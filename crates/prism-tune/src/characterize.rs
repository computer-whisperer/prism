//! `prism-tune characterize` — raw response-curve characterization.
//!
//! Where `calibrate` runs a closed-loop fit against a `gain × cmd^gamma`
//! model and writes per-channel correction back via IPC, `characterize`
//! makes no model assumptions: it sweeps one channel's commanded value
//! across a configurable range (more samples than calibrate's probe,
//! optionally log- or linear-spaced) and dumps the raw measurement as
//! a CSV for offline plotting / fitting.
//!
//! Use when you don't trust `calibrate`'s simple-power-law fit — e.g.
//! to confirm whether a panel's commanded-to-emitted curve really is
//! a single power law in the linear region, or to find where the
//! soft-knee starts before the saturation cliff.
//!
//! The tool calls `ResetColor` at start (unless `--keep-calibration`)
//! so the panel sees raw commanded values without any per-channel
//! correction or CTM in the way. In HDR mode it also lifts the IR
//! clamp to 10000 nits per channel so the buffer's nits aren't
//! pre-clipped before reaching the panel.

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use prism_ipc::OutputAction;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::thread;
use std::time::Duration;
use tristim_driver::{Colorimeter, measurement::raw_to_xyz};

use crate::common::{
    Channel, OutputBaseline, apply_border, apply_panel_peaks, open_patch_surface,
    query_output_baseline, send_action, set_channel_patch, set_patch_off, set_white_patch,
    show_alignment_patch,
};

#[derive(Args)]
pub struct CharacterizeArgs {
    #[command(subcommand)]
    pub mode: CharacterizeMode,
}

#[derive(Subcommand)]
pub enum CharacterizeMode {
    /// Sweep one channel's commanded value across a range, measuring
    /// the panel's response at each step. Produces a raw response
    /// curve for offline analysis — no fitting, no model assumptions.
    SingleChannel(SingleChannelArgs),
    /// Sweep BT.2020 D65 white at varying luminances, measuring the
    /// panel's actual chromaticity + Y at each step. Use to diagnose
    /// non-additive panel behaviour by comparing the measured response
    /// to the sum-of-per-channel-sweep prediction.
    WhiteSweep(WhiteSweepArgs),
}

#[derive(Args)]
pub struct SingleChannelArgs {
    /// Connector to characterize (e.g. `DisplayPort-4`, `HDMI-A-1`).
    /// Use the long form — recent prism builds match the connector-
    /// driver name verbatim, not the `DP-N` shorthand.
    #[arg(long)]
    pub output: String,
    /// Which channel(s) to sweep. Defaults to all three sequentially.
    #[arg(long, value_enum)]
    pub channel: Option<ChannelArg>,
    /// Number of samples per channel in the sweep.
    #[arg(long, default_value_t = 20)]
    pub steps: usize,
    /// Minimum commanded value (cd/m²). Below ~1 nit the colorimeter
    /// noise floor dominates; below ~5 some panels gate the backlight.
    #[arg(long, default_value_t = 5.0)]
    pub min_cmd: f64,
    /// Maximum commanded value (cd/m²). Default is generous so the
    /// sweep walks past the panel's saturation cliff — use a smaller
    /// value if you want to stay strictly in the linear region.
    #[arg(long, default_value_t = 1000.0)]
    pub max_cmd: f64,
    /// Use linear spacing between samples instead of log. Log spacing
    /// gives more density at the dim end where the soft-toe and
    /// noise floor live; linear gives even coverage of the full range.
    #[arg(long)]
    pub linear: bool,
    /// Skip the initial `ResetColor` — leave whatever calibration is
    /// already live (KDL config + runtime overrides) in place. Useful
    /// for measuring the panel as it appears with calibration applied,
    /// rather than the raw panel response.
    #[arg(long)]
    pub keep_calibration: bool,
    /// Colorimeter calibration index (0..=6). 0 = the "General" preset.
    #[arg(long, default_value_t = 0)]
    pub cal: u8,
    /// Centered bright-window fraction (0..=1). Use 0.04–0.10 on
    /// ABL-throttled OLEDs; 1.0 fills the screen.
    #[arg(long, default_value_t = 0.10)]
    pub window: f64,
    /// Seconds to wait for the puck before the first sample.
    #[arg(long, default_value_t = 5)]
    pub prep_secs: u64,
    /// Settle time after each color change before measuring (ms).
    #[arg(long, default_value_t = 32)]
    pub settle_ms: u64,
    /// Per-sample CSV log path. Defaults to
    /// `prism-tune-characterize-<output>.csv` in the current directory.
    #[arg(long)]
    pub log: Option<PathBuf>,
    /// Skip writing the CSV log entirely.
    #[arg(long)]
    pub no_log: bool,
    /// Border luminance (cd/m²) painted around the centered patch.
    /// Same role as `calibrate --border-nits` — keeps panels' CABL
    /// from gating off during low-intensity measurements.
    #[arg(long, default_value_t = 50.0)]
    pub border_nits: f64,
    /// Disable the border entirely (black surround).
    #[arg(long)]
    pub no_border: bool,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum ChannelArg {
    R,
    G,
    B,
}

impl ChannelArg {
    fn to_channel(self) -> Channel {
        match self {
            Self::R => Channel::R,
            Self::G => Channel::G,
            Self::B => Channel::B,
        }
    }
}

pub fn run(args: CharacterizeArgs) -> Result<()> {
    match args.mode {
        CharacterizeMode::SingleChannel(a) => run_single_channel(a),
        CharacterizeMode::WhiteSweep(a) => run_white_sweep(a),
    }
}

fn run_single_channel(args: SingleChannelArgs) -> Result<()> {
    let baseline = query_output_baseline(&args.output)
        .context("query baseline output state via prism IPC")?;
    eprintln!(
        "Baseline for {}: mode={}, panel_peak={:?}, sdr_ref={}, prior_curve={}",
        args.output,
        if baseline.hdr_active { "HDR" } else { "SDR" },
        baseline.initial_panel_peak_nits,
        baseline.sdr_reference_nits,
        if baseline.initial_response_curve.is_some() {
            "present"
        } else {
            "identity"
        }
    );

    if !args.keep_calibration {
        send_action(&args.output, OutputAction::ResetColor)
            .context("initial ResetColor")?;
        eprintln!("Cleared runtime color overrides (--keep-calibration not set).");
    } else {
        eprintln!("Keeping live calibration in place (--keep-calibration).");
    }

    // Lift the IR clamp during HDR characterization so the buffer's
    // nits aren't pre-clipped — we want to see the panel's raw
    // response across the sweep range, even past its emitted peak.
    if baseline.hdr_active && !args.keep_calibration {
        apply_panel_peaks(&args.output, [10_000.0, 10_000.0, 10_000.0])?;
    }

    // Hardware setup.
    let mut device = Colorimeter::open_any().context("open colorimeter")?;
    let info = device.get_info().context("read colorimeter info")?;
    eprintln!(
        "Colorimeter: Spyder SN {} HW {}.{:02}",
        info.serial, info.hw_version.0, info.hw_version.1
    );
    let cal = device.get_calibration(args.cal).context("download cal matrix")?;
    let setup = device.get_setup(&cal).context("download setup")?;

    let mut patch = open_patch_surface(&args.output, baseline.hdr_active)?;
    patch.set_window_fraction(args.window).context("set window fraction")?;

    if !args.no_border {
        apply_border(&mut patch, &baseline, args.border_nits)?;
        eprintln!(
            "Anti-CABL border set to {:.1} cd/m² (override with --border-nits or --no-border).",
            args.border_nits
        );
    } else {
        eprintln!("Border disabled (--no-border).");
    }

    let alignment_nits = (args.border_nits * 2.0).max(40.0);
    show_alignment_patch(&mut patch, &baseline, alignment_nits)?;

    let mut log = open_log(&args, &baseline)?;

    eprintln!(
        "Place the puck flat on the centred patch on {} now. Sweep starts in {}s.",
        args.output, args.prep_secs
    );
    for s in (1..=args.prep_secs).rev() {
        eprintln!("  starting in {s}s...");
        thread::sleep(Duration::from_secs(1));
    }

    set_patch_off(&mut patch, baseline.hdr_active)?;

    let channels: Vec<Channel> = match args.channel {
        Some(c) => vec![c.to_channel()],
        None => Channel::ALL.to_vec(),
    };

    let targets = generate_targets(args.min_cmd, args.max_cmd, args.steps, args.linear);
    let settle = Duration::from_millis(args.settle_ms);

    eprintln!(
        "\nSweep: {} samples per channel, {} spacing, range [{:.2}, {:.2}] cd/m²",
        targets.len(),
        if args.linear { "linear" } else { "log" },
        args.min_cmd,
        args.max_cmd,
    );

    for channel in channels {
        eprintln!("\n--- {} channel ---", channel.label());
        for (idx, &cmd) in targets.iter().enumerate() {
            set_channel_patch(&mut patch, &baseline, channel, cmd)?;
            thread::sleep(settle);
            let raw = device.measure_raw(&setup).context("measure")?;
            let xyz = raw_to_xyz(&raw, &setup, &cal);
            let (cx, cy) = xyz.chromaticity().unwrap_or((0.0, 0.0));
            eprintln!(
                "  {} cmd {:>8.2} cd/m² → X={:>8.3}  Y={:>8.3}  Z={:>8.3}  xy=({:.4}, {:.4})",
                channel.label(), cmd, xyz.x, xyz.y, xyz.z, cx, cy,
            );
            if let Some((_, w)) = log.as_mut() {
                writeln!(
                    w,
                    "{},{},{:.4},{:.4},{:.4},{:.4},{:.6},{:.6}",
                    channel.label(),
                    idx + 1,
                    cmd,
                    xyz.x,
                    xyz.y,
                    xyz.z,
                    cx,
                    cy,
                )?;
            }
        }
    }

    set_patch_off(&mut patch, baseline.hdr_active)?;

    if let Some((path, mut w)) = log {
        w.flush().ok();
        eprintln!("\nCSV log written to {}", path.display());
    }

    // Restore the bringup state — always, since characterize is purely
    // diagnostic and shouldn't leave the panel in a weird intermediate
    // state.
    send_action(&args.output, OutputAction::ResetColor)
        .context("final ResetColor")?;

    Ok(())
}

#[derive(Args)]
pub struct WhiteSweepArgs {
    /// Connector to characterize (e.g. `DisplayPort-4`, `HDMI-A-1`).
    /// Use the long form — recent prism builds match the connector-
    /// driver name verbatim, not the `DP-N` shorthand.
    #[arg(long)]
    pub output: String,
    /// Number of samples in the white sweep.
    #[arg(long, default_value_t = 20)]
    pub steps: usize,
    /// Minimum target luminance (cd/m²) for D65 white.
    #[arg(long, default_value_t = 5.0)]
    pub min_nits: f64,
    /// Maximum target luminance (cd/m²) for D65 white. Default is
    /// generous so the sweep walks past where the panel can physically
    /// emit D65 (which is capped by the weakest subpixel's peak ÷
    /// its D65 weight). For DP-4-class panels this lands near 100 nits;
    /// for OLEDs it can land well above 1000.
    #[arg(long, default_value_t = 1000.0)]
    pub max_nits: f64,
    /// Use linear spacing between samples instead of log.
    #[arg(long)]
    pub linear: bool,
    /// Skip the initial `ResetColor` — leave whatever calibration is
    /// already live in place.
    #[arg(long)]
    pub keep_calibration: bool,
    /// Colorimeter calibration index (0..=6).
    #[arg(long, default_value_t = 0)]
    pub cal: u8,
    /// Centered bright-window fraction (0..=1).
    #[arg(long, default_value_t = 0.10)]
    pub window: f64,
    /// Seconds to wait for the puck before the first sample.
    #[arg(long, default_value_t = 5)]
    pub prep_secs: u64,
    /// Settle time after each color change before measuring (ms).
    #[arg(long, default_value_t = 32)]
    pub settle_ms: u64,
    /// Per-sample CSV log path. Defaults to
    /// `prism-tune-characterize-white-<output>.csv`.
    #[arg(long)]
    pub log: Option<PathBuf>,
    /// Skip writing the CSV log entirely.
    #[arg(long)]
    pub no_log: bool,
    /// Border luminance (cd/m²) painted around the centered patch.
    #[arg(long, default_value_t = 50.0)]
    pub border_nits: f64,
    /// Disable the border entirely (black surround).
    #[arg(long)]
    pub no_border: bool,
}

fn run_white_sweep(args: WhiteSweepArgs) -> Result<()> {
    let baseline = query_output_baseline(&args.output)
        .context("query baseline output state via prism IPC")?;
    eprintln!(
        "Baseline for {}: mode={}, panel_peak={:?}, sdr_ref={}, prior_curve={}",
        args.output,
        if baseline.hdr_active { "HDR" } else { "SDR" },
        baseline.initial_panel_peak_nits,
        baseline.sdr_reference_nits,
        if baseline.initial_response_curve.is_some() {
            "present"
        } else {
            "identity"
        }
    );

    if !args.keep_calibration {
        send_action(&args.output, OutputAction::ResetColor)
            .context("initial ResetColor")?;
        eprintln!("Cleared runtime color overrides (--keep-calibration not set).");
    } else {
        eprintln!("Keeping live calibration in place (--keep-calibration).");
    }

    if baseline.hdr_active && !args.keep_calibration {
        apply_panel_peaks(&args.output, [10_000.0, 10_000.0, 10_000.0])?;
    }

    let mut device = Colorimeter::open_any().context("open colorimeter")?;
    let info = device.get_info().context("read colorimeter info")?;
    eprintln!(
        "Colorimeter: Spyder SN {} HW {}.{:02}",
        info.serial, info.hw_version.0, info.hw_version.1
    );
    let cal = device.get_calibration(args.cal).context("download cal matrix")?;
    let setup = device.get_setup(&cal).context("download setup")?;

    let mut patch = open_patch_surface(&args.output, baseline.hdr_active)?;
    patch.set_window_fraction(args.window).context("set window fraction")?;

    if !args.no_border {
        apply_border(&mut patch, &baseline, args.border_nits)?;
        eprintln!(
            "Anti-CABL border set to {:.1} cd/m² (override with --border-nits or --no-border).",
            args.border_nits
        );
    } else {
        eprintln!("Border disabled (--no-border).");
    }

    let alignment_nits = (args.border_nits * 2.0).max(40.0);
    show_alignment_patch(&mut patch, &baseline, alignment_nits)?;

    let mut log = open_white_log(&args, &baseline)?;

    eprintln!(
        "Place the puck flat on the centred patch on {} now. Sweep starts in {}s.",
        args.output, args.prep_secs
    );
    for s in (1..=args.prep_secs).rev() {
        eprintln!("  starting in {s}s...");
        thread::sleep(Duration::from_secs(1));
    }

    set_patch_off(&mut patch, baseline.hdr_active)?;

    let targets = generate_targets(args.min_nits, args.max_nits, args.steps, args.linear);
    let settle = Duration::from_millis(args.settle_ms);

    eprintln!(
        "\nWhite sweep: {} samples, {} spacing, BT.2020 D65 white from {:.2} → {:.2} cd/m²",
        targets.len(),
        if args.linear { "linear" } else { "log" },
        args.min_nits,
        args.max_nits,
    );
    eprintln!("    (each patch is set_nits([L, L, L]) — equal R/G/B in linear nits → D65 white in BT.2020)");

    // D65 reference for inline Δu'v' / Δxy reporting.
    let (d65_up, d65_vp) = xy_to_uv_prime((0.3127, 0.3290));

    for (idx, &target) in targets.iter().enumerate() {
        set_white_patch(&mut patch, &baseline, target)?;
        thread::sleep(settle);
        let raw = device.measure_raw(&setup).context("measure")?;
        let xyz = raw_to_xyz(&raw, &setup, &cal);
        let (cx, cy) = xyz.chromaticity().unwrap_or((0.0, 0.0));
        let (up, vp) = xy_to_uv_prime((cx, cy));
        let duv = ((up - d65_up).powi(2) + (vp - d65_vp).powi(2)).sqrt();
        let y_err_pct = (xyz.y - target) / target.max(0.01) * 100.0;
        eprintln!(
            "  W target {:>8.2} cd/m² → X={:>8.3}  Y={:>8.3}  Z={:>8.3}  xy=({:.4}, {:.4})  Δu'v'={:.4}  Y_err={:+.1}%",
            target, xyz.x, xyz.y, xyz.z, cx, cy, duv, y_err_pct,
        );
        if let Some((_, w)) = log.as_mut() {
            writeln!(
                w,
                "W,{},{:.4},{:.4},{:.4},{:.4},{:.6},{:.6},{:.5},{:+.3}",
                idx + 1,
                target,
                xyz.x,
                xyz.y,
                xyz.z,
                cx,
                cy,
                duv,
                y_err_pct,
            )?;
        }
    }

    set_patch_off(&mut patch, baseline.hdr_active)?;

    if let Some((path, mut w)) = log {
        w.flush().ok();
        eprintln!("\nCSV log written to {}", path.display());
    }

    send_action(&args.output, OutputAction::ResetColor)
        .context("final ResetColor")?;

    Ok(())
}

/// CIE 1976 uniform chromaticity (u', v') from xy. Duplicated here so
/// characterize doesn't depend on calibrate's private helpers.
fn xy_to_uv_prime(xy: (f64, f64)) -> (f64, f64) {
    let (x, y) = xy;
    let denom = -2.0 * x + 12.0 * y + 3.0;
    if denom.abs() < 1e-9 {
        return (0.0, 0.0);
    }
    (4.0 * x / denom, 9.0 * y / denom)
}

/// Generate `n` sample targets in `[lo, hi]`, either log- or linear-spaced.
/// Returns at least 2 samples even if `n < 2` was requested (degenerate
/// sweeps aren't useful and would crash on the first iteration).
fn generate_targets(lo: f64, hi: f64, n: usize, linear: bool) -> Vec<f64> {
    let n = n.max(2);
    let lo = lo.max(1e-3);
    let hi = hi.max(lo * 1.01);
    if linear {
        let step = (hi - lo) / (n - 1) as f64;
        (0..n).map(|i| lo + step * i as f64).collect()
    } else {
        let lo_ln = lo.ln();
        let hi_ln = hi.ln();
        let step = (hi_ln - lo_ln) / (n - 1) as f64;
        (0..n).map(|i| (lo_ln + step * i as f64).exp()).collect()
    }
}

fn open_log(
    args: &SingleChannelArgs,
    baseline: &OutputBaseline,
) -> Result<Option<(PathBuf, BufWriter<File>)>> {
    if args.no_log {
        return Ok(None);
    }
    let path = args.log.clone().unwrap_or_else(|| {
        let safe = args.output.replace('/', "_");
        PathBuf::from(format!("prism-tune-characterize-{safe}.csv"))
    });
    if path.as_os_str() == "/dev/null" {
        return Ok(None);
    }
    let file = File::create(&path)
        .with_context(|| format!("create log file {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "# prism-tune characterize single-channel — output={} mode={} steps={} settle_ms={} window={} spacing={} border_nits={} keep_calibration={}",
        args.output,
        if baseline.hdr_active { "HDR" } else { "SDR" },
        args.steps,
        args.settle_ms,
        args.window,
        if args.linear { "linear" } else { "log" },
        args.border_nits,
        args.keep_calibration,
    )?;
    writeln!(
        w,
        "# baseline: panel_peak_nits={:?} sdr_reference_nits={} prior_response_curve={:?}",
        baseline.initial_panel_peak_nits,
        baseline.sdr_reference_nits,
        baseline.initial_response_curve,
    )?;
    writeln!(
        w,
        "# range: [{:.4}, {:.4}] cd/m²",
        args.min_cmd, args.max_cmd
    )?;
    writeln!(w, "channel,sample_idx,commanded_nits,X,Y,Z,x,y")?;
    eprintln!("Logging per-sample CSV to {}", path.display());
    Ok(Some((path, w)))
}

fn open_white_log(
    args: &WhiteSweepArgs,
    baseline: &OutputBaseline,
) -> Result<Option<(PathBuf, BufWriter<File>)>> {
    if args.no_log {
        return Ok(None);
    }
    let path = args.log.clone().unwrap_or_else(|| {
        let safe = args.output.replace('/', "_");
        PathBuf::from(format!("prism-tune-characterize-white-{safe}.csv"))
    });
    if path.as_os_str() == "/dev/null" {
        return Ok(None);
    }
    let file = File::create(&path)
        .with_context(|| format!("create log file {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "# prism-tune characterize white-sweep — output={} mode={} steps={} settle_ms={} window={} spacing={} border_nits={} keep_calibration={}",
        args.output,
        if baseline.hdr_active { "HDR" } else { "SDR" },
        args.steps,
        args.settle_ms,
        args.window,
        if args.linear { "linear" } else { "log" },
        args.border_nits,
        args.keep_calibration,
    )?;
    writeln!(
        w,
        "# baseline: panel_peak_nits={:?} sdr_reference_nits={} prior_response_curve={:?}",
        baseline.initial_panel_peak_nits,
        baseline.sdr_reference_nits,
        baseline.initial_response_curve,
    )?;
    writeln!(
        w,
        "# range: [{:.4}, {:.4}] cd/m² (BT.2020 D65 reference white)",
        args.min_nits, args.max_nits
    )?;
    writeln!(
        w,
        "channel,sample_idx,target_nits,X,Y,Z,x,y,delta_uv_from_d65,y_err_pct"
    )?;
    eprintln!("Logging per-sample CSV to {}", path.display());
    Ok(Some((path, w)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_spacing_endpoints_and_count() {
        let targets = generate_targets(10.0, 1000.0, 5, false);
        assert_eq!(targets.len(), 5);
        assert!((targets[0] - 10.0).abs() < 1e-6);
        assert!((targets[4] - 1000.0).abs() < 1e-6);
        // Log spacing: each step is a constant ratio.
        let ratio = targets[1] / targets[0];
        for i in 1..targets.len() {
            let r = targets[i] / targets[i - 1];
            assert!((r - ratio).abs() < 1e-6, "non-uniform log spacing at {i}");
        }
    }

    #[test]
    fn linear_spacing_endpoints_and_count() {
        let targets = generate_targets(10.0, 1000.0, 5, true);
        assert_eq!(targets.len(), 5);
        assert!((targets[0] - 10.0).abs() < 1e-6);
        assert!((targets[4] - 1000.0).abs() < 1e-6);
        let step = targets[1] - targets[0];
        for i in 1..targets.len() {
            let s = targets[i] - targets[i - 1];
            assert!((s - step).abs() < 1e-6, "non-uniform linear spacing at {i}");
        }
    }

    #[test]
    fn degenerate_steps_clamped_to_minimum() {
        let targets = generate_targets(10.0, 1000.0, 1, false);
        assert!(targets.len() >= 2);
    }
}
