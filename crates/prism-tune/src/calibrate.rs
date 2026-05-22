//! `prism-tune calibrate` — closed-loop per-output color calibration.
//!
//! **Role:** make prism's f32 intermediate + capability metadata
//! match physical panel output for a given connector. Single tool,
//! single invocation per panel, three phases per run.
//!
//! **Phase 0 — query state via prism IPC.** Mode (`hdr_active`),
//! current `panel_peak_nits[3]`, current `sdr_reference_nits`,
//! current `response_curve`. Branch on mode.
//!
//! **Phase 1 — per-channel saturation discovery.** For each of R, G,
//! B independently: drive only that channel, walk an ascending
//! commanded-nits ramp until the measured Y plateaus. Last non-
//! saturated commanded value = that channel's measured peak.
//!
//! - HDR mode: lift the compositor clamp first (set `PanelPeakNits`
//!   to `[10_000; 3]`) so the buffer's nits aren't pre-clipped by
//!   prism's decode stage before reaching the panel. Apply
//!   discovered per-channel peaks via `OutputAction::PanelPeakNits`
//!   immediately — this also rebuilds the HDR_OUTPUT_METADATA
//!   infoframe so the sink tonemaps against measured reality.
//! - SDR mode: don't override the user's existing panel-peak policy
//!   (it represents a vibrancy preference, not a measurement).
//!   Sweep targets derived via `sRGB_OETF(target / sdr_ref)`. Log
//!   per-channel peaks; warn if any channel can't reach the
//!   configured `sdr_reference_nits`.
//!
//! **Phase 2 — per-channel response refinement.** Within each
//! channel's discovered cap: 5 targets, 5 measure → fit → apply
//! iterations. Each channel's `(gain, gamma)` fit independently from
//! its pure-color sweep. Final per-channel curve applied via
//! `OutputAction::ResponseCurve`.
//!
//! **CSV log:** one row per measurement; `phase` + `channel` columns
//! identify context. Chromaticity x/y per row captures per-primary
//! positions for future CTM-from-measured-primaries work without
//! extra hold time.

use anyhow::{Context, Result};
use clap::Args;
use prism_ipc::socket::Socket;
use prism_ipc::{ColorState, OutputAction, Request, Response};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::thread;
use std::time::Duration;
use tristim_display::{PatchSurface, PqDescriptionParams};
use tristim_driver::{Calibration, Colorimeter, Setup, measurement::raw_to_xyz};

#[derive(Args)]
pub struct CalibrateArgs {
    /// Connector to calibrate (e.g. `DP-4`).
    #[arg(long)]
    pub output: String,
    /// Colorimeter calibration index (0..=6). 0 = the "General" preset.
    #[arg(long, default_value_t = 0)]
    pub cal: u8,
    /// Centered bright-window fraction (0..=1). Use 0.04–0.10 on
    /// ABL-throttled OLEDs to measure peak response; 1.0 fills the
    /// screen (right for LCDs without ABL).
    #[arg(long, default_value_t = 0.10)]
    pub window: f64,
    /// Refinement iterations per channel.
    #[arg(long, default_value_t = 5)]
    pub iterations: u32,
    /// Seconds to wait for the puck before the first sweep.
    #[arg(long, default_value_t = 5)]
    pub prep_secs: u64,
    /// Settle time after each color change before measuring (ms).
    #[arg(long, default_value_t = 32)]
    pub settle_ms: u64,
    /// Leave the tuned curve + per-channel peak active on exit.
    /// Default: send ResetColor so the persisted KDL config wins again
    /// and re-runs start from a clean baseline.
    #[arg(long)]
    pub keep: bool,
    /// Per-measurement CSV log path. Defaults to
    /// `prism-tune-calibrate-<output>.csv` in the current directory.
    #[arg(long)]
    pub log: Option<PathBuf>,
    /// Skip writing the CSV log entirely.
    #[arg(long)]
    pub no_log: bool,
}

/// Channel identifier — used to pick a patch driver and to label CSV
/// rows. Order matches array indices everywhere: R=0, G=1, B=2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Channel {
    R,
    G,
    B,
}

impl Channel {
    fn idx(self) -> usize {
        match self {
            Self::R => 0,
            Self::G => 1,
            Self::B => 2,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::R => "R",
            Self::G => "G",
            Self::B => "B",
        }
    }

    const ALL: [Self; 3] = [Self::R, Self::G, Self::B];
}

/// What the calibrator is doing at the moment a row is logged. Used
/// to keep the CSV log self-describing — `probe` rows are the
/// saturation discovery phase, `refine` rows are the iterative gain/
/// gamma fit.
#[derive(Clone, Copy, Debug)]
enum Phase {
    Probe,
    Refine,
}

impl Phase {
    fn label(self) -> &'static str {
        match self {
            Self::Probe => "probe",
            Self::Refine => "refine",
        }
    }
}

/// What the compositor reports about the output. The full per-output
/// `ColorState` from IPC, plus a one-shot snapshot taken at the start
/// of the run so phase transitions can compare against the baseline.
struct OutputBaseline {
    hdr_active: bool,
    sdr_reference_nits: f64,
    /// Per-channel panel peak as the compositor sees it at run start.
    /// We don't read this back later — Phase 1 replaces it.
    initial_panel_peak_nits: [f64; 3],
    /// Pre-existing response curve, if any. Reported for context;
    /// not consumed (the run resets to identity at start).
    initial_response_curve: Option<([f64; 3], [f64; 3])>,
}

/// Current state of the calibrator's running curve estimate. The
/// refinement loop reads/writes this; the IPC application is what
/// makes the compositor see it.
#[derive(Clone, Copy, Debug)]
struct PerChannelCurve {
    gain: [f64; 3],
    gamma: [f64; 3],
}

impl PerChannelCurve {
    const IDENTITY: Self = Self {
        gain: [1.0, 1.0, 1.0],
        gamma: [1.0, 1.0, 1.0],
    };
}

// ─── Top-level entrypoint ──────────────────────────────────────────────────

pub fn run(args: CalibrateArgs) -> Result<()> {
    // Phase 0 — query baseline state via IPC. Failing here means the
    // compositor isn't running or PRISM_SOCKET isn't set; bail before
    // touching the colorimeter or display surface.
    let baseline = query_output_baseline(&args.output)
        .context("query baseline output state via prism IPC")?;
    eprintln!(
        "Baseline for {}: mode={}, panel_peak={:?}, sdr_ref={}, prior_curve={}",
        args.output,
        if baseline.hdr_active { "HDR" } else { "SDR" },
        baseline.initial_panel_peak_nits,
        baseline.sdr_reference_nits,
        if baseline.initial_response_curve.is_some() {
            "present (will reset for clean baseline)"
        } else {
            "identity"
        }
    );

    // Reset any prior runtime overrides so we always start from KDL
    // defaults (modulo what's actually compiled in). The fit math
    // assumes the curve is identity before the first iteration.
    send_action(&args.output, OutputAction::ResetColor)
        .context("initial ResetColor")?;

    // Open hardware. Probe colorimeter first — if the puck isn't
    // plugged in / udev rule missing we want to fail before the user
    // has held it for 30 seconds.
    let mut device = Colorimeter::open_any().context("open colorimeter")?;
    let info = device.get_info().context("read colorimeter info")?;
    eprintln!(
        "Colorimeter: Spyder SN {} HW {}.{:02}",
        info.serial, info.hw_version.0, info.hw_version.1
    );
    let cal = device.get_calibration(args.cal).context("download cal matrix")?;
    let setup = device.get_setup(&cal).context("download setup")?;

    // Patch surface. Mode picks the constructor.
    let mut patch = open_patch_surface(&args, &baseline)?;
    patch.set_window_fraction(args.window).context("set window fraction")?;
    set_patch_off(&mut patch, baseline.hdr_active)?;

    // CSV log — open up front so a permission/path error fails before
    // the user has held the puck.
    let mut log = open_log(&args, &baseline)?;

    eprintln!(
        "Place the puck flat on {} now. Calibration starts in {}s.",
        args.output, args.prep_secs
    );
    for s in (1..=args.prep_secs).rev() {
        eprintln!("  starting in {s}s...");
        thread::sleep(Duration::from_secs(1));
    }

    // ─── Phase 1: per-channel saturation discovery ────────────────────
    let (discovered_peaks, probe_peak_y) = discover_per_channel_peaks(
        &args,
        &baseline,
        &mut device,
        &mut patch,
        &setup,
        &cal,
        log.as_mut(),
    )?;
    eprintln!(
        "\nDiscovered per-channel commanded peaks (cd/m²): R={:.1}  G={:.1}  B={:.1}",
        discovered_peaks[0], discovered_peaks[1], discovered_peaks[2]
    );
    eprintln!(
        "  saturation-Y for filtering refinement samples: R={:.1}  G={:.1}  B={:.1}",
        probe_peak_y[0], probe_peak_y[1], probe_peak_y[2]
    );

    // HDR mode: apply the measured peaks via IPC so subsequent
    // refinement runs with the compositor clamp at calibrated reality
    // (and the HDR_OUTPUT_METADATA reflects the new ceiling). SDR
    // mode: log only — sdr_reference_nits is policy, not measurement,
    // and overriding it would conflict with the user's vibrancy
    // preference. Warn if a channel can't reach sdr_reference_nits.
    if baseline.hdr_active {
        apply_panel_peaks(&args.output, discovered_peaks)?;
    } else {
        for c in Channel::ALL {
            if discovered_peaks[c.idx()] < baseline.sdr_reference_nits * 0.9 {
                eprintln!(
                    "  WARN: {} channel peaks at {:.1} cd/m² but sdr-reference-nits is {:.1} — \
                     this channel cannot reach SDR white. Lower sdr-reference-nits or accept \
                     clipping on pure {}.",
                    c.label(),
                    discovered_peaks[c.idx()],
                    baseline.sdr_reference_nits,
                    c.label(),
                );
            }
        }
    }
    if let Some((_, w)) = log.as_mut() {
        writeln!(
            w,
            "# phase 1 complete: discovered peaks R={:.3} G={:.3} B={:.3}",
            discovered_peaks[0], discovered_peaks[1], discovered_peaks[2]
        )?;
    }

    // ─── Phase 2: per-channel response refinement ─────────────────────
    let curve = refine_per_channel_curve(
        &args,
        &baseline,
        &discovered_peaks,
        &probe_peak_y,
        &mut device,
        &mut patch,
        &setup,
        &cal,
        log.as_mut(),
    )?;

    // Black before the user lifts the puck — polite + ABL-friendly.
    set_patch_off(&mut patch, baseline.hdr_active)?;

    if let Some((path, mut w)) = log {
        writeln!(
            w,
            "# final: panel_peak_nits=[{:.3},{:.3},{:.3}] response_curve gain=[{:.4},{:.4},{:.4}] gamma=[{:.4},{:.4},{:.4}]",
            discovered_peaks[0], discovered_peaks[1], discovered_peaks[2],
            curve.gain[0], curve.gain[1], curve.gain[2],
            curve.gamma[0], curve.gamma[1], curve.gamma[2],
        )?;
        w.flush().ok();
        eprintln!("CSV log written to {}", path.display());
    }

    // ─── Print KDL block for paste-in ─────────────────────────────────
    print_kdl_block(&args.output, baseline.hdr_active, discovered_peaks, curve);

    if args.keep {
        eprintln!(
            "\n--keep: discovered panel peak + tuned curve remain active until prism restart."
        );
    } else {
        eprintln!(
            "\nRestoring KDL config defaults (use --keep to leave the tuned values active)."
        );
        send_action(&args.output, OutputAction::ResetColor)
            .context("final ResetColor")?;
    }

    Ok(())
}

// ─── Phase 0 helpers ───────────────────────────────────────────────────────

fn query_output_baseline(name: &str) -> Result<OutputBaseline> {
    let mut socket = Socket::connect().context("connect to PRISM_SOCKET")?;
    let reply = socket
        .send(Request::Outputs)
        .context("Request::Outputs")?;
    let outputs: HashMap<String, prism_ipc::Output> = match reply {
        Ok(Response::Outputs(map)) => map,
        Ok(other) => anyhow::bail!("unexpected reply to Outputs: {other:?}"),
        Err(e) => anyhow::bail!("prism returned error: {e}"),
    };
    let output = outputs
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("no output named {name:?} (connected: {:?})", outputs.keys().collect::<Vec<_>>()))?;
    let ColorState {
        hdr_active,
        panel_peak_nits,
        sdr_reference_nits,
        response_curve,
    } = output.color;
    let initial_response_curve = response_curve.map(|c| (c.gain, c.gamma));
    Ok(OutputBaseline {
        hdr_active,
        sdr_reference_nits,
        initial_panel_peak_nits: panel_peak_nits,
        initial_response_curve,
    })
}

fn open_patch_surface(args: &CalibrateArgs, baseline: &OutputBaseline) -> Result<PatchSurface> {
    if baseline.hdr_active {
        // Mastering peak high enough that the patch's declared envelope
        // doesn't pre-clip the probe ramp. We rebuild HDR_OUTPUT_METADATA
        // on the compositor side via PanelPeakNits after discovery, so
        // the sink ends up tonemapping against measured reality.
        let probe_peak = 10_000;
        let params = PqDescriptionParams {
            mastering_min_lum_ticks: 5,
            mastering_max_lum: probe_peak,
            max_cll: probe_peak,
            max_fall: probe_peak / 2,
        };
        PatchSurface::open_hdr(&args.output, params)
            .with_context(|| format!("open HDR patch on {}", args.output))
    } else {
        PatchSurface::open(&args.output)
            .with_context(|| format!("open SDR patch on {}", args.output))
    }
}

/// Drive the patch to black using the right setter for the current mode.
/// Used at the start and end of the run so the panel isn't left glaring.
fn set_patch_off(patch: &mut PatchSurface, hdr_active: bool) -> Result<()> {
    if hdr_active {
        patch.set_nits([0.0, 0.0, 0.0]).context("set black (HDR)")
    } else {
        patch.set_color([0.0, 0.0, 0.0]).context("set black (SDR)")
    }
}

// ─── Phase 1 helpers ───────────────────────────────────────────────────────

/// Walk each channel ascending; find where measured Y plateaus.
///
/// Returns `(peaks, probe_peak_y)`:
/// - `peaks[c]` = highest commanded value that produced a non-saturated
///   measurement (used as the panel-peak ceiling for clamping the
///   intermediate buffer).
/// - `probe_peak_y[c]` = highest measured Y observed for that channel
///   during probe (used by refinement to filter out saturation-
///   polluted samples — if `Y >= 0.95 × probe_peak_y[c]` the panel was
///   clipping and the data point doesn't reflect linear response).
fn discover_per_channel_peaks(
    args: &CalibrateArgs,
    baseline: &OutputBaseline,
    device: &mut Colorimeter,
    patch: &mut PatchSurface,
    setup: &Setup,
    cal: &Calibration,
    mut log: Option<&mut (PathBuf, BufWriter<File>)>,
) -> Result<([f64; 3], [f64; 3])> {
    // HDR mode: lift the compositor clamp temporarily so the buffer's
    // nits aren't pre-clipped before reaching the panel. SDR mode:
    // leave the user's policy alone — sdr_reference_nits + panel peak
    // are theirs to choose, our job is to measure under those.
    if baseline.hdr_active {
        apply_panel_peaks(&args.output, [10_000.0, 10_000.0, 10_000.0])?;
    }

    // Probe ramp. HDR: ramp in absolute nits up to a generous ceiling
    // (no panel exceeds 4000 in practice; 6000 here gives at least one
    // clipped sample so saturation detection has something to bite on).
    // SDR: ramp in equivalent nits derived from RGB via sRGB OETF.
    let probe_targets: Vec<f64> = if baseline.hdr_active {
        vec![25.0, 100.0, 400.0, 1500.0, 6000.0]
    } else {
        // SDR equivalents — values in nits, converted to RGB at patch time.
        // Cap at sdr_reference_nits since RGB=1.0 maps to sdr_ref exactly.
        let r = baseline.sdr_reference_nits;
        vec![
            r * 0.05,
            r * 0.15,
            r * 0.35,
            r * 0.65,
            r,
        ]
    };

    let settle = Duration::from_millis(args.settle_ms);
    let mut peaks = [0.0f64; 3];
    let mut probe_peak_y = [0.0f64; 3];

    for c in Channel::ALL {
        eprintln!("\n--- phase 1 probe: {} channel ---", c.label());
        // measure_raw() resets per-call — Argyll's auto-zero behaviour.
        // Skipping it (the earlier --no-reset optimisation) made dim
        // single-channel reads unreliable on real LCDs; honest dark-
        // current refresh per measurement is the only safe default.
        let mut measurements: Vec<(f64, f64)> = Vec::with_capacity(probe_targets.len());
        for (patch_idx, &target_nits) in probe_targets.iter().enumerate() {
            set_channel_patch(patch, baseline, c, target_nits)?;
            thread::sleep(settle);
            let raw = device.measure_raw(setup).context("measure")?;
            let xyz = raw_to_xyz(&raw, setup, cal);
            let (cx, cy) = xyz.chromaticity().unwrap_or((0.0, 0.0));
            eprintln!(
                "  {} target {:>7.1} cd/m²  →  X={:>7.2}  Y={:>7.2}  Z={:>7.2}",
                c.label(), target_nits, xyz.x, xyz.y, xyz.z,
            );
            measurements.push((target_nits, xyz.y));
            probe_peak_y[c.idx()] = probe_peak_y[c.idx()].max(xyz.y);
            if let Some((_, w)) = log.as_mut() {
                writeln!(
                    w,
                    "{},{},1,{},{:.4},1.0,1.0,1.0,1.0,1.0,1.0,{:.4},{:.4},{:.4},{:.6},{:.6}",
                    Phase::Probe.label(),
                    c.label(),
                    patch_idx + 1,
                    target_nits,
                    xyz.x, xyz.y, xyz.z, cx, cy,
                )?;
            }
        }
        // Saturation detection. Walk ascending; first index where the
        // Y-ratio falls well below the commanded ratio means we're past
        // the panel's ceiling for this channel. Threshold 1.2 is loose
        // — real panels deviate from clean power-law near the top, and
        // a tighter threshold would prematurely flag the soft knee.
        let mut saturated_at: Option<usize> = None;
        for i in 1..measurements.len() {
            let (prev_t, prev_y) = measurements[i - 1];
            let (this_t, this_y) = measurements[i];
            let t_ratio = this_t / prev_t.max(0.01);
            let y_ratio = this_y / prev_y.max(0.01);
            if t_ratio >= 2.0 && y_ratio < 1.2 {
                saturated_at = Some(i);
                eprintln!(
                    "  {} saturation detected at target {:.1} (Y ratio {:.2} vs target ratio {:.2})",
                    c.label(), this_t, y_ratio, t_ratio,
                );
                break;
            }
        }

        // Bisection sub-probe: when the initial coarse ramp brackets
        // the cliff in a single decade (e.g. (400, 1500) for a panel
        // that actually caps at ~700), bisect 3 times in log space to
        // tighten the discovered peak. Without this, the refinement
        // phase uses 0.85 × coarse_last_good — which could be either
        // too conservative (leave headroom unused) or too aggressive
        // (sit on the soft knee). Three bisections narrows the cliff
        // to ~12% relative width, giving safe_peak room to land in.
        let (left_t, left_y) = match saturated_at {
            Some(i) => {
                // measurements[i-1] is last non-sat, [i] is first sat.
                let right_t = measurements[i].0;
                let lt = measurements[i - 1].0;
                let ly = measurements[i - 1].1;
                eprintln!(
                    "  {} bisecting cliff in ({:.1}, {:.1}) cd/m² (3 steps)…",
                    c.label(), lt, right_t,
                );
                let mut right_t = right_t;
                let mut left_t = lt;
                let mut left_y = ly;
                let mut patch_idx = measurements.len();
                for _step in 0..3 {
                    patch_idx += 1;
                    // Geometric midpoint — log-space bisection matches
                    // how the coarse ramp was spaced and the panel's
                    // power-law response.
                    let mid_t = (left_t * right_t).sqrt();
                    set_channel_patch(patch, baseline, c, mid_t)?;
                    thread::sleep(settle);
                    let raw = device.measure_raw(setup).context("measure (bisect)")?;
                    let xyz = raw_to_xyz(&raw, setup, cal);
                    let (cx, cy) = xyz.chromaticity().unwrap_or((0.0, 0.0));
                    let mid_y = xyz.y;
                    // Saturated when mid_y is barely above left_y
                    // (Y_ratio < 1.2). Sub-saturation means the panel
                    // is still ramping at mid_t.
                    let saturated = mid_y / left_y.max(0.01) < 1.2;
                    eprintln!(
                        "    bisect target {:>7.1} cd/m² → Y={:>7.2} ({})",
                        mid_t,
                        mid_y,
                        if saturated { "saturated" } else { "ramping" },
                    );
                    if let Some((_, w)) = log.as_mut() {
                        writeln!(
                            w,
                            "{},{},1,{},{:.4},1.0,1.0,1.0,1.0,1.0,1.0,{:.4},{:.4},{:.4},{:.6},{:.6}",
                            Phase::Probe.label(),
                            c.label(),
                            patch_idx,
                            mid_t,
                            xyz.x, xyz.y, xyz.z, cx, cy,
                        )?;
                    }
                    probe_peak_y[c.idx()] = probe_peak_y[c.idx()].max(mid_y);
                    if saturated {
                        right_t = mid_t;
                    } else {
                        left_t = mid_t;
                        left_y = mid_y;
                    }
                }
                (left_t, left_y)
            }
            None => {
                // No saturation in the coarse probe — the top sample
                // is the best estimate we have. (Unlikely in practice;
                // even 6000 cd/m² targets on real panels hit clipping
                // somewhere in our buffer/sink chain.)
                let last = measurements.len() - 1;
                (measurements[last].0, measurements[last].1)
            }
        };
        let _ = left_y; // silence unused; semantic value is the target

        peaks[c.idx()] = left_t;

        if let Some((_, w)) = log.as_mut() {
            writeln!(
                w,
                "# probe {} measured peak commanded = {:.3} cd/m² (after bisection), peak_y = {:.3}",
                c.label(),
                left_t,
                probe_peak_y[c.idx()],
            )?;
        }
    }

    Ok((peaks, probe_peak_y))
}

/// Drive a single-channel patch using the right setter for the mode.
/// SDR uses sRGB OETF to convert target nits → RGB 0..=1.
fn set_channel_patch(
    patch: &mut PatchSurface,
    baseline: &OutputBaseline,
    channel: Channel,
    target_nits: f64,
) -> Result<()> {
    if baseline.hdr_active {
        let mut rgb = [0.0_f64; 3];
        rgb[channel.idx()] = target_nits;
        patch.set_nits(rgb).with_context(|| {
            format!("set HDR nits for {} = {:.2}", channel.label(), target_nits)
        })
    } else {
        // SDR: convert target nits → linear → sRGB-encoded RGB.
        let linear = (target_nits / baseline.sdr_reference_nits).clamp(0.0, 1.0);
        let encoded = srgb_oetf(linear);
        let mut rgb = [0.0_f64; 3];
        rgb[channel.idx()] = encoded;
        patch.set_color(rgb).with_context(|| {
            format!("set SDR RGB for {} = {:.4} (target {:.2} cd/m²)", channel.label(), encoded, target_nits)
        })
    }
}

/// sRGB OETF (linear → encoded). Inverse of the EOTF in the decode shader.
fn srgb_oetf(linear: f64) -> f64 {
    if linear <= 0.0031308 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

// ─── Phase 2 helpers ───────────────────────────────────────────────────────

/// Per-channel iterative refinement of (gain, gamma). Each channel
/// fits independently from its own pure-color sweep; the resulting
/// curves are applied together at the end of each iteration via a
/// single ResponseCurve IPC.
///
/// Key design: **targets are adaptive per iteration**. We pick five
/// log-spaced *commanded* values in `[COMMANDED_LO, 0.85 × discovered_peak]`
/// and map them to intermediate-space targets `T = gain_k × commanded^gamma_k`
/// using the *current* curve V_k. That way the commanded values
/// arriving at the panel always land in its linear working range —
/// no matter how the curve has drifted — so fit data is always
/// inside the panel's well-behaved region. Without this, fixed
/// targets + iterative curve updates push commanded values into the
/// panel's saturation cliff after one or two iterations, poisoning
/// every subsequent fit (see the divergent DP-4 run in the prior
/// CSV log).
///
/// Filters before fit:
///   - **Noise floor:** drop samples where `Y < 1 cd/m²` (Spyder's
///     practical reliability threshold; on panels with per-channel
///     dead zones at low commanded values, this drops samples that
///     read effectively black).
///   - **Saturation polluted:** drop samples where `Y ≥ 0.95 × probe_peak_y[c]`
///     (the panel was clipping; the reading doesn't describe linear
///     response and would pull the fit toward gamma ≈ 0).
///
/// Plausibility check: reject fits where `panel_gamma` is outside
/// `[0.5, 2.5]` or where predicted top-Y deviates from measured by
/// more than 2× — both signs the data is too corrupted to trust.
///
/// Apply fit *directly* (no damping). With clean data the first
/// iteration converges very close to the panel's true response;
/// damping moved us *away* from the answer in the prior version.
/// Subsequent iterations re-validate and refine.
///
/// Early-exit on convergence: if all three channels' (gain, gamma)
/// move by < 5% from the previous iteration, we're done. Avoids
/// wasted iterations on cleanly-behaved panels.
fn refine_per_channel_curve(
    args: &CalibrateArgs,
    baseline: &OutputBaseline,
    discovered_peaks: &[f64; 3],
    probe_peak_y: &[f64; 3],
    device: &mut Colorimeter,
    patch: &mut PatchSurface,
    setup: &Setup,
    cal: &Calibration,
    mut log: Option<&mut (PathBuf, BufWriter<File>)>,
) -> Result<PerChannelCurve> {
    /// Lowest commanded value to sample. 50 cd/m² is well above the
    /// Spyder's Y reliability floor for all three primaries (B's peak
    /// Y on a typical panel is ~25, so commanded 50 → Y ~3 even
    /// near-identity; well above 1). Below this we hit dead zones on
    /// some panels (the Samsung LU28R55 in HDR mode doesn't emit red
    /// below ~95 commanded), and the noise-floor filter drops those
    /// samples anyway.
    const COMMANDED_LO: f64 = 50.0;
    /// Top end of the commanded sweep — leaves 15% margin below the
    /// discovered saturation cliff so the soft-knee doesn't pollute
    /// the top samples.
    const COMMANDED_HI_FRAC: f64 = 0.85;
    /// Saturation filter threshold — measurements above this fraction
    /// of the channel's probed peak Y are treated as clipping events
    /// rather than valid response data.
    const SATURATION_FILTER_FRAC: f64 = 0.95;
    /// Noise floor — measurements below this Y are below the
    /// Spyder's reliable range (dark-current drift + ambient pickup
    /// dominate).
    const NOISE_FLOOR_Y: f64 = 1.0;
    /// Plausibility window for the fitted panel gamma. Real panels
    /// in their working range fall well inside this — outside means
    /// the data is corrupted (saturation, dead zone, or measurement
    /// glitch).
    const GAMMA_PLAUSIBLE: std::ops::RangeInclusive<f64> = 0.5..=2.5;
    /// Convergence threshold for early-exit. When the max per-channel
    /// relative change in either gain or gamma drops below this, we
    /// stop iterating.
    const CONVERGED_REL_CHANGE: f64 = 0.05;

    let mut curve = PerChannelCurve::IDENTITY;
    let mut last_curve = curve;
    let settle = Duration::from_millis(args.settle_ms);

    for iter in 0..args.iterations {
        eprintln!(
            "\n--- phase 2 refine iter {} / {} (curve gain={:?}, gamma={:?}) ---",
            iter + 1,
            args.iterations,
            curve.gain,
            curve.gamma,
        );

        for c in Channel::ALL {
            // Adaptive targets: pick commanded values in the panel's
            // working range, then back-solve targets via the current
            // curve so commanded actually lands where we want it.
            let gain_k = curve.gain[c.idx()];
            let gamma_k = curve.gamma[c.idx()];
            let cmd_hi = (discovered_peaks[c.idx()] * COMMANDED_HI_FRAC).max(COMMANDED_LO + 10.0);
            let cmd_log_lo = COMMANDED_LO.ln();
            let cmd_log_hi = cmd_hi.ln();
            let targets: Vec<f64> = (0..5)
                .map(|i| {
                    let f = i as f64 / 4.0;
                    let cmd = (cmd_log_lo + f * (cmd_log_hi - cmd_log_lo)).exp();
                    gain_k * cmd.powf(gamma_k)
                })
                .collect();

            let mut samples: Vec<(f64, f64)> = Vec::with_capacity(targets.len());
            for (patch_idx, &t) in targets.iter().enumerate() {
                set_channel_patch(patch, baseline, c, t)?;
                thread::sleep(settle);
                let raw = device.measure_raw(setup).context("measure")?;
                let xyz = raw_to_xyz(&raw, setup, cal);
                let (cx, cy) = xyz.chromaticity().unwrap_or((0.0, 0.0));
                eprintln!(
                    "  {} target {:>7.2} cd/m² → X={:>7.2} Y={:>7.2} Z={:>7.2}",
                    c.label(), t, xyz.x, xyz.y, xyz.z,
                );
                samples.push((t, xyz.y));
                if let Some((_, w)) = log.as_mut() {
                    writeln!(
                        w,
                        "{},{},{},{},{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4},{:.4},{:.4},{:.6},{:.6}",
                        Phase::Refine.label(),
                        c.label(),
                        iter + 1,
                        patch_idx + 1,
                        t,
                        curve.gain[0], curve.gain[1], curve.gain[2],
                        curve.gamma[0], curve.gamma[1], curve.gamma[2],
                        xyz.x, xyz.y, xyz.z, cx, cy,
                    )?;
                }
            }

            // Filter both ends: noise-floor + saturation-polluted.
            let y_sat = probe_peak_y[c.idx()] * SATURATION_FILTER_FRAC;
            let useful: Vec<(f64, f64)> = samples
                .iter()
                .copied()
                .filter(|(_, y)| *y >= NOISE_FLOOR_Y && *y <= y_sat)
                .collect();
            if useful.len() < 2 {
                eprintln!(
                    "  {} only {} useful sample(s) after Y in [{}, {:.1}] filter; skipping fit",
                    c.label(),
                    useful.len(),
                    NOISE_FLOOR_Y,
                    y_sat,
                );
                if let Some((_, w)) = log.as_mut() {
                    writeln!(
                        w,
                        "# refine {} iter {} fit: SKIPPED (insufficient samples in [{}, {:.3}])",
                        c.label(),
                        iter + 1,
                        NOISE_FLOOR_Y,
                        y_sat,
                    )?;
                }
                continue;
            }

            // Log-linear fit. With current curve V_k = (gain_k, gamma_k)
            // already applied by the compositor:
            //   measured_Y = panel_gain * commanded^panel_gamma  where
            //   commanded = (T / gain_k)^(1/gamma_k)
            // So log(Y) = log(panel_gain) + (panel_gamma/gamma_k) * (log T - log gain_k)
            // Slope b, intercept a → panel_gamma = b * gamma_k,
            //                        panel_gain = exp(a) * gain_k^b.
            let (slope, intercept) = log_linear_fit(&useful);
            let fit_gamma = slope * gamma_k;
            let fit_gain = intercept.exp() * gain_k.powf(slope);

            // Plausibility: panel gamma must be in a reasonable range.
            if !GAMMA_PLAUSIBLE.contains(&fit_gamma) {
                eprintln!(
                    "  {} fit rejected: panel_gamma={:.3} outside {:?}",
                    c.label(), fit_gamma, GAMMA_PLAUSIBLE,
                );
                if let Some((_, w)) = log.as_mut() {
                    writeln!(
                        w,
                        "# refine {} iter {} fit: REJECTED (gamma={:.4} out of plausible range)",
                        c.label(), iter + 1, fit_gamma,
                    )?;
                }
                continue;
            }

            // Plausibility: predicted top-Y vs measured top-Y.
            let (top_target, top_measured_y) = *useful.last().unwrap();
            let predicted_top_y = fit_gain * top_target.powf(fit_gamma);
            let ratio = predicted_top_y / top_measured_y.max(0.01);
            if !(0.5..=2.0).contains(&ratio) {
                eprintln!(
                    "  {} fit rejected: predicted_top_Y={:.2} vs measured_top_Y={:.2} (ratio {:.2})",
                    c.label(), predicted_top_y, top_measured_y, ratio,
                );
                if let Some((_, w)) = log.as_mut() {
                    writeln!(
                        w,
                        "# refine {} iter {} fit: REJECTED (predicted_y={:.4} measured_y={:.4} ratio={:.4})",
                        c.label(), iter + 1, predicted_top_y, top_measured_y, ratio,
                    )?;
                }
                continue;
            }

            // Apply directly — no damping. Clean data + adaptive
            // targets means iter 1 is already very close to the true
            // panel response; damping just slows the obvious answer.
            curve.gain[c.idx()] = fit_gain;
            curve.gamma[c.idx()] = fit_gamma;
            eprintln!(
                "  {} fit applied: gain={:.4}, gamma={:.4}",
                c.label(), fit_gain, fit_gamma,
            );
            if let Some((_, w)) = log.as_mut() {
                writeln!(
                    w,
                    "# refine {} iter {} fit: applied gain={:.6} gamma={:.6}",
                    c.label(), iter + 1, fit_gain, fit_gamma,
                )?;
            }
        }

        // Push the full per-channel triple as a single IPC. All three
        // channels update together so the next iteration's
        // measurements are consistent with the curve we've committed.
        send_action(
            &args.output,
            OutputAction::ResponseCurve {
                gain_r: curve.gain[0],
                gain_g: curve.gain[1],
                gain_b: curve.gain[2],
                gamma_r: curve.gamma[0],
                gamma_g: curve.gamma[1],
                gamma_b: curve.gamma[2],
            },
        )
        .context("apply ResponseCurve")?;

        // Convergence check: skip wasted iterations once the curve
        // has stabilised. Identity → real curve in iter 1 is a huge
        // delta; subsequent moves should be < 5% on cleanly-behaved
        // panels. Don't bother on iter 1 (always huge delta from
        // identity).
        if iter > 0 {
            let max_rel = (0..3)
                .map(|i| {
                    let gain_d = (curve.gain[i] - last_curve.gain[i]).abs()
                        / last_curve.gain[i].abs().max(1e-3);
                    let gamma_d = (curve.gamma[i] - last_curve.gamma[i]).abs()
                        / last_curve.gamma[i].abs().max(1e-3);
                    gain_d.max(gamma_d)
                })
                .fold(0.0_f64, f64::max);
            eprintln!(
                "  iter {} convergence: max per-channel relative change = {:.2}%",
                iter + 1,
                max_rel * 100.0,
            );
            if let Some((_, w)) = log.as_mut() {
                writeln!(
                    w,
                    "# iter {} convergence: max_rel_change={:.4}",
                    iter + 1, max_rel,
                )?;
            }
            if max_rel < CONVERGED_REL_CHANGE {
                eprintln!(
                    "\nConverged after iter {} ({:.2}% < {:.2}% threshold).",
                    iter + 1, max_rel * 100.0, CONVERGED_REL_CHANGE * 100.0,
                );
                if let Some((_, w)) = log.as_mut() {
                    writeln!(
                        w,
                        "# converged at iter {} (max_rel_change={:.4} < {:.4})",
                        iter + 1, max_rel, CONVERGED_REL_CHANGE,
                    )?;
                }
                break;
            }
        }
        last_curve = curve;
    }

    Ok(curve)
}

/// Ordinary-least-squares fit on log-log. Skips non-positive points.
/// Returns `(slope, intercept)` of `log(y) = intercept + slope * log(x)`.
fn log_linear_fit(samples: &[(f64, f64)]) -> (f64, f64) {
    let pts: Vec<(f64, f64)> = samples
        .iter()
        .filter(|(t, y)| *t > 0.0 && *y > 0.0)
        .map(|(t, y)| (t.ln(), y.ln()))
        .collect();
    assert!(
        pts.len() >= 2,
        "need at least 2 positive (target, measured) samples to fit"
    );
    let n = pts.len() as f64;
    let mean_x = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let num: f64 = pts.iter().map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();
    let den: f64 = pts.iter().map(|(x, _)| (x - mean_x).powi(2)).sum();
    let slope = num / den;
    let intercept = mean_y - slope * mean_x;
    (slope, intercept)
}

// ─── IPC + CSV log glue ────────────────────────────────────────────────────

fn apply_panel_peaks(output: &str, peaks: [f64; 3]) -> Result<()> {
    send_action(
        output,
        OutputAction::PanelPeakNits {
            nits_r: peaks[0],
            nits_g: peaks[1],
            nits_b: peaks[2],
        },
    )
    .context("apply PanelPeakNits")
}

/// Fire a one-shot IPC request against the running prism. The server
/// closes after each reply, so each request opens a fresh connection.
fn send_action(output: &str, action: OutputAction) -> Result<()> {
    let mut socket = Socket::connect().context("connect to PRISM_SOCKET")?;
    let reply = socket
        .send(Request::Output {
            output: output.to_string(),
            action,
        })
        .context("send request / read reply")?;
    match reply {
        Ok(Response::OutputConfigChanged(_)) => Ok(()),
        Ok(other) => anyhow::bail!("unexpected reply: {other:?}"),
        Err(e) => anyhow::bail!("prism returned error: {e}"),
    }
}

fn open_log(
    args: &CalibrateArgs,
    baseline: &OutputBaseline,
) -> Result<Option<(PathBuf, BufWriter<File>)>> {
    if args.no_log {
        return Ok(None);
    }
    let path = args.log.clone().unwrap_or_else(|| {
        // Substitute slashes in the connector name so a hypothetical
        // "DP-0/1" doesn't escape into a subdirectory.
        let safe = args.output.replace('/', "_");
        PathBuf::from(format!("prism-tune-calibrate-{safe}.csv"))
    });
    if path.as_os_str() == "/dev/null" {
        return Ok(None);
    }
    let file = File::create(&path)
        .with_context(|| format!("create log file {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "# prism-tune calibrate — output={} mode={} iterations={} settle_ms={} window={}",
        args.output,
        if baseline.hdr_active { "HDR" } else { "SDR" },
        args.iterations, args.settle_ms, args.window,
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
        "phase,channel,iter,patch_idx,target_nits,applied_gain_r,applied_gain_g,applied_gain_b,applied_gamma_r,applied_gamma_g,applied_gamma_b,X,Y,Z,x,y"
    )?;
    eprintln!("Logging per-measurement CSV to {}", path.display());
    Ok(Some((path, w)))
}

// ─── KDL output ────────────────────────────────────────────────────────────

fn print_kdl_block(
    output_name: &str,
    hdr_active: bool,
    peaks: [f64; 3],
    curve: PerChannelCurve,
) {
    println!();
    println!("# Paste into the matching output block in your prism config:");
    println!("output \"{}\" {{", output_name);
    println!("    color {{");
    if hdr_active {
        // Per-channel panel peak is the calibration's headline output
        // in HDR mode — it's what aligns the f32 IR clamp with measured
        // reality and what the HDR_OUTPUT_METADATA infoframe carries.
        println!(
            "        panel-peak-nits r={:.1} g={:.1} b={:.1}",
            peaks[0], peaks[1], peaks[2]
        );
    }
    println!(
        "        response-curve gain-r={:.4} gain-g={:.4} gain-b={:.4} gamma-r={:.4} gamma-g={:.4} gamma-b={:.4}",
        curve.gain[0], curve.gain[1], curve.gain[2],
        curve.gamma[0], curve.gamma[1], curve.gamma[2],
    );
    println!("    }}");
    println!("}}");
    if !hdr_active {
        // For SDR-only panels the per-channel peaks are diagnostic, not
        // applied. Print them as a commented note so the reader can spot
        // panels that can't reach sdr_reference_nits.
        println!(
            "# Measured per-channel SDR peaks (not auto-applied; sdr-reference-nits is policy):"
        );
        println!(
            "#   R={:.1} G={:.1} B={:.1} cd/m²",
            peaks[0], peaks[1], peaks[2]
        );
    }
}
