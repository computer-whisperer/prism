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
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::thread;
use std::time::Duration;
use tristim_display::PatchSurface;
use tristim_driver::{measurement::raw_to_xyz, Calibration, Colorimeter, Setup, Xyz};

use crate::common::{
    apply_border, apply_panel_peaks, open_patch_surface, query_output_baseline, send_action,
    set_channel_patch, set_patch_off, set_white_patch, show_alignment_patch, Channel,
    OutputBaseline,
};
use prism_ipc::OutputAction;

#[derive(Args)]
pub struct CalibrateArgs {
    /// Connector to calibrate (e.g. `DisplayPort-4`, `HDMI-A-1`).
    /// Use the long form — recent prism builds match the connector-
    /// driver name verbatim, not the `DP-N` shorthand.
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
    /// Border luminance (cd/m²) painted around the centered patch.
    /// Default 50 — high enough to keep most panels' content-adaptive
    /// backlight from gating off during low-intensity measurements
    /// (the LU28R55 turns the backlight off entirely below ~30 cd/m²
    /// frame average, which then reads dim patches as black). The
    /// colorimeter only sees the centered patch so the border doesn't
    /// pollute measurements.
    ///
    /// In SDR mode the value is converted to an equivalent sRGB
    /// fraction via `border_nits / sdr_reference_nits`.
    #[arg(long, default_value_t = 50.0)]
    pub border_nits: f64,
    /// Disable the border entirely (legacy black surround). Use if a
    /// panel's CABL isn't an issue and you want the cleanest signal
    /// — small benefit, but the option is here.
    #[arg(long)]
    pub no_border: bool,
    /// Per-stage D65 white-check target luminance (cd/m²). After each
    /// refine iteration's ResponseCurve push, and at each stage
    /// boundary (baseline, post-peaks, post-CTM), the calibrator
    /// briefly renders BT.2020 D65 white at this luminance and
    /// measures it. Use to diagnose where in the pipeline the white
    /// point drifts. Default 10 — below typical weakest-subpixel peak
    /// (so no per-channel clamp interaction) and above the puck's
    /// noise floor.
    ///
    /// Pre-CTM stages produce non-D65 readings by design (BT.2020-
    /// equal-RGB hits the panel's native white, not D65) — the value
    /// to watch in those stages is *stability across iterations*. The
    /// post-CTM check should land near D65.
    #[arg(long, default_value_t = 10.0)]
    pub white_check_nits: f64,
}

/// What the calibrator is doing at the moment a row is logged. Used
/// to keep the CSV log self-describing — `probe` rows are the
/// saturation discovery phase, `refine` rows are the iterative gain/
/// gamma fit.
#[derive(Clone, Copy, Debug)]
enum Phase {
    Probe,
    Refine,
    Verify,
    /// Diagnostic D65 white probe inserted between stages. CSV rows
    /// use the `iter` column to disambiguate (0 = stage boundary,
    /// N = after refine iter N's ResponseCurve push); a preceding
    /// `# white_check stage=...` comment carries the human-readable
    /// label.
    WhiteCheck,
}

impl Phase {
    fn label(self) -> &'static str {
        match self {
            Self::Probe => "probe",
            Self::Refine => "refine",
            Self::Verify => "verify",
            Self::WhiteCheck => "white_check",
        }
    }
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
    let baseline =
        query_output_baseline(&args.output).context("query baseline output state via prism IPC")?;
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
    send_action(&args.output, OutputAction::ResetColor).context("initial ResetColor")?;

    // Open hardware. Probe colorimeter first — if the puck isn't
    // plugged in / udev rule missing we want to fail before the user
    // has held it for 30 seconds.
    let mut device = Colorimeter::open_any().context("open colorimeter")?;
    let info = device.get_info().context("read colorimeter info")?;
    eprintln!(
        "Colorimeter: Spyder SN {} HW {}.{:02}",
        info.serial, info.hw_version.0, info.hw_version.1
    );
    let cal = device
        .get_calibration(args.cal)
        .context("download cal matrix")?;
    let setup = device.get_setup(&cal).context("download setup")?;

    // Patch surface. Mode picks the constructor.
    let mut patch = open_patch_surface(&args.output, baseline.hdr_active)?;
    patch
        .set_window_fraction(args.window)
        .context("set window fraction")?;

    // Configure the border (anti-CABL surround) unless --no-border.
    // The border colour is held across all subsequent set_color /
    // set_nits calls — only the centred patch changes per measurement.
    if !args.no_border {
        apply_border(&mut patch, &baseline, args.border_nits)?;
        eprintln!(
            "Anti-CABL border set to {:.1} cd/m² (override with --border-nits or --no-border).",
            args.border_nits
        );
    } else {
        eprintln!("Border disabled (--no-border); surround is black.");
    }

    // Alignment patch during the countdown: paint a moderately-bright
    // gray in the centered window so the user can see exactly where
    // the patch will appear and position the puck precisely. Without
    // this the countdown happens with both centre and border at the
    // border colour (or pure black with --no-border), giving the user
    // nothing to align against.
    let alignment_nits = (args.border_nits * 2.0).max(40.0);
    show_alignment_patch(&mut patch, &baseline, alignment_nits)?;

    // CSV log — open up front so a permission/path error fails before
    // the user has held the puck.
    let mut log = open_log(&args, &baseline)?;

    eprintln!(
        "Place the puck flat on the centred patch on {} now. Calibration starts in {}s.",
        args.output, args.prep_secs
    );
    for s in (1..=args.prep_secs).rev() {
        eprintln!("  starting in {s}s...");
        thread::sleep(Duration::from_secs(1));
    }

    // Now go to black before the probe begins — the probe expects to
    // start from a known state. Border stays as configured.
    set_patch_off(&mut patch, baseline.hdr_active)?;

    // ─── Phase 1: per-channel saturation discovery ────────────────────
    let (discovered_peaks, probe_peak_y, measured_primaries, probe_cabl_count) =
        discover_per_channel_peaks(
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
    eprintln!(
        "  measured panel primaries (xy): R=({:.4}, {:.4})  G=({:.4}, {:.4})  B=({:.4}, {:.4})",
        measured_primaries[0].0,
        measured_primaries[0].1,
        measured_primaries[1].0,
        measured_primaries[1].1,
        measured_primaries[2].0,
        measured_primaries[2].1,
    );

    // HDR mode: DON'T set the clamp to probe_peak_y here. Refine
    // needs to probe commanded values above probe_peak_y per channel
    // — the per-channel curve fit characterises the panel by
    // inverting `emitted = gain × commanded^gamma`, which means to
    // make the panel emit 0.5×peak we have to command MORE than
    // 0.5×peak. If the IR is clamped at probe_peak_y mid-run, every
    // refine sample at a higher target collapses to identical IR,
    // identical commanded, identical Y → degenerate fit. We apply
    // probe_peak_y as the real clamp AFTER refine completes (just
    // before verify), so the verify phase tests the realistic
    // pipeline while refine sees the full panel response.
    //
    // SDR mode: log only — sdr_reference_nits is policy, not
    // measurement, and overriding it would conflict with the user's
    // vibrancy preference. Warn if a channel can't reach
    // sdr_reference_nits.
    if !baseline.hdr_active {
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

    // Baseline white check — identity curve, default IR clamp, no CTM.
    // BT.2020-equal-RGB on this panel renders as the panel's native
    // white (not D65). The reading anchors the pre-calibration state
    // so subsequent stage transitions are interpretable.
    let settle = Duration::from_millis(args.settle_ms);
    white_check_d65(
        "baseline_pre_refine",
        args.white_check_nits,
        settle,
        &PerChannelCurve::IDENTITY,
        &baseline,
        &mut device,
        &mut patch,
        &setup,
        &cal,
        log.as_mut(),
    )?;

    // ─── Phase 2: per-channel response refinement (CTM in the loop) ───
    // Each iter measures with the previous iter's (curve, CTM) active,
    // refits gain/gamma, recomputes the CTM from this iter's measured
    // primaries, and pushes both. Convergence is on the end-of-iter D65
    // white-check Δu'v', so the loop is gradient-descending on the
    // *full* calibration state, not just the curve.
    let (curve, ctm_inner, final_primaries, refine_cabl_count) = refine_per_channel_curve(
        &args,
        &baseline,
        &discovered_peaks,
        &probe_peak_y,
        &measured_primaries,
        &mut device,
        &mut patch,
        &setup,
        &cal,
        log.as_mut(),
    )?;

    // SDR mode: refine ran with identity CTM throughout (panel-native
    // path doesn't go through CTM in the SDR pipeline). Discard the
    // identity CTM rather than persisting it.
    let ctm: Option<[[f64; 3]; 3]> = if baseline.hdr_active {
        eprintln!("\nFinal CTM (BT.2020 → panel-native, row-major):");
        for row in &ctm_inner {
            eprintln!("  {:>9.5}  {:>9.5}  {:>9.5}", row[0], row[1], row[2]);
        }
        eprintln!(
            "Final measured primaries (xy): R=({:.4},{:.4}) G=({:.4},{:.4}) B=({:.4},{:.4})",
            final_primaries[0].0,
            final_primaries[0].1,
            final_primaries[1].0,
            final_primaries[1].1,
            final_primaries[2].0,
            final_primaries[2].1,
        );
        Some(ctm_inner)
    } else {
        eprintln!("\nSDR mode — no CTM applied.");
        None
    };

    // Now that refine has seen the full panel response unhindered,
    // commit the measured per-channel emitted peaks as the real IR
    // clamp. This is what aligns the f32 IR with calibrated reality
    // (and rebuilds the HDR_OUTPUT_METADATA infoframe). Done HERE,
    // between refine and verify, so verify tests the realistic
    // pipeline. HDR mode only — SDR's clamp is sdr_reference_nits
    // and isn't ours to override.
    if baseline.hdr_active {
        apply_panel_peaks(&args.output, probe_peak_y)?;
    }

    // Post-panel-peaks white check — same fitted curve + CTM, but IR
    // clamp dropped from default to per-channel measured peaks. A
    // white-point shift here means the clamp is interacting with the
    // calibrated pipeline.
    white_check_d65(
        "after_panel_peaks",
        args.white_check_nits,
        settle,
        &curve,
        &baseline,
        &mut device,
        &mut patch,
        &setup,
        &cal,
        log.as_mut(),
    )?;

    // ─── Phase 4: verify white point + luminance under final pipeline ─
    // Renders a D65 reference-white sweep with the live curve + CTM
    // applied and measures the result. Δu'v' from D65 reports how close
    // the calibrated stack lands on the reference white; Y-error reports
    // how close measured luminance lands on requested. Passive
    // diagnostic — no further IPC writes — so a bad result tells us to
    // re-look at the primaries / per-channel fit rather than auto-
    // correcting (which would mask the root cause).
    let verify = verify_white_point(
        &args,
        &baseline,
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
            probe_peak_y[0], probe_peak_y[1], probe_peak_y[2],
            curve.gain[0], curve.gain[1], curve.gain[2],
            curve.gamma[0], curve.gamma[1], curve.gamma[2],
        )?;
        if let Some(m) = ctm {
            writeln!(
                w,
                "# final CTM (row-major): {:.6} {:.6} {:.6}  {:.6} {:.6} {:.6}  {:.6} {:.6} {:.6}",
                m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
            )?;
        }
        w.flush().ok();
        eprintln!("CSV log written to {}", path.display());
    }

    // Backlight-off summary across the whole run. The fit filters out
    // affected samples but the user might still want to know — if the
    // count is non-zero, the border was too dim for this panel's CABL
    // threshold and a re-run with --border-nits N (N > current) would
    // produce more confident readings.
    let total_cabl = probe_cabl_count + refine_cabl_count;
    if total_cabl > 0 {
        eprintln!(
            "\n⚠  {} backlight-off event(s) detected during run (probe {} + refine {}).",
            total_cabl, probe_cabl_count, refine_cabl_count,
        );
        eprintln!("   The panel's content-adaptive backlight gated samples to ambient noise.");
        eprintln!(
            "   Affected samples were dropped from the fit; current curve uses surviving data."
        );
        eprintln!(
            "   Re-run with `--border-nits {}` (or higher) for cleaner results.",
            (args.border_nits * 2.0).round(),
        );
    } else {
        eprintln!(
            "\nNo backlight-off events detected — border at {:.0} cd/m² was sufficient.",
            args.border_nits
        );
    }

    // ─── Verify summary ────────────────────────────────────────────────
    // Δu'v' < 0.01 is "indistinguishable from reference" for most
    // viewers; < 0.02 is "good enough for general use"; > 0.04 is
    // visibly off. Y-error < 5% is within the colorimeter's own
    // uncertainty for desktop-class panels.
    let verdict = if verify.max_duv < 0.01 && verify.max_y_err_pct < 5.0 {
        "✓ EXCELLENT — calibration verified within colorimeter noise."
    } else if verify.max_duv < 0.02 && verify.max_y_err_pct < 10.0 {
        "✓ ACCEPTABLE — minor drift, usable for general desktop work."
    } else {
        "⚠ POOR — investigate measured primaries + per-channel curve before trusting."
    };
    eprintln!(
        "\nVerify: max Δu'v' from D65 = {:.4}, max Y-error = {:.1}%",
        verify.max_duv, verify.max_y_err_pct,
    );
    eprintln!("        {}", verdict);

    // ─── Print KDL block for paste-in ─────────────────────────────────
    print_kdl_block(&args.output, baseline.hdr_active, probe_peak_y, curve, ctm);

    if args.keep {
        eprintln!(
            "\n--keep: discovered panel peak + tuned curve remain active until prism restart."
        );
    } else {
        eprintln!("\nRestoring KDL config defaults (use --keep to leave the tuned values active).");
        send_action(&args.output, OutputAction::ResetColor).context("final ResetColor")?;
    }

    Ok(())
}

// ─── Phase 1 helpers ───────────────────────────────────────────────────────

/// Walk each channel ascending; find where measured Y plateaus.
///
/// Returns `(peaks, probe_peak_y, measured_primaries, backlight_off_count)`:
/// - `peaks[c]` = highest commanded value that produced a non-saturated
///   measurement (used as the panel-peak ceiling for clamping the
///   intermediate buffer).
/// - `probe_peak_y[c]` = highest measured Y observed for that channel
///   during probe (used by refinement to filter out saturation-
///   polluted samples — if `Y >= 0.95 × probe_peak_y[c]` the panel was
///   clipping and the data point doesn't reflect linear response).
/// - `measured_primaries[c]` = chromaticity (x, y) of the panel's
///   actual emission for channel `c`, taken from the highest-Y sample
///   in the channel's sweep. Used by Phase 3 to derive the gamut-
///   correction CTM. The brightest sample is the most reliable
///   chromaticity reading (chromaticity noise scales inversely with Y).
/// - `backlight_off_count` = number of measurements flagged by
///   [`is_backlight_off`] — surfaced to the final summary so the user
///   knows to bump `--border-nits` if CABL is gating samples.
fn discover_per_channel_peaks(
    args: &CalibrateArgs,
    baseline: &OutputBaseline,
    device: &mut Colorimeter,
    patch: &mut PatchSurface,
    setup: &Setup,
    cal: &Calibration,
    mut log: Option<&mut (PathBuf, BufWriter<File>)>,
) -> Result<([f64; 3], [f64; 3], [(f64, f64); 3], usize)> {
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
        vec![r * 0.05, r * 0.15, r * 0.35, r * 0.65, r]
    };

    let settle = Duration::from_millis(args.settle_ms);
    let mut peaks = [0.0f64; 3];
    let mut probe_peak_y = [0.0f64; 3];
    let mut measured_primaries = [(0.0f64, 0.0f64); 3];
    let mut backlight_off_count = 0usize;

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
            let cabl = is_backlight_off(target_nits, &xyz);
            eprintln!(
                "  {} target {:>7.1} cd/m²  →  X={:>7.2}  Y={:>7.2}  Z={:>7.2}{}",
                c.label(),
                target_nits,
                xyz.x,
                xyz.y,
                xyz.z,
                if cabl { "  ⚠ backlight-off" } else { "" },
            );
            if cabl {
                backlight_off_count += 1;
                if let Some((_, w)) = log.as_mut() {
                    writeln!(
                        w,
                        "# WARN backlight-off detected (probe {} patch_idx {}, target {:.2}): xy=({:.4},{:.4}) Y={:.4}",
                        c.label(), patch_idx + 1, target_nits, cx, cy, xyz.y,
                    )?;
                }
            }
            measurements.push((target_nits, xyz.y));
            if xyz.y > probe_peak_y[c.idx()] {
                probe_peak_y[c.idx()] = xyz.y;
                measured_primaries[c.idx()] = (cx, cy);
            }
            if let Some((_, w)) = log.as_mut() {
                writeln!(
                    w,
                    "{},{},1,{},{:.4},1.0,1.0,1.0,1.0,1.0,1.0,{:.4},{:.4},{:.4},{:.6},{:.6}",
                    Phase::Probe.label(),
                    c.label(),
                    patch_idx + 1,
                    target_nits,
                    xyz.x,
                    xyz.y,
                    xyz.z,
                    cx,
                    cy,
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
        //
        // Walk-back: the coarse-ramp saturation test (`y_ratio < 1.2`)
        // only catches *clear* saturation between consecutive samples,
        // not the soft knee. So `measurements[i-1]` (the supposed
        // "last non-saturated" sample) can itself sit on the knee, with
        // its Y already nearly at the saturation plateau. If we bisect
        // (measurements[i-1], measurements[i]) in that case, every
        // bisection sample saturates and we never find a true non-sat
        // upper bound — `discovered_peaks` ends up overestimated.
        //
        // Detect this by checking if measurements[i-1].y is within 5%
        // of the saturation Y (= measurements[i].y). If so, walk
        // further back until we find a sample whose Y is clearly less,
        // and bisect from THERE. Result: bisection starts with a
        // clearly-non-saturated left, narrows in toward the true
        // cliff.
        let (left_t, left_y) = match saturated_at {
            Some(sat_i) => {
                let sat_y = measurements[sat_i].1;
                let mut left_idx = sat_i - 1;
                while left_idx > 0 && measurements[left_idx].1 / sat_y.max(0.01) > 0.95 {
                    left_idx -= 1;
                }
                if left_idx < sat_i - 1 {
                    eprintln!(
                        "  {} walk-back: sample at {:.1} cd/m² is within 5% of saturation \
                         Y ({:.2} vs {:.2}); bisecting from {:.1} cd/m² instead",
                        c.label(),
                        measurements[sat_i - 1].0,
                        measurements[sat_i - 1].1,
                        sat_y,
                        measurements[left_idx].0,
                    );
                }
                let right_t = measurements[left_idx + 1].0;
                let lt = measurements[left_idx].0;
                let ly = measurements[left_idx].1;
                eprintln!(
                    "  {} bisecting cliff in ({:.1}, {:.1}) cd/m² (3 steps)…",
                    c.label(),
                    lt,
                    right_t,
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
                    if is_backlight_off(mid_t, &xyz) {
                        backlight_off_count += 1;
                        if let Some((_, w)) = log.as_mut() {
                            writeln!(
                                w,
                                "# WARN backlight-off detected during {} bisection (target {:.2}): xy=({:.4},{:.4}) Y={:.4}",
                                c.label(), mid_t, cx, cy, mid_y,
                            )?;
                        }
                    }
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
                    if mid_y > probe_peak_y[c.idx()] {
                        probe_peak_y[c.idx()] = mid_y;
                        measured_primaries[c.idx()] = (cx, cy);
                    }
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
            let (px, py) = measured_primaries[c.idx()];
            writeln!(
                w,
                "# probe {} measured peak commanded = {:.3} cd/m² (after bisection), peak_y = {:.3}, primary xy = ({:.4}, {:.4})",
                c.label(),
                left_t,
                probe_peak_y[c.idx()],
                px, py,
            )?;
        }
    }

    Ok((peaks, probe_peak_y, measured_primaries, backlight_off_count))
}

/// Detect a backlight-off / dead measurement on a pure-channel patch.
///
/// Some panels (the LU28R55 in HDR mode is the example we hit during
/// DP-4 calibration) gate the backlight off entirely when frame-
/// average brightness falls below a threshold. If the colorimeter is
/// looking at the centred patch while this happens, the reading is
/// ambient noise rather than the intended colour.
///
/// Detection is pure magnitude: when the panel's backlight gates off
/// on a pure-channel patch, Y collapses to ambient floor (~0.08 on the
/// LU28R55). A previous version also checked chromaticity against
/// hardcoded sRGB-ish primary anchors, but real panel primaries can
/// land far from those anchors (the PG27UCDM QD-OLED's G primary is
/// at xy=(0.254, 0.704), ~0.1 from a 0.27/0.60 sRGB-G anchor),
/// causing every real measurement on that panel to be flagged as
/// CABL and zeroing the fit. Magnitude alone is panel-agnostic and
/// catches every real CABL event we've observed.
///
/// Returns `true` when the sample looks like a CABL event:
/// - target is non-trivial (> 5 nits — below this we can't
///   distinguish "panel emits little" from "backlight off"), AND
/// - measured Y is below the absolute noise floor (`Y < 0.3`).
fn is_backlight_off(target_nits: f64, xyz: &Xyz) -> bool {
    target_nits > 5.0 && xyz.y < 0.3
}

// ─── Phase 2 helpers ───────────────────────────────────────────────────────

/// Full-pipeline iterative refinement — each iter measures with the
/// previous iter's (curve, CTM) active, refits per-channel gain/gamma,
/// recomputes the CTM from this iter's measured primaries, and pushes
/// **both** as updates to the live compositor state. The next iter
/// then measures through the refined pipeline. The loop is gradient-
/// descending on the full calibration state rather than just the
/// curve; convergence is decided by the end-of-iter D65 white-check
/// Δu'v', since that's the actual quality signal we care about.
///
/// Inputs: per-channel `discovered_peaks` and `probe_peak_y` from
/// Phase 1 (used for adaptive-target range and saturation filtering);
/// `initial_primaries` also from Phase 1 (seeds the first iter's CTM
/// derivation, refined each iter from highest-Y per-channel samples).
///
/// Key design: **targets are adaptive per iteration AND CTM-aware**.
/// We pick five log-spaced *commanded* values in `[COMMANDED_LO,
/// 0.85 × discovered_peak]` and map them to panel-native targets
/// `T = gain_k × commanded^gamma_k` via the current curve V_k. Each
/// target is then sent as a BT.2020 patch of value `T / ctm[c][c]`,
/// which after the compositor's CTM produces panel-native c-channel
/// drive = `T`. The fit math (log-linear on `Y` vs panel-native
/// target) is unchanged from the pre-CTM version — the CTM only
/// affects what BT.2020 value we send.
///
/// HDR vs SDR: CTM is HDR-only (SDR pipeline doesn't route through
/// it). In SDR mode the CTM stays identity throughout, no CTM IPC is
/// emitted, and the BT.2020-send compensation collapses to identity.
/// Fit semantics in SDR are unchanged from pre-CTM calibrate.
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
/// Early-exit on convergence: when the end-of-iter white-check
/// Δu'v' from D65 drops below `CONVERGED_DUV` (0.005), the full
/// pipeline has landed on the reference white and we stop.
fn refine_per_channel_curve(
    args: &CalibrateArgs,
    baseline: &OutputBaseline,
    discovered_peaks: &[f64; 3],
    probe_peak_y: &[f64; 3],
    initial_primaries: &[(f64, f64); 3],
    device: &mut Colorimeter,
    patch: &mut PatchSurface,
    setup: &Setup,
    cal: &Calibration,
    mut log: Option<&mut (PathBuf, BufWriter<File>)>,
) -> Result<(PerChannelCurve, [[f64; 3]; 3], [(f64, f64); 3], usize)> {
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
    /// Convergence threshold for early-exit, in Δu'v' from D65 at the
    /// end-of-iter white-check. 0.005 is below the visual just-
    /// noticeable-difference; the per-iter measurement noise on a
    /// clean run is ~0.001-0.002.
    const CONVERGED_DUV: f64 = 0.005;
    /// Per-iter Δu'v' improvement below this is "no useful progress."
    /// Two consecutive non-improving iters → exit even if we haven't
    /// hit `CONVERGED_DUV`. Some panels have a hard residual floor
    /// (multi-channel ABL, intensity-dependent primaries) that no
    /// number of iterations will push past; this stops us burning
    /// minutes on iters that aren't moving the needle.
    const NO_IMPROVEMENT_DELTA: f64 = 0.001;
    /// Consecutive non-improving iters required to declare "stuck."
    const STABLE_ITERS_FOR_EXIT: usize = 2;
    /// Fraction of `probe_peak_y[c]` we aim at when picking which
    /// per-channel sample provides the primary-chromaticity reading
    /// for CTM derivation. Real LCDs/OLEDs show meaningful primary
    /// drift with intensity (e.g. R primary x can move by 0.04+ over
    /// the operating range); using the peak-Y reading builds a CTM
    /// matched only to peak emission. 0.4 lands closer to the panel's
    /// typical operating Y for desktop content, trading a small
    /// near-peak miss for a much-improved mid-Y white point.
    const PRIMARY_SAMPLE_Y_FRAC: f64 = 0.4;
    const IDENTITY_CTM: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let mut curve = PerChannelCurve::IDENTITY;
    let mut primaries = *initial_primaries;
    // CTM starts identity. End of iter 1 derives the first real CTM
    // from this iter's measured primaries and pushes it; iter 2 then
    // measures through that CTM, refines, and pushes a new one. The
    // outer convergence on Δu'v' decides when we're done.
    let mut ctm: [[f64; 3]; 3] = IDENTITY_CTM;
    let settle = Duration::from_millis(args.settle_ms);
    let mut backlight_off_count = 0usize;
    // Track last iter's white-check Δu'v' to detect plateau (panel
    // physics has a floor we can't beat; stop once we hit it).
    let mut last_duv: Option<f64> = None;
    let mut no_improvement_count: usize = 0;

    for iter in 0..args.iterations {
        eprintln!(
            "\n--- refine iter {} / {} (curve gain={:?}, gamma={:?}, ctm diag=[{:.3},{:.3},{:.3}]) ---",
            iter + 1,
            args.iterations,
            curve.gain,
            curve.gamma,
            ctm[0][0], ctm[1][1], ctm[2][2],
        );

        // Per-iter primary capture: pick the sample whose Y lies
        // closest to a *mid-range* operating point per channel, NOT the
        // highest-Y sample. Rationale: panels' per-primary chromaticity
        // drifts with commanded value (typical LCD R primary moves
        // from xy≈(0.64, 0.31) at low cmd to (0.68, 0.31) at peak).
        // Using peak-Y primaries builds a CTM that's correct only at
        // peak emission — at mid/low operating Y it predicts a
        // more-saturated emission than the panel actually produces,
        // shifting the white point away from D65 (typically toward
        // green/cyan as R becomes effectively less red). The mid-range
        // primary is a better match for the bulk of viewing
        // conditions. Falls back to previous iter's primary for
        // channels that have no useful sample.
        let mut iter_primary: [Option<(f64, f64, f64)>; 3] = [None; 3]; // (y_distance_to_target, x, y_chroma)

        for c in Channel::ALL {
            // Adaptive targets: pick commanded values in the panel's
            // working range, then back-solve targets via the current
            // curve so commanded actually lands where we want it.
            // `target` here is the panel-NATIVE per-primary desired
            // emission in nits — what the inverse-curve in the
            // compositor sees as its input AFTER the CTM has mapped
            // BT.2020 → panel-native.
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

            // BT.2020 send-value compensation: with CTM active in the
            // compositor, sending BT.2020 c-only at `s` produces
            // panel-native c-component of `ctm[c][c] * s` (off-diagonal
            // products are small for diagonal-dominant CTMs and clip
            // to zero in the shader for pure-primary inputs). To make
            // the panel-native desired-emit be `t`, send `t / ctm[c][c]`
            // through wp_color_management. When ctm is identity this
            // collapses to `t`, matching pre-CTM behaviour exactly.
            let ctm_diag = ctm[c.idx()][c.idx()].max(1e-6);

            let mut samples: Vec<(f64, f64, bool)> = Vec::with_capacity(targets.len());
            for (patch_idx, &t) in targets.iter().enumerate() {
                let bt2020_send = t / ctm_diag;
                set_channel_patch(patch, baseline, c, bt2020_send)?;
                thread::sleep(settle);
                let raw = device.measure_raw(setup).context("measure")?;
                let xyz = raw_to_xyz(&raw, setup, cal);
                let (cx, cy) = xyz.chromaticity().unwrap_or((0.0, 0.0));
                let cabl = is_backlight_off(t, &xyz);
                if cabl {
                    backlight_off_count += 1;
                }
                eprintln!(
                    "  {} target {:>7.2} (send {:>7.2}) → X={:>7.2} Y={:>7.2} Z={:>7.2}{}",
                    c.label(),
                    t,
                    bt2020_send,
                    xyz.x,
                    xyz.y,
                    xyz.z,
                    if cabl { "  ⚠ backlight-off" } else { "" },
                );
                samples.push((t, xyz.y, cabl));

                // Track this iter's primary chromaticity for this
                // channel — sample whose Y is closest to a mid-range
                // operating target wins (PRIMARY_SAMPLE_Y_FRAC of the
                // channel's probed peak). See iter_primary doc-comment
                // for the rationale (intensity-dependent primary drift).
                let y_sat_check = probe_peak_y[c.idx()] * SATURATION_FILTER_FRAC;
                if !cabl && xyz.y >= NOISE_FLOOR_Y && xyz.y <= y_sat_check {
                    let target_y = probe_peak_y[c.idx()] * PRIMARY_SAMPLE_Y_FRAC;
                    let dist = (xyz.y - target_y).abs();
                    if iter_primary[c.idx()].map_or(true, |(prev_dist, _, _)| dist < prev_dist) {
                        iter_primary[c.idx()] = Some((dist, cx, cy));
                    }
                }

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

            // Filter all three: backlight-off (CABL), noise-floor,
            // saturation-polluted. Each catches a different failure
            // mode of the underlying model assumption (clean power-law
            // response in the panel's linear region).
            let y_sat = probe_peak_y[c.idx()] * SATURATION_FILTER_FRAC;
            let useful: Vec<(f64, f64)> = samples
                .iter()
                .copied()
                .filter(|(_, y, cabl)| !*cabl && *y >= NOISE_FLOOR_Y && *y <= y_sat)
                .map(|(t, y, _)| (t, y))
                .collect();
            if useful.len() < 2 {
                eprintln!(
                    "  {} only {} useful sample(s) after CABL+noise+saturation filter; skipping fit",
                    c.label(),
                    useful.len(),
                );
                if let Some((_, w)) = log.as_mut() {
                    writeln!(
                        w,
                        "# refine {} iter {} fit: SKIPPED (insufficient samples after CABL+noise+saturation filter)",
                        c.label(),
                        iter + 1,
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
                    c.label(),
                    fit_gamma,
                    GAMMA_PLAUSIBLE,
                );
                if let Some((_, w)) = log.as_mut() {
                    writeln!(
                        w,
                        "# refine {} iter {} fit: REJECTED (gamma={:.4} out of plausible range)",
                        c.label(),
                        iter + 1,
                        fit_gamma,
                    )?;
                }
                continue;
            }

            // Plausibility: predicted top-Y vs measured top-Y. The
            // panel's response is `Y = fit_gain × commanded^fit_gamma`
            // (per the linearisation comment above), NOT
            // `Y = fit_gain × target^fit_gamma`. The previous version
            // used `target` directly, which only worked when the
            // current curve V_k was identity (target == commanded);
            // for iter 2+ with a non-identity curve, it silently
            // produced wrong predictions and rejected valid fits.
            // Convert target → commanded via the inverse of V_k.
            let (top_target, top_measured_y) = *useful.last().unwrap();
            let top_commanded = (top_target / gain_k.max(1e-6)).powf(1.0 / gamma_k.max(1e-3));
            let predicted_top_y = fit_gain * top_commanded.powf(fit_gamma);
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
                c.label(),
                fit_gain,
                fit_gamma,
            );
            if let Some((_, w)) = log.as_mut() {
                writeln!(
                    w,
                    "# refine {} iter {} fit: applied gain={:.6} gamma={:.6}",
                    c.label(),
                    iter + 1,
                    fit_gain,
                    fit_gamma,
                )?;
            }
        }

        // Update primaries from this iter's measurements. Channels
        // with no useful sample keep their previous-iter primary —
        // protects CTM derivation from a single degenerate channel.
        for i in 0..3 {
            if let Some((_, x, y)) = iter_primary[i] {
                primaries[i] = (x, y);
            }
        }

        // CTM is HDR-only — the SDR pipeline doesn't route through it.
        // In SDR mode we keep `ctm` at identity throughout, so the
        // bt2020_send compensation above is a no-op and we never push
        // a CTM IPC. Fit semantics in SDR are unchanged from pre-CTM
        // calibrate.
        if baseline.hdr_active {
            match compute_ctm(&primaries) {
                Ok(new_ctm) => ctm = new_ctm,
                Err(e) => {
                    eprintln!("  WARN: CTM recompute failed ({e:#}); keeping previous CTM.");
                    if let Some((_, w)) = log.as_mut() {
                        writeln!(
                            w,
                            "# refine iter {} CTM recompute FAILED: {} (keeping previous)",
                            iter + 1,
                            e,
                        )?;
                    }
                }
            }
        }

        if let Some((_, w)) = log.as_mut() {
            writeln!(
                w,
                "# refine iter {} primaries: R=({:.4},{:.4}) G=({:.4},{:.4}) B=({:.4},{:.4})",
                iter + 1,
                primaries[0].0,
                primaries[0].1,
                primaries[1].0,
                primaries[1].1,
                primaries[2].0,
                primaries[2].1,
            )?;
            if baseline.hdr_active {
                writeln!(
                    w,
                    "# refine iter {} ctm: R=({:.6},{:.6},{:.6}) G=({:.6},{:.6},{:.6}) B=({:.6},{:.6},{:.6})",
                    iter + 1,
                    ctm[0][0], ctm[0][1], ctm[0][2],
                    ctm[1][0], ctm[1][1], ctm[1][2],
                    ctm[2][0], ctm[2][1], ctm[2][2],
                )?;
            }
        }

        // Push CTM first (HDR only) so the ResponseCurve push sees a
        // freshly mapped IR — minimises the window where the panel
        // sees mismatched state.
        if baseline.hdr_active {
            apply_ctm(&args.output, &ctm).context("push iter CTM")?;
        }
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

        // Stage-boundary diagnostic — render BT.2020 D65 through the
        // just-pushed full pipeline (curve + CTM). This is the
        // convergence signal: Δu'v' from D65 below CONVERGED_DUV means
        // the full state has landed on the reference white.
        let duv = white_check_d65(
            &format!("refine_iter{}", iter + 1),
            args.white_check_nits,
            settle,
            &curve,
            baseline,
            device,
            patch,
            setup,
            cal,
            log.as_deref_mut(),
        )?;

        if duv < CONVERGED_DUV {
            eprintln!(
                "\nConverged after iter {} (Δu'v'={:.4} < {:.4} threshold).",
                iter + 1,
                duv,
                CONVERGED_DUV,
            );
            if let Some((_, w)) = log.as_mut() {
                writeln!(
                    w,
                    "# converged at iter {} (delta_uv={:.5} < {:.5})",
                    iter + 1,
                    duv,
                    CONVERGED_DUV,
                )?;
            }
            break;
        }

        // Plateau detection — if we're not making meaningful progress,
        // the panel has hit its calibration floor and more iterations
        // won't help. Counts any iter where Δu'v' improved by less
        // than NO_IMPROVEMENT_DELTA (regression also counts as "not
        // improving"). Two consecutive such iters → exit.
        if let Some(prev) = last_duv {
            let improvement = prev - duv;
            if improvement < NO_IMPROVEMENT_DELTA {
                no_improvement_count += 1;
                eprintln!(
                    "  iter {} no useful improvement (Δu'v' {:.4} → {:.4}, delta {:+.4}); stable_count={}/{}",
                    iter + 1, prev, duv, -improvement,
                    no_improvement_count, STABLE_ITERS_FOR_EXIT,
                );
                if no_improvement_count >= STABLE_ITERS_FOR_EXIT {
                    eprintln!(
                        "\nPlateau after iter {} (Δu'v' stable around {:.4} for {} iters; panel-physics floor reached).",
                        iter + 1, duv, no_improvement_count + 1,
                    );
                    if let Some((_, w)) = log.as_mut() {
                        writeln!(
                            w,
                            "# plateau at iter {} (delta_uv={:.5} stable for {} iters)",
                            iter + 1,
                            duv,
                            no_improvement_count + 1,
                        )?;
                    }
                    break;
                }
            } else {
                no_improvement_count = 0;
            }
        }
        last_duv = Some(duv);
    }

    Ok((curve, ctm, primaries, backlight_off_count))
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

// ─── Phase 4 helpers ───────────────────────────────────────────────────────

/// Verify the calibrated pipeline by sweeping reference white at a
/// range of luminances and measuring how close the panel lands on the
/// expected D65 chromaticity + commanded luminance.
///
/// Passive — does not iterate the curve or matrix. A miss here means
/// either:
///   - measured primaries are off (Δu'v' large, Y-error small),
///   - per-channel curve is off (Y-error large, Δu'v' small),
///   - or the panel's ABL / per-zone-dim is interfering at the test
///     luminance (both large, often with low-Y samples).
///
/// HDR vs SDR target selection differs:
///   - HDR: log-space `0.05 × hi` → `hi`, where `hi = 0.8 × min(probe_peak_y)`.
///     The cap matters: BT.2020 D65 white at `L` nits maps (after CTM)
///     to approximately `(L, L, L)` panel-native, so the weakest
///     subpixel's emitted-peak limits how bright we can render true
///     D65. On a panel with a weak B subpixel (e.g. DP-4 with peak_y_B
///     ≈ 12), the sweep tops out around 10 nits — that's a feature of
///     the panel, not the test. Going past `min(peak_y)` would clip
///     individual channels and skew the chromaticity away from D65,
///     turning verify into a measurement of the panel's max-drive
///     native white instead.
///   - SDR: fixed fractions of `sdr_reference_nits` (`[0.1, 0.25, 0.5,
///     0.75, 0.95]`) — staying inside the [0, 1] sRGB encoding range
///     so `set_color` doesn't clip.
fn verify_white_point(
    args: &CalibrateArgs,
    baseline: &OutputBaseline,
    probe_peak_y: &[f64; 3],
    device: &mut Colorimeter,
    patch: &mut PatchSurface,
    setup: &Setup,
    cal: &Calibration,
    mut log: Option<&mut (PathBuf, BufWriter<File>)>,
) -> Result<VerifyResult> {
    const D65: (f64, f64) = (0.3127, 0.3290);
    let (d65_up, d65_vp) = xy_to_uv_prime(D65);

    let targets: Vec<f64> = if baseline.hdr_active {
        let min_peak_y = probe_peak_y.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = (min_peak_y * 0.8).max(2.0);
        let lo = (hi * 0.05).max(1.0);
        let lo_ln = lo.ln();
        let hi_ln = hi.max(lo * 1.5).ln();
        (0..5)
            .map(|i| {
                let f = i as f64 / 4.0;
                (lo_ln + f * (hi_ln - lo_ln)).exp()
            })
            .collect()
    } else {
        let r = baseline.sdr_reference_nits;
        vec![0.10, 0.25, 0.50, 0.75, 0.95]
            .into_iter()
            .map(|f| f * r)
            .collect()
    };

    let settle = Duration::from_millis(args.settle_ms);
    let mut max_duv = 0.0_f64;
    let mut max_y_err_pct = 0.0_f64;

    eprintln!("\n--- phase 4 verify: D65 white sweep ---");
    if let Some((_, w)) = log.as_mut() {
        writeln!(
            w,
            "# phase 4 verify: D65 white sweep — Δu'v' from D65=({:.4},{:.4})",
            D65.0, D65.1,
        )?;
    }

    for (patch_idx, &t) in targets.iter().enumerate() {
        set_white_patch(patch, baseline, t)?;
        thread::sleep(settle);
        let raw = device.measure_raw(setup).context("measure (verify)")?;
        let xyz = raw_to_xyz(&raw, setup, cal);
        let (cx, cy) = xyz.chromaticity().unwrap_or((0.0, 0.0));
        let (up, vp) = xy_to_uv_prime((cx, cy));
        let duv = ((up - d65_up).powi(2) + (vp - d65_vp).powi(2)).sqrt();
        let y_err_pct = (xyz.y - t) / t.max(0.01) * 100.0;

        eprintln!(
            "  W target {:>7.1} cd/m² → Y={:>7.2}  xy=({:.4},{:.4})  Δu'v'={:.4}  Y_err={:+.1}%",
            t, xyz.y, cx, cy, duv, y_err_pct,
        );

        max_duv = max_duv.max(duv);
        max_y_err_pct = max_y_err_pct.max(y_err_pct.abs());

        if let Some((_, w)) = log.as_mut() {
            // Match the standard column layout: phase, channel ("W"
            // for the white-sweep), iter (always 1), patch_idx, target,
            // the 6 applied-curve cells (unknown to verify — log 1.0
            // because the live curve isn't surfaced here), then XYZ + xy.
            writeln!(
                w,
                "{},W,1,{},{:.4},1.0,1.0,1.0,1.0,1.0,1.0,{:.4},{:.4},{:.4},{:.6},{:.6}",
                Phase::Verify.label(),
                patch_idx + 1,
                t,
                xyz.x,
                xyz.y,
                xyz.z,
                cx,
                cy,
            )?;
            writeln!(
                w,
                "# verify W patch {}: target_nits={:.3} measured_y={:.3} delta_uv={:.5} y_err_pct={:+.3}",
                patch_idx + 1, t, xyz.y, duv, y_err_pct,
            )?;
        }
    }

    Ok(VerifyResult {
        max_duv,
        max_y_err_pct,
    })
}

/// CIE 1976 uniform chromaticity (u', v') from xy. Used for Δu'v'
/// distance against the D65 reference white — perceptually more
/// uniform than raw xy distance so a single threshold (Δu'v' < 0.01)
/// means roughly the same thing across the diagram.
fn xy_to_uv_prime(xy: (f64, f64)) -> (f64, f64) {
    let (x, y) = xy;
    let denom = -2.0 * x + 12.0 * y + 3.0;
    if denom.abs() < 1e-9 {
        return (0.0, 0.0);
    }
    (4.0 * x / denom, 9.0 * y / denom)
}

/// Summary of the verify-phase pass — max chromaticity drift and max
/// luminance error across the white sweep. The thresholds applied to
/// these in `run()` decide the verdict line.
struct VerifyResult {
    max_duv: f64,
    max_y_err_pct: f64,
}

/// Render a centered BT.2020 D65 white patch at `target_nits`, measure
/// it, and report. Used as a stability probe at stage boundaries — if
/// the panel's emission of a fixed white target swings between stages
/// (or between iterations of the same stage), the per-output color
/// pipeline is applying inconsistent state.
///
/// `stage` is a free-form label written to both the console output and
/// the CSV log so a stage-by-stage trace can be reconstructed offline.
///
/// `target_nits` should be low enough to stay under every per-channel
/// ceiling — at 10 cd/m² we're well below typical B-subpixel peaks
/// (~15 on the LU28R55) so no clamp interaction can confound the
/// reading.
///
/// **Δu'v' interpretation:** PRE-CTM, BT.2020-equal-RGB renders as the
/// panel's NATIVE white (which is not D65 on narrower-than-BT.2020
/// gamuts) — a non-zero Δu'v' here is expected. The value to watch
/// pre-CTM is *consistency across iterations*. POST-CTM the reading
/// should land near D65 (Δu'v' < 0.01).
fn white_check_d65(
    stage: &str,
    target_nits: f64,
    settle: Duration,
    curve: &PerChannelCurve,
    baseline: &OutputBaseline,
    device: &mut Colorimeter,
    patch: &mut PatchSurface,
    setup: &Setup,
    cal: &Calibration,
    log: Option<&mut (PathBuf, BufWriter<File>)>,
) -> Result<f64> {
    const D65: (f64, f64) = (0.3127, 0.3290);
    let (d65_up, d65_vp) = xy_to_uv_prime(D65);

    set_white_patch(patch, baseline, target_nits)?;
    thread::sleep(settle);
    let raw = device.measure_raw(setup).context("measure (white_check)")?;
    let xyz = raw_to_xyz(&raw, setup, cal);
    let (cx, cy) = xyz.chromaticity().unwrap_or((0.0, 0.0));
    let (up, vp) = xy_to_uv_prime((cx, cy));
    let duv = ((up - d65_up).powi(2) + (vp - d65_vp).powi(2)).sqrt();
    let y_err_pct = (xyz.y - target_nits) / target_nits.max(0.01) * 100.0;

    eprintln!(
        "  ⊕ white-check [{}] target {:.1} cd/m² → Y={:>6.2} xy=({:.4},{:.4}) Δu'v'={:.4} Y_err={:+.1}%",
        stage, target_nits, xyz.y, cx, cy, duv, y_err_pct,
    );

    if let Some((_, w)) = log {
        writeln!(
            w,
            "# white_check stage={} target_nits={:.3} measured_y={:.3} measured_xy=({:.6},{:.6}) delta_uv_from_d65={:.5} y_err_pct={:+.3}",
            stage, target_nits, xyz.y, cx, cy, duv, y_err_pct,
        )?;
        writeln!(
            w,
            "{},W,0,1,{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4},{:.4},{:.4},{:.6},{:.6}",
            Phase::WhiteCheck.label(),
            target_nits,
            curve.gain[0],
            curve.gain[1],
            curve.gain[2],
            curve.gamma[0],
            curve.gamma[1],
            curve.gamma[2],
            xyz.x,
            xyz.y,
            xyz.z,
            cx,
            cy,
        )?;
    }

    Ok(duv)
}

// ─── Phase 3 helpers ───────────────────────────────────────────────────────

/// Derive a 3×3 BT.2020-to-panel CTM from measured panel primaries
/// and apply it via IPC.
///
/// **Math (units matter — this got us once already):**
///
/// The IR holds BT.2020 RGB tristimulus values in the standard
/// normalized convention where `(L, L, L)` = D65 white at total
/// luminance `L`. The downstream per-channel curve, however, was
/// calibrated against *panel-native per-primary nits* — its input is
/// "desired emit per primary in cd/m²", not "panel-native normalized
/// RGB". So the CTM must convert from BT.2020 normalized space
/// directly to per-primary panel-native nits.
///
/// The correct CTM is `M_phys_panel⁻¹ · M_bt2020`, where:
/// - `M_phys_panel` is the panel's RGB→XYZ matrix with un-normalized
///   columns `(X_i/Y_i, 1, Z_i/Y_i)` (per-primary at Y=1, no D65
///   normalization). Inverse converts XYZ to per-primary panel nits.
/// - `M_bt2020` is the standard BT.2020 RGB→XYZ matrix (D65-normalized
///   so `(1, 1, 1)` → `XYZ_D65 = (0.9504, 1, 1.0890)`). Converts
///   BT.2020 normalized RGB to XYZ.
///
/// Composite: BT.2020 RGB → XYZ → panel-native per-primary nits.
///
/// For BT.2020 D65 white input `(L, L, L)`, the CTM produces
/// `(S_R_panel · L, S_G_panel · L, S_B_panel · L)` where
/// `(S_R, S_G, S_B)` are the panel's per-primary contributions to
/// D65 (sum = 1.0 in luminance). The per-channel curve then inverts
/// panel response so the panel actually emits those nits per primary
/// — giving total Y = L and chromaticity = D65.
///
/// **Earlier bug:** previous version used `M_panel_d65norm⁻¹` (D65-
/// normalized both sides), which made `CTM · (1, 1, 1)` produce
/// `(1, 1, 1)` — interpreted by the per-channel curve as "emit 1 nit
/// per primary" instead of "emit S_R/S_G/S_B per primary." Result was
/// the panel emitted ~3× the target luminance with non-D65
/// chromaticity. Caught during white-sweep characterization (2026-05-22).
///
/// In-gamut BT.2020 inputs produce non-negative panel-native outputs.
/// Out-of-gamut inputs (BT.2020 saturations the panel can't reach —
/// most edges of the BT.2020 triangle on a P3-class panel) produce
/// negative components which the per-channel curve clips to zero,
/// introducing hue rotation at the gamut boundary. Documented trade-off.
fn compute_ctm(measured_primaries: &[(f64, f64); 3]) -> Result<[[f64; 3]; 3]> {
    // BT.2020 primaries + D65 reference white (ITU-R BT.2020-2).
    const BT2020_R: (f64, f64) = (0.708, 0.292);
    const BT2020_G: (f64, f64) = (0.170, 0.797);
    const BT2020_B: (f64, f64) = (0.131, 0.046);
    const D65: (f64, f64) = (0.3127, 0.3290);

    let m_phys_panel = build_phys_matrix(measured_primaries);
    let m_bt2020 = build_rgb_to_xyz(&[BT2020_R, BT2020_G, BT2020_B], D65)
        .expect("BT.2020 primaries are well-formed (non-singular by construction)");
    let m_phys_panel_inv =
        mat3_inverse(&m_phys_panel).context("singular panel primaries (cannot derive CTM)")?;
    Ok(mat3_mul(&m_phys_panel_inv, &m_bt2020))
}

fn apply_ctm(output: &str, ctm: &[[f64; 3]; 3]) -> Result<()> {
    send_action(
        output,
        OutputAction::Ctm {
            rr: ctm[0][0],
            rg: ctm[0][1],
            rb: ctm[0][2],
            gr: ctm[1][0],
            gg: ctm[1][1],
            gb: ctm[1][2],
            br: ctm[2][0],
            bg: ctm[2][1],
            bb: ctm[2][2],
        },
    )
    .context("apply CTM")
}

/// Convert chromaticity `(x, y)` to XYZ with Y normalized to 1.
fn xy_to_xyz(xy: (f64, f64)) -> [f64; 3] {
    let (x, y) = xy;
    // Guard against degenerate (0, 0) input — `xyz.chromaticity()`
    // returns `None` for all-zero XYZ and we default to (0, 0).
    if y <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [x / y, 1.0, (1.0 - x - y) / y]
}

/// Un-normalized "physical" matrix with columns = per-primary XYZ at
/// `Y=1`. Multiplying this by `(R_emit, G_emit, B_emit)` per-primary
/// nits gives the panel's emitted XYZ in nits — the right form for
/// CTM derivation where the per-channel curve operates in per-primary
/// nits, not normalized RGB. Compare to [`build_rgb_to_xyz`] which
/// further scales columns to D65-normalize the matrix.
fn build_phys_matrix(primaries: &[(f64, f64); 3]) -> [[f64; 3]; 3] {
    let p_r = xy_to_xyz(primaries[0]);
    let p_g = xy_to_xyz(primaries[1]);
    let p_b = xy_to_xyz(primaries[2]);
    [
        [p_r[0], p_g[0], p_b[0]],
        [p_r[1], p_g[1], p_b[1]],
        [p_r[2], p_g[2], p_b[2]],
    ]
}

/// Standard RGB→XYZ matrix construction from primaries + white point.
///
/// Forms the un-normalized matrix with primary XYZ as columns, then
/// scales each column so that `M · (1, 1, 1)ᵀ = XYZ_white`. Returns
/// `None` if the primaries are degenerate (collinear → singular).
fn build_rgb_to_xyz(primaries: &[(f64, f64); 3], white: (f64, f64)) -> Option<[[f64; 3]; 3]> {
    let p_r = xy_to_xyz(primaries[0]);
    let p_g = xy_to_xyz(primaries[1]);
    let p_b = xy_to_xyz(primaries[2]);
    let m_unnorm: [[f64; 3]; 3] = [
        [p_r[0], p_g[0], p_b[0]],
        [p_r[1], p_g[1], p_b[1]],
        [p_r[2], p_g[2], p_b[2]],
    ];
    let m_inv = mat3_inverse(&m_unnorm)?;
    let xyz_white = xy_to_xyz(white);
    let s = mat3_mul_vec(&m_inv, &xyz_white);
    Some([
        [
            m_unnorm[0][0] * s[0],
            m_unnorm[0][1] * s[1],
            m_unnorm[0][2] * s[2],
        ],
        [
            m_unnorm[1][0] * s[0],
            m_unnorm[1][1] * s[1],
            m_unnorm[1][2] * s[2],
        ],
        [
            m_unnorm[2][0] * s[0],
            m_unnorm[2][1] * s[1],
            m_unnorm[2][2] * s[2],
        ],
    ])
}

/// 3×3 matrix determinant (cofactor expansion along row 0).
fn mat3_det(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// 3×3 matrix inverse via adjugate / determinant. Returns `None` when
/// the matrix is singular within `1e-12`.
fn mat3_inverse(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = mat3_det(m);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

/// 3×3 × 3×3 matrix multiplication.
fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// 3×3 matrix × 3-vector multiplication.
fn mat3_mul_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `build_rgb_to_xyz` with BT.2020 primaries + D65 must match the
    /// canonical published matrix (ITU-R BT.2020-2). Tolerance is 1e-3
    /// — the published 4-decimal matrix is itself rounded.
    #[test]
    fn build_rgb_to_xyz_bt2020_d65_matches_canonical() {
        let m = build_rgb_to_xyz(
            &[(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)],
            (0.3127, 0.3290),
        )
        .expect("BT.2020 primaries non-singular");
        // Canonical BT.2020 RGB→XYZ (ITU-R BT.2020-2, derived with same
        // D65 normalisation).
        let expected = [
            [0.6370, 0.1446, 0.1689],
            [0.2627, 0.6780, 0.0593],
            [0.0000, 0.0281, 1.0610],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m[i][j] - expected[i][j]).abs() < 1e-3,
                    "m[{i}][{j}] = {} vs expected {}",
                    m[i][j],
                    expected[i][j],
                );
            }
        }
    }

    /// `build_phys_matrix` produces the un-normalized form: columns
    /// `(X_i/Y_i, 1, Z_i/Y_i)` per primary. For BT.2020 primaries the
    /// first column is `(0.708/0.292, 1, 0/0.292) = (2.425, 1, 0)`.
    #[test]
    fn build_phys_matrix_bt2020_columns() {
        let m = build_phys_matrix(&[(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)]);
        // Column R: (0.708/0.292, 1, 0/0.292) = (2.425, 1, 0)
        assert!((m[0][0] - 2.425).abs() < 1e-3);
        assert!((m[1][0] - 1.0).abs() < 1e-9);
        assert!((m[2][0] - 0.0).abs() < 1e-3);
        // Column G: (0.170/0.797, 1, 0.033/0.797) = (0.213, 1, 0.0414)
        assert!((m[0][1] - 0.213).abs() < 1e-3);
        assert!((m[1][1] - 1.0).abs() < 1e-9);
        assert!((m[2][1] - 0.0414).abs() < 1e-3);
        // Column B: (0.131/0.046, 1, 0.823/0.046) = (2.848, 1, 17.891)
        assert!((m[0][2] - 2.848).abs() < 1e-2);
        assert!((m[1][2] - 1.0).abs() < 1e-9);
        assert!((m[2][2] - 17.891).abs() < 1e-2);
    }

    /// When the panel's primaries equal BT.2020's, the derived CTM
    /// must reduce to `diag(0.2627, 0.6780, 0.0593)` — the BT.2020 D65
    /// luminance coefficients. That's because BT.2020 D65 white
    /// `(1, 1, 1)` must map to per-primary panel-native nits that
    /// produce D65 white at total Y=1: those nits are exactly the
    /// per-primary D65 contributions, i.e., the BT.2020 S values.
    /// (No gamut correction needed, but per-primary normalization is.)
    #[test]
    fn ctm_panel_matching_bt2020_is_diag_d65_weights() {
        let primaries = [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)];
        let m_phys = build_phys_matrix(&primaries);
        let m_bt2020 = build_rgb_to_xyz(&primaries, (0.3127, 0.3290)).unwrap();
        let m_phys_inv = mat3_inverse(&m_phys).unwrap();
        let ctm = mat3_mul(&m_phys_inv, &m_bt2020);
        let expected = [[0.2627, 0.0, 0.0], [0.0, 0.6780, 0.0], [0.0, 0.0, 0.0593]];
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (ctm[i][j] - expected[i][j]).abs() < 1e-3,
                    "ctm[{i}][{j}] = {} vs expected {}",
                    ctm[i][j],
                    expected[i][j],
                );
            }
        }
    }

    /// The CTM must convert BT.2020 D65 white `(1, 1, 1)` into the
    /// panel's per-primary D65 contributions. For any panel, those
    /// contributions sum to 1 (so total Y=1 → D65 white at Y=1).
    #[test]
    fn ctm_d65_white_input_sums_to_one() {
        // Use the DP-4 measured primaries for a non-trivial check.
        let primaries = [(0.6747, 0.3133), (0.2725, 0.6048), (0.1396, 0.0744)];
        let m_phys = build_phys_matrix(&primaries);
        let m_bt2020 = build_rgb_to_xyz(
            &[(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)],
            (0.3127, 0.3290),
        )
        .unwrap();
        let m_phys_inv = mat3_inverse(&m_phys).unwrap();
        let ctm = mat3_mul(&m_phys_inv, &m_bt2020);
        let out = mat3_mul_vec(&ctm, &[1.0, 1.0, 1.0]);
        let sum = out[0] + out[1] + out[2];
        // Per-primary nits for D65 at Y=1 must sum to 1.
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "CTM·(1,1,1) row sum = {sum}, expected 1.0 (panel-native nits for D65 at Y=1)"
        );
        // And each component must be positive (in-gamut for the panel).
        assert!(out[0] > 0.0 && out[1] > 0.0 && out[2] > 0.0);
    }

    /// `mat3_inverse` returns `None` for singular input (collinear
    /// primaries = degenerate triangle).
    #[test]
    fn mat3_inverse_singular_returns_none() {
        let singular = [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0], // row 1 = 2 × row 0
            [0.0, 1.0, 0.0],
        ];
        assert!(mat3_inverse(&singular).is_none());
    }

    /// D65 in CIE 1976 u'v' space is (0.1978, 0.4683). Sanity-check the
    /// conversion so the verify phase's "distance from D65" math has a
    /// known anchor.
    #[test]
    fn d65_xy_to_uv_prime_canonical() {
        let (up, vp) = xy_to_uv_prime((0.3127, 0.3290));
        assert!((up - 0.1978).abs() < 1e-3, "u' = {up}");
        assert!((vp - 0.4683).abs() < 1e-3, "v' = {vp}");
    }

    /// `mat3_inverse` then `mat3_mul` against the original must give
    /// identity within fp tolerance.
    #[test]
    fn mat3_inverse_roundtrip() {
        let m = [
            [0.637, 0.145, 0.169],
            [0.263, 0.678, 0.059],
            [0.000, 0.028, 1.061],
        ];
        let m_inv = mat3_inverse(&m).unwrap();
        let prod = mat3_mul(&m, &m_inv);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[i][j] - expected).abs() < 1e-9,
                    "prod[{i}][{j}] = {} vs expected {}",
                    prod[i][j],
                    expected,
                );
            }
        }
    }
}

// ─── CSV log glue ──────────────────────────────────────────────────────────

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
    let file =
        File::create(&path).with_context(|| format!("create log file {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "# prism-tune calibrate — output={} mode={} iterations={} settle_ms={} window={}",
        args.output,
        if baseline.hdr_active { "HDR" } else { "SDR" },
        args.iterations,
        args.settle_ms,
        args.window,
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

/// Print a `output { color { … } }` block for paste-in. `peaks` here is
/// per-channel measured EMITTED peak luminance (cd/m²) — i.e.
/// `probe_peak_y`, not the commanded-nits saturation cliff. That's the
/// number the compositor's IR clamp wants.
fn print_kdl_block(
    output_name: &str,
    hdr_active: bool,
    peaks: [f64; 3],
    curve: PerChannelCurve,
    ctm: Option<[[f64; 3]; 3]>,
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
    if let Some(m) = ctm {
        // 9 positional values in row-major order — matches the Ctm
        // knuffel struct (`#[knuffel(arguments)] values: Vec<f64>`).
        println!(
            "        ctm {:.6} {:.6} {:.6}  {:.6} {:.6} {:.6}  {:.6} {:.6} {:.6}",
            m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
        );
    }
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
