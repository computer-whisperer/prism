//! `prism-tune calibrate-lut3d` — measurement-driven 3D LUT calibration.
//!
//! Where the legacy `calibrate` fits a closed-form `(gain, gamma)` model
//! per channel and then synthesizes a 3D LUT from that, this tool skips
//! the model entirely. Per-channel sweeps capture the panel's actual
//! XYZ response at many commanded values; the inverse 3D LUT is then
//! built by Newton-Raphson — for each grid point in BT.2020 space we
//! solve `panel_R(cmd_R) + panel_G(cmd_G) + panel_B(cmd_B) = target_XYZ`
//! and store the commanded values.
//!
//! Why it matters: the closed-form model assumes the per-primary
//! chromaticity is constant across luminance, but real panels (LCDs
//! especially, see the DP-4 G primary drifting y from 0.52 → 0.61 over
//! the working range) don't behave that way. A LUT built from XYZ
//! measurements captures the drift directly.
//!
//! Additivity assumption: we measure each channel alone, then assume
//! `panel(R+G+B) = panel(R) + panel(G) + panel(B)`. Reasonable for the
//! small-window patches the colorimeter sees with a constant-luminance
//! border around them — APL stays roughly constant so ABL doesn't
//! cross-couple. A direct-3D-sweep variant could verify this; not yet.
//!
//! Output: a binary `.lut` file written to disk plus a paste-ready KDL
//! snippet pointing the output's config at it.

use anyhow::{Context, Result};
use clap::Args;
use prism_ipc::OutputAction;
use prism_renderer::{LUT_FILE_IN_TF_PQ, pq_eotf, save_lut3d_file};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::thread;
use std::time::Duration;
use tristim_driver::{Colorimeter, Xyz, measurement::raw_to_xyz};

use crate::common::{
    Channel, OutputBaseline, apply_border, apply_panel_peaks, open_patch_surface,
    query_output_baseline, send_action, send_action_for_reply, set_channel_patch, set_patch_off,
    set_white_patch, show_alignment_patch,
};
use prism_ipc::Response;
use tristim_display::PatchSurface;
use tristim_driver::{Calibration, Setup};

#[derive(Args)]
pub struct CalibrateLut3dArgs {
    /// Connector to calibrate (e.g. `DisplayPort-4`, `HDMI-A-1`).
    #[arg(long)]
    pub output: String,
    /// LUT cube edge (grid points per axis). Default 17 matches the
    /// compositor's default texture size. 33 gives finer precision at
    /// 8× the storage cost; only useful if 17³ shows banding.
    #[arg(long, default_value_t = 17)]
    pub cube_edge: u32,
    /// Per-channel measurement count. Each channel sweeps this many
    /// log-spaced commanded values from `--min-cmd` to its measured
    /// saturation. Higher = better forward-LUT precision = more
    /// accurate inversion, but linear time cost.
    #[arg(long, default_value_t = 33)]
    pub samples_per_channel: usize,
    /// Lowest commanded value (cd/m²) in the per-channel sweep.
    /// Defaults to 1.0 — below this the Spyder noise floor dominates
    /// and the readings can't anchor the inversion at the dim end.
    #[arg(long, default_value_t = 1.0)]
    pub min_cmd: f64,
    /// Highest commanded value (cd/m²) the sweep walks toward. We
    /// stop early once Y plateaus (saturation), so this is just a
    /// generous ceiling. Default 10000 = PQ peak.
    #[arg(long, default_value_t = 10000.0)]
    pub max_cmd: f64,
    /// Colorimeter calibration index (0..=6). 0 = the "General" preset.
    #[arg(long, default_value_t = 0)]
    pub cal: u8,
    /// Centered bright-window fraction (0..=1).
    #[arg(long, default_value_t = 0.10)]
    pub window: f64,
    /// Seconds to wait for the puck before the first sweep.
    #[arg(long, default_value_t = 5)]
    pub prep_secs: u64,
    /// Settle time after each color change before measuring (ms).
    #[arg(long, default_value_t = 32)]
    pub settle_ms: u64,
    /// Border luminance (cd/m²) painted around the centered patch —
    /// keeps CABL panels from gating off during low-intensity samples.
    #[arg(long, default_value_t = 50.0)]
    pub border_nits: f64,
    /// Disable the border entirely (black surround).
    #[arg(long)]
    pub no_border: bool,
    /// Output `.lut` path. Defaults to `prism-calibrate-lut3d-<output>.lut`
    /// in the current directory.
    #[arg(long)]
    pub lut_path: Option<PathBuf>,
    /// Optional measurement-log CSV path. Defaults to
    /// `prism-calibrate-lut3d-<output>.csv` (set `--no-log` to skip).
    #[arg(long)]
    pub log: Option<PathBuf>,
    /// Skip writing the measurement-log CSV.
    #[arg(long)]
    pub no_log: bool,
    /// Leave the discovered per-channel panel peaks AND the live-pushed
    /// LUT active on exit. Default: ResetColor so the next session
    /// re-reads from KDL (which still points at the file just written).
    #[arg(long)]
    pub keep: bool,
    /// Skip the post-calibration verification sweep. Default: render
    /// D65 white at several luminances through the freshly-pushed LUT
    /// and report Δu'v' from D65 + Y error per patch — the LUT
    /// pipeline is otherwise feed-forward (measure → invert → write),
    /// so closing the loop with a measurement is the only way to
    /// catch math regressions or panel-state surprises.
    #[arg(long)]
    pub no_verify: bool,
}

/// One (commanded, XYZ) measurement from a per-channel sweep.
#[derive(Clone, Copy, Debug)]
struct ChannelSample {
    /// Commanded value handed to the patch surface (cd/m² for HDR PQ,
    /// nits-equivalent for SDR sRGB encode at patch time).
    commanded: f64,
    xyz: Xyz,
}

/// Forward 1D response LUT for one primary — sorted by commanded value
/// and capped at the highest commanded value where the panel was still
/// responding (saturation cutoff).
#[derive(Clone, Debug)]
struct ChannelResponse {
    samples: Vec<ChannelSample>,
    /// Inclusive upper bound on `commanded` past which we treat the
    /// panel as saturated. Used to clamp the Newton-Raphson search
    /// space so we don't extrapolate into the cliff.
    max_cmd: f64,
    /// Peak emitted Y observed during the sweep. Stored alongside so
    /// the final panel-peak push uses measured-emitted values rather
    /// than commanded (the latter would over-promise on weak subpixels).
    peak_y: f64,
}

impl ChannelResponse {
    /// Linear interpolation of `commanded → XYZ` from the sorted
    /// samples. Inputs clamp to `[samples[0].commanded, max_cmd]` so
    /// the inversion can't read garbage outside the measured range.
    fn xyz_at(&self, commanded: f64) -> Xyz {
        let c = commanded
            .max(self.samples[0].commanded)
            .min(self.max_cmd);
        // Binary search for the segment.
        let idx = self
            .samples
            .binary_search_by(|s| s.commanded.partial_cmp(&c).unwrap())
            .unwrap_or_else(|i| i);
        if idx == 0 {
            return self.samples[0].xyz;
        }
        let hi = idx.min(self.samples.len() - 1);
        let lo = hi.saturating_sub(1);
        let lo_s = &self.samples[lo];
        let hi_s = &self.samples[hi];
        let span = hi_s.commanded - lo_s.commanded;
        if span <= 0.0 {
            return hi_s.xyz;
        }
        let t = ((c - lo_s.commanded) / span).clamp(0.0, 1.0);
        Xyz {
            x: lo_s.xyz.x + t * (hi_s.xyz.x - lo_s.xyz.x),
            y: lo_s.xyz.y + t * (hi_s.xyz.y - lo_s.xyz.y),
            z: lo_s.xyz.z + t * (hi_s.xyz.z - lo_s.xyz.z),
        }
    }

    /// Coarse Y-per-cmd gain from the brightest non-clamped sample.
    /// Used by the inverter to seed its initial guess — a single
    /// number per channel that's roughly the slope of the linear
    /// region. Skips the very first sample (often dominated by the
    /// colorimeter noise floor) to avoid the toe inflating the gain.
    fn approx_gain_y_per_cmd(&self) -> f64 {
        // Find the last sample at or below max_cmd (i.e. pre-cliff).
        let last = self
            .samples
            .iter()
            .rev()
            .find(|s| s.commanded <= self.max_cmd)
            .unwrap_or(self.samples.last().unwrap());
        last.xyz.y / last.commanded.max(1e-6)
    }

    /// `d(XYZ)/d(cmd)` at `commanded` via the slope of the surrounding
    /// linear segment. Returns the segment that brackets `commanded`
    /// (clamped to the LUT's range at the ends).
    fn dxyz_dcmd_at(&self, commanded: f64) -> [f64; 3] {
        let c = commanded
            .max(self.samples[0].commanded)
            .min(self.max_cmd);
        let idx = self
            .samples
            .binary_search_by(|s| s.commanded.partial_cmp(&c).unwrap())
            .unwrap_or_else(|i| i);
        let hi = idx.min(self.samples.len() - 1).max(1);
        let lo = hi - 1;
        let lo_s = &self.samples[lo];
        let hi_s = &self.samples[hi];
        let span = hi_s.commanded - lo_s.commanded;
        if span <= 1e-12 {
            return [0.0, 0.0, 0.0];
        }
        [
            (hi_s.xyz.x - lo_s.xyz.x) / span,
            (hi_s.xyz.y - lo_s.xyz.y) / span,
            (hi_s.xyz.z - lo_s.xyz.z) / span,
        ]
    }
}

pub fn run(args: CalibrateLut3dArgs) -> Result<()> {
    let baseline = query_output_baseline(&args.output)
        .context("query baseline output state via prism IPC")?;
    eprintln!(
        "Baseline for {}: mode={}, panel_peak={:?}, sdr_ref={}",
        args.output,
        if baseline.hdr_active { "HDR" } else { "SDR" },
        baseline.initial_panel_peak_nits,
        baseline.sdr_reference_nits,
    );

    // Wipe any runtime overrides, then lift the IR clamp (HDR) so the
    // panel sees raw commanded values during the sweep. SDR clamp stays
    // at sdr_reference_nits — that's the user's policy and we shouldn't
    // override it during measurement.
    send_action(&args.output, OutputAction::ResetColor)
        .context("initial ResetColor")?;
    if baseline.hdr_active {
        apply_panel_peaks(&args.output, [10_000.0, 10_000.0, 10_000.0])?;
    }
    // ResetColor only clears IPC overrides; KDL `color { ctm …
    // response-curve … }` stays active and the encode shader's LUT
    // gets re-synthesized from those values. That would silently
    // pre-transform every commanded value the sweep sends — measuring
    // panel response through the existing calibration instead of raw.
    // IdentityLut3d forces the LUT to identity regardless of KDL.
    send_action(&args.output, OutputAction::IdentityLut3d)
        .context("force identity LUT for raw-cmd sweep")?;

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
        eprintln!("Border disabled (--no-border); surround is black.");
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

    // ─── Phase 1: per-channel sweeps ──────────────────────────────────────
    let settle = Duration::from_millis(args.settle_ms);
    let targets =
        log_spaced_targets(args.min_cmd, args.max_cmd, args.samples_per_channel);
    let mut responses: [Option<ChannelResponse>; 3] = [None, None, None];

    for channel in Channel::ALL {
        eprintln!("\n--- {} channel sweep ---", channel.label());
        let mut samples: Vec<ChannelSample> = Vec::with_capacity(targets.len());
        let mut max_cmd = targets[0];
        let mut peak_y = 0.0_f64;
        for (i, &cmd) in targets.iter().enumerate() {
            set_channel_patch(&mut patch, &baseline, channel, cmd)?;
            thread::sleep(settle);
            let raw = device.measure_raw(&setup).context("measure")?;
            let xyz = raw_to_xyz(&raw, &setup, &cal);
            eprintln!(
                "  {} cmd {:>8.2} → X={:>8.3}  Y={:>8.3}  Z={:>8.3}",
                channel.label(),
                cmd,
                xyz.x,
                xyz.y,
                xyz.z,
            );
            if let Some((_, w)) = log.as_mut() {
                writeln!(
                    w,
                    "{},{},{:.4},{:.4},{:.4},{:.4}",
                    channel.label(),
                    i + 1,
                    cmd,
                    xyz.x,
                    xyz.y,
                    xyz.z,
                )?;
            }
            // Saturation: if Y is no longer increasing meaningfully and
            // we've already taken at least 3 samples, stop early. Catches
            // the cliff without forcing the rest of the sweep through it.
            //
            // Guard against false positives at the noise floor: the
            // Spyder reads ~0.3 cd/m² of ambient even on black, so the
            // first couple of B samples on a weak-blue panel can show
            // Y in the [0.3, 0.5] range where consecutive-sample ratios
            // are pure noise. Requiring BOTH samples to be above 1.0
            // cd/m² keeps the cliff-detector from triggering on toe-
            // region wobble — the panel's actual saturation lives well
            // above 1 nit per channel.
            const SATURATION_NOISE_FLOOR_Y: f64 = 1.0;
            if let Some(prev) = samples.last() {
                let ratio = xyz.y / prev.xyz.y.max(0.01);
                let cmd_ratio = cmd / prev.commanded.max(0.01);
                let both_above_floor =
                    xyz.y > SATURATION_NOISE_FLOOR_Y && prev.xyz.y > SATURATION_NOISE_FLOOR_Y;
                if samples.len() >= 3
                    && cmd_ratio >= 1.2
                    && ratio < 1.05
                    && both_above_floor
                {
                    eprintln!(
                        "  {} saturation detected at cmd {:.1} (Y ratio {:.2} vs cmd ratio {:.2}); \
                         stopping sweep early",
                        channel.label(), cmd, ratio, cmd_ratio,
                    );
                    max_cmd = prev.commanded;
                    break;
                }
            }
            if xyz.y > peak_y {
                peak_y = xyz.y;
            }
            max_cmd = cmd;
            samples.push(ChannelSample { commanded: cmd, xyz });
        }
        if samples.len() < 4 {
            anyhow::bail!(
                "{} channel: only {} usable sample(s) before saturation — too few to invert",
                channel.label(),
                samples.len(),
            );
        }
        eprintln!(
            "  {} forward LUT: {} samples, max_cmd={:.1}, peak_y={:.2}",
            channel.label(),
            samples.len(),
            max_cmd,
            peak_y,
        );
        if let Some((_, w)) = log.as_mut() {
            writeln!(
                w,
                "# {} forward LUT: samples={} max_cmd={:.3} peak_y={:.3}",
                channel.label(),
                samples.len(),
                max_cmd,
                peak_y,
            )?;
        }
        responses[channel.idx()] = Some(ChannelResponse {
            samples,
            max_cmd,
            peak_y,
        });
    }
    let responses = [
        responses[0].take().unwrap(),
        responses[1].take().unwrap(),
        responses[2].take().unwrap(),
    ];

    // ─── Phase 2: inversion to 3D LUT ─────────────────────────────────────
    eprintln!("\nInverting forward LUTs to {}³ 3D LUT…", args.cube_edge);
    let (entries, residuals) = build_inverse_lut(args.cube_edge, &responses);
    if let Some((_, w)) = log.as_mut() {
        // Split residual stats by whether the target was in-gamut for
        // the panel (target_Y ≤ panel total peak). Out-of-gamut grids
        // always have huge residuals — the inverter clamps cmd to
        // max_cmd and the panel physically can't reach the target —
        // so mixing them into the percentile dilutes the signal we
        // actually care about: "did Newton converge for the points
        // verify will actually sample?"
        let panel_total_peak: f64 = responses.iter().map(|r| r.peak_y).sum();
        let n = args.cube_edge as usize;
        let denom = (args.cube_edge - 1) as f32;
        let mut in_gamut: Vec<f64> = Vec::new();
        let mut out_of_gamut: Vec<f64> = Vec::new();
        for k in 0..n {
            let bz_in = pq_eotf(k as f32 / denom) as f64;
            for j in 0..n {
                let g_in = pq_eotf(j as f32 / denom) as f64;
                for i in 0..n {
                    let r_in = pq_eotf(i as f32 / denom) as f64;
                    let target_y = bt2020_to_xyz(r_in, g_in, bz_in)[1];
                    let idx = (k * n + j) * n + i;
                    if target_y <= panel_total_peak {
                        in_gamut.push(residuals[idx]);
                    } else {
                        out_of_gamut.push(residuals[idx]);
                    }
                }
            }
        }
        let pct = |v: &mut Vec<f64>, p: f64| {
            if v.is_empty() {
                return f64::NAN;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v[((v.len() - 1) as f64 * p) as usize]
        };
        writeln!(
            w,
            "# inversion residuals (in-gamut, target_Y ≤ {:.1} cd/m²): n={} p50={:.4} p90={:.4} p99={:.4} max={:.4} cd/m²",
            panel_total_peak,
            in_gamut.len(),
            pct(&mut in_gamut, 0.50),
            pct(&mut in_gamut, 0.90),
            pct(&mut in_gamut, 0.99),
            pct(&mut in_gamut, 1.0),
        )?;
        writeln!(
            w,
            "# inversion residuals (out-of-gamut): n={} p50={:.4} p90={:.4} p99={:.4} max={:.4} cd/m² (expected — panel cap'd)",
            out_of_gamut.len(),
            pct(&mut out_of_gamut, 0.50),
            pct(&mut out_of_gamut, 0.90),
            pct(&mut out_of_gamut, 0.99),
            pct(&mut out_of_gamut, 1.0),
        )?;
    }
    let peak_nits = [
        responses[0].peak_y as f32,
        responses[1].peak_y as f32,
        responses[2].peak_y as f32,
    ];

    // ─── Phase 3: write LUT + restore ─────────────────────────────────────
    let lut_path = args.lut_path.clone().unwrap_or_else(|| {
        let safe = args.output.replace('/', "_");
        PathBuf::from(format!("prism-calibrate-lut3d-{safe}.lut"))
    });
    save_lut3d_file(&lut_path, args.cube_edge, peak_nits, &entries)
        .with_context(|| format!("write LUT file {}", lut_path.display()))?;
    eprintln!("\nWrote {} (cube_edge={}, peaks={:?})", lut_path.display(), args.cube_edge, peak_nits);

    // Apply discovered peaks (HDR mode only) so the IR clamp matches
    // measured reality. SDR keeps its policy-driven peak (sdr_reference_nits).
    if baseline.hdr_active {
        apply_panel_peaks(
            &args.output,
            [peak_nits[0] as f64, peak_nits[1] as f64, peak_nits[2] as f64],
        )?;
    }

    // Push the LUT live so verify (and the user's first impression)
    // sees the new calibration without a prism restart. The path
    // resolution happens server-side, so we hand over the absolute
    // form to be safe regardless of where prism's CWD is.
    let lut_abs_path = std::fs::canonicalize(&lut_path)
        .with_context(|| format!("canonicalize {} for IPC", lut_path.display()))?;
    send_action(
        &args.output,
        OutputAction::LoadLut3dFromFile {
            path: lut_abs_path.to_string_lossy().into_owned(),
        },
    )
    .context("push LUT via IPC")?;
    eprintln!("Pushed LUT live via IPC ({}).", lut_abs_path.display());

    // ─── Phase 4: verify the live LUT against D65 ─────────────────────────
    let verify_result = if !args.no_verify {
        eprintln!("\n--- phase 4 verify: D65 white sweep through live LUT ---");
        Some(verify_white_point(
            &args, &baseline, &peak_nits, &entries, &responses, &mut device, &mut patch, &setup, &cal, log.as_mut(),
        )?)
    } else {
        eprintln!("\n(verify skipped — --no-verify)");
        None
    };

    set_patch_off(&mut patch, baseline.hdr_active)?;

    if let Some((path, mut w)) = log {
        writeln!(
            w,
            "# inverse LUT written to {} (cube_edge={}, in_tf={}, peaks=R={:.3} G={:.3} B={:.3})",
            lut_path.display(),
            args.cube_edge,
            LUT_FILE_IN_TF_PQ,
            peak_nits[0],
            peak_nits[1],
            peak_nits[2],
        )?;
        w.flush().ok();
        eprintln!("Measurement log: {}", path.display());
    }

    if let Some(verify) = verify_result.as_ref() {
        let verdict = if verify.max_duv < 0.01 && verify.max_y_err_pct < 5.0 {
            "✓ EXCELLENT — calibration verified within colorimeter noise."
        } else if verify.max_duv < 0.02 && verify.max_y_err_pct < 10.0 {
            "✓ ACCEPTABLE — minor drift, usable for general desktop work."
        } else {
            "⚠ POOR — investigate forward-LUT measurements + Newton residuals before trusting."
        };
        eprintln!(
            "\nVerify: max Δu'v' from D65 = {:.4}, max Y-error = {:.1}%",
            verify.max_duv, verify.max_y_err_pct,
        );
        eprintln!("        {verdict}");
    }

    print_kdl_block(&args.output, baseline.hdr_active, peak_nits, &lut_path);

    if !args.keep {
        eprintln!(
            "\nRestoring KDL defaults (use --keep to leave the live-pushed LUT + peaks active)."
        );
        send_action(&args.output, OutputAction::ResetColor)
            .context("final ResetColor")?;
    } else {
        eprintln!(
            "\n--keep: live-pushed LUT and per-channel panel peaks remain active until prism restart."
        );
    }
    Ok(())
}

/// Result of the verify-phase D65 sweep. Same shape as `calibrate`'s
/// version — max chromaticity drift + max luminance error, used to
/// pick the verdict line.
struct VerifyResult {
    max_duv: f64,
    max_y_err_pct: f64,
}

/// Render BT.2020 D65 white at a range of luminances through the
/// freshly-pushed LUT and measure how close the panel lands on the
/// reference. Δu'v' large means the LUT's chromaticity inversion is
/// off; Y-error large means the LUT's luminance inversion is off.
/// Both means measurement noise or a non-additive panel that broke
/// the additivity assumption.
///
/// Targets mirror `calibrate`'s verify phase so reports are comparable
/// across the two pipelines:
/// - HDR: log-space `[0.05 × hi, hi]` with `hi = 0.8 × min(peak_y)`.
/// - SDR: fixed fractions of `sdr_reference_nits`.
#[allow(clippy::too_many_arguments)]
fn verify_white_point(
    args: &CalibrateLut3dArgs,
    baseline: &OutputBaseline,
    peak_nits: &[f32; 3],
    lut_entries: &[[f32; 3]],
    responses: &[ChannelResponse; 3],
    device: &mut tristim_driver::Colorimeter,
    patch: &mut PatchSurface,
    setup: &Setup,
    cal: &Calibration,
    mut log: Option<&mut (PathBuf, BufWriter<File>)>,
) -> Result<VerifyResult> {
    const D65: (f64, f64) = (0.3127, 0.3290);
    let (d65_up, d65_vp) = xy_to_uv_prime(D65);

    let targets: Vec<f64> = if baseline.hdr_active {
        let min_peak_y = peak_nits.iter().copied().fold(f32::INFINITY, f32::min) as f64;
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

        // Diagnostic 1: mirror the shader's LUT lookup for this verify
        // patch so we can attribute Y_err. Sample the LUT at the PQ-
        // encoded input coord (target nits per channel for D65 white),
        // compute the expected commanded cmd via trilinear interp,
        // then ask the per-channel forward LUTs what those cmds would
        // emit (additive sum). Compare to actual measurement:
        //   - additive_predicted ≈ measured → panel is additive,
        //     LUT entries are right.
        //   - additive_predicted ≠ measured → panel non-additive,
        //     additivity assumption is the source of Y_err.
        let coord = [pq_oetf_f64(t), pq_oetf_f64(t), pq_oetf_f64(t)];
        let cmd_predicted = trilinear_sample_lut(lut_entries, args.cube_edge, coord);
        let xyz_r = responses[0].xyz_at(cmd_predicted[0] as f64);
        let xyz_g = responses[1].xyz_at(cmd_predicted[1] as f64);
        let xyz_b = responses[2].xyz_at(cmd_predicted[2] as f64);
        let additive_y =
            xyz_r.y + xyz_g.y + xyz_b.y;
        let additive_y_err_pct = (xyz.y - additive_y) / additive_y.max(0.01) * 100.0;

        // Diagnostic 2: ask the compositor what its encode pipeline
        // actually emits for this input. Compares the CPU prediction
        // above against the GPU's real output — isolates shader-side
        // bugs (wrong LUT upload, wrong trilinear, wrong push constants)
        // from the additivity assumption. CPU `cmd_predicted` vs GPU
        // `scanout_decoded` should be ≈ identical if the LUT path is
        // correct; any drift is a bug in the renderer.
        let gpu_diag = send_action_for_reply(
            &args.output,
            OutputAction::EncodeDiagnose { r: t, g: t, b: t },
        )
        .context("EncodeDiagnose IPC")?;
        let scanout_decoded = match gpu_diag {
            Response::EncodeDiagnose(r) => r.scanout_nits,
            other => anyhow::bail!("unexpected EncodeDiagnose reply: {other:?}"),
        };
        // Predict from forward LUT: what additive Y would the GPU-
        // decoded cmd produce? This is what verify SHOULD see if the
        // panel is additive (separate from the CPU-cmd prediction
        // above, which depends on accurate trilinear).
        let gpu_xyz_r = responses[0].xyz_at(scanout_decoded[0]);
        let gpu_xyz_g = responses[1].xyz_at(scanout_decoded[1]);
        let gpu_xyz_b = responses[2].xyz_at(scanout_decoded[2]);
        let gpu_additive_y = gpu_xyz_r.y + gpu_xyz_g.y + gpu_xyz_b.y;

        eprintln!(
            "  W target {:>7.1} cd/m² → Y={:>7.2}  xy=({:.4},{:.4})  Δu'v'={:.4}  Y_err={:+.1}%",
            t, xyz.y, cx, cy, duv, y_err_pct,
        );
        eprintln!(
            "      CPU lut cmd=({:.2},{:.2},{:.2})  GPU scanout cmd=({:.2},{:.2},{:.2})  CPU vs GPU drift R/G/B={:+.2}/{:+.2}/{:+.2}",
            cmd_predicted[0], cmd_predicted[1], cmd_predicted[2],
            scanout_decoded[0], scanout_decoded[1], scanout_decoded[2],
            scanout_decoded[0] - cmd_predicted[0] as f64,
            scanout_decoded[1] - cmd_predicted[1] as f64,
            scanout_decoded[2] - cmd_predicted[2] as f64,
        );
        eprintln!(
            "      additive-predicted Y from GPU cmd={:.2}  measured Y={:.2}  panel additivity error={:+.1}%",
            gpu_additive_y, xyz.y,
            (xyz.y - gpu_additive_y) / gpu_additive_y.max(0.01) * 100.0,
        );
        let _ = additive_y;
        let _ = additive_y_err_pct;
        max_duv = max_duv.max(duv);
        max_y_err_pct = max_y_err_pct.max(y_err_pct.abs());

        if let Some((_, w)) = log.as_mut() {
            writeln!(
                w,
                "verify,W,{},{:.4},{:.4},{:.4},{:.4}",
                patch_idx + 1, t, xyz.x, xyz.y, xyz.z,
            )?;
            writeln!(
                w,
                "# verify W patch {}: target_nits={:.3} measured_y={:.3} delta_uv={:.5} y_err_pct={:+.3}",
                patch_idx + 1, t, xyz.y, duv, y_err_pct,
            )?;
            writeln!(
                w,
                "#   cpu_lut_cmd=({:.3},{:.3},{:.3}) gpu_scanout_cmd=({:.3},{:.3},{:.3})",
                cmd_predicted[0], cmd_predicted[1], cmd_predicted[2],
                scanout_decoded[0], scanout_decoded[1], scanout_decoded[2],
            )?;
            writeln!(
                w,
                "#   gpu_additive_predicted_y={:.4} panel_additivity_err_pct={:+.3}",
                gpu_additive_y,
                (xyz.y - gpu_additive_y) / gpu_additive_y.max(0.01) * 100.0,
            )?;
        }
    }

    Ok(VerifyResult { max_duv, max_y_err_pct })
}

/// SMPTE ST 2084 (PQ) OETF: linear nits → encoded `[0, 1]`. CPU mirror
/// of the encode-shader's PQ shaper, used by verify to compute the
/// coord the renderer would sample the LUT at for a given input nits.
fn pq_oetf_f64(nits: f64) -> f64 {
    const M1: f64 = 0.1593017578125;
    const M2: f64 = 78.84375;
    const C1: f64 = 0.8359375;
    const C2: f64 = 18.8515625;
    const C3: f64 = 18.6875;
    let yn = (nits.max(0.0) / 10000.0).powf(M1);
    let num = C1 + C2 * yn;
    let den = 1.0 + C3 * yn;
    (num / den).powf(M2)
}

/// Trilinear sample the 3D LUT entries at `coord ∈ [0,1]³`. CPU
/// mirror of the GPU's sampler3D LINEAR filter so verify can compute
/// "what cmd values the shader would emit" without a roundtrip
/// through Vulkan. Entries are X-fastest then Y then Z (matches the
/// upload + binary format).
fn trilinear_sample_lut(entries: &[[f32; 3]], cube_edge: u32, coord: [f64; 3]) -> [f32; 3] {
    let n = cube_edge as usize;
    let denom = (cube_edge - 1) as f64;
    // Map [0, 1] → [0, N-1] continuous index, then split into integer
    // base and fractional weight.
    let cx = coord[0].clamp(0.0, 1.0) * denom;
    let cy = coord[1].clamp(0.0, 1.0) * denom;
    let cz = coord[2].clamp(0.0, 1.0) * denom;
    let i0 = (cx.floor() as usize).min(n - 1);
    let j0 = (cy.floor() as usize).min(n - 1);
    let k0 = (cz.floor() as usize).min(n - 1);
    let i1 = (i0 + 1).min(n - 1);
    let j1 = (j0 + 1).min(n - 1);
    let k1 = (k0 + 1).min(n - 1);
    let tx = (cx - i0 as f64) as f32;
    let ty = (cy - j0 as f64) as f32;
    let tz = (cz - k0 as f64) as f32;
    let idx = |i: usize, j: usize, k: usize| (k * n + j) * n + i;
    let lerp3 = |a: [f32; 3], b: [f32; 3], t: f32| {
        [
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2]),
        ]
    };
    // Trilinear: lerp along x, then y, then z.
    let c000 = entries[idx(i0, j0, k0)];
    let c100 = entries[idx(i1, j0, k0)];
    let c010 = entries[idx(i0, j1, k0)];
    let c110 = entries[idx(i1, j1, k0)];
    let c001 = entries[idx(i0, j0, k1)];
    let c101 = entries[idx(i1, j0, k1)];
    let c011 = entries[idx(i0, j1, k1)];
    let c111 = entries[idx(i1, j1, k1)];
    let c00 = lerp3(c000, c100, tx);
    let c10 = lerp3(c010, c110, tx);
    let c01 = lerp3(c001, c101, tx);
    let c11 = lerp3(c011, c111, tx);
    let c0 = lerp3(c00, c10, ty);
    let c1 = lerp3(c01, c11, ty);
    lerp3(c0, c1, tz)
}

/// CIE 1976 (u', v') from (x, y). Same math as `calibrate`'s private
/// helper — duplicated here so the modules stay independent.
fn xy_to_uv_prime(xy: (f64, f64)) -> (f64, f64) {
    let (x, y) = xy;
    let denom = -2.0 * x + 12.0 * y + 3.0;
    if denom.abs() < 1e-9 {
        return (0.0, 0.0);
    }
    (4.0 * x / denom, 9.0 * y / denom)
}

/// Log-spaced sample targets in `[lo, hi]`. Used for the per-channel
/// commanded sweep — log spacing puts more samples at the dim end
/// where the response is least linear (toe + filter rolloff).
fn log_spaced_targets(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let n = n.max(2);
    let lo = lo.max(1e-3);
    let hi = hi.max(lo * 1.01);
    let lo_ln = lo.ln();
    let hi_ln = hi.ln();
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            (lo_ln + t * (hi_ln - lo_ln)).exp()
        })
        .collect()
}

/// Build the inverse 3D LUT from the per-channel forward responses.
/// Iteration order is X-fastest then Y then Z — matches the binary
/// file format + the GPU image memory walk.
///
/// Returns `(entries, per-grid residual L2 norms)`. The residuals
/// array lets the caller report convergence stats (mean / percentile /
/// worst) — important because the inverter is the only post-measurement
/// math step where bad data can sneak through silently.
fn build_inverse_lut(
    cube_edge: u32,
    responses: &[ChannelResponse; 3],
) -> (Vec<[f32; 3]>, Vec<f64>) {
    let n = cube_edge as usize;
    let denom = (cube_edge - 1) as f32;
    let mut entries = Vec::with_capacity(n * n * n);
    let mut residuals = Vec::with_capacity(n * n * n);
    let mut total_residual = 0.0_f64;
    let mut worst_residual = 0.0_f64;
    for k in 0..n {
        let bz_in = pq_eotf(k as f32 / denom) as f64;
        for j in 0..n {
            let g_in = pq_eotf(j as f32 / denom) as f64;
            for i in 0..n {
                let r_in = pq_eotf(i as f32 / denom) as f64;
                // BT.2020 RGB → BT.2020 XYZ (rows are X, Y, Z; cols R, G, B).
                let target_xyz = bt2020_to_xyz(r_in, g_in, bz_in);
                let (cmd, residual) = invert_one(responses, target_xyz);
                total_residual += residual;
                if residual > worst_residual {
                    worst_residual = residual;
                }
                residuals.push(residual);
                entries.push([cmd[0] as f32, cmd[1] as f32, cmd[2] as f32]);
            }
        }
    }
    let mean = total_residual / (n * n * n) as f64;
    eprintln!(
        "  inversion residuals: mean={:.4} cd/m², worst={:.4} cd/m²",
        mean, worst_residual,
    );
    (entries, residuals)
}

/// Damped-Newton-with-backtracking-line-search inversion of one grid
/// point. Returns the commanded triple plus the residual norm.
///
/// Why damped + line-search: forward responses are non-linear (toe at
/// the dim end, soft knee near saturation). Full Newton steps from a
/// poor initial guess overshoot wildly — a target Y of 12 needs cmd ≈
/// 30 but the slope-extrapolated step from cmd=12 would land near 65.
/// We accept the proposed step only if it actually reduces the
/// residual; if not, halve the step length until it does.
///
/// Initial guess: scale target_XYZ components into per-channel cmd
/// space using the response's brightest-unsaturated Y/cmd ratio.
/// Lands within ~30% of the answer for in-gamut targets so Newton
/// converges in a few iterations.
fn invert_one(responses: &[ChannelResponse; 3], target: [f64; 3]) -> ([f64; 3], f64) {
    // Per-channel approximate gain (Y per unit cmd) from the brightest
    // pre-cliff sample. Used as the per-channel scale for the initial
    // guess and as the fallback when the Jacobian goes singular.
    let approx_gain = [
        responses[0].approx_gain_y_per_cmd(),
        responses[1].approx_gain_y_per_cmd(),
        responses[2].approx_gain_y_per_cmd(),
    ];
    // Distribute target Y across primaries by BT.2020 D65 weights; this
    // is what a perfectly-additive panel would need at minimum. For
    // off-D65 targets it's still a reasonable seed because the dominant
    // contribution per channel is its own primary.
    const D65_WEIGHTS: [f64; 3] = [0.2627, 0.6780, 0.0593];
    let target_y = target[1];
    let mut cmd = [
        ((target_y * D65_WEIGHTS[0]) / approx_gain[0].max(1e-6))
            .clamp(0.0, responses[0].max_cmd),
        ((target_y * D65_WEIGHTS[1]) / approx_gain[1].max(1e-6))
            .clamp(0.0, responses[1].max_cmd),
        ((target_y * D65_WEIGHTS[2]) / approx_gain[2].max(1e-6))
            .clamp(0.0, responses[2].max_cmd),
    ];

    const MAX_ITERS: usize = 40;
    const TOL: f64 = 0.005;
    const MIN_STEP_FRACTION: f64 = 1.0 / 64.0;

    let mut res_norm = predicted_residual_norm(responses, &cmd, &target);
    if res_norm < TOL {
        return (cmd, res_norm);
    }

    for _ in 0..MAX_ITERS {
        // Jacobian at the current cmd: column c = dXYZ/dcmd_c.
        let j0 = responses[0].dxyz_dcmd_at(cmd[0]);
        let j1 = responses[1].dxyz_dcmd_at(cmd[1]);
        let j2 = responses[2].dxyz_dcmd_at(cmd[2]);
        let jac = [
            [j0[0], j1[0], j2[0]],
            [j0[1], j1[1], j2[1]],
            [j0[2], j1[2], j2[2]],
        ];
        // Current residual for the line-search comparison.
        let xyz_r = responses[0].xyz_at(cmd[0]);
        let xyz_g = responses[1].xyz_at(cmd[1]);
        let xyz_b = responses[2].xyz_at(cmd[2]);
        let predicted = [
            xyz_r.x + xyz_g.x + xyz_b.x,
            xyz_r.y + xyz_g.y + xyz_b.y,
            xyz_r.z + xyz_g.z + xyz_b.z,
        ];
        let residual = [
            predicted[0] - target[0],
            predicted[1] - target[1],
            predicted[2] - target[2],
        ];
        let Some(jac_inv) = mat3_inverse(&jac) else {
            // Singular Jacobian — channel sitting on its cliff. Stop
            // and report current state; caller-side residual stat
            // will surface it.
            return (cmd, res_norm);
        };
        let full_step = mat3_mul_vec(&jac_inv, &residual);

        // Backtracking line search: try alpha = 1, 1/2, 1/4, … until
        // the proposed step actually reduces the residual norm. If we
        // can't find one above MIN_STEP_FRACTION the inverter has
        // stalled — bail with the current best.
        let mut alpha = 1.0_f64;
        let mut accepted = false;
        while alpha >= MIN_STEP_FRACTION {
            let mut cmd_try = [0.0_f64; 3];
            for c in 0..3 {
                cmd_try[c] = (cmd[c] - alpha * full_step[c])
                    .clamp(0.0, responses[c].max_cmd);
            }
            let trial_norm = predicted_residual_norm(responses, &cmd_try, &target);
            if trial_norm < res_norm {
                cmd = cmd_try;
                res_norm = trial_norm;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            return (cmd, res_norm);
        }
        if res_norm < TOL {
            return (cmd, res_norm);
        }
    }
    (cmd, res_norm)
}

/// Sum of per-channel emissions minus target — L2 norm of the residual.
/// Helper for the inverter's line-search comparison; pulled out so the
/// loop reads cleanly.
fn predicted_residual_norm(
    responses: &[ChannelResponse; 3],
    cmd: &[f64; 3],
    target: &[f64; 3],
) -> f64 {
    let xyz_r = responses[0].xyz_at(cmd[0]);
    let xyz_g = responses[1].xyz_at(cmd[1]);
    let xyz_b = responses[2].xyz_at(cmd[2]);
    let dx = xyz_r.x + xyz_g.x + xyz_b.x - target[0];
    let dy = xyz_r.y + xyz_g.y + xyz_b.y - target[1];
    let dz = xyz_r.z + xyz_g.z + xyz_b.z - target[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// BT.2020 RGB (linear nits) → CIE XYZ via the canonical BT.2020-2
/// matrix. Input `(R, G, B) = (L, L, L)` produces D65 at total Y = L.
fn bt2020_to_xyz(r: f64, g: f64, b: f64) -> [f64; 3] {
    // ITU-R BT.2020-2 RGB → XYZ (D65 normalized).
    const M: [[f64; 3]; 3] = [
        [0.6370, 0.1446, 0.1689],
        [0.2627, 0.6780, 0.0593],
        [0.0000, 0.0281, 1.0610],
    ];
    [
        M[0][0] * r + M[0][1] * g + M[0][2] * b,
        M[1][0] * r + M[1][1] * g + M[1][2] * b,
        M[2][0] * r + M[2][1] * g + M[2][2] * b,
    ]
}

fn mat3_inverse(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
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

fn mat3_mul_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn open_log(
    args: &CalibrateLut3dArgs,
    baseline: &OutputBaseline,
) -> Result<Option<(PathBuf, BufWriter<File>)>> {
    if args.no_log {
        return Ok(None);
    }
    let path = args.log.clone().unwrap_or_else(|| {
        let safe = args.output.replace('/', "_");
        PathBuf::from(format!("prism-calibrate-lut3d-{safe}.csv"))
    });
    let file = File::create(&path)
        .with_context(|| format!("create log file {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "# prism-tune calibrate-lut3d — output={} mode={} cube_edge={} samples_per_channel={} settle_ms={} window={}",
        args.output,
        if baseline.hdr_active { "HDR" } else { "SDR" },
        args.cube_edge,
        args.samples_per_channel,
        args.settle_ms,
        args.window,
    )?;
    writeln!(w, "channel,sample_idx,commanded_nits,X,Y,Z")?;
    eprintln!("Logging per-sample CSV to {}", path.display());
    Ok(Some((path, w)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic ChannelResponse whose XYZ is `primary × cmd`
    /// — a perfectly-linear panel with the given primary coefficients.
    /// Used in the inversion round-trip tests.
    fn synth_response(primary_xyz: [f64; 3], max_cmd: f64) -> ChannelResponse {
        let cmds = [0.0, 1.0, 5.0, 25.0, 100.0, 500.0, max_cmd];
        let samples: Vec<ChannelSample> = cmds
            .iter()
            .map(|&cmd| ChannelSample {
                commanded: cmd,
                xyz: Xyz {
                    x: primary_xyz[0] * cmd,
                    y: primary_xyz[1] * cmd,
                    z: primary_xyz[2] * cmd,
                },
            })
            .collect();
        let peak_y = samples.last().unwrap().xyz.y;
        ChannelResponse { samples, max_cmd, peak_y }
    }

    /// Inversion against a "panel that exactly matches BT.2020" must
    /// recover the input commanded values. The per-channel responses
    /// are linear × the BT.2020 primary contributions, so BT.2020
    /// input (L, L, L) → target XYZ → invert → (L, L, L). Sanity
    /// anchor — if this drifts the Jacobian or BT.2020 matrix
    /// regressed.
    #[test]
    fn inversion_recovers_identity_for_bt2020_matching_panel() {
        // Columns of the BT.2020 RGB→XYZ matrix = per-primary XYZ at
        // unit RGB (i.e. at cmd=1). Match those exactly in the
        // synthetic responses.
        let r_pri = [0.6370, 0.2627, 0.0000];
        let g_pri = [0.1446, 0.6780, 0.0281];
        let b_pri = [0.1689, 0.0593, 1.0610];
        let responses = [
            synth_response(r_pri, 1000.0),
            synth_response(g_pri, 1000.0),
            synth_response(b_pri, 1000.0),
        ];

        for &l in &[1.0, 10.0, 50.0, 200.0] {
            let target = bt2020_to_xyz(l, l, l);
            let (cmd, residual) = invert_one(&responses, target);
            assert!(
                residual < 0.05,
                "L={l}: residual {residual} too large; cmd={cmd:?} target={target:?}"
            );
            for c in 0..3 {
                assert!(
                    (cmd[c] - l).abs() < 0.05,
                    "L={l}, channel {c}: cmd={} expected ~{l}",
                    cmd[c]
                );
            }
        }
    }

    /// Pure-channel input (red only) must produce cmd_R > 0 and
    /// cmd_G, cmd_B = 0 (within tolerance). Catches the case where
    /// the Newton solver hands off-axis a commanded value when the
    /// optimal solution is on-axis.
    #[test]
    fn pure_channel_input_stays_on_axis() {
        let r_pri = [0.6370, 0.2627, 0.0000];
        let g_pri = [0.1446, 0.6780, 0.0281];
        let b_pri = [0.1689, 0.0593, 1.0610];
        let responses = [
            synth_response(r_pri, 1000.0),
            synth_response(g_pri, 1000.0),
            synth_response(b_pri, 1000.0),
        ];
        let target = bt2020_to_xyz(50.0, 0.0, 0.0);
        let (cmd, residual) = invert_one(&responses, target);
        assert!(residual < 0.05);
        assert!((cmd[0] - 50.0).abs() < 0.05, "cmd_R={}", cmd[0]);
        // Solver should pick (50, 0, 0); allow a tiny numerical fuzz
        // but reject answers that meaningfully smear into G/B.
        assert!(cmd[1].abs() < 0.5, "cmd_G should be ~0, got {}", cmd[1]);
        assert!(cmd[2].abs() < 0.5, "cmd_B should be ~0, got {}", cmd[2]);
    }

    /// Forward LUT interpolation is linear-in-segment: querying at a
    /// point halfway between two samples returns the segment midpoint.
    /// Guards the inversion math from accidentally regressing to
    /// nearest-neighbor (which would crater precision in dim regions).
    #[test]
    fn channel_response_linear_interp() {
        let samples = vec![
            ChannelSample { commanded: 10.0, xyz: Xyz { x: 1.0, y: 2.0, z: 3.0 } },
            ChannelSample { commanded: 20.0, xyz: Xyz { x: 5.0, y: 8.0, z: 11.0 } },
        ];
        let resp = ChannelResponse { samples, max_cmd: 20.0, peak_y: 8.0 };
        let mid = resp.xyz_at(15.0);
        assert!((mid.x - 3.0).abs() < 1e-9, "x = {}", mid.x);
        assert!((mid.y - 5.0).abs() < 1e-9, "y = {}", mid.y);
        assert!((mid.z - 7.0).abs() < 1e-9, "z = {}", mid.z);
    }

    /// Clamp behaviour at the bounds: below the smallest sample
    /// returns the smallest sample; above `max_cmd` returns the
    /// sample at `max_cmd`. Inverter calls this often (Newton step
    /// can overshoot) so it must be defensive.
    #[test]
    fn channel_response_clamps_out_of_range() {
        let samples = vec![
            ChannelSample { commanded: 10.0, xyz: Xyz { x: 1.0, y: 2.0, z: 3.0 } },
            ChannelSample { commanded: 20.0, xyz: Xyz { x: 5.0, y: 8.0, z: 11.0 } },
        ];
        let resp = ChannelResponse { samples, max_cmd: 20.0, peak_y: 8.0 };
        let below = resp.xyz_at(5.0);
        assert!((below.x - 1.0).abs() < 1e-9);
        let above = resp.xyz_at(50.0);
        assert!((above.x - 5.0).abs() < 1e-9);
    }
}

fn print_kdl_block(output_name: &str, hdr_active: bool, peaks: [f32; 3], lut_path: &std::path::Path) {
    println!();
    println!("# Paste into the matching output block in your prism config:");
    println!("output \"{}\" {{", output_name);
    println!("    color {{");
    if hdr_active {
        println!(
            "        panel-peak-nits r={:.1} g={:.1} b={:.1}",
            peaks[0], peaks[1], peaks[2]
        );
    }
    println!("        lut3d \"{}\"", lut_path.display());
    println!("    }}");
    println!("}}");
}
