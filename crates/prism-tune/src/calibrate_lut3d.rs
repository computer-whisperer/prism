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
    query_output_baseline, send_action, set_channel_patch, set_patch_off, set_white_patch,
    show_alignment_patch,
};
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
    let entries = build_inverse_lut(args.cube_edge, &responses);
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
            &args, &baseline, &peak_nits, &mut device, &mut patch, &setup, &cal, log.as_mut(),
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

        eprintln!(
            "  W target {:>7.1} cd/m² → Y={:>7.2}  xy=({:.4},{:.4})  Δu'v'={:.4}  Y_err={:+.1}%",
            t, xyz.y, cx, cy, duv, y_err_pct,
        );
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
        }
    }

    Ok(VerifyResult { max_duv, max_y_err_pct })
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
fn build_inverse_lut(cube_edge: u32, responses: &[ChannelResponse; 3]) -> Vec<[f32; 3]> {
    let n = cube_edge as usize;
    let denom = (cube_edge - 1) as f32;
    let mut entries = Vec::with_capacity(n * n * n);
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
                entries.push([cmd[0] as f32, cmd[1] as f32, cmd[2] as f32]);
            }
        }
    }
    let mean = total_residual / (n * n * n) as f64;
    eprintln!(
        "  inversion residuals: mean={:.4} cd/m², worst={:.4} cd/m²",
        mean, worst_residual,
    );
    entries
}

/// Newton-Raphson invert one grid point. Returns the commanded triple
/// plus the residual norm so the caller can summarize convergence.
fn invert_one(responses: &[ChannelResponse; 3], target: [f64; 3]) -> ([f64; 3], f64) {
    // Initial guess: identity in nits. For an in-gamut sRGB-class panel
    // this is usually within an iteration or two of the answer.
    let mut cmd = [
        target[0].clamp(0.0, responses[0].max_cmd),
        target[1].clamp(0.0, responses[1].max_cmd),
        target[2].clamp(0.0, responses[2].max_cmd),
    ];
    const MAX_ITERS: usize = 25;
    const TOL: f64 = 0.005;
    for _ in 0..MAX_ITERS {
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
        let res_norm = (residual[0].powi(2) + residual[1].powi(2) + residual[2].powi(2)).sqrt();
        if res_norm < TOL {
            return (cmd, res_norm);
        }
        // Jacobian: column j = dXYZ/dcmd_j.
        let j0 = responses[0].dxyz_dcmd_at(cmd[0]);
        let j1 = responses[1].dxyz_dcmd_at(cmd[1]);
        let j2 = responses[2].dxyz_dcmd_at(cmd[2]);
        let jac = [
            [j0[0], j1[0], j2[0]],
            [j0[1], j1[1], j2[1]],
            [j0[2], j1[2], j2[2]],
        ];
        let Some(jac_inv) = mat3_inverse(&jac) else {
            // Singular Jacobian — typically a channel sitting on the
            // cliff with zero slope. Stop and report the current state.
            return (cmd, res_norm);
        };
        // delta = -J^(-1) · residual
        let delta = mat3_mul_vec(&jac_inv, &residual);
        for c in 0..3 {
            cmd[c] = (cmd[c] - delta[c]).clamp(0.0, responses[c].max_cmd);
        }
    }
    // Final residual after max iters.
    let xyz_r = responses[0].xyz_at(cmd[0]);
    let xyz_g = responses[1].xyz_at(cmd[1]);
    let xyz_b = responses[2].xyz_at(cmd[2]);
    let predicted = [
        xyz_r.x + xyz_g.x + xyz_b.x,
        xyz_r.y + xyz_g.y + xyz_b.y,
        xyz_r.z + xyz_g.z + xyz_b.z,
    ];
    let res_norm = ((predicted[0] - target[0]).powi(2)
        + (predicted[1] - target[1]).powi(2)
        + (predicted[2] - target[2]).powi(2))
    .sqrt();
    (cmd, res_norm)
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
