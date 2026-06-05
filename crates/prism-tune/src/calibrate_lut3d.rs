//! `prism-tune calibrate-lut3d` — measurement-driven 3D LUT calibration.
//!
//! Forward model: request patches on a 3D grid, diagnose the actual
//! scanout nits prism emits for each request, then measure the panel's
//! XYZ response at those diagnosed coordinates. The inverse 3D LUT is
//! built by Newton-Raphson against this measured grid: for each grid
//! point in BT.2020 target space, find the scanout triple that produces
//! the target XYZ.
//!
//! Why the 3D forward grid (vs. an additive per-channel model):
//! Real LCDs don't actually obey `panel(R+G+B) = panel(R) +
//! panel(G) + panel(B)`. Driver IC current limits, voltage rail
//! sag, and per-channel-pair optical crosstalk produce ~10-15%
//! sub-additivity when channels are driven together. The additive
//! model bottoms out at that residual no matter how good the per-
//! channel measurements are. A direct 3D grid captures cross-channel
//! interactions structurally; trilinear interpolation between grid
//! points fills in the rest.
//!
//! Pipeline:
//!   0. Black floor: single measurement at (R=G=B=0), subtracted
//!      from every subsequent sample. The colorimeter sees panel
//!      emission + ambient + dark current at black; folding that
//!      into per-channel data triple-counts it in the additive sum.
//!   1. Per-channel saturation discovery: short sweep per channel
//!      to find peak emission. Used to bound the 3D request-axis ranges
//!      (no wasted samples above saturation) and to seed Newton from
//!      diagnosed scanout nits.
//!   2. 3D grid sweep: `cube_edge_cmd³` patches at log-spaced
//!      per-axis requests. Each vertex stores diagnosed scanout nits
//!      plus black-subtracted XYZ in `ResponseGrid`.
//!   3. Inversion: Newton-Raphson against `grid.forward(cmd)` →
//!      17³ scanout-space inverse LUT.
//!   4. Verify: D65 white sweep through the live-pushed LUT,
//!      compare measured Y / Δu'v' to target.
//!
//! Output: a binary `.lut` file written to disk (carries the measured
//! `black_point_xyz` in its v2 header so the compositor can plumb it
//! into tone mapping + wp_color_management feedback) plus a paste-
//! ready KDL snippet pointing the output's config at it.

use anyhow::{Context, Result};
use clap::Args;
use prism_ipc::OutputAction;
use prism_renderer::{pq_eotf, save_lut3d_file, LUT_FILE_IN_TF_PQ};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::thread;
use std::time::Duration;
use tristim_driver::{measurement::raw_to_xyz, AdaptiveTier, Colorimeter, Xyz};

use crate::common::{
    apply_border, apply_panel_peaks, open_patch_surface, query_output_baseline,
    sanitize_for_filename, send_action, send_action_for_reply, set_channel_patch, set_patch_off,
    set_rgb_patch, set_white_patch, show_alignment_patch, Channel, OutputBaseline,
};
use prism_ipc::Response;
use tristim_display::PatchSurface;
use tristim_driver::{Calibration, Setup};

#[derive(Args)]
pub struct CalibrateLut3dArgs {
    /// Connector to calibrate (e.g. `DisplayPort-4`, `HDMI-A-1`).
    #[arg(long)]
    pub output: String,
    /// Inverse-LUT cube edge (grid points per axis). Default 33
    /// matches the compositor's `LUT_CUBE_EDGE` const — the
    /// compositor rejects files whose `cube_edge` doesn't match its
    /// compiled texture size. If you change one, change both.
    #[arg(long, default_value_t = 33)]
    pub cube_edge: u32,
    /// Measurement-grid edge for the 3D forward sweep (Phase 2).
    /// `cube_edge_cmd³` patches command the panel; each measurement
    /// captures one (R, G, B, X, Y, Z) sample. Default 9 = 729
    /// patches. 7 = 343 (faster but coarser cross-channel resolution);
    /// 11 = 1331 (finer but linearly slower).
    #[arg(long, default_value_t = 9)]
    pub cube_edge_cmd: usize,
    /// Per-channel saturation-discovery sample count (Phase 1). Used
    /// only to find each axis's peak emission for the 3D-sweep bounds
    /// and to seed Newton inversion. 9 log-spaced samples is plenty
    /// for cliff detection.
    #[arg(long, default_value_t = 9)]
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
    /// Repeats per gamut-probe vertex burst. Confidence + adaptive
    /// integration cover most of the noise slack, so 4 is the sweet
    /// spot — matches tristim's updated default. Bump for noisier
    /// setups.
    #[arg(long, default_value_t = 4)]
    pub gamut_repeats: usize,
    /// Enable adaptive per-measurement integration across every probe
    /// phase (black floor, per-channel saturation, gamut probe, 3D grid
    /// sweep, and verify). Each measurement first probes at this
    /// integration time and only re-measures at the calibration default
    /// if the fast result fails the trust gate. Bright easy points
    /// stay short; dim or saturated ones escalate. Typical Spyder
    /// default integration is ~1000 ms, so a fast tier of 100–250 ms
    /// is a meaningful win — Phase 2 (729 patches) gets the biggest
    /// payoff. Omit to keep the legacy single-tier behaviour.
    #[arg(long, alias = "gamut-fast-integration-ms")]
    pub fast_integration_ms: Option<u16>,
}

/// One (requested, diagnosed-scanout, XYZ) measurement from a per-channel sweep.
#[derive(Clone, Copy, Debug)]
struct ChannelSample {
    /// Requested value handed to the patch surface (cd/m² for HDR PQ,
    /// nits-equivalent for SDR sRGB encode at patch time).
    requested: f64,
    /// Actual scanout nits reported by `EncodeDiagnose`. In HDR this is
    /// the coordinate used by the forward model; in SDR it equals
    /// the requested value because `EncodeDiagnose` bypasses source-surface
    /// decode.
    scanout: f64,
    xyz: Xyz,
}

/// Forward 1D response LUT for one primary — sorted by diagnosed scanout value
/// and capped at the highest scanout value where the panel was still
/// responding (saturation cutoff).
#[derive(Clone, Debug)]
struct ChannelResponse {
    samples: Vec<ChannelSample>,
    /// Inclusive upper bound on diagnosed scanout past which we treat the
    /// panel as saturated. Used to clamp the Newton-Raphson search
    /// space so we don't extrapolate into the cliff.
    max_cmd: f64,
    /// Requested patch value that produced `max_cmd`. Phase 2 uses this
    /// as the request-space sweep bound, then re-diagnoses the actual
    /// scanout coordinate for every vertex.
    max_requested: f64,
    /// Peak emitted Y observed during the sweep. Stored alongside so
    /// the final panel-peak push uses measured-emitted values rather
    /// than commanded (the latter would over-promise on weak subpixels).
    peak_y: f64,
}

impl ChannelResponse {
    /// Coarse Y-per-cmd gain — the steepest per-sample secant
    /// `Y / scanout` across the sweep. Used by the inverter to seed
    /// its initial guess — a single number per channel that's roughly
    /// the slope of the linear region.
    ///
    /// Steepest (not last) because the sweep may have run deep into
    /// saturation before the cliff detector fired: the last sample's
    /// secant then under-reports the tracking-region slope severalfold,
    /// which over-scales every Newton seed into the flat zone where the
    /// Jacobian is near-singular (the root cause of the catastrophic
    /// garbage cells in the 2026-06 PG27UCDM bake). The steepest secant
    /// is the tracking-region slope regardless of how much saturated
    /// tail the sweep collected. Skips the very first sample (often
    /// dominated by the colorimeter noise floor) to avoid toe noise
    /// inflating the gain.
    fn approx_gain_y_per_cmd(&self) -> f64 {
        self.samples
            .iter()
            .skip(1)
            .map(|s| s.xyz.y / s.scanout.max(1e-6))
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1e-6)
    }
}

fn request_axis_for_scanout(response: &ChannelResponse, cube_edge: usize) -> Vec<f64> {
    let scanout_lo = response.samples.first().unwrap().scanout.max(1e-3);
    let scanout_axis = log_spaced_targets(scanout_lo, response.max_cmd, cube_edge);
    let mut requested_axis: Vec<f64> = scanout_axis
        .into_iter()
        .map(|target_scanout| request_for_scanout(response, target_scanout))
        .collect();
    if let Some(last) = requested_axis.last_mut() {
        *last = response.max_requested;
    }
    requested_axis
}

fn request_for_scanout(response: &ChannelResponse, target_scanout: f64) -> f64 {
    let samples = &response.samples;
    let first = samples.first().unwrap();
    if target_scanout <= first.scanout {
        return first.requested;
    }

    for pair in samples.windows(2) {
        let a = pair[0];
        let b = pair[1];
        if target_scanout <= b.scanout {
            let span = b.scanout - a.scanout;
            if span <= 1e-6 {
                return a.requested;
            }
            let t = ((target_scanout - a.scanout) / span).clamp(0.0, 1.0);
            return a.requested + t * (b.requested - a.requested);
        }
    }

    response.max_requested
}

/// 3D forward measurement grid in diagnosed scanout-space. Each entry stores the
/// panel's absolute emission at a specific (R, G, B) scanout triple. Per-axis values are
/// log-spaced from `args.min_cmd` up to each channel's discovered
/// saturation peak — so all `cube_edge³` measurements land in the
/// useful range and none are wasted above saturation.
///
/// Trilinear interpolation between grid points handles arbitrary cmd
/// queries during Newton inversion. The Jacobian within a cell is
/// constant — `(face_high - face_low) / axis_span` bilinearly
/// interpolated in the other two axes' weights — which gives Newton
/// an analytic derivative without finite-difference noise.
///
/// Storage order: `xyz[(k * N + j) * N + i]` for axis-c indices
/// `(i, j, k)`. X-fastest matches the existing LUT layout convention.
struct ResponseGrid {
    cube_edge: usize,
    /// Per-axis sorted ascending list of diagnosed scanout values.
    /// `axis_cmds[c][i]` is the actual scanout coordinate for channel `c`
    /// at the slice with axis-c index `i`.
    /// Each list has exactly `cube_edge` entries; entries are positive
    /// (no zero corner — handled at the inversion edge case instead).
    axis_cmds: [Vec<f64>; 3],
    /// Absolute (raw) XYZ at each grid point — the reformed bake works
    /// in absolute emission throughout; no black subtraction here.
    xyz: Vec<Xyz>,
}

impl ResponseGrid {
    fn lookup(&self, i: usize, j: usize, k: usize) -> Xyz {
        let n = self.cube_edge;
        let idx = (k * n + j) * n + i;
        self.xyz[idx]
    }

    /// Trilinear forward evaluation. `cmd` clamps per-axis to
    /// `[axis_cmds[c][0], axis_cmds[c][N-1]]` — Newton may overshoot
    /// past the bounds in its line search and we don't want to
    /// extrapolate into a region we haven't measured.
    fn forward(&self, cmd: [f64; 3]) -> Xyz {
        let (lo_r, hi_r, tr) = bracket_axis(&self.axis_cmds[0], cmd[0]);
        let (lo_g, hi_g, tg) = bracket_axis(&self.axis_cmds[1], cmd[1]);
        let (lo_b, hi_b, tb) = bracket_axis(&self.axis_cmds[2], cmd[2]);

        let c000 = self.lookup(lo_r, lo_g, lo_b);
        let c100 = self.lookup(hi_r, lo_g, lo_b);
        let c010 = self.lookup(lo_r, hi_g, lo_b);
        let c110 = self.lookup(hi_r, hi_g, lo_b);
        let c001 = self.lookup(lo_r, lo_g, hi_b);
        let c101 = self.lookup(hi_r, lo_g, hi_b);
        let c011 = self.lookup(lo_r, hi_g, hi_b);
        let c111 = self.lookup(hi_r, hi_g, hi_b);

        // Trilinear: lerp along R, then G, then B.
        let c00 = lerp_xyz(c000, c100, tr);
        let c10 = lerp_xyz(c010, c110, tr);
        let c01 = lerp_xyz(c001, c101, tr);
        let c11 = lerp_xyz(c011, c111, tr);
        let c0 = lerp_xyz(c00, c10, tg);
        let c1 = lerp_xyz(c01, c11, tg);
        lerp_xyz(c0, c1, tb)
    }

    /// Jacobian at `cmd`: column c = ∂XYZ/∂cmd_c. Within a single
    /// grid cell the trilinear interpolation's partial in axis c is
    /// `(face_high - face_low) / axis_span`, where the faces are
    /// bilinearly interpolated in the other two axes' weights.
    /// Returns a 3×3 matrix in [row=XYZ-component, col=cmd-channel]
    /// form so `mat3_inverse + mat3_mul_vec` apply cleanly.
    fn jacobian(&self, cmd: [f64; 3]) -> [[f64; 3]; 3] {
        let (lo_r, hi_r, tr) = bracket_axis(&self.axis_cmds[0], cmd[0]);
        let (lo_g, hi_g, tg) = bracket_axis(&self.axis_cmds[1], cmd[1]);
        let (lo_b, hi_b, tb) = bracket_axis(&self.axis_cmds[2], cmd[2]);
        let span_r = (self.axis_cmds[0][hi_r] - self.axis_cmds[0][lo_r]).max(1e-12);
        let span_g = (self.axis_cmds[1][hi_g] - self.axis_cmds[1][lo_g]).max(1e-12);
        let span_b = (self.axis_cmds[2][hi_b] - self.axis_cmds[2][lo_b]).max(1e-12);

        let c000 = self.lookup(lo_r, lo_g, lo_b);
        let c100 = self.lookup(hi_r, lo_g, lo_b);
        let c010 = self.lookup(lo_r, hi_g, lo_b);
        let c110 = self.lookup(hi_r, hi_g, lo_b);
        let c001 = self.lookup(lo_r, lo_g, hi_b);
        let c101 = self.lookup(hi_r, lo_g, hi_b);
        let c011 = self.lookup(lo_r, hi_g, hi_b);
        let c111 = self.lookup(hi_r, hi_g, hi_b);

        // ∂/∂R: high-R face vs low-R face, bilinearly interpolated in (G, B).
        // Low-R face corners by (j, k): (lo_g, lo_b)=c000, (hi_g, lo_b)=c010,
        // (lo_g, hi_b)=c001, (hi_g, hi_b)=c011.
        let face_lo_r = bilinear_xyz(c000, c010, c001, c011, tg, tb);
        let face_hi_r = bilinear_xyz(c100, c110, c101, c111, tg, tb);
        let d_r = sub_xyz_scaled(face_hi_r, face_lo_r, 1.0 / span_r);

        // ∂/∂G: high-G face vs low-G face, bilinear in (R, B).
        // Low-G face corners by (i, k): c000, c100, c001, c101.
        let face_lo_g = bilinear_xyz(c000, c100, c001, c101, tr, tb);
        let face_hi_g = bilinear_xyz(c010, c110, c011, c111, tr, tb);
        let d_g = sub_xyz_scaled(face_hi_g, face_lo_g, 1.0 / span_g);

        // ∂/∂B: high-B face vs low-B face, bilinear in (R, G).
        // Low-B face corners by (i, j): c000, c100, c010, c110.
        let face_lo_b = bilinear_xyz(c000, c100, c010, c110, tr, tg);
        let face_hi_b = bilinear_xyz(c001, c101, c011, c111, tr, tg);
        let d_b = sub_xyz_scaled(face_hi_b, face_lo_b, 1.0 / span_b);

        [
            [d_r.x, d_g.x, d_b.x],
            [d_r.y, d_g.y, d_b.y],
            [d_r.z, d_g.z, d_b.z],
        ]
    }

    /// Smallest emission anywhere in the grid — at the (min_cmd, min_cmd,
    /// min_cmd) corner. The bake's sub-floor projection uses the gamut
    /// mesh's measured `(0, 0, 0)` vertex (= true black including bleed)
    /// rather than this; kept for debug introspection.
    #[allow(dead_code)]
    fn min_emission(&self) -> Xyz {
        self.lookup(0, 0, 0)
    }

    /// Max cmd per axis (the saturation cap each axis was bounded at).
    fn max_cmd(&self) -> [f64; 3] {
        let n = self.cube_edge - 1;
        [
            self.axis_cmds[0][n],
            self.axis_cmds[1][n],
            self.axis_cmds[2][n],
        ]
    }
}

/// Locate the segment in `axis` (sorted ascending) bracketing `value`.
/// Returns `(lo_idx, hi_idx, t)` where `t ∈ [0, 1]` is the linear
/// weight from `axis[lo_idx]` to `axis[hi_idx]`. Below the range
/// clamps to `(0, 0, 0.0)`; above clamps to `(n-1, n-1, 0.0)` — both
/// degenerate cells where forward returns the corner value and
/// Jacobian returns zero (axis span guarded against div-by-zero).
fn bracket_axis(axis: &[f64], value: f64) -> (usize, usize, f64) {
    let n = axis.len();
    if n == 0 {
        return (0, 0, 0.0);
    }
    if value <= axis[0] {
        return (0, 0, 0.0);
    }
    if value >= axis[n - 1] {
        return (n - 1, n - 1, 0.0);
    }
    // Linear scan — `n` is small (typically 9, max ~17).
    for i in 1..n {
        if value <= axis[i] {
            let span = axis[i] - axis[i - 1];
            let t = if span > 0.0 {
                (value - axis[i - 1]) / span
            } else {
                0.0
            };
            return (i - 1, i, t);
        }
    }
    (n - 1, n - 1, 0.0)
}

fn lerp_xyz(a: Xyz, b: Xyz, t: f64) -> Xyz {
    Xyz {
        x: a.x + t * (b.x - a.x),
        y: a.y + t * (b.y - a.y),
        z: a.z + t * (b.z - a.z),
    }
}

fn bilinear_xyz(c00: Xyz, c10: Xyz, c01: Xyz, c11: Xyz, t0: f64, t1: f64) -> Xyz {
    let a = lerp_xyz(c00, c10, t0);
    let b = lerp_xyz(c01, c11, t0);
    lerp_xyz(a, b, t1)
}

fn sub_xyz_scaled(a: Xyz, b: Xyz, scale: f64) -> Xyz {
    Xyz {
        x: (a.x - b.x) * scale,
        y: (a.y - b.y) * scale,
        z: (a.z - b.z) * scale,
    }
}

/// Single-shot measurement with optional adaptive integration. Returns
/// `(xyz, tier)` so callers can roll per-phase tier totals into their
/// logging without bothering with `MeasurementConfidence` (the gamut
/// probe is the only site that consumes confidence). `fast_ms = None`
/// degrades to `SingleFull` (a plain default-integration measurement).
fn measure_single_adaptive(
    device: &mut Colorimeter,
    setup: &Setup,
    cal: &Calibration,
    fast_ms: Option<u16>,
) -> Result<(Xyz, AdaptiveTier)> {
    let m = device
        .measure_adaptive(setup, cal, 1, fast_ms)
        .context("measure_adaptive")?;
    let xyz = raw_to_xyz(&m.raws[0], &m.setup, &m.cal);
    Ok((xyz, m.tier))
}

/// Compact tier suffix for per-measurement log lines. Empty string for
/// `SingleFull` so non-adaptive runs keep the legacy log shape.
fn tier_suffix(tier: AdaptiveTier) -> &'static str {
    match tier {
        AdaptiveTier::Fast => " [fast]",
        AdaptiveTier::EscalatedFull => " [esc]",
        AdaptiveTier::SingleFull => "",
    }
}

/// Per-phase counters for the three adaptive outcomes. Printed at
/// phase end so big phases (Phase 2 especially) surface their
/// integration cost without spamming per-measurement.
#[derive(Default, Clone, Copy)]
struct TierTally {
    fast: usize,
    escalated: usize,
    single: usize,
}

impl TierTally {
    fn record(&mut self, tier: AdaptiveTier) {
        match tier {
            AdaptiveTier::Fast => self.fast += 1,
            AdaptiveTier::EscalatedFull => self.escalated += 1,
            AdaptiveTier::SingleFull => self.single += 1,
        }
    }

    /// `Some("...")` when adaptive was on (any non-single tier seen);
    /// `None` when every measurement was single-tier (legacy run).
    fn summary(&self) -> Option<String> {
        if self.fast == 0 && self.escalated == 0 {
            return None;
        }
        Some(format!("{} fast, {} escalated", self.fast, self.escalated,))
    }
}

/// Saturation noise-floor guard: the Spyder reads ~0.3 cd/m² of ambient
/// even on black, so the first couple of B samples on a weak-blue panel
/// can show Y in the [0.3, 0.5] range where consecutive-sample
/// comparisons are pure noise. Requiring both samples above 1.0 cd/m²
/// keeps the cliff detector off toe-region wobble — the panel's actual
/// saturation lives well above 1 nit per channel.
const SATURATION_NOISE_FLOOR_Y: f64 = 1.0;

/// Marginal-response efficiency below which a channel counts as
/// saturated: the secant slope ΔY/Δscanout between consecutive sweep
/// samples, relative to the steepest per-sample secant seen so far
/// (the tracking-region gain). The old detector compared consecutive Y
/// as a plain ratio (`< 1.05`), which a half-decade sweep step
/// straddling the knee sails past — the 2026-06 PG27UCDM run measured
/// Y ratio 1.099 across the 316→997 scanout step while the panel ran
/// at 4.6% marginal efficiency, so the entire flat zone entered the
/// forward grid and poisoned the inversion. 0.30 is deliberately
/// sensitive: a false positive only triggers the knee bisection, which
/// walks back up to wherever tracking actually ends.
const SATURATION_EFFICIENCY_MIN: f64 = 0.30;

/// Knee-refinement bisection steps after saturation triggers. Each is
/// one extra measurement; 3 narrows a half-decade bracket to ~15% in
/// cmd — plenty for an axis bound.
const KNEE_REFINE_STEPS: usize = 3;

/// Drive one per-channel patch and measure it: set the patch, diagnose
/// the actual scanout coordinate, settle, measure, black-subtract, and
/// emit the stderr line + CSV row. Shared between the phase-1 discovery
/// sweep and the knee-refinement bisection so refinement samples land
/// in the log with the same shape.
#[allow(clippy::too_many_arguments)]
fn measure_channel_patch(
    channel: Channel,
    cmd: f64,
    row_idx: usize,
    args: &CalibrateLut3dArgs,
    baseline: &OutputBaseline,
    settle: Duration,
    black_xyz: &Xyz,
    device: &mut Colorimeter,
    patch: &mut PatchSurface,
    setup: &Setup,
    cal: &Calibration,
    tally: &mut TierTally,
    log: Option<&mut (PathBuf, BufWriter<File>)>,
) -> Result<ChannelSample> {
    set_channel_patch(patch, baseline, channel, cmd)?;
    let mut requested_rgb = [0.0_f64; 3];
    requested_rgb[channel.idx()] = cmd;
    let scanout_rgb = diagnose_scanout_cmd(&args.output, baseline, requested_rgb)?;
    let scanout_cmd = scanout_rgb[channel.idx()];
    thread::sleep(settle);
    let (raw_xyz, tier) = measure_single_adaptive(device, setup, cal, args.fast_integration_ms)
        .context("phase 1 measure")?;
    tally.record(tier);
    // True emission above the black floor. Clamp to zero so the
    // toe can't go negative from measurement noise — the inverter
    // assumes non-negative XYZ for its line search.
    let xyz = Xyz {
        x: (raw_xyz.x - black_xyz.x).max(0.0),
        y: (raw_xyz.y - black_xyz.y).max(0.0),
        z: (raw_xyz.z - black_xyz.z).max(0.0),
    };
    eprintln!(
        "  {} cmd {:>8.2} scanout {:>8.2} → X={:>8.3}  Y={:>8.3}  Z={:>8.3}  (raw Y={:.3}, less black {:.3}){}",
        channel.label(),
        cmd,
        scanout_cmd,
        xyz.x,
        xyz.y,
        xyz.z,
        raw_xyz.y,
        black_xyz.y,
        tier_suffix(tier),
    );
    if let Some((_, w)) = log {
        // Log the BLACK-SUBTRACTED values — that's the model the rest
        // of the pipeline operates on. Raw values are still recoverable
        // as (logged + black_xyz from the header line written above).
        writeln!(
            w,
            "{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
            channel.label(),
            row_idx,
            cmd,
            scanout_cmd,
            xyz.x,
            xyz.y,
            xyz.z,
        )?;
    }
    Ok(ChannelSample {
        requested: cmd,
        scanout: scanout_cmd,
        xyz,
    })
}

pub fn run(args: CalibrateLut3dArgs) -> Result<()> {
    let baseline =
        query_output_baseline(&args.output).context("query baseline output state via prism IPC")?;
    eprintln!(
        "Baseline for {}: mode={}, panel_peak={:?}, sdr_ref={}",
        args.output,
        if baseline.hdr_active { "HDR" } else { "SDR" },
        baseline.initial_panel_peak_nits,
        baseline.sdr_reference_nits,
    );
    if let Some(ms) = args.fast_integration_ms {
        eprintln!(
            "Adaptive integration enabled: fast tier {} ms, escalate to calibration default on low trust.",
            ms,
        );
    }

    // Wipe any runtime overrides, then lift the IR clamp (HDR) so the
    // panel sees raw commanded values during the sweep. SDR clamp stays
    // at sdr_reference_nits — that's the user's policy and we shouldn't
    // override it during measurement.
    send_action(&args.output, OutputAction::ResetColor).context("initial ResetColor")?;
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
    let cal = device
        .get_calibration(args.cal)
        .context("download cal matrix")?;
    let setup = device.get_setup(&cal).context("download setup")?;

    let mut patch = open_patch_surface(&args.output, &baseline)?;
    patch
        .set_window_fraction(args.window)
        .context("set window fraction")?;

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

    // ─── Phase 0: black-floor measurement ─────────────────────────────────
    // The colorimeter sees panel_emission + ambient + Spyder dark current
    // at (R=G=B=0). Per-channel measurements below pick this up too —
    // each "cmd=1 nit" sample is really (true_emission + black_floor).
    // If we don't subtract it, the additive prediction
    // `panel(R+G+B) ≈ R + G + B` triple-counts the floor and the
    // inverter under-commands at the dim end.
    //
    // 4× normal settle: OLEDs (esp. QD-OLED) take a moment to fully
    // gate off subpixels after the patch swap, and the floor reading
    // is the noise-sensitive one — better to over-settle here than
    // bake measurement-transient junk into every downstream sample.
    let settle = Duration::from_millis(args.settle_ms);
    let settle_black = settle * 4;
    eprintln!("\n--- phase 0: black-floor measurement ---");
    // Patch was already driven off by the prep-countdown setup above —
    // just give the panel the extra settle window before measuring.
    thread::sleep(settle_black);
    let (black_xyz, black_tier) =
        measure_single_adaptive(&mut device, &setup, &cal, args.fast_integration_ms)
            .context("measure black floor")?;
    eprintln!(
        "  black floor: X={:.4}  Y={:.4}  Z={:.4} cd/m²{}",
        black_xyz.x,
        black_xyz.y,
        black_xyz.z,
        tier_suffix(black_tier),
    );
    if let Some((_, w)) = log.as_mut() {
        // Header line first so any reader scanning for "# black_floor"
        // gets it before the per-channel sample rows. Raw values
        // (pre-subtraction) — only place the raw floor is preserved.
        writeln!(
            w,
            "# black_floor (raw, pre-subtract): X={:.4} Y={:.4} Z={:.4} cd/m²",
            black_xyz.x, black_xyz.y, black_xyz.z,
        )?;
    }

    // ─── Phase 1: per-channel saturation discovery + Newton seed ──────────
    // Cheap pre-probe (~9 samples per channel, <30s) used purely to
    // (a) bound each cmd-axis of the 3D sweep at the channel's measured
    // peak, and (b) seed Newton inversion with a per-channel Y-per-cmd
    // ratio. The forward model is the 3D grid measured in Phase 2 —
    // these samples are NOT used as the per-channel additive model
    // any longer.
    //
    // SDR sweep cap: SDR's encode pipeline clamps `cmd /
    // sdr_reference_nits` to [0, 1] before sRGB-OETF encoding for
    // scanout. Cmds above `sdr_reference_nits` all produce the same
    // panel output (post-clip). Sampling that flat region would put
    // degenerate data into the 3D grid — Newton could converge to
    // phantom cmds the encode pipeline can't actually deliver. Cap
    // the sweep at sdr_reference_nits so the request-space sweep bounds (and
    // therefore the 3D sweep axis bounds) stay in the unclipped
    // regime. HDR has no equivalent clamp; full args.max_cmd applies.
    let effective_max_cmd = if baseline.hdr_active {
        args.max_cmd
    } else {
        args.max_cmd.min(baseline.sdr_reference_nits)
    };
    let targets = log_spaced_targets(args.min_cmd, effective_max_cmd, args.samples_per_channel);
    let mut responses: [Option<ChannelResponse>; 3] = [None, None, None];
    let mut phase1_tally = TierTally::default();

    for channel in Channel::ALL {
        eprintln!("\n--- {} channel sweep ---", channel.label());
        let mut samples: Vec<ChannelSample> = Vec::with_capacity(targets.len());
        let mut max_cmd = targets[0];
        let mut max_requested = targets[0];
        let mut peak_y = 0.0_f64;
        // Steepest per-sample secant Y/scanout seen so far — the
        // tracking-region gain the saturation detector measures
        // marginal response against.
        let mut best_gain = 0.0_f64;
        let mut row_idx = 0usize;
        for &cmd in targets.iter() {
            row_idx += 1;
            let sample = measure_channel_patch(
                channel,
                cmd,
                row_idx,
                &args,
                &baseline,
                settle,
                &black_xyz,
                &mut device,
                &mut patch,
                &setup,
                &cal,
                &mut phase1_tally,
                log.as_mut(),
            )?;
            if let Some(&prev) = samples.last() {
                let requested_ratio = sample.requested / prev.requested.max(0.01);
                let scanout_ratio = sample.scanout / prev.scanout.max(0.01);
                let both_above_floor = sample.xyz.y > SATURATION_NOISE_FLOOR_Y
                    && prev.xyz.y > SATURATION_NOISE_FLOOR_Y;
                // Compositor clamp: requests grow but the encode chain
                // emits the same scanout value. Nothing above prev is
                // reachable — stop, no refinement possible.
                let scanout_plateaued = requested_ratio >= 1.2 && scanout_ratio < 1.01;
                // Panel saturation: scanout grows but emission doesn't
                // follow. Judged on marginal efficiency (secant slope
                // between consecutive samples vs the tracking-region
                // gain), not a plain Y ratio — a half-decade sweep step
                // that straddles the knee still shows Y growth even
                // when the panel spends most of the step flat.
                let marginal =
                    (sample.xyz.y - prev.xyz.y) / (sample.scanout - prev.scanout).max(1e-6);
                let panel_saturated = requested_ratio >= 1.2
                    && both_above_floor
                    && best_gain > 0.0
                    && marginal < SATURATION_EFFICIENCY_MIN * best_gain;
                if samples.len() >= 3 && scanout_plateaued {
                    eprintln!(
                        "  {} scanout plateaued at request {:.1} (compositor clamp at scanout {:.1}); \
                         stopping sweep early",
                        channel.label(),
                        sample.requested,
                        sample.scanout,
                    );
                    peak_y = peak_y.max(sample.xyz.y);
                    max_cmd = prev.scanout;
                    max_requested = prev.requested;
                    break;
                }
                if samples.len() >= 3 && panel_saturated {
                    eprintln!(
                        "  {} saturation detected between scanout {:.1} and {:.1} \
                         (marginal efficiency {:.0}% of tracking gain); refining knee",
                        channel.label(),
                        prev.scanout,
                        sample.scanout,
                        100.0 * marginal / best_gain,
                    );
                    // The saturated sample still observes the channel's
                    // true peak emission — fold it in even though its
                    // cmd is past the usable range.
                    peak_y = peak_y.max(sample.xyz.y);
                    // The knee lies inside (prev, sample]. Bisect in
                    // log-request space, keeping the highest cmd whose
                    // marginal response against the tracking-side
                    // bracket still clears the efficiency bar. Each
                    // still-tracking midpoint is a valid sweep sample —
                    // push it so the request→scanout mapping gains
                    // resolution right where the axis bound lands.
                    let mut lo = prev;
                    let mut hi_requested = sample.requested;
                    for _ in 0..KNEE_REFINE_STEPS {
                        let mid_requested = (0.5 * (lo.requested.ln() + hi_requested.ln())).exp();
                        row_idx += 1;
                        let mid = measure_channel_patch(
                            channel,
                            mid_requested,
                            row_idx,
                            &args,
                            &baseline,
                            settle,
                            &black_xyz,
                            &mut device,
                            &mut patch,
                            &setup,
                            &cal,
                            &mut phase1_tally,
                            log.as_mut(),
                        )?;
                        peak_y = peak_y.max(mid.xyz.y);
                        let m = (mid.xyz.y - lo.xyz.y) / (mid.scanout - lo.scanout).max(1e-6);
                        if m >= SATURATION_EFFICIENCY_MIN * best_gain {
                            samples.push(mid);
                            lo = mid;
                        } else {
                            hi_requested = mid.requested;
                        }
                    }
                    eprintln!(
                        "  {} knee localized: max usable scanout {:.1} (request {:.1})",
                        channel.label(),
                        lo.scanout,
                        lo.requested,
                    );
                    max_cmd = lo.scanout;
                    max_requested = lo.requested;
                    break;
                }
            }
            peak_y = peak_y.max(sample.xyz.y);
            if sample.xyz.y > SATURATION_NOISE_FLOOR_Y {
                best_gain = best_gain.max(sample.xyz.y / sample.scanout.max(1e-6));
            }
            max_cmd = sample.scanout;
            max_requested = sample.requested;
            samples.push(sample);
        }
        if samples.len() < 4 {
            anyhow::bail!(
                "{} channel: only {} usable sample(s) before saturation — too few to invert",
                channel.label(),
                samples.len(),
            );
        }
        eprintln!(
            "  {} forward LUT: {} samples, max_request={:.1}, max_scanout={:.1}, peak_y={:.2}",
            channel.label(),
            samples.len(),
            max_requested,
            max_cmd,
            peak_y,
        );
        if let Some((_, w)) = log.as_mut() {
            writeln!(
                w,
                "# {} forward LUT: samples={} max_requested={:.3} max_scanout={:.3} peak_y={:.3}",
                channel.label(),
                samples.len(),
                max_requested,
                max_cmd,
                peak_y,
            )?;
        }
        responses[channel.idx()] = Some(ChannelResponse {
            samples,
            max_cmd,
            max_requested,
            peak_y,
        });
    }
    if let Some(s) = phase1_tally.summary() {
        eprintln!("  phase 1 adaptive: {s}");
    }
    let responses = [
        responses[0].take().unwrap(),
        responses[1].take().unwrap(),
        responses[2].take().unwrap(),
    ];
    let phase2_request_axes = [
        request_axis_for_scanout(&responses[0], args.cube_edge_cmd),
        request_axis_for_scanout(&responses[1], args.cube_edge_cmd),
        request_axis_for_scanout(&responses[2], args.cube_edge_cmd),
    ];
    let seed_gain = [
        responses[0].approx_gain_y_per_cmd(),
        responses[1].approx_gain_y_per_cmd(),
        responses[2].approx_gain_y_per_cmd(),
    ];

    // ─── Phase 1.5: cube-surface gamut probe ──────────────────────────────
    // Adaptive 14-point coarse + quadtree-refined boundary survey of
    // the reachable solid. Each vertex is a burst of repeats reduced
    // to a `MeasurementConfidence`; the refinement uses Lab-ΔE76 in
    // the measured-white frame to detect flat / folded / max-depth
    // / low-trust leaves. Low-trust corners stop subdivision so we
    // never chase a noise read into refinement.
    //
    // The mesh's `(0, 0, 0)` vertex is the panel's true bleed floor
    // — measured absolute, with the same burst confidence — and
    // supersedes the standalone phase-0 black measurement for the
    // bake's bottom-side projection.
    eprintln!("\n--- phase 1.5: cube-surface gamut probe ---");
    let probe_config = crate::gamut::ProbeConfig {
        cmd_axis_max_nits: [
            responses[0].max_requested,
            responses[1].max_requested,
            responses[2].max_requested,
        ],
        repeats: args.gamut_repeats,
        settle,
        settle_black,
        fast_integration_ms: args.fast_integration_ms,
    };
    let probe_params = crate::gamut::RefineParams::default();
    let gamut_mesh = crate::gamut::probe_gamut_refined(
        &probe_config,
        &probe_params,
        &baseline,
        &mut device,
        &mut patch,
        &setup,
        &cal,
        |evt| {
            let crate::gamut::GamutProbeEvent::Measured {
                index,
                code_value,
                cmd_nits,
                measured,
                flags,
                tier,
            } = evt;
            let flag_str = if flags.is_empty() {
                "ok".to_string()
            } else {
                flags
                    .iter()
                    .map(|f| match f {
                        tristim_driver::TrustFlag::Floor => "FLOOR",
                        tristim_driver::TrustFlag::Noisy => "NOISY",
                        tristim_driver::TrustFlag::Chroma => "DUV",
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            };
            // Tier annotation only when adaptive is on, so non-adaptive
            // runs keep the legacy log shape (no surprise [single]).
            let tier_str = match tier {
                tristim_driver::AdaptiveTier::Fast => " [fast]",
                tristim_driver::AdaptiveTier::EscalatedFull => " [esc]",
                tristim_driver::AdaptiveTier::SingleFull => "",
            };
            eprintln!(
                "  vertex {index:>3} cv=({:.2},{:.2},{:.2}) cmd=({:>6.1},{:>6.1},{:>6.1}) → \
                 XYZ=({:>7.3},{:>7.3},{:>7.3}) {flag_str}{tier_str}",
                code_value[0],
                code_value[1],
                code_value[2],
                cmd_nits[0],
                cmd_nits[1],
                cmd_nits[2],
                measured.x,
                measured.y,
                measured.z,
            );
        },
    )
    .context("gamut probe")?;
    eprintln!(
        "  gamut mesh: {} vertices, {} patches ({} flat, {} folded, {} max-depth, {} low-trust)",
        gamut_mesh.vertices.len(),
        gamut_mesh.patches.len(),
        gamut_mesh.count(crate::gamut::PatchStatus::Flat),
        gamut_mesh.count(crate::gamut::PatchStatus::Folded),
        gamut_mesh.count(crate::gamut::PatchStatus::MaxDepth),
        gamut_mesh.count(crate::gamut::PatchStatus::LowTrust),
    );

    // ─── Phase 2: 3D forward grid sweep ───────────────────────────────────
    // The per-channel data above just sized the cmd-axis bounds and
    // seeded Newton — it's deliberately not the forward model. Real
    // LCDs are ~10-15% sub-additive when channels are driven together
    // (driver-IC current limit, voltage-rail sag, per-channel optical
    // crosstalk), and the additive model bottoms out at that residual
    // no matter how dense the per-channel sweep is.
    //
    // Direct 3D sweep: cube_edge_cmd³ patches at log-spaced per-axis
    // cmds, bounded by each channel's measured peak. Trilinear
    // interpolation between grid points captures cross-channel
    // interactions structurally; no additivity assumption anywhere.
    let black_floor_xyz = [black_xyz.x, black_xyz.y, black_xyz.z];
    eprintln!(
        "\n--- phase 2: 3D forward sweep ({}³ = {} patches) ---",
        args.cube_edge_cmd,
        args.cube_edge_cmd.pow(3),
    );
    let grid = sweep_3d_grid(
        args.cube_edge_cmd,
        phase2_request_axes,
        settle,
        &mut device,
        &mut patch,
        &args.output,
        &baseline,
        &setup,
        &cal,
        args.fast_integration_ms,
        log.as_mut(),
    )?;

    // ─── Phase 3: inversion to 3D LUT ─────────────────────────────────────
    eprintln!(
        "\n--- phase 3: invert {}³ forward grid → {}³ inverse LUT ---",
        args.cube_edge_cmd, args.cube_edge
    );
    // The mesh's measured corners anchor the bake's projection: white
    // for the peak-luminance surface, black for the bleed floor (the
    // standalone phase-0 floor is the fallback when the mesh somehow
    // lacks its (0,0,0) vertex).
    let bake_white = [gamut_mesh.white.x, gamut_mesh.white.y, gamut_mesh.white.z];
    let bake_floor = gamut_mesh
        .black()
        .map(|b| [b.x, b.y, b.z])
        .unwrap_or(black_floor_xyz);
    let (entries, residuals) =
        build_inverse_lut(args.cube_edge, &grid, bake_white, bake_floor, seed_gain);
    // Bake health — computed unconditionally and surfaced on stderr.
    // The percentile split used to live only in the CSV, where a
    // catastrophic bake (in-gamut p50 of 72 cd/m²!) hid behind a verify
    // verdict that never sampled the broken range.
    let panel_total_peak: f64 = responses.iter().map(|r| r.peak_y).sum();
    let health = summarize_bake_health(
        args.cube_edge,
        &residuals,
        panel_total_peak,
        bake_white[1],
        grid.min_emission().y,
    );
    health.report(panel_total_peak);
    if let Some((_, w)) = log.as_mut() {
        health.write_csv(w, panel_total_peak)?;
    }
    let peak_nits = [
        responses[0].peak_y as f32,
        responses[1].peak_y as f32,
        responses[2].peak_y as f32,
    ];
    let black_point_f32 = [black_xyz.x as f32, black_xyz.y as f32, black_xyz.z as f32];

    // ─── Phase 4: write LUT + restore ─────────────────────────────────────
    // Default filename: derive from EDID (Make-Model-Serial) so the
    // file follows the physical monitor across port re-plugs.
    // Re-calibrations of the same unit overwrite cleanly. Fall back
    // to the connector name when EDID is missing any of make/model/
    // serial — without all three the identifier can't promise to
    // pick out a single unit.
    let lut_path = args.lut_path.clone().unwrap_or_else(|| {
        let stem = baseline
            .edid_filename_stem()
            .unwrap_or_else(|| sanitize_for_filename(&args.output));
        PathBuf::from(format!("prism-calibrate-lut3d-{stem}.lut"))
    });
    save_lut3d_file(
        &lut_path,
        args.cube_edge,
        peak_nits,
        black_point_f32,
        &entries,
    )
    .with_context(|| format!("write LUT file {}", lut_path.display()))?;
    eprintln!(
        "\nWrote {} (cube_edge={}, peaks={:?}, black_xyz={:?})",
        lut_path.display(),
        args.cube_edge,
        peak_nits,
        black_point_f32,
    );

    // Persist the measured gamut mesh as a sidecar JSON so the cube-
    // surface boundary travels with the LUT for inspection, validation,
    // and any future runtime consumers (tone-mapping, IPC).
    let gamut_path = lut_path.with_extension("gamut.json");
    save_gamut_json(&gamut_path, &gamut_mesh)
        .with_context(|| format!("write gamut sidecar {}", gamut_path.display()))?;
    eprintln!(
        "Wrote gamut sidecar {} ({} vertices, {} patches: {} flat, {} folded, {} max-depth, {} low-trust)",
        gamut_path.display(),
        gamut_mesh.vertices.len(),
        gamut_mesh.patches.len(),
        gamut_mesh.count(crate::gamut::PatchStatus::Flat),
        gamut_mesh.count(crate::gamut::PatchStatus::Folded),
        gamut_mesh.count(crate::gamut::PatchStatus::MaxDepth),
        gamut_mesh.count(crate::gamut::PatchStatus::LowTrust),
    );

    // Apply discovered peaks (HDR mode only) so the IR clamp matches
    // measured reality. SDR keeps its policy-driven peak (sdr_reference_nits).
    if baseline.hdr_active {
        apply_panel_peaks(
            &args.output,
            [
                peak_nits[0] as f64,
                peak_nits[1] as f64,
                peak_nits[2] as f64,
            ],
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

    // ─── Phase 5: verify the live LUT against D65 ─────────────────────────
    let verify_result = if !args.no_verify {
        eprintln!("\n--- phase 5 verify: D65 white sweep through live LUT ---");
        Some(verify_white_point(
            &args,
            &baseline,
            &entries,
            &grid,
            &mut device,
            &mut patch,
            &setup,
            &cal,
            log.as_mut(),
        )?)
    } else {
        eprintln!("\n(verify skipped — --no-verify)");
        None
    };

    set_patch_off(&mut patch, baseline.hdr_active)?;

    if let Some((path, mut w)) = log {
        writeln!(
            w,
            "# inverse LUT written to {} (cube_edge={}, in_tf={}, peaks=R={:.3} G={:.3} B={:.3}, black_xyz=({:.4},{:.4},{:.4}))",
            lut_path.display(),
            args.cube_edge,
            LUT_FILE_IN_TF_PQ,
            peak_nits[0],
            peak_nits[1],
            peak_nits[2],
            black_point_f32[0], black_point_f32[1], black_point_f32[2],
        )?;
        w.flush().ok();
        eprintln!("Measurement log: {}", path.display());
    }

    if let Some(verify) = verify_result.as_ref() {
        // The bake-health neutral check vetoes the verify verdict: a
        // sweep of N patches can land between broken cells, but the
        // residuals see every cell.
        let verdict = if !health.ok() {
            "⚠ POOR — neutral-axis inversion failures; the LUT contains broken cells \
             (see bake health above)."
        } else if verify.max_duv < 0.01 && verify.max_y_err_pct < 5.0 {
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

    // The KDL `output` block name must be one of:
    //   - the connector (e.g. "DP-4"), OR
    //   - the EDID-derived `<Make> <Model> <Serial>` triple, which
    //     `OutputName::matches` accepts on equal terms.
    // Prefer the EDID form so the calibration follows the monitor
    // across port changes; fall back to the connector when EDID is
    // incomplete. Also surface the alternative so the user can pick.
    let edid_id = baseline.edid_identifier();
    let kdl_name = edid_id.as_deref().unwrap_or(args.output.as_str());
    print_kdl_block(
        kdl_name,
        &args.output,
        edid_id.as_deref(),
        baseline.hdr_active,
        peak_nits,
        black_point_f32,
        &lut_path,
    );

    if !args.keep {
        eprintln!(
            "\nRestoring KDL defaults (use --keep to leave the live-pushed LUT + peaks active)."
        );
        send_action(&args.output, OutputAction::ResetColor).context("final ResetColor")?;
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

/// Drive the panel through the cube_edge³ Cartesian product of per-
/// axis log-spaced requests and record one (diagnosed scanout, XYZ)
/// measurement at each vertex. Returns the populated [`ResponseGrid`]
/// holding absolute (raw) XYZ per sample.
///
/// Request axes are chosen from the Phase 1 request→diagnosed-scanout
/// curve so the measured grid is log-spaced in actual emitted scanout
/// coordinates, not in raw requests that may plateau under compositor
/// clamps.
///
/// CSV log: each row is
/// `3D,i,j,k,requested_r,requested_g,requested_b,scanout_r,scanout_g,scanout_b,X,Y,Z` —
/// distinct prefix from the Phase 1 `R/G/B` rows so a log reader
/// can tell which phase each sample is from. XYZ is absolute (no
/// black subtraction); the bake operates in absolute throughout.
#[allow(clippy::too_many_arguments)]
fn sweep_3d_grid(
    cube_edge: usize,
    requested_axis_cmds: [Vec<f64>; 3],
    settle: Duration,
    device: &mut Colorimeter,
    patch: &mut PatchSurface,
    output: &str,
    baseline: &OutputBaseline,
    setup: &Setup,
    cal: &Calibration,
    fast_ms: Option<u16>,
    mut log: Option<&mut (PathBuf, BufWriter<File>)>,
) -> Result<ResponseGrid> {
    if cube_edge < 2 {
        anyhow::bail!("cube_edge_cmd must be ≥ 2 (degenerate 1D grid not supported)");
    }
    if requested_axis_cmds
        .iter()
        .any(|axis| axis.len() != cube_edge)
    {
        anyhow::bail!("internal error: phase 2 request axis length does not match cube_edge_cmd");
    }
    eprintln!(
        "  per-axis cmd ranges: R[{:.2}..{:.2}] G[{:.2}..{:.2}] B[{:.2}..{:.2}]",
        requested_axis_cmds[0][0],
        requested_axis_cmds[0][cube_edge - 1],
        requested_axis_cmds[1][0],
        requested_axis_cmds[1][cube_edge - 1],
        requested_axis_cmds[2][0],
        requested_axis_cmds[2][cube_edge - 1],
    );
    if let Some((_, w)) = log.as_mut() {
        writeln!(
            w,
            "# phase 2: 3D forward grid sweep, cube_edge_cmd={cube_edge} ({} patches)",
            cube_edge.pow(3),
        )?;
        writeln!(
            w,
            "phase,i,j,k,requested_r,requested_g,requested_b,scanout_r,scanout_g,scanout_b,X,Y,Z"
        )?;
    }

    let total = cube_edge.pow(3);
    let mut xyz: Vec<Xyz> = Vec::with_capacity(total);
    let mut axis_scanout_sum: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0_f64; cube_edge]);
    let mut axis_scanout_count: [Vec<usize>; 3] = std::array::from_fn(|_| vec![0_usize; cube_edge]);
    let mut axis_scanout_min: [Vec<f64>; 3] =
        std::array::from_fn(|_| vec![f64::INFINITY; cube_edge]);
    let mut axis_scanout_max: [Vec<f64>; 3] =
        std::array::from_fn(|_| vec![f64::NEG_INFINITY; cube_edge]);
    let mut patches_done = 0usize;
    let mut tally = TierTally::default();
    // Progress heartbeat at 10% increments — the 729-patch sweep takes
    // long enough that a silent stretch reads as a hang.
    let progress_step = (total / 10).max(1);
    let start = std::time::Instant::now();
    for k in 0..cube_edge {
        let cmd_b = requested_axis_cmds[2][k];
        for j in 0..cube_edge {
            let cmd_g = requested_axis_cmds[1][j];
            for i in 0..cube_edge {
                let cmd_r = requested_axis_cmds[0][i];
                set_rgb_patch(patch, baseline, [cmd_r, cmd_g, cmd_b])?;
                let scanout_rgb = diagnose_scanout_cmd(output, baseline, [cmd_r, cmd_g, cmd_b])?;
                for (axis, idx) in [(0, i), (1, j), (2, k)] {
                    let v = scanout_rgb[axis];
                    axis_scanout_sum[axis][idx] += v;
                    axis_scanout_count[axis][idx] += 1;
                    axis_scanout_min[axis][idx] = axis_scanout_min[axis][idx].min(v);
                    axis_scanout_max[axis][idx] = axis_scanout_max[axis][idx].max(v);
                }
                thread::sleep(settle);
                let (raw_xyz, tier) = measure_single_adaptive(device, setup, cal, fast_ms)
                    .context("3D-sweep measure")?;
                tally.record(tier);
                // Absolute emission — the reformed bake works in
                // absolute XYZ and projects sub-floor requests onto
                // the gamut mesh's measured bleed surface (cmd=0 ≡
                // true black). No black subtraction here.
                xyz.push(raw_xyz);
                if let Some((_, w)) = log.as_mut() {
                    writeln!(
                        w,
                        "3D,{i},{j},{k},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                        cmd_r,
                        cmd_g,
                        cmd_b,
                        scanout_rgb[0],
                        scanout_rgb[1],
                        scanout_rgb[2],
                        raw_xyz.x,
                        raw_xyz.y,
                        raw_xyz.z,
                    )?;
                }
                patches_done += 1;
                if patches_done % progress_step == 0 || patches_done == total {
                    let elapsed = start.elapsed().as_secs_f64();
                    let rate = patches_done as f64 / elapsed;
                    let eta_secs = (total - patches_done) as f64 / rate.max(1e-6);
                    let tier_str = match tally.summary() {
                        Some(s) => format!(" ({s})"),
                        None => String::new(),
                    };
                    eprintln!(
                        "  3D sweep: {}/{} patches ({:.0}%) — {:.1} patches/s, ETA {:.0}s{}",
                        patches_done,
                        total,
                        (patches_done as f64 / total as f64) * 100.0,
                        rate,
                        eta_secs,
                        tier_str,
                    );
                }
            }
        }
    }
    let tier_str = match tally.summary() {
        Some(s) => format!(" ({s})"),
        None => String::new(),
    };
    eprintln!(
        "  3D sweep complete: {} patches in {:.1}s{}",
        total,
        start.elapsed().as_secs_f64(),
        tier_str,
    );
    let axis_cmds = diagnosed_axes_from_stats(
        axis_scanout_sum,
        axis_scanout_count,
        axis_scanout_min,
        axis_scanout_max,
    )?;
    eprintln!(
        "  diagnosed scanout ranges: R[{:.2}..{:.2}] G[{:.2}..{:.2}] B[{:.2}..{:.2}]",
        axis_cmds[0][0],
        axis_cmds[0][cube_edge - 1],
        axis_cmds[1][0],
        axis_cmds[1][cube_edge - 1],
        axis_cmds[2][0],
        axis_cmds[2][cube_edge - 1],
    );
    Ok(ResponseGrid {
        cube_edge,
        axis_cmds,
        xyz,
    })
}

fn diagnose_scanout_cmd(
    output: &str,
    baseline: &OutputBaseline,
    requested_rgb: [f64; 3],
) -> Result<[f64; 3]> {
    if !baseline.hdr_active {
        // SDR calibration patches start as source-surface sRGB and are decoded
        // before entering BT.2020. EncodeDiagnose starts after that decode, so
        // using it as the coordinate would silently change domains.
        return Ok(requested_rgb);
    }
    let resp = send_action_for_reply(
        output,
        OutputAction::EncodeDiagnose {
            r: requested_rgb[0],
            g: requested_rgb[1],
            b: requested_rgb[2],
        },
    )
    .context("EncodeDiagnose IPC during calibration sweep")?;
    match resp {
        Response::EncodeDiagnose(r) => Ok(r.scanout_nits),
        other => anyhow::bail!("unexpected reply to EncodeDiagnose: {other:?}"),
    }
}

fn diagnosed_axes_from_stats(
    sum: [Vec<f64>; 3],
    count: [Vec<usize>; 3],
    min: [Vec<f64>; 3],
    max: [Vec<f64>; 3],
) -> Result<[Vec<f64>; 3]> {
    let out: [Vec<f64>; 3] = std::array::from_fn(|axis| {
        (0..sum[axis].len())
            .map(|idx| {
                let n = count[axis][idx].max(1) as f64;
                sum[axis][idx] / n
            })
            .collect()
    });

    for axis in 0..3 {
        for idx in 0..out[axis].len() {
            if count[axis][idx] == 0 {
                anyhow::bail!("missing EncodeDiagnose samples for axis {axis} index {idx}");
            }
            let spread = max[axis][idx] - min[axis][idx];
            let tolerance = (out[axis][idx].abs() * 0.002).max(0.05);
            if spread > tolerance {
                anyhow::bail!(
                    "EncodeDiagnose scanout is not separable for axis {axis} index {idx}: \
                     min={:.4}, max={:.4}, spread={:.4}, tolerance={:.4}",
                    min[axis][idx],
                    max[axis][idx],
                    spread,
                    tolerance,
                );
            }
        }
        for idx in 1..out[axis].len() {
            if out[axis][idx] <= out[axis][idx - 1] {
                anyhow::bail!(
                    "EncodeDiagnose scanout axis {axis} is not strictly increasing at index {idx}: \
                     prev={:.4}, current={:.4}",
                    out[axis][idx - 1],
                    out[axis][idx],
                );
            }
        }
    }

    Ok(out)
}

/// Verify-sweep target luminances. HDR spans the calibrated range up
/// to (almost) the measured white peak at the grid's max command. The
/// old bound — `0.8 × min(per-channel peak_y)` — was the *blue*
/// subpixel's solo Y share (~36 nits on a QD-OLED): it left the top
/// ~90% of the white range unverified, and a bake with garbage cells
/// at 230 nits sailed through with an "excellent" verdict. The 0.9
/// factor keeps the top patch off the exact gamut surface (where
/// legitimate projection slack lives) while sweeping the full usable
/// range; 7 log-spaced patches keep the gaps between them under one
/// octave.
fn verify_white_targets(hdr_active: bool, grid_white_y: f64, sdr_reference_nits: f64) -> Vec<f64> {
    if hdr_active {
        let hi = (grid_white_y * 0.9).max(2.0);
        let lo = (hi * 0.02).max(1.0);
        let lo_ln = lo.ln();
        let hi_ln = hi.max(lo * 1.5).ln();
        (0..7)
            .map(|i| {
                let f = i as f64 / 6.0;
                (lo_ln + f * (hi_ln - lo_ln)).exp()
            })
            .collect()
    } else {
        vec![0.10, 0.25, 0.50, 0.75, 0.95]
            .into_iter()
            .map(|f| f * sdr_reference_nits)
            .collect()
    }
}

/// Render BT.2020 D65 white at a range of luminances through the
/// freshly-pushed LUT and measure how close the panel lands on the
/// reference. Δu'v' large means the LUT's chromaticity inversion is
/// off; Y-error large means the LUT's luminance inversion is off.
///
/// Targets come from [`verify_white_targets`]:
/// - HDR: log-space up to 0.9 × the measured white peak.
/// - SDR: fixed fractions of `sdr_reference_nits` (mirrors
///   `calibrate`'s verify phase so reports are comparable).
#[allow(clippy::too_many_arguments)]
fn verify_white_point(
    args: &CalibrateLut3dArgs,
    baseline: &OutputBaseline,
    lut_entries: &[[f32; 3]],
    grid: &ResponseGrid,
    device: &mut tristim_driver::Colorimeter,
    patch: &mut PatchSurface,
    setup: &Setup,
    cal: &Calibration,
    mut log: Option<&mut (PathBuf, BufWriter<File>)>,
) -> Result<VerifyResult> {
    const D65: (f64, f64) = (0.3127, 0.3290);
    let (d65_up, d65_vp) = xy_to_uv_prime(D65);

    let targets = verify_white_targets(
        baseline.hdr_active,
        grid.forward(grid.max_cmd()).y,
        baseline.sdr_reference_nits,
    );

    let settle = Duration::from_millis(args.settle_ms);
    let mut max_duv = 0.0_f64;
    let mut max_y_err_pct = 0.0_f64;

    if let Some((_, w)) = log.as_mut() {
        writeln!(
            w,
            "# phase 5 verify: D65 white sweep — Δu'v' from D65=({:.4},{:.4})",
            D65.0, D65.1,
        )?;
    }

    for (patch_idx, &t) in targets.iter().enumerate() {
        set_white_patch(patch, baseline, t)?;
        thread::sleep(settle);
        let (xyz, _tier) = measure_single_adaptive(device, setup, cal, args.fast_integration_ms)
            .context("verify measure")?;
        let (cx, cy) = xyz.chromaticity().unwrap_or((0.0, 0.0));
        let (up, vp) = xy_to_uv_prime((cx, cy));
        let duv = ((up - d65_up).powi(2) + (vp - d65_vp).powi(2)).sqrt();
        let y_err_pct = (xyz.y - t) / t.max(0.01) * 100.0;

        // Diagnostic: mirror the shader's LUT lookup for this verify
        // patch so we can compare CPU prediction vs GPU actual.
        // CPU `cmd_predicted` vs GPU `scanout_decoded` should be ≈
        // identical (within f16 quantization) post-texel-center fix;
        // any drift means the renderer or upload path regressed.
        let coord = [pq_oetf_f64(t), pq_oetf_f64(t), pq_oetf_f64(t)];
        let cmd_predicted = trilinear_sample_lut(lut_entries, args.cube_edge, coord);
        let gpu_diag = send_action_for_reply(
            &args.output,
            OutputAction::EncodeDiagnose { r: t, g: t, b: t },
        )
        .context("EncodeDiagnose IPC")?;
        let scanout_decoded = match gpu_diag {
            Response::EncodeDiagnose(r) => r.scanout_nits,
            other => anyhow::bail!("unexpected EncodeDiagnose reply: {other:?}"),
        };
        // Predict measured Y by asking the 3D forward grid what it
        // would produce at the GPU's actual scanout cmds. With the
        // additive-model replaced by a directly-measured grid this
        // diagnostic now answers a different question: "does the
        // panel's behavior at verify time match its behavior at
        // calibration time?" A close match means the grid is
        // capturing reality and the LUT inversion is sound; large
        // gap means drift (thermal, time-since-calibration, ABL
        // behaving differently for this specific verify pattern).
        // Grid stores absolute XYZ post-reform, so `grid.forward.y` is
        // the absolute luminance prediction directly — no floor add-back.
        let grid_pred = grid.forward(scanout_decoded);
        let grid_predicted_y = grid_pred.y;

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
            "      grid-predicted Y from GPU cmd={:.2}  measured Y={:.2}  grid-vs-measured drift={:+.1}%",
            grid_predicted_y, xyz.y,
            (xyz.y - grid_predicted_y) / grid_predicted_y.max(0.01) * 100.0,
        );
        max_duv = max_duv.max(duv);
        max_y_err_pct = max_y_err_pct.max(y_err_pct.abs());

        if let Some((_, w)) = log.as_mut() {
            writeln!(
                w,
                "verify,W,{},{:.4},{:.4},{:.4},{:.4}",
                patch_idx + 1,
                t,
                xyz.x,
                xyz.y,
                xyz.z,
            )?;
            writeln!(
                w,
                "# verify W patch {}: target_nits={:.3} measured_y={:.3} delta_uv={:.5} y_err_pct={:+.3}",
                patch_idx + 1, t, xyz.y, duv, y_err_pct,
            )?;
            writeln!(
                w,
                "#   cpu_lut_cmd=({:.3},{:.3},{:.3}) gpu_scanout_cmd=({:.3},{:.3},{:.3})",
                cmd_predicted[0],
                cmd_predicted[1],
                cmd_predicted[2],
                scanout_decoded[0],
                scanout_decoded[1],
                scanout_decoded[2],
            )?;
            writeln!(
                w,
                "#   grid_predicted_y={:.4} grid_vs_measured_err_pct={:+.3}",
                grid_predicted_y,
                (xyz.y - grid_predicted_y) / grid_predicted_y.max(0.01) * 100.0,
            )?;
        }
    }

    Ok(VerifyResult {
        max_duv,
        max_y_err_pct,
    })
}

/// SMPTE ST 2084 (PQ) OETF: linear nits → encoded `[0, 1]`. CPU mirror
/// of the encode-shader's PQ shaper, used by verify to compute the
/// coord the renderer would sample the LUT at for a given input nits.
pub(crate) fn pq_oetf_f64(nits: f64) -> f64 {
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
pub(crate) fn trilinear_sample_lut(
    entries: &[[f32; 3]],
    cube_edge: u32,
    coord: [f64; 3],
) -> [f32; 3] {
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

/// Build the inverse 3D LUT by projecting each BT.2020 grid input onto
/// the panel's measured reachable volume and inverting against the
/// absolute-XYZ forward grid.
///
/// The reform vs. the older bake:
///
/// - **Absolute XYZ throughout.** The forward grid stores absolute
///   emission (no separate black subtraction). Sub-floor requests
///   project to the gamut mesh's measured `(0, 0, 0)` vertex — the
///   panel's actual bleed XYZ — rather than the old "subtract floor,
///   clamp to 0, hard-map to cmd=0" short-circuit. LCD backlight bleed
///   chromaticity now survives into rendered near-black.
///
/// - **Hue-preserving boundary projection.** Out-of-gamut requests
///   (Newton parks any cmd channel at its measured max) bisect a chroma
///   scale toward the measured white in `u'v'` space at fixed luminance,
///   re-invert at each candidate, and take the highest-chroma scale
///   that lands in-gamut. Compared with the older XYZ-Euclidean Newton
///   clamp (which silently shifted hue at the gamut boundary), this
///   pulls saturated requests toward neutral instead of toward whatever
///   nearest-XYZ corner the panel could reach.
///
/// `black_floor_xyz` is kept as a Newton sub-floor fallback for the rare
/// case the gamut mesh's `(0, 0, 0)` vertex didn't measure (unlikely on
/// hardware but possible in tests with synthetic data).
///
/// Returns `(entries, per-grid residual L2 norms)`. Caller reports
/// convergence stats so a bad inversion can't sneak through silently.
fn build_inverse_lut(
    cube_edge: u32,
    grid: &ResponseGrid,
    white: [f64; 3],
    floor: [f64; 3],
    seed_gain: [f64; 3],
) -> (Vec<[f32; 3]>, Vec<f64>) {
    let n = cube_edge as usize;
    let denom = (cube_edge - 1) as f32;
    let mut entries = Vec::with_capacity(n * n * n);
    let mut residuals = Vec::with_capacity(n * n * n);
    let mut total_residual = 0.0_f64;
    let mut worst_residual = 0.0_f64;
    let white_uv = uv_prime(white);
    // Warm-start chain: each cell offers its solved cmd as a fallback
    // seed to the next cell in scan order. Adjacent cells' targets
    // differ by one PQ grid step, so a converged neighbor is an
    // excellent second seed wherever the analytic one misbehaves.
    let mut prev_cmd: Option<[f64; 3]> = None;
    for k in 0..n {
        let bz_in = pq_eotf(k as f32 / denom) as f64;
        for j in 0..n {
            let g_in = pq_eotf(j as f32 / denom) as f64;
            for i in 0..n {
                let r_in = pq_eotf(i as f32 / denom) as f64;
                let target_xyz = bt2020_to_xyz(r_in, g_in, bz_in);
                let (cmd, residual) = project_and_invert(
                    grid, seed_gain, target_xyz, floor, white[1], white_uv, prev_cmd,
                );
                prev_cmd = Some(cmd);
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

/// Neutral-axis health check: gray-diagonal LUT cells with target Y at
/// or below this fraction of the measured white peak must invert
/// cleanly — they're unambiguously inside the reachable volume. The
/// margin keeps legitimate near-peak projection slack (the panel's
/// native white chromaticity isn't exactly D65) out of the check.
const NEUTRAL_CHECK_WHITE_FRAC: f64 = 0.8;
/// A checked neutral cell fails when its residual exceeds
/// `max(2% of target Y, 1 cd/m²)` — generous against colorimeter noise
/// and trilinear model error, far below visible breakage.
const NEUTRAL_TOL_FRAC: f64 = 0.02;
const NEUTRAL_TOL_MIN_NITS: f64 = 1.0;

/// Percentile summary of one residual population.
struct ResidualStats {
    n: usize,
    p50: f64,
    p90: f64,
    p99: f64,
    max: f64,
}

impl ResidualStats {
    fn from(mut v: Vec<f64>) -> Self {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pick = |p: f64| -> f64 {
            if v.is_empty() {
                f64::NAN
            } else {
                v[((v.len() - 1) as f64 * p) as usize]
            }
        };
        let (p50, p90, p99, max) = (pick(0.50), pick(0.90), pick(0.99), pick(1.0));
        ResidualStats {
            n: v.len(),
            p50,
            p90,
            p99,
            max,
        }
    }

    fn line(&self) -> String {
        format!(
            "n={} p50={:.4} p90={:.4} p99={:.4} max={:.4} cd/m²",
            self.n, self.p50, self.p90, self.p99, self.max
        )
    }
}

/// Bake-health summary from the inversion residuals: the in/out-of-
/// gamut percentile split plus a strict neutral-axis check. Computed
/// unconditionally (not just for the CSV) so a broken bake is loud on
/// stderr and can poison the final verdict.
struct BakeHealth {
    in_gamut: ResidualStats,
    out_of_gamut: ResidualStats,
    /// Gray-diagonal cells inside the strict neutral check (see
    /// [`NEUTRAL_CHECK_WHITE_FRAC`]) whose residual exceeded tolerance.
    /// Any failure here means broken LUT cells — this is exactly the
    /// shape of the 2026-06 PG27UCDM bug where the 230-nit white cell
    /// baked to cmd (0, 0, 157): pure blue.
    neutral_failures: usize,
    neutral_checked: usize,
    neutral_worst: f64,
    neutral_worst_y: f64,
}

impl BakeHealth {
    fn ok(&self) -> bool {
        self.neutral_failures == 0
    }

    fn report(&self, panel_total_peak: f64) {
        eprintln!(
            "  in-gamut (target_Y ≤ {:.1} cd/m²): {}",
            panel_total_peak,
            self.in_gamut.line(),
        );
        eprintln!(
            "  out-of-gamut: {} (expected — panel cap'd)",
            self.out_of_gamut.line(),
        );
        if self.ok() {
            eprintln!(
                "  neutral axis: {}/{} cells inverted within tolerance",
                self.neutral_checked, self.neutral_checked,
            );
        } else {
            eprintln!(
                "  ⚠ neutral axis: {}/{} cells FAILED inversion (worst {:.1} cd/m² at \
                 target {:.1} cd/m²) — the LUT contains broken cells; do not trust this bake",
                self.neutral_failures,
                self.neutral_checked,
                self.neutral_worst,
                self.neutral_worst_y,
            );
        }
    }

    fn write_csv(&self, w: &mut impl Write, panel_total_peak: f64) -> std::io::Result<()> {
        writeln!(
            w,
            "# inversion residuals (in-gamut, target_Y ≤ {:.1} cd/m²): {}",
            panel_total_peak,
            self.in_gamut.line(),
        )?;
        writeln!(
            w,
            "# inversion residuals (out-of-gamut): {} (expected — panel cap'd)",
            self.out_of_gamut.line(),
        )?;
        writeln!(
            w,
            "# neutral-axis health: failures={}/{} worst={:.4} cd/m² at target_y={:.4}",
            self.neutral_failures, self.neutral_checked, self.neutral_worst, self.neutral_worst_y,
        )
    }
}

/// Split the bake residuals into in/out-of-gamut percentile stats
/// (in-gamut = target_Y within the panel's summed per-channel peak —
/// out-of-gamut cells are *expected* to carry projection residuals)
/// and run the strict neutral-axis check against the measured white
/// peak `white_y`.
///
/// `neutral_lo_y` bounds the neutral check from below — pass the
/// grid's min-corner emission Y. Targets dimmer than the dimmest
/// measured sample are unrepresentable by the forward model (its
/// trilinear clamp over-predicts them by construction), so their
/// bounded sub-nit residuals are projection cost, not bake breakage.
fn summarize_bake_health(
    cube_edge: u32,
    residuals: &[f64],
    panel_total_peak: f64,
    white_y: f64,
    neutral_lo_y: f64,
) -> BakeHealth {
    let n = cube_edge as usize;
    let denom = (cube_edge - 1) as f32;
    let mut in_gamut: Vec<f64> = Vec::new();
    let mut out_of_gamut: Vec<f64> = Vec::new();
    let mut neutral_failures = 0usize;
    let mut neutral_checked = 0usize;
    let mut neutral_worst = 0.0_f64;
    let mut neutral_worst_y = 0.0_f64;
    for k in 0..n {
        let bz_in = pq_eotf(k as f32 / denom) as f64;
        for j in 0..n {
            let g_in = pq_eotf(j as f32 / denom) as f64;
            for i in 0..n {
                let r_in = pq_eotf(i as f32 / denom) as f64;
                let target_y = bt2020_to_xyz(r_in, g_in, bz_in)[1];
                let idx = (k * n + j) * n + i;
                let residual = residuals[idx];
                if target_y <= panel_total_peak {
                    in_gamut.push(residual);
                } else {
                    out_of_gamut.push(residual);
                }
                if i == j
                    && j == k
                    && target_y >= neutral_lo_y
                    && target_y <= NEUTRAL_CHECK_WHITE_FRAC * white_y
                {
                    neutral_checked += 1;
                    if residual > (target_y * NEUTRAL_TOL_FRAC).max(NEUTRAL_TOL_MIN_NITS) {
                        neutral_failures += 1;
                        if residual > neutral_worst {
                            neutral_worst = residual;
                            neutral_worst_y = target_y;
                        }
                    }
                }
            }
        }
    }
    BakeHealth {
        in_gamut: ResidualStats::from(in_gamut),
        out_of_gamut: ResidualStats::from(out_of_gamut),
        neutral_failures,
        neutral_checked,
        neutral_worst,
        neutral_worst_y,
    }
}

/// Tolerance for "in-gamut": residual XYZ error in cd/m². 0.5 cd/m² is
/// well below the colorimeter's noise floor, so it admits any honest
/// Newton convergence but rejects the "parked at max_cmd" residuals
/// that indicate the panel can't reach the target.
const PROJECT_TOL_NITS: f64 = 0.5;
/// Saturation margin: a cmd within this fraction of `max_cmd` is treated
/// as out-of-gamut so the bisection pulls chroma toward white rather
/// than baking the terminal cell's hue-shifted clip.
const PROJECT_SATURATION_FRAC: f64 = 0.995;
/// Chroma-compression bisection step count. 16 halvings ⇒ ~1e-5 fraction,
/// far below any meaningful u'v' precision.
const PROJECT_BISECTION_STEPS: usize = 16;

/// Default Newton seed for a target luminance: distribute Y across the
/// channels by BT.2020 D65 weights, scaled by each channel's
/// tracking-region gain. A sane additive guess even though the panel
/// itself is sub-additive — Newton refines from there.
fn default_seed_cmd(seed_gain: [f64; 3], max_cmd: [f64; 3], target_y: f64) -> [f64; 3] {
    const D65_WEIGHTS: [f64; 3] = [0.2627, 0.6780, 0.0593];
    [
        ((target_y * D65_WEIGHTS[0]) / seed_gain[0].max(1e-6)).clamp(0.0, max_cmd[0]),
        ((target_y * D65_WEIGHTS[1]) / seed_gain[1].max(1e-6)).clamp(0.0, max_cmd[1]),
        ((target_y * D65_WEIGHTS[2]) / seed_gain[2].max(1e-6)).clamp(0.0, max_cmd[2]),
    ]
}

/// Project an absolute-XYZ target into the panel's reachable volume and
/// invert. Four cases, applied in order:
///
/// 1. **Below floor** (`target_y ≤ floor_y`): hard-map to cmd=0 — the
///    panel can't go darker than its bleed point. The achieved emission
///    is `floor`; residual is the XYZ distance to it. Preserves bleed
///    chromaticity rather than crushing to neutral zero.
///
/// 2. **Above peak luminance** (`target_y > white_y`): roll the target
///    down to the panel's peak-luminance surface by scaling all three
///    XYZ components by `white_y / target_y` (preserves chromaticity,
///    clamps Y). Without this, the fixed-Y chroma bisection below has
///    no reachable scale at the requested Y and would fall back to a
///    cmd parked at max with a huge residual.
///
/// 3. **In-gamut on first try**: take it.
///
/// 4. **Out-of-gamut by chromaticity**: bisect a chroma scale `s ∈ [0, 1]`
///    where `chromaticity(s) = lerp(white_uv, target_uv, s)` at the
///    (clamped) luminance. Keep the largest `s` whose inversion lands
///    in-gamut. `s = 0` is neutral (always reachable at `Y ≤ white_y`);
///    `s = 1` is the full requested chroma.
///
/// The returned residual is the XYZ distance from `target_abs` to what
/// the panel actually emits at the returned cmd — so it quantifies the
/// full projection magnitude (Y-clamp + chroma compression), not just
/// Newton's local convergence against the projected target. That makes
/// the build-time residual percentiles a uniform "how much did the bake
/// have to project away from the request?" diagnostic.
///
/// `prev_cmd` is a warm-start candidate — typically the solved cmd of
/// the adjacent LUT cell, whose target differs by one grid step. Newton
/// runs from the default analytic seed first; if that fails to converge
/// it retries from `prev_cmd` and keeps the better result. This breaks
/// the deterministic failure mode where the analytic seed (a function
/// of target Y only) lands in an ill-conditioned region and every
/// chroma-bisection step replays the identical divergence.
fn project_and_invert(
    grid: &ResponseGrid,
    seed_gain: [f64; 3],
    target_abs: [f64; 3],
    floor: [f64; 3],
    white_y: f64,
    white_uv: Option<(f64, f64)>,
    prev_cmd: Option<[f64; 3]>,
) -> ([f64; 3], f64) {
    // Below-floor: emit panel black with its measured chromaticity.
    if target_abs[1] <= floor[1] {
        let residual = ((target_abs[0] - floor[0]).powi(2)
            + (target_abs[1] - floor[1]).powi(2)
            + (target_abs[2] - floor[2]).powi(2))
        .sqrt();
        return ([0.0, 0.0, 0.0], residual);
    }

    // Above-peak luminance: scale all of XYZ by `white_y / target_y` to
    // land on the peak-luminance surface at the same chromaticity. The
    // residual against the *original* target captures the roll-off cost.
    let working_target = if target_abs[1] > white_y && target_abs[1] > 0.0 {
        let s = white_y / target_abs[1];
        [target_abs[0] * s, white_y, target_abs[2] * s]
    } else {
        target_abs
    };

    let max_cmd = grid.max_cmd();
    let in_gamut = |cmd: &[f64; 3], residual_to_working: f64| -> bool {
        residual_to_working < PROJECT_TOL_NITS
            && !(0..3).any(|c| cmd[c] >= max_cmd[c] * PROJECT_SATURATION_FRAC)
    };
    // Residual against the *original* target — what the LUT cell will be
    // judged on. Forward-evaluates the grid at the final cmd and L2's
    // against target_abs.
    let residual_to_target = |cmd: &[f64; 3]| -> f64 {
        let emitted = grid.forward(*cmd);
        ((emitted.x - target_abs[0]).powi(2)
            + (emitted.y - target_abs[1]).powi(2)
            + (emitted.z - target_abs[2]).powi(2))
        .sqrt()
    };
    // Newton from the analytic seed; on non-convergence retry from the
    // neighbor-cell warm start and keep whichever lands closer.
    let invert_best = |target: [f64; 3]| -> ([f64; 3], f64) {
        let seed = default_seed_cmd(seed_gain, max_cmd, target[1]);
        let (cmd, residual) = invert_one(grid, seed, target);
        if residual < PROJECT_TOL_NITS {
            return (cmd, residual);
        }
        let Some(warm) = prev_cmd else {
            return (cmd, residual);
        };
        let (cmd_w, residual_w) = invert_one(grid, warm, target);
        if residual_w < residual {
            (cmd_w, residual_w)
        } else {
            (cmd, residual)
        }
    };

    let (cmd0, residual0_to_working) = invert_best(working_target);
    if in_gamut(&cmd0, residual0_to_working) {
        return (cmd0, residual_to_target(&cmd0));
    }

    // Bisect chroma toward neutral at fixed (clamped) luminance.
    let target_uv = uv_prime(working_target);
    let (Some((u_t, v_t)), Some((u_w, v_w))) = (target_uv, white_uv) else {
        return (cmd0, residual_to_target(&cmd0));
    };
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut best_cmd = cmd0;
    for _ in 0..PROJECT_BISECTION_STEPS {
        let scale = 0.5 * (lo + hi);
        let u = u_w + scale * (u_t - u_w);
        let v = v_w + scale * (v_t - v_w);
        let compressed = xyz_from_uv_y(u, v, working_target[1]);
        let (cmd, residual_to_compressed) = invert_best(compressed);
        if in_gamut(&cmd, residual_to_compressed) {
            best_cmd = cmd;
            lo = scale;
        } else {
            hi = scale;
        }
    }
    (best_cmd, residual_to_target(&best_cmd))
}

/// CIE 1976 `u'v'` chromaticity coordinates from absolute XYZ. Returns
/// `None` at true black, where the ratios are undefined.
fn uv_prime(xyz: [f64; 3]) -> Option<(f64, f64)> {
    let denom = xyz[0] + 15.0 * xyz[1] + 3.0 * xyz[2];
    if denom <= 1e-9 {
        None
    } else {
        Some((4.0 * xyz[0] / denom, 9.0 * xyz[1] / denom))
    }
}

/// XYZ at a given `u'v'` chromaticity and luminance Y. Standard inversion
/// of the `u'v'` definition; degenerate at `v' ≈ 0` so we guard.
fn xyz_from_uv_y(u: f64, v: f64, y: f64) -> [f64; 3] {
    let v = v.max(1e-9);
    let x = 9.0 * y * u / (4.0 * v);
    let z = y * (12.0 - 3.0 * u - 20.0 * v) / (4.0 * v);
    [x.max(0.0), y, z.max(0.0)]
}

/// Damped-Newton-with-backtracking-line-search inversion of one
/// target XYZ against the 3D measured forward grid, starting from an
/// explicit `seed_cmd` (see [`default_seed_cmd`] for the analytic
/// choice; callers may also warm-start from a neighboring cell's
/// solution). Returns the commanded triple plus the residual norm.
///
/// Why damped + line-search: the grid's forward is non-linear and
/// the Jacobian is constant only within a cell; stepping across
/// cell boundaries can produce wildly wrong predictions. Accept
/// only steps that reduce the residual norm; halve until they do.
fn invert_one(
    grid: &ResponseGrid,
    seed_cmd: [f64; 3],
    target_emission: [f64; 3],
) -> ([f64; 3], f64) {
    let max_cmd = grid.max_cmd();
    let mut cmd = [
        seed_cmd[0].clamp(0.0, max_cmd[0]),
        seed_cmd[1].clamp(0.0, max_cmd[1]),
        seed_cmd[2].clamp(0.0, max_cmd[2]),
    ];

    const MAX_ITERS: usize = 40;
    const TOL: f64 = 0.005;
    const MIN_STEP_FRACTION: f64 = 1.0 / 64.0;

    let mut res_norm = predicted_residual_norm(grid, &cmd, &target_emission);
    if res_norm < TOL {
        return (cmd, res_norm);
    }
    // Best command seen across the whole search — the singular-Jacobian
    // recovery below can transiently make things worse before Newton
    // re-converges, so never return anything but the best.
    let mut best_cmd = cmd;
    let mut best_res = res_norm;

    for _ in 0..MAX_ITERS {
        let jac = grid.jacobian(cmd);
        let xyz = grid.forward(cmd);
        let residual = [
            xyz.x - target_emission[0],
            xyz.y - target_emission[1],
            xyz.z - target_emission[2],
        ];
        let Some(jac_inv) = mat3_inverse(&jac) else {
            // Singular Jacobian: cmd sits in a saturated (flat) cell of
            // the grid — a whole Jacobian row vanishes wherever an axis
            // segment is past the panel's knee, which happens when the
            // sweep collected saturated samples (the seed can land
            // there directly). Flat regions live at the TOP of the
            // range, so pulling the command halfway toward black
            // re-enters responsive territory within a few steps; Newton
            // then walks the responsive channels back up. The old code
            // returned immediately here, baking the parked seed into
            // the LUT as a garbage cell.
            for c in cmd.iter_mut() {
                *c *= 0.5;
            }
            res_norm = predicted_residual_norm(grid, &cmd, &target_emission);
            if res_norm < best_res {
                best_res = res_norm;
                best_cmd = cmd;
            }
            continue;
        };
        let full_step = mat3_mul_vec(&jac_inv, &residual);

        let mut alpha = 1.0_f64;
        let mut accepted = false;
        while alpha >= MIN_STEP_FRACTION {
            let mut cmd_try = [0.0_f64; 3];
            for c in 0..3 {
                cmd_try[c] = (cmd[c] - alpha * full_step[c]).clamp(0.0, max_cmd[c]);
            }
            let trial_norm = predicted_residual_norm(grid, &cmd_try, &target_emission);
            if trial_norm < res_norm {
                cmd = cmd_try;
                res_norm = trial_norm;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            return if best_res < res_norm {
                (best_cmd, best_res)
            } else {
                (cmd, res_norm)
            };
        }
        if res_norm < best_res {
            best_res = res_norm;
            best_cmd = cmd;
        }
        if res_norm < TOL {
            return (cmd, res_norm);
        }
    }
    if best_res < res_norm {
        (best_cmd, best_res)
    } else {
        (cmd, res_norm)
    }
}

/// L2 norm of (grid.forward(cmd) - target_emission). Helper for
/// inverter's line-search comparison; pulled out so the loop reads
/// cleanly.
fn predicted_residual_norm(grid: &ResponseGrid, cmd: &[f64; 3], target_emission: &[f64; 3]) -> f64 {
    let xyz = grid.forward(*cmd);
    let dx = xyz.x - target_emission[0];
    let dy = xyz.y - target_emission[1];
    let dz = xyz.z - target_emission[2];
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
    // Mirror the LUT filename's EDID-keying so CSV + .lut sit next
    // to each other with matching stems; fall back to connector name
    // when EDID is incomplete.
    let path = args.log.clone().unwrap_or_else(|| {
        let stem = baseline
            .edid_filename_stem()
            .unwrap_or_else(|| sanitize_for_filename(&args.output));
        PathBuf::from(format!("prism-calibrate-lut3d-{stem}.csv"))
    });
    let file =
        File::create(&path).with_context(|| format!("create log file {}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "# prism-tune calibrate-lut3d — output={} mode={} cube_edge={} cube_edge_cmd={} samples_per_channel={} settle_ms={} window={}",
        args.output,
        if baseline.hdr_active { "HDR" } else { "SDR" },
        args.cube_edge,
        args.cube_edge_cmd,
        args.samples_per_channel,
        args.settle_ms,
        args.window,
    )?;
    // Phase 1 rows use this header (channel-axis per-channel sweep);
    // Phase 2 rows declare their own wider 3D header at the top of that
    // block. Verify rows are tagged `verify,W,…`. All XYZ values are
    // black-subtracted (raw floor recoverable from the `# black_floor`
    // comment line below).
    writeln!(w, "channel,sample_idx,requested_nits,scanout_nits,X,Y,Z")?;
    eprintln!("Logging per-sample CSV to {}", path.display());
    Ok(Some((path, w)))
}

/// Persist a `GamutMesh` as a JSON sidecar alongside the LUT file. The
/// schema is hand-rolled with `serde_json::json!` so we don't pull a
/// `Serialize` derive onto `Xyz` (which lives in tristim-driver and
/// might evolve at its own pace). Human-readable and easy to load from
/// any tool that wants the cube-surface boundary later.
fn save_gamut_json(path: &std::path::Path, mesh: &crate::gamut::GamutMesh) -> Result<()> {
    let vertices: Vec<serde_json::Value> = mesh
        .vertices
        .iter()
        .map(|v| {
            serde_json::json!({
                "code_value": v.code_value,
                "cmd_nits": v.cmd_nits,
                "xyz": [v.xyz.x, v.xyz.y, v.xyz.z],
                "lab": v.lab,
                "trustworthy": v.trustworthy,
            })
        })
        .collect();
    let patches: Vec<serde_json::Value> = mesh
        .patches
        .iter()
        .map(|p| {
            serde_json::json!({
                "face": p.face_label(),
                "axis": p.axis,
                "value": p.value,
                "corners": p.corners,
                "status": p.status.as_str(),
            })
        })
        .collect();
    let doc = serde_json::json!({
        "schema": "prism-gamut-mesh.v1",
        "white_xyz": [mesh.white.x, mesh.white.y, mesh.white.z],
        "cmd_axis_max_nits": mesh.cmd_axis_max_nits,
        "vertices": vertices,
        "patches": patches,
    });
    let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    serde_json::to_writer_pretty(BufWriter::new(f), &doc)
        .with_context(|| format!("serialize gamut mesh to {}", path.display()))?;
    Ok(())
}

fn print_kdl_block(
    kdl_name: &str,
    connector: &str,
    edid_id: Option<&str>,
    hdr_active: bool,
    peaks: [f32; 3],
    black_xyz: [f32; 3],
    lut_path: &std::path::Path,
) {
    println!();
    println!(
        "# Measured black floor for {connector}: X={:.4} Y={:.4} Z={:.4} cd/m²",
        black_xyz[0], black_xyz[1], black_xyz[2],
    );
    println!(
        "# (carried in the LUT file header; compositor exposes it via OutputContext for tone mapping)",
    );
    match edid_id {
        Some(_) => println!(
            "# Output block keyed by EDID (Make Model Serial) — the calibration\n\
             # follows the physical monitor if it moves to a different port.\n\
             # If you prefer port-keyed config, replace the block name with \"{connector}\".",
        ),
        None => println!(
            "# EDID make/model/serial incomplete on this output — block keyed by\n\
             # connector \"{connector}\". The calibration won't follow the\n\
             # monitor across port changes; manual update needed if you re-cable.",
        ),
    }
    println!("# Paste into the matching output block in your prism config:");
    println!("output \"{}\" {{", kdl_name);
    println!("    color {{");
    if hdr_active {
        println!(
            "        panel-peak-nits r={:.1} g={:.1} b={:.1}",
            peaks[0], peaks[1], peaks[2]
        );
    }
    println!("        lut3d \"{}\"", lut_path.display());
    // The measured gamut-surface sidecar written alongside the LUT. The
    // compositor doesn't use it in the pipeline; it serves it over IPC so
    // the prism-tune gamut-cloud inspector can overlay the panel's actual
    // reachable boundary as a lattice shell.
    println!(
        "        gamut \"{}\"",
        lut_path.with_extension("gamut.json").display()
    );
    println!("    }}");
    println!("}}");
}

// ── Offline rebake from a measurement CSV ────────────────────────────────────
//
// The forward measurements (phase-1 channel sweeps + phase-2 3D grid)
// are honest panel data even when the bake that followed them produced
// a broken LUT. `rebake-lut3d` re-runs the inversion with the current
// algorithm against a previous run's CSV — pure CPU math, no
// colorimeter, no running prism, no re-rigging a fragile panel/sensor
// setup. White/black anchors come from the `.gamut.json` sidecar the
// original run wrote next to the CSV.

#[derive(Args)]
pub struct RebakeLut3dArgs {
    /// Measurement-log CSV written by a previous `calibrate-lut3d` run.
    pub csv: PathBuf,
    /// Output `.lut` path. Defaults to the CSV path with its extension
    /// replaced by `.lut` — i.e. the same file the original run wrote.
    #[arg(long)]
    pub lut_path: Option<PathBuf>,
    /// Inverse-LUT cube edge. Must match the compositor's compiled
    /// LUT texture size (33).
    #[arg(long, default_value_t = 33)]
    pub cube_edge: u32,
}

/// One parsed `3D,…` grid row.
struct GridRow {
    i: usize,
    j: usize,
    k: usize,
    scanout: [f64; 3],
    xyz: Xyz,
}

/// Everything the rebake needs out of a calibrate-lut3d CSV.
struct ParsedCsvLog {
    /// Raw (pre-subtraction) floor from the `# black_floor` comment.
    black_floor_xyz: [f64; 3],
    /// Phase-1 channel sweeps, black-subtracted as logged.
    channel_samples: [Vec<ChannelSample>; 3],
    /// Phase-2 grid edge from the header comments.
    cube_edge_cmd: usize,
    grid_rows: Vec<GridRow>,
}

/// Pull the unsigned integer following `key` out of a comment line
/// (e.g. `cube_edge_cmd=9`).
fn parse_kv_usize(line: &str, key: &str) -> Option<usize> {
    let start = line.find(key)? + key.len();
    let digits: String = line[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// Pull `X=… Y=… Z=…` floats out of a comment line.
fn parse_xyz_tokens(line: &str) -> Option<[f64; 3]> {
    let grab = |key: &str| -> Option<f64> {
        let start = line.find(key)? + key.len();
        line[start..]
            .split_whitespace()
            .next()
            .and_then(|tok| tok.parse().ok())
    };
    Some([grab("X=")?, grab("Y=")?, grab("Z=")?])
}

/// Parse a calibrate-lut3d measurement CSV. Tolerant by construction:
/// comment lines are scanned only for the black floor + grid edge,
/// data lines are dispatched on their first field, and anything else
/// (column headers, verify rows, future row kinds) is skipped.
fn parse_csv_log(text: &str) -> Result<ParsedCsvLog> {
    let mut black_floor_xyz: Option<[f64; 3]> = None;
    let mut cube_edge_cmd: Option<usize> = None;
    let mut channel_samples: [Vec<ChannelSample>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut grid_rows: Vec<GridRow> = Vec::new();
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(comment) = line.strip_prefix('#') {
            if black_floor_xyz.is_none() && comment.contains("black_floor") {
                black_floor_xyz = parse_xyz_tokens(comment);
            }
            if cube_edge_cmd.is_none() {
                cube_edge_cmd = parse_kv_usize(comment, "cube_edge_cmd=");
            }
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        let num = |idx: usize| -> Result<f64> {
            fields
                .get(idx)
                .ok_or_else(|| anyhow::anyhow!("CSV line {}: missing field {idx}", lineno + 1))?
                .parse::<f64>()
                .map_err(|e| {
                    anyhow::anyhow!("CSV line {}: field {idx} not a number: {e}", lineno + 1)
                })
        };
        match fields[0] {
            "R" | "G" | "B" if fields.len() >= 7 => {
                let ch = match fields[0] {
                    "R" => 0,
                    "G" => 1,
                    _ => 2,
                };
                channel_samples[ch].push(ChannelSample {
                    requested: num(2)?,
                    scanout: num(3)?,
                    xyz: Xyz {
                        x: num(4)?,
                        y: num(5)?,
                        z: num(6)?,
                    },
                });
            }
            "3D" if fields.len() >= 13 => {
                let index = |idx: usize| -> Result<usize> {
                    fields[idx].parse::<usize>().map_err(|e| {
                        anyhow::anyhow!("CSV line {}: field {idx} not an index: {e}", lineno + 1)
                    })
                };
                grid_rows.push(GridRow {
                    i: index(1)?,
                    j: index(2)?,
                    k: index(3)?,
                    scanout: [num(7)?, num(8)?, num(9)?],
                    xyz: Xyz {
                        x: num(10)?,
                        y: num(11)?,
                        z: num(12)?,
                    },
                });
            }
            // Column headers, verify rows, future row kinds.
            _ => {}
        }
    }
    let black_floor_xyz = black_floor_xyz
        .ok_or_else(|| anyhow::anyhow!("CSV missing the `# black_floor` comment line"))?;
    let cube_edge_cmd = cube_edge_cmd
        .ok_or_else(|| anyhow::anyhow!("CSV missing `cube_edge_cmd=` in its header comments"))?;
    anyhow::ensure!(!grid_rows.is_empty(), "CSV contains no `3D,…` grid rows");
    Ok(ParsedCsvLog {
        black_floor_xyz,
        channel_samples,
        cube_edge_cmd,
        grid_rows,
    })
}

/// Rebuild the [`ResponseGrid`] from parsed 3D rows: XYZ placed by
/// (i, j, k), per-axis scanout coordinates re-derived through the same
/// averaging + separability validation the live sweep uses.
fn grid_from_rows(cube_edge_cmd: usize, rows: &[GridRow]) -> Result<ResponseGrid> {
    let n = cube_edge_cmd;
    anyhow::ensure!(n >= 2, "cube_edge_cmd must be ≥ 2");
    let total = n * n * n;
    anyhow::ensure!(
        rows.len() == total,
        "expected {total} 3D grid rows ({n}³), found {}",
        rows.len(),
    );
    let mut xyz: Vec<Option<Xyz>> = vec![None; total];
    let mut sum: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0_f64; n]);
    let mut count: [Vec<usize>; 3] = std::array::from_fn(|_| vec![0_usize; n]);
    let mut min: [Vec<f64>; 3] = std::array::from_fn(|_| vec![f64::INFINITY; n]);
    let mut max: [Vec<f64>; 3] = std::array::from_fn(|_| vec![f64::NEG_INFINITY; n]);
    for r in rows {
        anyhow::ensure!(
            r.i < n && r.j < n && r.k < n,
            "3D row index ({},{},{}) out of range for edge {n}",
            r.i,
            r.j,
            r.k,
        );
        let idx = (r.k * n + r.j) * n + r.i;
        anyhow::ensure!(
            xyz[idx].is_none(),
            "duplicate 3D row for ({},{},{})",
            r.i,
            r.j,
            r.k,
        );
        xyz[idx] = Some(r.xyz);
        for (axis, ai) in [(0, r.i), (1, r.j), (2, r.k)] {
            let v = r.scanout[axis];
            sum[axis][ai] += v;
            count[axis][ai] += 1;
            min[axis][ai] = min[axis][ai].min(v);
            max[axis][ai] = max[axis][ai].max(v);
        }
    }
    let axis_cmds = diagnosed_axes_from_stats(sum, count, min, max)?;
    Ok(ResponseGrid {
        cube_edge: n,
        axis_cmds,
        // Coverage is total: rows.len() == n³ with no duplicates.
        xyz: xyz
            .into_iter()
            .map(|o| o.expect("full grid coverage checked above"))
            .collect(),
    })
}

fn json_vec3(v: &serde_json::Value) -> Option<[f64; 3]> {
    let arr = v.as_array()?;
    if arr.len() != 3 {
        return None;
    }
    Some([arr[0].as_f64()?, arr[1].as_f64()?, arr[2].as_f64()?])
}

/// Load the white + black anchors from a `.gamut.json` sidecar. Only
/// the two corner measurements feed the bake; the rest of the mesh
/// stays on disk for the inspector.
fn load_gamut_anchors(path: &std::path::Path) -> Result<([f64; 3], Option<[f64; 3]>)> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let doc: serde_json::Value = serde_json::from_reader(std::io::BufReader::new(f))
        .with_context(|| format!("parse {}", path.display()))?;
    let white = json_vec3(&doc["white_xyz"])
        .ok_or_else(|| anyhow::anyhow!("{}: missing/invalid white_xyz", path.display()))?;
    let black = doc["vertices"].as_array().and_then(|vs| {
        vs.iter()
            .find(|v| json_vec3(&v["code_value"]) == Some([0.0, 0.0, 0.0]))
            .and_then(|v| json_vec3(&v["xyz"]))
    });
    Ok((white, black))
}

pub fn run_rebake(args: RebakeLut3dArgs) -> Result<()> {
    let text = std::fs::read_to_string(&args.csv)
        .with_context(|| format!("read {}", args.csv.display()))?;
    let parsed = parse_csv_log(&text)?;
    eprintln!(
        "Parsed {}: {}³ grid rows, channel sweeps R={} G={} B={}, black floor Y={:.4}",
        args.csv.display(),
        parsed.cube_edge_cmd,
        parsed.channel_samples[0].len(),
        parsed.channel_samples[1].len(),
        parsed.channel_samples[2].len(),
        parsed.black_floor_xyz[1],
    );

    // Rebuild the per-channel responses just enough for the Newton seed
    // gain + the header peaks. The grid axes come from the 3D rows
    // themselves, so the phase-1 saturation bound plays no role here.
    let mut responses: [Option<ChannelResponse>; 3] = [None, None, None];
    for ch in 0..3 {
        let samples = parsed.channel_samples[ch].clone();
        anyhow::ensure!(
            samples.len() >= 2,
            "{} channel: {} sweep row(s) in CSV — too few to derive a seed gain",
            ["R", "G", "B"][ch],
            samples.len(),
        );
        let peak_y = samples.iter().map(|s| s.xyz.y).fold(0.0_f64, f64::max);
        let last = *samples.last().unwrap();
        responses[ch] = Some(ChannelResponse {
            samples,
            max_cmd: last.scanout,
            max_requested: last.requested,
            peak_y,
        });
    }
    let responses = [
        responses[0].take().unwrap(),
        responses[1].take().unwrap(),
        responses[2].take().unwrap(),
    ];
    let seed_gain = [
        responses[0].approx_gain_y_per_cmd(),
        responses[1].approx_gain_y_per_cmd(),
        responses[2].approx_gain_y_per_cmd(),
    ];
    let grid = grid_from_rows(parsed.cube_edge_cmd, &parsed.grid_rows)?;

    // White/black anchors from the sidecar mesh when present; fall back
    // to the grid's max-cmd corner + the CSV's raw floor.
    let sidecar = args.csv.with_extension("gamut.json");
    let (bake_white, bake_floor) = match load_gamut_anchors(&sidecar) {
        Ok((white, black)) => {
            eprintln!(
                "Gamut anchors from {}: white Y={:.1}, black Y={:.4}",
                sidecar.display(),
                white[1],
                black.unwrap_or(parsed.black_floor_xyz)[1],
            );
            (white, black.unwrap_or(parsed.black_floor_xyz))
        }
        Err(e) => {
            let corner = grid.forward(grid.max_cmd());
            eprintln!(
                "No usable gamut sidecar ({e:#}); using the grid's max-cmd corner as \
                 white (Y={:.1}) and the CSV floor as black",
                corner.y,
            );
            ([corner.x, corner.y, corner.z], parsed.black_floor_xyz)
        }
    };

    eprintln!(
        "\n--- rebake: invert {}³ forward grid → {}³ inverse LUT ---",
        parsed.cube_edge_cmd, args.cube_edge,
    );
    let (entries, residuals) =
        build_inverse_lut(args.cube_edge, &grid, bake_white, bake_floor, seed_gain);
    let panel_total_peak: f64 = responses.iter().map(|r| r.peak_y).sum();
    let health = summarize_bake_health(
        args.cube_edge,
        &residuals,
        panel_total_peak,
        bake_white[1],
        grid.min_emission().y,
    );
    health.report(panel_total_peak);

    let peak_nits = [
        responses[0].peak_y as f32,
        responses[1].peak_y as f32,
        responses[2].peak_y as f32,
    ];
    let black_point_f32 = [
        parsed.black_floor_xyz[0] as f32,
        parsed.black_floor_xyz[1] as f32,
        parsed.black_floor_xyz[2] as f32,
    ];
    let lut_path = args
        .lut_path
        .clone()
        .unwrap_or_else(|| args.csv.with_extension("lut"));
    save_lut3d_file(
        &lut_path,
        args.cube_edge,
        peak_nits,
        black_point_f32,
        &entries,
    )
    .with_context(|| format!("write LUT file {}", lut_path.display()))?;
    eprintln!(
        "\nWrote {} (cube_edge={}, peaks={:?}, black_xyz={:?})",
        lut_path.display(),
        args.cube_edge,
        peak_nits,
        black_point_f32,
    );
    eprintln!(
        "Existing config pointing at this path picks it up on prism restart; \
         no measurement-side files were touched."
    );
    if !health.ok() {
        anyhow::bail!("bake health check failed — see the neutral-axis report above");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a perfectly-linear additive panel as a ResponseGrid:
    /// `XYZ(cmd_R, cmd_G, cmd_B) = r_pri × cmd_R + g_pri × cmd_G +
    /// b_pri × cmd_B`. Inversion against this should recover the
    /// input cmd to high precision (no sub-additivity to confound
    /// the grid interpolation).
    fn synth_grid(
        cube_edge: usize,
        r_pri: [f64; 3],
        g_pri: [f64; 3],
        b_pri: [f64; 3],
        max_cmd: f64,
    ) -> ResponseGrid {
        let axis = log_spaced_targets(1.0, max_cmd, cube_edge);
        let axis_cmds = [axis.clone(), axis.clone(), axis];
        let mut xyz = Vec::with_capacity(cube_edge.pow(3));
        for k in 0..cube_edge {
            let cb = axis_cmds[2][k];
            for j in 0..cube_edge {
                let cg = axis_cmds[1][j];
                for i in 0..cube_edge {
                    let cr = axis_cmds[0][i];
                    xyz.push(Xyz {
                        x: r_pri[0] * cr + g_pri[0] * cg + b_pri[0] * cb,
                        y: r_pri[1] * cr + g_pri[1] * cg + b_pri[1] * cb,
                        z: r_pri[2] * cr + g_pri[2] * cg + b_pri[2] * cb,
                    });
                }
            }
        }
        ResponseGrid {
            cube_edge,
            axis_cmds,
            xyz,
        }
    }

    /// Inversion against a "panel that exactly matches BT.2020" must
    /// recover the input commanded values. Sanity anchor — if this
    /// drifts, the Jacobian, trilinear interp, or BT.2020 matrix
    /// regressed.
    #[test]
    fn inversion_recovers_identity_for_bt2020_matching_panel() {
        // Columns of the BT.2020 RGB→XYZ matrix = per-primary XYZ at
        // unit RGB; the synthetic grid is exactly linear in those.
        let r_pri = [0.6370, 0.2627, 0.0000];
        let g_pri = [0.1446, 0.6780, 0.0281];
        let b_pri = [0.1689, 0.0593, 1.0610];
        let grid = synth_grid(9, r_pri, g_pri, b_pri, 1000.0);
        // Per-channel seed gain from the grid's diagonal slice (cmd_R
        // alone). For a perfectly-linear panel this is just the
        // primary's Y coefficient.
        let seed_gain = [r_pri[1], g_pri[1], b_pri[1]];

        for &l in &[10.0, 50.0, 200.0] {
            let target = bt2020_to_xyz(l, l, l);
            let seed = default_seed_cmd(seed_gain, grid.max_cmd(), target[1]);
            let (cmd, residual) = invert_one(&grid, seed, target);
            assert!(
                residual < 0.5,
                "L={l}: residual {residual} too large; cmd={cmd:?} target={target:?}"
            );
            for c in 0..3 {
                assert!(
                    (cmd[c] - l).abs() < 1.0,
                    "L={l}, channel {c}: cmd={} expected ~{l}",
                    cmd[c]
                );
            }
        }
    }

    /// A target whose luminance exceeds the panel's peak white must be
    /// rolled down onto the peak-luminance surface (Y-clamp at fixed
    /// chromaticity) before chroma bisection runs. Without the Y-clamp
    /// the fixed-Y bisection has no reachable scale and returns the
    /// initial Newton attempt with a huge residual; with it, the
    /// residual is bounded by the projection cost — basically
    /// ‖original − peak_white_at_same_chromaticity‖.
    #[test]
    fn project_clamps_above_peak_luminance_to_white_surface() {
        // BT.2020-primary synthetic panel — well-conditioned for the
        // Newton seed (all three channels contribute to Y, matching the
        // shape of any real panel). Full-cube peak white emits
        // bt2020_to_xyz(100, 100, 100) ≈ (95.0, 100.0, 109.0).
        let r_pri = [0.6370, 0.2627, 0.0000];
        let g_pri = [0.1446, 0.6780, 0.0281];
        let b_pri = [0.1689, 0.0593, 1.0610];
        let grid = synth_grid(9, r_pri, g_pri, b_pri, 100.0);
        let seed_gain = [r_pri[1], g_pri[1], b_pri[1]];
        let white_xyz = grid.forward([100.0, 100.0, 100.0]);
        let white_y = white_xyz.y;
        let white_uv = super::uv_prime([white_xyz.x, white_xyz.y, white_xyz.z]);
        let floor = [0.0, 0.0, 0.0];

        // Ask for 4× peak luminance at the panel's own white chromaticity.
        // Y-clamp pulls us back to the peak-white corner; residual is the
        // L2 gap between original and white_xyz — known and bounded.
        let target = [white_xyz.x * 4.0, white_xyz.y * 4.0, white_xyz.z * 4.0];
        let expected_gap = ((target[0] - white_xyz.x).powi(2)
            + (target[1] - white_xyz.y).powi(2)
            + (target[2] - white_xyz.z).powi(2))
        .sqrt();
        let (cmd, residual) =
            super::project_and_invert(&grid, seed_gain, target, floor, white_y, white_uv, None);
        // cmd should land at (or very near) the corner.
        for c in 0..3 {
            assert!(
                (cmd[c] - 100.0).abs() < 2.0,
                "above-peak target should park cmd[{c}] near corner, got {}",
                cmd[c]
            );
        }
        // Residual should be ≈ the projection-cost gap, NOT the thousands-
        // of-cd/m² value the pre-fix bake produced for this kind of input.
        // Tolerance accounts for Newton's small slack at the corner.
        assert!(
            (residual - expected_gap).abs() < 5.0,
            "residual {residual} should be ≈ {expected_gap} (projection cost)",
        );
    }

    /// Pure-channel target (red only) should produce cmd with most of
    /// its mass on the R axis. Trilinear interp from a sparse grid
    /// introduces small off-axis bleed — looser tolerance than the
    /// additive analytic version this replaced.
    #[test]
    fn pure_channel_input_stays_near_axis() {
        let r_pri = [0.6370, 0.2627, 0.0000];
        let g_pri = [0.1446, 0.6780, 0.0281];
        let b_pri = [0.1689, 0.0593, 1.0610];
        let grid = synth_grid(9, r_pri, g_pri, b_pri, 1000.0);
        let seed_gain = [r_pri[1], g_pri[1], b_pri[1]];
        let target = bt2020_to_xyz(50.0, 0.0, 0.0);
        let seed = default_seed_cmd(seed_gain, grid.max_cmd(), target[1]);
        let (cmd, _residual) = invert_one(&grid, seed, target);
        // cmd_R dominates (close to 50, within trilinear noise).
        assert!((cmd[0] - 50.0).abs() < 5.0, "cmd_R={}", cmd[0]);
        // cmd_G, cmd_B small relative to cmd_R.
        assert!(
            cmd[1] < cmd[0] * 0.2,
            "cmd_G should be small, got {}",
            cmd[1]
        );
        assert!(
            cmd[2] < cmd[0] * 0.2,
            "cmd_B should be small, got {}",
            cmd[2]
        );
    }

    /// ResponseGrid::forward at a grid corner returns the corner
    /// value exactly — no interpolation rounding when query lands on
    /// a sample point. Catches off-by-one indexing in bracket_axis or
    /// trilinear weight computation.
    #[test]
    fn grid_forward_at_corner_is_exact() {
        let grid = synth_grid(5, [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], 100.0);
        let n = grid.cube_edge;
        // Top corner (max cmd on all axes).
        let cmd_top = [
            grid.axis_cmds[0][n - 1],
            grid.axis_cmds[1][n - 1],
            grid.axis_cmds[2][n - 1],
        ];
        let xyz = grid.forward(cmd_top);
        let expected = Xyz {
            x: cmd_top[0],
            y: 2.0 * cmd_top[1],
            z: 3.0 * cmd_top[2],
        };
        assert!(
            (xyz.x - expected.x).abs() < 1e-9,
            "x: got {}, want {}",
            xyz.x,
            expected.x
        );
        assert!(
            (xyz.y - expected.y).abs() < 1e-9,
            "y: got {}, want {}",
            xyz.y,
            expected.y
        );
        assert!(
            (xyz.z - expected.z).abs() < 1e-9,
            "z: got {}, want {}",
            xyz.z,
            expected.z
        );
    }

    /// Out-of-bounds queries clamp to the nearest grid face. Tests
    /// both directions: below the min (cmd=0) returns the min-cmd
    /// corner; above max returns the max-cmd corner. Newton's line
    /// search can overshoot in either direction so this defense
    /// matters.
    #[test]
    fn grid_forward_clamps_out_of_range() {
        let grid = synth_grid(5, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 100.0);
        let n = grid.cube_edge;
        // Below: cmd=0 < axis[0]=1; should return the (1,1,1) corner.
        let below = grid.forward([0.0, 0.0, 0.0]);
        let corner = grid.lookup(0, 0, 0);
        assert!((below.x - corner.x).abs() < 1e-9);
        assert!((below.y - corner.y).abs() < 1e-9);
        // Above: cmd=10000 > axis[max]; should return the (max, max, max) corner.
        let above = grid.forward([10000.0, 10000.0, 10000.0]);
        let max_corner = grid.lookup(n - 1, n - 1, n - 1);
        assert!((above.x - max_corner.x).abs() < 1e-9);
        assert!((above.y - max_corner.y).abs() < 1e-9);
    }

    /// Build a hard-clipping panel: linear response up to `knee` nits
    /// of per-channel cmd, flat above — the shape of an OLED driven
    /// past its per-channel current limit. Axes span well past the
    /// knee, mirroring a sweep whose saturation detection missed.
    fn synth_clipping_grid(
        cube_edge: usize,
        r_pri: [f64; 3],
        g_pri: [f64; 3],
        b_pri: [f64; 3],
        max_cmd: f64,
        knee: f64,
    ) -> ResponseGrid {
        let axis = log_spaced_targets(1.0, max_cmd, cube_edge);
        let axis_cmds = [axis.clone(), axis.clone(), axis];
        let mut xyz = Vec::with_capacity(cube_edge.pow(3));
        for k in 0..cube_edge {
            let cb = axis_cmds[2][k].min(knee);
            for j in 0..cube_edge {
                let cg = axis_cmds[1][j].min(knee);
                for i in 0..cube_edge {
                    let cr = axis_cmds[0][i].min(knee);
                    xyz.push(Xyz {
                        x: r_pri[0] * cr + g_pri[0] * cg + b_pri[0] * cb,
                        y: r_pri[1] * cr + g_pri[1] * cg + b_pri[1] * cb,
                        z: r_pri[2] * cr + g_pri[2] * cg + b_pri[2] * cb,
                    });
                }
            }
        }
        ResponseGrid {
            cube_edge,
            axis_cmds,
            xyz,
        }
    }

    /// Regression for the 2026-06 PG27UCDM bake: a forward grid whose
    /// top segment is deep in panel saturation, combined with a seed
    /// gain computed from the saturated tail (2.5× under the tracking
    /// slope), must NOT produce broken neutral-axis cells. The pre-fix
    /// bake parked Newton in the flat zone and baked a 230-nit white
    /// to cmd (0, 0, 157) — pure blue — while reporting "excellent".
    #[test]
    fn saturated_grid_bake_keeps_neutral_axis_healthy() {
        let r_pri = [0.6370, 0.2627, 0.0000];
        let g_pri = [0.1446, 0.6780, 0.0281];
        let b_pri = [0.1689, 0.0593, 1.0610];
        let knee = 400.0;
        let grid = synth_clipping_grid(9, r_pri, g_pri, b_pri, 1000.0, knee);
        // The OLD broken seed gain: per-channel peak emission over the
        // saturated max scanout (pri × 400 / 1000).
        let bad_seed_gain = [
            r_pri[1] * knee / 1000.0,
            g_pri[1] * knee / 1000.0,
            b_pri[1] * knee / 1000.0,
        ];
        let white_xyz = grid.forward(grid.max_cmd());
        let white = [white_xyz.x, white_xyz.y, white_xyz.z];
        let cube_edge = 17_u32;
        let (entries, residuals) =
            build_inverse_lut(cube_edge, &grid, white, [0.0; 3], bad_seed_gain);
        let health = summarize_bake_health(
            cube_edge,
            &residuals,
            white_xyz.y,
            white_xyz.y,
            grid.min_emission().y,
        );
        assert!(
            health.ok(),
            "neutral-axis failures: {}/{} (worst {:.2} cd/m² at target {:.2} cd/m²)",
            health.neutral_failures,
            health.neutral_checked,
            health.neutral_worst,
            health.neutral_worst_y,
        );
        // And the gray diagonal must command near-neutral triples —
        // no (0, 0, B)-shaped garbage anywhere below the white peak.
        let n = cube_edge as usize;
        let denom = (cube_edge - 1) as f32;
        for d in 0..n {
            let target_y = bt2020_to_xyz(
                pq_eotf(d as f32 / denom) as f64,
                pq_eotf(d as f32 / denom) as f64,
                pq_eotf(d as f32 / denom) as f64,
            )[1];
            if target_y < 1.0 || target_y > 0.8 * white_xyz.y {
                continue;
            }
            let cmd = entries[(d * n + d) * n + d];
            let lo = cmd.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = cmd.iter().copied().fold(0.0_f32, f32::max);
            assert!(
                lo > 0.0 && hi / lo.max(1e-3) < 3.0,
                "diagonal cell {d} (target {target_y:.1} cd/m²) has wildly \
                 unbalanced cmd {cmd:?}",
            );
        }
    }

    /// The seed gain must come from the tracking region even when the
    /// sweep ran deep into saturation before stopping — the old
    /// last-sample pick under-reported by the saturation factor.
    #[test]
    fn approx_gain_uses_tracking_region_not_saturated_tail() {
        let slope = 0.3;
        let knee = 350.0;
        let samples: Vec<ChannelSample> = [1.0, 3.16, 10.0, 31.6, 100.0, 316.0, 1000.0]
            .iter()
            .map(|&s| ChannelSample {
                requested: s,
                scanout: s,
                xyz: Xyz {
                    x: 0.0,
                    y: slope * s.min(knee),
                    z: 0.0,
                },
            })
            .collect();
        let response = ChannelResponse {
            samples,
            max_cmd: 1000.0,
            max_requested: 1000.0,
            peak_y: slope * knee,
        };
        let gain = response.approx_gain_y_per_cmd();
        assert!(
            (gain - slope).abs() < slope * 0.05,
            "gain {gain} should be ≈ tracking slope {slope}, not the \
             saturated secant {}",
            slope * knee / 1000.0,
        );
    }

    /// Verify-sweep targets must span up to (almost) the measured
    /// white peak — including the mid-range zone the old
    /// min-channel-peak bound never sampled.
    #[test]
    fn verify_targets_span_measured_white_peak() {
        let t = verify_white_targets(true, 457.0, 203.0);
        assert_eq!(t.len(), 7);
        let top = *t.last().unwrap();
        assert!(
            (400.0..457.0).contains(&top),
            "top verify target {top} should approach the 457-nit white peak",
        );
        assert!(
            t.iter().any(|&v| (150.0..330.0).contains(&v)),
            "verify sweep must cover the mid-range whites: {t:?}",
        );
        // SDR branch unchanged: fractions of the reference white.
        let sdr = verify_white_targets(false, 457.0, 200.0);
        assert_eq!(sdr, vec![20.0, 50.0, 100.0, 150.0, 190.0]);
    }

    /// CSV log → ParsedCsvLog → ResponseGrid round-trip on a minimal
    /// synthetic log with the exact row shapes the live run writes
    /// (header comments, channel rows, 3D rows, verify rows).
    #[test]
    fn parse_csv_log_rebuilds_grid() {
        let csv = "\
# prism-tune calibrate-lut3d — output=TEST mode=HDR cube_edge=33 cube_edge_cmd=2 samples_per_channel=4 settle_ms=32 window=0.02
channel,sample_idx,requested_nits,scanout_nits,X,Y,Z
# black_floor (raw, pre-subtract): X=0.0500 Y=0.0700 Z=0.1200 cd/m²
R,1,1.0000,1.0000,0.6370,0.2627,0.0000
R,2,10.0000,10.0000,6.3700,2.6270,0.0000
G,1,1.0000,1.0000,0.1446,0.6780,0.0281
G,2,10.0000,10.0000,1.4460,6.7800,0.2810
B,1,1.0000,1.0000,0.1689,0.0593,1.0610
B,2,10.0000,10.0000,1.6890,0.5930,10.6100
# phase 2: 3D forward grid sweep, cube_edge_cmd=2 (8 patches)
phase,i,j,k,requested_r,requested_g,requested_b,scanout_r,scanout_g,scanout_b,X,Y,Z
3D,0,0,0,1,1,1,1.0,1.0,1.0,0.9505,1.0000,1.0891
3D,1,0,0,10,1,1,10.0,1.0,1.0,6.6835,3.3643,1.0891
3D,0,1,0,1,10,1,1.0,10.0,1.0,2.2519,7.1020,1.3420
3D,1,1,0,10,10,1,10.0,10.0,1.0,7.9849,9.4663,1.3420
3D,0,0,1,1,1,10,1.0,1.0,10.0,2.4706,1.5337,10.6381
3D,1,0,1,10,1,10,10.0,1.0,10.0,8.2036,3.8980,10.6381
3D,0,1,1,1,10,10,1.0,10.0,10.0,3.7720,7.6357,10.8910
3D,1,1,1,10,10,10,10.0,10.0,10.0,9.5050,10.0000,10.8910
verify,W,1,1.4394,1.4082,1.4885,1.6542
# inverse LUT written to whatever.lut (cube_edge=33, in_tf=1, peaks=R=1 G=1 B=1, black_xyz=(0,0,0))
";
        let parsed = parse_csv_log(csv).unwrap();
        assert_eq!(parsed.black_floor_xyz, [0.05, 0.07, 0.12]);
        assert_eq!(parsed.cube_edge_cmd, 2);
        for ch in 0..3 {
            assert_eq!(parsed.channel_samples[ch].len(), 2, "channel {ch}");
        }
        assert_eq!(parsed.grid_rows.len(), 8);

        let grid = grid_from_rows(parsed.cube_edge_cmd, &parsed.grid_rows).unwrap();
        for axis in 0..3 {
            assert_eq!(grid.axis_cmds[axis], vec![1.0, 10.0], "axis {axis}");
        }
        // Placement: the (1,0,1) row landed at its (i,j,k) slot.
        let v = grid.lookup(1, 0, 1);
        assert!((v.x - 8.2036).abs() < 1e-9);
        assert!((v.y - 3.8980).abs() < 1e-9);
        assert!((v.z - 10.6381).abs() < 1e-9);
        // Missing/duplicate coverage is rejected.
        assert!(grid_from_rows(3, &parsed.grid_rows).is_err());
    }

    /// Trilinear interp at the midpoint between two grid points
    /// returns the linear average. Specifically check along one axis
    /// (R) with G and B pinned at axis[0]. Guards against regression
    /// to nearest-neighbor or skipping the trilinear weight.
    #[test]
    fn grid_forward_midpoint_linear() {
        let grid = synth_grid(3, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 100.0);
        let r0 = grid.axis_cmds[0][0];
        let r1 = grid.axis_cmds[0][1];
        let g0 = grid.axis_cmds[1][0];
        let b0 = grid.axis_cmds[2][0];
        let mid_r = 0.5 * (r0 + r1);
        let xyz = grid.forward([mid_r, g0, b0]);
        // Synth grid: X = cmd_R + 0 + 0, so midpoint X = mid_r.
        assert!(
            (xyz.x - mid_r).abs() < 1e-9,
            "midpoint X: got {}, want {}",
            xyz.x,
            mid_r,
        );
    }
}
