//! `prism-tune calibrate-lut3d` — measurement-driven 3D LUT calibration.
//!
//! Forward model: measure the panel's response on a 3D grid in
//! command space — every combination of (cmd_R, cmd_G, cmd_B) at
//! `cube_edge_cmd` log-spaced values per axis. The inverse 3D LUT
//! is built by Newton-Raphson against this measured grid: for each
//! grid point in BT.2020 target space, find the cmd triple that
//! produces the target XYZ.
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
//!      to find peak emission. Used to bound the 3D cmd-axis ranges
//!      (no wasted samples above saturation) and to seed Newton.
//!   2. 3D grid sweep: `cube_edge_cmd³` patches at log-spaced
//!      per-axis cmds — black-subtracted, stored as `ResponseGrid`.
//!   3. Inversion: Newton-Raphson against `grid.forward(cmd)` →
//!      17³ cmd-space inverse LUT.
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
    set_rgb_patch, set_white_patch, show_alignment_patch,
};
use prism_ipc::Response;
use tristim_display::PatchSurface;
use tristim_driver::{Calibration, Setup};

#[derive(Args)]
pub struct CalibrateLut3dArgs {
    /// Connector to calibrate (e.g. `DisplayPort-4`, `HDMI-A-1`).
    #[arg(long)]
    pub output: String,
    /// Inverse-LUT cube edge (grid points per axis). Default 17 matches
    /// the compositor's default texture size. 33 gives finer precision at
    /// 8× the storage cost; only useful if 17³ shows banding.
    #[arg(long, default_value_t = 17)]
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
}

/// 3D forward measurement grid in cmd-space. Each entry stores the
/// panel's true emission (black already subtracted) at a specific
/// (cmd_R, cmd_G, cmd_B) command triple. Per-axis cmd values are
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
    /// Per-axis sorted ascending list of commanded values. `axis_cmds[c][i]`
    /// is the cmd handed to channel `c` for the slice at axis-c index `i`.
    /// Each list has exactly `cube_edge` entries; entries are positive
    /// (no zero corner — handled at the inversion edge case instead).
    axis_cmds: [Vec<f64>; 3],
    /// Black-subtracted XYZ at each grid point.
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
    /// min_cmd) corner. Used by the inverter to short-circuit "target is
    /// below what the panel can render above black" cases to cmd=0.
    fn min_emission(&self) -> Xyz {
        self.lookup(0, 0, 0)
    }

    /// Max cmd per axis (the saturation cap each axis was bounded at).
    fn max_cmd(&self) -> [f64; 3] {
        let n = self.cube_edge - 1;
        [self.axis_cmds[0][n], self.axis_cmds[1][n], self.axis_cmds[2][n]]
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
            let t = if span > 0.0 { (value - axis[i - 1]) / span } else { 0.0 };
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
    let raw_black = device.measure_raw(&setup).context("measure black floor")?;
    let black_xyz = raw_to_xyz(&raw_black, &setup, &cal);
    eprintln!(
        "  black floor: X={:.4}  Y={:.4}  Z={:.4} cd/m²",
        black_xyz.x, black_xyz.y, black_xyz.z,
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
            let raw_xyz = raw_to_xyz(&raw, &setup, &cal);
            // True emission above the black floor. Clamp to zero so the
            // toe can't go negative from measurement noise — the inverter
            // assumes non-negative XYZ for its line search.
            let xyz = Xyz {
                x: (raw_xyz.x - black_xyz.x).max(0.0),
                y: (raw_xyz.y - black_xyz.y).max(0.0),
                z: (raw_xyz.z - black_xyz.z).max(0.0),
            };
            eprintln!(
                "  {} cmd {:>8.2} → X={:>8.3}  Y={:>8.3}  Z={:>8.3}  (raw Y={:.3}, less black {:.3})",
                channel.label(),
                cmd,
                xyz.x,
                xyz.y,
                xyz.z,
                raw_xyz.y,
                black_xyz.y,
            );
            if let Some((_, w)) = log.as_mut() {
                // Log the BLACK-SUBTRACTED values — that's the model
                // the rest of the pipeline operates on. Raw values
                // are still recoverable as (logged + black_xyz from
                // the header line written above).
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
    let per_channel_peaks = [
        responses[0].max_cmd,
        responses[1].max_cmd,
        responses[2].max_cmd,
    ];
    let seed_gain = [
        responses[0].approx_gain_y_per_cmd(),
        responses[1].approx_gain_y_per_cmd(),
        responses[2].approx_gain_y_per_cmd(),
    ];

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
        args.min_cmd,
        per_channel_peaks,
        black_xyz,
        settle,
        &mut device,
        &mut patch,
        &baseline,
        &setup,
        &cal,
        log.as_mut(),
    )?;

    // ─── Phase 3: inversion to 3D LUT ─────────────────────────────────────
    eprintln!("\n--- phase 3: invert {}³ forward grid → {}³ inverse LUT ---",
              args.cube_edge_cmd, args.cube_edge);
    let (entries, residuals) =
        build_inverse_lut(args.cube_edge, &grid, seed_gain, black_floor_xyz);
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
    let black_point_f32 = [
        black_xyz.x as f32,
        black_xyz.y as f32,
        black_xyz.z as f32,
    ];

    // ─── Phase 4: write LUT + restore ─────────────────────────────────────
    let lut_path = args.lut_path.clone().unwrap_or_else(|| {
        let safe = args.output.replace('/', "_");
        PathBuf::from(format!("prism-calibrate-lut3d-{safe}.lut"))
    });
    save_lut3d_file(&lut_path, args.cube_edge, peak_nits, black_point_f32, &entries)
        .with_context(|| format!("write LUT file {}", lut_path.display()))?;
    eprintln!(
        "\nWrote {} (cube_edge={}, peaks={:?}, black_xyz={:?})",
        lut_path.display(), args.cube_edge, peak_nits, black_point_f32,
    );

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

    // ─── Phase 5: verify the live LUT against D65 ─────────────────────────
    let verify_result = if !args.no_verify {
        eprintln!("\n--- phase 5 verify: D65 white sweep through live LUT ---");
        Some(verify_white_point(
            &args, &baseline, &peak_nits, &entries, &grid, black_floor_xyz,
            &mut device, &mut patch, &setup, &cal, log.as_mut(),
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

    print_kdl_block(&args.output, baseline.hdr_active, peak_nits, black_point_f32, &lut_path);

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

/// Drive the panel through the cube_edge³ Cartesian product of per-
/// axis log-spaced cmds and record one (cmd, XYZ) measurement at each
/// vertex. Returns the populated [`ResponseGrid`], with the black
/// floor already subtracted from every sample.
///
/// Axis bounds: per-channel `[args_min_cmd, per_channel_peaks[c]]`.
/// Putting peaks-from-Phase-1 in as the upper bound means no
/// measurements are wasted above each channel's saturation, regardless
/// of how much they differ (typical LCD: R≈300, G≈420, B≈420).
///
/// CSV log: each row is `3D,i,j,k,cmd_r,cmd_g,cmd_b,X,Y,Z` —
/// distinct prefix from the Phase 1 `R/G/B` rows so a log reader
/// can tell which phase each sample is from. XYZ is black-subtracted.
#[allow(clippy::too_many_arguments)]
fn sweep_3d_grid(
    cube_edge: usize,
    min_cmd: f64,
    per_channel_peaks: [f64; 3],
    black_xyz: Xyz,
    settle: Duration,
    device: &mut Colorimeter,
    patch: &mut PatchSurface,
    baseline: &OutputBaseline,
    setup: &Setup,
    cal: &Calibration,
    mut log: Option<&mut (PathBuf, BufWriter<File>)>,
) -> Result<ResponseGrid> {
    if cube_edge < 2 {
        anyhow::bail!("cube_edge_cmd must be ≥ 2 (degenerate 1D grid not supported)");
    }
    let axis_cmds = [
        log_spaced_targets(min_cmd, per_channel_peaks[0], cube_edge),
        log_spaced_targets(min_cmd, per_channel_peaks[1], cube_edge),
        log_spaced_targets(min_cmd, per_channel_peaks[2], cube_edge),
    ];
    eprintln!(
        "  per-axis cmd ranges: R[{:.2}..{:.2}] G[{:.2}..{:.2}] B[{:.2}..{:.2}]",
        axis_cmds[0][0], axis_cmds[0][cube_edge - 1],
        axis_cmds[1][0], axis_cmds[1][cube_edge - 1],
        axis_cmds[2][0], axis_cmds[2][cube_edge - 1],
    );
    if let Some((_, w)) = log.as_mut() {
        writeln!(
            w,
            "# phase 2: 3D forward grid sweep, cube_edge_cmd={cube_edge} ({} patches)",
            cube_edge.pow(3),
        )?;
        writeln!(w, "phase,i,j,k,cmd_r,cmd_g,cmd_b,X,Y,Z")?;
    }

    let total = cube_edge.pow(3);
    let mut xyz: Vec<Xyz> = Vec::with_capacity(total);
    let mut patches_done = 0usize;
    // Progress heartbeat at 10% increments — the 729-patch sweep takes
    // long enough that a silent stretch reads as a hang.
    let progress_step = (total / 10).max(1);
    let start = std::time::Instant::now();
    for k in 0..cube_edge {
        let cmd_b = axis_cmds[2][k];
        for j in 0..cube_edge {
            let cmd_g = axis_cmds[1][j];
            for i in 0..cube_edge {
                let cmd_r = axis_cmds[0][i];
                set_rgb_patch(patch, baseline, [cmd_r, cmd_g, cmd_b])?;
                thread::sleep(settle);
                let raw = device.measure_raw(setup).context("3D-sweep measure")?;
                let raw_xyz = raw_to_xyz(&raw, setup, cal);
                let xyz_above_black = Xyz {
                    x: (raw_xyz.x - black_xyz.x).max(0.0),
                    y: (raw_xyz.y - black_xyz.y).max(0.0),
                    z: (raw_xyz.z - black_xyz.z).max(0.0),
                };
                xyz.push(xyz_above_black);
                if let Some((_, w)) = log.as_mut() {
                    writeln!(
                        w,
                        "3D,{i},{j},{k},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                        cmd_r, cmd_g, cmd_b,
                        xyz_above_black.x, xyz_above_black.y, xyz_above_black.z,
                    )?;
                }
                patches_done += 1;
                if patches_done % progress_step == 0 || patches_done == total {
                    let elapsed = start.elapsed().as_secs_f64();
                    let rate = patches_done as f64 / elapsed;
                    let eta_secs = (total - patches_done) as f64 / rate.max(1e-6);
                    eprintln!(
                        "  3D sweep: {}/{} patches ({:.0}%) — {:.1} patches/s, ETA {:.0}s",
                        patches_done, total,
                        (patches_done as f64 / total as f64) * 100.0,
                        rate, eta_secs,
                    );
                }
            }
        }
    }
    eprintln!(
        "  3D sweep complete: {} patches in {:.1}s",
        total, start.elapsed().as_secs_f64(),
    );
    Ok(ResponseGrid { cube_edge, axis_cmds, xyz })
}

/// Render BT.2020 D65 white at a range of luminances through the
/// freshly-pushed LUT and measure how close the panel lands on the
/// reference. Δu'v' large means the LUT's chromaticity inversion is
/// off; Y-error large means the LUT's luminance inversion is off.
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
    grid: &ResponseGrid,
    black_floor_xyz: [f64; 3],
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
            "# phase 5 verify: D65 white sweep — Δu'v' from D65=({:.4},{:.4})",
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
        let grid_pred = grid.forward(scanout_decoded);
        let grid_predicted_y = grid_pred.y + black_floor_xyz[1];

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
                "#   grid_predicted_y={:.4} grid_vs_measured_err_pct={:+.3}",
                grid_predicted_y,
                (xyz.y - grid_predicted_y) / grid_predicted_y.max(0.01) * 100.0,
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

/// Build the inverse 3D LUT from the measured 3D forward grid.
/// Iteration order is X-fastest then Y then Z — matches the binary
/// file format + the GPU image memory walk.
///
/// `black_floor_xyz` is the panel's emission at (R=G=B=0). The grid
/// stores emission above black; the user-facing target is the panel's
/// TOTAL reading (emission + black). So each grid point's invert
/// target is `target_xyz - black_floor_xyz`, floor-clamped to zero.
///
/// Fast path for sub-floor targets: if the requested emission is
/// below what the grid's smallest cmd produces, the panel can't
/// render any darker, so we emit cmd = (0, 0, 0) directly — the
/// panel renders its black floor and that's the closest we get.
///
/// Returns `(entries, per-grid residual L2 norms)`. Caller reports
/// convergence stats so a bad inversion can't sneak through silently.
fn build_inverse_lut(
    cube_edge: u32,
    grid: &ResponseGrid,
    seed_gain: [f64; 3],
    black_floor_xyz: [f64; 3],
) -> (Vec<[f32; 3]>, Vec<f64>) {
    let n = cube_edge as usize;
    let denom = (cube_edge - 1) as f32;
    let mut entries = Vec::with_capacity(n * n * n);
    let mut residuals = Vec::with_capacity(n * n * n);
    let mut total_residual = 0.0_f64;
    let mut worst_residual = 0.0_f64;
    let grid_floor = grid.min_emission();
    for k in 0..n {
        let bz_in = pq_eotf(k as f32 / denom) as f64;
        for j in 0..n {
            let g_in = pq_eotf(j as f32 / denom) as f64;
            for i in 0..n {
                let r_in = pq_eotf(i as f32 / denom) as f64;
                let target_xyz = bt2020_to_xyz(r_in, g_in, bz_in);
                let emission_target = [
                    (target_xyz[0] - black_floor_xyz[0]).max(0.0),
                    (target_xyz[1] - black_floor_xyz[1]).max(0.0),
                    (target_xyz[2] - black_floor_xyz[2]).max(0.0),
                ];
                // Sub-floor short-circuit: if all three target components
                // are below what the grid's dimmest cmd produces, the
                // panel can only render down to that floor. cmd=0 gets
                // us the panel's black emission, which is the closest
                // we can come — no Newton needed.
                let (cmd, residual) = if emission_target[0] <= grid_floor.x
                    && emission_target[1] <= grid_floor.y
                    && emission_target[2] <= grid_floor.z
                {
                    let residual = (emission_target[0].powi(2)
                        + emission_target[1].powi(2)
                        + emission_target[2].powi(2))
                    .sqrt();
                    ([0.0, 0.0, 0.0], residual)
                } else {
                    invert_one(grid, seed_gain, emission_target)
                };
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

/// Damped-Newton-with-backtracking-line-search inversion of one
/// target XYZ against the 3D measured forward grid. Returns the
/// commanded triple plus the residual norm.
///
/// `seed_gain` is the per-channel Y-per-cmd approximation from the
/// per-channel discovery sweep; used to seed Newton near the right
/// scale. The grid's trilinear forward + analytic Jacobian do the
/// actual convergence — full-step gradients aren't directly visible
/// here because they live inside `grid.jacobian`.
///
/// Why damped + line-search: the grid's forward is non-linear and
/// the Jacobian is constant only within a cell; stepping across
/// cell boundaries can produce wildly wrong predictions. Accept
/// only steps that reduce the residual norm; halve until they do.
fn invert_one(
    grid: &ResponseGrid,
    seed_gain: [f64; 3],
    target_emission: [f64; 3],
) -> ([f64; 3], f64) {
    // BT.2020 D65 weights distribute Y across primaries — a sane
    // additive seed even though the panel itself is sub-additive.
    // Newton refines from there.
    const D65_WEIGHTS: [f64; 3] = [0.2627, 0.6780, 0.0593];
    let max_cmd = grid.max_cmd();
    let target_y = target_emission[1];
    let mut cmd = [
        ((target_y * D65_WEIGHTS[0]) / seed_gain[0].max(1e-6)).clamp(0.0, max_cmd[0]),
        ((target_y * D65_WEIGHTS[1]) / seed_gain[1].max(1e-6)).clamp(0.0, max_cmd[1]),
        ((target_y * D65_WEIGHTS[2]) / seed_gain[2].max(1e-6)).clamp(0.0, max_cmd[2]),
    ];

    const MAX_ITERS: usize = 40;
    const TOL: f64 = 0.005;
    const MIN_STEP_FRACTION: f64 = 1.0 / 64.0;

    let mut res_norm = predicted_residual_norm(grid, &cmd, &target_emission);
    if res_norm < TOL {
        return (cmd, res_norm);
    }

    for _ in 0..MAX_ITERS {
        let jac = grid.jacobian(cmd);
        let xyz = grid.forward(cmd);
        let residual = [
            xyz.x - target_emission[0],
            xyz.y - target_emission[1],
            xyz.z - target_emission[2],
        ];
        let Some(jac_inv) = mat3_inverse(&jac) else {
            // Singular Jacobian: cmd is parked at a grid corner where
            // some axis span is degenerate. Surface the current best.
            return (cmd, res_norm);
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
            return (cmd, res_norm);
        }
        if res_norm < TOL {
            return (cmd, res_norm);
        }
    }
    (cmd, res_norm)
}

/// L2 norm of (grid.forward(cmd) - target_emission). Helper for
/// inverter's line-search comparison; pulled out so the loop reads
/// cleanly.
fn predicted_residual_norm(
    grid: &ResponseGrid,
    cmd: &[f64; 3],
    target_emission: &[f64; 3],
) -> f64 {
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
    let path = args.log.clone().unwrap_or_else(|| {
        let safe = args.output.replace('/', "_");
        PathBuf::from(format!("prism-calibrate-lut3d-{safe}.csv"))
    });
    let file = File::create(&path)
        .with_context(|| format!("create log file {}", path.display()))?;
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
    // Phase 2 rows use the `3D,i,j,k,cmd_r,cmd_g,cmd_b,X,Y,Z` shape
    // declared at the top of that block. Verify rows are tagged
    // `verify,W,…`. All XYZ values are black-subtracted (raw floor
    // recoverable from the `# black_floor` comment line below).
    writeln!(w, "channel,sample_idx,commanded_nits,X,Y,Z")?;
    eprintln!("Logging per-sample CSV to {}", path.display());
    Ok(Some((path, w)))
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
        ResponseGrid { cube_edge, axis_cmds, xyz }
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
            let (cmd, residual) = invert_one(&grid, seed_gain, target);
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
        let (cmd, _residual) = invert_one(&grid, seed_gain, target);
        // cmd_R dominates (close to 50, within trilinear noise).
        assert!((cmd[0] - 50.0).abs() < 5.0, "cmd_R={}", cmd[0]);
        // cmd_G, cmd_B small relative to cmd_R.
        assert!(cmd[1] < cmd[0] * 0.2, "cmd_G should be small, got {}", cmd[1]);
        assert!(cmd[2] < cmd[0] * 0.2, "cmd_B should be small, got {}", cmd[2]);
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
        assert!((xyz.x - expected.x).abs() < 1e-9, "x: got {}, want {}", xyz.x, expected.x);
        assert!((xyz.y - expected.y).abs() < 1e-9, "y: got {}, want {}", xyz.y, expected.y);
        assert!((xyz.z - expected.z).abs() < 1e-9, "z: got {}, want {}", xyz.z, expected.z);
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
            "midpoint X: got {}, want {}", xyz.x, mid_r,
        );
    }
}

fn print_kdl_block(
    output_name: &str,
    hdr_active: bool,
    peaks: [f32; 3],
    black_xyz: [f32; 3],
    lut_path: &std::path::Path,
) {
    println!();
    println!(
        "# Measured black floor for {output_name}: X={:.4} Y={:.4} Z={:.4} cd/m²",
        black_xyz[0], black_xyz[1], black_xyz[2],
    );
    println!(
        "# (carried in the LUT file header; compositor exposes it via OutputContext for tone mapping)",
    );
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
