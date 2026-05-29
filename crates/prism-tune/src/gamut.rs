//! Closed-loop cube-surface gamut probe.
//!
//! The panel's reachable gamut is the image of the native command cube
//! `[0, max]³` under the measured forward response. For a continuous
//! pipeline the *boundary* of that solid is the image of the cube's
//! surface, so we probe the surface directly — every measurement is a
//! boundary point of the reachable volume, no inversion needed. A
//! coarse pass measures the 8 corners + 6 face centres (14 vertices);
//! optional adaptive quadtree refinement subdivides each face where the
//! pipeline is curved and stops where it's already trustworthy-flat,
//! has measurably *folded* (consecutive corners collapse in measured
//! space — the pipeline is clamping), or where confidence is too low to
//! chase a deviation (a noise read, not real curvature).
//!
//! The mesh persists alongside the inverse LUT and supersedes the
//! scalar `black_point_xyz`: the volume's lower vertex IS its true
//! black (the `cv = (0,0,0)` corner), measured absolute. Out-of-gamut
//! and below-floor requests project onto this surface during the bake
//! rather than being subtracted-then-clamped.
//!
//! This is a deliberate re-implementation of tristim's cube-surface
//! gamut probe (`tristim-gather::gamut`), kept independent so tristim
//! remains a valid external check on prism's own gamut definition.

use crate::common::{set_patch_off, set_rgb_patch, OutputBaseline};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::thread;
use std::time::Duration;
use tristim_display::PatchSurface;
use tristim_driver::{
    AdaptiveTier, Calibration, Colorimeter, MeasurementConfidence, Setup, TrustFlag, Xyz,
};

/// The 14 coarse cube-surface probe points: 8 corners (black, three
/// primaries, three secondaries, white) + 6 face centres. Each entry
/// is `(label, code value in [0, 1]³)`. Reference set; the adaptive
/// refinement subdivides each face on demand rather than walking this
/// list directly.
#[allow(dead_code)]
pub const PROBE_POINTS: &[(&str, [f64; 3])] = &[
    ("black", [0.0, 0.0, 0.0]),
    ("red", [1.0, 0.0, 0.0]),
    ("green", [0.0, 1.0, 0.0]),
    ("blue", [0.0, 0.0, 1.0]),
    ("yellow", [1.0, 1.0, 0.0]),
    ("cyan", [0.0, 1.0, 1.0]),
    ("magenta", [1.0, 0.0, 1.0]),
    ("white", [1.0, 1.0, 1.0]),
    ("R=0", [0.0, 0.5, 0.5]),
    ("R=1", [1.0, 0.5, 0.5]),
    ("G=0", [0.5, 0.0, 0.5]),
    ("G=1", [0.5, 1.0, 0.5]),
    ("B=0", [0.5, 0.5, 0.0]),
    ("B=1", [0.5, 0.5, 1.0]),
];

const WHITE_CV: [f64; 3] = [1.0, 1.0, 1.0];

/// The 6 cube faces as `(fixed axis, value)`: axis 0=R, 1=G, 2=B.
const FACE_DEFS: [(usize, f64); 6] = [(0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0), (2, 0.0), (2, 1.0)];

/// One probe sample: what came back for a requested code value, plus
/// the coarse trust verdict the refinement gates on. Low-trust samples
/// cap subdivision rather than misleading it with noise.
#[derive(Debug, Clone, Copy)]
pub struct ProbeSample {
    pub measured: Xyz,
    pub trustworthy: bool,
}

/// Tunable thresholds for [`refine_gamut`]. All ΔE figures are CIE76 in
/// the measured-white-referenced Lab the mesh lives in.
///
/// Defaults mirror tristim's, calibrated on real captures: a patch is
/// flat when its centre is within ΔE 2 of its corners' bilinear average
/// (~the perceptual just-noticeable bound), and folded when the four
/// corners agree to within ΔE 0.5 over at least a quarter of a face
/// side — that's a measured-space collapse, not ordinary convergence.
#[derive(Debug, Clone, Copy)]
pub struct RefineParams {
    pub max_depth: u32,
    pub flat_eps: f64,
    pub fold_eps: f64,
    pub fold_min_side: f64,
}

impl Default for RefineParams {
    fn default() -> Self {
        Self {
            max_depth: 3,
            flat_eps: 2.0,
            fold_eps: 0.5,
            fold_min_side: 0.25,
        }
    }
}

/// Why a leaf patch stopped subdividing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchStatus {
    /// Planar enough (centre within `flat_eps` of the bilinear average).
    Flat,
    /// Corners collapsed in measured space over a non-trivial CV span —
    /// the pipeline clamped this region onto the reachable boundary.
    Folded,
    /// Hit the depth cap while still curved.
    MaxDepth,
    /// A corner was untrustworthy; stopped rather than chase noise.
    LowTrust,
}

impl PatchStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Folded => "folded",
            Self::MaxDepth => "max_depth",
            Self::LowTrust => "low_trust",
        }
    }

    /// Stable enum value for any future binary file format (currently
    /// only persisted via the JSON sidecar's `as_str` form).
    #[allow(dead_code)]
    pub fn as_u32(&self) -> u32 {
        match self {
            Self::Flat => 0,
            Self::Folded => 1,
            Self::MaxDepth => 2,
            Self::LowTrust => 3,
        }
    }

    #[allow(dead_code)]
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Flat,
            1 => Self::Folded,
            2 => Self::MaxDepth,
            3 => Self::LowTrust,
            _ => return None,
        })
    }
}

/// A measured boundary vertex: the code value we asked for, the actual
/// per-channel nits commanded (`cv * cmd_axis_max_nits`), and what came
/// back from the colorimeter. Lab is relative to the mesh's measured
/// white so refinement thresholds are perceptual.
#[derive(Debug, Clone, Copy)]
pub struct MeshVertex {
    pub code_value: [f64; 3],
    pub cmd_nits: [f64; 3],
    pub xyz: Xyz,
    pub lab: [f64; 3],
    pub trustworthy: bool,
}

/// A leaf patch of a refined face: its 4 corner vertex indices (CCW)
/// and why subdivision stopped.
#[derive(Debug, Clone, Copy)]
pub struct Patch {
    /// Face's fixed axis (0=R, 1=G, 2=B).
    pub axis: usize,
    /// Face's fixed value (0.0 or 1.0).
    pub value: f64,
    /// Indices into [`GamutMesh::vertices`].
    pub corners: [usize; 4],
    pub status: PatchStatus,
}

impl Patch {
    pub fn face_label(&self) -> String {
        let name = ["R", "G", "B"][self.axis];
        format!("{name}={}", self.value as u8)
    }
}

/// The refined measured gamut: deduped boundary vertices + quadtree
/// leaf patches, in absolute XYZ. The per-channel `cmd_axis_max_nits`
/// records what `cv = 1` meant on each axis at probe time — needed to
/// reconstruct the cmd-space cube anyone later wants to invert against.
#[derive(Debug, Clone)]
pub struct GamutMesh {
    pub white: Xyz,
    pub cmd_axis_max_nits: [f64; 3],
    pub vertices: Vec<MeshVertex>,
    pub patches: Vec<Patch>,
}

impl GamutMesh {
    /// Count leaf patches with a given status.
    pub fn count(&self, status: PatchStatus) -> usize {
        self.patches.iter().filter(|p| p.status == status).count()
    }

    /// The measured vertex for an exact code value, if it was probed.
    pub fn vertex_at(&self, cv: [f64; 3]) -> Option<&MeshVertex> {
        self.vertices.iter().find(|v| v.code_value == cv)
    }

    /// The measured black corner — `cv = (0,0,0)`. This supersedes the
    /// historical separate `black_floor_xyz` scalar: the lower vertex
    /// of the reachable volume is part of the mesh, measured absolute.
    pub fn black(&self) -> Option<Xyz> {
        self.vertex_at([0.0, 0.0, 0.0]).map(|v| v.xyz)
    }

    /// The measured white corner — `cv = (1,1,1)`.
    #[allow(dead_code)]
    pub fn measured_white(&self) -> Xyz {
        self.white
    }

    /// Number of leaf patches.
    #[allow(dead_code)]
    pub fn patch_count(&self) -> usize {
        self.patches.len()
    }
}

// ── refinement ───────────────────────────────────────────────────────────────

/// Adaptive cube-surface refinement, generic over how a code value is
/// measured. The hardware path passes the patch surface + colorimeter
/// in; tests inject a synthetic display.
///
/// Measures the white corner first to fix the Lab reference, then
/// refines each of the 6 faces by quadtree subdivision.
pub fn refine_gamut<M, E>(params: &RefineParams, mut measure: M) -> Result<GamutMesh, E>
where
    M: FnMut([f64; 3]) -> Result<ProbeSample, E>,
{
    let white_ps = measure(WHITE_CV)?;
    let white = white_ps.measured;

    let mut ctx = RefineCtx {
        params,
        white,
        vertices: Vec::new(),
        cache: HashMap::new(),
        patches: Vec::new(),
    };
    // Pre-insert white as vertex 0 so faces sharing it hit the cache.
    let wlab = xyz_to_lab([white.x, white.y, white.z], [white.x, white.y, white.z]);
    ctx.vertices.push(MeshVertex {
        code_value: WHITE_CV,
        cmd_nits: [0.0; 3], // filled in by the measurement wrapper
        xyz: white,
        lab: wlab,
        trustworthy: white_ps.trustworthy,
    });
    ctx.cache.insert(cv_key(WHITE_CV), 0);

    for &(axis, value) in &FACE_DEFS {
        ctx.refine(axis, value, 0.0, 1.0, 0.0, 1.0, 0, &mut measure)?;
    }

    Ok(GamutMesh {
        white,
        cmd_axis_max_nits: [0.0; 3], // populated by the hardware entry below
        vertices: ctx.vertices,
        patches: ctx.patches,
    })
}

/// Dedup key for a code value. The bisection only ever produces dyadic
/// rationals, so bit patterns are a stable exact key.
fn cv_key(cv: [f64; 3]) -> [u64; 3] {
    [cv[0].to_bits(), cv[1].to_bits(), cv[2].to_bits()]
}

/// Code value on face `(axis, value)` at face coordinates `(s, t)`: the
/// fixed axis is held at `value`, the other two sweep `s` and `t` in
/// axis order.
fn face_cv(axis: usize, value: f64, s: f64, t: f64) -> [f64; 3] {
    let mut cv = [0.0; 3];
    cv[axis] = value;
    let others = match axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    cv[others[0]] = s;
    cv[others[1]] = t;
    cv
}

struct RefineCtx<'a> {
    params: &'a RefineParams,
    white: Xyz,
    vertices: Vec<MeshVertex>,
    cache: HashMap<[u64; 3], usize>,
    patches: Vec<Patch>,
}

impl RefineCtx<'_> {
    /// Measure (or recall from the cache) the vertex at the requested
    /// code value, returning its index in `self.vertices`.
    fn sample<M, E>(&mut self, cv: [f64; 3], measure: &mut M) -> Result<usize, E>
    where
        M: FnMut([f64; 3]) -> Result<ProbeSample, E>,
    {
        if let Some(&idx) = self.cache.get(&cv_key(cv)) {
            return Ok(idx);
        }
        let ps = measure(cv)?;
        let lab = xyz_to_lab(
            [ps.measured.x, ps.measured.y, ps.measured.z],
            [self.white.x, self.white.y, self.white.z],
        );
        let idx = self.vertices.len();
        self.vertices.push(MeshVertex {
            code_value: cv,
            cmd_nits: [0.0; 3], // filled in by the hardware wrapper
            xyz: ps.measured,
            lab,
            trustworthy: ps.trustworthy,
        });
        self.cache.insert(cv_key(cv), idx);
        Ok(idx)
    }

    #[allow(clippy::too_many_arguments)]
    fn refine<M, E>(
        &mut self,
        axis: usize,
        value: f64,
        s0: f64,
        s1: f64,
        t0: f64,
        t1: f64,
        depth: u32,
        measure: &mut M,
    ) -> Result<(), E>
    where
        M: FnMut([f64; 3]) -> Result<ProbeSample, E>,
    {
        let c = [
            self.sample(face_cv(axis, value, s0, t0), measure)?,
            self.sample(face_cv(axis, value, s0, t1), measure)?,
            self.sample(face_cv(axis, value, s1, t1), measure)?,
            self.sample(face_cv(axis, value, s1, t0), measure)?,
        ];
        let labs: [[f64; 3]; 4] = [
            self.vertices[c[0]].lab,
            self.vertices[c[1]].lab,
            self.vertices[c[2]].lab,
            self.vertices[c[3]].lab,
        ];
        // Measured-space size of the patch = the widest corner-to-corner ΔE.
        let mut spread = 0.0_f64;
        for i in 0..4 {
            for j in (i + 1)..4 {
                spread = spread.max(delta_e76(labs[i], labs[j]));
            }
        }
        let side = s1 - s0;
        let emit = |this: &mut Self, status| {
            this.patches.push(Patch {
                axis,
                value,
                corners: c,
                status,
            });
        };

        if spread < self.params.fold_eps {
            let status = if side >= self.params.fold_min_side {
                PatchStatus::Folded
            } else {
                PatchStatus::Flat
            };
            emit(self, status);
            return Ok(());
        }
        if depth >= self.params.max_depth {
            emit(self, PatchStatus::MaxDepth);
            return Ok(());
        }
        if !c.iter().all(|&i| self.vertices[i].trustworthy) {
            emit(self, PatchStatus::LowTrust);
            return Ok(());
        }

        let sm = 0.5 * (s0 + s1);
        let tm = 0.5 * (t0 + t1);
        let center = self.sample(face_cv(axis, value, sm, tm), measure)?;
        let bilinear = [
            0.25 * (labs[0][0] + labs[1][0] + labs[2][0] + labs[3][0]),
            0.25 * (labs[0][1] + labs[1][1] + labs[2][1] + labs[3][1]),
            0.25 * (labs[0][2] + labs[1][2] + labs[2][2] + labs[3][2]),
        ];
        if delta_e76(self.vertices[center].lab, bilinear) < self.params.flat_eps {
            emit(self, PatchStatus::Flat);
            return Ok(());
        }

        // Curved and trustworthy: split into 4 quadrants.
        self.refine(axis, value, s0, sm, t0, tm, depth + 1, measure)?;
        self.refine(axis, value, sm, s1, t0, tm, depth + 1, measure)?;
        self.refine(axis, value, s0, sm, tm, t1, depth + 1, measure)?;
        self.refine(axis, value, sm, s1, tm, t1, depth + 1, measure)?;
        Ok(())
    }
}

// ── hardware driver ──────────────────────────────────────────────────────────

/// Configuration for the hardware gamut probe. `cmd_axis_max_nits` is
/// the per-channel saturation peak discovered in the per-channel
/// pre-probe; `cv = 1` on each axis maps to that nits value.
///
/// `fast_integration_ms` enables adaptive per-point integration: each
/// vertex first burst-measures at the override integration time, and
/// re-measures at the calibration default only if the fast result
/// fails the confidence gate. Bright easy points stay fast; dim or
/// saturated points pay the full integration. `None` keeps the
/// legacy single-tier behaviour. See `Colorimeter::measure_adaptive`.
#[derive(Debug, Clone)]
pub struct ProbeConfig {
    pub cmd_axis_max_nits: [f64; 3],
    pub repeats: usize,
    pub settle: Duration,
    pub settle_black: Duration,
    pub fast_integration_ms: Option<u16>,
}

/// Drive the colorimeter + patch surface through an adaptive cube-
/// surface gamut probe. The `cmd_axis_max_nits` config field tells the
/// probe what nits to drive at each axis's `cv = 1`. Returns the
/// refined [`GamutMesh`] with `cmd_axis_max_nits` already populated and
/// each vertex's `cmd_nits` filled in.
///
/// Black is given an extended settle so LCD/OLED gate-off transients
/// don't bleed into the most noise-sensitive vertex of the mesh.
#[allow(clippy::too_many_arguments)]
pub fn probe_gamut_refined(
    config: &ProbeConfig,
    params: &RefineParams,
    baseline: &OutputBaseline,
    device: &mut Colorimeter,
    patch: &mut PatchSurface,
    setup: &Setup,
    cal: &Calibration,
    mut on_event: impl FnMut(GamutProbeEvent),
) -> Result<GamutMesh> {
    let mut index = 0usize;
    let mut vertex_cmd_nits: HashMap<[u64; 3], [f64; 3]> = HashMap::new();
    let measure = |cv: [f64; 3]| -> Result<ProbeSample> {
        let cmd_nits = [
            cv[0] * config.cmd_axis_max_nits[0],
            cv[1] * config.cmd_axis_max_nits[1],
            cv[2] * config.cmd_axis_max_nits[2],
        ];
        set_rgb_patch(patch, baseline, cmd_nits)?;
        // Black is the noise-sensitive corner — OLED subpixels can take
        // a moment to fully gate off after a colour swap.
        let is_black = cv == [0.0, 0.0, 0.0];
        thread::sleep(if is_black {
            config.settle_black
        } else {
            config.settle
        });
        // Adaptive integration: bright vertices clear trust at the short
        // tier and skip the long integration entirely. Dim/saturated
        // vertices auto-escalate to the calibration default. `setup`/`cal`
        // come back paired with the actual raws (the fast tier uses a
        // scaled cal matrix), so they go straight into `from_repeats`.
        let m = device
            .measure_adaptive(setup, cal, config.repeats, config.fast_integration_ms)
            .context("gamut probe measure_adaptive")?;
        let confidence = MeasurementConfidence::from_repeats(&m.raws, &m.setup, &m.cal);
        let sample = ProbeSample {
            measured: confidence.mean,
            trustworthy: confidence.is_trustworthy(),
        };
        let flags = confidence.flags();
        on_event(GamutProbeEvent::Measured {
            index,
            code_value: cv,
            cmd_nits,
            measured: sample.measured,
            flags: flags.clone(),
            tier: m.tier,
        });
        vertex_cmd_nits.insert(cv_key(cv), cmd_nits);
        index += 1;
        Ok(sample)
    };

    // Avoid the closure capturing the HashMap by move: take the result
    // and patch up cmd_nits after the fact. This keeps the closure
    // small and avoids RefCell.
    let mut mesh = refine_gamut(params, measure)?;
    mesh.cmd_axis_max_nits = config.cmd_axis_max_nits;
    for v in &mut mesh.vertices {
        if let Some(&cmd) = vertex_cmd_nits.get(&cv_key(v.code_value)) {
            v.cmd_nits = cmd;
        }
    }
    let _ = set_patch_off(patch, baseline.hdr_active);
    Ok(mesh)
}

/// Progress events for the hardware probe.
#[derive(Debug, Clone)]
pub enum GamutProbeEvent {
    Measured {
        index: usize,
        code_value: [f64; 3],
        cmd_nits: [f64; 3],
        measured: Xyz,
        flags: Vec<TrustFlag>,
        /// Which integration tier produced this measurement — `Fast`
        /// (short integration cleared trust), `EscalatedFull` (short
        /// failed trust, re-measured at default), or `SingleFull`
        /// (adaptive disabled). Useful for surfacing where the probe
        /// is actually spending its time.
        tier: AdaptiveTier,
    },
}

// ── color helpers ────────────────────────────────────────────────────────────

/// CIELAB conversion using `wp` as the reference white. Standard CIE
/// formula with the cube-root + linear-toe split.
pub fn xyz_to_lab(xyz: [f64; 3], wp: [f64; 3]) -> [f64; 3] {
    let f = |t: f64| -> f64 {
        const DELTA: f64 = 6.0 / 29.0;
        if t > DELTA.powi(3) {
            t.cbrt()
        } else {
            t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
        }
    };
    let xn = (xyz[0] / wp[0].max(1e-12)).max(0.0);
    let yn = (xyz[1] / wp[1].max(1e-12)).max(0.0);
    let zn = (xyz[2] / wp[2].max(1e-12)).max(0.0);
    let fx = f(xn);
    let fy = f(yn);
    let fz = f(zn);
    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);
    [l, a, b]
}

/// CIE76 colour difference — Euclidean distance in Lab. Imperfect
/// perceptually (CIEDE2000 is better) but fine as a refinement
/// threshold; we just need "patches that look similar enough to stop
/// subdividing" and a few-percent metric error is far below the noise
/// floor of the measurement that drives it.
pub fn delta_e76(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dl = a[0] - b[0];
    let da = a[1] - b[1];
    let db = a[2] - b[2];
    (dl * dl + da * da + db * db).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    // sRGB linear-RGB → XYZ (D65). Smooth additive display for the
    // synthetic tests.
    const SRGB_TO_XYZ: [[f64; 3]; 3] = [
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ];

    fn mat_mul(m: [[f64; 3]; 3], v: [f64; 3]) -> Xyz {
        Xyz {
            x: m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            y: m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            z: m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        }
    }

    fn smooth(cv: [f64; 3]) -> Result<ProbeSample, Infallible> {
        Ok(ProbeSample {
            measured: mat_mul(SRGB_TO_XYZ, cv),
            trustworthy: true,
        })
    }

    /// Display whose gamut is smaller than the container: each channel
    /// clamps at 0.5, collapsing the high-saturation corner of every
    /// face onto one point — a fold the probe must detect rather than
    /// silently let collapse the mesh.
    fn clamped(cv: [f64; 3]) -> Result<ProbeSample, Infallible> {
        let disp = cv.map(|c| c.min(0.5));
        Ok(ProbeSample {
            measured: mat_mul(SRGB_TO_XYZ, disp),
            trustworthy: true,
        })
    }

    #[test]
    fn smooth_display_recovers_primaries_no_folds() {
        let mesh = refine_gamut(&RefineParams::default(), smooth).unwrap();
        assert!(
            mesh.patches.len() > FACE_DEFS.len(),
            "expected subdivision beyond the 6 face roots, got {} patches",
            mesh.patches.len()
        );
        assert_eq!(mesh.count(PatchStatus::Folded), 0, "no clamping expected");

        let red = mesh.vertex_at([1.0, 0.0, 0.0]).unwrap();
        let (x, y) = red.xyz.chromaticity().unwrap();
        assert!((x - 0.640).abs() < 1e-3, "red x {x}");
        assert!((y - 0.330).abs() < 1e-3, "red y {y}");
    }

    #[test]
    fn clamped_display_detects_folds() {
        let mesh = refine_gamut(&RefineParams::default(), clamped).unwrap();
        assert!(
            mesh.count(PatchStatus::Folded) > 0,
            "expected at least one folded (clamped) patch"
        );
        // White clamps to half-scale, so the gamut is genuinely smaller
        // — not collapsed to nothing.
        let total: f64 = SRGB_TO_XYZ[1].iter().sum();
        assert!(mesh.white.y > 0.0 && mesh.white.y < total);
    }

    #[test]
    fn lowtrust_corner_stops_refinement() {
        // Trustworthy for the white corner (so the Lab reference is
        // sane), low-trust everywhere else. Refinement must stop at
        // each face's first 4-corner check and emit LowTrust.
        let measure = |cv: [f64; 3]| -> Result<ProbeSample, Infallible> {
            Ok(ProbeSample {
                measured: mat_mul(SRGB_TO_XYZ, cv),
                trustworthy: cv == WHITE_CV,
            })
        };
        let mesh = refine_gamut(&RefineParams::default(), measure).unwrap();
        assert!(
            mesh.count(PatchStatus::LowTrust) > 0,
            "expected LowTrust patches; got {:?}",
            mesh.patches
                .iter()
                .map(|p| p.status.as_str())
                .collect::<Vec<_>>()
        );
        // Each face still produces exactly one leaf patch — no further
        // subdivision happens once a low-trust corner is detected.
        assert_eq!(mesh.patches.len(), FACE_DEFS.len());
    }

    #[test]
    fn xyz_to_lab_white_at_reference_is_l100() {
        let wp = [0.95047, 1.0, 1.08883]; // D65
        let lab = xyz_to_lab(wp, wp);
        assert!((lab[0] - 100.0).abs() < 1e-9, "L* of reference white");
        assert!(lab[1].abs() < 1e-9 && lab[2].abs() < 1e-9, "a*b* neutral");
    }

    #[test]
    fn delta_e76_is_euclidean() {
        let a = [50.0, 10.0, -5.0];
        let b = [55.0, 14.0, -2.0];
        let expected = (25.0_f64 + 16.0 + 9.0).sqrt();
        assert!((delta_e76(a, b) - expected).abs() < 1e-12);
    }
}
