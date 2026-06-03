//! Local color science for the 3D gamut view: BT.2020/sRGB/P3 → XYZ →
//! CIELAB, and gamut-cage wireframes. Reimplemented here (rather than
//! pulling `tristim-color`, which would force a bump of the pinned
//! tristim git rev) — the needed subset is small and standard.
//!
//! Two plot spaces ([`GamutSpace`]):
//!
//! - **CIELAB** — perceptual; world X = a*, Y = L* (up), Z = b*, D65.
//!   Lab is *relative by construction* (L* = 100 is "white"), so this
//!   mode anchors to a fixed [`REFERENCE_WHITE_NITS`] (not the output's
//!   `sdr_reference_nits`); content brighter than reference white extends
//!   above L* = 100. Each cage's white corner sits at L* = 100, so a
//!   reference-white-luminance sample lands on the cage white.
//!
//! - **BT.2020 RGB** — the raw intermediate buffer values in **absolute
//!   nits** (X=R, Y=G, Z=B), no normalization. This is the truly-absolute
//!   view; reference gamuts (sRGB, P3) appear as nested cubes (scaled to
//!   reference white) showing containment, with the BT.2020 cage as the
//!   axis-aligned outer box.

use std::collections::HashSet;

use damascene_core::scene::glam::Vec3;
use damascene_core::scene::{LabelPlacement, PointLabels};
use damascene_core::scene::{LineData, LinesHandle};
use damascene_core::scene::{LineSegment, PointData, PointsHandle, ScenePoint};

use crate::common::srgb_oetf;

/// Absolute reference white the Lab frame is anchored to: a sample at
/// this luminance maps to L* = 100. ITU-R BT.2408 HDR reference
/// ("graphics") white. Fixed (not the output's `sdr_reference_nits`) so
/// the gamut plot reads in absolute terms.
pub const REFERENCE_WHITE_NITS: f64 = 203.0;

/// D65 reference white in XYZ (Y normalized to 1).
const D65: [f64; 3] = [0.950_47, 1.0, 1.088_83];

/// BT.2020 linear RGB → XYZ (D65). Also the sample-cloud source matrix.
const BT2020_TO_XYZ: [[f64; 3]; 3] = [
    [0.636_958_0, 0.144_616_9, 0.168_881_0],
    [0.262_700_2, 0.677_998_1, 0.059_301_7],
    [0.000_000_0, 0.028_072_7, 1.060_985_1],
];

/// XYZ → BT.2020 linear RGB (D65) — inverse of [`BT2020_TO_XYZ`]. Used
/// to express other gamuts' primaries in the buffer's BT.2020 RGB space.
const XYZ_TO_BT2020: [[f64; 3]; 3] = [
    [1.716_651_2, -0.355_670_8, -0.253_366_3],
    [-0.666_684_4, 1.616_481_2, 0.015_768_5],
    [0.017_639_9, -0.042_770_6, 0.942_103_1],
];

/// sRGB / BT.709 linear RGB → XYZ (D65).
const SRGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.412_456_4, 0.357_576_1, 0.180_437_5],
    [0.212_672_9, 0.715_152_2, 0.072_175_0],
    [0.019_333_9, 0.119_192_0, 0.950_304_1],
];

/// Coordinate space for the gamut plot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GamutSpace {
    /// Perceptual CIELAB, anchored to [`REFERENCE_WHITE_NITS`] (white →
    /// L* = 100). Relative by construction — Lab needs a white reference.
    Cielab,
    /// Raw BT.2020 RGB in **absolute nits**, straight from the
    /// intermediate buffer (X=R, Y=G, Z=B). No normalization; reference
    /// gamuts appear as nested cubes scaled to [`REFERENCE_WHITE_NITS`].
    Bt2020Rgb,
}

/// Display-P3 linear RGB → XYZ (D65).
const P3_TO_XYZ: [[f64; 3]; 3] = [
    [0.486_570_9, 0.265_667_7, 0.198_217_3],
    [0.228_974_6, 0.691_738_5, 0.079_286_9],
    [0.000_000_0, 0.045_113_4, 1.043_944_4],
];

/// A built gamut scene: the measured-color point cloud, the enabled
/// reference gamut-cage wireframes, and one labelled anchor per cage (at
/// its green primary). All are damascene geometry handles (Arc'd,
/// versioned), built once per capture + reference set and cloned into a
/// fresh `SceneSpec` each frame.
pub struct GamutScene {
    pub points: PointsHandle,
    pub cages: LinesHandle,
    /// The measured gamut-surface lattice shell (patch quad-edges), empty
    /// when no mesh is supplied. Folded patches drawn hot.
    pub shell: LinesHandle,
    /// One marker point per enabled cage, at its green primary — the
    /// vertex that differs most between gamuts, so labels don't pile up at
    /// the shared white.
    pub cage_label_geo: PointsHandle,
    /// Persistent in-plot name labels, aligned with `cage_label_geo`.
    pub cage_labels: PointLabels,
    /// Distinct point count after voxel dedup (for the status line).
    pub point_count: usize,
    /// Reference-cage line-segment count. The damascene wgpu backend
    /// rejects empty geometry buffers, so callers must skip a mark whose
    /// count is zero rather than upload it.
    pub cage_segments: usize,
    /// Measured-shell line-segment count (see [`Self::cage_segments`]).
    pub shell_segments: usize,
    /// Cage-label marker count (== enabled cages; see [`Self::cage_segments`]).
    pub cage_label_count: usize,
}

/// The measured gamut shell's normal patch-edge color (a distinct magenta,
/// reading against both the neutral cloud and the colored reference cages).
const SHELL_COLOR: [f32; 4] = [0.93, 0.45, 0.85, 0.7];
/// Folded (clamped) patches of the shell — where pushing the code value
/// stopped moving the measurement; drawn hotter to flag the boundary hit.
const SHELL_FOLD_COLOR: [f32; 4] = [1.0, 0.40, 0.25, 0.9];

/// A reference gamut the plot can outline as a cage overlay.
pub struct RefGamut {
    /// Toggle route suffix (`cage:<key>`) and stable identity.
    pub key: &'static str,
    /// Button / in-plot label text.
    pub name: &'static str,
    /// This gamut's linear-RGB → XYZ matrix.
    matrix: &'static [[f64; 3]; 3],
    /// Cage + label color (authoring sRGBA).
    pub color: [f32; 4],
}

/// Number of reference-gamut overlays offered.
pub const N_REF_GAMUTS: usize = 3;

/// Which reference cages are enabled, parallel to [`REF_GAMUTS`].
pub type RefSet = [bool; N_REF_GAMUTS];

/// Reference gamuts overlaid as cage wireframes, in nesting order
/// (sRGB ⊂ Display P3 ⊂ Rec.2020).
pub const REF_GAMUTS: [RefGamut; N_REF_GAMUTS] = [
    RefGamut {
        key: "srgb",
        name: "sRGB",
        matrix: &SRGB_TO_XYZ,
        color: [0.42, 0.60, 1.0, 0.5],
    },
    RefGamut {
        key: "p3",
        name: "Display P3",
        matrix: &P3_TO_XYZ,
        color: [0.36, 0.85, 0.52, 0.5],
    },
    RefGamut {
        key: "bt2020",
        name: "Rec.2020",
        matrix: &BT2020_TO_XYZ,
        color: [1.0, 0.70, 0.30, 0.5],
    },
];

fn mat_mul(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// CIE XYZ → CIELAB against white `wp`.
fn xyz_to_lab(xyz: [f64; 3], wp: [f64; 3]) -> [f64; 3] {
    fn f(t: f64) -> f64 {
        const D: f64 = 6.0 / 29.0;
        if t > D * D * D {
            t.cbrt()
        } else {
            t / (3.0 * D * D) + 4.0 / 29.0
        }
    }
    let fx = f(xyz[0] / wp[0]);
    let fy = f(xyz[1] / wp[1]);
    let fz = f(xyz[2] / wp[2]);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// Lab `[L*, a*, b*]` → world position (X=a*, Y=L*, Z=b*).
fn lab_to_world(lab: [f64; 3]) -> Vec3 {
    Vec3::new(lab[1] as f32, lab[0] as f32, lab[2] as f32)
}

/// sRGB-encoded authoring-space color for a BT.2020-linear sample
/// (already white-normalized). Mirrors the preview tonemap so a point's
/// swatch matches its pixel.
fn point_color(bt2020: [f64; 3]) -> [f32; 4] {
    let sr = 1.660_491 * bt2020[0] - 0.587_641 * bt2020[1] - 0.072_850 * bt2020[2];
    let sg = -0.124_550 * bt2020[0] + 1.132_900 * bt2020[1] - 0.008_349 * bt2020[2];
    let sb = -0.018_151 * bt2020[0] - 0.100_579 * bt2020[1] + 1.118_730 * bt2020[2];
    let enc = |c: f64| srgb_oetf(c.clamp(0.0, 1.0)) as f32;
    [enc(sr), enc(sg), enc(sb), 1.0]
}

/// Build the point cloud + reference cages from BT.2020 absolute-nits
/// samples, in the requested [`GamutSpace`]. Points are deduplicated on a
/// voxel grid (in the plot's own coordinates) so large flat regions
/// collapse to one mark instead of thousands. Point swatch colors are the
/// same in both spaces (tonemapped relative to reference white so a mark
/// looks like its pixel); only the *positions* differ.
pub fn build_gamut_scene(
    samples: &[[f32; 3]],
    space: GamutSpace,
    refs: RefSet,
    shell: Option<&prism_ipc::GamutMesh>,
) -> GamutScene {
    let inv_ref = 1.0 / REFERENCE_WHITE_NITS;
    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut points: Vec<ScenePoint> = Vec::new();

    for s in samples {
        let nits = [s[0] as f64, s[1] as f64, s[2] as f64];
        // White-relative copy, only for the swatch color.
        let rel = [nits[0] * inv_ref, nits[1] * inv_ref, nits[2] * inv_ref];

        let (position, cell) = match space {
            GamutSpace::Cielab => {
                // Clamp tiny negatives from encode noise before Lab.
                let xyz = mat_mul(
                    &BT2020_TO_XYZ,
                    [rel[0].max(0.0), rel[1].max(0.0), rel[2].max(0.0)],
                );
                let lab = xyz_to_lab(xyz, D65);
                const CELL: f64 = 2.0; // Lab units
                let cell = (
                    (lab[1] / CELL).round() as i32,
                    (lab[0] / CELL).round() as i32,
                    (lab[2] / CELL).round() as i32,
                );
                (lab_to_world(lab), cell)
            }
            GamutSpace::Bt2020Rgb => {
                const CELL: f64 = 4.0; // nits
                let cell = (
                    (nits[0] / CELL).round() as i32,
                    (nits[1] / CELL).round() as i32,
                    (nits[2] / CELL).round() as i32,
                );
                (
                    Vec3::new(nits[0] as f32, nits[1] as f32, nits[2] as f32),
                    cell,
                )
            }
        };

        if seen.insert(cell) {
            points.push(ScenePoint {
                position,
                color: point_color(rel),
            });
        }
    }

    // Only the enabled cages are drawn; each gets a labelled marker at its
    // green primary so every outlined gamut is named where it's most
    // distinctive (green diverges most between gamuts).
    let mut cages: Vec<LineSegment> = Vec::new();
    let mut anchor_pts: Vec<ScenePoint> = Vec::new();
    let mut anchor_txt: Vec<String> = Vec::new();
    for (on, g) in refs.iter().zip(REF_GAMUTS.iter()) {
        if !*on {
            continue;
        }
        match space {
            GamutSpace::Cielab => cages.extend(gamut_cage_lab(g.matrix, g.color)),
            GamutSpace::Bt2020Rgb => cages.extend(gamut_cage_rgb(g.matrix, g.color)),
        }
        anchor_pts.push(ScenePoint {
            position: cage_green_world(g.matrix, space),
            color: g.color,
        });
        anchor_txt.push(g.name.to_string());
    }

    let shell_segs = shell.map(|m| shell_segments(m, space)).unwrap_or_default();

    let point_count = points.len();
    let cage_segments = cages.len();
    let shell_segments = shell_segs.len();
    let cage_label_count = anchor_pts.len();
    GamutScene {
        points: PointsHandle::new(PointData { points }),
        cages: LinesHandle::new(LineData { segments: cages }),
        shell: LinesHandle::new(LineData {
            segments: shell_segs,
        }),
        cage_label_geo: PointsHandle::new(PointData { points: anchor_pts }),
        cage_labels: PointLabels::new(anchor_txt)
            .always()
            .placement(LabelPlacement::Above),
        point_count,
        cage_segments,
        shell_segments,
        cage_label_count,
    }
}

/// World position of a measured vertex's absolute XYZ (cd/m²) in the given
/// plot space, anchored the same way as the point cloud so the shell sits
/// in one frame with it: CIELAB normalizes by [`REFERENCE_WHITE_NITS`]
/// (203-nit white → L* = 100); BT.2020 RGB is the absolute nits straight
/// from the inverse matrix.
fn mesh_vertex_world(xyz_abs: [f64; 3], space: GamutSpace) -> Vec3 {
    match space {
        GamutSpace::Cielab => {
            let inv_ref = 1.0 / REFERENCE_WHITE_NITS;
            let xyz_n = [
                xyz_abs[0] * inv_ref,
                xyz_abs[1] * inv_ref,
                xyz_abs[2] * inv_ref,
            ];
            lab_to_world(xyz_to_lab(xyz_n, D65))
        }
        GamutSpace::Bt2020Rgb => {
            let bt = mat_mul(&XYZ_TO_BT2020, xyz_abs);
            Vec3::new(bt[0] as f32, bt[1] as f32, bt[2] as f32)
        }
    }
}

/// The measured gamut shell as patch quad-edge segments in `space`. Each
/// patch contributes its 4 boundary edges (corners are stored CCW); folded
/// patches are drawn in [`SHELL_FOLD_COLOR`] to flag the clamp.
fn shell_segments(mesh: &prism_ipc::GamutMesh, space: GamutSpace) -> Vec<LineSegment> {
    let world: Vec<Vec3> = mesh
        .vertices
        .iter()
        .map(|v| mesh_vertex_world(v.xyz, space))
        .collect();
    let mut out = Vec::new();
    for p in &mesh.patches {
        let color = match p.status {
            prism_ipc::GamutPatchStatus::Folded => SHELL_FOLD_COLOR,
            _ => SHELL_COLOR,
        };
        for k in 0..4 {
            let (Some(&start), Some(&end)) = (
                world.get(p.corners[k] as usize),
                world.get(p.corners[(k + 1) % 4] as usize),
            ) else {
                continue;
            };
            out.push(LineSegment { start, end, color });
        }
    }
    out
}

/// World position of a gamut's green primary `[0, 1, 0]` in the given
/// plot space — the anchor for its in-plot name label.
fn cage_green_world(rgb_to_xyz: &[[f64; 3]; 3], space: GamutSpace) -> Vec3 {
    const GREEN: [f64; 3] = [0.0, 1.0, 0.0];
    match space {
        GamutSpace::Cielab => lab_to_world(xyz_to_lab(mat_mul(rgb_to_xyz, GREEN), D65)),
        GamutSpace::Bt2020Rgb => {
            let bt = mat_mul(&XYZ_TO_BT2020, mat_mul(rgb_to_xyz, GREEN));
            Vec3::new(
                (bt[0] * REFERENCE_WHITE_NITS) as f32,
                (bt[1] * REFERENCE_WHITE_NITS) as f32,
                (bt[2] * REFERENCE_WHITE_NITS) as f32,
            )
        }
    }
}

/// Wireframe of an RGB cube's 12 edges mapped into Lab — the edges curve
/// because Lab is nonlinear in the source RGB. White corner `[1,1,1]`
/// lands at L* = 100 (D65), matching the normalized point cloud.
fn gamut_cage_lab(rgb_to_xyz: &[[f64; 3]; 3], color: [f32; 4]) -> Vec<LineSegment> {
    const STEPS: usize = 16;
    let world_of = |rgb: [f64; 3]| lab_to_world(xyz_to_lab(mat_mul(rgb_to_xyz, rgb), D65));

    let mut segs = Vec::new();
    for_each_cube_edge(|ca, cb| {
        let mut prev = world_of(ca);
        for i in 1..=STEPS {
            let t = i as f64 / STEPS as f64;
            let rgb = [
                ca[0] + (cb[0] - ca[0]) * t,
                ca[1] + (cb[1] - ca[1]) * t,
                ca[2] + (cb[2] - ca[2]) * t,
            ];
            let cur = world_of(rgb);
            segs.push(LineSegment {
                start: prev,
                end: cur,
                color,
            });
            prev = cur;
        }
    });
    segs
}

/// The gamut's RGB unit cube expressed in the buffer's BT.2020 RGB space,
/// scaled to [`REFERENCE_WHITE_NITS`]. The transform RGB→XYZ→BT.2020-RGB
/// is linear, so edges stay straight (no subdivision). For BT.2020 itself
/// this collapses to the axis-aligned `[0, ref]³` cube; narrower gamuts
/// (sRGB, P3) become skewed boxes nested inside it.
fn gamut_cage_rgb(rgb_to_xyz: &[[f64; 3]; 3], color: [f32; 4]) -> Vec<LineSegment> {
    let world_of = |rgb: [f64; 3]| {
        let bt = mat_mul(&XYZ_TO_BT2020, mat_mul(rgb_to_xyz, rgb));
        Vec3::new(
            (bt[0] * REFERENCE_WHITE_NITS) as f32,
            (bt[1] * REFERENCE_WHITE_NITS) as f32,
            (bt[2] * REFERENCE_WHITE_NITS) as f32,
        )
    };
    let mut segs = Vec::new();
    for_each_cube_edge(|ca, cb| {
        segs.push(LineSegment {
            start: world_of(ca),
            end: world_of(cb),
            color,
        });
    });
    segs
}

/// Call `f(corner_a, corner_b)` for each of the unit cube's 12 edges
/// (corner pairs differing in exactly one channel).
fn for_each_cube_edge(mut f: impl FnMut([f64; 3], [f64; 3])) {
    let corner = |bits: usize| {
        [
            (bits & 1) as f64,
            ((bits >> 1) & 1) as f64,
            ((bits >> 2) & 1) as f64,
        ]
    };
    for a in 0..8usize {
        for b in (a + 1)..8usize {
            if (a ^ b).count_ones() == 1 {
                f(corner(a), corner(b));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn d65_white_is_lab_100_neutral() {
        let lab = xyz_to_lab(D65, D65);
        assert!(approx(lab[0], 100.0, 1e-6), "L*={}", lab[0]);
        assert!(approx(lab[1], 0.0, 1e-6), "a*={}", lab[1]);
        assert!(approx(lab[2], 0.0, 1e-6), "b*={}", lab[2]);
    }

    #[test]
    fn bt2020_white_maps_to_d65_and_l100() {
        // BT.2020's white point IS D65, so RGB [1,1,1] → XYZ ≈ D65.
        let xyz = mat_mul(&BT2020_TO_XYZ, [1.0, 1.0, 1.0]);
        assert!(approx(xyz[0], D65[0], 2e-3), "X={}", xyz[0]);
        assert!(approx(xyz[1], D65[1], 2e-3), "Y={}", xyz[1]);
        assert!(approx(xyz[2], D65[2], 2e-3), "Z={}", xyz[2]);
        let lab = xyz_to_lab(xyz, D65);
        assert!(approx(lab[0], 100.0, 0.1), "L*={}", lab[0]);
        assert!(
            approx(lab[1], 0.0, 0.5) && approx(lab[2], 0.0, 0.5),
            "ab=({},{})",
            lab[1],
            lab[2]
        );
    }

    #[test]
    fn reference_white_sample_lands_at_l100() {
        // A BT.2020 sample at exactly REFERENCE_WHITE_NITS is L*≈100,
        // independent of any per-output white.
        let w = REFERENCE_WHITE_NITS as f32;
        let scene = build_gamut_scene(&[[w, w, w]], GamutSpace::Cielab, [true; N_REF_GAMUTS], None);
        let (data, _rev) = scene.points.snapshot();
        assert_eq!(scene.point_count, 1);
        assert!(approx(data.points[0].position.y as f64, 100.0, 0.1));
    }

    #[test]
    fn cloud_dedups_identical_samples() {
        let samples = vec![[100.0f32, 50.0, 25.0]; 1000];
        for space in [GamutSpace::Cielab, GamutSpace::Bt2020Rgb] {
            let scene = build_gamut_scene(&samples, space, [true; N_REF_GAMUTS], None);
            assert_eq!(
                scene.point_count, 1,
                "identical samples collapse, {space:?}"
            );
        }
    }

    #[test]
    fn raw_rgb_position_is_absolute_nits() {
        // BT.2020 RGB mode plots the buffer values directly, no scaling.
        let scene = build_gamut_scene(
            &[[120.0, 250.0, 60.0]],
            GamutSpace::Bt2020Rgb,
            [true; N_REF_GAMUTS],
            None,
        );
        let (data, _rev) = scene.points.snapshot();
        let p = data.points[0].position;
        assert!(approx(p.x as f64, 120.0, 1e-3));
        assert!(approx(p.y as f64, 250.0, 1e-3));
        assert!(approx(p.z as f64, 60.0, 1e-3));
    }

    #[test]
    fn enabled_set_selects_cages_and_labels() {
        let sample = [[100.0f32, 50.0, 25.0]];
        // All three on: three labelled anchors, named in REF_GAMUTS order.
        let all = build_gamut_scene(&sample, GamutSpace::Cielab, [true; N_REF_GAMUTS], None);
        assert_eq!(all.cage_label_geo.snapshot().0.points.len(), 3);
        assert_eq!(all.cage_labels.get(0), Some("sRGB"));
        assert_eq!(all.cage_labels.get(1), Some("Display P3"));
        assert_eq!(all.cage_labels.get(2), Some("Rec.2020"));
        let all_segs = all.cages.snapshot().0.segments.len();

        // Only Rec.2020 on: one anchor labelled "Rec.2020", a third of the cages.
        let one = build_gamut_scene(&sample, GamutSpace::Cielab, [false, false, true], None);
        assert_eq!(one.cage_label_geo.snapshot().0.points.len(), 1);
        assert_eq!(one.cage_labels.get(0), Some("Rec.2020"));
        assert_eq!(one.cages.snapshot().0.segments.len(), all_segs / 3);

        // None on: no cages, no labels (the cloud still stands on its own).
        let none = build_gamut_scene(&sample, GamutSpace::Cielab, [false; N_REF_GAMUTS], None);
        assert_eq!(none.cage_label_geo.snapshot().0.points.len(), 0);
        assert_eq!(none.cages.snapshot().0.segments.len(), 0);
    }

    #[test]
    fn shell_segments_render_per_patch_with_fold_flag() {
        use prism_ipc::{GamutMesh, GamutPatch, GamutPatchStatus, GamutVertex};

        let w = REFERENCE_WHITE_NITS;
        let vert = |xyz: [f64; 3]| GamutVertex {
            code_value: [0.0; 3],
            cmd_nits: [0.0; 3],
            xyz,
            lab: [0.0; 3],
            trustworthy: true,
        };
        // One flat quad + one folded quad → 8 edge segments, the folded
        // four in the hot color.
        let mesh = GamutMesh {
            white_xyz: [w, w, w],
            cmd_axis_max_nits: [w, w, w],
            vertices: vec![
                vert([10.0, 10.0, 10.0]),
                vert([40.0, 10.0, 5.0]),
                vert([35.0, 70.0, 12.0]),
                vert([18.0, 7.0, 95.0]),
            ],
            patches: vec![
                GamutPatch {
                    axis: 0,
                    value: 0.0,
                    corners: [0, 1, 2, 3],
                    status: GamutPatchStatus::Flat,
                },
                GamutPatch {
                    axis: 0,
                    value: 1.0,
                    corners: [0, 1, 2, 3],
                    status: GamutPatchStatus::Folded,
                },
            ],
        };

        for space in [GamutSpace::Cielab, GamutSpace::Bt2020Rgb] {
            let scene = build_gamut_scene(&[], space, [false; N_REF_GAMUTS], Some(&mesh));
            let (lines, _) = scene.shell.snapshot();
            assert_eq!(lines.segments.len(), 8, "two quads → 8 edges, {space:?}");
            let folded = lines
                .segments
                .iter()
                .filter(|s| s.color == SHELL_FOLD_COLOR)
                .count();
            assert_eq!(folded, 4, "the folded patch's 4 edges are hot, {space:?}");
            for s in &lines.segments {
                assert!(s.start.is_finite() && s.end.is_finite());
            }
        }

        // No mesh ⇒ empty shell.
        let bare = build_gamut_scene(&[], GamutSpace::Cielab, [false; N_REF_GAMUTS], None);
        assert_eq!(bare.shell.snapshot().0.segments.len(), 0);
    }
}
