//! Local color science for the 3D gamut view: BT.2020/sRGB/P3 → XYZ →
//! CIELAB, and gamut-cage wireframes. Reimplemented here (rather than
//! pulling `tristim-color`, which would force a bump of the pinned
//! tristim git rev) — the needed subset is small and standard.
//!
//! The cloud and cages live in the same Lab frame: world X = a*, Y = L*
//! (up), Z = b*, all referenced to D65 (BT.2020's white). Sample colors
//! are normalized so the output's SDR-reference white lands at L* = 100,
//! matching the cube-corner white of the cages.

use std::collections::HashSet;

use damascene_core::scene::glam::Vec3;
use damascene_core::scene::{LineData, LinesHandle};
use damascene_core::scene::{LineSegment, PointData, PointsHandle, ScenePoint};

use crate::common::srgb_oetf;

/// D65 reference white in XYZ (Y normalized to 1).
const D65: [f64; 3] = [0.950_47, 1.0, 1.088_83];

/// BT.2020 linear RGB → XYZ (D65). Also the sample-cloud source matrix.
const BT2020_TO_XYZ: [[f64; 3]; 3] = [
    [0.636_958_0, 0.144_616_9, 0.168_881_0],
    [0.262_700_2, 0.677_998_1, 0.059_301_7],
    [0.000_000_0, 0.028_072_7, 1.060_985_1],
];

/// sRGB / BT.709 linear RGB → XYZ (D65).
const SRGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.412_456_4, 0.357_576_1, 0.180_437_5],
    [0.212_672_9, 0.715_152_2, 0.072_175_0],
    [0.019_333_9, 0.119_192_0, 0.950_304_1],
];

/// Display-P3 linear RGB → XYZ (D65).
const P3_TO_XYZ: [[f64; 3]; 3] = [
    [0.486_570_9, 0.265_667_7, 0.198_217_3],
    [0.228_974_6, 0.691_738_5, 0.079_286_9],
    [0.000_000_0, 0.045_113_4, 1.043_944_4],
];

/// A built gamut scene: the measured-color point cloud and the reference
/// gamut-cage wireframes. Both are damascene geometry handles (Arc'd,
/// versioned), built once per capture and cloned into a fresh `SceneSpec`
/// each frame.
pub struct GamutScene {
    pub points: PointsHandle,
    pub cages: LinesHandle,
    /// Distinct point count after Lab dedup (for the status line).
    pub point_count: usize,
}

/// Reference gamuts overlaid as cage wireframes.
const REF_GAMUTS: [(&str, &[[f64; 3]; 3], [f32; 4]); 3] = [
    ("sRGB", &SRGB_TO_XYZ, [0.42, 0.60, 1.0, 0.5]),
    ("Display P3", &P3_TO_XYZ, [0.36, 0.85, 0.52, 0.5]),
    ("Rec.2020", &BT2020_TO_XYZ, [1.0, 0.70, 0.30, 0.5]),
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

/// Build the point cloud + cages from white-normalizable BT.2020
/// absolute-nits samples. Points are deduplicated on a Lab voxel grid so
/// large flat regions collapse to one mark instead of thousands.
pub fn build_gamut_scene(samples: &[[f32; 3]], white_nits: f64) -> GamutScene {
    let scale = if white_nits > 0.0 {
        1.0 / white_nits
    } else {
        1.0
    };

    // Lab voxel dedup: quantize to `CELL`-sized cells, keep one per cell.
    const CELL: f64 = 2.0;
    let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
    let mut points: Vec<ScenePoint> = Vec::new();
    for s in samples {
        let bt2020 = [
            s[0] as f64 * scale,
            s[1] as f64 * scale,
            s[2] as f64 * scale,
        ];
        // Clamp tiny negatives from sensor/encode noise before Lab.
        let xyz = mat_mul(
            &BT2020_TO_XYZ,
            [bt2020[0].max(0.0), bt2020[1].max(0.0), bt2020[2].max(0.0)],
        );
        let lab = xyz_to_lab(xyz, D65);
        let cell = (
            (lab[1] / CELL).round() as i32,
            (lab[0] / CELL).round() as i32,
            (lab[2] / CELL).round() as i32,
        );
        if seen.insert(cell) {
            points.push(ScenePoint {
                position: lab_to_world(lab),
                color: point_color(bt2020),
            });
        }
    }

    let mut cages: Vec<LineSegment> = Vec::new();
    for (_name, matrix, color) in REF_GAMUTS {
        cages.extend(gamut_cage(matrix, color));
    }

    let point_count = points.len();
    GamutScene {
        points: PointsHandle::new(PointData { points }),
        cages: LinesHandle::new(LineData { segments: cages }),
        point_count,
    }
}

/// Wireframe of an RGB cube's 12 edges mapped into Lab — the edges curve
/// because Lab is nonlinear in the source RGB. White corner `[1,1,1]`
/// lands at L* = 100 (D65), matching the normalized point cloud.
fn gamut_cage(rgb_to_xyz: &[[f64; 3]; 3], color: [f32; 4]) -> Vec<LineSegment> {
    const STEPS: usize = 16;
    let corner = |bits: usize| {
        [
            (bits & 1) as f64,
            ((bits >> 1) & 1) as f64,
            ((bits >> 2) & 1) as f64,
        ]
    };
    let world_of = |rgb: [f64; 3]| lab_to_world(xyz_to_lab(mat_mul(rgb_to_xyz, rgb), D65));

    let mut segs = Vec::new();
    for a in 0..8usize {
        for b in (a + 1)..8usize {
            // Cube edge = corners differing in exactly one bit.
            if (a ^ b).count_ones() != 1 {
                continue;
            }
            let (ca, cb) = (corner(a), corner(b));
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
        }
    }
    segs
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
        // A BT.2020 sample at exactly the reference white nits is L*≈100.
        let scene = build_gamut_scene(&[[203.0, 203.0, 203.0]], 203.0);
        let (data, _rev) = scene.points.snapshot();
        assert_eq!(scene.point_count, 1);
        assert!(approx(data.points[0].position.y as f64, 100.0, 0.1));
    }

    #[test]
    fn cloud_dedups_identical_samples() {
        let samples = vec![[100.0f32, 50.0, 25.0]; 1000];
        let scene = build_gamut_scene(&samples, 203.0);
        assert_eq!(
            scene.point_count, 1,
            "identical samples collapse to one Lab cell"
        );
    }
}
