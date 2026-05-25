//! Color descriptions: primaries, transfer functions, mastering metadata.
//!
//! Every `Element` carries a `ColorDescription` describing its source content;
//! every `OutputState` carries one describing its scanout target. The renderer
//! is responsible for converting between them via decode + composite +
//! postprocess passes.

use std::num::NonZeroU32;

/// Color primaries: which RGB triangle the values live in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Primaries {
    /// Rec.709 / sRGB primaries.
    Srgb,
    /// DCI-P3 with D65 white (Display-P3).
    DisplayP3,
    /// Rec.2020 / BT.2020 primaries.
    Bt2020,
    /// Adobe RGB primaries.
    AdobeRgb,
    /// Custom CIE 1931 xy chromaticities. Each coord is fixed-point × 1e6 for
    /// hashability (e.g. D65 white = (313_000, 329_000)).
    Custom {
        red_xy: (i32, i32),
        green_xy: (i32, i32),
        blue_xy: (i32, i32),
        white_xy: (i32, i32),
    },
}

/// Optical-electronic transfer function applied to the values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransferFunction {
    /// Linear light. 1.0 = reference white.
    Linear,
    /// sRGB piecewise approximation of gamma 2.2.
    Srgb,
    /// BT.1886 (gamma 2.4-ish, broadcast reference).
    Bt1886,
    /// SMPTE ST 2084 (PQ). Normalized so 1.0 = 10000 nits.
    Pq,
    /// Hybrid Log-Gamma.
    Hlg,
    /// Pure gamma exponent. Stored × 100 (gamma 2.2 → 220).
    Gamma(GammaExponent),
}

/// Gamma exponent × 100 (e.g. 2.2 → 220) for hashability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GammaExponent(pub NonZeroU32);

impl GammaExponent {
    pub fn from_f32(g: f32) -> Option<Self> {
        let n = (g * 100.0).round() as u32;
        NonZeroU32::new(n).map(Self)
    }

    pub fn to_f32(self) -> f32 {
        self.0.get() as f32 / 100.0
    }
}

/// Mastering display metadata for HDR content. Drives tone-map decisions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MasteringInfo {
    /// Mastering display peak luminance in nits.
    pub display_max_nits: f32,
    /// Mastering display min luminance in nits.
    pub display_min_nits: f32,
    /// Maximum content light level in nits (CTA-861.G MaxCLL).
    pub max_cll: Option<f32>,
    /// Maximum frame-average light level in nits (CTA-861.G MaxFALL).
    pub max_fall: Option<f32>,
}

/// Complete description of how a buffer's pixel values map to light.
#[derive(Clone, Debug, PartialEq)]
pub struct ColorDescription {
    pub primaries: Primaries,
    pub transfer: TransferFunction,
    /// Reference white luminance in nits. SDR convention: 100. HDR clients
    /// may specify (typically 100–203).
    pub reference_luminance_nits: f32,
    /// HDR mastering metadata, if known. None for SDR or unknown.
    pub mastering: Option<MasteringInfo>,
}

impl ColorDescription {
    /// sRGB primaries, sRGB transfer, 100 nit ref white. The default for
    /// clients that don't speak `wp_color_management_v1`.
    pub const SRGB: Self = Self {
        primaries: Primaries::Srgb,
        transfer: TransferFunction::Srgb,
        reference_luminance_nits: 100.0,
        mastering: None,
    };

    /// BT.2020 primaries, PQ transfer, 100 nit ref white. The HDR scanout target.
    pub const BT2020_PQ: Self = Self {
        primaries: Primaries::Bt2020,
        transfer: TransferFunction::Pq,
        reference_luminance_nits: 100.0,
        mastering: None,
    };

    /// BT.2020 primaries, linear, 1.0 = 1 nit. The compositor's working
    /// intermediate color space — every element decodes into this, every
    /// output encodes from this.
    pub const BT2020_ABSOLUTE_NITS_LINEAR: Self = Self {
        primaries: Primaries::Bt2020,
        transfer: TransferFunction::Linear,
        reference_luminance_nits: 1.0,
        mastering: None,
    };
}

// ─── Primaries → BT.2020 conversion matrices ────────────────────────────────
//
// The compositor's working space is BT.2020 linear (see
// `BT2020_ABSOLUTE_NITS_LINEAR`). Every element must be converted from its
// own primaries into BT.2020 on the way in; this module builds the linear
// 3×3 that does it. Conversion is the standard RGB→XYZ→RGB chain with a
// Bradford chromatic-adaptation transform, so source white points other than
// D65 are mapped correctly (the transform degenerates to the identity when
// source and destination whites match, which is the common case).

/// Row-major 3×3 matrix. `m[row][col]`; `m * v` is `out[i] = Σ_k m[i][k]·v[k]`.
pub type Mat3 = [[f32; 3]; 3];

/// CIE 1931 xy chromaticities for an RGB primary set plus its white point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chromaticities {
    pub red: (f32, f32),
    pub green: (f32, f32),
    pub blue: (f32, f32),
    pub white: (f32, f32),
}

impl Chromaticities {
    /// Rec.709 / sRGB primaries, D65 white.
    pub const BT709: Self = Self {
        red: (0.640, 0.330),
        green: (0.300, 0.600),
        blue: (0.150, 0.060),
        white: (0.3127, 0.3290),
    };
    /// Rec.2020 primaries, D65 white.
    pub const BT2020: Self = Self {
        red: (0.708, 0.292),
        green: (0.170, 0.797),
        blue: (0.131, 0.046),
        white: (0.3127, 0.3290),
    };
    /// DCI-P3 with D65 white (Display-P3).
    pub const DISPLAY_P3: Self = Self {
        red: (0.680, 0.320),
        green: (0.265, 0.690),
        blue: (0.150, 0.060),
        white: (0.3127, 0.3290),
    };
    /// Adobe RGB (1998), D65 white.
    pub const ADOBE_RGB: Self = Self {
        red: (0.640, 0.330),
        green: (0.210, 0.710),
        blue: (0.150, 0.060),
        white: (0.3127, 0.3290),
    };
}

impl Primaries {
    /// The CIE xy chromaticities this primary set denotes. `Custom` carries
    /// its coordinates as fixed-point ×1e6 (see the enum); the named sets use
    /// the standard values with a D65 white.
    pub fn chromaticities(self) -> Chromaticities {
        match self {
            Primaries::Srgb => Chromaticities::BT709,
            Primaries::DisplayP3 => Chromaticities::DISPLAY_P3,
            Primaries::Bt2020 => Chromaticities::BT2020,
            Primaries::AdobeRgb => Chromaticities::ADOBE_RGB,
            Primaries::Custom {
                red_xy,
                green_xy,
                blue_xy,
                white_xy,
            } => {
                let f = |v: i32| v as f32 / 1_000_000.0;
                Chromaticities {
                    red: (f(red_xy.0), f(red_xy.1)),
                    green: (f(green_xy.0), f(green_xy.1)),
                    blue: (f(blue_xy.0), f(blue_xy.1)),
                    white: (f(white_xy.0), f(white_xy.1)),
                }
            }
        }
    }
}

fn mat3_mul(a: Mat3, b: Mat3) -> Mat3 {
    let mut out = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

fn mat3_vec(m: Mat3, v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Inverse of a row-major 3×3 via cofactor expansion. The matrices we invert
/// (normalized primary matrices, the Bradford cone matrix) are always
/// non-singular for valid chromaticities; `debug_assert` guards against a
/// degenerate primary set slipping through.
fn mat3_inverse(m: Mat3) -> Mat3 {
    let [[a, b, c], [d, e, f], [g, h, i]] = m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    debug_assert!(det.abs() > 1e-12, "singular 3×3 in color matrix inverse");
    let inv_det = 1.0 / det;
    [
        [
            (e * i - f * h) * inv_det,
            (c * h - b * i) * inv_det,
            (b * f - c * e) * inv_det,
        ],
        [
            (f * g - d * i) * inv_det,
            (a * i - c * g) * inv_det,
            (c * d - a * f) * inv_det,
        ],
        [
            (d * h - e * g) * inv_det,
            (b * g - a * h) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ]
}

/// XYZ of an xy chromaticity at unit luminance (Y = 1).
fn xy_to_xyz((x, y): (f32, f32)) -> [f32; 3] {
    [x / y, 1.0, (1.0 - x - y) / y]
}

/// Normalized primary matrix: source-RGB → CIE XYZ, scaled so RGB (1,1,1)
/// maps to the set's white point XYZ.
fn rgb_to_xyz(c: &Chromaticities) -> Mat3 {
    let r = xy_to_xyz(c.red);
    let g = xy_to_xyz(c.green);
    let b = xy_to_xyz(c.blue);
    // Columns are the primaries' XYZ.
    let m = [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]];
    // Per-primary scale so the column sum equals the white point.
    let s = mat3_vec(mat3_inverse(m), xy_to_xyz(c.white));
    // result = M · diag(s): scale column j by s[j].
    [
        [m[0][0] * s[0], m[0][1] * s[1], m[0][2] * s[2]],
        [m[1][0] * s[0], m[1][1] * s[1], m[1][2] * s[2]],
        [m[2][0] * s[0], m[2][1] * s[1], m[2][2] * s[2]],
    ]
}

/// Bradford cone-response matrix (XYZ → LMS-ish cone space).
const BRADFORD: Mat3 = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// Bradford chromatic-adaptation transform mapping XYZ referred to `src_white`
/// onto XYZ referred to `dst_white`. Identity when the whites coincide.
fn bradford_adapt(src_white: [f32; 3], dst_white: [f32; 3]) -> Mat3 {
    let s = mat3_vec(BRADFORD, src_white);
    let d = mat3_vec(BRADFORD, dst_white);
    let scale = [
        [d[0] / s[0], 0.0, 0.0],
        [0.0, d[1] / s[1], 0.0],
        [0.0, 0.0, d[2] / s[2]],
    ];
    mat3_mul(mat3_inverse(BRADFORD), mat3_mul(scale, BRADFORD))
}

/// Linear-light matrix converting `src`-primaries RGB into BT.2020 RGB,
/// Bradford-adapted to BT.2020's D65 white. Row-major; apply as `m * rgb`.
/// White is preserved exactly when `src.white == BT2020.white`.
pub fn primaries_to_bt2020(src: &Chromaticities) -> Mat3 {
    let m_src = rgb_to_xyz(src);
    let m_dst = rgb_to_xyz(&Chromaticities::BT2020);
    let cat = bradford_adapt(
        xy_to_xyz(src.white),
        xy_to_xyz(Chromaticities::BT2020.white),
    );
    mat3_mul(mat3_inverse(m_dst), mat3_mul(cat, m_src))
}

/// The sRGB/BT.709 → BT.2020 matrix — the conversion for legacy clients that
/// don't speak `wp_color_management_v1` (their content is sRGB by convention)
/// and for `srgb_to_bt2020_nits`. Cached: the derivation is identical every
/// call.
pub fn srgb_to_bt2020_matrix() -> Mat3 {
    use std::sync::LazyLock;
    static M: LazyLock<Mat3> = LazyLock::new(|| primaries_to_bt2020(&Chromaticities::BT709));
    *M
}

#[cfg(test)]
mod primaries_tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) {
        assert!((a - b).abs() < eps, "{a} vs {b} (eps {eps})");
    }
    fn approx_mat(m: Mat3, e: Mat3, eps: f32) {
        for i in 0..3 {
            for j in 0..3 {
                approx(m[i][j], e[i][j], eps);
            }
        }
    }
    const I: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    #[test]
    fn bt2020_source_is_identity() {
        approx_mat(primaries_to_bt2020(&Chromaticities::BT2020), I, 1e-4);
    }

    #[test]
    fn white_maps_to_white_for_named_sets() {
        for c in [
            Chromaticities::BT709,
            Chromaticities::DISPLAY_P3,
            Chromaticities::ADOBE_RGB,
            Chromaticities::BT2020,
        ] {
            let w = mat3_vec(primaries_to_bt2020(&c), [1.0, 1.0, 1.0]);
            approx(w[0], 1.0, 2e-4);
            approx(w[1], 1.0, 2e-4);
            approx(w[2], 1.0, 2e-4);
        }
    }

    #[test]
    fn bt709_matches_published_bt2087_matrix() {
        // ITU-R BT.2087-0 BT.709 → BT.2020 conversion coefficients.
        let expect = [
            [0.6274, 0.3293, 0.0433],
            [0.0691, 0.9195, 0.0114],
            [0.0164, 0.0880, 0.8956],
        ];
        approx_mat(primaries_to_bt2020(&Chromaticities::BT709), expect, 2e-3);
    }

    #[test]
    fn inverse_roundtrips() {
        let m = rgb_to_xyz(&Chromaticities::DISPLAY_P3);
        approx_mat(mat3_mul(m, mat3_inverse(m)), I, 1e-4);
    }

    #[test]
    fn non_d65_white_is_adapted_to_neutral() {
        // sRGB primaries but a D50 white: Bradford must still carry (1,1,1)
        // to BT.2020 (1,1,1) rather than tinting it.
        let d50 = Chromaticities {
            white: (0.34567, 0.35850),
            ..Chromaticities::BT709
        };
        let w = mat3_vec(primaries_to_bt2020(&d50), [1.0, 1.0, 1.0]);
        approx(w[0], 1.0, 2e-3);
        approx(w[1], 1.0, 2e-3);
        approx(w[2], 1.0, 2e-3);
    }
}
