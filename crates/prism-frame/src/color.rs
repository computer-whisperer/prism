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
