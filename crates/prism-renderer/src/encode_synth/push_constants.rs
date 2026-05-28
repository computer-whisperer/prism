//! Push-constant layout for the synthesized encode fragment shader.
//!
//! One fixed Rust struct mirrors a single SPIR-V `Push` block. Different
//! [`EncodeConfig`](super::EncodeConfig)s produce different shader code but
//! all read from the same push-constant layout — fragments use whichever
//! slots they need, and unused slots are dead code on the GPU side.
//!
//! Layout aims for std430 + Vulkan push-constants compatibility:
//!   - `mat4 cal_matrix` at offset 0   (64 B, MatrixStride 16, ColMajor)
//!   - `vec4 response_gain` at offset 64 (per-channel response gain,
//!     used by `EncodeFragment::PerChannelResponseGainGamma`. `.w`
//!     unused but reserved.)
//!   - `vec4 response_gamma` at offset 80 (per-channel response gamma
//!     exponent. `.w` unused but reserved.)
//!   - `float sdr_white_nits` at offset 96
//!   - `float target_peak_nits` at offset 100
//!   - `float dither_strength` at offset 104
//!   - `float _pad` at offset 108
//!
//! Total 112 bytes — comfortably below the 128-byte Vulkan minimum
//! push-constant size, so any conformant driver accepts this layout.
//!
//! A per-channel `lut_input_max_nits` vec4 used to sit at offset 96; v4
//! of the LUT pipeline removed it. The bake now projects out-of-gamut
//! and below-floor requests onto the measured reachable surface, so the
//! shader no longer needs a pre-LUT axis-aligned box clamp. The PQ
//! shaper still implicitly bounds inputs to `[0, 10000]` cd/m² via the
//! sampler's clamp-to-edge addressing.

use bytemuck::{Pod, Zeroable};

/// Push-constant block shared by every synthesized encode shader. Field
/// ordering and offsets MUST match the SPIR-V member layout in
/// `builder.rs::declare_push_block`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct EncodePushSynth {
    /// Per-output calibration matrix in the source color space (BT.2020
    /// absolute nits). 4×4 storage with the upper-left 3×3 used; the rest
    /// is zero-padded for std430 alignment.
    pub cal_matrix: [f32; 16],
    /// Per-channel response gain — multiplier the panel applies to a
    /// commanded value. Used by `EncodeFragment::PerChannelResponseGainGamma`
    /// to invert the panel's measured response (`commanded =
    /// (target/gain)^(1/gamma)`). Identity = `[1.0, 1.0, 1.0, 0.0]`.
    /// `.w` is unused/reserved.
    pub response_gain: [f32; 4],
    /// Per-channel response gamma — exponent. Identity = `[1.0, 1.0, 1.0, 0.0]`.
    pub response_gamma: [f32; 4],
    /// For SDR encode: how many nits is the input value `1.0`.
    pub sdr_white_nits: f32,
    /// For PQ encode and linear scanout: clip / normalize ceiling.
    pub target_peak_nits: f32,
    /// Per-pixel dither magnitude (units: encoded code values). Set 0 to
    /// disable visually even if the dither fragment is in the chain.
    pub dither_strength: f32,
    pub _pad: f32,
}

impl EncodePushSynth {
    /// Identity calibration + 80-nit SDR white, no dither, identity response.
    pub fn sdr_identity() -> Self {
        Self {
            cal_matrix: mat4_identity(),
            response_gain: identity_response_vec(),
            response_gamma: identity_response_vec(),
            sdr_white_nits: 80.0,
            target_peak_nits: 80.0,
            dither_strength: 0.0,
            _pad: 0.0,
        }
    }

    /// Identity calibration + 10000-nit PQ peak, no dither, identity response.
    pub fn pq_identity() -> Self {
        Self {
            cal_matrix: mat4_identity(),
            response_gain: identity_response_vec(),
            response_gamma: identity_response_vec(),
            sdr_white_nits: 80.0,
            target_peak_nits: 10000.0,
            dither_strength: 0.0,
            _pad: 0.0,
        }
    }

    /// Set the per-channel response correction values. `gain` and
    /// `gamma` are arrays of length 3 (R, G, B). The fragment shader
    /// computes `compensated = (target / gain)^(1/gamma)` per channel
    /// before the OETF, so the panel — which emits
    /// `gain * commanded^gamma` — ends up emitting the target value.
    pub fn set_response_gain_gamma(&mut self, gain: [f32; 3], gamma: [f32; 3]) {
        self.response_gain = [gain[0], gain[1], gain[2], 0.0];
        self.response_gamma = [gamma[0], gamma[1], gamma[2], 0.0];
    }

    /// Set the 3×3 calibration matrix from row-major rows. The shader
    /// applies `panel_rgb = M * bt2020_rgb` — i.e., row R of `m` is the
    /// coefficients `[in_R, in_G, in_B]` that contribute to output R.
    /// Stored as a mat4 in column-major order with the 4th row/column
    /// zeroed (the shader uses `mat3(cal_matrix)`).
    pub fn set_ctm(&mut self, m: [[f32; 3]; 3]) {
        self.cal_matrix = [
            m[0][0], m[1][0], m[2][0], 0.0, // col 0
            m[0][1], m[1][1], m[2][1], 0.0, // col 1
            m[0][2], m[1][2], m[2][2], 0.0, // col 2
            0.0, 0.0, 0.0, 1.0, // col 3 (unused; identity for safety)
        ];
    }
}

fn mat4_identity() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

/// Identity response: gain=1, gamma=1, .w slot unused.
fn identity_response_vec() -> [f32; 4] {
    [1.0, 1.0, 1.0, 0.0]
}

/// Push-constant byte size. Must equal the SPIR-V Push struct size.
pub const PUSH_CONSTANTS_SIZE: u32 = std::mem::size_of::<EncodePushSynth>() as u32;

// Byte offsets — used by the SPIR-V builder for OpMemberDecorate Offset.
pub const OFFSET_CAL_MATRIX: u32 = 0;
pub const OFFSET_RESPONSE_GAIN: u32 = 64;
pub const OFFSET_RESPONSE_GAMMA: u32 = 80;
pub const OFFSET_SDR_WHITE_NITS: u32 = 96;
pub const OFFSET_TARGET_PEAK_NITS: u32 = 100;
pub const OFFSET_DITHER_STRENGTH: u32 = 104;
pub const OFFSET_PAD: u32 = 108;

// Member indices within the SPIR-V struct (same order as Rust struct).
pub const MEMBER_CAL_MATRIX: u32 = 0;
pub const MEMBER_RESPONSE_GAIN: u32 = 1;
pub const MEMBER_RESPONSE_GAMMA: u32 = 2;
pub const MEMBER_SDR_WHITE_NITS: u32 = 3;
pub const MEMBER_TARGET_PEAK_NITS: u32 = 4;
pub const MEMBER_DITHER_STRENGTH: u32 = 5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_constant_size_under_minimum_limit() {
        // 112 < 128 (Vulkan minimum). Plenty of headroom if future
        // fragments need more slots — adjust _pad to align.
        assert_eq!(PUSH_CONSTANTS_SIZE, 112);
    }

    #[test]
    fn offsets_match_struct_layout() {
        let zero = EncodePushSynth::sdr_identity();
        let base = &zero as *const _ as usize;
        assert_eq!(
            (&zero.cal_matrix as *const _ as usize - base) as u32,
            OFFSET_CAL_MATRIX
        );
        assert_eq!(
            (&zero.response_gain as *const _ as usize - base) as u32,
            OFFSET_RESPONSE_GAIN
        );
        assert_eq!(
            (&zero.response_gamma as *const _ as usize - base) as u32,
            OFFSET_RESPONSE_GAMMA
        );
        assert_eq!(
            (&zero.sdr_white_nits as *const _ as usize - base) as u32,
            OFFSET_SDR_WHITE_NITS
        );
        assert_eq!(
            (&zero.target_peak_nits as *const _ as usize - base) as u32,
            OFFSET_TARGET_PEAK_NITS
        );
        assert_eq!(
            (&zero.dither_strength as *const _ as usize - base) as u32,
            OFFSET_DITHER_STRENGTH
        );
    }
}
