//! Push-constant layout for the synthesized encode fragment shader.
//!
//! One fixed Rust struct mirrors a single SPIR-V `Push` block. Different
//! [`EncodeConfig`](super::EncodeConfig)s produce different shader code but
//! all read from the same push-constant layout — fragments use whichever
//! slots they need, and unused slots are dead code on the GPU side.
//!
//! Layout aims for std430 + Vulkan push-constants compatibility:
//!   - `mat4 cal_matrix` at offset 0   (64 B, MatrixStride 16, ColMajor)
//!   - `vec4 fir_kernel_r` at offset 64 (xyz = 3-tap weights, w unused)
//!   - `vec4 fir_kernel_g` at offset 80
//!   - `vec4 fir_kernel_b` at offset 96
//!   - `float sdr_white_nits` at offset 112
//!   - `float target_peak_nits` at offset 116
//!   - `float dither_strength` at offset 120
//!   - `float _pad` at offset 124
//! Total 128 bytes — at the Vulkan minimum push-constant size, intentionally,
//! so any conformant driver accepts this layout.

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
    /// Subpixel FIR weights, per-channel. `xyz` = three horizontal taps
    /// (left, center, right). `w` is padding. Used by
    /// `EncodeFragment::SubpixelFir3Horizontal`.
    pub fir_kernel_r: [f32; 4],
    pub fir_kernel_g: [f32; 4],
    pub fir_kernel_b: [f32; 4],
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
    /// Identity calibration + 80-nit SDR white, no dither, no FIR.
    /// Matches the previous hard-coded `EncodePush::sdr_identity` defaults.
    pub fn sdr_identity() -> Self {
        Self {
            cal_matrix: mat4_identity(),
            fir_kernel_r: identity_fir_tap(),
            fir_kernel_g: identity_fir_tap(),
            fir_kernel_b: identity_fir_tap(),
            sdr_white_nits: 80.0,
            target_peak_nits: 80.0,
            dither_strength: 0.0,
            _pad: 0.0,
        }
    }

    /// Identity calibration + 10000-nit PQ peak, no dither, no FIR.
    pub fn pq_identity() -> Self {
        Self {
            cal_matrix: mat4_identity(),
            fir_kernel_r: identity_fir_tap(),
            fir_kernel_g: identity_fir_tap(),
            fir_kernel_b: identity_fir_tap(),
            sdr_white_nits: 80.0,
            target_peak_nits: 10000.0,
            dither_strength: 0.0,
            _pad: 0.0,
        }
    }
}

fn mat4_identity() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// 3-tap FIR pass-through: center weight = 1, neighbors = 0.
fn identity_fir_tap() -> [f32; 4] {
    [0.0, 1.0, 0.0, 0.0]
}

/// Push-constant byte size. Must equal the SPIR-V Push struct size.
pub const PUSH_CONSTANTS_SIZE: u32 = std::mem::size_of::<EncodePushSynth>() as u32;

// Byte offsets — used by the SPIR-V builder for OpMemberDecorate Offset.
pub const OFFSET_CAL_MATRIX: u32 = 0;
pub const OFFSET_FIR_KERNEL_R: u32 = 64;
pub const OFFSET_FIR_KERNEL_G: u32 = 80;
pub const OFFSET_FIR_KERNEL_B: u32 = 96;
pub const OFFSET_SDR_WHITE_NITS: u32 = 112;
pub const OFFSET_TARGET_PEAK_NITS: u32 = 116;
pub const OFFSET_DITHER_STRENGTH: u32 = 120;
pub const OFFSET_PAD: u32 = 124;

// Member indices within the SPIR-V struct (same order as Rust struct).
pub const MEMBER_CAL_MATRIX: u32 = 0;
pub const MEMBER_FIR_KERNEL_R: u32 = 1;
pub const MEMBER_FIR_KERNEL_G: u32 = 2;
pub const MEMBER_FIR_KERNEL_B: u32 = 3;
pub const MEMBER_SDR_WHITE_NITS: u32 = 4;
pub const MEMBER_TARGET_PEAK_NITS: u32 = 5;
pub const MEMBER_DITHER_STRENGTH: u32 = 6;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_constant_size_at_minimum_limit() {
        assert_eq!(PUSH_CONSTANTS_SIZE, 128);
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
            (&zero.fir_kernel_r as *const _ as usize - base) as u32,
            OFFSET_FIR_KERNEL_R
        );
        assert_eq!(
            (&zero.fir_kernel_g as *const _ as usize - base) as u32,
            OFFSET_FIR_KERNEL_G
        );
        assert_eq!(
            (&zero.fir_kernel_b as *const _ as usize - base) as u32,
            OFFSET_FIR_KERNEL_B
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
