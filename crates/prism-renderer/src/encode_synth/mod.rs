//! Per-output encode fragment shader synthesis (rspirv).
//!
//! The encode pass is *the* per-output customization point. Different
//! displays want different effects in different orders:
//!   - Standard SDR: `[CalibrationMatrix, OutputTransferSrgb]`
//!   - HDR PQ:       `[CalibrationMatrix, OutputTransferPq]`
//!   - QD-OLED text: `[CalibrationMatrix, OutputTransferSrgb, SubpixelFir3Horizontal]`
//!   - 8-bit panel:  `[..., InterleavedGradientNoiseDither]`
//!
//! Rather than a single mega-shader with runtime branches, we synthesize
//! the fragment shader per output from an `EncodeConfig`. SPIR-V emission
//! goes through `rspirv::dr::Builder`. The vertex shader is unchanged
//! (full-screen triangle) and stays statically compiled from GLSL.

pub mod builder;
pub mod fragment;
pub mod push_constants;

pub use push_constants::{EncodePushSynth, PUSH_CONSTANTS_SIZE};

use crate::error::{RendererError, Result};

/// Ordered list of effects the encode shader applies, in chain order.
#[derive(Clone, Debug)]
pub struct EncodeConfig {
    pub fragments: Vec<EncodeFragment>,
}

/// One step in the encode chain. Each variant maps to a `fragment::emit_*`
/// function that produces a block of SPIR-V instructions threading a vec3
/// color through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodeFragment {
    /// `out = mat3(push.cal_matrix) * in`. Identity by default.
    CalibrationMatrix,
    /// `out = srgb_oetf(clamp(in / sdr_white_nits, 0, 1))`.
    OutputTransferSrgb,
    /// `out = pq_oetf(clamp(in, 0, target_peak_nits))`.
    OutputTransferPq,
    /// `out = in / max(target_peak_nits, 1)`. For fp16 scanout.
    OutputTransferLinear,
    /// Per-channel response correction: `out = (in / gain)^(1/gamma)`.
    /// Inverts the panel's measured `emitted = gain * commanded^gamma`
    /// curve. Identity (gain=1, gamma=1) is a no-op. Stage before any
    /// of the OutputTransfer* fragments so the correction is in
    /// linear-nits domain.
    PerChannelResponseGainGamma,
    /// 3D color LUT lookup with a PQ shaper on input. The per-channel
    /// PQ OETF maps incoming linear BT.2020 nits into the LUT's `[0, 1]`
    /// coordinate space (allocating more precision near zero, where the
    /// eye is sensitive), then a trilinear sample returns panel-native
    /// commanded nits. Replaces the [`CalibrationMatrix`] +
    /// [`PerChannelResponseGainGamma`] pair: a single LUT captures both
    /// gamut correction AND per-channel response without assuming either
    /// is a closed-form function. Output stays in linear-nits domain so
    /// a downstream `OutputTransfer*` fragment encodes for scanout.
    Lut3d,
    /// Per-channel 3-tap horizontal FIR filter for non-stripe subpixel
    /// layouts (QD-OLED triangular). Requires multi-sample handling that
    /// the synthesizer doesn't implement yet — emitting this currently
    /// returns `MissingFeature`. See progress doc for the multi-sample
    /// fan-out plan.
    SubpixelFir3Horizontal,
    /// Per-pixel ordered dither via interleaved-gradient noise. Hides
    /// 8-bit quantization banding without needing a noise texture.
    /// Not implemented in the first synth cut.
    InterleavedGradientNoiseDither,
}

impl EncodeConfig {
    /// Today's SDR default: identity calibration + sRGB OETF. Matches the
    /// pre-synthesis hard-coded encode shader.
    pub fn default_srgb() -> Self {
        Self {
            fragments: vec![
                EncodeFragment::CalibrationMatrix,
                EncodeFragment::OutputTransferSrgb,
            ],
        }
    }

    /// HDR PQ default: identity calibration + identity response
    /// correction + PQ OETF. The response correction stage is always
    /// included for HDR outputs (even when no per-output calibration
    /// is configured) so runtime IPC can flip on a panel-specific
    /// curve without rebuilding the encode pipeline. Identity gain
    /// and gamma values make the stage a no-op; cost is one extra
    /// `pow` per pixel that the shader compiler can sometimes
    /// optimize away.
    pub fn default_pq() -> Self {
        Self {
            fragments: vec![
                EncodeFragment::CalibrationMatrix,
                EncodeFragment::PerChannelResponseGainGamma,
                EncodeFragment::OutputTransferPq,
            ],
        }
    }

    /// fp16-scanout default: identity calibration + linear pass-through.
    pub fn default_linear() -> Self {
        Self {
            fragments: vec![
                EncodeFragment::CalibrationMatrix,
                EncodeFragment::OutputTransferLinear,
            ],
        }
    }

    /// True if any fragment in this chain references the per-output 3D LUT
    /// (binding 1). The shader synthesizer + pipeline layout both honor
    /// this — when false, binding 1 is never declared and the pipeline
    /// doesn't expect an LUT image at draw time.
    pub fn uses_lut3d(&self) -> bool {
        self.fragments.iter().any(|f| matches!(f, EncodeFragment::Lut3d))
    }
}

/// Synthesize a SPIR-V fragment shader for the given `EncodeConfig`.
///
/// Returns the SPIR-V words (u32 sequence) suitable for `vkCreateShaderModule`.
pub fn synthesize_fragment_shader(config: &EncodeConfig) -> Result<Vec<u32>> {
    let mut ctx = builder::ShaderCtx::new(config);
    // Every encode chain starts by sampling the intermediate texture.
    let mut color = fragment::emit_sample_intermediate(&mut ctx);

    for frag in &config.fragments {
        color = match frag {
            EncodeFragment::CalibrationMatrix => fragment::emit_calibration_matrix(&mut ctx, color),
            EncodeFragment::OutputTransferSrgb => {
                fragment::emit_output_transfer_srgb(&mut ctx, color)
            }
            EncodeFragment::OutputTransferPq => fragment::emit_output_transfer_pq(&mut ctx, color),
            EncodeFragment::OutputTransferLinear => {
                fragment::emit_output_transfer_linear(&mut ctx, color)
            }
            EncodeFragment::PerChannelResponseGainGamma => {
                fragment::emit_per_channel_response_gain_gamma(&mut ctx, color)
            }
            EncodeFragment::Lut3d => fragment::emit_lut3d(&mut ctx, color),
            EncodeFragment::SubpixelFir3Horizontal => {
                return Err(RendererError::MissingFeature(
                    "SubpixelFir3Horizontal: multi-sample synthesis not implemented yet",
                ));
            }
            EncodeFragment::InterleavedGradientNoiseDither => {
                return Err(RendererError::MissingFeature(
                    "InterleavedGradientNoiseDither: not implemented in first synthesis cut",
                ));
            }
        };
    }

    Ok(ctx.finish(color))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `uses_lut3d` reports the LUT presence honestly across the built-in
    /// configs. Drives whether `ShaderCtx` declares binding 1 and whether
    /// the pipeline layout will need to provide an LUT descriptor.
    #[test]
    fn uses_lut3d_matches_chain_contents() {
        assert!(!EncodeConfig::default_srgb().uses_lut3d());
        assert!(!EncodeConfig::default_pq().uses_lut3d());
        assert!(!EncodeConfig::default_linear().uses_lut3d());
        let with_lut = EncodeConfig {
            fragments: vec![EncodeFragment::Lut3d, EncodeFragment::OutputTransferPq],
        };
        assert!(with_lut.uses_lut3d());
    }

    /// A Lut3d-bearing chain emits a non-empty SPIR-V module without
    /// panicking. Real validation needs vulkan or spirv-val (not in our
    /// dep tree); smoke-test that emission gets through end-to-end.
    #[test]
    fn lut3d_chain_synthesizes() {
        let config = EncodeConfig {
            fragments: vec![EncodeFragment::Lut3d, EncodeFragment::OutputTransferPq],
        };
        let spv = synthesize_fragment_shader(&config).expect("synthesize");
        // SPIR-V starts with magic 0x07230203 and isn't empty.
        assert!(!spv.is_empty(), "empty SPIR-V");
        assert_eq!(spv[0], 0x07230203, "missing SPIR-V magic");
    }

    /// Existing chains (no LUT) keep emitting valid SPIR-V — verifies that
    /// the conditional `u_lut_ptr` declaration in `ShaderCtx::new` doesn't
    /// regress configs that never reference the LUT.
    #[test]
    fn pq_chain_without_lut_still_synthesizes() {
        let spv = synthesize_fragment_shader(&EncodeConfig::default_pq()).expect("synthesize");
        assert!(!spv.is_empty());
        assert_eq!(spv[0], 0x07230203);
    }
}
