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

    /// HDR PQ default: identity calibration + PQ OETF.
    pub fn default_pq() -> Self {
        Self {
            fragments: vec![
                EncodeFragment::CalibrationMatrix,
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
}

/// Synthesize a SPIR-V fragment shader for the given `EncodeConfig`.
///
/// Returns the SPIR-V words (u32 sequence) suitable for `vkCreateShaderModule`.
pub fn synthesize_fragment_shader(config: &EncodeConfig) -> Result<Vec<u32>> {
    let mut ctx = builder::ShaderCtx::new();
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
