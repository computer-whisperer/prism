//! Per-output encode fragment shader synthesis (rspirv).
//!
//! The encode pass is *the* per-output customization point. Different
//! displays want different effects in different orders:
//!   - Standard SDR: `[Lut3d, OutputTransferSrgb]`
//!   - HDR PQ:       `[Lut3d, OutputTransferPq]`
//!   - QD-OLED text: `[Lut3d, OutputTransferSrgb, SubpixelFir3Horizontal]`
//!   - 8-bit panel:  `[..., InterleavedGradientNoiseDither]`
//!
//! The LUT output domain is chain-dependent and absolute in both modes:
//! PQ/linear chains expect linear nits, sRGB chains expect linear panel
//! drive in `[0, 1]` (the wire value pre-OETF). Either way the terminal
//! OutputTransfer fragment is a fixed function whose clamp exists only
//! to keep invalid control values off the wire — calibration meaning
//! lives entirely in the LUT and can't be re-scaled by runtime policy.
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
    /// `out = srgb_oetf(clamp(in, 0, 1))`. Input is linear panel drive
    /// (the LUT's output domain for SDR chains); the clamp only guards
    /// the wire against out-of-range control values.
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
    /// eye is sensitive), then a trilinear sample returns the panel-
    /// native command. Replaces the [`CalibrationMatrix`] +
    /// [`PerChannelResponseGainGamma`] pair: a single LUT captures both
    /// gamut correction AND per-channel response without assuming either
    /// is a closed-form function. The output domain matches the chain's
    /// terminal fragment: linear nits for PQ/linear chains, linear drive
    /// `[0, 1]` for sRGB chains.
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
    /// SDR default: 3D LUT (identity unless calibrated) + sRGB OETF.
    /// The LUT subsumes what was previously a separate `CalibrationMatrix`
    /// stage — calibration data flows through the LUT path uniformly with
    /// HDR. Identity LUT content gives the same output as the old
    /// identity-CTM chain modulo trilinear interpolation error.
    pub fn default_srgb() -> Self {
        Self {
            fragments: vec![EncodeFragment::Lut3d, EncodeFragment::OutputTransferSrgb],
        }
    }

    /// HDR PQ default: 3D LUT + PQ OETF. The LUT replaces the old
    /// `CalibrationMatrix` + `PerChannelResponseGainGamma` pair — a single
    /// trilinear sample captures both gamut correction AND per-channel
    /// response without assuming either is a closed-form function. The
    /// uniform LUT-only path also means HDR-mode calibration tools have
    /// one knob (LUT contents) instead of two (CTM + curve).
    pub fn default_pq() -> Self {
        Self {
            fragments: vec![EncodeFragment::Lut3d, EncodeFragment::OutputTransferPq],
        }
    }

    /// fp16-scanout default: 3D LUT + linear pass-through.
    pub fn default_linear() -> Self {
        Self {
            fragments: vec![EncodeFragment::Lut3d, EncodeFragment::OutputTransferLinear],
        }
    }

    /// True if any fragment in this chain references the per-output 3D LUT
    /// (binding 1). The shader synthesizer + pipeline layout both honor
    /// this — when false, binding 1 is never declared and the pipeline
    /// doesn't expect an LUT image at draw time.
    pub fn uses_lut3d(&self) -> bool {
        self.fragments
            .iter()
            .any(|f| matches!(f, EncodeFragment::Lut3d))
    }

    /// The output domain the chain's LUT must be baked/synthesized in,
    /// derived from the terminal OutputTransfer fragment: sRGB chains
    /// consume linear drive `[0, 1]`, PQ/linear chains consume linear
    /// nits. Chains without an OutputTransfer default to nits (the IR
    /// domain passes through untouched).
    pub fn lut_output_domain(&self) -> LutOutputDomain {
        for f in self.fragments.iter().rev() {
            match f {
                EncodeFragment::OutputTransferSrgb => return LutOutputDomain::Drive,
                EncodeFragment::OutputTransferPq | EncodeFragment::OutputTransferLinear => {
                    return LutOutputDomain::Nits
                }
                _ => {}
            }
        }
        LutOutputDomain::Nits
    }
}

/// Domain of the values a chain's 3D LUT outputs — what the terminal
/// OutputTransfer fragment expects as input. See
/// [`EncodeConfig::lut_output_domain`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LutOutputDomain {
    /// Linear nits (absolute). PQ + linear-scanout chains.
    Nits,
    /// Linear panel drive in `[0, 1]` (the wire value pre-OETF). sRGB
    /// chains.
    Drive,
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

    /// `uses_lut3d` reports the LUT presence honestly. All built-in
    /// default chains now route calibration through the LUT, so they all
    /// answer `true`; the negative case is a synthetic config without
    /// `Lut3d` (used for testing the matrix+curve fragments directly).
    #[test]
    fn uses_lut3d_matches_chain_contents() {
        assert!(EncodeConfig::default_srgb().uses_lut3d());
        assert!(EncodeConfig::default_pq().uses_lut3d());
        assert!(EncodeConfig::default_linear().uses_lut3d());
        let no_lut = EncodeConfig {
            fragments: vec![
                EncodeFragment::CalibrationMatrix,
                EncodeFragment::OutputTransferPq,
            ],
        };
        assert!(!no_lut.uses_lut3d());
    }

    /// Default PQ chain emits a non-empty, magic-headered SPIR-V module.
    /// Catches regressions in the conditional-binding emission path the
    /// LUT-aware ShaderCtx::new takes when `uses_lut3d` is true.
    #[test]
    fn default_pq_chain_synthesizes() {
        let spv = synthesize_fragment_shader(&EncodeConfig::default_pq()).expect("synthesize");
        assert!(!spv.is_empty(), "empty SPIR-V");
        assert_eq!(spv[0], 0x07230203, "missing SPIR-V magic");
    }

    /// Synthetic chain that omits Lut3d still synthesizes — exercises the
    /// no-LUT branch of `ShaderCtx::new` so a future refactor doesn't
    /// silently regress the matrix+curve path.
    #[test]
    fn no_lut_chain_still_synthesizes() {
        let config = EncodeConfig {
            fragments: vec![
                EncodeFragment::CalibrationMatrix,
                EncodeFragment::PerChannelResponseGainGamma,
                EncodeFragment::OutputTransferPq,
            ],
        };
        let spv = synthesize_fragment_shader(&config).expect("synthesize");
        assert!(!spv.is_empty());
        assert_eq!(spv[0], 0x07230203);
    }
}
