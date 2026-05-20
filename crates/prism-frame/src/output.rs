//! Per-output rendering target description.

use std::num::NonZeroU64;

use smithay::utils::{Physical, Size, Transform};

use crate::color::ColorDescription;

/// Stable identifier for an output. Typically derived from the DRM connector
/// id; persists across mode changes but not across hotplug.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OutputId(NonZeroU64);

impl OutputId {
    pub fn from_raw(id: NonZeroU64) -> Self {
        Self(id)
    }

    pub fn raw(self) -> u64 {
        self.0.get()
    }
}

/// What the renderer needs to know about an output to render to it this frame.
///
/// Set per-frame so changes (mode change, HDR enable/disable, tone-map policy)
/// take effect on the next frame.
#[derive(Clone, Debug)]
pub struct OutputState {
    /// Output framebuffer size in physical pixels.
    pub size: Size<i32, Physical>,
    /// Compositor-to-output transform (rotation/flip).
    pub transform: Transform,
    /// Target color description: how the scanout buffer is encoded.
    /// E.g. `ColorDescription::BT2020_PQ` for HDR outputs;
    /// `ColorDescription::SRGB` for SDR outputs.
    pub target: ColorDescription,
    /// Panel peak luminance in nits, for tone-map decisions. None = SDR/unknown.
    pub peak_luminance_nits: Option<f32>,
    /// Reference white luminance to map SDR content to on an HDR output (nits).
    /// Typically 100–250. Ignored on SDR outputs.
    pub sdr_reference_white_nits: f32,
}
