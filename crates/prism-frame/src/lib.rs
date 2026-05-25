//! Renderer-independent frame description.
//!
//! This crate is the API boundary between high-level code
//! (`prism-layout`, `prism-protocols`, `prism-input`) and the renderer
//! (`prism-renderer`). High-level code produces `FrameDescription`s;
//! the renderer consumes them.
//!
//! The boundary is a concrete data structure rather than a trait because
//! there is one renderer (Vulkan via `ash`) — no backend polymorphism axis
//! to maintain. Opaque `TextureHandle` / `ShaderHandle` types keep renderer
//! resource types from leaking out.
//!
//! See `docs/phase-2-backend-notes.md` and `docs/phase-2-reuse-map.md` for
//! the design rationale.

pub mod color;
pub mod dmabuf;
pub mod element;
pub mod handle;
pub mod output;

pub use color::{
    primaries_to_bt2020, srgb_to_bt2020_matrix, Chromaticities, ColorDescription, GammaExponent,
    MasteringInfo, Mat3, Primaries, TransferFunction,
};
pub use dmabuf::{Dmabuf, DmabufPlane};
pub use element::{Element, ElementId, ElementSource, ShaderUniform};
pub use handle::{ShaderHandle, TextureHandle};
pub use output::{OutputId, OutputState};

// Re-export fourcc types so downstream crates needn't depend on drm-fourcc
// directly to construct/inspect `Dmabuf`s.
pub use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};

// Re-export smithay geometry types so non-frame crates can use them through
// `prism_frame` without taking a direct smithay dependency for layout
// primitives.
pub use smithay::utils::{Buffer, Logical, Physical, Point, Rectangle, Scale, Size, Transform};

/// One frame's worth of state for one output.
///
/// Produced by `prism-layout` (and renderer-side tracer code, in the MVP).
/// Consumed by `prism-renderer`.
#[derive(Clone, Debug)]
pub struct FrameDescription {
    pub output: OutputId,
    pub output_state: OutputState,
    /// Elements back-to-front (first element drawn first / lowest in stack).
    pub elements: Vec<Element>,
}

impl FrameDescription {
    pub fn new(output: OutputId, output_state: OutputState) -> Self {
        Self {
            output,
            output_state,
            elements: Vec::new(),
        }
    }
}
