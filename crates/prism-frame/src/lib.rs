//! Shared low-level vocabulary for prism.
//!
//! A no-vk leaf crate carrying the types that several crates
//! (`prism-renderer`, `prism-protocols`, `prism-layout`, `prism-drm`) need
//! without taking a direct dependency on Vulkan or on each other:
//!
//! - color management ([`ColorDescription`], primaries / transfer functions,
//!   the BT.2020 working-space matrices),
//! - the [`Dmabuf`] descriptor for buffer import,
//! - stable cross-frame [`ElementId`] identity for damage tracking,
//! - re-exported geometry primitives.
//!
//! The renderer's frame model itself lives in `prism-renderer` and is
//! Vulkan-native (it carries `vk::ImageView`s directly) — prism is a
//! single-backend compositor (ash, Linux), so there is no backend-agnostic
//! boundary to maintain here.
//!
//! See `docs/color-management.md` and `docs/reuse-map.md` for
//! the design rationale.

pub mod color;
pub mod dmabuf;
pub mod id;

pub use color::{
    primaries_to_bt2020, srgb_to_bt2020_matrix, Chromaticities, ColorDescription, GammaExponent,
    MasteringInfo, Mat3, Primaries, TransferFunction,
};
pub use dmabuf::{Dmabuf, DmabufPlane};
pub use id::ElementId;

// Re-export fourcc types so downstream crates needn't depend on drm-fourcc
// directly to construct/inspect `Dmabuf`s.
pub use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};

// Re-export smithay geometry types so non-frame crates can use them through
// `prism_frame` without taking a direct smithay dependency for layout
// primitives.
pub use smithay::utils::{Buffer, Logical, Physical, Point, Rectangle, Scale, Size, Transform};
