//! Vulkan (ash) renderer.
//!
//! One renderer instance per GPU. Consumes `prism_frame::FrameDescription`,
//! produces scanout buffers for `prism_drm` to submit.

pub mod device;
pub mod dmabuf;
pub mod error;
pub mod instance;
pub mod oneshot;

pub use device::{Device, DrmDevId, PhysicalDeviceInfo};
pub use dmabuf::ImportedImage;
pub use error::{RendererError, Result};
pub use instance::Instance;
pub use oneshot::OneshotPool;

// Re-export ash::vk so binary / glue crates don't need a direct ash dep.
pub use ash::vk;
