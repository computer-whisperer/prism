//! Vulkan (ash) renderer.
//!
//! One renderer instance per GPU. Consumes `prism_frame::FrameDescription`,
//! produces scanout buffers for `prism_drm` to submit.

pub mod device;
pub mod error;
pub mod instance;

pub use device::{Device, DrmDevId, PhysicalDeviceInfo};
pub use error::{RendererError, Result};
pub use instance::Instance;
