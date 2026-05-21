//! Vulkan (ash) renderer.
//!
//! One renderer instance per GPU. Consumes `prism_frame::FrameDescription`,
//! produces scanout buffers for `prism_drm` to submit.

pub mod device;
pub mod dmabuf;
pub mod element;
pub mod encode_synth;
pub mod error;
pub mod instance;
pub mod intermediate;
pub mod oneshot;
pub mod pipeline;
pub mod renderer;
pub mod upload;

pub use device::{Device, DrmDevId, PhysicalDeviceInfo};
pub use dmabuf::ImportedImage;
pub use element::{BorderEl, RenderEl, SolidColorEl, SurfaceEl, srgb_to_bt2020_nits};
pub use encode_synth::{EncodeConfig, EncodeFragment, EncodePushSynth};
pub use error::{RendererError, Result};
pub use instance::Instance;
pub use intermediate::{DEFAULT_INTERMEDIATE_FORMAT, Intermediate, create_view};
pub use oneshot::OneshotPool;
pub use pipeline::decode::{DecodePipeline, DecodePush};
pub use pipeline::encode::{EncodePipeline, EncodePush};
pub use renderer::{ElementDraw, Renderer};
pub use upload::ShmTexture;

// Re-export ash::vk so binary / glue crates don't need a direct ash dep.
pub use ash::vk;
