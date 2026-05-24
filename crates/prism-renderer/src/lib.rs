//! Vulkan (ash) renderer.
//!
//! One renderer instance per GPU. Consumes `prism_frame::FrameDescription`,
//! produces scanout buffers for `prism_drm` to submit.

pub mod cross_gpu;
pub mod device;
pub mod diagnose;
pub mod dmabuf;
pub mod element;
pub mod encode_synth;
pub mod error;
pub mod instance;
pub mod intermediate;
pub mod lut3d;
pub mod oneshot;
pub mod pipeline;
pub mod renderer;
pub mod upload;

pub use cross_gpu::{ExportableImage, MirrorCopier, MirrorCopyOp};
pub use device::{Device, DrmDevId, DrmFormatModifierInfo, PhysicalDeviceInfo};
pub use diagnose::{DiagnosedNits, EncodeDiagnoseProbe, decode_scanout_texel};
pub use dmabuf::ImportedImage;
pub use element::{
    BorderEl, RenderEl, SolidColorEl, SurfaceColorParams, SurfaceEl, srgb_to_bt2020_nits,
};
pub use encode_synth::{EncodeConfig, EncodeFragment, EncodePushSynth};
pub use error::{RendererError, Result};
pub use instance::Instance;
pub use intermediate::{DEFAULT_INTERMEDIATE_FORMAT, Intermediate, create_view};
pub use lut3d::{
    LUT_FILE_HEADER_BYTES, LUT_FILE_IN_TF_PQ, LUT_FILE_MAGIC, LUT_FILE_TRIPLE_BYTES,
    LUT_FILE_VERSION, LUT_FORMAT, LoadedLut, Lut3dTexture, LutFileHeader, identity_lut,
    load_lut3d_file, pq_eotf, save_lut3d_file, synthesize_lut_from_matrix_curve,
};
pub use oneshot::OneshotPool;
pub use pipeline::decode::{DecodePipeline, DecodePush};
pub use pipeline::encode::{EncodePipeline, EncodePush};
pub use renderer::{ElementDraw, Renderer};
pub use upload::ShmTexture;

// Re-export ash::vk so binary / glue crates don't need a direct ash dep.
pub use ash::vk;
