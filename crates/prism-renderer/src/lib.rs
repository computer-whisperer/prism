//! Vulkan (ash) renderer.
//!
//! One renderer instance per GPU. Consumes the Vulkan-native frame model in
//! [`element`] (a back-to-front list of [`RenderEl`](element::RenderEl)s),
//! produces scanout buffers for `prism_drm` to submit.

pub mod capture;
pub mod cross_gpu;
pub mod damage;
pub mod device;
pub mod diagnose;
pub mod dmabuf;
pub mod element;
pub mod encode_synth;
pub mod error;
pub mod instance;
pub mod intermediate;
pub mod intermediate_capture;
pub mod lut3d;
pub mod oneshot;
pub mod pipeline;
pub mod renderer;
pub mod snapshot;
pub mod upload;

pub use capture::{CaptureEncoder, HostReadback};
pub use cross_gpu::{ExportableImage, MirrorCopier, MirrorCopyOp};
pub use damage::DamageTracker;
pub use device::{Device, DrmDevId, DrmFormatModifierInfo, PhysicalDeviceInfo};
pub use diagnose::{decode_scanout_texel, DiagnosedNits, EncodeDiagnoseProbe};
pub use dmabuf::{ImportedImage, YuvKind};
pub use element::{
    lower_elements, make_projector, srgb_to_bt2020_nits, AlphaMode, BorderEl, FrameElementMeta,
    LoweredFrame, RenderEl, SolidColorEl, SurfaceColorParams, SurfaceEl,
};
pub use encode_synth::{EncodeConfig, EncodeFragment, EncodePushSynth, LutOutputDomain};
pub use error::{RendererError, Result};
pub use instance::Instance;
pub use intermediate::{create_view, Intermediate, DEFAULT_INTERMEDIATE_FORMAT};
pub use lut3d::{
    drive_identity_lut, identity_lut, load_lut3d_file, pq_eotf, save_lut3d_file,
    synthesize_lut_from_matrix_curve, LoadedLut, Lut3dTexture, LutFileHeader,
    DEFAULT_DRIVE_WHITE_NITS, LUT_FILE_HEADER_BYTES, LUT_FILE_IN_TF_PQ, LUT_FILE_MAGIC,
    LUT_FILE_TRIPLE_BYTES, LUT_FILE_VERSION, LUT_FORMAT,
};
pub use oneshot::OneshotPool;
pub use pipeline::decode::{DecodePipeline, DecodePush};
pub use pipeline::encode::{EncodePipeline, EncodePush};
pub use renderer::{ElementDraw, Renderer, SnapshotCopy};
pub use snapshot::SnapshotTexture;
pub use upload::ShmTexture;

// Re-export ash::vk so binary / glue crates don't need a direct ash dep.
pub use ash::vk;
