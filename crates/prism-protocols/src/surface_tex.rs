//! Per-surface texture cache (multi-GPU, consumer-driven).
//!
//! A `SurfaceTexture` is a client's most recently committed buffer plus the
//! per-GPU materializations of its pixels. The model separates two things:
//!
//!   - **source** ([`TexSource`]): a GPU-agnostic description of the pixels
//!     — the dmabuf fds, or the shm buffer handle — kept so we can
//!     materialize on a GPU lazily, when (and only when) an output on that
//!     GPU actually displays the surface.
//!   - **materializations** (`by_gpu`): for each GPU that needs to sample
//!     this surface, the concrete Vulkan texture it samples ([`GpuTex`]).
//!
//! A GPU's materialization is one of:
//!   - [`GpuTex::Native`] — zero-copy dmabuf import. Available only on GPUs
//!     whose driver understands the buffer's DRM format modifier.
//!   - [`GpuTex::Mirror`] — for a GPU that *can't* natively import the
//!     buffer (different vendor/gen in a multi-GPU box): a LINEAR
//!     exportable scratch image on a "home" GPU that can read the buffer,
//!     copied each commit and re-imported on this GPU. The cross-GPU
//!     fallback; see [`prism_renderer::cross_gpu`].
//!   - [`GpuTex::Shm`] — client bytes uploaded into a per-GPU image.
//!
//! Unlike the previous design, materialization is **not** eager on every
//! registered GPU. The set of consuming GPUs is derived from where the
//! surface is displayed (placement + render-walk demand), so a window on a
//! single monitor only ever materializes on that monitor's GPU.
//!
//! Storage: one `SurfaceTexSlot` per `wl_surface` in the surface's
//! `data_map`. Source set by `process_surface_buffer` on commit;
//! per-GPU materializations built/refreshed by `ensure_surface_textures`.
//! Read by the render path via [`SurfaceTexture::view_for`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use prism_renderer::{
    vk, AlphaMode, DrmDevId, ExportableImage, ImportedImage, ShmTexture, YuvKind,
};
use smithay::backend::renderer::utils::CommitCounter;
use smithay::reexports::wayland_server::backend::ObjectId;
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;

/// GPU-agnostic description of a surface's current pixels, retained so we
/// can materialize the texture on any GPU on demand.
pub enum TexSource {
    /// dmabuf-backed. `dmabuf` owns dup'd fds and can be imported on any
    /// GPU whose driver supports its modifier; GPUs that can't get a
    /// [`GpuTex::Mirror`] instead. `format` is the Vulkan format the
    /// import uses. `buffer` is held so the render-path release tracking
    /// and reuse detection have the original `wl_buffer` identity.
    Dmabuf {
        dmabuf: Arc<prism_frame::Dmabuf>,
        format: vk::Format,
        buffer: WlBuffer,
        /// Whether the source fourcc carries meaningful alpha (`A`-format) vs
        /// none (`X`-format or YUV). Tracked from the fourcc because
        /// `vk::Format` conflates `Xrgb`/`Argb` (both `B8G8R8A8_UNORM`).
        has_alpha: bool,
    },
    /// shm-backed. We don't hold a CPU copy of the bytes; uploads read the
    /// live buffer (`buffer`) via `with_buffer_contents` at materialization
    /// time. `extent`/`format` describe the upload target.
    Shm {
        extent: vk::Extent2D,
        format: vk::Format,
        buffer: WlBuffer,
        /// Whether the source `wl_shm` format carries meaningful alpha — see
        /// the `Dmabuf` variant's `has_alpha`.
        has_alpha: bool,
    },
    /// A `wp_single_pixel_buffer` solid color (e.g. swaybg `-c`, GTK/Qt solid
    /// backgrounds). There is no texture to upload — the render walk lowers it
    /// to a color-managed [`SolidColorEl`](prism_renderer::SolidColorEl) from
    /// `rgba` (premultiplied sRGB per the protocol). `buffer` is held for
    /// reuse-detection / release-tracking parity with the textured variants.
    SolidColor { rgba: [u8; 4], buffer: WlBuffer },
}

impl TexSource {
    pub fn buffer(&self) -> &WlBuffer {
        match self {
            Self::Dmabuf { buffer, .. }
            | Self::Shm { buffer, .. }
            | Self::SolidColor { buffer, .. } => buffer,
        }
    }

    /// Whether the source's pixels carry meaningful alpha. `A`-format
    /// dmabuf/shm buffers do (premultiplied per the Wayland contract);
    /// `X`-format and YUV buffers don't. `SolidColor` reports `true` —
    /// `wp_single_pixel_buffer` is premultiplied — though it never samples a
    /// texture (the walk lowers it to a `SolidColorEl`).
    fn has_alpha(&self) -> bool {
        match self {
            Self::Dmabuf { has_alpha, .. } | Self::Shm { has_alpha, .. } => *has_alpha,
            Self::SolidColor { .. } => true,
        }
    }
    fn extent(&self) -> vk::Extent2D {
        match self {
            Self::Dmabuf { dmabuf, .. } => vk::Extent2D {
                width: dmabuf.width,
                height: dmabuf.height,
            },
            Self::Shm { extent, .. } => *extent,
            // No texture; the surface's logical dst (from its viewport) drives
            // the SolidColorEl rect, not this extent.
            Self::SolidColor { .. } => vk::Extent2D {
                width: 1,
                height: 1,
            },
        }
    }
}

/// One GPU's materialization of a surface's pixels.
pub enum GpuTex {
    /// Zero-copy dmabuf import on this GPU. The client writes the BO, we
    /// sample it — no per-frame copy.
    Native(Arc<ImportedImage>),
    /// Cross-GPU mirror: this GPU can't import the client buffer, so a
    /// "home" GPU that can holds the client import (`home_src`) and copies
    /// it into a LINEAR exportable scratch image (`scratch`) each commit;
    /// `target` is that scratch's dmabuf imported on *this* GPU. Sampling
    /// `target` returns the mirrored pixels.
    ///
    /// `scratch` + `target` depend only on extent/format and are **reused
    /// across buffer swaps** (a churning client that reallocates its
    /// dmabuf every frame keeps the same scratch + target import — we only
    /// re-import `home_src` and re-copy). `home_src_buffer` is the
    /// `wl_buffer` id `home_src` was imported from, so the refresh path
    /// knows when the client buffer changed and `home_src` must be
    /// re-imported.
    Mirror {
        home: DrmDevId,
        home_src: Arc<ImportedImage>,
        home_src_buffer: ObjectId,
        scratch: ExportableImage,
        target: Arc<ImportedImage>,
        /// Chroma plane for a YUV mirror (NV12/P010): a second, half-res
        /// scratch+target carrying the interleaved chroma. `None` for an
        /// RGB mirror. When set, `scratch`/`target` hold the luma plane and
        /// the consumer's decode shader recombines the two — the same path
        /// a native YUV import takes, just fed from mirrored memory.
        chroma: Option<MirrorChroma>,
    },
    /// Client bytes uploaded into an image on this GPU.
    Shm(ShmTexture),
}

/// The chroma half of a cross-GPU YUV mirror (see [`GpuTex::Mirror`]).
/// Mirrors the luma plane's `scratch`/`target` pair at half resolution;
/// `kind` selects the decode path (NV12 vs P010).
pub struct MirrorChroma {
    pub scratch: ExportableImage,
    pub target: Arc<ImportedImage>,
    pub kind: YuvKind,
}

impl GpuTex {
    pub fn view(&self) -> vk::ImageView {
        match self {
            Self::Native(img) => img.view(),
            Self::Mirror { target, .. } => target.view(),
            Self::Shm(t) => t.view(),
        }
    }

    /// Chroma plane view + YUV kind code (matching `DecodePush::yuv`:
    /// 1 = NV12, 2 = P010) for a YUV surface. `(None, 0)` for RGB native
    /// imports, RGB mirrors, and shm. Both the zero-copy native path and
    /// the cross-GPU mirror carry YUV; shm is RGB-only.
    pub fn yuv(&self) -> (Option<vk::ImageView>, i32) {
        let code = |kind| match kind {
            YuvKind::Nv12 => 1,
            YuvKind::P010 => 2,
        };
        match self {
            Self::Native(img) => match img.yuv_kind() {
                Some(kind) => (img.chroma_view(), code(kind)),
                None => (None, 0),
            },
            Self::Mirror {
                chroma: Some(c), ..
            } => (Some(c.target.view()), code(c.kind)),
            _ => (None, 0),
        }
    }
}

/// A client's committed buffer and its per-GPU materializations.
pub struct SurfaceTexture {
    pub source: TexSource,
    pub by_gpu: HashMap<DrmDevId, GpuTex>,
    /// Commit counter (from smithay's `RendererSurfaceState`) of the last shm
    /// upload, so the next upload can fetch only the damage since then via
    /// `damage_since`. `None` until the first shm upload, and carried across
    /// buffer swaps that reuse the per-GPU `ShmTexture`s (see
    /// `process_surface_buffer`). Unused for dmabuf/solid sources.
    pub shm_upload_commit: Option<CommitCounter>,
    /// GPUs whose render has already waited on the client's implicit write
    /// fence for the CURRENTLY committed (native dmabuf) buffer. Emptied on
    /// every dmabuf commit (`ensure_surface_textures`); a GPU is added by
    /// `mark_dmabuf_acquire_waited` only after a render submit that carries
    /// the imported fence in its wait list was actually queued — NOT at
    /// fence-import time, because `present()` can return FlipPending /
    /// SkippedNoDamage without submitting any GPU work, and clearing early
    /// would let the retry sample the buffer with no producer wait (the
    /// fa62fb9 blue-bleed race). Per-GPU so one output's wait doesn't eat
    /// the fence for the other GPUs' renders. A written buffer only needs
    /// syncing on its first sample per GPU, not every frame it stays on
    /// screen — keeping the per-frame sync_file/semaphore churn bounded.
    /// Unused for shm/solid.
    pub acquire_waited: std::collections::HashSet<DrmDevId>,
}

impl SurfaceTexture {
    pub fn new(source: TexSource) -> Self {
        Self {
            source,
            by_gpu: HashMap::new(),
            shm_upload_commit: None,
            acquire_waited: std::collections::HashSet::new(),
        }
    }

    /// View into this surface's texture on `gpu`, or `None` if `gpu` has
    /// no materialization yet (not a consumer, or import/upload pending).
    pub fn view_for(&self, gpu: DrmDevId) -> Option<vk::ImageView> {
        self.by_gpu.get(&gpu).map(|t| t.view())
    }

    /// Chroma view + YUV kind code for `gpu`, for YUV video surfaces.
    /// `(None, 0)` when `gpu` has no materialization or the texture is RGB.
    /// Pairs with [`Self::view_for`] (the luma/primary plane).
    pub fn yuv_for(&self, gpu: DrmDevId) -> (Option<vk::ImageView>, i32) {
        self.by_gpu.get(&gpu).map(|t| t.yuv()).unwrap_or((None, 0))
    }

    /// Whether this surface's texture on `gpu` is a cross-GPU mirror (vs a
    /// native import or shm upload). Mirrors need a per-frame copy +
    /// GPU-side sync before the render samples them; the render path uses
    /// this to collect them for `prepare_mirror_waits`.
    pub fn is_mirror_for(&self, gpu: DrmDevId) -> bool {
        matches!(self.by_gpu.get(&gpu), Some(GpuTex::Mirror { .. }))
    }

    /// Whether this surface's texture on `gpu` is a zero-copy native dmabuf
    /// import (vs a mirror or shm upload). The render path uses this to know
    /// which surfaces need their client's implicit write fence imported as a
    /// render wait — prism samples the client BO directly, so it must not read
    /// it mid-write. See `prepare_dmabuf_acquire_waits`.
    pub fn is_native_dmabuf_for(&self, gpu: DrmDevId) -> bool {
        matches!(self.by_gpu.get(&gpu), Some(GpuTex::Native(_)))
    }

    /// Width × height of the source pixels (GPU-independent).
    pub fn extent(&self) -> vk::Extent2D {
        self.source.extent()
    }

    /// The `wl_buffer` this texture is currently backed by.
    pub fn buffer(&self) -> &WlBuffer {
        self.source.buffer()
    }

    /// Premultiplied sRGB RGBA if this surface is a `wp_single_pixel_buffer`
    /// solid color (rendered as a color-managed `SolidColorEl`, no texture);
    /// `None` for textured surfaces.
    pub fn solid_color(&self) -> Option<[u8; 4]> {
        match &self.source {
            TexSource::SolidColor { rgba, .. } => Some(*rgba),
            _ => None,
        }
    }

    /// How the decode shader should interpret this surface's sampled alpha.
    /// A GPU-independent buffer-format property: `A`-format buffers are
    /// premultiplied (the Wayland contract), `X`-format and YUV buffers are
    /// opaque. (YUV is recorded with `has_alpha = false` at build time, so it
    /// falls into `Opaque` here without a per-GPU lookup.)
    pub fn alpha_mode(&self) -> AlphaMode {
        if self.source.has_alpha() {
            AlphaMode::Premultiplied
        } else {
            AlphaMode::Opaque
        }
    }

    /// Whether the source buffer is 8-bit-per-component RGB — the precondition
    /// for debanding (the decode pass's ±0.5 LSB clamp is defined on 8-bit
    /// codes; a 10-bit/fp16 buffer carries no 8-bit banding and must not be
    /// clamped against a 1/255 LSB). Derived from the source `vk::Format`;
    /// `SolidColor` has no sampled texture so it reports `false`.
    pub fn is_8bit(&self) -> bool {
        match &self.source {
            TexSource::Dmabuf { format, .. } | TexSource::Shm { format, .. } => {
                prism_renderer::format_is_8bit_rgb(*format)
            }
            TexSource::SolidColor { .. } => false,
        }
    }
}

/// Per-surface slot inserted into `SurfaceData::data_map`. Wrapped in
/// `Mutex` because the surface's data_map is shared (`&UserDataMap`) but
/// texture replacement / materialization happens from `commit`.
#[derive(Default)]
pub struct SurfaceTexSlot(pub Mutex<Option<SurfaceTexture>>);

/// Per-surface layout state — where the surface lives in logical space
/// and which output it currently belongs to. Inserted into the surface's
/// `SurfaceData::data_map` alongside [`SurfaceTexSlot`]; mutated from
/// the commit hook (which holds `&UserDataMap`, hence the interior
/// `Mutex`).
///
/// `current_output` is updated by `dispatch_surface_output_from_layout`
/// after each commit; on transitions we dispatch `wl_surface.enter` /
/// `.leave` events to the appropriate smithay `Output`s. Today's
/// containment rule is single-output (the layout assigns each window to
/// one monitor). The texture layer treats the consuming-GPU set as
/// plural regardless, so when floating-window spanning lands this becomes
/// a set with no churn to the materialization path.
#[derive(Default)]
pub struct SurfacePlacementSlot(pub Mutex<SurfacePlacement>);

/// Logical-space placement of a `wl_surface`.
#[derive(Default, Clone)]
pub struct SurfacePlacement {
    /// Top-left in logical pixels. Defaults to `(0, 0)`; today every
    /// new toplevel pins here until a config / layout layer assigns
    /// real coordinates.
    pub logical_pos: (i32, i32),
    /// `OutputId` of the wl_output the surface most recently entered,
    /// or `None` if not currently mapped to any output. Stored as the
    /// stable connector-name `String` to match `PrismState::wl_outputs`'s
    /// key. Single-output for now (center-containment); becomes a set
    /// when surfaces span outputs.
    pub current_output: Option<String>,
}
