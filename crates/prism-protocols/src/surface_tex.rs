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

use prism_renderer::{DrmDevId, ExportableImage, ImportedImage, ShmTexture, vk};
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
    },
    /// shm-backed. We don't hold a CPU copy of the bytes; uploads read the
    /// live buffer (`buffer`) via `with_buffer_contents` at materialization
    /// time. `extent`/`format` describe the upload target.
    Shm {
        extent: vk::Extent2D,
        format: vk::Format,
        buffer: WlBuffer,
    },
}

impl TexSource {
    pub fn buffer(&self) -> &WlBuffer {
        match self {
            Self::Dmabuf { buffer, .. } | Self::Shm { buffer, .. } => buffer,
        }
    }
    fn extent(&self) -> vk::Extent2D {
        match self {
            Self::Dmabuf { dmabuf, .. } => vk::Extent2D {
                width: dmabuf.width,
                height: dmabuf.height,
            },
            Self::Shm { extent, .. } => *extent,
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
    },
    /// Client bytes uploaded into an image on this GPU.
    Shm(ShmTexture),
}

impl GpuTex {
    pub fn view(&self) -> vk::ImageView {
        match self {
            Self::Native(img) => img.view(),
            Self::Mirror { target, .. } => target.view(),
            Self::Shm(t) => t.view(),
        }
    }
}

/// A client's committed buffer and its per-GPU materializations.
pub struct SurfaceTexture {
    pub source: TexSource,
    pub by_gpu: HashMap<DrmDevId, GpuTex>,
}

impl SurfaceTexture {
    pub fn new(source: TexSource) -> Self {
        Self {
            source,
            by_gpu: HashMap::new(),
        }
    }

    /// View into this surface's texture on `gpu`, or `None` if `gpu` has
    /// no materialization yet (not a consumer, or import/upload pending).
    pub fn view_for(&self, gpu: DrmDevId) -> Option<vk::ImageView> {
        self.by_gpu.get(&gpu).map(|t| t.view())
    }

    /// Whether this surface's texture on `gpu` is a cross-GPU mirror (vs a
    /// native import or shm upload). Mirrors need a per-frame copy +
    /// GPU-side sync before the render samples them; the render path uses
    /// this to collect them for `prepare_mirror_waits`.
    pub fn is_mirror_for(&self, gpu: DrmDevId) -> bool {
        matches!(self.by_gpu.get(&gpu), Some(GpuTex::Mirror { .. }))
    }

    /// Width × height of the source pixels (GPU-independent).
    pub fn extent(&self) -> vk::Extent2D {
        self.source.extent()
    }

    /// The `wl_buffer` this texture is currently backed by.
    pub fn buffer(&self) -> &WlBuffer {
        self.source.buffer()
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
