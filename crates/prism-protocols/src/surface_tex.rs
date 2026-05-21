//! Per-surface texture cache (multi-GPU aware).
//!
//! A `SurfaceTexture` represents a client's most recently committed buffer.
//! In both variants the buffer's pixels are available on **every registered
//! GPU**, so any output's render path can sample it via [`view_for`] without
//! caring which GPU rendered the frame.
//!
//!   - **Dmabuf**: imported as a `VkImage` on each GPU in `dmabuf_imported`
//!     (zero-copy import; the kernel dups the fd per device).
//!   - **Shm**: client bytes are read once and uploaded into a per-GPU
//!     staged `VkImage` on each commit. Costs N× upload bandwidth for N
//!     GPUs but keeps the render path uniform; surface-to-output mapping
//!     (#59.5) will later let us skip GPUs whose outputs don't host the
//!     surface.
//!
//! Storage: one `SurfaceTexSlot` per `wl_surface`, kept in the surface's
//! `data_map`. Populated / refreshed by `process_surface_buffer` on each
//! commit; read by the render path on each frame.
//!
//! [`view_for`]: SurfaceTexture::view_for

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use prism_renderer::{DrmDevId, ImportedImage, ShmTexture, vk};
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;

/// What the renderer samples from on a given commit.
pub enum SurfaceTexture {
    /// Bytes uploaded into a Vulkan image on every registered GPU. Lookup
    /// by GPU returns the matching upload target.
    Shm { by_gpu: HashMap<DrmDevId, ShmTexture> },
    /// VkImage backed by client-owned dmabuf (linux-dmabuf path), imported
    /// on every registered GPU. Lookup by GPU returns the matching import.
    ///
    /// `buffer` is held so we can send `wl_buffer.release` when the
    /// client commits a replacement buffer. Without that the client's
    /// dmabuf pool fills up and it stalls / errors out (mpv exits with
    /// "Error occurred on display fd" after ~4 unreleased commits).
    /// The release happens at "replacement" time, not at present-done
    /// time — so there's a small race where we may still be GPU-reading
    /// the BO when the client starts writing it. Acceptable for video
    /// (next frame masks any tearing); proper sync needs explicit
    /// fences (linux-drm-syncobj-v1) and lands later.
    Dmabuf {
        by_gpu: HashMap<DrmDevId, Arc<ImportedImage>>,
        buffer: WlBuffer,
    },
}

impl SurfaceTexture {
    /// View into this surface's texture on the given GPU, or `None` if the
    /// GPU doesn't have an applicable upload/import (e.g. import failed on
    /// this GPU for a dmabuf, or the GPU wasn't registered when this
    /// texture was built).
    pub fn view_for(&self, gpu: DrmDevId) -> Option<vk::ImageView> {
        match self {
            Self::Shm { by_gpu } => by_gpu.get(&gpu).map(|t| t.view()),
            Self::Dmabuf { by_gpu, .. } => by_gpu.get(&gpu).map(|t| t.view()),
        }
    }

    /// Width × height in pixels, GPU-independent (same across all
    /// uploads/imports of the same buffer).
    pub fn extent(&self) -> vk::Extent2D {
        match self {
            Self::Shm { by_gpu } => by_gpu
                .values()
                .next()
                .map(|t| t.extent())
                .unwrap_or_default(),
            Self::Dmabuf { by_gpu, .. } => by_gpu
                .values()
                .next()
                .map(|i| i.extent())
                .unwrap_or_default(),
        }
    }
}

/// Per-surface slot inserted into `SurfaceData::data_map`. Wrapped in
/// `Mutex` because the surface's data_map is shared (`&UserDataMap`) but
/// the texture replacement happens from `commit` (single-threaded today,
/// but the smithay API gives us a shared reference, so we lock to mutate).
#[derive(Default)]
pub struct SurfaceTexSlot(pub Mutex<Option<SurfaceTexture>>);

/// Per-surface layout state — where the surface lives in logical space
/// and which output it currently belongs to. Inserted into the surface's
/// `SurfaceData::data_map` alongside [`SurfaceTexSlot`]; mutated from
/// the commit hook (which holds `&UserDataMap`, hence the interior
/// `Mutex`).
///
/// `current_output` is updated by `process_surface_buffer` after each
/// commit; on transitions we dispatch `wl_surface.enter` / `.leave`
/// events to the appropriate smithay `Output`s. Today's containment
/// rule is "the output whose geometry contains the surface's center";
/// when surfaces span multiple outputs (real layout / overlapping
/// outputs / fractional scaling) this becomes a `Vec<OutputId>`.
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
