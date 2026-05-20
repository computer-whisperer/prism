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

/// What the renderer samples from on a given commit.
pub enum SurfaceTexture {
    /// Bytes uploaded into a Vulkan image on every registered GPU. Lookup
    /// by GPU returns the matching upload target.
    Shm { by_gpu: HashMap<DrmDevId, ShmTexture> },
    /// VkImage backed by client-owned dmabuf (linux-dmabuf path), imported
    /// on every registered GPU. Lookup by GPU returns the matching import.
    Dmabuf { by_gpu: HashMap<DrmDevId, Arc<ImportedImage>> },
}

impl SurfaceTexture {
    /// View into this surface's texture on the given GPU, or `None` if the
    /// GPU doesn't have an applicable upload/import (e.g. import failed on
    /// this GPU for a dmabuf, or the GPU wasn't registered when this
    /// texture was built).
    pub fn view_for(&self, gpu: DrmDevId) -> Option<vk::ImageView> {
        match self {
            Self::Shm { by_gpu } => by_gpu.get(&gpu).map(|t| t.view()),
            Self::Dmabuf { by_gpu } => by_gpu.get(&gpu).map(|t| t.view()),
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
            Self::Dmabuf { by_gpu } => by_gpu
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
