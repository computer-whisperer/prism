//! Cross-process buffer description for dmabuf import/export.
//!
//! A `Dmabuf` is the wire-format handoff between an allocator (GBM for our
//! own scanout buffers, Wayland clients for their textures) and the renderer
//! that imports it as a `VkImage`. The fds are owned — dropping the
//! description closes them, which releases the kernel-side BO reference.

use std::os::fd::OwnedFd;

use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};

/// One plane of a (possibly multi-planar) dmabuf.
#[derive(Debug)]
pub struct DmabufPlane {
    /// Dup'd fd owned by this plane. May be the same kernel BO as another
    /// plane's fd (multi-planar formats), but each plane carries its own
    /// owned fd to keep ownership simple.
    pub fd: OwnedFd,
    /// Byte offset from the start of the fd's BO to this plane's first pixel.
    pub offset: u32,
    /// Bytes per row.
    pub stride: u32,
}

/// A cross-API buffer description usable as a Vulkan import source.
///
/// Convention: planes\[0\] is the first/primary plane. For single-planar
/// formats like XRGB8888 there is exactly one plane.
#[derive(Debug)]
pub struct Dmabuf {
    pub width: u32,
    pub height: u32,
    pub format: DrmFourcc,
    pub modifier: DrmModifier,
    pub planes: Vec<DmabufPlane>,
}

impl Dmabuf {
    pub fn drm_format(&self) -> DrmFormat {
        DrmFormat {
            code: self.format,
            modifier: self.modifier,
        }
    }
}
