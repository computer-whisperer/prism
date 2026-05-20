//! GBM (Generic Buffer Manager) device + scanout-capable BO allocation.
//!
//! GBM is the standard way on Linux to allocate buffers that both a GPU and
//! the display controller can use. The flow is:
//!
//!   1. Open a DRM fd (render node is fine for non-scanout; primary node
//!      needed for actual scanout — but the BO itself doesn't care, only the
//!      eventual `addfb2` does).
//!   2. Build a `gbm::Device` on top of that fd.
//!   3. Allocate a `BufferObject` with `SCANOUT|RENDERING` usage and the set
//!      of DRM format modifiers we're willing to accept. GBM picks the best
//!      modifier the driver supports.
//!   4. Export the BO as a dmabuf (per plane), package into `prism_frame::Dmabuf`.
//!   5. Renderer imports the dmabuf as a `VkImage`.
//!
//! For the first-pass tracer we ask for `LINEAR` only — keeps the layout
//! trivially CPU-mappable for verification.

use std::fs::{File, OpenOptions};
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use drm_fourcc::{DrmFourcc, DrmModifier};
use gbm::{BufferObject, BufferObjectFlags};
use prism_frame::{Dmabuf, DmabufPlane};

/// A DRM fd owned for GBM use. `gbm::Device` requires `AsFd`.
pub struct GbmFd(File);

impl GbmFd {
    /// Open a DRM node (primary or render) for GBM. Does NOT acquire master.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        Ok(Self(file))
    }
}

impl AsFd for GbmFd {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

/// GBM device wrapper. Holds the DRM fd alive for the device's lifetime.
pub struct GbmDevice {
    inner: gbm::Device<GbmFd>,
}

impl GbmDevice {
    pub fn new(fd: GbmFd) -> Result<Self> {
        let inner = gbm::Device::new(fd).context("gbm::Device::new")?;
        Ok(Self { inner })
    }

    pub fn backend_name(&self) -> &str {
        self.inner.backend_name()
    }

    /// Allocate a scanout-usable BO and package its dmabuf handles.
    ///
    /// Caller provides a candidate modifier list; GBM picks one the driver
    /// supports. The chosen modifier is recorded on the returned `Dmabuf`.
    ///
    /// The returned `BufferObject` *owns the underlying gem object*. The
    /// `Dmabuf`'s plane fds are independent kernel references — dropping the
    /// `Dmabuf` does not free the BO, and dropping the BO does not invalidate
    /// the fds. Keep both alive for as long as the image is in use.
    pub fn allocate_scanout(
        &self,
        width: u32,
        height: u32,
        format: DrmFourcc,
        modifiers: &[DrmModifier],
    ) -> Result<(BufferObject<()>, Dmabuf)> {
        let usage = BufferObjectFlags::SCANOUT | BufferObjectFlags::RENDERING;

        let bo: BufferObject<()> = self
            .inner
            .create_buffer_object_with_modifiers2(
                width,
                height,
                format,
                modifiers.iter().copied(),
                usage,
            )
            .with_context(|| {
                format!(
                    "create_buffer_object_with_modifiers2 {width}x{height} {format:?} \
                     modifiers={modifiers:?}"
                )
            })?;

        let chosen_modifier = bo.modifier();
        let plane_count = bo.plane_count();
        if plane_count == 0 {
            return Err(anyhow!("GBM returned BO with 0 planes"));
        }

        let mut planes: Vec<DmabufPlane> = Vec::with_capacity(plane_count as usize);
        for i in 0..plane_count as i32 {
            let fd: OwnedFd = bo
                .fd_for_plane(i)
                .map_err(|_| anyhow!("gbm_bo_get_fd_for_plane({i}) returned -1"))?;
            planes.push(DmabufPlane {
                fd,
                offset: bo.offset(i),
                stride: bo.stride_for_plane(i),
            });
        }

        let dmabuf = Dmabuf {
            width,
            height,
            format,
            modifier: chosen_modifier,
            planes,
        };

        Ok((bo, dmabuf))
    }
}
