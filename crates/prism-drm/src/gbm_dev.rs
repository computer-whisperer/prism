//! GBM (Generic Buffer Manager) device + scanout-capable BO allocation.
//!
//! GBM is the standard way on Linux to allocate buffers that both a GPU and
//! the display controller can use. The flow is:
//!
//!   1. Open a DRM fd (render node is fine for headless tests; primary node
//!      needed for actual scanout — `addfb2` requires DRM master).
//!   2. Build a `gbm::Device` on top of that fd.
//!   3. Allocate a `BufferObject` with `SCANOUT|RENDERING` usage and the set
//!      of DRM format modifiers we're willing to accept. GBM picks the best
//!      modifier the driver supports.
//!   4. Export the BO as a dmabuf (per plane), package into `prism_frame::Dmabuf`.
//!   5. Renderer imports the dmabuf as a `VkImage`.
//!
//! Critical: GEM handles are PER-FD. When the same BO is to be used both for
//! Vulkan import (via dmabuf, fd-agnostic) AND for `addfb2` (via GEM handle,
//! fd-specific), the GBM device and the DRM master device MUST share the
//! same fd, or `addfb2` will return ENOENT looking up an unknown handle.
//! Use `from_device_fd` with the same `DeviceFd` the `DrmDevice` holds.

use std::fs::OpenOptions;
use std::os::fd::OwnedFd;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use drm_fourcc::{DrmFourcc, DrmModifier};
use gbm::{BufferObject, BufferObjectFlags};
use prism_frame::{Dmabuf, DmabufPlane};
use smithay::utils::DeviceFd;

/// GBM device wrapper. The inner fd is a `DeviceFd` (Arc<OwnedFd>) so it can
/// be shared with a smithay `DrmDevice` — GEM handles are per-fd, so anything
/// that wants to addfb2 from this BO has to be on the same fd.
pub struct GbmDevice {
    inner: gbm::Device<DeviceFd>,
}

impl GbmDevice {
    /// Open a DRM node by path and use it for GBM. For headless tracer use
    /// (render node, no master needed). Cannot be used for actual scanout
    /// because the fd isn't shared with anything that holds DRM master.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .with_context(|| format!("opening {} for GBM", path.display()))?;
        let owned: OwnedFd = file.into();
        Self::from_device_fd(DeviceFd::from(owned))
    }

    /// Build a GBM device on an existing `DeviceFd`. Use this when the same
    /// fd must back both a `DrmDevice` (for atomic commits / addfb2) and GBM
    /// (for BO allocation) — the underlying `OwnedFd` is ref-counted, so both
    /// owners keep it alive.
    pub fn from_device_fd(fd: DeviceFd) -> Result<Self> {
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

    /// Allocate a cursor BO: `CURSOR|WRITE` lets the CPU update the
    /// sprite via `bo.write`, `LINEAR` keeps the layout writable and
    /// scanout-able without driver-specific tiling.
    ///
    /// Always ARGB8888 — the only format universally supported by
    /// hardware cursor planes.
    pub fn allocate_cursor(&self, width: u32, height: u32) -> Result<BufferObject<()>> {
        let usage =
            BufferObjectFlags::CURSOR | BufferObjectFlags::WRITE | BufferObjectFlags::LINEAR;
        let bo = self
            .inner
            .create_buffer_object::<()>(width, height, DrmFourcc::Argb8888, usage)
            .with_context(|| format!("create_buffer_object cursor {width}x{height}"))?;
        Ok(bo)
    }
}
