//! dmabuf → VkImage import.
//!
//! Takes a `prism_frame::Dmabuf` and creates a Vulkan image backed by the
//! same kernel BO, via:
//!   - `VK_EXT_image_drm_format_modifier` — tells the driver the image has
//!     a specific DRM format modifier and per-plane layout (offset/stride).
//!   - `VK_EXT_external_memory_dma_buf` + `VK_KHR_external_memory_fd` —
//!     imports the dmabuf fd as Vulkan device memory.
//!
//! Single-planar formats (XRGB8888, ARGB8888, RGBA16F, ...) only: we pass one
//! fd into a single allocation, bound to one image with `plane_layouts` of
//! length 1. Multi-planar formats (NV12, P010, ...) will need a separate
//! import path with disjoint memory.

use std::os::fd::{AsFd, AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
use std::sync::Arc;

use ash::khr::external_memory_fd;
use ash::vk;
use prism_frame::Dmabuf;
use tracing::debug;

use crate::device::Device;
use crate::error::{RendererError, Result, VkResultExt};

/// A `VkImage` backed by imported dmabuf memory. Owns the image + memory and
/// destroys them on drop. Does NOT own the dmabuf fds — those live on the
/// caller's `Dmabuf`.
pub struct ImportedImage {
    device: Arc<Device>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    extent: vk::Extent2D,
    format: vk::Format,
}

impl ImportedImage {
    pub fn image(&self) -> vk::Image {
        self.image
    }
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Import a dmabuf as a `VkImage`.
    ///
    /// `vk_format` must match the dmabuf's `DrmFourcc` byte layout. E.g.
    /// `DrmFourcc::Xrgb8888` ↔ `vk::Format::B8G8R8A8_UNORM`: DRM is
    /// little-endian-byte-order, so XRGB-in-memory is B,G,R,X bytes, which
    /// is Vulkan's B8G8R8A8.
    pub fn import(
        device: Arc<Device>,
        dmabuf: &Dmabuf,
        vk_format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self> {
        if dmabuf.planes.len() != 1 {
            return Err(RendererError::MissingFeature(
                "multi-planar dmabuf import not implemented yet",
            ));
        }
        let plane = &dmabuf.planes[0];

        let plane_layouts = [vk::SubresourceLayout {
            offset: u64::from(plane.offset),
            size: 0,
            row_pitch: u64::from(plane.stride),
            array_pitch: 0,
            depth_pitch: 0,
        }];

        let mut modifier_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(u64::from(dmabuf.modifier))
            .plane_layouts(&plane_layouts);

        let mut external_image = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width: dmabuf.width,
                height: dmabuf.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_image)
            .push_next(&mut modifier_info);

        let image =
            unsafe { device.raw.create_image(&image_info, None) }.vk_ctx("create_image (dmabuf)")?;

        let memory = match allocate_imported_memory(&device, image, plane.fd.as_fd()) {
            Ok(m) => m,
            Err(e) => {
                unsafe { device.raw.destroy_image(image, None) };
                return Err(e);
            }
        };

        let bind_info = [vk::BindImageMemoryInfo::default()
            .image(image)
            .memory(memory)
            .memory_offset(0)];
        if let Err(e) = unsafe { device.raw.bind_image_memory2(&bind_info) } {
            unsafe {
                device.raw.free_memory(memory, None);
                device.raw.destroy_image(image, None);
            }
            return Err(RendererError::Vk {
                context: "bind_image_memory2 (dmabuf import)",
                result: e,
            });
        }

        debug!(
            "imported dmabuf as VkImage {}x{} format={:?} modifier={:#x}",
            dmabuf.width,
            dmabuf.height,
            vk_format,
            u64::from(dmabuf.modifier),
        );

        Ok(Self {
            device,
            image,
            memory,
            extent: vk::Extent2D {
                width: dmabuf.width,
                height: dmabuf.height,
            },
            format: vk_format,
        })
    }
}

impl Drop for ImportedImage {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_image(self.image, None);
            self.device.raw.free_memory(self.memory, None);
        }
    }
}

fn allocate_imported_memory(
    device: &Device,
    image: vk::Image,
    plane_fd: std::os::fd::BorrowedFd<'_>,
) -> Result<vk::DeviceMemory> {
    let mem_req2 = unsafe {
        let mut req = vk::MemoryRequirements2::default();
        let info = vk::ImageMemoryRequirementsInfo2::default().image(image);
        device.raw.get_image_memory_requirements2(&info, &mut req);
        req
    };

    let fd_loader = external_memory_fd::Device::new(device.instance_raw(), &device.raw);

    // Query which memory types accept this fd. The query does NOT consume
    // the fd, but takes a raw i32; dup so we don't borrow the caller's.
    let query_fd: OwnedFd = plane_fd.try_clone_to_owned().map_err(|_| RendererError::Vk {
        context: "dup dmabuf fd for memory-type query",
        result: vk::Result::ERROR_OUT_OF_HOST_MEMORY,
    })?;
    let mut fd_props = vk::MemoryFdPropertiesKHR::default();
    unsafe {
        fd_loader.get_memory_fd_properties(
            vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            query_fd.as_raw_fd(),
            &mut fd_props,
        )
    }
    .vk_ctx("get_memory_fd_properties (dmabuf)")?;
    drop(query_fd);

    let candidate_types = mem_req2.memory_requirements.memory_type_bits & fd_props.memory_type_bits;
    if candidate_types == 0 {
        return Err(RendererError::MissingFeature(
            "no memory type supports both this image and dmabuf import",
        ));
    }
    let mem_type_index = candidate_types.trailing_zeros();

    // Vulkan consumes this fd on success. Dup so the caller's Dmabuf stays
    // valid. On failure, reclaim into an OwnedFd to close it.
    let import_fd: OwnedFd = plane_fd.try_clone_to_owned().map_err(|_| RendererError::Vk {
        context: "dup dmabuf fd for import",
        result: vk::Result::ERROR_OUT_OF_HOST_MEMORY,
    })?;
    let raw_import_fd = import_fd.into_raw_fd();

    let mut import_info = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
        .fd(raw_import_fd);
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_req2.memory_requirements.size)
        .memory_type_index(mem_type_index)
        .push_next(&mut import_info)
        .push_next(&mut dedicated);

    match unsafe { device.raw.allocate_memory(&alloc_info, None) } {
        Ok(m) => Ok(m),
        Err(e) => {
            // Take the fd back so it's properly closed.
            unsafe { OwnedFd::from_raw_fd(raw_import_fd) };
            Err(RendererError::Vk {
                context: "allocate_memory (dmabuf import)",
                result: e,
            })
        }
    }
}
