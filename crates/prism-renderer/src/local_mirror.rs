//! Target-local mirror image — the OPTIMAL, device-local copy of a cross-GPU
//! mirror's LINEAR GTT import.
//!
//! Background (see `docs/async-render-rework.md`): a mirrored surface is held
//! in a LINEAR, host-visible (GTT) scratch the home GPU fills each commit; the
//! target GPU imports it ([`crate::dmabuf::ImportedImage`]) and historically
//! **sampled that GTT import directly** in the decode/deband passes — two
//! untiled PCIe scans per frame. Instead we copy the GTT import **once** into a
//! [`LocalImage`] in the target's own VRAM (OPTIMAL-tiled) at the start of the
//! render command buffer, and the passes sample that. One streaming copy
//! replaces two scattered remote scans, and the tiled local image restores
//! texture-cache locality.
//!
//! This first cut records the copy on the **graphics** queue (in
//! `Renderer::render_frame`); a later step moves it to an async-compute (ACE)
//! queue so the transfer overlaps graphics.

use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::error::{Result, VkResultExt};
use crate::intermediate::{create_view, pick_device_local_memory};

/// An OPTIMAL, device-local image used as the local destination of a mirror's
/// GTT→VRAM copy. Sampled by the decode/deband passes in place of the LINEAR
/// import. Created `CONCURRENT` across the graphics + transfer queue families
/// when they differ (so a future ACE-queue copy needs no ownership transfer);
/// `EXCLUSIVE` when the GPU exposes a single family.
pub struct LocalImage {
    device: Arc<Device>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    extent: vk::Extent2D,
}

impl LocalImage {
    pub fn new(device: Arc<Device>, extent: vk::Extent2D, format: vk::Format) -> Result<Self> {
        let gfx = device.physical.graphics_queue_family;
        let xfer = device.transfer_queue_family;
        let families: Vec<u32> = if gfx == xfer {
            vec![gfx]
        } else {
            vec![gfx, xfer]
        };
        let sharing = if families.len() > 1 {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(sharing)
            .queue_family_indices(&families)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.raw.create_image(&info, None) }
            .vk_ctx("create_image (local mirror)")?;

        let req = unsafe { device.raw.get_image_memory_requirements(image) };
        let mem_type = pick_device_local_memory(&device, req.memory_type_bits)?;
        let memory = unsafe {
            device.raw.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mem_type),
                None,
            )
        }
        .vk_ctx("allocate_memory (local mirror)")?;
        unsafe { device.raw.bind_image_memory(image, memory, 0) }
            .vk_ctx("bind_image_memory (local mirror)")?;

        let view = create_view(&device, image, format)?;
        Ok(Self {
            device,
            image,
            memory,
            view,
            extent,
        })
    }

    pub fn image(&self) -> vk::Image {
        self.image
    }
    pub fn view(&self) -> vk::ImageView {
        self.view
    }
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }
}

impl Drop for LocalImage {
    fn drop(&mut self) {
        // May be sampled by up to FRAMES_IN_FLIGHT in-flight submissions when a
        // mirror is torn down (surface destroyed / re-imported). Retire so the
        // deferred queue frees the handles once the slot fences prove those
        // submissions complete — same contract as `ImportedImage::drop`.
        self.device.retire(crate::device::Retired::Image {
            image: self.image,
            view: self.view,
            memory: self.memory,
        });
    }
}

/// One LINEAR(GTT)→OPTIMAL(local) image copy to record at the start of a
/// target render's command buffer. Plain Vulkan handles, so the renderer needs
/// no knowledge of the mirror bookkeeping. The submit's wait semaphores (the
/// home→GTT copy-done) gate the whole command buffer, so the copy never reads a
/// half-written GTT scratch.
#[derive(Clone, Copy)]
pub struct LocalMirrorCopy {
    /// LINEAR GTT import (the mirror `target`), in `GENERAL` layout.
    pub src: vk::Image,
    /// OPTIMAL device-local destination (the mirror `target_local`). Fully
    /// overwritten each frame, so its prior contents/layout are irrelevant
    /// (taken as `UNDEFINED`).
    pub dst: vk::Image,
    pub extent: vk::Extent2D,
}
