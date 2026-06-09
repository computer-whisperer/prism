//! A standalone copy of a tile-sized region of the intermediate, captured for
//! the window-close animation.
//!
//! When a window unmaps, its last composited frame still lives in the
//! persistent intermediate (BT.2020 absolute-nits, premultiplied). The close
//! animation needs to keep drawing that after the real surface is gone, so
//! `render_frame` copies the tile's region out of the intermediate into one of
//! these the frame the window is removed (before the decode pass repaints over
//! it). The copy stays in intermediate space, so replaying it is just a
//! pass-through decode draw (Linear transfer, identity primaries,
//! `sdr_white = 1.0`) — no re-decode, full HDR fidelity.
//!
//! Lifetime: held (via `Arc`) by the layout's `ClosingWindow` for the duration
//! of the animation (~250 ms). `Drop` retires the image into the device's
//! deferred-destroy queue, which frees it once the frame-slot fences prove no
//! in-flight frame still samples it.

use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::error::{Result, VkResultExt};
use crate::intermediate::{create_view, pick_device_local_memory};

/// Snapshot storage format. Deliberately fp16, NOT the intermediate's fp32:
/// the replay samples the snapshot **scaled** (real linear interpolation), and
/// `R32G32B32A32_SFLOAT` lacks `SAMPLED_IMAGE_FILTER_LINEAR` on common GPUs
/// (radv/AMD) — filtering it is undefined behaviour (garbage + GPU faults).
/// `R16G16B16A16_SFLOAT` supports linear filtering everywhere and holds the
/// intermediate's absolute-nits range with ample precision for a transient
/// animation. The capture is a format-converting blit (fp32 → fp16), not a copy.
pub const SNAPSHOT_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

pub struct SnapshotTexture {
    device: Arc<Device>,
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    extent: vk::Extent2D,
}

impl std::fmt::Debug for SnapshotTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SnapshotTexture")
            .field("extent", &self.extent)
            .finish_non_exhaustive()
    }
}

impl SnapshotTexture {
    /// Allocate a DEVICE_LOCAL `TRANSFER_DST | SAMPLED` [`SNAPSHOT_FORMAT`]
    /// (fp16) image of `extent`, ready to receive a format-converting blit from
    /// the fp32 intermediate.
    pub fn new(device: Arc<Device>, extent: vk::Extent2D) -> Result<Self> {
        let format = SNAPSHOT_FORMAT;
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width.max(1),
                height: extent.height.max(1),
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.raw.create_image(&image_info, None) }
            .vk_ctx("create_image (snapshot)")?;

        let req = unsafe { device.raw.get_image_memory_requirements(image) };
        let mem_type = pick_device_local_memory(&device, req.memory_type_bits)?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        let memory = unsafe { device.raw.allocate_memory(&alloc, None) }
            .vk_ctx("allocate_memory (snapshot)")?;
        unsafe { device.raw.bind_image_memory(image, memory, 0) }
            .vk_ctx("bind_image_memory (snapshot)")?;

        let view = create_view(&device, image, format)?;

        Ok(Self {
            device,
            image,
            view,
            memory,
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

impl Drop for SnapshotTexture {
    fn drop(&mut self) {
        // Retired, not destroyed: the deferred queue frees the image once the
        // frame-slot fences prove no in-flight frame still samples it. (The
        // old `device_wait_idle` here cost a full GPU drain at the end of
        // every close animation.)
        self.device.retire(crate::device::Retired::Image {
            image: self.image,
            view: self.view,
            memory: self.memory,
        });
    }
}
