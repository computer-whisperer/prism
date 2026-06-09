//! Per-output fp16 intermediate image (the BT.2020 absolute-nits buffer).
//!
//! The intermediate sits between the decode pass (writes) and the encode pass
//! (reads). One per active output; sized to match the output's mode. Created
//! eagerly when an output is added and recreated on mode change.

use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::error::{RendererError, Result, VkResultExt};

/// Default intermediate format: fp32 BT.2020 absolute-nits linear.
///
/// fp32 gives ~7 decimal digits of mantissa and infinite headroom for
/// accumulating semi-transparent layers, calibration matrices that compress
/// dynamic range, and steep-slope encode math (PQ near peak luminance).
///
/// Cost on our hardware is negligible: at 4K@60 a write+read pair burns
/// ~16 GB/s per output — single-digit % of either GPU's memory bandwidth.
/// fp16 stays available for outputs where measurement shows no difference
/// or where VRAM pressure becomes the binding constraint.
pub const DEFAULT_INTERMEDIATE_FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;

pub struct Intermediate {
    device: Arc<Device>,
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
}

impl Intermediate {
    pub fn new(device: Arc<Device>, extent: vk::Extent2D, format: vk::Format) -> Result<Self> {
        let image_info = vk::ImageCreateInfo::default()
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
            // TRANSFER_SRC so the window-close path can copy a tile-sized region
            // out into a `SnapshotTexture` for the close animation.
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.raw.create_image(&image_info, None) }
            .vk_ctx("create_image (intermediate)")?;

        let req = unsafe { device.raw.get_image_memory_requirements(image) };
        let mem_type = pick_device_local_memory(&device, req.memory_type_bits)?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        let memory = unsafe { device.raw.allocate_memory(&alloc, None) }
            .vk_ctx("allocate_memory (intermediate)")?;
        unsafe { device.raw.bind_image_memory(image, memory, 0) }
            .vk_ctx("bind_image_memory (intermediate)")?;

        let view = create_view(&device, image, format)?;

        Ok(Self {
            device,
            image,
            view,
            memory,
            extent,
            format,
        })
    }
}

impl Drop for Intermediate {
    fn drop(&mut self) {
        // Retired, not destroyed: on an in-place realloc
        // (`ensure_intermediate` on extent change) the in-flight previous
        // frame still samples this image; the deferred queue holds it until
        // the slot fences prove otherwise. At teardown `Device::drop` drains
        // after `device_wait_idle`, preserving the old guarantees.
        self.device.retire(crate::device::Retired::Image {
            image: self.image,
            view: self.view,
            memory: self.memory,
        });
    }
}

/// Create a 2D color image view covering all of the image.
pub fn create_view(device: &Device, image: vk::Image, format: vk::Format) -> Result<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    unsafe { device.raw.create_image_view(&info, None) }.vk_ctx("create_image_view")
}

pub(crate) fn pick_device_local_memory(device: &Device, type_bits: u32) -> Result<u32> {
    let props = unsafe {
        device
            .instance_raw()
            .get_physical_device_memory_properties(device.physical.raw)
    };
    for i in 0..props.memory_type_count {
        let mt = props.memory_types[i as usize];
        if (type_bits & (1 << i)) != 0
            && mt
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        {
            return Ok(i);
        }
    }
    Err(RendererError::MissingFeature(
        "no DEVICE_LOCAL memory type for intermediate image",
    ))
}
