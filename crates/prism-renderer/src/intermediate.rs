//! Per-output fp16 intermediate image (the BT.2020 absolute-nits buffer).
//!
//! The intermediate sits between the decode pass (writes) and the encode pass
//! (reads). One per active output; sized to match the output's mode. Created
//! eagerly when an output is added and recreated on mode change.

use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::error::{RendererError, Result, VkResultExt};

pub const INTERMEDIATE_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

pub struct Intermediate {
    device: Arc<Device>,
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub extent: vk::Extent2D,
}

impl Intermediate {
    pub fn new(device: Arc<Device>, extent: vk::Extent2D) -> Result<Self> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(INTERMEDIATE_FORMAT)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
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

        let view = create_view(&device, image, INTERMEDIATE_FORMAT)?;

        Ok(Self {
            device,
            image,
            view,
            memory,
            extent,
        })
    }
}

impl Drop for Intermediate {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_image_view(self.view, None);
            self.device.raw.destroy_image(self.image, None);
            self.device.raw.free_memory(self.memory, None);
        }
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

fn pick_device_local_memory(device: &Device, type_bits: u32) -> Result<u32> {
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
