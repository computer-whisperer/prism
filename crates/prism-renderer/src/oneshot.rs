//! One-shot Vulkan command submission helpers.
//!
//! Owns a transient command pool and offers `record_and_submit` — record a
//! single command buffer, submit, wait for the queue to go idle, free the
//! command buffer. Synchronous; for the tracer / smoke tests, not the hot
//! path.

use std::sync::Arc;

use ash::vk;
use tracing::trace;

use crate::device::Device;
use crate::error::{Result, VkResultExt};

/// A transient command pool tied to the device's graphics queue family.
pub struct OneshotPool {
    device: Arc<Device>,
    pool: vk::CommandPool,
}

impl OneshotPool {
    pub fn new(device: Arc<Device>) -> Result<Self> {
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.physical.graphics_queue_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let pool =
            unsafe { device.raw.create_command_pool(&info, None) }.vk_ctx("create_command_pool")?;
        Ok(Self { device, pool })
    }

    /// Record a command buffer via `record`, submit it, wait for the queue
    /// to go idle, then free the buffer.
    pub fn record_and_submit<F>(&self, record: F) -> Result<()>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer),
    {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cb = unsafe { self.device.raw.allocate_command_buffers(&alloc_info) }
            .vk_ctx("allocate_command_buffers")?[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.raw.begin_command_buffer(cb, &begin_info) }
            .vk_ctx("begin_command_buffer")?;
        record(&self.device.raw, cb);
        unsafe { self.device.raw.end_command_buffer(cb) }.vk_ctx("end_command_buffer")?;

        let cb_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
        let submit = [vk::SubmitInfo2::default().command_buffer_infos(&cb_infos)];
        unsafe {
            self.device
                .raw
                .queue_submit2(self.device.graphics_queue, &submit, vk::Fence::null())
        }
        .vk_ctx("queue_submit2 (oneshot)")?;

        unsafe { self.device.raw.queue_wait_idle(self.device.graphics_queue) }
            .vk_ctx("queue_wait_idle (oneshot)")?;

        unsafe { self.device.raw.free_command_buffers(self.pool, &[cb]) };
        trace!("oneshot submit complete");
        Ok(())
    }
}

impl Drop for OneshotPool {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.raw.device_wait_idle();
            self.device.raw.destroy_command_pool(self.pool, None);
        }
    }
}

/// Build an `ImageMemoryBarrier2` for the common UNDEFINED → TRANSFER_DST →
/// (whatever) transitions used by `cmd_clear_image_to_color`.
pub fn barrier_undef_to_transfer_dst(image: vk::Image) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
        .src_access_mask(vk::AccessFlags2::empty())
        .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
}

pub fn barrier_transfer_dst_to_general(image: vk::Image) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
        .dst_access_mask(vk::AccessFlags2::empty())
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
}

/// Convenience: record a clear-color into `image` (must be UNDEFINED-layout).
/// Leaves the image in GENERAL layout. Uses `cmd_clear_color_image`.
pub fn record_clear_color(
    raw: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    color: vk::ClearColorValue,
) {
    let pre = [barrier_undef_to_transfer_dst(image)];
    let post = [barrier_transfer_dst_to_general(image)];
    let dep_pre = vk::DependencyInfo::default().image_memory_barriers(&pre);
    let dep_post = vk::DependencyInfo::default().image_memory_barriers(&post);
    let range = [vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }];
    unsafe {
        raw.cmd_pipeline_barrier2(cb, &dep_pre);
        raw.cmd_clear_color_image(
            cb,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &color,
            &range,
        );
        raw.cmd_pipeline_barrier2(cb, &dep_post);
    }
}
