//! Top-level renderer: orchestrates the two-pass decode→encode pipeline for
//! a single frame, targeting a caller-provided scanout image.
//!
//! Scope today: single output, single element type (texture), SDR sRGB
//! encode. Multi-output and richer element types (solid color, custom
//! shader) follow in #49 + later tasks.

use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::error::{Result, VkResultExt};
use crate::intermediate::{Intermediate, create_view};
use crate::pipeline::decode::{DecodePipeline, DecodePush};
use crate::pipeline::encode::{EncodePipeline, EncodePush};

/// One element to draw in the decode pass.
pub struct ElementDraw {
    /// Sampled texture (must be in SHADER_READ_ONLY_OPTIMAL layout).
    pub texture_view: vk::ImageView,
    pub push: DecodePush,
}

/// Top-level renderer. Owns the two pipelines + a transient command pool.
/// `render_frame` is synchronous (waits for the queue to go idle) — fine
/// for the tracer demo; #49 will introduce real frame pacing + sync2 sems.
pub struct Renderer {
    device: Arc<Device>,
    decode: DecodePipeline,
    encode: EncodePipeline,
    intermediate: Option<Intermediate>,
    /// Scanout format the encode pipeline was built for. Recreating the
    /// pipeline for a new format isn't free; assert callers keep this stable.
    scanout_format: vk::Format,
    /// fp32 / fp16 / etc. — picked at Renderer-instance construction, which
    /// today maps 1:1 to per-output (one Renderer per scanout target).
    /// When multi-output lands this will move into per-output state.
    intermediate_format: vk::Format,
    command_pool: vk::CommandPool,
}

impl Renderer {
    pub fn new(
        device: Arc<Device>,
        scanout_format: vk::Format,
        intermediate_format: vk::Format,
    ) -> Result<Self> {
        let decode = DecodePipeline::new(device.clone(), intermediate_format)?;
        let encode = EncodePipeline::new(device.clone(), scanout_format)?;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.physical.graphics_queue_family)
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            );
        let command_pool = unsafe { device.raw.create_command_pool(&pool_info, None) }
            .vk_ctx("create_command_pool (renderer)")?;

        Ok(Self {
            device,
            decode,
            encode,
            intermediate: None,
            scanout_format,
            intermediate_format,
            command_pool,
        })
    }

    pub fn scanout_format(&self) -> vk::Format {
        self.scanout_format
    }

    pub fn intermediate_format(&self) -> vk::Format {
        self.intermediate_format
    }

    /// Ensure we have an intermediate image of the right size + format.
    /// Recreates on mismatch.
    fn ensure_intermediate(&mut self, extent: vk::Extent2D) -> Result<()> {
        if self.intermediate.as_ref().is_some_and(|i| {
            i.extent.width == extent.width
                && i.extent.height == extent.height
                && i.format == self.intermediate_format
        }) {
            return Ok(());
        }
        self.intermediate = Some(Intermediate::new(
            self.device.clone(),
            extent,
            self.intermediate_format,
        )?);
        Ok(())
    }

    /// Render one frame.
    ///
    /// Arguments:
    /// - `scanout_image`: the destination image (matching `scanout_format`,
    ///   `extent`, and must be in UNDEFINED layout — we transition it).
    /// - `extent`: scanout size in pixels.
    /// - `elements`: per-element draws (decode pass). Empty list is valid;
    ///   you'll get whatever was cleared into the intermediate.
    /// - `encode_push`: per-output encode parameters (CTM, transfer, etc.).
    ///
    /// Leaves the scanout image in `PRESENT_SRC_KHR` (suitable for KMS
    /// scanout) on success. Waits for queue idle before returning.
    pub fn render_frame(
        &mut self,
        scanout_image: vk::Image,
        extent: vk::Extent2D,
        elements: &[ElementDraw],
        encode_push: &EncodePush,
    ) -> Result<()> {
        self.ensure_intermediate(extent)?;
        let intermediate = self.intermediate.as_ref().unwrap();

        // Encode pass needs a descriptor set bound to the intermediate.
        // Both pools are reset at the start so we don't leak sets across frames.
        self.decode.reset_pool()?;
        self.encode.reset_pool()?;
        let encode_set = self.encode.allocate_descriptor_set(intermediate.view)?;

        // Decode descriptor sets — one per element.
        let mut decode_sets = Vec::with_capacity(elements.len());
        for el in elements {
            decode_sets.push(self.decode.allocate_descriptor_set(el.texture_view)?);
        }

        // Build a scanout image view for the dynamic-rendering attachment.
        let scanout_view = create_view(&self.device, scanout_image, self.scanout_format)?;

        let cb = self.alloc_cmd_buffer()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.raw.begin_command_buffer(cb, &begin) }
            .vk_ctx("begin_command_buffer (renderer)")?;

        // ── Decode pass ────────────────────────────────────────────────────
        // Intermediate: UNDEFINED → COLOR_ATTACHMENT_OPTIMAL.
        let pre_intermediate = [
            barrier_image(
                intermediate.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),
        ];
        unsafe {
            self.device.raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&pre_intermediate),
            );
        }

        let color_attach = [vk::RenderingAttachmentInfo::default()
            .image_view(intermediate.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            })];
        let render_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .layer_count(1)
            .color_attachments(&color_attach);
        unsafe {
            self.device.raw.cmd_begin_rendering(cb, &render_info);

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            };
            self.device.raw.cmd_set_viewport(cb, 0, &[viewport]);
            self.device.raw.cmd_set_scissor(cb, 0, &[scissor]);

            self.device
                .raw
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.decode.pipeline);

            for (el, set) in elements.iter().zip(decode_sets.iter()) {
                self.device.raw.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.decode.pipeline_layout,
                    0,
                    &[*set],
                    &[],
                );
                self.device.raw.cmd_push_constants(
                    cb,
                    self.decode.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&el.push),
                );
                // 4 vertices, triangle-strip → 1 quad.
                self.device.raw.cmd_draw(cb, 4, 1, 0, 0);
            }

            self.device.raw.cmd_end_rendering(cb);
        }

        // ── Barrier: intermediate becomes the encode-pass input ────────────
        // COLOR_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL.
        let mid_barrier = [barrier_image(
            intermediate.image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
        )];

        // Also bring the scanout image up: UNDEFINED → COLOR_ATTACHMENT_OPTIMAL.
        let pre_scanout = barrier_image(
            scanout_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        let mid_barriers = [mid_barrier[0], pre_scanout];
        unsafe {
            self.device.raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&mid_barriers),
            );
        }

        // ── Encode pass ───────────────────────────────────────────────────
        let encode_color_attach = [vk::RenderingAttachmentInfo::default()
            .image_view(scanout_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let encode_render_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .layer_count(1)
            .color_attachments(&encode_color_attach);
        unsafe {
            self.device.raw.cmd_begin_rendering(cb, &encode_render_info);
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            };
            self.device.raw.cmd_set_viewport(cb, 0, &[viewport]);
            self.device.raw.cmd_set_scissor(cb, 0, &[scissor]);
            self.device
                .raw
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.encode.pipeline);
            self.device.raw.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.encode.pipeline_layout,
                0,
                &[encode_set],
                &[],
            );
            self.device.raw.cmd_push_constants(
                cb,
                self.encode.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(encode_push),
            );
            self.device.raw.cmd_draw(cb, 3, 1, 0, 0);
            self.device.raw.cmd_end_rendering(cb);
        }

        // ── Final: scanout → GENERAL for the KMS handoff ───────────────────
        // PRESENT_SRC_KHR would require VK_KHR_swapchain (we don't use it —
        // KMS reads the dmabuf directly). GENERAL is the correct non-swapchain
        // layout when handing a Vulkan-written image to a non-Vulkan consumer.
        let final_barrier = [barrier_image(
            scanout_image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::AccessFlags2::empty(),
        )];
        unsafe {
            self.device.raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&final_barrier),
            );
        }

        unsafe { self.device.raw.end_command_buffer(cb) }.vk_ctx("end_command_buffer")?;

        let cb_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
        let submit = [vk::SubmitInfo2::default().command_buffer_infos(&cb_infos)];
        unsafe {
            self.device
                .raw
                .queue_submit2(self.device.graphics_queue, &submit, vk::Fence::null())
        }
        .vk_ctx("queue_submit2 (renderer)")?;
        unsafe { self.device.raw.queue_wait_idle(self.device.graphics_queue) }
            .vk_ctx("queue_wait_idle (renderer)")?;

        unsafe {
            self.device.raw.free_command_buffers(self.command_pool, &[cb]);
            self.device.raw.destroy_image_view(scanout_view, None);
        }
        Ok(())
    }

    fn alloc_cmd_buffer(&self) -> Result<vk::CommandBuffer> {
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cbs = unsafe { self.device.raw.allocate_command_buffers(&info) }
            .vk_ctx("allocate_command_buffers (renderer)")?;
        Ok(cbs[0])
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.raw.device_wait_idle();
            self.device.raw.destroy_command_pool(self.command_pool, None);
        }
    }
}

fn barrier_image(
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
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
