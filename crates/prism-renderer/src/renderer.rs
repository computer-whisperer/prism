//! Top-level renderer: orchestrates the two-pass decode→encode pipeline.
//!
//! Architecture: N frame slots in flight (default 2, matching double-buffered
//! scanout). Each slot owns a persistent command buffer + a fence. Per-frame:
//! wait for the slot's fence (gates resource reuse), reset the command buffer,
//! record the new frame's commands, submit signalling the fence. No
//! `queue_wait_idle`. No descriptor pool — push descriptors at draw time.
//!
//! The CPU can prepare frame N+1 while the GPU executes frame N; only stalls
//! if the GPU falls more than N frames behind. For our gradient at ~7 ms GPU
//! time on a 16.67 ms vblank budget there's plenty of slack, but real
//! workloads (full client compositing) need this overlap.

use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::dmabuf::ImportedImage;
use crate::encode_synth::EncodeConfig;
use crate::error::{Result, VkResultExt};
use crate::intermediate::Intermediate;
use crate::pipeline::decode::{DecodePipeline, DecodePush};
use crate::pipeline::encode::{EncodePipeline, EncodePush};
use crate::upload::ShmTexture;

/// Number of frames the renderer keeps in flight. Matches the scanout BO
/// count in `prism-drm::OutputContext`; the kernel only ever has one flip
/// pending and one currently scanning out, so two frames-worth of GPU
/// resources is the right size.
pub const FRAMES_IN_FLIGHT: usize = 2;

/// One element to draw in the decode pass.
pub struct ElementDraw {
    /// Sampled texture (must be in SHADER_READ_ONLY_OPTIMAL layout).
    pub texture_view: vk::ImageView,
    pub push: DecodePush,
}

/// Per-frame-in-flight resources. Owned by the renderer.
struct FrameSlot {
    cmd_buffer: vk::CommandBuffer,
    /// Signalled by the queue submission for this slot's frame. We wait on
    /// it at the start of the *next* time we use this slot, to ensure the
    /// GPU is done with the previous frame's resources (including the
    /// intermediate image, the scanout image's previous content, etc.).
    /// Created in the signalled state so the first wait is a no-op.
    fence: vk::Fence,
}

pub struct Renderer {
    device: Arc<Device>,
    decode: DecodePipeline,
    encode: EncodePipeline,
    intermediate: Option<Intermediate>,
    /// 1×1 RGBA8 white texture used as the texture binding for solid-color
    /// elements (window borders, layer-shell backgrounds, …). Sampled in
    /// the decode pipeline with `transfer = Linear`, `sdr_white_nits = 1.0`,
    /// and the actual color baked into `DecodePush::tint`. Lives for the
    /// renderer's full lifetime so the view handle is stable across frames.
    white_tex: ShmTexture,
    scanout_format: vk::Format,
    intermediate_format: vk::Format,
    command_pool: vk::CommandPool,
    slots: [FrameSlot; FRAMES_IN_FLIGHT],
    /// Index into `slots` for the *next* frame.
    next_slot: usize,
}

impl Renderer {
    pub fn new(
        device: Arc<Device>,
        scanout_format: vk::Format,
        intermediate_format: vk::Format,
        encode_config: &EncodeConfig,
    ) -> Result<Self> {
        let decode = DecodePipeline::new(device.clone(), intermediate_format)?;
        let encode = EncodePipeline::new(device.clone(), scanout_format, encode_config)?;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.physical.graphics_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.raw.create_command_pool(&pool_info, None) }
            .vk_ctx("create_command_pool (renderer)")?;

        // Allocate all N command buffers in one call.
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(FRAMES_IN_FLIGHT as u32);
        let cbs = unsafe { device.raw.allocate_command_buffers(&alloc_info) }
            .vk_ctx("allocate_command_buffers (renderer slots)")?;

        // One fence per slot, signalled at creation so the first wait is a no-op.
        let fence_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let mut slots = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for cb in cbs {
            let fence = unsafe { device.raw.create_fence(&fence_info, None) }
                .vk_ctx("create_fence (renderer slot)")?;
            slots.push(FrameSlot {
                cmd_buffer: cb,
                fence,
            });
        }
        let slots: [FrameSlot; FRAMES_IN_FLIGHT] = slots
            .try_into()
            .map_err(|_| crate::error::RendererError::MissingFeature("FrameSlot collect"))?;

        // Solid-color element scratch — one 1×1 white texel, uploaded once.
        let mut white_tex = ShmTexture::new(
            device.clone(),
            vk::Extent2D { width: 1, height: 1 },
            vk::Format::R8G8B8A8_UNORM,
        )?;
        white_tex.upload_bytes(&[255, 255, 255, 255], 4)?;

        Ok(Self {
            device,
            decode,
            encode,
            intermediate: None,
            white_tex,
            scanout_format,
            intermediate_format,
            command_pool,
            slots,
            next_slot: 0,
        })
    }

    /// View of the 1×1 RGBA white texture solid-color elements sample.
    /// Pair with `DecodePush::solid(dst, color)`.
    pub fn white_view(&self) -> vk::ImageView {
        self.white_tex.view()
    }

    pub fn scanout_format(&self) -> vk::Format {
        self.scanout_format
    }

    pub fn intermediate_format(&self) -> vk::Format {
        self.intermediate_format
    }

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

    /// Render one frame into `scanout` (which must match `scanout_format`).
    ///
    /// Waits on the next slot's fence (gates against the GPU still using its
    /// resources from N frames ago), records into its command buffer, submits
    /// signalling the fence. Does NOT wait for the GPU to finish — that's
    /// what frames-in-flight gives us. Caller should sequence the page-flip
    /// after this; the page-flip waits implicitly because Vulkan-emitted
    /// images get a sync-fd that the KMS path respects via dmabuf fences.
    /// (For now we still rely on implicit sync; explicit dmabuf fence
    /// integration is a follow-up.)
    pub fn render_frame(
        &mut self,
        scanout: &ImportedImage,
        elements: &[ElementDraw],
        encode_push: &EncodePush,
        tracer: Option<&dyn Fn(&str)>,
    ) -> Result<()> {
        let trace = |s: &str| {
            if let Some(t) = tracer {
                t(s);
            }
        };
        let extent = scanout.extent();
        self.ensure_intermediate(extent)?;
        let intermediate = self.intermediate.as_ref().unwrap();

        let slot_idx = self.next_slot;
        let slot = &self.slots[slot_idx];

        // Wait for this slot's previous use to finish. With N=2 frames in
        // flight and a 60Hz vblank cadence, this is essentially free —
        // the GPU has had ~16ms+ to drain.
        trace(&format!("rf: wait_for_fences slot={slot_idx}"));
        unsafe {
            self.device
                .raw
                .wait_for_fences(&[slot.fence], true, u64::MAX)
        }
        .vk_ctx("wait_for_fences (slot)")?;
        trace("rf: wait_for_fences done");
        unsafe { self.device.raw.reset_fences(&[slot.fence]) }
            .vk_ctx("reset_fences (slot)")?;

        let cb = slot.cmd_buffer;
        unsafe {
            self.device
                .raw
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
        }
        .vk_ctx("reset_command_buffer (slot)")?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.raw.begin_command_buffer(cb, &begin_info) }
            .vk_ctx("begin_command_buffer (renderer)")?;

        // ── Decode pass ────────────────────────────────────────────────────
        let pre_intermediate = [barrier_image(
            intermediate.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        )];

        // Global memory dependency picking up writes made outside our queue
        // submission stream — specifically client-side writes to dmabuf BOs
        // we're about to sample. The dmabuf's implicit-sync resv on radv
        // already gates queue execution on producer fences, but the
        // visibility barrier (MEMORY_WRITE → SHADER_SAMPLED_READ) is still
        // required so the fragment shader sees fresh pixels rather than
        // anything cached from a prior frame's sample.
        let producer_sync = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)];
        unsafe {
            self.device.raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default()
                    .memory_barriers(&producer_sync)
                    .image_memory_barriers(&pre_intermediate),
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
                offset: vk::Offset2D::default(),
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

            for (i, el) in elements.iter().enumerate() {
                trace(&format!("rf: decode el#{i} push_descriptor view={:?}", el.texture_view));
                let image_info = [vk::DescriptorImageInfo::default()
                    .sampler(self.decode.sampler)
                    .image_view(el.texture_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let writes = [self.decode.write_texture_binding(&image_info)];
                self.decode.push_loader.cmd_push_descriptor_set(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.decode.pipeline_layout,
                    0,
                    &writes,
                );
                self.device.raw.cmd_push_constants(
                    cb,
                    self.decode.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&el.push),
                );
                self.device.raw.cmd_draw(cb, 4, 1, 0, 0);
                trace(&format!("rf: decode el#{i} drawn"));
            }

            self.device.raw.cmd_end_rendering(cb);
        }

        // ── Barrier: intermediate → SHADER_READ; scanout → COLOR_ATTACHMENT ─
        let mid_barriers = [
            barrier_image(
                intermediate.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
            ),
            barrier_image(
                scanout.image(),
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
                &vk::DependencyInfo::default().image_memory_barriers(&mid_barriers),
            );
        }

        // ── Encode pass ───────────────────────────────────────────────────
        let encode_color_attach = [vk::RenderingAttachmentInfo::default()
            .image_view(scanout.view())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let encode_render_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
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

            let image_info = [vk::DescriptorImageInfo::default()
                .sampler(self.encode.sampler)
                .image_view(intermediate.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let writes = [self.encode.write_intermediate_binding(&image_info)];
            self.encode.push_loader.cmd_push_descriptor_set(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.encode.pipeline_layout,
                0,
                &writes,
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

        // ── Final: scanout → GENERAL for KMS handoff ──────────────────────
        let final_barrier = [barrier_image(
            scanout.image(),
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
        trace("rf: end_command_buffer");

        let cb_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
        let submit = [vk::SubmitInfo2::default().command_buffer_infos(&cb_infos)];
        trace("rf: queue_submit2 BEFORE");
        unsafe {
            self.device
                .raw
                .queue_submit2(self.device.graphics_queue, &submit, slot.fence)
        }
        .vk_ctx("queue_submit2 (renderer)")?;
        trace("rf: queue_submit2 AFTER");

        // Advance to the next slot. No GPU wait — the next call to
        // render_frame will wait on its slot's fence as needed.
        self.next_slot = (slot_idx + 1) % FRAMES_IN_FLIGHT;
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            // Drain all outstanding work before tearing down the pool / fences.
            let _ = self.device.raw.device_wait_idle();
            for slot in &self.slots {
                self.device.raw.destroy_fence(slot.fence, None);
            }
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
