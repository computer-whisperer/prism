//! Deband pre-pass: a separable Gaussian blur of an element's source codes.
//!
//! Runs before the decode pass. For each element flagged for debanding, two
//! passes (horizontal then vertical) blur the source texture in code space
//! into an fp16 scratch copy. The decode pass then samples that copy with the
//! element's existing UV and clamps it to ±0.5 LSB of the original code (see
//! `shaders/decode.frag`), so the 8-bit value never changes but smooth
//! gradients gain sub-LSB precision.
//!
//! Cost controls (per [`blur`](DebandPipeline::blur)): the blur runs on a
//! `1/downsample`-scale copy reached by repeated 2× bilinear halving (the
//! decode sampler upsamples the low-res scratch for free at clamp time), and
//! the shader pairs taps via linear sampling. The blur is gated by the
//! caller on the frame's damage rects so static elements aren't re-blurred.
//!
//! Scratch images are pooled per frame-slot: `begin_frame(slot)` frees that
//! slot's pool (safe because `render_frame` already waited the slot fence
//! before recording), and `blur` acquires from it. Steady state reuses the
//! same images frame to frame — no per-frame allocation once warmed.

use std::sync::Arc;

use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::device::{Device, Retired};
use crate::error::{Result, VkResultExt};
use crate::intermediate::{create_view, pick_device_local_memory};

use super::shader_module;

const VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/deband_blur.vert.spv"));
const FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/deband_blur.frag.spv"));

/// fp16 is ample for code-space [0,1] values: its mantissa resolves well
/// below the 0.5/255 clamp envelope everywhere in range, at half the
/// bandwidth of the fp32 intermediate.
const SCRATCH_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Cap the kernel half-width so the shader loop is bounded regardless of a
/// misconfigured strength. 128 covers σ≈32 (4·σ), past our measured knee.
const MAX_RADIUS: i32 = 128;

/// Push constants for one blur axis. Mirrors `Push` in `deband_blur.frag`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurPush {
    axis: [f32; 2],
    sigma: f32,
    radius: i32,
}

struct Scratch {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    extent: vk::Extent2D,
    in_use: bool,
}

#[derive(Default)]
struct ScratchPool {
    items: Vec<Scratch>,
}

pub struct DebandPipeline {
    device: Arc<Device>,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    push_loader: ash::khr::push_descriptor::Device,
    /// One pool per frame-slot; freed at `begin_frame` for that slot.
    pools: Vec<ScratchPool>,
}

impl DebandPipeline {
    pub fn new(device: Arc<Device>, frames_in_flight: usize) -> Result<Self> {
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR);
        let descriptor_set_layout =
            unsafe { device.raw.create_descriptor_set_layout(&dsl_info, None) }
                .vk_ctx("create_descriptor_set_layout (deband)")?;

        let push_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<BlurPush>() as u32)];
        let set_layouts = [descriptor_set_layout];
        let pl_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_range);
        let pipeline_layout = unsafe { device.raw.create_pipeline_layout(&pl_info, None) }
            .vk_ctx("create_pipeline_layout (deband)")?;

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST);
        let sampler = unsafe { device.raw.create_sampler(&sampler_info, None) }
            .vk_ctx("create_sampler (deband)")?;

        let vert = shader_module(&device, VERT_SPV)?;
        let frag = shader_module(&device, FRAG_SPV)?;
        let pipeline = build_pipeline(&device, pipeline_layout, vert, frag, SCRATCH_FORMAT)?;
        unsafe {
            device.raw.destroy_shader_module(vert, None);
            device.raw.destroy_shader_module(frag, None);
        }

        let push_loader =
            ash::khr::push_descriptor::Device::new(device.instance_raw(), &device.raw);

        let mut pools = Vec::with_capacity(frames_in_flight);
        pools.resize_with(frames_in_flight, ScratchPool::default);

        Ok(Self {
            device,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            sampler,
            push_loader,
            pools,
        })
    }

    /// Free `slot`'s scratch pool for reuse. Safe to drop the GPU references:
    /// `render_frame` waited this slot's fence before recording, so any prior
    /// frame that used these images has completed.
    pub fn begin_frame(&mut self, slot: usize) {
        for s in &mut self.pools[slot].items {
            s.in_use = false;
        }
    }

    /// Record the deband blur for one element. With `downsample > 1` the wide
    /// Gaussian is computed on a `1/downsample`-scale copy (reached by repeated
    /// 2× bilinear halving) — the decode sampler bilinearly upsamples the
    /// returned low-res scratch for free at clamp time, so quality holds while
    /// cost drops ~`downsample²`. Returns the final blurred view (layout
    /// `SHADER_READ_ONLY_OPTIMAL`). `source_view` must already be
    /// `SHADER_READ_ONLY_OPTIMAL`.
    pub fn blur(
        &mut self,
        cb: vk::CommandBuffer,
        slot: usize,
        source_view: vk::ImageView,
        extent: vk::Extent2D,
        sigma: f32,
        downsample: u32,
    ) -> Result<vk::ImageView> {
        // Downsample chain: halve via a radius-0 (bilinear-blit) pass until we
        // reach the 1/downsample scale (rounded down to a power of two). A
        // half-res blit samples each 2×2 source block's exact average.
        let halvings = downsample.max(1).ilog2();
        let mut cur_view = source_view;
        let mut cur_extent = extent;
        for _ in 0..halvings {
            let next = vk::Extent2D {
                width: (cur_extent.width / 2).max(1),
                height: (cur_extent.height / 2).max(1),
            };
            let (img, view) = self.acquire(slot, next)?;
            self.record_pass(
                cb,
                img,
                view,
                cur_view,
                next,
                BlurPush {
                    axis: [0.0, 0.0],
                    sigma: 1.0,
                    radius: 0,
                },
            );
            cur_view = view;
            cur_extent = next;
        }

        // σ scales down with the resolution; clamp ≥ 0.5 so a small low-res σ
        // still produces a valid (if mild) kernel. Radius is rounded up to an
        // even value so the linear-sampling shader pairs every tap exactly.
        let eff_d = (1u32 << halvings) as f32;
        let sigma_low = (sigma / eff_d).max(0.5);
        let r = (4.0 * sigma_low).ceil() as i32;
        let radius = ((r + 1) & !1).clamp(2, MAX_RADIUS);

        let (img_a, view_a) = self.acquire(slot, cur_extent)?;
        let (img_b, view_b) = self.acquire(slot, cur_extent)?;

        // Horizontal: cur → A.
        self.record_pass(
            cb,
            img_a,
            view_a,
            cur_view,
            cur_extent,
            BlurPush {
                axis: [1.0 / cur_extent.width as f32, 0.0],
                sigma: sigma_low,
                radius,
            },
        );
        // Vertical: A → B.
        self.record_pass(
            cb,
            img_b,
            view_b,
            view_a,
            cur_extent,
            BlurPush {
                axis: [0.0, 1.0 / cur_extent.height as f32],
                sigma: sigma_low,
                radius,
            },
        );
        Ok(view_b)
    }

    /// Record one axis: transition `dst` to color-attachment, render the blur
    /// sampling `src_view`, then transition `dst` to shader-read.
    fn record_pass(
        &self,
        cb: vk::CommandBuffer,
        dst_image: vk::Image,
        dst_view: vk::ImageView,
        src_view: vk::ImageView,
        extent: vk::Extent2D,
        push: BlurPush,
    ) {
        // Old contents are fully overwritten → UNDEFINED old layout.
        self.image_barrier(
            cb,
            dst_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );

        let area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        };
        let color_attach = [vk::RenderingAttachmentInfo::default()
            .image_view(dst_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let render_info = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(&color_attach);
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let src_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(src_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = [vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&src_info)];

        unsafe {
            self.device.raw.cmd_begin_rendering(cb, &render_info);
            self.device
                .raw
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.raw.cmd_set_viewport(cb, 0, &[viewport]);
            self.device.raw.cmd_set_scissor(cb, 0, &[area]);
            self.push_loader.cmd_push_descriptor_set(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &write,
            );
            self.device.raw.cmd_push_constants(
                cb,
                self.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            self.device.raw.cmd_draw(cb, 4, 1, 0, 0);
            self.device.raw.cmd_end_rendering(cb);
        }

        // Make the result sampleable by the next pass / the decode pass.
        self.image_barrier(
            cb,
            dst_image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn image_barrier(
        &self,
        cb: vk::CommandBuffer,
        image: vk::Image,
        old: vk::ImageLayout,
        new: vk::ImageLayout,
        src_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
    ) {
        let barrier = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .old_layout(old)
            .new_layout(new)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })];
        unsafe {
            self.device.raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&barrier),
            );
        }
    }

    /// Get a free scratch image of `extent` from `slot`'s pool, allocating or
    /// re-sizing as needed. Returns `(image, view)` and marks the entry in-use.
    fn acquire(&mut self, slot: usize, extent: vk::Extent2D) -> Result<(vk::Image, vk::ImageView)> {
        let pool = &mut self.pools[slot];
        // Reuse a free, same-size entry.
        if let Some(s) = pool
            .items
            .iter_mut()
            .find(|s| !s.in_use && s.extent == extent)
        {
            s.in_use = true;
            return Ok((s.image, s.view));
        }
        // Re-purpose a free, wrong-size entry (retire its old resources).
        if let Some(s) = pool.items.iter_mut().find(|s| !s.in_use) {
            self.device.retire(Retired::Image {
                image: s.image,
                view: s.view,
                memory: s.memory,
            });
            let (image, memory, view) = create_scratch(&self.device, extent)?;
            *s = Scratch {
                image,
                memory,
                view,
                extent,
                in_use: true,
            };
            return Ok((image, view));
        }
        // None free — grow the pool.
        let (image, memory, view) = create_scratch(&self.device, extent)?;
        pool.items.push(Scratch {
            image,
            memory,
            view,
            extent,
            in_use: true,
        });
        Ok((image, view))
    }
}

impl Drop for DebandPipeline {
    fn drop(&mut self) {
        self.device.wait_device_idle();
        unsafe {
            for pool in &self.pools {
                for s in &pool.items {
                    self.device.raw.destroy_image_view(s.view, None);
                    self.device.raw.destroy_image(s.image, None);
                    self.device.raw.free_memory(s.memory, None);
                }
            }
            self.device.raw.destroy_pipeline(self.pipeline, None);
            self.device
                .raw
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .raw
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.raw.destroy_sampler(self.sampler, None);
        }
    }
}

fn create_scratch(
    device: &Device,
    extent: vk::Extent2D,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(SCRATCH_FORMAT)
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
        .vk_ctx("create_image (deband scratch)")?;

    let req = unsafe { device.raw.get_image_memory_requirements(image) };
    let mem_type = pick_device_local_memory(device, req.memory_type_bits)?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.raw.allocate_memory(&alloc, None) }
        .vk_ctx("allocate_memory (deband scratch)")?;
    unsafe { device.raw.bind_image_memory(image, memory, 0) }
        .vk_ctx("bind_image_memory (deband scratch)")?;

    let view = create_view(device, image, SCRATCH_FORMAT)?;
    Ok((image, memory, view))
}

fn build_pipeline(
    device: &Device,
    layout: vk::PipelineLayout,
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
    color_format: vk::Format,
) -> Result<vk::Pipeline> {
    let entry = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert)
            .name(entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag)
            .name(entry),
    ];
    let vi = vk::PipelineVertexInputStateCreateInfo::default();
    let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_STRIP);
    let viewports = [vk::Viewport::default()];
    let scissors = [vk::Rect2D::default()];
    let vp = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    let rs = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let ms = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    // No blending — the blur overwrites the scratch.
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let cb_state = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dyn_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let color_formats = [color_format];
    let mut dynamic_info =
        vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vi)
        .input_assembly_state(&ia)
        .viewport_state(&vp)
        .rasterization_state(&rs)
        .multisample_state(&ms)
        .color_blend_state(&cb_state)
        .dynamic_state(&dyn_state)
        .layout(layout)
        .push_next(&mut dynamic_info);

    let pipelines = unsafe {
        device
            .raw
            .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
    }
    .map_err(|(_, e)| crate::error::RendererError::Vk {
        context: "create_graphics_pipelines (deband)",
        result: e,
    })?;
    Ok(pipelines[0])
}
