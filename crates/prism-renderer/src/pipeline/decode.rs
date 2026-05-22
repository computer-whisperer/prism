//! Per-element decode pipeline.
//!
//! Draws a single element as a 4-vertex triangle-strip quad onto the fp16/fp32
//! intermediate. Fragment shader applies the input transfer + primaries
//! conversion, writes BT.2020 linear absolute nits.
//!
//! Uses VK_KHR_push_descriptor — bindings are pushed at command-record time,
//! no descriptor pool, no per-frame allocate/free.

use std::sync::Arc;

use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::device::Device;
use crate::error::{Result, VkResultExt};

use super::shader_module;

const VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/decode.vert.spv"));
const FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/decode.frag.spv"));

/// Push constants for the decode pipeline. Mirrors GLSL `Push` in
/// `shaders/decode.frag`. Must be `repr(C)` and `Pod` for bytemuck.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DecodePush {
    pub dst_rect_clip: [f32; 4],
    pub src_rect_uv: [f32; 4],
    pub decode_matrix: [f32; 16], // mat4, column-major
    /// Per-element tint, identity = `[1.0; 4]`. Used by solid-color elements
    /// (window borders, layout backgrounds) that sample the renderer's 1×1
    /// white texture with `transfer = 0` (Linear) and have the actual color
    /// baked into this tint in BT.2020 linear nits.
    pub tint: [f32; 4],
    pub sdr_white_nits: f32,
    pub transfer: i32,
    /// Per-output panel luminance ceiling, in nits. The intermediate
    /// is display-referred: post-decode values get clamped to this.
    /// Defaults set by `identity_srgb`/`solid` to a very large value
    /// (PQ's 10000-nit max) so unconfigured callers get effectively-
    /// no-op behaviour; the render-path constructor overrides per-output.
    pub output_peak_nits: f32,
    pub _pad1: i32,
}

impl DecodePush {
    pub fn identity_srgb(dst: [f32; 4], src: [f32; 4]) -> Self {
        Self {
            dst_rect_clip: dst,
            src_rect_uv: src,
            decode_matrix: mat4_identity(),
            tint: [1.0, 1.0, 1.0, 1.0],
            sdr_white_nits: 80.0,
            // 0 = Linear (no decode). The smoke test feeds an already-linear
            // RGBA16_SFLOAT texture, so this is the right choice for #48.
            transfer: 0,
            output_peak_nits: 10_000.0,
            _pad1: 0,
        }
    }

    /// Solid-color draw: sample the renderer's 1×1 white texture in full,
    /// no decode (Linear transfer), tint with the supplied color in BT.2020
    /// linear nits. Caller supplies the destination clip-space rect.
    pub fn solid(dst: [f32; 4], color_bt2020_nits: [f32; 4]) -> Self {
        Self {
            dst_rect_clip: dst,
            src_rect_uv: [0.0, 0.0, 1.0, 1.0],
            decode_matrix: mat4_identity(),
            tint: color_bt2020_nits,
            sdr_white_nits: 1.0,
            transfer: 0,
            output_peak_nits: 10_000.0,
            _pad1: 0,
        }
    }
}

fn mat4_identity() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// Owns the decode pipeline + its descriptor set layout + sampler. No pool
/// because we push descriptors at draw time.
pub struct DecodePipeline {
    device: Arc<Device>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub sampler: vk::Sampler,
    pub push_loader: ash::khr::push_descriptor::Device,
}

impl DecodePipeline {
    pub fn new(device: Arc<Device>, intermediate_format: vk::Format) -> Result<Self> {
        // Descriptor set layout — combined image-sampler at binding 0. The
        // PUSH_DESCRIPTOR_KHR flag is what lets us bypass the pool.
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
                .vk_ctx("create_descriptor_set_layout (decode)")?;

        let push_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<DecodePush>() as u32)];
        let set_layouts = [descriptor_set_layout];
        let pl_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_range);
        let pipeline_layout = unsafe { device.raw.create_pipeline_layout(&pl_info, None) }
            .vk_ctx("create_pipeline_layout (decode)")?;

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST);
        let sampler = unsafe { device.raw.create_sampler(&sampler_info, None) }
            .vk_ctx("create_sampler (decode)")?;

        let vert = shader_module(&device, VERT_SPV)?;
        let frag = shader_module(&device, FRAG_SPV)?;

        let pipeline = build_decode_pipeline(
            &device,
            pipeline_layout,
            vert,
            frag,
            intermediate_format,
        )?;

        unsafe {
            device.raw.destroy_shader_module(vert, None);
            device.raw.destroy_shader_module(frag, None);
        }

        let push_loader =
            ash::khr::push_descriptor::Device::new(device.instance_raw(), &device.raw);

        Ok(Self {
            device,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            sampler,
            push_loader,
        })
    }

    /// Build the `WriteDescriptorSet` to push for a texture binding.
    /// Caller writes via `cmd_push_descriptor_set`. Useful as a helper
    /// to keep the per-draw recording terse.
    pub fn write_texture_binding<'a>(
        &self,
        image_info: &'a [vk::DescriptorImageInfo; 1],
    ) -> vk::WriteDescriptorSet<'a> {
        vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(image_info)
    }
}

impl Drop for DecodePipeline {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.raw.device_wait_idle();
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

fn build_decode_pipeline(
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

    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dyn_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

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
        .color_blend_state(&cb)
        .dynamic_state(&dyn_state)
        .layout(layout)
        .push_next(&mut dynamic_info);

    let pipelines = unsafe {
        device
            .raw
            .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
    }
    .map_err(|(_, e)| crate::error::RendererError::Vk {
        context: "create_graphics_pipelines (decode)",
        result: e,
    })?;
    Ok(pipelines[0])
}
