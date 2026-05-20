//! Per-element decode pipeline.
//!
//! Draws a single element as a 4-vertex triangle-strip quad onto the fp16
//! intermediate. Fragment shader applies the input transfer + primaries
//! conversion, writes BT.2020 linear absolute nits.

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
    pub sdr_white_nits: f32,
    pub transfer: i32,
    pub _pad0: i32,
    pub _pad1: i32,
}

impl DecodePush {
    pub fn identity_srgb(dst: [f32; 4], src: [f32; 4]) -> Self {
        Self {
            dst_rect_clip: dst,
            src_rect_uv: src,
            decode_matrix: mat4_identity(),
            sdr_white_nits: 80.0,
            // 0 = Linear (no decode). The smoke test feeds an already-linear
            // RGBA16_SFLOAT texture, so this is the right choice for #48.
            transfer: 0,
            _pad0: 0,
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

/// Owns the decode pipeline + its descriptor set layout, sampler, and pool.
/// One per-renderer (not per-output) — the same pipeline draws onto whichever
/// intermediate image we're targeting (via dynamic rendering).
pub struct DecodePipeline {
    device: Arc<Device>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub sampler: vk::Sampler,
    pub descriptor_pool: vk::DescriptorPool,
}

const POOL_MAX_SETS: u32 = 64;

impl DecodePipeline {
    pub fn new(device: Arc<Device>, intermediate_format: vk::Format) -> Result<Self> {
        // Descriptor set layout: one combined image-sampler at binding 0.
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { device.raw.create_descriptor_set_layout(&dsl_info, None) }
                .vk_ctx("create_descriptor_set_layout (decode)")?;

        // Pipeline layout: that DSL + push constants.
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

        // Linear sampler. NEAREST for the gradient test would be fine too; LINEAR
        // is what real surface textures will want for fractional scaling.
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST);
        let sampler = unsafe { device.raw.create_sampler(&sampler_info, None) }
            .vk_ctx("create_sampler (decode)")?;

        // Descriptor pool sized for the max elements we expect per-frame.
        // The smoke test only uses 1; the real compositor will need more.
        // TODO: dynamic sizing once we know real-world element counts.
        let pool_size = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(POOL_MAX_SETS)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(POOL_MAX_SETS)
            .pool_sizes(&pool_size)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let descriptor_pool = unsafe { device.raw.create_descriptor_pool(&pool_info, None) }
            .vk_ctx("create_descriptor_pool (decode)")?;

        let vert = shader_module(&device, VERT_SPV)?;
        let frag = shader_module(&device, FRAG_SPV)?;

        let pipeline = build_decode_pipeline(
            &device,
            pipeline_layout,
            vert,
            frag,
            intermediate_format,
        )?;

        // Shader modules can be destroyed once the pipeline is built.
        unsafe {
            device.raw.destroy_shader_module(vert, None);
            device.raw.destroy_shader_module(frag, None);
        }

        Ok(Self {
            device,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            sampler,
            descriptor_pool,
        })
    }

    /// Allocate + write a descriptor set bound to the given image view. The
    /// caller is responsible for freeing via `vkFreeDescriptorSets` or
    /// resetting the pool between frames.
    pub fn allocate_descriptor_set(&self, image_view: vk::ImageView) -> Result<vk::DescriptorSet> {
        let layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        let sets = unsafe { self.device.raw.allocate_descriptor_sets(&alloc_info) }
            .vk_ctx("allocate_descriptor_sets (decode)")?;
        let set = sets[0];

        let image_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info)];
        unsafe { self.device.raw.update_descriptor_sets(&write, &[]) };
        Ok(set)
    }

    /// Reset the descriptor pool (frees all sets allocated from it). Call
    /// between frames or after every render cycle.
    pub fn reset_pool(&self) -> Result<()> {
        unsafe {
            self.device
                .raw
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
        }
        .vk_ctx("reset_descriptor_pool (decode)")
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
            self.device
                .raw
                .destroy_descriptor_pool(self.descriptor_pool, None);
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

    // Pre-multiplied alpha blend: src + dst * (1 - src_alpha).
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

    // Dynamic rendering: declare the color attachment format directly.
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
