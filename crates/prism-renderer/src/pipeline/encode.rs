//! Output encode pipeline.
//!
//! Full-screen triangle. Samples the fp16 intermediate, applies per-output
//! calibration matrix + transfer encode, writes to the scanout image.

use std::sync::Arc;

use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::device::Device;
use crate::error::{Result, VkResultExt};

use super::shader_module;

const VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/encode.vert.spv"));
const FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/encode.frag.spv"));

/// Output transfer enum. Matches the GLSL `output_transfer` int in
/// `shaders/encode.frag`.
#[derive(Clone, Copy, Debug)]
#[repr(i32)]
pub enum OutputTransfer {
    /// fp16 / 32-bit float scanout — no encoding, just scale to [0,1].
    /// Not used yet; HDR-fp16-scanout is its own task.
    Linear = 0,
    /// sRGB OETF for SDR XRGB8888 / XBGR8888 scanout.
    Srgb = 1,
    /// PQ OETF for HDR A2RGB10 / fp16-PQ scanout. Plumbed but not wired
    /// to a real HDR scanout BO yet.
    Pq = 2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct EncodePush {
    pub cal_matrix: [f32; 16],   // mat4
    pub sdr_white_nits: f32,
    pub target_peak_nits: f32,
    pub output_transfer: i32,
    pub _pad: i32,
}

impl EncodePush {
    pub fn sdr_identity() -> Self {
        Self {
            cal_matrix: mat4_identity(),
            sdr_white_nits: 80.0,
            target_peak_nits: 80.0,
            output_transfer: OutputTransfer::Srgb as i32,
            _pad: 0,
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

pub struct EncodePipeline {
    device: Arc<Device>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub sampler: vk::Sampler,
    pub descriptor_pool: vk::DescriptorPool,
}

const POOL_MAX_SETS: u32 = 8;

impl EncodePipeline {
    pub fn new(device: Arc<Device>, scanout_format: vk::Format) -> Result<Self> {
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            unsafe { device.raw.create_descriptor_set_layout(&dsl_info, None) }
                .vk_ctx("create_descriptor_set_layout (encode)")?;

        let push_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<EncodePush>() as u32)];
        let set_layouts = [descriptor_set_layout];
        let pl_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_range);
        let pipeline_layout = unsafe { device.raw.create_pipeline_layout(&pl_info, None) }
            .vk_ctx("create_pipeline_layout (encode)")?;

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST);
        let sampler = unsafe { device.raw.create_sampler(&sampler_info, None) }
            .vk_ctx("create_sampler (encode)")?;

        let pool_size = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(POOL_MAX_SETS)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(POOL_MAX_SETS)
            .pool_sizes(&pool_size)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let descriptor_pool = unsafe { device.raw.create_descriptor_pool(&pool_info, None) }
            .vk_ctx("create_descriptor_pool (encode)")?;

        let vert = shader_module(&device, VERT_SPV)?;
        let frag = shader_module(&device, FRAG_SPV)?;
        let pipeline =
            build_encode_pipeline(&device, pipeline_layout, vert, frag, scanout_format)?;
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

    pub fn allocate_descriptor_set(&self, image_view: vk::ImageView) -> Result<vk::DescriptorSet> {
        let layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        let sets = unsafe { self.device.raw.allocate_descriptor_sets(&alloc_info) }
            .vk_ctx("allocate_descriptor_sets (encode)")?;
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

    pub fn reset_pool(&self) -> Result<()> {
        unsafe {
            self.device
                .raw
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
        }
        .vk_ctx("reset_descriptor_pool (encode)")
    }
}

impl Drop for EncodePipeline {
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

fn build_encode_pipeline(
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
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
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
    // No blending — encode writes the final framebuffer.
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
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
        context: "create_graphics_pipelines (encode)",
        result: e,
    })?;
    Ok(pipelines[0])
}
