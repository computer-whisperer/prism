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
    /// YUV plane layout of the bound texture: 0 = RGB (binding 1 unused),
    /// 1 = NV12 (8-bit), 2 = P010 (10-bit). When non-zero the shader
    /// samples the chroma plane at binding 1, range-expands, and applies
    /// `yuv_matrix` to recover nonlinear R′G′B′ before the transfer decode.
    pub yuv: i32,
    /// YUV→RGB coefficient set, by the source's primaries: 0 = BT.709
    /// (also sRGB-primaries SDR video), 1 = BT.2020. Ignored when `yuv == 0`.
    pub yuv_matrix: i32,
    /// Per-output, per-channel panel luminance ceiling, in nits. The
    /// intermediate is display-referred: post-decode values get
    /// clamped per-channel to this. `.a` is unused (vec4 only because
    /// std430 vec3 alignment is awkward in push constants). Defaults
    /// set by `identity_srgb`/`solid` to a very large value (PQ's
    /// 10000-nit max) broadcast to all three so unconfigured callers
    /// get effectively-no-op behaviour; the render-path constructor
    /// overrides per-output with the effective per-channel peaks.
    pub output_peak_nits_rgba: [f32; 4],
    /// Sampled-alpha handling (see [`crate::AlphaMode`] and
    /// `shaders/decode.frag`): 0 = opaque (ignore sampled alpha, force 1.0;
    /// `X`-formats and YUV), 1 = premultiplied (Wayland `A`-format default;
    /// un-premultiply before the transfer EOTF, re-premultiply at output).
    /// Placed last (after the trailing `vec4`) so it needs no std430 alignment
    /// padding.
    pub alpha_mode: i32,
    /// Rounded-corner SDF coverage mode (see `shaders/decode.frag`):
    /// 0 = off (sdf fields ignored), 1 = fill (alpha ×= rounded-box
    /// coverage), 2 = ring (alpha ×= outer − inner coverage; a hollow
    /// border band of per-side thickness `sdf_inset`), 3 = shadow
    /// (Gaussian-blurred rounded box per `sdf_sigma`, minus the
    /// `sdf_box2`/`sdf_radii2` cut-out).
    pub sdf_mode: i32,
    /// Logical size of the output view — lets the vertex shader recover
    /// fragment positions in logical pixels for the SDF. Set centrally by
    /// `lower_elements` on every draw; direct `ElementDraw` constructors
    /// (tests, capture probes) leave it zero, which is fine for
    /// `sdf_mode == 0`. Pairs with `sdf_mode` (two `i32`s then a `vec2`)
    /// so the trailing `vec4`s stay 16-aligned without padding.
    pub view_size_log: [f32; 2],
    /// Rounded box in output-space logical pixels: x_min, y_min, x_max, y_max.
    pub sdf_box: [f32; 4],
    /// Per-corner radii in logical pixels: top-left, top-right, bottom-right,
    /// bottom-left (matches `prism_config::CornerRadius` order).
    pub sdf_radii: [f32; 4],
    /// Ring mode only: per-side band thickness in logical pixels —
    /// top, right, bottom, left (CSS order, matches `BorderEl::thickness`).
    pub sdf_inset: [f32; 4],
    /// Shadow mode only: cut-out rounded box (the window rect the shadow
    /// must not paint behind), logical px min/max. Empty (max ≤ min)
    /// disables the cut-out.
    pub sdf_box2: [f32; 4],
    /// Shadow mode only: per-corner radii of the cut-out box.
    pub sdf_radii2: [f32; 4],
    /// Shadow mode only: Gaussian sigma in logical px (softness / 2, the
    /// CSS box-shadow convention). Below 0.1 → crisp rounded rect.
    pub sdf_sigma: f32,
}

/// The GLSL `Push` block lays out to exactly this size under std430 rules;
/// a drifting Rust-side struct (reordered fields, accidental padding) would
/// silently corrupt every per-element parameter, so pin it. 244 bytes is
/// within the 256-byte `maxPushConstantsSize` of all desktop drivers
/// (RADV / NVIDIA / ANV / llvmpipe).
const _: () = assert!(std::mem::size_of::<DecodePush>() == 244);

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
            yuv: 0,
            yuv_matrix: 0,
            output_peak_nits_rgba: [10_000.0, 10_000.0, 10_000.0, 0.0],
            // Opaque base; the surface path overrides via `SurfaceEl::to_draw`
            // from the buffer fourcc. Direct callers (the #48 smoke test) feed
            // an opaque texture, where forcing alpha = 1.0 is a no-op.
            alpha_mode: 0,
            sdf_mode: 0,
            view_size_log: [0.0; 2],
            sdf_box: [0.0; 4],
            sdf_radii: [0.0; 4],
            sdf_inset: [0.0; 4],
            sdf_box2: [0.0; 4],
            sdf_radii2: [0.0; 4],
            sdf_sigma: 0.0,
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
            yuv: 0,
            yuv_matrix: 0,
            output_peak_nits_rgba: [10_000.0, 10_000.0, 10_000.0, 0.0],
            // 1×1 white texel is opaque (alpha = 1.0); the element's own alpha
            // rides `tint.a`, which is applied independently of this mode.
            alpha_mode: 0,
            sdf_mode: 0,
            view_size_log: [0.0; 2],
            sdf_box: [0.0; 4],
            sdf_radii: [0.0; 4],
            sdf_inset: [0.0; 4],
            sdf_box2: [0.0; 4],
            sdf_radii2: [0.0; 4],
            sdf_sigma: 0.0,
        }
    }
}

fn mat4_identity() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
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
        // Descriptor set layout — combined image-samplers at binding 0 (the
        // primary/luma texture) and binding 1 (the chroma plane for YUV
        // imports; bound to the same view as binding 0 for RGB draws since
        // the shader references it statically). The PUSH_DESCRIPTOR_KHR flag
        // is what lets us bypass the pool.
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
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

        let pipeline =
            build_decode_pipeline(&device, pipeline_layout, vert, frag, intermediate_format)?;

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
    /// Caller writes via `cmd_push_descriptor_set`. `binding` is 0 for the
    /// primary/luma texture, 1 for the chroma plane.
    pub fn write_texture_binding<'a>(
        &self,
        binding: u32,
        image_info: &'a [vk::DescriptorImageInfo; 1],
    ) -> vk::WriteDescriptorSet<'a> {
        vk::WriteDescriptorSet::default()
            .dst_binding(binding)
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
