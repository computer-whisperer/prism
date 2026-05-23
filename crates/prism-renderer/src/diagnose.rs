//! GPU readback probe for the encode pipeline.
//!
//! Lets a calibration tool answer "does the LUT path actually produce
//! what I think it produces?" without trusting the colorimeter to be
//! the source of truth. Given an input in linear-nits BT.2020 (the
//! intermediate's domain), the probe runs the per-output encode
//! pipeline (PQ shaper → 3D LUT → OutputTransfer) against a tiny
//! offscreen image, reads it back, and decodes the scanout format
//! back to linear nits. The caller compares that against
//! `trilinear_sample_lut(entries, pq_oetf(input))` to attribute any
//! verify-time discrepancy to either the shader/LUT path or the
//! panel itself.
//!
//! Why a separate offscreen image rather than reading the real scanout
//! BO: the scanout is owned by KMS between vblanks. Coordinating a
//! readback with the page-flip cycle is fiddly and easy to race;
//! a dedicated 1×1 offscreen we always own is much simpler and the
//! shader behaviour is identical at any image size.

use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::encode_synth::EncodePushSynth as EncodePush;
use crate::error::{RendererError, Result, VkResultExt};
use crate::lut3d::{Lut3dTexture, pq_eotf};
use crate::oneshot::OneshotPool;
use crate::pipeline::encode::EncodePipeline;

/// Format-aware decoded readback value. RGB per-channel in linear cd/m².
pub type DiagnosedNits = [f64; 3];

/// 1×1 offscreen scratch for the encode-pipeline readback. All Vulkan
/// resources sized for a single pixel — encode shader produces the
/// same value at any sample position so a 1×1 sink is sufficient.
pub struct EncodeDiagnoseProbe {
    device: Arc<Device>,
    oneshot: OneshotPool,

    intermediate_format: vk::Format,
    scanout_format: vk::Format,

    // ─── 1×1 intermediate (encode pipeline input) ─────────────────────
    intermediate_image: vk::Image,
    intermediate_view: vk::ImageView,
    intermediate_memory: vk::DeviceMemory,

    // ─── 1×1 scanout-format offscreen (encode pipeline output) ────────
    offscreen_image: vk::Image,
    offscreen_view: vk::ImageView,
    offscreen_memory: vk::DeviceMemory,

    // ─── Staging buffers (host-visible, persistently mapped) ──────────
    /// Upload: f32 RGBA (16 bytes for one pixel of fp32 intermediate)
    /// or whatever the intermediate format demands.
    upload_buffer: vk::Buffer,
    upload_memory: vk::DeviceMemory,
    upload_ptr: *mut u8,
    upload_size: vk::DeviceSize,
    /// Readback: bytes of the scanout format for one pixel.
    readback_buffer: vk::Buffer,
    readback_memory: vk::DeviceMemory,
    readback_ptr: *mut u8,
    readback_size: vk::DeviceSize,
}

// Persistently-mapped host pointers — same Send/Sync rationale as
// ShmTexture: per-instance pointers, only touched under &mut self.
unsafe impl Send for EncodeDiagnoseProbe {}
unsafe impl Sync for EncodeDiagnoseProbe {}

impl EncodeDiagnoseProbe {
    pub fn new(
        device: Arc<Device>,
        intermediate_format: vk::Format,
        scanout_format: vk::Format,
    ) -> Result<Self> {
        let intermediate_texel_bytes = format_texel_bytes(intermediate_format)?;
        let scanout_texel_bytes = format_texel_bytes(scanout_format)?;

        // ── Allocate the 1×1 intermediate (sampled by encode shader) ──
        let (intermediate_image, intermediate_memory) = create_image_1x1(
            &device,
            intermediate_format,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let intermediate_view = create_view_1x1(&device, intermediate_image, intermediate_format)?;

        // ── Allocate the 1×1 offscreen (color attachment + readback) ──
        let (offscreen_image, offscreen_memory) = create_image_1x1(
            &device,
            scanout_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let offscreen_view = create_view_1x1(&device, offscreen_image, scanout_format)?;

        // ── Staging: upload + readback, host-visible/coherent ─────────
        let upload_size = intermediate_texel_bytes as vk::DeviceSize;
        let (upload_buffer, upload_memory, upload_ptr, upload_size) = create_host_buffer(
            &device,
            upload_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        let readback_size = scanout_texel_bytes as vk::DeviceSize;
        let (readback_buffer, readback_memory, readback_ptr, readback_size) = create_host_buffer(
            &device,
            readback_size,
            vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        let oneshot = OneshotPool::new(device.clone())?;

        Ok(Self {
            device,
            oneshot,
            intermediate_format,
            scanout_format,
            intermediate_image,
            intermediate_view,
            intermediate_memory,
            offscreen_image,
            offscreen_view,
            offscreen_memory,
            upload_buffer,
            upload_memory,
            upload_ptr,
            upload_size,
            readback_buffer,
            readback_memory,
            readback_ptr,
            readback_size,
        })
    }

    /// Run the encode pipeline against a 1×1 scratch with `input_nits`
    /// as the (linear, BT.2020) intermediate value, read back the
    /// scanout-format output, and decode it to linear nits.
    ///
    /// `encode_push` should match what the live render path uses for
    /// this output — target_peak_nits + sdr_white_nits influence the
    /// OutputTransfer stage and so the decoded result.
    ///
    /// `lut3d` is the per-output 3D LUT bound at descriptor set 0
    /// binding 1 when the encode chain includes `EncodeFragment::Lut3d`.
    /// Pass the live one so the diagnose mirrors the live shader exactly.
    pub fn diagnose(
        &mut self,
        encode: &EncodePipeline,
        encode_push: &EncodePush,
        lut3d: Option<&Lut3dTexture>,
        input_nits: [f64; 3],
    ) -> Result<DiagnosedNits> {
        // 1) Stage the input into the upload buffer, then copy to the
        //    1×1 intermediate. UNDEFINED → TRANSFER_DST → SAMPLED.
        self.upload_intermediate(input_nits)?;

        // 2) Run the encode pipeline against the 1×1 offscreen.
        self.run_encode(encode, encode_push, lut3d)?;

        // 3) Copy the offscreen → readback buffer + read on CPU.
        // SDR formats need `sdr_white_nits` for the scanout-to-nits
        // inverse (sRGB EOTF returns [0,1] linear, then ×nits); HDR
        // PQ formats decode straight to nits and ignore the param.
        // Source it from the encode_push the live render path uses
        // so the decode matches what the shader actually emitted.
        self.readback_and_decode(encode_push.sdr_white_nits as f64)
    }

    /// Push `input_nits` into the 1×1 intermediate image. The
    /// intermediate format determines the on-GPU layout — `R32G32B32A32_SFLOAT`
    /// is the only one we currently emit so we hard-code that path;
    /// add new format variants as the renderer grows.
    fn upload_intermediate(&mut self, input_nits: [f64; 3]) -> Result<()> {
        // SAFETY: persistently-mapped HOST_COHERENT buffer; only
        // touched under &mut self; no concurrent GPU read because we
        // wait on each oneshot submit before the next access.
        unsafe {
            let dst = self.upload_ptr;
            match self.intermediate_format {
                vk::Format::R32G32B32A32_SFLOAT => {
                    let r = input_nits[0] as f32;
                    let g = input_nits[1] as f32;
                    let b = input_nits[2] as f32;
                    let a = 1.0_f32;
                    std::ptr::copy_nonoverlapping(r.to_le_bytes().as_ptr(), dst, 4);
                    std::ptr::copy_nonoverlapping(g.to_le_bytes().as_ptr(), dst.add(4), 4);
                    std::ptr::copy_nonoverlapping(b.to_le_bytes().as_ptr(), dst.add(8), 4);
                    std::ptr::copy_nonoverlapping(a.to_le_bytes().as_ptr(), dst.add(12), 4);
                }
                vk::Format::R16G16B16A16_SFLOAT => {
                    let r = half::f16::from_f64(input_nits[0]);
                    let g = half::f16::from_f64(input_nits[1]);
                    let b = half::f16::from_f64(input_nits[2]);
                    let a = half::f16::from_f32(1.0);
                    std::ptr::copy_nonoverlapping(r.to_le_bytes().as_ptr(), dst, 2);
                    std::ptr::copy_nonoverlapping(g.to_le_bytes().as_ptr(), dst.add(2), 2);
                    std::ptr::copy_nonoverlapping(b.to_le_bytes().as_ptr(), dst.add(4), 2);
                    std::ptr::copy_nonoverlapping(a.to_le_bytes().as_ptr(), dst.add(6), 2);
                }
                _ => {
                    return Err(RendererError::MissingFeature(
                        "EncodeDiagnoseProbe: unsupported intermediate format",
                    ));
                }
            }
        }

        let upload_buffer = self.upload_buffer;
        let intermediate_image = self.intermediate_image;
        self.oneshot.record_and_submit(|raw, cb| unsafe {
            let to_xfer = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(intermediate_image)
                .subresource_range(color_subresource_range())];
            raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&to_xfer),
            );

            let region = [vk::BufferImageCopy::default()
                .buffer_offset(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D { width: 1, height: 1, depth: 1 })];
            raw.cmd_copy_buffer_to_image(
                cb,
                upload_buffer,
                intermediate_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &region,
            );

            let to_sampled = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(intermediate_image)
                .subresource_range(color_subresource_range())];
            raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&to_sampled),
            );
        })?;
        let _ = self.upload_size;
        Ok(())
    }

    /// Run the encode pipeline once against the 1×1 offscreen image
    /// with the 1×1 intermediate as source.
    fn run_encode(
        &mut self,
        encode: &EncodePipeline,
        encode_push: &EncodePush,
        lut3d: Option<&Lut3dTexture>,
    ) -> Result<()> {
        let offscreen_image = self.offscreen_image;
        let offscreen_view = self.offscreen_view;
        let intermediate_view = self.intermediate_view;
        let sampler = encode.sampler;
        let pipeline = encode.pipeline;
        let pipeline_layout = encode.pipeline_layout;
        let push_loader = encode.push_loader.clone();
        let uses_lut3d = encode.uses_lut3d;

        self.oneshot.record_and_submit(|raw, cb| unsafe {
            // Transition offscreen UNDEFINED → COLOR_ATTACHMENT_OPTIMAL.
            let pre = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(offscreen_image)
                .subresource_range(color_subresource_range())];
            raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&pre),
            );

            // Dynamic rendering against the 1×1 offscreen.
            let color_attach = [vk::RenderingAttachmentInfo::default()
                .image_view(offscreen_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)];
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D { width: 1, height: 1 },
                })
                .layer_count(1)
                .color_attachments(&color_attach);
            raw.cmd_begin_rendering(cb, &rendering_info);

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            raw.cmd_set_viewport(cb, 0, &[viewport]);
            raw.cmd_set_scissor(
                cb,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D { width: 1, height: 1 },
                }],
            );
            raw.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);

            // Push descriptors: binding 0 = 1×1 intermediate, binding 1
            // = LUT if encode chain uses it. Mirrors the live render
            // path exactly.
            let intermediate_info = [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(intermediate_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let lut_info = lut3d.map(|lut| {
                [vk::DescriptorImageInfo::default()
                    .sampler(sampler)
                    .image_view(lut.view())
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
            });
            let mut writes = vec![encode.write_intermediate_binding(&intermediate_info)];
            if let Some(ref info) = lut_info {
                if uses_lut3d {
                    writes.push(encode.write_lut3d_binding(info));
                }
            }
            push_loader.cmd_push_descriptor_set(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &writes,
            );

            raw.cmd_push_constants(
                cb,
                pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(encode_push),
            );
            raw.cmd_draw(cb, 3, 1, 0, 0);
            raw.cmd_end_rendering(cb);

            // COLOR_ATTACHMENT_OPTIMAL → TRANSFER_SRC_OPTIMAL for the readback copy.
            let post = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(offscreen_image)
                .subresource_range(color_subresource_range())];
            raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&post),
            );
        })?;
        Ok(())
    }

    /// Copy the 1×1 offscreen to the readback buffer, read it on CPU,
    /// decode the format back to linear cd/m². See
    /// [`decode_scanout_texel`] for the `sdr_white_nits` semantics.
    fn readback_and_decode(&mut self, sdr_white_nits: f64) -> Result<DiagnosedNits> {
        let offscreen_image = self.offscreen_image;
        let readback_buffer = self.readback_buffer;
        self.oneshot.record_and_submit(|raw, cb| unsafe {
            let region = [vk::BufferImageCopy::default()
                .buffer_offset(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D { width: 1, height: 1, depth: 1 })];
            raw.cmd_copy_image_to_buffer(
                cb,
                offscreen_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback_buffer,
                &region,
            );
        })?;

        // SAFETY: HOST_COHERENT readback memory, just submitted-and-
        // waited copy into it. Read texel and decode.
        let raw = unsafe { std::slice::from_raw_parts(self.readback_ptr, self.readback_size as usize) };
        let nits = decode_scanout_texel(self.scanout_format, raw, sdr_white_nits)?;
        let _ = self.readback_size;
        Ok(nits)
    }
}

impl Drop for EncodeDiagnoseProbe {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.raw.device_wait_idle();
            self.device.raw.unmap_memory(self.upload_memory);
            self.device.raw.unmap_memory(self.readback_memory);
            self.device.raw.destroy_buffer(self.upload_buffer, None);
            self.device.raw.destroy_buffer(self.readback_buffer, None);
            self.device.raw.free_memory(self.upload_memory, None);
            self.device.raw.free_memory(self.readback_memory, None);
            self.device.raw.destroy_image_view(self.offscreen_view, None);
            self.device.raw.destroy_image(self.offscreen_image, None);
            self.device.raw.free_memory(self.offscreen_memory, None);
            self.device.raw.destroy_image_view(self.intermediate_view, None);
            self.device.raw.destroy_image(self.intermediate_image, None);
            self.device.raw.free_memory(self.intermediate_memory, None);
        }
    }
}

// ── Format decode ───────────────────────────────────────────────────────────

/// CPU-side decode of one scanout texel back to absolute cd/m². The
/// encode shader's `OutputTransfer*` fragment is the producer; this
/// is its inverse, so what comes back is in the same "commanded
/// nits" units as the LUT entries the verify path compares against.
///
/// Mode-aware scaling: HDR PQ formats decode straight to linear nits
/// (PQ EOTF is absolute by construction). SDR sRGB formats decode
/// to linear `[0, 1]`, which then needs `× sdr_white_nits` to land
/// in absolute cd/m². The caller passes `sdr_white_nits` from the
/// EncodePush they handed to `diagnose`; for HDR formats the value
/// is ignored.
///
/// Supported today:
///   - `R16G16B16A16_SFLOAT` — HDR PQ fp16 scanout (DP-4 HDR mode)
///   - `R8G8B8A8_UNORM` / `R8G8B8A8_SRGB`,
///     `B8G8R8A8_UNORM` / `B8G8R8A8_SRGB` — 8bpc SDR scanout
///   - `A2R10G10B10_UNORM_PACK32` — 10bpc SDR scanout (default for
///     `depth=Bpc10`, which is prism's default when no `--depth`
///     CLI flag is given AND the output isn't in HDR mode)
///
/// Add new arms as the renderer grows scanout formats; missing
/// support manifests as a `MissingFeature` error on the first
/// `EncodeDiagnose` IPC call for that output.
pub fn decode_scanout_texel(
    format: vk::Format,
    bytes: &[u8],
    sdr_white_nits: f64,
) -> Result<DiagnosedNits> {
    match format {
        vk::Format::R16G16B16A16_SFLOAT => {
            // PQ-encoded half-floats in [0, 1]. Inverse: f16 → f32 →
            // PQ EOTF → linear nits.
            if bytes.len() < 8 {
                return Err(RendererError::MissingFeature(
                    "decode_scanout_texel: R16G16B16A16 needs ≥8 bytes",
                ));
            }
            let r = half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32();
            let g = half::f16::from_le_bytes([bytes[2], bytes[3]]).to_f32();
            let b = half::f16::from_le_bytes([bytes[4], bytes[5]]).to_f32();
            Ok([pq_eotf(r) as f64, pq_eotf(g) as f64, pq_eotf(b) as f64])
        }
        vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => {
            if bytes.len() < 4 {
                return Err(RendererError::MissingFeature(
                    "decode_scanout_texel: R8G8B8A8 needs ≥4 bytes",
                ));
            }
            Ok([
                srgb_eotf(bytes[0] as f64 / 255.0) * sdr_white_nits,
                srgb_eotf(bytes[1] as f64 / 255.0) * sdr_white_nits,
                srgb_eotf(bytes[2] as f64 / 255.0) * sdr_white_nits,
            ])
        }
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => {
            if bytes.len() < 4 {
                return Err(RendererError::MissingFeature(
                    "decode_scanout_texel: B8G8R8A8 needs ≥4 bytes",
                ));
            }
            Ok([
                srgb_eotf(bytes[2] as f64 / 255.0) * sdr_white_nits,
                srgb_eotf(bytes[1] as f64 / 255.0) * sdr_white_nits,
                srgb_eotf(bytes[0] as f64 / 255.0) * sdr_white_nits,
            ])
        }
        vk::Format::A2R10G10B10_UNORM_PACK32 => {
            // Single 32-bit little-endian word. Channel layout
            // (MSB→LSB): A[31:30] R[29:20] G[19:10] B[9:0].
            // Vulkan/DRM packed-format convention. Normalize each
            // 10-bit channel to [0, 1] then sRGB-decode.
            if bytes.len() < 4 {
                return Err(RendererError::MissingFeature(
                    "decode_scanout_texel: A2R10G10B10 needs ≥4 bytes",
                ));
            }
            let pack = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let r10 = ((pack >> 20) & 0x3FF) as f64;
            let g10 = ((pack >> 10) & 0x3FF) as f64;
            let b10 = (pack & 0x3FF) as f64;
            Ok([
                srgb_eotf(r10 / 1023.0) * sdr_white_nits,
                srgb_eotf(g10 / 1023.0) * sdr_white_nits,
                srgb_eotf(b10 / 1023.0) * sdr_white_nits,
            ])
        }
        _ => Err(RendererError::MissingFeature(
            "decode_scanout_texel: unsupported scanout format",
        )),
    }
}

/// sRGB encoded `[0, 1]` → linear `[0, 1]`. Standard IEC 61966-2-1
/// piecewise inverse; caller multiplies by sdr_white_nits to get
/// absolute cd/m² if needed.
fn srgb_eotf(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

// ── Vulkan helpers ──────────────────────────────────────────────────────────

fn format_texel_bytes(format: vk::Format) -> Result<usize> {
    Ok(match format {
        vk::Format::R32G32B32A32_SFLOAT => 16,
        vk::Format::R16G16B16A16_SFLOAT => 8,
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::B8G8R8A8_SRGB
        | vk::Format::A2R10G10B10_UNORM_PACK32 => 4,
        _ => {
            return Err(RendererError::MissingFeature(
                "EncodeDiagnoseProbe: unsupported format (intermediate or scanout)",
            ));
        }
    })
}

fn create_image_1x1(
    device: &Device,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    mem_props: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width: 1, height: 1, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.raw.create_image(&info, None) }
        .vk_ctx("create_image (diagnose 1×1)")?;
    let req = unsafe { device.raw.get_image_memory_requirements(image) };
    let mem_type = pick_memory(device, req.memory_type_bits, mem_props)?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.raw.allocate_memory(&alloc, None) }
        .vk_ctx("allocate_memory (diagnose 1×1)")?;
    unsafe { device.raw.bind_image_memory(image, memory, 0) }
        .vk_ctx("bind_image_memory (diagnose 1×1)")?;
    Ok((image, memory))
}

fn create_view_1x1(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
) -> Result<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(color_subresource_range());
    unsafe { device.raw.create_image_view(&info, None) }
        .vk_ctx("create_image_view (diagnose 1×1)")
}

fn create_host_buffer(
    device: &Device,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8, vk::DeviceSize)> {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.raw.create_buffer(&info, None) }
        .vk_ctx("create_buffer (diagnose staging)")?;
    let req = unsafe { device.raw.get_buffer_memory_requirements(buffer) };
    let mem_type = pick_memory(
        device,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.raw.allocate_memory(&alloc, None) }
        .vk_ctx("allocate_memory (diagnose staging)")?;
    unsafe { device.raw.bind_buffer_memory(buffer, memory, 0) }
        .vk_ctx("bind_buffer_memory (diagnose staging)")?;
    let ptr = unsafe {
        device.raw.map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty())
    }
    .vk_ctx("map_memory (diagnose staging)")? as *mut u8;
    Ok((buffer, memory, ptr, req.size))
}

fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn pick_memory(
    device: &Device,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let props = unsafe {
        device
            .instance_raw()
            .get_physical_device_memory_properties(device.physical.raw)
    };
    for i in 0..props.memory_type_count {
        let mt = props.memory_types[i as usize];
        if (type_bits & (1 << i)) != 0 && mt.property_flags.contains(required) {
            return Ok(i);
        }
    }
    Err(RendererError::MissingFeature(
        "EncodeDiagnoseProbe: no memory type matches required flags",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PQ EOTF round-trip via decode_scanout_texel. Pack a known PQ-
    /// encoded f16 value (anchor: 0.5083 ≈ 100 nits) and verify we
    /// decode back to ~100. sdr_white_nits is unused for HDR formats.
    #[test]
    fn decode_pq_anchor() {
        let v = half::f16::from_f32(0.5083);
        let bytes_per = v.to_le_bytes();
        let mut buf = [0u8; 8];
        buf[0..2].copy_from_slice(&bytes_per);
        buf[2..4].copy_from_slice(&bytes_per);
        buf[4..6].copy_from_slice(&bytes_per);
        let decoded = decode_scanout_texel(vk::Format::R16G16B16A16_SFLOAT, &buf, 80.0).unwrap();
        for c in 0..3 {
            assert!(
                (decoded[c] - 100.0).abs() < 2.0,
                "ch {c} decoded {} vs ~100",
                decoded[c]
            );
        }
    }

    /// sRGB white (255, 255, 255) at sdr_white_nits=80 should decode
    /// to ~80 cd/m² per channel. Anchors the SDR nits-scaling path
    /// (which was missing before — used to return [0, 1] linear).
    #[test]
    fn decode_srgb_white_at_80_nits() {
        let buf = [255u8, 255, 255, 255];
        let decoded = decode_scanout_texel(vk::Format::R8G8B8A8_UNORM, &buf, 80.0).unwrap();
        for c in 0..3 {
            assert!(
                (decoded[c] - 80.0).abs() < 1e-6,
                "ch {c} decoded {} vs ~80",
                decoded[c]
            );
        }
    }

    /// A2R10G10B10 packed: full white (R=G=B=1023, A=3) at
    /// sdr_white_nits=80 decodes to ~80 cd/m² per channel. Exercises
    /// the packed-10-bit channel extraction that DP-8 SDR scanout
    /// hits (depth=Bpc10 default whenever no `--depth` CLI flag is
    /// given AND output isn't HDR).
    #[test]
    fn decode_a2r10g10b10_white_at_80_nits() {
        let pack: u32 = (3 << 30) | (1023 << 20) | (1023 << 10) | 1023;
        let buf = pack.to_le_bytes();
        let decoded =
            decode_scanout_texel(vk::Format::A2R10G10B10_UNORM_PACK32, &buf, 80.0).unwrap();
        for c in 0..3 {
            assert!(
                (decoded[c] - 80.0).abs() < 1e-6,
                "ch {c} decoded {} vs ~80",
                decoded[c]
            );
        }
    }

    /// A2R10G10B10 channel order: pack distinct values into R/G/B and
    /// confirm we don't transpose. R=512 > G=256 > B=128 must round-
    /// trip into monotonically-decreasing decoded values — catches an
    /// off-by-one or swapped-shift mistake in the packed unpacker.
    #[test]
    fn decode_a2r10g10b10_channel_order() {
        let pack: u32 = (512 << 20) | (256 << 10) | 128;
        let buf = pack.to_le_bytes();
        let decoded =
            decode_scanout_texel(vk::Format::A2R10G10B10_UNORM_PACK32, &buf, 100.0).unwrap();
        assert!(decoded[0] > decoded[1], "R={} should > G={}", decoded[0], decoded[1]);
        assert!(decoded[1] > decoded[2], "G={} should > B={}", decoded[1], decoded[2]);
    }
}
