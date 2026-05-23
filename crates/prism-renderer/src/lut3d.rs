//! Per-output 3D color LUT texture.
//!
//! The shape that flows through the encode pipeline: a single trilinear
//! sample replaces the matrix + per-channel-curve pair, capturing both
//! gamut correction AND per-channel response in one table. The encode
//! shader's [`Lut3d`](crate::encode_synth::EncodeFragment::Lut3d) fragment
//! is the consumer; this module is the producer side.
//!
//! ## Memory layout
//!
//! `R16G16B16A16_SFLOAT` 3D image, cube edge configurable (typical 17 or
//! 33). Alpha is unused content-wise but uploaded as 1.0 so a stray
//! sampler doesn't pick up garbage. f16 precision is enough — the LUT's
//! own grid quantization dominates round-trip error long before f16
//! mantissa loss does.
//!
//! ## Coordinate space
//!
//! The shader hands the sampler PQ-encoded coords in `[0, 1]` (the
//! "shaper" stage). Each LUT entry `(i, j, k)` therefore stores the
//! commanded nits for input nits `pq_eotf((i, j, k) / (N-1))`. The
//! identity LUT — what every output starts with before any calibration —
//! just stores `pq_eotf(coord)` so the round trip `pq_oetf → sample →
//! identity_lut` returns the original linear nits (modulo trilinear
//! approximation error).
//!
//! ## Upload model
//!
//! One-shot synchronous: record a transition + copy + transition, submit,
//! wait for idle. Upload is rare — once at Renderer construction (identity
//! LUT) plus whenever a new calibration is pushed via IPC. We don't need
//! a persistent staging mapping with batched updates.

use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::error::{RendererError, Result, VkResultExt};
use crate::oneshot::OneshotPool;

/// Format of the 3D LUT texture. Half-float RGBA — the .a channel is
/// uploaded as 1.0 and unread by the encode fragment.
pub const LUT_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Bytes per LUT texel (half-float RGBA).
const TEXEL_BYTES: usize = 8;

/// Per-output 3D color LUT. Owns its image, view, memory, and the
/// staging buffer used for uploads.
pub struct Lut3dTexture {
    device: Arc<Device>,
    oneshot: OneshotPool,

    image: vk::Image,
    view: vk::ImageView,
    image_memory: vk::DeviceMemory,

    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    staging_ptr: *mut u8,
    staging_size: vk::DeviceSize,

    cube_edge: u32,
}

// Same justification as ShmTexture: the persistently-mapped staging pointer
// is per-instance and only touched inside `&mut self` methods.
unsafe impl Send for Lut3dTexture {}
unsafe impl Sync for Lut3dTexture {}

impl Lut3dTexture {
    /// Allocate the 3D image + staging buffer. `cube_edge` is the number
    /// of grid points per axis (typical 17 or 33). Total texel count is
    /// `cube_edge³`.
    pub fn new(device: Arc<Device>, cube_edge: u32) -> Result<Self> {
        if cube_edge < 2 {
            return Err(RendererError::MissingFeature(
                "Lut3dTexture: cube_edge must be >= 2 (1D-degenerate LUT not supported)",
            ));
        }
        let staging_size =
            (cube_edge as vk::DeviceSize).pow(3) * (TEXEL_BYTES as vk::DeviceSize);

        // ── 3D image ──────────────────────────────────────────────────────
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_3D)
            .format(LUT_FORMAT)
            .extent(vk::Extent3D {
                width: cube_edge,
                height: cube_edge,
                depth: cube_edge,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.raw.create_image(&image_info, None) }
            .vk_ctx("create_image (lut3d)")?;

        let img_req = unsafe { device.raw.get_image_memory_requirements(image) };
        let img_mem_type = pick_memory(
            &device,
            img_req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let img_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(img_req.size)
            .memory_type_index(img_mem_type);
        let image_memory = unsafe { device.raw.allocate_memory(&img_alloc, None) }
            .vk_ctx("allocate_memory (lut3d image)")?;
        unsafe { device.raw.bind_image_memory(image, image_memory, 0) }
            .vk_ctx("bind_image_memory (lut3d)")?;

        // 3D image view. View type must match image type for sampling.
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_3D)
            .format(LUT_FORMAT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let view = unsafe { device.raw.create_image_view(&view_info, None) }
            .vk_ctx("create_image_view (lut3d)")?;

        // ── Staging buffer ────────────────────────────────────────────────
        let buf_info = vk::BufferCreateInfo::default()
            .size(staging_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buffer = unsafe { device.raw.create_buffer(&buf_info, None) }
            .vk_ctx("create_buffer (lut3d staging)")?;
        let buf_req = unsafe { device.raw.get_buffer_memory_requirements(staging_buffer) };
        let buf_mem_type = pick_memory(
            &device,
            buf_req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let buf_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(buf_req.size)
            .memory_type_index(buf_mem_type);
        let staging_memory = unsafe { device.raw.allocate_memory(&buf_alloc, None) }
            .vk_ctx("allocate_memory (lut3d staging)")?;
        unsafe { device.raw.bind_buffer_memory(staging_buffer, staging_memory, 0) }
            .vk_ctx("bind_buffer_memory (lut3d staging)")?;
        let staging_ptr = unsafe {
            device.raw.map_memory(
                staging_memory,
                0,
                buf_req.size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .vk_ctx("map_memory (lut3d staging)")? as *mut u8;

        let oneshot = OneshotPool::new(device.clone())?;

        Ok(Self {
            device,
            oneshot,
            image,
            view,
            image_memory,
            staging_buffer,
            staging_memory,
            staging_ptr,
            staging_size,
            cube_edge,
        })
    }

    /// View bound at descriptor set 0, binding 1 of the encode pipeline.
    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    /// Cube edge length (grid points per axis).
    pub fn cube_edge(&self) -> u32 {
        self.cube_edge
    }

    /// Upload `entries` (length `cube_edge³`, RGB triples in linear nits)
    /// to the image. Synchronous: waits for the GPU to idle before
    /// returning, so the image is in `SHADER_READ_ONLY_OPTIMAL` and safe
    /// to sample by the time this returns. Each upload's old content is
    /// discarded (UNDEFINED → TRANSFER_DST → SHADER_READ_ONLY).
    ///
    /// Index order: `idx = (z * N + y) * N + x` for grid point `(x, y, z)`.
    /// X-fastest matches how Vulkan walks 3D image memory.
    pub fn upload(&mut self, entries: &[[f32; 3]]) -> Result<()> {
        let expected = (self.cube_edge as usize).pow(3);
        if entries.len() != expected {
            return Err(RendererError::MissingFeature(
                "Lut3dTexture::upload: entries length mismatches cube_edge³",
            ));
        }

        // Convert f32 RGB to f16 RGBA (alpha = 1.0) in-place into the
        // staging buffer. SAFETY: staging is HOST_COHERENT, sized
        // staging_size, persistently mapped; we touch it under &mut self
        // and the GPU side of any prior upload was waited on by the
        // previous oneshot submit.
        unsafe {
            let dst = self.staging_ptr as *mut half::f16;
            let one = half::f16::from_f32(1.0);
            for (i, rgb) in entries.iter().enumerate() {
                let off = i * 4;
                *dst.add(off) = half::f16::from_f32(rgb[0]);
                *dst.add(off + 1) = half::f16::from_f32(rgb[1]);
                *dst.add(off + 2) = half::f16::from_f32(rgb[2]);
                *dst.add(off + 3) = one;
            }
        }

        let image = self.image;
        let staging_buffer = self.staging_buffer;
        let cube_edge = self.cube_edge;
        self.oneshot.record_and_submit(|raw, cb| unsafe {
            let to_xfer = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_subresource_range())];
            raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&to_xfer),
            );

            let region = [vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: cube_edge,
                    height: cube_edge,
                    depth: cube_edge,
                })];
            raw.cmd_copy_buffer_to_image(
                cb,
                staging_buffer,
                image,
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
                .image(image)
                .subresource_range(color_subresource_range())];
            raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&to_sampled),
            );
        })?;

        let _ = self.staging_size;
        Ok(())
    }
}

impl Drop for Lut3dTexture {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.raw.device_wait_idle();
            self.device.raw.unmap_memory(self.staging_memory);
            self.device.raw.destroy_buffer(self.staging_buffer, None);
            self.device.raw.free_memory(self.staging_memory, None);
            self.device.raw.destroy_image_view(self.view, None);
            self.device.raw.destroy_image(self.image, None);
            self.device.raw.free_memory(self.image_memory, None);
        }
    }
}

/// SMPTE ST 2084 (PQ) EOTF: encoded `[0, 1]` → linear nits `[0, 10000]`.
/// CPU-side mirror of the shader's `pq_oetf` inverse — used for LUT
/// generation (identity LUT computation, and future synthesis paths
/// that need to know what nits a given LUT coord represents).
pub fn pq_eotf(v: f32) -> f32 {
    const M1: f32 = 0.1593017578125;
    const M2: f32 = 78.84375;
    const C1: f32 = 0.8359375;
    const C2: f32 = 18.8515625;
    const C3: f32 = 18.6875;
    let v = v.max(0.0);
    let vm = v.powf(1.0 / M2);
    let num = (vm - C1).max(0.0);
    let den = C2 - C3 * vm;
    let y = (num / den).powf(1.0 / M1);
    y * 10000.0
}

/// Generate an identity LUT (no calibration). Each grid point `(i, j, k)`
/// gets the linear-nits value the shader's PQ shaper would decode the
/// coord `(i, j, k) / (N-1)` to — so the round trip `pq_oetf(input) →
/// trilinear_sample → identity_lut` returns approximately `input`.
///
/// Trilinear interpolation between adjacent grid points approximates
/// `pq_eotf` piecewise-linearly; precision is best with finer LUTs. At
/// `cube_edge = 17` round-trip error peaks near a few percent at very
/// low luminance; `cube_edge = 33` brings it down to sub-percent.
pub fn identity_lut(cube_edge: u32) -> Vec<[f32; 3]> {
    let n = cube_edge as usize;
    let denom = (cube_edge - 1) as f32;
    let mut out = Vec::with_capacity(n * n * n);
    for k in 0..n {
        let bz = pq_eotf(k as f32 / denom);
        for j in 0..n {
            let g = pq_eotf(j as f32 / denom);
            for i in 0..n {
                let r = pq_eotf(i as f32 / denom);
                out.push([r, g, bz]);
            }
        }
    }
    out
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
        "Lut3dTexture: no memory type matches required flags",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PQ EOTF anchor points from the SMPTE ST 2084 spec.
    #[test]
    fn pq_eotf_anchors() {
        // V = 0 → 0 nits exactly.
        assert!(pq_eotf(0.0).abs() < 1e-3);
        // V = 1 → 10000 nits exactly.
        assert!((pq_eotf(1.0) - 10000.0).abs() < 0.1);
        // V = 0.5083 ≈ 100 nits (the SDR reference white anchor).
        let y = pq_eotf(0.5083);
        assert!(
            (y - 100.0).abs() < 2.0,
            "PQ(0.5083) = {y}, expected ~100"
        );
    }

    /// Identity LUT vertex count matches `cube_edge³`, vertices in
    /// X-fastest order, and corner values are 0 and 10000 nits.
    #[test]
    fn identity_lut_shape() {
        let lut = identity_lut(17);
        assert_eq!(lut.len(), 17 * 17 * 17);
        // (0, 0, 0) = black.
        assert!(lut[0].iter().all(|&v| v.abs() < 1e-3));
        // Last entry = (10000, 10000, 10000).
        let last = lut[17 * 17 * 17 - 1];
        for c in 0..3 {
            assert!(
                (last[c] - 10000.0).abs() < 0.1,
                "last[{c}] = {}, expected ~10000",
                last[c]
            );
        }
    }

    /// Identity LUT respects the shaper: at any axis-aligned grid point
    /// the entry equals `pq_eotf(coord)` per channel, so the shader-side
    /// round trip is identity to within trilinear-interpolation error.
    #[test]
    fn identity_lut_axis_grid_points() {
        let n = 17u32;
        let lut = identity_lut(n);
        let denom = (n - 1) as f32;
        // X-axis: walk i from 0 to n-1 at (j=0, k=0).
        for i in 0..n {
            let idx = i as usize;
            let expected = pq_eotf(i as f32 / denom);
            assert!(
                (lut[idx][0] - expected).abs() < 1e-3,
                "x-axis i={i}: lut[{idx}].r = {}, expected {expected}",
                lut[idx][0]
            );
            assert!(lut[idx][1].abs() < 1e-3, "y at (i, 0, 0) should be 0");
            assert!(lut[idx][2].abs() < 1e-3, "z at (i, 0, 0) should be 0");
        }
    }
}
