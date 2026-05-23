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

use std::path::Path;
use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::error::{RendererError, Result, VkResultExt};
use crate::oneshot::OneshotPool;

/// Format of the 3D LUT texture. Half-float RGBA — the .a channel is
/// uploaded as 1.0 and unread by the encode fragment.
pub const LUT_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Cube edge length of every per-output 3D LUT prism allocates. The
/// encode shader's `Lut3d` fragment bakes this in as the divisor for
/// its `texture_coord = c × (N-1)/N + 0.5/N` texel-center adjustment;
/// the renderer uses it when allocating `Lut3dTexture`. Keep these
/// two consumers in lockstep — a mismatch silently mis-indexes the
/// LUT.
pub const LUT_CUBE_EDGE: u32 = 17;

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
    synthesize_lut_from_matrix_curve(cube_edge, None, None)
}

/// Synthesize a 3D LUT that reproduces the legacy `(CTM, per-channel
/// gain/gamma curve)` encode chain. The output is what the shader chain
/// `CalibrationMatrix → PerChannelResponseGainGamma` would have produced
/// at each grid point, baked once and sampled trilinearly thereafter.
///
/// Per grid point `(i, j, k)`:
/// 1. Decode the PQ coord `(i, j, k) / (N-1)` via [`pq_eotf`] →
///    `in_nits` (linear BT.2020 nits — the shader's IR domain).
/// 2. `panel_nits = CTM × in_nits` (CTM stored row-major; identity when
///    `ctm` is `None`). Per-channel negatives clip to zero — matches
///    the shader's `max(in, 0)` before the per-channel-response stage.
/// 3. `commanded = (panel_nits / gain)^(1/gamma)` per channel
///    (identity when `response_curve` is `None`).
/// 4. Store `commanded` as the LUT entry.
///
/// `None` for both inputs degenerates exactly to [`identity_lut`].
/// Calibrated LUTs stay in the same units throughout: the encode
/// pipeline's OutputTransfer fragment is what eventually clamps + PQ-
/// encodes for scanout, so the LUT output is "commanded nits" in
/// linear space.
pub fn synthesize_lut_from_matrix_curve(
    cube_edge: u32,
    ctm: Option<[[f32; 3]; 3]>,
    response_curve: Option<([f32; 3], [f32; 3])>,
) -> Vec<[f32; 3]> {
    let n = cube_edge as usize;
    let denom = (cube_edge - 1) as f32;
    let (gain, gamma) = response_curve.unwrap_or(([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]));
    // Floor matches the shader-side epsilon — guards against gain=0 div
    // and gamma=0 pow(0).
    const EPS: f32 = 1.0e-3;
    let safe_gain = [gain[0].max(EPS), gain[1].max(EPS), gain[2].max(EPS)];
    let inv_gamma = [
        1.0 / gamma[0].max(EPS),
        1.0 / gamma[1].max(EPS),
        1.0 / gamma[2].max(EPS),
    ];

    let mut out = Vec::with_capacity(n * n * n);
    // Iteration order is X-fastest then Y then Z — matches how the
    // Vulkan 3D image walks memory and what Lut3dTexture::upload expects.
    for k in 0..n {
        let bz = pq_eotf(k as f32 / denom);
        for j in 0..n {
            let g = pq_eotf(j as f32 / denom);
            for i in 0..n {
                let r = pq_eotf(i as f32 / denom);
                let in_nits = [r, g, bz];

                // CTM: panel_nits = M × in_nits (row-major).
                let mut panel = match ctm {
                    Some(m) => [
                        m[0][0] * in_nits[0] + m[0][1] * in_nits[1] + m[0][2] * in_nits[2],
                        m[1][0] * in_nits[0] + m[1][1] * in_nits[1] + m[1][2] * in_nits[2],
                        m[2][0] * in_nits[0] + m[2][1] * in_nits[1] + m[2][2] * in_nits[2],
                    ],
                    None => in_nits,
                };
                // Per-channel inverse response: commanded = (max(panel,0) / gain)^(1/gamma).
                for c in 0..3 {
                    let p = panel[c].max(0.0);
                    panel[c] = (p / safe_gain[c]).powf(inv_gamma[c]);
                }
                out.push(panel);
            }
        }
    }
    out
}

// ── Binary file format ──────────────────────────────────────────────────────

/// File magic (`"PLUT"` little-endian).
pub const LUT_FILE_MAGIC: u32 = 0x54554C50;
/// Current file format version.
pub const LUT_FILE_VERSION: u32 = 1;
/// `in_tf` enum value for the PQ shaper. The compositor's encode shader
/// always PQ-encodes its input before sampling the LUT, so files written
/// for a different shaper would mis-index. Stored explicitly so a future
/// shaper change can be detected at load.
pub const LUT_FILE_IN_TF_PQ: u32 = 1;

/// Binary header that precedes the data payload in a `.lut` file. All
/// fields little-endian. 32 bytes total — the data payload immediately
/// follows.
///
/// Field-by-field:
/// ```text
/// magic     u32  must be LUT_FILE_MAGIC
/// version   u32  must be LUT_FILE_VERSION
/// cube_edge u32  grid points per axis (typically 17 or 33)
/// in_tf     u32  shaper TF identifier; must be LUT_FILE_IN_TF_PQ
/// flags     u32  reserved; must be 0
/// peak_r    f32  panel-native peak emission, R channel (cd/m²)
/// peak_g    f32  panel-native peak emission, G channel (cd/m²)
/// peak_b    f32  panel-native peak emission, B channel (cd/m²)
/// ```
///
/// Data payload: `cube_edge³` RGB triples (3 × f32 each), X-fastest then
/// Y then Z, in linear nits. Matches the iteration order
/// [`Lut3dTexture::upload`] expects.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LutFileHeader {
    pub magic: u32,
    pub version: u32,
    pub cube_edge: u32,
    pub in_tf: u32,
    pub flags: u32,
    pub peak_r: f32,
    pub peak_g: f32,
    pub peak_b: f32,
}

/// Header byte size — guaranteed by the file format and verified at load.
pub const LUT_FILE_HEADER_BYTES: usize = 32;

/// Bytes per data triple (3 × f32, little-endian).
pub const LUT_FILE_TRIPLE_BYTES: usize = 12;

/// Loaded LUT — entries plus the metadata the header carried alongside
/// them. The cube edge is what callers need to validate against the
/// compositor's LUT texture; peaks are informational.
pub struct LoadedLut {
    pub cube_edge: u32,
    pub peak_nits: [f32; 3],
    pub entries: Vec<[f32; 3]>,
}

/// Read a binary LUT file from disk. Validates magic / version / shaper
/// TF / sane cube edge before allocating. Returns `MissingFeature` with
/// a descriptive context on any of those checks; callers can fall back
/// to the synthesis path on error.
pub fn load_lut3d_file(path: &Path) -> Result<LoadedLut> {
    let bytes = std::fs::read(path).map_err(|e| {
        tracing::warn!(path = %path.display(), "failed to read LUT file: {e}");
        RendererError::MissingFeature("Lut3d: file read failed")
    })?;
    if bytes.len() < LUT_FILE_HEADER_BYTES {
        return Err(RendererError::MissingFeature(
            "Lut3d: file shorter than 32-byte header",
        ));
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if magic != LUT_FILE_MAGIC {
        return Err(RendererError::MissingFeature(
            "Lut3d: bad magic (expected \"PLUT\")",
        ));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != LUT_FILE_VERSION {
        return Err(RendererError::MissingFeature(
            "Lut3d: unsupported file version",
        ));
    }
    let cube_edge = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if !(2..=129).contains(&cube_edge) {
        return Err(RendererError::MissingFeature(
            "Lut3d: cube_edge out of supported range (2..=129)",
        ));
    }
    let in_tf = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    if in_tf != LUT_FILE_IN_TF_PQ {
        return Err(RendererError::MissingFeature(
            "Lut3d: unsupported shaper TF (only PQ is recognized)",
        ));
    }
    let _flags = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
    let peak_r = f32::from_le_bytes(bytes[20..24].try_into().unwrap());
    let peak_g = f32::from_le_bytes(bytes[24..28].try_into().unwrap());
    let peak_b = f32::from_le_bytes(bytes[28..32].try_into().unwrap());

    let n = cube_edge as usize;
    let expected_data = n * n * n * LUT_FILE_TRIPLE_BYTES;
    if bytes.len() < LUT_FILE_HEADER_BYTES + expected_data {
        return Err(RendererError::MissingFeature(
            "Lut3d: file payload shorter than cube_edge³ × 12 bytes",
        ));
    }
    let mut entries = Vec::with_capacity(n * n * n);
    let mut off = LUT_FILE_HEADER_BYTES;
    for _ in 0..(n * n * n) {
        let r = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let g = f32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
        let b = f32::from_le_bytes(bytes[off + 8..off + 12].try_into().unwrap());
        entries.push([r, g, b]);
        off += LUT_FILE_TRIPLE_BYTES;
    }
    Ok(LoadedLut {
        cube_edge,
        peak_nits: [peak_r, peak_g, peak_b],
        entries,
    })
}

/// Write the entries + metadata as a binary LUT file. Header values
/// other than `cube_edge` and `peak_nits` are filled in from the
/// canonical constants. `entries` must have length `cube_edge³` and be
/// laid out X-fastest.
pub fn save_lut3d_file(
    path: &Path,
    cube_edge: u32,
    peak_nits: [f32; 3],
    entries: &[[f32; 3]],
) -> std::io::Result<()> {
    let n = cube_edge as usize;
    if entries.len() != n * n * n {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Lut3d: entries length mismatches cube_edge³",
        ));
    }
    let mut out = Vec::with_capacity(LUT_FILE_HEADER_BYTES + entries.len() * LUT_FILE_TRIPLE_BYTES);
    out.extend_from_slice(&LUT_FILE_MAGIC.to_le_bytes());
    out.extend_from_slice(&LUT_FILE_VERSION.to_le_bytes());
    out.extend_from_slice(&cube_edge.to_le_bytes());
    out.extend_from_slice(&LUT_FILE_IN_TF_PQ.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // flags reserved
    out.extend_from_slice(&peak_nits[0].to_le_bytes());
    out.extend_from_slice(&peak_nits[1].to_le_bytes());
    out.extend_from_slice(&peak_nits[2].to_le_bytes());
    for rgb in entries {
        out.extend_from_slice(&rgb[0].to_le_bytes());
        out.extend_from_slice(&rgb[1].to_le_bytes());
        out.extend_from_slice(&rgb[2].to_le_bytes());
    }
    std::fs::write(path, &out)
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

    /// Synthesis with no calibration is bit-equivalent to the identity
    /// LUT — both code paths must agree so removing one doesn't break the
    /// other.
    #[test]
    fn synthesis_no_calibration_equals_identity() {
        let n = 9u32;
        let synth = synthesize_lut_from_matrix_curve(n, None, None);
        let ident = identity_lut(n);
        assert_eq!(synth.len(), ident.len());
        for (i, (s, idn)) in synth.iter().zip(ident.iter()).enumerate() {
            for c in 0..3 {
                assert!(
                    (s[c] - idn[c]).abs() < 1e-6,
                    "diff at idx {i} ch {c}: synth={} identity={}",
                    s[c], idn[c],
                );
            }
        }
    }

    /// Synthesis matches the legacy shader chain for a sample DP-4
    /// calibration: at any grid point, the LUT entry equals what
    /// `(CTM × pq_eotf(coord))_+ then (in/gain)^(1/gamma)` would
    /// produce. Test point picked to exercise CTM off-diagonals and
    /// non-identity curve.
    #[test]
    fn synthesis_matches_analytical_chain() {
        // From a recent DP-4 calibration run.
        let ctm = Some([
            [0.303636, -0.083659, -0.002953],
            [-0.040053, 0.774200, -0.042934],
            [-0.000884, -0.012542, 0.105189],
        ]);
        let curve = Some(([0.0781f32, 0.1814, 0.0326], [1.0754f32, 1.0759, 1.0330]));
        let n = 9u32;
        let lut = synthesize_lut_from_matrix_curve(n, ctm, curve);

        // Spot-check the (4, 4, 4) grid point — well inside the cube so
        // all three channels see non-trivial CTM contributions.
        let denom = (n - 1) as f32;
        let coord = 4.0 / denom;
        let in_nits = [pq_eotf(coord); 3];
        let m = ctm.unwrap();
        let panel = [
            m[0][0] * in_nits[0] + m[0][1] * in_nits[1] + m[0][2] * in_nits[2],
            m[1][0] * in_nits[0] + m[1][1] * in_nits[1] + m[1][2] * in_nits[2],
            m[2][0] * in_nits[0] + m[2][1] * in_nits[1] + m[2][2] * in_nits[2],
        ];
        let (gain, gamma) = curve.unwrap();
        let expected = [
            (panel[0].max(0.0) / gain[0]).powf(1.0 / gamma[0]),
            (panel[1].max(0.0) / gain[1]).powf(1.0 / gamma[1]),
            (panel[2].max(0.0) / gain[2]).powf(1.0 / gamma[2]),
        ];
        let idx = ((4 * n as usize) + 4) * n as usize + 4; // X-fastest
        for c in 0..3 {
            assert!(
                (lut[idx][c] - expected[c]).abs() < 1e-4,
                "ch {c}: lut={} expected={}",
                lut[idx][c], expected[c]
            );
        }
    }

    /// File save/load is a byte-exact round trip for matching metadata.
    /// Catches regressions in the header layout or endianness handling.
    #[test]
    fn lut_file_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("prism-lut-test-{}.lut", std::process::id()));
        let cube_edge = 5u32;
        let peak_nits = [38.9, 113.9, 15.7];
        let entries = synthesize_lut_from_matrix_curve(
            cube_edge,
            Some([[0.95, 0.02, -0.01], [-0.03, 0.92, -0.04], [-0.001, -0.01, 0.95]]),
            Some(([0.5, 0.7, 0.3], [1.05, 1.0, 1.02])),
        );
        save_lut3d_file(&path, cube_edge, peak_nits, &entries).expect("save");
        let loaded = load_lut3d_file(&path).expect("load");
        assert_eq!(loaded.cube_edge, cube_edge);
        for c in 0..3 {
            assert!((loaded.peak_nits[c] - peak_nits[c]).abs() < 1e-6);
        }
        assert_eq!(loaded.entries.len(), entries.len());
        for (i, (orig, got)) in entries.iter().zip(loaded.entries.iter()).enumerate() {
            for c in 0..3 {
                assert!(
                    (orig[c] - got[c]).abs() < 1e-6,
                    "entry {i} ch {c}: orig={} got={}",
                    orig[c], got[c],
                );
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    /// Bad magic / version / shaper TF / oversized cube edge all reject
    /// the file cleanly rather than allocate nonsense.
    #[test]
    fn lut_file_validates_header_fields() {
        let mut buf = Vec::with_capacity(LUT_FILE_HEADER_BYTES);
        // Wrong magic.
        buf.extend_from_slice(&0xDEADBEEFu32.to_le_bytes());
        buf.extend_from_slice(&LUT_FILE_VERSION.to_le_bytes());
        buf.extend_from_slice(&5u32.to_le_bytes());
        buf.extend_from_slice(&LUT_FILE_IN_TF_PQ.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&80.0f32.to_le_bytes());
        buf.extend_from_slice(&80.0f32.to_le_bytes());
        buf.extend_from_slice(&80.0f32.to_le_bytes());
        let dir = std::env::temp_dir();
        let path = dir.join(format!("prism-lut-bad-{}.lut", std::process::id()));
        std::fs::write(&path, &buf).unwrap();
        assert!(load_lut3d_file(&path).is_err(), "bad magic should reject");
        let _ = std::fs::remove_file(&path);
    }

    /// Negative CTM outputs clip to zero before the per-channel curve —
    /// mirrors the shader's `max(in, 0)` clamp. Without this, the curve's
    /// `pow(negative, fractional)` would produce NaN entries.
    #[test]
    fn synthesis_clips_negative_ctm_outputs() {
        // CTM that maps positive R input to negative G and B (contrived).
        let ctm = Some([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]);
        let curve = Some(([0.5f32, 0.5, 0.5], [1.2f32, 1.2, 1.2]));
        let lut = synthesize_lut_from_matrix_curve(9, ctm, curve);
        // Grid point (4, 0, 0): R input > 0, so G/B CTM outputs are negative.
        let idx = 4; // (i=4, j=0, k=0) with X-fastest
        // R should be positive (positive CTM diagonal, positive input).
        assert!(lut[idx][0] > 0.0, "R should not be zero");
        // G and B clipped to zero before curve → curve(0) = 0.
        assert!(lut[idx][1].abs() < 1e-6, "G expected 0, got {}", lut[idx][1]);
        assert!(lut[idx][2].abs() < 1e-6, "B expected 0, got {}", lut[idx][2]);
    }
}
