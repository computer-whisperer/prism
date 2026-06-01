//! Screen capture: render the live intermediate through an sRGB *capture*
//! encode and read it back to host memory.
//!
//! This is the shared primitive behind every capture frontend (debug dump,
//! `wlr-screencopy`, `ext-image-copy-capture`, PipeWire screencast). It takes
//! the per-output fp32 BT.2020 absolute-nits intermediate — panel-independent
//! real light — and encodes a colorimetric sRGB rendition with **no** panel
//! correction, the opposite of the live per-output encode (which is panel
//! *realization*: 3D LUT + panel transfer). See `docs/screen-capture.md`.
//!
//! Capture chain: `[CalibrationMatrix(BT.2020 → BT.709), OutputTransferSrgb]`.
//! `sdr_white_nits` (the output's reference-white level) maps diffuse white to
//! `1.0`; the sRGB OETF encodes for an ordinary sRGB viewer. Highlights above
//! reference white and out-of-BT.709-gamut colors are hard-clipped by the
//! shader's `clamp` — a crude but viewer-safe first-cut tone/gamut map (the doc
//! tracks the roll-off follow-up).
//!
//! Phase 1 uses the immediate (out-of-band) path: a single one-shot submit that
//! samples the intermediate where it already sits (`SHADER_READ_ONLY_OPTIMAL`
//! after the last frame's encode), renders into an owned offscreen, and copies
//! that into a host-visible buffer. Must be called when no frame for this output
//! is mid-flight (the live render path's own submit must have drained); the
//! caller's `OneshotPool::record_and_submit` `queue_wait_idle` then orders this
//! capture before the next frame's decode rewrites the intermediate.

use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Arc;

use ash::khr::external_semaphore_fd;
use ash::vk;

use crate::device::Device;
use crate::encode_synth::{EncodeConfig, EncodeFragment, EncodePushSynth as EncodePush};
use crate::error::{RendererError, Result, VkResultExt};
use crate::intermediate::{create_view, pick_device_local_memory};
use crate::oneshot::OneshotPool;
use crate::pipeline::encode::EncodePipeline;

/// Whether `format` is a capture target this module can produce. Both are
/// 8-bit-per-channel UNORM (not `_SRGB`): the `OutputTransferSrgb` fragment
/// applies the sRGB OETF in-shader, so the stored bytes are the final sRGB code
/// values — a `_SRGB` view would double-encode. The two byte orders cover the
/// common Wayland buffer formats: `R8G8B8A8_UNORM` = memory `R,G,B,A` (wl_shm
/// `Abgr8888`), `B8G8R8A8_UNORM` = memory `B,G,R,A` (wl_shm `Xrgb8888`/
/// `Argb8888`, the universally-supported screencopy format).
fn is_supported_format(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM
    )
}

/// A captured frame: tightly packed, row-major, 4 bytes/pixel in `format`'s
/// channel order (sRGB-encoded), no row padding. `pixels.len() == width *
/// height * 4`.
pub struct CaptureImage {
    pub width: u32,
    pub height: u32,
    /// The pixel byte order — `R8G8B8A8_UNORM` or `B8G8R8A8_UNORM`.
    pub format: vk::Format,
    pub pixels: Vec<u8>,
}

impl CaptureImage {
    /// Serialize as a binary PPM (`P6`, 8-bit RGB), dropping alpha. A
    /// zero-dependency debug format for the phase-1 dump path; real frontends
    /// hand `pixels` to a client buffer instead. Swizzles per `format` so the
    /// PPM is always RGB regardless of capture byte order.
    pub fn to_ppm(&self) -> Vec<u8> {
        let bgr = self.format == vk::Format::B8G8R8A8_UNORM;
        let header = format!("P6\n{} {}\n255\n", self.width, self.height);
        let mut out = Vec::with_capacity(header.len() + (self.width * self.height * 3) as usize);
        out.extend_from_slice(header.as_bytes());
        for px in self.pixels.chunks_exact(4) {
            if bgr {
                out.extend_from_slice(&[px[2], px[1], px[0]]);
            } else {
                out.extend_from_slice(&px[..3]);
            }
        }
        out
    }
}

/// Owned offscreen target + host-visible readback buffer, sized to one output.
/// Reallocated when the output extent changes (mode change); owns the device
/// handle so its `Drop` frees its own resources (the realloc in
/// `ensure_target` relies on this).
struct Target {
    device: Arc<Device>,
    extent: vk::Extent2D,
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    buffer_ptr: *mut u8,
    buffer_size: vk::DeviceSize,
}

/// Renders the intermediate through the capture encode and reads it back.
/// Lazily allocated by the [`Renderer`](crate::Renderer) on first capture; the
/// offscreen/readback target is then sized on demand to the output extent.
pub struct CaptureEncoder {
    device: Arc<Device>,
    encode: EncodePipeline,
    oneshot: OneshotPool,
    target: Option<Target>,
    /// Async-submit resources for the dmabuf path (renders into a client
    /// dmabuf, returns a sync_fd). Lazily created on first dmabuf capture.
    async_slot: Option<AsyncSlot>,
    /// The target pixel format this encoder's pipeline was built for. The
    /// renderer rebuilds the encoder if a capture asks for a different one.
    format: vk::Format,
}

// SAFETY: `buffer_ptr` is a persistently-mapped HOST_COHERENT pointer, only
// touched under `&mut self` while no GPU access is outstanding (each capture
// waits on its one-shot submit before reading). Same rationale as
// `EncodeDiagnoseProbe`.
unsafe impl Send for CaptureEncoder {}
unsafe impl Sync for CaptureEncoder {}

impl CaptureEncoder {
    /// Build a capture encoder targeting `format` (`R8G8B8A8_UNORM` or
    /// `B8G8R8A8_UNORM`). The sRGB encode pipeline is format-specific, so a
    /// different target format needs a different encoder.
    pub fn new(device: Arc<Device>, format: vk::Format) -> Result<Self> {
        if !is_supported_format(format) {
            return Err(RendererError::MissingFeature(
                "capture: unsupported target format (want R8G8B8A8_UNORM or B8G8R8A8_UNORM)",
            ));
        }
        // Colorimetric sRGB capture chain — no Lut3d (panel correction must not
        // leak into a screenshot), so the pipeline declares no LUT binding.
        let config = EncodeConfig {
            fragments: vec![
                EncodeFragment::CalibrationMatrix,
                EncodeFragment::OutputTransferSrgb,
            ],
        };
        let encode = EncodePipeline::new(device.clone(), format, &config)?;
        let oneshot = OneshotPool::new(device.clone())?;
        Ok(Self {
            device,
            encode,
            oneshot,
            target: None,
            async_slot: None,
            format,
        })
    }

    /// The target pixel format this encoder produces.
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Capture `intermediate_view` (the live per-output intermediate, currently
    /// in `SHADER_READ_ONLY_OPTIMAL`) of size `extent` as an sRGB RGBA8 image.
    /// `sdr_white_nits` is the output's reference-white level (the value 1.0
    /// after normalization) — pass `effective_sdr_reference_nits()`.
    pub fn capture(
        &mut self,
        intermediate_view: vk::ImageView,
        extent: vk::Extent2D,
        sdr_white_nits: f32,
    ) -> Result<CaptureImage> {
        self.ensure_target(extent)?;
        let push = capture_push(sdr_white_nits);
        let target = self.target.as_ref().unwrap();

        let encode = &self.encode;
        let image = target.image;
        let view = target.view;
        let buffer = target.buffer;
        self.oneshot.record_and_submit(|raw, cb| unsafe {
            // Make the prior frame's writes to the intermediate visible to our
            // sample (cross-submit: the live render path's encode left it in
            // SHADER_READ_ONLY, but a fresh submit needs the visibility edge),
            // and bring the offscreen UNDEFINED → COLOR_ATTACHMENT.
            let to_attach = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range())];
            let intermediate_vis = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)];
            raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default()
                    .memory_barriers(&intermediate_vis)
                    .image_memory_barriers(&to_attach),
            );

            record_fullscreen_encode(raw, encode, cb, intermediate_view, view, extent, &push);

            // COLOR_ATTACHMENT → TRANSFER_SRC, then copy the whole image into
            // the tightly-packed host buffer (buffer_row_length 0 ⇒ packed).
            let to_src = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range())];
            raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&to_src),
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
                .image_extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })];
            raw.cmd_copy_image_to_buffer(
                cb,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer,
                &region,
            );
        })?;

        let target = self.target.as_ref().unwrap();
        // SAFETY: HOST_COHERENT mapping; the one-shot submit above has been
        // waited on, so the copy is complete and no GPU access is outstanding.
        let pixels = unsafe {
            std::slice::from_raw_parts(target.buffer_ptr, target.buffer_size as usize).to_vec()
        };
        Ok(CaptureImage {
            width: extent.width,
            height: extent.height,
            format: self.format,
            pixels,
        })
    }

    /// Capture `intermediate_view` (size `extent`) directly into a caller-
    /// provided dmabuf-backed image (`dst_image`/`dst_view`, same `extent`,
    /// imported with `COLOR_ATTACHMENT` usage and this encoder's `format`) — the
    /// zero-copy path for `wlr-screencopy` dmabuf clients and (later) PipeWire.
    ///
    /// Unlike [`Self::capture`], this does **not** block: it submits the encode
    /// and returns a Linux `SYNC_FD` `OwnedFd` that signals when the GPU is done.
    /// The caller gates the protocol's completion event on that fd and must keep
    /// the dmabuf import alive until it fires. The image is left in `GENERAL`
    /// layout for external (cross-API / KMS-style) consumption, matching the
    /// scanout handoff.
    ///
    /// Captures whole-output only (`dst` extent must equal the intermediate);
    /// region/scaled capture is the SHM path's job for now.
    ///
    /// **Ordering requirement.** This submits a *separate* command buffer that
    /// samples the persistent intermediate; the in-cb `MemoryBarrier2` below only
    /// orders work *within this submission* (it makes prior writes visible to our
    /// sample), not against the per-frame render submits that also touch the
    /// intermediate. Correct ordering therefore depends on **the caller invoking
    /// this right after the output's `present()`**, from the same (calloop)
    /// thread: the capture submit then lands between frame N and frame N+1 in
    /// submission order, and the single graphics queue's in-order execution (as
    /// the renderer already relies on for the intermediate across frames) means
    /// the capture samples the completed frame N and finishes before frame N+1's
    /// decode overwrites it. The screencopy path does this via
    /// `submit_pending_screencopy`. Called out of that sequence (e.g. mid-frame
    /// from an arbitrary event), the worst case is a torn captured frame — not a
    /// crash, since the intermediate is never freed.
    pub fn capture_into_dmabuf(
        &mut self,
        intermediate_view: vk::ImageView,
        dst_image: vk::Image,
        dst_view: vk::ImageView,
        extent: vk::Extent2D,
        sdr_white_nits: f32,
    ) -> Result<OwnedFd> {
        if self.async_slot.is_none() {
            self.async_slot = Some(AsyncSlot::new(&self.device)?);
        }
        let push = capture_push(sdr_white_nits);
        let encode = &self.encode;
        let slot = self.async_slot.as_ref().unwrap();
        let cb = slot.cmd_buffer;
        let raw = &self.device.raw;

        // Gate reuse of this slot's cb/semaphore against the previous dmabuf
        // capture still being in flight.
        unsafe { raw.wait_for_fences(&[slot.fence], true, u64::MAX) }
            .vk_ctx("wait_for_fences (capture async slot)")?;
        unsafe { raw.reset_fences(&[slot.fence]) }.vk_ctx("reset_fences (capture async slot)")?;
        unsafe { raw.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()) }
            .vk_ctx("reset_command_buffer (capture async slot)")?;

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { raw.begin_command_buffer(cb, &begin) }.vk_ctx("begin_command_buffer (capture)")?;

        unsafe {
            // Make writes to the intermediate visible to our sample, and bring
            // the dst dmabuf UNDEFINED → COLOR_ATTACHMENT. NB: this is a
            // *within-submission* visibility edge only — execution ordering vs.
            // the per-frame submits relies on same-queue ordering (see the
            // method doc's "Ordering caveat").
            let to_attach = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(dst_image)
                .subresource_range(color_range())];
            let intermediate_vis = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)];
            raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default()
                    .memory_barriers(&intermediate_vis)
                    .image_memory_barriers(&to_attach),
            );

            record_fullscreen_encode(raw, encode, cb, intermediate_view, dst_view, extent, &push);

            // COLOR_ATTACHMENT → GENERAL for external consumption (the client's
            // reader / KMS), matching the scanout's final handoff transition.
            let to_general = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                .dst_access_mask(vk::AccessFlags2::empty())
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(dst_image)
                .subresource_range(color_range())];
            raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&to_general),
            );

            raw.end_command_buffer(cb)
        }
        .vk_ctx("end_command_buffer (capture)")?;

        // Submit, signalling both the fence (slot-reuse gate) and the exportable
        // binary semaphore (we export it as the returned sync_fd).
        let cb_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(slot.semaphore)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submit = [vk::SubmitInfo2::default()
            .command_buffer_infos(&cb_infos)
            .signal_semaphore_infos(&signal)];
        unsafe { raw.queue_submit2(self.device.graphics_queue, &submit, slot.fence) }
            .vk_ctx("queue_submit2 (capture async)")?;

        // Export the just-signalled semaphore as a sync_file fd (transfers the
        // sync state to the fd and unsignals the semaphore for next reuse).
        let get_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(slot.semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let raw_fd = unsafe { slot.fd_loader.get_semaphore_fd(&get_info) }
            .vk_ctx("vkGetSemaphoreFdKHR (capture SYNC_FD)")?;
        if raw_fd < 0 {
            return Err(RendererError::MissingFeature(
                "capture: vkGetSemaphoreFdKHR returned a negative fd",
            ));
        }
        // SAFETY: a fresh, owned sync_file fd from a successful export.
        Ok(unsafe { OwnedFd::from_raw_fd(raw_fd) })
    }

    /// (Re)allocate the offscreen + readback target if it doesn't match `extent`.
    fn ensure_target(&mut self, extent: vk::Extent2D) -> Result<()> {
        if self
            .target
            .as_ref()
            .is_some_and(|t| t.extent.width == extent.width && t.extent.height == extent.height)
        {
            return Ok(());
        }
        // Drop the old target first so its allocations are freed before we make
        // new (potentially larger) ones.
        self.target = None;
        self.target = Some(Target::new(&self.device, extent, self.format)?);
        Ok(())
    }
}

impl Target {
    fn new(device: &Arc<Device>, extent: vk::Extent2D, format: vk::Format) -> Result<Self> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.raw.create_image(&image_info, None) }
            .vk_ctx("create_image (capture offscreen)")?;
        let req = unsafe { device.raw.get_image_memory_requirements(image) };
        let mem_type = pick_device_local_memory(device, req.memory_type_bits)?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        let memory = unsafe { device.raw.allocate_memory(&alloc, None) }
            .vk_ctx("allocate_memory (capture offscreen)")?;
        unsafe { device.raw.bind_image_memory(image, memory, 0) }
            .vk_ctx("bind_image_memory (capture offscreen)")?;
        let view = create_view(device, image, format)?;

        // Host-visible readback buffer, tightly packed RGBA8.
        let buffer_size = (extent.width as vk::DeviceSize) * (extent.height as vk::DeviceSize) * 4;
        let buf_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.raw.create_buffer(&buf_info, None) }
            .vk_ctx("create_buffer (capture readback)")?;
        let breq = unsafe { device.raw.get_buffer_memory_requirements(buffer) };
        let bmem_type = pick_host_visible_memory(device, breq.memory_type_bits)?;
        let balloc = vk::MemoryAllocateInfo::default()
            .allocation_size(breq.size)
            .memory_type_index(bmem_type);
        let buffer_memory = unsafe { device.raw.allocate_memory(&balloc, None) }
            .vk_ctx("allocate_memory (capture readback)")?;
        unsafe { device.raw.bind_buffer_memory(buffer, buffer_memory, 0) }
            .vk_ctx("bind_buffer_memory (capture readback)")?;
        let buffer_ptr = unsafe {
            device
                .raw
                .map_memory(buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
        }
        .vk_ctx("map_memory (capture readback)")? as *mut u8;

        Ok(Self {
            device: device.clone(),
            extent,
            image,
            view,
            memory,
            buffer,
            buffer_memory,
            buffer_ptr,
            buffer_size,
        })
    }
}

impl Drop for Target {
    fn drop(&mut self) {
        // No GPU work referencing this target is outstanding: each capture waits
        // on its one-shot submit, and a realloc/teardown only happens between
        // captures (and the encoder drains the device before dropping).
        unsafe {
            self.device.raw.unmap_memory(self.buffer_memory);
            self.device.raw.destroy_buffer(self.buffer, None);
            self.device.raw.free_memory(self.buffer_memory, None);
            self.device.raw.destroy_image_view(self.view, None);
            self.device.raw.destroy_image(self.image, None);
            self.device.raw.free_memory(self.memory, None);
        }
    }
}

impl Drop for CaptureEncoder {
    fn drop(&mut self) {
        // Drain before the fields (target's mapped memory, async slot's pool /
        // fence / semaphore) tear down in their own Drop impls.
        unsafe {
            let _ = self.device.raw.device_wait_idle();
        }
    }
}

/// Async-submit resources for the dmabuf capture path: a reusable command
/// buffer + a reuse-gate fence + an exportable binary semaphore we hand to the
/// caller as a sync_fd. One in flight at a time (the fence serializes reuse).
struct AsyncSlot {
    device: Arc<Device>,
    pool: vk::CommandPool,
    cmd_buffer: vk::CommandBuffer,
    /// Signalled by each submit; waited at the start of the next to gate reuse.
    /// Created signalled so the first wait is a no-op.
    fence: vk::Fence,
    /// Exportable (SYNC_FD) binary semaphore, signalled by each submit and
    /// exported as the returned sync_fd. The export unsignals it for reuse.
    semaphore: vk::Semaphore,
    fd_loader: external_semaphore_fd::Device,
}

impl AsyncSlot {
    fn new(device: &Arc<Device>) -> Result<Self> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.physical.graphics_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { device.raw.create_command_pool(&pool_info, None) }
            .vk_ctx("create_command_pool (capture async)")?;
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buffer = unsafe { device.raw.allocate_command_buffers(&alloc) }
            .vk_ctx("allocate_command_buffers (capture async)")?[0];
        let fence = unsafe {
            device.raw.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }
        .vk_ctx("create_fence (capture async)")?;
        let mut export = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut export);
        let semaphore = unsafe { device.raw.create_semaphore(&sem_info, None) }
            .vk_ctx("create_semaphore (capture async, exportable SYNC_FD)")?;
        let fd_loader = external_semaphore_fd::Device::new(device.instance_raw(), &device.raw);
        Ok(Self {
            device: device.clone(),
            pool,
            cmd_buffer,
            fence,
            semaphore,
            fd_loader,
        })
    }
}

impl Drop for AsyncSlot {
    fn drop(&mut self) {
        // CaptureEncoder::drop drained the device before we get here.
        unsafe {
            self.device.raw.destroy_semaphore(self.semaphore, None);
            self.device.raw.destroy_fence(self.fence, None);
            self.device.raw.destroy_command_pool(self.pool, None);
        }
    }
}

/// Build the capture encode push constants for a given reference-white level:
/// the BT.2020 → BT.709 primaries matrix plus `sdr_white_nits` for the sRGB
/// OETF's normalization. Shared by the SHM and dmabuf paths.
fn capture_push(sdr_white_nits: f32) -> EncodePush {
    let mut push = EncodePush::sdr_identity();
    push.sdr_white_nits = sdr_white_nits;
    push.target_peak_nits = sdr_white_nits;
    // CalibrationMatrix does `out = mat3(cal_matrix) * in`; set it to the
    // BT.2020 → BT.709 primaries conversion so the sRGB OETF that follows
    // receives sRGB-primary light.
    push.set_ctm(prism_frame::bt2020_to_srgb_matrix());
    push
}

/// Record the capture encode pass: a full-screen triangle sampling
/// `intermediate_view` (`SHADER_READ_ONLY_OPTIMAL`) and writing the sRGB capture
/// into `dst_view` (must already be `COLOR_ATTACHMENT_OPTIMAL`, `extent`-sized).
/// Emits no barriers — the caller wraps this with the layout transitions and
/// submit appropriate to its destination (owned offscreen vs. client dmabuf).
///
/// # Safety
/// `cb` must be in the recording state; `dst_view`/`intermediate_view` must be
/// live and in the layouts above for the duration of the submitted work.
unsafe fn record_fullscreen_encode(
    raw: &ash::Device,
    encode: &EncodePipeline,
    cb: vk::CommandBuffer,
    intermediate_view: vk::ImageView,
    dst_view: vk::ImageView,
    extent: vk::Extent2D,
    push: &EncodePush,
) {
    let color_attach = [vk::RenderingAttachmentInfo::default()
        .image_view(dst_view)
        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)];
    let rendering_info = vk::RenderingInfo::default()
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        })
        .layer_count(1)
        .color_attachments(&color_attach);
    raw.cmd_begin_rendering(cb, &rendering_info);

    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    raw.cmd_set_viewport(cb, 0, &[viewport]);
    raw.cmd_set_scissor(
        cb,
        0,
        &[vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        }],
    );
    raw.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, encode.pipeline);

    // Binding 0 = intermediate. No LUT binding (capture chain omits it).
    let intermediate_info = [vk::DescriptorImageInfo::default()
        .sampler(encode.sampler)
        .image_view(intermediate_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    let writes = [encode.write_intermediate_binding(&intermediate_info)];
    encode.push_loader.cmd_push_descriptor_set(
        cb,
        vk::PipelineBindPoint::GRAPHICS,
        encode.pipeline_layout,
        0,
        &writes,
    );
    raw.cmd_push_constants(
        cb,
        encode.pipeline_layout,
        vk::ShaderStageFlags::FRAGMENT,
        0,
        bytemuck::bytes_of(push),
    );
    raw.cmd_draw(cb, 3, 1, 0, 0);
    raw.cmd_end_rendering(cb);
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn pick_host_visible_memory(device: &Device, type_bits: u32) -> Result<u32> {
    let props = unsafe {
        device
            .instance_raw()
            .get_physical_device_memory_properties(device.physical.raw)
    };
    let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    for i in 0..props.memory_type_count {
        let mt = props.memory_types[i as usize];
        if (type_bits & (1 << i)) != 0 && mt.property_flags.contains(want) {
            return Ok(i);
        }
    }
    Err(RendererError::MissingFeature(
        "capture: no HOST_VISIBLE|HOST_COHERENT memory type for readback buffer",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PPM serialization: header + RGB triples, alpha dropped, packed.
    #[test]
    fn ppm_header_and_pixels() {
        let img = CaptureImage {
            width: 2,
            height: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            pixels: vec![10, 20, 30, 255, 40, 50, 60, 128],
        };
        let ppm = img.to_ppm();
        let header = b"P6\n2 1\n255\n";
        assert_eq!(&ppm[..header.len()], header);
        assert_eq!(&ppm[header.len()..], &[10, 20, 30, 40, 50, 60]);
    }

    /// BGRA capture swizzles to RGB in the PPM (the screencopy path captures
    /// `B8G8R8A8_UNORM` to fill `Xrgb8888` buffers).
    #[test]
    fn ppm_swizzles_bgra() {
        let img = CaptureImage {
            width: 1,
            height: 1,
            format: vk::Format::B8G8R8A8_UNORM,
            // memory B,G,R,A = 30,20,10 → RGB 10,20,30
            pixels: vec![30, 20, 10, 255],
        };
        let ppm = img.to_ppm();
        let header = b"P6\n1 1\n255\n";
        assert_eq!(&ppm[header.len()..], &[10, 20, 30]);
    }

    /// The BT.2020 → sRGB capture matrix maps neutral grey to neutral grey
    /// (white point preserved): equal-energy BT.2020 RGB stays equal-energy.
    /// Catches a transposed/mis-scaled matrix that would tint captures.
    #[test]
    fn capture_matrix_preserves_neutral() {
        let m = prism_frame::bt2020_to_srgb_matrix();
        let row_sum = |r: usize| m[r][0] + m[r][1] + m[r][2];
        for r in 0..3 {
            assert!(
                (row_sum(r) - 1.0).abs() < 1e-4,
                "row {r} sums to {} (neutral not preserved)",
                row_sum(r)
            );
        }
    }
}
