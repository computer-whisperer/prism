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

use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Arc;

use ash::khr::external_semaphore_fd;
use ash::vk;

use crate::device::Device;
use crate::diagnose::{DiagnosedNits, EncodeDiagnoseProbe};
use crate::dmabuf::ImportedImage;
use crate::encode_synth::EncodeConfig;
use crate::error::{RendererError, Result, VkResultExt};
use crate::intermediate::Intermediate;
use crate::lut3d::{identity_lut, Lut3dTexture, LUT_CUBE_EDGE};
use crate::pipeline::decode::{DecodePipeline, DecodePush};
use crate::pipeline::encode::{EncodePipeline, EncodePush};
use crate::snapshot::SnapshotTexture;
use crate::upload::ShmTexture;
use prism_frame::{Physical, Rectangle};

/// Default 3D LUT cube edge. See `lut3d::LUT_CUBE_EDGE` for the
/// renderer-wide constant; this is the per-output binding that mirrors
/// it. Must stay in sync with the value the encode shader bakes into
/// its texel-center adjustment.
const DEFAULT_LUT_CUBE_EDGE: u32 = LUT_CUBE_EDGE;

/// Number of frames the renderer keeps in flight. Matches the scanout BO
/// count in `prism-drm::OutputContext`; the kernel only ever has one flip
/// pending and one currently scanning out, so two frames-worth of GPU
/// resources is the right size.
pub const FRAMES_IN_FLIGHT: usize = 2;

/// Timestamps written per frame by the GPU profiler: start, after decode,
/// after encode. `decode = t1 - t0`, `encode = t2 - t1`.
const PROFILE_TIMESTAMP_COUNT: u32 = 3;

/// Cap on the number of separate scissored encode passes per frame. Beyond this,
/// the encode falls back to a single bounding-box pass — the per-pass overhead
/// (begin/end rendering) stops being worth the saved fragment work. Sized for the
/// common cases (a window region, a 4-rect border ring) with headroom.
const MAX_ENCODE_PASSES: usize = 8;

/// Cap on the number of separate scissored decode passes per frame. As
/// [`MAX_ENCODE_PASSES`], but decode passes are heavier (a per-element draw loop +
/// an inter-pass barrier each), so the bbox fallback matters more for busy frames.
const MAX_DECODE_PASSES: usize = 8;

/// Per-frame GPU time for the two compositing passes, in microseconds.
/// Produced by the timestamp-query profiler when the `PRISM_GPU_PROFILE`
/// environment variable is set; `None` everywhere otherwise (no query pools
/// are even allocated). This is prism's *own* per-output cost — what it adds
/// on top of whatever the client rendered — isolated from app GPU load.
#[derive(Clone, Copy, Debug)]
pub struct GpuFrameTiming {
    pub decode_us: f32,
    pub encode_us: f32,
}

/// Outcome of [`Renderer::render_frame`].
pub struct RenderedFrame {
    /// Present-completion sync as a Linux SYNC_FD — the caller hands this to
    /// the DRM atomic commit as `IN_FENCE_FD`.
    pub present_sync: OwnedFd,
    /// Whether the encode pass covered the full output this frame (vs a
    /// buffer-age sub-region). Lets the caller keep its per-BO damage carry
    /// exact: a full encode makes every *other* scanout BO fully stale.
    pub encoded_full: bool,
}

/// Timestamp-query GPU profiler. Brackets the decode and encode passes with
/// `vkCmdWriteTimestamp2` into a per-slot query pool, reads the prior frame's
/// result back once the slot's fence proves completion (no GPU stall), and
/// exposes a 1 Hz EWMA report. Fully disabled — zero commands, no pools —
/// unless `PRISM_GPU_PROFILE` is set, so production paths pay nothing.
struct GpuProfiler {
    enabled: bool,
    /// Nanoseconds per timestamp tick (`limits.timestampPeriod`).
    period_ns: f32,
    /// Valid-bit mask for timestamp values on the graphics queue family.
    mask: u64,
    /// Smoothed (decode_us, encode_us); `None` until the first sample.
    ewma: Option<(f32, f32)>,
    last_log: std::time::Instant,
    /// Throttled report, set at most once per second; drained by the caller.
    report: Option<GpuFrameTiming>,
}

impl GpuProfiler {
    fn new(device: &Device) -> Self {
        let want = std::env::var_os("PRISM_GPU_PROFILE").is_some();
        let period_ns = device.physical.properties.limits.timestamp_period;
        // Timestamp validity is per queue family.
        let valid_bits = unsafe {
            device
                .instance_raw()
                .get_physical_device_queue_family_properties(device.physical.raw)
        }
        .get(device.physical.graphics_queue_family as usize)
        .map(|q| q.timestamp_valid_bits)
        .unwrap_or(0);
        let enabled = want && period_ns > 0.0 && valid_bits > 0;
        if want && !enabled {
            tracing::warn!(
                "PRISM_GPU_PROFILE set but the graphics queue lacks timestamp support; \
                 GPU profiling disabled"
            );
        }
        let mask = if valid_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << valid_bits) - 1
        };
        GpuProfiler {
            enabled,
            period_ns,
            mask,
            ewma: None,
            last_log: std::time::Instant::now(),
            report: None,
        }
    }

    /// Fold one frame's three timestamps into the EWMA and, at most once per
    /// second, stage a report for the caller to log.
    fn ingest(&mut self, ts: [u64; PROFILE_TIMESTAMP_COUNT as usize]) {
        let dec = (ts[1] & self.mask).wrapping_sub(ts[0] & self.mask);
        let enc = (ts[2] & self.mask).wrapping_sub(ts[1] & self.mask);
        let to_us = |ticks: u64| (ticks as f32 * self.period_ns) / 1000.0;
        let (d, e) = (to_us(dec), to_us(enc));
        const ALPHA: f32 = 0.1;
        let (ed, ee) = match self.ewma {
            Some((pd, pe)) => (pd + ALPHA * (d - pd), pe + ALPHA * (e - pe)),
            None => (d, e),
        };
        self.ewma = Some((ed, ee));
        if self.last_log.elapsed() >= std::time::Duration::from_secs(1) {
            self.report = Some(GpuFrameTiming {
                decode_us: ed,
                encode_us: ee,
            });
            self.last_log = std::time::Instant::now();
        }
    }
}

/// One element to draw in the decode pass.
pub struct ElementDraw {
    /// Sampled texture (must be in SHADER_READ_ONLY_OPTIMAL layout). For YUV
    /// elements this is the luma plane.
    pub texture_view: vk::ImageView,
    /// Chroma plane for YUV elements (`push.yuv != 0`); `None` for RGB. When
    /// `None`, binding 1 is bound to `texture_view` so it stays valid (the
    /// shader references it statically but ignores it when `yuv == 0`).
    pub chroma_view: Option<vk::ImageView>,
    pub push: DecodePush,
}

/// A request to capture a region of the intermediate into a [`SnapshotTexture`]
/// at the start of the frame, before the decode pass repaints over it — used by
/// the window-close animation (approach B: the copy rides the frame's command
/// buffer, so it's naturally ordered before the decode without a second submit).
pub struct SnapshotCopy {
    /// Destination, sized to `src.extent`. The renderer records a copy into it.
    pub dst: Arc<SnapshotTexture>,
    /// Region of the intermediate (physical pixels) to capture. Caller clamps
    /// it to the output extent.
    pub src: vk::Rect2D,
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
    /// Binary semaphore signalled by the same submission as `fence`, and
    /// exportable as a Linux SYNC_FD via VK_KHR_external_semaphore_fd. We
    /// hand the exported fd to the DRM atomic commit as `IN_FENCE_FD` so
    /// the kernel can schedule the page-flip without blocking on the
    /// dmabuf's implicit-sync reservation. Per spec the export
    /// **unsignals** the semaphore, so on each frame we re-signal it via
    /// the submit then re-export.
    present_semaphore: vk::Semaphore,
    /// Device submission serial of this slot's last submit
    /// ([`Device::note_submit`]). When the slot's fence wait returns, this
    /// serial is reported to [`Device::note_completed`], driving the
    /// deferred-destroy queue. 0 = never submitted (note_completed no-op).
    submit_serial: u64,
    /// Timestamp query pool ([`PROFILE_TIMESTAMP_COUNT`] queries) for this
    /// slot, or `None` when GPU profiling is off. Each slot owns its own pool
    /// so reset/write/read never races another in-flight frame.
    timing_pool: Option<vk::QueryPool>,
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
    /// Per-output 3D color LUT. `Some` whenever the encode pipeline's
    /// chain includes `EncodeFragment::Lut3d`; bound at descriptor set 0 /
    /// binding 1 at every draw. Identity content at construction; replaced
    /// by [`Self::upload_lut3d`] when calibration changes.
    lut3d: Option<Lut3dTexture>,
    /// 1×1 offscreen scratch for `encode_diagnose` — runs the encode
    /// pipeline against a known input, reads back the scanout-format
    /// output. Allocated lazily on first diagnose call so non-
    /// calibration sessions don't pay for it.
    diagnose: Option<EncodeDiagnoseProbe>,
    /// Screen-capture encoder (sRGB capture chain + offscreen readback).
    /// Allocated lazily on first capture so non-capturing sessions don't
    /// pay for the offscreen/readback target. See [`crate::capture`].
    capture: Option<crate::capture::CaptureEncoder>,
    scanout_format: vk::Format,
    intermediate_format: vk::Format,
    command_pool: vk::CommandPool,
    slots: [FrameSlot; FRAMES_IN_FLIGHT],
    /// Index into `slots` for the *next* frame.
    next_slot: usize,
    /// GPU timestamp profiler (no-op unless `PRISM_GPU_PROFILE` is set).
    profiler: GpuProfiler,
    /// Loader for VK_KHR_external_semaphore_fd — exports the per-slot
    /// `present_semaphore` as a Linux sync_file fd for KMS.
    semaphore_fd_loader: external_semaphore_fd::Device,
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
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        // One exportable binary semaphore per slot. Marked exportable as
        // SYNC_FD via the pNext export-info chain — required by Vulkan to
        // later call vkGetSemaphoreFdKHR. Starts unsignalled (the default
        // for VkSemaphore); the first signal happens at the first submit
        // that uses the slot.
        let mut export_info = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut export_info);
        let profiler = GpuProfiler::new(&device);
        let mut slots = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for cb in cbs {
            let fence = unsafe { device.raw.create_fence(&fence_info, None) }
                .vk_ctx("create_fence (renderer slot)")?;
            let present_semaphore = unsafe { device.raw.create_semaphore(&sem_info, None) }
                .vk_ctx("create_semaphore (renderer slot, exportable SYNC_FD)")?;
            let timing_pool = if profiler.enabled {
                let info = vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(PROFILE_TIMESTAMP_COUNT);
                Some(
                    unsafe { device.raw.create_query_pool(&info, None) }
                        .vk_ctx("create_query_pool (gpu profile)")?,
                )
            } else {
                None
            };
            slots.push(FrameSlot {
                cmd_buffer: cb,
                fence,
                present_semaphore,
                submit_serial: 0,
                timing_pool,
            });
        }
        let slots: [FrameSlot; FRAMES_IN_FLIGHT] = slots
            .try_into()
            .map_err(|_| crate::error::RendererError::MissingFeature("FrameSlot collect"))?;

        let semaphore_fd_loader =
            external_semaphore_fd::Device::new(device.instance_raw(), &device.raw);

        // Solid-color element scratch — one 1×1 white texel, uploaded once.
        let mut white_tex = ShmTexture::new(
            device.clone(),
            vk::Extent2D {
                width: 1,
                height: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
        )?;
        // 1×1, uploaded once: pass &[] (new texture → full upload).
        white_tex.upload_bytes(&[255, 255, 255, 255], 4, &[])?;

        // Per-output 3D LUT — only allocate when the configured encode
        // chain actually samples it. Default content at construction
        // matches the chain's LUT-output domain: nits chains (PQ/linear)
        // get the nits identity (pq_oetf → sample → pq_eotf is a visual
        // no-op); drive chains (sRGB) get the nominal nits → drive
        // mapping anchored at 80-nit reference white, so an uncalibrated
        // output renders like a standard sRGB panel. Calibration data
        // (or a per-output reference-white re-synthesis) arrives later
        // via upload_lut3d.
        let lut3d = if encode.uses_lut3d {
            let mut tex = Lut3dTexture::new(device.clone(), DEFAULT_LUT_CUBE_EDGE)?;
            let entries = match encode_config.lut_output_domain() {
                crate::encode_synth::LutOutputDomain::Nits => identity_lut(DEFAULT_LUT_CUBE_EDGE),
                crate::encode_synth::LutOutputDomain::Drive => crate::lut3d::drive_identity_lut(
                    DEFAULT_LUT_CUBE_EDGE,
                    crate::lut3d::DEFAULT_DRIVE_WHITE_NITS,
                ),
            };
            tex.upload(&entries)?;
            Some(tex)
        } else {
            None
        };

        Ok(Self {
            device,
            decode,
            encode,
            intermediate: None,
            white_tex,
            lut3d,
            diagnose: None,
            capture: None,
            scanout_format,
            intermediate_format,
            command_pool,
            slots,
            next_slot: 0,
            profiler,
            semaphore_fd_loader,
        })
    }

    /// Run the encode pipeline against a 1×1 scratch with `input_nits`
    /// as the synthetic intermediate value, read back the scanout-format
    /// output, and decode to linear cd/m². Lets calibration tools
    /// verify the LUT path end-to-end (shader emission + LUT contents +
    /// output transfer) against an independently-computed prediction
    /// — closes the loop the colorimeter alone can't close.
    ///
    /// Lazy-allocates the probe on first call so non-calibration
    /// sessions don't pay for the scratch images.
    pub fn encode_diagnose(
        &mut self,
        input_nits: [f64; 3],
        encode_push: &EncodePush,
    ) -> Result<DiagnosedNits> {
        if self.diagnose.is_none() {
            self.diagnose = Some(EncodeDiagnoseProbe::new(
                self.device.clone(),
                self.intermediate_format,
                self.scanout_format,
            )?);
        }
        let probe = self.diagnose.as_mut().unwrap();
        probe.diagnose(&self.encode, encode_push, self.lut3d.as_ref(), input_nits)
    }

    /// Capture this output's last composited frame as an sRGB image in `format`
    /// (`R8G8B8A8_UNORM` or `B8G8R8A8_UNORM` — the latter fills `Xrgb8888`/
    /// `Argb8888` client buffers without a CPU swizzle) into host memory,
    /// **asynchronously** (the SHM screencopy path).
    ///
    /// Renders the persistent intermediate (the last frame's BT.2020 absolute-
    /// nits composite, still resident) through the capture encode chain — a
    /// colorimetric sRGB target with no panel correction — into an owned
    /// [`HostReadback`](crate::capture::HostReadback), and returns a Linux
    /// `SYNC_FD` that signals on GPU completion plus that buffer. The caller
    /// reads the bytes once the fd fires. `sdr_white_nits` is the output's
    /// reference-white level (`effective_sdr_reference_nits()`).
    ///
    /// Must be called from the render loop right after the output's `present()`
    /// (same ordering requirement as [`Self::capture_into_dmabuf`]). Errors if no
    /// frame has been rendered yet (no intermediate to capture). Lazy-allocates
    /// the capture encoder; rebuilds it if a different `format` is requested.
    pub fn capture_to_host(
        &mut self,
        format: vk::Format,
        sdr_white_nits: f32,
    ) -> Result<(OwnedFd, crate::capture::HostReadback)> {
        let (view, extent) = {
            let intermediate =
                self.intermediate
                    .as_ref()
                    .ok_or(crate::error::RendererError::MissingFeature(
                        "capture: no intermediate (no frame rendered yet)",
                    ))?;
            (intermediate.view, intermediate.extent)
        };
        let need_rebuild = self.capture.as_ref().is_none_or(|c| c.format() != format);
        if need_rebuild {
            self.capture = Some(crate::capture::CaptureEncoder::new(
                self.device.clone(),
                format,
            )?);
        }
        self.capture
            .as_mut()
            .unwrap()
            .capture_to_host_async(view, extent, sdr_white_nits)
    }

    /// Capture this output's last composited frame directly into `dst` — a
    /// dmabuf-backed image (e.g. a `wlr-screencopy` client buffer) imported with
    /// `COLOR_ATTACHMENT` usage and a supported capture format — and return a
    /// Linux `SYNC_FD` that signals when the GPU finishes.
    ///
    /// Non-blocking, zero-copy: unlike [`Self::capture`] there's no host
    /// readback. The caller gates the protocol completion event on the returned
    /// fd and must keep `dst` alive until it fires. `dst`'s extent must equal
    /// this output's (whole-output capture only). Errors if no frame has been
    /// rendered yet.
    pub fn capture_into_dmabuf(
        &mut self,
        dst: &ImportedImage,
        sdr_white_nits: f32,
    ) -> Result<OwnedFd> {
        let (view, extent) = {
            let intermediate =
                self.intermediate
                    .as_ref()
                    .ok_or(crate::error::RendererError::MissingFeature(
                        "capture_into_dmabuf: no intermediate (no frame rendered yet)",
                    ))?;
            (intermediate.view, intermediate.extent)
        };
        if dst.extent().width != extent.width || dst.extent().height != extent.height {
            return Err(crate::error::RendererError::MissingFeature(
                "capture_into_dmabuf: dst extent != output extent (whole-output capture only)",
            ));
        }
        let format = dst.format();
        let need_rebuild = self.capture.as_ref().is_none_or(|c| c.format() != format);
        if need_rebuild {
            self.capture = Some(crate::capture::CaptureEncoder::new(
                self.device.clone(),
                format,
            )?);
        }
        self.capture.as_mut().unwrap().capture_into_dmabuf(
            view,
            dst.image(),
            dst.view(),
            extent,
            sdr_white_nits,
        )
    }

    /// Capture this output's last composited frame — the raw BT.2020
    /// absolute-nits *intermediate* (before LUT / response-curve / OETF
    /// panel correction) — into a memfd, returning the fd + geometry.
    ///
    /// On-demand and synchronous (a single ~one-frame hitch): meant for
    /// the prism-tune frame inspector behind an explicit "fetch frame"
    /// IPC, NOT for the render path. Errors if no frame has been
    /// rendered yet. See [`crate::intermediate_capture`].
    pub fn capture_intermediate(&self) -> Result<crate::intermediate_capture::CapturedFrame> {
        let intermediate =
            self.intermediate
                .as_ref()
                .ok_or(crate::error::RendererError::MissingFeature(
                    "capture_intermediate: no intermediate (no frame rendered yet)",
                ))?;
        crate::intermediate_capture::capture_intermediate_to_memfd(
            &self.device,
            intermediate.image,
            intermediate.extent,
            intermediate.format,
        )
    }

    /// Replace this output's 3D LUT content. No-op (and returns Ok) when
    /// the encode chain doesn't include `EncodeFragment::Lut3d`. `entries`
    /// must be `cube_edge³` RGB triples in linear nits, X-fastest
    /// (see [`crate::lut3d::Lut3dTexture::upload`]).
    pub fn upload_lut3d(&mut self, entries: &[[f32; 3]]) -> Result<()> {
        if let Some(lut) = self.lut3d.as_mut() {
            lut.upload(entries)?;
        }
        Ok(())
    }

    /// Cube edge length of this renderer's 3D LUT, or 0 when none is
    /// allocated. Calibration callers need this to size the entries
    /// vector they pass to [`Self::upload_lut3d`].
    pub fn lut3d_cube_edge(&self) -> u32 {
        self.lut3d.as_ref().map(|l| l.cube_edge()).unwrap_or(0)
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

    /// Take the throttled (≤1 Hz) GPU-timing report, if one is pending. Always
    /// `None` unless `PRISM_GPU_PROFILE` is set. The caller logs it with the
    /// output's identity — this is prism's own per-output decode/encode cost.
    pub fn take_gpu_profile_report(&mut self) -> Option<GpuFrameTiming> {
        self.profiler.report.take()
    }

    /// The renderer's device handle — lets the integrator allocate
    /// `SnapshotTexture`s for the close animation without holding the renderer.
    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }

    /// Allocate a [`SnapshotTexture`] of `extent`, ready to receive a
    /// [`SnapshotCopy`] (a format-converting blit) in the next `render_frame`.
    /// Used by the window-close animation to capture a tile's last composited
    /// frame.
    pub fn create_snapshot(&self, extent: vk::Extent2D) -> Result<SnapshotTexture> {
        SnapshotTexture::new(self.device.clone(), extent)
    }

    /// Ensure the persistent intermediate matches `extent`/format. Returns
    /// `true` if it was (re)allocated this call — meaning its contents are
    /// undefined and the caller must paint the full frame (no preservation).
    fn ensure_intermediate(&mut self, extent: vk::Extent2D) -> Result<bool> {
        if self.intermediate.as_ref().is_some_and(|i| {
            i.extent.width == extent.width
                && i.extent.height == extent.height
                && i.format == self.intermediate_format
        }) {
            return Ok(false);
        }
        self.intermediate = Some(Intermediate::new(
            self.device.clone(),
            extent,
            self.intermediate_format,
        )?);
        Ok(true)
    }

    /// Render one frame into `scanout` (which must match `scanout_format`).
    ///
    /// Waits on the next slot's fence (gates against the GPU still using
    /// its resources from N frames ago), records into its command buffer,
    /// submits signalling both the fence (for slot reuse) and the slot's
    /// binary `present_semaphore` (exported below). Does NOT wait for the
    /// GPU to finish.
    ///
    /// Returns the present-completion sync as a Linux SYNC_FD `OwnedFd` —
    /// the caller passes it to the DRM atomic commit as `IN_FENCE_FD` so
    /// the kernel sequences the page-flip after the GPU finishes writing
    /// the scanout BO, without falling back to dmabuf implicit-sync
    /// (which makes `page_flip` itself block).
    #[allow(clippy::too_many_arguments)] // cohesive low-level frame-submit entry point
    pub fn render_frame(
        &mut self,
        scanout: &ImportedImage,
        elements: &[ElementDraw],
        // Damaged regions in physical pixels (from the per-output DamageTracker).
        // The decode pass repaints only their bounding box, preserving the rest
        // of the persistent intermediate. Empty ⇒ nothing changed ⇒ the decode
        // pass is skipped entirely (the intermediate already holds the correct
        // composite). Ignored on the first frame / after a realloc, where the
        // intermediate is uninitialized and must be painted in full.
        damage: &[Rectangle<i32, Physical>],
        // Regions of *this scanout BO* to re-encode, in physical pixels. The BO
        // is buffer-age stale (the caller renders into alternating BOs, so each
        // is ~2 presents behind), so this is the caller's per-BO accumulated
        // damage union — what this particular BO is missing. The encode pass
        // scissors to its bounding box; the rest of the BO keeps its prior
        // content. Empty ⇒ full-output encode (first use of the BO, or the
        // caller forcing a full repaint). `force_full*` also force full.
        encode_damage: &[Rectangle<i32, Physical>],
        encode_push: &EncodePush,
        // Binary semaphores the render must wait on before sampling — used
        // for cross-GPU mirror copies (a home GPU's copy into the shared
        // scratch must complete before this GPU samples it). Empty for the
        // common case (native textures, no mirror). Consumed by the wait;
        // the caller hands them to `Device::retire` after this returns (the
        // spec forbids destroying a semaphore before the batch waiting on
        // it completes execution — the deferred-destroy queue holds them
        // until this submission's slot fence proves that).
        wait_semaphores: &[vk::Semaphore],
        // Window-close snapshots to capture from the intermediate this frame,
        // recorded before the decode pass (which would otherwise repaint over
        // the region). Empty in the common case.
        snapshots: &[SnapshotCopy],
        // Repaint the whole frame instead of just the damage bbox. Set while a
        // closing window animates: it leaves a large clear-only damage ring
        // (the area the shrinking snapshot vacates), and a *sub-region*
        // `load_op = CLEAR` of the persistent intermediate doesn't reliably
        // clear it on radv (partial fast-clear / DCC). A full-frame decode
        // clears reliably; the cost is bounded (close animations are brief).
        force_full_repaint: bool,
    ) -> Result<RenderedFrame> {
        let extent = scanout.extent();
        // `force_full`: the intermediate was just (re)allocated, so its contents
        // are undefined and the whole frame must be painted regardless of damage.
        let force_full = self.ensure_intermediate(extent)?;
        let full_area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        };
        // Per-rect render-pass plans. Both passes scissor to the individual damage
        // rects (disjoint regions like a border ring around an opaque window each
        // cost only their own area, not their bounding box); past the cap they
        // collapse to one bbox pass to bound per-pass overhead. `force_full*` /
        // realloc force the whole output.
        //
        // Decode repaints the persistent intermediate; an empty plan ⇒ skip decode
        // (no damage, intermediate already valid). Encode targets the scanout BO;
        // an empty plan ⇒ full encode (first use / forced), which also selects the
        // discard-from-UNDEFINED mid-barrier below.
        let decode_passes: Vec<vk::Rect2D> = if force_full || force_full_repaint {
            vec![full_area]
        } else {
            plan_passes(damage, extent, MAX_DECODE_PASSES)
        };
        let decoded = !decode_passes.is_empty();

        let encode_planned: Vec<vk::Rect2D> = if force_full || force_full_repaint {
            Vec::new()
        } else {
            plan_passes(encode_damage, extent, MAX_ENCODE_PASSES)
        };
        let encode_full = encode_planned.is_empty();
        let encode_passes: Vec<vk::Rect2D> = if encode_full {
            vec![full_area]
        } else {
            encode_planned
        };
        let intermediate = self.intermediate.as_ref().unwrap();

        let slot_idx = self.next_slot;
        let (slot_fence, slot_cb, slot_present_semaphore, slot_prev_serial) = {
            let slot = &self.slots[slot_idx];
            (
                slot.fence,
                slot.cmd_buffer,
                slot.present_semaphore,
                slot.submit_serial,
            )
        };

        // Wait for this slot's previous use to finish. With N=2 frames in
        // flight and a 60Hz vblank cadence, this is essentially free —
        // the GPU has had ~16ms+ to drain.
        unsafe {
            self.device
                .raw
                .wait_for_fences(&[slot_fence], true, u64::MAX)
        }
        .vk_ctx("wait_for_fences (slot)")?;
        unsafe { self.device.raw.reset_fences(&[slot_fence]) }.vk_ctx("reset_fences (slot)")?;
        // This slot's previous submission has provably completed — advance the
        // device's deferred-destroy queue (frees retired dmabuf imports, shm
        // textures, wait semaphores, … that no in-flight frame references).
        self.device.note_completed(slot_prev_serial);

        // The fence above proves this slot's prior frame (incl. its timestamp
        // writes) finished, so the query results are available without a GPU
        // stall. Read them back before we reset the pool for this frame.
        if let (Some(pool), true) = (self.slots[slot_idx].timing_pool, slot_prev_serial != 0) {
            let mut ts = [0u64; PROFILE_TIMESTAMP_COUNT as usize];
            let got = unsafe {
                self.device.raw.get_query_pool_results(
                    pool,
                    0,
                    &mut ts,
                    vk::QueryResultFlags::TYPE_64,
                )
            };
            if got.is_ok() {
                self.profiler.ingest(ts);
            }
        }

        let cb = slot_cb;
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

        // GPU profiling: reset this slot's timestamp pool up front (must be
        // outside any render pass). `t0` is written just before the decode
        // pass below, `t1` after it, `t2` after encode — so snapshot copies
        // are excluded and we measure the two compositing passes proper.
        let timing_pool = self.slots[slot_idx].timing_pool;
        if let Some(pool) = timing_pool {
            unsafe {
                self.device
                    .raw
                    .cmd_reset_query_pool(cb, pool, 0, PROFILE_TIMESTAMP_COUNT);
            }
        }

        // ── Snapshot capture (window close) ─────────────────────────────────
        // Copy each requested region out of the intermediate *before* the decode
        // pass repaints over it. The intermediate holds last frame's composite
        // here (it's persistent and still in SHADER_READ_ONLY from last frame's
        // encode) — except on the first frame / realloc (`force_full`), where it
        // is undefined and there's nothing to capture; then we clear the
        // snapshots to transparent so the close replay simply draws nothing.
        if !snapshots.is_empty() {
            let subresource = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1);

            if force_full {
                // Bring each snapshot to TRANSFER_DST, clear, then to SHADER_READ.
                let to_dst: Vec<_> = snapshots
                    .iter()
                    .map(|s| {
                        barrier_image(
                            s.dst.image(),
                            vk::ImageLayout::UNDEFINED,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            vk::PipelineStageFlags2::TOP_OF_PIPE,
                            vk::AccessFlags2::empty(),
                            vk::PipelineStageFlags2::ALL_TRANSFER,
                            vk::AccessFlags2::TRANSFER_WRITE,
                        )
                    })
                    .collect();
                unsafe {
                    self.device.raw.cmd_pipeline_barrier2(
                        cb,
                        &vk::DependencyInfo::default().image_memory_barriers(&to_dst),
                    );
                    let clear = vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    };
                    let range = vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    };
                    for s in snapshots {
                        self.device.raw.cmd_clear_color_image(
                            cb,
                            s.dst.image(),
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &clear,
                            &[range],
                        );
                    }
                }
            } else {
                // Pre-copy: intermediate SHADER_READ → TRANSFER_SRC, each
                // snapshot UNDEFINED → TRANSFER_DST.
                let mut pre = vec![barrier_image(
                    intermediate.image,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                    vk::PipelineStageFlags2::ALL_TRANSFER,
                    vk::AccessFlags2::TRANSFER_READ,
                )];
                for s in snapshots {
                    pre.push(barrier_image(
                        s.dst.image(),
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::AccessFlags2::empty(),
                        vk::PipelineStageFlags2::ALL_TRANSFER,
                        vk::AccessFlags2::TRANSFER_WRITE,
                    ));
                }
                unsafe {
                    self.device.raw.cmd_pipeline_barrier2(
                        cb,
                        &vk::DependencyInfo::default().image_memory_barriers(&pre),
                    );
                    for s in snapshots {
                        // Blit, not copy: the snapshot is fp16 while the
                        // intermediate is fp32, so the capture must format-
                        // convert (a same-size 1:1 blit, NEAREST filter — no
                        // scaling here; the *replay* scales later, sampling the
                        // fp16 snapshot, which supports linear filtering).
                        let region = vk::ImageBlit::default()
                            .src_subresource(subresource)
                            .src_offsets([
                                vk::Offset3D {
                                    x: s.src.offset.x,
                                    y: s.src.offset.y,
                                    z: 0,
                                },
                                vk::Offset3D {
                                    x: s.src.offset.x + s.src.extent.width as i32,
                                    y: s.src.offset.y + s.src.extent.height as i32,
                                    z: 1,
                                },
                            ])
                            .dst_subresource(subresource)
                            .dst_offsets([
                                vk::Offset3D { x: 0, y: 0, z: 0 },
                                vk::Offset3D {
                                    x: s.src.extent.width as i32,
                                    y: s.src.extent.height as i32,
                                    z: 1,
                                },
                            ]);
                        self.device.raw.cmd_blit_image(
                            cb,
                            intermediate.image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            s.dst.image(),
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[region],
                            vk::Filter::NEAREST,
                        );
                    }
                }
                // Post-copy: restore intermediate to SHADER_READ (the decode
                // pass barrier below assumes that), each snapshot → SHADER_READ.
                let mut post = vec![barrier_image(
                    intermediate.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::PipelineStageFlags2::ALL_TRANSFER,
                    vk::AccessFlags2::TRANSFER_READ,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                )];
                for s in snapshots {
                    post.push(barrier_image(
                        s.dst.image(),
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags2::ALL_TRANSFER,
                        vk::AccessFlags2::TRANSFER_WRITE,
                        vk::PipelineStageFlags2::FRAGMENT_SHADER,
                        vk::AccessFlags2::SHADER_SAMPLED_READ,
                    ));
                }
                unsafe {
                    self.device.raw.cmd_pipeline_barrier2(
                        cb,
                        &vk::DependencyInfo::default().image_memory_barriers(&post),
                    );
                }
            }

            // The force_full branch left snapshots in TRANSFER_DST; finish the
            // transition to SHADER_READ so the decode pass can sample them.
            if force_full {
                let to_read: Vec<_> = snapshots
                    .iter()
                    .map(|s| {
                        barrier_image(
                            s.dst.image(),
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            vk::PipelineStageFlags2::ALL_TRANSFER,
                            vk::AccessFlags2::TRANSFER_WRITE,
                            vk::PipelineStageFlags2::FRAGMENT_SHADER,
                            vk::AccessFlags2::SHADER_SAMPLED_READ,
                        )
                    })
                    .collect();
                unsafe {
                    self.device.raw.cmd_pipeline_barrier2(
                        cb,
                        &vk::DependencyInfo::default().image_memory_barriers(&to_read),
                    );
                }
            }
        }

        // GPU profile t0: start of the decode pass.
        if let Some(pool) = timing_pool {
            unsafe {
                self.device.raw.cmd_write_timestamp2(
                    cb,
                    vk::PipelineStageFlags2::TOP_OF_PIPE,
                    pool,
                    0,
                );
            }
        }

        // ── Decode pass (per damage rect) ───────────────────────────────────
        // Repaint only the damaged regions of the persistent intermediate,
        // preserving the rest (last frame's composite stays valid). One render
        // pass per `decode_passes` rect, so a border ring around an opaque window
        // repaints only the ring, not its bounding box. Skipped entirely when
        // there's no damage and the intermediate is valid.
        if decoded {
            // Bring the intermediate to COLOR_ATTACHMENT once. On the full-paint
            // frame its prior contents are undefined (discard from UNDEFINED);
            // otherwise preserve them — the previous frame's encode left it in
            // SHADER_READ_ONLY_OPTIMAL, and we only overwrite the damaged rects.
            let (old_layout, src_stage, src_access) = if force_full {
                (
                    vk::ImageLayout::UNDEFINED,
                    vk::PipelineStageFlags2::TOP_OF_PIPE,
                    vk::AccessFlags2::empty(),
                )
            } else {
                (
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                )
            };
            let pre_intermediate = [barrier_image(
                intermediate.image,
                old_layout,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                src_stage,
                src_access,
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

            // CLEAR scopes to each pass's render_area, so only the damaged rect is
            // cleared; pixels outside it keep last frame's composite.
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
            // Viewport spans the whole output (element vertices are in clip space
            // for the full framebuffer); each pass's scissor confines fragments to
            // its rect. Pipeline + viewport are command-buffer state that persists
            // across the per-pass render passes, so bind them once.
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            unsafe {
                self.device.raw.cmd_bind_pipeline(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.decode.pipeline,
                );
                self.device.raw.cmd_set_viewport(cb, 0, &[viewport]);

                for (pass_idx, area) in decode_passes.iter().enumerate() {
                    // `plan_passes` returns a disjoint cover, so the passes don't
                    // actually overlap; this write-after-write barrier is cheap
                    // insurance that keeps each pass's CLEAR + recomposite correct
                    // even if that invariant ever changes. (≤8 small passes.)
                    if pass_idx > 0 {
                        let waw = [vk::MemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                            .dst_access_mask(
                                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                                    | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                            )];
                        self.device.raw.cmd_pipeline_barrier2(
                            cb,
                            &vk::DependencyInfo::default().memory_barriers(&waw),
                        );
                    }
                    let render_info = vk::RenderingInfo::default()
                        .render_area(*area)
                        .layer_count(1)
                        .color_attachments(&color_attach);
                    self.device.raw.cmd_begin_rendering(cb, &render_info);
                    self.device.raw.cmd_set_scissor(cb, 0, &[*area]);

                    for el in elements {
                        // Skip elements whose quad doesn't touch this rect — they
                        // would be scissored to zero fragments anyway, so this just
                        // drops the wasted descriptor push + draw call.
                        if !clip_rect_overlaps(el.push.dst_rect_clip, *area, extent) {
                            continue;
                        }
                        let luma_info = [vk::DescriptorImageInfo::default()
                            .sampler(self.decode.sampler)
                            .image_view(el.texture_view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                        // Binding 1 = chroma for YUV; for RGB bind the same view as
                        // binding 0 so the statically-referenced sampler stays valid.
                        let chroma_info = [vk::DescriptorImageInfo::default()
                            .sampler(self.decode.sampler)
                            .image_view(el.chroma_view.unwrap_or(el.texture_view))
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                        let writes = [
                            self.decode.write_texture_binding(0, &luma_info),
                            self.decode.write_texture_binding(1, &chroma_info),
                        ];
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
                    }

                    self.device.raw.cmd_end_rendering(cb);
                }
            }
        }

        // GPU profile t1: decode done (or skipped → t1 ≈ t0).
        if let Some(pool) = timing_pool {
            unsafe {
                self.device.raw.cmd_write_timestamp2(
                    cb,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    pool,
                    1,
                );
            }
        }

        // ── Barrier: scanout → COLOR_ATTACHMENT; intermediate → SHADER_READ ──
        // The scanout always becomes the encode target. The intermediate only
        // needs the COLOR→SHADER_READ transition when we actually decoded into
        // it this frame; when the decode was skipped it is still in SHADER_READ
        // from the previous frame.
        // Full encode discards the BO's prior contents (UNDEFINED→COLOR — every
        // pixel is overwritten). Partial encode preserves the unrendered
        // remainder (GENERAL→COLOR): the previous frame's final barrier left the
        // BO in GENERAL, and only an UNDEFINED old-layout would discard it.
        let (scan_old_layout, scan_src_stage, scan_src_access) = if encode_full {
            (
                vk::ImageLayout::UNDEFINED,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::empty(),
            )
        } else {
            (
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ,
            )
        };
        let mut mid_barriers = vec![barrier_image(
            scanout.image(),
            scan_old_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            scan_src_stage,
            scan_src_access,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        )];
        if decoded {
            mid_barriers.push(barrier_image(
                intermediate.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
            ));
        }
        unsafe {
            self.device.raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&mid_barriers),
            );
        }

        // ── Encode pass ───────────────────────────────────────────────────
        // One small render pass per `encode_passes` rect (disjoint damage regions
        // — e.g. a border ring around an opaque window — so each pays only for its
        // own area, not their bounding box). The pipeline / descriptors / push
        // constants / viewport are command-buffer state that persists across
        // `cmd_begin_rendering` boundaries, so bind them once and loop the passes.
        // Each pass's full-screen triangle covers its render area entirely → no
        // LOAD needed; areas outside every pass keep the BO's prior content.
        let encode_color_attach = [vk::RenderingAttachmentInfo::default()
            .image_view(scanout.view())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let intermediate_info = [vk::DescriptorImageInfo::default()
            .sampler(self.encode.sampler)
            .image_view(intermediate.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        // Binding 1 (LUT) is conditional on the encode chain including
        // `EncodeFragment::Lut3d`. We stage the LUT descriptor info outside the
        // optional so its lifetime covers the push_descriptor call.
        let lut_info = self.lut3d.as_ref().map(|lut| {
            [vk::DescriptorImageInfo::default()
                .sampler(self.encode.sampler)
                .image_view(lut.view())
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
        });
        let mut writes = vec![self.encode.write_intermediate_binding(&intermediate_info)];
        if let Some(ref info) = lut_info {
            writes.push(self.encode.write_lut3d_binding(info));
        }
        unsafe {
            self.device.raw.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.encode.pipeline,
            );
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
            // Viewport spans the whole output (the triangle is in full-framebuffer
            // clip space); each pass's scissor confines fragments to its rect.
            self.device.raw.cmd_set_viewport(cb, 0, &[viewport]);
            for pass in &encode_passes {
                let encode_render_info = vk::RenderingInfo::default()
                    .render_area(*pass)
                    .layer_count(1)
                    .color_attachments(&encode_color_attach);
                self.device.raw.cmd_begin_rendering(cb, &encode_render_info);
                self.device.raw.cmd_set_scissor(cb, 0, &[*pass]);
                self.device.raw.cmd_draw(cb, 3, 1, 0, 0);
                self.device.raw.cmd_end_rendering(cb);
            }
        }

        // GPU profile t2: encode done. decode = t1-t0, encode = t2-t1.
        if let Some(pool) = timing_pool {
            unsafe {
                self.device.raw.cmd_write_timestamp2(
                    cb,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    pool,
                    2,
                );
            }
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

        let cb_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
        // Signal the slot's exportable binary semaphore alongside the
        // fence. The fence stays for our internal slot-reuse gate; the
        // semaphore exists so we can export a sync_file fd handle to KMS.
        let signal_sem = [vk::SemaphoreSubmitInfo::default()
            .semaphore(slot_present_semaphore)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        // Wait on any cross-GPU mirror copies before the decode pass samples
        // their scratch textures. Fragment-shader stage = where the decode
        // pipeline samples.
        let wait_sems: Vec<vk::SemaphoreSubmitInfo> = wait_semaphores
            .iter()
            .map(|&s| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(s)
                    .stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            })
            .collect();
        let submit = [vk::SubmitInfo2::default()
            .command_buffer_infos(&cb_infos)
            .wait_semaphore_infos(&wait_sems)
            .signal_semaphore_infos(&signal_sem)];
        let serial = self.device.note_submit();
        unsafe {
            self.device
                .raw
                .queue_submit2(self.device.graphics_queue, &submit, slot_fence)
        }
        .vk_ctx("queue_submit2 (renderer)")?;
        self.slots[slot_idx].submit_serial = serial;

        // Export the just-signalled semaphore as a Linux sync_file fd.
        // Per VK_KHR_external_semaphore_fd spec, the export transfers
        // ownership of the underlying sync state to the returned fd and
        // unsignals the VkSemaphore — so the next queue_submit2 for this
        // slot is free to re-signal it.
        let get_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(slot_present_semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let raw_fd = unsafe { self.semaphore_fd_loader.get_semaphore_fd(&get_info) }
            .vk_ctx("vkGetSemaphoreFdKHR (SYNC_FD)")?;
        if raw_fd < 0 {
            return Err(RendererError::MissingFeature(
                "vkGetSemaphoreFdKHR returned a negative fd",
            ));
        }
        let present_sync_fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

        // Advance to the next slot. No GPU wait — the next call to
        // render_frame will wait on its slot's fence as needed.
        self.next_slot = (slot_idx + 1) % FRAMES_IN_FLIGHT;
        Ok(RenderedFrame {
            present_sync: present_sync_fd,
            encoded_full: encode_full,
        })
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            // Drain all outstanding work before tearing down the pool / fences.
            let _ = self.device.raw.device_wait_idle();
            for slot in &self.slots {
                self.device.raw.destroy_fence(slot.fence, None);
                self.device
                    .raw
                    .destroy_semaphore(slot.present_semaphore, None);
                if let Some(pool) = slot.timing_pool {
                    self.device.raw.destroy_query_pool(pool, None);
                }
            }
            self.device
                .raw
                .destroy_command_pool(self.command_pool, None);
        }
    }
}

/// Bounding box of the damage rects, clipped to the output. `None` if there is
/// no damage, or it lies entirely outside the output. The decode pass repaints
/// this single box (per-rect scissoring is a later refinement).
fn damage_bbox(damage: &[Rectangle<i32, Physical>], extent: vk::Extent2D) -> Option<vk::Rect2D> {
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    for r in damage {
        min_x = min_x.min(r.loc.x);
        min_y = min_y.min(r.loc.y);
        max_x = max_x.max(r.loc.x + r.size.w);
        max_y = max_y.max(r.loc.y + r.size.h);
    }
    let x0 = min_x.max(0);
    let y0 = min_y.max(0);
    let x1 = max_x.min(extent.width as i32);
    let y1 = max_y.min(extent.height as i32);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(vk::Rect2D {
        offset: vk::Offset2D { x: x0, y: y0 },
        extent: vk::Extent2D {
            width: (x1 - x0) as u32,
            height: (y1 - y0) as u32,
        },
    })
}

/// Plan the per-rect render passes for a set of physical damage rects: reduce the
/// input to a *disjoint* cover of its union, clamp each piece to `extent`, and if
/// more than `max` survive, collapse to a single bounding-box pass (bounding the
/// per-pass overhead). An empty result means no on-screen damage — the caller
/// decides whether that skips the pass or forces a full one.
///
/// Making the cover disjoint matters even though both passes are idempotent under
/// overlap: an overlapping pass *redoes* the work. The buffer-age encode region is
/// `carry ∪ this-frame-damage`, and on a steady region (a bare wallpaper, or a
/// border ring that changes the same way each frame) those are identical — so
/// without this, the encode would run the whole region twice every frame.
fn plan_passes(
    rects: &[Rectangle<i32, Physical>],
    extent: vk::Extent2D,
    max: usize,
) -> Vec<vk::Rect2D> {
    // Accumulate a disjoint set: each rect contributes only the part not already
    // covered. `subtract_rects` can split a rect into several pieces; the `max`
    // fallback below bounds any blow-up.
    let mut disjoint: Vec<Rectangle<i32, Physical>> = Vec::new();
    for &r in rects {
        if disjoint.is_empty() {
            disjoint.push(r);
        } else {
            disjoint.extend(r.subtract_rects(disjoint.iter().copied()));
        }
    }
    let clamped: Vec<vk::Rect2D> = disjoint
        .iter()
        .filter_map(|r| clamp_rect_to_extent(r, extent))
        .collect();
    if clamped.len() > max {
        match damage_bbox(rects, extent) {
            Some(b) => vec![b],
            None => Vec::new(),
        }
    } else {
        clamped
    }
}

/// Whether an element's clip-space quad (`dst_rect_clip`, `[x0,y0,x1,y1]` in
/// `[-1,1]` over the full framebuffer) touches the physical-pixel `area`. Used to
/// skip decode draws that a pass's scissor would discard anyway. The quad is
/// converted to pixels and rounded *outward*, so a touching element is never
/// dropped (over-inclusion only costs a redundant, fully-scissored draw).
fn clip_rect_overlaps(dst_rect_clip: [f32; 4], area: vk::Rect2D, extent: vk::Extent2D) -> bool {
    let to_px = |c: f32, dim: u32| (c + 1.0) * 0.5 * dim as f32;
    let xa = to_px(dst_rect_clip[0], extent.width);
    let xb = to_px(dst_rect_clip[2], extent.width);
    let ya = to_px(dst_rect_clip[1], extent.height);
    let yb = to_px(dst_rect_clip[3], extent.height);
    let x0 = xa.min(xb).floor().max(0.0) as i32;
    let x1 = (xa.max(xb).ceil() as i32).min(extent.width as i32);
    let y0 = ya.min(yb).floor().max(0.0) as i32;
    let y1 = (ya.max(yb).ceil() as i32).min(extent.height as i32);
    if x1 <= x0 || y1 <= y0 {
        return false; // degenerate / fully offscreen quad — contributes nothing
    }
    let ax1 = area.offset.x + area.extent.width as i32;
    let ay1 = area.offset.y + area.extent.height as i32;
    x0 < ax1 && area.offset.x < x1 && y0 < ay1 && area.offset.y < y1
}

/// Clamp a physical-pixel damage rect to the output extent, as a `vk::Rect2D`.
/// `None` if it lies entirely outside the output (or is empty).
fn clamp_rect_to_extent(r: &Rectangle<i32, Physical>, extent: vk::Extent2D) -> Option<vk::Rect2D> {
    let x0 = r.loc.x.max(0);
    let y0 = r.loc.y.max(0);
    let x1 = (r.loc.x + r.size.w).min(extent.width as i32);
    let y1 = (r.loc.y + r.size.h).min(extent.height as i32);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(vk::Rect2D {
        offset: vk::Offset2D { x: x0, y: y0 },
        extent: vk::Extent2D {
            width: (x1 - x0) as u32,
            height: (y1 - y0) as u32,
        },
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use prism_frame::{Point, Size};

    fn ext(w: u32, h: u32) -> vk::Extent2D {
        vk::Extent2D {
            width: w,
            height: h,
        }
    }
    fn area(x: i32, y: i32, w: u32, h: u32) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D {
                width: w,
                height: h,
            },
        }
    }
    fn pr(x: i32, y: i32, w: i32, h: i32) -> Rectangle<i32, Physical> {
        Rectangle::new(Point::from((x, y)), Size::from((w, h)))
    }

    #[test]
    fn clip_rect_overlaps_basic() {
        let e = ext(100, 100);
        // Full-screen quad touches any on-screen rect.
        assert!(clip_rect_overlaps(
            [-1., -1., 1., 1.],
            area(90, 90, 10, 10),
            e
        ));
        // Left-half quad ([-1,0] clip x → [0,50] px) overlaps a left rect only.
        assert!(clip_rect_overlaps(
            [-1., -1., 0., 1.],
            area(0, 0, 10, 100),
            e
        ));
        assert!(!clip_rect_overlaps(
            [-1., -1., 0., 1.],
            area(60, 0, 10, 100),
            e
        ));
        // Edge-adjacent (no pixel overlap) → not counted.
        assert!(!clip_rect_overlaps(
            [-1., -1., 0., 1.],
            area(50, 0, 10, 100),
            e
        ));
    }

    #[test]
    fn clamp_rect_to_extent_clips_and_rejects() {
        let e = ext(100, 100);
        let inside = clamp_rect_to_extent(&pr(10, 10, 20, 20), e).unwrap();
        assert_eq!((inside.offset.x, inside.offset.y), (10, 10));
        assert_eq!((inside.extent.width, inside.extent.height), (20, 20));
        let overhang = clamp_rect_to_extent(&pr(90, 90, 20, 20), e).unwrap();
        assert_eq!((overhang.extent.width, overhang.extent.height), (10, 10));
        assert!(clamp_rect_to_extent(&pr(200, 0, 10, 10), e).is_none());
    }

    #[test]
    fn plan_passes_caps_and_filters() {
        let e = ext(100, 100);
        assert!(plan_passes(&[], e, 8).is_empty());
        // Two on-screen rects → two passes.
        assert_eq!(
            plan_passes(&[pr(0, 0, 10, 10), pr(50, 50, 10, 10)], e, 8).len(),
            2
        );
        // Offscreen rects are dropped.
        assert!(plan_passes(&[pr(200, 200, 10, 10)], e, 8).is_empty());
        // Past the cap → a single bounding-box pass.
        let many: Vec<_> = (0..10).map(|i| pr(i * 5, 0, 4, 4)).collect();
        assert_eq!(plan_passes(&many, e, 8).len(), 1);
    }

    #[test]
    fn plan_passes_dedups_overlap() {
        let e = ext(100, 100);
        // Duplicate full-output rects (carry ∪ damage on a bare wallpaper) collapse
        // to a single pass — not two — so the encode runs once, not twice.
        let full = pr(0, 0, 100, 100);
        let passes = plan_passes(&[full, full], e, 8);
        assert_eq!(passes.len(), 1);
        assert_eq!(passes[0].extent, ext(100, 100));
        // Identical sub-rects likewise collapse.
        let r = pr(10, 10, 20, 20);
        assert_eq!(plan_passes(&[r, r], e, 8).len(), 1);
        // A disjoint cover never double-counts area: total ≤ the output.
        let area: i64 = plan_passes(&[full, pr(20, 20, 30, 30)], e, 8)
            .iter()
            .map(|p| p.extent.width as i64 * p.extent.height as i64)
            .sum();
        assert_eq!(area, 100 * 100);
    }
}
