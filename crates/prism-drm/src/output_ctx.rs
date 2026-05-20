//! Per-output runtime state — the integrated thing that owns "this display
//! is currently scanning out, here are all the resources holding it together."
//!
//! This is the closest prism comes to "an output." It bundles:
//!   - The libseat session that holds DRM master.
//!   - The smithay `DrmDevice` + `DrmSurface` for this output.
//!   - The GBM allocator + TWO BOs (double-buffered) backing the scanout images.
//!   - The Vulkan `ImportedImage` view of each BO + DRM framebuffer handles.
//!   - The renderer instance (per-output because its pipelines bake in the
//!     scanout format and the per-output `EncodeConfig`).
//!
//! Double-buffering rationale: the AMD display engine reads continuously from
//! whatever BO is currently being scanned out. If we render directly into that
//! same BO every frame (single-buffered), the 3D engine writes and the display
//! engine reads contend through implicit synchronization, which on
//! amdgpu+RADV can fully wedge the system at 60Hz (system-wide kernel hang,
//! even input layer stops responding). With two BOs the render targets the
//! *back* buffer while the display reads the *front*; page_flip swaps them
//! at vblank.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use drm_fourcc::DrmModifier;
use prism_renderer::{
    ElementDraw, EncodeConfig, EncodePush, ImportedImage, Renderer, vk,
};
use smithay::backend::drm::{DrmDevice, DrmDeviceNotifier, DrmSurface, PlaneConfig, PlaneState};
use smithay::backend::session::libseat::LibSeatSessionNotifier;
use smithay::reexports::drm::control::framebuffer;
use smithay::utils::{Rectangle, Transform};

use crate::{
    GbmDevice, ScanoutDepth, SeatSession, add_framebuffer_for_bo, pick_by_name,
    pick_first_connected, set_connector_max_bpc,
};

/// Bundle of calloop event sources the caller MUST insert into its loop.
/// Failure to drain either source causes a hard system lockup.
pub struct OutputNotifiers {
    pub drm: DrmDeviceNotifier,
    pub session: LibSeatSessionNotifier,
}

/// Builder-style input to `OutputContext::new`.
pub struct OutputSetup<'a> {
    /// Path to the DRM primary node (e.g. `/dev/dri/card0`).
    pub drm_path: &'a str,
    /// Optional connector name. `None` → first connected.
    pub output_name: Option<&'a str>,
    /// Scanout depth + matching Vulkan format.
    pub depth: ScanoutDepth,
    /// Vulkan format that matches `depth.drm_fourcc()` byte layout.
    pub vk_format: vk::Format,
    /// Intermediate fp16 / fp32 format for the renderer.
    pub intermediate_format: vk::Format,
    /// Per-output encode shader composition.
    pub encode_config: &'a EncodeConfig,
}

/// One BO + Vulkan view + DRM framebuffer handle. Two of these live in
/// `OutputContext` for double buffering. Field order matters for Drop:
/// image (Vulkan) → BO (GBM); the FB is a kernel-side handle freed by the
/// DRM device drop, no Rust-side cleanup needed here.
struct ScanoutBuffer {
    image: ImportedImage,
    _bo: gbm::BufferObject<()>,
    fb: framebuffer::Handle,
}

/// All the per-output state. Drop releases scanout cleanly.
pub struct OutputContext {
    // Field order matters: surface → scanout buffers → renderer → GBM → drm → session.
    pub surface: DrmSurface,
    /// Two-element ring; `back_index` selects which one to render into next.
    /// On first present we render `buffers[0]` and mode-set to it.
    buffers: [ScanoutBuffer; 2],
    /// Which buffer is currently the *back* (safe to render into). After a
    /// successful page-flip the kernel will switch at next vblank; we wait
    /// for `mark_vblank()` to advance this index so we never render into
    /// the buffer the display is actively reading.
    back_index: usize,
    pub renderer: Renderer,
    _gbm: GbmDevice,
    /// DRM device — kept alive so the surface, FBs, and BOs remain valid.
    pub drm: DrmDevice,
    /// libseat session — last out, so the DRM fd stays open through the others' drops.
    pub session: SeatSession,
    /// Width × height in pixels.
    pub extent: vk::Extent2D,
    /// Connector name for logging.
    pub connector_name: String,
    /// Set on first present to switch from `commit` (mode-set) to `page_flip`
    /// (just-swap-fb) for subsequent frames.
    mode_set_done: bool,
    /// True between submitting a page-flip and receiving its vblank event.
    /// Submitting another flip while one is pending causes the kernel to
    /// reject with ENOMEM. Don't re-enter present() until `mark_vblank()`.
    frame_pending: bool,
}

impl OutputContext {
    /// Bring up the integrated output. Returns `(context, OutputNotifiers)`.
    /// Both notifiers MUST be inserted into the caller's calloop event loop.
    pub fn new(
        device: Arc<prism_renderer::Device>,
        setup: OutputSetup<'_>,
    ) -> Result<(Self, OutputNotifiers)> {
        let (mut session, session_notifier) = SeatSession::new()?;
        if !session.is_active() {
            return Err(anyhow!(
                "libseat session not active — must be run from a foreground VT"
            ));
        }
        let drm_fd = session.open_drm(setup.drm_path)?;
        let (mut drm, drm_notifier) = smithay::backend::drm::DrmDevice::new(drm_fd, false)
            .with_context(|| format!("DrmDevice::new({})", setup.drm_path))?;

        let pick = match setup.output_name {
            Some(name) => pick_by_name(&drm, name)?,
            None => pick_first_connected(&drm)?,
        };
        tracing::info!(
            "output bringup: {} mode={}x{}@{}Hz crtc={:?} depth={:?}",
            pick.connector_name,
            pick.mode.size().0,
            pick.mode.size().1,
            pick.mode.vrefresh(),
            pick.crtc,
            setup.depth,
        );

        match set_connector_max_bpc(&drm, pick.connector, setup.depth.max_bpc()) {
            Ok(true) => tracing::info!("connector max bpc set to {}", setup.depth.max_bpc()),
            Ok(false) => tracing::warn!(
                "connector doesn't expose 'max bpc'; link depth driver-controlled"
            ),
            Err(e) => tracing::warn!("set max bpc failed: {e:#}"),
        }

        let surface = drm
            .create_surface(pick.crtc, pick.mode, &[pick.connector])
            .with_context(|| format!("create_surface on {:?}", pick.crtc))?;

        let gbm = GbmDevice::from_device_fd(drm.device_fd().device_fd())?;
        let (w, h) = pick.mode.size();
        let extent = vk::Extent2D {
            width: w as u32,
            height: h as u32,
        };

        // Two scanout buffers (double-buffered).
        let buffer_a =
            alloc_scanout_buffer(&device, &drm, &gbm, &setup, extent, "buffer A")?;
        let buffer_b =
            alloc_scanout_buffer(&device, &drm, &gbm, &setup, extent, "buffer B")?;
        let buffers = [buffer_a, buffer_b];

        let renderer = Renderer::new(
            device.clone(),
            setup.vk_format,
            setup.intermediate_format,
            setup.encode_config,
        )?;

        Ok((
            Self {
                session,
                drm,
                _gbm: gbm,
                buffers,
                back_index: 0,
                renderer,
                surface,
                extent,
                connector_name: pick.connector_name,
                mode_set_done: false,
                frame_pending: false,
            },
            OutputNotifiers {
                drm: drm_notifier,
                session: session_notifier,
            },
        ))
    }

    /// Clear the `frame_pending` flag AND advance the back-buffer index.
    /// Call this when the DRM notifier surfaces a VBlank event for our CRTC.
    ///
    /// At this point the just-flipped buffer is being scanned out; the OTHER
    /// buffer is no longer in use by the display and is safe to render into.
    pub fn mark_vblank(&mut self) {
        self.frame_pending = false;
        // The buffer we just flipped TO is now front. Toggle so back_index
        // points at the *other* one (the new back).
        self.back_index = 1 - self.back_index;
    }

    /// True if a flip is in flight (`present` will be a no-op).
    pub fn is_frame_pending(&self) -> bool {
        self.frame_pending
    }

    /// Render the supplied `elements` (with the supplied encode parameters)
    /// into the *back* scanout image and submit it for display.
    ///
    /// Returns `Ok(false)` (no-op) if a previous flip is still pending —
    /// the caller should wait for the next VBlank event before retrying.
    /// Returns `Ok(true)` if a frame was submitted.
    pub fn present(
        &mut self,
        elements: &[ElementDraw],
        encode_push: &EncodePush,
    ) -> Result<bool> {
        if self.frame_pending {
            return Ok(false);
        }

        let back = &self.buffers[self.back_index];
        self.renderer.render_frame(
            back.image.image(),
            self.extent,
            elements,
            encode_push,
        )?;

        let src = Rectangle::from_size(
            (self.extent.width as i32, self.extent.height as i32).into(),
        )
        .to_f64();
        let dst =
            Rectangle::from_size((self.extent.width as i32, self.extent.height as i32).into());
        let plane_state = [PlaneState {
            handle: self.surface.plane(),
            config: Some(PlaneConfig {
                src,
                dst,
                transform: Transform::Normal,
                alpha: 1.0,
                damage_clips: None,
                fb: back.fb,
                fence: None,
            }),
        }];

        if !self.mode_set_done {
            self.surface
                .commit(plane_state.iter().cloned(), true)
                .context("DrmSurface::commit (initial mode-set)")?;
            self.mode_set_done = true;
        } else {
            self.surface
                .page_flip(plane_state.iter().cloned(), true)
                .context("DrmSurface::page_flip")?;
        }
        self.frame_pending = true;
        Ok(true)
    }
}

fn alloc_scanout_buffer(
    device: &Arc<prism_renderer::Device>,
    drm: &DrmDevice,
    gbm: &GbmDevice,
    setup: &OutputSetup<'_>,
    extent: vk::Extent2D,
    label: &str,
) -> Result<ScanoutBuffer> {
    let (bo, dmabuf) = gbm
        .allocate_scanout(
            extent.width,
            extent.height,
            setup.depth.drm_fourcc(),
            &[DrmModifier::Linear],
        )
        .with_context(|| {
            format!(
                "GBM allocate {} {}×{} {:?}",
                label,
                extent.width,
                extent.height,
                setup.depth.drm_fourcc()
            )
        })?;
    let image = ImportedImage::import(
        device.clone(),
        &dmabuf,
        setup.vk_format,
        vk::ImageUsageFlags::COLOR_ATTACHMENT,
    )?;
    let fb = add_framebuffer_for_bo(drm, &bo)?;
    Ok(ScanoutBuffer {
        image,
        _bo: bo,
        fb,
    })
}

impl Drop for OutputContext {
    fn drop(&mut self) {
        // Best-effort scanout clear so the desktop session reclaims a known
        // state. Ignoring the EINVAL we observed earlier — already documented
        // as a follow-up.
        let _ = self.surface.clear();
    }
}
