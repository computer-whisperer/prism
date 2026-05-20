//! Per-output runtime state — the integrated thing that owns "this display
//! is currently scanning out, here are all the resources holding it together."
//!
//! This is the closest prism comes to "an output." It bundles:
//!   - The libseat session that holds DRM master.
//!   - The smithay `DrmDevice` + `DrmSurface` for this output.
//!   - The GBM allocator + the BO backing the scanout image.
//!   - The Vulkan `ImportedImage` view of that BO + the DRM framebuffer handle.
//!   - The renderer instance (per-output because its pipelines bake in the
//!     scanout format and the per-output `EncodeConfig`).
//!
//! Lifetime contract: BO + GBM + ImportedImage + DRM device must all outlive
//! the surface; the surface must outlive any frame queued against it. We keep
//! everything as fields of this struct so drop order does the right thing
//! (Rust drops fields in declaration order: surface → image → BO → GBM → drm
//! → session).

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use drm_fourcc::DrmModifier;
use prism_renderer::{
    ElementDraw, EncodeConfig, EncodePush, ImportedImage, Renderer, vk,
};
use smithay::backend::drm::{DrmDevice, DrmDeviceNotifier, DrmSurface, PlaneConfig, PlaneState};
use smithay::reexports::drm::control::framebuffer;
use smithay::utils::{Rectangle, Transform};

use crate::{
    GbmDevice, ScanoutDepth, SeatSession, add_framebuffer_for_bo, pick_by_name,
    pick_first_connected, set_connector_max_bpc,
};

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

/// All the per-output state. Drop releases scanout cleanly.
pub struct OutputContext {
    // Field order matters: surface → image → BO → renderer → GBM → drm → session.
    pub surface: DrmSurface,
    pub fb: framebuffer::Handle,
    pub scanout_image: ImportedImage,
    pub renderer: Renderer,
    /// GBM BO backing the scanout image. Kept alive so the FB handle stays valid.
    _bo: gbm::BufferObject<()>,
    _gbm: GbmDevice,
    /// DRM device — kept alive so the surface, FB, and BOs remain valid.
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
    /// reject with ENOMEM as its event-allocation pool fills; on a 60Hz
    /// timer that's ~3000 errors/minute which locks up the system. Don't
    /// re-enter present() until `mark_vblank()` has been called.
    frame_pending: bool,
}

impl OutputContext {
    /// Bring up the integrated output. Returns `(context, drm_notifier)` — the
    /// caller MUST insert the notifier into a calloop event loop and route
    /// VBlank events back to `mark_vblank()`. Failure to drain notifier events
    /// causes a kernel-side event-allocation cascade that locks the system.
    pub fn new(
        device: Arc<prism_renderer::Device>,
        setup: OutputSetup<'_>,
    ) -> Result<(Self, DrmDeviceNotifier)> {
        let mut session = SeatSession::new()?;
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
        let (bo, dmabuf) = gbm
            .allocate_scanout(
                extent.width,
                extent.height,
                setup.depth.drm_fourcc(),
                &[DrmModifier::Linear],
            )
            .with_context(|| {
                format!(
                    "GBM allocate scanout {}×{} {:?}",
                    extent.width,
                    extent.height,
                    setup.depth.drm_fourcc()
                )
            })?;
        let scanout_image = ImportedImage::import(
            device.clone(),
            &dmabuf,
            setup.vk_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )?;
        let fb = add_framebuffer_for_bo(&drm, &bo)?;

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
                _bo: bo,
                scanout_image,
                renderer,
                fb,
                surface,
                extent,
                connector_name: pick.connector_name,
                mode_set_done: false,
                frame_pending: false,
            },
            drm_notifier,
        ))
    }

    /// Clear the `frame_pending` flag. Call this when the DRM notifier
    /// surfaces a VBlank event for this output's CRTC.
    pub fn mark_vblank(&mut self) {
        self.frame_pending = false;
    }

    /// True if a flip is in flight (`present` will be a no-op).
    pub fn is_frame_pending(&self) -> bool {
        self.frame_pending
    }

    /// Render the supplied `elements` (with the supplied encode parameters)
    /// into the scanout image and submit it for display.
    ///
    /// Returns `Ok(false)` (no-op) if a previous flip is still pending —
    /// the caller should wait for the next VBlank event before retrying.
    /// Returns `Ok(true)` if a frame was submitted.
    ///
    /// The first call does a full atomic `commit` (mode-set). Subsequent
    /// calls do `page_flip` (buffer swap). Both request a vblank event from
    /// the kernel; the caller MUST drain them by feeding the DrmDeviceNotifier
    /// into calloop and routing VBlank back to `mark_vblank()`.
    pub fn present(
        &mut self,
        elements: &[ElementDraw],
        encode_push: &EncodePush,
    ) -> Result<bool> {
        if self.frame_pending {
            return Ok(false);
        }

        self.renderer.render_frame(
            self.scanout_image.image(),
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
                fb: self.fb,
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

impl Drop for OutputContext {
    fn drop(&mut self) {
        // Best-effort scanout clear so the desktop session reclaims a known
        // state. Ignoring the EINVAL we observed earlier — already documented
        // as a follow-up.
        let _ = self.surface.clear();
    }
}
