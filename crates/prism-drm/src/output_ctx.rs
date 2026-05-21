//! Per-output runtime state — one per active connector.
//!
//! Owns the per-connector scanout pipeline: the DrmSurface (CRTC + mode +
//! connector), the double-buffered scanout BOs (front/back + Vulkan
//! `ImportedImage` view + DRM framebuffer handle for each), and the
//! per-output `Renderer` (one per output because its encode pipeline bakes
//! in the per-output `EncodeConfig`).
//!
//! Does NOT own: the libseat session (per-process, see [`crate::SeatSession`])
//! or the DRM device + GBM (per-card, see [`crate::DrmCardContext`]). Multiple
//! `OutputContext`s on the same card share their card context.
//!
//! Double-buffering rationale: the AMD display engine reads continuously
//! from whatever BO is currently being scanned out. If we render directly
//! into that same BO every frame (single-buffered), the 3D engine writes
//! and the display engine reads contend through implicit synchronization,
//! which on amdgpu+RADV can fully wedge the system at 60Hz (system-wide
//! kernel hang, even input layer stops responding). With two BOs the render
//! targets the *back* buffer while the display reads the *front*; page_flip
//! swaps them at vblank.

use std::sync::Arc;

use anyhow::{Context, Result};
use drm_fourcc::DrmModifier;
use prism_renderer::{
    Device, DrmDevId, ElementDraw, EncodePush, ImportedImage, Renderer, vk,
};
use smithay::backend::drm::{DrmSurface, PlaneConfig, PlaneState};
use smithay::reexports::drm::control::{Mode, connector, crtc, framebuffer};
use smithay::utils::{Rectangle, Transform};

use crate::{
    DrmCardContext, OutputConfig, OutputPick, add_framebuffer_for_bo, breadcrumb::breadcrumb,
    set_connector_max_bpc,
};

/// One BO + Vulkan view + DRM framebuffer handle. Two of these live in
/// `OutputContext` for double buffering. Field order matters for Drop:
/// image (Vulkan) → BO (GBM); the FB is a kernel-side handle freed by the
/// DRM device drop, no Rust-side cleanup needed here.
struct ScanoutBuffer {
    image: ImportedImage,
    _bo: gbm::BufferObject<()>,
    fb: framebuffer::Handle,
}

/// The per-output state. Drop releases scanout cleanly.
///
/// Construction (`OutputContext::new`) takes a pre-opened [`DrmCardContext`]
/// (borrowed for construction only), the [`Arc<Device>`] for the GPU that
/// will render frames for this output, a pre-resolved [`OutputPick`]
/// (connector + crtc + mode + connector_name), and the static [`OutputConfig`].
pub struct OutputContext {
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
    /// Width × height in pixels.
    pub extent: vk::Extent2D,
    /// Active DRM mode (size + vrefresh). Kept so the wayland side can
    /// derive `smithay::output::Mode` without re-querying the connector.
    pub mode: Mode,
    /// Connector name for logging.
    pub connector_name: String,
    /// Connector handle (for routing / config queries).
    pub connector: connector::Handle,
    /// CRTC bound to this output. The vblank event from `DrmDeviceNotifier`
    /// carries the CRTC handle; the main loop uses this to route to the
    /// right OutputContext on a multi-output card.
    pub crtc: crtc::Handle,
    /// DrmDevId of the GPU whose `Device` this output's renderer was built
    /// from. The render path uses this to look up the correct per-GPU
    /// texture import (or per-GPU shm upload) when sampling client surfaces.
    pub gpu_id: DrmDevId,
    /// The static config used at construction. Held so HDR / calibration
    /// reconfig later can read what we currently have.
    pub config: OutputConfig,
    /// Set on first present to switch from `commit` (mode-set) to `page_flip`
    /// (just-swap-fb) for subsequent frames.
    mode_set_done: bool,
    /// True between submitting a page-flip and receiving its vblank event.
    /// Submitting another flip while one is pending causes the kernel to
    /// reject with ENOMEM. Don't re-enter present() until `mark_vblank()`.
    frame_pending: bool,
}

impl OutputContext {
    /// Bring up an output on the given card+GPU with the given connector pick
    /// and static config. Allocates the scanout buffers + builds the renderer.
    ///
    /// The card is borrowed mutably for construction only (smithay's
    /// `DrmDevice::create_surface` takes `&mut`); once allocated, the
    /// OutputContext doesn't reference the card directly (DrmSurface keeps
    /// its own internal handle to the device fd).
    pub fn new(
        card: &mut DrmCardContext,
        device: Arc<Device>,
        pick: OutputPick,
        config: &OutputConfig,
    ) -> Result<Self> {
        let gpu_id = device
            .physical
            .drm_primary
            .or(device.physical.drm_render)
            .ok_or_else(|| {
                anyhow::anyhow!("renderer Device has no DRM node id; cannot build OutputContext")
            })?;

        tracing::info!(
            "output bringup: {} mode={}x{}@{}Hz crtc={:?} depth={:?} gpu={}:{}",
            pick.connector_name,
            pick.mode.size().0,
            pick.mode.size().1,
            pick.mode.vrefresh(),
            pick.crtc,
            config.depth,
            gpu_id.major,
            gpu_id.minor,
        );

        match set_connector_max_bpc(&card.drm, pick.connector, config.depth.max_bpc()) {
            Ok(true) => tracing::info!("connector max bpc set to {}", config.depth.max_bpc()),
            Ok(false) => tracing::warn!(
                "connector doesn't expose 'max bpc'; link depth driver-controlled"
            ),
            Err(e) => tracing::warn!("set max bpc failed: {e:#}"),
        }

        tracing::info!(connector = %pick.connector_name, "OutputContext::new step: create_surface");
        let surface = card
            .drm
            .create_surface(pick.crtc, pick.mode, &[pick.connector])
            .with_context(|| format!("create_surface on {:?}", pick.crtc))?;

        let (w, h) = pick.mode.size();
        let extent = vk::Extent2D {
            width: w as u32,
            height: h as u32,
        };

        // Two scanout buffers (double-buffered). See module doc.
        tracing::info!(connector = %pick.connector_name, "OutputContext::new step: alloc buffer A");
        let buffer_a = alloc_scanout_buffer(&device, card, config, extent, "buffer A")?;
        tracing::info!(connector = %pick.connector_name, "OutputContext::new step: alloc buffer B");
        let buffer_b = alloc_scanout_buffer(&device, card, config, extent, "buffer B")?;
        let buffers = [buffer_a, buffer_b];

        tracing::info!(connector = %pick.connector_name, "OutputContext::new step: Renderer::new");
        let renderer = Renderer::new(
            device.clone(),
            config.vk_format,
            config.intermediate_format,
            &config.encode_config,
        )?;
        tracing::info!(connector = %pick.connector_name, "OutputContext::new step: done");

        Ok(Self {
            surface,
            buffers,
            back_index: 0,
            renderer,
            extent,
            mode: pick.mode,
            connector_name: pick.connector_name,
            connector: pick.connector,
            crtc: pick.crtc,
            gpu_id,
            config: config.clone(),
            mode_set_done: false,
            frame_pending: false,
        })
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
        self.renderer
            .render_frame(&back.image, elements, encode_push)?;

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
    device: &Arc<Device>,
    card: &DrmCardContext,
    config: &OutputConfig,
    extent: vk::Extent2D,
    label: &str,
) -> Result<ScanoutBuffer> {
    let (bo, dmabuf) = card
        .gbm
        .allocate_scanout(
            extent.width,
            extent.height,
            config.depth.drm_fourcc(),
            &[DrmModifier::Linear],
        )
        .with_context(|| {
            format!(
                "GBM allocate {} {}×{} {:?}",
                label,
                extent.width,
                extent.height,
                config.depth.drm_fourcc()
            )
        })?;
    let image = ImportedImage::import(
        device.clone(),
        &dmabuf,
        config.vk_format,
        vk::ImageUsageFlags::COLOR_ATTACHMENT,
    )?;
    let fb = add_framebuffer_for_bo(&card.drm, &bo)?;
    Ok(ScanoutBuffer {
        image,
        _bo: bo,
        fb,
    })
}

impl Drop for OutputContext {
    fn drop(&mut self) {
        // Best-effort scanout clear so the desktop session reclaims a known
        // state. May still fail with EINVAL (smithay's clear_state quirk)
        // or EACCES if libseat released master before us — but the latter
        // is a Drop-order bug; complain loudly so it gets fixed.
        //
        // Breadcrumb wrapping (vs just tracing) because a hang here gets
        // SIGKILLed by the watchdog and tracing's stdio buffer is lost.
        // The breadcrumbs are fsync'd per line so we can attribute the
        // hang to clear() vs to the subsequent implicit field drops.
        breadcrumb(&format!(
            "OutputContext::Drop entry: {} (crtc {:?})",
            self.connector_name, self.crtc
        ));
        let t0 = std::time::Instant::now();
        let clear_res = self.surface.clear();
        breadcrumb(&format!(
            "OutputContext::Drop surface.clear() returned in {}ms: {}",
            t0.elapsed().as_millis(),
            match &clear_res {
                Ok(()) => "Ok".to_string(),
                Err(e) => format!("Err({e})"),
            }
        ));
        match clear_res {
            Ok(()) => tracing::debug!(
                connector = %self.connector_name,
                "OutputContext drop: surface.clear() OK in {}ms",
                t0.elapsed().as_millis()
            ),
            Err(e) => tracing::warn!(
                connector = %self.connector_name,
                "OutputContext drop: surface.clear() failed in {}ms: {e}",
                t0.elapsed().as_millis()
            ),
        }
        // Function returns here → DrmSurface drops, then buffers drop
        // (ImportedImage + GBM BO each), then Renderer drops (persistent
        // CB + fences + Vulkan device). main.rs's per-output "dropped
        // output X in Yms" breadcrumb wraps the entire chain, so if it
        // doesn't fire after our "clear returned" breadcrumb the hang
        // is in one of those implicit drops.
    }
}
