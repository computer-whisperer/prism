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

use std::os::fd::AsFd;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use drm_fourcc::DrmModifier;
use prism_renderer::{
    Device, DrmDevId, ElementDraw, EncodePush, ImportedImage, Renderer,
    synthesize_lut_from_matrix_curve, vk,
};
use smithay::backend::drm::{DrmSurface, PlaneConfig, PlaneState};
use smithay::reexports::drm::control::{Mode, connector, crtc, framebuffer};

use crate::frame_clock::FrameClock;
use smithay::utils::{Rectangle, Transform};

/// Headroom multiplier on the derived per-channel BT.2020 decode clamp.
/// The clamp's purpose is to keep the BT.2020 fp16 intermediate honest
/// — bound to "panel-realizable as BT.2020 content" — not to be the
/// final tone-map. Going slightly above the panel's physical limit is
/// fine: the encoder's response inverse + PQ OETF clamp soft-clip the
/// excess at the panel ceiling. With margin, the buffer absorbs minor
/// color-management drift (e.g. fp roundoff pushing a 100%-channel value
/// to 1.0001 × sdr_white) without hard-clipping content authored in good
/// faith. 1.5× chosen as a conservative default; the precision of the
/// underlying panel-peak fit doesn't need to be tight at this margin.
const BT2020_DECODE_CAP_MARGIN: f32 = 1.5;

use crate::{
    CursorPlane, DrmCardContext, OutputConfig, OutputPick, add_framebuffer_for_bo,
    breadcrumb::{breadcrumb, flip_trace},
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
    /// Runtime color overrides set via IPC (calibration tooling, etc.).
    /// Sticky across config-file reloads — only `OutputAction::ResetColor`
    /// or a compositor restart clears them. When set, the render path
    /// uses these values in preference to the equivalents in `config`.
    pub color_override: ColorOverride,
    /// Set on first present to switch from `commit` (mode-set) to `page_flip`
    /// (just-swap-fb) for subsequent frames.
    mode_set_done: bool,
    /// True between submitting a page-flip and receiving its vblank event.
    /// Submitting another flip while one is pending causes the kernel to
    /// reject with ENOMEM. Don't re-enter present() until `mark_vblank()`.
    frame_pending: bool,
    /// VRR-aware predictor for the next vblank. Updated on every vblank
    /// with the actual presentation time the kernel reports; the redraw
    /// pass reads `next_presentation_time()` to pick the
    /// `target_presentation_time` it hands to clients via
    /// `wp_presentation_feedback`.
    pub frame_clock: FrameClock,
    /// Hardware cursor plane for this output, if the driver exposed
    /// one and we could claim it. `None` ⇒ no cursor on this output
    /// until software cursor lands. The position/visibility/sprite are
    /// driven by `prism_protocols::update_output_cursors`; we just
    /// include its `to_plane_state()` in the page-flip below.
    pub cursor: Option<CursorPlane>,
    /// Parsed EDID — make / model / serial / physical mm size / HDR
    /// capabilities / default primaries. Always populated; fields
    /// inside are `None` when the panel didn't advertise them. Read
    /// at bringup and stashed so per-output config defaulting +
    /// `wl_output` advertisement can pick it up.
    pub edid: crate::EdidInfo,
    /// KMS HDR property handles + currently-installed blob ID.
    /// `Some` whenever the connector exposes HDR_OUTPUT_METADATA +
    /// Colorspace, regardless of whether HDR is *currently* enabled
    /// — we hold the handles either way so toggling on/off later
    /// doesn't need to re-walk properties. `None` if the connector
    /// can't carry HDR signaling (some virtual / dock outputs).
    pub hdr_props: Option<crate::HdrProps>,
    /// The (DRM fourcc, modifier) candidate list this output's scanout
    /// pipeline can directly accept — driven by the same modifier
    /// negotiation that picked the format for our internal buffers.
    /// First entry is the chosen / preferred modifier; LINEAR comes
    /// last as the universal fallback. Exposed so the wayland-side
    /// `wp_linux_dmabuf_v1` feedback path can advertise per-output
    /// direct-scanout-friendly tranches to clients.
    pub scanout_formats: Vec<(drm_fourcc::DrmFourcc, drm_fourcc::DrmModifier)>,
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

        let edid = crate::EdidInfo::read(&card.drm, pick.connector);
        tracing::info!(
            connector = %pick.connector_name,
            "EDID: {}",
            edid.log_line()
        );

        // HDR property discovery. Failing to find the props is not
        // an error — many connectors don't carry HDR signaling.
        let mut hdr_props = match crate::HdrProps::lookup(&card.drm, pick.connector) {
            Ok(Some(p)) => Some(p),
            Ok(None) => {
                if config.hdr.is_some() {
                    tracing::warn!(
                        connector = %pick.connector_name,
                        "HDR configured but connector exposes no HDR_OUTPUT_METADATA / Colorspace"
                    );
                }
                None
            }
            Err(e) => {
                tracing::warn!(
                    connector = %pick.connector_name,
                    "HDR property lookup failed: {e:#}"
                );
                None
            }
        };
        // Either install or explicitly clear at bringup — clearing
        // is what prevents stickiness from the prior session
        // (phase-1 "DP-4 stuck on PQ" bug). Both branches no-op
        // gracefully if hdr_props is None.
        if let Some(props) = hdr_props.as_mut() {
            match &config.hdr {
                Some(signaling) => {
                    if let Err(e) = props.set_hdr(&card.drm, signaling) {
                        tracing::warn!(
                            connector = %pick.connector_name,
                            "set HDR signaling failed: {e:#}"
                        );
                    } else {
                        tracing::info!(
                            connector = %pick.connector_name,
                            ?signaling,
                            "HDR signaling installed"
                        );
                    }
                }
                None => {
                    // Best-effort clear; failure logs only.
                    let _ = props.clear(&card.drm);
                }
            }
        }

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

        // Two scanout buffers (double-buffered). See module doc. Both
        // are allocated with the same modifier list; the candidate set
        // returned with `buffer A` is what we expose for the wayland
        // dmabuf-feedback path (per-output direct-scanout tranche).
        tracing::info!(connector = %pick.connector_name, "OutputContext::new step: alloc buffer A");
        let alloc_a = alloc_scanout_buffer(&device, card, config, extent, "buffer A")?;
        tracing::info!(connector = %pick.connector_name, "OutputContext::new step: alloc buffer B");
        let alloc_b = alloc_scanout_buffer(&device, card, config, extent, "buffer B")?;
        let scanout_fourcc = config.depth.drm_fourcc();
        let scanout_formats: Vec<_> = alloc_a
            .candidates
            .iter()
            .copied()
            .map(|m| (scanout_fourcc, m))
            .collect();
        let buffers = [alloc_a.buffer, alloc_b.buffer];

        tracing::info!(connector = %pick.connector_name, "OutputContext::new step: Renderer::new");
        let renderer = Renderer::new(
            device.clone(),
            config.vk_format,
            config.intermediate_format,
            &config.encode_config,
        )?;
        tracing::info!(connector = %pick.connector_name, "OutputContext::new step: done");

        // VRR / Adaptive Sync. Probe support per-connector first; if the
        // kernel says it's not supported on this connector, don't try to
        // turn it on — we'd hit an EINVAL on the next atomic commit. The
        // value the config wanted is logged so the user can see when their
        // request is being silently downgraded.
        let vrr_actual = if config.vrr {
            match surface.vrr_supported(pick.connector) {
                Ok(support) => {
                    use smithay::backend::drm::VrrSupport;
                    let supported = matches!(
                        support,
                        VrrSupport::Supported | VrrSupport::RequiresModeset
                    );
                    if !supported {
                        tracing::warn!(
                            connector = %pick.connector_name,
                            "VRR configured on but connector advertises NotSupported; \
                             leaving fixed-refresh"
                        );
                        false
                    } else {
                        match surface.use_vrr(true) {
                            Ok(()) => {
                                tracing::info!(
                                    connector = %pick.connector_name,
                                    "VRR enabled"
                                );
                                true
                            }
                            Err(e) => {
                                tracing::warn!(
                                    connector = %pick.connector_name,
                                    "use_vrr(true) rejected: {e:#}; leaving fixed-refresh"
                                );
                                false
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        connector = %pick.connector_name,
                        "vrr_supported query failed: {e:#}; assuming unsupported"
                    );
                    false
                }
            }
        } else {
            false
        };

        // FrameClock seed: refresh interval from the picked mode (vrefresh
        // is Hz). `vrr=true` lets the predictor stretch past the nominal
        // interval when no flip is pending — the kernel honors that on
        // VRR-enabled scanout.
        let vrefresh = pick.mode.vrefresh().max(1);
        let refresh_interval = Duration::from_nanos(1_000_000_000 / u64::from(vrefresh));
        let frame_clock = FrameClock::new(Some(refresh_interval), vrr_actual);

        // Try to claim a hardware cursor plane. Failure is non-fatal —
        // the output renders fine without one, just with no cursor.
        let cursor = match CursorPlane::try_new(card, &card.gbm, &surface) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("cursor plane init failed: {e:#}");
                None
            }
        };

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
            color_override: ColorOverride::default(),
            mode_set_done: false,
            frame_pending: false,
            frame_clock,
            cursor,
            edid,
            hdr_props,
            scanout_formats,
        })
    }

    /// Effective SDR reference luminance for this output, taking any
    /// runtime IPC override into account before falling back to the
    /// KDL-config value.
    pub fn effective_sdr_reference_nits(&self) -> f32 {
        self.color_override
            .sdr_reference_nits
            .unwrap_or(self.config.sdr_reference_nits)
    }

    /// Effective response curve, override-then-config.
    pub fn effective_response_curve(&self) -> Option<([f32; 3], [f32; 3])> {
        self.color_override
            .response_curve
            .or(self.config.response_curve)
    }

    /// Effective 3×3 calibration matrix, override-then-config. `None`
    /// = identity (no gamut correction in the encode shader).
    pub fn effective_ctm(&self) -> Option<[[f32; 3]; 3]> {
        self.color_override.ctm.or(self.config.ctm)
    }

    /// Rebuild this output's 3D LUT from the legacy `(CTM, response curve)`
    /// representation. Called at bringup and whenever an IPC handler
    /// mutates `color_override`'s CTM / response-curve / panel-peak
    /// fields. The shader chain reads only the LUT, so this is what
    /// makes calibration actually visible.
    ///
    /// No-op when the renderer's encode chain doesn't include
    /// `EncodeFragment::Lut3d` (e.g. legacy chains kept for tests) —
    /// detected via `lut3d_cube_edge() == 0`.
    ///
    /// **Per-mode curve handling.** SDR outputs skip the per-channel
    /// response curve when baking the LUT, mirroring the legacy
    /// `[CalibrationMatrix, OutputTransferSrgb]` shader chain that never
    /// included `PerChannelResponseGainGamma`. Applying it would push
    /// commanded values past `sdr_reference_nits` whenever the panel's
    /// peak emission is below sdr_white — the inverse-gamma blows up
    /// commanded to several × the cap, then `OutputTransferSrgb` clamps
    /// the entire range to byte 0xff and the panel renders pinned-peak
    /// garbage. The calibrate tool today still writes SDR response
    /// curves to KDL but those values are derived from a feedback loop
    /// that didn't actually apply them, so they're not trustworthy; we
    /// honor the previous shader's "ignore" semantics until the SDR
    /// calibration path is rebuilt. HDR outputs use the full
    /// `(CTM, curve)` pair as before — that path is what the LUT
    /// pipeline was designed around.
    pub fn resynthesize_color_lut(&mut self) -> Result<()> {
        let cube_edge = self.renderer.lut3d_cube_edge();
        if cube_edge == 0 {
            return Ok(());
        }
        let response_curve = if self.config.hdr.is_some() {
            self.effective_response_curve()
        } else {
            None
        };
        let entries = synthesize_lut_from_matrix_curve(
            cube_edge,
            self.effective_ctm(),
            response_curve,
        );
        self.renderer
            .upload_lut3d(&entries)
            .context("upload synthesized color LUT")
    }

    /// Effective per-channel panel-NATIVE peak nits (i.e. the peak
    /// emission of each physical primary on the panel itself).
    /// Resolution order:
    ///   1. Runtime IPC override (set by calibration tools)
    ///   2. KDL config `panel-peak-nits-r/-g/-b` (from prior calibration)
    ///   3. Broadcast of HDR `max_luminance` (HDR-mode outputs) or
    ///      effective `sdr_reference_nits` (SDR-mode outputs)
    ///
    /// **Do not feed this directly to the decoder clamp.** The decoder
    /// clamps in the BT.2020 domain; use [`Self::effective_decode_clamp_bt2020_rgb`]
    /// instead, which translates panel-native peaks through the CTM.
    /// This getter is for tooling (telemetry, calibrate's curve fit)
    /// and for the BT.2020 derivation itself.
    pub fn effective_panel_peak_nits_rgb(&self) -> [f32; 3] {
        if let Some(rgb) = self.color_override.panel_peak_nits_rgb {
            return rgb;
        }
        // Cached config value is already the per-channel value (KDL
        // override or bringup-time broadcast — main.rs::bringup
        // resolves both before we get here).
        self.config.panel_peak_nits_rgb
    }

    /// Per-channel BT.2020-domain clamp the decoder applies to the fp16
    /// intermediate. See [`derive_bt2020_decode_clamp`] for the math;
    /// this is a thin getter that plumbs in the effective panel peak,
    /// effective CTM, and standard margin.
    pub fn effective_decode_clamp_bt2020_rgb(&self) -> [f32; 3] {
        derive_bt2020_decode_clamp(
            self.effective_panel_peak_nits_rgb(),
            self.effective_ctm(),
            BT2020_DECODE_CAP_MARGIN,
        )
    }

    /// Re-push the bringup `HDR_OUTPUT_METADATA` infoframe to the
    /// connector. Used after runtime color overrides change, mostly
    /// as a no-op safety re-push — the actual signaling values come
    /// from the KDL `hdr { … }` block and do NOT track per-channel
    /// panel peaks.
    ///
    /// **History (2026-05-22 fix):** an earlier version projected
    /// the IPC-set `panel_peak_nits_rgb` into the infoframe's
    /// `max_display_mastering_luminance`, on the theory that a
    /// calibration pass discovering tighter per-channel peaks should
    /// tell the sink to tonemap against measured reality. That was
    /// based on the wrong meaning of the field: CTA-861's
    /// `max_display_mastering_luminance` is "what reference display
    /// was the **content** authored for", not "what can this panel
    /// physically emit". Empirically, advertising a low mastering
    /// peak (e.g. 112 nits, derived from per-channel emission) made
    /// the DP-4 Samsung HDR400 panel apply aggressive dim-content
    /// boost — verify-phase D65 sweeps emitted ~8× the requested
    /// luminance. The metadata blob now stays at whatever the user
    /// set in KDL `hdr { max-luminance … }` (typically the panel's
    /// HDR class peak: 400, 600, 1000), regardless of per-channel
    /// calibration measurements. The IR clamp at `probe_peak_y` is
    /// the only thing that downscales emission — keep that internal.
    ///
    /// No-op on non-HDR outputs (no `config.hdr` block) or connectors
    /// that don't expose HDR signaling (no `hdr_props`).
    pub fn rebuild_hdr_infoframe(&mut self) -> Result<()> {
        let Some(signaling) = self.config.hdr else {
            return Ok(()); // non-HDR output — nothing to push
        };
        if let Some(props) = self.hdr_props.as_mut() {
            props
                .set_hdr(&self.surface, &signaling)
                .context("re-push HDR_OUTPUT_METADATA from bringup config")?;
            tracing::debug!(
                connector = %self.connector_name,
                ?signaling,
                "HDR infoframe re-pushed (bringup config, no per-peak override)"
            );
        }
        Ok(())
    }
}

/// Runtime color overrides — see [`OutputContext::color_override`].
/// Each field is `None` until an IPC request sets it; setting any field
/// shadows the matching `OutputConfig` value in the render path.
#[derive(Debug, Default, Clone, Copy)]
pub struct ColorOverride {
    pub sdr_reference_nits: Option<f32>,
    pub response_curve: Option<([f32; 3], [f32; 3])>,
    /// Per-channel panel luminance ceiling override. Set by
    /// calibration tools after the per-channel saturation discovery
    /// phase produces measured per-subpixel maxima.
    pub panel_peak_nits_rgb: Option<[f32; 3]>,
    /// Per-output 3×3 gamut-correction matrix override. Set by
    /// calibration tools after measured primaries are known; the
    /// encode shader applies `panel_rgb = M * bt2020_rgb` to map IR
    /// values into the panel's native-primary space.
    pub ctm: Option<[[f32; 3]; 3]>,
}

impl OutputContext {

    /// Clear the `frame_pending` flag, advance the back-buffer index, and
    /// feed the actual kernel-reported presentation time into the
    /// `FrameClock` so the next render can predict the upcoming vblank.
    /// Call this when the DRM notifier surfaces a VBlank event for our
    /// CRTC.
    ///
    /// At this point the just-flipped buffer is being scanned out; the
    /// other buffer is no longer in use by the display and is safe to
    /// render into.
    pub fn mark_vblank(&mut self, presentation_time: Duration) {
        self.frame_pending = false;
        // The buffer we just flipped TO is now front. Toggle so back_index
        // points at the *other* one (the new back).
        self.back_index = 1 - self.back_index;
        self.frame_clock.presented(presentation_time);
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
        // render_frame returns the present-completion sync as a Linux
        // sync_file fd; we hand it to the DRM atomic commit as
        // IN_FENCE_FD so the kernel sequences the page-flip after our
        // GPU writes complete, without falling back to dmabuf
        // implicit-sync (which makes page_flip itself stall ~16ms on
        // radv). The fd is BorrowedFd-lifetime tied to `plane_state`
        // below; it's closed when `present_sync` drops at the end of
        // this function.
        let present_sync = self
            .renderer
            .render_frame(&back.image, elements, encode_push)?;

        let src = Rectangle::from_size(
            (self.extent.width as i32, self.extent.height as i32).into(),
        )
        .to_f64();
        let dst =
            Rectangle::from_size((self.extent.width as i32, self.extent.height as i32).into());
        // Build the plane state vector: primary first, then the
        // cursor plane if we own one. Cursor visibility/position are
        // owned by `prism_protocols::update_output_cursors`; we just
        // serialize whatever's there.
        let mut plane_state: Vec<PlaneState<'_>> = Vec::with_capacity(2);
        plane_state.push(PlaneState {
            handle: self.surface.plane(),
            config: Some(PlaneConfig {
                src,
                dst,
                transform: Transform::Normal,
                alpha: 1.0,
                damage_clips: None,
                fb: back.fb,
                fence: Some(present_sync.as_fd()),
            }),
        });
        if let Some(cursor) = self.cursor.as_ref() {
            plane_state.push(cursor.to_plane_state());
        }

        if !self.mode_set_done {
            flip_trace(&format!(
                "submit modeset {} crtc={:?} back={}",
                self.connector_name, self.crtc, self.back_index
            ));
            let res = self
                .surface
                .commit(plane_state.iter().cloned(), true)
                .context("DrmSurface::commit (initial mode-set)");
            flip_trace(&format!(
                "result modeset {} crtc={:?} -> {}",
                self.connector_name,
                self.crtc,
                match &res {
                    Ok(()) => "Ok".to_string(),
                    Err(e) => format!("Err({e})"),
                }
            ));
            res?;
            self.mode_set_done = true;
        } else {
            flip_trace(&format!(
                "submit page_flip {} crtc={:?} back={}",
                self.connector_name, self.crtc, self.back_index
            ));
            let res = self
                .surface
                .page_flip(plane_state.iter().cloned(), true)
                .context("DrmSurface::page_flip");
            flip_trace(&format!(
                "result page_flip {} crtc={:?} -> {}",
                self.connector_name,
                self.crtc,
                match &res {
                    Ok(()) => "Ok".to_string(),
                    Err(e) => format!("Err({e})"),
                }
            ));
            res?;
        }
        self.frame_pending = true;
        Ok(true)
    }
}

/// Result of `alloc_scanout_buffer` — the buffer itself plus the
/// per-format/modifier list that's compatible with this output's
/// scanout pipeline (chosen modifier first, LINEAR fallback last).
/// Returned so the caller can stash the candidate set on
/// `OutputContext.scanout_formats` for the wayland dmabuf-feedback
/// path; allocation is the natural place to compute it because we
/// run modifier negotiation here anyway.
struct AllocResult {
    buffer: ScanoutBuffer,
    candidates: Vec<DrmModifier>,
}

fn alloc_scanout_buffer(
    device: &Arc<Device>,
    card: &DrmCardContext,
    config: &OutputConfig,
    extent: vk::Extent2D,
    label: &str,
) -> Result<AllocResult> {
    // Modifier negotiation. Query what the Vulkan driver will accept as
    // a color attachment for this format, pass the resulting candidate
    // list through `pick_scanout_modifiers` (drops multi-plane / non-
    // renderable, orders tiled-first with LINEAR fallback), and let GBM
    // pick one that's also acceptable to the scanout pipe.
    //
    // The first allocate_scanout pass uses the full candidate list. If
    // it fails (driver/GBM couldn't find a mutually-supported tiled
    // modifier under the SCANOUT|RENDERING constraint), we retry with
    // LINEAR-only as the explicit fallback. LINEAR is universally
    // scanout-capable for the formats we use; the only cost is the
    // bandwidth we were trying to avoid.
    let render_mods = device.supported_drm_format_modifiers(config.vk_format);
    let candidates = crate::pick_scanout_modifiers(&render_mods);
    let fourcc = config.depth.drm_fourcc();
    let (bo, dmabuf) = match card.gbm.allocate_scanout(
        extent.width,
        extent.height,
        fourcc,
        &candidates,
    ) {
        Ok(pair) => pair,
        Err(first_err) => {
            tracing::warn!(
                buffer = label,
                ?candidates,
                "scanout alloc with tiled-modifier candidates failed ({first_err:#}); \
                 retrying LINEAR-only"
            );
            card.gbm
                .allocate_scanout(
                    extent.width,
                    extent.height,
                    fourcc,
                    &[DrmModifier::Linear],
                )
                .with_context(|| {
                    format!(
                        "GBM allocate {} {}×{} {:?} (LINEAR fallback after tiled failed)",
                        label, extent.width, extent.height, fourcc
                    )
                })?
        }
    };
    tracing::debug!(
        buffer = label,
        modifier = ?bo.modifier(),
        "scanout buffer allocated"
    );
    let image = ImportedImage::import(
        device.clone(),
        &dmabuf,
        config.vk_format,
        vk::ImageUsageFlags::COLOR_ATTACHMENT,
    )?;
    let fb = add_framebuffer_for_bo(&card.drm, &bo)?;
    Ok(AllocResult {
        buffer: ScanoutBuffer {
            image,
            _bo: bo,
            fb,
        },
        candidates,
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
        // Clear HDR signaling first so the next session sees a fresh
        // SDR connector. DrmSurface impls ControlDevice so we can
        // run the property writes through it without needing the
        // owning DrmDevice. Failure is logged and ignored — the
        // kernel will reset on fd close anyway, this just avoids
        // the cross-session "stuck on PQ" handoff bug.
        if let Some(hdr) = self.hdr_props.as_mut() {
            let _ = hdr.clear(&self.surface);
        }
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

/// Derive the per-channel BT.2020-domain clamp from panel-native peaks
/// and the calibration matrix.
///
/// Math: for each BT.2020 channel `c`, the panel-native commanded value
/// for panel-native channel `p` is `CTM[p][c] * bt2020[c]`. The largest
/// `bt2020[c]` we can pass through without driving any panel-native
/// channel past its peak is `min over p of (panel[p] / CTM[p][c])` —
/// restricted to entries where `CTM[p][c] > 0`, because negative entries
/// gamut-clip to zero in the encoder and don't constrain the cap.
///
/// `margin` is multiplied on top of the derived value to give the buffer
/// soft-clip headroom (see [`BT2020_DECODE_CAP_MARGIN`]).
///
/// `ctm = None` (un-calibrated identity) returns `panel` unchanged.
///
/// Defensive against pathological CTMs (all-negative column): falls back
/// to the matching panel-native value so we don't accidentally clamp the
/// buffer to zero.
pub fn derive_bt2020_decode_clamp(
    panel: [f32; 3],
    ctm: Option<[[f32; 3]; 3]>,
    margin: f32,
) -> [f32; 3] {
    let Some(ctm) = ctm else {
        return panel;
    };
    let mut cap = [0.0f32; 3];
    for c in 0..3 {
        let mut min_cap = f32::INFINITY;
        for p in 0..3 {
            if ctm[p][c] > 0.0 {
                let cap_p = panel[p] / ctm[p][c];
                if cap_p < min_cap {
                    min_cap = cap_p;
                }
            }
        }
        cap[c] = if min_cap.is_finite() {
            min_cap * margin
        } else {
            panel[c]
        };
    }
    cap
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity CTM (un-calibrated path) returns panel-native peaks
    /// unchanged. BT.2020 ≡ panel-native when no transform applies.
    #[test]
    fn identity_ctm_passes_panel_peaks_through() {
        let panel = [100.0, 200.0, 50.0];
        let cap = derive_bt2020_decode_clamp(panel, None, 1.5);
        assert_eq!(cap, panel);
    }

    /// Real DP-4 darker calibration data. Panel native peaks
    /// (37.842, 109.423, 15.256). CTM diagonal entries dominate, so the
    /// per-channel cap collapses to `panel[c] / ctm[c][c] * margin`.
    /// Before this fix, blue clamped to 15.256 nits and crushed any
    /// blue-rich content; with derivation the BT.2020-B cap is ~220 nits.
    #[test]
    fn dp4_darker_clamp_derivation() {
        let panel = [37.842, 109.423, 15.256];
        let ctm = Some([
            [0.307197, -0.086296, -0.001263],
            [-0.043766, 0.776906, -0.043953],
            [-0.000731, -0.012612, 0.104517],
        ]);
        let cap = derive_bt2020_decode_clamp(panel, ctm, 1.5);
        // Diagonal-only contributors (off-diagonals all negative for
        // this CTM, so they're clipped out of the cap derivation).
        let expect_r = 37.842 / 0.307197 * 1.5;
        let expect_g = 109.423 / 0.776906 * 1.5;
        let expect_b = 15.256 / 0.104517 * 1.5;
        assert!((cap[0] - expect_r).abs() < 1e-3, "R: {} vs {}", cap[0], expect_r);
        assert!((cap[1] - expect_g).abs() < 1e-3, "G: {} vs {}", cap[1], expect_g);
        assert!((cap[2] - expect_b).abs() < 1e-3, "B: {} vs {}", cap[2], expect_b);
        // Sanity: blue cap is ~10× the panel-native blue peak.
        assert!(cap[2] > 200.0, "blue cap should be far above panel-native peak, got {}", cap[2]);
    }

    /// When a CTM column has multiple positive entries, the tightest
    /// constraint wins (binding panel-native primary is whichever
    /// saturates first). Constructed example: column 0 has positive
    /// contributions to both panel R and panel G; R hits its peak first.
    #[test]
    fn min_constraint_wins_on_positive_columns() {
        let panel = [100.0, 50.0, 200.0];
        let ctm = Some([
            [2.0, 0.0, 0.0], // panel R = 2 * bt2020 R; cap_r_from_R = 100/2 = 50
            [1.0, 1.0, 0.0], // panel G = 1 * bt2020 R; cap_r_from_G = 50/1 = 50
            [0.0, 0.0, 1.0],
        ]);
        let cap = derive_bt2020_decode_clamp(panel, ctm, 1.0);
        // Tightest: both rows give cap_r = 50; either way result = 50.
        assert!((cap[0] - 50.0).abs() < 1e-4);
    }

    /// Margin is applied as a simple multiplier.
    #[test]
    fn margin_scales_output() {
        let panel = [100.0, 100.0, 100.0];
        let ctm = Some([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ]);
        let cap_no_margin = derive_bt2020_decode_clamp(panel, ctm, 1.0);
        let cap_margined = derive_bt2020_decode_clamp(panel, ctm, 2.0);
        for c in 0..3 {
            assert!((cap_no_margin[c] - 200.0).abs() < 1e-4);
            assert!((cap_margined[c] - 400.0).abs() < 1e-4);
        }
    }

    /// Pathological CTM (all-zero or all-negative column): fall back to
    /// the matching panel-native value rather than clamp the buffer to
    /// zero. Margin is intentionally NOT applied in the fallback path —
    /// the derivation couldn't constrain anything, so report the
    /// conservative panel-native value as-is.
    #[test]
    fn no_positive_column_falls_back_to_panel() {
        let panel = [42.0, 99.0, 7.0];
        let ctm = Some([
            [-1.0, 0.0, 0.0], // column 0: no positive entries
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let cap = derive_bt2020_decode_clamp(panel, ctm, 1.5);
        assert!((cap[0] - 42.0).abs() < 1e-4, "fallback R: {}", cap[0]);
    }
}
