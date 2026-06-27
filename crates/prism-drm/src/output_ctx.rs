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
    synthesize_lut_from_matrix_curve, vk, DamageTracker, Device, DrmDevId, EncodePush,
    ImportedImage, LoweredFrame, Renderer, SnapshotCopy,
};
use smithay::backend::drm::{DrmDevice, DrmSurface, PlaneConfig, PlaneState};
use smithay::reexports::drm::control::{connector, crtc, framebuffer, Mode};

use crate::frame_clock::FrameClock;
use smithay::utils::{Logical, Physical, Point, Rectangle, Scale, Size, Transform};

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
    add_framebuffer_for_bo,
    breadcrumb::{breadcrumb, flip_trace},
    set_connector_max_bpc, CursorPlane, DrmCardContext, OutputConfig, OutputPick,
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
    /// Per-output region damage tracker — diffs each frame's element metadata
    /// against the last to derive the changed region.
    damage_tracker: DamageTracker,
    /// Per-scanout-BO accumulated damage (buffer age). We render into alternating
    /// BOs, so each is ~2 presents stale; this tracks, for each BO, the region it
    /// is missing since it was last rendered. The encode pass scissors to it.
    /// Indexed like `buffers`; `Full` ⇒ the BO needs a full-output encode (never
    /// rendered, or content changed everywhere). See [`DamageCarry`].
    damage_carry: [DamageCarry; 2],
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
    /// LUT entries loaded from KDL `color.lut3d` at bringup, if any.
    /// Used by [`Self::resynthesize_color_lut`] as the fallback when no
    /// IPC color override is active — bypasses the synthesis path and
    /// pushes the file content directly. `None` ⇒ no file configured,
    /// synthesis takes over.
    pub kdl_lut3d_entries: Option<Vec<[f32; 3]>>,
    /// Panel black-point measurement (X, Y, Z in cd/m²) parsed out of
    /// the KDL-loaded LUT file's v2 header at bringup. `None` ⇒ no
    /// KDL LUT, or the file's `black_point_xyz` was all zeros (the
    /// "unknown" sentinel — calibrate-lut3d always writes a real
    /// measurement). Read via [`Self::effective_black_point_xyz`],
    /// which prefers the IPC override.
    pub kdl_black_point_xyz: Option<[f32; 3]>,
    /// Path to the measured gamut-surface sidecar from KDL `color.gamut
    /// "file"`, if configured. Not used by the render pipeline — held so
    /// the `GamutMesh` IPC can load + serve it on demand for the
    /// gamut-cloud inspector. `None` ⇒ no gamut file configured.
    pub kdl_gamut_path: Option<std::path::PathBuf>,
    /// Set on first present to switch from `commit` (mode-set) to `page_flip`
    /// (just-swap-fb) for subsequent frames.
    mode_set_done: bool,
    /// Forces the next present to render even with empty element damage, then
    /// self-clears. Set via [`Self::force_next_present`] when the encode output
    /// changed without any element changing — a new color LUT or encode params
    /// (calibration / HDR). Without it the zero-damage skip would drop the
    /// frame and the recolor wouldn't reach the screen until something moved.
    /// (The decode still scissors to the empty damage, so only the cheap
    /// full-screen encode re-runs.)
    force_present: bool,
    /// True between submitting a page-flip and receiving its vblank event.
    /// Submitting another flip while one is pending causes the kernel to
    /// reject with ENOMEM. Don't re-enter present() until `mark_vblank()`.
    frame_pending: bool,
    /// DPMS power state. `true` after [`Self::power_off`] (CRTC cleared,
    /// panel asleep) until [`Self::power_on`]. The render path skips
    /// powered-off outputs so a stray commit-/animation-driven redraw can't
    /// wake the panel; the next present after power-on re-modesets (because
    /// `power_off` also resets `mode_set_done`).
    powered_off: bool,
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
        // gracefully if hdr_props is None. The same routine runs on
        // session resume (see `reapply_color_signaling`), since
        // `DrmDevice::activate` resets these connector properties.
        apply_color_signaling(
            &card.drm,
            &pick.connector_name,
            pick.connector,
            &mut hdr_props,
            config,
        );

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
                    let supported =
                        matches!(support, VrrSupport::Supported | VrrSupport::RequiresModeset);
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
            damage_tracker: DamageTracker::new(),
            // Both BOs start undefined → full encode on first use.
            damage_carry: [DamageCarry::Full; 2],
            extent,
            mode: pick.mode,
            connector_name: pick.connector_name,
            connector: pick.connector,
            crtc: pick.crtc,
            gpu_id,
            config: config.clone(),
            color_override: ColorOverride::default(),
            kdl_lut3d_entries: None,
            kdl_black_point_xyz: None,
            kdl_gamut_path: None,
            mode_set_done: false,
            force_present: false,
            frame_pending: false,
            powered_off: false,
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

    /// Effective measured panel black-point in CIE XYZ (cd/m²) —
    /// what the colorimeter reads at (R=G=B=0). Precedence:
    /// IPC override > KDL-loaded LUT header > `None` (unmeasured).
    ///
    /// Consumers:
    /// - `wp_color_management_v1` feedback `min_luminance` event,
    ///   which standardizes on the Y component (the only one HDR
    ///   metadata transports treat as load-bearing).
    /// - Future tone mapping: PQ-encoded content authored at "0 nits"
    ///   should map to the panel's actual floor rather than absolute
    ///   zero, so the toe doesn't crush.
    /// - Cross-display matching (future): lifting the OLED's floor
    ///   to match the brightest LCD's would let same-grey patches
    ///   read consistently across the 6-monitor setup.
    ///
    /// Returns `None` when no calibration has measured it. Callers
    /// MUST handle that — there's no sensible default (EDID min-lum
    /// is essentially fictional on consumer panels).
    pub fn effective_black_point_xyz(&self) -> Option<[f32; 3]> {
        self.color_override
            .black_point_xyz
            .or(self.kdl_black_point_xyz)
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
        let Some((entries, source)) = self.effective_lut3d_entries() else {
            return Ok(());
        };
        let what = match source {
            LutSource::IpcOverride => "upload IPC-pushed color LUT",
            LutSource::KdlFile => "upload KDL-loaded color LUT",
            LutSource::Synthesized => "upload synthesized color LUT",
        };
        self.renderer.upload_lut3d(&entries).context(what)
    }

    /// The 3D LUT entries the encode pass should be running right now,
    /// plus which precedence level produced them. This is the single
    /// source of truth [`Self::resynthesize_color_lut`] uploads from; the
    /// IPC `Lut3d` inspector request reads it directly so clients see
    /// exactly what the GPU sees (which may never have existed on disk).
    ///
    /// Returns `None` when the renderer's encode chain has no LUT slot
    /// (`lut3d_cube_edge() == 0`). Always returns owned entries — the
    /// stored-override branches clone (~0.4 MB at cube_edge 33, dwarfed
    /// by the GPU upload that typically follows).
    ///
    /// Precedence (highest to lowest):
    ///   1. IPC LUT override (LoadLut3dFromFile) — measurement-derived
    ///      data pushed live by a calibration tool.
    ///   2. IPC override on CTM or response-curve → synthesize from
    ///      effective values (override-wins).
    ///   3. KDL-loaded LUT file (`color.lut3d "…"`).
    ///   4. KDL CTM + response-curve synthesis.
    pub fn effective_lut3d_entries(&self) -> Option<(Vec<[f32; 3]>, LutSource)> {
        let cube_edge = self.renderer.lut3d_cube_edge();
        if cube_edge == 0 {
            return None;
        }
        if let Some(entries) = self.color_override.lut3d_entries.as_ref() {
            if entries.len() == (cube_edge as usize).pow(3) {
                return Some((entries.clone(), LutSource::IpcOverride));
            }
            tracing::warn!(
                connector = %self.connector_name,
                "IPC LUT entry count {} doesn't match renderer cube_edge {}; \
                 falling back to next precedence level",
                entries.len(),
                cube_edge,
            );
        }
        let ipc_curve_override_active =
            self.color_override.ctm.is_some() || self.color_override.response_curve.is_some();
        if !ipc_curve_override_active {
            if let Some(entries) = self.kdl_lut3d_entries.as_ref() {
                if entries.len() == (cube_edge as usize).pow(3) {
                    return Some((entries.clone(), LutSource::KdlFile));
                }
                tracing::warn!(
                    connector = %self.connector_name,
                    "KDL-loaded LUT entry count {} doesn't match renderer cube_edge {}; \
                     falling back to synthesis",
                    entries.len(),
                    cube_edge,
                );
            }
        }
        let response_curve = if self.config.hdr.is_some() {
            self.effective_response_curve()
        } else {
            None
        };
        // Drive-domain chains (sRGB) bake the nits → drive normalization
        // into the synthesized fallback, anchored at the output's
        // effective reference white *at synthesis time*. This is the only
        // place `sdr-reference-nits` touches the encode side, and only
        // for uncalibrated outputs — a measured `.lut` (precedence 1/3
        // above) carries its own absolute drive mapping that no runtime
        // policy knob can re-scale.
        let drive_white = match self.config.encode_config.lut_output_domain() {
            prism_renderer::LutOutputDomain::Drive => Some(self.effective_sdr_reference_nits()),
            prism_renderer::LutOutputDomain::Nits => None,
        };
        let entries = synthesize_lut_from_matrix_curve(
            cube_edge,
            self.effective_ctm(),
            response_curve,
            drive_white,
        );
        Some((entries, LutSource::Synthesized))
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

    /// Resolve the peak luminance (cd/m²) to advertise to
    /// color-management clients as the display's mastering ceiling.
    /// Resolution order:
    ///   1. Runtime IPC override (`ColorOverride::advertised_peak_nits`)
    ///   2. KDL config `advertised-peak-nits`
    ///   3. KDL `hdr { max-luminance … }`
    ///
    /// Returns `None` for SDR outputs (no `hdr` block ⇒ no mastering
    /// metadata is advertised).
    ///
    /// Distinct from [`Self::effective_panel_peak_nits_rgb`] and the
    /// panel-facing `max-luminance`: this only changes what we *tell*
    /// clients the display reaches, not the HDR_OUTPUT_METADATA
    /// infoframe or the encode clamp.
    pub fn effective_advertised_peak_nits(&self) -> Option<u32> {
        let hdr = self.config.hdr?;
        Some(
            self.color_override
                .advertised_peak_nits
                .or(self.config.advertised_peak_nits)
                .unwrap_or(hdr.max_luminance as u32),
        )
    }

    /// Per-channel BT.2020-domain clamp the decoder applies to the fp16
    /// intermediate. The job here is only to prevent fp32 garbage from
    /// reaching the encode stage on adversarial content — gamut / range
    /// mapping happens inside the LUT (or the CTM-derived legacy clamp
    /// when no LUT is in play). See [`derive_bt2020_decode_clamp`].
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

/// Which precedence level of [`OutputContext::effective_lut3d_entries`]
/// produced the effective LUT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LutSource {
    /// IPC-pushed override (`LoadLut3dFromFile` / `IdentityLut3d`).
    IpcOverride,
    /// KDL-configured `color.lut3d` file.
    KdlFile,
    /// Synthesized from the effective CTM + response curve.
    Synthesized,
}

/// Runtime color overrides — see [`OutputContext::color_override`].
/// Each field is `None` until an IPC request sets it; setting any field
/// shadows the matching `OutputConfig` value in the render path.
#[derive(Debug, Default, Clone)]
pub struct ColorOverride {
    pub sdr_reference_nits: Option<f32>,
    pub response_curve: Option<([f32; 3], [f32; 3])>,
    /// Per-channel panel luminance ceiling override. Set by
    /// calibration tools after the per-channel saturation discovery
    /// phase produces measured per-subpixel maxima.
    pub panel_peak_nits_rgb: Option<[f32; 3]>,
    /// Override for the color-management advertised peak luminance
    /// (cd/m²) — the `mastering_luminance` max in the output's preferred
    /// image description. Set by calibration tooling to tune what
    /// color-managed clients are told the display reaches, without
    /// touching the HDR_OUTPUT_METADATA infoframe or the encode path.
    /// Shadows `OutputConfig::advertised_peak_nits`.
    pub advertised_peak_nits: Option<u32>,
    /// Per-output 3×3 gamut-correction matrix override. Set by
    /// calibration tools after measured primaries are known; the
    /// encode shader applies `panel_rgb = M * bt2020_rgb` to map IR
    /// values into the panel's native-primary space.
    pub ctm: Option<[[f32; 3]; 3]>,
    /// Runtime per-output 3D LUT override — entries in linear nits,
    /// X-fastest. When `Some`, takes precedence over both the
    /// (CTM, response-curve) overrides AND the KDL-loaded LUT. Used
    /// by `calibrate-lut3d` to push a freshly-measured LUT live
    /// without restarting prism. Set via `OutputAction::LoadLut3dFromFile`.
    pub lut3d_entries: Option<Vec<[f32; 3]>>,
    /// Panel black-point measurement that came in alongside `lut3d_entries`
    /// (or, separately, was set by some future IPC action). Same
    /// XYZ units as the LUT file header. Overrides `kdl_black_point_xyz`
    /// when present.
    pub black_point_xyz: Option<[f32; 3]>,
}

/// Outcome of [`OutputContext::present`].
pub enum PresentOutcome {
    /// Rendered and submitted a page-flip; carries the present-completion
    /// SYNC_FD (handed to KMS as IN_FENCE_FD, returned for release wiring).
    Presented(std::os::fd::OwnedFd),
    /// Rendered — the submit is on the GPU queue — but the atomic commit /
    /// page-flip failed. Carries the render-completion SYNC_FD (what
    /// `Presented` would have carried) so the caller can still account for
    /// the in-flight GPU work (mirror back-sync, acquire-wait marking)
    /// before propagating `error`. `frame_pending` stays false and the
    /// damage baseline is not advanced, so the next attempt re-renders.
    FlipFailed {
        render_done: std::os::fd::OwnedFd,
        error: anyhow::Error,
    },
    /// Nothing changed since the last presented frame (empty damage) and a
    /// valid image is already on screen, so no render or flip happened. The
    /// caller should arm an estimated-vblank instead of waiting for a real one.
    SkippedNoDamage,
    /// A previous flip is still in flight; the caller should retry after the
    /// next vblank (was `Ok(None)` before).
    FlipPending,
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

    /// True while the output is in DPMS-off (see [`Self::power_off`]). The
    /// render path skips powered-off outputs.
    pub fn is_powered_off(&self) -> bool {
        self.powered_off
    }

    /// DPMS-off this output: clear the CRTC (panel sleeps, all planes
    /// disabled) and mark it powered-off. Idempotent. Resets the mode-set
    /// flag so the next present after [`Self::power_on`] re-establishes the
    /// mode via `commit` rather than a (now-invalid) page-flip.
    ///
    /// `DrmSurface::clear` re-enables on the next `commit`/`page_flip`, so a
    /// powered-off surface is safe to leave until power-on. While off the
    /// CRTC emits no vblanks, so the vblank-driven redraw loop stops on its
    /// own; the explicit `powered_off` guard covers non-vblank redraw
    /// sources (commit handlers, animation ticks).
    pub fn power_off(&mut self) -> Result<()> {
        if self.powered_off {
            return Ok(());
        }
        self.surface
            .clear()
            .with_context(|| format!("DrmSurface::clear ({})", self.connector_name))?;
        self.powered_off = true;
        self.mode_set_done = false;
        // A flip may have been in flight; clearing the CRTC means its vblank
        // will never arrive, so drop the pending guard or present() would
        // refuse to ever render again after power-on.
        self.frame_pending = false;
        tracing::info!(connector = %self.connector_name, "DPMS off");
        Ok(())
    }

    /// DPMS-on this output. The actual mode-set happens on the next
    /// `present` (driven by a queued redraw), which commits because
    /// `power_off` reset `mode_set_done`. Idempotent.
    pub fn power_on(&mut self) {
        if !self.powered_off {
            return;
        }
        self.powered_off = false;
        tracing::info!(connector = %self.connector_name, "DPMS on (re-modeset on next present)");
    }

    /// Force the next [`Self::present`] to render even if no element changed.
    /// Call after mutating something the encode pass consumes but the damage
    /// tracker doesn't see — a new color LUT or encode parameters (calibration,
    /// HDR) — so the recolor reaches the screen instead of being dropped by the
    /// zero-damage skip. One-shot: cleared by the next render.
    pub fn force_next_present(&mut self) {
        self.force_present = true;
    }

    /// Prepare this output for a session resume (VT switch back). On pause the
    /// DRM device released master; on re-activate, `DrmDevice::activate(true)`
    /// ran `reset_state` over every surface, clearing the kernel-side CRTC and
    /// plane config. Mirror that in our own view: reset `mode_set_done` so the
    /// next present re-establishes the mode via `commit` rather than a
    /// now-invalid page-flip, and drop any in-flight-flip guard whose
    /// completion vblank was lost when master went away (same reasoning as
    /// [`Self::power_off`]). A powered-off output stays off — its queued
    /// redraw is dropped by the `powered_off` guard before present.
    pub fn mark_for_resume(&mut self) {
        self.mode_set_done = false;
        self.frame_pending = false;
    }

    /// Re-apply this connector's HDR / Colorspace / max-bpc signaling after a
    /// session resume. These are written as plain connector properties (not
    /// through the surface commit), so `DrmDevice::activate(true)`'s state
    /// reset drops them back to defaults — without this the panel returns from
    /// a VT switch in SDR instead of the HDR mode it had before. Shares the
    /// exact bringup routine. Pairs with [`Self::mark_for_resume`].
    pub fn reapply_color_signaling(&mut self, drm: &DrmDevice) {
        apply_color_signaling(
            drm,
            &self.connector_name,
            self.connector,
            &mut self.hdr_props,
            &self.config,
        );
    }

    /// Render the supplied frame (with the supplied encode parameters) into the
    /// *back* scanout image and submit it for display. Returns a
    /// [`PresentOutcome`]:
    ///   - `FlipPending`: a previous flip is still in flight; the caller should
    ///     wait for the next VBlank event before retrying.
    ///   - `SkippedNoDamage`: nothing changed since the last presented frame, so
    ///     no render or flip happened; the caller should arm an estimated vblank
    ///     (no real vblank will arrive).
    ///   - `Presented(fd)`: frame submitted. `fd` is the present-completion
    ///     `SYNC_FD` from `render_frame` — readable once the GPU finishes our
    ///     submit. It's already been handed to the DRM atomic commit as
    ///     `IN_FENCE_FD` (kernel dup'd internally), and is returned so the caller
    ///     can also time post-submit work — e.g. signaling `wp_linux_drm_syncobj`
    ///     release points on the input dmabufs. Caller may drop it if not needed.
    pub fn present(
        &mut self,
        frame: &LoweredFrame,
        // Output view size in logical pixels — converts the frame's logical
        // element geometry to physical for the damage diff.
        view_size: Size<f64, Logical>,
        encode_push: &EncodePush,
        // Cross-GPU mirror copy-done semaphores the render must wait on
        // before sampling. Empty for outputs with no mirrored surfaces.
        wait_semaphores: &[vk::Semaphore],
        // Window-close snapshots to capture from the intermediate this frame
        // (before the decode pass repaints over the region). Empty in the
        // common case.
        snapshots: &[SnapshotCopy],
        // Force a full-frame decode this frame (bypass the damage optimization).
        // Set while a closing window animates — see `render_frame` /
        // `Layout::ensure_close_snapshots`.
        force_full_repaint: bool,
    ) -> Result<PresentOutcome> {
        if self.frame_pending {
            return Ok(PresentOutcome::FlipPending);
        }

        // Region damage for this frame. Logged for now (Stage 1b); Stage 2
        // scissors the decode/encode to it. Computed here — after the
        // frame_pending early-out, so it only advances on an actual render —
        // but note its state advances unconditionally; once it gates real
        // work, the advance must move to after a successful flip.
        let scale = Scale::from((
            self.extent.width as f64 / view_size.w.max(1.0),
            self.extent.height as f64 / view_size.h.max(1.0),
        ));
        let damage = self.damage_tracker.compute(&frame.meta, scale);
        let damage_area: i64 = damage
            .iter()
            .map(|r| r.size.w as i64 * r.size.h as i64)
            .sum();
        tracing::debug!(
            target: "damage",
            output = %self.connector_name,
            rects = damage.len(),
            area_px = damage_area,
            full_px = self.extent.width as i64 * self.extent.height as i64,
            "frame damage",
        );

        // Zero-damage skip: nothing changed and we've already presented a valid
        // frame (mode_set_done). The front buffer still holds the correct image,
        // so rendering + flipping would just reproduce it — skip both. We do NOT
        // call `damage_tracker.commit()`: `compute` only staged `pending`, so the
        // baseline stays at the on-screen frame and the next real change diffs
        // against what's actually displayed. The caller arms an estimated vblank.
        // `force_present` overrides the skip when the encode output changed
        // without any element moving (recolor / recalibration).
        // A pending snapshot must render even with no element damage — the copy
        // out of the intermediate rides this frame's command buffer.
        if damage.is_empty() && self.mode_set_done && !self.force_present && snapshots.is_empty() {
            tracing::debug!(target: "damage", output = %self.connector_name, "skip: no damage");
            return Ok(PresentOutcome::SkippedNoDamage);
        }

        let back_index = self.back_index;
        // Buffer-age encode region for this BO: the region it's been missing
        // since it was last rendered (its carry) ∪ this frame's damage. The
        // encode pass scissors to the bounding box of this; passing it empty
        // requests a full-output encode. `content_full` marks frames whose
        // output changed *everywhere* (recolor via force_present, close-anim
        // repaint, or the first frame) — captured here, before force_present /
        // mode_set_done are mutated below, and used to roll the carry forward.
        let content_full = self.force_present || force_full_repaint || !self.mode_set_done;
        let frame_bbox = rects_bbox(&damage, self.extent);
        let encode_rects: Vec<Rectangle<i32, Physical>> = match self.damage_carry[back_index] {
            // BO never rendered / fully stale, or whole-output change ⇒ full.
            DamageCarry::Full => Vec::new(),
            DamageCarry::Region(_) if content_full => Vec::new(),
            // Re-encode the BO's missing region ∪ this frame's damage.
            DamageCarry::Region(carry) => {
                let mut v = Vec::with_capacity(carry.is_some() as usize + damage.len());
                v.extend(carry);
                v.extend_from_slice(&damage);
                v
            }
        };

        let back = &self.buffers[back_index];
        // render_frame returns the present-completion sync as a Linux
        // sync_file fd; we hand it to the DRM atomic commit as
        // IN_FENCE_FD so the kernel sequences the page-flip after our
        // GPU writes complete, without falling back to dmabuf
        // implicit-sync (which makes page_flip itself stall ~16ms on
        // radv). The fd is `BorrowedFd`-borrowed by `plane_state`
        // below for the duration of the atomic commit; after the
        // commit ioctl returns, the kernel holds its own dup and
        // the `OwnedFd` is free to be returned to the caller.
        let rendered = self.renderer.render_frame(
            &back.image,
            &frame.draws,
            &damage,
            &encode_rects,
            encode_push,
            wait_semaphores,
            snapshots,
            force_full_repaint,
        )?;
        let present_sync = rendered.present_sync;
        tracing::trace!(
            target: "damage",
            output = %self.connector_name,
            bo = back_index,
            encoded_full = rendered.encoded_full,
            "encode region",
        );

        // GPU profiling (PRISM_GPU_PROFILE): prism's own per-output compositing
        // cost, isolated from app load. Throttled to ≤1 Hz inside the renderer.
        if let Some(t) = self.renderer.take_gpu_profile_report() {
            tracing::info!(
                output = %self.connector_name,
                decode_us = t.decode_us,
                encode_us = t.encode_us,
                total_us = t.decode_us + t.encode_us,
                "gpu profile (1s ewma)"
            );
        }

        let src =
            Rectangle::from_size((self.extent.width as i32, self.extent.height as i32).into())
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
            if let Err(error) = res {
                return Ok(PresentOutcome::FlipFailed {
                    render_done: present_sync,
                    error,
                });
            }
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
            if let Err(error) = res {
                return Ok(PresentOutcome::FlipFailed {
                    render_done: present_sync,
                    error,
                });
            }
        }
        self.frame_pending = true;
        // The flip is submitted — this frame's damage has reached the scanout,
        // so advance the tracker's baseline. (On the `?` early-returns above the
        // commit is skipped, so a failed flip re-damages next frame.)
        self.damage_tracker.commit();
        // Roll the per-BO damage carry forward on the same successful-flip gate.
        // The BO we just rendered is now current; every *other* BO falls further
        // behind by this frame's content change (the whole output if the change
        // was global, else this frame's damage bbox added to what it already
        // missed). Keying the other BO off `content_full` — not off whether *this*
        // BO did a full encode — is what stops two BOs ping-ponging full encodes
        // when one merely needed a refresh for being stale.
        let other = 1 - back_index;
        self.damage_carry[other] = if content_full {
            DamageCarry::Full
        } else {
            self.damage_carry[other].extended(frame_bbox)
        };
        self.damage_carry[back_index] = DamageCarry::Region(None);
        // Consume the one-shot force flag only now that the flip actually
        // succeeded — clearing it earlier would drop a forced cursor/recolor
        // present on a transient page-flip error (the `?`s above), with nothing
        // to retry it until the next independent damage.
        self.force_present = false;
        Ok(PresentOutcome::Presented(present_sync))
    }
}

/// Apply a connector's color-signaling state from its [`OutputConfig`]:
/// install HDR_OUTPUT_METADATA + BT.2020 Colorspace when HDR is configured,
/// explicitly clear to SDR otherwise, and set max-bpc for the link depth.
///
/// These are direct connector-property writes, independent of the surface's
/// atomic commit — so they must be (re-)applied both at bringup and after a
/// session resume, where `DrmDevice::activate` has reset connector state.
/// All failures are best-effort (logged, not fatal): a connector with no HDR
/// props no-ops, and a missing max-bpc prop just leaves link depth to the
/// driver.
fn apply_color_signaling(
    drm: &DrmDevice,
    connector_name: &str,
    connector: connector::Handle,
    hdr_props: &mut Option<crate::HdrProps>,
    config: &OutputConfig,
) {
    if let Some(props) = hdr_props.as_mut() {
        match &config.hdr {
            Some(signaling) => {
                if let Err(e) = props.set_hdr(drm, signaling) {
                    tracing::warn!(connector = %connector_name, "set HDR signaling failed: {e:#}");
                } else {
                    tracing::info!(connector = %connector_name, ?signaling, "HDR signaling installed");
                }
            }
            // Best-effort clear; failure logs only.
            None => {
                let _ = props.clear(drm);
            }
        }
    }

    match set_connector_max_bpc(drm, connector, config.depth.max_bpc()) {
        Ok(true) => {
            tracing::info!(connector = %connector_name, "connector max bpc set to {}", config.depth.max_bpc())
        }
        Ok(false) => {
            tracing::warn!(connector = %connector_name, "connector doesn't expose 'max bpc'; link depth driver-controlled")
        }
        Err(e) => tracing::warn!(connector = %connector_name, "set max bpc failed: {e:#}"),
    }
}

/// Per-scanout-BO accumulated damage (buffer age). Because we render into
/// alternating BOs, each is ~2 presents stale when we next render it; this
/// records the region it's missing so the encode pass can repaint just that
/// instead of the whole output every frame.
#[derive(Clone, Copy)]
enum DamageCarry {
    /// The BO needs a full-output encode: never rendered, or the output content
    /// changed everywhere since it last saw a frame.
    Full,
    /// The BO is missing (at most) this bounding box of physical-pixel regions.
    /// `None` ⇒ nothing missing (the BO is current). Stored as a single
    /// bounding box because the encode pass collapses its scissor region to one
    /// anyway; per-rect scissoring is a later refinement.
    Region(Option<Rectangle<i32, Physical>>),
}

impl DamageCarry {
    /// Add `rect` to what this BO is missing. `Full` absorbs everything; a
    /// `Region` grows to the bounding box enclosing its current contents and
    /// `rect`.
    fn extended(self, rect: Option<Rectangle<i32, Physical>>) -> Self {
        match self {
            DamageCarry::Full => DamageCarry::Full,
            DamageCarry::Region(existing) => DamageCarry::Region(union_bbox(existing, rect)),
        }
    }
}

/// Bounding box of `rects`, clipped to `extent`. `None` if empty or entirely
/// offscreen. Mirrors the renderer's `damage_bbox`, in `smithay` `Rectangle`s.
fn rects_bbox(
    rects: &[Rectangle<i32, Physical>],
    extent: vk::Extent2D,
) -> Option<Rectangle<i32, Physical>> {
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    for r in rects {
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
    Some(Rectangle::new(
        Point::from((x0, y0)),
        Size::from((x1 - x0, y1 - y0)),
    ))
}

/// Bounding box enclosing two optional rectangles.
fn union_bbox(
    a: Option<Rectangle<i32, Physical>>,
    b: Option<Rectangle<i32, Physical>>,
) -> Option<Rectangle<i32, Physical>> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(a), Some(b)) => {
            let x0 = a.loc.x.min(b.loc.x);
            let y0 = a.loc.y.min(b.loc.y);
            let x1 = (a.loc.x + a.size.w).max(b.loc.x + b.size.w);
            let y1 = (a.loc.y + a.size.h).max(b.loc.y + b.size.h);
            Some(Rectangle::new(
                Point::from((x0, y0)),
                Size::from((x1 - x0, y1 - y0)),
            ))
        }
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
    let (bo, dmabuf) =
        match card
            .gbm
            .allocate_scanout(extent.width, extent.height, fourcc, &candidates)
        {
            Ok(pair) => pair,
            Err(first_err) => {
                tracing::warn!(
                    buffer = label,
                    ?candidates,
                    "scanout alloc with tiled-modifier candidates failed ({first_err:#}); \
                 retrying LINEAR-only"
                );
                card.gbm
                    .allocate_scanout(extent.width, extent.height, fourcc, &[DrmModifier::Linear])
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
        buffer: ScanoutBuffer { image, _bo: bo, fb },
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
        // Breadcrumb wrapping (vs just tracing) because a hang here may
        // need an external SIGKILL (or a hard kernel wedge), and tracing's
        // stdio buffer is lost on an ungraceful kill. The breadcrumbs are
        // fsync'd per line so we can attribute the hang to clear() vs to
        // the subsequent implicit field drops.
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
/// **CTM-active case** (`ctm = Some(_)`): for each BT.2020 channel
/// `c`, the panel-native commanded value for panel-native channel
/// `p` is `CTM[p][c] * bt2020[c]`. The largest `bt2020[c]` we can
/// pass through without driving any panel-native channel past its
/// peak is `min over p of (panel[p] / CTM[p][c])` — restricted to
/// entries where `CTM[p][c] > 0`, because negative entries gamut-
/// clip to zero in the encoder and don't constrain the cap.
///
/// `margin` is multiplied on top of the derived value to give the
/// buffer soft-clip headroom (see [`BT2020_DECODE_CAP_MARGIN`]).
/// Defensive against pathological CTMs (all-negative column): falls
/// back to the matching panel-native value so we don't accidentally
/// clamp the buffer to zero.
///
/// **3D-LUT case** (`ctm = None`): the LUT carries the gamut
/// transform end-to-end — including the BT.2020→panel-native
/// chromaticity remap that CTM used to do. In this mode the decode
/// clamp's only job is keeping the fp32 intermediate from
/// overflowing on adversarial content; gamut/range mapping happens
/// inside the LUT. Returns a uniform PQ-peak cap (10000 cd/m² per
/// channel) — no realistic content goes above that, and the LUT
/// handles whatever the panel can actually emit.
///
/// **Why this matters** (history): pre-3D-LUT, this function
/// returned `panel` directly for the `None` case under the
/// assumption "BT.2020 ≡ panel-native when no transform applies".
/// That assumption is false on every consumer panel — BT.2020 has
/// wider primaries than any LCD/OLED actually ships, so panel-
/// native B-peak (e.g. 15 cd/m² on a Samsung HDR400 LCD) is much
/// smaller than the BT.2020 B-value a typical blue pixel translates
/// to (~75 cd/m² for full sRGB blue through the sRGB→BT.2020
/// matrix). Per-channel clamping at panel-native peaks aggressively
/// truncated the blue channel, leaving G dominant — visible as
/// "blue UI accents render green" on calibrated HDR outputs.
pub fn derive_bt2020_decode_clamp(
    panel: [f32; 3],
    ctm: Option<[[f32; 3]; 3]>,
    margin: f32,
) -> [f32; 3] {
    let Some(ctm) = ctm else {
        // PQ peak: 10000 nits per channel. The LUT does the gamut /
        // range mapping; this just stops fp32 garbage from getting
        // through. `panel` is intentionally unused here.
        let _ = panel;
        return [10_000.0; 3];
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

    fn r(x: i32, y: i32, w: i32, h: i32) -> Rectangle<i32, Physical> {
        Rectangle::new(Point::from((x, y)), Size::from((w, h)))
    }
    fn ext(w: u32, h: u32) -> vk::Extent2D {
        vk::Extent2D {
            width: w,
            height: h,
        }
    }

    #[test]
    fn rects_bbox_empty_is_none() {
        assert!(rects_bbox(&[], ext(100, 100)).is_none());
    }

    #[test]
    fn rects_bbox_encloses_and_clips() {
        // Two disjoint rects → enclosing box; the right one overhangs the
        // 100×100 output and is clipped to the edge.
        let got = rects_bbox(&[r(10, 10, 5, 5), r(80, 80, 40, 40)], ext(100, 100));
        assert_eq!(got, Some(r(10, 10, 90, 90)));
    }

    #[test]
    fn rects_bbox_fully_offscreen_is_none() {
        assert!(rects_bbox(&[r(200, 200, 10, 10)], ext(100, 100)).is_none());
    }

    #[test]
    fn union_bbox_handles_none_and_encloses() {
        assert_eq!(union_bbox(None, Some(r(1, 2, 3, 4))), Some(r(1, 2, 3, 4)));
        assert_eq!(union_bbox(Some(r(1, 2, 3, 4)), None), Some(r(1, 2, 3, 4)));
        assert_eq!(union_bbox(None, None), None);
        // (0,0,10,10) ∪ (20,20,10,10) → (0,0,30,30).
        assert_eq!(
            union_bbox(Some(r(0, 0, 10, 10)), Some(r(20, 20, 10, 10))),
            Some(r(0, 0, 30, 30))
        );
    }

    #[test]
    fn damage_carry_extended() {
        // Full absorbs anything.
        assert!(matches!(
            DamageCarry::Full.extended(Some(r(0, 0, 5, 5))),
            DamageCarry::Full
        ));
        // Empty region picks up the first rect.
        assert!(matches!(
            DamageCarry::Region(None).extended(Some(r(1, 1, 2, 2))),
            DamageCarry::Region(Some(g)) if g == r(1, 1, 2, 2)
        ));
        // Existing region grows to the enclosing box.
        assert!(matches!(
            DamageCarry::Region(Some(r(0, 0, 4, 4))).extended(Some(r(10, 10, 4, 4))),
            DamageCarry::Region(Some(g)) if g == r(0, 0, 14, 14)
        ));
        // Extending by nothing is a no-op.
        assert!(matches!(
            DamageCarry::Region(Some(r(0, 0, 4, 4))).extended(None),
            DamageCarry::Region(Some(g)) if g == r(0, 0, 4, 4)
        ));
    }

    /// `ctm = None` is the 3D-LUT pipeline (the LUT carries the
    /// chromaticity transform). The clamp is loose — PQ peak (10000
    /// nits per channel) — because the LUT handles the actual gamut
    /// / range mapping and this clamp only exists to prevent fp32
    /// overflow on adversarial input.
    ///
    /// Regression guard for the "blue renders green" bug: before this
    /// fix, the None branch returned `panel` unchanged, which was
    /// fine for a fictional BT.2020-primary panel but wildly wrong
    /// for every real consumer panel (panel-native B-peak << BT.2020
    /// B-value of a typical sRGB blue pixel → blue got truncated to
    /// the panel's tiny B headroom → green dominated).
    #[test]
    fn no_ctm_returns_pq_peak() {
        let panel = [100.0, 200.0, 50.0];
        let cap = derive_bt2020_decode_clamp(panel, None, 1.5);
        // panel intentionally unused in the None branch; cap is the
        // uniform PQ peak regardless of per-channel headroom.
        assert_eq!(cap, [10_000.0; 3]);

        // Even pathologically tiny panel B (the DP-4 HDR LCD case
        // that triggered the original bug at 15.3 cd/m²) gets the
        // same uniform cap — the LUT, not this clamp, decides how
        // to map blue content into the panel's capabilities.
        let dp4 = [37.3, 112.1, 15.3];
        let cap_dp4 = derive_bt2020_decode_clamp(dp4, None, 1.5);
        assert_eq!(cap_dp4, [10_000.0; 3]);
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
        assert!(
            (cap[0] - expect_r).abs() < 1e-3,
            "R: {} vs {}",
            cap[0],
            expect_r
        );
        assert!(
            (cap[1] - expect_g).abs() < 1e-3,
            "G: {} vs {}",
            cap[1],
            expect_g
        );
        assert!(
            (cap[2] - expect_b).abs() < 1e-3,
            "B: {} vs {}",
            cap[2],
            expect_b
        );
        // Sanity: blue cap is ~10× the panel-native blue peak.
        assert!(
            cap[2] > 200.0,
            "blue cap should be far above panel-native peak, got {}",
            cap[2]
        );
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
        let ctm = Some([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]);
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
