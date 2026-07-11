//! Main-thread mirror of an output — identity, geometry, config, and the
//! color policy (`docs/async-render-rework.md` §2.2).
//!
//! The async-render rework moves each [`OutputContext`] (the execution
//! state: DrmSurface, scanout buffers, `Renderer`, damage tracking, cursor
//! plane) onto its GPU's render thread. Main-thread code — the layout walk
//! and frame lowering, IPC queries, protocol advertisements, the redraw
//! state machine — keeps reading per-output data constantly, so it keeps an
//! `OutputShadow` per output: everything main needs, with **no** reference
//! into the execution state.
//!
//! Ownership rules:
//! - **Identity / geometry / config** (`connector_name`, `crtc`, `extent`,
//!   `mode`, `config`, `edid`, …): immutable after bringup; plain copies.
//! - **Color policy** (`color_override`, `kdl_*`, the `effective_*`
//!   getters): the shadow is the *owner*, not a replica. IPC calibration
//!   mutates it here on main; derived data (LUT entries, encode push
//!   values) is computed here and shipped to the render thread as plain
//!   values. The `OutputContext` no longer holds any of it.
//! - **Renderer-derived caches** (`white_view`, `lut3d_cube_edge`,
//!   `profile_enabled`): stable for the life of a `Renderer`; captured at
//!   construction and refreshed via [`Self::refresh_renderer_caches`] when
//!   device-loss recovery rebuilds the renderer.
//! - **`powered_off`**: a replica of execution state, updated by the
//!   main-side power paths that drive it (DPMS actions, the session-lock
//!   blank path, recovery re-power). The context's flag remains the truth
//!   the render path acts on; the replica serves main-side gates.

use prism_renderer::{synthesize_lut_from_matrix_curve, vk, DrmDevId};
use smithay::reexports::drm::control::{connector, crtc, Mode};

use crate::output_ctx::derive_bt2020_decode_clamp;
use crate::{ColorOverride, EdidInfo, OutputConfig, OutputContext};

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

/// Which precedence level of [`OutputShadow::effective_lut3d_entries`]
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

impl LutSource {
    /// Context string for the GPU upload that typically follows.
    pub fn upload_context(self) -> &'static str {
        match self {
            LutSource::IpcOverride => "upload IPC-pushed color LUT",
            LutSource::KdlFile => "upload KDL-loaded color LUT",
            LutSource::Synthesized => "upload synthesized color LUT",
        }
    }
}

/// Main-thread per-output state — see the module docs.
///
/// Public field names deliberately mirror [`OutputContext`]'s so the many
/// main-side readers (`o.gpu_id`, `o.extent`, `o.config`, …) are untouched
/// by the map's type change.
pub struct OutputShadow {
    /// Connector name — also the `OutputId` key of the maps holding this.
    pub connector_name: String,
    /// Connector handle (for routing / config queries).
    pub connector: connector::Handle,
    /// CRTC bound to this output; routes vblank events (per-card handle —
    /// must be matched together with `gpu_id`).
    pub crtc: crtc::Handle,
    /// DrmDevId of the GPU rendering this output — keys `gpus`, `cards`,
    /// and (after the thread split) the owning render thread.
    pub gpu_id: DrmDevId,
    /// Width × height in pixels.
    pub extent: vk::Extent2D,
    /// Active DRM mode (size + vrefresh).
    pub mode: Mode,
    /// Parsed EDID (make / model / serial / physical size / HDR caps).
    pub edid: EdidInfo,
    /// Scanout-compatible (fourcc, modifier) list for dmabuf-feedback.
    pub scanout_formats: Vec<(drm_fourcc::DrmFourcc, drm_fourcc::DrmModifier)>,
    /// The static config resolved at bringup.
    pub config: OutputConfig,

    // ── Color policy (owned here; see module docs) ─────────────────────────
    /// Runtime color overrides set via IPC (calibration tooling, etc.).
    /// Sticky across config-file reloads — only `OutputAction::ResetColor`
    /// or a compositor restart clears them. When set, the render path
    /// uses these values in preference to the equivalents in `config`.
    pub color_override: ColorOverride,
    /// LUT entries loaded from KDL `color.lut3d` at bringup, if any.
    /// Fallback for [`Self::effective_lut3d_entries`] when no IPC override
    /// is active. `None` ⇒ no file configured, synthesis takes over.
    pub kdl_lut3d_entries: Option<Vec<[f32; 3]>>,
    /// Panel black-point measurement (X, Y, Z in cd/m²) from the KDL LUT
    /// file's v2 header. `None` ⇒ no KDL LUT or unmeasured (all-zero
    /// sentinel). Read via [`Self::effective_black_point_xyz`].
    pub kdl_black_point_xyz: Option<[f32; 3]>,
    /// Path to the measured gamut-surface sidecar from KDL `color.gamut`,
    /// served on demand by the `GamutMesh` IPC.
    pub kdl_gamut_path: Option<std::path::PathBuf>,

    // ── Renderer-derived caches ────────────────────────────────────────────
    /// The renderer's 1×1 white texel view — bound by solid-color/border
    /// draws during lowering. Stable per `Renderer`; refreshed on rebuild.
    pub white_view: vk::ImageView,
    /// The renderer's LUT cube edge (0 ⇒ encode chain has no LUT slot).
    pub lut3d_cube_edge: u32,
    /// Whether per-frame phase profiling is collected (static per renderer,
    /// from `PRISM_NO_PROFILE`).
    pub profile_enabled: bool,
    /// Shared handle to the renderer's 256-frame profile ring (§2.7): the
    /// render thread pushes a record per frame; the IPC profiling query
    /// reads percentiles/timeline from here without a bridge round-trip.
    /// Swapped on recovery rebuild (new renderer ⇒ new ring).
    pub profile_ring: std::sync::Arc<std::sync::Mutex<prism_renderer::ProfileRing>>,

    // ── Cursor-plane replica ───────────────────────────────────────────────
    /// Whether the output claimed a hardware cursor plane at bringup.
    /// Static for the output's lifetime (recovery rebuilds don't touch the
    /// KMS side).
    pub has_cursor_plane: bool,
    /// Replica of the plane's `visible` flag. The plane's setters are
    /// plain field writes with no clamping, so diffing desired state
    /// against this replica is exactly equivalent to diffing on the plane
    /// itself — which is what lets cursor updates be fire-and-forget
    /// messages instead of a blocking round-trip per pointer motion.
    pub cursor_visible: bool,
    /// Replica of the plane's position (physical CRTC pixels, hotspot
    /// already subtracted). Meaningless while `cursor_visible` is false.
    pub cursor_pos: (i32, i32),

    // ── Frame-clock estimate replica ───────────────────────────────────────
    /// Last-known prediction of the next presentation time, mirrored from
    /// the thread-owned `FrameClock` via every [`crate::SubmitReply`]
    /// (`docs/async-render-rework.md` §2.3 `next_present_estimate`).
    /// Consumers: the estimated-vblank timer (armed right after a
    /// `SkippedNoDamage` reply, so the value is fresh) and the feedback
    /// stash's diagnostic `target_time` (stale-by-one between a vblank and
    /// the next reply — acceptable, it's comparison-only).
    pub next_present_estimate: std::time::Duration,
    /// Refresh interval from the same snapshot (static per mode in
    /// practice; `None` only if the clock was built without one).
    pub refresh_interval: Option<std::time::Duration>,

    /// DPMS replica — see module docs.
    powered_off: bool,
}

impl OutputShadow {
    /// Capture a shadow from a freshly built (or rebuilt) context. Color
    /// policy starts empty; bringup fills the `kdl_*` fields afterwards.
    pub fn new(ctx: &OutputContext) -> Self {
        Self {
            connector_name: ctx.connector_name.clone(),
            connector: ctx.connector,
            crtc: ctx.crtc,
            gpu_id: ctx.gpu_id,
            extent: ctx.extent,
            mode: ctx.mode,
            edid: ctx.edid.clone(),
            scanout_formats: ctx.scanout_formats.clone(),
            config: ctx.config.clone(),
            color_override: ColorOverride::default(),
            kdl_lut3d_entries: None,
            kdl_black_point_xyz: None,
            kdl_gamut_path: None,
            white_view: ctx.renderer.white_view(),
            lut3d_cube_edge: ctx.renderer.lut3d_cube_edge(),
            profile_enabled: ctx.renderer.profile_enabled(),
            profile_ring: ctx.renderer.profile_ring(),
            has_cursor_plane: ctx.cursor.is_some(),
            // Matches CursorPlane's construction state ((0,0), hidden).
            cursor_visible: false,
            cursor_pos: (0, 0),
            next_present_estimate: ctx.frame_clock.next_presentation_time(),
            refresh_interval: ctx.frame_clock.refresh_interval(),
            powered_off: ctx.is_powered_off(),
        }
    }

    /// Mirror a fresh frame-clock snapshot from a render-thread reply.
    pub fn note_clock(&mut self, snapshot: crate::FrameClockSnapshot) {
        self.next_present_estimate = snapshot.next_presentation_time;
        self.refresh_interval = snapshot.refresh_interval;
    }

    /// Re-set the renderer-derived caches after device-loss recovery
    /// swapped the context's `Renderer` (`rebuild_render_state`). Raw
    /// values because the rebuild runs on the render thread and only these
    /// come back over the bridge.
    pub fn set_renderer_caches(
        &mut self,
        white_view: vk::ImageView,
        lut3d_cube_edge: u32,
        profile_enabled: bool,
        profile_ring: std::sync::Arc<std::sync::Mutex<prism_renderer::ProfileRing>>,
    ) {
        self.white_view = white_view;
        self.lut3d_cube_edge = lut3d_cube_edge;
        self.profile_enabled = profile_enabled;
        self.profile_ring = profile_ring;
    }

    /// DPMS replica — true while the output is powered off. Kept in sync by
    /// the power paths via [`Self::note_powered`].
    pub fn is_powered_off(&self) -> bool {
        self.powered_off
    }

    /// Record a successful power transition (`on = true` ⇒ panel awake).
    pub fn note_powered(&mut self, on: bool) {
        self.powered_off = !on;
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

    /// The 3D LUT entries the encode pass should be running right now,
    /// plus which precedence level produced them. This is the single
    /// source of truth for the render thread's LUT texture (the caller
    /// ships the entries to `Renderer::upload_lut3d`); the IPC `Lut3d`
    /// inspector request reads it directly so clients see exactly what
    /// the GPU sees (which may never have existed on disk).
    ///
    /// Returns `None` when the renderer's encode chain has no LUT slot
    /// (`lut3d_cube_edge == 0`). Always returns owned entries — the
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
    pub fn effective_lut3d_entries(&self) -> Option<(Vec<[f32; 3]>, LutSource)> {
        let cube_edge = self.lut3d_cube_edge;
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
}
