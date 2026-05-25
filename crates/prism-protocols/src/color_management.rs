//! `wp_color_management_v1` server dispatch.
//!
//! Implements the staging protocol from `wayland-protocols` (the XML
//! lives in `protocols/staging/color-management/color-management-v1.xml`,
//! generated bindings re-exported by smithay under
//! `smithay::reexports::wayland_protocols::wp::color_management::v1::server`).
//!
//! **Scope:** parametric image descriptions only. We deliberately defer:
//! - ICC profiles (`wp_image_description_creator_icc_v1`) — needs file
//!   I/O + ICC parser; calibration use case doesn't need it.
//! - `create_windows_scrgb` — niche; can add when needed.
//!
//! Output color advertising (`wp_color_management_output_v1`) and surface
//! feedback (`wp_color_management_surface_feedback_v1`) are implemented:
//! clients can query an output's preferred description (Firefox uses the
//! output path to decide whether to drive HDR).
//!
//! **What this gives us today:** clients can declare the color encoding
//! of their surface contents (e.g. "PQ-encoded BT.2020 mastered to 400
//! nits"). The render path consumes it: a surface's committed
//! description is mapped to [`prism_renderer::SurfaceColorParams`] via
//! [`description_to_params`] and lowered into the decode shader's push
//! constants (transfer fn + primaries→BT.2020 matrix). Surfaces with no
//! description fall back to the sRGB default. The remaining gap for HDR
//! *video* clients (e.g. Firefox) is YUV dmabuf import (NV12/P010), not
//! the color decode — once a YUV sampler yields nonlinear RGB, this same
//! per-surface decode applies.
//!
//! **Identity policy:** descriptions get a monotonically-increasing 64-bit
//! ID assigned at creation. Identity is opaque to clients today
//! (only used by `preferred_changed2`).

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

use smithay::reexports::wayland_protocols::wp::color_management::v1::server::{
    wp_color_management_surface_v1::{self, WpColorManagementSurfaceV1},
    wp_color_manager_v1::{
        self, Feature, Primaries, RenderIntent, TransferFunction, WpColorManagerV1,
    },
    wp_image_description_creator_params_v1::{self, WpImageDescriptionCreatorParamsV1},
    wp_image_description_info_v1::WpImageDescriptionInfoV1,
    wp_image_description_v1::{self, WpImageDescriptionV1},
};
use smithay::reexports::wayland_server::{
    backend::ClientId, protocol::wl_surface::WlSurface, Client, DataInit, Dispatch, DisplayHandle,
    GlobalDispatch, New, Resource,
};
use smithay::output::Output;
use smithay::wayland::compositor::{self, SurfaceData};

use crate::state::PrismState;
use crate::surface_tex::SurfacePlacementSlot;

// ─── Public types ──────────────────────────────────────────────────────────

/// What the compositor will accept in a parametric description.
/// Anything outside these sets gets rejected with the relevant
/// protocol error at parse time (so the client sees a clear failure
/// instead of a confusing `unsupported` later).
const SUPPORTED_INTENTS: &[RenderIntent] = &[RenderIntent::Perceptual];

/// Features we advertise via `supported_feature`. Notable absences:
/// `IccV2V4` (ICC creator deferred), `SetTfPower` (no use case yet),
/// `ExtendedTargetVolume` (renderer can't represent it), `WindowsScrgb`.
const SUPPORTED_FEATURES: &[Feature] = &[
    Feature::Parametric,
    Feature::SetPrimaries,
    Feature::SetLuminances,
    Feature::SetMasteringDisplayPrimaries,
];

/// Named transfer functions we advertise via `supported_tf_named`.
/// Includes the deprecated `Srgb` for backwards compatibility with
/// clients that still send it (Firefox, Chromium have shipped it).
const SUPPORTED_TFS: &[TransferFunction] = &[
    TransferFunction::Srgb,
    TransferFunction::Gamma22,
    TransferFunction::Bt1886,
    TransferFunction::St2084Pq,
    TransferFunction::ExtLinear,
];

/// Named primaries we advertise via `supported_primaries_named`.
/// sRGB/BT.709 (SDR), BT.2020 (HDR), and Display-P3 (P3-D65 — wide-gamut
/// web/video, the increasingly common middle ground). Each MUST have an
/// explicit arm in `chromaticities_for_named` — the catch-all there
/// silently resolves to sRGB, so adding a primary here without its
/// chromaticities would render it as the wrong gamut.
const SUPPORTED_PRIMARIES: &[Primaries] =
    &[Primaries::Srgb, Primaries::Bt2020, Primaries::DisplayP3];

/// A complete, validated image description. Created by the params
/// creator's `create` request after all required fields are set.
/// Stored behind `Arc` so surfaces can hold cheap clones.
#[derive(Debug, Clone)]
pub struct ImageDescription {
    /// Monotonic identity assigned at creation. Opaque to clients.
    pub identity: u64,
    /// Transfer characteristic (PQ, sRGB, gamma22, BT.1886, ext-linear).
    pub tf: TransferFunction,
    /// Primary color volume — named set or explicit chromaticities.
    pub primaries: PrimaryVolume,
    /// Primary color volume luminance range + reference white.
    /// `None` ⇒ defaults implied by `tf` (sRGB: 0.2 / 80 / 80;
    /// PQ: 0.005 / 10000 / 203; etc.).
    pub luminances: Option<Luminances>,
    /// Mastering display chromaticities (target color volume). `None`
    /// ⇒ same as `primaries`.
    pub mastering_primaries: Option<PrimaryChromaticities>,
    /// Mastering luminance range. `None` ⇒ inferred from `tf`.
    pub mastering_luminance: Option<MasteringLuminance>,
    /// Max content / frame-average light level, in cd/m². Optional
    /// per spec — `None` means the client didn't supply them.
    pub max_cll: Option<u32>,
    pub max_fall: Option<u32>,
}

/// Primary color volume — named or explicit.
#[derive(Debug, Clone, Copy)]
pub enum PrimaryVolume {
    Named(Primaries),
    Explicit(PrimaryChromaticities),
}

/// Eight signed integers from `set_primaries` / `set_mastering_display_primaries`.
/// Each is the CIE xy coordinate × 1,000,000 (6 decimal precision).
#[derive(Debug, Clone, Copy)]
pub struct PrimaryChromaticities {
    pub r_x: i32,
    pub r_y: i32,
    pub g_x: i32,
    pub g_y: i32,
    pub b_x: i32,
    pub b_y: i32,
    pub w_x: i32,
    pub w_y: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct Luminances {
    /// Minimum luminance (cd/m²) × 10000.
    pub min_lum_ticks: u32,
    /// Maximum luminance (cd/m², unscaled).
    pub max_lum: u32,
    /// Reference white luminance (cd/m², unscaled).
    pub reference_lum: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct MasteringLuminance {
    pub min_lum_ticks: u32,
    pub max_lum: u32,
}

/// Per-surface attachment. The currently-committed image description
/// (after a `wl_surface.commit` that follows a
/// `wp_color_management_surface_v1.set_image_description`).
///
/// Stored as a `Mutex` inside `SurfaceData` via smithay's `data_map`
/// extension pattern. The render path (Step 4) reads this to pick
/// the per-surface decode shader; today nothing reads it yet.
#[derive(Default)]
pub struct SurfaceColorSlot(pub Mutex<SurfaceColorState>);

#[derive(Default, Clone)]
pub struct SurfaceColorState {
    /// `Some` ⇔ client set a description; `None` ⇔ never set, or
    /// `unset_image_description` was called. Render path treats
    /// `None` as "assume sRGB" — see module doc.
    pub description: Option<Arc<ImageDescription>>,
    /// The rendering intent the client requested alongside the
    /// description. Today only `Perceptual` is advertised; stored
    /// for completeness when more intents land.
    pub intent: Option<RenderIntent>,
    /// Pending (not yet committed) description from a
    /// `set_image_description` request. Applied on `wl_surface.commit`.
    pub pending: Option<(Arc<ImageDescription>, RenderIntent)>,
    /// Same shape for `unset_image_description` (also double-buffered
    /// per spec).
    pub pending_unset: bool,
}

/// Per-surface tracker for active `wp_color_management_surface_feedback_v1`
/// resources. Stored on `SurfaceData` like `SurfaceDmabufFeedbackState` is.
/// Mirrors the smithay pattern: hold weak refs to the protocol
/// objects so dropped clients don't keep us alive, and push
/// `preferred_changed2` to the live ones when the surface's output
/// changes.
#[derive(Default)]
pub struct SurfaceColorFeedbackSlot(pub Mutex<SurfaceColorFeedbackInner>);

#[derive(Default)]
pub struct SurfaceColorFeedbackInner {
    pub instances: Vec<smithay::reexports::wayland_server::Weak<
        smithay::reexports::wayland_protocols::wp::color_management::v1::server::wp_color_management_surface_feedback_v1::WpColorManagementSurfaceFeedbackV1,
    >>,
    /// Identity of the last description we sent via
    /// `preferred_changed2`. Skip the send if it matches the current
    /// preferred — `preferred_changed2` is the *change* notification,
    /// the get_preferred* requests are how clients fetch initial
    /// state.
    pub last_sent_identity: Option<u64>,
}

impl SurfaceColorFeedbackSlot {
    /// Push `preferred_changed2` to every live feedback instance on
    /// this surface, if the identity differs from what we last sent.
    /// No-op if no feedback was ever requested for the surface.
    pub fn notify_preferred_changed(states: &SurfaceData, identity: u64) {
        let Some(slot) = states.data_map.get::<SurfaceColorFeedbackSlot>() else {
            return;
        };
        let mut st = slot.0.lock().unwrap();
        if st.last_sent_identity == Some(identity) {
            return;
        }
        st.instances.retain(|w| w.upgrade().is_ok());
        let hi = (identity >> 32) as u32;
        let lo = (identity & 0xffff_ffff) as u32;
        for w in &st.instances {
            if let Ok(inst) = w.upgrade() {
                inst.preferred_changed2(hi, lo);
            }
        }
        st.last_sent_identity = Some(identity);
    }
}

/// Map a committed image description to the renderer's
/// `SurfaceColorParams`. This is the bridge between the protocol-side
/// description (semantic: "PQ encoded BT.2020 mastered to 400 nits")
/// and the shader-side decode parameters (mechanical: "shader code 2,
/// 80-nit reference white").
///
/// `None` description ⇒ `None` (caller falls back to
/// `SurfaceColorParams::default()`). The mapping is deliberately
/// total over every TF we advertise in `SUPPORTED_TFS`; unsupported
/// TFs can never reach a committed description (the params creator
/// rejects them with `invalid_tf`).
pub fn description_to_params(desc: &ImageDescription) -> prism_renderer::SurfaceColorParams {
    let transfer = match desc.tf {
        // Linear path — pixels already in linear-light. Caller
        // anchors via sdr_white_nits.
        TransferFunction::ExtLinear => 0,
        // sRGB piecewise EOTF — matches the deprecated `srgb`
        // protocol value that some toolkits still ship.
        TransferFunction::Srgb => 1,
        // PQ absolute-nits domain.
        TransferFunction::St2084Pq => 2,
        // Gamma 2.2 — modern SDR default in protocol v2.
        TransferFunction::Gamma22 => 4,
        // BT.1886 — fragment shader implements the default-Lw/Lb
        // pure-pow 2.4 degenerate.
        TransferFunction::Bt1886 => 5,
        // Anything else (HLG, st240, log_*, etc.) isn't in
        // SUPPORTED_TFS so shouldn't reach here; if it does
        // (future expansion bug) fall back to sRGB rather than
        // silently rendering wrong.
        _ => 1,
    };
    let sdr_white_nits = desc
        .luminances
        .map(|l| l.reference_lum as f32)
        // sRGB default reference white per the protocol spec.
        .unwrap_or(80.0);
    // Convert the surface's primaries into the BT.2020 working space. Named
    // sets resolve to their standard chromaticities; explicit sets are used
    // verbatim. `primaries_to_bt2020` Bradford-adapts any non-D65 white.
    let chroma = match desc.primaries {
        PrimaryVolume::Named(p) => frame_chromaticities(chromaticities_for_named(p)),
        PrimaryVolume::Explicit(c) => frame_chromaticities(c),
    };
    // YUV→RGB coefficients for YUV-sampled surfaces follow the primaries:
    // BT.2020 → the BT.2020 NCL matrix; everything else (sRGB/BT.709, the
    // SDR-video default) → BT.709. Ignored unless the surface is YUV.
    let yuv_matrix = match desc.primaries {
        PrimaryVolume::Named(Primaries::Bt2020) => 1,
        _ => 0,
    };
    prism_renderer::SurfaceColorParams {
        transfer,
        sdr_white_nits,
        primaries_to_bt2020: prism_frame::primaries_to_bt2020(&chroma),
        yuv_matrix,
    }
}

/// Convert protocol chromaticities (CIE xy × 1,000,000) into the renderer's
/// floating-point [`prism_frame::Chromaticities`].
fn frame_chromaticities(c: PrimaryChromaticities) -> prism_frame::Chromaticities {
    let f = |v: i32| v as f32 / 1_000_000.0;
    prism_frame::Chromaticities {
        red: (f(c.r_x), f(c.r_y)),
        green: (f(c.g_x), f(c.g_y)),
        blue: (f(c.b_x), f(c.b_y)),
        white: (f(c.w_x), f(c.w_y)),
    }
}

impl SurfaceColorSlot {
    /// Fetch the committed image description for a surface, if any.
    /// Render path entry point.
    pub fn current(states: &SurfaceData) -> Option<Arc<ImageDescription>> {
        states
            .data_map
            .get::<SurfaceColorSlot>()
            .and_then(|slot| slot.0.lock().unwrap().description.clone())
    }

    /// Apply any pending description change. Called from the
    /// compositor's commit handler for surfaces that have one.
    pub fn commit_pending(states: &SurfaceData) {
        let Some(slot) = states.data_map.get::<SurfaceColorSlot>() else {
            return;
        };
        let mut st = slot.0.lock().unwrap();
        if st.pending_unset {
            st.description = None;
            st.intent = None;
            st.pending_unset = false;
        }
        if let Some((desc, intent)) = st.pending.take() {
            st.description = Some(desc);
            st.intent = Some(intent);
        }
    }
}

// ─── Server-side state ─────────────────────────────────────────────────────

/// Holds the description-identity counter, the global handle, and
/// the per-output preferred-description cache used by
/// `wp_color_management_surface_feedback_v1`.
/// Single instance lives on `PrismState`.
pub struct ColorManagementState {
    next_identity: AtomicU64,
    /// Per-output preferred image description. Built once at output
    /// bringup via [`Self::set_output_preferred`] from the HDR config +
    /// EDID. Cleared when an output drops. Identity is stable for
    /// the lifetime of the cached `Arc` (re-derivation only happens
    /// when HDR config changes, which today is bringup-static).
    output_preferred:
        std::sync::Mutex<std::collections::HashMap<crate::state::OutputId, Arc<ImageDescription>>>,
    /// `wp_image_description_info_v1` resources whose info events have
    /// been emitted but whose terminating `done()` is pending.
    ///
    /// `done` is a **destructor event** — calling it synchronously
    /// frees the resource's user data under `wayland-backend`, and
    /// since the resource was just created in this same dispatch turn
    /// the dispatcher's post-call code (`mod.rs:1651`) then writes
    /// into freed memory → use-after-free → SIGSEGV. We avoid the
    /// race by queuing the resource here from the request handler and
    /// draining (calling `done()` on each) after the dispatch returns,
    /// from the main loop (see [`Self::drain_pending_info_done`]).
    pending_info_done: std::sync::Mutex<Vec<WpImageDescriptionInfoV1>>,
}

impl ColorManagementState {
    pub fn new(dh: &DisplayHandle) -> Self {
        // Two versions advertised. v2 added preferred_changed2,
        // get_image_description, and compound_power_2_4 / absolute_no_adaptation.
        let _global = dh.create_global::<PrismState, WpColorManagerV1, ()>(2, ());
        Self {
            next_identity: AtomicU64::new(1),
            output_preferred: std::sync::Mutex::new(Default::default()),
            pending_info_done: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn next_identity(&self) -> u64 {
        self.next_identity.fetch_add(1, Ordering::Relaxed)
    }

    /// Register the preferred image description for an output. Called
    /// from `attach_output` after building from the output's
    /// HDR/EDID state. Returns the registered Arc so the caller can
    /// also push it to any feedback resources already bound on
    /// surfaces newly-mapped to this output.
    pub fn set_output_preferred(&self, id: crate::state::OutputId, desc: Arc<ImageDescription>) {
        self.output_preferred.lock().unwrap().insert(id, desc);
    }

    /// Look up an output's preferred description.
    pub fn output_preferred(&self, id: &crate::state::OutputId) -> Option<Arc<ImageDescription>> {
        self.output_preferred.lock().unwrap().get(id).cloned()
    }

    /// Queue a `wp_image_description_info_v1` resource for its
    /// terminating `done()` event to be sent from the main loop.
    /// See the field doc on [`Self::pending_info_done`] for why this
    /// can't be done inline from the request handler.
    pub fn queue_info_done(&self, info: WpImageDescriptionInfoV1) {
        self.pending_info_done.lock().unwrap().push(info);
    }

    /// Send `done()` on every queued info resource and clear the
    /// queue. Call once per main-loop turn, after wayland dispatch
    /// returns and before the matching `flush_clients`.
    pub fn drain_pending_info_done(&self) {
        // Take the vec while holding the lock briefly, then send
        // events outside the lock — `done()` calls into libwayland
        // which may itself dispatch (and re-enter our code), and we
        // don't want our own lock held across that.
        let drained = std::mem::take(&mut *self.pending_info_done.lock().unwrap());
        for info in drained {
            info.done();
        }
    }
}

/// Build the preferred parametric image description for an output
/// from its HDR / EDID state. HDR-configured outputs get PQ + BT.2020
/// with the mastering values we'd push to the panel via
/// HDR_OUTPUT_METADATA; SDR outputs get sRGB primaries + gamma22
/// with default luminances.
///
/// This is what `wp_color_management_surface_feedback_v1` advertises
/// as `preferred_changed2` — clients that match it can hit the
/// pass-through render path (Step 4) instead of going through
/// transfer-function conversion.
pub fn build_output_preferred(
    ctx: &prism_drm::OutputContext,
    cm: &ColorManagementState,
) -> Arc<ImageDescription> {
    let identity = cm.next_identity();
    if let Some(hdr) = ctx.config.hdr.as_ref() {
        // PQ HDR. Mastering primaries default to BT.2020 (same as
        // primary color volume) when the panel doesn't advertise
        // narrower ones via EDID — matches what build_hdr_metadata_blob
        // ships in the HDR_OUTPUT_METADATA infoframe.
        let max_lum = hdr.max_luminance as u32;
        // Prefer the measured panel floor (calibrate-lut3d writes it
        // to the .lut header, OutputContext exposes it via
        // effective_black_point_xyz) over the KDL-configured min
        // luminance. The measurement is what the colorimeter
        // actually read at (R=G=B=0); the KDL value is a guess or
        // a copy of an EDID claim. Convert cd/m² → 1/10000-cd/m²
        // ticks for the protocol.
        let min_lum_ticks = match ctx.effective_black_point_xyz() {
            Some(black) => (black[1] * 10_000.0).round().clamp(0.0, u32::MAX as f32) as u32,
            None => hdr.min_luminance_ticks as u32,
        };
        Arc::new(ImageDescription {
            identity,
            tf: TransferFunction::St2084Pq,
            primaries: PrimaryVolume::Named(Primaries::Bt2020),
            luminances: Some(Luminances {
                min_lum_ticks,
                // spec: max_lum is ignored for st2084_pq (always
                // implies min + 10000); we set it anyway for clarity.
                max_lum: 10_000,
                reference_lum: 203,
            }),
            mastering_primaries: None,
            mastering_luminance: Some(MasteringLuminance {
                min_lum_ticks,
                max_lum,
            }),
            max_cll: Some(hdr.max_cll as u32),
            max_fall: Some(hdr.max_fall as u32),
        })
    } else {
        // SDR. gamma22 is the spec-recommended modern default
        // (Srgb is deprecated since v2 for being ambiguous).
        Arc::new(ImageDescription {
            identity,
            tf: TransferFunction::Gamma22,
            primaries: PrimaryVolume::Named(Primaries::Srgb),
            luminances: None,
            mastering_primaries: None,
            mastering_luminance: None,
            max_cll: None,
            max_fall: None,
        })
    }
}

// ─── User data attached to resources ───────────────────────────────────────

/// User data for `WpImageDescriptionCreatorParamsV1`. Mutex-guarded
/// accumulator + flag tracking which fields have been set. The
/// `create` request reads + validates, then moves the values into a
/// freshly-built `ImageDescription`.
#[derive(Default)]
pub struct ParamsCreatorData {
    inner: Mutex<ParamsCreatorInner>,
}

#[derive(Default)]
struct ParamsCreatorInner {
    /// True once `create` was called — further requests on this
    /// object are protocol-incorrect (object becomes inert / a
    /// destructor in the protocol).
    consumed: bool,
    tf: Option<TransferFunction>,
    primaries: Option<PrimaryVolume>,
    luminances: Option<Luminances>,
    mastering_primaries: Option<PrimaryChromaticities>,
    mastering_luminance: Option<MasteringLuminance>,
    max_cll: Option<u32>,
    max_fall: Option<u32>,
}

/// User data for `WpImageDescriptionV1`. `Some` for descriptions
/// successfully created from the params creator; `None` for ones
/// where validation failed (the resource exists only long enough to
/// emit `failed` then is destroyed by the client).
pub struct ImageDescriptionData {
    pub description: Option<Arc<ImageDescription>>,
}

/// User data for `WpColorManagementSurfaceV1`. Holds a `Weak` to the
/// underlying `WlSurface` so we can resolve to live state on each
/// set/unset request — and detect inertness if the surface was
/// destroyed.
pub struct ColorSurfaceData {
    pub surface: smithay::reexports::wayland_server::Weak<WlSurface>,
}

/// User data for `WpColorManagementSurfaceFeedbackV1`. Same shape as
/// `ColorSurfaceData` — a Weak to the underlying surface so the
/// get_preferred / get_preferred_parametric requests can resolve the
/// surface's current output.
pub struct ColorSurfaceFeedbackData {
    pub surface: smithay::reexports::wayland_server::Weak<WlSurface>,
}

/// User data for `WpColorManagementOutputV1`. Holds the `OutputId`
/// (connector name) resolved from the `wl_output` passed to
/// `get_output`, so `get_image_description` can look up that output's
/// preferred description. `None` ⇔ the wl_output was foreign or already
/// dead → the resource is inert and `get_image_description` sends `failed`.
pub struct ColorOutputData {
    pub output_id: Option<crate::state::OutputId>,
}

// ─── wp_color_manager_v1 ───────────────────────────────────────────────────

impl GlobalDispatch<WpColorManagerV1, ()> for PrismState {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<WpColorManagerV1>,
        _global_data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        let manager = data_init.init(resource, ());
        // Spec: immediately send one event per supported intent /
        // feature / transfer function / primaries, then `done`.
        for &intent in SUPPORTED_INTENTS {
            manager.supported_intent(intent);
        }
        for &feature in SUPPORTED_FEATURES {
            manager.supported_feature(feature);
        }
        for &tf in SUPPORTED_TFS {
            manager.supported_tf_named(tf);
        }
        for &p in SUPPORTED_PRIMARIES {
            manager.supported_primaries_named(p);
        }
        manager.done();
    }
}

impl Dispatch<WpColorManagerV1, ()> for PrismState {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &WpColorManagerV1,
        request: <WpColorManagerV1 as Resource>::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        use wp_color_manager_v1::Request;
        match request {
            Request::Destroy => {}
            Request::GetOutput { id, output } => {
                // Resolve the wl_output to our OutputId (connector name)
                // and stash it on the resource, so the resulting
                // wp_color_management_output_v1.get_image_description can
                // hand back that output's preferred description. A
                // foreign / dead wl_output yields `None` → inert (its
                // get_image_description sends `failed`).
                let output_id = Output::from_resource(&output).map(|o| o.name());
                let _ = data_init.init(id, ColorOutputData { output_id });
            }
            Request::GetSurface { id, surface } => {
                let data = ColorSurfaceData {
                    surface: surface.downgrade(),
                };
                let _ = data_init.init(id, data);
            }
            Request::GetSurfaceFeedback { id, surface } => {
                // Step 3: real handler. Resource carries a Weak to
                // the wl_surface so get_preferred* can look up which
                // output the surface is currently on. The actual
                // protocol object is also added to the surface's
                // SurfaceColorFeedbackSlot so future output
                // transitions can push preferred_changed2.
                let data = ColorSurfaceFeedbackData {
                    surface: surface.downgrade(),
                };
                let instance = data_init.init(id, data);
                compositor::with_states(&surface, |states| {
                    states
                        .data_map
                        .insert_if_missing_threadsafe(SurfaceColorFeedbackSlot::default);
                    let slot = states.data_map.get::<SurfaceColorFeedbackSlot>().unwrap();
                    let mut st = slot.0.lock().unwrap();
                    st.instances.push(instance.downgrade());
                    // Send the current preferred (if known) so
                    // clients that bind feedback after the surface
                    // already has an output see the right state
                    // without needing an explicit get_preferred*.
                    let current_output = states
                        .data_map
                        .get::<SurfacePlacementSlot>()
                        .and_then(|s| s.0.lock().unwrap().current_output.clone());
                    if let Some(out_id) = current_output {
                        if let Some(desc) = state.color_management.output_preferred(&out_id) {
                            let hi = (desc.identity >> 32) as u32;
                            let lo = (desc.identity & 0xffff_ffff) as u32;
                            instance.preferred_changed2(hi, lo);
                            st.last_sent_identity = Some(desc.identity);
                        }
                    }
                });
            }
            Request::CreateIccCreator { obj } => {
                // We don't advertise IccV2V4 so a spec-compliant
                // client never sends this. Raise the documented
                // protocol error and skip resource init.
                resource.post_error(
                    wp_color_manager_v1::Error::UnsupportedFeature,
                    "ICC profiles not supported".to_string(),
                );
                let _ = obj; // dropped without init — wayland-server closes the new_id
                let _ = state;
            }
            Request::CreateParametricCreator { obj } => {
                let _ = data_init.init(obj, ParamsCreatorData::default());
            }
            Request::CreateWindowsScrgb { image_description } => {
                resource.post_error(
                    wp_color_manager_v1::Error::UnsupportedFeature,
                    "windows_scrgb not supported".to_string(),
                );
                let _ = image_description;
            }
            Request::GetImageDescription {
                image_description,
                reference: _,
            } => {
                // v2 addition — for clients holding a reference (we
                // don't expose any reference-yielding requests today),
                // re-materialize the description. Until reference-
                // yielding paths land we just init an inert resource.
                let _ = data_init.init(
                    image_description,
                    ImageDescriptionData { description: None },
                );
            }
            _ => {}
        }
    }
}

use smithay::reexports::wayland_protocols::wp::color_management::v1::server::wp_color_management_output_v1::{
    self, WpColorManagementOutputV1,
};
impl Dispatch<WpColorManagementOutputV1, ColorOutputData> for PrismState {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &WpColorManagementOutputV1,
        request: <WpColorManagementOutputV1 as Resource>::Request,
        data: &ColorOutputData,
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        use wp_color_management_output_v1::Request;
        match request {
            Request::Destroy => {}
            Request::GetImageDescription { image_description } => {
                // Hand back the output's preferred description (same one
                // surface_feedback advertises). Mirrors the get_preferred
                // path: init with the description + `ready`, or `failed`
                // if we don't know this output.
                //
                // We don't emit `image_description_changed` because an
                // output's color description is static for prism's
                // lifetime today (HDR config is fixed at startup); wire
                // it up here if runtime HDR toggling lands.
                let preferred = data
                    .output_id
                    .as_ref()
                    .and_then(|id| state.color_management.output_preferred(id));
                let Some(desc) = preferred else {
                    tracing::debug!(
                        output = ?data.output_id,
                        "color-mgmt: output get_image_description → failed (no description)"
                    );
                    let d = data_init.init(
                        image_description,
                        ImageDescriptionData { description: None },
                    );
                    d.failed(
                        wp_image_description_v1::Cause::NoOutput,
                        "no color description for this output".to_string(),
                    );
                    return;
                };
                let identity = desc.identity;
                tracing::debug!(
                    output = ?data.output_id,
                    tf = ?desc.tf,
                    identity,
                    "color-mgmt: output get_image_description → ready"
                );
                let d = data_init.init(
                    image_description,
                    ImageDescriptionData {
                        description: Some(desc),
                    },
                );
                let lo = (identity & 0xffff_ffff) as u32;
                d.ready(lo);
            }
            _ => {}
        }
    }
}

use smithay::reexports::wayland_protocols::wp::color_management::v1::server::wp_color_management_surface_feedback_v1::{
    self, WpColorManagementSurfaceFeedbackV1,
};
impl Dispatch<WpColorManagementSurfaceFeedbackV1, ColorSurfaceFeedbackData> for PrismState {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &WpColorManagementSurfaceFeedbackV1,
        request: <WpColorManagementSurfaceFeedbackV1 as Resource>::Request,
        data: &ColorSurfaceFeedbackData,
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        use wp_color_management_surface_feedback_v1::Request;
        match request {
            Request::Destroy => {}
            Request::GetPreferred { image_description }
            | Request::GetPreferredParametric { image_description } => {
                // Both variants behave the same for us — every
                // description we mint is parametric. Resolve the
                // surface's current output, look up the cached
                // preferred description, init the resource with it,
                // send `ready` (low 32 bits of identity).
                let Ok(surface) = data.surface.upgrade() else {
                    let desc = data_init.init(
                        image_description,
                        ImageDescriptionData { description: None },
                    );
                    desc.failed(
                        wp_image_description_v1::Cause::NoOutput,
                        "wl_surface gone".to_string(),
                    );
                    let _ = resource;
                    return;
                };
                let preferred = compositor::with_states(&surface, |states| {
                    states
                        .data_map
                        .get::<SurfacePlacementSlot>()
                        .and_then(|s| s.0.lock().unwrap().current_output.clone())
                })
                .and_then(|out_id| state.color_management.output_preferred(&out_id));
                let Some(desc) = preferred else {
                    // Surface isn't on any known output (unmapped or
                    // pre-first-commit). Spec wants us to still
                    // produce a description — emit failed so the
                    // client knows.
                    let d = data_init.init(
                        image_description,
                        ImageDescriptionData { description: None },
                    );
                    d.failed(
                        wp_image_description_v1::Cause::NoOutput,
                        "surface not yet mapped to an output".to_string(),
                    );
                    return;
                };
                let identity = desc.identity;
                let d = data_init.init(
                    image_description,
                    ImageDescriptionData {
                        description: Some(desc),
                    },
                );
                let lo = (identity & 0xffff_ffff) as u32;
                d.ready(lo);
            }
            _ => {}
        }
    }

    fn destroyed(
        _state: &mut Self,
        _client: ClientId,
        resource: &WpColorManagementSurfaceFeedbackV1,
        data: &ColorSurfaceFeedbackData,
    ) {
        if let Ok(surface) = data.surface.upgrade() {
            compositor::with_states(&surface, |states| {
                if let Some(slot) = states.data_map.get::<SurfaceColorFeedbackSlot>() {
                    let mut st = slot.0.lock().unwrap();
                    st.instances
                        .retain(|w| w.upgrade().is_ok() && w.id() != resource.id());
                }
            });
        }
    }
}

// ─── wp_image_description_creator_params_v1 ────────────────────────────────

impl Dispatch<WpImageDescriptionCreatorParamsV1, ParamsCreatorData> for PrismState {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &WpImageDescriptionCreatorParamsV1,
        request: <WpImageDescriptionCreatorParamsV1 as Resource>::Request,
        data: &ParamsCreatorData,
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        use wp_image_description_creator_params_v1::Request;
        let mut inner = data.inner.lock().unwrap();

        // Once `create` was called the object is destroyed — these
        // post-create requests are protocol-incorrect but the
        // wayland-server backend already filters destructor-after-
        // destruction. The flag here guards against re-entry within
        // the same dispatch call.
        if inner.consumed {
            return;
        }

        match request {
            Request::Create { image_description } => {
                inner.consumed = true;
                let outcome = build_description(&mut inner, &state.color_management);
                let desc_resource = match &outcome {
                    Ok(desc) => data_init.init(
                        image_description,
                        ImageDescriptionData {
                            description: Some(desc.clone()),
                        },
                    ),
                    Err(_) => data_init.init(
                        image_description,
                        ImageDescriptionData { description: None },
                    ),
                };
                match outcome {
                    Ok(desc) => {
                        // `ready` (v1, deprecated since v2) carries
                        // the low 32 bits of the identity. Once we
                        // start emitting `ready2` (the v2 64-bit
                        // form) here, switch + drop the truncation.
                        let lo = (desc.identity & 0xffff_ffff) as u32;
                        desc_resource.ready(lo);
                    }
                    Err(reason) => {
                        desc_resource
                            .failed(wp_image_description_v1::Cause::Unsupported, reason.into());
                    }
                }
            }
            Request::SetTfNamed { tf } => {
                let tf = match into_tf(tf) {
                    Some(t) if SUPPORTED_TFS.contains(&t) => t,
                    _ => {
                        resource.post_error(
                            wp_image_description_creator_params_v1::Error::InvalidTf,
                            "unsupported transfer function".to_string(),
                        );
                        return;
                    }
                };
                if inner.tf.is_some() {
                    resource.post_error(
                        wp_image_description_creator_params_v1::Error::AlreadySet,
                        "tf already set".to_string(),
                    );
                    return;
                }
                inner.tf = Some(tf);
            }
            Request::SetTfPower { eexp: _ } => {
                resource.post_error(
                    wp_image_description_creator_params_v1::Error::UnsupportedFeature,
                    "set_tf_power not supported".to_string(),
                );
            }
            Request::SetPrimariesNamed { primaries } => {
                let p = match into_primaries(primaries) {
                    Some(p) if SUPPORTED_PRIMARIES.contains(&p) => p,
                    _ => {
                        resource.post_error(
                            wp_image_description_creator_params_v1::Error::InvalidPrimariesNamed,
                            "unsupported primaries".to_string(),
                        );
                        return;
                    }
                };
                if inner.primaries.is_some() {
                    resource.post_error(
                        wp_image_description_creator_params_v1::Error::AlreadySet,
                        "primaries already set".to_string(),
                    );
                    return;
                }
                inner.primaries = Some(PrimaryVolume::Named(p));
            }
            Request::SetPrimaries {
                r_x,
                r_y,
                g_x,
                g_y,
                b_x,
                b_y,
                w_x,
                w_y,
            } => {
                if inner.primaries.is_some() {
                    resource.post_error(
                        wp_image_description_creator_params_v1::Error::AlreadySet,
                        "primaries already set".to_string(),
                    );
                    return;
                }
                inner.primaries = Some(PrimaryVolume::Explicit(PrimaryChromaticities {
                    r_x,
                    r_y,
                    g_x,
                    g_y,
                    b_x,
                    b_y,
                    w_x,
                    w_y,
                }));
            }
            Request::SetLuminances {
                min_lum,
                max_lum,
                reference_lum,
            } => {
                if inner.luminances.is_some() {
                    resource.post_error(
                        wp_image_description_creator_params_v1::Error::AlreadySet,
                        "luminances already set".to_string(),
                    );
                    return;
                }
                // Per spec: max_lum and reference_lum must be > min_lum/10000.
                let min_cdm2 = min_lum as f64 / 10000.0;
                if (max_lum as f64) <= min_cdm2 || (reference_lum as f64) <= min_cdm2 {
                    resource.post_error(
                        wp_image_description_creator_params_v1::Error::InvalidLuminance,
                        "max_lum or reference_lum <= min_lum".to_string(),
                    );
                    return;
                }
                inner.luminances = Some(Luminances {
                    min_lum_ticks: min_lum,
                    max_lum,
                    reference_lum,
                });
            }
            Request::SetMasteringDisplayPrimaries {
                r_x,
                r_y,
                g_x,
                g_y,
                b_x,
                b_y,
                w_x,
                w_y,
            } => {
                if inner.mastering_primaries.is_some() {
                    resource.post_error(
                        wp_image_description_creator_params_v1::Error::AlreadySet,
                        "mastering primaries already set".to_string(),
                    );
                    return;
                }
                inner.mastering_primaries = Some(PrimaryChromaticities {
                    r_x,
                    r_y,
                    g_x,
                    g_y,
                    b_x,
                    b_y,
                    w_x,
                    w_y,
                });
            }
            Request::SetMasteringLuminance { min_lum, max_lum } => {
                if inner.mastering_luminance.is_some() {
                    resource.post_error(
                        wp_image_description_creator_params_v1::Error::AlreadySet,
                        "mastering luminance already set".to_string(),
                    );
                    return;
                }
                let min_cdm2 = min_lum as f64 / 10000.0;
                if (max_lum as f64) <= min_cdm2 {
                    resource.post_error(
                        wp_image_description_creator_params_v1::Error::InvalidLuminance,
                        "mastering max_lum <= min_lum".to_string(),
                    );
                    return;
                }
                inner.mastering_luminance = Some(MasteringLuminance {
                    min_lum_ticks: min_lum,
                    max_lum,
                });
            }
            Request::SetMaxCll { max_cll } => {
                if inner.max_cll.is_some() {
                    resource.post_error(
                        wp_image_description_creator_params_v1::Error::AlreadySet,
                        "max_cll already set".to_string(),
                    );
                    return;
                }
                inner.max_cll = Some(max_cll);
            }
            Request::SetMaxFall { max_fall } => {
                if inner.max_fall.is_some() {
                    resource.post_error(
                        wp_image_description_creator_params_v1::Error::AlreadySet,
                        "max_fall already set".to_string(),
                    );
                    return;
                }
                inner.max_fall = Some(max_fall);
            }
            _ => {}
        }
    }
}

/// Pull validated values out of the accumulator and build an
/// `ImageDescription`. Returns `Err(reason)` if required fields are
/// missing or post-create cross-field validation fails — in that
/// case the resulting description object delivers `failed`.
fn build_description(
    inner: &mut ParamsCreatorInner,
    cm: &ColorManagementState,
) -> Result<Arc<ImageDescription>, &'static str> {
    let tf = inner.tf.ok_or("incomplete: tf not set")?;
    let primaries = inner.primaries.ok_or("incomplete: primaries not set")?;
    // Cross-field: max_fall must be ≤ max_cll if both set.
    if let (Some(cll), Some(fall)) = (inner.max_cll, inner.max_fall) {
        if fall > cll {
            return Err("max_fall > max_cll");
        }
    }
    Ok(Arc::new(ImageDescription {
        identity: cm.next_identity(),
        tf,
        primaries,
        luminances: inner.luminances,
        mastering_primaries: inner.mastering_primaries,
        mastering_luminance: inner.mastering_luminance,
        max_cll: inner.max_cll,
        max_fall: inner.max_fall,
    }))
}

/// Emit the `wp_image_description_info_v1` event sequence for a
/// description, per the protocol's `get_information` contract. The
/// terminating `done()` is NOT sent here — the caller is responsible
/// for queuing that via [`ColorManagementState::queue_info_done`].
///
/// Sends (in spec order):
///   - `primaries` — always, with explicit chromaticities
///   - `primaries_named` — when the description used a named set
///   - `tf_named` — always (we never produce `tf_power` descriptions)
///   - `luminances` — always, defaulting per the TF when unset
///   - `target_primaries` — only when mastering primaries differ from
///     the primary color volume (per spec)
///   - `target_luminance` — always, defaulting per the TF when unset
///   - `target_max_cll` / `target_max_fall` — only when set
fn emit_info_events(info: &WpImageDescriptionInfoV1, desc: &ImageDescription) {
    let (chroma, named) = match desc.primaries {
        PrimaryVolume::Named(p) => (chromaticities_for_named(p), Some(p)),
        PrimaryVolume::Explicit(c) => (c, None),
    };
    info.primaries(
        chroma.r_x, chroma.r_y, chroma.g_x, chroma.g_y, chroma.b_x, chroma.b_y, chroma.w_x,
        chroma.w_y,
    );
    if let Some(p) = named {
        info.primaries_named(p);
    }

    info.tf_named(desc.tf);

    let lums = desc
        .luminances
        .unwrap_or_else(|| default_luminances_for_tf(desc.tf));
    info.luminances(lums.min_lum_ticks, lums.max_lum, lums.reference_lum);

    if let Some(target) = desc.mastering_primaries {
        let equal = target.r_x == chroma.r_x
            && target.r_y == chroma.r_y
            && target.g_x == chroma.g_x
            && target.g_y == chroma.g_y
            && target.b_x == chroma.b_x
            && target.b_y == chroma.b_y
            && target.w_x == chroma.w_x
            && target.w_y == chroma.w_y;
        if !equal {
            info.target_primaries(
                target.r_x, target.r_y, target.g_x, target.g_y, target.b_x, target.b_y, target.w_x,
                target.w_y,
            );
        }
    }

    let target_lum = desc.mastering_luminance.unwrap_or_else(|| {
        let l = default_luminances_for_tf(desc.tf);
        MasteringLuminance {
            min_lum_ticks: l.min_lum_ticks,
            max_lum: l.max_lum,
        }
    });
    info.target_luminance(target_lum.min_lum_ticks, target_lum.max_lum);

    if let Some(cll) = desc.max_cll {
        info.target_max_cll(cll);
    }
    if let Some(fall) = desc.max_fall {
        info.target_max_fall(fall);
    }
}

/// CIE 1931 xy chromaticities × 1,000,000 for the named primary sets
/// we advertise. Values pulled from the standards (BT.709 for sRGB,
/// BT.2020 for the wide-gamut entry). Unsupported entries fall back
/// to sRGB — they can't reach here today (only `SUPPORTED_PRIMARIES`
/// is accepted by the params creator) but the fallback keeps the
/// helper total for future expansions.
fn chromaticities_for_named(p: Primaries) -> PrimaryChromaticities {
    match p {
        Primaries::Bt2020 => PrimaryChromaticities {
            r_x: 708_000,
            r_y: 292_000,
            g_x: 170_000,
            g_y: 797_000,
            b_x: 131_000,
            b_y: 46_000,
            w_x: 312_700,
            w_y: 329_000,
        },
        // Display-P3 (P3-D65): wider than sRGB, D65 white (shared with
        // BT.2020, so the conversion is a pure gamut rotation). Matches
        // prism_frame::Chromaticities::DISPLAY_P3.
        Primaries::DisplayP3 => PrimaryChromaticities {
            r_x: 680_000,
            r_y: 320_000,
            g_x: 265_000,
            g_y: 690_000,
            b_x: 150_000,
            b_y: 60_000,
            w_x: 312_700,
            w_y: 329_000,
        },
        // sRGB / BT.709 — also the safe fallback for any primary we don't
        // tabulate. Every entry in SUPPORTED_PRIMARIES should have its own
        // arm above so a wide-gamut surface never silently lands here.
        _ => PrimaryChromaticities {
            r_x: 640_000,
            r_y: 330_000,
            g_x: 300_000,
            g_y: 600_000,
            b_x: 150_000,
            b_y: 60_000,
            w_x: 312_700,
            w_y: 329_000,
        },
    }
}

/// Spec-defined default luminances for the named TFs we support.
/// Pulled from the `wp_color_manager_v1.transfer_function` enum and
/// `set_luminances` request docs in `color-management-v1.xml`.
fn default_luminances_for_tf(tf: TransferFunction) -> Luminances {
    match tf {
        // PQ: 0.005 cd/m² min, 10000 cd/m² max, 203 cd/m² reference.
        TransferFunction::St2084Pq => Luminances {
            min_lum_ticks: 50, // 0.005 × 10000
            max_lum: 10_000,
            reference_lum: 203,
        },
        // BT.1886 per ITU-R BT.2035: 0.01 / 100 / 100.
        TransferFunction::Bt1886 => Luminances {
            min_lum_ticks: 100, // 0.01 × 10000
            max_lum: 100,
            reference_lum: 100,
        },
        // sRGB / Gamma22 / ExtLinear all default to the IEC sRGB
        // canonical range (0.2 / 80 / 80).
        _ => Luminances {
            min_lum_ticks: 2_000, // 0.2 × 10000
            max_lum: 80,
            reference_lum: 80,
        },
    }
}

// ─── wp_image_description_v1 ───────────────────────────────────────────────

impl Dispatch<WpImageDescriptionV1, ImageDescriptionData> for PrismState {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &WpImageDescriptionV1,
        request: <WpImageDescriptionV1 as Resource>::Request,
        data: &ImageDescriptionData,
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        use wp_image_description_v1::Request;
        match request {
            Request::Destroy => {}
            Request::GetInformation { information } => {
                // Per spec, `get_information` MUST emit a defined set
                // of events followed by `done`. We always init the
                // info resource (the New<R> can't be dropped) and
                // emit events when we have a real description backing
                // this parent. For an inert/failed description we
                // emit nothing and just enqueue the terminating
                // `done`, since the protocol object is unusable
                // anyway and the client shouldn't be querying it.
                //
                // `done` is a destructor event and would UAF the
                // freshly-created resource if sent inline (see
                // [`ColorManagementState::pending_info_done`]); the
                // main loop drains the queue after dispatch returns.
                let info = data_init.init(information, ());
                if let Some(desc) = data.description.as_ref() {
                    emit_info_events(&info, desc);
                }
                state.color_management.queue_info_done(info);
            }
            _ => {}
        }
    }
}

impl Dispatch<WpImageDescriptionInfoV1, ()> for PrismState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WpImageDescriptionInfoV1,
        _request: <WpImageDescriptionInfoV1 as Resource>::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
    }
}

// ─── wp_color_management_surface_v1 ────────────────────────────────────────

impl Dispatch<WpColorManagementSurfaceV1, ColorSurfaceData> for PrismState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        resource: &WpColorManagementSurfaceV1,
        request: <WpColorManagementSurfaceV1 as Resource>::Request,
        data: &ColorSurfaceData,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        use wp_color_management_surface_v1::Request;
        match request {
            Request::Destroy => {
                // Spec: destroying the surface-extension also does the
                // equivalent of unset_image_description. If the
                // underlying wl_surface is still alive, queue an unset
                // for next commit.
                if let Ok(surface) = data.surface.upgrade() {
                    queue_unset(&surface);
                }
            }
            Request::SetImageDescription {
                image_description,
                render_intent,
            } => {
                let Ok(surface) = data.surface.upgrade() else {
                    resource.post_error(
                        wp_color_management_surface_v1::Error::Inert,
                        "wl_surface gone".to_string(),
                    );
                    return;
                };
                let intent = match render_intent.into_result() {
                    Ok(i) if SUPPORTED_INTENTS.contains(&i) => i,
                    _ => {
                        resource.post_error(
                            wp_color_management_surface_v1::Error::RenderIntent,
                            "unsupported render intent".to_string(),
                        );
                        return;
                    }
                };
                let desc = match image_description.data::<ImageDescriptionData>() {
                    Some(d) => match d.description.as_ref() {
                        Some(desc) => desc.clone(),
                        None => {
                            resource.post_error(
                                wp_color_management_surface_v1::Error::ImageDescription,
                                "image description is not in ready state".to_string(),
                            );
                            return;
                        }
                    },
                    None => {
                        resource.post_error(
                            wp_color_management_surface_v1::Error::ImageDescription,
                            "image description has no user data".to_string(),
                        );
                        return;
                    }
                };
                queue_set(&surface, desc, intent);
            }
            Request::UnsetImageDescription => {
                let Ok(surface) = data.surface.upgrade() else {
                    resource.post_error(
                        wp_color_management_surface_v1::Error::Inert,
                        "wl_surface gone".to_string(),
                    );
                    return;
                };
                queue_unset(&surface);
            }
            _ => {}
        }
    }

    fn destroyed(
        _state: &mut Self,
        _client: ClientId,
        _resource: &WpColorManagementSurfaceV1,
        data: &ColorSurfaceData,
    ) {
        // Same intent as the Destroy request — if the resource went
        // away by client disconnect rather than explicit destroy,
        // still clean up.
        if let Ok(surface) = data.surface.upgrade() {
            queue_unset(&surface);
        }
    }
}

fn queue_set(surface: &WlSurface, desc: Arc<ImageDescription>, intent: RenderIntent) {
    compositor::with_states(surface, |states| {
        states
            .data_map
            .insert_if_missing_threadsafe(SurfaceColorSlot::default);
        let slot = states.data_map.get::<SurfaceColorSlot>().unwrap();
        let mut st = slot.0.lock().unwrap();
        st.pending = Some((desc, intent));
        st.pending_unset = false;
    });
}

fn queue_unset(surface: &WlSurface) {
    compositor::with_states(surface, |states| {
        states
            .data_map
            .insert_if_missing_threadsafe(SurfaceColorSlot::default);
        let slot = states.data_map.get::<SurfaceColorSlot>().unwrap();
        let mut st = slot.0.lock().unwrap();
        st.pending_unset = true;
        st.pending = None;
    });
}

// ─── Helpers: WEnum conversions ────────────────────────────────────────────

fn into_tf(
    v: smithay::reexports::wayland_server::WEnum<TransferFunction>,
) -> Option<TransferFunction> {
    v.into_result().ok()
}

fn into_primaries(v: smithay::reexports::wayland_server::WEnum<Primaries>) -> Option<Primaries> {
    v.into_result().ok()
}
