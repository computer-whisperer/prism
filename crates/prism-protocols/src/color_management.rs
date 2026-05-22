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
//! - `wp_color_management_output_v1` — per-output description advertising;
//!   surface_feedback (Step 3) is what clients actually use.
//! - `wp_color_management_surface_feedback_v1` — Step 3 of the plan.
//! - `create_windows_scrgb` — niche; can add when needed.
//! - `wp_image_description_info_v1` (`get_information` events) — only
//!   needed for descriptions obtained via the upcoming `get_image_description`
//!   path; descriptions created via the params creator can't be queried.
//!
//! **What this gives us today:** clients can declare the color encoding
//! of their surface contents (e.g. "PQ-encoded BT.2020 mastered to 400
//! nits"). The compositor stores the description on the surface but
//! does **not yet** consume it in the render path — that's Step 4.
//! Until then, treating a colour-managed surface as sRGB is the
//! existing behaviour; the protocol surface lands first so subsequent
//! work can light up incrementally.
//!
//! **Identity policy:** descriptions get a monotonically-increasing 64-bit
//! ID assigned at creation. Identity is opaque to clients today
//! (only used by `preferred_changed2` in Step 3). Step 3 may revisit
//! to content-hash so the same parametric description gets a stable ID.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use smithay::reexports::wayland_protocols::wp::color_management::v1::server::{
    wp_color_management_surface_v1::{
        self, WpColorManagementSurfaceV1,
    },
    wp_color_manager_v1::{
        self, Feature, Primaries, RenderIntent, TransferFunction, WpColorManagerV1,
    },
    wp_image_description_creator_params_v1::{
        self, WpImageDescriptionCreatorParamsV1,
    },
    wp_image_description_v1::{self, WpImageDescriptionV1},
};
use smithay::reexports::wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
    backend::ClientId,
    protocol::wl_surface::WlSurface,
};
use smithay::wayland::compositor::{self, SurfaceData};

use crate::state::PrismState;

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
/// Spyder + general clients minimally need sRGB and BT.2020; more
/// can be added as use cases arrive (DCI-P3 / Display P3 for video).
const SUPPORTED_PRIMARIES: &[Primaries] =
    &[Primaries::Srgb, Primaries::Bt2020];

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

/// Holds the description-identity counter and the global handle.
/// Single instance lives on `PrismState`.
pub struct ColorManagementState {
    next_identity: AtomicU64,
}

impl ColorManagementState {
    pub fn new(dh: &DisplayHandle) -> Self {
        // Two versions advertised. v2 added preferred_changed2,
        // get_image_description, and compound_power_2_4 / absolute_no_adaptation.
        let _global = dh.create_global::<PrismState, WpColorManagerV1, ()>(2, ());
        Self {
            next_identity: AtomicU64::new(1),
        }
    }

    fn next_identity(&self) -> u64 {
        self.next_identity.fetch_add(1, Ordering::Relaxed)
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
            Request::GetOutput { id, output: _ } => {
                // Spec: posts a protocol error if we don't support it,
                // but we *do* need to construct *something* — for now
                // create the resource and let it be inert (its
                // get_image_description sends `failed`). Step 3 wires
                // this up properly.
                let _ = data_init.init(id, ());
                tracing::debug!(
                    "wp_color_manager_v1.get_output: not yet implemented; \
                     resource will be inert"
                );
            }
            Request::GetSurface { id, surface } => {
                let data = ColorSurfaceData {
                    surface: surface.downgrade(),
                };
                let _ = data_init.init(id, data);
            }
            Request::GetSurfaceFeedback { id, surface: _ } => {
                // Step 3.
                let _ = data_init.init(id, ());
                tracing::debug!(
                    "wp_color_manager_v1.get_surface_feedback: not yet implemented"
                );
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
            Request::GetImageDescription { image_description, reference: _ } => {
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

// Inert placeholders for resources we initialize but don't (yet)
// fully implement. They satisfy the wayland-server dispatch
// requirement; their requests are no-ops or, where the spec
// mandates a response, send the appropriate inert/failed signal.

use smithay::reexports::wayland_protocols::wp::color_management::v1::server::wp_color_management_output_v1::{
    self, WpColorManagementOutputV1,
};
impl Dispatch<WpColorManagementOutputV1, ()> for PrismState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WpColorManagementOutputV1,
        request: <WpColorManagementOutputV1 as Resource>::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        use wp_color_management_output_v1::Request;
        match request {
            Request::Destroy => {}
            Request::GetImageDescription { image_description } => {
                let desc =
                    data_init.init(image_description, ImageDescriptionData { description: None });
                // Per spec for an inert protocol object: immediately
                // deliver `failed`.
                desc.failed(
                    wp_image_description_v1::Cause::NoOutput,
                    "wp_color_management_output_v1 not implemented".to_string(),
                );
            }
            _ => {}
        }
    }
}

use smithay::reexports::wayland_protocols::wp::color_management::v1::server::wp_color_management_surface_feedback_v1::{
    self, WpColorManagementSurfaceFeedbackV1,
};
impl Dispatch<WpColorManagementSurfaceFeedbackV1, ()> for PrismState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        resource: &WpColorManagementSurfaceFeedbackV1,
        request: <WpColorManagementSurfaceFeedbackV1 as Resource>::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        use wp_color_management_surface_feedback_v1::Request;
        match request {
            Request::Destroy => {}
            Request::GetPreferred { image_description }
            | Request::GetPreferredParametric { image_description } => {
                let desc =
                    data_init.init(image_description, ImageDescriptionData { description: None });
                desc.failed(
                    wp_image_description_v1::Cause::NoOutput,
                    "surface_feedback not implemented yet".to_string(),
                );
                let _ = resource;
            }
            _ => {}
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
                        desc_resource.failed(
                            wp_image_description_v1::Cause::Unsupported,
                            reason.into(),
                        );
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
                r_x, r_y, g_x, g_y, b_x, b_y, w_x, w_y,
            } => {
                if inner.primaries.is_some() {
                    resource.post_error(
                        wp_image_description_creator_params_v1::Error::AlreadySet,
                        "primaries already set".to_string(),
                    );
                    return;
                }
                inner.primaries = Some(PrimaryVolume::Explicit(PrimaryChromaticities {
                    r_x, r_y, g_x, g_y, b_x, b_y, w_x, w_y,
                }));
            }
            Request::SetLuminances {
                min_lum, max_lum, reference_lum,
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
                r_x, r_y, g_x, g_y, b_x, b_y, w_x, w_y,
            } => {
                if inner.mastering_primaries.is_some() {
                    resource.post_error(
                        wp_image_description_creator_params_v1::Error::AlreadySet,
                        "mastering primaries already set".to_string(),
                    );
                    return;
                }
                inner.mastering_primaries = Some(PrimaryChromaticities {
                    r_x, r_y, g_x, g_y, b_x, b_y, w_x, w_y,
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

// ─── wp_image_description_v1 ───────────────────────────────────────────────

impl Dispatch<WpImageDescriptionV1, ImageDescriptionData> for PrismState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &WpImageDescriptionV1,
        request: <WpImageDescriptionV1 as Resource>::Request,
        _data: &ImageDescriptionData,
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        use wp_image_description_v1::Request;
        match request {
            Request::Destroy => {}
            Request::GetInformation { information } => {
                // Per spec, descriptions created via the params
                // creator do NOT allow get_information. Spec doesn't
                // define a protocol error for this so we init the
                // resource and immediately send the destructor `done`
                // (well-behaved clients won't call this).
                let info = data_init.init(information, ());
                info.done();
            }
            _ => {}
        }
    }
}

use smithay::reexports::wayland_protocols::wp::color_management::v1::server::wp_image_description_info_v1::WpImageDescriptionInfoV1;
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
            Request::SetImageDescription { image_description, render_intent } => {
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

fn into_tf(v: smithay::reexports::wayland_server::WEnum<TransferFunction>) -> Option<TransferFunction> {
    v.into_result().ok()
}

fn into_primaries(
    v: smithay::reexports::wayland_server::WEnum<Primaries>,
) -> Option<Primaries> {
    v.into_result().ok()
}
