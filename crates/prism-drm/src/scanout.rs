//! Helpers for the scanout path: picking an output, building a framebuffer
//! from a GBM BO, and producing the `PlaneState` for an atomic commit.

use anyhow::{Context, Result, anyhow};
use drm_fourcc::DrmFourcc;
use gbm::BufferObject;
use smithay::backend::drm::DrmDevice;
use smithay::reexports::drm::buffer::PlanarBuffer;
use smithay::reexports::drm::control::{
    Device as ControlDevice, FbCmd2Flags, Mode, ModeTypeFlags, ResourceHandle, connector, crtc,
    framebuffer, property,
};

/// Bit depth + format selection for a scanout BO. Picks the matching DRM
/// fourcc and the Vulkan format that interprets the same memory layout.
///
/// `Bpc8` → DRM `XR24` ↔ Vulkan `B8G8R8A8_UNORM`. Standard SDR scanout.
/// `Bpc10` → DRM `XR30` ↔ Vulkan `A2R10G10B10_UNORM_PACK32`. Higher
///   precision; required for HDR and for SDR-without-banding on smooth
///   gradients. Pair with `max_bpc=10` on the connector to actually push
///   10 bits over the wire (else driver dithers down).
///
/// Choice is per-output; some displays don't support 10-bit links (cheap
/// 1080p panels). Negotiation belongs in the per-output config layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanoutDepth {
    Bpc8,
    Bpc10,
}

impl ScanoutDepth {
    pub fn drm_fourcc(self) -> DrmFourcc {
        match self {
            Self::Bpc8 => DrmFourcc::Xrgb8888,
            Self::Bpc10 => DrmFourcc::Xrgb2101010,
        }
    }

    /// The `max bpc` value to push to the connector for this depth.
    pub fn max_bpc(self) -> u64 {
        match self {
            Self::Bpc8 => 8,
            Self::Bpc10 => 10,
        }
    }
}

/// One connected output's wiring choices for the tracer.
#[derive(Debug)]
pub struct OutputPick {
    pub connector: connector::Handle,
    pub mode: Mode,
    pub crtc: crtc::Handle,
    pub connector_name: String,
}

/// Pick a connected output: first connected connector with a preferred mode
/// and a compatible (currently-unused) CRTC. Good enough for the single-screen
/// scanout smoke test; the real compositor will allow user-driven assignments.
pub fn pick_first_connected(drm: &DrmDevice) -> Result<OutputPick> {
    pick_matching(drm, |_name| true)
}

/// Pick a specific output by name. Accepts the full connector name
/// (`DisplayPort-6`) or the common short alias (`DP-6`, `HDMI-1`),
/// case-insensitively.
pub fn pick_by_name(drm: &DrmDevice, want: &str) -> Result<OutputPick> {
    let want_lc = want.to_lowercase();
    let want_normalized = expand_alias(&want_lc);
    pick_matching(drm, |name| {
        let lc = name.to_lowercase();
        lc == want_lc || lc == want_normalized
    })
    .with_context(|| format!("no connected output matched {want:?}"))
}

/// Expand short-form connector aliases to the kernel-reported full names.
fn expand_alias(input: &str) -> String {
    if let Some(rest) = input.strip_prefix("dp-") {
        format!("displayport-{rest}")
    } else if let Some(rest) = input.strip_prefix("hdmi-") {
        format!("hdmi-a-{rest}")
    } else {
        input.to_string()
    }
}

fn pick_matching<F>(drm: &DrmDevice, matches: F) -> Result<OutputPick>
where
    F: Fn(&str) -> bool,
{
    let resources = drm.resource_handles().context("resource_handles")?;

    // Build the set of CRTCs currently bound to *other* connectors (a prior
    // desktop session usually leaves these bound to its assignments). Reusing
    // one would require atomically disabling the other connector's CRTC in
    // the same commit, which we don't do — the kernel rejects the test
    // commit with "Atomic Test failed for crtc X". So treat them as occupied.
    let mut occupied_crtcs: Vec<crtc::Handle> = Vec::new();
    for &c in resources.connectors() {
        let info = drm.get_connector(c, false).ok();
        let Some(info) = info else { continue };
        if info.state() != connector::State::Connected {
            continue;
        }
        let Some(enc_h) = info.current_encoder() else {
            continue;
        };
        let Ok(enc) = drm.get_encoder(enc_h) else {
            continue;
        };
        if let Some(crtc_h) = enc.crtc() {
            occupied_crtcs.push(crtc_h);
        }
    }

    for &conn_h in resources.connectors() {
        let info = drm
            .get_connector(conn_h, false)
            .with_context(|| format!("get_connector {conn_h:?}"))?;
        if info.state() != connector::State::Connected {
            continue;
        }
        let name = format!("{:?}-{}", info.interface(), info.interface_id());
        if !matches(&name) {
            continue;
        }
        let mode = info
            .modes()
            .iter()
            .find(|m| m.mode_type().contains(ModeTypeFlags::PREFERRED))
            .or_else(|| info.modes().first())
            .copied();
        let Some(mode) = mode else {
            continue;
        };

        // Allowed: our connector's own current CRTC (if any), plus any free
        // CRTC. Reject CRTCs bound to *other* connectors.
        let own_crtc: Option<crtc::Handle> = info
            .current_encoder()
            .and_then(|enc_h| drm.get_encoder(enc_h).ok())
            .and_then(|enc| enc.crtc());

        let mut chosen_crtc: Option<crtc::Handle> = None;
        'pick: for &enc_h in info.encoders() {
            let enc = drm
                .get_encoder(enc_h)
                .with_context(|| format!("get_encoder {enc_h:?}"))?;
            for candidate in resources.filter_crtcs(enc.possible_crtcs()) {
                let occupied_by_other = occupied_crtcs.contains(&candidate)
                    && Some(candidate) != own_crtc;
                if !occupied_by_other {
                    chosen_crtc = Some(candidate);
                    break 'pick;
                }
            }
        }
        let Some(crtc_h) = chosen_crtc else {
            return Err(anyhow!(
                "no free CRTC available for {name} (all compatible CRTCs are bound to other connectors)"
            ));
        };

        return Ok(OutputPick {
            connector: conn_h,
            mode,
            crtc: crtc_h,
            connector_name: name,
        });
    }
    Err(anyhow!("no connected connector with a usable mode + CRTC"))
}

/// Add a framebuffer for a GBM BO. The BO must have a non-INVALID modifier
/// (LINEAR / explicit-modifier BOs from `GbmDevice::allocate_scanout` qualify).
pub fn add_framebuffer_for_bo<T: 'static>(
    drm: &DrmDevice,
    bo: &BufferObject<T>,
) -> Result<framebuffer::Handle>
where
    BufferObject<T>: PlanarBuffer,
{
    let fb = drm
        .add_planar_framebuffer(bo, FbCmd2Flags::MODIFIERS)
        .context("add_planar_framebuffer")?;
    Ok(fb)
}

/// Find a named property on a resource by walking its property list.
/// Returns `None` if no such property exists on this object.
pub fn find_property<H: ResourceHandle>(
    drm: &DrmDevice,
    handle: H,
    name: &str,
) -> Result<Option<property::Handle>> {
    let props = drm.get_properties(handle).context("get_properties")?;
    for (&prop_h, _) in &props {
        let info = drm.get_property(prop_h).context("get_property")?;
        if info.name().to_string_lossy() == name {
            return Ok(Some(prop_h));
        }
    }
    Ok(None)
}

/// Set `max bpc` on a connector via the legacy property API.
///
/// `max bpc` controls the bit depth used on the physical link to the
/// display. Default is usually 8; setting it to 10 lets us send full
/// 10-bit scanout (paired with an A2R10G10B10 framebuffer). Without this
/// the driver dithers our 10-bit framebuffer down to 8 bits on the wire.
///
/// Returns `Ok(false)` if the property isn't exposed on this connector
/// (some drivers omit it for HDMI/DP variants); the caller can treat
/// that as "use whatever depth the link defaulted to". Returns `Ok(true)`
/// on a successful set.
pub fn set_connector_max_bpc(
    drm: &DrmDevice,
    connector: connector::Handle,
    value: u64,
) -> Result<bool> {
    let Some(prop) = find_property(drm, connector, "max bpc")? else {
        return Ok(false);
    };
    drm.set_property(connector, prop, value)
        .with_context(|| format!("set_property max_bpc={value}"))?;
    Ok(true)
}
