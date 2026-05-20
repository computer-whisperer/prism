//! Helpers for the scanout path: picking an output, building a framebuffer
//! from a GBM BO, and producing the `PlaneState` for an atomic commit.

use anyhow::{Context, Result, anyhow};
use gbm::BufferObject;
use smithay::backend::drm::DrmDevice;
use smithay::reexports::drm::buffer::PlanarBuffer;
use smithay::reexports::drm::control::{
    Device as ControlDevice, FbCmd2Flags, Mode, ModeTypeFlags, connector, crtc, framebuffer,
};

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
