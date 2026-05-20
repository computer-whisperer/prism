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
    let resources = drm.resource_handles().context("resource_handles")?;

    for &conn_h in resources.connectors() {
        let info = drm
            .get_connector(conn_h, false)
            .with_context(|| format!("get_connector {conn_h:?}"))?;
        if info.state() != connector::State::Connected {
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

        // Find a CRTC that any of this connector's encoders supports. We use
        // the first compatible CRTC reported by the DRM resources; for a real
        // compositor with multiple outputs we'd need to do CRTC assignment
        // properly to avoid conflicts.
        let mut chosen_crtc: Option<crtc::Handle> = None;
        for &enc_h in info.encoders() {
            let enc = drm
                .get_encoder(enc_h)
                .with_context(|| format!("get_encoder {enc_h:?}"))?;
            for candidate in resources.filter_crtcs(enc.possible_crtcs()) {
                chosen_crtc = Some(candidate);
                break;
            }
            if chosen_crtc.is_some() {
                break;
            }
        }
        let Some(crtc_h) = chosen_crtc else {
            continue;
        };

        return Ok(OutputPick {
            connector: conn_h,
            mode,
            crtc: crtc_h,
            connector_name: format!("{:?}-{}", info.interface(), info.interface_id()),
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
