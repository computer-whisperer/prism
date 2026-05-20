//! Read-only DRM resource enumeration.
//!
//! Opens a DRM device without acquiring master, lists connectors / modes /
//! CRTCs / planes. For inspection and target-output selection before the
//! compositor takes over scanout.

use std::os::fd::AsFd;
use std::path::Path;

use anyhow::{Context, Result};
use smithay::reexports::drm::control::{
    Device as ControlDevice, ModeTypeFlags, connector, crtc, plane,
};
use smithay::reexports::drm::{ClientCapability, Device as BasicDevice};

/// File-descriptor newtype implementing the drm-rs traits.
///
/// Read-only enumeration path. The real compositor opens its DRM device via a
/// libseat session (so it can acquire master); this type is for inspection
/// while the desktop session still owns the device.
pub struct DrmFd(std::fs::File);

impl AsFd for DrmFd {
    fn as_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl BasicDevice for DrmFd {}
impl ControlDevice for DrmFd {}

/// Open a DRM device by path. Does NOT acquire master. Enables the
/// universal-planes client capability so plane enumeration returns overlay
/// + cursor planes, not just the primary.
pub fn open_for_enumeration(path: impl AsRef<Path>) -> Result<DrmFd> {
    let path = path.as_ref();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let dev = DrmFd(file);
    // Failing to enable universal planes is non-fatal; we just see fewer planes.
    let _ = dev.set_client_capability(ClientCapability::UniversalPlanes, true);
    Ok(dev)
}

#[derive(Debug)]
pub struct ConnectorSummary {
    pub handle: connector::Handle,
    pub kind: connector::Interface,
    pub kind_id: u32,
    pub state: connector::State,
    pub modes: Vec<smithay::reexports::drm::control::Mode>,
    pub current_encoder: Option<smithay::reexports::drm::control::encoder::Handle>,
    pub physical_size_mm: Option<(u32, u32)>,
}

impl ConnectorSummary {
    /// Display-friendly name (e.g. `DisplayPort-4`).
    pub fn name(&self) -> String {
        format!("{:?}-{}", self.kind, self.kind_id)
    }

    /// The preferred mode if the EDID/driver tagged one; else the first mode.
    pub fn preferred_mode(&self) -> Option<&smithay::reexports::drm::control::Mode> {
        self.modes
            .iter()
            .find(|m| m.mode_type().contains(ModeTypeFlags::PREFERRED))
            .or_else(|| self.modes.first())
    }
}

#[derive(Debug)]
pub struct DeviceSummary {
    pub connectors: Vec<ConnectorSummary>,
    pub crtcs: Vec<crtc::Handle>,
    pub planes: Vec<plane::Handle>,
}

pub fn summarize(dev: &DrmFd) -> Result<DeviceSummary> {
    let resources = dev.resource_handles().context("resource_handles")?;

    let mut connectors = Vec::new();
    for &handle in resources.connectors() {
        let info = dev
            .get_connector(handle, false)
            .with_context(|| format!("get_connector {handle:?}"))?;
        connectors.push(ConnectorSummary {
            handle,
            kind: info.interface(),
            kind_id: info.interface_id(),
            state: info.state(),
            modes: info.modes().to_vec(),
            current_encoder: info.current_encoder(),
            physical_size_mm: info.size(),
        });
    }

    let crtcs = resources.crtcs().to_vec();
    let planes = dev.plane_handles().context("plane_handles")?;

    Ok(DeviceSummary {
        connectors,
        crtcs,
        planes,
    })
}
