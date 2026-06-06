//! `wp_drm_lease_device_v1` — DRM leasing for VR headsets.
//!
//! The kernel flags HMD connectors `non-desktop` (EDID quirk list).
//! Bringup skips those as desktop outputs and instead reserves a free
//! CRTC per headset (`prism_drm::scanout::scan_non_desktop`), recorded
//! here per card. One lease global is created per card; a VR runtime
//! (SteamVR's vrcompositor, Monado) binds it, sees the advertised
//! connector, and requests a lease. We grant the connector, its
//! reserved CRTC, and a claimed primary plane; the kernel hands the
//! client a limited DRM fd with direct modeset/page-flip rights on
//! exactly those resources. Dropping our [`DrmLease`] (or the client
//! exiting) revokes the lease.
//!
//! The advertised connectors are withdrawn while the session is paused
//! (VT switched away) and re-advertised on resume — `main.rs` drives
//! that from the libseat session events, mirroring the `card.drm`
//! pause/activate calls.
//!
//! Connector hotplug is not handled (prism has no connector hotplug at
//! all yet): the headset must be plugged in before compositor launch,
//! and stays advertised if unplugged after.

use std::collections::HashMap;

use prism_renderer::DrmDevId;
use smithay::backend::drm::DrmNode;
use smithay::delegate_drm_lease;
use smithay::wayland::drm_lease::{
    DrmLease, DrmLeaseBuilder, DrmLeaseHandler, DrmLeaseRequest, DrmLeaseState, LeaseRejected,
};

use crate::state::PrismState;

/// Per-card leasing state: the wayland global plus the connectors it
/// advertises and the leases currently handed out.
pub struct CardLeaseState {
    pub lease_state: DrmLeaseState,
    /// Non-desktop connectors advertised on this card's global, each
    /// with its CRTC reserved at bringup.
    pub non_desktop: Vec<prism_drm::NonDesktopConnector>,
    /// Currently-active leases. Holding the [`DrmLease`] keeps the
    /// kernel lease alive; dropping one revokes it.
    pub active_leases: Vec<DrmLease>,
}

/// smithay keys lease callbacks by [`DrmNode`]; our card map is keyed
/// by [`DrmDevId`]. Both are the primary node's major/minor.
fn dev_id_of(node: DrmNode) -> DrmDevId {
    DrmDevId {
        major: node.major() as i64,
        minor: node.minor() as i64,
    }
}

/// Create one `wp_drm_lease_device_v1` global per attached card and
/// advertise its non-desktop connectors. Called once at startup, after
/// cards are attached and before the wayland socket goes live. Cards
/// whose global can't be created (no openable non-master fd) are
/// skipped with a warning — leasing is then unavailable on that card.
pub fn init(
    state: &mut PrismState,
    mut non_desktop_by_card: HashMap<DrmDevId, Vec<prism_drm::NonDesktopConnector>>,
) {
    let dh = state.display_handle.clone();
    for (dev_id, card) in &state.cards {
        let mut lease_state = match DrmLeaseState::new::<PrismState>(&dh, &card.node) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("card {}: no DRM lease global: {e}", card.path);
                continue;
            }
        };
        let non_desktop = non_desktop_by_card.remove(dev_id).unwrap_or_default();
        for ndc in &non_desktop {
            tracing::info!(
                "card {}: advertising {} ({}) for DRM leasing",
                card.path,
                ndc.connector_name,
                ndc.description
            );
            lease_state.add_connector::<PrismState>(
                ndc.connector,
                ndc.connector_name.clone(),
                ndc.description.clone(),
            );
        }
        state.drm_lease.insert(
            *dev_id,
            CardLeaseState {
                lease_state,
                non_desktop,
                active_leases: Vec::new(),
            },
        );
    }
}

impl DrmLeaseHandler for PrismState {
    fn drm_lease_state(&mut self, node: DrmNode) -> &mut DrmLeaseState {
        // The global's data carries the node we created it with, so a
        // missing entry is unreachable while cards are never detached.
        &mut self
            .drm_lease
            .get_mut(&dev_id_of(node))
            .expect("drm_lease callback for a card without lease state")
            .lease_state
    }

    fn lease_request(
        &mut self,
        node: DrmNode,
        request: DrmLeaseRequest,
    ) -> Result<DrmLeaseBuilder, LeaseRejected> {
        let dev_id = dev_id_of(node);
        let card = self.cards.get(&dev_id).ok_or_else(LeaseRejected::default)?;
        let lease = self
            .drm_lease
            .get(&dev_id)
            .ok_or_else(LeaseRejected::default)?;
        tracing::info!(
            "DRM lease request on {} for {} connector(s)",
            card.path,
            request.connectors.len()
        );
        let mut builder = DrmLeaseBuilder::new(&card.drm);
        for conn in request.connectors {
            let ndc = lease
                .non_desktop
                .iter()
                .find(|n| n.connector == conn)
                .ok_or_else(|| {
                    tracing::warn!(
                        "lease request for a connector that isn't non-desktop; rejecting"
                    );
                    LeaseRejected::default()
                })?;
            builder.add_connector(conn);
            builder.add_crtc(ndc.crtc);
            // The headset client needs at least a primary plane on its
            // CRTC to scan out (same policy as niri).
            let planes = card
                .drm
                .planes(&ndc.crtc)
                .map_err(LeaseRejected::with_cause)?;
            let (primary, claim) = planes
                .primary
                .iter()
                .find_map(|p| card.drm.claim_plane(p.handle, ndc.crtc).map(|c| (p, c)))
                .ok_or_else(LeaseRejected::default)?;
            builder.add_plane(primary.handle, claim);
        }
        Ok(builder)
    }

    fn new_active_lease(&mut self, node: DrmNode, lease: DrmLease) {
        tracing::info!("DRM lease {} granted", lease.id());
        if let Some(cls) = self.drm_lease.get_mut(&dev_id_of(node)) {
            cls.active_leases.push(lease);
        }
    }

    fn lease_destroyed(&mut self, node: DrmNode, lease_id: u32) {
        tracing::info!("DRM lease {lease_id} destroyed");
        if let Some(cls) = self.drm_lease.get_mut(&dev_id_of(node)) {
            cls.active_leases.retain(|l| l.id() != lease_id);
        }
    }
}

delegate_drm_lease!(PrismState);
