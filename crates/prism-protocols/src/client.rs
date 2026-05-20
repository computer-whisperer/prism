//! Per-client state attached to each connected Wayland client.
//!
//! Smithay needs us to provide a `ClientData` for each new client so it can
//! track per-client compositor state (most importantly, per-client surface
//! scale for fractional scaling). Anything we want to carry per-client lives
//! here too — security context, sandboxing tags, etc., in the future.

use smithay::reexports::wayland_server::backend::{ClientData, ClientId, DisconnectReason};
use smithay::wayland::compositor::CompositorClientState;

#[derive(Default, Debug)]
pub struct PrismClient {
    pub compositor: CompositorClientState,
}

impl ClientData for PrismClient {
    fn initialized(&self, client_id: ClientId) {
        tracing::debug!(?client_id, "wayland client initialized");
    }

    fn disconnected(&self, client_id: ClientId, reason: DisconnectReason) {
        tracing::debug!(?client_id, ?reason, "wayland client disconnected");
    }
}
