//! `zwlr_output_power_management_v1` — DPMS control for clients like
//! `wlopm` (the usual swayidle display-sleep command).
//!
//! Hand-rolled: smithay ships no output-power-management. A client binds the
//! manager and calls `get_output_power(wl_output)` to obtain a
//! `zwlr_output_power_v1` handle, then `set_mode(off|on)` to drive that
//! output's DPMS. We resolve the `wl_output` to our [`OutputId`] and call
//! [`PrismState::set_monitor_powered`], which clears / re-modesets the CRTC
//! (see [`prism_drm::OutputContext::power_off`]). We echo the resulting
//! `mode` event and broadcast `mode` to every bound power object whenever
//! the state changes by *any* means (protocol, IPC action, keybind), so a
//! watching client stays in sync.

use smithay::output::Output;
use smithay::reexports::wayland_protocols_wlr::output_power_management::v1::server::{
    zwlr_output_power_manager_v1::{self, ZwlrOutputPowerManagerV1},
    zwlr_output_power_v1::{self, Mode, ZwlrOutputPowerV1},
};
use smithay::reexports::wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource, WEnum,
};

use crate::state::{OutputId, PrismState};

/// Per-`zwlr_output_power_v1` user data: which output it controls. `None`
/// ⇒ the client passed a `wl_output` we don't own (foreign / disconnected);
/// the object is inert and we sent `failed` at creation.
pub struct OutputPowerData {
    pub output_id: Option<OutputId>,
}

/// Create the `zwlr_output_power_manager_v1` global. The returned id is
/// dropped — the display keeps the global alive, and we never remove it.
pub fn create_output_power_global(dh: &DisplayHandle) {
    dh.create_global::<PrismState, ZwlrOutputPowerManagerV1, ()>(1, ());
}

impl PrismState {
    /// Send the current power `mode` of `output_id` to every bound power
    /// object controlling it. Called after any DPMS change so protocol
    /// clients observe state set via IPC / keybinds too. Prunes dead objects.
    pub fn broadcast_output_power_mode(&mut self, output_id: &OutputId) {
        let Some(mode) = self.outputs.get(output_id).map(|ctx| {
            if ctx.is_powered_off() {
                Mode::Off
            } else {
                Mode::On
            }
        }) else {
            return;
        };
        self.output_power_objects.retain(|(_, obj)| obj.is_alive());
        for (id, obj) in &self.output_power_objects {
            if id == output_id {
                obj.mode(mode);
            }
        }
    }
}

impl GlobalDispatch<ZwlrOutputPowerManagerV1, ()> for PrismState {
    fn bind(
        _state: &mut Self,
        _dh: &DisplayHandle,
        _client: &Client,
        resource: New<ZwlrOutputPowerManagerV1>,
        _global_data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        data_init.init(resource, ());
    }
}

impl Dispatch<ZwlrOutputPowerManagerV1, ()> for PrismState {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &ZwlrOutputPowerManagerV1,
        request: <ZwlrOutputPowerManagerV1 as Resource>::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut DataInit<'_, Self>,
    ) {
        use zwlr_output_power_manager_v1::Request;
        match request {
            Request::GetOutputPower { id, output } => {
                // Resolve the wl_output to our connector name. A foreign /
                // disconnected output yields `None` → inert object + `failed`.
                let output_id = Output::from_resource(&output).map(|o| o.name());
                let known = output_id
                    .as_ref()
                    .is_some_and(|id| state.outputs.contains_key(id));
                let power = data_init.init(
                    id,
                    OutputPowerData {
                        output_id: output_id.clone(),
                    },
                );
                match output_id.filter(|_| known) {
                    Some(id) => {
                        // Spec: send the current mode right after creation.
                        let off = state.outputs.get(&id).is_some_and(|c| c.is_powered_off());
                        power.mode(if off { Mode::Off } else { Mode::On });
                        state.output_power_objects.push((id, power));
                    }
                    None => power.failed(),
                }
            }
            Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ZwlrOutputPowerV1, OutputPowerData> for PrismState {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &ZwlrOutputPowerV1,
        request: <ZwlrOutputPowerV1 as Resource>::Request,
        data: &OutputPowerData,
        _dh: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        use zwlr_output_power_v1::Request;
        match request {
            Request::SetMode { mode } => {
                let Some(output_id) = data.output_id.clone() else {
                    return; // inert object (output we don't own)
                };
                let on = match mode {
                    WEnum::Value(Mode::On) => true,
                    WEnum::Value(Mode::Off) => false,
                    _ => {
                        resource.post_error(
                            zwlr_output_power_v1::Error::InvalidMode,
                            "nonexistent power save mode",
                        );
                        return;
                    }
                };
                // Drives the CRTC and broadcasts the resulting `mode` to all
                // bound objects (including this one).
                state.set_monitor_powered(&output_id, on);
            }
            Request::Destroy => {}
            _ => {}
        }
    }
}
