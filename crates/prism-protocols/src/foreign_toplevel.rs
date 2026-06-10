//! Foreign-toplevel window lists for taskbars / docks / status bars.
//!
//! Two globals, one tracking structure:
//!
//! - `ext_foreign_toplevel_list_v1` — the modern read-only window list
//!   (identifier + title + app_id).
//! - `zwlr_foreign_toplevel_manager_v1` (v3) — the wlr list with state
//!   (activated/maximized/fullscreen), per-output enter/leave, parent
//!   links (dialog grouping), and control requests (activate, close,
//!   set_fullscreen, set_maximized).
//!
//! Ported from niri `src/protocols/foreign_toplevel.rs`, restructured to
//! prism idiom: `Dispatch`/`GlobalDispatch` are implemented directly on
//! [`PrismState`] (no handler trait / delegate macros), there is no client
//! filter (prism has no security-context restriction concept yet), and
//! request handling drives the layout through the same helpers as the xdg
//! request paths.
//!
//! Data flow: [`refresh`] runs once per dispatch cycle (after
//! `update_keyboard_focus`) and diffs the layout's window set against the
//! protocol objects, emitting only deltas. The wlr `activated` state is
//! deliberately keyboard focus, not the xdg `Activated` set — see
//! [`to_state_vec`].

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use prism_layout::layout::ActivateWindow;
use prism_layout::utils::with_toplevel_role_and_current;
use prism_layout::window::mapped::MappedId;
use smithay::output::Output;
use smithay::reexports::wayland_protocols::ext::foreign_toplevel_list::v1::server::{
    ext_foreign_toplevel_handle_v1::{self, ExtForeignToplevelHandleV1},
    ext_foreign_toplevel_list_v1::{self, ExtForeignToplevelListV1},
};
use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel;
use smithay::reexports::wayland_protocols_wlr::foreign_toplevel::v1::server::{
    zwlr_foreign_toplevel_handle_v1::{self, ZwlrForeignToplevelHandleV1},
    zwlr_foreign_toplevel_manager_v1::{self, ZwlrForeignToplevelManagerV1},
};
use smithay::reexports::wayland_server::backend::ClientId;
use smithay::reexports::wayland_server::protocol::wl_output::WlOutput;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
};
use smithay::wayland::shell::xdg::{
    ToplevelState, ToplevelStateSet, XdgToplevelSurfaceRoleAttributes,
};

use crate::state::{queue_redraw_all, queue_redraw_for_surface, PrismState};

const EXT_LIST_VERSION: u32 = 1;
const WLR_MANAGEMENT_VERSION: u32 = 3;

pub struct ForeignToplevelManagerState {
    display: DisplayHandle,
    ext_list_instances: HashSet<ExtForeignToplevelListV1>,
    wlr_management_instances: HashSet<ZwlrForeignToplevelManagerV1>,
    toplevels: HashMap<WlSurface, ToplevelData>,
}

struct ToplevelData {
    identifier: MappedId,
    title: Option<String>,
    app_id: Option<String>,
    states: Vec<u32>,
    output: Option<Output>,

    ext_list_instances: HashSet<ExtForeignToplevelHandleV1>,
    /// Per wlr handle, the wl_outputs we sent `output_enter` for (so an
    /// output change can send the matching `output_leave`).
    wlr_management_instances: HashMap<ZwlrForeignToplevelHandleV1, Vec<WlOutput>>,
    /// The *effective* parent last sent on the wlr handles (v3 `parent`
    /// event): the xdg parent if that surface is itself a tracked
    /// toplevel, else None. Storing the effective value means a parent
    /// that maps (or closes) later flips it and re-emits. See
    /// [`refresh_parents`].
    parent: Option<WlSurface>,
}

impl ForeignToplevelManagerState {
    pub fn new(display: &DisplayHandle) -> Self {
        display.create_global::<PrismState, ExtForeignToplevelListV1, _>(EXT_LIST_VERSION, ());
        display.create_global::<PrismState, ZwlrForeignToplevelManagerV1, _>(
            WLR_MANAGEMENT_VERSION,
            (),
        );
        Self {
            display: display.clone(),
            ext_list_instances: HashSet::new(),
            wlr_management_instances: HashSet::new(),
            toplevels: HashMap::new(),
        }
    }
}

/// Diff the layout's window set against the protocol objects and emit
/// deltas. Called once per dispatch cycle, after `update_keyboard_focus`
/// (the wlr `activated` state mirrors keyboard focus).
pub fn refresh(state: &mut PrismState) {
    let protocol_state = &mut state.foreign_toplevel_state;

    // Handle closed windows.
    protocol_state.toplevels.retain(|surface, data| {
        if state.layout.find_window_and_output(surface).is_some() {
            return true;
        }

        for instance in data.ext_list_instances.iter() {
            instance.closed();
        }

        for instance in data.wlr_management_instances.keys() {
            instance.closed();
        }

        false
    });

    // Handle new and existing windows.
    //
    // Save the focused window for last, this way when the focus changes, we
    // will first deactivate the previous window and only then activate the
    // newly focused window.
    let mut focused = None;
    let mut parent_updates = Vec::new();
    state.layout.with_windows(|mapped, output, _, _| {
        let toplevel = mapped.toplevel();
        let wl_surface = toplevel.wl_surface();
        with_toplevel_role_and_current(toplevel, |role, cur| {
            let Some(cur) = cur else {
                tracing::error!("mapped must have had initial commit");
                return;
            };

            parent_updates.push((wl_surface.clone(), role.parent.clone()));

            if state.keyboard_focus.surface() == Some(wl_surface) {
                focused = Some((mapped.id(), mapped.window.clone(), output.cloned()));
            } else {
                refresh_toplevel(
                    protocol_state,
                    wl_surface,
                    mapped.id(),
                    role,
                    cur,
                    output,
                    false,
                );
            }
        });
    });

    // Finally, refresh the focused window.
    if let Some((identifier, window, output)) = focused {
        let toplevel = window.toplevel().expect("no X11 support");
        let wl_surface = toplevel.wl_surface();
        with_toplevel_role_and_current(toplevel, |role, cur| {
            let Some(cur) = cur else {
                tracing::error!("mapped must have had initial commit");
                return;
            };

            refresh_toplevel(
                protocol_state,
                wl_surface,
                identifier,
                role,
                cur,
                output.as_ref(),
                true,
            );
        });
    }

    // Parent links go last so that a parent mapped in this same cycle
    // already has its handles created.
    refresh_parents(protocol_state, parent_updates);
}

/// Emit wlr v3 `parent` deltas. The effective parent is the xdg parent
/// surface if it is itself a tracked toplevel, else null — so a parent
/// closing (or a child outliving an unmapped parent) re-emits null, and a
/// parent appearing re-emits the handle. Each child handle is paired with
/// the parent's handle for the same client (wlroots matches by client
/// too); if a client somehow has no handle for the parent, null is sent.
fn refresh_parents(
    protocol_state: &mut ForeignToplevelManagerState,
    updates: Vec<(WlSurface, Option<WlSurface>)>,
) {
    for (surface, raw_parent) in updates {
        let effective = raw_parent.filter(|p| protocol_state.toplevels.contains_key(p));
        let Some(data) = protocol_state.toplevels.get(&surface) else {
            continue;
        };
        if data.parent == effective {
            continue;
        }

        let parent_handles: Vec<ZwlrForeignToplevelHandleV1> = effective
            .as_ref()
            .map(|p| {
                protocol_state.toplevels[p]
                    .wlr_management_instances
                    .keys()
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        let events: Vec<_> = data
            .wlr_management_instances
            .keys()
            .filter(|i| i.version() >= zwlr_foreign_toplevel_handle_v1::EVT_PARENT_SINCE)
            .map(|i| {
                let parent = parent_handles
                    .iter()
                    .find(|p| p.client() == i.client())
                    .cloned();
                (i.clone(), parent)
            })
            .collect();
        for (instance, parent) in events {
            instance.parent(parent.as_ref());
            instance.done();
        }

        protocol_state.toplevels.get_mut(&surface).unwrap().parent = effective;
    }
}

/// A client bound a new wl_output: send `output_enter` to its wlr handles
/// for toplevels on that output. Called from `OutputHandler::output_bound`.
pub fn on_output_bound(state: &mut PrismState, output: &Output, wl_output: &WlOutput) {
    let Some(client) = wl_output.client() else {
        return;
    };

    let protocol_state = &mut state.foreign_toplevel_state;
    for data in protocol_state.toplevels.values_mut() {
        if data.output.as_ref() != Some(output) {
            continue;
        }

        for (instance, outputs) in &mut data.wlr_management_instances {
            if instance.client().as_ref() != Some(&client) {
                continue;
            }

            instance.output_enter(wl_output);
            instance.done();
            outputs.push(wl_output.clone());
        }
    }
}

fn refresh_toplevel(
    protocol_state: &mut ForeignToplevelManagerState,
    wl_surface: &WlSurface,
    identifier: MappedId,
    role: &XdgToplevelSurfaceRoleAttributes,
    current: &ToplevelState,
    output: Option<&Output>,
    has_focus: bool,
) {
    let states = to_state_vec(&current.states, has_focus);

    match protocol_state.toplevels.entry(wl_surface.clone()) {
        Entry::Occupied(entry) => {
            // Existing window, check if anything changed.
            let data = entry.into_mut();

            let mut new_title = None;
            if data.title != role.title {
                data.title.clone_from(&role.title);
                new_title = role.title.as_deref();

                if new_title.is_none() {
                    tracing::error!("toplevel title changed to None");
                }
            }

            let mut new_app_id = None;
            if data.app_id != role.app_id {
                data.app_id.clone_from(&role.app_id);
                new_app_id = role.app_id.as_deref();

                if new_app_id.is_none() {
                    tracing::error!("toplevel app_id changed to None");
                }
            }

            let mut states_changed = false;
            if data.states != states {
                data.states = states;
                states_changed = true;
            }

            let mut output_changed = false;
            if data.output.as_ref() != output {
                data.output = output.cloned();
                output_changed = true;
            }

            let something_changed_for_ext = new_title.is_some() || new_app_id.is_some();
            let something_changed_for_wlr =
                new_title.is_some() || new_app_id.is_some() || states_changed || output_changed;

            if something_changed_for_ext {
                for instance in &data.ext_list_instances {
                    if let Some(new_title) = new_title {
                        instance.title(new_title.to_owned());
                    }
                    if let Some(new_app_id) = new_app_id {
                        instance.app_id(new_app_id.to_owned());
                    }
                    instance.done();
                }
            }

            if something_changed_for_wlr {
                for (instance, outputs) in &mut data.wlr_management_instances {
                    if let Some(new_title) = new_title {
                        instance.title(new_title.to_owned());
                    }
                    if let Some(new_app_id) = new_app_id {
                        instance.app_id(new_app_id.to_owned());
                    }
                    if states_changed {
                        instance.state(data.states.iter().flat_map(|x| x.to_ne_bytes()).collect());
                    }
                    if output_changed {
                        for wl_output in outputs.drain(..) {
                            instance.output_leave(&wl_output);
                        }
                        if let Some(output) = &data.output {
                            if let Some(client) = instance.client() {
                                for wl_output in output.client_outputs(&client) {
                                    instance.output_enter(&wl_output);
                                    outputs.push(wl_output);
                                }
                            }
                        }
                    }
                    instance.done();
                }
            }

            for outputs in data.wlr_management_instances.values_mut() {
                // Clean up dead wl_outputs.
                outputs.retain(|x| x.is_alive());
            }
        }
        Entry::Vacant(entry) => {
            // New window, start tracking it.
            let mut data = ToplevelData {
                identifier,
                title: role.title.clone(),
                app_id: role.app_id.clone(),
                states,
                output: output.cloned(),
                ext_list_instances: HashSet::new(),
                wlr_management_instances: HashMap::new(),
                // The real parent (if any) is emitted by the parent pass
                // later in this same refresh cycle.
                parent: None,
            };

            for manager in &protocol_state.ext_list_instances {
                if let Some(client) = manager.client() {
                    data.add_ext_instance(&protocol_state.display, &client, manager);
                }
            }

            for manager in &protocol_state.wlr_management_instances {
                if let Some(client) = manager.client() {
                    data.add_wlr_instance(&protocol_state.display, &client, manager);
                }
            }

            entry.insert(data);
        }
    }
}

impl ToplevelData {
    fn add_ext_instance(
        &mut self,
        handle: &DisplayHandle,
        client: &Client,
        manager: &ExtForeignToplevelListV1,
    ) {
        let toplevel = client
            .create_resource::<ExtForeignToplevelHandleV1, _, PrismState>(
                handle,
                manager.version(),
                (),
            )
            .unwrap();
        manager.toplevel(&toplevel);

        toplevel.identifier(self.identifier.to_protocol_identifier());

        if let Some(title) = &self.title {
            toplevel.title(title.clone());
        }
        if let Some(app_id) = &self.app_id {
            toplevel.app_id(app_id.clone());
        }

        toplevel.done();

        self.ext_list_instances.insert(toplevel);
    }

    fn add_wlr_instance(
        &mut self,
        handle: &DisplayHandle,
        client: &Client,
        manager: &ZwlrForeignToplevelManagerV1,
    ) {
        let toplevel = client
            .create_resource::<ZwlrForeignToplevelHandleV1, _, PrismState>(
                handle,
                manager.version(),
                (),
            )
            .unwrap();
        manager.toplevel(&toplevel);

        if let Some(title) = &self.title {
            toplevel.title(title.clone());
        }
        if let Some(app_id) = &self.app_id {
            toplevel.app_id(app_id.clone());
        }

        toplevel.state(self.states.iter().flat_map(|x| x.to_ne_bytes()).collect());

        let mut outputs = Vec::new();
        if let Some(output) = &self.output {
            for wl_output in output.client_outputs(client) {
                toplevel.output_enter(&wl_output);
                outputs.push(wl_output);
            }
        }

        toplevel.done();

        self.wlr_management_instances.insert(toplevel, outputs);
    }
}

// ─── request handling: drive the layout ─────────────────────────────────────
//
// Same shape as the xdg_toplevel request paths in state.rs
// (set_window_fullscreen / set_window_maximized): resolve the surface to its
// layout window first (cloning out of the immutable borrow), then mutate.

fn activate(state: &mut PrismState, wl_surface: WlSurface) {
    let window = state
        .layout
        .find_window_and_output(&wl_surface)
        .map(|(mapped, _)| mapped.window.clone());
    if let Some(w) = window {
        state.layout.activate_window(&w);
        // Keyboard focus is derived from the layout per frame; the next
        // update_keyboard_focus() pass picks this up.
        queue_redraw_all(state);
    }
}

fn close(state: &mut PrismState, wl_surface: WlSurface) {
    if let Some((mapped, _)) = state.layout.find_window_and_output(&wl_surface) {
        mapped.toplevel().send_close();
    }
}

fn set_fullscreen(state: &mut PrismState, wl_surface: WlSurface, wl_output: Option<WlOutput>) {
    let found = state
        .layout
        .find_window_and_output(&wl_surface)
        .map(|(mapped, output)| (mapped.window.clone(), output.cloned()));
    let Some((window, current_output)) = found else {
        return;
    };

    // Filter the resolved output through the layout's monitor set: right
    // after an output global is disabled (but before the client drops its
    // wl_output) `Output::from_resource` still resolves an output the layout
    // has already forgotten, and `move_to_output` panics on unknown outputs.
    // niri guards the same race via `output_from_resource`/`output_exists`.
    let requested_output = wl_output
        .as_ref()
        .and_then(Output::from_resource)
        .filter(|requested| state.layout.outputs().any(|o| o == requested));
    if let Some(requested_output) = requested_output {
        if Some(&requested_output) != current_output.as_ref() {
            state.layout.move_to_output(
                Some(&window),
                &requested_output,
                None,
                ActivateWindow::Smart,
            );
        }
    }

    state.layout.set_fullscreen(&window, true);
    queue_redraw_for_surface(state, &wl_surface);
}

fn unset_fullscreen(state: &mut PrismState, wl_surface: WlSurface) {
    let window = state
        .layout
        .find_window_and_output(&wl_surface)
        .map(|(mapped, _)| mapped.window.clone());
    if let Some(w) = window {
        state.layout.set_fullscreen(&w, false);
        queue_redraw_for_surface(state, &wl_surface);
    }
}

fn set_maximized(state: &mut PrismState, wl_surface: WlSurface, maximize: bool) {
    let window = state
        .layout
        .find_window_and_output(&wl_surface)
        .map(|(mapped, _)| mapped.window.clone());
    if let Some(w) = window {
        state.layout.set_maximized(&w, maximize);
        queue_redraw_for_surface(state, &wl_surface);
    }
}

// ─── ext_foreign_toplevel_list_v1 ────────────────────────────────────────────

impl GlobalDispatch<ExtForeignToplevelListV1, ()> for PrismState {
    fn bind(
        state: &mut Self,
        handle: &DisplayHandle,
        client: &Client,
        resource: New<ExtForeignToplevelListV1>,
        _global_data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        let manager = data_init.init(resource, ());

        let protocol_state = &mut state.foreign_toplevel_state;
        for data in protocol_state.toplevels.values_mut() {
            data.add_ext_instance(handle, client, &manager);
        }

        protocol_state.ext_list_instances.insert(manager);
    }
}

impl Dispatch<ExtForeignToplevelListV1, ()> for PrismState {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &ExtForeignToplevelListV1,
        request: <ExtForeignToplevelListV1 as Resource>::Request,
        _data: &(),
        _dhandle: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            ext_foreign_toplevel_list_v1::Request::Stop => {
                resource.finished();

                // Remove the instance here so we won't send any more events.
                state
                    .foreign_toplevel_state
                    .ext_list_instances
                    .remove(resource);
            }
            ext_foreign_toplevel_list_v1::Request::Destroy => {}
            _ => unreachable!(),
        }
    }

    fn destroyed(
        state: &mut Self,
        _client: ClientId,
        resource: &ExtForeignToplevelListV1,
        _data: &(),
    ) {
        // Also remove the instance here, in case `stop` was never sent, e.g.
        // sudden disconnect.
        state
            .foreign_toplevel_state
            .ext_list_instances
            .remove(resource);
    }
}

impl Dispatch<ExtForeignToplevelHandleV1, ()> for PrismState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &ExtForeignToplevelHandleV1,
        request: <ExtForeignToplevelHandleV1 as Resource>::Request,
        _data: &(),
        _dhandle: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            ext_foreign_toplevel_handle_v1::Request::Destroy => {}
            _ => unreachable!(),
        }
    }

    fn destroyed(
        state: &mut Self,
        _client: ClientId,
        resource: &ExtForeignToplevelHandleV1,
        _data: &(),
    ) {
        for data in state.foreign_toplevel_state.toplevels.values_mut() {
            data.ext_list_instances.remove(resource);
        }
    }
}

// ─── zwlr_foreign_toplevel_manager_v1 ────────────────────────────────────────

impl GlobalDispatch<ZwlrForeignToplevelManagerV1, ()> for PrismState {
    fn bind(
        state: &mut Self,
        handle: &DisplayHandle,
        client: &Client,
        resource: New<ZwlrForeignToplevelManagerV1>,
        _global_data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        let manager = data_init.init(resource, ());

        let protocol_state = &mut state.foreign_toplevel_state;
        for data in protocol_state.toplevels.values_mut() {
            data.add_wlr_instance(handle, client, &manager);
        }

        // Parent links are sent after the creation loop so the parent
        // handles for this client exist regardless of map iteration
        // order. Re-sending to the client's pre-existing handles (manager
        // bound twice) is a harmless duplicate state event.
        send_parents_to_client(protocol_state, client);

        protocol_state.wlr_management_instances.insert(manager);
    }
}

/// Initial-state `parent` events for a freshly bound wlr manager: every
/// tracked toplevel with a (tracked) parent gets the link sent on this
/// client's v3 handles. Null parents are skipped — that's already the
/// protocol default for a new handle.
fn send_parents_to_client(protocol_state: &ForeignToplevelManagerState, client: &Client) {
    for data in protocol_state.toplevels.values() {
        let Some(parent) = &data.parent else {
            continue;
        };
        let Some(parent_data) = protocol_state.toplevels.get(parent) else {
            continue;
        };
        let parent_handle = parent_data
            .wlr_management_instances
            .keys()
            .find(|p| p.client().as_ref() == Some(client));
        let Some(parent_handle) = parent_handle else {
            continue;
        };

        for instance in data.wlr_management_instances.keys() {
            if instance.client().as_ref() != Some(client) {
                continue;
            }
            if instance.version() < zwlr_foreign_toplevel_handle_v1::EVT_PARENT_SINCE {
                continue;
            }
            instance.parent(Some(parent_handle));
            instance.done();
        }
    }
}

impl Dispatch<ZwlrForeignToplevelManagerV1, ()> for PrismState {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &ZwlrForeignToplevelManagerV1,
        request: <ZwlrForeignToplevelManagerV1 as Resource>::Request,
        _data: &(),
        _dhandle: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            zwlr_foreign_toplevel_manager_v1::Request::Stop => {
                resource.finished();

                // Remove the instance here so we won't send any more events.
                state
                    .foreign_toplevel_state
                    .wlr_management_instances
                    .remove(resource);
            }
            _ => unreachable!(),
        }
    }

    fn destroyed(
        state: &mut Self,
        _client: ClientId,
        resource: &ZwlrForeignToplevelManagerV1,
        _data: &(),
    ) {
        // Also remove the instance here, in case `stop` was never sent, e.g.
        // sudden disconnect.
        state
            .foreign_toplevel_state
            .wlr_management_instances
            .remove(resource);
    }
}

impl Dispatch<ZwlrForeignToplevelHandleV1, ()> for PrismState {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &ZwlrForeignToplevelHandleV1,
        request: <ZwlrForeignToplevelHandleV1 as Resource>::Request,
        _data: &(),
        _dhandle: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        let protocol_state = &state.foreign_toplevel_state;

        let Some((surface, _)) = protocol_state
            .toplevels
            .iter()
            .find(|(_, data)| data.wlr_management_instances.contains_key(resource))
        else {
            return;
        };
        let surface = surface.clone();

        match request {
            zwlr_foreign_toplevel_handle_v1::Request::SetMaximized => {
                set_maximized(state, surface, true);
            }
            zwlr_foreign_toplevel_handle_v1::Request::UnsetMaximized => {
                set_maximized(state, surface, false);
            }
            // No minimized concept in the layout.
            zwlr_foreign_toplevel_handle_v1::Request::SetMinimized => (),
            zwlr_foreign_toplevel_handle_v1::Request::UnsetMinimized => (),
            zwlr_foreign_toplevel_handle_v1::Request::Activate { .. } => {
                activate(state, surface);
            }
            zwlr_foreign_toplevel_handle_v1::Request::Close => {
                close(state, surface);
            }
            zwlr_foreign_toplevel_handle_v1::Request::SetRectangle { .. } => (),
            zwlr_foreign_toplevel_handle_v1::Request::Destroy => (),
            zwlr_foreign_toplevel_handle_v1::Request::SetFullscreen { output } => {
                set_fullscreen(state, surface, output);
            }
            zwlr_foreign_toplevel_handle_v1::Request::UnsetFullscreen => {
                unset_fullscreen(state, surface);
            }
            _ => unreachable!(),
        }
    }

    fn destroyed(
        state: &mut Self,
        _client: ClientId,
        resource: &ZwlrForeignToplevelHandleV1,
        _data: &(),
    ) {
        for data in state.foreign_toplevel_state.toplevels.values_mut() {
            data.wlr_management_instances.remove(resource);
        }
    }
}

fn to_state_vec(states: &ToplevelStateSet, has_focus: bool) -> Vec<u32> {
    let mut rv = Vec::with_capacity(3);
    if states.contains(xdg_toplevel::State::Maximized) {
        rv.push(zwlr_foreign_toplevel_handle_v1::State::Maximized as u32);
    }
    if states.contains(xdg_toplevel::State::Fullscreen) {
        rv.push(zwlr_foreign_toplevel_handle_v1::State::Fullscreen as u32);
    }

    // HACK: wlr-foreign-toplevel-management states:
    //
    // These have the same meaning as the states with the same names defined in xdg-toplevel
    //
    // However, clients such as sfwbar and fcitx seem to treat the activated state as keyboard
    // focus, i.e. they don't expect multiple windows to have it set at once. Even Waybar which
    // handles multiple activated windows correctly uses it in its design in such a way that
    // keyboard focus would make more sense. Let's do what the clients expect.
    if has_focus {
        rv.push(zwlr_foreign_toplevel_handle_v1::State::Activated as u32);
    }

    rv
}
