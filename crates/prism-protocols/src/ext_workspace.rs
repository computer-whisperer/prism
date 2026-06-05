//! ext-workspace-v1 — workspace observation/control for status bars.
//!
//! Mapping of protocol concepts (same as niri's):
//!
//! - Workspace groups are outputs.
//! - Workspace coordinates: X = 0, Y = workspace index. They need to be
//!   two-dimensional because 1D coordinates are defined to be a plain list
//!   without a geometric interpretation, while we do order workspaces in a
//!   vertical line.
//! - Workspace id: name for named workspaces, unset for unnamed. Because ids
//!   in this protocol are expected to be stable across sessions.
//! - Workspace name: name for named workspaces, index for unnamed.
//!
//! Ported from niri `src/protocols/ext_workspace.rs`, restructured to prism
//! idiom: `Dispatch`/`GlobalDispatch` are implemented directly on
//! [`PrismState`] (no handler trait / delegate macros) and there is no
//! client filter. [`refresh`] runs once per dispatch cycle and diffs the
//! layout's workspace set against the protocol objects.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::mem;

use prism_layout::layout::monitor::Monitor;
use prism_layout::layout::workspace::Workspace;
use prism_layout::layout::WorkspaceId;
use prism_layout::window::Mapped;
use smithay::output::{Output, WeakOutput};
use smithay::reexports::wayland_protocols::ext::workspace::v1::server::{
    ext_workspace_group_handle_v1::{self, ExtWorkspaceGroupHandleV1},
    ext_workspace_handle_v1::{self, ExtWorkspaceHandleV1},
    ext_workspace_manager_v1::{self, ExtWorkspaceManagerV1},
};
use smithay::reexports::wayland_server::backend::ClientId;
use smithay::reexports::wayland_server::protocol::wl_output::WlOutput;
use smithay::reexports::wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
};

use crate::state::{queue_redraw_all, PrismState};

const VERSION: u32 = 1;

/// Requests on workspace handles are buffered per manager instance and
/// applied atomically on `commit`.
enum Action {
    Assign(WorkspaceId, WeakOutput),
    Activate(WorkspaceId),
}

impl Action {
    fn order(&self) -> u8 {
        // First assign everything (move across outputs), then activate.
        match self {
            Action::Assign(_, _) => 0,
            Action::Activate(_) => 1,
        }
    }
}

pub struct ExtWorkspaceManagerState {
    display: DisplayHandle,
    instances: HashMap<ExtWorkspaceManagerV1, Vec<Action>>,
    workspace_groups: HashMap<Output, ExtWorkspaceGroupData>,
    workspaces: HashMap<WorkspaceId, ExtWorkspaceData>,
}

struct ExtWorkspaceGroupData {
    instances: Vec<ExtWorkspaceGroupHandleV1>,
}

struct ExtWorkspaceData {
    // id cannot change once set.
    id: Option<String>,
    name: String,
    coordinates: [u32; 2],
    state: ext_workspace_handle_v1::State,
    instances: Vec<ExtWorkspaceHandleV1>,
    output: Option<Output>,
}

impl ExtWorkspaceManagerState {
    pub fn new(display: &DisplayHandle) -> Self {
        display.create_global::<PrismState, ExtWorkspaceManagerV1, _>(VERSION, ());
        Self {
            display: display.clone(),
            instances: HashMap::new(),
            workspace_groups: HashMap::new(),
            workspaces: HashMap::new(),
        }
    }
}

/// Diff the layout's workspace set against the protocol objects and emit
/// deltas. Called once per dispatch cycle.
pub fn refresh(state: &mut PrismState) {
    let protocol_state = &mut state.ext_workspace_state;

    let mut changed = false;

    // Remove workspaces that no longer exist (sending workspace_leave to
    // workspace groups).
    let mut seen_workspaces = HashMap::new();
    for (mon, _, ws) in state.layout.workspaces() {
        let output = mon.map(|mon| mon.output());
        seen_workspaces.insert(ws.id(), output);
    }

    protocol_state.workspaces.retain(|id, workspace| {
        if seen_workspaces.contains_key(id) {
            return true;
        }

        remove_workspace_instances(&protocol_state.workspace_groups, workspace);
        changed = true;
        false
    });

    // Remove workspace groups for outputs that no longer exist.
    let layout_outputs: Vec<Output> = state.layout.outputs().cloned().collect();
    protocol_state.workspace_groups.retain(|output, data| {
        if layout_outputs.contains(output) {
            return true;
        }

        for group in &data.instances {
            // Send workspace_leave for all workspaces in this group with
            // matching manager.
            let manager: &ExtWorkspaceManagerV1 = group.data().unwrap();
            for ws in protocol_state.workspaces.values() {
                if ws.output.as_ref() == Some(output) {
                    for workspace in &ws.instances {
                        if workspace.data() == Some(manager) {
                            group.workspace_leave(workspace);
                        }
                    }
                }
            }

            group.removed();
        }

        changed = true;
        false
    });

    // Update existing workspaces and create new ones.
    for (mon, ws_idx, ws) in state.layout.workspaces() {
        changed |= refresh_workspace(protocol_state, mon, ws_idx, ws);
    }

    // Update workspace groups and create new ones, sending workspace_enter
    // events as needed.
    for output in &layout_outputs {
        changed |= refresh_workspace_group(protocol_state, output);
    }

    if changed {
        for manager in protocol_state.instances.keys() {
            manager.done();
        }
    }
}

/// A client bound a new wl_output: send `output_enter` to its group handles
/// for that output. Called from `OutputHandler::output_bound`.
pub fn on_output_bound(state: &mut PrismState, output: &Output, wl_output: &WlOutput) {
    let Some(client) = wl_output.client() else {
        return;
    };

    let mut sent = false;

    let protocol_state = &mut state.ext_workspace_state;
    if let Some(data) = protocol_state.workspace_groups.get_mut(output) {
        for group in &mut data.instances {
            if group.client().as_ref() != Some(&client) {
                continue;
            }

            group.output_enter(wl_output);
            sent = true;
        }
    }

    if !sent {
        return;
    }

    for manager in protocol_state.instances.keys() {
        if manager.client().as_ref() == Some(&client) {
            manager.done();
        }
    }
}

fn refresh_workspace_group(protocol_state: &mut ExtWorkspaceManagerState, output: &Output) -> bool {
    if protocol_state.workspace_groups.contains_key(output) {
        // Existing workspace group. Nothing can actually change since our
        // workspace groups are tied to an output.
        return false;
    }

    // New workspace group, start tracking it.
    let mut data = ExtWorkspaceGroupData {
        instances: Vec::new(),
    };

    // Create workspace group handle for each manager instance.
    for manager in protocol_state.instances.keys() {
        if let Some(client) = manager.client() {
            data.add_instance(&protocol_state.display, &client, manager, output);
        }
    }

    // Send workspace_enter for all existing workspaces on this output.
    for group in &data.instances {
        let manager: &ExtWorkspaceManagerV1 = group.data().unwrap();
        for (_, ws) in protocol_state.workspaces.iter() {
            if ws.output.as_ref() != Some(output) {
                continue;
            }
            for workspace in &ws.instances {
                if workspace.data() == Some(manager) {
                    group.workspace_enter(workspace);
                }
            }
        }
    }

    protocol_state.workspace_groups.insert(output.clone(), data);
    true
}

fn send_workspace_enter_leave(
    workspace_groups: &HashMap<Output, ExtWorkspaceGroupData>,
    data: &ExtWorkspaceData,
    enter: bool,
) {
    if let Some(output) = &data.output {
        if let Some(group_data) = workspace_groups.get(output) {
            for group in &group_data.instances {
                let manager: &ExtWorkspaceManagerV1 = group.data().unwrap();
                for workspace in &data.instances {
                    if workspace.data() == Some(manager) {
                        if enter {
                            group.workspace_enter(workspace);
                        } else {
                            group.workspace_leave(workspace);
                        }
                    }
                }
            }
        }
    }
}

fn remove_workspace_instances(
    workspace_groups: &HashMap<Output, ExtWorkspaceGroupData>,
    data: &ExtWorkspaceData,
) {
    send_workspace_enter_leave(workspace_groups, data, false);

    for workspace in &data.instances {
        workspace.removed();
    }
}

fn build_name(ws: &Workspace<Mapped>, ws_idx: usize) -> String {
    ws.name().cloned().unwrap_or_else(|| {
        // Add 1 since this is a human-readable name, and our action indexing
        // is 1-based.
        (ws_idx + 1).to_string()
    })
}

fn refresh_workspace(
    protocol_state: &mut ExtWorkspaceManagerState,
    mon: Option<&Monitor<Mapped>>,
    ws_idx: usize,
    ws: &Workspace<Mapped>,
) -> bool {
    let mut state = ext_workspace_handle_v1::State::empty();
    if mon.is_some_and(|mon| mon.active_workspace_idx() == ws_idx) {
        state |= ext_workspace_handle_v1::State::Active;
    }
    if ws.is_urgent() {
        state |= ext_workspace_handle_v1::State::Urgent;
    }

    let output = mon.map(|mon| mon.output());

    match protocol_state.workspaces.entry(ws.id()) {
        Entry::Occupied(entry) => {
            // Existing workspace, check if anything changed.
            let data = entry.into_mut();

            let mut id_set = false;
            let mut recreate = false;
            let id = ws.name();
            if data.id.as_ref() != id {
                if data.id.is_some() {
                    recreate = true;
                } else {
                    id_set = true;
                }
                data.id = id.cloned();
            }

            let mut coordinates_changed = false;
            if data.coordinates[1] != ws_idx as u32 {
                data.coordinates[1] = ws_idx as u32;
                coordinates_changed = true;
            }

            let mut state_changed = false;
            if data.state != state {
                data.state = state;
                state_changed = true;
            }

            // Recreate means name got changed or unset (meaning data.name is
            // back to ws_idx).
            let check = recreate
                || if data.id.is_some() {
                    // True means workspace got named, going from ws_idx to name.
                    id_set
                } else {
                    // The workspace is unnamed, check if ws_idx changed.
                    coordinates_changed
                };
            let mut name_changed = false;
            if check {
                let new_name = build_name(ws, ws_idx);
                // This will likely be true, except if the workspace got named
                // its index.
                if data.name != new_name {
                    data.name = new_name;
                    name_changed = true;
                }
            }

            let mut output_changed = false;
            if data.output.as_ref() != output {
                send_workspace_enter_leave(&protocol_state.workspace_groups, data, false);
                data.output = output.cloned();
                output_changed = true;
            }

            if recreate {
                remove_workspace_instances(&protocol_state.workspace_groups, data);
                data.instances.clear();

                for manager in protocol_state.instances.keys() {
                    if let Some(client) = manager.client() {
                        data.add_instance(&protocol_state.display, &client, manager);
                    }
                }

                send_workspace_enter_leave(&protocol_state.workspace_groups, data, true);
                return true;
            }

            if output_changed {
                // Send workspace_enter to the new output's group. If the
                // group doesn't exist yet (new groups are created after
                // refreshing workspaces), then workspace_enter() will be sent
                // when the group is created.
                send_workspace_enter_leave(&protocol_state.workspace_groups, data, true);
            }

            let something_changed = id_set || name_changed || coordinates_changed || state_changed;
            if something_changed {
                for instance in &data.instances {
                    if id_set {
                        instance.id(data.id.clone().unwrap());
                    }
                    if name_changed {
                        instance.name(data.name.clone());
                    }
                    if coordinates_changed {
                        instance.coordinates(
                            data.coordinates
                                .iter()
                                .flat_map(|x| x.to_ne_bytes())
                                .collect(),
                        );
                    }
                    if state_changed {
                        instance.state(data.state);
                    }
                }
            }

            output_changed || something_changed
        }
        Entry::Vacant(entry) => {
            // New workspace, start tracking it.
            let mut data = ExtWorkspaceData {
                id: ws.name().cloned(),
                name: build_name(ws, ws_idx),
                coordinates: [0, ws_idx as u32],
                state,
                instances: Vec::new(),
                output: output.cloned(),
            };

            for manager in protocol_state.instances.keys() {
                if let Some(client) = manager.client() {
                    data.add_instance(&protocol_state.display, &client, manager);
                }
            }

            send_workspace_enter_leave(&protocol_state.workspace_groups, &data, true);
            entry.insert(data);
            true
        }
    }
}

impl ExtWorkspaceGroupData {
    fn add_instance(
        &mut self,
        handle: &DisplayHandle,
        client: &Client,
        manager: &ExtWorkspaceManagerV1,
        output: &Output,
    ) -> &ExtWorkspaceGroupHandleV1 {
        let group = client
            .create_resource::<ExtWorkspaceGroupHandleV1, _, PrismState>(
                handle,
                manager.version(),
                manager.clone(),
            )
            .unwrap();
        manager.workspace_group(&group);

        group.capabilities(ext_workspace_group_handle_v1::GroupCapabilities::empty());

        for wl_output in output.client_outputs(client) {
            group.output_enter(&wl_output);
        }

        self.instances.push(group);
        self.instances.last().unwrap()
    }
}

impl ExtWorkspaceData {
    fn add_instance(
        &mut self,
        handle: &DisplayHandle,
        client: &Client,
        manager: &ExtWorkspaceManagerV1,
    ) -> &ExtWorkspaceHandleV1 {
        let workspace = client
            .create_resource::<ExtWorkspaceHandleV1, _, PrismState>(
                handle,
                manager.version(),
                manager.clone(),
            )
            .unwrap();
        manager.workspace(&workspace);

        if let Some(id) = self.id.clone() {
            workspace.id(id);
        }

        workspace.name(self.name.clone());
        workspace.coordinates(
            self.coordinates
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect(),
        );
        workspace.state(self.state);
        workspace.capabilities(
            ext_workspace_handle_v1::WorkspaceCapabilities::Activate
                | ext_workspace_handle_v1::WorkspaceCapabilities::Assign,
        );

        self.instances.push(workspace);
        self.instances.last().unwrap()
    }
}

// ─── request handling: drive the layout ─────────────────────────────────────

fn activate_workspace(state: &mut PrismState, id: WorkspaceId) {
    // Resolve the workspace id to its (output, index) in the layout. Mirrors
    // niri's find_output_and_workspace_index for the by-id case.
    let mut found = None;
    for (mon, ws_idx, ws) in state.layout.workspaces() {
        if ws.id() == id {
            found = Some((mon.map(|mon| mon.output().clone()), ws_idx));
            break;
        }
    }
    let Some((mut output, ws_idx)) = found else {
        return;
    };

    // Switching on the already-active output must not re-focus it (that
    // could pull focus from a more recently focused floating window etc.).
    if let Some(active) = state.layout.active_output() {
        if output.as_ref() == Some(active) {
            output = None;
        }
    }

    if let Some(output) = output {
        state.layout.focus_output(&output);
    }
    state.layout.switch_workspace(ws_idx);
    // No mouse warp: assuming the layer-shell bar workspaces use-case.

    queue_redraw_all(state);
}

fn assign_workspace(state: &mut PrismState, ws_id: WorkspaceId, output: Output) {
    let mut found = None;
    for (mon, ws_idx, ws) in state.layout.workspaces() {
        if ws.id() == ws_id {
            found = Some((mon.map(|mon| mon.output().clone()), ws_idx));
            break;
        }
    }
    let Some((old_output, old_idx)) = found else {
        return;
    };

    state
        .layout
        .move_workspace_to_output_by_id(old_idx, old_output, &output);
    queue_redraw_all(state);
}

// ─── ext_workspace_manager_v1 ────────────────────────────────────────────────

impl GlobalDispatch<ExtWorkspaceManagerV1, ()> for PrismState {
    fn bind(
        state: &mut Self,
        handle: &DisplayHandle,
        client: &Client,
        resource: New<ExtWorkspaceManagerV1>,
        _global_data: &(),
        data_init: &mut DataInit<'_, Self>,
    ) {
        let manager = data_init.init(resource, ());

        let protocol_state = &mut state.ext_workspace_state;

        // Send existing workspaces to the new client.
        let mut new_workspaces: HashMap<_, Vec<_>> = HashMap::new();
        for data in protocol_state.workspaces.values_mut() {
            let output = data.output.clone();
            let workspace = data.add_instance(handle, client, &manager);

            if let Some(output) = output {
                new_workspaces.entry(output).or_default().push(workspace);
            }
        }

        // Create workspace groups for all outputs.
        for (output, group_data) in &mut protocol_state.workspace_groups {
            let group = group_data.add_instance(handle, client, &manager, output);

            for workspace in new_workspaces.get(output).into_iter().flatten() {
                group.workspace_enter(workspace);
            }
        }

        manager.done();
        protocol_state.instances.insert(manager, Vec::new());
    }
}

impl Dispatch<ExtWorkspaceManagerV1, ()> for PrismState {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &ExtWorkspaceManagerV1,
        request: <ExtWorkspaceManagerV1 as Resource>::Request,
        _data: &(),
        _dhandle: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            ext_workspace_manager_v1::Request::Commit => {
                let protocol_state = &mut state.ext_workspace_state;
                let actions = protocol_state.instances.get_mut(resource).unwrap();
                let mut actions = mem::take(actions);

                actions.sort_by_key(Action::order);

                for action in actions {
                    match action {
                        Action::Assign(ws_id, output) => {
                            if let Some(output) = output.upgrade() {
                                assign_workspace(state, ws_id, output);
                            }
                        }
                        Action::Activate(id) => activate_workspace(state, id),
                    }
                }
            }
            ext_workspace_manager_v1::Request::Stop => {
                resource.finished();

                let protocol_state = &mut state.ext_workspace_state;
                protocol_state.instances.retain(|x, _| x != resource);

                for data in protocol_state.workspace_groups.values_mut() {
                    data.instances
                        .retain(|instance| instance.data() != Some(resource));
                }

                for data in protocol_state.workspaces.values_mut() {
                    data.instances
                        .retain(|instance| instance.data() != Some(resource));
                }
            }
            _ => unreachable!(),
        }
    }

    fn destroyed(
        state: &mut Self,
        _client: ClientId,
        resource: &ExtWorkspaceManagerV1,
        _data: &(),
    ) {
        state
            .ext_workspace_state
            .instances
            .retain(|x, _| x != resource);
    }
}

impl Dispatch<ExtWorkspaceHandleV1, ExtWorkspaceManagerV1> for PrismState {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &ExtWorkspaceHandleV1,
        request: <ExtWorkspaceHandleV1 as Resource>::Request,
        data: &ExtWorkspaceManagerV1,
        _dhandle: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        let protocol_state = &mut state.ext_workspace_state;

        let Some((workspace, _)) = protocol_state
            .workspaces
            .iter()
            .find(|(_, data)| data.instances.contains(resource))
        else {
            return;
        };
        let workspace = *workspace;

        match request {
            ext_workspace_handle_v1::Request::Activate => {
                let actions = protocol_state.instances.get_mut(data).unwrap();
                actions.push(Action::Activate(workspace));
            }
            ext_workspace_handle_v1::Request::Deactivate => (),
            ext_workspace_handle_v1::Request::Assign { workspace_group } => {
                if let Some(output) = protocol_state
                    .workspace_groups
                    .iter()
                    .find(|(_, data)| data.instances.contains(&workspace_group))
                    .map(|(output, _)| output.clone())
                {
                    let actions = protocol_state.instances.get_mut(data).unwrap();
                    actions.push(Action::Assign(workspace, output.downgrade()));
                }
            }
            ext_workspace_handle_v1::Request::Remove => (),
            ext_workspace_handle_v1::Request::Destroy => (),
            _ => unreachable!(),
        }
    }

    fn destroyed(
        state: &mut Self,
        _client: ClientId,
        resource: &ExtWorkspaceHandleV1,
        _data: &ExtWorkspaceManagerV1,
    ) {
        for data in state.ext_workspace_state.workspaces.values_mut() {
            data.instances.retain(|instance| instance != resource);
        }
    }
}

impl Dispatch<ExtWorkspaceGroupHandleV1, ExtWorkspaceManagerV1> for PrismState {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &ExtWorkspaceGroupHandleV1,
        request: <ExtWorkspaceGroupHandleV1 as Resource>::Request,
        _data: &ExtWorkspaceManagerV1,
        _dhandle: &DisplayHandle,
        _data_init: &mut DataInit<'_, Self>,
    ) {
        match request {
            // We don't advertise the CreateWorkspace capability.
            ext_workspace_group_handle_v1::Request::CreateWorkspace { .. } => (),
            ext_workspace_group_handle_v1::Request::Destroy => (),
            _ => unreachable!(),
        }
    }

    fn destroyed(
        state: &mut Self,
        _client: ClientId,
        resource: &ExtWorkspaceGroupHandleV1,
        _data: &ExtWorkspaceManagerV1,
    ) {
        for data in state.ext_workspace_state.workspace_groups.values_mut() {
            data.instances.retain(|instance| instance != resource);
        }
    }
}
