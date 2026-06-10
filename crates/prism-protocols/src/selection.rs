//! Clipboard + primary selection + drag-and-drop wiring.
//!
//! Backs five protocols:
//!   - `wl_data_device_manager` (v3) — standard clipboard + DnD.
//!   - `wp_primary_selection_device_manager_v1` — middle-click paste.
//!   - `zwlr_data_control_manager_v1` + `ext_data_control_manager_v1` —
//!     clipboard managers (cliphist, `wl-paste --watch`, clipman).
//!   - DnD grab handling (smithay's `input::dnd`) — pointer + touch.
//!
//! ## Why this exists at all
//!
//! GTK4 ≥ 4.22 *hard-requires* `wl_data_device_manager` as one of its
//! mandatory wayland globals. Without it, GTK's wayland backend
//! prints "does not provide one or more of the required interfaces"
//! and refuses the display, falling back to X11 — which on a TTY
//! session means the client just dies. Firefox, Nautilus, and every
//! other GTK4 app on the system were silently failing this check
//! before this module landed.
//!
//! ## How the pieces fit
//!
//! Clipboard ownership follows keyboard focus: the
//! [`smithay::input::SeatHandler::focus_changed`] callback in
//! `state.rs` calls [`set_data_device_focus`] +
//! [`set_primary_focus`], which tells smithay which client gets
//! the per-seat data offers. Reads and writes go directly
//! client-to-client over pipes — the compositor only sees them when
//! it sets a *server-side* selection (e.g. `set_data_device_selection`
//! after a screenshot copies its result). We don't do that today, so
//! [`SelectionHandler::send_selection`] is a defensive no-op
//! placeholder pattern.
//!
//! DnD is handled by smithay's [`DnDGrab`]: when a client calls
//! `wl_data_device.start_drag`, [`WaylandDndGrabHandler::dnd_requested`]
//! fires, we stash the icon surface for the render path, and
//! construct + install a pointer (or touch) grab that drives
//! enter/leave/motion/drop events to potential drop targets. While the
//! grab is live, the render walk draws the icon surface at the cursor
//! position on the output under the pointer (`render_output_now`), the
//! commit handler accumulates the icon's `wl_surface.offset` deltas and
//! queues repaints, and `send_frame_callbacks` keeps the icon's frame
//! callbacks fed. When the grab ends, [`DndGrabHandler::dropped`]
//! activates the drop-target window (or the output under the drop) and
//! clears the icon; `cancelled` just clears.
//!
//! ## Deferred — see TODO comments for details
//!
//!   - **Icon `wl_surface.enter`/`leave`.** The icon surface is never
//!     sent output enter/leave, so fractional-scale clients can't pick
//!     a preferred scale for it (niri dispatches these from its
//!     render-walk visibility tracking).
//!   - **Compositor-set selections.** [`SelectionHandler`] requires
//!     us to declare a `SelectionUserData` type for clipboard data
//!     the *compositor* writes (e.g. after a screenshot). We use
//!     `Arc<[u8]>` to match niri's shape, but `send_selection` is
//!     unreachable today because nothing calls
//!     `set_data_device_selection` from our side.
//!   - **Primary selection filtering.** niri lets per-client config
//!     suppress primary selection (sandbox isolation). We advertise
//!     to everyone; revisit when prism gains an analogous policy
//!     hook.

use std::fs::File;
use std::io::Write;
use std::os::fd::{AsRawFd, OwnedFd};
use std::sync::Arc;
use std::thread;

use smithay::input::dnd::{self, DnDGrab, DndGrabHandler, DndTarget};
use smithay::input::pointer::Focus;
use smithay::input::Seat;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{Logical, Point};
use smithay::wayland::selection::data_device::{
    DataDeviceHandler, DataDeviceState, WaylandDndGrabHandler,
};
use smithay::wayland::selection::ext_data_control::{
    DataControlHandler as ExtDataControlHandler, DataControlState as ExtDataControlState,
};
use smithay::wayland::selection::primary_selection::{
    PrimarySelectionHandler, PrimarySelectionState,
};
use smithay::wayland::selection::wlr_data_control::{
    DataControlHandler as WlrDataControlHandler, DataControlState as WlrDataControlState,
};
use smithay::wayland::selection::SelectionHandler;
use smithay::wayland::selection::SelectionTarget;

use crate::state::PrismState;

/// DnD cursor icon stashed for the duration of a drag. Lives in
/// [`PrismState::dnd_icon`] from [`WaylandDndGrabHandler::dnd_requested`]
/// (drag start) until [`DndGrabHandler::dropped`] or `cancelled`.
///
/// The render walk draws the icon surface tree at
/// `pointer_pos + offset`, topmost, on the output under the pointer.
#[derive(Debug)]
pub struct DndIcon {
    /// The wl_surface the client wants drawn under the cursor while
    /// the drag is active. May have its own subsurfaces and damage
    /// events like any other surface.
    pub surface: WlSurface,
    /// Offset of the icon's buffer top-left relative to the cursor
    /// position. Starts at zero and accumulates the dnd_icon role's
    /// `wl_surface.offset` deltas as the client commits (the commit
    /// handler in `state.rs` applies them), matching niri.
    pub offset: Point<i32, Logical>,
}

// ─── SelectionHandler ───────────────────────────────────────────────────────

impl SelectionHandler for PrismState {
    // Matches niri's choice: an Arc<[u8]> is the natural container
    // for compositor-set clipboard payloads (screenshot bytes,
    // synthesized text). Cheap to clone into the send-thread.
    type SelectionUserData = Arc<[u8]>;

    fn send_selection(
        &mut self,
        _ty: SelectionTarget,
        _mime_type: String,
        fd: OwnedFd,
        _seat: Seat<Self>,
        user_data: &Self::SelectionUserData,
    ) {
        // Spawn a thread so the wayland dispatch loop never blocks
        // on a slow reader. Clients open the fd with O_NONBLOCK by
        // default; clear it so write_all doesn't hit EAGAIN
        // half-way through a large payload.
        let buf = user_data.clone();
        thread::spawn(move || {
            // SAFETY: fcntl(F_SETFL, 0) on an owned fd is sound; we
            // hold OwnedFd and use its raw descriptor.
            let rc = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFL, 0) };
            if rc < 0 {
                tracing::warn!(
                    "selection: clearing O_NONBLOCK failed: {}",
                    std::io::Error::last_os_error()
                );
            }
            if let Err(err) = File::from(fd).write_all(&buf) {
                tracing::warn!("selection: writing payload to target fd: {err:?}");
            }
        });
    }
}

// ─── DataDeviceHandler + DnD grab ───────────────────────────────────────────

impl DataDeviceHandler for PrismState {
    fn data_device_state(&mut self) -> &mut DataDeviceState {
        &mut self.data_device_state
    }
}

impl WaylandDndGrabHandler for PrismState {
    fn dnd_requested<S: dnd::Source>(
        &mut self,
        source: S,
        icon: Option<WlSurface>,
        seat: Seat<Self>,
        serial: smithay::utils::Serial,
        type_: dnd::GrabType,
    ) {
        self.dnd_icon = icon.map(|surface| DndIcon {
            surface,
            offset: Point::from((0, 0)),
        });
        let icon_present = self.dnd_icon.is_some();

        // The seat is guaranteed to have the corresponding device — smithay
        // validates this before reaching the handler.
        match type_ {
            dnd::GrabType::Pointer => {
                let pointer = seat
                    .get_pointer()
                    .expect("dnd grab dispatched without pointer on seat");
                let start_data = pointer
                    .grab_start_data()
                    .expect("dnd_requested without a pointer grab in flight");
                let grab = DnDGrab::new_pointer(&self.display_handle, start_data, source, seat);
                pointer.set_grab(self, grab, serial, Focus::Keep);
            }
            dnd::GrabType::Touch => {
                let touch = seat
                    .get_touch()
                    .expect("dnd grab dispatched without touch on seat");
                let start_data = touch
                    .grab_start_data()
                    .expect("dnd_requested without a touch grab in flight");
                let grab = DnDGrab::new_touch(&self.display_handle, start_data, source, seat);
                touch.set_grab(self, grab, serial);
            }
        }

        // Show the icon right away — if it committed its buffer before
        // start_drag, no commit will queue the repaint for us. After
        // this, redraws are sustained by pointer motion
        // (`maybe_dnd_update`) and by icon commits (the dnd-icon branch
        // in the commit handler), whichever happens.
        if icon_present {
            let ids: Vec<_> = self.outputs.keys().cloned().collect();
            for id in ids {
                self.output_redraw.entry(id).or_default().queue_redraw();
            }
        }
    }
}

impl DndGrabHandler for PrismState {
    fn dropped(
        &mut self,
        target: Option<DndTarget<'_, Self>>,
        validated: bool,
        _seat: Seat<Self>,
        location: Point<f64, Logical>,
    ) {
        let target: Option<&WlSurface> = target.map(DndTarget::into_inner);
        tracing::trace!("dnd dropped, target: {target:?}, validated: {validated}");

        // End DnD before activating a specific window below so that the
        // activation takes precedence (niri handlers/mod.rs:363).
        self.on_dnd_ended();

        // Activate the target output — that's how Firefox's
        // drag-tab-into-new-window works, for example. On a successful
        // drop, additionally activate the target window.
        let mut activate_output = true;
        if let Some(target) = validated.then_some(target).flatten() {
            let root = self.find_root_shell_surface(target);
            let window = self
                .layout
                .find_window_and_output(&root)
                .map(|(mapped, _)| mapped.window.clone());
            if let Some(window) = window {
                self.layout.activate_window(&window);
                // Drop any transient on-demand layer focus so the
                // keyboard reconcile hands focus to the layout window.
                self.on_demand_layer_focus = None;
                activate_output = false;
            }
        }

        if activate_output {
            // Find the output from the drop coordinates.
            let id = self.output_containing((location.x as i32, location.y as i32));
            if let Some(output) = id.and_then(|id| self.wl_outputs.get(&id)).cloned() {
                self.layout.focus_output(&output);
            }
        }
    }

    fn cancelled(&mut self, _seat: Seat<Self>, _location: Point<f64, Logical>) {
        self.on_dnd_ended();
    }
}

impl PrismState {
    /// A data-device drag finished (drop or cancel): clear the layout's
    /// DnD state — it was fed by the pointer-motion handlers via
    /// `Layout::dnd_update` to drive edge scrolling and the overview's
    /// hover-to-switch — and drop the icon. Mirrors niri's
    /// `on_maybe_dnd_ended` (handlers/mod.rs:396).
    fn on_dnd_ended(&mut self) {
        self.layout.dnd_end();
        self.dnd_icon = None;
        let ids: Vec<_> = self.outputs.keys().cloned().collect();
        for id in ids {
            self.output_redraw.entry(id).or_default().queue_redraw();
        }
    }
}

// ─── PrimarySelectionHandler ────────────────────────────────────────────────

impl PrimarySelectionHandler for PrismState {
    fn primary_selection_state(&mut self) -> &mut PrimarySelectionState {
        &mut self.primary_selection_state
    }
}

// ─── Data-control (clipboard managers) ──────────────────────────────────────
//
// Both variants ship the same protocol with different namespaces:
// wlr is the legacy one everything supports, ext is the standardized
// successor. Routed by the blanket `delegate_dispatch2!` in state.rs.

impl WlrDataControlHandler for PrismState {
    fn data_control_state(&mut self) -> &mut WlrDataControlState {
        &mut self.wlr_data_control_state
    }
}

impl ExtDataControlHandler for PrismState {
    fn data_control_state(&mut self) -> &mut ExtDataControlState {
        &mut self.ext_data_control_state
    }
}
