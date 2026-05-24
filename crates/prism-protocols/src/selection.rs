//! Clipboard + primary selection + drag-and-drop wiring.
//!
//! Backs three protocols:
//!   - `wl_data_device_manager` (v3) — standard clipboard + DnD.
//!   - `wp_primary_selection_device_manager_v1` — middle-click paste.
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
//! enter/leave/motion/drop events to potential drop targets. When
//! the grab ends, [`DndGrabHandler::dropped`] or `cancelled` clears
//! the icon and queues a redraw.
//!
//! ## Deferred — see TODO comments for details
//!
//!   - **DnD icon rendering.** We *store* the icon surface in
//!     [`PrismState::dnd_icon`] but the render path doesn't draw it
//!     yet. Drag operations work functionally; the cursor just has no
//!     visual drag preview attached. Wiring needs render-path access
//!     to (a) the icon's WlSurface texture, (b) the current cursor
//!     position, (c) per-frame redraw on cursor motion.
//!   - **wlr_data_control / ext_data_control.** Clipboard *manager*
//!     protocols used by tools like `cliphist`, `wl-paste --watch`,
//!     and `clipman`. The core clipboard works without them; add when
//!     a clipboard manager workflow is desired.
//!   - **Cross-monitor drop activation.** niri's `dropped` handler
//!     activates the target output + window so that "drag a Firefox
//!     tab to another monitor" lands focus correctly. Skipped pending
//!     a concrete use case in the prism layout.
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

use smithay::delegate_data_device;
use smithay::delegate_primary_selection;
use smithay::input::Seat;
use smithay::input::dnd::{self, DnDGrab, DndGrabHandler, DndTarget};
use smithay::input::pointer::Focus;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{Logical, Point};
use smithay::wayland::selection::SelectionHandler;
use smithay::wayland::selection::SelectionTarget;
use smithay::wayland::selection::data_device::{
    DataDeviceHandler, DataDeviceState, WaylandDndGrabHandler,
};
use smithay::wayland::selection::primary_selection::{
    PrimarySelectionHandler, PrimarySelectionState,
};

use crate::state::PrismState;

/// DnD cursor icon stashed for the duration of a drag. Lives in
/// [`PrismState::dnd_icon`] from [`WaylandDndGrabHandler::dnd_requested`]
/// (drag start) until [`DndGrabHandler::dropped`] or `cancelled`.
///
/// Currently unread by the render path — see the module-level TODO.
/// When wired, the icon surface should be drawn at
/// `cursor_position + offset` on top of the cursor sprite.
#[derive(Debug)]
pub struct DndIcon {
    /// The wl_surface the client wants drawn under the cursor while
    /// the drag is active. May have its own subsurfaces and damage
    /// events like any other surface.
    pub surface: WlSurface,
    /// Hot-spot offset relative to the cursor position. Today this
    /// is always zero — wl_data_device's start_drag doesn't carry
    /// an explicit offset and we don't yet apply any.
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

        // TODO: per-frame redraw while dragging. Today the cursor
        // doesn't move with the drag from the compositor's POV
        // because we don't render the icon, so this is a no-op
        // worth doing as part of the icon-rendering follow-up.
    }
}

impl DndGrabHandler for PrismState {
    fn dropped(
        &mut self,
        _target: Option<DndTarget<'_, Self>>,
        _validated: bool,
        _seat: Seat<Self>,
        _location: Point<f64, Logical>,
    ) {
        // TODO: niri activates the drop-target output/window here so
        // that drag-tab-into-new-window lands focus on the target.
        // We just clear the icon today.
        self.dnd_icon = None;
    }

    fn cancelled(&mut self, _seat: Seat<Self>, _location: Point<f64, Logical>) {
        self.dnd_icon = None;
    }
}

delegate_data_device!(PrismState);

// ─── PrimarySelectionHandler ────────────────────────────────────────────────

impl PrimarySelectionHandler for PrismState {
    fn primary_selection_state(&mut self) -> &mut PrimarySelectionState {
        &mut self.primary_selection_state
    }
}

delegate_primary_selection!(PrismState);
