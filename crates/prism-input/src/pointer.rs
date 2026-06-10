//! Pointer dispatch — motion, button, axis.
//!
//! Minimal MVP port of niri's `on_pointer_motion` /
//! `on_pointer_motion_absolute` / `on_pointer_button` /
//! `on_pointer_axis` (input/mod.rs lines 2414, 2658, 2750, 3074).
//!
//! Pointer constraints (lock / confine) and relative-pointer deltas are
//! handled in `on_pointer_motion`; activation/teardown lives in
//! [`prism_protocols::PrismState::maybe_activate_pointer_constraint`].
//!
//! Carries the hot-corner trigger, the overview's pointer bindings
//! (LMB click/drag, wheel, finger scroll), the move/resize/spatial
//! grab triggers, and the DnD → layout feed.
//!
//! What's intentionally not here (yet):
//!   - Tablet integration
//!   - pick_window / pick_color grabs (niri's screenshot-UI helpers)
//!   - Cursor auto-hide / pointer-inactivity timer
//!
//! These can all bolt onto this file as their backing state lands.

use prism_protocols::PrismState;
use smithay::backend::input::{
    AbsolutePositionEvent, Axis, AxisSource, ButtonState, PointerAxisEvent, PointerButtonEvent,
    PointerMotionEvent,
};
use smithay::input::pointer::{AxisFrame, ButtonEvent, MotionEvent, RelativeMotionEvent};
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{Logical, Point, Rectangle, Size, SERIAL_COUNTER};
use smithay::wayland::pointer_constraints::{with_pointer_constraint, PointerConstraint};

use crate::backend_ext::PrismInputBackend;

pub fn on_pointer_motion<I: PrismInputBackend>(
    state: &mut PrismState,
    event: I::PointerMotionEvent,
) {
    // Any early return below means the pointer is not inside the hot
    // corner (niri does the same reset-first dance).
    let was_inside_hot_corner = state.pointer_inside_hot_corner;
    state.pointer_inside_hot_corner = false;

    let serial = SERIAL_COUNTER.next_serial();
    let time = smithay::backend::input::Event::time_msec(&event);

    let Some(pointer) = state.seat.get_pointer() else {
        return;
    };

    // Both accelerated and raw deltas, plus the microsecond timestamp, are
    // needed for relative-motion events (zwp_relative_pointer_v1).
    let delta = event.delta();
    let delta_unaccel = event.delta_unaccel();
    let utime = smithay::backend::input::Event::time(&event);

    let pos = state.pointer_pos;
    // Tentative new position: only committed to `state.pointer_pos` once we
    // know a pointer lock / confinement doesn't forbid the move.
    let new_pos = pos + delta;

    // Pointer constraint check against the CURRENT pointer focus. The
    // constraint deactivates automatically (in smithay) when focus leaves the
    // surface, so an active constraint here means focus is still on it.
    let current = state.pointer_contents.clone();
    let mut pointer_confined = None;
    if let Some(under) = &current {
        let pos_within_surface = pos - under.1;

        let mut pointer_locked = false;
        with_pointer_constraint(&under.0, &pointer, |constraint| {
            let Some(constraint) = constraint else {
                return;
            };
            if !constraint.is_active() {
                return;
            }
            // Constraint does not apply if the pointer is outside its region.
            if let Some(region) = constraint.region() {
                if !region.contains(pos_within_surface.to_i32_round()) {
                    return;
                }
            }
            match &*constraint {
                PointerConstraint::Locked(_locked) => pointer_locked = true,
                PointerConstraint::Confined(confine) => {
                    pointer_confined = Some((under.clone(), confine.region().cloned()));
                }
            }
        });

        // Locked: the pointer stays put. Deliver only relative motion (this is
        // what FPS mouselook reads) and leave `pointer_pos` untouched.
        if pointer_locked {
            pointer.relative_motion(
                state,
                Some(under.clone()),
                &RelativeMotionEvent {
                    delta,
                    delta_unaccel,
                    utime,
                },
            );
            pointer.frame(state);
            return;
        }
    }

    // Clamp the tentative position into the union of all output rects, then
    // resolve what's under it. Without clamping the pointer can drift forever
    // and the focus query stops finding anything.
    let new_pos = clamp_point_to_outputs(state, new_pos);
    let focus = state.contents_under(new_pos);

    // Confined: prevent the pointer from leaving the focused surface (or its
    // confine region). Like the locked case, deliver relative motion only and
    // don't move the pointer when the move would breach the confinement.
    if let Some((focus_surface, region)) = pointer_confined {
        let mut prevent = false;
        if Some(&focus_surface.0) != focus.as_ref().map(|(s, _)| s) {
            prevent = true;
        }
        if let Some(region) = region {
            let new_within_surface = new_pos - focus_surface.1;
            if !region.contains(new_within_surface.to_i32_round()) {
                prevent = true;
            }
        }
        if prevent {
            pointer.relative_motion(
                state,
                Some(focus_surface),
                &RelativeMotionEvent {
                    delta,
                    delta_unaccel,
                    utime,
                },
            );
            pointer.frame(state);
            return;
        }
    }

    state.pointer_pos = new_pos;
    // Keep the tracked contents in sync so the post-dispatch
    // `refresh_pointer_focus` doesn't see a spurious change and re-fire.
    state.pointer_contents = focus.clone();

    pointer.motion(
        state,
        focus.clone(),
        &MotionEvent {
            location: new_pos,
            serial,
            time,
        },
    );
    pointer.relative_motion(
        state,
        focus,
        &RelativeMotionEvent {
            delta,
            delta_unaccel,
            utime,
        },
    );
    pointer.frame(state);
    maybe_focus_follows_mouse(state);
    maybe_trigger_hot_corner(state, was_inside_hot_corner);
    maybe_dnd_update(state, &pointer);
    // Walk the cursor plane on every output: show on the output the
    // pointer is in, hide on the rest, queue redraws on changes.
    prism_protocols::state::update_output_cursors(state);
    // A constraint may want to activate now that focus settled here.
    state.maybe_activate_pointer_constraint();
}

pub fn on_pointer_motion_absolute<I: PrismInputBackend>(
    state: &mut PrismState,
    event: I::PointerMotionAbsoluteEvent,
) {
    // See on_pointer_motion: reset-first, edge-triggered.
    let was_inside_hot_corner = state.pointer_inside_hot_corner;
    state.pointer_inside_hot_corner = false;

    let serial = SERIAL_COUNTER.next_serial();
    let time = smithay::backend::input::Event::time_msec(&event);

    // For an absolute event, transform the [0..1] device-space
    // coordinates into the global bounding rect.
    let Some(bounds) = global_bounding_rect(state) else {
        return;
    };
    let pos = event.position_transformed(bounds.size);
    state.pointer_pos = (bounds.loc.x as f64 + pos.x, bounds.loc.y as f64 + pos.y).into();
    clamp_pointer_to_outputs(state);

    let focus = surface_under_pointer(state);
    // Keep the tracked contents in sync so the post-dispatch
    // `refresh_pointer_focus` doesn't see a spurious change and re-fire.
    state.pointer_contents = focus.clone();
    let new_pos = state.pointer_pos;

    let Some(pointer) = state.seat.get_pointer() else {
        return;
    };
    pointer.motion(
        state,
        focus,
        &MotionEvent {
            location: new_pos,
            serial,
            time,
        },
    );
    pointer.frame(state);
    maybe_focus_follows_mouse(state);
    maybe_trigger_hot_corner(state, was_inside_hot_corner);
    maybe_dnd_update(state, &pointer);
    prism_protocols::state::update_output_cursors(state);
    // Absolute motion doesn't enforce locks (no meaningful raw delta), but it
    // can still settle focus onto a surface that wants to activate a
    // constraint — e.g. a confine. Matches niri's absolute-motion handler.
    state.maybe_activate_pointer_constraint();
}

pub fn on_pointer_button<I: PrismInputBackend>(
    state: &mut PrismState,
    event: I::PointerButtonEvent,
) {
    let Some(pointer) = state.seat.get_pointer() else {
        return;
    };
    let serial = SERIAL_COUNTER.next_serial();
    let time = smithay::backend::input::Event::time_msec(&event);
    // `event.button()` returns Option<MouseButton> (an enum); casting
    // its discriminant via `as u32` gives 0/1/2/… — not the linux
    // input event code (`BTN_LEFT=0x110`, `BTN_RIGHT=0x111`, …) that
    // clients and our grab triggers actually expect. Use button_code()
    // for the raw kernel value.
    let button = event.button_code();
    let state_pressed = event.state() == ButtonState::Pressed;

    // A button whose press was consumed by a `Mouse*` bind: swallow
    // the matching release too, so the client never sees a dangling
    // release (niri's `suppressed_buttons`; the pointer analogue of
    // `suppressed_keys`).
    if state.suppressed_buttons.remove(&button) {
        return;
    }

    // Configured mouse-button binds (`Mod+MouseMiddle { ... }`) fire
    // before any other press handling, mirroring niri's
    // `on_pointer_button` order. Deliberately not gated on
    // `is_grabbed`: niri dispatches mouse binds during grabs too.
    if state_pressed {
        use prism_config::Trigger;
        use smithay::backend::input::MouseButton;
        let trigger = match event.button() {
            Some(MouseButton::Left) => Some(Trigger::MouseLeft),
            Some(MouseButton::Right) => Some(Trigger::MouseRight),
            Some(MouseButton::Middle) => Some(Trigger::MouseMiddle),
            Some(MouseButton::Back) => Some(Trigger::MouseBack),
            Some(MouseButton::Forward) => Some(Trigger::MouseForward),
            _ => None,
        };
        if let Some(trigger) = trigger {
            let (snapshot, mod_key) = binds_snapshot(state);
            let mods = state
                .seat
                .get_keyboard()
                .map(|kb| kb.modifier_state())
                .unwrap_or_default();
            if let Some(bind) = crate::dispatch::find_bind(&snapshot, mod_key, trigger, mods) {
                state.suppressed_buttons.insert(button);
                crate::dispatch::handle_bind(state, bind);
                return;
            }
        }
    }

    // Click-to-focus: on press, take keyboard focus to the surface
    // under the pointer AND make that surface's output the layout's
    // active monitor. Without the focus_output call the focus ring
    // would stay drawn on whichever output happened to be active at
    // startup (typically the first in connector-name sort order,
    // DP-4 on the current hardware), even when the user clicks
    // somewhere else. niri runs the same `focus_output` from its
    // input handlers.
    if state_pressed && !pointer.is_grabbed() {
        // Spatial-movement drags run before click-to-focus: RMB in the
        // overview pans the workspace view; Mod+MMB pans / switches
        // workspaces anywhere. niri deliberately does NOT activate the
        // window under the cursor for these (avoids surprise scrolling
        // when Mod+MMB-clicking a partially off-screen window).
        if spatial_movement_press(state, button, serial) {
            return;
        }

        if let Some((surface, _)) = surface_under_pointer(state) {
            // Click-to-focus is for switching between WINDOWS. Clicking a
            // popup (menu item) must NOT move keyboard focus onto the popup:
            // that sends wl_keyboard.leave to the parent toplevel, and
            // grab-less clients like Firefox dismiss their own menu when the
            // toplevel loses focus — so the click would tear the menu down
            // before its item activates. The pointer button is still
            // delivered below; we just skip the focus switch. Popup keyboard
            // focus, when a client wants it, comes from a real popup grab.
            if !state.surface_is_popup(&surface) {
                // Activate (and raise) the clicked window; this also makes its
                // output the layout's active monitor, so no separate
                // `focus_output` is needed. `update_keyboard_focus` (run every
                // frame) then hands it the keyboard. niri's click path is the
                // same single `activate_window`.
                focus_surface(state, &surface, true);
            }
        }

        // Overview: plain LeftClick interacts with the zoomed cards
        // (niri input/mod.rs overview branches). On a window it starts
        // the move grab — a click (release before the layout's move
        // threshold) then activates the window and zooms to its
        // workspace, handled in `MoveGrab::end`; a drag interactively
        // moves it. On a workspace card's background it zooms straight
        // to that workspace. (RMB/MMB drags were handled above by
        // `spatial_movement_press`.)
        if state.layout.is_overview_open()
            && button == BTN_LEFT
            && overview_lmb_press(state, serial)
        {
            // Press consumed by the grab; the release is delivered by
            // the grab's own button handler when it unsets.
            return;
        }

        // Mod+LeftClick / Mod+RightClick on a window installs an
        // interactive grab — move / resize respectively. Mirrors
        // niri's `on_pointer_button` triggers (input/mod.rs:2895+).
        if try_begin_window_grab(state, button, serial) {
            // Don't forward this button press to clients — the press
            // is consumed by the grab. The release will be delivered
            // by the grab's own button handler when it unsets.
            return;
        }
    }

    pointer.button(
        state,
        &ButtonEvent {
            button,
            state: event.state(),
            serial,
            time,
        },
    );
    pointer.frame(state);
}

/// Linux input-event codes for the mouse buttons we trigger grabs on.
/// Match niri's hardcoded constants in `input/mod.rs::on_pointer_button`.
const BTN_LEFT: u32 = 0x110;
const BTN_RIGHT: u32 = 0x111;
const BTN_MIDDLE: u32 = 0x112;

/// Try to start a spatial-movement grab (niri input/mod.rs:2825-2889):
/// RMB in the overview pans the workspace view under the cursor (the
/// view-offset gesture begins immediately); Mod+MMB starts the
/// recognizing variant whose first 8px of travel picks view-pan
/// (horizontal) or workspace-switch (vertical). Returns `true` if a
/// grab consumed the press.
fn spatial_movement_press(
    state: &mut PrismState,
    button: u32,
    serial: smithay::utils::Serial,
) -> bool {
    use smithay::input::pointer::{CursorIcon, CursorImageStatus, Focus};

    let overview_open = state.layout.is_overview_open();

    let is_rmb_overview = button == BTN_RIGHT && overview_open;
    let is_mod_mmb = button == BTN_MIDDLE && {
        use prism_config::ModKey;
        let mod_key = state.config.borrow().input.mod_key.unwrap_or(ModKey::Super);
        state
            .seat
            .get_keyboard()
            .map(|kb| {
                crate::dispatch::modifiers_from_state(kb.modifier_state())
                    .contains(mod_key.to_modifiers())
            })
            .unwrap_or(false)
    };
    if !is_rmb_overview && !is_mod_mmb {
        return false;
    }

    let Some((out, pos_within_output)) = output_under_pointer(state) else {
        return false;
    };

    // RMB-in-overview targets the workspace under the cursor (extended,
    // full-width bounds); Mod+MMB does too in the overview, but outside
    // it uses the active workspace — hit-testing during animations
    // could catch the wrong one (niri's comment).
    let ws_id = if overview_open {
        state
            .layout
            .monitor_for_output(&out)
            .and_then(|mon| mon.workspace_under(pos_within_output))
            .map(|(ws, _geo)| ws.id())
    } else {
        state
            .layout
            .monitor_for_output(&out)
            .map(|mon| mon.active_workspace_ref().id())
    };
    let Some(ws_id) = ws_id else {
        return false;
    };

    state.layout.focus_output(&out);

    if is_rmb_overview {
        let Some((ws_idx, _)) = state.layout.find_workspace_by_id(ws_id) else {
            return false;
        };
        state
            .layout
            .view_offset_gesture_begin(&out, Some(ws_idx), false);
    }

    let Some(pointer) = state.seat.get_pointer() else {
        return false;
    };
    let start_data = smithay::input::pointer::GrabStartData {
        focus: None,
        button,
        location: state.pointer_pos,
    };
    let grab = crate::SpatialMovementGrab::new(start_data, out, ws_id, is_rmb_overview);
    pointer.set_grab(state, grab, serial, Focus::Clear);

    state
        .cursor_manager
        .set_cursor_image(CursorImageStatus::Named(CursorIcon::AllScroll));
    state.cursor_dirty = true;
    prism_protocols::state::update_output_cursors(state);
    crate::move_grab::queue_redraw_all(state);
    true
}

/// Handle an unmodified LeftClick press while the overview is open.
/// On a window (the zoom-aware `window_under` resolves hits on the
/// scaled cards): start the move grab — `MoveGrab::end` turns a click
/// into activate-and-zoom-to-workspace. On a workspace card's
/// background (narrow bounds — clicks in the gaps/backdrop do
/// nothing): zoom to that workspace. Returns `true` if a grab consumed
/// the press.
fn overview_lmb_press(state: &mut PrismState, serial: smithay::utils::Serial) -> bool {
    use smithay::input::pointer::Focus;

    let Some((out, pos_within_output)) = output_under_pointer(state) else {
        return false;
    };

    if let Some((mapped, _hit)) = state.layout.window_under(&out, pos_within_output) {
        let window = mapped.window.clone();
        let Some(pointer) = state.seat.get_pointer() else {
            return false;
        };
        let start_data = smithay::input::pointer::GrabStartData {
            focus: None,
            button: BTN_LEFT,
            location: state.pointer_pos,
        };
        let Some(grab) = crate::MoveGrab::new(state, start_data, window, pos_within_output, out)
        else {
            return false;
        };
        pointer.set_grab(state, grab, serial, Focus::Clear);
        // niri deliberately keeps the normal cursor here: in the
        // overview a press is usually a click-to-activate, and
        // flashing a grab cursor would be distracting.
        return true;
    }

    // No window: a click on a workspace card's background switches to
    // it and closes the overview (niri input/mod.rs:3002).
    let ws_id = state
        .layout
        .monitor_for_output(&out)
        .and_then(|mon| mon.workspace_under_narrow(pos_within_output))
        .map(|ws| ws.id());
    if let Some(ws_id) = ws_id {
        if let Some((ws_idx, _)) = state.layout.find_workspace_by_id(ws_id) {
            state.layout.focus_output(&out);
            state.layout.toggle_overview_to_workspace(ws_idx);
            crate::move_grab::queue_redraw_all(state);
        }
    }
    false
}

/// Try to start an interactive move (Mod+LeftClick) or resize
/// (Mod+RightClick) grab. Returns `true` if a grab was installed (in
/// which case the caller should not forward the originating button
/// press to clients).
fn try_begin_window_grab(
    state: &mut PrismState,
    button: u32,
    serial: smithay::utils::Serial,
) -> bool {
    use prism_config::ModKey;
    use smithay::input::pointer::Focus;

    // Only the two buttons we use as gesture triggers.
    if button != BTN_LEFT && button != BTN_RIGHT {
        return false;
    }

    // Mod-down check. Without it, plain LeftClick would also start a
    // move grab — not what the user wants.
    let mod_key = state.config.borrow().input.mod_key.unwrap_or(ModKey::Super);
    let Some(keyboard) = state.seat.get_keyboard() else {
        return false;
    };
    let mods = keyboard.modifier_state();
    let mods_bits = crate::dispatch::modifiers_from_state(mods);
    if !mods_bits.contains(mod_key.to_modifiers()) {
        return false;
    }

    // Window + output + position-within-output under the cursor.
    let Some((out, pos_within_output)) = output_under_pointer(state) else {
        return false;
    };
    let Some((mapped, _hit)) = state.layout.window_under(&out, pos_within_output) else {
        return false;
    };
    let window = mapped.window.clone();

    let Some(pointer) = state.seat.get_pointer() else {
        return false;
    };
    let location = state.pointer_pos;
    let start_data = smithay::input::pointer::GrabStartData {
        focus: None,
        button,
        location,
    };

    if button == BTN_LEFT {
        let Some(grab) = crate::MoveGrab::new(state, start_data, window, pos_within_output, out)
        else {
            return false;
        };
        pointer.set_grab(state, grab, serial, Focus::Clear);
        true
    } else {
        // BTN_RIGHT — resize. Need the edge under the cursor; if the
        // pointer is dead-center of the window there's no edge to
        // resize, so we bail (niri does the same).
        let edges = state
            .layout
            .resize_edges_under(&out, pos_within_output)
            .unwrap_or_else(prism_layout::utils::ResizeEdge::empty);
        if edges.is_empty() {
            return false;
        }
        if !state.layout.interactive_resize_begin(window.clone(), edges) {
            return false;
        }
        let grab = crate::ResizeGrab::new(start_data, window);
        pointer.set_grab(state, grab, serial, Focus::Clear);
        true
    }
}

pub fn on_pointer_axis<I: PrismInputBackend>(state: &mut PrismState, event: I::PointerAxisEvent) {
    let Some(pointer) = state.seat.get_pointer() else {
        return;
    };
    let time = smithay::backend::input::Event::time_msec(&event);

    let source = event.source();

    // Overview hardcoded wheel bindings: unmodified wheel up/down
    // switches workspaces, left/right moves column focus, instead of
    // scrolling clients (niri input/mod.rs:3115+ synthesizes
    // `*UnderMouse` binds for this; prism dispatches the active-monitor
    // actions — identical on a single monitor, and prism's multi-
    // monitor focus tracking isn't wired yet anyway). Scrolls consumed
    // here never reach Wayland clients.
    if source == AxisSource::Wheel {
        if state.layout.is_overview_open() && overview_wheel_scroll::<I>(state, &event) {
            return;
        }
        // Configured `WheelScroll*` binds (Mod+wheel workspace switch
        // and friends). Consumes the whole event when a bind exists
        // for the held modifiers.
        if wheel_bind_scroll::<I>(state, &event) {
            return;
        }
    }

    // Touchpad two-finger (and continuous-device) scroll in the
    // overview drives the workspace-switch / view-offset gestures
    // continuously instead of scrolling clients (niri
    // input/mod.rs:3296). Outside the overview the helper winds down
    // any gesture left over from an overview scroll and lets the
    // event through.
    if matches!(source, AxisSource::Finger | AxisSource::Continuous) {
        if overview_finger_scroll::<I>(state, &event) {
            return;
        }
        // Configured `TouchpadScroll*` binds.
        if finger_bind_scroll::<I>(state, &event) {
            return;
        }
    }

    // Pass-through to the client under the pointer, scaled by the
    // device scroll-factor (`mouse { scroll-factor }` for wheels,
    // `touchpad { scroll-factor }` for finger scrolls) × the window
    // rule's `scroll-factor` (niri input/mod.rs:3486).
    let device_scroll_factor = {
        let cfg = state.config.borrow();
        match source {
            AxisSource::Wheel => cfg.input.mouse.scroll_factor,
            AxisSource::Finger => cfg.input.touchpad.scroll_factor,
            _ => None,
        }
    };
    let window_scroll_factor = {
        use prism_layout::layout::LayoutElement as _;
        state
            .pointer_contents
            .as_ref()
            .map(|(surface, _)| state.find_root_shell_surface(surface))
            .and_then(|root| state.layout.find_window_and_output(&root))
            .and_then(|(mapped, _)| mapped.rules().scroll_factor)
            .unwrap_or(1.)
    };
    let (h_factor, v_factor) = device_scroll_factor
        .map(|f| f.h_v_factors())
        .unwrap_or((1.0, 1.0));
    let (h_factor, v_factor) = (
        h_factor * window_scroll_factor,
        v_factor * window_scroll_factor,
    );

    let mut frame = AxisFrame::new(time).source(source);

    for axis in [Axis::Horizontal, Axis::Vertical] {
        let factor = match axis {
            Axis::Horizontal => h_factor,
            Axis::Vertical => v_factor,
        };
        if let Some(discrete) = event.amount_v120(axis) {
            // v120 increments are smithay's preferred high-resolution
            // discrete scroll signal.
            frame = frame.v120(axis, (discrete * factor) as i32);
        }
        if let Some(amount) = event.amount(axis) {
            frame = frame.value(axis, amount * factor);
        } else if let Some(amount_discrete) = event.amount_v120(axis) {
            // Some backends only give discrete; convert to a smooth
            // value at ~10 px per notch, niri's default ratio.
            frame = frame.value(axis, amount_discrete / 120.0 * 10.0 * factor);
        }
        // niri stops a wheel "frame" with a stop event for finger
        // scrolls when the amount is exactly zero — we forward that
        // through with `stop`.
        if event.amount(axis) == Some(0.0) && matches!(source, AxisSource::Finger) {
            frame = frame.stop(axis);
        }
    }

    pointer.axis(state, frame);
    pointer.frame(state);
}

/// Snapshot the bind table + resolved Mod key out of the config
/// RefCell, so bind matching never holds the config borrow across an
/// action (actions re-borrow config freely).
fn binds_snapshot(state: &PrismState) -> (Vec<prism_config::Bind>, prism_config::ModKey) {
    let cfg = state.config.borrow();
    (
        cfg.binds.0.clone(),
        cfg.input.mod_key.unwrap_or(prism_config::ModKey::Super),
    )
}

/// Dispatch configured `WheelScroll{Up,Down,Left,Right}` binds for a
/// wheel-source axis event. Returns `true` when the event is consumed:
/// if any wheel bind exists for the held modifier combination, the
/// whole event is swallowed (sub-tick amounts accumulate in the wheel
/// trackers rather than reaching clients) — niri's
/// `mods_with_wheel_binds` gate. With no bind for these modifiers the
/// trackers reset and the event passes through.
fn wheel_bind_scroll<I: PrismInputBackend>(
    state: &mut PrismState,
    event: &I::PointerAxisEvent,
) -> bool {
    use prism_config::Trigger;

    const WHEEL_TRIGGERS: &[Trigger] = &[
        Trigger::WheelScrollUp,
        Trigger::WheelScrollDown,
        Trigger::WheelScrollLeft,
        Trigger::WheelScrollRight,
    ];

    let mods = state
        .seat
        .get_keyboard()
        .map(|kb| kb.modifier_state())
        .unwrap_or_default();
    let modifiers = crate::dispatch::modifiers_from_state(mods);
    let (snapshot, mod_key) = binds_snapshot(state);

    if !crate::dispatch::binds_have_trigger_for_mods(&snapshot, mod_key, WHEEL_TRIGGERS, modifiers)
    {
        state.horizontal_wheel_tracker.reset();
        state.vertical_wheel_tracker.reset();
        return false;
    }

    let horizontal = event.amount_v120(Axis::Horizontal).unwrap_or(0.);
    let ticks = state.horizontal_wheel_tracker.accumulate(horizontal);
    if ticks != 0 {
        let left = crate::dispatch::find_bind(&snapshot, mod_key, Trigger::WheelScrollLeft, mods);
        let right = crate::dispatch::find_bind(&snapshot, mod_key, Trigger::WheelScrollRight, mods);
        if let Some(right) = right {
            for _ in 0..ticks {
                crate::dispatch::handle_bind(state, right.clone());
            }
        }
        if let Some(left) = left {
            for _ in ticks..0 {
                crate::dispatch::handle_bind(state, left.clone());
            }
        }
    }

    let vertical = event.amount_v120(Axis::Vertical).unwrap_or(0.);
    let ticks = state.vertical_wheel_tracker.accumulate(vertical);
    if ticks != 0 {
        let up = crate::dispatch::find_bind(&snapshot, mod_key, Trigger::WheelScrollUp, mods);
        let down = crate::dispatch::find_bind(&snapshot, mod_key, Trigger::WheelScrollDown, mods);
        if let Some(down) = down {
            for _ in 0..ticks {
                crate::dispatch::handle_bind(state, down.clone());
            }
        }
        if let Some(up) = up {
            for _ in ticks..0 {
                crate::dispatch::handle_bind(state, up.clone());
            }
        }
    }

    true
}

/// Dispatch configured `TouchpadScroll{Up,Down,Left,Right}` binds for
/// a finger/continuous-source axis event. Same consume-vs-passthrough
/// shape as [`wheel_bind_scroll`], but accumulating pixel deltas
/// (tick = 10) instead of v120 units.
fn finger_bind_scroll<I: PrismInputBackend>(
    state: &mut PrismState,
    event: &I::PointerAxisEvent,
) -> bool {
    use prism_config::Trigger;

    const FINGER_TRIGGERS: &[Trigger] = &[
        Trigger::TouchpadScrollUp,
        Trigger::TouchpadScrollDown,
        Trigger::TouchpadScrollLeft,
        Trigger::TouchpadScrollRight,
    ];

    let mods = state
        .seat
        .get_keyboard()
        .map(|kb| kb.modifier_state())
        .unwrap_or_default();
    let modifiers = crate::dispatch::modifiers_from_state(mods);
    let (snapshot, mod_key) = binds_snapshot(state);

    if !crate::dispatch::binds_have_trigger_for_mods(&snapshot, mod_key, FINGER_TRIGGERS, modifiers)
    {
        state.horizontal_finger_scroll_tracker.reset();
        state.vertical_finger_scroll_tracker.reset();
        return false;
    }

    let horizontal = event.amount(Axis::Horizontal).unwrap_or(0.);
    let ticks = state
        .horizontal_finger_scroll_tracker
        .accumulate(horizontal);
    if ticks != 0 {
        let left =
            crate::dispatch::find_bind(&snapshot, mod_key, Trigger::TouchpadScrollLeft, mods);
        let right =
            crate::dispatch::find_bind(&snapshot, mod_key, Trigger::TouchpadScrollRight, mods);
        if let Some(right) = right {
            for _ in 0..ticks {
                crate::dispatch::handle_bind(state, right.clone());
            }
        }
        if let Some(left) = left {
            for _ in ticks..0 {
                crate::dispatch::handle_bind(state, left.clone());
            }
        }
    }

    let vertical = event.amount(Axis::Vertical).unwrap_or(0.);
    let ticks = state.vertical_finger_scroll_tracker.accumulate(vertical);
    if ticks != 0 {
        let up = crate::dispatch::find_bind(&snapshot, mod_key, Trigger::TouchpadScrollUp, mods);
        let down =
            crate::dispatch::find_bind(&snapshot, mod_key, Trigger::TouchpadScrollDown, mods);
        if let Some(down) = down {
            for _ in 0..ticks {
                crate::dispatch::handle_bind(state, down.clone());
            }
        }
        if let Some(up) = up {
            for _ in ticks..0 {
                crate::dispatch::handle_bind(state, up.clone());
            }
        }
    }

    true
}

/// Handle a wheel-source axis event while the overview is open.
/// Returns `true` when the event was consumed (any unmodified wheel
/// scroll — even sub-tick amounts accumulate silently rather than
/// reaching clients, matching niri). Modified scrolls, and scrolls
/// over a Top/Overlay layer surface (a bar's volume scroll keeps
/// working in the overview), pass through untouched.
fn overview_wheel_scroll<I: PrismInputBackend>(
    state: &mut PrismState,
    event: &I::PointerAxisEvent,
) -> bool {
    use prism_config::Action;

    // Unmodified wheel and Shift+wheel are the overview bindings;
    // anything else passes through.
    let Some(keyboard) = state.seat.get_keyboard() else {
        return false;
    };
    let mods = crate::dispatch::modifiers_from_state(keyboard.modifier_state());
    let shift = mods == prism_config::Modifiers::SHIFT;
    if !mods.is_empty() && !shift {
        return false;
    }

    // Scrolling a Top/Overlay layer surface (bar, launcher) keeps its
    // normal meaning (niri's `should_handle_in_overview` gate).
    if pointer_over_top_or_overlay_layer(state) {
        return false;
    }

    // Vertical, unmodified: workspace switch, with niri's 50ms
    // cooldown so one flick doesn't skip several workspaces.
    // Vertical, Shift: column focus (niri's wheel-only-mouse
    // affordance for horizontal movement).
    let vertical = event.amount_v120(Axis::Vertical).unwrap_or(0.);
    let ticks = state.vertical_wheel_tracker.accumulate(vertical);
    if ticks != 0 {
        if shift {
            let action = if ticks > 0 {
                Action::FocusColumnRight
            } else {
                Action::FocusColumnLeft
            };
            for _ in 0..ticks.unsigned_abs() {
                crate::actions::handle_action(state, action.clone());
            }
        } else {
            let now = std::time::Instant::now();
            let cooled = state.overview_wheel_last_switch.is_none_or(|last| {
                now.duration_since(last) >= std::time::Duration::from_millis(50)
            });
            if cooled {
                // One workspace per event regardless of tick count —
                // the cooldown would swallow the extras anyway.
                let action = if ticks > 0 {
                    Action::FocusWorkspaceDown
                } else {
                    Action::FocusWorkspaceUp
                };
                crate::actions::handle_action(state, action);
                state.overview_wheel_last_switch = Some(now);
            }
        }
    }

    // Horizontal: column focus (no cooldown, matching niri).
    let horizontal = event.amount_v120(Axis::Horizontal).unwrap_or(0.);
    let ticks = state.horizontal_wheel_tracker.accumulate(horizontal);
    if ticks != 0 {
        let action = if ticks > 0 {
            Action::FocusColumnRight
        } else {
            Action::FocusColumnLeft
        };
        for _ in 0..ticks.unsigned_abs() {
            crate::actions::handle_action(state, action.clone());
        }
    }

    true
}

/// Handle a Finger/Continuous-source axis event. Returns `true` when
/// consumed: in the overview (unmodified, not over a bar) the scroll
/// drives the workspace-switch (vertical) or view-offset (horizontal)
/// gesture as a continuous swipe — `ScrollSwipeGesture` supplies the
/// begin/update/end edges that wheel events don't have. When not
/// handling, any gesture still ongoing from an overview scroll is
/// wound down and the event passes to clients.
fn overview_finger_scroll<I: PrismInputBackend>(
    state: &mut PrismState,
    event: &I::PointerAxisEvent,
) -> bool {
    let timestamp = std::time::Duration::from_micros(smithay::backend::input::Event::time(event));
    let horizontal = event.amount(Axis::Horizontal).unwrap_or(0.);
    let vertical = event.amount(Axis::Vertical).unwrap_or(0.);

    let mods_empty = state
        .seat
        .get_keyboard()
        .map(|kb| crate::dispatch::modifiers_from_state(kb.modifier_state()).is_empty())
        .unwrap_or(false);

    if state.layout.is_overview_open() && mods_empty && !pointer_over_top_or_overlay_layer(state) {
        let action = state
            .overview_scroll_swipe_gesture
            .update(horizontal, vertical);
        let is_vertical = state.overview_scroll_swipe_gesture.is_vertical();
        let mut redraw = false;

        if action.end() {
            if is_vertical {
                redraw |= state
                    .layout
                    .workspace_switch_gesture_end(Some(true))
                    .is_some();
            } else {
                redraw |= state.layout.view_offset_gesture_end(Some(true)).is_some();
            }
        } else if is_vertical {
            if action.begin() {
                if let Some((out, _)) = output_under_pointer(state) {
                    state.layout.workspace_switch_gesture_begin(&out, true);
                    redraw = true;
                }
            }
            if let Some(Some(_)) = state
                .layout
                .workspace_switch_gesture_update(vertical, timestamp, true)
            {
                redraw = true;
            }
        } else {
            if action.begin() {
                if let Some((out, pos_within_output)) = output_under_pointer(state) {
                    // Extended (full-width) workspace bounds: a
                    // horizontal scroll between cards still targets
                    // the nearest one (niri workspace_under_cursor(true)).
                    let ws_id = state
                        .layout
                        .monitor_for_output(&out)
                        .and_then(|mon| mon.workspace_under(pos_within_output))
                        .map(|(ws, _geo)| ws.id());
                    if let Some(ws_id) = ws_id {
                        if let Some((ws_idx, _)) = state.layout.find_workspace_by_id(ws_id) {
                            state
                                .layout
                                .view_offset_gesture_begin(&out, Some(ws_idx), true);
                            redraw = true;
                        }
                    }
                }
            }
            if let Some(Some(_)) = state
                .layout
                .view_offset_gesture_update(horizontal, timestamp, true)
            {
                redraw = true;
            }
        }

        if redraw {
            crate::move_grab::queue_redraw_all(state);
        }
        return true;
    }

    // Not handling (overview closed mid-scroll, modifier pressed, or
    // over a bar): wind down whichever gesture the overview scroll had
    // going so it doesn't stay grabbed (niri's reset branch).
    if state.overview_scroll_swipe_gesture.reset() {
        let redraw = if state.overview_scroll_swipe_gesture.is_vertical() {
            state
                .layout
                .workspace_switch_gesture_end(Some(true))
                .is_some()
        } else {
            state.layout.view_offset_gesture_end(Some(true)).is_some()
        };
        if redraw {
            crate::move_grab::queue_redraw_all(state);
        }
    }
    false
}

/// Hot-corner trigger, run after a motion event delivered its focus.
/// Toggles the overview when the pointer *enters* a configured hot
/// corner (edge-triggered via `pointer_inside_hot_corner`, which the
/// caller captured-and-reset at the top of the handler).
///
/// Conditions, mirroring niri (input/mod.rs:2628):
///   - the corner pixel must not be focus-owned (`contents_under`
///     makes it dead, so normally focus is None there; an implicit
///     click grab keeps focus on its window, correctly blocking the
///     trigger mid-click);
///   - an active grab must allow it — resize is blocklisted; the move
///     grab is allowed so dragging a window into the corner opens the
///     overview; DnD likewise (drag across workspaces).
fn maybe_trigger_hot_corner(state: &mut PrismState, was_inside: bool) {
    // No overview from a locked screen. (The `current_focus` guard
    // below doesn't cover the corner of an output whose lock surface
    // is missing — focus is None there.)
    if state.is_locked() {
        return;
    }
    let Some(pointer) = state.seat.get_pointer() else {
        return;
    };
    let Some((out, pos_within_output)) = output_under_pointer(state) else {
        return;
    };
    if !state.is_inside_hot_corner(&out, pos_within_output) || pointer.current_focus().is_some() {
        return;
    }

    if !was_inside
        && pointer
            .with_grab(|_, grab| !grab.as_any().is::<crate::ResizeGrab>())
            .unwrap_or(true)
    {
        state.layout.toggle_overview();
        crate::move_grab::queue_redraw_all(state);
    }
    state.pointer_inside_hot_corner = true;
}

/// Inform the layout of an ongoing data-device DnD drag (niri
/// input/mod.rs:2644): it drives the DnD edge view-scroll and, in the
/// overview, hovering a workspace card edge-switches to it. The layout
/// state is cleared from the drop/cancel handlers (`Layout::dnd_end`).
fn maybe_dnd_update(
    state: &mut PrismState,
    pointer: &smithay::input::pointer::PointerHandle<PrismState>,
) {
    use smithay::input::dnd::DnDGrab;
    use smithay::reexports::wayland_server::protocol::wl_data_source::WlDataSource;

    let is_dnd = pointer
        .with_grab(|_, grab| {
            let grab = grab.as_any();
            // Normal DnD, plus null-source DnD (weston-dnd --self-only).
            grab.is::<DnDGrab<PrismState, WlDataSource, WlSurface>>()
                || grab.is::<DnDGrab<PrismState, WlSurface, WlSurface>>()
        })
        .unwrap_or(false);
    if !is_dnd {
        return;
    }

    if let Some((out, pos_within_output)) = output_under_pointer(state) {
        state.layout.dnd_update(out, pos_within_output);
        // The DnD scroll/switch gestures animate; one queued redraw
        // starts the cycle, are_animations_ongoing keeps it running.
        crate::move_grab::queue_redraw_all(state);
    }
}

/// The output under the pointer, and the pointer position within it
/// (output-local logical coordinates).
pub(crate) fn output_under_pointer(
    state: &PrismState,
) -> Option<(smithay::output::Output, Point<f64, Logical>)> {
    let px = state.pointer_pos.x as i32;
    let py = state.pointer_pos.y as i32;
    let output_id = state.output_containing((px, py))?;
    let out = state.wl_outputs.get(&output_id)?.clone();
    let origin = out.current_location();
    let pos_within_output = Point::<f64, Logical>::from((
        state.pointer_pos.x - origin.x as f64,
        state.pointer_pos.y - origin.y as f64,
    ));
    Some((out, pos_within_output))
}

/// Whether the pointer currently rests on a Top/Overlay layer surface
/// (a bar, a launcher). Overview scroll bindings stand down there so
/// the surface keeps its own scroll semantics (niri's
/// `should_handle_in_overview` gate).
///
/// Deliberately narrower than niri's any-mapped-layer check: niri
/// blocks pointer input to backdrop-placed Background layers in the
/// overview, so its wallpaper never holds pointer focus there and the
/// gate never fires for it. Prism's wallpaper IS the backdrop (the
/// place-within-backdrop look) but stays a normal input-receiving
/// layer — gating only Top/Overlay reproduces niri's *effective*
/// behavior: scrolling over the wallpaper between cards switches
/// workspaces.
fn pointer_over_top_or_overlay_layer(state: &PrismState) -> bool {
    let Some((surface, _)) = &state.pointer_contents else {
        return false;
    };
    use smithay::wayland::shell::wlr_layer::Layer;
    for out in state.wl_outputs.values() {
        let map = smithay::desktop::layer_map_for_output(out);
        if let Some(ls) = map.layer_for_surface(surface, smithay::desktop::WindowSurfaceType::ALL) {
            if matches!(ls.layer(), Layer::Top | Layer::Overlay) {
                return true;
            }
        }
    }
    false
}

// ─── helpers ─────────────────────────────────────────────────────

/// Union of all advertised output geometries in global logical
/// coordinates. None if no outputs are advertised.
///
/// Per-output logical size is `physical_mode_size / fractional_scale`,
/// matching `PrismState::output_containing`. The clamp inside
/// `clamp_pointer_to_outputs` keeps the pointer within the union, so
/// per-output scale changes show up immediately as a smaller addressable
/// area on that output.
fn global_bounding_rect(state: &PrismState) -> Option<Rectangle<i32, Logical>> {
    let mut acc: Option<Rectangle<i32, Logical>> = None;
    for output in state.wl_outputs.values() {
        let loc = output.current_location();
        let Some(mode) = output.current_mode() else {
            continue;
        };
        let scale = output.current_scale().fractional_scale().max(0.01);
        let lw = ((mode.size.w as f64) / scale).round() as i32;
        let lh = ((mode.size.h as f64) / scale).round() as i32;
        let size: Size<i32, Logical> = (lw, lh).into();
        let rect = Rectangle::new(loc, size);
        acc = Some(acc.map(|a| a.merge(rect)).unwrap_or(rect));
    }
    acc
}

/// Clamp a point into the union of all output rects. Returns the point
/// unchanged if no outputs are advertised.
fn clamp_point_to_outputs(state: &PrismState, mut p: Point<f64, Logical>) -> Point<f64, Logical> {
    let Some(bounds) = global_bounding_rect(state) else {
        return p;
    };
    let max_x = (bounds.loc.x + bounds.size.w - 1) as f64;
    let max_y = (bounds.loc.y + bounds.size.h - 1) as f64;
    p.x = p.x.clamp(bounds.loc.x as f64, max_x);
    p.y = p.y.clamp(bounds.loc.y as f64, max_y);
    p
}

fn clamp_pointer_to_outputs(state: &mut PrismState) {
    state.pointer_pos = clamp_point_to_outputs(state, state.pointer_pos);
}

/// Look up the surface (and its global origin) under the current
/// pointer position. Delegates to [`PrismState::contents_under`], which
/// resolves layer-shell and layout windows in render order and descends
/// popups + subsurfaces to the deepest input-accepting surface.
fn surface_under_pointer(state: &PrismState) -> Option<(WlSurface, Point<f64, Logical>)> {
    state.contents_under(state.pointer_pos)
}

/// Re-evaluate what's under the pointer after a surface/layout change and
/// deliver enter/leave/motion if it differs from what the client last saw.
///
/// Unlike the motion handlers above, this is driven from the post-dispatch
/// refresh, not from input: a pointer that never moved still needs an
/// `enter` when a window slides, resizes, or restacks under it, or when a
/// subsurface commit changes the input geometry beneath it. Mirrors niri's
/// `update_pointer_contents` (niri.rs:1054) — recompute `contents_under` at
/// the current location, bail if unchanged, otherwise re-send `motion` so
/// smithay emits the right enter/leave pair.
///
/// Skipped while a grab (interactive move/resize, popup, or an implicit
/// button grab) owns the pointer: the grab decides focus, so we leave it
/// alone and re-sync once the grab ends.
pub fn refresh_pointer_focus(state: &mut PrismState) {
    let Some(pointer) = state.seat.get_pointer() else {
        return;
    };
    if pointer.is_grabbed() {
        return;
    }

    let under = surface_under_pointer(state);
    if under == state.pointer_contents {
        return;
    }
    state.pointer_contents = under.clone();

    let serial = SERIAL_COUNTER.next_serial();
    let time = state.clock.now().as_millis() as u32;
    let location = state.pointer_pos;
    pointer.motion(
        state,
        under,
        &MotionEvent {
            location,
            serial,
            time,
        },
    );
    pointer.frame(state);
    // NOTE: focus-follows-mouse is deliberately *not* run here. niri drives
    // ffm only from real pointer-motion events, never from a contents refresh
    // (its `update_pointer_contents`). Re-homing focus when windows move under
    // a *stationary* cursor means that during a column-scroll or
    // workspace-switch animation, whatever tile slides under the cursor gets
    // activated — which re-targets the scroll and cancels it. A window that
    // opens under the cursor still gets focus: via its activation policy in
    // `add_window`, reconciled by the per-frame `update_keyboard_focus` — not
    // via ffm.
    // Contents settled on a (possibly new) surface — give any pointer constraint
    // there a chance to activate immediately, instead of waiting for the next
    // motion event's normal path (which would move the pointer first). Mirrors
    // niri calling `maybe_activate_pointer_constraint` after every contents
    // update (niri.rs:877, :1095); prism previously only did so in the live
    // motion handlers, so a lock that dropped couldn't re-engage until the
    // pointer moved — by which time it may have left the surface.
    state.maybe_activate_pointer_constraint();
    prism_protocols::state::update_output_cursors(state);
}

/// If `input { focus-follows-mouse }` is enabled, update the layout's
/// active monitor + active window + keyboard focus to track the pointer.
///
/// Mirrors niri's `handle_focus_follows_mouse` (niri.rs:6175):
///   - skip when disabled
///   - skip when the pointer is in a grab (drag/resize)
///   - update active output when the pointer crosses output boundaries
///   - activate the window under the pointer (without raising), gated by
///     `should_trigger_focus_follows_mouse_on` (don't cancel an in-progress
///     workspace switch) and `max_scroll_amount` (don't yank the view to
///     activate a barely-visible off-screen window); the per-frame
///     `update_keyboard_focus` reconcile then hands it the keyboard
///
/// Called only from real pointer-motion handlers, never from the contents
/// refresh — see the note in [`refresh_pointer_focus`].
///
/// niri's tab-indicator guard has no analogue here: prism's `contents_under`
/// only resolves `HitType::Input` hits (pointer_focus.rs), so the tab
/// indicator — an `Activate`-only hit — never surfaces a window to ffm in the
/// first place.
fn maybe_focus_follows_mouse(state: &mut PrismState) {
    // Locked: ffm reaches into the layout directly (focus_output /
    // window activation), bypassing the lock-gated contents_under —
    // don't move session focus from under the lock screen.
    if state.is_locked() {
        return;
    }
    let Some(ffm) = state.config.borrow().input.focus_follows_mouse else {
        return;
    };
    if let Some(pointer) = state.seat.get_pointer() {
        if pointer.is_grabbed() {
            return;
        }
    }

    // Output under pointer → active monitor.
    let px = state.pointer_pos.x as i32;
    let py = state.pointer_pos.y as i32;
    if let Some(output_id) = state.output_containing((px, py)) {
        let target_output = state.wl_outputs.get(&output_id).cloned();
        if let Some(out) = target_output.as_ref() {
            if state.layout.active_output() != Some(out) {
                state.layout.focus_output(out);
            }
        }
    }

    // Window under pointer → activate without raising; keyboard focus
    // to its surface. Skip the keyboard re-focus if we're already on
    // the right surface (avoids a no-op enter/leave dance on every
    // motion event inside the same window).
    let under = surface_under_pointer(state);
    if let Some((surface, _)) = under {
        // Don't let focus-follows-mouse refocus while the pointer is over a
        // popup: moving keyboard focus onto a popup surface (or away from the
        // toplevel that owns it) makes grab-less clients like Firefox dismiss
        // their own menus. The menu stays the user's focus until it closes.
        if state.surface_is_popup(&surface) {
            return;
        }
        // A layer-shell surface under the cursor takes on-demand focus
        // directly — no workspace concept applies.
        if state.layer_surface_for(&surface).is_some() {
            focus_surface(state, &surface, false);
            return;
        }
        // A layout window: only activate it if focus-follows-mouse should
        // trigger on its workspace. While a workspace switch animates, the
        // window under a (moving) cursor may belong to the *outgoing*
        // workspace; activating it would cancel the switch. niri guards the
        // same way via `should_trigger_focus_follows_mouse_on`. The per-frame
        // `update_keyboard_focus` reconcile then hands it the keyboard.
        if let Some((mapped, _)) = state.layout.find_window_and_output(&surface) {
            let window = mapped.window.clone();
            if !state.layout.should_trigger_focus_follows_mouse_on(&window) {
                return;
            }
            // Don't let a hover over a sliver of an off-screen window yank the
            // view: if activating it would scroll the viewport by more than the
            // configured fraction of a screen, leave focus alone. niri parity.
            if let Some(threshold) = ffm.max_scroll_amount {
                if state.layout.scroll_amount_to_activate(&window) > threshold.0 {
                    return;
                }
            }
            state.on_demand_layer_focus = None;
            state.layout.activate_window_without_raising(&window);
        }
    }
}

/// Route a click / focus-follows-mouse hit (the surface under the pointer)
/// to keyboard focus.
///
/// Keyboard focus for layout windows is *derived* from the layout's active
/// window by [`PrismState::update_keyboard_focus`], reconciled every frame.
/// So for a window we don't poke keyboard focus directly — we just make it
/// the layout's active window and let that reconcile follow. `raise`
/// distinguishes a click (raise the window above its column-mates) from
/// focus-follows-mouse (activate without raising), mirroring niri.
///
/// Layer-shell surfaces don't live in the layout, so they're handled here:
///   - an `OnDemand` surface is remembered as the on-demand focus (the
///     arbiter holds the keyboard there until it unmaps or focus moves on);
///   - an `Exclusive` surface already holds focus via the arbiter, and a
///     `None`-interactivity surface (bar, wallpaper) never takes the keyboard
///     — both are left untouched.
fn focus_surface(state: &mut PrismState, surface: &WlSurface, raise: bool) {
    use smithay::wayland::shell::wlr_layer::KeyboardInteractivity;

    if let Some(ls) = state.layer_surface_for(surface) {
        if ls.cached_state().keyboard_interactivity == KeyboardInteractivity::OnDemand {
            state.on_demand_layer_focus = Some(ls.wl_surface().clone());
        }
        // Exclusive / None: the arbiter owns it; leave the layout alone.
        return;
    }

    // A layout window (or a child surface of one) — make it the layout's
    // active window; `update_keyboard_focus` hands it the keyboard on the
    // next reconcile. Drop any transient on-demand layer focus first.
    state.on_demand_layer_focus = None;
    let window = state
        .layout
        .find_window_and_output(surface)
        .map(|(mapped, _)| mapped.window.clone());
    if let Some(w) = window {
        if raise {
            state.layout.activate_window(&w);
        } else {
            state.layout.activate_window_without_raising(&w);
        }
    }
}

// ─── WLCS test-harness entry points ──────────────────────────────
//
// The handlers above are generic over the libinput backend
// (`I::PointerMotionEvent`, …). WLCS has no libinput events — only
// already-decoded coordinates and button codes — so these thin entry
// points drive the same `state.pointer_pos` → `surface_under_pointer` →
// seat path. That makes WLCS observe exactly prism's real pointer
// behavior, including its current toplevel-only / output-origin hit-test
// approximation. Used only by the prism-wlcs harness.

/// Warp the pointer to an absolute global logical position and deliver the
/// resulting motion (and any enter/leave) to clients.
pub fn wlcs_pointer_motion_absolute(
    state: &mut PrismState,
    location: Point<f64, Logical>,
    time: u32,
) {
    state.pointer_pos = location;
    wlcs_pointer_reposition(state, time);
}

/// Move the pointer by a relative delta and deliver the resulting motion.
pub fn wlcs_pointer_motion_relative(state: &mut PrismState, delta: Point<f64, Logical>, time: u32) {
    state.pointer_pos.x += delta.x;
    state.pointer_pos.y += delta.y;
    wlcs_pointer_reposition(state, time);
}

fn wlcs_pointer_reposition(state: &mut PrismState, time: u32) {
    let serial = SERIAL_COUNTER.next_serial();
    clamp_pointer_to_outputs(state);
    let focus = surface_under_pointer(state);
    // Keep the tracked contents in sync so the post-dispatch
    // `refresh_pointer_focus` doesn't see a spurious change and re-fire.
    state.pointer_contents = focus.clone();
    let new_pos = state.pointer_pos;
    let Some(pointer) = state.seat.get_pointer() else {
        return;
    };
    pointer.motion(
        state,
        focus,
        &MotionEvent {
            location: new_pos,
            serial,
            time,
        },
    );
    pointer.frame(state);
    maybe_focus_follows_mouse(state);
    prism_protocols::state::update_output_cursors(state);
}

/// Press or release a pointer button by raw linux event code (e.g.
/// `BTN_LEFT = 0x110`). On press, focus follows to the surface under the
/// pointer, mirroring [`on_pointer_button`] minus the Mod+click grab
/// triggers (WLCS never sends modifiers).
pub fn wlcs_pointer_button(state: &mut PrismState, button: u32, pressed: bool, time: u32) {
    let serial = SERIAL_COUNTER.next_serial();
    let Some(pointer) = state.seat.get_pointer() else {
        return;
    };
    if pressed && !pointer.is_grabbed() {
        if let Some((surface, _)) = surface_under_pointer(state) {
            focus_surface(state, &surface, true);
        }
    }
    pointer.button(
        state,
        &ButtonEvent {
            button,
            state: if pressed {
                ButtonState::Pressed
            } else {
                ButtonState::Released
            },
            serial,
            time,
        },
    );
    pointer.frame(state);
}
