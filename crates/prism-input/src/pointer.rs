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
//! What's intentionally not here (yet):
//!   - Hot corners
//!   - Tablet integration
//!   - Move/resize/spatial/pick_window/pick_color grabs (niri's 7 grab files)
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

    // Click-to-focus: on press, take keyboard focus to the surface
    // under the pointer AND make that surface's output the layout's
    // active monitor. Without the focus_output call the focus ring
    // would stay drawn on whichever output happened to be active at
    // startup (typically the first in connector-name sort order,
    // DP-4 on the current hardware), even when the user clicks
    // somewhere else. niri runs the same `focus_output` from its
    // input handlers.
    if state_pressed && !pointer.is_grabbed() {
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
                // Resolve the surface's output for focus_output.
                let output_for_focus = state
                    .layout
                    .find_window_and_output(&surface)
                    .and_then(|(_, out)| out.cloned());
                set_keyboard_focus(state, Some(surface));
                if let Some(out) = output_for_focus {
                    state.layout.focus_output(&out);
                }
            }
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
    let px = state.pointer_pos.x as i32;
    let py = state.pointer_pos.y as i32;
    let Some(output_id) = state.output_containing((px, py)) else {
        return false;
    };
    let Some(out) = state.wl_outputs.get(&output_id).cloned() else {
        return false;
    };
    let origin = out.current_location();
    let pos_within_output = Point::<f64, Logical>::from((
        state.pointer_pos.x - origin.x as f64,
        state.pointer_pos.y - origin.y as f64,
    ));
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
    let mut frame = AxisFrame::new(time).source(source);

    for axis in [Axis::Horizontal, Axis::Vertical] {
        if let Some(discrete) = event.amount_v120(axis) {
            // v120 increments are smithay's preferred high-resolution
            // discrete scroll signal.
            frame = frame.v120(axis, discrete as i32);
        }
        if let Some(amount) = event.amount(axis) {
            frame = frame.value(axis, amount);
        } else if let Some(amount_discrete) = event.amount_v120(axis) {
            // Some backends only give discrete; convert to a smooth
            // value at ~10 px per notch, niri's default ratio.
            frame = frame.value(axis, amount_discrete / 120.0 * 10.0);
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
    // The contents under a *stationary* pointer just changed — e.g. a window
    // opened/closed/restacked beneath it. With focus-follows-mouse this must
    // re-home keyboard focus, exactly as a real motion event would: otherwise a
    // window mapped under the cursor stays unfocused until the user wiggles the
    // mouse. `maybe_focus_follows_mouse` no-ops when ffm is disabled, and the
    // early return above means this only runs when something actually changed —
    // a window opening *elsewhere* leaves the contents here unchanged and never
    // steals focus.
    maybe_focus_follows_mouse(state);
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
///   - activate the window under the pointer (without raising), and
///     hand keyboard focus to its surface
///
/// `max_scroll_amount` (ignore the focus update while the cursor is
/// inside a scrolling viewport beyond N% of one screen) is not yet
/// honored — needs scroll-amount tracking that we don't have.
fn maybe_focus_follows_mouse(state: &mut PrismState) {
    let ffm = state.config.borrow().input.focus_follows_mouse;
    if ffm.is_none() {
        return;
    }
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
        let window = state
            .layout
            .find_window_and_output(&surface)
            .map(|(mapped, _)| mapped.window.clone());
        if let Some(w) = window {
            state.layout.activate_window_without_raising(&w);
        }
        set_keyboard_focus(state, Some(surface));
    }
}

/// Route a click / focus-follows-mouse hit to the focus arbiter.
///
/// `surface` is whatever lies under the pointer — a window, a layer-shell
/// surface, or one of their subsurfaces/popups (or `None` for empty space).
/// We don't set keyboard focus directly; we update the arbiter's inputs and
/// let [`PrismState::update_keyboard_focus`] decide:
///   - a layout window (or empty space) becomes the layout's focus and drops
///     any transient on-demand layer focus;
///   - an `OnDemand` layer surface is remembered as the on-demand focus;
///   - an `Exclusive` layer surface already holds focus via the arbiter, and
///     a `None`-interactivity surface (bar, wallpaper) is left alone — neither
///     disturbs the layout's focused window.
///
/// The arbiter handles the enter/leave round-trip (and skips it when the
/// effective surface is unchanged).
fn set_keyboard_focus(state: &mut PrismState, surface: Option<WlSurface>) {
    use smithay::wayland::shell::wlr_layer::KeyboardInteractivity;

    match surface {
        Some(surf) => match state.layer_surface_for(&surf) {
            Some(ls) => {
                if ls.cached_state().keyboard_interactivity == KeyboardInteractivity::OnDemand {
                    state.on_demand_layer_focus = Some(ls.wl_surface().clone());
                }
                // Exclusive surfaces already hold focus via the arbiter;
                // None-interactivity surfaces never take the keyboard. Either
                // way, leave the layout's focused window untouched.
            }
            None => {
                // A layout window (or a child surface of one).
                state.layout_focus_surface = Some(surf);
                state.on_demand_layer_focus = None;
            }
        },
        None => {
            state.layout_focus_surface = None;
            state.on_demand_layer_focus = None;
        }
    }
    state.update_keyboard_focus();
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
            let output_for_focus = state
                .layout
                .find_window_and_output(&surface)
                .and_then(|(_, out)| out.cloned());
            set_keyboard_focus(state, Some(surface));
            if let Some(out) = output_for_focus {
                state.layout.focus_output(&out);
            }
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
