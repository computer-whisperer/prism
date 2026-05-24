//! Pointer dispatch — motion, button, axis.
//!
//! Minimal MVP port of niri's `on_pointer_motion` /
//! `on_pointer_motion_absolute` / `on_pointer_button` /
//! `on_pointer_axis` (input/mod.rs lines 2414, 2658, 2750, 3074).
//!
//! What's intentionally not here (yet):
//!   - Pointer constraints / locked / confined regions
//!   - Hot corners
//!   - Tablet integration
//!   - Move/resize/spatial/pick_window/pick_color grabs (niri's 7 grab files)
//!   - Follow-pointer focus / MRU click-to-focus
//!   - Cursor auto-hide / pointer-inactivity timer
//!   - Sub-surface hit-testing (we hand the toplevel surface and treat
//!     the window placement as the surface origin)
//!
//! These can all bolt onto this file as their backing state lands.

use prism_protocols::PrismState;
use smithay::backend::input::{
    AbsolutePositionEvent, Axis, AxisSource, ButtonState, PointerAxisEvent, PointerButtonEvent,
    PointerMotionEvent,
};
use smithay::input::pointer::{AxisFrame, ButtonEvent, MotionEvent};
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::Resource;
use smithay::utils::{Logical, Point, Rectangle, Size, SERIAL_COUNTER};

use crate::backend_ext::PrismInputBackend;

pub fn on_pointer_motion<I: PrismInputBackend>(
    state: &mut PrismState,
    event: I::PointerMotionEvent,
) {
    let serial = SERIAL_COUNTER.next_serial();
    let time = smithay::backend::input::Event::time_msec(&event);

    // Advance the global pointer by the relative delta, then clamp
    // into the union of all output rects. Without clamping the
    // pointer can drift forever and the focus query stops finding
    // anything.
    state.pointer_pos.x += event.delta_x();
    state.pointer_pos.y += event.delta_y();
    clamp_pointer_to_outputs(state);

    let focus = surface_under_pointer(state);
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
    maybe_focus_follows_mouse(state, serial);
    // Walk the cursor plane on every output: show on the output the
    // pointer is in, hide on the rest, queue redraws on changes.
    prism_protocols::state::update_output_cursors(state);
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
    maybe_focus_follows_mouse(state, serial);
    prism_protocols::state::update_output_cursors(state);
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
            // Resolve the surface's output for focus_output.
            let output_for_focus = state
                .layout
                .find_window_and_output(&surface)
                .and_then(|(_, out)| out.cloned());
            set_keyboard_focus(state, Some(surface), serial);
            if let Some(out) = output_for_focus {
                state.layout.focus_output(&out);
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
            state: event.state().into(),
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

fn clamp_pointer_to_outputs(state: &mut PrismState) {
    let Some(bounds) = global_bounding_rect(state) else {
        return;
    };
    let max_x = (bounds.loc.x + bounds.size.w - 1) as f64;
    let max_y = (bounds.loc.y + bounds.size.h - 1) as f64;
    let min_x = bounds.loc.x as f64;
    let min_y = bounds.loc.y as f64;
    state.pointer_pos.x = state.pointer_pos.x.clamp(min_x, max_x);
    state.pointer_pos.y = state.pointer_pos.y.clamp(min_y, max_y);
}

/// Look up the surface (and its global origin) under the current
/// pointer position. MVP: returns the focused window's toplevel
/// surface; sub-surface walk + accurate origin land with popup /
/// sub-surface dispatch.
fn surface_under_pointer(state: &PrismState) -> Option<(WlSurface, Point<f64, Logical>)> {
    let px = state.pointer_pos.x as i32;
    let py = state.pointer_pos.y as i32;
    let output_id = state.output_containing((px, py))?;
    let output = state.wl_outputs.get(&output_id)?;
    let output_loc = output.current_location();
    let pos_within = Point::<f64, Logical>::from((
        state.pointer_pos.x - output_loc.x as f64,
        state.pointer_pos.y - output_loc.y as f64,
    ));
    let (mapped, _hit) = state.layout.window_under(output, pos_within)?;
    let toplevel = mapped.toplevel().clone();
    let wl_surface = toplevel.wl_surface().clone();
    // TODO(pointer hit-testing): use the layout's per-tile geometry
    // to derive the actual surface origin (output_loc + tile.loc).
    // Today we approximate as the output origin; works for the
    // common single-tile / full-output case.
    let surface_origin = Point::from((output_loc.x as f64, output_loc.y as f64));
    Some((wl_surface, surface_origin))
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
fn maybe_focus_follows_mouse(state: &mut PrismState, serial: smithay::utils::Serial) {
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
        let window = state
            .layout
            .find_window_and_output(&surface)
            .map(|(mapped, _)| mapped.window.clone());
        if let Some(w) = window {
            state.layout.activate_window_without_raising(&w);
        }
        set_keyboard_focus(state, Some(surface), serial);
    }
}

/// Move keyboard focus to `surface` (or clear it if None), sending
/// the usual enter/leave dance via smithay's KeyboardHandle.
fn set_keyboard_focus(
    state: &mut PrismState,
    surface: Option<WlSurface>,
    serial: smithay::utils::Serial,
) {
    use prism_protocols::input_state::KeyboardFocus;

    // Quick equality check: if focus is already on this surface,
    // skip the syscalls.
    let already_focused = match (&state.keyboard_focus, &surface) {
        (KeyboardFocus::Layout { surface: Some(a) }, Some(b)) => a.id() == b.id(),
        (KeyboardFocus::Layout { surface: None }, None) => true,
        _ => false,
    };
    if already_focused {
        return;
    }

    state.keyboard_focus = KeyboardFocus::Layout {
        surface: surface.clone(),
    };
    if let Some(kb) = state.seat.get_keyboard() {
        kb.set_focus(state, surface, serial);
    }
}
