//! Touchpad gesture handlers — niri `input/mod.rs` `on_gesture_*` port.
//!
//! Swipe gestures are where the compositor behavior lives:
//!
//!   - **3-finger swipe**: workspace switch (vertical) or view offset /
//!     column scroll (horizontal). The direction is decided once the
//!     cumulative travel passes a 16px threshold (GNOME Shell's), then
//!     the matching layout gesture runs for the rest of the swipe.
//!   - **4-finger swipe**: drives the overview zoom continuously
//!     (`Layout::overview_gesture_*` — rubber-banded, velocity-snapped
//!     on release).
//!
//! Anything the compositor doesn't consume is forwarded to the focused
//! client via `zwp_pointer_gestures_v1` (the seat pointer's `gesture_*`
//! methods); pinch and hold are pure forwarding.
//!
//! Deltas use libinput's unaccelerated values when available, and the
//! 3-finger/view gestures honor the device's natural-scroll setting —
//! the overview gesture deliberately does NOT (zoom direction is
//! spatial, not scroll-like), matching niri's `uninverted_delta_y`.

use std::any::Any;
use std::time::Duration;

use prism_protocols::PrismState;
use smithay::backend::input::{Event, GestureBeginEvent, GestureEndEvent};
use smithay::input::pointer::{
    GestureHoldBeginEvent, GestureHoldEndEvent, GesturePinchBeginEvent, GesturePinchEndEvent,
    GesturePinchUpdateEvent, GestureSwipeBeginEvent, GestureSwipeEndEvent, GestureSwipeUpdateEvent,
};
use smithay::utils::SERIAL_COUNTER;

use crate::backend_ext::PrismInputBackend;
use crate::move_grab::queue_redraw_all;
use crate::pointer::refresh_pointer_focus;

pub fn on_gesture_swipe_begin<I: PrismInputBackend>(
    state: &mut PrismState,
    event: I::GestureSwipeBeginEvent,
) {
    if event.fingers() == 3 {
        state.gesture_swipe_3f_cumulative = Some((0., 0.));
        // We handled this event.
        return;
    } else if event.fingers() == 4 {
        state.layout.overview_gesture_begin();
        queue_redraw_all(state);
        // We handled this event.
        return;
    }

    let serial = SERIAL_COUNTER.next_serial();
    let Some(pointer) = state.seat.get_pointer() else {
        return;
    };
    refresh_pointer_focus(state);
    pointer.gesture_swipe_begin(
        state,
        &GestureSwipeBeginEvent {
            serial,
            time: event.time_msec(),
            fingers: event.fingers(),
        },
    );
}

pub fn on_gesture_swipe_update<I: PrismInputBackend + 'static>(
    state: &mut PrismState,
    event: I::GestureSwipeUpdateEvent,
) where
    I::Device: 'static,
{
    use smithay::backend::input::GestureSwipeUpdateEvent as _;

    let mut delta_x = event.delta_x();
    let mut delta_y = event.delta_y();

    // Prefer libinput's unaccelerated deltas: gesture distances should
    // be physical finger travel, not pointer-acceleration output.
    if let Some(libinput_event) =
        (&event as &dyn Any).downcast_ref::<input::event::gesture::GestureSwipeUpdateEvent>()
    {
        use input::event::gesture::GestureEventCoordinates as _;
        delta_x = libinput_event.dx_unaccelerated();
        delta_y = libinput_event.dy_unaccelerated();
    }

    // The overview zoom is a spatial gesture, not a scroll: it ignores
    // natural-scroll inversion (niri's `uninverted_delta_y`).
    let uninverted_delta_y = delta_y;

    let device = event.device();
    if let Some(device) = (&device as &dyn Any).downcast_ref::<input::Device>() {
        if device.config_scroll_natural_scroll_enabled() {
            delta_x = -delta_x;
            delta_y = -delta_y;
        }
    }

    let is_overview_open = state.layout.is_overview_open();

    // Undecided 3-finger swipe: accumulate until the 16px decision
    // threshold, then start the gesture matching the dominant axis.
    if let Some((cx, cy)) = &mut state.gesture_swipe_3f_cumulative {
        *cx += delta_x;
        *cy += delta_y;

        let (cx, cy) = (*cx, *cy);
        if cx * cx + cy * cy >= 16. * 16. {
            state.gesture_swipe_3f_cumulative = None;

            if let Some((out, pos_within_output)) = crate::pointer::output_under_pointer(state) {
                if cx.abs() > cy.abs() {
                    // Horizontal: view offset. In the overview the
                    // workspace under the cursor (extended, full-width
                    // bounds) is the target; otherwise the active one —
                    // hit-testing during animations could catch the
                    // wrong workspace (niri's comment).
                    let ws_id = if is_overview_open {
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
                    if let Some(ws_id) = ws_id {
                        if let Some((ws_idx, _)) = state.layout.find_workspace_by_id(ws_id) {
                            state
                                .layout
                                .view_offset_gesture_begin(&out, Some(ws_idx), true);
                        }
                    }
                } else {
                    state.layout.workspace_switch_gesture_begin(&out, true);
                }
            }
        }
    }

    let timestamp = Duration::from_micros(event.time());

    // `Some(_)` = the gesture is ongoing (event consumed);
    // `Some(Some(_))` = and something changed visually (redraw).
    let mut handled = false;
    let res = state
        .layout
        .workspace_switch_gesture_update(delta_y, timestamp, true);
    if let Some(output) = res {
        if output.is_some() {
            queue_redraw_all(state);
        }
        handled = true;
    }

    let res = state
        .layout
        .view_offset_gesture_update(delta_x, timestamp, true);
    if let Some(output) = res {
        if output.is_some() {
            queue_redraw_all(state);
        }
        handled = true;
    }

    let res = state
        .layout
        .overview_gesture_update(-uninverted_delta_y, timestamp);
    if let Some(redraw) = res {
        if redraw {
            queue_redraw_all(state);
        }
        handled = true;
    }

    if handled {
        // We handled this event.
        return;
    }

    let Some(pointer) = state.seat.get_pointer() else {
        return;
    };
    refresh_pointer_focus(state);
    pointer.gesture_swipe_update(
        state,
        &GestureSwipeUpdateEvent {
            time: event.time_msec(),
            delta: event.delta(),
        },
    );
}

pub fn on_gesture_swipe_end<I: PrismInputBackend>(
    state: &mut PrismState,
    event: I::GestureSwipeEndEvent,
) {
    state.gesture_swipe_3f_cumulative = None;

    let mut handled = false;
    if state
        .layout
        .workspace_switch_gesture_end(Some(true))
        .is_some()
    {
        queue_redraw_all(state);
        handled = true;
    }

    if state.layout.view_offset_gesture_end(Some(true)).is_some() {
        queue_redraw_all(state);
        handled = true;
    }

    if state.layout.overview_gesture_end() {
        queue_redraw_all(state);
        handled = true;
    }

    if handled {
        // We handled this event.
        return;
    }

    let serial = SERIAL_COUNTER.next_serial();
    let Some(pointer) = state.seat.get_pointer() else {
        return;
    };
    refresh_pointer_focus(state);
    pointer.gesture_swipe_end(
        state,
        &GestureSwipeEndEvent {
            serial,
            time: event.time_msec(),
            cancelled: event.cancelled(),
        },
    );
}

pub fn on_gesture_pinch_begin<I: PrismInputBackend>(
    state: &mut PrismState,
    event: I::GesturePinchBeginEvent,
) {
    let serial = SERIAL_COUNTER.next_serial();
    let Some(pointer) = state.seat.get_pointer() else {
        return;
    };
    refresh_pointer_focus(state);
    pointer.gesture_pinch_begin(
        state,
        &GesturePinchBeginEvent {
            serial,
            time: event.time_msec(),
            fingers: event.fingers(),
        },
    );
}

pub fn on_gesture_pinch_update<I: PrismInputBackend>(
    state: &mut PrismState,
    event: I::GesturePinchUpdateEvent,
) {
    use smithay::backend::input::GesturePinchUpdateEvent as _;

    let Some(pointer) = state.seat.get_pointer() else {
        return;
    };
    refresh_pointer_focus(state);
    pointer.gesture_pinch_update(
        state,
        &GesturePinchUpdateEvent {
            time: event.time_msec(),
            delta: event.delta(),
            scale: event.scale(),
            rotation: event.rotation(),
        },
    );
}

pub fn on_gesture_pinch_end<I: PrismInputBackend>(
    state: &mut PrismState,
    event: I::GesturePinchEndEvent,
) {
    let serial = SERIAL_COUNTER.next_serial();
    let Some(pointer) = state.seat.get_pointer() else {
        return;
    };
    refresh_pointer_focus(state);
    pointer.gesture_pinch_end(
        state,
        &GesturePinchEndEvent {
            serial,
            time: event.time_msec(),
            cancelled: event.cancelled(),
        },
    );
}

pub fn on_gesture_hold_begin<I: PrismInputBackend>(
    state: &mut PrismState,
    event: I::GestureHoldBeginEvent,
) {
    let serial = SERIAL_COUNTER.next_serial();
    let Some(pointer) = state.seat.get_pointer() else {
        return;
    };
    refresh_pointer_focus(state);
    pointer.gesture_hold_begin(
        state,
        &GestureHoldBeginEvent {
            serial,
            time: event.time_msec(),
            fingers: event.fingers(),
        },
    );
}

pub fn on_gesture_hold_end<I: PrismInputBackend>(
    state: &mut PrismState,
    event: I::GestureHoldEndEvent,
) {
    let serial = SERIAL_COUNTER.next_serial();
    let Some(pointer) = state.seat.get_pointer() else {
        return;
    };
    refresh_pointer_focus(state);
    pointer.gesture_hold_end(
        state,
        &GestureHoldEndEvent {
            serial,
            time: event.time_msec(),
            cancelled: event.cancelled(),
        },
    );
}
