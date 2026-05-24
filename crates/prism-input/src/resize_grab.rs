//! Interactive resize grab — Mod+RightClick+drag a window edge.
//!
//! Direct port of niri's `input/resize_grab.rs`. The grab is started
//! from `on_pointer_button` after `Layout::resize_edges_under` picks
//! the edge(s) closest to the cursor (so dragging from a corner does
//! 2-axis resize, near a side does 1-axis).
//!
//! On every motion the absolute delta from the grab origin is fed to
//! `Layout::interactive_resize_update`, which clamps to the window's
//! min/max size and the workspace's column rules.

use prism_protocols::PrismState;
use smithay::desktop::Window;
use smithay::input::pointer::{
    AxisFrame, ButtonEvent, GestureHoldBeginEvent, GestureHoldEndEvent, GesturePinchBeginEvent,
    GesturePinchEndEvent, GesturePinchUpdateEvent, GestureSwipeBeginEvent, GestureSwipeEndEvent,
    GestureSwipeUpdateEvent, GrabStartData, MotionEvent, PointerGrab, PointerInnerHandle,
    RelativeMotionEvent,
};
use smithay::input::SeatHandler;
use smithay::utils::{IsAlive, Logical, Point};

use crate::move_grab::queue_redraw_all;

pub struct ResizeGrab {
    start_data: GrabStartData<PrismState>,
    window: Window,
}

impl ResizeGrab {
    pub fn new(start_data: GrabStartData<PrismState>, window: Window) -> Self {
        Self { start_data, window }
    }

    fn end(&mut self, state: &mut PrismState) {
        state.layout.interactive_resize_end(&self.window);
        queue_redraw_all(state);
    }
}

impl PointerGrab<PrismState> for ResizeGrab {
    fn motion(
        &mut self,
        data: &mut PrismState,
        handle: &mut PointerInnerHandle<'_, PrismState>,
        _focus: Option<(
            <PrismState as SeatHandler>::PointerFocus,
            Point<f64, Logical>,
        )>,
        event: &MotionEvent,
    ) {
        handle.motion(data, None, event);
        if !self.window.alive() {
            handle.unset_grab(self, data, event.serial, event.time, true);
            return;
        }
        let delta = event.location - self.start_data.location;
        let ongoing = data.layout.interactive_resize_update(&self.window, delta);
        if !ongoing {
            handle.unset_grab(self, data, event.serial, event.time, true);
            return;
        }
        queue_redraw_all(data);
    }

    fn relative_motion(
        &mut self,
        data: &mut PrismState,
        handle: &mut PointerInnerHandle<'_, PrismState>,
        _focus: Option<(
            <PrismState as SeatHandler>::PointerFocus,
            Point<f64, Logical>,
        )>,
        event: &RelativeMotionEvent,
    ) {
        handle.relative_motion(data, None, event);
    }

    fn button(
        &mut self,
        data: &mut PrismState,
        handle: &mut PointerInnerHandle<'_, PrismState>,
        event: &ButtonEvent,
    ) {
        handle.button(data, event);
        if handle.current_pressed().is_empty() {
            handle.unset_grab(self, data, event.serial, event.time, true);
        }
    }

    fn axis(
        &mut self,
        data: &mut PrismState,
        handle: &mut PointerInnerHandle<'_, PrismState>,
        details: AxisFrame,
    ) {
        handle.axis(data, details);
    }

    fn frame(&mut self, data: &mut PrismState, handle: &mut PointerInnerHandle<'_, PrismState>) {
        handle.frame(data);
    }

    fn gesture_swipe_begin(
        &mut self,
        data: &mut PrismState,
        handle: &mut PointerInnerHandle<'_, PrismState>,
        event: &GestureSwipeBeginEvent,
    ) {
        handle.gesture_swipe_begin(data, event);
    }

    fn gesture_swipe_update(
        &mut self,
        data: &mut PrismState,
        handle: &mut PointerInnerHandle<'_, PrismState>,
        event: &GestureSwipeUpdateEvent,
    ) {
        handle.gesture_swipe_update(data, event);
    }

    fn gesture_swipe_end(
        &mut self,
        data: &mut PrismState,
        handle: &mut PointerInnerHandle<'_, PrismState>,
        event: &GestureSwipeEndEvent,
    ) {
        handle.gesture_swipe_end(data, event);
    }

    fn gesture_pinch_begin(
        &mut self,
        data: &mut PrismState,
        handle: &mut PointerInnerHandle<'_, PrismState>,
        event: &GesturePinchBeginEvent,
    ) {
        handle.gesture_pinch_begin(data, event);
    }

    fn gesture_pinch_update(
        &mut self,
        data: &mut PrismState,
        handle: &mut PointerInnerHandle<'_, PrismState>,
        event: &GesturePinchUpdateEvent,
    ) {
        handle.gesture_pinch_update(data, event);
    }

    fn gesture_pinch_end(
        &mut self,
        data: &mut PrismState,
        handle: &mut PointerInnerHandle<'_, PrismState>,
        event: &GesturePinchEndEvent,
    ) {
        handle.gesture_pinch_end(data, event);
    }

    fn gesture_hold_begin(
        &mut self,
        data: &mut PrismState,
        handle: &mut PointerInnerHandle<'_, PrismState>,
        event: &GestureHoldBeginEvent,
    ) {
        handle.gesture_hold_begin(data, event);
    }

    fn gesture_hold_end(
        &mut self,
        data: &mut PrismState,
        handle: &mut PointerInnerHandle<'_, PrismState>,
        event: &GestureHoldEndEvent,
    ) {
        handle.gesture_hold_end(data, event);
    }

    fn start_data(&self) -> &GrabStartData<PrismState> {
        &self.start_data
    }

    fn unset(&mut self, data: &mut PrismState) {
        self.end(data);
    }
}
