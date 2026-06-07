//! Spatial-movement grab — drag to pan the view or switch workspaces.
//!
//! Port of niri's `input/spatial_movement_grab.rs`. Installed from two
//! pointer triggers:
//!
//!   - **RMB drag in the overview**: starts directly in `ViewOffset`
//!     (the workspace view pans horizontally under the cursor; the
//!     view-offset gesture was begun by the button handler).
//!   - **Mod+MMB drag** (anywhere): starts in `Recognizing`; after 8px
//!     (GTK 4's threshold) the dominant axis picks `ViewOffset`
//!     (horizontal) or `WorkspaceSwitch` (vertical).
//!
//! Motion is accumulated per event and applied once per pointer
//! *frame* (niri batches the same way); deltas are negated because
//! dragging the content moves it with the cursor, opposite to scroll
//! direction. Relative motion takes precedence over absolute motion
//! when both arrive in one frame.
//!
//! Divergences from niri, by design: no `view_offset_output` /
//! `workspace_switch_output` accessors (niri uses them for granular
//! per-output redraws; prism redraws whole-world), and the grab tracks
//! `last_time_msec` instead of calling a monotonic-clock helper for
//! the synthetic unset timestamp.

use std::time::Duration;

use prism_layout::layout::WorkspaceId;
use prism_protocols::PrismState;
use smithay::input::pointer::{
    AxisFrame, ButtonEvent, CursorImageStatus, GestureHoldBeginEvent, GestureHoldEndEvent,
    GesturePinchBeginEvent, GesturePinchEndEvent, GesturePinchUpdateEvent, GestureSwipeBeginEvent,
    GestureSwipeEndEvent, GestureSwipeUpdateEvent, GrabStartData, MotionEvent, PointerGrab,
    PointerInnerHandle, RelativeMotionEvent,
};
use smithay::input::SeatHandler;
use smithay::output::Output;
use smithay::utils::{Logical, Point, SERIAL_COUNTER};

use crate::move_grab::queue_redraw_all;

pub struct SpatialMovementGrab {
    start_data: GrabStartData<PrismState>,
    last_location: Point<f64, Logical>,
    output: Output,
    workspace_id: WorkspaceId,
    gesture: GestureState,

    // Accumulated and applied in frame().
    new_location: Point<f64, Logical>,
    event_timestamp: Option<Duration>,
    relative_delta: Option<Point<f64, Logical>>,
    last_time_msec: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GestureState {
    Recognizing,
    ViewOffset,
    WorkspaceSwitch,
}

impl SpatialMovementGrab {
    pub fn new(
        start_data: GrabStartData<PrismState>,
        output: Output,
        workspace_id: WorkspaceId,
        is_view_offset: bool,
    ) -> Self {
        let location = start_data.location;
        let gesture = if is_view_offset {
            GestureState::ViewOffset
        } else {
            GestureState::Recognizing
        };

        Self {
            last_location: location,
            start_data,
            output,
            workspace_id,
            gesture,
            new_location: location,
            event_timestamp: None,
            relative_delta: None,
            last_time_msec: 0,
        }
    }

    /// Apply the frame's accumulated motion to the active gesture.
    /// Returns `false` when the gesture is no longer ongoing and the
    /// grab should release itself.
    fn on_frame(&mut self, state: &mut PrismState) -> bool {
        let Some(timestamp) = self.event_timestamp.take() else {
            return true;
        };

        let delta = self
            .relative_delta
            .take()
            .unwrap_or(self.new_location - self.last_location);
        self.last_location = self.new_location;

        let layout = &mut state.layout;
        let res = match self.gesture {
            GestureState::Recognizing => {
                let c = self.new_location - self.start_data.location;

                // Check if the gesture moved far enough to decide.
                // Threshold copied from GTK 4 (niri's comment).
                if c.x * c.x + c.y * c.y >= 8. * 8. {
                    if c.x.abs() > c.y.abs() {
                        self.gesture = GestureState::ViewOffset;
                        if let Some((ws_idx, ws)) = layout.find_workspace_by_id(self.workspace_id) {
                            if ws.current_output() == Some(&self.output) {
                                layout.view_offset_gesture_begin(&self.output, Some(ws_idx), false);
                                layout.view_offset_gesture_update(-c.x, timestamp, false)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        self.gesture = GestureState::WorkspaceSwitch;
                        layout.workspace_switch_gesture_begin(&self.output, false);
                        layout.workspace_switch_gesture_update(-c.y, timestamp, false)
                    }
                } else {
                    Some(None)
                }
            }
            GestureState::ViewOffset => {
                layout.view_offset_gesture_update(-delta.x, timestamp, false)
            }
            GestureState::WorkspaceSwitch => {
                layout.workspace_switch_gesture_update(-delta.y, timestamp, false)
            }
        };

        if let Some(output) = res {
            if output.is_some() {
                queue_redraw_all(state);
            }
            true
        } else {
            false
        }
    }

    fn on_ungrab(&mut self, state: &mut PrismState) {
        let layout = &mut state.layout;
        let res = match self.gesture {
            GestureState::Recognizing => None,
            GestureState::ViewOffset => layout.view_offset_gesture_end(Some(false)),
            GestureState::WorkspaceSwitch => layout.workspace_switch_gesture_end(Some(false)),
        };

        if res.is_some() {
            queue_redraw_all(state);
        }

        state
            .cursor_manager
            .set_cursor_image(CursorImageStatus::default_named());
        state.cursor_dirty = true;
        prism_protocols::state::update_output_cursors(state);
    }
}

impl PointerGrab<PrismState> for SpatialMovementGrab {
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
        // While the grab is active, no client has pointer focus.
        handle.motion(data, None, event);

        self.new_location = event.location;
        self.last_time_msec = event.time;

        // Relative motion takes precedence over normal motion.
        if self.relative_delta.is_none() {
            self.event_timestamp = Some(Duration::from_millis(u64::from(event.time)));
        }
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
        // While the grab is active, no client has pointer focus.
        handle.relative_motion(data, None, event);

        *self.relative_delta.get_or_insert_default() += event.delta;
        self.event_timestamp = Some(Duration::from_micros(event.utime));
    }

    fn button(
        &mut self,
        data: &mut PrismState,
        handle: &mut PointerInnerHandle<'_, PrismState>,
        event: &ButtonEvent,
    ) {
        handle.button(data, event);
        self.last_time_msec = event.time;

        if handle.current_pressed().is_empty() {
            // No more buttons are pressed, release the grab.
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

        if !self.on_frame(data) {
            // The gesture is no longer ongoing.
            let time = self.last_time_msec;
            handle.unset_grab(self, data, SERIAL_COUNTER.next_serial(), time, true);
        }
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
        self.on_ungrab(data);
    }
}
