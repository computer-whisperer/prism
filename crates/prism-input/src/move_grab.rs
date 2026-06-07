//! Interactive move grab — Mod+LeftClick+drag a window with the pointer.
//!
//! Cut-down port of niri's `input/move_grab.rs`. Drops the view-offset
//! gesture (drag horizontally to scroll the workspace instead of moving
//! the window), the floating toggle on the opposite mouse button, and
//! the touch case. Those layer on top of the basic move once their
//! backing subsystems land.
//!
//! Overview integration: niri's grab tracks a `GestureState` whose
//! `Recognizing` stage (release before an 8px threshold = a click)
//! activates the window and zooms to its workspace. Prism's grab
//! begins the layout move on construction instead (entering the
//! layout's `Starting` stage immediately), but keeps niri's 8px
//! grab-side gate in `update()` — motion below it never reaches the
//! layout, so on release `Layout::interactive_move_is_starting`
//! answers the click-vs-drag question (see `end()`).
//!
//! What this *does* do:
//!   - Installs a smithay `PointerGrab` for the duration of the drag,
//!     so motion events route exclusively to the grab and don't leak
//!     to surfaces under the moving window.
//!   - Calls `Layout::interactive_move_begin` on construction, then
//!     `interactive_move_update` on every motion + `interactive_move_end`
//!     on release.
//!   - Lets the cursor cross output boundaries — the layout's
//!     `interactive_move_update` already handles "moved to a different
//!     output" by transferring the moving tile.

use prism_protocols::PrismState;
use smithay::desktop::Window;
use smithay::input::pointer::{
    AxisFrame, ButtonEvent, GestureHoldBeginEvent, GestureHoldEndEvent, GesturePinchBeginEvent,
    GesturePinchEndEvent, GesturePinchUpdateEvent, GestureSwipeBeginEvent, GestureSwipeEndEvent,
    GestureSwipeUpdateEvent, GrabStartData, MotionEvent, PointerGrab, PointerInnerHandle,
    RelativeMotionEvent,
};
use smithay::input::SeatHandler;
use smithay::output::Output;
use smithay::utils::{IsAlive, Logical, Point};

pub struct MoveGrab {
    start_data: GrabStartData<PrismState>,
    window: Window,
    // Cached so we can hand it to interactive_move_end on cleanup
    // even after the layout has stopped tracking the window (e.g.
    // window destroyed mid-drag).
    last_pointer: Point<f64, Logical>,
    /// Whether the pointer has travelled past the 8px recognizing
    /// threshold since the press (niri's `GestureState::Recognizing` →
    /// `Move` gate). Until then, motion is withheld from the layout so
    /// a click with sub-threshold jitter stays in `Starting` — the
    /// layout's own promotion threshold only covers tiled windows
    /// (floating ones promote on the first update), so the grab-side
    /// gate is what keeps click-to-activate working on floating
    /// windows in the overview.
    recognized: bool,
}

impl MoveGrab {
    /// Begin an interactive move. Returns `None` if the layout refuses
    /// to start the move (e.g. another move is already ongoing, or
    /// the window isn't tracked).
    pub fn new(
        state: &mut PrismState,
        start_data: GrabStartData<PrismState>,
        window: Window,
        pos_within_output: Point<f64, Logical>,
        output: Output,
    ) -> Option<Self> {
        let started =
            state
                .layout
                .interactive_move_begin(window.clone(), &output, pos_within_output);
        if !started {
            return None;
        }
        Some(Self {
            last_pointer: start_data.location,
            start_data,
            window,
            recognized: false,
        })
    }

    fn end(&mut self, state: &mut PrismState) {
        // A *click* — release before the layout promoted the move out
        // of `Starting` — in the overview activates the clicked window
        // and zooms to its workspace with a synchronized animation
        // (niri move_grab.rs `on_ungrab`, `GestureState::Recognizing`).
        let overview_click = state.layout.is_overview_open()
            && state.layout.interactive_move_is_starting(&self.window);

        state.layout.interactive_move_end(&self.window);

        if overview_click {
            let res = state.layout.workspaces().find_map(|(mon, ws_idx, ws)| {
                ws.windows()
                    .any(|w| w.window == self.window)
                    .then(|| (mon.map(|mon| mon.output().clone()), ws_idx))
            });
            if let Some((Some(output), ws_idx)) = res {
                state.layout.focus_output(&output);
                state.layout.toggle_overview_to_workspace(ws_idx);
            }
            state.layout.activate_window(&self.window);
        }

        // Drag changed window placement; queue a full redraw so source
        // + destination outputs both repaint. Granular per-output
        // invalidation can replace this once we wire per-window-output
        // tracking through the layout's interactive_move state.
        queue_redraw_all(state);
    }

    fn update(&mut self, state: &mut PrismState, location: Point<f64, Logical>) {
        if !self.window.alive() {
            return;
        }

        // Recognizing gate: withhold motion from the layout until the
        // pointer travels 8px from the press (niri move_grab.rs
        // `on_frame`). Sub-threshold deltas are dropped, not
        // accumulated — when the gate opens, the move starts from the
        // window's rest position, same as niri.
        if !self.recognized {
            let c = location - self.start_data.location;
            if c.x * c.x + c.y * c.y < 8. * 8. {
                self.last_pointer = location;
                return;
            }
            self.recognized = true;
        }

        let delta = location - self.last_pointer;
        self.last_pointer = location;

        // Resolve which output the pointer is currently over (cursor
        // may have crossed a monitor boundary mid-drag — the layout
        // handles the transfer).
        let px = location.x as i32;
        let py = location.y as i32;
        let Some(output_id) = state.output_containing((px, py)) else {
            return;
        };
        let Some(out) = state.wl_outputs.get(&output_id).cloned() else {
            return;
        };
        let origin = out.current_location();
        let pos_within_output = Point::<f64, Logical>::from((
            location.x - origin.x as f64,
            location.y - origin.y as f64,
        ));

        state
            .layout
            .interactive_move_update(&self.window, delta, out, pos_within_output);
        queue_redraw_all(state);
    }
}

impl PointerGrab<PrismState> for MoveGrab {
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
        // While a move grab is active, no client receives pointer
        // focus — pass `None` so the surface under the moving window
        // doesn't see misleading hover events.
        handle.motion(data, None, event);
        self.update(data, event.location);
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
        // Release of the button that started the grab ends it. Other
        // buttons during the grab are forwarded (handle.button above)
        // but don't end it — niri uses the opposite button to toggle
        // floating; we skip that for now.
        if !handle.current_pressed().contains(&self.start_data.button) {
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

/// Queue redraw on every output. Coarser than ideal, but conservative —
/// interactive move can move a window between outputs at any time.
pub(crate) fn queue_redraw_all(state: &mut PrismState) {
    let ids: Vec<_> = state.outputs.keys().cloned().collect();
    for id in ids {
        state.output_redraw.entry(id).or_default().queue_redraw();
    }
}
