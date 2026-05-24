//! Per-output redraw state machine.
//!
//! The render loop drives every output through three phases:
//!
//! ```text
//!     Idle ──(damage)──► Queued ──(redraw_pass)──► WaitingForVBlank ──┐
//!       ▲                                                              │
//!       └────────────────────(vblank, !redraw_needed)──────────────────┘
//! ```
//!
//! - **Idle**: nothing to do; the output won't render at the next
//!   vblank unless damage flips it to `Queued`.
//! - **Queued**: there's a render pending. The next pass through
//!   `redraw_queued_outputs` will perform it and submit the page-flip.
//! - **WaitingForVBlank**: we just submitted a page-flip and are
//!   waiting for the kernel to report it presented. `redraw_needed`
//!   tells the vblank handler whether to go back to `Queued` or to
//!   `Idle`.
//!
//! This shape lets us split the vblank handler (bookkeeping only —
//! fires `wp_presentation_feedback` for the just-presented frame with
//! the kernel-reported timestamp, advances the FrameClock, flips
//! state) from the actual render+page-flip (runs in a separate
//! event-loop tick via `redraw_queued_outputs`). Following niri's
//! pattern of doing the work *outside* the vblank handler so wayland
//! event servicing doesn't get blocked by GPU work.
//!
//! `PendingFeedback` is the per-output stash created when a render is
//! submitted: the `wl_callback.frame` and `wp_presentation_feedback`
//! objects extracted from this frame's surfaces, waiting to fire at
//! the next vblank with the actual presentation time the kernel
//! reports. Firing them at submit time would be a lie — the buffer
//! isn't on screen yet — and clients (mpv) that interpret it as "go
//! make another frame" will over-produce and stall the compositor.

use std::sync::Arc;
use std::time::Duration;

use smithay::reexports::wayland_server::protocol::wl_callback::WlCallback;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::wayland::presentation::PresentationFeedbackCallback;

use crate::drm_syncobj::CommitReleaseTracker;

/// What we plan to do for an output between now and its next vblank.
#[derive(Debug, Default)]
pub enum RedrawState {
    /// Nothing requested — the output will skip the next vblank
    /// unless damage / commit / animation arrives.
    #[default]
    Idle,
    /// A render is pending. The next `redraw_queued_outputs` pass
    /// will perform it.
    Queued,
    /// A page-flip is in flight. When the vblank arrives:
    ///   - `redraw_needed: true`  ⇒ transition back to `Queued`
    ///     (e.g. continuous animation, or damage since submit).
    ///   - `redraw_needed: false` ⇒ transition to `Idle`.
    WaitingForVBlank { redraw_needed: bool },
}

/// `wl_callback.frame` + `wp_presentation_feedback` objects extracted
/// from a frame's surfaces at submit time, deferred until the kernel
/// reports the corresponding vblank. Fired with the actual
/// presentation time.
pub struct PendingFeedback {
    pub frame_cbs: Vec<WlCallback>,
    pub presentation_cbs: Vec<PresentationFeedbackCallback>,
    /// The `target_presentation_time` we predicted via FrameClock when
    /// we did the render — kept for diagnostics / comparison against
    /// the actual vblank time.
    pub target_time: Duration,
}

/// Per-output redraw bookkeeping, kept on `PrismState`.
#[derive(Default)]
pub struct OutputRedrawState {
    pub redraw: RedrawState,
    /// Feedback for the in-flight page-flip, if any. `Some` while in
    /// `WaitingForVBlank`; taken + fired by the vblank handler.
    pub pending_feedback: Option<PendingFeedback>,
}

impl OutputRedrawState {
    /// Mark that this output needs to render at the earliest
    /// opportunity. From `Idle` ⇒ `Queued`. From `WaitingForVBlank`
    /// ⇒ flag `redraw_needed: true` so the vblank handler queues a
    /// follow-up render. From `Queued` ⇒ already queued, no-op.
    pub fn queue_redraw(&mut self) {
        match self.redraw {
            RedrawState::Idle => self.redraw = RedrawState::Queued,
            RedrawState::WaitingForVBlank { .. } => {
                self.redraw = RedrawState::WaitingForVBlank {
                    redraw_needed: true,
                };
            }
            RedrawState::Queued => {}
        }
    }
}

/// Walk the surface tree rooted at `root` (a toplevel or layer-shell root,
/// down through every subsurface) and harvest, from each surface's current
/// committed state, its pending `wl_callback.frame` callbacks, its
/// `wp_presentation_feedback` callbacks, and its `wp_linux_drm_syncobj`
/// release trackers. Appends to the caller's accumulators so one set can
/// span several roots.
///
/// Descending into subsurfaces is load-bearing: GTK4 / Firefox / Mesa
/// register frame callbacks on subsurface-backed content, and harvesting
/// only the root freezes their animation loops until something else nudges
/// the output (e.g. cursor motion).
///
/// CAREFUL: the visit callback runs *inside* smithay's per-surface
/// data_map lock. Anything that would re-enter `with_states` on the same
/// surface (e.g. the public `drm_syncobj::tracker_for_render` helper)
/// self-deadlocks here — read `SurfaceReleaseSlot` directly off the
/// `states` we already hold, as below.
///
/// Used by the DRM submit path (feeds [`PendingFeedback`] + the
/// release-after-submit wiring) and by the WLCS test harness, which
/// consumes only `frame_cbs` — it fires them off a timer, with no scanout
/// or explicit-sync behind the surfaces, so `presentation_cbs` and
/// `release_trackers` come back empty there.
pub fn harvest_surface_feedback(
    root: &WlSurface,
    frame_cbs: &mut Vec<WlCallback>,
    presentation_cbs: &mut Vec<PresentationFeedbackCallback>,
    release_trackers: &mut Vec<Arc<CommitReleaseTracker>>,
) {
    use smithay::wayland::compositor::{
        with_surface_tree_downward, SurfaceAttributes, TraversalAction,
    };
    use smithay::wayland::presentation::PresentationFeedbackCachedState;

    use crate::drm_syncobj::SurfaceReleaseSlot;

    with_surface_tree_downward(
        root,
        (),
        |_, _, &()| TraversalAction::DoChildren(()),
        |_surface, states, &()| {
            frame_cbs.append(&mut std::mem::take(
                &mut states
                    .cached_state
                    .get::<SurfaceAttributes>()
                    .current()
                    .frame_callbacks,
            ));
            presentation_cbs.append(&mut std::mem::take(
                &mut states
                    .cached_state
                    .get::<PresentationFeedbackCachedState>()
                    .current()
                    .callbacks,
            ));
            if let Some(t) = states
                .data_map
                .get::<SurfaceReleaseSlot>()
                .and_then(|slot| slot.current())
            {
                release_trackers.push(t);
            }
        },
        |_, _, &()| true,
    );
}
