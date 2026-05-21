//! Input dispatch (event routing, grabs, keybinds).
//!
//! Status: scaffold. Only the standalone helpers (scroll tracker, swipe
//! gesture state machine) are ported. The bulk of niri's input/ tree —
//! `mod.rs` (5497 LOC) + the 7 grab files — is fundamentally
//! `impl State`/`impl PointerGrab<State>` code referencing ~80 fields and
//! methods of `Niri` (seat, layout, output map, surface tracking, event
//! loop, screenshot UI, tablet state, etc.). That code can't be ported
//! cleanly until the PrismState scaffold from task #73 exists.
//!
//! What lives in the parent crate already:
//!   - `prism_layout::swipe_tracker` (touchpad gesture velocity tracking)
//!     was lifted to the lib root during the layout port because
//!     `ScrollingSpace` consumes it directly for view-offset gestures.

pub mod scroll_swipe_gesture;
pub mod scroll_tracker;

pub use scroll_swipe_gesture::ScrollSwipeGesture;
pub use scroll_tracker::ScrollTracker;
