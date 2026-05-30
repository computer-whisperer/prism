//! Input dispatch for prism — libinput events → seat → focused surface.
//!
//! ## Surface ported so far
//!
//! - [`backend_ext`]: extension traits over smithay's `InputBackend` /
//!   `Device` so handler signatures can stay generic across libinput
//!   and winit virtual devices.
//!
//! ## Still upstream-only
//!
//! The bulk of niri's `input/mod.rs` (5497 LOC) plus 7 grab files
//! (1840 LOC) is fundamentally `impl State` code referencing ~80
//! fields/methods of `Niri`. Porting incrementally as those
//! subsystems land on `PrismState`. See task #71 sub-steps (71a-71e)
//! in the task list for the staging plan.

pub mod actions;
pub mod backend_ext;
pub mod dispatch;
pub mod move_grab;
pub mod pointer;
pub mod resize_grab;

pub use actions::{set_child_env, spawn, spawn_sh};
pub use backend_ext::{PrismInputBackend, PrismInputDevice};
pub use dispatch::process_input_event;
pub use move_grab::MoveGrab;
pub use resize_grab::ResizeGrab;
