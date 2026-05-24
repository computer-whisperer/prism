//! Wayland protocol wiring.
//!
//! Implements smithay's protocol handler traits on `PrismState`, plus the
//! event-loop helpers needed to bring up a Wayland server socket.
//!
//! Scope: scaffolding only (task #46). Surface tracking and configure
//! lifecycle work; rendering / texture import / input come incrementally.

pub mod client;
pub mod color_management;
pub mod drm_syncobj;
pub mod input_state;
pub mod layer_shell;
pub mod redraw;
pub mod selection;
pub mod server;
pub mod state;
pub mod surface_tex;

pub use client::PrismClient;
pub use input_state::{KeyboardFocus, PointerVisibility};
pub use redraw::{OutputRedrawState, PendingFeedback, RedrawState};
pub use server::insert_wayland_sources;
pub use state::{PrismState, new_display};
pub use surface_tex::{SurfacePlacement, SurfacePlacementSlot, SurfaceTexSlot, SurfaceTexture};
