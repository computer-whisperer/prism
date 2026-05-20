//! Wayland protocol wiring.
//!
//! Implements smithay's protocol handler traits on `PrismState`, plus the
//! event-loop helpers needed to bring up a Wayland server socket.
//!
//! Scope: scaffolding only (task #46). Surface tracking and configure
//! lifecycle work; rendering / texture import / input come incrementally.

pub mod client;
pub mod server;
pub mod state;

pub use client::PrismClient;
pub use server::insert_wayland_sources;
pub use state::{PrismState, new_display};
