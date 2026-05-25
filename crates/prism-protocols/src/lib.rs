//! Wayland protocol wiring.
//!
//! Implements smithay's protocol handler traits on `PrismState`, plus the
//! event-loop helpers needed to bring up a Wayland server socket.
//!
//! Scope: scaffolding only (task #46). Surface tracking and configure
//! lifecycle work; rendering / texture import / input come incrementally.

pub mod client;
pub mod color_management;
pub mod dmabuf_sync;
pub mod drm_syncobj;
pub mod input_state;
pub mod layer_shell;
pub mod output_power;
pub mod pointer_focus;
pub mod redraw;
pub mod selection;
pub mod server;
pub mod state;
pub mod surface_tex;
pub mod xwayland;

pub use client::PrismClient;
pub use input_state::{KeyboardFocus, PointerVisibility};
pub use redraw::{OutputRedrawState, PendingFeedback, RedrawState};
pub use server::insert_wayland_sources;
pub use state::{
    destroy_render_wait_semaphores, materialize_surface_on_gpu, new_display,
    prepare_dmabuf_acquire_waits, prepare_mirror_waits, PrismState,
};
pub use surface_tex::{
    GpuTex, SurfacePlacement, SurfacePlacementSlot, SurfaceTexSlot, SurfaceTexture, TexSource,
};
