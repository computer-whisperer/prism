//! Wayland-server event-loop wiring.
//!
//! Two calloop sources are needed for a working wayland server:
//!   1. `ListeningSocketSource` — the AF_UNIX listening socket. Fires when
//!      a new client connects; we register it with `display.insert_client`.
//!   2. `Generic<Display>` on the display's poll fd — fires when any client
//!      has pending requests; we call `display.dispatch_clients()`.
//!
//! Plus a flush after every loop turn (`display.flush_clients()`) so we
//! don't accidentally hold replies in the kernel buffer.

use anyhow::{Context, Result};
use calloop::generic::Generic;
use calloop::{Interest, LoopHandle, Mode, PostAction};
use smithay::reexports::wayland_server::Display;
use smithay::wayland::socket::ListeningSocketSource;

use crate::state::{new_client_data, PrismState};

/// Insert the listening socket + display dispatch sources into the loop.
/// Returns the socket name (e.g. `"wayland-1"`) that clients should set as
/// `WAYLAND_DISPLAY`.
pub fn insert_wayland_sources(
    handle: &LoopHandle<'static, PrismState>,
    display: Display<PrismState>,
) -> Result<String> {
    let listening = ListeningSocketSource::new_auto().context("ListeningSocketSource::new_auto")?;
    let socket_name = listening.socket_name().to_string_lossy().into_owned();

    // Set WAYLAND_DISPLAY in our own process env so child processes
    // we later spawn (Mod+Return → alacritty, etc.) inherit it and
    // can connect to this socket. Without this, prism spawns succeed
    // but the child has no way to find our socket and exits
    // silently. SAFETY: set_var has soundness caveats on some
    // platforms when other threads read env concurrently, but we're
    // still single-threaded at server-startup time (event loop hasn't
    // started yet).
    // SAFETY: see comment above — single-threaded at this point.
    unsafe {
        std::env::set_var("WAYLAND_DISPLAY", &socket_name);
    }

    handle
        .insert_source(listening, |client_stream, _, state| {
            match state
                .display_handle
                .insert_client(client_stream, new_client_data())
            {
                Ok(client) => tracing::info!(client_id = ?client.id(), "client connected"),
                Err(e) => tracing::warn!("insert_client failed: {e}"),
            }
        })
        .map_err(|e| anyhow::anyhow!("insert listening source: {e}"))?;

    handle
        .insert_source(
            Generic::new(display, Interest::READ, Mode::Level),
            |_, display, state| {
                // SAFETY: we don't drop the Display; smithay's anvil follows
                // the same pattern.
                unsafe {
                    display
                        .get_mut()
                        .dispatch_clients(state)
                        .map_err(|e| std::io::Error::other(format!("dispatch_clients: {e}")))?;
                }
                Ok(PostAction::Continue)
            },
        )
        .map_err(|e| anyhow::anyhow!("insert display source: {e}"))?;

    tracing::info!(socket = %socket_name, "wayland server listening");
    Ok(socket_name)
}
