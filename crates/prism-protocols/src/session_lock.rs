//! `ext-session-lock-v1` — screen locking (swaylock, gtklock, …).
//!
//! Niri port (niri/src/niri.rs lock machinery + handlers/mod.rs). The
//! protocol's safety property is that `ext_session_lock_v1.locked` may
//! only be sent once the session is actually secured — so confirmation
//! is withheld until **every powered output has rendered a locked
//! frame** (lock surface or opaque backdrop). Until then a compositor
//! crash or client race could leak desktop content.
//!
//! State machine ([`LockState`]):
//!
//! ```text
//! Unlocked ──lock()──▶ WaitingForSurfaces ──all surfaces mapped──▶ Locking
//!                          │   (≤1 s deadline timer)                  │
//!                          └──────────deadline───────────────────────▶│
//!                                                                     ▼
//!                       every powered output rendered a locked frame: │
//!                       confirmation.lock() ──▶ Locked(ExtSessionLockV1)
//! ```
//!
//! `WaitingForSurfaces` exists so the first locked frame already shows
//! the client's lock image instead of flashing the solid backdrop:
//! swaylock needs a moment to draw a big background. The session does
//! not count as locked yet in that state (`is_locked()` is false) —
//! render and input gating start at `Locking`.
//!
//! Divergence from niri: niri gates the "all outputs rendered locked"
//! check on a global `monitors_active` flag; prism tracks DPMS per
//! output (`OutputContext::is_powered_off`), so powered-off outputs are
//! individually exempted — a DPMS'd panel can't show content and would
//! otherwise stall the confirmation forever.
//!
//! Lock-surface *input* routing (keyboard focus, pointer gating,
//! `allow-when-locked` binds) lives in the input increment; this module
//! owns the protocol + state machine + what the render path consults.

use std::time::Duration;

use smithay::output::Output;
use smithay::reexports::calloop::timer::{TimeoutAction, Timer};
use smithay::reexports::calloop::RegistrationToken;
use smithay::reexports::wayland_protocols::ext::session_lock::v1::server::ext_session_lock_v1::ExtSessionLockV1;
use smithay::reexports::wayland_server::protocol::wl_output::WlOutput;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::Resource;
use smithay::utils::{Logical, Size};
use smithay::wayland::compositor::with_states;
use smithay::wayland::session_lock::{
    LockSurface, SessionLockHandler, SessionLockManagerState, SessionLocker,
};

use crate::state::{queue_redraw_all, PrismState};

/// Backdrop drawn on every output while locked, under (or instead of)
/// the lock surface. Deliberately not black and not configurable — the
/// dark red reads as "locked, lock client has no surface here" rather
/// than "output died". Same color as niri's `CLEAR_COLOR_LOCKED`.
pub const CLEAR_COLOR_LOCKED: [f32; 4] = [0.3, 0.1, 0.1, 1.0];

/// How long `WaitingForSurfaces` waits for the lock client to commit a
/// surface on every output before blanking anyway (niri: 1 s — swaylock
/// can take its time painting a large background image).
const LOCK_SURFACES_DEADLINE: Duration = Duration::from_millis(1000);

/// The session-lock state machine. See the module doc for the diagram.
#[derive(Debug, Default)]
pub enum LockState {
    #[default]
    Unlocked,
    /// `lock()` arrived; waiting for the client to map lock surfaces on
    /// all outputs (or the deadline) before switching the render path.
    WaitingForSurfaces {
        confirmation: SessionLocker,
        deadline_token: RegistrationToken,
    },
    /// Render path is locked; waiting for every powered output to
    /// actually present a locked frame before confirming to the client.
    Locking(SessionLocker),
    /// Confirmed. The resource may be dead (lock client crashed) — the
    /// session then STAYS locked (backdrop only) until a new lock
    /// client replaces it and unlocks.
    Locked(ExtSessionLockV1),
}

/// What the last *presented* frame on an output showed. Consulted when
/// deciding whether the whole session is safely blanked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LockRenderState {
    #[default]
    Unlocked,
    Locked,
}

impl SessionLockHandler for PrismState {
    fn lock_state(&mut self) -> &mut SessionLockManagerState {
        &mut self.session_lock_state
    }

    fn lock(&mut self, confirmation: SessionLocker) {
        self.lock_session(confirmation);
    }

    fn unlock(&mut self) {
        self.unlock_session();
        // niri parity: unlocking wakes the displays and resets idle, so
        // the user isn't typing their password into a DPMS'd panel's
        // successor state.
        self.set_all_monitors_powered(true);
        self.notify_idle_activity();
    }

    fn new_surface(&mut self, surface: LockSurface, wl_output: WlOutput) {
        let Some(output) = Output::from_resource(&wl_output) else {
            return;
        };
        configure_lock_surface(&surface, &output);
        self.new_lock_surface(surface, &output);
    }
}

// No per-protocol delegate needed: the blanket
// `smithay::delegate_dispatch2!(PrismState)` in state.rs routes
// session-lock requests through this `SessionLockHandler` impl.

impl PrismState {
    /// True from the moment the render path switches to lock-only
    /// output (`Locking`) — NOT during `WaitingForSurfaces`, which is
    /// still showing the normal session.
    pub fn is_locked(&self) -> bool {
        matches!(
            self.lock_state,
            LockState::Locking(_) | LockState::Locked(_)
        )
    }

    fn lock_session(&mut self, confirmation: SessionLocker) {
        // Another client is mid-lock: refuse (dropping the confirmation
        // sends `finished`).
        if matches!(
            self.lock_state,
            LockState::WaitingForSurfaces { .. } | LockState::Locking(_)
        ) {
            tracing::info!("refusing lock: another client is currently locking");
            return;
        }

        if let LockState::Locked(lock) = &self.lock_state {
            if lock.is_alive() {
                tracing::info!("refusing lock: already locked with an active client");
                return;
            }
            // The previous lock client died while locked. Outputs are
            // already blanked, so the replacement locks right away.
            tracing::info!("locking session (replacing dead lock client)");
            let lock = confirmation.ext_session_lock().clone();
            confirmation.lock();
            self.lock_state = LockState::Locked(lock);
            return;
        }

        tracing::info!("locking session");

        if self.outputs.is_empty() {
            // Nothing to blank; lock immediately.
            let lock = confirmation.ext_session_lock().clone();
            confirmation.lock();
            self.lock_state = LockState::Locked(lock);
            return;
        }

        // Give the client up to a second to map lock surfaces on every
        // output, so the first locked frame is the lock image rather
        // than a backdrop flash. `maybe_continue_to_locking` short-cuts
        // the deadline once they're all mapped.
        let Some(handle) = self.loop_handle.clone() else {
            // No event loop (tests): skip the grace period.
            self.lock_state = LockState::Locking(confirmation);
            queue_redraw_all(self);
            return;
        };
        let timer = Timer::from_duration(LOCK_SURFACES_DEADLINE);
        match handle.insert_source(timer, |_, _, state: &mut PrismState| {
            tracing::trace!("lock-surface deadline expired, continuing to locking");
            state.continue_to_locking();
            TimeoutAction::Drop
        }) {
            Ok(deadline_token) => {
                self.lock_state = LockState::WaitingForSurfaces {
                    confirmation,
                    deadline_token,
                };
            }
            Err(e) => {
                tracing::warn!("lock deadline timer insert failed: {e}; locking without grace");
                self.lock_state = LockState::Locking(confirmation);
                queue_redraw_all(self);
            }
        }
    }

    /// Advance `WaitingForSurfaces → Locking` early if the lock client
    /// has mapped a surface on every output. Called on lock-surface
    /// commits; no-op in any other state.
    pub(crate) fn maybe_continue_to_locking(&mut self) {
        if !matches!(self.lock_state, LockState::WaitingForSurfaces { .. }) {
            return;
        }
        for id in self.outputs.keys() {
            let Some(surface) = self.lock_surfaces.get(id) else {
                return; // not created yet
            };
            // Mapped = has committed a buffer (niri's `is_mapped`).
            let mapped = smithay::backend::renderer::utils::with_renderer_surface_state(
                surface.wl_surface(),
                |s| s.buffer().is_some(),
            )
            .unwrap_or(false);
            if !mapped {
                return;
            }
        }
        tracing::trace!("lock surfaces are ready, continuing to locking");
        self.continue_to_locking();
    }

    fn continue_to_locking(&mut self) {
        match std::mem::take(&mut self.lock_state) {
            LockState::WaitingForSurfaces {
                confirmation,
                deadline_token,
            } => {
                if let Some(handle) = &self.loop_handle {
                    handle.remove(deadline_token);
                }
                // Reset any client-set cursor: the surface it belongs
                // to is about to stop receiving input. (niri also
                // closes its screenshot UI here; prism has none.)
                self.cursor_manager
                    .set_cursor_image(smithay::input::pointer::CursorImageStatus::default_named());
                self.cursor_dirty = true;

                if self.outputs.is_empty() {
                    let lock = confirmation.ext_session_lock().clone();
                    confirmation.lock();
                    self.lock_state = LockState::Locked(lock);
                } else {
                    self.lock_state = LockState::Locking(confirmation);
                    queue_redraw_all(self);
                }
            }
            other => {
                tracing::error!("continue_to_locking() in wrong lock state: {other:?}");
                self.lock_state = other;
            }
        }
    }

    /// Tear the lock down: from the client's `unlock` request, or as
    /// the failure path when a locked frame couldn't be rendered while
    /// `Locking` (the confirmation is dropped → client gets `finished`).
    pub(crate) fn unlock_session(&mut self) {
        tracing::info!("unlocking session");
        let prev = std::mem::take(&mut self.lock_state);
        if let LockState::WaitingForSurfaces { deadline_token, .. } = prev {
            if let Some(handle) = &self.loop_handle {
                handle.remove(deadline_token);
            }
        }
        self.lock_surfaces.clear();
        queue_redraw_all(self);
    }

    fn new_lock_surface(&mut self, surface: LockSurface, output: &Output) {
        let lock = match &self.lock_state {
            LockState::Unlocked => {
                tracing::error!("lock surface created on an unlocked session");
                return;
            }
            LockState::WaitingForSurfaces { confirmation, .. } => confirmation.ext_session_lock(),
            LockState::Locking(confirmation) => confirmation.ext_session_lock(),
            LockState::Locked(lock) => lock,
        };
        if lock.client() != surface.wl_surface().client() {
            tracing::debug!("ignoring lock surface from an unrelated client");
            return;
        }
        let Some(id) = self
            .wl_outputs
            .iter()
            .find_map(|(id, o)| (o == output).then(|| id.clone()))
        else {
            tracing::error!("lock surface for an unknown output");
            return;
        };
        self.lock_surfaces.insert(id, surface);
    }

    /// The lock surface keyboard input should go to: the one on the
    /// output under the cursor, else the layout's active output, else
    /// the first output. `None` while the lock client has no (live)
    /// surface there. Niri's `lock_surface_focus` (niri.rs:3647).
    pub fn lock_surface_focus(&self) -> Option<WlSurface> {
        let under_cursor =
            self.output_containing((self.pointer_pos.x as i32, self.pointer_pos.y as i32));
        let output_id = under_cursor
            .or_else(|| {
                let active = self.layout.active_output()?;
                self.wl_outputs
                    .iter()
                    .find_map(|(id, o)| (o == active).then(|| id.clone()))
            })
            .or_else(|| self.outputs.keys().next().cloned())?;
        self.lock_surfaces
            .get(&output_id)
            .filter(|ls| ls.alive())
            .map(|ls| ls.wl_surface().clone())
    }

    /// Commit on a surface with the `ext_session_lock_surface_v1` role:
    /// while waiting, a newly mapped surface may complete the set;
    /// afterwards it's a repaint of that output's lock screen.
    pub(crate) fn session_lock_surface_commit(&mut self, surface: &WlSurface) {
        let Some(id) = self
            .lock_surfaces
            .iter()
            .find_map(|(id, ls)| (ls.wl_surface() == surface).then(|| id.clone()))
        else {
            return;
        };
        if matches!(self.lock_state, LockState::WaitingForSurfaces { .. }) {
            self.maybe_continue_to_locking();
        } else {
            self.output_redraw.entry(id).or_default().queue_redraw();
        }
    }

    /// Post-render bookkeeping, called by the render path after a frame
    /// for `output_id` was successfully submitted (`rendered_locked` =
    /// it was the lock-only frame), or after a render FAILURE with
    /// `rendered_locked = false`. Advances `Locking → Locked` once every
    /// powered output shows a locked frame; a normal/failed frame while
    /// `Locking` aborts the lock (niri parity — never confirm a lock
    /// the screen doesn't reflect).
    pub fn note_lock_render(&mut self, output_id: &str, rendered_locked: bool) {
        self.lock_render_state.insert(
            output_id.to_owned(),
            if rendered_locked {
                LockRenderState::Locked
            } else {
                LockRenderState::Unlocked
            },
        );

        match std::mem::take(&mut self.lock_state) {
            LockState::Locking(confirmation) => {
                if !rendered_locked {
                    // Needed a locked frame on this output but didn't
                    // get one. Drop the confirmation (→ `finished`) and
                    // abort.
                    drop(confirmation);
                    self.unlock_session();
                } else {
                    let all_locked = self.outputs.iter().all(|(id, ctx)| {
                        ctx.is_powered_off()
                            || self.lock_render_state.get(id) == Some(&LockRenderState::Locked)
                    });
                    if all_locked {
                        let lock = confirmation.ext_session_lock().clone();
                        confirmation.lock();
                        tracing::info!("session locked");
                        self.lock_state = LockState::Locked(lock);
                    } else {
                        self.lock_state = LockState::Locking(confirmation);
                    }
                }
            }
            other => self.lock_state = other,
        }
    }
}

/// Size a lock surface to its output and send the configure (plus
/// preferred buffer scale/transform, so hidpi lock screens render
/// sharp). Called on surface creation; re-call if an output's
/// mode/scale ever changes while locked.
pub(crate) fn configure_lock_surface(surface: &LockSurface, output: &Output) {
    surface.with_pending_state(|states| {
        states.size = Some(output_logical_size(output));
    });
    let scale = output.current_scale();
    let transform = output.current_transform();
    let wl_surface = surface.wl_surface();
    with_states(wl_surface, |data| {
        smithay::wayland::compositor::send_surface_state(
            wl_surface,
            data,
            scale.integer_scale(),
            transform,
        );
    });
    surface.send_configure();
}

/// The output's size in logical pixels (mode size, transformed, scaled)
/// — what a fullscreen surface on it should be told to be.
fn output_logical_size(output: &Output) -> Size<u32, Logical> {
    let mode_size = output
        .current_mode()
        .map(|m| m.size)
        .unwrap_or_else(|| (0, 0).into());
    let logical = output
        .current_transform()
        .transform_size(mode_size)
        .to_f64()
        .to_logical(output.current_scale().fractional_scale())
        .to_i32_round::<i32>();
    Size::from((logical.w.max(1) as u32, logical.h.max(1) as u32))
}
