//! libseat session for DRM master acquisition.
//!
//! On Linux a process needs DRM master to set modes / commit framebuffers.
//! The display session (logind / seatd) hands master to whichever process
//! is the active VT's foreground. libseat is the abstraction that lets us
//! open device fds through the seat manager so master switches correctly
//! across VT switches / login state changes.
//!
//! For the tracer MVP we only do the bare minimum: open a session, open a
//! DRM fd, wrap it as a `DrmDeviceFd`. We ignore the notifier (VT-switch
//! events) for now — the smoke test holds the device for a fixed duration
//! and exits.

use std::path::Path;

use anyhow::{Context, Result};
use rustix::fs::OFlags;
use smithay::backend::drm::DrmDeviceFd;
use smithay::backend::session::Session;
use smithay::backend::session::libseat::{LibSeatSession, LibSeatSessionNotifier};
use smithay::utils::DeviceFd;

/// libseat-backed session. The companion [`LibSeatSessionNotifier`] returned
/// from `new()` MUST be inserted into a calloop event loop. Without it,
/// libseat can't process logind's "release the devices" messages during a
/// VT switch — the kernel waits for our ack, the VT switch hangs, and
/// the user can't escape (Ctrl+Alt+Fn does nothing, SIGINT delivery is
/// blocked too because the desktop session is stuck on the switch).
pub struct SeatSession {
    session: LibSeatSession,
}

impl SeatSession {
    /// Open a libseat session. Returns `(SeatSession, notifier)` — the
    /// caller MUST insert `notifier` into the calloop event loop. The
    /// callback typically logs the `SessionEvent` (Pause / Activate) but
    /// doesn't need to do more — libseat acknowledges the pause inside
    /// its own dispatch path.
    pub fn new() -> Result<(Self, LibSeatSessionNotifier)> {
        let (session, notifier) =
            LibSeatSession::new().context("LibSeatSession::new (is logind/seatd running?)")?;
        tracing::info!(
            "libseat session active, seat={}, active={}",
            session.seat(),
            session.is_active(),
        );
        Ok((Self { session }, notifier))
    }

    pub fn seat(&self) -> String {
        self.session.seat()
    }

    /// A clone of the underlying `LibSeatSession`. Needed for the
    /// libinput backend, which takes ownership of a `Session`-bound
    /// interface (`LibinputSessionInterface<S>`) for fd open/close.
    /// Cheap — `LibSeatSession` is an `Arc` under the hood.
    pub fn libseat_clone(&self) -> LibSeatSession {
        self.session.clone()
    }

    pub fn is_active(&self) -> bool {
        self.session.is_active()
    }

    /// Request a VT switch via libseat. The compositor's input
    /// dispatcher routes `Ctrl+Alt+Fn` (xkbcommon emits
    /// `XF86_Switch_VT_N` keysyms for these on TTY) here so users can
    /// jump to another VT without ssh-ing in to pkill prism. Returns
    /// an `io::Result` so the call site can `tracing::warn` failures
    /// without dragging the `smithay::backend::session::Error` type
    /// across the crate boundary.
    ///
    /// `&self`, not `&mut self`: clones the underlying
    /// `LibSeatSession` (cheap, Arc-backed) and calls
    /// `Session::change_vt` on the clone. The clone is necessary
    /// because `Session::change_vt` takes `&mut self` but the input
    /// dispatcher only has a `&` on its `Option<SeatSession>` field.
    pub fn change_vt(&self, vt: i32) -> std::io::Result<()> {
        let mut session = self.session.clone();
        session
            .change_vt(vt)
            .map_err(|e| {
                std::io::Error::other(format!("libseat change_vt({vt}): {e:?}"))
            })
    }

    /// Open a DRM device through the seat. The returned fd is master-capable
    /// when the session is active (i.e. when we're on the foreground VT).
    pub fn open_drm(&mut self, path: impl AsRef<Path>) -> Result<DrmDeviceFd> {
        let path = path.as_ref();
        let flags = OFlags::RDWR | OFlags::CLOEXEC | OFlags::NONBLOCK;
        let fd = self
            .session
            .open(path, flags)
            .map_err(|e| anyhow::anyhow!("libseat open({}): {e:?}", path.display()))?;
        Ok(DrmDeviceFd::new(DeviceFd::from(fd)))
    }
}
