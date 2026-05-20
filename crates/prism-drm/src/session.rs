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

    pub fn is_active(&self) -> bool {
        self.session.is_active()
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
