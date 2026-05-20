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

/// libseat-backed session. Owns the session handle; dropping releases it.
pub struct SeatSession {
    session: LibSeatSession,
    /// Held so the seat-fd stays open. Smithay needs us to keep this around
    /// even if we never poll it.
    _notifier: LibSeatSessionNotifier,
}

impl SeatSession {
    pub fn new() -> Result<Self> {
        let (session, notifier) =
            LibSeatSession::new().context("LibSeatSession::new (is logind/seatd running?)")?;
        tracing::info!(
            "libseat session active, seat={}, active={}",
            session.seat(),
            session.is_active(),
        );
        Ok(Self {
            session,
            _notifier: notifier,
        })
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
