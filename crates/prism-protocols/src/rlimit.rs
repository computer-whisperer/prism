//! Process open-files limit (`RLIMIT_NOFILE`).
//!
//! A compositor holds a file descriptor per client buffer — dmabuf planes, shm
//! pools, sync fences — so the default (~1024) soft limit is easily exhausted
//! by buffer-churning clients: Firefox/WebRender allocates hundreds of dmabufs
//! per second under scroll, and at ~1024 live fds `Dmabuf` import fails with
//! `EMFILE` and the client's connection dies.
//!
//! Raise the soft limit to the hard maximum at startup, and restore the
//! original for spawned children so legacy / `select()`-based programs (notably
//! X clients reached via Xwayland) don't inherit a giant limit. Mirrors niri's
//! `store_and_increase_nofile_rlimit` / `restore_nofile_rlimit`.

use std::sync::atomic::{AtomicU64, Ordering};

// Original limit, captured by `raise_nofile_to_max`. `cur == 0` means "never
// captured" (a real soft limit is never 0), so `restore_nofile` is then a no-op.
static ORIGINAL_CUR: AtomicU64 = AtomicU64::new(0);
static ORIGINAL_MAX: AtomicU64 = AtomicU64::new(0);

/// Raise the open-files soft limit to the hard maximum, storing the original
/// so [`restore_nofile`] can put it back for children. Call once at startup.
pub fn raise_nofile_to_max() {
    let mut rlim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: valid resource id, fully-initialized out-param.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut rlim) } != 0 {
        tracing::warn!(
            "getrlimit(RLIMIT_NOFILE) failed: {}",
            std::io::Error::last_os_error()
        );
        return;
    }

    ORIGINAL_CUR.store(rlim.rlim_cur, Ordering::SeqCst);
    ORIGINAL_MAX.store(rlim.rlim_max, Ordering::SeqCst);

    if rlim.rlim_cur >= rlim.rlim_max {
        return; // already at (or above) the hard cap — nothing to do
    }
    let from = rlim.rlim_cur;
    rlim.rlim_cur = rlim.rlim_max;
    // SAFETY: valid resource id, fully-initialized rlimit.
    if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &rlim) } != 0 {
        tracing::warn!(
            "setrlimit(RLIMIT_NOFILE) failed: {}",
            std::io::Error::last_os_error()
        );
    } else {
        tracing::info!(
            "raised RLIMIT_NOFILE soft limit {} -> {}",
            from,
            rlim.rlim_max
        );
    }
}

/// Restore the open-files limit captured by [`raise_nofile_to_max`]. Intended
/// for a child's `Command::pre_exec` (post-fork, pre-exec) so it doesn't
/// inherit the raised soft limit. No-op if the limit was never raised.
///
/// Async-signal-safe: only lock-free atomic loads and a single `setrlimit`
/// syscall — no allocation, locks, or logging — so it is sound in the
/// post-fork window.
pub fn restore_nofile() {
    let rlim_cur = ORIGINAL_CUR.load(Ordering::SeqCst);
    if rlim_cur == 0 {
        return;
    }
    let rlim = libc::rlimit {
        rlim_cur,
        rlim_max: ORIGINAL_MAX.load(Ordering::SeqCst),
    };
    // SAFETY: valid resource id, fully-initialized rlimit; setrlimit is a
    // single async-signal-safe syscall.
    unsafe {
        libc::setrlimit(libc::RLIMIT_NOFILE, &rlim);
    }
}
