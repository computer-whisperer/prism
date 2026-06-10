//! Xwayland support via [xwayland-satellite].
//!
//! prism does not embed an X window manager. Instead it integrates the
//! external `xwayland-satellite` process, which runs its own Xwayland and
//! re-exposes each X11 window to prism as an ordinary `xdg_toplevel`. From
//! prism's point of view the satellite is just another Wayland client, so no
//! window-, layout- or input-side glue is needed — only the process/socket
//! plumbing implemented here and in [`satellite`].
//!
//! This module owns the X11 listening sockets (prism picks the display
//! number and binds the sockets itself), so that the satellite — and by
//! extension Xwayland — can be spawned *on demand*, the first time an X11
//! client actually connects. See [`satellite::setup`].
//!
//! Ported from niri's `src/utils/xwayland/` (the satellite integration it
//! authored as the reference implementation), adapted to prism's
//! [`PrismState`](crate::PrismState) and dropped of niri-specific tracing.

use std::os::fd::OwnedFd;
use std::os::unix::net::{SocketAddr, UnixListener};

use anyhow::{anyhow, ensure, Context as _};
use rustix::fs::{lstat, mkdir, open, unlink, OFlags};
use rustix::io::Errno;
use rustix::process::{getpid, getuid};

pub mod satellite;

const TMP_UNIX_DIR: &str = "/tmp";
const X11_TMP_UNIX_DIR: &str = "/tmp/.X11-unix";

/// A bound pair of X11 listening sockets (abstract + filesystem) plus the
/// lock file that reserves the display number. The `Unlink` guards remove
/// the socket/lock files when the connection is dropped.
struct X11Connection {
    display_name: String,
    // Optional because there are no abstract sockets on FreeBSD.
    abstract_fd: Option<OwnedFd>,
    unix_fd: OwnedFd,
    _unix_guard: Unlink,
    _lock_guard: Unlink,
}

/// Unlinks a path on drop. Used to clean up the X11 lock and socket files.
struct Unlink(String);
impl Drop for Unlink {
    fn drop(&mut self) {
        let _ = unlink(&self.0);
    }
}

// Adapted from Mutter code:
// https://gitlab.gnome.org/GNOME/mutter/-/blob/48.3.1/src/wayland/meta-xwayland.c?ref_type=tags#L513
fn ensure_x11_unix_dir() -> anyhow::Result<()> {
    match mkdir(X11_TMP_UNIX_DIR, 0o1777.into()) {
        Ok(()) => Ok(()),
        Err(Errno::EXIST) => {
            ensure_x11_unix_perms().context("wrong X11 directory permissions")?;
            Ok(())
        }
        Err(err) => Err(err).context("error creating X11 directory"),
    }
}

fn ensure_x11_unix_perms() -> anyhow::Result<()> {
    let x11_tmp = lstat(X11_TMP_UNIX_DIR).context("error checking X11 directory permissions")?;
    let tmp = lstat(TMP_UNIX_DIR).context("error checking /tmp directory permissions")?;

    ensure!(
        x11_tmp.st_uid == tmp.st_uid || x11_tmp.st_uid == getuid().as_raw(),
        "wrong ownership for X11 directory"
    );
    ensure!(
        (x11_tmp.st_mode & 0o022) == 0o022,
        "X11 directory is not writable"
    );
    ensure!(
        (x11_tmp.st_mode & 0o1000) == 0o1000,
        "X11 directory is missing the sticky bit"
    );

    Ok(())
}

/// Find a free X11 display number starting at `start`, reserving it by
/// creating its `/tmp/.X{n}-lock` file. Returns the number, the open lock
/// fd, and an unlink guard for the lock file.
///
/// A lock file whose recorded owner is dead (unclean shutdown of a previous
/// session) is reclaimed: unlinked and re-created, Mutter-style. Without
/// this, every crash permanently burned a display number — and a crashed
/// `:0` made clients that hardcode `DISPLAY=:0` unable to connect until the
/// lock was removed by hand.
fn pick_x11_display(start: u32) -> anyhow::Result<(u32, OwnedFd, Unlink)> {
    for n in start..start + 50 {
        let lock_path = format!("/tmp/.X{n}-lock");
        let flags = OFlags::WRONLY | OFlags::CLOEXEC | OFlags::CREATE | OFlags::EXCL;
        // Second iteration retries once after reclaiming a stale lock.
        for attempt in 0..2 {
            match open(&lock_path, flags, 0o444.into()) {
                Ok(lock_fd) => return Ok((n, lock_fd, Unlink(lock_path))),
                Err(Errno::EXIST) if attempt == 0 && reclaim_stale_x11_lock(&lock_path) => {
                    continue;
                }
                Err(_) => break,
            }
        }
    }

    Err(anyhow!("no free X11 display found after 50 attempts"))
}

/// If `lock_path` records a PID that is no longer running, unlink the lock
/// and return true. Conservative on every doubt: unreadable contents, a
/// non-PID payload, or a `kill(pid, 0)` answer other than ESRCH (alive, or
/// alive-but-not-ours EPERM) all leave the lock in place.
fn reclaim_stale_x11_lock(lock_path: &str) -> bool {
    let Ok(contents) = std::fs::read_to_string(lock_path) else {
        return false;
    };
    // X11 lock format: "%10d\n".
    let Ok(pid) = contents.trim().parse::<i32>() else {
        return false;
    };
    let Some(pid) = rustix::process::Pid::from_raw(pid) else {
        return false;
    };
    if rustix::process::test_kill_process(pid) != Err(Errno::SRCH) {
        return false;
    }
    tracing::info!("reclaiming stale X11 lock {lock_path} (owner {pid:?} is dead)");
    unlink(lock_path).is_ok()
}

fn bind_to_socket(addr: &SocketAddr) -> anyhow::Result<UnixListener> {
    let listener = UnixListener::bind_addr(addr).context("error binding socket")?;
    Ok(listener)
}

#[cfg(target_os = "linux")]
fn bind_to_abstract_socket(display: u32) -> anyhow::Result<UnixListener> {
    use std::os::linux::net::SocketAddrExt;

    let name = format!("/tmp/.X11-unix/X{display}");
    let addr = SocketAddr::from_abstract_name(name).unwrap();
    bind_to_socket(&addr)
}

fn bind_to_unix_socket(display: u32) -> anyhow::Result<(UnixListener, Unlink)> {
    let name = format!("/tmp/.X11-unix/X{display}");
    let addr = SocketAddr::from_pathname(&name).unwrap();
    // Unlink old leftover socket if any.
    let _ = unlink(name.as_str());
    let guard = Unlink(name);
    bind_to_socket(&addr).map(|listener| (listener, guard))
}

fn open_display_sockets(
    display: u32,
) -> anyhow::Result<(Option<UnixListener>, UnixListener, Unlink)> {
    #[cfg(target_os = "linux")]
    let a = Some(bind_to_abstract_socket(display).context("error binding to abstract socket")?);
    #[cfg(not(target_os = "linux"))]
    let a = None;

    let (u, g) = bind_to_unix_socket(display).context("error binding to unix socket")?;
    Ok((a, u, g))
}

/// Pick a free X11 display number and bind its listening sockets. prism owns
/// these sockets and hands them to the satellite via `-listenfd` only once an
/// X11 client connects (see [`satellite`]).
fn setup_connection() -> anyhow::Result<X11Connection> {
    ensure_x11_unix_dir()?;

    let mut n = 0;
    let mut attempt = 0;
    let (display, lock_guard, a, u, unix_guard) = loop {
        let (display, lock_fd, lock_guard) = pick_x11_display(n)?;

        // Write our PID into the lock file.
        let pid_string = format!("{:>10}\n", getpid().as_raw_nonzero());
        if let Err(err) = rustix::io::write(&lock_fd, pid_string.as_bytes()) {
            return Err(err).context("error writing PID to X11 lock file");
        }
        drop(lock_fd);

        match open_display_sockets(display) {
            Ok((a, u, g)) => {
                break (display, lock_guard, a, u, g);
            }
            Err(err) => {
                if attempt == 50 {
                    return Err(err)
                        .context("error opening X11 sockets after creating a lock file");
                }

                n = display + 1;
                attempt += 1;
                continue;
            }
        }
    };

    let display_name = format!(":{display}");
    let abstract_fd = a.map(OwnedFd::from);
    let unix_fd = OwnedFd::from(u);

    Ok(X11Connection {
        display_name,
        abstract_fd,
        unix_fd,
        _unix_guard: unix_guard,
        _lock_guard: lock_guard,
    })
}
