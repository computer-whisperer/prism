//! On-demand spawning of `xwayland-satellite`.
//!
//! prism binds the X11 sockets itself (see [`super::setup_connection`]) and
//! watches them on the event loop. The satellite — and therefore Xwayland —
//! is only launched the first time an X11 client connects, by handing the
//! pre-bound sockets to the satellite via its `-listenfd` argument. When the
//! satellite exits (e.g. crashes, or its last X11 client quits) the socket
//! watch is re-armed so the next connection spawns it again.
//!
//! Ported from niri's `src/utils/xwayland/satellite.rs`, adapted to
//! [`PrismState`]: niri's `state.niri.event_loop` is prism's
//! [`PrismState::loop_handle`], and prism's spawned children inherit prism's
//! own `DISPLAY` (set in [`setup`]), so niri's `CHILD_DISPLAY` global is not
//! needed.

use std::io;
use std::os::fd::{AsRawFd as _, BorrowedFd, OwnedFd};
use std::os::unix::net::UnixListener;
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;

use rustix::io::{fcntl_setfd, FdFlags};
use smithay::reexports::calloop::channel::{channel, Event, Sender};
use smithay::reexports::calloop::generic::Generic;
use smithay::reexports::calloop::{Interest, Mode, PostAction, RegistrationToken};
use tracing::{debug, error, warn};

use super::X11Connection;
use crate::PrismState;

/// Live xwayland-satellite integration state: the bound X11 sockets and the
/// event-loop tokens watching them for the next client connection.
pub struct Satellite {
    x11: X11Connection,
    abstract_token: Option<RegistrationToken>,
    unix_token: Option<RegistrationToken>,
    to_main: Sender<ToMain>,
}

enum ToMain {
    SetupWatch,
}

impl Satellite {
    /// The X11 display name (e.g. `":1"`) clients should use as `$DISPLAY`.
    pub fn display_name(&self) -> &str {
        &self.x11.display_name
    }
}

/// Bring up xwayland-satellite integration: bind the X11 sockets, start
/// watching them for the first X11 client, and export `$DISPLAY` for prism's
/// children. Idempotent — a no-op if already set up. Disabled by the
/// `xwayland-satellite { off }` config or if the installed satellite is too
/// old to support on-demand activation.
///
/// Must be called during single-threaded startup: it mutates prism's process
/// environment (`$DISPLAY`) for children to inherit, which is only sound
/// while no other thread reads the environment (mirrors the `WAYLAND_DISPLAY`
/// handling in [`crate::server`]).
pub fn setup(state: &mut PrismState) {
    setup_sockets(state);
    export_display(state);
}

/// Set or clear prism's own `$DISPLAY` so spawned children connect to (or are
/// kept away from) the satellite. See [`setup`] for the safety contract.
fn export_display(state: &PrismState) {
    // SAFETY: single-threaded startup only — see `setup` docs and the
    // matching WAYLAND_DISPLAY comment in server.rs.
    unsafe {
        match &state.satellite {
            Some(satellite) => std::env::set_var("DISPLAY", satellite.display_name()),
            // Avoid spawning children into a host X11 server.
            None => std::env::remove_var("DISPLAY"),
        }
    }
}

fn setup_sockets(state: &mut PrismState) {
    if state.satellite.is_some() {
        return;
    }

    let config = state.config.borrow();
    let xwls_config = &config.xwayland_satellite;
    if xwls_config.off {
        return;
    }

    if !test_ondemand(&xwls_config.path) {
        return;
    }
    drop(config);

    let x11 = match super::setup_connection() {
        Ok(x11) => x11,
        Err(err) => {
            warn!("error opening X11 sockets, disabling xwayland-satellite integration: {err:?}");
            return;
        }
    };

    let Some(loop_handle) = state.loop_handle.clone() else {
        error!("xwayland: setup called before the loop handle was set");
        return;
    };

    let (to_main, rx) = channel();
    loop_handle
        .insert_source(rx, move |event, _, state| match event {
            Event::Msg(msg) => match msg {
                ToMain::SetupWatch => setup_watch(state),
            },
            Event::Closed => (),
        })
        .unwrap();

    state.satellite = Some(Satellite {
        x11,
        abstract_token: None,
        unix_token: None,
        to_main,
    });

    setup_watch(state);
}

/// Probe whether the configured satellite binary supports `-listenfd`
/// on-demand activation (`--test-listenfd-support` exits 0 if so). Disables
/// integration for too-old binaries rather than failing later.
fn test_ondemand(path: &str) -> bool {
    let path = expand_home(path);

    let mut process = Command::new(&path);
    process
        .args([":0", "--test-listenfd-support"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .env_remove("DISPLAY")
        .env_remove("RUST_BACKTRACE")
        .env_remove("RUST_LIB_BACKTRACE");

    let mut child = match process.spawn() {
        Ok(child) => child,
        Err(err) => {
            warn!("error spawning xwayland-satellite at {path:?}, disabling integration: {err}");
            return false;
        }
    };

    let status = match child.wait() {
        Ok(status) => status,
        Err(err) => {
            warn!("error waiting for xwayland-satellite, disabling integration: {err}");
            return false;
        }
    };

    if !status.success() {
        warn!("xwayland-satellite doesn't support on-demand activation, disabling integration");
        return false;
    }

    true
}

// When xwayland-satellite fails to start and accept a connection on the socket, the socket will
// keep triggering our event source, even after the X11 client quits, resulting in a busyloop of
// trying to start xwayland-satellite. This function will clear out (accept and drop) all pending
// connections on the socket before registering a new event source, working around this problem.
// When the problem happens, it's very likely that xwayland-satellite won't be able to accept the
// pending client (since it had just failed to do so), so it's fine to drop the connections.
fn clear_out_pending_connections(fd: OwnedFd) -> OwnedFd {
    let listener = UnixListener::from(fd);

    if let Err(err) = listener.set_nonblocking(true) {
        warn!("error setting X11 socket to nonblocking: {err:?}");
        return OwnedFd::from(listener);
    }

    while listener.accept().is_ok() {}

    if let Err(err) = listener.set_nonblocking(false) {
        warn!("error setting X11 socket to blocking: {err:?}");
    }

    OwnedFd::from(listener)
}

/// (Re-)arm the event-loop watch on both X11 sockets. The first socket to
/// become readable (an X11 client connecting) spawns the satellite and
/// removes the other watch.
fn setup_watch(state: &mut PrismState) {
    let Some(loop_handle) = state.loop_handle.clone() else {
        error!("xwayland: setup_watch called before the loop handle was set");
        return;
    };

    let Some(satellite) = state.satellite.as_mut() else {
        return;
    };

    if let Some(token) = satellite.abstract_token.take() {
        error!("abstract_token must be None in setup_watch()");
        loop_handle.remove(token);
    }
    if let Some(token) = satellite.unix_token.take() {
        error!("unix_token must be None in setup_watch()");
        loop_handle.remove(token);
    }

    if let Some(fd) = &satellite.x11.abstract_fd {
        let fd = fd.try_clone().unwrap();
        let fd = clear_out_pending_connections(fd);
        let source = Generic::new(fd, Interest::READ, Mode::Level);
        let token = loop_handle
            .insert_source(source, move |_, _, state| {
                if let Some(satellite) = &mut state.satellite {
                    // Remove the other source.
                    if let Some(token) = satellite.unix_token.take() {
                        state.loop_handle.as_ref().unwrap().remove(token);
                    }
                    // Clear this source.
                    satellite.abstract_token = None;

                    debug!("connection to X11 abstract socket; spawning xwayland-satellite");
                    let path = state.config.borrow().xwayland_satellite.path.clone();
                    spawn(path, satellite);
                }
                Ok(PostAction::Remove)
            })
            .unwrap();
        satellite.abstract_token = Some(token);
    }

    let fd = satellite.x11.unix_fd.try_clone().unwrap();
    let fd = clear_out_pending_connections(fd);
    let source = Generic::new(fd, Interest::READ, Mode::Level);
    let token = loop_handle
        .insert_source(source, move |_, _, state| {
            if let Some(satellite) = &mut state.satellite {
                // Remove the other source.
                if let Some(token) = satellite.abstract_token.take() {
                    state.loop_handle.as_ref().unwrap().remove(token);
                }
                // Clear this source.
                satellite.unix_token = None;

                debug!("connection to X11 unix socket; spawning xwayland-satellite");
                let path = state.config.borrow().xwayland_satellite.path.clone();
                spawn(path, satellite);
            }
            Ok(PostAction::Remove)
        })
        .unwrap();
    satellite.unix_token = Some(token);
}

fn spawn(path: String, xwl: &Satellite) {
    let abstract_fd = xwl
        .x11
        .abstract_fd
        .as_ref()
        .map(|fd| fd.try_clone().unwrap());
    let unix_fd = xwl.x11.unix_fd.try_clone().unwrap();
    let to_main = xwl.to_main.clone();

    let path = expand_home(&path);

    let mut process = Command::new(&path);
    process.arg(&xwl.x11.display_name).env_remove("DISPLAY");

    // We don't want it spamming the prism output.
    process
        .env_remove("RUST_BACKTRACE")
        .env_remove("RUST_LIB_BACKTRACE");
    process
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    // SAFETY: unblock_all only resets the child's signal mask (async-signal-safe).
    unsafe { process.pre_exec(unblock_all) };

    // Spawning and waiting takes some milliseconds, so do it in a thread.
    let res = thread::Builder::new()
        .name("Xwl-s Spawner".to_owned())
        .spawn(move || {
            spawn_and_wait(&path, process, abstract_fd, unix_fd);

            // Once xwayland-satellite crashes or fails to spawn, re-establish our X11 socket watch
            // to try again next time.
            let _ = to_main.send(ToMain::SetupWatch);
        });

    if let Err(err) = res {
        warn!("error spawning a thread to spawn xwayland-satellite: {err:?}");
        let _ = xwl.to_main.send(ToMain::SetupWatch);
    }
}

fn spawn_and_wait(
    path: &Path,
    mut process: Command,
    abstract_fd: Option<OwnedFd>,
    unix_fd: OwnedFd,
) {
    let abstract_raw = abstract_fd.as_ref().map(|fd| fd.as_raw_fd());
    let unix_raw = unix_fd.as_raw_fd();

    process.arg("-listenfd").arg(unix_raw.to_string());

    if let Some(abstract_raw) = abstract_raw {
        process.arg("-listenfd").arg(abstract_raw.to_string());
    }

    unsafe {
        process.pre_exec(move || {
            // We're about to exec xwl-s; perfect time to clear CLOEXEC on the file descriptors
            // that we want to pass it.

            // We're not dropping these until after spawn().
            let unix_fd = BorrowedFd::borrow_raw(unix_raw);
            fcntl_setfd(unix_fd, FdFlags::empty())?;

            if let Some(abstract_raw) = abstract_raw {
                let abstract_fd = BorrowedFd::borrow_raw(abstract_raw);
                fcntl_setfd(abstract_fd, FdFlags::empty())?;
            }

            Ok(())
        })
    };

    let mut child = match process.spawn() {
        Ok(child) => child,
        Err(err) => {
            warn!("error spawning {path:?}: {err:?}");
            return;
        }
    };

    // The process spawned, we can drop our fds.
    drop(abstract_fd);
    drop(unix_fd);

    let status = match child.wait() {
        Ok(status) => status,
        Err(err) => {
            warn!("error waiting for xwayland-satellite: {err:?}");
            return;
        }
    };

    // This is most likely a crash, hence warn!().
    warn!("xwayland-satellite exited with: {status}");
}

/// Reset the child's signal mask to empty before exec. calloop's `Signals`
/// source blocks SIGINT/SIGTERM/etc. on prism's main thread; without this the
/// satellite (and Xwayland) would inherit that blocked mask.
fn unblock_all() -> io::Result<()> {
    // SAFETY: called in the forked child before exec, where it is
    // single-threaded; sigemptyset/sigprocmask are async-signal-safe.
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        if libc::sigemptyset(&mut set) != 0 {
            return Err(io::Error::last_os_error());
        }
        if libc::sigprocmask(libc::SIG_SETMASK, &set, std::ptr::null_mut()) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// Expand a leading `~` / `~/` in a configured path to `$HOME`. `exec` does
/// not perform tilde expansion, so a configured `path "~/bin/xwls"` would
/// otherwise fail to launch.
fn expand_home(path: &str) -> PathBuf {
    if path == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home);
        }
    } else if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return Path::new(&home).join(rest);
        }
    }
    PathBuf::from(path)
}
