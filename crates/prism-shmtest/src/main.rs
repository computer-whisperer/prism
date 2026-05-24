//! Minimal wayland xdg-shell + shm client. Draws a single-color toplevel
//! surface and re-attaches its buffer once per second (to exercise prism's
//! shm upload path on every commit) until SIGINT or a max-frame cap.
//!
//! Usage:
//!   prism-shmtest [seconds=10] [width=640] [height=480]
//!
//! Uses `$WAYLAND_DISPLAY` from the env; respects `$XDG_RUNTIME_DIR`.

use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::time::{Duration, Instant};

use memmap2::MmapMut;
use wayland_client::globals::{registry_queue_init, GlobalListContents};
use wayland_client::protocol::{
    wl_buffer, wl_compositor, wl_registry, wl_shm, wl_shm_pool, wl_surface,
};
use wayland_client::{Connection, Dispatch, QueueHandle, WEnum};
use wayland_protocols::xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base};

const FORMAT: wl_shm::Format = wl_shm::Format::Xrgb8888;
const BPP: u32 = 4;

struct State {
    compositor: wl_compositor::WlCompositor,
    shm: wl_shm::WlShm,
    wm_base: xdg_wm_base::XdgWmBase,

    width: u32,
    height: u32,

    surface: Option<wl_surface::WlSurface>,
    xdg_surface: Option<xdg_surface::XdgSurface>,
    toplevel: Option<xdg_toplevel::XdgToplevel>,
    configured: bool,
    closed: bool,
    frames_drawn: u32,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let secs: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);
    let width: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(640);
    let height: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(480);

    let conn = Connection::connect_to_env().expect("connect to wayland: WAYLAND_DISPLAY set?");
    let (globals, mut event_queue) = registry_queue_init::<State>(&conn).expect("registry init");
    let qh = event_queue.handle();

    let compositor: wl_compositor::WlCompositor = globals
        .bind(&qh, 4..=6, ())
        .expect("wl_compositor not advertised");
    let shm: wl_shm::WlShm = globals.bind(&qh, 1..=2, ()).expect("wl_shm not advertised");
    let wm_base: xdg_wm_base::XdgWmBase = globals
        .bind(&qh, 1..=6, ())
        .expect("xdg_wm_base not advertised");

    let mut state = State {
        compositor,
        shm,
        wm_base,
        width,
        height,
        surface: None,
        xdg_surface: None,
        toplevel: None,
        configured: false,
        closed: false,
        frames_drawn: 0,
    };

    // Create surface + xdg_toplevel and commit-without-buffer to request
    // the initial configure.
    let surface = state.compositor.create_surface(&qh, ());
    let xdg_surface = state.wm_base.get_xdg_surface(&surface, &qh, ());
    let toplevel = xdg_surface.get_toplevel(&qh, ());
    toplevel.set_title("prism-shmtest".to_string());
    toplevel.set_app_id("prism.shmtest".to_string());
    surface.commit();
    state.surface = Some(surface);
    state.xdg_surface = Some(xdg_surface);
    state.toplevel = Some(toplevel);

    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut last_redraw = Instant::now();
    let poll_fd = conn.backend().poll_fd().as_raw_fd();

    eprintln!(
        "prism-shmtest: connected, {}x{} {}s, waiting for configure…",
        state.width, state.height, secs
    );

    // Tight poll loop: flush outgoing, wait briefly for incoming, dispatch,
    // tick redraw. We can't use blocking_dispatch because this compositor
    // doesn't auto-fire wl_callback.frame events, so after the first commit
    // the server has nothing to send and blocking_dispatch would never
    // return.
    while Instant::now() < deadline && !state.closed {
        event_queue.flush().expect("flush");

        // Wait briefly for events on the connection fd.
        let mut pollfd = libc::pollfd {
            fd: poll_fd,
            events: libc::POLLIN,
            revents: 0,
        };
        let _ = unsafe { libc::poll(&mut pollfd, 1, 50) };
        if (pollfd.revents & libc::POLLIN) != 0 {
            if let Some(guard) = event_queue.prepare_read() {
                let _ = guard.read();
            }
        }

        event_queue
            .dispatch_pending(&mut state)
            .expect("dispatch_pending");

        if state.configured && last_redraw.elapsed() >= Duration::from_secs(1) {
            draw_frame(&qh, &mut state);
            last_redraw = Instant::now();
        }
    }
    eprintln!(
        "prism-shmtest: exit (frames_drawn={}, closed={})",
        state.frames_drawn, state.closed
    );
}

/// Allocate a fresh shm-backed wl_buffer, fill it with a per-frame color,
/// attach it to the surface and commit.
fn draw_frame(qh: &QueueHandle<State>, state: &mut State) {
    let stride = state.width * BPP;
    let size = (stride * state.height) as usize;

    // memfd-backed pool; we mmap it so we can fill pixels, the server
    // mmaps it from the fd we pass.
    let fd: OwnedFd = rustix::fs::memfd_create(c"prism-shmtest", rustix::fs::MemfdFlags::CLOEXEC)
        .expect("memfd_create");
    rustix::fs::ftruncate(&fd, size as u64).expect("ftruncate");
    let mut mmap = unsafe { MmapMut::map_mut(&fd) }.expect("mmap");

    // Rotate hue per frame so successive uploads are visually distinct.
    let t = state.frames_drawn;
    let (r, g, b): (u8, u8, u8) = match t % 6 {
        0 => (0xff, 0x40, 0x40), // red
        1 => (0xff, 0xa0, 0x40), // orange
        2 => (0xff, 0xff, 0x40), // yellow
        3 => (0x40, 0xff, 0x60), // green
        4 => (0x40, 0xa0, 0xff), // blue
        _ => (0xa0, 0x60, 0xff), // purple
    };
    // wl_shm Xrgb8888 layout in memory: B, G, R, X (little-endian ARGB word).
    for pixel in mmap.chunks_exact_mut(BPP as usize) {
        pixel[0] = b;
        pixel[1] = g;
        pixel[2] = r;
        pixel[3] = 0xff;
    }
    mmap.flush().ok();

    let pool = state.shm.create_pool(fd.as_fd(), size as i32, qh, ());
    let buffer = pool.create_buffer(
        0,
        state.width as i32,
        state.height as i32,
        stride as i32,
        FORMAT,
        qh,
        (),
    );
    pool.destroy();

    let surface = state.surface.as_ref().unwrap();
    surface.attach(Some(&buffer), 0, 0);
    surface.damage_buffer(0, 0, state.width as i32, state.height as i32);
    surface.commit();

    state.frames_drawn += 1;
    eprintln!(
        "prism-shmtest: drew frame #{} ({:#04x}{:02x}{:02x})",
        state.frames_drawn, r, g, b
    );
}

// ─── Dispatch impls — all minimal, just sink unwanted events ────────────────

impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for State {
    fn event(
        _state: &mut Self,
        _proxy: &wl_registry::WlRegistry,
        _event: wl_registry::Event,
        _data: &GlobalListContents,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        // globals already bound via registry_queue_init.
    }
}

impl Dispatch<wl_compositor::WlCompositor, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &wl_compositor::WlCompositor,
        _event: wl_compositor::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_shm::WlShm, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &wl_shm::WlShm,
        event: wl_shm::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        // wl_shm::Format events tell us what additional formats the
        // server supports. We hardcode Xrgb8888 (mandatory) so we ignore.
        let _ = event;
    }
}

impl Dispatch<wl_shm_pool::WlShmPool, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &wl_shm_pool::WlShmPool,
        _event: wl_shm_pool::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_buffer::WlBuffer, ()> for State {
    fn event(
        _state: &mut Self,
        proxy: &wl_buffer::WlBuffer,
        event: wl_buffer::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let wl_buffer::Event::Release = event {
            // Server's done with it. Drop our reference so the memfd
            // closes and resources clean up.
            proxy.destroy();
        }
    }
}

impl Dispatch<wl_surface::WlSurface, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &wl_surface::WlSurface,
        _event: wl_surface::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<xdg_wm_base::XdgWmBase, ()> for State {
    fn event(
        _state: &mut Self,
        proxy: &xdg_wm_base::XdgWmBase,
        event: xdg_wm_base::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            proxy.pong(serial);
        }
    }
}

impl Dispatch<xdg_surface::XdgSurface, ()> for State {
    fn event(
        state: &mut Self,
        proxy: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        _data: &(),
        _conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            proxy.ack_configure(serial);
            state.configured = true;
            // Draw the first frame immediately on configure.
            draw_frame(qh, state);
        }
    }
}

impl Dispatch<xdg_toplevel::XdgToplevel, ()> for State {
    fn event(
        state: &mut Self,
        _proxy: &xdg_toplevel::XdgToplevel,
        event: xdg_toplevel::Event,
        _data: &(),
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        if let xdg_toplevel::Event::Close = event {
            state.closed = true;
        }
        // We accept any size the compositor proposes (don't track configure
        // dims separately) — our buffer stays at the requested width/height.
        let _ = event;
    }
}

// Allow WEnum unused in some configurations.
const _: Option<WEnum<wl_shm::Format>> = None;
