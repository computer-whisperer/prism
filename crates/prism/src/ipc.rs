//! IPC server — Unix-socket request/reply using the `prism_ipc` vocabulary.
//!
//! Protocol: one JSON `Request` per line in, one JSON `Reply` per line
//! out. Each connection is one-shot today (client sends request, gets
//! reply, both sides close). The long-lived event-stream form (niri's
//! `EventStream` request) is future work.
//!
//! Dispatch is single-threaded on the main calloop — requests cannot
//! interleave with each other or with the render loop, which is the
//! right semantics for state-mutating requests (e.g. a calibration
//! tool flipping color overrides between sweep frames). Connection
//! I/O, however, never blocks the loop: each accepted stream becomes
//! its own nonblocking calloop source that accumulates bytes until the
//! request line is complete, dispatches, and writes the reply —
//! re-arming on write interest if the socket buffer fills. A client
//! that connects and sends nothing (or trickles bytes) costs nothing
//! per se; a wall-clock deadline reaps connections that overstay so
//! they can't hold fds forever.
//!
//! Bringup wires in `insert_ipc_source` from main.rs after the
//! wayland sources are up. The socket path lives in `$XDG_RUNTIME_DIR`
//! and is exported via `PRISM_SOCKET` so child processes (and any
//! `prism-tune` invocation) can find us.

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{ErrorKind, Read};
use std::os::fd::OwnedFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Duration;

use anyhow::{Context, Result};
use calloop::generic::Generic;
use calloop::timer::{TimeoutAction, Timer};
use calloop::{Interest, LoopHandle, Mode, PostAction, RegistrationToken};
use prism_ipc::{
    self, ColorState, LogicalOutput, Output, OutputAction, OutputConfigChanged, Reply, Request,
    Response, ResponseCurveState, Transform,
};
use prism_layout::utils::with_toplevel_role;
use prism_protocols::PrismState;
use smithay::utils::Size;

/// Insert the IPC `UnixListener` calloop source. Binds a fresh socket
/// in `$XDG_RUNTIME_DIR` (falling back to `/tmp`), removes any stale
/// file at that path, sets `PRISM_SOCKET` for child processes, and
/// returns the path so the caller can log + arrange cleanup on exit.
pub fn insert_ipc_source(handle: &LoopHandle<'static, PrismState>) -> Result<PathBuf> {
    let path = default_socket_path();
    // Stale socket left over from a previous run: remove it so bind() doesn't
    // EADDRINUSE. (Other compositors do the same; correct because we're
    // the only writer of this path.)
    let _ = std::fs::remove_file(&path);
    let listener =
        UnixListener::bind(&path).with_context(|| format!("bind ipc socket {}", path.display()))?;
    listener.set_nonblocking(true).context("set_nonblocking")?;

    // SAFETY: set_var is sound while we're still single-threaded at
    // server-startup time (matches the WAYLAND_DISPLAY pattern in
    // prism-protocols::server).
    unsafe {
        std::env::set_var(prism_ipc::socket::SOCKET_PATH_ENV, &path);
    }

    let conn_handle = handle.clone();
    handle
        .insert_source(
            Generic::new(listener, Interest::READ, Mode::Level),
            move |_event, listener, _state| {
                loop {
                    match listener.accept() {
                        Ok((stream, _addr)) => {
                            if let Err(e) = insert_connection(&conn_handle, stream) {
                                tracing::warn!("ipc: failed to register connection: {e:#}");
                            }
                        }
                        Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                        Err(e) => {
                            tracing::warn!("ipc accept error: {e}");
                            break;
                        }
                    }
                }
                Ok(PostAction::Continue)
            },
        )
        .map_err(|e| anyhow::anyhow!("insert ipc source: {e}"))?;

    tracing::info!(path = %path.display(), "ipc socket listening");
    Ok(path)
}

/// Decide where the IPC socket lives. Prefers `$XDG_RUNTIME_DIR` (the
/// usual home for user-scoped sockets); falls back to `/tmp` if that
/// isn't set. Filename includes the PID so two prism instances on the
/// same machine don't collide; child processes inherit
/// `PRISM_SOCKET` and find ours by env var, not by guessing.
fn default_socket_path() -> PathBuf {
    let base: PathBuf = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    base.join(format!("prism-{}.sock", std::process::id()))
}

/// Ceiling on a connection's lifetime, accept to reply-flushed. Nothing
/// blocks the loop on a slow client — this is garbage collection, so an
/// idle or wedged peer can't hold its fd (and calloop slot) forever.
/// Well-behaved clients complete in single-digit milliseconds.
const CONNECTION_DEADLINE: Duration = Duration::from_secs(5);

/// Request lines are single small JSON objects; anything past this is a
/// confused (or hostile) client and the connection is dropped.
const MAX_REQUEST_BYTES: usize = 16 * 1024;

/// Register an accepted stream as its own nonblocking calloop source.
/// The source accumulates bytes until the request line is complete,
/// dispatches on the main loop, writes the reply, then closes — one
/// request per connection; clients that want more reconnect, which is
/// what `prism_ipc::socket::Socket` does anyway. A companion timer
/// enforces [`CONNECTION_DEADLINE`] over the whole exchange.
fn insert_connection(handle: &LoopHandle<'static, PrismState>, stream: UnixStream) -> Result<()> {
    stream.set_nonblocking(true).context("set_nonblocking")?;

    // Whichever source currently owns the connection (reader, then
    // possibly a reply writer). The deadline timer kills it through
    // this slot; every normal-completion path clears the slot and
    // removes the timer.
    let conn_token: Rc<RefCell<Option<RegistrationToken>>> = Rc::new(RefCell::new(None));

    let timer_handle = handle.clone();
    let timer_conn_token = conn_token.clone();
    let timer_token = handle
        .insert_source(
            Timer::from_duration(CONNECTION_DEADLINE),
            move |_deadline, _, _state| {
                if let Some(token) = timer_conn_token.borrow_mut().take() {
                    tracing::warn!(
                        "ipc: connection exceeded {CONNECTION_DEADLINE:?} deadline, dropping"
                    );
                    timer_handle.remove(token);
                }
                TimeoutAction::Drop
            },
        )
        .map_err(|e| anyhow::anyhow!("insert connection deadline timer: {e}"))?;

    let read_handle = handle.clone();
    let read_conn_token = conn_token.clone();
    let mut buf: Vec<u8> = Vec::new();
    let inserted = handle.insert_source(
        Generic::new(stream, Interest::READ, Mode::Level),
        move |_event, stream, state| {
            let mut chunk = [0u8; 4096];
            let line_end = loop {
                if let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    break pos;
                }
                let mut reader: &UnixStream = stream;
                match reader.read(&mut chunk) {
                    Ok(0) => {
                        // EOF before a complete request line: client gave up.
                        finish_connection(&read_handle, &read_conn_token, timer_token);
                        return Ok(PostAction::Remove);
                    }
                    Ok(n) => {
                        buf.extend_from_slice(&chunk[..n]);
                        if buf.len() > MAX_REQUEST_BYTES {
                            tracing::warn!(
                                "ipc: request exceeds {MAX_REQUEST_BYTES} bytes, dropping connection"
                            );
                            finish_connection(&read_handle, &read_conn_token, timer_token);
                            return Ok(PostAction::Remove);
                        }
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock => {
                        return Ok(PostAction::Continue);
                    }
                    Err(e) if e.kind() == ErrorKind::Interrupted => {}
                    Err(e) => {
                        tracing::warn!("ipc: read error: {e}");
                        finish_connection(&read_handle, &read_conn_token, timer_token);
                        return Ok(PostAction::Remove);
                    }
                }
            };

            let line = String::from_utf8_lossy(&buf[..line_end]);
            let (reply, fd): (Reply, Option<OwnedFd>) =
                match serde_json::from_str::<Request>(&line) {
                    Ok(req) => dispatch(state, req),
                    Err(e) => (Err(format!("parse request: {e}")), None),
                };
            let mut text = match serde_json::to_string(&reply) {
                Ok(text) => text,
                Err(e) => {
                    tracing::warn!("ipc: serialize reply: {e}");
                    finish_connection(&read_handle, &read_conn_token, timer_token);
                    return Ok(PostAction::Remove);
                }
            };
            text.push('\n');
            let mut pending = PendingReply {
                bytes: text.into_bytes(),
                sent: 0,
                fd,
            };

            match pending.try_write(stream) {
                Ok(true) => {
                    finish_connection(&read_handle, &read_conn_token, timer_token);
                    Ok(PostAction::Remove)
                }
                Ok(false) => {
                    // Socket send buffer full mid-reply (large reply and/or
                    // slow reader): hand the remainder to a write-interest
                    // source on a dup'd fd and retire this read source. The
                    // deadline timer stays armed and covers the writer too.
                    insert_reply_writer(
                        &read_handle,
                        stream,
                        pending,
                        &read_conn_token,
                        timer_token,
                    );
                    Ok(PostAction::Remove)
                }
                Err(e) => {
                    tracing::warn!("ipc: write error: {e}");
                    finish_connection(&read_handle, &read_conn_token, timer_token);
                    Ok(PostAction::Remove)
                }
            }
        },
    );
    match inserted {
        Ok(token) => {
            *conn_token.borrow_mut() = Some(token);
            Ok(())
        }
        Err(e) => {
            handle.remove(timer_token);
            Err(anyhow::anyhow!("insert connection source: {e}"))
        }
    }
}

/// Normal-completion teardown shared by every connection path: clear the
/// live-source slot (so the deadline timer finds nothing to kill) and
/// retire the timer. The calling source removes itself by returning
/// [`PostAction::Remove`].
fn finish_connection(
    handle: &LoopHandle<'static, PrismState>,
    conn_token: &RefCell<Option<RegistrationToken>>,
    timer_token: RegistrationToken,
) {
    conn_token.borrow_mut().take();
    handle.remove(timer_token);
}

/// A serialized reply mid-flight to a client that isn't draining its
/// socket as fast as we fill it. `fd` rides the first byte actually
/// accepted (see `write_chunk_with_fd`).
struct PendingReply {
    bytes: Vec<u8>,
    sent: usize,
    fd: Option<OwnedFd>,
}

impl PendingReply {
    /// Push as much of the reply as the socket accepts right now.
    /// `Ok(true)` = fully written; `Ok(false)` = `WouldBlock`, re-arm.
    fn try_write(&mut self, stream: &UnixStream) -> std::io::Result<bool> {
        while self.sent < self.bytes.len() {
            match prism_ipc::socket::write_chunk_with_fd(
                stream,
                &self.bytes[self.sent..],
                &mut self.fd,
            ) {
                Ok(n) => self.sent += n,
                Err(e) if e.kind() == ErrorKind::WouldBlock => return Ok(false),
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(true)
    }
}

/// Continue an interrupted reply write on a dedicated write-interest
/// source (the read source retires itself after calling this). Works on
/// a dup of the stream's fd — same open socket, independently owned by
/// the new source. On registration failure the connection just drops;
/// the client sees EOF mid-reply.
fn insert_reply_writer(
    handle: &LoopHandle<'static, PrismState>,
    stream: &UnixStream,
    mut pending: PendingReply,
    conn_token: &Rc<RefCell<Option<RegistrationToken>>>,
    timer_token: RegistrationToken,
) {
    let dup = match stream.try_clone() {
        Ok(dup) => dup,
        Err(e) => {
            tracing::warn!("ipc: dup for reply writer failed: {e}");
            finish_connection(handle, conn_token, timer_token);
            return;
        }
    };
    let writer_handle = handle.clone();
    let writer_conn_token = conn_token.clone();
    let inserted = handle.insert_source(
        Generic::new(dup, Interest::WRITE, Mode::Level),
        move |_event, stream, _state| match pending.try_write(stream) {
            Ok(true) => {
                finish_connection(&writer_handle, &writer_conn_token, timer_token);
                Ok(PostAction::Remove)
            }
            Ok(false) => Ok(PostAction::Continue),
            Err(e) => {
                tracing::warn!("ipc: write error: {e}");
                finish_connection(&writer_handle, &writer_conn_token, timer_token);
                Ok(PostAction::Remove)
            }
        },
    );
    match inserted {
        Ok(token) => {
            *conn_token.borrow_mut() = Some(token);
        }
        Err(e) => {
            tracing::warn!("ipc: failed to register reply writer: {e}");
            finish_connection(handle, conn_token, timer_token);
        }
    }
}

/// Route a parsed Request to the matching handler. Unsupported variants
/// return `Err("not implemented")` rather than panic — clients then know
/// which surface area is still future work.
fn dispatch(state: &mut PrismState, req: Request) -> (Reply, Option<OwnedFd>) {
    match req {
        Request::Version => (
            Ok(Response::Version(env!("CARGO_PKG_VERSION").to_string())),
            None,
        ),
        Request::Outputs => (Ok(Response::Outputs(collect_outputs(state))), None),
        Request::FocusedOutput => {
            // The layout's active monitor is the focused output (the one
            // carrying the focus ring).
            let focused = state.active_output().and_then(|active| {
                let id = state
                    .wl_outputs
                    .iter()
                    .find(|(_, wl)| **wl == active)
                    .map(|(id, _)| id.clone())?;
                collect_outputs(state).remove(&id)
            });
            (Ok(Response::FocusedOutput(focused)), None)
        }
        Request::Workspaces => (Ok(Response::Workspaces(collect_workspaces(state))), None),
        Request::Windows => (Ok(Response::Windows(collect_windows(state))), None),
        Request::Output { output, action } => (handle_output_action(state, &output, action), None),
        Request::CaptureFrame { output } => handle_capture_frame(state, &output),
        Request::GamutMesh { output } => (handle_gamut_mesh(state, &output), None),
        Request::Lut3d { output } => handle_lut3d(state, &output),
        Request::ProfilingStats { output } => handle_profiling_stats(state, &output),
        other => (
            Err(format!(
                "request {other:?} is not implemented in this build"
            )),
            None,
        ),
    }
}

/// Capture an output's intermediate frame into a memfd and return the
/// fd alongside a [`Response::FrameCaptured`] describing its layout.
/// The fd is passed out-of-band (`SCM_RIGHTS`) by `handle_connection`.
fn handle_capture_frame(state: &mut PrismState, name: &str) -> (Reply, Option<OwnedFd>) {
    let Some(output_id) = state
        .outputs
        .iter()
        .find(|(_id, ctx)| ctx.connector_name == name)
        .map(|(id, _)| id.clone())
    else {
        return (
            Err(format!("capture-frame: output {name:?} not found")),
            None,
        );
    };

    let output_ctx = state.outputs.get(&output_id).expect("just found above");
    match output_ctx.renderer.capture_intermediate() {
        Ok(frame) => {
            let meta = prism_ipc::FrameMeta {
                width: frame.width,
                height: frame.height,
                stride_bytes: frame.stride_bytes,
                byte_len: frame.byte_len,
                format: prism_ipc::FrameFormat::Rgba32Float,
            };
            (Ok(Response::FrameCaptured(meta)), Some(frame.fd))
        }
        Err(e) => (Err(format!("capture-frame: {e:#}")), None),
    }
}

/// Serve the effective 3D LUT for `name` — the exact entries
/// `resynthesize_color_lut` would upload (same precedence chain), so the
/// inspector sees what the GPU is running even when it was synthesized
/// and never written to disk. Entries travel out-of-band in a memfd
/// (three LE `f32`s each, X-fastest) alongside [`Response::Lut3d`].
fn handle_lut3d(state: &mut PrismState, name: &str) -> (Reply, Option<OwnedFd>) {
    let Some(ctx) = state
        .outputs
        .values()
        .find(|ctx| ctx.connector_name == name)
    else {
        return (Err(format!("lut3d: output {name:?} not found")), None);
    };
    let Some((entries, source)) = ctx.effective_lut3d_entries() else {
        return (
            Err(format!(
                "lut3d: output {name:?} has no LUT slot in its encode chain"
            )),
            None,
        );
    };
    let mut bytes = Vec::with_capacity(entries.len() * 12);
    for entry in &entries {
        for channel in entry {
            bytes.extend_from_slice(&channel.to_le_bytes());
        }
    }
    let fd = match prism_ipc::socket::memfd_from_bytes("prism-lut3d", &bytes) {
        Ok(fd) => fd,
        Err(e) => return (Err(format!("lut3d: memfd: {e}")), None),
    };
    let meta = prism_ipc::Lut3dMeta {
        cube_edge: ctx.renderer.lut3d_cube_edge(),
        byte_len: bytes.len() as u64,
        out_space: match ctx.config.encode_config.lut_output_domain() {
            prism_renderer::LutOutputDomain::Nits => prism_ipc::Lut3dDomain::Nits,
            prism_renderer::LutOutputDomain::Drive => prism_ipc::Lut3dDomain::Drive,
        },
        source: match source {
            prism_drm::LutSource::IpcOverride => prism_ipc::Lut3dSource::IpcOverride,
            prism_drm::LutSource::KdlFile => prism_ipc::Lut3dSource::KdlFile,
            prism_drm::LutSource::Synthesized => prism_ipc::Lut3dSource::Synthesized,
        },
    };
    (Ok(Response::Lut3d(meta)), Some(fd))
}

/// Snapshot an output's live render-profiling readout: the per-span percentile
/// aggregate inline, and the raw per-frame timeline out-of-band in a memfd
/// (`timeline_frames × N_SPANS` LE `f32`, oldest → newest — see
/// [`prism_ipc::ProfilingStats`]). Pull-based; prism-tune polls it while its
/// profiling panel is open.
fn handle_profiling_stats(state: &mut PrismState, name: &str) -> (Reply, Option<OwnedFd>) {
    let Some(ctx) = state
        .outputs
        .values()
        .find(|ctx| ctx.connector_name == name)
    else {
        return (Err(format!("profiling: output {name:?} not found")), None);
    };
    let Some(readout) = ctx.renderer.profile_readout() else {
        return (
            Err(format!(
                "profiling: output {name:?} has no profile data yet \
                 (just started, or PRISM_NO_PROFILE set)"
            )),
            None,
        );
    };
    // Timeline payload: one record of N_SPANS LE f32 per frame, oldest → newest.
    let mut bytes = Vec::with_capacity(readout.timeline.len() * prism_renderer::N_SPANS * 4);
    for frame in &readout.timeline {
        for v in frame {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let fd = match prism_ipc::socket::memfd_from_bytes("prism-profiling", &bytes) {
        Ok(fd) => fd,
        Err(e) => return (Err(format!("profiling: memfd: {e}")), None),
    };
    let s = &readout.summary;
    let stats = prism_ipc::ProfilingStats {
        span_names: prism_renderer::SPAN_NAMES
            .iter()
            .map(|n| n.to_string())
            .collect(),
        percentiles_us: s.spans.iter().map(|st| [st.p50, st.p95, st.p99]).collect(),
        frames: s.frames as u32,
        damage_ratio_p50: s.damage_ratio_p50,
        elements_p50: s.elements_p50,
        timeline_frames: readout.timeline.len() as u32,
        byte_len: bytes.len() as u64,
    };
    (Ok(Response::ProfilingStats(stats)), Some(fd))
}

/// Load + return the measured gamut-surface mesh configured for `name`
/// (KDL `color.gamut`), or `Response::GamutMesh(None)` when none is set.
/// Parses the `.gamut.json` sidecar on demand — it's inspector-only data,
/// not worth holding in memory per output.
fn handle_gamut_mesh(state: &PrismState, name: &str) -> Reply {
    let Some(ctx) = state
        .outputs
        .values()
        .find(|ctx| ctx.connector_name == name)
    else {
        return Err(format!("gamut-mesh: output {name:?} not found"));
    };
    let Some(path) = ctx.kdl_gamut_path.as_ref() else {
        return Ok(Response::GamutMesh(None));
    };
    match prism_ipc::GamutMesh::load_json(path) {
        Ok(mesh) => Ok(Response::GamutMesh(Some(mesh))),
        Err(e) => Err(format!("gamut-mesh: load {} failed: {e}", path.display())),
    }
}

/// Build the IPC `Output` info map for every live `OutputContext`.
fn collect_outputs(state: &PrismState) -> HashMap<String, Output> {
    let mut map = HashMap::new();
    for ctx in state.outputs.values() {
        let mode = ctx.mode;
        let (mw, mh) = mode.size();
        let modes = vec![prism_ipc::Mode {
            width: mw,
            height: mh,
            refresh_rate: mode.vrefresh() * 1000,
            is_preferred: true,
        }];
        let logical = state.wl_outputs.get(&ctx.connector_name).and_then(|wl| {
            let monitor = state.layout.monitor_for_output(wl)?;
            let view: Size<f64, smithay::utils::Logical> = monitor.view_size();
            // Position/scale/transform live on the smithay Output —
            // `layout_outputs()` assigns the location, the config the
            // rest. Region-capture tooling (slurp-style) keys off these.
            let loc = wl.current_location();
            Some(LogicalOutput {
                x: loc.x,
                y: loc.y,
                width: view.w as u32,
                height: view.h as u32,
                scale: wl.current_scale().fractional_scale(),
                transform: ipc_transform(wl.current_transform()),
            })
        });
        // Snapshot the effective color pipeline state so callers
        // (calibration tooling, debug UIs) can branch on HDR/SDR mode
        // and read the current per-channel peaks + response curve
        // without a second round-trip.
        let peaks = ctx.effective_panel_peak_nits_rgb();
        let curve = ctx
            .effective_response_curve()
            .map(|(gain, gamma)| ResponseCurveState {
                gain: [gain[0] as f64, gain[1] as f64, gain[2] as f64],
                gamma: [gamma[0] as f64, gamma[1] as f64, gamma[2] as f64],
            });
        let ctm = ctx.effective_ctm().map(|m| {
            [
                [m[0][0] as f64, m[0][1] as f64, m[0][2] as f64],
                [m[1][0] as f64, m[1][1] as f64, m[1][2] as f64],
                [m[2][0] as f64, m[2][1] as f64, m[2][2] as f64],
            ]
        });
        let color = ColorState {
            hdr_active: ctx.config.hdr.is_some(),
            panel_peak_nits: [peaks[0] as f64, peaks[1] as f64, peaks[2] as f64],
            sdr_reference_nits: ctx.effective_sdr_reference_nits() as f64,
            response_curve: curve,
            ctm,
            advertised_peak_nits: ctx.effective_advertised_peak_nits().map(|v| v as f64),
        };
        let info = Output {
            name: ctx.connector_name.clone(),
            make: ctx.edid.make.clone().unwrap_or_default(),
            model: ctx.edid.model.clone().unwrap_or_default(),
            serial: ctx.edid.serial.clone(),
            physical_size: ctx.edid.size_mm,
            modes,
            current_mode: Some(0),
            is_custom_mode: false,
            vrr_supported: ctx.config.vrr,
            vrr_enabled: ctx.config.vrr,
            logical,
            color,
        };
        map.insert(ctx.connector_name.clone(), info);
    }
    map
}

/// smithay output transform → IPC vocabulary.
fn ipc_transform(t: smithay::utils::Transform) -> Transform {
    use smithay::utils::Transform as T;
    match t {
        T::Normal => Transform::Normal,
        T::_90 => Transform::_90,
        T::_180 => Transform::_180,
        T::_270 => Transform::_270,
        T::Flipped => Transform::Flipped,
        T::Flipped90 => Transform::Flipped90,
        T::Flipped180 => Transform::Flipped180,
        T::Flipped270 => Transform::Flipped270,
    }
}

/// Build the IPC `Workspace` list straight from the layout. (niri serves
/// this from cached event-stream state because its IPC thread can't
/// touch the compositor; prism dispatches on the main calloop and reads
/// the layout directly.)
fn collect_workspaces(state: &PrismState) -> Vec<prism_ipc::Workspace> {
    let focused_ws_id = state.layout.active_workspace().map(|ws| ws.id().get());
    state
        .layout
        .workspaces()
        .map(|(mon, ws_idx, ws)| {
            let id = ws.id().get();
            prism_ipc::Workspace {
                id,
                idx: u8::try_from(ws_idx + 1).unwrap_or(u8::MAX),
                name: ws.name().cloned(),
                output: mon.map(|mon| mon.output_name().clone()),
                is_urgent: ws.is_urgent(),
                is_active: mon.is_some_and(|mon| mon.active_workspace_idx() == ws_idx),
                is_focused: Some(id) == focused_ws_id,
                active_window_id: ws.active_window().map(|win| win.id().get()),
            }
        })
        .collect()
}

/// Build the IPC `Window` list straight from the layout (same rationale
/// as [`collect_workspaces`]). The `WindowLayout` comes ready-made from
/// the layout walk.
fn collect_windows(state: &PrismState) -> Vec<prism_ipc::Window> {
    let mut windows = Vec::new();
    state.layout.with_windows(|mapped, _, ws_id, layout| {
        let window = with_toplevel_role(mapped.toplevel(), |role| prism_ipc::Window {
            id: mapped.id().get(),
            title: role.title.clone(),
            app_id: role.app_id.clone(),
            pid: mapped.credentials().map(|c| c.pid),
            workspace_id: ws_id.map(|id| id.get()),
            is_focused: mapped.is_focused(),
            is_floating: mapped.is_floating(),
            is_urgent: mapped.is_urgent(),
            layout,
            focus_timestamp: mapped.get_focus_timestamp().map(prism_ipc::Timestamp::from),
        });
        windows.push(window);
    });
    windows
}

/// Apply a state-mutating `OutputAction`. Color-related variants are
/// the only ones implemented today; modeset / scale / etc. return
/// `Err("not implemented")` until the corresponding compositor
/// machinery is plumbed.
fn handle_output_action(state: &mut PrismState, name: &str, action: OutputAction) -> Reply {
    let Some(output_id) = state
        .outputs
        .iter()
        .find(|(_id, ctx)| ctx.connector_name == name)
        .map(|(id, _)| id.clone())
    else {
        // Niri's "OutputWasMissing" semantic: a Request::Output for an
        // absent output isn't an error — it's just queued. We don't
        // have a queue today, so report it but still succeed.
        return Ok(Response::OutputConfigChanged(
            OutputConfigChanged::OutputWasMissing,
        ));
    };

    let output_ctx = state.outputs.get_mut(&output_id).expect("just found above");
    // Track whether this mutation changed something the per-output 3D LUT
    // depends on. CTM and response-curve obviously do, and so does
    // SdrReferenceNits on drive-domain (SDR) outputs — the synthesized
    // fallback LUT bakes the nits → drive anchor in. PanelPeakNits
    // affects the encode pipeline elsewhere (decode clamp) but not LUT
    // *contents*, so it skips the re-synthesis.
    let mut lut_dirty = false;
    // Set when a mutation changes what color-management clients are told
    // (the advertised mastering peak) without touching the LUT or encode
    // push — triggers a rebuild + re-advertise of the output's preferred
    // image description after the match.
    let mut readvertise = false;
    match action {
        OutputAction::SdrReferenceNits { nits } => {
            let v = (nits.clamp(1.0, 10_000.0)) as f32;
            output_ctx.color_override.sdr_reference_nits = Some(v);
            tracing::info!(connector = %name, sdr_reference_nits = v, "ipc: set sdr-reference-nits override");
            // Decode-side policy, plus the anchor of the *synthesized*
            // fallback LUT on drive-domain outputs. Measured LUTs
            // (IPC/KDL precedence levels) are absolute and unaffected —
            // resynthesize_color_lut re-uploads whichever source wins.
            lut_dirty = true;
        }
        OutputAction::ResponseCurve {
            gain_r,
            gain_g,
            gain_b,
            gamma_r,
            gamma_g,
            gamma_b,
        } => {
            let g_r = (gain_r as f32).clamp(0.01, 10.0);
            let g_g = (gain_g as f32).clamp(0.01, 10.0);
            let g_b = (gain_b as f32).clamp(0.01, 10.0);
            let y_r = (gamma_r as f32).clamp(0.1, 10.0);
            let y_g = (gamma_g as f32).clamp(0.1, 10.0);
            let y_b = (gamma_b as f32).clamp(0.1, 10.0);
            output_ctx.color_override.response_curve = Some(([g_r, g_g, g_b], [y_r, y_g, y_b]));
            tracing::info!(
                connector = %name,
                gain = ?[g_r, g_g, g_b],
                gamma = ?[y_r, y_g, y_b],
                "ipc: set response-curve override"
            );
            lut_dirty = true;
        }
        OutputAction::PanelPeakNits {
            nits_r,
            nits_g,
            nits_b,
        } => {
            let r = (nits_r as f32).clamp(1.0, 10_000.0);
            let g = (nits_g as f32).clamp(1.0, 10_000.0);
            let b = (nits_b as f32).clamp(1.0, 10_000.0);
            output_ctx.color_override.panel_peak_nits_rgb = Some([r, g, b]);
            tracing::info!(
                connector = %name,
                panel_peak_nits_rgb = ?[r, g, b],
                "ipc: set panel-peak-nits override"
            );
            // Honest mid-session: re-push the configured HDR infoframe
            // after the runtime color state changes. The infoframe uses
            // the KDL HDR signaling values, not measured subpixel peaks.
            // No-op for SDR outputs.
            if let Err(e) = output_ctx.rebuild_hdr_infoframe() {
                tracing::warn!(
                    connector = %name,
                    "panel-peak-nits applied but HDR infoframe rebuild failed: {e:#}"
                );
            }
        }
        OutputAction::AdvertisedPeakNits { nits } => {
            let v = (nits.round().clamp(1.0, 10_000.0)) as u32;
            output_ctx.color_override.advertised_peak_nits = Some(v);
            tracing::info!(
                connector = %name,
                advertised_peak_nits = v,
                "ipc: set advertised-peak-nits override"
            );
            // Client-facing only — no LUT/encode change, no infoframe
            // change. Just re-advertise the preferred description.
            readvertise = true;
        }
        OutputAction::Ctm {
            rr,
            rg,
            rb,
            gr,
            gg,
            gb,
            br,
            bg,
            bb,
        } => {
            let m = [
                [rr as f32, rg as f32, rb as f32],
                [gr as f32, gg as f32, gb as f32],
                [br as f32, bg as f32, bb as f32],
            ];
            output_ctx.color_override.ctm = Some(m);
            tracing::info!(
                connector = %name,
                ctm = ?m,
                "ipc: set ctm override"
            );
            lut_dirty = true;
        }
        OutputAction::LoadLut3dFromFile { path } => {
            let loaded = match prism_renderer::load_lut3d_file(std::path::Path::new(&path)) {
                Ok(l) => l,
                Err(e) => {
                    return Err(format!("load_lut3d_file({path}) failed: {e:#}"));
                }
            };
            let renderer_edge = output_ctx.renderer.lut3d_cube_edge();
            if loaded.cube_edge != renderer_edge {
                return Err(format!(
                    "LUT file cube_edge={} doesn't match renderer cube_edge={}",
                    loaded.cube_edge, renderer_edge,
                ));
            }
            // Domain check: the chain's terminal OutputTransfer dictates
            // what the LUT entries must mean (nits for PQ/linear, drive
            // for sRGB). A mismatched file would render garbage — most
            // insidiously, a pre-v5 nits-space file on an SDR output
            // would re-create the silent dimming this reform removed.
            let want = output_ctx.config.encode_config.lut_output_domain();
            if loaded.out_space != want {
                return Err(format!(
                    "LUT file out_space={:?} doesn't match the output's encode chain ({want:?}); \
                     rebake with the current prism-tune (SDR LUTs are drive-domain v5 now)",
                    loaded.out_space,
                ));
            }
            let cube_edge = loaded.cube_edge;
            let bp = loaded.black_point_xyz;
            output_ctx.color_override.lut3d_entries = Some(loaded.entries);
            // Carry the measured black-point alongside — the v2+ file
            // format pairs them and the compositor uses the floor for
            // tone-map decisions + wp_color_management feedback. All-
            // zero ⇒ unmeasured; we treat that as "leave the override
            // unset" so the KDL-loaded value (if any) still shows
            // through via effective_black_point_xyz.
            if bp[0] != 0.0 || bp[1] != 0.0 || bp[2] != 0.0 {
                output_ctx.color_override.black_point_xyz = Some(bp);
            }
            tracing::info!(
                connector = %name,
                path = %path,
                cube_edge,
                black_point_xyz = ?bp,
                "ipc: loaded color LUT from file"
            );
            lut_dirty = true;
        }
        OutputAction::IdentityLut3d => {
            let cube_edge = output_ctx.renderer.lut3d_cube_edge();
            if cube_edge == 0 {
                return Err(
                    "IdentityLut3d: encode chain has no LUT slot (legacy CTM+curve path)"
                        .to_owned(),
                );
            }
            // Mode-appropriate "identity": nits passthrough for PQ/linear
            // chains; for drive-domain (sRGB) chains the uncalibrated
            // nits → drive mapping anchored at the effective reference
            // white, so a calibration sweep's patch code values round-
            // trip to the wire unchanged.
            let entries = match output_ctx.config.encode_config.lut_output_domain() {
                prism_renderer::LutOutputDomain::Nits => prism_renderer::identity_lut(cube_edge),
                prism_renderer::LutOutputDomain::Drive => prism_renderer::drive_identity_lut(
                    cube_edge,
                    output_ctx.effective_sdr_reference_nits(),
                ),
            };
            output_ctx.color_override.lut3d_entries = Some(entries);
            tracing::info!(
                connector = %name,
                cube_edge,
                "ipc: forced LUT to identity (raw-cmd mode)"
            );
            lut_dirty = true;
        }
        OutputAction::EncodeDiagnose { r, g, b } => {
            // Build the encode push the live render path would use for
            // this output — target_peak_nits influences the PQ/linear
            // OutputTransfer stage, so the diagnose must mirror it or
            // it'd test a different shader configuration than what the
            // panel sees. (The sRGB transfer is parameter-free.)
            let mut p = match output_ctx.config.hdr {
                Some(hdr) => {
                    let mut p = prism_renderer::EncodePushSynth::pq_identity();
                    p.target_peak_nits = hdr.max_luminance as f32;
                    p
                }
                None => prism_renderer::EncodePushSynth::sdr_identity(),
            };
            // CTM + per-channel curve are no longer read by the
            // Lut3d-only encode chain, but mirror them anyway so any
            // legacy-chain output stays equivalent.
            if let Some((gain, gamma)) = output_ctx.effective_response_curve() {
                p.set_response_gain_gamma(gain, gamma);
            }
            if let Some(m) = output_ctx.effective_ctm() {
                p.set_ctm(m);
            }
            let result = output_ctx
                .renderer
                .encode_diagnose([r, g, b], &p)
                .map_err(|e| {
                    // Mirror to tracing as well as the IPC reply so the
                    // error is recoverable from prism.log alone when the
                    // client's stderr wasn't captured (which is the
                    // common case for interactive prism-tune runs).
                    tracing::warn!(
                        connector = %name,
                        input = ?[r, g, b],
                        "ipc: encode_diagnose failed: {e:#}"
                    );
                    format!("encode_diagnose failed: {e:#}")
                })?;
            tracing::info!(
                connector = %name,
                input = ?[r, g, b],
                scanout = ?result,
                "ipc: encode_diagnose"
            );
            return Ok(Response::EncodeDiagnose(prism_ipc::EncodeDiagnoseResult {
                scanout_nits: result,
            }));
        }
        OutputAction::ResetColor => {
            output_ctx.color_override = prism_drm::ColorOverride::default();
            tracing::info!(connector = %name, "ipc: cleared color overrides");
            // The cleared panel-peak-nits reverts to the bringup KDL
            // value — push the corresponding infoframe.
            if let Err(e) = output_ctx.rebuild_hdr_infoframe() {
                tracing::warn!(
                    connector = %name,
                    "color reset applied but HDR infoframe rebuild failed: {e:#}"
                );
            }
            lut_dirty = true;
            // Clearing the override reverts the advertised peak to the
            // KDL value — re-advertise so clients pick that back up.
            readvertise = true;
        }
        other => {
            return Err(format!(
                "OutputAction {other:?} is not implemented in this build"
            ));
        }
    }

    // CTM / response-curve / reset all change what the LUT should hold;
    // re-synthesize from the new effective values. Failure is logged but
    // doesn't bubble — keeping the previous LUT is the least-surprising
    // fallback (calibration just doesn't update this round).
    if lut_dirty {
        if let Err(e) = output_ctx.resynthesize_color_lut() {
            tracing::warn!(
                connector = %name,
                "color LUT re-synthesis failed: {e:#} (LUT keeps previous content)"
            );
        }
    }

    // Rebuild + re-advertise the output's preferred image description
    // when the advertised mastering peak changed (or was reset). This is
    // independent of the render path — it only touches what
    // color-management clients are told.
    if readvertise {
        readvertise_output_preferred(state, &output_id);
    }

    // The recolor changes the encode output without moving any element, so the
    // damage tracker sees nothing — force the next present past the zero-damage
    // skip, or the change wouldn't reach the screen until something moved.
    if let Some(output_ctx) = state.outputs.get_mut(&output_id) {
        output_ctx.force_next_present();
    }
    // Override won't be observable until the next frame — kick a
    // redraw so users see the change immediately even on otherwise-
    // idle outputs.
    state
        .output_redraw
        .entry(output_id)
        .or_default()
        .queue_redraw();

    Ok(Response::OutputConfigChanged(OutputConfigChanged::Applied))
}

/// Rebuild an output's preferred `wp_color_management_v1` image
/// description from its current (post-override) color state and push it
/// to clients. Called after a tunable that only changes the advertised
/// metadata (the mastering peak) — the render path is untouched.
///
/// The rebuild allocates a fresh image-description identity; we cache it
/// as the output's preferred and send `preferred_changed2` to every
/// color-managed toplevel currently mapped to the output so it re-queries
/// the new value. Surfaces without a feedback object pick the new
/// description up on their next `get_preferred`.
fn readvertise_output_preferred(state: &mut PrismState, output_id: &str) {
    let Some(output_ctx) = state.outputs.get(output_id) else {
        return;
    };
    let preferred = prism_protocols::color_management::build_output_preferred(
        output_ctx,
        &state.color_management,
    );
    let identity = preferred.identity;
    state
        .color_management
        .set_output_preferred(output_id.to_owned(), preferred);
    tracing::info!(
        connector = %output_id,
        identity,
        "color-mgmt: output preferred re-advertised after ipc tune"
    );

    // Notify color-managed toplevels on this output. windows_for_output
    // panics on an output the layout doesn't track, so guard on the
    // monitor existing first (wl_outputs and the layout are added
    // together at bringup, but be defensive in the IPC path).
    let Some(wl_output) = state.wl_outputs.get(output_id).cloned() else {
        return;
    };
    if state.layout.monitor_for_output(&wl_output).is_none() {
        return;
    }
    let surfaces: Vec<_> = state
        .layout
        .windows_for_output(&wl_output)
        .map(|w| w.toplevel().wl_surface().clone())
        .collect();
    for surface in surfaces {
        smithay::wayland::compositor::with_states(&surface, |states| {
            prism_protocols::color_management::SurfaceColorFeedbackSlot::notify_preferred_changed(
                states, identity,
            );
        });
    }
}

/// Best-effort socket cleanup on shutdown. Doesn't panic on failure —
/// the path was set with `prism_ipc::socket::SOCKET_PATH_ENV` so callers
/// already know where it lives if manual cleanup is needed.
pub fn remove_socket(path: &Path) {
    let _ = std::fs::remove_file(path);
}
