//! IPC server — Unix-socket request/reply using the `prism_ipc` vocabulary.
//!
//! Protocol: one JSON `Request` per line in, one JSON `Reply` per line
//! out. Each connection is one-shot today (client sends request, gets
//! reply, both sides close). The long-lived event-stream form (niri's
//! `EventStream` request) is future work.
//!
//! Dispatch is single-threaded on the main calloop — each connection
//! is handled synchronously inside the listener's read-event handler,
//! so requests cannot interleave with each other or with the render
//! loop. That's the right semantics for state-mutating requests (e.g.
//! a calibration tool flipping color overrides between sweep frames)
//! and is acceptable for read-only ones since the payloads are tiny.
//! A misbehaving client that connects but never sends data could
//! block the loop; the per-socket read timeout below bounds that.
//!
//! Bringup wires in `insert_ipc_source` from main.rs after the
//! wayland sources are up. The socket path lives in `$XDG_RUNTIME_DIR`
//! and is exported via `PRISM_SOCKET` so child processes (and any
//! `prism-tune` invocation) can find us.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, ErrorKind};
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use calloop::generic::Generic;
use calloop::{Interest, LoopHandle, Mode, PostAction};
use prism_ipc::{
    self, ColorState, LogicalOutput, Output, OutputAction, OutputConfigChanged, Reply, Request,
    Response, ResponseCurveState, Transform,
};
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

    handle
        .insert_source(
            Generic::new(listener, Interest::READ, Mode::Level),
            move |_event, listener, state| {
                loop {
                    match listener.accept() {
                        Ok((stream, _addr)) => {
                            if let Err(e) = handle_connection(stream, state) {
                                tracing::warn!("ipc connection handler error: {e:#}");
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

/// Read one JSON request from `stream`, dispatch, write one JSON reply,
/// then close. Connection lifetime is one request — clients that want
/// to make multiple requests reconnect, which is what
/// `prism_ipc::socket::Socket` does anyway.
fn handle_connection(stream: UnixStream, state: &mut PrismState) -> Result<()> {
    // Bound how long a slow / malicious client can hold the main loop.
    stream
        .set_read_timeout(Some(Duration::from_millis(500)))
        .context("set_read_timeout")?;
    stream
        .set_write_timeout(Some(Duration::from_millis(500)))
        .context("set_write_timeout")?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).context("read request line")?;

    let (reply, fd): (Reply, Option<OwnedFd>) = match serde_json::from_str::<Request>(&line) {
        Ok(req) => dispatch(state, req),
        Err(e) => (Err(format!("parse request: {e}")), None),
    };

    prism_ipc::socket::write_reply_with_fd(
        reader.get_mut(),
        &reply,
        fd.as_ref().map(|f| f.as_fd()),
    )
    .context("write reply")?;
    Ok(())
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
            // True focus tracking lives in input dispatch (not yet
            // multi-output-aware); for now report the first connected
            // output so the IPC at least has a defined answer.
            let map = collect_outputs(state);
            let first = map.into_values().next();
            (Ok(Response::FocusedOutput(first)), None)
        }
        Request::Output { output, action } => (handle_output_action(state, &output, action), None),
        Request::CaptureFrame { output } => handle_capture_frame(state, &output),
        Request::GamutMesh { output } => (handle_gamut_mesh(state, &output), None),
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
        let logical = state
            .wl_outputs
            .get(&ctx.connector_name)
            .and_then(|wl| state.layout.monitor_for_output(wl))
            .map(|monitor| {
                let view: Size<f64, smithay::utils::Logical> = monitor.view_size();
                LogicalOutput {
                    x: 0,
                    y: 0,
                    width: view.w as u32,
                    height: view.h as u32,
                    scale: 1.0,
                    transform: Transform::Normal,
                }
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
    // depends on. CTM and response-curve obviously do; SdrReferenceNits +
    // PanelPeakNits affect the encode pipeline elsewhere (target_peak +
    // decode clamp) but not LUT *contents*, so they skip the re-synthesis.
    let mut lut_dirty = false;
    match action {
        OutputAction::SdrReferenceNits { nits } => {
            let v = (nits.clamp(1.0, 10_000.0)) as f32;
            output_ctx.color_override.sdr_reference_nits = Some(v);
            tracing::info!(connector = %name, sdr_reference_nits = v, "ipc: set sdr-reference-nits override");
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
            let entries = prism_renderer::identity_lut(cube_edge);
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
            // this output — target_peak_nits + sdr_white_nits both
            // influence the OutputTransfer stage, so the diagnose must
            // mirror them or it'd test a different shader configuration
            // than what the panel sees.
            let mut p = match output_ctx.config.hdr {
                Some(hdr) => {
                    let mut p = prism_renderer::EncodePushSynth::pq_identity();
                    p.target_peak_nits = hdr.max_luminance as f32;
                    p
                }
                None => prism_renderer::EncodePushSynth::sdr_identity(),
            };
            p.sdr_white_nits = output_ctx.effective_sdr_reference_nits();
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

/// Best-effort socket cleanup on shutdown. Doesn't panic on failure —
/// the path was set with `prism_ipc::socket::SOCKET_PATH_ENV` so callers
/// already know where it lives if manual cleanup is needed.
pub fn remove_socket(path: &Path) {
    let _ = std::fs::remove_file(path);
}
