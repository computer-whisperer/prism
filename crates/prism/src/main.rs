use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use prism_frame::{DrmFourcc, DrmModifier};
use prism_renderer::vk;
use tracing_subscriber::EnvFilter;

mod ipc;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("prism=info,vulkan=info")),
        )
        .init();

    // Raise the open-files limit to the hard max before we (or any client)
    // start allocating fds. A compositor holds an fd per client buffer
    // (dmabuf planes, shm pools); the default ~1024 soft limit is exhausted by
    // buffer-churning clients (Firefox/WebRender under scroll) → EMFILE on
    // import → client crash. niri does the same at startup.
    prism_protocols::raise_nofile_to_max();

    // Capture panic messages to the (fsync'd) breadcrumb file so we can
    // still see them after a hard process exit that loses stderr buffer
    // contents — happens during TTY runs where stderr goes to a file the
    // script can't flush on our behalf.
    let default_panic = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = format!("PANIC: {info}");
        breadcrumb(&msg);
        default_panic(info);
    }));

    let args: Vec<String> = std::env::args().skip(1).collect();
    let output_name = args.get(1).map(String::as_str);
    let depth_arg = args.get(2).map(String::as_str);
    let result: Result<()> = match args.first().map(String::as_str) {
        None => run_headless_smoke_tests(),
        Some("scanout") => run_scanout_smoke_test(output_name),
        Some("gradient") => run_gradient_scanout(output_name, parse_depth(depth_arg)?),
        Some("wayland") => run_wayland_server(),
        Some("run") => run_integrated(output_name, parse_depth(depth_arg)?),
        Some(other) => Err(anyhow!(
            "unknown subcommand {other:?}; expected: (no args) | scanout [output] | gradient [output] [8|10] | wayland | run [output] [8|10]"
        )),
    };
    if let Err(e) = &result {
        // Mirror the error into the breadcrumb file so it survives a TTY
        // run where stderr buffering may eat the standard anyhow display.
        breadcrumb(&format!("EXIT ERROR: {e:#}"));
    }
    result
}

/// Resolve the prism config to load:
///   1. `$PRISM_CONFIG` if set (full path)
///   2. `$XDG_CONFIG_HOME/prism/config.kdl` (XDG default)
///   3. `~/.config/prism/config.kdl` (fallback)
///
/// On read / parse error: log loudly via `tracing::error!` AND a
/// `breadcrumb` (TTY runs lose stderr; the breadcrumb survives), then
/// fall back to `Config::default()` so the compositor still boots.
fn load_config() -> prism_config::Config {
    use std::path::PathBuf;

    let candidate: Option<PathBuf> = std::env::var_os("PRISM_CONFIG")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("XDG_CONFIG_HOME").map(|h| PathBuf::from(h).join("prism/config.kdl"))
        })
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config/prism/config.kdl"))
        });

    let Some(path) = candidate else {
        tracing::warn!(
            "no config path resolvable (PRISM_CONFIG / XDG_CONFIG_HOME / HOME all unset); using defaults"
        );
        return prism_config::Config::default();
    };

    if !path.exists() {
        tracing::info!(
            "no config file at {}; using defaults — set PRISM_CONFIG or create that file to customize",
            path.display()
        );
        return prism_config::Config::default();
    }

    let res = prism_config::Config::load(&path);
    if !res.includes.is_empty() {
        tracing::info!("config: loaded {} include(s)", res.includes.len());
    }
    match res.config {
        Ok(cfg) => {
            tracing::info!("loaded prism config from {}", path.display());
            cfg
        }
        Err(e) => {
            let msg = format!("config parse failed for {}: {e:?}", path.display());
            breadcrumb(&msg);
            tracing::error!("{msg}");
            tracing::error!("falling back to default config — fix the file and restart");
            // The default config has no user binds, which used to mean
            // "user is now trapped on this VT with no way to quit
            // prism short of sshing in to pkill". Hard-coded escape
            // hatches in prism-input::dispatch::on_keyboard cover
            // this — surface them in the log so the user knows the
            // escape sequences exist when they hit this path.
            tracing::error!(
                "emergency exits (hard-coded, always work): \
                 Ctrl+Alt+Backspace = quit prism, \
                 Ctrl+Alt+F1..F12 = switch VT"
            );
            prism_config::Config::default()
        }
    }
}

/// Find the `output "..."` config block for a kernel connector name (e.g.
/// `DisplayPort-4`). Accepts the short alias (`DP-4`) by expanding both
/// sides. Same logic as `prism_drm::scanout::match_config_for_connector`
/// Look up the per-connector KDL `output "..."` block. Matches by:
///   - Exact case-insensitive connector name (`output "DP-4"`)
///   - Legacy verbose spelling (`output "DisplayPort-4"`, normalized to `DP-4`)
///   - EDID `<Make> <Model> <Serial>` triple, when `edid` carries
///     all three fields — this is the form `prism-tune calibrate-lut3d`
///     writes for portable per-monitor calibration
///
/// The shared matcher lives in [`prism_config::output::block_matches_output`];
/// this is just the bringup-side wrapper that builds an `OutputName`
/// from the connector + EDID. Pre-EDID call sites pass `EdidInfo::default()`
/// (everything `None`); they get connector-only matching identical to the
/// pre-EDID behavior.
fn find_connector_config<'a>(
    connector_name: &str,
    edid: &prism_drm::EdidInfo,
    outputs_cfg: &'a [prism_config::output::Output],
) -> Option<&'a prism_config::output::Output> {
    let output_name = prism_config::output::OutputName {
        connector: connector_name.to_string(),
        make: edid.make.clone(),
        model: edid.model.clone(),
        serial: edid.serial.clone(),
    };
    outputs_cfg
        .iter()
        .find(|o| prism_config::output::block_matches_output(&o.name, &output_name))
}

/// Resolve a parsed `HdrConfig` (KDL) into a kernel-ready
/// `HdrSignaling`. Clamps user values to the u16 ranges the kernel
/// HDR_OUTPUT_METADATA infoframe uses; defaults max_cll to
/// max_luminance and max_fall to half max_luminance when omitted.
/// Mirrors niri's `ResolvedHdrConfig::from_config` (tty.rs:3323).
fn resolve_hdr_signaling(cfg: &prism_config::output::HdrConfig) -> prism_drm::HdrSignaling {
    use prism_config::output::HdrMode;
    let eotf = match cfg.mode {
        HdrMode::Pq => prism_drm::HdrEotf::Pq,
    };
    let clamp_u16 = |v: u32| v.min(u16::MAX as u32) as u16;
    let max_lum = clamp_u16(cfg.max_luminance);
    prism_drm::HdrSignaling {
        eotf,
        max_luminance: max_lum,
        // min_luminance is in nits; the kernel field is in 0.0001-nit
        // ticks (i.e. multiply by 10_000 before clamping to u16).
        min_luminance_ticks: (cfg.min_luminance * 10_000.0)
            .round()
            .clamp(0.0, u16::MAX as f64) as u16,
        max_cll: clamp_u16(cfg.max_cll.unwrap_or(cfg.max_luminance)),
        max_fall: clamp_u16(cfg.max_fall.unwrap_or(cfg.max_luminance / 2)),
    }
}

fn parse_depth(arg: Option<&str>) -> Result<prism_drm::ScanoutDepth> {
    match arg {
        None | Some("10") => Ok(prism_drm::ScanoutDepth::Bpc10),
        Some("8") => Ok(prism_drm::ScanoutDepth::Bpc8),
        Some(other) => Err(anyhow!("unknown depth {other:?}; expected 8 or 10")),
    }
}

fn vk_format_for_depth(depth: prism_drm::ScanoutDepth) -> prism_renderer::vk::Format {
    use prism_drm::ScanoutDepth::*;
    use prism_renderer::vk;
    match depth {
        Bpc8 => vk::Format::B8G8R8A8_UNORM,
        // DRM XR30 layout: 32-bit word with X(2) | R(10) | G(10) | B(10),
        // X in the high bits. Vulkan A2R10G10B10_UNORM_PACK32 uses the
        // exact same component ordering inside the 32-bit word.
        Bpc10 => vk::Format::A2R10G10B10_UNORM_PACK32,
        // DRM XBGR16161616F: little-endian half-floats laid out R, G, B,
        // X in memory order (the fourcc name reads the channels MSB-to-
        // LSB). Vulkan R16G16B16A16_SFLOAT reads the same byte layout.
        Fp16 => vk::Format::R16G16B16A16_SFLOAT,
    }
}

/// Bring up a Wayland server socket and dispatch protocol messages forever.
/// Clients can connect via `WAYLAND_DISPLAY=wayland-N`. No rendering yet —
/// surface lifecycle / configure / commit are logged, buffers are dropped.
fn run_wayland_server() -> Result<()> {
    use calloop::signals::{Signal, Signals};
    use calloop::EventLoop;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    tracing::info!("prism compositor — wayland server scaffolding");

    // Bring up Vulkan so the dmabuf handler can do real imports. Pick the
    // device that drives DP-4 (Vega 20) so client buffers and our scanout
    // path end up on the same GPU.
    let instance = prism_renderer::Instance::new()?;
    let device = prism_renderer::Device::new(
        instance.clone(),
        Some(prism_renderer::DrmDevId {
            major: 226,
            minor: 129,
        }),
    )?;
    tracing::info!("Vulkan device for dmabuf import: {}", device.physical.name);

    let display = prism_protocols::new_display()?;
    // Wayland-only mode: no DRM session, one GPU available for dmabuf import
    // validation. Key the gpu map by drm_primary (or drm_render fallback).
    let mut gpus = std::collections::HashMap::new();
    let key = device
        .physical
        .drm_primary
        .or(device.physical.drm_render)
        .ok_or_else(|| anyhow!("Vulkan device has no DRM node id; cannot index"))?;
    gpus.insert(key, device);
    // Single-GPU wayland mode → that's the primary.
    // No config file loaded in wayland-only mode — defaults give the
    // layout enough to bring up an empty workspace set.
    let mut state =
        prism_protocols::PrismState::new(&display, load_config(), None, gpus, Some(key));

    let mut event_loop: EventLoop<'static, prism_protocols::PrismState> =
        EventLoop::try_new().context("calloop EventLoop::try_new")?;

    // Stash the LoopHandle before client surfaces can appear, so the
    // drm_syncobj pre-commit hook can register eventfd sources.
    // `init_drm_syncobj` is a no-op in wayland-only mode (no card
    // attached), so we don't call it — the hook self-guards on
    // drm_syncobj_state.is_some().
    state.set_loop_handle(event_loop.handle());
    let socket = prism_protocols::insert_wayland_sources(&event_loop.handle(), display)?;
    // Bring up xwayland-satellite integration (binds X11 sockets, exports
    // $DISPLAY for children, spawns the satellite on-demand). Single-threaded
    // startup is required for the $DISPLAY env mutation — see its docs.
    prism_protocols::xwayland::satellite::setup(&mut state);
    tracing::info!(
        "WAYLAND_DISPLAY={socket}  — try: `WAYLAND_DISPLAY={socket} foot` (or weston-terminal)"
    );
    tracing::info!("Ctrl-C to exit");

    let running = Arc::new(AtomicBool::new(true));
    {
        let running = running.clone();
        let signals = Signals::new(&[Signal::SIGINT, Signal::SIGTERM])
            .context("Signals::new(SIGINT|SIGTERM)")?;
        event_loop
            .handle()
            .insert_source(signals, move |evt, _, _state| {
                tracing::info!(signal = ?evt.signal(), "signal received, shutting down");
                running.store(false, Ordering::SeqCst);
            })
            .map_err(|e| anyhow!("insert signals source: {e}"))?;
    }

    while running.load(Ordering::SeqCst) {
        event_loop
            .dispatch(Some(Duration::from_millis(100)), &mut state)
            .context("event_loop.dispatch")?;
        // Send deferred destructor events queued during dispatch
        // (see `ColorManagementState::pending_info_done`).
        state.color_management.drain_pending_info_done();
        // Flush replies queued during this turn.
        state
            .display_handle
            .flush_clients()
            .context("flush_clients")?;
    }

    tracing::info!("wayland server stopped");
    Ok(())
}

/// Default invocation: probe Vulkan + DRM + GBM↔Vulkan round-trip. All steps
/// run without DRM master and without a TTY.
fn run_headless_smoke_tests() -> Result<()> {
    tracing::info!("prism compositor — headless smoke tests");

    let instance = prism_renderer::Instance::new()?;

    // Default-picked device.
    {
        let device = prism_renderer::Device::new(instance.clone(), None)?;
        tracing::info!(
            "default device: {}, graphics queue family {}",
            device.physical.name,
            device.physical.graphics_queue_family,
        );
    }

    // Select by DRM node. Vega 20 is at render node 226:129 on this box
    // (the GPU driving DP-4 / LU28R55, our HDR tracer target).
    let want = prism_renderer::DrmDevId {
        major: 226,
        minor: 129,
    };
    let device = prism_renderer::Device::new(instance.clone(), Some(want))?;
    tracing::info!(
        "drm-preferred device: {}, drm_render={:?}",
        device.physical.name,
        device.physical.drm_render,
    );

    // Enumerate DRM connectors on both cards.
    for path in ["/dev/dri/card0", "/dev/dri/card1"] {
        match prism_drm::open_for_enumeration(path) {
            Ok(dev) => {
                let summary = prism_drm::summarize(&dev)?;
                tracing::info!(
                    "{path}: {} connectors, {} CRTCs, {} planes",
                    summary.connectors.len(),
                    summary.crtcs.len(),
                    summary.planes.len(),
                );
                for c in &summary.connectors {
                    let mode_str = c
                        .preferred_mode()
                        .map(|m| format!("{}x{}@{}Hz", m.size().0, m.size().1, m.vrefresh()))
                        .unwrap_or_else(|| "<no mode>".to_string());
                    tracing::info!(
                        "  {} {:?} {} modes, preferred {}",
                        c.name(),
                        c.state,
                        c.modes.len(),
                        mode_str,
                    );
                }
            }
            Err(e) => tracing::warn!("could not open {path}: {e:#}"),
        }
    }

    // GBM allocate + Vulkan import + clear-to-magenta + readback.
    tracer_clear(device.clone()).context("GBM→Vulkan tracer")?;

    // Same code path the wayland dmabuf protocol handler uses, exercised
    // without needing a real client to play along.
    tracer_dmabuf_protocol(device.clone()).context("dmabuf protocol-handler import path")?;

    // Full decode→encode pipeline rendering a linear gradient through the
    // intermediate, sRGB-encoded into a GBM BO we can map+inspect.
    tracer_render_gradient(device).context("render pipeline gradient test")?;

    Ok(())
}

/// Allocate a small XRGB8888 LINEAR buffer via GBM, import as a Vulkan image,
/// clear it to magenta via the Vulkan transfer queue, then map the BO from
/// the CPU and check that pixel (0,0) really is magenta.
///
/// This proves the GBM↔Vulkan handshake works end-to-end: format negotiation,
/// dmabuf fd handoff, memory-type matching, and that Vulkan commands actually
/// wrote to the same kernel BO the CPU can see.
fn tracer_clear(device: Arc<prism_renderer::Device>) -> Result<()> {
    use prism_renderer::{oneshot, ImportedImage, OneshotPool};

    let width: u32 = 256;
    let height: u32 = 16;

    let gbm = prism_drm::GbmDevice::open("/dev/dri/renderD129")
        .context("open /dev/dri/renderD129 for GBM")?;
    tracing::info!("GBM backend: {}", gbm.backend_name());

    let (bo, dmabuf) = gbm
        .allocate_scanout(width, height, DrmFourcc::Xrgb8888, &[DrmModifier::Linear])
        .context("GBM allocate XRGB8888 LINEAR")?;
    tracing::info!(
        "GBM BO: {}x{} {:?} modifier={:#x} planes={} stride[0]={}",
        dmabuf.width,
        dmabuf.height,
        dmabuf.format,
        u64::from(dmabuf.modifier),
        dmabuf.planes.len(),
        dmabuf.planes[0].stride,
    );

    let image = ImportedImage::import(
        device.clone(),
        &dmabuf,
        vk::Format::B8G8R8A8_UNORM,
        vk::ImageUsageFlags::TRANSFER_DST,
    )
    .context("import dmabuf as VkImage")?;
    tracing::info!(
        "imported VkImage {}x{} format={:?}",
        image.extent().width,
        image.extent().height,
        image.format(),
    );

    let pool = OneshotPool::new(device.clone())?;
    // Magenta in RGB. XRGB8888 in memory is B,G,R,X bytes per pixel.
    let color = vk::ClearColorValue {
        float32: [1.0, 0.0, 1.0, 1.0],
    };
    let vk_image = image.image();
    pool.record_and_submit(|raw, cb| {
        oneshot::record_clear_color(raw, cb, vk_image, color);
    })
    .context("clear-to-magenta submit")?;

    let probe = bo
        .map(0, 0, 1, 1, |mapped| {
            let b = mapped.buffer();
            (b[0], b[1], b[2], b[3])
        })
        .context("gbm map readback")?;
    tracing::info!(
        "BO pixel(0,0) after clear: B={:#04x} G={:#04x} R={:#04x} X={:#04x}",
        probe.0,
        probe.1,
        probe.2,
        probe.3
    );

    if probe.0 == 0xff && probe.1 == 0x00 && probe.2 == 0xff {
        tracing::info!("✓ GBM → Vulkan → clear → readback verified (magenta)");
    } else {
        return Err(anyhow!(
            "readback mismatch: expected B=ff G=00 R=ff, got B={:#04x} G={:#04x} R={:#04x}",
            probe.0,
            probe.1,
            probe.2
        ));
    }
    Ok(())
}

/// Exercise the same import path that `prism-protocols::DmabufHandler::dmabuf_imported`
/// runs, but synthesize the smithay::Dmabuf locally so we don't depend on a
/// real client. Validates that:
///   - smithay::Dmabuf → prism_frame::Dmabuf fd-dup conversion works
///   - ImportedImage::import succeeds with vk::ImageUsageFlags::SAMPLED
///     (vs the TRANSFER_DST usage the tracer_clear path uses)
fn tracer_dmabuf_protocol(device: Arc<prism_renderer::Device>) -> Result<()> {
    use smithay::backend::allocator::dmabuf::{Dmabuf as SmithayDmabuf, DmabufFlags};

    let width: u32 = 256;
    let height: u32 = 16;

    let gbm = prism_drm::GbmDevice::open("/dev/dri/renderD129")?;
    let (bo, _our_dmabuf) =
        gbm.allocate_scanout(width, height, DrmFourcc::Xrgb8888, &[DrmModifier::Linear])?;

    // Build a smithay::Dmabuf from the GBM BO, mirroring what
    // smithay::backend::allocator::gbm::GbmAllocator does internally — that
    // way the input to ImportedImage::import matches the shape the wayland
    // handler will hand us at runtime.
    let plane_fd = bo
        .fd_for_plane(0)
        .map_err(|_| anyhow!("gbm_bo_get_fd_for_plane(0) returned -1"))?;
    let mut builder = SmithayDmabuf::builder(
        (width as i32, height as i32),
        DrmFourcc::Xrgb8888,
        DrmModifier::Linear,
        DmabufFlags::empty(),
    );
    if !builder.add_plane(plane_fd, 0, bo.offset(0), bo.stride_for_plane(0)) {
        return Err(anyhow!("DmabufBuilder::add_plane returned false"));
    }
    let smithay_dmabuf: SmithayDmabuf = builder
        .build()
        .ok_or_else(|| anyhow!("DmabufBuilder::build returned None"))?;

    // Convert + import — same call shape as the wayland handler.
    let prism_dmabuf =
        prism_frame::Dmabuf::from_smithay(&smithay_dmabuf).context("Dmabuf::from_smithay")?;
    let _image = prism_renderer::ImportedImage::import(
        device,
        &prism_dmabuf,
        vk::Format::B8G8R8A8_UNORM,
        vk::ImageUsageFlags::SAMPLED,
    )
    .context("ImportedImage::import (SAMPLED, mirroring wayland handler)")?;
    tracing::info!("✓ dmabuf-handler import path verified (SAMPLED VkImage)");
    Ok(())
}

/// End-to-end pipeline check: build a small linear gradient texture, run it
/// through decode→intermediate→encode (sRGB OETF), readback the BGRA bytes,
/// validate at anchor points. Catches:
///   - shader compile / SPIR-V loading regressions
///   - descriptor / pipeline layout mismatches
///   - dynamic-rendering attachment setup mistakes
///   - sRGB OETF math (compare to known curve values)
fn tracer_render_gradient(device: Arc<prism_renderer::Device>) -> Result<()> {
    use prism_renderer::{vk, DecodePush, ElementDraw, EncodePush, ImportedImage, Renderer};

    let width: u32 = 256;
    let height: u32 = 1;

    // Scanout target: a GBM XRGB8888 LINEAR BO we can map for readback.
    let gbm = prism_drm::GbmDevice::open("/dev/dri/renderD129")?;
    let (bo, dmabuf) =
        gbm.allocate_scanout(width, height, DrmFourcc::Xrgb8888, &[DrmModifier::Linear])?;
    let scanout = ImportedImage::import(
        device.clone(),
        &dmabuf,
        vk::Format::B8G8R8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT,
    )?;

    // Source texture: 256×1 linear horizontal gradient, RGBA16_SFLOAT. Each
    // pixel = (x/255, x/255, x/255, 1.0). When fed through identity decode
    // (transfer=Linear) the intermediate holds linear values in [0,1] *
    // sdr_white_nits. The encode pass (Srgb, sdr_white_nits=80) normalizes
    // back to [0,1] and sRGB-encodes.
    let texture = build_gradient_texture(device.clone(), width)?;

    // Headless self-test uses the renderer's default fp32 intermediate + the
    // default-SDR encode config (identity calibration + sRGB OETF).
    let encode_config = prism_renderer::EncodeConfig::default_srgb();
    let mut renderer = Renderer::new(
        device.clone(),
        vk::Format::B8G8R8A8_UNORM,
        prism_renderer::DEFAULT_INTERMEDIATE_FORMAT,
        &encode_config,
    )?;

    // Single element covering the whole output.
    let element = ElementDraw {
        texture_view: texture.view,
        chroma_view: None,
        push: DecodePush::identity_srgb([-1.0, -1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]),
    };
    let encode_push = EncodePush::sdr_identity();

    // Headless readback path — the SYNC_FD returned by render_frame is
    // dropped, and we device_wait_idle for completeness. (One-shot test
    // doesn't use the page-flip path the fd is meant for.)
    // Damage `&[]`: a freshly built Renderer forces a full first-frame paint
    // regardless, so the empty damage list is moot here.
    let _present_sync = renderer.render_frame(&scanout, &[element], &[], &encode_push, &[])?;
    unsafe {
        let _ = device.raw.device_wait_idle();
    }

    // Read back via GBM map and check anchor points.
    bo.map(0, 0, width, 1, |mapped| {
        let stride = mapped.stride() as usize;
        let row = &mapped.buffer()[..stride];
        // Pixel at x=0 should be ~0 (sRGB-encoded 0.0 → 0.0).
        let p0 = bgra(row, 0);
        // Pixel at x=255 should be ~255 (sRGB-encoded 1.0 → 1.0).
        let p255 = bgra(row, 255);
        // Mid pixel: linear 127/255 ≈ 0.498. sRGB OETF ≈ 0.738. So encoded byte ≈ 188.
        let pmid = bgra(row, 127);
        tracing::info!(
            "gradient readback: x=0 BGRA={:?}  x=127 BGRA={:?}  x=255 BGRA={:?}",
            p0,
            pmid,
            p255
        );
        // Quick sanity bounds (allow small AMD sRGB-OETF rounding).
        let ok = p0.0 <= 4
            && p255.0 >= 250
            && (180..=196).contains(&pmid.0)
            && (180..=196).contains(&pmid.1)
            && (180..=196).contains(&pmid.2);
        if ok {
            tracing::info!("✓ render pipeline gradient verified (sRGB OETF anchor-points match)");
            Ok(())
        } else {
            Err(anyhow!("gradient anchor-point mismatch"))
        }
    })
    .context("gbm map for gradient readback")??;

    Ok(())
}

fn bgra(row: &[u8], x: usize) -> (u8, u8, u8, u8) {
    let off = x * 4;
    (row[off], row[off + 1], row[off + 2], row[off + 3])
}

/// Create a 256×1 RGBA16_SFLOAT texture pre-filled with a horizontal linear
/// gradient. Owns its own VkImage + memory; caller drops to free.
struct GradientTexture {
    device: Arc<prism_renderer::Device>,
    image: prism_renderer::vk::Image,
    memory: prism_renderer::vk::DeviceMemory,
    view: prism_renderer::vk::ImageView,
}

impl Drop for GradientTexture {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.raw.device_wait_idle();
            self.device.raw.destroy_image_view(self.view, None);
            self.device.raw.destroy_image(self.image, None);
            self.device.raw.free_memory(self.memory, None);
        }
    }
}

/// Minimal f32 → IEEE-754 half-precision bit pattern. Sufficient for the
/// non-negative finite values the test uses; doesn't handle subnormals or
/// rounding modes carefully.
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp_f32 = ((bits >> 23) & 0xff) as i32;
    let mantissa_f32 = bits & 0x007f_ffff;
    if exp_f32 == 0 {
        return sign << 15; // ±0
    }
    let exp_f16 = exp_f32 - 127 + 15;
    if exp_f16 <= 0 {
        return sign << 15; // underflow → zero (no subnormals)
    }
    if exp_f16 >= 31 {
        return (sign << 15) | (0x1f << 10); // ±inf
    }
    let mantissa_f16 = (mantissa_f32 >> 13) as u16;
    (sign << 15) | ((exp_f16 as u16) << 10) | mantissa_f16
}

fn build_gradient_texture(
    device: Arc<prism_renderer::Device>,
    width: u32,
) -> Result<GradientTexture> {
    use prism_renderer::vk;
    let height = 1;

    // Generate the data: 256 fp16 RGBA values, linear ramp 0..1.
    let pixels: Vec<u16> = (0..width)
        .flat_map(|x| {
            let v = x as f32 / (width - 1) as f32;
            let h = f32_to_f16_bits(v);
            let one = f32_to_f16_bits(1.0);
            [h, h, h, one]
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&pixels);

    // Staging buffer: HOST_VISIBLE, large enough for the upload.
    let buffer_info = vk::BufferCreateInfo::default()
        .size(bytes.len() as u64)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let staging = unsafe { device.raw.create_buffer(&buffer_info, None) }?;
    let req = unsafe { device.raw.get_buffer_memory_requirements(staging) };
    let mem_type = pick_memory(
        &device,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_type);
    let staging_mem = unsafe { device.raw.allocate_memory(&alloc, None) }?;
    unsafe { device.raw.bind_buffer_memory(staging, staging_mem, 0) }?;
    unsafe {
        let dst = device
            .raw
            .map_memory(staging_mem, 0, req.size, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst as *mut u8, bytes.len());
        device.raw.unmap_memory(staging_mem);
    }

    // Texture image: OPTIMAL, SAMPLED + TRANSFER_DST.
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R16G16B16A16_SFLOAT)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.raw.create_image(&image_info, None) }?;
    let req = unsafe { device.raw.get_image_memory_requirements(image) };
    let mem_type = pick_memory(
        &device,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.raw.allocate_memory(&alloc, None) }?;
    unsafe { device.raw.bind_image_memory(image, memory, 0) }?;

    // Upload: one-shot command buffer: transition → copy → transition.
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(device.physical.graphics_queue_family)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let pool = unsafe { device.raw.create_command_pool(&pool_info, None) }?;
    let cb_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cb = unsafe { device.raw.allocate_command_buffers(&cb_info) }?[0];
    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.raw.begin_command_buffer(cb, &begin) }?;

    let to_xfer = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
        .dst_stage_mask(vk::PipelineStageFlags2::COPY)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })];
    unsafe {
        device.raw.cmd_pipeline_barrier2(
            cb,
            &vk::DependencyInfo::default().image_memory_barriers(&to_xfer),
        );
    }
    let region = [vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })];
    unsafe {
        device.raw.cmd_copy_buffer_to_image(
            cb,
            staging,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &region,
        );
    }
    let to_sampled = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COPY)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })];
    unsafe {
        device.raw.cmd_pipeline_barrier2(
            cb,
            &vk::DependencyInfo::default().image_memory_barriers(&to_sampled),
        );
        device.raw.end_command_buffer(cb)?;
    }
    let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
    let submit = [vk::SubmitInfo2::default().command_buffer_infos(&cbs)];
    unsafe {
        device
            .raw
            .queue_submit2(device.graphics_queue, &submit, vk::Fence::null())?;
        device.raw.queue_wait_idle(device.graphics_queue)?;
        device.raw.destroy_command_pool(pool, None);
        device.raw.destroy_buffer(staging, None);
        device.raw.free_memory(staging_mem, None);
    }

    let view = prism_renderer::create_view(&device, image, vk::Format::R16G16B16A16_SFLOAT)?;
    Ok(GradientTexture {
        device,
        image,
        memory,
        view,
    })
}

fn pick_memory(
    device: &prism_renderer::Device,
    type_bits: u32,
    required: prism_renderer::vk::MemoryPropertyFlags,
) -> Result<u32> {
    let props = unsafe {
        device
            .instance_raw()
            .get_physical_device_memory_properties(device.physical.raw)
    };
    for i in 0..props.memory_type_count {
        let mt = props.memory_types[i as usize];
        if (type_bits & (1 << i)) != 0 && mt.property_flags.contains(required) {
            return Ok(i);
        }
    }
    Err(anyhow!("no memory type matches {:?}", required))
}

/// Append a one-line breadcrumb to the crumbs file and `fsync`. Used in
/// `prism run` to leave a trail across a TTY-test session that survives the
/// system locking up — tracing-via-stderr can't reach the user's eyes once
/// we own DRM master (the text console can't refresh), and any in-flight
/// stdio is lost when the kernel wedges.
///
/// Path: `$PRISM_CRUMBS` if set, otherwise `./prism.crumbs` (relative to
/// the cwd at process start). NOT `/tmp` — that's tmpfs on most distros
/// and gets wiped at the reboot we're specifically trying to debug.
fn breadcrumb(msg: &str) {
    use std::io::Write;
    use std::sync::OnceLock;
    static CRUMBS_PATH: OnceLock<std::path::PathBuf> = OnceLock::new();
    let path = CRUMBS_PATH.get_or_init(|| {
        if let Ok(p) = std::env::var("PRISM_CRUMBS") {
            std::path::PathBuf::from(p)
        } else {
            std::path::PathBuf::from("prism.crumbs")
        }
    });
    let line = format!(
        "{:.3}: {msg}\n",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0),
    );
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = f.write_all(line.as_bytes());
        let _ = f.sync_all();
    }
}

/// End-to-end integrated mode: wayland server + DRM scanout + per-frame
/// render in one event loop. Single process. Wayland clients can connect —
/// their surfaces are tracked but not yet rendered.
///
/// Requires DRM master — run from a free VT (Ctrl+Alt+F3). Ctrl-C to exit.
///
/// Rendering is vblank-driven: each DRM VBlank event triggers the next
/// present. Bootstrap is one explicit present before entering the loop;
/// after that the kernel's vblank cadence (the display's refresh rate) sets
/// the pace. This eliminates the half-rate pinning of the previous
/// timer-driven model (timer + frame_pending gate skipped every other
/// fire), and naturally drops frames if rendering can't keep up.
///
/// Diagnostic env vars (set before `prism run`):
///   PRISM_MAX_FRAMES=N      exit after N frames presented (default: unlimited).
///                            Use small values (e.g. 5) when testing on a TTY
///                            so the process self-terminates if rendering hangs.
///   PRISM_WATCHDOG_SECS=N   spawn a sleeper thread that SIGKILLs our PID
///                            after N seconds (default 10, 0 to disable).
///
/// Breadcrumbs are appended to ./prism.crumbs (override with $PRISM_CRUMBS)
/// with fsync per line, so they survive a system lockup.
fn run_integrated(output_name: Option<&str>, depth: prism_drm::ScanoutDepth) -> Result<()> {
    use calloop::signals::{Signal, Signals};
    use calloop::EventLoop;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    let max_frames: Option<u32> = std::env::var("PRISM_MAX_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok());
    // Wall-clock shutdown trigger. Cleaner than `max_frames` for
    // multi-output (frame counter rate scales with output count). When set,
    // a thread flips `running` to false after N seconds, the main dispatch
    // loop notices and exits cleanly.
    let max_runtime_secs: Option<u64> = std::env::var("PRISM_MAX_RUNTIME_SECS")
        .ok()
        .and_then(|s| s.parse().ok());
    // Hard self-kill watchdog. Spawns a sleeper thread that SIGKILLs our
    // own PID after N seconds — uncatchable, runs in a separate thread so
    // it fires even if our main thread is stuck on queue_wait_idle waiting
    // for a hung GPU. Default 10s so a misbehaving TTY test still recovers
    // before the kernel locks up. Set to 0 to disable.
    let watchdog_secs: u64 = std::env::var("PRISM_WATCHDOG_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    breadcrumb(&format!(
        "startup: vblank-driven, max_frames={max_frames:?}, max_runtime={max_runtime_secs:?}s, watchdog={watchdog_secs}s"
    ));
    if watchdog_secs > 0 {
        let secs = watchdog_secs;
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(secs));
            breadcrumb(&format!("WATCHDOG: {secs}s elapsed, SIGKILL self"));
            unsafe {
                libc::kill(libc::getpid(), libc::SIGKILL);
            }
        });
    }

    tracing::info!("prism — integrated mode (wayland + scanout)");

    // ── Vulkan instance (devices opened later, one per card) ──────────────
    let instance = prism_renderer::Instance::new()?;
    breadcrumb("vulkan instance up");

    // ── DRM session ────────────────────────────────────────────────────────
    let (mut session, session_notifier) = prism_drm::SeatSession::new()?;
    if !session.is_active() {
        return Err(anyhow!(
            "libseat session not active — must be run from a foreground VT"
        ));
    }

    // ── Load user config up-front ─────────────────────────────────────────
    // Both bringup (mode / off / depth selection) and PrismState::new need it;
    // load once and share. `load_config()` already falls back to defaults on
    // failure with a loud log line, so this is safe before anything else.
    let config = load_config();

    // ── Open every card we want to drive ───────────────────────────────────
    // CARDS env var overrides the hard-coded list (comma-separated paths,
    // e.g. CARDS=/dev/dri/card1). Default: both cards on this hardware.
    let card_paths: Vec<String> = match std::env::var("CARDS").ok() {
        Some(s) => s.split(',').map(|p| p.trim().to_string()).collect(),
        None => vec!["/dev/dri/card0".into(), "/dev/dri/card1".into()],
    };
    let mut cards: Vec<prism_drm::DrmCardContext> = Vec::new();
    let mut drm_notifiers: Vec<smithay::backend::drm::DrmDeviceNotifier> = Vec::new();
    for path in &card_paths {
        match prism_drm::DrmCardContext::open(&mut session, path) {
            Ok((card, notifier)) => {
                breadcrumb(&format!(
                    "card opened: {} (drm {}:{})",
                    card.path, card.drm_dev_id.major, card.drm_dev_id.minor
                ));
                cards.push(card);
                drm_notifiers.push(notifier);
            }
            Err(e) => {
                tracing::warn!("skipping card {path}: {e:#}");
                breadcrumb(&format!("card open FAILED: {path}: {e:#}"));
            }
        }
    }
    if cards.is_empty() {
        return Err(anyhow!("no DRM cards could be opened"));
    }

    // ── Build a Vulkan device per opened card ──────────────────────────────
    // Match Vulkan physical devices to DRM cards via DrmDevId. If a card
    // has no matching Vulkan device (driver mismatch), skip that card's
    // outputs but keep the rest of the bringup.
    let mut gpus: std::collections::HashMap<prism_renderer::DrmDevId, Arc<prism_renderer::Device>> =
        std::collections::HashMap::new();
    for card in &cards {
        match prism_renderer::Device::new(instance.clone(), Some(card.drm_dev_id)) {
            Ok(device) => {
                tracing::info!(
                    "GPU for card {} ({}:{}): {}",
                    card.path,
                    card.drm_dev_id.major,
                    card.drm_dev_id.minor,
                    device.physical.name
                );
                gpus.insert(card.drm_dev_id, device);
            }
            Err(e) => {
                tracing::warn!(
                    "no Vulkan device matches card {} ({}:{}): {e:#}",
                    card.path,
                    card.drm_dev_id.major,
                    card.drm_dev_id.minor
                );
            }
        }
    }
    if gpus.is_empty() {
        return Err(anyhow!("no Vulkan devices matched any opened card"));
    }
    breadcrumb(&format!("vulkan devices: {} GPU(s)", gpus.len()));

    // Default OutputConfig built from CLI depth. Per-connector overrides
    // (config.color.max_bpc → depth; config.variable_refresh_rate → vrr)
    // are applied just below, inside the per-output bringup loop, where
    // the connector name is known.
    let default_output_config = prism_drm::OutputConfig {
        depth,
        vk_format: vk_format_for_depth(depth),
        intermediate_format: prism_renderer::DEFAULT_INTERMEDIATE_FORMAT,
        encode_config: prism_renderer::EncodeConfig::default_srgb(),
        vrr: false,
        hdr: None,
        // IEC sRGB default: 1.0 client-side = 80 cd/m². Per-output
        // overrides applied below when a connector's KDL color block
        // sets `sdr-reference-nits`.
        sdr_reference_nits: 80.0,
        // Defaults to IEC sRGB ceiling broadcast to all three channels;
        // recalculated per-output below once HDR config / sdr-reference-nits
        // / explicit panel-peak-nits are known.
        panel_peak_nits_rgb: [80.0, 80.0, 80.0],
        response_curve: None,
        ctm: None,
    };

    // ── Pick connectors + bring up OutputContexts on every card ────────────
    // If OUTPUT specifies a connector name, search every card for it and
    // bring up only that one. Otherwise pick_all_connected on each card.
    let mut outputs: Vec<prism_drm::OutputContext> = Vec::new();
    for card in &mut cards {
        breadcrumb(&format!("bringup loop: entering card {}", card.path));
        let Some(device) = gpus.get(&card.drm_dev_id).cloned() else {
            tracing::warn!("card {} has no GPU; skipping all its outputs", card.path);
            breadcrumb(&format!(
                "bringup loop: {} has no matching GPU, skipping",
                card.path
            ));
            continue;
        };
        breadcrumb(&format!("bringup loop: {} picking connectors", card.path));
        let picks: Vec<prism_drm::OutputPick> = match output_name {
            Some(name) => {
                match prism_drm::pick_by_name_with_config(&card.drm, name, &config.outputs.0) {
                    Ok(p) => vec![p],
                    Err(_) => Vec::new(), // OUTPUT might be on a different card
                }
            }
            None => prism_drm::pick_all_connected_with_config(&card.drm, &config.outputs.0)
                .unwrap_or_default(),
        };
        breadcrumb(&format!(
            "bringup loop: {} got {} pick(s)",
            card.path,
            picks.len()
        ));
        for pick in picks {
            let name = pick.connector_name.clone();
            // Per-output config: start from the CLI default, then apply
            // any per-connector overrides from the KDL `output "…"` block.
            // EDID is what makes EDID-keyed `output "Make Model Serial"`
            // blocks resolvable — read it here so find_connector_config
            // can match them. OutputContext::new re-reads inside, but
            // EDID is a single DRM property read so the double-read is
            // negligible compared to bringup cost.
            let edid = prism_drm::EdidInfo::read(&card.drm, pick.connector);
            let mut output_config = default_output_config.clone();
            if let Some(cfg) = find_connector_config(&name, &edid, &config.outputs.0) {
                if let Some(color) = cfg.color.as_ref() {
                    // HDR-on overrides max_bpc to 10 + flips depth to
                    // fp16 + flips encode chain to PQ. Done before
                    // the bare-max_bpc branch so an explicit max_bpc
                    // in config is still honored as the ceiling.
                    if let Some(hdr_cfg) = color.hdr.as_ref() {
                        output_config.hdr = Some(resolve_hdr_signaling(hdr_cfg));
                        output_config.depth = prism_drm::ScanoutDepth::Fp16;
                        output_config.vk_format = vk_format_for_depth(output_config.depth);
                        output_config.encode_config = prism_renderer::EncodeConfig::default_pq();
                        tracing::info!(
                            connector = %name,
                            "HDR config present: fp16 scanout + PQ encode + KMS signaling"
                        );
                    } else if let Some(bpc) = color.max_bpc {
                        if bpc >= 10 {
                            output_config.depth = prism_drm::ScanoutDepth::Bpc10;
                        } else {
                            output_config.depth = prism_drm::ScanoutDepth::Bpc8;
                        }
                        output_config.vk_format = vk_format_for_depth(output_config.depth);
                    }
                    if let Some(nits) = color.sdr_reference_nits {
                        // Clamp to a sane physical range (1..=10000). Negative
                        // or zero would zero-out all color-unaware content;
                        // values above 10000 exceed PQ's encoding range.
                        let clamped = nits.clamp(1.0, 10_000.0) as f32;
                        output_config.sdr_reference_nits = clamped;
                        tracing::info!(
                            connector = %name,
                            sdr_reference_nits = clamped,
                            "per-output SDR reference luminance set"
                        );
                    }
                }
                // Resolve the decoder's display-referred clamp ceiling
                // per channel. Resolution order:
                //   1. Explicit KDL `color.panel-peak-nits r=… g=… b=…`
                //      (preferred — calibrated per-subpixel)
                //   2. Broadcast of HDR `max_luminance` (HDR mode)
                //   3. Broadcast of SDR reference (SDR mode)
                // The broadcast fallbacks are conservative all-channels-
                // equal guesses; calibrate replaces them.
                output_config.panel_peak_nits_rgb =
                    match cfg.color.as_ref().and_then(|c| c.panel_peak_nits) {
                        Some(p) => [p.r as f32, p.g as f32, p.b as f32],
                        None => {
                            let scalar = match output_config.hdr {
                                Some(hdr) => hdr.max_luminance as f32,
                                None => output_config.sdr_reference_nits,
                            };
                            [scalar, scalar, scalar]
                        }
                    };
                tracing::info!(
                    connector = %name,
                    panel_peak_nits_rgb = ?output_config.panel_peak_nits_rgb,
                    "per-output panel peak resolved"
                );
                if let Some(ctm_cfg) = cfg.color.as_ref().and_then(|c| c.ctm.as_ref()) {
                    if ctm_cfg.values.len() == 9 {
                        let v = &ctm_cfg.values;
                        output_config.ctm = Some([
                            [v[0] as f32, v[1] as f32, v[2] as f32],
                            [v[3] as f32, v[4] as f32, v[5] as f32],
                            [v[6] as f32, v[7] as f32, v[8] as f32],
                        ]);
                        tracing::info!(
                            connector = %name,
                            ctm = ?output_config.ctm,
                            "per-output CTM set from KDL"
                        );
                    } else {
                        tracing::warn!(
                            connector = %name,
                            got = ctm_cfg.values.len(),
                            "color.ctm needs exactly 9 row-major values; ignoring"
                        );
                    }
                }
                if let Some(curve) = cfg.color.as_ref().and_then(|c| c.response_curve.as_ref()) {
                    // Clamp to physically-meaningful ranges. Gain <= 0
                    // would zero-divide; gamma <= 0 would produce
                    // pow(x, +inf); silly-large values blow up
                    // commanded-nits to past PQ peak. The fragment is
                    // already in the encode chain for any HDR output
                    // (default_pq always includes it); we only need to
                    // stash the configured values so per-frame push
                    // construction picks them up.
                    let g_r = (curve.gain_r as f32).clamp(0.01, 10.0);
                    let g_g = (curve.gain_g as f32).clamp(0.01, 10.0);
                    let g_b = (curve.gain_b as f32).clamp(0.01, 10.0);
                    let y_r = (curve.gamma_r as f32).clamp(0.1, 10.0);
                    let y_g = (curve.gamma_g as f32).clamp(0.1, 10.0);
                    let y_b = (curve.gamma_b as f32).clamp(0.1, 10.0);
                    output_config.response_curve = Some(([g_r, g_g, g_b], [y_r, y_g, y_b]));
                    tracing::info!(
                        connector = %name,
                        gain = ?[g_r, g_g, g_b],
                        gamma = ?[y_r, y_g, y_b],
                        "per-output response correction set from KDL"
                    );
                }
                // Vrr always-on → wire through; on-demand currently
                // treated as off (needs content_type signaling).
                output_config.vrr = cfg.is_vrr_always_on();
                if cfg.is_vrr_on_demand() {
                    tracing::warn!(
                        connector = %name,
                        "VRR on_demand=true ignored — falling back to off until \
                         content_type signaling lands"
                    );
                }
            }
            breadcrumb(&format!("bringup loop: building OutputContext for {name}"));
            match prism_drm::OutputContext::new(card, device.clone(), pick, &output_config) {
                Ok(mut output) => {
                    // KDL `color.lut3d "file"` — load the binary LUT now
                    // so resynthesize_color_lut sees it as the fallback
                    // when no IPC override is active. Re-look up the
                    // per-connector config block; the earlier `cfg` is
                    // scoped to the `if let` above. Use the OutputContext's
                    // already-populated EDID so EDID-keyed blocks resolve
                    // here too (the bringup-side `edid` above is out of
                    // scope after OutputContext takes ownership).
                    if let Some(lut3d_cfg) =
                        find_connector_config(&name, &output.edid, &config.outputs.0)
                            .and_then(|c| c.color.as_ref())
                            .and_then(|c| c.lut3d.as_ref())
                    {
                        match prism_renderer::load_lut3d_file(std::path::Path::new(&lut3d_cfg.path))
                        {
                            Ok(loaded) => {
                                let renderer_edge = output.renderer.lut3d_cube_edge();
                                if loaded.cube_edge != renderer_edge {
                                    tracing::warn!(
                                        connector = %output.connector_name,
                                        path = %lut3d_cfg.path,
                                        "LUT file cube_edge={} doesn't match renderer cube_edge={}; \
                                         falling back to synthesis",
                                        loaded.cube_edge, renderer_edge,
                                    );
                                } else {
                                    let bp = loaded.black_point_xyz;
                                    tracing::info!(
                                        connector = %output.connector_name,
                                        path = %lut3d_cfg.path,
                                        cube_edge = loaded.cube_edge,
                                        black_point_xyz = ?bp,
                                        "loaded color LUT from file"
                                    );
                                    output.kdl_lut3d_entries = Some(loaded.entries);
                                    // All-zero ⇒ unmeasured (pre-v2 measurement
                                    // or calibrate skipped the floor). Leave
                                    // unset so effective_black_point_xyz returns
                                    // None and downstream consumers can fall
                                    // back to "no min-luminance signal."
                                    if bp[0] != 0.0 || bp[1] != 0.0 || bp[2] != 0.0 {
                                        output.kdl_black_point_xyz = Some(bp);
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    connector = %output.connector_name,
                                    path = %lut3d_cfg.path,
                                    "LUT file load failed ({e:#}); falling back to synthesis"
                                );
                            }
                        }
                    }
                    // Bake whichever color representation is current
                    // (loaded LUT or synthesis from CTM + curve) into the
                    // renderer's LUT texture. Failure here means the
                    // renderer can't accept LUT data (allocator OOM, lost
                    // device, etc.) — log and continue with whatever the
                    // identity LUT renders rather than fail the whole
                    // output, since the bringup just succeeded.
                    if let Err(e) = output.resynthesize_color_lut() {
                        tracing::warn!(
                            connector = %output.connector_name,
                            "initial color LUT synthesis failed: {e:#} \
                             (output stays on identity LUT)"
                        );
                    }
                    breadcrumb(&format!(
                        "output bringup ok: {} {}x{} on {}",
                        name, output.extent.width, output.extent.height, card.path
                    ));
                    outputs.push(output);
                }
                Err(e) => {
                    breadcrumb(&format!("output bringup FAILED for {name}: {e:#}"));
                    tracing::warn!("output bringup failed for {name}: {e:#}");
                }
            }
        }
        breadcrumb(&format!("bringup loop: finished card {}", card.path));
    }
    breadcrumb(&format!(
        "bringup loop: all cards done, {} outputs total",
        outputs.len()
    ));
    if outputs.is_empty() {
        return Err(anyhow!(
            "no outputs successfully brought up across any card"
        ));
    }

    // ── Wayland display + PrismState ───────────────────────────────────────
    let display = prism_protocols::new_display()?;
    // Pick the primary GPU for dmabuf-feedback's main_device:
    // - PRISM_PRIMARY_GPU env var (format "major:minor", e.g. "226:1") overrides
    // - Otherwise the highest-numbered DrmDevId, which on this hardware
    //   (and most modern setups where the discrete GPU is added later)
    //   resolves to Navi 21 (226:1). Documented in
    //   memory/project_hardware_allocation as the bandwidth-critical primary.
    let primary_gpu = std::env::var("PRISM_PRIMARY_GPU")
        .ok()
        .and_then(|s| {
            let mut parts = s.splitn(2, ':');
            let major = parts.next()?.parse::<i64>().ok()?;
            let minor = parts.next()?.parse::<i64>().ok()?;
            Some(prism_renderer::DrmDevId { major, minor })
        })
        .filter(|id| gpus.contains_key(id))
        .or_else(|| gpus.keys().max_by_key(|id| (id.major, id.minor)).copied());
    if let Some(id) = primary_gpu {
        tracing::info!("primary GPU for dmabuf-feedback: {}:{}", id.major, id.minor);
    }
    let mut state =
        prism_protocols::PrismState::new(&display, config, Some(session), gpus, primary_gpu);
    for card in cards {
        state.attach_card(card);
    }
    for output in outputs {
        // Advertise BEFORE moving the OutputContext into state — we only
        // need a borrow for the wl_output mode/extent/connector_name.
        state.advertise_output(&output);
        state.attach_output(output);
    }
    // Now that every output is advertised, assign logical positions
    // (horizontal stack by sorted connector name). Sends wl_output.done
    // events to any clients already bound.
    state.layout_outputs();
    breadcrumb(&format!(
        "wayland state up; {} card(s) + {} output(s) attached ({} wl_output globals)",
        state.cards.len(),
        state.outputs.len(),
        state.wl_outputs.len()
    ));

    // Event loop + sources.
    let mut event_loop: EventLoop<'static, prism_protocols::PrismState> =
        EventLoop::try_new().context("EventLoop::try_new")?;
    // Stash the LoopHandle on state before any client can connect:
    // drm_syncobj's pre-commit hook (registered from new_surface)
    // reads it to insert eventfd sources for acquire blockers, and
    // render_output_now reads it to schedule release-point signals
    // on the per-submit sync_fd. Must happen before
    // `insert_wayland_sources` (which makes the socket reachable).
    state.set_loop_handle(event_loop.handle());
    // Bring up the wp_linux_drm_syncobj global now that cards are
    // attached. Skipped silently if kernel lacks `syncobj_eventfd`
    // or the primary GPU's card isn't registered.
    state.init_drm_syncobj();
    let socket = prism_protocols::insert_wayland_sources(&event_loop.handle(), display)?;
    // Bring up xwayland-satellite integration (binds X11 sockets, exports
    // $DISPLAY for children, spawns the satellite on-demand). Single-threaded
    // startup is required for the $DISPLAY env mutation — see its docs.
    prism_protocols::xwayland::satellite::setup(&mut state);
    tracing::info!("WAYLAND_DISPLAY={socket}");
    // IPC socket for runtime control (prism-tune, future prism-msg, etc.).
    // Best-effort: a bringup failure here would lock us out of calibration
    // tooling but shouldn't take the compositor down.
    let ipc_socket_path = match ipc::insert_ipc_source(&event_loop.handle()) {
        Ok(path) => Some(path),
        Err(e) => {
            tracing::warn!("ipc bringup failed; runtime control disabled: {e:#}");
            None
        }
    };
    for output in state.outputs.values() {
        tracing::info!(
            "scanout target: {} {}×{} (crtc {:?})",
            output.connector_name,
            output.extent.width,
            output.extent.height,
            output.crtc
        );
    }

    // Shared shutdown flag, set by signal handlers AND by the vblank handler
    // once max_frames has been hit. Defined here so both can reference it.
    let running = Arc::new(AtomicBool::new(true));
    let frame_counter = Arc::new(AtomicU32::new(0));

    // Wall-clock shutdown timer (if PRISM_MAX_RUNTIME_SECS is set). Flips
    // `running` to false after N seconds. Cleaner than frame-count caps
    // for multi-output, where total frames-per-second scales with the
    // number of outputs.
    if let Some(secs) = max_runtime_secs {
        let running = running.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(secs));
            breadcrumb(&format!(
                "MAX_RUNTIME: {secs}s elapsed, requesting clean exit"
            ));
            running.store(false, Ordering::SeqCst);
        });
    }

    // DRM vblank handler — one per card. Strictly bookkeeping: fire the
    // wp_presentation_feedback that was stashed at the matching submit
    // (with the kernel-reported presentation time), advance the
    // FrameClock, transition the redraw state machine, queue another
    // redraw if needed. The actual render+page_flip happens later, in
    // `redraw_queued_outputs` called from the main loop after dispatch.
    // Keeping GPU work off the vblank thread is what lets wayland event
    // servicing keep up at refresh rate.
    let max_frames_copy = max_frames;
    for drm_notifier in drm_notifiers.drain(..) {
        let running_for_vblank = running.clone();
        let frame_counter_for_vblank = frame_counter.clone();
        event_loop
            .handle()
            .insert_source(drm_notifier, move |event, metadata, state| {
                use smithay::backend::drm::DrmEvent;
                match event {
                    DrmEvent::VBlank(crtc) => {
                        let presentation_time = metadata
                            .as_ref()
                            .map(|m| time_to_monotonic(m.time))
                            .unwrap_or_else(clock_monotonic_now);
                        on_vblank(state, crtc, presentation_time);
                        let n = frame_counter_for_vblank.fetch_add(1, Ordering::SeqCst) + 1;
                        if let Some(max) = max_frames_copy {
                            if n >= max {
                                breadcrumb(&format!("frame #{n}: max_frames reached, exit"));
                                running_for_vblank.store(false, Ordering::SeqCst);
                            }
                        }
                    }
                    DrmEvent::Error(e) => {
                        breadcrumb(&format!("DRM event ERROR: {e:#}"));
                        tracing::warn!("DRM event error: {e:#}");
                    }
                }
            })
            .map_err(|e| anyhow!("insert drm notifier: {e}"))?;
    }

    // Drain libseat session events. CRITICAL: without this, logind can't
    // request a VT switch (we never ack the "pause" message), which blocks
    // Ctrl+Alt+Fn AND blocks SIGINT delivery to us via the desktop session.
    // The callback can be a near-no-op — libseat acks the pause inside its
    // own dispatch path; we just need process_events to run.
    event_loop
        .handle()
        .insert_source(session_notifier, |event, _, _state| {
            use smithay::backend::session::Event as SessionEvent;
            match event {
                SessionEvent::PauseSession => {
                    breadcrumb("session PAUSE");
                    tracing::info!("libseat session paused (likely VT switch away)");
                    // TODO: properly suspend rendering / release DRM resources;
                    // for now we just let subsequent DRM ops fail and log.
                }
                SessionEvent::ActivateSession => {
                    breadcrumb("session ACTIVATE");
                    tracing::info!("libseat session activated");
                    // TODO: re-acquire DRM resources after a previous pause.
                }
            }
        })
        .map_err(|e| anyhow!("insert session notifier: {e}"))?;

    // libinput → LibinputInputBackend → calloop source.
    //
    // The PrismState carries a Weak<LibSeatSession>, but Libinput
    // wants a Session impl by value. The owning LibSeatSession lives
    // in the notifier inserted above; we clone the underlying
    // (Arc-backed) handle here for the libinput interface.
    //
    // udev_assign_seat enumerates every libinput-eligible device
    // (keyboards, mice, touchpads, tablets, touch) on the named seat
    // and emits a DeviceAdded for each — those drive the wl_seat
    // capability flips in prism_input::dispatch::on_device_added.
    {
        use input::Libinput;
        use smithay::backend::libinput::{LibinputInputBackend, LibinputSessionInterface};
        use smithay::backend::session::Session as _;

        let seat_session = state
            .session
            .as_ref()
            .expect("integrated mode always has a session")
            .libseat_clone();
        let seat_name = seat_session.seat();
        let mut libinput = Libinput::new_with_udev(LibinputSessionInterface::from(seat_session));
        libinput
            .udev_assign_seat(&seat_name)
            .map_err(|()| anyhow!("libinput.udev_assign_seat({seat_name}) failed"))?;
        let input_backend = LibinputInputBackend::new(libinput);
        event_loop
            .handle()
            .insert_source(input_backend, |event, _, state| {
                prism_input::process_input_event(state, event);
            })
            .map_err(|e| anyhow!("insert libinput source: {e}"))?;
        tracing::info!("libinput backend running on seat {seat_name}");
    }

    // SIGINT / SIGTERM → clean shutdown.
    {
        let running = running.clone();
        let signals = Signals::new(&[Signal::SIGINT, Signal::SIGTERM]).context("Signals::new")?;
        event_loop
            .handle()
            .insert_source(signals, move |evt, _, _state| {
                tracing::info!(signal = ?evt.signal(), "shutting down");
                running.store(false, Ordering::SeqCst);
            })
            .map_err(|e| anyhow!("insert signals source: {e}"))?;
    }

    // Bootstrap: mark every attached output Queued so the very first
    // redraw_queued_outputs pass below performs the mode-set commit and
    // kicks off the vblank → on_vblank → queue_redraw → render cycle.
    // Subsequent renders are paced by real vblanks (each output's
    // FrameClock predicts the next presentation time on the fly).
    let bootstrap_ids: Vec<_> = state.outputs.keys().cloned().collect();
    for output_id in bootstrap_ids {
        state
            .output_redraw
            .entry(output_id)
            .or_default()
            .queue_redraw();
    }
    // Seed cursor visibility/position on every output's cursor plane
    // before the first present so the cursor appears immediately
    // (otherwise it'd stay invisible until the first pointer event).
    prism_protocols::state::update_output_cursors(&mut state);
    breadcrumb(&format!(
        "bootstrap: {} output(s) queued",
        state.output_redraw.len()
    ));
    redraw_queued_outputs(&mut state);

    breadcrumb("entering dispatch loop");
    while running.load(Ordering::SeqCst) && !state.should_stop {
        event_loop
            .dispatch(Some(Duration::from_millis(100)), &mut state)
            .context("event_loop.dispatch")?;

        // Send the terminating `done()` on any wp_image_description_info_v1
        // resources whose info events were emitted during dispatch.
        // `done` is a destructor event and can't be sent inline from
        // the request handler — see field doc on
        // `ColorManagementState::pending_info_done`.
        state.color_management.drain_pending_info_done();

        // Advance every running animation (view-offset scrolls,
        // window movement, opening/closing fades, etc.). Without
        // this, `Layout::add_window` queues a view-offset animation
        // when a column overflows but the animation never progresses
        // — so a third window stays off-screen rather than scrolling
        // into view.
        state.layout.advance_animations();

        // Flush layout state to clients: walks every window and sends
        // any pending xdg_toplevel.configure (size, position,
        // activation). Without this, newly-mapped windows never learn
        // their column geometry and pile up at (0,0) at their own
        // preferred size. niri does the same via
        // `state.refresh_and_flush_clients()` after every dispatch.
        // `is_active=true` is the layout-focus flag — true while the
        // layout owns keyboard focus (no lock screen / overlay UI
        // intercepting); for prism today that's always.
        state.layout.refresh(true);

        // Re-evaluate what's under the pointer after this cycle's layout
        // changes. A window that moved, resized, or restacked under a
        // stationary pointer (or a subsurface commit changing input
        // geometry) needs an enter/leave the client wouldn't otherwise get,
        // since nothing generated a pointer-motion event.
        prism_input::pointer::refresh_pointer_focus(&mut state);

        state
            .display_handle
            .flush_clients()
            .context("flush_clients")?;
        // Drain any outputs queued by this iteration (vblank handlers,
        // commit handlers, etc.). One pass — if rendering itself sets
        // more outputs Queued (it shouldn't), they'll drain on the next
        // iteration.
        redraw_queued_outputs(&mut state);
        state
            .display_handle
            .flush_clients()
            .context("flush_clients (after redraw)")?;

        // Clear the cached monotonic time so the next iteration's
        // `advance_animations` / `update_render_elements` / `refresh`
        // pulls a fresh `gettime()` instead of re-reading the same
        // sample. Without this the layout's animation engine sees
        // zero elapsed time per tick and animations (view-offset
        // scroll, close-window reflow, etc.) never progress.
        // Mirrors niri/src/niri.rs:776 (`self.niri.clock.clear()`),
        // which is the last thing inside its post-dispatch refresh.
        state.clock.clear();
    }

    breadcrumb("dispatch loop exited cleanly");
    tracing::info!("integrated loop stopped");

    // Drop the IPC socket file so we don't leave a stale node in
    // $XDG_RUNTIME_DIR (next bringup would remove it anyway, but it's
    // tidier to clean up after ourselves).
    if let Some(path) = ipc_socket_path.as_deref() {
        ipc::remove_socket(path);
    }

    // Explicit, instrumented teardown.
    //
    // Order matters. The libseat-grant (DRM master) is held by
    // `LibSeatSessionImpl`, which is owned by `LibSeatSessionNotifier`
    // (inside event_loop). The `SeatSession` we stash in PrismState is
    // just a `Weak<LibSeatSessionImpl>` — dropping PrismState does NOT
    // release master. Master release happens when the libseat notifier
    // source inside event_loop drops.
    //
    // Same shape for DrmDevice: `DrmDeviceNotifier` holds an
    // `Arc<DrmDeviceInternal>`. `DrmDevice::Drop` (which tries its own
    // `clear_state`) only fires when the LAST Arc is gone — which means
    // after both PrismState's `DrmCardContext` AND event_loop's notifier
    // both drop.
    //
    // Therefore: drop PrismState FIRST while master is still held by
    // event_loop's libseat notifier — gives `OutputContext::Drop` a
    // chance to `surface.clear()` successfully. Then drop event_loop;
    // calloop drops sources in insertion order (drm_notifier first,
    // session_notifier second), so DrmDevice::Drop fires (its own
    // clear_state succeeds) BEFORE the libseat seat closes.
    breadcrumb(&format!(
        "shutdown: outputs={} cards={} gpus={} dmabuf_sources={}",
        state.outputs.len(),
        state.cards.len(),
        state.gpus.len(),
        state.dmabuf_sources.len()
    ));
    let t_start = std::time::Instant::now();

    // Take + drop outputs one at a time so we can attribute hangs to a
    // specific OutputContext (surface.clear, Renderer Drop, scanout buffer
    // Drop with imported image + GBM BO).
    let outputs = std::mem::take(&mut state.outputs);
    breadcrumb(&format!("shutdown: dropping {} outputs", outputs.len()));
    for (id, output) in outputs {
        let t = std::time::Instant::now();
        let crtc = output.crtc;
        drop(output);
        breadcrumb(&format!(
            "shutdown: dropped output {id} (crtc {crtc:?}) in {}ms",
            t.elapsed().as_millis()
        ));
    }

    let t = std::time::Instant::now();
    // Drop the dmabuf fd descriptions. The per-GPU VkImages live on
    // surfaces' SurfaceTexSlots and are dropped when `state` (→ Display →
    // clients → surfaces) drops below; their `Arc<Device>` keeps each
    // Device alive until then, so there's no images-outstanding hazard.
    state.dmabuf_sources.clear();
    breadcrumb(&format!(
        "shutdown: cleared dmabuf_sources in {}ms",
        t.elapsed().as_millis()
    ));

    let t = std::time::Instant::now();
    let cards = std::mem::take(&mut state.cards);
    let n_cards = cards.len();
    drop(cards);
    breadcrumb(&format!(
        "shutdown: dropped {n_cards} cards in {}ms",
        t.elapsed().as_millis()
    ));

    let t = std::time::Instant::now();
    let gpus = std::mem::take(&mut state.gpus);
    let n_gpus = gpus.len();
    drop(gpus);
    breadcrumb(&format!(
        "shutdown: dropped {n_gpus} gpus in {}ms",
        t.elapsed().as_millis()
    ));

    let t = std::time::Instant::now();
    drop(state);
    breadcrumb(&format!(
        "shutdown: dropped remaining state in {}ms (state total {}ms)",
        t.elapsed().as_millis(),
        t_start.elapsed().as_millis()
    ));

    let t = std::time::Instant::now();
    drop(event_loop);
    breadcrumb(&format!(
        "shutdown: dropped event_loop in {}ms",
        t.elapsed().as_millis()
    ));
    breadcrumb(&format!(
        "shutdown: returning from run_integrated (total {}ms)",
        t_start.elapsed().as_millis()
    ));
    Ok(())
}

/// Present one frame on a specific output (identified by CRTC handle).
/// Walks currently-mapped xdg toplevels, builds the element list from
/// their cached `SurfaceTexture`s, calls `OutputContext::present` on the
/// matching output.
///
/// Returns true if a flip was submitted, false if skipped (previous flip
/// still pending), Err if the output isn't found or presenting failed.
///
/// Layout-driven composition: walk this output's monitor in the layout,
/// emit `RenderEl`s with each tile projected to clip space, lower to
/// `ElementDraw`s, and submit through the OutputContext. Replaces the
/// pre-layout "first toplevel fills the framebuffer" bypass.
/// Lightweight DRM-vblank handler. Bookkeeping only — no render, no
/// page-flip. Called from the DrmEvent::VBlank dispatch path in the main
/// loop. The actual render+page-flip for the next frame is performed by
/// [`redraw_queued_outputs`] in the same loop iteration, *after*
/// `event_loop.dispatch` returns, so wayland clients also get serviced
/// in between.
///
/// Steps, in order:
/// 1. Advance the output's `frame_clock` and back-buffer / `frame_pending`
///    bookkeeping with the kernel-reported `presentation_time`.
/// 2. Take the `PendingFeedback` we stashed at the matching submit and
///    fire `wl_callback.frame` + `wp_presentation_feedback.presented`
///    with the *actual* presentation time. This is what stops clients
///    (mpv) from over-producing: the feedback signal goes out when the
///    flip actually landed on screen, not when we queued it.
/// 3. Transition the redraw state machine: `WaitingForVBlank { redraw_needed }`
///    decides whether we re-queue or go idle. Today we always re-queue
///    (matches the "render every vblank" behaviour we had before Stage B);
///    Stage D will replace that with damage-driven scheduling.
fn on_vblank(
    state: &mut prism_protocols::PrismState,
    crtc: smithay::reexports::drm::control::crtc::Handle,
    presentation_time: Duration,
) {
    use prism_protocols::redraw::RedrawState;

    // Resolve crtc → OutputId (small map; lookup is fine).
    let Some(output_id) = state
        .outputs
        .iter()
        .find(|(_, o)| o.crtc == crtc)
        .map(|(id, _)| id.clone())
    else {
        breadcrumb(&format!("vblank for unknown crtc {crtc:?}"));
        return;
    };

    prism_drm::flip_trace(&format!(
        "vblank {} crtc={:?} pres_us={}",
        output_id,
        crtc,
        presentation_time.as_micros()
    ));

    // Step 1: per-output DRM bookkeeping (frame_pending, back_buffer
    // toggle, frame_clock update).
    if let Some(ctx) = state.outputs.get_mut(&output_id) {
        ctx.mark_vblank(presentation_time);
    }

    // Step 2: advance this output's refresh-cycle counter and deliver frame
    // callbacks (throttled to one per surface per cycle), decoupled from the
    // page-flip. The mutable borrow for the bump is scoped so the immutable
    // `send_frame_callbacks(state, …)` borrow that follows doesn't conflict.
    {
        let entry = state.output_redraw.entry(output_id.clone()).or_default();
        entry.frame_callback_sequence = entry.frame_callback_sequence.wrapping_add(1);
    }
    send_frame_callbacks(state, &output_id, presentation_time);

    // Step 3: take + fire the stashed *presentation* feedback for the
    // just-presented frame (this is the one thing that genuinely belongs on the
    // real vblank). Split take-from-fire so the fire can hold immutable borrows.
    let pending = state
        .output_redraw
        .entry(output_id.clone())
        .or_default()
        .pending_feedback
        .take();

    if let Some(pending) = pending {
        if let (Some(smithay_output), Some(ctx)) = (
            state.wl_outputs.get(&output_id).cloned(),
            state.outputs.get(&output_id),
        ) {
            let hz = ctx.mode.vrefresh().max(1);
            let refresh = smithay::wayland::presentation::Refresh::fixed(Duration::from_nanos(
                1_000_000_000 / hz as u64,
            ));
            use smithay::reexports::wayland_protocols::wp::presentation_time::server::wp_presentation_feedback;
            for fb in pending.presentation_cbs {
                fb.presented(
                    &smithay_output,
                    presentation_time,
                    refresh,
                    // We don't track a real vblank sequence yet; the kernel
                    // gives us metadata.sequence but plumbing it through is
                    // a separate change. Monotonic-zero is sane for mpv's
                    // glitch-detection.
                    0,
                    wp_presentation_feedback::Kind::Vsync,
                );
            }
        }
    }

    // Step 4: transition the state machine. Damage-driven now —
    // commit handlers call `queue_redraw_for_surface` to flip a
    // WaitingForVBlank entry's `redraw_needed` to true; on this
    // vblank we honour that signal. If nothing requested a redraw
    // between submit and now, the output goes Idle until the next
    // commit lands UNLESS the layout has an animation in progress
    // on this output's monitor — in that case we need another frame
    // to advance the animation (close-window column reflow,
    // view-offset scroll-to-new-column, etc.). Without this check the
    // animation ticks once via `Layout::advance_animations` in the
    // main loop but no fresh frame ever renders.
    let smithay_output = state.wl_outputs.get(&output_id).cloned();
    let animations_ongoing = smithay_output
        .as_ref()
        .map(|o| state.layout.are_animations_ongoing(Some(o)))
        .unwrap_or(false);

    let entry = state.output_redraw.entry(output_id).or_default();
    let prev = std::mem::take(&mut entry.redraw);
    entry.redraw = match prev {
        RedrawState::WaitingForVBlank {
            redraw_needed: true,
        } => RedrawState::Queued,
        RedrawState::WaitingForVBlank {
            redraw_needed: false,
        } => {
            if animations_ongoing {
                RedrawState::Queued
            } else {
                RedrawState::Idle
            }
        }
        other => other,
    };
}

/// Drain every output whose `redraw` state is `Queued`: build its render
/// elements, render and submit the page-flip, stash the surfaces'
/// `wl_callback.frame` and `wp_presentation_feedback` objects so the
/// matching vblank handler can fire them with the actual presentation
/// timestamp. Called once per main-loop iteration, after `dispatch`.
fn redraw_queued_outputs(state: &mut prism_protocols::PrismState) {
    use prism_protocols::redraw::RedrawState;

    let to_render: Vec<_> = state
        .output_redraw
        .iter()
        .filter(|&(_id, st)| matches!(st.redraw, RedrawState::Queued))
        .map(|(id, _st)| id.clone())
        .collect();

    for output_id in to_render {
        render_one_queued(state, &output_id);
    }
}

/// Render one queued output, stash its presentation feedback, and
/// transition its redraw state (`WaitingForVBlank` on a real flip,
/// `WaitingForEstimatedVBlank` on a zero-damage skip).
///
/// Called only from [`redraw_queued_outputs`] (once per main-loop iteration,
/// draining every `Queued` output). Outputs typically reach `Queued` at their
/// own vblank (`on_vblank`), with non-vblank sources too (commit handlers,
/// animation ticks, bootstrap). Because per-CRTC vblanks are staggered across
/// the wall-clock, N outputs rarely become `Queued` for the same pass.
///
/// That staggering matters: bursting all per-card page-flips into one ~150µs
/// window overflowed amdgpu's atomic-commit allocator ceiling on Vega 20 + fp16
/// scanout (ENOMEM on the next submit). wlroots/sway commit per-output from
/// their own per-CRTC vblank handler for the same reason —
/// `backend/drm/drm.c:2086 wlr_output_send_frame`.
fn render_one_queued(state: &mut prism_protocols::PrismState, output_id: &str) {
    use prism_protocols::redraw::RedrawState;
    // Powered-off (DPMS) outputs never render — rendering would re-modeset
    // and wake the panel. Drop any queued redraw to Idle so a commit- or
    // animation-driven queue doesn't spin (these outputs emit no vblanks, so
    // nothing else clears the Queued state). power_on re-queues explicitly.
    if state
        .outputs
        .get(output_id)
        .is_some_and(|o| o.is_powered_off())
    {
        state
            .output_redraw
            .entry(output_id.to_owned())
            .or_default()
            .redraw = RedrawState::Idle;
        return;
    }
    match render_output_now(state, output_id) {
        Ok(RenderOutcome::Presented(pending)) => {
            let entry = state.output_redraw.entry(output_id.to_owned()).or_default();
            entry.pending_feedback = Some(pending);
            entry.redraw = RedrawState::WaitingForVBlank {
                redraw_needed: false,
            };
        }
        Ok(RenderOutcome::SkippedNoDamage) => {
            // Nothing changed, so no page-flip and thus no real vblank will
            // arrive to advance the frame-callback cycle or resume animations.
            // Arm an estimated-vblank timer in its place.
            queue_estimated_vblank(state, output_id);
        }
        Ok(RenderOutcome::FlipPending) => {
            // Flip still in flight. Shouldn't normally happen (we only enter
            // Queued after a vblank cleared frame_pending), but defensive:
            // leave Queued so the next pass retries.
            tracing::debug!(output = %output_id, "render_output_now: flip still pending, retry next pass");
        }
        Err(e) => {
            tracing::warn!("render_output_now({output_id}) failed: {e:#}");
            breadcrumb(&format!("render_output_now({output_id}) ERROR: {e:#}"));
            if let Some(entry) = state.output_redraw.get_mut(output_id) {
                entry.redraw = RedrawState::Idle;
            }
        }
    }
}

/// Arm an estimated-vblank timer for an output whose render was skipped for
/// lack of damage. A skipped frame submits no page-flip, so no real vblank will
/// arrive; this timer, fired at the predicted next vblank, substitutes for it —
/// advancing the frame-callback cycle (so clients keep getting callbacks) and
/// resuming any animation. Mirrors niri's `queue_estimated_vblank_timer`.
///
/// At most one timer is armed per output: if one is already pending we keep it.
fn queue_estimated_vblank(state: &mut prism_protocols::PrismState, output_id: &str) {
    use prism_protocols::redraw::RedrawState;
    use smithay::reexports::calloop::timer::{TimeoutAction, Timer};

    // Don't double-arm — keep an already-pending timer.
    if matches!(
        state.output_redraw.get(output_id).map(|s| &s.redraw),
        Some(RedrawState::WaitingForEstimatedVBlank(_))
            | Some(RedrawState::WaitingForEstimatedVBlankAndQueued(_))
    ) {
        return;
    }

    let Some((target, refresh)) = state.outputs.get(output_id).map(|o| {
        (
            o.frame_clock.next_presentation_time(),
            o.frame_clock.refresh_interval(),
        )
    }) else {
        return;
    };
    // Fire at the predicted vblank. If that's already due (zero), wait one
    // refresh — a zero-delay timer would just respin (we'd re-skip immediately).
    let now = clock_monotonic_now();
    let mut duration = target.saturating_sub(now);
    if duration.is_zero() {
        duration += refresh.unwrap_or(Duration::from_micros(16_667));
    }

    let Some(loop_handle) = state.loop_handle.clone() else {
        // No event loop yet (pre-bringup) — fall back to Idle rather than wedge.
        if let Some(entry) = state.output_redraw.get_mut(output_id) {
            entry.redraw = RedrawState::Idle;
        }
        return;
    };
    let oid = output_id.to_owned();
    let res = loop_handle.insert_source(Timer::from_duration(duration), move |_, _, state| {
        on_estimated_vblank(state, &oid);
        TimeoutAction::Drop
    });
    let entry = state.output_redraw.entry(output_id.to_owned()).or_default();
    entry.redraw = match res {
        Ok(token) => RedrawState::WaitingForEstimatedVBlank(token),
        Err(e) => {
            tracing::warn!(output = %output_id, "failed to arm estimated-vblank timer: {e}");
            RedrawState::Idle
        }
    };
}

/// Fired by the estimated-vblank timer (see [`queue_estimated_vblank`]). Stands
/// in for a real vblank on a frame we chose not to flip: advance the
/// frame-callback cycle, deliver callbacks, and re-queue if a redraw is now
/// wanted (a commit arrived, or an animation is ongoing) — else go idle.
fn on_estimated_vblank(state: &mut prism_protocols::PrismState, output_id: &str) {
    use prism_protocols::redraw::RedrawState;

    // Act only if we're still waiting on this estimated vblank (a later
    // transition may have superseded the timer that fired us).
    if !matches!(
        state.output_redraw.get(output_id).map(|s| &s.redraw),
        Some(RedrawState::WaitingForEstimatedVBlank(_))
            | Some(RedrawState::WaitingForEstimatedVBlankAndQueued(_))
    ) {
        return;
    }

    // Advance the refresh-cycle counter and deliver frame callbacks, exactly as
    // the real vblank handler does. Unlike niri (which sends in `redraw`), prism
    // only sends in the vblank handlers, so we MUST send here on every estimated
    // vblank — otherwise a client flooding frame-callback-only commits (which
    // keep us in the AndQueued↔Queued↔skip loop) would never be unblocked.
    {
        let entry = state.output_redraw.entry(output_id.to_owned()).or_default();
        entry.frame_callback_sequence = entry.frame_callback_sequence.wrapping_add(1);
    }
    send_frame_callbacks(state, output_id, clock_monotonic_now());

    // Transition: a redraw queued while we waited ⇒ Queued; an ongoing animation
    // ⇒ Queued (advance it next pass); otherwise the output is now idle.
    let smithay_output = state.wl_outputs.get(output_id).cloned();
    let animations_ongoing = smithay_output
        .as_ref()
        .map(|o| state.layout.are_animations_ongoing(Some(o)))
        .unwrap_or(false);
    let entry = state.output_redraw.entry(output_id.to_owned()).or_default();
    entry.redraw = match std::mem::take(&mut entry.redraw) {
        RedrawState::WaitingForEstimatedVBlankAndQueued(_) => RedrawState::Queued,
        RedrawState::WaitingForEstimatedVBlank(_) => {
            if animations_ongoing {
                RedrawState::Queued
            } else {
                RedrawState::Idle
            }
        }
        other => other,
    };
}

/// Render one output now and submit the page-flip. Returns a [`RenderOutcome`]:
/// `Presented` carries the `PendingFeedback` to stash for the matching vblank;
/// `SkippedNoDamage` if nothing changed (caller arms an estimated vblank);
/// `FlipPending` if the output's previous flip is still in flight (caller
/// retries).
fn render_output_now(
    state: &mut prism_protocols::PrismState,
    output_id: &str,
) -> Result<RenderOutcome> {
    use prism_layout::layout::RenderCtx;
    use prism_protocols::PendingFeedback;
    use prism_renderer::{vk, EncodePush, RenderEl};

    // Snapshot identity bits without holding any borrow into
    // state.outputs (we'll re-borrow mutably at present() time below).
    // The "effective" reads here pick runtime IPC overrides ahead of
    // the persisted KDL values, so calibration tools can iterate live.
    let (
        output_gpu_id,
        white_view,
        target_time,
        output_sdr_reference_nits,
        output_decode_clamp_bt2020_rgb,
    ) = {
        let output = state
            .outputs
            .get(output_id)
            .ok_or_else(|| anyhow!("no output bound to id {output_id}"))?;
        (
            output.gpu_id,
            output.renderer.white_view(),
            output.frame_clock.next_presentation_time(),
            output.effective_sdr_reference_nits(),
            output.effective_decode_clamp_bt2020_rgb(),
        )
    };

    // The smithay Output is the key the layout uses to find its
    // Monitor. wl_outputs is populated by advertise_output().
    let smithay_output = state
        .wl_outputs
        .get(output_id)
        .cloned()
        .ok_or_else(|| anyhow!("no smithay Output for {output_id}"))?;

    // Build the render walk's inputs:
    //   ctx.texture_lookup: WlSurface → vk::ImageView (per-GPU)
    //
    // It closes over things that don't touch &mut state, so the walk and the
    // present can sequence cleanly. `view_size` (logical pixels) is handed to
    // the renderer at lowering time, where it owns the logical → clip-space
    // projection; the walk itself emits output-space logical geometry.
    let view_size = match state.layout.monitor_for_output(&smithay_output) {
        Some(m) => m.view_size(),
        // Output not in the layout yet (race between add_output and the
        // first vblank). Leave it Queued and retry next pass (as the old
        // `Ok(None)` did); the next pass will find the monitor.
        None => return Ok(RenderOutcome::FlipPending),
    };

    let texture_lookup =
        |states: &smithay::wayland::compositor::SurfaceData| -> Option<vk::ImageView> {
            states
                .data_map
                .get::<prism_protocols::SurfaceTexSlot>()
                .and_then(|s| {
                    s.0.lock()
                        .unwrap()
                        .as_ref()
                        .and_then(|t| t.view_for(output_gpu_id))
                })
        };
    // Chroma plane + YUV kind for video surfaces, on this output's GPU.
    // Parallels texture_lookup (the luma/primary plane); `(None, 0)` for RGB.
    let yuv_lookup =
        |states: &smithay::wayland::compositor::SurfaceData| -> (Option<vk::ImageView>, i32) {
            states
                .data_map
                .get::<prism_protocols::SurfaceTexSlot>()
                .and_then(|s| {
                    s.0.lock()
                        .unwrap()
                        .as_ref()
                        .map(|t| t.yuv_for(output_gpu_id))
                })
                .unwrap_or((None, 0))
        };
    // How to interpret each surface's sampled alpha (opaque X-format/YUV vs
    // premultiplied A-format). A buffer-format property, GPU-independent.
    let alpha_mode_lookup =
        |states: &smithay::wayland::compositor::SurfaceData| -> prism_renderer::AlphaMode {
            states
                .data_map
                .get::<prism_protocols::SurfaceTexSlot>()
                .and_then(|s| s.0.lock().unwrap().as_ref().map(|t| t.alpha_mode()))
                .unwrap_or_default()
        };
    // Per-surface decode params from wp_color_management_v1. Falls
    // through to RenderCtx::color_for's default (sRGB + the output's
    // sdr_reference_nits) for surfaces with no description set —
    // that's the pre-color-management path every existing client
    // still uses, now scaled per the output's KDL config.
    let color_lookup = |states: &smithay::wayland::compositor::SurfaceData|
        -> Option<prism_renderer::SurfaceColorParams> {
        prism_protocols::color_management::SurfaceColorSlot::current(states)
            .as_deref()
            .map(prism_protocols::color_management::description_to_params)
    };
    // Render-demand safety net: surfaces the walk finds without a texture
    // on this output's GPU get collected here and materialized after the
    // walk (GPU work can't happen inside the walk — it holds surface
    // locks). See RenderCtx::report_missing_texture.
    let missing_textures: std::cell::RefCell<
        Vec<smithay::reexports::wayland_server::protocol::wl_surface::WlSurface>,
    > = std::cell::RefCell::new(Vec::new());
    let report_missing =
        |s: &smithay::reexports::wayland_server::protocol::wl_surface::WlSurface| {
            missing_textures.borrow_mut().push(s.clone());
        };
    // Surfaces drawn on this output, collected during the walk for pre-present
    // GPU-sync prep:
    //   - `mirror_surfaces`: texture is a cross-GPU mirror → async home→scratch
    //     copy before the present submit;
    //   - `acquire_surfaces`: zero-copy native dmabuf → import the client's
    //     implicit write fence as a render wait so we don't sample mid-write.
    let mirror_surfaces: std::cell::RefCell<
        Vec<smithay::reexports::wayland_server::protocol::wl_surface::WlSurface>,
    > = std::cell::RefCell::new(Vec::new());
    let acquire_surfaces: std::cell::RefCell<
        Vec<smithay::reexports::wayland_server::protocol::wl_surface::WlSurface>,
    > = std::cell::RefCell::new(Vec::new());
    let report_drawn_surface =
        |s: &smithay::reexports::wayland_server::protocol::wl_surface::WlSurface,
         states: &smithay::wayland::compositor::SurfaceData| {
            let Some(slot) = states.data_map.get::<prism_protocols::SurfaceTexSlot>() else {
                return;
            };
            let guard = slot.0.lock().unwrap();
            let Some(tex) = guard.as_ref() else { return };
            if tex.is_mirror_for(output_gpu_id) {
                mirror_surfaces.borrow_mut().push(s.clone());
            } else if tex.is_native_dmabuf_for(output_gpu_id) && tex.acquire_pending {
                // Only surfaces written since their last acquire wait — keeps
                // the per-frame sync_file/semaphore churn bounded to changed
                // tiles (a static buffer's write is long done).
                acquire_surfaces.borrow_mut().push(s.clone());
            }
        };
    // Solid color (wp_single_pixel_buffer) lookup — the walk lowers these to a
    // SolidColorEl instead of sampling a texture. Premultiplied sRGB RGBA.
    let solid_color_lookup =
        |states: &smithay::wayland::compositor::SurfaceData| -> Option<[u8; 4]> {
            states
                .data_map
                .get::<prism_protocols::SurfaceTexSlot>()
                .and_then(|s| s.0.lock().unwrap().as_ref().and_then(|t| t.solid_color()))
        };
    let ctx = RenderCtx {
        texture_lookup: &texture_lookup,
        yuv_lookup: &yuv_lookup,
        alpha_mode_lookup: &alpha_mode_lookup,
        color_lookup: &color_lookup,
        sdr_reference_nits: output_sdr_reference_nits,
        report_missing_texture: &report_missing,
        report_drawn_surface: &report_drawn_surface,
        solid_color_lookup: &solid_color_lookup,
    };

    // Refresh per-tile cached render elements (focus ring / border /
    // shadow geometry). Without this, the FocusRing's `cached` is
    // never populated and `render` early-returns — the ring is
    // invisible even when configured.
    state.layout.update_render_elements(Some(&smithay_output));

    // Layout walk into a flat RenderEl vector. Monitor is borrowed
    // immutably for the duration of render_workspaces; dropped before
    // the present below mutably re-borrows state.outputs.
    let mut render_els: Vec<RenderEl> = Vec::new();

    // wlr_layer_shell: walk each mapped layer surface (main + subsurfaces)
    // through the SAME color-managed surface-tree walk windows use, so layer
    // chrome (bars, wallpapers, notification daemons) gets identical
    // wp_color_management decode, cross-GPU mirror handling, and subsurface
    // z-ordering — no separate unmanaged path. Geometry comes from the
    // per-output LayerMap (anchors / margins / exclusive zones, arranged on
    // commit). Cross-layer Z is the append order into the back-to-front
    // render_els: Background + Bottom go BELOW the workspace, Top + Overlay
    // ABOVE it (and above any interactive-move tile).
    use smithay::wayland::shell::wlr_layer::Layer;
    let push_layers = |layers: &[Layer], out: &mut Vec<RenderEl>| {
        let map = smithay::desktop::layer_map_for_output(&smithay_output);
        for &which in layers {
            for ls in map.layers_on(which) {
                let Some(geo) = map.layer_geometry(ls) else {
                    continue;
                };
                prism_layout::layout::element::push_surface_tree_elements(
                    ls.wl_surface(),
                    geo.loc.to_f64(),
                    &ctx,
                    out,
                );
            }
        }
    };
    push_layers(&[Layer::Background, Layer::Bottom], &mut render_els);

    let monitor_found = if let Some(monitor) = state.layout.monitor_for_output(&smithay_output) {
        // focus_ring: this is the focused monitor's render — for
        // single-monitor configs it always is; multi-monitor focus
        // tracking lands when input dispatch does.
        monitor.render_workspaces(true, &ctx, &mut render_els);
        true
    } else {
        false
    };

    // During an interactive move, the moving tile is detached from its
    // workspace's normal layout — `render_workspaces` above does NOT
    // include it. The layout exposes the moving tile separately;
    // `render_interactive_move_for_output` early-returns unless the
    // tile is currently assigned to *this* output (the layout transfers
    // the assignment as the cursor crosses output boundaries during
    // the drag). Append after the workspace pass so the moving window
    // draws on top of normal tiles.
    state
        .layout
        .render_interactive_move_for_output(&smithay_output, &ctx, &mut render_els);

    // Top + Overlay layers: above the workspace walk and the interactive-move
    // tile. Same color-managed walk as Background/Bottom. Done before the
    // demand-materialize pass below so a layer surface missing a texture this
    // frame is caught + retried exactly like a window.
    push_layers(&[Layer::Top, Layer::Overlay], &mut render_els);

    // Render-demand safety net: materialize any surfaces the walk drew on
    // this output but had no texture for its GPU (spanning windows,
    // surfaces committed before placement, layer surfaces). They render
    // blank this frame; materialize now (outside the walk — safe to do GPU
    // work + with_states here) and queue a redraw so they draw next frame.
    let missing = missing_textures.take();
    if !missing.is_empty() {
        for surf in &missing {
            prism_protocols::materialize_surface_on_gpu(state, surf, output_gpu_id);
        }
        tracing::debug!(
            output = %output_id,
            count = missing.len(),
            "demand-materialized missing surface textures"
        );
        state
            .output_redraw
            .entry(output_id.to_string())
            .or_default()
            .queue_redraw();
    }

    // Lower RenderEls (output-space logical geometry + tint) into a
    // LoweredFrame: the flat ElementDraw stream (clip-space + push constants)
    // render_frame consumes, plus the per-element metadata the damage tracker
    // diffs. The renderer owns the logical → clip projection (built once from
    // `view_size`); SolidColor/Border elements bind the white texel, Surface
    // elements bind the per-surface view. The per-output panel peak is threaded
    // through so the decoder's display-referred clamp lands at the right value.
    let frame = prism_renderer::lower_elements(
        &render_els,
        view_size,
        white_view,
        output_decode_clamp_bt2020_rgb,
    );

    // Once per output, the first present that actually carries tiles —
    // a single tracing line we use as a regression sentinel for
    // "did this output's render walk see the layout's window?".
    static FIRST_WITH_TILES: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashSet<String>>,
    > = std::sync::OnceLock::new();
    let has_surface = render_els.iter().any(|e| matches!(e, RenderEl::Surface(_)));
    if has_surface {
        let seen = FIRST_WITH_TILES
            .get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
        if seen.lock().unwrap().insert(output_id.to_owned()) {
            let first_surface = render_els.iter().find_map(|e| match e {
                RenderEl::Surface(s) => Some(s.geometry),
                _ => None,
            });
            tracing::info!(
                output = %output_id,
                view_w = view_size.w,
                view_h = view_size.h,
                monitor_found,
                n_render_els = render_els.len(),
                n_draws = frame.draws.len(),
                ?first_surface,
                "FIRST present with tiles for output"
            );
        }
    }

    // Build the encode push from the output's config: PQ outputs clamp
    // at the panel's declared peak, and per-output response correction
    // (if configured) gets its push-constant slots filled.
    let encode_push = {
        let output = state
            .outputs
            .get(output_id)
            .ok_or_else(|| anyhow!("no output bound to id {output_id}"))?;
        let mut p = match output.config.hdr {
            Some(hdr) => {
                let mut p = EncodePush::pq_identity();
                p.target_peak_nits = hdr.max_luminance as f32;
                p
            }
            None => EncodePush::sdr_identity(),
        };
        p.sdr_white_nits = output.effective_sdr_reference_nits();
        if let Some((gain, gamma)) = output.effective_response_curve() {
            p.set_response_gain_gamma(gain, gamma);
        }
        if let Some(m) = output.effective_ctm() {
            p.set_ctm(m);
        }
        p
    };
    // Pre-present GPU-sync waits the render submit blocks on (GPU-side, not on
    // the event loop). Computed before the mutable borrow of state.outputs:
    //   - cross-GPU mirror: submit home→scratch copies async, wait on them;
    //   - native dmabuf: import each client's implicit write fence so we don't
    //     sample a buffer the client's GPU is still writing (the Vulkan analog
    //     of the implicit sync a GL compositor gets from Mesa for free).
    // Both empty in the trivial single-GPU, all-shm case.
    let mirror_surfaces = mirror_surfaces.take();
    let acquire_surfaces = acquire_surfaces.take();
    let mut render_waits =
        prism_protocols::prepare_mirror_waits(state, &mirror_surfaces, output_gpu_id);
    render_waits.extend(prism_protocols::prepare_dmabuf_acquire_waits(
        state,
        &acquire_surfaces,
        output_gpu_id,
    ));
    let outcome = {
        let output = state
            .outputs
            .get_mut(output_id)
            .ok_or_else(|| anyhow!("no output bound to id {output_id}"))?;
        output.present(&frame, view_size, &encode_push, &render_waits)?
    };
    // The render submit has been queued with the waits in its dependency list
    // (or, on skip / flip-pending, never used them); either way the imported
    // semaphores can be destroyed now.
    prism_protocols::destroy_render_wait_semaphores(state, output_gpu_id, render_waits);

    let present_sync_fd = match outcome {
        prism_drm::PresentOutcome::Presented(fd) => fd,
        // Flip still in flight — caller leaves the output Queued and retries.
        prism_drm::PresentOutcome::FlipPending => return Ok(RenderOutcome::FlipPending),
        // Nothing changed — caller arms an estimated vblank instead of waiting
        // for a real one. No harvest (no scanout, so no presentation feedback).
        prism_drm::PresentOutcome::SkippedNoDamage => return Ok(RenderOutcome::SkippedNoDamage),
    };

    // Extract pending `wp_presentation_feedback` from every surface mapped to
    // this output so we can fire it at the next vblank with the kernel-reported
    // presentation time. Firing it now (at submit, before scanout) would lie to
    // clients about when the buffer hit the screen and cause over-production /
    // stalls — see the redraw module's docs. (Frame callbacks are delivered
    // separately by `send_frame_callbacks`, throttled per refresh cycle.)
    //
    // Same walk also harvests `wp_linux_drm_syncobj` release trackers
    // for surfaces that opted into explicit sync: every surface we
    // just rendered contributes one Arc clone, and we hand the whole
    // batch + the present sync_fd to drm_syncobj's release wiring.
    // When the fd signals (Vulkan submit done), the Arcs drop; the
    // last drop across all outputs that sampled the surface signals
    // the client's release point.
    let surfaces: Vec<_> = state
        .xdg_shell
        .toplevel_surfaces()
        .iter()
        .map(|t| t.wl_surface().clone())
        .collect();

    let mut presentation_cbs = Vec::new();
    let mut release_trackers = Vec::new();

    // Harvest presentation feedback + drm_syncobj release trackers from each
    // surface we rendered. Frame callbacks are NOT harvested here (we pass
    // `None`): they're delivered by `send_frame_callbacks` at vblank, throttled
    // per refresh cycle — draining them now would steal them before that runs.
    // The subsurface-descending walk and deadlock-safe direct reads live in
    // `redraw::harvest_surface_feedback` so the WLCS harness reuses the traversal.
    for surface in &surfaces {
        let belongs_here = state
            .layout
            .find_window_and_output(surface)
            .and_then(|(_, out)| out)
            .map(|out| out == &smithay_output)
            .unwrap_or(false);
        if !belongs_here {
            continue;
        }
        prism_protocols::redraw::harvest_surface_feedback(
            surface,
            None,
            &mut presentation_cbs,
            &mut release_trackers,
        );
        // Popups are separate surface trees parented to this toplevel, not
        // part of its subsurface tree, so the walk above doesn't reach them.
        for (popup, _) in smithay::desktop::PopupManager::popups_for_surface(surface) {
            prism_protocols::redraw::harvest_surface_feedback(
                popup.wl_surface(),
                None,
                &mut presentation_cbs,
                &mut release_trackers,
            );
        }
    }
    // Same harvest for every layer-shell surface we just rendered (all four
    // layers now composite). harvest_surface_feedback descends each surface's
    // subsurface tree itself, so the layer roots are all we pass in.
    {
        let map = smithay::desktop::layer_map_for_output(&smithay_output);
        for ls in map.layers() {
            prism_protocols::redraw::harvest_surface_feedback(
                ls.wl_surface(),
                None,
                &mut presentation_cbs,
                &mut release_trackers,
            );
        }
    }

    if let Some(loop_handle) = state.loop_handle.as_ref() {
        prism_protocols::drm_syncobj::register_release_after_submit(
            loop_handle,
            present_sync_fd,
            release_trackers,
        );
    } else {
        // No LoopHandle stashed — drop the trackers (signals release
        // immediately, racy with in-flight GPU). Only reachable if
        // main forgot to call state.set_loop_handle before dispatch;
        // log loudly.
        if !release_trackers.is_empty() {
            tracing::error!(
                "drm_syncobj: state.loop_handle is None during render — \
                 release points will fire before GPU completes"
            );
        }
        drop(release_trackers);
        drop(present_sync_fd);
    }

    Ok(RenderOutcome::Presented(PendingFeedback {
        presentation_cbs,
        target_time,
    }))
}

/// Outcome of [`render_output_now`] — mirrors `prism_drm::PresentOutcome` but
/// carries the harvested `PendingFeedback` on the presented path.
enum RenderOutcome {
    /// Rendered + flipped; the stash to fire at the matching vblank.
    Presented(prism_protocols::PendingFeedback),
    /// Nothing changed; no flip happened. Caller arms an estimated vblank.
    SkippedNoDamage,
    /// A previous flip is still in flight; caller retries next pass.
    FlipPending,
}

/// Deliver `wl_callback.frame` callbacks to every surface mapped to `output_id`,
/// throttled to at most one per surface per output refresh cycle (the output's
/// `frame_callback_sequence`). Decoupled from the page-flip: callable from the
/// vblank handler (Stage A) and, later, from the zero-damage skip path (Stage B)
/// so a skipped frame still unblocks clients that throttle on frame callbacks.
///
/// `wp_presentation_feedback` is deliberately NOT sent here — it means "your
/// buffer reached the screen", so it stays tied to the real vblank (see
/// [`prism_protocols::PendingFeedback`]).
///
/// Resolution mirrors `render_output_now`'s harvest exactly (toplevels mapped to
/// this output + their popup trees + layer-shell surfaces) so the same surfaces
/// get callbacks. `throttle = None` neutralises smithay's own time-based
/// throttle in `send_frames_surface_tree`, leaving our per-surface sequence
/// check (the closure returning `Some(output)`) the sole send trigger.
fn send_frame_callbacks(
    state: &prism_protocols::PrismState,
    output_id: &str,
    time: std::time::Duration,
) {
    use prism_protocols::redraw::FrameCallbackThrottle;
    use smithay::desktop::utils::send_frames_surface_tree;

    let Some(smithay_output) = state.wl_outputs.get(output_id).cloned() else {
        return;
    };
    let sequence = state
        .output_redraw
        .get(output_id)
        .map(|s| s.frame_callback_sequence)
        .unwrap_or(0);

    let mut should_send =
        |_surface: &smithay::reexports::wayland_server::protocol::wl_surface::WlSurface,
         states: &smithay::wayland::compositor::SurfaceData| {
            states
                .data_map
                .insert_if_missing_threadsafe(FrameCallbackThrottle::default);
            let throttle = states.data_map.get::<FrameCallbackThrottle>().unwrap();
            if throttle.should_send(output_id, sequence) {
                Some(smithay_output.clone())
            } else {
                None
            }
        };

    // Toplevel windows mapped to this output, plus their popup trees.
    let toplevels: Vec<_> = state
        .xdg_shell
        .toplevel_surfaces()
        .iter()
        .map(|t| t.wl_surface().clone())
        .collect();
    for surface in &toplevels {
        let belongs_here = state
            .layout
            .find_window_and_output(surface)
            .and_then(|(_, out)| out)
            .map(|out| out == &smithay_output)
            .unwrap_or(false);
        if !belongs_here {
            continue;
        }
        send_frames_surface_tree(surface, &smithay_output, time, None, &mut should_send);
        for (popup, _) in smithay::desktop::PopupManager::popups_for_surface(surface) {
            send_frames_surface_tree(
                popup.wl_surface(),
                &smithay_output,
                time,
                None,
                &mut should_send,
            );
        }
    }

    // Layer-shell surfaces composited on this output.
    let map = smithay::desktop::layer_map_for_output(&smithay_output);
    for ls in map.layers() {
        send_frames_surface_tree(
            ls.wl_surface(),
            &smithay_output,
            time,
            None,
            &mut should_send,
        );
    }
}

/// Convert smithay's `DrmEventTime` (monotonic or realtime) to the
/// CLOCK_MONOTONIC `Duration` `wp_presentation_feedback` expects. If
/// the kernel handed us a realtime timestamp instead of monotonic (rare;
/// requires the driver to support DRM_CAP_TIMESTAMP_MONOTONIC = 1, which
/// AMDGPU always does), we fall back to `clock_monotonic_now()`.
fn time_to_monotonic(time: smithay::backend::drm::DrmEventTime) -> Duration {
    match time {
        smithay::backend::drm::DrmEventTime::Monotonic(d) => d,
        smithay::backend::drm::DrmEventTime::Realtime(_) => clock_monotonic_now(),
    }
}

/// CLOCK_MONOTONIC right now as a `Duration` since the kernel's boot
/// reference. Used for `wl_callback.frame` timestamps and
/// `wp_presentation_feedback.presented` times — both want the same
/// clock that `wp_presentation` advertises (CLOCK_MONOTONIC for us).
fn clock_monotonic_now() -> std::time::Duration {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: ts is a stack timespec we just zeroed; CLOCK_MONOTONIC
    // is always supported on Linux; we check return for the off chance
    // and fall back to zero (clients diff timestamps, so zero is fine
    // as long as we're consistent).
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    if rc != 0 {
        return std::time::Duration::ZERO;
    }
    std::time::Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
}

/// Same TTY-required mode-set as `scanout`, but instead of `vkCmdClearColorImage`
/// the scanout image is rendered through the two-pass decode→encode pipeline
/// using a horizontal-gradient texture. Visual verification: a smoothly
/// gamma-correct gradient (black on the left → white on the right).
fn run_gradient_scanout(output_name: Option<&str>, depth: prism_drm::ScanoutDepth) -> Result<()> {
    use prism_drm::scanout;
    use prism_renderer::{vk, DecodePush, ElementDraw, EncodePush, Renderer};
    use smithay::backend::drm::{DrmDevice, PlaneConfig, PlaneState};
    use smithay::utils::{Rectangle, Transform};
    use std::time::Duration;

    tracing::info!("prism — TTY gradient scanout (renderer pipeline), depth={depth:?}");

    let instance = prism_renderer::Instance::new()?;
    let device = prism_renderer::Device::new(
        instance.clone(),
        Some(prism_renderer::DrmDevId {
            major: 226,
            minor: 129,
        }),
    )?;
    tracing::info!("Vulkan device: {}", device.physical.name);

    let drm_path = "/dev/dri/card0";
    // One-shot subcommand: we hold the session + DRM only briefly (5s).
    // VT-switch and SIGINT may be blocked during that window because we
    // don't drain the libseat notifier (`_session_notifier`) — acceptable
    // for the diagnostic subcommands. Integrated `prism run` properly
    // drains both notifiers.
    let (mut session, _session_notifier) = prism_drm::SeatSession::new()?;
    if !session.is_active() {
        return Err(anyhow!(
            "libseat session not active. Switch to a free VT and rerun."
        ));
    }
    let drm_fd = session.open_drm(drm_path)?;
    let (mut drm, _drm_notifier) = DrmDevice::new(drm_fd, false)?;
    let pick = match output_name {
        Some(name) => scanout::pick_by_name(&drm, name)?,
        None => scanout::pick_first_connected(&drm)?,
    };
    tracing::info!(
        "scanout target: {} mode={}x{}@{}Hz",
        pick.connector_name,
        pick.mode.size().0,
        pick.mode.size().1,
        pick.mode.vrefresh(),
    );

    // Tell the connector to run the link at the depth we're scanning out.
    // Without this, a 10-bit framebuffer gets dithered down to 8 bits at
    // scanout — better than 8-bit-with-banding but still throws information
    // away. Most amdgpu connectors expose `max bpc` for DP and HDMI.
    match prism_drm::set_connector_max_bpc(&drm, pick.connector, depth.max_bpc()) {
        Ok(true) => tracing::info!("connector max bpc set to {}", depth.max_bpc()),
        Ok(false) => tracing::warn!(
            "connector doesn't expose `max bpc` property; scanout depth may be \
             driver-controlled (typically 8)"
        ),
        Err(e) => tracing::warn!("set max bpc failed: {e:#}"),
    }

    let surface = drm.create_surface(pick.crtc, pick.mode, &[pick.connector])?;
    let gbm = prism_drm::GbmDevice::from_device_fd(drm.device_fd().device_fd())?;

    let (w, h) = pick.mode.size();
    let w = w as u32;
    let h = h as u32;
    let fourcc = depth.drm_fourcc();
    let vk_format = vk_format_for_depth(depth);
    let (bo, dmabuf) = gbm.allocate_scanout(w, h, fourcc, &[DrmModifier::Linear])?;
    let scanout_image = prism_renderer::ImportedImage::import(
        device.clone(),
        &dmabuf,
        vk_format,
        vk::ImageUsageFlags::COLOR_ATTACHMENT,
    )?;
    tracing::info!(
        "scanout BO ready: {w}x{h} {:?} LINEAR (Vulkan {:?})",
        fourcc,
        vk_format
    );

    let texture = build_gradient_texture(device.clone(), 1024)?;
    // TTY gradient: fp32 intermediate + standard SDR encode. Per-output
    // EncodeConfig (FIR filter for the QD-OLED, calibration LUT per panel)
    // will come from the config layer once it exists.
    let encode_config = prism_renderer::EncodeConfig::default_srgb();
    let mut renderer = Renderer::new(
        device.clone(),
        vk_format,
        prism_renderer::DEFAULT_INTERMEDIATE_FORMAT,
        &encode_config,
    )?;

    let element = ElementDraw {
        texture_view: texture.view,
        chroma_view: None,
        push: DecodePush::identity_srgb([-1.0, -1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]),
    };
    let encode_push = EncodePush::sdr_identity();
    // One-shot TTY test: device_wait_idle below ensures the GPU work
    // committed by render_frame finishes before the page-flip; the
    // returned SYNC_FD is dropped (we don't use the IN_FENCE_FD path
    // here, the synchronous wait is simpler for a one-shot test).
    // Damage `&[]`: fresh Renderer → forced full first-frame paint anyway.
    let _present_sync =
        renderer.render_frame(&scanout_image, &[element], &[], &encode_push, &[])?;
    unsafe {
        let _ = device.raw.device_wait_idle();
    }
    tracing::info!("rendered gradient via decode→encode pipeline");

    let fb = scanout::add_framebuffer_for_bo(&drm, &bo)?;
    let src = Rectangle::from_size((w as i32, h as i32).into()).to_f64();
    let dst = Rectangle::from_size((w as i32, h as i32).into());
    let plane_state = [PlaneState {
        handle: surface.plane(),
        config: Some(PlaneConfig {
            src,
            dst,
            transform: Transform::Normal,
            alpha: 1.0,
            damage_clips: None,
            fb,
            fence: None,
        }),
    }];
    surface.commit(plane_state.iter().cloned(), true)?;
    tracing::info!("committed; holding 5s");
    std::thread::sleep(Duration::from_secs(5));
    surface.clear()?;
    tracing::info!("released");
    Ok(())
}

/// Light up the first connected display with a solid color for a few seconds,
/// then exit. Requires DRM master — must be run from a TTY where no Wayland /
/// X session owns the device. Run with:
///
///     sudo -E env "WAYLAND_DISPLAY=" "DISPLAY=" cargo run --bin prism -- scanout
///
/// (or simpler, after switching to a fresh VT with Ctrl+Alt+F3:
///   `./target/debug/prism scanout`, no sudo needed if you're in the `seat`
///    or `video` group and seatd/logind is running.)
fn run_scanout_smoke_test(output_name: Option<&str>) -> Result<()> {
    use prism_drm::scanout;
    use prism_renderer::{oneshot, ImportedImage, OneshotPool};
    use smithay::backend::drm::{DrmDevice, PlaneConfig, PlaneState};
    use smithay::utils::{Rectangle, Transform};

    tracing::info!("prism compositor — scanout smoke test (needs DRM master / TTY)");

    let instance = prism_renderer::Instance::new()?;
    // For the smoke test, render with the device that drives our scanout target.
    // Vega 20 (DP-4 / LU28R55) lives at primary 226:0, render 226:129. Path is
    // /dev/dri/card0 (primary node — needed for mode-set).
    let drm_path = "/dev/dri/card0";
    let device = prism_renderer::Device::new(
        instance.clone(),
        Some(prism_renderer::DrmDevId {
            major: 226,
            minor: 129,
        }),
    )?;
    tracing::info!("Vulkan device: {}", device.physical.name);

    // libseat session → DRM master.
    // One-shot subcommand: we hold the session + DRM only briefly (5s).
    // VT-switch and SIGINT may be blocked during that window because we
    // don't drain the libseat notifier (`_session_notifier`) — acceptable
    // for the diagnostic subcommands. Integrated `prism run` properly
    // drains both notifiers.
    let (mut session, _session_notifier) = prism_drm::SeatSession::new()?;
    if !session.is_active() {
        return Err(anyhow!(
            "libseat session not active. Switch to a free VT (Ctrl+Alt+F3) and rerun."
        ));
    }
    let drm_fd = session
        .open_drm(drm_path)
        .with_context(|| format!("open {drm_path} via libseat"))?;
    let (mut drm, _drm_notifier) =
        DrmDevice::new(drm_fd, false).with_context(|| format!("DrmDevice::new({drm_path})"))?;
    tracing::info!(
        "DRM atomic={} dev_id={:?}",
        drm.is_atomic(),
        drm.device_id()
    );

    // Pick a connected output: by name if specified, else the first one.
    let pick = match output_name {
        Some(name) => scanout::pick_by_name(&drm, name)?,
        None => scanout::pick_first_connected(&drm)?,
    };
    tracing::info!(
        "scanout target: {} mode={}x{}@{}Hz crtc={:?}",
        pick.connector_name,
        pick.mode.size().0,
        pick.mode.size().1,
        pick.mode.vrefresh(),
        pick.crtc,
    );

    // Create a surface (claims a primary plane, validates connector/crtc).
    let surface = drm
        .create_surface(pick.crtc, pick.mode, &[pick.connector])
        .context("DrmDevice::create_surface")?;
    tracing::info!(
        "DrmSurface ready, plane={:?} mode={:?}",
        surface.plane(),
        surface.current_mode().size(),
    );

    // GBM and DrmDevice MUST share the same fd: GEM handles are per-fd, so
    // BOs allocated through GBM on a different fd would be invisible to the
    // addfb2 ioctl called through the master fd (ENOENT). Pull the master
    // fd back out of the DrmDevice's DrmDeviceFd to share it.
    let gbm = prism_drm::GbmDevice::from_device_fd(drm.device_fd().device_fd())?;
    tracing::info!("GBM backend: {}", gbm.backend_name());

    let (w, h) = pick.mode.size();
    let w = w as u32;
    let h = h as u32;
    let (bo, dmabuf) = gbm
        .allocate_scanout(w, h, DrmFourcc::Xrgb8888, &[DrmModifier::Linear])
        .with_context(|| format!("GBM allocate {w}x{h} XRGB8888 LINEAR"))?;
    tracing::info!(
        "scanout BO: {}x{} modifier={:#x} stride={}",
        dmabuf.width,
        dmabuf.height,
        u64::from(dmabuf.modifier),
        dmabuf.planes[0].stride,
    );

    // Render solid green via Vulkan.
    let image = ImportedImage::import(
        device.clone(),
        &dmabuf,
        vk::Format::B8G8R8A8_UNORM,
        vk::ImageUsageFlags::TRANSFER_DST,
    )?;
    let pool = OneshotPool::new(device.clone())?;
    let color = vk::ClearColorValue {
        // Mid-green: r=0.0, g=0.5, b=0.0. Picked so HDR vs SDR processing is
        // visually distinguishable later (saturated primaries shift more).
        float32: [0.0, 0.5, 0.0, 1.0],
    };
    let vk_image = image.image();
    pool.record_and_submit(|raw, cb| {
        oneshot::record_clear_color(raw, cb, vk_image, color);
    })
    .context("clear-to-green submit")?;
    tracing::info!("Vulkan clear submitted");

    // Promote BO → DRM framebuffer handle.
    let fb = scanout::add_framebuffer_for_bo(&drm, &bo)
        .context("add_planar_framebuffer for scanout BO")?;
    tracing::info!("framebuffer handle: {:?}", fb);

    // Atomic commit: mode-set the surface to display this fb.
    let src = Rectangle::from_size((w as i32, h as i32).into()).to_f64();
    let dst = Rectangle::from_size((w as i32, h as i32).into());
    let plane_state = [PlaneState {
        handle: surface.plane(),
        config: Some(PlaneConfig {
            src,
            dst,
            transform: Transform::Normal,
            alpha: 1.0,
            damage_clips: None,
            fb,
            fence: None,
        }),
    }];

    surface
        .commit(plane_state.iter().cloned(), true)
        .context("DrmSurface::commit (mode-set)")?;
    tracing::info!("mode-set committed; holding output for 5 seconds…");

    std::thread::sleep(Duration::from_secs(5));

    // Clear out before releasing master so we don't leave the panel locked
    // to our framebuffer when the display session takes over again.
    surface.clear().context("DrmSurface::clear")?;
    tracing::info!("scanout cleared; releasing");

    Ok(())
}
