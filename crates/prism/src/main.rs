use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use prism_frame::{DrmFourcc, DrmModifier};
use prism_renderer::vk;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("prism=info,vulkan=info")),
        )
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => run_headless_smoke_tests(),
        Some("scanout") => run_scanout_smoke_test(),
        Some("wayland") => run_wayland_server(),
        Some(other) => Err(anyhow!(
            "unknown subcommand {other:?}; expected: (no args) | scanout | wayland"
        )),
    }
}

/// Bring up a Wayland server socket and dispatch protocol messages forever.
/// Clients can connect via `WAYLAND_DISPLAY=wayland-N`. No rendering yet —
/// surface lifecycle / configure / commit are logged, buffers are dropped.
fn run_wayland_server() -> Result<()> {
    use calloop::EventLoop;
    use calloop::signals::{Signal, Signals};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
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
    let mut state = prism_protocols::PrismState::new(&display, device);

    let mut event_loop: EventLoop<'static, prism_protocols::PrismState> =
        EventLoop::try_new().context("calloop EventLoop::try_new")?;

    let socket = prism_protocols::insert_wayland_sources(&event_loop.handle(), display)?;
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
                        .map(|m| {
                            format!("{}x{}@{}Hz", m.size().0, m.size().1, m.vrefresh())
                        })
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
    tracer_dmabuf_protocol(device).context("dmabuf protocol-handler import path")?;

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
    use prism_renderer::{ImportedImage, OneshotPool, oneshot};

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
        probe.0, probe.1, probe.2, probe.3
    );

    if probe.0 == 0xff && probe.1 == 0x00 && probe.2 == 0xff {
        tracing::info!("✓ GBM → Vulkan → clear → readback verified (magenta)");
    } else {
        return Err(anyhow!(
            "readback mismatch: expected B=ff G=00 R=ff, got B={:#04x} G={:#04x} R={:#04x}",
            probe.0, probe.1, probe.2
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
    let prism_dmabuf = prism_frame::Dmabuf::from_smithay(&smithay_dmabuf)
        .context("Dmabuf::from_smithay")?;
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

/// Light up the first connected display with a solid color for a few seconds,
/// then exit. Requires DRM master — must be run from a TTY where no Wayland /
/// X session owns the device. Run with:
///
///     sudo -E env "WAYLAND_DISPLAY=" "DISPLAY=" cargo run --bin prism -- scanout
///
/// (or simpler, after switching to a fresh VT with Ctrl+Alt+F3:
///   `./target/debug/prism scanout`, no sudo needed if you're in the `seat`
///    or `video` group and seatd/logind is running.)
fn run_scanout_smoke_test() -> Result<()> {
    use prism_drm::scanout;
    use prism_renderer::{ImportedImage, OneshotPool, oneshot};
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
    let mut session = prism_drm::SeatSession::new()?;
    if !session.is_active() {
        return Err(anyhow!(
            "libseat session not active. Switch to a free VT (Ctrl+Alt+F3) and rerun."
        ));
    }
    let drm_fd = session
        .open_drm(drm_path)
        .with_context(|| format!("open {drm_path} via libseat"))?;
    let (mut drm, _drm_notifier) = DrmDevice::new(drm_fd, false)
        .with_context(|| format!("DrmDevice::new({drm_path})"))?;
    tracing::info!("DRM atomic={} dev_id={:?}", drm.is_atomic(), drm.device_id());

    // Pick the first connected output.
    let pick = scanout::pick_first_connected(&drm)?;
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
