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
    // Wayland-only mode: no DRM session, one GPU available for dmabuf import
    // validation. Key the gpu map by drm_primary (or drm_render fallback).
    let mut gpus = std::collections::HashMap::new();
    let key = device
        .physical
        .drm_primary
        .or(device.physical.drm_render)
        .ok_or_else(|| anyhow!("Vulkan device has no DRM node id; cannot index"))?;
    gpus.insert(key, device);
    let mut state = prism_protocols::PrismState::new(&display, None, gpus);

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

/// End-to-end pipeline check: build a small linear gradient texture, run it
/// through decode→intermediate→encode (sRGB OETF), readback the BGRA bytes,
/// validate at anchor points. Catches:
///   - shader compile / SPIR-V loading regressions
///   - descriptor / pipeline layout mismatches
///   - dynamic-rendering attachment setup mistakes
///   - sRGB OETF math (compare to known curve values)
fn tracer_render_gradient(device: Arc<prism_renderer::Device>) -> Result<()> {
    use prism_renderer::{
        DecodePush, ElementDraw, EncodePush, ImportedImage, Renderer, vk,
    };

    let width: u32 = 256;
    let height: u32 = 1;

    // Scanout target: a GBM XRGB8888 LINEAR BO we can map for readback.
    let gbm = prism_drm::GbmDevice::open("/dev/dri/renderD129")?;
    let (bo, dmabuf) = gbm.allocate_scanout(
        width,
        height,
        DrmFourcc::Xrgb8888,
        &[DrmModifier::Linear],
    )?;
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
        push: DecodePush::identity_srgb(
            [-1.0, -1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ),
    };
    let encode_push = EncodePush::sdr_identity();

    renderer.render_frame(&scanout, &[element], &encode_push)?;
    // Headless readback wants to map the scanout BO; the GPU work above
    // was submitted with a fence we don't wait on, so make sure it's done
    // before we map. device_wait_idle is fine here (one-shot test).
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
            p0, pmid, p255
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
    let mem_type = pick_memory(&device, req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_type);
    let staging_mem = unsafe { device.raw.allocate_memory(&alloc, None) }?;
    unsafe { device.raw.bind_buffer_memory(staging, staging_mem, 0) }?;
    unsafe {
        let dst = device.raw.map_memory(staging_mem, 0, req.size, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst as *mut u8, bytes.len());
        device.raw.unmap_memory(staging_mem);
    }

    // Texture image: OPTIMAL, SAMPLED + TRANSFER_DST.
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R16G16B16A16_SFLOAT)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.raw.create_image(&image_info, None) }?;
    let req = unsafe { device.raw.get_image_memory_requirements(image) };
    let mem_type =
        pick_memory(&device, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
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
    let begin = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
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
        .image_extent(vk::Extent3D { width, height, depth: 1 })];
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
    Ok(GradientTexture { device, image, memory, view })
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
    use calloop::EventLoop;
    use calloop::signals::{Signal, Signals};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
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
    let mut gpus: std::collections::HashMap<
        prism_renderer::DrmDevId,
        Arc<prism_renderer::Device>,
    > = std::collections::HashMap::new();
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

    // For now every output uses the same OutputConfig (same depth, same
    // encode chain). Per-output config (from EDID + user config) is the
    // job of #59.x-later; today we let depth come from the CLI to keep
    // existing test flows working.
    let output_config = prism_drm::OutputConfig {
        depth,
        vk_format: vk_format_for_depth(depth),
        intermediate_format: prism_renderer::DEFAULT_INTERMEDIATE_FORMAT,
        encode_config: prism_renderer::EncodeConfig::default_srgb(),
    };

    // ── Pick connectors + bring up OutputContexts on every card ────────────
    // If OUTPUT specifies a connector name, search every card for it and
    // bring up only that one. Otherwise pick_all_connected on each card.
    let mut outputs: Vec<prism_drm::OutputContext> = Vec::new();
    for card in &mut cards {
        breadcrumb(&format!("bringup loop: entering card {}", card.path));
        let Some(device) = gpus.get(&card.drm_dev_id).cloned() else {
            tracing::warn!(
                "card {} has no GPU; skipping all its outputs",
                card.path
            );
            breadcrumb(&format!(
                "bringup loop: {} has no matching GPU, skipping",
                card.path
            ));
            continue;
        };
        breadcrumb(&format!("bringup loop: {} picking connectors", card.path));
        let picks: Vec<prism_drm::OutputPick> = match output_name {
            Some(name) => match prism_drm::pick_by_name(&card.drm, name) {
                Ok(p) => vec![p],
                Err(_) => Vec::new(), // OUTPUT might be on a different card
            },
            None => prism_drm::pick_all_connected(&card.drm).unwrap_or_default(),
        };
        breadcrumb(&format!(
            "bringup loop: {} got {} pick(s)",
            card.path,
            picks.len()
        ));
        for pick in picks {
            let name = pick.connector_name.clone();
            breadcrumb(&format!("bringup loop: building OutputContext for {name}"));
            match prism_drm::OutputContext::new(card, device.clone(), pick, &output_config) {
                Ok(output) => {
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
    breadcrumb(&format!("bringup loop: all cards done, {} outputs total", outputs.len()));
    if outputs.is_empty() {
        return Err(anyhow!("no outputs successfully brought up across any card"));
    }

    // ── Wayland display + PrismState ───────────────────────────────────────
    let display = prism_protocols::new_display()?;
    let mut state = prism_protocols::PrismState::new(&display, Some(session), gpus);
    for card in cards {
        state.attach_card(card);
    }
    for output in outputs {
        state.attach_output(output);
    }
    breadcrumb(&format!(
        "wayland state up; {} card(s) + {} output(s) attached",
        state.cards.len(),
        state.outputs.len()
    ));

    // Event loop + sources.
    let mut event_loop: EventLoop<'static, prism_protocols::PrismState> =
        EventLoop::try_new().context("EventLoop::try_new")?;
    let socket = prism_protocols::insert_wayland_sources(&event_loop.handle(), display)?;
    tracing::info!("WAYLAND_DISPLAY={socket}");
    for output in state.outputs.values() {
        tracing::info!(
            "scanout target: {} {}×{} (crtc {:?})",
            output.connector_name, output.extent.width, output.extent.height, output.crtc
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
            breadcrumb(&format!("MAX_RUNTIME: {secs}s elapsed, requesting clean exit"));
            running.store(false, Ordering::SeqCst);
        });
    }

    // DRM vblank handler — one per card (each card's DrmDeviceNotifier
    // fires only for its own CRTCs). Marks the just-flipped output's
    // page-flip done AND triggers its next render. This is the heartbeat
    // of vblank-driven pacing — present → wait for vblank → present again.
    // Bootstrapped below with one explicit present per output that does
    // the mode-set commit; every subsequent frame is kicked off by a
    // vblank from the prior flip.
    let max_frames_copy = max_frames;
    for drm_notifier in drm_notifiers.drain(..) {
        let running_for_vblank = running.clone();
        let frame_counter_for_vblank = frame_counter.clone();
        event_loop
            .handle()
            .insert_source(drm_notifier, move |event, _metadata, state| {
                use smithay::backend::drm::DrmEvent;
                match event {
                    DrmEvent::VBlank(crtc) => {
                        let Some(output) = state.output_for_crtc(crtc) else {
                            breadcrumb(&format!("vblank for unknown crtc {crtc:?}"));
                            return;
                        };
                        output.mark_vblank();
                        let n = frame_counter_for_vblank.fetch_add(1, Ordering::SeqCst) + 1;
                        let t0 = std::time::Instant::now();
                        let result = present_for_crtc(state, crtc);
                        let dt_us = t0.elapsed().as_micros();
                        match result {
                            Ok(true) => breadcrumb(&format!(
                                "frame #{n}: vblank({crtc:?}) → present submitted in {dt_us}µs"
                            )),
                            Ok(false) => {
                                breadcrumb(&format!(
                                    "frame #{n}: skipped after {dt_us}µs (still pending)"
                                ));
                            }
                            Err(e) => {
                                breadcrumb(&format!(
                                    "frame #{n}: ERROR after {dt_us}µs: {e:#}"
                                ));
                                tracing::warn!("present failed: {e:#}");
                            }
                        }
                        if let Some(max) = max_frames_copy {
                            if n >= max {
                                breadcrumb(&format!(
                                    "frame #{n}: max_frames reached, exit"
                                ));
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

    // SIGINT / SIGTERM → clean shutdown.
    {
        let running = running.clone();
        let signals = Signals::new(&[Signal::SIGINT, Signal::SIGTERM])
            .context("Signals::new")?;
        event_loop
            .handle()
            .insert_source(signals, move |evt, _, _state| {
                tracing::info!(signal = ?evt.signal(), "shutting down");
                running.store(false, Ordering::SeqCst);
            })
            .map_err(|e| anyhow!("insert signals source: {e}"))?;
    }

    // Bootstrap: render frame #1 on every attached output explicitly so
    // the kernel has a vblank to schedule. From here on, each output's
    // DRM vblank handler triggers its next present — true vblank-driven
    // pacing at the panel's refresh rate. Before any client connects, the
    // element list is empty: scanout shows a black frame, but the page-flip
    // cycle is up and running so a later commit lands in the next vblank.
    let bootstrap_crtcs: Vec<_> = state.outputs.values().map(|o| o.crtc).collect();
    for crtc in bootstrap_crtcs {
        let n0 = frame_counter.fetch_add(1, Ordering::SeqCst) + 1;
        breadcrumb(&format!("bootstrap({crtc:?}) → render frame #{n0}"));
        match present_for_crtc(&mut state, crtc) {
            Ok(true) => breadcrumb(&format!("frame #{n0}: submitted (mode-set commit)")),
            Ok(false) => {
                breadcrumb(&format!("frame #{n0}: skipped (unexpected at bootstrap)"))
            }
            Err(e) => {
                breadcrumb(&format!("frame #{n0}: ERROR {e:#}"));
                return Err(e).context("bootstrap present");
            }
        }
    }

    breadcrumb("entering dispatch loop");
    while running.load(Ordering::SeqCst) {
        event_loop
            .dispatch(Some(Duration::from_millis(100)), &mut state)
            .context("event_loop.dispatch")?;
        state
            .display_handle
            .flush_clients()
            .context("flush_clients")?;
    }

    breadcrumb("dispatch loop exited cleanly");
    tracing::info!("integrated loop stopped");

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
        "shutdown: outputs={} cards={} gpus={} dmabuf_textures={}",
        state.outputs.len(),
        state.cards.len(),
        state.gpus.len(),
        state.dmabuf_textures.len()
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
    state.dmabuf_textures.clear();
    breadcrumb(&format!(
        "shutdown: cleared dmabuf_textures in {}ms",
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
/// Layout (provisional, until #59.5): every output sees every surface;
/// first mapped toplevel covers the whole output.
fn present_for_crtc(
    state: &mut prism_protocols::PrismState,
    crtc: smithay::reexports::drm::control::crtc::Handle,
) -> Result<bool> {
    use prism_renderer::{DecodePush, ElementDraw, EncodePush};
    use smithay::wayland::compositor::with_states;

    // Find the output for this CRTC and capture its GPU id — we need it
    // to pick the right per-GPU texture import below. (We re-borrow the
    // output mutably at present() time.)
    let output_gpu_id = state
        .output_for_crtc(crtc)
        .ok_or_else(|| anyhow!("no output bound to crtc {crtc:?}"))?
        .gpu_id;

    // Snapshot the toplevel surface handles so we can drop the borrow on
    // state.xdg_shell before mutably borrowing state.outputs for present().
    let surfaces: Vec<_> = state
        .xdg_shell
        .toplevel_surfaces()
        .iter()
        .map(|t| t.wl_surface().clone())
        .collect();

    let mut elements: Vec<ElementDraw> = Vec::new();
    for surface in &surfaces {
        let view_opt = with_states(surface, |states| {
            states
                .data_map
                .get::<prism_protocols::SurfaceTexSlot>()
                .and_then(|s| s.0.lock().unwrap().as_ref()
                    .and_then(|t| t.view_for(output_gpu_id)))
        });
        if let Some(view) = view_opt {
            elements.push(ElementDraw {
                texture_view: view,
                push: DecodePush::identity_srgb(
                    [-1.0, -1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                ),
            });
            // Trivial layout: only the first mapped toplevel goes through.
            break;
        }
        // view_for can still return None if a per-GPU dmabuf import failed
        // on this GPU; in that case we silently skip this surface here.
    }

    let encode_push = EncodePush::sdr_identity();
    let output = state
        .output_for_crtc(crtc)
        .ok_or_else(|| anyhow!("no output bound to crtc {crtc:?}"))?;
    output.present(&elements, &encode_push)
}

/// Same TTY-required mode-set as `scanout`, but instead of `vkCmdClearColorImage`
/// the scanout image is rendered through the two-pass decode→encode pipeline
/// using a horizontal-gradient texture. Visual verification: a smoothly
/// gamma-correct gradient (black on the left → white on the right).
fn run_gradient_scanout(
    output_name: Option<&str>,
    depth: prism_drm::ScanoutDepth,
) -> Result<()> {
    use prism_drm::scanout;
    use prism_renderer::{DecodePush, ElementDraw, EncodePush, Renderer, vk};
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
        fourcc, vk_format
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
        push: DecodePush::identity_srgb([-1.0, -1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]),
    };
    let encode_push = EncodePush::sdr_identity();
    renderer.render_frame(&scanout_image, &[element], &encode_push)?;
    // One-shot TTY test: page-flip below needs the render to be done.
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
    let (mut drm, _drm_notifier) = DrmDevice::new(drm_fd, false)
        .with_context(|| format!("DrmDevice::new({drm_path})"))?;
    tracing::info!("DRM atomic={} dev_id={:?}", drm.is_atomic(), drm.device_id());

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
