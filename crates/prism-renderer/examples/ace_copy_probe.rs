//! Headless probe: does radv accept `vkCmdCopyImage` (LINEAR host-visible →
//! OPTIMAL device-local, i.e. a detiling copy) on a **family-1 queue**
//! (COMPUTE+TRANSFER, the async-compute engines), and does it produce correct
//! pixels?
//!
//! This is the one open risk gating the Phase 1 async-render rework
//! (`docs/async-render-rework.md`): the plan runs the cross-GPU mirror's
//! GTT→local copy on an ACE queue. The Vulkan spec allows transfer commands on
//! any TRANSFER-capable queue, but we want empirical confirmation from radv —
//! not spec-reading — before building the copier on it.
//!
//! Runs on EVERY physical device. For each: picks a queue family that is
//! COMPUTE+TRANSFER but NOT GRAPHICS (falls back to reporting if none),
//! creates a 64×64 RGBA8 LINEAR host image filled with a known pattern, copies
//! it into an OPTIMAL device image on the ACE queue, copies that back into a
//! LINEAR host image on the same queue, and verifies the round-trip pixels.
//! Validation layer is enabled, so any illegal queue/stage/access usage prints
//! to stderr.
//!
//! Run: `cargo run -p prism-renderer --example ace_copy_probe`

use ash::vk;
use std::ffi::{c_char, CStr};

const W: u32 = 64;
const H: u32 = 64;

fn main() {
    let entry = unsafe { ash::Entry::load() }.expect("load Vulkan");

    // Enable the validation layer if present.
    let want_layer = c"VK_LAYER_KHRONOS_validation";
    let have_layer = unsafe { entry.enumerate_instance_layer_properties() }
        .unwrap_or_default()
        .iter()
        .any(|l| {
            let name = unsafe { CStr::from_ptr(l.layer_name.as_ptr()) };
            name == want_layer
        });
    let layers: Vec<*const c_char> = if have_layer {
        vec![want_layer.as_ptr()]
    } else {
        eprintln!("(validation layer not found — running without)");
        vec![]
    };

    let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_layer_names(&layers),
            None,
        )
    }
    .expect("create_instance");

    let phys = unsafe { instance.enumerate_physical_devices() }.expect("enumerate devices");
    println!("Found {} physical device(s)\n", phys.len());

    for pd in phys {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        println!("=== {name} ===");
        match probe_device(&instance, pd) {
            Ok(msg) => println!("  RESULT: {msg}\n"),
            Err(e) => println!("  RESULT: FAIL — {e}\n"),
        }
    }

    unsafe { instance.destroy_instance(None) };
}

fn probe_device(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Result<String, String> {
    let qfp = unsafe { instance.get_physical_device_queue_family_properties(pd) };

    // Family 1 target: COMPUTE+TRANSFER, NOT GRAPHICS (the ACE queues).
    let ace = qfp.iter().enumerate().find(|(_, p)| {
        p.queue_flags
            .contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER)
            && !p.queue_flags.contains(vk::QueueFlags::GRAPHICS)
    });
    let (qfi, qprops) = match ace {
        Some((i, p)) => (i as u32, p),
        None => return Err("no COMPUTE+TRANSFER-without-GRAPHICS queue family".into()),
    };
    println!(
        "  using queue family {qfi} (count={}, flags={:?})",
        qprops.queue_count, qprops.queue_flags
    );

    let prio = [1.0_f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(qfi)
        .queue_priorities(&prio)];
    let device = unsafe {
        instance.create_device(
            pd,
            &vk::DeviceCreateInfo::default().queue_create_infos(&queue_info),
            None,
        )
    }
    .map_err(|e| format!("create_device: {e}"))?;

    let result = (|| {
        let queue = unsafe { device.get_device_queue(qfi, 0) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(pd) };

        let fmt = vk::Format::R8G8B8A8_UNORM;
        let extent = vk::Extent3D {
            width: W,
            height: H,
            depth: 1,
        };

        // --- src: LINEAR, host-visible, PREINITIALIZED so we can fill it. ---
        let src = create_image(
            &device,
            fmt,
            extent,
            vk::ImageTiling::LINEAR,
            vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageLayout::PREINITIALIZED,
        )?;
        let src_mem = bind_image_memory(
            &device,
            &mem_props,
            src,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // --- dst: OPTIMAL, device-local (the "local tiled" target). ---
        let dst = create_image(
            &device,
            fmt,
            extent,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED,
            vk::ImageLayout::UNDEFINED,
        )?;
        let _dst_mem = bind_image_memory(
            &device,
            &mem_props,
            dst,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // --- readback: LINEAR, host-visible, to verify the round trip. ---
        let rb = create_image(
            &device,
            fmt,
            extent,
            vk::ImageTiling::LINEAR,
            vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageLayout::UNDEFINED,
        )?;
        let rb_mem = bind_image_memory(
            &device,
            &mem_props,
            rb,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Fill src via its mapped LINEAR layout (respect rowPitch). Pattern:
        // r = x, g = y, b = 0x5a, a = 0xff.
        let layout = unsafe {
            device.get_image_subresource_layout(
                src,
                vk::ImageSubresource {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    array_layer: 0,
                },
            )
        };
        unsafe {
            let ptr = device
                .map_memory(src_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("map src: {e}"))? as *mut u8;
            for y in 0..H {
                for x in 0..W {
                    let off = layout.offset as usize
                        + y as usize * layout.row_pitch as usize
                        + x as usize * 4;
                    *ptr.add(off) = x as u8;
                    *ptr.add(off + 1) = y as u8;
                    *ptr.add(off + 2) = 0x5a;
                    *ptr.add(off + 3) = 0xff;
                }
            }
            device.unmap_memory(src_mem);
        }

        // --- command buffer on the ACE queue family ---
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(qfi),
                None,
            )
        }
        .map_err(|e| format!("create_command_pool: {e}"))?;
        let cb = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| format!("allocate_command_buffers: {e}"))?[0];

        unsafe {
            device
                .begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| format!("begin: {e}"))?;

            // src: PREINITIALIZED -> TRANSFER_SRC_OPTIMAL (preserve contents).
            // No HOST_WRITE src-access: host writes before a queue submit are
            // automatically visible to the device (and HOST_* access is illegal
            // on a compute/transfer queue — that's part of what this probe
            // established).
            barrier(
                &device,
                cb,
                src,
                vk::ImageLayout::PREINITIALIZED,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_READ,
            );
            // dst: UNDEFINED -> TRANSFER_DST_OPTIMAL.
            barrier(
                &device,
                cb,
                dst,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            );

            // THE OP UNDER TEST: LINEAR src -> OPTIMAL dst (detiling copy) on
            // the ACE queue.
            let region = [vk::ImageCopy::default()
                .src_subresource(color_layers())
                .dst_subresource(color_layers())
                .extent(extent)];
            device.cmd_copy_image(
                cb,
                src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &region,
            );

            // dst -> TRANSFER_SRC, rb -> TRANSFER_DST, copy back.
            barrier(
                &device,
                cb,
                dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            );
            barrier(
                &device,
                cb,
                rb,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            );
            device.cmd_copy_image(
                cb,
                dst,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                rb,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &region,
            );
            // rb -> GENERAL for host read. No HOST_READ dst-access: the fence
            // wait below makes the transfer writes available, and coherent
            // memory makes them visible to the host without a host-stage
            // barrier (which is unavailable on this queue anyway).
            barrier(
                &device,
                cb,
                rb,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::empty(),
            );

            device
                .end_command_buffer(cb)
                .map_err(|e| format!("end: {e}"))?;
        }

        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| format!("create_fence: {e}"))?;
        let cbs = [cb];
        let submit = [vk::SubmitInfo::default().command_buffers(&cbs)];
        unsafe {
            device
                .queue_submit(queue, &submit, fence)
                .map_err(|e| format!("queue_submit (ACE): {e}"))?;
            device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| format!("wait_for_fences: {e}"))?;
        }

        // Verify the round-trip pixels.
        let rb_layout = unsafe {
            device.get_image_subresource_layout(
                rb,
                vk::ImageSubresource {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    array_layer: 0,
                },
            )
        };
        let mut bad = 0u32;
        let mut first_bad = String::new();
        unsafe {
            let ptr = device
                .map_memory(rb_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("map rb: {e}"))? as *const u8;
            for y in 0..H {
                for x in 0..W {
                    let off = rb_layout.offset as usize
                        + y as usize * rb_layout.row_pitch as usize
                        + x as usize * 4;
                    let got = [
                        *ptr.add(off),
                        *ptr.add(off + 1),
                        *ptr.add(off + 2),
                        *ptr.add(off + 3),
                    ];
                    let want = [x as u8, y as u8, 0x5a, 0xff];
                    if got != want {
                        if bad == 0 {
                            first_bad = format!("at ({x},{y}) got {got:?} want {want:?}");
                        }
                        bad += 1;
                    }
                }
            }
            device.unmap_memory(rb_mem);
        }

        // Cleanup.
        unsafe {
            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
            device.destroy_image(src, None);
            device.free_memory(src_mem, None);
            device.destroy_image(dst, None);
            device.free_memory(_dst_mem, None);
            device.destroy_image(rb, None);
            device.free_memory(rb_mem, None);
        }

        if bad == 0 {
            Ok(format!(
                "PASS — image copy accepted on family {qfi}, {}×{} round-trip pixels correct",
                W, H
            ))
        } else {
            Err(format!(
                "copy ran but {bad}/{} pixels wrong (first: {first_bad})",
                W * H
            ))
        }
    })();

    unsafe { device.destroy_device(None) };
    result
}

fn color_layers() -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1)
}

fn create_image(
    device: &ash::Device,
    format: vk::Format,
    extent: vk::Extent3D,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    initial: vk::ImageLayout,
) -> Result<vk::Image, String> {
    unsafe {
        device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(tiling)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(initial),
            None,
        )
    }
    .map_err(|e| format!("create_image: {e}"))
}

fn bind_image_memory(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    image: vk::Image,
    flags: vk::MemoryPropertyFlags,
) -> Result<vk::DeviceMemory, String> {
    let req = unsafe { device.get_image_memory_requirements(image) };
    let mem_type = (0..mem_props.memory_type_count)
        .find(|&i| {
            (req.memory_type_bits & (1 << i)) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(flags)
        })
        .ok_or_else(|| format!("no memory type for {flags:?}"))?;
    let mem = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type),
            None,
        )
    }
    .map_err(|e| format!("allocate_memory: {e}"))?;
    unsafe { device.bind_image_memory(image, mem, 0) }.map_err(|e| format!("bind: {e}"))?;
    Ok(mem)
}

#[allow(clippy::too_many_arguments)]
fn barrier(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) {
    let b = [vk::ImageMemoryBarrier::default()
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })];
    unsafe {
        device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &b,
        );
    }
}
