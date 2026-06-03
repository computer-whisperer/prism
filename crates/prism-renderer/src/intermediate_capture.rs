//! On-demand readback of a per-output BT.2020 intermediate frame into a
//! memfd, for the prism-tune frame inspector.
//!
//! The intermediate is the absolute-nits BT.2020 *linear* composite the
//! encode pass reads — before the LUT / response-curve / OETF that bake
//! in panel correction. Between frames it sits in
//! `SHADER_READ_ONLY_OPTIMAL` (last frame's content, persistent). We
//! transition it to `TRANSFER_SRC`, copy the whole image to a host
//! buffer, transition it back to the layout the render loop expects,
//! then write the raw `RGBA32F` texels into a fresh memfd and hand the
//! fd up to the IPC layer (passed to the client via `SCM_RIGHTS`).
//!
//! Synchronous — [`OneshotPool`] waits on the queue. This runs on the
//! main calloop thread in response to an explicit "fetch frame" IPC, so
//! a single ~one-frame hitch on demand is acceptable; it is NOT on the
//! per-frame render path. The capture is whole-output, full-resolution,
//! and uncompressed: a 4K frame is `3840*2160*16` ≈ 127 MiB of shared
//! memory, freed once both ends drop the fd. fp16 (halving that) is a
//! straightforward future add if transport size ever bites.

use std::fs::File;
use std::io::Write;
use std::os::fd::OwnedFd;
use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::diagnose::create_host_buffer;
use crate::error::{RendererError, Result};
use crate::oneshot::OneshotPool;

/// A captured intermediate frame. The memfd holds `height` rows of
/// `width` `RGBA32F` texels, tightly packed (`stride_bytes = width * 16`),
/// in BT.2020 absolute-nits linear.
pub struct CapturedFrame {
    pub fd: OwnedFd,
    pub width: u32,
    pub height: u32,
    pub stride_bytes: u32,
    pub byte_len: u64,
}

/// Copy `image` (the output's intermediate, expected `RGBA32F` and in
/// `SHADER_READ_ONLY_OPTIMAL`) to host memory and write the raw texels
/// into a fresh memfd, returning the fd + geometry. See module docs.
pub fn capture_intermediate_to_memfd(
    device: &Arc<Device>,
    image: vk::Image,
    extent: vk::Extent2D,
    format: vk::Format,
) -> Result<CapturedFrame> {
    // Only the default fp32 intermediate today; fp16 would be a small
    // add (convert on the host-copy below) when transport size matters.
    if format != vk::Format::R32G32B32A32_SFLOAT {
        return Err(RendererError::MissingFeature(
            "capture_intermediate: only R32G32B32A32_SFLOAT intermediate supported",
        ));
    }
    if extent.width == 0 || extent.height == 0 {
        return Err(RendererError::Io(
            "capture_intermediate: empty extent".into(),
        ));
    }
    let texels = extent.width as u64 * extent.height as u64;
    let byte_len = texels * 16;

    // Host-visible readback target — lives only for this capture.
    let (buffer, memory, ptr, _size) =
        create_host_buffer(device, byte_len, vk::BufferUsageFlags::TRANSFER_DST)?;

    let result = (|| -> Result<OwnedFd> {
        let oneshot = OneshotPool::new(device.clone())?;
        oneshot.record_and_submit(|raw, cb| unsafe {
            let to_src = [barrier(
                image,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::PipelineStageFlags2::ALL_TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            )];
            raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&to_src),
            );

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
                .image_extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })];
            raw.cmd_copy_image_to_buffer(
                cb,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer,
                &region,
            );

            // Restore the layout the render loop's next encode expects.
            let back = [barrier(
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::ALL_TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
            )];
            raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&back),
            );
        })?;

        // Copy the readback bytes into a fresh memfd. SAFETY:
        // host-coherent memory, just queue-waited; `byte_len` matches
        // the allocation size passed to `create_host_buffer`.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, byte_len as usize) };
        let memfd = rustix::fs::memfd_create("prism-frame", rustix::fs::MemfdFlags::CLOEXEC)
            .map_err(|e| RendererError::Io(format!("memfd_create: {e}")))?;
        let mut file = File::from(memfd);
        file.write_all(bytes)
            .map_err(|e| RendererError::Io(format!("write frame to memfd: {e}")))?;
        Ok(OwnedFd::from(file))
    })();

    // Tear down the transient readback buffer regardless of outcome.
    unsafe {
        device.raw.unmap_memory(memory);
        device.raw.destroy_buffer(buffer, None);
        device.raw.free_memory(memory, None);
    }

    let fd = result?;
    Ok(CapturedFrame {
        fd,
        width: extent.width,
        height: extent.height,
        stride_bytes: extent.width * 16,
        byte_len,
    })
}

#[allow(clippy::too_many_arguments)]
fn barrier(
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
}
