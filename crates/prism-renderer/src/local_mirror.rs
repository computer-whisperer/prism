//! Target-local mirror image — the OPTIMAL, device-local copy of a cross-GPU
//! mirror's LINEAR GTT import.
//!
//! Background (see `docs/async-render-rework.md`): a mirrored surface is held
//! in a LINEAR, host-visible (GTT) scratch the home GPU fills each commit; the
//! target GPU imports it ([`crate::dmabuf::ImportedImage`]) and historically
//! **sampled that GTT import directly** in the decode/deband passes — two
//! untiled PCIe scans per frame. Instead we copy the GTT import **once** into a
//! [`LocalImage`] in the target's own VRAM (OPTIMAL-tiled) at the start of the
//! render command buffer, and the passes sample that. One streaming copy
//! replaces two scattered remote scans, and the tiled local image restores
//! texture-cache locality.
//!
//! The copy runs on the **transfer (ACE) queue** ([`AceCopier`], rework
//! increment C) so it overlaps graphics work instead of serializing inside
//! the render command buffer. The render submit waits the copier's
//! `local_done` semaphore; the exported `ace_done` sync-file feeds the
//! cross-GPU scratch-overwrite gate (the next home→GTT copy waits it).

use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Arc;

use ash::khr::external_semaphore_fd;
use ash::vk;

use crate::device::Device;
use crate::error::{RendererError, Result, VkResultExt};
use crate::intermediate::{create_view, pick_device_local_memory};

/// An OPTIMAL, device-local image used as the local destination of a mirror's
/// GTT→VRAM copy. Sampled by the decode/deband passes in place of the LINEAR
/// import. Created `CONCURRENT` across the graphics + transfer queue families
/// when they differ (so a future ACE-queue copy needs no ownership transfer);
/// `EXCLUSIVE` when the GPU exposes a single family.
pub struct LocalImage {
    device: Arc<Device>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    extent: vk::Extent2D,
}

impl LocalImage {
    pub fn new(device: Arc<Device>, extent: vk::Extent2D, format: vk::Format) -> Result<Self> {
        let gfx = device.physical.graphics_queue_family;
        let xfer = device.transfer_queue_family;
        let families: Vec<u32> = if gfx == xfer {
            vec![gfx]
        } else {
            vec![gfx, xfer]
        };
        let sharing = if families.len() > 1 {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(sharing)
            .queue_family_indices(&families)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.raw.create_image(&info, None) }
            .vk_ctx("create_image (local mirror)")?;

        let req = unsafe { device.raw.get_image_memory_requirements(image) };
        let mem_type = pick_device_local_memory(&device, req.memory_type_bits)?;
        let memory = unsafe {
            device.raw.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mem_type),
                None,
            )
        }
        .vk_ctx("allocate_memory (local mirror)")?;
        unsafe { device.raw.bind_image_memory(image, memory, 0) }
            .vk_ctx("bind_image_memory (local mirror)")?;

        let view = create_view(&device, image, format)?;
        Ok(Self {
            device,
            image,
            memory,
            view,
            extent,
        })
    }

    pub fn image(&self) -> vk::Image {
        self.image
    }
    pub fn view(&self) -> vk::ImageView {
        self.view
    }
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }
}

impl Drop for LocalImage {
    fn drop(&mut self) {
        // May be sampled by up to FRAMES_IN_FLIGHT in-flight submissions when a
        // mirror is torn down (surface destroyed / re-imported). Retire so the
        // deferred queue frees the handles once the slot fences prove those
        // submissions complete — same contract as `ImportedImage::drop`.
        self.device.retire(crate::device::Retired::Image {
            image: self.image,
            view: self.view,
            memory: self.memory,
        });
    }
}

/// One LINEAR(GTT)→OPTIMAL(local) image copy, recorded on the transfer
/// (ACE) queue by [`AceCopier::submit`]. Plain Vulkan handles, so the
/// renderer needs no knowledge of the mirror bookkeeping. The ACE submit's
/// wait semaphores (the home→GTT copy-done) gate the batch, so the copy
/// never reads a half-written GTT scratch.
#[derive(Clone, Copy)]
pub struct LocalMirrorCopy {
    /// LINEAR GTT import (the mirror `target`), in `GENERAL` layout.
    pub src: vk::Image,
    /// OPTIMAL device-local destination (the mirror `target_local` back
    /// buffer this frame writes + samples). Fully overwritten, so its prior
    /// contents/layout are irrelevant (taken as `UNDEFINED`).
    pub dst: vk::Image,
    pub extent: vk::Extent2D,
}

/// Per-renderer submitter for the GTT→local mirror copies, on the device's
/// transfer (ACE) queue so the copies overlap graphics
/// (`docs/async-render-rework.md` §2.8 C). One reusable command buffer +
/// fence (the reuse gate — steady state it's long signalled), a `local_done`
/// binary semaphore the render submit waits on, and an exportable `ace_done`
/// semaphore whose SYNC_FD feeds the cross-GPU scratch-overwrite gate.
///
/// **Semaphore discipline.** `local_done` is signalled by every ACE submit
/// and must be waited exactly once (by the frame's render submit) before
/// the next signal. If the render submit never happens (an error between
/// the ACE submit and the graphics submit), the caller invokes
/// [`Self::abort_unconsumed`]: it waits the copier fence (severing the
/// broken transitive deferred-destroy proof — see `Device::submit_transfer`)
/// and flags `local_done` for recreation at the next use.
pub struct AceCopier {
    device: Arc<Device>,
    pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    local_done: vk::Semaphore,
    export_sem: vk::Semaphore,
    sem_fd_loader: external_semaphore_fd::Device,
    /// True between an ACE submit and the graphics submit that waits its
    /// `local_done` — see the struct docs.
    local_done_unconsumed: bool,
    /// True when a SYNC_FD export failed after a submit signalled
    /// `export_sem` — the semaphore is still signalled (only a successful
    /// export unsignals it) and must be recreated before the next submit.
    export_sem_stale: bool,
}

impl AceCopier {
    /// Submit-free construction (device-loss recovery builds replacement
    /// renderers while the poisoned device's fd is still open).
    pub fn new(device: Arc<Device>) -> Result<Self> {
        let pool = unsafe {
            device.raw.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(device.transfer_queue_family),
                None,
            )
        }
        .vk_ctx("create_command_pool (ace copier)")?;
        let cb = unsafe {
            device.raw.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .vk_ctx("allocate_command_buffers (ace copier)")?[0];
        let fence = unsafe {
            device.raw.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }
        .vk_ctx("create_fence (ace copier)")?;
        let local_done = unsafe {
            device
                .raw
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        }
        .vk_ctx("create_semaphore (ace local_done)")?;
        let mut export_info = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let export_sem = unsafe {
            device.raw.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut export_info),
                None,
            )
        }
        .vk_ctx("create_semaphore (ace_done export)")?;
        let sem_fd_loader = external_semaphore_fd::Device::new(device.instance_raw(), &device.raw);
        Ok(Self {
            device,
            pool,
            cb,
            fence,
            local_done,
            export_sem,
            sem_fd_loader,
            local_done_unconsumed: false,
            export_sem_stale: false,
        })
    }

    /// Record + submit `copies` on the transfer queue, waiting `waits` (the
    /// imported home→GTT copy-done semaphores) and signalling both
    /// `local_done` (returned — the render submit waits it) and the
    /// exportable `ace_done` (returned as a SYNC_FD for the scratch gate).
    pub fn submit(
        &mut self,
        copies: &[LocalMirrorCopy],
        waits: &[vk::Semaphore],
    ) -> Result<(vk::Semaphore, OwnedFd)> {
        let device = &self.device.raw;
        unsafe {
            // Reuse gate: the previous ACE batch must have finished before
            // its command buffer / semaphores are recycled. Long done in
            // steady state (the copy is small and a frame old).
            device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .vk_ctx("wait_for_fences (ace copier reuse gate)")?;
            device
                .reset_fences(&[self.fence])
                .vk_ctx("reset_fences (ace copier)")?;
            // A prior signal that no render submit consumed leaves a binary
            // semaphore signalled — signalling it again is invalid. Replace
            // it (cheap device-level op; error paths only).
            if self.local_done_unconsumed {
                device.destroy_semaphore(self.local_done, None);
                self.local_done = device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .vk_ctx("recreate_semaphore (ace local_done)")?;
                self.local_done_unconsumed = false;
            }
            // Same for a signalled export semaphore whose SYNC_FD export
            // failed (a successful export is what unsignals it).
            if self.export_sem_stale {
                device.destroy_semaphore(self.export_sem, None);
                let mut export_info = vk::ExportSemaphoreCreateInfo::default()
                    .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
                self.export_sem = device
                    .create_semaphore(
                        &vk::SemaphoreCreateInfo::default().push_next(&mut export_info),
                        None,
                    )
                    .vk_ctx("recreate_semaphore (ace_done export)")?;
                self.export_sem_stale = false;
            }
            device
                .reset_command_buffer(self.cb, vk::CommandBufferResetFlags::empty())
                .vk_ctx("reset_command_buffer (ace copier)")?;
            device
                .begin_command_buffer(
                    self.cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .vk_ctx("begin_command_buffer (ace copier)")?;

            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let layers = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1);
            for c in copies {
                let to_dst = [vk::ImageMemoryBarrier2::default()
                    .image(c.dst)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .src_access_mask(vk::AccessFlags2::empty())
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .subresource_range(range)];
                let region = [vk::ImageCopy::default()
                    .src_subresource(layers)
                    .dst_subresource(layers)
                    .extent(vk::Extent3D {
                        width: c.extent.width,
                        height: c.extent.height,
                        depth: 1,
                    })];
                // Transition to the sampled layout here on the ACE queue
                // (the image is CONCURRENT-shared across the two families,
                // so no ownership transfer). Cross-queue availability +
                // visibility toward the graphics FRAGMENT_SHADER stage ride
                // the local_done semaphore signal/wait, so the barrier's own
                // dst scope only needs stages valid on a compute queue.
                let to_sample = [vk::ImageMemoryBarrier2::default()
                    .image(c.dst)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                    .subresource_range(range)];
                device.cmd_pipeline_barrier2(
                    self.cb,
                    &vk::DependencyInfo::default().image_memory_barriers(&to_dst),
                );
                device.cmd_copy_image(
                    self.cb,
                    c.src,
                    vk::ImageLayout::GENERAL,
                    c.dst,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &region,
                );
                device.cmd_pipeline_barrier2(
                    self.cb,
                    &vk::DependencyInfo::default().image_memory_barriers(&to_sample),
                );
            }

            device
                .end_command_buffer(self.cb)
                .vk_ctx("end_command_buffer (ace copier)")?;

            let cb_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(self.cb)];
            let wait_infos: Vec<vk::SemaphoreSubmitInfo> = waits
                .iter()
                .map(|&sem| {
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(sem)
                        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                })
                .collect();
            let signals = [
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.local_done)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.export_sem)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            ];
            let submits = [vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_infos)
                .command_buffer_infos(&cb_infos)
                .signal_semaphore_infos(&signals)];
            self.device
                .submit_transfer(&submits, self.fence, "queue_submit2 (ace mirror copy)")?;
            self.local_done_unconsumed = true;
        }

        // Export the ace_done semaphore as a sync_file fd (unsignals the
        // VkSemaphore per spec, freeing it for the next frame). The batch is
        // already submitted at this point, so an export failure takes the
        // abort path itself — the caller propagating the error will never
        // issue the graphics submit that consumes `local_done` / extends the
        // deferred-destroy proof over the batch.
        let get_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(self.export_sem)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let raw_fd = match unsafe { self.sem_fd_loader.get_semaphore_fd(&get_info) }
            .vk_ctx("vkGetSemaphoreFdKHR (ace copy SYNC_FD)")
        {
            Ok(fd) if fd >= 0 => fd,
            Ok(_) => {
                self.abort_unconsumed();
                self.export_sem_stale = true;
                return Err(RendererError::MissingFeature(
                    "vkGetSemaphoreFdKHR returned a negative fd (ace copy)",
                ));
            }
            Err(e) => {
                self.abort_unconsumed();
                self.export_sem_stale = true;
                return Err(e);
            }
        };
        // SAFETY: a fresh fd owned by us per the export contract.
        Ok((self.local_done, unsafe { OwnedFd::from_raw_fd(raw_fd) }))
    }

    /// The graphics submit carrying the `local_done` wait was queued — the
    /// signal will be consumed; normal reuse next frame.
    pub fn note_consumed(&mut self) {
        self.local_done_unconsumed = false;
    }

    /// The graphics submit never happened after an ACE submit: wait the
    /// copier fence so the in-flight copy provably finishes before the
    /// caller lets the frame's referenced objects retire (the transitive
    /// deferred-destroy proof is severed on this path), and leave
    /// `local_done` flagged for recreation.
    pub fn abort_unconsumed(&mut self) {
        let _ = unsafe {
            self.device
                .raw
                .wait_for_fences(&[self.fence], true, u64::MAX)
        };
    }
}

impl Drop for AceCopier {
    fn drop(&mut self) {
        // The renderer (sole owner) is being torn down; the device idles
        // all queues before destruction, but this copier's work may still
        // be in flight right now — the fence bounds it.
        let _ = unsafe {
            self.device
                .raw
                .wait_for_fences(&[self.fence], true, u64::MAX)
        };
        unsafe {
            self.device.raw.destroy_semaphore(self.local_done, None);
            self.device.raw.destroy_semaphore(self.export_sem, None);
            self.device.raw.destroy_fence(self.fence, None);
            self.device.raw.destroy_command_pool(self.pool, None);
        }
    }
}
