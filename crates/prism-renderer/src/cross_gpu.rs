//! Cross-GPU texture mirroring.
//!
//! A client dmabuf imports zero-copy only on the GPU(s) whose driver
//! understands its DRM format modifier. When a surface is displayed on an
//! output driven by a *different* GPU (multi-GPU compositor), that GPU
//! cannot import the buffer at all — sampling it would return nothing.
//!
//! The fix is a **mirror**: on a "home" GPU that *can* read the client
//! buffer, we keep a LINEAR, dmabuf-exportable scratch image
//! ([`ExportableImage`]) and copy the client pixels into it each commit
//! ([`MirrorCopier`]). LINEAR is universally importable, so the exported
//! dmabuf re-imports cleanly on the target GPU via the normal
//! [`crate::dmabuf::ImportedImage`] path. The target then samples its
//! import like any other surface texture.
//!
//! Cost: one GPU→GPU copy per commit of a mirrored surface (snapshots
//! don't track later client writes the way a zero-copy import does). This
//! is the fallback path — once per-output dmabuf feedback steers a client
//! to allocate buffers the display GPU can read natively, no mirror is
//! built at all.
//!
//! ## Memory placement
//!
//! The exportable image's backing memory must be reachable by *both*
//! GPUs. Device-local VRAM generally is not (peer GPUs can't read each
//! other's VRAM without PCIe P2P), so we prefer a host-visible (GTT /
//! system) memory type — universally importable across same-driver
//! devices. The home GPU writes it as a copy target; the target GPU reads
//! it as a sampled image.
//!
//! ## Layout / sync
//!
//! LINEAR images carry no compression metadata, so the home-side scratch
//! image stays in `GENERAL` and the target-side import transitions once to
//! `SHADER_READ_ONLY_OPTIMAL` — both map to the same physical byte layout.
//! Cross-device visibility is provided by the copy submit's fence (radv
//! flushes external-memory writes at the submit boundary) plus our
//! single-threaded commit→render ordering: the copy completes during
//! `commit`, the target samples during a later `render`.

use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};
use std::sync::Arc;

use ash::khr::{external_memory_fd, external_semaphore_fd};
use ash::vk;
use prism_frame::{Dmabuf, DmabufPlane, DrmFourcc, DrmModifier};

use crate::device::Device;
use crate::error::{RendererError, Result, VkResultExt};

/// DRM_FORMAT_MOD_LINEAR. The one modifier every driver can import.
const DRM_FORMAT_MOD_LINEAR: u64 = 0;

/// A LINEAR, dmabuf-exportable `VkImage` on a "home" GPU, used as the
/// destination of a cross-GPU mirror copy. Owns the image + memory and
/// holds the exported [`Dmabuf`] description so a target GPU can import
/// the same memory. Destroys the Vulkan objects on drop; the exported fd
/// is owned by `exported` and closed with it.
pub struct ExportableImage {
    device: Arc<Device>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    extent: vk::Extent2D,
    /// dmabuf description of this image's memory, for import on another
    /// GPU. The fd is owned here; importers dup it (they never consume the
    /// caller's fd — see [`crate::dmabuf::ImportedImage::import`]).
    exported: Dmabuf,
}

impl ExportableImage {
    pub fn image(&self) -> vk::Image {
        self.image
    }
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }
    /// The exported dmabuf description, ready to hand to
    /// [`crate::dmabuf::ImportedImage::import`] on a target GPU.
    pub fn exported_dmabuf(&self) -> &Dmabuf {
        &self.exported
    }

    /// Create a LINEAR exportable scratch image on `device`, sized `extent`,
    /// in `vk_format` (must match the client image's format so the copy is a
    /// straight `vkCmdCopyImage`). `fourcc` is the DRM code paired with the
    /// exported dmabuf — it must correspond to `vk_format`.
    pub fn new(
        device: Arc<Device>,
        extent: vk::Extent2D,
        vk_format: vk::Format,
        fourcc: DrmFourcc,
    ) -> Result<Self> {
        // Create with an explicit single-element modifier list = LINEAR.
        // The driver picks LINEAR; we then query its exact plane layout.
        let modifiers = [DRM_FORMAT_MOD_LINEAR];
        let mut modifier_list =
            vk::ImageDrmFormatModifierListCreateInfoEXT::default().drm_format_modifiers(&modifiers);
        let mut external_image = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_image)
            .push_next(&mut modifier_list);

        let image = unsafe { device.raw.create_image(&image_info, None) }
            .vk_ctx("create_image (exportable mirror)")?;

        let cleanup_image = |device: &Device, image: vk::Image| unsafe {
            device.raw.destroy_image(image, None);
        };

        let memory = match allocate_exportable_memory(&device, image) {
            Ok(m) => m,
            Err(e) => {
                cleanup_image(&device, image);
                return Err(e);
            }
        };
        if let Err(e) = unsafe { device.raw.bind_image_memory(image, memory, 0) }
            .vk_ctx("bind_image_memory (exportable mirror)")
        {
            unsafe { device.raw.free_memory(memory, None) };
            cleanup_image(&device, image);
            return Err(e);
        }

        // Query the LINEAR plane layout the driver actually chose. For a
        // DRM-modifier image we ask for memory plane 0.
        let layout = unsafe {
            device.raw.get_image_subresource_layout(
                image,
                vk::ImageSubresource::default()
                    .aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT),
            )
        };

        let exported = match export_dmabuf(&device, memory, extent, fourcc, &layout) {
            Ok(d) => d,
            Err(e) => {
                unsafe {
                    device.raw.free_memory(memory, None);
                    device.raw.destroy_image(image, None);
                }
                return Err(e);
            }
        };

        // Move the scratch image into GENERAL once. It stays there: copies
        // write it as GENERAL, and the cross-device target reads its own
        // import (SHADER_READ_ONLY_OPTIMAL) — same LINEAR bytes either way.
        let img = Self {
            device,
            image,
            memory,
            extent,
            exported,
        };
        if let Err(e) = img.transition_to_general() {
            return Err(e);
        }

        tracing::debug!(
            "created exportable mirror {}x{} format={:?} stride={} offset={} size={}",
            extent.width,
            extent.height,
            vk_format,
            layout.row_pitch,
            layout.offset,
            layout.size,
        );
        Ok(img)
    }

    fn transition_to_general(&self) -> Result<()> {
        let device = &self.device.raw;
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.device.physical.graphics_queue_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }
            .vk_ctx("create_command_pool (mirror init)")?;
        let result = (|| -> Result<()> {
            let cb = unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .command_buffer_count(1)
                        .level(vk::CommandBufferLevel::PRIMARY),
                )
            }
            .vk_ctx("allocate_command_buffers (mirror init)")?[0];
            unsafe {
                device.begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
            }
            .vk_ctx("begin_command_buffer (mirror init)")?;
            let barrier = [vk::ImageMemoryBarrier2::default()
                .image(self.image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .subresource_range(color_subresource())];
            unsafe {
                device.cmd_pipeline_barrier2(
                    cb,
                    &vk::DependencyInfo::default().image_memory_barriers(&barrier),
                );
                device.end_command_buffer(cb)
            }
            .vk_ctx("end_command_buffer (mirror init)")?;
            submit_and_wait(&self.device, cb)
        })();
        unsafe { device.destroy_command_pool(pool, None) };
        result
    }
}

impl Drop for ExportableImage {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_image(self.image, None);
            self.device.raw.free_memory(self.memory, None);
        }
    }
}

/// One source→scratch copy for [`MirrorCopier::copy_batch_async`].
pub struct MirrorCopyOp {
    /// Client import on the home GPU (the copy source), in `UNDEFINED`
    /// layout (imported untransitioned — it's only ever copied from).
    pub src: vk::Image,
    /// The LINEAR exportable scratch image (copy destination), in `GENERAL`.
    pub dst: vk::Image,
    pub extent: vk::Extent2D,
}

/// Persistent per-GPU command infrastructure for the cross-GPU mirror
/// copy. Reuses one command pool / command buffer / fence / exportable
/// semaphore across frames (resetting rather than recreating), held for the
/// device's lifetime by the materialization layer.
///
/// The copy is **asynchronous**: [`copy_batch_async`] submits without
/// blocking and signals an exportable binary semaphore, which is exported as
/// a Linux `sync_file` fd. The target GPU imports that fd
/// ([`import_wait_semaphore`]) and its render submit waits on it — so the
/// cross-device dependency is enforced GPU-side, never by stalling the
/// compositor event loop.
///
/// [`copy_batch_async`]: MirrorCopier::copy_batch_async
/// [`import_wait_semaphore`]: MirrorCopier::import_wait_semaphore
pub struct MirrorCopier {
    device: Arc<Device>,
    pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    /// Gates command-buffer reuse: waited at the *start* of the next
    /// `copy_batch_async` (not after submit), so the event loop doesn't
    /// block on the copy. Created signalled so the first wait is a no-op.
    fence: vk::Fence,
    /// Signalled by the copy submit, exported as a `sync_file` fd (export
    /// unsignals it, so it's reusable next frame).
    sem: vk::Semaphore,
    sem_fd_loader: external_semaphore_fd::Device,
}

impl MirrorCopier {
    pub fn new(device: Arc<Device>) -> Result<Self> {
        let pool = unsafe {
            device.raw.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(device.physical.graphics_queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .vk_ctx("create_command_pool (mirror copier)")?;
        let cb = unsafe {
            device.raw.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .command_buffer_count(1)
                    .level(vk::CommandBufferLevel::PRIMARY),
            )
        }
        .vk_ctx("allocate_command_buffers (mirror copier)")?[0];
        let fence = unsafe {
            device.raw.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }
        .vk_ctx("create_fence (mirror copier)")?;
        // Exportable (SYNC_FD) binary semaphore — same pattern as the
        // renderer's present semaphore.
        let mut export_info = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let sem = unsafe {
            device.raw.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut export_info),
                None,
            )
        }
        .vk_ctx("create_semaphore (mirror copier, exportable SYNC_FD)")?;
        let sem_fd_loader = external_semaphore_fd::Device::new(device.instance_raw(), &device.raw);
        Ok(Self {
            device,
            pool,
            cb,
            fence,
            sem,
            sem_fd_loader,
        })
    }

    /// Record + submit all `copies` (client import → LINEAR scratch) on this
    /// (home) GPU in one non-blocking submit, signalling the exportable
    /// semaphore, and return it exported as a `sync_file` fd marking
    /// "copies complete". The caller imports the fd on the target GPU
    /// ([`import_wait_semaphore`]) and waits on it in the render submit.
    ///
    /// Each `src` is in `UNDEFINED` (untransitioned client import); we move
    /// it to `TRANSFER_SRC_OPTIMAL` for the copy. Each `dst` stays `GENERAL`.
    ///
    /// [`import_wait_semaphore`]: MirrorCopier::import_wait_semaphore
    pub fn copy_batch_async(&self, copies: &[MirrorCopyOp]) -> Result<OwnedFd> {
        let device = &self.device.raw;
        unsafe {
            // Gate reuse on the previous copy finishing (no-op in steady
            // state — the copy is long done by the next frame).
            device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .vk_ctx("wait_for_fences (mirror copy reuse gate)")?;
            device
                .reset_fences(&[self.fence])
                .vk_ctx("reset_fences (mirror copy)")?;
            device
                .reset_command_buffer(self.cb, vk::CommandBufferResetFlags::empty())
                .vk_ctx("reset_command_buffer (mirror copy)")?;
            device
                .begin_command_buffer(
                    self.cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .vk_ctx("begin_command_buffer (mirror copy)")?;

            for op in copies {
                let pre = [
                    vk::ImageMemoryBarrier2::default()
                        .image(op.src)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                        .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                        .subresource_range(color_subresource()),
                    vk::ImageMemoryBarrier2::default()
                        .image(op.dst)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                        .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .subresource_range(color_subresource()),
                ];
                device.cmd_pipeline_barrier2(
                    self.cb,
                    &vk::DependencyInfo::default().image_memory_barriers(&pre),
                );
                let region = [vk::ImageCopy::default()
                    .src_subresource(color_layers())
                    .dst_subresource(color_layers())
                    .extent(vk::Extent3D {
                        width: op.extent.width,
                        height: op.extent.height,
                        depth: 1,
                    })];
                device.cmd_copy_image(
                    self.cb,
                    op.src,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    op.dst,
                    vk::ImageLayout::GENERAL,
                    &region,
                );
                // Make the writes available toward the shared BO so the
                // target GPU sees them once it waits on our semaphore.
                let post = [vk::ImageMemoryBarrier2::default()
                    .image(op.dst)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                    .subresource_range(color_subresource())];
                device.cmd_pipeline_barrier2(
                    self.cb,
                    &vk::DependencyInfo::default().image_memory_barriers(&post),
                );
            }

            device
                .end_command_buffer(self.cb)
                .vk_ctx("end_command_buffer (mirror copy)")?;

            let cb_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(self.cb)];
            let signal = [vk::SemaphoreSubmitInfo::default()
                .semaphore(self.sem)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let submits = [vk::SubmitInfo2::default()
                .command_buffer_infos(&cb_infos)
                .signal_semaphore_infos(&signal)];
            device
                .queue_submit2(self.device.graphics_queue, &submits, self.fence)
                .vk_ctx("queue_submit2 (mirror copy)")?;
        }

        // Export the just-signalled semaphore as a sync_file fd. Per spec the
        // export unsignals the VkSemaphore, so it's free for the next frame.
        let get_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(self.sem)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let raw_fd = unsafe { self.sem_fd_loader.get_semaphore_fd(&get_info) }
            .vk_ctx("vkGetSemaphoreFdKHR (mirror copy SYNC_FD)")?;
        if raw_fd < 0 {
            return Err(RendererError::MissingFeature(
                "vkGetSemaphoreFdKHR returned a negative fd (mirror copy)",
            ));
        }
        // SAFETY: a fresh fd owned by us per the export contract.
        Ok(unsafe { OwnedFd::from_raw_fd(raw_fd) })
    }

    /// Import a `sync_file` fd (from another GPU's [`copy_batch_async`]) as a
    /// fresh binary semaphore on *this* (target) device. The caller passes
    /// it to the render submit's wait list and destroys it
    /// ([`destroy_imported_semaphore`]) after the submit. Temporary import:
    /// the wait consumes the payload.
    ///
    /// [`destroy_imported_semaphore`]: MirrorCopier::destroy_imported_semaphore
    pub fn import_wait_semaphore(&self, fd: OwnedFd) -> Result<vk::Semaphore> {
        let sem = unsafe {
            self.device
                .raw
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
        }
        .vk_ctx("create_semaphore (mirror wait import)")?;
        let info = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(sem)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
            .flags(vk::SemaphoreImportFlags::TEMPORARY)
            .fd(fd.into_raw_fd());
        match unsafe { self.sem_fd_loader.import_semaphore_fd(&info) } {
            Ok(()) => Ok(sem),
            Err(e) => {
                // import_semaphore_fd consumed the fd; on failure the payload
                // wasn't imported, so just destroy the empty semaphore.
                unsafe { self.device.raw.destroy_semaphore(sem, None) };
                Err(RendererError::Vk {
                    context: "vkImportSemaphoreFdKHR (mirror wait, SYNC_FD)",
                    result: e,
                })
            }
        }
    }

    /// Destroy a semaphore returned by [`import_wait_semaphore`], after the
    /// render submit that waited on it has been queued.
    ///
    /// [`import_wait_semaphore`]: MirrorCopier::import_wait_semaphore
    pub fn destroy_imported_semaphore(&self, sem: vk::Semaphore) {
        unsafe { self.device.raw.destroy_semaphore(sem, None) };
    }
}

impl Drop for MirrorCopier {
    fn drop(&mut self) {
        unsafe {
            let _ = self
                .device
                .raw
                .wait_for_fences(&[self.fence], true, u64::MAX);
            self.device.raw.destroy_semaphore(self.sem, None);
            self.device.raw.destroy_fence(self.fence, None);
            self.device.raw.destroy_command_pool(self.pool, None);
        }
    }
}

fn color_subresource() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn color_layers() -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// Submit a one-shot command buffer on the device's graphics queue and
/// block on a transient fence. For the rare one-time mirror-init
/// transition; the hot copy path uses [`MirrorCopier`]'s persistent fence.
fn submit_and_wait(device: &Device, cb: vk::CommandBuffer) -> Result<()> {
    let fence = unsafe {
        device
            .raw
            .create_fence(&vk::FenceCreateInfo::default(), None)
    }
    .vk_ctx("create_fence (mirror init submit)")?;
    let cb_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
    let submits = [vk::SubmitInfo2::default().command_buffer_infos(&cb_infos)];
    let res = unsafe {
        device
            .raw
            .queue_submit2(device.graphics_queue, &submits, fence)
    }
    .vk_ctx("queue_submit2 (mirror init)")
    .and_then(|_| {
        unsafe { device.raw.wait_for_fences(&[fence], true, u64::MAX) }
            .vk_ctx("wait_for_fences (mirror init)")
    });
    unsafe { device.raw.destroy_fence(fence, None) };
    res
}

/// Allocate exportable memory for `image`, preferring a host-visible (GTT)
/// memory type so the BO is importable by a peer GPU. The memory is
/// dedicated to the image and marked exportable as DMA_BUF.
fn allocate_exportable_memory(device: &Device, image: vk::Image) -> Result<vk::DeviceMemory> {
    let req = unsafe { device.raw.get_image_memory_requirements(image) };
    let mem_type = pick_exportable_memory(device, req.memory_type_bits)?;

    let mut export_info = vk::ExportMemoryAllocateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_type)
        .push_next(&mut export_info)
        .push_next(&mut dedicated);
    unsafe { device.raw.allocate_memory(&alloc, None) }
        .vk_ctx("allocate_memory (exportable mirror)")
}

/// Pick a memory type for the exportable mirror. Prefers host-visible
/// (GTT / system memory), which a peer GPU can import; falls back to any
/// allowed type if no host-visible one qualifies (single-GPU configs,
/// where cross-import never happens anyway).
fn pick_exportable_memory(device: &Device, type_bits: u32) -> Result<u32> {
    let props = unsafe {
        device
            .instance_raw()
            .get_physical_device_memory_properties(device.physical.raw)
    };
    let mut fallback: Option<u32> = None;
    for i in 0..props.memory_type_count {
        if (type_bits & (1 << i)) == 0 {
            continue;
        }
        fallback.get_or_insert(i);
        if props.memory_types[i as usize]
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            return Ok(i);
        }
    }
    fallback.ok_or(RendererError::MissingFeature(
        "no memory type for exportable mirror image",
    ))
}

/// Export `memory` as a dmabuf fd and build a single-plane LINEAR
/// [`Dmabuf`] describing it.
fn export_dmabuf(
    device: &Device,
    memory: vk::DeviceMemory,
    extent: vk::Extent2D,
    fourcc: DrmFourcc,
    layout: &vk::SubresourceLayout,
) -> Result<Dmabuf> {
    let fd_loader = external_memory_fd::Device::new(device.instance_raw(), &device.raw);
    let get_info = vk::MemoryGetFdInfoKHR::default()
        .memory(memory)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let raw_fd = unsafe { fd_loader.get_memory_fd(&get_info) }
        .vk_ctx("get_memory_fd (exportable mirror)")?;
    // SAFETY: get_memory_fd transfers ownership of a fresh fd to us.
    let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
    Ok(Dmabuf {
        width: extent.width,
        height: extent.height,
        format: fourcc,
        modifier: DrmModifier::Linear,
        planes: vec![DmabufPlane {
            fd,
            offset: layout.offset as u32,
            stride: layout.row_pitch as u32,
        }],
    })
}
