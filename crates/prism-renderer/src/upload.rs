//! CPU→Vulkan upload texture. Used for wl_shm client buffers (bytes in
//! shared memory) — copies through a persistent host-visible staging buffer
//! into a DEVICE_LOCAL sampled image.
//!
//! Fixed extent + format at construction. If the client resizes its surface,
//! drop the old `ShmTexture` and create a new one.
//!
//! Upload is asynchronous: the buffer→image copy is recorded into a
//! per-texture command buffer and submitted with a per-texture fence — no
//! `queue_wait_idle`. Correctness rests on the single per-device graphics
//! queue being submitted only from the main thread, with the upload always
//! submitted before the render that samples it (commit handlers run before
//! `redraw_queued_outputs` in the same loop iteration):
//!
//!   - vs the *next* render reading the new pixels: ordered by submission
//!     order plus the renderer's producer barrier (MEMORY_WRITE →
//!     SHADER_SAMPLED_READ) at frame start.
//!   - vs the *previous* render still sampling the old pixels: the copy's
//!     acquire barrier sources from FRAGMENT_SHADER / SHADER_READ_ONLY_OPTIMAL
//!     (after the first upload), which on a single queue synchronizes against
//!     all prior-submitted commands — so the copy waits for that sampling.
//!   - vs the CPU clobbering staging while a prior copy still reads it: the
//!     per-texture fence, waited before the next memcpy. Signalled long ago
//!     in steady state (one upload per frame), so the wait is free.

use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::error::{RendererError, Result, VkResultExt};
use crate::intermediate::create_view;

/// A sampled VkImage uploaded from CPU bytes via a persistent staging buffer.
pub struct ShmTexture {
    device: Arc<Device>,
    /// Per-texture command pool + one reused command buffer for the copy:
    /// one allocation, reset each upload (no per-commit allocate/free).
    command_pool: vk::CommandPool,
    cmd_buffer: vk::CommandBuffer,
    /// Signalled by each upload's submit; waited at the start of the next
    /// upload to gate staging-buffer reuse. Created signalled.
    upload_fence: vk::Fence,
    /// Device submission serial of the last upload submit; reported to
    /// `Device::note_completed` when the fence wait proves it finished
    /// (drives the deferred-destroy queue). 0 = never submitted.
    last_submit_serial: u64,
    /// False until the first upload has initialized the image contents and
    /// layout. The first upload is always full-extent and acquires from
    /// UNDEFINED; later uploads acquire from SHADER_READ_ONLY_OPTIMAL.
    initialized: bool,

    image: vk::Image,
    view: vk::ImageView,
    image_memory: vk::DeviceMemory,

    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    staging_ptr: *mut u8,
    staging_size: vk::DeviceSize,

    extent: vk::Extent2D,
    format: vk::Format,
    bytes_per_pixel: u32,
}

// The mapped staging pointer is per-instance; ShmTexture is otherwise Send
// + Sync the same way ImportedImage is (raw Vulkan handles guarded by the
// `Arc<Device>`). The pointer doesn't break that — we only dereference it
// inside `upload_bytes` which takes `&mut self`.
unsafe impl Send for ShmTexture {}
unsafe impl Sync for ShmTexture {}

impl ShmTexture {
    pub fn view(&self) -> vk::ImageView {
        self.view
    }
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Allocate the image + staging buffer for the given size. `format` must
    /// be a single-planar 32-bit packed format (currently the only formats
    /// we map wl_shm into). Extent is fixed for the lifetime of the texture.
    pub fn new(device: Arc<Device>, extent: vk::Extent2D, format: vk::Format) -> Result<Self> {
        let bytes_per_pixel = bytes_per_pixel_for(format)?;
        let staging_size = (extent.width as vk::DeviceSize)
            * (extent.height as vk::DeviceSize)
            * (bytes_per_pixel as vk::DeviceSize);

        // ── Image ──────────────────────────────────────────────────────────
        let image_info = vk::ImageCreateInfo::default()
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
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image =
            unsafe { device.raw.create_image(&image_info, None) }.vk_ctx("create_image (shm)")?;

        let img_req = unsafe { device.raw.get_image_memory_requirements(image) };
        let img_mem_type = pick_memory(
            &device,
            img_req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let img_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(img_req.size)
            .memory_type_index(img_mem_type);
        let image_memory = unsafe { device.raw.allocate_memory(&img_alloc, None) }
            .vk_ctx("allocate_memory (shm image)")?;
        unsafe { device.raw.bind_image_memory(image, image_memory, 0) }
            .vk_ctx("bind_image_memory (shm)")?;

        let view = create_view(&device, image, format)?;

        // ── Staging buffer ────────────────────────────────────────────────
        let buf_info = vk::BufferCreateInfo::default()
            .size(staging_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buffer = unsafe { device.raw.create_buffer(&buf_info, None) }
            .vk_ctx("create_buffer (shm staging)")?;
        let buf_req = unsafe { device.raw.get_buffer_memory_requirements(staging_buffer) };
        let buf_mem_type = pick_memory(
            &device,
            buf_req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let buf_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(buf_req.size)
            .memory_type_index(buf_mem_type);
        let staging_memory = unsafe { device.raw.allocate_memory(&buf_alloc, None) }
            .vk_ctx("allocate_memory (shm staging)")?;
        unsafe {
            device
                .raw
                .bind_buffer_memory(staging_buffer, staging_memory, 0)
        }
        .vk_ctx("bind_buffer_memory (shm staging)")?;
        let staging_ptr = unsafe {
            device
                .raw
                .map_memory(staging_memory, 0, buf_req.size, vk::MemoryMapFlags::empty())
        }
        .vk_ctx("map_memory (shm staging)")? as *mut u8;

        // Per-texture command pool + one reusable command buffer for the
        // copy. RESET_COMMAND_BUFFER lets us reset the buffer each upload
        // rather than reallocating.
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.physical.graphics_queue_family)
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            );
        let command_pool = unsafe { device.raw.create_command_pool(&pool_info, None) }
            .vk_ctx("create_command_pool (shm upload)")?;
        let cb_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buffer = unsafe { device.raw.allocate_command_buffers(&cb_alloc) }
            .vk_ctx("allocate_command_buffers (shm upload)")?[0];
        // Created signalled so the first upload's wait is a no-op.
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let upload_fence = unsafe { device.raw.create_fence(&fence_info, None) }
            .vk_ctx("create_fence (shm upload)")?;

        Ok(Self {
            device,
            command_pool,
            cmd_buffer,
            upload_fence,
            last_submit_serial: 0,
            initialized: false,
            image,
            view,
            image_memory,
            staging_buffer,
            staging_memory,
            staging_ptr,
            staging_size,
            extent,
            format,
            bytes_per_pixel,
        })
    }

    /// Copy `bytes` into the image, uploading only the rows covered by
    /// `damage` (image/buffer pixel coords; for wl_shm the two grids coincide).
    /// `bytes` is the full source buffer and `src_stride` its byte row stride
    /// (the wl_shm pool stride).
    ///
    /// Damage handling:
    ///   - first upload (image uninitialized): `damage` is ignored and the
    ///     whole extent is uploaded — a fresh image has no prior content and
    ///     must be fully written (and left SHADER_READ_ONLY_OPTIMAL for
    ///     sampling);
    ///   - later uploads: only the `damage` rects are copied, preserving the
    ///     rest of the persistent image. Empty `damage` is a no-op.
    ///
    /// Pass `&[]` to mean "no damage info": on a new texture that yields a
    /// full upload, on an already-populated one a no-op.
    pub fn upload_bytes(
        &mut self,
        bytes: &[u8],
        src_stride: usize,
        damage: &[vk::Rect2D],
    ) -> Result<()> {
        let row_bytes = (self.extent.width as usize) * (self.bytes_per_pixel as usize);
        let needed = row_bytes * (self.extent.height as usize);
        if bytes.len() < needed.max(src_stride * (self.extent.height as usize)) {
            // bytes too small for the declared geometry — bail rather than
            // read past the end.
            if bytes.len()
                < src_stride * (self.extent.height.saturating_sub(1) as usize) + row_bytes
            {
                return Err(RendererError::MissingFeature(
                    "shm upload: source buffer smaller than image extent",
                ));
            }
        }

        // Effective upload regions. A fresh image must be fully written; an
        // already-populated one with no damage needs no work at all.
        let full = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent,
        }];
        let regions: &[vk::Rect2D] = if !self.initialized {
            &full
        } else if damage.is_empty() {
            return Ok(());
        } else {
            damage
        };

        // Gate staging-buffer reuse: wait for the previous upload's copy to
        // finish reading staging before we overwrite it. Signalled at create
        // and (in steady state) long before now, so this rarely blocks.
        unsafe {
            self.device
                .raw
                .wait_for_fences(&[self.upload_fence], true, u64::MAX)
        }
        .vk_ctx("wait_for_fences (shm upload)")?;
        unsafe { self.device.raw.reset_fences(&[self.upload_fence]) }
            .vk_ctx("reset_fences (shm upload)")?;
        self.device.note_completed(self.last_submit_serial);

        // Copy the damaged rows into staging at their tightly-packed offsets.
        // Staging mirrors the image (row stride = row_bytes), so the GPU copy
        // addresses each rect by (y*row_bytes + x*bpp). Non-damaged staging
        // bytes may be stale — the GPU copy below only reads the damage rects.
        //
        // SAFETY: staging is persistently mapped HOST_COHERENT, sized for the
        // full image; we never touch it concurrently (`&mut self`); the prior
        // copy from staging has completed (the fence wait above). Source reads
        // stay in-bounds: rects are clamped to the extent by the caller and
        // `bytes` was bounds-checked against the full geometry above.
        let bpp = self.bytes_per_pixel as usize;
        unsafe {
            let dst = self.staging_ptr;
            let is_full = regions.len() == 1
                && regions[0].offset.x == 0
                && regions[0].offset.y == 0
                && regions[0].extent.width == self.extent.width
                && regions[0].extent.height == self.extent.height;
            if is_full && src_stride == row_bytes {
                // Fast path: one contiguous copy.
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, needed);
            } else {
                for r in regions {
                    let x = r.offset.x as usize;
                    let y = r.offset.y as usize;
                    let span = r.extent.width as usize * bpp;
                    for row in 0..r.extent.height as usize {
                        let src_off = (y + row) * src_stride + x * bpp;
                        let dst_off = (y + row) * row_bytes + x * bpp;
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr().add(src_off),
                            dst.add(dst_off),
                            span,
                        );
                    }
                }
            }
        }

        // Record the copy into our reusable command buffer and submit with
        // our fence — no queue idle. See the module doc for why this is
        // race-free on the single main-thread graphics queue.
        let image = self.image;
        let staging_buffer = self.staging_buffer;
        let extent = self.extent;
        let cb = self.cmd_buffer;

        // First upload: acquire from UNDEFINED (no prior content to preserve
        // or to synchronize against). Later uploads: acquire from
        // SHADER_READ_ONLY_OPTIMAL with a FRAGMENT_SHADER source scope, so the
        // copy waits for any prior render still sampling this image.
        let (old_layout, src_stage, src_access) = if self.initialized {
            (
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
            )
        } else {
            (
                vk::ImageLayout::UNDEFINED,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::empty(),
            )
        };

        unsafe {
            self.device
                .raw
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
        }
        .vk_ctx("reset_command_buffer (shm upload)")?;
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.raw.begin_command_buffer(cb, &begin_info) }
            .vk_ctx("begin_command_buffer (shm upload)")?;
        unsafe {
            let to_xfer = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_subresource_range())];
            self.device.raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&to_xfer),
            );

            // One copy region per damage rect. `buffer_row_length` is the
            // staging row stride in texels (full image width, tightly packed);
            // `buffer_offset` addresses the rect's top-left texel.
            let bpp_u64 = self.bytes_per_pixel as u64;
            let row_bytes_u64 = row_bytes as u64;
            let copies: Vec<vk::BufferImageCopy> = regions
                .iter()
                .map(|r| {
                    vk::BufferImageCopy::default()
                        .buffer_offset(
                            r.offset.y as u64 * row_bytes_u64 + r.offset.x as u64 * bpp_u64,
                        )
                        .buffer_row_length(extent.width)
                        .buffer_image_height(0)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_offset(vk::Offset3D {
                            x: r.offset.x,
                            y: r.offset.y,
                            z: 0,
                        })
                        .image_extent(vk::Extent3D {
                            width: r.extent.width,
                            height: r.extent.height,
                            depth: 1,
                        })
                })
                .collect();
            self.device.raw.cmd_copy_buffer_to_image(
                cb,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copies,
            );

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
                .subresource_range(color_subresource_range())];
            self.device.raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&to_sampled),
            );
        }
        unsafe { self.device.raw.end_command_buffer(cb) }
            .vk_ctx("end_command_buffer (shm upload)")?;

        let cb_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
        let submit = [vk::SubmitInfo2::default().command_buffer_infos(&cb_infos)];
        let serial = self.device.note_submit();
        unsafe {
            self.device
                .raw
                .queue_submit2(self.device.graphics_queue, &submit, self.upload_fence)
        }
        .vk_ctx("queue_submit2 (shm upload)")?;
        self.last_submit_serial = serial;

        self.initialized = true;
        let _ = self.staging_size; // silence dead-code on this field; useful for future asserts
        Ok(())
    }
}

impl Drop for ShmTexture {
    fn drop(&mut self) {
        use crate::device::Retired;
        // Retire everything the GPU may still reference instead of the old
        // `device_wait_idle` (a full-pipeline stall on every shm texture
        // realloc/close). Unmapping is a host-side operation and is safe
        // while the GPU reads the memory; the free is what's deferred.
        unsafe { self.device.raw.unmap_memory(self.staging_memory) };
        self.device.retire(Retired::Fence(self.upload_fence));
        self.device.retire(Retired::CommandPool(self.command_pool));
        self.device.retire(Retired::Buffer {
            buffer: self.staging_buffer,
            memory: self.staging_memory,
        });
        self.device.retire(Retired::Image {
            image: self.image,
            view: self.view,
            memory: self.image_memory,
        });
    }
}

fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn bytes_per_pixel_for(format: vk::Format) -> Result<u32> {
    Ok(match format {
        vk::Format::B8G8R8A8_UNORM
        | vk::Format::R8G8B8A8_UNORM
        | vk::Format::B8G8R8A8_SRGB
        | vk::Format::R8G8B8A8_SRGB => 4,
        vk::Format::R16G16B16A16_SFLOAT => 8,
        _ => {
            return Err(RendererError::MissingFeature(
                "ShmTexture: unsupported format",
            ));
        }
    })
}

fn pick_memory(device: &Device, type_bits: u32, required: vk::MemoryPropertyFlags) -> Result<u32> {
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
    Err(RendererError::MissingFeature(
        "ShmTexture: no memory type matches required flags",
    ))
}
