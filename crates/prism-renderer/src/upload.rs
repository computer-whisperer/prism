//! CPU→Vulkan upload texture. Used for wl_shm client buffers (bytes in
//! shared memory) — copies through a persistent host-visible staging buffer
//! into a DEVICE_LOCAL sampled image.
//!
//! Fixed extent + format at construction. If the client resizes its surface,
//! drop the old `ShmTexture` and create a new one.
//!
//! Upload is synchronous: the one-shot command buffer that runs the
//! buffer→image copy waits for the queue to go idle before returning. That
//! keeps each upload race-free against any in-flight render that may still
//! be sampling the previous content. Cost: one queue stall per commit. Cheap
//! at the workloads we care about; revisit if commit rates grow.

use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::error::{RendererError, Result, VkResultExt};
use crate::intermediate::create_view;
use crate::oneshot::OneshotPool;

/// A sampled VkImage uploaded from CPU bytes via a persistent staging buffer.
pub struct ShmTexture {
    device: Arc<Device>,
    oneshot: OneshotPool,

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
        let image = unsafe { device.raw.create_image(&image_info, None) }
            .vk_ctx("create_image (shm)")?;

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
        unsafe { device.raw.bind_buffer_memory(staging_buffer, staging_memory, 0) }
            .vk_ctx("bind_buffer_memory (shm staging)")?;
        let staging_ptr = unsafe {
            device.raw.map_memory(
                staging_memory,
                0,
                buf_req.size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .vk_ctx("map_memory (shm staging)")? as *mut u8;

        let oneshot = OneshotPool::new(device.clone())?;

        Ok(Self {
            device,
            oneshot,
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

    /// Copy `bytes` (in tightly packed row order matching the texture extent)
    /// into the image. Synchronous: waits for the GPU to idle before
    /// returning, so the image is in SHADER_READ_ONLY_OPTIMAL and safe to
    /// sample by the time this returns.
    ///
    /// `src_stride` is the byte stride between rows in `bytes` (the wl_shm
    /// pool stride). If `src_stride == extent.width * bytes_per_pixel`, we
    /// copy in one shot; otherwise we copy row by row.
    pub fn upload_bytes(&mut self, bytes: &[u8], src_stride: usize) -> Result<()> {
        let row_bytes = (self.extent.width as usize) * (self.bytes_per_pixel as usize);
        let needed = row_bytes * (self.extent.height as usize);
        if bytes.len() < needed.max(src_stride * (self.extent.height as usize)) {
            // bytes too small for the declared geometry — bail rather than
            // read past the end.
            if bytes.len() < src_stride * (self.extent.height.saturating_sub(1) as usize) + row_bytes
            {
                return Err(RendererError::MissingFeature(
                    "shm upload: source buffer smaller than image extent",
                ));
            }
        }

        // SAFETY: staging is persistently mapped HOST_COHERENT, sized
        // staging_size at construction; we never touch it concurrently
        // (`&mut self`); the GPU side of any prior upload has been waited on
        // by the oneshot's queue_wait_idle, so the GPU isn't reading from it.
        unsafe {
            let dst = self.staging_ptr;
            if src_stride == row_bytes {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, needed);
            } else {
                for row in 0..self.extent.height as usize {
                    let src_off = row * src_stride;
                    let dst_off = row * row_bytes;
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr().add(src_off),
                        dst.add(dst_off),
                        row_bytes,
                    );
                }
            }
        }

        // Record + submit + wait. UNDEFINED as the old layout discards
        // previous content, which is fine because we're about to overwrite
        // every texel.
        let image = self.image;
        let staging_buffer = self.staging_buffer;
        let extent = self.extent;
        self.oneshot.record_and_submit(|raw, cb| unsafe {
            let to_xfer = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_subresource_range())];
            raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&to_xfer),
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
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })];
            raw.cmd_copy_buffer_to_image(
                cb,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &region,
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
            raw.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&to_sampled),
            );
        })?;

        let _ = self.staging_size; // silence dead-code on this field; useful for future asserts
        Ok(())
    }
}

impl Drop for ShmTexture {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.raw.device_wait_idle();
            self.device.raw.unmap_memory(self.staging_memory);
            self.device.raw.destroy_buffer(self.staging_buffer, None);
            self.device.raw.free_memory(self.staging_memory, None);
            self.device.raw.destroy_image_view(self.view, None);
            self.device.raw.destroy_image(self.image, None);
            self.device.raw.free_memory(self.image_memory, None);
        }
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

fn pick_memory(
    device: &Device,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
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
    Err(RendererError::MissingFeature(
        "ShmTexture: no memory type matches required flags",
    ))
}
