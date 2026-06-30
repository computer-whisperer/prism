//! dmabuf → VkImage import.
//!
//! Takes a `prism_frame::Dmabuf` and creates a Vulkan image backed by the
//! same kernel BO, via:
//!   - `VK_EXT_image_drm_format_modifier` — tells the driver the image has
//!     a specific DRM format modifier and per-plane layout (offset/stride).
//!   - `VK_EXT_external_memory_dma_buf` + `VK_KHR_external_memory_fd` —
//!     imports the dmabuf fd as Vulkan device memory.
//!
//! Single-planar formats (XRGB8888, ARGB8888, RGBA16F, ...) go through
//! [`ImportedImage::import`]: one fd → one allocation → one image with
//! `plane_layouts` of length 1.
//!
//! Two-plane YUV (NV12, P010) goes through [`ImportedImage::import_yuv`],
//! which imports each plane as its **own** single-plane image (luma R8/R16
//! at full res, chroma R8G8/R16G16 at half res) and carries both in one
//! `ImportedImage`. The decode shader samples both and does YUV→RGB
//! manually — see `shaders/decode.frag`. This per-plane approach handles
//! the common VA-API case where each plane is exported as its own dmabuf;
//! a single BO shared across planes is addressed via each plane's
//! `offset` in its `plane_layouts` entry.

use std::os::fd::{AsFd, AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
use std::sync::Arc;

use ash::khr::external_memory_fd;
use ash::vk;
use prism_frame::{Dmabuf, DmabufPlane};
use tracing::debug;

use crate::device::Device;
use crate::error::{RendererError, Result, VkResultExt};

/// Planar YUV layout of an imported video buffer. Manual two-plane
/// sampling: a luma plane + an interleaved chroma plane, converted to
/// nonlinear RGB in the decode shader (`shaders/decode.frag`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YuvKind {
    /// 8-bit 4:2:0 — `R8` luma + `R8G8` chroma (half-res in both axes).
    Nv12,
    /// 10-bit 4:2:0 (10 bits in the high bits of each 16-bit sample) —
    /// `R16` luma + `R16G16` chroma (half-res in both axes).
    P010,
}

impl YuvKind {
    /// `(luma format, chroma format)` for the two single-plane images.
    pub fn plane_formats(self) -> (vk::Format, vk::Format) {
        match self {
            YuvKind::Nv12 => (vk::Format::R8_UNORM, vk::Format::R8G8_UNORM),
            YuvKind::P010 => (vk::Format::R16_UNORM, vk::Format::R16G16_UNORM),
        }
    }
}

/// The second (chroma) plane of a YUV import. Owns its own image/view/memory
/// alongside the luma plane held directly on [`ImportedImage`].
struct ChromaPlane {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
}

/// A `VkImage` backed by imported dmabuf memory. Owns the image + memory and
/// destroys them on drop. Does NOT own the dmabuf fds — those live on the
/// caller's `Dmabuf`.
///
/// For YUV imports the luma plane lives in the top-level fields and the
/// chroma plane in `chroma`; `yuv_kind` is then `Some`.
pub struct ImportedImage {
    device: Arc<Device>,
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    extent: vk::Extent2D,
    format: vk::Format,
    /// Chroma plane, present iff this is a YUV import.
    chroma: Option<ChromaPlane>,
    /// `Some` iff this is a YUV import; tells the render path to sample
    /// `chroma` and pick the YUV→RGB path.
    yuv_kind: Option<YuvKind>,
}

impl ImportedImage {
    pub fn image(&self) -> vk::Image {
        self.image
    }
    pub fn view(&self) -> vk::ImageView {
        self.view
    }
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }
    pub fn format(&self) -> vk::Format {
        self.format
    }
    /// The chroma plane's view, for YUV imports. `None` for RGB.
    pub fn chroma_view(&self) -> Option<vk::ImageView> {
        self.chroma.as_ref().map(|c| c.view)
    }
    /// The chroma plane's `VkImage`, for YUV imports. `None` for RGB.
    /// Used as the copy source when mirroring a YUV surface cross-GPU.
    pub fn chroma_image(&self) -> Option<vk::Image> {
        self.chroma.as_ref().map(|c| c.image)
    }
    /// `Some(kind)` iff this is a YUV import.
    pub fn yuv_kind(&self) -> Option<YuvKind> {
        self.yuv_kind
    }

    /// Import a single-plane dmabuf (RGB) as a `VkImage`.
    ///
    /// `vk_format` must match the dmabuf's `DrmFourcc` byte layout. E.g.
    /// `DrmFourcc::Xrgb8888` ↔ `vk::Format::B8G8R8A8_UNORM`: DRM is
    /// little-endian-byte-order, so XRGB-in-memory is B,G,R,X bytes, which
    /// is Vulkan's B8G8R8A8.
    pub fn import(
        device: Arc<Device>,
        dmabuf: &Dmabuf,
        vk_format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self> {
        if dmabuf.planes.len() != 1 {
            return Err(RendererError::MissingFeature(
                "ImportedImage::import is single-plane only; use import_yuv for NV12/P010",
            ));
        }
        let (image, memory, view) = import_plane(
            &device,
            u64::from(dmabuf.modifier),
            dmabuf.width,
            dmabuf.height,
            &dmabuf.planes[0],
            vk_format,
            usage,
        )?;

        debug!(
            "imported dmabuf as VkImage {}x{} format={:?} modifier={:#x}",
            dmabuf.width,
            dmabuf.height,
            vk_format,
            u64::from(dmabuf.modifier),
        );

        Ok(Self {
            device,
            image,
            view,
            memory,
            extent: vk::Extent2D {
                width: dmabuf.width,
                height: dmabuf.height,
            },
            format: vk_format,
            chroma: None,
            yuv_kind: None,
        })
    }

    /// Import a two-plane YUV dmabuf (NV12 / P010) as a luma image + a
    /// chroma image, both held in one `ImportedImage`. Each plane is
    /// imported independently as its own single-plane image (luma at full
    /// res, chroma at half res for 4:2:0); the decode shader recombines
    /// them. The caller maps the buffer's `DrmFourcc` to the `YuvKind`.
    pub fn import_yuv(
        device: Arc<Device>,
        dmabuf: &Dmabuf,
        kind: YuvKind,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self> {
        if dmabuf.planes.len() != 2 {
            return Err(RendererError::MissingFeature(
                "YUV import expects exactly two planes (luma + chroma)",
            ));
        }
        let modifier = u64::from(dmabuf.modifier);
        let (luma_fmt, chroma_fmt) = kind.plane_formats();

        let (l_image, l_memory, l_view) = import_plane(
            &device,
            modifier,
            dmabuf.width,
            dmabuf.height,
            &dmabuf.planes[0],
            luma_fmt,
            usage,
        )?;

        // 4:2:0 chroma is half-res in both axes; round up for odd sizes.
        let chroma_w = dmabuf.width.div_ceil(2);
        let chroma_h = dmabuf.height.div_ceil(2);
        let (c_image, c_memory, c_view) = match import_plane(
            &device,
            modifier,
            chroma_w,
            chroma_h,
            &dmabuf.planes[1],
            chroma_fmt,
            usage,
        ) {
            Ok(t) => t,
            Err(e) => {
                // Roll back the luma plane we already created.
                unsafe {
                    device.raw.destroy_image_view(l_view, None);
                    device.raw.destroy_image(l_image, None);
                    device.raw.free_memory(l_memory, None);
                }
                return Err(e);
            }
        };

        debug!(
            "imported {:?} dmabuf: luma {}x{} {:?} + chroma {}x{} {:?} modifier={:#x}",
            kind, dmabuf.width, dmabuf.height, luma_fmt, chroma_w, chroma_h, chroma_fmt, modifier,
        );

        Ok(Self {
            device,
            image: l_image,
            view: l_view,
            memory: l_memory,
            extent: vk::Extent2D {
                width: dmabuf.width,
                height: dmabuf.height,
            },
            format: luma_fmt,
            chroma: Some(ChromaPlane {
                image: c_image,
                view: c_view,
                memory: c_memory,
            }),
            yuv_kind: Some(kind),
        })
    }

    /// Transition this image from UNDEFINED → SHADER_READ_ONLY_OPTIMAL on the
    /// graphics queue, blocking until the transition completes.
    ///
    /// The image is created with `initial_layout = UNDEFINED`, but the render
    /// path binds it with `image_layout = SHADER_READ_ONLY_OPTIMAL` and never
    /// emits a layout-transition barrier of its own. Sampling from an image
    /// whose actual layout is UNDEFINED is undefined behaviour — on radv it
    /// hangs the queue. Doing the transition once at import time (when the
    /// producer's pixels are already in the BO) gets the image into a
    /// sampleable layout without us having to repeat the work every frame.
    ///
    /// For sampled dmabuf imports only. Color-attachment scanout images
    /// don't need this — they're transitioned per-frame in render_frame.
    pub fn transition_for_sampling(&self) -> Result<()> {
        self.transition_to(
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
        )
    }

    /// As [`transition_for_sampling`](Self::transition_for_sampling) but to
    /// `GENERAL` — for a mirror import that is read as a **copy source** (into
    /// the target-local image) rather than sampled directly. `GENERAL` is a
    /// valid `vkCmdCopyImage` source layout and lets the per-frame copy use a
    /// fixed source layout with no stateful re-transition.
    pub fn transition_to_general(&self) -> Result<()> {
        self.transition_to(
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags2::ALL_TRANSFER,
            vk::AccessFlags2::TRANSFER_READ,
        )
    }

    fn transition_to(
        &self,
        new_layout: vk::ImageLayout,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
    ) -> Result<()> {
        let device = &self.device.raw;
        let queue_family = self.device.physical.graphics_queue_family;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let pool = unsafe { device.create_command_pool(&pool_info, None) }
            .vk_ctx("create_command_pool (dmabuf transition)")?;

        let result = (|| -> Result<()> {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .command_buffer_count(1)
                .level(vk::CommandBufferLevel::PRIMARY);
            let cbs = unsafe { device.allocate_command_buffers(&alloc_info) }
                .vk_ctx("allocate_command_buffers (dmabuf transition)")?;
            let cb = cbs[0];

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe { device.begin_command_buffer(cb, &begin_info) }
                .vk_ctx("begin_command_buffer (dmabuf transition)")?;

            let make_barrier = |image: vk::Image| {
                vk::ImageMemoryBarrier2::default()
                    .image(image)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(new_layout)
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .src_access_mask(vk::AccessFlags2::empty())
                    .dst_stage_mask(dst_stage)
                    .dst_access_mask(dst_access)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
            };
            // YUV imports carry a second (chroma) image that must reach a
            // sampleable layout too.
            let mut barriers = vec![make_barrier(self.image)];
            if let Some(c) = self.chroma.as_ref() {
                barriers.push(make_barrier(c.image));
            }
            unsafe {
                device.cmd_pipeline_barrier2(
                    cb,
                    &vk::DependencyInfo::default().image_memory_barriers(&barriers),
                );
                device.end_command_buffer(cb)
            }
            .vk_ctx("end_command_buffer (dmabuf transition)")?;

            let cb_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
            let submits = [vk::SubmitInfo2::default().command_buffer_infos(&cb_infos)];
            let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
                .vk_ctx("create_fence (dmabuf transition)")?;
            let serial = self.device.note_submit();
            let submit_result =
                unsafe { device.queue_submit2(self.device.graphics_queue, &submits, fence) };
            if let Err(e) = submit_result {
                unsafe { device.destroy_fence(fence, None) };
                return Err(RendererError::Vk {
                    context: "queue_submit2 (dmabuf transition)",
                    result: e,
                });
            }
            let wait = unsafe { device.wait_for_fences(&[fence], true, u64::MAX) };
            unsafe { device.destroy_fence(fence, None) };
            if wait.is_ok() {
                self.device.note_completed(serial);
            }
            wait.vk_ctx("wait_for_fences (dmabuf transition)")
        })();

        unsafe { device.destroy_command_pool(pool, None) };
        result
    }
}

impl Drop for ImportedImage {
    fn drop(&mut self) {
        // Dropped on EVERY buffer swap of a double-buffered dmabuf client
        // (process_surface_buffer replaces the import when the committed
        // wl_buffer changes), while up to FRAMES_IN_FLIGHT submissions
        // referencing the view may still be executing. Retire instead of
        // destroying: the deferred queue frees the handles once the frame
        // slot fences prove every such submission complete. (Survived as an
        // immediate destroy only thanks to amdgpu's fence-deferred VM unmap.)
        self.device.retire(crate::device::Retired::Image {
            image: self.image,
            view: self.view,
            memory: self.memory,
        });
        if let Some(c) = self.chroma.as_ref() {
            self.device.retire(crate::device::Retired::Image {
                image: c.image,
                view: c.view,
                memory: c.memory,
            });
        }
    }
}

/// Import one dmabuf plane as a single-plane `VkImage` + cached view, returning
/// the raw handles (caller owns cleanup). Shared by single-plane RGB import and
/// each plane of a YUV import. `width`/`height` are this plane's dimensions
/// (chroma is half-res for 4:2:0), and `plane.offset` locates the plane within
/// its (possibly shared) BO.
fn import_plane(
    device: &Arc<Device>,
    modifier: u64,
    width: u32,
    height: u32,
    plane: &DmabufPlane,
    vk_format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let plane_layouts = [vk::SubresourceLayout {
        offset: u64::from(plane.offset),
        size: 0,
        row_pitch: u64::from(plane.stride),
        array_pitch: 0,
        depth_pitch: 0,
    }];

    let mut modifier_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(modifier)
        .plane_layouts(&plane_layouts);

    let mut external_image = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk_format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut external_image)
        .push_next(&mut modifier_info);

    let image =
        unsafe { device.raw.create_image(&image_info, None) }.vk_ctx("create_image (dmabuf)")?;

    let memory = match allocate_imported_memory(device, image, plane.fd.as_fd()) {
        Ok(m) => m,
        Err(e) => {
            unsafe { device.raw.destroy_image(image, None) };
            return Err(e);
        }
    };

    let bind_info = [vk::BindImageMemoryInfo::default()
        .image(image)
        .memory(memory)
        .memory_offset(0)];
    if let Err(e) = unsafe { device.raw.bind_image_memory2(&bind_info) } {
        unsafe {
            device.raw.free_memory(memory, None);
            device.raw.destroy_image(image, None);
        }
        return Err(RendererError::Vk {
            context: "bind_image_memory2 (dmabuf import)",
            result: e,
        });
    }

    // Cache the image view here so the renderer doesn't re-create one per
    // frame. The view's lifetime is tied to the image's.
    let view = match crate::intermediate::create_view(device, image, vk_format) {
        Ok(v) => v,
        Err(e) => {
            unsafe {
                device.raw.free_memory(memory, None);
                device.raw.destroy_image(image, None);
            }
            return Err(e);
        }
    };

    Ok((image, memory, view))
}

fn allocate_imported_memory(
    device: &Device,
    image: vk::Image,
    plane_fd: std::os::fd::BorrowedFd<'_>,
) -> Result<vk::DeviceMemory> {
    let mem_req2 = unsafe {
        let mut req = vk::MemoryRequirements2::default();
        let info = vk::ImageMemoryRequirementsInfo2::default().image(image);
        device.raw.get_image_memory_requirements2(&info, &mut req);
        req
    };

    let fd_loader = external_memory_fd::Device::new(device.instance_raw(), &device.raw);

    // Query which memory types accept this fd. The query does NOT consume
    // the fd, but takes a raw i32; dup so we don't borrow the caller's.
    let query_fd: OwnedFd = plane_fd
        .try_clone_to_owned()
        .map_err(|_| RendererError::Vk {
            context: "dup dmabuf fd for memory-type query",
            result: vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        })?;
    let mut fd_props = vk::MemoryFdPropertiesKHR::default();
    unsafe {
        fd_loader.get_memory_fd_properties(
            vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            query_fd.as_raw_fd(),
            &mut fd_props,
        )
    }
    .vk_ctx("get_memory_fd_properties (dmabuf)")?;
    drop(query_fd);

    let candidate_types = mem_req2.memory_requirements.memory_type_bits & fd_props.memory_type_bits;
    if candidate_types == 0 {
        return Err(RendererError::MissingFeature(
            "no memory type supports both this image and dmabuf import",
        ));
    }
    let mem_type_index = candidate_types.trailing_zeros();

    // Vulkan consumes this fd on success. Dup so the caller's Dmabuf stays
    // valid. On failure, reclaim into an OwnedFd to close it.
    let import_fd: OwnedFd = plane_fd
        .try_clone_to_owned()
        .map_err(|_| RendererError::Vk {
            context: "dup dmabuf fd for import",
            result: vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        })?;
    let raw_import_fd = import_fd.into_raw_fd();

    let mut import_info = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
        .fd(raw_import_fd);
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_req2.memory_requirements.size)
        .memory_type_index(mem_type_index)
        .push_next(&mut import_info)
        .push_next(&mut dedicated);

    match unsafe { device.raw.allocate_memory(&alloc_info, None) } {
        Ok(m) => Ok(m),
        Err(e) => {
            // Take the fd back so it's properly closed.
            unsafe { OwnedFd::from_raw_fd(raw_import_fd) };
            Err(RendererError::Vk {
                context: "allocate_memory (dmabuf import)",
                result: e,
            })
        }
    }
}
