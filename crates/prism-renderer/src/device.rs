use std::ffi::CStr;
use std::sync::Arc;

use ash::ext::{
    external_memory_dma_buf, external_memory_host, image_drm_format_modifier, physical_device_drm,
};
use ash::khr::{external_memory_fd, external_semaphore_fd, image_format_list, push_descriptor};
use ash::vk;
use tracing::{debug, info, warn};

use crate::error::{RendererError, Result, VkResultExt};
use crate::instance::Instance;

/// Extensions we require on the chosen physical device. Without these, the
/// renderer cannot do its job (importing client dmabufs as Vulkan textures
/// and exporting our scanout output with explicit DRM format modifiers).
const REQUIRED_DEVICE_EXTS: &[&CStr] = &[
    image_drm_format_modifier::NAME,
    external_memory_fd::NAME,
    external_memory_dma_buf::NAME,
    image_format_list::NAME,
    external_semaphore_fd::NAME,
    // Lets us bind descriptors at command-record time without allocating
    // from a pool — no per-frame allocate/free churn. RADV + recent NVidia
    // + Intel all support this; failure here means an unusably old GPU.
    push_descriptor::NAME,
];

/// Optional extensions. Enabled if available; absence is logged but not fatal.
const OPTIONAL_DEVICE_EXTS: &[&CStr] = &[physical_device_drm::NAME, external_memory_host::NAME];

/// DRM device-id (major, minor) for matching a Vulkan physical device to a
/// DRM node opened separately. Populated via `VK_EXT_physical_device_drm`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DrmDevId {
    pub major: i64,
    pub minor: i64,
}

/// Info about a physical device we've decided to keep. Stored on `Device`.
pub struct PhysicalDeviceInfo {
    pub raw: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub name: String,
    /// DRM primary node major/minor, if `VK_EXT_physical_device_drm` reports one.
    pub drm_primary: Option<DrmDevId>,
    /// DRM render node major/minor, ditto.
    pub drm_render: Option<DrmDevId>,
    pub graphics_queue_family: u32,
    /// A COMPUTE+TRANSFER queue family that is *not* the graphics family — the
    /// async-compute engines (ACEs) on AMD. Used for transfer work that should
    /// overlap graphics (the cross-GPU mirror copy; see
    /// `docs/async-render-rework.md`). `None` on GPUs that expose no such
    /// family (then the graphics queue does double duty).
    pub transfer_queue_family: Option<u32>,
    /// Extensions we enabled on this device (intersection of requested set
    /// with what's actually present — required exts always present; optional
    /// exts may or may not be).
    pub enabled_extensions: Vec<&'static CStr>,
}

/// A logical Vulkan device + the queue we use for graphics work. Owns the
/// lifetime; dropping waits for idle, then destroys.
pub struct Device {
    /// Kept to keep the parent instance alive while the device exists, and
    /// to give callers access to the raw instance handle without re-plumbing
    /// it everywhere.
    pub instance: Arc<Instance>,
    pub physical: PhysicalDeviceInfo,
    pub raw: ash::Device,
    pub graphics_queue: vk::Queue,
    /// Async-compute (ACE) queue for transfer work that should overlap
    /// graphics — the cross-GPU mirror copy (see `docs/async-render-rework.md`).
    /// Equals `graphics_queue` when the GPU exposes no dedicated ACE family, so
    /// callers can always submit here without a capability branch.
    ///
    /// NOTE: a distinct `VkQueue` still needs external synchronization — only
    /// one thread may submit to it at a time, same as `graphics_queue`.
    pub transfer_queue: vk::Queue,
    /// Queue family `transfer_queue` belongs to (for command-pool creation).
    /// Equals `physical.graphics_queue_family` in the fallback case.
    pub transfer_queue_family: u32,
    /// Deferred-destroy queue — see [`Device::retire`].
    deferred: std::sync::Mutex<DeferredDestroy>,
}

/// A GPU object handed to [`Device::retire`] for deferred destruction: it is
/// destroyed only once every queue submission issued *before* the retirement
/// has completed, so in-flight frames can still reference it safely.
#[derive(Debug)]
pub enum Retired {
    /// An image with its view and backing memory (e.g. a dmabuf import, an
    /// shm/snapshot texture, the persistent intermediate).
    Image {
        image: vk::Image,
        view: vk::ImageView,
        memory: vk::DeviceMemory,
    },
    /// A buffer with its backing memory (e.g. an upload staging buffer).
    Buffer {
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
    },
    /// A binary semaphore (e.g. an imported render-wait semaphore — the spec
    /// forbids destroying it before the batch that waited on it completes).
    Semaphore(vk::Semaphore),
    Fence(vk::Fence),
    CommandPool(vk::CommandPool),
}

/// Book-keeping for the deferred-destroy queue.
///
/// Correctness rests on two facts: prism puts all GPU work on the single
/// `graphics_queue`, and Vulkan's implicit ordering guarantees a fence
/// signal waits for *all* commands submitted to the queue before it
/// (spec §7.2, "Implicit Synchronization Guarantees"). So a monotonic
/// per-submission serial plus "fence for serial N signalled" proves every
/// submission ≤ N is complete, across every renderer/copier/uploader
/// sharing the device.
#[derive(Default)]
struct DeferredDestroy {
    /// Serial of the most recent submission to `graphics_queue`. Bumped via
    /// [`Device::note_submit`] immediately before each `vkQueueSubmit`.
    submitted: u64,
    /// Highest serial proven complete by a fence/idle wait
    /// ([`Device::note_completed`]).
    completed: u64,
    /// Retired objects, each stamped with the `submitted` serial at
    /// retirement time — i.e. the last submission that could reference it.
    retired: Vec<(u64, Retired)>,
}

impl Device {
    /// Allocate the serial for a submission about to be enqueued on
    /// `graphics_queue`. MUST be called (once) before **every**
    /// `vkQueueSubmit2` on this device, or a [`Self::retire`]d object that
    /// the unsequenced submission references could be destroyed under it.
    /// Returns the serial; pass it to [`Self::note_completed`] from
    /// whichever fence/idle wait later proves that submission finished.
    pub fn note_submit(&self) -> u64 {
        let mut d = self.deferred.lock().unwrap();
        d.submitted += 1;
        d.submitted
    }

    /// Record that the submission with `serial` (and, by single-queue fence
    /// ordering, every earlier one) has completed, then destroy any retired
    /// objects that are no longer referenced. Call after a successful
    /// `wait_for_fences` / `queue_wait_idle` tied to that submission.
    pub fn note_completed(&self, serial: u64) {
        let mut to_destroy = Vec::new();
        {
            let mut d = self.deferred.lock().unwrap();
            if serial > d.completed {
                d.completed = serial;
            }
            let completed = d.completed;
            let mut i = 0;
            while i < d.retired.len() {
                if d.retired[i].0 <= completed {
                    to_destroy.push(d.retired.swap_remove(i).1);
                } else {
                    i += 1;
                }
            }
        }
        // Destroy outside the lock (destruction can be slow; nothing else
        // can resurrect a popped entry).
        for r in to_destroy {
            unsafe { self.destroy_retired(r) };
        }
    }

    /// Queue a GPU object for destruction once every submission issued so
    /// far has completed. If the queue is already proven idle past the
    /// current serial, destroys immediately.
    pub fn retire(&self, r: Retired) {
        let mut d = self.deferred.lock().unwrap();
        if d.submitted <= d.completed {
            drop(d);
            unsafe { self.destroy_retired(r) };
            return;
        }
        let stamp = d.submitted;
        d.retired.push((stamp, r));
    }

    /// Destroy one retired object. Caller must guarantee no in-flight
    /// submission references it.
    unsafe fn destroy_retired(&self, r: Retired) {
        match r {
            Retired::Image {
                image,
                view,
                memory,
            } => {
                self.raw.destroy_image_view(view, None);
                self.raw.destroy_image(image, None);
                self.raw.free_memory(memory, None);
            }
            Retired::Buffer { buffer, memory } => {
                self.raw.destroy_buffer(buffer, None);
                self.raw.free_memory(memory, None);
            }
            Retired::Semaphore(sem) => self.raw.destroy_semaphore(sem, None),
            Retired::Fence(fence) => self.raw.destroy_fence(fence, None),
            Retired::CommandPool(pool) => self.raw.destroy_command_pool(pool, None),
        }
    }
}

/// One entry from `vkGetPhysicalDeviceFormatProperties2` +
/// `VkDrmFormatModifierPropertiesListEXT`. Describes one DRM format
/// modifier the Vulkan driver supports for a given Vulkan format,
/// along with how many memory planes the modifier uses and which
/// usage classes (color attachment / sampling / etc.) work with it.
#[derive(Clone, Copy, Debug)]
pub struct DrmFormatModifierInfo {
    pub modifier: u64,
    pub plane_count: u32,
    pub tiling_features: vk::FormatFeatureFlags,
}

impl Device {
    /// Access the raw `ash::Instance` for building extension loaders.
    pub fn instance_raw(&self) -> &ash::Instance {
        self.instance.raw()
    }

    /// Query the Vulkan driver for the DRM format modifiers it supports for
    /// `format`. Used to negotiate a tiled scanout layout we can both render
    /// into (via `ImportedImage`) and feed to KMS, instead of falling back
    /// to LINEAR which on amdgpu+fp16 burns enough scanout bandwidth to
    /// trigger transient `-ENOMEM` from the DCN validator under contention.
    ///
    /// Caller filters the returned list (e.g. by `plane_count == 1`,
    /// required `tiling_features`). An empty result means the driver
    /// doesn't advertise any modifier for this format — caller should
    /// fall back to LINEAR.
    ///
    /// Two-pass query: first call gets the count, second fills the buffer.
    /// Done unconditionally per-output at bringup; the cost is one
    /// `vkGetPhysicalDeviceFormatProperties2` round-trip.
    pub fn supported_drm_format_modifiers(&self, format: vk::Format) -> Vec<DrmFormatModifierInfo> {
        let phys = self.physical.raw;
        let instance = self.instance.raw();

        // Pass 1: query count with a null buffer.
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
        let mut props2 = vk::FormatProperties2::default().push_next(&mut list);
        unsafe {
            instance.get_physical_device_format_properties2(phys, format, &mut props2);
        }
        let count = list.drm_format_modifier_count as usize;
        if count == 0 {
            return Vec::new();
        }

        // Pass 2: fill the buffer. `props2` re-uses the pNext chain; we
        // rebuild it so `list` points at the freshly-allocated storage.
        let mut storage: Vec<vk::DrmFormatModifierPropertiesEXT> = vec![Default::default(); count];
        let filled = {
            let mut list = vk::DrmFormatModifierPropertiesListEXT::default()
                .drm_format_modifier_properties(&mut storage);
            let mut props2 = vk::FormatProperties2::default().push_next(&mut list);
            unsafe {
                instance.get_physical_device_format_properties2(phys, format, &mut props2);
            }
            list.drm_format_modifier_count as usize
        };
        // Truncate to what the second call actually filled (drivers may
        // report a smaller count on the second pass; not common but spec-allowed).
        storage.truncate(filled);

        storage
            .into_iter()
            .map(|p| DrmFormatModifierInfo {
                modifier: p.drm_format_modifier,
                plane_count: p.drm_format_modifier_plane_count,
                tiling_features: p.drm_format_modifier_tiling_features,
            })
            .collect()
    }

    /// Open a Vulkan logical device.
    ///
    /// Selection priority:
    ///   1. Physical device whose DRM node matches `prefer_drm_node` (major/minor).
    ///   2. Discrete GPU.
    ///   3. First device that meets the required-extension bar.
    pub fn new(instance: Arc<Instance>, prefer_drm_node: Option<DrmDevId>) -> Result<Arc<Self>> {
        let physicals = unsafe { instance.raw().enumerate_physical_devices() }
            .vk_ctx("enumerate_physical_devices")?;
        if physicals.is_empty() {
            return Err(RendererError::NoSuitableDevice);
        }

        let mut candidates: Vec<PhysicalDeviceInfo> = Vec::new();
        for &phys in &physicals {
            match probe_physical_device(&instance, phys) {
                Ok(info) => {
                    debug!(
                        "candidate: {} type={:?} drm_primary={:?} drm_render={:?}",
                        info.name, info.properties.device_type, info.drm_primary, info.drm_render
                    );
                    candidates.push(info);
                }
                Err(e) => debug!("skipping physical device: {e}"),
            }
        }
        if candidates.is_empty() {
            return Err(RendererError::NoSuitableDevice);
        }

        let pick_index = prefer_drm_node
            .and_then(|want| {
                candidates
                    .iter()
                    .position(|c| c.drm_primary == Some(want) || c.drm_render == Some(want))
            })
            .or_else(|| {
                candidates
                    .iter()
                    .position(|c| c.properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
            })
            .unwrap_or(0);

        let info = candidates.swap_remove(pick_index);
        info!(
            "Selected Vulkan device: {} (type={:?}, drm_primary={:?}, drm_render={:?})",
            info.name, info.properties.device_type, info.drm_primary, info.drm_render
        );

        let ext_ptrs: Vec<*const i8> = info.enabled_extensions.iter().map(|e| e.as_ptr()).collect();
        let queue_priorities = [1.0_f32];
        // Always request the graphics queue; additionally request one ACE queue
        // when the GPU exposes a dedicated async-compute family, for transfer
        // work that overlaps graphics (see `docs/async-render-rework.md`).
        let mut queue_infos = vec![vk::DeviceQueueCreateInfo::default()
            .queue_family_index(info.graphics_queue_family)
            .queue_priorities(&queue_priorities)];
        if let Some(tqf) = info.transfer_queue_family {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(tqf)
                    .queue_priorities(&queue_priorities),
            );
        }

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .descriptor_indexing(true);
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .dynamic_rendering(true);

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&ext_ptrs)
            .push_next(&mut features12)
            .push_next(&mut features13);

        let raw = unsafe { instance.raw().create_device(info.raw, &device_info, None) }
            .vk_ctx("create_device")?;

        let graphics_queue = unsafe { raw.get_device_queue(info.graphics_queue_family, 0) };
        // ACE queue when present, else the graphics queue does double duty so
        // callers never need a capability branch.
        let transfer_queue_family = info
            .transfer_queue_family
            .unwrap_or(info.graphics_queue_family);
        let transfer_queue = unsafe { raw.get_device_queue(transfer_queue_family, 0) };
        info!(
            "queues: graphics family {}, transfer family {} ({})",
            info.graphics_queue_family,
            transfer_queue_family,
            if info.transfer_queue_family.is_some() {
                "dedicated ACE"
            } else {
                "shared with graphics"
            },
        );

        Ok(Arc::new(Self {
            instance,
            physical: info,
            raw,
            graphics_queue,
            transfer_queue,
            transfer_queue_family,
            deferred: std::sync::Mutex::new(DeferredDestroy::default()),
        }))
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let _ = self.raw.device_wait_idle();
            // Idle ⇒ every submission completed; drain whatever the deferred
            // queue still holds before the device goes away.
            let retired = std::mem::take(&mut self.deferred.lock().unwrap().retired);
            for (_, r) in retired {
                self.destroy_retired(r);
            }
            self.raw.destroy_device(None);
        }
    }
}

fn probe_physical_device(
    instance: &Instance,
    phys: vk::PhysicalDevice,
) -> Result<PhysicalDeviceInfo> {
    let properties = unsafe { instance.raw().get_physical_device_properties(phys) };
    let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    if properties.api_version < vk::API_VERSION_1_3 {
        return Err(RendererError::MissingFeature(
            "Vulkan 1.3 not supported on this device",
        ));
    }

    let exts = unsafe { instance.raw().enumerate_device_extension_properties(phys) }
        .vk_ctx("enumerate_device_extension_properties")?;
    let has_ext = |name: &CStr| {
        exts.iter()
            .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == name)
    };

    let mut enabled_extensions: Vec<&'static CStr> = Vec::new();
    for required in REQUIRED_DEVICE_EXTS {
        if !has_ext(required) {
            return Err(RendererError::MissingExtension(
                required.to_string_lossy().into_owned(),
            ));
        }
        enabled_extensions.push(required);
    }
    for opt in OPTIONAL_DEVICE_EXTS {
        if has_ext(opt) {
            enabled_extensions.push(opt);
        } else {
            warn!(
                "{}: optional extension {} unavailable",
                name,
                opt.to_string_lossy()
            );
        }
    }

    let qfp = unsafe {
        instance
            .raw()
            .get_physical_device_queue_family_properties(phys)
    };
    let graphics_queue_family = qfp
        .iter()
        .enumerate()
        .find(|(_, p)| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .map(|(i, _)| i as u32)
        .ok_or(RendererError::MissingFeature("graphics queue family"))?;

    // A dedicated async-compute family (COMPUTE+TRANSFER, no GRAPHICS) — the
    // ACE queues on AMD. Used for transfer work meant to overlap graphics.
    // `None` ⇒ no such family; callers fall back to the graphics queue.
    let transfer_queue_family = qfp
        .iter()
        .enumerate()
        .find(|(_, p)| {
            p.queue_flags
                .contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER)
                && !p.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        })
        .map(|(i, _)| i as u32);

    let (drm_primary, drm_render) = if has_ext(physical_device_drm::NAME) {
        let mut drm_props = vk::PhysicalDeviceDrmPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut drm_props);
        unsafe {
            instance
                .raw()
                .get_physical_device_properties2(phys, &mut props2)
        };
        let primary = (drm_props.has_primary != 0).then_some(DrmDevId {
            major: drm_props.primary_major,
            minor: drm_props.primary_minor,
        });
        let render = (drm_props.has_render != 0).then_some(DrmDevId {
            major: drm_props.render_major,
            minor: drm_props.render_minor,
        });
        (primary, render)
    } else {
        (None, None)
    };

    Ok(PhysicalDeviceInfo {
        raw: phys,
        properties,
        name,
        drm_primary,
        drm_render,
        graphics_queue_family,
        transfer_queue_family,
        enabled_extensions,
    })
}
