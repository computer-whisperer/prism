use std::ffi::CStr;
use std::sync::Arc;

use ash::ext::{
    external_memory_dma_buf, external_memory_host, image_drm_format_modifier, physical_device_drm,
};
use ash::khr::{external_memory_fd, external_semaphore_fd, image_format_list};
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
];

/// Optional extensions. Enabled if available; absence is logged but not fatal.
const OPTIONAL_DEVICE_EXTS: &[&CStr] = &[physical_device_drm::NAME, external_memory_host::NAME];

/// DRM device-id (major, minor) for matching a Vulkan physical device to a
/// DRM node opened separately. Populated via `VK_EXT_physical_device_drm`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
}

impl Device {
    /// Access the raw `ash::Instance` for building extension loaders.
    pub fn instance_raw(&self) -> &ash::Instance {
        self.instance.raw()
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
                candidates.iter().position(|c| {
                    c.drm_primary == Some(want) || c.drm_render == Some(want)
                })
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
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(info.graphics_queue_family)
            .queue_priorities(&queue_priorities);
        let queue_infos = [queue_info];

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

        Ok(Arc::new(Self {
            instance,
            physical: info,
            raw,
            graphics_queue,
        }))
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let _ = self.raw.device_wait_idle();
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
        enabled_extensions,
    })
}
