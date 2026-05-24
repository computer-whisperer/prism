use std::ffi::CStr;
use std::sync::Arc;

use ash::ext::debug_utils;
use ash::vk;
use tracing::{debug, info, warn};

use crate::error::{Result, VkResultExt};

const APP_NAME: &CStr = c"prism";
const ENGINE_NAME: &CStr = c"prism";
const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// A Vulkan instance. Shared via `Arc` since every `Device` holds a reference
/// to keep its parent instance alive.
pub struct Instance {
    pub(crate) entry: ash::Entry,
    pub(crate) raw: ash::Instance,
    debug: Option<DebugMessenger>,
}

struct DebugMessenger {
    loader: debug_utils::Instance,
    handle: vk::DebugUtilsMessengerEXT,
}

impl Instance {
    pub fn new() -> Result<Arc<Self>> {
        let entry = unsafe { ash::Entry::load() }?;

        let api_version = vk::API_VERSION_1_3;
        let app_info = vk::ApplicationInfo::default()
            .application_name(APP_NAME)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(ENGINE_NAME)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(api_version);

        let available_exts = unsafe { entry.enumerate_instance_extension_properties(None) }
            .vk_ctx("enumerate_instance_extension_properties")?;
        let has_ext = |name: &CStr| {
            available_exts
                .iter()
                .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == name)
        };

        // Validation runs in debug builds, or in any build when
        // `PRISM_VK_VALIDATION` is set to a non-empty, non-"0" value — the
        // release-build escape hatch for debugging on real hardware (the DRM
        // scanout path only exists outside `cargo test`, so synchronization
        // hazards there can't be caught by a debug unit-test run).
        let force_validation = std::env::var("PRISM_VK_VALIDATION")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);
        let want_validation = cfg!(debug_assertions) || force_validation;

        let mut enable_exts: Vec<*const i8> = Vec::new();
        // Need the debug-utils messenger whenever validation runs so its
        // output routes through our tracing callback instead of stderr.
        let want_debug_utils = want_validation;
        let have_debug_utils = has_ext(debug_utils::NAME);
        if want_debug_utils {
            if have_debug_utils {
                enable_exts.push(debug_utils::NAME.as_ptr());
            } else {
                warn!("VK_EXT_debug_utils unavailable; Vulkan messages will not be logged");
            }
        }

        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }
            .vk_ctx("enumerate_instance_layer_properties")?;
        let have_validation = available_layers
            .iter()
            .any(|l| unsafe { CStr::from_ptr(l.layer_name.as_ptr()) } == VALIDATION_LAYER);
        let mut enable_layers: Vec<*const i8> = Vec::new();
        if want_validation {
            if have_validation {
                enable_layers.push(VALIDATION_LAYER.as_ptr());
            } else {
                warn!(
                    "Vulkan validation layer not installed; install vulkan-validation-layers \
                     for development-time checks"
                );
            }
        }

        // Synchronization validation is off by default even when the layer is
        // loaded. It statically analyzes the barrier / semaphore / fence graph
        // for missing-synchronization hazards — the class of bug behind
        // intermittent, timing-dependent corruption — so it flags a latent
        // missing barrier even in a run where the corruption doesn't surface.
        let sync_features = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
        let mut validation_features =
            vk::ValidationFeaturesEXT::default().enabled_validation_features(&sync_features);

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&enable_exts)
            .enabled_layer_names(&enable_layers);
        if want_validation && have_validation {
            create_info = create_info.push_next(&mut validation_features);
        }

        let raw = unsafe { entry.create_instance(&create_info, None) }.vk_ctx("create_instance")?;

        info!(
            "Vulkan instance created (target API {}.{}.{}, validation={}, debug_utils={})",
            vk::api_version_major(api_version),
            vk::api_version_minor(api_version),
            vk::api_version_patch(api_version),
            want_validation && have_validation,
            want_debug_utils && have_debug_utils,
        );

        let debug = if want_debug_utils && have_debug_utils {
            let loader = debug_utils::Instance::new(&entry, &raw);
            // Only WARN+ from validation/performance; GENERAL is mostly loader chatter.
            let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vk_debug_callback));
            let handle = unsafe { loader.create_debug_utils_messenger(&messenger_info, None) }
                .vk_ctx("create_debug_utils_messenger")?;
            Some(DebugMessenger { loader, handle })
        } else {
            None
        };

        Ok(Arc::new(Self { entry, raw, debug }))
    }

    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    pub fn raw(&self) -> &ash::Instance {
        &self.raw
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            if let Some(debug) = self.debug.take() {
                debug
                    .loader
                    .destroy_debug_utils_messenger(debug.handle, None);
            }
            self.raw.destroy_instance(None);
        }
    }
}

unsafe extern "system" fn vk_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    cb_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let data = unsafe { &*cb_data };
    let message = unsafe { CStr::from_ptr(data.p_message) }.to_string_lossy();
    let kind = format!("{:?}", msg_type);
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        tracing::error!(target: "vulkan", %kind, "{}", message);
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        warn!(target: "vulkan", %kind, "{}", message);
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
        info!(target: "vulkan", %kind, "{}", message);
    } else {
        debug!(target: "vulkan", %kind, "{}", message);
    }
    vk::FALSE
}
