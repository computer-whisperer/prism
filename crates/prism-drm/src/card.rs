//! Per-card DRM context.
//!
//! Layered above [`SeatSession`] (one per process, owns libseat) and below
//! [`OutputContext`] (one per active connector, owns the scanout pipeline).
//! A `DrmCardContext` is **one per `/dev/dri/cardN` we drive**: it holds
//! the `DrmDevice` (kernel-side card handle, opened through libseat so it
//! has DRM master when the session is active) and a matching `GbmDevice`
//! (same fd, required for GEM-handle compatibility with `addfb2`).
//!
//! Multiple outputs on the same card share their `DrmCardContext` —
//! one DRM device, many surfaces. Multi-GPU systems have one
//! `DrmCardContext` per card; each is matched to a Vulkan device via
//! [`DrmDevId`] (drm major/minor of the primary node).
//!
//! Owns the per-card `DrmDeviceNotifier`-companion at construction
//! time only — returned to the caller so it can be inserted into calloop
//! alongside the session notifier.

use anyhow::{Context, Result};
use prism_renderer::{DrmDevId, EncodeConfig, vk};
use smithay::backend::drm::{DrmDevice, DrmDeviceNotifier};

use crate::{GbmDevice, ScanoutDepth, SeatSession};

/// The per-card DRM + GBM state. Created via [`DrmCardContext::open`].
/// Drop releases the DRM device fd (kernel-side, via libseat).
pub struct DrmCardContext {
    /// Original path used to open the card (e.g. `/dev/dri/card0`).
    pub path: String,
    /// DRM kernel handle. Used to query connectors, create surfaces,
    /// register framebuffers. Shared across all outputs on this card.
    pub drm: DrmDevice,
    /// GBM allocator on the same fd. Per-card because GEM handles are
    /// per-fd; sharing the fd is required for `addfb2`.
    pub gbm: GbmDevice,
    /// DRM primary-node major/minor (extracted from the device fd). Used
    /// to match this card to its Vulkan device (`Device::physical.drm_primary`).
    pub drm_dev_id: DrmDevId,
}

impl DrmCardContext {
    /// Open a DRM card through the seat. Returns `(card, drm_notifier)` —
    /// the caller MUST insert `drm_notifier` into the calloop event loop.
    /// Without it, page-flip events accumulate kernel-side until ENOMEM
    /// cascade.
    pub fn open(session: &mut SeatSession, path: &str) -> Result<(Self, DrmDeviceNotifier)> {
        let drm_fd = session.open_drm(path)?;
        let (drm, drm_notifier) = DrmDevice::new(drm_fd, false)
            .with_context(|| format!("DrmDevice::new({path})"))?;

        // GBM must share the same fd as DrmDevice (GEM handles are per-fd).
        let gbm = GbmDevice::from_device_fd(drm.device_fd().device_fd())?;

        let dev_id_raw = drm.device_id();
        let drm_dev_id = DrmDevId {
            major: libc::major(dev_id_raw) as i64,
            minor: libc::minor(dev_id_raw) as i64,
        };

        tracing::info!(
            path = %path,
            major = drm_dev_id.major,
            minor = drm_dev_id.minor,
            "DRM card opened"
        );

        Ok((
            Self {
                path: path.to_string(),
                drm,
                gbm,
                drm_dev_id,
            },
            drm_notifier,
        ))
    }
}

/// Static-per-output configuration: everything the renderer + scanout path
/// needs to know about an output that doesn't change per-frame.
///
/// Bundled into one type because the same bundle gets consumed by:
///   - `OutputContext::new` (allocates BOs, builds renderer pipelines)
///   - KMS bringup (mode, max_bpc, HDR_OUTPUT_METADATA based on encode chain)
///   - Future config layer (load from disk, override by EDID / user input)
///
/// Today this carries: scanout depth, the Vulkan format that matches it,
/// the intermediate-pass format, and the encode shader composition.
/// Will grow to include:
///   - target color description (primaries + TF + ref white)
///   - panel peak luminance (from EDID HDR static metadata block)
///   - calibration matrix (3×3 in linear-light, post-decode, pre-OETF)
///   - 1D / 3D LUT corrections
///   - tone-map curve choice
///   - SDR-on-HDR reference white nits
#[derive(Clone, Debug)]
pub struct OutputConfig {
    /// Bit depth of the scanout BO + matching Vulkan format are coupled;
    /// this enum picks both.
    pub depth: ScanoutDepth,
    /// Vulkan format for the scanout image. Must match `depth.drm_fourcc()`
    /// byte layout. Today this is derived from `depth`; kept explicit so
    /// future fp16 scanout (`AB48` / `R16G16B16A16_SFLOAT`) which doesn't
    /// fit cleanly into `ScanoutDepth` has a slot.
    pub vk_format: vk::Format,
    /// fp16 / fp32 intermediate-pass format (the BT.2020 absolute-nits
    /// linear buffer between the decode and encode passes).
    pub intermediate_format: vk::Format,
    /// Encode-shader composition. Determines which OETF + calibration +
    /// post-process effects run in the per-output encode pass.
    pub encode_config: EncodeConfig,
    /// If true, enable variable-refresh-rate (Adaptive Sync / Freesync /
    /// HDMI VRR) on the connector at bringup and tell `FrameClock` it can
    /// stretch the interval past the nominal refresh when no frame is
    /// pending. Logs a warning and falls back to fixed refresh if the
    /// connector does not advertise VRR support.
    ///
    /// OnDemand (per niri config) is treated as `false` for now — needs
    /// content_type / fullscreen-window inspection to flip on/off
    /// dynamically.
    pub vrr: bool,
}
