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
use prism_renderer::{vk, DrmDevId, EncodeConfig};
use smithay::backend::drm::{DrmDevice, DrmDeviceNotifier, DrmNode};

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
    /// The same identity as smithay's [`DrmNode`], for protocols that key
    /// their per-device state by node (`wp_drm_lease_device_v1`).
    pub node: DrmNode,
}

impl DrmCardContext {
    /// Open a DRM card through the seat. Returns `(card, drm_notifier)` —
    /// the caller MUST insert `drm_notifier` into the calloop event loop.
    /// Without it, page-flip events accumulate kernel-side until ENOMEM
    /// cascade.
    pub fn open(session: &mut SeatSession, path: &str) -> Result<(Self, DrmDeviceNotifier)> {
        let drm_fd = session.open_drm(path)?;
        let (drm, drm_notifier) =
            DrmDevice::new(drm_fd, false).with_context(|| format!("DrmDevice::new({path})"))?;

        // GBM must share the same fd as DrmDevice (GEM handles are per-fd).
        let gbm = GbmDevice::from_device_fd(drm.device_fd().device_fd())?;

        let dev_id_raw = drm.device_id();
        let drm_dev_id = DrmDevId {
            major: libc::major(dev_id_raw) as i64,
            minor: libc::minor(dev_id_raw) as i64,
        };
        let node = DrmNode::from_dev_id(dev_id_raw)
            .with_context(|| format!("DrmNode::from_dev_id for {path}"))?;

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
                node,
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
    /// HDR signaling to push to the connector at bringup. `None` =
    /// SDR scanout. When `Some`, the bringup path also flips
    /// `depth`/`vk_format`/`encode_config` upstream (in main.rs) to
    /// fp16 + PQ encode. The signaling here only controls the KMS
    /// connector-side advertisement (HDR_OUTPUT_METADATA blob +
    /// Colorspace).
    pub hdr: Option<crate::HdrSignaling>,
    /// Absolute peak luminance (cd/m²) advertised to color-management
    /// clients as the display's mastering ceiling (the
    /// `mastering_luminance` max in the preferred `wp_color_management_v1`
    /// image description). Purely a client-facing advertisement — unlike
    /// `hdr.max_luminance` it does not feed the HDR_OUTPUT_METADATA
    /// infoframe or the encode clamp. `None` = advertise
    /// `hdr.max_luminance`. Only meaningful when `hdr` is `Some`.
    pub advertised_peak_nits: Option<u32>,
    /// What absolute luminance "SDR white" (= 1.0 from a color-unaware
    /// client) maps to on this output, in cd/m². Plumbed through to
    /// the per-surface decode push as the `sdr_white_nits` parameter
    /// fallback when the surface has no `wp_color_management_v1`
    /// description. IEC sRGB default is 80; HDR outputs may want
    /// higher (203 = BT.2408 reference white, or somewhere between
    /// for taste).
    pub sdr_reference_nits: f32,
    /// Per-channel panel luminance ceiling in cd/m² — the maximum
    /// value each subpixel channel of the display-referred
    /// intermediate is allowed to hold for this output. The decoder
    /// clamps post-EOTF values per-channel so the rest of the pipeline
    /// operates entirely within the panel's realizable range.
    /// Per-channel because real subpixel peaks differ — OLED ABL
    /// allocates power per subpixel, and LCD color-filter transmission
    /// varies per primary. Derived at bringup: KDL `panel-peak-nits
    /// r=… g=… b=…` (preferred, from calibration), else broadcast of
    /// `hdr.max_luminance` (HDR) or `sdr_reference_nits` (SDR).
    pub panel_peak_nits_rgb: [f32; 3],
    /// Per-channel response correction parameters: `(gain_rgb,
    /// gamma_rgb)`. The encoder applies `(in / gain)^(1/gamma)` per
    /// channel before the OETF to compensate for the panel's measured
    /// `emitted = gain * commanded^gamma` response. `None` =
    /// identity (no correction).
    pub response_curve: Option<([f32; 3], [f32; 3])>,
    /// Per-output 3×3 calibration matrix in row-major order, applied
    /// in the encode shader as `panel_rgb = M * bt2020_rgb`. Derived
    /// from measured panel primaries — maps BT.2020 IR values to the
    /// panel's native-primary linear nits so the per-channel response
    /// curve (which characterises the panel in its own primaries)
    /// works against panel-native input. `None` = identity (no gamut
    /// correction, panel emits its native primaries directly).
    pub ctm: Option<[[f32; 3]; 3]>,
    /// Sub-LSB deband strength for 8-bit SDR content on this output, from
    /// `output { tune { deband strength=… } }`. `None` = debanding off.
    /// Lowered into a per-surface blur request at render time (see
    /// `prism_renderer::lower_elements`); σ scales with source resolution.
    pub deband_strength: Option<f32>,
    /// Deband blur resolution divisor from `tune { deband downsample=… }`
    /// (rounded down to a power of two). `1` = full-res; higher is cheaper.
    /// Ignored when `deband_strength` is `None`.
    pub deband_downsample: u32,
}
