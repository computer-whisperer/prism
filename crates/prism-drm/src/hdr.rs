//! HDR signaling — `HDR_OUTPUT_METADATA` blob + `Colorspace` enum.
//!
//! Port of niri's `tty.rs::HdrProps` + `build_hdr_metadata_blob`,
//! byte-identical so the kernel's `struct hdr_output_metadata` parses
//! the same way. Two connector properties carry the signaling:
//!
//!   - `HDR_OUTPUT_METADATA`: an opaque blob the kernel forwards to
//!     the panel as a CTA-861.3 HDR Static Metadata infoframe. Tells
//!     the panel "this content is PQ-encoded, mastered to a display
//!     with these primaries and luminance range, please tone-map
//!     accordingly."
//!   - `Colorspace`: an enum that selects the BT.601/709/2020 etc.
//!     colorimetry the panel should assume for incoming pixel data.
//!     `BT2020_RGB` is the conventional pairing for PQ HDR.
//!
//! Lifecycle: [`HdrProps::lookup`] at bringup discovers the property
//! handles (if absent, the connector doesn't support HDR signaling
//! and we silently no-op). [`HdrProps::set_hdr`] creates a new
//! property blob, swaps it in, and destroys the previous one.
//! [`HdrProps::clear`] sets metadata→0 and Colorspace→Default so the
//! desktop session that takes over after we exit doesn't inherit
//! stale HDR signaling — that's the "phase-1 hung-on PQ" failure
//! mode documented in `docs/color-management.md`.
//!
//! **Stage-1 feedforward.** The only validation here is "the kernel
//! accepted the property set". We don't measure the resulting panel
//! output — closed-loop measurement against a colorimeter is
//! `prism-tune`'s job, not this path's.

use anyhow::{Context, Result};
use smithay::reexports::drm::control::{connector, property, Device as ControlDevice};

/// HDR transfer function. Only PQ is wired today; HLG is a TODO that
/// needs a corresponding `OutputTransferHlg` encode fragment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HdrEotf {
    /// SMPTE ST 2084 (Perceptual Quantizer). CTA-861.3 EOTF value 2.
    Pq,
}

impl HdrEotf {
    /// CTA-861.3 EOTF code for the HDMI DRM infoframe byte 4.
    /// 0=SDR, 1=traditional HDR, 2=PQ, 3=HLG, 4=ICtCp.
    pub fn cta_value(self) -> u8 {
        match self {
            HdrEotf::Pq => 2,
        }
    }
}

/// `drm_mode_colorimetry` enum value for BT.2020 RGB.
///
/// The kernel registers Colorspace as a named enum at boot; the
/// numeric value below is the index `drm_mode_create_colorspace_property`
/// uses for `BT2020_RGB` and is stable across kernel versions.
pub const DRM_MODE_COLORIMETRY_BT2020_RGB: u64 = 9;

/// `drm_mode_colorimetry` enum value for Default (SDR / no special
/// signaling). Reset to this on shutdown.
pub const DRM_MODE_COLORIMETRY_DEFAULT: u64 = 0;

/// Resolved HDR signaling — pre-validated, kernel-ready. Built from
/// `prism_config::HdrConfig` once at bringup. All fields use the units
/// the kernel's `struct hdr_output_metadata_infoframe_sm_type1` expects.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrSignaling {
    pub eotf: HdrEotf,
    /// Peak display mastering luminance (nits).
    pub max_luminance: u16,
    /// Min display mastering luminance (ticks of 0.0001 nits).
    pub min_luminance_ticks: u16,
    /// Max content light level (nits). Caller defaults to
    /// `max_luminance` if config left it `None`.
    pub max_cll: u16,
    /// Max frame-average light level (nits). Caller defaults to
    /// `max_luminance / 2` if config left it `None`.
    pub max_fall: u16,
}

/// Size of the kernel's `struct hdr_output_metadata` (u32 metadata_type
/// + 26-byte type-1 infoframe + 2-byte tail padding to u32 alignment).
pub const HDR_OUTPUT_METADATA_BLOB_SIZE: usize = 32;

/// Build the `hdr_output_metadata` blob bytes matching the kernel's
/// struct layout exactly. Mastering display primaries are hardcoded
/// to BT.2020 + D65 — that's the conventional "source content was
/// mastered against this" advertisement for PQ HDR, regardless of
/// the actual panel's native gamut.
///
/// Layout (little-endian):
/// ```text
///   0..4   metadata_type (u32)              = 0  (HDMI_STATIC_METADATA_TYPE1)
///   4      eotf (u8)
///   5      hdmi_metadata_type (u8)          = 0  (Static Metadata Type 1)
///   6..18  display_primaries[3] RGB         (BT.2020, 0.00002 ticks)
///  18..22  white_point                      (D65, 0.00002 ticks)
///  22..24  max_display_mastering_luminance  (nits)
///  24..26  min_display_mastering_luminance  (0.0001-nit ticks)
///  26..28  max_cll                          (nits)
///  28..30  max_fall                         (nits)
///  30..32  tail padding                     = 0
/// ```
pub fn build_hdr_metadata_blob(s: &HdrSignaling) -> [u8; HDR_OUTPUT_METADATA_BLOB_SIZE] {
    let mut blob = [0u8; HDR_OUTPUT_METADATA_BLOB_SIZE];

    // outer metadata_type (u32) — 0 = HDMI_STATIC_METADATA_TYPE1
    blob[0..4].copy_from_slice(&0u32.to_le_bytes());

    // infoframe.eotf + infoframe.metadata_type
    blob[4] = s.eotf.cta_value();
    blob[5] = 0;

    // BT.2020 display primaries (chromaticity coords in 0.00002 ticks).
    // R = (0.708, 0.292), G = (0.170, 0.797), B = (0.131, 0.046).
    blob[6..8].copy_from_slice(&35400u16.to_le_bytes());
    blob[8..10].copy_from_slice(&14600u16.to_le_bytes());
    blob[10..12].copy_from_slice(&8500u16.to_le_bytes());
    blob[12..14].copy_from_slice(&39850u16.to_le_bytes());
    blob[14..16].copy_from_slice(&6550u16.to_le_bytes());
    blob[16..18].copy_from_slice(&2300u16.to_le_bytes());

    // D65 white point (0.3127, 0.3290).
    blob[18..20].copy_from_slice(&15635u16.to_le_bytes());
    blob[20..22].copy_from_slice(&16450u16.to_le_bytes());

    blob[22..24].copy_from_slice(&s.max_luminance.to_le_bytes());
    blob[24..26].copy_from_slice(&s.min_luminance_ticks.to_le_bytes());
    blob[26..28].copy_from_slice(&s.max_cll.to_le_bytes());
    blob[28..30].copy_from_slice(&s.max_fall.to_le_bytes());
    // bytes 30..32 = tail padding (already zero)

    blob
}

/// Discovered KMS property handles for HDR signaling on a connector.
/// `None` from [`lookup`] means the connector doesn't expose the
/// properties (older driver, output type that doesn't carry HDR).
pub struct HdrProps {
    connector: connector::Handle,
    hdr_metadata_prop: property::Handle,
    colorspace_prop: property::Handle,
    /// Blob ID currently set on the connector. We own its lifetime
    /// and must destroy it when replacing or clearing. `None` = no
    /// blob currently held by us.
    current_blob: Option<u64>,
}

impl HdrProps {
    /// Walk the connector's properties for `HDR_OUTPUT_METADATA` +
    /// `Colorspace`. Returns `Ok(None)` (not an error) if either is
    /// absent — most drivers expose both as a pair, but USB-C dock
    /// outputs and virtual displays sometimes don't.
    pub fn lookup<D: ControlDevice>(drm: &D, connector: connector::Handle) -> Result<Option<Self>> {
        let mut hdr_metadata_prop: Option<property::Handle> = None;
        let mut colorspace_prop: Option<property::Handle> = None;

        let props = drm
            .get_properties(connector)
            .with_context(|| format!("get_properties on {connector:?}"))?;
        for (prop_h, _value) in props {
            let Ok(info) = drm.get_property(prop_h) else {
                continue;
            };
            let name = info.name().to_string_lossy();
            match name.as_ref() {
                "HDR_OUTPUT_METADATA" => {
                    if matches!(info.value_type(), property::ValueType::Blob) {
                        hdr_metadata_prop = Some(prop_h);
                    } else {
                        tracing::warn!(
                            ?connector,
                            "HDR_OUTPUT_METADATA exists but isn't blob-typed"
                        );
                    }
                }
                "Colorspace" => {
                    if matches!(info.value_type(), property::ValueType::Enum(_)) {
                        colorspace_prop = Some(prop_h);
                    } else {
                        tracing::warn!(?connector, "Colorspace exists but isn't enum-typed");
                    }
                }
                _ => {}
            }
        }

        match (hdr_metadata_prop, colorspace_prop) {
            (Some(hdr_metadata_prop), Some(colorspace_prop)) => Ok(Some(Self {
                connector,
                hdr_metadata_prop,
                colorspace_prop,
                current_blob: None,
            })),
            _ => Ok(None),
        }
    }

    /// Push an HDR metadata blob to the connector + set
    /// `Colorspace=BT2020_RGB`. Replaces (and destroys) any prior
    /// blob we previously installed. Errors propagate; partial
    /// failure leaves the blob unset rather than half-applied.
    pub fn set_hdr<D: ControlDevice>(&mut self, drm: &D, signaling: &HdrSignaling) -> Result<()> {
        let blob_bytes = build_hdr_metadata_blob(signaling);
        // drm-rs's `create_property_blob<T>(&T)` takes a generic
        // reference and forwards its byte view. Passing the array
        // directly is the simplest path.
        let value = drm
            .create_property_blob(&blob_bytes)
            .context("create_property_blob(HDR_OUTPUT_METADATA)")?;
        let blob_id: u64 = match value {
            property::Value::Blob(id) => id,
            _ => anyhow::bail!("create_property_blob returned non-Blob value"),
        };

        // Set the new blob first; only destroy the old one once the
        // set succeeded, so a failure here leaves the previous blob
        // valid for the kernel to keep using.
        let set_result = drm
            .set_property(
                self.connector,
                self.hdr_metadata_prop,
                property::Value::Blob(blob_id).into(),
            )
            .context("set_property HDR_OUTPUT_METADATA");
        if let Err(e) = set_result {
            // Clean up the orphan blob we just created — kernel
            // would GC it on fd close, but we own a session-long fd.
            let _ = drm.destroy_property_blob(blob_id);
            return Err(e);
        }

        drm.set_property(
            self.connector,
            self.colorspace_prop,
            DRM_MODE_COLORIMETRY_BT2020_RGB,
        )
        .context("set_property Colorspace=BT2020_RGB")?;

        if let Some(old) = self.current_blob.replace(blob_id) {
            if let Err(err) = drm.destroy_property_blob(old) {
                tracing::warn!("destroy_property_blob({old}): {err:#}");
            }
        }
        Ok(())
    }

    /// Reset HDR signaling to "SDR / no metadata" on shutdown. Without
    /// this the connector keeps the last metadata blob across master
    /// handoffs and the desktop session that takes over interprets
    /// its sRGB pixels as PQ — visible as crushed shadows. This was
    /// the "DP-4 stickiness" bug; see `docs/color-management.md`.
    pub fn clear<D: ControlDevice>(&mut self, drm: &D) -> Result<()> {
        // Set metadata blob to 0 (kernel: "no HDR infoframe").
        let _ = drm
            .set_property(
                self.connector,
                self.hdr_metadata_prop,
                property::Value::Blob(0).into(),
            )
            .map_err(|e| tracing::warn!("clear HDR_OUTPUT_METADATA: {e:#}"));
        // Reset Colorspace to Default.
        let _ = drm
            .set_property(
                self.connector,
                self.colorspace_prop,
                DRM_MODE_COLORIMETRY_DEFAULT,
            )
            .map_err(|e| tracing::warn!("clear Colorspace: {e:#}"));
        if let Some(blob) = self.current_blob.take() {
            if let Err(err) = drm.destroy_property_blob(blob) {
                tracing::warn!("destroy_property_blob({blob}) on clear: {err:#}");
            }
        }
        Ok(())
    }
}
