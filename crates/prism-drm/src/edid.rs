//! EDID parsing → human-readable identifiers + HDR/color capabilities.
//!
//! Reads the connector's `EDID` blob property, hands it to
//! `libdisplay-info` for the high-level parse, and lifts a few fields
//! we care about into prism-friendly types:
//!
//!   - Make / model / serial — fed into `PhysicalProperties` so clients
//!     see real values via `wl_output` and so `output "Make Model
//!     Serial"` config matching (niri's `OutputName::matches`) works
//!     without the user having to type `DisplayPort-4`.
//!
//!   - Physical mm size — also feeds `PhysicalProperties` so DPI-aware
//!     clients can pick correct font sizes etc.
//!
//!   - HDR static metadata block — panel peak / min luminance, frame
//!     average max, supported transfer curves (PQ / HLG / SDR / HDR
//!     traditional). These become the *defaults* for the HDR feedforward
//!     enable; the user's `output "..." { color hdr ... }` block can
//!     override.
//!
//!   - Default color primaries — CIE 1931 xy for R, G, B and the
//!     default white point. Used for HDR_OUTPUT_METADATA's
//!     display_primaries / white_point fields when the user hasn't
//!     specified them.
//!
//! Returns `EdidInfo::default()` (everything `None`) on parse failure
//! or absent EDID — bringup must not depend on EDID being present.
//! Some DRM connectors (virtual displays, projectors with broken
//! EDIDs) just don't have one.

use anyhow::{Context, Result};
use smithay::backend::drm::DrmDevice;
use smithay::reexports::drm::control::{Device as ControlDevice, connector};

/// Parsed EDID summary. Every field is optional so a missing-EDID
/// connector still produces a value the rest of the codebase can use
/// (defaulting through `EdidInfo::default()`).
#[derive(Debug, Default, Clone)]
pub struct EdidInfo {
    pub make: Option<String>,
    pub model: Option<String>,
    pub serial: Option<String>,
    /// Physical panel size in millimeters (width, height). From the
    /// connector — kernel parses it out of the EDID's basic display
    /// parameters block. `None` for sources with no EDID or with a
    /// 0x0 size field.
    pub size_mm: Option<(u32, u32)>,
    /// HDR static metadata block from the EDID (CTA-861 HDR Static
    /// Metadata Data Block). Absent → display doesn't advertise HDR.
    pub hdr: Option<HdrCapabilities>,
    /// Default color primaries the panel advertises. Often clipped
    /// from the actual gamut for compatibility reasons — niri's notes
    /// say "may not be display's physical primaries, but only the
    /// primaries of the default RGB colorimetry signal." Good enough
    /// for HDR signaling defaults.
    pub primaries: Option<ColorPrimaries>,
}

/// HDR capabilities advertised by the panel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HdrCapabilities {
    /// Supports SMPTE ST 2084 (PQ) EOTF.
    pub pq: bool,
    /// Supports Hybrid Log-Gamma (HLG) EOTF.
    pub hlg: bool,
    /// Supports the "traditional gamma — HDR luminance range" EOTF.
    /// Distinct from `traditional_sdr` (which is the SDR gamma curve).
    pub traditional_hdr: bool,
    /// Peak luminance the panel claims (nits). Often optimistic on
    /// consumer panels — represents the desired-content max, not what
    /// the panel sustains.
    pub max_luminance_nits: Option<f32>,
    /// Frame-average max luminance (nits). Most relevant for OLED
    /// where the panel power-limits brightness over a full white
    /// frame to avoid burn-in / supply current limits.
    pub max_frame_avg_luminance_nits: Option<f32>,
    /// Minimum luminance (nits). OLEDs claim ~0; LCDs ~0.05.
    pub min_luminance_nits: Option<f32>,
}

/// Color primaries + white point in CIE 1931 xy chromaticity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorPrimaries {
    pub red: (f32, f32),
    pub green: (f32, f32),
    pub blue: (f32, f32),
    pub white: (f32, f32),
}

impl EdidInfo {
    /// Read + parse the connector's EDID blob. Returns a fully-empty
    /// `EdidInfo` if there's no EDID property, the blob fails to
    /// parse, or any individual field is absent — failure is logged
    /// at debug/warn and bringup continues.
    pub fn read(drm: &DrmDevice, connector_handle: connector::Handle) -> EdidInfo {
        let info_opt = match read_edid_blob(drm, connector_handle) {
            Ok(blob) => match libdisplay_info::info::Info::parse_edid(&blob) {
                Ok(info) => Some(info),
                Err(e) => {
                    tracing::debug!("EDID parse failed for {connector_handle:?}: {e}");
                    None
                }
            },
            Err(e) => {
                tracing::debug!("no EDID for {connector_handle:?}: {e:#}");
                None
            }
        };

        // Physical size always comes from the connector — drm-rs
        // exposes the field, kernel-parsed.
        let size_mm = connector_size_mm(drm, connector_handle);

        let Some(info) = info_opt else {
            return EdidInfo {
                size_mm,
                ..EdidInfo::default()
            };
        };

        let hdr = parse_hdr(&info);
        let primaries = parse_primaries(&info);

        EdidInfo {
            make: info.make(),
            model: info.model(),
            serial: info.serial(),
            size_mm,
            hdr,
            primaries,
        }
    }

    /// One-line summary for log output during bringup. Empty fields
    /// are elided so the line stays short when the EDID is sparse.
    pub fn log_line(&self) -> String {
        let mut parts = Vec::with_capacity(6);
        if let Some(m) = &self.make {
            parts.push(format!("make={m:?}"));
        }
        if let Some(m) = &self.model {
            parts.push(format!("model={m:?}"));
        }
        if let Some(s) = &self.serial {
            parts.push(format!("serial={s:?}"));
        }
        if let Some((w, h)) = self.size_mm {
            parts.push(format!("size={w}x{h}mm"));
        }
        if let Some(hdr) = self.hdr {
            let mut tfs = Vec::with_capacity(3);
            if hdr.pq {
                tfs.push("PQ");
            }
            if hdr.hlg {
                tfs.push("HLG");
            }
            if hdr.traditional_hdr {
                tfs.push("trad-HDR");
            }
            let tfs = if tfs.is_empty() {
                "none".to_owned()
            } else {
                tfs.join("+")
            };
            let peak = hdr
                .max_luminance_nits
                .map(|n| format!("{n:.0}nits"))
                .unwrap_or_else(|| "?".to_owned());
            parts.push(format!("hdr=[{tfs}, peak={peak}]"));
        }
        if let Some(p) = self.primaries {
            parts.push(format!(
                "primaries=[r({:.3},{:.3}) g({:.3},{:.3}) b({:.3},{:.3}) w({:.3},{:.3})]",
                p.red.0, p.red.1, p.green.0, p.green.1, p.blue.0, p.blue.1, p.white.0, p.white.1,
            ));
        }
        if parts.is_empty() {
            "no EDID".to_owned()
        } else {
            parts.join(" ")
        }
    }
}

fn read_edid_blob(drm: &DrmDevice, connector_handle: connector::Handle) -> Result<Vec<u8>> {
    let props = drm
        .get_properties(connector_handle)
        .context("get_properties")?;
    for (prop_h, value) in props {
        let info = drm.get_property(prop_h).context("get_property")?;
        if info.name().to_string_lossy() != "EDID" {
            continue;
        }
        let blob = info
            .value_type()
            .convert_value(value)
            .as_blob()
            .context("EDID property not blob-typed")?;
        let data = drm
            .get_property_blob(blob)
            .context("get_property_blob(EDID)")?;
        return Ok(data);
    }
    anyhow::bail!("connector has no EDID property")
}

fn connector_size_mm(drm: &DrmDevice, connector_handle: connector::Handle) -> Option<(u32, u32)> {
    let info = drm.get_connector(connector_handle, false).ok()?;
    let (w, h) = info.size()?;
    if w == 0 || h == 0 {
        return None;
    }
    Some((w, h))
}

fn parse_hdr(info: &libdisplay_info::info::Info) -> Option<HdrCapabilities> {
    let meta = info.hdr_static_metadata();
    // If the panel doesn't claim *any* HDR-related transfer curve we
    // treat it as "no HDR" rather than "HDR with all-false flags"
    // — keeps downstream code's "if let Some(hdr) = …" simple.
    if !(meta.pq || meta.hlg || meta.traditional_hdr) {
        return None;
    }
    let zero_to_none = |v: f32| (v > 0.0).then_some(v);
    Some(HdrCapabilities {
        pq: meta.pq,
        hlg: meta.hlg,
        traditional_hdr: meta.traditional_hdr,
        max_luminance_nits: zero_to_none(meta.desired_content_max_luminance),
        max_frame_avg_luminance_nits: zero_to_none(meta.desired_content_max_frame_avg_luminance),
        min_luminance_nits: zero_to_none(meta.desired_content_min_luminance),
    })
}

fn parse_primaries(info: &libdisplay_info::info::Info) -> Option<ColorPrimaries> {
    let p = info.default_color_primaries();
    if !p.has_primaries || !p.has_default_white_point {
        return None;
    }
    Some(ColorPrimaries {
        red: (p.primary[0].x, p.primary[0].y),
        green: (p.primary[1].x, p.primary[1].y),
        blue: (p.primary[2].x, p.primary[2].y),
        white: (p.default_white.x, p.default_white.y),
    })
}
