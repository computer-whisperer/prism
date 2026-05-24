use std::str::FromStr;

use knuffel::ast::SpannedNode;
use knuffel::decode::Context;
use knuffel::errors::DecodeError;
use knuffel::traits::ErrorSpan;
use knuffel::Decode;
use prism_ipc::{ConfiguredMode, HSyncPolarity, Transform, VSyncPolarity};

use crate::gestures::HotCorners;
use crate::{Color, FloatOrInt, LayoutPart};

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Outputs(pub Vec<Output>);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mode {
    pub custom: bool,
    pub mode: ConfiguredMode,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Modeline {
    /// The rate at which pixels are drawn in MHz.
    pub clock: f64,
    /// Horizontal active pixels.
    pub hdisplay: u16,
    /// Horizontal sync pulse start position in pixels.
    pub hsync_start: u16,
    /// Horizontal sync pulse end position in pixels.
    pub hsync_end: u16,
    /// Total horizontal number of pixels before resetting the horizontal drawing position to
    /// zero.
    pub htotal: u16,

    /// Vertical active pixels.
    pub vdisplay: u16,
    /// Vertical sync pulse start position in pixels.
    pub vsync_start: u16,
    /// Vertical sync pulse end position in pixels.
    pub vsync_end: u16,
    /// Total vertical number of pixels before resetting the vertical drawing position to zero.
    pub vtotal: u16,
    /// Horizontal sync polarity: "+hsync" or "-hsync".
    pub hsync_polarity: prism_ipc::HSyncPolarity,
    /// Vertical sync polarity: "+vsync" or "-vsync".
    pub vsync_polarity: prism_ipc::VSyncPolarity,
}

#[derive(knuffel::Decode, Debug, Clone, PartialEq)]
pub struct Output {
    #[knuffel(child)]
    pub off: bool,
    #[knuffel(argument)]
    pub name: String,
    #[knuffel(child, unwrap(argument))]
    pub scale: Option<FloatOrInt<0, 10>>,
    #[knuffel(child, unwrap(argument, str), default = Transform::Normal)]
    pub transform: Transform,
    #[knuffel(child)]
    pub position: Option<Position>,
    #[knuffel(child)]
    pub mode: Option<Mode>,
    #[knuffel(child)]
    pub modeline: Option<Modeline>,
    #[knuffel(child)]
    pub variable_refresh_rate: Option<Vrr>,
    #[knuffel(child)]
    pub focus_at_startup: bool,
    // Deprecated; use layout.background_color.
    #[knuffel(child)]
    pub background_color: Option<Color>,
    #[knuffel(child)]
    pub backdrop_color: Option<Color>,
    #[knuffel(child)]
    pub hot_corners: Option<HotCorners>,
    #[knuffel(child)]
    pub layout: Option<LayoutPart>,
    #[knuffel(child)]
    pub color: Option<ColorConfig>,
}

impl Output {
    pub fn is_vrr_always_on(&self) -> bool {
        self.variable_refresh_rate == Some(Vrr { on_demand: false })
    }

    pub fn is_vrr_on_demand(&self) -> bool {
        self.variable_refresh_rate == Some(Vrr { on_demand: true })
    }

    pub fn is_vrr_always_off(&self) -> bool {
        self.variable_refresh_rate.is_none()
    }
}

impl Default for Output {
    fn default() -> Self {
        Self {
            off: false,
            focus_at_startup: false,
            name: String::new(),
            scale: None,
            transform: Transform::Normal,
            position: None,
            mode: None,
            modeline: None,
            variable_refresh_rate: None,
            background_color: None,
            backdrop_color: None,
            hot_corners: None,
            layout: None,
            color: None,
        }
    }
}

/// Per-output color management knobs.
///
/// Currently exposes:
///  - CRTC `CTM` (3x3 post-blend calibration matrix; gamma-encoded coefficients
///    expected since amdgpu has no DEGAMMA_LUT at the CRTC).
///  - Connector `max bpc`: request a deeper link to the panel (8 / 10 / 12).
///    Values outside the connector's advertised range are silently clamped by
///    the kernel during EDID-aware mode negotiation.
///  - Connector HDR signaling (`HDR_OUTPUT_METADATA` + `Colorspace`). NOTE
///    (slice C1): the render pipeline still emits sRGB-encoded content, so
///    setting this on a panel will visibly dim SDR content (crushed into the
///    bottom of the PQ signal range). Useful as a signaling probe before the
///    shader-side OETF lands.
#[derive(knuffel::Decode, Debug, Clone, PartialEq, Default)]
pub struct ColorConfig {
    #[knuffel(child)]
    pub ctm: Option<Ctm>,
    #[knuffel(child, unwrap(argument))]
    pub max_bpc: Option<u32>,
    #[knuffel(child)]
    pub hdr: Option<HdrConfig>,
    /// What absolute luminance (cd/m²) "SDR white" maps to for
    /// color-unaware clients on this output. Only meaningful when the
    /// output is in HDR mode (no effect on SDR scanout, which always
    /// uses the IEC sRGB 80 cd/m² convention). Color-managed clients
    /// (those that attach a `wp_color_management_v1` description) are
    /// unaffected — they declare their own absolute luminance via the
    /// protocol.
    ///
    /// Defaults to 80 (= IEC sRGB spec). BT.2408's "HDR reference
    /// white" is 203; Windows lets users pick 80–450 via the
    /// per-display "SDR content brightness" slider. Higher values
    /// brighten ordinary desktop/SDR content in HDR mode; lower values
    /// dim it.
    #[knuffel(child, unwrap(argument))]
    pub sdr_reference_nits: Option<f64>,
    /// Per-channel response correction. The encoder inverts the
    /// panel's measured response (`emitted = gain * commanded^gamma`)
    /// so commanded nits match emitted nits. Derive from a tristim
    /// sweep — fit `gain` and `gamma` per channel to the measured-vs-
    /// requested curve. Identity (gain=1, gamma=1) when unset.
    #[knuffel(child)]
    pub response_curve: Option<ResponseCurve>,
    /// Per-channel measured panel peak luminance, in cd/m². Each
    /// subpixel of a real panel has its own emission ceiling — OLED
    /// ABL allocates power per subpixel; LCD color-filter transmission
    /// varies per primary. The decoder's display-referred clamp uses
    /// these as per-channel ceilings so the f32 intermediate stays
    /// inside the panel's realizable range. When unset, derived from
    /// `hdr.max_luminance` (HDR mode) or `sdr_reference_nits` (SDR
    /// mode) broadcast to all three channels — a conservative guess
    /// that the calibration tool replaces with measured values.
    #[knuffel(child)]
    pub panel_peak_nits: Option<PanelPeakNits>,
    /// Path to a binary 3D LUT file produced by `prism-tune calibrate`.
    /// When present, the compositor loads the file at bringup and
    /// pushes the entries directly into the encode pipeline's LUT
    /// texture — bypasses the (CTM + response-curve) → LUT synthesis
    /// path. The file is the canonical output of the measurement-driven
    /// calibration loop and carries per-grid-point precision the
    /// closed-form (CTM, gain/gamma) model can't represent.
    ///
    /// Path is resolved as given: absolute, or relative to prism's CWD.
    /// File format documented in `prism-renderer::lut3d::LutFileHeader`.
    /// If both `lut3d` and `(ctm | response-curve)` are configured, the
    /// LUT file wins; the others stay readable for fallback if the
    /// file load fails.
    #[knuffel(child)]
    pub lut3d: Option<Lut3dFile>,
}

/// Per-output 3D LUT file reference. See [`ColorConfig::lut3d`].
#[derive(knuffel::Decode, Debug, Clone, PartialEq)]
pub struct Lut3dFile {
    /// Path to the `.lut` file. Absolute or relative to prism's CWD.
    #[knuffel(argument)]
    pub path: String,
}

/// Per-channel measured panel peak luminance. See [`ColorConfig::panel_peak_nits`].
#[derive(knuffel::Decode, Debug, Clone, Copy, PartialEq)]
pub struct PanelPeakNits {
    /// Red-channel measured peak (cd/m²).
    #[knuffel(property)]
    pub r: f64,
    /// Green-channel measured peak (cd/m²).
    #[knuffel(property)]
    pub g: f64,
    /// Blue-channel measured peak (cd/m²).
    #[knuffel(property)]
    pub b: f64,
}

/// Per-channel response correction parameters. See [`ColorConfig::response_curve`].
#[derive(knuffel::Decode, Debug, Clone, PartialEq)]
pub struct ResponseCurve {
    /// Per-channel response gain. Identity = 1.0.
    #[knuffel(property, default = 1.0)]
    pub gain_r: f64,
    #[knuffel(property, default = 1.0)]
    pub gain_g: f64,
    #[knuffel(property, default = 1.0)]
    pub gain_b: f64,
    /// Per-channel response gamma exponent. Identity = 1.0.
    #[knuffel(property, default = 1.0)]
    pub gamma_r: f64,
    #[knuffel(property, default = 1.0)]
    pub gamma_g: f64,
    #[knuffel(property, default = 1.0)]
    pub gamma_b: f64,
}

/// 3×3 gamut-correction matrix in row-major order
/// (`r0c0 r0c1 r0c2  r1c0 r1c1 r1c2  r2c0 r2c1 r2c2`).
///
/// Applied in the encode shader as `panel_rgb = CTM * bt2020_rgb`, before
/// the per-channel response curve and OETF. Maps BT.2020 IR values into the
/// panel's native-primary linear nits — derived by calibration from
/// measured panel primaries plus the BT.2020 reference (D65) white point.
/// Identity is `1 0 0  0 1 0  0 0 1` (no gamut correction).
#[derive(knuffel::Decode, Debug, Clone, PartialEq)]
pub struct Ctm {
    #[knuffel(arguments)]
    pub values: Vec<f64>,
}

/// HDR signaling config. Maps onto the kernel's `hdr_output_metadata` blob plus
/// `Colorspace=BT2020_RGB`. All luminance values are in nits except where noted.
#[derive(knuffel::Decode, Debug, Clone, PartialEq)]
pub struct HdrConfig {
    /// Transfer function. Currently only `"pq"` (SMPTE ST 2084) is supported.
    #[knuffel(argument, str)]
    pub mode: HdrMode,
    /// Peak luminance the panel claims to reach (nits). Goes into
    /// `max_display_mastering_luminance`. Typical: 400 for HDR400 LCDs, 1000
    /// for HDR1000 displays, 600-800 for OLEDs.
    #[knuffel(property, default = 1000)]
    pub max_luminance: u32,
    /// Black-level the panel claims (nits). Goes into
    /// `min_display_mastering_luminance` as ticks of 0.0001 nits. Typical:
    /// ~0.05 for LCDs, ~0.0005 for OLEDs.
    #[knuffel(property, default = 0.05)]
    pub min_luminance: f64,
    /// Maximum content light level (nits). Defaults to `max_luminance`.
    #[knuffel(property)]
    pub max_cll: Option<u32>,
    /// Maximum frame-average light level (nits). Defaults to half
    /// `max_luminance`.
    #[knuffel(property)]
    pub max_fall: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrMode {
    /// SMPTE ST 2084 (PQ). EOTF value 2 in HDMI DRM infoframe.
    Pq,
}

impl FromStr for HdrMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pq" => Ok(HdrMode::Pq),
            other => Err(format!("unknown HDR mode {other:?}, expected: pq")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OutputName {
    pub connector: String,
    pub make: Option<String>,
    pub model: Option<String>,
    pub serial: Option<String>,
}

#[derive(knuffel::Decode, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    #[knuffel(property)]
    pub x: i32,
    #[knuffel(property)]
    pub y: i32,
}

#[derive(knuffel::Decode, Debug, Clone, PartialEq, Default)]
pub struct Vrr {
    #[knuffel(property, default = false)]
    pub on_demand: bool,
}

impl FromIterator<Output> for Outputs {
    fn from_iter<T: IntoIterator<Item = Output>>(iter: T) -> Self {
        Self(Vec::from_iter(iter))
    }
}

impl Outputs {
    pub fn find(&self, name: &OutputName) -> Option<&Output> {
        self.0.iter().find(|o| block_matches_output(&o.name, name))
    }

    pub fn find_mut(&mut self, name: &OutputName) -> Option<&mut Output> {
        self.0
            .iter_mut()
            .find(|o| block_matches_output(&o.name, name))
    }
}

/// Whether the KDL `output "..."` block named `block_name` should
/// apply to the connector described by `output_name`. Single source
/// of truth for the three subsystems that needed to do this matching
/// before (bringup output_config, scanout pick, wl_output positioning),
/// each of which previously duplicated a connector-only variant of
/// this logic.
///
/// Tried in order:
///   1. Exact case-insensitive connector match OR EDID `<Make> <Model>
///      <Serial>` match (delegated to [`OutputName::matches`]).
///   2. Short-form alias expansion: `output "DP-4"` matches connector
///      `DisplayPort-4`; `output "HDMI-1"` matches `HDMI-A-1`. This
///      catches legacy configs written before the long kernel-name
///      convention; modern configs would use `DisplayPort-4` or the
///      EDID triple directly.
///
/// Returns false if `block_name` is in short-alias form for an
/// interface (`dp-`, `hdmi-`) that doesn't match the connector AND
/// no EDID match succeeds — i.e. the alias check is additive, never
/// substitutive for failed EDID checks.
pub fn block_matches_output(block_name: &str, output_name: &OutputName) -> bool {
    if output_name.matches(block_name) {
        return true;
    }
    // Alias expansion only matters for the short forms `DP-N` and
    // `HDMI-N` that legacy configs use. Anything else falls through
    // to "no match" — we already tried the full set above.
    let lc = block_name.to_lowercase();
    let expanded = if let Some(rest) = lc.strip_prefix("dp-") {
        format!("displayport-{rest}")
    } else if let Some(rest) = lc.strip_prefix("hdmi-") {
        format!("hdmi-a-{rest}")
    } else {
        return false;
    };
    output_name.matches(&expanded)
}

impl OutputName {
    pub fn from_ipc_output(output: &prism_ipc::Output) -> Self {
        Self {
            connector: output.name.clone(),
            make: (output.make != "Unknown").then(|| output.make.clone()),
            model: (output.model != "Unknown").then(|| output.model.clone()),
            serial: output.serial.clone(),
        }
    }

    /// Returns an output description matching what Smithay's `Output::new()` does.
    pub fn format_description(&self) -> String {
        format!(
            "{} - {} - {}",
            self.make.as_deref().unwrap_or("Unknown"),
            self.model.as_deref().unwrap_or("Unknown"),
            self.connector,
        )
    }

    /// Returns an output name that will match by make/model/serial or, if they are missing, by
    /// connector.
    pub fn format_make_model_serial_or_connector(&self) -> String {
        if self.make.is_none() && self.model.is_none() && self.serial.is_none() {
            self.connector.to_string()
        } else {
            self.format_make_model_serial()
        }
    }

    pub fn format_make_model_serial(&self) -> String {
        let make = self.make.as_deref().unwrap_or("Unknown");
        let model = self.model.as_deref().unwrap_or("Unknown");
        let serial = self.serial.as_deref().unwrap_or("Unknown");
        format!("{make} {model} {serial}")
    }

    pub fn matches(&self, target: &str) -> bool {
        // Match by connector.
        if target.eq_ignore_ascii_case(&self.connector) {
            return true;
        }

        // If no other fields are available, don't try to match by them.
        //
        // This is used by niri msg output.
        if self.make.is_none() && self.model.is_none() && self.serial.is_none() {
            return false;
        }

        // Match by "make model serial" with Unknown if something is missing.
        let make = self.make.as_deref().unwrap_or("Unknown");
        let model = self.model.as_deref().unwrap_or("Unknown");
        let serial = self.serial.as_deref().unwrap_or("Unknown");

        let Some(target_make) = target.get(..make.len()) else {
            return false;
        };
        let rest = &target[make.len()..];
        if !target_make.eq_ignore_ascii_case(make) {
            return false;
        }
        if !rest.starts_with(' ') {
            return false;
        }
        let rest = &rest[1..];

        let Some(target_model) = rest.get(..model.len()) else {
            return false;
        };
        let rest = &rest[model.len()..];
        if !target_model.eq_ignore_ascii_case(model) {
            return false;
        }
        if !rest.starts_with(' ') {
            return false;
        }

        let rest = &rest[1..];
        if !rest.eq_ignore_ascii_case(serial) {
            return false;
        }

        true
    }

    // Similar in spirit to Ord, but I don't want to derive Eq to avoid mistakes (you should use
    // `Self::match`, not Eq).
    pub fn compare(&self, other: &Self) -> std::cmp::Ordering {
        let self_missing_mms = self.make.is_none() && self.model.is_none() && self.serial.is_none();
        let other_missing_mms =
            other.make.is_none() && other.model.is_none() && other.serial.is_none();

        match (self_missing_mms, other_missing_mms) {
            (true, true) => self.connector.cmp(&other.connector),
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => self
                .make
                .cmp(&other.make)
                .then_with(|| self.model.cmp(&other.model))
                .then_with(|| self.serial.cmp(&other.serial))
                .then_with(|| self.connector.cmp(&other.connector)),
        }
    }
}

impl<S: ErrorSpan> knuffel::Decode<S> for Mode {
    fn decode_node(node: &SpannedNode<S>, ctx: &mut Context<S>) -> Result<Self, DecodeError<S>> {
        if let Some(type_name) = &node.type_name {
            ctx.emit_error(DecodeError::unexpected(
                type_name,
                "type name",
                "no type name expected for this node",
            ));
        }

        for child in node.children() {
            ctx.emit_error(DecodeError::unexpected(
                child,
                "node",
                format!("unexpected node `{}`", child.node_name.escape_default()),
            ));
        }

        let mut custom: Option<bool> = None;
        for (name, val) in &node.properties {
            match &***name {
                "custom" => {
                    if custom.is_some() {
                        ctx.emit_error(DecodeError::unexpected(
                            name,
                            "property",
                            "unexpected duplicate property `custom`",
                        ))
                    }
                    custom = Some(knuffel::traits::DecodeScalar::decode(val, ctx)?)
                }
                name_str => ctx.emit_error(DecodeError::unexpected(
                    node,
                    "property",
                    format!("unexpected property `{}`", name_str.escape_default()),
                )),
            }
        }
        let custom = custom.unwrap_or(false);

        let mut arguments = node.arguments.iter();
        let mode = if let Some(mode_str) = arguments.next() {
            let temp_mode: String = knuffel::traits::DecodeScalar::decode(mode_str, ctx)?;

            let res = ConfiguredMode::from_str(temp_mode.as_str()).and_then(|mode| {
                if custom {
                    if mode.refresh.is_none() {
                        return Err("no refresh rate found; required for custom mode");
                    } else if let Some(refresh) = mode.refresh {
                        if refresh <= 0. {
                            return Err("custom mode refresh rate must be > 0");
                        }
                    }
                }
                Ok(mode)
            });
            res.map_err(|err_msg| DecodeError::conversion(&mode_str.literal, err_msg))?
        } else {
            return Err(DecodeError::missing(node, "argument `mode` is required"));
        };

        if let Some(surplus) = arguments.next() {
            ctx.emit_error(DecodeError::unexpected(
                &surplus.literal,
                "argument",
                "unexpected argument",
            ))
        }

        Ok(Mode { custom, mode })
    }
}

macro_rules! ensure {
    ($cond:expr, $ctx:expr, $span:expr, $fmt:literal $($arg:tt)* ) => {
        if !$cond {
            $ctx.emit_error(DecodeError::Conversion {
                source: format!($fmt $($arg)*).into(),
                span: $span.literal.span().clone()
            });
        }
    };
}

impl<S: ErrorSpan> Decode<S> for Modeline {
    fn decode_node(node: &SpannedNode<S>, ctx: &mut Context<S>) -> Result<Self, DecodeError<S>> {
        if let Some(type_name) = &node.type_name {
            ctx.emit_error(DecodeError::unexpected(
                type_name,
                "type name",
                "no type name expected for this node",
            ));
        }

        for child in node.children() {
            ctx.emit_error(DecodeError::unexpected(
                child,
                "node",
                format!("unexpected node `{}`", child.node_name.escape_default()),
            ));
        }

        for span in node.properties.keys() {
            ctx.emit_error(DecodeError::unexpected(
                span,
                "node",
                format!("unexpected node `{}`", span.escape_default()),
            ));
        }

        let mut arguments = node.arguments.iter();

        macro_rules! m_required {
            // This could be one identifier if macro_metavar_expr_concat stabilizes
            ($field:ident, $value_field:ident) => {
                let $value_field = arguments.next().ok_or_else(|| {
                    DecodeError::missing(node, format!("missing {} argument", stringify!($value)))
                })?;
                let $field = knuffel::traits::DecodeScalar::decode($value_field, ctx)?;
            };
        }

        m_required!(clock, clock_value);
        m_required!(hdisplay, hdisplay_value);
        m_required!(hsync_start, hsync_start_value);
        m_required!(hsync_end, hsync_end_value);
        m_required!(htotal, htotal_value);
        m_required!(vdisplay, vdisplay_value);
        m_required!(vsync_start, vsync_start_value);
        m_required!(vsync_end, vsync_end_value);
        m_required!(vtotal, vtotal_value);
        m_required!(hsync_polarity, hsync_polarity_value);
        let hsync_polarity =
            HSyncPolarity::from_str(String::as_str(&hsync_polarity)).map_err(|msg| {
                DecodeError::Conversion {
                    span: hsync_polarity_value.literal.span().clone(),
                    source: msg.into(),
                }
            })?;

        m_required!(vsync_polarity, vsync_polarity_value);
        let vsync_polarity =
            VSyncPolarity::from_str(String::as_str(&vsync_polarity)).map_err(|msg| {
                DecodeError::Conversion {
                    span: vsync_polarity_value.literal.span().clone(),
                    source: msg.into(),
                }
            })?;

        ensure!(
            hdisplay < hsync_start,
            ctx,
            hdisplay_value,
            "hdisplay {} must be < hsync_start {}",
            hdisplay,
            hsync_start
        );
        ensure!(
            hsync_start < hsync_end,
            ctx,
            hsync_start_value,
            "hsync_start {} must be < hsync_end {}",
            hsync_start,
            hsync_end,
        );
        ensure!(
            hsync_end < htotal,
            ctx,
            hsync_end_value,
            "hsync_end {} must be < htotal {}",
            hsync_end,
            htotal,
        );
        ensure!(
            0u16 < htotal,
            ctx,
            htotal_value,
            "htotal {} must be > 0",
            htotal
        );
        ensure!(
            vdisplay < vsync_start,
            ctx,
            vdisplay_value,
            "vdisplay {} must be < vsync_start {}",
            vdisplay,
            vsync_start,
        );
        ensure!(
            vsync_start < vsync_end,
            ctx,
            vsync_start_value,
            "vsync_start {} must be < vsync_end {}",
            vsync_start,
            vsync_end,
        );
        ensure!(
            vsync_end < vtotal,
            ctx,
            vsync_end_value,
            "vsync_end {} must be < vtotal {}",
            vsync_end,
            vtotal,
        );
        ensure!(
            0u16 < vtotal,
            ctx,
            vtotal_value,
            "vtotal {} must be > 0",
            vtotal
        );

        if let Some(extra) = arguments.next() {
            ctx.emit_error(DecodeError::unexpected(
                &extra.literal,
                "argument",
                "unexpected argument, all possible arguments were already provided",
            ))
        }

        Ok(Modeline {
            clock,
            hdisplay,
            hsync_start,
            hsync_end,
            htotal,
            vdisplay,
            vsync_start,
            vsync_end,
            vtotal,
            hsync_polarity,
            vsync_polarity,
        })
    }
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot;

    use super::*;

    #[test]
    fn parse_mode() {
        assert_eq!(
            "2560x1600@165.004".parse::<ConfiguredMode>().unwrap(),
            ConfiguredMode {
                width: 2560,
                height: 1600,
                refresh: Some(165.004),
            },
        );

        assert_eq!(
            "1920x1080".parse::<ConfiguredMode>().unwrap(),
            ConfiguredMode {
                width: 1920,
                height: 1080,
                refresh: None,
            },
        );

        assert!("1920".parse::<ConfiguredMode>().is_err());
        assert!("1920x".parse::<ConfiguredMode>().is_err());
        assert!("1920x1080@".parse::<ConfiguredMode>().is_err());
        assert!("1920x1080@60Hz".parse::<ConfiguredMode>().is_err());
    }

    fn make_output_name(
        connector: &str,
        make: Option<&str>,
        model: Option<&str>,
        serial: Option<&str>,
    ) -> OutputName {
        OutputName {
            connector: connector.to_string(),
            make: make.map(|x| x.to_string()),
            model: model.map(|x| x.to_string()),
            serial: serial.map(|x| x.to_string()),
        }
    }

    #[test]
    fn test_output_name_match() {
        fn check(
            target: &str,
            connector: &str,
            make: Option<&str>,
            model: Option<&str>,
            serial: Option<&str>,
        ) -> bool {
            let name = make_output_name(connector, make, model, serial);
            name.matches(target)
        }

        assert!(check("dp-2", "DP-2", None, None, None));
        assert!(!check("dp-1", "DP-2", None, None, None));
        assert!(check("dp-2", "DP-2", Some("a"), Some("b"), Some("c")));
        assert!(check(
            "some company some monitor 1234",
            "DP-2",
            Some("Some Company"),
            Some("Some Monitor"),
            Some("1234")
        ));
        assert!(!check(
            "some other company some monitor 1234",
            "DP-2",
            Some("Some Company"),
            Some("Some Monitor"),
            Some("1234")
        ));
        assert!(!check(
            "make model serial ",
            "DP-2",
            Some("make"),
            Some("model"),
            Some("serial")
        ));
        assert!(check(
            "make  serial",
            "DP-2",
            Some("make"),
            Some(""),
            Some("serial")
        ));
        assert!(check(
            "make model unknown",
            "DP-2",
            Some("Make"),
            Some("Model"),
            None
        ));
        assert!(check(
            "unknown unknown serial",
            "DP-2",
            None,
            None,
            Some("Serial")
        ));
        assert!(!check("unknown unknown unknown", "DP-2", None, None, None));
    }

    /// `block_matches_output` is the superset matcher used by every
    /// bringup/scanout/protocol output-config lookup. Three behaviors:
    /// long-form connector match, short-alias expansion (legacy
    /// `output "DP-4"` ⇒ kernel `DisplayPort-4`), and EDID
    /// `Make Model Serial` match for portable per-monitor calibration.
    #[test]
    fn block_matches_long_short_and_edid() {
        let lu28r55 = make_output_name(
            "DisplayPort-4",
            Some("Samsung Electric Company"),
            Some("LU28R55"),
            Some("HCJT603937"),
        );

        // Long-form connector (kernel name verbatim).
        assert!(block_matches_output("DisplayPort-4", &lu28r55));
        assert!(block_matches_output("displayport-4", &lu28r55));

        // Short alias — legacy convenience for `output "DP-4"`.
        assert!(block_matches_output("DP-4", &lu28r55));
        assert!(block_matches_output("dp-4", &lu28r55));

        // EDID-keyed (the form `prism-tune calibrate-lut3d` writes).
        assert!(block_matches_output(
            "Samsung Electric Company LU28R55 HCJT603937",
            &lu28r55,
        ));
        assert!(block_matches_output(
            "samsung electric company lu28r55 hcjt603937",
            &lu28r55,
        ));

        // Different connector + different serial: no match. Catches
        // accidentally too-loose matching that would smear calibration
        // across same-model units.
        assert!(!block_matches_output("DisplayPort-6", &lu28r55));
        assert!(!block_matches_output(
            "Samsung Electric Company LU28R55 HCJT507693",
            &lu28r55,
        ));

        // Unrelated block name: not matched.
        assert!(!block_matches_output("HDMI-1", &lu28r55));
        assert!(!block_matches_output("eDP-1", &lu28r55));
    }

    /// EDID-incomplete output (e.g., a dock without an EDID block):
    /// only the connector paths can fire, EDID is never used. Make
    /// sure the matcher doesn't accidentally produce false positives
    /// when EDID fields are None.
    #[test]
    fn block_matches_without_edid() {
        let no_edid = make_output_name("DisplayPort-2", None, None, None);
        assert!(block_matches_output("DisplayPort-2", &no_edid));
        assert!(block_matches_output("DP-2", &no_edid));
        assert!(!block_matches_output("Samsung X Y", &no_edid));
        assert!(!block_matches_output("unknown unknown unknown", &no_edid));
    }

    #[test]
    fn test_output_name_sorting() {
        let mut names = vec![
            make_output_name("DP-2", None, None, None),
            make_output_name("DP-1", None, None, None),
            make_output_name("DP-3", Some("B"), Some("A"), Some("A")),
            make_output_name("DP-3", Some("A"), Some("B"), Some("A")),
            make_output_name("DP-3", Some("A"), Some("A"), Some("B")),
            make_output_name("DP-3", None, Some("A"), Some("A")),
            make_output_name("DP-3", Some("A"), None, Some("A")),
            make_output_name("DP-3", Some("A"), Some("A"), None),
            make_output_name("DP-5", Some("A"), Some("A"), Some("A")),
            make_output_name("DP-4", Some("A"), Some("A"), Some("A")),
        ];
        names.sort_by(|a, b| a.compare(b));
        let names = names
            .into_iter()
            .map(|name| {
                format!(
                    "{} | {}",
                    name.format_make_model_serial_or_connector(),
                    name.connector,
                )
            })
            .collect::<Vec<_>>();
        assert_debug_snapshot!(
            names,
            @r#"
        [
            "Unknown A A | DP-3",
            "A Unknown A | DP-3",
            "A A Unknown | DP-3",
            "A A A | DP-4",
            "A A A | DP-5",
            "A A B | DP-3",
            "A B A | DP-3",
            "B A A | DP-3",
            "DP-1 | DP-1",
            "DP-2 | DP-2",
        ]
        "#
        );
    }
}
