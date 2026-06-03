//! Shared helpers between `calibrate` and `characterize` subcommands.
//!
//! All of this used to live in `calibrate.rs`. It got extracted here
//! when `characterize` started needing the same patch-surface lifecycle,
//! IPC primitives, and baseline-query plumbing. Functions are kept
//! deliberately simple — no clap arg structs in here, so each
//! subcommand can pass values from its own arg surface without
//! cross-coupling.

use crate::calibrate_lut3d::pq_oetf_f64;
use anyhow::{Context, Result};
use prism_ipc::socket::Socket;
use prism_ipc::{ColorState, OutputAction, Request, Response};
use std::collections::HashMap;
use tristim_display::{BufferFormat, DescriptionRequest, Luminances, Mastering, PatchSurface};

// ─── Channel ──────────────────────────────────────────────────────────────────

/// Channel identifier — used to pick a patch driver and to label CSV
/// rows. Order matches array indices everywhere: R=0, G=1, B=2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    R,
    G,
    B,
}

impl Channel {
    pub fn idx(self) -> usize {
        match self {
            Self::R => 0,
            Self::G => 1,
            Self::B => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::R => "R",
            Self::G => "G",
            Self::B => "B",
        }
    }

    pub const ALL: [Self; 3] = [Self::R, Self::G, Self::B];
}

// ─── Baseline ─────────────────────────────────────────────────────────────────

/// What the compositor reports about the output. The full per-output
/// `ColorState` from IPC, plus a one-shot snapshot taken at the start
/// of the run so subcommands can compare against the baseline.
pub struct OutputBaseline {
    pub hdr_active: bool,
    pub sdr_reference_nits: f64,
    /// Per-channel panel peak as the compositor sees it at run start.
    pub initial_panel_peak_nits: [f64; 3],
    /// Pre-existing response curve, if any. Reported for context.
    pub initial_response_curve: Option<([f64; 3], [f64; 3])>,
    /// EDID make ("Unknown" if absent). Mirrors `prism_ipc::Output.make`.
    pub make: Option<String>,
    /// EDID model ("Unknown" if absent).
    pub model: Option<String>,
    /// EDID serial number ("Display Product Serial Number" descriptor).
    pub serial: Option<String>,
}

impl OutputBaseline {
    /// EDID-derived identifier in the same shape `OutputName::matches`
    /// accepts: `"<Make> <Model> <Serial>"`. Returns `None` when any
    /// of the three is missing — without all three the identifier
    /// can't promise to pick out a single physical unit (multiple
    /// monitors of the same model would all match), and callers
    /// should fall back to the connector name.
    pub fn edid_identifier(&self) -> Option<String> {
        let make = self.make.as_deref()?;
        let model = self.model.as_deref()?;
        let serial = self.serial.as_deref()?;
        Some(format!("{make} {model} {serial}"))
    }

    /// Filesystem-safe variant of [`Self::edid_identifier`]. Spaces
    /// become dashes; path-unsafe chars become underscores. Suitable
    /// as a filename stem across Linux/macOS/Windows. Returns `None`
    /// when [`Self::edid_identifier`] does.
    pub fn edid_filename_stem(&self) -> Option<String> {
        self.edid_identifier().map(|s| sanitize_for_filename(&s))
    }
}

/// Replace whitespace with `-` and path-unsafe characters with `_`.
/// Lossy but predictable — two distinct EDID strings can't collide
/// unless they were textually identical to begin with.
pub fn sanitize_for_filename(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            ' ' | '\t' => '-',
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect()
}

/// Query the prism IPC for the named output's current color state.
pub fn query_output_baseline(name: &str) -> Result<OutputBaseline> {
    let mut socket = Socket::connect().context("connect to PRISM_SOCKET")?;
    let reply = socket.send(Request::Outputs).context("Request::Outputs")?;
    let outputs: HashMap<String, prism_ipc::Output> = match reply {
        Ok(Response::Outputs(map)) => map,
        Ok(other) => anyhow::bail!("unexpected reply to Outputs: {other:?}"),
        Err(e) => anyhow::bail!("prism returned error: {e}"),
    };
    let output = outputs.get(name).ok_or_else(|| {
        anyhow::anyhow!(
            "no output named {name:?} (connected: {:?})",
            outputs.keys().collect::<Vec<_>>()
        )
    })?;
    let ColorState {
        hdr_active,
        panel_peak_nits,
        sdr_reference_nits,
        response_curve,
        ctm: _,
        advertised_peak_nits: _,
    } = output.color;
    let initial_response_curve = response_curve.map(|c| (c.gain, c.gamma));
    // "Unknown" is prism IPC's sentinel for "EDID didn't carry this
    // field" — strip it out so `edid_identifier` returns None rather
    // than synthesizing a useless "Unknown Unknown Unknown" string.
    let make = (output.make != "Unknown").then(|| output.make.clone());
    let model = (output.model != "Unknown").then(|| output.model.clone());
    let serial = output.serial.clone();
    Ok(OutputBaseline {
        hdr_active,
        sdr_reference_nits,
        initial_panel_peak_nits: panel_peak_nits,
        initial_response_curve,
        make,
        model,
        serial,
    })
}

// ─── Patch surface lifecycle ──────────────────────────────────────────────────

/// Open an HDR or SDR patch surface on the chosen output, mode-aware.
///
/// **HDR**: fp16 buffer with a PQ + BT.2020 description and a generous
/// mastering envelope (10000 nits) so the patch nits aren't pre-clipped
/// by the descriptor before reaching the panel.
///
/// **SDR**: 8-bit xRGB buffer that *also* declares BT.2020 primaries
/// (with sRGB transfer + reference luminance = `sdr_reference_nits`).
/// Why not unmanaged sRGB: an unmanaged SDR buffer is interpreted as
/// sRGB-primary, so `cv = (1, 0, 0)` routes through the decode pass's
/// sRGB→BT.2020 chromaticity remap and hits the panel as a mix
/// (~0.63 R + 0.07 G + 0.02 B in BT.2020), which the panel renders at
/// well under its native red saturation. Declaring BT.2020 primaries
/// makes the decode CTM identity, so `cv = (1, 0, 0)` drives the
/// panel's native R purely — same shape as the HDR probe. The probe
/// then measures the panel's actual reach, not the sRGB cube's image
/// under prism's pipeline.
pub fn open_patch_surface(output: &str, baseline: &OutputBaseline) -> Result<PatchSurface> {
    if baseline.hdr_active {
        let probe_peak = 10_000.0;
        let desc = DescriptionRequest {
            transfer_function: "st2084_pq".into(),
            primaries: "bt2020".into(),
            luminances: None,
            mastering: Some(Mastering {
                min_nits: 0.0005,
                max_nits: probe_peak,
                max_cll_nits: probe_peak,
                max_fall_nits: probe_peak / 2.0,
            }),
        };
        PatchSurface::open(output, BufferFormat::Xbgr16161616f, Some(desc))
            .with_context(|| format!("open HDR patch on {output}"))
    } else {
        let ref_nits = baseline.sdr_reference_nits;
        let desc = DescriptionRequest {
            transfer_function: "srgb".into(),
            primaries: "bt2020".into(),
            luminances: Some(Luminances {
                min_nits: 0.0,
                max_nits: ref_nits,
                reference_nits: ref_nits,
            }),
            mastering: None,
        };
        PatchSurface::open(output, BufferFormat::Xrgb8888, Some(desc))
            .with_context(|| format!("open SDR patch (BT.2020-described) on {output}"))
    }
}

/// Encode a nits triple to the patch surface's `[0, 1]` code-value space
/// for the current mode. HDR uses PQ OETF (10000-nit peak); SDR uses
/// sRGB OETF anchored at `sdr_reference_nits`. The compositor decodes
/// from this same convention.
pub fn code_values_for_nits(baseline: &OutputBaseline, nits_rgb: [f64; 3]) -> [f64; 3] {
    if baseline.hdr_active {
        [
            pq_oetf_f64(nits_rgb[0].clamp(0.0, 10_000.0)),
            pq_oetf_f64(nits_rgb[1].clamp(0.0, 10_000.0)),
            pq_oetf_f64(nits_rgb[2].clamp(0.0, 10_000.0)),
        ]
    } else {
        let ref_nits = baseline.sdr_reference_nits.max(1e-6);
        [
            srgb_oetf((nits_rgb[0] / ref_nits).clamp(0.0, 1.0)),
            srgb_oetf((nits_rgb[1] / ref_nits).clamp(0.0, 1.0)),
            srgb_oetf((nits_rgb[2] / ref_nits).clamp(0.0, 1.0)),
        ]
    }
}

/// Drive the patch to black. Used at start/end of runs so the panel
/// isn't left glaring.
pub fn set_patch_off(patch: &mut PatchSurface, _hdr_active: bool) -> Result<()> {
    patch
        .set_code_values([0.0, 0.0, 0.0])
        .context("set black patch")
}

/// Configure the patch's surround colour at a fixed luminance, mode-aware.
pub fn apply_border(
    patch: &mut PatchSurface,
    baseline: &OutputBaseline,
    border_nits: f64,
) -> Result<()> {
    let cv = code_values_for_nits(baseline, [border_nits, border_nits, border_nits]);
    patch.set_border(cv).with_context(|| {
        format!(
            "set border to {:.2} cd/m² ({} mode)",
            border_nits,
            if baseline.hdr_active { "HDR" } else { "SDR" },
        )
    })
}

/// Paint a clearly-visible gray patch in the centred window so the
/// user can see where to place the colorimeter puck during the prep
/// countdown.
pub fn show_alignment_patch(
    patch: &mut PatchSurface,
    baseline: &OutputBaseline,
    alignment_nits: f64,
) -> Result<()> {
    let cv = code_values_for_nits(baseline, [alignment_nits, alignment_nits, alignment_nits]);
    patch
        .set_code_values(cv)
        .with_context(|| format!("alignment patch at {alignment_nits:.2} cd/m²"))
}

/// Drive a single-channel patch at `target_nits` on the named channel,
/// other channels at zero.
pub fn set_channel_patch(
    patch: &mut PatchSurface,
    baseline: &OutputBaseline,
    channel: Channel,
    target_nits: f64,
) -> Result<()> {
    let mut nits = [0.0_f64; 3];
    nits[channel.idx()] = target_nits;
    let cv = code_values_for_nits(baseline, nits);
    patch.set_code_values(cv).with_context(|| {
        format!(
            "set {} channel patch = {:.2} cd/m²",
            channel.label(),
            target_nits,
        )
    })
}

/// Drive an arbitrary RGB patch in panel-native nits. The 3D-sweep
/// calibration needs to command independent per-channel values
/// (cmd_R, cmd_G, cmd_B) — encoded to the surface's code-value space
/// via the same mode-aware helper as the single-channel and white
/// setters so the conventions stay consistent.
pub fn set_rgb_patch(
    patch: &mut PatchSurface,
    baseline: &OutputBaseline,
    cmd_rgb: [f64; 3],
) -> Result<()> {
    let cv = code_values_for_nits(baseline, cmd_rgb);
    patch.set_code_values(cv).with_context(|| {
        format!(
            "set RGB patch ({:.2}, {:.2}, {:.2}) cd/m²",
            cmd_rgb[0], cmd_rgb[1], cmd_rgb[2],
        )
    })
}

/// Render BT.2020 D65 reference white at `target_nits` in the centred
/// patch. HDR: equal R/G/B in linear nits — BT.2020 is defined such that
/// equal R/G/B produces D65 by construction. SDR: equal sRGB-encoded
/// values where RGB=1.0 maps to `sdr_reference_nits`.
pub fn set_white_patch(
    patch: &mut PatchSurface,
    baseline: &OutputBaseline,
    target_nits: f64,
) -> Result<()> {
    let cv = code_values_for_nits(baseline, [target_nits, target_nits, target_nits]);
    patch
        .set_code_values(cv)
        .with_context(|| format!("set white patch = {target_nits:.2} cd/m²"))
}

/// sRGB OETF (linear → encoded). Inverse of the EOTF in the decode shader.
pub fn srgb_oetf(linear: f64) -> f64 {
    if linear <= 0.0031308 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

// ─── IPC primitives ───────────────────────────────────────────────────────────

/// Push per-channel panel-peak nits via IPC. Used to lift the IR clamp
/// (set to 10000) during free-range characterization, or apply
/// calibrated peaks afterward.
pub fn apply_panel_peaks(output: &str, peaks: [f64; 3]) -> Result<()> {
    send_action(
        output,
        OutputAction::PanelPeakNits {
            nits_r: peaks[0],
            nits_g: peaks[1],
            nits_b: peaks[2],
        },
    )
    .context("apply PanelPeakNits")
}

/// Fire a one-shot IPC request against the running prism. The server
/// closes after each reply, so each request opens a fresh connection.
pub fn send_action(output: &str, action: OutputAction) -> Result<()> {
    let mut socket = Socket::connect().context("connect to PRISM_SOCKET")?;
    let reply = socket
        .send(Request::Output {
            output: output.to_string(),
            action,
        })
        .context("send request / read reply")?;
    match reply {
        Ok(Response::OutputConfigChanged(_)) => Ok(()),
        Ok(other) => anyhow::bail!("unexpected reply: {other:?}"),
        Err(e) => anyhow::bail!("prism returned error: {e}"),
    }
}

/// Like [`send_action`] but returns the full `Response` to the caller.
/// Used by actions that ship measurement data back (e.g.
/// `OutputAction::EncodeDiagnose`).
pub fn send_action_for_reply(output: &str, action: OutputAction) -> Result<Response> {
    let mut socket = Socket::connect().context("connect to PRISM_SOCKET")?;
    let reply = socket
        .send(Request::Output {
            output: output.to_string(),
            action,
        })
        .context("send request / read reply")?;
    match reply {
        Ok(response) => Ok(response),
        Err(e) => anyhow::bail!("prism returned error: {e}"),
    }
}
