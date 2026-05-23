//! Shared helpers between `calibrate` and `characterize` subcommands.
//!
//! All of this used to live in `calibrate.rs`. It got extracted here
//! when `characterize` started needing the same patch-surface lifecycle,
//! IPC primitives, and baseline-query plumbing. Functions are kept
//! deliberately simple — no clap arg structs in here, so each
//! subcommand can pass values from its own arg surface without
//! cross-coupling.

use anyhow::{Context, Result};
use prism_ipc::socket::Socket;
use prism_ipc::{ColorState, OutputAction, Request, Response};
use std::collections::HashMap;
use tristim_display::{PatchSurface, PqDescriptionParams};

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
}

/// Query the prism IPC for the named output's current color state.
pub fn query_output_baseline(name: &str) -> Result<OutputBaseline> {
    let mut socket = Socket::connect().context("connect to PRISM_SOCKET")?;
    let reply = socket
        .send(Request::Outputs)
        .context("Request::Outputs")?;
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
    } = output.color;
    let initial_response_curve = response_curve.map(|c| (c.gain, c.gamma));
    Ok(OutputBaseline {
        hdr_active,
        sdr_reference_nits,
        initial_panel_peak_nits: panel_peak_nits,
        initial_response_curve,
    })
}

// ─── Patch surface lifecycle ──────────────────────────────────────────────────

/// Open an HDR or SDR patch surface on the chosen output, mode-aware.
/// For HDR, declares a generous mastering envelope (10000 nits) so the
/// patch buffer's nits aren't pre-clipped by the descriptor before
/// reaching the panel.
pub fn open_patch_surface(output: &str, hdr_active: bool) -> Result<PatchSurface> {
    if hdr_active {
        let probe_peak = 10_000;
        let params = PqDescriptionParams {
            mastering_min_lum_ticks: 5,
            mastering_max_lum: probe_peak,
            max_cll: probe_peak,
            max_fall: probe_peak / 2,
        };
        PatchSurface::open_hdr(output, params)
            .with_context(|| format!("open HDR patch on {output}"))
    } else {
        PatchSurface::open(output).with_context(|| format!("open SDR patch on {output}"))
    }
}

/// Drive the patch to black using the right setter for the current mode.
/// Used at start/end of runs so the panel isn't left glaring.
pub fn set_patch_off(patch: &mut PatchSurface, hdr_active: bool) -> Result<()> {
    if hdr_active {
        patch.set_nits([0.0, 0.0, 0.0]).context("set black (HDR)")
    } else {
        patch.set_color([0.0, 0.0, 0.0]).context("set black (SDR)")
    }
}

/// Configure the patch's surround colour at a fixed luminance, mode-aware.
pub fn apply_border(
    patch: &mut PatchSurface,
    baseline: &OutputBaseline,
    border_nits: f64,
) -> Result<()> {
    if baseline.hdr_active {
        patch
            .set_border_nits([border_nits, border_nits, border_nits])
            .context("set HDR border")
    } else {
        let linear = (border_nits / baseline.sdr_reference_nits).clamp(0.0, 1.0);
        let encoded = srgb_oetf(linear);
        patch
            .set_border_color([encoded, encoded, encoded])
            .context("set SDR border")
    }
}

/// Paint a clearly-visible gray patch in the centred window so the
/// user can see where to place the colorimeter puck during the prep
/// countdown.
pub fn show_alignment_patch(
    patch: &mut PatchSurface,
    baseline: &OutputBaseline,
    alignment_nits: f64,
) -> Result<()> {
    if baseline.hdr_active {
        patch
            .set_nits([alignment_nits, alignment_nits, alignment_nits])
            .context("alignment patch (HDR)")
    } else {
        let linear = (alignment_nits / baseline.sdr_reference_nits).clamp(0.0, 1.0);
        let encoded = srgb_oetf(linear);
        patch
            .set_color([encoded, encoded, encoded])
            .context("alignment patch (SDR)")
    }
}

/// Drive a single-channel patch using the right setter for the mode.
/// SDR uses sRGB OETF to convert target nits → RGB 0..=1.
pub fn set_channel_patch(
    patch: &mut PatchSurface,
    baseline: &OutputBaseline,
    channel: Channel,
    target_nits: f64,
) -> Result<()> {
    if baseline.hdr_active {
        let mut rgb = [0.0_f64; 3];
        rgb[channel.idx()] = target_nits;
        patch.set_nits(rgb).with_context(|| {
            format!("set HDR nits for {} = {:.2}", channel.label(), target_nits)
        })
    } else {
        let linear = (target_nits / baseline.sdr_reference_nits).clamp(0.0, 1.0);
        let encoded = srgb_oetf(linear);
        let mut rgb = [0.0_f64; 3];
        rgb[channel.idx()] = encoded;
        patch.set_color(rgb).with_context(|| {
            format!(
                "set SDR RGB for {} = {:.4} (target {:.2} cd/m²)",
                channel.label(),
                encoded,
                target_nits
            )
        })
    }
}

/// Render BT.2020 D65 reference white at `target_nits` in the centred
/// patch. HDR mode: `(R=L, G=L, B=L)` in linear nits — BT.2020 is
/// defined such that equal R/G/B produces D65 by construction. SDR mode:
/// convert to sRGB-encoded white where RGB=1.0 maps to
/// `sdr_reference_nits`.
pub fn set_white_patch(
    patch: &mut PatchSurface,
    baseline: &OutputBaseline,
    target_nits: f64,
) -> Result<()> {
    if baseline.hdr_active {
        patch
            .set_nits([target_nits, target_nits, target_nits])
            .with_context(|| format!("set HDR white = {target_nits:.2}"))
    } else {
        let linear = (target_nits / baseline.sdr_reference_nits).clamp(0.0, 1.0);
        let encoded = srgb_oetf(linear);
        patch.set_color([encoded, encoded, encoded]).with_context(|| {
            format!("set SDR white = {encoded:.4} (target {target_nits:.2} cd/m²)")
        })
    }
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
