//! Helpers for the scanout path: picking an output, building a framebuffer
//! from a GBM BO, and producing the `PlaneState` for an atomic commit.

use anyhow::{anyhow, Context, Result};
use drm_fourcc::DrmFourcc;
use gbm::BufferObject;
use prism_config::output::Output as OutputCfg;
use smithay::backend::drm::DrmDevice;
use smithay::output::Mode as WlMode;
use smithay::reexports::drm::buffer::PlanarBuffer;
use smithay::reexports::drm::control::{
    connector, crtc, framebuffer, property, Device as ControlDevice, FbCmd2Flags, Mode, ModeFlags,
    ModeTypeFlags, ResourceHandle,
};

/// Bit depth + format selection for a scanout BO. Picks the matching DRM
/// fourcc and the Vulkan format that interprets the same memory layout.
///
/// `Bpc8` → DRM `XR24` ↔ Vulkan `B8G8R8A8_UNORM`. Standard SDR scanout.
/// `Bpc10` → DRM `XR30` ↔ Vulkan `A2R10G10B10_UNORM_PACK32`. Higher
///   precision; required for HDR and for SDR-without-banding on smooth
///   gradients. Pair with `max_bpc=10` on the connector to actually push
///   10 bits over the wire (else driver dithers down).
/// `Fp16` → DRM `XB4F` (`XBGR16161616F`) ↔ Vulkan `R16G16B16A16_SFLOAT`.
///   16 bits per channel as half-floats; the only scanout format with
///   enough headroom for absolute-nits PQ encode (10000-nit peak). Used
///   for HDR-configured outputs. The kernel splits this back down to the
///   connector's negotiated link depth (8/10/12) on scanout.
///
/// Choice is per-output; some displays don't support 10-bit links or
/// fp16 framebuffers (cheap 1080p panels). Negotiation belongs in the
/// per-output config layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanoutDepth {
    Bpc8,
    Bpc10,
    Fp16,
}

impl ScanoutDepth {
    pub fn drm_fourcc(self) -> DrmFourcc {
        match self {
            Self::Bpc8 => DrmFourcc::Xrgb8888,
            Self::Bpc10 => DrmFourcc::Xrgb2101010,
            Self::Fp16 => DrmFourcc::Xbgr16161616f,
        }
    }

    /// The `max bpc` value to push to the connector for this depth.
    /// Fp16 stays at 10 because that's the highest most consumer
    /// HDR displays accept over DP/HDMI (12 is rare and the link
    /// negotiation will reject it on most panels).
    pub fn max_bpc(self) -> u64 {
        match self {
            Self::Bpc8 => 8,
            Self::Bpc10 | Self::Fp16 => 10,
        }
    }
}

/// One connected output's wiring choices for the tracer.
#[derive(Debug)]
pub struct OutputPick {
    pub connector: connector::Handle,
    pub mode: Mode,
    pub crtc: crtc::Handle,
    pub connector_name: String,
}

/// Pick a connected output: first connected connector with a preferred mode
/// and a compatible (currently-unused) CRTC. Good enough for the single-screen
/// scanout smoke test; the real compositor will allow user-driven assignments.
pub fn pick_first_connected(drm: &DrmDevice) -> Result<OutputPick> {
    pick_matching(drm, |_name| true, &[])
}

/// Pick a specific output by name. Accepts the full connector name
/// (`DisplayPort-6`) or the common short alias (`DP-6`, `HDMI-1`),
/// case-insensitively. Pass `&[]` for `outputs_cfg` when running outside the
/// integrated compositor (no KDL config to consult).
pub fn pick_by_name(drm: &DrmDevice, want: &str) -> Result<OutputPick> {
    pick_by_name_with_config(drm, want, &[])
}

/// Same as [`pick_by_name`] but also honors the config's `mode`/`off` for the
/// matched connector. `off` aborts with an explanatory error (since the user
/// explicitly asked for this connector by name).
pub fn pick_by_name_with_config(
    drm: &DrmDevice,
    want: &str,
    outputs_cfg: &[OutputCfg],
) -> Result<OutputPick> {
    let want_lc = want.to_lowercase();
    let want_normalized = expand_alias(&want_lc);
    pick_matching(
        drm,
        |name| {
            let lc = name.to_lowercase();
            lc == want_lc || lc == want_normalized
        },
        outputs_cfg,
    )
    .with_context(|| format!("no connected output matched {want:?}"))
}

/// Expand short-form connector aliases to the kernel-reported full names.
fn expand_alias(input: &str) -> String {
    if let Some(rest) = input.strip_prefix("dp-") {
        format!("displayport-{rest}")
    } else if let Some(rest) = input.strip_prefix("hdmi-") {
        format!("hdmi-a-{rest}")
    } else {
        input.to_string()
    }
}

fn pick_matching<F>(drm: &DrmDevice, matches: F, outputs_cfg: &[OutputCfg]) -> Result<OutputPick>
where
    F: Fn(&str) -> bool,
{
    let resources = drm.resource_handles().context("resource_handles")?;
    let occupied_by_other = collect_other_session_crtcs(drm, &resources);

    for &conn_h in resources.connectors() {
        let info = drm
            .get_connector(conn_h, false)
            .with_context(|| format!("get_connector {conn_h:?}"))?;
        if info.state() != connector::State::Connected {
            continue;
        }
        let name = format!("{:?}-{}", info.interface(), info.interface_id());
        if !matches(&name) {
            continue;
        }
        let edid = crate::EdidInfo::read(drm, conn_h);
        let cfg = match_config_for_connector(&name, &edid, outputs_cfg);
        let Some((mode, fallback)) = pick_mode(&info, cfg) else {
            continue;
        };
        if fallback {
            tracing::warn!("{name}: configured mode not available; falling back to preferred",);
        }
        let pick = resolve_pick(
            drm,
            &resources,
            conn_h,
            &info,
            &name,
            mode,
            &occupied_by_other,
            &[],
        )?;
        return Ok(pick);
    }
    Err(anyhow!("no connected connector with a usable mode + CRTC"))
}

/// Pick every connected connector with a usable mode + free CRTC.
/// No config consulted — used by the headless tracer paths.
pub fn pick_all_connected(drm: &DrmDevice) -> Result<Vec<OutputPick>> {
    pick_all_connected_with_config(drm, &[])
}

/// Same as [`pick_all_connected`] but honors per-output `off` and
/// `mode`/`modeline` from the KDL config.
///
/// Each successful pick reserves its CRTC against subsequent picks in the
/// same call, so two of our outputs can't accidentally collide on the same
/// CRTC. Connectors that can't be assigned (no free CRTC, no usable mode,
/// `off=true`, etc.) are skipped with a warning rather than aborting the
/// whole bringup — useful on hardware with more connectors than CRTCs (DP
/// MST splitters etc.) where partial bringup is the right answer.
///
/// `outputs_cfg` is matched against each connector's kernel name (e.g.
/// `DisplayPort-4`) case-insensitively, expanding short aliases like
/// `DP-4` to `DisplayPort-4` (mirrors what [`pick_by_name`] accepts).
///
/// Returns the picks in connector-order. Empty result is success-with-zero-
/// outputs (caller may treat that as an error of its own).
pub fn pick_all_connected_with_config(
    drm: &DrmDevice,
    outputs_cfg: &[OutputCfg],
) -> Result<Vec<OutputPick>> {
    let resources = drm.resource_handles().context("resource_handles")?;
    let occupied_by_other = collect_other_session_crtcs(drm, &resources);

    let mut picks: Vec<OutputPick> = Vec::new();
    for &conn_h in resources.connectors() {
        let info = match drm.get_connector(conn_h, false) {
            Ok(i) => i,
            Err(e) => {
                tracing::warn!("get_connector {conn_h:?} failed: {e:#}; skipping");
                continue;
            }
        };
        if info.state() != connector::State::Connected {
            continue;
        }
        let name = format!("{:?}-{}", info.interface(), info.interface_id());

        let edid = crate::EdidInfo::read(drm, conn_h);
        let cfg = match_config_for_connector(&name, &edid, outputs_cfg);
        if cfg.is_some_and(|c| c.off) {
            tracing::info!("{name}: config `off`; skipping");
            continue;
        }

        let Some((mode, fallback)) = pick_mode(&info, cfg) else {
            tracing::warn!("{name}: no usable mode; skipping");
            continue;
        };
        if fallback {
            tracing::warn!("{name}: configured mode not available; falling back to preferred",);
        }

        let used_by_us: Vec<crtc::Handle> = picks.iter().map(|p| p.crtc).collect();
        match resolve_pick(
            drm,
            &resources,
            conn_h,
            &info,
            &name,
            mode,
            &occupied_by_other,
            &used_by_us,
        ) {
            Ok(p) => {
                tracing::info!(
                    "{name}: assigned crtc {:?} mode={}x{}@{}Hz",
                    p.crtc,
                    p.mode.size().0,
                    p.mode.size().1,
                    p.mode.vrefresh()
                );
                picks.push(p);
            }
            Err(e) => {
                tracing::warn!("{name}: cannot bring up: {e:#}; skipping");
            }
        }
    }
    Ok(picks)
}

/// CRTCs currently bound to *other* sessions' connectors (a prior desktop
/// session usually leaves these bound to its assignments). Reusing one
/// would require atomically disabling the other connector's CRTC in the
/// same commit, which we don't do — the kernel rejects the test commit
/// with "Atomic Test failed for crtc X".
fn collect_other_session_crtcs(
    drm: &DrmDevice,
    resources: &smithay::reexports::drm::control::ResourceHandles,
) -> Vec<crtc::Handle> {
    let mut out = Vec::new();
    for &c in resources.connectors() {
        let Ok(info) = drm.get_connector(c, false) else {
            continue;
        };
        if info.state() != connector::State::Connected {
            continue;
        }
        let Some(enc_h) = info.current_encoder() else {
            continue;
        };
        let Ok(enc) = drm.get_encoder(enc_h) else {
            continue;
        };
        if let Some(crtc_h) = enc.crtc() {
            out.push(crtc_h);
        }
    }
    out
}

/// Find the config entry for a given DRM connector name (e.g.
/// `DisplayPort-4`). Matches against the user-typed `output "..."` argument
/// case-insensitively and accepts both the kernel-long form and the
/// short alias (e.g. `DP-4`).
///
/// Match a KDL `output "..."` block to a connector. Supports the
/// long kernel name (`DisplayPort-4`), short alias (`DP-4`), and the
/// EDID `<Make> <Model> <Serial>` triple — see
/// [`prism_config::output::block_matches_output`] for the matcher.
/// `edid` is `EdidInfo::default()` when the connector has no EDID
/// (rare; some virtual / dock outputs), in which case only the
/// connector-based paths can fire.
fn match_config_for_connector<'a>(
    connector_name: &str,
    edid: &crate::EdidInfo,
    outputs_cfg: &'a [OutputCfg],
) -> Option<&'a OutputCfg> {
    let output_name = prism_config::output::OutputName {
        connector: connector_name.to_string(),
        make: edid.make.clone(),
        model: edid.model.clone(),
        serial: edid.serial.clone(),
    };
    outputs_cfg
        .iter()
        .find(|o| prism_config::output::block_matches_output(&o.name, &output_name))
}

/// Pick a mode for `info`, honoring the config's optional `mode` override.
/// Returns `(mode, fallback)` where `fallback=true` means the configured
/// mode was not available and we fell back to the preferred mode.
///
/// Selection order (mirrors niri's `pick_mode`):
///   1. If config specifies `mode "WxH[@R]"`:
///       - find connector modes matching width × height; if `@R` is set
///         match refresh too (×1000 since smithay's `output::Mode` reports
///         refresh in millihertz); otherwise pick the highest-refresh
///         matching mode. Interlaced modes are excluded.
///       - if no match, set `fallback=true` and fall through to (2).
///   2. Pick the highest-refresh `PREFERRED` mode.
///   3. Last resort: first advertised mode.
///
/// `modeline` (custom CVT-derived modes) is not yet supported here —
/// would need to depend on `libdisplay-info` (already in our tree
/// transitively via smithay) and mirror niri's `calculate_mode_cvt` /
/// `create_mode_from_modeline` in `tty.rs`. Logs a warning if used.
fn pick_mode(info: &connector::Info, cfg: Option<&OutputCfg>) -> Option<(Mode, bool)> {
    let mut chosen: Option<&Mode> = None;
    let mut fallback = false;

    if let Some(cfg) = cfg {
        if cfg.modeline.is_some() {
            tracing::warn!("config `modeline` not yet implemented; falling back to preferred mode",);
            // Treat as a fallback too — the user asked for something
            // specific and we couldn't honor it.
            fallback = true;
        }

        if let Some(target) = cfg.mode {
            if target.custom {
                tracing::warn!(
                    "config `mode custom=true` not yet implemented (needs CVT); \
                     trying connector mode list, then preferred",
                );
            }
            let target_mode = target.mode;
            // smithay's output::Mode reports refresh in mHz (Hz × 1000).
            let refresh_mhz = target_mode.refresh.map(|r| (r * 1000.).round() as i32);
            for m in info.modes() {
                if m.size() != (target_mode.width, target_mode.height) {
                    continue;
                }
                if m.flags().contains(ModeFlags::INTERLACE) {
                    continue;
                }
                if let Some(refresh) = refresh_mhz {
                    let wl = WlMode::from(*m);
                    if wl.refresh == refresh {
                        chosen = Some(m);
                    }
                } else if let Some(curr) = chosen {
                    if curr.vrefresh() < m.vrefresh() {
                        chosen = Some(m);
                    }
                } else {
                    chosen = Some(m);
                }
            }
            if chosen.is_none() {
                fallback = true;
            }
        }
    }

    if chosen.is_none() {
        for m in info.modes() {
            if !m.mode_type().contains(ModeTypeFlags::PREFERRED) {
                continue;
            }
            match chosen {
                Some(curr) if curr.vrefresh() >= m.vrefresh() => {}
                _ => chosen = Some(m),
            }
        }
    }

    if chosen.is_none() {
        chosen = info.modes().first();
    }

    chosen.map(|m| (*m, fallback))
}

/// Find a free CRTC for this connector, building an `OutputPick`.
///
/// `occupied_by_other` are CRTCs we treat as off-limits because another
/// session owns them; `also_excluded` are CRTCs we've already picked in
/// the current bringup pass (multi-output uses this to avoid collisions).
/// A CRTC currently bound to *this* connector by another session is
/// still acceptable (we'll grab master and rebind it).
#[allow(clippy::too_many_arguments)] // CRTC resolution genuinely needs all of these
fn resolve_pick(
    drm: &DrmDevice,
    resources: &smithay::reexports::drm::control::ResourceHandles,
    conn_h: connector::Handle,
    info: &connector::Info,
    name: &str,
    mode: Mode,
    occupied_by_other: &[crtc::Handle],
    also_excluded: &[crtc::Handle],
) -> Result<OutputPick> {
    let own_crtc: Option<crtc::Handle> = info
        .current_encoder()
        .and_then(|enc_h| drm.get_encoder(enc_h).ok())
        .and_then(|enc| enc.crtc());

    for &enc_h in info.encoders() {
        let enc = drm
            .get_encoder(enc_h)
            .with_context(|| format!("get_encoder {enc_h:?}"))?;
        for candidate in resources.filter_crtcs(enc.possible_crtcs()) {
            if also_excluded.contains(&candidate) {
                continue;
            }
            let blocked_by_other =
                occupied_by_other.contains(&candidate) && Some(candidate) != own_crtc;
            if !blocked_by_other {
                return Ok(OutputPick {
                    connector: conn_h,
                    mode,
                    crtc: candidate,
                    connector_name: name.to_string(),
                });
            }
        }
    }
    Err(anyhow!(
        "no free CRTC available for {name} (all compatible CRTCs are bound to other connectors or already picked in this pass)"
    ))
}

/// Add a framebuffer for a GBM BO. The BO must have a non-INVALID modifier
/// (LINEAR / explicit-modifier BOs from `GbmDevice::allocate_scanout` qualify).
pub fn add_framebuffer_for_bo<T: 'static>(
    drm: &DrmDevice,
    bo: &BufferObject<T>,
) -> Result<framebuffer::Handle>
where
    BufferObject<T>: PlanarBuffer,
{
    let fb = drm
        .add_planar_framebuffer(bo, FbCmd2Flags::MODIFIERS)
        .context("add_planar_framebuffer")?;
    Ok(fb)
}

/// Find a named property on a resource by walking its property list.
/// Returns `None` if no such property exists on this object.
pub fn find_property<H: ResourceHandle>(
    drm: &DrmDevice,
    handle: H,
    name: &str,
) -> Result<Option<property::Handle>> {
    let props = drm.get_properties(handle).context("get_properties")?;
    for (&prop_h, _) in &props {
        let info = drm.get_property(prop_h).context("get_property")?;
        if info.name().to_string_lossy() == name {
            return Ok(Some(prop_h));
        }
    }
    Ok(None)
}

/// Set `max bpc` on a connector via the legacy property API.
///
/// `max bpc` controls the bit depth used on the physical link to the
/// display. Default is usually 8; setting it to 10 lets us send full
/// 10-bit scanout (paired with an A2R10G10B10 framebuffer). Without this
/// the driver dithers our 10-bit framebuffer down to 8 bits on the wire.
///
/// Returns `Ok(false)` if the property isn't exposed on this connector
/// (some drivers omit it for HDMI/DP variants); the caller can treat
/// that as "use whatever depth the link defaulted to". Returns `Ok(true)`
/// on a successful set.
pub fn set_connector_max_bpc(
    drm: &DrmDevice,
    connector: connector::Handle,
    value: u64,
) -> Result<bool> {
    let Some(prop) = find_property(drm, connector, "max bpc")? else {
        return Ok(false);
    };
    drm.set_property(connector, prop, value)
        .with_context(|| format!("set_property max_bpc={value}"))?;
    Ok(true)
}
