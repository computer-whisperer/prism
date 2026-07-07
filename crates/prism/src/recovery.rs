//! VkDevice-loss recovery orchestrator (docs/device-loss-recovery.md).
//!
//! Detection and quarantine live in `prism-protocols`
//! ([`PrismState::note_gpu_lost`] and the `gpu_is_lost` gates on every
//! render/materialize/mirror/upload path). This module is the main-loop
//! side: once per dispatch iteration, if a quarantine was flagged, replace
//! the lost GPU's `Device` with a fresh one and rebuild everything that
//! hung off it.
//!
//! The 07-05/07-07 outages showed a fresh device (fresh render-node fd +
//! amdgpu context) in THIS process being kernel-rejected exactly like the
//! poisoned one, 18 attempts straight, while a fresh process succeeded
//! seconds later — so recovery now assumes the poison may be ANCHORED to
//! the old device's open fd and drops every holder of it (exhaustive
//! surface sweep, ghosts, memos, copier, per-output render state) BEFORE
//! the replacement device submits anything (`recover_one` phase ordering;
//! `Renderer::new` is submit-free for this reason). Whether the old fd
//! actually closed is logged as the experiment readout, and a `gpu-probe`
//! child process is forked on failures to discriminate process-anchored
//! vs device-global rejection.
//!
//! Failure (e.g. the GPU is genuinely wedged, like a Vega SMU hang) leaves
//! the GPU quarantined and retries on an exponential-backoff timer; other
//! GPUs are unaffected.

use std::sync::Arc;

use anyhow::{Context, Result};
use calloop::timer::{TimeoutAction, Timer};
use prism_protocols::PrismState;
use prism_renderer::DrmDevId;

/// Service pending GPU quarantines. Called from the main dispatch loop
/// between passes (no state borrows outstanding, no frame mid-lowering)
/// whenever `gpu_recovery_pending` was set.
pub fn recover_lost_gpus(state: &mut PrismState) {
    let lost: Vec<DrmDevId> = state.lost_gpus.borrow().iter().copied().collect();
    for gpu in lost {
        let attempt = state.gpu_recovery_attempts.get(&gpu).copied().unwrap_or(0) + 1;
        match recover_one(state, gpu) {
            Ok(()) => {
                state.lost_gpus.borrow_mut().remove(&gpu);
                // Deliberately NOT clearing gpu_recovery_attempts here: a
                // successful rebuild only proves trivial submits work. The
                // counter clears on this GPU's first successful present
                // (perform_redraw), so a half-alive GPU that passes rebuild
                // but fails real renders keeps climbing the backoff ladder
                // instead of churning device creation at dispatch rate.
                let outputs: Vec<String> = state
                    .outputs
                    .iter()
                    .filter(|(_, c)| c.gpu_id == gpu)
                    .map(|(id, _)| id.clone())
                    .collect();
                for id in &outputs {
                    // Queued captures were imported on the dead device;
                    // fail them rather than submit cross-device work.
                    state.fail_pending_screencopy(id);
                    // An output the lock path blanked (CRTC clear) has no
                    // other wake-up: nothing else calls power_on for it,
                    // so it would stay dark after unlock forever. Re-power
                    // exactly those — while still locked the next render
                    // shows the lock surface, not desktop content. User
                    // DPMS state (not in this set) is untouched.
                    if state.lock_blanked_outputs.remove(id) {
                        if let Some(ctx) = state.outputs.get_mut(id) {
                            ctx.power_on();
                        }
                    }
                    state
                        .output_redraw
                        .entry(id.clone())
                        .or_default()
                        .queue_redraw();
                }
                tracing::info!(
                    ?gpu,
                    ?outputs,
                    attempt,
                    "GPU recovered: VkDevice replaced, outputs requeued"
                );
            }
            Err(e) => {
                state.gpu_recovery_attempts.insert(gpu, attempt);
                // 2s, 4s, 8s, 16s, 32s, then steady at 32s — a genuinely
                // dead GPU shouldn't spam device creation, but a recovering
                // one (e.g. post-reset SMU settling) should be retried soon.
                let delay = std::time::Duration::from_secs(2u64 << (attempt - 1).min(4));
                tracing::error!(
                    ?gpu,
                    attempt,
                    "GPU recovery failed: {e:#}; retry in {delay:?}"
                );
                // Diagnostic fork (observed 07-05/07-07: fresh devices in
                // THIS process are rejected while a fresh process succeeds
                // seconds later): a child process runs the same
                // device-create + first-submit and logs its verdict to the
                // journal next to ours. Child-OK + parent-failing proves
                // the poison is anchored to this process / its open fds;
                // child-failing means device-global. Attempts 1 and 5 (one
                // early sample, one after ~30s persistence).
                if attempt == 1 || attempt == 5 {
                    prism_input::actions::spawn(vec![
                        "prism".to_owned(),
                        "gpu-probe".to_owned(),
                        format!("{}:{}", gpu.major, gpu.minor),
                    ]);
                }
                schedule_retry(state, delay);
            }
        }
    }
}

/// Replace `gpu`'s VkDevice. Ordering is the whole point (07-05/07-07
/// evidence: fresh devices in this process were rejected while the
/// original dead device's fd was still open — the leading hypothesis is
/// kernel state anchored by that fd):
///
///   1. drop every holder of the OLD device — registry entry, copier, all
///      per-surface materializations (exhaustive sweep), snapshot ghosts,
///      import memos, queued captures, and each output's renderer +
///      scanout imports (swapped for submit-free replacements);
///   2. CLOSE the old fd (`drop(old)`) and log whether it actually closed
///      (`Arc::strong_count == 1`) — the experiment readout;
///   3. only then perform the first submits on the replacement device
///      (LUT re-synthesis here; the deferred renderer init at the first
///      redraw).
///
/// On failure the newest device is left in the registry (invariants hold;
/// every path is still gated by the quarantine) and the next attempt
/// re-runs the full sequence — `rebuild_render_state` is idempotent.
fn recover_one(state: &mut PrismState, gpu: DrmDevId) -> Result<()> {
    // ── Phase 1: tear down every holder of the old device ──────────────
    let old = state
        .gpus
        .remove(&gpu)
        .context("lost GPU not in registry")?;
    let instance = old.instance.clone();
    tracing::info!(
        ?gpu,
        old_device_refs = Arc::strong_count(&old),
        "GPU recovery: tearing down old device"
    );
    // The old copier waits its fence on drop — on a lost device the wait
    // returns (error, ignored) rather than blocking.
    drop(state.mirror_copiers.remove(&gpu));
    // Exhaustive: with the registry entry removed, the Arc-identity check
    // fails for every entry touching this GPU (incl. mirrors homed on it),
    // across ALL registered surfaces — not just ones the walk would visit.
    state.purge_all_surface_textures();
    // Snapshot ghosts captured on the old device: close animations end,
    // resize crossfades cancel (recapture would copy unrelated pixels).
    {
        let gpus = &state.gpus;
        state
            .layout
            .purge_stale_snapshots(&|s| !gpus.values().any(|d| Arc::ptr_eq(d, s.device())));
    }
    state.drop_native_import_memos(gpu);
    // Queued captures hold imports on the old device; fail them now.
    let out_ids: Vec<String> = state
        .outputs
        .iter()
        .filter(|(_, c)| c.gpu_id == gpu)
        .map(|(id, _)| id.clone())
        .collect();
    for id in &out_ids {
        state.fail_pending_screencopy(id);
    }

    // ── Phase 2: replacement device + submit-free rebuilds ─────────────
    // Restore-registry-on-failure keeps invariants (e.g. the
    // quarantined-home check in ensure_home_import consults the map).
    let restore = |state: &mut PrismState, dev: Arc<prism_renderer::Device>| {
        state.gpus.insert(gpu, dev);
    };
    let device = match prism_renderer::Device::new(instance, Some(gpu)) {
        Ok(d) => d,
        Err(e) => {
            restore(state, old);
            return Err(e).context("recreate VkDevice");
        }
    };
    // Device::new falls back to any discrete GPU when the preferred DRM
    // node doesn't match (bringup semantics). For recovery that would
    // silently rebuild this GPU's outputs on a *different* card — reject.
    if device.physical.drm_primary != Some(gpu) {
        let landed = device.physical.drm_primary;
        restore(state, old);
        anyhow::bail!(
            "replacement device landed on {landed:?}, not the lost GPU {gpu:?} (unplugged?)"
        );
    }
    // Rebuild every output (constructor is submit-free; the swap drops the
    // old renderer + scanout imports).
    let mut rebuild_err: Option<anyhow::Error> = None;
    for (id, ctx) in state.outputs.iter_mut() {
        if ctx.gpu_id == gpu {
            if let Err(e) = ctx.rebuild_render_state(device.clone()) {
                rebuild_err = Some(e.context(format!("rebuild output {id}")));
                break;
            }
        }
    }
    if let Some(e) = rebuild_err {
        restore(state, device);
        return Err(e);
    }

    // ── Phase 3: close the old fd, then first submits ──────────────────
    let old_refs = Arc::strong_count(&old);
    drop(old);
    if old_refs == 1 {
        tracing::info!(
            ?gpu,
            "old device fully released — fd closed before any submit on the replacement"
        );
    } else {
        // Not fatal (the remaining holders fade), but the fd stays open —
        // if recovery still fails, this line says the experiment didn't
        // actually test the fd-anchor hypothesis this round.
        tracing::warn!(
            ?gpu,
            refs = old_refs - 1,
            "old device still referenced; its fd stays open (in-flight capture? leak?)"
        );
    }
    state.gpus.insert(gpu, device.clone());
    match prism_renderer::MirrorCopier::new(device.clone()) {
        Ok(c) => {
            state.mirror_copiers.insert(gpu, c);
        }
        Err(e) => {
            // Without a copier this GPU can't be a mirror home/target, but
            // native rendering still works — degrade like bringup does.
            tracing::warn!(?gpu, "mirror copier rebuild failed: {e:#}");
        }
    }
    // First submits on the replacement device: the per-output LUT upload.
    // A rejection here means the new device is now lost too — fail the
    // attempt so the retry runs the full cycle again.
    for (id, ctx) in state.outputs.iter_mut() {
        if ctx.gpu_id == gpu {
            ctx.resynthesize_color_lut()
                .with_context(|| format!("post-rebuild LUT upload on {id}"))?;
        }
    }
    Ok(())
}

/// `prism gpu-probe <major>:<minor>`: the child half of the recovery
/// diagnostic fork. Creates a fresh VkInstance + Device on the given GPU
/// (by DRM primary-node id) and performs the exact submit that keeps
/// getting kernel-rejected in the parent — a 1×1 white shm-texture upload,
/// byte-for-byte the first submit of `Renderer::new` — then reports
/// loudly. stderr inherits the compositor's journald stream, so the
/// verdict lands next to the recovery-failure lines it diagnoses.
pub fn run_gpu_probe(arg: Option<&str>) -> Result<()> {
    let arg = arg.context("usage: prism gpu-probe <major>:<minor>")?;
    let (major, minor) = arg
        .split_once(':')
        .context("gpu-probe: expected <major>:<minor>")?;
    let gpu = DrmDevId {
        major: major.parse().context("gpu-probe: bad major")?,
        minor: minor.parse().context("gpu-probe: bad minor")?,
    };
    tracing::info!(?gpu, pid = std::process::id(), "gpu-probe: starting");
    let instance = prism_renderer::Instance::new().context("gpu-probe: Instance::new")?;
    let device =
        prism_renderer::Device::new(instance, Some(gpu)).context("gpu-probe: Device::new")?;
    if device.physical.drm_primary != Some(gpu) {
        anyhow::bail!(
            "gpu-probe: selection landed on {:?}, not {:?}",
            device.physical.drm_primary,
            gpu
        );
    }
    let mut tex = prism_renderer::ShmTexture::new(
        device.clone(),
        prism_renderer::vk::Extent2D {
            width: 1,
            height: 1,
        },
        prism_renderer::vk::Format::R8G8B8A8_UNORM,
    )
    .context("gpu-probe: ShmTexture::new")?;
    tex.upload_bytes(&[255, 255, 255, 255], 4, &[])
        .context("gpu-probe: upload submit REJECTED")?;
    tracing::info!(
        ?gpu,
        pid = std::process::id(),
        "gpu-probe: SUBMIT OK — fresh process can use this GPU"
    );
    Ok(())
}

/// Re-arm the recovery flag after `delay`, so the main loop retries the
/// still-quarantined GPUs. Without a loop handle (never happens after
/// startup) recovery would stall until the next device-lost detection.
fn schedule_retry(state: &PrismState, delay: std::time::Duration) {
    let Some(handle) = &state.loop_handle else {
        tracing::warn!("no loop handle; GPU recovery retry not scheduled");
        return;
    };
    let timer = Timer::from_duration(delay);
    if handle
        .insert_source(timer, |_, _, state: &mut PrismState| {
            state.gpu_recovery_pending.set(true);
            TimeoutAction::Drop
        })
        .is_err()
    {
        tracing::warn!("failed to arm GPU recovery retry timer");
    }
}
