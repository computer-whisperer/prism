//! VkDevice-loss recovery orchestrator (docs/device-loss-recovery.md).
//!
//! Detection and quarantine live in `prism-protocols`
//! ([`PrismState::note_gpu_lost`] and the `gpu_is_lost` gates on every
//! render/materialize/mirror/upload path). This module is the main-loop
//! side: once per dispatch iteration, if a quarantine was flagged, replace
//! the lost GPU's `Device` with a fresh one (fresh render-node fd + amdgpu
//! context — RADV's lost flag is per-VkDevice and sticky) and rebuild
//! everything that hung off it:
//!
//! - the `gpus` registry entry — replaced FIRST, so the R2 Arc-identity
//!   staleness purge flips for every cached per-surface materialization at
//!   once (they lazily rebuild on the new device at their next read);
//! - the GPU's `MirrorCopier`;
//! - each of its outputs' render state (`OutputContext::
//!   rebuild_render_state`: new `Renderer`, scanout BOs re-imported — the
//!   BOs/DRM FBs are kernel-side and survive, so panels keep their last
//!   frame throughout — LUT re-synthesized from retained state);
//! - layout snapshot ghosts and the dmabuf native-import memos.
//!
//! Failure (e.g. the GPU is genuinely wedged, like a Vega SMU hang) leaves
//! the GPU quarantined and retries on an exponential-backoff timer; other
//! GPUs are unaffected. The old `Device` fades via `Arc`: stale
//! materializations drop it as they purge, and `Device::drop` drains the
//! deferred-destroy queue unconditionally.

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
                state.gpu_recovery_attempts.remove(&gpu);
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
                schedule_retry(state, delay);
            }
        }
    }
}

fn recover_one(state: &mut PrismState, gpu: DrmDevId) -> Result<()> {
    let old = state
        .gpus
        .get(&gpu)
        .cloned()
        .context("lost GPU not in registry")?;
    tracing::info!(
        ?gpu,
        old_device_refs = Arc::strong_count(&old),
        "GPU recovery: recreating VkDevice"
    );
    let device = prism_renderer::Device::new(old.instance.clone(), Some(gpu))
        .context("recreate VkDevice")?;
    // Device::new falls back to any discrete GPU when the preferred DRM
    // node doesn't match (bringup semantics). For recovery that would
    // silently rebuild this GPU's outputs on a *different* card — reject.
    if device.physical.drm_primary != Some(gpu) {
        anyhow::bail!(
            "replacement device landed on {:?}, not the lost GPU {:?} (unplugged?)",
            device.physical.drm_primary,
            gpu
        );
    }

    // Registry swap FIRST: from here every stale materialization fails the
    // R2 Arc-identity check and lazily rebuilds on the new device.
    state.gpus.insert(gpu, device.clone());

    // The old copier waits its fence on drop — on a lost device the wait
    // returns (error, ignored) rather than blocking.
    match prism_renderer::MirrorCopier::new(device.clone()) {
        Ok(c) => {
            state.mirror_copiers.insert(gpu, c);
        }
        Err(e) => {
            // Without a copier this GPU can't be a mirror home/target, but
            // native rendering still works — degrade like bringup does.
            state.mirror_copiers.remove(&gpu);
            tracing::warn!(?gpu, "mirror copier rebuild failed: {e:#}");
        }
    }

    for (id, ctx) in state.outputs.iter_mut() {
        if ctx.gpu_id == gpu {
            ctx.rebuild_render_state(device.clone())
                .with_context(|| format!("rebuild output {id}"))?;
        }
    }

    // Snapshot ghosts captured on the old device: close animations end,
    // resize crossfades cancel (recapture would copy unrelated pixels).
    let gpus = &state.gpus;
    state
        .layout
        .purge_stale_snapshots(&|s| !gpus.values().any(|d| Arc::ptr_eq(d, s.device())));

    // Memo hygiene: the hit-site identity check already rejects these,
    // this just frees them now instead of at buffer destruction.
    state.drop_native_import_memos(gpu);
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
