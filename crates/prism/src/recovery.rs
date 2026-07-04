//! VkDevice-loss recovery orchestrator (docs/device-loss-recovery.md).
//!
//! Detection and quarantine live in `prism-protocols`
//! ([`PrismState::note_gpu_lost`] and the `gpu_is_lost` gates on every
//! render/materialize/mirror/upload path). This module is the main-loop
//! side: once per dispatch iteration, if a quarantine was flagged, tear
//! down the lost GPU's Vulkan state and rebuild it on a fresh `Device`
//! (fresh render-node fd + amdgpu context).
//!
//! Increment status: R1 (quarantine) — recovery itself is R2/R3 and not
//! yet implemented; a lost GPU currently stays parked until restart, which
//! is still a strict improvement over the pre-quarantine behavior (spammed
//! journal, wedged screencopy clients, endless failing rerenders).

use prism_protocols::PrismState;
use prism_renderer::DrmDevId;

/// Service pending GPU quarantines. Called from the main dispatch loop
/// between passes (no state borrows outstanding) whenever
/// `gpu_recovery_pending` was set.
pub fn recover_lost_gpus(state: &mut PrismState) {
    let lost: Vec<DrmDevId> = state.lost_gpus.borrow().iter().copied().collect();
    for gpu in lost {
        let outputs: Vec<String> = state
            .outputs
            .iter()
            .filter(|(_, o)| o.gpu_id == gpu)
            .map(|(id, _)| id.clone())
            .collect();
        tracing::error!(
            ?gpu,
            ?outputs,
            "GPU quarantined; device recovery not yet implemented (R3) — \
             outputs on it stay parked on their last frame until restart"
        );
    }
}
