//! DRM format-modifier selection for scanout buffers.
//!
//! Bridges the Vulkan side (`prism_renderer::DrmFormatModifierInfo` — what
//! the GPU's color-attachment path can write into) with the KMS side (GBM's
//! `create_buffer_object_with_modifiers2`, which intersects with what the
//! display engine can scan out). We don't query the display engine
//! directly; GBM handles that intersection internally when given the
//! `SCANOUT|RENDERING` usage.
//!
//! Why this exists: hardcoding `LINEAR` for fp16 4K scanout on Vega 20
//! puts ~4 GB/s of uncompressed DRAM fetch on the display engine, which
//! pushes amdgpu's DCN bandwidth validator over a transient ceiling and
//! returns `-ENOMEM` from nonblocking page_flips. A tiled modifier drops
//! that to ~400 MB/s effective fetch (cache-line aligned tile bursts vs
//! row-major scattered reads) and the issue disappears.
//!
//! See `docs/deferred-work.md` for deferred follow-ups —
//! multi-plane (DCC) compression import, per-plane damage clips, and
//! atomic test commits before mode-changing operations.

use drm_fourcc::DrmModifier;
use prism_renderer::{vk, DrmFormatModifierInfo};

/// Pick the modifier candidate list to feed `GbmDevice::allocate_scanout`,
/// from what the Vulkan side advertised. Filter + order:
///
/// 1. Drop modifiers that don't support `COLOR_ATTACHMENT` (can't render
///    into them).
/// 2. Drop multi-plane modifiers (`plane_count > 1`). Our
///    `ImportedImage` is single-plane only; multi-plane modifiers carry
///    auxiliary metadata planes (DCC compression, CCS) that need
///    separate Vulkan memory imports.
/// 3. Preserve the driver's stated order — drivers list their preferred
///    modifier first.
/// 4. Push `LINEAR` (modifier value 0) to the end, and always append it
///    as a fallback if the driver didn't include it. GBM will pick
///    `LINEAR` only if no tiled modifier in the list is mutually
///    supported by the scanout pipe — never as a preference.
///
/// Caller is expected to call this once per output at bringup. Result
/// is a `Vec<DrmModifier>` ready to hand to GBM.
pub fn pick_scanout_modifiers(renderer_modifiers: &[DrmFormatModifierInfo]) -> Vec<DrmModifier> {
    let required = vk::FormatFeatureFlags::COLOR_ATTACHMENT;

    let mut picked: Vec<DrmModifier> = renderer_modifiers
        .iter()
        .filter(|m| m.plane_count == 1)
        .filter(|m| m.tiling_features.contains(required))
        .map(|m| DrmModifier::from(m.modifier))
        // Driver order first; we'll re-sort to push LINEAR to the back
        // in a moment so the ordering of tiled modifiers vs each other
        // is preserved.
        .collect();

    // Stable partition: tiled first (in driver order), LINEAR at the end.
    let linear_present = picked.contains(&DrmModifier::Linear);
    picked.retain(|m| *m != DrmModifier::Linear);
    if linear_present {
        picked.push(DrmModifier::Linear);
    } else {
        // The driver didn't list LINEAR — add it ourselves as the last-
        // resort fallback. amdgpu's scanout pipe always accepts LINEAR
        // for the formats we care about; if no tiled mutual modifier
        // exists, this prevents allocate_scanout from failing outright.
        picked.push(DrmModifier::Linear);
    }
    picked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(mod_value: u64, planes: u32, color: bool) -> DrmFormatModifierInfo {
        let features = if color {
            vk::FormatFeatureFlags::COLOR_ATTACHMENT | vk::FormatFeatureFlags::SAMPLED_IMAGE
        } else {
            vk::FormatFeatureFlags::SAMPLED_IMAGE
        };
        DrmFormatModifierInfo {
            modifier: mod_value,
            plane_count: planes,
            tiling_features: features,
        }
    }

    #[test]
    fn empty_input_still_yields_linear_fallback() {
        let out = pick_scanout_modifiers(&[]);
        assert_eq!(out, vec![DrmModifier::Linear]);
    }

    #[test]
    fn multi_plane_modifiers_are_dropped() {
        // 0x500_0000_0000_0001 is an arbitrary AMD-style tiled value with 2 planes.
        let mods = [entry(0x500_0000_0000_0001, 2, true), entry(0, 1, true)];
        let out = pick_scanout_modifiers(&mods);
        assert_eq!(out, vec![DrmModifier::Linear]);
    }

    #[test]
    fn non_renderable_modifiers_are_dropped() {
        let mods = [entry(0x123, 1, false), entry(0, 1, true)];
        let out = pick_scanout_modifiers(&mods);
        assert_eq!(out, vec![DrmModifier::Linear]);
    }

    #[test]
    fn tiled_modifiers_ordered_before_linear() {
        let mods = [
            entry(0, 1, true),     // LINEAR first in driver list
            entry(0xAAA, 1, true), // some tiled mod
            entry(0xBBB, 1, true), // another tiled mod
        ];
        let out = pick_scanout_modifiers(&mods);
        // Tiled (in driver order), LINEAR last.
        assert_eq!(
            out,
            vec![
                DrmModifier::from(0xAAA),
                DrmModifier::from(0xBBB),
                DrmModifier::Linear,
            ]
        );
    }

    #[test]
    fn linear_appended_if_driver_omits_it() {
        let mods = [entry(0xAAA, 1, true)];
        let out = pick_scanout_modifiers(&mods);
        assert_eq!(out, vec![DrmModifier::from(0xAAA), DrmModifier::Linear]);
    }
}
