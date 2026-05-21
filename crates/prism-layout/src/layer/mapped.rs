//! `MappedLayer` — minimal scaffold.
//!
//! Full port from `niri/src/layer/mapped.rs` (333 LOC; per-frame
//! geometry, shadow rendering, popups, baba-is-float bob offset) is
//! deferred until step 7 of the niri port, when the layer-shell render
//! call sites in tile.rs / workspace.rs get wired through. Doing the
//! full port standalone is wasted effort — the shadow + popup pipelines
//! depend on infrastructure (offscreen rendering, the Vulkan equivalent
//! of niri's `BorderRenderElement` shader for shadows) we don't have
//! yet.
//!
//! Today this carries just enough surface for `super::ResolvedLayerRules`
//! and the handler scaffolding to construct + reference a `MappedLayer`:
//! the wrapped `LayerSurface` and a `ResolvedLayerRules`. Other methods
//! `unimplemented!` until step 7.

use smithay::desktop::LayerSurface;

use super::ResolvedLayerRules;

#[derive(Debug)]
pub struct MappedLayer {
    pub surface: LayerSurface,
    pub rules: ResolvedLayerRules,
}

impl MappedLayer {
    pub fn new(surface: LayerSurface, rules: ResolvedLayerRules) -> Self {
        Self { surface, rules }
    }

    pub fn surface(&self) -> &LayerSurface {
        &self.surface
    }

    pub fn rules(&self) -> &ResolvedLayerRules {
        &self.rules
    }
}
