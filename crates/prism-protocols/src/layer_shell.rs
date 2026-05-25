//! `wlr_layer_shell` — layer surfaces mapped through smithay's per-output
//! [`LayerMap`](smithay::desktop::layer_map_for_output).
//!
//! Each layer surface is mapped into `layer_map_for_output(output)`, which
//! owns its arranged geometry — anchors, margins, exclusive zones, and the
//! "size 0 ⇒ span the anchored axis" defaults — and recomputes it on
//! `arrange()`. Rendering (in the binary's output render path) iterates
//! `layers_on(layer)` per layer in z-order and walks each surface through
//! the shared color-managed surface-tree walk
//! (`prism_layout::layout::element::push_surface_tree_elements`), so
//! layer-shell chrome (bars, wallpapers, notification daemons) composites
//! exactly like ordinary windows: same `wp_color_management_v1` decode,
//! same cross-GPU mirror handling, same subsurface z-ordering. There is no
//! separate unmanaged-sRGB blit path for it.
//!
//! **Stage-1 scope (deliberate gaps):** all four layers render + arrange,
//! and exclusive zones shrink the tiling work area (via
//! `LayerMap::non_exclusive_zone`, consumed by `compute_working_area` in
//! prism-layout). Not yet: layer *popups*, layer *shadows*, and
//! `KeyboardInteractivity` (treated as `None` — `OnDemand`/`Exclusive`
//! land with the keyboard-focus work).

use smithay::desktop::{layer_map_for_output, LayerSurface, WindowSurfaceType};
use smithay::output::Output;
use smithay::reexports::wayland_server::protocol::{wl_output::WlOutput, wl_surface::WlSurface};
use smithay::wayland::compositor::with_states;
use smithay::wayland::shell::wlr_layer::{
    Layer, LayerSurface as WlrLayerSurface, LayerSurfaceData, WlrLayerShellHandler,
    WlrLayerShellState,
};

use crate::state::{OutputId, PrismState};

impl PrismState {
    /// Resolve a client-supplied `WlOutput` (from `get_layer_surface`) to one
    /// of our outputs, returning both its `OutputId` (connector name) and the
    /// smithay `Output`. `None` request ⇒ first available output (spec-
    /// permitted fallback). Returns `None` only if we have no outputs (the
    /// layer surface then stays inert until one appears).
    fn resolve_layer_output(&self, wl_output: Option<&WlOutput>) -> Option<(OutputId, Output)> {
        if let Some(req) = wl_output {
            for (id, output) in &self.wl_outputs {
                if output.owns(req) {
                    return Some((id.clone(), output.clone()));
                }
            }
            tracing::warn!(
                "layer_shell: client requested an output we don't own; falling back to first"
            );
        }
        self.wl_outputs
            .iter()
            .next()
            .map(|(id, o)| (id.clone(), o.clone()))
    }

    /// The `OutputId` (connector name) whose `LayerMap` hosts `surface`, if it
    /// is a mapped layer surface (or a subsurface of one). Scans each output's
    /// map. Used by the dmabuf / color-management feedback paths to find a
    /// layer surface's output, mirroring what placement does for toplevels.
    pub fn layer_surface_output_id(&self, surface: &WlSurface) -> Option<OutputId> {
        for (id, output) in &self.wl_outputs {
            if layer_map_for_output(output)
                .layer_for_surface(surface, WindowSurfaceType::ALL)
                .is_some()
            {
                return Some(id.clone());
            }
        }
        None
    }

    /// Map a freshly-created layer surface into its output's `LayerMap`.
    /// `map_layer` arranges immediately (computing geometry + sending
    /// `wl_surface.enter`); the initial configure is sent from the commit
    /// handler once the role's pending state has settled.
    pub fn layer_shell_new_surface(
        &mut self,
        surface: WlrLayerSurface,
        wl_output: Option<WlOutput>,
        _layer: Layer,
        namespace: String,
    ) {
        let Some((output_id, output)) = self.resolve_layer_output(wl_output.as_ref()) else {
            tracing::warn!(
                namespace = %namespace,
                "layer_shell: no outputs to host this surface; ignoring"
            );
            return;
        };
        let wl_surface = surface.wl_surface().clone();

        if let Err(e) =
            layer_map_for_output(&output).map_layer(&LayerSurface::new(surface, namespace.clone()))
        {
            tracing::warn!(namespace = %namespace, "layer_shell: map_layer failed: {e:?}");
            return;
        }

        // Mark placement so the buffer-import path materializes this surface's
        // texture on the hosting output's GPU, and so dmabuf/color feedback
        // resolves the right output (mirrors the toplevel placement slot).
        with_states(&wl_surface, |states| {
            states
                .data_map
                .insert_if_missing_threadsafe(crate::surface_tex::SurfacePlacementSlot::default);
            let slot = states
                .data_map
                .get::<crate::surface_tex::SurfacePlacementSlot>()
                .unwrap();
            slot.0.lock().unwrap().current_output = Some(output_id.clone());
        });

        tracing::info!(
            namespace = %namespace,
            connector = %output_id,
            "layer_shell: surface created + mapped"
        );
        self.output_redraw
            .entry(output_id)
            .or_default()
            .queue_redraw();
    }

    /// Commit-time arrange for a layer surface: recompute the per-output
    /// layout (so anchor/size/margin/exclusive-zone changes take effect) and
    /// send the initial configure exactly once. Called from the compositor
    /// commit handler for surfaces with the layer-surface role.
    pub fn layer_shell_commit(&mut self, surface: &WlSurface) {
        let Some(output) = self
            .wl_outputs
            .values()
            .find(|o| {
                layer_map_for_output(o)
                    .layer_for_surface(surface, WindowSurfaceType::TOPLEVEL)
                    .is_some()
            })
            .cloned()
        else {
            return;
        };

        let initial_configure_sent = with_states(surface, |states| {
            states
                .data_map
                .get::<LayerSurfaceData>()
                .map(|d| d.lock().unwrap().initial_configure_sent)
                .unwrap_or(false)
        });

        {
            let mut map = layer_map_for_output(&output);
            // Arrange before the initial configure so the client gets the
            // size we computed from its requested anchor/size.
            map.arrange();
            if !initial_configure_sent {
                if let Some(layer) = map.layer_for_surface(surface, WindowSurfaceType::TOPLEVEL) {
                    layer.layer_surface().send_configure();
                }
            }
        }

        if let Some(id) = self.layer_surface_output_id(surface) {
            self.output_redraw.entry(id).or_default().queue_redraw();
        }
    }

    /// Unmap a destroyed layer surface from whichever output's `LayerMap`
    /// holds it (`unmap_layer` re-arranges + sends `wl_surface.leave`).
    pub fn layer_shell_destroyed(&mut self, surface: WlrLayerSurface) {
        let found = self.wl_outputs.iter().find_map(|(id, o)| {
            let map = layer_map_for_output(o);
            let layer = map
                .layers()
                .find(|&l| l.layer_surface() == &surface)
                .cloned();
            layer.map(|l| (id.clone(), o.clone(), l))
        });
        if let Some((id, output, layer)) = found {
            layer_map_for_output(&output).unmap_layer(&layer);
            self.output_redraw.entry(id).or_default().queue_redraw();
        }
    }
}

impl WlrLayerShellHandler for PrismState {
    fn shell_state(&mut self) -> &mut WlrLayerShellState {
        &mut self.layer_shell_state
    }

    fn new_layer_surface(
        &mut self,
        surface: WlrLayerSurface,
        output: Option<WlOutput>,
        layer: Layer,
        namespace: String,
    ) {
        self.layer_shell_new_surface(surface, output, layer, namespace);
    }

    fn layer_destroyed(&mut self, surface: WlrLayerSurface) {
        self.layer_shell_destroyed(surface);
    }
}

// Re-export so state.rs can wire the delegate macro.
pub use smithay::delegate_layer_shell;
