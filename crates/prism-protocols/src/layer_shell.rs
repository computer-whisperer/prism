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
//! **Keyboard interactivity** is honored: [`Self::update_keyboard_focus`]
//! arbitrates focus between layer surfaces and the layout. An `Exclusive`
//! surface (launcher / lock-style) on the focused output grabs the keyboard
//! on map and releases it on unmap; an `OnDemand` surface takes focus when
//! clicked; a `None` surface (bars, wallpapers) never receives keyboard
//! input. See that method for the priority order.
//!
//! **Scope (deliberate gaps):** all four layers render + arrange, and
//! exclusive zones shrink the tiling work area (via
//! `LayerMap::non_exclusive_zone`, consumed by `compute_working_area` in
//! prism-layout). Not yet: layer *popups* and layer *shadows*. Exclusive
//! grab is scoped to the *focused* output (a surface on a non-focused
//! monitor waits until that monitor is focused).

use smithay::desktop::{layer_map_for_output, LayerSurface, WindowSurfaceType};
use smithay::output::Output;
use smithay::reexports::wayland_server::protocol::{wl_output::WlOutput, wl_surface::WlSurface};
use smithay::reexports::wayland_server::Resource as _;
use smithay::utils::{IsAlive, SERIAL_COUNTER};
use smithay::wayland::compositor::with_states;
use smithay::wayland::shell::wlr_layer::{
    KeyboardInteractivity, Layer, LayerSurface as WlrLayerSurface, LayerSurfaceData,
    WlrLayerShellHandler, WlrLayerShellState,
};

use crate::input_state::KeyboardFocus;
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
                "layer_shell: client requested an output we don't own; falling back to focused"
            );
        }
        // `output = None` (launchers like fuzzel/wofi, notification daemons):
        // place on the *focused* output so the surface appears on the monitor
        // the user is actually on, not an arbitrary `HashMap` entry. Fall back
        // to any output only if nothing is focused yet.
        if let Some(active) = self.active_output() {
            let id = active.name();
            if let Some(output) = self.wl_outputs.get(&id) {
                return Some((id, output.clone()));
            }
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

    /// The mapped [`LayerSurface`] that owns `surface` (or a subsurface /
    /// popup of it), scanning every output's `LayerMap`. Used by the focus
    /// arbiter and click-to-focus to read a surface's keyboard-interactivity.
    pub fn layer_surface_for(&self, surface: &WlSurface) -> Option<LayerSurface> {
        self.wl_outputs.values().find_map(|output| {
            layer_map_for_output(output)
                .layer_for_surface(surface, WindowSurfaceType::ALL)
                .cloned()
        })
    }

    /// The top-most `Exclusive`-interactivity layer surface on the *focused*
    /// output, searching Overlay above Top (the only layers we let grab the
    /// keyboard, matching niri). `None` if no exclusive surface is mapped
    /// there. Topmost-first mirrors [`LayerMap::layer_under`]'s `.rev()`.
    fn exclusive_layer_focus(&self) -> Option<WlSurface> {
        let output = self.active_output()?;
        let map = layer_map_for_output(&output);
        for layer in [Layer::Overlay, Layer::Top] {
            if let Some(ls) = map.layers_on(layer).rev().find(|ls| {
                ls.cached_state().keyboard_interactivity == KeyboardInteractivity::Exclusive
            }) {
                return Some(ls.wl_surface().clone());
            }
        }
        None
    }

    /// The remembered `OnDemand` layer-shell focus, but only if it is still
    /// mapped and still advertises `OnDemand` (a surface can change its
    /// interactivity or unmap out from under us).
    fn active_on_demand_layer_focus(&self) -> Option<WlSurface> {
        let surface = self.on_demand_layer_focus.as_ref()?;
        let ls = self.layer_surface_for(surface)?;
        (ls.cached_state().keyboard_interactivity == KeyboardInteractivity::OnDemand)
            .then(|| surface.clone())
    }

    /// Recompute the effective keyboard focus and push it to the seat.
    ///
    /// The single arbiter of [`Self::keyboard_focus`]. Priority (Stage-3
    /// layer-shell focus model, mirroring niri's `update_keyboard_focus`):
    ///   1. The top-most `Exclusive` layer surface on the focused output
    ///      (a launcher / lock-style surface grabs the keyboard on map).
    ///   2. A layer surface the user clicked while it was `OnDemand`, as
    ///      long as it is still mapped + still `OnDemand`.
    ///   3. The layout's active window's toplevel surface, read live from
    ///      [`Layout::focus`] — the single source of truth. Keyboard focus is
    ///      thus *derived* from layout state, not stored separately: any path
    ///      that moves the layout's active window (click, focus-follows-mouse,
    ///      keyboard navigation, window close) is reflected the next time this
    ///      runs. Mirrors niri's `update_keyboard_focus` (niri.rs:1167).
    ///
    /// `None`-interactivity surfaces (bars, wallpapers) are never candidates,
    /// so they never steal the keyboard — even when clicked. Idempotent:
    /// no-ops (no enter/leave round-trip) when the effective surface is
    /// unchanged, so it is cheap to call every frame.
    pub fn update_keyboard_focus(&mut self) {
        let target = self
            .exclusive_layer_focus()
            .or_else(|| self.active_on_demand_layer_focus())
            .or_else(|| {
                self.layout
                    .focus()
                    .map(|win| win.toplevel().wl_surface().clone())
                    .filter(IsAlive::alive)
            });

        let same = match (self.keyboard_focus.surface(), &target) {
            (Some(a), Some(b)) => a.id() == b.id(),
            (None, None) => true,
            _ => false,
        };
        if same {
            return;
        }

        let from_layer = target
            .as_ref()
            .is_some_and(|s| self.layer_surface_for(s).is_some());
        self.keyboard_focus = match (&target, from_layer) {
            (Some(s), true) => KeyboardFocus::LayerShell { surface: s.clone() },
            _ => KeyboardFocus::Layout {
                surface: target.clone(),
            },
        };

        if let Some(kb) = self.seat.get_keyboard() {
            let serial = SERIAL_COUNTER.next_serial();
            kb.set_focus(self, target, serial);
        }
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
        // An `Exclusive` surface (launcher / lock) grabs the keyboard the
        // moment it maps; recompute focus so it does.
        self.update_keyboard_focus();
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

        // A commit can change the surface's exclusive zone (a bar reserving
        // space along an edge). `arrange()` above updated the `LayerMap`'s
        // non-exclusive zone, but the layout's working area is derived from it
        // and must be recomputed or tiled windows would overlap the bar.
        // Mirrors niri's `output_resized` after arranging layers (niri.rs:2990).
        self.layout.update_output_size(&output);

        if let Some(id) = self.layer_surface_output_id(surface) {
            self.output_redraw.entry(id).or_default().queue_redraw();
        }
        // A commit can change `keyboard_interactivity` (e.g. None → Exclusive
        // once the client is ready), so re-arbitrate focus.
        self.update_keyboard_focus();
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
            // `unmap_layer` re-arranged the map; reclaim the bar's exclusive
            // zone for the layout's working area so tiled windows expand back.
            self.layout.update_output_size(&output);
            self.output_redraw.entry(id).or_default().queue_redraw();
        }
        // Drop a stale on-demand reference to the destroyed surface, then
        // re-arbitrate so the keyboard returns to the layout (or the next
        // exclusive surface) now that this one is gone.
        if self.on_demand_layer_focus.as_ref() == Some(surface.wl_surface()) {
            self.on_demand_layer_focus = None;
        }
        self.update_keyboard_focus();
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
