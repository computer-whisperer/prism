//! `wlr_layer_shell` MVP — just enough to host the Spyder calibration
//! patch surface.
//!
//! **Scope (deliberate gaps):**
//! - Only `Layer::Overlay` surfaces are rendered today. `Background`,
//!   `Bottom`, `Top` are accepted (we send configure + track them) but
//!   they don't appear on screen — the layout walk happens between
//!   "we'd draw Background/Bottom" and "we'd draw Top/Overlay" only
//!   for Overlay.
//! - `KeyboardInteractivity` is treated as None for everyone. Per spec
//!   that's what `KeyboardInteractivity::None` says happens anyway;
//!   `OnDemand` and `Exclusive` need real input dispatch wiring.
//! - `exclusive_zone` is honored only as far as "ignore me" (`-1`) vs
//!   not. Positive values would shrink the layout work area; that
//!   needs the layout to grow an output-work-area concept it doesn't
//!   have today.
//! - `margin` is accepted but ignored — we always render at the
//!   anchored position. Spyder anchors fullscreen so margin doesn't
//!   apply anyway.
//!
//! What works end-to-end: a client anchored to all four edges of an
//! output (Spyder's pattern) gets a configure with the output's full
//! size, attaches a buffer, gets rendered on top of the workspace.
//!
//! The full layer-shell version (status bars, notification daemons,
//! wallpapers) is a follow-up — see `docs/phase-2-scanout-followups.md`.

use smithay::reexports::wayland_server::protocol::{wl_output::WlOutput, wl_surface::WlSurface};
use smithay::utils::{Logical, Rectangle, Size};
use smithay::wayland::shell::wlr_layer::{
    Anchor, KeyboardInteractivity, Layer, LayerSurface, LayerSurfaceCachedState,
    WlrLayerShellHandler, WlrLayerShellState,
};

use crate::state::{OutputId, PrismState};

/// Per-output entry for tracking layer-shell surfaces. We keep both
/// the smithay-managed handle (for state queries + send_configure)
/// and the layer assignment (used for Z-order at render time).
///
/// `output_id` is the connector this surface targets — either chosen
/// by the client via `output: Option<WlOutput>` in `get_layer_surface`
/// or assigned by us as a fallback (first available output) when the
/// client passed None.
pub struct LayerEntry {
    pub surface: LayerSurface,
    pub output_id: OutputId,
    pub layer: Layer,
    /// Last computed render rect in the output's logical coordinates.
    /// Refreshed on configure / commit / output size change. `None`
    /// before the first configure ack.
    pub last_rect: Option<Rectangle<i32, Logical>>,
}

impl LayerEntry {
    /// Compute the placement + size to send in a configure event,
    /// based on the surface's anchor / exclusive_zone / size and the
    /// containing output's logical size. Spyder's
    /// "anchor all four + 0×0 size" case → output's full size at
    /// (0, 0). Other anchor combinations are computed per the
    /// wlr-layer-shell spec.
    pub fn compute_rect(
        anchor: Anchor,
        size: Size<i32, Logical>,
        output_size: Size<i32, Logical>,
    ) -> Rectangle<i32, Logical> {
        // Per spec: dimensions of 0 mean "compositor sets it from
        // the anchored span." If anchored to both opposite edges of
        // an axis, the size in that axis is the output's size.
        let want_w = if anchor.contains(Anchor::LEFT) && anchor.contains(Anchor::RIGHT) {
            output_size.w
        } else if size.w > 0 {
            size.w
        } else {
            output_size.w
        };
        let want_h = if anchor.contains(Anchor::TOP) && anchor.contains(Anchor::BOTTOM) {
            output_size.h
        } else if size.h > 0 {
            size.h
        } else {
            output_size.h
        };

        // Position from anchor flags. Both edges anchored on an axis
        // ⇒ origin = 0 (spans the full output). Single edge ⇒ flush
        // to that edge. Neither ⇒ centered.
        let x = if anchor.contains(Anchor::LEFT) {
            0
        } else if anchor.contains(Anchor::RIGHT) {
            output_size.w - want_w
        } else {
            (output_size.w - want_w) / 2
        };
        let y = if anchor.contains(Anchor::TOP) {
            0
        } else if anchor.contains(Anchor::BOTTOM) {
            output_size.h - want_h
        } else {
            (output_size.h - want_h) / 2
        };

        Rectangle::new((x, y).into(), (want_w, want_h).into())
    }
}

impl PrismState {
    /// Resolve a client-supplied `WlOutput` (from `get_layer_surface`)
    /// to one of our tracked outputs. `None` ⇒ pick the first
    /// available output (spec-permitted fallback). Returns `None` if
    /// we have no outputs (the layer surface stays inert until one
    /// appears).
    fn resolve_layer_output(&self, wl_output: Option<&WlOutput>) -> Option<OutputId> {
        if let Some(req) = wl_output {
            for (id, output) in &self.wl_outputs {
                if output.owns(req) {
                    return Some(id.clone());
                }
            }
            tracing::warn!(
                "layer_shell: client requested an output we don't own; falling back to first"
            );
        }
        self.wl_outputs.keys().next().cloned()
    }

    /// Find the output's logical size from the layout's tracked
    /// outputs (which has the per-output scale applied). Falls back
    /// to the wl_output's announced mode if the layout doesn't know
    /// about the output yet (e.g. just-added, layout::add_output
    /// hasn't run).
    fn output_logical_size(&self, id: &OutputId) -> Size<i32, Logical> {
        if let Some(monitor) = self
            .wl_outputs
            .get(id)
            .and_then(|out| self.layout.monitor_for_output(out))
        {
            let s: Size<i32, Logical> = monitor.view_size().to_i32_round();
            return Size::from((s.w.max(1), s.h.max(1)));
        }
        // Fallback: use the output's current mode + integer scale.
        if let Some(output) = self.wl_outputs.get(id) {
            if let Some(mode) = output.current_mode() {
                let scale = output.current_scale().integer_scale().max(1);
                return Size::from((mode.size.w / scale, mode.size.h / scale));
            }
        }
        Size::from((1, 1))
    }

    /// Insert a freshly-created layer surface into the per-output
    /// tracking + send the initial configure with the resolved
    /// placement.
    pub fn layer_shell_new_surface(
        &mut self,
        surface: LayerSurface,
        wl_output: Option<WlOutput>,
        layer: Layer,
        namespace: String,
    ) {
        let Some(output_id) = self.resolve_layer_output(wl_output.as_ref()) else {
            tracing::warn!(
                namespace = %namespace,
                ?layer,
                "layer_shell: no outputs to host this surface; ignoring"
            );
            return;
        };

        // Pull the requested anchor/size from the cached state to
        // compute the initial configure. The first configure must
        // honor what the client asked for so the buffer it allocates
        // matches.
        let wl_surface = surface.wl_surface().clone();
        let (anchor, want_size, ki, layer_actual) =
            smithay::wayland::compositor::with_states(&wl_surface, |states| {
                let mut guard = states.cached_state.get::<LayerSurfaceCachedState>();
                let pending = guard.pending();
                (
                    pending.anchor,
                    pending.size,
                    pending.keyboard_interactivity,
                    pending.layer,
                )
            });
        if !matches!(ki, KeyboardInteractivity::None) {
            tracing::warn!(
                namespace = %namespace,
                "layer_shell MVP: KeyboardInteractivity != None requested but \
                 input dispatch isn't wired; treating as None"
            );
        }

        let output_size = self.output_logical_size(&output_id);
        let rect = LayerEntry::compute_rect(anchor, want_size, output_size);

        // Send initial configure with the computed size. send_configure
        // bumps the serial; we ignore the return value (smithay tracks
        // ack via ack_configure).
        surface.with_pending_state(|state| {
            state.size = Some(rect.size);
        });
        let _ = surface.send_configure();

        tracing::info!(
            namespace = %namespace,
            connector = %output_id,
            layer = ?layer_actual,
            anchor = ?anchor,
            rect = ?rect,
            "layer_shell: surface created + initial configure sent"
        );

        // Mark the surface's placement so dmabuf-feedback +
        // color-mgmt-feedback paths know which output it lives on
        // (mirrors what dispatch_surface_output_from_layout does for
        // xdg toplevels).
        smithay::wayland::compositor::with_states(&wl_surface, |states| {
            states
                .data_map
                .insert_if_missing_threadsafe(crate::surface_tex::SurfacePlacementSlot::default);
            let slot = states
                .data_map
                .get::<crate::surface_tex::SurfacePlacementSlot>()
                .unwrap();
            slot.0.lock().unwrap().current_output = Some(output_id.clone());
        });
        // wl_surface.enter so the client knows which output it's on.
        if let Some(output) = self.wl_outputs.get(&output_id) {
            output.enter(&wl_surface);
        }

        let entry = LayerEntry {
            surface,
            output_id: output_id.clone(),
            layer: layer_actual,
            last_rect: Some(rect),
        };
        self.layer_surfaces
            .entry(output_id.clone())
            .or_default()
            .push(entry);

        // Queue a redraw — the surface won't actually render until
        // its first buffer commit, but Tab-completing the redraw
        // pipeline now means we don't miss the next vblank when it
        // does.
        self.output_redraw
            .entry(output_id.clone())
            .or_default()
            .queue_redraw();
    }

    /// Called from `WlrLayerShellHandler::layer_destroyed`. Find +
    /// drop the entry, queue a redraw on the affected output so the
    /// surface's pixels actually leave the screen.
    pub fn layer_shell_destroyed(&mut self, removed: LayerSurface) {
        let mut affected: Option<OutputId> = None;
        for (id, list) in self.layer_surfaces.iter_mut() {
            if let Some(pos) = list.iter().position(|e| e.surface == removed) {
                list.swap_remove(pos);
                affected = Some(id.clone());
                break;
            }
        }
        if let Some(id) = affected {
            self.output_redraw.entry(id).or_default().queue_redraw();
        }
    }

    /// Iterate every layer surface assigned to `output_id`, in
    /// render order (Background, Bottom, Top, Overlay). Each entry
    /// yields its `WlSurface` and the rect (in output-logical
    /// pixels) where it should be drawn.
    pub fn layer_surfaces_for_output(
        &self,
        output_id: &OutputId,
    ) -> impl Iterator<Item = (&WlSurface, Rectangle<i32, Logical>, Layer)> {
        self.layer_surfaces
            .get(output_id)
            .into_iter()
            .flat_map(|list| list.iter())
            .filter_map(|entry| {
                let rect = entry.last_rect?;
                Some((entry.surface.wl_surface(), rect, entry.layer))
            })
    }
}

impl WlrLayerShellHandler for PrismState {
    fn shell_state(&mut self) -> &mut WlrLayerShellState {
        &mut self.layer_shell_state
    }

    fn new_layer_surface(
        &mut self,
        surface: LayerSurface,
        output: Option<WlOutput>,
        layer: Layer,
        namespace: String,
    ) {
        self.layer_shell_new_surface(surface, output, layer, namespace);
    }

    fn layer_destroyed(&mut self, surface: LayerSurface) {
        self.layer_shell_destroyed(surface);
    }
}

// Re-export so state.rs can wire the delegate macro.
pub use smithay::delegate_layer_shell;
