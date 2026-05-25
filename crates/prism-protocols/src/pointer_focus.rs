//! Pointer/touch focus resolution — which surface, and where, lies under a
//! point in global logical space.
//!
//! Ported from niri's `Niri::contents_under` (src/niri.rs:3283), reduced to
//! the subsystems prism has today: layout windows and `wlr_layer_shell`
//! surfaces. niri additionally consults the session-lock surface, the
//! exit-confirm / screenshot / window-MRU UIs, hot corners, and the overview;
//! prism has none of those yet, so they're omitted (add them here, in render
//! order, as they land).
//!
//! The returned origin is in *global* logical coordinates — exactly the focus
//! tuple smithay's `PointerHandle::motion` wants, so the deepest surface and
//! its correct surface-local mapping fall straight out.

use smithay::desktop::{layer_map_for_output, WindowSurfaceType};
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{Logical, Point};
use smithay::wayland::compositor::get_parent;
use smithay::wayland::shell::wlr_layer::Layer;

use prism_layout::layout::HitType;
use prism_layout::window::Mapped;

use crate::state::{OutputId, PrismState};

impl PrismState {
    /// Whether `surface` is part of an xdg_popup tree (the popup surface
    /// itself or a subsurface of one).
    ///
    /// Used to suppress window keyboard-focus changes (click-to-focus,
    /// focus-follows-mouse) when the pointer is over a popup. Moving the
    /// keyboard focus onto a popup surface sends `wl_keyboard.leave` to its
    /// parent toplevel, and clients like Firefox dismiss their own (grab-less)
    /// menus when the toplevel loses keyboard focus — so doing it on every
    /// hover/click over the menu made menu items impossible to use. Popups
    /// take keyboard focus only through a real popup grab, which prism honors
    /// separately in `XdgShellHandler::grab`.
    pub fn surface_is_popup(&self, surface: &WlSurface) -> bool {
        let mut root = surface.clone();
        while let Some(parent) = get_parent(&root) {
            root = parent;
        }
        self.popups.find_popup(&root).is_some()
    }
    /// The topmost surface under `pos` (global logical coords) together with
    /// that surface's origin in global logical coords, or `None` if nothing
    /// accepts input there.
    ///
    /// Z-order mirrors the render order, top-most first: overlay then top
    /// layer-shell, then layout windows (an interactively-moved window floats
    /// above the rest), then bottom then background layer-shell. Within a
    /// window or layer surface, smithay's `surface_under` descends popups and
    /// subsurfaces to the deepest surface whose input region contains the
    /// point — so subsurfaces and popups receive input, and the returned
    /// origin is that child surface's, not the toplevel's.
    pub fn contents_under(
        &self,
        pos: Point<f64, Logical>,
    ) -> Option<(WlSurface, Point<f64, Logical>)> {
        let output_id = self.output_containing((pos.x as i32, pos.y as i32))?;
        let output = self.wl_outputs.get(&output_id)?;
        let output_loc = output.current_location();
        let output_loc = Point::<f64, Logical>::from((output_loc.x as f64, output_loc.y as f64));
        let pos_within_output = pos - output_loc;

        // Resolve a layout-window hit. `HitType::Input { win_pos }` carries
        // the window buffer's position within the output (same semantics as
        // niri's, since `Tile::hit` is a direct port); offset into the window
        // and let smithay walk down to the deepest surface, then shift the
        // result back into output-local space.
        let window_hit = |(mapped, hit): (&Mapped, HitType)| {
            let HitType::Input { win_pos } = hit else {
                return None;
            };
            mapped
                .window
                .surface_under(pos_within_output - win_pos, WindowSurfaceType::ALL)
                .map(|(surface, surface_pos)| (surface, surface_pos.to_f64() + win_pos))
        };

        let within_output = self
            .layer_under(&output_id, pos_within_output, Layer::Overlay)
            .or_else(|| self.layer_under(&output_id, pos_within_output, Layer::Top))
            .or_else(|| {
                self.layout
                    .interactive_moved_window_under(output, pos_within_output)
                    .and_then(&window_hit)
            })
            .or_else(|| {
                self.layout
                    .window_under(output, pos_within_output)
                    .and_then(&window_hit)
            })
            .or_else(|| self.layer_under(&output_id, pos_within_output, Layer::Bottom))
            .or_else(|| self.layer_under(&output_id, pos_within_output, Layer::Background))?;

        // Lift the output-local origin into global space.
        Some((within_output.0, within_output.1 + output_loc))
    }

    /// Hit-test the layer-shell surfaces on `output_id` belonging to `layer`.
    /// Returns the deepest surface under the point and its origin in
    /// output-local coordinates.
    ///
    /// Delegates to the per-output `LayerMap`: `layer_under` picks the
    /// top-most layer surface (by z-order) whose bbox-with-popups contains the
    /// point, then `LayerSurface::surface_under` descends popups + subsurfaces
    /// to the deepest input surface.
    fn layer_under(
        &self,
        output_id: &OutputId,
        pos_within_output: Point<f64, Logical>,
        layer: Layer,
    ) -> Option<(WlSurface, Point<f64, Logical>)> {
        let output = self.wl_outputs.get(output_id)?;
        let map = layer_map_for_output(output);
        let ls = map.layer_under(layer, pos_within_output)?;
        let loc = map.layer_geometry(ls)?.loc.to_f64();
        ls.surface_under(pos_within_output - loc, WindowSurfaceType::ALL)
            .map(|(surface, surface_pos)| (surface, surface_pos.to_f64() + loc))
    }
}
