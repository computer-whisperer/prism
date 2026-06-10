//! Pointer/touch focus resolution — which surface, and where, lies under a
//! point in global logical space.
//!
//! Ported from niri's `Niri::contents_under` (src/niri.rs:3283), reduced to
//! the subsystems prism has today: layout windows, `wlr_layer_shell`
//! surfaces, and the hot corner. niri additionally consults the session-lock
//! surface and the exit-confirm / screenshot / window-MRU UIs; prism has
//! none of those yet, so they're omitted (add them here, in render order,
//! as they land).
//!
//! The returned origin is in *global* logical coordinates — exactly the focus
//! tuple smithay's `PointerHandle::motion` wants, so the deepest surface and
//! its correct surface-local mapping fall straight out.

use smithay::desktop::{layer_map_for_output, WindowSurfaceType};
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{Logical, Point};
use smithay::wayland::compositor::get_parent;
use smithay::wayland::pointer_constraints::with_pointer_constraint;
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

        // Locked session: the lock surface is the only input target —
        // no layer chrome, no windows, no hot corner (niri.rs:3297).
        // With no (live) lock surface on this output, nothing accepts
        // input.
        if self.is_locked() {
            let ls = self.lock_surfaces.get(&output_id).filter(|ls| ls.alive())?;
            let (surface, surface_pos) = smithay::desktop::utils::under_from_surface_tree(
                ls.wl_surface(),
                pos_within_output,
                (0, 0),
                WindowSurfaceType::ALL,
            )?;
            return Some((surface, surface_pos.to_f64() + output_loc));
        }

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

        let overlay = self.layer_under(&output_id, pos_within_output, Layer::Overlay);

        // The hot-corner trigger pixel is dead to everything below the
        // Overlay layer (niri checks it between Overlay and Top,
        // niri.rs:3420): a bar anchored into the corner doesn't swallow
        // the trigger, and clients never see hover or clicks there. The
        // motion handlers re-check the geometry to do the actual toggle
        // (prism returns a plain surface tuple, not niri's
        // `PointContents` with a `hot_corner` flag).
        if overlay.is_none() && self.is_inside_hot_corner(output, pos_within_output) {
            return None;
        }

        let within_output = overlay
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

    /// Whether `pos_within_output` lies inside a configured hot corner of
    /// `output` — a 1×1 logical pixel at each enabled corner. Per-output
    /// `hot-corners` config overrides the global `gestures` section; when
    /// the user set no corner explicitly, top-left is the default. Ported
    /// from niri's `is_inside_hot_corner` (niri.rs:3057).
    pub fn is_inside_hot_corner(
        &self,
        output: &smithay::output::Output,
        pos_within_output: Point<f64, Logical>,
    ) -> bool {
        let config = self.config.borrow();
        let hot_corners = output
            .user_data()
            .get::<prism_config::output::OutputName>()
            .and_then(|name| config.outputs.find(name))
            .and_then(|c| c.hot_corners)
            .unwrap_or(config.gestures.hot_corners);

        if hot_corners.off {
            return false;
        }

        // Logical output size, same math as `output_containing` /
        // the pointer clamp (physical mode / fractional scale, rounded)
        // so the corner pixels are exactly reachable.
        let Some(mode) = output.current_mode() else {
            return false;
        };
        let scale = output.current_scale().fractional_scale().max(0.01);
        let w = (mode.size.w as f64 / scale).round();
        let h = (mode.size.h as f64 / scale).round();

        let contains = |corner: Point<f64, Logical>| {
            smithay::utils::Rectangle::new(corner, smithay::utils::Size::from((1., 1.)))
                .contains(pos_within_output)
        };

        if hot_corners.top_right && contains(Point::from((w - 1., 0.))) {
            return true;
        }
        if hot_corners.bottom_left && contains(Point::from((0., h - 1.))) {
            return true;
        }
        if hot_corners.bottom_right && contains(Point::from((w - 1., h - 1.))) {
            return true;
        }

        // If the user didn't explicitly set any corners, default to top-left.
        if (hot_corners.top_left
            || !(hot_corners.top_right || hot_corners.bottom_right || hot_corners.bottom_left))
            && contains(Point::from((0., 0.)))
        {
            return true;
        }

        false
    }

    /// Activate the pointer constraint on the surface under the pointer, if
    /// one exists and the conditions are met: the pointer must be focused on
    /// that surface and, for a constraint with a region, inside the region.
    ///
    /// Called after pointer focus settles (motion handlers) and when a client
    /// creates a new constraint. smithay handles *de*activation automatically
    /// when pointer focus leaves the surface. Ported from niri's
    /// `Niri::maybe_activate_pointer_constraint`.
    pub fn maybe_activate_pointer_constraint(&self) {
        let Some((surface, surface_loc)) = &self.pointer_contents else {
            return;
        };
        let Some(pointer) = self.seat.get_pointer() else {
            return;
        };
        // Only activate if this surface actually holds the pointer focus.
        if Some(surface) != pointer.current_focus().as_ref() {
            return;
        }

        with_pointer_constraint(surface, &pointer, |constraint| {
            let Some(constraint) = constraint else {
                return;
            };
            if constraint.is_active() {
                return;
            }
            // A region-limited constraint only applies while the pointer is
            // inside the region.
            if let Some(region) = constraint.region() {
                let pos_within_surface = self.pointer_pos - *surface_loc;
                if !region.contains(pos_within_surface.to_i32_round()) {
                    return;
                }
            }
            constraint.activate();
        });
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
