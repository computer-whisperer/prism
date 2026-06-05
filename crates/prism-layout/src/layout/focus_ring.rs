//! Focus / border ring around tiles.
//!
//! Simplified from `niri/src/layout/focus_ring.rs`: niri's version uses a
//! custom GLES `BorderRenderElement` shader (eight stripe/corner patches);
//! this port renders single-colour rings — sharp rings via the cheap
//! four-stripe [`prism_renderer::BorderEl`], rounded rings / fills via the
//! SDF-backed [`prism_renderer::RoundedBoxEl`] (one quad, per-corner
//! coverage in the decode shader). The split mirrors niri's
//! `use_border_shader` heuristic: only pay the full-bounding-box quad when
//! a corner radius actually asks for it.
//!
//! Visual deficit vs niri (revisit later):
//!   - no gradients (active_gradient / inactive_gradient / urgent_gradient
//!     all collapse to the corresponding flat colour)
//!   - no thicken-corners hack (the SDF has no corner-seam bleed to hide)
//!
//! API preserved so the layout port (`tile.rs`) compiles unchanged.

use prism_frame::ElementId;
use prism_renderer::{srgb_to_bt2020_nits, BorderEl, RenderEl, RoundedBoxEl};
use smithay::utils::{Logical, Point, Rectangle, Size};

use crate::utils::round_logical_in_physical_max1;

/// Default SDR-white nits used to project sRGB-described border colours
/// into the renderer's BT.2020 linear nits. Matches the niri default
/// (`80` is the historical sRGB reference white). Per-output SDR white
/// will eventually be threaded through here and replace this constant.
const DEFAULT_SDR_WHITE_NITS: f32 = 80.0;

#[derive(Debug)]
pub struct FocusRing {
    config: prism_config::FocusRing,
    /// Logical-pixel width of the ring (mirrors `config.width`). Cached
    /// because callers ask for it many times per frame.
    width: f64,
    /// Stable cross-frame element id, allocated once per ring. Lives here (not
    /// re-derived per frame) so the damage tracker sees the same id every
    /// frame — niri's "id in the cached SolidColorBuffer" pattern.
    id: ElementId,
    /// Geometry produced by the most recent `update_render_elements` call.
    /// `None` until first update or when the ring is off.
    cached: Option<CachedGeometry>,
}

#[derive(Debug, Clone)]
struct CachedGeometry {
    /// Outer rect of the ring, in *logical* coordinates relative to the
    /// window's visual top-left. With `width = W`, the outer rect is
    /// `((-W, -W), (win_w + 2W, win_h + 2W))`.
    outer: Rectangle<f64, Logical>,
    color_bt2020_nits: [f32; 4],
    /// Whether to render as a hollow ring (`is_border = true`, used when
    /// the window has a CSD border) or a filled rect (`is_border = false`,
    /// used as a backdrop behind windows without SSDs). Niri's name; we
    /// keep it for porting fidelity.
    is_border: bool,
    /// Per-side ring thickness in logical pixels. All four are `width`;
    /// kept as four fields to match `BorderEl::thickness` /
    /// `RoundedBoxEl::inset`.
    thickness_logical: [f64; 4],
    /// Per-corner radii of the ring's *outer* rect in logical pixels,
    /// `[tl, tr, br, bl]`. All zero → sharp ring (stripe path).
    radii: [f32; 4],
}

impl FocusRing {
    pub fn new(config: prism_config::FocusRing) -> Self {
        let width = config.width;
        Self {
            config,
            width,
            id: ElementId::alloc(),
            cached: None,
        }
    }

    pub fn update_config(&mut self, config: prism_config::FocusRing) {
        self.width = config.width;
        self.config = config;
    }

    /// No-op for now — niri uses this to invalidate per-element shader
    /// state after a config / shader-source reload. We don't have a
    /// shader cache to invalidate.
    pub fn update_shaders(&mut self) {}

    pub fn config(&self) -> &prism_config::FocusRing {
        &self.config
    }

    pub fn width(&self) -> f64 {
        self.width
    }

    pub fn is_off(&self) -> bool {
        self.config.off
    }

    /// Recompute the ring's geometry from the surrounding tile state.
    /// Called once per frame per tile.
    ///
    /// `radius` is the radius of the ring's *outer* corners — the caller
    /// (tile.rs) has already expanded the window's `geometry-corner-radius`
    /// by the border / ring widths, niri-style. `view_rect` is accepted to
    /// match niri's signature (future gradient support) but ignored.
    #[allow(clippy::too_many_arguments)] // mirrors niri's signature
    pub fn update_render_elements(
        &mut self,
        win_size: Size<f64, Logical>,
        is_active: bool,
        is_border: bool,
        is_urgent: bool,
        _view_rect: Rectangle<f64, Logical>,
        radius: prism_config::CornerRadius,
        scale: f64,
        alpha: f32,
    ) {
        if self.config.off || self.width == 0.0 {
            self.cached = None;
            return;
        }

        let color = if is_urgent {
            self.config.urgent_color
        } else if is_active {
            self.config.active_color
        } else {
            self.config.inactive_color
        };

        // Unpremul, because the renderer's shader multiplies the colour
        // by alpha at output time — feeding it premul would double-darken.
        let mut rgba = color.to_array_unpremul();
        rgba[3] *= alpha;

        let color_bt2020_nits =
            srgb_to_bt2020_nits(rgba[0], rgba[1], rgba[2], rgba[3], DEFAULT_SDR_WHITE_NITS);

        let w = self.width;
        let outer = Rectangle::new(
            Point::from((-w, -w)),
            Size::from((win_size.w + 2. * w, win_size.h + 2. * w)),
        );

        // Overlapping corner radii shrink CSS-style to fit the outer rect
        // (niri does the same against its full_size).
        let radius = radius.fit_to(outer.size.w as f32, outer.size.h as f32);
        let radii = [
            radius.top_left,
            radius.top_right,
            radius.bottom_right,
            radius.bottom_left,
        ];

        // Snap thickness to physical-pixel multiples so 1-px rings don't
        // disappear under fractional scaling.
        let snapped = round_logical_in_physical_max1(scale, w);
        let thickness_logical = [snapped; 4];

        self.cached = Some(CachedGeometry {
            outer,
            color_bt2020_nits,
            is_border,
            thickness_logical,
            radii,
        });
    }

    /// Append this ring's draw elements onto `out` in output-space logical
    /// pixels. `location` is the visual top-left of the owning window in
    /// logical pixels; the renderer projects to clip space at lowering time.
    pub fn render(&self, location: Point<f64, Logical>, out: &mut Vec<RenderEl>) {
        let Some(cached) = &self.cached else {
            return;
        };

        let outer_logical = Rectangle::new(cached.outer.loc + location, cached.outer.size);

        // Rounded rings/fills go through the SDF quad; sharp ones keep the
        // cheaper stripe / plain-quad elements (which also rasterize only the
        // painted area and, for the fill, occlude their full rect).
        let rounded = cached.radii.iter().any(|r| *r > 0.0);

        if rounded {
            out.push(RenderEl::RoundedBox(RoundedBoxEl {
                id: self.id,
                geometry: outer_logical,
                radii: cached.radii,
                inset: cached.is_border.then_some(cached.thickness_logical),
                color_bt2020_nits: cached.color_bt2020_nits,
            }));
        } else if cached.is_border {
            out.push(RenderEl::Border(BorderEl {
                id: self.id,
                geometry: outer_logical,
                thickness: cached.thickness_logical,
                color_bt2020_nits: cached.color_bt2020_nits,
            }));
        } else {
            out.push(RenderEl::SolidColor(prism_renderer::SolidColorEl {
                id: self.id,
                geometry: outer_logical,
                color_bt2020_nits: cached.color_bt2020_nits,
            }));
        }
    }
}
