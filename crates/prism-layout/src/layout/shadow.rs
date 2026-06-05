//! Drop-shadow rendering.
//!
//! Ported from `niri/src/layout/shadow.rs` against prism's SDF-backed
//! [`prism_renderer::ShadowEl`] (decode-pass Gaussian rounded-box shadow,
//! Evan Wallace's approximation). One difference from niri: when
//! `draw-behind-window false`, niri splits the shader quad into up to
//! eight rects that skirt the window's interior; prism draws one quad and
//! cuts the window region out in the shader (`sdf_box2`), which produces
//! identical pixels — the splitting is a GLES-era fill-rate optimization
//! the damage-scissored decode pass doesn't need.

use prism_frame::ElementId;
use prism_renderer::{srgb_to_bt2020_nits, RenderEl, ShadowEl, SurfaceClip};
use smithay::utils::{Logical, Point, Rectangle, Size};

/// Default SDR-white nits for projecting the configured sRGB shadow colour
/// into BT.2020 linear nits — same constant as the focus ring; a black
/// shadow only carries alpha, so this matters only for tinted shadows.
const DEFAULT_SDR_WHITE_NITS: f32 = 80.0;

#[derive(Debug)]
pub struct Shadow {
    config: prism_config::Shadow,
    /// Stable cross-frame element id (one shadow = one element).
    id: ElementId,
    /// Geometry from the most recent `update_render_elements`, in logical
    /// coordinates relative to the window's visual top-left. `None` until
    /// first update or when shadows are off.
    cached: Option<CachedShadow>,
}

#[derive(Debug, Clone)]
struct CachedShadow {
    /// Quad rect: the shadow box expanded by the blur reach (3σ).
    geometry: Rectangle<f64, Logical>,
    /// The casting box: window offset by the configured offset, grown by
    /// `spread`.
    shadow_box: Rectangle<f64, Logical>,
    /// Per-corner radii of `shadow_box`, `[tl, tr, br, bl]`.
    radii: [f32; 4],
    /// Gaussian sigma (`softness / 2`).
    sigma: f32,
    /// `draw-behind-window false`: the window's own rounded box to cut out.
    cutout: Option<SurfaceClip>,
    color_bt2020_nits: [f32; 4],
}

impl Shadow {
    pub fn new(config: prism_config::Shadow) -> Self {
        Self {
            config,
            id: ElementId::alloc(),
            cached: None,
        }
    }

    pub fn update_config(&mut self, config: prism_config::Shadow) {
        self.config = config;
    }

    pub fn update_shaders(&mut self) {}

    pub fn config(&self) -> &prism_config::Shadow {
        &self.config
    }

    /// Recompute the shadow geometry from the surrounding tile state.
    /// Mirrors niri's math exactly: σ = softness/2 (CSS box-shadow), the
    /// quad extends ceil(3σ) past the spread-expanded box, and offsets /
    /// spread are ceiled to physical pixels.
    pub fn update_render_elements(
        &mut self,
        win_size: Size<f64, Logical>,
        is_active: bool,
        radius: prism_config::CornerRadius,
        scale: f64,
        alpha: f32,
    ) {
        if !self.config.on {
            self.cached = None;
            return;
        }

        let ceil = |logical: f64| (logical * scale).ceil() / scale;

        // Like in CSS box-shadow.
        let sigma = self.config.softness / 2.;
        // Blur reach: draw all pixels the Gaussian meaningfully touches.
        let width = ceil(sigma * 3.);

        let offset = self.config.offset;
        let offset = Point::<f64, Logical>::from((ceil(offset.x.0), ceil(offset.y.0)));

        let spread = self.config.spread;
        let spread = ceil(spread.abs()).copysign(spread);
        let box_loc = offset - Point::from((spread, spread));

        let win_radius = radius.fit_to(win_size.w as f32, win_size.h as f32);

        let box_size = if spread >= 0. {
            win_size + Size::from((spread, spread)).upscale(2.)
        } else {
            // Saturating shrink.
            Size::from((
                (win_size.w + 2. * spread).max(0.),
                (win_size.h + 2. * spread).max(0.),
            ))
        };
        let shadow_box = Rectangle::new(box_loc, box_size);
        let box_radius = win_radius.expanded_by(spread as f32);

        let geometry = Rectangle::new(
            box_loc - Point::from((width, width)),
            box_size + Size::from((width, width)).upscale(2.),
        );

        let color = if is_active {
            self.config.color
        } else {
            // Default to slightly more transparent (niri: color * 0.75).
            self.config
                .inactive_color
                .unwrap_or(self.config.color * 0.75)
        };
        let mut rgba = color.to_array_unpremul();
        rgba[3] *= alpha;
        let color_bt2020_nits =
            srgb_to_bt2020_nits(rgba[0], rgba[1], rgba[2], rgba[3], DEFAULT_SDR_WHITE_NITS);

        let cutout = (!self.config.draw_behind_window).then_some(SurfaceClip {
            rect: Rectangle::new(Point::from((0., 0.)), win_size),
            radii: [
                win_radius.top_left,
                win_radius.top_right,
                win_radius.bottom_right,
                win_radius.bottom_left,
            ],
        });

        self.cached = Some(CachedShadow {
            geometry,
            shadow_box,
            radii: [
                box_radius.top_left,
                box_radius.top_right,
                box_radius.bottom_right,
                box_radius.bottom_left,
            ],
            sigma: sigma as f32,
            cutout,
            color_bt2020_nits,
        });
    }

    /// Append this shadow's element onto `out` in output-space logical
    /// pixels. `location` is the visual top-left of the owning window.
    pub fn render(&self, location: Point<f64, Logical>, out: &mut Vec<RenderEl>) {
        let Some(cached) = &self.cached else {
            return;
        };

        let offset_rect = |r: Rectangle<f64, Logical>| Rectangle::new(r.loc + location, r.size);

        out.push(RenderEl::Shadow(ShadowEl {
            id: self.id,
            geometry: offset_rect(cached.geometry),
            shadow_box: offset_rect(cached.shadow_box),
            radii: cached.radii,
            sigma: cached.sigma,
            cutout: cached.cutout.map(|c| SurfaceClip {
                rect: offset_rect(c.rect),
                radii: c.radii,
            }),
            color_bt2020_nits: cached.color_bt2020_nits,
        }));
    }
}
