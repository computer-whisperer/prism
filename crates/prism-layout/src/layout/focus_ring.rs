//! Focus / border ring around tiles.
//!
//! Heavily simplified from `niri/src/layout/focus_ring.rs`: niri's version
//! uses a custom GLES `BorderRenderElement` shader to draw per-corner
//! rounded gradient borders with optional urgent / active colour state and
//! gradient fades. We don't have a Vulkan equivalent of that shader yet,
//! so this port renders sharp-cornered, single-colour rings via
//! [`prism_renderer::BorderEl`].
//!
//! Visual deficit vs niri (revisit when the Vulkan border shader lands):
//!   - no rounded corners (config `geometry-corner-radius` ignored here)
//!   - no gradients (active_gradient / inactive_gradient / urgent_gradient
//!     all collapse to the corresponding flat colour)
//!   - no thicken-corners hack (we don't paint corners separately)
//!
//! API preserved so the layout port (`tile.rs`) compiles unchanged.

use prism_renderer::{srgb_to_bt2020_nits, BorderEl, RenderEl};
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
    /// Per-side ring thickness in logical pixels. All four are `width`
    /// for sharp-cornered, unrounded rings, but we keep them as four
    /// fields so the future rounded-corner port can drop in cleanly.
    thickness_logical: [f64; 4],
}

impl FocusRing {
    pub fn new(config: prism_config::FocusRing) -> Self {
        let width = config.width;
        Self {
            config,
            width,
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
    /// `view_rect` and `radius` are accepted to match niri's signature
    /// (for future rounded-corner / gradient support) but currently
    /// ignored.
    #[allow(clippy::too_many_arguments)] // mirrors niri's signature
    pub fn update_render_elements(
        &mut self,
        win_size: Size<f64, Logical>,
        is_active: bool,
        is_border: bool,
        is_urgent: bool,
        _view_rect: Rectangle<f64, Logical>,
        _radius: prism_config::CornerRadius,
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

        // Snap thickness to physical-pixel multiples so 1-px rings don't
        // disappear under fractional scaling.
        let snapped = round_logical_in_physical_max1(scale, w);
        let thickness_logical = [snapped; 4];

        self.cached = Some(CachedGeometry {
            outer,
            color_bt2020_nits,
            is_border,
            thickness_logical,
        });
    }

    /// Append this ring's draw elements onto `out`, projecting into the
    /// supplied output coord system. `location` is the visual top-left of
    /// the owning window in logical pixels; `project` is the
    /// logical→clip-space transform the caller provides (typically
    /// composed from the output extent + any tile-level scale/translate).
    pub fn render(
        &self,
        location: Point<f64, Logical>,
        project: &impl Fn(Rectangle<f64, Logical>) -> [f32; 4],
        out: &mut Vec<RenderEl>,
    ) {
        let Some(cached) = &self.cached else {
            return;
        };

        let outer_logical = Rectangle::new(cached.outer.loc + location, cached.outer.size);

        if cached.is_border {
            // Per-side thickness in clip space: project the outer rect
            // and the inner rect (outer shrunk by `thickness_logical` on
            // each side), then take the difference along each axis. This
            // routes the caller's logical→clip projection through
            // without needing to know the output's pixel scale.
            let outer_clip = project(outer_logical);
            let [t, r, b, l] = cached.thickness_logical;
            let inner_logical = Rectangle::new(
                outer_logical.loc + Point::from((l, t)),
                Size::from((
                    outer_logical.size.w - (l + r),
                    outer_logical.size.h - (t + b),
                )),
            );
            let inner_clip = project(inner_logical);
            let thickness_clip = [
                inner_clip[1] - outer_clip[1], // top
                outer_clip[2] - inner_clip[2], // right
                outer_clip[3] - inner_clip[3], // bottom
                inner_clip[0] - outer_clip[0], // left
            ];

            out.push(RenderEl::Border(BorderEl {
                rect_clip: outer_clip,
                thickness_clip,
                color_bt2020_nits: cached.color_bt2020_nits,
            }));
        } else {
            let rect_clip = project(outer_logical);
            out.push(RenderEl::SolidColor(prism_renderer::SolidColorEl {
                rect_clip,
                color_bt2020_nits: cached.color_bt2020_nits,
            }));
        }
    }
}
