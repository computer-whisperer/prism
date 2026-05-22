//! `LayoutElement` — the contract every window/tile-able thing implements.
//!
//! Ported from niri's `layout/mod.rs` trait, with the render-emit shape
//! reworked to produce `prism_renderer::RenderEl` directly (no
//! `NiriRenderer` generic, no push closure, no offscreen / xray / bg
//! effect parameters that needed render_helpers infrastructure we
//! aren't carrying across). Everything else is faithfully preserved so
//! the layout port (`tile.rs`, `workspace.rs`, …) and window/layer
//! impls plug in unchanged from the call-site side.
//!
//! Render-side parameters that survived: `location` in logical pixels,
//! `scale` (output fractional scale), `alpha` (per-element opacity
//! multiplier), and the output `Vec<RenderEl>`. Niri's `RenderCtx<R>`
//! is gone — there's no per-renderer cache to carry.
//!
//! Render-side parameters that were dropped: `XrayPos` (debug overlay),
//! `BackgroundEffectElement` (custom GLES shaders), `OffscreenData`
//! (offscreen render-to-texture). When prism eventually wants these
//! they'll be plumbed back in.

use prism_config::CornerRadius;
use prism_renderer::{vk, RenderEl, SurfaceColorParams};
use smithay::output::{self, Output};
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::{Logical, Point, Rectangle, Scale, Serial, Size, Transform};
use smithay::wayland::compositor::SurfaceData;

/// Side-channel a `LayoutElement` needs to actually emit content during a
/// render walk: how to project logical rects into clip space, and how to
/// resolve a surface (via its already-locked `SurfaceData`) to the
/// GPU-side texture view the render path will sample. Constructed once
/// per output at the top of the render walk by the integrator
/// (prism-protocols' `present_for_crtc`), then threaded through
/// `Monitor → Workspace → Tile → Mapped`.
///
/// We bundle the texture lookup behind a `&dyn Fn` instead of a typed
/// handle so prism-layout doesn't need to know how the integrator stores
/// per-surface textures (today: `SurfaceTexSlot` on the surface's
/// data_map; tomorrow possibly cached elsewhere).
///
/// **Why `&SurfaceData` and not `&WlSurface`**: the renderer's
/// per-surface walk uses `with_surface_tree_downward`, whose visit
/// callback already holds the surface's `SurfaceData` lock. Resolving
/// the texture by calling `with_states(surface, ...)` again would
/// re-acquire the same non-reentrant std::sync::Mutex inside the
/// callback and deadlock — exactly the hang we saw on the first
/// post-map present for the mpv-bearing output. Taking the already
/// borrowed `&SurfaceData` keeps the lookup inside the existing scope.
pub struct RenderCtx<'a> {
    pub texture_lookup: &'a dyn Fn(&SurfaceData) -> Option<vk::ImageView>,
    /// Look up the surface's color-decoding parameters (TF +
    /// reference white) from its `wp_color_management_v1` image
    /// description. Same shape as `texture_lookup`: closure over
    /// `&SurfaceData` so we don't double-acquire the `with_states`
    /// lock during a surface-tree walk. Returning `None` falls back
    /// to a sRGB EOTF with the output's `sdr_reference_nits` as the
    /// white-point luminance.
    pub color_lookup: &'a dyn Fn(&SurfaceData) -> Option<SurfaceColorParams>,
    /// Per-output luminance to map "color-unaware client white" to,
    /// in cd/m². Used as the default `sdr_white_nits` for surfaces
    /// with no `wp_color_management_v1` description. IEC sRGB default
    /// is 80; HDR-configured outputs typically want higher.
    pub sdr_reference_nits: f32,
}

impl<'a> RenderCtx<'a> {
    pub fn texture_for(&self, states: &SurfaceData) -> Option<vk::ImageView> {
        (self.texture_lookup)(states)
    }
    pub fn color_for(&self, states: &SurfaceData) -> SurfaceColorParams {
        (self.color_lookup)(states).unwrap_or(SurfaceColorParams {
            transfer: 1,
            sdr_white_nits: self.sdr_reference_nits,
        })
    }
}

use super::LayoutElementRenderSnapshot;
use crate::utils::transaction::Transaction;
use crate::utils::ResizeEdge;
use crate::window::ResolvedWindowRules;

/// Size-relative units marker. Used as the `Coordinate` parameter to
/// smithay's `Size<f64, SizeFrac>` for column widths expressed as a
/// fraction of the workspace.
#[derive(Debug, Clone, Copy)]
pub struct SizeFrac;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizingMode {
    Normal,
    Maximized,
    Fullscreen,
}

impl SizingMode {
    pub fn is_normal(self) -> bool {
        matches!(self, Self::Normal)
    }

    pub fn is_maximized(self) -> bool {
        matches!(self, Self::Maximized)
    }

    pub fn is_fullscreen(self) -> bool {
        matches!(self, Self::Fullscreen)
    }
}

/// Interactive-resize state held in `LayoutElement::set_interactive_resize`.
/// Carries the edge mask so the configure path knows which sides are
/// being dragged.
#[derive(Debug, Clone, Copy)]
pub struct InteractiveResizeData {
    pub edges: ResizeEdge,
}

#[derive(Debug, Clone, Copy)]
pub enum ConfigureIntent {
    /// A configure is not needed (no changes to server pending state).
    NotNeeded,
    /// A configure is throttled (due to resizing too fast for example).
    Throttled,
    /// Can send the configure if it isn't throttled externally (only size changed).
    CanSend,
    /// Should send the configure regardless of external throttling (something other than size
    /// changed).
    ShouldSend,
}

/// Renderer-agnostic, prism-flavoured LayoutElement trait. The contract
/// every tile-able window implements (today: `crate::window::Mapped`).
pub trait LayoutElement {
    /// Unique-ID type. PartialEq + Debug + Clone are the only requirements.
    type Id: PartialEq + std::fmt::Debug + Clone;

    fn id(&self) -> &Self::Id;

    /// Re-derive cached config-dependent state from a freshly-loaded
    /// `blur_config` (niri's name; the parameter type is general
    /// enough to carry future tunables).
    fn update_config(&mut self, blur_config: prism_config::Blur) {
        let _ = blur_config;
    }

    /// Visual size, excluding CSD shadows — matches `xdg_surface::set_window_geometry`.
    fn size(&self) -> Size<i32, Logical>;

    /// Buffer top-left relative to the element's visual geometry.
    /// Negative if the buffer extends past the geometry (CSD shadows).
    fn buf_loc(&self) -> Point<i32, Logical>;

    /// Hit-test the element's input region. `point` is relative to the
    /// element's visual top-left.
    fn is_in_input_region(&self, point: Point<f64, Logical>) -> bool;

    /// Emit all this element's draw calls into `out`. Default impl
    /// emits popups then the normal surface.
    fn render(
        &self,
        location: Point<f64, Logical>,
        scale: Scale<f64>,
        alpha: f32,
        project: &dyn Fn(Rectangle<f64, Logical>) -> [f32; 4],
        ctx: &RenderCtx<'_>,
        out: &mut Vec<RenderEl>,
    ) {
        self.render_popups(location, scale, alpha, project, ctx, out);
        self.render_normal(location, scale, alpha, project, ctx, out);
    }

    /// Emit the non-popup parts of the element. Default = no-op.
    fn render_normal(
        &self,
        location: Point<f64, Logical>,
        scale: Scale<f64>,
        alpha: f32,
        project: &dyn Fn(Rectangle<f64, Logical>) -> [f32; 4],
        ctx: &RenderCtx<'_>,
        out: &mut Vec<RenderEl>,
    ) {
        let _ = (location, scale, alpha, project, ctx, out);
    }

    /// Emit the popup-tree parts of the element. Default = no-op.
    fn render_popups(
        &self,
        location: Point<f64, Logical>,
        scale: Scale<f64>,
        alpha: f32,
        project: &dyn Fn(Rectangle<f64, Logical>) -> [f32; 4],
        ctx: &RenderCtx<'_>,
        out: &mut Vec<RenderEl>,
    ) {
        let _ = (location, scale, alpha, project, ctx, out);
    }

    fn request_size(
        &mut self,
        size: Size<i32, Logical>,
        mode: SizingMode,
        animate: bool,
        transaction: Option<Transaction>,
    );

    fn request_size_once(&mut self, size: Size<i32, Logical>, animate: bool) {
        self.request_size(size, SizingMode::Normal, animate, None);
    }

    fn min_size(&self) -> Size<i32, Logical>;
    fn max_size(&self) -> Size<i32, Logical>;

    fn is_wl_surface(&self, wl_surface: &WlSurface) -> bool;

    /// Whether the element draws its own server-side decorations.
    fn has_ssd(&self) -> bool;

    fn set_preferred_scale_transform(&self, scale: output::Scale, transform: Transform);

    fn output_enter(&self, output: &Output);
    fn output_leave(&self, output: &Output);

    fn set_activated(&mut self, active: bool);
    fn set_active_in_column(&mut self, active: bool);
    fn set_floating(&mut self, floating: bool);

    fn set_bounds(&self, bounds: Size<i32, Logical>);
    fn is_ignoring_opacity_window_rule(&self) -> bool;

    fn is_urgent(&self) -> bool;

    fn configure_intent(&self) -> ConfigureIntent;
    fn send_pending_configure(&mut self);

    /// Element's current sizing mode. *Does not* switch immediately on
    /// `request_size`; reflects what the client has actually applied.
    fn sizing_mode(&self) -> SizingMode;
    /// Sizing mode we're requesting. Switches immediately on `request_size`.
    fn pending_sizing_mode(&self) -> SizingMode;

    fn requested_size(&self) -> Option<Size<i32, Logical>>;

    /// Non-fullscreen size we expect this window has or will shortly have.
    /// `None` means there's no known expected size (e.g. fullscreen).
    fn expected_size(&self) -> Option<Size<i32, Logical>> {
        if self.sizing_mode().is_fullscreen() {
            return None;
        }
        let mut requested = self.requested_size().unwrap_or_default();
        let current = self.size();
        if requested.w == 0 {
            requested.w = current.w;
        }
        if requested.h == 0 {
            requested.h = current.h;
        }
        Some(requested)
    }

    fn is_windowed_fullscreen(&self) -> bool {
        false
    }
    fn is_pending_windowed_fullscreen(&self) -> bool {
        false
    }
    fn request_windowed_fullscreen(&mut self, value: bool) {
        let _ = value;
    }

    /// Effective corner radius. Returns zero in windowed fullscreen.
    /// (Other fullscreen / maximize modes are handled by the surrounding
    /// `Tile`, not the element itself.)
    fn geometry_corner_radius(&self) -> CornerRadius {
        if self.is_windowed_fullscreen() {
            return CornerRadius::default();
        }
        self.rules().geometry_corner_radius.unwrap_or_default()
    }

    fn is_child_of(&self, parent: &Self) -> bool;

    fn rules(&self) -> &ResolvedWindowRules;

    /// Periodic clean-up tick (called once per layout refresh).
    fn refresh(&self);

    fn set_interactive_resize(&mut self, data: Option<InteractiveResizeData>);
    fn cancel_interactive_resize(&mut self);
    fn interactive_resize_data(&self) -> Option<InteractiveResizeData>;

    fn on_commit(&mut self, serial: Serial);

    /// Pop a saved render snapshot for the about-to-happen resize
    /// animation. Niri returns a `LayoutElementRenderSnapshot` (a baked
    /// texture buffer); prism's `LayoutElementRenderSnapshot` is stubbed
    /// to `()` until the snapshot pipeline lands, so this returns
    /// `Option<()>` carrying just the "have a snapshot" bit. The
    /// surrounding `Tile` still consults it to decide whether to play
    /// the resize animation.
    fn take_animation_snapshot(&mut self) -> Option<LayoutElementRenderSnapshot>;
}

/// Workspace-view rect helper — niri's render path projects element
/// positions through this. Kept here so the small layout pieces and the
/// upcoming tile port share one definition.
pub type ViewRect = Rectangle<f64, Logical>;
