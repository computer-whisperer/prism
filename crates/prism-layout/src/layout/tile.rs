//! `Tile<W: LayoutElement>` — one window with its decorations.
//!
//! State-machine ported wholesale from niri/src/layout/tile.rs (sizing
//! mode dance, open/resize/move/alpha animations, fullscreen/maximize
//! geometry, interactive resize bookkeeping). Render emission ported in
//! a stripped form: niri's `render_inner` composed ~10 GLES-bound
//! element types into a `TileRenderElement`; prism's `render` produces a
//! flat `Vec<RenderEl>` via the per-element `render` calls already in
//! place on `FocusRing`/`Shadow` plus a single `Surface` push from the
//! window. Snapshot crossfades, the fullscreen backdrop, and
//! clipped-surface rounded corners (`clip-to-geometry`, via the decode
//! pass's SDF coverage) have since been wired up; offscreen-buffered
//! alpha and background blur remain dropped until the corresponding
//! Vulkan paths exist. See [`crate::layout::shadow`] and
//! [`crate::layout::focus_ring`] for the matching deficits.

use core::f64;
use std::rc::Rc;
use std::sync::Arc;

use prism_animation::{Animation, Clock};
use prism_config::utils::MergeWith as _;
use prism_frame::ElementId;
use prism_ipc::WindowLayout;
use prism_renderer::{AlphaMode, RenderEl, SurfaceColorParams, SurfaceEl};
use smithay::utils::{Logical, Point, Rectangle, Size};

use super::focus_ring::FocusRing;
use super::opening_window::OpenAnimation;
use super::shadow::Shadow;
use super::SizingMode;
use super::{
    HitType, LayoutElement, LayoutElementRenderSnapshot, Options, SizeFrac, SnapshotTexture,
    RESIZE_ANIMATION_THRESHOLD,
};
use crate::utils::transaction::Transaction;
use crate::utils::{
    baba_is_float_offset, round_logical_in_physical, round_logical_in_physical_max1,
};

/// SDR-white nits used when projecting the fullscreen-backdrop sRGB
/// colour into the renderer's BT.2020-linear-nits space. Matches the
/// `DEFAULT_SDR_WHITE_NITS` in [`super::focus_ring`]; will be replaced
/// by a per-output value once HDR / SDR-white-level routing lands.
const BACKDROP_SDR_WHITE_NITS: f32 = 80.0;

/// Toplevel window with decorations.
#[derive(Debug)]
pub struct Tile<W: LayoutElement> {
    /// The toplevel window itself.
    window: W,

    /// The border around the window.
    border: FocusRing,

    /// The focus ring around the window.
    focus_ring: FocusRing,

    /// The shadow around the window.
    shadow: Shadow,

    /// Stable cross-frame id for the fullscreen black backdrop this tile emits
    /// while fullscreen. Allocated once so the damage tracker tracks it across
    /// frames (niri carries the id inside its cached `SolidColorBuffer`).
    backdrop_id: ElementId,

    /// This tile's current sizing mode.
    ///
    /// This will update only when the `window` actually goes maximized or fullscreen, rather than
    /// right away, to avoid black backdrop flicker before the window has had a chance to resize.
    sizing_mode: SizingMode,

    // niri carries a `SolidColorBuffer` for the fullscreen black
    // backdrop here; prism's render path will draw the backdrop
    // directly from the tile's fullscreen rect in `render()`, so no
    // cached buffer is needed.
    /// Whether the tile should float upon unfullscreening.
    pub(super) restore_to_floating: bool,

    /// The size that the window should assume when going floating.
    ///
    /// This is generally the last size the window had when it was floating. It can be unknown if
    /// the window starts out in the tiling layout or fullscreen.
    pub(super) floating_window_size: Option<Size<i32, Logical>>,

    /// The position that the tile should assume when going floating, relative to the floating
    /// space working area.
    ///
    /// This is generally the last position the tile had when it was floating. It can be unknown if
    /// the window starts out in the tiling layout.
    pub(super) floating_pos: Option<Point<f64, SizeFrac>>,

    /// Currently selected preset width index when this tile is floating.
    pub(super) floating_preset_width_idx: Option<usize>,

    /// Currently selected preset height index when this tile is floating.
    pub(super) floating_preset_height_idx: Option<usize>,

    /// The animation upon opening a window.
    open_animation: Option<OpenAnimation>,

    /// The animation of the window resizing.
    resize_animation: Option<ResizeAnimation>,

    /// The animation of a tile visually moving horizontally.
    move_x_animation: Option<MoveAnimation>,

    /// The animation of a tile visually moving vertically.
    move_y_animation: Option<MoveAnimation>,

    /// The animation of the tile's opacity.
    pub(super) alpha_animation: Option<AlphaAnimation>,

    /// Offset during the initial interactive move rubberband.
    pub(super) interactive_move_offset: Point<f64, Logical>,

    /// Snapshot of the last render for use in the close animation.
    /// Stubbed (the `Option`-of-`()` carries just the "have a snapshot"
    /// bit) until prism's snapshot pipeline lands. See
    /// [`super::LayoutElementRenderSnapshot`].
    unmap_snapshot: Option<TileRenderSnapshot>,

    // niri keeps a `RoundedCornerDamage` here for clipped-surface
    // corner-radius damage tracking — that lives in render_helpers and
    // is part of the GLES path we're not porting. Damage tracking for
    // corner-radius changes will be added with the Vulkan rounded-
    // corner element.
    /// The view size for the tile's workspace.
    ///
    /// Used as the fullscreen target size.
    view_size: Size<f64, Logical>,

    /// Scale of the output the tile is on (and rounds its sizes to).
    scale: f64,

    /// Clock for driving animations.
    pub(super) clock: Clock,

    /// Configurable properties of the layout.
    pub(super) options: Rc<Options>,
}

// niri defines a `TileRenderElement` enum (~10 GLES variants) here via
// the `niri_render_elements!` macro. Prism's render path emits a flat
// `Vec<RenderEl>` instead, so no enum is needed.

/// Snapshot type for the close / unmap animation. Niri stores a baked
/// texture buffer; prism stubs it to `()` (carrying just the "have a
/// snapshot" bit through the `Option`) until the offscreen-render
/// pipeline lands. See [`super::LayoutElementRenderSnapshot`].
pub type TileRenderSnapshot = LayoutElementRenderSnapshot;

#[derive(Debug)]
struct ResizeAnimation {
    anim: Animation,
    size_from: Size<f64, Logical>,
    /// Geometry-only snapshot stub (carries just the pre-resize window
    /// size). The actual crossfade pixels live in `gpu_snapshot`, captured
    /// from the persistent intermediate by the integrator on the first
    /// render after the resize commit (see [`Tile::resize_snapshot_geo`]).
    #[allow(dead_code)]
    snapshot: LayoutElementRenderSnapshot,
    /// The captured pre-resize tile frame, replayed at the *animated* tile
    /// rect and fading 1→0 on top of the live (already size-animated) content.
    /// `None` until the integrator fills it; the live content alone renders
    /// for that first frame. niri instead renders both states into an
    /// `OffscreenBuffer` and crossfades; prism replays the snapshot through
    /// the decode pass (same mechanism as the close animation).
    gpu_snapshot: Option<Arc<SnapshotTexture>>,
    /// Stable element id for the replay, so the damage tracker follows the
    /// crossfade across frames.
    snapshot_id: ElementId,
    tile_size_from: Size<f64, Logical>,
    // If the resize involved the fullscreen state at some point, this is the progress toward the
    // fullscreen state. Used for things like fullscreen backdrop alpha.
    //
    // Note that this can be set even if this specific resize is between two non-fullscreen states,
    // for example when issuing a new resize during an unfullscreen resize.
    fullscreen_progress: Option<Animation>,
    // Similar to above but for fullscreen-or-maximized.
    expanded_progress: Option<Animation>,
}

#[derive(Debug)]
struct MoveAnimation {
    anim: Animation,
    from: f64,
}

#[derive(Debug)]
pub(super) struct AlphaAnimation {
    pub(super) anim: Animation,
    /// Whether the animation should persist after it's done.
    ///
    /// This is used by things like interactive move which need to animate alpha to
    /// semitransparent, then hold it at semitransparent for a while, until the operation
    /// completes.
    pub(super) hold_after_done: bool,
    // niri keeps an `OffscreenBuffer` here so the alpha animation can
    // composite the whole tile in one pass; prism applies the alpha
    // value directly to per-element draws instead, so no buffer is
    // needed.
}

impl<W: LayoutElement> Tile<W> {
    pub fn new(
        window: W,
        view_size: Size<f64, Logical>,
        scale: f64,
        clock: Clock,
        options: Rc<Options>,
    ) -> Self {
        let rules = window.rules();
        let border_config = options.layout.border.merged_with(&rules.border);
        let focus_ring_config = options.layout.focus_ring.merged_with(&rules.focus_ring);
        let shadow_config = options.layout.shadow.merged_with(&rules.shadow);
        let sizing_mode = window.sizing_mode();

        Self {
            window,
            border: FocusRing::new(border_config.into()),
            focus_ring: FocusRing::new(focus_ring_config),
            shadow: Shadow::new(shadow_config),
            backdrop_id: ElementId::alloc(),
            sizing_mode,
            restore_to_floating: false,
            floating_window_size: None,
            floating_pos: None,
            floating_preset_width_idx: None,
            floating_preset_height_idx: None,
            open_animation: None,
            resize_animation: None,
            move_x_animation: None,
            move_y_animation: None,
            alpha_animation: None,
            interactive_move_offset: Point::from((0., 0.)),
            unmap_snapshot: None,
            view_size,
            scale,
            clock,
            options,
        }
    }

    pub fn update_config(
        &mut self,
        view_size: Size<f64, Logical>,
        scale: f64,
        options: Rc<Options>,
    ) {
        // If preset widths or heights changed, clear our stored preset index.
        if self.options.layout.preset_column_widths != options.layout.preset_column_widths {
            self.floating_preset_width_idx = None;
        }
        if self.options.layout.preset_window_heights != options.layout.preset_window_heights {
            self.floating_preset_height_idx = None;
        }

        self.view_size = view_size;
        self.scale = scale;
        self.options = options;

        let round_max1 = |logical| round_logical_in_physical_max1(self.scale, logical);

        let rules = self.window.rules();

        let mut border_config = self.options.layout.border.merged_with(&rules.border);
        border_config.width = round_max1(border_config.width);
        self.border.update_config(border_config.into());

        let mut focus_ring_config = self
            .options
            .layout
            .focus_ring
            .merged_with(&rules.focus_ring);
        focus_ring_config.width = round_max1(focus_ring_config.width);
        self.focus_ring.update_config(focus_ring_config);

        let shadow_config = self.options.layout.shadow.merged_with(&rules.shadow);
        self.shadow.update_config(shadow_config);

        self.window.update_config(self.options.blur);
    }

    /// Hook for shader-hot-reload — niri invalidates per-element shader
    /// caches here. Prism's pipelines are immutable for the lifetime of
    /// the renderer, so this is a no-op. The method stays so callers
    /// in the broader layout port don't need to change.
    pub fn update_shaders(&mut self) {}

    pub fn update_window(&mut self) {
        let prev_sizing_mode = self.sizing_mode;
        self.sizing_mode = self.window.sizing_mode();

        if let Some(animate_from) = self.window.take_animation_snapshot() {
            let params = if let Some(resize) = self.resize_animation.take() {
                // Compute like in animated_window_size(), but using the snapshot geometry (since
                // the current one is already overwritten).
                let mut size = animate_from.size;

                let val = resize.anim.value();
                let size_from = resize.size_from;
                let tile_size_from = resize.tile_size_from;

                size.w = size_from.w + (size.w - size_from.w) * val;
                size.h = size_from.h + (size.h - size_from.h) * val;

                let mut tile_size = animate_from.size;
                if prev_sizing_mode.is_fullscreen() {
                    tile_size.w = f64::max(tile_size.w, self.view_size.w);
                    tile_size.h = f64::max(tile_size.h, self.view_size.h);
                } else if prev_sizing_mode.is_normal() && !self.border.is_off() {
                    let width = self.border.width();
                    tile_size.w += width * 2.;
                    tile_size.h += width * 2.;
                }

                tile_size.w = tile_size_from.w + (tile_size.w - tile_size_from.w) * val;
                tile_size.h = tile_size_from.h + (tile_size.h - tile_size_from.h) * val;

                let fullscreen_from = resize
                    .fullscreen_progress
                    .map(|anim| anim.clamped_value().clamp(0., 1.))
                    .unwrap_or(if prev_sizing_mode.is_fullscreen() {
                        1.
                    } else {
                        0.
                    });

                let expanded_from = resize
                    .expanded_progress
                    .map(|anim| anim.clamped_value().clamp(0., 1.))
                    .unwrap_or(if prev_sizing_mode.is_normal() { 0. } else { 1. });

                // niri also reused the existing offscreen buffer here —
                // we don't track one (the crossfade is stubbed).
                (size, tile_size, fullscreen_from, expanded_from)
            } else {
                let size = animate_from.size;

                // Compute like in tile_size().
                let mut tile_size = size;
                if prev_sizing_mode.is_fullscreen() {
                    tile_size.w = f64::max(tile_size.w, self.view_size.w);
                    tile_size.h = f64::max(tile_size.h, self.view_size.h);
                } else if prev_sizing_mode.is_normal() && !self.border.is_off() {
                    let width = self.border.width();
                    tile_size.w += width * 2.;
                    tile_size.h += width * 2.;
                }

                let fullscreen_from = if prev_sizing_mode.is_fullscreen() {
                    1.
                } else {
                    0.
                };

                let expanded_from = if prev_sizing_mode.is_normal() { 0. } else { 1. };

                (size, tile_size, fullscreen_from, expanded_from)
            };
            let (size_from, tile_size_from, fullscreen_from, expanded_from) = params;

            let change = self.window.size().to_f64().to_point() - size_from.to_point();
            let change = f64::max(change.x.abs(), change.y.abs());
            let tile_change = self.tile_size().to_f64().to_point() - tile_size_from.to_point();
            let tile_change = f64::max(tile_change.x.abs(), tile_change.y.abs());
            let change = f64::max(change, tile_change);
            if change > RESIZE_ANIMATION_THRESHOLD {
                let anim = Animation::new(
                    self.clock.clone(),
                    0.,
                    1.,
                    0.,
                    self.options.animations.window_resize.anim,
                );

                let fullscreen_to = if self.sizing_mode.is_fullscreen() {
                    1.
                } else {
                    0.
                };
                let expanded_to = if self.sizing_mode.is_normal() { 0. } else { 1. };
                let fullscreen_progress = (fullscreen_from != fullscreen_to)
                    .then(|| anim.restarted(fullscreen_from, fullscreen_to, 0.));
                let expanded_progress = (expanded_from != expanded_to)
                    .then(|| anim.restarted(expanded_from, expanded_to, 0.));

                self.resize_animation = Some(ResizeAnimation {
                    anim,
                    size_from,
                    snapshot: animate_from,
                    gpu_snapshot: None,
                    snapshot_id: ElementId::alloc(),
                    tile_size_from,
                    fullscreen_progress,
                    expanded_progress,
                });
            } else {
                self.resize_animation = None;
            }
        }

        let round_max1 = |logical| round_logical_in_physical_max1(self.scale, logical);

        let rules = self.window.rules();
        let mut border_config = self.options.layout.border.merged_with(&rules.border);
        border_config.width = round_max1(border_config.width);
        self.border.update_config(border_config.into());

        let mut focus_ring_config = self
            .options
            .layout
            .focus_ring
            .merged_with(&rules.focus_ring);
        focus_ring_config.width = round_max1(focus_ring_config.width);
        self.focus_ring.update_config(focus_ring_config);

        let shadow_config = self.options.layout.shadow.merged_with(&rules.shadow);
        self.shadow.update_config(shadow_config);

        // niri propagates the resolved corner radius into the
        // `RoundedCornerDamage` so the clipped-surface element can
        // damage exactly the corner cells that changed. We don't have
        // that element yet, so the radius computation is dropped
        // entirely (no field reads it).
    }

    pub fn advance_animations(&mut self) {
        if let Some(open) = &mut self.open_animation {
            if open.is_done() {
                self.open_animation = None;
            }
        }

        if let Some(resize) = &mut self.resize_animation {
            if resize.anim.is_done() {
                self.resize_animation = None;
            }
        }

        if let Some(move_) = &mut self.move_x_animation {
            if move_.anim.is_done() {
                self.move_x_animation = None;
            }
        }
        if let Some(move_) = &mut self.move_y_animation {
            if move_.anim.is_done() {
                self.move_y_animation = None;
            }
        }

        if let Some(alpha) = &mut self.alpha_animation {
            if !alpha.hold_after_done && alpha.anim.is_done() {
                self.alpha_animation = None;
            }
        }
    }

    pub fn are_animations_ongoing(&self) -> bool {
        self.are_transitions_ongoing() || self.window.rules().baba_is_float == Some(true)
    }

    pub fn are_transitions_ongoing(&self) -> bool {
        self.open_animation.is_some()
            || self.resize_animation.is_some()
            || self.move_x_animation.is_some()
            || self.move_y_animation.is_some()
            || self
                .alpha_animation
                .as_ref()
                .is_some_and(|alpha| !alpha.anim.is_done())
    }

    pub fn update_render_elements(&mut self, is_active: bool, view_rect: Rectangle<f64, Logical>) {
        let rules = self.window.rules();
        let animated_tile_size = self.animated_tile_size();
        let expanded_progress = self.expanded_progress();

        let draw_border_with_background = rules
            .draw_border_with_background
            .unwrap_or_else(|| !self.window.has_ssd());
        let border_width = self.visual_border_width().unwrap_or(0.);

        // Do the inverse of tile_size() in order to handle the unfullscreen animation for windows
        // that were smaller than the fullscreen size, and therefore their animated_window_size() is
        // currently much smaller than the tile size.
        let mut border_window_size = animated_tile_size;
        border_window_size.w -= border_width * 2.;
        border_window_size.h -= border_width * 2.;

        // FIXME: this takes into account the animation from normal sizing mode to
        // maximized/fullscreen, but it doesn't take into account the corner radius animation from
        // the window itself.
        //
        // Currently, an easy way to see the problem is to start from a window with a nonzero
        // radius, then go from windowed fullscreen (that forces 0 radius) to regular fullscreen.
        // At the start of the animation, windowed fullscreen becomes false, but the window hasn't
        // animated to the normal fullscreen yet, so the radius here jumps to its nonzero value,
        // even though it should remain zero throughout.
        //
        // Later, when windows get the surface shape protocol with radii, this issue will happen
        // when that changes between animated commits.
        let radius = self
            .window
            .geometry_corner_radius()
            .expanded_by(border_width as f32)
            .scaled_by(1. - expanded_progress as f32);
        self.border.update_render_elements(
            border_window_size,
            is_active,
            !draw_border_with_background,
            self.window.is_urgent(),
            Rectangle::new(
                view_rect.loc - Point::from((border_width, border_width)),
                view_rect.size,
            ),
            radius,
            self.scale,
            1. - expanded_progress as f32,
        );

        let radius = if self.visual_border_width().is_some() {
            radius
        } else {
            self.window
                .geometry_corner_radius()
                .scaled_by(1. - expanded_progress as f32)
        };
        self.shadow.update_render_elements(
            animated_tile_size,
            is_active,
            radius,
            self.scale,
            1. - expanded_progress as f32,
        );

        let draw_focus_ring_with_background = if self.border.is_off() {
            draw_border_with_background
        } else {
            false
        };
        let radius = radius.expanded_by(self.focus_ring.width() as f32);
        self.focus_ring.update_render_elements(
            animated_tile_size,
            is_active,
            !draw_focus_ring_with_background,
            self.window.is_urgent(),
            view_rect,
            radius,
            self.scale,
            1. - expanded_progress as f32,
        );

        // niri sizes its cached `fullscreen_backdrop: SolidColorBuffer`
        // here; prism's render path emits the fullscreen-backdrop rect
        // directly from `animated_tile_size`, no cached buffer needed.
        let _ = animated_tile_size;
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn render_offset(&self) -> Point<f64, Logical> {
        let mut offset = Point::from((0., 0.));

        if let Some(move_) = &self.move_x_animation {
            offset.x += move_.from * move_.anim.value();
        }
        if let Some(move_) = &self.move_y_animation {
            offset.y += move_.from * move_.anim.value();
        }

        offset += self.interactive_move_offset;

        offset
    }

    pub fn start_open_animation(&mut self) {
        self.open_animation = Some(OpenAnimation::new(Animation::new(
            self.clock.clone(),
            0.,
            1.,
            0.,
            self.options.animations.window_open.anim,
        )));
    }

    pub fn resize_animation(&self) -> Option<&Animation> {
        self.resize_animation.as_ref().map(|resize| &resize.anim)
    }

    /// The output-logical rect to capture for the resize crossfade: the tile's
    /// pre-resize size (`tile_size_from`) placed at its current render
    /// `location`. Returns `None` when there's no resize animation or its
    /// snapshot is already captured — so the integrator both gates on and sizes
    /// the capture from this one call. Mirrors `ClosingWindow::needs_snapshot` +
    /// `geometry`.
    pub fn resize_snapshot_geo(
        &self,
        location: Point<f64, Logical>,
    ) -> Option<Rectangle<f64, Logical>> {
        let r = self.resize_animation.as_ref()?;
        if r.gpu_snapshot.is_some() {
            return None;
        }
        Some(Rectangle::new(location, r.tile_size_from))
    }

    /// Store the captured pre-resize frame. The replay is placed at the
    /// animated tile rect each frame (see `render`), so the captured geometry
    /// itself isn't retained.
    pub fn set_resize_snapshot(&mut self, snapshot: Arc<SnapshotTexture>) {
        if let Some(r) = &mut self.resize_animation {
            r.gpu_snapshot = Some(snapshot);
        }
    }

    /// The captured pre-resize crossfade frame, while one is replaying. For
    /// the integrator's frame keepalive set (`docs/async-render-rework.md`
    /// §2.4 — a frame in flight must outlive an animation ending under it).
    pub fn resize_snapshot(&self) -> Option<&Arc<SnapshotTexture>> {
        self.resize_animation.as_ref()?.gpu_snapshot.as_ref()
    }

    /// Drop the resize animation outright. Used when the tile's workspace is
    /// off screen at snapshot-capture time: there is no pre-resize frame in
    /// the intermediate to crossfade from (and leaving `gpu_snapshot` empty
    /// would re-request the capture next frame, copying unrelated pixels once
    /// the workspace scrolls in). The tile isn't being rendered, so snapping
    /// to the final size is invisible — and if the workspace scrolls back in
    /// mid-animation, the settled size is more correct than a stale
    /// mid-flight crossfade.
    pub fn cancel_resize_animation(&mut self) {
        self.resize_animation = None;
    }

    pub fn animate_move_from(&mut self, from: Point<f64, Logical>) {
        self.animate_move_x_from(from.x);
        self.animate_move_y_from(from.y);
    }

    pub fn animate_move_x_from(&mut self, from: f64) {
        self.animate_move_x_from_with_config(from, self.options.animations.window_movement.0);
    }

    pub fn animate_move_x_from_with_config(&mut self, from: f64, config: prism_config::Animation) {
        let current_offset = self.render_offset().x;

        // Preserve the previous config if ongoing.
        let anim = self.move_x_animation.take().map(|move_| move_.anim);
        let anim = anim
            .map(|anim| anim.restarted(1., 0., 0.))
            .unwrap_or_else(|| Animation::new(self.clock.clone(), 1., 0., 0., config));

        self.move_x_animation = Some(MoveAnimation {
            anim,
            from: from + current_offset,
        });
    }

    pub fn animate_move_y_from(&mut self, from: f64) {
        self.animate_move_y_from_with_config(from, self.options.animations.window_movement.0);
    }

    pub fn animate_move_y_from_with_config(&mut self, from: f64, config: prism_config::Animation) {
        let current_offset = self.render_offset().y;

        // Preserve the previous config if ongoing.
        let anim = self.move_y_animation.take().map(|move_| move_.anim);
        let anim = anim
            .map(|anim| anim.restarted(1., 0., 0.))
            .unwrap_or_else(|| Animation::new(self.clock.clone(), 1., 0., 0., config));

        self.move_y_animation = Some(MoveAnimation {
            anim,
            from: from + current_offset,
        });
    }

    pub fn offset_move_y_anim_current(&mut self, offset: f64) {
        if let Some(move_) = self.move_y_animation.as_mut() {
            // If the anim is almost done, there's little point trying to offset it; we can let
            // things jump. If it turns out like a bad idea, we could restart the anim instead.
            let value = move_.anim.value();
            if value > 0.001 {
                move_.from += offset / value;
            }
        }
    }

    pub fn stop_move_animations(&mut self) {
        self.move_x_animation = None;
        self.move_y_animation = None;
    }

    pub fn animate_alpha(&mut self, from: f64, to: f64, config: prism_config::Animation) {
        let from = from.clamp(0., 1.);
        let to = to.clamp(0., 1.);

        let current = self
            .alpha_animation
            .take()
            .map(|alpha| alpha.anim.clamped_value())
            .unwrap_or(from);

        self.alpha_animation = Some(AlphaAnimation {
            anim: Animation::new(self.clock.clone(), current, to, 0., config),
            hold_after_done: false,
        });
    }

    pub fn ensure_alpha_animates_to_1(&mut self) {
        if let Some(alpha) = &self.alpha_animation {
            if alpha.anim.to() != 1. {
                // Cancel animation instead of starting a new one because the user likely wants to
                // see the tile right away.
                self.alpha_animation = None;
            }
        }
    }

    pub fn hold_alpha_animation_after_done(&mut self) {
        if let Some(alpha) = &mut self.alpha_animation {
            alpha.hold_after_done = true;
        }
    }

    pub fn window(&self) -> &W {
        &self.window
    }

    pub fn window_mut(&mut self) -> &mut W {
        &mut self.window
    }

    pub fn sizing_mode(&self) -> SizingMode {
        self.sizing_mode
    }

    fn fullscreen_progress(&self) -> f64 {
        if let Some(resize) = &self.resize_animation {
            if let Some(anim) = &resize.fullscreen_progress {
                return anim.clamped_value().clamp(0., 1.);
            }
        }

        if self.sizing_mode.is_fullscreen() {
            1.
        } else {
            0.
        }
    }

    fn expanded_progress(&self) -> f64 {
        if let Some(resize) = &self.resize_animation {
            if let Some(anim) = &resize.expanded_progress {
                return anim.clamped_value().clamp(0., 1.);
            }
        }

        if self.sizing_mode.is_normal() {
            0.
        } else {
            1.
        }
    }

    /// Returns `None` if the border is hidden and `Some(width)` if it should be shown.
    pub fn effective_border_width(&self) -> Option<f64> {
        if !self.sizing_mode.is_normal() {
            return None;
        }

        if self.border.is_off() {
            return None;
        }

        Some(self.border.width())
    }

    fn visual_border_width(&self) -> Option<f64> {
        if self.border.is_off() {
            return None;
        }

        let expanded_progress = self.expanded_progress();

        // Only hide the border when fully expanded to avoid jarring border appearance.
        if expanded_progress == 1. {
            return None;
        }

        // FIXME: would be cool to, like, gradually resize the border from full width to 0 during
        // fullscreening, but the rest of the code isn't quite ready for that yet. It needs to
        // handle things like computing intermediate tile size when an animated resize starts during
        // an animated unfullscreen resize.
        Some(self.border.width())
    }

    /// Returns the location of the window's visual geometry within this Tile.
    pub fn window_loc(&self) -> Point<f64, Logical> {
        let mut loc = Point::from((0., 0.));

        let window_size = self.animated_window_size();
        let target_size = self.animated_tile_size();

        // Center the window within its tile.
        //
        // - Without borders, the sizes match, so this difference is zero.
        // - Borders always match from all sides, so this difference is pre-rounded to physical.
        // - In fullscreen, if the window is smaller than the tile, then it gets centered, otherwise
        //   the tile size matches the window.
        // - During animations, the window remains centered within the tile; this is important for
        //   the to/from fullscreen animation.
        loc.x += (target_size.w - window_size.w) / 2.;
        loc.y += (target_size.h - window_size.h) / 2.;

        // Round to physical pixels.
        loc = loc
            .to_physical_precise_round(self.scale)
            .to_logical(self.scale);

        loc
    }

    pub fn tile_size(&self) -> Size<f64, Logical> {
        let mut size = self.window_size();

        if self.sizing_mode.is_fullscreen() {
            // Normally we'd just return the fullscreen size here, but this makes things a bit
            // nicer if a fullscreen window is bigger than the fullscreen size for some reason.
            size.w = f64::max(size.w, self.view_size.w);
            size.h = f64::max(size.h, self.view_size.h);
            return size;
        }

        if let Some(width) = self.effective_border_width() {
            size.w += width * 2.;
            size.h += width * 2.;
        }

        size
    }

    pub fn tile_expected_or_current_size(&self) -> Size<f64, Logical> {
        let mut size = self.window_expected_or_current_size();

        if self.sizing_mode.is_fullscreen() {
            // Normally we'd just return the fullscreen size here, but this makes things a bit
            // nicer if a fullscreen window is bigger than the fullscreen size for some reason.
            size.w = f64::max(size.w, self.view_size.w);
            size.h = f64::max(size.h, self.view_size.h);
            return size;
        }

        if let Some(width) = self.effective_border_width() {
            size.w += width * 2.;
            size.h += width * 2.;
        }

        size
    }

    pub fn window_size(&self) -> Size<f64, Logical> {
        let mut size = self.window.size().to_f64();
        size = size
            .to_physical_precise_round(self.scale)
            .to_logical(self.scale);
        size
    }

    pub fn window_expected_or_current_size(&self) -> Size<f64, Logical> {
        let size = self.window.expected_size();
        let mut size = size.unwrap_or_else(|| self.window.size()).to_f64();
        size = size
            .to_physical_precise_round(self.scale)
            .to_logical(self.scale);
        size
    }

    pub fn animated_window_size(&self) -> Size<f64, Logical> {
        let mut size = self.window_size();

        if let Some(resize) = &self.resize_animation {
            let val = resize.anim.value();
            let size_from = resize.size_from.to_f64();

            size.w = f64::max(1., size_from.w + (size.w - size_from.w) * val);
            size.h = f64::max(1., size_from.h + (size.h - size_from.h) * val);
            size = size
                .to_physical_precise_round(self.scale)
                .to_logical(self.scale);
        }

        size
    }

    pub fn animated_tile_size(&self) -> Size<f64, Logical> {
        let mut size = self.tile_size();

        if let Some(resize) = &self.resize_animation {
            let val = resize.anim.value();
            let size_from = resize.tile_size_from.to_f64();

            size.w = f64::max(1., size_from.w + (size.w - size_from.w) * val);
            size.h = f64::max(1., size_from.h + (size.h - size_from.h) * val);
            size = size
                .to_physical_precise_round(self.scale)
                .to_logical(self.scale);
        }

        size
    }

    pub fn buf_loc(&self) -> Point<f64, Logical> {
        let mut loc = Point::from((0., 0.));
        loc += self.window_loc();
        loc += self.window.buf_loc().to_f64();
        loc
    }

    /// Returns a partially-filled [`WindowLayout`].
    ///
    /// Only the sizing properties that a [`Tile`] can fill are filled.
    pub fn ipc_layout_template(&self) -> WindowLayout {
        WindowLayout {
            pos_in_scrolling_layout: None,
            tile_size: self.tile_size().into(),
            window_size: self.window().size().into(),
            tile_pos_in_workspace_view: None,
            window_offset_in_tile: self.window_loc().into(),
        }
    }

    fn is_in_input_region(&self, mut point: Point<f64, Logical>) -> bool {
        point -= self.window_loc().to_f64();
        self.window.is_in_input_region(point)
    }

    fn is_in_activation_region(&self, point: Point<f64, Logical>) -> bool {
        let activation_region = Rectangle::from_size(self.tile_size());
        activation_region.contains(point)
    }

    pub fn hit(&self, point: Point<f64, Logical>) -> Option<HitType> {
        let offset = self.bob_offset();
        let point = point - offset;

        if self.is_in_input_region(point) {
            let win_pos = self.buf_loc() + offset;
            Some(HitType::Input { win_pos })
        } else if self.is_in_activation_region(point) {
            Some(HitType::Activate {
                is_tab_indicator: false,
            })
        } else {
            None
        }
    }

    pub fn request_tile_size(
        &mut self,
        mut size: Size<f64, Logical>,
        animate: bool,
        transaction: Option<Transaction>,
    ) {
        // Can't go through effective_border_width() because we might be fullscreen.
        if !self.border.is_off() {
            let width = self.border.width();
            size.w = f64::max(1., size.w - width * 2.);
            size.h = f64::max(1., size.h - width * 2.);
        }

        // The size request has to be i32 unfortunately, due to Wayland. We floor here instead of
        // round to avoid situations where proportionally-sized columns don't fit on the screen
        // exactly.
        self.window.request_size(
            size.to_i32_floor(),
            SizingMode::Normal,
            animate,
            transaction,
        );
    }

    pub fn tile_width_for_window_width(&self, size: f64) -> f64 {
        if self.border.is_off() {
            size
        } else {
            size + self.border.width() * 2.
        }
    }

    pub fn tile_height_for_window_height(&self, size: f64) -> f64 {
        if self.border.is_off() {
            size
        } else {
            size + self.border.width() * 2.
        }
    }

    pub fn window_width_for_tile_width(&self, size: f64) -> f64 {
        if self.border.is_off() {
            size
        } else {
            size - self.border.width() * 2.
        }
    }

    pub fn window_height_for_tile_height(&self, size: f64) -> f64 {
        if self.border.is_off() {
            size
        } else {
            size - self.border.width() * 2.
        }
    }

    pub fn request_maximized(
        &mut self,
        size: Size<f64, Logical>,
        animate: bool,
        transaction: Option<Transaction>,
    ) {
        self.window.request_size(
            size.to_i32_round(),
            SizingMode::Maximized,
            animate,
            transaction,
        );
    }

    pub fn request_fullscreen(&mut self, animate: bool, transaction: Option<Transaction>) {
        self.window.request_size(
            self.view_size.to_i32_round(),
            SizingMode::Fullscreen,
            animate,
            transaction,
        );
    }

    pub fn min_size_nonfullscreen(&self) -> Size<f64, Logical> {
        let mut size = self.window.min_size().to_f64();

        // Can't go through effective_border_width() because we might be fullscreen.
        if !self.border.is_off() {
            let width = self.border.width();

            size.w = f64::max(1., size.w);
            size.h = f64::max(1., size.h);

            size.w += width * 2.;
            size.h += width * 2.;
        }

        size
    }

    pub fn max_size_nonfullscreen(&self) -> Size<f64, Logical> {
        let mut size = self.window.max_size().to_f64();

        // Can't go through effective_border_width() because we might be fullscreen.
        if !self.border.is_off() {
            let width = self.border.width();

            if size.w > 0. {
                size.w += width * 2.;
            }
            if size.h > 0. {
                size.h += width * 2.;
            }
        }

        size
    }

    pub fn bob_offset(&self) -> Point<f64, Logical> {
        if self.window.rules().baba_is_float != Some(true) {
            return Point::from((0., 0.));
        }

        let y = baba_is_float_offset(self.clock.now(), self.view_size.h);
        let y = round_logical_in_physical(self.scale, y);
        Point::from((0., y))
    }

    /// Emit this tile's draw calls into `out` in output-space logical
    /// pixels. `location` is the tile's visual top-left in logical
    /// coordinates (this is **not** the window's top-left — the window sits
    /// inside the tile, inset by the border on all sides when one is drawn).
    ///
    /// Ported from niri/src/layout/tile.rs's `render_inner`, adapted to
    /// prism's element vocabulary: the resize crossfade replays a
    /// screen-copy snapshot instead of niri's offscreen re-render, the
    /// tile-wide fade is a per-element `mul_alpha` instead of offscreen
    /// alpha compositing, and clip-to-geometry uses the SDF surface
    /// clip. Still missing vs niri: background blur, xray.
    ///
    /// `focus_ring_visible` lets the caller suppress the focus ring
    /// (used by niri for tabs / overview); mirrors niri's parameter
    /// name in `render_inner`.
    pub fn render(
        &self,
        location: Point<f64, Logical>,
        scale: smithay::utils::Scale<f64>,
        focus_ring_visible: bool,
        ctx: &crate::layout::RenderCtx<'_>,
        out: &mut Vec<RenderEl>,
    ) {
        let fullscreen_progress = self.fullscreen_progress();
        let expanded_progress = self.expanded_progress();

        let tile_alpha = self
            .alpha_animation
            .as_ref()
            .map_or(1., |a| a.anim.clamped_value()) as f32;

        // Window-content opacity (the opacity window rule). `tile_alpha`
        // is NOT folded in here — it multiplies the whole assembled tile
        // (decorations included) at the bottom of this function, so an
        // interactive-move / tab fade dims shadow, ring and border in
        // lockstep with the content instead of leaving them opaque.
        let win_alpha = if self.window.is_ignoring_opacity_window_rule() {
            1.
        } else {
            let a = self.window.rules().opacity.unwrap_or(1.).clamp(0., 1.);
            // Interpolate towards alpha = 1. at fullscreen.
            let p = fullscreen_progress as f32;
            a * (1. - p) + p
        };

        // Baba-is-float vertical bob — kept here rather than baked into
        // render_offset() because it's not reset by interactive move /
        // animation cancellation; see niri's note on `bob_offset`.
        let bob = self.bob_offset();
        let location = location + bob;
        let window_loc = self.window_loc();
        let window_render_loc = location + window_loc;

        // prism-renderer draws in vector order — earlier pushes paint
        // behind. Build back-to-front: shadow, focus ring, fullscreen
        // backdrop, border, window content (popups emit on top from
        // inside `LayoutElement::render`).
        //
        // Collect into a local vec rather than straight into `out`: an active
        // open animation transforms the whole tile (zoom + fade about its
        // centre) as a post-pass over these elements before they're appended.
        let mut els: Vec<RenderEl> = Vec::new();

        if expanded_progress < 1. {
            self.shadow.render(location, &mut els);
        }

        // Hide focus ring when maximized/fullscreened — niri's logic.
        if focus_ring_visible && expanded_progress < 1. {
            self.focus_ring.render(location, &mut els);
        }

        // Fullscreen black backdrop. Niri caches a `SolidColorBuffer`;
        // we emit a `SolidColorEl` directly. Uses the animated tile
        // size so the backdrop fades in/out with the resize animation.
        if fullscreen_progress > 0. {
            let backdrop = Rectangle::new(location, self.animated_tile_size());
            let alpha = fullscreen_progress as f32;
            let color_bt2020_nits =
                prism_renderer::srgb_to_bt2020_nits(0., 0., 0., alpha, BACKDROP_SDR_WHITE_NITS);
            els.push(RenderEl::SolidColor(prism_renderer::SolidColorEl {
                id: self.backdrop_id,
                geometry: backdrop,
                color_bt2020_nits,
                clip: None,
            }));
        }

        // Border ring around the window itself.
        if self.visual_border_width().is_some() {
            self.border.render(window_render_loc, &mut els);
        }

        // Window content, then popups on top (matching the order the trait's
        // combined `render` would use). Split so the clip-to-geometry pass
        // below applies to the window's own surface tree but never to popups
        // — niri clips the same way in `render_inner`.
        let window_els_start = els.len();
        self.window
            .render_normal(window_render_loc, scale, win_alpha, ctx, &mut els);

        // Clip the window content to its geometry, with the window-rule
        // corner radius (`clip-to-geometry`). Mirrors niri: keep clipping
        // through the fullscreen *animation* (buggy clients submit
        // full-sized buffers before acking the state — Firefox), drop it
        // only at full fullscreen; scale the radius out with the expanded
        // animation; fit overlapping radii to the window size. Elements the
        // clip provably can't affect are left untouched
        // (`clip_to_rounded_box` no-ops, keeping their opaque regions).
        let clip_to_geometry =
            fullscreen_progress < 1. && self.window.rules().clip_to_geometry == Some(true);
        if clip_to_geometry {
            let window_size = self.window_size();
            let radius = self
                .window
                .geometry_corner_radius()
                .scaled_by(1. - expanded_progress as f32)
                .fit_to(window_size.w as f32, window_size.h as f32);
            let clip = prism_renderer::SurfaceClip {
                rect: Rectangle::new(window_render_loc, window_size),
                radii: [
                    radius.top_left,
                    radius.top_right,
                    radius.bottom_right,
                    radius.bottom_left,
                ],
            };
            for el in &mut els[window_els_start..] {
                el.clip_to_rounded_box(clip);
            }
        }

        // Resize crossfade. The live content above is already drawn at the
        // animated size (inc1: `animated_tile_size`/`animated_window_size`
        // interpolate, and border/ring/window_loc follow). Here we replay the
        // captured pre-resize frame on TOP, stretched to the same animated
        // tile rect, fading 1→0. Because the live content underneath is
        // opaque, src-over of `snapshot·(1-v)` gives a true crossfade
        // (`snap·(1-v) + live·v`) with no background bleed — so we fade only
        // the snapshot, not both layers. niri renders both states into an
        // offscreen buffer and crossfades; prism replays the snapshot through
        // the decode pass, the same mechanism as the close animation. The
        // snapshot is in the intermediate's working space, so it replays as a
        // pass-through draw (no EOTF, identity primaries, premultiplied).
        // Pushed before the popups below so an open popup stays on top of
        // the fading old frame instead of being covered by it.
        if let Some(resize) = &self.resize_animation {
            if let Some(snapshot) = &resize.gpu_snapshot {
                let progress = resize.anim.clamped_value().clamp(0.0, 1.0);
                let fade = (1.0 - progress) as f32;
                if fade > 0.0 {
                    // Draw the old frame at the current animated tile rect, so
                    // it tracks the live content's size as the animation runs.
                    let target = Rectangle::new(location, self.animated_tile_size());
                    // `opaque` empty: a fading snapshot never occludes.
                    els.push(RenderEl::Surface(SurfaceEl {
                        id: resize.snapshot_id,
                        texture_view: snapshot.view(),
                        chroma_view: None,
                        yuv: 0,
                        // Intermediate-space snapshot (passthrough transfer):
                        // never debanded, so the source extent is irrelevant.
                        source_extent: prism_renderer::vk::Extent2D::default(),
                        source_8bit: false,
                        geometry: target,
                        content_commit: 0,
                        opaque: Vec::new(),
                        src_rect_uv: [0.0, 0.0, 1.0, 1.0],
                        color: SurfaceColorParams::passthrough(),
                        alpha_mode: AlphaMode::Premultiplied,
                        alpha: fade,
                        clip: None,
                    }));
                }
            }
        }

        self.window
            .render_popups(window_render_loc, scale, win_alpha, ctx, &mut els);

        // Tile-wide fade (interactive move at 0.75, tabbed-column fades):
        // multiplied over every element — decorations, backdrop, content,
        // crossfade, popups — so the whole tile dims as one. Surfaces that
        // turn translucent drop their opaque regions inside mul_alpha, so
        // occlusion culling stays correct.
        if tile_alpha < 1. {
            for el in &mut els {
                el.mul_alpha(tile_alpha);
            }
        }

        // Open animation: zoom + fade the assembled tile about its centre. The
        // animated tile size matches what shadow/backdrop/border were laid out
        // against this frame, so the centre is the tile rect's midpoint.
        if let Some(open) = &self.open_animation {
            let size = self.animated_tile_size();
            let center = location + Point::from((size.w / 2., size.h / 2.));
            open.transform(center, &mut els);
        }
        out.extend(els);
    }

    /// Snapshot bookkeeping for the close animation. Niri renders the
    /// tile into an offscreen texture here; prism only retains the
    /// size, since the renderer-side crossfade isn't wired up. The
    /// rest of the unmap-animation state machine still runs.
    pub fn store_unmap_snapshot_if_empty(&mut self) {
        if self.unmap_snapshot.is_some() {
            return;
        }
        self.unmap_snapshot = Some(LayoutElementRenderSnapshot {
            size: self.animated_tile_size(),
        });
    }

    pub fn take_unmap_snapshot(&mut self) -> Option<TileRenderSnapshot> {
        self.unmap_snapshot.take()
    }

    pub fn border(&self) -> &FocusRing {
        &self.border
    }

    pub fn focus_ring(&self) -> &FocusRing {
        &self.focus_ring
    }

    pub fn options(&self) -> &Rc<Options> {
        &self.options
    }

    #[cfg(test)]
    pub fn view_size(&self) -> Size<f64, Logical> {
        self.view_size
    }

    #[cfg(test)]
    pub fn verify_invariants(&self) {
        use approx::assert_abs_diff_eq;

        assert_eq!(self.sizing_mode, self.window.sizing_mode());

        let scale = self.scale;
        let size = self.tile_size();
        let rounded = size.to_physical_precise_round(scale).to_logical(scale);
        assert_abs_diff_eq!(size.w, rounded.w, epsilon = 1e-5);
        assert_abs_diff_eq!(size.h, rounded.h, epsilon = 1e-5);
    }
}
