//! Tab indicator for tabbed columns — stubbed for now.
//!
//! Niri draws a small vertical strip with one entry per tab in a column,
//! using `render_helpers::border::BorderRenderElement` (a custom GLES
//! shader) for the per-tab pill shape. Tabbed columns aren't a tier-1
//! daily-driver feature, so we stub the whole thing until the Vulkan
//! border shader lands; tabbed columns will just have no visual
//! indicator until then (the layout still tracks the active tab).
//!
//! API preserved so `tile.rs` compiles.

use prism_animation::{Animation, Clock};
use prism_renderer::RenderEl;
use smithay::utils::{Logical, Point, Rectangle, Size};

use super::tile::Tile;
use super::LayoutElement;

#[derive(Debug)]
pub struct TabIndicator {
    config: prism_config::TabIndicator,
    open_anim: Option<Animation>,
}

/// Per-tab metadata the indicator renders from. Niri's full version
/// resolves per-tab gradient colors here (active/inactive/urgent
/// branches against window rules + config defaults); the prism stub
/// just carries the source bits and leaves gradient resolution to a
/// future Vulkan border-shader port.
#[derive(Debug, Clone)]
pub struct TabInfo {
    /// Unique id of the tab (matches the column's window id, as a
    /// `u64` here so the stub doesn't have to be generic over `W::Id`).
    pub id: u64,
    pub is_active: bool,
    pub is_urgent: bool,
}

impl TabInfo {
    /// Build a `TabInfo` from a tile. Niri's version also resolves
    /// the gradient color from window rules + config; until the
    /// gradient renderer lands we just capture the discrete bits.
    pub fn from_tile<W: LayoutElement>(
        tile: &Tile<W>,
        _position: Point<f64, Logical>,
        is_active: bool,
        is_urgent: bool,
        _config: &prism_config::TabIndicator,
    ) -> Self {
        // `W::Id` may not be convertible to u64 in general; stub-id
        // until the gradient pipeline is wired. Once that lands the
        // ID will move into a separate per-tab cache keyed by
        // `W::Id` directly.
        let _ = tile;
        Self {
            id: 0,
            is_active,
            is_urgent,
        }
    }
}

impl TabIndicator {
    pub fn new(config: prism_config::TabIndicator) -> Self {
        Self {
            config,
            open_anim: None,
        }
    }

    pub fn update_config(&mut self, config: prism_config::TabIndicator) {
        self.config = config;
    }

    pub fn update_shaders(&mut self) {}

    pub fn config(&self) -> prism_config::TabIndicator {
        self.config
    }

    pub fn advance_animations(&mut self) {
        if let Some(anim) = &self.open_anim {
            if anim.is_done() {
                self.open_anim = None;
            }
        }
    }

    pub fn are_animations_ongoing(&self) -> bool {
        self.open_anim.is_some()
    }

    pub fn start_open_animation(&mut self, clock: Clock, config: prism_config::Animation) {
        self.open_anim = Some(Animation::new(clock, 0., 1., 0., config));
    }

    /// Cache per-tab render geometry. Matches niri's 7-arg
    /// signature so the upstream call site in `scrolling.rs` ports
    /// over unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn update_render_elements<I: IntoIterator<Item = TabInfo>>(
        &mut self,
        _enabled: bool,
        _area: Rectangle<f64, Logical>,
        _area_view_rect: Rectangle<f64, Logical>,
        _tab_count: usize,
        _tabs: I,
        _is_active: bool,
        _scale: f64,
    ) {
    }

    pub fn render(&self, _location: Point<f64, Logical>, _out: &mut Vec<RenderEl>) {}

    /// Niri uses this for click hit-test on the indicator strip.
    /// Returns the index of the hit tab, if any. Stub returns None.
    pub fn hit(
        &self,
        _area: Rectangle<f64, Logical>,
        _tab_count: usize,
        _scale: f64,
        _point: Point<f64, Logical>,
    ) -> Option<usize> {
        None
    }

    /// Extra logical pixels the indicator adds to the column footprint.
    /// Stub returns zero so the layout treats it as not-there.
    pub fn extra_size(&self, _tab_count: usize, _scale: f64) -> Size<f64, Logical> {
        Size::default()
    }

    /// Offset applied to the column's content when the indicator is
    /// present. Stub returns zero.
    pub fn content_offset(&self, _tab_count: usize, _scale: f64) -> Point<f64, Logical> {
        Point::default()
    }
}
