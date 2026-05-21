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

#[derive(Debug)]
pub struct TabIndicator {
    config: prism_config::TabIndicator,
    open_anim: Option<Animation>,
}

#[derive(Debug, Clone)]
pub struct TabInfo {
    /// Unique id of the tab (matches the column's window id).
    pub id: u64,
    pub is_active: bool,
    pub is_urgent: bool,
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

    pub fn update_render_elements(
        &mut self,
        _win_size: Size<f64, Logical>,
        _tabs: &[TabInfo],
        _scale: f64,
        _alpha: f32,
    ) {
    }

    pub fn render(
        &self,
        _location: Point<f64, Logical>,
        _project: &impl Fn(Rectangle<f64, Logical>) -> [f32; 4],
        _out: &mut Vec<RenderEl>,
    ) {
    }

    /// Niri uses this for click hit-test on the indicator strip.
    /// Returns the index of the hit tab, if any. Stub returns None.
    pub fn hit(
        &self,
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
