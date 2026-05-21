//! Drag-and-drop insert hint (the highlighted slot during interactive
//! window placement) — stubbed for now.
//!
//! Niri draws this with a custom GLES border shader; we don't have a
//! Vulkan equivalent yet. Drag-drop placement still works, just without
//! the visual hint.

use prism_renderer::RenderEl;
use smithay::utils::{Logical, Point, Rectangle, Size};

#[derive(Debug)]
pub struct InsertHintElement {
    config: prism_config::InsertHint,
}

impl InsertHintElement {
    pub fn new(config: prism_config::InsertHint) -> Self {
        Self { config }
    }

    pub fn update_config(&mut self, config: prism_config::InsertHint) {
        self.config = config;
    }

    pub fn update_shaders(&mut self) {}

    pub fn config(&self) -> prism_config::InsertHint {
        self.config
    }

    pub fn update_render_elements(
        &mut self,
        _area: Rectangle<f64, Logical>,
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

    pub fn extra_size(&self) -> Size<f64, Logical> {
        Size::default()
    }
}
