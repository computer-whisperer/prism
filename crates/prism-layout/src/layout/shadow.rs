//! Drop-shadow rendering — stubbed for now.
//!
//! Niri renders window shadows through a custom GLES shadow shader
//! (`render_helpers::shadow::ShadowRenderElement`) that draws a fading,
//! per-corner-radius shadow rect via SDF math. We don't have a Vulkan
//! equivalent yet, and shadows are pure polish (a daily driver works
//! fine without them), so this stub keeps the API surface
//! `tile.rs`/`workspace.rs`/`layer/mapped.rs` expect but emits nothing.
//!
//! When we want shadows: write `prism-renderer::ShadowEl` + a small SDF
//! fragment shader that produces an alpha-blended drop shadow, then make
//! `update_render_elements` cache parameters and `render` emit the
//! ShadowEl through the supplied output adapter. The `Shadow` API will
//! stay the same; only the body changes.

use prism_renderer::RenderEl;
use smithay::utils::{Logical, Point, Size};

#[derive(Debug)]
pub struct Shadow {
    config: prism_config::Shadow,
}

impl Shadow {
    pub fn new(config: prism_config::Shadow) -> Self {
        Self { config }
    }

    pub fn update_config(&mut self, config: prism_config::Shadow) {
        self.config = config;
    }

    pub fn update_shaders(&mut self) {}

    pub fn config(&self) -> &prism_config::Shadow {
        &self.config
    }

    /// Niri caches per-frame shadow geometry here. We accept the call so
    /// callers compile, but produce no draws.
    pub fn update_render_elements(
        &mut self,
        _win_size: Size<f64, Logical>,
        _is_active: bool,
        _radius: prism_config::CornerRadius,
        _scale: f64,
        _alpha: f32,
    ) {
    }

    /// No shadow draws today.
    pub fn render(&self, _location: Point<f64, Logical>, _out: &mut Vec<RenderEl>) {}
}
