//! Window-open animation — stubbed for now.
//!
//! Niri implements this with an offscreen GLES render of the window's
//! first frame plus a custom WGSL-ish open-effect shader. The
//! `prism-renderer` doesn't have offscreen render-to-texture or custom
//! shader pipelines yet, so this stub treats the animation as
//! "instantly done": newly mapped windows just appear.
//!
//! API preserved so `tile.rs` can construct + query `OpenAnimation`
//! state without touching the render path.

use prism_animation::Animation;
use prism_renderer::RenderEl;
use smithay::utils::{Logical, Point, Scale, Size};

#[derive(Debug)]
pub struct OpenAnimation {
    anim: Animation,
}

impl OpenAnimation {
    pub fn new(anim: Animation) -> Self {
        Self { anim }
    }

    /// Always returns true today — the stub has no visual to wait on so
    /// we report instantly complete after construction. Future port
    /// drives this off `self.anim.is_done()` once the effect shader and
    /// offscreen pipeline are wired.
    pub fn is_done(&self) -> bool {
        // FIXME: once a real open effect is wired, use:
        //   self.anim.is_done()
        let _ = &self.anim;
        true
    }

    pub fn render(
        &self,
        _geo_size: Size<f64, Logical>,
        _location: Point<f64, Logical>,
        _scale: Scale<f64>,
        _alpha: f32,
        _out: &mut Vec<RenderEl>,
    ) {
    }
}
