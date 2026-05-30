//! Window-open animation — a zoom-and-fade on the live window.
//!
//! Niri renders the window through an offscreen GLES buffer each frame and
//! runs a custom open-effect shader over it. prism doesn't need the offscreen
//! step: the `decode` pass already composites the live window into the
//! persistent intermediate every frame, so the animation is a purely visual
//! transform applied to the tile's already-emitted render elements — scale
//! about the tile centre + an alpha fade. This is niri's *fallback* (no custom
//! shader) effect: scale 0.5→1.0 (overshoot tracks the spring), alpha 0→1.
//!
//! The layout geometry is untouched, so input hit-testing and tiling operate
//! at full size while only the pixels zoom in.

use prism_animation::Animation;
use prism_renderer::RenderEl;
use smithay::utils::{Logical, Point};

#[derive(Debug)]
pub struct OpenAnimation {
    anim: Animation,
}

impl OpenAnimation {
    pub fn new(anim: Animation) -> Self {
        Self { anim }
    }

    /// Done once the underlying clock-driven animation reaches its end — the
    /// tile drops the `OpenAnimation` on the next `advance_animations` tick.
    pub fn is_done(&self) -> bool {
        self.anim.is_done()
    }

    /// Apply the open zoom+fade to the tile's render elements, in place.
    /// `center` is the tile centre in output-logical pixels (the zoom origin).
    pub fn transform(&self, center: Point<f64, Logical>, els: &mut [RenderEl]) {
        // `value()` may overshoot [0, 1] on a spring; the zoom follows it so the
        // window springs slightly past full size. The fade uses the clamped
        // value so alpha never exceeds 1.0.
        let progress = self.anim.value();
        let alpha = self.anim.clamped_value().clamp(0.0, 1.0) as f32;
        let scale = (progress / 2.0 + 0.5).max(0.0);
        for el in els {
            el.scale_about(center, scale);
            el.mul_alpha(alpha);
        }
    }
}
