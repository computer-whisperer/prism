//! Window-close animation — stubbed for now.
//!
//! Niri snapshots the closing window's last rendered frame (offscreen
//! GLES texture), then plays a custom shader-driven dismissal effect
//! against it. Same blockers as `opening_window.rs`: we don't have
//! offscreen render or custom shader pipelines on prism-renderer yet.
//! Windows just disappear when closed.
//!
//! The transaction blocker plumbing is preserved so the layout port
//! still routes close commits through the transaction system; the
//! visual side is just a no-op.

use prism_animation::Animation;
use prism_renderer::RenderEl;
use smithay::utils::{Logical, Point, Rectangle, Scale};

use crate::utils::transaction::TransactionBlocker;

/// Per-instance state — held in `tile.rs`'s `closing_animations` map.
#[derive(Debug)]
pub struct ClosingWindow {
    geometry: Rectangle<f64, Logical>,
    state: AnimationState,
}

#[derive(Debug)]
pub struct AnimationState {
    pub blocker: TransactionBlocker,
    pub anim: Animation,
}

impl AnimationState {
    pub fn new(blocker: TransactionBlocker, anim: Animation) -> Self {
        Self { blocker, anim }
    }
}

impl ClosingWindow {
    /// Construct without snapshotting anything. Real impl will capture
    /// the window's last frame; the stub records geometry only so the
    /// layout still tracks where the window was.
    pub fn new(
        geometry: Rectangle<f64, Logical>,
        blocker: TransactionBlocker,
        anim: Animation,
    ) -> Self {
        Self {
            geometry,
            state: AnimationState::new(blocker, anim),
        }
    }

    pub fn geometry(&self) -> Rectangle<f64, Logical> {
        self.geometry
    }

    pub fn advance_animations(&mut self) {
        // The blocker keeps the transaction alive until the anim ticks
        // forward; without a real animation effect the stub treats one
        // tick as enough and lets the transaction proceed.
        let _ = &self.state;
    }

    pub fn are_animations_ongoing(&self) -> bool {
        // Always false so the closing animation completes immediately —
        // the caller will drop us on the next tick.
        false
    }

    pub fn render(
        &self,
        _location: Point<f64, Logical>,
        _scale: Scale<f64>,
        _alpha: f32,
        _out: &mut Vec<RenderEl>,
    ) {
    }
}
