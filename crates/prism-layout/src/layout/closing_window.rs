//! Window-close animation — a shrink-and-fade of the window's last frame.
//!
//! When a window unmaps, its last composited frame still lives in the
//! persistent intermediate (BT.2020 absolute-nits, premultiplied). The
//! integrator copies that region into a [`SnapshotTexture`] the frame the
//! window is removed (before the decode pass repaints over it — see
//! `prism_renderer::SnapshotCopy`), and stores the `Arc` here via
//! [`ClosingWindow::set_snapshot`]. We then keep replaying the snapshot as a
//! pass-through decode draw (no re-decode, full HDR fidelity), shrinking
//! 1.0→0.8 about the tile centre and fading 1→0. niri's fallback effect.
//!
//! The snapshot `Arc` lives exactly as long as the `ClosingWindow`; when the
//! animation finishes the layout drops us and the texture frees itself.

use std::sync::Arc;

use prism_animation::Animation;
use prism_frame::ElementId;
use prism_renderer::{AlphaMode, RenderEl, SnapshotTexture, SurfaceColorParams, SurfaceEl};
use smithay::utils::{Logical, Point, Rectangle, Scale};

use crate::utils::transaction::TransactionBlocker;

/// Per-instance state — held in the scrolling / floating space's
/// `closing_windows` vector for the duration of the animation.
#[derive(Debug)]
pub struct ClosingWindow {
    /// Stable element id, so the damage tracker tracks the replay across frames.
    id: ElementId,
    /// Rect the window occupied (the snapshot's placement + size). In the
    /// scrolling space this is workspace-absolute (niri semantics) — the
    /// space subtracts its view position at capture and at replay; in the
    /// floating space positions are view-independent, so it is output-space
    /// as-is.
    geometry: Rectangle<f64, Logical>,
    /// The captured last frame. `None` until the integrator fills it on the
    /// first render after unmap (see [`Self::set_snapshot`]).
    snapshot: Option<Arc<SnapshotTexture>>,
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
    pub fn new(
        geometry: Rectangle<f64, Logical>,
        blocker: TransactionBlocker,
        anim: Animation,
    ) -> Self {
        Self {
            id: ElementId::alloc(),
            geometry,
            snapshot: None,
            state: AnimationState::new(blocker, anim),
        }
    }

    pub fn geometry(&self) -> Rectangle<f64, Logical> {
        self.geometry
    }

    /// Whether the GPU snapshot still needs capturing — the integrator calls
    /// [`Self::set_snapshot`] on the first render after unmap.
    pub fn needs_snapshot(&self) -> bool {
        self.snapshot.is_none()
    }

    pub fn set_snapshot(&mut self, snapshot: Arc<SnapshotTexture>) {
        self.snapshot = Some(snapshot);
    }

    pub fn advance_animations(&mut self) {
        // The clock drives `anim.value()`; nothing to advance by hand. The
        // blocker only gates the owning transaction, not the visual.
        let _ = &self.state.blocker;
    }

    pub fn are_animations_ongoing(&self) -> bool {
        !self.state.anim.is_done()
    }

    /// Emit the shrink+fade replay of the captured frame. `location` is the
    /// snapshot's top-left in output-logical pixels (the owning space passes
    /// `geometry().loc` corrected for its current view position); `_scale` is
    /// the output scale, unused here since the transform is expressed in
    /// logical space.
    pub fn render(
        &self,
        location: Point<f64, Logical>,
        _scale: Scale<f64>,
        out: &mut Vec<RenderEl>,
    ) {
        let Some(snapshot) = &self.snapshot else {
            // No capture yet (first frame after unmap, before the integrator
            // filled it) — draw nothing this one frame.
            return;
        };

        // niri's fallback close effect: scale 1.0→0.8, alpha 1→0.
        let progress = self.state.anim.clamped_value().clamp(0.0, 1.0);
        let anim_alpha = (1.0 - progress) as f32;
        let anim_scale = ((1.0 - progress) / 5.0 + 0.8).max(0.0);

        // The snapshot is already in the intermediate's working space, so it
        // replays through a pass-through decode (no EOTF, identity primaries,
        // premultiplied) — bit-identical to the window it captured.
        let geometry = Rectangle::new(location, self.geometry.size);
        let mut el = RenderEl::Surface(SurfaceEl {
            id: self.id,
            texture_view: snapshot.view(),
            chroma_view: None,
            yuv: 0,
            geometry,
            content_commit: 0,
            opaque: Vec::new(),
            src_rect_uv: [0.0, 0.0, 1.0, 1.0],
            color: SurfaceColorParams::passthrough(),
            alpha_mode: AlphaMode::Premultiplied,
            alpha: 1.0,
            clip: None,
        });

        let center =
            location + Point::from((self.geometry.size.w / 2.0, self.geometry.size.h / 2.0));
        el.scale_about(center, anim_scale);
        el.mul_alpha(anim_alpha);
        out.push(el);
    }
}
