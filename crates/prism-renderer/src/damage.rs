//! Per-output region damage tracking.
//!
//! Mirrors smithay's `OutputDamageTracker` last-state diff, adapted to prism's
//! plain-data [`FrameElementMeta`] stream: per element, detect
//! move/resize/restack (damage old ∪ new geometry) or content change (damage
//! geometry), and damage the old geometry of elements that disappeared.
//! Operates in physical pixels.
//!
//! Current scope (Stage 1b): whole-element granularity — a changed surface
//! damages its whole rect (no sub-rect `damage_since` yet), and no occlusion
//! culling. The result is only logged for now; Stage 2 scissors the render to
//! it. The diff is intentionally conservative — it never under-damages.

use std::collections::HashMap;

use prism_frame::{ElementId, Physical, Rectangle, Scale};

use crate::element::FrameElementMeta;

#[derive(Clone, Copy)]
struct Snapshot {
    /// Element geometry in physical pixels last frame.
    geometry: Rectangle<i32, Physical>,
    /// Stacking index last frame (position in the back-to-front list).
    z: usize,
    /// Content fingerprint last frame.
    content: u64,
}

/// Per-output damage tracker. Holds the previous frame's per-element state and
/// diffs the current frame against it.
#[derive(Default)]
pub struct DamageTracker {
    last: HashMap<ElementId, Snapshot>,
}

impl DamageTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Diff `meta` (this frame's back-to-front elements, in logical geometry)
    /// against the previous frame and return the damaged regions in physical
    /// pixels. `scale` converts logical → physical (the output has no rotation,
    /// so this is just per-axis scale + round).
    ///
    /// Advances the stored state immediately. NOTE: once the render is actually
    /// scissored to this result (Stage 2), the advance must move to *after* a
    /// successful present — otherwise a dropped or failed page-flip would lose
    /// the damage and leave stale pixels next frame.
    pub fn compute(
        &mut self,
        meta: &[FrameElementMeta],
        scale: Scale<f64>,
    ) -> Vec<Rectangle<i32, Physical>> {
        let mut damage: Vec<Rectangle<i32, Physical>> = Vec::new();
        let mut current: HashMap<ElementId, Snapshot> = HashMap::with_capacity(meta.len());

        for (z, m) in meta.iter().enumerate() {
            let geometry: Rectangle<i32, Physical> = m.geometry.to_physical_precise_round(scale);
            match self.last.get(&m.id) {
                // New element this frame.
                None => damage.push(geometry),
                Some(old) => {
                    if old.geometry != geometry || old.z != z {
                        // Moved / resized / restacked: damage both footprints.
                        damage.push(geometry);
                        damage.push(old.geometry);
                    } else if old.content != m.content_token {
                        // Same placement, changed pixels.
                        damage.push(geometry);
                    }
                    // else: identical — contributes no damage.
                }
            }
            current.insert(
                m.id,
                Snapshot {
                    geometry,
                    z,
                    content: m.content_token,
                },
            );
        }

        // Elements present last frame but gone now: their old footprint needs
        // repainting (whatever was behind them now shows through).
        for (id, old) in &self.last {
            if !current.contains_key(id) {
                damage.push(old.geometry);
            }
        }

        self.last = current;
        damage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_frame::{Point, Size};
    use std::num::NonZeroU64;

    fn id(n: u64) -> ElementId {
        ElementId::from_raw(NonZeroU64::new(n).unwrap())
    }

    fn meta(n: u64, x: f64, y: f64, w: f64, h: f64, content: u64) -> FrameElementMeta {
        FrameElementMeta {
            id: id(n),
            geometry: Rectangle::new(Point::from((x, y)), Size::from((w, h))),
            content_token: content,
        }
    }

    fn rect(x: i32, y: i32, w: i32, h: i32) -> Rectangle<i32, Physical> {
        Rectangle::new(Point::from((x, y)), Size::from((w, h)))
    }

    fn scale1() -> Scale<f64> {
        Scale::from(1.0)
    }

    #[test]
    fn first_frame_damages_every_element() {
        let mut t = DamageTracker::new();
        let d = t.compute(
            &[meta(1, 0., 0., 100., 100., 0), meta(2, 10., 10., 5., 5., 0)],
            scale1(),
        );
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn unchanged_frame_has_no_damage() {
        let mut t = DamageTracker::new();
        let frame = [meta(1, 0., 0., 100., 100., 7)];
        t.compute(&frame, scale1());
        assert!(t.compute(&frame, scale1()).is_empty());
    }

    #[test]
    fn content_change_damages_geometry() {
        let mut t = DamageTracker::new();
        t.compute(&[meta(1, 0., 0., 100., 100., 1)], scale1());
        let d = t.compute(&[meta(1, 0., 0., 100., 100., 2)], scale1());
        assert_eq!(d, vec![rect(0, 0, 100, 100)]);
    }

    #[test]
    fn move_damages_old_and_new() {
        let mut t = DamageTracker::new();
        t.compute(&[meta(1, 0., 0., 100., 100., 0)], scale1());
        let d = t.compute(&[meta(1, 50., 0., 100., 100., 0)], scale1());
        assert_eq!(d.len(), 2);
        assert!(d.contains(&rect(50, 0, 100, 100)));
        assert!(d.contains(&rect(0, 0, 100, 100)));
    }

    #[test]
    fn vanished_element_damages_its_old_footprint() {
        let mut t = DamageTracker::new();
        t.compute(
            &[
                meta(1, 0., 0., 100., 100., 0),
                meta(2, 200., 0., 50., 50., 0),
            ],
            scale1(),
        );
        let d = t.compute(&[meta(1, 0., 0., 100., 100., 0)], scale1());
        assert_eq!(d, vec![rect(200, 0, 50, 50)]);
    }

    #[test]
    fn restack_without_move_still_damages() {
        let mut t = DamageTracker::new();
        let a = meta(1, 0., 0., 10., 10., 0);
        let b = meta(2, 0., 0., 10., 10., 0);
        t.compute(&[a.clone(), b.clone()], scale1());
        // Swap z-order; geometry and content unchanged.
        assert!(!t.compute(&[b, a], scale1()).is_empty());
    }

    #[test]
    fn scale_converts_logical_to_physical() {
        let mut t = DamageTracker::new();
        let d = t.compute(&[meta(1, 1., 2., 10., 20., 0)], Scale::from(2.0));
        assert_eq!(d, vec![rect(2, 4, 20, 40)]);
    }
}
