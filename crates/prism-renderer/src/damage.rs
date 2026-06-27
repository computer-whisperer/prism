//! Per-output region damage tracking.
//!
//! Mirrors smithay's `OutputDamageTracker` last-state diff, adapted to prism's
//! plain-data [`FrameElementMeta`] stream: per element, detect
//! move/resize/restack (damage old ∪ new geometry) or content change (damage
//! geometry), and damage the old geometry of elements that disappeared.
//! Operates in physical pixels.
//!
//! Granularity is whole-element on the *input* side — a changed surface damages
//! its whole rect (no sub-rect `damage_since` from `wl_surface` buffer damage
//! yet). On the *output* side the result drives the decode/encode scissors.
//!
//! Occlusion: the diff subtracts the opaque regions of elements stacked in front
//! from each lower element's current-position damage (smithay's
//! `OutputDamageTracker` pattern), so an animating full-screen surface covered by
//! opaque windows damages only the visible remainder. Old-position damage (moved/
//! vanished footprints) is emitted in full. The diff is intentionally
//! conservative — opaque occluders round inward, so it never under-damages.

use std::collections::HashMap;

use prism_frame::{ElementId, Logical, Physical, Point, Rectangle, Scale, Size};

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
    /// State staged by the most recent [`compute`](Self::compute), promoted to
    /// `last` by [`commit`](Self::commit) once the frame is known to have been
    /// presented. Kept separate so a dropped or failed flip re-damages next
    /// frame instead of losing the damage.
    pending: Option<HashMap<ElementId, Snapshot>>,
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
    /// Stages the new state in `pending` but does *not* advance `last` — the
    /// caller must call [`commit`](Self::commit) after a successful present.
    /// This keeps the diff baselined on the last frame that actually reached
    /// the screen, so a skipped or failed flip re-damages rather than dropping.
    pub fn compute(
        &mut self,
        meta: &[FrameElementMeta],
        scale: Scale<f64>,
    ) -> Vec<Rectangle<i32, Physical>> {
        let n = meta.len();
        let mut damage: Vec<Rectangle<i32, Physical>> = Vec::new();
        let mut current: HashMap<ElementId, Snapshot> = HashMap::with_capacity(n);

        let geo: Vec<Rectangle<i32, Physical>> = meta
            .iter()
            .map(|m| m.geometry.to_physical_precise_round(scale))
            .collect();

        // Occlusion: walk front-to-back, accumulating the opaque regions of
        // elements drawn in front. Each element's *current-position* damage (new,
        // content-changed, or moved-to) is clipped by that accumulation — a region
        // hidden behind opaque content in front needs no repaint. This is what
        // collapses an animating full-screen wallpaper, covered by opaque windows,
        // down to the visible border.
        //
        // *Old-position* damage (a moved element's vacated footprint, a vanished
        // element) is emitted in full and NOT occlusion-clipped: the persistent
        // intermediate still holds the moved element's pixels there, so that area
        // must be recomposited regardless of what opaque content sits in front of
        // it now.
        //
        // `meta` is back-to-front, so a higher index is further front. Iterating
        // in reverse makes `occ` hold exactly the opaque regions of the elements
        // in front of the current one — its own opaque region is added only after
        // its damage is clipped.
        let mut occ: Vec<Rectangle<i32, Physical>> = Vec::new();
        for i in (0..n).rev() {
            let m = &meta[i];
            let g = geo[i];

            // Current-position damage (subject to occlusion clipping below).
            let mut cur: Vec<Rectangle<i32, Physical>> = Vec::new();
            match self.last.get(&m.id) {
                None => cur.push(g), // new element
                Some(old) => {
                    if old.geometry != g || old.z != i {
                        // Moved / resized / restacked: the new footprint is
                        // occludable; the old footprint is emitted in full.
                        cur.push(g);
                        damage.push(old.geometry);
                    } else if old.content != m.content_token {
                        // Same placement, changed pixels.
                        cur.push(g);
                    }
                    // else: identical — contributes no damage.
                }
            }

            if occ.is_empty() {
                damage.append(&mut cur);
            } else {
                for r in cur {
                    damage.extend(r.subtract_rects(occ.iter().copied()));
                }
            }

            // This element now occludes everything behind it (lower index).
            for r in &m.opaque {
                if let Some(p) = opaque_to_physical(*r, scale) {
                    occ.push(p);
                }
            }

            current.insert(
                m.id,
                Snapshot {
                    geometry: g,
                    z: i,
                    content: m.content_token,
                },
            );
        }

        // Elements present last frame but gone now: their old footprint needs
        // repainting (whatever was behind them now shows through). Emitted in
        // full — see the old-position note above.
        for (id, old) in &self.last {
            if !current.contains_key(id) {
                damage.push(old.geometry);
            }
        }

        self.pending = Some(current);
        damage
    }

    /// Promote the most recent [`compute`](Self::compute)'s state to current.
    /// Call only after the frame was actually presented (a successful flip); on
    /// a skipped or failed present, don't call it so the damage is recomputed
    /// against the same baseline next frame.
    pub fn commit(&mut self) {
        if let Some(pending) = self.pending.take() {
            self.last = pending;
        }
    }
}

/// Convert a logical opaque rect to physical pixels, rounding *inward* (shrink).
/// Conservative by design: an occluder rounded outward could subtract a
/// partially-covered edge pixel from a lower element's damage and leave it
/// stale, so we only ever treat fully-covered pixels as opaque. `None` if it
/// shrinks to nothing.
fn opaque_to_physical(
    r: Rectangle<f64, Logical>,
    scale: Scale<f64>,
) -> Option<Rectangle<i32, Physical>> {
    let x0 = (r.loc.x * scale.x).ceil();
    let y0 = (r.loc.y * scale.y).ceil();
    let x1 = ((r.loc.x + r.size.w) * scale.x).floor();
    let y1 = ((r.loc.y + r.size.h) * scale.y).floor();
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(Rectangle::new(
        Point::from((x0 as i32, y0 as i32)),
        Size::from(((x1 - x0) as i32, (y1 - y0) as i32)),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU64;

    fn id(n: u64) -> ElementId {
        ElementId::from_raw(NonZeroU64::new(n).unwrap())
    }

    fn meta(n: u64, x: f64, y: f64, w: f64, h: f64, content: u64) -> FrameElementMeta {
        FrameElementMeta {
            id: id(n),
            geometry: Rectangle::new(Point::from((x, y)), Size::from((w, h))),
            content_token: content,
            // Transparent by default: occludes nothing.
            opaque: Vec::new(),
        }
    }

    /// Like [`meta`] but fully opaque over its whole geometry (an opaque window).
    fn meta_opaque(n: u64, x: f64, y: f64, w: f64, h: f64, content: u64) -> FrameElementMeta {
        let mut m = meta(n, x, y, w, h, content);
        m.opaque = vec![m.geometry];
        m
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
        t.commit();
        assert!(t.compute(&frame, scale1()).is_empty());
    }

    #[test]
    fn content_change_damages_geometry() {
        let mut t = DamageTracker::new();
        t.compute(&[meta(1, 0., 0., 100., 100., 1)], scale1());
        t.commit();
        let d = t.compute(&[meta(1, 0., 0., 100., 100., 2)], scale1());
        assert_eq!(d, vec![rect(0, 0, 100, 100)]);
    }

    #[test]
    fn move_damages_old_and_new() {
        let mut t = DamageTracker::new();
        t.compute(&[meta(1, 0., 0., 100., 100., 0)], scale1());
        t.commit();
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
        t.commit();
        let d = t.compute(&[meta(1, 0., 0., 100., 100., 0)], scale1());
        assert_eq!(d, vec![rect(200, 0, 50, 50)]);
    }

    #[test]
    fn restack_without_move_still_damages() {
        let mut t = DamageTracker::new();
        let a = meta(1, 0., 0., 10., 10., 0);
        let b = meta(2, 0., 0., 10., 10., 0);
        t.compute(&[a.clone(), b.clone()], scale1());
        t.commit();
        // Swap z-order; geometry and content unchanged.
        assert!(!t.compute(&[b, a], scale1()).is_empty());
    }

    #[test]
    fn uncommitted_frame_redamages_until_committed() {
        let mut t = DamageTracker::new();
        let frame = [meta(1, 0., 0., 100., 100., 7)];
        assert_eq!(t.compute(&frame, scale1()).len(), 1); // new element
                                                          // No commit → baseline unchanged → still damaged next compute.
        assert_eq!(t.compute(&frame, scale1()).len(), 1);
        t.commit();
        assert!(t.compute(&frame, scale1()).is_empty()); // baseline advanced
    }

    #[test]
    fn scale_converts_logical_to_physical() {
        let mut t = DamageTracker::new();
        let d = t.compute(&[meta(1, 1., 2., 10., 20., 0)], Scale::from(2.0));
        assert_eq!(d, vec![rect(2, 4, 20, 40)]);
    }

    // ── Occlusion (layer 3a) ────────────────────────────────────────────────

    #[test]
    fn opaque_front_clips_animating_wallpaper_to_border() {
        // Full-screen wallpaper (back) animating every frame, with a centered
        // opaque window (front). The wallpaper's content-change damage should be
        // clipped to the visible border ring — never the window region.
        let mut t = DamageTracker::new();
        let wallpaper = |tok| meta(1, 0., 0., 100., 100., tok);
        let window = meta_opaque(2, 25., 25., 50., 50., 7);
        t.compute(&[wallpaper(0), window.clone()], scale1());
        t.commit();
        // Next frame: wallpaper pixels changed, window unchanged.
        let d = t.compute(&[wallpaper(1), window], scale1());
        let expected = rect(0, 0, 100, 100).subtract_rects(std::iter::once(rect(25, 25, 50, 50)));
        assert_eq!(d, expected);
        // The occluded center is not in the damage.
        assert!(!d.iter().any(|r| r.contains(Point::from((50, 50)))));
    }

    #[test]
    fn opaque_front_on_one_side_leaves_contiguous_remainder() {
        let mut t = DamageTracker::new();
        let wallpaper = |tok| meta(1, 0., 0., 100., 100., tok);
        let window = meta_opaque(2, 0., 0., 50., 100., 7); // left half
        t.compute(&[wallpaper(0), window.clone()], scale1());
        t.commit();
        let d = t.compute(&[wallpaper(1), window], scale1());
        assert_eq!(d, vec![rect(50, 0, 50, 100)]);
    }

    #[test]
    fn transparent_front_does_not_clip() {
        // A front element with no opaque region must not suppress damage behind it.
        let mut t = DamageTracker::new();
        let wallpaper = |tok| meta(1, 0., 0., 100., 100., tok);
        let glass = meta(2, 25., 25., 50., 50., 7); // not opaque
        t.compute(&[wallpaper(0), glass.clone()], scale1());
        t.commit();
        let d = t.compute(&[wallpaper(1), glass], scale1());
        assert_eq!(d, vec![rect(0, 0, 100, 100)]);
    }

    #[test]
    fn opaque_to_physical_rounds_inward() {
        let s = scale1();
        let rf = |x, y, w, h| Rectangle::new(Point::from((x, y)), Size::from((w, h)));
        // 1.4..3.4 / 1.6..3.6 → fully-covered pixels [2,3) → rect(2,2,1,1).
        assert_eq!(
            opaque_to_physical(rf(1.4, 1.6, 2.0, 2.0), s),
            Some(rect(2, 2, 1, 1))
        );
        // Sub-pixel sliver covers no whole pixel → None.
        assert_eq!(opaque_to_physical(rf(0.5, 0.5, 0.4, 0.4), s), None);
    }
}
