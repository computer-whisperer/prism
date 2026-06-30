//! Per-frame phase profiling.
//!
//! Builds visibility into the per-output compositing cost: a [`FrameProfile`]
//! captures the phase breakdown of one composited frame — CPU phases (the
//! layout walk, damage diff, element lowering, encode-push build, command
//! record/submit) and, once the GPU half lands, the decode/encode/etc. sub-pass
//! GPU times — plus per-frame counters (element count, damage rects, damaged
//! area). A per-output [`ProfileRing`] keeps the last [`RING_CAPACITY`] frames
//! so the readout is tail-aware (p50/p95/p99), not just a smoothed mean.
//!
//! Collection runs **always-on in the background**: the per-frame cost is a
//! handful of `Instant`s plus fire-and-forget GPU timestamp writes, negligible
//! enough to leave on through days-long sessions so prism-tune can read the
//! breakdown the moment something feels laggy — no restart to "turn profiling
//! on". `PRISM_NO_PROFILE` opts out (to measure profiling's own overhead).
//! Nothing is logged periodically; `PRISM_PROFILE_LOG` adds a throttled debug
//! line for bring-up only. See [`Renderer::profile_enabled`].
//!
//! [`Renderer::profile_enabled`]: crate::renderer::Renderer::profile_enabled

use std::time::{Duration, Instant};

/// Recent-frame history depth. ~2–4 s at 60–120 Hz — enough for a stable p99
/// and a readable scrolling timeline without unbounded growth.
pub const RING_CAPACITY: usize = 256;

/// Number of timed phases per frame ([`Span`] variants).
pub const N_SPANS: usize = 9;

/// Stable display names for the spans, indexed by [`Span`] `as usize`.
pub const SPAN_NAMES: [&str; N_SPANS] = [
    // CPU phases (filled by the compositor as it builds the frame).
    "walk", "damage", "lower", "encpush", "submit",
    // GPU phases (filled by the renderer's timestamp queries; 0 until wired).
    "snapshot", "decode", "deband", "encode",
];

/// Index into [`FrameProfile::spans`] / [`SPAN_NAMES`]. Order is CPU phases
/// first (in frame order), then GPU phases (in command-stream order).
#[derive(Clone, Copy, Debug)]
#[repr(usize)]
pub enum Span {
    /// Layout walk: surface-tree traversal building the `RenderEl` stream.
    Walk = 0,
    /// Damage diff against the previous frame.
    Damage,
    /// Lowering `RenderEl`s into the flat `ElementDraw` stream.
    Lower,
    /// Building the per-output `EncodePush`.
    EncodePush,
    /// CPU cost of recording + submitting the render command buffer.
    Submit,
    /// GPU: window-close snapshot capture (intermediate → snapshot copies).
    Snapshot,
    /// GPU: decode pass (composite into the fp32 intermediate).
    Decode,
    /// GPU: deband pre-pass (separable blur for 8-bit SDR content).
    Deband,
    /// GPU: encode pass (color transform → scanout buffer).
    Encode,
}

/// One composited frame's phase breakdown and counters. Microseconds throughout.
///
/// Assembled across the frame: the compositor fills the CPU spans and counters
/// inline as it works; the renderer fills the GPU spans (currently left at 0).
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameProfile {
    /// Per-phase time in microseconds, indexed by [`Span`].
    pub spans: [f32; N_SPANS],
    /// Elements drawn this frame (lowered `ElementDraw` count).
    pub elements: u32,
    /// Damage rectangles this frame.
    pub damage_rects: u32,
    /// Damaged area this frame, physical pixels.
    pub damage_area_px: u64,
    /// Full output area, physical pixels — the denominator for the damage ratio.
    pub full_area_px: u64,
}

impl FrameProfile {
    /// Record a phase duration (converted to microseconds).
    #[inline]
    pub fn set(&mut self, span: Span, dur: Duration) {
        self.spans[span as usize] = dur.as_secs_f32() * 1.0e6;
    }

    /// Sum of the CPU spans (walk..=submit), microseconds. The CPU phases run
    /// sequentially, so this is a real per-frame CPU cost.
    pub fn cpu_us(&self) -> f32 {
        self.spans[Span::Walk as usize..=Span::Submit as usize]
            .iter()
            .sum()
    }

    /// Sum of the GPU spans (snapshot..=encode), microseconds.
    pub fn gpu_us(&self) -> f32 {
        self.spans[Span::Snapshot as usize..=Span::Encode as usize]
            .iter()
            .sum()
    }

    /// Damaged fraction of the output in [0, 1] (0 if the area is unknown).
    pub fn damage_ratio(&self) -> f32 {
        if self.full_area_px == 0 {
            0.0
        } else {
            (self.damage_area_px as f64 / self.full_area_px as f64) as f32
        }
    }
}

/// p50/p95/p99 of one span over the ring, microseconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpanStat {
    pub p50: f32,
    pub p95: f32,
    pub p99: f32,
}

/// Aggregate of a [`ProfileRing`] over its current window. This is the shape
/// the IPC layer will serialize for prism-tune; for now it backs the 1 Hz log.
#[derive(Clone, Debug)]
pub struct ProfileSummary {
    /// Number of frames the aggregate covers.
    pub frames: usize,
    /// Per-span percentiles, indexed by [`Span`].
    pub spans: [SpanStat; N_SPANS],
    /// Median damaged fraction of the output, [0, 1].
    pub damage_ratio_p50: f32,
    /// Median element count.
    pub elements_p50: f32,
}

impl ProfileSummary {
    /// One-line human summary for the throttled log: per-span p50/p95/p99 in µs,
    /// plus frame count, median damage %, and median element count.
    pub fn format_line(&self) -> String {
        let mut s = String::with_capacity(256);
        for (i, st) in self.spans.iter().enumerate() {
            // Skip GPU spans entirely while they read 0 (not yet wired), so the
            // line stays about what's actually measured.
            if i >= Span::Snapshot as usize && st.p50 == 0.0 && st.p99 == 0.0 {
                continue;
            }
            s.push_str(SPAN_NAMES[i]);
            s.push('=');
            s.push_str(&format!("{:.0}/{:.0}/{:.0} ", st.p50, st.p95, st.p99));
        }
        format!(
            "{}(µs p50/p95/p99) frames={} dmg={:.0}% elems={:.0}",
            s,
            self.frames,
            self.damage_ratio_p50 * 100.0,
            self.elements_p50,
        )
    }
}

/// Fixed-capacity ring of recent [`FrameProfile`]s for one output. Oldest frame
/// is evicted on overflow. Also owns the 1 Hz log throttle.
pub struct ProfileRing {
    buf: Box<[FrameProfile]>,
    /// Next write position.
    head: usize,
    /// Valid entry count (saturates at capacity).
    len: usize,
    /// Throttle gate for the periodic summary.
    last_summary: Instant,
}

impl ProfileRing {
    pub fn new() -> Self {
        Self::with_capacity(RING_CAPACITY)
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: vec![FrameProfile::default(); cap.max(1)].into_boxed_slice(),
            head: 0,
            len: 0,
            last_summary: Instant::now(),
        }
    }

    pub fn push(&mut self, p: FrameProfile) {
        self.buf[self.head] = p;
        self.head = (self.head + 1) % self.buf.len();
        self.len = (self.len + 1).min(self.buf.len());
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Chronological iterator over valid entries (oldest → newest).
    pub fn iter(&self) -> impl Iterator<Item = &FrameProfile> + '_ {
        let cap = self.buf.len();
        let start = (self.head + cap - self.len) % cap;
        (0..self.len).map(move |i| &self.buf[(start + i) % cap])
    }

    /// Aggregate the current window into a [`ProfileSummary`].
    pub fn summary(&self) -> ProfileSummary {
        let mut scratch: Vec<f32> = Vec::with_capacity(self.len);
        let mut spans = [SpanStat::default(); N_SPANS];
        for (s, stat) in spans.iter_mut().enumerate() {
            scratch.clear();
            scratch.extend(self.iter().map(|p| p.spans[s]));
            sort_f32(&mut scratch);
            *stat = SpanStat {
                p50: pick(&scratch, 0.50),
                p95: pick(&scratch, 0.95),
                p99: pick(&scratch, 0.99),
            };
        }
        scratch.clear();
        scratch.extend(self.iter().map(|p| p.damage_ratio()));
        sort_f32(&mut scratch);
        let damage_ratio_p50 = pick(&scratch, 0.50);
        scratch.clear();
        scratch.extend(self.iter().map(|p| p.elements as f32));
        sort_f32(&mut scratch);
        let elements_p50 = pick(&scratch, 0.50);
        ProfileSummary {
            frames: self.len,
            spans,
            damage_ratio_p50,
            elements_p50,
        }
    }

    /// Return a fresh summary at most once per second; `None` otherwise (or when
    /// the ring is empty). Resets the throttle when it fires.
    pub fn summary_due(&mut self) -> Option<ProfileSummary> {
        if self.len == 0 || self.last_summary.elapsed() < Duration::from_secs(1) {
            return None;
        }
        self.last_summary = Instant::now();
        Some(self.summary())
    }
}

impl Default for ProfileRing {
    fn default() -> Self {
        Self::new()
    }
}

/// Sort ascending, treating NaN as equal (timings are never NaN in practice).
fn sort_f32(v: &mut [f32]) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// Nearest-rank percentile of an already-sorted slice. `q` in [0, 1].
fn pick(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (((sorted.len() - 1) as f32) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prof(walk_us: f32) -> FrameProfile {
        let mut p = FrameProfile::default();
        p.spans[Span::Walk as usize] = walk_us;
        p
    }

    #[test]
    fn ring_wraps_and_iterates_oldest_to_newest() {
        let mut ring = ProfileRing::with_capacity(3);
        assert!(ring.is_empty());
        for i in 1..=5u32 {
            ring.push(prof(i as f32));
        }
        // Capacity 3, pushed 1..=5 → holds the last three: 3, 4, 5.
        assert_eq!(ring.len(), 3);
        let seen: Vec<f32> = ring.iter().map(|p| p.spans[Span::Walk as usize]).collect();
        assert_eq!(seen, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        // Sorted 0,10,...,90 (n=10). Nearest-rank index = round((n-1)*q).
        let v: Vec<f32> = (0..10).map(|i| (i * 10) as f32).collect();
        assert_eq!(pick(&v, 0.50), 50.0); // round(9*0.5)=5 → 50
        assert_eq!(pick(&v, 0.95), 90.0); // round(9*0.95)=9 → 90
        assert_eq!(pick(&v, 0.99), 90.0); // round(9*0.99)=9 → 90
        assert_eq!(pick(&v, 0.0), 0.0);
        assert_eq!(pick(&[], 0.5), 0.0);
    }

    #[test]
    fn summary_aggregates_the_window() {
        let mut ring = ProfileRing::with_capacity(8);
        for i in 0..5u32 {
            let mut p = prof((i * 100) as f32);
            p.elements = 10;
            p.damage_area_px = 50;
            p.full_area_px = 100;
            ring.push(p);
        }
        let s = ring.summary();
        assert_eq!(s.frames, 5);
        // Walk values 0,100,200,300,400 → p50 = round(4*0.5)=2 → 200.
        assert_eq!(s.spans[Span::Walk as usize].p50, 200.0);
        assert_eq!(s.elements_p50, 10.0);
        assert!((s.damage_ratio_p50 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn damage_ratio_guards_zero_area() {
        assert_eq!(FrameProfile::default().damage_ratio(), 0.0);
    }
}
