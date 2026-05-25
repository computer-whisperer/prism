//! Stable cross-frame element identity.

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

/// Stable cross-frame identifier for an element. Used for damage tracking
/// and direct-scanout matching. Either derived from the underlying surface
/// (a per-`wl_surface` allocation stored in its data) or freshly [`alloc`](Self::alloc)ed
/// and held in the element's persistent owner (border / backdrop / background).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ElementId(NonZeroU64);

impl ElementId {
    pub fn from_raw(id: NonZeroU64) -> Self {
        Self(id)
    }

    pub fn raw(self) -> u64 {
        self.0.get()
    }

    /// Allocate a fresh, process-unique id from a monotonic counter. For
    /// elements with no natural wayland identity (borders, fullscreen
    /// backdrops, workspace backgrounds): the caller stores the result in the
    /// element's persistent owner so the id stays stable across frames.
    /// Monotonic and never recycled, so ids never collide.
    pub fn alloc() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        Self(NonZeroU64::new(n).expect("ElementId counter overflow"))
    }
}
