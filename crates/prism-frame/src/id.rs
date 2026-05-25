//! Stable cross-frame element identity.

use std::num::NonZeroU64;

/// Stable cross-frame identifier for an element. Used for damage tracking
/// and direct-scanout matching. Typically derived from the underlying surface
/// (wl_surface object id, cursor instance id, etc.).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ElementId(NonZeroU64);

impl ElementId {
    pub fn from_raw(id: NonZeroU64) -> Self {
        Self(id)
    }

    pub fn raw(self) -> u64 {
        self.0.get()
    }
}
