//! Opaque handles for renderer-owned resources.
//!
//! Layout/protocol code holds handles; only the renderer dereferences them.
//! This is what keeps `ash` and Vulkan types from leaking out of the renderer
//! crate.

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

/// Opaque renderer-owned texture handle.
///
/// Allocated by the renderer when a buffer is imported (dmabuf, shm, or
/// constructed texture). The renderer maintains the handle → `VkImage` mapping;
/// other crates pass handles through without dereferencing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TextureHandle(NonZeroU64);

impl TextureHandle {
    /// Allocate a fresh handle from a global counter. Renderer-internal API;
    /// non-renderer code receives handles from the renderer and must not call
    /// this directly.
    pub fn alloc() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        Self(NonZeroU64::new(n).expect("texture handle counter overflow"))
    }

    pub fn raw(self) -> u64 {
        self.0.get()
    }
}

/// Opaque renderer-owned shader handle.
///
/// Used by `ElementSource::CustomShader` to reference a shader program owned
/// by the renderer. Built-in decode/postprocess shaders are renderer-internal
/// and do not need handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShaderHandle(NonZeroU64);

impl ShaderHandle {
    pub fn alloc() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        Self(NonZeroU64::new(n).expect("shader handle counter overflow"))
    }

    pub fn raw(self) -> u64 {
        self.0.get()
    }
}
