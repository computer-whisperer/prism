//! Element: one thing to draw, back-to-front in a `FrameDescription`.

use std::num::NonZeroU64;

use smithay::utils::{Buffer, Physical, Point, Rectangle, Transform};

use crate::color::ColorDescription;
use crate::handle::{ShaderHandle, TextureHandle};

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

/// What kind of pixel source the element draws from.
#[derive(Clone, Debug)]
pub enum ElementSource {
    /// A texture (typically a client buffer). Sampled into the intermediate
    /// via the standard decode shader.
    Texture {
        handle: TextureHandle,
        /// Source rect in buffer coordinates (for cropping / viewport).
        src: Rectangle<f64, Buffer>,
        /// Per-element alpha multiplier. Combined with the alpha sampled
        /// from the texture in the decode shader.
        alpha: f32,
    },
    /// A solid color, in the color space declared by `Element::color`.
    SolidColor { rgba: [f32; 4] },
    /// A custom shader. Escape hatch for special effects (blur, color picker,
    /// screen recorder cursor highlight, etc.). The shader writes into the
    /// intermediate; its output should be in the element's declared `color`
    /// space, which the renderer decodes from like any other source.
    CustomShader {
        shader: ShaderHandle,
        textures: Vec<(&'static str, TextureHandle)>,
        uniforms: Vec<(&'static str, ShaderUniform)>,
    },
}

/// Uniform value for a custom shader binding.
#[derive(Clone, Copy, Debug)]
pub enum ShaderUniform {
    F32(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    I32(i32),
}

/// One element to draw. Back-to-front order in `FrameDescription::elements`.
#[derive(Clone, Debug)]
pub struct Element {
    /// Stable cross-frame ID. Same ID this frame as last frame = same element
    /// (used for damage diff, direct-scanout matching).
    pub id: ElementId,
    /// Output rect: where on the output this element draws.
    pub geometry: Rectangle<i32, Physical>,
    /// Buffer-to-output transform (rotation/flip).
    pub transform: Transform,
    /// Color description of the source content. Drives the decode shader.
    pub color: ColorDescription,
    /// The pixel source.
    pub source: ElementSource,
    /// Opaque sub-regions of this element, in output physical coords.
    /// Used by the renderer for occlusion culling of elements behind this one.
    /// Empty = treat as fully translucent.
    pub opaque_regions: Vec<Rectangle<i32, Physical>>,
}

impl Element {
    pub fn position(&self) -> Point<i32, Physical> {
        self.geometry.loc
    }
}
