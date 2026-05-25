//! Render element vocabulary — Vulkan-native, no GLES coupling.
//!
//! What niri's `render_helpers/` provided as a trait-laden
//! `RenderElement<R: NiriRenderer>` zoo, this provides as plain data: each
//! variant is the geometry + colorimetry the higher layers (window, layer,
//! layout) describe, and lowers to one or more
//! [`ElementDraw`](crate::ElementDraw)s for the actual decode pass.
//!
//! Lowering rule: every element samples a texture and writes the result
//! into the fp16 BT.2020-linear intermediate. Solid-color elements
//! (borders, layer backgrounds) sample the renderer's 1×1 white texel
//! ([`Renderer::white_view`](crate::Renderer::white_view)) with
//! `transfer = Linear` and bake their colour into the per-element tint.
//!
//! Coordinate space: elements carry their geometry in **output-space logical
//! pixels** (`Rectangle<f64, Logical>`). The layout walk produces them in that
//! space; the renderer owns the logical → Vulkan-clip-space projection, applied
//! once at lowering time ([`lower_elements`]) from the output's `view_size`.
//! This keeps clip-space out of the layout entirely and gives the renderer the
//! output-space geometry it needs for damage tracking.

use crate::pipeline::decode::DecodePush;
use crate::renderer::ElementDraw;
use ash::vk;
use prism_frame::{Logical, Point, Rectangle, Size};

/// Logical → Vulkan-clip-space projection for one output.
///
/// Clip space is `[-1, 1] × [-1, 1]` over the full framebuffer, so the mapping
/// depends only on the output's logical `view_size` — it is independent of the
/// physical (fractional-scaled) framebuffer extent, since `[-1, 1]` always
/// means "full framebuffer".
pub fn make_projector(
    view_size: Size<f64, Logical>,
) -> impl Fn(Rectangle<f64, Logical>) -> [f32; 4] {
    let w = view_size.w.max(1.0);
    let h = view_size.h.max(1.0);
    move |rect: Rectangle<f64, Logical>| -> [f32; 4] {
        let x0 = (2.0 * rect.loc.x / w - 1.0) as f32;
        let y0 = (2.0 * rect.loc.y / h - 1.0) as f32;
        let x1 = (2.0 * (rect.loc.x + rect.size.w) / w - 1.0) as f32;
        let y1 = (2.0 * (rect.loc.y + rect.size.h) / h - 1.0) as f32;
        [x0, y0, x1, y1]
    }
}

/// Project + flatten a back-to-front `RenderEl` list into the `ElementDraw`
/// stream `render_frame` consumes. Builds the per-output projector once from
/// `view_size`, then lowers each element. `white_view` backs solid-colour and
/// border draws; `output_peak_nits_rgb` is the per-output display-referred
/// decode clamp threaded into every draw's push constants.
pub fn lower_elements(
    elements: &[RenderEl],
    view_size: Size<f64, Logical>,
    white_view: vk::ImageView,
    output_peak_nits_rgb: [f32; 3],
) -> Vec<ElementDraw> {
    let project = make_projector(view_size);
    let mut out = Vec::with_capacity(elements.len());
    for el in elements {
        el.lower(&project, white_view, output_peak_nits_rgb, &mut out);
    }
    out
}

/// Color-decoding parameters for a [`SurfaceEl`]. Captures *what
/// colorspace the client's pixels are in* so the decode shader picks
/// the right inverse transfer function. Constructed by the integrator
/// (prism-protocols) from the surface's `wp_color_management_v1`
/// image description; the default mirrors the historical pre-color-
/// management behavior (sRGB with 80-nit white).
///
/// Transfer codes match `decode.frag`:
///
/// - 0 = Linear (already-linear pixels, e.g. ext_linear)
/// - 1 = sRGB piecewise EOTF (default for unmanaged surfaces)
/// - 2 = PQ (SMPTE ST 2084) — `sdr_white_nits` ignored, pixels
///   already in absolute nits after decode
/// - 4 = Gamma 2.2 (modern SDR default per wp_color_management v2)
/// - 5 = BT.1886 (with default Lw/Lb → pure pow 2.4)
#[derive(Clone, Copy, Debug)]
pub struct SurfaceColorParams {
    pub transfer: i32,
    /// Nits value the client's 1.0 white maps to. Ignored when
    /// `transfer == 2` (PQ); else multiplied into the linear result
    /// to anchor into the same absolute-nits coordinate system the
    /// intermediate buffer uses.
    pub sdr_white_nits: f32,
    /// Linear-light matrix converting the client's primaries into the
    /// BT.2020 working space, row-major (`out[i] = Σ_k m[i][k]·in[k]`).
    /// Built from the surface's `wp_color_management_v1` primaries (or
    /// the sRGB/BT.709 default for unmanaged clients) via
    /// [`prism_frame::primaries_to_bt2020`]. Near-identity for content
    /// already in BT.2020. `to_draw` lowers this into the decode shader's
    /// `decode_matrix`.
    pub primaries_to_bt2020: prism_frame::Mat3,
    /// YUV→RGB coefficient set for YUV-sampled surfaces, by the source's
    /// primaries: 0 = BT.709 (and sRGB-primaries SDR video), 1 = BT.2020.
    /// Lowered into `DecodePush::yuv_matrix`; ignored unless the surface's
    /// texture is YUV (`SurfaceEl::yuv != 0`).
    pub yuv_matrix: i32,
}

impl Default for SurfaceColorParams {
    fn default() -> Self {
        // Pre-color-management default: sRGB EOTF + 80-nit white, and the
        // sRGB/BT.709 → BT.2020 primaries conversion (legacy clients author
        // in sRGB by convention). Every client before wp_color_management_v1
        // lands in their toolkit flows through this path.
        Self {
            transfer: 1,
            sdr_white_nits: 80.0,
            primaries_to_bt2020: prism_frame::srgb_to_bt2020_matrix(),
            yuv_matrix: 0,
        }
    }
}

/// Sampled-texture surface element. Used for wl_surface content (xdg-shell
/// toplevels, popups, layer-shell content, subsurfaces) and for the cursor.
/// One per surface tree node; produced by walking the surface tree at
/// frame-build time.
pub struct SurfaceEl {
    /// Sampled texture. For YUV surfaces (`yuv != 0`) this is the luma plane.
    pub texture_view: vk::ImageView,
    /// Chroma plane for YUV surfaces; `None` for RGB. Pairs with `yuv`.
    pub chroma_view: Option<vk::ImageView>,
    /// YUV plane layout: 0 = RGB, 1 = NV12 (8-bit), 2 = P010 (10-bit).
    /// Set from the imported texture's `YuvKind`; lowered into
    /// `DecodePush::yuv`.
    pub yuv: i32,
    /// Output rect in logical pixels; projected to clip space at lowering.
    pub geometry: Rectangle<f64, Logical>,
    pub src_rect_uv: [f32; 4],
    pub color: SurfaceColorParams,
}

impl SurfaceEl {
    pub fn to_draw(
        &self,
        project: &dyn Fn(Rectangle<f64, Logical>) -> [f32; 4],
        output_peak_nits_rgb: [f32; 3],
    ) -> ElementDraw {
        let mut push = DecodePush::identity_srgb(project(self.geometry), self.src_rect_uv);
        push.transfer = self.color.transfer;
        push.sdr_white_nits = self.color.sdr_white_nits;
        push.decode_matrix = mat3_to_mat4_colmajor(self.color.primaries_to_bt2020);
        push.yuv = self.yuv;
        push.yuv_matrix = self.color.yuv_matrix;
        let [r, g, b] = output_peak_nits_rgb;
        push.output_peak_nits_rgba = [r, g, b, 0.0];
        ElementDraw {
            texture_view: self.texture_view,
            chroma_view: self.chroma_view,
            push,
        }
    }
}

/// Lay a row-major 3×3 into the upper-left of a column-major `mat4`
/// (`[f32; 16]`, the decode push's `decode_matrix` storage). The shader reads
/// `mat3(decode_matrix)`, so only the 3×3 block matters; the rest is set to
/// the identity tail. Column `j` of the mat4 holds `(m[0][j], m[1][j], m[2][j])`.
fn mat3_to_mat4_colmajor(m: prism_frame::Mat3) -> [f32; 16] {
    [
        m[0][0], m[1][0], m[2][0], 0.0, // column 0
        m[0][1], m[1][1], m[2][1], 0.0, // column 1
        m[0][2], m[1][2], m[2][2], 0.0, // column 2
        0.0, 0.0, 0.0, 1.0, // column 3
    ]
}

/// Uniformly-coloured rectangle. Backs window/layer backgrounds,
/// fullscreen-padding fills, debug overlays.
pub struct SolidColorEl {
    /// Output rect in logical pixels; projected to clip space at lowering.
    pub geometry: Rectangle<f64, Logical>,
    /// Colour in BT.2020 linear nits, RGBA. Use [`srgb_to_bt2020_nits`] to
    /// convert from configured sRGB hex values.
    pub color_bt2020_nits: [f32; 4],
}

impl SolidColorEl {
    pub fn to_draw(
        &self,
        project: &dyn Fn(Rectangle<f64, Logical>) -> [f32; 4],
        white_view: vk::ImageView,
        output_peak_nits_rgb: [f32; 3],
    ) -> ElementDraw {
        solid_color_draw(
            project(self.geometry),
            self.color_bt2020_nits,
            white_view,
            output_peak_nits_rgb,
        )
    }
}

/// Lower a single already-projected solid-colour rect to an `ElementDraw`.
/// Shared by [`SolidColorEl`] (one rect) and [`BorderEl`] (four stripes); both
/// sample the renderer's white texel and bake the colour into the tint.
fn solid_color_draw(
    rect_clip: [f32; 4],
    color_bt2020_nits: [f32; 4],
    white_view: vk::ImageView,
    output_peak_nits_rgb: [f32; 3],
) -> ElementDraw {
    let mut push = DecodePush::solid(rect_clip, color_bt2020_nits);
    let [r, g, b] = output_peak_nits_rgb;
    push.output_peak_nits_rgba = [r, g, b, 0.0];
    ElementDraw {
        texture_view: white_view,
        chroma_view: None,
        push,
    }
}

/// Window / layer border — four solid stripes around `rect_clip`, each side
/// independently thickened. Top / right / bottom / left order matches CSS.
///
/// Thicknesses are in clip-space units (already projected from logical
/// pixels by the caller). Zero-thickness sides emit no draws.
///
/// Rounded corners: not yet supported — when added, a per-corner SDF
/// fragment shader and a real radius field will land here as a separate
/// variant or extra fields. For now the border is sharp-cornered, which
/// matches niri's default config.
pub struct BorderEl {
    /// Outer rect (window + margin) in logical pixels.
    pub geometry: Rectangle<f64, Logical>,
    /// Per-side thickness in logical pixels, `[top, right, bottom, left]`
    /// (CSS order). Each side independently thickened; zero emits no stripe.
    pub thickness: [f64; 4],
    pub color_bt2020_nits: [f32; 4],
}

impl BorderEl {
    pub fn push_draws(
        &self,
        project: &dyn Fn(Rectangle<f64, Logical>) -> [f32; 4],
        white_view: vk::ImageView,
        output_peak_nits_rgb: [f32; 3],
        out: &mut Vec<ElementDraw>,
    ) {
        // Per-side thickness in clip space: project the outer rect and the
        // inner rect (outer shrunk by the logical thickness on each side),
        // then take the clip-space difference along each axis. This routes the
        // logical→clip projection through without needing the output's pixel
        // scale here.
        let [t_log, r_log, b_log, l_log] = self.thickness;
        let outer_clip = project(self.geometry);
        let inner_logical = Rectangle::new(
            self.geometry.loc + Point::from((l_log, t_log)),
            Size::from((
                self.geometry.size.w - (l_log + r_log),
                self.geometry.size.h - (t_log + b_log),
            )),
        );
        let inner_clip = project(inner_logical);

        let [x_min, y_min, x_max, y_max] = outer_clip;
        let t = inner_clip[1] - outer_clip[1]; // top
        let r = outer_clip[2] - inner_clip[2]; // right
        let b = outer_clip[3] - inner_clip[3]; // bottom
        let l = inner_clip[0] - outer_clip[0]; // left

        let color = self.color_bt2020_nits;
        // Top stripe — full width × t.
        if t > 0.0 {
            out.push(solid_color_draw(
                [x_min, y_min, x_max, y_min + t],
                color,
                white_view,
                output_peak_nits_rgb,
            ));
        }
        // Bottom stripe — full width × b.
        if b > 0.0 {
            out.push(solid_color_draw(
                [x_min, y_max - b, x_max, y_max],
                color,
                white_view,
                output_peak_nits_rgb,
            ));
        }
        // Left stripe — l × inner-height (between the horizontal stripes).
        if l > 0.0 {
            out.push(solid_color_draw(
                [x_min, y_min + t, x_min + l, y_max - b],
                color,
                white_view,
                output_peak_nits_rgb,
            ));
        }
        // Right stripe — r × inner-height.
        if r > 0.0 {
            out.push(solid_color_draw(
                [x_max - r, y_min + t, x_max, y_max - b],
                color,
                white_view,
                output_peak_nits_rgb,
            ));
        }
    }
}

/// Tagged dispatch over the element vocabulary. Callers build a
/// `Vec<RenderEl>` from the layout walk; the render path calls
/// [`RenderEl::lower`] on each to flatten into the [`ElementDraw`]
/// stream the renderer consumes.
pub enum RenderEl {
    Surface(SurfaceEl),
    SolidColor(SolidColorEl),
    Border(BorderEl),
}

impl RenderEl {
    pub fn lower(
        &self,
        project: &dyn Fn(Rectangle<f64, Logical>) -> [f32; 4],
        white_view: vk::ImageView,
        output_peak_nits_rgb: [f32; 3],
        out: &mut Vec<ElementDraw>,
    ) {
        match self {
            Self::Surface(s) => out.push(s.to_draw(project, output_peak_nits_rgb)),
            Self::SolidColor(s) => out.push(s.to_draw(project, white_view, output_peak_nits_rgb)),
            Self::Border(b) => b.push_draws(project, white_view, output_peak_nits_rgb, out),
        }
    }
}

/// Convert an unmultiplied sRGB-encoded RGBA colour into the BT.2020
/// linear-nits domain solid-color elements need.
///
/// `r`, `g`, `b` are in `[0, 1]` (sRGB-encoded); `a` passes through
/// unchanged. `sdr_white_nits` is the nits value the output's diffuse
/// white maps to — typically the per-output `sdr_white_nits` config
/// (commonly 80–200 for SDR-only setups, 203 for the BT.2408 reference).
///
/// Primaries conversion uses the shared [`prism_frame::srgb_to_bt2020_matrix`]
/// (BT.709 → BT.2020, Bradford-adapted) so solid colors and sampled surfaces
/// agree on the BT.709 → BT.2020 transform.
pub fn srgb_to_bt2020_nits(r: f32, g: f32, b: f32, a: f32, sdr_white_nits: f32) -> [f32; 4] {
    fn eotf(c: f32) -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }
    let lin = [eotf(r), eotf(g), eotf(b)];
    let m = prism_frame::srgb_to_bt2020_matrix();
    [
        (m[0][0] * lin[0] + m[0][1] * lin[1] + m[0][2] * lin[2]) * sdr_white_nits,
        (m[1][0] * lin[0] + m[1][1] * lin[1] + m[1][2] * lin[2]) * sdr_white_nits,
        (m[2][0] * lin[0] + m[2][1] * lin[1] + m[2][2] * lin[2]) * sdr_white_nits,
        a,
    ]
}
