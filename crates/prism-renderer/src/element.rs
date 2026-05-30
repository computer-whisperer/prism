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
use prism_frame::{ElementId, Logical, Point, Rectangle, Size};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a over the bit patterns of a slice of `f32`s — a cheap content
/// fingerprint for solid-colour / border elements. Exact-equality change
/// detection: identical colours hash identically.
fn fnv_f32s(xs: &[f32]) -> u64 {
    xs.iter().fold(FNV_OFFSET, |h, x| {
        (h ^ x.to_bits() as u64).wrapping_mul(FNV_PRIME)
    })
}

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

/// Per-element metadata the damage tracker keys on — one entry per `RenderEl`
/// (a border is one element even though it lowers to up to four stripe draws).
/// Decoupled from the flat [`ElementDraw`] stream because the tracker works at
/// element granularity while `render_frame` works at draw granularity.
#[derive(Clone, Debug)]
pub struct FrameElementMeta {
    pub id: ElementId,
    /// Output rect in logical pixels (the element's bounding box; for a border,
    /// the outer rect). The tracker converts to physical for its diff.
    pub geometry: Rectangle<f64, Logical>,
    /// Content fingerprint — changes iff the element's pixels changed without
    /// its geometry moving (surface re-commit, focus-colour change). The
    /// tracker re-damages the element's geometry when this differs from the
    /// stored value.
    pub content_token: u64,
}

/// One frame lowered for one output: the flat draw stream `render_frame`
/// consumes, plus the per-element metadata the damage tracker diffs. The two
/// are intentionally *not* index-aligned (one border → one `meta`, four
/// `draws`); they serve different consumers.
pub struct LoweredFrame {
    pub draws: Vec<ElementDraw>,
    pub meta: Vec<FrameElementMeta>,
}

/// Project + flatten a back-to-front `RenderEl` list into a [`LoweredFrame`].
/// Builds the per-output projector once from `view_size`, then lowers each
/// element to its draws and records its metadata. `white_view` backs
/// solid-colour and border draws; `output_peak_nits_rgb` is the per-output
/// display-referred decode clamp threaded into every draw's push constants.
///
/// **Occlusion culling.** Elements fully hidden behind opaque content drawn in
/// front of them emit no draws (see [`cull_occluded`]). This only affects the
/// `draws` stream (the decode work); `meta` still describes *every* element, so
/// the damage tracker keeps seeing hidden elements — when an occluder later
/// moves away, its old region is damaged and the now-revealed element repaints.
pub fn lower_elements(
    elements: &[RenderEl],
    view_size: Size<f64, Logical>,
    white_view: vk::ImageView,
    output_peak_nits_rgb: [f32; 3],
) -> LoweredFrame {
    let project = make_projector(view_size);
    let visible = cull_occluded(elements);
    let mut draws = Vec::with_capacity(elements.len());
    let mut meta = Vec::with_capacity(elements.len());
    for (i, el) in elements.iter().enumerate() {
        // Metadata covers every element (visible or culled) so the damage diff
        // tracks occluded elements too.
        meta.push(FrameElementMeta {
            id: el.id(),
            geometry: el.geometry(),
            content_token: el.content_token(),
        });
        if visible[i] {
            el.lower(&project, white_view, output_peak_nits_rgb, &mut draws);
        }
    }
    LoweredFrame { draws, meta }
}

/// Decide which elements are visible, by the painter's-algorithm occlusion test
/// smithay's `OutputDamageTracker` uses: walk front-to-back, accumulate the
/// opaque regions of everything in front, and mark an element culled when its
/// geometry is fully covered (`subtract_rects` leaves nothing).
///
/// `elements` is in prism's back-to-front draw order (earlier paints behind),
/// so we iterate in reverse to go front-to-back. Returns a visibility mask
/// index-aligned to `elements`.
///
/// Coordinate space is logical `f64` — the same space the geometry is projected
/// from. Abutting tiles share exact logical edges, so they rasterise to
/// adjacent pixels with no seam; culling here can't open a gap the draw path
/// wouldn't also close.
fn cull_occluded(elements: &[RenderEl]) -> Vec<bool> {
    let mut visible = vec![true; elements.len()];
    let mut occluders: Vec<Rectangle<f64, Logical>> = Vec::new();
    let mut culled = 0usize;
    for (i, el) in elements.iter().enumerate().rev() {
        let geometry = el.geometry();
        // Fully covered by the opaque content already collected from in front?
        if !occluders.is_empty()
            && geometry
                .subtract_rects(occluders.iter().copied())
                .is_empty()
        {
            visible[i] = false;
            culled += 1;
            // A hidden element contributes no new occlusion (we don't draw it),
            // so don't add its opaque regions.
            continue;
        }
        el.push_opaque_regions(&mut occluders);
    }
    if culled > 0 {
        // `debug!`, not `trace!`: release builds set `release_max_level_debug`,
        // so a `trace!` here would compile out — and the test rig runs release.
        // Opt in with `RUST_LOG=cull=debug` (mirrors the "damage" target).
        tracing::debug!(
            target: "cull",
            total = elements.len(),
            culled,
            "occlusion culling"
        );
    }
    visible
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
/// - 2 = PQ (SMPTE ST 2084) — EOTF yields absolute nits; `sdr_white_nits`
///   is 1.0 (pass-through — PQ is not anchored, see decode_luminance_scale)
/// - 4 = Gamma 2.2 (modern SDR default per wp_color_management v2)
/// - 5 = BT.1886 (with default Lw/Lb → pure pow 2.4)
#[derive(Clone, Copy, Debug)]
pub struct SurfaceColorParams {
    pub transfer: i32,
    /// Post-EOTF luminance multiplier into the intermediate's anchored
    /// absolute-nits space. For non-PQ transfers it is the nits the source's
    /// 1.0 maps to (the output reference-white level for the anchored intents,
    /// or the content's declared reference luminance for absolute). For PQ —
    /// whose EOTF already yields absolute nits — it is 1.0 (pass-through; PQ is
    /// not anchored). Computed by `description_to_params`; applied uniformly
    /// across transfers in `decode.frag`.
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

/// How the decode shader must interpret a sampled texture's alpha channel.
/// A buffer-format property (X-format vs A-format, YUV), not a per-element
/// opacity — see `shaders/decode.frag`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AlphaMode {
    /// No meaningful alpha: opaque `X`-formats (the `X` byte is undefined per
    /// the DRM/wl_shm contract) and YUV video. The shader ignores the sampled
    /// alpha and treats the texel as fully opaque (alpha = 1.0). Default so a
    /// surface whose format we can't classify never leaks garbage alpha.
    #[default]
    Opaque,
    /// Premultiplied alpha — the Wayland `wl_surface` contract for `A`-formats.
    /// Color channels are already multiplied by alpha, so the shader
    /// un-premultiplies before the transfer EOTF (which must run on straight
    /// color) and re-premultiplies at output for the premultiplied src-over
    /// blend. (The solid-color path also rides this: its sampled white texel
    /// has alpha = 1.0, so un-premultiply is a no-op.)
    Premultiplied,
}

impl AlphaMode {
    /// Shader code for `DecodePush::alpha_mode`. Keep in sync with
    /// `shaders/decode.frag`.
    pub fn code(self) -> i32 {
        match self {
            Self::Opaque => 0,
            Self::Premultiplied => 1,
        }
    }
}

/// Sampled-texture surface element. Used for wl_surface content (xdg-shell
/// toplevels, popups, layer-shell content, subsurfaces) and for the cursor.
/// One per surface tree node; produced by walking the surface tree at
/// frame-build time.
pub struct SurfaceEl {
    /// Stable cross-frame id (the owning surface's allocated id). Same id this
    /// frame as last = same element, for the damage diff.
    pub id: ElementId,
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
    /// Content version: the surface's buffer commit count. The damage tracker
    /// re-damages the surface when this advances (geometry unchanged but pixels
    /// changed). The walk derives it from the `wl_surface`'s `CommitCounter`.
    pub content_commit: u64,
    /// Fully-opaque sub-rects of this surface, in **absolute output-space
    /// logical** pixels (the walk offsets the surface-relative regions from
    /// smithay's `RendererSurfaceState::opaque_regions()` by `geometry.loc`).
    /// Used only for occlusion culling: elements drawn behind these rects can
    /// be skipped. Empty when the buffer has (or may have) alpha and the client
    /// declared no opaque region — the conservative case that never culls.
    ///
    /// NOTE: this reflects only the *buffer's* opacity. The per-element fade
    /// `alpha` animation multiplier is not yet plumbed into the decode path
    /// (mapped.rs drops it); once it is, a surface with effective `alpha < 1.0`
    /// must report no opaque regions, or culling will hide content behind a
    /// translucent fading tile.
    pub opaque: Vec<Rectangle<f64, Logical>>,
    pub src_rect_uv: [f32; 4],
    pub color: SurfaceColorParams,
    /// How to interpret the sampled alpha (X-format/YUV → opaque; A-format →
    /// premultiplied). Set from the buffer's DRM/wl_shm fourcc at frame-build
    /// time, since `vk::Format` alone can't distinguish `Xrgb` from `Argb`.
    pub alpha_mode: AlphaMode,
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
        push.alpha_mode = self.alpha_mode.code();
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
    /// Stable cross-frame id (surface id for single-pixel-buffer solids, or the
    /// owning layout element's allocated id for backdrops / backgrounds).
    pub id: ElementId,
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
    /// Stable cross-frame id (the owning `FocusRing`'s allocated id). The four
    /// lowered stripes share it — the border is one element for damage.
    pub id: ElementId,
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
    /// Stable cross-frame id of this element.
    pub fn id(&self) -> ElementId {
        match self {
            Self::Surface(s) => s.id,
            Self::SolidColor(s) => s.id,
            Self::Border(b) => b.id,
        }
    }

    /// Output rect in logical pixels (bounding box; for a border, the outer rect).
    pub fn geometry(&self) -> Rectangle<f64, Logical> {
        match self {
            Self::Surface(s) => s.geometry,
            Self::SolidColor(s) => s.geometry,
            Self::Border(b) => b.geometry,
        }
    }

    /// Append this element's fully-opaque rects (absolute output-space logical)
    /// to `out`, for occlusion culling. An element drawn behind the union of
    /// these rects contributes nothing visible and can be skipped.
    ///
    /// - `Surface`: the buffer's opaque regions (already absolute; see
    ///   [`SurfaceEl::opaque`]). Empty for translucent / unknown-alpha buffers.
    /// - `SolidColor`: the whole geometry iff the colour is fully opaque
    ///   (`alpha == 1.0`); a translucent fill occludes nothing.
    /// - `Border`: nothing — a border is hollow stripes, so its bounding box is
    ///   mostly transparent and never a safe occluder.
    pub fn push_opaque_regions(&self, out: &mut Vec<Rectangle<f64, Logical>>) {
        match self {
            Self::Surface(s) => out.extend_from_slice(&s.opaque),
            Self::SolidColor(s) => {
                if s.color_bt2020_nits[3] >= 1.0 {
                    out.push(s.geometry);
                }
            }
            Self::Border(_) => {}
        }
    }

    /// Content fingerprint for the damage diff (see [`FrameElementMeta`]).
    /// Surfaces use their buffer commit count; solids/borders fingerprint their
    /// colour (and per-side thickness), so a focus-colour change re-damages the
    /// ring even though its geometry is unchanged.
    pub fn content_token(&self) -> u64 {
        match self {
            Self::Surface(s) => s.content_commit,
            Self::SolidColor(s) => fnv_f32s(&s.color_bt2020_nits),
            Self::Border(b) => {
                let c = fnv_f32s(&b.color_bt2020_nits);
                b.thickness
                    .iter()
                    .fold(c, |h, t| (h ^ t.to_bits()).wrapping_mul(FNV_PRIME))
            }
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: f64, y: f64, w: f64, h: f64) -> Rectangle<f64, Logical> {
        Rectangle::new(Point::from((x, y)), Size::from((w, h)))
    }

    fn solid(geo: Rectangle<f64, Logical>, alpha: f32) -> RenderEl {
        RenderEl::SolidColor(SolidColorEl {
            id: ElementId::alloc(),
            geometry: geo,
            color_bt2020_nits: [10.0, 10.0, 10.0, alpha],
        })
    }

    fn surface(geo: Rectangle<f64, Logical>, opaque: Vec<Rectangle<f64, Logical>>) -> RenderEl {
        RenderEl::Surface(SurfaceEl {
            id: ElementId::alloc(),
            texture_view: vk::ImageView::null(),
            chroma_view: None,
            yuv: 0,
            geometry: geo,
            content_commit: 0,
            opaque,
            src_rect_uv: [0.0, 0.0, 1.0, 1.0],
            color: SurfaceColorParams::default(),
            alpha_mode: AlphaMode::Opaque,
        })
    }

    fn border(geo: Rectangle<f64, Logical>) -> RenderEl {
        RenderEl::Border(BorderEl {
            id: ElementId::alloc(),
            geometry: geo,
            thickness: [2.0; 4],
            color_bt2020_nits: [10.0, 10.0, 10.0, 1.0],
        })
    }

    // Element vecs are back-to-front: index 0 paints behind the last.

    #[test]
    fn opaque_front_culls_fully_covered_back() {
        let els = vec![
            solid(rect(0.0, 0.0, 100.0, 100.0), 1.0),
            solid(rect(0.0, 0.0, 100.0, 100.0), 1.0),
        ];
        assert_eq!(cull_occluded(&els), vec![false, true]);
    }

    #[test]
    fn partial_cover_keeps_back() {
        let els = vec![
            solid(rect(0.0, 0.0, 100.0, 100.0), 1.0),
            solid(rect(0.0, 0.0, 50.0, 100.0), 1.0),
        ];
        assert_eq!(cull_occluded(&els), vec![true, true]);
    }

    #[test]
    fn translucent_front_occludes_nothing() {
        let els = vec![
            solid(rect(0.0, 0.0, 100.0, 100.0), 1.0),
            solid(rect(0.0, 0.0, 100.0, 100.0), 0.5),
        ];
        assert_eq!(cull_occluded(&els), vec![true, true]);
    }

    #[test]
    fn border_never_occludes() {
        let els = vec![
            solid(rect(0.0, 0.0, 100.0, 100.0), 1.0),
            border(rect(0.0, 0.0, 100.0, 100.0)),
        ];
        assert_eq!(cull_occluded(&els), vec![true, true]);
    }

    #[test]
    fn surface_opaque_region_culls_back() {
        let els = vec![
            solid(rect(0.0, 0.0, 100.0, 100.0), 1.0),
            surface(
                rect(0.0, 0.0, 100.0, 100.0),
                vec![rect(0.0, 0.0, 100.0, 100.0)],
            ),
        ];
        assert_eq!(cull_occluded(&els), vec![false, true]);
    }

    #[test]
    fn surface_without_opaque_region_occludes_nothing() {
        let els = vec![
            solid(rect(0.0, 0.0, 100.0, 100.0), 1.0),
            surface(rect(0.0, 0.0, 100.0, 100.0), vec![]),
        ];
        assert_eq!(cull_occluded(&els), vec![true, true]);
    }

    /// Two opaque halves sharing an exact edge tile the background; their union
    /// fully covers it, so the background is culled even though no single
    /// occluder covers it alone.
    #[test]
    fn abutting_occluders_union_covers_background() {
        let els = vec![
            solid(rect(0.0, 0.0, 100.0, 100.0), 1.0),
            solid(rect(0.0, 0.0, 50.0, 100.0), 1.0),
            solid(rect(50.0, 0.0, 50.0, 100.0), 1.0),
        ];
        assert_eq!(cull_occluded(&els), vec![false, true, true]);
    }
}
