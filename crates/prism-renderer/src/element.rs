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

/// Fold an optional rect crop into a content token, expressed relative to
/// the element origin (same convention as `SurfaceClip::mix_token`): a pure
/// move stays a geometry-only diff, while a crop edge sweeping across the
/// element re-damages it in place.
fn mix_rect_clip(
    c: u64,
    clip: &Option<Rectangle<f64, Logical>>,
    origin: Point<f64, Logical>,
) -> u64 {
    match clip {
        None => c,
        Some(r) => {
            c ^ fnv_f32s(&[
                (r.loc.x - origin.x) as f32,
                (r.loc.y - origin.y) as f32,
                r.size.w as f32,
                r.size.h as f32,
            ])
        }
    }
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
    let view_size_log = [view_size.w as f32, view_size.h as f32];
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
            let start = draws.len();
            el.lower(&project, white_view, output_peak_nits_rgb, &mut draws);
            // The vertex shader inverts the projection to recover logical
            // pixel positions for the rounded-corner SDF; thread the view
            // size into every draw here rather than through each `to_draw`.
            for draw in &mut draws[start..] {
                draw.push.view_size_log = view_size_log;
            }
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
///   is then the anchoring ratio (1.0 = pass-through, i.e. absolute intent)
/// - 4 = Gamma 2.2 (modern SDR default per wp_color_management v2)
/// - 5 = BT.1886 (with default Lw/Lb → pure pow 2.4)
#[derive(Clone, Copy, Debug)]
pub struct SurfaceColorParams {
    pub transfer: i32,
    /// Post-EOTF luminance multiplier into the intermediate's anchored
    /// absolute-nits space. For non-PQ transfers it is the nits the source's
    /// 1.0 maps to (the output reference-white level for the anchored intents,
    /// or the content's declared reference luminance for absolute). For PQ —
    /// whose EOTF already yields absolute nits — it is the anchoring ratio
    /// (output-ref-white / content-ref-white, or 1.0 for absolute). Computed by
    /// `description_to_params`; applied uniformly across transfers in
    /// `decode.frag`.
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

impl SurfaceColorParams {
    /// Pass-through params for a texture already in the intermediate's working
    /// space (BT.2020 absolute nits, linear) — i.e. a `SnapshotTexture` for the
    /// close animation. Linear transfer (no EOTF), `sdr_white = 1.0` (no
    /// rescale), identity primaries (already BT.2020). The decode shader then
    /// just samples and re-emits, so the snapshot composites bit-identical to
    /// the window it captured (modulo the per-element fade in `tint.a`).
    pub fn passthrough() -> Self {
        Self {
            transfer: 0,
            sdr_white_nits: 1.0,
            primaries_to_bt2020: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            yuv_matrix: 0,
        }
    }
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

/// Rounded-rect clip applied to an element at decode time — the
/// `clip-to-geometry` window rule. The decode shader multiplies the
/// element's alpha by the SDF coverage of this box (`sdf_mode = 1`), so
/// pixels outside it (and outside its rounded corners) drop out with
/// one physical pixel of anti-aliasing.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceClip {
    /// Clip box in output-space logical pixels — the window's visual
    /// geometry, independent of this element's own rect (a subsurface can
    /// hang outside it and gets clipped away entirely).
    pub rect: Rectangle<f64, Logical>,
    /// Per-corner radii in logical pixels, `[tl, tr, br, bl]`.
    pub radii: [f32; 4],
}

impl SurfaceClip {
    /// Whether the clip would actually remove pixels from an element with
    /// this `geometry` — false when the element lies inside the clip box
    /// and clear of all four corner squares (niri's
    /// `ClippedSurfaceRenderElement::will_clip`). Callers skip clipping
    /// such elements: same pixels, but their opaque regions stay intact
    /// for occlusion culling and the shader skips the SDF.
    pub fn would_clip(&self, geometry: Rectangle<f64, Logical>) -> bool {
        if !self.rect.contains_rect(geometry) {
            return true;
        }
        let [tl, tr, br, bl] = self.radii.map(f64::from);
        let r = self.rect;
        let corner = |x: f64, y: f64, s: f64| {
            s > 0.0
                && geometry.overlaps(Rectangle::<f64, Logical>::new((x, y).into(), (s, s).into()))
        };
        corner(r.loc.x, r.loc.y, tl)
            || corner(r.loc.x + r.size.w - tr, r.loc.y, tr)
            || corner(r.loc.x + r.size.w - br, r.loc.y + r.size.h - br, br)
            || corner(r.loc.x, r.loc.y + r.size.h - bl, bl)
    }

    /// Fold this clip into a content fingerprint, relative to the element's
    /// own origin — a moving window keeps the same token (geometry diffs
    /// already re-damage moves), while a radius or relative-clip change
    /// re-damages in place (e.g. the corner-radius animation).
    fn mix_token(&self, token: u64, origin: Point<f64, Logical>) -> u64 {
        let rel = [
            (self.rect.loc.x - origin.x) as f32,
            (self.rect.loc.y - origin.y) as f32,
            self.rect.size.w as f32,
            self.rect.size.h as f32,
        ];
        token ^ fnv_f32s(&rel) ^ fnv_f32s(&self.radii)
    }

    fn apply_to_push(&self, push: &mut DecodePush) {
        push.sdf_mode = 1;
        push.sdf_box = rect_min_max(self.rect);
        push.sdf_radii = self.radii;
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
    /// NOTE: this reflects only the *buffer's* opacity. A surface with an
    /// effective per-element fade (`alpha < 1.0`) must report **no** opaque
    /// regions, or occlusion culling will hide content drawn behind the
    /// translucent fading tile. The layout walk clears this when it sets
    /// `alpha < 1.0` (see `push_surface_tree_elements`).
    pub opaque: Vec<Rectangle<f64, Logical>>,
    pub src_rect_uv: [f32; 4],
    pub color: SurfaceColorParams,
    /// How to interpret the sampled alpha (X-format/YUV → opaque; A-format →
    /// premultiplied). Set from the buffer's DRM/wl_shm fourcc at frame-build
    /// time, since `vk::Format` alone can't distinguish `Xrgb` from `Argb`.
    pub alpha_mode: AlphaMode,
    /// Per-element opacity multiplier in `[0, 1]`, `1.0` = fully opaque.
    /// Drives window fade animations (open/close, interactive-move dimming,
    /// the `opacity` window rule). Lowered into the decode push's `tint.a`,
    /// which the shader folds into the premultiplied output — scaling a
    /// premultiplied pixel by `alpha` is the correct fade regardless of the
    /// buffer's own (premultiplied) alpha. `1.0` is a no-op.
    pub alpha: f32,
    /// Rounded-rect clip (`clip-to-geometry`); `None` = unclipped. Set by
    /// the tile render pass on the window's normal surface tree. The
    /// reported opaque regions are shrunk accordingly at culling time
    /// (see [`RenderEl::push_opaque_regions`]).
    pub clip: Option<SurfaceClip>,
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
        // Per-element opacity rides `tint.a`. The shader computes
        // `alpha = sampled.a * tint.a` then outputs `vec4(bt2020 * alpha, alpha)`
        // (premultiplied), so multiplying `tint.a` by the fade scales the whole
        // premultiplied pixel — the correct fade for both opaque and
        // already-premultiplied buffers. `identity_srgb` left `tint = [1; 4]`.
        push.tint[3] = self.alpha;
        if let Some(clip) = &self.clip {
            clip.apply_to_push(&mut push);
        }
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
    /// Rounded-rect clip (`clip-to-geometry`); `None` = unclipped. Reached
    /// by single-pixel-buffer surfaces inside a clipped window's tree
    /// (e.g. video letterbox bars), which the walk lowers to solids.
    pub clip: Option<SurfaceClip>,
}

impl SolidColorEl {
    pub fn to_draw(
        &self,
        project: &dyn Fn(Rectangle<f64, Logical>) -> [f32; 4],
        white_view: vk::ImageView,
        output_peak_nits_rgb: [f32; 3],
    ) -> ElementDraw {
        let mut draw = solid_color_draw(
            project(self.geometry),
            self.color_bt2020_nits,
            white_view,
            output_peak_nits_rgb,
        );
        if let Some(clip) = &self.clip {
            clip.apply_to_push(&mut draw.push);
        }
        draw
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
/// Sharp corners only — the stripes rasterize exactly the painted band, so
/// this stays the cheap path for unrounded rings. Rounded rings use
/// [`RoundedBoxEl`], whose SDF quad covers the full bounding box.
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
    /// Rectangular crop (the overview/switch workspace-card band — see
    /// [`RenderEl::crop_to_rect`]); `None` = uncropped. Each lowered stripe
    /// is intersected with it, so a border straddling the card edge stops
    /// exactly at the band instead of bleeding into the neighbour card.
    pub clip: Option<Rectangle<f64, Logical>>,
}

impl BorderEl {
    pub fn push_draws(
        &self,
        project: &dyn Fn(Rectangle<f64, Logical>) -> [f32; 4],
        white_view: vk::ImageView,
        output_peak_nits_rgb: [f32; 3],
        out: &mut Vec<ElementDraw>,
    ) {
        // Build the four stripes in logical space (the projection is
        // affine, so projecting each stripe matches the old clip-space
        // arithmetic exactly), intersect with the optional crop, project,
        // emit. The horizontal stripes span the full width; the vertical
        // ones fill between them — same tiling as before.
        let [t, r, b, l] = self.thickness;
        let g = self.geometry;
        let stripes = [
            // Top — full width × t.
            Rectangle::new(g.loc, Size::from((g.size.w, t))),
            // Bottom — full width × b.
            Rectangle::new(
                g.loc + Point::from((0.0, g.size.h - b)),
                Size::from((g.size.w, b)),
            ),
            // Left — l × inner height (between the horizontal stripes).
            Rectangle::new(
                g.loc + Point::from((0.0, t)),
                Size::from((l, g.size.h - t - b)),
            ),
            // Right — r × inner height.
            Rectangle::new(
                g.loc + Point::from((g.size.w - r, t)),
                Size::from((r, g.size.h - t - b)),
            ),
        ];

        let color = self.color_bt2020_nits;
        for stripe in stripes {
            if stripe.size.w <= 0.0 || stripe.size.h <= 0.0 {
                continue;
            }
            let stripe = match self.clip {
                Some(clip) => match stripe.intersection(clip) {
                    Some(s) => s,
                    None => continue,
                },
                None => stripe,
            };
            out.push(solid_color_draw(
                project(stripe),
                color,
                white_view,
                output_peak_nits_rgb,
            ));
        }
    }
}

/// Rounded-corner box — a filled rounded rect, or a hollow rounded ring
/// (border / focus ring around a rounded window). One quad; the decode
/// shader's per-corner SDF computes coverage per fragment and folds it into
/// alpha (see `sdf_coverage` in `shaders/decode.frag`).
///
/// Used only when at least one corner radius is non-zero: the sharp case
/// stays on [`BorderEl`] / [`SolidColorEl`], whose stripe/quad draws
/// rasterize only the painted area instead of the full bounding box
/// (niri makes the same split — solid buffers unless the border shader is
/// actually needed).
pub struct RoundedBoxEl {
    /// Stable cross-frame id (the owning `FocusRing`'s allocated id).
    pub id: ElementId,
    /// Outer rect in logical pixels.
    pub geometry: Rectangle<f64, Logical>,
    /// Per-corner radii of the outer rect in logical pixels, clockwise from
    /// top-left: `[tl, tr, br, bl]` (matches `prism_config::CornerRadius`).
    /// The shader clamps each to half the shorter side.
    pub radii: [f32; 4],
    /// `Some([top, right, bottom, left])` (CSS order, logical pixels) draws a
    /// hollow ring of that per-side thickness; `None` draws a filled box.
    pub inset: Option<[f64; 4]>,
    pub color_bt2020_nits: [f32; 4],
    /// Rectangular crop (the overview/switch workspace-card band — see
    /// [`RenderEl::crop_to_rect`]); `None` = uncropped. Shrinks the drawn
    /// quad only — the SDF box stays at `geometry`, so the visible part of
    /// the rounded shape is unchanged, it just stops at the band edge.
    pub clip: Option<Rectangle<f64, Logical>>,
}

impl RoundedBoxEl {
    pub fn to_draw(
        &self,
        project: &dyn Fn(Rectangle<f64, Logical>) -> [f32; 4],
        white_view: vk::ImageView,
        output_peak_nits_rgb: [f32; 3],
    ) -> ElementDraw {
        // The crop shrinks the rasterized quad only; the SDF box below
        // keeps the full geometry, so the shape is unchanged — it just
        // stops at the crop edge. An empty intersection degenerates to a
        // zero-area quad (no fragments).
        let quad = match self.clip {
            Some(clip) => self
                .geometry
                .intersection(clip)
                .unwrap_or_else(|| Rectangle::new(self.geometry.loc, Size::default())),
            None => self.geometry,
        };
        let mut draw = solid_color_draw(
            project(quad),
            self.color_bt2020_nits,
            white_view,
            output_peak_nits_rgb,
        );
        draw.push.sdf_box = rect_min_max(self.geometry);
        draw.push.sdf_radii = self.radii;
        match self.inset {
            Some(inset) => {
                draw.push.sdf_mode = 2;
                draw.push.sdf_inset = inset.map(|v| v as f32);
            }
            None => draw.push.sdf_mode = 1,
        }
        draw
    }
}

/// The fully-covered interior of a rounded box, as up to two overlapping
/// bands (a horizontal one between the corner rows and a vertical one
/// between the corner columns), each inset by 1 logical px to stay clear of
/// the SDF's ≤ 1-physical-px anti-aliased edge — which is translucent even
/// on straight runs at fractional alignment. Shared by the opaque-region
/// reporting of filled [`RoundedBoxEl`]s and clip-shrunk surfaces/solids.
fn push_rounded_box_bands(
    g: Rectangle<f64, Logical>,
    radii: [f32; 4],
    out: &mut Vec<Rectangle<f64, Logical>>,
) {
    let [tl, tr, br, bl] = radii.map(f64::from);
    const AA: f64 = 1.0;
    // Horizontal band: full width, between the corner rows.
    let h_band = Rectangle::<f64, Logical>::new(
        Point::from((g.loc.x + AA, g.loc.y + tl.max(tr) + AA)),
        Size::from((
            g.size.w - 2.0 * AA,
            g.size.h - tl.max(tr) - bl.max(br) - 2.0 * AA,
        )),
    );
    // Vertical band: full height, between the corner columns.
    let v_band = Rectangle::<f64, Logical>::new(
        Point::from((g.loc.x + tl.max(bl) + AA, g.loc.y + AA)),
        Size::from((
            g.size.w - tl.max(bl) - tr.max(br) - 2.0 * AA,
            g.size.h - 2.0 * AA,
        )),
    );
    for band in [h_band, v_band] {
        if band.size.w > 0.0 && band.size.h > 0.0 {
            out.push(band);
        }
    }
}

/// Window / layer drop shadow — a Gaussian-blurred rounded box (Evan
/// Wallace's analytic approximation, ported from niri's shadow shader),
/// with the window region optionally cut out so translucent windows don't
/// sit on their own shadow. One quad; never occludes.
pub struct ShadowEl {
    /// Stable cross-frame id (the owning `Shadow`'s allocated id).
    pub id: ElementId,
    /// Quad rect in logical pixels — the shadow box expanded by the blur
    /// reach (3σ) on all sides.
    pub geometry: Rectangle<f64, Logical>,
    /// The box casting the shadow: window rect offset by the configured
    /// shadow offset and grown by `spread`. Logical pixels.
    pub shadow_box: Rectangle<f64, Logical>,
    /// Per-corner radii of `shadow_box`, `[tl, tr, br, bl]` logical px.
    /// The blur itself uses only `tl` (niri's single-radius limitation);
    /// the low-sigma crisp path honors all four.
    pub radii: [f32; 4],
    /// Gaussian sigma in logical px (`softness / 2`, CSS box-shadow
    /// convention). `< 0.1` renders a crisp rounded rect.
    pub sigma: f32,
    /// Cut-out (`draw-behind-window false`): the window's own rounded box,
    /// inside which the shadow does not paint.
    pub cutout: Option<SurfaceClip>,
    pub color_bt2020_nits: [f32; 4],
}

impl ShadowEl {
    pub fn to_draw(
        &self,
        project: &dyn Fn(Rectangle<f64, Logical>) -> [f32; 4],
        white_view: vk::ImageView,
        output_peak_nits_rgb: [f32; 3],
    ) -> ElementDraw {
        let mut draw = solid_color_draw(
            project(self.geometry),
            self.color_bt2020_nits,
            white_view,
            output_peak_nits_rgb,
        );
        draw.push.sdf_mode = 3;
        draw.push.sdf_box = rect_min_max(self.shadow_box);
        draw.push.sdf_radii = self.radii;
        draw.push.sdf_sigma = self.sigma;
        if let Some(cutout) = &self.cutout {
            draw.push.sdf_box2 = rect_min_max(cutout.rect);
            draw.push.sdf_radii2 = cutout.radii;
        }
        draw
    }
}

/// A logical rect as the `[x_min, y_min, x_max, y_max]` f32 quadruple the
/// SDF push constants use.
fn rect_min_max(r: Rectangle<f64, Logical>) -> [f32; 4] {
    [
        r.loc.x as f32,
        r.loc.y as f32,
        (r.loc.x + r.size.w) as f32,
        (r.loc.y + r.size.h) as f32,
    ]
}

/// Tagged dispatch over the element vocabulary. Callers build a
/// `Vec<RenderEl>` from the layout walk; the render path calls
/// [`RenderEl::lower`] on each to flatten into the [`ElementDraw`]
/// stream the renderer consumes.
pub enum RenderEl {
    Surface(SurfaceEl),
    SolidColor(SolidColorEl),
    Border(BorderEl),
    RoundedBox(RoundedBoxEl),
    Shadow(ShadowEl),
}

impl RenderEl {
    /// Stable cross-frame id of this element.
    pub fn id(&self) -> ElementId {
        match self {
            Self::Surface(s) => s.id,
            Self::SolidColor(s) => s.id,
            Self::Border(b) => b.id,
            Self::RoundedBox(r) => r.id,
            Self::Shadow(s) => s.id,
        }
    }

    /// Output rect in logical pixels (bounding box; for a border, the outer rect).
    pub fn geometry(&self) -> Rectangle<f64, Logical> {
        match self {
            Self::Surface(s) => s.geometry,
            Self::SolidColor(s) => s.geometry,
            Self::Border(b) => b.geometry,
            Self::RoundedBox(r) => r.geometry,
            Self::Shadow(s) => s.geometry,
        }
    }

    /// Multiply this element's opacity by `factor`, in place — the window
    /// open/close fade. Surfaces carry opacity in their dedicated `alpha`;
    /// solids and borders carry it in the alpha channel of their (straight)
    /// BT.2020 colour. Both lower into the decode push's `tint.a`, so scaling
    /// either is the same premultiplied fade, and a now-translucent solid
    /// correctly stops being an occluder (see [`Self::push_opaque_regions`]).
    pub fn mul_alpha(&mut self, factor: f32) {
        match self {
            Self::Surface(s) => {
                s.alpha *= factor;
                // A now-translucent surface no longer fully occludes anything.
                if s.alpha < 1.0 {
                    s.opaque.clear();
                }
            }
            Self::SolidColor(s) => s.color_bt2020_nits[3] *= factor,
            Self::Border(b) => b.color_bt2020_nits[3] *= factor,
            Self::RoundedBox(r) => r.color_bt2020_nits[3] *= factor,
            Self::Shadow(s) => s.color_bt2020_nits[3] *= factor,
        }
    }

    /// Scale this element's geometry about `center` by `factor`, in place — the
    /// window open/close zoom. Purely visual: the layout geometry is unchanged,
    /// only the emitted destination rect moves, so input hit-testing and tiling
    /// stay at full size. Border stripe thicknesses scale too so the ring stays
    /// proportional to the shrunk window.
    pub fn scale_about(&mut self, center: Point<f64, Logical>, factor: f64) {
        fn scaled(
            g: Rectangle<f64, Logical>,
            center: Point<f64, Logical>,
            factor: f64,
        ) -> Rectangle<f64, Logical> {
            let loc = Point::from((
                center.x + (g.loc.x - center.x) * factor,
                center.y + (g.loc.y - center.y) * factor,
            ));
            let size = Size::from((g.size.w * factor, g.size.h * factor));
            Rectangle::new(loc, size)
        }
        fn scaled_clip(clip: SurfaceClip, center: Point<f64, Logical>, factor: f64) -> SurfaceClip {
            SurfaceClip {
                rect: scaled(clip.rect, center, factor),
                radii: clip.radii.map(|r| r * factor as f32),
            }
        }
        match self {
            Self::Surface(s) => {
                s.geometry = scaled(s.geometry, center, factor);
                // The clip box follows the same transform so the rounded
                // corners stay glued to the zoomed window.
                s.clip = s.clip.map(|c| scaled_clip(c, center, factor));
                // The opaque rects were computed against the un-scaled
                // geometry; once moved they no longer describe where the
                // surface is solid, so drop them rather than mis-cull.
                s.opaque.clear();
            }
            Self::SolidColor(s) => {
                s.geometry = scaled(s.geometry, center, factor);
                s.clip = s.clip.map(|c| scaled_clip(c, center, factor));
            }
            Self::Border(b) => {
                b.geometry = scaled(b.geometry, center, factor);
                b.thickness = b.thickness.map(|t| t * factor);
                b.clip = b.clip.map(|c| scaled(c, center, factor));
            }
            Self::RoundedBox(r) => {
                r.geometry = scaled(r.geometry, center, factor);
                r.radii = r.radii.map(|v| v * factor as f32);
                r.inset = r.inset.map(|inset| inset.map(|v| v * factor));
                r.clip = r.clip.map(|c| scaled(c, center, factor));
            }
            Self::Shadow(s) => {
                s.geometry = scaled(s.geometry, center, factor);
                s.shadow_box = scaled(s.shadow_box, center, factor);
                s.radii = s.radii.map(|v| v * factor as f32);
                s.sigma *= factor as f32;
                s.cutout = s.cutout.map(|c| scaled_clip(c, center, factor));
            }
        }
    }

    /// Translate this element's geometry by `offset`, in place — the
    /// overview's workspace-card placement (and the workspace-switch
    /// slide). Companion to [`Self::scale_about`]; unlike scaling, a pure
    /// translation keeps the opaque rects valid (they're absolute
    /// output-space), so they shift along instead of being cleared.
    pub fn translate(&mut self, offset: Point<f64, Logical>) {
        match self {
            Self::Surface(s) => {
                s.geometry.loc += offset;
                if let Some(c) = &mut s.clip {
                    c.rect.loc += offset;
                }
                for r in &mut s.opaque {
                    r.loc += offset;
                }
            }
            Self::SolidColor(s) => {
                s.geometry.loc += offset;
                if let Some(c) = &mut s.clip {
                    c.rect.loc += offset;
                }
            }
            Self::Border(b) => {
                b.geometry.loc += offset;
                if let Some(c) = &mut b.clip {
                    c.loc += offset;
                }
            }
            Self::RoundedBox(r) => {
                r.geometry.loc += offset;
                if let Some(c) = &mut r.clip {
                    c.loc += offset;
                }
            }
            Self::Shadow(s) => {
                s.geometry.loc += offset;
                s.shadow_box.loc += offset;
                if let Some(c) = &mut s.cutout {
                    c.rect.loc += offset;
                }
            }
        }
    }

    /// Restrict this element to `bounds` (output-space logical), in place —
    /// the overview's workspace-card crop (niri's `CropRenderElement`).
    /// Returns `false` when the element lies fully outside `bounds` and
    /// should be dropped. Surfaces and solids get their clip box
    /// intersected with `bounds` (an existing rounded clip keeps its radii;
    /// a corner cut by the card edge keeps its rounding — a sub-pixel
    /// divergence from niri's exact crop at overview zoom). Borders and
    /// rounded boxes get their rect crop intersected (stripes / the drawn
    /// quad stop at the band edge, the SDF shape itself is unchanged); a
    /// shadow's quad is its own field separate from the SDF box, so its
    /// geometry shrinks directly.
    pub fn crop_to_rect(&mut self, bounds: Rectangle<f64, Logical>) -> bool {
        let geometry = self.geometry();
        if geometry.intersection(bounds).is_none() {
            return false;
        }
        if bounds.contains_rect(geometry) {
            return true;
        }
        fn intersect_rect_clip(
            clip: &mut Option<Rectangle<f64, Logical>>,
            bounds: Rectangle<f64, Logical>,
        ) -> bool {
            match clip {
                Some(c) => match c.intersection(bounds) {
                    Some(r) => {
                        *c = r;
                        true
                    }
                    None => false,
                },
                None => {
                    *clip = Some(bounds);
                    true
                }
            }
        }
        match self {
            Self::Surface(s) => match &mut s.clip {
                Some(c) => match c.rect.intersection(bounds) {
                    Some(r) => c.rect = r,
                    None => return false,
                },
                None => {
                    s.clip = Some(SurfaceClip {
                        rect: bounds,
                        radii: [0.0; 4],
                    });
                }
            },
            Self::SolidColor(s) => match &mut s.clip {
                Some(c) => match c.rect.intersection(bounds) {
                    Some(r) => c.rect = r,
                    None => return false,
                },
                None => {
                    s.clip = Some(SurfaceClip {
                        rect: bounds,
                        radii: [0.0; 4],
                    });
                }
            },
            Self::Border(b) => {
                if !intersect_rect_clip(&mut b.clip, bounds) {
                    return false;
                }
            }
            Self::RoundedBox(r) => {
                if !intersect_rect_clip(&mut r.clip, bounds) {
                    return false;
                }
            }
            Self::Shadow(s) => {
                // Geometry ∩ bounds is non-empty (checked above); the SDF
                // parameters live in shadow_box/radii/sigma, so shrinking
                // the quad crops the blur exactly.
                if let Some(g) = s.geometry.intersection(bounds) {
                    s.geometry = g;
                }
            }
        }
        true
    }

    /// Apply a rounded-rect clip (`clip-to-geometry`) to this element, in
    /// place. Skipped when the clip provably wouldn't remove any pixels
    /// (element inside the box, clear of the corner squares — niri's
    /// `will_clip` test), so unclipped elements keep their full opaque
    /// regions and the cheap shader path. Surfaces and solids only; the
    /// other variants are decorations that are never part of a window's
    /// surface tree.
    pub fn clip_to_rounded_box(&mut self, clip: SurfaceClip) {
        match self {
            Self::Surface(s) => {
                if clip.would_clip(s.geometry) {
                    s.clip = Some(clip);
                }
            }
            Self::SolidColor(s) => {
                if clip.would_clip(s.geometry) {
                    s.clip = Some(clip);
                }
            }
            Self::Border(_) | Self::RoundedBox(_) | Self::Shadow(_) => {}
        }
    }

    /// Append this element's fully-opaque rects (absolute output-space logical)
    /// to `out`, for occlusion culling. An element drawn behind the union of
    /// these rects contributes nothing visible and can be skipped.
    ///
    /// - `Surface`: the buffer's opaque regions (already absolute; see
    ///   [`SurfaceEl::opaque`]). Empty for translucent / unknown-alpha buffers.
    ///   Clipped surfaces intersect them with the clip's corner-free bands.
    /// - `SolidColor`: the whole geometry iff the colour is fully opaque
    ///   (`alpha == 1.0`); a translucent fill occludes nothing. Clipped solids
    ///   intersect with the clip's bands like surfaces.
    /// - `Border`: nothing — a border is hollow stripes, so its bounding box is
    ///   mostly transparent and never a safe occluder.
    /// - `RoundedBox`: a ring occludes nothing; a fully-opaque fill occludes
    ///   the cross of two bands that excludes the corner squares (see
    ///   [`push_rounded_box_bands`]).
    pub fn push_opaque_regions(&self, out: &mut Vec<Rectangle<f64, Logical>>) {
        match self {
            Self::Surface(s) => match &s.clip {
                None => out.extend_from_slice(&s.opaque),
                // A clipped surface is only opaque where the buffer is opaque
                // AND the clip keeps full coverage — intersect with the clip
                // box's corner-free cross bands.
                Some(clip) => {
                    let mut bands = Vec::with_capacity(2);
                    push_rounded_box_bands(clip.rect, clip.radii, &mut bands);
                    for r in &s.opaque {
                        out.extend(bands.iter().filter_map(|b| r.intersection(*b)));
                    }
                }
            },
            Self::SolidColor(s) => {
                if s.color_bt2020_nits[3] < 1.0 {
                    return;
                }
                match &s.clip {
                    None => out.push(s.geometry),
                    Some(clip) => {
                        let mut bands = Vec::with_capacity(2);
                        push_rounded_box_bands(clip.rect, clip.radii, &mut bands);
                        out.extend(bands.iter().filter_map(|b| s.geometry.intersection(*b)));
                    }
                }
            }
            Self::Border(_) => {}
            Self::RoundedBox(r) => {
                if r.inset.is_some() || r.color_bt2020_nits[3] < 1.0 {
                    return;
                }
                match r.clip {
                    None => push_rounded_box_bands(r.geometry, r.radii, out),
                    // Cropped: only the part of the interior the quad
                    // actually rasterizes is opaque.
                    Some(clip) => {
                        let mut bands = Vec::with_capacity(2);
                        push_rounded_box_bands(r.geometry, r.radii, &mut bands);
                        out.extend(bands.iter().filter_map(|b| b.intersection(clip)));
                    }
                }
            }
            // A shadow is translucent everywhere — never an occluder.
            Self::Shadow(_) => {}
        }
    }

    /// Content fingerprint for the damage diff (see [`FrameElementMeta`]).
    /// Surfaces use their buffer commit count; solids/borders fingerprint their
    /// colour (and per-side thickness), so a focus-colour change re-damages the
    /// ring even though its geometry is unchanged.
    pub fn content_token(&self) -> u64 {
        match self {
            // A clip folds into the token relative to the element origin, so
            // a radius / relative-clip change re-damages in place while a
            // plain move stays a pure geometry diff. The element opacity folds
            // in too: a fade / wp_alpha_modifier change repaints in place even
            // though the buffer commit count is unchanged (a multiplier-only
            // commit attaches no buffer, so `content_commit` doesn't advance).
            // Same reasoning for the color params and the viewport source
            // rect: a commit changing only the color-management image
            // description or wp_viewport source attaches no buffer either,
            // yet changes every rendered pixel.
            Self::Surface(s) => {
                let c = (s.content_commit ^ s.alpha.to_bits() as u64).wrapping_mul(FNV_PRIME);
                let c = (c ^ fnv_f32s(&s.src_rect_uv)).wrapping_mul(FNV_PRIME);
                let c = (c ^ s.color.transfer as u64).wrapping_mul(FNV_PRIME);
                let c = (c ^ s.color.yuv_matrix as u64).wrapping_mul(FNV_PRIME);
                let c = c ^ fnv_f32s(&[s.color.sdr_white_nits]);
                let c = s
                    .color
                    .primaries_to_bt2020
                    .iter()
                    .fold(c, |h, row| h ^ fnv_f32s(row));
                match &s.clip {
                    None => c,
                    Some(clip) => clip.mix_token(c, s.geometry.loc),
                }
            }
            Self::SolidColor(s) => {
                let c = fnv_f32s(&s.color_bt2020_nits);
                match &s.clip {
                    None => c,
                    Some(clip) => clip.mix_token(c, s.geometry.loc),
                }
            }
            Self::Border(b) => {
                let c = fnv_f32s(&b.color_bt2020_nits);
                let c = b
                    .thickness
                    .iter()
                    .fold(c, |h, t| (h ^ t.to_bits()).wrapping_mul(FNV_PRIME));
                // Crop folded in relative to the element origin, like the
                // surface clip: a pure move stays a geometry-only diff, a
                // band edge sweeping across re-damages in place.
                mix_rect_clip(c, &b.clip, b.geometry.loc)
            }
            Self::RoundedBox(r) => {
                let c = fnv_f32s(&r.color_bt2020_nits);
                let c = fnv_f32s(&r.radii) ^ c;
                let c = r
                    .inset
                    .unwrap_or([-1.0; 4]) // distinct from any real (≥ 0) inset
                    .iter()
                    .fold(c, |h, t| (h ^ t.to_bits()).wrapping_mul(FNV_PRIME));
                mix_rect_clip(c, &r.clip, r.geometry.loc)
            }
            Self::Shadow(s) => {
                // Shadow box + cut-out fingerprint relative to the quad
                // origin, like the surface clip: moves stay pure geometry
                // diffs, radius/sigma/color changes re-damage in place.
                let origin = s.geometry.loc;
                let rel = [
                    (s.shadow_box.loc.x - origin.x) as f32,
                    (s.shadow_box.loc.y - origin.y) as f32,
                    s.shadow_box.size.w as f32,
                    s.shadow_box.size.h as f32,
                    s.sigma,
                ];
                let c = fnv_f32s(&s.color_bt2020_nits) ^ fnv_f32s(&s.radii) ^ fnv_f32s(&rel);
                match &s.cutout {
                    None => c,
                    Some(cutout) => cutout.mix_token(c, origin),
                }
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
            Self::RoundedBox(r) => out.push(r.to_draw(project, white_view, output_peak_nits_rgb)),
            Self::Shadow(s) => out.push(s.to_draw(project, white_view, output_peak_nits_rgb)),
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
            clip: None,
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
            alpha: 1.0,
            clip: None,
        })
    }

    fn border(geo: Rectangle<f64, Logical>) -> RenderEl {
        RenderEl::Border(BorderEl {
            id: ElementId::alloc(),
            geometry: geo,
            thickness: [2.0; 4],
            color_bt2020_nits: [10.0, 10.0, 10.0, 1.0],
            clip: None,
        })
    }

    fn rounded(geo: Rectangle<f64, Logical>, inset: Option<[f64; 4]>, alpha: f32) -> RenderEl {
        RenderEl::RoundedBox(RoundedBoxEl {
            id: ElementId::alloc(),
            geometry: geo,
            radii: [20.0; 4],
            inset,
            color_bt2020_nits: [10.0, 10.0, 10.0, alpha],
            clip: None,
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

    /// A filled opaque rounded box occludes its inner cross (between the
    /// corner squares, minus the AA margin) but not its corner regions.
    #[test]
    fn rounded_fill_occludes_center_not_corners() {
        let center = vec![
            solid(rect(30.0, 30.0, 40.0, 40.0), 1.0),
            rounded(rect(0.0, 0.0, 100.0, 100.0), None, 1.0),
        ];
        assert_eq!(cull_occluded(&center), vec![false, true]);

        let corner = vec![
            solid(rect(0.0, 0.0, 10.0, 10.0), 1.0),
            rounded(rect(0.0, 0.0, 100.0, 100.0), None, 1.0),
        ];
        assert_eq!(cull_occluded(&corner), vec![true, true]);
    }

    #[test]
    fn rounded_ring_never_occludes() {
        let els = vec![
            solid(rect(30.0, 30.0, 40.0, 40.0), 1.0),
            rounded(rect(0.0, 0.0, 100.0, 100.0), Some([2.0; 4]), 1.0),
        ];
        assert_eq!(cull_occluded(&els), vec![true, true]);
    }

    #[test]
    fn translucent_rounded_fill_occludes_nothing() {
        let els = vec![
            solid(rect(30.0, 30.0, 40.0, 40.0), 1.0),
            rounded(rect(0.0, 0.0, 100.0, 100.0), None, 0.5),
        ];
        assert_eq!(cull_occluded(&els), vec![true, true]);
    }

    /// Field mapping of the SDF push constants: box rect in logical px,
    /// fill vs ring mode, per-side inset.
    #[test]
    fn rounded_box_to_draw_push_fields() {
        let project = make_projector(Size::from((200.0, 100.0)));

        let RenderEl::RoundedBox(fill) = rounded(rect(10.0, 20.0, 80.0, 40.0), None, 1.0) else {
            unreachable!()
        };
        let draw = fill.to_draw(&project, vk::ImageView::null(), [1000.0; 3]);
        assert_eq!(draw.push.sdf_mode, 1);
        assert_eq!(draw.push.sdf_box, [10.0, 20.0, 90.0, 60.0]);
        assert_eq!(draw.push.sdf_radii, [20.0; 4]);

        let RenderEl::RoundedBox(ring) = rounded(
            rect(10.0, 20.0, 80.0, 40.0),
            Some([1.0, 2.0, 3.0, 4.0]),
            1.0,
        ) else {
            unreachable!()
        };
        let draw = ring.to_draw(&project, vk::ImageView::null(), [1000.0; 3]);
        assert_eq!(draw.push.sdf_mode, 2);
        assert_eq!(draw.push.sdf_inset, [1.0, 2.0, 3.0, 4.0]);
    }

    fn clip(x: f64, y: f64, w: f64, h: f64, radius: f32) -> SurfaceClip {
        SurfaceClip {
            rect: rect(x, y, w, h),
            radii: [radius; 4],
        }
    }

    fn shadow(quad: Rectangle<f64, Logical>, cutout: Option<SurfaceClip>) -> RenderEl {
        RenderEl::Shadow(ShadowEl {
            id: ElementId::alloc(),
            geometry: quad,
            shadow_box: rect(
                quad.loc.x + 45.0,
                quad.loc.y + 45.0,
                quad.size.w - 90.0,
                quad.size.h - 90.0,
            ),
            radii: [16.0; 4],
            sigma: 15.0,
            cutout,
            color_bt2020_nits: [0.0, 0.0, 0.0, 0.47],
        })
    }

    /// A shadow never occludes, even at alpha 1 with no cut-out.
    #[test]
    fn shadow_never_occludes() {
        let els = vec![
            solid(rect(30.0, 30.0, 40.0, 40.0), 1.0),
            shadow(rect(0.0, 0.0, 200.0, 200.0), None),
        ];
        assert_eq!(cull_occluded(&els), vec![true, true]);
    }

    /// Shadow push-constant mapping: mode 3, shadow box, sigma, cut-out.
    #[test]
    fn shadow_to_draw_push_fields() {
        let project = make_projector(Size::from((400.0, 300.0)));
        let RenderEl::Shadow(s) = shadow(
            rect(0.0, 0.0, 200.0, 200.0),
            Some(clip(50.0, 50.0, 100.0, 100.0, 12.0)),
        ) else {
            unreachable!()
        };
        let draw = s.to_draw(&project, vk::ImageView::null(), [1000.0; 3]);
        assert_eq!(draw.push.sdf_mode, 3);
        assert_eq!(draw.push.sdf_box, [45.0, 45.0, 155.0, 155.0]);
        assert_eq!(draw.push.sdf_radii, [16.0; 4]);
        assert_eq!(draw.push.sdf_sigma, 15.0);
        assert_eq!(draw.push.sdf_box2, [50.0, 50.0, 150.0, 150.0]);
        assert_eq!(draw.push.sdf_radii2, [12.0; 4]);

        // Without a cut-out the box2 stays empty (max == min == 0 disables it).
        let RenderEl::Shadow(s) = shadow(rect(0.0, 0.0, 200.0, 200.0), None) else {
            unreachable!()
        };
        let draw = s.to_draw(&project, vk::ImageView::null(), [1000.0; 3]);
        assert_eq!(draw.push.sdf_box2, [0.0; 4]);
    }

    /// Shadow content token is origin-relative (moves are pure geometry
    /// diffs) and sensitive to sigma.
    #[test]
    fn shadow_token_origin_relative_and_sigma_sensitive() {
        let at = |loc: f64, sigma: f32| {
            let RenderEl::Shadow(mut s) = shadow(
                rect(loc, loc, 200.0, 200.0),
                Some(clip(loc + 50.0, loc + 50.0, 100.0, 100.0, 12.0)),
            ) else {
                unreachable!()
            };
            s.sigma = sigma;
            RenderEl::Shadow(s).content_token()
        };
        assert_eq!(at(0.0, 15.0), at(70.0, 15.0));
        assert_ne!(at(0.0, 15.0), at(0.0, 20.0));
    }

    /// `would_clip` (niri's `will_clip`): false only for elements inside the
    /// box and clear of every corner square.
    #[test]
    fn would_clip_semantics() {
        let c = clip(0.0, 0.0, 100.0, 100.0, 12.0);
        // Interior, clear of all corners.
        assert!(!c.would_clip(rect(20.0, 20.0, 60.0, 60.0)));
        // Inside the box but overlapping the top-left corner square.
        assert!(c.would_clip(rect(5.0, 5.0, 20.0, 20.0)));
        // Sticking out of the box.
        assert!(c.would_clip(rect(50.0, 50.0, 60.0, 20.0)));
        // Radius 0: anything inside the box is clear.
        let sharp = clip(0.0, 0.0, 100.0, 100.0, 0.0);
        assert!(!sharp.would_clip(rect(0.0, 0.0, 100.0, 100.0)));
        assert!(sharp.would_clip(rect(-1.0, 0.0, 100.0, 100.0)));
    }

    /// `clip_to_rounded_box` skips elements the clip can't affect, keeping
    /// their opaque regions intact; clipped elements shrink theirs to the
    /// corner-free cross.
    #[test]
    fn clip_shrinks_opaque_regions() {
        let full = rect(0.0, 0.0, 100.0, 100.0);
        let c = clip(0.0, 0.0, 100.0, 100.0, 20.0);

        let mut clipped = surface(full, vec![full]);
        clipped.clip_to_rounded_box(c);
        let mut regions = Vec::new();
        clipped.push_opaque_regions(&mut regions);
        // Corner square no longer claimed opaque…
        assert!(regions
            .iter()
            .all(|r| r.intersection(rect(0.0, 0.0, 10.0, 10.0)).is_none()));
        // …but the center still is.
        assert!(regions
            .iter()
            .any(|r| r.contains_rect(rect(40.0, 40.0, 20.0, 20.0))));

        let mut skipped = surface(
            rect(30.0, 30.0, 40.0, 40.0),
            vec![rect(30.0, 30.0, 40.0, 40.0)],
        );
        skipped.clip_to_rounded_box(c);
        let mut regions = Vec::new();
        skipped.push_opaque_regions(&mut regions);
        assert_eq!(regions, vec![rect(30.0, 30.0, 40.0, 40.0)]);
    }

    /// Clipped surfaces lower with fill-mode SDF push constants.
    #[test]
    fn clipped_surface_to_draw_push_fields() {
        let project = make_projector(Size::from((200.0, 100.0)));
        let RenderEl::Surface(mut s) = surface(rect(10.0, 20.0, 80.0, 40.0), vec![]) else {
            unreachable!()
        };
        s.clip = Some(clip(12.0, 22.0, 60.0, 30.0, 8.0));
        let draw = s.to_draw(&project, [1000.0; 3]);
        assert_eq!(draw.push.sdf_mode, 1);
        assert_eq!(draw.push.sdf_box, [12.0, 22.0, 72.0, 52.0]);
        assert_eq!(draw.push.sdf_radii, [8.0; 4]);
    }

    /// The clip folds into the content token relative to the element origin:
    /// moving box+element together keeps the token; changing the radius (or
    /// the clip relative to the element) changes it.
    #[test]
    fn clip_token_is_origin_relative() {
        let mk = |loc: f64, radius: f32| {
            let RenderEl::Surface(mut s) = surface(rect(loc, loc, 50.0, 50.0), vec![]) else {
                unreachable!()
            };
            s.clip = Some(clip(loc, loc, 50.0, 50.0, radius));
            RenderEl::Surface(s).content_token()
        };
        assert_eq!(mk(0.0, 8.0), mk(30.0, 8.0));
        assert_ne!(mk(0.0, 8.0), mk(0.0, 12.0));
    }

    /// Element opacity folds into the surface content token: an alpha-only
    /// change (fade frame, wp_alpha_modifier commit) re-damages in place.
    #[test]
    fn surface_token_alpha_sensitive() {
        let mk = |alpha: f32| {
            let RenderEl::Surface(mut s) = surface(rect(0.0, 0.0, 50.0, 50.0), vec![]) else {
                unreachable!()
            };
            s.alpha = alpha;
            RenderEl::Surface(s).content_token()
        };
        assert_eq!(mk(1.0), mk(1.0));
        assert_ne!(mk(1.0), mk(0.5));
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

    /// A border straddling the crop edge picks up the rect clip (so its
    /// stripes stop at the band) instead of being kept whole; one fully
    /// outside is dropped.
    #[test]
    fn crop_clips_straddling_border() {
        let mut el = border(rect(40.0, 40.0, 100.0, 100.0));
        assert!(el.crop_to_rect(rect(0.0, 0.0, 100.0, 100.0)));
        let RenderEl::Border(b) = &el else {
            unreachable!()
        };
        assert_eq!(b.clip, Some(rect(0.0, 0.0, 100.0, 100.0)));
        // Full geometry retained — the crop is the clip, not a reshape.
        assert_eq!(b.geometry, rect(40.0, 40.0, 100.0, 100.0));

        let mut outside = border(rect(200.0, 200.0, 10.0, 10.0));
        assert!(!outside.crop_to_rect(rect(0.0, 0.0, 100.0, 100.0)));
    }

    /// Cropping a rounded box keeps its SDF geometry (the shape is cut at
    /// the band edge, not reshaped) and shrinks its opaque bands to the
    /// clipped region only.
    #[test]
    fn crop_clips_rounded_box_and_its_opaque_bands() {
        let mut el = rounded(rect(40.0, 0.0, 100.0, 100.0), None, 1.0);
        assert!(el.crop_to_rect(rect(0.0, 0.0, 100.0, 100.0)));
        let RenderEl::RoundedBox(r) = &el else {
            unreachable!()
        };
        assert_eq!(r.geometry, rect(40.0, 0.0, 100.0, 100.0));
        assert_eq!(r.clip, Some(rect(0.0, 0.0, 100.0, 100.0)));

        let mut opaque = Vec::new();
        el.push_opaque_regions(&mut opaque);
        for region in &opaque {
            assert!(
                region.loc.x + region.size.w <= 100.0,
                "opaque region {region:?} extends past the crop edge"
            );
        }
    }

    /// Cropping a shadow shrinks the rasterized quad directly — its SDF
    /// box is the separate `shadow_box` field, which must stay put so the
    /// blur is cut at the band edge rather than reshaped. (The shrink
    /// itself is caught by the damage diff's geometry compare; the content
    /// token only needs to track the quad↔shadow_box *offset*, covered
    /// below.)
    #[test]
    fn crop_shrinks_shadow_quad_not_its_sdf_box() {
        let mk = || ShadowEl {
            id: ElementId::alloc(),
            geometry: rect(40.0, 0.0, 120.0, 120.0),
            shadow_box: rect(50.0, 10.0, 100.0, 100.0),
            radii: [8.0; 4],
            sigma: 5.0,
            cutout: None,
            color_bt2020_nits: [0.0, 0.0, 0.0, 0.7],
        };
        let mut el = RenderEl::Shadow(mk());
        assert!(el.crop_to_rect(rect(0.0, 0.0, 100.0, 100.0)));
        let RenderEl::Shadow(s) = &el else {
            unreachable!()
        };
        assert_eq!(s.geometry, rect(40.0, 0.0, 60.0, 100.0));
        assert_eq!(s.shadow_box, rect(50.0, 10.0, 100.0, 100.0));

        // A quad pinned in place while the casting box slides under it
        // (band-edge crop during a card scroll) must re-damage via the
        // token — the meta geometry is unchanged in that case.
        let mut slid = mk();
        slid.shadow_box.loc.x += 10.0;
        assert_ne!(
            RenderEl::Shadow(slid).content_token(),
            RenderEl::Shadow(mk()).content_token()
        );
    }
}
