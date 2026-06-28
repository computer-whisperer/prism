// Per-element decode pass: fragment shader.
//
// Pipeline stage 1 of 2. Inputs one element's texture (any transfer/primaries).
// Outputs BT.2020 linear absolute nits into the fp16 intermediate.
//
// Transfer enum (kept in sync with prism_renderer::SurfaceColorParams):
//   0 = Linear (no-op, assume input is already linear-light)
//   1 = sRGB piecewise EOTF (IEC 61966-2-1)
//   2 = PQ (SMPTE ST 2084) EOTF — output is absolute nits
//   3 = HLG (BT.2100) — TODO not implemented yet
//   4 = Gamma 2.2 (BT.470 / NTSC display assumption)
//   5 = BT.1886 (display gamma 2.4 with defaults; pure pow for now —
//       precise BT.1886 has Lb/Lw black-lift terms but most content
//       authors expect the pure-pow degenerate case)
//
// Every transfer scales the linear result by sdr_white_nits to reach the
// anchored absolute-nits working space. The CPU folds the per-transfer EOTF
// convention and the render intent into that one scalar (see
// description_to_params / decode_luminance_scale): for non-PQ it is the nits the
// source's 1.0 maps to; for PQ (already absolute after its EOTF) it is the
// anchoring ratio output-ref-white / content-ref-white (1.0 for absolute).

#version 450

// binding 0 = primary texture (RGB), or the luma plane for YUV (yuv != 0).
layout(set = 0, binding = 0) uniform sampler2D u_texture;
// binding 1 = chroma plane for YUV (half-res, interleaved Cb/Cr). For RGB
// draws this is bound to the same view as binding 0 and never sampled.
layout(set = 0, binding = 1) uniform sampler2D u_chroma;
// binding 2 = deband pre-blurred copy of the source (fp16, possibly at a
// lower resolution when downsampled — the LINEAR sampler upsamples it),
// sampled with the same v_uv as binding 0. Used only when `deband != 0`;
// otherwise bound to the binding-0 view and never sampled.
layout(set = 0, binding = 2) uniform sampler2D u_deband;

layout(push_constant) uniform Push {
    vec4 dst_rect_clip;
    vec4 src_rect_uv;
    mat4 decode_matrix;
    // Per-element tint, applied after decode + primaries conversion but before
    // alpha premultiplication. Identity = vec4(1.0). Used by solid-color
    // elements (window borders, layout backgrounds) which sample the
    // renderer's 1×1 white texture with transfer=Linear and have the actual
    // color baked into this tint in BT.2020 linear nits.
    vec4 tint;
    float sdr_white_nits;
    int transfer;
    // YUV plane layout: 0 = RGB (sample u_texture directly), 1 = NV12 (8-bit),
    // 2 = P010 (10-bit in the high bits of 16). Non-zero → sample luma+chroma
    // and convert to nonlinear R'G'B' before the transfer decode below.
    int yuv;
    // YUV→RGB coefficients: 0 = BT.709, 1 = BT.2020. Ignored when yuv == 0.
    int yuv_matrix;
    // Per-output, per-channel panel luminance ceiling, in nits. The
    // intermediate is display-referred: post-decode values are clamped
    // to this peak so downstream compositing operates entirely within
    // the panel's realizable range, and the encoder is responsible
    // only for emitting what the intermediate holds. Per-channel
    // because subpixel peaks differ on real panels — OLED ABL allocates
    // power per subpixel, and LCD color-filter transmission varies per
    // primary. `.a` is unused (vec4 to avoid std430 vec3 alignment
    // overhead). Set per output by the calibration pipeline; default
    // broadcasts hdr.max_luminance / sdr_reference_nits to all three.
    vec4 output_peak_nits_rgba;
    // Sampled-alpha handling (mirrors prism_renderer::AlphaMode):
    //   0 = opaque        — ignore the sampled alpha, force it to 1.0. For
    //                       X-formats (the X byte is undefined per the DRM /
    //                       wl_shm contract) and YUV video.
    //   1 = premultiplied — the Wayland wl_surface contract for A-formats. The
    //                       color is already multiplied by alpha, so we
    //                       un-premultiply before the transfer EOTF (which must
    //                       run on straight color) and re-premultiply at output.
    // Trailing scalar after the vec4, so no std430 padding is needed.
    int alpha_mode;
    // Rounded-corner SDF coverage (see sdf_coverage below):
    //   0 = off  — sdf_* fields ignored.
    //   1 = fill — multiply alpha by the coverage of the rounded box
    //              (sdf_box, sdf_radii). Filled rounded rects and, later,
    //              window-surface corner clipping.
    //   2 = ring — multiply alpha by (outer coverage − inner coverage):
    //              a hollow border band of per-side thickness sdf_inset
    //              inside sdf_box. Inner corner radii are derived as
    //              max(outer − adjacent insets, 0), matching niri.
    int sdf_mode;
    // Logical size of the output view; lets the vertex shader recover the
    // fragment's position in logical pixels from clip space (v_pos_log).
    vec2 view_size_log;
    // Rounded box in output-space logical pixels: x_min, y_min, x_max, y_max.
    vec4 sdf_box;
    // Per-corner radii in logical pixels: top-left, top-right, bottom-right,
    // bottom-left (clockwise from top-left; matches prism-config CornerRadius).
    vec4 sdf_radii;
    // Ring mode only: per-side band thickness in logical pixels,
    // top, right, bottom, left (CSS order; matches BorderEl::thickness).
    vec4 sdf_inset;
    // Shadow mode only: cut-out rounded box (the window rect — the shadow
    // is not drawn behind it), logical px min/max. Empty (max <= min)
    // disables the cut-out (`draw-behind-window true`).
    vec4 sdf_box2;
    // Shadow mode only: per-corner radii of the cut-out box.
    vec4 sdf_radii2;
    // Shadow mode only: Gaussian sigma in logical px (CSS box-shadow
    // convention: softness / 2). Below 0.1 the shadow degenerates to a
    // crisp rounded rect, matching niri.
    float sdf_sigma;
    // Deband: 0 = off, 1 = clamp the binding-2 pre-blurred source to
    // ±0.5 LSB of the original code before the transfer decode.
    int deband;
} push;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec2 v_pos_log;
layout(location = 0) out vec4 out_color;

// sRGB inverse EOTF (encoded → linear), per IEC 61966-2-1.
float srgb_eotf_component(float c) {
    return c <= 0.04045
        ? c / 12.92
        : pow((c + 0.055) / 1.055, 2.4);
}
vec3 srgb_eotf(vec3 c) {
    return vec3(srgb_eotf_component(c.r),
                srgb_eotf_component(c.g),
                srgb_eotf_component(c.b));
}

// PQ inverse EOTF (encoded V in [0,1] → linear Y in [0,1] scaled to 10000 nits).
// SMPTE ST 2084. TODO: implement and unit-test against anchor points
// (V=0.5 ≈ 92.25 nits; V=1.0 = 10000 nits).
vec3 pq_eotf(vec3 v) {
    const float m1 = 0.1593017578125;     // 2610/16384
    const float m2 = 78.84375;            // 2523/4096 * 128
    const float c1 = 0.8359375;           // 3424/4096
    const float c2 = 18.8515625;          // 2413/4096 * 32
    const float c3 = 18.6875;             // 2392/4096 * 32
    vec3 vm = pow(max(v, vec3(0.0)), vec3(1.0 / m2));
    vec3 num = max(vm - c1, vec3(0.0));
    vec3 den = c2 - c3 * vm;
    vec3 y = pow(num / den, vec3(1.0 / m1));
    return y * 10000.0;
}

// Signed distance to a rounded box centered at the origin. `p` is the sample
// position relative to the box center, `b` the half-size, `r` the per-corner
// radii (top-left, top-right, bottom-right, bottom-left). Quadrant-aware
// radius selection, then the classic Inigo Quilez rounded-box SDF. The caller
// clamps each radius to the half shorter side, so the field never degenerates.
// (Ported from damascene's rounded_rect.wgsl.)
float sdf_rounded_box(vec2 p, vec2 b, vec4 r) {
    float r_top = p.x > 0.0 ? r.y : r.x; // tr : tl
    float r_bot = p.x > 0.0 ? r.z : r.w; // br : bl
    float rd = p.y < 0.0 ? r_top : r_bot;
    vec2 q = abs(p) - b + vec2(rd);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - rd;
}

// Pixel coverage of the rounded box `box` (logical px, min/max corners) with
// per-corner radii `radii`, sampled at `pos` (logical px).
//
// Anti-aliasing: box-filter estimate `0.5 − d/aa` over the SDF, with `aa` the
// L2 norm of the screen-space SDF gradient — NOT fwidth's L1 norm, which is
// √2 wider at 45° and visibly fattens curved corners (damascene's lesson).
// Since `d` is in logical px and derivatives are per *physical* pixel, the
// gradient magnitude is exactly one physical pixel expressed in logical units,
// so the AA band is one physical pixel at any output scale. The box-filter
// form (rather than smoothstep) makes pixel-aligned straight edges resolve to
// exact 0/1 coverage — a radius-0 box rasterizes crisp, identical to a plain
// quad.
float sdf_coverage(vec2 pos, vec4 box, vec4 radii) {
    vec2 half_size = max((box.zw - box.xy) * 0.5, vec2(0.0));
    vec2 center = (box.xy + box.zw) * 0.5;
    float max_r = min(half_size.x, half_size.y);
    vec4 r = clamp(radii, vec4(0.0), vec4(max_r));
    float d = sdf_rounded_box(pos - center, half_size, r);
    float aa = max(length(vec2(dFdx(d), dFdy(d))), 1e-4);
    return clamp(0.5 - d / aa, 0.0, 1.0);
}

// ── Gaussian-blurred rounded-box shadow ────────────────────────────────
// Analytic-in-X, 4-sample-numeric-in-Y approximation of a rounded box
// convolved with a Gaussian. Ported from niri's shadow.frag, which is
// based on Evan Wallace's "fast rounded rectangle shadows"
// (https://madebyevan.com/shaders/fast-rounded-rectangle-shadows/, CC0).

// A standard gaussian function, used for weighting samples.
float gaussian(float x, float sigma) {
    const float pi = 3.141592653589793;
    return exp(-(x * x) / (2.0 * sigma * sigma)) / (sqrt(2.0 * pi) * sigma);
}

// This approximates the error function, needed for the gaussian integral.
vec2 erf(vec2 x) {
    vec2 s = sign(x), a = abs(x);
    x = 1.0 + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
    x *= x;
    return s - s / (x * x);
}

// The blurred mask along the x dimension, at height offset `y` from the
// box center.
float rounded_box_shadow_x(float x, float y, float sigma, float corner, vec2 half_size) {
    float delta = min(half_size.y - corner - abs(y), 0.0);
    float curved = half_size.x - corner + sqrt(max(0.0, corner * corner - delta * delta));
    vec2 integral = 0.5 + 0.5 * erf((x + vec2(-curved, curved)) * (sqrt(0.5) / sigma));
    return integral.y - integral.x;
}

// The shadow mask for the rounded box `box` (logical px min/max corners,
// uniform corner radius `corner`) blurred by `sigma`, sampled at `pos`.
float rounded_box_shadow(vec4 box, vec2 pos, float sigma, float corner) {
    // Center everything to make the math easier.
    vec2 center = (box.xy + box.zw) * 0.5;
    vec2 half_size = (box.zw - box.xy) * 0.5;
    pos -= center;

    // The signal is only non-zero in a limited range, so don't waste samples.
    float low = pos.y - half_size.y;
    float high = pos.y + half_size.y;
    float start = clamp(-3.0 * sigma, low, high);
    float end = clamp(3.0 * sigma, low, high);

    // Accumulate samples (we can get away with surprisingly few).
    float step = (end - start) / 4.0;
    float y = start + step * 0.5;
    float value = 0.0;
    for (int i = 0; i < 4; i++) {
        value += rounded_box_shadow_x(pos.x, pos.y - y, sigma, corner, half_size)
            * gaussian(y, sigma) * step;
        y += step;
    }

    return value;
}

// Recover nonlinear R'G'B' from a limited-range Y'CbCr buffer. Luma is in
// u_texture.r, interleaved chroma in u_chroma.rg (bilinearly upsampled by the
// LINEAR sampler). Range-expansion constants are in the *sampled* domain:
// UNORM8 = code/255 for NV12; UNORM16 = (code10 << 6)/65535 for P010, so its
// limited-range endpoints scale by 64. Coefficients are the BT.709 / BT.2020
// non-constant-luminance inverse matrices for full-range Y in [0,1],
// Cb/Cr in [-0.5, 0.5]. The result is still transfer-encoded (PQ/gamma) —
// main()'s transfer decode runs on it afterwards.
vec3 ycbcr_to_nonlinear_rgb() {
    float y_black, y_scale, c_mid, c_scale;
    if (push.yuv == 2) {
        // P010: 10-bit limited (Y 64..940, C 64..960) in the high bits of 16.
        y_black = 4096.0 / 65535.0;          // 64 << 6
        y_scale = 65535.0 / 56064.0;         // 1 / ((940-64) << 6)
        c_mid   = 32768.0 / 65535.0;         // 512 << 6
        c_scale = 65535.0 / 57344.0;         // 1 / ((960-64) << 6)
    } else {
        // NV12: 8-bit limited (Y 16..235, C 16..240).
        y_black = 16.0 / 255.0;
        y_scale = 255.0 / 219.0;
        c_mid   = 128.0 / 255.0;
        c_scale = 255.0 / 224.0;
    }

    float Y  = (texture(u_texture, v_uv).r - y_black) * y_scale;
    vec2  C  = texture(u_chroma, v_uv).rg;
    float Cb = (C.x - c_mid) * c_scale;
    float Cr = (C.y - c_mid) * c_scale;

    if (push.yuv_matrix == 1) {
        // BT.2020 NCL (Kr=0.2627, Kb=0.0593).
        return vec3(
            Y + 1.4746 * Cr,
            Y - 0.16455 * Cb - 0.57135 * Cr,
            Y + 1.8814 * Cb);
    }
    // BT.709 (Kr=0.2126, Kb=0.0722) — also used for sRGB-primaries SDR video.
    return vec3(
        Y + 1.5748 * Cr,
        Y - 0.1873 * Cb - 0.4681 * Cr,
        Y + 1.8556 * Cb);
}

void main() {
    // YUV surfaces reconstruct nonlinear R'G'B' (alpha = opaque); RGB surfaces
    // sample straight. Either way `sampled` is transfer-encoded RGBA from here.
    vec4 sampled = push.yuv != 0
        ? vec4(ycbcr_to_nonlinear_rgb(), 1.0)
        : texture(u_texture, v_uv);

    // Debanding (8-bit SDR RGB only — the integrator gates `deband` on
    // that). Replace the source codes with the wide-Gaussian-blurred copy
    // at binding 2, clamped to ±0.5 LSB of the original so re-rounding to
    // 8-bit is unchanged: the value the panel sees is never wrong, but
    // smooth gradients gain sub-LSB precision the panel can resolve. Runs
    // before alpha/transfer so the rest of the pipeline is untouched.
    if (push.deband != 0) {
        const float lsb = 0.5 / 255.0;
        vec3 blurred = texture(u_deband, v_uv).rgb;
        sampled.rgb = clamp(blurred, sampled.rgb - lsb, sampled.rgb + lsb);
    }

    // Alpha handling — BEFORE the transfer decode, which must operate on
    // straight (non-premultiplied) color. Opaque buffers (X-formats, YUV)
    // carry no meaningful alpha, so force it to 1.0 rather than trust the
    // undefined X byte. Premultiplied buffers (the Wayland A-format contract)
    // get un-premultiplied here; the `* alpha` at output re-premultiplies for
    // the pipeline's premultiplied src-over blend. The solid-color path samples
    // an opaque white texel, so it lands in either branch as a no-op and keeps
    // its color/opacity in `tint`.
    if (push.alpha_mode == 0) {
        sampled.a = 1.0;
    } else if (sampled.a > 0.0) {
        sampled.rgb /= sampled.a;
    }

    // Decode transfer → linear-light. Linear path leaves the (now straight)
    // color where it is.
    vec3 linear;
    if (push.transfer == 1) {
        linear = srgb_eotf(sampled.rgb);
    } else if (push.transfer == 2) {
        linear = pq_eotf(sampled.rgb);
    } else if (push.transfer == 4) {
        // Gamma 2.2. `pow` on the negative half is undefined; clamp.
        linear = pow(max(sampled.rgb, vec3(0.0)), vec3(2.2));
    } else if (push.transfer == 5) {
        // BT.1886 with defaults — degenerate pure-pow 2.4.
        linear = pow(max(sampled.rgb, vec3(0.0)), vec3(2.4));
    } else {
        // Linear (or unhandled transfer → identity for now).
        linear = sampled.rgb;
    }

    // Scale into the anchored absolute-nits working space. `sdr_white_nits` is
    // the post-EOTF multiplier the CPU computed from the surface's transfer and
    // render intent (decode_luminance_scale): non-PQ → the nits the source's 1.0
    // maps to; PQ → the anchoring ratio (1.0 for the absolute intent, i.e. the
    // former pass-through). Uniform across transfers so the intent policy lives
    // entirely on the CPU.
    linear *= push.sdr_white_nits;

    // Primaries → BT.2020. mat4 storage; the 3×3 lives in the upper-left.
    mat3 m = mat3(push.decode_matrix);
    vec3 bt2020 = m * linear;

    // Tint (multiplicative; identity = vec4(1.0)). Applied to color in
    // linear-light + alpha — lets solid-color elements drive arbitrary
    // hues through the white-texture path.
    bt2020 *= push.tint.rgb;
    float alpha = sampled.a * push.tint.a;

    // Rounded-corner SDF coverage folds into alpha; the premultiplied output
    // below then scales color by it, so the AA edge blends correctly in
    // linear light. The branch is on a push constant — uniform across the
    // draw — so the derivatives inside sdf_coverage are well-defined.
    if (push.sdf_mode == 1) {
        alpha *= sdf_coverage(v_pos_log, push.sdf_box, push.sdf_radii);
    } else if (push.sdf_mode == 2) {
        float outer = sdf_coverage(v_pos_log, push.sdf_box, push.sdf_radii);
        // Inner box: outer inset per side (top, right, bottom, left).
        vec4 inset = push.sdf_inset;
        vec4 inner_box = vec4(
            push.sdf_box.x + inset.w,
            push.sdf_box.y + inset.x,
            push.sdf_box.z - inset.y,
            push.sdf_box.w - inset.z);
        // Inner corner radius = outer minus the larger adjacent inset,
        // floored at zero (niri: max(outer_radius - border_width, 0)).
        vec4 inner_radii = max(
            push.sdf_radii - vec4(
                max(inset.x, inset.w),  // tl: top, left
                max(inset.x, inset.y),  // tr: top, right
                max(inset.z, inset.y),  // br: bottom, right
                max(inset.z, inset.w)), // bl: bottom, left
            vec4(0.0));
        float inner = inner_box.z > inner_box.x && inner_box.w > inner_box.y
            ? sdf_coverage(v_pos_log, inner_box, inner_radii)
            : 0.0;
        alpha *= clamp(outer - inner, 0.0, 1.0);
    } else if (push.sdf_mode == 3) {
        // Drop shadow: Gaussian-blurred rounded box, with the window
        // region optionally cut out (draw-behind-window false). The blur
        // uses a single corner radius (tl) — same limitation as niri;
        // per-corner blurring would need GTK's split approach.
        float v;
        if (push.sdf_sigma < 0.1) {
            // Low sigma degenerates to a crisp rounded rect.
            v = sdf_coverage(v_pos_log, push.sdf_box, push.sdf_radii);
        } else {
            v = rounded_box_shadow(push.sdf_box, v_pos_log, push.sdf_sigma, push.sdf_radii.x);
        }
        if (push.sdf_box2.z > push.sdf_box2.x && push.sdf_box2.w > push.sdf_box2.y) {
            v *= 1.0 - sdf_coverage(v_pos_log, push.sdf_box2, push.sdf_radii2);
        }
        alpha *= clamp(v, 0.0, 1.0);
    }

    // Display-referred clamp: the intermediate represents what the
    // panel can actually emit. Values authored beyond panel peak
    // (HDR content from clients that haven't tone-mapped to the
    // output, or sRGB content with an unusually high sdr_white_nits)
    // hard-clip here so the rest of the pipeline operates in-range.
    // Per-channel: a buffer carrying (1000, 0, 0) nits red is clipped
    // by red's ceiling, not green's — under-clipping bright pure
    // colors against an all-channel scalar would lose chromaticity
    // we could otherwise preserve. Chromaticity-preserving tone-map
    // would warrant a luminance-aware stage, which we don't have yet
    // (deferred — see docs/phase-2-scanout-followups.md).
    bt2020 = clamp(bt2020, vec3(0.0), push.output_peak_nits_rgba.rgb);

    // Output to fp16 intermediate. Alpha is passed through unchanged so
    // standard pre-multiplied blending composes correctly in linear space.
    out_color = vec4(bt2020 * alpha, alpha);
}
