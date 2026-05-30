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
// source's 1.0 maps to; for PQ (already absolute after its EOTF) it is 1.0
// (pass-through — PQ is not anchored).

#version 450

// binding 0 = primary texture (RGB), or the luma plane for YUV (yuv != 0).
layout(set = 0, binding = 0) uniform sampler2D u_texture;
// binding 1 = chroma plane for YUV (half-res, interleaved Cb/Cr). For RGB
// draws this is bound to the same view as binding 0 and never sampled.
layout(set = 0, binding = 1) uniform sampler2D u_chroma;

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
} push;

layout(location = 0) in vec2 v_uv;
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
    // maps to; PQ → 1.0 (its EOTF is already absolute and PQ is not anchored).
    // Uniform across transfers so the intent policy lives entirely on the CPU.
    linear *= push.sdr_white_nits;

    // Primaries → BT.2020. mat4 storage; the 3×3 lives in the upper-left.
    mat3 m = mat3(push.decode_matrix);
    vec3 bt2020 = m * linear;

    // Tint (multiplicative; identity = vec4(1.0)). Applied to color in
    // linear-light + alpha — lets solid-color elements drive arbitrary
    // hues through the white-texture path.
    bt2020 *= push.tint.rgb;
    float alpha = sampled.a * push.tint.a;

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
