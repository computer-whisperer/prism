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
// All transfers other than PQ scale the linear result by sdr_white_nits
// to get into absolute-nits domain.

#version 450

layout(set = 0, binding = 0) uniform sampler2D u_texture;

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
    // Per-output panel luminance ceiling, in nits. The intermediate
    // is display-referred: post-decode values are clamped to this
    // peak so downstream compositing operates entirely within the
    // panel's realizable range, and the encoder is responsible only
    // for emitting what the intermediate holds (no further clamping
    // needed). Set per output from the HDR max_luminance config or
    // sdr_reference_nits for SDR outputs.
    float output_peak_nits;
    int _pad1;
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

void main() {
    vec4 sampled = texture(u_texture, v_uv);

    // Decode transfer → linear-light. Linear path leaves alpha-unmultiplied
    // values where they are.
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

    // Scale into absolute-nits domain. For PQ the EOTF already produced
    // absolute nits; everything else interprets the source's 1.0 as
    // `sdr_white_nits`.
    if (push.transfer != 2) {
        linear *= push.sdr_white_nits;
    }

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
    // Per-channel clamp is fine for our additive-channel panels;
    // chromaticity preservation would warrant a luminance-aware
    // tone-map at this stage, which we don't have yet (deferred —
    // see docs/phase-2-scanout-followups.md).
    bt2020 = clamp(bt2020, vec3(0.0), vec3(push.output_peak_nits));

    // Output to fp16 intermediate. Alpha is passed through unchanged so
    // standard pre-multiplied blending composes correctly in linear space.
    out_color = vec4(bt2020 * alpha, alpha);
}
