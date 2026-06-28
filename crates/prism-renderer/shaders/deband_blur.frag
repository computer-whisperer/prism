// One axis of a separable Gaussian blur, in source *code* space.
//
// Stage 1 of the deband pre-pass: smooth the 8-bit source codes (sampled
// as normalized [0,1]) with a wide Gaussian so smooth gradients gain
// sub-LSB structure. Run twice (horizontal then vertical) into an fp16
// scratch copy the decode pass then clamps to +/-0.5 LSB of the original.
//
// No transfer decode here — a pure weighted average of the raw samples.
// Linear-sampling optimization: adjacent tap pairs (a, a+1) are folded into
// a single bilinear fetch placed at the weight-barycentric offset
// `a + wb/(wa+wb)`, which the LINEAR sampler interpolates back into
// `wa·t[a] + wb·t[a+1]` exactly. Halves the fetch count vs one fetch per
// integer tap, with identical output when `radius` is even (the Rust side
// guarantees it). `radius == 0` degenerates to a center-only fetch — that's
// the bilinear 2× downsample blit the deband chain reuses this shader for.
#version 450

layout(set = 0, binding = 0) uniform sampler2D u_src;

layout(push_constant) uniform Push {
    // UV step between taps: (1/width, 0) for the horizontal pass,
    // (0, 1/height) for the vertical pass.
    vec2 axis;
    // Gaussian standard deviation, in taps (= source pixels).
    float sigma;
    // Half-width of the kernel in taps; the loop runs [-radius, radius].
    int radius;
} push;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

void main() {
    float inv_two_sigma_sq = 1.0 / (2.0 * push.sigma * push.sigma);
    // Center tap (offset 0, weight exp(0) = 1).
    vec3 acc = texture(u_src, v_uv).rgb;
    float wsum = 1.0;
    // One bilinear fetch per pair (a, a+1), mirrored to ±. `radius` even.
    int pairs = push.radius / 2;
    for (int k = 1; k <= pairs; k++) {
        float a = float(2 * k - 1);
        float b = a + 1.0;
        float wa = exp(-a * a * inv_two_sigma_sq);
        float wb = exp(-b * b * inv_two_sigma_sq);
        float wab = wa + wb;
        vec2 d = (a + wb / wab) * push.axis;
        acc += (texture(u_src, v_uv + d).rgb + texture(u_src, v_uv - d).rgb) * wab;
        wsum += 2.0 * wab;
    }
    out_color = vec4(acc / wsum, 1.0);
}
