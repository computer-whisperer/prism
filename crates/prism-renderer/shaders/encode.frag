// Output encode pass: fragment shader.
//
// Pipeline stage 2 of 2. Inputs the fp16 BT.2020 absolute-nits intermediate.
// Outputs the per-output target color space + transfer.
//
// Output transfer enum (matches prism_frame::TransferFunction):
//   0 = Linear  (no encoding; for fp16 scanout — TODO not used yet)
//   1 = sRGB OETF (for SDR XRGB8888 scanout — current path)
//   2 = PQ OETF   (for HDR A2RGB10 / fp16 PQ scanout — TODO, plumbed but not used)
//
// Calibration is a per-output 3×3 matrix applied before encoding. Identity
// today (no calibration ingested); will become non-identity once we feed
// Spyder-derived corrections in.
//
// TODOs:
//   - Tone mapping when intermediate values exceed `target_peak_nits`.
//     Right now we just clamp. BT.2390 EETF or Hable are the candidates.
//   - HLG OETF for HLG-capable displays.

#version 450

layout(set = 0, binding = 0) uniform sampler2D u_intermediate;

layout(push_constant) uniform Push {
    mat4 cal_matrix;       // calibration CTM (in BT.2020 → out BT.2020 corrected)
    float sdr_white_nits;  // for sRGB output: how many nits is 1.0 of the input?
    float target_peak_nits;// display peak (for tone-mapping, currently unused)
    int output_transfer;
    int _pad;
} push;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

// sRGB OETF (linear → encoded), per IEC 61966-2-1.
float srgb_oetf_component(float c) {
    return c <= 0.0031308
        ? 12.92 * c
        : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}
vec3 srgb_oetf(vec3 c) {
    return vec3(srgb_oetf_component(c.r),
                srgb_oetf_component(c.g),
                srgb_oetf_component(c.b));
}

// PQ OETF (linear absolute nits → encoded V in [0,1]).
vec3 pq_oetf(vec3 y_nits) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    vec3 yn = pow(max(y_nits / 10000.0, vec3(0.0)), vec3(m1));
    vec3 num = c1 + c2 * yn;
    vec3 den = 1.0 + c3 * yn;
    return pow(num / den, vec3(m2));
}

void main() {
    vec3 sampled_nits = texture(u_intermediate, v_uv).rgb;

    // Per-output calibration (3×3, identity for now).
    vec3 cal = mat3(push.cal_matrix) * sampled_nits;

    vec3 encoded;
    if (push.output_transfer == 1) {
        // sRGB output: normalize nits to [0,1] using SDR white, clamp, encode.
        // TODO tone-map content above sdr_white_nits instead of hard-clipping.
        vec3 normalized = clamp(cal / max(push.sdr_white_nits, 1.0), vec3(0.0), vec3(1.0));
        encoded = srgb_oetf(normalized);
    } else if (push.output_transfer == 2) {
        // PQ output: absolute nits straight in, clamp to display peak.
        // TODO tone-map instead of hard-clipping.
        vec3 clamped = clamp(cal, vec3(0.0), vec3(push.target_peak_nits));
        encoded = pq_oetf(clamped);
    } else {
        // Linear output (fp16 scanout). Pass through normalized to peak.
        encoded = cal / max(push.target_peak_nits, 1.0);
    }

    out_color = vec4(encoded, 1.0);
}
