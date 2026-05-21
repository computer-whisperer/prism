// Per-element decode pass: vertex shader.
//
// Draws an axis-aligned quad on the fp16 BT.2020-linear intermediate image.
// The quad position is taken from push constants; we generate the 4 vertices
// from gl_VertexIndex (use triangle-strip topology with 4 vertices).

#version 450

layout(push_constant) uniform Push {
    // Destination rectangle on the intermediate image, in clip space.
    // [-1,1] × [-1,1] coords (post-NDC), so this is normalized + signed.
    vec4 dst_rect_clip;   // x_min, y_min, x_max, y_max
    // Source rectangle on the texture, normalized [0,1] × [0,1].
    vec4 src_rect_uv;     // u_min, v_min, u_max, v_max
    // Decode primaries → BT.2020 matrix (in linear-light space).
    // Stored as mat4 to keep push-constant alignment trivial.
    mat4 decode_matrix;
    // Per-element tint (see decode.frag).
    vec4 tint;
    // Scalar parameters.
    float sdr_white_nits;
    int transfer;         // see decode.frag
    int _pad0;
    int _pad1;
} push;

layout(location = 0) out vec2 v_uv;

void main() {
    // Triangle strip: (0,0), (1,0), (0,1), (1,1).
    vec2 corner = vec2(gl_VertexIndex & 1, (gl_VertexIndex >> 1) & 1);
    vec2 dst_xy = mix(push.dst_rect_clip.xy, push.dst_rect_clip.zw, corner);
    vec2 src_uv = mix(push.src_rect_uv.xy, push.src_rect_uv.zw, corner);
    gl_Position = vec4(dst_xy, 0.0, 1.0);
    v_uv = src_uv;
}
