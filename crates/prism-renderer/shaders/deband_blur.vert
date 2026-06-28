// Fullscreen-quad vertex for the deband separable-blur passes.
//
// 4-vertex triangle strip covering the whole scratch target; emits uv in
// [0, 1]. No vertex buffer — positions derived from gl_VertexIndex.
#version 450

layout(location = 0) out vec2 v_uv;

void main() {
    vec2 p = vec2(float(gl_VertexIndex & 1), float((gl_VertexIndex >> 1) & 1));
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
