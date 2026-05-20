// Output encode pass: vertex shader.
//
// Renders a full-screen triangle (single draw, 3 vertices). The "single
// oversized tri" trick covers the screen with one primitive and no clipping
// — slightly more pipeline-friendly than a quad's two tris.

#version 450

layout(location = 0) out vec2 v_uv;

void main() {
    // Vertex IDs 0, 1, 2 → corner positions covering the screen.
    // (-1,-1), (3,-1), (-1,3) in clip space; UVs (0,0)..(2,0)..(0,2).
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    v_uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
