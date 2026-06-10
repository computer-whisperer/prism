//! SPIR-V code generation for individual `EncodeFragment` effects.
//!
//! Each `emit_*` takes a `ShaderCtx` (which holds the builder + cached
//! type/constant IDs) plus the incoming vec3 color id, and returns the
//! outgoing vec3 color id. The synthesizer threads the value through.
//!
//! Fragments that need to read push-constant fields do their own access-chain.
//! Fragments that need to sample the texture at extra positions (e.g. FIR
//! filter) also handle that themselves — but a multi-sample fragment can't
//! be folded into the simple "vec3 in → vec3 out" chain, so for the first
//! cut only single-sample-in fragments live here.

use rspirv::spirv;

use super::builder::ShaderCtx;
use super::push_constants::*;

// GLSL.std.450 instruction numbers we use. From the spec.
const GLSL_POW: u32 = 26;
const GLSL_FCLAMP: u32 = 43;
const GLSL_FMAX: u32 = 40;

/// Sample `u_intermediate` at `v_uv` and return the .rgb of the sampled vec4.
/// This is the first thing every synthesized encode shader does — produces
/// the starting vec3 the rest of the chain operates on.
pub fn emit_sample_intermediate(ctx: &mut ShaderCtx) -> spirv::Word {
    let vec2_t = ctx.types.vec2;
    let v_uv_ptr = ctx.iface.v_uv_ptr;
    let sampled_image_t = ctx.types.sampled_image;
    let u_intermediate_ptr = ctx.iface.u_intermediate_ptr;
    let vec4_t = ctx.types.vec4;
    let f32_t = ctx.types.f32_t;
    let vec3_t = ctx.types.vec3;

    let uv = ctx
        .b
        .load(vec2_t, None, v_uv_ptr, None, [])
        .expect("load v_uv");
    let sampler = ctx
        .b
        .load(sampled_image_t, None, u_intermediate_ptr, None, [])
        .expect("load sampled image");
    let sampled = ctx
        .b
        .image_sample_implicit_lod(vec4_t, None, sampler, uv, None, [])
        .expect("image_sample_implicit_lod");
    let r = ctx
        .b
        .composite_extract(f32_t, None, sampled, [0])
        .expect("extract r");
    let g = ctx
        .b
        .composite_extract(f32_t, None, sampled, [1])
        .expect("extract g");
    let b = ctx
        .b
        .composite_extract(f32_t, None, sampled, [2])
        .expect("extract b");
    ctx.b
        .composite_construct(vec3_t, None, [r, g, b])
        .expect("composite_construct rgb")
}

/// Apply the 3×3 portion of `push.cal_matrix` to `in_rgb`.
///
/// Storage is mat4; we extend the input to vec4(in.xyz, 0), multiply by
/// the mat4, take .xyz of the result. The fourth column is multiplied by
/// 0 and contributes nothing. This matches what GLSL `mat3(mat4_value)` does.
pub fn emit_calibration_matrix(ctx: &mut ShaderCtx, in_rgb: spirv::Word) -> spirv::Word {
    let member_idx = ctx.const_u32(MEMBER_CAL_MATRIX);
    let mat4_t = ctx.types.mat4;
    let push_mat4_ptr_t = ctx.ptrs.push_constant_mat4;
    let push_ptr = ctx.iface.push_ptr;
    let f32_t = ctx.types.f32_t;
    let vec4_t = ctx.types.vec4;
    let vec3_t = ctx.types.vec3;
    let f_zero = ctx.consts.f_zero;

    let mat_ptr = ctx
        .b
        .access_chain(push_mat4_ptr_t, None, push_ptr, [member_idx])
        .expect("access_chain cal_matrix");
    let mat = ctx
        .b
        .load(mat4_t, None, mat_ptr, None, [])
        .expect("load cal_matrix");
    let in_r = ctx
        .b
        .composite_extract(f32_t, None, in_rgb, [0])
        .expect("extract r");
    let in_g = ctx
        .b
        .composite_extract(f32_t, None, in_rgb, [1])
        .expect("extract g");
    let in_b = ctx
        .b
        .composite_extract(f32_t, None, in_rgb, [2])
        .expect("extract b");
    let in_vec4 = ctx
        .b
        .composite_construct(vec4_t, None, [in_r, in_g, in_b, f_zero])
        .expect("composite_construct vec4 for mat mul");
    let out_vec4 = ctx
        .b
        .matrix_times_vector(vec4_t, None, mat, in_vec4)
        .expect("matrix_times_vector");
    let out_r = ctx
        .b
        .composite_extract(f32_t, None, out_vec4, [0])
        .expect("extract r");
    let out_g = ctx
        .b
        .composite_extract(f32_t, None, out_vec4, [1])
        .expect("extract g");
    let out_b = ctx
        .b
        .composite_extract(f32_t, None, out_vec4, [2])
        .expect("extract b");
    ctx.b
        .composite_construct(vec3_t, None, [out_r, out_g, out_b])
        .expect("composite_construct output rgb")
}

/// sRGB output transfer: clamp linear drive to [0, 1], apply sRGB OETF per
/// channel. Parameter-free by design — the upstream LUT outputs the panel's
/// linear drive value directly (the wire value pre-OETF), so the only job
/// left here is refusing to send out-of-range control values to the panel.
/// Mirrors how `OutputTransferPq` treats its input as absolute PQ nits: the
/// LUT owns the BT.2020→panel mapping end-to-end in both modes, and no
/// runtime policy knob (e.g. `sdr-reference-nits`) can silently re-scale a
/// baked calibration.
///
/// Exact piecewise sRGB OETF (IEC 61966-2-1): `12.92*c` for
/// `c <= 0.0031308`, else `1.055*c^(1/2.4) - 0.055`. Both segments are
/// computed unconditionally and merged with a component-wise `OpSelect` —
/// no branches, so no divergence cost. Exactness matters: the decode shader
/// and every CPU-side helper (`diagnose::srgb_eotf`, prism-tune's
/// `srgb_oetf`) are piecewise, and `drive_identity_lut` relies on
/// encode being the true inverse of decode for code-value-stable SDR
/// passthrough. The pure-pow form this replaced deviated up to ~2/255
/// in the toe (slight black crush).
pub fn emit_output_transfer_srgb(ctx: &mut ShaderCtx, in_drive: spirv::Word) -> spirv::Word {
    let vec3_t = ctx.types.vec3;
    let bool_t = ctx.types.bool_t;
    let f_zero = ctx.consts.f_zero;
    let f_one = ctx.consts.f_one;

    let zero_vec = ctx.vec3_splat(f_zero);
    let one_vec = ctx.vec3_splat(f_one);
    let clamped = ctx.glsl_call_vec3(GLSL_FCLAMP, [in_drive, zero_vec, one_vec]);

    // Linear toe: 12.92 * c.
    let toe_scale = ctx.const_f32(12.92);
    let toe_scale_vec = ctx.vec3_splat(toe_scale);
    let toe = ctx
        .b
        .f_mul(vec3_t, None, clamped, toe_scale_vec)
        .expect("12.92 * c");

    // Power segment: 1.055 * c^(1/2.4) - 0.055.
    let inv_24 = ctx.const_f32(1.0 / 2.4);
    let inv_24_vec = ctx.vec3_splat(inv_24);
    let c_pow = ctx.glsl_call_vec3(GLSL_POW, [clamped, inv_24_vec]);
    let scale = ctx.const_f32(1.055);
    let scale_vec = ctx.vec3_splat(scale);
    let scaled = ctx
        .b
        .f_mul(vec3_t, None, c_pow, scale_vec)
        .expect("c_pow * 1.055");
    let bias = ctx.const_f32(0.055);
    let bias_vec = ctx.vec3_splat(bias);
    let powered = ctx
        .b
        .f_sub(vec3_t, None, scaled, bias_vec)
        .expect("- 0.055");

    // Component-wise select on the spec threshold. `type_vector` dedupes,
    // so re-requesting bvec3 here is free.
    let bvec3_t = ctx.b.type_vector(bool_t, 3);
    let threshold = ctx.const_f32(0.003_130_8);
    let threshold_vec = ctx.vec3_splat(threshold);
    let in_toe = ctx
        .b
        .f_ord_less_than_equal(bvec3_t, None, clamped, threshold_vec)
        .expect("c <= 0.0031308");
    let result = ctx
        .b
        .select(vec3_t, None, in_toe, toe, powered)
        .expect("select toe / power segment");

    let zero_vec2 = ctx.vec3_splat(f_zero);
    let one_vec2 = ctx.vec3_splat(f_one);
    ctx.glsl_call_vec3(GLSL_FCLAMP, [result, zero_vec2, one_vec2])
}

/// PQ output transfer (SMPTE ST 2084): clamp nits to target peak, apply PQ OETF.
pub fn emit_output_transfer_pq(ctx: &mut ShaderCtx, in_nits: spirv::Word) -> spirv::Word {
    let vec3_t = ctx.types.vec3;
    let f_zero = ctx.consts.f_zero;
    let f_one = ctx.consts.f_one;

    let peak = load_push_f32(ctx, MEMBER_TARGET_PEAK_NITS);
    let zero_vec = ctx.vec3_splat(f_zero);
    let peak_vec = ctx.vec3_splat(peak);
    let clamped = ctx.glsl_call_vec3(GLSL_FCLAMP, [in_nits, zero_vec, peak_vec]);

    let inv_10k = ctx.const_f32(1.0 / 10000.0);
    let inv_10k_vec = ctx.vec3_splat(inv_10k);
    let yn = ctx
        .b
        .f_mul(vec3_t, None, clamped, inv_10k_vec)
        .expect("yn = clamped/10000");

    let m1 = ctx.const_f32(0.159_301_76);
    let m2 = ctx.const_f32(78.84375);
    let c1 = ctx.const_f32(0.8359375);
    let c2 = ctx.const_f32(18.851_563);
    let c3 = ctx.const_f32(18.6875);
    let m1_vec = ctx.vec3_splat(m1);
    let m2_vec = ctx.vec3_splat(m2);
    let c1_vec = ctx.vec3_splat(c1);
    let c2_vec = ctx.vec3_splat(c2);
    let c3_vec = ctx.vec3_splat(c3);
    let one_vec = ctx.vec3_splat(f_one);

    let yn_pow = ctx.glsl_call_vec3(GLSL_POW, [yn, m1_vec]);
    let c2_yn = ctx
        .b
        .f_mul(vec3_t, None, yn_pow, c2_vec)
        .expect("c2 * yn_pow");
    let num = ctx
        .b
        .f_add(vec3_t, None, c1_vec, c2_yn)
        .expect("c1 + c2*yn");
    let c3_yn = ctx
        .b
        .f_mul(vec3_t, None, yn_pow, c3_vec)
        .expect("c3 * yn_pow");
    let den = ctx
        .b
        .f_add(vec3_t, None, one_vec, c3_yn)
        .expect("1 + c3*yn");
    let ratio = ctx.b.f_div(vec3_t, None, num, den).expect("num/den");
    ctx.glsl_call_vec3(GLSL_POW, [ratio, m2_vec])
}

/// Linear output transfer: divide by target_peak_nits, no encoding. For fp16
/// scanout where the panel expects already-linear values, or for debugging.
pub fn emit_output_transfer_linear(ctx: &mut ShaderCtx, in_nits: spirv::Word) -> spirv::Word {
    let f32_t = ctx.types.f32_t;
    let vec3_t = ctx.types.vec3;
    let glsl_ext = ctx.iface.glsl_ext;
    let f_one = ctx.consts.f_one;

    let peak = load_push_f32(ctx, MEMBER_TARGET_PEAK_NITS);
    let big = ctx.const_f32(1.0e30);
    let peak_clamped = ctx
        .b
        .ext_inst(
            f32_t,
            None,
            glsl_ext,
            GLSL_FCLAMP,
            [
                rspirv::dr::Operand::IdRef(peak),
                rspirv::dr::Operand::IdRef(f_one),
                rspirv::dr::Operand::IdRef(big),
            ],
        )
        .expect("clamp peak");
    let peak_vec = ctx.vec3_splat(peak_clamped);
    ctx.b
        .f_div(vec3_t, None, in_nits, peak_vec)
        .expect("divide by peak")
}

/// Per-channel response correction: invert the panel's measured
/// `emitted = gain * commanded^gamma` curve. Computes
/// `out = (in / max(gain, eps))^(1 / max(gamma, eps))` per channel.
///
/// `gain` and `gamma` come from push constant members `response_gain`
/// and `response_gamma` (each `vec4`, only `.rgb` used). Identity values
/// are `gain = (1, 1, 1)` and `gamma = (1, 1, 1)` — those make this
/// fragment a no-op (`out = in`), which is what unconfigured panels
/// (e.g. the OLED in its linear range) want.
///
/// The epsilon floor (`1e-3`) prevents division by zero and zero^k
/// surprises for misconfigured panels. A panel-author who sets gain
/// or gamma below epsilon gets clipped to epsilon, which produces
/// huge but bounded commanded values that the encoder's downstream
/// PQ_OETF can still process.
pub fn emit_per_channel_response_gain_gamma(
    ctx: &mut ShaderCtx,
    in_nits: spirv::Word,
) -> spirv::Word {
    let vec3_t = ctx.types.vec3;
    let glsl_ext = ctx.iface.glsl_ext;

    let gain_vec4 = load_push_vec4(ctx, MEMBER_RESPONSE_GAIN);
    let gamma_vec4 = load_push_vec4(ctx, MEMBER_RESPONSE_GAMMA);
    let gain = vec4_xyz(ctx, gain_vec4);
    let gamma = vec4_xyz(ctx, gamma_vec4);

    let eps = ctx.const_f32(1.0e-3);
    let eps_vec = ctx.vec3_splat(eps);
    let f_zero = ctx.consts.f_zero;
    let zero_vec = ctx.vec3_splat(f_zero);
    let f_one = ctx.consts.f_one;
    let one_vec = ctx.vec3_splat(f_one);

    // safe_gain = max(gain, eps); safe_gamma = max(gamma, eps)
    let safe_gain = ctx.glsl_call_vec3(GLSL_FMAX, [gain, eps_vec]);
    let safe_gamma = ctx.glsl_call_vec3(GLSL_FMAX, [gamma, eps_vec]);

    // ratio = max(in_nits, 0) / safe_gain
    let in_clamped = ctx.glsl_call_vec3(GLSL_FMAX, [in_nits, zero_vec]);
    let ratio = ctx
        .b
        .f_div(vec3_t, None, in_clamped, safe_gain)
        .expect("ratio = in / gain");

    // inv_gamma = 1 / safe_gamma
    let inv_gamma = ctx
        .b
        .f_div(vec3_t, None, one_vec, safe_gamma)
        .expect("inv_gamma = 1 / gamma");

    // out = pow(ratio, inv_gamma)
    let _ = glsl_ext; // already referenced via ext_inst calls above
    ctx.glsl_call_vec3(GLSL_POW, [ratio, inv_gamma])
}

/// 3D LUT lookup with a per-channel PQ shaper on input.
///
/// Replaces the [`CalibrationMatrix`](super::EncodeFragment::CalibrationMatrix) +
/// [`PerChannelResponseGainGamma`](super::EncodeFragment::PerChannelResponseGainGamma)
/// pair. The math is:
///
/// ```text
/// clamped = clamp(in_nits, 0, 10000)         // PQ-domain overflow guard
/// coord   = pq_oetf(clamped)                  // [0, 10000] nits → [0, 1] LUT coord
/// out    = texture(u_lut, coord).rgb          // trilinear sample; nits
/// ```
///
/// `pq_oetf` (SMPTE ST 2084 inverse-EOTF) is the same function the
/// `OutputTransferPq` fragment uses for output encoding — reused here as a
/// "shaper" so the LUT grid is allocated perceptually. With the shaper, a
/// 17³ or 33³ LUT puts most of its precision in the dim region where the
/// eye is sensitive; without it we'd need a much larger LUT to get the same
/// quality at low luminance.
///
/// The LUT itself is calibration data, populated per-output: identity LUT
/// for uncalibrated displays, synthesized from CTM + per-channel curve
/// when those are configured, or loaded from a measured binary file. The
/// shader is agnostic — it just samples.
///
/// Sampler filter (LINEAR, set on the pipeline side) gives trilinear
/// interpolation between the 8 nearest grid points. `CLAMP_TO_EDGE` address
/// mode means inputs above the highest grid point clip to the LUT's
/// boundary value rather than wrapping or reading garbage.
///
/// **Gamut handling**: previous versions clamped per-channel via a
/// push-constant `lut_input_max_nits`. That box was the wrong shape for
/// the panel's reachable parallelepiped and pre-clipped colors the LUT's
/// own cross-channel compensation could handle. The bake now projects
/// out-of-gamut and below-floor requests onto the measured reachable
/// surface at calibration time, so the table degrades gracefully and no
/// shader-side per-channel pre-clamp is needed — only the loose PQ-domain
/// `[0, 10000]` overflow guard below.
pub fn emit_lut3d(ctx: &mut ShaderCtx, in_nits: spirv::Word) -> spirv::Word {
    let vec3_t = ctx.types.vec3;
    let vec4_t = ctx.types.vec4;
    let f32_t = ctx.types.f32_t;
    let f_zero = ctx.consts.f_zero;
    let f_one = ctx.consts.f_one;
    let sampled_image_3d_t = ctx.types.sampled_image_3d;
    let u_lut_ptr = ctx
        .iface
        .u_lut_ptr
        .expect("EncodeFragment::Lut3d requires builder to declare u_lut");

    // Shaper: PQ-encode the linear-nits input into [0, 1] coord space.
    // Loose [0, 10000] clamp keeps the PQ shaper's pow() out of trouble
    // on adversarial or accumulated-overflow inputs; the bake handles
    // gamut/range mapping inside the LUT, so we don't need a per-channel
    // box from push constants here.
    let zero_vec = ctx.vec3_splat(f_zero);
    let pq_peak = ctx.const_f32(10_000.0);
    let max_vec = ctx.vec3_splat(pq_peak);
    let in_clamped = ctx.glsl_call_vec3(GLSL_FCLAMP, [in_nits, zero_vec, max_vec]);

    let inv_10k = ctx.const_f32(1.0 / 10000.0);
    let inv_10k_vec = ctx.vec3_splat(inv_10k);
    let yn = ctx
        .b
        .f_mul(vec3_t, None, in_clamped, inv_10k_vec)
        .expect("yn = clamped/10000");

    // PQ OETF constants (SMPTE ST 2084).
    let m1 = ctx.const_f32(0.159_301_76);
    let m2 = ctx.const_f32(78.84375);
    let c1 = ctx.const_f32(0.8359375);
    let c2 = ctx.const_f32(18.851_563);
    let c3 = ctx.const_f32(18.6875);
    let m1_vec = ctx.vec3_splat(m1);
    let m2_vec = ctx.vec3_splat(m2);
    let c1_vec = ctx.vec3_splat(c1);
    let c2_vec = ctx.vec3_splat(c2);
    let c3_vec = ctx.vec3_splat(c3);
    let one_vec = ctx.vec3_splat(f_one);

    let yn_pow = ctx.glsl_call_vec3(GLSL_POW, [yn, m1_vec]);
    let c2_yn = ctx
        .b
        .f_mul(vec3_t, None, yn_pow, c2_vec)
        .expect("c2 * yn_pow");
    let num = ctx
        .b
        .f_add(vec3_t, None, c1_vec, c2_yn)
        .expect("c1 + c2*yn_pow");
    let c3_yn = ctx
        .b
        .f_mul(vec3_t, None, yn_pow, c3_vec)
        .expect("c3 * yn_pow");
    let den = ctx
        .b
        .f_add(vec3_t, None, one_vec, c3_yn)
        .expect("1 + c3*yn_pow");
    let ratio = ctx.b.f_div(vec3_t, None, num, den).expect("num / den");
    let logical_coord = ctx.glsl_call_vec3(GLSL_POW, [ratio, m2_vec]);

    // ── Texel-center adjustment ───────────────────────────────────────
    // LUT entries are stored at integer indices `i ∈ [0, N-1]`. The
    // GPU sampler's default convention puts texel `i`'s center at
    // texture coord `(i+0.5)/N` — so sampling at the "logical"
    // coord `i/(N-1)` would land between texel `i-1` and texel `i`
    // for most `i`, dragging in the wrong neighbours. The standard
    // 3D-LUT remedy:
    //     texture_coord = logical_coord × (N-1)/N + 0.5/N
    // which maps `logical_coord = i/(N-1) → texture_coord = (i+0.5)/N`
    // (i.e. exact texel `i` center) for every `i` and interpolates
    // linearly between adjacent entries in between. `N` is baked in
    // from `lut3d::LUT_CUBE_EDGE` at shader synthesis time; if that
    // ever needs to vary per-output we'd thread it through
    // `EncodeConfig`.
    let n = crate::lut3d::LUT_CUBE_EDGE as f32;
    let scale = ctx.const_f32((n - 1.0) / n);
    let bias = ctx.const_f32(0.5 / n);
    let scale_vec = ctx.vec3_splat(scale);
    let bias_vec = ctx.vec3_splat(bias);
    let scaled = ctx
        .b
        .f_mul(vec3_t, None, logical_coord, scale_vec)
        .expect("coord * (N-1)/N");
    let coord = ctx
        .b
        .f_add(vec3_t, None, scaled, bias_vec)
        .expect("coord * (N-1)/N + 0.5/N");

    // Sample the 3D LUT at `coord`. Linear filtering + CLAMP_TO_EDGE
    // (configured on the sampler at pipeline-create time) gives trilinear
    // interpolation between adjacent grid points and clamps out-of-range
    // coords to the boundary.
    let sampler = ctx
        .b
        .load(sampled_image_3d_t, None, u_lut_ptr, None, [])
        .expect("load u_lut sampler3D");
    let sampled = ctx
        .b
        .image_sample_implicit_lod(vec4_t, None, sampler, coord, None, [])
        .expect("image_sample_implicit_lod (lut3d)");
    // Drop alpha — the LUT carries RGB nits in .rgb; .a is unused but
    // forced to 1.0 at upload so a stray sampler isn't garbage.
    let r = ctx
        .b
        .composite_extract(f32_t, None, sampled, [0])
        .expect("extract r (lut)");
    let g = ctx
        .b
        .composite_extract(f32_t, None, sampled, [1])
        .expect("extract g (lut)");
    let b = ctx
        .b
        .composite_extract(f32_t, None, sampled, [2])
        .expect("extract b (lut)");
    ctx.b
        .composite_construct(vec3_t, None, [r, g, b])
        .expect("composite_construct lut rgb")
}

/// Helper: load a single f32 push-constant member.
fn load_push_f32(ctx: &mut ShaderCtx, member_index: u32) -> spirv::Word {
    let idx = ctx.const_u32(member_index);
    let push_f32_ptr_t = ctx.ptrs.push_constant_f32;
    let push_ptr = ctx.iface.push_ptr;
    let f32_t = ctx.types.f32_t;
    let ptr = ctx
        .b
        .access_chain(push_f32_ptr_t, None, push_ptr, [idx])
        .expect("access_chain push f32");
    ctx.b
        .load(f32_t, None, ptr, None, [])
        .expect("load push f32")
}

/// Helper: load a single vec4 push-constant member.
fn load_push_vec4(ctx: &mut ShaderCtx, member_index: u32) -> spirv::Word {
    let idx = ctx.const_u32(member_index);
    let push_vec4_ptr_t = ctx.ptrs.push_constant_vec4;
    let push_ptr = ctx.iface.push_ptr;
    let vec4_t = ctx.types.vec4;
    let ptr = ctx
        .b
        .access_chain(push_vec4_ptr_t, None, push_ptr, [idx])
        .expect("access_chain push vec4");
    ctx.b
        .load(vec4_t, None, ptr, None, [])
        .expect("load push vec4")
}

/// Helper: extract the xyz of a vec4 into a vec3.
fn vec4_xyz(ctx: &mut ShaderCtx, v: spirv::Word) -> spirv::Word {
    let f32_t = ctx.types.f32_t;
    let vec3_t = ctx.types.vec3;
    let x = ctx
        .b
        .composite_extract(f32_t, None, v, [0])
        .expect("extract x");
    let y = ctx
        .b
        .composite_extract(f32_t, None, v, [1])
        .expect("extract y");
    let z = ctx
        .b
        .composite_extract(f32_t, None, v, [2])
        .expect("extract z");
    ctx.b
        .composite_construct(vec3_t, None, [x, y, z])
        .expect("composite_construct vec3 from xyz")
}
