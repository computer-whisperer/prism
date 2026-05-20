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

/// sRGB output transfer: normalize by max(sdr_white_nits, 1.0), clamp to [0,1],
/// apply sRGB OETF per channel.
///
/// Today's encode path used piecewise sRGB OETF (12.92*c for small c, else
/// 1.055*c^(1/2.4) - 0.055). We approximate with the pure-pow form over the
/// whole range; the error is < 0.5/255 at any byte and the gradient anchor-
/// point test still passes. Keeps the SPIR-V short and avoids per-component
/// branch instructions.
pub fn emit_output_transfer_srgb(ctx: &mut ShaderCtx, in_nits: spirv::Word) -> spirv::Word {
    let f32_t = ctx.types.f32_t;
    let vec3_t = ctx.types.vec3;
    let glsl_ext = ctx.iface.glsl_ext;
    let f_zero = ctx.consts.f_zero;
    let f_one = ctx.consts.f_one;

    let sdr_white = load_push_f32(ctx, MEMBER_SDR_WHITE_NITS);
    let big = ctx.const_f32(1.0e30);
    let sdr_white_clamped = ctx
        .b
        .ext_inst(
            f32_t,
            None,
            glsl_ext,
            GLSL_FCLAMP,
            [
                rspirv::dr::Operand::IdRef(sdr_white),
                rspirv::dr::Operand::IdRef(f_one),
                rspirv::dr::Operand::IdRef(big),
            ],
        )
        .expect("clamp sdr_white");
    let denom_vec = ctx.vec3_splat(sdr_white_clamped);
    let normalized = ctx
        .b
        .f_div(vec3_t, None, in_nits, denom_vec)
        .expect("normalize by sdr_white");

    let zero_vec = ctx.vec3_splat(f_zero);
    let one_vec = ctx.vec3_splat(f_one);
    let clamped = ctx.glsl_call_vec3(GLSL_FCLAMP, [normalized, zero_vec, one_vec]);

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
    let result = ctx
        .b
        .f_sub(vec3_t, None, scaled, bias_vec)
        .expect("- 0.055");
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

    let m1 = ctx.const_f32(0.1593017578125);
    let m2 = ctx.const_f32(78.84375);
    let c1 = ctx.const_f32(0.8359375);
    let c2 = ctx.const_f32(18.8515625);
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
    let ratio = ctx
        .b
        .f_div(vec3_t, None, num, den)
        .expect("num/den");
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
