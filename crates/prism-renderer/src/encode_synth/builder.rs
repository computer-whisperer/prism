//! Thin wrapper around `rspirv::dr::Builder` that caches commonly-used types
//! and constants, and tracks the interface variables (texture, push constants,
//! input UV, output color) the fragments need to reference.
//!
//! All fragments emit instructions into the *currently-open block* of the
//! main function; this struct holds the IDs they need without each fragment
//! re-declaring them.

use rspirv::dr::Builder;
use rspirv::spirv;

use super::push_constants::*;
use super::EncodeConfig;

/// Cached SPIR-V Word IDs for types we use across the shader.
pub struct TypeIds {
    pub void: spirv::Word,
    pub bool_t: spirv::Word,
    pub f32_t: spirv::Word,
    pub u32_t: spirv::Word,
    pub i32_t: spirv::Word,
    pub vec2: spirv::Word,
    pub vec3: spirv::Word,
    pub vec4: spirv::Word,
    pub mat4: spirv::Word,
    /// `OpTypeImage %f32 2D 0 0 0 1 Unknown`.
    pub image: spirv::Word,
    pub sampled_image: spirv::Word,
    /// `OpTypeImage %f32 3D 0 0 0 1 Unknown` — the 3D LUT texture.
    pub image_3d: spirv::Word,
    pub sampled_image_3d: spirv::Word,
}

/// Pointer types we'll need for load/store / access-chain operations.
pub struct PointerIds {
    pub input_vec2: spirv::Word,
    pub output_vec4: spirv::Word,
    pub uniform_const_sampled_image: spirv::Word,
    pub uniform_const_sampled_image_3d: spirv::Word,
    pub push_constant_struct: spirv::Word,
    pub push_constant_mat4: spirv::Word,
    pub push_constant_vec4: spirv::Word,
    pub push_constant_f32: spirv::Word,
    pub function_vec3: spirv::Word,
    pub function_vec4: spirv::Word,
}

/// Common float constants we reuse.
pub struct ConstantIds {
    pub f_zero: spirv::Word,
    pub f_one: spirv::Word,
    pub u_zero: spirv::Word,
    pub u_one: spirv::Word,
    pub u_two: spirv::Word,
    pub u_three: spirv::Word,
}

/// Interface variables visible to all fragments.
pub struct InterfaceIds {
    /// `layout(location=0) in vec2 v_uv;` (pointer)
    pub v_uv_ptr: spirv::Word,
    /// `layout(location=0) out vec4 out_color;` (pointer)
    pub out_color_ptr: spirv::Word,
    /// `layout(set=0, binding=0) uniform sampler2D u_intermediate;` (pointer)
    pub u_intermediate_ptr: spirv::Word,
    /// `layout(set=0, binding=1) uniform sampler3D u_lut;` (pointer) — the
    /// per-output 3D color LUT. Declared only when the encode chain
    /// includes the [`Lut3d`](super::EncodeFragment::Lut3d) fragment; the
    /// pipeline-layout side mirrors this so binding 1 only exists when the
    /// shader actually samples it.
    pub u_lut_ptr: Option<spirv::Word>,
    /// `layout(push_constant) uniform Push { ... } push;` (pointer to struct)
    pub push_ptr: spirv::Word,
    /// `GLSL.std.450` extension import (needed for pow, fclamp, etc.)
    pub glsl_ext: spirv::Word,
}

/// Top-level builder context the fragment-emit functions all share.
pub struct ShaderCtx {
    pub b: Builder,
    pub types: TypeIds,
    pub ptrs: PointerIds,
    pub consts: ConstantIds,
    pub iface: InterfaceIds,
    /// `main` function id, set during construction.
    pub main_id: spirv::Word,
}

impl ShaderCtx {
    /// Initialize a fresh fragment-shader module with all the boilerplate
    /// (capability, memory model, types, interface vars, push constants
    /// struct, main function with one open block). Fragments then emit
    /// their work into the open block before `finish()` is called.
    ///
    /// `config` decides which optional interface variables get declared —
    /// e.g. binding 1 for the per-output 3D LUT is only emitted when the
    /// chain contains [`EncodeFragment::Lut3d`](super::EncodeFragment::Lut3d).
    /// This keeps shader interfaces minimal: a configuration that doesn't
    /// use the LUT doesn't force the pipeline layout to allocate a slot for it.
    pub fn new(config: &EncodeConfig) -> Self {
        let needs_lut = config.uses_lut3d();
        let mut b = Builder::new();
        // SPIR-V 1.5 is fine for Vulkan 1.3.
        b.set_version(1, 5);
        b.capability(spirv::Capability::Shader);
        let glsl_ext = b.ext_inst_import("GLSL.std.450");
        b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);

        // ── Types ──────────────────────────────────────────────────────────
        let void = b.type_void();
        let bool_t = b.type_bool();
        let f32_t = b.type_float(32, None);
        let u32_t = b.type_int(32, 0);
        let i32_t = b.type_int(32, 1);
        let vec2 = b.type_vector(f32_t, 2);
        let vec3 = b.type_vector(f32_t, 3);
        let vec4 = b.type_vector(f32_t, 4);
        let mat4 = b.type_matrix(vec4, 4);
        // Sampled 2D color image — depth=0, arrayed=0, MS=0, sampled=1.
        let image = b.type_image(
            f32_t,
            spirv::Dim::Dim2D,
            0,
            0,
            0,
            1,
            spirv::ImageFormat::Unknown,
            None,
        );
        let sampled_image = b.type_sampled_image(image);
        // Sampled 3D color image for the per-output color LUT — same shape
        // as the 2D version but `Dim::Dim3D`. Trilinear sampling happens at
        // sample-time, controlled by the sampler.
        let image_3d = b.type_image(
            f32_t,
            spirv::Dim::Dim3D,
            0,
            0,
            0,
            1,
            spirv::ImageFormat::Unknown,
            None,
        );
        let sampled_image_3d = b.type_sampled_image(image_3d);

        let types = TypeIds {
            void,
            bool_t,
            f32_t,
            u32_t,
            i32_t,
            vec2,
            vec3,
            vec4,
            mat4,
            image,
            sampled_image,
            image_3d,
            sampled_image_3d,
        };

        // ── Push-constant struct type ──────────────────────────────────────
        let push_struct = b.type_struct(vec![
            mat4,  // 0: cal_matrix
            vec4,  // 1: response_gain (rgb + reserved)
            vec4,  // 2: response_gamma (rgb + reserved)
            vec4,  // 3: lut_input_max_nits
            f32_t, // 4: sdr_white_nits
            f32_t, // 5: target_peak_nits
            f32_t, // 6: dither_strength
            f32_t, // 7: _pad
        ]);
        // Block decoration so SPIR-V treats it as a push-constant block.
        b.decorate(push_struct, spirv::Decoration::Block, []);
        // Per-member decorations: offsets + matrix layout.
        b.member_decorate(
            push_struct,
            MEMBER_CAL_MATRIX,
            spirv::Decoration::Offset,
            [rspirv::dr::Operand::LiteralBit32(OFFSET_CAL_MATRIX)],
        );
        b.member_decorate(
            push_struct,
            MEMBER_CAL_MATRIX,
            spirv::Decoration::MatrixStride,
            [rspirv::dr::Operand::LiteralBit32(16)],
        );
        b.member_decorate(
            push_struct,
            MEMBER_CAL_MATRIX,
            spirv::Decoration::ColMajor,
            [],
        );
        for (member, offset) in [
            (MEMBER_RESPONSE_GAIN, OFFSET_RESPONSE_GAIN),
            (MEMBER_RESPONSE_GAMMA, OFFSET_RESPONSE_GAMMA),
            (MEMBER_LUT_INPUT_MAX_NITS, OFFSET_LUT_INPUT_MAX_NITS),
            (MEMBER_SDR_WHITE_NITS, OFFSET_SDR_WHITE_NITS),
            (MEMBER_TARGET_PEAK_NITS, OFFSET_TARGET_PEAK_NITS),
            (MEMBER_DITHER_STRENGTH, OFFSET_DITHER_STRENGTH),
        ] {
            b.member_decorate(
                push_struct,
                member,
                spirv::Decoration::Offset,
                [rspirv::dr::Operand::LiteralBit32(offset)],
            );
        }
        // Padding member (no offset decoration needed since it isn't read,
        // but emitting one keeps the struct layout valid).
        b.member_decorate(
            push_struct,
            7,
            spirv::Decoration::Offset,
            [rspirv::dr::Operand::LiteralBit32(OFFSET_PAD)],
        );

        // ── Pointer types ──────────────────────────────────────────────────
        let input_vec2 = b.type_pointer(None, spirv::StorageClass::Input, vec2);
        let output_vec4 = b.type_pointer(None, spirv::StorageClass::Output, vec4);
        let uniform_const_sampled_image =
            b.type_pointer(None, spirv::StorageClass::UniformConstant, sampled_image);
        let uniform_const_sampled_image_3d =
            b.type_pointer(None, spirv::StorageClass::UniformConstant, sampled_image_3d);
        let push_constant_struct =
            b.type_pointer(None, spirv::StorageClass::PushConstant, push_struct);
        let push_constant_mat4 = b.type_pointer(None, spirv::StorageClass::PushConstant, mat4);
        let push_constant_vec4 = b.type_pointer(None, spirv::StorageClass::PushConstant, vec4);
        let push_constant_f32 = b.type_pointer(None, spirv::StorageClass::PushConstant, f32_t);
        let function_vec3 = b.type_pointer(None, spirv::StorageClass::Function, vec3);
        let function_vec4 = b.type_pointer(None, spirv::StorageClass::Function, vec4);

        let ptrs = PointerIds {
            input_vec2,
            output_vec4,
            uniform_const_sampled_image,
            uniform_const_sampled_image_3d,
            push_constant_struct,
            push_constant_mat4,
            push_constant_vec4,
            push_constant_f32,
            function_vec3,
            function_vec4,
        };

        // ── Constants ──────────────────────────────────────────────────────
        let f_zero = b.constant_bit32(f32_t, 0u32);
        let f_one = b.constant_bit32(f32_t, 1.0f32.to_bits());
        let u_zero = b.constant_bit32(u32_t, 0);
        let u_one = b.constant_bit32(u32_t, 1);
        let u_two = b.constant_bit32(u32_t, 2);
        let u_three = b.constant_bit32(u32_t, 3);
        let consts = ConstantIds {
            f_zero,
            f_one,
            u_zero,
            u_one,
            u_two,
            u_three,
        };

        // ── Interface variables ────────────────────────────────────────────
        let v_uv_ptr = b.variable(input_vec2, None, spirv::StorageClass::Input, None);
        let out_color_ptr = b.variable(output_vec4, None, spirv::StorageClass::Output, None);
        let u_intermediate_ptr = b.variable(
            uniform_const_sampled_image,
            None,
            spirv::StorageClass::UniformConstant,
            None,
        );
        let u_lut_ptr = if needs_lut {
            Some(b.variable(
                uniform_const_sampled_image_3d,
                None,
                spirv::StorageClass::UniformConstant,
                None,
            ))
        } else {
            None
        };
        let push_ptr = b.variable(
            push_constant_struct,
            None,
            spirv::StorageClass::PushConstant,
            None,
        );

        b.decorate(
            v_uv_ptr,
            spirv::Decoration::Location,
            [rspirv::dr::Operand::LiteralBit32(0)],
        );
        b.decorate(
            out_color_ptr,
            spirv::Decoration::Location,
            [rspirv::dr::Operand::LiteralBit32(0)],
        );
        b.decorate(
            u_intermediate_ptr,
            spirv::Decoration::DescriptorSet,
            [rspirv::dr::Operand::LiteralBit32(0)],
        );
        b.decorate(
            u_intermediate_ptr,
            spirv::Decoration::Binding,
            [rspirv::dr::Operand::LiteralBit32(0)],
        );
        if let Some(lut_ptr) = u_lut_ptr {
            b.decorate(
                lut_ptr,
                spirv::Decoration::DescriptorSet,
                [rspirv::dr::Operand::LiteralBit32(0)],
            );
            b.decorate(
                lut_ptr,
                spirv::Decoration::Binding,
                [rspirv::dr::Operand::LiteralBit32(1)],
            );
        }

        let iface = InterfaceIds {
            v_uv_ptr,
            out_color_ptr,
            u_intermediate_ptr,
            u_lut_ptr,
            push_ptr,
            glsl_ext,
        };

        // ── main() function ────────────────────────────────────────────────
        let void_fn_ty = b.type_function(void, vec![]);
        let main_id = b
            .begin_function(void, None, spirv::FunctionControl::NONE, void_fn_ty)
            .expect("begin_function main");
        b.begin_block(None).expect("begin_block entry");

        // Declare the entry point + execution mode now that main_id exists.
        // SPIR-V 1.4+ requires ALL interface variables (Input, Output, AND
        // referenced UniformConstant / PushConstant globals) be listed.
        // `u_lut_ptr` is included only when the chain references it — the
        // pipeline-layout side mirrors this conditional so an LUT-less
        // chain doesn't pay for an unused descriptor binding.
        let mut iface_list: Vec<spirv::Word> =
            vec![v_uv_ptr, out_color_ptr, u_intermediate_ptr, push_ptr];
        if let Some(lut_ptr) = u_lut_ptr {
            iface_list.push(lut_ptr);
        }
        b.entry_point(spirv::ExecutionModel::Fragment, main_id, "main", iface_list);
        b.execution_mode(main_id, spirv::ExecutionMode::OriginUpperLeft, []);

        Self {
            b,
            types,
            ptrs,
            consts,
            iface,
            main_id,
        }
    }

    /// Create or fetch an f32 constant with the given value.
    pub fn const_f32(&mut self, value: f32) -> spirv::Word {
        self.b.constant_bit32(self.types.f32_t, value.to_bits())
    }

    /// Create or fetch a u32 constant.
    pub fn const_u32(&mut self, value: u32) -> spirv::Word {
        self.b.constant_bit32(self.types.u32_t, value)
    }

    /// Splat an f32 into a vec3.
    pub fn vec3_splat(&mut self, x: spirv::Word) -> spirv::Word {
        let ty = self.types.vec3;
        self.b
            .composite_construct(ty, None, [x, x, x])
            .expect("composite_construct vec3 splat")
    }

    /// `GLSL.std.450` ext-inst call with one f32 result type.
    pub fn glsl_call_f32(
        &mut self,
        instruction: u32,
        args: impl IntoIterator<Item = spirv::Word>,
    ) -> spirv::Word {
        let ty = self.types.f32_t;
        let ext = self.iface.glsl_ext;
        let operands: Vec<_> = args.into_iter().map(rspirv::dr::Operand::IdRef).collect();
        self.b
            .ext_inst(ty, None, ext, instruction, operands)
            .expect("ext_inst")
    }

    /// `GLSL.std.450` ext-inst call with a vec3 result type.
    pub fn glsl_call_vec3(
        &mut self,
        instruction: u32,
        args: impl IntoIterator<Item = spirv::Word>,
    ) -> spirv::Word {
        let ty = self.types.vec3;
        let ext = self.iface.glsl_ext;
        let operands: Vec<_> = args.into_iter().map(rspirv::dr::Operand::IdRef).collect();
        self.b
            .ext_inst(ty, None, ext, instruction, operands)
            .expect("ext_inst")
    }

    /// Finalize the module: write the final `out_color`, end main, return SPIR-V words.
    pub fn finish(mut self, encoded_rgb: spirv::Word) -> Vec<u32> {
        // Compose vec4(encoded, 1.0) and store into out_color.
        let r = self
            .b
            .composite_extract(self.types.f32_t, None, encoded_rgb, [0])
            .expect("extract r");
        let g = self
            .b
            .composite_extract(self.types.f32_t, None, encoded_rgb, [1])
            .expect("extract g");
        let bz = self
            .b
            .composite_extract(self.types.f32_t, None, encoded_rgb, [2])
            .expect("extract b");
        let one = self.consts.f_one;
        let rgba = self
            .b
            .composite_construct(self.types.vec4, None, [r, g, bz, one])
            .expect("composite_construct rgba");
        self.b
            .store(self.iface.out_color_ptr, rgba, None, [])
            .expect("store out_color");
        self.b.ret().expect("ret");
        self.b.end_function().expect("end_function");

        use rspirv::binary::Assemble;
        self.b.module().assemble()
    }
}
