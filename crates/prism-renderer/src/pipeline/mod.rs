//! Vulkan pipeline state objects for the two-pass renderer.
//!
//! - `decode`: per-element draw, source primaries → BT.2020 linear nits
//!   into the fp16 intermediate image.
//! - `encode`: full-screen tri, BT.2020 intermediate → per-output target
//!   color space + transfer, written to the scanout image.
//!
//! Push-constant blocks (`DecodePush`, `EncodePush`) are mirrored exactly
//! by the GLSL `Push` struct in the corresponding shader; keep the field
//! ordering and types in sync. The shaders use `mat4` for the matrix even
//! though we logically only need mat3, to avoid std430 alignment pitfalls.

pub mod deband;
pub mod decode;
pub mod encode;

use ash::vk;

use crate::device::Device;
use crate::error::{Result, VkResultExt};

/// Build a `VkShaderModule` from raw SPIR-V bytes.
pub(crate) fn shader_module(device: &Device, spv: &[u8]) -> Result<vk::ShaderModule> {
    // SPIR-V is little-endian uint32; transmute 4-byte aligned bytes.
    assert!(spv.len() % 4 == 0, "SPIR-V byte length not 4-aligned");
    let code: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let info = vk::ShaderModuleCreateInfo::default().code(&code);
    unsafe { device.raw.create_shader_module(&info, None) }.vk_ctx("create_shader_module")
}
