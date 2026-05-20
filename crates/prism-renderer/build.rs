// Compile GLSL shaders → SPIR-V at build time using the system `glslangValidator`.
//
// Source files: crates/prism-renderer/shaders/*.{vert,frag}
// Outputs:      $OUT_DIR/<name>.<stage>.spv
//
// The renderer's pipeline modules `include_bytes!` from $OUT_DIR. If
// glslangValidator isn't installed we fail with a clear message rather
// than producing a broken build.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

// encode.frag is synthesized at runtime via rspirv (see encode_synth/).
// Only the static-GLSL shaders are listed here.
const SHADERS: &[(&str, &str)] = &[
    ("decode.vert", "vert"),
    ("decode.frag", "frag"),
    ("encode.vert", "vert"),
];

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shader_dir = manifest_dir.join("shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=build.rs");

    // Sanity: glslangValidator on PATH.
    if Command::new("glslangValidator")
        .arg("--version")
        .output()
        .is_err()
    {
        panic!(
            "glslangValidator not found on PATH. Install via `pacman -S glslang` \
             (Arch) or `apt install glslang-tools` (Debian/Ubuntu)."
        );
    }

    for (file, stage) in SHADERS {
        let src = shader_dir.join(file);
        let dst = out_dir.join(format!("{file}.spv"));
        compile_one(&src, &dst, stage);
        println!("cargo:rerun-if-changed={}", src.display());
    }
}

fn compile_one(src: &Path, dst: &Path, stage: &str) {
    let output = Command::new("glslangValidator")
        .args(["-V", "-S", stage, "-o"])
        .arg(dst)
        .arg(src)
        .output()
        .expect("invoking glslangValidator");
    if !output.status.success() {
        panic!(
            "glslangValidator failed for {}:\n--- stdout ---\n{}\n--- stderr ---\n{}",
            src.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}
