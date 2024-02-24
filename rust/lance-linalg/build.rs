// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::env;

fn main() {
    let rust_toolchain = env::var("RUSTUP_TOOLCHAIN").unwrap();
    if rust_toolchain.starts_with("nightly") {
        // enable the 'nightly' feature flag
        println!("cargo:rustc-cfg=feature=\"nightly\"");
    }

    println!("cargo:rerun-if-changed=src/simd/f16.c");

    if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        // Build a version with NEON
        build_f16_with_flags("neon", &["-mtune=apple-m1"]);
    }

    if cfg!(target_arch = "x86_64") {
        // Build a version with AVX512
        build_f16_with_flags("avx512", &["-march=sapphirerapids", "-ffast-math"]);
        // Build a version with AVX
        build_f16_with_flags("avx2", &["-march=broadwell", "-ffast-math"]);
        // There is no SSE instruction set for f16 -> f32 float conversion
    }

    // Build a version with no flags
    build_f16_with_flags("base", &[]);
}

fn build_f16_with_flags(suffix: &str, flags: &[&str]) {
    let mut builder = cc::Build::new();
    builder
        .compiler("clang")
        .std("c17")
        .file("src/simd/f16.c")
        .flag("-ffast-math")
        .flag("-O3")
        .flag("-Wall")
        .flag("-Werror")
        .flag("-Wextra")
        .flag("-Wpedantic")
        // We pass in the suffix to make sure the symbol names are unique
        .flag(&format!("-DSUFFIX=_{}", suffix));

    for flag in flags {
        builder.flag(flag);
    }

    builder.compile(&format!("f16_{}", suffix));
}
