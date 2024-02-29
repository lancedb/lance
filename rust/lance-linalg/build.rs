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

    if cfg!(any(not(feature = "fp16kernels"), target_os = "windows")) {
        // We only compile the f16 kernels if the feature is enabled
        // MSVC does not support the f16 type, so we also skip it on Windows.
        return;
    }

    if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        // Build a version with NEON
        build_f16_with_flags("neon", &["-mtune=apple-m1"]).unwrap();
    } else if cfg!(all(target_arch = "aarch64", target_os = "linux")) {
        // Build a version with NEON
        build_f16_with_flags("neon", &["-march=armv8.2-a+fp16"]).unwrap();
    }

    if cfg!(target_arch = "x86_64") {
        // Build a version with AVX512
        if let Err(err) = build_f16_with_flags("avx512", &["-march=sapphirerapids", "-mavx512fp16"])
        {
            // It's likely the compiler doesn't support the sapphirerapids architecture
            // Clang 12 and GCC 11 are the first versions with sapphire rapids support
            eprintln!("Skipping Sapphire Rapids build due to error: {}", err);
        } else {
            // We create a special cfg so that we can detect we have in fact
            // generated the AVX512 version of the f16 kernels.
            println!("cargo:rustc-cfg=kernel_suppport=\"avx512\"");
        };
        // Build a version with AVX
        // While GCC doesn't have support for _Float16 until GCC 12, clang
        // has support for __fp16 going back to at least clang 6.
        // We use haswell since it's the oldest CPUs on AWS.
        build_f16_with_flags("avx2", &["-march=haswell"]).unwrap();
        // There is no SSE instruction set for f16 -> f32 float conversion
    }
}

fn build_f16_with_flags(suffix: &str, flags: &[&str]) -> Result<(), cc::Error> {
    let mut builder = cc::Build::new();
    builder
        // TODO: why specify the compiler?
        // .compiler("clang")
        .std("c17")
        .file("src/simd/f16.c")
        .flag("-ffast-math")
        .flag("-funroll-loops")
        .flag("-O3")
        .flag("-Wall")
        // .flag("-Werror")
        .flag("-Wextra")
        // Pedantic will complain about _Float16 in some versions of GCC
        // .flag("-Wpedantic")
        // We pass in the suffix to make sure the symbol names are unique
        .flag(&format!("-DSUFFIX=_{}", suffix));

    for flag in flags {
        builder.flag(flag);
    }

    builder.try_compile(&format!("f16_{}", suffix))
}
