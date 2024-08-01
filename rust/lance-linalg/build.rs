// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::env;

fn main() -> Result<(), String> {
    let rust_toolchain = env::var("RUSTUP_TOOLCHAIN")
        .or_else(|e| match e {
            env::VarError::NotPresent => Ok("stable".into()),
            e => Err(e),
        })
        .unwrap();
    if rust_toolchain.starts_with("nightly") {
        // enable the 'nightly' feature flag
        println!("cargo:rustc-cfg=feature=\"nightly\"");
    }

    println!("cargo:rerun-if-changed=src/simd/f16.c");

    if cfg!(not(feature = "fp16kernels")) {
        println!(
            "cargo:warning=fp16kernels feature is not enabled, skipping build of fp16 kernels"
        );
        return Ok(());
    }

    if cfg!(target_os = "windows") {
        println!(
            "cargo:warning=fp16 kernels are not supported on Windows. Skipping compilation of kernels."
        );
        return Ok(());
    }

    if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        // Build a version with NEON
        build_f16_with_flags("neon", &["-mtune=apple-m1"]).unwrap();
    } else if cfg!(all(target_arch = "aarch64", target_os = "linux")) {
        // Build a version with NEON
        build_f16_with_flags("neon", &["-march=armv8.2-a+fp16"]).unwrap();
    } else if cfg!(target_arch = "x86_64") {
        // Build a version with AVX512
        if let Err(err) = build_f16_with_flags("avx512", &["-march=sapphirerapids", "-mavx512fp16"])
        {
            // It's likely the compiler doesn't support the sapphirerapids architecture
            // Clang 12 and GCC 11 are the first versions with sapphire rapids support
            println!(
                "cargo:warning=Skipping build of AVX-512 fp16 kernels.  Clang/GCC too old or compiler does not support sapphirerapids architecture.  Error: {}",
                err
            );
        } else {
            // We create a special cfg so that we can detect we have in fact
            // generated the AVX512 version of the f16 kernels.
            println!("cargo:rustc-cfg=kernel_suppport=\"avx512\"");
        };
        // Build a version with AVX
        // While GCC doesn't have support for _Float16 until GCC 12, clang
        // has support for __fp16 going back to at least clang 6.
        // We use haswell since it's the oldest CPUs on AWS.
        if let Err(err) = build_f16_with_flags("avx2", &["-march=haswell"]) {
            return Err(format!("Unable to build AVX2 f16 kernels.  Please use Clang >= 6 or GCC >= 12 or remove the fp16kernels feature.  Received error: {}", err));
        };
        // There is no SSE instruction set for f16 -> f32 float conversion
    } else if cfg!(target_arch = "loongarch64") {
        // Build a version with LSX and LASX
        build_f16_with_flags("lsx", &["-mlsx"]).unwrap();
        build_f16_with_flags("lasx", &["-mlasx"]).unwrap();
    } else {
        return Err("Unable to build f16 kernels on given target_arch.  Please use x86_64 or aarch64 or remove the fp16kernels feature".to_string());
    }
    Ok(())
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
