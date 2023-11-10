fn main() {
    println!("cargo:rerun-if-changed=src/simd/f16.c");

    if cfg!(all(target_os = "macos", target_feature = "neon")) {
        cc::Build::new()
            .compiler("clang")
            .file("src/simd/f16.c")
            .flag("-mcpu=apple-m1")
            .flag("-mtune=apple-m1")
            .flag("-ffast-math")
            .flag("-O3")
            .flag("-Wall")
            .flag("-Werror")
            .flag("-Wextra")
            .flag("-Wpedantic")
            // .flag("-mfp16")
            .compile("f16");
    }

    if cfg!(all(target_os = "linux", feature = "avx512fp16")) {
        // No fp16 without AVX512fp16
        cc::Build::new()
            .compiler("clang")
            .std("c17")
            .file("src/simd/f16.c")
            .flag("-march=sapphirerapids")
            .flag("-ffast-math")
            .flag("-mavx512f")
            .flag("-mavx512vl")
            .flag("-mavx512bw")
            .flag("-mavx512fp16")
            .flag("-O3")
            .flag("-Wall")
            .flag("-Werror")
            .flag("-Wextra")
            .flag("-Wpedantic")
            .compile("f16");
    }
}
