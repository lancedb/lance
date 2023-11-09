fn main() {
    println!("cargo:rerun-if-changed=src/simd/f16.c");

    if cfg!(all(target_os = "macos", target_feature = "neon")) {
        cc::Build::new()
            .compiler("clang")
            .file("src/simd/f16.c")
            .flag("-mcpu=apple-m1")
            .flag("-O3")
            .define("LANES", "4")
            .compile("f16");
    }

    if cfg!(all(target_os = "linux", target_feature = "avx512fp16")) {
        // No fp16 without AVX512fp16
        cc::Build::new()
            .compiler("clang")
            .file("src/simd/f16.c")
            .flag("-march=icelake-server")
            .flag("-mavx512fp16")
            .flag("-mllvm")
            .flag("-unroll-count=8")
            .flag("-O3")
            .compile("f16");
    }
}
