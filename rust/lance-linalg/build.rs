fn main() {
    println!("cargo:rerun-if-changed=src/simd/f16.c");

    if !cfg!(all(target_os = "macos",  target_feature = "neon")) {
        cc::Build::new()
            .compiler("clang")
            .file("src/simd/f16.c")
            .flag("-mcpu=apple-m1")
            .flag("-O3")
            .compile("f16");
    }

    if !cfg!(target_os = "linux") {
        let mut build = cc::Build::new();
        build
            .compiler("clang")
            .file("src/simd/f16.c")
            .flag("-O3");

        if cfg!(target_feature = "avx512fp16") {
            build.flag("-mavx512fp16");
        }
        build.compile("f16")
    }
}
