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

    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        cc::Build::new()
            .compiler("clang")
            .file("src/simd/f16.c")
            .flag("-mtune=apple-m1")
            .flag("-O3")
            .flag("-Wall")
            .flag("-Werror")
            .flag("-Wextra")
            .flag("-Wpedantic")
            .compile("f16");
    }

    if cfg!(all(target_os = "linux", feature = "avx512fp16")) {
        // No fp16 without AVX512fp16
        cc::Build::new()
            .compiler("clang")
            .std("c17")
            .file("src/simd/f16.c")
            .flag("-march=sapphirerapids")
            .flag("-mavx512f")
            .flag("-mavx512vl")
            .flag("-mavx512bw")
            .flag("-mavx512vnni")
            .flag("-mavx512fp16")
            .flag("-O3")
            .flag("-Wall")
            .flag("-Werror")
            .flag("-Wextra")
            .flag("-Wpedantic")
            .compile("f16");
    }
}
