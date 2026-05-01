// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::env;
use std::io::Result;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=protos");
    println!("cargo:rerun-if-changed=csrc/libsais.c");
    println!("cargo:rerun-if-changed=csrc/libsais.h");

    #[cfg(feature = "protoc")]
    // Use vendored protobuf compiler if requested.
    unsafe {
        std::env::set_var("PROTOC", protobuf_src::protoc());
    }

    let mut prost_build = prost_build::Config::new();
    prost_build.protoc_arg("--experimental_allow_proto3_optional");
    prost_build.enable_type_names();
    prost_build.compile_protos(
        &["./protos/index.proto", "./protos/index_old.proto"],
        &["./protos"],
    )?;

    // Compile vendored libsais (Ilya Grebnov's SA-IS implementation)
    // for fast O(N) suffix array construction.
    cc::Build::new()
        .file("csrc/libsais.c")
        .include("csrc")
        .opt_level(3)
        .flag_if_supported("-march=native")
        .flag_if_supported("-mtune=native")
        .compile("sais");

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

    Ok(())
}
