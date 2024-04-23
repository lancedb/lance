// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::env;
use std::io::Result;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=protos");

    let mut prost_build = prost_build::Config::new();
    prost_build.protoc_arg("--experimental_allow_proto3_optional");
    prost_build.compile_protos(&["./protos/index.proto"], &["./protos"])?;

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
