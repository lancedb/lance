// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io::Result;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=protos");

    let mut prost_build = prost_build::Config::new();
    prost_build.extern_path(".lance.encodings", "::lance_encoding::format::pb");
    prost_build.protoc_arg("--experimental_allow_proto3_optional");
    prost_build.enable_type_names();
    prost_build.compile_protos(&["./protos/encodings-df.proto"], &["./protos"])?;

    Ok(())
}
