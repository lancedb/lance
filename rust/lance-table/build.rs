// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io::Result;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=protos");

    #[cfg(feature = "protoc")]
    // Use vendored protobuf compiler if requested.
    unsafe {
        std::env::set_var("PROTOC", protobuf_src::protoc());
    }

    // `mut` is used only when the experimental feature adds to the list below.
    #[allow(unused_mut)]
    let mut protos = vec![
        "./protos/table.proto",
        "./protos/transaction.proto",
        "./protos/rowids.proto",
    ];
    // EXPERIMENTAL: only compile the action-based transaction messages when the
    // unstable feature is enabled, so they are absent from default builds.
    #[cfg(feature = "unstable-action-transactions")]
    protos.push("./protos/transaction_experimental.proto");

    let mut prost_build = prost_build::Config::new();
    prost_build.extern_path(".lance.file", "::lance_file::format::pb");
    prost_build.protoc_arg("--experimental_allow_proto3_optional");
    prost_build.enable_type_names();
    prost_build.compile_protos(&protos, &["./protos"])?;

    Ok(())
}
