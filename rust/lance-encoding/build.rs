// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io::Result;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=protos");

    #[cfg(feature = "protoc")]
    // Use vendored protobuf compiler if requested.
    std::env::set_var("PROTOC", protobuf_src::protoc());

    let mut prost_build = prost_build::Config::new();
    prost_build.protoc_arg("--experimental_allow_proto3_optional");
    prost_build.enable_type_names();
    prost_build.bytes(["."]); // Enable Bytes type for all messages to avoid Vec clones.

    // Implement DeepSizeOf so we can keep metadata in cache.
    // Once https://github.com/nhtyy/deepsize2/pull/2 is merged and released,
    // we can use that and just implement DeepSizeOf for `.`
    for path in &[
        "lance.encodings.ColumnEncoding",
        "lance.encodings.Blob",
        "lance.encodings.ZoneIndex",
        "lance.encodings.ArrayEncoding",
        "lance.encodings.Flat",
        "lance.encodings.Nullable",
        "lance.encodings.FixedSizeList",
        "lance.encodings.List",
        "lance.encodings.Struct",
        "lance.encodings.Binary",
        "lance.encodings.Dictionary",
        "lance.encodings.PackedStruct",
        "lance.encodings.SimpleStruct",
        "lance.encodings.Bitpacked",
        "lance.encodings.FixedSizeBinary",
        "lance.encodings.BitpackedForNonNeg",
        "lance.encodings.InlineBitpacking",
        "lance.encodings.OutOfLineBitpacking",
        "lance.encodings.Variable",
        "lance.encodings.PackedStructFixedWidthMiniBlock",
        "lance.encodings.Block",
        "lance.encodings.Rle",
        "lance.encodings.GeneralMiniBlock",
        "lance.encodings.ByteStreamSplit",
        "lance.encodings.Buffer",
        "lance.encodings.Compression",
        "lance.encodings.Nullable.NoNull",
        "lance.encodings.Nullable.AllNull",
        "lance.encodings.Nullable.SomeNull",
        "lance.encodings21.MiniBlockLayout",
        "lance.encodings21.CompressiveEncoding",
        "lance.encodings21.FullZipLayout",
        "lance.encodings21.AllNullLayout",
        "lance.encodings21.BlobLayout",
        "lance.encodings21.PageLayout",
        "lance.encodings21.BufferCompression",
        "lance.encodings21.Flat",
        "lance.encodings21.Variable",
        "lance.encodings21.OutOfLineBitpacking",
        "lance.encodings21.InlineBitpacking",
        "lance.encodings21.Dictionary",
        "lance.encodings21.Rle",
        "lance.encodings21.FixedSizeList",
        "lance.encodings21.PackedStruct",
        "lance.encodings21.General",
        "lance.encodings21.ByteStreamSplit",
    ] {
        prost_build.type_attribute(path, "#[derive(deepsize::DeepSizeOf)]");
    }
    for path in &[
        "lance.encodings.ArrayEncoding.array_encoding",
        "lance.encodings.ColumnEncoding.column_encoding",
        "lance.encodings.Nullable.nullability",
        "lance.encodings21.FullZipLayout.details",
        "lance.encodings21.PageLayout.layout",
        "lance.encodings21.CompressiveEncoding.compression",
    ] {
        prost_build.enum_attribute(path, "#[derive(deepsize::DeepSizeOf)]");
    }

    prost_build.compile_protos(&["./protos/encodings_v2_0.proto"], &["./protos"])?;
    prost_build.compile_protos(&["./protos/encodings_v2_1.proto"], &["./protos"])?;

    Ok(())
}
