use std::io::Result;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=protos");

    let mut prost_build = prost_build::Config::new();
    prost_build.extern_path(".lance.file", "::lance_file::format::pb");
    prost_build.protoc_arg("--experimental_allow_proto3_optional");
    prost_build.compile_protos(
        &["./protos/table.proto", "./protos/transaction.proto"],
        &["./protos"],
    )?;

    Ok(())
}
