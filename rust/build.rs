use std::io::Result;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=protos");
    println!("cargo:rerun-if-changed=../protos");

    prost_build::compile_protos(
        &["protos/index.proto", "../protos/format.proto"],
        &["protos/", "../protos"],
    )?;
    Ok(())
}
