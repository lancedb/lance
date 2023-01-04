use std::io::Result;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=protos");

    prost_build::compile_protos(&["protos/index.proto"], &["protos/"])?;
    Ok(())
}
