use std::io::Result;

use rustc_version::{version_meta, Channel};

fn main() -> Result<()> {
    match version_meta().unwrap().channel {
        Channel::Nightly => {
            println!("cargo:rustc-cfg=nightly");
        }
        _ => {},
    }

    println!("cargo:rerun-if-changed=protos");

    let mut prost_build = prost_build::Config::new();
    prost_build.protoc_arg("--experimental_allow_proto3_optional");
    prost_build.compile_protos(&["./protos/index.proto"], &["./protos"])?;

    Ok(())
}
