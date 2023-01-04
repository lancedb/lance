/// Protobuf definitions
#[allow(clippy::all)]
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/lance.format.pb.rs"));
}
