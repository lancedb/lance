mod fragment;
mod manifest;
mod metadata;
mod page_table;
pub use fragment::Fragment;
pub use manifest::Manifest;
pub use metadata::Metadata;
pub use page_table::{PageInfo, PageTable};

use prost::Message;

/// Protobuf definitions
#[allow(clippy::all)]
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/lance.format.pb.rs"));
}

/// Annotation on a struct that can be converted a Protobuf message.
pub trait ProtoStruct {
    type Proto: Message;
}
