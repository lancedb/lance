//! On-disk format

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

pub const MAJOR_VERSION: i16 = 0;
pub const MINOR_VERSION: i16 = 1;
pub const MAGIC: &[u8; 4] = b"LANC";
pub const INDEX_MAGIC: &[u8; 8] = b"LANC_IDX";

/// Annotation on a struct that can be converted a Protobuf message.
pub trait ProtoStruct {
    type Proto: Message;
}
