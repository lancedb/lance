//! On-disk format

use arrow_buffer::ToByteSlice;
use prost::Message;
use uuid::Uuid;

mod fragment;
mod index;
mod manifest;
mod metadata;
mod page_table;
use crate::{Error, Result};
pub use fragment::Fragment;
pub use index::Index;
pub use manifest::Manifest;
pub use metadata::Metadata;
pub use page_table::{PageInfo, PageTable};

/// Protobuf definitions
#[allow(clippy::all)]
pub mod pb {
    #![allow(clippy::all)]
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

impl TryFrom<&pb::Uuid> for Uuid {
    type Error = Error;

    fn try_from(p: &pb::Uuid) -> Result<Self> {
        if p.uuid.len() != 16 {
            return Err(Error::IO("Protobuf UUID is malformed".to_string()));
        }
        let mut buf: [u8; 16] = [0; 16];
        buf.copy_from_slice(p.uuid.to_byte_slice());
        Ok(Uuid::from_bytes(buf))
    }
}

impl From<&Uuid> for pb::Uuid {
    fn from(value: &Uuid) -> Self {
        Self {
            uuid: value.into_bytes().to_vec(),
        }
    }
}
