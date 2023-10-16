//! On-disk format

use arrow_buffer::ToByteSlice;
use uuid::Uuid;

mod fragment;
mod index;
mod manifest;
mod metadata;
mod page_table;
use crate::{Error, Result};
pub use fragment::*;
pub use index::Index;
pub use manifest::Manifest;
pub use metadata::{Metadata, StatisticsMetadata};
pub use page_table::{PageInfo, PageTable};
use snafu::{location, Location};

pub use lance_core::format::*;
/// Protobuf definitions

impl TryFrom<&pb::Uuid> for Uuid {
    type Error = Error;

    fn try_from(p: &pb::Uuid) -> Result<Self> {
        if p.uuid.len() != 16 {
            return Err(Error::IO {
                message: "Protobuf UUID is malformed".to_string(),
                location: location!(),
            });
        }
        let mut buf: [u8; 16] = [0; 16];
        buf.copy_from_slice(p.uuid.to_byte_slice());
        Ok(Self::from_bytes(buf))
    }
}

impl From<&Uuid> for pb::Uuid {
    fn from(value: &Uuid) -> Self {
        Self {
            uuid: value.into_bytes().to_vec(),
        }
    }
}
