// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use arrow_buffer::ToByteSlice;
use prost::Message;
use snafu::{location, Location};
use uuid::Uuid;

use crate::{Error, Result};

/// Protobuf definitions for Lance Format
pub mod pb {
    #![allow(clippy::all)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(unused)]
    #![allow(improper_ctypes)]
    #![allow(clippy::upper_case_acronyms)]
    #![allow(clippy::use_self)]
    include!(concat!(env!("OUT_DIR"), "/lance.format.pb.rs"));
}

pub const MAJOR_VERSION: i16 = 0;
pub const MINOR_VERSION: i16 = 1;
pub const MAGIC: &[u8; 4] = b"LANC";
pub const INDEX_MAGIC: &[u8; 8] = b"LANC_IDX";

pub trait ProtoStruct {
    type Proto: Message;
}

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
