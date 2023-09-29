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

//! I/O utilities.

use std::io::{Error, ErrorKind, Result};
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use arrow_array::UInt32Array;
use async_trait::async_trait;
use byteorder::{ByteOrder, LittleEndian};
use prost::bytes::Bytes;
use prost::Message;
use tokio::io::{AsyncWrite, AsyncWriteExt};

pub mod commit;
pub(crate) mod deletion;
pub(crate) mod exec;
pub mod local;
pub mod object_reader;
pub mod object_store;
pub mod object_writer;
pub(crate) mod reader;
pub(crate) mod remap;
mod stream;
mod writer;

use crate::format::{ProtoStruct, INDEX_MAGIC, MAGIC};

pub use self::object_store::ObjectStore;
pub use deletion::deletion_file_path;
pub use reader::read_manifest;
pub use reader::FileReader;
pub use stream::RecordBatchStream;
pub use writer::*;

#[async_trait]
pub trait AsyncWriteProtoExt {
    /// Write footer with the offset to the root metadata block.
    async fn write_footer(&mut self, offset: u64) -> Result<()>;
}

#[async_trait]
impl<T: AsyncWrite + Unpin + std::marker::Send> AsyncWriteProtoExt for T {
    async fn write_footer(&mut self, offset: u64) -> Result<()> {
        self.write_u64_le(offset).await?;
        self.write_all(INDEX_MAGIC).await?;
        Ok(())
    }
}

pub fn read_metadata_offset(bytes: &Bytes) -> Result<usize> {
    let len = bytes.len();
    if len < 16 {
        return Err(Error::new(
            ErrorKind::Interrupted,
            "does not have sufficient data",
        ));
    }
    let offset_bytes = bytes.slice(len - 16..len - 8);
    Ok(LittleEndian::read_u64(offset_bytes.as_ref()) as usize)
}

/// Read protobuf from a buffer.
pub fn read_message_from_buf<M: Message + Default>(buf: &Bytes) -> Result<M> {
    let msg_len = LittleEndian::read_u32(buf) as usize;
    Ok(M::decode(&buf[4..4 + msg_len])?)
}

/// Read a Protobuf-backed struct from a buffer.
pub fn read_struct_from_buf<M: Message + Default, T: ProtoStruct<Proto = M> + From<M>>(
    buf: &Bytes,
) -> Result<T> {
    let msg: M = read_message_from_buf(buf)?;
    Ok(T::from(msg))
}

/// Parameter to be used to read a batch.
#[derive(Debug, Clone)]
pub(crate) enum ReadBatchParams {
    Range(Range<usize>),

    RangeFull,

    RangeTo(RangeTo<usize>),

    RangeFrom(RangeFrom<usize>),

    Indices(UInt32Array),
}

/// Default of ReadBatchParams is reading the full batch.
impl Default for ReadBatchParams {
    fn default() -> Self {
        Self::RangeFull
    }
}

impl From<&[u32]> for ReadBatchParams {
    fn from(value: &[u32]) -> Self {
        Self::Indices(UInt32Array::from_iter_values(value.iter().copied()))
    }
}

impl From<UInt32Array> for ReadBatchParams {
    fn from(value: UInt32Array) -> Self {
        Self::Indices(value)
    }
}

impl From<RangeFull> for ReadBatchParams {
    fn from(_: RangeFull) -> Self {
        Self::RangeFull
    }
}

impl From<Range<usize>> for ReadBatchParams {
    fn from(r: Range<usize>) -> Self {
        Self::Range(r)
    }
}

impl From<RangeTo<usize>> for ReadBatchParams {
    fn from(r: RangeTo<usize>) -> Self {
        Self::RangeTo(r)
    }
}

impl From<RangeFrom<usize>> for ReadBatchParams {
    fn from(r: RangeFrom<usize>) -> Self {
        Self::RangeFrom(r)
    }
}

impl From<&Self> for ReadBatchParams {
    fn from(params: &Self) -> Self {
        params.clone()
    }
}
