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

use std::ops::Range;

use async_trait::async_trait;
use bytes::Bytes;
use object_store::path::Path;
use prost::Message;
use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::format::*;
use crate::Result;

/// A trait for writing to a file on local file system or object store.
#[async_trait]
pub trait Writer: AsyncWrite + Unpin + Send {
    /// Tell the current offset.
    fn tell(&self) -> usize;
}

#[async_trait]
pub trait LanceWrite: Writer {
    /// Write a protobuf message to the object, and returns the file position of the protobuf.
    async fn write_protobuf(&mut self, msg: &impl Message) -> Result<usize> {
        let offset = self.tell();

        let len = msg.encoded_len();

        self.write_u32_le(len as u32).await?;
        self.write_all(&msg.encode_to_vec()).await?;

        Ok(offset)
    }

    async fn write_struct<
        'b,
        M: Message + From<&'b T>,
        T: ProtoStruct<Proto = M> + Send + Sync + 'b,
    >(
        &mut self,
        obj: &'b T,
    ) -> Result<usize> {
        let msg: M = M::from(obj);
        self.write_protobuf(&msg).await
    }

    /// Write magics to the tail of a file before closing the file.
    async fn write_magics(&mut self, pos: usize) -> Result<()> {
        self.write_i64_le(pos as i64).await?;
        self.write_i16_le(MAJOR_VERSION).await?;
        self.write_i16_le(MINOR_VERSION).await?;
        self.write_all(MAGIC).await?;
        Ok(())
    }
}

#[async_trait]
pub trait Reader: Send + Sync {
    fn path(&self) -> &Path;

    /// Suggest optimal I/O size per storage device.
    fn block_size(&self) -> usize;

    /// Object/File Size.
    async fn size(&self) -> Result<usize>;

    /// Read a range of bytes from the object.
    ///
    /// TODO: change to read_at()?
    async fn get_range(&self, range: Range<usize>) -> Result<Bytes>;
}
