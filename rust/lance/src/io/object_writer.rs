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

use std::pin::Pin;
use std::task::{Context, Poll};

use arrow_array::Array;
use object_store::{path::Path, MultipartId};
use pin_project::pin_project;
use prost::Message;
use snafu::{location, Location};
use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::encodings::plain::PlainEncoder;
use crate::format::{ProtoStruct, MAGIC, MAJOR_VERSION, MINOR_VERSION};
use crate::io::ObjectStore;
use crate::{Error, Result};

/// AsyncWrite with the capability to tell the position the data is written.
///
#[pin_project]
pub struct ObjectWriter {
    store: ObjectStore,

    // TODO: wrap writer with a BufWriter.
    #[pin]
    writer: Box<dyn AsyncWrite + Unpin + Send>,

    pub(crate) multipart_id: MultipartId,

    cursor: usize,
}

impl ObjectWriter {
    pub async fn new(object_store: &ObjectStore, path: &Path) -> Result<Self> {
        let (multipart_id, writer) =
            object_store
                .inner
                .put_multipart(path)
                .await
                .map_err(|e| Error::IO {
                    message: format!("failed to create object writer for {}: {}", path, e),
                    location: location!(),
                })?;

        Ok(Self {
            store: object_store.clone(),
            writer,
            multipart_id,
            cursor: 0,
        })
    }

    /// Tell the current position (file size).
    pub fn tell(&self) -> usize {
        self.cursor
    }

    /// Write a protobuf message to the object, and returns the file position of the protobuf.
    pub async fn write_protobuf(&mut self, msg: &impl Message) -> Result<usize> {
        let offset = self.tell();

        let len = msg.encoded_len();

        self.write_u32_le(len as u32).await?;
        self.write_all(&msg.encode_to_vec()).await?;

        Ok(offset)
    }

    pub async fn write_struct<'b, M: Message + From<&'b T>, T: ProtoStruct<Proto = M> + 'b>(
        &mut self,
        obj: &'b T,
    ) -> Result<usize> {
        let msg: M = M::from(obj);
        self.write_protobuf(&msg).await
    }

    /// Write arrays, as single array, using plain encoding
    ///
    /// Returns the file position if success.
    pub async fn write_plain_encoded_array(&mut self, arrays: &[&dyn Array]) -> Result<usize> {
        if arrays.is_empty() {
            return Ok(self.tell());
        }
        let data_type = arrays[0].data_type();
        let mut encoder = PlainEncoder::new(self, data_type);
        encoder.encode(arrays).await
    }

    /// Write magics to the tail of a file before closing the file.
    pub async fn write_magics(&mut self, pos: usize) -> Result<()> {
        self.write_i64_le(pos as i64).await?;
        self.write_i16_le(MAJOR_VERSION).await?;
        self.write_i16_le(MINOR_VERSION).await?;
        self.write_all(MAGIC).await?;
        Ok(())
    }

    pub fn sizeof_magic() -> usize {
        16
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        Ok(self.writer.shutdown().await?)
    }
}

impl AsyncWrite for ObjectWriter {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        let mut this = self.project();
        this.writer.as_mut().poll_write(cx, buf).map_ok(|n| {
            *this.cursor += n;
            n
        })
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        self.project().writer.as_mut().poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        self.project().writer.as_mut().poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    use object_store::path::Path;
    use tokio::io::AsyncWriteExt;

    use crate::format::Metadata;
    use crate::io::object_reader::{read_struct, CloudObjectReader};
    use crate::io::ObjectStore;

    use super::*;

    #[tokio::test]
    async fn test_write() {
        let store = ObjectStore::memory();

        let mut object_writer = ObjectWriter::new(&store, &Path::from("/foo"))
            .await
            .unwrap();
        assert_eq!(object_writer.tell(), 0);

        let mut buf = Vec::<u8>::new();
        buf.resize(256, 0);
        assert_eq!(object_writer.write(buf.as_slice()).await.unwrap(), 256);
        assert_eq!(object_writer.tell(), 256);

        assert_eq!(object_writer.write(buf.as_slice()).await.unwrap(), 256);
        assert_eq!(object_writer.tell(), 512);

        assert_eq!(object_writer.write(buf.as_slice()).await.unwrap(), 256);
        assert_eq!(object_writer.tell(), 256 * 3);

        object_writer.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_write_proto_structs() {
        let store = ObjectStore::memory();
        let path = Path::from("/foo");

        let mut object_writer = ObjectWriter::new(&store, &path).await.unwrap();
        assert_eq!(object_writer.tell(), 0);

        let mut metadata = Metadata {
            manifest_position: Some(100),
            ..Default::default()
        };
        metadata.batch_offsets.extend([1, 2, 3, 4]);

        let pos = object_writer.write_struct(&metadata).await.unwrap();
        assert_eq!(pos, 0);
        object_writer.shutdown().await.unwrap();

        let object_reader = CloudObjectReader::new(&store, path, 1024).unwrap();
        let actual: Metadata = read_struct(&object_reader, pos).await.unwrap();
        assert_eq!(metadata, actual);
    }
}
