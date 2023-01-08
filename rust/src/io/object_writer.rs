// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::io::Error;
use std::pin::Pin;
use std::task::{Context, Poll};

use pin_project::pin_project;
use prost::Message;
use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::format::ProtoStruct;

/// AsyncWrite with the capability to tell the position the data is written.
///
#[pin_project]
pub struct ObjectWriter<'a> {
    // TODO: wrap writer with a BufWriter.
    #[pin]
    writer: &'a mut (dyn AsyncWrite + Unpin + Send),
    cursor: usize,
}

impl<'a> ObjectWriter<'a> {
    pub fn new(writer: &'a mut (dyn AsyncWrite + Unpin + Send)) -> ObjectWriter<'a> {
        ObjectWriter { writer, cursor: 0 }
    }

    /// Tell the current position (file size).
    pub fn tell(&self) -> usize {
        self.cursor
    }

    /// Write a protobuf message to the object, and returns the file position of the protobuf.
    pub async fn write_protobuf(&mut self, msg: &impl Message) -> Result<usize, Error> {
        let offset = self.tell();

        let len = msg.encoded_len();

        self.write_u32_le(len as u32).await?;
        self.write_all(&msg.encode_to_vec()).await?;

        Ok(offset)
    }

    pub async fn write_struct<'b, M: Message + From<&'b T>, T: ProtoStruct<Proto = M> + 'b>(
        &mut self,
        obj: &'b T,
    ) -> Result<usize, Error> {
        let msg: M = M::from(obj);
        self.write_protobuf(&msg).await
    }
}

impl AsyncWrite for ObjectWriter<'_> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, Error>> {
        let mut this = self.project();
        *this.cursor += buf.len();
        this.writer.as_mut().poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
        self.project().writer.as_mut().poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
        self.project().writer.as_mut().poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {

    use object_store::path::Path;
    use tokio::io::AsyncWriteExt;

    use crate::format::Metadata;
    use crate::io::object_reader::ObjectReader;
    use crate::io::ObjectStore;

    use super::*;

    #[tokio::test]
    async fn test_write() {
        let store = ObjectStore::new(":memory:").unwrap();
        let (_, mut writer) = store
            .inner
            .put_multipart(&Path::from("/foo"))
            .await
            .unwrap();

        let mut object_writer = ObjectWriter::new(writer.as_mut());
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

        assert_eq!(
            store.inner.head(&Path::from("/foo")).await.unwrap().size,
            256 * 3
        );
    }

    #[tokio::test]
    async fn test_write_proto_structs() {
        let store = ObjectStore::new(":memory:").unwrap();
        let path = Path::from("/foo");
        let (_, mut writer) = store.inner.put_multipart(&path).await.unwrap();

        let mut object_writer = ObjectWriter::new(writer.as_mut());
        assert_eq!(object_writer.tell(), 0);

        let mut metadata = Metadata::default();
        metadata.manifest_position = Some(100);
        metadata.batch_offsets.extend([1, 2, 3, 4]);

        let pos = object_writer.write_struct(&metadata).await.unwrap();
        assert_eq!(pos, 0);
        object_writer.shutdown().await.unwrap();

        let mut object_reader = ObjectReader::new(&store, path, 1024).unwrap();
        let actual: Metadata = object_reader.read_struct(pos).await.unwrap();
        assert_eq!(metadata, actual);
    }
}
