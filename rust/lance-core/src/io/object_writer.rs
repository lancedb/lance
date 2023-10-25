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

use async_trait::async_trait;
use object_store::{path::Path, MultipartId, ObjectStore};
use pin_project::pin_project;
use snafu::{location, Location};
use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::{io::Writer, Error, Result};

/// AsyncWrite with the capability to tell the position the data is written.
///
#[pin_project]
pub struct ObjectWriter {
    // TODO: wrap writer with a BufWriter.
    #[pin]
    writer: Box<dyn AsyncWrite + Send + Unpin>,

    // TODO: pub(crate)
    pub multipart_id: MultipartId,

    cursor: usize,
}

impl ObjectWriter {
    pub async fn new(object_store: &dyn ObjectStore, path: &Path) -> Result<Self> {
        let (multipart_id, writer) =
            object_store
                .put_multipart(path)
                .await
                .map_err(|e| Error::IO {
                    message: format!("failed to create object writer for {}: {}", path, e),
                    location: location!(),
                })?;

        Ok(Self {
            writer,
            multipart_id,
            cursor: 0,
        })
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        Ok(self.writer.as_mut().shutdown().await?)
    }
}

#[async_trait]
impl Writer for ObjectWriter {
    async fn tell(&mut self) -> Result<usize> {
        Ok(self.cursor)
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
    use std::sync::Arc;

    use object_store::{memory::InMemory, path::Path};
    use tokio::io::AsyncWriteExt;

    use crate::{
        format::Metadata,
        io::{read_struct, CloudObjectReader, WriteExt},
    };

    use super::*;

    #[tokio::test]
    async fn test_write() {
        let store = InMemory::new();

        let mut object_writer = ObjectWriter::new(&store, &Path::from("/foo"))
            .await
            .unwrap();
        assert_eq!(object_writer.tell().await.unwrap(), 0);

        let buf = vec![0; 256];
        assert_eq!(object_writer.write(buf.as_slice()).await.unwrap(), 256);
        assert_eq!(object_writer.tell().await.unwrap(), 256);

        assert_eq!(object_writer.write(buf.as_slice()).await.unwrap(), 256);
        assert_eq!(object_writer.tell().await.unwrap(), 512);

        assert_eq!(object_writer.write(buf.as_slice()).await.unwrap(), 256);
        assert_eq!(object_writer.tell().await.unwrap(), 256 * 3);

        object_writer.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_write_proto_structs() {
        let store = InMemory::new();
        let path = Path::from("/foo");

        let mut object_writer = ObjectWriter::new(&store, &path).await.unwrap();
        assert_eq!(object_writer.tell().await.unwrap(), 0);

        let mut metadata = Metadata {
            manifest_position: Some(100),
            ..Default::default()
        };
        metadata.batch_offsets.extend([1, 2, 3, 4]);

        let pos = object_writer.write_struct(&metadata).await.unwrap();
        assert_eq!(pos, 0);
        object_writer.shutdown().await.unwrap();

        let object_reader = CloudObjectReader::new(Arc::new(store), path, 1024).unwrap();
        let actual: Metadata = read_struct(&object_reader, pos).await.unwrap();
        assert_eq!(metadata, actual);
    }
}
