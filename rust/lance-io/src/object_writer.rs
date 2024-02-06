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
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures::FutureExt;
use object_store::{path::Path, MultipartId, ObjectStore};
use pin_project::pin_project;
use snafu::{location, Location};
use tokio::io::{AsyncWrite, AsyncWriteExt};

use lance_core::{Error, Result};

use crate::traits::Writer;

/// AsyncWrite with the capability to tell the position the data is written.
///
#[pin_project]
pub struct ObjectWriter {
    /// This writer is behind a Mutex because it is used both by the caller
    /// to write data and by the background task to flush the data. The background
    /// task never holds the mutex for longer than it takes to poll the flush
    /// future once, so it should never block the caller for long.
    ///
    /// Note: this is a std Mutex. It MUST NOT be held across await points.
    #[pin]
    writer: Arc<Mutex<Pin<Box<dyn AsyncWrite + Send + Unpin>>>>,

    /// A task that flushes the data every 500ms. This is to make sure that the
    /// futures within the writer are polled at least every 500ms. This is
    /// necessary because the internal writer buffers data and holds up to 10
    /// write request futures in FuturesUnordered. These futures only make
    /// progress when polled, and if they are not polled for a while, they can
    /// cause the requests to timeout.
    background_flusher: tokio::task::JoinHandle<()>,

    /// When calling flush(), the background task may receive a ready error.
    /// This channel is used to send the error to the main task.
    background_error: tokio::sync::oneshot::Receiver<std::io::Error>,

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

        let writer = Arc::new(Mutex::new(Pin::new(writer)));

        // If background task encounters an error, we use a channel to send the error
        // to the main task.
        let (error_sender, background_error) = tokio::sync::oneshot::channel();

        let writer_ref = writer.clone();
        let background_flusher = tokio::task::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                match writer_ref.lock().unwrap().flush().now_or_never() {
                    None => continue,
                    Some(Ok(_)) => continue,
                    Some(Err(e)) => {
                        let _ = error_sender.send(e);
                        break;
                    }
                }
            }
        });

        Ok(Self {
            writer,
            background_flusher,
            background_error,
            multipart_id,
            cursor: 0,
        })
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        Ok(AsyncWriteExt::shutdown(self).await?)
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
        let this = self.project();
        if let Ok(err) = this.background_error.try_recv() {
            return Poll::Ready(Err(err));
        }
        let mut writer = this.writer.lock().unwrap();
        writer.as_mut().poll_write(cx, buf).map_ok(|n| {
            *this.cursor += n;
            n
        })
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        let this = self.project();
        if let Ok(err) = this.background_error.try_recv() {
            return Poll::Ready(Err(err));
        }
        let mut writer = this.writer.lock().unwrap();
        writer.as_mut().poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        let this = self.project();
        if let Ok(err) = this.background_error.try_recv() {
            return Poll::Ready(Err(err));
        }
        let mut writer = this.writer.lock().unwrap();
        writer.as_mut().poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {

    use object_store::{memory::InMemory, path::Path};
    use tokio::io::AsyncWriteExt;

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
}
