// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{pin::Pin, sync::Arc, task::Poll};

use futures::{Stream, StreamExt};
use object_store::{path::Path, ObjectMeta, ObjectStore};
use tokio::task::JoinHandle;
use tracing::Instrument;

/// ObjectStore::list() and ObjectStore::list_with_offset() return a stream
/// where the lifetime is tied to the object store. This makes it hard to wrap.
/// So here we put it inside a tokio task and return a channel receiver.
struct StaticListStream {
    rx: tokio::sync::mpsc::Receiver<Result<ObjectMeta, object_store::Error>>,
    handle: JoinHandle<()>,
}

impl StaticListStream {
    fn new(object_store: Arc<dyn ObjectStore>, prefix: Option<Path>, offset: Option<Path>) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let handle = tokio::spawn(
            (async move {
                let mut stream = if let Some(offset) = offset {
                    object_store.list_with_offset(prefix.as_ref(), &offset)
                } else {
                    object_store.list(prefix.as_ref())
                };
                while let Some(item) = stream.next().await {
                    if tx.send(item).await.is_err() {
                        break;
                    }
                }
            })
            .in_current_span(),
        );
        Self { rx, handle }
    }

    fn abort(&self) {
        self.handle.abort();
    }
}

impl Stream for StaticListStream {
    type Item = Result<ObjectMeta, object_store::Error>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.rx.poll_recv(cx) {
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// A stream that does outer retries on list operations.
///
/// This is to handle request responses that ObjectStore doesn't handle, such as
/// the error `error decoding response body` from queries to GCS.
pub struct ListRetryStream {
    object_store: Arc<dyn ObjectStore>,
    current_stream: StaticListStream,
    prefix: Option<Path>,
    last_successful_key: Option<Path>,
    max_retries: usize,
    current_retries: usize,
}

impl ListRetryStream {
    pub fn new(
        object_store: Arc<dyn ObjectStore>,
        prefix: Option<Path>,
        max_retries: usize,
    ) -> Self {
        let current_stream = StaticListStream::new(object_store.clone(), prefix.clone(), None);
        Self {
            object_store,
            current_stream,
            prefix,
            last_successful_key: None,
            max_retries,
            current_retries: 0,
        }
    }

    fn is_retryable(error: &object_store::Error) -> bool {
        !matches!(
            error,
            object_store::Error::NotFound { .. }
                | object_store::Error::InvalidPath { .. }
                | object_store::Error::NotSupported { .. }
                | object_store::Error::NotImplemented
        )
    }
}

impl Stream for ListRetryStream {
    type Item = Result<ObjectMeta, object_store::Error>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            match this.current_stream.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(meta))) => {
                    this.last_successful_key = Some(meta.location.clone());
                    return Poll::Ready(Some(Ok(meta)));
                }
                Poll::Ready(None) => {
                    // If the stream is done, return None
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Err(error))) if Self::is_retryable(&error) => {
                    if this.current_retries < this.max_retries {
                        this.current_retries += 1;

                        this.current_stream.abort();
                        this.current_stream = StaticListStream::new(
                            this.object_store.clone(),
                            this.prefix.clone(),
                            this.last_successful_key.clone(),
                        );

                        continue;
                    } else {
                        return Poll::Ready(Some(Err(error)));
                    }
                }
                Poll::Ready(Some(Err(error))) => {
                    return Poll::Ready(Some(Err(error)));
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    #[test]
    fn test_list_retry_stream_send() {
        // Ensure that ListRetryStream is Send
        assert_send::<ListRetryStream>();
    }
}
