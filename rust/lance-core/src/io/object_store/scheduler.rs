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

//! Wrapper around object store that handles scheduling of reads.
use std::{ops::Range, sync::Arc};

use bytes::Bytes;
use futures::StreamExt;
use object_store::{path::Path, ObjectStore};
use snafu::{location, Location};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tracing::Instrument;

use crate::{io::Reader, Error, Result};

/// A scheduler for read requests.
///
/// Callers can create Readers that reference this scheduler. When callers make
/// get_range requests to the reader, they are added to a queue and a future
/// is returned. A background task will read requests from the queue and submit
/// them, keeping a maximum number of requests in flight at a time. This keeps
/// concurrency control centralized.
///
/// When the scheduler is dropped, the background tasks are aborted. The readers
/// keep an `Arc<Scheduler>` so that the scheduler will not be dropped until
/// they are finished.
#[derive(Debug)]
pub struct Scheduler {
    pub object_store: Arc<dyn ObjectStore>,
    /// A queue of pending get_range requests.
    queue: mpsc::Sender<Request>,
    /// A handle to the background task that reads from the queue.
    handle: tokio::task::JoinHandle<()>,
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

// TODO: coalesce adjacent requests
// TODO: adaptively choose max_concurrent_requests
// TODO: optionally make backup requests if one is slow.

impl Scheduler {
    /// Create a new scheduler with the parameters:
    ///
    /// * `max_concurrent_requests`: The maximum number of requests to have in flight at a time.
    pub fn new(object_store: Arc<dyn ObjectStore>, max_concurrent_requests: usize) -> Self {
        // Right now the queue size doesn't really matter, since the callers either wait to
        // enter the queue or wait for us to get to them in the queue. In the future though,
        // we might want a configurable queue size when the queue is used for things like
        // request range coalescing.
        let (sender, receiver) = mpsc::channel::<Request>(100);
        let store_ref = object_store.clone();
        // This is the background task that reads from the queue and submits requests.
        let handle = tokio::task::spawn(async move {
            let mut stream = ReceiverStream::new(receiver)
                .map(|request| {
                    let store = store_ref.clone();
                    async move {
                        // Make sure the receiver still wants this request. If it
                        // has been cancelled, then the reciever will have been dropped.
                        if !request.result_sender.is_closed() {
                            let result = store
                                .get_range(&request.path, request.range)
                                .await
                                .map_err(|err| Error::IO {
                                    message: err.to_string(),
                                    location: location!(),
                                });
                            request.result_sender.send(result)
                        } else {
                            Ok(())
                        }
                    }
                    .instrument(request.span)
                })
                .buffer_unordered(max_concurrent_requests);

            while let Some(res) = stream.next().await {
                if res.is_err() {
                    // This might happen if the listening end of the channel is dropped,
                    // due to cancellation.
                    log::info!("Read task failed");
                }
            }
        });

        Self {
            object_store,
            queue: sender,
            handle,
        }
    }

    /// Submit a single get_range request to the queue, returning a future
    /// that will resolve to the result of the request.
    ///
    /// The request may not immediately be submitted to the object store, depending
    /// on the number of requests already in flight. If the future is dropped
    /// before it initiates, the request will be ignored.
    async fn get_range(&self, path: Arc<Path>, range: Range<usize>) -> Result<Bytes> {
        let (result_sender, result_receiver) = oneshot::channel();
        let request = Request {
            path,
            range,
            result_sender,
            span: tracing::Span::current(),
        };
        self.queue
            .send(request)
            .await
            .map_err(|err| Error::Internal {
                message: format!("Failed to send read task: {err}"),
                location: location!(),
            })?;
        match result_receiver.await {
            Ok(result) => result,
            Err(_) => Err(Error::Internal {
                message: "Read task failed".to_string(),
                location: location!(),
            }),
        }
    }
}

/// A request to read a range of bytes from an object store.
struct Request {
    /// The path of the object to read.
    path: Arc<Path>,
    /// The range of bytes to read.
    range: Range<usize>,
    /// A oneshot channel to send the result of the read to.
    result_sender: oneshot::Sender<Result<Bytes>>,
    /// The span of the caller. This will be attached to the future that makes
    /// the read request.
    span: tracing::Span,
}

/// A wrapper around a [Reader] that uses a [Scheduler] to schedule reads.
pub struct ScheduledReader {
    scheduler: Arc<Scheduler>,
    path: Arc<Path>,
    inner: Box<dyn Reader>,
}

impl ScheduledReader {
    pub fn new(scheduler: Arc<Scheduler>, path: Arc<Path>, inner: Box<dyn Reader>) -> Self {
        Self {
            scheduler,
            path,
            inner,
        }
    }
}

#[async_trait::async_trait]
impl Reader for ScheduledReader {
    fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    fn path(&self) -> &Path {
        self.path.as_ref()
    }

    async fn size(&self) -> Result<usize> {
        self.inner.size().await
    }

    async fn get_range(&self, range: Range<usize>) -> Result<Bytes> {
        self.scheduler.get_range(self.path.clone(), range).await
    }
}
