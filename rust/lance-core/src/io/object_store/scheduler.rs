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
use std::{
    ops::Range,
    sync::{Arc, Mutex},
};

use bytes::Bytes;
use futures::StreamExt;
use object_store::{path::Path, ObjectStore};
use snafu::{location, Location};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tracing::Instrument;

use crate::{io::Reader, Error, Result};

/// A wrapper around a reader that schedules IO.
///
/// When callers make read requests, they are added to a queue and a future
/// is returned. A background task will read requests from the queue and submit
/// them, keeping a maximum number of requests in flight at a time. This keeps
/// concurrency control centralized.
///
/// When the scheduler is dropped, the background tasks are aborted.
#[derive(Clone, Debug)]
pub struct Scheduler {
    pub object_store: Arc<dyn ObjectStore>,
    queue: mpsc::Sender<Request>,
    handle: Arc<Mutex<tokio::task::JoinHandle<()>>>,
}

// TODO: coalesce adjacent requests
// TODO: adaptively choose max_concurrent_requests
// TODO: optionally make backup requests if one is slow.

impl Scheduler {
    pub fn new(
        object_store: Arc<dyn ObjectStore>,
        max_concurrent_requests: usize,
        max_queue_size: usize,
    ) -> Self {
        let (sender, receiver) = mpsc::channel::<Request>(max_queue_size);
        let store_ref = object_store.clone();
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
            handle: Arc::new(Mutex::new(handle)),
        }
    }

    pub fn open_reader(&self, path: Arc<Path>, inner: Box<dyn Reader>) -> ScheduledReader {
        ScheduledReader {
            scheduler: self.clone(),
            path,
            inner,
        }
    }

    pub async fn get_range(&self, path: Arc<Path>, range: Range<usize>) -> Result<Bytes> {
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

impl Drop for Scheduler {
    fn drop(&mut self) {
        // If we're the last reference to the handle, abort the task.
        if Arc::strong_count(&self.handle) == 1 {
            self.handle.lock().unwrap().abort();
        }
    }
}

pub struct ScheduledReader {
    scheduler: Scheduler,
    path: Arc<Path>,
    inner: Box<dyn Reader>,
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

struct Request {
    path: Arc<Path>,
    range: Range<usize>,
    result_sender: oneshot::Sender<Result<Bytes>>,
    span: tracing::Span,
}
