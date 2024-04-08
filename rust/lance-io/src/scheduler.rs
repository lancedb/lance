// Copyright 2024 Lance Developers.
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

use bytes::Bytes;
use futures::channel::oneshot;
use futures::{FutureExt, StreamExt, TryFutureExt};
use object_store::path::Path;
use snafu::{location, Location};
use std::future::Future;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use lance_core::{Error, Result};

use crate::object_store::ObjectStore;
use crate::traits::Reader;

// There is one instance of MutableBatch shared by all the I/O operations
// that make up a single request.  When all the I/O operations complete
// then the MutableBatch goes out of scope and the batch request is considered
// complete
struct MutableBatch<F: FnOnce(Result<Vec<Bytes>>) + Send> {
    when_done: Option<F>,
    data_buffers: Vec<Bytes>,
    err: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
}

impl<F: FnOnce(Result<Vec<Bytes>>) + Send> MutableBatch<F> {
    fn new(when_done: F, num_data_buffers: u32) -> Self {
        Self {
            when_done: Some(when_done),
            data_buffers: vec![Bytes::default(); num_data_buffers as usize],
            err: None,
        }
    }
}

// Rather than keep track of when all the I/O requests are finished so that we
// can deliver the batch of data we let Rust do that for us.  When all I/O's are
// done then the MutableBatch will go out of scope and we know we have all the
// data.
impl<F: FnOnce(Result<Vec<Bytes>>) + Send> Drop for MutableBatch<F> {
    fn drop(&mut self) {
        // If we have an error, return that.  Otherwise return the data
        let result = if self.err.is_some() {
            Err(Error::Wrapped {
                error: self.err.take().unwrap(),
                location: location!(),
            })
        } else {
            let mut data = Vec::new();
            std::mem::swap(&mut data, &mut self.data_buffers);
            Ok(data)
        };
        // We don't really care if no one is around to receive it, just let
        // the result go out of scope and get cleaned up
        (self.when_done.take().unwrap())(result);
    }
}

trait DataSink: Send {
    fn deliver_data(&mut self, data: Result<(usize, Bytes)>);
}

impl<F: FnOnce(Result<Vec<Bytes>>) + Send> DataSink for MutableBatch<F> {
    // Called by worker tasks to add data to the MutableBatch
    fn deliver_data(&mut self, data: Result<(usize, Bytes)>) {
        match data {
            Ok(data) => {
                self.data_buffers[data.0] = data.1;
            }
            Err(err) => {
                // This keeps the original error, if present
                self.err.get_or_insert(Box::new(err));
            }
        }
    }
}

struct IoTask {
    reader: Arc<dyn Reader>,
    to_read: Range<u64>,
    when_done: Box<dyn FnOnce(Result<Bytes>) + Send>,
}

impl IoTask {
    async fn run(self) {
        let bytes = self
            .reader
            .get_range(self.to_read.start as usize..self.to_read.end as usize)
            .await;
        (self.when_done)(bytes);
    }
}

// Every time a scheduler starts up it launches a task to run the I/O loop.  This loop
// repeats endlessly until the scheduler is destroyed.
async fn run_io_loop(tasks: mpsc::UnboundedReceiver<IoTask>, io_capacity: u32) {
    let io_stream = UnboundedReceiverStream::new(tasks);
    let tokio_task_stream = io_stream.map(|task| tokio::spawn(task.run()));
    let mut tokio_task_stream = tokio_task_stream.buffer_unordered(io_capacity as usize);
    while tokio_task_stream.next().await.is_some() {
        // We don't actually do anything with the results here, they are sent
        // via the io tasks's when_done.  Instead we just keep chugging away
        // indefinitely until the tasks receiver returns none (scheduler has
        // been shut down)
    }
}

/// An I/O scheduler which wraps an ObjectStore and throttles the amount of\
/// parallel I/O that can be run.
///
/// TODO: This will also add coalescing
pub struct StoreScheduler {
    object_store: Arc<ObjectStore>,
    io_submitter: mpsc::UnboundedSender<IoTask>,
}

impl StoreScheduler {
    /// Create a new scheduler with the given I/O capacity
    ///
    /// # Arguments
    ///
    /// * object_store - the store to wrap
    /// * io_capacity - the maximum number of parallel requests that will be allowed
    pub fn new(object_store: Arc<ObjectStore>, io_capacity: u32) -> Arc<Self> {
        // TODO: we don't have any backpressure in place if the compute thread falls
        // behind.  The scheduler thread will schedule ALL of the I/O and then the
        // loaded data will eventually pile up.
        //
        // We could bound this channel but that wouldn't help.  If the decode thread
        // was paused then the I/O loop would keep running and reading from this channel.
        //
        // Once the reader is finished we should revisit.  We will probably want to convert
        // from `when_done` futures to delivering data into a queue.  That queue should fill
        // up, causing the I/O loop to pause.
        let (reg_tx, reg_rx) = mpsc::unbounded_channel();
        let scheduler = Self {
            object_store,
            io_submitter: reg_tx,
        };
        tokio::task::spawn(async move { run_io_loop(reg_rx, io_capacity).await });
        Arc::new(scheduler)
    }

    /// Open a file for reading
    pub async fn open_file(self: &Arc<Self>, path: &Path) -> Result<FileScheduler> {
        let reader = self.object_store.open(path).await?;
        Ok(FileScheduler {
            reader: reader.into(),
            root: self.clone(),
        })
    }

    fn do_submit_request(
        &self,
        reader: Arc<dyn Reader>,
        request: Vec<Range<u64>>,
        tx: oneshot::Sender<Result<Vec<Bytes>>>,
    ) {
        let num_iops = request.len() as u32;

        let when_all_io_done = move |bytes| {
            // We don't care if the receiver has given up
            let _ = tx.send(bytes);
        };

        let dest = Arc::new(Mutex::new(Box::new(MutableBatch::new(
            when_all_io_done,
            num_iops,
        ))));

        for (task_idx, iop) in request.into_iter().enumerate() {
            let dest = dest.clone();
            let task = IoTask {
                reader: reader.clone(),
                to_read: iop,
                when_done: Box::new(move |bytes| {
                    let mut dest = dest.lock().unwrap();
                    dest.deliver_data(bytes.map(|bytes| (task_idx, bytes)));
                }),
            };
            if self.io_submitter.send(task).is_err() {
                panic!("unable to submit I/O because the I/O thread has panic'd");
            }
        }
    }

    fn submit_request(
        &self,
        reader: Arc<dyn Reader>,
        request: Vec<Range<u64>>,
    ) -> impl Future<Output = Result<Vec<Bytes>>> + Send {
        let (tx, rx) = oneshot::channel::<Result<Vec<Bytes>>>();

        self.do_submit_request(reader, request, tx);

        // Right now, it isn't possible for I/O to be cancelled so a cancel error should
        // not occur
        rx.map(|wrapped_err| wrapped_err.unwrap())
    }
}

/// A throttled file reader
#[derive(Clone)]
pub struct FileScheduler {
    reader: Arc<dyn Reader>,
    root: Arc<StoreScheduler>,
}

impl FileScheduler {
    /// Submit a batch of I/O requests to the reader
    ///
    /// The requests will be queued in a FIFO manner and, when all requests
    /// have been fulfilled, the returned future will be completed.
    pub fn submit_request(
        &self,
        request: Vec<Range<u64>>,
    ) -> impl Future<Output = Result<Vec<Bytes>>> + Send {
        self.root.submit_request(self.reader.clone(), request)
    }

    /// Submit a single IOP to the reader
    ///
    /// If you have multpile IOPS to perform then [`Self::submit_request`] is going
    /// to be more efficient.
    pub fn submit_single(&self, range: Range<u64>) -> impl Future<Output = Result<Bytes>> + Send {
        self.submit_request(vec![range])
            .map_ok(|vec_bytes| vec_bytes.into_iter().next().unwrap())
    }

    /// Provides access to the underlying reader
    ///
    /// Do not use this for reading data as it will bypass any I/O scheduling!
    /// This is mainly exposed to allow metadata operations (e.g size, block_size,)
    /// which either aren't IOPS or we don't throttle
    pub fn reader(&self) -> &Arc<dyn Reader> {
        &self.reader
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use rand::RngCore;
    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn test_full_seq_read() {
        let tmpdir = tempdir().unwrap();
        let tmp_path = tmpdir.path().to_str().unwrap();
        let tmp_path = Path::parse(tmp_path).unwrap();
        let tmp_file = tmp_path.child("foo.file");

        let obj_store = Arc::new(ObjectStore::local());

        // Write 1MiB of data
        const DATA_SIZE: u64 = 1024 * 1024;
        let mut some_data = vec![0; DATA_SIZE as usize];
        rand::thread_rng().fill_bytes(&mut some_data);
        obj_store.put(&tmp_file, &some_data).await.unwrap();

        let scheduler = StoreScheduler::new(obj_store, 16);

        let file_scheduler = scheduler.open_file(&tmp_file).await.unwrap();

        // Read it back 4KiB at a time
        const READ_SIZE: u64 = 4 * 1024;
        let mut reqs = VecDeque::new();
        let mut offset = 0;
        while offset < DATA_SIZE {
            reqs.push_back(
                #[allow(clippy::single_range_in_vec_init)]
                file_scheduler
                    .submit_request(vec![offset..offset + READ_SIZE])
                    .await
                    .unwrap(),
            );
            offset += READ_SIZE;
        }

        offset = 0;
        // Note: we should get parallel I/O even though we are consuming serially
        while offset < DATA_SIZE {
            let data = reqs.pop_front().unwrap();
            let actual = &data[0];
            let expected = &some_data[offset as usize..(offset + READ_SIZE) as usize];
            assert_eq!(expected, actual);
            offset += READ_SIZE;
        }
    }
}
