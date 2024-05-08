// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use bytes::Bytes;
use futures::channel::oneshot;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt, TryFutureExt};
use object_store::path::Path;
use snafu::{location, Location};
use std::cmp::Reverse;
use std::fmt::Debug;
use std::future::Future;
use std::ops::Range;
use std::sync::{Arc, Mutex};

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

fn receiver_to_stream<T: Send + 'static, P: Ord + Send + 'static>(
    tasks: async_priority_channel::Receiver<T, P>,
) -> BoxStream<'static, T> {
    futures::stream::unfold(tasks, |state| async move {
        match state.recv().await {
            Ok(val) => Some((val.0, state)),
            Err(async_priority_channel::RecvError) => None,
        }
    })
    .boxed()
}

// Every time a scheduler starts up it launches a task to run the I/O loop.  This loop
// repeats endlessly until the scheduler is destroyed.
async fn run_io_loop(
    tasks: async_priority_channel::Receiver<IoTask, Reverse<u128>>,
    io_capacity: u32,
) {
    let io_stream = receiver_to_stream(tasks);
    let tokio_task_stream = io_stream.map(|task| tokio::spawn(task.run()));
    let mut tokio_task_stream = tokio_task_stream.buffer_unordered(io_capacity as usize);
    while tokio_task_stream.next().await.is_some() {
        // We don't actually do anything with the results here, they are sent
        // via the io tasks's when_done.  Instead we just keep chugging away
        // indefinitely until the tasks receiver returns none (scheduler has
        // been shut down)
    }
}

/// An I/O scheduler which wraps an ObjectStore and throttles the amount of
/// parallel I/O that can be run.
///
/// TODO: This will also add coalescing
pub struct ScanScheduler {
    object_store: Arc<ObjectStore>,
    io_submitter: async_priority_channel::Sender<IoTask, Reverse<u128>>,
    file_counter: Mutex<u32>,
}

impl Debug for ScanScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanScheduler")
            .field("object_store", &self.object_store)
            .field("file_counter", &self.file_counter)
            .finish()
    }
}

impl ScanScheduler {
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
        let (reg_tx, reg_rx) = async_priority_channel::unbounded();
        let scheduler = Self {
            object_store,
            io_submitter: reg_tx,
            file_counter: Mutex::new(0),
        };
        tokio::task::spawn(async move { run_io_loop(reg_rx, io_capacity).await });
        Arc::new(scheduler)
    }

    /// Open a file for reading
    pub async fn open_file(self: &Arc<Self>, path: &Path) -> Result<FileScheduler> {
        let reader = self.object_store.open(path).await?;
        let mut file_counter = self.file_counter.lock().unwrap();
        let file_index = *file_counter;
        *file_counter += 1;
        Ok(FileScheduler {
            reader: reader.into(),
            root: self.clone(),
            file_index,
        })
    }

    fn do_submit_request(
        &self,
        reader: Arc<dyn Reader>,
        request: Vec<Range<u64>>,
        tx: oneshot::Sender<Result<Vec<Bytes>>>,
        priority: u128,
    ) {
        let num_iops = request.len() as u32;

        let when_all_io_done = move |bytes| {
            // We don't care if the receiver has given up so discard the result
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
            if self.io_submitter.try_send(task, Reverse(priority)).is_err() {
                panic!("unable to submit I/O because the I/O thread has panic'd");
            }
        }
    }

    fn submit_request(
        &self,
        reader: Arc<dyn Reader>,
        request: Vec<Range<u64>>,
        priority: u128,
    ) -> impl Future<Output = Result<Vec<Bytes>>> + Send {
        let (tx, rx) = oneshot::channel::<Result<Vec<Bytes>>>();

        self.do_submit_request(reader, request, tx, priority);

        // Right now, it isn't possible for I/O to be cancelled so a cancel error should
        // not occur
        rx.map(|wrapped_err| wrapped_err.unwrap())
    }
}

/// A throttled file reader
#[derive(Clone, Debug)]
pub struct FileScheduler {
    reader: Arc<dyn Reader>,
    root: Arc<ScanScheduler>,
    file_index: u32,
}

impl FileScheduler {
    /// Submit a batch of I/O requests to the reader
    ///
    /// The requests will be queued in a FIFO manner and, when all requests
    /// have been fulfilled, the returned future will be completed.
    pub fn submit_request(
        &self,
        request: Vec<Range<u64>>,
        priority: u64,
    ) -> impl Future<Output = Result<Vec<Bytes>>> + Send {
        // The final priority is a combination of the row offset and the file number
        let priority = ((self.file_index as u128) << 64) + priority as u128;
        self.root
            .submit_request(self.reader.clone(), request, priority)
    }

    /// Submit a single IOP to the reader
    ///
    /// If you have multpile IOPS to perform then [`Self::submit_request`] is going
    /// to be more efficient.
    pub fn submit_single(
        &self,
        range: Range<u64>,
        priority: u64,
    ) -> impl Future<Output = Result<Bytes>> + Send {
        self.submit_request(vec![range], priority)
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
    use std::{collections::VecDeque, time::Duration};

    use futures::poll;
    use rand::RngCore;
    use tempfile::tempdir;

    use object_store::{memory::InMemory, ObjectStore as OSObjectStore};
    use tokio::time::timeout;
    use url::Url;

    use crate::testing::MockObjectStore;

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

        let scheduler = ScanScheduler::new(obj_store, 16);

        let file_scheduler = scheduler.open_file(&tmp_file).await.unwrap();

        // Read it back 4KiB at a time
        const READ_SIZE: u64 = 4 * 1024;
        let mut reqs = VecDeque::new();
        let mut offset = 0;
        while offset < DATA_SIZE {
            reqs.push_back(
                #[allow(clippy::single_range_in_vec_init)]
                file_scheduler
                    .submit_request(vec![offset..offset + READ_SIZE], 0)
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

    #[tokio::test]
    async fn test_priority() {
        let some_path = Path::parse("foo").unwrap();
        let base_store = Arc::new(InMemory::new());
        base_store
            .put(&some_path, Bytes::from(vec![0; 1000]))
            .await
            .unwrap();

        let semaphore = Arc::new(tokio::sync::Semaphore::new(0));
        let mut obj_store = MockObjectStore::default();
        let semaphore_copy = semaphore.clone();
        obj_store
            .expect_get_opts()
            .returning(move |location, options| {
                let semaphore = semaphore.clone();
                let base_store = base_store.clone();
                let location = location.clone();
                async move {
                    semaphore.acquire().await.unwrap().forget();
                    let res = base_store.get_opts(&location, options).await;
                    res
                }
                .boxed()
            });
        let obj_store = Arc::new(ObjectStore::new(
            Arc::new(obj_store),
            Url::parse("mem://").unwrap(),
            None,
            None,
        ));

        let scan_scheduler = ScanScheduler::new(obj_store, 1);

        let file_scheduler = scan_scheduler
            .open_file(&Path::parse("foo").unwrap())
            .await
            .unwrap();

        // Issue a request, priority doesn't matter, it will be submitted
        // immediately (it will go pending)
        // Note: the timeout is to prevent a deadlock if the test fails.
        let first_fut = timeout(
            Duration::from_secs(10),
            file_scheduler.submit_single(0..10, 0),
        )
        .boxed();

        // Issue another low priority request (it will go in queue)
        let mut second_fut = timeout(
            Duration::from_secs(10),
            file_scheduler.submit_single(0..20, 100),
        )
        .boxed();

        // Issue a high priority request (it will go in queue and should bump
        // the other queued request down)
        let mut third_fut = timeout(
            Duration::from_secs(10),
            file_scheduler.submit_single(0..30, 0),
        )
        .boxed();

        // Finish one file, should be the in-flight first request
        semaphore_copy.add_permits(1);
        assert!(first_fut.await.unwrap().unwrap().len() == 10);
        // Other requests should not be finished
        assert!(poll!(&mut second_fut).is_pending());
        assert!(poll!(&mut third_fut).is_pending());

        // Next should be high priority request
        semaphore_copy.add_permits(1);
        assert!(third_fut.await.unwrap().unwrap().len() == 30);
        assert!(poll!(&mut second_fut).is_pending());

        // Finally, the low priority request
        semaphore_copy.add_permits(1);
        assert!(second_fut.await.unwrap().unwrap().len() == 20);
    }
}
