// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use bytes::Bytes;
use futures::channel::oneshot;
use futures::stream::FuturesUnordered;
use futures::{FutureExt, StreamExt, TryFutureExt};
use object_store::path::Path;
use snafu::{location, Location};
use std::cmp::Reverse;
use std::fmt::Debug;
use std::future::Future;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{AcquireError, Semaphore, TryAcquireError};
use tokio::time::error::Elapsed;

use lance_core::{Error, Result};

use crate::object_store::ObjectStore;
use crate::traits::Reader;

// There is one instance of MutableBatch shared by all the I/O operations
// that make up a single request.  When all the I/O operations complete
// then the MutableBatch goes out of scope and the batch request is considered
// complete
struct MutableBatch<F: FnOnce(Response) + Send> {
    when_done: Option<F>,
    data_buffers: Vec<Bytes>,
    permits_consumed: AtomicU64,
    err: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
}

impl<F: FnOnce(Response) + Send> MutableBatch<F> {
    fn new(when_done: F, num_data_buffers: u32) -> Self {
        Self {
            when_done: Some(when_done),
            data_buffers: vec![Bytes::default(); num_data_buffers as usize],
            permits_consumed: AtomicU64::from(0),
            err: None,
        }
    }
}

// Rather than keep track of when all the I/O requests are finished so that we
// can deliver the batch of data we let Rust do that for us.  When all I/O's are
// done then the MutableBatch will go out of scope and we know we have all the
// data.
impl<F: FnOnce(Response) + Send> Drop for MutableBatch<F> {
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
        let response = Response {
            data: result,
            permits_acquired: self.permits_consumed.load(Ordering::Acquire),
        };
        (self.when_done.take().unwrap())(response);
    }
}

struct DataChunk {
    task_idx: usize,
    permits_acquired: u64,
    data: Result<Bytes>,
}

trait DataSink: Send {
    fn deliver_data(&mut self, data: DataChunk);
}

impl<F: FnOnce(Response) + Send> DataSink for MutableBatch<F> {
    // Called by worker tasks to add data to the MutableBatch
    fn deliver_data(&mut self, data: DataChunk) {
        self.permits_consumed
            .fetch_add(data.permits_acquired, Ordering::Release);
        match data.data {
            Ok(data_bytes) => {
                self.data_buffers[data.task_idx] = data_bytes;
            }
            Err(err) => {
                // This keeps the original error, if present
                self.err.get_or_insert(Box::new(err));
            }
        }
    }
}

// Don't log backpressure warnings more than once / minute
const BACKPRESSURE_DEBOUNCE: f64 = 60.0;

struct BackpressureThrottle {
    semaphore: Semaphore,
    capacity: u64,
    start: Instant,
    last_warn: Mutex<f64>,
    deadlock_prevention_timeout: Option<Duration>,
}

impl BackpressureThrottle {
    fn new(
        semaphore: Semaphore,
        capacity: u64,
        deadlock_prevention_timeout: Option<Duration>,
    ) -> Self {
        Self {
            semaphore,
            capacity,
            last_warn: Mutex::new(0_f64),
            start: Instant::now(),
            deadlock_prevention_timeout,
        }
    }

    fn warn_if_needed(&self) {
        let seconds_elapsed = self.start.elapsed().as_secs_f64();
        let mut last_warn = self.last_warn.lock().unwrap();
        let since_last_warn = seconds_elapsed - *last_warn;
        if *last_warn == 0.0 || since_last_warn > BACKPRESSURE_DEBOUNCE {
            log::warn!("Backpressure throttle is full, I/O will pause until buffer is drained.  Max I/O bandwidth will not be achieved because CPU is falling behind");
            *last_warn = seconds_elapsed;
        }
    }

    async fn acquire_permit(&self, num_bytes: u64) -> u64 {
        // First, try and acquire the permit without waiting
        let permits_needed = num_bytes.min(self.capacity).min(u32::MAX as u64);
        if permits_needed < num_bytes {
            log::warn!(
                "I/O request for {} bytes exceeds the I/O buffer size of {}",
                num_bytes,
                self.capacity
            );
        }
        let permit = self.semaphore.try_acquire_many(permits_needed as u32);
        match permit {
            Ok(permit) => {
                permit.forget();
                return permits_needed;
            }
            Err(TryAcquireError::Closed) => {
                // If we're shutting down the scan, ignore backpressure
                return 0;
            }
            Err(TryAcquireError::NoPermits) => {}
        };
        // If backpressure is full we need to alert the user and wait
        self.warn_if_needed();
        let wait_for_backpressure = self.semaphore.acquire_many(permits_needed as u32);
        if let Some(deadline) = self.deadlock_prevention_timeout {
            match tokio::time::timeout(deadline, wait_for_backpressure).await {
                Ok(Ok(permit)) => {
                    permit.forget();
                    permits_needed
                }
                Ok(Err(AcquireError { .. })) => 0,
                Err(Elapsed { .. }) => {
                    log::error!(
                        concat!(
                            "Waited over {} seconds for backpressure throttle to clear. ",
                            "Deadlock prevention is kicking in and we will release the I/O. ",
                            "If your data processing is simply very slow then increase the ",
                            "deadlock prevention timeout or disable it entirely.  If your ",
                            "data processing is fast then you may try increasing the size ",
                            "of your I/O buffer or there may be a bug."
                        ),
                        deadline.as_secs()
                    );
                    0
                }
            }
        } else {
            match wait_for_backpressure.await {
                Ok(permit) => {
                    permit.forget();
                    permits_needed
                }
                Err(AcquireError { .. }) => 0,
            }
        }
    }

    fn release(&self, num_permits: u64) {
        self.semaphore.add_permits(num_permits as usize);
    }
}

struct IoTask {
    reader: Arc<dyn Reader>,
    to_read: Range<u64>,
    when_done: Box<dyn FnOnce((Result<Bytes>, u64)) + Send>,
    permits_to_realease: u64,
}

impl IoTask {
    fn num_bytes(&self) -> u64 {
        self.to_read.end - self.to_read.start
    }

    fn set_permits_to_release(&mut self, permits: u64) {
        self.permits_to_realease = permits;
    }

    async fn run(self) {
        let bytes_fut = self
            .reader
            .get_range(self.to_read.start as usize..self.to_read.end as usize);
        let bytes = bytes_fut.await.map_err(Error::from);
        (self.when_done)((bytes, self.permits_to_realease));
    }
}

// Every time a scheduler starts up it launches a task to run the I/O loop.  This loop
// repeats endlessly until the scheduler is destroyed.
async fn run_io_loop(
    tasks: async_priority_channel::Receiver<IoTask, Reverse<u128>>,
    backpressure_throttle: Arc<BackpressureThrottle>,
    io_capacity: u32,
) {
    let mut in_process = FuturesUnordered::new();

    // First, prime the queue up to io_capacity
    for _ in 0..io_capacity {
        let next_task = tasks.recv().await;
        match next_task {
            Ok(task) => {
                let mut task = task.0;
                let permits_acquired = backpressure_throttle.acquire_permit(task.num_bytes()).await;
                task.set_permits_to_release(permits_acquired);
                let handle = tokio::spawn(task.run());
                in_process.push(handle);
            }
            Err(async_priority_channel::RecvError) => {
                // The sender has been dropped, we are done
                return;
            }
        }
    }
    // Pop the first finished task off the queue and submit another until
    // we are done
    loop {
        let res = in_process.next().await;
        res.unwrap().unwrap();
        let next_task = tasks.recv().await;
        match next_task {
            Ok(task) => {
                let handle = tokio::spawn(task.0.run());
                in_process.push(handle);
            }
            Err(async_priority_channel::RecvError) => {
                // The sender has been dropped, we are done
                return;
            }
        }
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
    backpressure_throttle: Arc<BackpressureThrottle>,
}

impl Debug for ScanScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanScheduler")
            .field("object_store", &self.object_store)
            .field("file_counter", &self.file_counter)
            .finish()
    }
}

struct Response {
    data: Result<Vec<Bytes>>,
    permits_acquired: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    /// the # of bytes that can be buffered but not yet requested.
    /// This controls back pressure.  If data is not processed quickly enough then this
    /// buffer will fill up and the I/O loop will pause until the buffer is drained.
    pub io_buffer_size_bytes: u64,
    /// The backpressure mechanism can potentially lead to deadlock if there are bugs.
    /// A complete hang is a pretty bad user experience so we log an error and panic
    /// if the I/O loop is paused for longer than this timeout.
    ///
    /// On the other hand, the user might be doing some very expensive and slow processing
    /// and they are just reading the data very very slowly (or maybe they are doing some
    /// kind of paging).  In these cases we don't want to panic so we allow this to be
    /// disabled.
    pub deadlock_prevention_timeout: Option<Duration>,
}

impl SchedulerConfig {
    /// Default configuration for that should be suitable for most unit testing purposes
    ///
    /// Note, we intentionally do not implement Default here because callers should think
    /// about these values.
    pub fn default_for_testing() -> Self {
        Self {
            io_buffer_size_bytes: 256 * 1024 * 1024,
            deadlock_prevention_timeout: Some(Duration::from_secs(30)),
        }
    }

    /// We read files for a number of tasks (e.g. creating indices, updating indices, running
    /// compaction, etc.)  In these cases, if the user has no input on the I/O buffer size we
    /// should choose something fairly reasonable (256MiB) and our internal scans should all
    /// be pretty fast and not trigger deadlock.
    pub fn fast_and_not_too_ram_intensive() -> Self {
        Self {
            io_buffer_size_bytes: 256 * 1024 * 1024,
            deadlock_prevention_timeout: Some(Duration::from_secs(60)),
        }
    }
}

impl ScanScheduler {
    /// Create a new scheduler with the given I/O capacity
    ///
    /// # Arguments
    ///
    /// * object_store - the store to wrap
    /// * config - configuration settings for the scheduler
    pub fn new(object_store: Arc<ObjectStore>, config: SchedulerConfig) -> Arc<Self> {
        let (reg_tx, reg_rx) = async_priority_channel::unbounded();
        let io_capacity = object_store.io_parallelism().unwrap();
        let backpressure_throttle = Arc::new(BackpressureThrottle::new(
            Semaphore::new(config.io_buffer_size_bytes as usize),
            config.io_buffer_size_bytes,
            config.deadlock_prevention_timeout,
        ));
        let scheduler = Self {
            object_store,
            io_submitter: reg_tx,
            file_counter: Mutex::new(0),
            backpressure_throttle: backpressure_throttle.clone(),
        };
        tokio::task::spawn(
            async move { run_io_loop(reg_rx, backpressure_throttle, io_capacity).await },
        );
        Arc::new(scheduler)
    }

    /// Open a file for reading
    pub async fn open_file(self: &Arc<Self>, path: &Path) -> Result<FileScheduler> {
        let reader = self.object_store.open(path).await?;
        let mut file_counter = self.file_counter.lock().unwrap();
        let file_index = *file_counter;
        let block_size = self.object_store.block_size() as u64;
        *file_counter += 1;
        Ok(FileScheduler {
            reader: reader.into(),
            block_size,
            root: self.clone(),
            file_index,
        })
    }

    fn do_submit_request(
        &self,
        reader: Arc<dyn Reader>,
        request: Vec<Range<u64>>,
        tx: oneshot::Sender<Response>,
        priority: u128,
    ) {
        let num_iops = request.len() as u32;

        let when_all_io_done = move |bytes_and_permits| {
            // We don't care if the receiver has given up so discard the result
            let _ = tx.send(bytes_and_permits);
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
                // This will be set by run_io_loop
                permits_to_realease: 0,
                when_done: Box::new(move |(data, permits_acquired)| {
                    let mut dest = dest.lock().unwrap();
                    let chunk = DataChunk {
                        data,
                        task_idx,
                        permits_acquired,
                    };
                    dest.deliver_data(chunk);
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
        let (tx, rx) = oneshot::channel::<Response>();

        self.do_submit_request(reader, request, tx, priority);

        let backpressure_throttle = self.backpressure_throttle.clone();
        rx.map(move |wrapped_rsp| {
            // Right now, it isn't possible for I/O to be cancelled so a cancel error should
            // not occur
            let rsp = wrapped_rsp.unwrap();
            backpressure_throttle.release(rsp.permits_acquired);
            rsp.data
        })
    }
}

/// A throttled file reader
#[derive(Clone, Debug)]
pub struct FileScheduler {
    reader: Arc<dyn Reader>,
    root: Arc<ScanScheduler>,
    block_size: u64,
    file_index: u32,
}

fn is_close_together(range1: &Range<u64>, range2: &Range<u64>, block_size: u64) -> bool {
    // Note that range1.end <= range2.start is possible (e.g. when decoding string arrays)
    range2.start <= (range1.end + block_size)
}

fn is_overlapping(range1: &Range<u64>, range2: &Range<u64>) -> bool {
    range1.start < range2.end && range2.start < range1.end
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

        let mut updated_requests = Vec::with_capacity(request.len());

        if !request.is_empty() {
            let mut curr_interval = request[0].clone();

            for req in request.iter().skip(1) {
                if is_close_together(&curr_interval, req, self.block_size) {
                    curr_interval.end = curr_interval.end.max(req.end);
                } else {
                    updated_requests.push(curr_interval);
                    curr_interval = req.clone();
                }
            }

            updated_requests.push(curr_interval);
        }

        let bytes_vec_fut =
            self.root
                .submit_request(self.reader.clone(), updated_requests.clone(), priority);

        let mut updated_index = 0;
        let mut final_bytes = Vec::with_capacity(request.len());

        async move {
            let bytes_vec = bytes_vec_fut.await?;

            let mut orig_index = 0;
            while (updated_index < updated_requests.len()) && (orig_index < request.len()) {
                let updated_range = &updated_requests[updated_index];
                let orig_range = &request[orig_index];
                let byte_offset = updated_range.start as usize;

                if is_overlapping(updated_range, orig_range) {
                    // Rescale the ranges since they correspond to the entire set of bytes, while
                    // But we need to slice into a subset of the bytes in a particular index of bytes_vec
                    let start = orig_range.start as usize - byte_offset;
                    let end = orig_range.end as usize - byte_offset;

                    let sliced_range = bytes_vec[updated_index].slice(start..end);
                    final_bytes.push(sliced_range);
                    orig_index += 1;
                } else {
                    updated_index += 1;
                }
            }

            Ok(final_bytes)
        }
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

    use object_store::{memory::InMemory, GetRange, ObjectStore as OSObjectStore};
    use tokio::{runtime::Handle, time::timeout};
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

        let config = SchedulerConfig {
            deadlock_prevention_timeout: None,
            io_buffer_size_bytes: 1024 * 1024,
        };

        let scheduler = ScanScheduler::new(obj_store, config);

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
            .put(&some_path, vec![0; 1000].into())
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
                    base_store.get_opts(&location, options).await
                }
                .boxed()
            });
        let obj_store = Arc::new(ObjectStore::new(
            Arc::new(obj_store),
            Url::parse("mem://").unwrap(),
            None,
            None,
            false,
            1,
        ));

        let config = SchedulerConfig {
            deadlock_prevention_timeout: None,
            io_buffer_size_bytes: 1024 * 1024,
        };

        let scan_scheduler = ScanScheduler::new(obj_store, config);

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

    #[tokio::test(flavor = "multi_thread")]
    async fn test_backpressure() {
        let some_path = Path::parse("foo").unwrap();
        let base_store = Arc::new(InMemory::new());
        base_store
            .put(&some_path, vec![0; 100000].into())
            .await
            .unwrap();

        let bytes_read = Arc::new(AtomicU64::from(0));
        let mut obj_store = MockObjectStore::default();
        let bytes_read_copy = bytes_read.clone();
        // Wraps the obj_store to keep track of how many bytes have been read
        obj_store
            .expect_get_opts()
            .returning(move |location, options| {
                let range = options.range.as_ref().unwrap();
                let num_bytes = match range {
                    GetRange::Bounded(bounded) => bounded.end - bounded.start,
                    _ => panic!(),
                };
                bytes_read_copy.fetch_add(num_bytes as u64, Ordering::Release);
                let location = location.clone();
                let base_store = base_store.clone();
                async move { base_store.get_opts(&location, options).await }.boxed()
            });
        let obj_store = Arc::new(ObjectStore::new(
            Arc::new(obj_store),
            Url::parse("mem://").unwrap(),
            None,
            None,
            false,
            1,
        ));

        let config = SchedulerConfig {
            deadlock_prevention_timeout: Some(Duration::from_millis(50)),
            io_buffer_size_bytes: 10,
        };

        let scan_scheduler = ScanScheduler::new(obj_store.clone(), config);

        let file_scheduler = scan_scheduler
            .open_file(&Path::parse("foo").unwrap())
            .await
            .unwrap();

        let wait_for_idle = || async move {
            let handle = Handle::current();
            while handle.metrics().num_alive_tasks() != 1 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        };
        let wait_for_bytes_read_and_idle = |target_bytes: u64| {
            // We need to move `target` but don't want to move `bytes_read`
            let bytes_read = &bytes_read;
            async move {
                let bytes_read_copy = bytes_read.clone();
                while bytes_read_copy.load(Ordering::Acquire) < target_bytes {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                wait_for_idle().await;
            }
        };

        // This read will begin immediately
        let first_fut = file_scheduler.submit_single(0..5, 0);
        // This read should also begin immediately
        let second_fut = file_scheduler.submit_single(0..5, 0);
        // This read will be throttled
        let third_fut = file_scheduler.submit_single(0..3, 0);
        // Two tasks (third_fut and unit test)
        wait_for_bytes_read_and_idle(10).await;

        assert_eq!(first_fut.await.unwrap().len(), 5);
        // One task (unit test)
        wait_for_bytes_read_and_idle(13).await;

        // 2 bytes are ready but 5 bytes requested, read will be blocked
        let fourth_fut = file_scheduler.submit_single(0..5, 0);
        wait_for_bytes_read_and_idle(13).await;

        // Out of order completion is ok, will unblock backpressure
        assert_eq!(third_fut.await.unwrap().len(), 3);
        wait_for_bytes_read_and_idle(18).await;

        assert_eq!(second_fut.await.unwrap().len(), 5);
        // At this point there are 5 bytes available in backpressure queue
        // Now we issue multi-read that can be partially fulfilled, it will read some bytes but
        // not all of them. (using large range gap to ensure request not coalesced)
        //
        // I'm actually not sure this behavior is great.  It's possible that we should just
        // block until we can fulfill the entire request.
        let fifth_fut = file_scheduler.submit_request(vec![0..3, 90000..90007], 0);
        wait_for_bytes_read_and_idle(21).await;

        // Fifth future should eventually finish due to deadlock prevention
        let fifth_bytes = tokio::time::timeout(Duration::from_secs(10), fifth_fut)
            .await
            .unwrap();
        assert_eq!(
            fifth_bytes.unwrap().iter().map(|b| b.len()).sum::<usize>(),
            10
        );

        // And now let's just make sure that we can read the rest of the data
        assert_eq!(fourth_fut.await.unwrap().len(), 5);
        wait_for_bytes_read_and_idle(28).await;

        // Ensure deadlock prevention timeout can be disabled
        let config = SchedulerConfig {
            deadlock_prevention_timeout: None,
            io_buffer_size_bytes: 10,
        };

        let scan_scheduler = ScanScheduler::new(obj_store, config);
        let file_scheduler = scan_scheduler
            .open_file(&Path::parse("foo").unwrap())
            .await
            .unwrap();

        let first_fut = file_scheduler.submit_single(0..10, 0);
        let second_fut = file_scheduler.submit_single(0..10, 0);

        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(first_fut.await.unwrap().len(), 10);
        assert_eq!(second_fut.await.unwrap().len(), 10);
    }

    #[test_log::test(tokio::test(flavor = "multi_thread"))]
    async fn stress_backpressure() {
        // This test ensures that the backpressure mechanism works correctly with
        // regards to priority.  In other words, as long as all requests are consumed
        // in priority order then the backpressure mechanism should not deadlock
        let some_path = Path::parse("foo").unwrap();
        let obj_store = Arc::new(ObjectStore::memory());
        obj_store
            .put(&some_path, vec![0; 100000].as_slice())
            .await
            .unwrap();

        // Only one request will be allowed in
        let config = SchedulerConfig {
            deadlock_prevention_timeout: Some(Duration::from_millis(1000)),
            io_buffer_size_bytes: 1,
        };
        let scan_scheduler = ScanScheduler::new(obj_store.clone(), config);
        let file_scheduler = scan_scheduler.open_file(&some_path).await.unwrap();

        let mut futs = Vec::with_capacity(10000);
        for idx in 0..10000 {
            futs.push(file_scheduler.submit_single(idx..idx + 1, idx));
        }

        for fut in futs {
            fut.await.unwrap();
        }
    }
}
