// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use bytes::Bytes;
use futures::channel::oneshot;
use futures::{FutureExt, TryFutureExt};
use object_store::path::Path;
use snafu::{location, Location};
use std::collections::BinaryHeap;
use std::fmt::Debug;
use std::future::Future;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::{Notify, Semaphore, SemaphorePermit};

use lance_core::{Error, Result};

use crate::object_store::ObjectStore;
use crate::traits::Reader;

// Don't log backpressure warnings until at least this many seconds have passed
const BACKPRESSURE_MIN: u64 = 5;
// Don't log backpressure warnings more than once / minute
const BACKPRESSURE_DEBOUNCE: u64 = 60;

// There are two structures that control the I/O scheduler concurrency.  First,
// we have a hard limit on the number of IOPS that can be issued concurrently.
// This limit is process-wide.
//
// Second, we try and limit how many I/O requests can be buffered in memory without
// being consumed by a decoder of some kind.  This limit is per-scheduler.  We cannot
// make this limit process wide without introducing deadlock (because the decoder for
// file 0 might be waiting on IOPS blocked by a queue filled with requests for file 1)
// and vice-versa.
//
// There is also a per-scan limit on the number of IOPS that can be issued concurrently.
//
// The process-wide limit exists when users need a hard limit on the number of parallel
// IOPS, e.g. due to port availability limits or to prevent multiple scans from saturating
// the network.  (Note: a process-wide limit of X will not necessarily limit the number of
// open TCP connections to exactly X.  The underlying object store may open more connections
// anyways)
//
// However, it can be too tough in some cases, e.g. when some scans are reading from
// cloud storage and other scans are reading from local disk.  In these cases users don't
// need to set a process-limit and can rely on the per-scan limits.

// The IopsQuota enforces the first of the above limits, it is the per-process hard cap
// on the number of IOPS that can be issued concurrently.
//
// The per-scan limits are enforced by IoQueue
struct IopsQuota {
    // An Option is used here to avoid mutex overhead if no limit is set
    iops_avail: Option<Semaphore>,
}

/// A reservation on the global IOPS quota
///
/// When the reservation is dropped, the IOPS quota is released unless
/// [`Self::forget`] is called.
struct IopsReservation<'a> {
    value: Option<SemaphorePermit<'a>>,
}

impl IopsReservation<'_> {
    // Forget the reservation, so it won't be released on drop
    fn forget(&mut self) {
        if let Some(value) = self.value.take() {
            value.forget();
        }
    }
}

impl IopsQuota {
    // By default, there is no process-wide limit on IOPS
    //
    // However, the user can request one by setting the environment variable
    // LANCE_PROCESS_IO_THREADS_LIMIT to a positive integer.
    fn new() -> Self {
        let initial_capacity = std::env::var("LANCE_PROCESS_IO_THREADS_LIMIT")
            .map(|s| {
                let limit = s
                    .parse::<i32>()
                    .expect("LANCE_PROCESS_IO_THREADS_LIMIT must be a positive integer");
                if limit <= 0 {
                    panic!("LANCE_PROCESS_IO_THREADS_LIMIT must be a positive integer.  To disable the limit, unset the environment variable");
                }
                limit
            })
            // The default (-1) does not apply any limit
            .unwrap_or(-1);
        let iops_avail = if initial_capacity < 0 {
            None
        } else {
            Some(Semaphore::new(initial_capacity as usize))
        };
        Self { iops_avail }
    }

    // Return a reservation on the global IOPS quota
    fn release(&self) {
        if let Some(iops_avail) = self.iops_avail.as_ref() {
            iops_avail.add_permits(1);
        }
    }

    // Acquire a reservation on the global IOPS quota
    async fn acquire(&self) -> IopsReservation {
        if let Some(iops_avail) = self.iops_avail.as_ref() {
            IopsReservation {
                value: Some(iops_avail.acquire().await.unwrap()),
            }
        } else {
            IopsReservation { value: None }
        }
    }
}

lazy_static::lazy_static! {
    static ref IOPS_QUOTA: IopsQuota = IopsQuota::new();
}

// We want to allow requests that have a lower priority than any
// currently in-flight request.  This helps avoid potential deadlocks
// related to backpressure.  Unfortunately, it is quite expensive to
// keep track of which priorities are in-flight.
//
// TODO: At some point it would be nice if we can optimize this away but
// in_flight should remain relatively small (generally less than 256 items)
// and has not shown itself to be a bottleneck yet.
struct PrioritiesInFlight {
    in_flight: Vec<u128>,
}

impl PrioritiesInFlight {
    fn new(capacity: u32) -> Self {
        Self {
            in_flight: Vec::with_capacity(capacity as usize * 2),
        }
    }

    fn min_in_flight(&self) -> u128 {
        self.in_flight.first().copied().unwrap_or(u128::MAX)
    }

    fn push(&mut self, prio: u128) {
        let pos = match self.in_flight.binary_search(&prio) {
            Ok(pos) => pos,
            Err(pos) => pos,
        };
        self.in_flight.insert(pos, prio);
    }

    fn remove(&mut self, prio: u128) {
        if let Ok(pos) = self.in_flight.binary_search(&prio) {
            self.in_flight.remove(pos);
        } else {
            unreachable!();
        }
    }
}

struct IoQueueState {
    // Number of IOPS we can issue concurrently before pausing I/O
    iops_avail: u32,
    // Number of bytes we are allowed to buffer in memory before pausing I/O
    //
    // This can dip below 0 due to I/O prioritization
    bytes_avail: i64,
    // Pending I/O requests
    pending_requests: BinaryHeap<IoTask>,
    // Priorities of in-flight requests
    priorities_in_flight: PrioritiesInFlight,
    // Set when the scheduler is finished to notify the I/O loop to shut down
    // once all outstanding requests have been completed.
    done_scheduling: bool,
    // Time when the scheduler started
    start: Instant,
    // Last time we warned about backpressure
    last_warn: AtomicU64,
}

impl IoQueueState {
    fn new(io_capacity: u32, io_buffer_size: u64) -> Self {
        Self {
            iops_avail: io_capacity,
            bytes_avail: io_buffer_size as i64,
            pending_requests: BinaryHeap::new(),
            priorities_in_flight: PrioritiesInFlight::new(io_capacity),
            done_scheduling: false,
            start: Instant::now(),
            last_warn: AtomicU64::from(0),
        }
    }

    fn finished(&self) -> bool {
        self.done_scheduling && self.pending_requests.is_empty()
    }

    fn warn_if_needed(&self) {
        let seconds_elapsed = self.start.elapsed().as_secs();
        let last_warn = self.last_warn.load(Ordering::Acquire);
        let since_last_warn = seconds_elapsed - last_warn;
        if (last_warn == 0
            && seconds_elapsed > BACKPRESSURE_MIN
            && seconds_elapsed < BACKPRESSURE_DEBOUNCE)
            || since_last_warn > BACKPRESSURE_DEBOUNCE
        {
            tracing::event!(tracing::Level::WARN, "Backpressure throttle exceeded");
            log::warn!("Backpressure throttle is full, I/O will pause until buffer is drained.  Max I/O bandwidth will not be achieved because CPU is falling behind");
            self.last_warn
                .store(seconds_elapsed.max(1), Ordering::Release);
        }
    }

    fn can_deliver(&self, task: &IoTask) -> bool {
        if self.iops_avail == 0 {
            false
        } else if task.priority <= self.priorities_in_flight.min_in_flight() {
            true
        } else if task.num_bytes() as i64 > self.bytes_avail {
            self.warn_if_needed();
            false
        } else {
            true
        }
    }

    fn next_task(&mut self) -> Option<IoTask> {
        let task = self.pending_requests.peek()?;
        if self.can_deliver(task) {
            self.priorities_in_flight.push(task.priority);
            self.iops_avail -= 1;
            self.bytes_avail -= task.num_bytes() as i64;
            if self.bytes_avail < 0 {
                // This can happen when we admit special priority requests
                log::debug!(
                    "Backpressure throttle temporarily exceeded by {} bytes due to priority I/O",
                    -self.bytes_avail
                );
            }
            Some(self.pending_requests.pop().unwrap())
        } else {
            None
        }
    }
}

// This is modeled after the MPSC queue described here: https://docs.rs/tokio/latest/tokio/sync/struct.Notify.html
//
// However, it only needs to be SPSC since there is only one "scheduler thread"
// and one I/O loop.
struct IoQueue {
    // Queue state
    state: Mutex<IoQueueState>,
    // Used to signal new I/O requests have arrived that might potentially be runnable
    notify: Notify,
}

impl IoQueue {
    fn new(io_capacity: u32, io_buffer_size: u64) -> Self {
        Self {
            state: Mutex::new(IoQueueState::new(io_capacity, io_buffer_size)),
            notify: Notify::new(),
        }
    }

    fn push(&self, task: IoTask) {
        log::trace!(
            "Inserting I/O request for {} bytes with priority ({},{}) into I/O queue",
            task.num_bytes(),
            task.priority >> 64,
            task.priority & 0xFFFFFFFFFFFFFFFF
        );
        let mut state = self.state.lock().unwrap();
        state.pending_requests.push(task);
        drop(state);

        self.notify.notify_one();
    }

    async fn pop(&self) -> Option<IoTask> {
        loop {
            {
                // First, grab a reservation on the global IOPS quota
                // If we then get a task to run, transfer the reservation
                // to the task.  Otherwise, the reservation will be released
                // when iop_res is dropped.
                let mut iop_res = IOPS_QUOTA.acquire().await;
                // Next, try and grab a reservation from the queue
                let mut state = self.state.lock().unwrap();
                if let Some(task) = state.next_task() {
                    // Reservation successfully acquired, we will release the global
                    // global reservation after task has run.
                    iop_res.forget();
                    return Some(task);
                }

                if state.finished() {
                    return None;
                }
            }

            self.notify.notified().await;
        }
    }

    fn on_iop_complete(&self) {
        let mut state = self.state.lock().unwrap();
        state.iops_avail += 1;
        drop(state);

        self.notify.notify_one();
    }

    fn on_bytes_consumed(&self, bytes: u64, priority: u128, num_reqs: usize) {
        let mut state = self.state.lock().unwrap();
        state.bytes_avail += bytes as i64;
        for _ in 0..num_reqs {
            state.priorities_in_flight.remove(priority);
        }
        drop(state);

        self.notify.notify_one();
    }

    fn close(&self) {
        let mut state = self.state.lock().unwrap();
        state.done_scheduling = true;
        drop(state);

        self.notify.notify_one();
    }
}

// There is one instance of MutableBatch shared by all the I/O operations
// that make up a single request.  When all the I/O operations complete
// then the MutableBatch goes out of scope and the batch request is considered
// complete
struct MutableBatch<F: FnOnce(Response) + Send> {
    when_done: Option<F>,
    data_buffers: Vec<Bytes>,
    num_bytes: u64,
    priority: u128,
    num_reqs: usize,
    err: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
}

impl<F: FnOnce(Response) + Send> MutableBatch<F> {
    fn new(when_done: F, num_data_buffers: u32, priority: u128, num_reqs: usize) -> Self {
        Self {
            when_done: Some(when_done),
            data_buffers: vec![Bytes::default(); num_data_buffers as usize],
            num_bytes: 0,
            priority,
            num_reqs,
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
            num_bytes: self.num_bytes,
            priority: self.priority,
            num_reqs: self.num_reqs,
        };
        (self.when_done.take().unwrap())(response);
    }
}

struct DataChunk {
    task_idx: usize,
    num_bytes: u64,
    data: Result<Bytes>,
}

trait DataSink: Send {
    fn deliver_data(&mut self, data: DataChunk);
}

impl<F: FnOnce(Response) + Send> DataSink for MutableBatch<F> {
    // Called by worker tasks to add data to the MutableBatch
    fn deliver_data(&mut self, data: DataChunk) {
        self.num_bytes += data.num_bytes;
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

struct IoTask {
    reader: Arc<dyn Reader>,
    to_read: Range<u64>,
    when_done: Box<dyn FnOnce(Result<Bytes>) + Send>,
    priority: u128,
}

impl Eq for IoTask {}

impl PartialEq for IoTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl PartialOrd for IoTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IoTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // This is intentionally inverted.  We want a min-heap
        other.priority.cmp(&self.priority)
    }
}

impl IoTask {
    fn num_bytes(&self) -> u64 {
        self.to_read.end - self.to_read.start
    }

    async fn run(self) {
        let bytes = if self.to_read.start == self.to_read.end {
            Ok(Bytes::new())
        } else {
            let bytes_fut = self
                .reader
                .get_range(self.to_read.start as usize..self.to_read.end as usize);
            bytes_fut.await.map_err(Error::from)
        };
        IOPS_QUOTA.release();
        (self.when_done)(bytes);
    }
}

// Every time a scheduler starts up it launches a task to run the I/O loop.  This loop
// repeats endlessly until the scheduler is destroyed.
async fn run_io_loop(tasks: Arc<IoQueue>) {
    // Pop the first finished task off the queue and submit another until
    // we are done
    loop {
        let next_task = tasks.pop().await;
        match next_task {
            Some(task) => {
                tokio::spawn(task.run());
            }
            None => {
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
    io_queue: Arc<IoQueue>,
}

impl Debug for ScanScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanScheduler")
            .field("object_store", &self.object_store)
            .finish()
    }
}

struct Response {
    data: Result<Vec<Bytes>>,
    priority: u128,
    num_reqs: usize,
    num_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    /// the # of bytes that can be buffered but not yet requested.
    /// This controls back pressure.  If data is not processed quickly enough then this
    /// buffer will fill up and the I/O loop will pause until the buffer is drained.
    pub io_buffer_size_bytes: u64,
}

impl SchedulerConfig {
    /// Big enough for unit testing
    pub fn default_for_testing() -> Self {
        Self {
            io_buffer_size_bytes: 256 * 1024 * 1024,
        }
    }

    /// Configuration that should generally maximize bandwidth (not trying to save RAM
    /// at all).  We assume a max page size of 32MiB and then allow 32MiB per I/O thread
    pub fn max_bandwidth(store: &ObjectStore) -> Self {
        Self {
            io_buffer_size_bytes: 32 * 1024 * 1024 * store.io_parallelism() as u64,
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
        let io_capacity = object_store.io_parallelism();
        let io_queue = Arc::new(IoQueue::new(
            io_capacity as u32,
            config.io_buffer_size_bytes,
        ));
        let scheduler = Self {
            object_store,
            io_queue: io_queue.clone(),
        };
        tokio::task::spawn(async move { run_io_loop(io_queue).await });
        Arc::new(scheduler)
    }

    /// Open a file for reading
    ///
    /// # Arguments
    ///
    /// * path - the path to the file to open
    /// * base_priority - the base priority for I/O requests submitted to this file scheduler
    ///                   this will determine the upper 64 bits of priority (the lower 64 bits
    ///                   come from `submit_request` and `submit_single`)
    pub async fn open_file_with_priority(
        self: &Arc<Self>,
        path: &Path,
        base_priority: u64,
    ) -> Result<FileScheduler> {
        let reader = self.object_store.open(path).await?;
        let block_size = self.object_store.block_size() as u64;
        Ok(FileScheduler {
            reader: reader.into(),
            block_size,
            root: self.clone(),
            base_priority,
        })
    }

    /// Open a file with a default priority of 0
    ///
    /// See [`Self::open_file_with_priority`] for more information on the priority
    pub async fn open_file(self: &Arc<Self>, path: &Path) -> Result<FileScheduler> {
        self.open_file_with_priority(path, 0).await
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
            priority,
            request.len(),
        ))));

        for (task_idx, iop) in request.into_iter().enumerate() {
            let dest = dest.clone();
            let io_queue = self.io_queue.clone();
            let num_bytes = iop.end - iop.start;
            let task = IoTask {
                reader: reader.clone(),
                to_read: iop,
                priority,
                when_done: Box::new(move |data| {
                    io_queue.on_iop_complete();
                    let mut dest = dest.lock().unwrap();
                    let chunk = DataChunk {
                        data,
                        task_idx,
                        num_bytes,
                    };
                    dest.deliver_data(chunk);
                }),
            };
            self.io_queue.push(task);
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

        let io_queue = self.io_queue.clone();

        rx.map(move |wrapped_rsp| {
            // Right now, it isn't possible for I/O to be cancelled so a cancel error should
            // not occur
            let rsp = wrapped_rsp.unwrap();
            io_queue.on_bytes_consumed(rsp.num_bytes, rsp.priority, rsp.num_reqs);
            rsp.data
        })
    }
}

impl Drop for ScanScheduler {
    fn drop(&mut self) {
        self.io_queue.close();
    }
}

/// A throttled file reader
#[derive(Clone, Debug)]
pub struct FileScheduler {
    reader: Arc<dyn Reader>,
    root: Arc<ScanScheduler>,
    block_size: u64,
    base_priority: u64,
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
    ///
    /// Each request has a given priority.  If the I/O loop is full then requests
    /// will be buffered and requests with the *lowest* priority will be released
    /// from the buffer first.
    ///
    /// Each request has a backpressure ID which controls which backpressure throttle
    /// is applied to the request.  Requests made to the same backpressure throttle
    /// will be throttled together.
    pub fn submit_request(
        &self,
        request: Vec<Range<u64>>,
        priority: u64,
    ) -> impl Future<Output = Result<Vec<Bytes>>> + Send {
        // The final priority is a combination of the row offset and the file number
        let priority = ((self.base_priority as u128) << 64) + priority as u128;

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
    /// If you have multiple IOPS to perform then [`Self::submit_request`] is going
    /// to be more efficient.
    ///
    /// See [`Self::submit_request`] for more information on the priority and backpressure.
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

    use crate::{object_store::DEFAULT_DOWNLOAD_RETRY_COUNT, testing::MockObjectStore};

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

        let config = SchedulerConfig::default_for_testing();

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
            false,
            1,
            DEFAULT_DOWNLOAD_RETRY_COUNT,
        ));

        let config = SchedulerConfig {
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
            false,
            1,
            DEFAULT_DOWNLOAD_RETRY_COUNT,
        ));

        let config = SchedulerConfig {
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
