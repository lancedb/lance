// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use bytes::Bytes;
use futures::task::noop_waker;
use futures::{FutureExt, TryFutureExt};
use object_store::path::Path;
use snafu::location;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt::Debug;
use std::future::Future;
use std::num::NonZero;
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};
use std::time::Instant;

use lance_core::{Error, Result};

use crate::object_store::ObjectStore;
use crate::traits::Reader;
use crate::utils::CachedFileSize;

// Global counter of how many IOPS we have issued
static IOPS_COUNTER: AtomicU64 = AtomicU64::new(0);
// Global counter of how many bytes were read by the scheduler
static BYTES_READ_COUNTER: AtomicU64 = AtomicU64::new(0);
// Don't log backpressure warnings until at least this many seconds have passed
const BACKPRESSURE_MIN: u64 = 5;
// Don't log backpressure warnings more than once / minute
const BACKPRESSURE_DEBOUNCE: u64 = 60;

// By default, we limit the number of IOPS across the entire process to 128
//
// In theory this is enough for ~10GBps on S3 following the guidelines to issue
// 1 IOP per 80MBps.  In practice, I have noticed slightly better performance going
// up to 256.
//
// However, non-S3 stores (e.g. GCS, Azure) can suffer significantly from too many
// concurrent IOPS.  For safety, we set the default to 128 and let the user override
// this if needed.
//
// Note: this only limits things that run through the scheduler.  It does not limit
// IOPS from other sources like writing or commits.
static DEFAULT_PROCESS_IOPS_LIMIT: u64 = 128;

pub fn iops_counter() -> u64 {
    IOPS_COUNTER.load(Ordering::Acquire)
}

pub fn bytes_read_counter() -> u64 {
    BYTES_READ_COUNTER.load(Ordering::Acquire)
}

type RunFn = Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = Result<Bytes>> + Send>> + Send>;

enum TaskState {
    Broken,
    Initial {
        idle_waker: Option<Waker>,
        run_fn: RunFn,
    },
    Reserved {
        idle_waker: Option<Waker>,
        backpressure_reservation: BackpressureReservation,
        run_fn: RunFn,
    },
    Running {
        backpressure_reservation: BackpressureReservation,
        inner: Pin<Box<dyn Future<Output = Result<Bytes>> + Send>>,
        polled: bool,
    },
    Finished {
        backpressure_reservation: BackpressureReservation,
        data: Result<Bytes>,
    },
}

struct IoTask {
    id: u64,
    num_bytes: u64,
    priority: u128,
    state: TaskState,
}

impl IoTask {
    fn is_reserved(&self) -> bool {
        match &self.state {
            TaskState::Initial { .. } => false,
            _ => true,
        }
    }

    fn cancel(&mut self) -> bool {
        let was_running = matches!(self.state, TaskState::Running { .. });
        self.state = TaskState::Finished {
            backpressure_reservation: BackpressureReservation {
                num_bytes: 0,
                priority: 0,
            },
            data: Err(Error::IO {
                source: Box::new(Error::IO {
                    source: "I/O Task cancelled".to_string().into(),
                    location: location!(),
                }),
                location: location!(),
            }),
        };
        was_running
    }

    fn reserve(&mut self, backpressure_reservation: BackpressureReservation) -> Result<()> {
        let state = std::mem::replace(&mut self.state, TaskState::Broken);
        let TaskState::Initial { idle_waker, run_fn } = state else {
            return Err(Error::Internal {
                message: format!("Task with id {} not in initial state", self.id),
                location: location!(),
            });
        };
        self.state = TaskState::Reserved {
            idle_waker: idle_waker,
            backpressure_reservation,
            run_fn: run_fn,
        };
        Ok(())
    }

    fn start(&mut self) -> Result<()> {
        let state = std::mem::replace(&mut self.state, TaskState::Broken);
        let TaskState::Reserved {
            backpressure_reservation,
            idle_waker,
            run_fn,
        } = state
        else {
            return Err(Error::Internal {
                message: format!("Task with id {} not in reserved state", self.id),
                location: location!(),
            });
        };
        let mut inner = run_fn();
        // Poll task immediately to get it started
        let noop_waker = noop_waker();
        let mut dummy_cx = Context::from_waker(&noop_waker);
        match inner.as_mut().poll(&mut dummy_cx) {
            Poll::Ready(data) => {
                self.state = TaskState::Finished {
                    data,
                    backpressure_reservation,
                };
            }
            Poll::Pending => {
                self.state = TaskState::Running {
                    backpressure_reservation,
                    inner,
                    polled: false,
                };
            }
        }
        // If someone is already waiting for this task let them know it is now running
        // so they can poll it
        if let Some(idle_waker) = idle_waker {
            idle_waker.wake();
        }
        Ok(())
    }

    // Quick check to see if the task is finished or if it needs to be polled
    // at least once more
    fn is_finished(&self) -> bool {
        matches!(self.state, TaskState::Broken | TaskState::Finished { .. })
    }

    fn poll(&mut self, cx: &mut Context<'_>, is_babysitter: bool) -> Poll<bool> {
        match &mut self.state {
            TaskState::Broken => Poll::Ready(false),
            TaskState::Initial { idle_waker, .. } | TaskState::Reserved { idle_waker, .. } => {
                idle_waker.replace(cx.waker().clone());
                Poll::Pending
            }
            TaskState::Running {
                inner,
                polled,
                backpressure_reservation,
            } => {
                match (*polled, is_babysitter) {
                    (true, true) => {
                        // Decoder is already polling this task, so mark that we don't need to
                        // babysit it any longer
                        return Poll::Ready(false);
                    }
                    (_, false) => {
                        // This is a decoder polling the task, so mark that decoder is interested
                        *polled = true;
                    }
                    _ => {}
                };

                match inner.as_mut().poll(cx) {
                    Poll::Ready(data) => {
                        self.state = TaskState::Finished {
                            data,
                            backpressure_reservation: *backpressure_reservation,
                        };
                        Poll::Ready(true)
                    }
                    Poll::Pending => Poll::Pending,
                }
            }
            TaskState::Finished { .. } => Poll::Ready(false),
        }
    }

    fn consume(self) -> Result<(Result<Bytes>, BackpressureReservation)> {
        let TaskState::Finished {
            data,
            backpressure_reservation,
        } = self.state
        else {
            return Err(Error::Internal {
                message: format!("Task with id {} not in finished state", self.id),
                location: location!(),
            });
        };
        Ok((data, backpressure_reservation))
    }
}

static PROCESS_CONCURRENCY_LIMIT: LazyLock<Mutex<u64>> = LazyLock::new(|| {
    let initial_capacity = std::env::var("LANCE_PROCESS_IO_THREADS_LIMIT")
        .map(|s| {
            s.parse::<u64>().unwrap_or_else(|_| {
                log::warn!("Ignoring invalid LANCE_PROCESS_IO_THREADS_LIMIT: {}", s);
                DEFAULT_PROCESS_IOPS_LIMIT
            })
        })
        .unwrap_or(DEFAULT_PROCESS_IOPS_LIMIT);
    Mutex::new(initial_capacity)
});

/// A throttle to control how many IOPS can be issued concurrently
trait ConcurrencyThrottle: Send {
    fn try_acquire(&mut self) -> bool;
    fn release(&mut self);
}

/// The default concurrency throttle combines a per-scan limit with a per-process limit
struct SimpleConcurrencyThrottle {
    concurrency_available: u64,
}

impl SimpleConcurrencyThrottle {
    fn new(max_concurrency: u64) -> Self {
        Self {
            concurrency_available: max_concurrency,
        }
    }
}

impl ConcurrencyThrottle for SimpleConcurrencyThrottle {
    fn try_acquire(&mut self) -> bool {
        if self.concurrency_available > 0 {
            let mut process_concurrency_limit = PROCESS_CONCURRENCY_LIMIT.lock().unwrap();
            if *process_concurrency_limit == 0 {
                return false;
            }
            *process_concurrency_limit -= 1;
            self.concurrency_available -= 1;
            true
        } else {
            false
        }
    }

    fn release(&mut self) {
        let mut process_concurrency_limit = PROCESS_CONCURRENCY_LIMIT.lock().unwrap();
        *process_concurrency_limit += 1;
        self.concurrency_available += 1;
    }
}

#[derive(Debug, Clone, Copy)]
struct BackpressureReservation {
    num_bytes: u64,
    priority: u128,
}

/// A throttle to control how many bytes can be read before we pause to let compute catch up
trait BackpressureThrottle: Send {
    fn try_acquire(&mut self, num_bytes: u64, priority: u128) -> Option<BackpressureReservation>;
    fn release(&mut self, reservation: BackpressureReservation);
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
    fn new(capacity: u64) -> Self {
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
        }
    }
}

struct SimpleBackpressureThrottle {
    start: Instant,
    last_warn: AtomicU64,
    bytes_available: i64,
    priorities_in_flight: PrioritiesInFlight,
}

impl SimpleBackpressureThrottle {
    fn try_new(max_bytes: u64, max_concurrency: u64) -> Result<Self> {
        if max_bytes > i64::MAX as u64 {
            return Err(Error::Internal {
                message: format!("Max bytes must be less than {}", i64::MAX),
                location: location!(),
            });
        }
        Ok(Self {
            start: Instant::now(),
            last_warn: AtomicU64::new(0),
            bytes_available: max_bytes as i64,
            priorities_in_flight: PrioritiesInFlight::new(max_concurrency),
        })
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
            tracing::event!(tracing::Level::DEBUG, "Backpressure throttle exceeded");
            log::debug!("Backpressure throttle is full, I/O will pause until buffer is drained.  Max I/O bandwidth will not be achieved because CPU is falling behind");
            self.last_warn
                .store(seconds_elapsed.max(1), Ordering::Release);
        }
    }
}

impl BackpressureThrottle for SimpleBackpressureThrottle {
    fn try_acquire(&mut self, num_bytes: u64, priority: u128) -> Option<BackpressureReservation> {
        if self.bytes_available >= num_bytes as i64
            || self.priorities_in_flight.min_in_flight() >= priority
        {
            self.bytes_available -= num_bytes as i64;
            self.priorities_in_flight.push(priority);
            Some(BackpressureReservation {
                num_bytes,
                priority,
            })
        } else {
            self.warn_if_needed();
            None
        }
    }

    fn release(&mut self, reservation: BackpressureReservation) {
        self.bytes_available += reservation.num_bytes as i64;
        self.priorities_in_flight.remove(reservation.priority);
    }
}

struct TaskEntry {
    task_id: u64,
    priority: u128,
    reserved: bool,
}

impl Ord for TaskEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Prefer reserved tasks over unreserved tasks and then highest priority tasks over lowest
        // priority tasks.
        //
        // This is a max-heap so we sort by reserved in normal order (true > false) and priority
        // in reverse order (lowest priority first)
        self.reserved
            .cmp(&other.reserved)
            .then(other.priority.cmp(&self.priority))
    }
}

impl PartialOrd for TaskEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for TaskEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for TaskEntry {}

struct IoQueueState {
    concurrency_throttle: Box<dyn ConcurrencyThrottle>,
    backpressure_throttle: Box<dyn BackpressureThrottle>,
    pending_tasks: BinaryHeap<TaskEntry>,
    tasks: HashMap<u64, IoTask>,
    tasks_to_babysit: HashSet<u64>,
    wake_babysitter: Option<Waker>,
    next_task_id: u64,
}

impl IoQueueState {
    fn try_new(max_concurrency: u64, max_bytes: u64) -> Result<Self> {
        Ok(Self {
            concurrency_throttle: Box::new(SimpleConcurrencyThrottle::new(max_concurrency)),
            backpressure_throttle: Box::new(SimpleBackpressureThrottle::try_new(
                max_bytes,
                max_concurrency,
            )?),
            pending_tasks: BinaryHeap::new(),
            tasks: HashMap::new(),
            tasks_to_babysit: HashSet::new(),
            wake_babysitter: None,
            next_task_id: 0,
        })
    }
}

/// A single-producer, single-consumer queue of I/O tasks to be shared between
/// the I/O scheduler and the I/O decoder.  There is also a third actor, the babysitter, which
/// interacts with the queue as well.
///
/// The queue is protected by a throttle to control how many IOPS can be issued concurrently.
///
/// The implementation utilizes three queues.  The first is a priority queue of tasks that have not
/// yet been started because they are waiting on the throttle.  The second is a FIFO queue of tasks
/// that have been started and are in progress.  The third is a FIFO queue of tasks that have been
/// completed.
///
/// All of these queues, and the throttle, are protected by a mutex, so only one of the three actors
/// can interact with the queue at a time.
///
/// When a task is added to the queue, we first check the throttle to see if we can run the task.  If
/// there is space then we start the task and place it in the FIFO queue.  If there is no space then
/// we place the task in the priority queue.
///
/// When the decoder requests a task, we poll the FIFO queue for a task.  If there is no task then
/// the decoder is asynchronously blocked until one becomes available.
///
/// The babysitter's job is to ensure we are periodically polling I/O tasks from the FIFO queue so that
/// these tasks do not pause if the decoder is busy.  If the babysitter, or the scheduler, complete a
/// task, then the task is put into the finished tasks FIFO.
///
/// When a task is finished, we partially release the reservation from the throttle.  This could happen
/// from any thread (scheduler, decoder, and babysitter).  When the task is consumed, we fully release
/// the reservation.  This only happens on the decoder thread.
///
/// In all of these cases, we may now have enough space to run another task.  We check the throttle to
/// see if this is true, and if so, we start another task, moving it from the priority queue to the FIFO
/// queue.
struct IoQueue {
    state: Arc<Mutex<IoQueueState>>,
}

impl IoQueue {
    fn try_new(max_concurrency: u64, max_bytes: u64) -> Result<Self> {
        Ok(Self {
            state: Arc::new(Mutex::new(IoQueueState::try_new(
                max_concurrency,
                max_bytes,
            )?)),
        })
    }

    fn push(&self, mut task: IoTask, mut state: MutexGuard<IoQueueState>) -> Result<()> {
        let task_id = task.id;
        if let Some(reservation) = state
            .backpressure_throttle
            .try_acquire(task.num_bytes, task.priority)
        {
            task.reserve(reservation)?;
            if state.concurrency_throttle.try_acquire() {
                task.start()?;
                // If the underlying I/O is synchronous (e.g. in-memory I/O) then it will
                // already be finished at this point
                //
                // Otherwise, we need to add it to the list of tasks to babysit and wake the babysitter
                let finished = task.is_finished();
                log::trace!(
                    "Started I/O task with id {} and finished={}",
                    task_id,
                    finished
                );
                state.tasks.insert(task_id, task);
                if finished {
                    state.concurrency_throttle.release();
                } else {
                    state.tasks_to_babysit.insert(task_id);
                    let waker = state.wake_babysitter.take();
                    drop(state);
                    if let Some(waker) = waker {
                        waker.wake();
                    }
                }
                return Ok(());
            }
        }

        state.pending_tasks.push(TaskEntry {
            task_id,
            priority: task.priority,
            reserved: task.is_reserved(),
        });
        state.tasks.insert(task_id, task);
        Ok(())
    }

    fn submit(
        self: Arc<Self>,
        range: Range<u64>,
        priority: u128,
        run_fn: RunFn,
    ) -> Result<TaskHandle> {
        log::trace!(
            "Submitting I/O task with range {:?}, priority {:?}",
            range,
            priority
        );
        let mut state = self.state.lock().unwrap();
        let task_id = state.next_task_id;
        state.next_task_id += 1;

        let task = IoTask {
            id: task_id,
            num_bytes: range.end - range.start,
            priority,
            state: TaskState::Initial {
                idle_waker: None,
                run_fn,
            },
        };
        self.push(task, state)?;
        Ok(TaskHandle {
            task_id,
            queue: self,
        })
    }

    fn on_task_complete(&self, mut state: MutexGuard<IoQueueState>) -> Result<()> {
        let mut has_new_babysitting_task = false;
        let state_ref = &mut *state;
        while !state_ref.pending_tasks.is_empty() {
            // Unwrap safe here since we just checked the queue is not empty
            let next_task = state_ref.pending_tasks.peek().unwrap();
            let Some(task) = state_ref.tasks.get_mut(&next_task.task_id) else {
                log::warn!("Task with id {} was lost", next_task.task_id);
                continue;
            };
            if !task.is_reserved() {
                let Some(reservation) = state_ref
                    .backpressure_throttle
                    .try_acquire(task.num_bytes, task.priority)
                else {
                    break;
                };
                task.reserve(reservation)?;
            }
            if !state_ref.concurrency_throttle.try_acquire() {
                break;
            };
            state_ref.pending_tasks.pop();
            task.start()?;
            if task.is_finished() {
                state_ref.concurrency_throttle.release();
            } else {
                state_ref.tasks_to_babysit.insert(task.id);
                has_new_babysitting_task = true;
            }
        }

        // If we started any tasks then wake the babysitter to start babysitting them
        if has_new_babysitting_task {
            let waker = state.wake_babysitter.take();
            drop(state);
            if let Some(waker) = waker {
                waker.wake();
            }
        }
        Ok(())
    }

    fn poll(&self, task_id: u64, cx: &mut Context<'_>) -> Poll<Result<Bytes>> {
        let mut state = self.state.lock().unwrap();
        let Some(task) = state.tasks.get_mut(&task_id) else {
            // This should never happen and indicates a bug
            return Poll::Ready(Err(Error::Internal {
                message: format!("Task with id {} was lost", task_id),
                location: location!(),
            }));
        };
        match task.poll(cx, false) {
            Poll::Ready(newly_finished) => {
                if newly_finished {
                    // Only release the concurrency throttle if we just finished the task
                    state.concurrency_throttle.release();
                }
                let task = state.tasks.remove(&task_id).unwrap();
                // This may be a no-op if the task was finished by babysitter but leaving it in
                // for completeness
                state.tasks_to_babysit.remove(&task_id);
                let (bytes, reservation) = task.consume()?;
                state.backpressure_throttle.release(reservation);
                // We run on_task_complete even if not newly finished because we released the backpressure reservation
                match self.on_task_complete(state) {
                    Ok(_) => Poll::Ready(bytes),
                    Err(e) => Poll::Ready(Err(e)),
                }
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn babysit(&self, cx: &mut Context<'_>) {
        let mut state = self.state.lock().unwrap();
        let mut tasks_to_babysit = std::mem::take(&mut state.tasks_to_babysit);
        let mut finished_tasks = false;
        tasks_to_babysit.retain(|task_id| {
            let Some(task) = state.tasks.get_mut(task_id) else {
                log::warn!("Task with id {} was lost", task_id);
                return false;
            };
            match task.poll(cx, true) {
                Poll::Ready(true) => {
                    finished_tasks = true;
                    state.concurrency_throttle.release();
                    false
                }
                Poll::Ready(false) => false,
                Poll::Pending => true,
            }
        });
        state.tasks_to_babysit = tasks_to_babysit;
        state.wake_babysitter.replace(cx.waker().clone());
        if finished_tasks {
            // Even though we haven't released pressure on the backpressure throttle we have
            // released the concurrency throttle and so more tasks might be able to start
            if let Err(e) = self.on_task_complete(state) {
                log::warn!("Error completing I/O tasks in babysitter: {:?}", e);
            }
        }
    }

    fn close(&self) {
        let mut state = self.state.lock().unwrap();
        for task in std::mem::take(&mut state.tasks).values_mut() {
            if task.cancel() {
                state.concurrency_throttle.release();
            }
        }
    }
}

struct BabysitFuture<'a> {
    queue: &'a IoQueue,
}

impl<'a> Future for BabysitFuture<'a> {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.queue.babysit(cx);
        Poll::Pending
    }
}

async fn babysitter_loop(queue: Arc<IoQueue>) {
    loop {
        BabysitFuture {
            queue: queue.as_ref(),
        }
        .await;
    }
}

struct TaskHandle {
    task_id: u64,
    queue: Arc<IoQueue>,
}

impl Future for TaskHandle {
    type Output = Result<Bytes>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.queue.poll(self.task_id, cx)
    }
}

#[derive(Debug)]
struct StatsCollector {
    iops: AtomicU64,
    requests: AtomicU64,
    bytes_read: AtomicU64,
}

impl StatsCollector {
    fn new() -> Self {
        Self {
            iops: AtomicU64::new(0),
            requests: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
        }
    }

    fn iops(&self) -> u64 {
        self.iops.load(Ordering::Relaxed)
    }

    fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }

    fn requests(&self) -> u64 {
        self.requests.load(Ordering::Relaxed)
    }

    fn record_request(&self, request: &[Range<u64>]) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.iops.fetch_add(request.len() as u64, Ordering::Relaxed);
        self.bytes_read.fetch_add(
            request.iter().map(|r| r.end - r.start).sum::<u64>(),
            Ordering::Relaxed,
        );
    }
}

pub struct ScanStats {
    pub iops: u64,
    pub requests: u64,
    pub bytes_read: u64,
}

impl ScanStats {
    fn new(stats: &StatsCollector) -> Self {
        Self {
            iops: stats.iops(),
            requests: stats.requests(),
            bytes_read: stats.bytes_read(),
        }
    }
}

/// An I/O scheduler which wraps an ObjectStore and throttles the amount of
/// parallel I/O that can be run.
///
/// The ScanScheduler will cancel any outstanding I/O requests when it is dropped.
/// For this reason it should be kept alive until all I/O has finished.
///
/// Note: The 2.X file readers already do this so this is only a concern if you are
/// using the ScanScheduler directly.
pub struct ScanScheduler {
    object_store: Arc<ObjectStore>,
    io_queue: Arc<IoQueue>,
    stats: Arc<StatsCollector>,
}

impl Debug for ScanScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanScheduler")
            .field("object_store", &self.object_store)
            .finish()
    }
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
    pub fn try_new(object_store: Arc<ObjectStore>, config: SchedulerConfig) -> Result<Arc<Self>> {
        let io_capacity = object_store.io_parallelism();
        let io_queue = Arc::new(IoQueue::try_new(
            io_capacity as u64,
            config.io_buffer_size_bytes,
        )?);
        let slf = Arc::new(Self {
            object_store,
            io_queue: io_queue.clone(),
            stats: Arc::new(StatsCollector::new()),
        });
        // Best we can do here is fire and forget.  If the I/O loop is still running when the scheduler is
        // dropped we can't wait for it to finish or we'd block a tokio thread.  We could spawn a blocking task
        // to wait for it to finish but that doesn't seem helpful.
        tokio::task::spawn(async move { babysitter_loop(io_queue).await });
        Ok(slf)
    }

    /// Open a file for reading
    ///
    /// # Arguments
    ///
    /// * path - the path to the file to open
    /// * base_priority - the base priority for I/O requests submitted to this file scheduler
    ///   this will determine the upper 64 bits of priority (the lower 64 bits
    ///   come from `submit_request` and `submit_single`)
    pub async fn open_file_with_priority(
        self: &Arc<Self>,
        path: &Path,
        base_priority: u64,
        file_size_bytes: &CachedFileSize,
    ) -> Result<FileScheduler> {
        let file_size_bytes = if let Some(size) = file_size_bytes.get() {
            u64::from(size)
        } else {
            let size = self.object_store.size(path).await?;
            if let Some(size) = NonZero::new(size) {
                file_size_bytes.set(size);
            }
            size
        };
        let reader = self
            .object_store
            .open_with_size(path, file_size_bytes as usize)
            .await?;
        let block_size = self.object_store.block_size() as u64;
        let max_iop_size = self.object_store.max_iop_size();
        Ok(FileScheduler {
            reader: reader.into(),
            block_size,
            root: self.clone(),
            base_priority,
            max_iop_size,
        })
    }

    /// Open a file with a default priority of 0
    ///
    /// See [`Self::open_file_with_priority`] for more information on the priority
    pub async fn open_file(
        self: &Arc<Self>,
        path: &Path,
        file_size_bytes: &CachedFileSize,
    ) -> Result<FileScheduler> {
        self.open_file_with_priority(path, 0, file_size_bytes).await
    }

    fn submit_request(
        &self,
        reader: Arc<dyn Reader>,
        request: Vec<Range<u64>>,
        priority: u128,
    ) -> impl Future<Output = Result<Vec<Bytes>>> + Send {
        let maybe_tasks = request
            .into_iter()
            .map(|task| {
                let reader = reader.clone();
                let queue = self.io_queue.clone();
                let run_fn = Box::new(move || {
                    async move {
                        reader
                            .get_range(task.start as usize..task.end as usize)
                            .map_err(Error::from)
                            .await
                    }
                    .boxed()
                });
                queue.submit(task, priority, run_fn)
            })
            .collect::<Result<Vec<_>>>();
        match maybe_tasks {
            Ok(tasks) => async move {
                let mut results = Vec::with_capacity(tasks.len());
                for task in tasks {
                    results.push(task.await?);
                }
                Ok(results)
            }
            .boxed(),
            Err(e) => async move { Err(e) }.boxed(),
        }
    }

    pub fn stats(&self) -> ScanStats {
        ScanStats::new(self.stats.as_ref())
    }
}

impl Drop for ScanScheduler {
    fn drop(&mut self) {
        // If the user is dropping the ScanScheduler then they _should_ be done with I/O.  This can happen
        // even when I/O is in progress if, for example, the user is dropping a scan mid-read because they found
        // the data they wanted (limit after filter or some other example).
        //
        // Closing the I/O queue will cancel any requests that have not yet been sent to the I/O loop.  However,
        // it will not terminate the I/O loop itself.  This is to help prevent deadlock and ensure that all I/O
        // requests that are submitted will terminate.
        //
        // In theory, this isn't strictly necessary, as callers should drop any task expecting I/O before they
        // drop the scheduler.  In practice, this can be difficult to do, and it is better to spend a little bit
        // of time letting the I/O loop drain so that we can avoid any potential deadlocks.
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
    max_iop_size: u64,
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

        let mut merged_requests = Vec::with_capacity(request.len());

        if !request.is_empty() {
            let mut curr_interval = request[0].clone();

            for req in request.iter().skip(1) {
                if is_close_together(&curr_interval, req, self.block_size) {
                    curr_interval.end = curr_interval.end.max(req.end);
                } else {
                    merged_requests.push(curr_interval);
                    curr_interval = req.clone();
                }
            }

            merged_requests.push(curr_interval);
        }

        let mut updated_requests = Vec::with_capacity(merged_requests.len());
        for req in merged_requests {
            if req.is_empty() {
                updated_requests.push(req);
            } else {
                let num_requests = (req.end - req.start).div_ceil(self.max_iop_size);
                let bytes_per_request = (req.end - req.start) / num_requests;
                for i in 0..num_requests {
                    let start = req.start + i * bytes_per_request;
                    let end = if i == num_requests - 1 {
                        // Last request is a bit bigger due to rounding
                        req.end
                    } else {
                        start + bytes_per_request
                    };
                    updated_requests.push(start..end);
                }
            }
        }

        self.root.stats.record_request(&updated_requests);

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
                    // We need to undo the coalescing and splitting done earlier
                    let start = orig_range.start as usize - byte_offset;
                    if orig_range.end <= updated_range.end {
                        // The original range is fully contained in the updated range, can do
                        // zero-copy slice
                        let end = orig_range.end as usize - byte_offset;
                        final_bytes.push(bytes_vec[updated_index].slice(start..end));
                    } else {
                        // The original read was split into multiple requests, need to copy
                        // back into a single buffer
                        let orig_size = orig_range.end - orig_range.start;
                        let mut merged_bytes = Vec::with_capacity(orig_size as usize);
                        merged_bytes.extend_from_slice(&bytes_vec[updated_index].slice(start..));
                        let mut copy_offset = merged_bytes.len() as u64;
                        while copy_offset < orig_size {
                            updated_index += 1;
                            let next_range = &updated_requests[updated_index];
                            let bytes_to_take =
                                (orig_size - copy_offset).min(next_range.end - next_range.start);
                            merged_bytes.extend_from_slice(
                                &bytes_vec[updated_index].slice(0..bytes_to_take as usize),
                            );
                            copy_offset += bytes_to_take;
                        }
                        final_bytes.push(Bytes::from(merged_bytes));
                    }
                    orig_index += 1;
                } else {
                    updated_index += 1;
                }
            }

            Ok(final_bytes)
        }
    }

    pub fn with_priority(&self, priority: u64) -> Self {
        Self {
            reader: self.reader.clone(),
            root: self.root.clone(),
            block_size: self.block_size,
            max_iop_size: self.max_iop_size,
            base_priority: priority,
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
    use lance_core::utils::tempfile::TempObjFile;
    use rand::RngCore;

    use object_store::{memory::InMemory, GetRange, ObjectStore as OSObjectStore};
    use tokio::{runtime::Handle, time::timeout};
    use url::Url;

    use crate::{
        object_store::{DEFAULT_DOWNLOAD_RETRY_COUNT, DEFAULT_MAX_IOP_SIZE},
        testing::MockObjectStore,
    };

    use super::*;

    #[tokio::test]
    async fn test_full_seq_read() {
        let tmp_file = TempObjFile::default();

        let obj_store = Arc::new(ObjectStore::local());

        // Write 1MiB of data
        const DATA_SIZE: u64 = 1024 * 1024;
        let mut some_data = vec![0; DATA_SIZE as usize];
        rand::rng().fill_bytes(&mut some_data);
        obj_store.put(&tmp_file, &some_data).await.unwrap();

        let config = SchedulerConfig::default_for_testing();

        let scheduler = ScanScheduler::try_new(obj_store, config).unwrap();

        let file_scheduler = scheduler
            .open_file(&tmp_file, &CachedFileSize::unknown())
            .await
            .unwrap();

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
    async fn test_split_coalesce() {
        let tmp_file = TempObjFile::default();

        let obj_store = Arc::new(ObjectStore::local());

        // Write 75MiB of data
        const DATA_SIZE: u64 = 75 * 1024 * 1024;
        let mut some_data = vec![0; DATA_SIZE as usize];
        rand::rng().fill_bytes(&mut some_data);
        obj_store.put(&tmp_file, &some_data).await.unwrap();

        let config = SchedulerConfig::default_for_testing();

        let scheduler = ScanScheduler::try_new(obj_store, config).unwrap();

        let file_scheduler = scheduler
            .open_file(&tmp_file, &CachedFileSize::unknown())
            .await
            .unwrap();

        // These 3 requests should be coalesced into a single I/O because they are within 4KiB
        // of each other
        let req =
            file_scheduler.submit_request(vec![50_000..51_000, 52_000..53_000, 54_000..55_000], 0);

        let bytes = req.await.unwrap();

        assert_eq!(bytes[0], &some_data[50_000..51_000]);
        assert_eq!(bytes[1], &some_data[52_000..53_000]);
        assert_eq!(bytes[2], &some_data[54_000..55_000]);

        assert_eq!(1, scheduler.stats().iops);

        // This should be split into 5 requests because it is so large
        let req = file_scheduler.submit_request(vec![0..DATA_SIZE], 0);
        let bytes = req.await.unwrap();
        assert!(bytes[0] == some_data, "data is not the same");

        assert_eq!(6, scheduler.stats().iops);

        // None of these requests are bigger than the max IOP size but they will be coalesced into
        // one IOP that is bigger and then split back into 2 requests that don't quite align with the original
        // ranges.
        let chunk_size = *DEFAULT_MAX_IOP_SIZE;
        let req = file_scheduler.submit_request(
            vec![
                10..chunk_size,
                chunk_size + 10..(chunk_size * 2) - 20,
                chunk_size * 2..(chunk_size * 2) + 10,
            ],
            0,
        );

        let bytes = req.await.unwrap();
        let chunk_size = chunk_size as usize;
        assert!(
            bytes[0] == some_data[10..chunk_size],
            "data is not the same"
        );
        assert!(
            bytes[1] == some_data[chunk_size + 10..(chunk_size * 2) - 20],
            "data is not the same"
        );
        assert!(
            bytes[2] == some_data[chunk_size * 2..(chunk_size * 2) + 10],
            "data is not the same"
        );
        assert_eq!(8, scheduler.stats().iops);

        let reads = (0..44)
            .map(|i| i * 1_000_000..(i + 1) * 1_000_000)
            .collect::<Vec<_>>();
        let req = file_scheduler.submit_request(reads, 0);
        let bytes = req.await.unwrap();
        for (i, bytes) in bytes.iter().enumerate() {
            assert!(
                bytes == &some_data[i * 1_000_000..(i + 1) * 1_000_000],
                "data is not the same"
            );
        }
        assert_eq!(11, scheduler.stats().iops);
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
            Some(500),
            None,
            false,
            false,
            1,
            DEFAULT_DOWNLOAD_RETRY_COUNT,
            None,
        ));

        let config = SchedulerConfig {
            io_buffer_size_bytes: 1024 * 1024,
        };

        let scan_scheduler = ScanScheduler::try_new(obj_store, config).unwrap();

        let file_scheduler = scan_scheduler
            .open_file(&Path::parse("foo").unwrap(), &CachedFileSize::new(1000))
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
                bytes_read_copy.fetch_add(num_bytes, Ordering::Release);
                let location = location.clone();
                let base_store = base_store.clone();
                async move { base_store.get_opts(&location, options).await }.boxed()
            });
        let obj_store = Arc::new(ObjectStore::new(
            Arc::new(obj_store),
            Url::parse("mem://").unwrap(),
            Some(500),
            None,
            false,
            false,
            1,
            DEFAULT_DOWNLOAD_RETRY_COUNT,
            None,
        ));

        let config = SchedulerConfig {
            io_buffer_size_bytes: 10,
        };

        let scan_scheduler = ScanScheduler::try_new(obj_store.clone(), config).unwrap();

        let file_scheduler = scan_scheduler
            .open_file(&Path::parse("foo").unwrap(), &CachedFileSize::new(100000))
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

        let scan_scheduler = ScanScheduler::try_new(obj_store, config).unwrap();
        let file_scheduler = scan_scheduler
            .open_file(&Path::parse("foo").unwrap(), &CachedFileSize::new(100000))
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
        let scan_scheduler = ScanScheduler::try_new(obj_store.clone(), config).unwrap();
        let file_scheduler = scan_scheduler
            .open_file(&some_path, &CachedFileSize::unknown())
            .await
            .unwrap();

        let mut futs = Vec::with_capacity(10000);
        for idx in 0..10000 {
            futs.push(file_scheduler.submit_single(idx..idx + 1, idx));
        }

        for fut in futs {
            fut.await.unwrap();
        }
    }
}
