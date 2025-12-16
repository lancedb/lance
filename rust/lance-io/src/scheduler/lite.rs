use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    fmt::Debug,
    future::Future,
    ops::Range,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, LazyLock, Mutex, MutexGuard,
    },
    task::{Context, Poll, Waker},
    time::Instant,
};

use bytes::Bytes;
use futures::task::noop_waker;
use lance_core::{Error, Result};
use snafu::location;

use super::{BACKPRESSURE_DEBOUNCE, BACKPRESSURE_MIN, DEFAULT_PROCESS_IOPS_LIMIT};

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

pub(super) struct IoTask {
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
                DEFAULT_PROCESS_IOPS_LIMIT as u64
            })
        })
        .unwrap_or(DEFAULT_PROCESS_IOPS_LIMIT as u64);
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
pub(super) struct IoQueue {
    state: Arc<Mutex<IoQueueState>>,
}

impl IoQueue {
    pub fn try_new(max_concurrency: u64, max_bytes: u64) -> Result<Self> {
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

    pub(super) fn submit(
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

    pub(super) fn close(&self) {
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

pub(super) async fn babysitter_loop(queue: Arc<IoQueue>) {
    loop {
        BabysitFuture {
            queue: queue.as_ref(),
        }
        .await;
    }
}

pub(super) struct TaskHandle {
    task_id: u64,
    queue: Arc<IoQueue>,
}

impl Future for TaskHandle {
    type Output = Result<Bytes>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.queue.poll(self.task_id, cx)
    }
}
