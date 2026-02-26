// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A lightweight I/O scheduler primarily intended for use with I/O uring.
//!
//! This scheduler attempts to avoid any kind of task switching whenever possible
//! to minimize context switching overhead.
//!
//! There are a few limitations compared to the standard scheduler:
//!
//! * There is no concurrency limit.  The scheduler will allow as many IOPS to run
//!   as possible as long as the backpressure throttle is not exceeded.
//! * There is no "babysitting" of IOPS.  An I/O task will only be polled when its
//!   future is polled.  The standard scheduler will `spawn` I/O tasks and so they
//!   are always polled by tokio's runtime.  This is important for operations like
//!   cloud requests where intermittent polling is required to clear out network
//!   buffers and keep the TCP connection moving.

use std::{
    collections::{BinaryHeap, HashMap},
    fmt::Debug,
    future::Future,
    ops::Range,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, MutexGuard,
    },
    task::{Context, Poll, Waker},
    time::Instant,
};

use bytes::Bytes;
use lance_core::{Error, Result};
use snafu::location;

use super::{BACKPRESSURE_DEBOUNCE, BACKPRESSURE_MIN};

type RunFn = Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = Result<Bytes>> + Send>> + Send>;

/// The state of an I/O task
///
/// The state machine is as follows:
///
/// * `Broken` - The task is in an error state and cannot be run, should never happen
/// * `Initial` - The task has been submitted but does not have a backpressure reservation
/// * `Reserved` - The task has a backpressure reservation
/// * `Running` - The task is running and has a future to poll
/// * `Finished` - The task has finished and has a result
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
    },
    Finished {
        backpressure_reservation: BackpressureReservation,
        data: Result<Bytes>,
    },
}

/// A custom error type that might have a backpressure reservation
///
/// This is used instead of Lance's standard error type so we can ensure
/// we release the reservation before returning the error.
struct BrokenTaskError {
    message: String,
    backpressure_reservation: Option<BackpressureReservation>,
}

/// The result type corresponding to BrokenTaskError
type TaskResult = std::result::Result<(), BrokenTaskError>;

impl BrokenTaskError {
    // Create a BrokenTaskError from a task state
    //
    // This will capture any backpressure reservation the task has and put it into the
    // error so we make sure to release it when returning the error.
    fn new(task_state: TaskState, message: String) -> Self {
        match task_state {
            TaskState::Reserved {
                backpressure_reservation,
                ..
            }
            | TaskState::Running {
                backpressure_reservation,
                ..
            }
            | TaskState::Finished {
                backpressure_reservation,
                ..
            } => Self {
                message,
                backpressure_reservation: Some(backpressure_reservation),
            },
            TaskState::Broken | TaskState::Initial { .. } => Self {
                message,
                backpressure_reservation: None,
            },
        }
    }
}

/// An I/O task represents a single read operation
struct IoTask {
    /// The unique identifier of the task (only used for debugging)
    id: u64,
    /// The number of bytes to read
    num_bytes: u64,
    /// The priority of the task, lower values are higher priority
    priority: u128,
    /// The current state of the task
    state: TaskState,
}

impl IoTask {
    fn is_reserved(&self) -> bool {
        !matches!(self.state, TaskState::Initial { .. })
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

    fn reserve(&mut self, backpressure_reservation: BackpressureReservation) -> TaskResult {
        let state = std::mem::replace(&mut self.state, TaskState::Broken);
        let TaskState::Initial { idle_waker, run_fn } = state else {
            return Err(BrokenTaskError::new(
                state,
                format!("Task with id {} not in initial state", self.id),
            ));
        };
        self.state = TaskState::Reserved {
            idle_waker,
            backpressure_reservation,
            run_fn,
        };
        Ok(())
    }

    fn start(&mut self) -> TaskResult {
        let state = std::mem::replace(&mut self.state, TaskState::Broken);
        let TaskState::Reserved {
            backpressure_reservation,
            idle_waker,
            run_fn,
        } = state
        else {
            return Err(BrokenTaskError::new(
                state,
                format!("Task with id {} not in reserved state", self.id),
            ));
        };
        let inner = run_fn();
        self.state = TaskState::Running {
            backpressure_reservation,
            inner,
        };

        // If someone is already waiting for this task let them know it is now running
        // so they can poll it
        if let Some(idle_waker) = idle_waker {
            idle_waker.wake();
        }
        Ok(())
    }

    fn poll(&mut self, cx: &mut Context<'_>) -> Poll<()> {
        match &mut self.state {
            TaskState::Broken => Poll::Ready(()),
            TaskState::Initial { idle_waker, .. } | TaskState::Reserved { idle_waker, .. } => {
                idle_waker.replace(cx.waker().clone());
                Poll::Pending
            }
            TaskState::Running {
                inner,
                backpressure_reservation,
            } => match inner.as_mut().poll(cx) {
                Poll::Ready(data) => {
                    self.state = TaskState::Finished {
                        data,
                        backpressure_reservation: *backpressure_reservation,
                    };
                    Poll::Ready(())
                }
                Poll::Pending => Poll::Pending,
            },
            TaskState::Finished { .. } => Poll::Ready(()),
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
    fn new(max_bytes: u64, max_concurrency: u64) -> Self {
        if max_bytes > i64::MAX as u64 {
            // This is unlikely to ever be an issue
            panic!("Max bytes must be less than {}", i64::MAX);
        }
        Self {
            start: Instant::now(),
            last_warn: AtomicU64::new(0),
            bytes_available: max_bytes as i64,
            priorities_in_flight: PrioritiesInFlight::new(max_concurrency),
        }
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
    backpressure_throttle: Box<dyn BackpressureThrottle>,
    pending_tasks: BinaryHeap<TaskEntry>,
    tasks: HashMap<u64, IoTask>,
    next_task_id: u64,
}

impl IoQueueState {
    fn new(max_concurrency: u64, max_bytes: u64) -> Self {
        Self {
            backpressure_throttle: Box::new(SimpleBackpressureThrottle::new(
                max_bytes,
                max_concurrency,
            )),
            pending_tasks: BinaryHeap::new(),
            tasks: HashMap::new(),
            next_task_id: 0,
        }
    }

    // If a task is in an unexpected state then we need to release any reservations that were made
    // before we return an error.
    //
    // Note: this is perhaps a bit paranoid as a task should never be in an unexpected state.
    fn handle_result(&mut self, result: TaskResult) -> Result<()> {
        if let Err(error) = result {
            if let Some(reservation) = error.backpressure_reservation {
                self.backpressure_throttle.release(reservation);
            }
            Err(Error::Internal {
                message: error.message,
                location: location!(),
            })
        } else {
            Ok(())
        }
    }
}

/// A queue of I/O tasks to be shared between the I/O scheduler and the I/O decoder.
///
/// The queue is protected by two different throttles.  The first controls memory backpressure, and
/// will only allow a certain number of bytes to be allocated for reads.  This throttle is released
/// as soon as the decoder consumes the bytes (not when the bytes have been fully processed).  This
/// throttle is currently scoped to the scheduler and not shared across the process.  This will likely
/// change in the future.
///
/// The second throttle controls how many IOPS can be issued concurrently.  This throttle is released
/// as soon as the IOP is finished.  This throttle has both a local per-scheduler limit and also a
/// process-wide limit.
///
/// Note: unlike the standard scheduler, there is no dedicated I/O loop thread.  If the decoder is not
/// polling the I/O tasks then nothing else will.  This scheduler is currently intended for use with I/O
/// uring where I/O tasks are bunched together and polling one task advances all outstanding I/O.  It
/// would not be suitable for cloud storage where each task is an independent HTTP request and needs to
/// be polled individually (though presumably one could use I/O uring for networked cloud storage some
/// day as well)
pub(super) struct IoQueue {
    state: Arc<Mutex<IoQueueState>>,
}

impl IoQueue {
    pub fn new(max_concurrency: u64, max_bytes: u64) -> Self {
        Self {
            state: Arc::new(Mutex::new(IoQueueState::new(max_concurrency, max_bytes))),
        }
    }

    fn push(&self, mut task: IoTask, mut state: MutexGuard<IoQueueState>) -> Result<()> {
        let task_id = task.id;
        if let Some(reservation) = state
            .backpressure_throttle
            .try_acquire(task.num_bytes, task.priority)
        {
            state.handle_result(task.reserve(reservation))?;
            state.handle_result(task.start())?;
            state.tasks.insert(task_id, task);
            return Ok(());
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

    // When a task completes we should check to see if any other tasks are now runnable
    fn on_task_complete(&self, mut state: MutexGuard<IoQueueState>) -> Result<()> {
        let state_ref = &mut *state;
        let mut task_result = TaskResult::Ok(());
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
                if let Err(e) = task.reserve(reservation) {
                    task_result = Err(e);
                    break;
                }
            }
            state_ref.pending_tasks.pop();
            if let Err(e) = task.start() {
                task_result = Err(e);
                break;
            }
        }
        state_ref.handle_result(task_result)
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
        match task.poll(cx) {
            Poll::Ready(_) => {
                let task = state.tasks.remove(&task_id).unwrap();
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

    pub(super) fn close(&self) {
        let mut state = self.state.lock().unwrap();
        for task in std::mem::take(&mut state.tasks).values_mut() {
            task.cancel();
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::oneshot;

    #[tokio::test]
    async fn test_priority_ordering() {
        // Backpressure budget of 10 bytes: only one 10-byte task runs at a time.
        let queue = Arc::new(IoQueue::new(128, 10));

        // Records the priority of each task when its run_fn is invoked (i.e. when
        // the task transitions to Running).
        let start_order: Arc<Mutex<Vec<u128>>> = Arc::new(Mutex::new(Vec::new()));

        // Helper: builds a RunFn that records `prio` in start_order and then
        // waits on the oneshot receiver for its result bytes.
        let make_run_fn =
            |prio: u128, rx: oneshot::Receiver<Bytes>, order: Arc<Mutex<Vec<u128>>>| -> RunFn {
                Box::new(move || {
                    order.lock().unwrap().push(prio);
                    Box::pin(async move { Ok(rx.await.unwrap()) })
                })
            };

        // Submit a blocker task (priority 0, 10 bytes).
        // It starts immediately because there is enough backpressure budget.
        let (blocker_tx, blocker_rx) = oneshot::channel();
        let blocker = queue
            .clone()
            .submit(0..10, 0, make_run_fn(0, blocker_rx, start_order.clone()))
            .unwrap();

        // Submit four tasks with out-of-order priorities.
        // All are queued because the blocker consumed the full budget.
        let (tx_30, rx_30) = oneshot::channel();
        let h30 = queue
            .clone()
            .submit(0..10, 30, make_run_fn(30, rx_30, start_order.clone()))
            .unwrap();

        let (tx_10, rx_10) = oneshot::channel();
        let h10 = queue
            .clone()
            .submit(0..10, 10, make_run_fn(10, rx_10, start_order.clone()))
            .unwrap();

        let (tx_50, rx_50) = oneshot::channel();
        let h50 = queue
            .clone()
            .submit(0..10, 50, make_run_fn(50, rx_50, start_order.clone()))
            .unwrap();

        let (tx_20, rx_20) = oneshot::channel();
        let h20 = queue
            .clone()
            .submit(0..10, 20, make_run_fn(20, rx_20, start_order.clone()))
            .unwrap();

        // Only the blocker has started so far.
        assert_eq!(*start_order.lock().unwrap(), vec![0]);

        // Complete the blocker -> frees budget -> starts priority 10 (lowest value = highest priority).
        blocker_tx.send(Bytes::from_static(b"x")).unwrap();
        blocker.await.unwrap();
        assert_eq!(*start_order.lock().unwrap(), vec![0, 10]);

        // Complete priority 10 -> starts priority 20.
        tx_10.send(Bytes::from_static(b"x")).unwrap();
        h10.await.unwrap();
        assert_eq!(*start_order.lock().unwrap(), vec![0, 10, 20]);

        // Complete priority 20 -> starts priority 30.
        tx_20.send(Bytes::from_static(b"x")).unwrap();
        h20.await.unwrap();
        assert_eq!(*start_order.lock().unwrap(), vec![0, 10, 20, 30]);

        // Complete priority 30 -> starts priority 50.
        tx_30.send(Bytes::from_static(b"x")).unwrap();
        h30.await.unwrap();
        assert_eq!(*start_order.lock().unwrap(), vec![0, 10, 20, 30, 50]);

        // Complete priority 50 -> no more pending tasks.
        tx_50.send(Bytes::from_static(b"x")).unwrap();
        h50.await.unwrap();
        assert_eq!(*start_order.lock().unwrap(), vec![0, 10, 20, 30, 50]);
    }
}
