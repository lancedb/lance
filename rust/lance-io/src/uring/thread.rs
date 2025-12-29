// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dedicated thread for io_uring operations.
//!
//! This module provides a background thread that owns an io_uring instance
//! and processes read requests from a channel. Readers send requests via
//! an MPSC channel, and the thread handles submission and completion processing.

use super::requests::IoRequest;
use super::DEFAULT_URING_QUEUE_DEPTH;
use io_uring::{opcode, types, IoUring};
use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

/// Handle to the io_uring background thread.
///
/// This provides a channel sender for submitting read requests to the thread.
pub(super) struct UringThreadHandle {
    pub request_tx: SyncSender<Arc<IoRequest>>,
}

/// Lazy-initialized io_uring thread pool.
///
/// Multiple threads are spawned on first access and run until process exit.
pub(super) static URING_THREADS: LazyLock<Vec<UringThreadHandle>> = LazyLock::new(|| {
    let queue_depth = get_queue_depth();
    let thread_count = get_thread_count();

    let mut threads = Vec::with_capacity(thread_count);

    for i in 0..thread_count {
        let (tx, rx) = sync_channel(queue_depth);

        std::thread::Builder::new()
            .name(format!("lance-uring-{}", i))
            .spawn(move || run_uring_thread(rx, queue_depth, i))
            .expect("Failed to spawn io_uring thread");

        threads.push(UringThreadHandle { request_tx: tx });
    }

    log::info!(
        "io_uring thread pool spawned ({} threads, queue_depth={})",
        thread_count,
        queue_depth
    );

    threads
});

/// Atomic counter for round-robin thread selection.
pub(super) static THREAD_SELECTOR: AtomicU64 = AtomicU64::new(0);

/// Counter for generating unique user_data values.
///
/// Each io_uring operation needs a unique user_data ID to match completions
/// with their corresponding requests.
static USER_DATA_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Counter for requests that have been submitted to the thread but not yet received.
///
/// This tracks requests sitting in the channel queue waiting to be received by the thread.
pub(super) static SUBMITTED_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Default batch size for submission - how many requests to batch before calling submit().
const DEFAULT_SUBMIT_BATCH_SIZE: usize = 128;

/// Default number of io_uring threads.
const DEFAULT_URING_THREAD_COUNT: usize = 2;

/// Get the configured queue depth from environment variable.
fn get_queue_depth() -> usize {
    std::env::var("LANCE_URING_QUEUE_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_URING_QUEUE_DEPTH)
}

/// Get the configured poll timeout from environment variable.
fn get_poll_timeout() -> Duration {
    let timeout_ms = std::env::var("LANCE_URING_POLL_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    Duration::from_millis(timeout_ms)
}

/// Get the configured submit batch size from environment variable.
fn get_submit_batch_size() -> usize {
    std::env::var("LANCE_URING_SUBMIT_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SUBMIT_BATCH_SIZE)
}

/// Get the configured number of uring threads from environment variable.
fn get_thread_count() -> usize {
    std::env::var("LANCE_URING_THREAD_COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_URING_THREAD_COUNT)
}

/// Main loop for the io_uring thread.
///
/// This thread:
/// 1. Receives requests from the channel
/// 2. Submits them to io_uring
/// 3. Processes completions
/// 4. Wakes futures via their wakers
fn run_uring_thread(request_rx: Receiver<Arc<IoRequest>>, queue_depth: usize, thread_id: usize) {
    // Pin to core if configured
    // Format: LANCE_URING_CORE=0,1,2 or LANCE_URING_CORE=0 (applies to thread 0 only)
    if let Ok(core_str) = std::env::var("LANCE_URING_CORE") {
        let cores: Vec<&str> = core_str.split(',').collect();
        if let Some(core_str) = cores.get(thread_id).or_else(|| cores.first()) {
            if let Ok(core) = core_str.parse() {
                if let Err(e) = pin_to_core(core) {
                    log::warn!(
                        "Failed to pin io_uring thread {} to core {}: {}",
                        thread_id,
                        core,
                        e
                    );
                } else {
                    log::info!("io_uring thread {} pinned to core {}", thread_id, core);
                }
            }
        }
    }

    // Create local io_uring instance
    let mut ring = IoUring::builder()
        // .setup_sqpoll(100)
        .build(queue_depth as u32)
        .expect("Failed to create io_uring");

    let mut pending: HashMap<u64, Arc<IoRequest>> = HashMap::with_capacity(queue_depth);
    let poll_timeout = get_poll_timeout();
    let submit_batch_size = get_submit_batch_size();
    let mut last_log = Instant::now();
    let log_interval = Duration::from_millis(100);
    let mut completed_iops = 0usize;
    let mut completed_sectors = 0usize;
    let mut min_in_flight = usize::MAX;

    loop {
        // Track minimum in-flight count
        let in_flight = pending.len();
        min_in_flight = min_in_flight.min(in_flight);

        // Log in-flight requests every 100ms
        let now = Instant::now();
        if now.duration_since(last_log) >= log_interval {
            let submitted = SUBMITTED_COUNTER.load(Ordering::Relaxed);
            log::info!(
                "io_uring[{}]: {} submitted, {} in flight (min {}), {} iops completed, {} sectors completed",
                thread_id,
                submitted,
                in_flight,
                min_in_flight,
                completed_iops,
                completed_sectors
            );
            last_log = now;
            completed_iops = 0; // Reset counter after logging
            completed_sectors = 0; // Reset counter after logging
            min_in_flight = usize::MAX; // Reset min tracker
        }

        // Process all available completions first
        let completions = process_completions(&mut ring, &mut pending);
        match completions {
            Ok(count) => {
                completed_iops += count.iops;
                completed_sectors += count.sectors;
            }
            Err(e) => {
                log::error!("Error processing io_uring completions: {}", e);
            }
        }

        min_in_flight = min_in_flight.min(pending.len());

        // Batch submit requests - keep pulling from channel and pushing to SQ
        // until we hit batch size or channel is empty
        let mut batch_count = 0;
        loop {
            // Try to receive new request
            // Use recv_timeout only when pending is empty, otherwise use try_recv
            let recv_result = if pending.is_empty() && batch_count == 0 {
                // No operations in flight and no batch started - we can afford to wait with timeout
                request_rx.recv_timeout(poll_timeout).map_err(|e| match e {
                    RecvTimeoutError::Timeout => std::sync::mpsc::TryRecvError::Empty,
                    RecvTimeoutError::Disconnected => std::sync::mpsc::TryRecvError::Disconnected,
                })
            } else {
                // Operations in flight or batch in progress - busy loop with try_recv
                request_rx.try_recv()
            };

            match recv_result {
                Ok(request) => {
                    // Decrement submitted counter when we receive the request from channel
                    SUBMITTED_COUNTER.fetch_sub(1, Ordering::Relaxed);

                    // Push to submission queue (but don't submit yet)
                    if let Err(e) = push_to_sq(&mut ring, &mut pending, request) {
                        log::error!("Failed to push to io_uring SQ: {}", e);
                    } else {
                        batch_count += 1;
                    }

                    // Break if we've hit the batch size limit
                    if batch_count >= submit_batch_size {
                        break;
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // No more requests in channel - break to submit the batch
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    // All senders dropped - submit batch and shutdown
                    if batch_count > 0 {
                        if let Err(e) = ring.submit() {
                            log::error!(
                                "io_uring[{}]: Failed to submit io_uring batch: {}",
                                thread_id,
                                e
                            );
                        }
                    }
                    log::info!(
                        "io_uring thread {} shutting down (channel disconnected)",
                        thread_id
                    );
                    return;
                }
            }
        }

        // Submit the batch if we have any requests
        if batch_count > 0 {
            if let Err(e) = ring.submit() {
                log::error!(
                    "Failed to submit io_uring batch of {} requests: {}",
                    batch_count,
                    e
                );
            }
        }
    }
}

/// Push a read request to the io_uring submission queue (without submitting).
///
/// This generates a unique user_data ID, prepares the read operation,
/// and pushes it to the SQ. The caller is responsible for calling ring.submit().
fn push_to_sq(
    ring: &mut IoUring,
    pending: &mut HashMap<u64, Arc<IoRequest>>,
    request: Arc<IoRequest>,
) -> io::Result<()> {
    // Generate unique user_data
    let user_data = USER_DATA_COUNTER.fetch_add(1, Ordering::Relaxed);

    // Get buffer pointer from request state
    let buffer_ptr = {
        let state = request.state.lock().unwrap();
        state.buffer.as_ptr() as *mut u8
    };

    // Prepare read operation
    let read_op = opcode::Read::new(types::Fd(request.fd), buffer_ptr, request.length as u32)
        .offset(request.offset);

    // Get submission queue
    let mut sq = ring.submission();

    // Check if SQ has space
    if sq.is_full() {
        drop(sq);
        return Err(io::Error::new(
            io::ErrorKind::WouldBlock,
            "io_uring submission queue full",
        ));
    }

    // Push to SQ
    unsafe {
        sq.push(&read_op.build().user_data(user_data))
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "Failed to push to SQ"))?;
    }
    drop(sq);

    // Track request in pending map
    pending.insert(user_data, request);

    Ok(())
}

struct CompletionStats {
    iops: usize,
    sectors: usize,
}

/// Process all available completions from the io_uring.
///
/// This iterates through the completion queue, matches completions to requests,
/// updates their state, and wakes any waiting futures.
///
/// Returns the number of completions processed.
fn process_completions(
    ring: &mut IoUring,
    pending: &mut HashMap<u64, Arc<IoRequest>>,
) -> io::Result<CompletionStats> {
    let mut iops = 0;
    let mut sectors = 0;

    // Process all available completions
    for cqe in ring.completion() {
        let user_data = cqe.user_data();
        let result = cqe.result();

        // Look up request
        if let Some(request) = pending.remove(&user_data) {
            let mut state = request.state.lock().unwrap();
            state.completed = true;

            // Handle result
            if result < 0 {
                state.err = Some(io::Error::from_raw_os_error(-result));
            } else if request.length > 0 {
                let first_sector = request.offset / 4096;
                let last_sector = (request.offset + request.length as u64 - 1) / 4096;
                let num_sectors = (last_sector - first_sector + 1) as usize;
                sectors += num_sectors;
            }

            // Wake the future if it's waiting
            if let Some(waker) = state.waker.take() {
                drop(state); // Release lock before waking
                waker.wake();
            }

            iops += 1;
        } else {
            log::warn!("Received completion for unknown user_data: {}", user_data);
        }
    }

    Ok(CompletionStats { iops, sectors })
}

/// Pin the current thread to a specific CPU core.
///
/// This uses Linux's sched_setaffinity to improve cache locality.
fn pin_to_core(core: usize) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};
        use std::mem;

        unsafe {
            let mut cpuset: cpu_set_t = mem::zeroed();
            CPU_ZERO(&mut cpuset);
            CPU_SET(core, &mut cpuset);

            let result = sched_setaffinity(
                0, // current thread
                mem::size_of::<cpu_set_t>(),
                &cpuset,
            );

            if result != 0 {
                return Err(io::Error::last_os_error());
            }
        }

        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = core;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "CPU pinning only supported on Linux",
        ))
    }
}
