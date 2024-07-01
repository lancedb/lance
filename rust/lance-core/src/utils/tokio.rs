// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{sync::atomic::AtomicUsize, time::Duration};

use crate::Result;

use core_affinity::CoreId;
use futures::{Future, FutureExt};
use tokio::runtime::{Builder, Runtime};
use tracing::Span;

lazy_static::lazy_static! {
    // TODO: handle systems with less than 2 cores
    pub static ref IO_CORE_RESERVATION: usize = std::env::var("LANCE_IO_CORE_RESERVATION").unwrap_or("2".to_string()).parse().unwrap();

    pub static ref CPU_RUNTIME_THREAD_START_COUNTER: AtomicUsize = AtomicUsize::new(*IO_CORE_RESERVATION);

    pub static ref CPU_RUNTIME: Runtime = Builder::new_multi_thread()
        .thread_name("lance-cpu")
        .max_blocking_threads(num_cpus::get() - *IO_CORE_RESERVATION - 1)
        .worker_threads(1)
        .on_thread_start(|| {
            let id = CPU_RUNTIME_THREAD_START_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            core_affinity::set_for_current(CoreId{id});
        })
        // keep the thread alive "forever"
        .thread_keep_alive(Duration::from_secs(u64::MAX))
        .build()
        .unwrap();
}

/// Spawn a CPU intensive task
///
/// This task will be put onto a thread pool dedicated for CPU-intensive work
/// This keeps the tokio thread pool free so that we can always be ready to service
/// cheap I/O & control requests.
///
/// This can also be used to convert a big chunk of synchronous work into a future
/// so that it can be run in parallel with something like StreamExt::buffered()
pub fn spawn_cpu<F: FnOnce() -> Result<R> + Send + 'static, R: Send + 'static>(
    func: F,
) -> impl Future<Output = Result<R>> {
    let (send, recv) = tokio::sync::oneshot::channel();
    // Propagate the current span into the task
    let span = Span::current();
    CPU_RUNTIME.spawn_blocking(move || {
        let _span_guard = span.enter();
        let result = func();
        let _ = send.send(result);
    });
    recv.map(|res| res.unwrap())
}
