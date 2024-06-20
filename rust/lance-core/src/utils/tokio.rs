// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::time::Duration;

use crate::Result;

use futures::{Future, FutureExt};
use tokio::runtime::{Builder, Runtime};
use tracing::Span;

fn get_num_compute_intensive_cpus() -> usize {
    let cpus = num_cpus::get();

    if cpus <= *IO_CORE_RESERVATION {
        // on systems with only 1 CPU there is no point in warning
        if cpus > 1 {
            log::warn!(
                "Number of CPUs is less than or equal to the number of IO core reservations. \
                This is not a supported configuration. using 1 CPU for compute intensive tasks."
            );
        }
        return 1;
    }

    num_cpus::get() - *IO_CORE_RESERVATION
}

lazy_static::lazy_static! {
    pub static ref IO_CORE_RESERVATION: usize = std::env::var("LANCE_IO_CORE_RESERVATION").unwrap_or("2".to_string()).parse().unwrap();

    pub static ref CPU_RUNTIME: Runtime = Builder::new_multi_thread()
        .thread_name("lance-cpu")
        .max_blocking_threads(get_num_compute_intensive_cpus())
        .worker_threads(1)
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
