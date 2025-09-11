// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::LazyLock;
use std::time::Duration;

use crate::Result;

use futures::{Future, FutureExt};
use tokio::runtime::{Builder, Runtime};
use tracing::Span;

pub fn get_num_compute_intensive_cpus() -> usize {
    if let Ok(user_specified) = std::env::var("LANCE_CPU_THREADS") {
        return user_specified.parse().unwrap();
    }

    let cpus = num_cpus::get();

    if cpus <= *IO_CORE_RESERVATION {
        // If the user is not setting a custom value for LANCE_IO_CORE_RESERVATION then we don't emit
        // a warning because they're just on a small machine and there isn't much they can do about it.
        if cpus > 2 {
            log::warn!(
                "Number of CPUs is less than or equal to the number of IO core reservations. \
                This is not a supported configuration. using 1 CPU for compute intensive tasks."
            );
        }
        return 1;
    }

    num_cpus::get() - *IO_CORE_RESERVATION
}

pub static IO_CORE_RESERVATION: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("LANCE_IO_CORE_RESERVATION")
        .unwrap_or("2".to_string())
        .parse()
        .unwrap()
});

pub static CPU_RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    Builder::new_multi_thread()
        .thread_name("lance-cpu")
        .max_blocking_threads(get_num_compute_intensive_cpus())
        .worker_threads(1)
        // keep the thread alive "forever"
        .thread_keep_alive(Duration::from_secs(u64::MAX))
        .build()
        .unwrap()
});

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
