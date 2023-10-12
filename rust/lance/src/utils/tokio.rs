use crate::Result;

use futures::{Future, FutureExt};
use tokio::runtime::{Builder, Runtime};

lazy_static::lazy_static! {
    static ref CPU_RUNTIME: Runtime = Builder::new_multi_thread()
        .thread_name("lance-cpu")
        .max_blocking_threads(num_cpus::get())
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
    CPU_RUNTIME.spawn_blocking(|| {
        let result = func();
        let _ = send.send(result);
    });
    recv.map(|res| res.unwrap())
}
