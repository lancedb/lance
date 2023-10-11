use crate::Result;

use futures::{Future, FutureExt};

/// Spawn a CPU intensive task
///
/// This task will be put onto the rayon thread pool dedicated for CPU-intensive work
pub fn spawn_cpu<F: FnOnce() -> Result<R> + Send + 'static, R: Send + 'static>(
    func: F,
) -> impl Future<Output = Result<R>> {
    let (send, recv) = tokio::sync::oneshot::channel();
    rayon::spawn(|| {
        let result = func();
        let _ = send.send(result);
    });
    // res will be an error if func() panic'd.  We propagate that panic via unwrap
    // in the same way we do with a mutex
    recv.map(|res| res.unwrap())
}
