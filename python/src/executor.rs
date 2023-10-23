// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::mpsc::RecvTimeoutError;

use futures::Future;
use pyo3::{exceptions::PyRuntimeError, PyResult, Python};

pub const SIGNAL_CHECK_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

/// A wrapper around tokio runtime.
///
/// This is used to spawn tasks in the background and wait synchronously for them
/// to complete. This is important for cases where we want to avoid nested
/// block_on() calls.
///
/// The methods also make sure that the GIL is released before spawning the task.
pub struct BackgroundExecutor {
    pub runtime: tokio::runtime::Runtime,
}

impl BackgroundExecutor {
    /// Creates a tokio runtime and spawns a thread to run it.
    pub fn new() -> Self {
        // Create a new Runtime to run tasks
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("lance_background_thread")
            .build()
            .expect("Creating Tokio runtime");

        Self { runtime }
    }

    /// Spawn a task and wait for it to complete.
    pub fn spawn<T>(&self, py: Option<Python<'_>>, task: T) -> PyResult<T::Output>
    where
        T: Future + Send + 'static,
        T::Output: Send + 'static,
    {
        if let Some(py) = py {
            py.allow_threads(|| self.spawn_impl(task))
        } else {
            // Python::with_gil is a no-op if the GIL is already held by the thread.
            Python::with_gil(|py| py.allow_threads(|| self.spawn_impl(task)))
        }
    }

    fn spawn_impl<T>(&self, task: T) -> PyResult<T::Output>
    where
        T: Future + Send + 'static,
        T::Output: Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::channel::<T::Output>();

        let fut = Box::pin(async move {
            let task_output = task.await;
            tokio::task::spawn_blocking(move || {
                tx.send(task_output).ok();
            })
            .await
            .unwrap();
        });

        let handle = self.runtime.spawn(fut);

        loop {
            // Check for keyboard interrupts
            match Python::with_gil(|py| py.check_signals()) {
                Ok(_) => {}
                Err(err) => {
                    handle.abort();
                    return Err(err);
                }
            }
            // Wait for 100ms before checking signals again
            match rx.recv_timeout(SIGNAL_CHECK_INTERVAL) {
                Ok(output) => return Ok(output),
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => {
                    handle.abort();
                    return Err(PyRuntimeError::new_err("Task was aborted"));
                }
            }
        }
    }

    /// Spawn a task in the background
    pub fn spawn_background<T>(&self, py: Option<Python<'_>>, task: T)
    where
        T: Future + Send + 'static,
        T::Output: Send + 'static,
    {
        if let Some(py) = py {
            py.allow_threads(|| {
                self.runtime.spawn(task);
            })
        } else {
            // Python::with_gil is a no-op if the GIL is already held by the thread.
            Python::with_gil(|py| {
                py.allow_threads(|| {
                    self.runtime.spawn(task);
                })
            })
        }
    }

    /// Block on a future and wait for it to complete.
    ///
    /// This helper method also frees the GIL before blocking.
    pub fn block_on<F: Future + Send>(
        &self,
        py: Option<Python<'_>>,
        future: F,
    ) -> PyResult<F::Output>
    where
        F::Output: Send,
    {
        let future = Self::result_or_interrupt(future);
        if let Some(py) = py {
            py.allow_threads(move || self.runtime.block_on(future))
        } else {
            // Python::with_gil is a no-op if the GIL is already held by the thread.
            Python::with_gil(|py| py.allow_threads(|| self.runtime.block_on(future)))
        }
    }

    async fn result_or_interrupt<F>(future: F) -> PyResult<F::Output>
    where
        F: Future + Send,
        F::Output: Send,
    {
        let interrupt_future = async {
            loop {
                // Check for keyboard interrupts
                match Python::with_gil(|py| py.check_signals()) {
                    Ok(_) => {
                        // Wait for 100ms before checking signals again
                        tokio::time::sleep(SIGNAL_CHECK_INTERVAL).await;
                    }
                    Err(err) => return Err(err),
                }
            }
        };

        tokio::select! {
            result = future => Ok(result),
            err = interrupt_future => err,
        }
    }
}
