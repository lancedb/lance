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

//! Another thread that can run async tasks in the background.

// See: https://thenewstack.io/using-rustlangs-async-tokio-runtime-for-cpu-bound-tasks/

use std::sync::{Arc, Mutex};

use futures::{future::BoxFuture, Future};

/// A background executor which allows running tasks on a tokio runtime
/// in a separate thread.
pub struct BackgroundExecutor {
    state: Arc<Mutex<State>>,
}

struct State {
    /// Channel for requests -- the dedicated executor takes requests                                                                  
    /// from here and runs them.                                                                                                       
    requests: Option<std::sync::mpsc::Sender<BoxFuture<'static, ()>>>,

    /// Thread which has a different Tokio runtime
    /// installed and spawns tasks there                                                                                            
    _thread: std::thread::JoinHandle<()>,
}

impl BackgroundExecutor {
    /// Creates a tokio runtime and spawns a thread to run it.                                                                                      
    pub fn new() -> Self {
        let thread_name = "lance_background_thread".to_string();

        let (tx, rx) = std::sync::mpsc::channel::<BoxFuture<'static, ()>>();

        let thread = std::thread::spawn(move || {
            // Create a new Runtime to run tasks
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name(&thread_name)
                .build()
                .expect("Creating Tokio runtime");

            // Pull task requests off the channel and send them to the executor
            runtime.block_on(async move {
                while let Ok(task) = rx.recv() {
                    tokio::task::spawn(async move {
                        task.await;
                    });
                }
            });
        });

        let state = State {
            requests: Some(tx),
            _thread: thread,
        };

        Self {
            state: Arc::new(Mutex::new(state)),
        }
    }

    pub fn block_on<T>(&self, task: T) -> T::Output
    where
        T: Future + Send + 'static,
        T::Output: Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::channel::<T::Output>();

        let fut = Box::pin(async move {
            println!("Running task on background thread");
            let task_output = task.await;
            tokio::task::spawn_blocking(move || {
                tx.send(task_output).ok();
            })
            .await
            .unwrap();
        });

        let mut state = self.state.lock().unwrap();
        println!("Sending task to background thread");
        if let Some(requests) = &mut state.requests {
            requests.send(fut).unwrap();
        } else {
            panic!("Background thread has exited");
        }

        // Drop the lock while we wait for the task to complete
        std::mem::drop(state);

        println!("Waiting for task to complete");

        let out = rx.recv().unwrap();

        println!("Task completed");

        out
    }
}
