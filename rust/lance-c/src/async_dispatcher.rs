// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Async callback dispatcher for non-blocking scan operations.
//!
//! Inspired by the Java JNI dispatcher (PR #6102). A dedicated background thread
//! receives completion messages from Tokio tasks and invokes C callbacks
//! sequentially, avoiding reentrancy and Tokio thread blocking.

use std::ffi::c_void;
use std::sync::{mpsc, LazyLock};

/// C callback function pointer type for async operations.
/// - `ctx`: opaque pointer passed back to the caller
/// - `status`: 0 = success, -1 = error (check `lance_last_error_*`)
/// - `result`: operation-specific result pointer (e.g., `*mut ArrowArrayStream`)
pub type LanceCallback = unsafe extern "C" fn(ctx: *mut c_void, status: i32, result: *mut c_void);

// Safety: LanceCallback is a C function pointer (Send by definition for FFI).
// The ctx pointer is transferred to the dispatcher thread which calls the callback.
unsafe impl Send for DispatcherMessage {}

pub(crate) struct DispatcherMessage {
    pub callback: LanceCallback,
    pub callback_ctx: *mut c_void,
    pub status: i32,
    pub result: *mut c_void,
}

struct Dispatcher {
    tx: mpsc::Sender<DispatcherMessage>,
}

impl Dispatcher {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel::<DispatcherMessage>();

        std::thread::Builder::new()
            .name("lance-c-dispatcher".to_string())
            .spawn(move || {
                log::debug!("Lance C dispatcher thread started");
                while let Ok(msg) = rx.recv() {
                    // Invoke the C callback on this dedicated thread.
                    // This ensures callbacks are serialized and don't run on Tokio I/O threads.
                    unsafe {
                        (msg.callback)(msg.callback_ctx, msg.status, msg.result);
                    }
                }
                log::debug!("Lance C dispatcher thread shutting down");
            })
            .expect("Failed to spawn lance-c dispatcher thread");

        Self { tx }
    }

    fn send(&self, msg: DispatcherMessage) {
        let _ = self.tx.send(msg);
    }
}

static DISPATCHER: LazyLock<Dispatcher> = LazyLock::new(Dispatcher::new);

/// Send a completion message to the dispatcher thread, which will invoke the callback.
pub(crate) fn dispatch_callback(
    callback: LanceCallback,
    callback_ctx: *mut c_void,
    status: i32,
    result: *mut c_void,
) {
    DISPATCHER.send(DispatcherMessage {
        callback,
        callback_ctx,
        status,
        result,
    });
}
