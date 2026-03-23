// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Global Tokio runtime for the C FFI layer.

use std::sync::LazyLock;

/// Global multi-threaded Tokio runtime, shared across all FFI calls.
/// Initialized lazily on first access.
pub static RT: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime")
});

/// Block the current thread on an async future using the global runtime.
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    RT.block_on(f)
}
