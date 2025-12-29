// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! io_uring-based I/O for high-performance local file access.
//!
//! This module provides a [`UringReader`] that implements the [`Reader`](crate::traits::Reader) trait
//! using Linux's io_uring interface for asynchronous I/O. It uses a dedicated background thread
//! that owns an io_uring instance and processes read requests from a channel.
//!
//! # Architecture
//!
//! - A dedicated background thread owns a local io_uring instance
//! - Readers submit requests via an MPSC channel to the thread
//! - The thread submits requests to io_uring and processes completions
//! - Futures are woken by the thread when operations complete
//! - Proper async integration using wakers (no busy-looping)
//!
//! # Configuration
//!
//! The io_uring reader is enabled by using the `file+uring://` URI scheme instead of `file://`.
//! Additional tuning parameters are controlled by environment variables:
//!
//! - `LANCE_URING_BLOCK_SIZE` - Block size in bytes (default: 64KB)
//! - `LANCE_URING_IO_PARALLELISM` - Max concurrent operations (default: 32)
//! - `LANCE_URING_QUEUE_DEPTH` - io_uring queue depth (default: 16K)
//! - `LANCE_URING_CORE` - Pin io_uring thread to specific CPU core (optional)
//! - `LANCE_URING_POLL_TIMEOUT_MS` - Thread poll timeout in milliseconds (default: 10)
//!
//! # Platform Support
//!
//! This module is only available on Linux and requires kernel 5.1 or newer.
//! On other platforms, the code falls back to [`LocalObjectReader`](crate::local::LocalObjectReader).
//!
//! # Example
//!
//! ```no_run
//! # use lance_io::object_store::ObjectStore;
//! # async fn example() -> lance_core::Result<()> {
//! // Enable io_uring by using the file+uring:// scheme
//! let uri = "file+uring:///path/to/file.dat";
//! let (store, path) = ObjectStore::from_uri(uri).await?;
//! let reader = store.open(&path).await?;
//!
//! // Reader will use io_uring
//! let data = reader.get_range(0..1024).await?;
//! # Ok(())
//! # }
//! ```

mod future;
mod reader;
mod requests;
mod thread;

// Thread-local io_uring implementation for current-thread runtimes
pub(crate) mod current_thread;
pub(crate) mod current_thread_future;

#[cfg(test)]
mod tests;

pub(crate) use current_thread::UringCurrentThreadReader;
pub use reader::UringReader;

/// Default block size for io_uring reads (64KB)
pub const DEFAULT_URING_BLOCK_SIZE: usize = 64 * 1024;

/// Default I/O parallelism for io_uring (32 concurrent operations)
pub const DEFAULT_URING_IO_PARALLELISM: usize = 32;

/// Default io_uring queue depth (16K entries)
pub const DEFAULT_URING_QUEUE_DEPTH: usize = 16 * 1024;
