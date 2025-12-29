// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Protocol types for communication between UringReader and the io_uring thread.

use bytes::BytesMut;
use std::io;
use std::os::unix::io::RawFd;
use std::sync::Mutex;
use std::task::Waker;
use std::thread::ThreadId;

pub(super) struct RequestState {
    pub completed: bool,
    pub waker: Option<Waker>,
    pub err: Option<io::Error>,
    pub buffer: BytesMut,
}

/// I/O request object that contains all state for a single read operation.
/// This is shared between the submitter, uring thread, and future via Arc.
pub(super) struct IoRequest {
    /// File descriptor to read from.
    pub fd: RawFd,

    /// Byte offset to start reading from.
    pub offset: u64,

    /// Number of bytes to read.
    pub length: usize,

    pub thread_id: ThreadId,

    /// Completion flag - set to true when operation completes.
    pub state: Mutex<RequestState>,
}
