// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{ops::Deref, ptr::NonNull, sync::Arc};

use arrow_buffer::Buffer;

// A copy-on-write version of Buffer / MutableBuffer or Bytes / BytesMut
//
// It can be created from read-only buffers (e.g. bytes::Bytes or arrow-rs' Buffer), e.g. "borrowed"
// or from writeable buffers (e.g. Vec<u8>, arrow-rs' MutableBuffer, or bytes::BytesMut), e.g. "owned"
#[derive(Debug)]
pub enum LanceBuffer {
    Borrowed(Buffer),
    Owned(Vec<u8>),
}

impl LanceBuffer {
    // Convert into a mutable buffer.  If this is a borrowed buffer, the data will be copied.
    pub fn into_owned(self) -> Vec<u8> {
        match self {
            LanceBuffer::Borrowed(buffer) => buffer.to_vec(),
            LanceBuffer::Owned(buffer) => buffer,
        }
    }

    // Convert into an Arrow buffer.  Never copies data.
    pub fn into_buffer(self) -> Buffer {
        match self {
            LanceBuffer::Borrowed(buffer) => buffer,
            LanceBuffer::Owned(buffer) => Buffer::from_vec(buffer),
        }
    }

    pub fn from_bytes(bytes: bytes::Bytes, bytes_per_value: u64) -> LanceBuffer {
        if bytes.as_ptr().align_offset(bytes_per_value as usize) != 0 {
            // The original buffer is not aligned, cannot zero-copy
            let mut buf = Vec::with_capacity(bytes.len());
            buf.extend_from_slice(&bytes);
            LanceBuffer::Owned(buf)
        } else {
            // The original buffer is aligned, can zero-copy
            // SAFETY: the alignment is correct we can make this conversion
            unsafe {
                LanceBuffer::Borrowed(Buffer::from_custom_allocation(
                    NonNull::new(bytes.as_ptr() as _).expect("should be a valid pointer"),
                    bytes.len(),
                    Arc::new(bytes),
                ))
            }
        }
    }
}

impl AsRef<[u8]> for LanceBuffer {
    fn as_ref(&self) -> &[u8] {
        match self {
            LanceBuffer::Borrowed(buffer) => buffer.as_slice(),
            LanceBuffer::Owned(buffer) => buffer.as_slice(),
        }
    }
}

impl Deref for LanceBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl From<Vec<u8>> for LanceBuffer {
    fn from(buffer: Vec<u8>) -> Self {
        LanceBuffer::Owned(buffer)
    }
}

impl From<Buffer> for LanceBuffer {
    fn from(buffer: Buffer) -> Self {
        LanceBuffer::Borrowed(buffer)
    }
}
