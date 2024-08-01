// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for byte arrays

use std::{ops::Deref, ptr::NonNull, sync::Arc};

use arrow_buffer::Buffer;
use snafu::{location, Location};

use lance_core::{Error, Result};

/// A copy-on-write byte buffer
///
/// It can be created from read-only buffers (e.g. bytes::Bytes or arrow_buffer::Buffer), e.g. "borrowed"
/// or from writeable buffers (e.g. Vec<u8>), e.g. "owned"
///
/// The buffer can switch to borrowed mode without a copy of the data
///
/// LanceBuffer does not implement Clone because doing could potentially silently trigger a copy of the data
/// and we want to make sure that the user is aware of this operation.
///
/// If you need to clone a LanceBuffer you can use borrow_and_clone() which will make sure that the buffer
/// is in borrowed mode before cloning.  This is a zero copy operation (but requires &mut self).
#[derive(Debug)]
pub enum LanceBuffer {
    Borrowed(Buffer),
    Owned(Vec<u8>),
}

impl LanceBuffer {
    /// Convert into a mutable buffer.  If this is a borrowed buffer, the data will be copied.
    pub fn into_owned(self) -> Vec<u8> {
        match self {
            Self::Borrowed(buffer) => buffer.to_vec(),
            Self::Owned(buffer) => buffer,
        }
    }

    /// Convert into an Arrow buffer.  Never copies data.
    pub fn into_buffer(self) -> Buffer {
        match self {
            Self::Borrowed(buffer) => buffer,
            Self::Owned(buffer) => Buffer::from_vec(buffer),
        }
    }

    /// Create a LanceBuffer from a bytes::Bytes object
    ///
    /// The alignment must be specified (as `bytes_per_value`) since we want to make
    /// sure we can safely reinterpret the buffer.
    ///
    /// If the buffer is properly aligned this will be zero-copy.  If not, a copy
    /// will be made and an owned buffer returned.
    pub fn from_bytes(bytes: bytes::Bytes, bytes_per_value: u64) -> Self {
        if bytes.as_ptr().align_offset(bytes_per_value as usize) != 0 {
            // The original buffer is not aligned, cannot zero-copy
            let mut buf = Vec::with_capacity(bytes.len());
            buf.extend_from_slice(&bytes);
            Self::Owned(buf)
        } else {
            // The original buffer is aligned, can zero-copy
            // SAFETY: the alignment is correct we can make this conversion
            unsafe {
                Self::Borrowed(Buffer::from_custom_allocation(
                    NonNull::new(bytes.as_ptr() as _).expect("should be a valid pointer"),
                    bytes.len(),
                    Arc::new(bytes),
                ))
            }
        }
    }

    /// Convert into a borrowed buffer, this is a zero-copy operation
    ///
    /// This is often called before cloning the buffer
    pub fn into_borrowed(self) -> Self {
        match self {
            Self::Borrowed(_) => self,
            Self::Owned(buffer) => Self::Borrowed(Buffer::from_vec(buffer)),
        }
    }

    /// Creates an owned copy of the buffer, will always involve a full copy of the bytes
    pub fn to_owned(&self) -> Self {
        match self {
            Self::Borrowed(buffer) => Self::Owned(buffer.to_vec()),
            Self::Owned(buffer) => Self::Owned(buffer.clone()),
        }
    }

    /// Creates a clone of the buffer but also puts the buffer into borrowed mode
    ///
    /// This is a zero-copy operation
    pub fn borrow_and_clone(&mut self) -> Self {
        match self {
            Self::Borrowed(buffer) => Self::Borrowed(buffer.clone()),
            Self::Owned(buffer) => {
                let buf_data = std::mem::take(buffer);
                let buffer = Buffer::from_vec(buf_data);
                *self = Self::Borrowed(buffer.clone());
                Self::Borrowed(buffer)
            }
        }
    }

    /// Clones the buffer but fails if the buffer is in owned mode
    pub fn try_clone(&self) -> Result<Self> {
        match self {
            Self::Borrowed(buffer) => Ok(Self::Borrowed(buffer.clone())),
            Self::Owned(_) => Err(Error::Internal {
                message: "try_clone called on an owned buffer".to_string(),
                location: location!(),
            }),
        }
    }
}

impl AsRef<[u8]> for LanceBuffer {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Borrowed(buffer) => buffer.as_slice(),
            Self::Owned(buffer) => buffer.as_slice(),
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
        Self::Owned(buffer)
    }
}

impl From<Buffer> for LanceBuffer {
    fn from(buffer: Buffer) -> Self {
        Self::Borrowed(buffer)
    }
}
