// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for byte arrays

use std::{ops::Deref, ptr::NonNull, sync::Arc};

use arrow_buffer::{ArrowNativeType, Buffer, ScalarBuffer};
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
pub enum LanceBuffer {
    Borrowed(Buffer),
    Owned(Vec<u8>),
}

// Compares equality of the buffers, ignoring owned / unowned status
impl PartialEq for LanceBuffer {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Borrowed(l0), Self::Borrowed(r0)) => l0 == r0,
            (Self::Owned(l0), Self::Owned(r0)) => l0 == r0,
            (Self::Borrowed(l0), Self::Owned(r0)) => l0.as_slice() == r0.as_slice(),
            (Self::Owned(l0), Self::Borrowed(r0)) => l0.as_slice() == r0.as_slice(),
        }
    }
}

impl Eq for LanceBuffer {}

impl std::fmt::Debug for LanceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let preview = if self.len() > 10 {
            format!("0x{}...", hex::encode_upper(&self[..10]))
        } else {
            format!("0x{}", hex::encode_upper(self.as_ref()))
        };
        match self {
            Self::Borrowed(buffer) => write!(
                f,
                "LanceBuffer::Borrowed(bytes={} #bytes={})",
                preview,
                buffer.len()
            ),
            Self::Owned(buffer) => {
                write!(
                    f,
                    "LanceBuffer::Owned(bytes={} #bytes={})",
                    preview,
                    buffer.len()
                )
            }
        }
    }
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

    /// Returns an owned buffer of the given size with all bits set to 0
    pub fn all_unset(len: usize) -> Self {
        Self::Owned(vec![0; len])
    }

    /// Returns an owned buffer of the given size with all bits set to 1
    pub fn all_set(len: usize) -> Self {
        Self::Owned(vec![0xff; len])
    }

    /// Creates an empty buffer
    pub fn empty() -> Self {
        Self::Owned(Vec::new())
    }

    /// Converts the buffer into a hex string
    pub fn as_hex(&self) -> String {
        hex::encode_upper(self)
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

    /// Make an owned copy of the buffer (always does a copy of the data)
    pub fn deep_copy(&self) -> Self {
        match self {
            Self::Borrowed(buffer) => Self::Owned(buffer.to_vec()),
            Self::Owned(buffer) => Self::Owned(buffer.clone()),
        }
    }

    /// Reinterprets a Vec<T> as a LanceBuffer
    ///
    /// Note that this creates a borrowed buffer.  It is not possible to safely
    /// reinterpret a Vec<T> into a Vec<u8> in rust due to this constraint from
    /// [`Vec::from_raw_parts`]:
    ///
    /// > `T` needs to have the same alignment as what `ptr` was allocated with.
    /// > (`T` having a less strict alignment is not sufficient, the alignment really
    /// > needs to be equal to satisfy the [`dealloc`] requirement that memory must be
    /// > allocated and deallocated with the same layout.)
    ///
    /// However, we can safely reinterpret Vec<T> into &[u8] which is what happens here.
    pub fn reinterpret_vec<T: ArrowNativeType>(vec: Vec<T>) -> Self {
        Self::Borrowed(Buffer::from_vec(vec))
    }

    /// Reinterprets a LanceBuffer into a Vec<T>
    ///
    /// Unfortunately, there is no way to do this safely in Rust without a copy, even if
    /// the source is Vec<u8>.
    pub fn borrow_to_typed_slice<T: ArrowNativeType>(&mut self) -> impl AsRef<[T]> {
        ScalarBuffer::<T>::from(self.borrow_and_clone().into_buffer())
    }

    /// Concatenates multiple buffers into a single buffer, consuming the input buffers
    ///
    /// If there is only one buffer, it will be returned as is
    pub fn concat_into_one(buffers: Vec<Self>) -> Self {
        if buffers.len() == 1 {
            return buffers.into_iter().next().unwrap();
        }

        let mut total_len = 0;
        for buffer in &buffers {
            total_len += buffer.len();
        }

        let mut data = Vec::with_capacity(total_len);
        for buffer in buffers {
            data.extend_from_slice(buffer.as_ref());
        }

        Self::Owned(data)
    }

    /// Zips multiple buffers into a single buffer, consuming the input buffers
    ///
    /// Unlike concat_into_one this "zips" the buffers, interleaving the values
    pub fn zip_into_one(buffers: Vec<(Self, u64)>, num_values: u64) -> Result<Self> {
        let bytes_per_value = buffers.iter().map(|(_, bits_per_value)| {
            if bits_per_value % 8 == 0 {
                Ok(bits_per_value / 8)
            } else {
                Err(Error::InvalidInput { source: format!("LanceBuffer::zip_into_one only supports full-byte buffers currently and received a buffer with {} bits per value", bits_per_value).into(), location: location!() })
            }
        }).collect::<Result<Vec<_>>>()?;
        let total_bytes_per_value = bytes_per_value.iter().sum::<u64>();
        let total_bytes = (total_bytes_per_value * num_values) as usize;

        let mut zipped = vec![0_u8; total_bytes];
        let mut buffer_ptrs = buffers
            .iter()
            .zip(bytes_per_value)
            .map(|((buffer, _), bytes_per_value)| (buffer.as_ptr(), bytes_per_value as usize))
            .collect::<Vec<_>>();

        let mut zipped_ptr = zipped.as_mut_ptr();
        unsafe {
            let end = zipped_ptr.add(total_bytes);
            while zipped_ptr < end {
                for (buf, bytes_per_value) in buffer_ptrs.iter_mut() {
                    std::ptr::copy_nonoverlapping(*buf, zipped_ptr, *bytes_per_value);
                    zipped_ptr = zipped_ptr.add(*bytes_per_value);
                    *buf = buf.add(*bytes_per_value);
                }
            }
        }

        Ok(Self::Owned(zipped))
    }

    /// Create a LanceBuffer from a slice
    ///
    /// This is NOT a zero-copy operation.  We can't even create a borrowed buffer because
    /// we have no way of extending the lifetime of the slice.
    pub fn copy_slice(slice: &[u8]) -> Self {
        Self::Owned(slice.to_vec())
    }

    /// Create a LanceBuffer from an array (fixed-size slice)
    ///
    /// This is NOT a zero-copy operation.  The slice memory could be on the stack and
    /// thus we can't forget it.
    pub fn copy_array<const N: usize>(array: [u8; N]) -> Self {
        Self::Owned(Vec::from(array))
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

// All `From` implementations are zero-copy

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

#[cfg(test)]
mod tests {
    use arrow_buffer::Buffer;

    use super::LanceBuffer;

    #[test]
    fn test_eq() {
        let buf = LanceBuffer::Borrowed(Buffer::from_vec(vec![1_u8, 2, 3]));
        let buf2 = LanceBuffer::Owned(vec![1, 2, 3]);
        assert_eq!(buf, buf2);
    }

    #[test]
    fn test_reinterpret_vec() {
        let vec = vec![1_u32, 2, 3];
        let mut buf = LanceBuffer::reinterpret_vec(vec);

        let mut expected = Vec::with_capacity(12);
        expected.extend_from_slice(&1_u32.to_ne_bytes());
        expected.extend_from_slice(&2_u32.to_ne_bytes());
        expected.extend_from_slice(&3_u32.to_ne_bytes());
        let expected = LanceBuffer::Owned(expected);

        assert_eq!(expected, buf);
        assert_eq!(buf.borrow_to_typed_slice::<u32>().as_ref(), vec![1, 2, 3]);
    }

    #[test]
    fn test_concat() {
        let buf1 = LanceBuffer::Owned(vec![1_u8, 2, 3]);
        let buf2 = LanceBuffer::Owned(vec![4_u8, 5, 6]);
        let buf3 = LanceBuffer::Owned(vec![7_u8, 8, 9]);

        let expected = LanceBuffer::Owned(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            expected,
            LanceBuffer::concat_into_one(vec![buf1, buf2, buf3])
        );

        let empty = LanceBuffer::empty();
        assert_eq!(
            LanceBuffer::empty(),
            LanceBuffer::concat_into_one(vec![empty])
        );

        let expected = LanceBuffer::Owned(vec![1, 2, 3]);
        assert_eq!(
            expected,
            LanceBuffer::concat_into_one(vec![expected.deep_copy(), LanceBuffer::empty()])
        );
    }

    #[test]
    fn test_zip() {
        let buf1 = LanceBuffer::Owned(vec![1_u8, 2, 3]);
        let buf2 = LanceBuffer::reinterpret_vec(vec![1_u16, 2, 3]);
        let buf3 = LanceBuffer::reinterpret_vec(vec![1_u32, 2, 3]);

        let zipped = LanceBuffer::zip_into_one(vec![(buf1, 8), (buf2, 16), (buf3, 32)], 3).unwrap();

        assert_eq!(zipped.len(), 21);

        let mut expected = Vec::with_capacity(21);
        for i in 1..4 {
            expected.push(i as u8);
            expected.extend_from_slice(&(i as u16).to_ne_bytes());
            expected.extend_from_slice(&(i as u32).to_ne_bytes());
        }
        let expected = LanceBuffer::Owned(expected);

        assert_eq!(expected, zipped);
    }

    #[test]
    fn test_hex() {
        let buf = LanceBuffer::Owned(vec![1, 2, 15, 20]);
        assert_eq!("01020F14", buf.as_hex());
    }
}
