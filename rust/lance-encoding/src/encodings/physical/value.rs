use arrow_array::ArrayRef;
use bytes::Bytes;
use futures::{future::BoxFuture, FutureExt};

use crate::{
    decoder::{PhysicalPageDecoder, PhysicalPageScheduler},
    encoder::{BufferEncoder, EncodedBuffer},
    EncodingsIo,
};

use lance_core::Result;

/// Scheduler for a simple encoding where buffers of fixed-size items are stored as-is on disk
#[derive(Debug, Clone, Copy)]
pub struct ValuePageScheduler {
    // TODO: do we really support values greater than 2^32 bytes per value?
    // I think we want to, in theory, but will need to test this case.
    bytes_per_value: u64,
    buffer_offset: u64,
}

impl ValuePageScheduler {
    /// Create a new instance
    ///
    /// # Arguments
    ///
    /// * `bytes_per_value` - the size, in bytes, of each item
    pub fn new(bytes_per_value: u64, buffer_offset: u64) -> Self {
        Self {
            bytes_per_value,
            buffer_offset,
        }
    }
}

impl PhysicalPageScheduler for ValuePageScheduler {
    fn schedule_range(
        &self,
        range: std::ops::Range<u32>,
        scheduler: &dyn EncodingsIo,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>> {
        let start = self.buffer_offset + (range.start as u64 * self.bytes_per_value);
        let end = self.buffer_offset + (range.end as u64 * self.bytes_per_value);
        let byte_range = start..end;

        let bytes = scheduler.submit_request(vec![byte_range]);
        let bytes_per_value = self.bytes_per_value;

        async move {
            let bytes = bytes.await?;
            Ok(Box::new(ValuePageDecoder {
                bytes_per_value,
                data: bytes,
            }) as Box<dyn PhysicalPageDecoder>)
        }
        .boxed()
    }
}

struct ValuePageDecoder {
    bytes_per_value: u64,
    data: Vec<Bytes>,
}

impl PhysicalPageDecoder for ValuePageDecoder {
    fn update_capacity(&self, _rows_to_skip: u32, num_rows: u32, buffers: &mut [(u64, bool)]) {
        buffers[0].0 = self.bytes_per_value * num_rows as u64;
        buffers[0].1 = true;
    }

    fn decode_into(&self, rows_to_skip: u32, num_rows: u32, dest_buffers: &mut [bytes::BytesMut]) {
        let mut bytes_to_skip = rows_to_skip as u64 * self.bytes_per_value;
        let mut bytes_to_take = num_rows as u64 * self.bytes_per_value;

        let dest = &mut dest_buffers[0];

        debug_assert!(dest.capacity() as u64 >= bytes_to_take);

        for buf in &self.data {
            let buf_len = buf.len() as u64;
            if bytes_to_skip > buf_len {
                bytes_to_skip -= buf_len;
            } else {
                let bytes_to_take_here = buf_len.min(bytes_to_take);
                bytes_to_take -= bytes_to_take_here;
                let start = bytes_to_skip as usize;
                let end = start + bytes_to_take_here as usize;
                dest.extend_from_slice(&buf.slice(start..end));
                bytes_to_skip = 0;
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct ValueEncoder {}

impl BufferEncoder for ValueEncoder {
    fn encode(&self, arrays: &[ArrayRef]) -> Result<EncodedBuffer> {
        let parts = arrays
            .iter()
            .map(|arr| arr.to_data().buffers()[0].clone())
            .collect::<Vec<_>>();
        Ok(EncodedBuffer {
            is_data: true,
            parts,
        })
    }
}
