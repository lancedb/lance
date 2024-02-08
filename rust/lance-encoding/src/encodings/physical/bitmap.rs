use std::ops::Range;

use arrow_array::{cast::AsArray, ArrayRef};
use arrow_buffer::BooleanBufferBuilder;
use arrow_schema::DataType;
use bytes::{Bytes, BytesMut};

use futures::{future::BoxFuture, FutureExt};
use lance_core::Result;

use crate::{
    decoder::{PhysicalPageDecoder, PhysicalPageScheduler},
    encoder::{BufferEncoder, EncodedBuffer},
    EncodingsIo,
};

/// A physical scheduler for bitmap buffers encoded densely as 1 bit per value
/// with bit-endianess (e.g. what Arrow uses for validity bitmaps and boolean arrays)
///
/// This decoder decodes from one buffer of disk data into one buffer of memory data
#[derive(Debug, Clone, Copy)]
struct DenseBitmapScheduler {}

impl PhysicalPageScheduler for DenseBitmapScheduler {
    fn schedule_range(
        &self,
        range: Range<u32>,
        scheduler: &dyn EncodingsIo,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>> {
        debug_assert_ne!(range.start, range.end);
        let start = range.start as u64 / 8;
        let end = (range.end as u64 / 8) + 1;
        let byte_range = start..end;

        let bytes = scheduler.submit_request(vec![byte_range]);

        async move {
            let bytes = bytes.await?;
            Ok(Box::new(BitmapDecoder { data: bytes }) as Box<dyn PhysicalPageDecoder>)
        }
        .boxed()
    }
}

struct BitmapDecoder {
    data: Vec<Bytes>,
}

impl PhysicalPageDecoder for BitmapDecoder {
    fn update_capacity(&self, _rows_to_skip: u32, num_rows: u32, buffers: &mut [(u64, bool)]) {
        buffers[0].0 = arrow_buffer::bit_util::ceil(num_rows as usize, 8) as u64;
        // This decoder has no concept of "optional" buffers
    }

    fn decode_into(&self, rows_to_skip: u32, num_rows: u32, dest_buffers: &mut [BytesMut]) {
        let mut bytes_to_fully_skip = rows_to_skip as u64 / 8;
        let mut bits_to_skip = rows_to_skip % 8;

        let mut dest_builder = BooleanBufferBuilder::new(num_rows as usize);

        let mut rows_remaining = num_rows;
        for buf in &self.data {
            let buf_len = buf.len() as u64;
            if bytes_to_fully_skip > buf_len {
                bytes_to_fully_skip -= buf_len;
            } else {
                let num_vals = (buf_len * 8) as u32;
                let num_vals_to_take = rows_remaining.min(num_vals);
                let end = (num_vals_to_take + bits_to_skip) as usize;
                dest_builder.append_packed_range(bits_to_skip as usize..end, buf);
                bytes_to_fully_skip = 0;
                bits_to_skip = 0;
                rows_remaining -= num_vals_to_take;
            }
        }

        let bool_buffer = dest_builder.finish().into_inner();
        unsafe { dest_buffers[0].set_len(bool_buffer.len()) }
        // TODO: This requires an extra copy.  First we copy the data from the read buffer(s)
        // into dest_builder (one copy is inevitable).  Then we copy the data from dest_builder
        // into dest_buffers.  This second copy could be avoided (e.g. BooleanBufferBuilder
        // has a new_from_buffer but that requires MutableBuffer and we can't easily get there
        // from BytesMut [or can we?])
        //
        // Worst case, we vendor our own copy of BooleanBufferBuilder based on BytesMut.  We could
        // also use MutableBuffer ourselves instead of BytesMut but arrow-rs claims MutableBuffer may
        // be deprecated in the future (though that discussion seems to have died)

        // TODO: Will this work at the boundaries?  If we have to skip 3 bits for example then the first
        // bytes of bool_buffer.as_slice will be 000XXXXX and if we copy it on top of YYY00000 then the YYY
        // will be clobbered.
        dest_buffers[0].copy_from_slice(bool_buffer.as_slice());
    }
}

// Encoder for writing boolean arrays as dense bitmaps
#[derive(Debug, Default)]
pub struct BitmapEncoder {}

impl BufferEncoder for BitmapEncoder {
    fn encode(&self, arrays: &[ArrayRef]) -> Result<EncodedBuffer> {
        debug_assert!(arrays
            .iter()
            .all(|arr| *arr.data_type() == DataType::Boolean));
        let num_rows: u32 = arrays.iter().map(|arr| arr.len() as u32).sum();
        // Empty pages don't make sense, this should be prevented before we
        // get here
        debug_assert_ne!(num_rows, 0);
        // We can't just write the inner value buffers one after the other because
        // bitmaps can have junk padding at the end (e.g. a boolean array with 12
        // values will be 2 bytes but the last four bits of the second byte are
        // garbage).  So we go ahead and pay the cost of a copy (we could avoid this
        // if we really needed to, at the expense of more complicated code and a slightly
        // larger encoded size but writer cost generally doesn't matter all that much)
        let mut builder = BooleanBufferBuilder::new(num_rows as usize);
        for arr in arrays {
            let bool_arr = arr.as_boolean();
            builder.append_buffer(bool_arr.values());
        }
        let buffer = builder.finish().into_inner();
        let parts = vec![buffer];
        let buffer = EncodedBuffer {
            is_data: true,
            parts,
        };
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field};

    use crate::encodings::physical::basic::BasicEncoder;
    use crate::encodings::physical::{basic::BasicPageScheduler, bitmap::DenseBitmapScheduler};
    use crate::testing::check_round_trip_encoding;

    #[tokio::test]
    async fn test_bitmap_boolean() {
        let encoder = BasicEncoder::new(0);
        let decoder = Arc::new(BasicPageScheduler::new_non_nullable(Box::new(
            DenseBitmapScheduler {},
        )));
        let field = Field::new("", DataType::Boolean, false);

        check_round_trip_encoding(&encoder, &[decoder], field).await;
    }
}
