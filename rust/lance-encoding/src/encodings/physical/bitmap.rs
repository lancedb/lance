// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

use arrow_buffer::BooleanBufferBuilder;
use bytes::{Bytes, BytesMut};

use futures::{future::BoxFuture, FutureExt};
use lance_core::Result;
use log::trace;

use crate::{
    decoder::{PhysicalPageDecoder, PhysicalPageScheduler},
    EncodingsIo,
};

/// A physical scheduler for bitmap buffers encoded densely as 1 bit per value
/// with bit-endianess (e.g. what Arrow uses for validity bitmaps and boolean arrays)
///
/// This decoder decodes from one buffer of disk data into one buffer of memory data
#[derive(Debug, Clone, Copy)]
pub struct DenseBitmapScheduler {
    buffer_offset: u64,
}

impl DenseBitmapScheduler {
    pub fn new(buffer_offset: u64) -> Self {
        Self { buffer_offset }
    }
}

impl PhysicalPageScheduler for DenseBitmapScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[Range<u32>],
        scheduler: &dyn EncodingsIo,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>> {
        let mut min = u64::MAX;
        let mut max = 0;
        let chunk_reqs = ranges
            .iter()
            .map(|range| {
                debug_assert_ne!(range.start, range.end);
                let start = self.buffer_offset + range.start as u64 / 8;
                let bit_offset = range.start % 8;
                let end = self.buffer_offset + range.end.div_ceil(8) as u64;
                let byte_range = start..end;
                min = min.min(start);
                max = max.max(end);
                (byte_range, bit_offset, range.end - range.start)
            })
            .collect::<Vec<_>>();

        let byte_ranges = chunk_reqs
            .iter()
            .map(|(range, _, _)| range.clone())
            .collect::<Vec<_>>();
        trace!(
            "Scheduling I/O for {} ranges across byte range {}..{}",
            byte_ranges.len(),
            min,
            max
        );
        let bytes = scheduler.submit_request(byte_ranges, top_level_row);

        async move {
            let bytes = bytes.await?;
            let chunks = bytes
                .into_iter()
                .zip(chunk_reqs)
                .map(|(bytes, (_, bit_offset, length))| BitmapData {
                    data: bytes,
                    bit_offset,
                    length,
                })
                .collect::<Vec<_>>();
            Ok(Box::new(BitmapDecoder { chunks }) as Box<dyn PhysicalPageDecoder>)
        }
        .boxed()
    }
}

struct BitmapData {
    data: Bytes,
    bit_offset: u32,
    length: u32,
}

struct BitmapDecoder {
    chunks: Vec<BitmapData>,
}

impl PhysicalPageDecoder for BitmapDecoder {
    fn update_capacity(
        &self,
        _rows_to_skip: u32,
        num_rows: u32,
        buffers: &mut [(u64, bool)],
        _all_null: &mut bool,
    ) {
        buffers[0].0 = arrow_buffer::bit_util::ceil(num_rows as usize, 8) as u64;
        buffers[0].1 = true;
    }

    fn decode_into(
        &self,
        rows_to_skip: u32,
        num_rows: u32,
        dest_buffers: &mut [BytesMut],
    ) -> Result<()> {
        let mut rows_to_skip = rows_to_skip;

        let mut dest_builder = BooleanBufferBuilder::new(num_rows as usize);

        let mut rows_remaining = num_rows;
        for chunk in &self.chunks {
            if chunk.length <= rows_to_skip {
                rows_to_skip -= chunk.length;
            } else {
                let start = rows_to_skip + chunk.bit_offset;
                let num_vals_to_take = rows_remaining.min(chunk.length);
                let end = start + num_vals_to_take;
                dest_builder.append_packed_range(start as usize..end as usize, &chunk.data);
                rows_to_skip = 0;
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
        //
        // It's a moot point at the moment since we don't support page bridging
        dest_buffers[0].copy_from_slice(bool_buffer.as_slice());
        Ok(())
    }

    fn num_buffers(&self) -> u32 {
        1
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::{DataType, Field};
    use bytes::{Bytes, BytesMut};

    use crate::decoder::PhysicalPageDecoder;
    use crate::encodings::physical::bitmap::BitmapData;
    use crate::testing::check_round_trip_encoding_random;

    use super::BitmapDecoder;

    #[test_log::test(tokio::test)]
    async fn test_bitmap_boolean() {
        let field = Field::new("", DataType::Boolean, false);
        check_round_trip_encoding_random(field).await;
    }

    #[test]
    fn test_bitmap_decoder_edge_cases() {
        // Regression for a case where the row skip and the bit offset
        // require us to read from the second Bytes instead of the first
        let decoder = BitmapDecoder {
            chunks: vec![
                BitmapData {
                    data: Bytes::from_static(&[0b11111111]),
                    bit_offset: 4,
                    length: 4,
                },
                BitmapData {
                    data: Bytes::from_static(&[0b00000000]),
                    bit_offset: 4,
                    length: 4,
                },
            ],
        };
        let mut dest = vec![BytesMut::with_capacity(1)];
        let result = decoder.decode_into(5, 1, &mut dest);
        assert!(result.is_ok());
    }
}
