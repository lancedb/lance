// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::array::ArrayData;
use arrow::datatypes::{ArrowPrimitiveType, UInt16Type, UInt32Type, UInt64Type, UInt8Type};
use arrow_array::{cast::AsArray, Array, ArrayRef, PrimitiveArray};
use arrow_schema::DataType;
use bytes::{Bytes, BytesMut};
use futures::future::{BoxFuture, FutureExt};
use log::trace;
use num_traits::{AsPrimitive, PrimInt};
use snafu::{location, Location};

use lance_arrow::DataTypeExt;
use lance_core::{Error, Result};

use crate::encoder::EncodedBufferMeta;
use crate::{
    decoder::{PageScheduler, PrimitivePageDecoder},
    encoder::{BufferEncoder, EncodedBuffer},
};

// Compute the number of bits to use for each item, if this array can be encoded using
// bitpacking encoding. Returns `None` if the type or array data is not supported.
pub fn num_compressed_bits(arr: ArrayRef) -> Option<u64> {
    match arr.data_type() {
        DataType::UInt8 => num_bits_for_type::<UInt8Type>(arr.as_primitive()),
        DataType::UInt16 => num_bits_for_type::<UInt16Type>(arr.as_primitive()),
        DataType::UInt32 => num_bits_for_type::<UInt32Type>(arr.as_primitive()),
        DataType::UInt64 => num_bits_for_type::<UInt64Type>(arr.as_primitive()),
        _ => None,
    }
}

// Compute the number of bits to to use for bitpacking generically.
// Returns `None` if the array is empty or only contains null values.
fn num_bits_for_type<T>(arr: &PrimitiveArray<T>) -> Option<u64>
where
    T: ArrowPrimitiveType,
    T::Native: PrimInt + AsPrimitive<u64>,
{
    let max = arrow::compute::bit_or(arr);
    let num_bits: Option<u64> =
        max.map(|x| arr.data_type().byte_width() as u64 * 8 - x.leading_zeros() as u64);

    // we can't bitpack into 0 bits, so the minimum is 1
    num_bits.map(|x| x.max(1))
}

#[derive(Debug, Default)]
pub struct BitpackingBufferEncoder {}

impl BufferEncoder for BitpackingBufferEncoder {
    fn encode(&self, arrays: &[ArrayRef]) -> Result<(EncodedBuffer, EncodedBufferMeta)> {
        // TODO -- num bits can be a struct field now that we have the strategy
        let mut num_bits = 0;
        for arr in arrays {
            let arr_max = num_compressed_bits(arr.clone()).ok_or(Error::InvalidInput {
                source: format!("Cannot compute num bits for array: {:?}", arr).into(),
                location: location!(),
            })?;
            num_bits = num_bits.max(arr_max);
        }

        // calculate the total number of bytes we need to allocate for the destination.
        // this will be the number of items in the source array times the number of bits.
        let count_items = count_items_to_pack(arrays);
        // TODO: make function for count & round up
        let mut dst_bytes_total = count_items * num_bits as usize / 8;
        // if if there's a partial byte at the end, we need to allocate one more byte
        if (count_items * num_bits as usize) % 8 != 0 {
            dst_bytes_total += 1;
        }

        let mut dst_buffer = vec![0u8; dst_bytes_total];
        let mut dst_idx = 0;
        let mut dst_offset = 0;
        for arr in arrays {
            pack_array(
                arr.clone(),
                num_bits,
                &mut dst_buffer,
                &mut dst_idx,
                &mut dst_offset,
            )?;
            // packed_arrays.push(packed.into());
        }

        let data_type = arrays[0].data_type();
        Ok((
            EncodedBuffer {
                parts: vec![dst_buffer.into()],
                // bits_per_value: (data_type.byte_width() * 8) as u64,
                // bitpacked_bits_per_value: Some(num_bits),
                // compression_scheme: None,
            },
            EncodedBufferMeta {
                bits_per_value: (data_type.byte_width() * 8) as u64,
                bitpacked_bits_per_value: Some(num_bits),
                compression_scheme: None,
            },
        ))
    }
}

fn count_items_to_pack(arrays: &[ArrayRef]) -> usize {
    let mut count = 0;
    for arr in arrays {
        let data = arr.to_data();
        let buffers = data.buffers();
        for buffer in buffers {
            count += buffer.len();
        }
    }

    count
}

fn pack_array(
    arr: ArrayRef,
    num_bits: u64,
    dst: &mut [u8],
    dst_idx: &mut usize,
    dst_offset: &mut u8,
) -> Result<()> {
    match arr.data_type() {
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            pack_buffers(
                arr.to_data(),
                num_bits,
                arr.data_type().byte_width(),
                dst,
                dst_idx,
                dst_offset,
            );

            Ok(())
        }
        _ => Err(Error::InvalidInput {
            source: format!("Invalid data type for bitpacking: {}", arr.data_type()).into(),
            location: location!(),
        }),
    }
}

fn pack_buffers(
    data: ArrayData,
    num_bits: u64,
    byte_len: usize,
    dst: &mut [u8],
    dst_idx: &mut usize,
    dst_offset: &mut u8,
) {
    let buffers = data.buffers();
    for buffer in buffers {
        pack_bits(buffer, num_bits, byte_len, dst, dst_idx, dst_offset);
    }
}

fn pack_bits(
    src: &[u8],
    num_bits: u64,
    byte_len: usize,
    dst: &mut [u8],
    dst_idx: &mut usize,
    dst_offset: &mut u8,
) {
    let bit_len = byte_len as u64 * 8;

    let mut mask = 0u64;
    for _ in 0..num_bits {
        mask = mask << 1 | 1;
    }

    let mut src_idx = 0;
    while src_idx < src.len() {
        let mut curr_mask = mask;
        let mut curr_src = src[src_idx] & curr_mask as u8;
        let mut src_offset = 0;
        let mut src_bits_written = 0;

        while src_bits_written < num_bits {
            dst[*dst_idx] += (curr_src >> src_offset) << *dst_offset as u64;
            let bits_written = (num_bits - src_bits_written)
                .min(8 - src_offset)
                .min(8 - *dst_offset as u64);
            src_bits_written += bits_written;
            *dst_offset += bits_written as u8;
            src_offset += bits_written;

            if *dst_offset == 8 {
                *dst_idx += 1;
                *dst_offset = 0;
            }

            if src_offset == 8 {
                src_idx += 1;
                src_offset = 0;
                curr_mask >>= 8;
                if src_idx == src.len() {
                    break;
                }
                curr_src = src[src_idx] & curr_mask as u8;
            }
        }

        // advance source_offset to the next byte if we're not at the end..
        // note that we don't need to do this if we wrote the full number of bits
        // because source index would have been advanced by the inner loop above
        if bit_len != num_bits {
            let mut partial_bytes_written = num_bits / 8;

            // if we didn't write the full byte for the last byte, increment by one because
            // we wrote a partial byte. Also increment by one if it's a partial byte where
            // the num bits is < 8
            if bit_len % num_bits != 0 || partial_bytes_written == 0 {
                partial_bytes_written += 1;
            }

            // we also want to the next location in src, unless we wrote something
            // byte-aligned in which case the logic above would have already advanced
            let mut to_next_byte = 1;
            if num_bits % 8 == 0 {
                to_next_byte = 0;
            }

            src_idx += (byte_len as u64 - partial_bytes_written + to_next_byte) as usize;
        }
    }
}

// A physical scheduler for bitpacked buffers
#[derive(Debug, Clone, Copy)]
pub struct BitpackedScheduler {
    bits_per_value: u64,
    uncompressed_bits_per_value: u64,
    buffer_offset: u64,
}

impl BitpackedScheduler {
    pub fn new(bits_per_value: u64, uncompressed_bits_per_value: u64, buffer_offset: u64) -> Self {
        Self {
            bits_per_value,
            uncompressed_bits_per_value,
            buffer_offset,
        }
    }
}

impl PageScheduler for BitpackedScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u64>],
        scheduler: &Arc<dyn crate::EncodingsIo>,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>> {
        let mut min = u64::MAX;
        let mut max = 0;

        let mut buffer_bit_start_offsets: Vec<u8> = vec![];
        let mut buffer_bit_end_offsets: Vec<Option<u8>> = vec![];
        let byte_ranges = ranges
            .iter()
            .map(|range| {
                let start_byte_offset = range.start * self.bits_per_value / 8;
                let mut end_byte_offset = range.end * self.bits_per_value / 8;
                if range.end * self.bits_per_value % 8 != 0 {
                    // If the end of the range is not byte-aligned, we need to read one more byte
                    end_byte_offset += 1;

                    let end_bit_offset = range.end * self.bits_per_value % 8;
                    buffer_bit_end_offsets.push(Some(end_bit_offset as u8));
                } else {
                    buffer_bit_end_offsets.push(None);
                }

                let start_bit_offset = range.start * self.bits_per_value % 8;
                buffer_bit_start_offsets.push(start_bit_offset as u8);

                let start = self.buffer_offset + start_byte_offset;
                let end = self.buffer_offset + end_byte_offset;
                min = min.min(start);
                max = max.max(end);

                start..end
            })
            .collect::<Vec<_>>();

        trace!(
            "Scheduling I/O for {} ranges spread across byte range {}..{}",
            byte_ranges.len(),
            min,
            max
        );

        let bytes = scheduler.submit_request(byte_ranges, top_level_row);

        let bits_per_value = self.bits_per_value;
        let uncompressed_bits_per_value = self.uncompressed_bits_per_value;
        async move {
            let bytes = bytes.await?;
            Ok(Box::new(BitpackedPageDecoder {
                buffer_bit_start_offsets,
                buffer_bit_end_offsets,
                bits_per_value,
                uncompressed_bits_per_value,
                data: bytes,
            }) as Box<dyn PrimitivePageDecoder>)
        }
        .boxed()
    }
}

#[derive(Debug)]
struct BitpackedPageDecoder {
    // bit offsets of the first value within each buffer
    buffer_bit_start_offsets: Vec<u8>,

    // bit offsets of the last value within each buffer. e.g. if there was a buffer
    // with 2 values, packed into 5 bits, this would be [Some(3)], indicating that
    // the bits from the 3rd->8th bit in the last byte shouldn't be decoded.
    buffer_bit_end_offsets: Vec<Option<u8>>,

    // the number of bits used to represent a compressed value. E.g. if the max value
    // in the page was 7 (0b111), then this will be 3
    bits_per_value: u64,

    // number of bits in the uncompressed value. E.g. this will be 32 for u32
    uncompressed_bits_per_value: u64,

    data: Vec<Bytes>,
}

impl PrimitivePageDecoder for BitpackedPageDecoder {
    fn decode(
        &self,
        rows_to_skip: u64,
        num_rows: u64,
        _all_null: &mut bool,
        // dest_buffers: &mut [BytesMut],
    ) -> Result<Vec<BytesMut>> {
        let num_bytes = self.uncompressed_bits_per_value / 8 * num_rows;
        let mut dest_buffers = vec![BytesMut::with_capacity(num_bytes as usize)];

        // current maximum supported bits per value = 64
        debug_assert!(self.bits_per_value <= 64);

        let mut rows_to_skip = rows_to_skip;
        let mut rows_taken = 0;
        let byte_len = self.uncompressed_bits_per_value / 8;
        let dst = &mut dest_buffers[0];
        let mut dst_idx = dst.len(); // index for current byte being written to destination buffer

        // create bit mask for source bits
        let mut mask = 0u64;
        for _ in 0..self.bits_per_value {
            mask = mask << 1 | 1;
        }

        for i in 0..self.data.len() {
            let src = &self.data[i];
            let (mut src_idx, mut src_offset) = match compute_start_offset(
                rows_to_skip,
                src.len(),
                self.bits_per_value,
                self.buffer_bit_start_offsets[i],
                self.buffer_bit_end_offsets[i],
            ) {
                StartOffset::SkipFull(rows_to_skip_here) => {
                    rows_to_skip -= rows_to_skip_here;
                    continue;
                }
                StartOffset::SkipSome(buffer_start_offset) => (
                    buffer_start_offset.index,
                    buffer_start_offset.bit_offset as u64,
                ),
            };

            while src_idx < src.len() && rows_taken < num_rows {
                rows_taken += 1;
                let mut curr_mask = mask; // copy mask

                // current source byte being written to destination
                let mut curr_src = src[src_idx] & (curr_mask << src_offset) as u8;

                // how many bits from the current source value have been written to destination
                let mut src_bits_written = 0;

                // the offset within the current destination byte to write to
                let mut dst_offset = 0;

                while src_bits_written < self.bits_per_value {
                    // add extra byte to buffer to hold next location
                    dst.extend([0].repeat(dst_idx + 1 - dst.len()));

                    // write bits from current source byte into destination
                    dst[dst_idx] += (curr_src >> src_offset) << dst_offset;
                    let bits_written = (self.bits_per_value - src_bits_written)
                        .min(8 - src_offset)
                        .min(8 - dst_offset);
                    src_bits_written += bits_written;
                    dst_offset += bits_written;
                    src_offset += bits_written;
                    curr_mask >>= bits_written;

                    if dst_offset == 8 {
                        dst_idx += 1;
                        dst_offset = 0;
                    }

                    if src_offset == 8 {
                        src_idx += 1;
                        src_offset = 0;
                        if src_idx == src.len() {
                            break;
                        }
                        curr_src = src[src_idx] & curr_mask as u8;
                    }
                }

                // advance destination offset to the next location
                // note that we don't need to do this if we wrote the full number of bits
                // because source index would have been advanced by the inner loop above
                if self.uncompressed_bits_per_value != self.bits_per_value {
                    let mut partial_bytes_written = self.bits_per_value / 8;

                    // if we didn't write the full byte for the last byte, increment by one because
                    // we wrote a partial byte. Also increment by one if it's a partial byte written where
                    // num bits is less than 8
                    if self.uncompressed_bits_per_value % self.bits_per_value != 0
                        || partial_bytes_written == 0
                    {
                        partial_bytes_written += 1;
                    }

                    // we also want to move one location to the next location in destination,
                    // unless we wrote something byte-aligned in which case the logic above
                    // would have already advanced dst_idx
                    let mut to_next_byte = 1;
                    if self.bits_per_value % 8 == 0 {
                        to_next_byte = 0;
                    }
                    dst_idx += (byte_len - partial_bytes_written + to_next_byte) as usize;
                }

                // If we've reached the last byte, there may be some extra bits from the
                // next value outside the range. We don't want to be taking those.
                if let Some(buffer_bit_end_offset) = self.buffer_bit_end_offsets[i] {
                    if src_idx == src.len() - 1 && src_offset >= buffer_bit_end_offset as u64 {
                        break;
                    }
                }
            }
        }

        // add pad any extra needed 0s onto end of buffer
        dst.extend([0].repeat(dst_idx + 1 - dst.len()));

        Ok(dest_buffers)
    }

    fn num_buffers(&self) -> u32 {
        1
    }
}

#[derive(Debug, PartialEq)]
struct BufferStartOffset {
    index: usize,
    bit_offset: u8,
}

#[derive(Debug, PartialEq)]
enum StartOffset {
    // skip the full buffer. The value is how many rows are skipped
    // by skipping the full buffer (e.g., # rows in buffer)
    SkipFull(u64),

    // skip to some start offset in the buffer
    SkipSome(BufferStartOffset),
}

/// compute how far ahead in this buffer should we skip ahead and start reading
///
/// * `rows_to_skip` - how many rows to skip
/// * `buffer_len` - length buf buffer (in bytes)
/// * `bits_per_value` - number of bits used to represent a single bitpacked value
/// * `buffer_start_bit_offset` - offset of the start of the first value within the
///     buffer's  first byte
/// * `buffer_end_bit_offset` - end bit of the last value within the buffer. Can be
///     `None` if the end of the last value is byte aligned with end of buffer.
fn compute_start_offset(
    rows_to_skip: u64,
    buffer_len: usize,
    bits_per_value: u64,
    buffer_start_bit_offset: u8,
    buffer_end_bit_offset: Option<u8>,
) -> StartOffset {
    let rows_in_buffer = rows_in_buffer(
        buffer_len,
        bits_per_value,
        buffer_start_bit_offset,
        buffer_end_bit_offset,
    );
    if rows_to_skip >= rows_in_buffer {
        return StartOffset::SkipFull(rows_in_buffer);
    }

    let start_bit = rows_to_skip * bits_per_value + buffer_start_bit_offset as u64;
    let start_byte = start_bit / 8;

    StartOffset::SkipSome(BufferStartOffset {
        index: start_byte as usize,
        bit_offset: (start_bit % 8) as u8,
    })
}

/// calculates the number of rows in a buffer
fn rows_in_buffer(
    buffer_len: usize,
    bits_per_value: u64,
    buffer_start_bit_offset: u8,
    buffer_end_bit_offset: Option<u8>,
) -> u64 {
    let mut bits_in_buffer = (buffer_len * 8) as u64 - buffer_start_bit_offset as u64;

    // if the end of the last value of the buffer isn't byte aligned, subtract the
    // end offset from the total number of bits in buffer
    if let Some(buffer_end_bit_offset) = buffer_end_bit_offset {
        bits_in_buffer -= (8 - buffer_end_bit_offset) as u64;
    }

    bits_in_buffer / bits_per_value
}

#[cfg(test)]
pub mod test {
    use super::*;
    use std::{io::Read, sync::Arc};

    use arrow_array::{
        types::{UInt16Type, UInt8Type},
        Float64Array, UInt64Array,
    };

    use lance_datagen::{array::fill, gen, ArrayGenerator, ArrayGeneratorExt, RowCount};

    #[test]
    fn test_round_trip() {
        let arrays = vec![Arc::new(UInt64Array::from_iter_values(vec![
            1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
        ])) as ArrayRef];
        let encoder = BitpackingBufferEncoder::default();
        let (result, _) = encoder.encode(&arrays).unwrap();

        let parts = result.parts.clone();
        let part_0 = parts[0].clone();
        let byte_slice = part_0.bytes();
        let bytes_raw: Vec<u8> = byte_slice.into_iter().map(|e| e.unwrap()).collect();

        let bytes = Bytes::copy_from_slice(&bytes_raw);

        let decoder = BitpackedPageDecoder {
            buffer_bit_start_offsets: vec![0],
            buffer_bit_end_offsets: vec![None],
            bits_per_value: 2,
            uncompressed_bits_per_value: 64,
            data: vec![bytes],
        };

        // let dest = BytesMut::new();
        // let mut dests = vec![dest];
        let mut all_nulls = false;
        let dests = decoder.decode(0, 8, &mut all_nulls).unwrap();

        println!("{:?}", dests[0])
    }

    #[test]
    fn test_num_compressed_bits() {
        fn gen_array(generator: Box<dyn ArrayGenerator>) -> ArrayRef {
            let arr = gen()
                .anon_col(generator)
                .into_batch_rows(RowCount::from(10000))
                .unwrap()
                .column(0)
                .clone();
            arr
        }

        macro_rules! do_test {
            ($num_bits:expr, $data_type:ident, $null_probability:expr) => {
                let max = 1 << $num_bits - 1;
                let arr = gen_array(fill::<$data_type>(max).with_random_nulls($null_probability));
                let result = num_compressed_bits(arr);
                assert_eq!(Some($num_bits), result);
            };
        }

        let test_cases = vec![
            (5u64, 0.0f64),
            (5u64, 0.9f64),
            (1u64, 0.0f64),
            (1u64, 0.5f64),
            (8u64, 0.0f64),
            (8u64, 0.5f64),
        ];

        for (num_bits, null_probability) in &test_cases {
            do_test!(*num_bits, UInt8Type, *null_probability);
            do_test!(*num_bits, UInt16Type, *null_probability);
            do_test!(*num_bits, UInt32Type, *null_probability);
            do_test!(*num_bits, UInt64Type, *null_probability);
        }

        // do some test cases that that will only work on larger types
        let test_cases = vec![
            (13u64, 0.0f64),
            (13u64, 0.5f64),
            (16u64, 0.0f64),
            (16u64, 0.5f64),
        ];
        for (num_bits, null_probability) in &test_cases {
            do_test!(*num_bits, UInt16Type, *null_probability);
            do_test!(*num_bits, UInt32Type, *null_probability);
            do_test!(*num_bits, UInt64Type, *null_probability);
        }
        let test_cases = vec![
            (25u64, 0.0f64),
            (25u64, 0.5f64),
            (32u64, 0.0f64),
            (32u64, 0.5f64),
        ];
        for (num_bits, null_probability) in &test_cases {
            do_test!(*num_bits, UInt32Type, *null_probability);
            do_test!(*num_bits, UInt64Type, *null_probability);
        }
        let test_cases = vec![
            (48u64, 0.0f64),
            (48u64, 0.5f64),
            (64u64, 0.0f64),
            (64u64, 0.5f64),
        ];
        for (num_bits, null_probability) in &test_cases {
            do_test!(*num_bits, UInt64Type, *null_probability);
        }

        // test that it returns None for datatypes that don't support bitpacking
        let arr = Float64Array::from_iter_values(vec![0.1, 0.2, 0.3]);
        let result = num_compressed_bits(Arc::new(arr));
        assert_eq!(None, result);

        // test that it returns None for the all null case
        let nulls = vec![true, true];
        let arr = gen_array(fill::<UInt16Type>(2).with_nulls(&nulls));
        let result = num_compressed_bits(Arc::new(arr));
        assert_eq!(None, result);
    }

    #[test]
    fn test_rows_in_buffer() {
        let test_cases = vec![
            (5usize, 5u64, 0u8, None, 8u64),
            (2, 3, 0, Some(5), 4),
            (2, 3, 7, Some(6), 2),
        ];

        for (
            buffer_len,
            bits_per_value,
            buffer_start_bit_offset,
            buffer_end_bit_offset,
            expected,
        ) in test_cases
        {
            let result = rows_in_buffer(
                buffer_len,
                bits_per_value,
                buffer_start_bit_offset,
                buffer_end_bit_offset,
            );
            assert_eq!(expected, result);
        }
    }

    #[test]
    fn test_compute_start_offset() {
        let result = compute_start_offset(0, 5, 5, 0, None);
        assert_eq!(
            StartOffset::SkipSome(BufferStartOffset {
                index: 0,
                bit_offset: 0
            }),
            result
        );

        let result = compute_start_offset(10, 5, 5, 0, None);
        assert_eq!(StartOffset::SkipFull(8), result);
    }
}
