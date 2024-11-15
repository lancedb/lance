// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::datatypes::{
    Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};
use arrow_array::{Array, PrimitiveArray};
use arrow_schema::DataType;
use byteorder::{ByteOrder, LittleEndian};
use bytes::Bytes;
use futures::future::{BoxFuture, FutureExt};
use log::trace;
use snafu::{location, Location};

use lance_arrow::DataTypeExt;
use lance_core::{Error, Result};

use crate::buffer::LanceBuffer;
use crate::compression_algo::fastlanes::BitPacking;
use crate::data::BlockInfo;
use crate::data::{DataBlock, FixedWidthDataBlock, NullableDataBlock};
use crate::decoder::{MiniBlockDecompressor, PageScheduler, PrimitivePageDecoder};
use crate::encoder::{
    ArrayEncoder, EncodedArray, MiniBlockChunk, MiniBlockCompressed, MiniBlockCompressor,
};
use crate::format::{pb, ProtobufUtils};
use crate::statistics::{GetStat, Stat};
use arrow::array::ArrayRef;
use bytemuck::cast_slice;
const LOG_ELEMS_PER_CHUNK: u8 = 10;
const ELEMS_PER_CHUNK: u64 = 1 << LOG_ELEMS_PER_CHUNK;

// Compute the compressed_bit_width for a given array of integers
// todo: compute all statistics before encoding
// todo: see how to use rust macro to rewrite this function
pub fn compute_compressed_bit_width_for_non_neg(arrays: &[ArrayRef]) -> u64 {
    debug_assert!(!arrays.is_empty());

    let res;

    match arrays[0].data_type() {
        DataType::UInt8 => {
            let mut global_max: u8 = 0;
            for array in arrays {
                let primitive_array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<UInt8Type>>()
                    .unwrap();
                let array_max = arrow::compute::bit_or(primitive_array);
                global_max = global_max.max(array_max.unwrap_or(0));
            }
            let num_bits =
                arrays[0].data_type().byte_width() as u64 * 8 - global_max.leading_zeros() as u64;
            // we will have constant encoding later
            if num_bits == 0 {
                res = 1;
            } else {
                res = num_bits;
            }
        }

        DataType::Int8 => {
            let mut global_max_width: u64 = 0;
            for array in arrays {
                let primitive_array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Int8Type>>()
                    .unwrap();
                let array_max_width = arrow::compute::bit_or(primitive_array).unwrap_or(0);
                global_max_width = global_max_width.max(8 - array_max_width.leading_zeros() as u64);
            }
            if global_max_width == 0 {
                res = 1;
            } else {
                res = global_max_width;
            }
        }

        DataType::UInt16 => {
            let mut global_max: u16 = 0;
            for array in arrays {
                let primitive_array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<UInt16Type>>()
                    .unwrap();
                let array_max = arrow::compute::bit_or(primitive_array).unwrap_or(0);
                global_max = global_max.max(array_max);
            }
            let num_bits =
                arrays[0].data_type().byte_width() as u64 * 8 - global_max.leading_zeros() as u64;
            if num_bits == 0 {
                res = 1;
            } else {
                res = num_bits;
            }
        }

        DataType::Int16 => {
            let mut global_max_width: u64 = 0;
            for array in arrays {
                let primitive_array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Int16Type>>()
                    .unwrap();
                let array_max_width = arrow::compute::bit_or(primitive_array).unwrap_or(0);
                global_max_width =
                    global_max_width.max(16 - array_max_width.leading_zeros() as u64);
            }
            if global_max_width == 0 {
                res = 1;
            } else {
                res = global_max_width;
            }
        }

        DataType::UInt32 => {
            let mut global_max: u32 = 0;
            for array in arrays {
                let primitive_array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<UInt32Type>>()
                    .unwrap();
                let array_max = arrow::compute::bit_or(primitive_array).unwrap_or(0);
                global_max = global_max.max(array_max);
            }
            let num_bits =
                arrays[0].data_type().byte_width() as u64 * 8 - global_max.leading_zeros() as u64;
            if num_bits == 0 {
                res = 1;
            } else {
                res = num_bits;
            }
        }

        DataType::Int32 => {
            let mut global_max_width: u64 = 0;
            for array in arrays {
                let primitive_array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Int32Type>>()
                    .unwrap();
                let array_max_width = arrow::compute::bit_or(primitive_array).unwrap_or(0);
                global_max_width =
                    global_max_width.max(32 - array_max_width.leading_zeros() as u64);
            }
            if global_max_width == 0 {
                res = 1;
            } else {
                res = global_max_width;
            }
        }

        DataType::UInt64 => {
            let mut global_max: u64 = 0;
            for array in arrays {
                let primitive_array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<UInt64Type>>()
                    .unwrap();
                let array_max = arrow::compute::bit_or(primitive_array).unwrap_or(0);
                global_max = global_max.max(array_max);
            }
            let num_bits =
                arrays[0].data_type().byte_width() as u64 * 8 - global_max.leading_zeros() as u64;
            if num_bits == 0 {
                res = 1;
            } else {
                res = num_bits;
            }
        }

        DataType::Int64 => {
            let mut global_max_width: u64 = 0;
            for array in arrays {
                let primitive_array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<Int64Type>>()
                    .unwrap();
                let array_max_width = arrow::compute::bit_or(primitive_array).unwrap_or(0);
                global_max_width =
                    global_max_width.max(64 - array_max_width.leading_zeros() as u64);
            }
            if global_max_width == 0 {
                res = 1;
            } else {
                res = global_max_width;
            }
        }
        _ => {
            panic!("BitpackedForNonNegArrayEncoder only supports data types of UInt8, Int8, UInt16, Int16, UInt32, Int32, UInt64, Int64");
        }
    };
    res
}

// Bitpack integers using fastlanes algorithm, the input is sliced into chunks of 1024 integers, and bitpacked
// chunk by chunk. when the input is not a multiple of 1024, the last chunk is padded with zeros, this is fine because
// we also know the number of rows we have.
// Here self is a borrow of BitpackedForNonNegArrayEncoder, unpacked is a mutable borrow of FixedWidthDataBlock,
// data_type can be  one of u8, u16, u32, or u64.
// buffer_index is a mutable borrow of u32, indicating the buffer index of the output EncodedArray.
// It outputs an fastlanes bitpacked EncodedArray
macro_rules! encode_fixed_width {
    ($self:expr, $unpacked:expr, $data_type:ty, $buffer_index:expr) => {{
        let num_chunks = ($unpacked.num_values + ELEMS_PER_CHUNK - 1) / ELEMS_PER_CHUNK;
        let num_full_chunks = $unpacked.num_values / ELEMS_PER_CHUNK;
        let uncompressed_bit_width = std::mem::size_of::<$data_type>() as u64 * 8;

        // the output vector type is the same as the input type, for example, when input is u16, output is Vec<u16>
        let packed_chunk_size = 1024 * $self.compressed_bit_width as usize / uncompressed_bit_width as usize;

        let input_slice = $unpacked.data.borrow_to_typed_slice::<$data_type>();
        let input = input_slice.as_ref();

        let mut output = Vec::with_capacity(num_chunks as usize * packed_chunk_size);

        // Loop over all but the last chunk.
        (0..num_full_chunks).for_each(|i| {
            let start_elem = (i * ELEMS_PER_CHUNK) as usize;

            let output_len = output.len();
            unsafe {
                output.set_len(output_len + packed_chunk_size);
                BitPacking::unchecked_pack(
                    $self.compressed_bit_width,
                    &input[start_elem..][..ELEMS_PER_CHUNK as usize],
                    &mut output[output_len..][..packed_chunk_size],
                );
            }
        });

        if num_chunks != num_full_chunks {
            let last_chunk_elem_num = $unpacked.num_values % ELEMS_PER_CHUNK;
            let mut last_chunk = vec![0 as $data_type; ELEMS_PER_CHUNK as usize];
            last_chunk[..last_chunk_elem_num as usize].clone_from_slice(
                &input[$unpacked.num_values as usize - last_chunk_elem_num as usize..],
            );

            let output_len = output.len();
            unsafe {
                output.set_len(output_len + packed_chunk_size);
                BitPacking::unchecked_pack(
                    $self.compressed_bit_width,
                    &last_chunk,
                    &mut output[output_len..][..packed_chunk_size],
                );
            }
        }

        let bitpacked_for_non_neg_buffer_index = *$buffer_index;
        *$buffer_index += 1;

        let encoding = ProtobufUtils::bitpacked_for_non_neg_encoding(
            $self.compressed_bit_width as u64,
            uncompressed_bit_width,
            bitpacked_for_non_neg_buffer_index,
        );
        let packed = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: $self.compressed_bit_width as u64,
            data: LanceBuffer::reinterpret_vec(output),
            num_values: $unpacked.num_values,
            block_info: BlockInfo::new(),
        });

        Result::Ok(EncodedArray {
            data: packed,
            encoding,
        })
    }};
}

#[derive(Debug)]
pub struct BitpackedForNonNegArrayEncoder {
    pub compressed_bit_width: usize,
    pub original_data_type: DataType,
}

impl BitpackedForNonNegArrayEncoder {
    pub fn new(compressed_bit_width: usize, data_type: DataType) -> Self {
        Self {
            compressed_bit_width,
            original_data_type: data_type,
        }
    }
}

impl ArrayEncoder for BitpackedForNonNegArrayEncoder {
    fn encode(
        &self,
        data: DataBlock,
        data_type: &DataType,
        buffer_index: &mut u32,
    ) -> Result<EncodedArray> {
        match data {
            DataBlock::AllNull(_) => {
                let encoding = ProtobufUtils::basic_all_null_encoding();
                Ok(EncodedArray { data, encoding })
            }
            DataBlock::FixedWidth(mut unpacked) => {
                match data_type {
                    DataType::UInt8 | DataType::Int8 => encode_fixed_width!(self, unpacked, u8, buffer_index),
                    DataType::UInt16 | DataType::Int16 => encode_fixed_width!(self, unpacked, u16, buffer_index),
                    DataType::UInt32 | DataType::Int32 => encode_fixed_width!(self, unpacked, u32, buffer_index),
                    DataType::UInt64 | DataType::Int64 => encode_fixed_width!(self, unpacked, u64, buffer_index),
                    _ => unreachable!("BitpackedForNonNegArrayEncoder only supports data types of UInt8, Int8, UInt16, Int16, UInt32, Int32, UInt64, Int64"),
                }
            }
            DataBlock::Nullable(nullable) => {
                let validity_buffer_index = *buffer_index;
                *buffer_index += 1;

                let validity_desc = ProtobufUtils::flat_encoding(
                    1,
                    validity_buffer_index,
                    /*compression=*/ None,
                );
                let encoded_values: EncodedArray;
                match *nullable.data {
                    DataBlock::FixedWidth(mut unpacked) => {
                        match data_type {
                            DataType::UInt8 | DataType::Int8 => encoded_values = encode_fixed_width!(self, unpacked, u8, buffer_index)?,
                            DataType::UInt16 | DataType::Int16 => encoded_values = encode_fixed_width!(self, unpacked, u16, buffer_index)?,
                            DataType::UInt32 | DataType::Int32 => encoded_values = encode_fixed_width!(self, unpacked, u32, buffer_index)?,
                            DataType::UInt64 | DataType::Int64 => encoded_values = encode_fixed_width!(self, unpacked, u64, buffer_index)?,
                            _ => unreachable!("BitpackedForNonNegArrayEncoder only supports data types of UInt8, Int8, UInt16, Int16, UInt32, Int32, UInt64, Int64"),
                        }
                    }
                    _ => {
                        return Err(Error::InvalidInput {
                            source: "Bitpacking only supports fixed width data blocks or a nullable data block with fixed width data block inside or a all null data block".into(),
                            location: location!(),
                        });
                    }
                }
                let encoding =
                    ProtobufUtils::basic_some_null_encoding(validity_desc, encoded_values.encoding);
                let encoded = DataBlock::Nullable(NullableDataBlock {
                    data: Box::new(encoded_values.data),
                    nulls: nullable.nulls,
                    block_info: BlockInfo::new(),
                });
                Ok(EncodedArray {
                    data: encoded,
                    encoding,
                })
            }
            _ => {
                Err(Error::InvalidInput {
                    source: "Bitpacking only supports fixed width data blocks or a nullable data block with fixed width data block inside or a all null data block".into(),
                    location: location!(),
                })
            }
        }
    }
}

#[derive(Debug)]
pub struct BitpackedForNonNegScheduler {
    compressed_bit_width: u64,
    uncompressed_bits_per_value: u64,
    buffer_offset: u64,
}

impl BitpackedForNonNegScheduler {
    pub fn new(
        compressed_bit_width: u64,
        uncompressed_bits_per_value: u64,
        buffer_offset: u64,
    ) -> Self {
        Self {
            compressed_bit_width,
            uncompressed_bits_per_value,
            buffer_offset,
        }
    }

    fn locate_chunk_start(&self, relative_row_num: u64) -> u64 {
        let chunk_size = ELEMS_PER_CHUNK * self.compressed_bit_width / 8;
        self.buffer_offset + (relative_row_num / ELEMS_PER_CHUNK * chunk_size)
    }

    fn locate_chunk_end(&self, relative_row_num: u64) -> u64 {
        let chunk_size = ELEMS_PER_CHUNK * self.compressed_bit_width / 8;
        self.buffer_offset + (relative_row_num / ELEMS_PER_CHUNK * chunk_size) + chunk_size
    }
}

impl PageScheduler for BitpackedForNonNegScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u64>],
        scheduler: &Arc<dyn crate::EncodingsIo>,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>> {
        assert!(!ranges.is_empty());

        let mut byte_ranges = vec![];

        // map one bytes to multiple ranges, one bytes has at least one range corresponding to it
        let mut bytes_idx_to_range_indices = vec![];
        let first_byte_range = std::ops::Range {
            start: self.locate_chunk_start(ranges[0].start),
            end: self.locate_chunk_end(ranges[0].end - 1),
        }; // the ranges are half-open
        byte_ranges.push(first_byte_range);
        bytes_idx_to_range_indices.push(vec![ranges[0].clone()]);

        for (i, range) in ranges.iter().enumerate().skip(1) {
            let this_start = self.locate_chunk_start(range.start);
            let this_end = self.locate_chunk_end(range.end - 1);

            // when the current range start is in the same chunk as the previous range's end, we colaesce this two bytes ranges
            // when the current range start is not in the same chunk as the previous range's end, we create a new bytes range
            if this_start == self.locate_chunk_start(ranges[i - 1].end - 1) {
                byte_ranges.last_mut().unwrap().end = this_end;
                bytes_idx_to_range_indices
                    .last_mut()
                    .unwrap()
                    .push(range.clone());
            } else {
                byte_ranges.push(this_start..this_end);
                bytes_idx_to_range_indices.push(vec![range.clone()]);
            }
        }

        trace!(
            "Scheduling I/O for {} ranges spread across byte range {}..{}",
            byte_ranges.len(),
            byte_ranges[0].start,
            byte_ranges.last().unwrap().end
        );

        let bytes = scheduler.submit_request(byte_ranges.clone(), top_level_row);

        // copy the necessary data from `self` to move into the async block
        let compressed_bit_width = self.compressed_bit_width;
        let uncompressed_bits_per_value = self.uncompressed_bits_per_value;
        let num_rows = ranges.iter().map(|range| range.end - range.start).sum();

        async move {
            let bytes = bytes.await?;
            let decompressed_output = bitpacked_for_non_neg_decode(
                compressed_bit_width,
                uncompressed_bits_per_value,
                &bytes,
                &bytes_idx_to_range_indices,
                num_rows,
            );
            Ok(Box::new(BitpackedForNonNegPageDecoder {
                uncompressed_bits_per_value,
                decompressed_buf: decompressed_output,
            }) as Box<dyn PrimitivePageDecoder>)
        }
        .boxed()
    }
}

#[derive(Debug)]
struct BitpackedForNonNegPageDecoder {
    // number of bits in the uncompressed value. E.g. this will be 32 for DataType::UInt32
    uncompressed_bits_per_value: u64,

    decompressed_buf: LanceBuffer,
}

impl PrimitivePageDecoder for BitpackedForNonNegPageDecoder {
    fn decode(&self, rows_to_skip: u64, num_rows: u64) -> Result<DataBlock> {
        if ![8, 16, 32, 64].contains(&self.uncompressed_bits_per_value) {
            return Err(Error::InvalidInput {
                source: "BitpackedForNonNegPageDecoder should only has uncompressed_bits_per_value of 8, 16, 32, or 64".into(),
                location: location!(),
            });
        }

        let elem_size_in_bytes = self.uncompressed_bits_per_value / 8;

        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: self.decompressed_buf.slice_with_length(
                (rows_to_skip * elem_size_in_bytes) as usize,
                (num_rows * elem_size_in_bytes) as usize,
            ),
            bits_per_value: self.uncompressed_bits_per_value,
            num_values: num_rows,
            block_info: BlockInfo::new(),
        }))
    }
}

macro_rules! bitpacked_decode {
    ($uncompressed_type:ty, $compressed_bit_width:expr, $data:expr, $bytes_idx_to_range_indices:expr, $num_rows:expr) => {{
        let mut decompressed: Vec<$uncompressed_type> = Vec::with_capacity($num_rows as usize);
        let packed_chunk_size_in_byte: usize = (ELEMS_PER_CHUNK * $compressed_bit_width) as usize / 8;
        let mut decompress_chunk_buf = vec![0 as $uncompressed_type; ELEMS_PER_CHUNK as usize];

        for (i, bytes) in $data.iter().enumerate() {
            let mut ranges_idx = 0;
            let mut curr_range_start = $bytes_idx_to_range_indices[i][0].start;
            let mut chunk_num = 0;

            while chunk_num * packed_chunk_size_in_byte < bytes.len() {
                // Copy for memory alignment
                let chunk_in_u8: Vec<u8> = bytes[chunk_num * packed_chunk_size_in_byte..]
                    [..packed_chunk_size_in_byte]
                    .to_vec();
                chunk_num += 1;
                let chunk = cast_slice(&chunk_in_u8);
                unsafe {
                    BitPacking::unchecked_unpack(
                        $compressed_bit_width as usize,
                        chunk,
                        &mut decompress_chunk_buf,
                    );
                }

                loop {
                    // Case 1: All the elements after (curr_range_start % ELEMS_PER_CHUNK) inside this chunk are needed.
                    let elems_after_curr_range_start_in_this_chunk =
                        ELEMS_PER_CHUNK - curr_range_start % ELEMS_PER_CHUNK;
                    if curr_range_start + elems_after_curr_range_start_in_this_chunk
                        <= $bytes_idx_to_range_indices[i][ranges_idx].end
                    {
                        decompressed.extend_from_slice(
                            &decompress_chunk_buf[(curr_range_start % ELEMS_PER_CHUNK) as usize..],
                        );
                        curr_range_start += elems_after_curr_range_start_in_this_chunk;
                        break;
                    } else {
                        // Case 2: Only part of the elements after (curr_range_start % ELEMS_PER_CHUNK) inside this chunk are needed.
                        let elems_this_range_needed_in_this_chunk =
                            ($bytes_idx_to_range_indices[i][ranges_idx].end - curr_range_start)
                                .min(ELEMS_PER_CHUNK - curr_range_start % ELEMS_PER_CHUNK);
                        decompressed.extend_from_slice(
                            &decompress_chunk_buf[(curr_range_start % ELEMS_PER_CHUNK) as usize..]
                                [..elems_this_range_needed_in_this_chunk as usize],
                        );
                        if curr_range_start + elems_this_range_needed_in_this_chunk
                            == $bytes_idx_to_range_indices[i][ranges_idx].end
                        {
                            ranges_idx += 1;
                            if ranges_idx == $bytes_idx_to_range_indices[i].len() {
                                break;
                            }
                            curr_range_start = $bytes_idx_to_range_indices[i][ranges_idx].start;
                        } else {
                            curr_range_start += elems_this_range_needed_in_this_chunk;
                        }
                    }
                }
            }
        }

        LanceBuffer::reinterpret_vec(decompressed)
    }};
}

fn bitpacked_for_non_neg_decode(
    compressed_bit_width: u64,
    uncompressed_bits_per_value: u64,
    data: &[Bytes],
    bytes_idx_to_range_indices: &[Vec<std::ops::Range<u64>>],
    num_rows: u64,
) -> LanceBuffer {
    match uncompressed_bits_per_value {
        8 => bitpacked_decode!(
            u8,
            compressed_bit_width,
            data,
            bytes_idx_to_range_indices,
            num_rows
        ),
        16 => bitpacked_decode!(
            u16,
            compressed_bit_width,
            data,
            bytes_idx_to_range_indices,
            num_rows
        ),
        32 => bitpacked_decode!(
            u32,
            compressed_bit_width,
            data,
            bytes_idx_to_range_indices,
            num_rows
        ),
        64 => bitpacked_decode!(
            u64,
            compressed_bit_width,
            data,
            bytes_idx_to_range_indices,
            num_rows
        ),
        _ => unreachable!(
            "bitpacked_for_non_neg_decode only supports 8, 16, 32, 64 uncompressed_bits_per_value"
        ),
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    // use arrow::array::{
    //     Int16Array, Int32Array, Int64Array, Int8Array, UInt16Array, UInt32Array, UInt64Array,
    //     UInt8Array,
    // };
    // use arrow::datatypes::DataType;

    // #[test_log::test(tokio::test)]
    // async fn test_compute_compressed_bit_width_for_non_neg() {}

    // use std::collections::HashMap;

    // use lance_datagen::RowCount;

    // use crate::testing::{check_round_trip_encoding_of_data, TestCases};
    // use crate::version::LanceFileVersion;

    // async fn check_round_trip_bitpacked(array: Arc<dyn Array>) {
    //     let test_cases = TestCases::default().with_file_version(LanceFileVersion::V2_1);
    //     check_round_trip_encoding_of_data(vec![array], &test_cases, HashMap::new()).await;
    // }

    // #[test_log::test(tokio::test)]
    // async fn test_bitpack_fastlanes_u8() {
    //     let values: Vec<u8> = vec![5; 1024];
    //     let array = UInt8Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<u8> = vec![66; 1000];
    //     let array = UInt8Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);

    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<u8> = vec![77; 2000];
    //     let array = UInt8Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);

    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<u8> = vec![0; 10000];
    //     let array = UInt8Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u8> = vec![88; 10000];
    //     let array = UInt8Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt8))
    //         .into_batch_rows(RowCount::from(1))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt8))
    //         .into_batch_rows(RowCount::from(20))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt8))
    //         .into_batch_rows(RowCount::from(50))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt8))
    //         .into_batch_rows(RowCount::from(100))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt8))
    //         .into_batch_rows(RowCount::from(1000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt8))
    //         .into_batch_rows(RowCount::from(1024))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt8))
    //         .into_batch_rows(RowCount::from(2000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt8))
    //         .into_batch_rows(RowCount::from(3000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;
    // }

    // #[test_log::test(tokio::test)]
    // async fn test_bitpack_fastlanes_u16() {
    //     let values: Vec<u16> = vec![5; 1024];
    //     let array = UInt16Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<u16> = vec![66; 1000];
    //     let array = UInt16Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);

    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<u16> = vec![77; 2000];
    //     let array = UInt16Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);

    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<u16> = vec![0; 10000];
    //     let array = UInt16Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u16> = vec![88; 10000];
    //     let array = UInt16Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u16> = vec![300; 100];
    //     let array = UInt16Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u16> = vec![800; 100];
    //     let array = UInt16Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt16))
    //         .into_batch_rows(RowCount::from(1))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt16))
    //         .into_batch_rows(RowCount::from(20))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt16))
    //         .into_batch_rows(RowCount::from(100))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt16))
    //         .into_batch_rows(RowCount::from(1000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt16))
    //         .into_batch_rows(RowCount::from(1024))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt16))
    //         .into_batch_rows(RowCount::from(2000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt16))
    //         .into_batch_rows(RowCount::from(3000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;
    // }

    // #[test_log::test(tokio::test)]
    // async fn test_bitpack_fastlanes_u32() {
    //     let values: Vec<u32> = vec![5; 1024];
    //     let array = UInt32Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<u32> = vec![7; 2000];
    //     let array = UInt32Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<u32> = vec![66; 1000];
    //     let array = UInt32Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<u32> = vec![666; 1000];
    //     let array = UInt32Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<u32> = vec![77; 2000];
    //     let array = UInt32Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<u32> = vec![0; 10000];
    //     let array = UInt32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u32> = vec![1; 10000];
    //     let array = UInt32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u32> = vec![88; 10000];
    //     let array = UInt32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u32> = vec![300; 100];
    //     let array = UInt32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u32> = vec![3000; 100];
    //     let array = UInt32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u32> = vec![800; 100];
    //     let array = UInt32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u32> = vec![8000; 100];
    //     let array = UInt32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u32> = vec![65536; 100];
    //     let array = UInt32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u32> = vec![655360; 100];
    //     let array = UInt32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt32))
    //         .into_batch_rows(RowCount::from(1))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt32))
    //         .into_batch_rows(RowCount::from(20))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt32))
    //         .into_batch_rows(RowCount::from(50))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt32))
    //         .into_batch_rows(RowCount::from(100))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt32))
    //         .into_batch_rows(RowCount::from(1000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt32))
    //         .into_batch_rows(RowCount::from(1024))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt32))
    //         .into_batch_rows(RowCount::from(2000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt32))
    //         .into_batch_rows(RowCount::from(3000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;
    // }

    // #[test_log::test(tokio::test)]
    // async fn test_bitpack_fastlanes_u64() {
    //     let values: Vec<u64> = vec![5; 1024];
    //     let array = UInt64Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<u64> = vec![7; 2000];
    //     let array = UInt64Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<u64> = vec![66; 1000];
    //     let array = UInt64Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<u64> = vec![666; 1000];
    //     let array = UInt64Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<u64> = vec![77; 2000];
    //     let array = UInt64Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<u64> = vec![0; 10000];
    //     let array = UInt64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u64> = vec![1; 10000];
    //     let array = UInt64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u64> = vec![88; 10000];
    //     let array = UInt64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u64> = vec![300; 100];
    //     let array = UInt64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u64> = vec![3000; 100];
    //     let array = UInt64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u64> = vec![800; 100];
    //     let array = UInt64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u64> = vec![8000; 100];
    //     let array = UInt64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u64> = vec![65536; 100];
    //     let array = UInt64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<u64> = vec![655360; 100];
    //     let array = UInt64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt64))
    //         .into_batch_rows(RowCount::from(1))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt64))
    //         .into_batch_rows(RowCount::from(20))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt64))
    //         .into_batch_rows(RowCount::from(50))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt64))
    //         .into_batch_rows(RowCount::from(100))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt64))
    //         .into_batch_rows(RowCount::from(1000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt64))
    //         .into_batch_rows(RowCount::from(1024))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt64))
    //         .into_batch_rows(RowCount::from(2000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::UInt64))
    //         .into_batch_rows(RowCount::from(3000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;
    // }

    // #[test_log::test(tokio::test)]
    // async fn test_bitpack_fastlanes_i8() {
    //     let values: Vec<i8> = vec![-5; 1024];
    //     let array = Int8Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<i8> = vec![66; 1000];
    //     let array = Int8Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);

    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<i8> = vec![77; 2000];
    //     let array = Int8Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);

    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<i8> = vec![0; 10000];
    //     let array = Int8Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i8> = vec![88; 10000];
    //     let array = Int8Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i8> = vec![-88; 10000];
    //     let array = Int8Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int8))
    //         .into_batch_rows(RowCount::from(1))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int8))
    //         .into_batch_rows(RowCount::from(20))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int8))
    //         .into_batch_rows(RowCount::from(50))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int8))
    //         .into_batch_rows(RowCount::from(100))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int8))
    //         .into_batch_rows(RowCount::from(1000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int8))
    //         .into_batch_rows(RowCount::from(1024))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int8))
    //         .into_batch_rows(RowCount::from(2000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int8))
    //         .into_batch_rows(RowCount::from(3000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;
    // }

    // #[test_log::test(tokio::test)]
    // async fn test_bitpack_fastlanes_i16() {
    //     let values: Vec<i16> = vec![-5; 1024];
    //     let array = Int16Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<i16> = vec![66; 1000];
    //     let array = Int16Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);

    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<i16> = vec![77; 2000];
    //     let array = Int16Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);

    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<i16> = vec![0; 10000];
    //     let array = Int16Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i16> = vec![88; 10000];
    //     let array = Int16Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i16> = vec![300; 100];
    //     let array = Int16Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i16> = vec![800; 100];
    //     let array = Int16Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int16))
    //         .into_batch_rows(RowCount::from(1))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int16))
    //         .into_batch_rows(RowCount::from(20))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int16))
    //         .into_batch_rows(RowCount::from(50))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int16))
    //         .into_batch_rows(RowCount::from(100))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int16))
    //         .into_batch_rows(RowCount::from(1000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int16))
    //         .into_batch_rows(RowCount::from(1024))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int16))
    //         .into_batch_rows(RowCount::from(2000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int16))
    //         .into_batch_rows(RowCount::from(3000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;
    // }

    // #[test_log::test(tokio::test)]
    // async fn test_bitpack_fastlanes_i32() {
    //     let values: Vec<i32> = vec![-5; 1024];
    //     let array = Int32Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<i32> = vec![66; 1000];
    //     let array = Int32Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<i32> = vec![-66; 1000];
    //     let array = Int32Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<i32> = vec![77; 2000];
    //     let array = Int32Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<i32> = vec![-77; 2000];
    //     let array = Int32Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<i32> = vec![0; 10000];
    //     let array = Int32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i32> = vec![88; 10000];
    //     let array = Int32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i32> = vec![-88; 10000];
    //     let array = Int32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i32> = vec![300; 100];
    //     let array = Int32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i32> = vec![-300; 100];
    //     let array = Int32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i32> = vec![800; 100];
    //     let array = Int32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i32> = vec![-800; 100];
    //     let array = Int32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i32> = vec![65536; 100];
    //     let array = Int32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i32> = vec![-65536; 100];
    //     let array = Int32Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int32))
    //         .into_batch_rows(RowCount::from(1))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int32))
    //         .into_batch_rows(RowCount::from(20))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int32))
    //         .into_batch_rows(RowCount::from(50))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int32))
    //         .into_batch_rows(RowCount::from(100))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int32))
    //         .into_batch_rows(RowCount::from(1000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int32))
    //         .into_batch_rows(RowCount::from(1024))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int32))
    //         .into_batch_rows(RowCount::from(2000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int32))
    //         .into_batch_rows(RowCount::from(3000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;
    // }

    // #[test_log::test(tokio::test)]
    // async fn test_bitpack_fastlanes_i64() {
    //     let values: Vec<i64> = vec![-5; 1024];
    //     let array = Int64Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<i64> = vec![66; 1000];
    //     let array = Int64Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<i64> = vec![-66; 1000];
    //     let array = Int64Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<i64> = vec![77; 2000];
    //     let array = Int64Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<i64> = vec![-77; 2000];
    //     let array = Int64Array::from(values);
    //     let array: Arc<dyn arrow_array::Array> = Arc::new(array);
    //     check_round_trip_bitpacked(array).await;

    //     let values: Vec<i64> = vec![0; 10000];
    //     let array = Int64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i64> = vec![88; 10000];
    //     let array = Int64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i64> = vec![-88; 10000];
    //     let array = Int64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i64> = vec![300; 100];
    //     let array = Int64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i64> = vec![-300; 100];
    //     let array = Int64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i64> = vec![800; 100];
    //     let array = Int64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i64> = vec![-800; 100];
    //     let array = Int64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i64> = vec![65536; 100];
    //     let array = Int64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let values: Vec<i64> = vec![-65536; 100];
    //     let array = Int64Array::from(values);
    //     let arr = Arc::new(array) as ArrayRef;
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int64))
    //         .into_batch_rows(RowCount::from(1))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int64))
    //         .into_batch_rows(RowCount::from(20))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int64))
    //         .into_batch_rows(RowCount::from(50))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int64))
    //         .into_batch_rows(RowCount::from(100))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int64))
    //         .into_batch_rows(RowCount::from(1000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int64))
    //         .into_batch_rows(RowCount::from(1024))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int64))
    //         .into_batch_rows(RowCount::from(2000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;

    //     let arr = lance_datagen::gen()
    //         .anon_col(lance_datagen::array::rand_type(&DataType::Int64))
    //         .into_batch_rows(RowCount::from(3000))
    //         .unwrap()
    //         .column(0)
    //         .clone();
    //     check_round_trip_bitpacked(arr).await;
    // }
}

// This macro chunks the FixedWidth DataBlock, bitpacks them with 1024 values per chunk,
// it puts the bit-width parameter in front of each chunk,
// and the bit-width parameter has the same bit-width as the uncompressed DataBlock
// for example, if the input DataBlock has `bits_per_value` of `16`, there will be 2 bytes(16 bits)
// in front of each chunk storing the `bit-width` parameter.
macro_rules! chunk_data_impl {
    ($data:expr, $data_type:ty) => {{
        let data_buffer = $data.data.borrow_to_typed_slice::<$data_type>();
        let data_buffer = data_buffer.as_ref();

        let bit_widths = $data
            .get_stat(Stat::BitWidth)
            .expect("FixedWidthDataBlock should have valid bit width statistics");
        let bit_widths_array = bit_widths
            .as_any()
            .downcast_ref::<PrimitiveArray<UInt64Type>>()
            .unwrap();

        let (packed_chunk_sizes, total_size) = bit_widths_array
            .values()
            .iter()
            .map(|&bit_width| {
                let chunk_size = ((1024 * bit_width) / $data.bits_per_value) as usize;
                (chunk_size, chunk_size + 1)
            })
            .fold(
                (Vec::with_capacity(bit_widths_array.len()), 0),
                |(mut sizes, total), (size, inc)| {
                    sizes.push(size);
                    (sizes, total + inc)
                },
            );

        let mut output: Vec<$data_type> = Vec::with_capacity(total_size);
        let mut chunks = Vec::with_capacity(bit_widths_array.len());

        for i in 0..bit_widths_array.len() - 1 {
            let start_elem = i * ELEMS_PER_CHUNK as usize;
            let bit_width = bit_widths_array.value(i) as $data_type;
            output.push(bit_width);
            let output_len = output.len();
            unsafe {
                output.set_len(output_len + packed_chunk_sizes[i]);
                BitPacking::unchecked_pack(
                    bit_width as usize,
                    &data_buffer[start_elem..][..ELEMS_PER_CHUNK as usize],
                    &mut output[output_len..][..packed_chunk_sizes[i]],
                );
            }
            chunks.push(MiniBlockChunk {
                num_bytes: ((1 + packed_chunk_sizes[i]) * std::mem::size_of::<$data_type>()) as u16,
                log_num_values: LOG_ELEMS_PER_CHUNK,
            });
        }

        // Handle the last chunk
        let last_chunk_elem_num = if $data.num_values % ELEMS_PER_CHUNK == 0 {
            1024
        } else {
            $data.num_values % ELEMS_PER_CHUNK
        };
        let mut last_chunk = vec![0; ELEMS_PER_CHUNK as usize];
        last_chunk[..last_chunk_elem_num as usize].clone_from_slice(
            &data_buffer[$data.num_values as usize - last_chunk_elem_num as usize..],
        );
        let bit_width = bit_widths_array.value(bit_widths_array.len() - 1) as $data_type;
        output.push(bit_width);
        let output_len = output.len();
        unsafe {
            output.set_len(output_len + packed_chunk_sizes[bit_widths_array.len() - 1]);
            BitPacking::unchecked_pack(
                bit_width as usize,
                &last_chunk,
                &mut output[output_len..][..packed_chunk_sizes[bit_widths_array.len() - 1]],
            );
        }
        chunks.push(MiniBlockChunk {
            num_bytes: ((1 + packed_chunk_sizes[bit_widths_array.len() - 1])
                * std::mem::size_of::<$data_type>()) as u16,
            log_num_values: 0,
        });

        (
            MiniBlockCompressed {
                data: LanceBuffer::reinterpret_vec(output),
                chunks,
                num_values: $data.num_values,
            },
            ProtobufUtils::bitpack2($data.bits_per_value),
        )
    }};
}

#[derive(Debug, Default)]
pub struct BitpackMiniBlockEncoder {}

impl BitpackMiniBlockEncoder {
    fn chunk_data(
        &self,
        mut data: FixedWidthDataBlock,
    ) -> (MiniBlockCompressed, crate::format::pb::ArrayEncoding) {
        assert!(data.bits_per_value % 8 == 0);
        match data.bits_per_value {
            8 => chunk_data_impl!(data, u8),
            16 => chunk_data_impl!(data, u16),
            32 => chunk_data_impl!(data, u32),
            64 => chunk_data_impl!(data, u64),
            _ => unreachable!(),
        }
    }
}

impl MiniBlockCompressor for BitpackMiniBlockEncoder {
    fn compress(
        &self,
        chunk: DataBlock,
    ) -> Result<(MiniBlockCompressed, crate::format::pb::ArrayEncoding)> {
        match chunk {
            DataBlock::FixedWidth(fixed_width) => Ok(self.chunk_data(fixed_width)),
            _ => Err(Error::InvalidInput {
                source: format!(
                    "Cannot compress a data block of type {} with BitpackMiniBlockEncoder",
                    chunk.name()
                )
                .into(),
                location: location!(),
            }),
        }
    }
}

/// A decompressor for fixed-width data that has
/// been written, as-is, to disk in single contiguous array
#[derive(Debug)]
pub struct BitpackMiniBlockDecompressor {
    uncompressed_bit_width: u64,
}

impl BitpackMiniBlockDecompressor {
    pub fn new(description: &pb::Bitpack2) -> Self {
        Self {
            uncompressed_bit_width: description.uncompressed_bits_per_value,
        }
    }
}

impl MiniBlockDecompressor for BitpackMiniBlockDecompressor {
    fn decompress(&self, data: LanceBuffer, num_values: u64) -> Result<DataBlock> {
        assert!(data.len() >= 8);
        assert!(num_values <= ELEMS_PER_CHUNK);

        // This macro decompresses a chunk(1024 values) of bitpacked values.
        macro_rules! decompress_impl {
            ($type:ty) => {{
                let uncompressed_bit_width = std::mem::size_of::<$type>() * 8;
                let mut decompressed = vec![0 as $type; ELEMS_PER_CHUNK as usize];

                // Copy for memory alignment
                let chunk_in_u8: Vec<u8> = data.to_vec();
                let bit_width_bytes = &chunk_in_u8[..std::mem::size_of::<$type>()];
                let bit_width_value = LittleEndian::read_uint(bit_width_bytes, std::mem::size_of::<$type>());
                let chunk = cast_slice(&chunk_in_u8[std::mem::size_of::<$type>()..]);

                // The bit-packed chunk should have number of bytes (bit_width_value * ELEMS_PER_CHUNK / 8)
                assert!(chunk.len() * std::mem::size_of::<$type>() == (bit_width_value * ELEMS_PER_CHUNK as u64) as usize / 8);

                unsafe {
                    BitPacking::unchecked_unpack(
                        bit_width_value as usize,
                        chunk,
                        &mut decompressed,
                    );
                }

                decompressed.shrink_to(num_values as usize);
                Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
                    data: LanceBuffer::reinterpret_vec(decompressed),
                    bits_per_value: uncompressed_bit_width as u64,
                    num_values,
                    block_info: BlockInfo::new(),
                }))
            }};
        }

        match self.uncompressed_bit_width {
            8 => decompress_impl!(u8),
            16 => decompress_impl!(u16),
            32 => decompress_impl!(u32),
            64 => decompress_impl!(u64),
            _ => todo!(),
        }
    }
}

#[cfg(test)]
mod test {
    use std::{collections::HashMap, sync::Arc};

    use arrow_array::{Int64Array, Int8Array};

    use arrow_schema::DataType;

    use arrow_array::Array;

    use crate::{
        testing::{check_round_trip_encoding_of_data, TestCases},
        version::LanceFileVersion,
    };

    #[test_log::test(tokio::test)]
    async fn test_miniblock_bitpack() {
        let test_cases = TestCases::default().with_file_version(LanceFileVersion::V2_1);

        let arrays = vec![
            Arc::new(Int8Array::from(vec![100; 1024])) as Arc<dyn Array>,
            Arc::new(Int8Array::from(vec![1; 1024])) as Arc<dyn Array>,
            Arc::new(Int8Array::from(vec![16; 1024])) as Arc<dyn Array>,
            Arc::new(Int8Array::from(vec![-1; 1024])) as Arc<dyn Array>,
            Arc::new(Int8Array::from(vec![5; 1])) as Arc<dyn Array>,
        ];
        check_round_trip_encoding_of_data(arrays, &test_cases, HashMap::new()).await;

        for data_type in [DataType::Int16, DataType::Int32, DataType::Int64] {
            let int64_arrays = vec![
                Int64Array::from(vec![3; 1024]),
                Int64Array::from(vec![8; 1024]),
                Int64Array::from(vec![16; 1024]),
                Int64Array::from(vec![100; 1024]),
                Int64Array::from(vec![512; 1024]),
                Int64Array::from(vec![1000; 1024]),
                Int64Array::from(vec![2000; 1024]),
                Int64Array::from(vec![-1; 10]),
            ];

            let mut arrays = vec![];
            for int64_array in int64_arrays {
                arrays.push(arrow_cast::cast(&int64_array, &data_type).unwrap());
            }
            check_round_trip_encoding_of_data(arrays, &test_cases, HashMap::new()).await;
        }
    }
}
