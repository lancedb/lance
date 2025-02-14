// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::datatypes::{
    Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};
use arrow_array::{Array, PrimitiveArray};
use arrow_buffer::ArrowNativeType;
use arrow_schema::DataType;
use byteorder::{ByteOrder, LittleEndian};
use bytes::Bytes;
use futures::future::{BoxFuture, FutureExt};
use log::trace;
use snafu::location;

use lance_arrow::DataTypeExt;
use lance_core::{Error, Result};

use crate::buffer::LanceBuffer;
use crate::compression_algo::fastlanes::BitPacking;
use crate::data::BlockInfo;
use crate::data::{DataBlock, FixedWidthDataBlock, NullableDataBlock};
use crate::decoder::{
    BlockDecompressor, FixedPerValueDecompressor, MiniBlockDecompressor, PageScheduler,
    PrimitivePageDecoder,
};
use crate::encoder::{
    ArrayEncoder, BlockCompressor, EncodedArray, MiniBlockChunk, MiniBlockCompressed,
    MiniBlockCompressor, PerValueCompressor, PerValueDataBlock,
};
use crate::format::{pb, ProtobufUtils};
use crate::statistics::{GetStat, Stat};
use arrow::array::ArrayRef;
use bytemuck::{cast_slice, AnyBitPattern};

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
        let num_chunks = $unpacked.num_values.div_ceil(ELEMS_PER_CHUNK);
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

#[derive(Debug, Default)]
pub struct InlineBitpacking {
    uncompressed_bit_width: u64,
}

impl InlineBitpacking {
    pub fn new(uncompressed_bit_width: u64) -> Self {
        Self {
            uncompressed_bit_width,
        }
    }

    pub fn from_description(description: &pb::InlineBitpacking) -> Self {
        Self {
            uncompressed_bit_width: description.uncompressed_bits_per_value,
        }
    }

    pub fn min_size_bytes(bit_width: u64) -> u64 {
        (ELEMS_PER_CHUNK * bit_width).div_ceil(8)
    }

    /// Bitpacks a FixedWidthDataBlock into compressed chunks of 1024 values
    ///
    /// Each chunk can have a different bit width
    ///
    /// Each chunk has the compressed bit width stored inline in the chunk itself.
    fn bitpack_chunked<T: ArrowNativeType + BitPacking>(
        mut data: FixedWidthDataBlock,
    ) -> MiniBlockCompressed {
        let data_buffer = data.data.borrow_to_typed_slice::<T>();
        let data_buffer = data_buffer.as_ref();

        let bit_widths = data.expect_stat(Stat::BitWidth);
        let bit_widths_array = bit_widths
            .as_any()
            .downcast_ref::<PrimitiveArray<UInt64Type>>()
            .unwrap();

        let (packed_chunk_sizes, total_size) = bit_widths_array
            .values()
            .iter()
            .map(|&bit_width| {
                let chunk_size = ((1024 * bit_width) / data.bits_per_value) as usize;
                (chunk_size, chunk_size + 1)
            })
            .fold(
                (Vec::with_capacity(bit_widths_array.len()), 0),
                |(mut sizes, total), (size, inc)| {
                    sizes.push(size);
                    (sizes, total + inc)
                },
            );

        let mut output: Vec<T> = Vec::with_capacity(total_size);
        let mut chunks = Vec::with_capacity(bit_widths_array.len());

        for i in 0..bit_widths_array.len() - 1 {
            let start_elem = i * ELEMS_PER_CHUNK as usize;
            let bit_width = bit_widths_array.value(i) as usize;
            output.push(T::from_usize(bit_width).unwrap());
            let output_len = output.len();
            unsafe {
                output.set_len(output_len + packed_chunk_sizes[i]);
                BitPacking::unchecked_pack(
                    bit_width,
                    &data_buffer[start_elem..][..ELEMS_PER_CHUNK as usize],
                    &mut output[output_len..][..packed_chunk_sizes[i]],
                );
            }
            chunks.push(MiniBlockChunk {
                buffer_sizes: vec![((1 + packed_chunk_sizes[i]) * std::mem::size_of::<T>()) as u16],
                log_num_values: LOG_ELEMS_PER_CHUNK,
            });
        }

        // Handle the last chunk
        let last_chunk_elem_num = if data.num_values % ELEMS_PER_CHUNK == 0 {
            1024
        } else {
            data.num_values % ELEMS_PER_CHUNK
        };
        let mut last_chunk: Vec<T> = vec![T::from_usize(0).unwrap(); ELEMS_PER_CHUNK as usize];
        last_chunk[..last_chunk_elem_num as usize].clone_from_slice(
            &data_buffer[data.num_values as usize - last_chunk_elem_num as usize..],
        );
        let bit_width = bit_widths_array.value(bit_widths_array.len() - 1) as usize;
        output.push(T::from_usize(bit_width).unwrap());
        let output_len = output.len();
        unsafe {
            output.set_len(output_len + packed_chunk_sizes[bit_widths_array.len() - 1]);
            BitPacking::unchecked_pack(
                bit_width,
                &last_chunk,
                &mut output[output_len..][..packed_chunk_sizes[bit_widths_array.len() - 1]],
            );
        }
        chunks.push(MiniBlockChunk {
            buffer_sizes: vec![
                ((1 + packed_chunk_sizes[bit_widths_array.len() - 1]) * std::mem::size_of::<T>())
                    as u16,
            ],
            log_num_values: 0,
        });

        MiniBlockCompressed {
            data: vec![LanceBuffer::reinterpret_vec(output)],
            chunks,
            num_values: data.num_values,
        }
    }

    fn chunk_data(
        &self,
        data: FixedWidthDataBlock,
    ) -> (MiniBlockCompressed, crate::format::pb::ArrayEncoding) {
        assert!(data.bits_per_value % 8 == 0);
        assert_eq!(data.bits_per_value, self.uncompressed_bit_width);
        let bits_per_value = data.bits_per_value;
        let compressed = match bits_per_value {
            8 => Self::bitpack_chunked::<u8>(data),
            16 => Self::bitpack_chunked::<u16>(data),
            32 => Self::bitpack_chunked::<u32>(data),
            64 => Self::bitpack_chunked::<u64>(data),
            _ => unreachable!(),
        };
        (compressed, ProtobufUtils::inline_bitpacking(bits_per_value))
    }

    fn unchunk<T: ArrowNativeType + BitPacking + AnyBitPattern>(
        data: LanceBuffer,
        num_values: u64,
    ) -> Result<DataBlock> {
        assert!(data.len() >= 8);
        assert!(num_values <= ELEMS_PER_CHUNK);

        // This macro decompresses a chunk(1024 values) of bitpacked values.
        let uncompressed_bit_width = std::mem::size_of::<T>() * 8;
        let mut decompressed = vec![T::from_usize(0).unwrap(); ELEMS_PER_CHUNK as usize];

        // Copy for memory alignment
        let chunk_in_u8: Vec<u8> = data.to_vec();
        let bit_width_bytes = &chunk_in_u8[..std::mem::size_of::<T>()];
        let bit_width_value = LittleEndian::read_uint(bit_width_bytes, std::mem::size_of::<T>());
        let chunk = cast_slice(&chunk_in_u8[std::mem::size_of::<T>()..]);

        // The bit-packed chunk should have number of bytes (bit_width_value * ELEMS_PER_CHUNK / 8)
        assert!(
            chunk.len() * std::mem::size_of::<T>()
                == (bit_width_value * ELEMS_PER_CHUNK as u64) as usize / 8
        );

        unsafe {
            BitPacking::unchecked_unpack(bit_width_value as usize, chunk, &mut decompressed);
        }

        decompressed.truncate(num_values as usize);
        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::reinterpret_vec(decompressed),
            bits_per_value: uncompressed_bit_width as u64,
            num_values,
            block_info: BlockInfo::new(),
        }))
    }
}

impl MiniBlockCompressor for InlineBitpacking {
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

impl BlockCompressor for InlineBitpacking {
    fn compress(&self, data: DataBlock) -> Result<LanceBuffer> {
        println!("Compressing rep-def levels with inline bitpacking");
        let fixed_width = data.as_fixed_width().unwrap();
        let (chunked, _) = self.chunk_data(fixed_width);
        Ok(chunked.data.into_iter().next().unwrap())
    }
}

impl MiniBlockDecompressor for InlineBitpacking {
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        assert_eq!(data.len(), 1);
        let data = data.into_iter().next().unwrap();
        match self.uncompressed_bit_width {
            8 => Self::unchunk::<u8>(data, num_values),
            16 => Self::unchunk::<u16>(data, num_values),
            32 => Self::unchunk::<u32>(data, num_values),
            64 => Self::unchunk::<u64>(data, num_values),
            _ => unimplemented!("Bitpacking word size must be 8, 16, 32, or 64"),
        }
    }
}

impl BlockDecompressor for InlineBitpacking {
    fn decompress(&self, data: LanceBuffer, num_values: u64) -> Result<DataBlock> {
        match self.uncompressed_bit_width {
            8 => Self::unchunk::<u8>(data, num_values),
            16 => Self::unchunk::<u16>(data, num_values),
            32 => Self::unchunk::<u32>(data, num_values),
            64 => Self::unchunk::<u64>(data, num_values),
            _ => unimplemented!("Bitpacking word size must be 8, 16, 32, or 64"),
        }
    }
}

/// Bitpacks a FixedWidthDataBlock with a given bit width
///
/// This function is simpler as it does not do any chunking, but slightly less efficient.
/// The compressed bits per value is constant across the entire buffer.
///
/// Note: even though we are not strictly "chunking" we are still operating on chunks of
/// 1024 values because that's what the bitpacking primitives expect.  They just don't
/// have a unique bit width for each chunk.
fn bitpack_out_of_line<T: ArrowNativeType + BitPacking>(
    mut data: FixedWidthDataBlock,
    bit_width: usize,
) -> LanceBuffer {
    let data_buffer = data.data.borrow_to_typed_slice::<T>();
    let data_buffer = data_buffer.as_ref();

    let num_chunks = data_buffer.len().div_ceil(ELEMS_PER_CHUNK as usize);
    let last_chunk_is_runt = data_buffer.len() % ELEMS_PER_CHUNK as usize != 0;
    let words_per_chunk =
        (ELEMS_PER_CHUNK as usize * bit_width).div_ceil(data.bits_per_value as usize);
    let mut output: Vec<T> = Vec::with_capacity(num_chunks * words_per_chunk);
    unsafe {
        output.set_len(num_chunks * words_per_chunk);
    }

    let num_whole_chunks = if last_chunk_is_runt {
        num_chunks - 1
    } else {
        num_chunks
    };

    // Simple case for complete chunks
    for i in 0..num_whole_chunks {
        let input_start = i * ELEMS_PER_CHUNK as usize;
        let input_end = input_start + ELEMS_PER_CHUNK as usize;
        let output_start = i * words_per_chunk;
        let output_end = output_start + words_per_chunk;
        unsafe {
            BitPacking::unchecked_pack(
                bit_width,
                &data_buffer[input_start..input_end],
                &mut output[output_start..output_end],
            );
        }
    }

    if !last_chunk_is_runt {
        return LanceBuffer::reinterpret_vec(output);
    }

    // If we get here then the last chunk needs to be padded with zeros
    let remaining_items = data_buffer.len() % ELEMS_PER_CHUNK as usize;
    let last_chunk_start = num_whole_chunks * ELEMS_PER_CHUNK as usize;

    let mut last_chunk: Vec<T> = vec![T::from_usize(0).unwrap(); ELEMS_PER_CHUNK as usize];
    last_chunk[..remaining_items as usize].clone_from_slice(&data_buffer[last_chunk_start..]);
    let output_start = num_whole_chunks * words_per_chunk;
    unsafe {
        BitPacking::unchecked_pack(bit_width, &last_chunk, &mut output[output_start..]);
    }

    LanceBuffer::reinterpret_vec(output)
}

/// Unpacks a FixedWidthDataBlock that has been bitpacked with a constant bit width
fn unpack_out_of_line<T: ArrowNativeType + BitPacking>(
    mut data: FixedWidthDataBlock,
    num_values: usize,
    bits_per_value: usize,
) -> FixedWidthDataBlock {
    let words_per_chunk =
        (ELEMS_PER_CHUNK as usize * bits_per_value).div_ceil(data.bits_per_value as usize);
    let compressed_words = data.data.borrow_to_typed_slice::<T>();

    let num_chunks = data.num_values as usize / words_per_chunk;
    debug_assert_eq!(data.num_values as usize % words_per_chunk, 0);

    // This is slightly larger than num_values because the last chunk has some padding, we will truncate at the end
    let mut decompressed = Vec::with_capacity(num_chunks * ELEMS_PER_CHUNK as usize);
    unsafe {
        decompressed.set_len(num_chunks * ELEMS_PER_CHUNK as usize);
    }

    for chunk_idx in 0..num_chunks {
        let input_start = chunk_idx * words_per_chunk;
        let input_end = input_start + words_per_chunk;
        let output_start = chunk_idx * ELEMS_PER_CHUNK as usize;
        let output_end = output_start + ELEMS_PER_CHUNK as usize;
        unsafe {
            BitPacking::unchecked_unpack(
                bits_per_value,
                &compressed_words[input_start..input_end],
                &mut decompressed[output_start..output_end],
            );
        }
    }

    decompressed.truncate(num_values);

    FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(decompressed),
        bits_per_value: data.bits_per_value,
        num_values: num_values as u64,
        block_info: BlockInfo::new(),
    }
}

/// A transparent compressor that bit packs data
///
/// In order for the encoding to be transparent we must have a fixed bit width
/// across the entire array.  Chunking within the buffer is not supported.  This
/// means that we will be slightly less efficient than something like the mini-block
/// approach.
///
/// WARNING: DO NOT USE YET.
///
/// This was an interesting experiment but it can't be used as a per-value compressor
/// at the moment.  The resulting data IS transparent but it's not quite so simple.  We
/// compress in blocks of 1024 and each block has a fixed size but also has some padding.
///
/// In other words, if we try the simple math to access the item at index `i` we will be
/// out of luck because `bits_per_value * i` is not the location.  What we need is something
/// like:
///
/// ```ignore
/// let chunk_idx = i / 1024;
/// let chunk_offset = i % 1024;
/// bits_per_chunk * chunk_idx + bits_per_value * chunk_offset
/// ```
///
/// However, this logic isn't expressible with the per-value traits we have today.  We can
/// enhance these traits should we need to support it at some point in the future.
#[derive(Debug)]
pub struct OutOfLineBitpacking {
    compressed_bit_width: usize,
}

impl PerValueCompressor for OutOfLineBitpacking {
    fn compress(
        &self,
        data: DataBlock,
    ) -> Result<(crate::encoder::PerValueDataBlock, pb::ArrayEncoding)> {
        let fixed_width = data.as_fixed_width().unwrap();
        let num_values = fixed_width.num_values;
        let word_size = fixed_width.bits_per_value;
        let compressed = match word_size {
            8 => bitpack_out_of_line::<u8>(fixed_width, self.compressed_bit_width),
            16 => bitpack_out_of_line::<u16>(fixed_width, self.compressed_bit_width),
            32 => bitpack_out_of_line::<u32>(fixed_width, self.compressed_bit_width),
            64 => bitpack_out_of_line::<u64>(fixed_width, self.compressed_bit_width),
            _ => panic!("Bitpacking word size must be 8,16,32,64"),
        };
        let compressed = FixedWidthDataBlock {
            data: compressed,
            bits_per_value: self.compressed_bit_width as u64,
            num_values,
            block_info: BlockInfo::new(),
        };
        let encoding =
            ProtobufUtils::out_of_line_bitpacking(word_size, self.compressed_bit_width as u64);
        Ok((PerValueDataBlock::Fixed(compressed), encoding))
    }
}

impl FixedPerValueDecompressor for OutOfLineBitpacking {
    fn decompress(&self, data: FixedWidthDataBlock, num_values: u64) -> Result<DataBlock> {
        let unpacked = match data.bits_per_value {
            8 => unpack_out_of_line::<u8>(data, num_values as usize, self.compressed_bit_width),
            16 => unpack_out_of_line::<u16>(data, num_values as usize, self.compressed_bit_width),
            32 => unpack_out_of_line::<u32>(data, num_values as usize, self.compressed_bit_width),
            64 => unpack_out_of_line::<u64>(data, num_values as usize, self.compressed_bit_width),
            _ => panic!("Bitpacking word size must be 8,16,32,64"),
        };
        Ok(DataBlock::FixedWidth(unpacked))
    }

    fn bits_per_value(&self) -> u64 {
        self.compressed_bit_width as u64
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
