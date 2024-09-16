// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::datatypes::{
    ArrowPrimitiveType, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type,
    UInt64Type, UInt8Type,
};
use arrow::util::bit_util::ceil;
use arrow_array::{cast::AsArray, Array, PrimitiveArray};
use arrow_schema::DataType;
use bytes::Bytes;
use futures::future::{BoxFuture, FutureExt};
use log::trace;
use num_traits::{AsPrimitive, PrimInt, ToPrimitive};
use snafu::{location, Location};

use lance_arrow::DataTypeExt;
use lance_core::{Error, Result};

use crate::buffer::LanceBuffer;
use crate::data::{DataBlock, FixedWidthDataBlock};
use crate::decoder::{PageScheduler, PrimitivePageDecoder};
use crate::encoder::{ArrayEncoder, EncodedArray};
use crate::format::ProtobufUtils;
use arrow::array::ArrayRef;
use fastlanes::BitPacking;

// Compute the compressed_bit_width for a given array of unsigned integers
// the vortex approach is better, they compute all stastistics before encoding
pub fn compute_compressed_bit_width_for_non_neg(arrays: &[ArrayRef]) -> u64 {
    // is it possible to get here?
    if arrays.len() == 0 {
        return 0;
    }

    let res;

    match arrays[0].data_type() {
        DataType::UInt8 | DataType::Int8 => {
            let mut global_max: u8 = 0;
            for array in arrays {
                // at this point we know that the array doesn't contain any negative values
                let primitive_array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<UInt8Type>>()
                    .unwrap();
                //println!("primitive_array: {:?}", primitive_array);
                let array_max = arrow::compute::bit_or(primitive_array);
                global_max = global_max.max(array_max.unwrap_or(0));
            }
            let num_bits =
                arrays[0].data_type().byte_width() as u64 * 8 - global_max.leading_zeros() as u64;
            if num_bits == 0 {
                res = 1;
            } else {
                res = num_bits;
            }
        }
        DataType::UInt16 | DataType::Int16 => {
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
        DataType::UInt32 | DataType::Int32 => {
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
        DataType::UInt64 | DataType::Int64 => {
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
        _ => {
            res = 8;
        }
    };
    res
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
        _data_type: &DataType,
        buffer_index: &mut u32,
    ) -> Result<EncodedArray> {
        let DataBlock::FixedWidth(mut unpacked) = data else {
            return Err(Error::InvalidInput {
                source: "Bitpacking only supports fixed width data blocks".into(),
                location: location!(),
            });
        };
        match _data_type {
            DataType::UInt8 | DataType::Int8 => {
                let num_chunks = (unpacked.num_values + 1023) / 1024;
                let num_full_chunks = unpacked.num_values / 1024;
                // there is no ceiling needed to calculate the size of the packed chunk because 1024 has divisor 8
                let packed_chunk_size = 1024 * self.compressed_bit_width / 8;

                let input = unpacked.data.reinterpret_to_rust_native::<u8>().unwrap();
                let mut output = Vec::with_capacity(num_chunks as usize * packed_chunk_size);

                // Loop over all but the last chunk.
                (0..num_full_chunks).for_each(|i| {
                    let start_elem = i as usize * 1024 as usize;

                    let output_len = output.len();
                    unsafe {
                        output.set_len(output_len + packed_chunk_size);
                        BitPacking::unchecked_pack(
                            self.compressed_bit_width,
                            &input[start_elem as usize..][..1024],
                            &mut output[output_len..][..packed_chunk_size],
                        );
                    };
                });

                if num_chunks != num_full_chunks {
                    let last_chunk_elem_num = unpacked.num_values % 1024;
                    let mut last_chunk = vec![0u8; 1024];
                    last_chunk[..last_chunk_elem_num as usize].clone_from_slice(
                        &input[unpacked.num_values as usize - last_chunk_elem_num as usize..],
                    );

                    let output_len = output.len();
                    unsafe {
                        output.set_len(output.len() + packed_chunk_size);
                        BitPacking::unchecked_pack(
                            self.compressed_bit_width,
                            &last_chunk,
                            &mut output[output_len..][..packed_chunk_size],
                        );
                    }
                }
                let bitpacked_for_non_neg_buffer_index = *buffer_index;
                *buffer_index += 1;

                let encoding = ProtobufUtils::bitpacked_for_non_neg_encoding(
                    self.compressed_bit_width as u64,
                    _data_type.byte_width() as u64 * 8,
                    bitpacked_for_non_neg_buffer_index,
                );
                let packed = DataBlock::FixedWidth(FixedWidthDataBlock {
                    bits_per_value: self.compressed_bit_width as u64,
                    data: LanceBuffer::Owned(output),
                    num_values: unpacked.num_values,
                });

                return Ok(EncodedArray {
                    data: packed,
                    encoding,
                });
            }
            _ => todo!(),
        }
    }
}

#[derive(Debug)]
pub struct BitpackedForNonNegScheduler {
    compressed_bit_width: u64,
    uncompressed_bits_per_value: u64,
    buffer_offset: u64,
}

fn locate_chunk_start(scheduler: &BitpackedForNonNegScheduler, relative_row_num: u64) -> u64 {
    let elems_per_chunk = 1024;
    let chunk_size = elems_per_chunk * scheduler.compressed_bit_width / 8;
    relative_row_num / elems_per_chunk * chunk_size
}

fn locate_chunk_end(scheduler: &BitpackedForNonNegScheduler, relative_row_num: u64) -> u64 {
    let elems_per_chunk: u64 = 1024;
    let chunk_size = elems_per_chunk * scheduler.compressed_bit_width / 8;
    relative_row_num / elems_per_chunk * chunk_size + chunk_size
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
}

impl PageScheduler for BitpackedForNonNegScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u64>],
        scheduler: &Arc<dyn crate::EncodingsIo>,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>> {
        // can we get here?
        if ranges.is_empty() {
            panic!("cannot schedule empty ranges");
        }
        let mut byte_ranges = vec![];
        let mut bytes_idx_to_range_indices = vec![];
        let first_byte_range = std::ops::Range {
            start: self.buffer_offset + locate_chunk_start(self, ranges[0].start),
            end: self.buffer_offset + locate_chunk_end(self, ranges[0].end - 1),
        }; // the ranges are half-open
        byte_ranges.push(first_byte_range);
        bytes_idx_to_range_indices.push(vec![ranges[0].clone()]);
        for (i, range) in ranges.iter().enumerate().skip(1) {
            let this_start = locate_chunk_start(self, range.start);
            let this_end = locate_chunk_end(self, range.end - 1);
            if this_start == locate_chunk_start(self, ranges[i - 1].end - 1) {
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
        /*
        println!(
            "Scheduling I/O for {} ranges spread across byte range {}..{}",
            byte_ranges.len(),
            byte_ranges[0].start,
            byte_ranges[byte_ranges.len() - 1].end
        );
        println!("ranges: {:?}", ranges);
        println!("byte_ranges: {:?}", byte_ranges);
        */

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

    decompressed_buf: Vec<u8>,
}

impl PrimitivePageDecoder for BitpackedForNonNegPageDecoder {
    fn decode(&self, rows_to_skip: u64, num_rows: u64) -> Result<DataBlock> {
        match self.uncompressed_bits_per_value {
            8 => {
                // I did an extra copy here, not sure how to avoid it and whether it's safe to avoid it
                let mut output = Vec::with_capacity(num_rows as usize);
                output.extend_from_slice(
                    &self.decompressed_buf[rows_to_skip as usize..][..num_rows as usize],
                );
                let res = Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
                    data: LanceBuffer::from(output),
                    bits_per_value: self.uncompressed_bits_per_value,
                    num_values: num_rows,
                }));
                return res;
            }
            _ => panic!("Unsupported data type"),
        }
    }
}

fn bitpacked_for_non_neg_decode(
    compressed_bit_width: u64,
    uncompressed_bits_per_value: u64,
    data: &Vec<Bytes>,
    bytes_idx_to_range_indices: &Vec<Vec<std::ops::Range<u64>>>,
    num_rows: u64,
) -> Vec<u8> {
    match uncompressed_bits_per_value {
        8 => {
            let mut decompressed: Vec<u8> = Vec::with_capacity(num_rows as usize);
            let packed_chunk_size: usize = 1024 * compressed_bit_width as usize / 8;
            let mut decompress_chunk_buf = vec![0_u8; 1024];
            for (i, bytes) in data.iter().enumerate() {
                let mut j = 0;
                let mut ranges_idx = 0;
                let mut curr_range_start = bytes_idx_to_range_indices[i][0].start;
                while j * packed_chunk_size < bytes.len() {
                    let chunk: &[u8] = &bytes[j * packed_chunk_size..][..packed_chunk_size];
                    unsafe {
                        BitPacking::unchecked_unpack(
                            compressed_bit_width as usize,
                            chunk,
                            &mut decompress_chunk_buf[..1024],
                        );
                    }
                    loop {
                        if curr_range_start + 1024 < bytes_idx_to_range_indices[i][ranges_idx].end {
                            let this_part_len = 1024 - curr_range_start % 1024;
                            decompressed.extend_from_slice(
                                &decompress_chunk_buf[curr_range_start as usize % 1024..],
                            );
                            curr_range_start += this_part_len;
                            break;
                        } else {
                            let this_part_len =
                                bytes_idx_to_range_indices[i][ranges_idx].end - curr_range_start;
                            decompressed.extend_from_slice(
                                &decompress_chunk_buf[curr_range_start as usize % 1024..]
                                    [..this_part_len as usize],
                            );
                            ranges_idx += 1;
                            if ranges_idx == bytes_idx_to_range_indices[i].len() {
                                break;
                            }
                            curr_range_start = bytes_idx_to_range_indices[i][ranges_idx].start;
                        }
                    }
                    j += 1;
                }
            }
            decompressed
        }
        _ => panic!("Unsupported data type"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, PrimitiveArray};
    use arrow::buffer::Buffer;
    use arrow::datatypes::DataType;
    use arrow::record_batch::RecordBatch;
    use arrow::util::bit_util::ceil;

    use crate::decoder::decode_batch;
    use crate::decoder::DecoderMiddlewareChain;
    use crate::decoder::FilterExpression;
    use crate::encoder::encode_batch;
    use crate::encoder::CoreFieldEncodingStrategy;
    use crate::encoder::EncodingOptions;
    use arrow::array::UInt8Array;
    use arrow::datatypes::Field;
    use arrow::datatypes::Schema;

    #[test_log::test(tokio::test)]
    async fn test_compute_compressed_bit_width_for_non_neg() {}

    use std::collections::HashMap;

    use lance_datagen::{ByteCount, RowCount};

    use crate::testing::{check_round_trip_encoding_of_data, TestCases};

    #[test_log::test(tokio::test)]
    async fn test_bitpack_fastlanes() {
        let values: Vec<u8> = vec![5; 1024];
        let array = UInt8Array::from(values);
        let array: Arc<dyn arrow_array::Array> = Arc::new(array);
        check_round_trip_encoding_of_data(vec![array], &TestCases::default(), HashMap::new()).await;

        let values: Vec<u8> = vec![66; 1000];
        let array = UInt8Array::from(values);
        let array: Arc<dyn arrow_array::Array> = Arc::new(array);

        check_round_trip_encoding_of_data(vec![array], &TestCases::default(), HashMap::new()).await;

        let values: Vec<u8> = vec![77; 2000];
        let array = UInt8Array::from(values);
        let array: Arc<dyn arrow_array::Array> = Arc::new(array);

        check_round_trip_encoding_of_data(vec![array], &TestCases::default(), HashMap::new()).await;

        let values: Vec<u8> = vec![0; 10000];
        let array = UInt8Array::from(values);
        let arr = Arc::new(array) as ArrayRef;
        check_round_trip_encoding_of_data(vec![arr], &TestCases::default(), HashMap::new()).await;

        let values: Vec<u8> = vec![88; 10000];
        let array = UInt8Array::from(values);
        let arr = Arc::new(array) as ArrayRef;
        check_round_trip_encoding_of_data(vec![arr], &TestCases::default(), HashMap::new()).await;

        let arr = lance_datagen::gen()
            .anon_col(lance_datagen::array::rand_primitive::<UInt8Type>(
                arrow::datatypes::UInt8Type::DATA_TYPE,
            ))
            .into_batch_rows(RowCount::from(1))
            .unwrap()
            .column(0)
            .clone();
        check_round_trip_encoding_of_data(vec![arr], &TestCases::default(), HashMap::new()).await;

        let arr = lance_datagen::gen()
            .anon_col(lance_datagen::array::rand_primitive::<UInt8Type>(
                arrow::datatypes::UInt8Type::DATA_TYPE,
            ))
            .into_batch_rows(RowCount::from(20))
            .unwrap()
            .column(0)
            .clone();
        check_round_trip_encoding_of_data(vec![arr], &TestCases::default(), HashMap::new()).await;

        let arr = lance_datagen::gen()
            .anon_col(lance_datagen::array::rand_primitive::<UInt8Type>(
                arrow::datatypes::UInt8Type::DATA_TYPE,
            ))
            .into_batch_rows(RowCount::from(50))
            .unwrap()
            .column(0)
            .clone();
        check_round_trip_encoding_of_data(vec![arr], &TestCases::default(), HashMap::new()).await;

        let arr = lance_datagen::gen()
            .anon_col(lance_datagen::array::rand_primitive::<UInt8Type>(
                arrow::datatypes::UInt8Type::DATA_TYPE,
            ))
            .into_batch_rows(RowCount::from(100))
            .unwrap()
            .column(0)
            .clone();
        check_round_trip_encoding_of_data(vec![arr], &TestCases::default(), HashMap::new()).await;

        let arr = lance_datagen::gen()
            .anon_col(lance_datagen::array::rand_primitive::<UInt8Type>(
                arrow::datatypes::UInt8Type::DATA_TYPE,
            ))
            .into_batch_rows(RowCount::from(1000))
            .unwrap()
            .column(0)
            .clone();
        check_round_trip_encoding_of_data(vec![arr], &TestCases::default(), HashMap::new()).await;

        let arr = lance_datagen::gen()
            .anon_col(lance_datagen::array::rand_primitive::<UInt8Type>(
                arrow::datatypes::UInt8Type::DATA_TYPE,
            ))
            .into_batch_rows(RowCount::from(1024))
            .unwrap()
            .column(0)
            .clone();
        check_round_trip_encoding_of_data(vec![arr], &TestCases::default(), HashMap::new()).await;

        let arr = lance_datagen::gen()
            .anon_col(lance_datagen::array::rand_primitive::<UInt8Type>(
                arrow::datatypes::UInt8Type::DATA_TYPE,
            ))
            .into_batch_rows(RowCount::from(2000))
            .unwrap()
            .column(0)
            .clone();
        check_round_trip_encoding_of_data(vec![arr], &TestCases::default(), HashMap::new()).await;

        let arr = lance_datagen::gen()
            .anon_col(lance_datagen::array::rand_primitive::<UInt8Type>(
                arrow::datatypes::UInt8Type::DATA_TYPE,
            ))
            .into_batch_rows(RowCount::from(3000))
            .unwrap()
            .column(0)
            .clone();
        check_round_trip_encoding_of_data(vec![arr], &TestCases::default(), HashMap::new()).await;
    }
}
