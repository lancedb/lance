// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use core::panic;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::ArrayRef;
use arrow_buffer::{bit_util, BooleanBuffer, BooleanBufferBuilder, NullBuffer, ScalarBuffer};
use futures::TryFutureExt;
use futures::{future::BoxFuture, FutureExt};

use crate::decoder::LogicalPageDecoder;
use crate::encodings::logical::primitive::PrimitiveFieldDecoder;

use crate::buffer::LanceBuffer;
use crate::data::{
    BlockInfo, DataBlock, FixedWidthDataBlock, NullableDataBlock, UsedEncoding, VariableWidthBlock,
};
use crate::format::ProtobufUtils;
use crate::{
    decoder::{PageScheduler, PrimitivePageDecoder},
    encoder::{ArrayEncoder, EncodedArray},
    EncodingsIo,
};

use arrow_array::{PrimitiveArray, UInt64Array};
use arrow_schema::DataType;
use lance_core::Result;

use super::block_compress::{BufferCompressor, CompressionScheme, GeneralBufferCompressor};

struct IndicesNormalizer {
    indices: Vec<u64>,
    validity: BooleanBufferBuilder,
    null_adjustment: u64,
}

impl IndicesNormalizer {
    fn new(num_rows: u64, null_adjustment: u64) -> Self {
        let mut indices = Vec::with_capacity(num_rows as usize);
        indices.push(0);
        Self {
            indices,
            validity: BooleanBufferBuilder::new(num_rows as usize),
            null_adjustment,
        }
    }

    fn normalize(&self, val: u64) -> (bool, u64) {
        if val >= self.null_adjustment {
            (false, val - self.null_adjustment)
        } else {
            (true, val)
        }
    }

    fn extend(&mut self, new_indices: &PrimitiveArray<UInt64Type>, is_start: bool) {
        let mut last = *self.indices.last().unwrap();
        if is_start {
            let (is_valid, val) = self.normalize(new_indices.value(0));
            self.indices.push(val);
            self.validity.append(is_valid);
            last += val;
        }
        let mut prev = self.normalize(*new_indices.values().first().unwrap()).1;
        for w in new_indices.values().windows(2) {
            let (is_valid, val) = self.normalize(w[1]);
            let next = val - prev + last;
            self.indices.push(next);
            self.validity.append(is_valid);
            prev = val;
            last = next;
        }
    }

    fn into_parts(mut self) -> (Vec<u64>, BooleanBuffer) {
        (self.indices, self.validity.finish())
    }
}

#[derive(Debug)]
pub struct BinaryPageScheduler {
    indices_scheduler: Arc<dyn PageScheduler>,
    bytes_scheduler: Arc<dyn PageScheduler>,
    offsets_type: DataType,
    null_adjustment: u64,
}

impl BinaryPageScheduler {
    pub fn new(
        indices_scheduler: Arc<dyn PageScheduler>,
        bytes_scheduler: Arc<dyn PageScheduler>,
        offsets_type: DataType,
        null_adjustment: u64,
    ) -> Self {
        Self {
            indices_scheduler,
            bytes_scheduler,
            offsets_type,
            null_adjustment,
        }
    }

    fn decode_indices(decoder: Arc<dyn PrimitivePageDecoder>, num_rows: u64) -> Result<ArrayRef> {
        let mut primitive_wrapper =
            PrimitiveFieldDecoder::new_from_data(decoder, DataType::UInt64, num_rows, false);
        let drained_task = primitive_wrapper.drain(num_rows)?;
        let indices_decode_task = drained_task.task;
        indices_decode_task.decode()
    }
}

struct IndirectData {
    decoded_indices: UInt64Array,
    offsets_type: DataType,
    validity: BooleanBuffer,
    bytes_decoder_fut: BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>>,
}

impl PageScheduler for BinaryPageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u64>],
        scheduler: &Arc<dyn EncodingsIo>,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>> {
        // ranges corresponds to row ranges that the user wants to fetch.
        // if user wants row range a..b
        // Case 1: if a != 0, we need indices a-1..b to decode
        // Case 2: if a = 0, we need indices 0..b to decode
        let indices_ranges = ranges
            .iter()
            .map(|range| {
                if range.start != 0 {
                    (range.start - 1)..range.end
                } else {
                    0..range.end
                }
            })
            .collect::<Vec<std::ops::Range<u64>>>();

        // We schedule all the indices for decoding together
        // This is more efficient compared to scheduling them one by one (reduces speed significantly for random access)
        let indices_page_decoder =
            self.indices_scheduler
                .schedule_ranges(&indices_ranges, scheduler, top_level_row);

        let num_rows = ranges.iter().map(|r| r.end - r.start).sum::<u64>();
        let indices_num_rows = indices_ranges.iter().map(|r| r.end - r.start).sum::<u64>();

        let ranges = ranges.to_vec();
        let copy_scheduler = scheduler.clone();
        let copy_bytes_scheduler = self.bytes_scheduler.clone();
        let null_adjustment = self.null_adjustment;
        let offsets_type = self.offsets_type.clone();

        tokio::spawn(async move {
            // For the following data:
            // "abcd", "hello", "abcd", "apple", "hello", "abcd"
            //   4,        9,     13,      18,      23,     27
            // e.g. want to scan rows 0, 2, 4
            // i.e. offsets are 4 | 9, 13 | 18, 23
            // Normalization is required for decoding later on
            // Normalize each part: 0, 4 | 0, 4 | 0, 5
            // Remove leading zeros except first one: 0, 4 | 4 | 5
            // Cumulative sum: 0, 4 | 8 | 13
            // These are the normalized offsets stored in decoded_indices
            // Rest of the workflow is continued later in BinaryPageDecoder
            let indices_decoder = Arc::from(indices_page_decoder.await?);
            let indices = Self::decode_indices(indices_decoder, indices_num_rows)?;
            let decoded_indices = indices.as_primitive::<UInt64Type>();

            let mut indices_builder = IndicesNormalizer::new(num_rows, null_adjustment);
            let mut bytes_ranges = Vec::new();
            let mut curr_offset_index = 0;

            for curr_row_range in ranges.iter() {
                let row_start = curr_row_range.start;
                let curr_range_len = (curr_row_range.end - row_start) as usize;

                let curr_indices;

                if row_start == 0 {
                    curr_indices = decoded_indices.slice(0, curr_range_len);
                    curr_offset_index = curr_range_len;
                } else {
                    curr_indices = decoded_indices.slice(curr_offset_index, curr_range_len + 1);
                    curr_offset_index += curr_range_len + 1;
                }

                let first = if row_start == 0 {
                    0
                } else {
                    indices_builder
                        .normalize(*curr_indices.values().first().unwrap())
                        .1
                };
                let last = indices_builder
                    .normalize(*curr_indices.values().last().unwrap())
                    .1;
                if first != last {
                    bytes_ranges.push(first..last);
                }

                indices_builder.extend(&curr_indices, row_start == 0);
            }

            let (indices, validity) = indices_builder.into_parts();
            let decoded_indices = UInt64Array::from(indices);

            // In the indirect task we schedule the bytes, but we do not await them.  We don't want to
            // await the bytes until the decoder is ready for them so that we don't release the backpressure
            // too early
            let bytes_decoder_fut =
                copy_bytes_scheduler.schedule_ranges(&bytes_ranges, &copy_scheduler, top_level_row);

            Ok(IndirectData {
                decoded_indices,
                validity,
                offsets_type,
                bytes_decoder_fut,
            })
        })
        // Propagate join panic
        .map(|join_handle| join_handle.unwrap())
        .and_then(|indirect_data| {
            async move {
                // Later, this will be called once the decoder actually starts polling.  At that point
                // we await the bytes (releasing the backpressure)
                let bytes_decoder = indirect_data.bytes_decoder_fut.await?;
                Ok(Box::new(BinaryPageDecoder {
                    decoded_indices: indirect_data.decoded_indices,
                    offsets_type: indirect_data.offsets_type,
                    validity: indirect_data.validity,
                    bytes_decoder,
                }) as Box<dyn PrimitivePageDecoder>)
            }
        })
        .boxed()
    }
}

struct BinaryPageDecoder {
    decoded_indices: UInt64Array,
    offsets_type: DataType,
    validity: BooleanBuffer,
    bytes_decoder: Box<dyn PrimitivePageDecoder>,
}

impl PrimitivePageDecoder for BinaryPageDecoder {
    // Continuing the example from BinaryPageScheduler
    // Suppose batch_size = 2. Then first, rows_to_skip=0, num_rows=2
    // Need to scan 2 rows
    // First row will be 4-0=4 bytes, second also 8-4=4 bytes.
    // Allocate 8 bytes capacity.
    // Next rows_to_skip=2, num_rows=1
    // Skip 8 bytes. Allocate 5 bytes capacity.
    //
    // The normalized offsets are [0, 4, 8, 13]
    // We only need [8, 13] to decode in this case.
    // These need to be normalized in order to build the string later
    // So return [0, 5]
    fn decode(&self, rows_to_skip: u64, num_rows: u64) -> Result<DataBlock> {
        // STEP 1: validity buffer
        let target_validity = self
            .validity
            .slice(rows_to_skip as usize, num_rows as usize);
        let has_nulls = target_validity.count_set_bits() < target_validity.len();

        let validity_buffer = if has_nulls {
            let num_validity_bits = arrow_buffer::bit_util::ceil(num_rows as usize, 8);
            let mut validity_buffer = Vec::with_capacity(num_validity_bits);

            if rows_to_skip == 0 {
                validity_buffer.extend_from_slice(target_validity.inner().as_slice());
            } else {
                // Need to copy the buffer because there may be a bit offset in first byte
                let target_validity = BooleanBuffer::from_iter(target_validity.iter());
                validity_buffer.extend_from_slice(target_validity.inner().as_slice());
            }
            Some(validity_buffer)
        } else {
            None
        };

        // STEP 2: offsets buffer
        // Currently we always do a copy here, we need to cast to the appropriate type
        // and we go ahead and normalize so the starting offset is 0 (though we could skip
        // this)
        let bytes_per_offset = match self.offsets_type {
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            _ => panic!("Unsupported offsets type"),
        };

        let target_offsets = self
            .decoded_indices
            .slice(rows_to_skip as usize, (num_rows + 1) as usize);

        // Normalize and cast (TODO: could fuse these into one pass for micro-optimization)
        let target_vec = target_offsets.values();
        let start = target_vec[0];
        let offsets_buffer =
            match bytes_per_offset {
                4 => ScalarBuffer::from_iter(target_vec.iter().map(|x| (x - start) as i32))
                    .into_inner(),
                8 => ScalarBuffer::from_iter(target_vec.iter().map(|x| (x - start) as i64))
                    .into_inner(),
                _ => panic!("Unsupported offsets type"),
            };

        let bytes_to_skip = self.decoded_indices.value(rows_to_skip as usize);
        let num_bytes = self
            .decoded_indices
            .value((rows_to_skip + num_rows) as usize)
            - bytes_to_skip;

        let bytes = self.bytes_decoder.decode(bytes_to_skip, num_bytes)?;
        let bytes = bytes.as_fixed_width().unwrap();
        debug_assert_eq!(bytes.bits_per_value, 8);

        let string_data = DataBlock::VariableWidth(VariableWidthBlock {
            bits_per_offset: bytes_per_offset * 8,
            data: bytes.data,
            num_values: num_rows,
            offsets: LanceBuffer::from(offsets_buffer),
            block_info: BlockInfo::new(),
            used_encodings: UsedEncoding::new(),
        });
        if let Some(validity) = validity_buffer {
            Ok(DataBlock::Nullable(NullableDataBlock {
                data: Box::new(string_data),
                nulls: LanceBuffer::from(validity),
                block_info: BlockInfo::new(),
                used_encoding: UsedEncoding::new(),
            }))
        } else {
            Ok(string_data)
        }
    }
}

#[derive(Debug)]
pub struct BinaryEncoder {
    indices_encoder: Box<dyn ArrayEncoder>,
    compression_scheme: Option<CompressionScheme>,
    buffer_compressor: Option<Box<dyn BufferCompressor>>,
}

impl BinaryEncoder {
    pub fn new(
        indices_encoder: Box<dyn ArrayEncoder>,
        compression_scheme: Option<CompressionScheme>,
    ) -> Self {
        let buffer_compressor = compression_scheme
            .map(|scheme| GeneralBufferCompressor::get_compressor(&scheme.to_string()));
        Self {
            indices_encoder,
            compression_scheme,
            buffer_compressor,
        }
    }

    // In 2.1 we will materialize nulls higher up (in the primitive encoder).  Unfortunately,
    // in 2.0 we actually need to write the offsets.
    fn all_null_variable_width(data_type: &DataType, num_values: u64) -> VariableWidthBlock {
        if matches!(data_type, DataType::Binary | DataType::Utf8) {
            VariableWidthBlock {
                bits_per_offset: 32,
                data: LanceBuffer::empty(),
                num_values,
                offsets: LanceBuffer::reinterpret_vec(vec![0_u32; num_values as usize + 1]),
                block_info: BlockInfo::new(),
                used_encodings: UsedEncoding::new(),
            }
        } else {
            VariableWidthBlock {
                bits_per_offset: 64,
                data: LanceBuffer::empty(),
                num_values,
                offsets: LanceBuffer::reinterpret_vec(vec![0_u64; num_values as usize + 1]),
                block_info: BlockInfo::new(),
                used_encodings: UsedEncoding::new(),
            }
        }
    }
}

// Creates indices arrays from string arrays
// Strings are a vector of arrays corresponding to each record batch
// Zero offset is removed from the start of the offsets array
// The indices array is computed across all arrays in the vector
fn get_indices_from_string_arrays(
    mut offsets: LanceBuffer,
    bits_per_offset: u8,
    nulls: Option<LanceBuffer>,
    num_rows: usize,
) -> (DataBlock, u64) {
    let mut indices = Vec::with_capacity(num_rows);
    let mut last_offset = 0_u64;
    if bits_per_offset == 32 {
        let offsets = offsets.borrow_to_typed_slice::<i32>();
        indices.extend(offsets.as_ref().windows(2).map(|w| {
            let strlen = (w[1] - w[0]) as u64;
            last_offset += strlen;
            last_offset
        }));
    } else if bits_per_offset == 64 {
        let offsets = offsets.borrow_to_typed_slice::<i64>();
        indices.extend(offsets.as_ref().windows(2).map(|w| {
            let strlen = (w[1] - w[0]) as u64;
            last_offset += strlen;
            last_offset
        }));
    }

    if indices.is_empty() {
        return (
            DataBlock::FixedWidth(FixedWidthDataBlock {
                bits_per_value: 64,
                data: LanceBuffer::empty(),
                num_values: 0,
                block_info: BlockInfo::new(),
                used_encoding: UsedEncoding::new(),
            }),
            0,
        );
    }

    let last_offset = *indices.last().expect("Indices array is empty");
    // 8 exabytes in a single array seems unlikely but...just in case
    assert!(
        last_offset < u64::MAX / 2,
        "Indices array with strings up to 2^63 is too large for this encoding"
    );
    let null_adjustment: u64 = *indices.last().expect("Indices array is empty") + 1;

    if let Some(nulls) = nulls {
        let nulls = NullBuffer::new(BooleanBuffer::new(nulls.into_buffer(), 0, num_rows));
        indices
            .iter_mut()
            .zip(nulls.iter())
            .for_each(|(index, is_valid)| {
                if !is_valid {
                    *index += null_adjustment;
                }
            });
    }
    let indices = DataBlock::FixedWidth(FixedWidthDataBlock {
        bits_per_value: 64,
        data: LanceBuffer::reinterpret_vec(indices),
        num_values: num_rows as u64,
        block_info: BlockInfo::new(),
        used_encoding: UsedEncoding::new(),
    });
    (indices, null_adjustment)
}

impl ArrayEncoder for BinaryEncoder {
    fn encode(
        &self,
        data: DataBlock,
        data_type: &DataType,
        buffer_index: &mut u32,
    ) -> Result<EncodedArray> {
        let (mut data, nulls) = match data {
            DataBlock::Nullable(nullable) => {
                let data = nullable.data.as_variable_width().unwrap();
                (data, Some(nullable.nulls))
            }
            DataBlock::VariableWidth(variable) => (variable, None),
            DataBlock::AllNull(all_null) => {
                let data = Self::all_null_variable_width(data_type, all_null.num_values);
                let validity =
                    LanceBuffer::all_unset(bit_util::ceil(all_null.num_values as usize, 8));
                (data, Some(validity))
            }
            _ => panic!("Expected variable width data block but got {}", data.name()),
        };

        let (indices, null_adjustment) = get_indices_from_string_arrays(
            data.offsets,
            data.bits_per_offset,
            nulls,
            data.num_values as usize,
        );
        let encoded_indices =
            self.indices_encoder
                .encode(indices, &DataType::UInt64, buffer_index)?;

        let encoded_indices_data = encoded_indices.data.as_fixed_width().unwrap();

        assert!(encoded_indices_data.bits_per_value <= 64);

        if let Some(buffer_compressor) = &self.buffer_compressor {
            let mut compressed_data = Vec::with_capacity(data.data.len());
            buffer_compressor.compress(&data.data, &mut compressed_data)?;
            data.data = LanceBuffer::Owned(compressed_data);
        }

        let data = DataBlock::VariableWidth(VariableWidthBlock {
            bits_per_offset: encoded_indices_data.bits_per_value as u8,
            offsets: encoded_indices_data.data,
            data: data.data,
            num_values: data.num_values,
            block_info: BlockInfo::new(),
            used_encodings: UsedEncoding::new(),
        });

        let bytes_buffer_index = *buffer_index;
        *buffer_index += 1;

        let bytes_encoding = ProtobufUtils::flat_encoding(
            /*bits_per_value=*/ 8,
            bytes_buffer_index,
            self.compression_scheme,
        );

        let encoding =
            ProtobufUtils::binary(encoded_indices.encoding, bytes_encoding, null_adjustment);

        Ok(EncodedArray { data, encoding })
    }
}

#[cfg(test)]
pub mod tests {

    use arrow_array::{
        builder::{LargeStringBuilder, StringBuilder},
        ArrayRef, StringArray,
    };
    use arrow_schema::{DataType, Field};
    use std::{collections::HashMap, sync::Arc, vec};

    use crate::{
        buffer::LanceBuffer,
        data::DataBlock,
        testing::{check_round_trip_encoding_of_data, check_round_trip_encoding_random, TestCases},
        version::LanceFileVersion,
    };

    use super::get_indices_from_string_arrays;

    #[test_log::test(tokio::test)]
    async fn test_utf8_binary() {
        let field = Field::new("", DataType::Utf8, false);
        check_round_trip_encoding_random(field, LanceFileVersion::V2_0).await;
    }

    #[test]
    fn test_encode_indices_adjusts_nulls() {
        // Null entries in string arrays should be adjusted
        let string_array = Arc::new(StringArray::from(vec![
            None,
            Some("foo"),
            Some("foo"),
            None,
            None,
            None,
        ])) as ArrayRef;
        let string_data = DataBlock::from(string_array).as_nullable().unwrap();
        let nulls = string_data.nulls;
        let string_data = string_data.data.as_variable_width().unwrap();

        let (indices, null_adjustment) = get_indices_from_string_arrays(
            string_data.offsets,
            string_data.bits_per_offset,
            Some(nulls),
            string_data.num_values as usize,
        );

        let indices = indices.as_fixed_width().unwrap();
        assert_eq!(indices.bits_per_value, 64);
        assert_eq!(
            indices.data,
            LanceBuffer::reinterpret_vec(vec![7_u64, 3, 6, 13, 13, 13])
        );
        assert_eq!(null_adjustment, 7);
    }

    #[test_log::test(tokio::test)]
    async fn test_binary() {
        let field = Field::new("", DataType::Binary, false);
        check_round_trip_encoding_random(field, LanceFileVersion::V2_0).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_large_binary() {
        let field = Field::new("", DataType::LargeBinary, true);
        check_round_trip_encoding_random(field, LanceFileVersion::V2_0).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_large_utf8() {
        let field = Field::new("", DataType::LargeUtf8, true);
        check_round_trip_encoding_random(field, LanceFileVersion::V2_0).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_simple_utf8_binary() {
        let string_array = StringArray::from(vec![Some("abc"), None, Some("pqr"), None, Some("m")]);

        let test_cases = TestCases::default()
            .with_range(0..2)
            .with_range(0..3)
            .with_range(1..3)
            .with_indices(vec![0, 1, 3, 4]);
        check_round_trip_encoding_of_data(
            vec![Arc::new(string_array)],
            &test_cases,
            HashMap::new(),
        )
        .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_sliced_utf8() {
        let string_array = StringArray::from(vec![Some("abc"), Some("de"), None, Some("fgh")]);
        let string_array = string_array.slice(1, 3);

        let test_cases = TestCases::default()
            .with_range(0..1)
            .with_range(0..2)
            .with_range(1..2);
        check_round_trip_encoding_of_data(
            vec![Arc::new(string_array)],
            &test_cases,
            HashMap::new(),
        )
        .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_bigger_than_max_page_size() {
        // Create an array with one single 32MiB string
        let big_string = String::from_iter((0..(32 * 1024 * 1024)).map(|_| '0'));
        let string_array = StringArray::from(vec![
            Some(big_string),
            Some("abc".to_string()),
            None,
            None,
            Some("xyz".to_string()),
        ]);

        // Drop the max page size to 1MiB
        let test_cases = TestCases::default().with_max_page_size(1024 * 1024);

        check_round_trip_encoding_of_data(
            vec![Arc::new(string_array)],
            &test_cases,
            HashMap::new(),
        )
        .await;

        // This is a regression testing the case where a page with X rows is split into Y parts
        // where the number of parts is not evenly divisible by the number of rows.  In this
        // case we are splitting 90 rows into 4 parts.
        let big_string = String::from_iter((0..(1000 * 1000)).map(|_| '0'));
        let string_array = StringArray::from_iter_values((0..90).map(|_| big_string.clone()));

        check_round_trip_encoding_of_data(
            vec![Arc::new(string_array)],
            &TestCases::default(),
            HashMap::new(),
        )
        .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_empty_strings() {
        // Scenario 1: Some strings are empty

        let values = [Some("abc"), Some(""), None];
        // Test empty list at beginning, middle, and end
        for order in [[0, 1, 2], [1, 0, 2], [2, 0, 1]] {
            let mut string_builder = StringBuilder::new();
            for idx in order {
                string_builder.append_option(values[idx]);
            }
            let string_array = Arc::new(string_builder.finish());
            let test_cases = TestCases::default()
                .with_indices(vec![1])
                .with_indices(vec![0])
                .with_indices(vec![2])
                .with_indices(vec![0, 1]);
            check_round_trip_encoding_of_data(
                vec![string_array.clone()],
                &test_cases,
                HashMap::new(),
            )
            .await;
            let test_cases = test_cases.with_batch_size(1);
            check_round_trip_encoding_of_data(vec![string_array], &test_cases, HashMap::new())
                .await;
        }

        // Scenario 2: All strings are empty

        // When encoding an array of empty strings there are no bytes to encode
        // which is strange and we want to ensure we handle it
        let string_array = Arc::new(StringArray::from(vec![Some(""), None, Some("")]));

        let test_cases = TestCases::default().with_range(0..2).with_indices(vec![1]);
        check_round_trip_encoding_of_data(vec![string_array.clone()], &test_cases, HashMap::new())
            .await;
        let test_cases = test_cases.with_batch_size(1);
        check_round_trip_encoding_of_data(vec![string_array], &test_cases, HashMap::new()).await;
    }

    #[test_log::test(tokio::test)]
    #[ignore] // This test is quite slow in debug mode
    async fn test_jumbo_string() {
        // This is an overflow test.  We have a list of lists where each list
        // has 1Mi items.  We encode 5000 of these lists and so we have over 4Gi in the
        // offsets range
        let mut string_builder = LargeStringBuilder::new();
        // a 1 MiB string
        let giant_string = String::from_iter((0..(1024 * 1024)).map(|_| '0'));
        for _ in 0..5000 {
            string_builder.append_option(Some(&giant_string));
        }
        let giant_array = Arc::new(string_builder.finish()) as ArrayRef;
        let arrs = vec![giant_array];

        // // We can't validate because our validation relies on concatenating all input arrays
        let test_cases = TestCases::default().without_validation();
        check_round_trip_encoding_of_data(arrs, &test_cases, HashMap::new()).await;
    }
}
