// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use core::panic;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, ArrayRef};
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder, ScalarBuffer};
use bytes::BytesMut;
use futures::stream::StreamExt;
use futures::{future::BoxFuture, stream::FuturesOrdered, FutureExt};

use crate::{
    decoder::{PageScheduler, PrimitivePageDecoder},
    encoder::{ArrayEncoder, EncodedArray},
    format::pb,
    EncodingsIo,
};

use crate::decoder::LogicalPageDecoder;
use crate::encodings::logical::primitive::PrimitiveFieldDecoder;

use arrow_array::{PrimitiveArray, UInt64Array, UInt8Array};
use arrow_schema::DataType;
use lance_core::Result;

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
}

impl BinaryPageScheduler {
    fn decode_indices(decoder: Arc<dyn PrimitivePageDecoder>, num_rows: u64) -> Result<ArrayRef> {
        let mut primitive_wrapper =
            PrimitiveFieldDecoder::new_from_data(decoder, DataType::UInt64, num_rows);
        let drained_task = primitive_wrapper.drain(num_rows)?;
        let indices_decode_task = drained_task.task;
        indices_decode_task.decode()
    }
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
                    (range.start - 1)..(range.end)
                } else {
                    0..(range.end)
                }
            })
            .collect::<Vec<std::ops::Range<u64>>>();

        let num_rows = ranges.iter().map(|r| r.end - r.start).sum::<u64>();

        let mut futures_ordered = indices_ranges
            .iter()
            .map(|range| {
                self.indices_scheduler
                    .schedule_ranges(&[range.clone()], scheduler, top_level_row)
            })
            .collect::<FuturesOrdered<_>>();

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

            let mut indices_builder = IndicesNormalizer::new(num_rows, null_adjustment);
            let mut bytes_ranges = Vec::new();
            let mut curr_range_idx = 0;
            while let Some(indices_page_decoder) = futures_ordered.next().await {
                let decoder = Arc::from(indices_page_decoder?);

                // Build and run decode task for offsets
                let curr_indices_range = indices_ranges[curr_range_idx].clone();
                let curr_row_range = ranges[curr_range_idx].clone();
                let indices_num_rows = curr_indices_range.end - curr_indices_range.start;

                let indices = Self::decode_indices(decoder, indices_num_rows)?;
                let indices = indices.as_primitive::<UInt64Type>();

                let first = if curr_row_range.start == 0 {
                    0
                } else {
                    indices_builder
                        .normalize(*indices.values().first().unwrap())
                        .1
                };
                let last = indices_builder
                    .normalize(*indices.values().last().unwrap())
                    .1;
                bytes_ranges.push(first..last);

                indices_builder.extend(indices, curr_row_range.start == 0);

                curr_range_idx += 1;
            }

            let (indices, validity) = indices_builder.into_parts();
            let decoded_indices = UInt64Array::from(indices);

            // Schedule the bytes for decoding
            let bytes_page_decoder =
                copy_bytes_scheduler.schedule_ranges(&bytes_ranges, &copy_scheduler, top_level_row);

            let bytes_decoder: Box<dyn PrimitivePageDecoder> = bytes_page_decoder.await?;

            Ok(Box::new(BinaryPageDecoder {
                decoded_indices,
                validity,
                offsets_type,
                bytes_decoder,
            }) as Box<dyn PrimitivePageDecoder>)
        })
        // Propagate join panic
        .map(|join_handle| join_handle.unwrap())
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
    fn decode(
        &self,
        rows_to_skip: u64,
        num_rows: u64,
        all_null: &mut bool,
    ) -> Result<Vec<BytesMut>> {
        // Buffers[0] == validity buffer
        // Buffers[1] == offsets buffer
        // Buffers[2] == null buffer // TODO: Micro-optimization, can we get rid of this?  Doesn't hurt much though
        //                              This buffer is always empty since bytes are not allowed to contain nulls
        // Buffers[3] == bytes buffer

        // STEP 1: validity buffer
        let target_validity = self
            .validity
            .slice(rows_to_skip as usize, num_rows as usize);
        let has_nulls = target_validity.count_set_bits() < target_validity.len();

        let validity_buffer = if has_nulls {
            let num_validity_bits = arrow_buffer::bit_util::ceil(num_rows as usize, 8);
            let mut validity_buffer = BytesMut::with_capacity(num_validity_bits);

            if rows_to_skip == 0 {
                validity_buffer.extend_from_slice(target_validity.inner().as_slice());
            } else {
                // Need to copy the buffer because there may be a bit offset in first byte
                let target_validity = BooleanBuffer::from_iter(target_validity.iter());
                validity_buffer.extend_from_slice(target_validity.inner().as_slice());
            }
            validity_buffer
        } else {
            BytesMut::new()
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
        // TODO: This forces a second copy, which is unfortunate, try and remove in the future
        let offsets_buf = BytesMut::from(offsets_buffer.as_slice());

        let bytes_to_skip = self.decoded_indices.value(rows_to_skip as usize);
        let num_bytes = self
            .decoded_indices
            .value((rows_to_skip + num_rows) as usize)
            - bytes_to_skip;

        let mut output_buffers = vec![validity_buffer, offsets_buf];

        // Copy decoded bytes into dest_buffers[2..]
        // Currently an empty null buffer is the first one
        // The actual bytes are in the second buffer
        // Including the indices this results in 4 buffers in total
        output_buffers.extend(
            self.bytes_decoder
                .decode(bytes_to_skip, num_bytes, all_null)?,
        );

        Ok(output_buffers)
    }

    fn num_buffers(&self) -> u32 {
        self.bytes_decoder.num_buffers() + 2
    }
}

#[derive(Debug)]
pub struct BinaryEncoder {
    indices_encoder: Box<dyn ArrayEncoder>,
    bytes_encoder: Box<dyn ArrayEncoder>,
}

impl BinaryEncoder {
    pub fn new(
        indices_encoder: Box<dyn ArrayEncoder>,
        bytes_encoder: Box<dyn ArrayEncoder>,
    ) -> Self {
        Self {
            indices_encoder,
            bytes_encoder,
        }
    }
}

// Creates indices arrays from string arrays
// Strings are a vector of arrays corresponding to each record batch
// Zero offset is removed from the start of the offsets array
// The indices array is computed across all arrays in the vector
fn get_indices_from_string_arrays(arrays: &[ArrayRef]) -> (ArrayRef, u64) {
    let num_rows = arrays.iter().map(|arr| arr.len()).sum::<usize>();
    let mut indices = Vec::with_capacity(num_rows);
    let mut last_offset = 0_u64;
    for array in arrays {
        if let Some(array) = array.as_string_opt::<i32>() {
            let offsets = array.offsets().inner();
            indices.extend(offsets.windows(2).map(|w| {
                let strlen = (w[1] - w[0]) as u64;
                let off = strlen + last_offset;
                last_offset = off;
                off
            }));
        } else if let Some(array) = array.as_string_opt::<i64>() {
            let offsets = array.offsets().inner();
            indices.extend(offsets.windows(2).map(|w| {
                let strlen = (w[1] - w[0]) as u64;
                let off = strlen + last_offset;
                last_offset = off;
                off
            }));
        } else if let Some(array) = array.as_binary_opt::<i32>() {
            let offsets = array.offsets().inner();
            indices.extend(offsets.windows(2).map(|w| {
                let strlen = (w[1] - w[0]) as u64;
                let off = strlen + last_offset;
                last_offset = off;
                off
            }));
        } else if let Some(array) = array.as_binary_opt::<i64>() {
            let offsets = array.offsets().inner();
            indices.extend(offsets.windows(2).map(|w| {
                let strlen = (w[1] - w[0]) as u64;
                let off = strlen + last_offset;
                last_offset = off;
                off
            }));
        } else {
            panic!("Array is not a string array");
        }
    }
    let last_offset = *indices.last().expect("Indices array is empty");
    // 8 exabytes in a single array seems unlikely but...just in case
    assert!(
        last_offset < u64::MAX / 2,
        "Indices array with strings up to 2^63 is too large for this encoding"
    );
    let null_adjustment: u64 = *indices.last().expect("Indices array is empty") + 1;

    let mut indices_offset = 0;
    for array in arrays {
        if let Some(nulls) = array.nulls() {
            let indices_slice = &mut indices[indices_offset..indices_offset + array.len()];
            indices_slice
                .iter_mut()
                .zip(nulls.iter())
                .for_each(|(index, is_valid)| {
                    if !is_valid {
                        *index += null_adjustment;
                    }
                });
        }
        indices_offset += array.len();
    }

    (Arc::new(UInt64Array::from(indices)), null_adjustment)
}

// Bytes computed across all string arrays, similar to indices above
fn get_bytes_from_string_arrays(arrays: &[ArrayRef]) -> Vec<ArrayRef> {
    arrays
        .iter()
        .map(|arr| {
            let (values_buffer, start, stop) = if let Some(arr) = arr.as_string_opt::<i32>() {
                (
                    arr.values(),
                    arr.offsets()[0] as usize,
                    arr.offsets()[arr.len()] as usize,
                )
            } else if let Some(arr) = arr.as_string_opt::<i64>() {
                (
                    arr.values(),
                    arr.offsets()[0] as usize,
                    arr.offsets()[arr.len()] as usize,
                )
            } else if let Some(arr) = arr.as_binary_opt::<i32>() {
                (
                    arr.values(),
                    arr.offsets()[0] as usize,
                    arr.offsets()[arr.len()] as usize,
                )
            } else if let Some(arr) = arr.as_binary_opt::<i64>() {
                (
                    arr.values(),
                    arr.offsets()[0] as usize,
                    arr.offsets()[arr.len()] as usize,
                )
            } else {
                panic!("Array is not a string / binary array");
            };
            let values = ScalarBuffer::new(values_buffer.clone(), start, stop - start);
            Arc::new(UInt8Array::new(values, None)) as ArrayRef
        })
        .collect()
}

impl ArrayEncoder for BinaryEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: &mut u32) -> Result<EncodedArray> {
        let (index_array, null_adjustment) = get_indices_from_string_arrays(arrays);
        let encoded_indices = self.indices_encoder.encode(&[index_array], buffer_index)?;

        let byte_arrays = get_bytes_from_string_arrays(arrays);
        let encoded_bytes = self.bytes_encoder.encode(&byte_arrays, buffer_index)?;

        let mut encoded_buffers = encoded_indices.buffers;
        encoded_buffers.extend(encoded_bytes.buffers);

        Ok(EncodedArray {
            buffers: encoded_buffers,
            encoding: pb::ArrayEncoding {
                array_encoding: Some(pb::array_encoding::ArrayEncoding::Binary(Box::new(
                    pb::Binary {
                        indices: Some(Box::new(encoded_indices.encoding)),
                        bytes: Some(Box::new(encoded_bytes.encoding)),
                        null_adjustment,
                    },
                ))),
            },
        })
    }
}

#[cfg(test)]
pub mod tests {

    use arrow_array::{
        builder::{LargeStringBuilder, StringBuilder},
        ArrayRef, LargeStringArray, StringArray, UInt64Array,
    };
    use arrow_schema::{DataType, Field};
    use std::{sync::Arc, vec};

    use crate::testing::{
        check_round_trip_encoding_of_data, check_round_trip_encoding_random, TestCases,
    };

    use super::get_indices_from_string_arrays;

    #[test_log::test(tokio::test)]
    async fn test_utf8() {
        let field = Field::new("", DataType::Utf8, false);
        check_round_trip_encoding_random(field).await;
    }

    #[test]
    fn test_encode_indices_stitches_offsets() {
        // Given two string arrays we might have offsets [5, 10, 15] and [0, 3, 7]
        //
        // We need to stitch them to [0, 5, 10, 13, 17]
        let string_array1 = StringArray::from(vec![Some("abcde"), Some("abcde"), Some("abcde")]);
        let string_array1 = Arc::new(string_array1.slice(1, 2));
        let string_array2 = Arc::new(StringArray::from(vec![Some("abc"), Some("abcd")]));
        let (offsets, null_adjustment) =
            get_indices_from_string_arrays(&[string_array1, string_array2]);

        let expected = Arc::new(UInt64Array::from(vec![5, 10, 13, 17])) as ArrayRef;
        assert_eq!(&offsets, &expected);
        assert_eq!(null_adjustment, 18);
    }

    #[test]
    fn test_encode_indices_adjusts_nulls() {
        // Null entries in string arrays should be adjusted
        let string_array1 = Arc::new(StringArray::from(vec![None, Some("foo")]));
        let string_array2 = Arc::new(StringArray::from(vec![Some("foo"), None]));
        let string_array3 = Arc::new(StringArray::from(vec![None as Option<&str>, None]));
        let (offsets, null_adjustment) =
            get_indices_from_string_arrays(&[string_array1, string_array2, string_array3]);

        let expected = Arc::new(UInt64Array::from(vec![7, 3, 6, 13, 13, 13])) as ArrayRef;
        assert_eq!(&offsets, &expected);
        assert_eq!(null_adjustment, 7);
    }

    #[test]
    fn test_encode_indices_string_types() {
        let string_array = Arc::new(LargeStringArray::from(vec![Some("foo")]));
        let large_string_array = Arc::new(LargeStringArray::from(vec![Some("foo")]));
        let binary_array = Arc::new(LargeStringArray::from(vec![Some("foo")]));
        let large_binary_array = Arc::new(LargeStringArray::from(vec![Some("foo")]));

        for arr in [
            string_array,
            large_string_array,
            binary_array,
            large_binary_array,
        ] {
            let (offsets, null_adjustment) = get_indices_from_string_arrays(&[arr]);
            let expected = Arc::new(UInt64Array::from(vec![3])) as ArrayRef;
            assert_eq!(&offsets, &expected);
            assert_eq!(null_adjustment, 4);
        }
    }

    #[test_log::test(tokio::test)]
    async fn test_binary() {
        let field = Field::new("", DataType::Binary, false);
        check_round_trip_encoding_random(field).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_large_binary() {
        let field = Field::new("", DataType::LargeBinary, true);
        check_round_trip_encoding_random(field).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_large_utf8() {
        let field = Field::new("", DataType::LargeUtf8, true);
        check_round_trip_encoding_random(field).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_simple_utf8() {
        let string_array = StringArray::from(vec![Some("abc"), Some("de"), None, Some("fgh")]);

        let test_cases = TestCases::default()
            .with_range(0..2)
            .with_range(0..3)
            .with_range(1..3)
            .with_indices(vec![1, 3]);
        check_round_trip_encoding_of_data(vec![Arc::new(string_array)], &test_cases).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_sliced_utf8() {
        let string_array = StringArray::from(vec![Some("abc"), Some("de"), None, Some("fgh")]);
        let string_array = string_array.slice(1, 3);

        let test_cases = TestCases::default()
            .with_range(0..1)
            .with_range(0..2)
            .with_range(1..2);
        check_round_trip_encoding_of_data(vec![Arc::new(string_array)], &test_cases).await;
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
                .with_indices(vec![2]);
            check_round_trip_encoding_of_data(vec![string_array.clone()], &test_cases).await;
            let test_cases = test_cases.with_batch_size(1);
            check_round_trip_encoding_of_data(vec![string_array], &test_cases).await;
        }

        // Scenario 2: All strings are empty

        // When encoding an array of empty strings there are no bytes to encode
        // which is strange and we want to ensure we handle it
        let string_array = Arc::new(StringArray::from(vec![Some(""), None, Some("")]));

        let test_cases = TestCases::default().with_range(0..2).with_indices(vec![1]);
        check_round_trip_encoding_of_data(vec![string_array.clone()], &test_cases).await;
        let test_cases = test_cases.with_batch_size(1);
        check_round_trip_encoding_of_data(vec![string_array], &test_cases).await;
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
        check_round_trip_encoding_of_data(arrs, &test_cases).await;
    }
}
