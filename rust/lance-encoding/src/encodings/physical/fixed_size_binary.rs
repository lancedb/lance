// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{cast::AsArray, Array, ArrayRef, UInt8Array};
use arrow_buffer::{Buffer, ScalarBuffer};
use futures::{future::BoxFuture, FutureExt};
use lance_core::Result;

use crate::{
    buffer::LanceBuffer,
    data::{DataBlock, DataBlockExt, FixedWidthDataBlock, VariableWidthBlock},
    decoder::{PageScheduler, PrimitivePageDecoder},
    encoder::{ArrayEncoder, EncodedArray},
    format::pb,
    EncodingsIo,
};

/// A scheduler for fixed size lists of primitive values
///
/// This scheduler is, itself, primitive
#[derive(Debug)]
pub struct FixedSizeBinaryPageScheduler {
    bytes_scheduler: Box<dyn PageScheduler>,
    byte_width: u32,
    bytes_per_offset: u32,
}

impl FixedSizeBinaryPageScheduler {
    pub fn new(
        bytes_scheduler: Box<dyn PageScheduler>,
        byte_width: u32,
        bytes_per_offset: u32,
    ) -> Self {
        Self {
            bytes_scheduler,
            byte_width,
            bytes_per_offset,
        }
    }
}

impl PageScheduler for FixedSizeBinaryPageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u64>],
        scheduler: &Arc<dyn EncodingsIo>,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>> {
        let expanded_ranges = ranges
            .iter()
            .map(|range| {
                (range.start * self.byte_width as u64)..(range.end * self.byte_width as u64)
            })
            .collect::<Vec<_>>();

        let bytes_page_decoder =
            self.bytes_scheduler
                .schedule_ranges(&expanded_ranges, scheduler, top_level_row);

        let byte_width = self.byte_width;
        let bytes_per_offset = self.bytes_per_offset;

        async move {
            let bytes_decoder = bytes_page_decoder.await?;
            Ok(Box::new(FixedSizeBinaryDecoder {
                bytes_decoder,
                byte_width: byte_width as u64,
                bytes_per_offset,
            }) as Box<dyn PrimitivePageDecoder>)
        }
        .boxed()
    }
}

pub struct FixedSizeBinaryDecoder {
    bytes_decoder: Box<dyn PrimitivePageDecoder>,
    byte_width: u64,
    bytes_per_offset: u32,
}

impl PrimitivePageDecoder for FixedSizeBinaryDecoder {
    fn decode(&self, rows_to_skip: u64, num_rows: u64) -> Result<Box<dyn DataBlock>> {
        let rows_to_skip = rows_to_skip * self.byte_width;
        let num_bytes = num_rows * self.byte_width;
        let bytes = self.bytes_decoder.decode(rows_to_skip, num_bytes)?;
        let bytes = bytes.try_into_layout::<FixedWidthDataBlock>()?;
        debug_assert_eq!(bytes.bits_per_value, 8);

        let offsets_buffer = if self.bytes_per_offset == 8 {
            let offsets_vec = (0..(num_rows + 1))
                .map(|i| i * self.byte_width)
                .collect::<Vec<_>>();
            Buffer::from_slice_ref(offsets_vec)
        } else {
            let offsets_vec = (0..(num_rows as u32 + 1))
                .map(|i| i * self.byte_width as u32)
                .collect::<Vec<_>>();
            Buffer::from_slice_ref(offsets_vec)
        };

        let string_data = Box::new(VariableWidthBlock {
            bits_per_offset: (self.bytes_per_offset * 8) as u8,
            data: bytes.data,
            num_values: num_rows,
            offsets: LanceBuffer::from(offsets_buffer),
        });

        Ok(string_data)
    }
}

#[derive(Debug)]
pub struct FixedSizeBinaryEncoder {
    bytes_encoder: Box<dyn ArrayEncoder>,
}

impl FixedSizeBinaryEncoder {
    pub fn new(bytes_encoder: Box<dyn ArrayEncoder>) -> Self {
        Self { bytes_encoder }
    }
}

pub fn get_offset_info_from_binary_arrays(arrays: &[ArrayRef]) -> usize {
    // TODO: add case here for when the first array in arrays is empty (or has only null values)
    let arr = &arrays[0];

    if let Some(arr) = arr.as_string_opt::<i32>() {
        (arr.offsets()[1] - arr.offsets()[0]) as usize
    } else if let Some(arr) = arr.as_string_opt::<i64>() {
        (arr.offsets()[1] - arr.offsets()[0]) as usize
    } else if let Some(arr) = arr.as_binary_opt::<i32>() {
        (arr.offsets()[1] - arr.offsets()[0]) as usize
    } else if let Some(arr) = arr.as_binary_opt::<i64>() {
        (arr.offsets()[1] - arr.offsets()[0]) as usize
    } else {
        panic!("Array is not a string / binary array");
    }
}

pub fn get_bytes_from_binary_arrays(arrays: &[ArrayRef], byte_width: usize) -> Vec<ArrayRef> {
    arrays
        .iter()
        .map(|arr| {
            let values_buffer = if let Some(arr) = arr.as_string_opt::<i32>() {
                let mut values_vec = Vec::with_capacity(arr.len() * byte_width);
                for i in 0..arr.len() {
                    let start = arr.offsets()[i] as usize;
                    let end = arr.offsets()[i + 1] as usize;
                    if start == end {
                        // Null value, add byte_width bytes
                        values_vec.extend(std::iter::repeat(0).take(byte_width))
                    } else {
                        values_vec.extend_from_slice(&arr.values()[start..end]);
                    }
                }

                values_vec
            } else if let Some(arr) = arr.as_string_opt::<i64>() {
                let mut values_vec = Vec::with_capacity(arr.len() * byte_width);
                for i in 0..arr.len() {
                    let start = arr.offsets()[i] as usize;
                    let end = arr.offsets()[i + 1] as usize;
                    if start == end {
                        // Null value, add byte_width bytes
                        values_vec.extend(std::iter::repeat(0).take(byte_width))
                    } else {
                        values_vec.extend_from_slice(&arr.values()[start..end]);
                    }
                }

                values_vec
            } else if let Some(arr) = arr.as_binary_opt::<i32>() {
                let mut values_vec = Vec::with_capacity(arr.len() * byte_width);
                for i in 0..arr.len() {
                    let start = arr.offsets()[i] as usize;
                    let end = arr.offsets()[i + 1] as usize;
                    if start == end {
                        // Null value, add byte_width bytes
                        values_vec.extend(std::iter::repeat(0).take(byte_width))
                    } else {
                        values_vec.extend_from_slice(&arr.values()[start..end]);
                    }
                }

                values_vec
            } else if let Some(arr) = arr.as_binary_opt::<i64>() {
                let mut values_vec = Vec::with_capacity(arr.len() * byte_width);
                for i in 0..arr.len() {
                    let start = arr.offsets()[i] as usize;
                    let end = arr.offsets()[i + 1] as usize;
                    if start == end {
                        // Null value, add byte_width bytes
                        values_vec.extend(std::iter::repeat(0).take(byte_width))
                    } else {
                        values_vec.extend_from_slice(&arr.values()[start..end]);
                    }
                }

                values_vec
            } else {
                panic!("Array is not a string / binary array");
            };

            let len_buffer = values_buffer.len();
            let values = ScalarBuffer::new(Buffer::from(values_buffer), 0, len_buffer);
            Arc::new(UInt8Array::new(values, None)) as ArrayRef
        })
        .collect()
}

impl ArrayEncoder for FixedSizeBinaryEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: &mut u32) -> Result<EncodedArray> {
        let byte_width = get_offset_info_from_binary_arrays(arrays);
        let byte_arrays = get_bytes_from_binary_arrays(arrays, byte_width);
        let encoded_bytes = self.bytes_encoder.encode(&byte_arrays, buffer_index)?;

        Ok(EncodedArray {
            buffers: encoded_bytes.buffers,
            encoding: pb::ArrayEncoding {
                array_encoding: Some(pb::array_encoding::ArrayEncoding::FixedSizeBinary(
                    Box::new(pb::FixedSizeBinary {
                        bytes: Some(Box::new(encoded_bytes.encoding)),
                        byte_width: byte_width as u32,
                    }),
                )),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow::array::LargeStringBuilder;
    use arrow_array::{ArrayRef, LargeStringArray, StringArray};
    use arrow_schema::{DataType, Field};

    use crate::testing::{
        check_round_trip_encoding_of_data, check_round_trip_encoding_random, TestCases,
    };

    #[test_log::test(tokio::test)]
    async fn test_fixed_size_utf8_binary() {
        let field = Field::new("", DataType::Utf8, false);
        // This test only generates fixed size binary arrays anyway
        check_round_trip_encoding_random(field, HashMap::new()).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_fixed_size_binary() {
        let field = Field::new("", DataType::Binary, false);
        check_round_trip_encoding_random(field, HashMap::new()).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_fixed_size_large_binary() {
        let field = Field::new("", DataType::LargeBinary, true);
        check_round_trip_encoding_random(field, HashMap::new()).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_fixed_size_large_utf8() {
        let field = Field::new("", DataType::LargeUtf8, true);
        check_round_trip_encoding_random(field, HashMap::new()).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_simple_fixed_size_utf8() {
        let string_array = StringArray::from(vec![
            Some("abc"),
            Some("def"),
            Some("ghi"),
            Some("jkl"),
            Some("mno"),
        ]);

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
    async fn test_simple_fixed_size_with_nulls_utf8() {
        let string_array =
            LargeStringArray::from(vec![Some("abc"), None, Some("ghi"), None, Some("mno")]);

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
    async fn test_fixed_size_sliced_utf8() {
        let string_array = StringArray::from(vec![Some("abc"), Some("def"), None, Some("fgh")]);
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
    async fn test_fixed_size_empty_strings() {
        // All strings are empty

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
