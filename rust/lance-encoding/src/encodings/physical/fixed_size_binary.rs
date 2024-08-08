// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{cast::AsArray, Array, ArrayRef, UInt64Array, UInt8Array};
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
}

impl FixedSizeBinaryPageScheduler {
    pub fn new(bytes_scheduler: Box<dyn PageScheduler>, byte_width: u32) -> Self {
        Self {
            bytes_scheduler,
            byte_width,
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
        async move {
            let bytes_decoder = bytes_page_decoder.await?;
            Ok(Box::new(FixedSizeBinaryDecoder {
                bytes_decoder,
                byte_width: byte_width as u64,
            }) as Box<dyn PrimitivePageDecoder>)
        }
        .boxed()
    }
}

pub struct FixedSizeBinaryDecoder {
    bytes_decoder: Box<dyn PrimitivePageDecoder>,
    byte_width: u64,
}

impl PrimitivePageDecoder for FixedSizeBinaryDecoder {
    fn decode(&self, rows_to_skip: u64, num_rows: u64) -> Result<Box<dyn DataBlock>> {
        let rows_to_skip = rows_to_skip * self.byte_width;
        let num_bytes = num_rows * self.byte_width;
        let bytes = self.bytes_decoder.decode(rows_to_skip, num_bytes)?;
        let bytes = bytes.try_into_layout::<FixedWidthDataBlock>()?;
        debug_assert_eq!(bytes.bits_per_value, 8);
        println!("Bytes data: {:?}", bytes.data);
        println!("Bits per value: {:?}", bytes.bits_per_value);

        let offsets_vec = (0..(num_rows as u32 + 1))
            .map(|i| i * self.byte_width as u32)
            .collect::<Vec<_>>();
        println!("{:?}", offsets_vec);
        let offsets_buffer = Buffer::from_slice_ref(&offsets_vec);
        println!("{:?}", offsets_buffer);

        let string_data = Box::new(VariableWidthBlock {
            bits_per_offset: 32,
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

// Bytes computed across all string arrays, similar to indices above
pub fn get_byte_width_from_binary_arrays(arrays: &[ArrayRef]) -> usize {
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
        }
        else if let Some(arr) = arr.as_string_opt::<i64>() {
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
        }
        else if let Some(arr) = arr.as_binary_opt::<i32>() {
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
        }
        else if let Some(arr) = arr.as_binary_opt::<i64>() {
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
        }
        else {
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
        let byte_width = get_byte_width_from_binary_arrays(arrays);
        let byte_arrays = get_bytes_from_binary_arrays(arrays, byte_width);
        let encoded_bytes = self.bytes_encoder.encode(&byte_arrays, buffer_index)?;

        Ok(EncodedArray {
            buffers: encoded_bytes.buffers,
            encoding: pb::ArrayEncoding {
                array_encoding: Some(pb::array_encoding::ArrayEncoding::FixedSizeBinary(
                    Box::new(pb::FixedSizeBinary {
                        byte_width: byte_width as u32,
                        bytes: Some(Box::new(encoded_bytes.encoding)),
                    }),
                )),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow_array::{Array, FixedSizeBinaryArray, LargeStringArray, StringArray};
    use arrow_buffer::Buffer;
    use arrow_schema::{DataType, Field};

    use crate::testing::{
        check_round_trip_encoding_of_data, check_round_trip_encoding_random, TestCases,
    };

    #[test_log::test(tokio::test)]
    async fn test_simple_fixed_size_utf8() {
        let string_array = StringArray::from(vec![
            Some("abc"),
            Some("def"),
            Some("ghi"),
            Some("jkl"),
            Some("mno"),
        ]);
        println!(
            "offsets: {:?}",
            string_array.offsets().inner().inner().clone()
        );

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
        let string_array = LargeStringArray::from(vec![
            Some("abc"),
            None,
            Some("ghi"),
            None,
            Some("mno"),
        ]);
        // println!(
        //     "offsets: {:?}",
        //     string_array.offsets().inner().inner().clone()
        // );
        // println!("nulls: {:?}", string_array.nulls());
        let (_, values, nulls) = string_array.clone().into_parts();
        println!("{:?}", values);
        println!("{:?}", nulls);
        // let fixed_size_binary_array = FixedSizeBinaryArray::new(3, values, nulls);
        // println!("{:?}", fixed_size_binary_array);

        let test_cases = TestCases::default()
            .with_range(0..2)
            .with_range(0..3)
            .with_range(1..3)
            .with_indices(vec![0, 1, 3, 4]);

        println!("Running test...");
        check_round_trip_encoding_of_data(
            vec![Arc::new(string_array)],
            &test_cases,
            HashMap::new(),
        )
        .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_sliced_utf8() {
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
}
