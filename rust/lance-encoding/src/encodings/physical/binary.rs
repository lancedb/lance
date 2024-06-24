// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use core::panic;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt32Type;
// use arrow::compute::concat;
use arrow_array::{
    builder::{ArrayBuilder, Int32Builder, UInt32Builder, UInt8Builder},
    Array, ArrayRef, Int32Array, UInt32Array,
};
use bytes::BytesMut;
use futures::stream::StreamExt;
use futures::{future::BoxFuture, stream::FuturesOrdered, FutureExt};
// use rand::seq::index;

use crate::{
    decoder::{PageScheduler, PrimitivePageDecoder},
    encoder::{ArrayEncoder, EncodedArray},
    format::pb,
    EncodingsIo,
};

use crate::decoder::LogicalPageDecoder;
use crate::encodings::logical::primitive::PrimitiveFieldDecoder;

use arrow_array::PrimitiveArray;
use arrow_schema::DataType;
use lance_core::Result;
use std::ops::Deref;

#[derive(Debug)]
pub struct BinaryPageScheduler {
    indices_scheduler: Arc<dyn PageScheduler>,
    bytes_scheduler: Arc<dyn PageScheduler>,
}

impl BinaryPageScheduler {
    pub fn new(
        indices_scheduler: Arc<dyn PageScheduler>,
        bytes_scheduler: Arc<dyn PageScheduler>,
    ) -> Self {
        Self {
            indices_scheduler,
            bytes_scheduler,
        }
    }
}

impl PageScheduler for BinaryPageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u32>],
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
            .collect::<Vec<std::ops::Range<u32>>>();

        let mut futures_ordered = FuturesOrdered::new();
        for range in indices_ranges.iter() {
            let indices_page_decoder =
                self.indices_scheduler
                    .schedule_ranges(&[range.clone()], scheduler, top_level_row);
            futures_ordered.push_back(indices_page_decoder);
        }

        let ranges = ranges.to_vec();
        let copy_scheduler = scheduler.clone();
        let copy_bytes_scheduler = self.bytes_scheduler.clone();
        let copy_indices_ranges = indices_ranges.to_vec();

        async move {
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

            let mut builder = UInt32Builder::new();
            let mut bytes_ranges = Vec::new();
            let mut curr_range_idx = 0;
            let mut last = 0;
            while let Some(indices_page_decoder) = futures_ordered.next().await {
                let indices: Arc<dyn PrimitivePageDecoder> = Arc::from(indices_page_decoder?);

                // Build and run decode task for offsets
                let curr_indices_range = copy_indices_ranges[curr_range_idx].clone();
                let curr_row_range = ranges[curr_range_idx].clone();
                let indices_num_rows = curr_indices_range.end - curr_indices_range.start;
                let mut primitive_wrapper = PrimitiveFieldDecoder::new_from_data(
                    indices,
                    DataType::UInt32,
                    indices_num_rows,
                );
                let drained_task = primitive_wrapper.drain(indices_num_rows)?;
                let indices_decode_task = drained_task.task;
                let decoded_part = indices_decode_task.decode()?;

                let indices_array = decoded_part.as_primitive::<UInt32Type>();
                let mut indices_vec = indices_array.values().to_vec();

                // Pad a zero at the start if the first row is requested
                // This is because the offsets do not start with 0 by default
                if curr_row_range.start == 0 {
                    indices_vec.insert(0, 0);
                }

                // Normalize the indices as described above
                let normalized_indices: PrimitiveArray<UInt32Type> = indices_vec
                    .iter()
                    .map(|x| x - indices_vec[0] + last)
                    .collect();
                last = normalized_indices.value(normalized_indices.len() - 1);
                let normalized_vec = normalized_indices.values().to_vec();

                // The first vector to be normalized should not have the leading zero removed
                let truncated_vec = if curr_range_idx == 0 {
                    normalized_vec.as_slice()
                } else {
                    &normalized_vec[1..]
                };

                builder.append_slice(truncated_vec);

                // get bytes range from the index range
                let bytes_range = if curr_row_range.start != 0 {
                    indices_array.value(0)..indices_array.value(indices_array.len() - 1)
                } else {
                    0..indices_array.value(indices_array.len() - 1)
                };

                bytes_ranges.push(bytes_range);
                curr_range_idx += 1;
            }

            let decoded_indices = Arc::new(builder.finish());

            let bytes_ranges_slice = bytes_ranges.as_slice();

            // Schedule the bytes for decoding
            let bytes_page_decoder = copy_bytes_scheduler.schedule_ranges(
                bytes_ranges_slice,
                &copy_scheduler,
                top_level_row,
            );

            let bytes_decoder: Box<dyn PrimitivePageDecoder> = bytes_page_decoder.await?;

            Ok(Box::new(BinaryPageDecoder {
                decoded_indices,
                bytes_decoder,
            }) as Box<dyn PrimitivePageDecoder>)
        }
        .boxed()
    }
}

struct BinaryPageDecoder {
    decoded_indices: Arc<dyn Array>,
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
    // The normalized offsets are [0, 4, 8, 13]
    // We only need [8, 13] to decode in this case.
    // These need to be normalized in order to build the string later
    // So return [0, 5]
    fn decode_into(
        &self,
        rows_to_skip: u32,
        num_rows: u32,
        all_null: &mut bool,
    ) -> Result<Vec<BytesMut>> {
        let offsets = self
            .decoded_indices
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();

        // let mut capacities = vec![(0, false); self.num_buffers() as usize];
        // 32 bits or 4 bytes per value.
        // capacities[0].0 = (num_rows as u64) * 4;
        // capacities[0].1 = true;

        let bytes_to_skip = offsets.value(rows_to_skip as usize);
        let num_bytes = offsets.value((rows_to_skip + num_rows) as usize) - bytes_to_skip;        
        let target_offsets = offsets.slice(
            rows_to_skip.try_into().unwrap(),
            (num_rows + 1).try_into().unwrap(),
        );

        let mut bytes_buffers = self
            .bytes_decoder
            .decode_into(bytes_to_skip, num_bytes, all_null)?;

        // Normalize offsets
        let target_vec = target_offsets.values();
        let normalized_array: PrimitiveArray<UInt32Type> =
            target_vec.iter().map(|x| x - target_vec[0]).collect();
        let normalized_values = normalized_array.values();

        let byte_slice = normalized_values.inner().deref();
        let mut dest_buffers = vec![BytesMut::from(byte_slice)];
        dest_buffers.append(&mut bytes_buffers);

        // let mut dest_buffers = create_buffers_from_capacities(capacities);

        // copy target_offsets into dest_buffers[0]
        // dest_buffers[0].extend_from_slice(byte_slice);

        // Copy decoded bytes into dest_buffers[1..]
        // Currently an empty null buffer is the first one
        // The actual bytes are in the second buffer
        // Including the indices this results in 3 buffers in total
        // dest_buffers.append(&mut bytes_buffers);

        Ok(dest_buffers)
    }

    fn num_buffers(&self) -> u32 {
        self.bytes_decoder.num_buffers() + 1
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
fn get_indices_from_string_arrays(arrays: &[ArrayRef]) -> ArrayRef {
    let mut indices_builder = Int32Builder::new();
    let mut last_offset = 0;
    arrays.iter().for_each(|arr| {
        let string_arr = arrow_array::cast::as_string_array(arr);
        let offsets = string_arr.offsets().inner();
        let mut offsets = offsets.slice(1, offsets.len() - 1).to_vec();

        if indices_builder.len() == 0 {
            last_offset = offsets[offsets.len() - 1];
        } else {
            offsets = offsets
                .iter()
                .map(|offset| offset + last_offset)
                .collect::<Vec<i32>>();
            last_offset = offsets[offsets.len() - 1];
        }

        let new_int_arr = Int32Array::from(offsets);
        indices_builder.append_slice(new_int_arr.values());
    });

    Arc::new(indices_builder.finish()) as ArrayRef
}

// Bytes computed across all string arrays, similar to indices above
fn get_bytes_from_string_arrays(arrays: &[ArrayRef]) -> ArrayRef {
    let mut bytes_builder = UInt8Builder::new();
    arrays.iter().for_each(|arr| {
        let string_arr = arrow_array::cast::as_string_array(arr);
        let values = string_arr.values();
        bytes_builder.append_slice(values);
    });

    Arc::new(bytes_builder.finish()) as ArrayRef
}

impl ArrayEncoder for BinaryEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: &mut u32) -> Result<EncodedArray> {
        let (null_count, _row_count) = arrays
            .iter()
            .map(|arr| (arr.null_count() as u32, arr.len() as u32))
            .fold((0, 0), |acc, val| (acc.0 + val.0, acc.1 + val.1));

        if null_count != 0 {
            panic!("Data contains null values, not currently supported for binary data.")
        } else {
            let index_array = get_indices_from_string_arrays(arrays);
            let encoded_indices = self.indices_encoder.encode(&[index_array], buffer_index)?;

            let byte_array = get_bytes_from_string_arrays(arrays);
            let encoded_bytes = self.bytes_encoder.encode(&[byte_array], buffer_index)?;

            let mut encoded_buffers = encoded_indices.buffers;
            encoded_buffers.extend(encoded_bytes.buffers);

            Ok(EncodedArray {
                buffers: encoded_buffers,
                encoding: pb::ArrayEncoding {
                    array_encoding: Some(pb::array_encoding::ArrayEncoding::Binary(Box::new(
                        pb::Binary {
                            indices: Some(Box::new(encoded_indices.encoding)),
                            bytes: Some(Box::new(encoded_bytes.encoding)),
                        },
                    ))),
                },
            })
        }
        // Currently not handling all null cases in this array encoder.
        // TODO: Separate behavior for all null rows vs some null rows
        // else if null_count == row_count {
        //     let nullability = pb::nullable::Nullability::AllNulls(pb::nullable::AllNull {});

        //     Ok(EncodedArray {
        //         buffers: vec![],
        //         encoding: pb::ArrayEncoding {
        //             array_encoding: Some(pb::array_encoding::ArrayEncoding::Nullable(Box::new(
        //                 pb::Nullable {
        //                     nullability: Some(nullability),
        //                 },
        //             ))),
        //         },
        //     })
        // }

        // let arr_encoding = self.values_encoder.encode(arrays, buffer_index)?;
        // let encoding = pb::nullable::Nullability::NoNulls(Box::new(pb::nullable::NoNull {
        //     values: Some(Box::new(arr_encoding.encoding)),
        // }));
        // (arr_encoding.buffers, encoding)
        // } else if null_count == row_count {
        // let encoding = pb::nullable::Nullability::AllNulls(pb::nullable::AllNull {});
        // (vec![], encoding)
        // } else {
        //     let validity_as_arrays = arrays
        //         .iter()
        //         .map(|arr| {
        //             if let Some(nulls) = arr.nulls() {
        //                 Arc::new(BooleanArray::new(nulls.inner().clone(), None)) as ArrayRef
        //             } else {
        //                 let buff = BooleanBuffer::new_set(arr.len());
        //                 Arc::new(BooleanArray::new(buff, None)) as ArrayRef
        //             }
        //         })
        //         .collect::<Vec<_>>();

        //     let validity_buffer_index = *buffer_index;
        //     *buffer_index += 1;
        //     let validity = BitmapBufferEncoder::default().encode(&validity_as_arrays)?;
        //     let validity_encoding = Box::new(pb::ArrayEncoding {
        //         array_encoding: Some(pb::array_encoding::ArrayEncoding::Flat(pb::Flat {
        //             bits_per_value: 1,
        //             buffer: Some(pb::Buffer {
        //                 buffer_index: validity_buffer_index,
        //                 buffer_type: pb::buffer::BufferType::Page as i32,
        //             }),
        //             compression: None,
        //         })),
        //     });

        //     let arr_encoding = self.values_encoder.encode(arrays, buffer_index)?;
        //     let encoding = pb::nullable::Nullability::SomeNulls(Box::new(pb::nullable::SomeNull {
        //         validity: Some(validity_encoding),
        //         values: Some(Box::new(arr_encoding.encoding)),
        //     }));

        //     let mut buffers = arr_encoding.buffers;
        //     buffers.push(EncodedArrayBuffer {
        //         parts: validity.parts,
        //         index: validity_buffer_index,
        //     });
        //     (buffers, encoding)
        // };

        // Ok(EncodedArray {
        //     buffers,
        //     encoding: pb::ArrayEncoding {
        //         array_encoding: Some(pb::array_encoding::ArrayEncoding::Nullable(Box::new(
        //             pb::Nullable {
        //                 nullability: Some(nullability),
        //             },
        //         ))),
        //     },
        // })
    }
}
