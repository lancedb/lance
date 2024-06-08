// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use core::panic;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt32Type;
use arrow_array::{
    builder::{ArrayBuilder, Int32Builder, UInt8Builder},
    Array, ArrayRef, Int32Array, StringArray, UInt32Array, UInt8Array,
};
use futures::{future::BoxFuture, FutureExt};
// use rand::seq::index;

use crate::{
    decoder::{PageScheduler, PhysicalPageDecoder},
    encoder::{ArrayEncoder, EncodedArray},
    format::pb,
    EncodingsIo,
};

use crate::decoder::LogicalPageDecoder;
use crate::encodings::logical::primitive::PrimitiveFieldDecoder;

// use arrow_cast::cast::cast;
use arrow_schema::DataType;
use lance_core::Result;

#[derive(Debug)]
pub struct BinaryPageScheduler {
    indices_scheduler: Box<dyn PageScheduler>,
    bytes_scheduler: Arc<dyn PageScheduler>,
}

impl BinaryPageScheduler {
    pub fn new(
        indices_scheduler: Box<dyn PageScheduler>,
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
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>> {
        println!("Inside BinaryPageScheduler");

        // if user wants rows a..b, or range a..b+1
        // Case 1: if a != 0, we need indices a-1..b inclusive
        // Case 2: if a = 0, we need indices 0..b inclusive

        let indices_ranges = ranges
            .iter()
            .map(|range| {
                if range.start != 0 {
                    (range.start - 1)..range.end
                } else {
                    0..range.end
                }
            })
            .collect::<Vec<std::ops::Range<u32>>>();

        let indices_page_decoder =
            self.indices_scheduler
                .schedule_ranges(&indices_ranges, scheduler, top_level_row);

        let ranges = ranges.to_vec();
        println!("Ranges: {:?}", ranges);
        let copy_scheduler = scheduler.clone();
        let copy_bytes_scheduler = Arc::clone(&self.bytes_scheduler);

        async move {
            println!("Started async move");
            let indices_page_decoder = indices_page_decoder.await?;
            let indices: Arc<dyn PhysicalPageDecoder> = Arc::from(indices_page_decoder);

            let indices_num_rows = indices_ranges
                .iter()
                .map(|range| range.end - range.start)
                .sum();
            // let indices_num_rows = ranges[ranges.len() - 1].end - ranges[0].start;
            println!("Indices num rows: {:?}", indices_num_rows);
            let mut primitive_wrapper =
                PrimitiveFieldDecoder::new_from_data(indices, DataType::UInt32, indices_num_rows);
            let drained_task = primitive_wrapper.drain(indices_num_rows)?;
            let indices_decode_task = drained_task.task;

            let decoded_indices = indices_decode_task.decode()?;
            println!("Decoded indices: {:?}", decoded_indices);
            let indices_array = decoded_indices.as_primitive::<UInt32Type>();

            let mut net_bytes = 0;
            let mut bytes_ranges: Vec<std::ops::Range<u32>> = Vec::new();

            // want range [1..3] or rows 1, 2
            //      0        1        2        3       4        5
            // "abcd", "hello", "abcd", "apple", "hello", "abcd"
            //   4,        9,     13,      18,      23,     27
            // decoded indices = [4, 9, 13]
            // if we wanted rows 0..3 even then the decoded indices would be the same as above.
            // But then len(indices) = len(range), so we know that we want row 0 as well.
            // Normally len(indices) = len(range) + 1
            // So, non-zero case (a != 0): net bytes for range a->b, inclusive = indices[b] - indices[a-1]
            // Zero case: net bytes for range 0->b, inclusive = indices[b]

            // want range [1..2] or row 1
            // want indices [1]

            ranges.iter().for_each(|range| {
                let end_index = usize::try_from(range.end).unwrap() - 1; // range.end=2, end_index=1
                let start_index = usize::try_from(range.start).unwrap(); // range.start=1, start_index=1

                let offset_start = if start_index == 0 {
                    0
                } else {
                    indices_array.value(start_index - 1) // indices[1] = 9
                };

                let offset_end = indices_array.value(end_index); // indices[2] = 13
                net_bytes += offset_end - offset_start;
                bytes_ranges.push(offset_start..(offset_end + 1));
            });

            println!("Net bytes: {:?}", net_bytes);
            let bytes_ranges_slice = bytes_ranges.as_slice();
            for range in bytes_ranges_slice {
                println!("Bytes range: {:?}, {:?}", range.start, range.end);
            }

            let bytes_page_decoder = copy_bytes_scheduler.schedule_ranges(
                bytes_ranges_slice,
                &copy_scheduler,
                top_level_row,
            );

            let bytes_decoder: Box<dyn PhysicalPageDecoder> = bytes_page_decoder.await?;

            Ok(Box::new(BinaryPageDecoder {
                decoded_indices,
                bytes_decoder,
            }) as Box<dyn PhysicalPageDecoder>)
        }
        .boxed()
    }
}

struct BinaryPageDecoder {
    decoded_indices: Arc<dyn Array>,
    bytes_decoder: Box<dyn PhysicalPageDecoder>,
}

fn get_bytes_from_rows(
    rows_to_skip: u32,
    num_rows: u32,
    offsets: &UInt32Array,
) -> (u32, u32, UInt32Array) {
    let bytes_to_skip;
    let num_bytes;
    let target_offsets;

    if rows_to_skip == 0 {
        target_offsets = offsets.slice(0, num_rows as usize);
        bytes_to_skip = 0;
        num_bytes = offsets.value(num_rows as usize - 1);
    } else {
        target_offsets = offsets.slice(
            rows_to_skip.try_into().unwrap(),
            num_rows.try_into().unwrap(),
        ); // 4, 9
        bytes_to_skip = offsets.value(rows_to_skip as usize);
        num_bytes = offsets.value((rows_to_skip + num_rows) as usize) - bytes_to_skip;
    }

    (bytes_to_skip, num_bytes, target_offsets)
}

impl PhysicalPageDecoder for BinaryPageDecoder {
    fn update_capacity(
        &self,
        rows_to_skip: u32,
        num_rows: u32,
        buffers: &mut [(u64, bool)],
        all_null: &mut bool,
    ) {
        println!("Inside BinaryPageDecoder update_capacity");
        println!("Rows to skip: {:?}, Num rows: {:?}", rows_to_skip, num_rows);
        let offsets = self
            .decoded_indices
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap(); // 4, 9, 13, 18, 23, 27
        let (bytes_to_skip, num_bytes, _) = get_bytes_from_rows(rows_to_skip, num_rows, offsets);

        // 32 bits or 4 bytes per value.
        buffers[0].0 = (num_rows as u64) * 4;
        buffers[0].1 = true;

        println!(
            "Capacity update: Bytes to skip: {:?}, Num bytes: {:?}",
            bytes_to_skip, num_bytes
        );
        self.bytes_decoder
            .update_capacity(bytes_to_skip, num_bytes, &mut buffers[1..], all_null);
    }

    fn decode_into(
        &self,
        rows_to_skip: u32,
        num_rows: u32,
        dest_buffers: &mut [bytes::BytesMut],
    ) -> Result<()> {
        println!("Rows to skip: {:?}, Num rows: {:?}", rows_to_skip, num_rows);
        let offsets = self
            .decoded_indices
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap(); // 4, 9, 13, 18, 23, 27
        println!("Offsets: {:?}", offsets);

        let (bytes_to_skip, num_bytes, target_offsets) =
            get_bytes_from_rows(rows_to_skip, num_rows, offsets);

        println!("Target offsets: {:?}", target_offsets);
        println!(
            "Bytes to skip: {:?}, Num bytes: {:?}",
            bytes_to_skip, num_bytes
        );

        // copy target_offsets into dest_buffers[0]
        let values = target_offsets.values();
        let byte_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                values.as_ptr() as *const u8,
                values.len() * std::mem::size_of::<u32>(),
            )
        };

        dest_buffers[0].extend_from_slice(byte_slice);
        self.bytes_decoder
            .decode_into(bytes_to_skip, num_bytes, &mut dest_buffers[1..])?;

        // for i in 0..3 {
        //     println!(
        //         "length, capacity of buffer: {:?}, {:?} ",
        //         dest_buffers[i].len(),
        //         dest_buffers[i].capacity()
        //     );
        // }

        Ok(())
    }

    fn num_buffers(&self) -> u32 {
        self.bytes_decoder.num_buffers() + 1
        // 0
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

// Returns indices arrays from string arrays
fn get_indices_from_string_arrays(arrays: &[ArrayRef]) -> Vec<ArrayRef> {
    let mut indices_builder = Int32Builder::new();
    let mut last_offset = 0;
    arrays.iter().for_each(|arr| {
        let string_arr = arrow_array::cast::as_string_array(arr);
        println!("String array: {:?}", string_arr);
        let mut offsets = string_arr.value_offsets().to_vec();
        offsets = offsets[1..].to_vec();
        println!("Offsets: {:?}", offsets);

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

        // match cast(&new_int_arr, &DataType::UInt64) {
        //     Ok(res) => Some(Arc::new(res) as ArrayRef),
        //     Err(_) => panic!("Failed to cast to uint64"),
        // }
        // Some(Arc::new(new_int_arr) as ArrayRef)
    });

    let final_array: ArrayRef = Arc::new(indices_builder.finish()) as ArrayRef;

    vec![final_array]
}

fn get_bytes_from_string_arrays(arrays: &[ArrayRef]) -> Vec<ArrayRef> {
    let mut bytes_builder = UInt8Builder::new();
    arrays.iter().for_each(|arr| {
        let string_arr = arrow_array::cast::as_string_array(arr);
        let values = string_arr.values().to_vec();
        bytes_builder.append_slice(&values);
        // let bytes_arr = Arc::new(UInt8Array::from(values)) as ArrayRef;
        // Some(bytes_arr)
    });

    let final_array = Arc::new(bytes_builder.finish()) as ArrayRef;

    vec![final_array]
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
            let index_arrays = get_indices_from_string_arrays(arrays);
            let encoded_indices = self.indices_encoder.encode(&index_arrays, buffer_index)?;
            for arr in &index_arrays {
                println!("indices: {:?}", arr);
            }

            let byte_arrays = get_bytes_from_string_arrays(arrays);
            for arr in &byte_arrays {
                println!("arr: {:?}", arr);
            }
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
