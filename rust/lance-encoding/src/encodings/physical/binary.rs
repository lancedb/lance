// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use core::panic;
use std::sync::Arc;

use arrow_array::{
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
        let indices_page_decoder =
            self.indices_scheduler
                .schedule_ranges(&ranges, scheduler, top_level_row);

        let ranges = ranges.to_vec();
        let copy_scheduler = scheduler.clone();
        let copy_bytes_scheduler = Arc::clone(&self.bytes_scheduler);

        async move {
            println!("Started async move");
            let indices_page_decoder = indices_page_decoder.await?;
            let indices: Arc<dyn PhysicalPageDecoder> = Arc::from(indices_page_decoder);

            // assuming ranges is in sorted order
            let indices_num_rows = ranges[ranges.len() - 1].end - ranges[0].start;
            let mut primitive_wrapper =
                PrimitiveFieldDecoder::new_from_data(indices, DataType::UInt32, indices_num_rows);
            let drained_task = primitive_wrapper.drain(indices_num_rows)?;
            let indices_decode_task = drained_task.task;

            let decoded_indices = indices_decode_task.decode()?;
            println!("Decoded indices: {:?}", decoded_indices);

            let mut net_bytes = 0;
            let mut bytes_ranges: Vec<std::ops::Range<u32>> = Vec::new();

            // want rows 0-2, 4-6. Or rows 0, 1, 4, 5.
            //      0        1        2        3       4        5
            // "abcd", "hello", "abcd", "apple", "hello", "abcd"
            //   4,        9,     13,      18,      23,     27
            // row i = indices[i-1] -> indices[i]. Except when i = 0, then it's 0 -> indices[i].
            // Bytes for row[i] = indices[i] - indices[i-1]
            // rows a -> b. a != 0. inclusive.
            // Bytes = indices[b] - indices[a]
            // rows a -> b. a = 0. inclusive.
            // Bytes = indices[b]

            ranges.iter().for_each(|range| {
                let end_index = usize::try_from(range.end).unwrap() - 1;
                let start_index = usize::try_from(range.start).unwrap();

                if let Some(indices_array) = decoded_indices.as_any().downcast_ref::<UInt32Array>()
                {
                    if start_index == 0 {
                        let offset_end = indices_array.value(end_index);
                        net_bytes += offset_end;
                        bytes_ranges.push(0..(offset_end + 1));
                    } else {
                        let offset_start = indices_array.value(start_index);
                        let offset_end = indices_array.value(end_index);
                        net_bytes += offset_end - offset_start;
                        bytes_ranges.push(offset_start..(offset_end + 1));
                    }
                }
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
// These arrays are padded with a 0 at the beginning
fn get_indices_from_string_arrays(arrays: &[ArrayRef]) -> Vec<ArrayRef> {
    let index_arrays: Vec<ArrayRef> = arrays
        .iter()
        .filter_map(|arr| {
            if let Some(string_arr) = arr.as_any().downcast_ref::<StringArray>() {
                println!("String array: {:?}", string_arr);
                let offsets = string_arr.value_offsets().to_vec();

                // Not using UInt64, since the range values during decoding expect u32's 
                let mut new_int_arr = Int32Array::from(offsets);
                new_int_arr = new_int_arr.slice(1, new_int_arr.len() - 1);
                // match cast(&new_int_arr, &DataType::UInt64) {
                //     Ok(res) => Some(Arc::new(res) as ArrayRef),
                //     Err(_) => panic!("Failed to cast to uint64"),
                // }
                Some(Arc::new(new_int_arr) as ArrayRef)
            } else {
                panic!("Failed to downcast data to string array");
            }
        })
        .collect();

    index_arrays
}

fn get_bytes_from_string_arrays(arrays: &[ArrayRef]) -> Vec<ArrayRef> {
    let byte_arrays: Vec<ArrayRef> = arrays
        .iter()
        .filter_map(|arr| {
            if let Some(string_arr) = arr.as_any().downcast_ref::<StringArray>() {
                let values = string_arr.values().to_vec();
                let bytes_arr = Arc::new(UInt8Array::from(values)) as ArrayRef;
                Some(bytes_arr)
            } else {
                panic!("Failed to downcast data to string array");
            }
        })
        .collect();

    byte_arrays
}

impl ArrayEncoder for BinaryEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: &mut u32) -> Result<EncodedArray> {
        let (null_count, row_count) = arrays
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
