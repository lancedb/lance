// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use core::panic;
use std::sync::Arc;

// use arrow_array::types::{UInt32Type, UInt64Type};
// use arrow::compute::concat;
use arrow_array::{
    // builder::{ArrayBuilder, Int32Builder, UInt32Builder, UInt8Builder},
    Array,
    ArrayRef,
    StringArray,
    UInt64Array,
};
// use futures::stream::StreamExt;
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

use arrow_schema::DataType;
use lance_core::Result;
use std::collections::HashMap;
// use std::ops::Deref;

#[derive(Debug)]
pub struct DictionaryPageScheduler {
    indices_scheduler: Arc<dyn PageScheduler>,
    items_scheduler: Arc<dyn PageScheduler>,
    size: u32,
}

impl DictionaryPageScheduler {
    pub fn new(
        indices_scheduler: Arc<dyn PageScheduler>,
        items_scheduler: Arc<dyn PageScheduler>,
        size: u32,
    ) -> Self {
        Self {
            indices_scheduler,
            items_scheduler,
            size,
        }
    }
}

impl PageScheduler for DictionaryPageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u32>],
        scheduler: &Arc<dyn EncodingsIo>,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>> {
        // Run schedule ranges on all indices together, decode all indices together.
        // we need to decode certain elements of the dictionary that are
        // present in distinct indices, i.e. distinct values from indices[a]...indices[b] and indices[c]...indices[d].
        // So we need to decode the *distinct* dictionary items in the same order of appearance in the indices.
        // Let's say this gives schedule dictionary[m...n]
        // Now during update_capacity, allocate capacity for the items encoder on all the relevant rows
        // During decode_into, decode all the dict items.
        // When reconstructing:
        // get indices buffers, items buffers.
        // Need to rebuild the large array.
        // have a bunch of random indices, bunch of items in the same order of appearance as indices
        // process all indices sequentially to build the array. use a hashmap to keep track of dict items that have been processed.
        // Should we make a ListArray<StringArray>? Since the return type of primitive_array_from_buffers() is ArrayRef

        // can rebuild the array by iterating over the indices and the distinct dict items.
        // decoded indices 1, 0, 1, 3. We need dict[0], dict[1], dict[3] of the original dict.
        // The decoded dictionary will have 2 elements. We should process it sequentially -

        // Schedule items for decoding
        let items_page_decoder =
            self.items_scheduler
                .schedule_ranges(&[0..self.size], scheduler, top_level_row);

        let copy_scheduler = scheduler.clone();
        let ranges = ranges.to_vec();
        let copy_indices_scheduler = self.indices_scheduler.clone();
        let copy_items_scheduler = self.items_scheduler.clone();
        let copy_size = self.size.clone();

        async move {
            let items: Arc<dyn PhysicalPageDecoder> = Arc::from(items_page_decoder.await?);

            let mut primitive_wrapper =
                PrimitiveFieldDecoder::new_from_data(items, DataType::Utf8, copy_size);

            // Decode all items
            let drained_task = primitive_wrapper.drain(copy_size)?;
            let items_decode_task = drained_task.task;
            let decoded_dict = items_decode_task.decode()?;
            println!("Decoded dict: {:?}", decoded_dict);

            // Schedule indices for decoding
            let indices_page_decoder =
                copy_indices_scheduler.schedule_ranges(&ranges, &copy_scheduler, top_level_row);

            let indices_decoder: Box<dyn PhysicalPageDecoder> = indices_page_decoder.await?;

            let copy_items_page_decoder = copy_items_scheduler.schedule_ranges(
                &[0..copy_size],
                &copy_scheduler,
                top_level_row,
            );
            let items_decoder: Box<dyn PhysicalPageDecoder> = copy_items_page_decoder.await?;

            Ok(Box::new(DictionaryPageDecoder {
                decoded_dict,
                indices_decoder,
                items_decoder,
            }) as Box<dyn PhysicalPageDecoder>)
        }
        .boxed()
    }
}

struct DictionaryPageDecoder {
    decoded_dict: Arc<dyn Array>,
    indices_decoder: Box<dyn PhysicalPageDecoder>,
    items_decoder: Box<dyn PhysicalPageDecoder>,
}

impl PhysicalPageDecoder for DictionaryPageDecoder {
    // Continuing the example from BinaryPageScheduler
    // Suppose batch_size = 2. Then first, rows_to_skip=0, num_rows=2
    // Need to scan 2 rows
    // First row will be 4-0=4 bytes, second also 8-4=4 bytes.
    // Allocate 8 bytes capacity.
    // Next rows_to_skip=2, num_rows=1
    // Skip 8 bytes. Allocate 5 bytes capacity.
    fn update_capacity(
        &self,
        rows_to_skip: u32,
        num_rows: u32,
        buffers: &mut [(u64, bool)],
        all_null: &mut bool,
    ) {
        // Allocate capacity for all indices
        self.indices_decoder
            .update_capacity(rows_to_skip, num_rows, &mut buffers[..2], all_null);

        let dictionary = self
            .decoded_dict
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let dict_bytes = dictionary.values().as_slice().len();
        println!("Dict bytes: {:?}", dict_bytes);

        // Allocate capacity for dictionary
        self.items_decoder.update_capacity(
            0,
            self.decoded_dict.len() as u32,
            &mut buffers[2..],
            all_null,
        );

        println!("Capacity of decoded dict buffer 0: {:?}", buffers[0].0);
        println!("Capacity of decoded dict buffer 1: {:?}", buffers[1].0);
        println!("Capacity of decoded dict buffer 2: {:?}", buffers[2].0);
        println!("Capacity of decoded dict buffer 3: {:?}", buffers[3].0);
        println!("Capacity of decoded dict buffer 4: {:?}", buffers[4].0);
    }

    // Continuing from update_capacity:
    // When rows_to_skip=2, num_rows=1
    // The normalized offsets are [0, 4, 8, 13]
    // We only need [8, 13] to decode in this case.
    // These need to be normalized in order to build the string later
    // So return [0, 5]
    fn decode_into(
        &self,
        rows_to_skip: u32,
        num_rows: u32,
        dest_buffers: &mut [bytes::BytesMut],
    ) -> Result<()> {
        self.indices_decoder
            .decode_into(rows_to_skip, num_rows, &mut dest_buffers[3..])?;

        // println!("Length of decoded indices: {:?}", dest_buffers[4].len());
        // println!(
        //     "Capacity of decoded indices: {:?}",
        // dest_buffers[4].capacity()
        // );

        self.items_decoder.decode_into(
            0,
            self.decoded_dict.len() as u32,
            &mut dest_buffers[..3],
        )?;

        println!("Buffer 0 contents: {:?}", dest_buffers[0].as_ref());
        println!("Buffer 1 contents: {:?}", dest_buffers[1].as_ref());
        println!("Buffer 2 contents: {:?}", dest_buffers[2].as_ref());
        println!("Buffer 3 contents: {:?}", dest_buffers[3].as_ref());
        println!("Buffer 4 contents: {:?}", dest_buffers[4].as_ref());

        Ok(())
    }

    fn num_buffers(&self) -> u32 {
        self.indices_decoder.num_buffers() + self.items_decoder.num_buffers() // 2 + 3 = 5
    }
}

#[derive(Debug)]
pub struct DictionaryEncoder {
    indices_encoder: Box<dyn ArrayEncoder>,
    items_encoder: Box<dyn ArrayEncoder>,
}

impl DictionaryEncoder {
    pub fn new(
        indices_encoder: Box<dyn ArrayEncoder>,
        items_encoder: Box<dyn ArrayEncoder>,
    ) -> Self {
        Self {
            indices_encoder,
            items_encoder,
        }
    }
}

// Creates indices arrays from string arrays
// Strings are a vector of arrays corresponding to each record batch
// Zero offset is removed from the start of the offsets array
// The indices array is computed across all arrays in the vector
// fn get_indices_from_string_arrays(arrays: &[ArrayRef]) -> ArrayRef {
//     let mut indices_builder = Int32Builder::new();
//     let mut last_offset = 0;
//     arrays.iter().for_each(|arr| {
//         let string_arr = arrow_array::cast::as_string_array(arr);
//         let offsets = string_arr.offsets().inner();
//         let mut offsets = offsets.slice(1, offsets.len() - 1).to_vec();

//         if indices_builder.len() == 0 {
//             last_offset = offsets[offsets.len() - 1];
//         } else {
//             offsets = offsets
//                 .iter()
//                 .map(|offset| offset + last_offset)
//                 .collect::<Vec<i32>>();
//             last_offset = offsets[offsets.len() - 1];
//         }

//         let new_int_arr = Int32Array::from(offsets);
//         indices_builder.append_slice(new_int_arr.values());
//     });

//     Arc::new(indices_builder.finish()) as ArrayRef
// }

// Bytes computed across all string arrays, similar to indices above
// fn get_bytes_from_string_arrays(arrays: &[ArrayRef]) -> ArrayRef {
//     let mut bytes_builder = UInt8Builder::new();
//     arrays.iter().for_each(|arr| {
//         let string_arr = arrow_array::cast::as_string_array(arr);
//         let values = string_arr.values();
//         bytes_builder.append_slice(values);
//     });

//     Arc::new(bytes_builder.finish()) as ArrayRef
// }

fn get_indices_items_from_arrays(arrays: &[ArrayRef]) -> (ArrayRef, ArrayRef) {
    let mut arr_hashmap: HashMap<&str, u64> = HashMap::new();
    let mut curr_dict_index = 0;
    let mut dict_indices = Vec::new();
    let mut dict_elements = Vec::new();

    for arr in arrays.iter() {
        let string_array = arrow_array::cast::as_string_array(arr);
        for i in 0..string_array.len() {
            let st = string_array.value(i);

            if arr_hashmap.contains_key(st) {
                dict_indices.push(arr_hashmap.get(st).unwrap().clone());
            } else {
                // insert into hashmap
                arr_hashmap.insert(string_array.value(i), curr_dict_index);
                dict_indices.push(curr_dict_index);
                dict_elements.push(st);
                curr_dict_index += 1;
            }
        }
    }

    let array_dict_indices = Arc::new(UInt64Array::from(dict_indices)) as ArrayRef;
    let array_dict_elements = Arc::new(StringArray::from(dict_elements)) as ArrayRef;

    (array_dict_indices, array_dict_elements)
}

impl ArrayEncoder for DictionaryEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: &mut u32) -> Result<EncodedArray> {
        let (null_count, _row_count) = arrays
            .iter()
            .map(|arr| (arr.null_count() as u32, arr.len() as u32))
            .fold((0, 0), |acc, val| (acc.0 + val.0, acc.1 + val.1));

        if null_count != 0 {
            panic!("Data contains null values, not currently supported for binary data.")
        } else {
            let (index_array, items_array) = get_indices_items_from_arrays(arrays);
            let encoded_indices = self
                .indices_encoder
                .encode(&[index_array.clone()], buffer_index)?;
            println!("dict indices: {:?}", index_array);

            let encoded_items: EncodedArray = self
                .items_encoder
                .encode(&[items_array.clone()], buffer_index)?;
            println!("dict items: {:?}", items_array);

            let mut encoded_buffers = encoded_indices.buffers;
            encoded_buffers.extend(encoded_items.buffers);

            let dict_size = items_array.len() as u32;

            Ok(EncodedArray {
                buffers: encoded_buffers,
                encoding: pb::ArrayEncoding {
                    array_encoding: Some(pb::array_encoding::ArrayEncoding::Dictionary(Box::new(
                        pb::Dictionary {
                            indices: Some(Box::new(encoded_indices.encoding)),
                            items: Some(Box::new(encoded_items.encoding)),
                            size: dict_size,
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
