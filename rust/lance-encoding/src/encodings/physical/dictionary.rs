// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use core::panic;
use std::sync::Arc;

use arrow_array::types::UInt64Type;
use arrow_array::{Array, ArrayRef, DictionaryArray, StringArray, UInt64Array};
use futures::{future::BoxFuture, FutureExt};

use crate::{
    decoder::{PageScheduler, PrimitivePageDecoder},
    encoder::{ArrayEncoder, EncodedArray},
    format::pb,
    EncodingsIo,
};

use crate::decoder::LogicalPageDecoder;
use crate::encodings::logical::primitive::PrimitiveFieldDecoder;

use arrow_schema::DataType;
use bytes::BytesMut;
use lance_core::Result;
use std::collections::HashMap;

use crate::encodings::utils::new_primitive_array;
use arrow_array::cast::AsArray;

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
    ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>> {
        // We want to decode indices and items
        // e.g. indices [0, 1, 2, 0, 1, 0]
        // items (dictionary) ["abcd", "hello", "apple"]
        // This will map to ["abcd", "hello", "apple", "abcd", "hello", "abcd"]
        // We decode all the items during scheduling itself
        // These are used to rebuild the string later

        // Schedule indices for decoding
        let indices_page_decoder =
            self.indices_scheduler
                .schedule_ranges(ranges, scheduler, top_level_row);

        // Schedule items for decoding
        let items_page_decoder =
            self.items_scheduler
                .schedule_ranges(&[0..self.size], scheduler, top_level_row);

        let copy_size = self.size;

        async move {
            let items_decoder: Arc<dyn PrimitivePageDecoder> = Arc::from(items_page_decoder.await?);

            let mut primitive_wrapper = PrimitiveFieldDecoder::new_from_data(
                items_decoder.clone(),
                DataType::Utf8,
                copy_size,
            );

            // Decode all items
            let drained_task = primitive_wrapper.drain(copy_size)?;
            let items_decode_task = drained_task.task;
            let decoded_dict = items_decode_task.decode()?;

            let indices_decoder: Box<dyn PrimitivePageDecoder> = indices_page_decoder.await?;

            Ok(Box::new(DictionaryPageDecoder {
                decoded_dict,
                indices_decoder,
                items_decoder,
            }) as Box<dyn PrimitivePageDecoder>)
        }
        .boxed()
    }
}

struct DictionaryPageDecoder {
    decoded_dict: Arc<dyn Array>,
    indices_decoder: Box<dyn PrimitivePageDecoder>,
    items_decoder: Arc<dyn PrimitivePageDecoder>,
}

impl PrimitivePageDecoder for DictionaryPageDecoder {
    fn decode(
        &self,
        rows_to_skip: u32,
        num_rows: u32,
        all_null: &mut bool,
    ) -> Result<Vec<BytesMut>> {
        // Decode the indices
        let indices_buffers = self
            .indices_decoder
            .decode(rows_to_skip, num_rows, all_null)?;

        let indices_array =
            new_primitive_array::<UInt64Type>(indices_buffers.clone(), num_rows, &DataType::UInt64);
        let indices_array = indices_array.as_primitive::<UInt64Type>().clone();

        let dictionary = self.decoded_dict.clone();

        // Build dictionary array using indices and items
        let dict_array = DictionaryArray::<UInt64Type>::try_new(indices_array, dictionary).unwrap();
        let str_array = arrow_cast::cast(&dict_array, &DataType::Utf8).unwrap();
        let string_arr = str_array.as_any().downcast_ref::<StringArray>().unwrap();

        let (offsets, bytes, nulls) = string_arr.clone().into_parts();

        let offsets = offsets.inner().inner().as_slice();
        let bytes = bytes.as_slice();

        let final_nulls = nulls.map_or(BytesMut::default(), |nb| {
            let nulls = nb.buffer().as_slice();
            BytesMut::from(nulls)
        });

        let final_offsets = BytesMut::from(offsets);
        let final_bytes = BytesMut::from(bytes);

        Ok(vec![final_offsets, final_nulls, final_bytes])
    }

    fn num_buffers(&self) -> u32 {
        self.items_decoder.num_buffers()
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
                dict_indices.push(*arr_hashmap.get(st).unwrap());
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

            let encoded_items: EncodedArray = self
                .items_encoder
                .encode(&[items_array.clone()], buffer_index)?;

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
