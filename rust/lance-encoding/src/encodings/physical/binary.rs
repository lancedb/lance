// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, StringArray, UInt32Array, UInt8Array};
use arrow_buffer::BooleanBuffer;
use futures::{future::BoxFuture, FutureExt};
use log::trace;
// use rand::seq::index;

use crate::{
    decoder::{PhysicalPageDecoder, PhysicalPageScheduler},
    encoder::{ArrayEncoder, BufferEncoder, EncodedArray, EncodedArrayBuffer},
    format::pb,
    EncodingsIo,
};

use lance_core::Result;

use super::buffers::BitmapBufferEncoder;

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

fn get_indices_from_string_arrays(arrays: &[ArrayRef]) -> Vec<ArrayRef> {
    let index_arrays: Vec<ArrayRef> = arrays
        .iter()
        .filter_map(|arr| {
            if let Some(string_arr) = arr.as_any().downcast_ref::<StringArray>() {
                let mut int_arr = Vec::new();

                let mut cum_sz: u32 = 0;
                for i in 0..string_arr.len() {
                    let s = string_arr.value(i);
                    let sz = s.len() as u32;
                    cum_sz += sz;
                    int_arr.push(cum_sz);
                }
                // Arc::new(UInt32Array::from(int_arr));
                Some(Arc::new(UInt32Array::from(int_arr)) as ArrayRef)
            } else {
                None
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
                let mut byte_arr = Vec::new();

                for i in 0..string_arr.len() {
                    let s = string_arr.value(i).as_bytes();
                    byte_arr.extend_from_slice(s);
                }
                Some(Arc::new(UInt8Array::from(byte_arr)) as ArrayRef)
            } else {
                None
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

        if null_count == 0 {
            let index_arrays = get_indices_from_string_arrays(arrays);
            let encoded_indices = self.indices_encoder.encode(&index_arrays, buffer_index)?;
            // for arr in &index_arrays {
            //     println!("indices: {:?}", arr);
            // }

            let byte_arrays = get_bytes_from_string_arrays(arrays);
            // for arr in &byte_arrays {
            //     println!("arr: {:?}", arr);
            // }
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
        else {
            let nullability = pb::nullable::Nullability::AllNulls(pb::nullable::AllNull {});

            Ok(EncodedArray {
                buffers: vec![],
                encoding: pb::ArrayEncoding {
                    array_encoding: Some(pb::array_encoding::ArrayEncoding::Nullable(Box::new(
                        pb::Nullable {
                            nullability: Some(nullability),
                        },
                    ))),
                },
            })
        }

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
