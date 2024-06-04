// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use core::panic;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, StringArray, UInt32Array, UInt8Array, Int32Array};
use arrow_buffer::BooleanBuffer;
use futures::{future::BoxFuture, FutureExt};
use log::trace;
// use rand::seq::index;

use crate::{
    decoder::{PhysicalPageDecoder, PageScheduler},
    encoder::{ArrayEncoder, BufferEncoder, EncodedArray, EncodedArrayBuffer},
    format::pb,
    EncodingsIo,
};

use lance_core::Result;
use arrow_schema::DataType;
use arrow_cast::cast::cast;

use super::buffers::BitmapBufferEncoder;

#[derive(Debug)]
pub struct BinaryPageScheduler {
    indices_scheduler: Box<dyn PageScheduler>,
    bytes_scheduler: Box<dyn PageScheduler>,
}

impl BinaryPageScheduler {
    pub fn new(indices_scheduler: Box<dyn PageScheduler>, bytes_scheduler: Box<dyn PageScheduler>) -> Self {
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
        scheduler: &dyn EncodingsIo,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PhysicalPageDecoder>>> {

        let indices_page_decoder =
            self.indices_scheduler
                .schedule_ranges(&ranges, scheduler, top_level_row);
        
        let bytes_page_decoder = 
            self.bytes_scheduler
                .schedule_ranges(&ranges, scheduler, top_level_row);
        
        async move {
            let indices_decoder = indices_page_decoder.await?;
            let bytes_decoder = bytes_page_decoder.await?;

            Ok(Box::new(BinaryPageDecoder {
                indices_decoder,
                bytes_decoder,
            }) as Box<dyn PhysicalPageDecoder>)
        }
        .boxed()
    }
}

struct BinaryPageDecoder {
    indices_decoder: Box<dyn PhysicalPageDecoder>,
    bytes_decoder: Box<dyn PhysicalPageDecoder>,
}

impl PhysicalPageDecoder for BinaryPageDecoder {
    fn update_capacity(
        &self,
        rows_to_skip: u32,
        num_rows: u32,
        buffers: &mut [(u64, bool)],
        all_null: &mut bool,
    ) {

        self.indices_decoder
            .update_capacity(rows_to_skip, num_rows, buffers, all_null);
        self.bytes_decoder
            .update_capacity(rows_to_skip, num_rows, buffers, all_null);
    }

    fn decode_into(
        &self,
        rows_to_skip: u32,
        num_rows: u32,
        dest_buffers: &mut [bytes::BytesMut],
    ) -> Result<()> {
        
        self.indices_decoder.decode_into(rows_to_skip, num_rows, &mut dest_buffers[..1])?;
        self.bytes_decoder.decode_into(rows_to_skip, num_rows, &mut dest_buffers[1..])?;

        Ok(())
    }

    fn num_buffers(&self) -> u32 {
        self.indices_decoder.num_buffers() + self.bytes_decoder.num_buffers()
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

                let offsets = string_arr.value_offsets().to_vec();

                let new_int_arr = Int32Array::from(offsets);
                match cast(&new_int_arr, &DataType::UInt64) {
                    Ok(res) => Some(Arc::new(res) as ArrayRef),
                    Err(_) => panic!("Failed to cast to uint64"),
                }
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
        }
        else {
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
