// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, StructArray};
use futures::{future::BoxFuture, FutureExt};

use crate::{
    decoder::{PageScheduler, PrimitivePageDecoder},
    encoder::{ArrayEncoder, EncodedArray, EncodedArrayBuffer},
    format::pb::{self},
    EncodingsIo,
};

use arrow_buffer::buffer::Buffer;
use arrow_schema::DataType;
use bytes::Bytes;
use bytes::BytesMut;
use lance_core::Result;

#[derive(Debug)]
pub struct PackedStructPageScheduler {
    _inner_schedulers: Vec<Box<dyn PageScheduler>>,
    struct_datatype: DataType,
}

impl PackedStructPageScheduler {
    pub fn new(_inner_schedulers: Vec<Box<dyn PageScheduler>>, struct_datatype: DataType) -> Self {
        Self {
            _inner_schedulers,
            struct_datatype,
        }
    }
}

impl PageScheduler for PackedStructPageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u64>],
        scheduler: &Arc<dyn EncodingsIo>,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>> {
        let DataType::Struct(fields) = &self.struct_datatype else {
            panic!("Struct datatype expected");
        };

        let mut total_bytes_per_row: u64 = 0;

        for field in fields {
            let bits_per_field = get_bits_per_value_from_datatype(field.data_type()) as u64;
            total_bytes_per_row += bits_per_field / 8;
        }

        println!("Ranges: {:?}", ranges);
        let adjusted_ranges = ranges
            .iter()
            .map(|range| {
                let start = range.start * total_bytes_per_row;
                let end = range.end * total_bytes_per_row;
                start..end
            })
            .collect::<Vec<_>>();
        println!("Adjusted ranges: {:?}", adjusted_ranges);

        let bytes = scheduler.submit_request(adjusted_ranges, top_level_row);

        let copy_struct_datatype = self.struct_datatype.clone();

        tokio::spawn(async move {
            let bytes = bytes.await?;

            Ok(Box::new(PackedStructPageDecoder {
                data: bytes,
                struct_datatype: copy_struct_datatype,
                total_bytes_per_row: total_bytes_per_row as usize,
            }) as Box<dyn PrimitivePageDecoder>)
        })
        .map(|join_handle| join_handle.unwrap())
        .boxed()
    }
}

struct PackedStructPageDecoder {
    // inner_decoders: Vec<Box<dyn PrimitivePageDecoder>>,
    data: Vec<Bytes>,
    struct_datatype: DataType,
    total_bytes_per_row: usize,
}

impl PrimitivePageDecoder for PackedStructPageDecoder {
    fn decode(
        &self,
        rows_to_skip: u64,
        num_rows: u64,
        _all_null: &mut bool,
    ) -> Result<Vec<BytesMut>> {
        // e.g.
        // rows 0-2: {x: [1, 2, 3], y: [4, 5, 6], z: [7, 8, 9]}
        // rows 3-5: {x: [10, 11, 12], y: [13, 14, 15], z: [16, 17, 18]}
        // packed encoding: [
        // [1, 4, 7, 2, 5, 8, 3, 6, 9],
        // [10, 13, 16, 11, 14, 17, 12, 15, 18]
        // ]

        println!("Num rows: {:?}", num_rows);
        println!("Rows to skip: {:?}", rows_to_skip);
        println!("Total bytes per row: {:?}", self.total_bytes_per_row);
        println!("Length of Bytes: {:?}", self.data[0].len());
        let DataType::Struct(fields) = &self.struct_datatype else {
            panic!("Struct datatype expected");
        };

        let total_len: usize = self.data.iter().map(|b| b.len()).sum();
        let mut bytes_data = BytesMut::with_capacity(total_len);

        for byte_slice in &self.data {
            bytes_data.extend_from_slice(byte_slice);
        }

        let bytes_data = Bytes::from(bytes_data);

        let mut start_offset = 0;
        let mut struct_bytes = Vec::new();
        for field in fields {
            let bytes_per_field = get_bits_per_value_from_datatype(field.data_type()) / 8;
            let mut field_bytes = BytesMut::default();
            let mut byte_index = start_offset;

            while byte_index + bytes_per_field <= bytes_data.len() {
                println!(
                    "Byte range: {:?} - {:?}",
                    byte_index,
                    byte_index + bytes_per_field
                );
                let byte_slice = bytes_data.slice(byte_index..(byte_index + bytes_per_field));
                field_bytes.extend_from_slice(&byte_slice);
                byte_index += self.total_bytes_per_row;
                println!("Byte index: {:?}", byte_index);
            }

            start_offset += bytes_per_field;
            struct_bytes.push(field_bytes);
        }

        Ok(struct_bytes)
    }

    fn num_buffers(&self) -> u32 {
        let DataType::Struct(fields) = &self.struct_datatype else {
            panic!("Struct datatype expected");
        };

        fields.len() as u32
    }
}

#[derive(Debug)]
pub struct PackedStructEncoder {
    inner_encoders: Vec<Box<dyn ArrayEncoder>>,
}

impl PackedStructEncoder {
    pub fn new(inner_encoders: Vec<Box<dyn ArrayEncoder>>) -> Self {
        Self { inner_encoders }
    }
}

fn get_bits_per_value_from_datatype(datatype: &DataType) -> usize {
    match datatype {
        DataType::UInt64 => 64,
        DataType::UInt32 => 32,
        DataType::UInt16 => 16,
        DataType::UInt8 => 8,
        DataType::Int64 => 64,
        DataType::Int32 => 32,
        DataType::Int16 => 16,
        DataType::Int8 => 8,
        _ => panic!("Unsupported datatype"),
    }
}

fn pack(encoded_fields: Vec<EncodedArray>, fields_bits_per_value: Vec<usize>) -> Vec<Buffer> {
    let num_fields = encoded_fields.len();
    println!("Num fields: {:?}", num_fields);

    // Each EncodedArray can have several EncodedArrayBuffers (e.g. validity, offsets, bytes, etc)
    // Each EncodedArrayBuffer object has several parts. Each part is a Vec<Buffer>
    // The code below assumes that for all fields:
    // (i) Each EncodedArray has only one EncodedArrayBuffer
    // (ii) The total number of buffers across all parts in the EncodedArrayBuffer is the same
    // This workflow will have to change as we adapt the packed encoding to support more complex datatypes
    let num_buffers_per_field: usize = encoded_fields[0].buffers[0]
        .parts
        .iter()
        .map(|buf| buf.len() / (fields_bits_per_value[0] / 8))
        .sum();

    let mut packed_vec: Vec<Option<Buffer>> = vec![None; num_buffers_per_field * num_fields];

    println!("Num encoded fields: {:?}", encoded_fields.len());
    for (field_index, encoded_field) in encoded_fields.iter().enumerate() {
        let bytes_per_value = fields_bits_per_value[field_index] / 8;
        let parts = &encoded_field.buffers[0].parts;
        println!("Parts: {:?}", parts.len());

        let mut packed_global_index = 0;
        for buf in parts {
            println!("Processing buffer: {:?}", buf);
            let num_values = buf.len() / bytes_per_value;
            for value_index in 0..num_values {
                let start = value_index * bytes_per_value;
                let packed_index = packed_global_index * num_fields + field_index;
                println!("Packed index: {:}", packed_index);

                let buffer_slice = Some(buf.slice_with_length(start, bytes_per_value));

                packed_vec[packed_index].clone_from(&buffer_slice);
                println!("added buffer: {:?}", buffer_slice.as_slice());
                packed_global_index += 1;
            }
        }
    }

    packed_vec
        .iter()
        .map(|buf| buf.clone().unwrap())
        .collect::<Vec<Buffer>>()
}

impl ArrayEncoder for PackedStructEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: &mut u32) -> Result<EncodedArray> {
        // let packed_arrays = encode_packed_struct(arrays);
        let num_struct_fields = arrays[0]
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .num_columns();

        let mut encoded_buffers = Vec::new();
        let mut inner_encodings = Vec::new();

        for (arr_index, arr) in arrays.iter().enumerate() {
            let struct_array = arr.as_any().downcast_ref::<StructArray>().unwrap();

            let mut encoded_fields = Vec::new();
            let mut field_bits_per_value = Vec::new();

            for field_index in 0..num_struct_fields {
                let field_datatype = struct_array.column(field_index).data_type();
                let field_array = struct_array.column(field_index).clone();

                // Compute encoded inner arrays
                let encoded_field =
                    self.inner_encoders[field_index].encode(&[field_array], &mut 0)?;
                let field_buffers = encoded_field.clone().buffers;

                encoded_fields.push(encoded_field.clone());

                // We assume there is only one outer buffer per field
                assert_eq!(field_buffers.len(), 1);

                // Compute bits per value for each field
                let bits_per_value = get_bits_per_value_from_datatype(field_datatype);
                field_bits_per_value.push(bits_per_value);

                if arr_index == 0 {
                    inner_encodings.push(encoded_field.encoding);
                }
            }

            let packed_vec = pack(encoded_fields, field_bits_per_value);

            let packed_buffer = EncodedArrayBuffer {
                parts: packed_vec,
                index: 0,
            };

            encoded_buffers.push(packed_buffer);
        }

        Ok(EncodedArray {
            buffers: encoded_buffers,
            encoding: pb::ArrayEncoding {
                array_encoding: Some(pb::array_encoding::ArrayEncoding::PackedStruct(
                    pb::PackedStruct {
                        inner: inner_encodings,
                        num_struct_fields: num_struct_fields as u32,
                    },
                )),
            },
        })
    }
}

#[cfg(test)]
pub mod tests {

    use arrow_array::{ArrayRef, Int32Array, StructArray, UInt64Array, UInt8Array};
    use arrow_schema::{DataType, Field, Fields};
    use std::{collections::HashMap, sync::Arc, vec};

    use crate::testing::{
        check_round_trip_encoding_of_data_with_metadata,
        check_round_trip_encoding_random_with_metadata, TestCases,
    };

    #[test_log::test(tokio::test)]
    async fn test_random_packed_struct() {
        let data_type = DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::UInt64, false),
            Field::new("b", DataType::UInt32, false),
        ]));
        let field = Field::new("", data_type, false);

        let mut metadata = HashMap::new();
        metadata.insert("packed".to_string(), "true".to_string());

        check_round_trip_encoding_random_with_metadata(field, metadata).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_specific_packed_struct() {
        let array1 = Arc::new(UInt64Array::from(vec![1, 2, 3, 4]));
        let array2 = Arc::new(Int32Array::from(vec![5, 6, 7, 8]));
        let array3 = Arc::new(UInt8Array::from(vec![9, 10, 11, 12]));

        let struct_array1 = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("x", DataType::UInt64, false)),
                array1.clone() as ArrayRef,
            ),
            (
                Arc::new(Field::new("y", DataType::Int32, false)),
                array2.clone() as ArrayRef,
            ),
            (
                Arc::new(Field::new("z", DataType::UInt8, false)),
                array3.clone() as ArrayRef,
            ),
        ]));

        let array4 = Arc::new(UInt64Array::from(vec![13, 14, 15, 16]));
        let array5 = Arc::new(Int32Array::from(vec![17, 18, 19, 20]));
        let array6 = Arc::new(UInt8Array::from(vec![21, 22, 23, 24]));

        let struct_array2 = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("x", DataType::UInt64, false)),
                array4.clone() as ArrayRef,
            ),
            (
                Arc::new(Field::new("y", DataType::Int32, false)),
                array5.clone() as ArrayRef,
            ),
            (
                Arc::new(Field::new("z", DataType::UInt8, false)),
                array6.clone() as ArrayRef,
            ),
        ]));

        let test_cases = TestCases::default()
            .with_range(0..2)
            .with_range(0..6)
            .with_range(1..4)
            .with_indices(vec![1, 3, 7]);

        let mut metadata = HashMap::new();
        metadata.insert("packed".to_string(), "true".to_string());

        check_round_trip_encoding_of_data_with_metadata(
            vec![struct_array1, struct_array2],
            &test_cases,
            metadata,
        )
        .await;
    }
}
