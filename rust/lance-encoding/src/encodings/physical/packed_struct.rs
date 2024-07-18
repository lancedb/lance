// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, StructArray};
use futures::{future::BoxFuture, FutureExt};
use lance_arrow::DataTypeExt;

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
    buffer_offset: u64,
}

impl PackedStructPageScheduler {
    pub fn new(
        _inner_schedulers: Vec<Box<dyn PageScheduler>>,
        struct_datatype: DataType,
        buffer_offset: u64,
    ) -> Self {
        Self {
            _inner_schedulers,
            struct_datatype,
            buffer_offset,
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
            let bytes_per_field = field.data_type().byte_width() as u64;
            total_bytes_per_row += bytes_per_field;
        }

        // To account for potential slicing of the arrays
        // i.e. parts of the arrays may be encoded in different encoding tasks
        // In that case ranges won't be enough to slice the bytes on it's own
        // We need to know the position of the buffer in the page (i.e. the buffer offset)
        let byte_ranges = ranges
            .iter()
            .map(|range| {
                let start = self.buffer_offset + (range.start * total_bytes_per_row);
                let end = self.buffer_offset + (range.end * total_bytes_per_row);
                start..end
            })
            .collect::<Vec<_>>();

        // Directly creates a future to decode the bytes
        let bytes = scheduler.submit_request(byte_ranges, top_level_row);

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
        // Decoding workflow:
        // rows 0-2: {x: [1, 2, 3], y: [4, 5, 6], z: [7, 8, 9]}
        // rows 3-5: {x: [10, 11, 12], y: [13, 14, 15], z: [16, 17, 18]}
        // packed encoding: [
        // [1, 4, 7, 2, 5, 8, 3, 6, 9],
        // [10, 13, 16, 11, 14, 17, 12, 15, 18]
        // ]
        // suppose bytes_per_field=1, 4, 8 for fields x, y, and z, respectively.
        // Then total_bytes_per_row = 13
        // Suppose rows_to_skip=1 and num_rows=2. Then we will slice bytes 13 to 39.
        // Now we have [2, 5, 8, 3, 6, 9]
        // We rearrange this to get [BytesMut(2, 3), BytesMut(5, 6), BytesMut(8, 9)] as a Vec<BytesMut>
        // This is used to reconstruct the struct array later

        let DataType::Struct(fields) = &self.struct_datatype else {
            panic!("Struct datatype expected");
        };

        let bytes_to_skip = (rows_to_skip as usize) * self.total_bytes_per_row;
        let bytes_to_take = (num_rows as usize) * self.total_bytes_per_row;

        let mut bytes = BytesMut::default();
        for byte_slice in &self.data {
            bytes.extend_from_slice(byte_slice);
        }

        let bytes = Bytes::from(bytes);
        let bytes = bytes.slice(bytes_to_skip..(bytes_to_skip + bytes_to_take));

        let mut struct_bytes = Vec::new();

        let mut start_index = 0;
        for field in fields {
            let bytes_per_field = field.data_type().byte_width();
            let mut field_bytes = BytesMut::default();

            let mut byte_index = start_index;

            for _ in 0..num_rows {
                field_bytes
                    .extend_from_slice(&bytes.slice(byte_index..(byte_index + bytes_per_field)));
                byte_index += self.total_bytes_per_row;
            }

            start_index += bytes_per_field;
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

fn pack(encoded_fields: Vec<EncodedArray>, fields_bytes_per_value: Vec<usize>) -> Vec<Buffer> {
    let num_fields = encoded_fields.len();

    // Each EncodedArray can have several EncodedArrayBuffers (e.g. validity, offsets, bytes, etc)
    // Each EncodedArrayBuffer object has several parts. Each part is a Vec<Buffer>
    // The code below assumes that for all fields:
    // (i) Each EncodedArray has only one EncodedArrayBuffer
    // (ii) The total number of buffers across all parts in the EncodedArrayBuffer is the same
    // This workflow will have to change as we adapt the packed encoding to support more complex datatypes
    let num_buffers_per_field: usize = encoded_fields[0].buffers[0]
        .parts
        .iter()
        .map(|buf| buf.len() / fields_bytes_per_value[0])
        .sum();

    let mut packed_vec: Vec<Option<Buffer>> = vec![None; num_buffers_per_field * num_fields];

    for (field_index, encoded_field) in encoded_fields.iter().enumerate() {
        let bytes_per_value = fields_bytes_per_value[field_index];
        let parts = &encoded_field.buffers[0].parts;

        let mut packed_global_index = 0;
        for buf in parts {
            let num_values = buf.len() / bytes_per_value;
            for value_index in 0..num_values {
                let start = value_index * bytes_per_value;
                let packed_index = packed_global_index * num_fields + field_index;

                let buffer_slice = Some(buf.slice_with_length(start, bytes_per_value));

                packed_vec[packed_index].clone_from(&buffer_slice);
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
        let num_struct_fields = arrays[0]
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .num_columns();

        let mut inner_encodings = Vec::new();
        let mut global_packed_vec: Vec<Buffer> = Vec::new();

        for (arr_index, arr) in arrays.iter().enumerate() {
            let struct_array = arr.as_any().downcast_ref::<StructArray>().unwrap();

            let mut encoded_fields = Vec::new();
            let mut field_bytes_per_value = Vec::new();

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

                // Compute bytes per value for each field
                let bytes_per_value = field_datatype.byte_width();
                field_bytes_per_value.push(bytes_per_value);

                if arr_index == 0 {
                    inner_encodings.push(encoded_field.encoding);
                }
            }

            let packed_vec = pack(encoded_fields, field_bytes_per_value);
            global_packed_vec.extend(packed_vec);
        }

        let index = *buffer_index;
        *buffer_index += 1;

        let packed_buffer = EncodedArrayBuffer {
            parts: global_packed_vec,
            index,
        };

        Ok(EncodedArray {
            buffers: vec![packed_buffer],
            encoding: pb::ArrayEncoding {
                array_encoding: Some(pb::array_encoding::ArrayEncoding::PackedStruct(
                    pb::PackedStruct {
                        inner: inner_encodings,
                        buffer_index: index,
                    },
                )),
            },
        })
    }
}

#[cfg(test)]
pub mod tests {

    use arrow::array::ArrayData;
    use arrow_array::{
        Array, ArrayRef, FixedSizeListArray, Int32Array, StructArray, UInt64Array, UInt8Array,
    };
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

    #[test_log::test(tokio::test)]
    async fn test_fsl_packed_struct() {
        let int_array = Int32Array::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        let array2 = Arc::new(Int32Array::from(vec![12, 13, 14, 15]));

        let list_data_type =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 3);
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(4)
            .add_child_data(int_array.into_data())
            .build()
            .unwrap();
        let list_array = FixedSizeListArray::from(list_data);

        let struct_array = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("x", list_data_type.clone(), false)),
                Arc::new(list_array) as ArrayRef,
            ),
            (
                Arc::new(Field::new("x", DataType::Int32, false)),
                array2 as ArrayRef,
            ),
        ]));

        let test_cases = TestCases::default()
            .with_range(1..3)
            .with_range(0..1)
            .with_range(2..4)
            .with_indices(vec![0, 2, 3]);

        let mut metadata = HashMap::new();
        metadata.insert("packed".to_string(), "true".to_string());

        check_round_trip_encoding_of_data_with_metadata(vec![struct_array], &test_cases, metadata)
            .await;
    }
}
