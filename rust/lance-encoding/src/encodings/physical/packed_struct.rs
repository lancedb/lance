// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{cast::AsArray, types::UInt64Type, Array, ArrayRef, StructArray, UInt64Array};
use futures::{future::BoxFuture, FutureExt};

use crate::{
    decoder::{PageScheduler, PrimitivePageDecoder},
    encoder::{ArrayEncoder, EncodedArray},
    format::pb,
    EncodingsIo,
};

use arrow_schema::DataType;
use bytes::BytesMut;
use lance_core::Result;

#[derive(Debug)]
pub struct PackedStructPageScheduler {
    inner_scheduler: Arc<dyn PageScheduler>,
    num_struct_fields: u64,
    struct_datatype: DataType,
}

impl PackedStructPageScheduler {
    pub fn new(
        inner_scheduler: Arc<dyn PageScheduler>,
        num_struct_fields: u64,
        struct_datatype: DataType,
    ) -> Self {
        Self {
            inner_scheduler,
            num_struct_fields,
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
        // Schedule inner values for decoding
        let adjusted_ranges = ranges
            .iter()
            .map(|range| {
                let start = range.start * self.num_struct_fields;
                let end = range.end * self.num_struct_fields;
                start..end
            })
            .collect::<Vec<_>>();

        let inner_page_decoder =
            self.inner_scheduler
                .schedule_ranges(&adjusted_ranges, scheduler, top_level_row);

        // let copy_num_struct_fields = self.num_struct_fields.clone();
        let copy_struct_datatype = self.struct_datatype.clone();

        tokio::spawn(async move {
            let inner_decoder = Arc::from(inner_page_decoder.await?);

            Ok(Box::new(PackedStructPageDecoder {
                inner_decoder,
                struct_datatype: copy_struct_datatype,
            }) as Box<dyn PrimitivePageDecoder>)
        })
        .map(|join_handle| join_handle.unwrap())
        .boxed()
    }
}

struct PackedStructPageDecoder {
    inner_decoder: Arc<dyn PrimitivePageDecoder>,
    struct_datatype: DataType,
}

impl PrimitivePageDecoder for PackedStructPageDecoder {
    fn decode(
        &self,
        rows_to_skip: u64,
        num_rows: u64,
        all_null: &mut bool,
    ) -> Result<Vec<BytesMut>> {
        // e.g.
        // rows 0-2: {x: [1, 2, 3], y: [4, 5, 6], z: [7, 8, 9]}
        // rows 3-5: {x: [10, 11, 12], y: [13, 14, 15], z: [16, 17, 18]}
        // packed encoding: [
        // [1, 4, 7, 2, 5, 8, 3, 6, 9],
        // [10, 13, 16, 11, 14, 17, 12, 15, 18]
        // ]
        // If user asks for rows i..j, we should decode num_struct_fields*i..num_struct_fields*j

        let DataType::Struct(fields) = &self.struct_datatype else {
            panic!("Struct datatype expected");
        };

        let num_struct_fields = fields.len() as u64;

        let inner_buffers = self.inner_decoder.decode(
            rows_to_skip * num_struct_fields,
            num_rows * num_struct_fields,
            all_null,
        )?;

        Ok(inner_buffers)
    }

    fn num_buffers(&self) -> u32 {
        self.inner_decoder.num_buffers() + 2
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

// Encodes the struct in a packed row format
// TODO: support different inner datatypes for the struct
// Right now only supports struct of uint64
// fn encode_packed_struct(arrays: &[ArrayRef]) -> Vec<ArrayRef> {
//     let mut packed_vec = Vec::new();
//     for arr in arrays {
//         let struct_array = arr.as_any().downcast_ref::<StructArray>().unwrap();
//         // let inner_datatype = struct_array.column(0).data_type();

//         let num_rows = struct_array.len();
//         let num_columns = struct_array.num_columns();

//         let mut packed_row = Vec::with_capacity(num_rows * num_columns);

//         for row in 0..num_rows {
//             for column in struct_array.columns() {
//                 let inner_array = column.as_any().downcast_ref::<UInt64Array>().unwrap();
//                 packed_row.push(inner_array.value(row));
//             }
//         }

//         let packed_array = Arc::new(UInt64Array::from(packed_row)) as ArrayRef;
//         packed_vec.push(packed_array);
//     }

//     packed_vec
// }

fn get_bits_per_value_from_datatype(datatype: &DataType) -> u64 {
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

fn pack(encoded_fields: Vec<EncodedArray>, children_bits_per_value: Vec<u64>) {
    // let num_rows = encoded_children[0].buffers[0].len() as u64;
    let num_fields = encoded_fields.len() as u64;

    // let mut packed_buffers = Vec::new();
    for i in 0..num_fields {
        // let encoded_field = &encoded_fields[i as usize].into_parts(); // encoded array for field i

        let encoded_field = &encoded_fields[i as usize];
        let child_encoded_buffer = &encoded_field.buffers[0].parts; // encoded buffers for field i
        // let child_buffer_index = encoded_child.encoding; // encoding for field i

        // let page_index = child_encoded_buffer
    }
}

impl ArrayEncoder for PackedStructEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: &mut u32) -> Result<EncodedArray> {
        // let packed_arrays = encode_packed_struct(arrays);
        let num_struct_fields = arrays[0]
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .num_columns();
        
        for arr in arrays {
            println!("Processing array: {:?}", arr);
            let struct_array = arr.as_any().downcast_ref::<StructArray>().unwrap();

            let mut encoded_children = Vec::new();
            let mut children_bits_per_value = Vec::new();
            for i in 0..num_struct_fields {
                let field_datatype = struct_array.column(i).data_type();
                
                let child_array = struct_array.column(i).clone();
                let encoded_child = self.inner_encoders[i].encode(&[child_array.clone()], &mut 0)?;
                let child_buffers = encoded_child.buffers;
                println!("Length of child {:?}: {:?}", i, child_buffers.len());

                let bits_per_value = get_bits_per_value_from_datatype(field_datatype);
                children_bits_per_value.push(bits_per_value);

                for j in 0..child_buffers.len() {
                    println!("Buffer: {:?}", child_buffers[j].parts);
                }
                let encoded_child = self.inner_encoders[i].encode(&[child_array], &mut 0)?;
                encoded_children.push(encoded_child);

                // let encoded_buffer = encoded_child.buffers;
            }

            // let packed_array = pack(encoded_children);
        }

        // println!("Packed arrays: {:?}", packed_arrays.clone());
        let encoded_packed_struct = self.inner_encoders[0].encode(&vec![], buffer_index)?;
        let encoded_buffers = encoded_packed_struct.buffers;

        Ok(EncodedArray {
            buffers: encoded_buffers,
            encoding: pb::ArrayEncoding {
                array_encoding: Some(pb::array_encoding::ArrayEncoding::PackedStruct(Box::new(
                    pb::PackedStruct {
                        inner: Some(Box::new(encoded_packed_struct.encoding)),
                        num_struct_fields: num_struct_fields as u32,
                    },
                ))),
            },
        })
    }
}

#[cfg(test)]
pub mod tests {

    use arrow_array::{
        builder::{LargeStringBuilder, StringBuilder}, ArrayRef, Int32Array, StringArray, StructArray, UInt64Array
    };
    use arrow_schema::{DataType, Field, Fields};
    use std::{collections::HashMap, sync::Arc, vec};

    use crate::testing::{
        check_round_trip_encoding_of_data, check_round_trip_encoding_of_data_with_metadata, check_round_trip_encoding_random, TestCases
    };

    #[test_log::test(tokio::test)]
    async fn test_random_packed_struct() {
        let data_type = DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::UInt64, false),
            Field::new("b", DataType::UInt64, false),
        ]));
        let field = Field::new("", data_type, false);
        check_round_trip_encoding_random(field).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_specific_packed_struct() {
        let array1 = Arc::new(UInt64Array::from(vec![1, 2, 3, 4]));
        let array2 = Arc::new(Int32Array::from(vec![5, 6, 7, 8]));

        let struct_array1 = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("x", DataType::UInt64, false)),
                array1.clone() as ArrayRef,
            ),
            (
                Arc::new(Field::new("y", DataType::Int32, false)),
                array2.clone() as ArrayRef,
            ),
        ]));

        let array3 = Arc::new(UInt64Array::from(vec![10, 11, 12, 13]));
        let array4 = Arc::new(Int32Array::from(vec![14, 15, 16, 17]));

        let struct_array2 = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("x", DataType::UInt64, false)),
                array3.clone() as ArrayRef,
            ),
            (
                Arc::new(Field::new("y", DataType::Int32, false)),
                array4.clone() as ArrayRef,
            ),
        ]));

        let test_cases = TestCases::default()
            .with_range(0..2)
            .with_range(0..6)
            .with_range(1..5)
            .with_indices(vec![1, 3, 7]);

        let mut metadata = HashMap::new();
        metadata.insert("packed".to_string(), "true".to_string());

        check_round_trip_encoding_of_data_with_metadata(vec![struct_array1, struct_array2], &test_cases, metadata).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_utf8() {
        let field = Field::new("", DataType::Utf8, false);
        check_round_trip_encoding_random(field).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_binary() {
        let field = Field::new("", DataType::Binary, false);
        check_round_trip_encoding_random(field).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_large_binary() {
        let field = Field::new("", DataType::LargeBinary, true);
        check_round_trip_encoding_random(field).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_large_utf8() {
        let field = Field::new("", DataType::LargeUtf8, true);
        check_round_trip_encoding_random(field).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_simple_utf8() {
        let string_array = StringArray::from(vec![Some("abc"), Some("de"), None, Some("fgh")]);

        let test_cases = TestCases::default()
            .with_range(0..2)
            .with_range(0..3)
            .with_range(1..3)
            .with_indices(vec![1, 3]);
        check_round_trip_encoding_of_data(vec![Arc::new(string_array)], &test_cases).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_sliced_utf8() {
        let string_array = StringArray::from(vec![Some("abc"), Some("de"), None, Some("fgh")]);
        let string_array = string_array.slice(1, 3);

        let test_cases = TestCases::default()
            .with_range(0..1)
            .with_range(0..2)
            .with_range(1..2);
        check_round_trip_encoding_of_data(vec![Arc::new(string_array)], &test_cases).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_empty_strings() {
        // Scenario 1: Some strings are empty

        let values = [Some("abc"), Some(""), None];
        // Test empty list at beginning, middle, and end
        for order in [[0, 1, 2], [1, 0, 2], [2, 0, 1]] {
            let mut string_builder = StringBuilder::new();
            for idx in order {
                string_builder.append_option(values[idx]);
            }
            let string_array = Arc::new(string_builder.finish());
            let test_cases = TestCases::default()
                .with_indices(vec![1])
                .with_indices(vec![0])
                .with_indices(vec![2]);
            check_round_trip_encoding_of_data(vec![string_array.clone()], &test_cases).await;
            let test_cases = test_cases.with_batch_size(1);
            check_round_trip_encoding_of_data(vec![string_array], &test_cases).await;
        }

        // Scenario 2: All strings are empty

        // When encoding an array of empty strings there are no bytes to encode
        // which is strange and we want to ensure we handle it
        let string_array = Arc::new(StringArray::from(vec![Some(""), None, Some("")]));

        let test_cases = TestCases::default().with_range(0..2).with_indices(vec![1]);
        check_round_trip_encoding_of_data(vec![string_array.clone()], &test_cases).await;
        let test_cases = test_cases.with_batch_size(1);
        check_round_trip_encoding_of_data(vec![string_array], &test_cases).await;
    }

    #[test_log::test(tokio::test)]
    #[ignore] // This test is quite slow in debug mode
    async fn test_jumbo_string() {
        // This is an overflow test.  We have a list of lists where each list
        // has 1Mi items.  We encode 5000 of these lists and so we have over 4Gi in the
        // offsets range
        let mut string_builder = LargeStringBuilder::new();
        // a 1 MiB string
        let giant_string = String::from_iter((0..(1024 * 1024)).map(|_| '0'));
        for _ in 0..5000 {
            string_builder.append_option(Some(&giant_string));
        }
        let giant_array = Arc::new(string_builder.finish()) as ArrayRef;
        let arrs = vec![giant_array];

        // // We can't validate because our validation relies on concatenating all input arrays
        let test_cases = TestCases::default().without_validation();
        check_round_trip_encoding_of_data(arrs, &test_cases).await;
    }
}
