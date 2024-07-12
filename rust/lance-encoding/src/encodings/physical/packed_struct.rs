// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::builder::{ArrayBuilder, StringBuilder};
use arrow_array::types::{UInt64Type, UInt8Type};
use arrow_array::{
    Array, ArrayRef, DictionaryArray, PrimitiveArray, StringArray, StructArray, UInt64Array,
    UInt8Array,
};
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

use crate::encodings::utils::{new_primitive_array, primitive_array_from_buffers};
use arrow_array::cast::AsArray;

#[derive(Debug)]
pub struct PackedStructPageScheduler {
    inner_scheduler: Arc<dyn PageScheduler>,
    num_elements_per_field: u64,
    struct_datatype: DataType,
}

impl PackedStructPageScheduler {
    pub fn new(
        inner_scheduler: Arc<dyn PageScheduler>,
        num_elements_per_field: u64,
        struct_datatype: DataType,
    ) -> Self {
        Self {
            inner_scheduler,
            num_elements_per_field,
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
        println!("Ranges: {:?}", ranges);

        let inner_page_decoder =
            self.inner_scheduler
                .schedule_ranges(&ranges, scheduler, top_level_row);

        let copy_num_elements_per_field = self.num_elements_per_field.clone();
        let copy_struct_datatype = self.struct_datatype.clone();

        tokio::spawn(async move {
            let inner_decoder = Arc::from(inner_page_decoder.await?);

            Ok(Box::new(PackedStructPageDecoder {
                inner_decoder,
                num_elements_per_field: copy_num_elements_per_field,
                struct_datatype: copy_struct_datatype,
            }) as Box<dyn PrimitivePageDecoder>)
        })
        .map(|join_handle| join_handle.unwrap())
        .boxed()
    }
}

struct PackedStructPageDecoder {
    inner_decoder: Arc<dyn PrimitivePageDecoder>,
    num_elements_per_field: u64,
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
        // row 0 {x: [1, 2, 3], y: [4, 5, 6], z: [7, 8, 9]}
        // row 1 {a: [10, 11, 12], b: [13, 14, 15], c: [16, 17, 18]}
        // packed encoding: [
        // [1, 4, 7, 2, 5, 8, 3, 6, 9],
        // [10, 13, 16, 11, 14, 17, 12, 15, 18]
        // ]
        // If user asks for row #1, we should decode elements from num_struct_elements*1 to num_struct_elements*2

        let DataType::Struct(fields) = &self.struct_datatype else {
            panic!("Struct datatype expected");
        };
        
        let num_struct_fields = fields.len();
        let num_elements_per_struct = (num_struct_fields as u64) * self.num_elements_per_field; 

        println!("Num struct fields: {:?}", num_struct_fields);
        println!("Num elements per field: {:?}", self.num_elements_per_field);
        println!("Num elements per struct: {:?}", num_elements_per_struct);
        println!("Rows to skip: {:?}", rows_to_skip);
        println!("Num rows: {:?}", num_rows);

        let inner_buffers = self
            .inner_decoder
            .decode(rows_to_skip*num_elements_per_struct, num_rows*num_elements_per_struct, all_null)?;

        let num_fields = fields.len();
        let inner_datatype = fields[0].data_type();
        let packed_array = primitive_array_from_buffers(inner_datatype, inner_buffers, num_rows).unwrap();
        let inner_array = packed_array.as_any().downcast_ref::<UInt64Array>().unwrap();
        println!("Inner array: {:?}", inner_array);

        let mut child_vecs = vec![Vec::new(); num_fields];

        for (i, value) in inner_array.iter().enumerate() {
            if let Some(v) = value {
                child_vecs[i % num_fields].push(v);
            }
        }

        let child_arrays = child_vecs
            .into_iter()
            .map(|field_data| Arc::new(PrimitiveArray::from(field_data)) as ArrayRef)
            .collect::<Vec<_>>();

        for arr in &child_arrays {
            println!("Child array: {:?}", arr);
        }

        let struct_array = Arc::new(StructArray::try_new(fields.clone(), child_arrays, None).unwrap());

        // Ok(inner_buffers)
        Ok(vec![])
    }

    fn num_buffers(&self) -> u32 {
        self.inner_decoder.num_buffers() + 2
    }
}

#[derive(Debug)]
pub struct PackedStructEncoder {
    inner_encoder: Box<dyn ArrayEncoder>,
}

impl PackedStructEncoder {
    pub fn new(inner_encoder: Box<dyn ArrayEncoder>) -> Self {
        Self { inner_encoder }
    }
}

// Encodes the struct in a packed row format
// TODO: support different inner datatypes for the struct
// Right now only supports struct of uint64
fn encode_packed_struct(arrays: &[ArrayRef]) -> Vec<ArrayRef> {
    let mut packed_vec = Vec::new();
    for arr in arrays {
        let struct_array = arr.as_any().downcast_ref::<StructArray>().unwrap();
        // let inner_datatype = struct_array.column(0).data_type();

        let num_rows = struct_array.len();
        let num_columns = struct_array.num_columns();

        let mut packed_row = Vec::with_capacity(num_rows * num_columns);

        for row in 0..num_rows {
            for column in struct_array.columns() {
                let inner_array = column.as_any().downcast_ref::<UInt64Array>().unwrap();
                packed_row.push(inner_array.value(row));
            }
        }

        let packed_array = Arc::new(UInt64Array::from(packed_row)) as ArrayRef;
        packed_vec.push(packed_array);
    }

    packed_vec
}

impl ArrayEncoder for PackedStructEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: &mut u32) -> Result<EncodedArray> {
        let packed_arrays = encode_packed_struct(arrays);
        let num_elements_per_field = arrays[0]
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .column(0).len();

        println!("Packed arrays length: {:?}", packed_arrays.len());
        println!("Packed arrays: {:?}", packed_arrays.clone());
        let encoded_packed_struct = self.inner_encoder.encode(&packed_arrays, buffer_index)?;
        let encoded_buffers = encoded_packed_struct.buffers;

        println!("Encoded packed struct");
        Ok(EncodedArray {
            buffers: encoded_buffers,
            encoding: pb::ArrayEncoding {
                array_encoding: Some(pb::array_encoding::ArrayEncoding::PackedStruct(Box::new(
                    pb::PackedStruct {
                        inner: Some(Box::new(encoded_packed_struct.encoding)),
                        num_elements_per_field: num_elements_per_field as u32,
                    },
                ))),
            },
        })
    }
}

#[cfg(test)]
pub mod tests {

    use arrow_array::{
        builder::{LargeStringBuilder, StringBuilder},
        ArrayRef, StringArray, StructArray, UInt64Array,
    };
    use arrow_schema::{DataType, Field, Fields};
    use std::{sync::Arc, vec};

    use crate::testing::{
        check_round_trip_encoding_of_data, check_round_trip_encoding_random, TestCases,
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
        let array1 = Arc::new(UInt64Array::from(vec![1, 2, 3]));
        let array2 = Arc::new(UInt64Array::from(vec![4, 5, 6]));

        let struct_array1 = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("x", DataType::UInt64, false)),
                array1.clone() as ArrayRef,
            ),
            (
                Arc::new(Field::new("y", DataType::UInt64, false)),
                array2.clone() as ArrayRef,
            ),
        ]));

        let array3 = Arc::new(UInt64Array::from(vec![10, 11, 12]));
        let array4 = Arc::new(UInt64Array::from(vec![13, 14, 15]));

        let struct_array2 = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("x", DataType::UInt64, false)),
                array3.clone() as ArrayRef,
            ),
            (
                Arc::new(Field::new("y", DataType::UInt64, false)),
                array4.clone() as ArrayRef,
            ),
        ]));

        let test_cases = TestCases::default()
        .with_range(0..1);

        check_round_trip_encoding_of_data(vec![struct_array1, struct_array2], &test_cases).await;
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
