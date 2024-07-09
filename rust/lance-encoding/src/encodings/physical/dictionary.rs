// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::builder::{ArrayBuilder, StringBuilder};
use arrow_array::types::UInt8Type;
use arrow_array::{Array, ArrayRef, DictionaryArray, StringArray, UInt8Array};
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
    num_dictionary_items: u32,
}

impl DictionaryPageScheduler {
    pub fn new(
        indices_scheduler: Arc<dyn PageScheduler>,
        items_scheduler: Arc<dyn PageScheduler>,
        num_dictionary_items: u32,
    ) -> Self {
        Self {
            indices_scheduler,
            items_scheduler,
            num_dictionary_items,
        }
    }
}

impl PageScheduler for DictionaryPageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u64>],
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
        let items_range = 0..(self.num_dictionary_items as u64);
        let items_page_decoder = self.items_scheduler.schedule_ranges(
            std::slice::from_ref(&items_range),
            scheduler,
            top_level_row,
        );

        let copy_size = self.num_dictionary_items as u64;

        tokio::spawn(async move {
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
        })
        .map(|join_handle| join_handle.unwrap())
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
        rows_to_skip: u64,
        num_rows: u64,
        all_null: &mut bool,
    ) -> Result<Vec<BytesMut>> {
        // Decode the indices
        let indices_buffers = self
            .indices_decoder
            .decode(rows_to_skip, num_rows, all_null)?;

        let indices_array =
            new_primitive_array::<UInt8Type>(indices_buffers.clone(), num_rows, &DataType::UInt8);

        let indices_array = indices_array.as_primitive::<UInt8Type>().clone();
        let dictionary = self.decoded_dict.clone();

        let adjusted_indices: UInt8Array = indices_array
            .iter()
            .map(|x| match x {
                Some(0) => None,
                Some(x) => Some(x - 1),
                None => None,
            })
            .collect();

        // Build dictionary array using indices and items
        let dict_array =
            DictionaryArray::<UInt8Type>::try_new(adjusted_indices, dictionary).unwrap();
        let string_array = arrow_cast::cast(&dict_array, &DataType::Utf8).unwrap();
        let string_array = string_array.as_any().downcast_ref::<StringArray>().unwrap();

        // This workflow is not ideal, since we go from DictionaryArray -> StringArray -> nulls, offsets, and bytes buffers (BytesMut)
        // and later in primitive_array_from_buffers() we will go from nulls, offsets, and bytes buffers -> StringArray again.
        // Creating the BytesMut is an unnecessary copy. But it is the best we can do in the current structure
        let null_buffer = string_array
            .nulls()
            .map(|n| BytesMut::from(n.buffer().as_slice()))
            .unwrap_or_else(BytesMut::new);

        let offsets_buffer = BytesMut::from(string_array.offsets().inner().inner().as_slice());

        // Empty buffer for nulls of bytes
        let empty_buffer = BytesMut::new();

        let bytes_buffer = BytesMut::from_iter(string_array.values().iter().copied());

        Ok(vec![
            null_buffer,
            offsets_buffer,
            empty_buffer,
            bytes_buffer,
        ])
    }

    fn num_buffers(&self) -> u32 {
        self.items_decoder.num_buffers() + 2
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

fn encode_dict_indices_and_items(arrays: &[ArrayRef]) -> (ArrayRef, ArrayRef) {
    let mut arr_hashmap: HashMap<&str, u8> = HashMap::new();
    // We start with a dict index of 1 because the value 0 is reserved for nulls
    // The dict indices are adjusted by subtracting 1 later during decode
    let mut curr_dict_index = 1;
    let total_capacity = arrays.iter().map(|arr| arr.len()).sum();

    let mut dict_indices = Vec::with_capacity(total_capacity);
    let mut dict_builder = StringBuilder::new();

    for arr in arrays.iter() {
        let string_array = arrow_array::cast::as_string_array(arr);

        for i in 0..string_array.len() {
            if !string_array.is_valid(i) {
                // null value
                dict_indices.push(0);
                continue;
            }

            let st = string_array.value(i);

            let hashmap_entry = *arr_hashmap.entry(st).or_insert(curr_dict_index);
            dict_indices.push(hashmap_entry);

            // if item didn't exist in the hashmap, add it to the dictionary
            // and increment the dictionary index
            if hashmap_entry == curr_dict_index {
                dict_builder.append_value(st);
                curr_dict_index += 1;
            }
        }
    }

    let array_dict_indices = Arc::new(UInt8Array::from(dict_indices)) as ArrayRef;

    // If there is an empty dictionary:
    // Either there is an array of nulls or an empty array altogether
    // In this case create the dictionary with a single null element
    // Because decoding [] is not currently supported by the binary decoder
    if dict_builder.is_empty() {
        dict_builder.append_option(Option::<&str>::None);
    }

    let dict_elements = dict_builder.finish();
    let array_dict_elements = arrow_cast::cast(&dict_elements, &DataType::Utf8).unwrap();

    (array_dict_indices, array_dict_elements)
}

impl ArrayEncoder for DictionaryEncoder {
    fn encode(&self, arrays: &[ArrayRef], buffer_index: &mut u32) -> Result<EncodedArray> {
        let (index_array, items_array) = encode_dict_indices_and_items(arrays);

        let encoded_indices = self
            .indices_encoder
            .encode(&[index_array.clone()], buffer_index)?;

        let encoded_items = self
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
                        num_dictionary_items: dict_size,
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
        ArrayRef, StringArray, UInt8Array,
    };
    use arrow_schema::{DataType, Field};
    use std::{sync::Arc, vec};

    use crate::testing::{
        check_round_trip_encoding_of_data, check_round_trip_encoding_random, TestCases,
    };

    use super::encode_dict_indices_and_items;

    #[test]
    fn test_encode_dict_nulls() {
        // Null entries in string arrays should be adjusted
        let string_array1 = Arc::new(StringArray::from(vec![None, Some("foo"), Some("bar")]));
        let string_array2 = Arc::new(StringArray::from(vec![Some("bar"), None, Some("foo")]));
        let string_array3 = Arc::new(StringArray::from(vec![None as Option<&str>, None]));
        let (dict_indices, dict_items) =
            encode_dict_indices_and_items(&[string_array1, string_array2, string_array3]);

        let expected_indices = Arc::new(UInt8Array::from(vec![0, 1, 2, 2, 0, 1, 0, 0])) as ArrayRef;
        let expected_items = Arc::new(StringArray::from(vec!["foo", "bar"])) as ArrayRef;
        assert_eq!(&dict_indices, &expected_indices);
        assert_eq!(&dict_items, &expected_items);
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
