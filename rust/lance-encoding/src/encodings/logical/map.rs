// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, ListArray, MapArray};
use arrow_schema::DataType;
use futures::future::BoxFuture;
use lance_core::{Error, Result};
use snafu::location;

use crate::{
    decoder::{DecodedArray, StructuralDecodeArrayTask, StructuralFieldDecoder},
    encoder::{EncodeTask, FieldEncoder, OutOfLineBuffers},
    encodings::logical::list::ListStructuralEncoder,
    repdef::RepDefBuilder,
};
/// A structural encoder for map fields
///
/// Map in Arrow is represented as List<Struct<key, value>>
/// This encoder uses the [`ListStructuralEncoder`] to encode the data
pub struct MapStructuralEncoder {
    list_encoder: ListStructuralEncoder,
}

impl MapStructuralEncoder {
    pub fn new(keep_original_array: bool, child: Box<dyn FieldEncoder>) -> Self {
        Self {
            list_encoder: ListStructuralEncoder::new(keep_original_array, child),
        }
    }
}

impl FieldEncoder for MapStructuralEncoder {
    fn maybe_encode(
        &mut self,
        array: ArrayRef,
        external_buffers: &mut OutOfLineBuffers,
        repdef: RepDefBuilder,
        row_number: u64,
        num_rows: u64,
    ) -> Result<Vec<EncodeTask>> {
        let map_array = array
            .as_any()
            .downcast_ref::<MapArray>()
            .expect("MapEncoder used for non-map data");

        let (entries_field, offsets, entries, validity, _keys_sorted) =
            map_array.clone().into_parts();

        let list_array = ListArray::new(entries_field, offsets, Arc::new(entries), validity);

        self.list_encoder.maybe_encode(
            Arc::new(list_array),
            external_buffers,
            repdef,
            row_number,
            num_rows,
        )
    }

    fn flush(&mut self, external_buffers: &mut OutOfLineBuffers) -> Result<Vec<EncodeTask>> {
        self.list_encoder.flush(external_buffers)
    }

    fn num_columns(&self) -> u32 {
        self.list_encoder.num_columns()
    }

    fn finish(
        &mut self,
        external_buffers: &mut OutOfLineBuffers,
    ) -> BoxFuture<'_, Result<Vec<crate::encoder::EncodedColumn>>> {
        self.list_encoder.finish(external_buffers)
    }
}

/// A structural decoder for map fields
///
/// This decoder uses the [`StructuralListDecoder`] to decode the data as a list.
/// It then simply casts the list array to a map array.
#[derive(Debug)]
pub struct StructuralMapDecoder {
    list_decoder: Box<dyn StructuralFieldDecoder>,
    data_type: DataType,
}

impl StructuralMapDecoder {
    pub fn new(list_decoder: Box<dyn StructuralFieldDecoder>, data_type: DataType) -> Self {
        Self {
            list_decoder,
            data_type,
        }
    }
}

impl StructuralFieldDecoder for StructuralMapDecoder {
    fn accept_page(&mut self, list_decoder: crate::decoder::LoadedPageShard) -> Result<()> {
        self.list_decoder.accept_page(list_decoder)
    }

    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn StructuralDecodeArrayTask>> {
        let child_task = self.list_decoder.drain(num_rows)?;
        Ok(Box::new(StructuralMapDecodeTask::new(child_task)))
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }
}

#[derive(Debug)]
struct StructuralMapDecodeTask {
    list_decode_task: Box<dyn StructuralDecodeArrayTask>,
}

impl StructuralMapDecodeTask {
    fn new(list_decode_task: Box<dyn StructuralDecodeArrayTask>) -> Self {
        Self { list_decode_task }
    }
}

impl StructuralDecodeArrayTask for StructuralMapDecodeTask {
    fn decode(self: Box<Self>) -> Result<DecodedArray> {
        let DecodedArray { array, repdef } = self.list_decode_task.decode()?;

        let list_array =
            array
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| Error::Schema {
                    message: format!(
                        "Expected list array from map's inner decoder, got: {:?}",
                        array.data_type()
                    ),
                    location: location!(),
                })?;

        // Extract the entries field and keys_sorted from the map data type
        let entries_field = match list_array.data_type() {
            DataType::List(field) => field.clone(),
            _ => {
                return Err(Error::Schema {
                    message: "List array did not have list data type".to_string(),
                    location: location!(),
                });
            }
        };

        // Convert the decoded array to StructArray
        let entries = list_array
            .values()
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .ok_or_else(|| Error::Schema {
                message: "Map entries should be a StructArray".to_string(),
                location: location!(),
            })?
            .clone();

        // Build the MapArray from offsets, entries, validity, and keys_sorted
        let map_array = MapArray::new(
            entries_field,
            list_array.offsets().clone(),
            entries,
            list_array.nulls().cloned(),
            false, // keys_sorted is always false at the moment
        );

        Ok(DecodedArray {
            array: Arc::new(map_array),
            repdef,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow_array::{
        builder::{Int32Builder, MapBuilder, StringBuilder},
        Array, Int32Array, MapArray, StringArray, StructArray,
    };
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};
    use arrow_schema::{DataType, Field, Fields};

    use crate::encoder::{default_encoding_strategy, ColumnIndexSequence, EncodingOptions};
    use crate::{
        testing::{check_round_trip_encoding_of_data, TestCases},
        version::LanceFileVersion,
    };
    use arrow_schema::Field as ArrowField;
    use lance_core::datatypes::Field as LanceField;

    fn make_map_type(key_type: DataType, value_type: DataType) -> DataType {
        // Note: Arrow MapBuilder uses "keys" and "values" as field names (plural)
        let entries = Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("keys", key_type, false),
                Field::new("values", value_type, true),
            ])),
            false,
        );
        DataType::Map(Arc::new(entries), false)
    }

    #[test_log::test(tokio::test)]
    async fn test_simple_map() {
        // Create a simple Map<String, Int32>
        let string_builder = StringBuilder::new();
        let int_builder = Int32Builder::new();
        let mut map_builder = MapBuilder::new(None, string_builder, int_builder);

        // Map 1: {"key1": 10, "key2": 20}
        map_builder.keys().append_value("key1");
        map_builder.values().append_value(10);
        map_builder.keys().append_value("key2");
        map_builder.values().append_value(20);
        map_builder.append(true).unwrap();

        // Map 2: {"key3": 30}
        map_builder.keys().append_value("key3");
        map_builder.values().append_value(30);
        map_builder.append(true).unwrap();

        let map_array = map_builder.finish();

        let test_cases = TestCases::default()
            .with_range(0..2)
            .with_min_file_version(LanceFileVersion::V2_2);

        check_round_trip_encoding_of_data(vec![Arc::new(map_array)], &test_cases, HashMap::new())
            .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_empty_maps() {
        // Test maps with empty entries
        let string_builder = StringBuilder::new();
        let int_builder = Int32Builder::new();
        let mut map_builder = MapBuilder::new(None, string_builder, int_builder);

        // Map 1: {"a": 1}
        map_builder.keys().append_value("a");
        map_builder.values().append_value(1);
        map_builder.append(true).unwrap();

        // Map 2: {} (empty)
        map_builder.append(true).unwrap();

        // Map 3: null
        map_builder.append(false).unwrap();

        // Map 4: {} (empty)
        map_builder.append(true).unwrap();

        let map_array = map_builder.finish();

        let test_cases = TestCases::default()
            .with_range(0..4)
            .with_indices(vec![1])
            .with_indices(vec![2])
            .with_min_file_version(LanceFileVersion::V2_2);

        check_round_trip_encoding_of_data(vec![Arc::new(map_array)], &test_cases, HashMap::new())
            .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_map_with_null_values() {
        // Test Map<String, Int32> with null values
        let string_builder = StringBuilder::new();
        let int_builder = Int32Builder::new();
        let mut map_builder = MapBuilder::new(None, string_builder, int_builder);

        // Map 1: {"key1": 10, "key2": null}
        map_builder.keys().append_value("key1");
        map_builder.values().append_value(10);
        map_builder.keys().append_value("key2");
        map_builder.values().append_null();
        map_builder.append(true).unwrap();

        // Map 2: {"key3": null}
        map_builder.keys().append_value("key3");
        map_builder.values().append_null();
        map_builder.append(true).unwrap();

        let map_array = map_builder.finish();

        let test_cases = TestCases::default()
            .with_range(0..2)
            .with_indices(vec![0])
            .with_indices(vec![1])
            .with_min_file_version(LanceFileVersion::V2_2);

        check_round_trip_encoding_of_data(vec![Arc::new(map_array)], &test_cases, HashMap::new())
            .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_map_in_struct() {
        // Test Struct containing Map
        // Struct<id: Int32, properties: Map<String, String>>

        let string_key_builder = StringBuilder::new();
        let string_val_builder = StringBuilder::new();
        let mut map_builder = MapBuilder::new(None, string_key_builder, string_val_builder);

        // First struct: id=1, properties={"name": "Alice", "city": "NYC"}
        map_builder.keys().append_value("name");
        map_builder.values().append_value("Alice");
        map_builder.keys().append_value("city");
        map_builder.values().append_value("NYC");
        map_builder.append(true).unwrap();

        // Second struct: id=2, properties={"name": "Bob"}
        map_builder.keys().append_value("name");
        map_builder.values().append_value("Bob");
        map_builder.append(true).unwrap();

        // Third struct: id=3, properties=null
        map_builder.append(false).unwrap();

        let map_array = Arc::new(map_builder.finish());
        let id_array = Arc::new(Int32Array::from(vec![1, 2, 3]));

        let struct_array = StructArray::new(
            Fields::from(vec![
                Field::new("id", DataType::Int32, false),
                Field::new(
                    "properties",
                    make_map_type(DataType::Utf8, DataType::Utf8),
                    true,
                ),
            ]),
            vec![id_array, map_array],
            None,
        );

        let test_cases = TestCases::default()
            .with_range(0..3)
            .with_indices(vec![0, 2])
            .with_min_file_version(LanceFileVersion::V2_2);

        check_round_trip_encoding_of_data(
            vec![Arc::new(struct_array)],
            &test_cases,
            HashMap::new(),
        )
        .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_list_of_maps() {
        // Test List<Map<String, Int32>>
        use arrow_array::builder::ListBuilder;

        let string_builder = StringBuilder::new();
        let int_builder = Int32Builder::new();
        let map_builder = MapBuilder::new(None, string_builder, int_builder);
        let mut list_builder = ListBuilder::new(map_builder);

        // List 1: [{"a": 1}, {"b": 2}]
        list_builder.values().keys().append_value("a");
        list_builder.values().values().append_value(1);
        list_builder.values().append(true).unwrap();

        list_builder.values().keys().append_value("b");
        list_builder.values().values().append_value(2);
        list_builder.values().append(true).unwrap();

        list_builder.append(true);

        // List 2: [{"c": 3}]
        list_builder.values().keys().append_value("c");
        list_builder.values().values().append_value(3);
        list_builder.values().append(true).unwrap();

        list_builder.append(true);

        // List 3: [] (empty list)
        list_builder.append(true);

        let list_array = list_builder.finish();

        let test_cases = TestCases::default()
            .with_range(0..3)
            .with_indices(vec![0, 2])
            .with_min_file_version(LanceFileVersion::V2_2);

        check_round_trip_encoding_of_data(vec![Arc::new(list_array)], &test_cases, HashMap::new())
            .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_nested_map() {
        // Test Map<String, Map<String, Int32>>
        // This is more complex as we need to build nested maps manually

        // Build inner maps first
        let inner_string_builder = StringBuilder::new();
        let inner_int_builder = Int32Builder::new();
        let mut inner_map_builder1 = MapBuilder::new(None, inner_string_builder, inner_int_builder);

        // Inner map 1: {"x": 10}
        inner_map_builder1.keys().append_value("x");
        inner_map_builder1.values().append_value(10);
        inner_map_builder1.append(true).unwrap();

        // Inner map 2: {"y": 20, "z": 30}
        inner_map_builder1.keys().append_value("y");
        inner_map_builder1.values().append_value(20);
        inner_map_builder1.keys().append_value("z");
        inner_map_builder1.values().append_value(30);
        inner_map_builder1.append(true).unwrap();

        let inner_maps = Arc::new(inner_map_builder1.finish());

        // Build outer map keys
        let outer_keys = Arc::new(StringArray::from(vec!["key1", "key2"]));

        // Build outer map structure
        let entries_struct = StructArray::new(
            Fields::from(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new(
                    "value",
                    make_map_type(DataType::Utf8, DataType::Int32),
                    true,
                ),
            ]),
            vec![outer_keys, inner_maps],
            None,
        );

        let offsets = OffsetBuffer::new(ScalarBuffer::<i32>::from(vec![0, 2]));
        let entries_field = Field::new("entries", entries_struct.data_type().clone(), false);

        let outer_map = MapArray::new(
            Arc::new(entries_field),
            offsets,
            entries_struct,
            None,
            false,
        );

        let test_cases = TestCases::default()
            .with_range(0..1)
            .with_min_file_version(LanceFileVersion::V2_2);

        check_round_trip_encoding_of_data(vec![Arc::new(outer_map)], &test_cases, HashMap::new())
            .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_map_different_key_types() {
        // Test Map<Int32, String> (integer keys)
        let int_builder = Int32Builder::new();
        let string_builder = StringBuilder::new();
        let mut map_builder = MapBuilder::new(None, int_builder, string_builder);

        // Map 1: {1: "one", 2: "two"}
        map_builder.keys().append_value(1);
        map_builder.values().append_value("one");
        map_builder.keys().append_value(2);
        map_builder.values().append_value("two");
        map_builder.append(true).unwrap();

        // Map 2: {3: "three"}
        map_builder.keys().append_value(3);
        map_builder.values().append_value("three");
        map_builder.append(true).unwrap();

        let map_array = map_builder.finish();

        let test_cases = TestCases::default()
            .with_range(0..2)
            .with_indices(vec![0, 1])
            .with_min_file_version(LanceFileVersion::V2_2);

        check_round_trip_encoding_of_data(vec![Arc::new(map_array)], &test_cases, HashMap::new())
            .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_map_with_extreme_sizes() {
        // Test maps with large number of entries
        let string_builder = StringBuilder::new();
        let int_builder = Int32Builder::new();
        let mut map_builder = MapBuilder::new(None, string_builder, int_builder);

        // Create a map with many entries
        for i in 0..100 {
            map_builder.keys().append_value(format!("key{}", i));
            map_builder.values().append_value(i);
        }
        map_builder.append(true).unwrap();

        // Create a second map with no entries
        map_builder.append(true).unwrap();

        let map_array = map_builder.finish();

        let test_cases = TestCases::default()
            .with_range(0..2)
            .with_min_file_version(LanceFileVersion::V2_2);

        check_round_trip_encoding_of_data(vec![Arc::new(map_array)], &test_cases, HashMap::new())
            .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_map_all_null() {
        // Test map where all entries are null
        let string_builder = StringBuilder::new();
        let int_builder = Int32Builder::new();
        let mut map_builder = MapBuilder::new(None, string_builder, int_builder);

        // All null maps
        map_builder.append(false).unwrap(); // null
        map_builder.append(false).unwrap(); // null

        let map_array = map_builder.finish();

        let test_cases = TestCases::default()
            .with_range(0..2)
            .with_min_file_version(LanceFileVersion::V2_2);

        check_round_trip_encoding_of_data(vec![Arc::new(map_array)], &test_cases, HashMap::new())
            .await;
    }

    #[test_log::test(tokio::test)]
    async fn test_map_encoder_keep_original_array_scenarios() {
        // Test scenarios that highlight the difference between keep_original_array=true/false
        // This test focuses on round-trip behavior which should be equivalent in both cases
        let string_builder = StringBuilder::new();
        let int_builder = Int32Builder::new();
        let mut map_builder = MapBuilder::new(None, string_builder, int_builder);

        // Create a map with mixed null and non-null values to test both scenarios
        // Map 1: {"key1": 10, "key2": null}
        map_builder.keys().append_value("key1");
        map_builder.values().append_value(10);
        map_builder.keys().append_value("key2");
        map_builder.values().append_null();
        map_builder.append(true).unwrap();

        // Map 2: null
        map_builder.append(false).unwrap();

        // Map 3: {"key3": 30}
        map_builder.keys().append_value("key3");
        map_builder.values().append_value(30);
        map_builder.append(true).unwrap();

        let map_array = map_builder.finish();

        let test_cases = TestCases::default()
            .with_range(0..3)
            .with_indices(vec![0, 1, 2])
            .with_min_file_version(LanceFileVersion::V2_2);

        // This test ensures that regardless of the internal keep_original_array setting,
        // the end-to-end behavior produces equivalent results
        check_round_trip_encoding_of_data(vec![Arc::new(map_array)], &test_cases, HashMap::new())
            .await;
    }

    #[test]
    fn test_map_not_supported_write_in_v2_1() {
        // Create a map field using Arrow Field first, then convert to Lance Field
        let map_arrow_field = ArrowField::new(
            "map_field",
            make_map_type(DataType::Utf8, DataType::Int32),
            true,
        );
        let map_field = LanceField::try_from(&map_arrow_field).unwrap();

        // Test encoder: Try to create encoder with V2_1 version - should fail
        let encoder_strategy = default_encoding_strategy(LanceFileVersion::V2_1);
        let mut column_index = ColumnIndexSequence::default();
        let options = EncodingOptions::default();

        let encoder_result = encoder_strategy.create_field_encoder(
            encoder_strategy.as_ref(),
            &map_field,
            &mut column_index,
            &options,
        );

        assert!(
            encoder_result.is_err(),
            "Map type should not be supported in V2_1 for encoder"
        );
        let Err(encoder_err) = encoder_result else {
            panic!("Expected error but got Ok")
        };

        let encoder_err_msg = format!("{}", encoder_err);
        assert!(
            encoder_err_msg.contains("2.2"),
            "Encoder error message should mention version 2.2, got: {}",
            encoder_err_msg
        );
        assert!(
            encoder_err_msg.contains("Map data type"),
            "Encoder error message should mention Map data type, got: {}",
            encoder_err_msg
        );
    }
}
