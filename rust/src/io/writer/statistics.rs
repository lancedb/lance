// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Statistics collection utilities

use std::collections::HashMap;

use arrow_array::{
    builder::{make_builder, ArrayBuilder, PrimitiveBuilder},
    types::{Int64Type, UInt32Type},
    Array, ArrayRef, RecordBatch,
};
use arrow_schema::DataType;

use crate::datatypes::Field;
use crate::error::Result;

/// Max number of bytes that are included in statistics for binary columns.
const BINARY_PREFIX_LENGTH: usize = 16;

/// Statistics for a single column chunk.
#[derive(Debug, PartialEq)]
pub struct StatisticsRow {
    /// Number of nulls in this column chunk.
    pub(crate) null_count: u32,
    /// Minimum value in this column chunk, if any
    pub(crate) min_value: Option<Box<dyn Array>>,
    /// Maximum value in this column chunk, if any
    pub(crate) max_value: Option<Box<dyn Array>>,
}

pub fn collect_statistics(arrays: &[&ArrayRef]) -> StatisticsRow {
    todo!();
}

pub struct StatisticsCollector {
    num_rows: PrimitiveBuilder<Int64Type>,
    builders: HashMap<i32, (Field, StatisticsBuilder)>,
}

impl StatisticsCollector {
    pub fn new(fields: &[Field]) -> Self {
        let builders = fields
            .iter()
            .map(|f| (f.id, (f.clone(), StatisticsBuilder::new(&f.data_type()))))
            .collect();
        Self {
            builders,
            num_rows: PrimitiveBuilder::<Int64Type>::new(),
        }
    }

    pub fn get_builder(&mut self, field_id: i32) -> Option<&mut StatisticsBuilder> {
        self.builders.get_mut(&field_id).map(|(_, b)| b)
    }

    pub fn append_num_values(&mut self, num_rows: i64) {
        self.num_rows.append_value(num_rows)
    }

    pub fn finish(&mut self) -> Result<RecordBatch> {
        todo!()
    }
}

pub struct StatisticsBuilder {
    null_count: PrimitiveBuilder<UInt32Type>,
    min_value: Box<dyn ArrayBuilder>,
    max_value: Box<dyn ArrayBuilder>,
}

impl StatisticsBuilder {
    fn new(data_type: &DataType) -> Self {
        let null_count = PrimitiveBuilder::<UInt32Type>::new();
        let min_value = make_builder(data_type, 1);
        let max_value = make_builder(data_type, 1);
        Self {
            null_count,
            min_value,
            max_value,
        }
    }

    pub fn append(&mut self, row: StatisticsRow) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        builder::StringDictionaryBuilder, BinaryArray, BooleanArray, Date32Array, Decimal128Array,
        DurationMillisecondArray, Float32Array, Int32Array, Int64Array, StringArray, StructArray,
        TimestampMicrosecondArray,
    };

    use crate::datatypes::Schema;
    use arrow_schema::{Field as ArrowField, Fields as ArrowFields, Schema as ArrowSchema};

    use super::*;

    #[test]
    fn test_collect_primitive_stats() {
        struct TestCase {
            source_arrays: Vec<ArrayRef>,
            expected_min: Option<Box<dyn Array>>,
            expected_max: Option<Box<dyn Array>>,
        }

        let cases: [TestCase; 6] = [
            // Int64
            TestCase {
                source_arrays: vec![
                    Arc::new(Int64Array::from(vec![4, 3, 7, 2])),
                    Arc::new(Int64Array::from(vec![-10, 3, 5])),
                ],
                expected_min: Some(Box::new(Int64Array::from(vec![-10]))),
                expected_max: Some(Box::new(Int64Array::from(vec![7]))),
            },
            // Boolean
            TestCase {
                source_arrays: vec![Arc::new(BooleanArray::from(vec![true, false]))],
                expected_min: Some(Box::new(BooleanArray::from(vec![false]))),
                expected_max: Some(Box::new(BooleanArray::from(vec![true]))),
            },
            // Date
            TestCase {
                source_arrays: vec![
                    Arc::new(Date32Array::from(vec![53, 42])),
                    Arc::new(Date32Array::from(vec![68, 32])),
                ],
                expected_min: Some(Box::new(Date32Array::from(vec![32]))),
                expected_max: Some(Box::new(Date32Array::from(vec![68]))),
            },
            // Timestamp
            TestCase {
                source_arrays: vec![
                    Arc::new(TimestampMicrosecondArray::from(vec![53, 42])),
                    Arc::new(TimestampMicrosecondArray::from(vec![68, 32])),
                ],
                expected_min: Some(Box::new(TimestampMicrosecondArray::from(vec![32]))),
                expected_max: Some(Box::new(TimestampMicrosecondArray::from(vec![68]))),
            },
            // Duration
            TestCase {
                source_arrays: vec![
                    Arc::new(DurationMillisecondArray::from(vec![53, 42])),
                    Arc::new(DurationMillisecondArray::from(vec![68, 32])),
                ],
                expected_min: Some(Box::new(DurationMillisecondArray::from(vec![32]))),
                expected_max: Some(Box::new(DurationMillisecondArray::from(vec![68]))),
            },
            // Decimal
            TestCase {
                source_arrays: vec![
                    Arc::new(Decimal128Array::from(vec![53, 42])),
                    Arc::new(Decimal128Array::from(vec![68, 32])),
                ],
                expected_min: Some(Box::new(Decimal128Array::from(vec![32]))),
                expected_max: Some(Box::new(Decimal128Array::from(vec![68]))),
            },
        ];

        for case in cases {
            let array_refs = case.source_arrays.iter().collect::<Vec<_>>();
            let stats = collect_statistics(&array_refs);
            assert_eq!(
                stats,
                StatisticsRow {
                    null_count: 0,
                    min_value: case.expected_min,
                    max_value: case.expected_max,
                },
                "Statistics are wrong for input data: {:?}",
                case.source_arrays
            );
        }
    }

    #[test]
    fn test_collect_float_stats() {
        // NaN values are ignored in statistics
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(Float32Array::from(vec![4.0f32, 3.0, std::f32::NAN, 2.0])),
            Arc::new(Float32Array::from(vec![-10.0f32, 3.0, 5.0, std::f32::NAN])),
        ];
        let array_refs = arrays.iter().collect::<Vec<_>>();
        let stats = collect_statistics(&array_refs);
        assert_eq!(
            stats,
            StatisticsRow {
                null_count: 0,
                min_value: Some(Box::new(Float32Array::from(vec![-10.0]))),
                max_value: Some(Box::new(Float32Array::from(vec![7.0]))),
            }
        );

        // (Negative) Infinity can be min or max.
        let arrays: Vec<ArrayRef> = vec![Arc::new(Float32Array::from(vec![
            4.0f32,
            std::f32::INFINITY,
            std::f32::NEG_INFINITY,
        ]))];
        let array_refs = arrays.iter().collect::<Vec<_>>();
        let stats = collect_statistics(&array_refs);
        assert_eq!(
            stats,
            StatisticsRow {
                null_count: 0,
                min_value: Some(Box::new(Float32Array::from(vec![std::f32::NEG_INFINITY]))),
                max_value: Some(Box::new(Float32Array::from(vec![std::f32::INFINITY]))),
            }
        );

        // Zero is always positive
        let arrays: Vec<ArrayRef> = vec![Arc::new(Float32Array::from(vec![4.0f32, -0.0]))];
        let array_refs = arrays.iter().collect::<Vec<_>>();
        let stats = collect_statistics(&array_refs);
        assert_eq!(
            stats,
            StatisticsRow {
                null_count: 0,
                min_value: Some(Box::new(Float32Array::from(vec![0.0f32]))),
                max_value: Some(Box::new(Float32Array::from(vec![4.0f32]))),
            }
        );
    }

    #[test]
    fn test_collect_binary_stats() {
        // Test string, binary with truncation and null values.

        // Whole strings are used if short enough
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec![Some("foo"), None, Some("bar")])),
            Arc::new(StringArray::from(vec!["yee", "haw"])),
        ];
        let array_refs = arrays.iter().collect::<Vec<_>>();
        let stats = collect_statistics(&array_refs);
        assert_eq!(
            stats,
            StatisticsRow {
                null_count: 1,
                min_value: Some(Box::new(StringArray::from(vec!["bar"]))),
                max_value: Some(Box::new(StringArray::from(vec!["yee"]))),
            }
        );

        // Prefixes are used if strings are too long. Multi-byte characters are
        // not split.
        let arrays: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![
            "bacteriologists🧑‍🔬",
            "terrestial planet",
        ]))];
        let array_refs = arrays.iter().collect::<Vec<_>>();
        let stats = collect_statistics(&array_refs);
        assert_eq!(
            stats,
            StatisticsRow {
                null_count: 0,
                // Bacteriologists is just 15 bytes, but the next character is multi-byte
                // so we truncate before.
                min_value: Some(Box::new(StringArray::from(vec!["bacteriologists"]))),
                // Increment the last character to make sure it's greater than max value
                max_value: Some(Box::new(StringArray::from(vec!["terrestial pland"]))),
            }
        );

        // If not truncated max value exists (in the edge case where the value is
        // 0xFF up until the limit), just return null as max.)
        let arrays: Vec<ArrayRef> = vec![Arc::new(BinaryArray::from(vec![vec![
            0xFFu8;
            BINARY_PREFIX_LENGTH
                + 5
        ]
        .as_ref()]))];
        let array_refs = arrays.iter().collect::<Vec<_>>();
        let stats = collect_statistics(&array_refs);
        assert_eq!(
            stats,
            StatisticsRow {
                null_count: 0,
                // We can truncate the minimum value, since the prefix is less than the full value
                min_value: Some(Box::new(BinaryArray::from(vec![vec![
                    0xFFu8;
                    BINARY_PREFIX_LENGTH
                ]
                .as_ref()]))),
                // We can't truncate the max value, so we return None
                max_value: None,
            }
        );
    }

    #[test]
    fn test_collect_dictionary_stats() {
        // Dictionary stats are collected from the underlying values
        let dictionary_values = StringArray::from(vec![None, Some("abc"), Some("def")]);
        let mut builder =
            StringDictionaryBuilder::<UInt32Type>::new_with_dictionary(3, &dictionary_values)
                .unwrap();
        builder.append("def").unwrap();
        builder.append_null();
        builder.append("abc").unwrap();

        let arr = builder.finish();
        let arrays: Vec<ArrayRef> = vec![Arc::new(arr)];
        let array_refs = arrays.iter().collect::<Vec<_>>();
        let stats = collect_statistics(&array_refs);
        assert_eq!(
            stats,
            StatisticsRow {
                null_count: 1,
                min_value: Some(Box::new(StringArray::from(vec!["abc"]))),
                max_value: Some(Box::new(StringArray::from(vec!["def"]))),
            }
        );
    }

    #[test]
    fn test_stats_collector() {
        // Check the output schema is correct
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("b", DataType::Utf8, true),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let mut collector = StatisticsCollector::new(&schema.fields);

        // Collect stats for a
        let id = schema.field("a").unwrap().id;
        let builder = collector.get_builder(id).unwrap();
        builder.append(StatisticsRow {
            null_count: 2,
            min_value: Some(Box::new(Int32Array::from(vec![1]))),
            max_value: Some(Box::new(Int32Array::from(vec![3]))),
        });
        builder.append(StatisticsRow {
            null_count: 0,
            min_value: None,
            max_value: None,
        });

        // If we try to finish at this point, it will error since we don't have
        // stats for b yet.
        assert!(collector.finish().is_err());

        // Collect stats for b
        let id = schema.field("b").unwrap().id;
        let builder = collector.get_builder(id).unwrap();
        builder.append(StatisticsRow {
            null_count: 6,
            min_value: Some(Box::new(StringArray::from(vec!["aaa"]))),
            max_value: Some(Box::new(StringArray::from(vec!["bbb"]))),
        });
        builder.append(StatisticsRow {
            null_count: 0,
            min_value: None,
            max_value: None,
        });

        collector.append_num_values(42);
        collector.append_num_values(64);

        // Now we can finish
        let batch = collector.finish().unwrap();

        let expected_schema = ArrowSchema::new(vec![
            ArrowField::new("num_values", DataType::Int64, false),
            ArrowField::new(
                "0",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("null_count", DataType::Int64, false),
                    ArrowField::new("min_value", DataType::Int32, false),
                    ArrowField::new("max_value", DataType::Int32, false),
                ])),
                false,
            ),
            ArrowField::new(
                "1",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("null_count", DataType::Int64, false),
                    ArrowField::new("min_value", DataType::Utf8, true),
                    ArrowField::new("max_value", DataType::Utf8, true),
                ])),
                false,
            ),
        ]);

        assert_eq!(batch.schema().as_ref(), &expected_schema);

        let expected_batch = RecordBatch::try_new(
            Arc::new(expected_schema),
            vec![
                Arc::new(Int32Array::from(vec![42, 64])),
                Arc::new(StructArray::from(vec![
                    (
                        Arc::new(ArrowField::new("null_count", DataType::Int64, false)),
                        Arc::new(Int64Array::from(vec![2, 0])) as ArrayRef,
                    ),
                    (
                        Arc::new(ArrowField::new("min_value", DataType::Int32, false)),
                        Arc::new(Int32Array::from(vec![1, std::i32::MIN])) as ArrayRef,
                    ),
                    (
                        Arc::new(ArrowField::new("max_value", DataType::Int32, false)),
                        Arc::new(Int32Array::from(vec![3, std::i32::MAX])) as ArrayRef,
                    ),
                ])),
                Arc::new(StructArray::from(vec![
                    (
                        Arc::new(ArrowField::new("null_count", DataType::Int64, false)),
                        Arc::new(Int32Array::from(vec![6, 0])) as ArrayRef,
                    ),
                    (
                        Arc::new(ArrowField::new("min_value", DataType::Utf8, true)),
                        Arc::new(StringArray::from(vec![Some("aaa"), None])) as ArrayRef,
                    ),
                    (
                        Arc::new(ArrowField::new("max_value", DataType::Utf8, true)),
                        Arc::new(StringArray::from(vec![Some("bbb"), None])) as ArrayRef,
                    ),
                ])),
            ],
        )
        .unwrap();

        assert_eq!(batch, expected_batch);
    }
}
