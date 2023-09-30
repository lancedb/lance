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

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use crate::datatypes::{Field, Schema};
use crate::error::Result;
use arrow_array::{
    builder::{make_builder, ArrayBuilder, PrimitiveBuilder},
    types::{
        Date32Type, Date64Type, DurationMillisecondType, Float32Type, Float64Type, Int16Type,
        Int32Type, Int64Type, Int8Type, TimestampMicrosecondType, UInt16Type, UInt32Type,
        UInt64Type, UInt8Type,
    },
    Array, ArrayRef, ArrowNumericType, ArrowPrimitiveType, PrimitiveArray, RecordBatch,
};
use arrow_schema::DataType;
use arrow_schema::{ArrowError, Field as ArrowField, Schema as ArrowSchema};
use datafusion::common::ScalarValue;
use num_traits::bounds::Bounded;

/// Max number of bytes that are included in statistics for binary columns.
const BINARY_PREFIX_LENGTH: usize = 16;

/// Statistics for a single column chunk.
#[derive(Debug, PartialEq)]
pub struct StatisticsRow {
    /// Number of nulls in this column chunk.
    pub(crate) null_count: ScalarValue,
    /// Minimum value in this column chunk, if any
    // pub(crate) min_value: Option<Box<dyn Array>>,
    pub(crate) min_value: ScalarValue,
    /// Maximum value in this column chunk, if any
    pub(crate) max_value: ScalarValue,
    // pub(crate) max_value: Option<Box<dyn Array>>,
}

fn get_statistics<T: ArrowNumericType>(arrays: &[&ArrayRef]) -> StatisticsRow
where
    T::Native: Bounded + PartialOrd,
    datafusion_common::scalar::ScalarValue: From<<T as ArrowPrimitiveType>::Native>,
{
    let arr = arrays
        .iter()
        .map(|x| x.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap())
        .collect::<Vec<_>>();
    let mut min_value = T::Native::max_value();
    let mut max_value = T::Native::min_value();
    let mut null_count: u32 = 0;

    for array in arr.iter() {
        array.iter().for_each(|value| {
            if let Some(value) = value {
                if let Some(Ordering::Greater) = value.partial_cmp(&max_value) {
                    max_value = value;
                } else if let Some(Ordering::Less) = value.partial_cmp(&min_value) {
                    min_value = value;
                }
            };
        });
        null_count += array.null_count() as u32;
    }
    // TODO: correct for -0.0 when float
    return StatisticsRow {
        null_count: ScalarValue::UInt32(Some(null_count)),
        min_value: ScalarValue::from(min_value),
        max_value: ScalarValue::from(max_value),
    };
}

pub fn collect_statistics(arrays: &[&ArrayRef]) -> StatisticsRow {
    match arrays[0].data_type() {
        DataType::Int8 => get_statistics::<Int8Type>(arrays),
        DataType::UInt8 => get_statistics::<UInt8Type>(arrays),
        DataType::Int16 => get_statistics::<Int16Type>(arrays),
        DataType::UInt16 => get_statistics::<UInt16Type>(arrays),
        DataType::Int32 => get_statistics::<Int32Type>(arrays),
        DataType::UInt32 => get_statistics::<UInt32Type>(arrays),
        DataType::Int64 => get_statistics::<Int64Type>(arrays),
        DataType::UInt64 => get_statistics::<UInt64Type>(arrays),
        DataType::Float32 => get_statistics::<Float32Type>(arrays),
        DataType::Float64 => get_statistics::<Float64Type>(arrays),
        DataType::Date32 => get_statistics::<Date32Type>(arrays),
        DataType::Date64 => get_statistics::<Date64Type>(arrays),
        DataType::Timestamp(_, _) => get_statistics::<TimestampMicrosecondType>(arrays), // TODO: timezones
        DataType::Duration(_) => get_statistics::<DurationMillisecondType>(arrays),
        // TODO: cover all types
        // DataType::Decimal128(_, _) => { get_statistics::<Decimal128Type>(arrays) }
        // DataType::Binary => { get_statistics::<BinaryType>(arrays) }
        // DataType::Utf8 => { get_statistics::<Utf8Type>(arrays) }
        // DataType::Boolean => { get_statistics::<BooleanType>(arrays) }
        // DataType::Struct(_) => {  }
        _ => {
            println!(
                "Stats collection for {} is not supported yet",
                arrays[0].data_type()
            );
            todo!()
        }
    }
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

    // TODO: handle types
    pub fn finish(&mut self) -> Result<RecordBatch> {
        let mut fields: Vec<ArrowField> = Vec::with_capacity(self.builders.len() + 1);

        fields.push(ArrowField::new("num_values", DataType::Int64, false));
        let _ = self.builders.iter().map(|(_, (field, builder))| {
            fields.push(ArrowField::new(
                &field.name,
                field.data_type(),
                field.nullable,
            ));
        });
        let arrow_schema = ArrowSchema::new(fields);

        let num_rows = self.num_rows.finish();
        let min_values = self
            .get_builder(0)
            .unwrap()
            .min_value
            .as_any_mut()
            .downcast_mut::<PrimitiveBuilder<Int64Type>>()
            .unwrap()
            .finish();
        let max_values = self
            .get_builder(0)
            .unwrap()
            .max_value
            .as_any_mut()
            .downcast_mut::<PrimitiveBuilder<Int64Type>>()
            .unwrap()
            .finish();
        let null_count = self
            .get_builder(0)
            .unwrap()
            .null_count
            .as_any_mut()
            .downcast_mut::<PrimitiveBuilder<Int64Type>>()
            .unwrap()
            .finish();

        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema),
            vec![
                Arc::new(num_rows),
                Arc::new(min_values),
                Arc::new(max_values),
                Arc::new(null_count),
            ],
        )
        .unwrap();
        todo!()
    }
}

pub struct StatisticsBuilder {
    null_count: PrimitiveBuilder<UInt32Type>,
    min_value: Box<dyn ArrayBuilder>,
    max_value: Box<dyn ArrayBuilder>,
    dt: DataType,
}

impl StatisticsBuilder {
    fn new(data_type: &DataType) -> Self {
        let null_count = PrimitiveBuilder::<UInt32Type>::new();
        let min_value = make_builder(data_type, 1);
        let max_value = make_builder(data_type, 1);
        let dt = data_type.clone();
        Self {
            null_count,
            min_value,
            max_value,
            dt,
        }
    }

    fn statistics_appender<T: ArrowNumericType>(&mut self, row: StatisticsRow) {
        let ScalarValue::UInt32(Some(null_count)) = row.null_count else {
            todo!()
        };
        self.null_count.append_value(null_count);
        if let ScalarValue::Boolean(Some(min_value)) = row.min_value {
            let min_value = min_value;
            let min_builder = self
                .min_value
                .as_any_mut()
                .downcast_mut::<PrimitiveBuilder<T>>()
                .unwrap();
            // min_builder.append_value(min_value);
        }
        if let ScalarValue::Boolean(Some(max_value)) = row.max_value {
            let max_builder = self
                .max_value
                .as_any_mut()
                .downcast_mut::<PrimitiveBuilder<T>>()
                .unwrap();
            // max_builder.append_value(max_value);
        }
        todo!();
    }

    pub fn append(&mut self, row: StatisticsRow) {
        let dt = match self.dt {
            DataType::Int32 => {
                self.statistics_appender::<Int32Type>(row);
            }
            DataType::Int64 => {
                self.statistics_appender::<Int64Type>(row);
            }
            _ => todo!(),
        };
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{
        builder::StringDictionaryBuilder, BinaryArray, BooleanArray, Date32Array, Decimal128Array,
        DurationMillisecondArray, Float32Array, Int32Array, Int64Array, StringArray, StructArray,
        TimestampMicrosecondArray,
    };

    use arrow_schema::{Field as ArrowField, Fields as ArrowFields, Schema as ArrowSchema};
    use datafusion::common::ScalarValue::Utf8;

    use super::*;

    #[test]
    fn test_collect_primitive_stats() {
        struct TestCase {
            source_arrays: Vec<ArrayRef>,
            expected_min: ScalarValue,
            expected_max: ScalarValue,
        }

        let cases: [TestCase; 7] = [
            // Int64
            TestCase {
                source_arrays: vec![
                    Arc::new(Int64Array::from(vec![4, 3, 7, 2])),
                    Arc::new(Int64Array::from(vec![-10, 3, 5])),
                ],
                expected_min: ScalarValue::from(-10 as i64),
                expected_max: ScalarValue::from(7 as i64),
            },
            // Int32
            TestCase {
                source_arrays: vec![
                    Arc::new(Int32Array::from(vec![4, 3, 7, 2])),
                    Arc::new(Int32Array::from(vec![-10, 3, 5])),
                ],
                expected_min: ScalarValue::from(-10 as i32),
                expected_max: ScalarValue::from(7 as i32),
            },
            // Boolean
            TestCase {
                source_arrays: vec![Arc::new(BooleanArray::from(vec![true, false]))],
                expected_min: ScalarValue::from(false),
                expected_max: ScalarValue::from(true),
            },
            // Date
            TestCase {
                source_arrays: vec![
                    Arc::new(Date32Array::from(vec![53, 42])),
                    Arc::new(Date32Array::from(vec![68, 32])),
                ],
                expected_min: ScalarValue::Date32(Some(32)),
                expected_max: ScalarValue::Date32(Some(68)),
            },
            // Timestamp
            TestCase {
                source_arrays: vec![
                    Arc::new(TimestampMicrosecondArray::from_vec(
                        vec![53, 42],
                        Some("UTC".into()),
                    )),
                    Arc::new(TimestampMicrosecondArray::from_vec(
                        vec![68, 32],
                        Some("UTC".into()),
                    )),
                ],
                expected_min: ScalarValue::TimestampMicrosecond(Some(32), Some("UTC".into())),
                expected_max: ScalarValue::TimestampMicrosecond(Some(68), Some("UTC".into())),
            },
            // Duration
            TestCase {
                source_arrays: vec![
                    Arc::new(DurationMillisecondArray::from(vec![53, 42])),
                    Arc::new(DurationMillisecondArray::from(vec![68, 32])),
                ],
                expected_min: ScalarValue::DurationMillisecond(Some(32)),
                expected_max: ScalarValue::DurationMillisecond(Some(68)),
            },
            // Decimal
            TestCase {
                source_arrays: vec![
                    Arc::new(Decimal128Array::from(vec![53, 42])),
                    Arc::new(Decimal128Array::from(vec![68, 32])),
                ],
                expected_min: ScalarValue::try_new_decimal128(32, 8, 8).unwrap(),
                expected_max: ScalarValue::try_new_decimal128(68, 8, 8).unwrap(),
            },
        ];

        for case in cases {
            let array_refs = case.source_arrays.iter().collect::<Vec<_>>();
            let stats = collect_statistics(&array_refs);
            assert_eq!(
                stats,
                StatisticsRow {
                    null_count: ScalarValue::UInt32(Some(0)),
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
                null_count: ScalarValue::from(0 as u32),
                min_value: ScalarValue::from(-10.0 as f32),
                max_value: ScalarValue::from(5.0 as f32),
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
                null_count: ScalarValue::from(0 as u32),
                min_value: ScalarValue::from(std::f32::NEG_INFINITY),
                max_value: ScalarValue::from(std::f32::INFINITY),
            }
        );

        // Zero is always positive
        let arrays: Vec<ArrayRef> = vec![Arc::new(Float32Array::from(vec![4.0f32, -0.0]))];
        let array_refs = arrays.iter().collect::<Vec<_>>();
        let stats = collect_statistics(&array_refs);
        assert_eq!(
            stats,
            StatisticsRow {
                null_count: ScalarValue::from(0 as u32),
                min_value: ScalarValue::from(0.0f32),
                max_value: ScalarValue::from(4.0f32),
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
                null_count: ScalarValue::from(1 as u32),
                min_value: ScalarValue::from("bar"),
                max_value: ScalarValue::from("yee"),
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
                null_count: ScalarValue::from(0 as u32),
                // Bacteriologists is just 15 bytes, but the next character is multi-byte
                // so we truncate before.
                min_value: ScalarValue::from("bacteriologists"),
                // Increment the last character to make sure it's greater than max value
                max_value: ScalarValue::from("terrestial pland"),
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
        let min_value: Vec<u8> = vec![0xFFu8; BINARY_PREFIX_LENGTH];
        assert_eq!(
            stats,
            StatisticsRow {
                null_count: ScalarValue::from(0 as u32),
                // We can truncate the minimum value, since the prefix is less than the full value
                min_value: ScalarValue::Binary(Some(min_value)),
                // We can't truncate the max value, so we return None
                max_value: ScalarValue::Binary(None),
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
                null_count: ScalarValue::from(1 as u32),
                min_value: ScalarValue::from("abc"),
                max_value: ScalarValue::from("def"),
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
            null_count: ScalarValue::from(2 as u32),
            min_value: ScalarValue::from(1 as i32),
            max_value: ScalarValue::from(3 as i32),
        });
        builder.append(StatisticsRow {
            null_count: ScalarValue::from(0 as u32),
            min_value: ScalarValue::Int32(None),
            max_value: ScalarValue::Int32(None),
        });

        // If we try to finish at this point, it will error since we don't have
        // stats for b yet.
        assert!(collector.finish().is_err());

        // Collect stats for b
        let id = schema.field("b").unwrap().id;
        let builder = collector.get_builder(id).unwrap();
        builder.append(StatisticsRow {
            null_count: ScalarValue::from(6 as u32),
            min_value: ScalarValue::from("aaa"),
            max_value: ScalarValue::from("bbb"),
        });
        builder.append(StatisticsRow {
            null_count: ScalarValue::from(0 as u32),
            min_value: ScalarValue::Utf8(Some(String::from(""))),
            max_value: ScalarValue::Utf8(Some(String::from(""))),
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
