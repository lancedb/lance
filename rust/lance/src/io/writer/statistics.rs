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
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::datatypes::Field;
use crate::error::Result;
use arrow_array::{
    builder::{
        make_builder, ArrayBuilder, BinaryBuilder, BooleanBuilder, PrimitiveBuilder, StringBuilder,
    },
    types::{
        Date32Type, Date64Type, Decimal128Type, DurationMillisecondType, Float32Type, Float64Type,
        Int16Type, Int32Type, Int64Type, Int8Type, TimestampMicrosecondType, UInt16Type,
        UInt32Type, UInt64Type, UInt8Type,
    },
    Array, ArrayRef, ArrowNumericType, ArrowPrimitiveType, BinaryArray, BooleanArray,
    Decimal128Array, DictionaryArray, Int32Array, Int64Array, PrimitiveArray, RecordBatch,
    StringArray, StructArray,
};
use std::str;

use arrow_schema::DataType;
use arrow_schema::{ArrowError, Field as ArrowField, Schema as ArrowSchema};
use datafusion::common::ScalarValue;
use num_traits::bounds::Bounded;

/// Max number of bytes that are included in statistics for binary columns.
const BINARY_PREFIX_LENGTH: usize = 64;

/// Statistics for a single column chunk.
#[derive(Debug, PartialEq)]
pub struct StatisticsRow {
    /// Number of nulls in this column chunk.
    pub(crate) null_count: ScalarValue,
    /// Minimum value in this column chunk, if any
    pub(crate) min_value: ScalarValue,
    /// Maximum value in this column chunk, if any
    pub(crate) max_value: ScalarValue,
}

fn get_statistics<T: ArrowNumericType>(arrays: &[&ArrayRef]) -> StatisticsRow
where
    T::Native: Bounded + PartialOrd,
    datafusion::scalar::ScalarValue: From<<T as ArrowPrimitiveType>::Native>,
{
    let arr = arrays
        .iter()
        .map(|x| x.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap())
        .collect::<Vec<_>>();
    let mut min_value = T::Native::max_value();
    let mut max_value = T::Native::min_value();
    let mut null_count: i64 = 0;

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
        null_count += array.null_count() as i64;
    }

    let mut scalar_min_value = ScalarValue::try_from(min_value).unwrap();
    let mut scalar_max_value = ScalarValue::try_from(max_value).unwrap();

    // TODO: can type be inferred from scalar? Currently we'd lose timezones etc
    match arrays[0].data_type() {
        // TODO: add more float types?
        DataType::Float32 => {
            if scalar_min_value == ScalarValue::Float32(Some(0.0)) {
                scalar_min_value = ScalarValue::Float32(Some(-0.0));
            }
            if scalar_max_value == ScalarValue::Float32(Some(-0.0)) {
                scalar_max_value = ScalarValue::Float32(Some(0.0));
            }
        }
        DataType::Float64 => {
            if scalar_min_value == ScalarValue::Float64(Some(0.0)) {
                scalar_min_value = ScalarValue::Float64(Some(-0.0));
            }
            if scalar_max_value == ScalarValue::Float64(Some(-0.0)) {
                scalar_max_value = ScalarValue::Float64(Some(0.0));
            }
        }
        DataType::Date32 => {
            let min_value_date32 = scalar_min_value
                .to_array()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0);
            let max_value_date32 = scalar_max_value
                .to_array()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0);
            scalar_min_value = ScalarValue::Date32(Some(min_value_date32));
            scalar_max_value = ScalarValue::Date32(Some(max_value_date32));
        }
        DataType::Timestamp(_, _) => {
            // TODO: Grab the correct timezone from the array or should we ignore it for stats?
            // let arr = arrays[0].as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap();
            // let tz = Some(Arc::new(arr.timezone().unwrap()));
            let tz = Some("UTC".into());

            let min_value_timestamp = scalar_min_value
                .to_array()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0);
            let max_value_timestamp = scalar_max_value
                .to_array()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0);
            scalar_min_value =
                ScalarValue::TimestampMicrosecond(Some(min_value_timestamp), tz.clone());
            scalar_max_value =
                ScalarValue::TimestampMicrosecond(Some(max_value_timestamp), tz.clone());
        }
        DataType::Duration(_) => {
            let min_value_duration = scalar_min_value
                .to_array()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0);
            let max_value_duration = scalar_max_value
                .to_array()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0);
            scalar_min_value = ScalarValue::DurationMillisecond(Some(min_value_duration));
            scalar_max_value = ScalarValue::DurationMillisecond(Some(max_value_duration));
        }
        _ => {}
    }

    StatisticsRow {
        null_count: ScalarValue::Int64(Some(null_count)),
        min_value: scalar_min_value,
        max_value: scalar_max_value,
    }
}

fn get_decimal_statistics(arrays: &[&ArrayRef]) -> StatisticsRow {
    // TODO: should this be done with Decimal128 and not i128?
    let mut min_value = i128::MAX;
    let mut max_value = i128::MIN;
    let mut null_count: i64 = 0;

    let arr = arrays
        .iter()
        .map(|x| x.as_any().downcast_ref::<Decimal128Array>().unwrap())
        .collect::<Vec<_>>();

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
        null_count += array.null_count() as i64;
    }

    StatisticsRow {
        null_count: ScalarValue::Int64(Some(null_count)),
        min_value: ScalarValue::Decimal128(Some(min_value), 8, 8),
        max_value: ScalarValue::Decimal128(Some(max_value), 8, 8),
    }
}

/// Truncate a UTF8 slice to the longest prefix that is still a valid UTF8 string, while being less than `length` bytes.
fn truncate_utf8(data: &str, length: usize) -> Option<&str> {
    // We return values like that at an earlier stage in the process.
    assert!(data.len() >= length);
    let mut char_indices = data.char_indices();

    // We know `data` is a valid UTF8 encoded string, which means it has at least one valid UTF8 byte, which will make this loop exist.
    while let Some((idx, c)) = char_indices.next_back() {
        let split_point = idx + c.len_utf8();
        if split_point <= length {
            return Some(&data[0..split_point]);
        }
    }

    None
}

/// Truncate a binary slice to make sure its length is less than `length`
fn truncate_binary(data: &[u8], length: usize) -> Option<Vec<u8>> {
    // We return values like that at an earlier stage in the process.
    assert!(data.len() >= length);
    // If all bytes are already maximal, no need to truncate

    Some(data[0..length].to_vec())
}

/// Try and increment the bytes from right to left.
///
/// Returns `None` if all bytes are set to `u8::MAX`.
fn increment(mut data: Vec<u8>) -> Option<Vec<u8>> {
    for byte in data.iter_mut().rev() {
        let (incremented, overflow) = byte.overflowing_add(1);
        *byte = incremented;

        if !overflow {
            return Some(data);
        }
    }

    None
}

/// Try and increment the the string's bytes from right to left, returning when the result is a valid UTF8 string.
/// Returns `None` when it can't increment any byte.
fn increment_utf8(mut data: Vec<u8>) -> Option<Vec<u8>> {
    for idx in (0..data.len()).rev() {
        let original = data[idx];
        let (mut byte, mut overflow) = data[idx].overflowing_add(1);

        // Until overflow: 0xFF -> 0x00
        while !overflow {
            data[idx] = byte;

            if str::from_utf8(&data).is_ok() {
                return Some(data);
            }
            (byte, overflow) = data[idx].overflowing_add(1);
        }

        data[idx] = original;
    }

    None
}

fn truncate_min_value(data: &[u8]) -> Vec<u8> {
    match str::from_utf8(data) {
        Ok(str_data) => truncate_utf8(str_data, BINARY_PREFIX_LENGTH),
        Err(_) => truncate_binary(data, BINARY_PREFIX_LENGTH),
    }
    .unwrap_or_else(|| data.to_vec())
}

fn truncate_max_value(data: &[u8]) -> Vec<u8> {
    match str::from_utf8(data) {
        Ok(str_data) => truncate_utf8(str_data, BINARY_PREFIX_LENGTH).and_then(increment_utf8),
        Err(_) => truncate_binary(data, BINARY_PREFIX_LENGTH).and_then(increment),
    }
    .unwrap_or_else(|| data.to_vec())
}

fn get_string_statistics(arrays: &[&ArrayRef]) -> StatisticsRow {
    let mut min_value = vec![u8::MAX; BINARY_PREFIX_LENGTH];
    let mut max_value = vec![u8::MIN; BINARY_PREFIX_LENGTH];
    let mut null_count: i64 = 0;
    let mut total_count: i64 = 0;

    let arr = arrays
        .iter()
        .map(|x| x.as_any().downcast_ref::<StringArray>().unwrap())
        .collect::<Vec<_>>();

    for array in arr.iter() {
        total_count += array.len() as i64;
        array.iter().for_each(|value| {
            if let Some(value) = value {
                let value = truncate_utf8(value);
                if let Some(Ordering::Greater) = value.partial_cmp(max_value) {
                    max_value = value;
                }
                if let Some(Ordering::Less) = value.partial_cmp(min_value.as_slice()) {
                    min_value = value.to_vec();
                }
            }
        });
        null_count += array.null_count() as i64;
    }

    if total_count == 0 {
        return StatisticsRow {
            null_count: ScalarValue::Int64(Some(null_count)),
            min_value: ScalarValue::Utf8(None),
            max_value: ScalarValue::Utf8(None),
        };
    }

    if let Some(max_value) = &mut max_value {
        max_value = increment_utf8(max_value);
    }
    let min_value = str::from_utf8(&min_value).unwrap();
    let max_value = str::from_utf8(&max_value).unwrap();

    StatisticsRow {
        null_count: ScalarValue::Int64(Some(null_count)),
        min_value: ScalarValue::Utf8(Some(min_value.to_string())),
        max_value: ScalarValue::Utf8(Some(max_value.to_string())),
    }
}

fn get_binary_statistics(arrays: &[&ArrayRef]) -> StatisticsRow {
    let mut min_value = vec![u8::MAX; BINARY_PREFIX_LENGTH];
    let mut max_value = vec![u8::MIN; BINARY_PREFIX_LENGTH];
    let mut null_count: i64 = 0;

    let arr = arrays
        .iter()
        .map(|x| x.as_any().downcast_ref::<BinaryArray>().unwrap())
        .collect::<Vec<_>>();

    for array in arr.iter() {
        array.iter().for_each(|value| {
            if let Some(value) = value {
                if let Some(Ordering::Greater) = value.partial_cmp(max_value.as_slice()) {
                    max_value = value.to_vec();
                }
                if let Some(Ordering::Less) = value.partial_cmp(min_value.as_slice()) {
                    min_value = value.to_vec();
                }
            };
        });
        null_count += array.null_count() as i64;
    }
    let min_value_scalar = if min_value.len() > BINARY_PREFIX_LENGTH {
        None
    } else {
        Some(min_value)
    };
    let max_value_scalar = if max_value.len() > BINARY_PREFIX_LENGTH {
        None
    } else {
        Some(max_value)
    };

    StatisticsRow {
        null_count: ScalarValue::Int64(Some(null_count)),
        min_value: ScalarValue::Binary(min_value_scalar),
        max_value: ScalarValue::Binary(max_value_scalar),
    }
}

fn get_boolean_statistics(arrays: &[&ArrayRef]) -> StatisticsRow {
    let mut true_present = false;
    let mut false_present = false;
    let mut null_count: i64 = 0;

    let arrs = arrays
        .iter()
        .map(|x| {
            null_count += x.null_count() as i64;
            x.as_any().downcast_ref::<BooleanArray>().unwrap()
        })
        .collect::<Vec<_>>();

    for array in arrs.iter() {
        array.iter().for_each(|value| {
            if let Some(value) = value {
                if value {
                    true_present = true;
                } else {
                    false_present = true;
                }
            };
        });
        if true_present && false_present {
            break;
        }
    }

    StatisticsRow {
        null_count: ScalarValue::Int64(Some(null_count)),
        min_value: if false_present {
            ScalarValue::Boolean(Some(false))
        } else if true_present {
            ScalarValue::Boolean(Some(true))
        } else {
            ScalarValue::Boolean(None)
        },
        max_value: if true_present {
            ScalarValue::Boolean(Some(true))
        } else {
            ScalarValue::Boolean(None)
        },
    }
}

fn get_dictionary_statistics(arrays: &[&ArrayRef]) -> StatisticsRow {
    let arr = arrays
        .iter()
        .map(|x| {
            x.as_any()
                .downcast_ref::<DictionaryArray<UInt32Type>>()
                .unwrap()
        })
        .map(|x| x.values())
        .collect::<Vec<_>>();
    get_string_statistics(&arr)
}

pub fn collect_statistics(arrays: &[&ArrayRef]) -> StatisticsRow {
    match arrays[0].data_type() {
        DataType::Boolean => get_boolean_statistics(arrays),
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
        DataType::Decimal128(_, _) => get_decimal_statistics(arrays),
        DataType::Binary => get_binary_statistics(arrays),
        DataType::Utf8 => get_string_statistics(arrays),
        DataType::Dictionary(_, _) => get_dictionary_statistics(arrays),
        // TODO: struct type
        // DataType::Struct(_) =>
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
    builders: BTreeMap<i32, (Field, StatisticsBuilder)>,
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
        let mut arrays: Vec<ArrayRef> = vec![];
        let mut fields: Vec<ArrowField> = vec![];
        let num_rows = Arc::new(self.num_rows.finish());
        arrays.push(num_rows);
        fields.push(ArrowField::new("num_values", DataType::Int64, false));

        self.builders.iter_mut().for_each(|(_, (field, builder))| {
            let null_count = Arc::new(builder.null_count.finish());
            let min_value = Arc::new(builder.min_value.finish());
            let max_value = Arc::new(builder.max_value.finish());
            let struct_fields = vec![
                ArrowField::new("null_count", DataType::Int64, false),
                ArrowField::new("min_value", field.data_type(), field.nullable),
                ArrowField::new("max_value", field.data_type(), field.nullable),
            ];

            let stats = StructArray::new(
                struct_fields.clone().into(),
                vec![null_count.clone(), min_value, max_value],
                null_count.nulls().cloned(),
            );
            let field = ArrowField::new_struct(field.id.to_string(), struct_fields, false);
            fields.push(field);
            arrays.push(Arc::new(stats));
        });
        let schema = Arc::new(ArrowSchema::new(fields));
        let batch = RecordBatch::try_new(schema.clone(), arrays);
        match batch {
            Ok(batch) => Ok(batch),
            _ => Err(ArrowError::SchemaError(
                "all columns in a record batch must have the same length".to_string(),
            )
            .into()),
        }
    }
}

pub struct StatisticsBuilder {
    null_count: PrimitiveBuilder<Int64Type>,
    min_value: Box<dyn ArrayBuilder>,
    max_value: Box<dyn ArrayBuilder>,
    dt: DataType,
}

impl StatisticsBuilder {
    fn new(data_type: &DataType) -> Self {
        let null_count = PrimitiveBuilder::<Int64Type>::new();
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

    fn string_statistics_appender(&mut self, row: StatisticsRow) {
        let ScalarValue::Int64(Some(null_count)) = row.null_count else {
            todo!()
        };
        self.null_count.append_value(null_count);

        let min_value_builder = self
            .min_value
            .as_any_mut()
            .downcast_mut::<StringBuilder>()
            .unwrap();
        if let ScalarValue::Utf8(Some(min_value)) = row.min_value {
            min_value_builder.append_value(min_value);
        } else {
            min_value_builder.append_null();
        }

        let max_value_builder = self
            .max_value
            .as_any_mut()
            .downcast_mut::<StringBuilder>()
            .unwrap();
        if let ScalarValue::Utf8(Some(max_value)) = row.max_value {
            max_value_builder.append_value(max_value);
        } else {
            max_value_builder.append_null();
        }
    }

    fn binary_statistics_appender(&mut self, row: StatisticsRow) {
        let ScalarValue::Int64(Some(null_count)) = row.null_count else {
            todo!()
        };
        self.null_count.append_value(null_count);

        let min_value_builder = self
            .min_value
            .as_any_mut()
            .downcast_mut::<BinaryBuilder>()
            .unwrap();
        let binding = row.min_value.to_array();
        let min_value = binding
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap()
            .value(0);

        min_value_builder.append_value(min_value);

        let max_value_builder = self
            .max_value
            .as_any_mut()
            .downcast_mut::<BinaryBuilder>()
            .unwrap();
        let binding = row.max_value.to_array();
        let max_value = binding
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap()
            .value(0);
        max_value_builder.append_value(max_value);
    }

    fn statistics_appender<T: arrow_array::ArrowPrimitiveType>(&mut self, row: StatisticsRow) {
        let ScalarValue::Int64(Some(null_count)) = row.null_count else {
            todo!()
        };
        self.null_count.append_value(null_count);

        let min_value_builder = self
            .min_value
            .as_any_mut()
            .downcast_mut::<PrimitiveBuilder<T>>()
            .unwrap();
        let min_value = row
            .min_value
            .to_array()
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .unwrap()
            .value(0);
        min_value_builder.append_value(min_value);

        let max_value_builder = self
            .max_value
            .as_any_mut()
            .downcast_mut::<PrimitiveBuilder<T>>()
            .unwrap();
        let max_value = row
            .max_value
            .to_array()
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .unwrap()
            .value(0);
        max_value_builder.append_value(max_value);
    }

    fn boolean_appender(&mut self, row: StatisticsRow) {
        let ScalarValue::Int64(Some(null_count)) = row.null_count else {
            todo!()
        };
        self.null_count.append_value(null_count);

        let min_value_builder = self
            .min_value
            .as_any_mut()
            .downcast_mut::<BooleanBuilder>()
            .unwrap();
        let min_value = row
            .min_value
            .to_array()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .value(0);
        min_value_builder.append_value(min_value);

        let max_value_builder = self
            .max_value
            .as_any_mut()
            .downcast_mut::<BooleanBuilder>()
            .unwrap();
        let max_value = row
            .max_value
            .to_array()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .value(0);
        max_value_builder.append_value(max_value);
    }

    pub fn append(&mut self, row: StatisticsRow) {
        match self.dt {
            DataType::Boolean => self.boolean_appender(row),
            DataType::Int8 => self.statistics_appender::<Int8Type>(row),
            DataType::UInt8 => self.statistics_appender::<UInt8Type>(row),
            DataType::Int16 => self.statistics_appender::<Int16Type>(row),
            DataType::UInt16 => self.statistics_appender::<UInt16Type>(row),
            DataType::Int32 => self.statistics_appender::<Int32Type>(row),
            DataType::UInt32 => self.statistics_appender::<UInt32Type>(row),
            DataType::Int64 => self.statistics_appender::<Int64Type>(row),
            DataType::UInt64 => self.statistics_appender::<UInt64Type>(row),
            DataType::Float32 => self.statistics_appender::<Float32Type>(row),
            DataType::Float64 => self.statistics_appender::<Float64Type>(row),
            DataType::Date32 => self.statistics_appender::<Date32Type>(row),
            DataType::Date64 => self.statistics_appender::<Date64Type>(row),
            DataType::Timestamp(_, _) => self.statistics_appender::<TimestampMicrosecondType>(row),
            DataType::Duration(_) => self.statistics_appender::<DurationMillisecondType>(row),
            DataType::Decimal128(_, _) => self.statistics_appender::<Decimal128Type>(row),
            DataType::Binary => self.binary_statistics_appender(row),
            DataType::Utf8 => self.string_statistics_appender(row),
            DataType::Dictionary(_, _) => self.string_statistics_appender(row),
            // TODO: struct type
            // DataType::Struct(_) =>
            _ => {
                println!("Stats collection for {} is not supported yet", self.dt);
                todo!()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{
        builder::StringDictionaryBuilder, BinaryArray, BooleanArray, Date32Array,
        DurationMillisecondArray, Float32Array, Int32Array, Int64Array, StringArray, StructArray,
        TimestampMicrosecondArray,
    };

    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};

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
                expected_min: ScalarValue::from(-10_i64),
                expected_max: ScalarValue::from(7_i64),
            },
            // Int32
            TestCase {
                source_arrays: vec![
                    Arc::new(Int32Array::from(vec![4, 3, 7, 2])),
                    Arc::new(Int32Array::from(vec![-10, 3, 5])),
                ],
                expected_min: ScalarValue::from(-10_i32),
                expected_max: ScalarValue::from(7_i32),
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
                    Arc::new(TimestampMicrosecondArray::with_timezone_opt(
                        vec![53, 42].into(),
                        Some("UTC"),
                    )),
                    Arc::new(TimestampMicrosecondArray::with_timezone_opt(
                        vec![68, 32].into(),
                        Some("UTC"),
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
                    null_count: ScalarValue::Int64(Some(0)),
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
                null_count: ScalarValue::from(0_i64),
                min_value: ScalarValue::from(-10.0_f32),
                max_value: ScalarValue::from(5.0_f32),
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
                null_count: ScalarValue::from(0_i64),
                min_value: ScalarValue::from(std::f32::NEG_INFINITY),
                max_value: ScalarValue::from(std::f32::INFINITY),
            }
        );

        // Max value for zero is always positive, min value for zero is always negative
        let arrays: Vec<ArrayRef> = vec![Arc::new(Float32Array::from(vec![-0.0, 0.0]))];
        let array_refs = arrays.iter().collect::<Vec<_>>();
        let stats = collect_statistics(&array_refs);
        assert_eq!(
            stats,
            StatisticsRow {
                null_count: ScalarValue::from(0_i64),
                min_value: ScalarValue::from(-0.0_f32),
                max_value: ScalarValue::from(0.0_f32),
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
                null_count: ScalarValue::from(1_i64),
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
                null_count: ScalarValue::from(0_i64),
                // Bacteriologists is just 15 bytes, but the next character is multi-byte
                // so we truncate before.
                min_value: ScalarValue::from("bacteriologists"),
                // Increment the last character to make sure it's greater than max value
                max_value: ScalarValue::from("terrestial planf"), // TODO: Should this end pland?
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
                null_count: ScalarValue::from(0_i64),
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
                null_count: ScalarValue::from(1_i64),
                min_value: ScalarValue::from("abc"),
                max_value: ScalarValue::from("def"),
            }
        );
    }

    #[test]
    fn test_stats_collector() {
        use crate::datatypes::Schema;
        use arrow_schema::Fields as ArrowFields;

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
            null_count: ScalarValue::from(2_i64),
            min_value: ScalarValue::from(1_i32),
            max_value: ScalarValue::from(3_i32),
        });
        builder.append(StatisticsRow {
            null_count: ScalarValue::from(0_i64),
            min_value: ScalarValue::Int32(Some(std::i32::MIN)),
            max_value: ScalarValue::Int32(Some(std::i32::MAX)),
        });

        // If we try to finish at this point, it will error since we don't have
        // stats for b yet.
        assert!(collector.finish().is_err());

        // We cannot reuse old collector as it's builders were finished.
        let mut collector = StatisticsCollector::new(&schema.fields);

        let id = schema.field("a").unwrap().id;
        let builder = collector.get_builder(id).unwrap();
        builder.append(StatisticsRow {
            null_count: ScalarValue::from(2_i64),
            min_value: ScalarValue::from(1_i32),
            max_value: ScalarValue::from(3_i32),
        });
        builder.append(StatisticsRow {
            null_count: ScalarValue::from(0_i64),
            min_value: ScalarValue::Int32(Some(std::i32::MIN)),
            max_value: ScalarValue::Int32(Some(std::i32::MAX)),
        });

        // Collect stats for b
        let id = schema.field("b").unwrap().id;
        let builder = collector.get_builder(id).unwrap();
        builder.append(StatisticsRow {
            null_count: ScalarValue::from(6_i64),
            min_value: ScalarValue::from("aaa"),
            max_value: ScalarValue::from("bbb"),
        });
        builder.append(StatisticsRow {
            null_count: ScalarValue::from(0_i64),
            min_value: ScalarValue::Utf8(None),
            max_value: ScalarValue::Utf8(None),
        });

        collector.append_num_values(42);
        collector.append_num_values(64);

        // Now we can finish
        let batch = collector.finish().unwrap();

        // TODO: Order of schema is not guaranteed and stable so this ocassionaly fails.
        // Should we have a helper to compare batches?
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
                Arc::new(Int64Array::from(vec![42, 64])),
                Arc::new(StructArray::from(vec![
                    (
                        Arc::new(ArrowField::new("null_count", DataType::Int64, false)),
                        Arc::new(Int64Array::from(vec![6, 0])) as ArrayRef,
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
            ],
        )
        .unwrap();

        assert_eq!(batch, expected_batch);
    }
}
