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
    builder::{make_builder, ArrayBuilder, BooleanBuilder, PrimitiveBuilder},
    builder::{BinaryBuilder, StringBuilder},
    cast::{as_generic_binary_array, as_primitive_array, AsArray},
    types::{
        ArrowDictionaryKeyType, Date32Type, Date64Type, Decimal128Type, DurationMicrosecondType,
        DurationMillisecondType, DurationNanosecondType, DurationSecondType, Float32Type,
        Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, Time32MillisecondType,
        Time32SecondType, Time64MicrosecondType, Time64NanosecondType, TimestampMicrosecondType,
        TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt16Type,
        UInt32Type, UInt64Type, UInt8Type,
    },
    Array, ArrayRef, ArrowNumericType, ArrowPrimitiveType, OffsetSizeTrait, PrimitiveArray,
    RecordBatch, StructArray,
};
use arrow_schema::{ArrowError, DataType, Field as ArrowField, Schema as ArrowSchema, TimeUnit};
use datafusion::common::ScalarValue;
use lance_arrow::as_fixed_size_binary_array;
use num_traits::bounds::Bounded;
use std::str;

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

impl Default for StatisticsRow {
    fn default() -> Self {
        Self {
            null_count: ScalarValue::Int64(Some(0)),
            min_value: ScalarValue::Null,
            max_value: ScalarValue::Null,
        }
    }
}

fn get_primitive_statistics<T: ArrowNumericType>(
    arrays: &[&ArrayRef],
) -> (T::Native, T::Native, i64)
where
    T::Native: Bounded + PartialOrd,
{
    let mut min_value = T::Native::max_value();
    let mut max_value = T::Native::min_value();
    let mut null_count: i64 = 0;
    let mut all_values_null = true;
    let arrays_iterator = arrays.iter().map(|x| as_primitive_array::<T>(x));

    for array in arrays_iterator {
        null_count += array.null_count() as i64;
        if array.null_count() == array.len() {
            continue;
        }
        all_values_null = false;

        array.iter().for_each(|value| {
            if let Some(value) = value {
                if let Some(Ordering::Greater) = value.partial_cmp(&max_value) {
                    max_value = value;
                }
                if let Some(Ordering::Less) = value.partial_cmp(&min_value) {
                    min_value = value;
                }
            };
        });
    }

    if all_values_null {
        return (T::Native::min_value(), T::Native::max_value(), null_count);
    }
    (min_value, max_value, null_count)
}

fn get_statistics<T: ArrowNumericType>(arrays: &[&ArrayRef]) -> StatisticsRow
where
    T::Native: Bounded,
    datafusion::scalar::ScalarValue: From<<T as ArrowPrimitiveType>::Native>,
{
    let (min_value, max_value, null_count) = get_primitive_statistics::<T>(arrays);

    let mut scalar_min_value = ScalarValue::try_from(min_value).unwrap();
    let mut scalar_max_value = ScalarValue::try_from(max_value).unwrap();

    match arrays[0].data_type() {
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
        _ => {}
    }

    StatisticsRow {
        null_count: ScalarValue::Int64(Some(null_count)),
        min_value: scalar_min_value,
        max_value: scalar_max_value,
    }
}

fn get_decimal_statistics(arrays: &[&ArrayRef]) -> StatisticsRow {
    let (min_value, max_value, null_count) = get_primitive_statistics::<Decimal128Type>(arrays);
    let array = as_primitive_array::<Decimal128Type>(arrays[0]);
    let precision = array.precision();
    let scale = array.scale();

    StatisticsRow {
        null_count: ScalarValue::Int64(Some(null_count)),
        min_value: ScalarValue::Decimal128(Some(min_value), precision, scale),
        max_value: ScalarValue::Decimal128(Some(max_value), precision, scale),
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

/// Truncate a binary slice to make sure its length is less than `length`
fn truncate_binary(data: &[u8], length: usize) -> Option<&[u8]> {
    // We return values like that at an earlier stage in the process.
    assert!(data.len() >= length);
    // If all bytes are already maximal, no need to truncate

    Some(&data[0..length])
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

fn get_string_statistics<T: OffsetSizeTrait>(arrays: &[&ArrayRef]) -> StatisticsRow {
    let mut min_value: Option<&str> = None;
    let mut max_value: Option<&str> = None;
    let mut null_count: i64 = 0;

    let array_iterator = arrays.iter().map(|x| x.as_string::<T>());

    for array in array_iterator {
        null_count += array.null_count() as i64;
        if array.null_count() == array.len() {
            continue;
        }

        array.iter().for_each(|value| {
            if let Some(val) = value {
                // TODO: don't compare full strings
                // for i in (BINARY_PREFIX_LENGTH..val.len()).rev() {
                //     if val.is_char_boundary(i) {
                //         val = &val[..i];
                //         break;
                //     }
                // }
                // TODO: if we discovered a max value greater than expressable
                //  with BINARY_PREFIX_LENGTH we can skip comparing remaining values
                //  in the array.
                //  Same for min value.

                if let Some(v) = min_value {
                    if let Some(Ordering::Less) = val[..].partial_cmp(v) {
                        min_value = Some(val);
                    }
                } else {
                    min_value = Some(val);
                }

                if let Some(v) = max_value {
                    if let Some(Ordering::Greater) = val.partial_cmp(v) {
                        max_value = Some(val);
                    }
                } else {
                    max_value = Some(val);
                }
            }
        });
    }

    if let Some(v) = min_value {
        if v.len() > BINARY_PREFIX_LENGTH {
            min_value = truncate_utf8(v, BINARY_PREFIX_LENGTH);
        }
    }

    let max_value_bound: Vec<u8>;
    if let Some(v) = max_value {
        if v.len() > BINARY_PREFIX_LENGTH {
            max_value = truncate_utf8(v, BINARY_PREFIX_LENGTH);
            max_value_bound = increment_utf8(max_value.unwrap().as_bytes().to_vec()).unwrap();
            max_value = Some(str::from_utf8(&max_value_bound).unwrap());
        }
    }

    match arrays[0].data_type() {
        DataType::Utf8 => StatisticsRow {
            null_count: ScalarValue::Int64(Some(null_count)),
            min_value: ScalarValue::Utf8(Some(min_value.unwrap().to_string())),
            max_value: ScalarValue::Utf8(Some(max_value.unwrap().to_string())),
        },
        DataType::LargeUtf8 => StatisticsRow {
            null_count: ScalarValue::Int64(Some(null_count)),
            min_value: ScalarValue::LargeUtf8(Some(min_value.unwrap().to_string())),
            max_value: ScalarValue::LargeUtf8(Some(max_value.unwrap().to_string())),
        },
        _ => {
            todo!()
        }
    }
}

fn get_binary_statistics<T: OffsetSizeTrait>(arrays: &[&ArrayRef]) -> StatisticsRow {
    let mut min_value: Option<&[u8]> = None;
    let mut max_value: Option<&[u8]> = None;
    let mut null_count: i64 = 0;

    let array_iterator = arrays.iter().map(|x| as_generic_binary_array::<T>(x));

    for array in array_iterator {
        null_count += array.null_count() as i64;
        if array.null_count() == array.len() {
            continue;
        }

        array.iter().for_each(|value| {
            if let Some(val) = value {
                // don't compare full buffers if possible
                let val = &val[..std::cmp::min(BINARY_PREFIX_LENGTH + 4, val.len())];

                if let Some(v) = min_value {
                    if let Some(Ordering::Less) = val.partial_cmp(v) {
                        min_value = Some(val);
                    }
                } else {
                    min_value = Some(val);
                }

                if let Some(v) = max_value {
                    if let Some(Ordering::Greater) = val.partial_cmp(v) {
                        max_value = Some(val);
                    }
                } else {
                    max_value = Some(val);
                }
            }
        });
    }

    if let Some(v) = min_value {
        if v.len() > BINARY_PREFIX_LENGTH {
            min_value = truncate_binary(v, BINARY_PREFIX_LENGTH);
        }
    }

    let max_value_bound: Vec<u8>;
    if let Some(v) = max_value {
        if v.len() > BINARY_PREFIX_LENGTH {
            max_value = truncate_binary(v, BINARY_PREFIX_LENGTH);
            if let Some(x) = increment(max_value.unwrap().to_vec()) {
                max_value_bound = x;
                max_value = Some(&max_value_bound);
            } else {
                max_value = None;
            }
        }
    }

    match arrays[0].data_type() {
        DataType::Binary => StatisticsRow {
            null_count: ScalarValue::Int64(Some(null_count)),
            min_value: ScalarValue::Binary(min_value.map(|x| x.to_vec())),
            max_value: ScalarValue::Binary(max_value.map(|x| x.to_vec())),
        },
        DataType::LargeBinary => StatisticsRow {
            null_count: ScalarValue::Int64(Some(null_count)),
            min_value: ScalarValue::LargeBinary(min_value.map(|x| x.to_vec())),
            max_value: ScalarValue::LargeBinary(max_value.map(|x| x.to_vec())),
        },
        _ => {
            todo!()
        }
    }
}

fn get_fixed_size_binary_statistics(arrays: &[&ArrayRef]) -> StatisticsRow {
    let mut min_value: Option<&[u8]> = None;
    let mut max_value: Option<&[u8]> = None;
    let mut null_count: i64 = 0;

    let array_iterator = arrays.iter().map(|x| as_fixed_size_binary_array(x));

    let length = as_fixed_size_binary_array(arrays[0]).value_length() as usize;
    let length = std::cmp::min(BINARY_PREFIX_LENGTH, length);

    for array in array_iterator {
        null_count += array.null_count() as i64;
        if array.null_count() == array.len() {
            continue;
        }

        array.iter().for_each(|value| {
            if let Some(val) = value {
                // don't compare full buffers if possible
                let val = &val[..length];

                if let Some(v) = min_value {
                    if let Some(Ordering::Less) = val.partial_cmp(v) {
                        min_value = Some(val);
                    }
                } else {
                    min_value = Some(val);
                }

                if let Some(v) = max_value {
                    if let Some(Ordering::Greater) = val.partial_cmp(v) {
                        max_value = Some(val);
                    }
                } else {
                    max_value = Some(val);
                }
            }
        });
    }

    if let Some(v) = min_value {
        if v.len() > BINARY_PREFIX_LENGTH {
            min_value = truncate_binary(v, BINARY_PREFIX_LENGTH);
        }
    }

    let max_value_bound: Vec<u8>;
    if let Some(v) = max_value {
        if v.len() > BINARY_PREFIX_LENGTH {
            max_value = truncate_binary(v, BINARY_PREFIX_LENGTH);
            if let Some(x) = increment(max_value.unwrap().to_vec()) {
                max_value_bound = x;
                max_value = Some(&max_value_bound);
            } else {
                max_value = None;
            }
        }
    }

    StatisticsRow {
        null_count: ScalarValue::Int64(Some(null_count)),
        min_value: ScalarValue::FixedSizeBinary(length as i32, min_value.map(|x| x.to_vec())),
        max_value: ScalarValue::FixedSizeBinary(length as i32, max_value.map(|x| x.to_vec())),
    }
}

fn get_boolean_statistics(arrays: &[&ArrayRef]) -> StatisticsRow {
    let mut true_present = false;
    let mut false_present = false;
    let mut null_count: i64 = 0;

    let array_iterator = arrays.iter().map(|x| x.as_boolean());

    for array in array_iterator {
        null_count += array.null_count() as i64;
        if array.null_count() == array.len() {
            continue;
        }

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
        min_value: ScalarValue::Boolean(Some(true_present && !false_present)),
        max_value: ScalarValue::Boolean(Some(true_present || !false_present)),
    }
}
fn get_list_statistics(arrays: &[&ArrayRef]) -> StatisticsRow {
    let mut null_count: i64 = 0;
    let mut stats: StatisticsRow;
    match arrays[0].data_type() {
        DataType::List(_) => {
            let arrays = arrays
                .iter()
                .map(|x| {
                    null_count += x.null_count() as i64;
                    x.as_list::<i32>().values()
                })
                .collect::<Vec<_>>();
            stats = collect_statistics(&arrays);
        }
        DataType::LargeList(_) => {
            let arrays = arrays
                .iter()
                .map(|x| {
                    null_count += x.null_count() as i64;
                    x.as_list::<i64>().values()
                })
                .collect::<Vec<_>>();
            stats = collect_statistics(&arrays);
        }
        _ => {
            todo!()
        }
    }
    stats.null_count = ScalarValue::Int64(Some(null_count));
    stats
}

fn cast_dictionary_arrays<'a, T: ArrowDictionaryKeyType + 'static>(
    arrays: &'a [&'a ArrayRef],
) -> Vec<&Arc<dyn Array>> {
    arrays
        .iter()
        .map(|x| x.as_dictionary::<T>().values())
        .collect::<Vec<_>>()
}

fn get_dictionary_statistics(arrays: &[&ArrayRef]) -> StatisticsRow {
    let data_type = arrays[0].data_type();
    match data_type {
        DataType::Dictionary(key_type, _) => match key_type.as_ref() {
            DataType::Int8 => collect_statistics(&cast_dictionary_arrays::<Int8Type>(arrays)),
            DataType::Int16 => collect_statistics(&cast_dictionary_arrays::<Int16Type>(arrays)),
            DataType::Int32 => collect_statistics(&cast_dictionary_arrays::<Int32Type>(arrays)),
            DataType::Int64 => collect_statistics(&cast_dictionary_arrays::<Int64Type>(arrays)),
            DataType::UInt8 => collect_statistics(&cast_dictionary_arrays::<UInt8Type>(arrays)),
            DataType::UInt16 => collect_statistics(&cast_dictionary_arrays::<UInt16Type>(arrays)),
            DataType::UInt32 => collect_statistics(&cast_dictionary_arrays::<UInt32Type>(arrays)),
            DataType::UInt64 => collect_statistics(&cast_dictionary_arrays::<UInt64Type>(arrays)),
            _ => {
                panic!("Unsupported dictionary key type: {}", key_type);
            }
        },
        _ => {
            panic!("Unsupported data type for dictionary: {}", data_type);
        }
    }
}

fn get_temporal_statistics(arrays: &[&ArrayRef]) -> StatisticsRow {
    match arrays[0].data_type() {
        DataType::Time32(TimeUnit::Second) => {
            let (min_value, max_value, null_count) =
                get_primitive_statistics::<Time32SecondType>(arrays);
            StatisticsRow {
                null_count: ScalarValue::Int64(Some(null_count)),
                min_value: ScalarValue::Time32Second(Some(min_value)),
                max_value: ScalarValue::Time32Second(Some(max_value)),
            }
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            let (min_value, max_value, null_count) =
                get_primitive_statistics::<Time32MillisecondType>(arrays);
            StatisticsRow {
                null_count: ScalarValue::Int64(Some(null_count)),
                min_value: ScalarValue::Time32Millisecond(Some(min_value)),
                max_value: ScalarValue::Time32Millisecond(Some(max_value)),
            }
        }
        DataType::Date32 => {
            let (min_value, max_value, null_count) = get_primitive_statistics::<Date32Type>(arrays);
            StatisticsRow {
                null_count: ScalarValue::Int64(Some(null_count)),
                min_value: ScalarValue::Date32(Some(min_value)),
                max_value: ScalarValue::Date32(Some(max_value)),
            }
        }

        DataType::Timestamp(TimeUnit::Second, tz) => {
            let (min_value, max_value, null_count) =
                get_primitive_statistics::<TimestampSecondType>(arrays);
            StatisticsRow {
                null_count: ScalarValue::Int64(Some(null_count)),
                min_value: ScalarValue::TimestampSecond(Some(min_value), tz.clone()),
                max_value: ScalarValue::TimestampSecond(Some(max_value), tz.clone()),
            }
        }
        DataType::Timestamp(TimeUnit::Millisecond, tz) => {
            let (min_value, max_value, null_count) =
                get_primitive_statistics::<TimestampMillisecondType>(arrays);
            StatisticsRow {
                null_count: ScalarValue::Int64(Some(null_count)),
                min_value: ScalarValue::TimestampMillisecond(Some(min_value), tz.clone()),
                max_value: ScalarValue::TimestampMillisecond(Some(max_value), tz.clone()),
            }
        }
        DataType::Timestamp(TimeUnit::Microsecond, tz) => {
            let (min_value, max_value, null_count) =
                get_primitive_statistics::<TimestampMicrosecondType>(arrays);
            StatisticsRow {
                null_count: ScalarValue::Int64(Some(null_count)),
                min_value: ScalarValue::TimestampMicrosecond(Some(min_value), tz.clone()),
                max_value: ScalarValue::TimestampMicrosecond(Some(max_value), tz.clone()),
            }
        }
        DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
            let (min_value, max_value, null_count) =
                get_primitive_statistics::<TimestampNanosecondType>(arrays);
            StatisticsRow {
                null_count: ScalarValue::Int64(Some(null_count)),
                min_value: ScalarValue::TimestampNanosecond(Some(min_value), tz.clone()),
                max_value: ScalarValue::TimestampNanosecond(Some(max_value), tz.clone()),
            }
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            let (min_value, max_value, null_count) =
                get_primitive_statistics::<Time64MicrosecondType>(arrays);
            StatisticsRow {
                null_count: ScalarValue::Int64(Some(null_count)),
                min_value: ScalarValue::Time64Microsecond(Some(min_value)),
                max_value: ScalarValue::Time64Microsecond(Some(max_value)),
            }
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            let (min_value, max_value, null_count) =
                get_primitive_statistics::<Time64NanosecondType>(arrays);
            StatisticsRow {
                null_count: ScalarValue::Int64(Some(null_count)),
                min_value: ScalarValue::Time64Nanosecond(Some(min_value)),
                max_value: ScalarValue::Time64Nanosecond(Some(max_value)),
            }
        }
        DataType::Date64 => {
            let (min_value, max_value, null_count) = get_primitive_statistics::<Date64Type>(arrays);
            StatisticsRow {
                null_count: ScalarValue::Int64(Some(null_count)),
                min_value: ScalarValue::Date64(Some(min_value)),
                max_value: ScalarValue::Date64(Some(max_value)),
            }
        }
        DataType::Duration(TimeUnit::Second) => {
            let (min_value, max_value, null_count) =
                get_primitive_statistics::<DurationSecondType>(arrays);
            StatisticsRow {
                null_count: ScalarValue::Int64(Some(null_count)),
                min_value: ScalarValue::DurationSecond(Some(min_value)),
                max_value: ScalarValue::DurationSecond(Some(max_value)),
            }
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            let (min_value, max_value, null_count) =
                get_primitive_statistics::<DurationMillisecondType>(arrays);
            StatisticsRow {
                null_count: ScalarValue::Int64(Some(null_count)),
                min_value: ScalarValue::DurationMillisecond(Some(min_value)),
                max_value: ScalarValue::DurationMillisecond(Some(max_value)),
            }
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            let (min_value, max_value, null_count) =
                get_primitive_statistics::<DurationMicrosecondType>(arrays);
            StatisticsRow {
                null_count: ScalarValue::Int64(Some(null_count)),
                min_value: ScalarValue::DurationMicrosecond(Some(min_value)),
                max_value: ScalarValue::DurationMicrosecond(Some(max_value)),
            }
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            let (min_value, max_value, null_count) =
                get_primitive_statistics::<DurationNanosecondType>(arrays);
            StatisticsRow {
                null_count: ScalarValue::Int64(Some(null_count)),
                min_value: ScalarValue::DurationNanosecond(Some(min_value)),
                max_value: ScalarValue::DurationNanosecond(Some(max_value)),
            }
        }
        _ => {
            todo!()
        }
    }
}

pub fn collect_statistics(arrays: &[&ArrayRef]) -> StatisticsRow {
    if arrays.is_empty() {
        return StatisticsRow::default();
    }
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
        DataType::Date32 => get_temporal_statistics(arrays),
        DataType::Time32(_) => get_temporal_statistics(arrays),
        DataType::Date64 => get_temporal_statistics(arrays),
        DataType::Time64(_) => get_temporal_statistics(arrays),
        DataType::Timestamp(_, _) => get_temporal_statistics(arrays),
        DataType::Duration(_) => get_temporal_statistics(arrays),
        DataType::Decimal128(_, _) => get_decimal_statistics(arrays),
        DataType::Binary => get_binary_statistics::<i32>(arrays),
        DataType::LargeBinary => get_binary_statistics::<i64>(arrays),
        DataType::FixedSizeBinary(_) => get_fixed_size_binary_statistics(arrays),
        DataType::Utf8 => get_string_statistics::<i32>(arrays),
        DataType::LargeUtf8 => get_string_statistics::<i64>(arrays),
        DataType::Dictionary(_, _) => get_dictionary_statistics(arrays),
        DataType::List(_) => get_list_statistics(arrays),
        DataType::LargeList(_) => get_list_statistics(arrays),
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
    builders: BTreeMap<i32, (Field, StatisticsBuilder)>,
}

impl StatisticsCollector {
    pub fn new(fields: &[Field]) -> Self {
        let builders = fields
            .iter()
            .map(|f| (f.id, (f.clone(), StatisticsBuilder::new(&f.data_type()))))
            .collect();
        Self { builders }
    }

    pub fn get_builder(&mut self, field_id: i32) -> Option<&mut StatisticsBuilder> {
        self.builders.get_mut(&field_id).map(|(_, b)| b)
    }

    pub fn finish(&mut self) -> Result<RecordBatch> {
        let mut arrays: Vec<ArrayRef> = vec![];
        let mut fields: Vec<ArrowField> = vec![];

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
        let min_value_builder = self
            .min_value
            .as_any_mut()
            .downcast_mut::<StringBuilder>()
            .unwrap();
        let max_value_builder = self
            .max_value
            .as_any_mut()
            .downcast_mut::<StringBuilder>()
            .unwrap();

        if let ScalarValue::Int64(Some(null_count)) = row.null_count {
            self.null_count.append_value(null_count);
        } else {
            todo!()
        };

        if let ScalarValue::Utf8(Some(min_value)) = row.min_value {
            min_value_builder.append_value(min_value);
        } else {
            min_value_builder.append_null();
        }

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
        let ScalarValue::Binary(Some(min_value)) = row.min_value else {
            todo!()
        };
        let ScalarValue::Binary(Some(max_value)) = row.max_value else {
            todo!()
        };

        let min_value_builder = self
            .min_value
            .as_any_mut()
            .downcast_mut::<BinaryBuilder>()
            .unwrap();
        let max_value_builder = self
            .max_value
            .as_any_mut()
            .downcast_mut::<BinaryBuilder>()
            .unwrap();

        self.null_count.append_value(null_count);
        min_value_builder.append_value(min_value);
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
        let ScalarValue::Boolean(Some(max_value)) = row.max_value else {
            todo!()
        };
        let ScalarValue::Boolean(Some(min_value)) = row.min_value else {
            todo!()
        };

        let min_value_builder = self
            .min_value
            .as_any_mut()
            .downcast_mut::<BooleanBuilder>()
            .unwrap();
        let max_value_builder = self
            .max_value
            .as_any_mut()
            .downcast_mut::<BooleanBuilder>()
            .unwrap();

        self.null_count.append_value(null_count);
        min_value_builder.append_value(min_value);
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
            DataType::Time32(TimeUnit::Second) => self.statistics_appender::<Time32SecondType>(row),
            DataType::Time32(TimeUnit::Millisecond) => {
                self.statistics_appender::<Time32MillisecondType>(row)
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                self.statistics_appender::<Time64MicrosecondType>(row)
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                self.statistics_appender::<Time64NanosecondType>(row)
            }
            DataType::Timestamp(TimeUnit::Second, _) => {
                self.statistics_appender::<TimestampSecondType>(row)
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                self.statistics_appender::<TimestampMillisecondType>(row)
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                self.statistics_appender::<TimestampMicrosecondType>(row)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                self.statistics_appender::<TimestampNanosecondType>(row)
            }
            DataType::Duration(TimeUnit::Second) => {
                self.statistics_appender::<DurationSecondType>(row)
            }
            DataType::Duration(TimeUnit::Millisecond) => {
                self.statistics_appender::<DurationMillisecondType>(row)
            }
            DataType::Duration(TimeUnit::Microsecond) => {
                self.statistics_appender::<DurationMicrosecondType>(row)
            }
            DataType::Duration(TimeUnit::Nanosecond) => {
                self.statistics_appender::<DurationNanosecondType>(row)
            }
            DataType::Decimal128(_, _) => self.statistics_appender::<Decimal128Type>(row),
            DataType::Binary => self.binary_statistics_appender(row),
            DataType::Utf8 => self.string_statistics_appender(row),
            DataType::LargeUtf8 => self.string_statistics_appender(row),
            // Dictionary type is not needed here. We collected stats for values.
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
        builder::StringDictionaryBuilder, BinaryArray, BooleanArray, Date32Array, Date64Array,
        Decimal128Array, DictionaryArray, DurationMicrosecondArray, DurationMillisecondArray,
        DurationNanosecondArray, DurationSecondArray, FixedSizeBinaryArray, Float32Array,
        Float64Array, Int16Array, Int32Array, Int64Array, Int8Array, LargeBinaryArray,
        LargeListArray, LargeStringArray, ListArray, StringArray, StructArray,
        Time32MillisecondArray, Time32SecondArray, Time64MicrosecondArray, Time64NanosecondArray,
        TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    };

    use super::*;
    use arrow_schema::{Field as ArrowField, Fields as ArrowFields, Schema as ArrowSchema};

    #[test]
    fn test_edge_cases() {
        // No arrays, datatype can't be inferred
        let arrays: Vec<ArrayRef> = vec![];
        let array_refs = arrays.iter().collect::<Vec<_>>();
        assert_eq!(
            collect_statistics(array_refs.as_slice()),
            StatisticsRow {
                null_count: ScalarValue::Int64(Some(0)),
                min_value: ScalarValue::Null,
                max_value: ScalarValue::Null,
            }
        );

        // Empty arrays, default min/max values
        let arrays: Vec<ArrayRef> = vec![Arc::new(UInt32Array::from_iter_values(vec![]))];
        let array_refs = arrays.iter().collect::<Vec<_>>();
        assert_eq!(
            collect_statistics(array_refs.as_slice()),
            StatisticsRow {
                null_count: ScalarValue::Int64(Some(0)),
                min_value: ScalarValue::from(u32::MIN),
                max_value: ScalarValue::from(u32::MAX),
            }
        );
    }

    #[test]
    fn test_collect_primitive_stats() {
        struct TestCase {
            source_arrays: Vec<ArrayRef>,
            expected_min: ScalarValue,
            expected_max: ScalarValue,
            expected_null_count: i64,
        }

        let cases: [TestCase; 24] = [
            // Int8
            TestCase {
                source_arrays: vec![
                    Arc::new(Int8Array::from(vec![4, 3, 7, 2])),
                    Arc::new(Int8Array::from(vec![-10, 3, 5])),
                ],
                expected_min: ScalarValue::from(-10_i8),
                expected_max: ScalarValue::from(7_i8),
                expected_null_count: 0,
            },
            // UInt8
            TestCase {
                source_arrays: vec![
                    Arc::new(UInt8Array::from(vec![4, 3, 7, 2])),
                    Arc::new(UInt8Array::from(vec![10, 3, 5])),
                ],
                expected_min: ScalarValue::from(2_u8),
                expected_max: ScalarValue::from(10_u8),
                expected_null_count: 0,
            },
            // Int16
            TestCase {
                source_arrays: vec![
                    Arc::new(Int16Array::from(vec![4, 3, 7, 2])),
                    Arc::new(Int16Array::from(vec![-10, 3, 5])),
                ],
                expected_min: ScalarValue::from(-10_i16),
                expected_max: ScalarValue::from(7_i16),
                expected_null_count: 0,
            },
            // UInt16
            TestCase {
                source_arrays: vec![
                    Arc::new(UInt16Array::from(vec![4, 3, 7, 2])),
                    Arc::new(UInt16Array::from(vec![10, 3, 5])),
                ],
                expected_min: ScalarValue::from(2_u16),
                expected_max: ScalarValue::from(10_u16),
                expected_null_count: 0,
            },
            // Int32
            TestCase {
                source_arrays: vec![
                    Arc::new(Int32Array::from(vec![4, 3, 7, 2])),
                    Arc::new(Int32Array::from(vec![-10, 3, 5])),
                ],
                expected_min: ScalarValue::from(-10_i32),
                expected_max: ScalarValue::from(7_i32),
                expected_null_count: 0,
            },
            // UInt32
            TestCase {
                source_arrays: vec![
                    Arc::new(UInt32Array::from(vec![4, 3, 7, 2])),
                    Arc::new(UInt32Array::from(vec![10, 3, 5])),
                ],
                expected_min: ScalarValue::from(2_u32),
                expected_max: ScalarValue::from(10_u32),
                expected_null_count: 0,
            },
            // Int64
            TestCase {
                source_arrays: vec![
                    Arc::new(Int64Array::from(vec![4, 3, 7, 2])),
                    Arc::new(Int64Array::from(vec![-10, 3, 5])),
                ],
                expected_min: ScalarValue::from(-10_i64),
                expected_max: ScalarValue::from(7_i64),
                expected_null_count: 0,
            },
            // UInt64
            TestCase {
                source_arrays: vec![
                    Arc::new(UInt64Array::from(vec![4, 3, 7, 2])),
                    Arc::new(UInt64Array::from(vec![10, 3, 5])),
                ],
                expected_min: ScalarValue::from(2_u64),
                expected_max: ScalarValue::from(10_u64),
                expected_null_count: 0,
            },
            // Boolean
            TestCase {
                source_arrays: vec![Arc::new(BooleanArray::from(vec![true, false]))],
                expected_min: ScalarValue::from(false),
                expected_max: ScalarValue::from(true),
                expected_null_count: 0,
            },
            // Date
            TestCase {
                source_arrays: vec![
                    Arc::new(Date32Array::from(vec![53, 42])),
                    Arc::new(Date32Array::from(vec![68, 32])),
                ],
                expected_min: ScalarValue::Date32(Some(32)),
                expected_max: ScalarValue::Date32(Some(68)),
                expected_null_count: 0,
            },
            TestCase {
                source_arrays: vec![
                    Arc::new(Date64Array::from(vec![53, 42])),
                    Arc::new(Date64Array::from(vec![68, 32])),
                ],
                expected_min: ScalarValue::Date64(Some(32)),
                expected_max: ScalarValue::Date64(Some(68)),
                expected_null_count: 0,
            },
            // Time
            TestCase {
                source_arrays: vec![
                    Arc::new(Time32SecondArray::from(vec![53, 42])),
                    Arc::new(Time32SecondArray::from(vec![68, 32])),
                ],
                expected_min: ScalarValue::Time32Second(Some(32)),
                expected_max: ScalarValue::Time32Second(Some(68)),
                expected_null_count: 0,
            },
            TestCase {
                source_arrays: vec![
                    Arc::new(Time32MillisecondArray::from(vec![53, 42])),
                    Arc::new(Time32MillisecondArray::from(vec![68, 32])),
                ],
                expected_min: ScalarValue::Time32Millisecond(Some(32)),
                expected_max: ScalarValue::Time32Millisecond(Some(68)),
                expected_null_count: 0,
            },
            TestCase {
                source_arrays: vec![
                    Arc::new(Time64MicrosecondArray::from(vec![53, 42])),
                    Arc::new(Time64MicrosecondArray::from(vec![68, 32])),
                ],
                expected_min: ScalarValue::Time64Microsecond(Some(32)),
                expected_max: ScalarValue::Time64Microsecond(Some(68)),
                expected_null_count: 0,
            },
            TestCase {
                source_arrays: vec![
                    Arc::new(Time64NanosecondArray::from(vec![53, 42])),
                    Arc::new(Time64NanosecondArray::from(vec![68, 32])),
                ],
                expected_min: ScalarValue::Time64Nanosecond(Some(32)),
                expected_max: ScalarValue::Time64Nanosecond(Some(68)),
                expected_null_count: 0,
            },
            // Timestamp
            TestCase {
                source_arrays: vec![
                    Arc::new(TimestampSecondArray::with_timezone_opt(
                        vec![53, 42].into(),
                        Some("UTC"),
                    )),
                    Arc::new(TimestampSecondArray::with_timezone_opt(
                        vec![68, 32].into(),
                        Some("UTC"),
                    )),
                ],
                expected_min: ScalarValue::TimestampSecond(Some(32), Some("UTC".into())),
                expected_max: ScalarValue::TimestampSecond(Some(68), Some("UTC".into())),
                expected_null_count: 0,
            },
            TestCase {
                source_arrays: vec![
                    Arc::new(TimestampMillisecondArray::with_timezone_opt(
                        vec![53, 42].into(),
                        Some("UTC"),
                    )),
                    Arc::new(TimestampMillisecondArray::with_timezone_opt(
                        vec![68, 32].into(),
                        Some("UTC"),
                    )),
                ],
                expected_min: ScalarValue::TimestampMillisecond(Some(32), Some("UTC".into())),
                expected_max: ScalarValue::TimestampMillisecond(Some(68), Some("UTC".into())),
                expected_null_count: 0,
            },
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
                expected_null_count: 0,
            },
            TestCase {
                source_arrays: vec![
                    Arc::new(TimestampNanosecondArray::with_timezone_opt(
                        vec![53, 42].into(),
                        Some("UTC"),
                    )),
                    Arc::new(TimestampNanosecondArray::with_timezone_opt(
                        vec![68, 32].into(),
                        Some("UTC"),
                    )),
                ],
                expected_min: ScalarValue::TimestampNanosecond(Some(32), Some("UTC".into())),
                expected_max: ScalarValue::TimestampNanosecond(Some(68), Some("UTC".into())),
                expected_null_count: 0,
            },
            // Duration
            TestCase {
                source_arrays: vec![
                    Arc::new(DurationSecondArray::from(vec![53, 42])),
                    Arc::new(DurationSecondArray::from(vec![68, 32])),
                ],
                expected_min: ScalarValue::DurationSecond(Some(32)),
                expected_max: ScalarValue::DurationSecond(Some(68)),
                expected_null_count: 0,
            },
            TestCase {
                source_arrays: vec![
                    Arc::new(DurationMillisecondArray::from(vec![53, 42])),
                    Arc::new(DurationMillisecondArray::from(vec![68, 32])),
                ],
                expected_min: ScalarValue::DurationMillisecond(Some(32)),
                expected_max: ScalarValue::DurationMillisecond(Some(68)),
                expected_null_count: 0,
            },
            TestCase {
                source_arrays: vec![
                    Arc::new(DurationMicrosecondArray::from(vec![53, 42])),
                    Arc::new(DurationMicrosecondArray::from(vec![68, 32])),
                ],
                expected_min: ScalarValue::DurationMicrosecond(Some(32)),
                expected_max: ScalarValue::DurationMicrosecond(Some(68)),
                expected_null_count: 0,
            },
            TestCase {
                source_arrays: vec![
                    Arc::new(DurationNanosecondArray::from(vec![53, 42])),
                    Arc::new(DurationNanosecondArray::from(vec![68, 32])),
                ],
                expected_min: ScalarValue::DurationNanosecond(Some(32)),
                expected_max: ScalarValue::DurationNanosecond(Some(68)),
                expected_null_count: 0,
            },
            // Decimal
            TestCase {
                source_arrays: vec![
                    Arc::new(Decimal128Array::from(vec![53, 42])),
                    Arc::new(Decimal128Array::from(vec![68, 32])),
                ],
                expected_min: ScalarValue::try_new_decimal128(32, 38, 10).unwrap(),
                expected_max: ScalarValue::try_new_decimal128(68, 38, 10).unwrap(),
                expected_null_count: 0,
            },
        ];

        for case in cases {
            let array_refs = case.source_arrays.iter().collect::<Vec<_>>();
            let stats = collect_statistics(&array_refs);
            assert_eq!(
                stats,
                StatisticsRow {
                    min_value: case.expected_min,
                    max_value: case.expected_max,
                    null_count: case.expected_null_count.into(),
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
        let arrays: Vec<ArrayRef> = vec![Arc::new(Float64Array::from(vec![
            4.0f64,
            std::f64::INFINITY,
            std::f64::NEG_INFINITY,
        ]))];
        let array_refs = arrays.iter().collect::<Vec<_>>();
        let stats = collect_statistics(&array_refs);
        assert_eq!(
            stats,
            StatisticsRow {
                null_count: ScalarValue::from(0_i64),
                min_value: ScalarValue::from(std::f64::NEG_INFINITY),
                max_value: ScalarValue::from(std::f64::INFINITY),
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
    fn test_collect_list_stats() {
        let data1 = vec![
            Some(vec![Some(0)]),
            Some(vec![Some(9)]),
            Some(vec![Some(9), Some(2), Some(2)]),
        ];
        let data2 = vec![
            Some(vec![Some(0), Some(1), Some(2)]),
            None,
            Some(vec![Some(3), None]),
            Some(vec![Some(6), Some(7), Some(8)]),
        ];

        let expected_stats = StatisticsRow {
            null_count: ScalarValue::from(1_i64),
            min_value: ScalarValue::from(0_i16),
            max_value: ScalarValue::from(9_i16),
        };
        let arrays = vec![
            Arc::new(ListArray::from_iter_primitive::<Int16Type, _, _>(
                data1.clone(),
            )) as ArrayRef,
            Arc::new(ListArray::from_iter_primitive::<Int16Type, _, _>(
                data2.clone(),
            )) as ArrayRef,
        ];

        let binding = arrays.iter().collect::<Vec<_>>();
        let array_refs = binding.as_slice();
        let stats = collect_statistics(array_refs);
        assert_eq!(stats, expected_stats);
    }

    #[test]
    fn test_collect_large_list_stats() {
        let data1 = vec![
            Some(vec![Some(0)]),
            Some(vec![Some(9)]),
            Some(vec![Some(9), Some(2), Some(2)]),
        ];
        let data2 = vec![
            Some(vec![Some(0), Some(1), Some(2)]),
            None,
            Some(vec![Some(3), None]),
            Some(vec![Some(6), Some(7), Some(8)]),
        ];
        let expected_stats = StatisticsRow {
            null_count: ScalarValue::from(1_i64),
            min_value: ScalarValue::from(0_i64),
            max_value: ScalarValue::from(9_i64),
        };
        let arrays = vec![
            Arc::new(LargeListArray::from_iter_primitive::<Int64Type, _, _>(
                data1.clone(),
            )) as ArrayRef,
            Arc::new(LargeListArray::from_iter_primitive::<Int64Type, _, _>(
                data2.clone(),
            )) as ArrayRef,
        ];

        let binding = arrays.iter().collect::<Vec<_>>();
        let array_refs = binding.as_slice();
        let stats = collect_statistics(array_refs);
        assert_eq!(stats, expected_stats);
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

        let filler = "48 chars of filler                              ";
        // Prefixes are used if strings are too long. Multi-byte characters are
        // not split.
        let arrays: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![
            format!("{}{}", filler, "bacteriologists🧑‍🔬"),
            format!("{}{}", filler, "terrestial planet"),
        ]))];
        let array_refs = arrays.iter().collect::<Vec<_>>();
        let stats = collect_statistics(&array_refs);
        assert_eq!(
            stats,
            StatisticsRow {
                null_count: ScalarValue::from(0_i64),
                // Bacteriologists is just 15 bytes, but the next character is multi-byte
                // so we truncate before.
                min_value: ScalarValue::from(format!("{}{}", filler, "bacteriologists").as_str()),
                // Increment the last character to make sure it's greater than max value
                max_value: ScalarValue::from(format!("{}{}", filler, "terrestial planf").as_str()),
            }
        );

        // Whole strings are used if short enough
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(LargeStringArray::from(vec![Some("foo"), None, Some("bar")])),
            Arc::new(LargeStringArray::from(vec!["yee", "haw"])),
        ];
        let array_refs = arrays.iter().collect::<Vec<_>>();
        let stats = collect_statistics(&array_refs);
        assert_eq!(
            stats,
            StatisticsRow {
                null_count: ScalarValue::from(1_i64),
                min_value: ScalarValue::LargeUtf8(Some("bar".to_string())),
                max_value: ScalarValue::LargeUtf8(Some("yee".to_string())),
            }
        );

        let filler = "48 chars of filler                              ";
        // Prefixes are used if strings are too long. Multi-byte characters are
        // not split.
        let arrays: Vec<ArrayRef> = vec![Arc::new(LargeStringArray::from(vec![
            format!("{}{}", filler, "bacteriologists🧑‍🔬"),
            format!("{}{}", filler, "terrestial planet"),
        ]))];
        let array_refs = arrays.iter().collect::<Vec<_>>();
        let stats = collect_statistics(&array_refs);
        assert_eq!(
            stats,
            StatisticsRow {
                null_count: ScalarValue::from(0_i64),
                // Bacteriologists is just 15 bytes, but the next character is multi-byte
                // so we truncate before.
                min_value: ScalarValue::LargeUtf8(Some(format!("{}{}", filler, "bacteriologists"))),
                // Increment the last character to make sure it's greater than max value
                max_value: ScalarValue::LargeUtf8(Some(format!(
                    "{}{}",
                    filler, "terrestial planf"
                ))),
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

        let arrays: Vec<ArrayRef> = vec![Arc::new(LargeBinaryArray::from(vec![vec![
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
                min_value: ScalarValue::LargeBinary(Some(min_value)),
                // We can't truncate the max value, so we return None
                max_value: ScalarValue::LargeBinary(None),
            }
        );

        let arrays: Vec<ArrayRef> = vec![Arc::new(FixedSizeBinaryArray::from(vec![
            Some(vec![0, 1].as_slice()),
            Some(vec![2, 3].as_slice()),
            Some(vec![4, 5].as_slice()),
            Some(vec![6, 7].as_slice()),
            Some(vec![8, 9].as_slice()),
        ]))];
        let array_refs = arrays.iter().collect::<Vec<_>>();
        let stats = collect_statistics(&array_refs);
        let min_value: Vec<u8> = vec![0, 1];
        let max_value: Vec<u8> = vec![8, 9];
        assert_eq!(
            stats,
            StatisticsRow {
                null_count: ScalarValue::from(0_i64),
                min_value: ScalarValue::FixedSizeBinary(2, Some(min_value)),
                max_value: ScalarValue::FixedSizeBinary(2, Some(max_value)),
            }
        );
    }

    #[test]
    fn test_collect_dictionary_stats() {
        // Dictionary stats are collected from the underlying values
        let dictionary_values = StringArray::from(vec![None, Some("abc"), Some("def")]);

        let mut builder =
            StringDictionaryBuilder::<Int32Type>::new_with_dictionary(3, &dictionary_values)
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

        let dictionary = Arc::new(StringArray::from(vec!["A", "C", "G", "T"]));
        let indices_1 = UInt32Array::from(vec![1, 0, 2, 1]);
        let indices_2 = UInt32Array::from(vec![0, 1, 3, 0]);
        let dictionary_array_1 =
            Arc::new(DictionaryArray::try_new(indices_1, dictionary.clone()).unwrap()) as ArrayRef;
        let dictionary_array_2 =
            Arc::new(DictionaryArray::try_new(indices_2, dictionary.clone()).unwrap()) as ArrayRef;
        let array_refs = vec![&dictionary_array_1, &dictionary_array_2];
        let stats = collect_statistics(&array_refs);
        assert_eq!(
            stats,
            StatisticsRow {
                null_count: ScalarValue::from(0_i64),
                min_value: ScalarValue::from("A"),
                max_value: ScalarValue::from("T"),
            }
        );
    }

    #[test]
    fn test_stats_collector() {
        use crate::datatypes::Schema;

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

        // Now we can finish
        let batch = collector.finish().unwrap();

        let expected_schema = ArrowSchema::new(vec![
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
            ],
        )
        .unwrap();

        assert_eq!(batch, expected_batch);
    }
}
