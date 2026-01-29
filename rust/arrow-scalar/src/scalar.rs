// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Core Scalar enum definition for representing single Arrow values.

use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_buffer::{i256, Buffer};
use arrow_schema::{DataType, Field, Fields, IntervalUnit, TimeUnit};
use half::f16;

/// A single value in Arrow format.
///
/// This enum represents a single scalar value for each Arrow data type.
/// It is similar to DataFusion's `ScalarValue` but without DataFusion dependencies.
///
/// For primitive types, the value is wrapped in `Option` where `None` represents null.
/// For complex types like List and Struct, null values are represented within the
/// contained arrays or vectors.
#[derive(Clone, Debug)]
pub enum Scalar {
    /// Null value with unknown type
    Null,

    /// Boolean value
    Boolean(Option<bool>),

    /// Signed 8-bit integer
    Int8(Option<i8>),
    /// Signed 16-bit integer
    Int16(Option<i16>),
    /// Signed 32-bit integer
    Int32(Option<i32>),
    /// Signed 64-bit integer
    Int64(Option<i64>),

    /// Unsigned 8-bit integer
    UInt8(Option<u8>),
    /// Unsigned 16-bit integer
    UInt16(Option<u16>),
    /// Unsigned 32-bit integer
    UInt32(Option<u32>),
    /// Unsigned 64-bit integer
    UInt64(Option<u64>),

    /// 16-bit floating point
    Float16(Option<f16>),
    /// 32-bit floating point
    Float32(Option<f32>),
    /// 64-bit floating point
    Float64(Option<f64>),

    /// 128-bit decimal with precision and scale
    Decimal128(Option<i128>, u8, i8),
    /// 256-bit decimal with precision and scale
    Decimal256(Option<i256>, u8, i8),

    /// UTF-8 encoded string
    Utf8(Option<Buffer>),
    /// UTF-8 encoded string with 64-bit offsets
    LargeUtf8(Option<Buffer>),
    /// UTF-8 encoded string view
    Utf8View(Option<Buffer>),

    /// Variable-length binary data
    Binary(Option<Buffer>),
    /// Variable-length binary data with 64-bit offsets
    LargeBinary(Option<Buffer>),
    /// Binary view
    BinaryView(Option<Buffer>),
    /// Fixed-size binary data (size in bytes, value)
    FixedSizeBinary(i32, Option<Buffer>),

    /// Days since Unix epoch
    Date32(Option<i32>),
    /// Milliseconds since Unix epoch
    Date64(Option<i64>),

    /// Time of day with specified unit (only Second and Millisecond valid)
    Time32(Option<i32>, TimeUnit),
    /// Time of day with specified unit (only Microsecond and Nanosecond valid)
    Time64(Option<i64>, TimeUnit),

    /// Timestamp with time unit and optional timezone
    Timestamp(Option<i64>, TimeUnit, Option<Arc<str>>),

    /// Duration with time unit
    Duration(Option<i64>, TimeUnit),

    /// Interval in months
    IntervalYearMonth(Option<i32>),
    /// Interval in days and milliseconds (stored as i64: days in lower 32 bits, ms in upper 32 bits)
    IntervalDayTime(Option<i64>),
    /// Interval in months, days, and nanoseconds (stored as i128)
    IntervalMonthDayNano(Option<i128>),

    /// List array (stored as a length-1 array for the element)
    List(ArrayRef),
    /// Large list array (stored as a length-1 array for the element)
    LargeList(ArrayRef),
    /// Fixed-size list array (stored as a length-1 array for the element)
    FixedSizeList(ArrayRef),

    /// Struct with named fields
    Struct(Fields, Vec<Scalar>),

    /// Map array (stored as a length-1 array)
    Map(ArrayRef),

    /// Dictionary-encoded value (key type, value)
    Dictionary(Box<DataType>, Box<Scalar>),
}

impl Scalar {
    /// Returns the Arrow [`DataType`] for this scalar.
    pub fn data_type(&self) -> DataType {
        match self {
            Self::Null => DataType::Null,
            Self::Boolean(_) => DataType::Boolean,
            Self::Int8(_) => DataType::Int8,
            Self::Int16(_) => DataType::Int16,
            Self::Int32(_) => DataType::Int32,
            Self::Int64(_) => DataType::Int64,
            Self::UInt8(_) => DataType::UInt8,
            Self::UInt16(_) => DataType::UInt16,
            Self::UInt32(_) => DataType::UInt32,
            Self::UInt64(_) => DataType::UInt64,
            Self::Float16(_) => DataType::Float16,
            Self::Float32(_) => DataType::Float32,
            Self::Float64(_) => DataType::Float64,
            Self::Decimal128(_, precision, scale) => DataType::Decimal128(*precision, *scale),
            Self::Decimal256(_, precision, scale) => DataType::Decimal256(*precision, *scale),
            Self::Utf8(_) => DataType::Utf8,
            Self::LargeUtf8(_) => DataType::LargeUtf8,
            Self::Utf8View(_) => DataType::Utf8View,
            Self::Binary(_) => DataType::Binary,
            Self::LargeBinary(_) => DataType::LargeBinary,
            Self::BinaryView(_) => DataType::BinaryView,
            Self::FixedSizeBinary(size, _) => DataType::FixedSizeBinary(*size),
            Self::Date32(_) => DataType::Date32,
            Self::Date64(_) => DataType::Date64,
            Self::Time32(_, unit) => DataType::Time32(*unit),
            Self::Time64(_, unit) => DataType::Time64(*unit),
            Self::Timestamp(_, unit, tz) => DataType::Timestamp(*unit, tz.clone()),
            Self::Duration(_, unit) => DataType::Duration(*unit),
            Self::IntervalYearMonth(_) => DataType::Interval(IntervalUnit::YearMonth),
            Self::IntervalDayTime(_) => DataType::Interval(IntervalUnit::DayTime),
            Self::IntervalMonthDayNano(_) => DataType::Interval(IntervalUnit::MonthDayNano),
            Self::List(arr) => {
                let list_arr = arr
                    .as_any()
                    .downcast_ref::<arrow_array::ListArray>()
                    .expect("List scalar must contain ListArray");
                DataType::List(Arc::new(Field::new("item", list_arr.value_type(), true)))
            }
            Self::LargeList(arr) => {
                let list_arr = arr
                    .as_any()
                    .downcast_ref::<arrow_array::LargeListArray>()
                    .expect("LargeList scalar must contain LargeListArray");
                DataType::LargeList(Arc::new(Field::new("item", list_arr.value_type(), true)))
            }
            Self::FixedSizeList(arr) => {
                let list_arr = arr
                    .as_any()
                    .downcast_ref::<arrow_array::FixedSizeListArray>()
                    .expect("FixedSizeList scalar must contain FixedSizeListArray");
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", list_arr.value_type(), true)),
                    list_arr.value_length(),
                )
            }
            Self::Struct(fields, _) => DataType::Struct(fields.clone()),
            Self::Map(arr) => {
                let map_arr = arr
                    .as_any()
                    .downcast_ref::<arrow_array::MapArray>()
                    .expect("Map scalar must contain MapArray");
                DataType::Map(map_arr.entries().fields().first().unwrap().clone(), false)
            }
            Self::Dictionary(key_type, value) => {
                DataType::Dictionary(key_type.clone(), Box::new(value.data_type()))
            }
        }
    }

    /// Returns `true` if this scalar is null.
    pub fn is_null(&self) -> bool {
        match self {
            Self::Null => true,
            Self::Boolean(v) => v.is_none(),
            Self::Int8(v) => v.is_none(),
            Self::Int16(v) => v.is_none(),
            Self::Int32(v) => v.is_none(),
            Self::Int64(v) => v.is_none(),
            Self::UInt8(v) => v.is_none(),
            Self::UInt16(v) => v.is_none(),
            Self::UInt32(v) => v.is_none(),
            Self::UInt64(v) => v.is_none(),
            Self::Float16(v) => v.is_none(),
            Self::Float32(v) => v.is_none(),
            Self::Float64(v) => v.is_none(),
            Self::Decimal128(v, _, _) => v.is_none(),
            Self::Decimal256(v, _, _) => v.is_none(),
            Self::Utf8(v) => v.is_none(),
            Self::LargeUtf8(v) => v.is_none(),
            Self::Utf8View(v) => v.is_none(),
            Self::Binary(v) => v.is_none(),
            Self::LargeBinary(v) => v.is_none(),
            Self::BinaryView(v) => v.is_none(),
            Self::FixedSizeBinary(_, v) => v.is_none(),
            Self::Date32(v) => v.is_none(),
            Self::Date64(v) => v.is_none(),
            Self::Time32(v, _) => v.is_none(),
            Self::Time64(v, _) => v.is_none(),
            Self::Timestamp(v, _, _) => v.is_none(),
            Self::Duration(v, _) => v.is_none(),
            Self::IntervalYearMonth(v) => v.is_none(),
            Self::IntervalDayTime(v) => v.is_none(),
            Self::IntervalMonthDayNano(v) => v.is_none(),
            Self::List(arr) => arr.is_null(0),
            Self::LargeList(arr) => arr.is_null(0),
            Self::FixedSizeList(arr) => arr.is_null(0),
            Self::Struct(_, values) => values.is_empty(),
            Self::Map(arr) => arr.is_null(0),
            Self::Dictionary(_, v) => v.is_null(),
        }
    }

    /// Returns an estimate of the memory size in bytes.
    pub fn size(&self) -> usize {
        std::mem::size_of::<Self>()
            + match self {
                Self::Null
                | Self::Boolean(_)
                | Self::Int8(_)
                | Self::Int16(_)
                | Self::Int32(_)
                | Self::Int64(_)
                | Self::UInt8(_)
                | Self::UInt16(_)
                | Self::UInt32(_)
                | Self::UInt64(_)
                | Self::Float16(_)
                | Self::Float32(_)
                | Self::Float64(_)
                | Self::Decimal128(_, _, _)
                | Self::Decimal256(_, _, _)
                | Self::Date32(_)
                | Self::Date64(_)
                | Self::Time32(_, _)
                | Self::Time64(_, _)
                | Self::Timestamp(_, _, _)
                | Self::Duration(_, _)
                | Self::IntervalYearMonth(_)
                | Self::IntervalDayTime(_)
                | Self::IntervalMonthDayNano(_) => 0,
                Self::Utf8(v)
                | Self::LargeUtf8(v)
                | Self::Utf8View(v)
                | Self::Binary(v)
                | Self::LargeBinary(v)
                | Self::BinaryView(v)
                | Self::FixedSizeBinary(_, v) => v.as_ref().map(|b| b.len()).unwrap_or(0),
                Self::List(arr)
                | Self::LargeList(arr)
                | Self::FixedSizeList(arr)
                | Self::Map(arr) => arr.get_array_memory_size(),
                Self::Struct(fields, values) => {
                    fields.iter().map(|f| f.size()).sum::<usize>()
                        + values.iter().map(|v| v.size()).sum::<usize>()
                }
                Self::Dictionary(_, v) => v.size(),
            }
    }

    /// Creates a null scalar for the given data type.
    pub fn null_for_type(data_type: &DataType) -> Self {
        match data_type {
            DataType::Null => Self::Null,
            DataType::Boolean => Self::Boolean(None),
            DataType::Int8 => Self::Int8(None),
            DataType::Int16 => Self::Int16(None),
            DataType::Int32 => Self::Int32(None),
            DataType::Int64 => Self::Int64(None),
            DataType::UInt8 => Self::UInt8(None),
            DataType::UInt16 => Self::UInt16(None),
            DataType::UInt32 => Self::UInt32(None),
            DataType::UInt64 => Self::UInt64(None),
            DataType::Float16 => Self::Float16(None),
            DataType::Float32 => Self::Float32(None),
            DataType::Float64 => Self::Float64(None),
            DataType::Decimal128(p, s) => Self::Decimal128(None, *p, *s),
            DataType::Decimal256(p, s) => Self::Decimal256(None, *p, *s),
            DataType::Utf8 => Self::Utf8(None),
            DataType::LargeUtf8 => Self::LargeUtf8(None),
            DataType::Utf8View => Self::Utf8View(None),
            DataType::Binary => Self::Binary(None),
            DataType::LargeBinary => Self::LargeBinary(None),
            DataType::BinaryView => Self::BinaryView(None),
            DataType::FixedSizeBinary(size) => Self::FixedSizeBinary(*size, None),
            DataType::Date32 => Self::Date32(None),
            DataType::Date64 => Self::Date64(None),
            DataType::Time32(unit) => Self::Time32(None, *unit),
            DataType::Time64(unit) => Self::Time64(None, *unit),
            DataType::Timestamp(unit, tz) => Self::Timestamp(None, *unit, tz.clone()),
            DataType::Duration(unit) => Self::Duration(None, *unit),
            DataType::Interval(IntervalUnit::YearMonth) => Self::IntervalYearMonth(None),
            DataType::Interval(IntervalUnit::DayTime) => Self::IntervalDayTime(None),
            DataType::Interval(IntervalUnit::MonthDayNano) => Self::IntervalMonthDayNano(None),
            DataType::List(_) => {
                let empty = arrow_array::new_null_array(data_type, 1);
                Self::List(empty)
            }
            DataType::LargeList(_) => {
                let empty = arrow_array::new_null_array(data_type, 1);
                Self::LargeList(empty)
            }
            DataType::FixedSizeList(_, _) => {
                let empty = arrow_array::new_null_array(data_type, 1);
                Self::FixedSizeList(empty)
            }
            DataType::Struct(fields) => {
                let values = fields
                    .iter()
                    .map(|f| Self::null_for_type(f.data_type()))
                    .collect();
                Self::Struct(fields.clone(), values)
            }
            DataType::Map(_, _) => {
                let empty = arrow_array::new_null_array(data_type, 1);
                Self::Map(empty)
            }
            DataType::Dictionary(key_type, value_type) => {
                Self::Dictionary(key_type.clone(), Box::new(Self::null_for_type(value_type)))
            }
            _ => Self::Null,
        }
    }
}
