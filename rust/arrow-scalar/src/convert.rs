// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Conversions between Arrow arrays and Scalar values.

use std::sync::Arc;

use arrow_array::{
    cast::AsArray,
    types::{
        BinaryViewType, Date32Type, Date64Type, Decimal128Type, Decimal256Type,
        DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType,
        DurationSecondType, Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type,
        Int8Type, IntervalDayTimeType, IntervalMonthDayNanoType, IntervalYearMonthType,
        StringViewType, Time32MillisecondType, Time32SecondType, Time64MicrosecondType,
        Time64NanosecondType, TimestampMicrosecondType, TimestampMillisecondType,
        TimestampNanosecondType, TimestampSecondType, UInt16Type, UInt32Type, UInt64Type,
        UInt8Type,
    },
    Array, ArrayRef, BooleanArray, FixedSizeBinaryArray, GenericByteArray, GenericByteViewArray,
    PrimitiveArray, StructArray,
};
use arrow_buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use arrow_data::transform::MutableArrayData;
use arrow_schema::{ArrowError, DataType, IntervalUnit, TimeUnit};

use crate::Scalar;

type Result<T> = std::result::Result<T, ArrowError>;

/// Extracts a scalar value from an array at the given index.
pub fn try_from_array(array: &dyn Array, index: usize) -> Result<Scalar> {
    if index >= array.len() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "Index {} out of bounds for array of length {}",
            index,
            array.len()
        )));
    }

    if array.is_null(index) {
        return Ok(Scalar::null_for_type(array.data_type()));
    }

    match array.data_type() {
        DataType::Null => Ok(Scalar::Null),
        DataType::Boolean => {
            let arr = array.as_boolean();
            Ok(Scalar::Boolean(Some(arr.value(index))))
        }
        DataType::Int8 => {
            let arr = array.as_primitive::<Int8Type>();
            Ok(Scalar::Int8(Some(arr.value(index))))
        }
        DataType::Int16 => {
            let arr = array.as_primitive::<Int16Type>();
            Ok(Scalar::Int16(Some(arr.value(index))))
        }
        DataType::Int32 => {
            let arr = array.as_primitive::<Int32Type>();
            Ok(Scalar::Int32(Some(arr.value(index))))
        }
        DataType::Int64 => {
            let arr = array.as_primitive::<Int64Type>();
            Ok(Scalar::Int64(Some(arr.value(index))))
        }
        DataType::UInt8 => {
            let arr = array.as_primitive::<UInt8Type>();
            Ok(Scalar::UInt8(Some(arr.value(index))))
        }
        DataType::UInt16 => {
            let arr = array.as_primitive::<UInt16Type>();
            Ok(Scalar::UInt16(Some(arr.value(index))))
        }
        DataType::UInt32 => {
            let arr = array.as_primitive::<UInt32Type>();
            Ok(Scalar::UInt32(Some(arr.value(index))))
        }
        DataType::UInt64 => {
            let arr = array.as_primitive::<UInt64Type>();
            Ok(Scalar::UInt64(Some(arr.value(index))))
        }
        DataType::Float16 => {
            let arr = array.as_primitive::<Float16Type>();
            Ok(Scalar::Float16(Some(arr.value(index))))
        }
        DataType::Float32 => {
            let arr = array.as_primitive::<Float32Type>();
            Ok(Scalar::Float32(Some(arr.value(index))))
        }
        DataType::Float64 => {
            let arr = array.as_primitive::<Float64Type>();
            Ok(Scalar::Float64(Some(arr.value(index))))
        }
        DataType::Decimal128(precision, scale) => {
            let arr = array.as_primitive::<Decimal128Type>();
            Ok(Scalar::Decimal128(
                Some(arr.value(index)),
                *precision,
                *scale,
            ))
        }
        DataType::Decimal256(precision, scale) => {
            let arr = array.as_primitive::<Decimal256Type>();
            Ok(Scalar::Decimal256(
                Some(arr.value(index)),
                *precision,
                *scale,
            ))
        }
        DataType::Utf8 => {
            let arr = array.as_string::<i32>();
            let offsets = arr.value_offsets();
            let start = offsets[index] as usize;
            let end = offsets[index + 1] as usize;
            let buf = arr.values().slice_with_length(start, end - start);
            Ok(Scalar::Utf8(Some(buf)))
        }
        DataType::LargeUtf8 => {
            let arr = array.as_string::<i64>();
            let offsets = arr.value_offsets();
            let start = offsets[index] as usize;
            let end = offsets[index + 1] as usize;
            let buf = arr.values().slice_with_length(start, end - start);
            Ok(Scalar::LargeUtf8(Some(buf)))
        }
        DataType::Utf8View => {
            let arr = array.as_string_view();
            Ok(Scalar::Utf8View(Some(Buffer::from(
                arr.value(index).as_bytes(),
            ))))
        }
        DataType::Binary => {
            let arr = array.as_binary::<i32>();
            let offsets = arr.value_offsets();
            let start = offsets[index] as usize;
            let end = offsets[index + 1] as usize;
            let buf = arr.values().slice_with_length(start, end - start);
            Ok(Scalar::Binary(Some(buf)))
        }
        DataType::LargeBinary => {
            let arr = array.as_binary::<i64>();
            let offsets = arr.value_offsets();
            let start = offsets[index] as usize;
            let end = offsets[index + 1] as usize;
            let buf = arr.values().slice_with_length(start, end - start);
            Ok(Scalar::LargeBinary(Some(buf)))
        }
        DataType::BinaryView => {
            let arr = array.as_binary_view();
            Ok(Scalar::BinaryView(Some(Buffer::from(arr.value(index)))))
        }
        DataType::FixedSizeBinary(size) => {
            let arr = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or_else(|| {
                    ArrowError::InvalidArgumentError("Expected FixedSizeBinaryArray".to_string())
                })?;
            let offset = arr.value_offset(index) as usize;
            let length = arr.value_length() as usize;
            let buf = arr.to_data().buffers()[0].slice_with_length(offset, length);
            Ok(Scalar::FixedSizeBinary(*size, Some(buf)))
        }
        DataType::Date32 => {
            let arr = array.as_primitive::<Date32Type>();
            Ok(Scalar::Date32(Some(arr.value(index))))
        }
        DataType::Date64 => {
            let arr = array.as_primitive::<Date64Type>();
            Ok(Scalar::Date64(Some(arr.value(index))))
        }
        DataType::Time32(TimeUnit::Second) => {
            let arr = array.as_primitive::<Time32SecondType>();
            Ok(Scalar::Time32(Some(arr.value(index)), TimeUnit::Second))
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            let arr = array.as_primitive::<Time32MillisecondType>();
            Ok(Scalar::Time32(
                Some(arr.value(index)),
                TimeUnit::Millisecond,
            ))
        }
        DataType::Time32(unit) => Err(ArrowError::InvalidArgumentError(format!(
            "Invalid time unit for Time32: {:?}",
            unit
        ))),
        DataType::Time64(TimeUnit::Microsecond) => {
            let arr = array.as_primitive::<Time64MicrosecondType>();
            Ok(Scalar::Time64(
                Some(arr.value(index)),
                TimeUnit::Microsecond,
            ))
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            let arr = array.as_primitive::<Time64NanosecondType>();
            Ok(Scalar::Time64(Some(arr.value(index)), TimeUnit::Nanosecond))
        }
        DataType::Time64(unit) => Err(ArrowError::InvalidArgumentError(format!(
            "Invalid time unit for Time64: {:?}",
            unit
        ))),
        DataType::Timestamp(TimeUnit::Second, tz) => {
            let arr = array.as_primitive::<TimestampSecondType>();
            Ok(Scalar::Timestamp(
                Some(arr.value(index)),
                TimeUnit::Second,
                tz.clone(),
            ))
        }
        DataType::Timestamp(TimeUnit::Millisecond, tz) => {
            let arr = array.as_primitive::<TimestampMillisecondType>();
            Ok(Scalar::Timestamp(
                Some(arr.value(index)),
                TimeUnit::Millisecond,
                tz.clone(),
            ))
        }
        DataType::Timestamp(TimeUnit::Microsecond, tz) => {
            let arr = array.as_primitive::<TimestampMicrosecondType>();
            Ok(Scalar::Timestamp(
                Some(arr.value(index)),
                TimeUnit::Microsecond,
                tz.clone(),
            ))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
            let arr = array.as_primitive::<TimestampNanosecondType>();
            Ok(Scalar::Timestamp(
                Some(arr.value(index)),
                TimeUnit::Nanosecond,
                tz.clone(),
            ))
        }
        DataType::Duration(TimeUnit::Second) => {
            let arr = array.as_primitive::<DurationSecondType>();
            Ok(Scalar::Duration(Some(arr.value(index)), TimeUnit::Second))
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            let arr = array.as_primitive::<DurationMillisecondType>();
            Ok(Scalar::Duration(
                Some(arr.value(index)),
                TimeUnit::Millisecond,
            ))
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            let arr = array.as_primitive::<DurationMicrosecondType>();
            Ok(Scalar::Duration(
                Some(arr.value(index)),
                TimeUnit::Microsecond,
            ))
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            let arr = array.as_primitive::<DurationNanosecondType>();
            Ok(Scalar::Duration(
                Some(arr.value(index)),
                TimeUnit::Nanosecond,
            ))
        }
        DataType::Interval(IntervalUnit::YearMonth) => {
            let arr = array.as_primitive::<IntervalYearMonthType>();
            Ok(Scalar::IntervalYearMonth(Some(arr.value(index))))
        }
        DataType::Interval(IntervalUnit::DayTime) => {
            let arr = array.as_primitive::<IntervalDayTimeType>();
            let val = arr.value(index);
            // IntervalDayTime is stored as days (lower 32 bits) and ms (upper 32 bits)
            let combined = ((val.milliseconds as i64) << 32) | (val.days as i64 & 0xFFFFFFFF);
            Ok(Scalar::IntervalDayTime(Some(combined)))
        }
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            let arr = array.as_primitive::<IntervalMonthDayNanoType>();
            let val = arr.value(index);
            // IntervalMonthDayNano: months (lower 32), days (next 32), nanos (upper 64)
            let combined = ((val.nanoseconds as i128) << 64)
                | ((val.days as i128 & 0xFFFFFFFF) << 32)
                | (val.months as i128 & 0xFFFFFFFF);
            Ok(Scalar::IntervalMonthDayNano(Some(combined)))
        }
        DataType::List(_) => {
            let arr = extract_scalar_element(array, index)?;
            Ok(Scalar::List(arr))
        }
        DataType::LargeList(_) => {
            let arr = extract_scalar_element(array, index)?;
            Ok(Scalar::LargeList(arr))
        }
        DataType::FixedSizeList(_, _) => {
            let arr = extract_scalar_element(array, index)?;
            Ok(Scalar::FixedSizeList(arr))
        }
        DataType::Struct(fields) => {
            let struct_arr = array.as_struct();
            let values = struct_arr
                .columns()
                .iter()
                .map(|col| try_from_array(col.as_ref(), index))
                .collect::<Result<Vec<_>>>()?;
            Ok(Scalar::Struct(fields.clone(), values))
        }
        DataType::Map(_, _) => {
            let arr = extract_scalar_element(array, index)?;
            Ok(Scalar::Map(arr))
        }
        DataType::Dictionary(key_type, _) => {
            let dict = array.as_any_dictionary();
            let key_idx = dict.keys().as_primitive::<UInt32Type>().value(index) as usize;
            let value_scalar = try_from_array(dict.values().as_ref(), key_idx)?;
            Ok(Scalar::Dictionary(key_type.clone(), Box::new(value_scalar)))
        }
        dt => Err(ArrowError::NotYetImplemented(format!(
            "Scalar conversion not implemented for {:?}",
            dt
        ))),
    }
}

/// Extracts a single element from an array as a length-1 array.
fn extract_scalar_element(array: &dyn Array, index: usize) -> Result<ArrayRef> {
    let data = array.to_data();
    let mut mutable = MutableArrayData::new(vec![&data], true, 1);
    mutable.extend(0, index, index + 1);
    Ok(arrow_array::make_array(mutable.freeze()))
}

impl Scalar {
    /// Converts this scalar to a length-1 Arrow array.
    pub fn to_array(&self) -> ArrayRef {
        match self {
            Self::Null => arrow_array::new_null_array(&DataType::Null, 1),
            Self::Boolean(v) => Arc::new(BooleanArray::from(vec![*v])),
            Self::Int8(v) => Arc::new(PrimitiveArray::<Int8Type>::from(vec![*v])),
            Self::Int16(v) => Arc::new(PrimitiveArray::<Int16Type>::from(vec![*v])),
            Self::Int32(v) => Arc::new(PrimitiveArray::<Int32Type>::from(vec![*v])),
            Self::Int64(v) => Arc::new(PrimitiveArray::<Int64Type>::from(vec![*v])),
            Self::UInt8(v) => Arc::new(PrimitiveArray::<UInt8Type>::from(vec![*v])),
            Self::UInt16(v) => Arc::new(PrimitiveArray::<UInt16Type>::from(vec![*v])),
            Self::UInt32(v) => Arc::new(PrimitiveArray::<UInt32Type>::from(vec![*v])),
            Self::UInt64(v) => Arc::new(PrimitiveArray::<UInt64Type>::from(vec![*v])),
            Self::Float16(v) => Arc::new(PrimitiveArray::<Float16Type>::from(vec![*v])),
            Self::Float32(v) => Arc::new(PrimitiveArray::<Float32Type>::from(vec![*v])),
            Self::Float64(v) => Arc::new(PrimitiveArray::<Float64Type>::from(vec![*v])),
            Self::Decimal128(v, precision, scale) => {
                let arr = PrimitiveArray::<Decimal128Type>::from(vec![*v])
                    .with_precision_and_scale(*precision, *scale)
                    .expect("Invalid decimal precision/scale");
                Arc::new(arr)
            }
            Self::Decimal256(v, precision, scale) => {
                let arr = PrimitiveArray::<Decimal256Type>::from(vec![*v])
                    .with_precision_and_scale(*precision, *scale)
                    .expect("Invalid decimal precision/scale");
                Arc::new(arr)
            }
            Self::Utf8(v) => match v {
                Some(buf) => {
                    let offsets =
                        OffsetBuffer::new(ScalarBuffer::from(vec![0i32, buf.len() as i32]));
                    Arc::new(GenericByteArray::<arrow_array::types::Utf8Type>::new(
                        offsets,
                        buf.clone(),
                        None,
                    ))
                }
                None => arrow_array::new_null_array(&DataType::Utf8, 1),
            },
            Self::LargeUtf8(v) => match v {
                Some(buf) => {
                    let offsets =
                        OffsetBuffer::new(ScalarBuffer::from(vec![0i64, buf.len() as i64]));
                    Arc::new(GenericByteArray::<arrow_array::types::LargeUtf8Type>::new(
                        offsets,
                        buf.clone(),
                        None,
                    ))
                }
                None => arrow_array::new_null_array(&DataType::LargeUtf8, 1),
            },
            Self::Utf8View(v) => {
                let s = v.as_deref().map(|b| {
                    std::str::from_utf8(b).expect("Utf8View scalar must contain valid UTF-8")
                });
                Arc::new(GenericByteViewArray::<StringViewType>::from(vec![s]))
            }
            Self::Binary(v) => match v {
                Some(buf) => {
                    let offsets =
                        OffsetBuffer::new(ScalarBuffer::from(vec![0i32, buf.len() as i32]));
                    Arc::new(GenericByteArray::<arrow_array::types::BinaryType>::new(
                        offsets,
                        buf.clone(),
                        None,
                    ))
                }
                None => arrow_array::new_null_array(&DataType::Binary, 1),
            },
            Self::LargeBinary(v) => match v {
                Some(buf) => {
                    let offsets =
                        OffsetBuffer::new(ScalarBuffer::from(vec![0i64, buf.len() as i64]));
                    Arc::new(
                        GenericByteArray::<arrow_array::types::LargeBinaryType>::new(
                            offsets,
                            buf.clone(),
                            None,
                        ),
                    )
                }
                None => arrow_array::new_null_array(&DataType::LargeBinary, 1),
            },
            Self::BinaryView(v) => Arc::new(GenericByteViewArray::<BinaryViewType>::from(vec![
                v.as_deref()
            ])),
            Self::FixedSizeBinary(size, v) => {
                let arr = match v {
                    Some(buf) => FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                        std::iter::once(Some(buf.as_ref())),
                        *size,
                    )
                    .expect("Invalid fixed size binary"),
                    None => FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                        std::iter::once(None::<&[u8]>),
                        *size,
                    )
                    .expect("Invalid fixed size binary"),
                };
                Arc::new(arr)
            }
            Self::Date32(v) => Arc::new(PrimitiveArray::<Date32Type>::from(vec![*v])),
            Self::Date64(v) => Arc::new(PrimitiveArray::<Date64Type>::from(vec![*v])),
            Self::Time32(v, unit) => match unit {
                TimeUnit::Second => Arc::new(PrimitiveArray::<Time32SecondType>::from(vec![*v])),
                TimeUnit::Millisecond => {
                    Arc::new(PrimitiveArray::<Time32MillisecondType>::from(vec![*v]))
                }
                _ => panic!("Invalid time unit for Time32: {:?}", unit),
            },
            Self::Time64(v, unit) => match unit {
                TimeUnit::Microsecond => {
                    Arc::new(PrimitiveArray::<Time64MicrosecondType>::from(vec![*v]))
                }
                TimeUnit::Nanosecond => {
                    Arc::new(PrimitiveArray::<Time64NanosecondType>::from(vec![*v]))
                }
                _ => panic!("Invalid time unit for Time64: {:?}", unit),
            },
            Self::Timestamp(v, unit, tz) => match unit {
                TimeUnit::Second => Arc::new(
                    PrimitiveArray::<TimestampSecondType>::from(vec![*v])
                        .with_timezone_opt(tz.clone()),
                ),
                TimeUnit::Millisecond => Arc::new(
                    PrimitiveArray::<TimestampMillisecondType>::from(vec![*v])
                        .with_timezone_opt(tz.clone()),
                ),
                TimeUnit::Microsecond => Arc::new(
                    PrimitiveArray::<TimestampMicrosecondType>::from(vec![*v])
                        .with_timezone_opt(tz.clone()),
                ),
                TimeUnit::Nanosecond => Arc::new(
                    PrimitiveArray::<TimestampNanosecondType>::from(vec![*v])
                        .with_timezone_opt(tz.clone()),
                ),
            },
            Self::Duration(v, unit) => match unit {
                TimeUnit::Second => Arc::new(PrimitiveArray::<DurationSecondType>::from(vec![*v])),
                TimeUnit::Millisecond => {
                    Arc::new(PrimitiveArray::<DurationMillisecondType>::from(vec![*v]))
                }
                TimeUnit::Microsecond => {
                    Arc::new(PrimitiveArray::<DurationMicrosecondType>::from(vec![*v]))
                }
                TimeUnit::Nanosecond => {
                    Arc::new(PrimitiveArray::<DurationNanosecondType>::from(vec![*v]))
                }
            },
            Self::IntervalYearMonth(v) => {
                Arc::new(PrimitiveArray::<IntervalYearMonthType>::from(vec![*v]))
            }
            Self::IntervalDayTime(v) => {
                let v = v.map(|combined| {
                    let days = (combined & 0xFFFFFFFF) as i32;
                    let ms = (combined >> 32) as i32;
                    arrow_buffer::IntervalDayTime::new(days, ms)
                });
                Arc::new(PrimitiveArray::<IntervalDayTimeType>::from(vec![v]))
            }
            Self::IntervalMonthDayNano(v) => {
                let v = v.map(|combined| {
                    let months = (combined & 0xFFFFFFFF) as i32;
                    let days = ((combined >> 32) & 0xFFFFFFFF) as i32;
                    let ns = (combined >> 64) as i64;
                    arrow_buffer::IntervalMonthDayNano::new(months, days, ns)
                });
                Arc::new(PrimitiveArray::<IntervalMonthDayNanoType>::from(vec![v]))
            }
            Self::List(arr) => arr.clone(),
            Self::LargeList(arr) => arr.clone(),
            Self::FixedSizeList(arr) => arr.clone(),
            Self::Struct(fields, values) => {
                let arrays: Vec<ArrayRef> = values.iter().map(|v| v.to_array()).collect();
                Arc::new(StructArray::new(fields.clone(), arrays, None))
            }
            Self::Map(arr) => arr.clone(),
            Self::Dictionary(key_type, value) => {
                let value_arr = value.to_array();
                match key_type.as_ref() {
                    DataType::Int8 => {
                        let keys = PrimitiveArray::<Int8Type>::from(vec![Some(0i8)]);
                        Arc::new(
                            arrow_array::DictionaryArray::try_new(keys, value_arr)
                                .expect("Invalid dictionary"),
                        )
                    }
                    DataType::Int16 => {
                        let keys = PrimitiveArray::<Int16Type>::from(vec![Some(0i16)]);
                        Arc::new(
                            arrow_array::DictionaryArray::try_new(keys, value_arr)
                                .expect("Invalid dictionary"),
                        )
                    }
                    DataType::Int32 => {
                        let keys = PrimitiveArray::<Int32Type>::from(vec![Some(0i32)]);
                        Arc::new(
                            arrow_array::DictionaryArray::try_new(keys, value_arr)
                                .expect("Invalid dictionary"),
                        )
                    }
                    DataType::Int64 => {
                        let keys = PrimitiveArray::<Int64Type>::from(vec![Some(0i64)]);
                        Arc::new(
                            arrow_array::DictionaryArray::try_new(keys, value_arr)
                                .expect("Invalid dictionary"),
                        )
                    }
                    DataType::UInt8 => {
                        let keys = PrimitiveArray::<UInt8Type>::from(vec![Some(0u8)]);
                        Arc::new(
                            arrow_array::DictionaryArray::try_new(keys, value_arr)
                                .expect("Invalid dictionary"),
                        )
                    }
                    DataType::UInt16 => {
                        let keys = PrimitiveArray::<UInt16Type>::from(vec![Some(0u16)]);
                        Arc::new(
                            arrow_array::DictionaryArray::try_new(keys, value_arr)
                                .expect("Invalid dictionary"),
                        )
                    }
                    DataType::UInt32 => {
                        let keys = PrimitiveArray::<UInt32Type>::from(vec![Some(0u32)]);
                        Arc::new(
                            arrow_array::DictionaryArray::try_new(keys, value_arr)
                                .expect("Invalid dictionary"),
                        )
                    }
                    DataType::UInt64 => {
                        let keys = PrimitiveArray::<UInt64Type>::from(vec![Some(0u64)]);
                        Arc::new(
                            arrow_array::DictionaryArray::try_new(keys, value_arr)
                                .expect("Invalid dictionary"),
                        )
                    }
                    _ => panic!("Invalid dictionary key type: {:?}", key_type),
                }
            }
        }
    }
}

/// Converts an iterator of scalars to an Arrow array.
///
/// All scalars must have the same data type.
pub fn iter_to_array(iter: impl Iterator<Item = Scalar>) -> Result<ArrayRef> {
    let scalars: Vec<Scalar> = iter.collect();
    if scalars.is_empty() {
        return Err(ArrowError::InvalidArgumentError(
            "Cannot create array from empty iterator".to_string(),
        ));
    }

    let arrays: Vec<ArrayRef> = scalars.iter().map(|s| s.to_array()).collect();
    let refs: Vec<&dyn Array> = arrays.iter().map(|a| a.as_ref()).collect();
    arrow_select::concat::concat(&refs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_buffer::Buffer;
    use rstest::rstest;

    #[rstest]
    #[case::int32(Arc::new(Int32Array::from(vec![Some(42), None, Some(7)])) as ArrayRef, 0, Scalar::Int32(Some(42)))]
    #[case::int32_null(Arc::new(Int32Array::from(vec![Some(42), None, Some(7)])) as ArrayRef, 1, Scalar::Int32(None))]
    #[case::string(Arc::new(StringArray::from(vec![Some("hello"), None])) as ArrayRef, 0, Scalar::Utf8(Some(Buffer::from("hello".as_bytes()))))]
    fn test_try_from_array(
        #[case] array: ArrayRef,
        #[case] index: usize,
        #[case] expected: Scalar,
    ) {
        let result = try_from_array(array.as_ref(), index).unwrap();
        assert_eq!(result.data_type(), expected.data_type());
        assert_eq!(result.is_null(), expected.is_null());
    }

    #[test]
    fn test_round_trip_primitives() {
        let original: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));
        for i in 0..original.len() {
            let scalar = try_from_array(original.as_ref(), i).unwrap();
            let arr = scalar.to_array();
            assert_eq!(arr.len(), 1);
            let back = try_from_array(arr.as_ref(), 0).unwrap();
            assert_eq!(back.data_type(), scalar.data_type());
        }
    }

    #[test]
    fn test_iter_to_array() {
        let scalars = vec![
            Scalar::Int32(Some(1)),
            Scalar::Int32(Some(2)),
            Scalar::Int32(None),
            Scalar::Int32(Some(4)),
        ];
        let arr = iter_to_array(scalars.into_iter()).unwrap();
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.null_count(), 1);
    }
}
