// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for working with datafusion expressions

use std::sync::Arc;

use arrow::compute::cast;
use arrow_array::{cast::AsArray, ArrayRef};
use arrow_schema::{DataType, TimeUnit};
use datafusion_common::ScalarValue;

const MS_PER_DAY: i64 = 86400000;

// This is slightly tedious but when we convert expressions from SQL strings to logical
// datafusion expressions there is no type coercion that happens.  In other words "x = 7"
// will always yield "x = 7_u64" regardless of the type of the column "x".  As a result, we
// need to do that literal coercion ourselves.
pub fn safe_coerce_scalar(value: &ScalarValue, ty: &DataType) -> Option<ScalarValue> {
    match value {
        ScalarValue::Int8(val) => match ty {
            DataType::Int8 => Some(value.clone()),
            DataType::Int16 => val.map(|v| ScalarValue::Int16(Some(i16::from(v)))),
            DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(i32::from(v)))),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(i64::from(v)))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => {
                val.and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok())
            }
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::Int16(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => Some(value.clone()),
            DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(i32::from(v)))),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(i64::from(v)))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => {
                val.and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok())
            }
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::Int32(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => Some(value.clone()),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(i64::from(v)))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => {
                val.and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok())
            }
            // These conversions are inherently lossy as the full range of i32 cannot
            // be represented in f32.  However, there is no f32::TryFrom(i32) and its not
            // clear users would want that anyways
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
            _ => None,
        },
        ScalarValue::Int64(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => {
                val.and_then(|v| i32::try_from(v).map(|v| ScalarValue::Int32(Some(v))).ok())
            }
            DataType::Int64 => Some(value.clone()),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => {
                val.and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok())
            }
            // See above warning about lossy float conversion
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
            _ => None,
        },
        ScalarValue::UInt8(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => val.map(|v| ScalarValue::Int16(Some(v.into()))),
            DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(v.into()))),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(v.into()))),
            DataType::UInt8 => Some(value.clone()),
            DataType::UInt16 => val.map(|v| ScalarValue::UInt16(Some(u16::from(v)))),
            DataType::UInt32 => val.map(|v| ScalarValue::UInt32(Some(u32::from(v)))),
            DataType::UInt64 => val.map(|v| ScalarValue::UInt64(Some(u64::from(v)))),
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::UInt16(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(v.into()))),
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(v.into()))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => Some(value.clone()),
            DataType::UInt32 => val.map(|v| ScalarValue::UInt32(Some(u32::from(v)))),
            DataType::UInt64 => val.map(|v| ScalarValue::UInt64(Some(u64::from(v)))),
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::UInt32(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => {
                val.and_then(|v| i32::try_from(v).map(|v| ScalarValue::Int32(Some(v))).ok())
            }
            DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(v.into()))),
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => Some(value.clone()),
            DataType::UInt64 => val.map(|v| ScalarValue::UInt64(Some(u64::from(v)))),
            // See above warning about lossy float conversion
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
            _ => None,
        },
        ScalarValue::UInt64(val) => match ty {
            DataType::Int8 => {
                val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
            }
            DataType::Int16 => {
                val.and_then(|v| i16::try_from(v).map(|v| ScalarValue::Int16(Some(v))).ok())
            }
            DataType::Int32 => {
                val.and_then(|v| i32::try_from(v).map(|v| ScalarValue::Int32(Some(v))).ok())
            }
            DataType::Int64 => {
                val.and_then(|v| i64::try_from(v).map(|v| ScalarValue::Int64(Some(v))).ok())
            }
            DataType::UInt8 => {
                val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
            }
            DataType::UInt16 => {
                val.and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok())
            }
            DataType::UInt32 => {
                val.and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok())
            }
            DataType::UInt64 => Some(value.clone()),
            // See above warning about lossy float conversion
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
            _ => None,
        },
        ScalarValue::Float32(val) => match ty {
            DataType::Float32 => Some(value.clone()),
            DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
            _ => None,
        },
        ScalarValue::Float64(val) => match ty {
            DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
            DataType::Float64 => Some(value.clone()),
            _ => None,
        },
        ScalarValue::Utf8(val) => match ty {
            DataType::Utf8 => Some(value.clone()),
            DataType::LargeUtf8 => Some(ScalarValue::LargeUtf8(val.clone())),
            _ => None,
        },
        ScalarValue::LargeUtf8(val) => match ty {
            DataType::Utf8 => Some(ScalarValue::Utf8(val.clone())),
            DataType::LargeUtf8 => Some(value.clone()),
            _ => None,
        },
        ScalarValue::Boolean(_) => match ty {
            DataType::Boolean => Some(value.clone()),
            _ => None,
        },
        ScalarValue::Null => Some(value.clone()),
        ScalarValue::List(values) => {
            let values = values.clone() as ArrayRef;
            let new_values = cast(&values, ty).ok()?;
            match ty {
                DataType::List(_) => {
                    Some(ScalarValue::List(Arc::new(new_values.as_list().clone())))
                }
                DataType::LargeList(_) => Some(ScalarValue::LargeList(Arc::new(
                    new_values.as_list().clone(),
                ))),
                DataType::FixedSizeList(_, _) => Some(ScalarValue::FixedSizeList(Arc::new(
                    new_values.as_fixed_size_list().clone(),
                ))),
                _ => None,
            }
        }
        ScalarValue::TimestampSecond(seconds, _) => match ty {
            DataType::Timestamp(TimeUnit::Second, _) => Some(value.clone()),
            DataType::Timestamp(TimeUnit::Millisecond, tz) => seconds
                .and_then(|v| v.checked_mul(1000))
                .map(|val| ScalarValue::TimestampMillisecond(Some(val), tz.clone())),
            DataType::Timestamp(TimeUnit::Microsecond, tz) => seconds
                .and_then(|v| v.checked_mul(1000000))
                .map(|val| ScalarValue::TimestampMicrosecond(Some(val), tz.clone())),
            DataType::Timestamp(TimeUnit::Nanosecond, tz) => seconds
                .and_then(|v| v.checked_mul(1000000000))
                .map(|val| ScalarValue::TimestampNanosecond(Some(val), tz.clone())),
            _ => None,
        },
        ScalarValue::TimestampMillisecond(millis, _) => match ty {
            DataType::Timestamp(TimeUnit::Second, tz) => {
                millis.map(|val| ScalarValue::TimestampSecond(Some(val / 1000), tz.clone()))
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => Some(value.clone()),
            DataType::Timestamp(TimeUnit::Microsecond, tz) => millis
                .and_then(|v| v.checked_mul(1000))
                .map(|val| ScalarValue::TimestampMicrosecond(Some(val), tz.clone())),
            DataType::Timestamp(TimeUnit::Nanosecond, tz) => millis
                .and_then(|v| v.checked_mul(1000000))
                .map(|val| ScalarValue::TimestampNanosecond(Some(val), tz.clone())),
            _ => None,
        },
        ScalarValue::TimestampMicrosecond(micros, _) => match ty {
            DataType::Timestamp(TimeUnit::Second, tz) => {
                micros.map(|val| ScalarValue::TimestampSecond(Some(val / 1000000), tz.clone()))
            }
            DataType::Timestamp(TimeUnit::Millisecond, tz) => {
                micros.map(|val| ScalarValue::TimestampMillisecond(Some(val / 1000), tz.clone()))
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => Some(value.clone()),
            DataType::Timestamp(TimeUnit::Nanosecond, tz) => micros
                .and_then(|v| v.checked_mul(1000))
                .map(|val| ScalarValue::TimestampNanosecond(Some(val), tz.clone())),
            _ => None,
        },
        ScalarValue::TimestampNanosecond(nanos, _) => {
            match ty {
                DataType::Timestamp(TimeUnit::Second, tz) => nanos
                    .map(|val| ScalarValue::TimestampSecond(Some(val / 1000000000), tz.clone())),
                DataType::Timestamp(TimeUnit::Millisecond, tz) => nanos
                    .map(|val| ScalarValue::TimestampMillisecond(Some(val / 1000000), tz.clone())),
                DataType::Timestamp(TimeUnit::Microsecond, tz) => {
                    nanos.map(|val| ScalarValue::TimestampMicrosecond(Some(val / 1000), tz.clone()))
                }
                DataType::Timestamp(TimeUnit::Nanosecond, _) => Some(value.clone()),
                _ => None,
            }
        }
        ScalarValue::Date32(ticks) => match ty {
            DataType::Date32 => Some(value.clone()),
            DataType::Date64 => Some(ScalarValue::Date64(
                ticks.map(|v| i64::from(v) * MS_PER_DAY),
            )),
            _ => None,
        },
        ScalarValue::Date64(ticks) => match ty {
            DataType::Date32 => Some(ScalarValue::Date32(ticks.map(|v| (v / MS_PER_DAY) as i32))),
            DataType::Date64 => Some(value.clone()),
            _ => None,
        },
        ScalarValue::Time32Second(seconds) => {
            match ty {
                DataType::Time32(TimeUnit::Second) => Some(value.clone()),
                DataType::Time32(TimeUnit::Millisecond) => {
                    seconds.map(|val| ScalarValue::Time32Millisecond(Some(val * 1000)))
                }
                DataType::Time64(TimeUnit::Microsecond) => seconds
                    .map(|val| ScalarValue::Time64Microsecond(Some(i64::from(val) * 1000000))),
                DataType::Time64(TimeUnit::Nanosecond) => seconds
                    .map(|val| ScalarValue::Time64Nanosecond(Some(i64::from(val) * 1000000000))),
                _ => None,
            }
        }
        ScalarValue::Time32Millisecond(millis) => match ty {
            DataType::Time32(TimeUnit::Second) => {
                millis.map(|val| ScalarValue::Time32Second(Some(val / 1000)))
            }
            DataType::Time32(TimeUnit::Millisecond) => Some(value.clone()),
            DataType::Time64(TimeUnit::Microsecond) => {
                millis.map(|val| ScalarValue::Time64Microsecond(Some(i64::from(val) * 1000)))
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                millis.map(|val| ScalarValue::Time64Nanosecond(Some(i64::from(val) * 1000000)))
            }
            _ => None,
        },
        ScalarValue::Time64Microsecond(micros) => match ty {
            DataType::Time32(TimeUnit::Second) => {
                micros.map(|val| ScalarValue::Time32Second(Some((val / 1000000) as i32)))
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                micros.map(|val| ScalarValue::Time32Millisecond(Some((val / 1000) as i32)))
            }
            DataType::Time64(TimeUnit::Microsecond) => Some(value.clone()),
            DataType::Time64(TimeUnit::Nanosecond) => {
                micros.map(|val| ScalarValue::Time64Nanosecond(Some(val * 1000)))
            }
            _ => None,
        },
        ScalarValue::Time64Nanosecond(nanos) => match ty {
            DataType::Time32(TimeUnit::Second) => {
                nanos.map(|val| ScalarValue::Time32Second(Some((val / 1000000000) as i32)))
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                nanos.map(|val| ScalarValue::Time32Millisecond(Some((val / 1000000) as i32)))
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                nanos.map(|val| ScalarValue::Time64Microsecond(Some(val / 1000)))
            }
            DataType::Time64(TimeUnit::Nanosecond) => Some(value.clone()),
            _ => None,
        },
        ScalarValue::LargeList(values) => {
            let values = values.clone() as ArrayRef;
            let new_values = cast(&values, ty).ok()?;
            match ty {
                DataType::List(_) => {
                    Some(ScalarValue::List(Arc::new(new_values.as_list().clone())))
                }
                DataType::LargeList(_) => Some(ScalarValue::LargeList(Arc::new(
                    new_values.as_list().clone(),
                ))),
                DataType::FixedSizeList(_, _) => Some(ScalarValue::FixedSizeList(Arc::new(
                    new_values.as_fixed_size_list().clone(),
                ))),
                _ => None,
            }
        }
        ScalarValue::FixedSizeList(values) => {
            let values = values.clone() as ArrayRef;
            let new_values = cast(&values, ty).ok()?;
            match ty {
                DataType::List(_) => {
                    Some(ScalarValue::List(Arc::new(new_values.as_list().clone())))
                }
                DataType::LargeList(_) => Some(ScalarValue::LargeList(Arc::new(
                    new_values.as_list().clone(),
                ))),
                DataType::FixedSizeList(_, _) => Some(ScalarValue::FixedSizeList(Arc::new(
                    new_values.as_fixed_size_list().clone(),
                ))),
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_coerce() {
        // Conversion from timestamps in one resolution to timestamps in another resolution is allowed
        // s->s
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampSecond(Some(5), None),
                &DataType::Timestamp(TimeUnit::Second, None),
            ),
            Some(ScalarValue::TimestampSecond(Some(5), None))
        );
        // s->ms
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampSecond(Some(5), None),
                &DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            Some(ScalarValue::TimestampMillisecond(Some(5000), None))
        );
        // s->us
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampSecond(Some(5), None),
                &DataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            Some(ScalarValue::TimestampMicrosecond(Some(5000000), None))
        );
        // s->ns
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampSecond(Some(5), None),
                &DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            Some(ScalarValue::TimestampNanosecond(Some(5000000000), None))
        );
        // ms->s
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMillisecond(Some(5000), None),
                &DataType::Timestamp(TimeUnit::Second, None),
            ),
            Some(ScalarValue::TimestampSecond(Some(5), None))
        );
        // ms->ms
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMillisecond(Some(5000), None),
                &DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            Some(ScalarValue::TimestampMillisecond(Some(5000), None))
        );
        // ms->us
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMillisecond(Some(5000), None),
                &DataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            Some(ScalarValue::TimestampMicrosecond(Some(5000000), None))
        );
        // ms->ns
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMillisecond(Some(5000), None),
                &DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            Some(ScalarValue::TimestampNanosecond(Some(5000000000), None))
        );
        // us->s
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMicrosecond(Some(5000000), None),
                &DataType::Timestamp(TimeUnit::Second, None),
            ),
            Some(ScalarValue::TimestampSecond(Some(5), None))
        );
        // us->ms
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMicrosecond(Some(5000000), None),
                &DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            Some(ScalarValue::TimestampMillisecond(Some(5000), None))
        );
        // us->us
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMicrosecond(Some(5000000), None),
                &DataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            Some(ScalarValue::TimestampMicrosecond(Some(5000000), None))
        );
        // us->ns
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampMicrosecond(Some(5000000), None),
                &DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            Some(ScalarValue::TimestampNanosecond(Some(5000000000), None))
        );
        // ns->s
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampNanosecond(Some(5000000000), None),
                &DataType::Timestamp(TimeUnit::Second, None),
            ),
            Some(ScalarValue::TimestampSecond(Some(5), None))
        );
        // ns->ms
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampNanosecond(Some(5000000000), None),
                &DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            Some(ScalarValue::TimestampMillisecond(Some(5000), None))
        );
        // ns->us
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampNanosecond(Some(5000000000), None),
                &DataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            Some(ScalarValue::TimestampMicrosecond(Some(5000000), None))
        );
        // ns->ns
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampNanosecond(Some(5000000000), None),
                &DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            Some(ScalarValue::TimestampNanosecond(Some(5000000000), None))
        );
        // Precision loss on coercion is allowed (truncation)
        // ns->s
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::TimestampNanosecond(Some(5987654321), None),
                &DataType::Timestamp(TimeUnit::Second, None),
            ),
            Some(ScalarValue::TimestampSecond(Some(5), None))
        );
        // Conversions from date-32 to date-64 is allowed
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::Date32(Some(5)), &DataType::Date32,),
            Some(ScalarValue::Date32(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::Date32(Some(5)), &DataType::Date64,),
            Some(ScalarValue::Date64(Some(5 * MS_PER_DAY)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Date64(Some(5 * MS_PER_DAY)),
                &DataType::Date32,
            ),
            Some(ScalarValue::Date32(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(&ScalarValue::Date64(Some(5)), &DataType::Date64,),
            Some(ScalarValue::Date64(Some(5)))
        );
        // Time-32 to time-64 (and within time-32 and time-64) is allowed
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Second(Some(5)),
                &DataType::Time32(TimeUnit::Second),
            ),
            Some(ScalarValue::Time32Second(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Second(Some(5)),
                &DataType::Time32(TimeUnit::Millisecond),
            ),
            Some(ScalarValue::Time32Millisecond(Some(5000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Second(Some(5)),
                &DataType::Time64(TimeUnit::Microsecond),
            ),
            Some(ScalarValue::Time64Microsecond(Some(5000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Second(Some(5)),
                &DataType::Time64(TimeUnit::Nanosecond),
            ),
            Some(ScalarValue::Time64Nanosecond(Some(5000000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Millisecond(Some(5000)),
                &DataType::Time32(TimeUnit::Second),
            ),
            Some(ScalarValue::Time32Second(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Millisecond(Some(5000)),
                &DataType::Time32(TimeUnit::Millisecond),
            ),
            Some(ScalarValue::Time32Millisecond(Some(5000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Millisecond(Some(5000)),
                &DataType::Time64(TimeUnit::Microsecond),
            ),
            Some(ScalarValue::Time64Microsecond(Some(5000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time32Millisecond(Some(5000)),
                &DataType::Time64(TimeUnit::Nanosecond),
            ),
            Some(ScalarValue::Time64Nanosecond(Some(5000000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Microsecond(Some(5000000)),
                &DataType::Time32(TimeUnit::Second),
            ),
            Some(ScalarValue::Time32Second(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Microsecond(Some(5000000)),
                &DataType::Time32(TimeUnit::Millisecond),
            ),
            Some(ScalarValue::Time32Millisecond(Some(5000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Microsecond(Some(5000000)),
                &DataType::Time64(TimeUnit::Microsecond),
            ),
            Some(ScalarValue::Time64Microsecond(Some(5000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Microsecond(Some(5000000)),
                &DataType::Time64(TimeUnit::Nanosecond),
            ),
            Some(ScalarValue::Time64Nanosecond(Some(5000000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Nanosecond(Some(5000000000)),
                &DataType::Time32(TimeUnit::Second),
            ),
            Some(ScalarValue::Time32Second(Some(5)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Nanosecond(Some(5000000000)),
                &DataType::Time32(TimeUnit::Millisecond),
            ),
            Some(ScalarValue::Time32Millisecond(Some(5000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Nanosecond(Some(5000000000)),
                &DataType::Time64(TimeUnit::Microsecond),
            ),
            Some(ScalarValue::Time64Microsecond(Some(5000000)))
        );
        assert_eq!(
            safe_coerce_scalar(
                &ScalarValue::Time64Nanosecond(Some(5000000000)),
                &DataType::Time64(TimeUnit::Nanosecond),
            ),
            Some(ScalarValue::Time64Nanosecond(Some(5000000000)))
        );
    }
}
