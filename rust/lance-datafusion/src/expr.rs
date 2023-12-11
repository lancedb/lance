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

//! Utilities for working with datafusion expressions

use arrow_array::ListArray;
use arrow_schema::DataType;
use datafusion_common::{Result, ScalarValue};

// This is slightly tedious but when we convert expressions from SQL strings to logical
// datafusion expressions there is no type coercion that happens.  In other words "x = 7"
// will always yield "x = 7_u64" regardless of the type of the column "x".  As a result, we
// need to do that literal coercion ourselves.
pub fn safe_coerce_scalar(value: &ScalarValue, ty: &DataType) -> Result<Option<ScalarValue>> {
    let value =
        match value {
            ScalarValue::Int8(val) => {
                match ty {
                    DataType::Int8 => Some(value.clone()),
                    DataType::Int16 => val.map(|v| ScalarValue::Int16(Some(i16::from(v)))),
                    DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(i32::from(v)))),
                    DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(i64::from(v)))),
                    DataType::UInt8 => {
                        val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
                    }
                    DataType::UInt16 => val
                        .and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok()),
                    DataType::UInt32 => val
                        .and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok()),
                    DataType::UInt64 => val
                        .and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok()),
                    DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
                    DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
                    _ => None,
                }
            }
            ScalarValue::Int16(val) => {
                match ty {
                    DataType::Int8 => {
                        val.and_then(|v| i8::try_from(v).map(|v| ScalarValue::Int8(Some(v))).ok())
                    }
                    DataType::Int16 => Some(value.clone()),
                    DataType::Int32 => val.map(|v| ScalarValue::Int32(Some(i32::from(v)))),
                    DataType::Int64 => val.map(|v| ScalarValue::Int64(Some(i64::from(v)))),
                    DataType::UInt8 => {
                        val.and_then(|v| u8::try_from(v).map(|v| ScalarValue::UInt8(Some(v))).ok())
                    }
                    DataType::UInt16 => val
                        .and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok()),
                    DataType::UInt32 => val
                        .and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok()),
                    DataType::UInt64 => val
                        .and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok()),
                    DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(f32::from(v)))),
                    DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(f64::from(v)))),
                    _ => None,
                }
            }
            ScalarValue::Int32(val) => {
                match ty {
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
                    DataType::UInt16 => val
                        .and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok()),
                    DataType::UInt32 => val
                        .and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok()),
                    DataType::UInt64 => val
                        .and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok()),
                    // These conversions are inherently lossy as the full range of i32 cannot
                    // be represented in f32.  However, there is no f32::TryFrom(i32) and its not
                    // clear users would want that anyways
                    DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
                    DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
                    _ => None,
                }
            }
            ScalarValue::Int64(val) => {
                match ty {
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
                    DataType::UInt16 => val
                        .and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok()),
                    DataType::UInt32 => val
                        .and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok()),
                    DataType::UInt64 => val
                        .and_then(|v| u64::try_from(v).map(|v| ScalarValue::UInt64(Some(v))).ok()),
                    // See above warning about lossy float conversion
                    DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
                    DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
                    _ => None,
                }
            }
            ScalarValue::UInt8(val) => match ty {
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
                DataType::Int32 => {
                    val.and_then(|v| i32::try_from(v).map(|v| ScalarValue::Int32(Some(v))).ok())
                }
                DataType::Int64 => {
                    val.and_then(|v| i64::try_from(v).map(|v| ScalarValue::Int64(Some(v))).ok())
                }
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
                DataType::Int64 => {
                    val.and_then(|v| i64::try_from(v).map(|v| ScalarValue::Int64(Some(v))).ok())
                }
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
            ScalarValue::UInt64(val) => {
                match ty {
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
                    DataType::UInt16 => val
                        .and_then(|v| u16::try_from(v).map(|v| ScalarValue::UInt16(Some(v))).ok()),
                    DataType::UInt32 => val
                        .and_then(|v| u32::try_from(v).map(|v| ScalarValue::UInt32(Some(v))).ok()),
                    DataType::UInt64 => Some(value.clone()),
                    // See above warning about lossy float conversion
                    DataType::Float32 => val.map(|v| ScalarValue::Float32(Some(v as f32))),
                    DataType::Float64 => val.map(|v| ScalarValue::Float64(Some(v as f64))),
                    _ => None,
                }
            }
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
            ScalarValue::Boolean(_) => match ty {
                DataType::Boolean => Some(value.clone()),
                _ => None,
            },
            ScalarValue::Null => Some(value.clone()),
            ScalarValue::List(vals) => match ty {
                DataType::List(_) => Some(ScalarValue::List(vals.clone())),
                DataType::FixedSizeList(field, size) => {
                    let inner = vals.as_any().downcast_ref::<ListArray>().unwrap();
                    let vals = inner.values();
                    if vals.len() == *size as usize {
                        let mut values = Vec::new();
                        for row in 0..vals.len() {
                            let scalar = ScalarValue::try_from_array(vals.as_ref(), row)?;
                            let scalar = safe_coerce_scalar(&scalar, field.data_type())?;
                            values.push(scalar);
                        }
                        let values = values.into_iter().collect::<Option<Vec<_>>>();
                        Some(ScalarValue::Fixedsizelist(values, field.clone(), *size))
                    } else {
                        None
                    }
                }
                _ => None,
            },
            _ => None,
        };
    Ok(value)
}
