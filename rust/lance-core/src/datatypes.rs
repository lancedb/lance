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

//! Lance data types, [Schema] and [Field]

use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_schema::{DataType, Field as ArrowField, TimeUnit};
use snafu::{location, Location};

mod field;
mod schema;

use crate::format::pb;
use crate::{Error, Result};
pub use field::Field;
pub use schema::Schema;

/// LogicalType is a string presentation of arrow type.
/// to be serialized into protobuf.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalType(String);

impl fmt::Display for LogicalType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl LogicalType {
    fn is_list(&self) -> bool {
        self.0 == "list" || self.0 == "list.struct"
    }

    fn is_large_list(&self) -> bool {
        self.0 == "large_list" || self.0 == "large_list.struct"
    }

    fn is_struct(&self) -> bool {
        self.0 == "struct"
    }

    /// Check whether the logical type is a FixedSizeList.
    fn is_fsl(&self) -> bool {
        self.0.starts_with("fixed_size_list:")
    }

    /// Check whether the logical type is a FixedSizeList, and if so, return the size.
    fn fsl_size(&self) -> Option<i32> {
        // Might be of format "fixed_size_list:{size}" or (legacy)
        // "fixed_size_list:{child_type}:{size}".
        if self.0.starts_with("fixed_size_list:") {
            let (_, size) = self.0.rsplit_once(':').unwrap();
            size.parse::<i32>().ok()
        } else {
            None
        }
    }
}

impl From<&str> for LogicalType {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

fn timeunit_to_str(unit: &TimeUnit) -> &'static str {
    match unit {
        TimeUnit::Second => "s",
        TimeUnit::Millisecond => "ms",
        TimeUnit::Microsecond => "us",
        TimeUnit::Nanosecond => "ns",
    }
}

fn parse_timeunit(unit: &str) -> Result<TimeUnit> {
    match unit {
        "s" => Ok(TimeUnit::Second),
        "ms" => Ok(TimeUnit::Millisecond),
        "us" => Ok(TimeUnit::Microsecond),
        "ns" => Ok(TimeUnit::Nanosecond),
        _ => Err(Error::Arrow {
            message: format!("Unsupported TimeUnit: {unit}"),
            location: location!(),
        }),
    }
}

impl TryFrom<&DataType> for LogicalType {
    type Error = Error;

    fn try_from(dt: &DataType) -> Result<Self> {
        let type_str = match dt {
            DataType::Null => "null".to_string(),
            DataType::Boolean => "bool".to_string(),
            DataType::Int8 => "int8".to_string(),
            DataType::UInt8 => "uint8".to_string(),
            DataType::Int16 => "int16".to_string(),
            DataType::UInt16 => "uint16".to_string(),
            DataType::Int32 => "int32".to_string(),
            DataType::UInt32 => "uint32".to_string(),
            DataType::Int64 => "int64".to_string(),
            DataType::UInt64 => "uint64".to_string(),
            DataType::Float16 => "halffloat".to_string(),
            DataType::Float32 => "float".to_string(),
            DataType::Float64 => "double".to_string(),
            DataType::Decimal128(precision, scale) => format!("decimal:128:{precision}:{scale}"),
            DataType::Decimal256(precision, scale) => format!("decimal:256:{precision}:{scale}"),
            DataType::Utf8 => "string".to_string(),
            DataType::Binary => "binary".to_string(),
            DataType::LargeUtf8 => "large_string".to_string(),
            DataType::LargeBinary => "large_binary".to_string(),
            DataType::Date32 => "date32:day".to_string(),
            DataType::Date64 => "date64:ms".to_string(),
            DataType::Time32(tu) => format!("time32:{}", timeunit_to_str(tu)),
            DataType::Time64(tu) => format!("time64:{}", timeunit_to_str(tu)),
            DataType::Timestamp(tu, tz) => format!(
                "timestamp:{}:{}",
                timeunit_to_str(tu),
                tz.as_ref()
                    .map(|v| v.to_string())
                    .unwrap_or("-".to_string())
            ),
            DataType::Duration(tu) => format!("duration:{}", timeunit_to_str(tu)),
            DataType::Struct(_) => "struct".to_string(),
            DataType::Dictionary(key_type, value_type) => {
                format!(
                    "dict:{}:{}:{}",
                    Self::try_from(value_type.as_ref())?.0,
                    Self::try_from(key_type.as_ref())?.0,
                    // Arrow C++ Dictionary has "ordered:bool" field, but it does not exist in `arrow-rs`.
                    false
                )
            }
            DataType::List(elem) => match elem.data_type() {
                _ => "list".to_string(),
            },
            DataType::LargeList(elem) => match elem.data_type() {
                _ => "large_list".to_string(),
            },
            DataType::FixedSizeList(_dt, len) => format!("fixed_size_list:{}", *len),
            DataType::FixedSizeBinary(len) => format!("fixed_size_binary:{}", *len),
            _ => {
                return Err(Error::Schema {
                    message: format!("Unsupported data type: {:?}", dt),
                    location: location!(),
                })
            }
        };

        Ok(Self(type_str))
    }
}

impl TryFrom<&LogicalType> for DataType {
    type Error = Error;

    fn try_from(lt: &LogicalType) -> Result<Self> {
        use DataType::*;
        if let Some(t) = match lt.0.as_str() {
            "null" => Some(Null),
            "bool" => Some(Boolean),
            "int8" => Some(Int8),
            "uint8" => Some(UInt8),
            "int16" => Some(Int16),
            "uint16" => Some(UInt16),
            "int32" => Some(Int32),
            "uint32" => Some(UInt32),
            "int64" => Some(Int64),
            "uint64" => Some(UInt64),
            "halffloat" => Some(Float16),
            "float" => Some(Float32),
            "double" => Some(Float64),
            "string" => Some(Utf8),
            "binary" => Some(Binary),
            "large_string" => Some(LargeUtf8),
            "large_binary" => Some(LargeBinary),
            "date32:day" => Some(Date32),
            "date64:ms" => Some(Date64),
            "time32:s" => Some(Time32(TimeUnit::Second)),
            "time32:ms" => Some(Time32(TimeUnit::Millisecond)),
            "time64:us" => Some(Time64(TimeUnit::Microsecond)),
            "time64:ns" => Some(Time64(TimeUnit::Nanosecond)),
            "duration:s" => Some(Duration(TimeUnit::Second)),
            "duration:ms" => Some(Duration(TimeUnit::Millisecond)),
            "duration:us" => Some(Duration(TimeUnit::Microsecond)),
            "duration:ns" => Some(Duration(TimeUnit::Nanosecond)),
            _ => None,
        } {
            Ok(t)
        } else {
            let splits = lt.0.split(':').collect::<Vec<_>>();
            match splits[0] {
                "fixed_size_binary" => {
                    if splits.len() != 2 {
                        Err(Error::Schema {
                            message: format!("Unsupported logical type: {}", lt),
                            location: location!(),
                        })
                    } else {
                        let size: i32 = splits[1].parse::<i32>().map_err(|e: _| Error::Schema {
                            message: e.to_string(),
                            location: location!(),
                        })?;
                        Ok(FixedSizeBinary(size))
                    }
                }
                "dict" => {
                    if splits.len() != 4 {
                        Err(Error::Schema {
                            message: format!("Unsupport dictionary type: {}", lt),
                            location: location!(),
                        })
                    } else {
                        let value_type: Self = (&LogicalType::from(splits[1])).try_into()?;
                        let index_type: Self = (&LogicalType::from(splits[2])).try_into()?;
                        Ok(Dictionary(Box::new(index_type), Box::new(value_type)))
                    }
                }
                "decimal" => {
                    if splits.len() != 4 {
                        Err(Error::Schema {
                            message: format!("Unsupport decimal type: {}", lt),
                            location: location!(),
                        })
                    } else {
                        let bits: i16 = splits[1].parse::<i16>().map_err(|err| Error::Schema {
                            message: err.to_string(),
                            location: location!(),
                        })?;
                        let precision: u8 =
                            splits[2].parse::<u8>().map_err(|err| Error::Schema {
                                message: err.to_string(),
                                location: location!(),
                            })?;
                        let scale: i8 = splits[3].parse::<i8>().map_err(|err| Error::Schema {
                            message: err.to_string(),
                            location: location!(),
                        })?;

                        if bits == 128 {
                            Ok(Decimal128(precision, scale))
                        } else if bits == 256 {
                            Ok(Decimal256(precision, scale))
                        } else {
                            Err(Error::Schema {
                                message: format!(
                                    "Only Decimal128 and Decimal256 is supported. Found {bits}"
                                ),
                                location: location!(),
                            })
                        }
                    }
                }
                "timestamp" => {
                    if splits.len() != 3 {
                        Err(Error::Schema {
                            message: format!("Unsupported timestamp type: {}", lt),
                            location: location!(),
                        })
                    } else {
                        let timeunit = parse_timeunit(splits[1])?;
                        let tz: Option<Arc<str>> = if splits[2] == "-" {
                            None
                        } else {
                            Some(splits[2].into())
                        };
                        Ok(Timestamp(timeunit, tz))
                    }
                }
                _ => Err(Error::Schema {
                    message: format!("Unsupported logical type: {}", lt),
                    location: location!(),
                }),
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Dictionary {
    pub offset: usize,

    pub length: usize,

    pub values: Option<ArrayRef>,
}

impl PartialEq for Dictionary {
    fn eq(&self, other: &Self) -> bool {
        match (&self.values, &other.values) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }
}

impl From<&pb::Dictionary> for Dictionary {
    fn from(proto: &pb::Dictionary) -> Self {
        Self {
            offset: proto.offset as usize,
            length: proto.length as usize,
            values: None,
        }
    }
}

impl From<&Dictionary> for pb::Dictionary {
    fn from(d: &Dictionary) -> Self {
        Self {
            offset: d.offset as i64,
            length: d.length as i64,
        }
    }
}

/// Returns true if Lance supports writing this datatype with nulls.
pub fn lance_supports_nulls(datatype: &DataType) -> bool {
    matches!(
        datatype,
        DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::List(_)
            | DataType::FixedSizeBinary(_)
            | DataType::FixedSizeList(_, _)
    )
}
