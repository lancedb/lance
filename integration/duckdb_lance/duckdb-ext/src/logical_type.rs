// Copyright 2023 Lance Developers
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

use std::ffi::{c_char, CString};
use std::fmt::Debug;

use crate::ffi::*;

#[repr(u32)]
#[derive(Debug, PartialEq, Eq)]
pub enum LogicalTypeId {
    Boolean = DUCKDB_TYPE_DUCKDB_TYPE_BOOLEAN,
    Tinyint = DUCKDB_TYPE_DUCKDB_TYPE_TINYINT,
    Smallint = DUCKDB_TYPE_DUCKDB_TYPE_SMALLINT,
    Integer = DUCKDB_TYPE_DUCKDB_TYPE_INTEGER,
    Bigint = DUCKDB_TYPE_DUCKDB_TYPE_BIGINT,
    UTinyint = DUCKDB_TYPE_DUCKDB_TYPE_UTINYINT,
    USmallint = DUCKDB_TYPE_DUCKDB_TYPE_USMALLINT,
    UInteger = DUCKDB_TYPE_DUCKDB_TYPE_UINTEGER,
    UBigint = DUCKDB_TYPE_DUCKDB_TYPE_UBIGINT,
    Float = DUCKDB_TYPE_DUCKDB_TYPE_FLOAT,
    Double = DUCKDB_TYPE_DUCKDB_TYPE_DOUBLE,
    Timestamp = DUCKDB_TYPE_DUCKDB_TYPE_TIMESTAMP,
    Date = DUCKDB_TYPE_DUCKDB_TYPE_DATE,
    Time = DUCKDB_TYPE_DUCKDB_TYPE_TIME,
    Interval = DUCKDB_TYPE_DUCKDB_TYPE_INTERVAL,
    Hugeint = DUCKDB_TYPE_DUCKDB_TYPE_HUGEINT,
    Varchar = DUCKDB_TYPE_DUCKDB_TYPE_VARCHAR,
    Blob = DUCKDB_TYPE_DUCKDB_TYPE_BLOB,
    Decimal = DUCKDB_TYPE_DUCKDB_TYPE_DECIMAL,
    TimestampS = DUCKDB_TYPE_DUCKDB_TYPE_TIMESTAMP_S,
    TimestampMs = DUCKDB_TYPE_DUCKDB_TYPE_TIMESTAMP_MS,
    TimestampNs = DUCKDB_TYPE_DUCKDB_TYPE_TIMESTAMP_NS,
    Enum = DUCKDB_TYPE_DUCKDB_TYPE_ENUM,
    List = DUCKDB_TYPE_DUCKDB_TYPE_LIST,
    Struct = DUCKDB_TYPE_DUCKDB_TYPE_STRUCT,
    Map = DUCKDB_TYPE_DUCKDB_TYPE_MAP,
    Uuid = DUCKDB_TYPE_DUCKDB_TYPE_UUID,
    Union = DUCKDB_TYPE_DUCKDB_TYPE_UNION,
}

impl From<u32> for LogicalTypeId {
    fn from(value: u32) -> Self {
        match value {
            DUCKDB_TYPE_DUCKDB_TYPE_BOOLEAN => Self::Boolean,
            DUCKDB_TYPE_DUCKDB_TYPE_TINYINT => Self::Tinyint,
            DUCKDB_TYPE_DUCKDB_TYPE_SMALLINT => Self::Smallint,
            DUCKDB_TYPE_DUCKDB_TYPE_INTEGER => Self::Integer,
            DUCKDB_TYPE_DUCKDB_TYPE_BIGINT => Self::Bigint,
            DUCKDB_TYPE_DUCKDB_TYPE_UTINYINT => Self::UTinyint,
            DUCKDB_TYPE_DUCKDB_TYPE_USMALLINT => Self::USmallint,
            DUCKDB_TYPE_DUCKDB_TYPE_UINTEGER => Self::UInteger,
            DUCKDB_TYPE_DUCKDB_TYPE_UBIGINT => Self::UBigint,
            DUCKDB_TYPE_DUCKDB_TYPE_FLOAT => Self::Float,
            DUCKDB_TYPE_DUCKDB_TYPE_DOUBLE => Self::Double,
            DUCKDB_TYPE_DUCKDB_TYPE_VARCHAR => Self::Varchar,
            DUCKDB_TYPE_DUCKDB_TYPE_BLOB => Self::Blob,
            DUCKDB_TYPE_DUCKDB_TYPE_TIMESTAMP => Self::Timestamp,
            DUCKDB_TYPE_DUCKDB_TYPE_DATE => Self::Date,
            DUCKDB_TYPE_DUCKDB_TYPE_TIME => Self::Time,
            DUCKDB_TYPE_DUCKDB_TYPE_INTERVAL => Self::Interval,
            DUCKDB_TYPE_DUCKDB_TYPE_HUGEINT => Self::Hugeint,
            DUCKDB_TYPE_DUCKDB_TYPE_DECIMAL => Self::Decimal,
            DUCKDB_TYPE_DUCKDB_TYPE_TIMESTAMP_S => Self::TimestampS,
            DUCKDB_TYPE_DUCKDB_TYPE_TIMESTAMP_MS => Self::TimestampMs,
            DUCKDB_TYPE_DUCKDB_TYPE_TIMESTAMP_NS => Self::TimestampNs,
            DUCKDB_TYPE_DUCKDB_TYPE_ENUM => Self::Enum,
            DUCKDB_TYPE_DUCKDB_TYPE_LIST => Self::List,
            DUCKDB_TYPE_DUCKDB_TYPE_STRUCT => Self::Struct,
            DUCKDB_TYPE_DUCKDB_TYPE_MAP => Self::Map,
            DUCKDB_TYPE_DUCKDB_TYPE_UUID => Self::Uuid,
            DUCKDB_TYPE_DUCKDB_TYPE_UNION => Self::Union,
            _ => panic!(),
        }
    }
}

/// DuckDB Logical Type.
///
/// https://duckdb.org/docs/sql/data_types/overview
pub struct LogicalType {
    pub(crate) ptr: duckdb_logical_type,
}

impl Debug for LogicalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let id = self.id();
        match id {
            LogicalTypeId::Struct => {
                write!(f, "struct<")?;
                for i in 0..self.num_children() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {:?}", self.child_name(i), self.child(i))?;
                }
                write!(f, ">")
            }
            _ => write!(f, "{:?}", self.id()),
        }
    }
}

impl Drop for LogicalType {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                duckdb_destroy_logical_type(&mut self.ptr);
            }
        }

        self.ptr = std::ptr::null_mut();
    }
}

/// Wrap a DuckDB logical type from C API
impl From<duckdb_logical_type> for LogicalType {
    fn from(ptr: duckdb_logical_type) -> Self {
        Self { ptr }
    }
}

impl LogicalType {
    /// Create a new [LogicalType] from [LogicalTypeId]
    pub fn new(id: LogicalTypeId) -> Self {
        unsafe {
            Self {
                ptr: duckdb_create_logical_type(id as u32),
            }
        }
    }

    /// Creates a list type from its child type.
    ///
    pub fn list_type(child_type: &LogicalType) -> Self {
        unsafe {
            Self {
                ptr: duckdb_create_list_type(child_type.ptr),
            }
        }
    }

    /// Make a `LogicalType` for `struct`
    ///
    pub fn struct_type(fields: &[(&str, LogicalType)]) -> Self {
        let keys: Vec<CString> = fields.iter().map(|f| CString::new(f.0).unwrap()).collect();
        let values: Vec<duckdb_logical_type> = fields.iter().map(|it| it.1.ptr).collect();
        let name_ptrs = keys
            .iter()
            .map(|it| it.as_ptr())
            .collect::<Vec<*const c_char>>();

        unsafe {
            Self {
                ptr: duckdb_create_struct_type(
                    fields.len() as idx_t,
                    name_ptrs.as_slice().as_ptr().cast_mut(),
                    values.as_slice().as_ptr(),
                ),
            }
        }
    }

    /// Logical type ID
    pub fn id(&self) -> LogicalTypeId {
        let duckdb_type_id = unsafe { duckdb_get_type_id(self.ptr) };
        duckdb_type_id.into()
    }

    pub fn num_children(&self) -> usize {
        match self.id() {
            LogicalTypeId::Struct => unsafe { duckdb_struct_type_child_count(self.ptr) as usize },
            LogicalTypeId::List => 1,
            _ => 0,
        }
    }

    pub fn child_name(&self, idx: usize) -> String {
        assert_eq!(self.id(), LogicalTypeId::Struct);
        unsafe {
            let child_name_ptr = duckdb_struct_type_child_name(self.ptr, idx as u64);
            let c_str = CString::from_raw(child_name_ptr);
            let name = c_str.to_str().unwrap();
            name.to_string()
        }
    }

    pub fn child(&self, idx: usize) -> Self {
        let c_logical_type = unsafe { duckdb_struct_type_child_type(self.ptr, idx as u64) };
        Self::from(c_logical_type)
    }
}
