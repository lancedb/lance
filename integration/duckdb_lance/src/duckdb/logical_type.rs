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

use std::fmt::Debug;

use crate::duckdb::ffi::*;

#[repr(u32)]
#[derive(Debug)]
pub enum LogicalTypeId {
    Boolean = DUCKDB_TYPE_DUCKDB_TYPE_BOOLEAN,
    Tinyint = DUCKDB_TYPE_DUCKDB_TYPE_TINYINT,
    Smallint = DUCKDB_TYPE_DUCKDB_TYPE_SMALLINT,
    Integer = DUCKDB_TYPE_DUCKDB_TYPE_INTEGER,
    Bigint = DUCKDB_TYPE_DUCKDB_TYPE_BIGINT,
    UTinyint = DUCKDB_TYPE_DUCKDB_TYPE_UTINYINT,
    USmallint = DUCKDB_TYPE_DUCKDB_TYPE_USMALLINT,
    Uinteger = DUCKDB_TYPE_DUCKDB_TYPE_UINTEGER,
    Ubigint = DUCKDB_TYPE_DUCKDB_TYPE_UBIGINT,
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
    Json = DUCKDB_TYPE_DUCKDB_TYPE_JSON,
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
            DUCKDB_TYPE_DUCKDB_TYPE_UINTEGER => Self::Uinteger,
            DUCKDB_TYPE_DUCKDB_TYPE_UBIGINT => Self::Ubigint,
            DUCKDB_TYPE_DUCKDB_TYPE_FLOAT => Self::Float,
            DUCKDB_TYPE_DUCKDB_TYPE_DOUBLE => Self::Double,
            DUCKDB_TYPE_DUCKDB_TYPE_TIMESTAMP => Self::Timestamp,
            DUCKDB_TYPE_DUCKDB_TYPE_DATE => Self::Date,
            /** .. */
            DUCKDB_TYPE_DUCKDB_TYPE_LIST => Self::List,
            DUCKDB_TYPE_DUCKDB_TYPE_STRUCT => Self::Struct,
            _ => panic!()
        }
    }
}

/// DuckDB Logical Type.
pub struct LogicalType {
    ptr: duckdb_logical_type,
}

impl Debug for LogicalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{:?}", self.id())
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

impl From<duckdb_logical_type> for LogicalType {
    fn from(ptr: duckdb_logical_type) -> Self {
        Self { ptr }
    }
}

impl LogicalType {
    fn id(&self) -> LogicalTypeId {
        let duckdb_type_id = unsafe {
            duckdb_get_type_id(self.ptr)
        };
        duckdb_type_id.into()
    }
}
