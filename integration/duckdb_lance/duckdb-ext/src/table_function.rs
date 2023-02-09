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

use std::ffi::CString;

use crate::ffi::{
    duckdb_bind_add_result_column, duckdb_bind_info, duckdb_create_table_function,
    duckdb_destroy_table_function, duckdb_init_info, duckdb_init_set_error, duckdb_table_function,
    duckdb_table_function_set_name,
};
use crate::{Error, LogicalType};

/// DuckDB BindInfo.
pub struct BindInfo {
    ptr: duckdb_bind_info,
}

impl From<duckdb_bind_info> for BindInfo {
    fn from(ptr: duckdb_bind_info) -> Self {
        Self { ptr }
    }
}

impl BindInfo {
    /// Add a result column to the output of the table function.
    ///
    ///  - `name`: The name of the column
    ///  - `logical_type`: The [LogicalType] of the new column.
    pub fn add_result_column(&self, name: &str, logical_type: LogicalType) {
        let c_string = CString::new(name).unwrap();
        unsafe {
            duckdb_bind_add_result_column(self.ptr, c_string.as_ptr(), logical_type.ptr);
        }
    }
}

#[derive(Debug)]
pub struct InitInfo {
    ptr: duckdb_init_info,
}

impl InitInfo {
    /// Report that an error has occurred while calling init.
    ///
    /// # Arguments
    /// * `error`: The error message
    pub fn set_error(&self, error: Error) {
        unsafe { duckdb_init_set_error(self.ptr, error.c_str().as_ptr()) }
    }
}

/// A function that returns a queryable table
#[derive(Debug)]
pub struct TableFunction {
    pub(crate) ptr: duckdb_table_function,
}

impl Drop for TableFunction {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                duckdb_destroy_table_function(&mut self.ptr);
            }
        }
        self.ptr = std::ptr::null_mut();
    }
}

impl TableFunction {
    /// Creates a new empty table function.
    pub fn new(name: &str) -> Self {
        let this = Self {
            ptr: unsafe { duckdb_create_table_function() },
        };
        this.set_name(name);
        this
    }

    pub fn set_name(&self, name: &str) -> &Self {
        unsafe {
            let string = CString::new(name).unwrap();
            duckdb_table_function_set_name(self.ptr, string.as_ptr());
        }
        self
    }
}
