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

use std::ffi::{c_void, CString};

use crate::ffi::{
    duckdb_bind_add_result_column, duckdb_bind_get_parameter, duckdb_bind_get_parameter_count,
    duckdb_bind_info, duckdb_bind_set_bind_data, duckdb_bind_set_cardinality,
    duckdb_bind_set_error, duckdb_create_table_function, duckdb_delete_callback_t,
    duckdb_destroy_table_function, duckdb_init_get_bind_data, duckdb_init_info,
    duckdb_init_set_error, duckdb_init_set_init_data, duckdb_table_function,
    duckdb_table_function_add_parameter, duckdb_table_function_bind_t,
    duckdb_table_function_init_t, duckdb_table_function_set_bind,
    duckdb_table_function_set_function, duckdb_table_function_set_init,
    duckdb_table_function_set_name, duckdb_table_function_supports_projection_pushdown,
    duckdb_table_function_t,
};
use crate::{Error, LogicalType, Value};

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
    ///
    /// # Safety
    pub fn add_result_column(&self, name: &str, logical_type: LogicalType) {
        let c_string = CString::new(name).unwrap();
        unsafe {
            duckdb_bind_add_result_column(self.ptr, c_string.as_ptr(), logical_type.ptr);
        }
    }

    /// Sets the user-provided bind data in the bind object. This object can be retrieved again during execution.
    ///
    /// # Arguments
    ///  * `extra_data`: The bind data object.
    ///  * `destroy`: The callback that will be called to destroy the bind data (if any)
    ///
    /// # Safety
    ///
    pub fn set_bind_data(
        &self,
        data: *mut c_void,
        free_function: Option<unsafe extern "C" fn(*mut c_void)>,
    ) {
        unsafe {
            duckdb_bind_set_bind_data(self.ptr, data, free_function);
        }
    }

    /// Get the number of regular (non-named) parameters to the function.
    pub fn num_parameters(&self) -> u64 {
        unsafe { duckdb_bind_get_parameter_count(self.ptr) }
    }

    /// Get the parameter at the given index.
    ///
    /// # Arguments
    ///  * `index`: The index of the parameter to get
    ///
    /// returns: The value of the parameter
    pub fn parameter(&self, index: usize) -> Value {
        unsafe { Value::from(duckdb_bind_get_parameter(self.ptr, index as u64)) }
    }

    /// Sets the cardinality estimate for the table function, used for optimization.
    ///
    /// * `cardinality`: The cardinality estimate
    /// * `is_exact`: Whether or not the cardinality estimate is exact, or an approximation
    pub fn set_cardinality(&self, cardinality: usize, is_exact: bool) {
        unsafe { duckdb_bind_set_cardinality(self.ptr, cardinality as u64, is_exact) }
    }

    pub fn set_error(&self, error: Error) {
        unsafe {
            duckdb_bind_set_error(self.ptr, error.c_str().as_ptr());
        }
    }
}

#[derive(Debug)]
pub struct InitInfo {
    ptr: duckdb_init_info,
}

impl From<duckdb_init_info> for InitInfo {
    fn from(ptr: duckdb_init_info) -> Self {
        Self { ptr }
    }
}

impl InitInfo {
    /// # Safety
    pub fn set_init_data(&self, data: *mut c_void, freeer: duckdb_delete_callback_t) {
        unsafe {
            duckdb_init_set_init_data(self.ptr, data, freeer);
        }
    }

    pub fn bind_data<T>(&self) -> *mut T {
        unsafe { duckdb_init_get_bind_data(self.ptr).cast() }
    }

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

    /// Adds a parameter to the table function.
    ///
    pub fn add_parameter(&self, logical_type: &LogicalType) -> &Self {
        unsafe {
            duckdb_table_function_add_parameter(self.ptr, logical_type.ptr);
        }
        self
    }

    /// Enable project pushdown.
    pub fn pushdown(&self, supports: bool) -> &Self {
        unsafe {
            duckdb_table_function_supports_projection_pushdown(self.ptr, supports);
        }
        self
    }

    /// Sets the main function of the table function
    ///
    pub fn set_function(&self, func: duckdb_table_function_t) -> &Self {
        unsafe {
            duckdb_table_function_set_function(self.ptr, func);
        }
        self
    }

    /// Sets the init function of the table function
    ///
    /// # Arguments
    ///  * `function`: The init function
    pub fn set_init(&self, init_func: duckdb_table_function_init_t) -> &Self {
        unsafe {
            duckdb_table_function_set_init(self.ptr, init_func);
        }
        self
    }

    /// Sets the bind function of the table function
    ///
    /// # Arguments
    ///  * `bind_func`: The bind function
    pub fn set_bind(&self, bind_func: duckdb_table_function_bind_t) -> &Self {
        unsafe {
            duckdb_table_function_set_bind(self.ptr, bind_func);
        }
        self
    }
}
