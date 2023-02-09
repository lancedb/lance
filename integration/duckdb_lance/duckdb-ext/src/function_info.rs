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

use crate::ffi::{duckdb_function_get_init_data, duckdb_function_info, duckdb_function_set_error};
use crate::Error;

/// UDF
pub struct FunctionInfo {
    ptr: duckdb_function_info,
}

impl From<duckdb_function_info> for FunctionInfo {
    fn from(ptr: duckdb_function_info) -> Self {
        Self { ptr }
    }
}

impl FunctionInfo {
    pub fn init_data<T>(&self) -> *mut T {
        unsafe { duckdb_function_get_init_data(self.ptr).cast() }
    }

    pub fn set_error(&self, error: Error) {
        unsafe {
            duckdb_function_set_error(self.ptr, error.c_str().as_ptr());
        }
    }
}
