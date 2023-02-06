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

use std::ffi::{c_char, c_void, CStr, CString};

use duckdb_extension_framework::duckly::{duckdb_data_chunk, duckdb_function_info};
use duckdb_extension_framework::table_functions::{FunctionInfo, TableFunction};
use duckdb_extension_framework::{DataChunk, LogicalType, LogicalTypeId};

#[repr(C)]
struct ScanBindData {
    /// Dataset URI
    uri: *mut c_char,
}

#[repr(C)]
struct ScanInitData {
    done: bool,
}

#[no_mangle]
unsafe extern "C" fn read_lance(info: duckdb_function_info, output: duckdb_data_chunk) {
    let info = FunctionInfo::from(info);
    let output = DataChunk::from(output);

    let bind_data = info.get_bind_data::<ScanBindData>();
    let mut init_data = info.get_init_data::<ScanInitData>();

    let uri = CStr::from_ptr((*bind_data).uri);
    (*init_data).done = true;
    output.set_size(0);
}

pub fn scan_table_function() -> TableFunction {
    let table_function = TableFunction::new();
    table_function.set_name("read_delta");
    let logical_type = LogicalType::new(LogicalTypeId::Varchar);
    table_function.add_parameter(&logical_type);

    table_function.set_function(Some(read_lance));
    table_function.supports_pushdown(true);
    table_function
}
