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

use duckdb_extension_framework::duckly::{
    duckdb_bind_info, duckdb_data_chunk, duckdb_free, duckdb_function_info, duckdb_init_info,
};
use duckdb_extension_framework::table_functions::{
    BindInfo, FunctionInfo, InitInfo, TableFunction,
};
use duckdb_extension_framework::{malloc_struct, DataChunk, LogicalType, LogicalTypeId};
use lance::dataset::scanner::ScannerStream;
use lance::dataset::Dataset;

use crate::arrow::to_duckdb_logical_type;

#[repr(C)]
struct ScanBindData {
    /// Dataset URI
    uri: *mut c_char,
}

/// Drop the ScanBindData from C.
unsafe extern "C" fn drop_scan_bind_data_c(v: *mut c_void) {
    let actual = v.cast::<ScanBindData>();

    // drop(Arc::from_raw(*actual).stream));
    duckdb_free(v);
}

#[repr(C)]
struct ScanInitData {
    stream: ScannerStream,

    done: bool,
}

#[no_mangle]
unsafe extern "C" fn read_lance(info: duckdb_function_info, output: duckdb_data_chunk) {
    let info = FunctionInfo::from(info);
    let output = DataChunk::from(output);

    // let bind_data = info.get_bind_data::<ScanBindData>();
    let mut init_data = info.get_init_data::<ScanInitData>();

    // let uri = CStr::from_ptr((*bind_data).uri);
    (*init_data).done = true;
    output.set_size(0);
}

#[no_mangle]
unsafe extern "C" fn read_lance_init(info: duckdb_init_info) {
    let info = InitInfo::from(info);
    let bind_data = info.get_bind_data::<ScanBindData>();

    let column_indices = info.get_column_indices();
    let uri = CStr::from_ptr((*bind_data).uri);
    let dataset =
        match crate::RUNTIME.block_on(async { Dataset::open(uri.to_str().unwrap()).await }) {
            Ok(d) => d,
            Err(e) => {
                info.set_error(CString::new(e.to_string()).expect("Create error message"));
                return;
            }
        };
    println!("Open dataset: {}\n", dataset.schema());

    let mut init_data = malloc_struct::<ScanInitData>();
    (*init_data).done = false;
    info.set_init_data(init_data.cast(), Some(duckdb_free));
}

#[no_mangle]
unsafe extern "C" fn read_lance_bind_c(bind_info: duckdb_bind_info) {
    let bind_info = BindInfo::from(bind_info);
    assert!(bind_info.get_parameter_count() >= 1);

    read_lance_bind(&bind_info);
}

fn read_lance_bind(bind: &BindInfo) {
    let uri_param = bind.get_parameter(0).get_varchar();

    let uri = uri_param.to_str().unwrap();
    let dataset =
        match crate::RUNTIME.block_on(async { Dataset::open(uri).await }) {
            Ok(d) => d,
            Err(e) => {
                bind.set_error(e.to_string().as_str());
                return;
            }
        };

    let schema = dataset.schema();
    for field in schema.fields.iter() {
        bind.add_result_column(
            &field.name,
            to_duckdb_logical_type(&field.data_type()).unwrap(),
        );
    }

    unsafe {
        let bind_data = malloc_struct::<ScanBindData>();
        (*bind_data).uri = CString::new(uri).expect("Bind uri").into_raw();

        bind.set_bind_data(bind_data.cast(), Some(drop_scan_bind_data_c));
    }
}

pub fn scan_table_function() -> TableFunction {
    let table_function = TableFunction::new();
    table_function.set_name("lance_scan");
    let logical_type = LogicalType::new(LogicalTypeId::Varchar);
    table_function.add_parameter(&logical_type);

    table_function.set_function(Some(read_lance));
    table_function.set_init(Some(read_lance_init));
    table_function.set_bind(Some(read_lance_bind_c));
    table_function.supports_pushdown(true);
    // TODO: add filter push down.
    table_function
}
