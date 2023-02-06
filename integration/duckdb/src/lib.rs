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

use std::ffi::{c_char, c_void};

use duckdb_extension_framework::duckly::duckdb_library_version;
use duckdb_extension_framework::Database;

pub mod error;

use error::Result;

#[no_mangle]
pub extern "C" fn lance_version_rust() -> *const c_char {
    unsafe { duckdb_library_version() }
}

#[no_mangle]
pub unsafe extern "C" fn lance_init_rust(db: *mut c_void) {
    init(db).expect("init failed");
}

unsafe fn init(db: *mut c_void) -> Result<()> {
    let db = Database::from_cpp_duckdb(db);
    let table_function = build_table_function_def();
    let connection = db.connect()?;
    connection.register_table_function(table_function)?;
    Ok(())
}

#[cfg(test)]
mod tests {}
