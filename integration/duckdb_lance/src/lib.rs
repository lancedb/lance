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

use duckdb_ext::ffi::{duckdb_library_version, _duckdb_database};
use duckdb_ext::Database;
use tokio::runtime::Runtime;

mod arrow;
pub mod error;
mod scan;

use crate::scan::scan_table_function;
use error::{Error, Result};

lazy_static::lazy_static! {
    static ref RUNTIME: Runtime = tokio::runtime::Runtime::new()
            .expect("Creating Tokio runtime");
}

#[no_mangle]
pub extern "C" fn lance_version_rust() -> *const c_char {
    unsafe { duckdb_library_version() }
}

#[no_mangle]
pub unsafe extern "C" fn lance_init_rust(db: *mut _duckdb_database) {
    init(db).expect("duckdb lance extension init failed");
}

unsafe fn init(db: *mut _duckdb_database) -> Result<()> {
    let db = Database::from(db);
    let table_function = scan_table_function();
    let connection = db.connect()?;
    connection.register_table_function(table_function)?;
    Ok(())
}

#[cfg(test)]
mod tests {}
