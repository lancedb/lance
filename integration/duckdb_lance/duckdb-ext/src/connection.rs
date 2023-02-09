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

use crate::ffi::{duckdb_connection, duckdb_register_table_function};
use crate::table_function::TableFunction;

/// A connection to a database. This represents a (client) connection that can
/// be used to query the database.
#[derive(Debug)]
pub struct Connection {
    ptr: duckdb_connection,
}

impl From<duckdb_connection> for Connection {
    fn from(ptr: duckdb_connection) -> Self {
        Self { ptr }
    }
}

impl Connection {
    pub fn register_table_function(
        &self,
        table_function: TableFunction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            duckdb_register_table_function(self.ptr, table_function.ptr);
        }
        Ok(())
    }
}
