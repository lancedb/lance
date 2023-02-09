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

use crate::ffi::{duckdb_connect, duckdb_connection, duckdb_database, duckdb_state_DuckDBError};
use crate::{Connection, Error, Result};

pub struct Database {
    ptr: duckdb_database,
}

impl From<duckdb_database> for Database {
    fn from(ptr: duckdb_database) -> Self {
        Self { ptr }
    }
}

impl Database {
    pub fn connect(&self) -> Result<Connection> {
        let mut connection: duckdb_connection = std::ptr::null_mut();

        let state = unsafe { duckdb_connect(self.ptr, &mut connection) };
        if state == duckdb_state_DuckDBError {
            return Err(Error::DuckDB("Connection error".to_string()));
        }

        Ok(Connection::from(connection))
    }
}
