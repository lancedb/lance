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