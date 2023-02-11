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

use super::vector::{FlatVector, ListVector, StructVector};
use crate::{
    ffi::{
        duckdb_create_data_chunk, duckdb_data_chunk, duckdb_data_chunk_get_size,
        duckdb_data_chunk_get_vector, duckdb_data_chunk_set_size, duckdb_destroy_data_chunk,
        duckdb_data_chunk_get_column_count,
    },
    LogicalType,
};

/// DataChunk in DuckDB.
pub struct DataChunk {
    /// Pointer to the DataChunk in duckdb C API.
    ptr: duckdb_data_chunk,

    /// Whether this [DataChunk] own the [DataChunk::ptr].
    owned: bool,
}

impl DataChunk {
    pub fn new(logical_types: &[LogicalType]) -> Self {
        let num_columns = logical_types.len();
        let mut c_types = logical_types.iter().map(|t| t.ptr).collect::<Vec<_>>();
        let ptr = unsafe { duckdb_create_data_chunk(c_types.as_mut_ptr(), num_columns as u64) };
        DataChunk { ptr, owned: true }
    }

    /// Get the vector at the specific column index: `idx`.
    ///
    pub fn flat_vector(&self, idx: usize) -> FlatVector {
        FlatVector::from(unsafe { duckdb_data_chunk_get_vector(self.ptr, idx as u64) })
    }

    /// Get a list vector from the column index.
    pub fn list_vector(&self, idx: usize) -> ListVector {
        ListVector::from(unsafe { duckdb_data_chunk_get_vector(self.ptr, idx as u64) })
    }

    /// Get struct vector at the column index: `idx`.
    pub fn struct_vector(&self, idx: usize) -> StructVector {
        StructVector::from(unsafe { duckdb_data_chunk_get_vector(self.ptr, idx as u64) })
    }

    /// Set the size of the data chunk
    pub fn set_len(&self, new_len: usize) {
        unsafe { duckdb_data_chunk_set_size(self.ptr, new_len as u64) };
    }

    /// Get the length / the number of rows in this [DataChunk].
    pub fn len(&self) -> usize {
        unsafe { duckdb_data_chunk_get_size(self.ptr) as usize }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_columns(&self) -> usize {
        unsafe { duckdb_data_chunk_get_column_count(self.ptr) as usize }
    }
}

impl From<duckdb_data_chunk> for DataChunk {
    fn from(ptr: duckdb_data_chunk) -> Self {
        Self { ptr, owned: false }
    }
}

impl Drop for DataChunk {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            unsafe { duckdb_destroy_data_chunk(&mut self.ptr) }
            self.ptr = std::ptr::null_mut();
        }
    }
}
