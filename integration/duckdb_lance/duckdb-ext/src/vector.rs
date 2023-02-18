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

use std::any::Any;
use std::ffi::CString;
use std::slice;

use crate::ffi::{
    duckdb_list_entry, duckdb_list_vector_get_child, duckdb_list_vector_get_size,
    duckdb_list_vector_reserve, duckdb_list_vector_set_size, duckdb_struct_type_child_count,
    duckdb_struct_type_child_name, duckdb_struct_vector_get_child, duckdb_vector,
    duckdb_vector_assign_string_element, duckdb_vector_get_column_type, duckdb_vector_get_data,
    duckdb_vector_size,
};
use crate::LogicalType;

/// Vector trait.
pub trait Vector {
    fn as_any(&self) -> &dyn Any;

    fn as_mut_any(&mut self) -> &mut dyn Any;
}

pub struct FlatVector {
    ptr: duckdb_vector,
    capacity: usize,
}

impl From<duckdb_vector> for FlatVector {
    fn from(ptr: duckdb_vector) -> Self {
        Self {
            ptr,
            capacity: unsafe { duckdb_vector_size() as usize },
        }
    }
}

impl Vector for FlatVector {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn Any {
        self
    }
}

impl FlatVector {
    fn with_capacity(ptr: duckdb_vector, capacity: usize) -> Self {
        Self { ptr, capacity }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns an unsafe mutable pointer to the vector’s
    pub fn as_mut_ptr<T>(&self) -> *mut T {
        unsafe { duckdb_vector_get_data(self.ptr).cast() }
    }

    pub fn as_slice<T>(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_mut_ptr(), self.capacity()) }
    }

    pub fn as_mut_slice<T>(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.capacity()) }
    }

    pub fn logical_type(&self) -> LogicalType {
        LogicalType::from(unsafe { duckdb_vector_get_column_type(self.ptr) })
    }

    pub fn copy<T: Copy>(&mut self, data: &[T]) {
        assert!(data.len() <= self.capacity());
        self.as_mut_slice::<T>()[0..data.len()].copy_from_slice(data);
    }
}

pub trait Inserter<T> {
    fn insert(&self, index: usize, value: T);
}

impl Inserter<&str> for FlatVector {
    fn insert(&self, index: usize, value: &str) {
        let cstr = CString::new(value.as_bytes()).unwrap();
        unsafe {
            duckdb_vector_assign_string_element(self.ptr, index as u64, cstr.as_ptr());
        }
    }
}

pub struct ListVector {
    /// ListVector does not own the vector pointer.
    entries: FlatVector,
}

impl From<duckdb_vector> for ListVector {
    fn from(ptr: duckdb_vector) -> Self {
        Self {
            entries: FlatVector::from(ptr),
        }
    }
}

impl ListVector {
    pub fn len(&self) -> usize {
        unsafe { duckdb_list_vector_get_size(self.entries.ptr) as usize }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // TODO: not ideal interface. Where should we keep capacity.
    pub fn child(&self, capacity: usize) -> FlatVector {
        self.reserve(capacity);
        FlatVector::with_capacity(
            unsafe { duckdb_list_vector_get_child(self.entries.ptr) },
            capacity,
        )
    }

    /// Set primitive data to the child node.
    pub fn set_child<T: Copy>(&self, data: &[T]) {
        self.child(data.len()).copy(data);
        self.set_len(data.len());
    }

    pub fn set_entry(&mut self, idx: usize, offset: usize, length: usize) {
        self.entries.as_mut_slice::<duckdb_list_entry>()[idx].offset = offset as u64;
        self.entries.as_mut_slice::<duckdb_list_entry>()[idx].length = length as u64;
    }

    /// Reserve the capacity for its child node.
    fn reserve(&self, capacity: usize) {
        unsafe { duckdb_list_vector_reserve(self.entries.ptr, capacity as u64); }
    }

    pub fn set_len(&self, new_len: usize) {
        unsafe { duckdb_list_vector_set_size(self.entries.ptr, new_len as u64); }
    }
}

pub struct StructVector {
    /// ListVector does not own the vector pointer.
    ptr: duckdb_vector,
}

impl From<duckdb_vector> for StructVector {
    fn from(ptr: duckdb_vector) -> Self {
        Self { ptr }
    }
}

impl StructVector {
    pub fn child(&self, idx: usize) -> FlatVector {
        FlatVector::from(unsafe { duckdb_struct_vector_get_child(self.ptr, idx as u64) })
    }

    /// Take the child as [StructVector].
    pub fn struct_vector_child(&self, idx: usize) -> StructVector {
        Self::from(unsafe { duckdb_struct_vector_get_child(self.ptr, idx as u64) })
    }

    pub fn list_vector_child(&self, idx: usize) -> ListVector {
        ListVector::from(unsafe { duckdb_struct_vector_get_child(self.ptr, idx as u64) })
    }

    /// Get the logical type of this struct vector.
    pub fn logical_type(&self) -> LogicalType {
        LogicalType::from(unsafe { duckdb_vector_get_column_type(self.ptr) })
    }

    pub fn child_name(&self, idx: usize) -> String {
        let logical_type = self.logical_type();
        unsafe {
            let child_name_ptr = duckdb_struct_type_child_name(logical_type.ptr, idx as u64);
            let c_str = CString::from_raw(child_name_ptr);
            let name = c_str.to_str().unwrap();
            // duckdb_free(child_name_ptr.cast());
            name.to_string()
        }
    }

    pub fn num_children(&self) -> usize {
        let logical_type = self.logical_type();
        unsafe { duckdb_struct_type_child_count(logical_type.ptr) as usize }
    }
}
