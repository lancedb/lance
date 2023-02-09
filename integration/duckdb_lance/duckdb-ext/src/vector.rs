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

use std::ffi::CString;
use std::ops::{Index, IndexMut};
use std::{marker::PhantomData, slice};

use crate::ffi::{
    duckdb_list_vector_get_child, duckdb_list_vector_get_size, duckdb_struct_type_child_count,
    duckdb_struct_type_child_name, duckdb_struct_vector_get_child, duckdb_vector,
    duckdb_vector_assign_string_element, duckdb_vector_get_column_type, duckdb_vector_get_data,
    duckdb_vector_size,
};
use crate::LogicalType;

pub struct Vector<T: Copy> {
    ptr: duckdb_vector,
    phantom: PhantomData<T>,
}

impl<T: Copy> From<duckdb_vector> for Vector<T> {
    fn from(ptr: duckdb_vector) -> Self {
        Self {
            ptr,
            phantom: PhantomData {},
        }
    }
}

impl<T: Copy> Vector<T> {
    pub fn capacity(&self) -> usize {
        unsafe { duckdb_vector_size() as usize }
    }

    /// Returns an unsafe mutable pointer to the vector’s
    pub fn as_mut_ptr(&self) -> *mut T {
        unsafe { duckdb_vector_get_data(self.ptr).cast() }
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_mut_ptr(), self.capacity()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.capacity()) }
    }

    pub fn logical_type(&self) -> LogicalType {
        LogicalType::from(unsafe { duckdb_vector_get_column_type(self.ptr) })
    }

    pub fn copy(&mut self, data: &[T]) {
        assert!(data.len() <= self.capacity());
        println!("Data len: {} capacity={}", data.len(), self.capacity());
        println!("Self len: {}", self.as_mut_slice().len());
        println!("self.ptr : {:p} data: {:p}", self.ptr, data.as_ptr());
        self.as_mut_slice().copy_from_slice(data);
        println!("Done assignment");
    }
}

impl<T: Copy> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<T: Copy> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

pub trait Inserter<T> {
    fn insert(&self, index: usize, value: T);
}

impl Inserter<&str> for Vector<&str> {
    fn insert(&self, index: usize, value: &str) {
        let cstr = CString::new(value.as_bytes()).unwrap();
        unsafe {
            duckdb_vector_assign_string_element(self.ptr, index as u64, cstr.as_ptr());
        }
    }
}

pub struct ListVector {
    /// ListVector does not own the vector pointer.
    ptr: duckdb_vector,
}

impl From<duckdb_vector> for ListVector {
    fn from(ptr: duckdb_vector) -> Self {
        Self { ptr }
    }
}

impl ListVector {
    pub fn len(&self) -> usize {
        unsafe { duckdb_list_vector_get_size(self.ptr) as usize }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn child<T: Copy>(&self) -> Vector<T> {
        Vector::from(unsafe { duckdb_list_vector_get_child(self.ptr) })
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
    pub fn child<T: Copy>(&self, idx: usize) -> Vector<T> {
        Vector::from(unsafe { duckdb_struct_vector_get_child(self.ptr, idx as u64) })
    }


    /// Take the child as [StructVector].
    pub fn struct_vector_child(&self, idx: usize) -> StructVector {
        Self::from(unsafe { duckdb_struct_vector_get_child(self.ptr, idx as u64) })
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
