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
use std::{marker::PhantomData, slice};
use std::ops::{Index, IndexMut};

use crate::duckdb::ffi::{duckdb_vector, duckdb_vector_get_data, duckdb_vector_size, duckdb_vector_assign_string_element};

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

    pub fn as_mut_slice(&self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.capacity()) }
    }

    pub fn assign(&self, data: &[T]) {
        assert!(data.len() <= self.capacity());

        for i in 0..data.len() {
            self.as_mut_slice()[i] = data[i];
        }
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
    ptr: duckdb_vector,
}

impl From<duckdb_vector> for ListVector {
    fn from(ptr: duckdb_vector) -> Self {
        Self {
            ptr,
        }
    }
}