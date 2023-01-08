// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Extend Arrow Functionality
//!
//! To improve Arrow-RS egonomitic

use arrow_array::types::UInt8Type;
use arrow_array::{
    Array, FixedSizeBinaryArray, FixedSizeListArray, Int32Array, ListArray, UInt8Array,
};
use arrow_data::{ArrayData, ArrayDataBuilder};
use arrow_schema::DataType::FixedSizeBinary;
use arrow_schema::{DataType, Field};

use crate::error::Result;
pub trait ListArrayExt {
    /// Create an [`ListArray`] from values and offsets.
    ///
    /// ```
    /// use arrow_array::{Int32Array, Int64Array, ListArray};
    /// use arrow_array::types::Int64Type;
    /// use lance::arrow::ListArrayExt;
    ///
    /// let offsets = Int32Array::from_iter([0, 2, 7, 10]);
    /// let int_values = Int64Array::from_iter(0..10);
    /// let list_arr = ListArray::new(int_values, &offsets).unwrap();
    /// assert_eq!(list_arr,
    ///     ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
    ///         Some(vec![Some(0), Some(1)]),
    ///         Some(vec![Some(2), Some(3), Some(4), Some(5), Some(6)]),
    ///         Some(vec![Some(7), Some(8), Some(9)]),
    /// ]))
    /// ```
    fn new<T: Array>(values: T, offsets: &Int32Array) -> Result<ListArray>;
}

impl ListArrayExt for ListArray {
    fn new<T: Array>(values: T, offsets: &Int32Array) -> Result<Self> {
        let data = ArrayDataBuilder::new(DataType::List(Box::new(Field::new(
            "item",
            values.data_type().clone(),
            true,
        ))))
        .len(offsets.len() - 1)
        .add_buffer(offsets.into_data().buffers()[0].clone())
        .add_child_data(values.into_data().clone())
        .build()?;

        Ok(Self::from(data))
    }
}

pub trait FixedSizeListArrayExt {
    /// Create an [`FixedSizeListArray`] from values and list size.
    ///
    /// ```
    /// use arrow_array::{Int64Array, FixedSizeListArray};
    /// use arrow_array::types::Int64Type;
    /// use lance::arrow::FixedSizeListArrayExt;
    ///
    /// let int_values = Int64Array::from_iter(0..10);
    /// let fixed_size_list_arr = FixedSizeListArray::new(int_values, 2).unwrap();
    /// assert_eq!(fixed_size_list_arr,
    ///     FixedSizeListArray::from_iter_primitive::<Int64Type, _, _>(vec![
    ///         Some(vec![Some(0), Some(1)]),
    ///         Some(vec![Some(2), Some(3)]),
    ///         Some(vec![Some(4), Some(5)]),
    ///         Some(vec![Some(6), Some(7)]),
    ///         Some(vec![Some(8), Some(9)])
    /// ], 2))
    /// ```
    fn new<T: Array>(values: T, list_size: i32) -> Result<FixedSizeListArray>;
}

impl FixedSizeListArrayExt for FixedSizeListArray {
    fn new<T: Array>(values: T, list_size: i32) -> Result<Self> {
        let list_type = DataType::FixedSizeList(
            Box::new(Field::new("item", values.data_type().clone(), true)),
            list_size,
        );
        let data = ArrayDataBuilder::new(list_type)
            .len(values.len() / list_size as usize)
            .add_child_data(values.data().clone())
            .build()?;

        Ok(Self::from(data))
    }
}

pub trait FixedSizeBinaryArrayExt {
    /// Create an [`FixedSizeBinaryArray`] from values and stride.
    ///
    /// ```
    /// use arrow_array::{UInt8Array, FixedSizeBinaryArray};
    /// use arrow_array::types::UInt8Type;
    /// use lance::arrow::FixedSizeBinaryArrayExt;
    ///
    /// let int_values = UInt8Array::from_iter(0..10);
    /// let fixed_size_list_arr = FixedSizeBinaryArray::new(&int_values, 2).unwrap();
    /// assert_eq!(fixed_size_list_arr,
    ///     FixedSizeBinaryArray::from(vec![
    ///         Some(vec![0, 1].as_slice()),
    ///         Some(vec![2, 3].as_slice()),
    ///         Some(vec![4, 5].as_slice()),
    ///         Some(vec![6, 7].as_slice()),
    ///         Some(vec![8, 9].as_slice())
    /// ]))
    /// ```
    fn new(values: &UInt8Array, stride: i32) -> Result<FixedSizeBinaryArray>;
}

impl FixedSizeBinaryArrayExt for FixedSizeBinaryArray {
    fn new(values: &UInt8Array, stride: i32) -> Result<Self> {
        let data_type = DataType::FixedSizeBinary(stride);
        let data = ArrayDataBuilder::new(data_type)
            .len(values.len() / stride as usize)
            .add_buffer(values.data().buffers()[0].clone())
            .build()?;
        Ok(Self::from(data))
    }
}
