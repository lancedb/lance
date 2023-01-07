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

use arrow_array::{Array, Int32Array, ListArray};
use arrow_data::ArrayDataBuilder;
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
