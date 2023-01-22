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

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeBinaryArray, FixedSizeListArray, Int32Array, Int64Array,
    LargeListArray, ListArray, RecordBatch, UInt8Array,
};
use arrow_data::ArrayDataBuilder;
use arrow_schema::{DataType, Field, Schema};

mod kernels;
mod record_batch;
use crate::error::{Error, Result};
pub use kernels::*;
pub use record_batch::*;

pub trait DataTypeExt {
    /// Returns true if the data type is binary-like, such as (Large)Utf8 and (Large)Binary.
    ///
    /// ```
    /// use lance::arrow::*;
    /// use arrow_schema::DataType;
    ///
    /// assert!(DataType::Utf8.is_binary_like());
    /// assert!(DataType::Binary.is_binary_like());
    /// assert!(DataType::LargeUtf8.is_binary_like());
    /// assert!(DataType::LargeBinary.is_binary_like());
    /// assert!(!DataType::Int32.is_binary_like());
    /// ```
    fn is_binary_like(&self) -> bool;

    /// Returns true if the data type is a struct.
    fn is_struct(&self) -> bool;

    /// Check whether the given Arrow DataType is fixed stride.
    ///
    /// A fixed stride type has the same byte width for all array elements
    /// This includes all PrimitiveType's Boolean, FixedSizeList, FixedSizeBinary, and Decimals
    fn is_fixed_stride(&self) -> bool;

    /// Returns true if the [DataType] is a dictionary type.
    fn is_dictionary(&self) -> bool;
}

impl DataTypeExt for DataType {
    fn is_binary_like(&self) -> bool {
        use DataType::*;
        matches!(self, Utf8 | Binary | LargeUtf8 | LargeBinary)
    }

    fn is_struct(&self) -> bool {
        matches!(self, Self::Struct(_))
    }

    fn is_fixed_stride(&self) -> bool {
        use DataType::*;
        matches!(
            self,
            Boolean
                | UInt8
                | UInt16
                | UInt32
                | UInt64
                | Int8
                | Int16
                | Int32
                | Int64
                | Float16
                | Float32
                | Float64
                | Decimal128(_, _)
                | Decimal256(_, _)
                | FixedSizeList(_, _)
                | FixedSizeBinary(_)
        )
    }

    fn is_dictionary(&self) -> bool {
        matches!(self, Self::Dictionary(_, _))
    }
}

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
    /// let list_arr = ListArray::try_new(int_values, &offsets).unwrap();
    /// assert_eq!(list_arr,
    ///     ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
    ///         Some(vec![Some(0), Some(1)]),
    ///         Some(vec![Some(2), Some(3), Some(4), Some(5), Some(6)]),
    ///         Some(vec![Some(7), Some(8), Some(9)]),
    /// ]))
    /// ```
    fn try_new<T: Array>(values: T, offsets: &Int32Array) -> Result<ListArray>;
}

impl ListArrayExt for ListArray {
    fn try_new<T: Array>(values: T, offsets: &Int32Array) -> Result<Self> {
        let data = ArrayDataBuilder::new(DataType::List(Box::new(Field::new(
            "item",
            values.data_type().clone(),
            true,
        ))))
        .len(offsets.len() - 1)
        .add_buffer(offsets.into_data().buffers()[0].clone())
        .add_child_data(values.into_data())
        .build()?;

        Ok(Self::from(data))
    }
}

// TODO: merge with ListArrayExt?;
pub trait LargeListArrayExt {
    fn try_new<T: Array>(values: T, offsets: &Int64Array) -> Result<LargeListArray>;
}

impl LargeListArrayExt for LargeListArray {
    fn try_new<T: Array>(values: T, offsets: &Int64Array) -> Result<Self> {
        let data = ArrayDataBuilder::new(DataType::LargeList(Box::new(Field::new(
            "item",
            values.data_type().clone(),
            true,
        ))))
        .len(offsets.len() - 1)
        .add_buffer(offsets.into_data().buffers()[0].clone())
        .add_child_data(values.into_data())
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
    /// let fixed_size_list_arr = FixedSizeListArray::try_new(int_values, 2).unwrap();
    /// assert_eq!(fixed_size_list_arr,
    ///     FixedSizeListArray::from_iter_primitive::<Int64Type, _, _>(vec![
    ///         Some(vec![Some(0), Some(1)]),
    ///         Some(vec![Some(2), Some(3)]),
    ///         Some(vec![Some(4), Some(5)]),
    ///         Some(vec![Some(6), Some(7)]),
    ///         Some(vec![Some(8), Some(9)])
    /// ], 2))
    /// ```
    fn try_new<T: Array>(values: T, list_size: i32) -> Result<FixedSizeListArray>;
}

impl FixedSizeListArrayExt for FixedSizeListArray {
    fn try_new<T: Array>(values: T, list_size: i32) -> Result<Self> {
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

/// Force downcast of an [`Array`], such as an [`ArrayRef`], to
/// [`FixedSizeListArray`], panic'ing on failure.
pub fn as_fixed_size_list_array(arr: &dyn Array) -> &FixedSizeListArray {
    arr.as_any().downcast_ref::<FixedSizeListArray>().unwrap()
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
    /// let fixed_size_list_arr = FixedSizeBinaryArray::try_new(&int_values, 2).unwrap();
    /// assert_eq!(fixed_size_list_arr,
    ///     FixedSizeBinaryArray::from(vec![
    ///         Some(vec![0, 1].as_slice()),
    ///         Some(vec![2, 3].as_slice()),
    ///         Some(vec![4, 5].as_slice()),
    ///         Some(vec![6, 7].as_slice()),
    ///         Some(vec![8, 9].as_slice())
    /// ]))
    /// ```
    fn try_new(values: &UInt8Array, stride: i32) -> Result<FixedSizeBinaryArray>;
}

impl FixedSizeBinaryArrayExt for FixedSizeBinaryArray {
    fn try_new(values: &UInt8Array, stride: i32) -> Result<Self> {
        let data_type = DataType::FixedSizeBinary(stride);
        let data = ArrayDataBuilder::new(data_type)
            .len(values.len() / stride as usize)
            .add_buffer(values.data().buffers()[0].clone())
            .build()?;
        Ok(Self::from(data))
    }
}

/// Extends Arrow's [RecordBatch].
pub trait RecordBatchExt {
    /// Append a new column to this [`RecordBatch`] and returns a new RecordBatch.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use arrow_array::{RecordBatch, Int32Array, StringArray};
    /// use arrow_schema::{Schema, Field, DataType};
    /// use lance::arrow::*;
    ///
    /// let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
    /// let int_arr = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
    /// let record_batch = RecordBatch::try_new(schema, vec![int_arr.clone()]).unwrap();
    ///
    /// let new_field = Field::new("s", DataType::Utf8, true);
    /// let str_arr = Arc::new(StringArray::from(vec!["a", "b", "c", "d"]));
    /// let new_record_batch = record_batch.try_with_column(new_field, str_arr.clone()).unwrap();
    ///
    /// assert_eq!(
    ///     new_record_batch,
    ///     RecordBatch::try_new(
    ///         Arc::new(Schema::new(
    ///             vec![
    ///                 Field::new("a", DataType::Int32, true),
    ///                 Field::new("s", DataType::Utf8, true)
    ///             ])
    ///         ),
    ///         vec![int_arr, str_arr],
    ///     ).unwrap()
    /// )
    /// ```
    fn try_with_column(&self, field: Field, arr: ArrayRef) -> Result<RecordBatch>;

    /// Merge with another [`RecordBatch`] and returns a new one.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use arrow_array::*;
    /// use arrow_schema::{Schema, Field, DataType};
    /// use lance::arrow::*;
    ///
    /// let left_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
    /// let int_arr = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
    /// let left = RecordBatch::try_new(left_schema, vec![int_arr.clone()]).unwrap();
    ///
    /// let right_schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
    /// let str_arr = Arc::new(StringArray::from(vec!["a", "b", "c", "d"]));
    /// let right = RecordBatch::try_new(right_schema, vec![str_arr.clone()]).unwrap();
    ///
    /// let new_record_batch = left.merge(&right).unwrap();
    ///
    /// assert_eq!(
    ///     new_record_batch,
    ///     RecordBatch::try_new(
    ///         Arc::new(Schema::new(
    ///             vec![
    ///                 Field::new("a", DataType::Int32, true),
    ///                 Field::new("s", DataType::Utf8, true)
    ///             ])
    ///         ),
    ///         vec![int_arr, str_arr],
    ///     ).unwrap()
    /// )
    /// ```
    ///
    /// TODO: add merge nested fields support.
    fn merge(&self, other: &RecordBatch) -> Result<RecordBatch>;

    /// Drop one column specified with the name and return the new [`RecordBatch`].
    ///
    /// If the named column does not exist, it returns a copy of this [`RecordBatch`].
    fn drop_column(&self, name: &str) -> Result<RecordBatch>;
}

impl RecordBatchExt for RecordBatch {
    fn try_with_column(&self, field: Field, arr: ArrayRef) -> Result<Self> {
        let mut new_fields = self.schema().fields.clone();
        new_fields.push(field);
        let new_schema = Arc::new(Schema::new_with_metadata(
            new_fields,
            self.schema().metadata.clone(),
        ));
        let mut new_columns = self.columns().to_vec();
        new_columns.push(arr);
        Ok(Self::try_new(new_schema, new_columns)?)
    }

    fn merge(&self, other: &RecordBatch) -> Result<RecordBatch> {
        if self.num_rows() != other.num_rows() {
            return Err(Error::Arrow(format!(
                "Attempt to merge two RecordBatch with different sizes: {} != {}",
                self.num_rows(),
                other.num_rows()
            )));
        }

        let mut fields = self.schema().fields.clone();
        let mut columns = Vec::from(self.columns());
        for field in other.schema().fields.as_slice() {
            if !fields.iter().any(|f| f.name() == field.name()) {
                fields.push(field.clone());
                columns.push(
                    other
                        .column_by_name(field.name())
                        .ok_or_else(|| {
                            Error::Arrow(format!(
                                "Column {} does not exist: schema={}",
                                field.name(),
                                other.schema()
                            ))
                        })?
                        .clone(),
                );
            }
        }
        Ok(Self::try_new(Arc::new(Schema::new(fields)), columns)?)
    }

    fn drop_column(&self, name: &str) -> Result<RecordBatch> {
        let mut fields = vec![];
        let mut columns = vec![];
        for i in 0..self.schema().fields.len() {
            if self.schema().field(i).name() != name {
                fields.push(self.schema().field(i).clone());
                columns.push(self.column(i).clone());
            }
        }
        Ok(RecordBatch::try_new(
            Arc::new(Schema::new_with_metadata(
                fields,
                self.schema().metadata().clone(),
            )),
            columns,
        )?)
    }
}
