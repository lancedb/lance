// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Extend Arrow Functionality
//!
//! To improve Arrow-RS ergonomic

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    cast::AsArray, Array, ArrayRef, ArrowNumericType, FixedSizeBinaryArray, FixedSizeListArray, Float16Array, Float32Array, Float64Array, GenericListArray, Int16Array, Int32Array, Int64Array, Int8Array, OffsetSizeTrait, PrimitiveArray, RecordBatch, StructArray, UInt32Array, UInt8Array
};
use arrow_data::ArrayDataBuilder;
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Fields, IntervalUnit, Schema};
use arrow_select::take::take;
use rand::prelude::*;

pub mod deepcopy;
pub mod schema;
pub use schema::*;
pub mod bfloat16;
pub mod floats;
pub use floats::*;
pub mod cast;

type Result<T> = std::result::Result<T, ArrowError>;

pub trait DataTypeExt {
    /// Returns true if the data type is binary-like, such as (Large)Utf8 and (Large)Binary.
    ///
    /// ```
    /// use lance_arrow::*;
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

    fn byte_width(&self) -> usize;
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
                | Duration(_)
                | Timestamp(_, _)
                | Date32
                | Date64
                | Time32(_)
                | Time64(_)
        )
    }

    fn is_dictionary(&self) -> bool {
        matches!(self, Self::Dictionary(_, _))
    }

    fn byte_width(&self) -> usize {
        match self {
            Self::Int8 => 1,
            Self::Int16 => 2,
            Self::Int32 => 4,
            Self::Int64 => 8,
            Self::UInt8 => 1,
            Self::UInt16 => 2,
            Self::UInt32 => 4,
            Self::UInt64 => 8,
            Self::Float16 => 2,
            Self::Float32 => 4,
            Self::Float64 => 8,
            Self::Date32 => 4,
            Self::Date64 => 8,
            Self::Time32(_) => 4,
            Self::Time64(_) => 8,
            Self::Timestamp(_, _) => 8,
            Self::Duration(_) => 8,
            Self::Decimal128(_, _) => 16,
            Self::Decimal256(_, _) => 32,
            Self::Interval(unit) => match unit {
                IntervalUnit::YearMonth => 4,
                IntervalUnit::DayTime => 8,
                IntervalUnit::MonthDayNano => 16,
            },
            Self::FixedSizeBinary(s) => *s as usize,
            Self::FixedSizeList(dt, s) => *s as usize * dt.data_type().byte_width(),
            _ => panic!("Does not support get byte width on type {self}"),
        }
    }
}

/// Create an [`GenericListArray`] from values and offsets.
///
/// ```
/// use arrow_array::{Int32Array, Int64Array, ListArray};
/// use arrow_array::types::Int64Type;
/// use lance_arrow::try_new_generic_list_array;
///
/// let offsets = Int32Array::from_iter([0, 2, 7, 10]);
/// let int_values = Int64Array::from_iter(0..10);
/// let list_arr = try_new_generic_list_array(int_values, &offsets).unwrap();
/// assert_eq!(list_arr,
///     ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
///         Some(vec![Some(0), Some(1)]),
///         Some(vec![Some(2), Some(3), Some(4), Some(5), Some(6)]),
///         Some(vec![Some(7), Some(8), Some(9)]),
/// ]))
/// ```
pub fn try_new_generic_list_array<T: Array, Offset: ArrowNumericType>(
    values: T,
    offsets: &PrimitiveArray<Offset>,
) -> Result<GenericListArray<Offset::Native>>
where
    Offset::Native: OffsetSizeTrait,
{
    let data_type = if Offset::Native::IS_LARGE {
        DataType::LargeList(Arc::new(Field::new(
            "item",
            values.data_type().clone(),
            true,
        )))
    } else {
        DataType::List(Arc::new(Field::new(
            "item",
            values.data_type().clone(),
            true,
        )))
    };
    let data = ArrayDataBuilder::new(data_type)
        .len(offsets.len() - 1)
        .add_buffer(offsets.into_data().buffers()[0].clone())
        .add_child_data(values.into_data())
        .build()?;

    Ok(GenericListArray::from(data))
}

pub fn fixed_size_list_type(list_width: i32, inner_type: DataType) -> DataType {
    DataType::FixedSizeList(Arc::new(Field::new("item", inner_type, true)), list_width)
}

pub trait FixedSizeListArrayExt {
    /// Create an [`FixedSizeListArray`] from values and list size.
    ///
    /// ```
    /// use arrow_array::{Int64Array, FixedSizeListArray};
    /// use arrow_array::types::Int64Type;
    /// use lance_arrow::FixedSizeListArrayExt;
    ///
    /// let int_values = Int64Array::from_iter(0..10);
    /// let fixed_size_list_arr = FixedSizeListArray::try_new_from_values(int_values, 2).unwrap();
    /// assert_eq!(fixed_size_list_arr,
    ///     FixedSizeListArray::from_iter_primitive::<Int64Type, _, _>(vec![
    ///         Some(vec![Some(0), Some(1)]),
    ///         Some(vec![Some(2), Some(3)]),
    ///         Some(vec![Some(4), Some(5)]),
    ///         Some(vec![Some(6), Some(7)]),
    ///         Some(vec![Some(8), Some(9)])
    /// ], 2))
    /// ```
    fn try_new_from_values<T: Array + 'static>(
        values: T,
        list_size: i32,
    ) -> Result<FixedSizeListArray>;

    /// Sample `n` rows from the [FixedSizeListArray]
    ///
    /// ```
    /// use arrow_array::{Int64Array, FixedSizeListArray, Array};
    /// use lance_arrow::FixedSizeListArrayExt;
    ///
    /// let int_values = Int64Array::from_iter(0..256);
    /// let fixed_size_list_arr = FixedSizeListArray::try_new_from_values(int_values, 16).unwrap();
    /// let sampled = fixed_size_list_arr.sample(10).unwrap();
    /// assert_eq!(sampled.len(), 10);
    /// assert_eq!(sampled.value_length(), 16);
    /// assert_eq!(sampled.values().len(), 160);
    /// ```
    fn sample(&self, n: usize) -> Result<FixedSizeListArray>;

    /// Convert FixedSizeListArray content to floating type
    fn convert_to_floating_point (self) -> Result<FixedSizeListArray>;
}

impl FixedSizeListArrayExt for FixedSizeListArray {
    fn try_new_from_values<T: Array + 'static>(values: T, list_size: i32) -> Result<Self> {
        let field = Arc::new(Field::new("item", values.data_type().clone(), true));
        let values = Arc::new(values);

        Self::try_new(field, list_size, values, None)
    }

    fn sample(&self, n: usize) -> Result<FixedSizeListArray> {
        if n >= self.len() {
            return Ok(self.clone());
        }
        let mut rng = SmallRng::from_entropy();
        let chosen = (0..self.len() as u32).choose_multiple(&mut rng, n);
        take(self, &UInt32Array::from(chosen), None).map(|arr| arr.as_fixed_size_list().clone())
    }

    fn convert_to_floating_point (self) -> Result<FixedSizeListArray> {
        match self.data_type() {
            DataType::FixedSizeList(field, size) => {
                match field.data_type() {
                    DataType::Float16 | DataType::Float32 | DataType::Float64 => Ok(self),
                    DataType::Int8 => Ok(
                        FixedSizeListArray::new(
                            Arc::new(arrow_schema::Field::new(
                                field.name(), 
                                DataType::Float32, 
                                field.is_nullable())),
                            *size, 
                            Arc::new(Float32Array::from_iter_values(
                                self
                                .values()
                                .as_any()
                                .downcast_ref::<Int8Array>()
                                .ok_or(ArrowError::ParseError(format!("Fail to cast primitive array to Int8Type")))?
                                .into_iter()
                                .filter_map(|x| x.map(|y| y as f32)))), 
                            self
                            .nulls()
                            .map(|x| x.clone()))
                    ),
                    DataType::Int16 => Ok(
                        FixedSizeListArray::new(
                            Arc::new(arrow_schema::Field::new(
                                field.name(), 
                                DataType::Float32, 
                                field.is_nullable())),
                            *size, 
                            Arc::new(Float32Array::from_iter_values(
                                self
                                .values()
                                .as_any()
                                .downcast_ref::<Int16Array>()
                                .ok_or(ArrowError::ParseError(format!("Fail to cast primitive array to Int8Type")))?
                                .into_iter()
                                .filter_map(|x| x.map(|y| y as f32)))), 
                            self
                            .nulls()
                            .map(|x| x.clone()))
                    ),
                    DataType::Int32 => Ok(
                        FixedSizeListArray::new(
                            Arc::new(arrow_schema::Field::new(
                                field.name(), 
                                DataType::Float32, 
                                field.is_nullable())),
                            *size, 
                            Arc::new(Float32Array::from_iter_values(
                                self
                                .values()
                                .as_any()
                                .downcast_ref::<Int32Array>()
                                .ok_or(ArrowError::ParseError(format!("Fail to cast primitive array to Int8Type")))?
                                .into_iter()
                                .filter_map(|x| x.map(|y| y as f32)))), 
                            self
                            .nulls()
                            .map(|x| x.clone()))
                    ),
                    DataType::Int64 => Ok(
                        FixedSizeListArray::new(
                            Arc::new(arrow_schema::Field::new(
                                field.name(), 
                                DataType::Float64, 
                                field.is_nullable())),
                            *size, 
                            Arc::new(Float64Array::from_iter_values(
                                self
                                .values()
                                .as_any()
                                .downcast_ref::<Int64Array>()
                                .ok_or(ArrowError::ParseError(format!("Fail to cast primitive array to Int8Type")))?
                                .into_iter()
                                .filter_map(|x| x.map(|y| y as f64)))), 
                            self
                            .nulls()
                            .map(|x| x.clone()))
                    ),
                    data_type => Err(ArrowError::ParseError(format!("Expect either floating type or integer got {:?}", data_type)))
                }
            }
            data_type => Err(ArrowError::ParseError(format!("Expect either FixedSizeList got {:?}", data_type)))
        }
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
    /// use lance_arrow::FixedSizeBinaryArrayExt;
    ///
    /// let int_values = UInt8Array::from_iter(0..10);
    /// let fixed_size_list_arr = FixedSizeBinaryArray::try_new_from_values(&int_values, 2).unwrap();
    /// assert_eq!(fixed_size_list_arr,
    ///     FixedSizeBinaryArray::from(vec![
    ///         Some(vec![0, 1].as_slice()),
    ///         Some(vec![2, 3].as_slice()),
    ///         Some(vec![4, 5].as_slice()),
    ///         Some(vec![6, 7].as_slice()),
    ///         Some(vec![8, 9].as_slice())
    /// ]))
    /// ```
    fn try_new_from_values(values: &UInt8Array, stride: i32) -> Result<FixedSizeBinaryArray>;
}

impl FixedSizeBinaryArrayExt for FixedSizeBinaryArray {
    fn try_new_from_values(values: &UInt8Array, stride: i32) -> Result<Self> {
        let data_type = DataType::FixedSizeBinary(stride);
        let data = ArrayDataBuilder::new(data_type)
            .len(values.len() / stride as usize)
            .add_buffer(values.into_data().buffers()[0].clone())
            .build()?;
        Ok(Self::from(data))
    }
}

pub fn as_fixed_size_binary_array(arr: &dyn Array) -> &FixedSizeBinaryArray {
    arr.as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap()
}

/// Extends Arrow's [RecordBatch].
pub trait RecordBatchExt {
    /// Append a new column to this [`RecordBatch`] and returns a new RecordBatch.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use arrow_array::{RecordBatch, Int32Array, StringArray};
    /// use arrow_schema::{Schema, Field, DataType};
    /// use lance_arrow::*;
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

    /// Created a new RecordBatch with column at index.
    fn try_with_column_at(&self, index: usize, field: Field, arr: ArrayRef) -> Result<RecordBatch>;

    /// Creates a new [`RecordBatch`] from the provided  [`StructArray`].
    ///
    /// The fields on the [`StructArray`] need to match this [`RecordBatch`] schema
    fn try_new_from_struct_array(&self, arr: StructArray) -> Result<RecordBatch>;

    /// Merge with another [`RecordBatch`] and returns a new one.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use arrow_array::*;
    /// use arrow_schema::{Schema, Field, DataType};
    /// use lance_arrow::*;
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

    /// Replace a column (specified by name) and return the new [`RecordBatch`].
    fn replace_column_by_name(&self, name: &str, column: Arc<dyn Array>) -> Result<RecordBatch>;

    /// Get (potentially nested) column by qualified name.
    fn column_by_qualified_name(&self, name: &str) -> Option<&ArrayRef>;

    /// Project the schema over the [RecordBatch].
    fn project_by_schema(&self, schema: &Schema) -> Result<RecordBatch>;

    /// metadata of the schema.
    fn metadata(&self) -> &HashMap<String, String>;

    /// Add metadata to the schema.
    fn add_metadata(&self, key: String, value: String) -> Result<RecordBatch> {
        let mut metadata = self.metadata().clone();
        metadata.insert(key, value);
        self.with_metadata(metadata)
    }

    /// Replace the schema metadata with the provided one.
    fn with_metadata(&self, metadata: HashMap<String, String>) -> Result<RecordBatch>;

    /// Take selected rows from the [RecordBatch].
    fn take(&self, indices: &UInt32Array) -> Result<RecordBatch>;
}

impl RecordBatchExt for RecordBatch {
    fn try_with_column(&self, field: Field, arr: ArrayRef) -> Result<Self> {
        let new_schema = Arc::new(self.schema().as_ref().try_with_column(field)?);
        let mut new_columns = self.columns().to_vec();
        new_columns.push(arr);
        Self::try_new(new_schema, new_columns)
    }

    fn try_with_column_at(&self, index: usize, field: Field, arr: ArrayRef) -> Result<Self> {
        let new_schema = Arc::new(self.schema().as_ref().try_with_column_at(index, field)?);
        let mut new_columns = self.columns().to_vec();
        new_columns.insert(index, arr);
        Self::try_new(new_schema, new_columns)
    }

    fn try_new_from_struct_array(&self, arr: StructArray) -> Result<Self> {
        let schema = Arc::new(Schema::new_with_metadata(
            arr.fields().to_vec(),
            self.schema().metadata.clone(),
        ));
        let batch = Self::from(arr);
        batch.with_schema(schema)
    }

    fn merge(&self, other: &Self) -> Result<Self> {
        if self.num_rows() != other.num_rows() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Attempt to merge two RecordBatch with different sizes: {} != {}",
                self.num_rows(),
                other.num_rows()
            )));
        }
        let left_struct_array: StructArray = self.clone().into();
        let right_struct_array: StructArray = other.clone().into();
        self.try_new_from_struct_array(merge(&left_struct_array, &right_struct_array))
    }

    fn drop_column(&self, name: &str) -> Result<Self> {
        let mut fields = vec![];
        let mut columns = vec![];
        for i in 0..self.schema().fields.len() {
            if self.schema().field(i).name() != name {
                fields.push(self.schema().field(i).clone());
                columns.push(self.column(i).clone());
            }
        }
        Self::try_new(
            Arc::new(Schema::new_with_metadata(
                fields,
                self.schema().metadata().clone(),
            )),
            columns,
        )
    }

    fn replace_column_by_name(&self, name: &str, column: Arc<dyn Array>) -> Result<RecordBatch> {
        let mut columns = self.columns().to_vec();
        let field_i = self
            .schema()
            .fields()
            .iter()
            .position(|f| f.name() == name)
            .ok_or_else(|| ArrowError::SchemaError(format!("Field {} does not exist", name)))?;
        columns[field_i] = column;
        Self::try_new(self.schema().clone(), columns)
    }

    fn column_by_qualified_name(&self, name: &str) -> Option<&ArrayRef> {
        let split = name.split('.').collect::<Vec<_>>();
        if split.is_empty() {
            return None;
        }

        self.column_by_name(split[0])
            .and_then(|arr| get_sub_array(arr, &split[1..]))
    }

    fn project_by_schema(&self, schema: &Schema) -> Result<Self> {
        let struct_array: StructArray = self.clone().into();
        self.try_new_from_struct_array(project(&struct_array, schema.fields())?)
    }

    fn metadata(&self) -> &HashMap<String, String> {
        self.schema_ref().metadata()
    }

    fn with_metadata(&self, metadata: HashMap<String, String>) -> Result<RecordBatch> {
        let mut schema = self.schema_ref().as_ref().clone();
        schema.metadata = metadata;
        Self::try_new(schema.into(), self.columns().into())
    }

    fn take(&self, indices: &UInt32Array) -> Result<Self> {
        let struct_array: StructArray = self.clone().into();
        let taken = take(&struct_array, indices, None)?;
        self.try_new_from_struct_array(taken.as_struct().clone())
    }
}

fn project(struct_array: &StructArray, fields: &Fields) -> Result<StructArray> {
    if fields.is_empty() {
        return Ok(StructArray::new_empty_fields(
            struct_array.len(),
            struct_array.nulls().cloned(),
        ));
    }
    let mut columns: Vec<ArrayRef> = vec![];
    for field in fields.iter() {
        if let Some(col) = struct_array.column_by_name(field.name()) {
            match field.data_type() {
                // TODO handle list-of-struct
                DataType::Struct(subfields) => {
                    let projected = project(col.as_struct(), subfields)?;
                    columns.push(Arc::new(projected));
                }
                _ => {
                    columns.push(col.clone());
                }
            }
        } else {
            return Err(ArrowError::SchemaError(format!(
                "field {} does not exist in the RecordBatch",
                field.name()
            )));
        }
    }
    StructArray::try_new(fields.clone(), columns, None)
}

/// Merge the fields and columns of two RecordBatch's recursively
fn merge(left_struct_array: &StructArray, right_struct_array: &StructArray) -> StructArray {
    let mut fields: Vec<Field> = vec![];
    let mut columns: Vec<ArrayRef> = vec![];
    let right_fields = right_struct_array.fields();
    let right_columns = right_struct_array.columns();

    // iterate through the fields on the left hand side
    for (left_field, left_column) in left_struct_array
        .fields()
        .iter()
        .zip(left_struct_array.columns().iter())
    {
        match right_fields
            .iter()
            .position(|f| f.name() == left_field.name())
        {
            // if the field exists on the right hand side, merge them recursively if appropriate
            Some(right_index) => {
                let right_field = right_fields.get(right_index).unwrap();
                let right_column = right_columns.get(right_index).unwrap();
                // if both fields are struct, merge them recursively
                match (left_field.data_type(), right_field.data_type()) {
                    (DataType::Struct(_), DataType::Struct(_)) => {
                        let left_sub_array = left_column.as_struct();
                        let right_sub_array = right_column.as_struct();
                        let merged_sub_array = merge(left_sub_array, right_sub_array);
                        fields.push(Field::new(
                            left_field.name(),
                            merged_sub_array.data_type().clone(),
                            left_field.is_nullable(),
                        ));
                        columns.push(Arc::new(merged_sub_array) as ArrayRef);
                    }
                    // otherwise, just use the field on the left hand side
                    _ => {
                        // TODO handle list-of-struct and other types
                        fields.push(left_field.as_ref().clone());
                        columns.push(left_column.clone());
                    }
                }
            }
            None => {
                fields.push(left_field.as_ref().clone());
                columns.push(left_column.clone());
            }
        }
    }

    // now iterate through the fields on the right hand side
    right_fields
        .iter()
        .zip(right_columns.iter())
        .for_each(|(field, column)| {
            // add new columns on the right
            if !left_struct_array
                .fields()
                .iter()
                .any(|f| f.name() == field.name())
            {
                fields.push(field.as_ref().clone());
                columns.push(column.clone() as ArrayRef);
            }
        });

    let zipped: Vec<(FieldRef, ArrayRef)> = fields
        .iter()
        .cloned()
        .map(Arc::new)
        .zip(columns.iter().cloned())
        .collect::<Vec<_>>();
    StructArray::from(zipped)
}

fn get_sub_array<'a>(array: &'a ArrayRef, components: &[&str]) -> Option<&'a ArrayRef> {
    if components.is_empty() {
        return Some(array);
    }
    if !matches!(array.data_type(), DataType::Struct(_)) {
        return None;
    }
    let struct_arr = array.as_struct();
    struct_arr
        .column_by_name(components[0])
        .and_then(|arr| get_sub_array(arr, &components[1..]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};

    #[test]
    fn test_merge_recursive() {
        let a_array = Int32Array::from(vec![Some(1), Some(2), Some(3)]);
        let e_array = Int32Array::from(vec![Some(4), Some(5), Some(6)]);
        let c_array = Int32Array::from(vec![Some(7), Some(8), Some(9)]);
        let d_array = StringArray::from(vec![Some("a"), Some("b"), Some("c")]);

        let left_schema = Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new(
                "b",
                DataType::Struct(vec![Field::new("c", DataType::Int32, true)].into()),
                true,
            ),
        ]);
        let left_batch = RecordBatch::try_new(
            Arc::new(left_schema),
            vec![
                Arc::new(a_array.clone()),
                Arc::new(StructArray::from(vec![(
                    Arc::new(Field::new("c", DataType::Int32, true)),
                    Arc::new(c_array.clone()) as ArrayRef,
                )])),
            ],
        )
        .unwrap();

        let right_schema = Schema::new(vec![
            Field::new("e", DataType::Int32, true),
            Field::new(
                "b",
                DataType::Struct(vec![Field::new("d", DataType::Utf8, true)].into()),
                true,
            ),
        ]);
        let right_batch = RecordBatch::try_new(
            Arc::new(right_schema),
            vec![
                Arc::new(e_array.clone()),
                Arc::new(StructArray::from(vec![(
                    Arc::new(Field::new("d", DataType::Utf8, true)),
                    Arc::new(d_array.clone()) as ArrayRef,
                )])) as ArrayRef,
            ],
        )
        .unwrap();

        let merged_schema = Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new(
                "b",
                DataType::Struct(
                    vec![
                        Field::new("c", DataType::Int32, true),
                        Field::new("d", DataType::Utf8, true),
                    ]
                    .into(),
                ),
                true,
            ),
            Field::new("e", DataType::Int32, true),
        ]);
        let merged_batch = RecordBatch::try_new(
            Arc::new(merged_schema),
            vec![
                Arc::new(a_array) as ArrayRef,
                Arc::new(StructArray::from(vec![
                    (
                        Arc::new(Field::new("c", DataType::Int32, true)),
                        Arc::new(c_array) as ArrayRef,
                    ),
                    (
                        Arc::new(Field::new("d", DataType::Utf8, true)),
                        Arc::new(d_array) as ArrayRef,
                    ),
                ])) as ArrayRef,
                Arc::new(e_array) as ArrayRef,
            ],
        )
        .unwrap();

        let result = left_batch.merge(&right_batch).unwrap();
        assert_eq!(result, merged_batch);
    }

    #[test]
    fn test_take_record_batch() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..20)),
                Arc::new(StringArray::from_iter_values(
                    (0..20).map(|i| format!("str-{}", i)),
                )),
            ],
        )
        .unwrap();
        let taken = batch.take(&(vec![1_u32, 5_u32, 10_u32].into())).unwrap();
        assert_eq!(
            taken,
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int32Array::from(vec![1, 5, 10])),
                    Arc::new(StringArray::from(vec!["str-1", "str-5", "str-10"])),
                ],
            )
            .unwrap()
        )
    }

    #[test]
    fn test_schema_project_by_schema() {
        let metadata = [("key".to_string(), "value".to_string())];
        let schema = Arc::new(
            Schema::new(vec![
                Field::new("a", DataType::Int32, true),
                Field::new("b", DataType::Utf8, true),
            ])
            .with_metadata(metadata.clone().into()),
        );
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..20)),
                Arc::new(StringArray::from_iter_values(
                    (0..20).map(|i| format!("str-{}", i)),
                )),
            ],
        )
        .unwrap();

        // Empty schema
        let empty_schema = Schema::empty();
        let empty_projected = batch.project_by_schema(&empty_schema).unwrap();
        let expected_schema = empty_schema.with_metadata(metadata.clone().into());
        assert_eq!(
            empty_projected,
            RecordBatch::from(StructArray::new_empty_fields(batch.num_rows(), None))
                .with_schema(Arc::new(expected_schema))
                .unwrap()
        );

        // Re-ordered schema
        let reordered_schema = Schema::new(vec![
            Field::new("b", DataType::Utf8, true),
            Field::new("a", DataType::Int32, true),
        ]);
        let reordered_projected = batch.project_by_schema(&reordered_schema).unwrap();
        let expected_schema = Arc::new(reordered_schema.with_metadata(metadata.clone().into()));
        assert_eq!(
            reordered_projected,
            RecordBatch::try_new(
                expected_schema,
                vec![
                    Arc::new(StringArray::from_iter_values(
                        (0..20).map(|i| format!("str-{}", i)),
                    )),
                    Arc::new(Int32Array::from_iter_values(0..20)),
                ],
            )
            .unwrap()
        );

        // Sub schema
        let sub_schema = Schema::new(vec![Field::new("a", DataType::Int32, true)]);
        let sub_projected = batch.project_by_schema(&sub_schema).unwrap();
        let expected_schema = Arc::new(sub_schema.with_metadata(metadata.clone().into()));
        assert_eq!(
            sub_projected,
            RecordBatch::try_new(
                expected_schema,
                vec![Arc::new(Int32Array::from_iter_values(0..20))],
            )
            .unwrap()
        );
    }
}
