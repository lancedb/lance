// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::array::AsArray;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::{FromPyArrow, PyArrowType};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, LargeListArray, ListArray, MapArray, RecordBatch,
    RecordBatchReader, StructArray,
};
use arrow_schema::{ArrowError, DataType, Schema as ArrowSchema};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::rt;
use crate::scanner::Scanner;

pub(crate) fn convert_reader(reader: &Bound<PyAny>) -> PyResult<Box<dyn RecordBatchReader + Send>> {
    let py = reader.py();
    if reader.is_instance_of::<Scanner>() {
        let scanner: Scanner = reader.extract()?;
        Ok(Box::new(
            rt().spawn(Some(py), async move { scanner.to_reader().await })?
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
        ))
    } else {
        let inner = Box::new(ArrowArrayStreamReader::from_pyarrow_bound(reader)?);
        with_pyarrow_schema(reader, inner)
    }
}

struct SchemaOverrideReader {
    inner: Box<dyn RecordBatchReader + Send>,
    schema: Arc<ArrowSchema>,
}

impl Iterator for SchemaOverrideReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|batch| batch.and_then(|batch| restore_batch_schema(batch, self.schema.clone())))
    }
}

impl RecordBatchReader for SchemaOverrideReader {
    fn schema(&self) -> Arc<ArrowSchema> {
        self.schema.clone()
    }
}

fn with_pyarrow_schema(
    reader: &Bound<PyAny>,
    inner: Box<dyn RecordBatchReader + Send>,
) -> PyResult<Box<dyn RecordBatchReader + Send>> {
    let Ok(schema_obj) = reader.getattr("schema") else {
        return Ok(inner);
    };
    let expected_schema: PyArrowType<ArrowSchema> = schema_obj.extract()?;
    if !schema_has_sorted_map(&expected_schema.0) && inner.schema().as_ref() == &expected_schema.0 {
        Ok(inner)
    } else {
        Ok(Box::new(SchemaOverrideReader {
            inner,
            schema: Arc::new(expected_schema.0),
        }))
    }
}

fn schema_has_sorted_map(schema: &ArrowSchema) -> bool {
    schema
        .fields()
        .iter()
        .any(|field| data_type_has_sorted_map(field.data_type()))
}

fn data_type_has_sorted_map(data_type: &DataType) -> bool {
    match data_type {
        DataType::Map(field, keys_sorted) => {
            *keys_sorted || data_type_has_sorted_map(field.data_type())
        }
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| data_type_has_sorted_map(field.data_type())),
        DataType::List(field) | DataType::LargeList(field) | DataType::FixedSizeList(field, _) => {
            data_type_has_sorted_map(field.data_type())
        }
        _ => false,
    }
}

fn restore_batch_schema(
    batch: RecordBatch,
    expected_schema: Arc<ArrowSchema>,
) -> Result<RecordBatch, ArrowError> {
    let columns = batch
        .columns()
        .iter()
        .zip(expected_schema.fields())
        .map(|(array, field)| restore_array_data_type(array.clone(), field.data_type()))
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(expected_schema, columns)
}

fn restore_array_data_type(
    array: ArrayRef,
    expected_type: &DataType,
) -> Result<ArrayRef, ArrowError> {
    if array.data_type() == expected_type {
        return Ok(array);
    }

    match expected_type {
        DataType::Map(entries_field, keys_sorted) => {
            let map_array = array
                .as_any()
                .downcast_ref::<MapArray>()
                .ok_or_else(|| schema_mismatch(array.data_type(), expected_type))?;
            let entries = restore_array_data_type(
                Arc::new(map_array.entries().clone()),
                entries_field.data_type(),
            )?;
            let entries = entries
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| schema_mismatch(entries.data_type(), entries_field.data_type()))?
                .clone();
            Ok(Arc::new(MapArray::new(
                entries_field.clone(),
                map_array.offsets().clone(),
                entries,
                map_array.nulls().cloned(),
                *keys_sorted,
            )))
        }
        DataType::Struct(fields) => {
            let struct_array = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| schema_mismatch(array.data_type(), expected_type))?;
            let columns = struct_array
                .columns()
                .iter()
                .zip(fields)
                .map(|(array, field)| restore_array_data_type(array.clone(), field.data_type()))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Arc::new(StructArray::new(
                fields.clone(),
                columns,
                struct_array.nulls().cloned(),
            )))
        }
        DataType::List(field) => {
            let list_array = array.as_list::<i32>();
            let values = restore_array_data_type(list_array.values().clone(), field.data_type())?;
            Ok(Arc::new(ListArray::new(
                field.clone(),
                list_array.offsets().clone(),
                values,
                list_array.nulls().cloned(),
            )))
        }
        DataType::LargeList(field) => {
            let list_array = array.as_list::<i64>();
            let values = restore_array_data_type(list_array.values().clone(), field.data_type())?;
            Ok(Arc::new(LargeListArray::new(
                field.clone(),
                list_array.offsets().clone(),
                values,
                list_array.nulls().cloned(),
            )))
        }
        DataType::FixedSizeList(field, size) => {
            let list_array = array.as_fixed_size_list();
            let values = restore_array_data_type(list_array.values().clone(), field.data_type())?;
            Ok(Arc::new(FixedSizeListArray::new(
                field.clone(),
                *size,
                values,
                list_array.nulls().cloned(),
            )))
        }
        _ => Err(schema_mismatch(array.data_type(), expected_type)),
    }
}

fn schema_mismatch(actual: &DataType, expected: &DataType) -> ArrowError {
    ArrowError::SchemaError(format!(
        "PyArrow reader schema mismatch: expected {expected}, got {actual}"
    ))
}
