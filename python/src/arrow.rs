// Copyright 2023 Lance Developers.
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

use std::sync::Arc;

use arrow::ffi_stream::{export_reader_into_raw, FFI_ArrowArrayStream};
use arrow::pyarrow::*;
use arrow_array::{RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::{DataType, Field, Schema};
use half::bf16;
use lance::arrow::bfloat16::BFloat16Array;
use pyo3::{
    exceptions::PyValueError,
    ffi::Py_uintptr_t,
    prelude::*,
    pyclass::CompareOp,
    types::{PyTuple, PyType},
};

#[pyclass]
pub struct BFloat16(bf16);

#[pymethods]
impl BFloat16 {
    #[new]
    fn new(value: f32) -> Self {
        Self(bf16::from_f32(value))
    }

    #[classmethod]
    fn from_bytes(_cls: &PyType, bytes: &[u8]) -> PyResult<Self> {
        if bytes.len() != 2 {
            PyValueError::new_err(format!(
                "BFloat16::from_bytes: expected 2 bytes, got {}",
                bytes.len()
            ));
        }
        Ok(Self(bf16::from_bits(u16::from_ne_bytes([
            bytes[0], bytes[1],
        ]))))
    }

    fn as_float(&self) -> f32 {
        self.0.to_f32()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.0))
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> bool {
        match op {
            CompareOp::Eq => self.0 == other.0,
            CompareOp::Ge => self.0 >= other.0,
            CompareOp::Le => self.0 <= other.0,
            CompareOp::Gt => self.0 > other.0,
            CompareOp::Lt => self.0 < other.0,
            CompareOp::Ne => self.0 != other.0,
        }
    }
}

const EXPORT_METADATA: [(&str, &str); 2] = [
    ("ARROW:extension:name", "lance.bfloat16"),
    ("ARROW:extension:metadata", ""),
];

#[pyfunction]
pub fn bfloat16_array(values: Vec<Option<f32>>, py: Python<'_>) -> PyResult<PyObject> {
    let array = BFloat16Array::from_iter(values.into_iter().map(|v| v.map(bf16::from_f32)));

    // Create a record batch with a single column and an annotated schema
    let field = Field::new("bfloat16", DataType::FixedSizeBinary(2), true).with_metadata(
        EXPORT_METADATA
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    );
    let schema = Schema::new(vec![field]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)])
        .map_err(|err| PyValueError::new_err(format!("Failed to build array: {}", err)))?;

    let pyarrow_batch = batch.to_pyarrow(py)?;
    pyarrow_batch.call_method1(py, "__getitem__", ("bfloat16",))
}

/// Temporary solution until arrow-rs 47.0.0 is released.
/// https://github.com/apache/arrow-rs/blob/878217b9e330b4f1ed13e798a214ea11fbeb2bbb/arrow/src/pyarrow.rs#L318
///
/// TODO: replace this method with `Box<RecordBatchReader>::into_pyarrow()`
/// once arrow-rs 47.0.0 is released.
pub fn reader_to_pyarrow(
    py: Python,
    reader: Box<dyn RecordBatchReader + Send>,
) -> PyResult<PyObject> {
    let mut stream = FFI_ArrowArrayStream::empty();
    unsafe { export_reader_into_raw(reader, &mut stream) };

    let stream_ptr = (&mut stream) as *mut FFI_ArrowArrayStream;
    let module = py.import("pyarrow")?;
    let class = module.getattr("RecordBatchReader")?;
    let args = PyTuple::new(py, [stream_ptr as Py_uintptr_t]);
    let reader = class.call_method1("_import_from_c", args)?;

    Ok(PyObject::from(reader))
}

/// Temporary solution until arrow-rs 47.0.0 is released.
/// https://github.com/apache/arrow-rs/pull/4806
///
/// TODO: replace this method once arrow-rs 47.0.0 is released.
pub fn record_batch_to_pyarrow(py: Python<'_>, batch: &RecordBatch) -> PyResult<PyObject> {
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema().clone());
    let reader: Box<dyn RecordBatchReader + Send> = Box::new(reader);
    let py_reader = reader_to_pyarrow(py, reader)?;
    py_reader.call_method0(py, "read_next_batch")
}
