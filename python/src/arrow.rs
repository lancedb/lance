use std::sync::Arc;

use arrow::pyarrow::PyArrowConvert;
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use half::bf16;
use lance::arrow::bfloat16::BFloat16Array;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

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

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

const EXPORT_METADATA: [(&str, &str); 2] = [
    ("ARROW:extension:name", "lance.bfloat16"),
    ("ARROW:extension:metadata", ""),
];

#[pyfunction]
pub fn bfloat16_array(values: Vec<Option<f32>>, py: Python<'_>) -> PyResult<PyObject> {
    let array = BFloat16Array::from_iter(values.into_iter().map(|v| v.map(|v| bf16::from_f32(v))));

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
