// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::pyarrow::PyArrowType;
use arrow_schema::Schema as ArrowSchema;
use lance::datatypes::{Field, Schema};
use lance_file::datatypes::Fields;
use lance_file::format::pb;
use prost::Message;
use pyo3::{
    basic::CompareOp,
    exceptions::{PyNotImplementedError, PyValueError},
    prelude::*,
    types::PyTuple,
};

/// A Lance Schema.
///
/// This is valid for a particular dataset. It contains the field ids for each
/// column in the dataset.
#[pyclass(name = "LanceSchema", module = "lance.schema")]
#[derive(Clone)]
pub struct LanceSchema(pub Schema);

#[pymethods]
impl LanceSchema {
    pub fn __repr__(&self) -> PyResult<String> {
        // TODO: we should make a more succinct representation
        Ok(format!("{:?}", self.0))
    }

    pub fn __richcmp__(&self, other: LanceSchema, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.0 == other.0),
            CompareOp::Ne => Ok(self.0 != other.0),
            _ => Err(PyNotImplementedError::new_err(
                "Only == and != are supported",
            )),
        }
    }

    pub fn to_pyarrow(&self) -> PyArrowType<ArrowSchema> {
        PyArrowType(ArrowSchema::from(&self.0))
    }

    pub fn __reduce__(&self, py: Python<'_>) -> PyResult<(PyObject, PyObject)> {
        // Each field is a protobuf message
        let mut states = Vec::new();

        let fields = Fields::from(&self.0);
        for field in fields.0.iter() {
            states.push(field.encode_to_vec());
        }
        // The metadata is its own JSON object
        let metadata_str = serde_json::to_string(&self.0.metadata)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("{}", e)))?;
        states.push(metadata_str.as_bytes().to_vec());

        let state = PyTuple::new(py, states).extract()?;
        let from_json = PyModule::import(py, "lance.schema")?
            .getattr("LanceSchema")?
            .getattr("_from_protos")?
            .extract()?;
        Ok((from_json, state))
    }

    #[staticmethod]
    #[pyo3(signature = (*args))]
    pub fn _from_protos(mut args: Vec<Vec<u8>>) -> PyResult<Self> {
        let metadata = args
            .pop()
            .ok_or_else(|| PyValueError::new_err("Must have at least two arguments"))?;
        let metadata = serde_json::from_slice(&metadata)
            .map_err(|err| PyValueError::new_err(format!("Failed to parse metadata: {}", err)))?;

        let mut fields = Vec::new();
        for arg in args {
            let field = pb::Field::decode(arg.as_slice()).map_err(|e| {
                PyValueError::new_err(format!("Failed to parse field proto: {}", e))
            })?;
            fields.push(Field::from(&field));
        }
        let schema = Schema { fields, metadata };
        Ok(Self(schema))
    }
}
