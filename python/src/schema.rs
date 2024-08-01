// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::pyarrow::PyArrowType;
use arrow_schema::Schema as ArrowSchema;
use lance::datatypes::Schema;
use lance_file::datatypes::{Fields, FieldsWithMeta};
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
/// Unlike a PyArrow schema, a Lance schema assigns every field an integer id.
/// This is used to track fields across versions. This assignment of fields to
/// ids is initially done in depth-first order, but as a schema evolves the
/// assignment may change.
///
/// The assignment of field ids is particular to each dataset, so these schemas
/// cannot be used interchangeably between datasets.
#[pyclass(name = "LanceSchema", module = "lance.schema")]
#[derive(Clone)]
pub struct LanceSchema(pub Schema);

#[pymethods]
impl LanceSchema {
    pub fn __repr__(&self) -> PyResult<String> {
        // TODO: we should make a more succinct representation
        Ok(format!("{:?}", self.0))
    }

    pub fn __richcmp__(&self, other: Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.0 == other.0),
            CompareOp::Ne => Ok(self.0 != other.0),
            _ => Err(PyNotImplementedError::new_err(
                "Only == and != are supported",
            )),
        }
    }

    /// Convert the schema to a PyArrow schema.
    pub fn to_pyarrow(&self) -> PyArrowType<ArrowSchema> {
        PyArrowType(ArrowSchema::from(&self.0))
    }

    /// Create a Lance schema from a PyArrow schema.
    ///
    /// This will assign field ids in depth-first order. Be aware this may not
    /// match the correct schema for a particular table.
    #[staticmethod]
    pub fn from_pyarrow(schema: PyArrowType<ArrowSchema>) -> PyResult<Self> {
        let schema = Schema::try_from(&schema.0)
            .map_err(|err| PyValueError::new_err(format!("Failed to convert schema: {}", err)))?;
        Ok(Self(schema))
    }

    pub fn __reduce__(&self, py: Python<'_>) -> PyResult<(PyObject, PyObject)> {
        // We don't have a single message for the schema, just protobuf message
        // for a field. So, the state will be:
        // (metadata_json, field_protos...)
        let fields_with_meta = FieldsWithMeta::from(&self.0);

        let mut states = Vec::new();
        let metadata_str = serde_json::to_string(&fields_with_meta.metadata)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("{}", e)))?
            .into_py(py);
        states.push(metadata_str);

        for field in fields_with_meta.fields.0.iter() {
            states.push(field.encode_to_vec().into_py(py));
        }

        let state = PyTuple::new(py, states).extract()?;
        let from_protos = PyModule::import(py, "lance.schema")?
            .getattr("LanceSchema")?
            .getattr("_from_protos")?
            .extract()?;
        Ok((from_protos, state))
    }

    #[staticmethod]
    #[pyo3(signature = (metadata_json, *field_protos))]
    pub fn _from_protos(metadata_json: String, field_protos: Vec<Vec<u8>>) -> PyResult<Self> {
        let metadata = serde_json::from_str(&metadata_json)
            .map_err(|err| PyValueError::new_err(format!("Failed to parse metadata: {}", err)))?;

        let mut fields = Vec::new();
        for proto in field_protos {
            let field = pb::Field::decode(proto.as_slice()).map_err(|e| {
                PyValueError::new_err(format!("Failed to parse field proto: {}", e))
            })?;
            fields.push(field);
        }
        let fields_with_meta = FieldsWithMeta {
            fields: Fields(fields),
            metadata,
        };
        let schema = Schema::from(fields_with_meta);
        Ok(Self(schema))
    }
}
