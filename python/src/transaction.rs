// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::pyarrow::PyArrowType;
use arrow_schema::Schema as ArrowSchema;
use lance::dataset::transaction::{
    DataReplacementGroup, Operation, RewriteGroup, RewrittenIndex, Transaction,
};
use lance::datatypes::Schema;
use lance_table::format::{DataFile, Fragment, Index};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyNone};
use pyo3::{intern, prelude::*};
use pyo3::{Bound, FromPyObject, PyAny, PyObject, PyResult, Python, ToPyObject};
use uuid::Uuid;

use crate::schema::LanceSchema;
use crate::utils::{class_name, export_vec, extract_vec, PyLance};

impl FromPyObject<'_> for PyLance<DataReplacementGroup> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let fragment_id = ob.getattr("fragment_id")?.extract::<u64>()?;
        let new_file = &ob.getattr("new_file")?.extract::<PyLance<DataFile>>()?;

        Ok(Self(DataReplacementGroup(fragment_id, new_file.0.clone())))
    }
}

impl ToPyObject for PyLance<&DataReplacementGroup> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let namespace = py
            .import_bound(intern!(py, "lance"))
            .and_then(|module| module.getattr(intern!(py, "LanceOperation")))
            .expect("Failed to import LanceOperation namespace");

        let fragment_id = self.0 .0;
        let new_file = PyLance(&self.0 .1).to_object(py);

        let cls = namespace
            .getattr("DataReplacementGroup")
            .expect("Failed to get DataReplacementGroup class");
        cls.call1((fragment_id, new_file)).unwrap().to_object(py)
    }
}

impl FromPyObject<'_> for PyLance<Index> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let uuid = ob.get_item("uuid")?.extract()?;
        let name = ob.get_item("name")?.extract()?;
        let fields = ob.get_item("fields")?.extract()?;
        let dataset_version = ob.get_item("dataset_version")?.extract()?;

        let fragment_ids = ob.get_item("fragment_ids")?;
        let fragment_ids = fragment_ids.iter()?.map(|id| id?.extract::<u32>())
            .collect::<PyResult<Vec<u32>>>()?;
        let fragment_bitmap = Some(fragment_ids.into_iter().collect());
        Ok(Self(Index {
            uuid: Uuid::parse_str(uuid)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
            name,
            fields,
            dataset_version,
            fragment_bitmap,
            // TODO: we should use lance::dataset::Dataset::commit_existing_index once
            // we have a way to determine index details from an existing index.
            index_details: None,
        }))
    }
}

impl ToPyObject for PyLance<&Index> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let uuid = self.0.uuid.to_string().to_object(py);
        let name = self.0.name.to_object(py);
        let fields = export_vec(py, &self.0.fields).to_object(py);
        let dataset_version = self.0.dataset_version.to_object(py);
        let fragment_ids = match &self.0.fragment_bitmap {
            Some(bitmap) => bitmap.into_iter().collect::<Vec<_>>().to_object(py),
            None => PyNone::get_bound(py).to_object(py),
        };

        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("uuid", uuid).unwrap();
        kwargs.set_item("name", name).unwrap();
        kwargs.set_item("fields", fields).unwrap();
        kwargs.set_item("dataset_version", dataset_version).unwrap();
        kwargs.set_item("fragment_ids", fragment_ids).unwrap();
        kwargs.into()
    }
}

impl ToPyObject for PyLance<Index> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        PyLance(&self.0).to_object(py)
    }
}

impl FromPyObject<'_> for PyLance<Operation> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        match class_name(ob)? {
            "Overwrite" => {
                let schema = extract_schema(&ob.getattr("new_schema")?)?;

                let fragments = extract_vec(&ob.getattr("fragments")?)?;

                let op = Operation::Overwrite {
                    schema,
                    fragments,
                    config_upsert_values: None,
                };
                Ok(Self(op))
            }
            "Append" => {
                let fragments = extract_vec(&ob.getattr("fragments")?)?;
                let op = Operation::Append { fragments };
                Ok(Self(op))
            }
            "Delete" => {
                let updated_fragments = extract_vec(&ob.getattr("updated_fragments")?)?;
                let deleted_fragment_ids = ob.getattr("deleted_fragment_ids")?.extract()?;
                let predicate = ob.getattr("predicate")?.extract()?;

                let op = Operation::Delete {
                    updated_fragments,
                    deleted_fragment_ids,
                    predicate,
                };
                Ok(Self(op))
            }
            "Update" => {
                let removed_fragment_ids = ob.getattr("removed_fragment_ids")?.extract()?;

                let updated_fragments = extract_vec(&ob.getattr("updated_fragments")?)?;

                let new_fragments = extract_vec(&ob.getattr("new_fragments")?)?;

                let op = Operation::Update {
                    removed_fragment_ids,
                    updated_fragments,
                    new_fragments,
                };
                Ok(Self(op))
            }
            "Merge" => {
                let schema = extract_schema(&ob.getattr("schema")?)?;

                let fragments = ob
                    .getattr("fragments")?
                    .extract::<Vec<PyLance<Fragment>>>()?;
                let fragments = fragments.into_iter().map(|f| f.0).collect();

                let op = Operation::Merge { schema, fragments };
                Ok(Self(op))
            }
            "Restore" => {
                let version = ob.getattr("version")?.extract()?;
                let op = Operation::Restore { version };
                Ok(Self(op))
            }
            "Rewrite" => {
                let groups = extract_vec(&ob.getattr("groups")?)?;
                let rewritten_indices = extract_vec(&ob.getattr("rewritten_indices")?)?;
                let op = Operation::Rewrite {
                    groups,
                    rewritten_indices,
                };
                Ok(Self(op))
            }
            "CreateIndex" => {
                let removed_indices = extract_vec(&ob.getattr("removed_indices")?)?;
                let new_indices = extract_vec(&ob.getattr("new_indices")?)?;
                let op = Operation::CreateIndex {
                    removed_indices,
                    new_indices,
                };
                Ok(Self(op))
            }
            "DataReplacement" => {
                let replacements = extract_vec(&ob.getattr("replacements")?)?;

                let op = Operation::DataReplacement { replacements };

                Ok(Self(op))
            }
            unsupported => Err(PyValueError::new_err(format!(
                "Unsupported operation: {unsupported}",
            ))),
        }
    }
}

impl ToPyObject for PyLance<&Operation> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let namespace = py
            .import_bound(intern!(py, "lance"))
            .and_then(|module| module.getattr(intern!(py, "LanceOperation")))
            .expect("Failed to import LanceOperation namespace");

        match self.0 {
            Operation::Append { ref fragments } => {
                let fragments = export_vec(py, fragments.as_slice());
                let cls = namespace
                    .getattr("Append")
                    .expect("Failed to get Append class");
                cls.call1((fragments,)).unwrap().to_object(py)
            }
            Operation::Overwrite {
                ref fragments,
                ref schema,
                ..
            } => {
                let fragments_py = export_vec(py, fragments.as_slice());

                let schema_py = LanceSchema(schema.clone());

                let cls = namespace
                    .getattr("Overwrite")
                    .expect("Failed to get Overwrite class");

                cls.call1((schema_py, fragments_py))
                    .expect("Failed to create Overwrite instance")
                    .to_object(py)
            }
            Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
            } => {
                let removed_fragment_ids = removed_fragment_ids.to_object(py);
                let updated_fragments = export_vec(py, updated_fragments.as_slice());
                let new_fragments = export_vec(py, new_fragments.as_slice());
                let cls = namespace
                    .getattr("Update")
                    .expect("Failed to get Update class");
                cls.call1((removed_fragment_ids, updated_fragments, new_fragments))
                    .unwrap()
                    .to_object(py)
            }
            Operation::CreateIndex {
                removed_indices,
                new_indices,
            } => {
                let removed_indices = export_vec(py, removed_indices.as_slice());
                let new_indices = export_vec(py, new_indices.as_slice());
                let cls = namespace
                    .getattr("CreateIndex")
                    .expect("Failed to get CreateIndex class");
                cls.call1((removed_indices, new_indices))
                    .unwrap()
                    .to_object(py)
            }
            Operation::DataReplacement { replacements } => {
                let replacements = export_vec(py, replacements.as_slice());
                let cls = namespace
                    .getattr("DataReplacement")
                    .expect("Failed to get DataReplacement class");
                cls.call1((replacements,)).unwrap().to_object(py)
            }
            _ => todo!(),
        }
    }
}

impl FromPyObject<'_> for PyLance<Transaction> {
    fn extract_bound(ob: &pyo3::Bound<'_, PyAny>) -> PyResult<Self> {
        let read_version = ob.getattr("read_version")?.extract()?;
        let uuid = ob.getattr("uuid")?.extract()?;
        let operation = ob.getattr("operation")?.extract::<PyLance<Operation>>()?.0;
        let blobs_op = ob
            .getattr("blobs_op")?
            .extract::<Option<PyLance<Operation>>>()?
            .map(|op| op.0);
        Ok(Self(Transaction {
            read_version,
            uuid,
            operation,
            blobs_op,
            tag: None,
        }))
    }
}

impl ToPyObject for PyLance<&Transaction> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let namespace = py
            .import_bound(intern!(py, "lance"))
            .expect("Failed to import lance module");

        let read_version = self.0.read_version;
        let uuid = &self.0.uuid;
        let operation = PyLance(&self.0.operation).to_object(py);
        let blobs_op = self.0.blobs_op.as_ref().map(|op| PyLance(op).to_object(py));

        let cls = namespace
            .getattr("Transaction")
            .expect("Failed to get Transaction class");
        cls.call1((read_version, operation, uuid, blobs_op))
            .unwrap()
            .to_object(py)
    }
}

impl ToPyObject for PyLance<Transaction> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        PyLance(&self.0).to_object(py)
    }
}

impl FromPyObject<'_> for PyLance<RewriteGroup> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self(RewriteGroup {
            old_fragments: extract_vec(&ob.getattr("old_fragments")?)?,
            new_fragments: extract_vec(&ob.getattr("new_fragments")?)?,
        }))
    }
}

impl ToPyObject for PyLance<&RewriteGroup> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let cls = py
            .import_bound(intern!(py, "lance"))
            .and_then(|module| module.getattr(intern!(py, "LanceTransaction")))
            .and_then(|cls| cls.getattr(intern!(py, "RewriteGroup")))
            .expect("Failed to get RewriteGroup class");

        let old_fragments = export_vec(py, self.0.old_fragments.as_slice());
        let new_fragments = export_vec(py, self.0.new_fragments.as_slice());

        cls.call1((old_fragments, new_fragments))
            .unwrap()
            .to_object(py)
    }
}

impl FromPyObject<'_> for PyLance<RewrittenIndex> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let old_id: &str = ob.getattr("old_id")?.extract()?;
        let new_id: &str = ob.getattr("new_id")?.extract()?;
        let old_id = Uuid::parse_str(old_id)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse UUID: {}", e)))?;
        let new_id = Uuid::parse_str(new_id)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse UUID: {}", e)))?;
        Ok(Self(RewrittenIndex { old_id, new_id }))
    }
}

impl ToPyObject for PyLance<&RewrittenIndex> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let cls = py
            .import_bound(intern!(py, "lance"))
            .and_then(|module| module.getattr(intern!(py, "LanceTransaction")))
            .and_then(|cls| cls.getattr(intern!(py, "RewrittenIndex")))
            .expect("Failed to get RewrittenIndex class");

        let old_id = self.0.old_id.to_string();
        let new_id = self.0.new_id.to_string();
        cls.call1((old_id, new_id)).unwrap().to_object(py)
    }
}

fn extract_schema(schema: &Bound<'_, PyAny>) -> PyResult<Schema> {
    match schema.downcast::<LanceSchema>() {
        Ok(schema) => Ok(schema.borrow().0.clone()),
        Err(_) => {
            let arrow_schema = schema.extract::<PyArrowType<ArrowSchema>>()?.0;
            convert_schema(&arrow_schema)
        }
    }
}

fn convert_schema(arrow_schema: &ArrowSchema) -> PyResult<Schema> {
    // Note: the field ids here are wrong.
    Schema::try_from(arrow_schema).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to convert Arrow schema to Lance schema: {}",
            e
        ))
    })
}
