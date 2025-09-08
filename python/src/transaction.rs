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
use pyo3::types::PySet;
use pyo3::{intern, prelude::*};
use pyo3::{Bound, FromPyObject, PyAny, PyResult, Python};
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::schema::LanceSchema;
use crate::utils::{class_name, export_vec, extract_vec, PyLance};

// Add Index bindings
impl FromPyObject<'_> for PyLance<Index> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let uuid = ob.getattr("uuid")?.to_string();
        let name = ob.getattr("name")?.extract()?;
        let fields = ob.getattr("fields")?.extract()?;
        let dataset_version = ob.getattr("dataset_version")?.extract()?;
        let index_version = ob.getattr("index_version")?.extract()?;
        let fragment_ids = ob.getattr("fragment_ids")?;
        let created_at = ob.getattr("created_at")?.extract()?;

        let fragment_ids_ref: &Bound<'_, PySet> = fragment_ids.downcast()?;
        let fragment_bitmap = Some(
            fragment_ids_ref
                .into_iter()
                .map(|id| id.extract::<u32>())
                .collect::<PyResult<RoaringBitmap>>()?,
        );
        let base_id: Option<u32> = ob
            .getattr("base_id")?
            .extract::<Option<i64>>()?
            .map(|id| id as u32);

        Ok(Self(Index {
            uuid: Uuid::parse_str(&uuid).map_err(|e| PyValueError::new_err(e.to_string()))?,
            name,
            fields,
            dataset_version,
            fragment_bitmap,
            index_details: None,
            index_version,
            created_at,
            base_id,
        }))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&Index> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let namespace = py
            .import(intern!(py, "lance"))
            .expect("Failed to import lance module");

        let uuid = self.0.uuid.to_string();
        let name = &self.0.name;
        let fields = &self.0.fields;
        let dataset_version = self.0.dataset_version;
        let index_version = self.0.index_version;
        let fragment_ids = self.0.fragment_bitmap.as_ref().map_or_else(
            || PySet::empty(py).unwrap(),
            |bitmap| {
                let set = PySet::empty(py).unwrap();
                for id in bitmap.iter() {
                    set.add(id).unwrap();
                }
                set
            },
        );
        let created_at = self.0.created_at;
        let base_id = self.0.base_id.map(|id| id as i64);

        let cls = namespace
            .getattr("Index")
            .expect("Failed to get Index class");
        cls.call1((
            uuid,
            name.clone(),
            fields.clone(),
            dataset_version,
            fragment_ids,
            index_version,
            created_at,
            base_id,
        ))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<Index> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        PyLance(&self.0).into_pyobject(py)
    }
}

impl FromPyObject<'_> for PyLance<DataReplacementGroup> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let fragment_id = ob.getattr("fragment_id")?.extract::<u64>()?;
        let new_file = &ob.getattr("new_file")?.extract::<PyLance<DataFile>>()?;

        Ok(Self(DataReplacementGroup(fragment_id, new_file.0.clone())))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&DataReplacementGroup> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let namespace = py
            .import(intern!(py, "lance"))
            .and_then(|module| module.getattr(intern!(py, "LanceOperation")))
            .expect("Failed to import LanceOperation namespace");

        let fragment_id = self.0 .0;
        let new_file = PyLance(&self.0 .1).into_pyobject(py)?;

        let cls = namespace
            .getattr("DataReplacementGroup")
            .expect("Failed to get DataReplacementGroup class");
        cls.call1((fragment_id, new_file))
    }
}

impl FromPyObject<'_> for PyLance<Operation> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        match class_name(ob)?.as_str() {
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

                let fields_modified = ob.getattr("fields_modified")?.extract()?;

                let op = Operation::Update {
                    removed_fragment_ids,
                    updated_fragments,
                    new_fragments,
                    fields_modified,
                    mem_wal_to_flush: None,
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
                    // TODO: pass frag_reuse_index when available
                    frag_reuse_index: None,
                };
                Ok(Self(op))
            }
            "CreateIndex" => {
                let new_indices_py = ob.getattr("new_indices")?;
                let removed_indices_py = ob.getattr("removed_indices")?;

                let new_indices = extract_vec(&new_indices_py)?;
                let removed_indices = extract_vec(&removed_indices_py)?;

                let op = Operation::CreateIndex {
                    new_indices,
                    removed_indices,
                };
                Ok(Self(op))
            }
            "DataReplacement" => {
                let replacements = extract_vec(&ob.getattr("replacements")?)?;

                let op = Operation::DataReplacement { replacements };

                Ok(Self(op))
            }
            "Project" => {
                let schema = extract_schema(&ob.getattr("schema")?)?;

                let op = Operation::Project { schema };
                Ok(Self(op))
            }
            unsupported => Err(PyValueError::new_err(format!(
                "Unsupported operation: {unsupported}",
            ))),
        }
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&Operation> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let namespace = py
            .import(intern!(py, "lance"))
            .and_then(|module| module.getattr(intern!(py, "LanceOperation")))
            .expect("Failed to import LanceOperation namespace");

        match self.0 {
            Operation::Append { ref fragments } => {
                let fragments = export_vec(py, fragments.as_slice())?;
                let cls = namespace
                    .getattr("Append")
                    .expect("Failed to get Append class");
                cls.call1((fragments,))
            }
            Operation::Overwrite {
                ref fragments,
                ref schema,
                ..
            } => {
                let fragments_py = export_vec(py, fragments.as_slice())?;

                let schema_py = LanceSchema(schema.clone());

                let cls = namespace
                    .getattr("Overwrite")
                    .expect("Failed to get Overwrite class");

                cls.call1((schema_py, fragments_py))
            }
            Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
                fields_modified,
                ..
            } => {
                let removed_fragment_ids = removed_fragment_ids.into_pyobject(py)?;
                let updated_fragments = export_vec(py, updated_fragments.as_slice())?;
                let new_fragments = export_vec(py, new_fragments.as_slice())?;
                let fields_modified = fields_modified.into_pyobject(py)?;
                let cls = namespace
                    .getattr("Update")
                    .expect("Failed to get Update class");
                cls.call1((
                    removed_fragment_ids,
                    updated_fragments,
                    new_fragments,
                    fields_modified,
                ))
            }
            Operation::DataReplacement { replacements } => {
                let replacements = export_vec(py, replacements.as_slice())?;
                let cls = namespace
                    .getattr("DataReplacement")
                    .expect("Failed to get DataReplacement class");
                cls.call1((replacements,))
            }
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                predicate,
            } => {
                let updated_fragments = export_vec(py, updated_fragments.as_slice())?;
                let deleted_fragment_ids = deleted_fragment_ids.into_pyobject(py)?;
                let cls = namespace
                    .getattr("Delete")
                    .expect("Failed to get Delete class");
                cls.call1((updated_fragments, deleted_fragment_ids, predicate))
            }
            Operation::Merge {
                ref fragments,
                ref schema,
            } => {
                let fragments_py = export_vec(py, fragments.as_slice())?;
                let schema_py = LanceSchema(schema.clone());
                let cls = namespace
                    .getattr("Merge")
                    .expect("Failed to get Merge class");
                cls.call1((fragments_py, schema_py))
            }
            Operation::Restore { version } => {
                let cls = namespace
                    .getattr("Restore")
                    .expect("Failed to get Restore class");
                cls.call1((version,))
            }
            Operation::Rewrite {
                ref groups,
                ref rewritten_indices,
                ..
            } => {
                let groups_py = export_vec(py, groups.as_slice())?;
                let rewritten_indices_py = export_vec(py, rewritten_indices.as_slice())?;
                let cls = namespace
                    .getattr("Rewrite")
                    .expect("Failed to get Rewrite class");
                cls.call1((groups_py, rewritten_indices_py))
            }
            Operation::CreateIndex {
                ref new_indices,
                ref removed_indices,
            } => {
                let new_indices_py = export_vec(py, new_indices.as_slice())?;
                let removed_indices_py = export_vec(py, removed_indices.as_slice())?;

                let cls = namespace
                    .getattr("CreateIndex")
                    .expect("Failed to get CreateIndex class");
                cls.call1((new_indices_py, removed_indices_py))
            }
            Operation::Project { ref schema } => {
                let schema_py = LanceSchema(schema.clone());
                let cls = namespace
                    .getattr("Project")
                    .expect("Failed to get Project class");
                cls.call1((schema_py,))
            }
            Operation::ReserveFragments { num_fragments } => {
                if let Ok(cls) = namespace.getattr("ReserveFragments") {
                    cls.call1((num_fragments,))
                } else {
                    let base_op = namespace.getattr("BaseOperation")?;
                    base_op.call0()
                }
            }
            Operation::UpdateConfig {
                ref upsert_values,
                ref delete_keys,
                ref schema_metadata,
                ref field_metadata,
            } => {
                if let Ok(cls) = namespace.getattr("UpdateConfig") {
                    cls.call1((upsert_values, delete_keys, schema_metadata, field_metadata))
                } else {
                    let base_op = namespace.getattr("BaseOperation")?;
                    base_op.call0()
                }
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
        let transaction_properties = ob
            .getattr("transaction_properties")?
            .extract::<Option<HashMap<String, String>>>()?
            .filter(|map| !map.is_empty())
            .map(Arc::new);
        Ok(Self(Transaction {
            read_version,
            uuid,
            operation,
            blobs_op,
            tag: None,
            transaction_properties,
        }))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&Transaction> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let namespace = py
            .import(intern!(py, "lance"))
            .expect("Failed to import lance module");

        let read_version = self.0.read_version;
        let uuid = &self.0.uuid;
        let operation = PyLance(&self.0.operation).into_pyobject(py)?;
        let blobs_op = self
            .0
            .blobs_op
            .as_ref()
            .map(|op| PyLance(op).into_pyobject(py))
            .transpose()?;

        let cls = namespace
            .getattr("Transaction")
            .expect("Failed to get Transaction class");

        let py_transaction = cls.call1((read_version, operation, uuid, blobs_op))?;

        if let Some(transaction_properties_arc) = &self.0.transaction_properties {
            let py_dict = transaction_properties_arc.as_ref().into_pyobject(py)?;
            py_transaction.setattr("transaction_properties", py_dict)?;
        }
        // Unwrap due to infallible
        Ok(py_transaction.into_pyobject(py).unwrap())
    }
}

impl<'py> IntoPyObject<'py> for PyLance<Transaction> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        PyLance(&self.0).into_pyobject(py)
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

impl<'py> IntoPyObject<'py> for PyLance<&RewriteGroup> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let cls = py
            .import(intern!(py, "lance"))
            .and_then(|module| module.getattr(intern!(py, "LanceTransaction")))
            .and_then(|cls| cls.getattr(intern!(py, "RewriteGroup")))
            .expect("Failed to get RewriteGroup class");

        let old_fragments = export_vec(py, self.0.old_fragments.as_slice())?;
        let new_fragments = export_vec(py, self.0.new_fragments.as_slice())?;

        cls.call1((old_fragments, new_fragments))
    }
}

impl FromPyObject<'_> for PyLance<RewrittenIndex> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let old_id: String = ob.getattr("old_id")?.extract()?;
        let new_id: String = ob.getattr("new_id")?.extract()?;
        let old_id = Uuid::parse_str(&old_id)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse UUID: {}", e)))?;
        let new_id = Uuid::parse_str(&new_id)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse UUID: {}", e)))?;
        let new_details_type_url: String = ob.getattr("new_details_type_url")?.extract()?;
        let new_details_value: Vec<u8> = ob.getattr("new_details_value")?.extract()?;
        let new_index_version: u32 = ob.getattr("new_index_version")?.extract()?;
        Ok(Self(RewrittenIndex {
            old_id,
            new_id,
            new_index_details: prost_types::Any {
                type_url: new_details_type_url,
                value: new_details_value,
            },
            new_index_version,
        }))
    }
}

impl<'py> IntoPyObject<'py> for PyLance<&RewrittenIndex> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let cls = py
            .import(intern!(py, "lance"))
            .and_then(|module| module.getattr(intern!(py, "LanceTransaction")))
            .and_then(|cls| cls.getattr(intern!(py, "RewrittenIndex")))
            .expect("Failed to get RewrittenIndex class");

        let old_id = self.0.old_id.to_string();
        let new_id = self.0.new_id.to_string();
        cls.call1((old_id, new_id))
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
