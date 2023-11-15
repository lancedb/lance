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

//! Internal APIs for Lance format.
//!
//! Python API binding that exposes various

use lance::dataset::transaction::Transaction;
use lance::io::commit::read_transaction_file;
use lance::io::ObjectStore;
use lance_core::format::Manifest;
use lance_core::io::reader::read_manifest;
use pyo3::prelude::*;

use crate::RT;

#[pyclass(name = "Manifest", module = "lance.internal")]
#[derive(Debug)]
pub struct PyManifest {
    manifest: Manifest,
}

#[pymethods]
impl PyManifest {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        RT.runtime.block_on(async {
            // the store created is always at the root of the file system
            let (store, _) = ObjectStore::from_uri(".").await.map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "Failed to create object store: {}",
                    e
                ))
            })?;

            let path = std::fs::canonicalize(path)?;

            let manifest = read_manifest(
                &store,
                &object_store::path::Path::from_filesystem_path(path).map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!(
                        "Failed to create object store: {}",
                        e
                    ))
                })?,
            )
            .await
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "Failed to create object store: {}",
                    e
                ))
            })?;

            Ok(Self { manifest })
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.manifest)
    }

    // TODO: compile protobuf def into python and allow converting this class into python protobuf message
}

#[pyclass(name = "Transaction", module = "lance.internal")]
#[derive(Debug)]
pub struct PyTransaction {
    txn: Transaction,
}

#[pymethods]
impl PyTransaction {
    #[new]
    fn new(path: String, txn_name: String) -> PyResult<Self> {
        RT.runtime.block_on(async {
            // the store created is always at the root of the file system
            let (store, _) = ObjectStore::from_uri(".").await.map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "Failed to create object store: {}",
                    e
                ))
            })?;

            let path = std::fs::canonicalize(path)?;
            let path = object_store::path::Path::from_filesystem_path(path).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "Failed to create object store: {}",
                    e
                ))
            })?;

            let txn = read_transaction_file(&store, &path, &txn_name)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!(
                        "Failed to create object store: {}",
                        e
                    ))
                })?;

            Ok(Self { txn })
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.txn)
    }

    // TODO: compile protobuf def into python and allow converting this class into python protobuf message

    fn operation(&self) -> String {
        format!("{:?}", self.txn.operation)
    }
}

pub fn register_internal_apis(py: Python, m: &PyModule) -> PyResult<()> {
    let internal = PyModule::new(py, "internal")?;
    // internal debugging APIs
    internal.add_class::<PyManifest>()?;
    internal.add_class::<PyTransaction>()?;
    m.add_submodule(internal)?;
    Ok(())
}
