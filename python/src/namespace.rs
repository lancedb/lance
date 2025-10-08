// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use lance_namespace::LanceNamespace;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::RT;

/// Python wrapper for LanceNamespace implementations
#[pyclass(name = "_Namespace", module = "_lib")]
pub struct PyNamespace {
    inner: Arc<dyn LanceNamespace>,
}

impl PyNamespace {
    pub fn new(inner: Arc<dyn LanceNamespace>) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> Arc<dyn LanceNamespace> {
        self.inner.clone()
    }
}

#[pymethods]
impl PyNamespace {
    fn __repr__(&self) -> String {
        "Namespace".to_string()
    }

    /// Describe a table and get its metadata including storage credentials
    ///
    /// Args:
    ///     table_id: List of strings representing the table identifier
    ///     version: Optional version number
    ///
    /// Returns:
    ///     dict with table metadata including storage_options
    fn describe_table(
        &self,
        py: Python,
        table_id: Vec<String>,
        version: Option<u64>,
    ) -> PyResult<PyObject> {
        use lance_namespace::models::DescribeTableRequest;

        let namespace = self.inner.clone();
        let request = DescribeTableRequest {
            id: Some(table_id),
            version: version.map(|v| v as i64),
        };

        let response = RT.block_on(Some(py), async move {
            namespace
                .describe_table(request)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to describe table: {}",
                        e
                    ))
                })
        })??;

        // Convert response to Python dict
        let result = PyDict::new(py);

        if let Some(location) = response.location {
            result.set_item("location", location)?;
        }

        if let Some(version) = response.version {
            result.set_item("version", version as i64)?;
        }

        if let Some(storage_options) = response.storage_options {
            let py_storage_options = PyDict::new(py);
            for (k, v) in storage_options {
                py_storage_options.set_item(k, v)?;
            }
            result.set_item("storage_options", py_storage_options)?;
        }

        Ok(result.into())
    }
}

/// Connect to a namespace implementation
///
/// Args:
///     impl_name: Implementation name ("dir" for directory-based)
///     properties: Dictionary of configuration properties
///
/// Returns:
///     Namespace instance
#[pyfunction]
pub fn connect_namespace(
    py: Python,
    impl_name: String,
    properties: HashMap<String, String>,
) -> PyResult<PyNamespace> {
    // Import lance-namespace-impls connect function
    use lance_namespace_impls::connect;

    let namespace = RT.block_on(Some(py), async move {
        connect(&impl_name, properties)
            .await
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to connect to namespace: {}",
                    e
                ))
            })
    })??;

    Ok(PyNamespace::new(namespace))
}
