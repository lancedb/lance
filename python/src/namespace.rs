// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Python bindings for Lance Namespace implementations

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use lance_namespace_impls::DirectoryNamespaceBuilder;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use pythonize::{depythonize, pythonize};

use crate::error::PythonErrorExt;

/// Convert Python dict to HashMap<String, String>
fn dict_to_hashmap(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, String>> {
    let mut map = HashMap::new();
    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;
        let value_str: String = value.extract()?;
        map.insert(key_str, value_str);
    }
    Ok(map)
}

/// Python wrapper for DirectoryNamespace
#[pyclass(name = "PyDirectoryNamespace", module = "lance.lance")]
pub struct PyDirectoryNamespace {
    inner: Arc<dyn lance_namespace::LanceNamespace>,
}

#[pymethods]
impl PyDirectoryNamespace {
    /// Create a new DirectoryNamespace
    #[new]
    #[pyo3(signature = (root, storage_options=None, manifest_enabled=None, dir_listing_enabled=None))]
    fn new(
        root: String,
        storage_options: Option<&Bound<'_, PyDict>>,
        manifest_enabled: Option<bool>,
        dir_listing_enabled: Option<bool>,
    ) -> PyResult<Self> {
        let mut builder = DirectoryNamespaceBuilder::new(root);

        if let Some(opts) = storage_options {
            let opts_map = dict_to_hashmap(opts)?;
            for (key, value) in opts_map {
                builder = builder.storage_option(key, value);
            }
        }

        if let Some(enabled) = manifest_enabled {
            builder = builder.manifest_enabled(enabled);
        }

        if let Some(enabled) = dir_listing_enabled {
            builder = builder.dir_listing_enabled(enabled);
        }

        let namespace = crate::rt().block_on(None, builder.build())?.infer_error()?;

        Ok(Self {
            inner: Arc::new(namespace),
        })
    }

    /// Get the namespace ID
    fn namespace_id(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("PyDirectoryNamespace({})", self.namespace_id())
    }

    // Namespace operations

    fn list_namespaces(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_namespaces(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    fn describe_namespace(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_namespace(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    fn create_namespace(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_namespace(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    fn drop_namespace(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.drop_namespace(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    fn namespace_exists(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let request = depythonize(request)?;
        crate::rt()
            .block_on(Some(py), self.inner.namespace_exists(request))?
            .infer_error()?;
        Ok(())
    }

    // Table operations

    fn list_tables(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_tables(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    fn describe_table(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_table(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    fn register_table(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.register_table(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    fn table_exists(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let request = depythonize(request)?;
        crate::rt()
            .block_on(Some(py), self.inner.table_exists(request))?
            .infer_error()?;
        Ok(())
    }

    fn drop_table(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.drop_table(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    fn deregister_table(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.deregister_table(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    fn create_table(
        &self,
        py: Python,
        request: &Bound<'_, PyAny>,
        request_data: &Bound<'_, PyBytes>,
    ) -> PyResult<PyObject> {
        let request = depythonize(request)?;
        let data = Bytes::copy_from_slice(request_data.as_bytes());
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_table(request, data))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    fn create_empty_table(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_empty_table(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }
}
