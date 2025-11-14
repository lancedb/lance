// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Python bindings for Lance Namespace implementations

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use lance_namespace_impls::DirectoryNamespaceBuilder;
#[cfg(feature = "rest")]
use lance_namespace_impls::RestNamespaceBuilder;
#[cfg(feature = "rest-adapter")]
use lance_namespace_impls::{ConnectBuilder, RestAdapter, RestAdapterConfig};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use pythonize::{depythonize, pythonize};

use crate::error::PythonErrorExt;
use crate::session::Session;

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
    /// Create a new DirectoryNamespace from properties
    #[new]
    #[pyo3(signature = (session = None, **properties))]
    fn new(
        session: Option<&Bound<'_, Session>>,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut props = HashMap::new();

        if let Some(dict) = properties {
            props = dict_to_hashmap(dict)?;
        }

        let session_arc = session.map(|s| s.borrow().inner.clone());

        let builder =
            DirectoryNamespaceBuilder::from_properties(props, session_arc).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to create DirectoryNamespace: {}",
                    e
                ))
            })?;

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

#[cfg(feature = "rest")]
/// Python wrapper for RestNamespace
#[pyclass(name = "PyRestNamespace", module = "lance.lance")]
pub struct PyRestNamespace {
    inner: Arc<dyn lance_namespace::LanceNamespace>,
}

#[cfg(feature = "rest")]
#[pymethods]
impl PyRestNamespace {
    /// Create a new RestNamespace from properties
    #[new]
    #[pyo3(signature = (**properties))]
    fn new(properties: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut props = HashMap::new();

        if let Some(dict) = properties {
            props = dict_to_hashmap(dict)?;
        }

        let builder = RestNamespaceBuilder::from_properties(props).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to create RestNamespace: {}",
                e
            ))
        })?;

        let namespace = builder.build();

        Ok(Self {
            inner: Arc::new(namespace),
        })
    }

    /// Get the namespace ID
    fn namespace_id(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("PyRestNamespace({})", self.namespace_id())
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

#[cfg(feature = "rest-adapter")]
/// Python wrapper for REST adapter server
#[pyclass(name = "PyRestAdapter", module = "lance.lance")]
pub struct PyRestAdapter {
    backend: Arc<dyn lance_namespace::LanceNamespace>,
    config: RestAdapterConfig,
}

#[cfg(feature = "rest-adapter")]
#[pymethods]
impl PyRestAdapter {
    /// Create a new REST adapter server with namespace configuration
    #[new]
    #[pyo3(signature = (namespace_impl, namespace_properties, session = None, host = "127.0.0.1".to_string(), port = 2333))]
    fn new(
        namespace_impl: String,
        namespace_properties: Option<&Bound<'_, PyDict>>,
        session: Option<&Bound<'_, Session>>,
        host: String,
        port: u16,
    ) -> PyResult<Self> {
        let mut props = HashMap::new();

        if let Some(dict) = namespace_properties {
            props = dict_to_hashmap(dict)?;
        }

        // Use ConnectBuilder to build namespace from impl and properties
        let mut builder = ConnectBuilder::new(namespace_impl);
        for (k, v) in props {
            builder = builder.property(k, v);
        }

        // Add session if provided
        if let Some(sess) = session {
            builder = builder.session(sess.borrow().inner.clone());
        }

        let backend = crate::rt()
            .block_on(None, builder.connect())?
            .infer_error()?;

        let config = RestAdapterConfig { host, port };

        Ok(Self { backend, config })
    }

    /// Start the REST server in the background
    fn serve(&mut self, py: Python) -> PyResult<()> {
        let adapter = RestAdapter::new(self.backend.clone(), self.config.clone());

        crate::rt().spawn_background(Some(py), async move {
            let _ = adapter.serve().await;
        });

        // Give server time to start
        py.allow_threads(|| {
            std::thread::sleep(std::time::Duration::from_millis(500));
        });

        Ok(())
    }

    /// Stop the REST server
    fn stop(&mut self) {
        // Server will be stopped when dropped
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        slf.stop();
        Ok(false)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyRestAdapter(host='{}', port={})",
            self.config.host, self.config.port
        )
    }
}
