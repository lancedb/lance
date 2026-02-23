// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Python bindings for Lance Namespace implementations

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use lance_namespace::models::{
    CreateTableVersionRequest, CreateTableVersionResponse, DescribeTableVersionRequest,
    DescribeTableVersionResponse, ListTableVersionsRequest, ListTableVersionsResponse,
};
use lance_namespace::LanceNamespace as LanceNamespaceTrait;
use lance_namespace_impls::RestNamespaceBuilder;
use lance_namespace_impls::{ConnectBuilder, RestAdapter, RestAdapterConfig, RestAdapterHandle};
use lance_namespace_impls::{DirectoryNamespaceBuilder, DynamicContextProvider, OperationInfo};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use pythonize::{depythonize, pythonize};

use crate::error::PythonErrorExt;
use crate::session::Session;

/// Python-implemented dynamic context provider.
///
/// Wraps a Python object that has a `provide_context(info: dict) -> dict` method.
/// For RestNamespace, context keys that start with `headers.` are converted to
/// HTTP headers by stripping the prefix.
pub struct PyDynamicContextProvider {
    provider: Py<PyAny>,
}

impl Clone for PyDynamicContextProvider {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            provider: self.provider.clone_ref(py),
        })
    }
}

impl PyDynamicContextProvider {
    /// Create a new Python context provider wrapper.
    pub fn new(provider: Py<PyAny>) -> Self {
        Self { provider }
    }
}

impl std::fmt::Debug for PyDynamicContextProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PyDynamicContextProvider")
    }
}

impl DynamicContextProvider for PyDynamicContextProvider {
    fn provide_context(&self, info: &OperationInfo) -> HashMap<String, String> {
        Python::attach(|py| {
            // Create Python dict for operation info
            let py_info = PyDict::new(py);
            if py_info.set_item("operation", &info.operation).is_err() {
                return HashMap::new();
            }
            if py_info.set_item("object_id", &info.object_id).is_err() {
                return HashMap::new();
            }

            // Call the provider's provide_context method
            let result = self
                .provider
                .call_method1(py, "provide_context", (py_info,));

            match result {
                Ok(headers_py) => {
                    // Convert Python dict to Rust HashMap
                    let bound_headers = headers_py.bind(py);
                    if let Ok(dict) = bound_headers.downcast::<PyDict>() {
                        dict_to_hashmap(dict).unwrap_or_default()
                    } else {
                        log::warn!("Context provider did not return a dict");
                        HashMap::new()
                    }
                }
                Err(e) => {
                    log::error!("Failed to call context provider: {}", e);
                    HashMap::new()
                }
            }
        })
    }
}

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
    pub(crate) inner: Arc<dyn lance_namespace::LanceNamespace>,
}

#[pymethods]
impl PyDirectoryNamespace {
    /// Create a new DirectoryNamespace from properties
    ///
    /// # Arguments
    ///
    /// * `session` - Optional Lance session for sharing storage connections
    /// * `context_provider` - Optional object with `provide_context(info: dict) -> dict` method
    ///   for providing dynamic per-request context
    /// * `**properties` - Namespace configuration properties
    #[new]
    #[pyo3(signature = (session = None, context_provider = None, **properties))]
    fn new(
        session: Option<&Bound<'_, Session>>,
        context_provider: Option<&Bound<'_, PyAny>>,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut props = HashMap::new();

        if let Some(dict) = properties {
            props = dict_to_hashmap(dict)?;
        }

        let session_arc = session.map(|s| s.borrow().inner.clone());

        let mut builder =
            DirectoryNamespaceBuilder::from_properties(props, session_arc).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to create DirectoryNamespace: {}",
                    e
                ))
            })?;

        // Add context provider if provided
        if let Some(provider) = context_provider {
            let py_provider = PyDynamicContextProvider::new(provider.clone().unbind());
            builder = builder.context_provider(Arc::new(py_provider));
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

    fn list_namespaces<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_namespaces(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    fn describe_namespace<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_namespace(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    fn create_namespace<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_namespace(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn drop_namespace<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.drop_namespace(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn namespace_exists(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let request = depythonize(request)?;
        crate::rt()
            .block_on(Some(py), self.inner.namespace_exists(request))?
            .infer_error()?;
        Ok(())
    }

    // Table operations

    fn list_tables<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_tables(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn describe_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn register_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.register_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn table_exists(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let request = depythonize(request)?;
        crate::rt()
            .block_on(Some(py), self.inner.table_exists(request))?
            .infer_error()?;
        Ok(())
    }

    fn drop_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.drop_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn deregister_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.deregister_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn create_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
        request_data: &Bound<'_, PyBytes>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let data = Bytes::copy_from_slice(request_data.as_bytes());
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_table(request, data))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[allow(deprecated)]
    fn create_empty_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_empty_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn declare_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.declare_table(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    // Table version operations

    fn list_table_versions<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_table_versions(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn create_table_version<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_table_version(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn describe_table_version<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_table_version(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn batch_delete_table_versions<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.batch_delete_table_versions(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

/// Python wrapper for RestNamespace
#[pyclass(name = "PyRestNamespace", module = "lance.lance")]
pub struct PyRestNamespace {
    pub(crate) inner: Arc<dyn lance_namespace::LanceNamespace>,
}

#[pymethods]
impl PyRestNamespace {
    /// Create a new RestNamespace from properties
    ///
    /// # Arguments
    ///
    /// * `context_provider` - Optional object with `provide_context(info: dict) -> dict` method
    ///   for providing dynamic per-request context. Context keys that start with `headers.`
    ///   are converted to HTTP headers by stripping the prefix. For example,
    ///   `{"headers.Authorization": "Bearer token"}` becomes the `Authorization` header.
    /// * `**properties` - Namespace configuration properties (uri, delimiter, header.*, etc.)
    #[new]
    #[pyo3(signature = (context_provider = None, **properties))]
    fn new(
        context_provider: Option<&Bound<'_, PyAny>>,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut props = HashMap::new();

        if let Some(dict) = properties {
            props = dict_to_hashmap(dict)?;
        }

        let mut builder = RestNamespaceBuilder::from_properties(props).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to create RestNamespace: {}",
                e
            ))
        })?;

        // Add context provider if provided
        if let Some(provider) = context_provider {
            let py_provider = PyDynamicContextProvider::new(provider.clone().unbind());
            builder = builder.context_provider(Arc::new(py_provider));
        }

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

    fn list_namespaces<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_namespaces(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn describe_namespace<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_namespace(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn create_namespace<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_namespace(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn drop_namespace<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.drop_namespace(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn namespace_exists(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let request = depythonize(request)?;
        crate::rt()
            .block_on(Some(py), self.inner.namespace_exists(request))?
            .infer_error()?;
        Ok(())
    }

    // Table operations

    fn list_tables<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_tables(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn describe_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn register_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.register_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn table_exists(&self, py: Python, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let request = depythonize(request)?;
        crate::rt()
            .block_on(Some(py), self.inner.table_exists(request))?
            .infer_error()?;
        Ok(())
    }

    fn drop_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.drop_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn deregister_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.deregister_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn create_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
        request_data: &Bound<'_, PyBytes>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let data = Bytes::copy_from_slice(request_data.as_bytes());
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_table(request, data))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[allow(deprecated)]
    fn create_empty_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_empty_table(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn declare_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.declare_table(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    fn rename_table<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.rename_table(request))?
            .infer_error()?;
        Ok(pythonize(py, &response)?.into())
    }

    // Table version operations

    fn list_table_versions<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.list_table_versions(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn create_table_version<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.create_table_version(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn describe_table_version<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.describe_table_version(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn batch_delete_table_versions<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request = depythonize(request)?;
        let response = crate::rt()
            .block_on(Some(py), self.inner.batch_delete_table_versions(request))?
            .infer_error()?;
        pythonize(py, &response).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

/// Wrapper that allows any Python object implementing LanceNamespace protocol
/// to be used as a Rust LanceNamespace.
///
/// This is similar to JavaLanceNamespace in the Java bindings - it wraps a Python
/// object and calls back into Python when namespace methods are invoked.
///
/// We use `Arc<Py<PyAny>>` instead of `Py<PyAny>` directly because cloning `Py`
/// requires the GIL, but cloning `Arc` does not. This allows us to pass the
/// namespace reference to `spawn_blocking` without holding the GIL.
pub struct PyLanceNamespace {
    py_namespace: Arc<Py<PyAny>>,
    namespace_id: String,
}

impl PyLanceNamespace {
    /// Create a new PyLanceNamespace wrapper around a Python namespace object.
    pub fn new(_py: Python<'_>, py_namespace: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Get the namespace_id by calling the Python method
        let namespace_id = py_namespace
            .call_method0("namespace_id")?
            .extract::<String>()?;

        Ok(Self {
            py_namespace: Arc::new(py_namespace.clone().unbind()),
            namespace_id,
        })
    }

    /// Create an Arc<dyn LanceNamespace> from a Python namespace object.
    pub fn create_arc(
        py: Python<'_>,
        py_namespace: &Bound<'_, PyAny>,
    ) -> PyResult<Arc<dyn LanceNamespaceTrait>> {
        let wrapper = Self::new(py, py_namespace)?;
        Ok(Arc::new(wrapper))
    }
}

impl std::fmt::Debug for PyLanceNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PyLanceNamespace {{ id: {} }}", self.namespace_id)
    }
}

#[async_trait]
impl LanceNamespaceTrait for PyLanceNamespace {
    fn namespace_id(&self) -> String {
        self.namespace_id.clone()
    }

    async fn describe_table_version(
        &self,
        request: DescribeTableVersionRequest,
    ) -> lance_core::Result<DescribeTableVersionResponse> {
        // Clone the Arc (doesn't need GIL) to pass to spawn_blocking
        let py_namespace = self.py_namespace.clone();
        let request_json = serde_json::to_string(&request).map_err(|e| lance_core::Error::IO {
            source: Box::new(std::io::Error::other(format!(
                "Failed to serialize request: {}",
                e
            ))),
            location: snafu::location!(),
        })?;

        let response_json = tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                let result =
                    py_namespace.call_method1(py, "describe_table_version_json", (request_json,));

                match result {
                    Ok(response_py) => {
                        let response_str: String =
                            response_py.extract(py).map_err(|e| lance_core::Error::IO {
                                source: Box::new(std::io::Error::other(format!(
                                    "Failed to extract response string: {}",
                                    e
                                ))),
                                location: snafu::location!(),
                            })?;
                        Ok(response_str)
                    }
                    Err(e) => Err(lance_core::Error::IO {
                        source: Box::new(std::io::Error::other(format!(
                            "Failed to call describe_table_version_json: {}",
                            e
                        ))),
                        location: snafu::location!(),
                    }),
                }
            })
        })
        .await
        .map_err(|e| lance_core::Error::IO {
            source: Box::new(std::io::Error::other(format!("Task join error: {}", e))),
            location: snafu::location!(),
        })??;

        serde_json::from_str(&response_json).map_err(|e| lance_core::Error::IO {
            source: Box::new(std::io::Error::other(format!(
                "Failed to deserialize response: {}",
                e
            ))),
            location: snafu::location!(),
        })
    }

    async fn create_table_version(
        &self,
        request: CreateTableVersionRequest,
    ) -> lance_core::Result<CreateTableVersionResponse> {
        // Clone the Arc (doesn't need GIL) to pass to spawn_blocking
        let py_namespace = self.py_namespace.clone();
        let request_json = serde_json::to_string(&request).map_err(|e| lance_core::Error::IO {
            source: Box::new(std::io::Error::other(format!(
                "Failed to serialize request: {}",
                e
            ))),
            location: snafu::location!(),
        })?;

        let response_json = tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                let result =
                    py_namespace.call_method1(py, "create_table_version_json", (request_json,));

                match result {
                    Ok(response_py) => {
                        let response_str: String =
                            response_py.extract(py).map_err(|e| lance_core::Error::IO {
                                source: Box::new(std::io::Error::other(format!(
                                    "Failed to extract response string: {}",
                                    e
                                ))),
                                location: snafu::location!(),
                            })?;
                        Ok(response_str)
                    }
                    Err(e) => Err(lance_core::Error::IO {
                        source: Box::new(std::io::Error::other(format!(
                            "Failed to call create_table_version_json: {}",
                            e
                        ))),
                        location: snafu::location!(),
                    }),
                }
            })
        })
        .await
        .map_err(|e| lance_core::Error::IO {
            source: Box::new(std::io::Error::other(format!("Task join error: {}", e))),
            location: snafu::location!(),
        })??;

        serde_json::from_str(&response_json).map_err(|e| lance_core::Error::IO {
            source: Box::new(std::io::Error::other(format!(
                "Failed to deserialize response: {}",
                e
            ))),
            location: snafu::location!(),
        })
    }

    async fn list_table_versions(
        &self,
        request: ListTableVersionsRequest,
    ) -> lance_core::Result<ListTableVersionsResponse> {
        // Clone the Arc (doesn't need GIL) to pass to spawn_blocking
        let py_namespace = self.py_namespace.clone();
        let request_json = serde_json::to_string(&request).map_err(|e| lance_core::Error::IO {
            source: Box::new(std::io::Error::other(format!(
                "Failed to serialize request: {}",
                e
            ))),
            location: snafu::location!(),
        })?;

        let response_json = tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                let result =
                    py_namespace.call_method1(py, "list_table_versions_json", (request_json,));

                match result {
                    Ok(response_py) => {
                        let response_str: String =
                            response_py.extract(py).map_err(|e| lance_core::Error::IO {
                                source: Box::new(std::io::Error::other(format!(
                                    "Failed to extract response string: {}",
                                    e
                                ))),
                                location: snafu::location!(),
                            })?;
                        Ok(response_str)
                    }
                    Err(e) => Err(lance_core::Error::IO {
                        source: Box::new(std::io::Error::other(format!(
                            "Failed to call list_table_versions_json: {}",
                            e
                        ))),
                        location: snafu::location!(),
                    }),
                }
            })
        })
        .await
        .map_err(|e| lance_core::Error::IO {
            source: Box::new(std::io::Error::other(format!("Task join error: {}", e))),
            location: snafu::location!(),
        })??;

        serde_json::from_str(&response_json).map_err(|e| lance_core::Error::IO {
            source: Box::new(std::io::Error::other(format!(
                "Failed to deserialize response: {}",
                e
            ))),
            location: snafu::location!(),
        })
    }
}

/// Python wrapper for REST adapter server
#[pyclass(name = "PyRestAdapter", module = "lance.lance")]
pub struct PyRestAdapter {
    backend: Arc<dyn lance_namespace::LanceNamespace>,
    config: RestAdapterConfig,
    handle: Option<RestAdapterHandle>,
}

#[pymethods]
impl PyRestAdapter {
    /// Create a new REST adapter server with namespace configuration.
    /// Default port is 2333 per REST spec. Use port 0 to let OS assign an ephemeral port.
    /// Use `port` property after `start()` to get the actual port.
    #[new]
    #[pyo3(signature = (namespace_impl, namespace_properties, session = None, host = None, port = None))]
    fn new(
        namespace_impl: String,
        namespace_properties: Option<&Bound<'_, PyDict>>,
        session: Option<&Bound<'_, Session>>,
        host: Option<String>,
        port: Option<u16>,
    ) -> PyResult<Self> {
        let mut props = HashMap::new();

        if let Some(dict) = namespace_properties {
            props = dict_to_hashmap(dict)?;
        }

        let mut builder = ConnectBuilder::new(namespace_impl);
        for (k, v) in props {
            builder = builder.property(k, v);
        }

        if let Some(sess) = session {
            builder = builder.session(sess.borrow().inner.clone());
        }

        let backend = crate::rt()
            .block_on(None, builder.connect())?
            .infer_error()?;

        let mut config = RestAdapterConfig::default();
        if let Some(h) = host {
            config.host = h;
        }
        if let Some(p) = port {
            config.port = p;
        }

        Ok(Self {
            backend,
            config,
            handle: None,
        })
    }

    /// Get the actual port the server is listening on.
    /// Returns 0 if server is not started yet.
    #[getter]
    fn port(&self) -> u16 {
        self.handle.as_ref().map(|h| h.port()).unwrap_or(0)
    }

    /// Start the REST server in the background
    fn start(&mut self, py: Python) -> PyResult<()> {
        let adapter = RestAdapter::new(self.backend.clone(), self.config.clone());
        let handle = crate::rt()
            .block_on(Some(py), adapter.start())?
            .infer_error()?;

        self.handle = Some(handle);
        Ok(())
    }

    /// Stop the REST server
    fn stop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.shutdown();
        }
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
