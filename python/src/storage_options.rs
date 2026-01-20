// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use lance_io::object_store::{StorageOptionsAccessor, StorageOptionsProvider};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::rt;

/// Internal wrapper for Python storage options providers
///
/// This is not exposed to Python. Users pass their Python objects directly
/// to dataset functions, and we wrap them internally with this struct.
pub struct PyStorageOptionsProvider {
    /// The Python object implementing get_storage_options()
    inner: Py<PyAny>,
}

impl std::fmt::Debug for PyStorageOptionsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Note: We can't call provider_id() here because this is PyStorageOptionsProvider,
        // not PyStorageOptionsProviderWrapper. Just use a simple format.
        write!(f, "PyStorageOptionsProvider")
    }
}

impl Clone for PyStorageOptionsProvider {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            inner: self.inner.clone_ref(py),
        })
    }
}

impl PyStorageOptionsProvider {
    pub fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Verify the object has a fetch_storage_options method
        if !obj.hasattr("fetch_storage_options")? {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "StorageOptionsProvider must implement fetch_storage_options() method",
            ));
        }
        Ok(Self {
            inner: obj.clone().unbind(),
        })
    }
}

/// Rust wrapper that implements StorageOptionsProvider trait for Python objects
pub struct PyStorageOptionsProviderWrapper {
    py_provider: PyStorageOptionsProvider,
}

impl std::fmt::Debug for PyStorageOptionsProviderWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.provider_id())
    }
}

impl std::fmt::Display for PyStorageOptionsProviderWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.provider_id())
    }
}

impl PyStorageOptionsProviderWrapper {
    pub fn new(py_provider: PyStorageOptionsProvider) -> Self {
        Self { py_provider }
    }
}

#[async_trait]
impl StorageOptionsProvider for PyStorageOptionsProviderWrapper {
    async fn fetch_storage_options(&self) -> lance_core::Result<Option<HashMap<String, String>>> {
        // Call Python method from async context
        let py_provider = self.py_provider.clone();

        rt().runtime
            .spawn_blocking(move || {
                Python::attach(|py| {
                    // Call the Python fetch_storage_options method
                    let result = py_provider
                        .inner
                        .bind(py)
                        .call_method0("fetch_storage_options")
                        .map_err(|e| lance_core::Error::IO {
                            source: Box::new(std::io::Error::other(format!(
                                "Failed to call fetch_storage_options: {}",
                                e
                            ))),
                            location: snafu::location!(),
                        })?;

                    // If result is None, return None
                    if result.is_none() {
                        return Ok(None);
                    }

                    // Extract the result dict - should be a flat Map<String, String>
                    let result_dict = result.downcast::<PyDict>().map_err(|_| {
                        lance_core::Error::InvalidInput {
                            source:
                                "fetch_storage_options() must return None or a dict of string key-value pairs"
                                    .into(),
                            location: snafu::location!(),
                        }
                    })?;

                    // Convert all entries to HashMap<String, String>
                    let mut storage_options = HashMap::new();
                    for (key, value) in result_dict.iter() {
                        let key_str: String =
                            key.extract().map_err(|e| lance_core::Error::InvalidInput {
                                source: format!("storage option keys must be strings: {}", e).into(),
                                location: snafu::location!(),
                            })?;
                        let value_str: String =
                            value
                                .extract()
                                .map_err(|e| lance_core::Error::InvalidInput {
                                    source: format!("storage option values must be strings: {}", e)
                                        .into(),
                                    location: snafu::location!(),
                                })?;
                        storage_options.insert(key_str, value_str);
                    }

                    Ok(Some(storage_options))
                })
            })
            .await
            .map_err(|e| lance_core::Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Failed to call Python fetch_storage_options: {}",
                    e
                ))),
                location: snafu::location!(),
            })?
    }

    fn provider_id(&self) -> String {
        Python::attach(|py| {
            // Call provider_id() method on the Python object
            // This should always succeed since StorageOptionsProvider.provider_id() has a default implementation
            let obj = self.py_provider.inner.bind(py);
            obj.call_method0("provider_id")
                .and_then(|result| result.extract::<String>())
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to call provider_id() on Python StorageOptionsProvider: {}",
                        e
                    )
                })
        })
    }
}

/// Convert a Python object to an Arc<dyn StorageOptionsProvider>
/// This is the main entry point for converting Python storage options providers to Rust
pub fn py_object_to_storage_options_provider(
    py_obj: &Bound<'_, PyAny>,
) -> PyResult<Arc<dyn StorageOptionsProvider>> {
    let py_provider = PyStorageOptionsProvider::new(py_obj)?;
    Ok(Arc::new(PyStorageOptionsProviderWrapper::new(py_provider)))
}

/// Python wrapper for StorageOptionsAccessor
///
/// This wraps a Rust StorageOptionsAccessor and exposes it to Python.
#[pyclass(name = "StorageOptionsAccessor")]
#[derive(Clone)]
pub struct PyStorageOptionsAccessor {
    inner: Arc<StorageOptionsAccessor>,
}

impl PyStorageOptionsAccessor {
    pub fn new(accessor: Arc<StorageOptionsAccessor>) -> Self {
        Self { inner: accessor }
    }

    pub fn inner(&self) -> Arc<StorageOptionsAccessor> {
        self.inner.clone()
    }
}

#[pymethods]
impl PyStorageOptionsAccessor {
    /// Create an accessor with only static options (no refresh capability)
    #[staticmethod]
    fn with_static_options(options: HashMap<String, String>) -> Self {
        Self {
            inner: Arc::new(StorageOptionsAccessor::with_static_options(options)),
        }
    }

    /// Create an accessor with a dynamic provider (no initial options)
    ///
    /// The refresh offset is extracted from storage options using the `refresh_offset_millis` key.
    #[staticmethod]
    fn with_provider(provider: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rust_provider = py_object_to_storage_options_provider(provider)?;
        Ok(Self {
            inner: Arc::new(StorageOptionsAccessor::with_provider(rust_provider)),
        })
    }

    /// Create an accessor with initial options and a dynamic provider
    ///
    /// The refresh offset is extracted from initial_options using the `refresh_offset_millis` key.
    #[staticmethod]
    fn with_initial_and_provider(
        initial_options: HashMap<String, String>,
        provider: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let rust_provider = py_object_to_storage_options_provider(provider)?;
        Ok(Self {
            inner: Arc::new(StorageOptionsAccessor::with_initial_and_provider(
                initial_options,
                rust_provider,
            )),
        })
    }

    /// Get current valid storage options
    fn get_storage_options(&self, py: Python<'_>) -> PyResult<HashMap<String, String>> {
        let accessor = self.inner.clone();
        let options = rt()
            .block_on(Some(py), accessor.get_storage_options())?
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(options.0)
    }

    /// Get the initial storage options without refresh
    fn initial_storage_options(&self) -> Option<HashMap<String, String>> {
        self.inner.initial_storage_options().cloned()
    }

    /// Get the accessor ID for equality/hashing
    fn accessor_id(&self) -> String {
        self.inner.accessor_id()
    }

    /// Check if this accessor has a dynamic provider
    fn has_provider(&self) -> bool {
        self.inner.has_provider()
    }

    /// Get the refresh offset in seconds
    fn refresh_offset_secs(&self) -> u64 {
        self.inner.refresh_offset().as_secs()
    }

    fn __repr__(&self) -> String {
        format!(
            "StorageOptionsAccessor(id={}, has_provider={})",
            self.inner.accessor_id(),
            self.inner.has_provider()
        )
    }
}

/// Create a StorageOptionsAccessor from Python parameters
///
/// This handles the conversion from Python types to Rust StorageOptionsAccessor.
/// The refresh offset is extracted from storage_options using the `refresh_offset_millis` key.
#[allow(dead_code)]
pub fn create_accessor_from_python(
    storage_options: Option<HashMap<String, String>>,
    storage_options_provider: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Arc<StorageOptionsAccessor>>> {
    match (storage_options, storage_options_provider) {
        (Some(opts), Some(provider)) => {
            let rust_provider = py_object_to_storage_options_provider(provider)?;
            Ok(Some(Arc::new(
                StorageOptionsAccessor::with_initial_and_provider(opts, rust_provider),
            )))
        }
        (None, Some(provider)) => {
            let rust_provider = py_object_to_storage_options_provider(provider)?;
            Ok(Some(Arc::new(StorageOptionsAccessor::with_provider(
                rust_provider,
            ))))
        }
        (Some(opts), None) => Ok(Some(Arc::new(StorageOptionsAccessor::with_static_options(
            opts,
        )))),
        (None, None) => Ok(None),
    }
}
