// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use lance_io::object_store::StorageOptionsProvider;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::rt;

/// Internal wrapper for Python storage options providers
///
/// This is not exposed to Python. Users pass their Python objects directly
/// to dataset functions, and we wrap them internally with this struct.
pub struct PyStorageOptionsProvider {
    /// The Python object implementing get_storage_options()
    inner: PyObject,
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
        Python::with_gil(|py| Self {
            inner: self.inner.clone_ref(py),
        })
    }
}

impl PyStorageOptionsProvider {
    pub fn new(obj: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            // Verify the object has a fetch_storage_options method
            if !obj.bind(py).hasattr("fetch_storage_options")? {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "StorageOptionsProvider must implement fetch_storage_options() method",
                ));
            }
            Ok(Self { inner: obj })
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
                Python::with_gil(|py| {
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
        Python::with_gil(|py| {
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
    py_obj: PyObject,
) -> PyResult<Arc<dyn StorageOptionsProvider>> {
    let py_provider = PyStorageOptionsProvider::new(py_obj)?;
    Ok(Arc::new(PyStorageOptionsProviderWrapper::new(py_provider)))
}
