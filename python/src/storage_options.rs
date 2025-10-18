// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use lance_io::object_store::StorageOptionsProvider;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::rt;

/// Internal wrapper for Python credential vendors
///
/// This is not exposed to Python. Users pass their Python objects directly
/// to dataset functions, and we wrap them internally with this struct.
pub struct PyStorageOptionsProvider {
    /// The Python object implementing get_storage_options()
    inner: PyObject,
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
            // Verify the object has a get_storage_options method
            if !obj.bind(py).hasattr("get_storage_options")? {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "StorageOptionsProvider must implement get_storage_options() method",
                ));
            }
            Ok(Self { inner: obj })
        })
    }
}

/// Rust wrapper that implements StorageOptionsProvider trait for Python objects
pub struct PyStorageOptionsProviderWrapper {
    py_vendor: PyStorageOptionsProvider,
}

impl PyStorageOptionsProviderWrapper {
    pub fn new(py_vendor: PyStorageOptionsProvider) -> Self {
        Self { py_vendor }
    }
}

#[async_trait]
impl StorageOptionsProvider for PyStorageOptionsProviderWrapper {
    async fn get_storage_options(&self) -> lance_core::Result<(HashMap<String, String>, u64)> {
        // Call Python method from async context
        let py_vendor = self.py_vendor.clone();

        rt().runtime
            .spawn_blocking(move || {
                Python::with_gil(|py| {
                    // Call the Python get_storage_options method
                    let result = py_vendor
                        .inner
                        .bind(py)
                        .call_method0("get_storage_options")
                        .map_err(|e| lance_core::Error::IO {
                            source: Box::new(std::io::Error::other(format!(
                                "Failed to call get_storage_options: {}",
                                e
                            ))),
                            location: snafu::location!(),
                        })?;

                    // Extract the result dict - should be a flat Map<String, String>
                    let result_dict = result.downcast::<PyDict>().map_err(|_| {
                        lance_core::Error::InvalidInput {
                            source:
                                "get_storage_options() must return a dict of string key-value pairs"
                                    .into(),
                            location: snafu::location!(),
                        }
                    })?;

                    // Convert all entries to HashMap<String, String>
                    let mut credentials = HashMap::new();
                    for (key, value) in result_dict.iter() {
                        let key_str: String =
                            key.extract().map_err(|e| lance_core::Error::InvalidInput {
                                source: format!("credential keys must be strings: {}", e).into(),
                                location: snafu::location!(),
                            })?;
                        let value_str: String =
                            value
                                .extract()
                                .map_err(|e| lance_core::Error::InvalidInput {
                                    source: format!("credential values must be strings: {}", e)
                                        .into(),
                                    location: snafu::location!(),
                                })?;
                        credentials.insert(key_str, value_str);
                    }

                    // Extract and parse expires_at_millis
                    let expires_at_millis_str =
                        credentials.get("expires_at_millis").ok_or_else(|| {
                            lance_core::Error::InvalidInput {
                                source:
                                    "get_storage_options() result must contain 'expires_at_millis' key"
                                        .into(),
                                location: snafu::location!(),
                            }
                        })?;

                    let expires_at_millis: u64 = expires_at_millis_str.parse().map_err(|e| {
                        lance_core::Error::InvalidInput {
                            source: format!(
                                "expires_at_millis must be a valid integer string: {}",
                                e
                            )
                            .into(),
                            location: snafu::location!(),
                        }
                    })?;

                    // Remove expires_at_millis from credentials map as it's returned separately
                    credentials.remove("expires_at_millis");

                    Ok((credentials, expires_at_millis))
                })
            })
            .await
            .map_err(|e| lance_core::Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Failed to call Python get_storage_options: {}",
                    e
                ))),
                location: snafu::location!(),
            })?
    }
}

/// Convert a Python object to an Arc<dyn StorageOptionsProvider>
/// This is the main entry point for converting Python credential vendors to Rust
pub fn py_object_to_storage_options_provider(py_obj: PyObject) -> PyResult<Arc<dyn StorageOptionsProvider>> {
    let py_vendor = PyStorageOptionsProvider::new(py_obj)?;
    Ok(Arc::new(PyStorageOptionsProviderWrapper::new(py_vendor)))
}
