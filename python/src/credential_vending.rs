// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use lance_io::object_store::CredentialVendor;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::rt;

/// Internal wrapper for Python credential vendors
///
/// This is not exposed to Python. Users pass their Python objects directly
/// to dataset functions, and we wrap them internally with this struct.
#[derive(Clone)]
pub struct PyCredentialVendor {
    /// The Python object implementing get_credentials()
    inner: PyObject,
}

impl PyCredentialVendor {
    pub fn new(obj: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            // Verify the object has a get_credentials method
            if !obj.bind(py).hasattr("get_credentials")? {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "CredentialVendor must implement get_credentials() method",
                ));
            }
            Ok(Self { inner: obj })
        })
    }
}

/// Rust wrapper that implements CredentialVendor trait for Python objects
pub struct PyCredentialVendorWrapper {
    py_vendor: PyCredentialVendor,
}

impl PyCredentialVendorWrapper {
    pub fn new(py_vendor: PyCredentialVendor) -> Self {
        Self { py_vendor }
    }
}

#[async_trait]
impl CredentialVendor for PyCredentialVendorWrapper {
    async fn get_credentials(
        &self,
    ) -> lance_core::Result<(HashMap<String, String>, u64)> {
        // Call Python method from async context
        let py_vendor = self.py_vendor.clone();

        rt().runtime.spawn_blocking(move || {
            Python::with_gil(|py| {
                // Call the Python get_credentials method
                let result = py_vendor
                    .inner
                    .bind(py)
                    .call_method0("get_credentials")
                    .map_err(|e| lance_core::Error::IO {
                        source: Box::new(std::io::Error::other(format!(
                            "Failed to call get_credentials: {}",
                            e
                        ))),
                        location: snafu::location!(),
                    })?;

                // Extract the result dict
                let result_dict = result.downcast::<PyDict>().map_err(|_| {
                    lance_core::Error::InvalidInput {
                        source: "get_credentials() must return a dict with 'storage_options' and 'expires_at_millis' keys".into(),
                        location: snafu::location!(),
                    }
                })?;

                // Extract storage_options
                let storage_options_obj = result_dict
                    .get_item("storage_options")
                    .map_err(|e| lance_core::Error::InvalidInput {
                        source: format!("Failed to get storage_options from result: {}", e).into(),
                        location: snafu::location!(),
                    })?
                    .ok_or_else(|| lance_core::Error::InvalidInput {
                        source: "get_credentials() result must contain 'storage_options' key".into(),
                        location: snafu::location!(),
                    })?;

                let storage_options_dict = storage_options_obj.downcast::<PyDict>().map_err(|_| {
                    lance_core::Error::InvalidInput {
                        source: "storage_options must be a dict".into(),
                        location: snafu::location!(),
                    }
                })?;

                let mut storage_options = HashMap::new();
                for (key, value) in storage_options_dict.iter() {
                    let key_str: String = key.extract().map_err(|e| lance_core::Error::InvalidInput {
                        source: format!("storage_options keys must be strings: {}", e).into(),
                        location: snafu::location!(),
                    })?;
                    let value_str: String = value.extract().map_err(|e| lance_core::Error::InvalidInput {
                        source: format!("storage_options values must be strings: {}", e).into(),
                        location: snafu::location!(),
                    })?;
                    storage_options.insert(key_str, value_str);
                }

                // Extract expires_at_millis
                let expires_at_millis_obj = result_dict
                    .get_item("expires_at_millis")
                    .map_err(|e| lance_core::Error::InvalidInput {
                        source: format!("Failed to get expires_at_millis from result: {}", e).into(),
                        location: snafu::location!(),
                    })?
                    .ok_or_else(|| lance_core::Error::InvalidInput {
                        source: "get_credentials() result must contain 'expires_at_millis' key".into(),
                        location: snafu::location!(),
                    })?;

                let expires_at_millis: u64 = expires_at_millis_obj.extract().map_err(|e| {
                    lance_core::Error::InvalidInput {
                        source: format!("expires_at_millis must be an integer: {}", e).into(),
                        location: snafu::location!(),
                    }
                })?;

                Ok((storage_options, expires_at_millis))
            })
        })
        .await
        .map_err(|e| lance_core::Error::IO {
            source: Box::new(std::io::Error::other(format!(
                "Failed to call Python get_credentials: {}",
                e
            ))),
            location: snafu::location!(),
        })?
    }
}

/// Convert a Python object to an Arc<dyn CredentialVendor>
/// This is the main entry point for converting Python credential vendors to Rust
pub fn py_object_to_credential_vendor(
    py_obj: PyObject,
) -> PyResult<Arc<dyn CredentialVendor>> {
    let py_vendor = PyCredentialVendor::new(py_obj)?;
    Ok(Arc::new(PyCredentialVendorWrapper::new(py_vendor)))
}
