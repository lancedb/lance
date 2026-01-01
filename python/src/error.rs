// Copyright 2024 Lance Developers.
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

use lance_namespace::error::NamespaceError;
use pyo3::{
    exceptions::{PyIOError, PyNotImplementedError, PyRuntimeError, PyValueError},
    types::{PyAnyMethods, PyModule},
    BoundObject, PyErr, PyResult, Python,
};

use lance::Error as LanceError;

/// Try to convert a NamespaceError to the corresponding Python exception.
/// Returns the appropriate Python exception from lance_namespace.errors module.
fn namespace_error_to_pyerr(py: Python<'_>, ns_err: &NamespaceError) -> PyErr {
    let code = ns_err.code().as_u32();
    let message = ns_err.to_string();

    // Try to import the lance_namespace.errors module and use from_error_code
    match PyModule::import(py, "lance_namespace.errors") {
        Ok(module) => {
            match module.getattr("from_error_code") {
                Ok(from_error_code) => {
                    match from_error_code.call1((code, message.clone())) {
                        Ok(exc) => {
                            // Create a PyErr from the exception object
                            PyErr::from_value(exc.into_bound())
                        }
                        Err(_) => PyRuntimeError::new_err(format!(
                            "[NamespaceError code={}] {}",
                            code, message
                        )),
                    }
                }
                Err(_) => {
                    PyRuntimeError::new_err(format!("[NamespaceError code={}] {}", code, message))
                }
            }
        }
        Err(_) => {
            // lance_namespace module not available, use RuntimeError with code prefix
            PyRuntimeError::new_err(format!("[NamespaceError code={}] {}", code, message))
        }
    }
}

pub trait PythonErrorExt<T> {
    /// Convert to a python error based on the Lance error type
    fn infer_error(self) -> PyResult<T>;
    /// Convert to RuntimeError
    fn runtime_error(self) -> PyResult<T>;
    /// Convert to ValueError
    fn value_error(self) -> PyResult<T>;
    /// Convert to PyNotImplementedError
    fn not_implemented(self) -> PyResult<T>;
    /// Convert to PyIoError
    fn io_error(self) -> PyResult<T>;
}

impl<T> PythonErrorExt<T> for std::result::Result<T, LanceError> {
    fn infer_error(self) -> PyResult<T> {
        match &self {
            Ok(_) => Ok(self.unwrap()),
            Err(err) => match err {
                LanceError::InvalidInput { .. } => self.value_error(),
                LanceError::NotSupported { .. } => self.not_implemented(),
                LanceError::IO { .. } => self.io_error(),
                LanceError::NotFound { .. } => self.value_error(),
                LanceError::RefNotFound { .. } => self.value_error(),
                LanceError::VersionNotFound { .. } => self.value_error(),
                LanceError::Namespace { source, .. } => {
                    // Try to downcast to NamespaceError and convert to proper Python exception
                    if let Some(ns_err) = source.downcast_ref::<NamespaceError>() {
                        Python::with_gil(|py| Err(namespace_error_to_pyerr(py, ns_err)))
                    } else {
                        log::warn!(
                            "Failed to downcast NamespaceError source, falling back to runtime error. \
                             This may indicate a version mismatch. Source type: {:?}",
                            source
                        );
                        self.runtime_error()
                    }
                }
                _ => self.runtime_error(),
            },
        }
    }

    fn runtime_error(self) -> PyResult<T> {
        self.map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn value_error(self) -> PyResult<T> {
        self.map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn not_implemented(self) -> PyResult<T> {
        self.map_err(|err| PyNotImplementedError::new_err(err.to_string()))
    }

    fn io_error(self) -> PyResult<T> {
        self.map_err(|err| PyIOError::new_err(err.to_string()))
    }
}
