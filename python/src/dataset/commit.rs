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

use std::fmt::Debug;

use lance::{
    io::commit::{CommitError, CommitLease, CommitLock},
    Error,
};
use pyo3::prelude::*;

fn handle_error(py_err: PyErr) -> Error {
    Error::Internal {
        message: format!("Error from commit handler: {}", py_err),
    }
}

pub struct PyCommitLock {
    inner: PyObject,
}

impl PyCommitLock {
    pub fn new(inner: PyObject) -> Self {
        Self { inner }
    }
}

impl Debug for PyCommitLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let repr = Python::with_gil(|py| {
            self.inner
                .call_method0(py, "__repr__")?
                .extract::<String>(py)
        })
        .ok();
        f.debug_struct("PyCommitLock")
            .field("inner", &repr)
            .finish()
    }
}

#[async_trait::async_trait]
impl CommitLock for PyCommitLock {
    type Lease = PyCommitLease;

    async fn lock(&self, version: u64) -> Result<Self::Lease, CommitError> {
        let lease = Python::with_gil(|py| {
            let py_conflict_error = py
                .import("lance")
                .map_err(handle_error)?
                .getattr("commit")
                .map_err(handle_error)?
                .getattr("CommitConflictError")
                .map_err(handle_error)?
                .get_type();
            self.inner
                .call_method1(py, "lock", (version,))
                .map_err(|err| {
                    if err.is_instance(py, py_conflict_error) {
                        CommitError::CommitConflict
                    } else {
                        CommitError::OtherError(handle_error(err))
                    }
                })
        });
        Ok(PyCommitLease { inner: lease? })
    }
}

pub struct PyCommitLease {
    inner: PyObject,
}

#[async_trait::async_trait]
impl CommitLease for PyCommitLease {
    async fn release(&self, success: bool) -> Result<(), CommitError> {
        Python::with_gil(|py| {
            self.inner
                .call_method1(py, "release", (success,))
                .map_err(handle_error)
        })?;
        Ok(())
    }
}
