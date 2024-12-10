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

use lance_table::io::commit::{CommitError, CommitLease, CommitLock};
use snafu::{location, Location};

use lance_core::Error;

use pyo3::{exceptions::PyIOError, prelude::*};

lazy_static! {
    static ref PY_CONFLICT_ERROR: PyResult<PyObject> = {
        Python::with_gil(|py| {
            py.import_bound("lance")
                .and_then(|lance| lance.getattr("commit"))
                .and_then(|commit| commit.getattr("CommitConflictError"))
                .map(|error| error.to_object(py))
        })
    };
}

fn handle_error(py_err: PyErr, py: Python) -> CommitError {
    let conflict_err_type = match &*PY_CONFLICT_ERROR {
        Ok(err) => err.bind(py).get_type(),
        Err(import_error) => {
            return CommitError::OtherError(Error::Internal {
                message: format!("Error importing from pylance {}", import_error),
                location: location!(),
            })
        }
    };

    if py_err.is_instance_bound(py, &conflict_err_type) {
        CommitError::CommitConflict
    } else {
        CommitError::OtherError(Error::Internal {
            message: format!("Error from commit handler: {}", py_err),
            location: location!(),
        })
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
        let lease = Python::with_gil(|py| -> Result<_, CommitError> {
            let lease = self
                .inner
                .call1(py, (version,))
                .map_err(|err| handle_error(err, py))?;
            lease
                .call_method0(py, "__enter__")
                .map_err(|err| handle_error(err, py))?;
            Ok(lease)
        })?;
        Ok(PyCommitLease { inner: lease })
    }
}

pub struct PyCommitLease {
    inner: PyObject,
}

#[async_trait::async_trait]
impl CommitLease for PyCommitLease {
    async fn release(&self, success: bool) -> Result<(), CommitError> {
        Python::with_gil(|py| {
            if success {
                self.inner
                    .call_method1(py, "__exit__", (py.None(), py.None(), py.None()))
                    .map_err(|err| handle_error(err, py))
            } else {
                // If the commit failed, we pass up an exception to the
                // context manager.
                PyIOError::new_err("commit failed").restore(py);
                let args = py
                    .import_bound("sys")
                    .unwrap()
                    .getattr("exc_info")
                    .unwrap()
                    .call0()
                    .unwrap();
                self.inner
                    .call_method1(
                        py,
                        "__exit__",
                        (
                            args.get_item(0).unwrap(),
                            args.get_item(1).unwrap(),
                            args.get_item(2).unwrap(),
                        ),
                    )
                    .map_err(|err| handle_error(err, py))
            }
        })?;
        Ok(())
    }
}
