// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::time::Duration;

use chrono::{DateTime, Utc};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyAny};

use crate::{error::PythonErrorExt, rt};

/// Python wrapper for a renewable dataset version lease.
#[pyclass(name = "VersionLease", module = "_lib", skip_from_py_object)]
pub struct PyVersionLease {
    inner: Option<lance::dataset::VersionLease>,
}

impl PyVersionLease {
    pub fn new(inner: lance::dataset::VersionLease) -> Self {
        Self { inner: Some(inner) }
    }

    pub(crate) fn ttl(ttl_micros: i64) -> PyResult<Duration> {
        let ttl_micros = u64::try_from(ttl_micros).map_err(|_| {
            PyValueError::new_err(format!(
                "version lease TTL must be greater than zero, got {ttl_micros} microseconds"
            ))
        })?;
        if ttl_micros == 0 {
            return Err(PyValueError::new_err(
                "version lease TTL must be greater than zero, got 0 microseconds",
            ));
        }
        Ok(Duration::from_micros(ttl_micros))
    }

    fn inner(&self) -> PyResult<&lance::dataset::VersionLease> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("version lease has been released"))
    }
}

#[pymethods]
impl PyVersionLease {
    /// The dataset version protected by this lease.
    #[getter]
    fn version(&self) -> PyResult<u64> {
        Ok(self.inner()?.version())
    }

    /// The time after which cleanup may remove the protected version.
    #[getter]
    fn expires_at(&self) -> PyResult<DateTime<Utc>> {
        Ok(self.inner()?.expires_at())
    }

    /// Renew this lease for the given duration from now.
    fn renew(&mut self, py: Python<'_>, ttl: Duration) -> PyResult<()> {
        if ttl.is_zero() {
            return Err(PyValueError::new_err(
                "version lease TTL must be greater than zero",
            ));
        }
        let lease = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("version lease has been released"))?;
        rt().block_on(Some(py), lease.renew(ttl))?.infer_error()?;
        Ok(())
    }

    /// Release this lease before its TTL expires.
    fn release(&mut self, py: Python<'_>) -> PyResult<()> {
        if let Some(lease) = self.inner.take() {
            rt().block_on(Some(py), lease.release())?.infer_error()?;
        }
        Ok(())
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        slf.release(py)?;
        Ok(false)
    }
}
