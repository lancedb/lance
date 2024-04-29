use lance::dataset::blob::Blob as LanceBlob;
use lance_io::traits::Reader;

use pyo3::{exceptions::PyValueError, pyclass, pymethods, types::PyBytes, PyRef, PyResult};

use crate::{error::PythonErrorExt, RT};

#[pyclass]
pub struct BlobReader {
    inner: Option<LanceBlob>,
}

impl BlobReader {
    pub fn new(inner: LanceBlob) -> Self {
        Self { inner: Some(inner) }
    }

    fn inner(&self) -> PyResult<&LanceBlob> {
        self.inner.as_ref().ok_or(PyValueError::new_err(
            "BlobReader was used after being closed",
        ))
    }
}

#[pymethods]
impl BlobReader {
    pub fn close(&mut self) {
        self.inner = None;
    }

    pub fn is_closed(&self) -> bool {
        self.inner.is_none()
    }

    pub fn read_range(self_: PyRef<'_, Self>, start: u64, end: u64) -> PyResult<&PyBytes> {
        RT.runtime
            .block_on(self_.inner()?.get_range(start as usize..end as usize))
            .infer_error()
            // Forces a copy :(, could add pyo3 support for bytes maybe?
            .map(|bytes| PyBytes::new(self_.py(), &bytes))
    }

    pub fn size(&self) -> PyResult<u64> {
        Ok(self.inner()?.size_bytes())
    }

    pub fn block_size(&self) -> PyResult<usize> {
        Ok(self.inner()?.block_size())
    }
}
