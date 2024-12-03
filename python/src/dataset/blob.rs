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

use std::sync::Arc;

use pyo3::{
    exceptions::PyValueError,
    pyclass, pymethods,
    types::{PyByteArray, PyByteArrayMethods, PyBytes},
    Bound, PyResult, Python,
};

use lance::dataset::BlobFile as InnerBlobFile;

use crate::{error::PythonErrorExt, RT};

#[pyclass]
pub struct LanceBlobFile {
    inner: Arc<InnerBlobFile>,
}

#[pymethods]
impl LanceBlobFile {
    pub fn close(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        RT.block_on(Some(py), inner.close())?.infer_error()
    }

    pub fn is_closed(&self, py: Python<'_>) -> PyResult<bool> {
        let inner = self.inner.clone();
        RT.block_on(Some(py), inner.is_closed())
    }

    pub fn seek(&self, py: Python<'_>, position: u64) -> PyResult<()> {
        let inner = self.inner.clone();
        RT.block_on(Some(py), inner.seek(position))?.infer_error()
    }

    pub fn tell(&self, py: Python<'_>) -> PyResult<u64> {
        let inner = self.inner.clone();
        RT.block_on(Some(py), inner.tell())?.infer_error()
    }

    pub fn size(&self) -> u64 {
        self.inner.size()
    }

    pub fn readall<'a>(&'a self, py: Python<'a>) -> PyResult<Bound<'a, PyBytes>> {
        let inner = self.inner.clone();
        let data = RT.block_on(Some(py), inner.read())?.infer_error()?;
        Ok(PyBytes::new_bound(py, &data))
    }

    pub fn read_into(&self, dst: Bound<'_, PyByteArray>) -> PyResult<usize> {
        let inner = self.inner.clone();

        let data = RT
            .block_on(Some(dst.py()), inner.read_up_to(dst.len()))?
            .infer_error()?;

        // Need to re-check length because the buffer could have been resized
        // by Python code while we were reading.
        if dst.len() < data.len() {
            Err(PyValueError::new_err("Buffer too small"))
        } else {
            // Safety: We've checked the buffer size above.  We've held the
            // GIL since then and so no other Python code could have modified
            // the buffer.
            unsafe {
                dst.as_bytes_mut()[0..data.len()].copy_from_slice(&data);
            }
            Ok(data.len())
        }
    }
}

impl From<InnerBlobFile> for LanceBlobFile {
    fn from(inner: InnerBlobFile) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}
