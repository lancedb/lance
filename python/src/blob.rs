// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::file::object_store_from_uri_or_path_with_provider;
use crate::namespace::extract_namespace_arc;
use crate::{error::PythonErrorExt, rt};
use arrow::pyarrow::ToPyArrow;
use bytes::Bytes;
use lance::{
    BlobRange, DedicatedBlobWriter, LanceBlobSession, LanceBlobWriter, PackedBlobWriter,
    PreparedBlobValue,
};
use lance_io::object_store::LanceNamespaceStorageOptionsProvider;
use pyo3::{
    Bound, PyResult,
    exceptions::PyValueError,
    pyclass, pymethods,
    types::{PyAny, PyAnyMethods, PyDict, PyList, PyListMethods, PyModule},
};
use std::collections::HashMap;
use std::sync::Arc;

#[pyclass(name = "PreparedBlobValue", skip_from_py_object)]
#[derive(Clone)]
pub struct PyPreparedBlobValue {
    inner: PreparedBlobValue,
}

impl From<PreparedBlobValue> for PyPreparedBlobValue {
    fn from(inner: PreparedBlobValue) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyPreparedBlobValue {
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(name = "LanceBlobSession", skip_from_py_object)]
pub struct PyLanceBlobSession {
    inner: LanceBlobSession,
}

#[pymethods]
impl PyLanceBlobSession {
    #[new]
    #[pyo3(signature = (data_file_path, storage_options=None, namespace_client=None, table_id=None))]
    pub fn new(
        data_file_path: String,
        storage_options: Option<HashMap<String, String>>,
        namespace_client: Option<&Bound<'_, PyAny>>,
        table_id: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let provider = if let (Some(ns_client), Some(tid)) = (&namespace_client, &table_id) {
            let ns_client = extract_namespace_arc(ns_client.py(), ns_client)?;
            Some(Arc::new(LanceNamespaceStorageOptionsProvider::new(
                ns_client,
                tid.clone(),
            ))
                as Arc<dyn lance_io::object_store::StorageOptionsProvider>)
        } else {
            None
        };

        let (object_store, path) = rt().block_on(
            None,
            object_store_from_uri_or_path_with_provider(data_file_path, storage_options, provider),
        )??;
        let inner = LanceBlobSession::try_new(object_store.as_ref().clone(), path).infer_error()?;
        Ok(Self { inner })
    }

    pub fn open_writer(&self, column: String) -> PyLanceBlobWriter {
        let inner = self.inner.open_writer(column);
        PyLanceBlobWriter {
            field: inner.field().clone(),
            inner: Some(inner),
        }
    }

    pub fn blob_path(&self, blob_id: u32) -> PyResult<String> {
        Ok(self.inner.blob_path(blob_id).infer_error()?.to_string())
    }

    #[getter]
    pub fn data_file_key(&self) -> String {
        self.inner.data_file_key().to_string()
    }
}

#[pyclass(name = "LanceBlobWriter", skip_from_py_object)]
pub struct PyLanceBlobWriter {
    field: arrow_schema::Field,
    inner: Option<LanceBlobWriter>,
}

impl PyLanceBlobWriter {
    fn inner(&self) -> PyResult<&LanceBlobWriter> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("LanceBlobWriter is already finished"))
    }

    fn inner_mut(&mut self) -> PyResult<&mut LanceBlobWriter> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("LanceBlobWriter is already finished"))
    }
}

#[pymethods]
impl PyLanceBlobWriter {
    #[getter]
    pub fn field<'py>(&self, py: pyo3::Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pyarrow = PyModule::import(py, "pyarrow")?;
        let child_fields = PyList::empty(py);
        for (name, type_fn) in [
            ("kind", "uint8"),
            ("data", "large_binary"),
            ("uri", "utf8"),
            ("blob_id", "uint32"),
            ("blob_size", "uint64"),
            ("position", "uint64"),
        ] {
            let data_type = pyarrow.getattr(type_fn)?.call0()?;
            let child = pyarrow.call_method1("field", (name, data_type, true))?;
            child_fields.append(child)?;
        }
        let data_type = pyarrow.call_method1("struct", (child_fields,))?;
        let metadata = PyDict::new(py);
        metadata.set_item("ARROW:extension:name", "lance.blob.v2")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("nullable", self.field.is_nullable())?;
        kwargs.set_item("metadata", metadata)?;
        pyarrow.call_method(
            "field",
            (self.field.name().as_str(), data_type),
            Some(&kwargs),
        )
    }

    pub fn new_packed(&self) -> PyResult<PyPackedBlobWriter> {
        let packed = rt()
            .block_on(None, self.inner()?.new_packed())?
            .infer_error()?;
        Ok(PyPackedBlobWriter {
            inner: Some(packed),
        })
    }

    pub fn new_dedicated(&self) -> PyResult<PyDedicatedBlobWriter> {
        let dedicated = rt()
            .block_on(None, self.inner()?.new_dedicated())?
            .infer_error()?;
        Ok(PyDedicatedBlobWriter {
            inner: Some(dedicated),
        })
    }

    pub fn load_packed(
        &self,
        blob_id: u32,
        offsets: Vec<u64>,
        sizes: Vec<u64>,
    ) -> PyResult<Vec<PyPreparedBlobValue>> {
        if offsets.len() != sizes.len() {
            return Err(PyValueError::new_err(format!(
                "offsets and sizes must have the same length, got {} and {}",
                offsets.len(),
                sizes.len()
            )));
        }
        let ranges = offsets
            .into_iter()
            .zip(sizes)
            .map(|(offset, size)| BlobRange { offset, size })
            .collect::<Vec<_>>();
        let values = rt()
            .block_on(None, self.inner()?.load_packed(blob_id, ranges))?
            .infer_error()?;
        Ok(values.into_iter().map(Into::into).collect())
    }

    pub fn load_dedicated(&self, blob_id: u32) -> PyResult<PyPreparedBlobValue> {
        let value = rt()
            .block_on(None, self.inner()?.load_dedicated(blob_id))?
            .infer_error()?;
        Ok(value.into())
    }

    pub fn push(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = value.extract::<pyo3::PyRef<'_, PyPreparedBlobValue>>()?;
        self.inner_mut()?.push(value.inner.clone()).infer_error()
    }

    pub fn extend(&mut self, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let iter = values.try_iter()?;
        for value in iter {
            let value = value?.extract::<pyo3::PyRef<'_, PyPreparedBlobValue>>()?;
            self.inner_mut()?.push(value.inner.clone()).infer_error()?;
        }
        Ok(())
    }

    pub fn push_inline(&mut self, data: Vec<u8>) -> PyResult<()> {
        self.inner_mut()?
            .push_inline(Bytes::from(data))
            .infer_error()
    }

    pub fn push_null(&mut self) -> PyResult<()> {
        self.inner_mut()?.push_null().infer_error()
    }

    pub fn finish<'py>(&mut self, py: pyo3::Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("LanceBlobWriter is already finished"))?;
        let column = inner.finish().infer_error()?;
        column.array().to_data().to_pyarrow(py)
    }
}

#[pyclass(name = "PackedBlobWriter", skip_from_py_object, unsendable)]
pub struct PyPackedBlobWriter {
    inner: Option<PackedBlobWriter>,
}

impl PyPackedBlobWriter {
    fn inner(&self) -> PyResult<&PackedBlobWriter> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("PackedBlobWriter is already finished"))
    }

    fn inner_mut(&mut self) -> PyResult<&mut PackedBlobWriter> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("PackedBlobWriter is already finished"))
    }
}

#[pymethods]
impl PyPackedBlobWriter {
    #[getter]
    pub fn blob_id(&self) -> PyResult<u32> {
        Ok(self.inner()?.blob_id())
    }

    #[getter]
    pub fn path(&self) -> PyResult<String> {
        Ok(self.inner()?.path().to_string())
    }

    pub fn write_blob(&mut self, data: Vec<u8>) -> PyResult<()> {
        rt().block_on(None, self.inner_mut()?.write_blob(data))?
            .infer_error()
    }

    pub fn finish(&mut self) -> PyResult<Vec<PyPreparedBlobValue>> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("PackedBlobWriter is already finished"))?;
        let values = rt().block_on(None, inner.finish())?.infer_error()?;
        Ok(values.into_iter().map(Into::into).collect())
    }
}

#[pyclass(name = "DedicatedBlobWriter", skip_from_py_object, unsendable)]
pub struct PyDedicatedBlobWriter {
    inner: Option<DedicatedBlobWriter>,
}

impl PyDedicatedBlobWriter {
    fn inner(&self) -> PyResult<&DedicatedBlobWriter> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("DedicatedBlobWriter is already finished"))
    }

    fn inner_mut(&mut self) -> PyResult<&mut DedicatedBlobWriter> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("DedicatedBlobWriter is already finished"))
    }
}

#[pymethods]
impl PyDedicatedBlobWriter {
    #[getter]
    pub fn blob_id(&self) -> PyResult<u32> {
        Ok(self.inner()?.blob_id())
    }

    #[getter]
    pub fn path(&self) -> PyResult<String> {
        Ok(self.inner()?.path().to_string())
    }

    pub fn write(&mut self, data: Vec<u8>) -> PyResult<()> {
        rt().block_on(None, self.inner_mut()?.write(data))?
            .infer_error()
    }

    pub fn finish(&mut self) -> PyResult<PyPreparedBlobValue> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("DedicatedBlobWriter is already finished"))?;
        let value = rt().block_on(None, inner.finish())?.infer_error()?;
        Ok(value.into())
    }
}
