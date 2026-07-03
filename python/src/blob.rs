// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{error::PythonErrorExt, rt};
use arrow::pyarrow::ToPyArrow;
use bytes::Bytes;
use lance::{
    BlobDescriptor, BlobDescriptorArrayBuilder, BlobRange, DedicatedBlobWriter, PackedBlobWriter,
};
use pyo3::{
    Bound, PyResult,
    exceptions::PyValueError,
    pyclass, pymethods,
    types::{PyAny, PyAnyMethods, PyDict, PyList, PyListMethods, PyModule},
};
use std::sync::Arc;

#[pyclass(name = "BlobDescriptor", skip_from_py_object)]
#[derive(Clone)]
pub struct PyBlobDescriptor {
    inner: BlobDescriptor,
}

impl From<BlobDescriptor> for PyBlobDescriptor {
    fn from(inner: BlobDescriptor) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyBlobDescriptor {
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(name = "BlobDescriptorArrayBuilder", skip_from_py_object)]
pub struct PyBlobDescriptorArrayBuilder {
    field: arrow_schema::Field,
    inner: Option<BlobDescriptorArrayBuilder>,
}

impl PyBlobDescriptorArrayBuilder {
    fn inner_mut(&mut self) -> PyResult<&mut BlobDescriptorArrayBuilder> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("BlobDescriptorArrayBuilder is already finished"))
    }
}

#[pymethods]
impl PyBlobDescriptorArrayBuilder {
    #[new]
    pub fn new(column: String) -> Self {
        let inner = BlobDescriptorArrayBuilder::new(column);
        Self {
            field: inner.field().clone(),
            inner: Some(inner),
        }
    }

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

    pub fn extend_packed(
        &mut self,
        blob_id: u32,
        offsets: Vec<u64>,
        sizes: Vec<u64>,
    ) -> PyResult<()> {
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
        self.inner_mut()?
            .extend_packed(blob_id, ranges)
            .infer_error()
    }

    pub fn append_dedicated(&mut self, blob_id: u32, size: u64) -> PyResult<()> {
        self.inner_mut()?
            .push_dedicated(blob_id, size)
            .infer_error()
    }

    pub fn append(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = value.extract::<pyo3::PyRef<'_, PyBlobDescriptor>>()?;
        self.inner_mut()?.push(value.inner.clone()).infer_error()
    }

    pub fn extend(&mut self, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let iter = values.try_iter()?;
        for value in iter {
            let value = value?.extract::<pyo3::PyRef<'_, PyBlobDescriptor>>()?;
            self.inner_mut()?.push(value.inner.clone()).infer_error()?;
        }
        Ok(())
    }

    pub fn append_inline(&mut self, data: Vec<u8>) -> PyResult<()> {
        self.inner_mut()?
            .push_inline(Bytes::from(data))
            .infer_error()
    }

    pub fn append_null(&mut self) -> PyResult<()> {
        self.inner_mut()?.push_null().infer_error()
    }

    pub fn finish<'py>(&mut self, py: pyo3::Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.take().ok_or_else(|| {
            PyValueError::new_err("BlobDescriptorArrayBuilder is already finished")
        })?;
        let column = inner.finish().infer_error()?;
        column.array().to_data().to_pyarrow(py)
    }
}

#[pyclass(name = "PackedBlobWriter", skip_from_py_object, unsendable)]
pub struct PyPackedBlobWriter {
    inner: Option<PackedBlobWriter>,
}

impl PyPackedBlobWriter {
    pub(crate) async fn try_new(
        object_store: Arc<lance_io::object_store::ObjectStore>,
        data_file_path: object_store::path::Path,
        blob_id: u32,
    ) -> PyResult<Self> {
        let inner =
            PackedBlobWriter::try_new(object_store.as_ref().clone(), data_file_path, blob_id)
                .await
                .infer_error()?;
        Ok(Self { inner: Some(inner) })
    }

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

    pub fn finish(&mut self) -> PyResult<Vec<PyBlobDescriptor>> {
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
    pub(crate) async fn try_new(
        object_store: Arc<lance_io::object_store::ObjectStore>,
        data_file_path: object_store::path::Path,
        blob_id: u32,
    ) -> PyResult<Self> {
        let inner =
            DedicatedBlobWriter::try_new(object_store.as_ref().clone(), data_file_path, blob_id)
                .await
                .infer_error()?;
        Ok(Self { inner: Some(inner) })
    }

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

    pub fn finish(&mut self) -> PyResult<PyBlobDescriptor> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("DedicatedBlobWriter is already finished"))?;
        let value = rt().block_on(None, inner.finish())?.infer_error()?;
        Ok(value.into())
    }
}
