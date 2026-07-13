// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{error::PythonErrorExt, rt};
use arrow::{
    array::{Array, ArrayRef, GenericBinaryArray, OffsetSizeTrait, cast::AsArray, make_array},
    buffer::Buffer,
    pyarrow::{FromPyArrow, ToPyArrow},
};
use arrow_data::ArrayData;
use arrow_schema::{DataType, Field};
use bytes::{Bytes, BytesMut};
use lance::{
    BlobDescriptor, BlobDescriptorArrayBuilder, BlobRange, DedicatedBlobWriter, PackedBlobWriter,
};
use pyo3::{
    Bound, PyResult,
    exceptions::{PyRuntimeError, PyValueError},
    pyclass, pymethods,
    types::{PyAny, PyAnyMethods, PyDict, PyList, PyListMethods, PyModule, PyTypeMethods},
};
use std::{ops::Range, sync::Arc};

/// Reconstruct the PyArrow equivalent of [`BlobDescriptorArrayBuilder::field`].
///
/// Arrow's array bridge does not carry the enclosing extension field, so this
/// rebuilds the canonical six nullable blob-v2 children and
/// `ARROW:extension:name = lance.blob.v2` metadata.
fn descriptor_field_to_pyarrow<'py>(
    field: &Field,
    py: pyo3::Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
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
    kwargs.set_item("nullable", field.is_nullable())?;
    kwargs.set_item("metadata", metadata)?;
    pyarrow.call_method("field", (field.name().as_str(), data_type), Some(&kwargs))
}

/// Normalize inputs accepted by [`PyPackedBlobWriter::write_blobs`] into Arrow arrays.
///
/// BinaryArray, LargeBinaryArray, and ChunkedArray values of either binary type
/// are accepted. Chunk boundaries, nulls, and empty values remain in the arrays;
/// row-level blob semantics are prepared by the binding before invoking the
/// byte-level core writer.
fn extract_blob_payloads(payloads: &Bound<'_, PyAny>) -> PyResult<Vec<ArrayRef>> {
    match ArrayData::from_pyarrow_bound(payloads) {
        Ok(data) => {
            let array = make_array(data);
            if !matches!(array.data_type(), DataType::Binary | DataType::LargeBinary) {
                return Err(PyValueError::new_err(format!(
                    "Packed blob payloads must have Arrow type Binary or LargeBinary, got {}",
                    array.data_type()
                )));
            }
            Ok(vec![array])
        }
        Err(_) => {
            let pyarrow = PyModule::import(payloads.py(), "pyarrow")?;
            let chunked_array_type = pyarrow.getattr("ChunkedArray")?;
            if !payloads.is_instance(&chunked_array_type)? {
                return Err(PyValueError::new_err(format!(
                    "payloads must be a pyarrow BinaryArray, LargeBinaryArray, or ChunkedArray, got {}",
                    payloads.get_type().name()?
                )));
            }

            let chunked_data_type = DataType::from_pyarrow_bound(&payloads.getattr("type")?)?;
            if !matches!(chunked_data_type, DataType::Binary | DataType::LargeBinary) {
                return Err(PyValueError::new_err(format!(
                    "Packed blob payloads must have Arrow type Binary or LargeBinary, got {chunked_data_type}"
                )));
            }

            let chunks = payloads.getattr("chunks")?;
            let mut arrays = Vec::with_capacity(chunks.len()?);
            for chunk in chunks.try_iter()? {
                arrays.push(make_array(ArrayData::from_pyarrow_bound(&chunk?)?));
            }
            for (chunk_index, array) in arrays.iter().enumerate() {
                if !matches!(array.data_type(), DataType::Binary | DataType::LargeBinary) {
                    return Err(PyValueError::new_err(format!(
                        "Packed blob payload chunk {chunk_index} must have Arrow type Binary or LargeBinary, got {}",
                        array.data_type()
                    )));
                }
            }
            Ok(arrays)
        }
    }
}

struct OwnedArrowBuffer(Buffer);

impl AsRef<[u8]> for OwnedArrowBuffer {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

struct PackedBlobBatch {
    bytes: Bytes,
    sizes: Vec<usize>,
    row_validity: Vec<bool>,
}

fn packed_blob_batch(payloads: &dyn Array) -> PyResult<PackedBlobBatch> {
    match payloads.data_type() {
        DataType::Binary => packed_binary_blob_batch(payloads.as_binary::<i32>()),
        DataType::LargeBinary => packed_binary_blob_batch(payloads.as_binary::<i64>()),
        data_type => Err(PyValueError::new_err(format!(
            "Packed blob payloads must have Arrow type Binary or LargeBinary, got {data_type}"
        ))),
    }
}

/// Prepare an Arrow binary array for the byte-level core writer.
///
/// The usual case retains the Arrow value buffer through a zero-copy `Bytes` owner.
/// If non-empty physical data for null rows separates valid values, only the valid
/// ranges are compacted so null payload bytes are never written to the sidecar.
fn packed_binary_blob_batch<O: OffsetSizeTrait>(
    payloads: &GenericBinaryArray<O>,
) -> PyResult<PackedBlobBatch> {
    let value_offsets = payloads.value_offsets();
    let value_data = payloads.value_data();
    let mut sizes = Vec::with_capacity(payloads.len() - payloads.null_count());
    let mut row_validity = Vec::with_capacity(payloads.len());
    let mut source_ranges: Vec<Range<usize>> = Vec::new();

    for row in 0..payloads.len() {
        let is_valid = payloads.is_valid(row);
        row_validity.push(is_valid);
        if !is_valid {
            continue;
        }

        let source_start = value_offsets[row].as_usize();
        let source_end = value_offsets[row + 1].as_usize();
        sizes.push(source_end - source_start);
        if source_start == source_end {
            continue;
        }

        if let Some(last_range) = source_ranges.last_mut()
            && last_range.end == source_start
        {
            last_range.end = source_end;
        } else {
            source_ranges.push(source_start..source_end);
        }
    }

    let total_size = source_ranges.iter().map(Range::len).sum::<usize>();
    let bytes = match source_ranges.as_slice() {
        [] => Bytes::new(),
        [source_range] => {
            let value_buffer = Bytes::from_owner(OwnedArrowBuffer(payloads.values().clone()));
            value_buffer.slice(source_range.clone())
        }
        source_ranges => {
            let mut compacted = BytesMut::with_capacity(total_size);
            for source_range in source_ranges {
                compacted.extend_from_slice(&value_data[source_range.clone()]);
            }
            compacted.freeze()
        }
    };
    debug_assert_eq!(
        bytes.len(),
        total_size,
        "packed blob byte preparation must preserve the valid payload size"
    );

    Ok(PackedBlobBatch {
        bytes,
        sizes,
        row_validity,
    })
}

fn align_blob_descriptors(
    descriptors: Vec<BlobDescriptor>,
    row_validity: &[bool],
) -> PyResult<Vec<BlobDescriptor>> {
    let descriptor_count = descriptors.len();
    let expected_descriptor_count = row_validity.iter().filter(|is_valid| **is_valid).count();
    if descriptor_count != expected_descriptor_count {
        return Err(PyRuntimeError::new_err(format!(
            "Packed blob descriptor count does not match valid input rows: \
             descriptors={descriptor_count}, valid_rows={expected_descriptor_count}"
        )));
    }
    if expected_descriptor_count == row_validity.len() {
        return Ok(descriptors);
    }

    let mut descriptors = descriptors.into_iter();
    let mut aligned = Vec::with_capacity(row_validity.len());
    for is_valid in row_validity {
        if *is_valid {
            let Some(descriptor) = descriptors.next() else {
                return Err(PyRuntimeError::new_err(
                    "Packed blob descriptor alignment ended unexpectedly",
                ));
            };
            aligned.push(descriptor);
        } else {
            aligned.push(BlobDescriptor::Null);
        }
    }
    debug_assert!(
        descriptors.next().is_none(),
        "packed blob descriptor alignment must consume every descriptor"
    );
    Ok(aligned)
}

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
        descriptor_field_to_pyarrow(&self.field, py)
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
    blob_id: u32,
    path: String,
    field: Option<Field>,
    inner: Option<PackedBlobWriter>,
    row_validity: Vec<bool>,
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
        Ok(Self {
            blob_id: inner.blob_id(),
            path: inner.path().to_string(),
            field: None,
            inner: Some(inner),
            row_validity: Vec::new(),
        })
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
    pub fn blob_id(&self) -> u32 {
        self.blob_id
    }

    #[getter]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The descriptor field associated with the array returned by
    /// :meth:`finish_array`.
    ///
    /// The field uses the name passed to ``finish_array`` and carries the
    /// ``lance.blob.v2`` extension metadata. It is available only after
    /// ``finish_array`` succeeds; accessing it earlier raises ``ValueError``.
    #[getter]
    pub fn field<'py>(&self, py: pyo3::Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let field = self.field.as_ref().ok_or_else(|| {
            PyValueError::new_err("PackedBlobWriter field is available after finish_array")
        })?;
        descriptor_field_to_pyarrow(field, py)
    }

    pub fn write_blob(&mut self, data: Vec<u8>) -> PyResult<()> {
        rt().block_on(None, self.inner_mut()?.write_blob(data))?
            .infer_error()?;
        self.row_validity.push(true);
        Ok(())
    }

    /// Append a batch of packed blob payloads.
    ///
    /// Parameters
    /// ----------
    /// payloads : pyarrow.BinaryArray, pyarrow.LargeBinaryArray, or pyarrow.ChunkedArray
    ///     A binary Arrow array. Every chunk of a chunked array must be binary.
    ///     Each input row produces one descriptor row, in order, across chunks
    ///     and repeated calls. Null rows produce null descriptors; empty but
    ///     non-null byte strings produce valid zero-length blobs.
    ///
    /// Examples
    /// --------
    /// >>> import pyarrow as pa
    /// >>> payloads = pa.array([b"first", None, b""], type=pa.large_binary())
    /// >>> writer.write_blobs(payloads)
    /// >>> descriptors = writer.finish_array("blob")
    /// >>> len(descriptors)
    /// 3
    pub fn write_blobs(&mut self, payloads: &Bound<'_, PyAny>) -> PyResult<()> {
        let payloads = extract_blob_payloads(payloads)?;
        let result = {
            let writer = self
                .inner
                .as_mut()
                .ok_or_else(|| PyValueError::new_err("PackedBlobWriter is already finished"))?;
            let row_validity = &mut self.row_validity;
            rt().block_on(None, async {
                for payloads in payloads {
                    let batch = packed_blob_batch(payloads.as_ref())?;
                    writer
                        .write_packed_blobs(batch.bytes, batch.sizes)
                        .await
                        .infer_error()?;
                    row_validity.extend(batch.row_validity);
                }
                Ok(())
            })
        };
        match result {
            Ok(result) => result,
            Err(error) => {
                // KeyboardInterrupt drops the async batch future. Remove the core
                // writer as well so a completed prefix cannot be reused as a new batch.
                self.inner.take();
                Err(error)
            }
        }
    }

    pub fn finish(&mut self) -> PyResult<Vec<PyBlobDescriptor>> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("PackedBlobWriter is already finished"))?;
        let values = rt().block_on(None, inner.finish())?.infer_error()?;
        let values = align_blob_descriptors(values, &self.row_validity)?;
        Ok(values.into_iter().map(Into::into).collect())
    }

    /// Finish the upload and return its blob descriptors as a PyArrow array.
    ///
    /// The returned ``pyarrow.StructArray`` has one row per payload previously
    /// passed to :meth:`write_blob` or :meth:`write_blobs`. The writer is consumed
    /// by this call. After it succeeds, :attr:`field` returns the matching
    /// extension field with ``field_name`` as its name.
    ///
    /// Parameters
    /// ----------
    /// field_name : str
    ///     Name for the descriptor field exposed by :attr:`field`.
    ///
    /// Returns
    /// -------
    /// pyarrow.StructArray
    ///     Row-aligned blob descriptors, including null rows from bulk input.
    ///
    /// Examples
    /// --------
    /// >>> import pyarrow as pa
    /// >>> writer.write_blobs(pa.array([b"value", None]))
    /// >>> descriptors = writer.finish_array("payload")
    /// >>> descriptors.is_null().to_pylist()
    /// [False, True]
    /// >>> writer.field.name
    /// 'payload'
    pub fn finish_array<'py>(
        &mut self,
        py: pyo3::Python<'py>,
        field_name: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("PackedBlobWriter is already finished"))?;
        let values = rt().block_on(None, inner.finish())?.infer_error()?;
        let values = align_blob_descriptors(values, &self.row_validity)?;
        let mut builder = BlobDescriptorArrayBuilder::new(field_name);
        builder.extend(values).infer_error()?;
        let column = builder.finish().infer_error()?;
        let (field, array) = column.into_parts();
        self.field = Some(field);
        array.to_data().to_pyarrow(py)
    }
}

#[pyclass(name = "DedicatedBlobWriter", skip_from_py_object, unsendable)]
pub struct PyDedicatedBlobWriter {
    blob_id: u32,
    path: String,
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
        Ok(Self {
            blob_id: inner.blob_id(),
            path: inner.path().to_string(),
            inner: Some(inner),
        })
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
    pub fn blob_id(&self) -> u32 {
        self.blob_id
    }

    #[getter]
    pub fn path(&self) -> &str {
        &self.path
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
