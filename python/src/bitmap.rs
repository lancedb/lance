// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::pyarrow::FromPyArrow;
use arrow_array::{cast::AsArray, make_array};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use roaring::RoaringBitmap;

/// A lazy, streaming iterator over a `Bitmap`'s values — yields one Python
/// `int` per `__next__` call rather than materializing them all up front.
#[pyclass(name = "BitmapIterator", module = "lance.bitmap")]
pub struct PyBitmapIter(roaring::bitmap::IntoIter);

#[pymethods]
impl PyBitmapIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<u32> {
        slf.0.next()
    }
}

/// A set of non-negative integers backed by a `RoaringBitmap`.
///
/// Cheap to clone (an `Arc` bump) and to pass into Lance APIs that accept a
/// bitmap, since no per-value Python object is created. Mutating methods
/// (`add`, `discard`, `update`) copy-on-write: cloning a `Bitmap` and mutating
/// one copy never affects the other.
#[pyclass(name = "Bitmap", module = "lance.bitmap", from_py_object)]
#[derive(Clone, Debug, Default)]
pub struct PyBitmap(pub Arc<RoaringBitmap>);

impl PyBitmap {
    pub fn new(bitmap: RoaringBitmap) -> Self {
        Self(Arc::new(bitmap))
    }
}

fn i64_to_u32(value: i64) -> PyResult<u32> {
    u32::try_from(value).map_err(|_| {
        PyValueError::new_err(format!(
            "Bitmap values must fit in an unsigned 32-bit integer, got {value}"
        ))
    })
}

fn u64_to_u32(value: u64) -> PyResult<u32> {
    u32::try_from(value).map_err(|_| {
        PyValueError::new_err(format!(
            "Bitmap values must fit in an unsigned 32-bit integer, got {value}"
        ))
    })
}

/// Read an integer pyarrow array's values into a `RoaringBitmap`, without
/// going through per-value Python objects or an intermediate `Vec` — each
/// arm collects straight from the array's native buffer into the bitmap.
fn bitmap_from_pyarrow(ob: &Bound<'_, PyAny>) -> PyResult<RoaringBitmap> {
    let data = ArrayData::from_pyarrow_bound(ob)?;
    if data.null_count() > 0 {
        return Err(PyValueError::new_err(
            "Bitmap cannot be constructed from an array containing nulls",
        ));
    }
    let array = make_array(data);
    match array.data_type() {
        DataType::UInt8 => Ok(array
            .as_primitive::<arrow::datatypes::UInt8Type>()
            .values()
            .iter()
            .map(|v| *v as u32)
            .collect()),
        DataType::UInt16 => Ok(array
            .as_primitive::<arrow::datatypes::UInt16Type>()
            .values()
            .iter()
            .map(|v| *v as u32)
            .collect()),
        DataType::UInt32 => Ok(array
            .as_primitive::<arrow::datatypes::UInt32Type>()
            .values()
            .iter()
            .copied()
            .collect()),
        DataType::UInt64 => array
            .as_primitive::<arrow::datatypes::UInt64Type>()
            .values()
            .iter()
            .map(|v| u64_to_u32(*v))
            .collect(),
        DataType::Int8 => array
            .as_primitive::<arrow::datatypes::Int8Type>()
            .values()
            .iter()
            .map(|v| i64_to_u32(*v as i64))
            .collect(),
        DataType::Int16 => array
            .as_primitive::<arrow::datatypes::Int16Type>()
            .values()
            .iter()
            .map(|v| i64_to_u32(*v as i64))
            .collect(),
        DataType::Int32 => array
            .as_primitive::<arrow::datatypes::Int32Type>()
            .values()
            .iter()
            .map(|v| i64_to_u32(*v as i64))
            .collect(),
        DataType::Int64 => array
            .as_primitive::<arrow::datatypes::Int64Type>()
            .values()
            .iter()
            .map(|v| i64_to_u32(*v))
            .collect(),
        other => Err(PyValueError::new_err(format!(
            "Bitmap can only be constructed from an integer pyarrow array, got {other:?}"
        ))),
    }
}

#[pymethods]
impl PyBitmap {
    /// Construct a Bitmap from an iterable of non-negative ints (list, set,
    /// range, generator, ...) or an integer pyarrow Array/ChunkedArray.
    #[new]
    #[pyo3(signature = (values=None))]
    fn new_py(values: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(values) = values else {
            return Ok(Self::default());
        };
        if let Ok(existing) = values.extract::<Self>() {
            return Ok(existing);
        }
        // A pyarrow `Array` supports the `__arrow_c_array__` Arrow C Data
        // Interface export directly; a `ChunkedArray` doesn't (it only
        // supports the streaming `__arrow_c_stream__` form), so detect it by
        // its `combine_chunks()` method and flatten it to a single `Array`
        // first.
        if values.hasattr("combine_chunks")? {
            let combined = values.call_method0("combine_chunks")?;
            return Ok(Self::new(bitmap_from_pyarrow(&combined)?));
        }
        if values.hasattr("__arrow_c_array__")? {
            return Ok(Self::new(bitmap_from_pyarrow(values)?));
        }
        let bitmap = values
            .try_iter()?
            .map(|item| item?.extract::<u32>())
            .collect::<PyResult<RoaringBitmap>>()?;
        Ok(Self::new(bitmap))
    }

    fn __len__(&self) -> usize {
        self.0.len() as usize
    }

    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        match value.extract::<u32>() {
            Ok(v) => Ok(self.0.contains(v)),
            Err(_) => Ok(false),
        }
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyBitmapIter>> {
        // Cloning the bitmap here is a native Rust-side copy of its compressed
        // containers, not a per-value Python allocation — the point is that
        // `PyBitmapIter` then streams values lazily instead of eagerly
        // building a Python list of boxed ints up front.
        Py::new(py, PyBitmapIter((*self.0).clone().into_iter()))
    }

    fn __repr__(&self) -> String {
        const MAX_VALUES_SHOWN: usize = 20;
        let len = self.0.len();
        let values: Vec<String> = self
            .0
            .iter()
            .take(MAX_VALUES_SHOWN)
            .map(|v| v.to_string())
            .collect();
        if (len as usize) > MAX_VALUES_SHOWN {
            format!("Bitmap({{{}, ...}}, len={})", values.join(", "), len)
        } else {
            format!("Bitmap({{{}}})", values.join(", "))
        }
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<bool> {
        // A value that isn't a Bitmap or an iterable of ints (e.g. `5`) is
        // simply unequal, matching `__contains__` and normal Python `==`
        // semantics — it shouldn't raise just because the type differs.
        let other_bitmap = if let Ok(other) = other.extract::<Self>() {
            Some((*other.0).clone())
        } else {
            other.try_iter().ok().and_then(|it| {
                it.map(|item| item?.extract::<u32>())
                    .collect::<PyResult<RoaringBitmap>>()
                    .ok()
            })
        };
        match (op, other_bitmap) {
            (CompareOp::Eq, Some(other)) => Ok(*self.0 == other),
            (CompareOp::Eq, None) => Ok(false),
            (CompareOp::Ne, Some(other)) => Ok(*self.0 != other),
            (CompareOp::Ne, None) => Ok(true),
            _ => Err(PyNotImplementedError::new_err(
                "Only == and != are supported",
            )),
        }
    }

    /// Add a value, cloning the underlying bitmap first if it is shared with
    /// another `Bitmap`.
    fn add(&mut self, value: u32) {
        Arc::make_mut(&mut self.0).insert(value);
    }

    /// Remove a value if present, cloning the underlying bitmap first if it
    /// is shared with another `Bitmap`.
    fn discard(&mut self, value: u32) {
        Arc::make_mut(&mut self.0).remove(value);
    }

    /// Add all values from an iterable, cloning the underlying bitmap first
    /// if it is shared with another `Bitmap`.
    fn update(&mut self, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let bitmap = Arc::make_mut(&mut self.0);
        for item in values.try_iter()? {
            bitmap.insert(item?.extract::<u32>()?);
        }
        Ok(())
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (Py<PyAny>,))> {
        let mut buf = Vec::new();
        self.0
            .serialize_into(&mut buf)
            .map_err(|e| PyValueError::new_err(format!("Failed to serialize Bitmap: {e}")))?;
        let ctor = py
            .import("lance.bitmap")?
            .getattr("Bitmap")?
            .getattr("_from_bytes")?;
        let bytes = PyBytes::new(py, &buf);
        Ok((ctor.unbind(), (bytes.into(),)))
    }

    #[staticmethod]
    fn _from_bytes(data: &[u8]) -> PyResult<Self> {
        let bitmap = RoaringBitmap::deserialize_from(data)
            .map_err(|e| PyValueError::new_err(format!("Failed to deserialize Bitmap: {e}")))?;
        Ok(Self::new(bitmap))
    }
}
