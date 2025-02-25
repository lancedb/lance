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

use std::fmt::Write as _;
use std::sync::Arc;

use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::{FromPyArrow, PyArrowType, ToPyArrow};
use arrow_array::RecordBatchReader;
use arrow_schema::Schema as ArrowSchema;
use futures::TryFutureExt;
use lance::dataset::fragment::FileFragment as LanceFragment;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{InsertBuilder, NewColumnTransform, WriteDestination};
use lance::Error;
use lance_table::format::{DataFile, DeletionFile, DeletionFileType, Fragment, RowIdMeta};
use lance_table::io::deletion::deletion_file_path;
use object_store::path::Path;
use pyo3::{exceptions::*, types::PyDict};
use pyo3::{intern, prelude::*};
use snafu::location;

use crate::dataset::{get_write_params, transforms_from_python};
use crate::error::PythonErrorExt;
use crate::schema::LanceSchema;
use crate::utils::{export_vec, extract_vec, PyLance};
use crate::{Dataset, Scanner, RT};

#[pyclass(name = "_Fragment", module = "_lib")]
#[derive(Clone)]
pub struct FileFragment {
    fragment: LanceFragment,
}

impl FileFragment {
    pub fn new(frag: LanceFragment) -> Self {
        Self { fragment: frag }
    }
}

#[pymethods]
impl FileFragment {
    fn __repr__(&self) -> PyResult<String> {
        let mut s = String::new();
        write!(
            s,
            "LanceFileFragment(id={}, data_files=[",
            self.fragment.id()
        )
        .unwrap();
        let file_path = self
            .fragment
            .metadata()
            .files
            .iter()
            .map(|f| format!("'{}'", f.path))
            .collect::<Vec<_>>()
            .join(", ");
        write!(s, "{}]", file_path).unwrap();
        if let Some(deletion) = &self.fragment.metadata().deletion_file {
            let path = deletion_file_path(&Default::default(), self.id() as u64, deletion);
            write!(s, ", deletion_file='{}'", path).unwrap();
        }
        write!(s, ")").unwrap();
        Ok(s)
    }

    #[staticmethod]
    #[pyo3(signature = (filename, dataset, fragment_id))]
    fn create_from_file(
        filename: &str,
        dataset: &Dataset,
        fragment_id: usize,
    ) -> PyResult<PyLance<Fragment>> {
        let metadata = RT.block_on(None, async {
            LanceFragment::create_from_file(filename, dataset.ds.as_ref(), fragment_id, None)
                .await
                .map_err(|err| PyIOError::new_err(err.to_string()))
        })??;
        Ok(PyLance(metadata))
    }

    #[staticmethod]
    #[pyo3(signature = (dataset_uri, fragment_id, reader, **kwargs))]
    fn create(
        dataset_uri: &str,
        fragment_id: Option<usize>,
        reader: &Bound<PyAny>,
        kwargs: Option<&PyDict>,
    ) -> PyResult<PyLance<Fragment>> {
        let params = if let Some(kw_params) = kwargs {
            get_write_params(kw_params)?
        } else {
            None
        };

        let batches = convert_reader(reader)?;

        reader.py().allow_threads(|| {
            RT.runtime.block_on(async move {
                let metadata =
                    LanceFragment::create(dataset_uri, fragment_id.unwrap_or(0), batches, params)
                        .await
                        .map_err(|err| PyIOError::new_err(err.to_string()))?;

                Ok(PyLance(metadata))
            })
        })
    }

    fn id(&self) -> usize {
        self.fragment.id()
    }

    pub fn metadata(&self) -> PyLance<Fragment> {
        PyLance(self.fragment.metadata().clone())
    }

    #[pyo3(signature=(filter=None))]
    fn count_rows(&self, filter: Option<String>) -> PyResult<usize> {
        RT.runtime.block_on(async {
            self.fragment
                .count_rows(filter)
                .await
                .map_err(|e| PyIOError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature=(row_indices, columns=None))]
    fn take(
        self_: PyRef<'_, Self>,
        row_indices: Vec<usize>,
        columns: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let dataset_schema = self_.fragment.dataset().schema();
        let projection = if let Some(columns) = columns {
            dataset_schema
                .project(&columns)
                .map_err(|e| PyIOError::new_err(e.to_string()))?
        } else {
            dataset_schema.clone()
        };

        let indices = row_indices.iter().map(|v| *v as u32).collect::<Vec<_>>();
        let fragment = self_.fragment.clone();
        let batch = RT
            .spawn(Some(self_.py()), async move {
                fragment.take(&indices, &projection).await
            })?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        batch.to_pyarrow(self_.py())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(columns=None, columns_with_transform=None, batch_size=None, filter=None, limit=None, offset=None, with_row_id=None, with_row_address=None, batch_readahead=None))]
    fn scanner(
        self_: PyRef<'_, Self>,
        columns: Option<Vec<String>>,
        columns_with_transform: Option<Vec<(String, String)>>,
        batch_size: Option<usize>,
        filter: Option<String>,
        limit: Option<i64>,
        offset: Option<i64>,
        with_row_id: Option<bool>,
        with_row_address: Option<bool>,
        batch_readahead: Option<usize>,
    ) -> PyResult<Scanner> {
        let mut scanner = self_.fragment.scan();

        match (columns, columns_with_transform) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "Cannot specify both `columns` and `columns_with_transform`",
                ));
            }
            (Some(cols), None) => {
                scanner
                    .project(&cols)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
            }
            (None, Some(cols_with_transform)) => {
                scanner
                    .project_with_transform(&cols_with_transform)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
            }
            (None, None) => {}
        }

        if let Some(batch_size) = batch_size {
            scanner.batch_size(batch_size);
        }
        if let Some(f) = filter {
            scanner
                .filter(&f)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }

        scanner
            .limit(limit, offset)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        if with_row_id.unwrap_or(false) {
            scanner.with_row_id();
        }
        if with_row_address.unwrap_or(false) {
            scanner.with_row_address();
        }
        if let Some(batch_readahead) = batch_readahead {
            scanner.batch_readahead(batch_readahead);
        }

        let scn = Arc::new(scanner);
        Ok(Scanner::new(scn))
    }

    #[pyo3(signature=(reader, batch_size=None))]
    fn add_columns_from_reader(
        &mut self,
        reader: &Bound<PyAny>,
        batch_size: Option<u32>,
    ) -> PyResult<(PyLance<Fragment>, LanceSchema)> {
        let batches = ArrowArrayStreamReader::from_pyarrow_bound(reader)?;

        let transforms = NewColumnTransform::Reader(Box::new(batches));

        let fragment = self.fragment.clone();
        let (fragment, schema) = RT
            .spawn(None, async move {
                fragment.add_columns(transforms, None, batch_size).await
            })?
            .infer_error()?;

        Ok((PyLance(fragment), LanceSchema(schema)))
    }

    #[pyo3(signature=(transforms, read_columns=None, batch_size=None))]
    fn add_columns(
        &mut self,
        transforms: &Bound<'_, PyAny>,
        read_columns: Option<Vec<String>>,
        batch_size: Option<u32>,
    ) -> PyResult<(PyLance<Fragment>, LanceSchema)> {
        let transforms = transforms_from_python(transforms)?;

        let fragment = self.fragment.clone();
        let (fragment, schema) = RT
            .spawn(None, async move {
                fragment
                    .add_columns(transforms, read_columns, batch_size)
                    .await
            })?
            .infer_error()?;

        Ok((PyLance(fragment), LanceSchema(schema)))
    }

    fn merge(
        &mut self,
        reader: PyArrowType<ArrowArrayStreamReader>,
        left_on: String,
        right_on: String,
        max_field_id: i32,
    ) -> PyResult<(PyLance<Fragment>, LanceSchema)> {
        let mut fragment = self.fragment.clone();
        let (fragment, schema) = RT
            .spawn(None, async move {
                fragment
                    .merge_columns(reader.0, &left_on, &right_on, max_field_id)
                    .await
            })?
            .infer_error()?;

        Ok((PyLance(fragment), LanceSchema(schema)))
    }

    fn delete(&self, predicate: &str) -> PyResult<Option<Self>> {
        let old_fragment = self.fragment.clone();
        let updated_fragment = RT
            .block_on(None, async { old_fragment.delete(predicate).await })?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;

        match updated_fragment {
            Some(frag) => Ok(Some(Self::new(frag))),
            None => Ok(None),
        }
    }

    fn schema(self_: PyRef<'_, Self>) -> PyResult<PyObject> {
        let schema = self_.fragment.dataset().schema();
        let arrow_schema: ArrowSchema = schema.into();
        arrow_schema.to_pyarrow(self_.py())
    }

    /// Returns the data file objects associated with this fragment.
    fn data_files(self_: PyRef<'_, Self>) -> PyResult<Vec<PyLance<DataFile>>> {
        let data_files: Vec<_> = self_
            .fragment
            .metadata()
            .files
            .iter()
            .map(|f| PyLance(f.clone()))
            .collect();
        Ok(data_files)
    }

    fn deletion_file(&self) -> PyResult<Option<String>> {
        let deletion = self.fragment.metadata().deletion_file.clone();
        Ok(deletion
            .map(|d| deletion_file_path(&Default::default(), self.id() as u64, &d).to_string()))
    }

    #[getter]
    fn num_deletions(&self) -> PyResult<usize> {
        RT.block_on(None, self.fragment.count_deletions())?
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    #[getter]
    fn physical_rows(&self) -> PyResult<usize> {
        RT.block_on(None, self.fragment.physical_rows())?
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }
}

impl From<FileFragment> for LanceFragment {
    fn from(fragment: FileFragment) -> Self {
        fragment.fragment
    }
}

fn do_write_fragments(
    dest: &Bound<PyAny>,
    reader: &Bound<PyAny>,
    kwargs: Option<&PyDict>,
) -> PyResult<Transaction> {
    let batches = convert_reader(reader)?;

    let params = kwargs
        .and_then(|params| get_write_params(params).transpose())
        .transpose()?
        .unwrap_or_default();

    let dest = if dest.is_instance_of::<Dataset>() {
        let dataset: Dataset = dest.extract()?;
        WriteDestination::Dataset(dataset.ds.clone())
    } else {
        WriteDestination::Uri(dest.extract()?)
    };

    RT.block_on(
        Some(reader.py()),
        InsertBuilder::new(dest)
            .with_params(&params)
            .execute_uncommitted_stream(batches),
    )?
    .map_err(|err| PyIOError::new_err(err.to_string()))
}

#[pyfunction(name = "_write_fragments")]
#[pyo3(signature = (dest, reader, **kwargs))]
pub fn write_fragments(
    dest: &Bound<PyAny>,
    reader: &Bound<PyAny>,
    kwargs: Option<&PyDict>,
) -> PyResult<Vec<PyObject>> {
    let written = do_write_fragments(dest, reader, kwargs)?;

    assert!(
        written.blobs_op.is_none(),
        "Blob writing is not yet supported by the python _write_fragments API"
    );

    let get_fragments = |operation| match operation {
        Operation::Overwrite { fragments, .. } => Ok(fragments),
        Operation::Append { fragments, .. } => Ok(fragments),
        _ => Err(Error::Internal {
            message: "Unexpected operation".into(),
            location: location!(),
        }),
    };
    let fragments =
        get_fragments(written.operation).map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    Ok(export_vec(reader.py(), &fragments))
}

#[pyfunction(name = "_write_fragments_transaction")]
#[pyo3(signature = (dest, reader, **kwargs))]
pub fn write_fragments_transaction(
    dest: &Bound<PyAny>,
    reader: &Bound<PyAny>,
    kwargs: Option<&PyDict>,
) -> PyResult<PyObject> {
    let written = do_write_fragments(dest, reader, kwargs)?;

    Ok(PyLance(written).to_object(reader.py()))
}

fn convert_reader(reader: &Bound<PyAny>) -> PyResult<Box<dyn RecordBatchReader + Send + 'static>> {
    if reader.is_instance_of::<Scanner>() {
        let scanner: Scanner = reader.extract()?;
        let reader = RT.block_on(
            Some(reader.py()),
            scanner
                .to_reader()
                .map_err(|err| PyValueError::new_err(err.to_string())),
        )??;
        Ok(Box::new(reader))
    } else {
        Ok(Box::new(ArrowArrayStreamReader::from_pyarrow_bound(
            reader,
        )?))
    }
}

#[pyclass(name = "DeletionFile", module = "lance.fragment")]
pub struct PyDeletionFile(pub DeletionFile);

#[pymethods]
impl PyDeletionFile {
    #[new]
    fn new(read_version: u64, id: u64, file_type: &str, num_deleted_rows: usize) -> PyResult<Self> {
        let file_type = match file_type {
            "array" => DeletionFileType::Array,
            "bitmap" => DeletionFileType::Bitmap,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "file_type must be either 'array' or 'bitmap', got '{}'",
                    file_type
                )))
            }
        };
        Ok(Self(DeletionFile {
            read_version,
            id,
            file_type,
            num_deleted_rows: Some(num_deleted_rows),
        }))
    }

    fn asdict(slf: PyRef<'_, Self>) -> PyResult<Bound<'_, PyDict>> {
        let dict = PyDict::new_bound(slf.py());

        dict.set_item(intern!(slf.py(), "read_version"), slf.0.read_version)?;
        dict.set_item(intern!(slf.py(), "id"), slf.0.id)?;
        dict.set_item(intern!(slf.py(), "file_type"), slf.file_type())?;
        dict.set_item(
            intern!(slf.py(), "num_deleted_rows"),
            slf.0.num_deleted_rows,
        )?;

        Ok(dict)
    }

    fn __repr__(&self) -> String {
        let mut repr = "DeletionFile(".to_string();
        write!(repr, "type='{}'", self.file_type()).unwrap();
        if let Some(num_deleted_rows) = self.0.num_deleted_rows {
            write!(repr, ", num_deleted_rows={}", num_deleted_rows).unwrap();
        }
        write!(repr, ")").unwrap();
        repr
    }

    #[getter]
    fn read_version(&self) -> u64 {
        self.0.read_version
    }

    #[getter]
    fn id(&self) -> u64 {
        self.0.id
    }

    #[getter]
    fn num_deleted_rows(&self) -> Option<usize> {
        self.0.num_deleted_rows
    }

    #[getter]
    fn file_type(&self) -> &str {
        match self.0.file_type {
            DeletionFileType::Array => "array",
            DeletionFileType::Bitmap => "bitmap",
        }
    }

    #[pyo3(signature = (fragment_id, base_uri=None))]
    fn path(&self, fragment_id: u64, base_uri: Option<&str>) -> PyResult<String> {
        let base_path = if let Some(base_uri) = base_uri {
            Path::from_url_path(base_uri).map_err(|e| {
                PyValueError::new_err(format!("Invalid base URI: {}: {}", base_uri, e))
            })?
        } else {
            Path::default()
        };
        Ok(deletion_file_path(&base_path, fragment_id, &self.0).to_string())
    }
}

#[pyclass(name = "RowIdMeta", module = "lance.fragment")]
pub struct PyRowIdMeta(pub RowIdMeta);

#[pymethods]
impl PyRowIdMeta {
    fn asdict(&self) -> PyResult<Bound<'_, PyDict>> {
        Err(PyNotImplementedError::new_err(
            "PyRowIdMeta.asdict is not yet supported.s",
        ))
    }
}

impl FromPyObject<'_> for PyLance<Fragment> {
    fn extract_bound(ob: &pyo3::Bound<'_, PyAny>) -> PyResult<Self> {
        let files = extract_vec(&ob.getattr("files")?)?;

        let deletion_file: Option<PyRef<PyDeletionFile>> =
            ob.getattr("deletion_file")?.extract()?;
        let deletion_file = deletion_file.map(|f| f.0.clone());

        let row_id_meta: Option<PyRef<PyRowIdMeta>> = ob.getattr("row_id_meta")?.extract()?;
        let row_id_meta = row_id_meta.map(|r| r.0.clone());

        Ok(Self(Fragment {
            id: ob.getattr("id")?.extract()?,
            files,
            deletion_file,
            physical_rows: ob.getattr("physical_rows")?.extract()?,
            row_id_meta,
        }))
    }
}

impl ToPyObject for PyLance<&Fragment> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let cls = py
            .import_bound(intern!(py, "lance.fragment"))
            .and_then(|m| m.getattr("FragmentMetadata"))
            .expect("FragmentMetadata class not found");

        let files = export_vec(py, &self.0.files);
        let deletion_file = self
            .0
            .deletion_file
            .as_ref()
            .map(|f| PyDeletionFile(f.clone()));
        let row_id_meta = self.0.row_id_meta.as_ref().map(|r| PyRowIdMeta(r.clone()));

        cls.call1((
            self.0.id,
            files,
            self.0.physical_rows,
            deletion_file,
            row_id_meta,
        ))
        .unwrap()
        .to_object(py)
    }
}

impl ToPyObject for PyLance<Fragment> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        PyLance(&self.0).to_object(py)
    }
}

impl FromPyObject<'_> for PyLance<DataFile> {
    fn extract_bound(ob: &pyo3::Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self(DataFile {
            path: ob.getattr("path")?.extract()?,
            fields: ob.getattr("fields")?.extract()?,
            column_indices: ob.getattr("column_indices")?.extract()?,
            file_major_version: ob.getattr("file_major_version")?.extract()?,
            file_minor_version: ob.getattr("file_minor_version")?.extract()?,
        }))
    }
}

impl ToPyObject for PyLance<&DataFile> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let cls = py
            .import_bound(intern!(py, "lance.fragment"))
            .and_then(|m| m.getattr("DataFile"))
            .expect("DataFile class not found");

        cls.call1((
            &self.0.path,
            self.0.fields.clone(),
            self.0.column_indices.clone(),
            self.0.file_major_version,
            self.0.file_minor_version,
        ))
        .unwrap()
        .to_object(py)
    }
}

impl ToPyObject for PyLance<DataFile> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        PyLance(&self.0).to_object(py)
    }
}
