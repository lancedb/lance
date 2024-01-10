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
use lance::datatypes::Schema;
use lance::format::{pb, DataFile as LanceDataFile, Fragment as LanceFragmentMetadata};
use lance_core::io::{deletion::deletion_file_path, object_store::ObjectStore};
use object_store::path::Path;
use prost::Message;
use pyo3::prelude::*;
use pyo3::{
    exceptions::*,
    pyclass::CompareOp,
    types::{PyBytes, PyDict},
};

use crate::dataset::get_write_params;
use crate::updater::Updater;
use crate::{Scanner, RT};

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
    #[pyo3(signature = (filename, schema, fragment_id))]
    fn create_from_file(
        filename: &str,
        schema: PyArrowType<ArrowSchema>,
        fragment_id: usize,
    ) -> PyResult<FragmentMetadata> {
        let arrow_schema = schema.0;
        let schema = Schema::try_from(&arrow_schema).map_err(|e| {
            PyValueError::new_err(format!(
                "Failed to convert Arrow schema to Lance schema: {}",
                e
            ))
        })?;
        let metadata = RT.block_on(None, async {
            LanceFragment::create_from_file(filename, &schema, fragment_id, None)
                .await
                .map_err(|err| PyIOError::new_err(err.to_string()))
        })??;
        Ok(FragmentMetadata::new(metadata, schema))
    }

    #[staticmethod]
    #[pyo3(signature = (dataset_uri, fragment_id, reader, **kwargs))]
    fn create(
        dataset_uri: &str,
        fragment_id: Option<usize>,
        reader: &PyAny,
        kwargs: Option<&PyDict>,
    ) -> PyResult<FragmentMetadata> {
        let params = if let Some(kw_params) = kwargs {
            get_write_params(kw_params)?
        } else {
            None
        };

        let batches = convert_reader(reader)?;

        reader.py().allow_threads(|| {
            RT.runtime.block_on(async move {
                let schema = batches.schema().clone();

                let metadata =
                    LanceFragment::create(dataset_uri, fragment_id.unwrap_or(0), batches, params)
                        .await
                        .map_err(|err| PyIOError::new_err(err.to_string()))?;
                let schema = Schema::try_from(schema.as_ref())
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;

                Ok(FragmentMetadata::new(metadata, schema))
            })
        })
    }

    fn id(&self) -> usize {
        self.fragment.id()
    }

    fn metadata(&self) -> FragmentMetadata {
        FragmentMetadata::new(
            self.fragment.metadata().clone(),
            self.fragment.dataset().schema().clone(),
        )
    }

    fn count_rows(&self, _filter: Option<String>) -> PyResult<usize> {
        RT.runtime.block_on(async {
            self.fragment
                .count_rows()
                .await
                .map_err(|e| PyIOError::new_err(e.to_string()))
        })
    }

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
    fn scanner(
        self_: PyRef<'_, Self>,
        columns: Option<Vec<String>>,
        batch_size: Option<usize>,
        filter: Option<String>,
        limit: Option<i64>,
        offset: Option<i64>,
        with_row_id: Option<bool>,
        batch_readahead: Option<usize>,
    ) -> PyResult<Scanner> {
        let mut scanner = self_.fragment.scan();
        if let Some(cols) = columns {
            scanner
                .project(&cols)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
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
        if let Some(batch_readahead) = batch_readahead {
            scanner.batch_readahead(batch_readahead);
        }

        let scn = Arc::new(scanner);
        Ok(Scanner::new(scn))
    }

    fn updater(&self, columns: Option<Vec<String>>) -> PyResult<Updater> {
        let cols = columns.as_deref();
        let inner = RT
            .block_on(None, async { self.fragment.updater(cols).await })?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Ok(Updater::new(inner))
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
    fn data_files(self_: PyRef<'_, Self>) -> PyResult<Vec<DataFile>> {
        let data_files: Vec<DataFile> = self_
            .fragment
            .metadata()
            .files
            .iter()
            .map(|f| DataFile::new(f.clone()))
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

/// Metadata of a DataFile.
#[pyclass(name = "_DataFile", module = "_lib")]
pub struct DataFile {
    pub(crate) inner: LanceDataFile,
}

impl DataFile {
    fn new(inner: LanceDataFile) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl DataFile {
    fn __repr__(&self) -> String {
        format!("DataFile({})", self.path())
    }

    fn path(&self) -> String {
        self.inner.path.clone()
    }

    fn field_ids(&self) -> Vec<i32> {
        self.inner.fields.clone()
    }
}

#[pyclass(name = "_FragmentMetadata", module = "lance")]
#[derive(Clone, Debug)]
pub struct FragmentMetadata {
    pub(crate) inner: LanceFragmentMetadata,
    schema: Schema,
}

impl FragmentMetadata {
    pub(crate) fn new(inner: LanceFragmentMetadata, full_schema: Schema) -> Self {
        Self {
            inner,
            schema: full_schema,
        }
    }
}

#[pymethods]
impl FragmentMetadata {
    #[new]
    fn init() -> Self {
        Self {
            inner: LanceFragmentMetadata::new(0),
            schema: Schema::from(&Vec::<pb::Field>::new()),
        }
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let metadata = LanceFragmentMetadata::from_json(json).map_err(|err| {
            PyValueError::new_err(format!("Invalid metadata json payload: {json}: {}", err))
        })?;

        Ok(Self {
            inner: metadata,
            schema: Schema::from(&Vec::<pb::Field>::new()),
        })
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Lt => Ok(self.inner.id < other.inner.id),
            CompareOp::Le => Ok(self.inner.id <= other.inner.id),
            CompareOp::Eq => Ok(self.inner.id == other.inner.id && self.schema == other.schema),
            CompareOp::Ne => self.__richcmp__(other, CompareOp::Eq).map(|v| !v),
            CompareOp::Gt => self.__richcmp__(other, CompareOp::Le).map(|v| !v),
            CompareOp::Ge => self.__richcmp__(other, CompareOp::Lt).map(|v| !v),
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(bytes) => {
                let bytes = bytes.as_bytes();
                let manifest = pb::Manifest::decode(bytes).map_err(|e| {
                    PyValueError::new_err(format!("Unable to unpickle FragmentMetadata: {}", e))
                })?;
                self.schema = Schema::from(&manifest.fields);
                self.inner = LanceFragmentMetadata::from(&manifest.fragments[0]);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn __getstate__(self_: PyRef<'_, Self>) -> PyResult<PyObject> {
        let container = pb::Manifest {
            fields: (&self_.schema).into(),
            fragments: vec![pb::DataFragment::from(&self_.inner)],
            ..Default::default()
        };

        Ok(PyBytes::new(self_.py(), container.encode_to_vec().as_slice()).to_object(self_.py()))
    }

    fn schema(self_: PyRef<'_, Self>) -> PyResult<PyObject> {
        let arrow_schema: ArrowSchema = (&self_.schema).into();
        arrow_schema.to_pyarrow(self_.py())
    }

    fn json(self_: PyRef<'_, Self>) -> PyResult<String> {
        let json = serde_json::to_string(&self_.inner).map_err(|e| {
            PyValueError::new_err(format!("Unable to serialize FragmentMetadata: {}", e))
        })?;
        Ok(json)
    }

    /// Returns the data file objects associated with this fragment.
    fn data_files(self_: PyRef<'_, Self>) -> PyResult<Vec<DataFile>> {
        let data_files: Vec<DataFile> = self_
            .inner
            .files
            .iter()
            .map(|f| DataFile::new(f.clone()))
            .collect();
        Ok(data_files)
    }

    fn deletion_file(&self) -> PyResult<Option<String>> {
        let deletion = self.inner.deletion_file.clone();
        Ok(
            deletion
                .map(|d| deletion_file_path(&Default::default(), self.inner.id, &d).to_string()),
        )
    }

    #[getter]
    fn id(&self) -> PyResult<u64> {
        Ok(self.inner.id)
    }
}

#[pyfunction(name = "_cleanup_partial_writes")]
pub fn cleanup_partial_writes(base_uri: &str, files: Vec<(String, String)>) -> PyResult<()> {
    let (store, _) = RT
        .runtime
        .block_on(ObjectStore::from_uri(base_uri))
        .map_err(|err| PyIOError::new_err(format!("Failed to create object store: {}", err)))?;

    let files: Vec<(Path, String)> = files
        .into_iter()
        .map(|(path, multipart_id)| (Path::from(path.as_str()), multipart_id))
        .collect();

    #[allow(clippy::map_identity)]
    async fn inner(store: ObjectStore, files: Vec<(Path, String)>) -> Result<(), ::lance::Error> {
        let files_iter = files
            .iter()
            .map(|(path, multipart_id)| (path, multipart_id));
        lance::dataset::cleanup::cleanup_partial_writes(&store, files_iter).await
    }

    RT.runtime
        .block_on(inner(store, files))
        .map_err(|err| PyIOError::new_err(format!("Failed to cleanup files: {}", err)))
}

#[pyfunction(name = "_write_fragments")]
#[pyo3(signature = (dataset_uri, reader, **kwargs))]
pub fn write_fragments(
    dataset_uri: &str,
    reader: &PyAny,
    kwargs: Option<&PyDict>,
) -> PyResult<Vec<FragmentMetadata>> {
    let batches = convert_reader(reader)?;

    let schema = batches.schema();
    let schema =
        Schema::try_from(schema.as_ref()).map_err(|err| PyValueError::new_err(err.to_string()))?;

    let params = kwargs
        .and_then(|params| get_write_params(params).ok().flatten())
        .unwrap_or_default();

    let fragments = RT
        .block_on(Some(reader.py()), async {
            lance::dataset::write_fragments(dataset_uri, batches, params).await
        })?
        .map_err(|err| PyIOError::new_err(err.to_string()))?;

    fragments
        .into_iter()
        .map(|f| Ok(FragmentMetadata::new(f, schema.clone())))
        .collect()
}

fn convert_reader(reader: &PyAny) -> PyResult<Box<dyn RecordBatchReader + Send + 'static>> {
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
        Ok(Box::new(ArrowArrayStreamReader::from_pyarrow(reader)?))
    }
}
