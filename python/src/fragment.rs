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

use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::PyArrowConvert;
use arrow_array::RecordBatchReader;
use arrow_schema::Schema as ArrowSchema;
use lance::dataset::Dataset;
use lance::dataset::fragment::FileFragment as LanceFragment;
use lance::datatypes::Schema;
use lance::format::pb;
use lance::format::{DataFile as LanceDataFile, Fragment as LanceFragmentMetadata};
use tokio::runtime::Runtime;
use prost::Message;
use pyo3::exceptions::*;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyBytes, PyDict};

use crate::dataset::get_write_params;

use crate::updater::Updater;
use crate::Scanner;

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
    fn __repr__(&self) -> String {
        format!("LanceFileFragment(id={})", self.fragment.id())
    }

    #[staticmethod]
    #[pyo3(signature = (dataset_uri, datafile_uri, fragment_id))]
    fn create_from_file(
        dataset_uri: &str,
        datafile_uri: &str,
        fragment_id: usize,
    ) -> PyResult<FragmentMetadata> {
        let rt = Runtime::new()?;
        let dataset = rt.block_on(async {
            Dataset::open(dataset_uri).await.unwrap()
        });
        let schema = dataset.schema();
        let metadata = rt.block_on(async {
            LanceFragment::create_from_file(dataset_uri, datafile_uri, fragment_id)
            .await
            .map_err(|err| PyIOError::new_err(err.to_string()))
        })?;
        Ok(FragmentMetadata::new(metadata, schema.clone()))
    }

    #[staticmethod]
    #[pyo3(signature = (dataset_uri, fragment_id, reader, **kwargs))]
    fn create(
        dataset_uri: &str,
        fragment_id: usize,
        reader: &PyAny,
        kwargs: Option<&PyDict>,
    ) -> PyResult<FragmentMetadata> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let mut batches: Box<dyn RecordBatchReader> = if reader.is_instance_of::<Scanner>()? {
                let scanner: Scanner = reader.extract()?;
                Box::new(
                    scanner
                        .to_reader()
                        .await
                        .map_err(|err| PyValueError::new_err(err.to_string()))?,
                )
            } else {
                Box::new(ArrowArrayStreamReader::from_pyarrow(reader)?)
            };
            let schema = Schema::try_from(batches.schema().as_ref())
                .map_err(|err| PyValueError::new_err(err.to_string()))?;

            let params = if let Some(kw_params) = kwargs {
                get_write_params(kw_params)?
            } else {
                None
            };

            let metadata =
                LanceFragment::create(dataset_uri, fragment_id, batches.as_mut(), params)
                    .await
                    .map_err(|err| PyIOError::new_err(err.to_string()))?;
            Ok(FragmentMetadata::new(metadata, schema))
        })
    }

    fn id(&self) -> usize {
        self.fragment.id()
    }

    fn count_rows(&self, _filter: Option<String>) -> PyResult<usize> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
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
        let rt = Arc::new(tokio::runtime::Runtime::new()?);
        let dataset_schema = self_.fragment.dataset().schema();
        let projection = if let Some(columns) = columns {
            dataset_schema
                .project(&columns)
                .map_err(|e| PyIOError::new_err(e.to_string()))?
        } else {
            dataset_schema.clone()
        };

        let indices = row_indices.iter().map(|v| *v as u32).collect::<Vec<_>>();
        let batch = rt
            .block_on(async { self_.fragment.take(&indices, &projection).await })
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        batch.to_pyarrow(self_.py())
    }

    fn scanner(
        self_: PyRef<'_, Self>,
        columns: Option<Vec<String>>,
        filter: Option<String>,
        limit: Option<i64>,
        offset: Option<i64>,
    ) -> PyResult<Scanner> {
        let rt = Arc::new(tokio::runtime::Runtime::new()?);
        let mut scanner = self_.fragment.scan();
        if let Some(cols) = columns {
            scanner
                .project(&cols)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        if let Some(f) = filter {
            scanner
                .filter(&f)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }

        scanner
            .limit(limit, offset)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let scn = Arc::new(scanner);
        Ok(Scanner::new(scn, rt))
    }

    fn updater(self_: PyRef<'_, Self>, columns: Option<Vec<String>>) -> PyResult<Updater> {
        let rt = tokio::runtime::Runtime::new()?;
        let cols = columns.as_ref().map(|col| col.as_slice());
        let inner = rt
            .block_on(async { self_.fragment.updater(cols).await })
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Ok(Updater::new(inner))
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
        self.inner.path.to_string()
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
                self.schema = Schema::try_from(&manifest.fields).map_err(|e| {
                    PyValueError::new_err(format!("Unable to unpickle FragmentMetadata: {}", e))
                })?;
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
}
