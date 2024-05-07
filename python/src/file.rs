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

use std::{pin::Pin, sync::Arc};

use arrow::pyarrow::PyArrowType;
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::Schema as ArrowSchema;
use futures::stream::StreamExt;
use lance::io::{ObjectStore, RecordBatchStream};
use lance_file::v2::{
    reader::{BufferDescriptor, CachedFileMetadata, FileReader},
    writer::{FileWriter, FileWriterOptions},
};
use lance_io::{scheduler::StoreScheduler, ReadBatchParams};
use object_store::path::Path;
use pyo3::{
    exceptions::{PyIOError, PyRuntimeError, PyValueError},
    pyclass, pymethods, IntoPy, PyObject, PyResult, Python,
};
use serde::Serialize;
use url::Url;

use crate::{error::PythonErrorExt, RT};

#[pyclass(get_all)]
#[derive(Clone, Debug, Serialize)]
pub struct LanceBufferDescriptor {
    /// The byte offset of the buffer in the file
    pub position: u64,
    /// The size (in bytes) of the buffer
    pub size: u64,
}

impl LanceBufferDescriptor {
    fn new(inner: &BufferDescriptor) -> Self {
        Self {
            position: inner.position,
            size: inner.size,
        }
    }

    fn new_from_parts(position: u64, size: u64) -> Self {
        Self { position, size }
    }
}

#[pyclass(get_all)]
#[derive(Clone, Debug, Serialize)]
pub struct LancePageMetadata {
    /// The buffers in the page
    pub buffers: Vec<LanceBufferDescriptor>,
    /// A description of the encoding used to encode the page
    pub encoding: String,
}

impl LancePageMetadata {
    fn new(inner: &lance_file::format::pbfile::column_metadata::Page) -> Self {
        let buffers = inner
            .buffer_offsets
            .iter()
            .zip(inner.buffer_sizes.iter())
            .map(|(pos, size)| LanceBufferDescriptor::new_from_parts(*pos, *size))
            .collect();
        Self {
            buffers,
            encoding: lance_file::v2::reader::describe_encoding(inner),
        }
    }
}

#[pyclass(get_all)]
#[derive(Clone, Debug, Serialize)]
pub struct LanceColumnMetadata {
    /// The column-wide buffers
    pub column_buffers: Vec<LanceBufferDescriptor>,
    /// The data pages in the column
    pub pages: Vec<LancePageMetadata>,
}

impl LanceColumnMetadata {
    fn new(inner: &lance_file::format::pbfile::ColumnMetadata) -> Self {
        let column_buffers = inner
            .buffer_offsets
            .iter()
            .zip(inner.buffer_sizes.iter())
            .map(|(pos, size)| LanceBufferDescriptor::new_from_parts(*pos, *size))
            .collect();
        Self {
            column_buffers,
            pages: inner.pages.iter().map(LancePageMetadata::new).collect(),
        }
    }
}

#[pyclass(get_all)]
#[derive(Clone, Debug, Serialize)]
pub struct LanceFileMetadata {
    /// The schema of the file
    #[serde(skip)]
    pub schema: Option<PyObject>,
    /// The major version of the file
    pub major_version: u16,
    /// The minor version of the file
    pub minor_version: u16,
    /// The number of rows in the file
    pub num_rows: u64,
    /// The number of bytes in the data section of the file
    pub num_data_bytes: u64,
    /// The number of bytes in the column metadata section of the file
    pub num_column_metadata_bytes: u64,
    /// The number of bytes in the global buffer section of the file
    pub num_global_buffer_bytes: u64,
    /// The global buffers
    pub global_buffers: Vec<LanceBufferDescriptor>,
    /// The column metadata, an entry might be None if the metadata for a column
    /// was not loaded into memory when the file was opened.
    pub columns: Vec<Option<LanceColumnMetadata>>,
}

impl LanceFileMetadata {
    fn new(inner: &CachedFileMetadata, py: Python) -> Self {
        let arrow_schema = arrow_schema::Schema::from(inner.file_schema.as_ref());
        let schema = Some(PyArrowType(arrow_schema).into_py(py));
        Self {
            major_version: inner.major_version,
            minor_version: inner.minor_version,
            schema,
            num_rows: inner.num_rows,
            num_data_bytes: inner.num_data_bytes,
            num_column_metadata_bytes: inner.num_column_metadata_bytes,
            num_global_buffer_bytes: inner.num_global_buffer_bytes,
            global_buffers: inner
                .file_buffers
                .iter()
                .map(LanceBufferDescriptor::new)
                .collect(),
            columns: inner
                .column_metadatas
                .iter()
                .map(LanceColumnMetadata::new)
                .map(Some)
                .collect(),
        }
    }
}

#[pymethods]
impl LanceFileMetadata {
    pub fn __repr__(&self) -> PyResult<String> {
        serde_yaml::to_string(self).map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass]
pub struct LanceFileWriter {
    inner: Box<FileWriter>,
}

impl LanceFileWriter {
    async fn open(
        uri_or_path: String,
        schema: PyArrowType<ArrowSchema>,
        data_cache_bytes: Option<u64>,
        keep_original_array: Option<bool>,
    ) -> PyResult<Self> {
        let (object_store, path) = object_store_from_uri_or_path(uri_or_path).await?;
        let object_writer = object_store.create(&path).await.infer_error()?;
        let lance_schema = lance_core::datatypes::Schema::try_from(&schema.0).infer_error()?;
        let inner = FileWriter::try_new(
            object_writer,
            path.to_string(),
            lance_schema,
            FileWriterOptions {
                data_cache_bytes,
                keep_original_array,
            },
        )
        .infer_error()?;
        Ok(Self {
            inner: Box::new(inner),
        })
    }
}

#[pymethods]
impl LanceFileWriter {
    #[new]
    pub fn new(
        path: String,
        schema: PyArrowType<ArrowSchema>,
        data_cache_bytes: Option<u64>,
        keep_original_array: Option<bool>,
    ) -> PyResult<Self> {
        RT.runtime.block_on(Self::open(
            path,
            schema,
            data_cache_bytes,
            keep_original_array,
        ))
    }

    pub fn write_batch(&mut self, batch: PyArrowType<RecordBatch>) -> PyResult<()> {
        RT.runtime
            .block_on(self.inner.write_batch(&batch.0))
            .infer_error()
    }

    pub fn finish(&mut self) -> PyResult<u64> {
        RT.runtime.block_on(self.inner.finish()).infer_error()
    }
}

fn path_to_parent(path: &Path) -> PyResult<(Path, String)> {
    let mut parts = path.parts().collect::<Vec<_>>();
    if parts.is_empty() {
        return Err(PyValueError::new_err(format!(
            "Path {} is not a valid path to a file",
            path,
        )));
    }
    let filename = parts.pop().unwrap().as_ref().to_owned();
    Ok((Path::from_iter(parts), filename))
}

// The ObjectStore::from_uri_or_path expects a path to a directory (and it creates it if it does
// not exist).  We are given a path to a file and so we need to strip the last component
// before creating the object store.  We then return the object store and the new relative path
// to the file.
async fn object_store_from_uri_or_path(uri_or_path: String) -> PyResult<(ObjectStore, Path)> {
    if let Ok(mut url) = Url::parse(&uri_or_path) {
        let path = object_store::path::Path::parse(url.path())
            .map_err(|e| PyIOError::new_err(format!("Invalid URL path `{}`: {}", url.path(), e)))?;
        let (parent_path, filename) = path_to_parent(&path)?;
        url.set_path(parent_path.as_ref());

        let (object_store, dir_path) = ObjectStore::from_uri(url.as_str()).await.infer_error()?;
        let child_path = dir_path.child(filename);
        Ok((object_store, child_path))
    } else {
        let path = Path::parse(&uri_or_path)
            .map_err(|e| PyIOError::new_err(format!("Invalid path `{}`: {}", uri_or_path, e)))?;
        let object_store = ObjectStore::local();
        Ok((object_store, path))
    }
}

#[pyclass]
pub struct LanceFileReader {
    inner: Arc<FileReader>,
}

impl LanceFileReader {
    async fn open(uri_or_path: String) -> PyResult<Self> {
        let (object_store, path) = object_store_from_uri_or_path(uri_or_path).await?;
        let io_parallelism = std::env::var("IO_THREADS")
            .map(|val| val.parse::<u32>().unwrap_or(8))
            .unwrap_or(8);
        let scheduler = StoreScheduler::new(Arc::new(object_store), io_parallelism);
        let file = scheduler.open_file(&path).await.infer_error()?;
        let inner = FileReader::try_open(file, None).await.infer_error()?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

struct LanceReaderAdapter(Pin<Box<dyn RecordBatchStream>>);

impl Iterator for LanceReaderAdapter {
    type Item = std::result::Result<RecordBatch, arrow::error::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = RT.runtime.block_on(self.0.next());
        batch.map(|b| b.map_err(|e| e.into()))
    }
}

impl RecordBatchReader for LanceReaderAdapter {
    fn schema(&self) -> std::sync::Arc<arrow_schema::Schema> {
        self.0.schema().clone()
    }
}

impl LanceFileReader {
    fn read_stream(
        &mut self,
        params: ReadBatchParams,
        batch_size: u32,
        batch_readahead: u32,
    ) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        // read_stream is a synchronous method but it launches tasks and needs to be
        // run in the context of a tokio runtime
        let inner = self.inner.clone();
        let _guard = RT.runtime.enter();
        let stream = inner
            .read_stream(params, batch_size, batch_readahead)
            .infer_error()?;
        Ok(PyArrowType(Box::new(LanceReaderAdapter(stream))))
    }
}

#[pymethods]
impl LanceFileReader {
    #[new]
    pub fn new(path: String) -> PyResult<Self> {
        RT.runtime.block_on(Self::open(path))
    }

    pub fn read_all(
        &mut self,
        batch_size: u32,
        batch_readahead: u32,
    ) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        self.read_stream(
            lance_io::ReadBatchParams::RangeFull,
            batch_size,
            batch_readahead,
        )
    }

    pub fn read_range(
        &mut self,
        offset: u64,
        num_rows: u64,
        batch_size: u32,
        batch_readahead: u32,
    ) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        self.read_stream(
            lance_io::ReadBatchParams::Range((offset as usize)..(offset + num_rows) as usize),
            batch_size,
            batch_readahead,
        )
    }

    pub fn metadata(&mut self, py: Python) -> LanceFileMetadata {
        let inner_meta = self.inner.metadata();
        LanceFileMetadata::new(inner_meta, py)
    }
}
