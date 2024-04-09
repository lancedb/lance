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
use lance_io::scheduler::StoreScheduler;
use object_store::path::Path;
use pyo3::{exceptions::PyRuntimeError, pyclass, pymethods, IntoPy, PyObject, PyResult, Python};
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
        let schema = Some(PyArrowType(inner.file_schema.clone()).into_py(py));
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
    async fn open(uri_or_path: String, schema: PyArrowType<ArrowSchema>) -> PyResult<Self> {
        let (object_store, path) = if Url::parse(&uri_or_path).is_ok() {
            ObjectStore::from_uri(&uri_or_path).await.infer_error()?
        } else {
            (
                ObjectStore::local(),
                Path::parse(uri_or_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )
        };
        let object_writer = object_store.create(&path).await.infer_error()?;
        let inner = FileWriter::try_new(
            object_writer,
            schema.0.clone(),
            FileWriterOptions::default(),
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
    pub fn new(path: String, schema: PyArrowType<ArrowSchema>) -> PyResult<Self> {
        RT.runtime.block_on(Self::open(path, schema))
    }

    pub fn write_batch(&mut self, batch: PyArrowType<RecordBatch>) -> PyResult<()> {
        RT.runtime
            .block_on(self.inner.write_batch(&batch.0))
            .infer_error()
    }

    pub fn finish(&mut self) -> PyResult<()> {
        RT.runtime.block_on(self.inner.finish()).infer_error()
    }
}

#[pyclass]
pub struct LanceFileReader {
    inner: Box<FileReader>,
}

impl LanceFileReader {
    async fn open(uri_or_path: String, schema: PyArrowType<ArrowSchema>) -> PyResult<Self> {
        let (object_store, path) = if Url::parse(&uri_or_path).is_ok() {
            ObjectStore::from_uri(&uri_or_path).await.infer_error()?
        } else {
            (
                ObjectStore::local(),
                Path::parse(uri_or_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            )
        };
        let scheduler = StoreScheduler::new(Arc::new(object_store), 8);
        let file = scheduler.open_file(&path).await.infer_error()?;
        let inner = FileReader::try_open(file, schema.0.clone())
            .await
            .infer_error()?;
        Ok(Self {
            inner: Box::new(inner),
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

#[pymethods]
impl LanceFileReader {
    #[new]
    pub fn new(path: String, schema: PyArrowType<ArrowSchema>) -> PyResult<Self> {
        RT.runtime.block_on(Self::open(path, schema))
    }

    pub fn read_all(
        &mut self,
        batch_size: u32,
    ) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        let stream = RT.runtime.block_on(
            self.inner
                .read_stream(lance_io::ReadBatchParams::RangeFull, batch_size),
        );
        Ok(PyArrowType(Box::new(LanceReaderAdapter(stream))))
    }

    pub fn read_range(
        &mut self,
        offset: u64,
        num_rows: u64,
        batch_size: u32,
    ) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        let stream = RT.runtime.block_on(self.inner.read_stream(
            lance_io::ReadBatchParams::Range((offset as usize)..(offset + num_rows) as usize),
            batch_size,
        ));
        Ok(PyArrowType(Box::new(LanceReaderAdapter(stream))))
    }

    pub fn metadata(&mut self, py: Python) -> LanceFileMetadata {
        let inner_meta = self.inner.metadata();
        LanceFileMetadata::new(inner_meta, py)
    }
}
