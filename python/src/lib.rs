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

//! Lance Columnar Data Format
//!
//! Lance columnar data format is an alternative to Parquet. It provides 100x faster for random access,
//! automatic versioning, optimized for computer vision, bioinformatics, spatial and ML data.
//! [Apache Arrow](https://arrow.apache.org/) and DuckDB compatible.

// Workaround for https://github.com/rust-lang/rust-clippy/issues/12039
// Remove after upgrading pyo3 to 0.23
#![allow(clippy::useless_conversion)]

use std::env;
use std::sync::Arc;

use std::ffi::CString;
use ::arrow::array::ArrayRef;
use ::arrow::array::Int32Array;

use ::arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use ::arrow::pyarrow::PyArrowType;
use ::arrow_schema::Schema as ArrowSchema;
use ::lance::arrow::json::ArrowJsonExt;
use arrow_array::{RecordBatch, RecordBatchIterator};
use arrow_schema::ArrowError;
use datafusion_ffi::table_provider::FFI_TableProvider;
use datafusion::error::{DataFusionError, Result};
use datafusion::datasource::MemTable;
use datafusion::arrow::{
    datatypes::{DataType, Field}
};

#[cfg(feature = "datagen")]
use datagen::register_datagen;
use dataset::blob::LanceBlobFile;
use dataset::cleanup::CleanupStats;
use dataset::optimize::{
    PyCompaction, PyCompactionMetrics, PyCompactionPlan, PyCompactionTask, PyRewriteResult,
};
use dataset::MergeInsertBuilder;
use env_logger::{Builder, Env};
use file::{
    LanceBufferDescriptor, LanceColumnMetadata, LanceFileMetadata, LanceFileReader,
    LanceFileStatistics, LanceFileWriter, LancePageMetadata,
};
use futures::StreamExt;
use lance_index::DatasetIndexExt;
use pyo3::exceptions::{PyIOError, PyValueError, PyRuntimeError, PyTypeError};
use log::Level;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use ::lance::datafusion::LanceTableProvider;
use ::lance::dataset::*;
use session::Session;
use tokio::runtime::Runtime;

#[macro_use]
extern crate lazy_static;

pub(crate) mod arrow;
#[cfg(feature = "datagen")]
pub(crate) mod datagen;
pub(crate) mod dataset;
pub(crate) mod debug;
pub(crate) mod error;
pub(crate) mod executor;
pub(crate) mod file;
pub(crate) mod fragment;
pub(crate) mod indices;
pub(crate) mod reader;
pub(crate) mod scanner;
pub(crate) mod schema;
pub(crate) mod session;
pub(crate) mod tracing;
pub(crate) mod transaction;
pub(crate) mod utils;

pub use crate::arrow::{bfloat16_array, BFloat16};
use crate::fragment::{write_fragments, write_fragments_transaction};
pub use crate::tracing::{trace_to_chrome, TraceGuard};
use crate::utils::Hnsw;
use crate::utils::KMeans;
pub use dataset::write_dataset;
pub use dataset::Dataset;
use fragment::{FileFragment, PyDeletionFile, PyRowIdMeta};
pub use indices::register_indices;
pub use reader::LanceReader;
pub use scanner::Scanner;

use crate::executor::BackgroundExecutor;

#[cfg(not(feature = "datagen"))]
#[pyfunction]
pub fn is_datagen_supported() -> bool {
    false
}

// A fallback module for when datagen is not enabled
#[cfg(not(feature = "datagen"))]
fn register_datagen(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let datagen = PyModule::new(py, "datagen")?;
    datagen.add_wrapped(wrap_pyfunction!(is_datagen_supported))?;
    m.add_submodule(&datagen)?;
    Ok(())
}

// TODO: make this runtime configurable (e.g. num threads)
lazy_static! {
    static ref RT: BackgroundExecutor = BackgroundExecutor::new();
}

pub fn init_logging(mut log_builder: Builder) {
    let logger = log_builder.build();

    let max_level = logger.filter();

    let log_level = max_level.to_level().unwrap_or(Level::Error);

    tracing::initialize_tracing(log_level);
    log::set_boxed_logger(Box::new(logger)).unwrap();
    log::set_max_level(max_level);
}

#[pymodule]
fn lance(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let env = Env::new()
        .filter_or("LANCE_LOG", "warn")
        .write_style("LANCE_LOG_STYLE");
    let log_builder = env_logger::Builder::from_env(env);
    init_logging(log_builder);

    m.add_class::<MyTableProvider>()?;
    m.add_class::<MyLanceTableProvider>()?;
    println!("✅ lance module loaded");
    m.add_class::<Scanner>()?;
    m.add_class::<Dataset>()?;
    m.add_class::<FileFragment>()?;
    m.add_class::<PyDeletionFile>()?;
    m.add_class::<PyRowIdMeta>()?;
    m.add_class::<MergeInsertBuilder>()?;
    m.add_class::<LanceBlobFile>()?;
    m.add_class::<LanceFileReader>()?;
    m.add_class::<LanceFileWriter>()?;
    m.add_class::<LanceFileMetadata>()?;
    m.add_class::<LanceFileStatistics>()?;
    m.add_class::<LanceColumnMetadata>()?;
    m.add_class::<LancePageMetadata>()?;
    m.add_class::<LanceBufferDescriptor>()?;
    m.add_class::<BFloat16>()?;
    m.add_class::<CleanupStats>()?;
    m.add_class::<KMeans>()?;
    m.add_class::<Hnsw>()?;
    m.add_class::<PyCompactionTask>()?;
    m.add_class::<PyCompaction>()?;
    m.add_class::<PyCompactionPlan>()?;
    m.add_class::<PyRewriteResult>()?;
    m.add_class::<PyCompactionMetrics>()?;
    m.add_class::<Session>()?;
    m.add_class::<TraceGuard>()?;
    m.add_class::<schema::LanceSchema>()?;
    m.add_wrapped(wrap_pyfunction!(bfloat16_array))?;
    m.add_wrapped(wrap_pyfunction!(write_dataset))?;
    m.add_wrapped(wrap_pyfunction!(write_fragments))?;
    m.add_wrapped(wrap_pyfunction!(write_fragments_transaction))?;
    m.add_wrapped(wrap_pyfunction!(schema_to_json))?;
    m.add_wrapped(wrap_pyfunction!(json_to_schema))?;
    m.add_wrapped(wrap_pyfunction!(infer_tfrecord_schema))?;
    m.add_wrapped(wrap_pyfunction!(read_tfrecord))?;
    m.add_wrapped(wrap_pyfunction!(trace_to_chrome))?;
    m.add_wrapped(wrap_pyfunction!(manifest_needs_migration))?;
    m.add_wrapped(wrap_pyfunction!(language_model_home))?;
    m.add_wrapped(wrap_pyfunction!(bytes_read_counter))?;
    m.add_wrapped(wrap_pyfunction!(iops_counter))?;
    // Debug functions
    m.add_wrapped(wrap_pyfunction!(debug::format_schema))?;
    m.add_wrapped(wrap_pyfunction!(debug::format_manifest))?;
    m.add_wrapped(wrap_pyfunction!(debug::format_fragment))?;
    m.add_wrapped(wrap_pyfunction!(debug::list_transactions))?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    register_datagen(py, m)?;
    register_indices(py, m)?;
    Ok(())
}

#[pyfunction(name = "iops_counter")]
fn iops_counter() -> PyResult<u64> {
    Ok(::lance::io::iops_counter())
}

#[pyfunction(name = "bytes_read_counter")]
fn bytes_read_counter() -> PyResult<u64> {
    Ok(::lance::io::bytes_read_counter())
}

#[pyfunction(name = "_schema_to_json")]
fn schema_to_json(schema: PyArrowType<ArrowSchema>) -> PyResult<String> {
    schema.0.to_json().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Failed to convert schema to json: {}", e))
    })
}

#[pyfunction(name = "_json_to_schema")]
fn json_to_schema(json: &str) -> PyResult<PyArrowType<ArrowSchema>> {
    let schema = ArrowSchema::from_json(json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Failed to convert json to schema: {}, json={}",
            e, json
        ))
    })?;
    Ok(schema.into())
}

#[pyfunction]
pub fn language_model_home() -> PyResult<String> {
    let Some(p) = lance_index::scalar::inverted::language_model_home() else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Failed to get language model home",
        ));
    };
    let Some(pstr) = p.to_str() else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Failed to convert language model home to str",
        ));
    };
    Ok(String::from(pstr))
}

/// Infer schema from tfrecord file
///
/// Parameters
/// ----------
/// uri: str
///     URI of the tfrecord file
/// tensor_features: Optional[List[str]]
///     Names of features that should be treated as tensors. Currently only
///     fixed-shape tensors are supported.
/// string_features: Optional[List[str]]
///     Names of features that should be treated as strings. Otherwise they
///     will be treated as binary.
/// batch_size: Optional[int], default None
///     Number of records to read to infer the schema. If None, will read the
///    entire file.
///
/// Returns
/// -------
/// pyarrow.Schema
///     An Arrow schema inferred from the tfrecord file. The schema is
///     alphabetically sorted by field names, since TFRecord doesn't have
///     a concept of field order.
#[pyfunction]
#[pyo3(signature = (uri, *, tensor_features = None, string_features = None, num_rows = None))]
fn infer_tfrecord_schema(
    uri: &str,
    tensor_features: Option<Vec<String>>,
    string_features: Option<Vec<String>>,
    num_rows: Option<usize>,
) -> PyResult<PyArrowType<ArrowSchema>> {
    let tensor_features = tensor_features.unwrap_or_default();
    let tensor_features = tensor_features
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>();
    let string_features = string_features.unwrap_or_default();
    let string_features = string_features
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>();
    let schema = RT
        .runtime
        .block_on(::lance::utils::tfrecord::infer_tfrecord_schema(
            uri,
            &tensor_features,
            &string_features,
            num_rows,
        ))
        .map_err(|err| PyIOError::new_err(err.to_string()))?;
    Ok(PyArrowType(schema))
}

/// Read tfrecord file as an Arrow stream
///
/// Parameters
/// ----------
/// uri: str
///     URI of the tfrecord file
/// schema: pyarrow.Schema
///     Arrow schema of the tfrecord file. Use :py:func:`infer_tfrecord_schema`
///     to infer the schema. The schema is allowed to be a subset of fields; the
///     reader will only parse the fields that are present in the schema.
/// batch_size: int, default 10k
///     Number of records to read per batch.
///
/// Returns
/// -------
/// pyarrow.RecordBatchReader
///     An Arrow reader, which can be passed directly to
///     :py:func:`lance.write_dataset`. The output schema will match the schema
///     provided, including field order.
#[pyfunction]
#[pyo3(signature = (uri, schema, *, batch_size = 10_000))]
fn read_tfrecord(
    uri: String,
    schema: PyArrowType<ArrowSchema>,
    batch_size: usize,
) -> PyResult<PyArrowType<ArrowArrayStreamReader>> {
    let schema = Arc::new(schema.0);

    let (init_sender, init_receiver) = std::sync::mpsc::channel::<Result<(), ::lance::Error>>();
    let (batch_sender, batch_receiver) =
        std::sync::mpsc::channel::<std::result::Result<RecordBatch, ArrowError>>();

    let schema_ref = schema.clone();
    RT.spawn_background(None, async move {
        let mut stream =
            match ::lance::utils::tfrecord::read_tfrecord(&uri, schema_ref, Some(batch_size)).await
            {
                Ok(stream) => {
                    init_sender.send(Ok(())).unwrap();
                    stream
                }
                Err(err) => {
                    init_sender.send(Err(err)).unwrap();
                    return;
                }
            };

        while let Some(batch) = stream.next().await {
            let batch = batch.map_err(|err| ArrowError::ExternalError(Box::new(err)));
            batch_sender.send(batch).unwrap();
        }
    });

    // Verify initialization happened successfully
    init_receiver.recv().unwrap().map_err(|err| {
        PyIOError::new_err(format!("Failed to initialize tfrecord reader: {}", err))
    })?;

    let batch_reader = RecordBatchIterator::new(batch_receiver, schema);

    // TODO: this should be handled by upstream
    let stream = FFI_ArrowArrayStream::new(Box::new(batch_reader));
    let stream_reader = ArrowArrayStreamReader::try_new(stream).map_err(|err| {
        PyValueError::new_err(format!("Failed to export record batch reader: {}", err))
    })?;

    Ok(PyArrowType(stream_reader))
}

#[pyfunction]
#[pyo3(signature = (dataset,))]
fn manifest_needs_migration(dataset: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = dataset.py();
    let dataset = dataset.getattr("_ds")?.extract::<Py<Dataset>>()?;
    let dataset_ref = &dataset.bind(py).borrow().ds;
    let indices = RT
        .block_on(Some(py), dataset_ref.load_indices())?
        .map_err(|err| PyIOError::new_err(format!("Could not read dataset metadata: {}", err)))?;
    let (manifest, _) = RT
        .block_on(Some(py), dataset_ref.latest_manifest())?
        .map_err(|err| PyIOError::new_err(format!("Could not read dataset metadata: {}", err)))?;
    Ok(::lance::io::commit::manifest_needs_migration(
        &manifest, &indices,
    ))
}

#[pyclass(name = "MyLanceTableProvider", module = "lance", subclass)]
#[derive(Clone)]
struct MyLanceTableProvider {
    dataset: Arc<::lance::Dataset>,
}

// impl MyLanceTableProvider {
//     // Asynchronous initialization function
//     async fn async_initialize(&self) -> Result<::lance::Dataset, String>
//     {
//         let fields: Vec<_> = (0..1)
//             .map(|idx| (b'A' + idx as u8) as char)
//             .map(|col_name| Field::new(col_name, DataType::Int32, true))
//             .collect();
//
//         let schema = Arc::new(ArrowSchema::new(fields));
//
//         let batch = RecordBatch::try_new(
//             schema.clone(),
//             vec![Arc::new(Int32Array::from_iter_values(0..10_i32))],
//         )
//             .unwrap();
//
//         let dataset = InsertBuilder::new("memory://test")
//             .execute(vec![batch])
//             .await
//             .map_err(|e| e.to_string())?;
//
//         Ok(dataset)
//         // self.dataset = Arc::new(dataset);
//     }
// }

#[pymethods]
impl MyLanceTableProvider {
    #[new]
    fn new() -> PyResult<Self> {
        let rt = Runtime::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let dataset = rt.block_on(async {
            let fields = vec![
                Field::new("A", DataType::Int32, true),
            ];
            let schema = Arc::new(ArrowSchema::new(fields));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from_iter_values(0..10))],
            ).map_err(|e| e.to_string())?;

            let dataset = InsertBuilder::new("memory://test")
                .execute(vec![batch])
                .await
                .map_err(|e| e.to_string())?;

            Ok::<_, String>(Arc::new(dataset))
        }).map_err(|e| PyRuntimeError::new_err(e))?;
        println!("dataset created {}", dataset.schema());

        let someTableProvider = Arc::new(LanceTableProvider::new(
            dataset.clone(),
            false,
            false,
        ));
        println!("LanceTableProvider created");

        Ok(Self {
            dataset
        })
    }

    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyCapsule>>
    {
        let name = CString::new("datafusion_table_provider").unwrap();
        let a_lance_table_provider = Arc::new(LanceTableProvider::new(
            self.dataset.clone(),
            false,
            false,
        ));
        println!("LanceTableProvider created");

        let ffi_provider = FFI_TableProvider::new(a_lance_table_provider, false);
        println!("lance_table_ffi_provider");
        let capsule = PyCapsule::new_bound(py, ffi_provider, Some(name.clone()));
        println!("Lance PyCapsule created");
        capsule
    }
}

#[pyclass(name = "MyTableProvider", module = "lance", subclass)]
#[derive(Clone)]
struct MyTableProvider {
    num_cols: usize,
    num_rows: usize,
    num_batches: usize,
}

#[pymethods]
impl MyTableProvider
{
    #[new]
    fn new(num_cols: usize, num_rows: usize, num_batches: usize) -> Self {
        Self {
            num_cols,
            num_rows,
            num_batches,
        }
    }

    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyCapsule>>
    {
        let name = CString::new("datafusion_table_provider").unwrap();

        let provider = self
            .create_table()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let ffi_provider = FFI_TableProvider::new(Arc::new(provider), false);
        let capsule = PyCapsule::new_bound(py, ffi_provider, Some(name.clone()));
        capsule
    }
}

impl MyTableProvider {
    fn create_table(&self) -> Result<MemTable> {
        let fields: Vec<_> = (0..self.num_cols)
        // let fields: Vec<_> = (0..0)
            .map(|idx| (b'A' + idx as u8) as char)
            .map(|col_name| Field::new(col_name, DataType::Int32, true))
            .collect();

        let schema = Arc::new(ArrowSchema::new(fields));

        let batches: Result<Vec<_>> = (0..self.num_batches)
        // let batches: Result<Vec<_>> = (0..0)
            .map(|batch_idx| {
                let start_value = batch_idx * self.num_rows;
                create_record_batch(
                    &schema,
                    self.num_cols,
                    start_value as i32,
                    self.num_rows + batch_idx,
                )
            })
            .collect();

        MemTable::try_new(schema, vec![batches?])
    }
}

fn create_record_batch(
    schema: &Arc<ArrowSchema>,
    num_cols: usize,
    start_value: i32,
    num_values: usize,
) -> Result<RecordBatch> {
    let end_value = start_value + num_values as i32;
    let row_values: Vec<i32> = (start_value..end_value).collect();

    let columns: Vec<_> = (0..num_cols)
        .map(|_| {
            std::sync::Arc::new(Int32Array::from(row_values.clone())) as ArrayRef
        })
        .collect();

    RecordBatch::try_new(Arc::clone(schema), columns).map_err(DataFusionError::from)
}