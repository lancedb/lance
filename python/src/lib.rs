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

use std::env;
use std::sync::Arc;

use ::arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use ::arrow::pyarrow::PyArrowType;
use ::arrow_schema::Schema as ArrowSchema;
use ::lance::arrow::json::ArrowJsonExt;
use arrow_array::{RecordBatch, RecordBatchIterator};
use arrow_schema::ArrowError;
use dataset::optimize::{
    PyCompaction, PyCompactionMetrics, PyCompactionPlan, PyCompactionTask, PyRewriteResult,
};
use env_logger::Env;
use futures::StreamExt;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

#[macro_use]
extern crate lazy_static;

pub(crate) mod arrow;
pub(crate) mod dataset;
pub(crate) mod errors;
pub(crate) mod executor;
pub(crate) mod fragment;
pub(crate) mod reader;
pub(crate) mod scanner;
pub(crate) mod updater;
pub(crate) mod utils;

pub use crate::arrow::{bfloat16_array, BFloat16};
use crate::fragment::cleanup_partial_writes;
use crate::utils::KMeans;
pub use dataset::write_dataset;
pub use dataset::{Dataset, Operation};
pub use fragment::FragmentMetadata;
use fragment::{DataFile, FileFragment};
pub use reader::LanceReader;
pub use scanner::Scanner;

use crate::executor::BackgroundExecutor;

// TODO: make this runtime configurable (e.g. num threads)
lazy_static! {
    static ref RT: BackgroundExecutor = BackgroundExecutor::new();
}

#[pymodule]
fn lance(_py: Python, m: &PyModule) -> PyResult<()> {
    let env = Env::new()
        .filter_or("LANCE_LOG", "warn")
        .write_style("LANCE_LOG_STYLE");
    env_logger::init_from_env(env);

    m.add_class::<Scanner>()?;
    m.add_class::<Dataset>()?;
    m.add_class::<Operation>()?;
    m.add_class::<FileFragment>()?;
    m.add_class::<FragmentMetadata>()?;
    m.add_class::<DataFile>()?;
    m.add_class::<BFloat16>()?;
    m.add_class::<KMeans>()?;
    m.add_class::<PyCompactionTask>()?;
    m.add_class::<PyCompaction>()?;
    m.add_class::<PyCompactionPlan>()?;
    m.add_class::<PyRewriteResult>()?;
    m.add_class::<PyCompactionMetrics>()?;
    m.add_wrapped(wrap_pyfunction!(bfloat16_array))?;
    m.add_wrapped(wrap_pyfunction!(write_dataset))?;
    m.add_wrapped(wrap_pyfunction!(schema_to_json))?;
    m.add_wrapped(wrap_pyfunction!(json_to_schema))?;
    m.add_wrapped(wrap_pyfunction!(infer_tfrecord_schema))?;
    m.add_wrapped(wrap_pyfunction!(read_tfrecord))?;
    m.add_wrapped(wrap_pyfunction!(cleanup_partial_writes))?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
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
