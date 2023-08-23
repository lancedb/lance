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

use ::arrow::pyarrow::PyArrowType;
use ::arrow::pyarrow::{FromPyArrow, ToPyArrow};
use ::arrow_schema::Schema as ArrowSchema;
use ::lance::arrow::json::ArrowJsonExt;
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use env_logger::Env;
use pyo3::exceptions::PyIOError;
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

pub use crate::arrow::{bfloat16_array, BFloat16};
pub use dataset::write_dataset;
pub use dataset::Dataset;
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
        .filter("LANCE_LOG")
        .write_style("LANCE_LOG_STYLE");
    env_logger::init_from_env(env);

    m.add_class::<Scanner>()?;
    m.add_class::<Dataset>()?;
    m.add_class::<FileFragment>()?;
    m.add_class::<FragmentMetadata>()?;
    m.add_class::<DataFile>()?;
    m.add_class::<BFloat16>()?;
    m.add_wrapped(wrap_pyfunction!(bfloat16_array))?;
    m.add_wrapped(wrap_pyfunction!(write_dataset))?;
    m.add_wrapped(wrap_pyfunction!(schema_to_json))?;
    m.add_wrapped(wrap_pyfunction!(json_to_schema))?;
    m.add_wrapped(wrap_pyfunction!(infer_tfrecord_schema))?;
    m.add_wrapped(wrap_pyfunction!(read_tfrecord))?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[pyfunction(name = "_schema_to_json")]
fn schema_to_json(py_schema: &PyAny) -> PyResult<String> {
    let schema = Schema::from_pyarrow(py_schema)?;
    schema.to_json().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Failed to convert schema to json: {}", e))
    })
}

#[pyfunction(name = "_json_to_schema")]
fn json_to_schema(py: Python<'_>, json: &str) -> PyResult<PyObject> {
    let schema = Schema::from_json(json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Failed to convert json to schema: {}, json={}",
            e, json
        ))
    })?;
    schema.to_pyarrow(py)
}

#[pyfunction]
#[pyo3(signature = (uri, *, tensor_features = None, string_features = None))]
fn infer_tfrecord_schema(
    uri: &str,
    tensor_features: Option<Vec<String>>,
    string_features: Option<Vec<String>>,
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
        ))
        .map_err(|err| PyIOError::new_err(err.to_string()))?;
    Ok(PyArrowType(schema))
}

#[pyfunction]
fn read_tfrecord(
    uri: String,
    schema: PyArrowType<ArrowSchema>,
) -> PyResult<PyArrowType<RecordBatch>> {
    let schema = schema.0;
    let record_batch = RT
        .spawn(None, async move {
            ::lance::utils::tfrecord::read_tfrecord(&uri, schema).await
        })
        .map_err(|err| PyIOError::new_err(err.to_string()))?;
    Ok(PyArrowType(record_batch))
}
