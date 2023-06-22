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

use ::lance::arrow::json::ArrowJsonExt;
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_schema::Schema;
use env_logger::Env;
use pyo3::prelude::*;

pub(crate) mod arrow;
pub(crate) mod dataset;
pub(crate) mod errors;
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
