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
use std::sync::{Arc, LazyLock};

use std::ffi::CString;

use ::arrow::pyarrow::PyArrowType;
use ::arrow_schema::Schema as ArrowSchema;
use ::lance::arrow::json::ArrowJsonExt;
use ::lance::datafusion::LanceTableProvider;
use datafusion_ffi::table_provider::FFI_TableProvider;
#[cfg(feature = "datagen")]
use datagen::register_datagen;
use dataset::blob::LanceBlobFile;
use dataset::cleanup::CleanupStats;
use dataset::optimize::{
    PyCompaction, PyCompactionMetrics, PyCompactionPlan, PyCompactionTask, PyRewriteResult,
};
use dataset::{MergeInsertBuilder, PyFullTextQuery};
use env_logger::{Builder, Env};
use file::{
    LanceBufferDescriptor, LanceColumnMetadata, LanceFileMetadata, LanceFileReader,
    LanceFileStatistics, LanceFileWriter, LancePageMetadata,
};
use lance_index::DatasetIndexExt;
use log::Level;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyCapsule};
use scanner::ScanStatistics;
use session::Session;

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
use crate::file::LanceFileSession;
use crate::fragment::{write_fragments, write_fragments_transaction};
use crate::tracing::{capture_trace_events, shutdown_tracing, PyTraceEvent};
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

const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");

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
static RT: LazyLock<BackgroundExecutor> = LazyLock::new(BackgroundExecutor::new);

pub fn init_logging(mut log_builder: Builder) {
    let logger = log_builder.build();

    let max_level = logger.filter();

    let log_level = max_level.to_level().unwrap_or(Level::Error);

    tracing::initialize_tracing(log_level);
    log::set_boxed_logger(Box::new(logger)).unwrap();
    log::set_max_level(max_level);
}

fn set_timestamp_precision(builder: &mut env_logger::Builder) {
    if let Ok(timestamp_precision) = env::var("LANCE_LOG_TS_PRECISION") {
        match timestamp_precision.as_str() {
            "ns" => {
                builder.format_timestamp_nanos();
            }
            "us" => {
                builder.format_timestamp_micros();
            }
            "ms" => {
                builder.format_timestamp_millis();
            }
            "s" => {
                builder.format_timestamp_secs();
            }
            _ => {
                // Can't log here because logging is not initialized yet
                println!(
                    "Invalid timestamp precision (valid values: ns, us, ms, s): {}, using default",
                    timestamp_precision
                );
            }
        };
    }
}

#[pymodule]
fn lance(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let env = Env::new()
        .filter_or("LANCE_LOG", "warn")
        .write_style("LANCE_LOG_STYLE");
    let mut log_builder = env_logger::Builder::from_env(env);
    set_timestamp_precision(&mut log_builder);
    init_logging(log_builder);

    m.add_class::<FFILanceTableProvider>()?;
    m.add_class::<Scanner>()?;
    m.add_class::<Dataset>()?;
    m.add_class::<FileFragment>()?;
    m.add_class::<PyDeletionFile>()?;
    m.add_class::<PyRowIdMeta>()?;
    m.add_class::<MergeInsertBuilder>()?;
    m.add_class::<LanceBlobFile>()?;
    m.add_class::<LanceFileReader>()?;
    m.add_class::<LanceFileWriter>()?;
    m.add_class::<LanceFileSession>()?;
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
    m.add_class::<ScanStatistics>()?;
    m.add_class::<Session>()?;
    m.add_class::<PyTraceEvent>()?;
    m.add_class::<TraceGuard>()?;
    m.add_class::<schema::LanceSchema>()?;
    m.add_class::<PyFullTextQuery>()?;
    m.add_wrapped(wrap_pyfunction!(bfloat16_array))?;
    m.add_wrapped(wrap_pyfunction!(write_dataset))?;
    m.add_wrapped(wrap_pyfunction!(write_fragments))?;
    m.add_wrapped(wrap_pyfunction!(write_fragments_transaction))?;
    m.add_wrapped(wrap_pyfunction!(schema_to_json))?;
    m.add_wrapped(wrap_pyfunction!(json_to_schema))?;
    m.add_wrapped(wrap_pyfunction!(trace_to_chrome))?;
    m.add_wrapped(wrap_pyfunction!(capture_trace_events))?;
    m.add_wrapped(wrap_pyfunction!(shutdown_tracing))?;
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

#[pyclass(name = "FFILanceTableProvider", module = "lance", subclass)]
#[derive(Clone)]
struct FFILanceTableProvider {
    dataset: Arc<::lance::Dataset>,
    with_row_id: bool,
    with_row_addr: bool,
}

#[pymethods]
impl FFILanceTableProvider {
    #[new]
    #[pyo3(signature = (dataset, *, with_row_id = false, with_row_addr = false))]
    fn new(dataset: &Bound<'_, PyAny>, with_row_id: bool, with_row_addr: bool) -> PyResult<Self> {
        let py = dataset.py();
        let dataset = dataset.getattr("_ds")?.extract::<Py<Dataset>>()?;
        let dataset_ref = &dataset.bind(py).borrow().ds;
        // TODO: https://github.com/lancedb/lance/issues/3966 remove this workaround
        let _ = RT.block_on(Some(py), dataset_ref.load_indices())?;
        Ok(Self {
            dataset: dataset_ref.clone(),
            with_row_id,
            with_row_addr,
        })
    }

    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let name = CString::new("datafusion_table_provider").unwrap();
        let a_lance_table_provider = Arc::new(LanceTableProvider::new(
            self.dataset.clone(),
            self.with_row_id,
            self.with_row_addr,
        ));

        let ffi_provider =
            FFI_TableProvider::new(a_lance_table_provider, true, RT.get_runtime_handle());
        let capsule = PyCapsule::new(py, ffi_provider, Some(name.clone()));
        capsule
    }
}
