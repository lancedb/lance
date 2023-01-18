// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Lance Columnar Data Format
//!
//! Lance columnar data format is an alternative to Parquet. It provides 100x faster for random access,
//! automatic versioning, optimized for computer vision, bioinformatics, spatial and ML data.
//! [Apache Arrow](https://arrow.apache.org/) and DuckDB compatible.

#[allow(unused_imports)]
use arrow::pyarrow::*;
use arrow_array::RecordBatch;
use futures::stream::StreamExt;
use object_store::path::Path;
use pyo3::prelude::*;
use tokio::runtime::Runtime;

use ::lance::dataset::Dataset;
use ::lance::datatypes::Schema;
use ::lance::io::{FileWriter, ObjectStore, object_writer::ObjectWriter};

#[pyfunction]
#[pyo3(pass_module)]
fn read_batch(m: &PyModule, uri: &str) -> PyResult<PyObject> {
    let rt = Runtime::new().unwrap();
    let batch : RecordBatch = rt.block_on(async {
        let dataset = Dataset::open(uri).await.unwrap();
        let mut stream = dataset.scan().into_stream();
        let batch = stream.next().await.unwrap().unwrap();
        batch
    });
    let rs = batch.to_pyarrow(m.py());
    rs
}

#[pyfunction]
fn write_batch(py_batch: &PyAny, uri: &str) -> PyResult<bool> {
    let rt = Runtime::new().unwrap();
    let object_store = ObjectStore::new(uri).unwrap();
    let path = Path::from(uri);
    let batch: RecordBatch = RecordBatch::from_pyarrow(py_batch).unwrap();
    let arrow_schema = batch.schema();
    let schema = Schema::try_from(arrow_schema.as_ref()).unwrap();
    let rs = Ok(rt.block_on(async {
        let object_writer = ObjectWriter::new(&object_store, &path).await.unwrap();
        let mut file_writer = FileWriter::new(object_writer, &schema);
        file_writer.write(&batch).await.unwrap();
        file_writer.finish().await
    }).is_ok());
    rs
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn lance(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_batch, m)?)?;
    m.add_function(wrap_pyfunction!(write_batch, m)?)?;
    Ok(())
}