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
use std::sync::Arc;

use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::*;
use arrow_schema::Schema as ArrowSchema;
use pyo3::{pyclass, PyObject, PyResult};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::runtime::Runtime;

use lance::dataset::{WriteMode, WriteParams};
use ::lance::dataset::Dataset as LanceDataset;
use crate::Scanner;


/// Lance Dataset that will be wrapped by another class in Python
#[pyclass(name = "_Dataset", module = "_lib")]
pub struct Dataset {
    #[pyo3(get)]
    uri: String,
    ds: Arc<LanceDataset>,
    rt: Arc<Runtime>,
}

#[pymethods]
impl Dataset {
    #[new]
    fn new(uri: String) -> PyResult<Self> {
        let rt = Runtime::new()?;
        let dataset = rt.block_on(async { LanceDataset::open(uri.as_str()).await });
        match dataset {
            Ok(ds) => Ok(Self {
                uri,
                ds: Arc::new(ds),
                rt: Arc::new(rt),
            }),
            Err(err) => Err(PyValueError::new_err(err.to_string())),
        }
    }

    #[getter(schema)]
    fn schema(self_: PyRef<'_, Self>) -> PyResult<PyObject> {
        let arrow_schema = ArrowSchema::from(self_.ds.schema());
        arrow_schema.to_pyarrow(self_.py())
    }

    fn scanner(
        &self,
        columns: Option<Vec<String>>,
        limit: i64,
        offset: Option<i64>,
    ) -> PyResult<Scanner> {
        let scanner = Scanner::new(self.ds.clone(), columns, offset, limit, self.rt.clone());
        Ok(scanner)
    }
}

#[pyfunction(name = "_write_dataset", module = "_lib")]
pub fn write_dataset(reader: &PyAny, uri: &str, options: &PyDict) -> PyResult<bool> {
    let mut reader = ArrowArrayStreamReader::from_pyarrow(reader)?;
    let params = if options.is_none() {
        None
    } else {
        let mut p = WriteParams::default();
        if let Some(mode) = options.get_item("mode") {
            match mode.to_string().to_lowercase().as_str() {
                "create" => Ok(WriteMode::Create),
                "append" => Ok(WriteMode::Append),
                "overwrite" => Ok(WriteMode::Overwrite),
                _ => Err(PyValueError::new_err(format!("Invalid mode {mode}"))),
            }?;
        }
        if let Some(maybe_nrows) = options.get_item("max_rows_per_file") {
            p.max_rows_per_file = usize::extract(maybe_nrows)?;
        }
        if let Some(maybe_nrows) = options.get_item("max_rows_per_group") {
            p.max_rows_per_group = usize::extract(maybe_nrows)?;
        }
        Some(p)
    };
    Runtime::new()?
        .block_on(async move { LanceDataset::create(&mut reader, uri, params).await })
        .map(|_| true)
        .map_err(|err| PyIOError::new_err(err.to_string()))
}
