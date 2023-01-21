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

use arrow::ffi_stream::*;
use arrow::pyarrow::*;
use arrow_schema::Schema as ArrowSchema;
use pyo3::prelude::*;
use pyo3::{pyclass, PyObject, PyResult};
use pyo3::exceptions::PyValueError;
use tokio::runtime::Runtime;

use ::lance::dataset::scanner::Scanner as LanceScanner;
use ::lance::dataset::Dataset as LanceDataset;

use crate::errors::ioerror;
use crate::reader::LanceReader;


/// This will be wrapped by a python class to provide
/// additional functionality
#[pyclass(name = "_Scanner", module = "_lib")]
pub struct Scanner {
    dataset: Arc<LanceDataset>,
    columns: Option<Vec<String>>,
    offset: Option<i64>,
    limit: i64,
    rt: Arc<Runtime>,
}

impl Scanner {
    pub fn new(
        dataset: Arc<LanceDataset>,
        columns: Option<Vec<String>>,
        offset: Option<i64>,
        limit: i64,
        rt: Arc<Runtime>,
    ) -> Self {
        Self {
            dataset,
            columns,
            offset,
            limit,
            rt,
        }
    }
}

#[pymethods]
impl Scanner {
    #[getter(dataset_schema)]
    fn dataset_schema(self_: PyRef<'_, Self>) -> PyResult<PyObject> {
        ArrowSchema::from(self_.dataset.schema()).to_pyarrow(self_.py())
    }

    fn to_reader(self_: PyRef<'_, Self>) -> PyResult<PyObject> {
        self_.rt.block_on(async {
            let mut scanner: LanceScanner = self_.dataset.scan();
            if let Some(c) = &self_.columns {
                let proj: Vec<&str> = c.iter().map(|s| s.as_str()).collect();
                scanner
                    .project(&proj)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
            }
            scanner.limit(self_.limit, self_.offset);
            let reader = LanceReader::new(scanner, self_.rt.clone());
            // Export a `RecordBatchReader` through `FFI_ArrowArrayStream`
            let stream = Arc::new(FFI_ArrowArrayStream::empty());
            let stream_ptr = Arc::into_raw(stream) as *mut FFI_ArrowArrayStream;
            unsafe {
                export_reader_into_raw(Box::new(reader), stream_ptr);
                match ArrowArrayStreamReader::from_raw(stream_ptr) {
                    Ok(reader) => reader.to_pyarrow(self_.py()),
                    Err(err) => Err(ioerror(self_.py(), err.to_string())),
                }
            }
        })
    }
}
