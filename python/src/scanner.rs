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
use pyo3::prelude::*;
use pyo3::{pyclass, PyObject, PyResult};

use ::lance::dataset::scanner::Scanner as LanceScanner;
use pyo3::exceptions::PyValueError;

use crate::errors::ioerror;
use crate::reader::LanceReader;
use crate::RT;

/// This will be wrapped by a python class to provide
/// additional functionality
#[pyclass(name = "_Scanner", module = "_lib")]
#[derive(Clone)]
pub struct Scanner {
    scanner: Arc<LanceScanner>,
}

impl Scanner {
    pub fn new(scanner: Arc<LanceScanner>) -> Self {
        Self { scanner }
    }

    pub(crate) async fn to_reader(&self) -> ::lance::error::Result<LanceReader> {
        LanceReader::try_new(self.scanner.clone()).await
    }
}

#[pymethods]
impl Scanner {
    #[getter(schema)]
    fn schema(self_: PyRef<'_, Self>) -> PyResult<PyObject> {
        self_
            .scanner
            .schema()
            .map(|s| s.to_pyarrow(self_.py()))
            .map_err(|err| PyValueError::new_err(err.to_string()))?
    }

    #[pyo3(signature = (*, verbose = false))]
    fn explain_plan(self_: PyRef<'_, Self>, verbose: bool) -> PyResult<String> {
        let scanner = self_.scanner.clone();
        let res = RT
            .spawn(Some(self_.py()), async move {
                scanner.explain_plan(verbose).await
            })
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        Ok(res)
    }

    fn count_rows(self_: PyRef<'_, Self>) -> PyResult<u64> {
        let scanner = self_.scanner.clone();
        RT.spawn(Some(self_.py()), async move { scanner.count_rows().await })
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn to_pyarrow(self_: PyRef<'_, Self>) -> PyResult<PyObject> {
        let scanner = self_.scanner.clone();
        let reader = RT
            .spawn(Some(self_.py()), async move {
                LanceReader::try_new(scanner).await
            })
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        // Export a `RecordBatchReader` through `FFI_ArrowArrayStream`
        let mut ffi_stream = FFI_ArrowArrayStream::empty();
        unsafe {
            export_reader_into_raw(Box::new(reader), &mut ffi_stream);
            match ArrowArrayStreamReader::from_raw(&mut ffi_stream) {
                Ok(reader) => reader.into_pyarrow(self_.py()),
                Err(err) => Err(ioerror(self_.py(), err.to_string())),
            }
        }
    }
}
