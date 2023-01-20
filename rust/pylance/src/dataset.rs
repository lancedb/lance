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
use arrow::pyarrow::*;
use arrow_schema::Schema as ArrowSchema;

use pyo3::prelude::*;
use pyo3::{pyclass, PyObject, PyResult};
use pyo3::exceptions::PyValueError;

use tokio::runtime::Runtime;

use ::lance::dataset::Dataset as LanceDataset;
use crate::Scanner;


/// Lance Dataset that will inherit from pyarrow dataset
/// to trick duckdb
#[pyclass(name="_Dataset")]
pub struct Dataset {
    #[pyo3(get)]
    uri: String,
    ds: Arc<LanceDataset>,
    rt: Arc<Runtime>
}

#[pymethods]
impl Dataset {
    #[new]
    fn new(uri: String) -> PyResult<Self> {
        let rt = Runtime::new()?;
        let dataset = rt.block_on(async { LanceDataset::open(uri.as_str()).await });
        match dataset {
            Ok(ds) => Ok(Self { uri, ds: Arc::new(ds), rt: Arc::new(rt) }),
            Err(err) => Err(PyValueError::new_err(err.to_string()))
        }
    }

    #[getter(schema)]
    fn schema(self_: PyRef<'_, Self>) -> PyResult<PyObject> {
        let arrow_schema = ArrowSchema::from(self_.ds.schema());
        arrow_schema.to_pyarrow(self_.py())
    }

    fn scanner(&self) -> PyResult<Scanner> {
        let scanner = Scanner::new(self.ds.clone(), self.rt.clone());
        Ok(scanner)
    }
}