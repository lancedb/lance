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
use arrow_array::{Float32Array, RecordBatchReader};
use arrow_data::ArrayData;
use arrow_schema::Schema as ArrowSchema;
use pyo3::exceptions::{PyIOError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyLong};
use pyo3::{pyclass, PyObject, PyResult};
use tokio::runtime::Runtime;

use crate::Scanner;
use ::lance::dataset::scanner::Scanner as LanceScanner;
use ::lance::dataset::Dataset as LanceDataset;
use lance::dataset::{Version, WriteMode, WriteParams};

const DEFAULT_NPROBS: usize = 1;

/// Lance Dataset that will be wrapped by another class in Python
#[pyclass(name = "_Dataset", module = "_lib")]
#[derive(Clone)]
pub struct Dataset {
    #[pyo3(get)]
    uri: String,
    ds: Arc<LanceDataset>,
    rt: Arc<Runtime>,
}

#[pymethods]
impl Dataset {
    #[new]
    fn new(uri: String, version: Option<u64>) -> PyResult<Self> {
        let rt = Runtime::new()?;
        let dataset = rt.block_on(async {
            if let Some(ver) = version {
                LanceDataset::checkout(uri.as_str(), ver).await
            } else {
                LanceDataset::open(uri.as_str()).await
            }
        });
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
        self_: PyRef<'_, Self>,
        columns: Option<Vec<String>>,
        limit: i64,
        offset: Option<i64>,
        nearest: Option<&PyDict>,
    ) -> PyResult<Scanner> {
        let mut scanner: LanceScanner = self_.ds.scan();
        if let Some(c) = columns {
            let proj: Vec<&str> = c.iter().map(|s| s.as_str()).collect();
            scanner
                .project(&proj)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        scanner
            .limit(limit, offset)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        if let Some(nearest) = nearest {
            let column = nearest
                .get_item("column")
                .ok_or_else(|| PyKeyError::new_err("Need column for nearest"))?
                .to_string();

            let qval = nearest
                .get_item("q")
                .ok_or_else(|| PyKeyError::new_err("Need q for nearest"))?;
            let data = ArrayData::from_pyarrow(qval)?;
            let q = Float32Array::from(data);

            let k: usize = if let Some(k) = nearest.get_item("k") {
                if k.is_none() {
                    10
                } else {
                    PyAny::downcast::<PyLong>(k)?.extract()?
                }
            } else {
                10
            };

            let nprobes: usize = if let Some(nprobes) = nearest.get_item("nprobes") {
                if nprobes.is_none() {
                    DEFAULT_NPROBS
                } else {
                    PyAny::downcast::<PyLong>(nprobes)?.extract()?
                }
            } else {
                DEFAULT_NPROBS
            };

            // When refine factor is specified, a final Refine stage will be added to the I/O plan,
            // and use Flat index over the raw vectors to refine the results.
            // By default, `refine_factor` is None to not involve extra I/O exec node and random access.
            let refine_factor: Option<u32> = if let Some(rf) = nearest.get_item("refine_factor") {
                if rf.is_none() {
                    None
                } else {
                    PyAny::downcast::<PyLong>(rf)?.extract()?
                }
            } else {
                None
            };
            scanner
                .nearest(column.as_str(), &q, k)
                .map(|s| {
                    let mut s = s.nprobs(nprobes);
                    if let Some(factor) = refine_factor {
                        s = s.refine(factor);
                    }
                    s
                })
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }

        let scn = Arc::new(scanner);
        Ok(Scanner::new(scn, self_.rt.clone()))
    }

    fn count_rows(&self) -> PyResult<usize> {
        self.rt.block_on(async {
            Ok(self
                .ds
                .count_rows()
                .await
                .map_err(|err| PyIOError::new_err(err.to_string()))?)
        })
    }

    fn versions(self_: PyRef<'_, Self>) -> PyResult<Vec<PyObject>> {
        let versions = self_
            .list_versions()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Python::with_gil(|py| {
            let pyvers: Vec<PyObject> = versions
                .iter()
                .map(|v| {
                    let dict = PyDict::new(py);
                    dict.set_item("version", v.version).unwrap();
                    dict.set_item("timestamp", v.timestamp.timestamp()).unwrap();
                    let tup: Vec<(&String, &String)> = v.metadata.iter().collect();
                    dict.set_item("metadata", tup.into_py_dict(py)).unwrap();
                    dict.to_object(py)
                })
                .collect::<Vec<_>>()
                .into_iter()
                .collect();
            Ok(pyvers)
        })
    }
}

impl Dataset {
    fn list_versions(&self) -> ::lance::error::Result<Vec<Version>> {
        self.rt.block_on(async { self.ds.versions().await })
    }
}

#[pyfunction(name = "_write_dataset", module = "_lib")]
pub fn write_dataset(reader: &PyAny, uri: &str, options: &PyDict) -> PyResult<bool> {
    let params = get_write_params(options)?;
    Runtime::new()?.block_on(async move {
        let mut batches: Box<dyn RecordBatchReader> = if reader.is_instance_of::<Scanner>()? {
            let scanner: Scanner = reader.extract()?;
            Box::new(
                scanner
                    .to_reader()
                    .await
                    .map_err(|err| PyValueError::new_err(err.to_string()))?,
            )
        } else {
            Box::new(ArrowArrayStreamReader::from_pyarrow(reader)?)
        };

        LanceDataset::write(&mut batches, uri, params)
            .await
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Ok(true)
    })
}

fn get_write_params(options: &PyDict) -> PyResult<Option<WriteParams>> {
    let params = if options.is_none() {
        None
    } else {
        let mut p = WriteParams::default();
        if let Some(mode) = options.get_item("mode") {
            p.mode = match mode.to_string().to_lowercase().as_str() {
                "create" => Ok(WriteMode::Create),
                "append" => Ok(WriteMode::Append),
                "overwrite" => Ok(WriteMode::Overwrite),
                _ => Err(PyValueError::new_err(format!("Invalid mode {mode}"))),
            }?;
        };
        if let Some(maybe_nrows) = options.get_item("max_rows_per_file") {
            p.max_rows_per_file = usize::extract(maybe_nrows)?;
        }
        if let Some(maybe_nrows) = options.get_item("max_rows_per_group") {
            p.max_rows_per_group = usize::extract(maybe_nrows)?;
        }
        Some(p)
    };
    Ok(params)
}
