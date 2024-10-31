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
use arrow_array::RecordBatchReader;
use lance::dataset::scanner::stats::ScannerStats;
use lance::dataset::scanner::stats::ThroughputUnit;
use pyo3::prelude::*;
use pyo3::pyclass;

use ::lance::dataset::scanner::Scanner as LanceScanner;
use pyo3::exceptions::PyValueError;

use crate::reader::LanceReader;
use crate::RT;

#[pyclass]
pub struct LanceScanStats {
    inner: ScannerStats,
}

impl LanceScanStats {
    pub fn new(inner: ScannerStats) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl LanceScanStats {
    /// The start of the scan
    ///
    /// This is when the stream is constructed, not when it is first consumed.
    ///
    /// Returned as milliseconds since the UNIX epoch
    #[getter]
    fn start(&self) -> PyResult<u64> {
        Ok(self
            .inner
            .start
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
            .try_into()
            .unwrap())
    }

    /// The end of the scan
    ///
    /// This is when the last batch is provided to the consumer which may be
    /// well after the I/O has finished (if there is a slow consumer or expensive
    /// decode).
    ///
    /// Returned as milliseconds since the UNIX epoch
    #[getter]
    fn end(&self) -> PyResult<u64> {
        Ok(self
            .inner
            .end
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
            .try_into()
            .unwrap())
    }

    /// The wall clock duration of the scan
    ///
    /// NOTE: This is not the time that the scanner was actually doing work, and not the amount
    /// of time spent in I/O but simply the time from when the scanner was created to when the
    /// last batch was provided to the consumer.
    ///
    /// As an example, if a consumer is slow to consume the data (e.g. they are writing the data
    /// back out to disk or doing expensive processing) then this will be much larger than the
    /// actual time to read the data.
    ///
    /// Returned as floating point seconds
    #[getter]
    fn wall_clock_duration(&self) -> PyResult<f64> {
        Ok(self.inner.wall_clock_duration.as_secs_f64())
    }

    /// This is an estimate of the "wall clock throughput" in GiB/s
    ///
    /// Note: this is based both on :ref:`wall_clock_duration` (see note on that method) and
    /// :ref:`estimated_output_bytes` (see note on that field).
    ///
    /// It is not safe, for example, to assume that this is the rate at which data was pulled down
    /// from storage.
    ///
    /// Returned as floating point GiB/s
    #[getter]
    fn wall_clock_throughput(&self) -> PyResult<f64> {
        Ok(self.inner.wall_clock_throughput().gigabytes_per_second())
    }

    /// The number of rows output by the scanner
    #[getter]
    fn output_rows(&self) -> PyResult<u64> {
        Ok(self.inner.output_rows)
    }

    /// The estimated size of the output in bytes
    ///
    /// "Estimated" is used here because there may be some instances where multiple
    /// batches will share the same underlying buffer (e.g. a dictionary) and so the
    /// actual data size may be less than the reported size.
    ///
    /// Also, this is very different than "input bytes" which may be much smaller since
    /// the input may be compressed or encoded.
    ///
    /// This will always be greater than or equal to the actual size.
    #[getter]
    fn estimated_output_bytes(&self) -> PyResult<u64> {
        Ok(self.inner.estimated_output_bytes)
    }

    /// The plan that was used to generate the scan
    ///
    /// There are some instances where we generate a scan without a plan and some handlers
    /// do not need the plan and so we may not gather it.  In these cases this will be None.
    #[getter]
    fn plan(&self) -> PyResult<Option<String>> {
        Ok(self.inner.plan.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "LanceScanStats(start={}, end={}, wall_clock_duration={}, wall_clock_throughput={}, output_rows={}, estimated_output_bytes={}, plan={:?})",
            self.start().unwrap(),
            self.end().unwrap(),
            self.wall_clock_duration().unwrap(),
            self.wall_clock_throughput().unwrap(),
            self.output_rows().unwrap(),
            self.estimated_output_bytes().unwrap(),
            self.plan().unwrap()
        )
    }
}

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
        let scanner = self_.scanner.clone();
        RT.spawn(Some(self_.py()), async move { scanner.schema().await })?
            .map(|s| s.to_pyarrow(self_.py()))
            .map_err(|err| PyValueError::new_err(err.to_string()))?
    }

    #[pyo3(signature = (*, verbose = false))]
    fn explain_plan(self_: PyRef<'_, Self>, verbose: bool) -> PyResult<String> {
        let scanner = self_.scanner.clone();
        let res = RT
            .spawn(Some(self_.py()), async move {
                scanner.explain_plan(verbose).await
            })?
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        Ok(res)
    }

    fn count_rows(self_: PyRef<'_, Self>) -> PyResult<u64> {
        let scanner = self_.scanner.clone();
        RT.spawn(Some(self_.py()), async move { scanner.count_rows().await })?
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn to_pyarrow(
        self_: PyRef<'_, Self>,
    ) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        let scanner = self_.scanner.clone();
        let reader = RT
            .spawn(Some(self_.py()), async move {
                LanceReader::try_new(scanner).await
            })?
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        Ok(PyArrowType(Box::new(reader)))
    }
}
