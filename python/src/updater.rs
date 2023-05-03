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

use std::sync::Arc;

use arrow::pyarrow::PyArrowConvert;
use arrow_array::RecordBatch;
use pyo3::{exceptions::*, prelude::*};

use lance::dataset::fragment::FileFragment as LanceFragment;
use lance::dataset::updater::Updater as LanceUpdater;

use crate::fragment::FragmentMetadata;

use super::fragment::FileFragment;

#[pyclass(name = "_Updater", module = "_lib")]
pub struct Updater {
    inner: LanceUpdater,
    rt: tokio::runtime::Runtime,
}

impl Updater {
    pub(super) fn new(updater: LanceUpdater) -> Self {
        Self {
            inner: updater,
            rt: tokio::runtime::Runtime::new().unwrap(),
        }
    }
}

#[pymethods]
impl Updater {
    /// Return the next batch as input data.
    #[pyo3(signature=())]
    fn next(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let batch = {
            self.rt
                .block_on(async { self.inner.next().await })
                .map_err(|err| PyIOError::new_err(err.to_string()))?
        };
        if let Some(batch) = batch {
            Ok(Some(batch.to_pyarrow(py)?))
        } else {
            Ok(None)
        }
    }

    /// Update one batch
    fn update(&mut self, batch: &PyAny) -> PyResult<()> {
        let batch = RecordBatch::from_pyarrow(batch)?;
        self.rt.block_on(async {
            self.inner
                .update(&batch)
                .await
                .map_err(|e| PyIOError::new_err(e.to_string()))
        })
    }

    fn finish(&mut self) -> PyResult<FragmentMetadata> {
        let fragment = self.rt.block_on(async {
            self.inner
                .finish()
                .await
                .map_err(|e| PyIOError::new_err(e.to_string()))
        })?;

        Ok(FragmentMetadata::new(fragment))
    }
}
