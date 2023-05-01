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

use arrow::pyarrow::PyArrowConvert;
use std::sync::Arc;

use lance::dataset::fragment::FileFragment as LanceFragment;
use pyo3::exceptions::*;
use pyo3::prelude::*;

use crate::Scanner;

#[pyclass(name = "_Fragment", module = "_lib")]
#[derive(Clone)]
pub struct FileFragment {
    fragment: LanceFragment,
}

impl FileFragment {
    pub fn new(frag: LanceFragment) -> Self {
        Self { fragment: frag }
    }
}

#[pymethods]
impl FileFragment {
    fn __repr__(&self) -> String {
        format!("LanceFileFragment(id={})", self.fragment.id())
    }

    fn id(&self) -> usize {
        self.fragment.id()
    }

    fn count_rows(&self, _filter: Option<String>) -> PyResult<usize> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            self.fragment
                .count_rows()
                .await
                .map_err(|e| PyIOError::new_err(e.to_string()))
        })
    }

    fn take(
        self_: PyRef<'_, Self>,
        row_indices: Vec<usize>,
        columns: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let rt = Arc::new(tokio::runtime::Runtime::new()?);
        let dataset_schema = self_.fragment.dataset().schema();
        let projection = if let Some(columns) = columns {
            dataset_schema
                .project(&columns)
                .map_err(|e| PyIOError::new_err(e.to_string()))?
        } else {
            dataset_schema.clone()
        };

        let indices = row_indices.iter().map(|v| *v as u32).collect::<Vec<_>>();
        let batch = rt
            .block_on(async { self_.fragment.take(&indices, &projection).await })
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        batch.to_pyarrow(self_.py())
    }

    fn scanner(
        self_: PyRef<'_, Self>,
        columns: Option<Vec<String>>,
        filter: Option<String>,
        limit: Option<i64>,
        offset: Option<i64>,
    ) -> PyResult<Scanner> {
        let rt = Arc::new(tokio::runtime::Runtime::new()?);
        let mut scanner = self_.fragment.scan();
        if let Some(cols) = columns {
            scanner
                .project(&cols)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        if let Some(f) = filter {
            scanner
                .filter(&f)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        if let Some(l) = limit {
            scanner
                .limit(l, offset)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        let scn = Arc::new(scanner);
        Ok(Scanner::new(scn, rt))
    }

    fn updater(self_: PyRef<'_, Self>) -> PyResult<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async { self_.fragment.updater().await })
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }
}
