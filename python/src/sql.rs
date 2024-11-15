// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::pyarrow::PyArrowType;
use arrow_array::RecordBatchReader;
use pyo3::{pyclass, pymethods, PyRef, PyResult};

use lance::datafusion::sql::SqlPlan;

use crate::{error::PythonErrorExt, LanceReader, RT};

#[pyclass]
pub struct SqlQueryBuilder {
    pub query: String,
}

#[pymethods]
impl SqlQueryBuilder {
    #[new]
    pub fn new(query: String) -> Self {
        Self { query }
    }

    fn with_lance_dataset(self: PyRef<'_, Self>) -> PyRef<'_, Self> {
        self
    }

    fn execute(self_: PyRef<'_, Self>) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        let query = SqlPlan::new(self_.query.clone());
        let reader = RT
            .spawn(Some(self_.py()), async move {
                Ok(LanceReader::from_stream(query.execute().await?))
            })?
            .infer_error()?;

        Ok(PyArrowType(Box::new(reader)))
    }
}
