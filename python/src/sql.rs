// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use arrow::pyarrow::PyArrowType;
use arrow_array::RecordBatchReader;
use pyo3::{pyclass, pymethods, PyRef, PyResult};

use lance::datafusion::sql::SqlPlan;
use lance::Dataset as LanceDataset;

use crate::{error::PythonErrorExt, Dataset, LanceReader, RT};

struct QueryBuilderState {
    datasets: HashMap<String, Arc<LanceDataset>>,
}

#[pyclass]
pub struct SqlQueryBuilder {
    pub query: String,
    state: Arc<Mutex<QueryBuilderState>>,
}

#[pymethods]
impl SqlQueryBuilder {
    #[new]
    pub fn new(query: String) -> Self {
        Self {
            query,
            state: Arc::new(Mutex::new(QueryBuilderState {
                datasets: HashMap::new(),
            })),
        }
    }

    fn with_dataset<'a>(slf: PyRef<'a, Self>, alias: String, dataset: &Dataset) -> PyRef<'a, Self> {
        {
            let mut state = slf.state.lock().unwrap();
            state.datasets.insert(alias, dataset.ds.clone());
        }
        slf
    }

    fn execute(slf: PyRef<'_, Self>) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        let context = {
            let state = slf.state.lock().unwrap();
            state.datasets.clone()
        };
        let query = SqlPlan::new(slf.query.clone(), context);
        let reader = RT
            .spawn(Some(slf.py()), async move {
                Ok(LanceReader::from_stream(query.execute().await?))
            })?
            .infer_error()?;

        Ok(PyArrowType(Box::new(reader)))
    }
}
