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

use arrow::pyarrow::FromPyArrow;
use arrow_array::{cast::AsArray, RecordBatch};
use arrow_schema::DataType;
use pyo3::{exceptions::PyValueError, prelude::*};

use lance_linalg::{
    distance::MetricType,
    kmeans::{KMeans as LanceKMeans, KMeansParams},
};

#[pyclass(name = "_KMeans")]
pub struct KMeans {
    /// Number of clusters
    k: usize,

    /// Metric type
    metric_type: MetricType,

    max_iters: usize,
}

#[pymethods]
impl KMeans {
    #[new]
    #[pyo3(signature = (k, metric_type="l2", max_iters=50))]
    fn new(k: usize, metric_type: &str, max_iters: usize) -> PyResult<Self> {
        Ok(Self {
            k,
            metric_type: metric_type.try_into().unwrap(),
            max_iters,
        })
    }

    /// Train the model
    fn fit(&mut self, reader: &PyAny) -> PyResult<()> {
        let batch = RecordBatch::from_pyarrow(reader)?;
        let Some(array) = batch.column_by_name("_kmeans_data") else {
            return Err(PyValueError::new_err("Missing column '_kmeans_data'"));
        };
        let dim: i32 = match array.data_type() {
            DataType::FixedSizeList(_, dim) => *dim,
            _ => {
                return Err(PyValueError::new_err(
                    "Column '_kmeans_data' must be a FixedSizeList",
                ))
            }
        };
        let fixed_size_arr = array.as_fixed_size_list();
        let params = KMeansParams {
            metric_type: self.metric_type,
            max_iters: self.max_iters as u32,
            ..Default::default()
        };
        let kmeans = LanceKMeans::new_with_params(fixed_size_arr, self.k, &params);
        Ok(())
    }

    fn centroids(&self) -> PyResult<PyObject> {
        Ok(Python::with_gil(|py| py.None()))
    }
}
