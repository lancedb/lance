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

use arrow::compute::{concat, concat_batches};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::{
    cast::AsArray, Array, FixedSizeListArray, Float32Array, UInt32Array, UInt64Array,
};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use lance::Result;
use lance::{datatypes::Schema, index::vector::sq, io::ObjectStore};
use lance_arrow::FixedSizeListArrayExt;
use lance_file::writer::FileWriter;
use lance_index::scalar::IndexWriter;
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_index::vector::{
    hnsw::{builder::HnswBuildParams, HNSW},
    storage::VectorStore,
};
use lance_linalg::kmeans::compute_partitions;
use lance_linalg::{
    distance::DistanceType,
    kmeans::{KMeans as LanceKMeans, KMeansParams},
};
use lance_table::io::manifest::ManifestDescribing;
use object_store::path::Path;
use pyo3::{
    exceptions::{PyIOError, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyIterator, PyTuple},
};

use crate::RT;

#[pyclass(name = "_KMeans")]
pub struct KMeans {
    /// Number of clusters
    k: usize,

    /// Metric type
    metric_type: DistanceType,

    max_iters: u32,

    /// A trained KMean model. This is set after calling `fit`.
    trained_kmeans: Option<LanceKMeans>,
}

#[pymethods]
impl KMeans {
    #[new]
    #[pyo3(signature = (k, metric_type="l2", max_iters=50))]
    fn new(k: usize, metric_type: &str, max_iters: u32) -> PyResult<Self> {
        Ok(Self {
            k,
            metric_type: metric_type.try_into().unwrap(),
            max_iters,
            trained_kmeans: None,
        })
    }

    /// Train the model
    fn fit(&mut self, _py: Python, arr: &PyAny) -> PyResult<()> {
        let data = ArrayData::from_pyarrow(arr)?;
        if !matches!(data.data_type(), DataType::FixedSizeList(_, _)) {
            return Err(PyValueError::new_err("Must be a FixedSizeList"));
        }
        let fixed_size_arr = FixedSizeListArray::from(data);
        let params = KMeansParams {
            distance_type: self.metric_type,
            max_iters: self.max_iters,
            ..Default::default()
        };
        let kmeans = LanceKMeans::new_with_params(&fixed_size_arr, self.k, &params)
            .map_err(|e| PyRuntimeError::new_err(format!("Error training KMeans: {}", e)))?;
        self.trained_kmeans = Some(kmeans);
        Ok(())
    }

    fn predict(&self, py: Python, array: &PyAny) -> PyResult<PyObject> {
        let Some(kmeans) = self.trained_kmeans.as_ref() else {
            return Err(PyRuntimeError::new_err("KMeans must fit (train) first"));
        };
        let data = ArrayData::from_pyarrow(array)?;
        if !matches!(data.data_type(), DataType::FixedSizeList(_, _)) {
            return Err(PyValueError::new_err("Must be a FixedSizeList"));
        }
        let fixed_size_arr = FixedSizeListArray::from(data);
        if kmeans.dimension != fixed_size_arr.value_length() as usize {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: kmean model {} != data {}",
                kmeans.dimension,
                fixed_size_arr.value_length()
            )));
        };
        if !matches!(fixed_size_arr.value_type(), DataType::Float32) {
            return Err(PyValueError::new_err("Must be a FixedSizeList of Float32"));
        };
        let values: Arc<Float32Array> = fixed_size_arr.values().as_primitive().clone().into();
        let centroids: &Float32Array = kmeans.centroids.as_primitive();
        let cluster_ids = UInt32Array::from(compute_partitions(
            centroids.values(),
            values.values(),
            kmeans.dimension,
            kmeans.distance_type,
        ));
        cluster_ids.into_data().to_pyarrow(py)
    }

    fn centroids(&self, py: Python) -> PyResult<PyObject> {
        if let Some(kmeans) = self.trained_kmeans.as_ref() {
            let centroids: Float32Array = kmeans.centroids.as_primitive().clone();
            let fixed_size_arr =
                FixedSizeListArray::try_new_from_values(centroids, kmeans.dimension as i32)
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "Error converting centroids to FixedSizeListArray: {}",
                            e
                        ))
                    })?;
            fixed_size_arr.into_data().to_pyarrow(py)
        } else {
            Ok(py.None())
        }
    }
}

#[pyclass(name = "_Hnsw")]
pub struct Hnsw {
    hnsw: lance_index::vector::hnsw::HNSW,
    vectors: Arc<dyn Array>,
}

#[pymethods]
impl Hnsw {
    #[staticmethod]
    #[pyo3(signature = (
        vectors_array,
        max_level=7,
        m=20,
        ef_construction=100,
    ))]
    fn build(
        vectors_array: &PyIterator,
        max_level: u16,
        m: usize,
        ef_construction: usize,
    ) -> PyResult<Self> {
        let params = HnswBuildParams::default()
            .max_level(max_level)
            .num_edges(m)
            .ef_construction(ef_construction);

        let mut data: Vec<Arc<dyn Array>> = Vec::new();
        for vectors in vectors_array {
            let vectors = ArrayData::from_pyarrow(vectors?)?;
            if !matches!(vectors.data_type(), DataType::FixedSizeList(_, _)) {
                return Err(PyValueError::new_err("Must be a FixedSizeList"));
            }
            data.push(Arc::new(FixedSizeListArray::from(vectors)));
        }
        let array_refs = data.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
        let vectors = concat(&array_refs).map_err(|e| PyIOError::new_err(e.to_string()))?;
        std::mem::drop(data);

        let hnsw = RT
            .runtime
            .block_on(params.build(vectors.clone()))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self { hnsw, vectors })
    }

    #[pyo3(signature = (file_path))]
    fn to_lance_file(&self, py: Python, file_path: &str) -> PyResult<()> {
        let object_store = ObjectStore::local();
        let path = Path::parse(file_path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        let mut writer = RT
            .block_on(
                Some(py),
                FileWriter::<ManifestDescribing>::try_new(
                    &object_store,
                    &path,
                    Schema::try_from(HNSW::schema().as_ref())
                        .map_err(|e| PyIOError::new_err(e.to_string()))?,
                    &Default::default(),
                ),
            )?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        RT.block_on(Some(py), async {
            let batch = self.hnsw.to_batch()?;
            let metadata = batch.schema_ref().metadata().clone();
            writer.write_record_batch(batch).await?;
            writer.finish_with_metadata(&metadata).await?;
            Result::Ok(())
        })?
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    fn vectors(&self, py: Python) -> PyResult<PyObject> {
        self.vectors.to_data().to_pyarrow(py)
    }
}

#[pyfunction(name = "_build_sq_storage")]
pub fn build_sq_storage(
    py: Python,
    row_ids_array: &PyIterator,
    vectors: &PyAny,
    dim: usize,
    bounds: &PyTuple,
) -> PyResult<PyObject> {
    let mut row_ids_arr: Vec<Arc<dyn Array>> = Vec::new();
    for row_ids in row_ids_array {
        let row_ids = ArrayData::from_pyarrow(row_ids?)?;
        if !matches!(row_ids.data_type(), DataType::UInt64) {
            return Err(PyValueError::new_err("Must be a UInt64"));
        }
        row_ids_arr.push(Arc::new(UInt64Array::from(row_ids)));
    }
    let row_ids_refs = row_ids_arr.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
    let row_ids = concat(&row_ids_refs).map_err(|e| PyIOError::new_err(e.to_string()))?;
    std::mem::drop(row_ids_arr);

    let vectors = Arc::new(FixedSizeListArray::from(ArrayData::from_pyarrow(vectors)?));

    let lower_bound = bounds.get_item(0)?.extract::<f64>()?;
    let upper_bound = bounds.get_item(1)?.extract::<f64>()?;
    let quantizer =
        lance_index::vector::sq::ScalarQuantizer::with_bounds(8, dim, lower_bound..upper_bound);
    let storage = sq::build_sq_storage(DistanceType::L2, row_ids, vectors, quantizer)
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    let batches = storage
        .to_batches()
        .map_err(|e| PyIOError::new_err(e.to_string()))?
        .collect::<Vec<_>>();
    let batch = concat_batches(&batches[0].schema(), &batches)
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    batch.to_pyarrow(py)
}
