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

use std::sync::{Arc, Mutex};

use arrow::compute::concat;
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::{cast::AsArray, Array, FixedSizeListArray, Float32Array, UInt32Array};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use lance::utils::GenericProgressCallback;
use lance::Result;
use lance::{datatypes::Schema, io::ObjectStore};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::utils::progress::NoopProgressCallback;
use lance_file::writer::FileWriter;
use lance_index::scalar::IndexWriter;
use lance_index::vector::hnsw::{builder::HnswBuildParams, HNSW};
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_linalg::kmeans::compute_partitions;
use lance_linalg::{
    distance::DistanceType,
    kmeans::{KMeans as LanceKMeans, KMeansParams},
};
use lance_table::io::manifest::ManifestDescribing;
use log::warn;
use object_store::path::Path;
use pyo3::types::PyDict;
use pyo3::{
    exceptions::{PyIOError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyIterator,
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
        let kmeans = LanceKMeans::new_with_params(
            &fixed_size_arr,
            self.k,
            &params,
            &NoopProgressCallback::default(),
        )
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
        distance_type="l2",
    ))]
    fn build(
        vectors_array: &PyIterator,
        max_level: u16,
        m: usize,
        ef_construction: usize,
        distance_type: &str,
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

        let dt = DistanceType::try_from(distance_type)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let hnsw = RT
            .runtime
            .block_on(params.build(vectors.clone(), dt))
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

#[derive(Default)]
pub struct TqdmProgressCallback {
    progress: Arc<Mutex<Option<PyObject>>>,
}

impl TqdmProgressCallback {
    fn do_begin(&self, total_units: u64) -> PyResult<()> {
        let mut prog = self.progress.lock().unwrap();
        let prog_obj = Python::with_gil(|py| {
            let tqdm = py.import("tqdm")?;
            let tqdm = tqdm.getattr("tqdm")?;
            let kwargs = vec![("total", total_units.to_object(py))].to_object(py);
            let kwargs = PyDict::from_sequence(py, kwargs)?;
            let progress = tqdm.call((), Some(kwargs))?;
            PyResult::Ok(progress.to_object(py))
        })?;
        *prog = Some(prog_obj);
        Ok(())
    }

    fn do_update(&self, new_units_completed: u64) -> PyResult<()> {
        let prog = self.progress.lock().unwrap();
        if let Some(prog) = prog.as_ref() {
            Python::with_gil(|py| {
                prog.call_method1(py, "update", (new_units_completed,))
                    .unwrap();
            });
        }
        Ok(())
    }
}

impl GenericProgressCallback for TqdmProgressCallback {
    fn begin(&self, total_units: u64) {
        if let Err(err) = self.do_begin(total_units) {
            warn!(
                "Failed to report progress, is tqdm installed?  Error: {}",
                err
            );
        }
    }

    fn update(&self, new_units_completed: u64) {
        if let Err(err) = self.do_update(new_units_completed) {
            warn!("Failed to report progress.  Error: {}", err);
        }
    }
}
