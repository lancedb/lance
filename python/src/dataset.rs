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

use std::str;
use std::sync::Arc;

use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::*;
use arrow_array::{Float32Array, RecordBatch};
use arrow_data::ArrayData;
use arrow_schema::Schema as ArrowSchema;
use lance::arrow::as_fixed_size_list_array;
use lance::dataset::fragment::FileFragment as LanceFileFragment;
use lance::dataset::ReadParams;
use lance::datatypes::Schema;
use lance::format::Fragment;
use lance::index::vector::ivf::IvfBuildParams;
use lance::index::vector::pq::PQBuildParams;
use lance::index::DatasetIndexExt;
use lance::io::object_store::ObjectStoreParams;
use pyo3::exceptions::{PyIOError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBool, PyDict, PyFloat, PyInt, PyLong};
use pyo3::{pyclass, PyObject, PyResult};

use crate::fragment::{FileFragment, FragmentMetadata};
use crate::Scanner;
use crate::RT;
use lance::dataset::{
    scanner::Scanner as LanceScanner, Dataset as LanceDataset, Version, WriteMode, WriteParams,
};
use lance::index::{
    vector::diskann::DiskANNParams,
    vector::{MetricType, VectorIndexParams},
    IndexType,
};

use self::commit::PyCommitLock;

pub mod commit;

const DEFAULT_NPROBS: usize = 1;
const DEFAULT_INDEX_CACHE_SIZE: usize = 256;
const DEFAULT_METADATA_CACHE_SIZE: usize = 256;

/// Lance Dataset that will be wrapped by another class in Python
#[pyclass(name = "_Dataset", module = "_lib")]
#[derive(Clone)]
pub struct Dataset {
    #[pyo3(get)]
    uri: String,
    ds: Arc<LanceDataset>,
}

#[pymethods]
impl Dataset {
    #[new]
    fn new(
        uri: String,
        version: Option<u64>,
        block_size: Option<usize>,
        index_cache_size: Option<usize>,
        metadata_cache_size: Option<usize>,
        commit_handler: Option<PyObject>,
    ) -> PyResult<Self> {
        let mut params = ReadParams {
            block_size,
            index_cache_size: index_cache_size.unwrap_or(DEFAULT_INDEX_CACHE_SIZE),
            metadata_cache_size: metadata_cache_size.unwrap_or(DEFAULT_METADATA_CACHE_SIZE),
            session: None,
            store_options: None,
        };

        if let Some(commit_handler) = commit_handler {
            let py_commit_lock = PyCommitLock::new(commit_handler);
            let object_store_params = ObjectStoreParams {
                commit_handler: Some(Arc::new(py_commit_lock)),
                ..Default::default()
            };
            params.store_options = Some(object_store_params);
        }
        let uri_ref = uri.clone();
        let dataset = if let Some(ver) = version {
            RT.block_on(
                async move { LanceDataset::checkout_with_params(&uri_ref, ver, &params).await },
            )
        } else {
            RT.block_on(async move { LanceDataset::open_with_params(&uri_ref, &params).await })
        };
        match dataset {
            Ok(ds) => Ok(Self {
                uri,
                ds: Arc::new(ds),
            }),
            Err(err) => Err(PyValueError::new_err(err.to_string())),
        }
    }

    #[getter(schema)]
    fn schema(self_: PyRef<'_, Self>) -> PyResult<PyObject> {
        let arrow_schema = ArrowSchema::from(self_.ds.schema());
        arrow_schema.to_pyarrow(self_.py())
    }

    /// Get index statistics
    fn index_statistics(&self, index_name: String) -> PyResult<String> {
        let index_statistics = self
            .rt
            .block_on(self.ds.index_statistics(index_name.clone()))
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        if let Some(s) = index_statistics {
            Ok(s)
        } else {
            Err(PyKeyError::new_err(format!(
                "Index \"{}\" not found",
                index_name
            )))
        }
    }

    /// Load index metadata
    fn load_indices(&self) -> PyResult<Vec<PyObject>> {
        let ds = self.ds.clone();
        let index_metadata = RT
            .block_on(async move { ds.load_indices().await })
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Python::with_gil(|py| {
            Ok(index_metadata
                .iter()
                .map(|idx| {
                    let dict = PyDict::new(py);
                    let schema = self.ds.schema();
                    let field_names = schema
                        .project_by_ids(idx.fields.as_slice())
                        .unwrap()
                        .fields
                        .iter()
                        .map(|f| f.name.clone())
                        .collect::<Vec<_>>();

                    dict.set_item("name", idx.name.clone()).unwrap();
                    // TODO: once we add more than vector indices, we need to:
                    // 1. Change protos and write path to persist index type
                    // 2. Use the new field from idx instead of hard coding it to Vector
                    dict.set_item("type", IndexType::Vector.to_string())
                        .unwrap();
                    dict.set_item("uuid", idx.uuid.to_string()).unwrap();
                    dict.set_item("fields", field_names).unwrap();
                    dict.set_item("version", idx.dataset_version).unwrap();
                    dict.to_object(py)
                })
                .collect::<Vec<_>>())
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn scanner(
        self_: PyRef<'_, Self>,
        columns: Option<Vec<String>>,
        filter: Option<String>,
        limit: Option<i64>,
        offset: Option<i64>,
        nearest: Option<&PyDict>,
        batch_size: Option<usize>,
        batch_readahead: Option<usize>,
        fragment_readahead: Option<usize>,
        scan_in_order: Option<bool>,
        fragments: Option<Vec<FileFragment>>,
    ) -> PyResult<Scanner> {
        let mut scanner: LanceScanner = self_.ds.scan();
        if let Some(c) = columns {
            scanner
                .project(&c)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        if let Some(f) = filter {
            scanner
                .filter(f.as_str())
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }

        scanner
            .limit(limit, offset)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        if let Some(batch_size) = batch_size {
            scanner.batch_size(batch_size);
        }
        if let Some(batch_readahead) = batch_readahead {
            scanner.batch_readahead(batch_readahead);
        }

        if let Some(fragment_readahead) = fragment_readahead {
            scanner.fragment_readahead(fragment_readahead);
        }

        scanner.scan_in_order(scan_in_order.unwrap_or(true));

        if let Some(fragments) = fragments {
            let fragments = fragments
                .into_iter()
                .map(|f| {
                    let file_fragment = LanceFileFragment::from(f);
                    file_fragment.into()
                })
                .collect();
            scanner.with_fragments(fragments);
        }

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

            let metric_type: Option<MetricType> = if let Some(metric) = nearest.get_item("metric") {
                if metric.is_none() {
                    None
                } else {
                    Some(
                        MetricType::try_from(metric.to_string().to_lowercase().as_str())
                            .map_err(|err| PyValueError::new_err(err.to_string()))?,
                    )
                }
            } else {
                None
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

            let use_index: bool = if let Some(idx) = nearest.get_item("use_index") {
                PyAny::downcast::<PyBool>(idx)?.extract()?
            } else {
                true
            };

            scanner
                .nearest(column.as_str(), &q, k)
                .map(|s| {
                    let mut s = s.nprobs(nprobes);
                    if let Some(factor) = refine_factor {
                        s = s.refine(factor);
                    }
                    if let Some(m) = metric_type {
                        s = s.distance_metric(m);
                    }
                    s.use_index(use_index);
                    s
                })
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }

        let scan = Arc::new(scanner);
        Ok(Scanner::new(scan))
    }

    fn count_rows(&self) -> PyResult<usize> {
        let ds = self.ds.clone();
        RT.block_on(async move {
            ds.count_rows()
                .await
                .map_err(|err| PyIOError::new_err(err.to_string()))
        })
    }

    fn take(&self, row_indices: Vec<usize>) -> PyResult<PyObject> {
        let projection = self.ds.schema().clone();
        let ds = self.ds.clone();
        let batch = RT
            .block_on(async move { ds.take(&row_indices, &projection).await })
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Python::with_gil(|py| batch.to_pyarrow(py))
    }

    fn merge(
        &mut self,
        reader: PyArrowType<ArrowArrayStreamReader>,
        left_on: String,
        right_on: String,
        py: Python<'_>,
    ) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        let new_self = Python::allow_threads(py, || {
            RT.block_on(async move {
                match new_self.merge(reader.0, &left_on, &right_on).await {
                    Ok(()) => Ok(new_self),
                    Err(err) => Err(PyIOError::new_err(err.to_string())),
                }
            })
        })?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    fn delete(&mut self, predicate: String, py: Python<'_>) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        let new_self = Python::allow_threads(py, || {
            RT.block_on(async move {
                match new_self.delete(&predicate).await {
                    Ok(()) => Ok(new_self),
                    Err(err) => Err(PyIOError::new_err(err.to_string())),
                }
            })
        })?;
        self.ds = Arc::new(new_self);
        Ok(())
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
                    dict.set_item("timestamp", v.timestamp.timestamp_nanos())
                        .unwrap();
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

    /// Fetches the currently checked out version of the dataset.
    fn version(&self) -> PyResult<u64> {
        Ok(self.ds.version().version)
    }

    /// Restore the current version
    fn restore(&mut self) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        let new_self = RT.block_on(async move {
            match new_self.restore(None).await {
                Ok(()) => Ok(new_self),
                Err(err) => Err(PyIOError::new_err(err.to_string())),
            }
        })?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    fn create_index(
        self_: PyRef<'_, Self>,
        columns: Vec<String>,
        index_type: String,
        name: Option<String>,
        metric_type: Option<&str>,
        kwargs: Option<&PyDict>,
    ) -> PyResult<()> {
        let idx_type = match index_type.to_uppercase().as_str() {
            "IVF_PQ" | "DISKANN" => IndexType::Vector,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Index type '{index_type}' is not supported."
                )))
            }
        };

        let m_type = match metric_type {
            Some(mt) => MetricType::try_from(mt.to_string().to_lowercase().as_str())
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
            None => MetricType::L2,
        };

        let replace = if let Some(replace) = kwargs.and_then(|k| k.get_item("replace")) {
            PyAny::downcast::<PyBool>(replace)?.extract()?
        } else {
            false
        };

        // Only VectorParams are supported.
        let params = match index_type.to_uppercase().as_str() {
            "IVF_PQ" => {
                let mut ivf_params = IvfBuildParams::default();
                let mut pq_params = PQBuildParams::default();
                if let Some(kwargs) = kwargs {
                    if let Some(n) = kwargs.get_item("num_partitions") {
                        ivf_params.num_partitions = PyAny::downcast::<PyInt>(n)?.extract()?
                    };

                    if let Some(n) = kwargs.get_item("num_bits") {
                        pq_params.num_bits = PyAny::downcast::<PyInt>(n)?.extract()?
                    };

                    if let Some(n) = kwargs.get_item("num_sub_vectors") {
                        pq_params.num_sub_vectors = PyAny::downcast::<PyInt>(n)?.extract()?
                    };

                    if let Some(o) = kwargs.get_item("use_opq") {
                        #[cfg(not(feature = "opq"))]
                        if PyAny::downcast::<PyBool>(o)?.extract()? {
                            return Err(PyValueError::new_err(
                                "Feature 'opq' is not installed.".to_string(),
                            ));
                        }
                        pq_params.use_opq = PyAny::downcast::<PyBool>(o)?.extract()?
                    };

                    if let Some(o) = kwargs.get_item("max_opq_iterations") {
                        pq_params.max_opq_iters = PyAny::downcast::<PyInt>(o)?.extract()?
                    };

                    if let Some(c) = kwargs.get_item("ivf_centroids") {
                        let batch = RecordBatch::from_pyarrow(c)?;
                        if "_ivf_centroids" != batch.schema().field(0).name() {
                            return Err(PyValueError::new_err(
                                "Expected '_ivf_centroids' as the first column name.",
                            ));
                        }
                        let centroids = as_fixed_size_list_array(batch.column(0));
                        ivf_params.centroids = Some(Arc::new(centroids.clone()))
                    };
                }
                VectorIndexParams::with_ivf_pq_params(m_type, ivf_params, pq_params)
            }
            "DISKANN" => {
                let mut params = DiskANNParams::default();
                if let Some(kwargs) = kwargs {
                    if let Some(n) = kwargs.get_item("r") {
                        params.r = PyAny::downcast::<PyInt>(n)?.extract()?
                    };

                    if let Some(n) = kwargs.get_item("alpha") {
                        params.alpha = PyAny::downcast::<PyFloat>(n)?.extract()?
                    };

                    if let Some(n) = kwargs.get_item("l") {
                        params.l = PyAny::downcast::<PyInt>(n)?.extract()?
                    };
                }
                VectorIndexParams::with_diskann_params(m_type, params)
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Index type '{index_type}' is not supported."
                )))
            }
        };
        let ds = self_.ds.clone();
        RT.block_on(async move {
            let cols_slice: Vec<&str> = columns.iter().map(|col| col.as_str()).collect();
            ds.create_index(cols_slice.as_slice(), idx_type, name, &params, replace)
                .await
        })
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    fn get_fragments(self_: PyRef<'_, Self>) -> PyResult<Vec<FileFragment>> {
        let core_fragments = self_.ds.get_fragments();

        Python::with_gil(|_| {
            let fragments: Vec<FileFragment> = core_fragments
                .iter()
                .map(|f| FileFragment::new(f.clone()))
                .collect::<Vec<_>>();
            Ok(fragments)
        })
    }

    fn get_fragment(self_: PyRef<'_, Self>, fragment_id: usize) -> PyResult<Option<FileFragment>> {
        if let Some(fragment) = self_.ds.get_fragment(fragment_id) {
            Ok(Some(FileFragment::new(fragment)))
        } else {
            Ok(None)
        }
    }

    #[staticmethod]
    fn commit(
        dataset_uri: String,
        schema: &PyAny,
        fragments: Vec<&PyAny>,
        mode: Option<String>,
    ) -> PyResult<Self> {
        let arrow_schema = ArrowSchema::from_pyarrow(schema)?;
        let schema = Schema::try_from(&arrow_schema).map_err(|e| {
            PyValueError::new_err(format!(
                "Failed to convert Arrow schema to Lance schema: {}",
                e
            ))
        })?;
        let fragment_metadata = fragments
            .iter()
            .map(|f| f.extract::<FragmentMetadata>().map(|fm| fm.inner))
            .collect::<PyResult<Vec<Fragment>>>()?;
        let mode = parse_write_mode(mode.as_deref().unwrap_or("create"))?;
        let uri = dataset_uri.clone();
        let ds = RT
            .block_on(async move {
                LanceDataset::commit(&dataset_uri, &schema, &fragment_metadata, mode).await
            })
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self {
            ds: Arc::new(ds),
            uri,
        })
    }
}

impl Dataset {
    fn list_versions(&self) -> ::lance::error::Result<Vec<Version>> {
        let ds = self.ds.clone();
        RT.block_on(async move { ds.versions().await })
    }
}

#[pyfunction(name = "_write_dataset")]
pub fn write_dataset(reader: &PyAny, uri: String, options: &PyDict) -> PyResult<bool> {
    let params = get_write_params(options)?;
    if reader.is_instance_of::<Scanner>() {
        let scanner: Scanner = reader.extract()?;
        let batches = RT
            .block_on(async move { scanner.to_reader().await })
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Python::allow_threads(reader.py(), || {
            RT.block_on(async move { LanceDataset::write(batches, &uri, params).await })
                .map_err(|err| PyIOError::new_err(err.to_string()))
        })?;
        Ok(true)
    } else {
        let batches = ArrowArrayStreamReader::from_pyarrow(reader)?;
        // The reader itself might try to acquire the GIL, so it's imperative we
        // aren't holding it when we call `LanceDataset::write`.
        Python::allow_threads(reader.py(), || {
            RT.block_on(async move { LanceDataset::write(batches, &uri, params).await })
                .map_err(|err| PyIOError::new_err(err.to_string()))
        })?;
        Ok(true)
    }
}

fn parse_write_mode(mode: &str) -> PyResult<WriteMode> {
    match mode.to_string().to_lowercase().as_str() {
        "create" => Ok(WriteMode::Create),
        "append" => Ok(WriteMode::Append),
        "overwrite" => Ok(WriteMode::Overwrite),
        _ => Err(PyValueError::new_err(format!("Invalid mode {mode}"))),
    }
}

pub(crate) fn get_write_params(options: &PyDict) -> PyResult<Option<WriteParams>> {
    let params = if options.is_none() {
        None
    } else {
        let mut p = WriteParams::default();
        if let Some(mode) = options.get_item("mode") {
            p.mode = parse_write_mode(mode.extract::<String>()?.as_str())?;
        };
        if let Some(maybe_nrows) = options.get_item("max_rows_per_file") {
            p.max_rows_per_file = usize::extract(maybe_nrows)?;
        }
        if let Some(maybe_nrows) = options.get_item("max_rows_per_group") {
            p.max_rows_per_group = usize::extract(maybe_nrows)?;
        }

        if let Some(commit_handler) = options.get_item("commit_handler") {
            let py_commit_lock = PyCommitLock::new(commit_handler.to_object(options.py()));
            let object_store_params = ObjectStoreParams {
                commit_handler: Some(Arc::new(py_commit_lock)),
                ..Default::default()
            };
            p.store_params = Some(object_store_params);
        }

        Some(p)
    };
    Ok(params)
}
