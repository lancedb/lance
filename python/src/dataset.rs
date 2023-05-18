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
use arrow_array::{Float32Array, RecordBatchReader};
use arrow_data::ArrayData;
use arrow_schema::Schema as ArrowSchema;
use lance::dataset::ReadParams;
use lance::datatypes::Schema;
use lance::format::Fragment;
use lance::index::vector::ivf::IvfBuildParams;
use lance::index::vector::pq::PQBuildParams;
use pyo3::exceptions::{PyIOError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBool, PyDict, PyFloat, PyInt, PyLong};
use pyo3::{pyclass, PyObject, PyResult};
use tokio::runtime::Runtime;

use crate::fragment::{FileFragment, FragmentMetadata};
use crate::Scanner;
use lance::dataset::{
    scanner::Scanner as LanceScanner, Dataset as LanceDataset, Version, WriteMode, WriteParams,
};
use lance::index::{
    vector::diskann::DiskANNParams,
    vector::{MetricType, VectorIndexParams},
    DatasetIndexExt, IndexType,
};

const DEFAULT_NPROBS: usize = 1;
const DEFAULT_INDEX_CACHE_SIZE: usize = 256;

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
    fn new(
        uri: String,
        version: Option<u64>,
        block_size: Option<usize>,
        index_cache_size: Option<usize>,
    ) -> PyResult<Self> {
        let rt = Runtime::new()?;
        let params = ReadParams {
            block_size,
            index_cache_size: index_cache_size.unwrap_or(DEFAULT_INDEX_CACHE_SIZE),
            session: None,
        };
        let dataset = rt.block_on(async {
            if let Some(ver) = version {
                LanceDataset::checkout_with_params(uri.as_str(), ver, &params).await
            } else {
                LanceDataset::open_with_params(uri.as_str(), &params).await
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

    /// Load index metadata
    fn load_indices(self_: PyRef<'_, Self>) -> PyResult<Vec<PyObject>> {
        let index_metadata = self_
            .rt
            .block_on(async { self_.ds.load_indices().await })
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let py = self_.py();
        Ok(index_metadata
            .iter()
            .map(|idx| {
                let dict = PyDict::new(py);
                let schema = self_.ds.schema();
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
                dict.set_item("version", idx.dataset_version.clone())
                    .unwrap();
                dict.to_object(py)
            })
            .collect::<Vec<_>>())
    }

    fn scanner(
        self_: PyRef<'_, Self>,
        columns: Option<Vec<String>>,
        filter: Option<String>,
        limit: Option<i64>,
        offset: Option<i64>,
        nearest: Option<&PyDict>,
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
        if let Some(limit) = limit {
            scanner
                .limit(limit, offset)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
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
                .iter()
                .map(|f| f.fragment().metadata().clone())
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

    fn take(self_: PyRef<'_, Self>, row_indices: Vec<usize>) -> PyResult<PyObject> {
        let projection = self_.ds.schema();
        let batch = self_
            .rt
            .block_on(async { self_.ds.take(&row_indices, &projection).await })
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        batch.to_pyarrow(self_.py())
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

    fn create_index(
        self_: PyRef<'_, Self>,
        columns: Vec<&str>,
        index_type: &str,
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
                        pq_params.use_opq = PyAny::downcast::<PyBool>(o)?.extract()?
                    };

                    if let Some(o) = kwargs.get_item("max_opq_iterations") {
                        pq_params.max_opq_iters = PyAny::downcast::<PyInt>(o)?.extract()?
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

        self_
            .rt
            .block_on(async {
                self_
                    .ds
                    .create_index(columns.as_slice(), idx_type, name, &params)
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
            Ok(Some(FileFragment::new(fragment.clone())))
        } else {
            Ok(None)
        }
    }

    #[staticmethod]
    fn commit(
        dataset_uri: &str,
        schema: &PyAny,
        fragments: Vec<&PyAny>,
        mode: Option<&str>,
    ) -> PyResult<Self> {
        let rt = Arc::new(Runtime::new()?);
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
        let mode = parse_write_mode(mode.unwrap_or("create"))?;
        let ds = rt
            .block_on(async {
                LanceDataset::commit(dataset_uri, &schema, &fragment_metadata, mode).await
            })
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self {
            ds: Arc::new(ds),
            rt,
            uri: dataset_uri.to_string(),
        })
    }
}

impl Dataset {
    fn list_versions(&self) -> ::lance::error::Result<Vec<Version>> {
        self.rt.block_on(async { self.ds.versions().await })
    }
}

#[pyfunction(name = "_write_dataset")]
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
        Some(p)
    };
    Ok(params)
}
