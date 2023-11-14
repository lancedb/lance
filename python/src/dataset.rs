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
use arrow::pyarrow::{ToPyArrow, *};
use arrow_array::{Float32Array, RecordBatch, RecordBatchReader};
use arrow_data::ArrayData;
use arrow_schema::Schema as ArrowSchema;
use async_trait::async_trait;
use chrono::Duration;

use futures::StreamExt;
use lance::dataset::{
    fragment::FileFragment as LanceFileFragment, progress::WriteFragmentProgress,
    scanner::Scanner as LanceScanner, transaction::Operation as LanceOperation,
    Dataset as LanceDataset, ReadParams, Version, WriteMode, WriteParams,
};
use lance::index::IndexParams;
use lance::index::{
    scalar::ScalarIndexParams,
    vector::{diskann::DiskANNParams, VectorIndexParams},
    DatasetIndexExt,
};
use lance_arrow::as_fixed_size_list_array;
use lance_core::{datatypes::Schema, format::Fragment, io::object_store::ObjectStoreParams};
use lance_index::{
    vector::{ivf::IvfBuildParams, pq::PQBuildParams},
    IndexType,
};
use lance_linalg::distance::MetricType;
use pyo3::exceptions::PyStopIteration;
use pyo3::prelude::*;
use pyo3::types::PySet;
use pyo3::{
    exceptions::{PyIOError, PyKeyError, PyValueError},
    pyclass,
    types::{IntoPyDict, PyBool, PyDict, PyFloat, PyInt, PyLong},
    PyObject, PyResult,
};
use snafu::{location, Location};

use crate::fragment::{FileFragment, FragmentMetadata};
use crate::RT;
use crate::{LanceReader, Scanner};

use self::cleanup::CleanupStats;
use self::commit::PyCommitLock;

pub mod cleanup;
pub mod commit;
pub mod optimize;

const DEFAULT_NPROBS: usize = 1;
const DEFAULT_INDEX_CACHE_SIZE: usize = 256;
const DEFAULT_METADATA_CACHE_SIZE: usize = 256;

#[pyclass(name = "_Operation", module = "_lib")]
#[derive(Clone)]
pub struct Operation(LanceOperation);

fn into_fragments(fragments: Vec<FragmentMetadata>) -> Vec<Fragment> {
    fragments
        .into_iter()
        .map(|f| f.inner)
        .collect::<Vec<Fragment>>()
}

fn convert_schema(arrow_schema: &ArrowSchema) -> PyResult<Schema> {
    Schema::try_from(arrow_schema).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to convert Arrow schema to Lance schema: {}",
            e
        ))
    })
}

#[pymethods]
impl Operation {
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    #[staticmethod]
    fn overwrite(
        schema: PyArrowType<ArrowSchema>,
        fragments: Vec<FragmentMetadata>,
    ) -> PyResult<Self> {
        let schema = convert_schema(&schema.0)?;
        let fragments = into_fragments(fragments);
        let op = LanceOperation::Overwrite { fragments, schema };
        Ok(Self(op))
    }

    #[staticmethod]
    fn append(fragments: Vec<FragmentMetadata>) -> PyResult<Self> {
        let fragments = into_fragments(fragments);
        let op = LanceOperation::Append { fragments };
        Ok(Self(op))
    }

    #[staticmethod]
    fn delete(
        updated_fragments: Vec<FragmentMetadata>,
        deleted_fragment_ids: Vec<u64>,
        predicate: String,
    ) -> PyResult<Self> {
        let updated_fragments = into_fragments(updated_fragments);
        let op = LanceOperation::Delete {
            updated_fragments,
            deleted_fragment_ids,
            predicate,
        };
        Ok(Self(op))
    }

    #[staticmethod]
    fn merge(fragments: Vec<FragmentMetadata>, schema: PyArrowType<ArrowSchema>) -> PyResult<Self> {
        let schema = convert_schema(&schema.0)?;
        let fragments = into_fragments(fragments);
        let op = LanceOperation::Merge { fragments, schema };
        Ok(Self(op))
    }

    #[staticmethod]
    fn restore(version: u64) -> PyResult<Self> {
        let op = LanceOperation::Restore { version };
        Ok(Self(op))
    }
}

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
            let mut object_store_params = ObjectStoreParams::default();
            object_store_params.set_commit_lock(Arc::new(py_commit_lock));
            params.store_options = Some(object_store_params);
        }
        let dataset = if let Some(ver) = version {
            RT.runtime
                .block_on(LanceDataset::checkout_with_params(&uri, ver, &params))
        } else {
            RT.runtime
                .block_on(LanceDataset::open_with_params(&uri, &params))
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
        let index_statistics = RT
            .runtime
            .block_on(self.ds.index_statistics(&index_name))
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
    fn load_indices(self_: PyRef<'_, Self>) -> PyResult<Vec<PyObject>> {
        let index_metadata = RT
            .block_on(Some(self_.py()), self_.ds.load_indices())?
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let py = self_.py();
        Ok(index_metadata
            .iter()
            .map(|idx| {
                let dict = PyDict::new(py);
                let schema = self_.ds.schema();
                let field_names = schema
                    .project_by_ids(idx.fields.as_slice())
                    .fields
                    .iter()
                    .map(|f| f.name.clone())
                    .collect::<Vec<_>>();

                let fragment_set = PySet::empty(py).unwrap();
                if let Some(bitmap) = &idx.fragment_bitmap {
                    for fragment_id in bitmap.iter() {
                        fragment_set.add(fragment_id).unwrap();
                    }
                }

                dict.set_item("name", idx.name.clone()).unwrap();
                // TODO: once we add more than vector indices, we need to:
                // 1. Change protos and write path to persist index type
                // 2. Use the new field from idx instead of hard coding it to Vector
                dict.set_item("type", IndexType::Vector.to_string())
                    .unwrap();
                dict.set_item("uuid", idx.uuid.to_string()).unwrap();
                dict.set_item("fields", field_names).unwrap();
                dict.set_item("version", idx.dataset_version).unwrap();
                dict.set_item("fragment_ids", fragment_set).unwrap();
                dict.to_object(py)
            })
            .collect::<Vec<_>>())
    }

    #[allow(clippy::too_many_arguments)]
    fn scanner(
        self_: PyRef<'_, Self>,
        columns: Option<Vec<String>>,
        filter: Option<String>,
        prefilter: Option<bool>,
        limit: Option<i64>,
        offset: Option<i64>,
        nearest: Option<&PyDict>,
        batch_size: Option<usize>,
        batch_readahead: Option<usize>,
        fragment_readahead: Option<usize>,
        scan_in_order: Option<bool>,
        fragments: Option<Vec<FileFragment>>,
        with_row_id: Option<bool>,
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
        if let Some(prefilter) = prefilter {
            scanner.prefilter(prefilter);
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

        if with_row_id.unwrap_or(false) {
            scanner.with_row_id();
        }

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
                    // Use limit if k is not specified, default to 10.
                    limit.unwrap_or(10) as usize
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
        RT.runtime
            .block_on(self.ds.count_rows())
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    fn take(
        self_: PyRef<'_, Self>,
        row_indices: Vec<u64>,
        columns: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let projection = if let Some(columns) = columns {
            self_.ds.schema().project(&columns)
        } else {
            Ok(self_.ds.schema().clone())
        }
        .map_err(|err| PyIOError::new_err(err.to_string()))?;
        let batch = RT
            .block_on(Some(self_.py()), self_.ds.take(&row_indices, &projection))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;

        crate::arrow::record_batch_to_pyarrow(self_.py(), &batch)
    }

    fn take_rows(
        self_: PyRef<'_, Self>,
        row_indices: Vec<u64>,
        columns: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let projection = if let Some(columns) = columns {
            self_.ds.schema().project(&columns)
        } else {
            Ok(self_.ds.schema().clone())
        }
        .map_err(|err| {
            PyIOError::new_err(format!(
                "TakeRows: failed to run projection over schema: {}",
                err
            ))
        })?;

        let batch = RT
            .block_on(
                Some(self_.py()),
                self_.ds.take_rows(&row_indices, &projection),
            )?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;

        crate::arrow::record_batch_to_pyarrow(self_.py(), &batch)
    }

    #[pyo3(signature = (row_slices, columns = None, batch_readahead = 10))]
    fn take_scan(
        &self,
        row_slices: PyObject,
        columns: Option<Vec<String>>,
        batch_readahead: usize,
    ) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        let projection = if let Some(columns) = columns {
            Arc::new(
                self.ds
                    .schema()
                    .project(&columns)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?,
            )
        } else {
            Arc::new(self.ds.schema().clone())
        };

        // Call into the Python iterable, only holding the GIL as necessary.
        let py_iter = Python::with_gil(|py| row_slices.call_method0(py, "__iter__"))?;
        let slice_iter = std::iter::from_fn(move || {
            Python::with_gil(|py| {
                match py_iter
                    .call_method0(py, "__next__")
                    .and_then(|range| range.extract::<(u64, u64)>(py))
                {
                    Ok((start, end)) => Some(Ok(start..end)),
                    Err(err) if err.is_instance_of::<PyStopIteration>(py) => None,
                    Err(err) => Some(Err(lance::Error::InvalidInput {
                        source: Box::new(err),
                        location: location!(),
                    })),
                }
            })
        });

        let slice_stream = futures::stream::iter(slice_iter).boxed();

        let stream = self.ds.take_scan(slice_stream, projection, batch_readahead);

        Ok(PyArrowType(Box::new(LanceReader::from_stream(stream))))
    }

    fn merge(
        &mut self,
        reader: PyArrowType<ArrowArrayStreamReader>,
        left_on: &str,
        right_on: &str,
    ) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        RT.block_on(None, new_self.merge(reader.0, left_on, right_on))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    fn delete(&mut self, predicate: String) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        RT.block_on(None, new_self.delete(&predicate))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    fn count_deleted_rows(&self) -> PyResult<usize> {
        RT.block_on(None, self.ds.count_deleted_rows())?
            .map_err(|err| PyIOError::new_err(err.to_string()))
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
                    dict.set_item(
                        "timestamp",
                        v.timestamp.timestamp_nanos_opt().unwrap_or_default(),
                    )
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

    fn latest_version(self_: PyRef<'_, Self>) -> PyResult<u64> {
        RT.block_on(Some(self_.py()), self_.ds.latest_version_id())?
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    /// Restore the current version
    fn restore(&mut self) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        RT.block_on(None, new_self.restore(None))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    /// Cleanup old versions from the dataset
    fn cleanup_old_versions(
        &self,
        older_than_micros: i64,
        delete_unverified: Option<bool>,
    ) -> PyResult<CleanupStats> {
        let older_than = Duration::microseconds(older_than_micros);
        let cleanup_stats = RT
            .block_on(
                None,
                self.ds.cleanup_old_versions(older_than, delete_unverified),
            )?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Ok(CleanupStats {
            bytes_removed: cleanup_stats.bytes_removed,
            old_versions: cleanup_stats.old_versions,
        })
    }

    fn optimize_indices(&mut self, _kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        RT.block_on(None, new_self.optimize_indices())?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    fn create_index(
        &mut self,
        columns: Vec<&str>,
        index_type: &str,
        name: Option<String>,
        replace: Option<bool>,
        metric_type: Option<&str>,
        kwargs: Option<&PyDict>,
    ) -> PyResult<()> {
        let idx_type = match index_type.to_uppercase().as_str() {
            "BTREE" => IndexType::Scalar,
            "IVF_PQ" | "DISKANN" => IndexType::Vector,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Index type '{index_type}' is not supported."
                )))
            }
        };

        // Only VectorParams are supported.
        let params: Box<dyn IndexParams> = match index_type.to_uppercase().as_str() {
            "BTREE" => Box::<ScalarIndexParams>::default(),
            "IVF_PQ" => {
                let mut ivf_params = IvfBuildParams::default();
                let mut pq_params = PQBuildParams::default();
                let mut m_type = MetricType::L2;
                if let Some(kwargs) = kwargs {
                    if let Some(mt) = kwargs.get_item("metric_type") {
                        m_type = MetricType::try_from(mt.to_string().to_lowercase().as_str())
                            .map_err(|err| PyValueError::new_err(err.to_string()))?;
                    }

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
                Box::new(VectorIndexParams::with_ivf_pq_params(
                    m_type, ivf_params, pq_params,
                ))
            }
            "DISKANN" => {
                let mut params = DiskANNParams::default();
                let mut m_type = MetricType::L2;
                if let Some(kwargs) = kwargs {
                    if let Some(mt) = kwargs.get_item("metric_type") {
                        m_type = MetricType::try_from(mt.to_string().to_lowercase().as_str())
                            .map_err(|err| PyValueError::new_err(err.to_string()))?;
                    }

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
                Box::new(VectorIndexParams::with_diskann_params(m_type, params))
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Index type '{index_type}' is not supported."
                )))
            }
        };

        let replace = replace.unwrap_or(true);

        let mut new_self = self.ds.as_ref().clone();
        RT.block_on(
            None,
            new_self.create_index(&columns, idx_type, name, params.as_ref(), replace),
        )?
        .map_err(|err| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);

        Ok(())
    }

    fn count_fragments(&self) -> usize {
        self.ds.count_fragments()
    }

    fn num_small_files(&self, max_rows_per_group: usize) -> PyResult<usize> {
        RT.block_on(None, self.ds.num_small_files(max_rows_per_group))
            .map_err(|err| PyIOError::new_err(err.to_string()))
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

    fn count_unindexed_rows(&self, index_name: String) -> PyResult<Option<usize>> {
        let idx = RT.block_on(None, self.ds.load_index_by_name(index_name.as_str()))?;
        if let Some(index) = idx {
            RT.block_on(
                None,
                self.ds
                    .count_unindexed_rows(index.uuid.to_string().as_str()),
            )?
            .map_err(|err| PyIOError::new_err(err.to_string()))
        } else {
            Err(PyIOError::new_err(format!(
                "Index {} not found",
                index_name
            )))
        }
    }

    fn count_indexed_rows(&self, index_name: String) -> PyResult<Option<usize>> {
        let idx = RT.block_on(None, self.ds.load_index_by_name(index_name.as_str()))?;
        if let Some(index) = idx {
            RT.block_on(
                None,
                self.ds.count_indexed_rows(index.uuid.to_string().as_str()),
            )?
            .map_err(|err| PyIOError::new_err(err.to_string()))
        } else {
            Err(PyIOError::new_err(format!(
                "Index {} not found",
                index_name
            )))
        }
    }

    fn index_cache_size(&self) -> PyResult<usize> {
        Ok(self.ds.index_cache_size())
    }

    #[staticmethod]
    fn commit(
        dataset_uri: &str,
        operation: Operation,
        read_version: Option<u64>,
        commit_lock: Option<&PyAny>,
    ) -> PyResult<Self> {
        let store_params = if let Some(commit_handler) = commit_lock {
            let py_commit_lock = PyCommitLock::new(commit_handler.to_object(commit_handler.py()));
            let mut object_store_params = ObjectStoreParams::default();
            object_store_params.set_commit_lock(Arc::new(py_commit_lock));
            Some(object_store_params)
        } else {
            None
        };
        let ds = RT
            .block_on(
                commit_lock.map(|cl| cl.py()),
                LanceDataset::commit(dataset_uri, operation.0, read_version, store_params),
            )?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self {
            ds: Arc::new(ds),
            uri: dataset_uri.to_string(),
        })
    }

    fn validate(&self) -> PyResult<()> {
        RT.block_on(None, self.ds.validate())?
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }
}

impl Dataset {
    fn list_versions(&self) -> ::lance::error::Result<Vec<Version>> {
        RT.runtime.block_on(self.ds.versions())
    }
}

#[pyfunction(name = "_write_dataset")]
pub fn write_dataset(reader: &PyAny, uri: String, options: &PyDict) -> PyResult<bool> {
    let params = get_write_params(options)?;
    let py = options.py();
    if reader.is_instance_of::<Scanner>() {
        let scanner: Scanner = reader.extract()?;
        let batches = RT
            .block_on(Some(py), scanner.to_reader())?
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        RT.block_on(Some(py), LanceDataset::write(batches, &uri, params))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Ok(true)
    } else {
        let batches = ArrowArrayStreamReader::from_pyarrow(reader)?;
        RT.block_on(Some(py), LanceDataset::write(batches, &uri, params))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
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

pub fn get_object_store_params(options: &PyDict) -> Option<ObjectStoreParams> {
    if options.is_none() {
        None
    } else if let Some(commit_handler) = options.get_item("commit_handler") {
        let py_commit_lock = PyCommitLock::new(commit_handler.to_object(options.py()));
        let mut object_store_params = ObjectStoreParams::default();
        object_store_params.set_commit_lock(Arc::new(py_commit_lock));
        Some(object_store_params)
    } else {
        None
    }
}

pub fn get_write_params(options: &PyDict) -> PyResult<Option<WriteParams>> {
    let params = if options.is_none() {
        None
    } else {
        let mut p = WriteParams::default();
        if let Some(mode) = options.get_item("mode") {
            p.mode = parse_write_mode(mode.extract::<String>()?.as_str())?;
        };
        if let Some(index_cache_size) = options.get_item("index_cache_size") {
            p.index_cache_size = usize::extract(index_cache_size)?;
        }
        if let Some(maybe_nrows) = options.get_item("max_rows_per_file") {
            p.max_rows_per_file = usize::extract(maybe_nrows)?;
        }
        if let Some(maybe_nrows) = options.get_item("max_rows_per_group") {
            p.max_rows_per_group = usize::extract(maybe_nrows)?;
        }
        if let Some(maybe_nbytes) = options.get_item("max_bytes_per_file") {
            p.max_bytes_per_file = usize::extract(maybe_nbytes)?;
        }
        if let Some(progress) = options.get_item("progress") {
            if !progress.is_none() {
                p.progress = Arc::new(PyWriteProgress::new(progress.to_object(options.py())));
            }
        }

        p.store_params = get_object_store_params(options);

        Some(p)
    };
    Ok(params)
}

#[pyclass(name = "_FragmentWriteProgress", module = "_lib")]
#[derive(Debug)]
pub struct PyWriteProgress {
    /// A Python object that implements the `WriteFragmentProgress` trait.
    py_obj: PyObject,
}

impl PyWriteProgress {
    fn new(obj: PyObject) -> Self {
        Self { py_obj: obj }
    }
}

#[async_trait]
impl WriteFragmentProgress for PyWriteProgress {
    async fn begin(&self, fragment: &Fragment, multipart_id: &str) -> lance::Result<()> {
        let json_str = serde_json::to_string(fragment)?;

        Python::with_gil(|py| -> PyResult<()> {
            let kwargs = PyDict::new(py);
            kwargs.set_item("multipart_id", multipart_id)?;
            self.py_obj
                .call_method(py, "_do_begin", (json_str,), Some(kwargs))?;
            Ok(())
        })
        .map_err(|e| lance::Error::IO {
            message: format!("Failed to call begin() on WriteFragmentProgress: {}", e),
            location: location!(),
        })?;
        Ok(())
    }

    async fn complete(&self, fragment: &Fragment) -> lance::Result<()> {
        let json_str = serde_json::to_string(fragment)?;

        Python::with_gil(|py| -> PyResult<()> {
            self.py_obj
                .call_method(py, "_do_complete", (json_str,), None)?;
            Ok(())
        })
        .map_err(|e| lance::Error::IO {
            message: format!("Failed to call complete() on WriteFragmentProgress: {}", e),
            location: location!(),
        })?;
        Ok(())
    }
}
