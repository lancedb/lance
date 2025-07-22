// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::str;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::UInt8Type;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::*;
use arrow_array::Array;
use arrow_array::{make_array, RecordBatch, RecordBatchReader};
use arrow_data::ArrayData;
use arrow_schema::{DataType, Schema as ArrowSchema};
use async_trait::async_trait;
use blob::LanceBlobFile;
use chrono::{Duration, TimeDelta};
use futures::{StreamExt, TryFutureExt};
use log::error;
use object_store::path::Path;
use pyo3::exceptions::{PyStopIteration, PyTypeError};
use pyo3::types::{PyBytes, PyInt, PyList, PySet, PyString};
use pyo3::{
    exceptions::{PyIOError, PyKeyError, PyValueError},
    pybacked::PyBackedStr,
    pyclass,
    types::{IntoPyDict, PyDict},
    PyObject, PyResult,
};
use pyo3::{prelude::*, IntoPyObjectExt};
use snafu::location;

use lance::dataset::refs::{Ref, TagContents};
use lance::dataset::scanner::{
    DatasetRecordBatchStream, ExecutionStatsCallback, MaterializationStyle,
};
use lance::dataset::statistics::{DataStatistics, DatasetStatisticsExt};
use lance::dataset::AutoCleanupParams;
use lance::dataset::{
    fragment::FileFragment as LanceFileFragment,
    progress::WriteFragmentProgress,
    scanner::Scanner as LanceScanner,
    transaction::{Operation, Transaction},
    Dataset as LanceDataset, MergeInsertBuilder as LanceMergeInsertBuilder, ReadParams,
    UncommittedMergeInsert, UpdateBuilder, Version, WhenMatched, WhenNotMatched,
    WhenNotMatchedBySource, WriteMode, WriteParams,
};
use lance::dataset::{
    BatchInfo, BatchUDF, CommitBuilder, MergeStats, NewColumnTransform, UDFCheckpointStore,
    WriteDestination,
};
use lance::dataset::{ColumnAlteration, ProjectionRequest};
use lance::index::vector::utils::get_vector_type;
use lance::index::{vector::VectorIndexParams, DatasetIndexInternalExt};
use lance::{dataset::builder::DatasetBuilder, index::vector::IndexFileVersion};
use lance_arrow::as_fixed_size_list_array;
use lance_index::scalar::inverted::query::{
    BooleanQuery, BoostQuery, FtsQuery, MatchQuery, MultiMatchQuery, Operator, PhraseQuery,
};
use lance_index::{
    infer_system_index_type, metrics::NoOpMetricsCollector, scalar::inverted::query::Occur,
};
use lance_index::{
    optimize::OptimizeOptions,
    scalar::{FullTextSearchQuery, InvertedIndexParams, ScalarIndexParams, ScalarIndexType},
    vector::{
        hnsw::builder::HnswBuildParams, ivf::IvfBuildParams, pq::PQBuildParams,
        sq::builder::SQBuildParams,
    },
    DatasetIndexExt, IndexParams, IndexType,
};
use lance_io::object_store::ObjectStoreParams;
use lance_linalg::distance::MetricType;
use lance_table::format::Fragment;
use lance_table::io::commit::CommitHandler;

use crate::error::PythonErrorExt;
use crate::file::object_store_from_uri_or_path;
use crate::fragment::FileFragment;
use crate::scanner::ScanStatistics;
use crate::schema::LanceSchema;
use crate::session::Session;
use crate::utils::PyLance;
use crate::RT;
use crate::{LanceReader, Scanner};

use self::cleanup::CleanupStats;
use self::commit::PyCommitLock;

pub mod blob;
pub mod cleanup;
pub mod commit;
pub mod optimize;
pub mod stats;

const DEFAULT_NPROBS: usize = 20;

fn convert_reader(reader: &Bound<PyAny>) -> PyResult<Box<dyn RecordBatchReader + Send>> {
    let py = reader.py();
    if reader.is_instance_of::<Scanner>() {
        let scanner: Scanner = reader.extract()?;
        Ok(Box::new(
            RT.spawn(Some(py), async move { scanner.to_reader().await })?
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
        ))
    } else {
        Ok(Box::new(ArrowArrayStreamReader::from_pyarrow_bound(
            reader,
        )?))
    }
}

#[pyclass(name = "_MergeInsertBuilder", module = "_lib", subclass)]
pub struct MergeInsertBuilder {
    builder: LanceMergeInsertBuilder,
    dataset: Py<Dataset>,
}

#[pymethods]
impl MergeInsertBuilder {
    #[new]
    pub fn new(dataset: &Bound<'_, PyAny>, on: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dataset: Py<Dataset> = dataset.extract()?;
        let ds = dataset.borrow(on.py()).ds.clone();
        // Either a single string, which we put in a vector or an iterator
        // of strings, which we collect into a vector
        let on = on
            .downcast::<PyString>()
            .map(|val| vec![val.to_string()])
            .or_else(|_| {
                let iterator = on.try_iter().map_err(|_| {
                    PyTypeError::new_err(
                        "The `on` argument to merge_insert must be a str or iterable of str",
                    )
                })?;
                let mut keys = Vec::new();
                for key in iterator {
                    keys.push(key?.downcast::<PyString>()?.to_string());
                }
                PyResult::Ok(keys)
            })?;

        let mut builder = LanceMergeInsertBuilder::try_new(ds, on)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        // We don't have do_nothing methods in python so we start with a blank slate
        builder
            .when_matched(WhenMatched::DoNothing)
            .when_not_matched(WhenNotMatched::DoNothing);

        Ok(Self { builder, dataset })
    }

    #[pyo3(signature=(condition=None))]
    pub fn when_matched_update_all<'a>(
        mut slf: PyRefMut<'a, Self>,
        condition: Option<&str>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let new_val = if let Some(expr) = condition {
            let dataset = slf.dataset.borrow(slf.py());
            WhenMatched::update_if(&dataset.ds, expr)
                .map_err(|err| PyValueError::new_err(err.to_string()))?
        } else {
            WhenMatched::UpdateAll
        };
        slf.builder.when_matched(new_val);
        Ok(slf)
    }

    pub fn when_not_matched_insert_all(mut slf: PyRefMut<Self>) -> PyResult<PyRefMut<Self>> {
        slf.builder.when_not_matched(WhenNotMatched::InsertAll);
        Ok(slf)
    }

    #[pyo3(signature=(expr=None))]
    pub fn when_not_matched_by_source_delete<'a>(
        mut slf: PyRefMut<'a, Self>,
        expr: Option<&str>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let new_val = if let Some(expr) = expr {
            let dataset = slf.dataset.borrow(slf.py());
            WhenNotMatchedBySource::delete_if(&dataset.ds, expr)
                .map_err(|err| PyValueError::new_err(err.to_string()))?
        } else {
            WhenNotMatchedBySource::Delete
        };
        slf.builder.when_not_matched_by_source(new_val);
        Ok(slf)
    }

    pub fn conflict_retries(
        mut slf: PyRefMut<'_, Self>,
        max_retries: u32,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.builder.conflict_retries(max_retries);
        Ok(slf)
    }

    pub fn retry_timeout(
        mut slf: PyRefMut<'_, Self>,
        timeout: std::time::Duration,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.builder.retry_timeout(timeout);
        Ok(slf)
    }

    pub fn execute(&mut self, new_data: &Bound<PyAny>) -> PyResult<PyObject> {
        let py = new_data.py();
        let new_data = convert_reader(new_data)?;

        let job = self
            .builder
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let (new_dataset, stats) = RT
            .spawn(Some(py), job.execute_reader(new_data))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;

        let dataset = self.dataset.bind(py);

        dataset.borrow_mut().ds = new_dataset;

        Ok(Self::build_stats(&stats, py)?.into())
    }

    pub fn execute_uncommitted<'a>(
        &mut self,
        new_data: &Bound<'a, PyAny>,
    ) -> PyResult<(PyLance<Transaction>, Bound<'a, PyDict>)> {
        let py = new_data.py();
        let new_data = convert_reader(new_data)?;

        let job = self
            .builder
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let UncommittedMergeInsert {
            transaction, stats, ..
        } = RT
            .spawn(Some(py), job.execute_uncommitted(new_data))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;

        let stats = Self::build_stats(&stats, py)?;

        Ok((PyLance(transaction), stats))
    }
}

impl MergeInsertBuilder {
    fn build_stats<'a>(stats: &MergeStats, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("num_inserted_rows", stats.num_inserted_rows)?;
        dict.set_item("num_updated_rows", stats.num_updated_rows)?;
        dict.set_item("num_deleted_rows", stats.num_deleted_rows)?;
        Ok(dict)
    }
}

pub fn transforms_from_python(transforms: &Bound<'_, PyAny>) -> PyResult<NewColumnTransform> {
    if let Ok(transforms) = transforms.downcast::<PyDict>() {
        let expressions = transforms
            .iter()
            .map(|(k, v)| {
                let col = k.extract::<String>()?;
                let expr = v.extract::<String>()?;
                Ok((col, expr))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(NewColumnTransform::SqlExpressions(expressions))
    } else {
        let append_schema: PyArrowType<ArrowSchema> =
            transforms.getattr("output_schema")?.extract()?;
        let output_schema = Arc::new(append_schema.0);

        let result_checkpoint: Option<PyObject> = transforms.getattr("cache")?.extract()?;
        let result_checkpoint = result_checkpoint.map(|c| PyBatchUDFCheckpointWrapper { inner: c });

        let udf_obj = transforms.into_py_any(transforms.py())?;
        let mapper = move |batch: &RecordBatch| -> lance::Result<RecordBatch> {
            Python::with_gil(|py| {
                let py_batch: PyArrowType<RecordBatch> = PyArrowType(batch.clone());
                let result = udf_obj
                    .call_method1(py, "_call", (py_batch,))
                    .map_err(|err| {
                        lance::Error::io(format_python_error(err, py).unwrap(), location!())
                    })?;
                let result_batch: PyArrowType<RecordBatch> = result
                    .extract(py)
                    .map_err(|err| lance::Error::io(err.to_string(), location!()))?;
                Ok(result_batch.0)
            })
        };

        Ok(NewColumnTransform::BatchUDF(BatchUDF {
            mapper: Box::new(mapper),
            output_schema,
            result_checkpoint: result_checkpoint
                .map(|c| Arc::new(c) as Arc<dyn UDFCheckpointStore>),
        }))
    }
}

/// Lance Dataset that will be wrapped by another class in Python
#[pyclass(name = "_Dataset", module = "_lib")]
#[derive(Clone)]
pub struct Dataset {
    #[pyo3(get)]
    uri: String,
    pub(crate) ds: Arc<LanceDataset>,
}

#[pymethods]
impl Dataset {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature=(uri, version=None, block_size=None, index_cache_size=None, metadata_cache_size=None, commit_handler=None, storage_options=None, manifest=None, metadata_cache_size_bytes=None))]
    fn new(
        py: Python,
        uri: String,
        version: Option<PyObject>,
        block_size: Option<usize>,
        index_cache_size: Option<usize>,
        metadata_cache_size: Option<usize>,
        commit_handler: Option<PyObject>,
        storage_options: Option<HashMap<String, String>>,
        manifest: Option<&[u8]>,
        metadata_cache_size_bytes: Option<usize>,
    ) -> PyResult<Self> {
        let mut params = ReadParams::default();
        if let Some(metadata_cache_size_bytes) = metadata_cache_size_bytes {
            params.metadata_cache_size_bytes(metadata_cache_size_bytes);
        } else if let Some(metadata_cache_size) = metadata_cache_size {
            #[allow(deprecated)]
            params.metadata_cache_size(metadata_cache_size);
        }
        if let Some(index_cache_size) = index_cache_size {
            params.index_cache_size(index_cache_size);
        }
        if let Some(block_size) = block_size {
            params.store_options = Some(ObjectStoreParams {
                block_size: Some(block_size),
                ..Default::default()
            });
        };
        if let Some(commit_handler) = commit_handler {
            let py_commit_lock = PyCommitLock::new(commit_handler);
            params.set_commit_lock(Arc::new(py_commit_lock));
        }

        let mut builder = DatasetBuilder::from_uri(&uri).with_read_params(params);
        if let Some(ver) = version {
            if let Ok(i) = ver.downcast_bound::<PyInt>(py) {
                let v: u64 = i.extract()?;
                builder = builder.with_version(v);
            } else if let Ok(v) = ver.downcast_bound::<PyString>(py) {
                let t: &str = &v.to_string_lossy();
                builder = builder.with_tag(t);
            } else {
                return Err(PyIOError::new_err(
                    "version must be an integer or a string.",
                ));
            };
        }
        if let Some(mut storage_options) = storage_options {
            if let Some(user_agent) = storage_options.get_mut("user_agent") {
                user_agent.push_str(&format!(" pylance/{}", env!("CARGO_PKG_VERSION")));
            } else {
                storage_options.insert(
                    "user_agent".to_string(),
                    format!("pylance/{}", env!("CARGO_PKG_VERSION")),
                );
            }

            builder = builder.with_storage_options(storage_options);
        }
        if let Some(manifest) = manifest {
            builder = builder.with_serialized_manifest(manifest).infer_error()?;
        }

        let dataset = RT.runtime.block_on(builder.load());

        match dataset {
            Ok(ds) => Ok(Self {
                uri,
                ds: Arc::new(ds),
            }),
            // TODO: return an appropriate error type, such as IOError or NotFound.
            Err(err) => Err(PyValueError::new_err(err.to_string())),
        }
    }

    pub fn __copy__(&self) -> Self {
        self.clone()
    }

    #[getter(max_field_id)]
    fn max_field_id(self_: PyRef<'_, Self>) -> PyResult<i32> {
        Ok(self_.ds.manifest().max_field_id())
    }

    #[getter(schema)]
    fn schema(self_: PyRef<'_, Self>) -> PyResult<PyObject> {
        let arrow_schema = ArrowSchema::from(self_.ds.schema());
        arrow_schema.to_pyarrow(self_.py())
    }

    #[getter(lance_schema)]
    fn lance_schema(self_: PyRef<'_, Self>) -> LanceSchema {
        LanceSchema(self_.ds.schema().clone())
    }

    fn replace_schema_metadata(&mut self, metadata: HashMap<String, String>) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        RT.block_on(None, new_self.replace_schema_metadata(metadata))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    fn replace_field_metadata(
        &mut self,
        field_name: &str,
        metadata: HashMap<String, String>,
    ) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        let field = new_self
            .schema()
            .field(field_name)
            .ok_or_else(|| PyKeyError::new_err(format!("Field \"{}\" not found", field_name)))?;
        let new_field_meta: HashMap<u32, HashMap<String, String>> =
            HashMap::from_iter(vec![(field.id as u32, metadata)]);
        RT.block_on(None, new_self.replace_field_metadata(new_field_meta))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    #[getter(data_storage_version)]
    fn data_storage_version(&self) -> PyResult<String> {
        Ok(self.ds.manifest().data_storage_format.version.clone())
    }

    /// Get index statistics
    fn index_statistics(&self, index_name: String) -> PyResult<String> {
        RT.runtime
            .block_on(self.ds.index_statistics(&index_name))
            .map_err(|err| match err {
                lance::Error::IndexNotFound { .. } => {
                    PyKeyError::new_err(format!("Index \"{}\" not found", index_name))
                }
                _ => PyIOError::new_err(format!(
                    "Failed to get index statistics for index {}: {}",
                    index_name, err
                )),
            })
    }

    fn serialized_manifest(&self, py: Python) -> PyObject {
        let manifest_bytes = self.ds.manifest().serialized();
        PyBytes::new(py, &manifest_bytes).into()
    }

    /// Load index metadata.
    ///
    /// This call will open the index and return its concrete index type.
    fn load_indices(self_: PyRef<'_, Self>) -> PyResult<Vec<PyObject>> {
        let index_metadata = RT
            .block_on(Some(self_.py()), self_.ds.load_indices())?
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let py = self_.py();
        index_metadata
            .iter()
            .map(|idx| {
                let dict = PyDict::new(py);
                let schema = self_.ds.schema();

                let idx_schema = schema.project_by_ids(idx.fields.as_slice(), true);

                let ds = self_.ds.clone();
                let idx_type = match RT.block_on(Some(self_.py()), async {
                    if let Some(system_index_type) = infer_system_index_type(idx) {
                        Ok::<_, lance::Error>(system_index_type.to_string())
                    } else {
                        let idx = ds
                            .open_generic_index(
                                &idx_schema.fields[0].name,
                                &idx.uuid.to_string(),
                                &NoOpMetricsCollector,
                            )
                            .await?;
                        Ok::<_, lance::Error>(idx.index_type().to_string())
                    }
                })? {
                    Ok(r) => r,
                    Err(error) => {
                        log::warn!("Cannot derive index type for {:?}: {}", idx, error);
                        // mark the type as unknown for any new index type
                        "Unknown".to_owned()
                    }
                };

                let field_names = idx_schema
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
                dict.set_item("type", idx_type).unwrap();
                dict.set_item("uuid", idx.uuid.to_string()).unwrap();
                dict.set_item("fields", field_names).unwrap();
                dict.set_item("version", idx.dataset_version).unwrap();
                dict.set_item("fragment_ids", fragment_set).unwrap();
                dict.into_py_any(py)
            })
            .collect::<PyResult<Vec<_>>>()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(columns=None, columns_with_transform=None, filter=None, prefilter=None, limit=None, offset=None, nearest=None, batch_size=None, io_buffer_size=None, batch_readahead=None, fragment_readahead=None, scan_in_order=None, fragments=None, with_row_id=None, with_row_address=None, use_stats=None, substrait_filter=None, fast_search=None, full_text_query=None, late_materialization=None, use_scalar_index=None, include_deleted_rows=None, scan_stats_callback=None, strict_batch_size=None))]
    fn scanner(
        self_: PyRef<'_, Self>,
        columns: Option<Vec<String>>,
        columns_with_transform: Option<Vec<(String, String)>>,
        filter: Option<String>,
        prefilter: Option<bool>,
        limit: Option<i64>,
        offset: Option<i64>,
        nearest: Option<&Bound<PyDict>>,
        batch_size: Option<usize>,
        io_buffer_size: Option<u64>,
        batch_readahead: Option<usize>,
        fragment_readahead: Option<usize>,
        scan_in_order: Option<bool>,
        fragments: Option<Vec<FileFragment>>,
        with_row_id: Option<bool>,
        with_row_address: Option<bool>,
        use_stats: Option<bool>,
        substrait_filter: Option<Vec<u8>>,
        fast_search: Option<bool>,
        full_text_query: Option<&Bound<'_, PyAny>>,
        late_materialization: Option<PyObject>,
        use_scalar_index: Option<bool>,
        include_deleted_rows: Option<bool>,
        scan_stats_callback: Option<&Bound<'_, PyAny>>,
        strict_batch_size: Option<bool>,
    ) -> PyResult<Scanner> {
        let mut scanner: LanceScanner = self_.ds.scan();

        if with_row_id.unwrap_or(false) {
            scanner.with_row_id();
        }

        if with_row_address.unwrap_or(false) {
            scanner.with_row_address();
        }

        match (columns, columns_with_transform) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "Cannot specify both columns and columns_with_transform",
                ))
            }
            (Some(c), None) => {
                scanner
                    .project(&c)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
            }
            (None, Some(ct)) => {
                scanner
                    .project_with_transform(&ct)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
            }
            (None, None) => {}
        }
        if let Some(f) = filter {
            if substrait_filter.is_some() {
                return Err(PyValueError::new_err(
                    "cannot specify both a string filter and a substrait filter",
                ));
            }
            scanner
                .filter(f.as_str())
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        if let Some(full_text_query) = full_text_query {
            let fts_query = if let Ok(full_text_query) = full_text_query.downcast::<PyDict>() {
                let mut query = full_text_query
                    .get_item("query")?
                    .ok_or_else(|| PyKeyError::new_err("query must be specified"))?
                    .to_string();
                let columns = if let Some(columns) = full_text_query.get_item("columns")? {
                    if columns.is_none() {
                        None
                    } else {
                        Some(
                            columns
                                .downcast::<PyList>()?
                                .iter()
                                .map(|c| c.extract::<String>())
                                .collect::<PyResult<Vec<String>>>()?,
                        )
                    }
                } else {
                    None
                };

                let is_phrase = query.len() >= 2 && query.starts_with('"') && query.ends_with('"');
                let is_multi_match = columns.as_ref().map(|cols| cols.len() > 1).unwrap_or(false);

                if is_phrase {
                    // Remove the surrounding quotes for phrase queries
                    query = query[1..query.len() - 1].to_string();
                }

                let query: FtsQuery = match (is_phrase, is_multi_match) {
                    (false, _) => MatchQuery::new(query).into(),
                    (true, false) => PhraseQuery::new(query).into(),
                    (true, true) => {
                        return Err(PyValueError::new_err(
                            "Phrase queries cannot be used with multiple columns.",
                        ));
                    }
                };
                let mut query = FullTextSearchQuery::new_query(query);
                if let Some(cols) = columns {
                    query = query.with_columns(&cols).map_err(|e| {
                        PyValueError::new_err(format!(
                            "Failed to set full text search columns: {}",
                            e
                        ))
                    })?;
                }
                query
            } else if let Ok(query) = full_text_query.downcast::<PyFullTextQuery>() {
                let query = query.borrow();
                FullTextSearchQuery::new_query(query.inner.clone())
            } else {
                return Err(PyValueError::new_err(
                    "query must be a string or a Query object",
                ));
            };

            scanner
                .full_text_search(fts_query)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        if let Some(f) = substrait_filter {
            scanner.filter_substrait(f.as_slice()).infer_error()?;
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
        if let Some(io_buffer_size) = io_buffer_size {
            scanner.io_buffer_size(io_buffer_size);
        }
        if let Some(batch_readahead) = batch_readahead {
            scanner.batch_readahead(batch_readahead);
        }

        if let Some(fragment_readahead) = fragment_readahead {
            scanner.fragment_readahead(fragment_readahead);
        }

        scanner.scan_in_order(scan_in_order.unwrap_or(true));

        if let Some(use_stats) = use_stats {
            scanner.use_stats(use_stats);
        }

        if let Some(true) = fast_search {
            scanner.fast_search();
        }

        if let Some(true) = include_deleted_rows {
            scanner.include_deleted_rows();
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

        if let Some(scan_stats_callback) = scan_stats_callback {
            let callback = Self::make_scan_stats_callback(scan_stats_callback.clone())?;
            scanner.scan_stats_callback(callback);
        }

        if let Some(late_materialization) = late_materialization {
            if let Ok(style_as_bool) = late_materialization.extract::<bool>(self_.py()) {
                if style_as_bool {
                    scanner.materialization_style(MaterializationStyle::AllLate);
                } else {
                    scanner.materialization_style(MaterializationStyle::AllEarly);
                }
            } else if let Ok(columns) = late_materialization.extract::<Vec<String>>(self_.py()) {
                scanner.materialization_style(
                    MaterializationStyle::all_early_except(&columns, self_.ds.schema())
                        .infer_error()?,
                );
            } else {
                return Err(PyValueError::new_err(
                    "late_materialization must be a bool or a list of strings",
                ));
            }
        }

        if let Some(use_scalar_index) = use_scalar_index {
            scanner.use_scalar_index(use_scalar_index);
        }

        if let Some(strict_batch_size) = strict_batch_size {
            scanner.strict_batch_size(strict_batch_size);
        }

        if let Some(nearest) = nearest {
            let column = nearest
                .get_item("column")?
                .ok_or_else(|| PyKeyError::new_err("Need column for nearest"))?
                .to_string();

            let qval = nearest
                .get_item("q")?
                .ok_or_else(|| PyKeyError::new_err("Need q for nearest"))?;
            let data = ArrayData::from_pyarrow_bound(&qval)?;
            let q = make_array(data);

            let k: usize = if let Some(k) = nearest.get_item("k")? {
                if k.is_none() {
                    // Use limit if k is not specified, default to 10.
                    limit.unwrap_or(10) as usize
                } else {
                    k.extract()?
                }
            } else {
                10
            };

            let mut minimum_nprobes = DEFAULT_NPROBS;
            let mut maximum_nprobes = None;

            if let Some(nprobes) = nearest.get_item("nprobes")? {
                if !nprobes.is_none() {
                    minimum_nprobes = nprobes.extract()?;
                    maximum_nprobes = Some(minimum_nprobes);
                }
            }

            if let Some(min_nprobes) = nearest.get_item("minimum_nprobes")? {
                if !min_nprobes.is_none() {
                    minimum_nprobes = min_nprobes.extract()?;
                }
            }

            if let Some(max_nprobes) = nearest.get_item("maximum_nprobes")? {
                if !max_nprobes.is_none() {
                    maximum_nprobes = Some(max_nprobes.extract()?);
                }
            }

            if minimum_nprobes > maximum_nprobes.unwrap_or(usize::MAX) {
                return Err(PyValueError::new_err(
                    "minimum_nprobes must be <= maximum_nprobes",
                ));
            }

            if minimum_nprobes < 1 {
                return Err(PyValueError::new_err("minimum_nprobes must be >= 1"));
            }

            if maximum_nprobes.unwrap_or(usize::MAX) < 1 {
                return Err(PyValueError::new_err("maximum_nprobes must be >= 1"));
            }

            let metric_type: Option<MetricType> =
                if let Some(metric) = nearest.get_item("metric")? {
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
            let refine_factor: Option<u32> = if let Some(rf) = nearest.get_item("refine_factor")? {
                if rf.is_none() {
                    None
                } else {
                    rf.extract()?
                }
            } else {
                None
            };

            let use_index: bool = if let Some(idx) = nearest.get_item("use_index")? {
                idx.extract()?
            } else {
                true
            };

            let ef: Option<usize> = if let Some(ef) = nearest.get_item("ef")? {
                if ef.is_none() {
                    None
                } else {
                    ef.extract()?
                }
            } else {
                None
            };

            let (_, element_type) = get_vector_type(self_.ds.schema(), &column)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let scanner = match element_type {
                DataType::UInt8 => {
                    let q = arrow::compute::cast(&q, &DataType::UInt8).map_err(|e| {
                        PyValueError::new_err(format!("Failed to cast q to binary vector: {}", e))
                    })?;
                    let q = q.as_primitive::<UInt8Type>();
                    scanner.nearest(&column, q, k)
                }
                _ => scanner.nearest(&column, &q, k),
            };
            scanner
                .map(|s| {
                    let mut s = s.minimum_nprobes(minimum_nprobes);
                    if let Some(maximum_nprobes) = maximum_nprobes {
                        s = s.maximum_nprobes(maximum_nprobes);
                    }
                    if let Some(factor) = refine_factor {
                        s = s.refine(factor);
                    }
                    if let Some(m) = metric_type {
                        s = s.distance_metric(m);
                    }
                    if let Some(ef) = ef {
                        s = s.ef(ef);
                    }
                    s.use_index(use_index);
                    s
                })
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }

        let scan = Arc::new(scanner);
        Ok(Scanner::new(scan))
    }

    #[pyo3(signature=(filter=None))]
    fn count_rows(&self, filter: Option<String>) -> PyResult<usize> {
        RT.runtime
            .block_on(self.ds.count_rows(filter))
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    #[pyo3(signature=(row_indices, columns = None, columns_with_transform = None))]
    fn take(
        self_: PyRef<'_, Self>,
        row_indices: Vec<u64>,
        columns: Option<Vec<String>>,
        columns_with_transform: Option<Vec<(String, String)>>,
    ) -> PyResult<PyObject> {
        let projection = match (columns, columns_with_transform) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "Cannot specify both columns and columns_with_transform",
                ))
            }
            (Some(columns), None) => {
                Ok(ProjectionRequest::from_columns(columns, self_.ds.schema()))
            }
            (None, Some(sql_exprs)) => Ok(ProjectionRequest::from_sql(sql_exprs)),
            (None, None) => Ok(ProjectionRequest::from_schema(self_.ds.schema().clone())),
        }
        .infer_error()?;
        let batch = RT
            .block_on(Some(self_.py()), self_.ds.take(&row_indices, projection))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;

        batch.to_pyarrow(self_.py())
    }

    #[pyo3(signature=(row_indices, columns = None, columns_with_transform = None))]
    fn take_rows(
        self_: PyRef<'_, Self>,
        row_indices: Vec<u64>,
        columns: Option<Vec<String>>,
        columns_with_transform: Option<Vec<(String, String)>>,
    ) -> PyResult<PyObject> {
        let projection = match (columns, columns_with_transform) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "Cannot specify both columns and columns_with_transform",
                ))
            }
            (Some(columns), None) => {
                Ok(ProjectionRequest::from_columns(columns, self_.ds.schema()))
            }
            (None, Some(sql_exprs)) => Ok(ProjectionRequest::from_sql(sql_exprs)),
            (None, None) => Ok(ProjectionRequest::from_schema(self_.ds.schema().clone())),
        }
        .infer_error()?;

        let batch = RT
            .block_on(
                Some(self_.py()),
                self_.ds.take_rows(&row_indices, projection),
            )?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;

        batch.to_pyarrow(self_.py())
    }

    fn take_blobs(
        self_: PyRef<'_, Self>,
        row_indices: Vec<u64>,
        blob_column: &str,
    ) -> PyResult<Vec<LanceBlobFile>> {
        let blobs = RT
            .block_on(
                Some(self_.py()),
                self_.ds.take_blobs(&row_indices, blob_column),
            )?
            .infer_error()?;
        Ok(blobs.into_iter().map(LanceBlobFile::from).collect())
    }

    fn take_blobs_by_indices(
        self_: PyRef<'_, Self>,
        row_indices: Vec<u64>,
        blob_column: &str,
    ) -> PyResult<Vec<LanceBlobFile>> {
        let blobs = RT
            .block_on(
                Some(self_.py()),
                self_.ds.take_blobs_by_indices(&row_indices, blob_column),
            )?
            .infer_error()?;
        Ok(blobs.into_iter().map(LanceBlobFile::from).collect())
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

    fn alter_columns(&mut self, alterations: &Bound<'_, PyList>) -> PyResult<()> {
        let alterations = alterations
            .iter()
            .map(|obj| {
                let obj = obj.downcast::<PyDict>()?;
                let path: String = obj
                    .get_item("path")?
                    .ok_or_else(|| PyValueError::new_err("path is required"))?
                    .extract()?;
                let name: Option<String> =
                    obj.get_item("name")?.map(|n| n.extract()).transpose()?;
                let nullable: Option<bool> =
                    obj.get_item("nullable")?.map(|n| n.extract()).transpose()?;
                let data_type: Option<PyArrowType<DataType>> = obj
                    .get_item("data_type")?
                    .map(|n| n.extract())
                    .transpose()?;

                for key in obj.keys().iter().map(|k| k.extract::<String>()) {
                    let k = key?;
                    if k != "path" && k != "name" && k != "nullable" && k != "data_type" {
                        return Err(PyValueError::new_err(format!(
                            "Unknown key: {}. Valid keys are name, nullable, and data_type.",
                            k
                        )));
                    }
                }

                if name.is_none() && nullable.is_none() && data_type.is_none() {
                    return Err(PyValueError::new_err(
                        "At least one of name, nullable, or data_type must be specified",
                    ));
                }

                let mut alteration = ColumnAlteration::new(path);
                if let Some(name) = name {
                    alteration = alteration.rename(name);
                }
                if let Some(nullable) = nullable {
                    alteration = alteration.set_nullable(nullable);
                }
                if let Some(data_type) = data_type {
                    alteration = alteration.cast_to(data_type.0);
                }
                Ok(alteration)
            })
            .collect::<PyResult<Vec<_>>>()?;

        let mut new_self = self.ds.as_ref().clone();
        new_self = RT
            .spawn(None, async move {
                new_self.alter_columns(&alterations).await.map(|_| new_self)
            })?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    fn merge(
        &mut self,
        reader: PyArrowType<ArrowArrayStreamReader>,
        left_on: String,
        right_on: String,
    ) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        let new_self = RT
            .spawn(None, async move {
                new_self
                    .merge(reader.0, &left_on, &right_on)
                    .await
                    .map(|_| new_self)
            })?
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

    #[pyo3(signature=(updates, predicate=None, conflict_retries=None, retry_timeout=None))]
    fn update(
        &mut self,
        updates: &Bound<'_, PyDict>,
        predicate: Option<&str>,
        conflict_retries: Option<u32>,
        retry_timeout: Option<std::time::Duration>,
    ) -> PyResult<PyObject> {
        let mut builder = UpdateBuilder::new(self.ds.clone());
        if let Some(predicate) = predicate {
            builder = builder
                .update_where(predicate)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }

        if let Some(retries) = conflict_retries {
            builder = builder.conflict_retries(retries);
        }

        if let Some(timeout) = retry_timeout {
            builder = builder.retry_timeout(timeout);
        }

        for (key, value) in updates {
            let column: PyBackedStr = key.downcast::<PyString>()?.clone().try_into()?;
            let expr: PyBackedStr = value.downcast::<PyString>()?.clone().try_into()?;

            builder = builder
                .set(column, &expr)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }

        let operation = builder
            .build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let new_self = RT
            .block_on(None, operation.execute())?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;

        self.ds = new_self.new_dataset;
        let update_dict = PyDict::new(updates.py());
        let num_rows_updated = new_self.rows_updated;
        update_dict.set_item("num_rows_updated", num_rows_updated)?;
        Ok(update_dict.into())
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
                    dict.set_item("metadata", tup.into_py_dict(py)?).unwrap();
                    dict.into_py_any(py)
                })
                .collect::<PyResult<Vec<_>>>()?;
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

    fn checkout_version(&self, py: Python, version: PyObject) -> PyResult<Self> {
        if let Ok(i) = version.downcast_bound::<PyInt>(py) {
            let ref_: u64 = i.extract()?;
            self._checkout_version(ref_)
        } else if let Ok(v) = version.downcast_bound::<PyString>(py) {
            let ref_: &str = &v.to_string_lossy();
            self._checkout_version(ref_)
        } else {
            Err(PyIOError::new_err(
                "version must be an integer or a string.",
            ))
        }
    }

    /// Restore the current version
    fn restore(&mut self) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        RT.block_on(None, new_self.restore())?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    /// Cleanup old versions from the dataset
    #[pyo3(signature = (older_than_micros, delete_unverified = None, error_if_tagged_old_versions = None))]
    fn cleanup_old_versions(
        &self,
        older_than_micros: i64,
        delete_unverified: Option<bool>,
        error_if_tagged_old_versions: Option<bool>,
    ) -> PyResult<CleanupStats> {
        let older_than = Duration::microseconds(older_than_micros);
        let cleanup_stats = RT
            .block_on(
                None,
                self.ds.cleanup_old_versions(
                    older_than,
                    delete_unverified,
                    error_if_tagged_old_versions,
                ),
            )?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Ok(CleanupStats {
            bytes_removed: cleanup_stats.bytes_removed,
            old_versions: cleanup_stats.old_versions,
        })
    }

    fn tags_ordered(self_: PyRef<'_, Self>, order: Option<String>) -> PyResult<PyObject> {
        let tags = self_
            .list_tags_ordered(order.as_deref())
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        Python::with_gil(|py| {
            let pylist = PyList::empty(py);

            for (tag_name, tag_content) in tags {
                let dict = PyDict::new(py);
                dict.set_item("version", tag_content.version)?;
                dict.set_item("manifest_size", tag_content.manifest_size)?;

                pylist.append((tag_name.as_str(), dict))?;
            }

            Ok(PyObject::from(pylist))
        })
    }

    fn tags(self_: PyRef<'_, Self>) -> PyResult<PyObject> {
        let tags = self_
            .list_tags()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        Python::with_gil(|py| {
            let pytags = PyDict::new(py);
            for (k, v) in tags.iter() {
                let dict = PyDict::new(py);
                dict.set_item("version", v.version).unwrap();
                dict.set_item("manifest_size", v.manifest_size).unwrap();
                pytags.set_item(k, dict.into_py_any(py)?).unwrap();
            }
            pytags.into_py_any(py)
        })
    }

    fn get_version(self_: PyRef<'_, Self>, tag: String) -> PyResult<u64> {
        let inner_result = RT.block_on(None, self_.ds.tags.get_version(&tag))?;

        inner_result.map_err(|err: lance::Error| match err {
            lance::Error::NotFound { .. } => {
                PyValueError::new_err(format!("Tag not found: {}", err))
            }
            lance::Error::RefNotFound { .. } => {
                PyValueError::new_err(format!("Ref not found: {}", err))
            }
            _ => PyIOError::new_err(format!("Storage error: {}", err)),
        })
    }

    fn create_tag(&mut self, tag: String, version: u64) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        RT.block_on(None, new_self.tags.create(tag.as_str(), version))?
            .map_err(|err| match err {
                lance::Error::NotFound { .. } => PyValueError::new_err(err.to_string()),
                lance::Error::RefConflict { .. } => PyValueError::new_err(err.to_string()),
                lance::Error::VersionNotFound { .. } => PyValueError::new_err(err.to_string()),
                _ => PyIOError::new_err(err.to_string()),
            })?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    fn delete_tag(&mut self, tag: String) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        RT.block_on(None, new_self.tags.delete(tag.as_str()))?
            .map_err(|err| match err {
                lance::Error::NotFound { .. } => PyValueError::new_err(err.to_string()),
                lance::Error::RefNotFound { .. } => PyValueError::new_err(err.to_string()),
                _ => PyIOError::new_err(err.to_string()),
            })?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    fn update_tag(&mut self, tag: String, version: u64) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        RT.block_on(None, new_self.tags.update(tag.as_str(), version))?
            .infer_error()?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    #[pyo3(signature = (**kwargs))]
    fn optimize_indices(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        let mut options: OptimizeOptions = Default::default();
        if let Some(kwargs) = kwargs {
            if let Some(num_indices_to_merge) = kwargs.get_item("num_indices_to_merge")? {
                options.num_indices_to_merge = num_indices_to_merge.extract()?;
            }
            if let Some(index_names) = kwargs.get_item("index_names")? {
                options.index_names = Some(
                    index_names
                        .extract::<Vec<String>>()
                        .map_err(|err| PyValueError::new_err(err.to_string()))?,
                );
            }
            if let Some(retrain) = kwargs.get_item("retrain")? {
                options.retrain = retrain.extract()?;
            }
        }
        RT.block_on(
            None,
            new_self
                .optimize_indices(&options)
                .map_err(|err| PyIOError::new_err(err.to_string())),
        )??;

        self.ds = Arc::new(new_self);
        Ok(())
    }

    #[pyo3(signature = (columns, index_type, name = None, replace = None, storage_options = None, kwargs = None))]
    fn create_index(
        &mut self,
        columns: Vec<PyBackedStr>,
        index_type: &str,
        name: Option<String>,
        replace: Option<bool>,
        storage_options: Option<HashMap<String, String>>,
        kwargs: Option<&Bound<PyDict>>,
    ) -> PyResult<()> {
        let columns: Vec<&str> = columns.iter().map(|s| &**s).collect();
        let index_type = index_type.to_uppercase();
        let idx_type = match index_type.as_str() {
            "BTREE" => IndexType::Scalar,
            "BITMAP" => IndexType::Bitmap,
            "NGRAM" => IndexType::NGram,
            "LABEL_LIST" => IndexType::LabelList,
            "INVERTED" | "FTS" => IndexType::Inverted,
            "IVF_FLAT" | "IVF_PQ" | "IVF_SQ" | "IVF_HNSW_FLAT" | "IVF_HNSW_PQ" | "IVF_HNSW_SQ" => {
                IndexType::Vector
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Index type '{index_type}' is not supported."
                )))
            }
        };

        log::info!("Creating index: type={}", index_type);
        let params: Box<dyn IndexParams> = match index_type.as_str() {
            "BTREE" => Box::<ScalarIndexParams>::default(),
            "BITMAP" => Box::new(ScalarIndexParams {
                // Temporary workaround until we add support for auto-detection of scalar index type
                force_index_type: Some(ScalarIndexType::Bitmap),
            }),
            "NGRAM" => Box::new(ScalarIndexParams {
                force_index_type: Some(ScalarIndexType::NGram),
            }),
            "LABEL_LIST" => Box::new(ScalarIndexParams {
                force_index_type: Some(ScalarIndexType::LabelList),
            }),
            "INVERTED" | "FTS" => {
                let mut params = InvertedIndexParams::default();
                if let Some(kwargs) = kwargs {
                    if let Some(with_position) = kwargs.get_item("with_position")? {
                        params = params.with_position(with_position.extract()?);
                    }
                    if let Some(base_tokenizer) = kwargs.get_item("base_tokenizer")? {
                        params = params.base_tokenizer(base_tokenizer.extract()?);
                    }
                    if let Some(language) = kwargs.get_item("language")? {
                        let language: PyBackedStr =
                            language.downcast::<PyString>()?.clone().try_into()?;
                        params = params.language(&language).map_err(|e| {
                            PyValueError::new_err(format!(
                                "can't set tokenizer language to {}: {:?}",
                                language, e
                            ))
                        })?;
                    }
                    if let Some(max_token_length) = kwargs.get_item("max_token_length")? {
                        params = params.max_token_length(max_token_length.extract()?);
                    }
                    if let Some(lower_case) = kwargs.get_item("lower_case")? {
                        params = params.lower_case(lower_case.extract()?);
                    }
                    if let Some(stem) = kwargs.get_item("stem")? {
                        params = params.stem(stem.extract()?);
                    }
                    if let Some(remove_stop_words) = kwargs.get_item("remove_stop_words")? {
                        params = params.remove_stop_words(remove_stop_words.extract()?);
                    }
                    if let Some(ascii_folding) = kwargs.get_item("ascii_folding")? {
                        params = params.ascii_folding(ascii_folding.extract()?);
                    }
                    if let Some(min_ngram_length) = kwargs.get_item("min_ngram_length")? {
                        params = params.ngram_min_length(min_ngram_length.extract()?);
                    }
                    if let Some(max_ngram_length) = kwargs.get_item("max_ngram_length")? {
                        params = params.ngram_max_length(max_ngram_length.extract()?);
                    }
                    if let Some(prefix_only) = kwargs.get_item("prefix_only")? {
                        params = params.ngram_prefix_only(prefix_only.extract()?);
                    }
                }
                Box::new(params)
            }
            _ => {
                let column_type = match self.ds.schema().field(columns[0]) {
                    Some(f) => f.data_type().clone(),
                    None => {
                        return Err(PyValueError::new_err("Column not found in dataset schema."))
                    }
                };
                prepare_vector_index_params(&index_type, &column_type, storage_options, kwargs)?
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

    fn drop_index(&mut self, name: &str) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        RT.block_on(None, new_self.drop_index(name))?
            .infer_error()?;
        self.ds = Arc::new(new_self);

        Ok(())
    }

    fn prewarm_index(&self, name: &str) -> PyResult<()> {
        RT.block_on(None, self.ds.prewarm_index(name))?
            .infer_error()
    }

    fn count_fragments(&self) -> usize {
        self.ds.count_fragments()
    }

    fn num_small_files(&self, max_rows_per_group: usize) -> PyResult<usize> {
        RT.block_on(None, self.ds.num_small_files(max_rows_per_group))
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    fn data_stats(&self) -> PyResult<PyLance<DataStatistics>> {
        RT.block_on(None, self.ds.calculate_data_stats())?
            .infer_error()
            .map(PyLance)
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

    fn index_cache_entry_count(&self) -> PyResult<usize> {
        Ok(self.ds.index_cache_entry_count())
    }

    fn index_cache_hit_rate(&self) -> PyResult<f32> {
        Ok(self.ds.index_cache_hit_rate())
    }

    fn session(&self) -> Session {
        Session::new(self.ds.session())
    }

    #[staticmethod]
    #[pyo3(signature = (dest, storage_options = None, ignore_not_found = None))]
    fn drop(
        dest: String,
        storage_options: Option<HashMap<String, String>>,
        ignore_not_found: Option<bool>,
    ) -> PyResult<()> {
        RT.spawn(None, async move {
            let (object_store, path) =
                object_store_from_uri_or_path(&dest, storage_options).await?;
            let result = object_store.remove_dir_all(path).await;

            match result {
                Ok(_) => Ok(()),
                Err(e) => {
                    let is_not_found = matches!(&e, lance_core::Error::NotFound { .. });

                    if let Some(true) = ignore_not_found {
                        if is_not_found {
                            Ok(())
                        } else {
                            Err(PyIOError::new_err(e.to_string()))
                        }
                    } else {
                        Err(PyIOError::new_err(e.to_string()))
                    }
                }
            }
        })?
    }

    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    #[pyo3(signature = (dest, operation, blobs_op=None, read_version = None, commit_lock = None, storage_options = None, enable_v2_manifest_paths = None, detached = None, max_retries = None))]
    fn commit(
        dest: PyWriteDest,
        operation: PyLance<Operation>,
        blobs_op: Option<PyLance<Operation>>,
        read_version: Option<u64>,
        commit_lock: Option<&Bound<'_, PyAny>>,
        storage_options: Option<HashMap<String, String>>,
        enable_v2_manifest_paths: Option<bool>,
        detached: Option<bool>,
        max_retries: Option<u32>,
    ) -> PyResult<Self> {
        let transaction = Transaction::new(
            read_version.unwrap_or_default(),
            operation.0,
            blobs_op.map(|op| op.0),
            None,
        );

        Self::commit_transaction(
            dest,
            PyLance(transaction),
            commit_lock,
            storage_options,
            enable_v2_manifest_paths,
            detached,
            max_retries,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    #[pyo3(signature = (dest, transaction, commit_lock = None, storage_options = None, enable_v2_manifest_paths = None, detached = None, max_retries = None))]
    fn commit_transaction(
        dest: PyWriteDest,
        transaction: PyLance<Transaction>,
        commit_lock: Option<&Bound<'_, PyAny>>,
        storage_options: Option<HashMap<String, String>>,
        enable_v2_manifest_paths: Option<bool>,
        detached: Option<bool>,
        max_retries: Option<u32>,
    ) -> PyResult<Self> {
        let object_store_params =
            storage_options
                .as_ref()
                .map(|storage_options| ObjectStoreParams {
                    storage_options: Some(storage_options.clone()),
                    ..Default::default()
                });

        let commit_handler = commit_lock
            .as_ref()
            .map(|commit_lock| {
                commit_lock
                    .into_py_any(commit_lock.py())
                    .map(|cl| Arc::new(PyCommitLock::new(cl)) as Arc<dyn CommitHandler>)
            })
            .transpose()?;

        let mut builder = CommitBuilder::new(dest.as_dest())
            .enable_v2_manifest_paths(enable_v2_manifest_paths.unwrap_or(false))
            .with_detached(detached.unwrap_or(false))
            .with_max_retries(max_retries.unwrap_or(20));

        if let Some(store_params) = object_store_params {
            builder = builder.with_store_params(store_params);
        }

        if let Some(commit_handler) = commit_handler {
            builder = builder.with_commit_handler(commit_handler);
        }

        let ds = RT
            .block_on(
                commit_lock.map(|cl| cl.py()),
                builder.execute(transaction.0),
            )?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;

        let uri = ds.uri().to_string();
        Ok(Self {
            ds: Arc::new(ds),
            uri,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (dest, transactions, commit_lock = None, storage_options = None, enable_v2_manifest_paths = None, detached = None, max_retries = None))]
    fn commit_batch(
        dest: PyWriteDest,
        transactions: Vec<PyLance<Transaction>>,
        commit_lock: Option<&Bound<'_, PyAny>>,
        storage_options: Option<HashMap<String, String>>,
        enable_v2_manifest_paths: Option<bool>,
        detached: Option<bool>,
        max_retries: Option<u32>,
    ) -> PyResult<(Self, PyLance<Transaction>)> {
        let object_store_params =
            storage_options
                .as_ref()
                .map(|storage_options| ObjectStoreParams {
                    storage_options: Some(storage_options.clone()),
                    ..Default::default()
                });

        let commit_handler = commit_lock
            .map(|commit_lock| {
                commit_lock
                    .into_py_any(commit_lock.py())
                    .map(|cl| Arc::new(PyCommitLock::new(cl)) as Arc<dyn CommitHandler>)
            })
            .transpose()?;

        let mut builder = CommitBuilder::new(dest.as_dest())
            .enable_v2_manifest_paths(enable_v2_manifest_paths.unwrap_or(false))
            .with_detached(detached.unwrap_or(false))
            .with_max_retries(max_retries.unwrap_or(20));

        if let Some(store_params) = object_store_params {
            builder = builder.with_store_params(store_params);
        }

        if let Some(commit_handler) = commit_handler {
            builder = builder.with_commit_handler(commit_handler);
        }

        let transactions = transactions
            .into_iter()
            .map(|transaction| transaction.0)
            .collect();

        let res = RT
            .block_on(None, builder.execute_batch(transactions))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        let uri = res.dataset.uri().to_string();
        let ds = Self {
            ds: Arc::new(res.dataset),
            uri,
        };

        Ok((ds, PyLance(res.merged)))
    }

    fn validate(&self) -> PyResult<()> {
        RT.block_on(None, self.ds.validate())?
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

    fn migrate_manifest_paths_v2(&mut self) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        RT.block_on(None, new_self.migrate_manifest_paths_v2())?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    fn drop_columns(&mut self, columns: Vec<String>) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        let columns: Vec<_> = columns.iter().map(|s| s.as_str()).collect();
        RT.block_on(None, new_self.drop_columns(&columns))?
            .map_err(|err| match err {
                lance::Error::InvalidInput { source, .. } => {
                    PyValueError::new_err(source.to_string())
                }
                _ => PyIOError::new_err(err.to_string()),
            })?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    #[pyo3(signature = (reader, batch_size = None))]
    fn add_columns_from_reader(
        &mut self,
        reader: &Bound<'_, PyAny>,
        batch_size: Option<u32>,
    ) -> PyResult<()> {
        let batches = ArrowArrayStreamReader::from_pyarrow_bound(reader)?;

        let transforms = NewColumnTransform::Reader(Box::new(batches));

        let mut new_self = self.ds.as_ref().clone();
        let new_self = RT
            .spawn(None, async move {
                new_self.add_columns(transforms, None, batch_size).await?;
                Ok(new_self)
            })?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);

        Ok(())
    }

    #[pyo3(signature = (transforms, read_columns = None, batch_size = None))]
    fn add_columns(
        &mut self,
        transforms: &Bound<'_, PyAny>,
        read_columns: Option<Vec<String>>,
        batch_size: Option<u32>,
    ) -> PyResult<()> {
        let transforms = transforms_from_python(transforms)?;

        let mut new_self = self.ds.as_ref().clone();
        let new_self = RT
            .spawn(None, async move {
                new_self
                    .add_columns(transforms, read_columns, batch_size)
                    .await?;
                Ok(new_self)
            })?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);

        Ok(())
    }

    /// Add NULL columns with only ArrowSchema.
    #[pyo3(signature = (schema))]
    fn add_columns_with_schema(&mut self, schema: PyArrowType<ArrowSchema>) -> PyResult<()> {
        let arrow_schema: &ArrowSchema = &schema.0;
        let transform = NewColumnTransform::AllNulls(Arc::new(arrow_schema.clone()));

        let mut new_self = self.ds.as_ref().clone();
        let new_self = RT
            .spawn(None, async move {
                new_self.add_columns(transform, None, None).await?;
                Ok(new_self)
            })?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    #[pyo3(signature = (index_name,partition_id, with_vector=false))]
    fn read_index_partition(
        &self,
        index_name: String,
        partition_id: usize,
        with_vector: bool,
    ) -> PyResult<PyArrowType<Box<dyn RecordBatchReader + Send>>> {
        let stream = RT
            .block_on(
                None,
                self.ds
                    .read_index_partition(&index_name, partition_id, with_vector),
            )?
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let reader = Box::new(LanceReader::from_stream(DatasetRecordBatchStream::new(
            stream,
        )));
        Ok(PyArrowType(reader))
    }

    #[pyo3(signature = (upsert_values))]
    fn update_config(&mut self, upsert_values: &Bound<'_, PyDict>) -> PyResult<()> {
        let upsert: HashMap<String, String> = upsert_values
            .iter()
            .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<String>()?)))
            .collect::<PyResult<_>>()?;

        let mut new_self = self.ds.as_ref().clone();
        RT.block_on(None, new_self.update_config(upsert))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    #[pyo3(signature = (keys))]
    fn delete_config_keys(&mut self, keys: Vec<String>) -> PyResult<()> {
        let key_refs: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
        let mut new_self = self.ds.as_ref().clone();
        RT.block_on(None, new_self.delete_config_keys(&key_refs))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);
        Ok(())
    }

    #[pyo3(signature = ())]
    fn config(&mut self) -> PyResult<PyObject> {
        let new_self = self.ds.as_ref().clone();

        let config = new_self
            .config()
            .map_err(|err| PyIOError::new_err(err.to_string()))?;

        self.ds = Arc::new(new_self);

        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for (k, v) in config {
                dict.set_item(k, v)?;
            }
            Ok(dict.into())
        })
    }

    #[pyo3(signature = (index_name))]
    fn get_ivf_model(
        &self,
        py: Python<'_>,
        index_name: &str,
    ) -> PyResult<Py<crate::indices::PyIvfModel>> {
        use crate::indices::PyIvfModel;
        let ivf_model = crate::RT.block_on(Some(py), async {
            use lance::index::DatasetIndexInternalExt;
            use lance_index::metrics::NoOpMetricsCollector;

            // Load index metadata and find the requested index
            let idx_metas = self.ds.load_indices().await.infer_error()?;
            let idx_meta = idx_metas
                .iter()
                .find(|idx| idx.name == index_name)
                .ok_or_else(|| {
                    PyValueError::new_err(format!("Index \"{}\" not found", index_name))
                })?;

            if idx_meta.fields.is_empty() {
                return Err(PyValueError::new_err("Index has no fields"));
            }

            let schema = self.ds.schema();
            let field = schema
                .field_by_id(idx_meta.fields[0])
                .ok_or_else(|| PyValueError::new_err("Failed to resolve index field"))?;
            let column_name = &field.name;

            let vindex = self
                .ds
                .open_vector_index(
                    column_name,
                    &idx_meta.uuid.to_string(),
                    &NoOpMetricsCollector,
                )
                .await
                .infer_error()?;

            Ok::<lance_index::vector::ivf::storage::IvfModel, pyo3::PyErr>(
                vindex.ivf_model().clone(),
            )
        })??;

        Py::new(py, PyIvfModel { inner: ivf_model })
    }

    #[pyo3(signature=(sql))]
    fn sql(&self, sql: String) -> PyResult<SqlQueryBuilder> {
        let mut ds = self.ds.as_ref().clone();
        let builder = ds.sql(&sql);
        Ok(SqlQueryBuilder { builder })
    }
}

#[pyclass(name = "SqlQuery", module = "_lib", subclass)]
#[derive(Clone)]
pub struct SqlQuery {
    builder: lance::dataset::sql::SqlQueryBuilder,
}

#[pymethods]
impl SqlQuery {
    /// Execute the query and return a list of RecordBatches.
    ///
    /// This is an eager operation that will load all results into memory.
    /// This corresponds to `into_batch_records` in Rust.
    fn to_batch_records(&self) -> PyResult<Vec<PyObject>> {
        use arrow::pyarrow::ToPyArrow;

        let builder = self.builder.clone();
        let batches = RT
            .block_on(None, async move {
                let query = builder.build().await?;
                query.into_batch_records().await
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))? // Handles tokio::JoinError
            .map_err(|e| PyValueError::new_err(e.to_string()))?; // Handles lance::Error

        Python::with_gil(|py| {
            batches
                .iter()
                .map(|rb| rb.to_pyarrow(py))
                .collect::<PyResult<Vec<PyObject>>>()
        })
    }

    /// Execute the query and return a RecordBatchReader.
    ///
    /// This is a lazy operation that will stream results.
    fn to_stream_reader(&self) -> PyResult<PyObject> {
        use crate::reader::LanceReader;
        use arrow::pyarrow::IntoPyArrow;
        use arrow_array::RecordBatchReader;
        use std::pin::Pin;

        let builder = self.builder.clone();
        let fut = Box::pin(async move {
            let query = builder.build().await?;
            let stream = query.into_stream().await;
            Ok::<Pin<Box<dyn datafusion::execution::RecordBatchStream + Send>>, lance::Error>(
                stream,
            )
        });

        let stream = RT
            .block_on(None, fut)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .map_err(|e: lance::Error| PyIOError::new_err(e.to_string()))?;

        let dataset_stream = DatasetRecordBatchStream::new(stream);
        let reader: Box<dyn RecordBatchReader + Send> =
            Box::new(LanceReader::from_stream(dataset_stream));
        Python::with_gil(|py| reader.into_pyarrow(py))
    }

    #[pyo3(signature = (verbose=false, analyze=false))]
    fn explain_plan(&self, verbose: bool, analyze: bool) -> PyResult<String> {
        let builder = self.builder.clone();
        let plan = RT
            .block_on(None, async move {
                let query = builder.build().await?;
                query.into_explain_plan(verbose, analyze).await
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(plan)
    }
}

#[pyclass(name = "SqlQueryBuilder", module = "_lib", subclass)]
#[derive(Clone)]
pub struct SqlQueryBuilder {
    builder: lance::dataset::sql::SqlQueryBuilder,
}

#[pymethods]
impl SqlQueryBuilder {
    #[pyo3(signature = (table_name))]
    fn table_name(&self, table_name: &str) -> Self {
        Self {
            builder: self.builder.clone().table_name(table_name),
        }
    }

    #[pyo3(signature = (with_row_id))]
    fn with_row_id(&self, with_row_id: bool) -> Self {
        Self {
            builder: self.builder.clone().with_row_id(with_row_id),
        }
    }

    #[pyo3(signature = (with_row_addr))]
    fn with_row_addr(&self, with_row_addr: bool) -> Self {
        Self {
            builder: self.builder.clone().with_row_addr(with_row_addr),
        }
    }

    /// Build the SQL query.
    fn build(&self) -> PyResult<SqlQuery> {
        Ok(SqlQuery {
            builder: self.builder.clone(),
        })
    }
}

#[derive(FromPyObject)]
pub enum PyWriteDest {
    Dataset(Dataset),
    Uri(PyBackedStr),
}

impl PyWriteDest {
    pub fn as_dest(&self) -> WriteDestination<'_> {
        match self {
            Self::Dataset(ds) => WriteDestination::Dataset(ds.ds.clone()),
            Self::Uri(uri) => WriteDestination::Uri(uri),
        }
    }
}

impl Dataset {
    fn _checkout_version(&self, version: impl Into<Ref> + std::marker::Send) -> PyResult<Self> {
        let ds = RT
            .block_on(None, self.ds.checkout_version(version))?
            .map_err(|err| match err {
                lance::Error::NotFound { .. } => PyValueError::new_err(err.to_string()),
                _ => PyIOError::new_err(err.to_string()),
            })?;

        Ok(Self {
            ds: Arc::new(ds),
            uri: self.uri.clone(),
        })
    }

    fn list_versions(&self) -> ::lance::error::Result<Vec<Version>> {
        RT.runtime.block_on(self.ds.versions())
    }

    fn list_tags(&self) -> ::lance::error::Result<HashMap<String, TagContents>> {
        RT.runtime.block_on(self.ds.tags.list())
    }

    fn list_tags_ordered(
        &self,
        order: Option<&str>,
    ) -> ::lance::error::Result<Vec<(String, TagContents)>> {
        let ordering = match order {
            Some("asc") => Some(std::cmp::Ordering::Less),
            Some("desc") => Some(std::cmp::Ordering::Greater),
            Some(invalid_order) => {
                let error_msg = format!(
                    "Invalid sort order '{}'. Valid values are: asc, desc",
                    invalid_order
                );
                return Err(::lance::error::Error::InvalidInput {
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        error_msg,
                    )),
                    location: location!(),
                });
            }
            None => None,
        };
        RT.runtime
            .block_on(async { self.ds.tags.list_tags_ordered(ordering).await })
    }

    fn make_scan_stats_callback(callback: Bound<'_, PyAny>) -> PyResult<ExecutionStatsCallback> {
        if !callback.is_callable() {
            return Err(PyValueError::new_err("Callback must be callable"));
        }

        let callback = callback.unbind();

        Ok(Arc::new(move |stats| {
            Python::with_gil(|py| {
                let stats = ScanStatistics::from_lance(stats);
                match callback.call1(py, (stats,)) {
                    Ok(_) => (),
                    Err(e) => {
                        // Don't fail scan if callback fails
                        error!("Error in scan stats callback: {}", e);
                    }
                }
            });
        }))
    }
}

#[pyfunction(name = "_write_dataset")]
pub fn write_dataset(
    reader: &Bound<'_, PyAny>,
    dest: PyWriteDest,
    options: &Bound<'_, PyDict>,
) -> PyResult<Dataset> {
    let params = get_write_params(options)?;
    let py = options.py();
    let ds = if reader.is_instance_of::<Scanner>() {
        let scanner: Scanner = reader.extract()?;
        let batches = RT
            .block_on(Some(py), scanner.to_reader())?
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        RT.block_on(
            Some(py),
            LanceDataset::write(batches, dest.as_dest(), params),
        )?
        .map_err(|err| PyIOError::new_err(err.to_string()))?
    } else {
        let batches = ArrowArrayStreamReader::from_pyarrow_bound(reader)?;
        RT.block_on(
            Some(py),
            LanceDataset::write(batches, dest.as_dest(), params),
        )?
        .map_err(|err| PyIOError::new_err(err.to_string()))?
    };
    Ok(Dataset {
        uri: ds.uri().to_string(),
        ds: Arc::new(ds),
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

pub fn get_commit_handler(options: &Bound<'_, PyDict>) -> PyResult<Option<Arc<dyn CommitHandler>>> {
    Ok(if options.is_none() {
        None
    } else if let Ok(Some(commit_handler)) = options.get_item("commit_handler") {
        Some(Arc::new(PyCommitLock::new(
            commit_handler.into_pyobject(options.py())?.into(),
        )))
    } else {
        None
    })
}

// Gets a value from the dictionary and attempts to extract it to
// the desired type.  If the value is None then it treats it as if
// it were never present in the dictionary.  If the value is not
// None it will try and parse it and parsing failures will be
// returned (e.g. a parsing failure is not considered `None`)
fn get_dict_opt<'a, 'py, D: FromPyObject<'a>>(
    dict: &'a Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<D>> {
    let value = dict.get_item(key)?;
    value
        .and_then(|v| {
            if v.is_none() {
                None
            } else {
                Some(v.extract::<D>())
            }
        })
        .transpose()
}

pub fn get_write_params(options: &Bound<'_, PyDict>) -> PyResult<Option<WriteParams>> {
    let params = if options.is_none() {
        None
    } else {
        let mut p = WriteParams::default();
        if let Some(mode) = get_dict_opt::<String>(options, "mode")? {
            p.mode = parse_write_mode(mode.as_str())?;
        };
        if let Some(maybe_nrows) = get_dict_opt::<usize>(options, "max_rows_per_file")? {
            p.max_rows_per_file = maybe_nrows;
        }
        if let Some(maybe_nrows) = get_dict_opt::<usize>(options, "max_rows_per_group")? {
            p.max_rows_per_group = maybe_nrows;
        }
        if let Some(maybe_nbytes) = get_dict_opt::<usize>(options, "max_bytes_per_file")? {
            p.max_bytes_per_file = maybe_nbytes;
        }
        if let Some(data_storage_version) = get_dict_opt::<String>(options, "data_storage_version")?
        {
            p.data_storage_version = Some(data_storage_version.parse().infer_error()?);
        }
        if let Some(progress) = get_dict_opt::<PyObject>(options, "progress")? {
            p.progress = Arc::new(PyWriteProgress::new(progress.into_py_any(options.py())?));
        }

        if let Some(storage_options) =
            get_dict_opt::<HashMap<String, String>>(options, "storage_options")?
        {
            p.store_params = Some(ObjectStoreParams {
                storage_options: Some(storage_options),
                ..Default::default()
            });
        }

        if let Some(enable_move_stable_row_ids) =
            get_dict_opt::<bool>(options, "enable_move_stable_row_ids")?
        {
            p.enable_move_stable_row_ids = enable_move_stable_row_ids;
        }
        if let Some(enable_v2_manifest_paths) =
            get_dict_opt::<bool>(options, "enable_v2_manifest_paths")?
        {
            p.enable_v2_manifest_paths = enable_v2_manifest_paths;
        }

        if let Some(auto_cleanup) = get_dict_opt::<Bound<PyAny>>(options, "auto_cleanup_options")? {
            let mut auto_cleanup_params = AutoCleanupParams::default();

            auto_cleanup_params.interval = auto_cleanup
                .get_item("interval")
                .and_then(|i| i.extract::<usize>())
                .ok()
                .unwrap_or(auto_cleanup_params.interval);

            auto_cleanup_params.older_than = auto_cleanup
                .get_item("older_than_seconds")
                .and_then(|i| i.extract::<i64>())
                .ok()
                .map(TimeDelta::seconds)
                .unwrap_or(auto_cleanup_params.older_than);

            p.auto_cleanup = Some(auto_cleanup_params);
        } else {
            p.auto_cleanup = None;
        }

        p.commit_handler = get_commit_handler(options)?;

        Some(p)
    };
    Ok(params)
}

fn prepare_vector_index_params(
    index_type: &str,
    column_type: &DataType,
    storage_options: Option<HashMap<String, String>>,
    kwargs: Option<&Bound<PyDict>>,
) -> PyResult<Box<dyn IndexParams>> {
    let mut m_type = MetricType::L2;
    let mut ivf_params = IvfBuildParams::default();
    let mut hnsw_params = HnswBuildParams::default();
    let mut pq_params = PQBuildParams::default();
    let mut sq_params = SQBuildParams::default();
    let mut index_file_version = IndexFileVersion::V3;

    if let Some(kwargs) = kwargs {
        // Parse metric type
        if let Some(mt) = kwargs.get_item("metric_type")? {
            m_type = MetricType::try_from(mt.to_string().to_lowercase().as_str())
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }

        // Parse sample rate
        if let Some(sample_rate) = kwargs.get_item("sample_rate")? {
            let sample_rate: usize = sample_rate.extract()?;
            ivf_params.sample_rate = sample_rate;
            pq_params.sample_rate = sample_rate;
            sq_params.sample_rate = sample_rate;
        }

        // Parse IVF params
        if let Some(n) = kwargs.get_item("num_partitions")? {
            ivf_params.num_partitions = n.extract()?
        };

        if let Some(n) = kwargs.get_item("shuffle_partition_concurrency")? {
            ivf_params.shuffle_partition_concurrency = n.extract()?
        };

        if let Some(c) = kwargs.get_item("ivf_centroids")? {
            let batch = RecordBatch::from_pyarrow_bound(&c)?;
            if "_ivf_centroids" != batch.schema().field(0).name() {
                return Err(PyValueError::new_err(
                    "Expected '_ivf_centroids' as the first column name.",
                ));
            }

            // It's important that the centroids are the same data type
            // as the vectors that will be indexed.
            let mut centroids: Arc<dyn Array> = batch.column(0).clone();
            if centroids.data_type() != column_type {
                centroids = lance_arrow::cast::cast_with_options(
                    centroids.as_ref(),
                    column_type,
                    &Default::default(),
                )
                .map_err(|e| {
                    PyValueError::new_err(format!("Failed to cast centroids to column type: {}", e))
                })?;
            }
            let centroids = as_fixed_size_list_array(centroids.as_ref());

            ivf_params.centroids = Some(Arc::new(centroids.clone()))
        };

        if let Some(f) = kwargs.get_item("precomputed_partitions_file")? {
            ivf_params.precomputed_partitions_file = Some(f.to_string());
        };

        if let Some(storage_options) = storage_options {
            ivf_params.storage_options = Some(storage_options);
        }

        match (
                kwargs.get_item("precomputed_shuffle_buffers")?,
                kwargs.get_item("precomputed_shuffle_buffers_path")?
            ) {
                (Some(l), Some(p)) => {
                    let path = Path::parse(p.to_string()).map_err(|e| {
                        PyValueError::new_err(format!(
                            "Failed to parse precomputed_shuffle_buffers_path: {}",
                            e
                        ))
                    })?;
                    let list = l.downcast::<PyList>()?
                        .iter()
                        .map(|f| f.to_string())
                        .collect();
                    ivf_params.precomputed_shuffle_buffers = Some((path, list));
                },
                (None, None) => {},
                _ => {
                    return Err(PyValueError::new_err(
                        "precomputed_shuffle_buffers and precomputed_shuffle_buffers_path must be specified together."
                    ))
                }
            }

        // Parse HNSW params
        if let Some(max_level) = kwargs.get_item("max_level")? {
            hnsw_params.max_level = max_level.extract()?;
        }

        if let Some(m) = kwargs.get_item("m")? {
            hnsw_params.m = m.extract()?;
        }

        if let Some(ef_c) = kwargs.get_item("ef_construction")? {
            hnsw_params.ef_construction = ef_c.extract()?;
        }

        // Parse PQ params
        if let Some(n) = kwargs.get_item("num_bits")? {
            pq_params.num_bits = n.extract()?
        };

        if let Some(n) = kwargs.get_item("num_sub_vectors")? {
            pq_params.num_sub_vectors = n.extract()?
        };

        if let Some(c) = kwargs.get_item("pq_codebook")? {
            let batch = RecordBatch::from_pyarrow_bound(&c)?;
            if "_pq_codebook" != batch.schema().field(0).name() {
                return Err(PyValueError::new_err(
                    "Expected '_pq_codebook' as the first column name.",
                ));
            }
            let codebook = as_fixed_size_list_array(batch.column(0));
            pq_params.codebook = Some(codebook.values().clone())
        };

        if let Some(version) = kwargs.get_item("index_file_version")? {
            let version: String = version.extract()?;
            index_file_version = IndexFileVersion::try_from(&version)
                .map_err(|e| PyValueError::new_err(format!("Invalid index_file_version: {e}")))?;
        }
    }

    let mut params = match index_type {
        "IVF_FLAT" => Ok(Box::new(VectorIndexParams::ivf_flat(
            ivf_params.num_partitions,
            m_type,
        ))),

        "IVF_PQ" => Ok(Box::new(VectorIndexParams::with_ivf_pq_params(
            m_type, ivf_params, pq_params,
        ))),

        "IVF_SQ" => Ok(Box::new(VectorIndexParams::with_ivf_sq_params(
            m_type, ivf_params, sq_params,
        ))),

        "IVF_HNSW_FLAT" => Ok(Box::new(VectorIndexParams::ivf_hnsw(
            m_type,
            ivf_params,
            hnsw_params,
        ))),

        "IVF_HNSW_PQ" => Ok(Box::new(VectorIndexParams::with_ivf_hnsw_pq_params(
            m_type,
            ivf_params,
            hnsw_params,
            pq_params,
        ))),

        "IVF_HNSW_SQ" => Ok(Box::new(VectorIndexParams::with_ivf_hnsw_sq_params(
            m_type,
            ivf_params,
            hnsw_params,
            sq_params,
        ))),

        _ => Err(PyValueError::new_err(format!(
            "Index type '{index_type}' is not supported."
        ))),
    }?;
    params.version(index_file_version);
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
    async fn begin(&self, fragment: &Fragment) -> lance::Result<()> {
        let json_str = serde_json::to_string(fragment)?;

        Python::with_gil(|py| -> PyResult<()> {
            self.py_obj
                .call_method(py, "_do_begin", (json_str,), None)?;
            Ok(())
        })
        .map_err(|e| {
            lance::Error::io(
                format!("Failed to call begin() on WriteFragmentProgress: {}", e),
                location!(),
            )
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
        .map_err(|e| {
            lance::Error::io(
                format!("Failed to call complete() on WriteFragmentProgress: {}", e),
                location!(),
            )
        })?;
        Ok(())
    }
}

/// Formats a Python error just as it would in Python interpreter.
fn format_python_error(e: PyErr, py: Python) -> PyResult<String> {
    let sys_mod = py.import("sys")?;
    // the traceback is the third element of the tuple returned by sys.exc_info()
    let traceback = sys_mod.call_method0("exc_info")?.get_item(2)?;

    let tracback_mod = py.import("traceback")?;
    let fmt_func = tracback_mod.getattr("format_exception")?;
    let e_type = e.get_type(py).to_owned();
    let formatted = fmt_func.call1((e_type, &e, traceback))?;
    let lines: Vec<String> = formatted.extract()?;
    Ok(lines.join(""))
}

struct PyBatchUDFCheckpointWrapper {
    inner: PyObject,
}

impl PyBatchUDFCheckpointWrapper {
    fn batch_info_to_py(&self, info: &BatchInfo, py: Python) -> PyResult<PyObject> {
        self.inner
            .getattr(py, "BatchInfo")?
            .call1(py, (info.fragment_id, info.batch_index))
    }
}

impl UDFCheckpointStore for PyBatchUDFCheckpointWrapper {
    fn get_batch(&self, info: &BatchInfo) -> lance::Result<Option<RecordBatch>> {
        Python::with_gil(|py| {
            let info = self.batch_info_to_py(info, py)?;
            let batch = self.inner.call_method1(py, "get_batch", (info,))?;
            let batch: Option<PyArrowType<RecordBatch>> = batch.extract(py)?;
            Ok(batch.map(|b| b.0))
        })
        .map_err(|err: PyErr| {
            lance_core::Error::io(
                format!("Failed to call get_batch() on UDFCheckpointer: {}", err),
                location!(),
            )
        })
    }

    fn get_fragment(&self, fragment_id: u32) -> lance::Result<Option<Fragment>> {
        let fragment_data = Python::with_gil(|py| {
            let fragment = self
                .inner
                .call_method1(py, "get_fragment", (fragment_id,))?;
            let fragment: Option<String> = fragment.extract(py)?;
            Ok(fragment)
        })
        .map_err(|err: PyErr| {
            lance_core::Error::io(
                format!("Failed to call get_fragment() on UDFCheckpointer: {}", err),
                location!(),
            )
        })?;
        fragment_data
            .map(|data| {
                serde_json::from_str(&data).map_err(|err| {
                    lance::Error::io(
                        format!("Failed to deserialize fragment data: {}", err),
                        location!(),
                    )
                })
            })
            .transpose()
    }

    fn insert_batch(&self, info: BatchInfo, batch: RecordBatch) -> lance::Result<()> {
        Python::with_gil(|py| {
            let info = self.batch_info_to_py(&info, py)?;
            let batch = PyArrowType(batch);
            self.inner.call_method1(py, "insert_batch", (info, batch))?;
            Ok(())
        })
        .map_err(|err: PyErr| {
            lance_core::Error::io(
                format!("Failed to call insert_batch() on UDFCheckpointer: {}", err),
                location!(),
            )
        })
    }

    fn insert_fragment(&self, fragment: Fragment) -> lance_core::Result<()> {
        let data = serde_json::to_string(&fragment).map_err(|err| {
            lance_core::Error::io(
                format!("Failed to serialize fragment data: {}", err),
                location!(),
            )
        })?;
        Python::with_gil(|py| {
            self.inner
                .call_method1(py, "insert_fragment", (fragment.id, data))?;
            Ok(())
        })
        .map_err(|err: PyErr| {
            lance_core::Error::io(
                format!(
                    "Failed to call insert_fragment() on UDFCheckpointer: {}",
                    err
                ),
                location!(),
            )
        })
    }
}

#[pyclass(name = "PyFullTextQuery")]
#[derive(Debug, Clone)]
pub struct PyFullTextQuery {
    pub(crate) inner: FtsQuery,
}

#[pymethods]
impl PyFullTextQuery {
    #[staticmethod]
    #[pyo3(signature = (query, column, boost=1.0, fuzziness=Some(0), max_expansions=50, operator="OR", prefix_length=0))]
    fn match_query(
        query: String,
        column: String,
        boost: f32,
        fuzziness: Option<u32>,
        max_expansions: usize,
        operator: &str,
        prefix_length: u32,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MatchQuery::new(query)
                .with_column(Some(column))
                .with_boost(boost)
                .with_fuzziness(fuzziness)
                .with_max_expansions(max_expansions)
                .with_operator(
                    Operator::try_from(operator)
                        .map_err(|e| PyValueError::new_err(format!("Invalid operator: {}", e)))?,
                )
                .with_prefix_length(prefix_length)
                .into(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (query, column, slop))]
    fn phrase_query(query: String, column: String, slop: u32) -> PyResult<Self> {
        Ok(Self {
            inner: PhraseQuery::new(query)
                .with_column(Some(column))
                .with_slop(slop)
                .into(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (positive, negative,negative_boost=None))]
    fn boost_query(positive: Self, negative: Self, negative_boost: Option<f32>) -> PyResult<Self> {
        Ok(Self {
            inner: BoostQuery::new(positive.inner, negative.inner, negative_boost).into(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (query, columns, boosts=None, operator="OR"))]
    fn multi_match_query(
        query: String,
        columns: Vec<String>,
        boosts: Option<Vec<f32>>,
        operator: &str,
    ) -> PyResult<Self> {
        let q = MultiMatchQuery::try_new(query, columns)
            .map_err(|e| PyValueError::new_err(format!("Invalid query: {}", e)))?;
        let q = if let Some(boosts) = boosts {
            q.try_with_boosts(boosts)
                .map_err(|e| PyValueError::new_err(format!("Invalid boosts: {}", e)))?
        } else {
            q
        };

        let op = Operator::try_from(operator)
            .map_err(|e| PyValueError::new_err(format!("Invalid operator: {}", e)))?;

        Ok(Self {
            inner: q.with_operator(op).into(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (queries))]
    fn boolean_query(queries: Vec<(String, Self)>) -> PyResult<Self> {
        let mut sub_queries = Vec::with_capacity(queries.len());
        for (occur, q) in queries {
            let occur = Occur::try_from(occur.as_str())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            sub_queries.push((occur, q.inner));
        }

        Ok(Self {
            inner: BooleanQuery::new(sub_queries).into(),
        })
    }
}
