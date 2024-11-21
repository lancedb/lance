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

use std::collections::HashMap;
use std::str;
use std::sync::Arc;

use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::*;
use arrow_array::{Float32Array, RecordBatch, RecordBatchReader};
use arrow_data::ArrayData;
use arrow_schema::{DataType, Schema as ArrowSchema};
use async_trait::async_trait;
use blob::LanceBlobFile;
use chrono::Duration;

use arrow_array::Array;
use futures::{StreamExt, TryFutureExt};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::refs::{Ref, TagContents};
use lance::dataset::scanner::MaterializationStyle;
use lance::dataset::transaction::{
    RewriteGroup as LanceRewriteGroup, RewrittenIndex as LanceRewrittenIndex, Transaction,
};
use lance::dataset::{
    fragment::FileFragment as LanceFileFragment, progress::WriteFragmentProgress,
    scanner::Scanner as LanceScanner, transaction::Operation as LanceOperation,
    Dataset as LanceDataset, MergeInsertBuilder as LanceMergeInsertBuilder, ReadParams,
    UpdateBuilder, Version, WhenMatched, WhenNotMatched, WhenNotMatchedBySource, WriteMode,
    WriteParams,
};
use lance::dataset::{
    BatchInfo, BatchUDF, CommitBuilder, NewColumnTransform, UDFCheckpointStore, WriteDestination,
};
use lance::dataset::{ColumnAlteration, ProjectionRequest};
use lance::index::{vector::VectorIndexParams, DatasetIndexInternalExt};
use lance_arrow::as_fixed_size_list_array;
use lance_core::datatypes::Schema;
use lance_index::scalar::InvertedIndexParams;
use lance_index::{
    optimize::OptimizeOptions,
    scalar::{FullTextSearchQuery, ScalarIndexParams, ScalarIndexType},
    vector::{
        hnsw::builder::HnswBuildParams, ivf::IvfBuildParams, pq::PQBuildParams,
        sq::builder::SQBuildParams,
    },
    DatasetIndexExt, IndexParams, IndexType,
};
use lance_io::object_store::ObjectStoreParams;
use lance_linalg::distance::MetricType;
use lance_table::format::Fragment;
use lance_table::format::Index;
use lance_table::io::commit::CommitHandler;
use object_store::path::Path;
use pyo3::exceptions::{PyNotImplementedError, PyStopIteration, PyTypeError};
use pyo3::types::{PyBytes, PyInt, PyList, PySet, PyString, PyTuple};
use pyo3::{
    exceptions::{PyIOError, PyKeyError, PyValueError},
    pyclass,
    types::{IntoPyDict, PyDict},
    PyObject, PyResult,
};
use pyo3::{intern, prelude::*};
use snafu::{location, Location};
use uuid::Uuid;

use crate::error::PythonErrorExt;
use crate::fragment::{FileFragment, FragmentMetadata};
use crate::schema::LanceSchema;
use crate::session::Session;
use crate::RT;
use crate::{LanceReader, Scanner};

use self::cleanup::CleanupStats;
use self::commit::PyCommitLock;

pub mod blob;
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
    // Note: the field ids here are wrong.
    Schema::try_from(arrow_schema).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to convert Arrow schema to Lance schema: {}",
            e
        ))
    })
}

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
    pub fn new(dataset: &PyAny, on: &PyAny) -> PyResult<Self> {
        let dataset: Py<Dataset> = dataset.extract()?;
        let ds = dataset.borrow(on.py()).ds.clone();
        // Either a single string, which we put in a vector or an iterator
        // of strings, which we collect into a vector
        let on = PyAny::downcast::<PyString>(on)
            .map(|val| vec![val.to_string()])
            .or_else(|_| {
                let iterator = on.iter().map_err(|_| {
                    PyTypeError::new_err(
                        "The `on` argument to merge_insert must be a str or iterable of str",
                    )
                })?;
                let mut keys = Vec::new();
                for key in iterator {
                    keys.push(PyAny::downcast::<PyString>(key?)?.to_string());
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

    pub fn execute(&mut self, new_data: &Bound<PyAny>) -> PyResult<PyObject> {
        let py = new_data.py();
        let new_data = convert_reader(new_data)?;

        let job = self
            .builder
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let new_self = RT
            .spawn(Some(py), job.execute_reader(new_data))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;

        let dataset = self.dataset.as_ref(py);

        dataset.borrow_mut().ds = new_self.0;
        let merge_stats = new_self.1;
        let merge_dict = PyDict::new(py);
        merge_dict.set_item("num_inserted_rows", merge_stats.num_inserted_rows)?;
        merge_dict.set_item("num_updated_rows", merge_stats.num_updated_rows)?;
        merge_dict.set_item("num_deleted_rows", merge_stats.num_deleted_rows)?;

        Ok(merge_dict.into())
    }
}

#[pyclass(name = "_RewriteGroup", module = "_lib")]
#[derive(Clone)]
pub struct RewriteGroup(LanceRewriteGroup);

#[pymethods]
impl RewriteGroup {
    #[new]
    pub fn new(old_fragments: Vec<FragmentMetadata>, new_fragments: Vec<FragmentMetadata>) -> Self {
        let old_fragments = into_fragments(old_fragments);
        let new_fragments = into_fragments(new_fragments);
        Self(LanceRewriteGroup {
            old_fragments,
            new_fragments,
        })
    }
}

#[pyclass(name = "_RewrittenIndex", module = "_lib")]
#[derive(Clone)]
pub struct RewrittenIndex(LanceRewrittenIndex);

#[pymethods]
impl RewrittenIndex {
    #[new]
    pub fn new(old_index: String, new_index: String) -> PyResult<Self> {
        let old_id: Uuid = old_index
            .parse()
            .map_err(|e: uuid::Error| PyValueError::new_err(e.to_string()))?;
        let new_id: Uuid = new_index
            .parse()
            .map_err(|e: uuid::Error| PyValueError::new_err(e.to_string()))?;
        Ok(Self(LanceRewrittenIndex { old_id, new_id }))
    }
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
        let op = LanceOperation::Overwrite {
            fragments,
            schema,
            config_upsert_values: None,
        };
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
    fn merge(fragments: Vec<FragmentMetadata>, schema: LanceSchema) -> PyResult<Self> {
        let schema = schema.0;
        let fragments = into_fragments(fragments);
        let op = LanceOperation::Merge { fragments, schema };
        Ok(Self(op))
    }

    #[staticmethod]
    fn restore(version: u64) -> PyResult<Self> {
        let op = LanceOperation::Restore { version };
        Ok(Self(op))
    }

    #[staticmethod]
    fn rewrite(
        groups: Vec<RewriteGroup>,
        rewritten_indices: Vec<RewrittenIndex>,
    ) -> PyResult<Self> {
        let groups = groups.into_iter().map(|g| g.0).collect();
        let rewritten_indices = rewritten_indices.into_iter().map(|r| r.0).collect();
        let op = LanceOperation::Rewrite {
            groups,
            rewritten_indices,
        };
        Ok(Self(op))
    }

    #[staticmethod]
    fn create_index(
        uuid: String,
        name: String,
        fields: Vec<i32>,
        dataset_version: u64,
        fragment_ids: &PySet,
    ) -> PyResult<Self> {
        let fragment_ids: Vec<u32> = fragment_ids
            .iter()
            .map(|item| item.extract::<u32>())
            .collect::<PyResult<Vec<u32>>>()?;
        let new_indices = vec![Index {
            uuid: Uuid::parse_str(&uuid).map_err(|e| PyValueError::new_err(e.to_string()))?,
            name,
            fields,
            dataset_version,
            fragment_bitmap: Some(fragment_ids.into_iter().collect()),
            // TODO: we should use lance::dataset::Dataset::commit_existing_index once
            // we have a way to determine index details from an existing index.
            index_details: None,
        }];
        let op = LanceOperation::CreateIndex {
            new_indices,
            removed_indices: vec![],
        };
        Ok(Self(op))
    }

    /// Convert to a pydict that can be used as kwargs into the Operation dataclasses
    fn to_dict<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        let dict = PyDict::new_bound(py);
        match &self.0 {
            LanceOperation::Append { fragments } => {
                let fragments = fragments
                    .iter()
                    .cloned()
                    .map(FragmentMetadata::new)
                    .map(|f| f.into_py(py))
                    .collect::<Vec<_>>();
                dict.set_item("fragments", fragments).unwrap();
            }
            _ => {
                return Err(PyNotImplementedError::new_err(format!(
                    "Operation.to_dict is not implemented for this operation: {:?}",
                    self.0
                )));
            }
        }

        Ok(dict)
    }
}

pub fn transforms_from_python(transforms: &PyAny) -> PyResult<NewColumnTransform> {
    if let Ok(transforms) = transforms.extract::<&PyDict>() {
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

        let udf_obj = transforms.to_object(transforms.py());
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
    ) -> PyResult<Self> {
        let mut params = ReadParams {
            index_cache_size: index_cache_size.unwrap_or(DEFAULT_INDEX_CACHE_SIZE),
            metadata_cache_size: metadata_cache_size.unwrap_or(DEFAULT_METADATA_CACHE_SIZE),
            store_options: Some(ObjectStoreParams {
                block_size,
                ..Default::default()
            }),
            ..Default::default()
        };

        if let Some(commit_handler) = commit_handler {
            let py_commit_lock = PyCommitLock::new(commit_handler);
            params.set_commit_lock(Arc::new(py_commit_lock));
        }

        let mut builder = DatasetBuilder::from_uri(&uri).with_read_params(params);
        if let Some(ver) = version {
            if let Ok(i) = ver.downcast::<PyInt>(py) {
                let v: u64 = i.extract()?;
                builder = builder.with_version(v);
            } else if let Ok(v) = ver.downcast::<PyString>(py) {
                let t: &str = v.extract()?;
                builder = builder.with_tag(t);
            } else {
                return Err(PyIOError::new_err(
                    "version must be an integer or a string.",
                ));
            };
        }
        if let Some(storage_options) = storage_options {
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

    #[getter(schema)]
    fn schema(self_: PyRef<'_, Self>) -> PyResult<PyObject> {
        let arrow_schema = ArrowSchema::from(self_.ds.schema());
        arrow_schema.to_pyarrow(self_.py())
    }

    #[getter(lance_schema)]
    fn lance_schema(self_: PyRef<'_, Self>) -> LanceSchema {
        LanceSchema(self_.ds.schema().clone())
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

                let is_vector = idx_schema
                    .fields
                    .iter()
                    .any(|f| matches!(f.data_type(), DataType::FixedSizeList(_, _)));

                let idx_type = if is_vector {
                    IndexType::Vector
                } else {
                    let ds = self_.ds.clone();
                    RT.block_on(Some(self_.py()), async {
                        let scalar_idx = ds
                            .open_scalar_index(&idx_schema.fields[0].name, &idx.uuid.to_string())
                            .await?;
                        Ok::<_, lance::Error>(scalar_idx.index_type())
                    })?
                    .map_err(|e| PyIOError::new_err(e.to_string()))?
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
                dict.set_item("type", idx_type.to_string()).unwrap();
                dict.set_item("uuid", idx.uuid.to_string()).unwrap();
                dict.set_item("fields", field_names).unwrap();
                dict.set_item("version", idx.dataset_version).unwrap();
                dict.set_item("fragment_ids", fragment_set).unwrap();
                Ok(dict.to_object(py))
            })
            .collect::<PyResult<Vec<_>>>()
    }

    #[allow(clippy::too_many_arguments)]
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
        full_text_query: Option<&PyDict>,
        late_materialization: Option<PyObject>,
        use_scalar_index: Option<bool>,
    ) -> PyResult<Scanner> {
        let mut scanner: LanceScanner = self_.ds.scan();
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
            let query = full_text_query
                .get_item("query")?
                .ok_or_else(|| PyKeyError::new_err("Need column for full text search"))?
                .to_string();
            let columns = if let Some(columns) = full_text_query.get_item("columns")? {
                if columns.is_none() {
                    None
                } else {
                    Some(
                        PyAny::downcast::<PyList>(columns)?
                            .iter()
                            .map(|c| c.extract::<String>())
                            .collect::<PyResult<Vec<String>>>()?,
                    )
                }
            } else {
                None
            };
            let full_text_query = FullTextSearchQuery::new(query).columns(columns);
            scanner
                .full_text_search(full_text_query)
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

        if with_row_id.unwrap_or(false) {
            scanner.with_row_id();
        }

        if with_row_address.unwrap_or(false) {
            scanner.with_row_address();
        }

        if let Some(use_stats) = use_stats {
            scanner.use_stats(use_stats);
        }

        if let Some(true) = fast_search {
            scanner.fast_search();
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

        if let Some(nearest) = nearest {
            let column = nearest
                .get_item("column")?
                .ok_or_else(|| PyKeyError::new_err("Need column for nearest"))?
                .to_string();

            let qval = nearest
                .get_item("q")?
                .ok_or_else(|| PyKeyError::new_err("Need q for nearest"))?;
            let data = ArrayData::from_pyarrow_bound(&qval)?;
            let q = Float32Array::from(data);

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

            let nprobes: usize = if let Some(nprobes) = nearest.get_item("nprobes")? {
                if nprobes.is_none() {
                    DEFAULT_NPROBS
                } else {
                    nprobes.extract()?
                }
            } else {
                DEFAULT_NPROBS
            };

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

    fn count_rows(&self, filter: Option<String>) -> PyResult<usize> {
        RT.runtime
            .block_on(self.ds.count_rows(filter))
            .map_err(|err| PyIOError::new_err(err.to_string()))
    }

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

    fn alter_columns(&mut self, alterations: &PyList) -> PyResult<()> {
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

    fn update(&mut self, updates: &PyDict, predicate: Option<&str>) -> PyResult<PyObject> {
        let mut builder = UpdateBuilder::new(self.ds.clone());
        if let Some(predicate) = predicate {
            builder = builder
                .update_where(predicate)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }

        for (key, value) in updates {
            let column: &str = key.extract()?;
            let expr: &str = value.extract()?;

            builder = builder
                .set(column, expr)
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

    fn checkout_version(&self, py: Python, version: PyObject) -> PyResult<Self> {
        if let Ok(i) = version.downcast::<PyInt>(py) {
            let ref_: u64 = i.extract()?;
            self._checkout_version(ref_)
        } else if let Ok(v) = version.downcast::<PyString>(py) {
            let ref_: &str = v.extract()?;
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
                dict.to_object(py);
                pytags.set_item(k, dict).unwrap();
            }
            Ok(pytags.to_object(py))
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
    fn optimize_indices(&mut self, kwargs: Option<&PyDict>) -> PyResult<()> {
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

    fn create_index(
        &mut self,
        columns: Vec<&str>,
        index_type: &str,
        name: Option<String>,
        replace: Option<bool>,
        storage_options: Option<HashMap<String, String>>,
        kwargs: Option<&Bound<PyDict>>,
    ) -> PyResult<()> {
        let index_type = index_type.to_uppercase();
        let idx_type = match index_type.as_str() {
            "BTREE" => IndexType::Scalar,
            "BITMAP" => IndexType::Bitmap,
            "LABEL_LIST" => IndexType::LabelList,
            "INVERTED" | "FTS" => IndexType::Inverted,
            "IVF_PQ" | "IVF_HNSW_PQ" | "IVF_HNSW_SQ" => IndexType::Vector,
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
            "LABEL_LIST" => Box::new(ScalarIndexParams {
                force_index_type: Some(ScalarIndexType::LabelList),
            }),
            "INVERTED" | "FTS" => {
                let mut params = InvertedIndexParams::default();
                if let Some(kwargs) = kwargs {
                    if let Some(with_position) = kwargs.get_item("with_position")? {
                        params.with_position = with_position.extract()?;
                    }
                    if let Some(base_tokenizer) = kwargs.get_item("base_tokenizer")? {
                        params.tokenizer_config = params
                            .tokenizer_config
                            .base_tokenizer(base_tokenizer.extract()?);
                    }
                    if let Some(language) = kwargs.get_item("language")? {
                        let language = language.extract()?;
                        params.tokenizer_config =
                            params.tokenizer_config.language(language).map_err(|e| {
                                PyValueError::new_err(format!(
                                    "can't set tokenizer language to {}: {:?}",
                                    language, e
                                ))
                            })?;
                    }
                    if let Some(max_token_length) = kwargs.get_item("max_token_length")? {
                        params.tokenizer_config = params
                            .tokenizer_config
                            .max_token_length(max_token_length.extract()?);
                    }
                    if let Some(lower_case) = kwargs.get_item("lower_case")? {
                        params.tokenizer_config =
                            params.tokenizer_config.lower_case(lower_case.extract()?);
                    }
                    if let Some(stem) = kwargs.get_item("stem")? {
                        params.tokenizer_config = params.tokenizer_config.stem(stem.extract()?);
                    }
                    if let Some(remove_stop_words) = kwargs.get_item("remove_stop_words")? {
                        params.tokenizer_config = params
                            .tokenizer_config
                            .remove_stop_words(remove_stop_words.extract()?);
                    }
                    if let Some(ascii_folding) = kwargs.get_item("ascii_folding")? {
                        params.tokenizer_config = params
                            .tokenizer_config
                            .ascii_folding(ascii_folding.extract()?);
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

    fn index_cache_entry_count(&self) -> PyResult<usize> {
        Ok(self.ds.index_cache_entry_count())
    }

    fn index_cache_hit_rate(&self) -> PyResult<f32> {
        Ok(self.ds.index_cache_hit_rate())
    }

    fn session(&self) -> Session {
        Session::new(self.ds.session())
    }

    #[allow(clippy::too_many_arguments)]
    #[staticmethod]
    fn commit(
        dest: &Bound<PyAny>,
        operation: Operation,
        read_version: Option<u64>,
        commit_lock: Option<&PyAny>,
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

        let commit_handler = commit_lock.map(|commit_lock| {
            Arc::new(PyCommitLock::new(commit_lock.to_object(commit_lock.py())))
                as Arc<dyn CommitHandler>
        });

        let dest = if dest.is_instance_of::<Self>() {
            let dataset: Self = dest.extract()?;
            WriteDestination::Dataset(dataset.ds.clone())
        } else {
            WriteDestination::Uri(dest.extract()?)
        };

        let transaction =
            Transaction::new(read_version.unwrap_or_default(), operation.0, None, None);

        let mut builder = CommitBuilder::new(dest)
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
            .block_on(commit_lock.map(|cl| cl.py()), builder.execute(transaction))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;

        let uri = ds.uri().to_string();
        Ok(Self {
            ds: Arc::new(ds),
            uri,
        })
    }

    #[staticmethod]
    fn commit_batch<'py>(
        dest: &Bound<'py, PyAny>,
        transactions: Vec<Bound<'py, PyAny>>,
        commit_lock: Option<&'py PyAny>,
        storage_options: Option<HashMap<String, String>>,
        enable_v2_manifest_paths: Option<bool>,
        detached: Option<bool>,
        max_retries: Option<u32>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let object_store_params =
            storage_options
                .as_ref()
                .map(|storage_options| ObjectStoreParams {
                    storage_options: Some(storage_options.clone()),
                    ..Default::default()
                });

        let commit_handler = commit_lock.map(|commit_lock| {
            Arc::new(PyCommitLock::new(commit_lock.to_object(commit_lock.py())))
                as Arc<dyn CommitHandler>
        });

        let py = dest.py();
        let dest = if dest.is_instance_of::<Dataset>() {
            let dataset: Dataset = dest.extract()?;
            WriteDestination::Dataset(dataset.ds.clone())
        } else {
            WriteDestination::Uri(dest.extract()?)
        };

        let mut builder = CommitBuilder::new(dest)
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
            .map(|transaction| extract_transaction(&transaction))
            .collect::<PyResult<Vec<_>>>()?;

        let res = RT
            .block_on(Some(py), builder.execute_batch(transactions))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        let uri = res.dataset.uri().to_string();
        let ds = Self {
            ds: Arc::new(res.dataset),
            uri,
        };
        let merged = export_transaction(&res.merged, py)?.to_object(py);
        let ds = ds.into_py(py);
        Ok(PyTuple::new_bound(py, [ds, merged]))
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

    fn drop_columns(&mut self, columns: Vec<&str>) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
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

    fn add_columns_from_reader(
        &mut self,
        reader: &Bound<PyAny>,
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

    fn add_columns(
        &mut self,
        transforms: &PyAny,
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
}

#[pyfunction(name = "_write_dataset")]
pub fn write_dataset(
    reader: &Bound<PyAny>,
    dest: &Bound<PyAny>,
    options: &PyDict,
) -> PyResult<Dataset> {
    let params = get_write_params(options)?;
    let py = options.py();
    let dest = if dest.is_instance_of::<Dataset>() {
        let dataset: Dataset = dest.extract()?;
        WriteDestination::Dataset(dataset.ds.clone())
    } else {
        WriteDestination::Uri(dest.extract()?)
    };
    let ds = if reader.is_instance_of::<Scanner>() {
        let scanner: Scanner = reader.extract()?;
        let batches = RT
            .block_on(Some(py), scanner.to_reader())?
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        RT.block_on(Some(py), LanceDataset::write(batches, dest, params))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?
    } else {
        let batches = ArrowArrayStreamReader::from_pyarrow_bound(reader)?;
        RT.block_on(Some(py), LanceDataset::write(batches, dest, params))?
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

pub fn get_commit_handler(options: &PyDict) -> Option<Arc<dyn CommitHandler>> {
    if options.is_none() {
        None
    } else if let Ok(Some(commit_handler)) = options.get_item("commit_handler") {
        Some(Arc::new(PyCommitLock::new(
            commit_handler.to_object(options.py()),
        )))
    } else {
        None
    }
}

// Gets a value from the dictionary and attempts to extract it to
// the desired type.  If the value is None then it treats it as if
// it were never present in the dictionary.  If the value is not
// None it will try and parse it and parsing failures will be
// returned (e.g. a parsing failure is not considered `None`)
fn get_dict_opt<'a, D: FromPyObject<'a>>(dict: &'a PyDict, key: &str) -> PyResult<Option<D>> {
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

pub fn get_write_params(options: &PyDict) -> PyResult<Option<WriteParams>> {
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
            p.progress = Arc::new(PyWriteProgress::new(progress.to_object(options.py())));
        }

        if let Some(storage_options) =
            get_dict_opt::<HashMap<String, String>>(options, "storage_options")?
        {
            p.store_params = Some(ObjectStoreParams {
                storage_options: Some(storage_options),
                ..Default::default()
            });
        }

        if let Some(enable_v2_manifest_paths) =
            get_dict_opt::<bool>(options, "enable_v2_manifest_paths")?
        {
            p.enable_v2_manifest_paths = enable_v2_manifest_paths;
        }

        p.commit_handler = get_commit_handler(options);

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
    }

    match index_type {
        "IVF_PQ" => Ok(Box::new(VectorIndexParams::with_ivf_pq_params(
            m_type, ivf_params, pq_params,
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
    }
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

/// py_transaction is a dataclass with attributes
/// read_version: int
/// uuid: str
/// operation: LanceOperation.BaseOperation
/// blobs_op: Optional[LanceOperation.BaseOperation] = None
fn extract_transaction(py_transaction: &Bound<PyAny>) -> PyResult<Transaction> {
    let py = py_transaction.py();
    let read_version = py_transaction.getattr("read_version")?.extract()?;
    let uuid = py_transaction.getattr("uuid")?.extract()?;
    let operation: Operation = py_transaction
        .getattr("operation")?
        .call_method0(intern!(py, "_to_inner"))?
        .extract()?;
    let operation = operation.0;
    let blobs_op: Option<Operation> = {
        let blobs_op: Option<Bound<PyAny>> = py_transaction.getattr("blobs_op")?.extract()?;
        if let Some(blobs_op) = blobs_op {
            Some(blobs_op.call_method0(intern!(py, "_to_inner"))?.extract()?)
        } else {
            None
        }
    };
    let blobs_op = blobs_op.map(|op| op.0);
    Ok(Transaction {
        read_version,
        uuid,
        operation,
        blobs_op,
        tag: None,
    })
}

// Exports to a pydict of kwargs to instantiation the python Transaction dataclass.
fn export_transaction<'a>(
    transaction: &Transaction,
    py: Python<'a>,
) -> PyResult<Bound<'a, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("read_version", transaction.read_version)?;
    dict.set_item("uuid", transaction.uuid.clone())?;
    dict.set_item(
        "operation",
        Operation(transaction.operation.clone()).to_dict(py)?,
    )?;
    dict.set_item(
        "blobs_op",
        transaction
            .blobs_op
            .clone()
            .map(Operation)
            .map(|op| op.to_dict(py))
            .transpose()?,
    )?;
    Ok(dict)
}
