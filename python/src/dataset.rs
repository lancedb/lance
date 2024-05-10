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
use chrono::Duration;

use arrow_array::Array;
use futures::{StreamExt, TryFutureExt};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::transaction::validate_operation;
use lance::dataset::ColumnAlteration;
use lance::dataset::{
    fragment::FileFragment as LanceFileFragment, progress::WriteFragmentProgress,
    scanner::Scanner as LanceScanner, transaction::Operation as LanceOperation,
    Dataset as LanceDataset, MergeInsertBuilder as LanceMergeInsertBuilder, ReadParams,
    UpdateBuilder, Version, WhenMatched, WhenNotMatched, WhenNotMatchedBySource, WriteMode,
    WriteParams,
};
use lance::dataset::{BatchInfo, BatchUDF, NewColumnTransform, UDFCheckpointStore};
use lance::index::{scalar::ScalarIndexParams, vector::VectorIndexParams};
use lance_arrow::as_fixed_size_list_array;
use lance_core::datatypes::Schema;
use lance_index::optimize::OptimizeOptions;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::{
    vector::{ivf::IvfBuildParams, pq::PQBuildParams},
    DatasetIndexExt, IndexParams, IndexType,
};
use lance_io::object_store::ObjectStoreParams;
use lance_linalg::distance::MetricType;
use lance_table::format::Fragment;
use lance_table::io::commit::CommitHandler;
use object_store::path::Path;
use pyo3::exceptions::{PyStopIteration, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySet, PyString};
use pyo3::{
    exceptions::{PyIOError, PyKeyError, PyValueError},
    pyclass,
    types::{IntoPyDict, PyBool, PyDict, PyInt, PyLong},
    PyObject, PyResult,
};
use snafu::{location, Location};

use crate::fragment::{FileFragment, FragmentMetadata};
use crate::schema::LanceSchema;
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
    // Note: the field ids here are wrong.
    Schema::try_from(arrow_schema).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to convert Arrow schema to Lance schema: {}",
            e
        ))
    })
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

    pub fn execute(&mut self, new_data: &PyAny) -> PyResult<()> {
        let py = new_data.py();

        let new_data: Box<dyn RecordBatchReader + Send> = if new_data.is_instance_of::<Scanner>() {
            let scanner: Scanner = new_data.extract()?;
            Box::new(
                RT.spawn(Some(py), async move { scanner.to_reader().await })?
                    .map_err(|err| PyValueError::new_err(err.to_string()))?,
            )
        } else {
            Box::new(ArrowArrayStreamReader::from_pyarrow(new_data)?)
        };

        let job = self
            .builder
            .try_build()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let new_self = RT
            .spawn(Some(py), job.execute_reader(new_data))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;

        let dataset = self.dataset.as_ref(py);

        dataset.borrow_mut().ds = new_self;

        Ok(())
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
    #[new]
    fn new(
        uri: String,
        version: Option<u64>,
        block_size: Option<usize>,
        index_cache_size: Option<usize>,
        metadata_cache_size: Option<usize>,
        commit_handler: Option<PyObject>,
        storage_options: Option<HashMap<String, String>>,
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
            builder = builder.with_version(ver);
        }
        if let Some(storage_options) = storage_options {
            builder = builder.with_storage_options(storage_options);
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

                let idx_schema = schema.project_by_ids(idx.fields.as_slice());

                let is_vector = idx_schema
                    .fields
                    .iter()
                    .any(|f| matches!(f.data_type(), DataType::FixedSizeList(_, _)));

                let idx_type = if is_vector {
                    IndexType::Vector
                } else {
                    IndexType::Scalar
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
                dict.to_object(py)
            })
            .collect::<Vec<_>>())
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
        nearest: Option<&PyDict>,
        batch_size: Option<usize>,
        batch_readahead: Option<usize>,
        fragment_readahead: Option<usize>,
        scan_in_order: Option<bool>,
        fragments: Option<Vec<FileFragment>>,
        with_row_id: Option<bool>,
        use_stats: Option<bool>,
        substrait_filter: Option<Vec<u8>>,
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
        if let Some(f) = substrait_filter {
            RT.runtime
                .block_on(scanner.filter_substrait(f.as_slice()))
                .map_err(|err| PyIOError::new_err(err.to_string()))?;
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

        if let Some(use_stats) = use_stats {
            scanner.use_stats(use_stats);
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
                .get_item("column")?
                .ok_or_else(|| PyKeyError::new_err("Need column for nearest"))?
                .to_string();

            let qval = nearest
                .get_item("q")?
                .ok_or_else(|| PyKeyError::new_err("Need q for nearest"))?;
            let data = ArrayData::from_pyarrow(qval)?;
            let q = Float32Array::from(data);

            let k: usize = if let Some(k) = nearest.get_item("k")? {
                if k.is_none() {
                    // Use limit if k is not specified, default to 10.
                    limit.unwrap_or(10) as usize
                } else {
                    PyAny::downcast::<PyLong>(k)?.extract()?
                }
            } else {
                10
            };

            let nprobes: usize = if let Some(nprobes) = nearest.get_item("nprobes")? {
                if nprobes.is_none() {
                    DEFAULT_NPROBS
                } else {
                    PyAny::downcast::<PyLong>(nprobes)?.extract()?
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
                    PyAny::downcast::<PyLong>(rf)?.extract()?
                }
            } else {
                None
            };

            let use_index: bool = if let Some(idx) = nearest.get_item("use_index")? {
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

    fn count_rows(&self, filter: Option<String>) -> PyResult<usize> {
        RT.runtime
            .block_on(self.ds.count_rows(filter))
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

        batch.to_pyarrow(self_.py())
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

        batch.to_pyarrow(self_.py())
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

    fn update(&mut self, updates: &PyDict, predicate: Option<&str>) -> PyResult<()> {
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

        self.ds = new_self;

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

    fn checkout_version(&self, version: u64) -> PyResult<Self> {
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

    #[pyo3(signature = (**kwargs))]
    fn optimize_indices(&mut self, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut new_self = self.ds.as_ref().clone();
        let mut options: OptimizeOptions = Default::default();
        if let Some(kwargs) = kwargs {
            if let Some(num_indices_to_merge) = kwargs.get_item("num_indices_to_merge")? {
                options.num_indices_to_merge = num_indices_to_merge.extract()?;
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
        kwargs: Option<&PyDict>,
    ) -> PyResult<()> {
        let index_type = index_type.to_uppercase();
        let idx_type = match index_type.as_str() {
            "BTREE" => IndexType::Scalar,
            "IVF_PQ" | "IVF_HNSW_PQ" | "IVF_HNSW_SQ" => IndexType::Vector,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Index type '{index_type}' is not supported."
                )))
            }
        };

        // Only VectorParams are supported.
        let params: Box<dyn IndexParams> = if index_type == "BTREE" {
            Box::<ScalarIndexParams>::default()
        } else {
            let column_type = match self.ds.schema().field(columns[0]) {
                Some(f) => f.data_type().clone(),
                None => return Err(PyValueError::new_err("Column not found in dataset schema.")),
            };
            prepare_vector_index_params(&index_type, &column_type, kwargs)?
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

    #[staticmethod]
    fn commit(
        dataset_uri: &str,
        operation: Operation,
        read_version: Option<u64>,
        commit_lock: Option<&PyAny>,
    ) -> PyResult<Self> {
        let commit_handler = commit_lock.map(|commit_lock| {
            Arc::new(PyCommitLock::new(commit_lock.to_object(commit_lock.py())))
                as Arc<dyn CommitHandler>
        });
        let ds = RT
            .block_on(commit_lock.map(|cl| cl.py()), async move {
                let dataset = match DatasetBuilder::from_uri(dataset_uri).load().await {
                    Ok(ds) => Some(ds),
                    Err(lance::Error::DatasetNotFound { .. }) => None,
                    Err(err) => return Err(err),
                };
                let manifest = dataset.as_ref().map(|ds| ds.manifest());
                validate_operation(manifest, &operation.0)?;
                LanceDataset::commit(dataset_uri, operation.0, read_version, None, commit_handler)
                    .await
            })?
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

    fn add_columns(
        &mut self,
        transforms: &PyAny,
        read_columns: Option<Vec<String>>,
    ) -> PyResult<()> {
        let transforms = if let Ok(transforms) = transforms.extract::<&PyDict>() {
            let expressions = transforms
                .iter()
                .map(|(k, v)| {
                    let col = k.extract::<String>()?;
                    let expr = v.extract::<String>()?;
                    Ok((col, expr))
                })
                .collect::<PyResult<Vec<_>>>()?;
            NewColumnTransform::SqlExpressions(expressions)
        } else {
            let append_schema: PyArrowType<ArrowSchema> =
                transforms.getattr("output_schema")?.extract()?;
            let output_schema = Arc::new(append_schema.0);

            let result_checkpoint: Option<PyObject> = transforms.getattr("cache")?.extract()?;
            let result_checkpoint =
                result_checkpoint.map(|c| PyBatchUDFCheckpointWrapper { inner: c });

            let udf_obj = transforms.to_object(transforms.py());
            let mapper = move |batch: &RecordBatch| -> lance::Result<RecordBatch> {
                Python::with_gil(|py| {
                    let py_batch: PyArrowType<RecordBatch> = PyArrowType(batch.clone());
                    let result = udf_obj
                        .call_method1(py, "_call", (py_batch,))
                        .map_err(|err| lance::Error::IO {
                            message: format_python_error(err, py).unwrap(),
                            location: location!(),
                        })?;
                    let result_batch: PyArrowType<RecordBatch> =
                        result.extract(py).map_err(|err| lance::Error::IO {
                            message: err.to_string(),
                            location: location!(),
                        })?;
                    Ok(result_batch.0)
                })
            };

            NewColumnTransform::BatchUDF(BatchUDF {
                mapper: Box::new(mapper),
                output_schema,
                result_checkpoint: result_checkpoint
                    .map(|c| Arc::new(c) as Arc<dyn UDFCheckpointStore>),
            })
        };

        let mut new_self = self.ds.as_ref().clone();
        let new_self = RT
            .spawn(None, async move {
                new_self.add_columns(transforms, read_columns).await?;
                Ok(new_self)
            })?
            .map_err(|err: lance::Error| PyIOError::new_err(err.to_string()))?;
        self.ds = Arc::new(new_self);

        Ok(())
    }
}

impl Dataset {
    fn list_versions(&self) -> ::lance::error::Result<Vec<Version>> {
        RT.runtime.block_on(self.ds.versions())
    }
}

#[pyfunction(name = "_write_dataset")]
pub fn write_dataset(reader: &PyAny, uri: String, options: &PyDict) -> PyResult<Dataset> {
    let params = get_write_params(options)?;
    let py = options.py();
    let ds = if reader.is_instance_of::<Scanner>() {
        let scanner: Scanner = reader.extract()?;
        let batches = RT
            .block_on(Some(py), scanner.to_reader())?
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        RT.block_on(Some(py), LanceDataset::write(batches, &uri, params))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?
    } else {
        let batches = ArrowArrayStreamReader::from_pyarrow(reader)?;
        RT.block_on(Some(py), LanceDataset::write(batches, &uri, params))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?
    };
    Ok(Dataset {
        uri,
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
        if let Some(use_experimental_writer) =
            get_dict_opt::<bool>(options, "use_experimental_writer")?
        {
            p.use_experimental_writer = use_experimental_writer;
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

        p.commit_handler = get_commit_handler(options);

        Some(p)
    };
    Ok(params)
}

fn prepare_vector_index_params(
    index_type: &str,
    column_type: &DataType,
    kwargs: Option<&PyDict>,
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
            let sample_rate = PyAny::downcast::<PyInt>(sample_rate)?.extract()?;
            ivf_params.sample_rate = sample_rate;
            pq_params.sample_rate = sample_rate;
            sq_params.sample_rate = sample_rate;
        }

        // Parse IVF params
        if let Some(n) = kwargs.get_item("num_partitions")? {
            ivf_params.num_partitions = PyAny::downcast::<PyInt>(n)?.extract()?
        };

        if let Some(c) = kwargs.get_item("ivf_centroids")? {
            let batch = RecordBatch::from_pyarrow(c)?;
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
            ivf_params.precomputed_partitons_file = Some(f.to_string());
        };

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
                    let list = PyAny::downcast::<PyList>(l)?
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
            hnsw_params.max_level = PyAny::downcast::<PyInt>(max_level)?.extract()?;
        }

        if let Some(m) = kwargs.get_item("m")? {
            hnsw_params.m = PyAny::downcast::<PyInt>(m)?.extract()?;
        }

        if let Some(m_max) = kwargs.get_item("m_max")? {
            hnsw_params.m_max = PyAny::downcast::<PyInt>(m_max)?.extract()?;
        }

        if let Some(ef_c) = kwargs.get_item("ef_construction")? {
            hnsw_params.ef_construction = PyAny::downcast::<PyInt>(ef_c)?.extract()?;
        }

        // Parse PQ params
        if let Some(n) = kwargs.get_item("num_bits")? {
            pq_params.num_bits = PyAny::downcast::<PyInt>(n)?.extract()?
        };

        if let Some(n) = kwargs.get_item("num_sub_vectors")? {
            pq_params.num_sub_vectors = PyAny::downcast::<PyInt>(n)?.extract()?
        };

        if let Some(c) = kwargs.get_item("pq_codebook")? {
            let batch = RecordBatch::from_pyarrow(c)?;
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
        .map_err(|err: PyErr| lance_core::Error::IO {
            message: format!("Failed to call get_batch() on UDFCheckpointer: {}", err),
            location: location!(),
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
        .map_err(|err: PyErr| lance_core::Error::IO {
            message: format!("Failed to call get_fragment() on UDFCheckpointer: {}", err),
            location: location!(),
        })?;
        fragment_data
            .map(|data| {
                serde_json::from_str(&data).map_err(|err| lance::Error::IO {
                    message: format!("Failed to deserialize fragment data: {}", err),
                    location: location!(),
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
        .map_err(|err: PyErr| lance_core::Error::IO {
            message: format!("Failed to call insert_batch() on UDFCheckpointer: {}", err),
            location: location!(),
        })
    }

    fn insert_fragment(&self, fragment: Fragment) -> lance_core::Result<()> {
        let data = serde_json::to_string(&fragment).map_err(|err| lance_core::Error::IO {
            message: format!("Failed to serialize fragment data: {}", err),
            location: location!(),
        })?;
        Python::with_gil(|py| {
            self.inner
                .call_method1(py, "insert_fragment", (fragment.id, data))?;
            Ok(())
        })
        .map_err(|err: PyErr| lance_core::Error::IO {
            message: format!(
                "Failed to call insert_fragment() on UDFCheckpointer: {}",
                err
            ),
            location: location!(),
        })
    }
}
