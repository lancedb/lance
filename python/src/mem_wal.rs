// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::*;
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, RecordBatchReader,
    StructArray, make_array,
};
use arrow_data::ArrayData;
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::ScalarValue;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use lance::dataset::Dataset as LanceDataset;
use lance::dataset::mem_wal::scanner::{
    FlushedGeneration, LsmDataSourceCollector, LsmPointLookupPlanner, LsmVectorSearchPlanner,
};
use lance::dataset::mem_wal::write::{MemTableStats, WriteStatsSnapshot};
use lance::dataset::mem_wal::{
    LsmScanner, ShardSnapshot as RegionSnapshot, ShardWriter as RegionWriter,
};
use lance_index::mem_wal::MergedGeneration as LanceMergedGeneration;
use lance_linalg::distance::DistanceType;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tokio::sync::Mutex as TokioMutex;
use uuid::Uuid;

use crate::dataset::Dataset as PyDataset;
use crate::rt;

/// Represents a single generation of a MemWAL region that has been merged
/// into the base table. Used with `MergeInsertBuilder.mark_generations_as_merged()`.
#[pyclass(name = "_MergedGeneration", module = "_lib")]
pub struct PyMergedGeneration {
    pub region_id: String,
    pub generation: u64,
}

#[pymethods]
impl PyMergedGeneration {
    #[new]
    pub fn new(region_id: String, generation: u64) -> Self {
        Self {
            region_id,
            generation,
        }
    }

    #[getter]
    pub fn region_id(&self) -> &str {
        &self.region_id
    }

    #[getter]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn __repr__(&self) -> String {
        format!(
            "_MergedGeneration(region_id='{}', generation={})",
            self.region_id, self.generation
        )
    }
}

impl PyMergedGeneration {
    pub fn to_lance(&self) -> PyResult<LanceMergedGeneration> {
        let uuid = Uuid::parse_str(&self.region_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid region_id UUID: {}", e)))?;
        Ok(LanceMergedGeneration::new(uuid, self.generation))
    }
}

/// Snapshot of a MemWAL region's state at a point in time.
///
/// Used to specify which flushed generations to include when creating an
/// `_LsmScanner`. Supports a builder pattern for adding generations.
#[pyclass(name = "_RegionSnapshot", module = "_lib", skip_from_py_object)]
#[derive(Clone)]
pub struct PyRegionSnapshot {
    pub inner: RegionSnapshot,
}

#[pymethods]
impl PyRegionSnapshot {
    #[new]
    pub fn new(region_id: String) -> PyResult<Self> {
        let uuid = Uuid::parse_str(&region_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid region_id UUID: {}", e)))?;
        Ok(Self {
            inner: RegionSnapshot::new(uuid),
        })
    }

    /// Set the RegionSpec ID for this snapshot.
    pub fn with_spec_id(mut slf: PyRefMut<'_, Self>, spec_id: u32) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().with_spec_id(spec_id);
        slf
    }

    /// Set the current (active) generation number.
    pub fn with_current_generation(
        mut slf: PyRefMut<'_, Self>,
        generation: u64,
    ) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().with_current_generation(generation);
        slf
    }

    /// Add a flushed generation by its generation number and storage path.
    pub fn with_flushed_generation(
        mut slf: PyRefMut<'_, Self>,
        generation: u64,
        path: String,
    ) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().with_flushed_generation(generation, path);
        slf
    }

    #[getter]
    pub fn region_id(&self) -> String {
        self.inner.shard_id.to_string()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "_RegionSnapshot(region_id='{}', current_gen={}, flushed_gens={})",
            self.inner.shard_id,
            self.inner.current_generation,
            self.inner.flushed_generations.len()
        )
    }
}

/// Long-lived stateful writer for a MemWAL region.
///
/// Supports writing batches, querying statistics, creating LSM scanners,
/// and graceful shutdown. Supports the Python context manager protocol.
#[pyclass(name = "_RegionWriter", module = "_lib")]
pub struct PyRegionWriter {
    inner: Arc<TokioMutex<Option<RegionWriter>>>,
    closed_state: Arc<TokioMutex<Option<ClosedRegionWriterState>>>,
    region_id: Uuid,
    dataset: Arc<LanceDataset>,
}

#[derive(Clone)]
struct ClosedRegionWriterState {
    stats: WriteStatsSnapshot,
    memtable_stats: MemTableStats,
}

#[pymethods]
impl PyRegionWriter {
    /// Write data batches to the MemWAL.
    ///
    /// Accepts any PyArrow-compatible data source (RecordBatch, Table,
    /// or an Arrow stream reader).
    pub fn put(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let reader = ArrowArrayStreamReader::from_pyarrow_bound(data)
            .map_err(|e| PyValueError::new_err(format!("Cannot read data as Arrow: {}", e)))?;
        let batches: Vec<RecordBatch> = reader
            .collect::<Result<_, _>>()
            .map_err(|e| PyIOError::new_err(format!("Failed to read batches: {}", e)))?;

        if batches.is_empty() {
            return Ok(());
        }

        let inner = self.inner.clone();
        rt().block_on(Some(py), async move {
            let guard = inner.lock().await;
            match guard.as_ref() {
                Some(writer) => writer.put(batches).await.map(|_| ()),
                None => Err(lance_core::Error::invalid_input(
                    "RegionWriter is already closed",
                )),
            }
        })?
        .map_err(|e: lance::Error| PyIOError::new_err(e.to_string()))
    }

    /// Flush pending data and close the writer.
    ///
    /// After close(), calling put() will raise an error.
    /// This is called automatically when using the context manager.
    pub fn close(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        let closed_state = self.closed_state.clone();
        rt().block_on(Some(py), async move {
            let mut guard = inner.lock().await;
            if let Some(writer) = guard.take() {
                let stats_handle = writer.stats_handle();
                // Snapshot stats before close so the captured state reflects
                // what was written, not any internal bookkeeping done by close().
                let stats_snapshot = stats_handle.snapshot();
                let memtable_stats_before_close = writer.memtable_stats().await?;
                writer.close().await?;
                let closed_memtable_stats = closed_memtable_stats(memtable_stats_before_close);
                let mut closed_guard = closed_state.lock().await;
                *closed_guard = Some(ClosedRegionWriterState {
                    stats: stats_snapshot,
                    memtable_stats: closed_memtable_stats,
                });
                Ok(())
            } else {
                Ok(())
            }
        })?
        .map_err(|e: lance::Error| PyIOError::new_err(e.to_string()))
    }

    /// Return a snapshot of current write statistics.
    ///
    /// Returns a dict with keys: put_count, put_time_ms, wal_flush_count,
    /// wal_flush_bytes, memtable_flush_count, memtable_flush_rows.
    pub fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        let closed_state = self.closed_state.clone();
        let stats = rt()
            .block_on(Some(py), async move {
                let guard = inner.lock().await;
                if let Some(writer) = guard.as_ref() {
                    Ok(writer.stats())
                } else {
                    let closed_guard = closed_state.lock().await;
                    closed_guard
                        .as_ref()
                        .map(|state| state.stats.clone())
                        .ok_or_else(|| {
                            lance_core::Error::invalid_input("RegionWriter is already closed")
                        })
                }
            })?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        write_stats_to_pydict(py, &stats)
    }

    /// Return current MemTable statistics.
    ///
    /// Returns a dict with keys: row_count, batch_count, estimated_size_bytes,
    /// generation.
    pub fn memtable_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        let closed_state = self.closed_state.clone();
        let stats = rt()
            .block_on(Some(py), async move {
                let guard = inner.lock().await;
                match guard.as_ref() {
                    Some(w) => w.memtable_stats().await,
                    None => {
                        let closed_guard = closed_state.lock().await;
                        closed_guard
                            .as_ref()
                            .map(|state| state.memtable_stats.clone())
                            .ok_or_else(|| {
                                lance_core::Error::invalid_input("RegionWriter is already closed")
                            })
                    }
                }
            })?
            .map_err(|e: lance::Error| PyIOError::new_err(e.to_string()))?;

        memtable_stats_to_pydict(py, &stats)
    }

    /// Create an LSM scanner that includes the active MemTable for strong consistency.
    ///
    /// The scanner covers: base table + given flushed generations + current active MemTable.
    #[pyo3(signature = (region_snapshots=vec![]))]
    pub fn lsm_scanner(
        &self,
        py: Python<'_>,
        region_snapshots: Vec<Bound<'_, PyRegionSnapshot>>,
    ) -> PyResult<PyLsmScanner> {
        let mut snapshots: Vec<RegionSnapshot> = region_snapshots
            .iter()
            .map(|s| s.borrow().inner.clone())
            .collect();

        let pk_columns = get_pk_columns(&self.dataset)?;
        let inner = self.inner.clone();
        let dataset = self.dataset.clone();
        let region_id = self.region_id;

        let (active_ref, writer_snapshot) = rt()
            .block_on(Some(py), async move {
                let guard = inner.lock().await;
                match guard.as_ref() {
                    Some(w) => {
                        let active_ref = w.active_memtable_ref().await?;
                        let writer_snapshot = w
                            .manifest()
                            .await?
                            .map(region_snapshot_from_manifest)
                            .unwrap_or_else(|| RegionSnapshot::new(region_id));
                        Ok((active_ref, writer_snapshot))
                    }
                    None => Err(lance_core::Error::invalid_input(
                        "RegionWriter is already closed",
                    )),
                }
            })?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        snapshots.retain(|snapshot| snapshot.shard_id != region_id);
        snapshots.push(writer_snapshot);

        let scanner = LsmScanner::new(dataset, snapshots, pk_columns)
            .with_active_memtable(region_id, active_ref);

        Ok(PyLsmScanner {
            inner: Some(scanner),
        })
    }

    /// Return the region ID as a UUID string.
    #[getter]
    pub fn region_id(&self) -> String {
        self.region_id.to_string()
    }

    pub fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_val: &Bound<'_, PyAny>,
        _exc_tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        self.close(py)?;
        Ok(false)
    }
}

impl PyRegionWriter {
    /// Create from a Rust RegionWriter and dataset reference.
    pub fn new(writer: RegionWriter, region_id: Uuid, dataset: Arc<LanceDataset>) -> Self {
        Self {
            inner: Arc::new(TokioMutex::new(Some(writer))),
            closed_state: Arc::new(TokioMutex::new(None)),
            region_id,
            dataset,
        }
    }
}

/// Python wrapper around a DataFusion physical execution plan.
#[pyclass(name = "_ExecutionPlan", module = "_lib", skip_from_py_object)]
#[derive(Clone)]
pub struct PyExecutionPlan {
    plan: Arc<dyn ExecutionPlan>,
    dataset_schema: Arc<ArrowSchema>,
}

impl PyExecutionPlan {
    pub fn new(plan: Arc<dyn ExecutionPlan>, dataset_schema: Arc<ArrowSchema>) -> Self {
        Self {
            plan,
            dataset_schema,
        }
    }
}

#[pymethods]
impl PyExecutionPlan {
    #[getter]
    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.plan.schema().to_pyarrow(py)
    }

    #[getter]
    fn dataset_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.dataset_schema.to_pyarrow(py)
    }

    fn explain(&self) -> String {
        format!("{}", displayable(self.plan.as_ref()).indent(true))
    }

    fn to_batches<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let plan = self.plan.clone();
        let batches = rt()
            .block_on(Some(py), async move {
                let ctx = SessionContext::new();
                collect(plan, ctx.task_ctx()).await
            })?
            .map_err(|e| PyIOError::new_err(format!("Plan execution failed: {}", e)))?;

        let py_batches: Vec<Bound<'py, PyAny>> = batches
            .into_iter()
            .map(|batch| {
                PyArrowType(batch)
                    .into_pyobject(py)
                    .map(|batch| batch.into_any())
                    .map_err(|e| PyIOError::new_err(e.to_string()))
            })
            .collect::<PyResult<_>>()?;
        PyList::new(py, py_batches)
    }

    fn to_reader<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let plan = self.plan.clone();
        let batches = rt()
            .block_on(Some(py), async move {
                let ctx = SessionContext::new();
                collect(plan, ctx.task_ctx()).await
            })?
            .map_err(|e| PyIOError::new_err(format!("Plan execution failed: {}", e)))?;

        let schema = self.plan.schema().clone();
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
            batches.into_iter().map(Ok),
            schema,
        ));
        reader.into_pyarrow(py)
    }
}

/// LSM-aware scanner covering base table, flushed MemTables, and active MemTable.
///
/// Provides deduplication by primary key, always returning the newest version
/// of each row across all LSM levels.
#[pyclass(name = "_LsmScanner", module = "_lib")]
pub struct PyLsmScanner {
    inner: Option<LsmScanner>,
}

#[pymethods]
impl PyLsmScanner {
    /// Create a scanner from dataset and region snapshots (without active MemTable).
    #[staticmethod]
    pub fn from_snapshots(
        dataset: &Bound<'_, PyDataset>,
        region_snapshots: Vec<Bound<'_, PyRegionSnapshot>>,
    ) -> PyResult<Self> {
        let ds = dataset.borrow().ds.clone();
        let snapshots: Vec<RegionSnapshot> = region_snapshots
            .iter()
            .map(|s| s.borrow().inner.clone())
            .collect();
        let pk_columns = get_pk_columns(&ds)?;
        Ok(Self {
            inner: Some(LsmScanner::new(ds, snapshots, pk_columns)),
        })
    }

    /// Select specific columns to return.
    pub fn project(
        mut slf: PyRefMut<'_, Self>,
        columns: Vec<String>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let scanner = slf
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Scanner has already been consumed"))?;
        let cols: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
        slf.inner = Some(scanner.project(&cols));
        Ok(slf)
    }

    /// Set a SQL filter expression.
    pub fn filter(mut slf: PyRefMut<'_, Self>, expr: String) -> PyResult<PyRefMut<'_, Self>> {
        let scanner = slf
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Scanner has already been consumed"))?;
        slf.inner = Some(
            scanner
                .filter(&expr)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
        Ok(slf)
    }

    /// Limit the number of rows returned.
    #[pyo3(signature = (n, offset=None))]
    pub fn limit(
        mut slf: PyRefMut<'_, Self>,
        n: usize,
        offset: Option<usize>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let scanner = slf
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Scanner has already been consumed"))?;
        slf.inner = Some(scanner.limit(n, offset));
        Ok(slf)
    }

    /// Include the `_rowaddr` internal column in results.
    pub fn with_row_address(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        let scanner = slf
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Scanner has already been consumed"))?;
        slf.inner = Some(scanner.with_row_address());
        Ok(slf)
    }

    /// Include the `_memtable_gen` internal column in results.
    pub fn with_memtable_gen(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        let scanner = slf
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Scanner has already been consumed"))?;
        slf.inner = Some(scanner.with_memtable_gen());
        Ok(slf)
    }

    /// Execute the scan and return a single PyArrow RecordBatch.
    pub fn to_batch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let scanner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Scanner has already been consumed"))?;
        let batch = rt()
            .block_on(Some(py), scanner.try_into_batch())
            .map_err(|e| PyIOError::new_err(e.to_string()))?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        PyArrowType(batch).into_pyobject(py).map(|b| b.into_any())
    }

    /// Execute the scan and return all batches as a Python list.
    pub fn to_batches<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let scanner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Scanner has already been consumed"))?;
        let stream = rt()
            .block_on(Some(py), scanner.try_into_stream())
            .map_err(|e| PyIOError::new_err(e.to_string()))?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let batches: Vec<RecordBatch> = rt()
            .block_on(Some(py), stream.try_collect())
            .map_err(|e| PyIOError::new_err(e.to_string()))?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        let py_batches: Vec<Bound<'py, PyAny>> = batches
            .into_iter()
            .map(|b| {
                PyArrowType(b)
                    .into_pyobject(py)
                    .map(|b| b.into_any())
                    .map_err(|e| PyIOError::new_err(e.to_string()))
            })
            .collect::<PyResult<_>>()?;
        PyList::new(py, py_batches)
    }

    /// Return the row count without loading all data.
    pub fn count_rows(&self, py: Python<'_>) -> PyResult<u64> {
        let scanner = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Scanner has already been consumed"))?;
        rt().block_on(Some(py), scanner.count_rows())
            .map_err(|e| PyIOError::new_err(e.to_string()))?
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }
}

/// Plans and executes primary key point lookups across all LSM levels.
///
/// More efficient than `_LsmScanner` for known-PK lookups due to bloom filter
/// optimizations and short-circuit evaluation.
#[pyclass(name = "_LsmPointLookupPlanner", module = "_lib")]
pub struct PyLsmPointLookupPlanner {
    planner: LsmPointLookupPlanner,
    dataset_schema: Arc<ArrowSchema>,
    pk_columns: Vec<String>,
}

#[pymethods]
impl PyLsmPointLookupPlanner {
    #[new]
    #[pyo3(signature = (dataset, region_snapshots, pk_columns=None))]
    pub fn new(
        dataset: &Bound<'_, PyDataset>,
        region_snapshots: Vec<Bound<'_, PyRegionSnapshot>>,
        pk_columns: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let ds = dataset.borrow().ds.clone();
        let snapshots: Vec<RegionSnapshot> = region_snapshots
            .iter()
            .map(|s| s.borrow().inner.clone())
            .collect();
        let pk_cols = match pk_columns {
            Some(cols) => cols,
            None => get_pk_columns(&ds)?,
        };
        let base_schema = Arc::new(ArrowSchema::from(ds.schema()));
        let collector = LsmDataSourceCollector::new(ds.clone(), snapshots);
        let planner = LsmPointLookupPlanner::new(collector, pk_cols.clone(), base_schema.clone());
        Ok(Self {
            planner,
            dataset_schema: base_schema,
            pk_columns: pk_cols,
        })
    }

    /// Plan a single-row point lookup by primary key.
    ///
    /// For single-column primary keys, `pk_value` should be a PyArrow array
    /// with exactly one element. For composite primary keys, `pk_value` must
    /// be a StructArray with exactly one row and one field per PK column.
    #[pyo3(signature = (pk_value, columns=None))]
    pub fn plan_lookup(
        &self,
        py: Python<'_>,
        pk_value: PyArrowType<ArrayData>,
        columns: Option<Vec<String>>,
    ) -> PyResult<PyExecutionPlan> {
        let array = make_array(pk_value.0);
        let pk_values = scalar_values_from_pk_value(array.as_ref(), &self.pk_columns)?;
        let proj: Option<Vec<String>> = columns;
        let planner_ref = &self.planner;
        let plan = rt()
            .block_on(Some(py), async {
                planner_ref.plan_lookup(&pk_values, proj.as_deref()).await
            })?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        Ok(PyExecutionPlan::new(plan, self.dataset_schema.clone()))
    }
}

/// Plans and executes vector KNN search across all LSM levels.
///
/// Only supports IVF-PQ vector indexes maintained in MemWAL.
/// Results include staleness filtering to return only the latest version of each row.
#[pyclass(name = "_LsmVectorSearchPlanner", module = "_lib")]
pub struct PyLsmVectorSearchPlanner {
    planner: LsmVectorSearchPlanner,
    vector_dim: usize,
    dataset_schema: Arc<ArrowSchema>,
}

#[pymethods]
impl PyLsmVectorSearchPlanner {
    #[new]
    #[pyo3(signature = (dataset, region_snapshots, vector_column, pk_columns=None, distance_type=None))]
    pub fn new(
        dataset: &Bound<'_, PyDataset>,
        region_snapshots: Vec<Bound<'_, PyRegionSnapshot>>,
        vector_column: String,
        pk_columns: Option<Vec<String>>,
        distance_type: Option<String>,
    ) -> PyResult<Self> {
        let ds = dataset.borrow().ds.clone();
        let snapshots: Vec<RegionSnapshot> = region_snapshots
            .iter()
            .map(|s| s.borrow().inner.clone())
            .collect();
        let pk_cols = match pk_columns {
            Some(cols) => cols,
            None => get_pk_columns(&ds)?,
        };
        let base_schema = Arc::new(ArrowSchema::from(ds.schema()));

        let dist_type = parse_distance_type(distance_type.as_deref().unwrap_or("l2"))?;

        let vector_dim = get_vector_dim(&ds, &vector_column)?;

        let collector = LsmDataSourceCollector::new(ds, snapshots);
        let planner = LsmVectorSearchPlanner::new(
            collector,
            pk_cols,
            base_schema.clone(),
            vector_column,
            dist_type,
        );

        Ok(Self {
            planner,
            vector_dim,
            dataset_schema: base_schema,
        })
    }

    /// Plan a KNN vector search.
    ///
    /// `query` should be a flat PyArrow Float32Array with `vector_dim` elements.
    #[pyo3(signature = (query, k=10, nprobes=20, columns=None))]
    pub fn plan_search(
        &self,
        py: Python<'_>,
        query: PyArrowType<ArrayData>,
        k: usize,
        nprobes: usize,
        columns: Option<Vec<String>>,
    ) -> PyResult<PyExecutionPlan> {
        let query_array = make_array(query.0);
        let float32_array = query_array
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| {
                PyValueError::new_err(
                    "query must be a Float32Array. Use pa.array(values, type=pa.float32())",
                )
            })?;

        if float32_array.len() != self.vector_dim {
            return Err(PyValueError::new_err(format!(
                "Query vector has {} dimensions, expected {}",
                float32_array.len(),
                self.vector_dim
            )));
        }

        // Wrap the flat array into a FixedSizeListArray with one row
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let fsl = FixedSizeListArray::try_new(
            field,
            self.vector_dim as i32,
            Arc::new(float32_array.clone()),
            None,
        )
        .map_err(|e| PyValueError::new_err(format!("Cannot create query vector: {}", e)))?;

        let planner_ref = &self.planner;
        let plan = rt()
            .block_on(Some(py), async {
                planner_ref
                    .plan_search(&fsl, k, nprobes, columns.as_deref())
                    .await
            })?
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        Ok(PyExecutionPlan::new(plan, self.dataset_schema.clone()))
    }
}

/// Extract primary key column names from dataset schema.
pub(crate) fn get_pk_columns(ds: &LanceDataset) -> PyResult<Vec<String>> {
    let pk_fields = ds.schema().unenforced_primary_key();
    if pk_fields.is_empty() {
        return Err(PyValueError::new_err(
            "Dataset has no primary key. Set 'lance-schema:unenforced-primary-key' metadata \
             on the primary key field(s).",
        ));
    }
    Ok(pk_fields.iter().map(|f| f.name.clone()).collect())
}

/// Parse distance type string to DistanceType enum.
fn parse_distance_type(s: &str) -> PyResult<DistanceType> {
    match s.to_lowercase().as_str() {
        "l2" | "euclidean" => Ok(DistanceType::L2),
        "cosine" => Ok(DistanceType::Cosine),
        "dot" | "inner_product" => Ok(DistanceType::Dot),
        "hamming" => Ok(DistanceType::Hamming),
        _ => Err(PyValueError::new_err(format!(
            "Unknown distance_type '{}'. Valid values: 'l2', 'cosine', 'dot', 'hamming'",
            s
        ))),
    }
}

/// Get the vector dimension from the dataset schema for a given column.
fn get_vector_dim(ds: &LanceDataset, column: &str) -> PyResult<usize> {
    let schema = ArrowSchema::from(ds.schema());
    let field = schema.field_with_name(column).map_err(|_| {
        PyValueError::new_err(format!("Column '{}' not found in dataset schema", column))
    })?;
    match field.data_type() {
        DataType::FixedSizeList(_, size) => Ok(*size as usize),
        other => Err(PyValueError::new_err(format!(
            "Column '{}' is not a FixedSizeList (got {:?}). \
             Vector columns must be FixedSizeList<float32>.",
            column, other
        ))),
    }
}

fn write_stats_to_pydict(py: Python<'_>, stats: &WriteStatsSnapshot) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("put_count", stats.put_count)?;
    dict.set_item("put_time_ms", stats.put_time.as_millis() as u64)?;
    dict.set_item("wal_flush_count", stats.wal_flush_count)?;
    dict.set_item("wal_flush_bytes", stats.wal_flush_bytes)?;
    dict.set_item("wal_flush_time_ms", stats.wal_flush_time.as_millis() as u64)?;
    dict.set_item("memtable_flush_count", stats.memtable_flush_count)?;
    dict.set_item("memtable_flush_rows", stats.memtable_flush_rows)?;
    dict.set_item(
        "memtable_flush_time_ms",
        stats.memtable_flush_time.as_millis() as u64,
    )?;
    Ok(dict.into_any().unbind())
}

fn memtable_stats_to_pydict(py: Python<'_>, stats: &MemTableStats) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("row_count", stats.row_count)?;
    dict.set_item("batch_count", stats.batch_count)?;
    dict.set_item("estimated_size_bytes", stats.estimated_size)?;
    dict.set_item("generation", stats.generation)?;
    Ok(dict.into_any().unbind())
}

fn scalar_values_from_pk_value(
    pk_value: &dyn Array,
    pk_columns: &[String],
) -> PyResult<Vec<ScalarValue>> {
    if pk_value.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "pk_value must contain exactly one row, got {}",
            pk_value.len()
        )));
    }

    if pk_columns.len() == 1 {
        let scalar = ScalarValue::try_from_array(pk_value, 0)
            .map_err(|e| PyValueError::new_err(format!("Cannot convert pk_value: {}", e)))?;
        return Ok(vec![scalar]);
    }

    let struct_array = pk_value.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
        PyValueError::new_err(format!(
            "Composite primary key lookup requires a StructArray with exactly one row and {} fields",
            pk_columns.len()
        ))
    })?;

    if struct_array.num_columns() != pk_columns.len() {
        return Err(PyValueError::new_err(format!(
            "Composite primary key lookup expected {} fields, got {}",
            pk_columns.len(),
            struct_array.num_columns()
        )));
    }

    let mut pk_values = Vec::with_capacity(pk_columns.len());
    for column_name in pk_columns {
        let column = struct_array.column_by_name(column_name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Composite primary key lookup requires field '{}' in pk_value",
                column_name
            ))
        })?;
        let scalar = ScalarValue::try_from_array(column.as_ref(), 0).map_err(|e| {
            PyValueError::new_err(format!("Cannot convert composite pk_value: {}", e))
        })?;
        pk_values.push(scalar);
    }
    Ok(pk_values)
}

fn region_snapshot_from_manifest(manifest: lance_index::mem_wal::ShardManifest) -> RegionSnapshot {
    RegionSnapshot {
        shard_id: manifest.shard_id,
        spec_id: manifest.shard_spec_id,
        current_generation: manifest.current_generation,
        flushed_generations: manifest
            .flushed_generations
            .into_iter()
            .map(|generation| FlushedGeneration {
                generation: generation.generation,
                path: generation.path,
            })
            .collect(),
    }
}

fn closed_memtable_stats(stats_before_close: MemTableStats) -> MemTableStats {
    if stats_before_close.batch_count == 0 {
        return stats_before_close;
    }

    MemTableStats {
        row_count: 0,
        batch_count: 0,
        estimated_size: 0,
        generation: stats_before_close.generation.saturating_add(1),
    }
}
