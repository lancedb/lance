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

use lance::dataset::{
    index::DatasetIndexRemapperOptions,
    optimize::{
        commit_compaction, compact_files, plan_compaction, CompactionMetrics, CompactionOptions,
        CompactionPlan, CompactionTask, RewriteResult,
    },
};
use pyo3::{exceptions::PyNotImplementedError, pyclass::CompareOp, types::PyTuple};

use super::*;

fn parse_compaction_options(options: &PyDict) -> PyResult<CompactionOptions> {
    let mut opts = CompactionOptions::default();

    for (key, value) in options.into_iter() {
        let key: String = key.extract()?;

        match key.as_str() {
            "target_rows_per_fragment" => {
                opts.target_rows_per_fragment = value.extract()?;
            }
            "max_rows_per_group" => {
                opts.max_rows_per_group = value.extract()?;
            }
            "materialize_deletions" => {
                opts.materialize_deletions = value.extract()?;
            }
            "materialize_deletions_threshold" => {
                opts.materialize_deletions_threshold = value.extract()?;
            }
            "num_threads" => {
                opts.num_threads = value
                    .extract::<Option<usize>>()?
                    .unwrap_or_else(num_cpus::get);
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid compaction option: {}",
                    key
                )));
            }
        }
    }

    Ok(opts)
}

fn unwrap_dataset(dataset: PyObject) -> PyResult<Py<Dataset>> {
    Python::with_gil(|py| dataset.getattr(py, "_ds")?.extract::<Py<Dataset>>(py))
}

fn wrap_fragment(py: Python<'_>, fragment: &Fragment) -> PyResult<PyObject> {
    let fragment_metadata = PyModule::import(py, "lance.fragment")?.getattr("FragmentMetadata")?;
    let fragment_json = serde_json::to_string(&fragment).map_err(|x| {
        PyValueError::new_err(format!("failed to serialize fragment metadata: {}", x))
    })?;

    Ok(fragment_metadata
        .call_method1("from_json", (fragment_json,))?
        .to_object(py))
}

#[pyclass(name = "CompactionMetrics", module = "lance.optimize")]
pub struct PyCompactionMetrics {
    /// int : The number of fragments that have been overwritten.
    #[pyo3(get)]
    pub fragments_removed: usize,
    /// int : The number of new fragments that have been added.
    #[pyo3(get)]
    pub fragments_added: usize,
    /// int : The number of files that have been removed, including deletion files.
    #[pyo3(get)]
    pub files_removed: usize,
    /// int : The number of files that have been added, which is always equal to the
    /// number of fragments.
    #[pyo3(get)]
    pub files_added: usize,
}

#[pymethods]
impl PyCompactionMetrics {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "CompactionMetrics(fragments_removed={}, fragments_added={}, files_removed={}, files_added={})",
            self.fragments_removed, self.fragments_added, self.files_removed, self.files_added
        ))
    }
}

impl From<CompactionMetrics> for PyCompactionMetrics {
    fn from(metrics: CompactionMetrics) -> Self {
        Self {
            fragments_removed: metrics.fragments_removed,
            fragments_added: metrics.fragments_added,
            files_removed: metrics.files_removed,
            files_added: metrics.files_added,
        }
    }
}

/// A plan to compact small dataset fragments into larger ones.
///
/// Created by :py:meth:`lance.optimize.Compaction.plan`.
#[pyclass(name = "CompactionPlan", module = "lance.optimize")]
pub struct PyCompactionPlan(CompactionPlan);

#[pymethods]
impl PyCompactionPlan {
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "CompactionPlan(read_version={}, tasks=<{} compaction tasks>)",
            self.0.read_version(),
            self.num_tasks()
        ))
    }

    /// int : The read version of the dataset that this plan was created from.
    #[getter]
    pub fn read_version(&self) -> u64 {
        self.0.read_version()
    }

    /// int : The number of compaction tasks in the plan.
    pub fn num_tasks(&self) -> usize {
        self.0.num_tasks()
    }

    /// List[CompactionTask] : The individual tasks in the plan.
    #[getter]
    pub fn tasks(&self) -> Vec<PyCompactionTask> {
        self.0.compaction_tasks().map(PyCompactionTask).collect()
    }

    /// Get a JSON representation of the plan.
    ///
    /// Returns
    /// -------
    /// str
    ///
    /// Warning
    /// -------
    /// The JSON representation is not guaranteed to be stable across versions.
    pub fn json(&self) -> PyResult<String> {
        serde_json::to_string(&self.0).map_err(|err| {
            PyValueError::new_err(format!(
                "Could not dump CompactionPlan due to error: {}",
                err
            ))
        })
    }

    /// Load a plan from a JSON representation.
    ///
    /// Parameters
    /// ----------
    /// json : str
    ///     The JSON representation of the plan.
    ///
    /// Returns
    /// -------
    /// CompactionPlan
    #[staticmethod]
    pub fn from_json(json: String) -> PyResult<Self> {
        let task = serde_json::from_str(&json).map_err(|err| {
            PyValueError::new_err(format!(
                "Could not load CompactionPlan due to error: {}",
                err
            ))
        })?;
        Ok(Self(task))
    }

    pub fn __reduce__(&self, py: Python<'_>) -> PyResult<(PyObject, PyObject)> {
        let state = self.json()?;
        let state = PyTuple::new(py, vec![state]).extract()?;
        let from_json = PyModule::import(py, "lance.optimize")?
            .getattr("CompactionPlan")?
            .getattr("from_json")?
            .extract()?;
        Ok((from_json, state))
    }

    pub fn __richcmp__(&self, other: PyRef<'_, Self>, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.0 == other.0),
            CompareOp::Ne => Ok(self.0 != other.0),
            _ => Err(PyNotImplementedError::new_err(
                "Only == and != are supported for CompactionTask",
            )),
        }
    }
}

#[pyclass(name = "CompactionTask", module = "lance.optimize")]
#[derive(Clone)]
pub struct PyCompactionTask(CompactionTask);

#[pymethods]
impl PyCompactionTask {
    pub fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let fragment_reprs: String = self
            .fragments(py)?
            .iter()
            .map(|f| f.call_method0(py, "__repr__")?.extract(py))
            .collect::<PyResult<Vec<String>>>()?
            .join(", ");
        Ok(format!(
            "CompactionTask(read_version={}, fragments=[{}])",
            self.0.read_version, fragment_reprs
        ))
    }

    /// int : The read version of the dataset that this task was created from.
    #[getter]
    pub fn read_version(&self) -> u64 {
        self.0.read_version
    }

    /// List[lance.fragment.FragmentMetadata] : The fragments that will be compacted.
    #[getter]
    pub fn fragments(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.0
            .task
            .fragments
            .iter()
            .map(|f| wrap_fragment(py, f))
            .collect()
    }

    /// Execute the compaction task and return the :py:class:`RewriteResult`.
    ///
    /// The rewrite result should be passed onto :py:meth:`lance.optimize.Compaction.commit`.
    pub fn execute(&self, dataset: PyObject) -> PyResult<PyRewriteResult> {
        let dataset = unwrap_dataset(dataset)?;
        let dataset = Python::with_gil(|py| dataset.borrow(py).clone());
        let result = RT
            .block_on(
                None,
                async move { self.0.execute(dataset.ds.as_ref()).await },
            )?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;

        Ok(PyRewriteResult(result))
    }

    /// Get a JSON representation of the task.
    ///
    /// Returns
    /// -------
    /// str
    ///
    /// Warning
    /// -------
    /// The JSON representation is not guaranteed to be stable across versions.
    pub fn json(&self) -> PyResult<String> {
        serde_json::to_string(&self.0).map_err(|err| {
            PyValueError::new_err(format!(
                "Could not dump CompactionTask due to error: {}",
                err
            ))
        })
    }

    /// Load a task from a JSON representation.
    ///
    /// Parameters
    /// ----------
    /// json : str
    ///     The JSON representation of the task.
    ///
    /// Returns
    /// -------
    /// CompactionTask
    #[staticmethod]
    pub fn from_json(json: String) -> PyResult<Self> {
        let task = serde_json::from_str(&json).map_err(|err| {
            PyValueError::new_err(format!(
                "Could not load CompactionTask due to error: {}",
                err
            ))
        })?;
        Ok(Self(task))
    }

    pub fn __reduce__(&self, py: Python<'_>) -> PyResult<(PyObject, PyObject)> {
        let state = self.json()?;
        let state = PyTuple::new(py, vec![state]).extract()?;
        let from_json = PyModule::import(py, "lance.optimize")?
            .getattr("CompactionTask")?
            .getattr("from_json")?
            .extract()?;
        Ok((from_json, state))
    }

    pub fn __richcmp__(&self, other: Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.0 == other.0),
            CompareOp::Ne => Ok(self.0 != other.0),
            _ => Err(PyNotImplementedError::new_err(
                "Only == and != are supported for CompactionTask",
            )),
        }
    }
}

/// The result of a single compaction task.
///
/// Created by :py:meth:`lance.optimize.CompactionTask.execute`.
///
/// This result is pickle-able, so it can be serialized and sent back to the
/// main process to be passed to :py:meth:`lance.optimize.Compaction.commit`.
#[pyclass(name = "RewriteResult", module = "lance.optimize")]
#[derive(Clone)]
pub struct PyRewriteResult(RewriteResult);

#[pymethods]
impl PyRewriteResult {
    pub fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let orig_fragment_reprs: String = self
            .original_fragments(py)?
            .iter()
            .map(|f| f.call_method0(py, "__repr__")?.extract(py))
            .collect::<PyResult<Vec<String>>>()?
            .join(", ");
        let new_fragment_reprs: String = self
            .original_fragments(py)?
            .iter()
            .map(|f| f.call_method0(py, "__repr__")?.extract(py))
            .collect::<PyResult<Vec<String>>>()?
            .join(", ");

        Ok(format!(
            "RewriteResult(read_version={}, new_fragments=[{}], old_fragments=[{}])",
            self.0.read_version, new_fragment_reprs, orig_fragment_reprs,
        ))
    }

    /// int : The version of the dataset the optimize operation is based on.
    #[getter]
    pub fn read_version(&self) -> u64 {
        self.0.read_version
    }

    /// List[lance.fragment.FragmentMetadata] : The metadata for fragments that are being replaced.
    #[getter]
    pub fn original_fragments(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.0
            .original_fragments
            .iter()
            .map(|f| wrap_fragment(py, f))
            .collect()
    }

    /// List[lance.fragment.FragmentMetadata] : The metadata for fragments that are being added.
    #[getter]
    pub fn new_fragments(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.0
            .new_fragments
            .iter()
            .map(|f| wrap_fragment(py, f))
            .collect()
    }

    /// Get a JSON representation of the result.
    ///
    /// Returns
    /// -------
    /// str
    ///
    /// Warning
    /// -------
    /// The JSON representation is not guaranteed to be stable across versions.
    pub fn json(&self) -> PyResult<String> {
        serde_json::to_string(&self.0).map_err(|err| {
            PyValueError::new_err(format!(
                "Could not dump RewriteResult due to error: {}",
                err
            ))
        })
    }

    /// Load a result from a JSON representation.
    #[staticmethod]
    pub fn from_json(json: String) -> PyResult<Self> {
        let result = serde_json::from_str(&json).map_err(|err| {
            PyValueError::new_err(format!(
                "Could not load RewriteResult due to error: {}",
                err
            ))
        })?;
        Ok(Self(result))
    }

    /// CompactionMetrics : The metrics from this compaction task.
    #[getter]
    pub fn metrics(&self) -> PyResult<PyCompactionMetrics> {
        Ok(self.0.metrics.clone().into())
    }

    pub fn __reduce__(&self, py: Python<'_>) -> PyResult<(PyObject, PyObject)> {
        let state = self.json()?;
        let state = PyTuple::new(py, vec![state]).extract()?;
        let from_json = PyModule::import(py, "lance.optimize")?
            .getattr("RewriteResult")?
            .getattr("from_json")?
            .extract()?;
        Ok((from_json, state))
    }

    pub fn __richcmp__(&self, other: Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.0 == other.0),
            CompareOp::Ne => Ok(self.0 != other.0),
            _ => Err(PyNotImplementedError::new_err(
                "Only == and != are supported for RewriteResult",
            )),
        }
    }
}

/// File compaction operation.
///
/// To run with multiple threads in a single process, just use :py:meth:`execute()`.
///
/// To run with multiple processes, first use :py:meth:`plan()` to construct a
/// plan, then execute the tasks in parallel, and finally use :py:meth:`commit()`.
/// The :py:class:`CompactionPlan` contains many :py:class:`CompactionTask` objects,
/// which can be pickled and sent to other processes. The tasks produce
/// :py:class:`RewriteResult` objects, which can be pickled and sent back to the
/// main process to be passed to :py:meth:`commit()`.
#[pyclass(name = "Compaction", module = "lance.optimize")]
pub struct PyCompaction;

#[pymethods]
impl PyCompaction {
    /// Execute a full compaction operation.
    ///
    /// Parameters
    /// ----------
    /// dataset : lance.Dataset
    ///    The dataset to compact. The dataset instance will be updated to the
    ///    new version once complete.
    /// options : CompactionOptions
    ///    The compaction options.
    ///
    /// Returns
    /// -------
    /// CompactionMetrics
    ///     The metrics from the compaction operation.
    #[staticmethod]
    pub fn execute(dataset: PyObject, options: PyObject, merge_indices: Option<&PyAny>,
        index_new_data: Option<&PyAny>,) -> PyResult<PyCompactionMetrics> {
        let dataset_ref = unwrap_dataset(dataset)?;
        let dataset = Python::with_gil(|py| dataset_ref.borrow(py).clone());

        let index_opts = if let Some((merge_indices, index_new_data)) = merge_indices.zip(index_new_data) {
            let mut index_opts: OptimizeOptions = Default::default();
        if let Ok(merge_indices_bool) = merge_indices.extract::<bool>() {
            if merge_indices_bool {
                index_opts.index_handling = IndexHandling::MergeAll;
            } else {
                index_opts.index_handling = IndexHandling::NewDelta;
            }
        } else if let Ok(merge_indices_int) = merge_indices.extract::<usize>() {
            index_opts.index_handling = IndexHandling::MergeLatestN(merge_indices_int as usize);
        } else if let Ok(merge_indices_ids) = merge_indices.extract::<Vec<String>>() {
            let index_ids = merge_indices_ids
                .iter()
                .map(|id_str| {
                    Uuid::parse_str(id_str).map_err(|err| PyValueError::new_err(err.to_string()))
                })
                .collect::<PyResult<Vec<Uuid>>>()?;
            index_opts.index_handling = IndexHandling::MergeIndices(index_ids);
        } else {
            return Err(PyValueError::new_err(
                "merge_indices must be a boolean value, integer, or list of str.",
            ));
        }

        if let Ok(index_new_data_bool) = index_new_data.extract::<bool>() {
            if index_new_data_bool {
                index_opts.new_data_handling = NewDataHandling::IndexAll;
            } else {
                index_opts.new_data_handling = NewDataHandling::Ignore;
            }
        } else if let Ok(index_new_data_ids) = index_new_data.extract::<Vec<u32>>() {
            index_opts.new_data_handling = NewDataHandling::Fragments(index_new_data_ids);
        } else {
            return Err(PyValueError::new_err(
                "index_new_data must be a boolean value.",
            ));
        }
        Some(index_opts)
        } else {
            None
        };
        

        // Make sure we parse the options within a scoped GIL context, so we
        // aren't holding the GIL while blocking the thread on the operation.
        let opts = Python::with_gil(|py| {
            let options = options.downcast::<PyDict>(py)?;
            parse_compaction_options(options)
        })?;
        let mut new_ds = dataset.ds.as_ref().clone();
        let fut = compact_files(&mut new_ds, opts, None, index_opts);
        let metrics = RT.block_on(None, async move {
            fut.await.map_err(|err| PyIOError::new_err(err.to_string()))
        })??;
        Python::with_gil(|py| {
            dataset_ref.borrow_mut(py).ds = Arc::new(new_ds);
        });
        Ok(metrics.into())
    }

    /// Plan a compaction operation.
    ///
    /// This is intended for users who want to run compaction in a distributed
    /// fashion. For running on a single process, use :py:meth:`execute()`
    /// instead.
    ///
    /// Parameters
    /// ----------
    /// dataset : lance.Dataset
    ///   The dataset to compact.
    /// options : CompactionOptions
    ///   The compaction options.
    ///
    /// Returns
    /// -------
    /// CompactionPlan
    #[staticmethod]
    pub fn plan(dataset: PyObject, options: PyObject) -> PyResult<PyCompactionPlan> {
        let dataset = unwrap_dataset(dataset)?;
        let dataset = Python::with_gil(|py| dataset.borrow(py).clone());
        // Make sure we parse the options within a scoped GIL context, so we
        // aren't holding the GIL while blocking the thread on the operation.
        let opts = Python::with_gil(|py| {
            let options = options.downcast::<PyDict>(py)?;
            parse_compaction_options(options)
        })?;
        let plan = RT
            .block_on(None, async move {
                plan_compaction(dataset.ds.as_ref(), &opts).await
            })?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Ok(PyCompactionPlan(plan))
    }

    /// Commit a compaction operation.
    ///
    /// Once tasks from :py:meth:`plan()` have been executed, the results can
    /// be passed to this method to commit the compaction. It is not required
    /// that all of the original tasks are passed. For example, if only a subset
    /// were successful or completed before a deadline, you can pass just those.
    ///
    /// Parameters
    /// ----------
    /// dataset : lance.Dataset
    ///     The dataset to compact. The dataset instance will be updated to the
    ///     new version once committed.
    /// rewrites : List[RewriteResult]
    ///     The results of the compaction tasks to include in the commit.
    /// merge_indices : Union[bool, int, List[str]]
    ///    How to handle merging indices. If a boolean, it will merge all
    ///    indices. If an integer, it will merge the latest N indices. If a
    ///    list of strings, it will merge the indices with those IDs.
    /// index_new_data : Union[bool, List[int]]
    ///    How to handle indexing new data. If a True, it will index all new
    ///    data. If a list of integers, it will index the new data with those
    ///    fragment IDs. If False, it will not index any new data.
    ///
    /// Returns
    /// -------
    /// CompactionMetrics
    #[staticmethod]
    pub fn commit(
        dataset: PyObject,
        rewrites: Vec<PyRewriteResult>,
        merge_indices: &PyAny,
        index_new_data: &PyAny,
    ) -> PyResult<PyCompactionMetrics> {
        let dataset_ref = unwrap_dataset(dataset)?;
        let dataset = Python::with_gil(|py| dataset_ref.borrow(py).clone());
        let rewrites: Vec<RewriteResult> = rewrites.into_iter().map(|r| r.0).collect();

        let mut index_opts: OptimizeOptions = Default::default();
        if let Ok(merge_indices_bool) = merge_indices.extract::<bool>() {
            if merge_indices_bool {
                index_opts.index_handling = IndexHandling::MergeAll;
            } else {
                index_opts.index_handling = IndexHandling::NewDelta;
            }
        } else if let Ok(merge_indices_int) = merge_indices.extract::<usize>() {
            index_opts.index_handling = IndexHandling::MergeLatestN(merge_indices_int as usize);
        } else if let Ok(merge_indices_ids) = merge_indices.extract::<Vec<String>>() {
            let index_ids = merge_indices_ids
                .iter()
                .map(|id_str| {
                    Uuid::parse_str(id_str).map_err(|err| PyValueError::new_err(err.to_string()))
                })
                .collect::<PyResult<Vec<Uuid>>>()?;
            index_opts.index_handling = IndexHandling::MergeIndices(index_ids);
        } else {
            return Err(PyValueError::new_err(
                "merge_indices must be a boolean value, integer, or list of str.",
            ));
        }

        if let Ok(index_new_data_bool) = index_new_data.extract::<bool>() {
            if index_new_data_bool {
                index_opts.new_data_handling = NewDataHandling::IndexAll;
            } else {
                index_opts.new_data_handling = NewDataHandling::Ignore;
            }
        } else if let Ok(index_new_data_ids) = index_new_data.extract::<Vec<u32>>() {
            index_opts.new_data_handling = NewDataHandling::Fragments(index_new_data_ids);
        } else {
            return Err(PyValueError::new_err(
                "index_new_data must be a boolean value.",
            ));
        }

        let mut new_ds = dataset.ds.as_ref().clone();
        let fut = commit_compaction(
            &mut new_ds,
            rewrites,
            Arc::new(DatasetIndexRemapperOptions::default()),
            Some(index_opts),
        );
        let metrics = RT
            .block_on(None, fut)?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Python::with_gil(|py| {
            dataset_ref.borrow_mut(py).ds = Arc::new(new_ds);
        });
        Ok(metrics.into())
    }
}
