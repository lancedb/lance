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

use lance::dataset::cleanup::{CleanupPlan, CleanupSweepResult, CleanupSweepTask, SweepOptions};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{pyclass, pymethods};

use super::optimize::unwrap_dataset;
use super::{cleanup_explanation, cleanup_stats};
use crate::rt;

#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct CleanupStats {
    pub bytes_removed: u64,
    pub old_versions: u64,
    pub data_files_removed: u64,
    pub transaction_files_removed: u64,
    pub index_files_removed: u64,
    pub deletion_files_removed: u64,
    /// Files that could not be deleted; not counted as removed.
    pub failed_deletes: u64,
}

#[pymethods]
impl CleanupStats {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct CleanupCandidateFile {
    pub path: String,
    pub kind: String,
    pub unverified: bool,
    pub size_bytes: u64,
}

#[pymethods]
impl CleanupCandidateFile {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct CleanupReferencedBranch {
    pub name: String,
    pub referenced_version: u64,
    pub cleanup_candidate: bool,
}

#[pymethods]
impl CleanupReferencedBranch {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct CleanupExplanation {
    pub read_version: u64,
    pub stats: CleanupStats,
    pub candidate_files: Vec<CleanupCandidateFile>,
    pub candidate_files_truncated: bool,
    pub candidate_file_limit: usize,
    pub referenced_branches: Vec<CleanupReferencedBranch>,
    pub warnings: Vec<String>,
}

#[pymethods]
impl CleanupExplanation {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

fn to_json<T: serde::Serialize>(value: &T, name: &str) -> PyResult<String> {
    serde_json::to_string(value)
        .map_err(|err| PyValueError::new_err(format!("Could not dump {name} due to error: {err}")))
}

fn from_json<T: serde::de::DeserializeOwned>(json: &str, name: &str) -> PyResult<T> {
    serde_json::from_str(json)
        .map_err(|err| PyValueError::new_err(format!("Could not load {name} due to error: {err}")))
}

fn reduce(py: Python<'_>, class: &str, json: String) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let state = PyTuple::new(py, vec![json])?.extract()?;
    let from_json = PyModule::import(py, "lance.cleanup")?
        .getattr(class)?
        .getattr("from_json")?
        .extract()?;
    Ok((from_json, state))
}

/// A cleanup whose listing and deletion can be split across processes.
///
/// Created by :py:meth:`lance.LanceDataset.plan_cleanup_old_versions`. Run every
/// task from :py:meth:`tasks`, then pass all of their results to :py:meth:`finish`.
/// The plan stays on the process that created it; only its tasks are sent to
/// workers.
#[pyclass(name = "CleanupPlan", module = "lance.cleanup")]
pub struct PyCleanupPlan(pub(crate) CleanupPlan);

#[pymethods]
impl PyCleanupPlan {
    fn __repr__(&self) -> String {
        format!(
            "CleanupPlan(read_version={}, num_old_versions={})",
            self.0.read_version(),
            self.0.num_old_versions()
        )
    }

    /// int : The dataset version the plan was created from.
    #[getter]
    fn read_version(&self) -> u64 {
        self.0.read_version()
    }

    /// int : The number of manifests :py:meth:`finish` removes.
    #[getter]
    fn num_old_versions(&self) -> usize {
        self.0.num_old_versions()
    }

    /// Split the sweep into tasks.
    ///
    /// Every listed directory gets at least one task, so there can be more than
    /// ``max_tasks`` of them.
    ///
    /// Returns
    /// -------
    /// List[CleanupSweepTask]
    fn tasks(&self, max_tasks: usize) -> PyResult<Vec<PyCleanupSweepTask>> {
        let tasks = self
            .0
            .tasks(max_tasks)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(tasks.into_iter().map(PyCleanupSweepTask).collect())
    }
    /// Remove the expired manifests once every task has run.
    ///
    /// ``results`` must hold exactly one result for each task of one
    /// :py:meth:`tasks` call; otherwise nothing is removed.
    ///
    /// Returns
    /// -------
    /// CleanupStats
    ///     The statistics of the whole cleanup.
    fn finish(
        &self,
        py: Python<'_>,
        dataset: Bound<PyAny>,
        results: Vec<PyCleanupSweepResult>,
    ) -> PyResult<CleanupStats> {
        let dataset = unwrap_dataset(dataset)?.borrow().ds.clone();
        let results = results
            .into_iter()
            .map(|result| result.0)
            .collect::<Vec<_>>();
        let stats = rt()
            .block_on(Some(py), self.0.finish(dataset.as_ref(), &results))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Ok(cleanup_stats(stats))
    }

    /// Like :py:meth:`finish` for a plan created with ``explain=True``: removes
    /// nothing and explains what the cleanup would remove.
    ///
    /// Returns
    /// -------
    /// CleanupExplanation
    fn finish_explain(
        &self,
        py: Python<'_>,
        dataset: Bound<PyAny>,
        results: Vec<PyCleanupSweepResult>,
    ) -> PyResult<CleanupExplanation> {
        let dataset = unwrap_dataset(dataset)?.borrow().ds.clone();
        let results = results
            .into_iter()
            .map(|result| result.0)
            .collect::<Vec<_>>();
        let explanation = rt()
            .block_on(Some(py), self.0.finish_explain(dataset.as_ref(), &results))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Ok(cleanup_explanation(explanation))
    }
}
/// One key range of one directory to sweep, created by :py:meth:`CleanupPlan.tasks`.
///
/// Pickle-able, so it can be sent to a worker. A task only runs on the Lance
/// version that planned it.
#[pyclass(name = "CleanupSweepTask", module = "lance.cleanup", from_py_object)]
#[derive(Clone)]
pub struct PyCleanupSweepTask(CleanupSweepTask);

#[pymethods]
impl PyCleanupSweepTask {
    fn __repr__(&self) -> String {
        format!(
            "CleanupSweepTask(index={}, count={}, subtree={}, start_after={:?}, \
             end_inclusive={:?})",
            self.0.index(),
            self.0.count(),
            self.0.subtree(),
            self.0.start_after(),
            self.0.end_inclusive()
        )
    }

    /// int : Position of this task among the tasks of its plan.
    #[getter]
    fn index(&self) -> u32 {
        self.0.index()
    }

    /// str : The dataset-relative directory this task lists.
    #[getter]
    fn subtree(&self) -> String {
        self.0.subtree().to_string()
    }

    /// Optional[str] : Dataset-relative path after which this task's range starts.
    #[getter]
    fn start_after(&self) -> Option<String> {
        self.0.start_after().map(str::to_string)
    }

    /// Optional[str] : Dataset-relative path at which this task's range ends.
    #[getter]
    fn end_inclusive(&self) -> Option<String> {
        self.0.end_inclusive().map(str::to_string)
    }

    /// Remove the unreferenced files in this task's range.
    ///
    /// ``dataset`` must be the planned dataset; any version of it will do. The
    /// limits apply to this task only.
    ///
    /// Returns
    /// -------
    /// CleanupSweepResult
    #[pyo3(signature = (dataset, *, delete_concurrency = None, delete_rate_limit = None))]
    fn execute(
        &self,
        py: Python<'_>,
        dataset: Bound<PyAny>,
        delete_concurrency: Option<usize>,
        delete_rate_limit: Option<u64>,
    ) -> PyResult<PyCleanupSweepResult> {
        let dataset = unwrap_dataset(dataset)?.borrow().ds.clone();
        let mut options = SweepOptions::default();
        if let Some(value) = delete_concurrency {
            options = options
                .with_delete_concurrency(value)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        if let Some(value) = delete_rate_limit {
            options = options
                .with_delete_rate_limit(value)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        let result = rt()
            .block_on(Some(py), self.0.execute(dataset.as_ref(), &options))?
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
        Ok(PyCleanupSweepResult(result))
    }

    /// Get a JSON representation of the task.
    ///
    /// Warning
    /// -------
    /// The JSON representation is not guaranteed to be stable across versions.
    fn json(&self) -> PyResult<String> {
        to_json(&self.0, "CleanupSweepTask")
    }

    /// Load a task from a JSON representation.
    #[staticmethod]
    fn from_json(json: String) -> PyResult<Self> {
        from_json(&json, "CleanupSweepTask").map(Self)
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        reduce(py, "CleanupSweepTask", self.json()?)
    }
}
/// What one :py:class:`CleanupSweepTask` removed, to pass to
/// :py:meth:`CleanupPlan.finish`. Pickle-able.
#[pyclass(name = "CleanupSweepResult", module = "lance.cleanup", from_py_object)]
#[derive(Clone)]
pub struct PyCleanupSweepResult(CleanupSweepResult);

#[pymethods]
impl PyCleanupSweepResult {
    fn __repr__(&self) -> String {
        format!(
            "CleanupSweepResult(index={}, stats={:?})",
            self.0.index(),
            self.0.stats()
        )
    }

    /// int : Index of the task that produced this result.
    #[getter]
    fn index(&self) -> u32 {
        self.0.index()
    }

    /// CleanupStats : The files the task removed.
    #[getter]
    fn stats(&self) -> CleanupStats {
        cleanup_stats(self.0.stats().clone())
    }

    /// int : Manifests committed after the plan that the task read first.
    #[getter]
    fn refreshed_versions(&self) -> u64 {
        self.0.refreshed_versions()
    }

    /// List[str] : Non-fatal warnings, such as a directory the task did not sweep.
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.0.warnings().to_vec()
    }

    /// Get a JSON representation of the result.
    ///
    /// Warning
    /// -------
    /// The JSON representation is not guaranteed to be stable across versions.
    fn json(&self) -> PyResult<String> {
        to_json(&self.0, "CleanupSweepResult")
    }

    /// Load a result from a JSON representation.
    #[staticmethod]
    fn from_json(json: String) -> PyResult<Self> {
        from_json(&json, "CleanupSweepResult").map(Self)
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        reduce(py, "CleanupSweepResult", self.json()?)
    }
}
