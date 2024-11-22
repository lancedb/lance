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

use std::sync::Arc;

use pyo3::{
    exceptions::PyValueError,
    pyclass, pymethods,
    types::{PyModule, PyTuple},
    PyObject, PyRef, PyResult, Python,
};

use lance::dataset::align::{
    AlignFragmentsPlan as LanceAlignFragmentsPlan, AlignFragmentsTask as LanceAlignFragmentsTask,
    AlignFragmentsTaskResult as LanceAlignFragmentsTaskResult,
};

use crate::{error::PythonErrorExt, RT};

use super::Dataset;

#[pyclass(module = "lance.lance.align")]
pub struct AlignFragmentsPlan {
    inner: Arc<LanceAlignFragmentsPlan>,
}

#[pymethods]
impl AlignFragmentsPlan {
    #[staticmethod]
    pub fn create(
        py: Python<'_>,
        source: &Dataset,
        target: &Dataset,
        join_key: &str,
    ) -> PyResult<Self> {
        let inner = RT
            .block_on(
                Some(py),
                LanceAlignFragmentsPlan::new(source.ds.clone(), target.ds.clone(), join_key),
            )?
            .infer_error()?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    #[getter]
    pub fn tasks(&self) -> Vec<AlignFragmentsTask> {
        self.inner
            .tasks()
            .iter()
            .map(|task| AlignFragmentsTask {
                inner: Arc::new(task.clone()),
            })
            .collect()
    }

    pub fn commit(
        slf: PyRef<'_, Self>,
        results: Vec<AlignFragmentsTaskResult>,
        source: &Dataset,
        target: &Dataset,
    ) -> PyResult<()> {
        let results = results
            .into_iter()
            .map(|result| result.inner.as_ref().clone())
            .collect::<Vec<_>>();
        RT.block_on(
            Some(slf.py()),
            slf.inner
                .commit(results, source.ds.clone(), target.ds.clone()),
        )?
        .infer_error()?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("AlignFragmentsPlan({} tasks)", self.tasks().len())
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
        serde_json::to_string(self.inner.as_ref()).map_err(|err| {
            PyValueError::new_err(format!(
                "Could not dump AlignFragmentsPlan due to error: {}",
                err
            ))
        })
    }

    /// Load a plan from a JSON representation.
    #[staticmethod]
    pub fn from_json(json: String) -> PyResult<Self> {
        let result = serde_json::from_str(&json).map_err(|err| {
            PyValueError::new_err(format!(
                "Could not load AlignFragmentsPlan due to error: {}",
                err
            ))
        })?;
        Ok(Self {
            inner: Arc::new(result),
        })
    }

    pub fn __reduce__(&self, py: Python<'_>) -> PyResult<(PyObject, PyObject)> {
        let state = self.json()?;
        let state = PyTuple::new(py, vec![state]).extract()?;
        let from_json = PyModule::import(py, "lance.lance.align")?
            .getattr("AlignFragmentsPlan")?
            .getattr("from_json")?
            .extract()?;
        Ok((from_json, state))
    }
}

#[pyclass(module = "lance.lance.align")]
pub struct AlignFragmentsTask {
    inner: Arc<LanceAlignFragmentsTask>,
}

#[pymethods]
impl AlignFragmentsTask {
    pub fn execute(
        slf: PyRef<'_, Self>,
        source: &Dataset,
        target: &Dataset,
    ) -> PyResult<AlignFragmentsTaskResult> {
        let inner = RT
            .block_on(
                Some(slf.py()),
                slf.inner.execute(source.ds.clone(), target.ds.clone()),
            )?
            .infer_error()?;
        Ok(AlignFragmentsTaskResult {
            inner: Arc::new(inner),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "AlignFragmentsTask(target_frag_id={})",
            self.inner.target_id
        )
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
        serde_json::to_string(self.inner.as_ref()).map_err(|err| {
            PyValueError::new_err(format!(
                "Could not dump AlignFragmentsTask due to error: {}",
                err
            ))
        })
    }

    /// Load a task from a JSON representation.
    #[staticmethod]
    pub fn from_json(json: String) -> PyResult<Self> {
        let result = serde_json::from_str(&json).map_err(|err| {
            PyValueError::new_err(format!(
                "Could not load AlignFragmentsTask due to error: {}",
                err
            ))
        })?;
        Ok(Self {
            inner: Arc::new(result),
        })
    }

    pub fn __reduce__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<(PyObject, PyObject)> {
        let state = slf.json()?;
        let state = PyTuple::new(py, vec![state]).extract()?;
        let from_json = PyModule::import(py, "lance.lance.align")?
            .getattr("AlignFragmentsTask")?
            .getattr("from_json")?
            .extract()?;
        Ok((from_json, state))
    }
}

#[pyclass(module = "lance.lance.align")]
#[derive(Clone)]
pub struct AlignFragmentsTaskResult {
    inner: Arc<LanceAlignFragmentsTaskResult>,
}

#[pymethods]
impl AlignFragmentsTaskResult {
    fn __repr__(&self) -> String {
        format!(
            "AlignFragmentsTaskResult(path={}, target_frag_id={})",
            self.inner.data_file.path, self.inner.target_id
        )
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
        serde_json::to_string(self.inner.as_ref()).map_err(|err| {
            PyValueError::new_err(format!(
                "Could not dump AlignFragmentsTaskResult due to error: {}",
                err
            ))
        })
    }

    /// Load a task from a JSON representation.
    #[staticmethod]
    pub fn from_json(json: String) -> PyResult<Self> {
        let result = serde_json::from_str(&json).map_err(|err| {
            PyValueError::new_err(format!(
                "Could not load AlignFragmentsTaskResult due to error: {}",
                err
            ))
        })?;
        Ok(Self {
            inner: Arc::new(result),
        })
    }

    pub fn __reduce__(&self, py: Python<'_>) -> PyResult<(PyObject, PyObject)> {
        let state = self.json()?;
        let state = PyTuple::new(py, vec![state]).extract()?;
        let from_json = PyModule::import(py, "lance.lance.align")?
            .getattr("AlignFragmentsTaskResult")?
            .getattr("from_json")?
            .extract()?;
        Ok((from_json, state))
    }
}

pub fn register_align(py: Python, m: &PyModule) -> PyResult<()> {
    let align = PyModule::new(py, "align")?;
    align.add_class::<AlignFragmentsPlan>()?;
    align.add_class::<AlignFragmentsTask>()?;
    align.add_class::<AlignFragmentsTaskResult>()?;
    m.add_submodule(align)?;
    // See https://github.com/PyO3/pyo3/issues/759#issuecomment-977835119
    py.import("sys")?
        .getattr("modules")?
        .set_item("lance.lance.align", align)?;
    Ok(())
}
