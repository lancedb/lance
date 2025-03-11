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

use lance::dataset::statistics::{DataStatistics, FieldStatistics};
use pyo3::{intern, types::PyAnyMethods, Bound, IntoPyObject, PyAny, PyErr, Python};

use crate::utils::{export_vec, PyLance};

impl<'py> IntoPyObject<'py> for PyLance<&FieldStatistics> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let cls = py
            .import(intern!(py, "lance"))
            .and_then(|m| m.getattr("FieldStatistics"))
            .expect("FieldStatistics class not found");

        let id = self.0.id;
        let bytes_on_disk = self.0.bytes_on_disk;

        // unwrap due to infallible
        Ok(cls.call1((id, bytes_on_disk)).unwrap())
    }
}

impl<'py> IntoPyObject<'py> for PyLance<DataStatistics> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let cls = py
            .import(intern!(py, "lance"))
            .and_then(|m| m.getattr("DataStatistics"))
            .expect("DataStatistics class not found");

        let fields = export_vec(py, &self.0.fields)?;

        // unwrap due to infallible
        Ok(cls.call1((fields,)).unwrap())
    }
}
