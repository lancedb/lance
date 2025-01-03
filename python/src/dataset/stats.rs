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
use pyo3::{intern, types::PyAnyMethods, PyObject, Python, ToPyObject};

use crate::utils::{export_vec, PyLance};

impl ToPyObject for PyLance<&FieldStatistics> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let cls = py
            .import_bound(intern!(py, "lance"))
            .and_then(|m| m.getattr("FieldStatistics"))
            .expect("FieldStatistics class not found");

        let id = self.0.id;
        let bytes_on_disk = self.0.bytes_on_disk;

        cls.call1((id, bytes_on_disk)).unwrap().to_object(py)
    }
}

impl ToPyObject for PyLance<DataStatistics> {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let cls = py
            .import_bound(intern!(py, "lance"))
            .and_then(|m| m.getattr("DataStatistics"))
            .expect("DataStatistics class not found");

        let fields = export_vec(py, &self.0.fields);

        cls.call1((fields,)).unwrap().to_object(py)
    }
}
