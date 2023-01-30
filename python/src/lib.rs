// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Lance Columnar Data Format
//!
//! Lance columnar data format is an alternative to Parquet. It provides 100x faster for random access,
//! automatic versioning, optimized for computer vision, bioinformatics, spatial and ML data.
//! [Apache Arrow](https://arrow.apache.org/) and DuckDB compatible.

use pyo3::prelude::*;

pub(crate) mod dataset;
pub(crate) mod errors;
pub(crate) mod reader;
pub(crate) mod scanner;

pub use dataset::write_dataset;
pub use dataset::Dataset;
pub use reader::LanceReader;
pub use scanner::Scanner;

#[pymodule]
fn lance(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Scanner>()?;
    m.add_class::<Dataset>()?;
    m.add_wrapped(wrap_pyfunction!(write_dataset))?;
    Ok(())
}
