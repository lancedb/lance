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

use pyo3::exceptions::PyAssertionError;
use pyo3::exceptions::PyValueError;
use pyo3::pyclass;
use pyo3::pyfunction;
use pyo3::pymethods;
use pyo3::PyResult;
use tracing::subscriber;
use tracing_chrome::{ChromeLayerBuilder, TraceStyle};
use tracing_subscriber::filter;
use tracing_subscriber::prelude::*;
use tracing_subscriber::Registry;

#[pyclass]
pub struct TraceGuard {
    guard: Option<tracing_chrome::FlushGuard>,
}

#[pymethods]
impl TraceGuard {
    pub fn finish_tracing(&mut self) {
        self.guard.take();
    }
}

fn get_filter(level: Option<&str>) -> PyResult<filter::LevelFilter> {
    match level {
        Some("trace") => Ok(filter::LevelFilter::TRACE),
        Some("debug") => Ok(filter::LevelFilter::DEBUG),
        Some("info") => Ok(filter::LevelFilter::INFO),
        Some("warn") => Ok(filter::LevelFilter::WARN),
        Some("error") => Ok(filter::LevelFilter::ERROR),
        None => Ok(filter::LevelFilter::INFO),
        _ => Err(PyValueError::new_err(format!(
            "Unexpected tracing level: {}",
            level.unwrap()
        ))),
    }
}

#[pyfunction]
pub fn trace_to_chrome(path: Option<&str>, level: Option<&str>) -> PyResult<TraceGuard> {
    let mut builder = ChromeLayerBuilder::new().trace_style(TraceStyle::Async);
    if let Some(path) = path {
        builder = builder.file(path);
    }
    let (chrome_layer, guard) = builder.build();
    let level_filter = get_filter(level)?;
    // Narrow down to just our targets, otherwise we get a lot of spam from
    // our dependencies.
    let filter = filter::Targets::new()
        .with_target("lance", level_filter)
        .with_target("lance_arrow", level_filter)
        .with_target("lance_core", level_filter)
        .with_target("lance_datagen", level_filter)
        .with_target("lance_index", level_filter)
        .with_target("lance_linalg", level_filter)
        .with_target("pylance", level_filter);
    let subscriber = Registry::default().with(chrome_layer.with_filter(filter));
    subscriber::set_global_default(subscriber)
        .map(move |_| TraceGuard { guard: Some(guard) })
        .map_err(|_| {
            PyAssertionError::new_err("Attempt to configure tracing when it is already configured")
        })
}
