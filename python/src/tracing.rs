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

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::RwLock;

use pyo3::pyclass;
use pyo3::pyfunction;
use pyo3::pymethods;
use pyo3::PyResult;
use tracing::field::Visit;
use tracing::span;
use tracing::subscriber;
use tracing::Event;
use tracing_chrome::ChromeLayer;
use tracing_chrome::{ChromeLayerBuilder, TraceStyle};
use tracing_subscriber::filter;
use tracing_subscriber::layer::Context;
use tracing_subscriber::prelude::*;
use tracing_subscriber::Registry;

static SUBSCRIBER: LazyLock<Arc<RwLock<Option<LoggingPassthroughState>>>> =
    LazyLock::new(|| Arc::new(RwLock::new(None)));

struct LoggingPassthroughState {
    inner: Option<ChromeLayer<Registry>>,
    level: log::Level,
}

impl LoggingPassthroughState {
    fn new(level: log::Level) -> Self {
        Self { inner: None, level }
    }

    fn set_inner(&mut self, inner: ChromeLayer<Registry>) {
        self.inner = Some(inner);
    }
}

#[derive(Default)]
struct EventToStr {
    str: String,
}

impl Visit for EventToStr {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.str += &format!("{}={:?} ", field.name(), value);
    }
}

#[derive(Clone)]
pub struct LoggingPassthroughRef(Arc<RwLock<Option<LoggingPassthroughState>>>);

impl LoggingPassthroughRef {
    fn inner_do<F: FnOnce(&ChromeLayer<Registry>)>(&self, f: F) {
        let state_guard = self.0.read().unwrap();
        if let Some(state) = state_guard.as_ref() {
            if let Some(inner) = state.inner.as_ref() {
                f(inner)
            }
        }
    }
}

impl tracing_subscriber::Layer<Registry> for LoggingPassthroughRef {
    fn on_enter(&self, id: &span::Id, ctx: Context<'_, Registry>) {
        self.inner_do(|inner| inner.on_enter(id, ctx));
    }

    fn on_record(&self, id: &span::Id, values: &span::Record<'_>, ctx: Context<'_, Registry>) {
        self.inner_do(|inner| inner.on_record(id, values, ctx));
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, Registry>) {
        let state_guard = self.0.read().unwrap();
        let state = state_guard.as_ref().unwrap();

        let mut fields = EventToStr::default();
        event.record(&mut fields);
        log::log!(target: "lance::events", state.level, "target=\"{}\" {}", event.metadata().target(), fields.str);

        if let Some(inner) = state.inner.as_ref() {
            inner.on_event(event, ctx);
        }
    }

    fn on_exit(&self, id: &span::Id, ctx: Context<'_, Registry>) {
        self.inner_do(|inner| inner.on_exit(id, ctx));
    }

    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, Registry>) {
        self.inner_do(|inner| inner.on_new_span(attrs, id, ctx));
    }

    fn on_close(&self, id: span::Id, ctx: Context<'_, Registry>) {
        self.inner_do(|inner| inner.on_close(id, ctx));
    }
}

#[pyclass]
pub struct TraceGuard {
    guard: Arc<Mutex<Option<tracing_chrome::FlushGuard>>>,
}

#[pymethods]
impl TraceGuard {
    pub fn finish_tracing(&self) {
        // We're exiting anyways, so discard the result
        let _ = self.guard.lock().map(|mut g| g.take());
    }
}

#[pyfunction]
#[pyo3(signature=(path=None))]
pub fn trace_to_chrome(path: Option<&str>) -> PyResult<TraceGuard> {
    let mut builder = ChromeLayerBuilder::new()
        .trace_style(TraceStyle::Async)
        .include_args(true);
    if let Some(path) = path {
        builder = builder.file(path);
    }
    let (chrome_layer, guard) = builder.build();

    SUBSCRIBER
        .write()
        .unwrap()
        .as_mut()
        .unwrap()
        .set_inner(chrome_layer);

    Ok(TraceGuard {
        guard: Arc::new(Mutex::new(Some(guard))),
    })
}

pub fn initialize_tracing(level: log::Level) {
    let level_filter = match level {
        log::Level::Trace => filter::LevelFilter::TRACE,
        log::Level::Debug => filter::LevelFilter::DEBUG,
        log::Level::Info => filter::LevelFilter::INFO,
        log::Level::Warn => filter::LevelFilter::WARN,
        log::Level::Error => filter::LevelFilter::ERROR,
    };
    // Narrow down to just our targets, otherwise we get a lot of spam from
    // our dependencies. The target check is based on a prefix, so `lance` is
    // sufficient to match `lance_*`.
    let filter = filter::Targets::new()
        .with_target("lance", level_filter)
        .with_target("pylance", level_filter);

    SUBSCRIBER
        .write()
        .unwrap()
        .replace(LoggingPassthroughState::new(level));

    let subscriber =
        Registry::default().with(LoggingPassthroughRef(SUBSCRIBER.clone()).with_filter(filter));
    subscriber::set_global_default(subscriber).unwrap();
}
