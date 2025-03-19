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
use std::sync::Mutex;
use std::sync::RwLock;

use pyo3::exceptions::PyValueError;
use pyo3::pyclass;
use pyo3::pyfunction;
use pyo3::pymethods;
use pyo3::PyResult;
use tracing::field::Visit;
use tracing::span;
use tracing::subscriber;
use tracing::Level;
use tracing::Subscriber;
use tracing_chrome::ChromeLayer;
use tracing_chrome::{ChromeLayerBuilder, TraceStyle};
use tracing_subscriber::filter;
use tracing_subscriber::filter::Filtered;
use tracing_subscriber::filter::Targets;
use tracing_subscriber::layer::Layered;
use tracing_subscriber::prelude::*;
use tracing_subscriber::Registry;

pub type TracingSubscriber = Layered<Filtered<ChromeLayer<Registry>, Targets, Registry>, Registry>;

lazy_static::lazy_static! {
    static ref SUBSCRIBER: LoggingSubscriberRef = LoggingPassthrough::init();
}

struct LoggingPassthroughState {
    inner: Option<TracingSubscriber>,
    level: Level,
}

impl Default for LoggingPassthroughState {
    fn default() -> Self {
        Self {
            inner: None,
            // This value doesn't matter, we'll override it in `initialize_tracing`
            level: Level::INFO,
        }
    }
}

#[derive(Default)]
struct LoggingPassthrough {
    state: RwLock<LoggingPassthroughState>,
}

impl LoggingPassthrough {
    fn init() -> LoggingSubscriberRef {
        let subscriber = LoggingSubscriberRef(Arc::new(LoggingPassthrough::default()));
        subscriber::set_global_default(subscriber.clone()).unwrap();
        subscriber
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
pub struct LoggingSubscriberRef(Arc<LoggingPassthrough>);

impl Subscriber for LoggingSubscriberRef {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.is_event() || self.0.state.read().unwrap().inner.is_some()
    }

    fn new_span(&self, span: &span::Attributes<'_>) -> span::Id {
        let state = self.0.state.read().unwrap();
        if let Some(inner) = &state.inner {
            inner.new_span(span)
        } else {
            span::Id::from_u64(0)
        }
    }

    fn record(&self, span: &span::Id, values: &span::Record<'_>) {
        let state = self.0.state.read().unwrap();
        if let Some(inner) = &state.inner {
            inner.record(span, values);
        }
    }

    fn record_follows_from(&self, span: &span::Id, follows: &span::Id) {
        let state = self.0.state.read().unwrap();
        if let Some(inner) = &state.inner {
            inner.record_follows_from(span, follows);
        }
    }

    fn event(&self, event: &tracing::Event<'_>) {
        let state = self.0.state.read().unwrap();

        if event.metadata().level() <= &state.level {
            let log_level = match *event.metadata().level() {
                Level::TRACE => log::Level::Trace,
                Level::DEBUG => log::Level::Debug,
                Level::INFO => log::Level::Info,
                Level::WARN => log::Level::Warn,
                Level::ERROR => log::Level::Error,
            };
            let mut fields = EventToStr::default();
            event.record(&mut fields);
            log::log!(target: "lance::events", log_level, "target=\"{}\" {}", event.metadata().target(), fields.str);
        }

        if let Some(inner) = &state.inner {
            inner.event(event);
        }
    }

    fn enter(&self, span: &span::Id) {
        let state = self.0.state.read().unwrap();
        if let Some(inner) = &state.inner {
            inner.enter(span);
        }
    }

    fn exit(&self, span: &span::Id) {
        let state = self.0.state.read().unwrap();
        if let Some(inner) = &state.inner {
            inner.exit(span);
        }
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
#[pyo3(signature=(path=None, level=None))]
pub fn trace_to_chrome(path: Option<&str>, level: Option<&str>) -> PyResult<TraceGuard> {
    let mut builder = ChromeLayerBuilder::new()
        .trace_style(TraceStyle::Async)
        .include_args(true);
    if let Some(path) = path {
        builder = builder.file(path);
    }
    let (chrome_layer, guard) = builder.build();
    let level_filter = get_filter(level)?;
    // Narrow down to just our targets, otherwise we get a lot of spam from
    // our dependencies. The target check is based on a prefix, so `lance` is
    // sufficient to match `lance_*`.
    let filter = filter::Targets::new()
        .with_target("lance", level_filter)
        .with_target("pylance", level_filter);
    let subscriber = Registry::default().with(chrome_layer.with_filter(filter));

    let mut state = SUBSCRIBER.0.state.write().unwrap();
    state.inner.replace(subscriber);

    Ok(TraceGuard {
        guard: Arc::new(Mutex::new(Some(guard))),
    })
}

pub fn initialize_tracing(level: log::Level) {
    let tracing_level = match level {
        log::Level::Trace => Level::TRACE,
        log::Level::Debug => Level::DEBUG,
        log::Level::Info => Level::INFO,
        log::Level::Warn => Level::WARN,
        log::Level::Error => Level::ERROR,
    };

    let mut state = SUBSCRIBER.0.state.write().unwrap();
    state.level = tracing_level;
}
