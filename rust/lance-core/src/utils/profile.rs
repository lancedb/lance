// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-query performance aggregator built on top of `tracing`.
//!
//! [`QueryProfileLayer`] is a [`tracing_subscriber::Layer`] that watches for
//! the `query` root span installed by the scanner stream entry points,
//! accumulates wall-clock time spent inside `phase.*` child spans, and
//! aggregates IO events from two sources:
//!
//! * [`TRACE_IO_EVENTS`] — high-level "index/partition open" events.
//! * [`TRACE_IO_PHYSICAL`] — per-read events emitted by the scheduler with
//!   `bytes` and `duration_us` fields. Used for distribution stats
//!   (count, sum, min, p50, p95, max for bytes / duration / throughput),
//!   broken down by file type (`manifest`, `index`, `data`, `deletion`,
//!   `other`).
//!
//! When the scanner stream is dropped the layer flushes one
//! `tracing::info!(target = TRACE_QUERY_PROFILE, ...)` event with all phase
//! and IO statistics.
//!
//! The layer is opt-in: [`QueryProfileLayer::from_env`] returns a no-op
//! instance unless `LANCE_QUERY_PROFILE` is truthy (`1`, `true`, `yes`,
//! `on`).
//!
//! Example:
//! ```
//! use lance_core::utils::profile::QueryProfileLayer;
//! use tracing_subscriber::prelude::*;
//!
//! let _ = tracing_subscriber::registry()
//!     .with(QueryProfileLayer::from_env())
//!     .try_init();
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::fmt::Write as _;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

use super::tracing::{
    PHASE_INDEX_SEARCH_ANN, PHASE_INDEX_SEARCH_FTS, PHASE_INDEX_SEARCH_SCALAR,
    PHASE_LOAD_DATA_FILTERED_READ, PHASE_LOAD_DATA_LANCE_SCAN, PHASE_LOAD_DATA_TAKE,
    PHASE_OPEN_INDEX_FRAG_REUSE, PHASE_OPEN_INDEX_MEM_WAL, PHASE_OPEN_INDEX_SCALAR,
    PHASE_OPEN_INDEX_VECTOR, PHASE_PLAN, PHASE_POSTPROCESS, QUERY_PROFILE_ROOT, TRACE_IO_EVENTS,
    TRACE_IO_PHYSICAL, TRACE_QUERY_PROFILE,
};

/// Internal target used by the scanner stream's `Drop` impl to signal that a
/// query has finished from the user's perspective. The profile layer flushes
/// the per-query summary when it sees this event, instead of waiting for the
/// `query` span to close — which can be delayed indefinitely by background
/// tasks that captured the span via `Span::current()`.
pub const TRACE_QUERY_DONE: &str = "lance::query_done";

thread_local! {
    /// Per-thread queue of query summaries waiting to be emitted. Tracing
    /// silently drops events emitted re-entrantly from inside another layer
    /// callback, so the layer pushes here during `on_event` and the
    /// scanner-side `QueryDoneGuard` calls [`drain_pending_summaries`] after
    /// the originating event has been fully dispatched.
    static PENDING_SUMMARIES: RefCell<Vec<QueryProfileSummary>> =
        const { RefCell::new(Vec::new()) };
}

/// Snapshot of one query's accumulated statistics. Field set is fixed; the
/// per-file-type IO distribution is rendered as a JSON object string for
/// downstream parsing.
#[derive(Debug, Clone)]
pub struct QueryProfileSummary {
    pub total_us: u64,
    pub phase_plan_us: u64,
    pub phase_open_index_us: u64,
    pub phase_open_index_scalar_us: u64,
    pub phase_open_index_vector_us: u64,
    pub phase_open_index_frag_reuse_us: u64,
    pub phase_open_index_mem_wal_us: u64,
    pub phase_index_search_us: u64,
    pub phase_index_search_scalar_us: u64,
    pub phase_index_search_fts_us: u64,
    pub phase_index_search_ann_us: u64,
    pub phase_load_data_us: u64,
    pub phase_load_data_filtered_read_us: u64,
    pub phase_load_data_lance_scan_us: u64,
    pub phase_load_data_take_us: u64,
    pub phase_postprocess_us: u64,
    /// Legacy logical-IO breakdown (`open_scalar_index/...`,
    /// `load_vector_part/...` etc). `type[/index_type]=count`, comma-joined.
    pub logical_io_breakdown: String,
    /// JSON object keyed by file type (`manifest`, `index`, `data`,
    /// `deletion`, `other`). See module docs for the shape.
    pub io_stats: String,
}

/// Drain any summaries collected on the current thread by
/// [`QueryProfileLayer`] and emit one `tracing::info!(target =
/// TRACE_QUERY_PROFILE, ...)` event per summary.
pub fn drain_pending_summaries() {
    let summaries: Vec<QueryProfileSummary> =
        PENDING_SUMMARIES.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
    for s in summaries {
        tracing::info!(
            target: TRACE_QUERY_PROFILE,
            total_us = s.total_us,
            phase_plan_us = s.phase_plan_us,
            phase_open_index_us = s.phase_open_index_us,
            phase_open_index_scalar_us = s.phase_open_index_scalar_us,
            phase_open_index_vector_us = s.phase_open_index_vector_us,
            phase_open_index_frag_reuse_us = s.phase_open_index_frag_reuse_us,
            phase_open_index_mem_wal_us = s.phase_open_index_mem_wal_us,
            phase_index_search_us = s.phase_index_search_us,
            phase_index_search_scalar_us = s.phase_index_search_scalar_us,
            phase_index_search_fts_us = s.phase_index_search_fts_us,
            phase_index_search_ann_us = s.phase_index_search_ann_us,
            phase_load_data_us = s.phase_load_data_us,
            phase_load_data_filtered_read_us = s.phase_load_data_filtered_read_us,
            phase_load_data_lance_scan_us = s.phase_load_data_lance_scan_us,
            phase_load_data_take_us = s.phase_load_data_take_us,
            phase_postprocess_us = s.phase_postprocess_us,
            logical_io_breakdown = %s.logical_io_breakdown,
            io_stats = %s.io_stats,
        );
    }
}

/// Aggregating tracing layer that emits one summary event per query.
#[derive(Debug)]
pub struct QueryProfileLayer {
    enabled: bool,
    state: Mutex<HashMap<Id, QueryState>>,
    phase_enter: Mutex<HashMap<Id, PhasePending>>,
}

#[derive(Debug)]
struct PhasePending {
    root: Id,
    phase: &'static str,
    enter: Option<Instant>,
}

#[derive(Debug)]
struct QueryState {
    started: Instant,
    phases: HashMap<&'static str, Duration>,
    /// Logical IO counters: (io_type, index_type) -> count. Tracks the
    /// existing `lance::io_events` so callers still see open/load_part
    /// statistics broken down by index type.
    logical_io: HashMap<(String, String), u64>,
    /// Physical IO samples keyed by file type. Each sample is
    /// `(bytes, duration_us)`.
    physical_io: HashMap<&'static str, Vec<(u64, u64)>>,
}

impl QueryProfileLayer {
    /// Construct a layer that always aggregates. Use in tests; production
    /// code should prefer [`Self::from_env`].
    pub fn enabled() -> Self {
        Self::with_enabled(true)
    }

    /// Construct a no-op layer.
    pub fn disabled() -> Self {
        Self::with_enabled(false)
    }

    /// Construct a layer whose enabled state is read from the
    /// `LANCE_QUERY_PROFILE` environment variable (`1`, `true`, `yes`, `on`
    /// — case-insensitive — turn it on).
    pub fn from_env() -> Self {
        Self::with_enabled(env_truthy("LANCE_QUERY_PROFILE"))
    }

    fn with_enabled(enabled: bool) -> Self {
        Self {
            enabled,
            state: Mutex::new(HashMap::new()),
            phase_enter: Mutex::new(HashMap::new()),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn flush_query(&self, root: &Id) {
        let Some(state) = self.state.lock().unwrap().remove(root) else {
            return;
        };
        emit_summary(state);
    }
}

fn env_truthy(name: &str) -> bool {
    match env::var(name) {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

/// Recognize a phase span name and return its canonical static reference.
/// Names not in this set are ignored. Rollup names (`phase.open_index`
/// without subkind) are intentionally NOT recognized at the span layer —
/// callers should always emit the most specific name; rollups are computed
/// at emission time.
fn phase_for(name: &str) -> Option<&'static str> {
    match name {
        PHASE_PLAN => Some(PHASE_PLAN),
        PHASE_OPEN_INDEX_SCALAR => Some(PHASE_OPEN_INDEX_SCALAR),
        PHASE_OPEN_INDEX_VECTOR => Some(PHASE_OPEN_INDEX_VECTOR),
        PHASE_OPEN_INDEX_FRAG_REUSE => Some(PHASE_OPEN_INDEX_FRAG_REUSE),
        PHASE_OPEN_INDEX_MEM_WAL => Some(PHASE_OPEN_INDEX_MEM_WAL),
        PHASE_INDEX_SEARCH_SCALAR => Some(PHASE_INDEX_SEARCH_SCALAR),
        PHASE_INDEX_SEARCH_FTS => Some(PHASE_INDEX_SEARCH_FTS),
        PHASE_INDEX_SEARCH_ANN => Some(PHASE_INDEX_SEARCH_ANN),
        PHASE_LOAD_DATA_FILTERED_READ => Some(PHASE_LOAD_DATA_FILTERED_READ),
        PHASE_LOAD_DATA_LANCE_SCAN => Some(PHASE_LOAD_DATA_LANCE_SCAN),
        PHASE_LOAD_DATA_TAKE => Some(PHASE_LOAD_DATA_TAKE),
        PHASE_POSTPROCESS => Some(PHASE_POSTPROCESS),
        _ => None,
    }
}

/// Map a filesystem path to one of the five file-type buckets used in the
/// IO distribution stats. Pure heuristic on path segments — keeps the
/// scheduler emit path free of policy.
pub fn classify_file_path(path: &str) -> &'static str {
    if path.contains("/_versions/") || path.ends_with(".manifest") {
        "manifest"
    } else if path.contains("/_indices/") {
        "index"
    } else if path.contains("/_deletions/") {
        "deletion"
    } else if path.contains("/data/") || path.ends_with(".lance") {
        "data"
    } else {
        "other"
    }
}

fn find_query_root<S>(start: &Id, ctx: &Context<'_, S>) -> Option<Id>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    let mut current = ctx.span(start)?;
    loop {
        if current.name() == QUERY_PROFILE_ROOT {
            return Some(current.id());
        }
        current = current.parent()?;
    }
}

impl<S> Layer<S> for QueryProfileLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if !self.enabled {
            return;
        }
        let name = attrs.metadata().name();
        if name == QUERY_PROFILE_ROOT {
            self.state.lock().unwrap().insert(
                id.clone(),
                QueryState {
                    started: Instant::now(),
                    phases: HashMap::new(),
                    logical_io: HashMap::new(),
                    physical_io: HashMap::new(),
                },
            );
            return;
        }
        if let Some(phase) = phase_for(name)
            && let Some(root) = find_query_root(id, &ctx)
        {
            self.phase_enter.lock().unwrap().insert(
                id.clone(),
                PhasePending {
                    root,
                    phase,
                    enter: None,
                },
            );
        }
    }

    fn on_enter(&self, id: &Id, _ctx: Context<'_, S>) {
        if !self.enabled {
            return;
        }
        if let Some(pending) = self.phase_enter.lock().unwrap().get_mut(id)
            && pending.enter.is_none()
        {
            pending.enter = Some(Instant::now());
        }
    }

    fn on_exit(&self, id: &Id, _ctx: Context<'_, S>) {
        if !self.enabled {
            return;
        }
        let mut pending_guard = self.phase_enter.lock().unwrap();
        if let Some(pending) = pending_guard.get_mut(id)
            && let Some(t0) = pending.enter.take()
        {
            let elapsed = t0.elapsed();
            let root = pending.root.clone();
            let phase = pending.phase;
            drop(pending_guard);
            if let Some(state) = self.state.lock().unwrap().get_mut(&root) {
                *state.phases.entry(phase).or_insert(Duration::ZERO) += elapsed;
            }
        }
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        if !self.enabled {
            return;
        }
        let target = event.metadata().target();
        match target {
            t if t == TRACE_QUERY_DONE => {
                let Some(span) = ctx.event_span(event) else {
                    return;
                };
                let Some(root) = find_query_root(&span.id(), &ctx) else {
                    return;
                };
                self.flush_query(&root);
            }
            t if t == TRACE_IO_EVENTS => {
                let Some(span) = ctx.event_span(event) else {
                    return;
                };
                let Some(root) = find_query_root(&span.id(), &ctx) else {
                    return;
                };
                let mut visitor = IoFieldVisitor::default();
                event.record(&mut visitor);
                let Some(io_type) = visitor.io_type else {
                    return;
                };
                let mut state_guard = self.state.lock().unwrap();
                let Some(state) = state_guard.get_mut(&root) else {
                    return;
                };
                *state
                    .logical_io
                    .entry((io_type, visitor.index_type.unwrap_or_default()))
                    .or_default() += 1;
            }
            t if t == TRACE_IO_PHYSICAL => {
                let Some(span) = ctx.event_span(event) else {
                    return;
                };
                let Some(root) = find_query_root(&span.id(), &ctx) else {
                    return;
                };
                let mut visitor = PhysicalIoVisitor::default();
                event.record(&mut visitor);
                let bytes = visitor.bytes.unwrap_or(0);
                let duration_us = visitor.duration_us.unwrap_or(0);
                let file_type = visitor
                    .file_path
                    .as_deref()
                    .map(classify_file_path)
                    .unwrap_or("other");
                let mut state_guard = self.state.lock().unwrap();
                let Some(state) = state_guard.get_mut(&root) else {
                    return;
                };
                state
                    .physical_io
                    .entry(file_type)
                    .or_default()
                    .push((bytes, duration_us));
            }
            _ => {}
        }
    }

    fn on_close(&self, id: Id, _ctx: Context<'_, S>) {
        if !self.enabled {
            return;
        }
        // Drop any leftover phase entries belonging to this query.
        let mut guard = self.phase_enter.lock().unwrap();
        guard.retain(|_, p| p.root != id);
        drop(guard);
        // If a stream-drop already flushed this query the entry is gone;
        // otherwise emit now (e.g., a query that errored during planning).
        self.flush_query(&id);
    }
}

#[derive(Default, Debug)]
struct IoFieldVisitor {
    io_type: Option<String>,
    index_type: Option<String>,
}

impl Visit for IoFieldVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "type" => self.io_type = Some(value.to_string()),
            "index_type" => self.index_type = Some(value.to_string()),
            _ => {}
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        match field.name() {
            "type" if self.io_type.is_none() => {
                self.io_type = Some(strip_quotes(format!("{:?}", value)));
            }
            "index_type" if self.index_type.is_none() => {
                self.index_type = Some(strip_quotes(format!("{:?}", value)));
            }
            _ => {}
        }
    }
}

#[derive(Default, Debug)]
struct PhysicalIoVisitor {
    bytes: Option<u64>,
    duration_us: Option<u64>,
    file_path: Option<String>,
}

impl Visit for PhysicalIoVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "file_path" {
            self.file_path = Some(value.to_string());
        }
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "bytes" => self.bytes = Some(value),
            "duration_us" => self.duration_us = Some(value),
            _ => {}
        }
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        if value < 0 {
            return;
        }
        match field.name() {
            "bytes" => self.bytes = Some(value as u64),
            "duration_us" => self.duration_us = Some(value as u64),
            _ => {}
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "file_path" && self.file_path.is_none() {
            self.file_path = Some(strip_quotes(format!("{:?}", value)));
        }
    }
}

fn strip_quotes(mut s: String) -> String {
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        s.pop();
        s.remove(0);
    }
    s
}

fn emit_summary(state: QueryState) {
    let total_us = state.started.elapsed().as_micros() as u64;

    let p_plan = duration_us(&state, PHASE_PLAN);
    let p_oi_scalar = duration_us(&state, PHASE_OPEN_INDEX_SCALAR);
    let p_oi_vector = duration_us(&state, PHASE_OPEN_INDEX_VECTOR);
    let p_oi_frag = duration_us(&state, PHASE_OPEN_INDEX_FRAG_REUSE);
    let p_oi_mem_wal = duration_us(&state, PHASE_OPEN_INDEX_MEM_WAL);
    let p_oi_rollup = p_oi_scalar + p_oi_vector + p_oi_frag + p_oi_mem_wal;

    let p_is_scalar = duration_us(&state, PHASE_INDEX_SEARCH_SCALAR);
    let p_is_fts = duration_us(&state, PHASE_INDEX_SEARCH_FTS);
    let p_is_ann = duration_us(&state, PHASE_INDEX_SEARCH_ANN);
    let p_is_rollup = p_is_scalar + p_is_fts + p_is_ann;

    let p_ld_fr = duration_us(&state, PHASE_LOAD_DATA_FILTERED_READ);
    let p_ld_ls = duration_us(&state, PHASE_LOAD_DATA_LANCE_SCAN);
    let p_ld_take = duration_us(&state, PHASE_LOAD_DATA_TAKE);
    let p_ld_rollup = p_ld_fr + p_ld_ls + p_ld_take;

    let p_post = duration_us(&state, PHASE_POSTPROCESS);

    let logical_io_breakdown = render_logical_io(&state.logical_io);
    let io_stats = render_io_stats(&state.physical_io);

    let summary = QueryProfileSummary {
        total_us,
        phase_plan_us: p_plan,
        phase_open_index_us: p_oi_rollup,
        phase_open_index_scalar_us: p_oi_scalar,
        phase_open_index_vector_us: p_oi_vector,
        phase_open_index_frag_reuse_us: p_oi_frag,
        phase_open_index_mem_wal_us: p_oi_mem_wal,
        phase_index_search_us: p_is_rollup,
        phase_index_search_scalar_us: p_is_scalar,
        phase_index_search_fts_us: p_is_fts,
        phase_index_search_ann_us: p_is_ann,
        phase_load_data_us: p_ld_rollup,
        phase_load_data_filtered_read_us: p_ld_fr,
        phase_load_data_lance_scan_us: p_ld_ls,
        phase_load_data_take_us: p_ld_take,
        phase_postprocess_us: p_post,
        logical_io_breakdown,
        io_stats,
    };
    PENDING_SUMMARIES.with(|cell| cell.borrow_mut().push(summary));
}

fn duration_us(state: &QueryState, phase: &str) -> u64 {
    state
        .phases
        .get(phase)
        .copied()
        .unwrap_or_default()
        .as_micros() as u64
}

fn render_logical_io(map: &HashMap<(String, String), u64>) -> String {
    let mut pairs: Vec<_> = map.iter().collect();
    pairs.sort_by(|a, b| a.0.cmp(b.0));
    pairs
        .into_iter()
        .map(|((t, idx), c)| {
            if idx.is_empty() {
                format!("{}={}", t, c)
            } else {
                format!("{}/{}={}", t, idx, c)
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Render the per-file-type physical IO stats as a JSON object.
///
/// Shape:
/// ```json
/// {
///   "data": {
///     "count": 42,
///     "bytes": {"sum":..., "min":..., "p50":..., "p90":..., "p95":..., "p99":..., "max":..., "avg":...},
///     "dur_us": {"sum":..., ...},
///     "throughput_mb_s": {"min":..., "p50":..., ...}
///   },
///   "index": { ... }
/// }
/// ```
/// Throughput is computed per-sample as `bytes / duration_s`, then summarized;
/// samples with `duration_us == 0` are dropped from the throughput distribution.
fn render_io_stats(map: &HashMap<&'static str, Vec<(u64, u64)>>) -> String {
    if map.is_empty() {
        return "{}".to_string();
    }
    let mut keys: Vec<_> = map.keys().collect();
    keys.sort();
    let mut out = String::with_capacity(256);
    out.push('{');
    for (i, file_type) in keys.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let samples = &map[*file_type];
        let count = samples.len() as u64;
        let mut bytes: Vec<u64> = samples.iter().map(|(b, _)| *b).collect();
        let mut durs: Vec<u64> = samples.iter().map(|(_, d)| *d).collect();
        let mut throughputs: Vec<u64> = samples
            .iter()
            .filter_map(|(b, d)| {
                if *d == 0 {
                    None
                } else {
                    // bytes/s; integer (whole bytes per second).
                    Some((*b as u128 * 1_000_000 / *d as u128) as u64)
                }
            })
            .collect();
        bytes.sort_unstable();
        durs.sort_unstable();
        throughputs.sort_unstable();

        write!(out, "\"{}\":{{\"count\":{}", file_type, count).unwrap();
        out.push_str(",\"bytes\":");
        append_dist(&mut out, &bytes);
        out.push_str(",\"dur_us\":");
        append_dist(&mut out, &durs);
        out.push_str(",\"bytes_per_s\":");
        append_dist(&mut out, &throughputs);
        out.push('}');
    }
    out.push('}');
    out
}

/// Append `{"sum":...,"min":...,"p50":...,"p90":...,"p95":...,"p99":...,"max":...,"avg":...}`.
/// Caller must pre-sort `sorted`. Empty input is rendered as null fields.
fn append_dist(out: &mut String, sorted: &[u64]) {
    if sorted.is_empty() {
        out.push_str("{\"count\":0}");
        return;
    }
    let sum: u128 = sorted.iter().map(|v| *v as u128).sum();
    let avg = (sum / sorted.len() as u128) as u64;
    let min = sorted[0];
    let max = *sorted.last().unwrap();
    let p = |q: f64| -> u64 {
        let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
        sorted[idx]
    };
    write!(
        out,
        "{{\"sum\":{},\"min\":{},\"p50\":{},\"p90\":{},\"p95\":{},\"p99\":{},\"max\":{},\"avg\":{}}}",
        sum,
        min,
        p(0.50),
        p(0.90),
        p(0.95),
        p(0.99),
        max,
        avg,
    )
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing::{info, info_span};
    use tracing_subscriber::layer::SubscriberExt;

    use std::sync::{Arc, Mutex};

    #[derive(Default, Clone)]
    struct CaptureLayer {
        summaries: Arc<Mutex<Vec<HashMap<String, String>>>>,
    }

    impl<S> Layer<S> for CaptureLayer
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
            if event.metadata().target() != TRACE_QUERY_PROFILE {
                return;
            }
            let mut v = MapVisitor::default();
            event.record(&mut v);
            self.summaries.lock().unwrap().push(v.fields);
        }
    }

    #[derive(Default)]
    struct MapVisitor {
        fields: HashMap<String, String>,
    }

    impl Visit for MapVisitor {
        fn record_str(&mut self, field: &Field, value: &str) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }
        fn record_u64(&mut self, field: &Field, value: u64) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }
        fn record_i64(&mut self, field: &Field, value: i64) {
            self.fields
                .insert(field.name().to_string(), value.to_string());
        }
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.fields
                .insert(field.name().to_string(), format!("{:?}", value));
        }
    }

    #[test]
    fn aggregates_sub_phases_and_physical_io_into_json() {
        let capture = CaptureLayer::default();
        let summaries = capture.summaries.clone();
        let subscriber = tracing_subscriber::registry()
            .with(QueryProfileLayer::enabled())
            .with(capture);
        let _guard = tracing::subscriber::set_default(subscriber);

        {
            let root = info_span!(QUERY_PROFILE_ROOT);
            let _root_guard = root.enter();
            {
                let plan = info_span!(PHASE_PLAN);
                let _g = plan.enter();
                std::thread::sleep(Duration::from_millis(2));
            }
            {
                let oi = info_span!(PHASE_OPEN_INDEX_VECTOR);
                let _g = oi.enter();
                std::thread::sleep(Duration::from_millis(1));
            }
            {
                let ld = info_span!(PHASE_LOAD_DATA_FILTERED_READ);
                let _g = ld.enter();
                // Three physical reads at different durations + sizes.
                info!(
                    target: TRACE_IO_PHYSICAL,
                    file_path = "tbl/data/abc.lance",
                    bytes = 1000u64,
                    duration_us = 100u64,
                );
                info!(
                    target: TRACE_IO_PHYSICAL,
                    file_path = "tbl/data/abc.lance",
                    bytes = 2000u64,
                    duration_us = 200u64,
                );
                info!(
                    target: TRACE_IO_PHYSICAL,
                    file_path = "tbl/_indices/aaa/index.idx",
                    bytes = 500u64,
                    duration_us = 50u64,
                );
                info!(
                    target: TRACE_IO_EVENTS,
                    r#type = "open_vector_index",
                    index_type = "IVF_PQ",
                );
            }
            info!(target: TRACE_QUERY_DONE, "");
        }
        drain_pending_summaries();

        let captured = summaries.lock().unwrap();
        assert_eq!(captured.len(), 1, "expected one summary, got {captured:?}");
        let s = &captured[0];

        // Sub-phase + rollup are both populated.
        assert!(s.get("phase_plan_us").unwrap().parse::<u64>().unwrap() > 0);
        assert!(
            s.get("phase_open_index_vector_us")
                .unwrap()
                .parse::<u64>()
                .unwrap()
                > 0
        );
        let rollup_oi = s
            .get("phase_open_index_us")
            .unwrap()
            .parse::<u64>()
            .unwrap();
        let vec_oi = s
            .get("phase_open_index_vector_us")
            .unwrap()
            .parse::<u64>()
            .unwrap();
        assert_eq!(rollup_oi, vec_oi, "rollup should equal sum of subs");

        // logical_io_breakdown picks up the IO_EVENTS event.
        assert!(
            s.get("logical_io_breakdown")
                .unwrap()
                .contains("open_vector_index/IVF_PQ=1")
        );

        // io_stats JSON has both file types and required keys.
        let io = s.get("io_stats").unwrap();
        assert!(io.contains("\"data\""), "got {io}");
        assert!(io.contains("\"index\""), "got {io}");
        assert!(io.contains("\"bytes\""), "got {io}");
        assert!(io.contains("\"dur_us\""), "got {io}");
        assert!(io.contains("\"bytes_per_s\""), "got {io}");
        // Two data samples: sum should be 3000.
        assert!(io.contains("\"sum\":3000"), "got {io}");
    }

    #[test]
    fn disabled_layer_emits_nothing() {
        let capture = CaptureLayer::default();
        let summaries = capture.summaries.clone();
        let subscriber = tracing_subscriber::registry()
            .with(QueryProfileLayer::disabled())
            .with(capture);
        let _guard = tracing::subscriber::set_default(subscriber);
        {
            let root = info_span!(QUERY_PROFILE_ROOT);
            let _g = root.enter();
            info!(target: TRACE_QUERY_DONE, "");
        }
        drain_pending_summaries();
        assert!(summaries.lock().unwrap().is_empty());
    }

    #[test]
    fn ignores_phase_spans_outside_query_root() {
        let capture = CaptureLayer::default();
        let summaries = capture.summaries.clone();
        let subscriber = tracing_subscriber::registry()
            .with(QueryProfileLayer::enabled())
            .with(capture);
        let _guard = tracing::subscriber::set_default(subscriber);
        {
            let plan = info_span!(PHASE_PLAN);
            let _g = plan.enter();
            info!(target: TRACE_IO_PHYSICAL, file_path = "x", bytes = 1u64, duration_us = 1u64);
        }
        drain_pending_summaries();
        assert!(summaries.lock().unwrap().is_empty());
    }

    #[test]
    fn classify_file_path_buckets() {
        assert_eq!(classify_file_path("foo/_versions/123.manifest"), "manifest");
        assert_eq!(classify_file_path("foo.manifest"), "manifest");
        assert_eq!(classify_file_path("foo/_indices/aaa/index.idx"), "index");
        assert_eq!(classify_file_path("foo/_deletions/1.arrow"), "deletion");
        assert_eq!(classify_file_path("foo/data/0011aabb.lance"), "data");
        assert_eq!(classify_file_path("standalone.lance"), "data");
        assert_eq!(classify_file_path("random/path.txt"), "other");
    }
}
