// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator,
    StringArray,
};
use lance::Dataset;
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::utils::profile::QueryProfileLayer;
use lance_core::utils::tempfile::TempStrDir;
use lance_core::utils::tracing::TRACE_QUERY_PROFILE;
use lance_index::IndexType;
use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_linalg::distance::MetricType;
use tracing::Subscriber;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

use crate::utils::build_multi_fragment_dataset;

#[derive(Default, Clone)]
struct CaptureLayer {
    summaries: Arc<Mutex<Vec<HashMap<String, String>>>>,
}

impl<S> Layer<S> for CaptureLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        if event.metadata().target() != TRACE_QUERY_PROFILE {
            return;
        }
        let mut visitor = MapVisitor::default();
        event.record(&mut visitor);
        self.summaries.lock().unwrap().push(visitor.fields);
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

async fn run_with_profile<F, Fut>(body: F) -> Vec<HashMap<String, String>>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let capture = CaptureLayer::default();
    let summaries = capture.summaries.clone();
    let subscriber = tracing_subscriber::registry()
        .with(QueryProfileLayer::enabled())
        .with(capture);
    let _guard = tracing::subscriber::set_default(subscriber);

    body().await;
    // Yield repeatedly so spawned tasks holding span clones drop them before
    // we snapshot. Vector / FTS execution paths spawn parallel sub-searches
    // that briefly outlive the stream's `try_collect` completion.
    for _ in 0..32 {
        tokio::task::yield_now().await;
    }
    summaries.lock().unwrap().clone()
}

fn scan_batch() -> RecordBatch {
    let ids = Int32Array::from((0..12).collect::<Vec<_>>());
    let values = Int32Array::from((100..112).collect::<Vec<_>>());
    RecordBatch::try_from_iter(vec![
        ("id", Arc::new(ids) as ArrayRef),
        ("value", Arc::new(values) as ArrayRef),
    ])
    .unwrap()
}

fn vector_batch() -> RecordBatch {
    let n = 300i32;
    let ids = Int32Array::from((0..n).collect::<Vec<_>>());
    let mut vecs = Vec::with_capacity(n as usize * 4);
    for i in 0..n {
        vecs.extend_from_slice(&[i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32]);
    }
    let v = FixedSizeListArray::try_new_from_values(Float32Array::from(vecs), 4).unwrap();
    RecordBatch::try_from_iter(vec![
        ("id", Arc::new(ids) as ArrayRef),
        ("vector", Arc::new(v) as ArrayRef),
    ])
    .unwrap()
}

fn fts_batch() -> RecordBatch {
    let ids = Int32Array::from((0..5).collect::<Vec<_>>());
    let texts = StringArray::from(vec![
        Some("lance database"),
        Some("lance vector"),
        Some("random text"),
        Some("lance"),
        None,
    ]);
    RecordBatch::try_from_iter(vec![
        ("id", Arc::new(ids) as ArrayRef),
        ("text", Arc::new(texts) as ArrayRef),
    ])
    .unwrap()
}

fn parse_u64(s: &str) -> u64 {
    s.parse().unwrap_or_else(|_| panic!("not u64: {s}"))
}

#[tokio::test(flavor = "current_thread")]
async fn profile_event_emitted_for_scan() {
    let original = scan_batch();
    let ds = build_multi_fragment_dataset(original.clone()).await;

    let summaries = run_with_profile(|| async {
        let mut scanner = ds.scan();
        scanner.project(&["id", "value"]).unwrap();
        let _ = scanner.try_into_batch().await.unwrap();
    })
    .await;

    assert_eq!(
        summaries.len(),
        1,
        "expected one profile event, got {summaries:?}"
    );
    let s = &summaries[0];
    assert!(
        parse_u64(s.get("phase_plan_us").unwrap()) > 0,
        "phase_plan_us should be > 0: {s:?}"
    );
    // Either lance_scan or filtered_read path runs; their rollup must move.
    assert!(
        parse_u64(s.get("phase_load_data_us").unwrap()) > 0,
        "phase_load_data_us should be > 0: {s:?}"
    );
    // io_stats is a JSON string, never empty for a real scan.
    let io = s.get("io_stats").unwrap();
    assert!(
        io.starts_with('{') && io.ends_with('}'),
        "io_stats should be a JSON object: {io}"
    );
    assert!(
        io.contains("\"data\""),
        "expected at least one 'data' read in io_stats: {io}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn profile_event_for_vector_search_includes_index_phases() {
    let test_dir = TempStrDir::default();
    let batch = vector_batch();
    let schema = batch.schema();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut ds = Dataset::write(reader, test_dir.as_str(), None)
        .await
        .unwrap();
    let params = VectorIndexParams::with_ivf_pq_params(
        MetricType::L2,
        IvfBuildParams::new(2),
        PQBuildParams::new(4, 8),
    );
    ds.create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();

    let q = Float32Array::from(vec![10.0f32, 20.0, 30.0, 40.0]);
    let summaries = run_with_profile(|| {
        let ds = ds.clone();
        async move {
            let mut scanner = ds.scan();
            scanner.nearest("vector", &q, 5).unwrap().nprobes(2);
            let _ = scanner.try_into_batch().await.unwrap();
            drop(scanner);
            drop(ds);
        }
    })
    .await;

    assert_eq!(
        summaries.len(),
        1,
        "expected one profile event, got {summaries:?}"
    );
    let s = &summaries[0];
    assert!(parse_u64(s.get("phase_plan_us").unwrap()) > 0);
    // Vector path must populate the vector sub-phase AND the rollup.
    let oi_vec = parse_u64(s.get("phase_open_index_vector_us").unwrap());
    let oi_roll = parse_u64(s.get("phase_open_index_us").unwrap());
    assert!(oi_vec > 0, "open_index.vector should be > 0: {s:?}");
    assert_eq!(
        oi_roll,
        oi_vec
            + parse_u64(s.get("phase_open_index_scalar_us").unwrap())
            + parse_u64(s.get("phase_open_index_frag_reuse_us").unwrap())
            + parse_u64(s.get("phase_open_index_mem_wal_us").unwrap()),
        "rollup must equal sum of sub-phases: {s:?}"
    );
    assert!(
        parse_u64(s.get("phase_index_search_ann_us").unwrap()) > 0,
        "ann sub-phase should be > 0: {s:?}"
    );
    let breakdown = s.get("logical_io_breakdown").unwrap();
    assert!(
        breakdown.contains("open_vector_index"),
        "expected open_vector_index in breakdown: {breakdown}"
    );
    let io = s.get("io_stats").unwrap();
    assert!(io.contains("\"index\""), "expected index file reads: {io}");
}

#[tokio::test(flavor = "current_thread")]
async fn profile_event_for_fts_includes_open_scalar() {
    let test_dir = TempStrDir::default();
    let batch = fts_batch();
    let schema = batch.schema();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut ds = Dataset::write(reader, test_dir.as_str(), None)
        .await
        .unwrap();
    let params = InvertedIndexParams::default();
    ds.create_index_builder(&["text"], IndexType::Inverted, &params)
        .await
        .unwrap();

    let summaries = run_with_profile(|| async {
        let mut scanner = ds.scan();
        scanner
            .full_text_search(
                FullTextSearchQuery::new("lance".to_string())
                    .with_column("text".to_string())
                    .unwrap(),
            )
            .unwrap();
        let _ = scanner.try_into_batch().await.unwrap();
    })
    .await;

    assert_eq!(
        summaries.len(),
        1,
        "expected one profile event, got {summaries:?}"
    );
    let s = &summaries[0];
    let breakdown = s.get("logical_io_breakdown").unwrap();
    assert!(
        breakdown.contains("open_scalar_index"),
        "expected open_scalar_index in breakdown: {breakdown}"
    );
    assert!(
        parse_u64(s.get("phase_open_index_scalar_us").unwrap()) > 0,
        "open_index.scalar should be > 0: {s:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn disabled_layer_emits_nothing() {
    let original = scan_batch();
    let ds = build_multi_fragment_dataset(original.clone()).await;

    let capture = CaptureLayer::default();
    let summaries = capture.summaries.clone();
    let subscriber = tracing_subscriber::registry()
        .with(QueryProfileLayer::disabled())
        .with(capture);
    let _guard = tracing::subscriber::set_default(subscriber);

    let mut scanner = ds.scan();
    scanner.project(&["id", "value"]).unwrap();
    let _ = scanner.try_into_batch().await.unwrap();
    tokio::task::yield_now().await;

    assert!(summaries.lock().unwrap().is_empty());
}
