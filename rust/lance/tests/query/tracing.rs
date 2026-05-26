// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::{Arc, Mutex};

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator,
    StringArray, UInt32Array,
};
use lance::Dataset;
use lance::dataset::scanner::{ColumnOrdering, QueryFilter};
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_arrow::FixedSizeListArrayExt;
use lance_index::IndexType;
use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
use lance_index::vector::Query;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_linalg::distance::MetricType;
use tracing::{Id, Instrument, Subscriber};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

use lance_core::utils::tempfile::TempStrDir;

use super::{strip_score_column, test_scan};
use crate::utils::build_multi_fragment_dataset;

const ROW_COUNT: i32 = 12;
const VALUE_OFFSET: i32 = 100;

#[derive(Clone, Debug)]
struct RecordedSpan {
    name: String,
    parent: Option<Id>,
    active_count: usize,
}

#[derive(Clone, Default, Debug)]
struct SpanTreeRecorder {
    spans: Arc<Mutex<HashMap<Id, RecordedSpan>>>,
}

impl SpanTreeRecorder {
    fn snapshot(&self) -> HashMap<Id, RecordedSpan> {
        self.spans.lock().unwrap().clone()
    }
}

impl<S> Layer<S> for SpanTreeRecorder
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let parent = attrs.parent().cloned().or_else(|| {
            if attrs.is_contextual() {
                ctx.current_span().id().cloned()
            } else {
                None
            }
        });

        let span = RecordedSpan {
            name: attrs.metadata().name().to_string(),
            parent,
            active_count: 0,
        };
        self.spans.lock().unwrap().insert(id.clone(), span);
    }

    fn on_enter(&self, id: &Id, _ctx: Context<'_, S>) {
        if let Some(span) = self.spans.lock().unwrap().get_mut(id) {
            span.active_count += 1;
        }
    }

    fn on_exit(&self, id: &Id, _ctx: Context<'_, S>) {
        if let Some(span) = self.spans.lock().unwrap().get_mut(id) {
            span.active_count = span.active_count.saturating_sub(1);
        }
    }
}

fn multi_fragment_batch() -> RecordBatch {
    let ids = Int32Array::from((0..ROW_COUNT).collect::<Vec<_>>());
    let values = Int32Array::from((VALUE_OFFSET..(VALUE_OFFSET + ROW_COUNT)).collect::<Vec<_>>());

    RecordBatch::try_from_iter(vec![
        ("id", Arc::new(ids) as ArrayRef),
        ("value", Arc::new(values) as ArrayRef),
    ])
    .unwrap()
}

fn orphan_chain_error(
    spans: &HashMap<Id, RecordedSpan>,
    span_id: &Id,
    root_id: &Id,
) -> Option<String> {
    let mut current_id = span_id.clone();
    let mut visited = HashSet::new();
    let mut chain = Vec::new();

    loop {
        let span = spans
            .get(&current_id)
            .unwrap_or_else(|| panic!("Missing recorded span for {:?}", current_id));
        chain.push(span.name.clone());

        if &current_id == root_id {
            return None;
        }

        if !visited.insert(current_id.clone()) {
            return Some(format!(
                "cycle detected for span '{}' via chain {}",
                chain.first().cloned().unwrap_or_default(),
                chain.join(" -> ")
            ));
        }

        let Some(parent) = span.parent.clone() else {
            return Some(format!(
                "span '{}' was orphaned before reaching test root; observed chain {}",
                chain.first().cloned().unwrap_or_default(),
                chain.join(" -> ")
            ));
        };
        current_id = parent;
    }
}

async fn assert_query_has_no_orphan_spans<Fut>(query: Fut)
where
    Fut: Future<Output = ()>,
{
    let recorder = SpanTreeRecorder::default();
    let subscriber = tracing_subscriber::registry().with(recorder.clone());
    let _guard = tracing::subscriber::set_default(subscriber);

    let root_span = tracing::info_span!("lance_query_test_root");
    let root_id = root_span
        .id()
        .expect("root span should be recorded by the test subscriber");

    {
        let instrumented_query = query.instrument(root_span.clone());
        instrumented_query.await;
    }

    drop(root_span);
    tokio::task::yield_now().await;

    let spans = recorder.snapshot();
    assert!(spans.contains_key(&root_id), "root span was not recorded");

    let descendants = spans
        .keys()
        .filter_map(|id| {
            if id == &root_id {
                None
            } else {
                Some(id.clone())
            }
        })
        .collect::<Vec<_>>();
    assert!(
        !descendants.is_empty(),
        "query produced no descendant spans under the test root"
    );

    let orphan_errors = descendants
        .iter()
        .filter_map(|span_id| orphan_chain_error(&spans, span_id, &root_id))
        .collect::<Vec<_>>();
    assert!(
        orphan_errors.is_empty(),
        "orphan spans detected: {:?}",
        orphan_errors
    );

    let active_spans = spans
        .values()
        .filter(|span| span.active_count > 0)
        .map(|span| span.name.clone())
        .collect::<Vec<_>>();
    assert!(
        active_spans.is_empty(),
        "spans still active after query finished: {:?}",
        active_spans
    );
}

fn multi_fragment_fts_batch() -> RecordBatch {
    let ids = Int32Array::from((0..5).collect::<Vec<_>>());
    let texts = StringArray::from(vec![
        Some("lance database"),
        Some("lance vector"),
        Some("random text"),
        Some("lance"),
        None,
    ]);
    let categories = StringArray::from(vec![
        Some("keep"),
        Some("drop"),
        Some("keep"),
        Some("keep"),
        Some("keep"),
    ]);

    RecordBatch::try_from_iter(vec![
        ("id", Arc::new(ids) as ArrayRef),
        ("text", Arc::new(texts) as ArrayRef),
        ("category", Arc::new(categories) as ArrayRef),
    ])
    .unwrap()
}

async fn test_fts_with_filter(original: &RecordBatch, ds: &Dataset) {
    let query = FullTextSearchQuery::new("lance".to_string())
        .with_column("text".to_string())
        .unwrap();

    let mut scanner = ds.scan();
    scanner.full_text_search(query).unwrap();
    scanner.filter("category = 'keep'").unwrap();
    scanner
        .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
            "id".to_string(),
        )]))
        .unwrap();

    let scanned: RecordBatch = scanner.try_into_batch().await.unwrap();
    let scanned = strip_score_column(&scanned, original.schema().as_ref());

    let expected_ids = UInt32Array::from(vec![0, 3]);
    let expected = arrow::compute::take_record_batch(original, &expected_ids).unwrap();
    assert_eq!(expected, scanned);
}

#[tokio::test(flavor = "current_thread")]
async fn test_multi_fragment_scan_does_not_create_orphan_spans() {
    let original = multi_fragment_batch();
    let ds = build_multi_fragment_dataset(original.clone()).await;

    assert_query_has_no_orphan_spans(async {
        test_scan(&original, &ds).await;
    })
    .await;
}

#[tokio::test(flavor = "current_thread")]
async fn test_multi_fragment_fts_filter_does_not_create_orphan_spans() {
    let original = multi_fragment_fts_batch();
    let mut ds = build_multi_fragment_dataset(original.clone()).await;
    let params = InvertedIndexParams::default();
    ds.create_index_builder(&["text"], IndexType::Inverted, &params)
        .await
        .unwrap();

    assert_query_has_no_orphan_spans(async {
        test_fts_with_filter(&original, &ds).await;
    })
    .await;
}

// -- Vector + Hybrid query helpers and data builders --

fn vector_batch() -> RecordBatch {
    let num_rows = 300i32;
    let ids = Int32Array::from((0..num_rows).collect::<Vec<_>>());
    let mut vectors = Vec::with_capacity(num_rows as usize * 4);
    for i in 0..num_rows {
        vectors.extend_from_slice(&[i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32]);
    }
    let vector_values = Float32Array::from(vectors);
    let vector_array = FixedSizeListArray::try_new_from_values(vector_values, 4).unwrap();

    RecordBatch::try_from_iter(vec![
        ("id", Arc::new(ids) as ArrayRef),
        ("vector", Arc::new(vector_array) as ArrayRef),
    ])
    .unwrap()
}

fn hybrid_batch() -> RecordBatch {
    let num_rows = 300i32;
    let ids = Int32Array::from((0..num_rows).collect::<Vec<_>>());
    let mut vectors = Vec::with_capacity(num_rows as usize * 4);
    for i in 0..num_rows {
        vectors.extend_from_slice(&[i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32]);
    }
    let vector_values = Float32Array::from(vectors);
    let vector_array = FixedSizeListArray::try_new_from_values(vector_values, 4).unwrap();

    let texts: Vec<String> = (0..num_rows)
        .map(|i| {
            if i % 3 == 0 {
                format!("lance database {}", i)
            } else {
                format!("other text {}", i)
            }
        })
        .collect();
    let text_array = StringArray::from(texts);

    let categories: Vec<String> = (0..num_rows)
        .map(|i| {
            if i % 2 == 0 {
                "keep".to_string()
            } else {
                "drop".to_string()
            }
        })
        .collect();
    let category_array = StringArray::from(categories);

    RecordBatch::try_from_iter(vec![
        ("id", Arc::new(ids) as ArrayRef),
        ("vector", Arc::new(vector_array) as ArrayRef),
        ("text", Arc::new(text_array) as ArrayRef),
        ("category", Arc::new(category_array) as ArrayRef),
    ])
    .unwrap()
}

async fn build_vector_dataset(batch: RecordBatch, uri: &str) -> Dataset {
    let schema = batch.schema();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut ds = Dataset::write(reader, uri, None).await.unwrap();
    let params = VectorIndexParams::with_ivf_pq_params(
        MetricType::L2,
        IvfBuildParams::new(2),
        PQBuildParams::new(4, 8),
    );
    ds.create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    ds
}

async fn build_hybrid_dataset(batch: RecordBatch, uri: &str) -> Dataset {
    let schema = batch.schema();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut ds = Dataset::write(reader, uri, None).await.unwrap();

    // Vector index
    let vector_params = VectorIndexParams::with_ivf_pq_params(
        MetricType::L2,
        IvfBuildParams::new(2),
        PQBuildParams::new(4, 8),
    );
    ds.create_index(&["vector"], IndexType::Vector, None, &vector_params, true)
        .await
        .unwrap();

    // FTS inverted index
    let fts_params = InvertedIndexParams::default();
    ds.create_index_builder(&["text"], IndexType::Inverted, &fts_params)
        .await
        .unwrap();

    ds
}

#[tokio::test(flavor = "current_thread")]
async fn test_vector_search_does_not_create_orphan_spans() {
    let batch = vector_batch();
    let ds = build_vector_dataset(batch, "memory://vector_tracing_test").await;

    let query_vector = Float32Array::from(vec![10.0f32, 20.0, 30.0, 40.0]);

    assert_query_has_no_orphan_spans(async {
        let mut scanner = ds.scan();
        scanner
            .nearest("vector", &query_vector, 5)
            .unwrap()
            .nprobes(2);
        let _batch = scanner.try_into_batch().await.unwrap();
    })
    .await;
}

#[tokio::test(flavor = "current_thread")]
async fn test_hybrid_fts_vector_rerank_does_not_create_orphan_spans() {
    let batch = hybrid_batch();
    let ds = build_hybrid_dataset(batch, "memory://hybrid_tracing_test").await;

    let query_vector = Arc::new(Float32Array::from(vec![10.0f32, 20.0, 30.0, 40.0]));
    let vector_query = Query {
        column: "vector".to_string(),
        key: query_vector,
        k: 10,
        lower_bound: None,
        upper_bound: None,
        minimum_nprobes: 2,
        maximum_nprobes: None,
        ef: None,
        refine_factor: None,
        metric_type: Some(MetricType::L2),
        use_index: true,
        dist_q_c: 0.0,
        query_parallelism: 0,
    };

    assert_query_has_no_orphan_spans(async {
        // FTS as primary search, vector as filter → triggers fts_rerank()
        let mut scanner = ds.scan();
        scanner
            .full_text_search(
                FullTextSearchQuery::new("lance".to_string())
                    .with_column("text".to_string())
                    .unwrap(),
            )
            .unwrap();
        scanner.prefilter(true);
        scanner
            .filter_query(QueryFilter::Vector(vector_query))
            .unwrap();
        let _batch = scanner.try_into_batch().await.unwrap();
    })
    .await;
}

// -- Local filesystem tests (exercises spawn_blocking in local.rs) --

#[tokio::test(flavor = "current_thread")]
async fn test_local_scan_does_not_create_orphan_spans() {
    let test_dir = TempStrDir::default();
    let original = multi_fragment_batch();
    let schema = original.schema();
    let reader = RecordBatchIterator::new(vec![Ok(original.clone())], schema);
    let ds = Dataset::write(reader, test_dir.as_str(), None)
        .await
        .unwrap();

    assert_query_has_no_orphan_spans(async {
        test_scan(&original, &ds).await;
    })
    .await;
}

#[tokio::test(flavor = "current_thread")]
async fn test_local_vector_search_does_not_create_orphan_spans() {
    let test_dir = TempStrDir::default();
    let batch = vector_batch();
    let ds = build_vector_dataset(batch, test_dir.as_str()).await;

    let query_vector = Float32Array::from(vec![10.0f32, 20.0, 30.0, 40.0]);

    assert_query_has_no_orphan_spans(async {
        let mut scanner = ds.scan();
        scanner
            .nearest("vector", &query_vector, 5)
            .unwrap()
            .nprobes(2);
        let _batch = scanner.try_into_batch().await.unwrap();
    })
    .await;
}

#[tokio::test(flavor = "current_thread")]
async fn test_local_hybrid_fts_vector_rerank_does_not_create_orphan_spans() {
    let test_dir = TempStrDir::default();
    let batch = hybrid_batch();
    let ds = build_hybrid_dataset(batch, test_dir.as_str()).await;

    let query_vector = Arc::new(Float32Array::from(vec![10.0f32, 20.0, 30.0, 40.0]));
    let vector_query = Query {
        column: "vector".to_string(),
        key: query_vector,
        k: 10,
        lower_bound: None,
        upper_bound: None,
        minimum_nprobes: 2,
        maximum_nprobes: None,
        ef: None,
        refine_factor: None,
        metric_type: Some(MetricType::L2),
        use_index: true,
        dist_q_c: 0.0,
        query_parallelism: 0,
    };

    assert_query_has_no_orphan_spans(async {
        let mut scanner = ds.scan();
        scanner
            .full_text_search(
                FullTextSearchQuery::new("lance".to_string())
                    .with_column("text".to_string())
                    .unwrap(),
            )
            .unwrap();
        scanner.prefilter(true);
        scanner
            .filter_query(QueryFilter::Vector(vector_query))
            .unwrap();
        let _batch = scanner.try_into_batch().await.unwrap();
    })
    .await;
}
