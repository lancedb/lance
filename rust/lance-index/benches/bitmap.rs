// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of Bitmap scalar index.
//!
//! This benchmark measures the performance of Bitmap index with:
//! - 50 million data points
//! - Float and String data types
//! - High cardinality (unique values) and low cardinality (100 unique values)
//! - Equality filters
//! - Range filters with varying selectivity (few/many/most rows match)

mod common;

use std::{ops::Bound, sync::Arc, time::Duration};

use common::{Selectivity, LOW_CARDINALITY_COUNT, TOTAL_ROWS};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use datafusion_common::ScalarValue;
use lance_core::cache::LanceCache;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::pbold;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::registry::ScalarIndexPlugin;
use lance_index::scalar::{bitmap::BitmapIndexPlugin, SargableQuery, ScalarIndex};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

/// Container for all benchmark indices
struct BenchmarkIndices {
    float_unique: Arc<dyn ScalarIndex>,
    float_low_card: Arc<dyn ScalarIndex>,
    string_unique: Arc<dyn ScalarIndex>,
    string_low_card: Arc<dyn ScalarIndex>,
    // Keep temp directories alive for the lifetime of the indices
    _temp_dirs: Vec<tempfile::TempDir>,
}

/// Create and train a Bitmap index for float data with unique values
async fn create_float_unique_index(store: Arc<LanceIndexStore>) -> Arc<dyn ScalarIndex> {
    let stream = common::generate_float_unique_stream();

    BitmapIndexPlugin::train_bitmap_index(stream, store.as_ref())
        .await
        .unwrap();

    let details = prost_types::Any::from_msg(&pbold::BitmapIndexDetails::default()).unwrap();
    let index = BitmapIndexPlugin
        .load_index(store, &details, None, &LanceCache::no_cache())
        .await
        .unwrap();

    index
}

/// Create and train a Bitmap index for float data with low cardinality
async fn create_float_low_card_index(store: Arc<LanceIndexStore>) -> Arc<dyn ScalarIndex> {
    let stream = common::generate_float_low_cardinality_stream();

    BitmapIndexPlugin::train_bitmap_index(stream, store.as_ref())
        .await
        .unwrap();

    let details = prost_types::Any::from_msg(&pbold::BitmapIndexDetails::default()).unwrap();
    let index = BitmapIndexPlugin
        .load_index(store, &details, None, &LanceCache::no_cache())
        .await
        .unwrap();

    index
}

/// Create and train a Bitmap index for string data with unique values
async fn create_string_unique_index(store: Arc<LanceIndexStore>) -> Arc<dyn ScalarIndex> {
    let stream = common::generate_string_unique_stream();

    BitmapIndexPlugin::train_bitmap_index(stream, store.as_ref())
        .await
        .unwrap();

    let details = prost_types::Any::from_msg(&pbold::BitmapIndexDetails::default()).unwrap();
    let index = BitmapIndexPlugin
        .load_index(store, &details, None, &LanceCache::no_cache())
        .await
        .unwrap();

    index
}

/// Create and train a Bitmap index for string data with low cardinality
async fn create_string_low_card_index(store: Arc<LanceIndexStore>) -> Arc<dyn ScalarIndex> {
    let stream = common::generate_string_low_cardinality_stream();

    BitmapIndexPlugin::train_bitmap_index(stream, store.as_ref())
        .await
        .unwrap();

    let details = prost_types::Any::from_msg(&pbold::BitmapIndexDetails::default()).unwrap();
    let index = BitmapIndexPlugin
        .load_index(store, &details, None, &LanceCache::no_cache())
        .await
        .unwrap();

    index
}

/// Set up all benchmark indices
fn setup_indices(rt: &tokio::runtime::Runtime) -> BenchmarkIndices {
    println!(
        "Setting up bitmap benchmark indices with {} rows...",
        TOTAL_ROWS
    );

    let indices = rt.block_on(async {
        // Create temporary directories for each index
        let tempdir_float_unique = tempfile::tempdir().unwrap();
        let tempdir_float_low_card = tempfile::tempdir().unwrap();
        let tempdir_string_unique = tempfile::tempdir().unwrap();
        let tempdir_string_low_card = tempfile::tempdir().unwrap();

        let store_float_unique = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tempdir_float_unique.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        let store_float_low_card = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tempdir_float_low_card.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        let store_string_unique = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tempdir_string_unique.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        let store_string_low_card = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tempdir_string_low_card.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        println!("Creating float unique bitmap index...");
        let float_unique = create_float_unique_index(store_float_unique).await;

        println!("Creating float low cardinality bitmap index...");
        let float_low_card = create_float_low_card_index(store_float_low_card).await;

        println!("Creating string unique bitmap index...");
        let string_unique = create_string_unique_index(store_string_unique).await;

        println!("Creating string low cardinality bitmap index...");
        let string_low_card = create_string_low_card_index(store_string_low_card).await;

        BenchmarkIndices {
            float_unique,
            float_low_card,
            string_unique,
            string_low_card,
            // Keep temp directories alive to prevent deletion of index data
            _temp_dirs: vec![
                tempdir_float_unique,
                tempdir_float_low_card,
                tempdir_string_unique,
                tempdir_string_low_card,
            ],
        }
    });

    println!("Bitmap setup complete!");
    indices
}

fn bench_equality(c: &mut Criterion, indices: &BenchmarkIndices) {
    let rt = tokio::runtime::Builder::new_multi_thread().build().unwrap();

    let mut group = c.benchmark_group("bitmap_equality");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));

    // Float unique
    group.bench_function(BenchmarkId::from_parameter("float_unique"), |b| {
        b.to_async(&rt).iter(|| {
            let index = indices.float_unique.clone();
            async move {
                let query = SargableQuery::Equals(ScalarValue::Float64(Some(25_000_000.0)));
                black_box(index.search(&query, &NoOpMetricsCollector).await.unwrap());
            }
        })
    });

    // Float low cardinality
    group.bench_function(BenchmarkId::from_parameter("float_low_card"), |b| {
        b.to_async(&rt).iter(|| {
            let index = indices.float_low_card.clone();
            async move {
                let query = SargableQuery::Equals(ScalarValue::Float64(Some(50.0)));
                black_box(index.search(&query, &NoOpMetricsCollector).await.unwrap());
            }
        })
    });

    // String unique
    group.bench_function(BenchmarkId::from_parameter("string_unique"), |b| {
        b.to_async(&rt).iter(|| {
            let index = indices.string_unique.clone();
            async move {
                let query =
                    SargableQuery::Equals(ScalarValue::Utf8(Some("string_0025000000".to_string())));
                black_box(index.search(&query, &NoOpMetricsCollector).await.unwrap());
            }
        })
    });

    // String low cardinality
    group.bench_function(BenchmarkId::from_parameter("string_low_card"), |b| {
        b.to_async(&rt).iter(|| {
            let index = indices.string_low_card.clone();
            async move {
                let query = SargableQuery::Equals(ScalarValue::Utf8(Some("value_050".to_string())));
                black_box(index.search(&query, &NoOpMetricsCollector).await.unwrap());
            }
        })
    });

    group.finish();
}

/// Helper function to count results from a range query
fn count_range_results(
    rt: &tokio::runtime::Runtime,
    index: &Arc<dyn ScalarIndex>,
    query: SargableQuery,
) -> usize {
    rt.block_on(async {
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        match result {
            lance_index::scalar::SearchResult::Exact(row_ids) => {
                row_ids.len().expect("Expected exact row count") as usize
            }
            _ => panic!("Expected exact search result"),
        }
    })
}

fn bench_range(c: &mut Criterion, indices: &BenchmarkIndices, selectivity: Selectivity) {
    let rt = tokio::runtime::Builder::new_multi_thread().build().unwrap();

    let group_name = format!("bitmap_range_{}", selectivity.name());
    let mut group = c.benchmark_group(&group_name);
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));

    let pct = selectivity.percentage();

    // Float unique - range queries
    let float_range_size = (TOTAL_ROWS as f64 * pct) as u64;
    let float_start = (TOTAL_ROWS / 2) - (float_range_size / 2);
    let float_end = float_start + float_range_size;

    // Sanity check: verify float unique range returns expected count
    let float_unique_query = SargableQuery::Range(
        Bound::Included(ScalarValue::Float64(Some(float_start as f64))),
        Bound::Included(ScalarValue::Float64(Some(float_end as f64))),
    );
    let float_unique_count = count_range_results(&rt, &indices.float_unique, float_unique_query);
    let expected_count = (float_end - float_start + 1) as usize;
    println!(
        "[{}] Bitmap Float unique range [{}, {}]: expected ~{} rows, got {} rows ({}%)",
        selectivity.name(),
        float_start,
        float_end,
        expected_count,
        float_unique_count,
        (float_unique_count as f64 / expected_count as f64 * 100.0)
    );
    assert!(
        (float_unique_count as f64 - expected_count as f64).abs() / (expected_count as f64) < 0.01,
        "Float unique count mismatch: expected {}, got {}",
        expected_count,
        float_unique_count
    );

    group.bench_function(BenchmarkId::from_parameter("float_unique"), |b| {
        b.to_async(&rt).iter(|| {
            let index = indices.float_unique.clone();
            async move {
                let query = SargableQuery::Range(
                    Bound::Included(ScalarValue::Float64(Some(float_start as f64))),
                    Bound::Included(ScalarValue::Float64(Some(float_end as f64))),
                );
                black_box(index.search(&query, &NoOpMetricsCollector).await.unwrap());
            }
        })
    });

    // Float low cardinality - range queries
    let low_card_range_size = (LOW_CARDINALITY_COUNT as f64 * pct) as usize;
    let low_card_start = (LOW_CARDINALITY_COUNT / 2) - (low_card_range_size / 2);
    let low_card_end = low_card_start + low_card_range_size;

    // Sanity check: verify float low cardinality range returns expected count
    let float_low_card_query = SargableQuery::Range(
        Bound::Included(ScalarValue::Float64(Some(low_card_start as f64))),
        Bound::Included(ScalarValue::Float64(Some(low_card_end as f64))),
    );
    let float_low_card_count =
        count_range_results(&rt, &indices.float_low_card, float_low_card_query);
    let rows_per_value = TOTAL_ROWS / LOW_CARDINALITY_COUNT as u64;
    let expected_low_card_count =
        ((low_card_end - low_card_start + 1) as u64 * rows_per_value) as usize;
    println!(
        "[{}] Bitmap Float low cardinality range [{}, {}]: expected ~{} rows, got {} rows ({}%)",
        selectivity.name(),
        low_card_start,
        low_card_end,
        expected_low_card_count,
        float_low_card_count,
        (float_low_card_count as f64 / expected_low_card_count as f64 * 100.0)
    );
    assert!(
        (float_low_card_count as f64 - expected_low_card_count as f64).abs()
            / (expected_low_card_count as f64)
            < 0.01,
        "Float low cardinality count mismatch: expected {}, got {}",
        expected_low_card_count,
        float_low_card_count
    );

    group.bench_function(BenchmarkId::from_parameter("float_low_card"), |b| {
        b.to_async(&rt).iter(|| {
            let index = indices.float_low_card.clone();
            async move {
                let query = SargableQuery::Range(
                    Bound::Included(ScalarValue::Float64(Some(low_card_start as f64))),
                    Bound::Included(ScalarValue::Float64(Some(low_card_end as f64))),
                );
                black_box(index.search(&query, &NoOpMetricsCollector).await.unwrap());
            }
        })
    });

    // String unique - range queries
    let string_start_row = float_start;
    let string_end_row = float_end;

    // Sanity check: verify string unique range returns expected count
    let string_unique_query = SargableQuery::Range(
        Bound::Included(ScalarValue::Utf8(Some(format!(
            "string_{:010}",
            string_start_row
        )))),
        Bound::Included(ScalarValue::Utf8(Some(format!(
            "string_{:010}",
            string_end_row
        )))),
    );
    let string_unique_count = count_range_results(&rt, &indices.string_unique, string_unique_query);
    let expected_string_count = (string_end_row - string_start_row + 1) as usize;
    println!(
        "[{}] Bitmap String unique range [string_{:010}, string_{:010}]: expected ~{} rows, got {} rows ({}%)",
        selectivity.name(),
        string_start_row,
        string_end_row,
        expected_string_count,
        string_unique_count,
        (string_unique_count as f64 / expected_string_count as f64 * 100.0)
    );
    assert!(
        (string_unique_count as f64 - expected_string_count as f64).abs()
            / (expected_string_count as f64)
            < 0.01,
        "String unique count mismatch: expected {}, got {}",
        expected_string_count,
        string_unique_count
    );

    group.bench_function(BenchmarkId::from_parameter("string_unique"), |b| {
        b.to_async(&rt).iter(|| {
            let index = indices.string_unique.clone();
            async move {
                let query = SargableQuery::Range(
                    Bound::Included(ScalarValue::Utf8(Some(format!(
                        "string_{:010}",
                        string_start_row
                    )))),
                    Bound::Included(ScalarValue::Utf8(Some(format!(
                        "string_{:010}",
                        string_end_row
                    )))),
                );
                black_box(index.search(&query, &NoOpMetricsCollector).await.unwrap());
            }
        })
    });

    // String low cardinality - range queries
    // Sanity check: verify string low cardinality range returns expected count
    let string_low_card_query = SargableQuery::Range(
        Bound::Included(ScalarValue::Utf8(Some(format!(
            "value_{:03}",
            low_card_start
        )))),
        Bound::Included(ScalarValue::Utf8(Some(format!(
            "value_{:03}",
            low_card_end
        )))),
    );
    let string_low_card_count =
        count_range_results(&rt, &indices.string_low_card, string_low_card_query);
    let expected_string_low_card_count =
        ((low_card_end - low_card_start + 1) as u64 * rows_per_value) as usize;
    println!(
        "[{}] Bitmap String low cardinality range [value_{:03}, value_{:03}]: expected ~{} rows, got {} rows ({}%)",
        selectivity.name(),
        low_card_start,
        low_card_end,
        expected_string_low_card_count,
        string_low_card_count,
        (string_low_card_count as f64 / expected_string_low_card_count as f64 * 100.0)
    );
    assert!(
        (string_low_card_count as f64 - expected_string_low_card_count as f64).abs()
            / (expected_string_low_card_count as f64)
            < 0.01,
        "String low cardinality count mismatch: expected {}, got {}",
        expected_string_low_card_count,
        string_low_card_count
    );

    group.bench_function(BenchmarkId::from_parameter("string_low_card"), |b| {
        b.to_async(&rt).iter(|| {
            let index = indices.string_low_card.clone();
            async move {
                let query = SargableQuery::Range(
                    Bound::Included(ScalarValue::Utf8(Some(format!(
                        "value_{:03}",
                        low_card_start
                    )))),
                    Bound::Included(ScalarValue::Utf8(Some(format!(
                        "value_{:03}",
                        low_card_end
                    )))),
                );
                black_box(index.search(&query, &NoOpMetricsCollector).await.unwrap());
            }
        })
    });

    group.finish();
}

fn bench_bitmap(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread().build().unwrap();

    // Set up all indices once
    let indices = setup_indices(&rt);

    // Run equality benchmarks
    bench_equality(c, &indices);

    // Run range benchmarks with different selectivities
    bench_range(c, &indices, Selectivity::Few);
    bench_range(c, &indices, Selectivity::Many);
    bench_range(c, &indices, Selectivity::Most);
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_bitmap);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10);
    targets = bench_bitmap);

criterion_main!(benches);
