// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of Compound BTree scalar index.
//!
//! This benchmark measures the performance of compound (multi-column) BTree index with:
//! - 1 million data points
//! - 2-column and 3-column indexes
//! - Various query patterns: full key lookup, prefix lookup, prefix + range
//! - Cached and uncached variants

mod common;

use std::{
    ops::Bound,
    sync::{Arc, OnceLock},
    time::Duration,
};

use arrow_schema::DataType;
use common::{
    generate_compound_2col_int_unique_stream, generate_compound_2col_tenant_stream,
    generate_compound_3col_stream, COMPOUND_STATUS_COUNT, COMPOUND_TENANT_COUNT,
    COMPOUND_TIMESTAMPS_PER_STATUS, COMPOUND_TIMESTAMPS_PER_TENANT,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use datafusion_common::ScalarValue;
use lance_core::cache::LanceCache;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::pb;
use lance_index::scalar::compound::{CompoundIndexSchema, CompoundSargableQuery};
use lance_index::scalar::compound_btree::{
    train_compound_btree_index, CompoundBTreeIndexPlugin, CompoundFlatIndexMetadata,
    DEFAULT_COMPOUND_BATCH_SIZE,
};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::registry::ScalarIndexPlugin;
use lance_index::scalar::{AnyQuery, ScalarIndex};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

/// Selectivity level for range queries
#[derive(Clone, Copy, Debug)]
enum Selectivity {
    Few,  // ~0.1% of rows
    Many, // ~10% of rows
    Most, // ~90% of rows
}

impl Selectivity {
    fn name(&self) -> &'static str {
        match self {
            Self::Few => "few",
            Self::Many => "many",
            Self::Most => "most",
        }
    }

    /// Get the approximate percentage of rows that should match
    fn percentage(&self) -> f64 {
        match self {
            Self::Few => 0.001,
            Self::Many => 0.10,
            Self::Most => 0.90,
        }
    }
}

// Lazy static runtime - only created once
static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

// Lazy static cache - only created when cached benchmarks are run
static CACHE: OnceLock<Arc<LanceCache>> = OnceLock::new();

// Lazy static indices - only created when first accessed
// 2-column int unique index
static COMPOUND_2COL_INT_INDEX_NO_CACHE: OnceLock<Arc<dyn ScalarIndex>> = OnceLock::new();
static COMPOUND_2COL_INT_INDEX_CACHED: OnceLock<Arc<dyn ScalarIndex>> = OnceLock::new();
// 2-column tenant (string + int) index
static COMPOUND_2COL_TENANT_INDEX_NO_CACHE: OnceLock<Arc<dyn ScalarIndex>> = OnceLock::new();
static COMPOUND_2COL_TENANT_INDEX_CACHED: OnceLock<Arc<dyn ScalarIndex>> = OnceLock::new();
// 3-column index
static COMPOUND_3COL_INDEX_NO_CACHE: OnceLock<Arc<dyn ScalarIndex>> = OnceLock::new();
static COMPOUND_3COL_INDEX_CACHED: OnceLock<Arc<dyn ScalarIndex>> = OnceLock::new();

/// Get or create the tokio runtime
fn get_runtime() -> &'static tokio::runtime::Runtime {
    RUNTIME.get_or_init(|| tokio::runtime::Builder::new_multi_thread().build().unwrap())
}

/// Get the cache - either a singleton cache or no_cache based on use_cache parameter
fn get_cache(use_cache: bool, key_prefix: &str) -> Arc<LanceCache> {
    if use_cache {
        Arc::new(
            CACHE
                .get_or_init(|| Arc::new(LanceCache::with_capacity(1024 * 1024 * 1024)))
                .with_key_prefix(key_prefix),
        )
    } else {
        Arc::new(LanceCache::no_cache())
    }
}

/// Create and train a 2-column compound index with int keys
async fn create_compound_2col_int_index(
    store: Arc<LanceIndexStore>,
    use_cache: bool,
) -> Arc<dyn ScalarIndex> {
    let stream = generate_compound_2col_int_unique_stream();

    let column_names = vec!["key1".to_string(), "key2".to_string()];
    let data_types = vec![DataType::Int64, DataType::Int64];

    let compound_schema = CompoundIndexSchema::new(column_names.clone(), data_types.clone())
        .expect("Failed to create compound schema");
    let sub_index = CompoundFlatIndexMetadata::new(column_names.clone(), data_types)
        .expect("Failed to create sub index metadata");

    train_compound_btree_index(
        stream,
        &sub_index,
        store.as_ref(),
        &compound_schema,
        DEFAULT_COMPOUND_BATCH_SIZE,
        None,
    )
    .await
    .expect("Failed to train compound index");

    let cache = get_cache(use_cache, "compound_2col_int");
    let details = prost_types::Any::from_msg(&pb::CompoundBTreeIndexDetails {
        column_names,
        num_columns: 2,
    })
    .unwrap();

    CompoundBTreeIndexPlugin
        .load_index(store, &details, None, &cache)
        .await
        .expect("Failed to load compound index")
}

/// Create and train a 2-column compound index with tenant (string + int)
async fn create_compound_2col_tenant_index(
    store: Arc<LanceIndexStore>,
    use_cache: bool,
) -> Arc<dyn ScalarIndex> {
    let stream = generate_compound_2col_tenant_stream();

    let column_names = vec!["tenant".to_string(), "timestamp".to_string()];
    let data_types = vec![DataType::Utf8, DataType::Int64];

    let compound_schema = CompoundIndexSchema::new(column_names.clone(), data_types.clone())
        .expect("Failed to create compound schema");
    let sub_index = CompoundFlatIndexMetadata::new(column_names.clone(), data_types)
        .expect("Failed to create sub index metadata");

    train_compound_btree_index(
        stream,
        &sub_index,
        store.as_ref(),
        &compound_schema,
        DEFAULT_COMPOUND_BATCH_SIZE,
        None,
    )
    .await
    .expect("Failed to train compound index");

    let cache = get_cache(use_cache, "compound_2col_tenant");
    let details = prost_types::Any::from_msg(&pb::CompoundBTreeIndexDetails {
        column_names,
        num_columns: 2,
    })
    .unwrap();

    CompoundBTreeIndexPlugin
        .load_index(store, &details, None, &cache)
        .await
        .expect("Failed to load compound index")
}

/// Create and train a 3-column compound index
async fn create_compound_3col_index(
    store: Arc<LanceIndexStore>,
    use_cache: bool,
) -> Arc<dyn ScalarIndex> {
    let stream = generate_compound_3col_stream();

    let column_names = vec![
        "tenant".to_string(),
        "status".to_string(),
        "timestamp".to_string(),
    ];
    let data_types = vec![DataType::Utf8, DataType::Utf8, DataType::Int64];

    let compound_schema = CompoundIndexSchema::new(column_names.clone(), data_types.clone())
        .expect("Failed to create compound schema");
    let sub_index = CompoundFlatIndexMetadata::new(column_names.clone(), data_types)
        .expect("Failed to create sub index metadata");

    train_compound_btree_index(
        stream,
        &sub_index,
        store.as_ref(),
        &compound_schema,
        DEFAULT_COMPOUND_BATCH_SIZE,
        None,
    )
    .await
    .expect("Failed to train compound index");

    let cache = get_cache(use_cache, "compound_3col");
    let details = prost_types::Any::from_msg(&pb::CompoundBTreeIndexDetails {
        column_names,
        num_columns: 3,
    })
    .unwrap();

    CompoundBTreeIndexPlugin
        .load_index(store, &details, None, &cache)
        .await
        .expect("Failed to load compound index")
}

/// Setup function for 2-column int index
fn setup_compound_2col_int_index(
    rt: &tokio::runtime::Runtime,
    use_cache: bool,
) -> Arc<dyn ScalarIndex> {
    let static_ref = if use_cache {
        &COMPOUND_2COL_INT_INDEX_CACHED
    } else {
        &COMPOUND_2COL_INT_INDEX_NO_CACHE
    };

    static_ref
        .get_or_init(|| {
            rt.block_on(async {
                let tempdir = tempfile::tempdir().unwrap();
                let store = Arc::new(LanceIndexStore::new(
                    Arc::new(ObjectStore::local()),
                    Path::from_filesystem_path(tempdir.path()).unwrap(),
                    get_cache(use_cache, "compound_2col_int"),
                ));
                let index = create_compound_2col_int_index(store, use_cache).await;
                let _ = tempdir.keep();
                index
            })
        })
        .clone()
}

/// Setup function for 2-column tenant index
fn setup_compound_2col_tenant_index(
    rt: &tokio::runtime::Runtime,
    use_cache: bool,
) -> Arc<dyn ScalarIndex> {
    let static_ref = if use_cache {
        &COMPOUND_2COL_TENANT_INDEX_CACHED
    } else {
        &COMPOUND_2COL_TENANT_INDEX_NO_CACHE
    };

    static_ref
        .get_or_init(|| {
            rt.block_on(async {
                let tempdir = tempfile::tempdir().unwrap();
                let store = Arc::new(LanceIndexStore::new(
                    Arc::new(ObjectStore::local()),
                    Path::from_filesystem_path(tempdir.path()).unwrap(),
                    get_cache(use_cache, "compound_2col_tenant"),
                ));
                let index = create_compound_2col_tenant_index(store, use_cache).await;
                let _ = tempdir.keep();
                index
            })
        })
        .clone()
}

/// Setup function for 3-column index
fn setup_compound_3col_index(
    rt: &tokio::runtime::Runtime,
    use_cache: bool,
) -> Arc<dyn ScalarIndex> {
    let static_ref = if use_cache {
        &COMPOUND_3COL_INDEX_CACHED
    } else {
        &COMPOUND_3COL_INDEX_NO_CACHE
    };

    static_ref
        .get_or_init(|| {
            rt.block_on(async {
                let tempdir = tempfile::tempdir().unwrap();
                let store = Arc::new(LanceIndexStore::new(
                    Arc::new(ObjectStore::local()),
                    Path::from_filesystem_path(tempdir.path()).unwrap(),
                    get_cache(use_cache, "compound_3col"),
                ));
                let index = create_compound_3col_index(store, use_cache).await;
                let _ = tempdir.keep();
                index
            })
        })
        .clone()
}

fn bench_full_key_lookup(c: &mut Criterion) {
    let rt = get_runtime();

    // Test values - middle of each range
    let int_key1 = 500i64; // Middle of 0..1000
    let int_key2 = 500i64; // Middle of 0..1000
    let tenant = format!("tenant_{:03}", COMPOUND_TENANT_COUNT / 2);
    let timestamp = (COMPOUND_TIMESTAMPS_PER_TENANT / 2) as i64;
    let status = format!("status_{}", COMPOUND_STATUS_COUNT / 2);
    let timestamp_3col = (COMPOUND_TIMESTAMPS_PER_STATUS / 2) as i64;

    let mut group = c.benchmark_group("compound_full_key_lookup");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));

    for use_cache in [false, true] {
        let cache_label = if use_cache { "cached" } else { "no_cache" };

        // 2-column int index - full key lookup via PrefixLookup with all columns
        // (PrefixLookup with all columns is equivalent to FullKeyLookup)
        group.bench_function(BenchmarkId::new("2col_int", cache_label), |b| {
            let index = setup_compound_2col_int_index(rt, use_cache);
            b.to_async(rt).iter(|| {
                let index = index.clone();
                async move {
                    let query = CompoundSargableQuery::PrefixLookup {
                        prefix: vec![
                            ScalarValue::Int64(Some(int_key1)),
                            ScalarValue::Int64(Some(int_key2)),
                        ],
                        range: None,
                    };
                    black_box(
                        index
                            .search(&query as &dyn AnyQuery, &NoOpMetricsCollector)
                            .await
                            .unwrap(),
                    );
                }
            })
        });

        // 2-column tenant index - full key lookup
        group.bench_function(BenchmarkId::new("2col_tenant", cache_label), |b| {
            let index = setup_compound_2col_tenant_index(rt, use_cache);
            let tenant = tenant.clone();
            b.to_async(rt).iter(|| {
                let index = index.clone();
                let tenant = tenant.clone();
                async move {
                    let query = CompoundSargableQuery::PrefixLookup {
                        prefix: vec![
                            ScalarValue::Utf8(Some(tenant)),
                            ScalarValue::Int64(Some(timestamp)),
                        ],
                        range: None,
                    };
                    black_box(
                        index
                            .search(&query as &dyn AnyQuery, &NoOpMetricsCollector)
                            .await
                            .unwrap(),
                    );
                }
            })
        });

        // 3-column index - full key lookup
        group.bench_function(BenchmarkId::new("3col", cache_label), |b| {
            let index = setup_compound_3col_index(rt, use_cache);
            let tenant = tenant.clone();
            let status = status.clone();
            b.to_async(rt).iter(|| {
                let index = index.clone();
                let tenant = tenant.clone();
                let status = status.clone();
                async move {
                    let query = CompoundSargableQuery::PrefixLookup {
                        prefix: vec![
                            ScalarValue::Utf8(Some(tenant)),
                            ScalarValue::Utf8(Some(status)),
                            ScalarValue::Int64(Some(timestamp_3col)),
                        ],
                        range: None,
                    };
                    black_box(
                        index
                            .search(&query as &dyn AnyQuery, &NoOpMetricsCollector)
                            .await
                            .unwrap(),
                    );
                }
            })
        });
    }

    group.finish();
}

fn bench_prefix_lookup(c: &mut Criterion) {
    let rt = get_runtime();

    let int_key1 = 500i64;
    let tenant = format!("tenant_{:03}", COMPOUND_TENANT_COUNT / 2);
    let status = format!("status_{}", COMPOUND_STATUS_COUNT / 2);

    let mut group = c.benchmark_group("compound_prefix_lookup");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));

    for use_cache in [false, true] {
        let cache_label = if use_cache { "cached" } else { "no_cache" };

        // 2-column int index - single column prefix
        group.bench_function(BenchmarkId::new("2col_int_single_prefix", cache_label), |b| {
            let index = setup_compound_2col_int_index(rt, use_cache);
            b.to_async(rt).iter(|| {
                let index = index.clone();
                async move {
                    let query = CompoundSargableQuery::PrefixLookup {
                        prefix: vec![ScalarValue::Int64(Some(int_key1))],
                        range: None,
                    };
                    black_box(
                        index
                            .search(&query as &dyn AnyQuery, &NoOpMetricsCollector)
                            .await
                            .unwrap(),
                    );
                }
            })
        });

        // 2-column tenant index - single column prefix (tenant only)
        group.bench_function(
            BenchmarkId::new("2col_tenant_single_prefix", cache_label),
            |b| {
                let index = setup_compound_2col_tenant_index(rt, use_cache);
                let tenant = tenant.clone();
                b.to_async(rt).iter(|| {
                    let index = index.clone();
                    let tenant = tenant.clone();
                    async move {
                        let query = CompoundSargableQuery::PrefixLookup {
                            prefix: vec![ScalarValue::Utf8(Some(tenant))],
                            range: None,
                        };
                        black_box(
                            index
                                .search(&query as &dyn AnyQuery, &NoOpMetricsCollector)
                                .await
                                .unwrap(),
                        );
                    }
                })
            },
        );

        // 3-column index - single column prefix (tenant only)
        group.bench_function(BenchmarkId::new("3col_single_prefix", cache_label), |b| {
            let index = setup_compound_3col_index(rt, use_cache);
            let tenant = tenant.clone();
            b.to_async(rt).iter(|| {
                let index = index.clone();
                let tenant = tenant.clone();
                async move {
                    let query = CompoundSargableQuery::PrefixLookup {
                        prefix: vec![ScalarValue::Utf8(Some(tenant))],
                        range: None,
                    };
                    black_box(
                        index
                            .search(&query as &dyn AnyQuery, &NoOpMetricsCollector)
                            .await
                            .unwrap(),
                    );
                }
            })
        });

        // 3-column index - two column prefix (tenant + status)
        group.bench_function(BenchmarkId::new("3col_two_prefix", cache_label), |b| {
            let index = setup_compound_3col_index(rt, use_cache);
            let tenant = tenant.clone();
            let status = status.clone();
            b.to_async(rt).iter(|| {
                let index = index.clone();
                let tenant = tenant.clone();
                let status = status.clone();
                async move {
                    let query = CompoundSargableQuery::PrefixLookup {
                        prefix: vec![
                            ScalarValue::Utf8(Some(tenant)),
                            ScalarValue::Utf8(Some(status)),
                        ],
                        range: None,
                    };
                    black_box(
                        index
                            .search(&query as &dyn AnyQuery, &NoOpMetricsCollector)
                            .await
                            .unwrap(),
                    );
                }
            })
        });
    }

    group.finish();
}

fn bench_prefix_range(c: &mut Criterion, selectivity: Selectivity) {
    let rt = get_runtime();

    let group_name = format!("compound_prefix_range_{}", selectivity.name());
    let mut group = c.benchmark_group(&group_name);
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(10));

    let pct = selectivity.percentage();

    // For 2-column int index: key1 = 500, range on key2
    let int_key1 = 500i64;
    let key2_range_size = (1000.0 * pct) as i64;
    let key2_start = 500 - (key2_range_size / 2);

    // For 2-column tenant index: tenant = mid, range on timestamp
    let tenant = format!("tenant_{:03}", COMPOUND_TENANT_COUNT / 2);
    let ts_range_size = (COMPOUND_TIMESTAMPS_PER_TENANT as f64 * pct) as i64;
    let ts_start = (COMPOUND_TIMESTAMPS_PER_TENANT / 2) as i64 - (ts_range_size / 2);

    // For 3-column index: tenant + status = mid, range on timestamp
    let status = format!("status_{}", COMPOUND_STATUS_COUNT / 2);
    let ts_3col_range_size = (COMPOUND_TIMESTAMPS_PER_STATUS as f64 * pct) as i64;
    let ts_3col_start = (COMPOUND_TIMESTAMPS_PER_STATUS / 2) as i64 - (ts_3col_range_size / 2);

    for use_cache in [false, true] {
        let cache_label = if use_cache { "cached" } else { "no_cache" };

        // 2-column int index - prefix + range
        group.bench_function(BenchmarkId::new("2col_int", cache_label), |b| {
            let index = setup_compound_2col_int_index(rt, use_cache);
            b.to_async(rt).iter(|| {
                let index = index.clone();
                async move {
                    let query = CompoundSargableQuery::PrefixLookup {
                        prefix: vec![ScalarValue::Int64(Some(int_key1))],
                        range: Some((
                            Bound::Included(ScalarValue::Int64(Some(key2_start))),
                            Bound::Unbounded,
                        )),
                    };
                    black_box(
                        index
                            .search(&query as &dyn AnyQuery, &NoOpMetricsCollector)
                            .await
                            .unwrap(),
                    );
                }
            })
        });

        // 2-column tenant index - prefix + range
        group.bench_function(BenchmarkId::new("2col_tenant", cache_label), |b| {
            let index = setup_compound_2col_tenant_index(rt, use_cache);
            let tenant = tenant.clone();
            b.to_async(rt).iter(|| {
                let index = index.clone();
                let tenant = tenant.clone();
                async move {
                    let query = CompoundSargableQuery::PrefixLookup {
                        prefix: vec![ScalarValue::Utf8(Some(tenant))],
                        range: Some((
                            Bound::Included(ScalarValue::Int64(Some(ts_start))),
                            Bound::Unbounded,
                        )),
                    };
                    black_box(
                        index
                            .search(&query as &dyn AnyQuery, &NoOpMetricsCollector)
                            .await
                            .unwrap(),
                    );
                }
            })
        });

        // 3-column index - prefix + range
        group.bench_function(BenchmarkId::new("3col", cache_label), |b| {
            let index = setup_compound_3col_index(rt, use_cache);
            let tenant = tenant.clone();
            let status = status.clone();
            b.to_async(rt).iter(|| {
                let index = index.clone();
                let tenant = tenant.clone();
                let status = status.clone();
                async move {
                    let query = CompoundSargableQuery::PrefixLookup {
                        prefix: vec![
                            ScalarValue::Utf8(Some(tenant)),
                            ScalarValue::Utf8(Some(status)),
                        ],
                        range: Some((
                            Bound::Included(ScalarValue::Int64(Some(ts_3col_start))),
                            Bound::Unbounded,
                        )),
                    };
                    black_box(
                        index
                            .search(&query as &dyn AnyQuery, &NoOpMetricsCollector)
                            .await
                            .unwrap(),
                    );
                }
            })
        });
    }

    group.finish();
}

fn bench_compound_btree(c: &mut Criterion) {
    // Run full key lookup benchmarks
    bench_full_key_lookup(c);

    // Run prefix lookup benchmarks
    bench_prefix_lookup(c);

    // Run prefix + range benchmarks with different selectivities
    bench_prefix_range(c, Selectivity::Few);
    bench_prefix_range(c, Selectivity::Many);
    bench_prefix_range(c, Selectivity::Most);
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_compound_btree);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10);
    targets = bench_compound_btree);

criterion_main!(benches);
