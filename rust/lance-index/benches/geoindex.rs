// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmarks for geo index vs brute force scanning

use arrow_schema::{DataType, Field};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use lance_core::cache::LanceCache;
use lance_core::utils::tempfile::TempObjDir;
use lance_io::object_store::ObjectStore;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

use lance_index::metrics::{MetricsCollector, NoOpMetricsCollector};
use lance_index::scalar::geo::geoindex::{GeoIndex, GeoIndexBuilder, GeoIndexBuilderParams};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{GeoQuery, ScalarIndex};
use std::sync::atomic::{AtomicUsize, Ordering};

struct LeafCounter {
    leaves_visited: AtomicUsize,
}

impl LeafCounter {
    fn new() -> Self {
        Self {
            leaves_visited: AtomicUsize::new(0),
        }
    }

    fn get_count(&self) -> usize {
        self.leaves_visited.load(Ordering::Relaxed)
    }

    fn reset(&self) {
        self.leaves_visited.store(0, Ordering::Relaxed);
    }
}

impl MetricsCollector for LeafCounter {
    fn record_parts_loaded(&self, num_parts: usize) {
        self.leaves_visited.fetch_add(num_parts, Ordering::Relaxed);
    }

    fn record_index_loads(&self, _num_indexes: usize) {}
    fn record_comparisons(&self, _num_comparisons: usize) {}
}

fn create_test_points(num_points: usize) -> Vec<(f64, f64, u64)> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut points = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let x = rng.random_range(0.0..1000.0);
        let y = rng.random_range(0.0..1000.0);
        points.push((x, y, i as u64));
    }

    points
}

async fn create_geo_index(
    points: Vec<(f64, f64, u64)>,
) -> (Arc<GeoIndex>, Arc<LanceIndexStore>, TempObjDir) {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));

    let params = GeoIndexBuilderParams {
        max_points_per_leaf: 1000,
        batches_per_file: 10,
    };

    let mut builder =
        GeoIndexBuilder::try_new(params, DataType::Struct(Vec::<Field>::new().into())).unwrap();

    builder.points = points;
    builder.write_index(store.as_ref()).await.unwrap();

    let index = GeoIndex::load(store.clone(), None, &LanceCache::no_cache())
        .await
        .unwrap();

    (index, store, tmpdir)
}

fn bench_geo_index_intersects_pruning(c: &mut Criterion) {
    let mut group = c.benchmark_group("geo_intersects_pruning_vs_scan_all");
    group.sample_size(10); // Reduce samples for faster benchmarks
    group.measurement_time(std::time::Duration::from_secs(60));

    // Compares BKD tree spatial pruning vs scanning all leaves for intersects queries
    // Both approaches do lazy loading - we're measuring pruning efficiency

    // Test with different dataset sizes
    for num_points in [10_000, 100_000, 1_000_000] {
        let points = create_test_points(num_points);

        // Create index (do this once outside the benchmark)
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (index, _store, _tmpdir) = rt.block_on(create_geo_index(points.clone()));

        // Generate random queries
        let mut rng = StdRng::seed_from_u64(42);
        let queries: Vec<[f64; 4]> = (0..10)
            .map(|_| {
                let width = rng.random_range(10.0..50.0);
                let height = rng.random_range(10.0..50.0);
                let min_x = rng.random_range(0.0..(1000.0 - width));
                let min_y = rng.random_range(0.0..(1000.0 - height));
                [min_x, min_y, min_x + width, min_y + height]
            })
            .collect();

        // Benchmark: scan all leaves (no spatial pruning)
        let scan_all_counter = Arc::new(LeafCounter::new());
        group.bench_with_input(
            BenchmarkId::new("scan_all_leaves", num_points),
            &(&index, &queries, scan_all_counter.clone()),
            |b, (index, queries, counter)| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.iter(|| {
                    counter.reset();
                    rt.block_on(async {
                        for query_bbox in queries.iter() {
                            let _result = index
                                .search_all_leaves(*query_bbox, counter.as_ref())
                                .await
                                .unwrap();
                        }
                    });
                });
            },
        );

        // Benchmark: BKD tree with spatial pruning
        let pruned_counter = Arc::new(LeafCounter::new());
        group.bench_with_input(
            BenchmarkId::new("with_pruning", num_points),
            &(&index, &queries, pruned_counter.clone()),
            |b, (index, queries, counter)| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.iter(|| {
                    counter.reset();
                    rt.block_on(async {
                        for query_bbox in queries.iter() {
                            let query = GeoQuery::Intersects(
                                query_bbox[0],
                                query_bbox[1],
                                query_bbox[2],
                                query_bbox[3],
                            );
                            let _result = index.search(&query, counter.as_ref()).await.unwrap();
                        }
                    });
                });
            },
        );

        // Print statistics with speedup estimate
        let scan_avg = scan_all_counter.get_count() as f64 / queries.len() as f64;
        let pruned_avg = pruned_counter.get_count() as f64 / queries.len() as f64;

        // Quick timing for speedup calculation
        let rt = tokio::runtime::Runtime::new().unwrap();
        let scan_start = std::time::Instant::now();
        for query_bbox in queries.iter() {
            rt.block_on(async {
                let _ = index
                    .search_all_leaves(*query_bbox, &NoOpMetricsCollector)
                    .await;
            });
        }
        let scan_time = scan_start.elapsed().as_secs_f64() / queries.len() as f64;

        let prune_start = std::time::Instant::now();
        for query_bbox in queries.iter() {
            rt.block_on(async {
                let query = GeoQuery::Intersects(
                    query_bbox[0],
                    query_bbox[1],
                    query_bbox[2],
                    query_bbox[3],
                );
                let _ = index.search(&query, &NoOpMetricsCollector).await;
            });
        }
        let prune_time = prune_start.elapsed().as_secs_f64() / queries.len() as f64;

        println!(
            "\n  {} points: {} leaves | scan_all: {:.1} leaves/query ({:.1}ms) | with_pruning: {:.1} leaves/query ({:.1}ms) | {:.1}x faster, {:.1}% reduction",
            num_points,
            index.num_leaves(),
            scan_avg,
            scan_time * 1000.0,
            pruned_avg,
            prune_time * 1000.0,
            scan_time / prune_time,
            100.0 * (1.0 - pruned_avg / scan_avg)
        );
    }

    group.finish();
}

fn bench_geo_intersects_query_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("geo_intersects_query_size");
    group.sample_size(10);

    // Fixed dataset size, varying intersects query box sizes
    let points = create_test_points(1_000_000);
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (index, _store, _tmpdir) = rt.block_on(create_geo_index(points));

    let mut rng = StdRng::seed_from_u64(42);

    // Test different query sizes
    for query_size in [1.0, 10.0, 50.0, 100.0, 200.0] {
        let queries: Vec<[f64; 4]> = (0..10)
            .map(|_| {
                let min_x = rng.random_range(0.0..(1000.0 - query_size));
                let min_y = rng.random_range(0.0..(1000.0 - query_size));
                [min_x, min_y, min_x + query_size, min_y + query_size]
            })
            .collect();

        let leaf_counter = Arc::new(LeafCounter::new());
        group.bench_with_input(
            BenchmarkId::new("query_size", format!("{}x{}", query_size, query_size)),
            &(&index, &queries, leaf_counter.clone()),
            |b, (index, queries, counter)| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.iter(|| {
                    counter.reset();
                    rt.block_on(async {
                        for query_bbox in queries.iter() {
                            let query = GeoQuery::Intersects(
                                query_bbox[0],
                                query_bbox[1],
                                query_bbox[2],
                                query_bbox[3],
                            );
                            let _result = index.search(&query, counter.as_ref()).await.unwrap();
                        }
                    });
                });
            },
        );

        let avg_leaves = leaf_counter.get_count() as f64 / queries.len() as f64;
        println!(
            "  Query {}x{}: avg {:.1} leaves visited (out of {} total, {:.1}% selectivity)",
            query_size,
            query_size,
            avg_leaves,
            index.num_leaves(),
            100.0 * avg_leaves / index.num_leaves() as f64
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_geo_index_intersects_pruning,
    bench_geo_intersects_query_size
);
criterion_main!(benches);
