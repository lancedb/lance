// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Direct object_store benchmark to isolate S3 vs S3 Express performance.
//!
//! This benchmark reads actual manifest files from a Lance dataset and measures:
//! - PUT latency (write to _validate directory)
//! - LIST first page latency (list with pagination)
//! - LIST all latency (list entire directory)
//! - GET latency (read back)
//!
//! ## Running against S3 Express
//!
//! ```bash
//! export AWS_REGION=us-east-1
//! export DATASET_URI=s3://jack-lancedb-devland-az6--use1-az6--x-s3/bench/manifest_commit_XXXX
//! cargo bench --bench object_store_bench
//! ```
//!
//! ## Running against S3 Standard
//!
//! ```bash
//! export AWS_REGION=us-east-1
//! export DATASET_URI=s3://jack-lancedb-devland-us-east-1/bench/manifest_commit_XXXX
//! cargo bench --bench object_store_bench
//! ```
//!
//! ## Configuration
//!
//! - `DATASET_URI`: URI to Lance dataset with _versions directory (required)
//! - `AWS_REGION`: AWS region (required for S3)
//! - `MAX_MANIFESTS`: Maximum number of manifests to test (default: all)

#![allow(clippy::print_stdout)]

use bytes::Bytes;
use criterion::{criterion_group, criterion_main, Criterion};
use futures::{StreamExt, TryStreamExt};
use object_store::aws::AmazonS3Builder;
use object_store::path::Path;
use object_store::{ObjectMeta, ObjectStore};
use std::sync::Arc;
use std::time::Instant;
use tokio::runtime::Runtime;
use url::Url;

fn get_max_manifests() -> Option<usize> {
    std::env::var("MAX_MANIFESTS")
        .ok()
        .and_then(|s| s.parse().ok())
}

fn get_storage_label(uri: &str) -> &'static str {
    if uri.contains("--x-s3") {
        "s3express"
    } else if uri.starts_with("s3://") {
        "s3"
    } else if uri.starts_with("gs://") {
        "gcs"
    } else {
        "local"
    }
}

async fn create_object_store(uri: &str) -> Arc<dyn ObjectStore> {
    let url = Url::parse(uri).expect("Invalid URL");
    let bucket = url.host_str().expect("No bucket in URL");

    let mut builder = AmazonS3Builder::from_env().with_bucket_name(bucket);

    if uri.contains("--x-s3") {
        builder = builder.with_s3_express(true);
    }

    Arc::new(builder.build().expect("Failed to build S3 client"))
}

async fn list_manifests(store: &dyn ObjectStore, versions_path: &Path) -> Vec<ObjectMeta> {
    let stream = store.list(Some(versions_path));
    let mut objects: Vec<ObjectMeta> = stream
        .try_collect()
        .await
        .expect("Failed to list manifests");

    // Filter for .manifest files and sort by version number
    objects.retain(|o| o.location.as_ref().ends_with(".manifest"));
    objects.sort_by(|a, b| {
        let version_a = extract_version(&a.location);
        let version_b = extract_version(&b.location);
        version_a.cmp(&version_b)
    });

    objects
}

fn extract_version(path: &Path) -> u64 {
    let filename = path.filename().unwrap_or("");
    filename
        .strip_suffix(".manifest")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

async fn list_first_page(store: &dyn ObjectStore, path: &Path) -> usize {
    // List with a small limit to simulate first page
    let stream = store.list(Some(path));
    let objects: Vec<ObjectMeta> = stream.try_collect().await.expect("Failed to list");
    // We can't easily limit in object_store, so we just count
    // The latency measurement captures the full list time
    objects.len()
}

async fn list_all(store: &dyn ObjectStore, path: &Path) -> usize {
    let stream = store.list(Some(path));
    let objects: Vec<ObjectMeta> = stream.try_collect().await.expect("Failed to list");
    objects.len()
}

fn bench_direct_s3(c: &mut Criterion) {
    let runtime = Runtime::new().expect("Failed to create runtime");

    let dataset_uri = std::env::var("DATASET_URI").expect(
        "DATASET_URI environment variable is required.\n\
         Example: export DATASET_URI=s3://bucket/bench/manifest_commit_XXXX",
    );

    let storage_label = get_storage_label(&dataset_uri);
    let max_manifests = get_max_manifests();

    let url = Url::parse(&dataset_uri).expect("Invalid URL");
    let base_path = url.path().trim_start_matches('/');

    println!("=== Direct S3 Operations Benchmark ===");
    println!("Dataset: {}", dataset_uri);
    println!("Storage: {}", storage_label);
    if let Some(max) = max_manifests {
        println!("Max manifests: {}", max);
    }
    println!();

    let store = runtime.block_on(create_object_store(&dataset_uri));

    // List all manifests from _versions directory
    let versions_path = Path::from(format!("{}/_versions", base_path));
    let manifests = runtime.block_on(list_manifests(store.as_ref(), &versions_path));

    let manifest_count = if let Some(max) = max_manifests {
        manifests.len().min(max)
    } else {
        manifests.len()
    };

    println!(
        "Found {} manifests, testing {}",
        manifests.len(),
        manifest_count
    );
    println!();

    // Prepare validate directory path
    let validate_path = Path::from(format!("{}/_validate", base_path));

    // Clean up any existing _validate directory
    runtime.block_on(async {
        let stream = store.list(Some(&validate_path));
        if let Ok(objects) = stream.try_collect::<Vec<_>>().await {
            for obj in objects {
                let _ = store.delete(&obj.location).await;
            }
        }
    });

    println!("version,size_bytes,put_ms,list_first_ms,list_all_ms,get_ms,validate_count");

    let mut put_times = Vec::with_capacity(manifest_count);
    let mut list_first_times = Vec::with_capacity(manifest_count);
    let mut list_all_times = Vec::with_capacity(manifest_count);
    let mut get_times = Vec::with_capacity(manifest_count);
    let mut sizes = Vec::with_capacity(manifest_count);

    for (i, manifest) in manifests.iter().take(manifest_count).enumerate() {
        let version = extract_version(&manifest.location);
        let size = manifest.size;

        // Read the manifest
        let data: Bytes = runtime.block_on(async {
            let result = store
                .get(&manifest.location)
                .await
                .expect("Failed to read manifest");
            result.bytes().await.expect("Failed to get bytes")
        });

        // Write to _validate directory - measure PUT
        let validate_file = Path::from(format!("{}/_validate/{}.manifest", base_path, version));
        let put_time = runtime.block_on(async {
            let start = Instant::now();
            store
                .put(&validate_file, data.clone().into())
                .await
                .expect("PUT failed");
            start.elapsed()
        });

        // List _validate with first page simulation - measure LIST first page
        // Since object_store doesn't have built-in pagination limit, we measure full list
        // but in practice this represents the first page latency for small counts
        let (list_first_time, _) = runtime.block_on(async {
            let start = Instant::now();
            let count = list_first_page(store.as_ref(), &validate_path).await;
            (start.elapsed(), count)
        });

        // List _validate completely - measure LIST all
        let (list_all_time, validate_count) = runtime.block_on(async {
            let start = Instant::now();
            let count = list_all(store.as_ref(), &validate_path).await;
            (start.elapsed(), count)
        });

        // Read back from _validate - measure GET
        let get_time = runtime.block_on(async {
            let start = Instant::now();
            let result = store.get(&validate_file).await.expect("GET failed");
            let _bytes = result.bytes().await.expect("Failed to get bytes");
            start.elapsed()
        });

        put_times.push(put_time);
        list_first_times.push(list_first_time);
        list_all_times.push(list_all_time);
        get_times.push(get_time);
        sizes.push(size);

        println!(
            "{},{},{:.2},{:.2},{:.2},{:.2},{}",
            version,
            size,
            put_time.as_secs_f64() * 1000.0,
            list_first_time.as_secs_f64() * 1000.0,
            list_all_time.as_secs_f64() * 1000.0,
            get_time.as_secs_f64() * 1000.0,
            validate_count
        );

        // Progress indicator every 500 iterations
        if (i + 1) % 500 == 0 {
            eprintln!("Progress: {}/{}", i + 1, manifest_count);
        }
    }

    // Summary statistics
    println!();
    println!("=== Summary Statistics ===");

    let avg_put = put_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / manifest_count as f64;
    let avg_list_first = list_first_times
        .iter()
        .map(|d| d.as_secs_f64())
        .sum::<f64>()
        / manifest_count as f64;
    let avg_list_all =
        list_all_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / manifest_count as f64;
    let avg_get = get_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / manifest_count as f64;

    let min_put = put_times.iter().min().unwrap().as_secs_f64();
    let min_list_first = list_first_times.iter().min().unwrap().as_secs_f64();
    let min_list_all = list_all_times.iter().min().unwrap().as_secs_f64();
    let min_get = get_times.iter().min().unwrap().as_secs_f64();

    let max_put = put_times.iter().max().unwrap().as_secs_f64();
    let max_list_first = list_first_times.iter().max().unwrap().as_secs_f64();
    let max_list_all = list_all_times.iter().max().unwrap().as_secs_f64();
    let max_get = get_times.iter().max().unwrap().as_secs_f64();

    println!(
        "PUT:        avg={:.2}ms, min={:.2}ms, max={:.2}ms",
        avg_put * 1000.0,
        min_put * 1000.0,
        max_put * 1000.0
    );
    println!(
        "LIST first: avg={:.2}ms, min={:.2}ms, max={:.2}ms",
        avg_list_first * 1000.0,
        min_list_first * 1000.0,
        max_list_first * 1000.0
    );
    println!(
        "LIST all:   avg={:.2}ms, min={:.2}ms, max={:.2}ms",
        avg_list_all * 1000.0,
        min_list_all * 1000.0,
        max_list_all * 1000.0
    );
    println!(
        "GET:        avg={:.2}ms, min={:.2}ms, max={:.2}ms",
        avg_get * 1000.0,
        min_get * 1000.0,
        max_get * 1000.0
    );

    // Linear regression for LIST latency vs file count
    let n = manifest_count as f64;
    let file_counts: Vec<f64> = (1..=manifest_count).map(|x| x as f64).collect();
    let list_all_ms: Vec<f64> = list_all_times
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .collect();

    let sum_x: f64 = file_counts.iter().sum();
    let sum_y: f64 = list_all_ms.iter().sum();
    let sum_xx: f64 = file_counts.iter().map(|x| x * x).sum();
    let sum_xy: f64 = file_counts
        .iter()
        .zip(list_all_ms.iter())
        .map(|(x, y)| x * y)
        .sum();

    let list_slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let list_intercept = (sum_y - list_slope * sum_x) / n;

    println!();
    println!("=== Linear Regression (LIST all vs file count) ===");
    println!(
        "LIST all = {:.2}ms + {:.4}ms × files",
        list_intercept, list_slope
    );
    println!("Per-file overhead: {:.4}ms", list_slope);

    // First 10 vs last 10 comparison
    let first_10_put = put_times
        .iter()
        .take(10)
        .map(|d| d.as_secs_f64())
        .sum::<f64>()
        / 10.0;
    let last_10_put = put_times
        .iter()
        .rev()
        .take(10)
        .map(|d| d.as_secs_f64())
        .sum::<f64>()
        / 10.0;
    let first_10_list = list_all_times
        .iter()
        .take(10)
        .map(|d| d.as_secs_f64())
        .sum::<f64>()
        / 10.0;
    let last_10_list = list_all_times
        .iter()
        .rev()
        .take(10)
        .map(|d| d.as_secs_f64())
        .sum::<f64>()
        / 10.0;
    let first_10_get = get_times
        .iter()
        .take(10)
        .map(|d| d.as_secs_f64())
        .sum::<f64>()
        / 10.0;
    let last_10_get = get_times
        .iter()
        .rev()
        .take(10)
        .map(|d| d.as_secs_f64())
        .sum::<f64>()
        / 10.0;

    println!();
    println!("=== First 10 vs Last 10 ===");
    println!(
        "PUT:      first={:.2}ms, last={:.2}ms, ratio={:.2}x",
        first_10_put * 1000.0,
        last_10_put * 1000.0,
        last_10_put / first_10_put
    );
    println!(
        "LIST all: first={:.2}ms, last={:.2}ms, ratio={:.2}x",
        first_10_list * 1000.0,
        last_10_list * 1000.0,
        last_10_list / first_10_list
    );
    println!(
        "GET:      first={:.2}ms, last={:.2}ms, ratio={:.2}x",
        first_10_get * 1000.0,
        last_10_get * 1000.0,
        last_10_get / first_10_get
    );

    // Cleanup _validate directory
    println!();
    println!("Cleaning up _validate directory...");
    runtime.block_on(async {
        let stream = store.list(Some(&validate_path));
        if let Ok(objects) = stream.try_collect::<Vec<_>>().await {
            for obj in objects {
                let _ = store.delete(&obj.location).await;
            }
        }
    });
    println!("Cleanup complete.");

    // Criterion benchmarks for tracking
    let mut group = c.benchmark_group(format!("direct_s3_{}", storage_label));

    group.bench_function("avg_put_ms", |b| b.iter(|| avg_put * 1000.0));
    group.bench_function("avg_list_all_ms", |b| b.iter(|| avg_list_all * 1000.0));
    group.bench_function("avg_get_ms", |b| b.iter(|| avg_get * 1000.0));
    group.bench_function("list_slope_ms_per_file", |b| b.iter(|| list_slope));

    group.finish();
}

criterion_group!(benches, bench_direct_s3);
criterion_main!(benches);
