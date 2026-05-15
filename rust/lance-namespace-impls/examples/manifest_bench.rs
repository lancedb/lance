// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark for directory manifest read/write latency and throughput.
//!
//! Usage:
//!   cargo run -p lance-namespace-impls --release --example manifest_bench -- \
//!     --root /tmp/manifest_bench \
//!     --concurrency 1,2,5,10,20,50,100 \
//!     --operations 500 \
//!     --warmup 50
//!
//! Output: JSON lines per (operation, concurrency) pair.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use lance_namespace::LanceNamespace;
use lance_namespace::models::{
    CreateNamespaceRequest, CreateTableRequest, DescribeTableRequest, ListNamespacesRequest,
    ListTablesRequest,
};
use lance_namespace_impls::DirectoryNamespaceBuilder;
use serde::Serialize;
use tokio::sync::Barrier;

#[derive(Clone, Copy, Debug)]
struct BenchConfig {
    operations: usize,
    warmup: usize,
    concurrency: usize,
}

#[derive(Serialize)]
struct BenchResult {
    variant: String,
    operation: String,
    concurrency: usize,
    total_operations: usize,
    total_duration_ms: f64,
    throughput_ops_per_sec: f64,
    avg_latency_ms: f64,
    p50_latency_ms: f64,
    p90_latency_ms: f64,
    p99_latency_ms: f64,
    min_latency_ms: f64,
    max_latency_ms: f64,
    errors: usize,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn compute_result(
    variant: &str,
    operation: &str,
    concurrency: usize,
    wall_duration: Duration,
    mut latencies: Vec<f64>,
    errors: usize,
) -> BenchResult {
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total = latencies.len();
    let total_ms = wall_duration.as_secs_f64() * 1000.0;
    let throughput = if total_ms > 0.0 {
        total as f64 / (total_ms / 1000.0)
    } else {
        0.0
    };
    BenchResult {
        variant: variant.to_string(),
        operation: operation.to_string(),
        concurrency,
        total_operations: total,
        total_duration_ms: total_ms,
        throughput_ops_per_sec: throughput,
        avg_latency_ms: if total > 0 {
            latencies.iter().sum::<f64>() / total as f64
        } else {
            0.0
        },
        p50_latency_ms: percentile(&latencies, 0.50),
        p90_latency_ms: percentile(&latencies, 0.90),
        p99_latency_ms: percentile(&latencies, 0.99),
        min_latency_ms: latencies.first().copied().unwrap_or(0.0),
        max_latency_ms: latencies.last().copied().unwrap_or(0.0),
        errors,
    }
}

fn create_test_ipc_data() -> Vec<u8> {
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::StreamWriter;
    use arrow::record_batch::RecordBatch;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }
    buffer
}

async fn setup_namespace(root: &str, inline_optimization: bool) -> Arc<dyn LanceNamespace> {
    let ns = DirectoryNamespaceBuilder::new(root)
        .dir_listing_enabled(false)
        .inline_optimization_enabled(inline_optimization)
        .build()
        .await
        .expect("Failed to build namespace");
    Arc::new(ns)
}

/// Seed the manifest with some initial data for read benchmarks.
async fn seed_data(ns: &dyn LanceNamespace, num_namespaces: usize, num_tables: usize) {
    let ipc_data = create_test_ipc_data();
    for i in 0..num_namespaces {
        let mut req = CreateNamespaceRequest::new();
        req.id = Some(vec![format!("ns_{}", i)]);
        let _ = ns.create_namespace(req).await;
    }
    for i in 0..num_tables {
        let mut req = CreateTableRequest::new();
        req.id = Some(vec![format!("table_{}", i)]);
        let _ = ns.create_table(req, Bytes::from(ipc_data.clone())).await;
    }
}

async fn bench_write_create_namespace(
    ns: Arc<dyn LanceNamespace>,
    config: BenchConfig,
    variant: &str,
) -> BenchResult {
    let counter = Arc::new(AtomicU64::new(0));
    let barrier = Arc::new(Barrier::new(config.concurrency));

    // Warmup
    for i in 0..config.warmup {
        let mut req = CreateNamespaceRequest::new();
        req.id = Some(vec![format!("warmup_ns_{}", i)]);
        let _ = ns.create_namespace(req).await;
    }

    let wall_start = Instant::now();
    let handles: Vec<_> = (0..config.concurrency)
        .map(|_| {
            let ns = ns.clone();
            let counter = counter.clone();
            let barrier = barrier.clone();
            let ops_per_worker = config.operations / config.concurrency;
            tokio::spawn(async move {
                barrier.wait().await;
                let mut latencies = Vec::with_capacity(ops_per_worker);
                let mut errors = 0usize;
                for _ in 0..ops_per_worker {
                    let id = counter.fetch_add(1, Ordering::Relaxed);
                    let mut req = CreateNamespaceRequest::new();
                    req.id = Some(vec![format!("bench_ns_{}", id)]);
                    let start = Instant::now();
                    match ns.create_namespace(req).await {
                        Ok(_) => latencies.push(start.elapsed().as_secs_f64() * 1000.0),
                        Err(_) => errors += 1,
                    }
                }
                (latencies, errors)
            })
        })
        .collect();

    let mut all_latencies = Vec::new();
    let mut total_errors = 0;
    for h in handles {
        let (lats, errs) = h.await.unwrap();
        all_latencies.extend(lats);
        total_errors += errs;
    }
    let wall_duration = wall_start.elapsed();
    compute_result(
        variant,
        "write_create_namespace",
        config.concurrency,
        wall_duration,
        all_latencies,
        total_errors,
    )
}

async fn bench_write_create_table(
    ns: Arc<dyn LanceNamespace>,
    config: BenchConfig,
    variant: &str,
) -> BenchResult {
    let counter = Arc::new(AtomicU64::new(0));
    let barrier = Arc::new(Barrier::new(config.concurrency));
    let ipc_data = Bytes::from(create_test_ipc_data());

    // Warmup
    for i in 0..config.warmup {
        let mut req = CreateTableRequest::new();
        req.id = Some(vec![format!("warmup_table_{}", i)]);
        let _ = ns.create_table(req, ipc_data.clone()).await;
    }

    let wall_start = Instant::now();
    let handles: Vec<_> = (0..config.concurrency)
        .map(|_| {
            let ns = ns.clone();
            let counter = counter.clone();
            let barrier = barrier.clone();
            let ipc_data = ipc_data.clone();
            let ops_per_worker = config.operations / config.concurrency;
            tokio::spawn(async move {
                barrier.wait().await;
                let mut latencies = Vec::with_capacity(ops_per_worker);
                let mut errors = 0usize;
                for _ in 0..ops_per_worker {
                    let id = counter.fetch_add(1, Ordering::Relaxed);
                    let mut req = CreateTableRequest::new();
                    req.id = Some(vec![format!("bench_table_{}", id)]);
                    let start = Instant::now();
                    match ns.create_table(req, ipc_data.clone()).await {
                        Ok(_) => latencies.push(start.elapsed().as_secs_f64() * 1000.0),
                        Err(_) => errors += 1,
                    }
                }
                (latencies, errors)
            })
        })
        .collect();

    let mut all_latencies = Vec::new();
    let mut total_errors = 0;
    for h in handles {
        let (lats, errs) = h.await.unwrap();
        all_latencies.extend(lats);
        total_errors += errs;
    }
    let wall_duration = wall_start.elapsed();
    compute_result(
        variant,
        "write_create_table",
        config.concurrency,
        wall_duration,
        all_latencies,
        total_errors,
    )
}

async fn bench_read_list_namespaces(
    ns: Arc<dyn LanceNamespace>,
    config: BenchConfig,
    variant: &str,
) -> BenchResult {
    let barrier = Arc::new(Barrier::new(config.concurrency));

    // Warmup
    for _ in 0..config.warmup {
        let mut req = ListNamespacesRequest::new();
        req.id = Some(vec![]);
        let _ = ns.list_namespaces(req).await;
    }

    let wall_start = Instant::now();
    let handles: Vec<_> = (0..config.concurrency)
        .map(|_| {
            let ns = ns.clone();
            let barrier = barrier.clone();
            let ops_per_worker = config.operations / config.concurrency;
            tokio::spawn(async move {
                barrier.wait().await;
                let mut latencies = Vec::with_capacity(ops_per_worker);
                let mut errors = 0usize;
                for _ in 0..ops_per_worker {
                    let mut req = ListNamespacesRequest::new();
                    req.id = Some(vec![]);
                    let start = Instant::now();
                    match ns.list_namespaces(req).await {
                        Ok(_) => latencies.push(start.elapsed().as_secs_f64() * 1000.0),
                        Err(_) => errors += 1,
                    }
                }
                (latencies, errors)
            })
        })
        .collect();

    let mut all_latencies = Vec::new();
    let mut total_errors = 0;
    for h in handles {
        let (lats, errs) = h.await.unwrap();
        all_latencies.extend(lats);
        total_errors += errs;
    }
    let wall_duration = wall_start.elapsed();
    compute_result(
        variant,
        "read_list_namespaces",
        config.concurrency,
        wall_duration,
        all_latencies,
        total_errors,
    )
}

async fn bench_read_list_tables(
    ns: Arc<dyn LanceNamespace>,
    config: BenchConfig,
    variant: &str,
) -> BenchResult {
    let barrier = Arc::new(Barrier::new(config.concurrency));

    // Warmup
    for _ in 0..config.warmup {
        let mut req = ListTablesRequest::new();
        req.id = Some(vec![]);
        let _ = ns.list_tables(req).await;
    }

    let wall_start = Instant::now();
    let handles: Vec<_> = (0..config.concurrency)
        .map(|_| {
            let ns = ns.clone();
            let barrier = barrier.clone();
            let ops_per_worker = config.operations / config.concurrency;
            tokio::spawn(async move {
                barrier.wait().await;
                let mut latencies = Vec::with_capacity(ops_per_worker);
                let mut errors = 0usize;
                for _ in 0..ops_per_worker {
                    let mut req = ListTablesRequest::new();
                    req.id = Some(vec![]);
                    let start = Instant::now();
                    match ns.list_tables(req).await {
                        Ok(_) => latencies.push(start.elapsed().as_secs_f64() * 1000.0),
                        Err(_) => errors += 1,
                    }
                }
                (latencies, errors)
            })
        })
        .collect();

    let mut all_latencies = Vec::new();
    let mut total_errors = 0;
    for h in handles {
        let (lats, errs) = h.await.unwrap();
        all_latencies.extend(lats);
        total_errors += errs;
    }
    let wall_duration = wall_start.elapsed();
    compute_result(
        variant,
        "read_list_tables",
        config.concurrency,
        wall_duration,
        all_latencies,
        total_errors,
    )
}

async fn bench_read_describe_table(
    ns: Arc<dyn LanceNamespace>,
    config: BenchConfig,
    num_tables: usize,
    variant: &str,
) -> BenchResult {
    let counter = Arc::new(AtomicU64::new(0));
    let barrier = Arc::new(Barrier::new(config.concurrency));

    // Warmup
    for i in 0..config.warmup.min(num_tables) {
        let req = DescribeTableRequest {
            id: Some(vec![format!("table_{}", i)]),
            ..Default::default()
        };
        let _ = ns.describe_table(req).await;
    }

    let wall_start = Instant::now();
    let handles: Vec<_> = (0..config.concurrency)
        .map(|_| {
            let ns = ns.clone();
            let counter = counter.clone();
            let barrier = barrier.clone();
            let ops_per_worker = config.operations / config.concurrency;
            tokio::spawn(async move {
                barrier.wait().await;
                let mut latencies = Vec::with_capacity(ops_per_worker);
                let mut errors = 0usize;
                for _ in 0..ops_per_worker {
                    let id = counter.fetch_add(1, Ordering::Relaxed);
                    let table_idx = id as usize % num_tables;
                    let req = DescribeTableRequest {
                        id: Some(vec![format!("table_{}", table_idx)]),
                        ..Default::default()
                    };
                    let start = Instant::now();
                    match ns.describe_table(req).await {
                        Ok(_) => latencies.push(start.elapsed().as_secs_f64() * 1000.0),
                        Err(_) => errors += 1,
                    }
                }
                (latencies, errors)
            })
        })
        .collect();

    let mut all_latencies = Vec::new();
    let mut total_errors = 0;
    for h in handles {
        let (lats, errs) = h.await.unwrap();
        all_latencies.extend(lats);
        total_errors += errs;
    }
    let wall_duration = wall_start.elapsed();
    compute_result(
        variant,
        "read_describe_table",
        config.concurrency,
        wall_duration,
        all_latencies,
        total_errors,
    )
}

fn parse_concurrency_list(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .collect()
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let mut root = String::new();
    let mut concurrency_list = vec![1, 2, 5, 10, 20, 50, 100];
    let mut operations: usize = 500;
    let mut warmup: usize = 20;
    let mut seed_namespaces: usize = 50;
    let mut seed_tables: usize = 100;
    let mut inline_optimization = true;
    let mut variant = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--root" => {
                root = args[i + 1].clone();
                i += 2;
            }
            "--concurrency" => {
                concurrency_list = parse_concurrency_list(&args[i + 1]);
                i += 2;
            }
            "--operations" => {
                operations = args[i + 1].parse().expect("invalid --operations");
                i += 2;
            }
            "--warmup" => {
                warmup = args[i + 1].parse().expect("invalid --warmup");
                i += 2;
            }
            "--seed-namespaces" => {
                seed_namespaces = args[i + 1].parse().expect("invalid --seed-namespaces");
                i += 2;
            }
            "--seed-tables" => {
                seed_tables = args[i + 1].parse().expect("invalid --seed-tables");
                i += 2;
            }
            "--inline-optimization" => {
                inline_optimization = args[i + 1]
                    .parse::<bool>()
                    .expect("invalid --inline-optimization (true/false)");
                i += 2;
            }
            "--variant" => {
                variant = args[i + 1].clone();
                i += 2;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    if root.is_empty() {
        root = std::env::temp_dir()
            .join("manifest_bench")
            .to_string_lossy()
            .to_string();
    }
    if variant.is_empty() {
        variant = if inline_optimization {
            "default".to_string()
        } else {
            "no_inline_opt".to_string()
        };
    }

    eprintln!("=== Manifest Benchmark ===");
    eprintln!("variant: {}", variant);
    eprintln!("root: {}", root);
    eprintln!("inline_optimization: {}", inline_optimization);
    eprintln!("concurrency: {:?}", concurrency_list);
    eprintln!("operations per concurrency level: {}", operations);
    eprintln!("warmup: {}", warmup);
    eprintln!("seed: {} namespaces, {} tables", seed_namespaces, seed_tables);

    // ---- READ benchmarks: single shared namespace with seeded data ----
    eprintln!("\n--- Seeding read benchmark data ---");
    let read_root = format!("{}/read", root);
    let _ = std::fs::remove_dir_all(&read_root);
    std::fs::create_dir_all(&read_root).expect("failed to create read root");
    let read_ns = setup_namespace(&read_root, inline_optimization).await;
    seed_data(read_ns.as_ref(), seed_namespaces, seed_tables).await;
    eprintln!("Seeded {} namespaces and {} tables", seed_namespaces, seed_tables);

    for &concurrency in &concurrency_list {
        let actual_ops = (operations / concurrency) * concurrency;
        let config = BenchConfig {
            operations: actual_ops,
            warmup,
            concurrency,
        };

        let result = bench_read_list_namespaces(read_ns.clone(), config, &variant).await;
        println!("{}", serde_json::to_string(&result).unwrap());

        let result = bench_read_list_tables(read_ns.clone(), config, &variant).await;
        println!("{}", serde_json::to_string(&result).unwrap());

        let result =
            bench_read_describe_table(read_ns.clone(), config, seed_tables, &variant).await;
        println!("{}", serde_json::to_string(&result).unwrap());
    }

    // ---- WRITE benchmarks: fresh namespace per concurrency level ----
    for &concurrency in &concurrency_list {
        let actual_ops = (operations / concurrency) * concurrency;
        let config = BenchConfig {
            operations: actual_ops,
            warmup: warmup.min(20),
            concurrency,
        };

        // Fresh namespace for each write benchmark
        let write_ns_root = format!("{}/write_ns_c{}", root, concurrency);
        let _ = std::fs::remove_dir_all(&write_ns_root);
        std::fs::create_dir_all(&write_ns_root).expect("failed to create write root");
        let write_ns = setup_namespace(&write_ns_root, inline_optimization).await;
        let result = bench_write_create_namespace(write_ns, config, &variant).await;
        println!("{}", serde_json::to_string(&result).unwrap());

        let write_tbl_root = format!("{}/write_tbl_c{}", root, concurrency);
        let _ = std::fs::remove_dir_all(&write_tbl_root);
        std::fs::create_dir_all(&write_tbl_root).expect("failed to create write root");
        let write_ns = setup_namespace(&write_tbl_root, inline_optimization).await;
        let result = bench_write_create_table(write_ns, config, &variant).await;
        println!("{}", serde_json::to_string(&result).unwrap());
    }

    eprintln!("\n=== Benchmark complete ===");
}
