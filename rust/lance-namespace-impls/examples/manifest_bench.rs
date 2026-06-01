// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Multi-process manifest benchmark with S3 support.
//!
//! Modes:
//!   seed       — populate a manifest with N entries via namespace API
//!   seed-large — write a __manifest Lance table directly with N rows
//!   run        — benchmark read/write operations with multi-process concurrency
//!   worker     — (internal) single-process worker spawned by `run`
//!
//! Examples:
//!   # Seed 1000 entries via namespace API
//!   manifest_bench seed --root s3://bucket/bench/test1 --count 1000
//!
//!   # Seed 500K rows directly into __manifest table
//!   manifest_bench seed-large --root s3://bucket/bench/scale \
//!     --count 500000 --inline-optimization true
//!
//!   # Run scale benchmark at 500K initial entries
//!   manifest_bench run --root s3://bucket/bench/scale \
//!     --concurrency 1,10,100 --operations 200

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow::array::{RecordBatch, RecordBatchIterator, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use lance::dataset::{InsertBuilder, ReadParams, WriteMode, WriteParams, builder::DatasetBuilder};
use lance_arrow::json::JsonArray;
use lance_core::datatypes::LANCE_UNENFORCED_PRIMARY_KEY_POSITION;
use lance_namespace::LanceNamespace;
use lance_namespace::models::{
    CreateNamespaceRequest, CreateTableRequest, DeclareTableRequest, DescribeTableRequest,
    ListNamespacesRequest, ListTablesRequest,
};
use lance_namespace_impls::DirectoryNamespaceBuilder;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
struct LatencyRecord {
    operation: String,
    latency_ms: f64,
    error: bool,
}

#[derive(Serialize)]
struct BenchResult {
    variant: String,
    operation: String,
    concurrency: usize,
    initial_entries: usize,
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
    initial_entries: usize,
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
        initial_entries,
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
    use arrow::array::Int32Array;
    use arrow::ipc::writer::StreamWriter;

    let schema = Arc::new(ArrowSchema::new(vec![
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

fn manifest_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("object_id", DataType::Utf8, false).with_metadata(
            [(
                LANCE_UNENFORCED_PRIMARY_KEY_POSITION.to_string(),
                "0".to_string(),
            )]
            .into_iter()
            .collect(),
        ),
        Field::new("object_type", DataType::Utf8, false),
        Field::new("location", DataType::Utf8, true),
        lance_arrow::json::json_field("metadata", true),
    ]))
}

async fn build_namespace(
    root: &str,
    inline_optimization: bool,
    manifest_shard_count: usize,
    storage_options: &HashMap<String, String>,
) -> Box<dyn LanceNamespace> {
    let mut properties = HashMap::new();
    properties.insert("root".to_string(), root.to_string());
    properties.insert("dir_listing_enabled".to_string(), "false".to_string());
    properties.insert(
        "inline_optimization_enabled".to_string(),
        inline_optimization.to_string(),
    );
    if manifest_shard_count > 0 {
        properties.insert(
            "manifest_shard_count".to_string(),
            manifest_shard_count.to_string(),
        );
    }
    for (k, v) in storage_options {
        properties.insert(format!("storage.{}", k), v.clone());
    }
    let builder = DirectoryNamespaceBuilder::from_properties(properties, None)
        .expect("Failed to create namespace builder from properties");
    Box::new(builder.build().await.expect("Failed to build namespace"))
}

// ──────────────────── seed mode ────────────────────

async fn seed(
    root: &str,
    count: usize,
    inline_optimization: bool,
    manifest_shard_count: usize,
    storage_options: &HashMap<String, String>,
) {
    eprintln!("Seeding {} entries at {}", count, root);
    let ns = build_namespace(
        root,
        inline_optimization,
        manifest_shard_count,
        storage_options,
    )
    .await;
    let ipc_data = Bytes::from(create_test_ipc_data());

    let ns_count = count / 3;
    let table_count = count - ns_count;

    for i in 0..ns_count {
        let mut req = CreateNamespaceRequest::new();
        req.id = Some(vec![format!("ns_{}", i)]);
        if let Err(e) = ns.create_namespace(req).await {
            eprintln!("seed ns_{}: {}", i, e);
        }
        if (i + 1) % 100 == 0 {
            eprintln!("  seeded {}/{} namespaces", i + 1, ns_count);
        }
    }
    for i in 0..table_count {
        let mut req = CreateTableRequest::new();
        req.id = Some(vec![format!("table_{}", i)]);
        if let Err(e) = ns.create_table(req, ipc_data.clone()).await {
            eprintln!("seed table_{}: {}", i, e);
        }
        if (i + 1) % 100 == 0 {
            eprintln!("  seeded {}/{} tables", i + 1, table_count);
        }
    }
    eprintln!(
        "Seed complete: {} namespaces, {} tables",
        ns_count, table_count
    );
}

// ──────────────────── seed-large mode ────────────────────
// Writes a __manifest Lance table directly with N rows, bypassing the namespace API.

const SEED_LARGE_BATCH_SIZE: usize = 10_000;
const SEED_LARGE_SHARD_FLUSH_BYTES: usize = 0;
struct ManifestSeedRow {
    object_id: String,
    object_type: String,
    location: Option<String>,
    metadata: Option<String>,
}

fn stable_hash(value: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn manifest_shard_marker_row(shard_index: usize) -> ManifestSeedRow {
    ManifestSeedRow {
        object_id: format!("__manifest_shard_marker_{:06}", shard_index),
        object_type: "table_version".to_string(),
        location: None,
        metadata: None,
    }
}

fn manifest_seed_row(i: usize, total_count: usize) -> ManifestSeedRow {
    let ns_count = total_count / 3;
    if i < ns_count {
        ManifestSeedRow {
            object_id: format!("ns_{}", i),
            object_type: "namespace".to_string(),
            location: None,
            metadata: None,
        }
    } else {
        let table_idx = i - ns_count;
        ManifestSeedRow {
            object_id: format!("table_{}", table_idx),
            object_type: "table".to_string(),
            location: Some(format!("table_{}", table_idx)),
            metadata: Some(r#"{"bench":"true"}"#.to_string()),
        }
    }
}

fn manifest_batch_from_rows(schema: &Arc<ArrowSchema>, rows: &[ManifestSeedRow]) -> RecordBatch {
    let metadata_array = Arc::new(
        JsonArray::try_from_iter(rows.iter().map(|row| row.metadata.as_deref()))
            .expect("Failed to encode metadata as JSON")
            .into_inner(),
    );

    RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.object_id.clone())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.object_type.clone())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.location.clone())
                    .collect::<Vec<_>>(),
            )),
            metadata_array,
        ],
    )
    .expect("Failed to create manifest batch")
}

async fn write_manifest_dataset(
    manifest_uri: &str,
    batches: Vec<RecordBatch>,
    schema: Arc<ArrowSchema>,
    storage_options: HashMap<String, String>,
    max_bytes_per_file: Option<usize>,
) {
    let mut write_params = WriteParams {
        mode: WriteMode::Create,
        ..WriteParams::default()
    };
    if let Some(max_bytes_per_file) = max_bytes_per_file {
        write_params.max_bytes_per_file = max_bytes_per_file;
    }
    if !storage_options.is_empty() {
        let accessor = Arc::new(
            lance_io::object_store::StorageOptionsAccessor::with_static_options(storage_options),
        );
        write_params.store_params = Some(lance_io::object_store::ObjectStoreParams {
            storage_options_accessor: Some(accessor),
            ..Default::default()
        });
    }

    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    InsertBuilder::new(manifest_uri)
        .with_params(&write_params)
        .execute_stream(reader)
        .await
        .expect("Failed to write manifest dataset");
}

async fn validate_sharded_manifest_seed(
    root: &str,
    manifest_shard_count: usize,
    storage_options: &HashMap<String, String>,
) {
    let mut read_params = ReadParams::default();
    if !storage_options.is_empty() {
        let accessor = Arc::new(
            lance_io::object_store::StorageOptionsAccessor::with_static_options(
                storage_options.clone(),
            ),
        );
        read_params.store_options = Some(lance_io::object_store::ObjectStoreParams {
            storage_options_accessor: Some(accessor),
            ..Default::default()
        });
    }
    let manifest_uri = format!("{}/__manifest", root);
    let dataset = DatasetBuilder::from_uri(manifest_uri)
        .with_read_params(read_params)
        .load()
        .await
        .expect("Failed to load seeded sharded manifest");
    assert_eq!(
        dataset.manifest().fragments.len(),
        manifest_shard_count,
        "seed-large sharded manifest must contain exactly one fragment per shard"
    );
}

fn generate_manifest_batch(
    schema: &Arc<ArrowSchema>,
    start_idx: usize,
    batch_size: usize,
    total_count: usize,
) -> RecordBatch {
    let ns_count = total_count / 3;
    let actual_size = batch_size.min(total_count - start_idx);

    let mut object_ids = Vec::with_capacity(actual_size);
    let mut object_types = Vec::with_capacity(actual_size);
    let mut locations: Vec<Option<String>> = Vec::with_capacity(actual_size);
    let mut metadatas: Vec<Option<&str>> = Vec::with_capacity(actual_size);

    for i in start_idx..start_idx + actual_size {
        if i < ns_count {
            object_ids.push(format!("ns_{}", i));
            object_types.push("namespace".to_string());
            locations.push(None);
            metadatas.push(None);
        } else {
            let table_idx = i - ns_count;
            object_ids.push(format!("table_{}", table_idx));
            object_types.push("table".to_string());
            locations.push(Some(format!("table_{}", table_idx)));
            metadatas.push(Some(r#"{"bench":"true"}"#));
        }
    }

    let metadata_array = Arc::new(
        JsonArray::try_from_iter(metadatas.into_iter())
            .expect("Failed to encode metadata as JSON")
            .into_inner(),
    );

    RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(object_ids)),
            Arc::new(StringArray::from(object_types)),
            Arc::new(StringArray::from(locations)),
            metadata_array,
        ],
    )
    .expect("Failed to create manifest batch")
}

async fn seed_large(
    root: &str,
    count: usize,
    inline_optimization: bool,
    manifest_shard_count: usize,
    storage_options: &HashMap<String, String>,
) {
    if manifest_shard_count > 0 {
        seed_large_sharded(
            root,
            count,
            inline_optimization,
            manifest_shard_count,
            storage_options,
        )
        .await;
        return;
    }

    let manifest_uri = format!("{}/{}", root, "__manifest");
    eprintln!(
        "Seed-large: writing {} rows directly to {}",
        count, manifest_uri
    );

    let schema = manifest_schema();

    // Generate batches
    let mut batches = Vec::new();
    let mut offset = 0;
    while offset < count {
        let batch_size = SEED_LARGE_BATCH_SIZE.min(count - offset);
        batches.push(generate_manifest_batch(&schema, offset, batch_size, count));
        offset += batch_size;
    }
    eprintln!("  generated {} batches", batches.len());

    write_manifest_dataset(
        manifest_uri.as_str(),
        batches,
        schema.clone(),
        storage_options.clone(),
        None,
    )
    .await;

    eprintln!("  wrote Lance dataset");

    // Now open via namespace API to trigger the first CoW rewrite with indices
    if inline_optimization {
        eprintln!("  triggering initial CoW rewrite to build indices...");
        let start = Instant::now();
        let ns = build_namespace(root, true, manifest_shard_count, storage_options).await;
        let mut req = CreateNamespaceRequest::new();
        req.id = Some(vec!["__seed_trigger__".to_string()]);
        ns.create_namespace(req)
            .await
            .expect("Failed to trigger CoW rewrite");
        eprintln!(
            "  CoW rewrite with index build took {:.1}s",
            start.elapsed().as_secs_f64()
        );
    }

    let ns_count = count / 3;
    let table_count = count - ns_count;
    eprintln!(
        "Seed-large complete: {} total rows ({} namespaces, {} tables)",
        count, ns_count, table_count
    );
}

async fn seed_large_sharded(
    root: &str,
    count: usize,
    inline_optimization: bool,
    manifest_shard_count: usize,
    storage_options: &HashMap<String, String>,
) {
    eprintln!(
        "Seed-large: writing {} rows directly across {} manifest fragments",
        count, manifest_shard_count
    );

    let schema = manifest_schema();
    let mut rows_by_shard: Vec<Vec<ManifestSeedRow>> =
        (0..manifest_shard_count).map(|_| Vec::new()).collect();

    for i in 0..count {
        let row = manifest_seed_row(i, count);
        let shard_index = stable_hash(&row.object_id) as usize % manifest_shard_count;
        rows_by_shard[shard_index].push(row);
    }

    let batches = rows_by_shard
        .into_iter()
        .enumerate()
        .map(|(shard_index, rows)| {
            let mut shard_rows = Vec::with_capacity(rows.len() + 1);
            shard_rows.push(manifest_shard_marker_row(shard_index));
            shard_rows.extend(rows);
            manifest_batch_from_rows(&schema, &shard_rows)
        })
        .collect::<Vec<_>>();

    let manifest_uri = format!("{}/__manifest", root);
    write_manifest_dataset(
        &manifest_uri,
        batches,
        schema,
        storage_options.clone(),
        Some(SEED_LARGE_SHARD_FLUSH_BYTES),
    )
    .await;
    validate_sharded_manifest_seed(root, manifest_shard_count, storage_options).await;

    if inline_optimization {
        eprintln!(
            "  sharded seed-large wrote raw manifest fragments; fragment writes do not rebuild manifest indices"
        );
    }

    let ns_count = count / 3;
    let table_count = count - ns_count;
    eprintln!(
        "Seed-large complete: {} total rows ({} namespaces, {} tables)",
        count, ns_count, table_count
    );
}

// ──────────────────── worker mode ────────────────────

async fn worker(
    root: &str,
    operation: &str,
    operations: usize,
    warmup: usize,
    worker_id: usize,
    table_count: usize,
    inline_optimization: bool,
    manifest_shard_count: usize,
    ready_path: Option<&str>,
    start_path: Option<&str>,
    storage_options: &HashMap<String, String>,
) {
    let ns = build_namespace(
        root,
        inline_optimization,
        manifest_shard_count,
        storage_options,
    )
    .await;
    let ipc_data = Bytes::from(create_test_ipc_data());

    // Warmup (only for warm-read operations)
    if operation.starts_with("warm-read") {
        for _ in 0..warmup {
            let _ =
                run_operation(ns.as_ref(), operation, worker_id, 0, table_count, &ipc_data).await;
        }
    }

    wait_for_coordinator(ready_path, start_path);

    for i in 0..operations {
        let start = Instant::now();
        let err = run_operation(ns.as_ref(), operation, worker_id, i, table_count, &ipc_data)
            .await
            .is_err();
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let record = LatencyRecord {
            operation: operation.to_string(),
            latency_ms,
            error: err,
        };
        println!("{}", serde_json::to_string(&record).unwrap());
    }
}

async fn run_operation(
    ns: &dyn LanceNamespace,
    operation: &str,
    worker_id: usize,
    op_idx: usize,
    table_count: usize,
    ipc_data: &Bytes,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match operation {
        "cold-read-list-namespaces" | "warm-read-list-namespaces" => {
            let mut req = ListNamespacesRequest::new();
            req.id = Some(vec![]);
            ns.list_namespaces(req).await?;
        }
        "cold-read-list-tables" | "warm-read-list-tables" => {
            let mut req = ListTablesRequest::new();
            req.id = Some(vec![]);
            ns.list_tables(req).await?;
        }
        "cold-read-describe-table" | "warm-read-describe-table" => {
            let table_idx = (worker_id * 1000 + op_idx) % table_count.max(1);
            let req = DescribeTableRequest {
                id: Some(vec![format!("table_{}", table_idx)]),
                ..Default::default()
            };
            ns.describe_table(req).await?;
        }
        "write-create-namespace" => {
            let mut req = CreateNamespaceRequest::new();
            req.id = Some(vec![format!("bench_w{}_{}", worker_id, op_idx)]);
            ns.create_namespace(req).await?;
        }
        "write-create-table" => {
            let mut req = CreateTableRequest::new();
            req.id = Some(vec![format!("bench_t{}_{}", worker_id, op_idx)]);
            ns.create_table(req, ipc_data.clone()).await?;
        }
        "write-declare-table" => {
            let req = DeclareTableRequest {
                id: Some(vec![format!("bench_d{}_{}", worker_id, op_idx)]),
                ..Default::default()
            };
            ns.declare_table(req).await?;
        }
        _ => {
            return Err(format!("unknown operation: {}", operation).into());
        }
    }
    Ok(())
}

// ──────────────────── cold-read worker ────────────────────
// For cold reads, each operation opens a FRESH namespace to avoid caching.

async fn cold_read_worker(
    root: &str,
    operation: &str,
    operations: usize,
    worker_id: usize,
    table_count: usize,
    inline_optimization: bool,
    manifest_shard_count: usize,
    ready_path: Option<&str>,
    start_path: Option<&str>,
    storage_options: &HashMap<String, String>,
) {
    let ipc_data = Bytes::from(create_test_ipc_data());

    wait_for_coordinator(ready_path, start_path);

    for i in 0..operations {
        // Fresh namespace for each operation — simulates cold start
        let start = Instant::now();
        let ns = build_namespace(
            root,
            inline_optimization,
            manifest_shard_count,
            storage_options,
        )
        .await;
        let err = run_operation(ns.as_ref(), operation, worker_id, i, table_count, &ipc_data)
            .await
            .is_err();
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let record = LatencyRecord {
            operation: operation.to_string(),
            latency_ms,
            error: err,
        };
        println!("{}", serde_json::to_string(&record).unwrap());
    }
}

fn wait_for_coordinator(ready_path: Option<&str>, start_path: Option<&str>) {
    if let Some(ready_path) = ready_path {
        fs::write(ready_path, b"ready").expect("failed to write worker ready marker");
    }
    if let Some(start_path) = start_path {
        let start_path = Path::new(start_path);
        while !start_path.exists() {
            thread::sleep(Duration::from_millis(10));
        }
    }
}

// ──────────────────── run mode (coordinator) ────────────────────

fn run_workers(
    self_exe: &str,
    root: &str,
    operation: &str,
    concurrency: usize,
    operations: usize,
    warmup: usize,
    table_count: usize,
    initial_entries: usize,
    inline_optimization: bool,
    manifest_shard_count: usize,
    variant: &str,
    storage_options: &HashMap<String, String>,
) -> BenchResult {
    let ops_per_worker = operations / concurrency.max(1);
    if ops_per_worker == 0 {
        return compute_result(
            variant,
            operation,
            concurrency,
            initial_entries,
            Duration::ZERO,
            vec![],
            0,
        );
    }

    let sync_dir = create_sync_dir();
    let start_path = sync_dir.join("start");
    let ready_paths: Vec<_> = (0..concurrency)
        .map(|worker_id| sync_dir.join(format!("ready_{}", worker_id)))
        .collect();

    let mut children: Vec<_> = (0..concurrency)
        .map(|worker_id| {
            let mut cmd = Command::new(self_exe);
            let unique_worker_id = concurrency * 1_000_000 + worker_id;
            cmd.arg("worker")
                .arg("--root")
                .arg(root)
                .arg("--operation")
                .arg(operation)
                .arg("--operations")
                .arg(ops_per_worker.to_string())
                .arg("--warmup")
                .arg(warmup.to_string())
                .arg("--worker-id")
                .arg(unique_worker_id.to_string())
                .arg("--table-count")
                .arg(table_count.to_string())
                .arg("--inline-optimization")
                .arg(inline_optimization.to_string())
                .arg("--manifest-shard-count")
                .arg(manifest_shard_count.to_string())
                .arg("--ready-path")
                .arg(ready_paths[worker_id].to_string_lossy().to_string())
                .arg("--start-path")
                .arg(start_path.to_string_lossy().to_string());
            for (k, v) in storage_options {
                cmd.arg("--storage-option").arg(format!("{}={}", k, v));
            }
            cmd.stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .expect("Failed to spawn worker")
        })
        .collect();

    let output_readers: Vec<_> = children
        .iter_mut()
        .map(|child| {
            let stdout = child.stdout.take().unwrap();
            thread::spawn(move || {
                let reader = BufReader::new(stdout);
                let mut latencies = Vec::new();
                let mut errors = 0;
                for line in reader.lines() {
                    let line = line.expect("failed to read worker output");
                    if let Ok(record) = serde_json::from_str::<LatencyRecord>(&line) {
                        if record.error {
                            errors += 1;
                        } else {
                            latencies.push(record.latency_ms);
                        }
                    }
                }
                (latencies, errors)
            })
        })
        .collect();

    wait_for_workers_ready(&mut children, &ready_paths);
    let wall_start = Instant::now();
    fs::write(&start_path, b"start").expect("failed to write worker start marker");

    for mut child in children {
        let status = child.wait().expect("failed to wait for worker");
        if !status.success() {
            eprintln!("Worker exited with status: {}", status);
        }
    }

    let mut all_latencies = Vec::new();
    let mut total_errors = 0;
    for reader in output_readers {
        let (mut latencies, errors) = reader.join().expect("failed to join worker output reader");
        all_latencies.append(&mut latencies);
        total_errors += errors;
    }

    let wall_duration = wall_start.elapsed();
    let _ = fs::remove_dir_all(&sync_dir);
    compute_result(
        variant,
        operation,
        concurrency,
        initial_entries,
        wall_duration,
        all_latencies,
        total_errors,
    )
}

fn create_sync_dir() -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before unix epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "manifest_bench_sync_{}_{}",
        std::process::id(),
        now
    ));
    fs::create_dir_all(&dir).expect("failed to create worker sync directory");
    dir
}

fn wait_for_workers_ready(children: &mut [std::process::Child], ready_paths: &[PathBuf]) {
    let start = Instant::now();
    loop {
        if ready_paths.iter().all(|path| path.exists()) {
            return;
        }

        for child in children.iter_mut() {
            if let Some(status) = child
                .try_wait()
                .expect("failed to check worker readiness status")
            {
                panic!("worker exited before benchmark start: {}", status);
            }
        }

        if start.elapsed() > Duration::from_secs(30 * 60) {
            panic!("timed out waiting for workers to open namespace");
        }

        thread::sleep(Duration::from_millis(50));
    }
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
    if args.len() < 2 {
        eprintln!("Usage: manifest_bench <seed|seed-large|run|worker> [options]");
        std::process::exit(1);
    }

    let mode = args[1].as_str();
    let mut root = String::new();
    let mut operation = String::new();
    let mut operations: usize = 100;
    let mut warmup: usize = 10;
    let mut concurrency_list = vec![1, 2, 5, 10, 20, 50, 100];
    let mut count: usize = 1000;
    let mut worker_id: usize = 0;
    let mut table_count: usize = 667; // default for 1000 seed: 1000 - 1000/3
    let mut initial_entries: usize = 0;
    let mut inline_optimization = true;
    let mut manifest_shard_count: usize = 0;
    let mut variant = String::new();
    let mut ready_path: Option<String> = None;
    let mut start_path: Option<String> = None;
    let mut storage_options: HashMap<String, String> = HashMap::new();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--root" => {
                root = args[i + 1].clone();
                i += 2;
            }
            "--operation" => {
                operation = args[i + 1].clone();
                i += 2;
            }
            "--operations" => {
                operations = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--warmup" => {
                warmup = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--concurrency" => {
                concurrency_list = parse_concurrency_list(&args[i + 1]);
                i += 2;
            }
            "--count" => {
                count = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--worker-id" => {
                worker_id = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--table-count" => {
                table_count = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--initial-entries" => {
                initial_entries = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--inline-optimization" => {
                inline_optimization = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--manifest-shard-count" => {
                manifest_shard_count = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--variant" => {
                variant = args[i + 1].clone();
                i += 2;
            }
            "--ready-path" => {
                ready_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--start-path" => {
                start_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--storage-option" => {
                let kv = &args[i + 1];
                if let Some((k, v)) = kv.split_once('=') {
                    storage_options.insert(k.to_string(), v.to_string());
                }
                i += 2;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    if variant.is_empty() {
        variant = if inline_optimization {
            "default".to_string()
        } else {
            "no_inline_opt".to_string()
        };
    }

    match mode {
        "seed" => {
            seed(
                &root,
                count,
                inline_optimization,
                manifest_shard_count,
                &storage_options,
            )
            .await;
        }
        "seed-large" => {
            seed_large(
                &root,
                count,
                inline_optimization,
                manifest_shard_count,
                &storage_options,
            )
            .await;
        }
        "worker" => {
            if operation.starts_with("cold-read") {
                cold_read_worker(
                    &root,
                    &operation,
                    operations,
                    worker_id,
                    table_count,
                    inline_optimization,
                    manifest_shard_count,
                    ready_path.as_deref(),
                    start_path.as_deref(),
                    &storage_options,
                )
                .await;
            } else {
                worker(
                    &root,
                    &operation,
                    operations,
                    warmup,
                    worker_id,
                    table_count,
                    inline_optimization,
                    manifest_shard_count,
                    ready_path.as_deref(),
                    start_path.as_deref(),
                    &storage_options,
                )
                .await;
            }
        }
        "run" => {
            let self_exe = std::env::current_exe()
                .expect("failed to get self exe path")
                .to_string_lossy()
                .to_string();

            let operations_list = [
                "cold-read-list-namespaces",
                "cold-read-list-tables",
                "cold-read-describe-table",
                "warm-read-list-namespaces",
                "warm-read-list-tables",
                "warm-read-describe-table",
                "write-create-namespace",
                "write-declare-table",
                "write-create-table",
            ];

            // If --operation is set, only run that one
            let ops: Vec<&str> = if operation.is_empty() {
                operations_list.to_vec()
            } else {
                vec![operation.as_str()]
            };

            eprintln!("=== Manifest Benchmark ===");
            eprintln!("variant: {}", variant);
            eprintln!("root: {}", root);
            eprintln!("inline_optimization: {}", inline_optimization);
            eprintln!("manifest_shard_count: {}", manifest_shard_count);
            eprintln!("initial_entries: {}", initial_entries);
            eprintln!("concurrency: {:?}", concurrency_list);
            eprintln!("operations per level: {}", operations);
            eprintln!("warmup: {}", warmup);
            eprintln!("table_count: {}", table_count);

            for op in &ops {
                for &concurrency in &concurrency_list {
                    let actual_ops = (operations / concurrency) * concurrency;
                    eprintln!("  {} concurrency={} ops={}", op, concurrency, actual_ops);
                    let result = run_workers(
                        &self_exe,
                        &root,
                        op,
                        concurrency,
                        actual_ops,
                        warmup,
                        table_count,
                        initial_entries,
                        inline_optimization,
                        manifest_shard_count,
                        &variant,
                        &storage_options,
                    );
                    println!("{}", serde_json::to_string(&result).unwrap());
                }
            }
            eprintln!("=== Benchmark complete ===");
        }
        _ => {
            eprintln!(
                "Unknown mode: {}. Use seed, seed-large, run, or worker.",
                mode
            );
            std::process::exit(1);
        }
    }
}
