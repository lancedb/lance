// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark for LSM Scanner read performance.
//!
//! This benchmark compares scanning performance between:
//! - A single Lance table (baseline)
//! - LSM scan across base table + flushed MemTables + active MemTable
//!
//! ## Benchmark Groups
//!
//! - **LSM Scan**: Full table scan with and without memtables
//! - **LSM Scan Projected**: Scan with column projection
//! - **LSM Point Lookup**: Primary key-based point lookups
//! - **LSM Vector Search**: KNN search across LSM levels
//!
//! ## Running against S3
//!
//! ```bash
//! export AWS_DEFAULT_REGION=us-east-1
//! export DATASET_PREFIX=s3://your-bucket/bench/mem_wal_read
//! cargo bench --bench mem_wal_read
//! ```
//!
//! ## Running against local filesystem (with temp directory)
//!
//! ```bash
//! cargo bench --bench mem_wal_read
//! ```
//!
//! ## Running against specific local directory
//!
//! ```bash
//! export DATASET_PREFIX=/tmp/bench/mem_wal_read
//! cargo bench --bench mem_wal_read
//! ```
//!
//! ## Configuration
//!
//! - `DATASET_PREFIX`: Base URI for datasets (optional, e.g. s3://bucket/prefix or /tmp/bench).
//!   If not set, uses a temporary directory.
//! - `BASE_ROWS`: Number of rows in base table (default: 10000)
//! - `MEMTABLE_ROWS`: Number of rows per MemTable generation (default: 1000)
//! - `BATCH_SIZE`: Rows per write batch (default: 100)
//! - `SAMPLE_SIZE`: Number of benchmark iterations (default: 100)
//! - `VECTOR_DIM`: Vector dimension for vector search benchmark (default: 128)

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::sync::Arc;
use std::time::Duration;

use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
use arrow_array::{FixedSizeListArray, Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use datafusion::common::ScalarValue;
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use lance::dataset::mem_wal::scanner::{
    ActiveMemTableRef, LsmDataSourceCollector, LsmPointLookupPlanner, LsmScanner,
    LsmVectorSearchPlanner, RegionSnapshot,
};
use lance::dataset::mem_wal::{DatasetMemWalExt, MemWalConfig, RegionWriterConfig};
use lance::dataset::{Dataset, WriteParams};
use lance_linalg::distance::DistanceType;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use uuid::Uuid;

const DEFAULT_BASE_ROWS: usize = 10000;
const DEFAULT_MEMTABLE_ROWS: usize = 1000;
const DEFAULT_BATCH_SIZE: usize = 100;
const DEFAULT_VECTOR_DIM: usize = 128;

fn get_base_rows() -> usize {
    std::env::var("BASE_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BASE_ROWS)
}

fn get_memtable_rows() -> usize {
    std::env::var("MEMTABLE_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MEMTABLE_ROWS)
}

fn get_batch_size() -> usize {
    std::env::var("BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BATCH_SIZE)
}

fn get_sample_size() -> usize {
    std::env::var("SAMPLE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100)
        .max(10)
}

fn get_vector_dim() -> usize {
    std::env::var("VECTOR_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_VECTOR_DIM)
}

/// Get or create dataset prefix directory.
/// Uses DATASET_PREFIX environment variable if set, otherwise creates a temporary directory.
fn get_dataset_prefix() -> String {
    std::env::var("DATASET_PREFIX").unwrap_or_else(|_| {
        let temp_dir = std::env::temp_dir().join(format!("lance_bench_read_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir).expect("Failed to create temp directory");
        temp_dir.to_string_lossy().to_string()
    })
}

/// Get storage label from dataset prefix (e.g. "s3" or "local").
fn get_storage_label(prefix: &str) -> &'static str {
    if prefix.starts_with("s3://") {
        "s3"
    } else if prefix.starts_with("gs://") {
        "gcs"
    } else if prefix.starts_with("az://") {
        "azure"
    } else {
        "local"
    }
}

/// Create test schema: (id: Int64, name: Utf8)
fn create_schema() -> Arc<ArrowSchema> {
    use std::collections::HashMap;

    let mut id_metadata = HashMap::new();
    id_metadata.insert(
        "lance-schema:unenforced-primary-key".to_string(),
        "true".to_string(),
    );
    let id_field = Field::new("id", DataType::Int64, false).with_metadata(id_metadata);

    Arc::new(ArrowSchema::new(vec![
        id_field,
        Field::new("name", DataType::Utf8, true),
    ]))
}

/// Create a test batch with sequential IDs.
fn create_batch(schema: &ArrowSchema, start_id: i64, num_rows: usize) -> RecordBatch {
    let ids: Vec<i64> = (start_id..start_id + num_rows as i64).collect();
    let names: Vec<String> = ids.iter().map(|id| format!("name_{}", id)).collect();

    RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(names)),
        ],
    )
    .unwrap()
}

/// Setup context for benchmarks.
struct BenchContext {
    /// Base dataset (for baseline scan).
    base_dataset: Arc<Dataset>,
    /// Dataset with MemWAL for LSM scan.
    lsm_dataset: Arc<Dataset>,
    /// Region snapshots with flushed generations.
    region_snapshots: Vec<RegionSnapshot>,
    /// Active memtable reference.
    active_memtable: Option<(Uuid, ActiveMemTableRef)>,
    /// Total rows across all sources.
    total_rows: usize,
    /// Primary key columns.
    pk_columns: Vec<String>,
}

/// Create benchmark context with:
/// - Base table with base_rows
/// - 2 flushed MemTables with memtable_rows each
/// - 1 active MemTable with memtable_rows
async fn setup_benchmark(
    base_rows: usize,
    memtable_rows: usize,
    batch_size: usize,
    dataset_prefix: &str,
) -> BenchContext {
    let schema = create_schema();
    let pk_columns = vec!["id".to_string()];

    // Use short random suffix for unique dataset names
    let short_id = &Uuid::new_v4().to_string()[..8];
    let prefix = dataset_prefix.trim_end_matches('/');

    // Create base dataset (for baseline comparison)
    let base_uri = format!("{}/base_{}", prefix, short_id);
    let base_batches: Vec<RecordBatch> = (0..base_rows.div_ceil(batch_size))
        .map(|i| {
            let start = (i * batch_size) as i64;
            let rows = batch_size.min(base_rows - i * batch_size);
            create_batch(&schema, start, rows)
        })
        .collect();

    let reader = RecordBatchIterator::new(base_batches.into_iter().map(Ok), schema.clone());
    let base_dataset = Arc::new(
        Dataset::write(reader, &base_uri, Some(WriteParams::default()))
            .await
            .unwrap(),
    );

    // Create LSM dataset with same base data
    let lsm_uri = format!("{}/lsm_{}", prefix, short_id);
    let lsm_base_batches: Vec<RecordBatch> = (0..base_rows.div_ceil(batch_size))
        .map(|i| {
            let start = (i * batch_size) as i64;
            let rows = batch_size.min(base_rows - i * batch_size);
            create_batch(&schema, start, rows)
        })
        .collect();

    let reader = RecordBatchIterator::new(lsm_base_batches.into_iter().map(Ok), schema.clone());
    let mut lsm_dataset = Dataset::write(reader, &lsm_uri, Some(WriteParams::default()))
        .await
        .unwrap();

    // Initialize MemWAL
    lsm_dataset
        .initialize_mem_wal(MemWalConfig {
            region_spec: None,
            maintained_indexes: vec![],
        })
        .await
        .unwrap();

    let lsm_dataset = Arc::new(lsm_dataset);

    // Create RegionWriter with small memtable size to trigger flushes
    let region_id = Uuid::new_v4();
    let config = RegionWriterConfig {
        region_id,
        region_spec_id: 0,
        durable_write: false,
        sync_indexed_write: false,
        max_memtable_size: memtable_rows * 50, // ~50 bytes per row, triggers flush after memtable_rows
        max_memtable_rows: memtable_rows,
        max_wal_flush_interval: Some(Duration::from_secs(60)), // Long interval to avoid time-based flushes
        ..RegionWriterConfig::default()
    };

    let writer = lsm_dataset
        .as_ref()
        .mem_wal_writer(region_id, config)
        .await
        .unwrap();

    // Determine flush wait time based on storage type (cloud storage needs more time)
    let is_cloud = dataset_prefix.starts_with("s3://")
        || dataset_prefix.starts_with("gs://")
        || dataset_prefix.starts_with("az://");
    let flush_wait = if is_cloud {
        Duration::from_secs(5)
    } else {
        Duration::from_millis(500)
    };

    // Write data for generation 1 (will be flushed)
    let gen1_start = base_rows as i64;
    for i in 0..memtable_rows.div_ceil(batch_size) {
        let start = gen1_start + (i * batch_size) as i64;
        let rows = batch_size.min(memtable_rows - i * batch_size);
        let batch = create_batch(&schema, start, rows);
        writer.put(vec![batch]).await.unwrap();
    }

    // Wait for memtable flush
    tokio::time::sleep(flush_wait).await;

    // Write data for generation 2 (will be flushed)
    let gen2_start = gen1_start + memtable_rows as i64;
    for i in 0..memtable_rows.div_ceil(batch_size) {
        let start = gen2_start + (i * batch_size) as i64;
        let rows = batch_size.min(memtable_rows - i * batch_size);
        let batch = create_batch(&schema, start, rows);
        writer.put(vec![batch]).await.unwrap();
    }

    // Wait for memtable flush
    tokio::time::sleep(flush_wait).await;

    // Write data for generation 3 (active memtable, not flushed)
    let gen3_start = gen2_start + memtable_rows as i64;
    let gen3_rows = memtable_rows / 2; // Smaller to keep in memory
    for i in 0..gen3_rows.div_ceil(batch_size) {
        let start = gen3_start + (i * batch_size) as i64;
        let rows = batch_size.min(gen3_rows - i * batch_size);
        let batch = create_batch(&schema, start, rows);
        writer.put(vec![batch]).await.unwrap();
    }

    // Get manifest to find flushed generations
    let manifest = writer.manifest().await.unwrap();

    // Get active memtable reference
    let active_memtable_ref = writer.active_memtable_ref().await;

    // Build region snapshot
    let mut region_snapshot = RegionSnapshot::new(region_id);
    if let Some(ref m) = manifest {
        region_snapshot = region_snapshot.with_current_generation(m.current_generation);
        for fg in &m.flushed_generations {
            region_snapshot =
                region_snapshot.with_flushed_generation(fg.generation, fg.path.clone());
        }
    }

    let num_flushed = manifest
        .as_ref()
        .map(|m| m.flushed_generations.len())
        .unwrap_or(0);

    println!("Setup complete:");
    println!("  Base table: {} rows", base_rows);
    println!("  LSM dataset URI: {}", lsm_dataset.uri());
    println!("  Flushed MemTables: {} generations", num_flushed);
    if let Some(ref m) = manifest {
        for fg in &m.flushed_generations {
            println!("    - Gen {}: path={}", fg.generation, fg.path);
        }
    }
    println!("  Active MemTable: {} rows", gen3_rows);
    println!(
        "  Total LSM rows: {}",
        base_rows + memtable_rows * 2 + gen3_rows
    );

    // Don't close writer - keep active memtable alive
    // We'll leak it for the benchmark (acceptable for benchmarks)
    std::mem::forget(writer);

    BenchContext {
        base_dataset,
        lsm_dataset,
        region_snapshots: vec![region_snapshot],
        active_memtable: Some((region_id, active_memtable_ref)),
        total_rows: base_rows + memtable_rows * 2 + gen3_rows,
        pk_columns,
    }
}

/// Benchmark scan operations.
fn bench_scan(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let base_rows = get_base_rows();
    let memtable_rows = get_memtable_rows();
    let batch_size = get_batch_size();
    let sample_size = get_sample_size();
    let dataset_prefix = get_dataset_prefix();
    let storage_label = get_storage_label(&dataset_prefix);

    println!("=== LSM Read Benchmark ===");
    println!("Storage: {} ({})", dataset_prefix, storage_label);
    println!("Base rows: {}", base_rows);
    println!("MemTable rows: {}", memtable_rows);
    println!("Batch size: {}", batch_size);
    println!();

    // Setup benchmark context
    let ctx = rt.block_on(setup_benchmark(
        base_rows,
        memtable_rows,
        batch_size,
        &dataset_prefix,
    ));

    let mut group = c.benchmark_group("LSM Scan");
    group.throughput(Throughput::Elements(ctx.total_rows as u64));
    group.sample_size(sample_size);

    let label = format!("{}_total_rows", ctx.total_rows);

    // Baseline: Scan base table only
    group.bench_with_input(BenchmarkId::new("BaseTable_Only", &label), &(), |b, _| {
        let dataset = ctx.base_dataset.clone();
        b.to_async(&rt).iter(|| async {
            let batches: Vec<RecordBatch> = dataset
                .scan()
                .try_into_stream()
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            let total: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert!(total > 0);
        });
    });

    // LSM scan: base + flushed (without active memtable for fair comparison)
    group.bench_with_input(
        BenchmarkId::new("LSM_Base_Plus_Flushed", &label),
        &(),
        |b, _| {
            let dataset = ctx.lsm_dataset.clone();
            let region_snapshots = ctx.region_snapshots.clone();
            let pk_columns = ctx.pk_columns.clone();
            b.to_async(&rt).iter(|| {
                let dataset = dataset.clone();
                let region_snapshots = region_snapshots.clone();
                let pk_columns = pk_columns.clone();
                async move {
                    let scanner = LsmScanner::new(dataset, region_snapshots, pk_columns);
                    let batches: Vec<RecordBatch> = scanner
                        .try_into_stream()
                        .await
                        .unwrap()
                        .try_collect()
                        .await
                        .unwrap();
                    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
                    assert!(total > 0);
                }
            });
        },
    );

    // LSM scan: base + flushed + active memtable
    if let Some((region_id, ref active_memtable)) = ctx.active_memtable {
        group.bench_with_input(BenchmarkId::new("LSM_Full", &label), &(), |b, _| {
            let dataset = ctx.lsm_dataset.clone();
            let region_snapshots = ctx.region_snapshots.clone();
            let pk_columns = ctx.pk_columns.clone();
            let active = active_memtable.clone();
            b.to_async(&rt).iter(|| {
                let dataset = dataset.clone();
                let region_snapshots = region_snapshots.clone();
                let pk_columns = pk_columns.clone();
                let active = active.clone();
                async move {
                    let scanner = LsmScanner::new(dataset, region_snapshots, pk_columns)
                        .with_active_memtable(region_id, active);
                    let batches: Vec<RecordBatch> = scanner
                        .try_into_stream()
                        .await
                        .unwrap()
                        .try_collect()
                        .await
                        .unwrap();
                    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
                    assert!(total > 0);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark with projection.
fn bench_scan_with_projection(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let base_rows = get_base_rows();
    let memtable_rows = get_memtable_rows();
    let batch_size = get_batch_size();
    let sample_size = get_sample_size();
    let dataset_prefix = get_dataset_prefix();

    // Setup benchmark context
    let ctx = rt.block_on(setup_benchmark(
        base_rows,
        memtable_rows,
        batch_size,
        &dataset_prefix,
    ));

    let mut group = c.benchmark_group("LSM Scan Projected");
    group.throughput(Throughput::Elements(ctx.total_rows as u64));
    group.sample_size(sample_size);

    let label = format!("{}_total_rows", ctx.total_rows);

    // Baseline: Scan base table with projection
    group.bench_with_input(
        BenchmarkId::new("BaseTable_Projected", &label),
        &(),
        |b, _| {
            let dataset = ctx.base_dataset.clone();
            b.to_async(&rt).iter(|| async {
                let batches: Vec<RecordBatch> = dataset
                    .scan()
                    .project(&["id"])
                    .unwrap()
                    .try_into_stream()
                    .await
                    .unwrap()
                    .try_collect()
                    .await
                    .unwrap();
                let total: usize = batches.iter().map(|b| b.num_rows()).sum();
                assert!(total > 0);
            });
        },
    );

    // LSM scan with projection
    if let Some((region_id, ref active_memtable)) = ctx.active_memtable {
        group.bench_with_input(
            BenchmarkId::new("LSM_Full_Projected", &label),
            &(),
            |b, _| {
                let dataset = ctx.lsm_dataset.clone();
                let region_snapshots = ctx.region_snapshots.clone();
                let pk_columns = ctx.pk_columns.clone();
                let active = active_memtable.clone();
                b.to_async(&rt).iter(|| {
                    let dataset = dataset.clone();
                    let region_snapshots = region_snapshots.clone();
                    let pk_columns = pk_columns.clone();
                    let active = active.clone();
                    async move {
                        let scanner = LsmScanner::new(dataset, region_snapshots, pk_columns)
                            .with_active_memtable(region_id, active)
                            .project(&["id"]);
                        let batches: Vec<RecordBatch> = scanner
                            .try_into_stream()
                            .await
                            .unwrap()
                            .try_collect()
                            .await
                            .unwrap();
                        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
                        assert!(total > 0);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark point lookup operations.
fn bench_point_lookup(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let base_rows = get_base_rows();
    let memtable_rows = get_memtable_rows();
    let batch_size = get_batch_size();
    let sample_size = get_sample_size();
    let dataset_prefix = get_dataset_prefix();

    let ctx = rt.block_on(setup_benchmark(
        base_rows,
        memtable_rows,
        batch_size,
        &dataset_prefix,
    ));

    let mut group = c.benchmark_group("LSM Point Lookup");
    group.throughput(Throughput::Elements(1));
    group.sample_size(sample_size);

    let label = format!("{}_total_rows", ctx.total_rows);

    // Lookup IDs from different locations:
    // - base_lookup_id: exists in base table
    // - flushed_lookup_id: exists in flushed memtable (gen1)
    // - active_lookup_id: exists in active memtable (gen3)
    let base_lookup_id = (base_rows / 2) as i64;
    let flushed_lookup_id = (base_rows + memtable_rows / 2) as i64;
    let active_lookup_id = (base_rows + memtable_rows * 2 + memtable_rows / 4) as i64;

    // Baseline: Filter scan on base table for point lookup
    group.bench_with_input(
        BenchmarkId::new("BaseTable_FilterScan", &label),
        &(),
        |b, _| {
            let dataset = ctx.base_dataset.clone();
            let lookup_id = base_lookup_id;
            let filter_str = format!("id = {}", lookup_id);
            b.to_async(&rt).iter(|| {
                let dataset = dataset.clone();
                let filter = filter_str.clone();
                async move {
                    let batches: Vec<RecordBatch> = dataset
                        .scan()
                        .filter(filter.as_str())
                        .unwrap()
                        .limit(Some(1), None)
                        .unwrap()
                        .try_into_stream()
                        .await
                        .unwrap()
                        .try_collect()
                        .await
                        .unwrap();
                    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
                    assert_eq!(total, 1);
                }
            });
        },
    );

    // LSM point lookup: key in base table
    if let Some((region_id, ref active_memtable)) = ctx.active_memtable {
        let arrow_schema: Arc<ArrowSchema> = Arc::new(ctx.lsm_dataset.schema().into());

        group.bench_with_input(
            BenchmarkId::new("LSM_Lookup_BaseKey", &label),
            &(),
            |b, _| {
                let dataset = ctx.lsm_dataset.clone();
                let region_snapshots = ctx.region_snapshots.clone();
                let pk_columns = ctx.pk_columns.clone();
                let schema = arrow_schema.clone();
                let active = active_memtable.clone();
                let lookup_id = base_lookup_id;
                b.to_async(&rt).iter(|| {
                    let dataset = dataset.clone();
                    let region_snapshots = region_snapshots.clone();
                    let pk_columns = pk_columns.clone();
                    let schema = schema.clone();
                    let active = active.clone();
                    async move {
                        let collector = LsmDataSourceCollector::new(dataset, region_snapshots)
                            .with_active_memtable(region_id, active);
                        let planner = LsmPointLookupPlanner::new(collector, pk_columns, schema);
                        let plan = planner
                            .plan_lookup(&[ScalarValue::Int64(Some(lookup_id))], None)
                            .await
                            .unwrap();
                        let session_ctx = SessionContext::new();
                        let stream = plan.execute(0, session_ctx.task_ctx()).unwrap();
                        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
                        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
                        assert!(total <= 1);
                    }
                });
            },
        );

        // LSM point lookup: key in flushed memtable
        group.bench_with_input(
            BenchmarkId::new("LSM_Lookup_FlushedKey", &label),
            &(),
            |b, _| {
                let dataset = ctx.lsm_dataset.clone();
                let region_snapshots = ctx.region_snapshots.clone();
                let pk_columns = ctx.pk_columns.clone();
                let schema = arrow_schema.clone();
                let active = active_memtable.clone();
                let lookup_id = flushed_lookup_id;
                b.to_async(&rt).iter(|| {
                    let dataset = dataset.clone();
                    let region_snapshots = region_snapshots.clone();
                    let pk_columns = pk_columns.clone();
                    let schema = schema.clone();
                    let active = active.clone();
                    async move {
                        let collector = LsmDataSourceCollector::new(dataset, region_snapshots)
                            .with_active_memtable(region_id, active);
                        let planner = LsmPointLookupPlanner::new(collector, pk_columns, schema);
                        let plan = planner
                            .plan_lookup(&[ScalarValue::Int64(Some(lookup_id))], None)
                            .await
                            .unwrap();
                        let session_ctx = SessionContext::new();
                        let stream = plan.execute(0, session_ctx.task_ctx()).unwrap();
                        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
                        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
                        assert!(total <= 1);
                    }
                });
            },
        );

        // LSM point lookup: key in active memtable
        group.bench_with_input(
            BenchmarkId::new("LSM_Lookup_ActiveKey", &label),
            &(),
            |b, _| {
                let dataset = ctx.lsm_dataset.clone();
                let region_snapshots = ctx.region_snapshots.clone();
                let pk_columns = ctx.pk_columns.clone();
                let schema = arrow_schema.clone();
                let active = active_memtable.clone();
                let lookup_id = active_lookup_id;
                b.to_async(&rt).iter(|| {
                    let dataset = dataset.clone();
                    let region_snapshots = region_snapshots.clone();
                    let pk_columns = pk_columns.clone();
                    let schema = schema.clone();
                    let active = active.clone();
                    async move {
                        let collector = LsmDataSourceCollector::new(dataset, region_snapshots)
                            .with_active_memtable(region_id, active);
                        let planner = LsmPointLookupPlanner::new(collector, pk_columns, schema);
                        let plan = planner
                            .plan_lookup(&[ScalarValue::Int64(Some(lookup_id))], None)
                            .await
                            .unwrap();
                        let session_ctx = SessionContext::new();
                        let stream = plan.execute(0, session_ctx.task_ctx()).unwrap();
                        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
                        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
                        assert!(total <= 1);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Create vector schema: (id: Int64, vector: FixedSizeList[Float32])
fn create_vector_schema(dim: usize) -> Arc<ArrowSchema> {
    use std::collections::HashMap;

    let mut id_metadata = HashMap::new();
    id_metadata.insert(
        "lance-schema:unenforced-primary-key".to_string(),
        "true".to_string(),
    );
    let id_field = Field::new("id", DataType::Int64, false).with_metadata(id_metadata);

    Arc::new(ArrowSchema::new(vec![
        id_field,
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]))
}

/// Create a batch with sequential IDs and random vectors.
fn create_vector_batch(
    schema: &ArrowSchema,
    start_id: i64,
    num_rows: usize,
    dim: usize,
) -> RecordBatch {
    let ids: Vec<i64> = (start_id..start_id + num_rows as i64).collect();

    let mut vector_builder = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
    for id in &ids {
        for d in 0..dim {
            let val = ((*id as f32) * 0.001 + (d as f32) * 0.0001) % 1.0;
            vector_builder.values().append_value(val);
        }
        vector_builder.append(true);
    }

    RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(vector_builder.finish()),
        ],
    )
    .unwrap()
}

/// Create a query vector.
fn create_query_vector(dim: usize) -> FixedSizeListArray {
    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
    for d in 0..dim {
        builder.values().append_value(0.5 + (d as f32) * 0.001);
    }
    builder.append(true);
    builder.finish()
}

/// Setup context for vector search benchmarks.
struct VectorBenchContext {
    base_dataset: Arc<Dataset>,
    lsm_dataset: Arc<Dataset>,
    region_snapshots: Vec<RegionSnapshot>,
    active_memtable: Option<(Uuid, ActiveMemTableRef)>,
    total_rows: usize,
    pk_columns: Vec<String>,
    vector_dim: usize,
}

/// Create benchmark context for vector search.
async fn setup_vector_benchmark(
    base_rows: usize,
    memtable_rows: usize,
    batch_size: usize,
    dataset_prefix: &str,
    dim: usize,
) -> VectorBenchContext {
    let schema = create_vector_schema(dim);
    let pk_columns = vec!["id".to_string()];

    let short_id = &Uuid::new_v4().to_string()[..8];
    let prefix = dataset_prefix.trim_end_matches('/');

    // Create base dataset
    let base_uri = format!("{}/vec_base_{}", prefix, short_id);
    let base_batches: Vec<RecordBatch> = (0..base_rows.div_ceil(batch_size))
        .map(|i| {
            let start = (i * batch_size) as i64;
            let rows = batch_size.min(base_rows - i * batch_size);
            create_vector_batch(&schema, start, rows, dim)
        })
        .collect();

    let reader = RecordBatchIterator::new(base_batches.into_iter().map(Ok), schema.clone());
    let base_dataset = Arc::new(
        Dataset::write(reader, &base_uri, Some(WriteParams::default()))
            .await
            .unwrap(),
    );

    // Create LSM dataset
    let lsm_uri = format!("{}/vec_lsm_{}", prefix, short_id);
    let lsm_base_batches: Vec<RecordBatch> = (0..base_rows.div_ceil(batch_size))
        .map(|i| {
            let start = (i * batch_size) as i64;
            let rows = batch_size.min(base_rows - i * batch_size);
            create_vector_batch(&schema, start, rows, dim)
        })
        .collect();

    let reader = RecordBatchIterator::new(lsm_base_batches.into_iter().map(Ok), schema.clone());
    let mut lsm_dataset = Dataset::write(reader, &lsm_uri, Some(WriteParams::default()))
        .await
        .unwrap();

    // Initialize MemWAL
    lsm_dataset
        .initialize_mem_wal(MemWalConfig {
            region_spec: None,
            maintained_indexes: vec![],
        })
        .await
        .unwrap();

    let lsm_dataset = Arc::new(lsm_dataset);

    let region_id = Uuid::new_v4();
    let config = RegionWriterConfig {
        region_id,
        region_spec_id: 0,
        durable_write: false,
        sync_indexed_write: false,
        max_memtable_size: memtable_rows * (dim * 4 + 8),
        max_memtable_rows: memtable_rows,
        max_wal_flush_interval: Some(Duration::from_secs(60)),
        ..RegionWriterConfig::default()
    };

    let writer = lsm_dataset
        .as_ref()
        .mem_wal_writer(region_id, config)
        .await
        .unwrap();

    let is_cloud = dataset_prefix.starts_with("s3://")
        || dataset_prefix.starts_with("gs://")
        || dataset_prefix.starts_with("az://");
    let flush_wait = if is_cloud {
        Duration::from_secs(5)
    } else {
        Duration::from_millis(500)
    };

    // Write flushed generations
    let gen1_start = base_rows as i64;
    for i in 0..memtable_rows.div_ceil(batch_size) {
        let start = gen1_start + (i * batch_size) as i64;
        let rows = batch_size.min(memtable_rows - i * batch_size);
        let batch = create_vector_batch(&schema, start, rows, dim);
        writer.put(vec![batch]).await.unwrap();
    }
    tokio::time::sleep(flush_wait).await;

    let gen2_start = gen1_start + memtable_rows as i64;
    for i in 0..memtable_rows.div_ceil(batch_size) {
        let start = gen2_start + (i * batch_size) as i64;
        let rows = batch_size.min(memtable_rows - i * batch_size);
        let batch = create_vector_batch(&schema, start, rows, dim);
        writer.put(vec![batch]).await.unwrap();
    }
    tokio::time::sleep(flush_wait).await;

    // Write active memtable
    let gen3_start = gen2_start + memtable_rows as i64;
    let gen3_rows = memtable_rows / 2;
    for i in 0..gen3_rows.div_ceil(batch_size) {
        let start = gen3_start + (i * batch_size) as i64;
        let rows = batch_size.min(gen3_rows - i * batch_size);
        let batch = create_vector_batch(&schema, start, rows, dim);
        writer.put(vec![batch]).await.unwrap();
    }

    let manifest = writer.manifest().await.unwrap();
    let active_memtable_ref = writer.active_memtable_ref().await;

    let mut region_snapshot = RegionSnapshot::new(region_id);
    if let Some(ref m) = manifest {
        region_snapshot = region_snapshot.with_current_generation(m.current_generation);
        for fg in &m.flushed_generations {
            region_snapshot =
                region_snapshot.with_flushed_generation(fg.generation, fg.path.clone());
        }
    }

    println!("Vector benchmark setup complete:");
    println!("  Vector dimension: {}", dim);
    println!("  Base table: {} rows", base_rows);
    println!(
        "  Total LSM rows: {}",
        base_rows + memtable_rows * 2 + gen3_rows
    );

    std::mem::forget(writer);

    VectorBenchContext {
        base_dataset,
        lsm_dataset,
        region_snapshots: vec![region_snapshot],
        active_memtable: Some((region_id, active_memtable_ref)),
        total_rows: base_rows + memtable_rows * 2 + gen3_rows,
        pk_columns,
        vector_dim: dim,
    }
}

/// Benchmark vector search operations.
fn bench_vector_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let base_rows = get_base_rows();
    let memtable_rows = get_memtable_rows();
    let batch_size = get_batch_size();
    let sample_size = get_sample_size();
    let dataset_prefix = get_dataset_prefix();
    let vector_dim = get_vector_dim();

    let ctx = rt.block_on(setup_vector_benchmark(
        base_rows,
        memtable_rows,
        batch_size,
        &dataset_prefix,
        vector_dim,
    ));

    let mut group = c.benchmark_group("LSM Vector Search");
    group.throughput(Throughput::Elements(10));
    group.sample_size(sample_size);

    let label = format!("{}_rows_{}d", ctx.total_rows, ctx.vector_dim);
    let k = 10;
    let nprobes = 1;

    // Baseline: KNN on base table
    group.bench_with_input(BenchmarkId::new("BaseTable_KNN", &label), &(), |b, _| {
        let dataset = ctx.base_dataset.clone();
        let query = create_query_vector(ctx.vector_dim);
        b.to_async(&rt).iter(|| {
            let dataset = dataset.clone();
            let query = query.clone();
            async move {
                let batches: Vec<RecordBatch> = dataset
                    .scan()
                    .nearest("vector", &query, k)
                    .unwrap()
                    .nprobes(nprobes)
                    .try_into_stream()
                    .await
                    .unwrap()
                    .try_collect()
                    .await
                    .unwrap();
                let total: usize = batches.iter().map(|b| b.num_rows()).sum();
                assert!(total <= k);
            }
        });
    });

    // LSM vector search
    if let Some((region_id, ref active_memtable)) = ctx.active_memtable {
        let arrow_schema: Arc<ArrowSchema> = Arc::new(ctx.lsm_dataset.schema().into());

        group.bench_with_input(BenchmarkId::new("LSM_KNN", &label), &(), |b, _| {
            let dataset = ctx.lsm_dataset.clone();
            let region_snapshots = ctx.region_snapshots.clone();
            let pk_columns = ctx.pk_columns.clone();
            let schema = arrow_schema.clone();
            let active = active_memtable.clone();
            let query = create_query_vector(ctx.vector_dim);
            b.to_async(&rt).iter(|| {
                let dataset = dataset.clone();
                let region_snapshots = region_snapshots.clone();
                let pk_columns = pk_columns.clone();
                let schema = schema.clone();
                let active = active.clone();
                let query = query.clone();
                async move {
                    let collector = LsmDataSourceCollector::new(dataset, region_snapshots)
                        .with_active_memtable(region_id, active);
                    let planner = LsmVectorSearchPlanner::new(
                        collector,
                        pk_columns,
                        schema,
                        "vector".to_string(),
                        DistanceType::L2,
                    );
                    let plan = planner.plan_search(&query, k, nprobes, None).await.unwrap();
                    let session_ctx = SessionContext::new();
                    let stream = plan.execute(0, session_ctx.task_ctx()).unwrap();
                    let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
                    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
                    assert!(total <= k);
                }
            });
        });
    }

    group.finish();
}

fn all_benchmarks(c: &mut Criterion) {
    bench_scan(c);
    bench_scan_with_projection(c);
    bench_point_lookup(c);
    bench_vector_search(c);
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .significance_level(0.05)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = all_benchmarks
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.05);
    targets = all_benchmarks
);

criterion_main!(benches);
