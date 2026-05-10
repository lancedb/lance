// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Focused MemWAL vector-search benchmark.
//!
//! Schema is intentionally minimal — `(id Int64, vector FixedSizeList<Float32>[dim])`
//! with `id` as the unenforced primary key. Only a vector index is created on
//! the base table and only that index is maintained in the MemWAL. This
//! isolates the in-memory vector index implementation (IVF-PQ on `main`,
//! HNSW on `jack/mem-wal-hnsw`) from the surrounding scan / point-lookup
//! machinery.
//!
//! ## Layout
//!
//! - Base table: `BASE_ROWS` rows with an IVF-PQ index `vector_idx`.
//! - MemWAL initialized with `maintained_indexes = ["vector_idx"]`.
//! - Two flushed MemTables (`MEMTABLE_ROWS` rows each).
//! - One active MemTable (`MEMTABLE_ROWS / 2` rows, kept in memory).
//!
//! ## Running against S3
//!
//! ```bash
//! export AWS_DEFAULT_REGION=us-east-1
//! export DATASET_PREFIX=s3://your-bucket/bench/mem_wal_vector
//! cargo bench --bench mem_wal_vector
//! ```
//!
//! ## Tunables
//!
//! - `DATASET_PREFIX`: base URI for datasets (optional)
//! - `BASE_ROWS`: rows in the base table (default 50_000)
//! - `MEMTABLE_ROWS`: rows per flushed memtable generation (default 5_000)
//! - `BATCH_SIZE`: rows per write batch (default 200)
//! - `VECTOR_DIM`: vector dimension (default 128)
//! - `SAMPLE_SIZE`: criterion sample count (default 30)
//! - `IVF_PARTITIONS`: number of IVF partitions on the base index (default 16)
//! - `PQ_SUBVECTORS`: number of PQ subvectors on the base index (default 16)

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::sync::Arc;
use std::time::Duration;

use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
use arrow_array::{FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use lance::dataset::mem_wal::scanner::{
    ActiveMemTableRef, LsmDataSourceCollector, LsmVectorSearchPlanner, ShardSnapshot,
};
use lance::dataset::mem_wal::{DatasetMemWalExt, MemWalConfig, ShardWriterConfig};
use lance::dataset::{Dataset, WriteParams};
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_index::IndexType;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::builder::PQBuildParams;
use lance_linalg::distance::{DistanceType, MetricType};
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};
use uuid::Uuid;

const DEFAULT_BASE_ROWS: usize = 50_000;
const DEFAULT_MEMTABLE_ROWS: usize = 5_000;
const DEFAULT_BATCH_SIZE: usize = 200;
const DEFAULT_VECTOR_DIM: usize = 128;
const DEFAULT_SAMPLE_SIZE: usize = 30;
const DEFAULT_IVF_PARTITIONS: usize = 16;
const DEFAULT_PQ_SUBVECTORS: usize = 16;

const VECTOR_INDEX_NAME: &str = "vector_idx";

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn dataset_prefix() -> String {
    std::env::var("DATASET_PREFIX").unwrap_or_else(|_| {
        let temp = std::env::temp_dir().join(format!("lance_bench_vec_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp).expect("create temp dir");
        temp.to_string_lossy().to_string()
    })
}

fn storage_label(prefix: &str) -> &'static str {
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

fn create_schema(dim: usize) -> Arc<ArrowSchema> {
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

fn create_batch(schema: &ArrowSchema, start_id: i64, n: usize, dim: usize) -> RecordBatch {
    let ids: Vec<i64> = (start_id..start_id + n as i64).collect();
    let mut vector_builder = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
    for id in &ids {
        for d in 0..dim {
            // Deterministic-looking-but-not-too-clustered values.
            let v = ((*id as f32) * 0.001 + (d as f32) * 0.0001) % 1.0;
            vector_builder.values().append_value(v);
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

/// Build a flat Float32 query vector. `dataset.scan().nearest(...)` expects
/// the query as a flat `Float32Array` for indexed search.
fn create_query_flat(dim: usize) -> Float32Array {
    let v: Vec<f32> = (0..dim).map(|d| 0.5 + (d as f32) * 0.001).collect();
    Float32Array::from(v)
}

/// Build a FixedSizeList query vector. The LSM planner takes a single-row
/// FSL.
fn create_query_fsl(dim: usize) -> FixedSizeListArray {
    let mut b = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
    for d in 0..dim {
        b.values().append_value(0.5 + (d as f32) * 0.001);
    }
    b.append(true);
    b.finish()
}

struct BenchContext {
    base_dataset: Arc<Dataset>,
    lsm_dataset: Arc<Dataset>,
    shard_snapshots: Vec<ShardSnapshot>,
    active_memtable: Option<(Uuid, ActiveMemTableRef)>,
    total_rows: usize,
    vector_dim: usize,
    pk_columns: Vec<String>,
}

async fn build_base_dataset(
    uri: &str,
    schema: Arc<ArrowSchema>,
    base_rows: usize,
    batch_size: usize,
    dim: usize,
    ivf_partitions: usize,
    pq_subvectors: usize,
) -> Dataset {
    let batches: Vec<RecordBatch> = (0..base_rows.div_ceil(batch_size))
        .map(|i| {
            let start = (i * batch_size) as i64;
            let rows = batch_size.min(base_rows - i * batch_size);
            create_batch(&schema, start, rows, dim)
        })
        .collect();
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(reader, uri, Some(WriteParams::default()))
        .await
        .unwrap();

    let ivf_params = IvfBuildParams::new(ivf_partitions);
    let pq_params = PQBuildParams::new(pq_subvectors, 8);
    let params = VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf_params, pq_params);
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some(VECTOR_INDEX_NAME.to_string()),
            &params,
            true,
        )
        .await
        .unwrap();

    dataset
}

async fn setup_benchmark(
    base_rows: usize,
    memtable_rows: usize,
    batch_size: usize,
    dim: usize,
    ivf_partitions: usize,
    pq_subvectors: usize,
    prefix: &str,
) -> BenchContext {
    let schema = create_schema(dim);
    let pk_columns = vec!["id".to_string()];
    let short = &Uuid::new_v4().to_string()[..8];
    let prefix = prefix.trim_end_matches('/');

    let base_uri = format!("{}/base_{}", prefix, short);
    let base_dataset = Arc::new(
        build_base_dataset(
            &base_uri,
            schema.clone(),
            base_rows,
            batch_size,
            dim,
            ivf_partitions,
            pq_subvectors,
        )
        .await,
    );

    let lsm_uri = format!("{}/lsm_{}", prefix, short);
    let mut lsm_dataset = build_base_dataset(
        &lsm_uri,
        schema.clone(),
        base_rows,
        batch_size,
        dim,
        ivf_partitions,
        pq_subvectors,
    )
    .await;
    lsm_dataset
        .initialize_mem_wal(MemWalConfig {
            shard_spec: None,
            maintained_indexes: vec![VECTOR_INDEX_NAME.to_string()],
        })
        .await
        .unwrap();
    let lsm_dataset = Arc::new(lsm_dataset);

    let shard_id = Uuid::new_v4();
    let row_size_estimate = dim * 4 + 8;
    let config = ShardWriterConfig {
        shard_id,
        shard_spec_id: 0,
        durable_write: false,
        sync_indexed_write: false,
        max_memtable_size: memtable_rows * row_size_estimate * 2,
        max_memtable_rows: memtable_rows,
        max_wal_flush_interval: Some(Duration::from_secs(60)),
        ..ShardWriterConfig::default()
    };

    let writer = lsm_dataset
        .as_ref()
        .mem_wal_writer(shard_id, config)
        .await
        .unwrap();

    let is_cloud =
        prefix.starts_with("s3://") || prefix.starts_with("gs://") || prefix.starts_with("az://");
    let flush_wait = if is_cloud {
        Duration::from_secs(8)
    } else {
        Duration::from_millis(800)
    };

    // Generation 1 (will flush)
    let g1_start = base_rows as i64;
    for i in 0..memtable_rows.div_ceil(batch_size) {
        let start = g1_start + (i * batch_size) as i64;
        let rows = batch_size.min(memtable_rows - i * batch_size);
        writer
            .put(vec![create_batch(&schema, start, rows, dim)])
            .await
            .unwrap();
    }
    tokio::time::sleep(flush_wait).await;

    // Generation 2 (will flush)
    let g2_start = g1_start + memtable_rows as i64;
    for i in 0..memtable_rows.div_ceil(batch_size) {
        let start = g2_start + (i * batch_size) as i64;
        let rows = batch_size.min(memtable_rows - i * batch_size);
        writer
            .put(vec![create_batch(&schema, start, rows, dim)])
            .await
            .unwrap();
    }
    tokio::time::sleep(flush_wait).await;

    // Active MemTable (smaller, stays in memory).
    let active_rows = memtable_rows / 2;
    let g3_start = g2_start + memtable_rows as i64;
    for i in 0..active_rows.div_ceil(batch_size) {
        let start = g3_start + (i * batch_size) as i64;
        let rows = batch_size.min(active_rows - i * batch_size);
        writer
            .put(vec![create_batch(&schema, start, rows, dim)])
            .await
            .unwrap();
    }

    let manifest = writer.manifest().await.unwrap();
    let active_memtable_ref = writer.active_memtable_ref().await.unwrap();

    let mut shard_snapshot = ShardSnapshot::new(shard_id);
    if let Some(ref m) = manifest {
        shard_snapshot = shard_snapshot.with_current_generation(m.current_generation);
        for fg in &m.flushed_generations {
            shard_snapshot = shard_snapshot.with_flushed_generation(fg.generation, fg.path.clone());
        }
    }

    let total_rows = base_rows + memtable_rows * 2 + active_rows;
    println!("=== Vector bench setup ===");
    println!("Storage: {} ({})", prefix, storage_label(prefix));
    println!("Base rows: {}", base_rows);
    println!("MemTable rows (per gen): {}", memtable_rows);
    println!("Active MemTable rows: {}", active_rows);
    println!("Vector dim: {}", dim);
    println!("Total LSM rows: {}", total_rows);

    // Keep writer alive so the active MemTable is reachable.
    std::mem::forget(writer);

    BenchContext {
        base_dataset,
        lsm_dataset,
        shard_snapshots: vec![shard_snapshot],
        active_memtable: Some((shard_id, active_memtable_ref)),
        total_rows,
        vector_dim: dim,
        pk_columns,
    }
}

fn bench_vector_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let base_rows = env_usize("BASE_ROWS", DEFAULT_BASE_ROWS);
    let memtable_rows = env_usize("MEMTABLE_ROWS", DEFAULT_MEMTABLE_ROWS);
    let batch_size = env_usize("BATCH_SIZE", DEFAULT_BATCH_SIZE);
    let vector_dim = env_usize("VECTOR_DIM", DEFAULT_VECTOR_DIM);
    let sample_size = env_usize("SAMPLE_SIZE", DEFAULT_SAMPLE_SIZE).max(10);
    let ivf_partitions = env_usize("IVF_PARTITIONS", DEFAULT_IVF_PARTITIONS);
    let pq_subvectors = env_usize("PQ_SUBVECTORS", DEFAULT_PQ_SUBVECTORS);
    let prefix = dataset_prefix();

    let ctx = rt.block_on(setup_benchmark(
        base_rows,
        memtable_rows,
        batch_size,
        vector_dim,
        ivf_partitions,
        pq_subvectors,
        &prefix,
    ));

    let mut group = c.benchmark_group("MemWAL Vector Search");
    group.throughput(Throughput::Elements(10));
    group.sample_size(sample_size);

    let label = format!("{}_rows_{}d", ctx.total_rows, ctx.vector_dim);
    let k = 10;
    let nprobes = 4;

    // Baseline: KNN on base table only.
    group.bench_with_input(BenchmarkId::new("BaseTable_KNN", &label), &(), |b, _| {
        let dataset = ctx.base_dataset.clone();
        let query = create_query_flat(ctx.vector_dim);
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

    // LSM vector search across base + flushed + active MemTables.
    if let Some((shard_id, ref active_memtable)) = ctx.active_memtable {
        let arrow_schema: Arc<ArrowSchema> = Arc::new(ctx.lsm_dataset.schema().into());

        group.bench_with_input(BenchmarkId::new("LSM_KNN", &label), &(), |b, _| {
            let dataset = ctx.lsm_dataset.clone();
            let shard_snapshots = ctx.shard_snapshots.clone();
            let pk_columns = ctx.pk_columns.clone();
            let schema = arrow_schema.clone();
            let active = active_memtable.clone();
            let query = create_query_fsl(ctx.vector_dim);
            b.to_async(&rt).iter(|| {
                let dataset = dataset.clone();
                let shard_snapshots = shard_snapshots.clone();
                let pk_columns = pk_columns.clone();
                let schema = schema.clone();
                let active = active.clone();
                let query = query.clone();
                async move {
                    let collector = LsmDataSourceCollector::new(dataset, shard_snapshots)
                        .with_active_memtable(shard_id, active);
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

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .significance_level(0.05)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_vector_search
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.05);
    targets = bench_vector_search
);

criterion_main!(benches);
