// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Microbench for the in-memory MemTable vector index.
//!
//! Two measurements:
//!
//! 1. **Pure MemTable write + query**: insert N rows into a single MemTable
//!    (no flush). Record cumulative insert time and median query latency at
//!    several fill checkpoints. The MemTable is configured with one vector
//!    index — HNSW on `jack/mem-wal-hnsw`, IVF-PQ on `main` — and that index
//!    is the one being measured.
//!
//! 2. **Flush time**: explicitly flush a fully-populated MemTable to local
//!    disk via `MemTableFlusher::flush_with_indexes`. Measure wall time at
//!    each fill checkpoint.
//!
//! ## Configuration (env vars)
//!
//! - `BENCH_DIM` — vector dimension (default 1024)
//! - `BENCH_BATCH` — rows per insert batch (default 1000)
//! - `BENCH_NUM_QUERIES` — queries to run per checkpoint (default 50)
//! - `BENCH_CHECKPOINTS` — comma-separated row counts (default
//!   `100000,500000,1000000`)
//!
//! Output is plain stdout, one section per checkpoint. Captured by the
//! runner script for comparison.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{
    ArrayRef, FixedSizeListArray, Int64Array, RecordBatch,
    builder::{FixedSizeListBuilder, Float32Builder},
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use lance::dataset::mem_wal::ShardManifestStore;
use lance::dataset::mem_wal::write::{
    HnswIndexConfig, IndexStore, MemTable, MemTableFlusher, MemTableScanner, MemIndexConfig,
};
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_io::object_store::ObjectStore;
use lance_linalg::distance::DistanceType;
use uuid::Uuid;

const VECTOR_COL: &str = "vector";

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_checkpoints() -> Vec<usize> {
    let raw =
        std::env::var("BENCH_CHECKPOINTS").unwrap_or_else(|_| "100000,500000,1000000".to_string());
    raw.split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect()
}

fn schema(dim: usize) -> Arc<ArrowSchema> {
    use std::collections::HashMap;
    let mut id_meta = HashMap::new();
    id_meta.insert(
        "lance-schema:unenforced-primary-key".to_string(),
        "true".to_string(),
    );
    let id = Field::new("id", DataType::Int64, false).with_metadata(id_meta);
    Arc::new(ArrowSchema::new(vec![
        id,
        Field::new(
            VECTOR_COL,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]))
}

fn make_batch(start_id: i64, n: usize, dim: usize) -> RecordBatch {
    let s = schema(dim);
    let ids: Vec<i64> = (start_id..start_id + n as i64).collect();
    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
    for &id in &ids {
        for d in 0..dim {
            // Deterministic but spread out so distances vary.
            let v = ((id as f32) * 0.000_173 + (d as f32) * 0.000_011).fract();
            builder.values().append_value(v);
        }
        builder.append(true);
    }
    RecordBatch::try_new(
        s,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(builder.finish()),
        ],
    )
    .unwrap()
}

fn make_query_fsl(dim: usize, seed: u64) -> FixedSizeListArray {
    let mut b = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32);
    for d in 0..dim {
        let v = (((seed.wrapping_mul(2654435761)) as f32) * 1e-9 + (d as f32) * 7e-5).fract();
        b.values().append_value(v);
    }
    b.append(true);
    b.finish()
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> lance_core::Result<()> {
    let dim = env_usize("BENCH_DIM", 1024);
    let batch_size = env_usize("BENCH_BATCH", 1000);
    let num_queries = env_usize("BENCH_NUM_QUERIES", 50);
    let checkpoints = env_checkpoints();
    let max_rows = *checkpoints.iter().max().unwrap_or(&1_000_000);

    println!("=== mem_wal_index_micro [HNSW] ===");
    println!(
        "dim={} batch={} num_queries={} checkpoints={:?} (HNSW backend)",
        dim, batch_size, num_queries, checkpoints
    );

    // ---- Setup ----
    let s = schema(dim);
    let index_configs = vec![MemIndexConfig::Hnsw(Box::new(
        HnswIndexConfig::new(
            "vector_idx".to_string(),
            1,
            VECTOR_COL.to_string(),
            DistanceType::L2,
        )
        .with_build_params(HnswBuildParams::default()),
    ))];

    let mut memtable = MemTable::new(s.clone(), 1, vec![]).unwrap();
    let registry = IndexStore::from_configs(&index_configs, max_rows).unwrap();
    memtable.set_indexes(registry);

    // ---- Insert + query loop ----
    let mut total_inserted: usize = 0;
    let mut total_insert_wall = Duration::ZERO;
    let mut next_cp_idx = 0;
    let mut wal_position: u64 = 0;

    let total_batches = max_rows.div_ceil(batch_size);
    println!("write phase: {} batches of {} rows", total_batches, batch_size);

    for i in 0..total_batches {
        let start = (i * batch_size) as i64;
        let rows = batch_size.min(max_rows - i * batch_size);
        let batch = make_batch(start, rows, dim);
        let t = Instant::now();
        let frag_id = memtable.insert(batch).await?;
        // Pretend WAL flushed each batch immediately so flush() will succeed.
        memtable.mark_wal_flushed(&[frag_id], wal_position + 1, &[i]);
        wal_position += 1;
        total_insert_wall += t.elapsed();
        total_inserted += rows;

        // Hit checkpoint?
        while next_cp_idx < checkpoints.len() && total_inserted >= checkpoints[next_cp_idx] {
            let cp = checkpoints[next_cp_idx];
            let throughput = cp as f64 / total_insert_wall.as_secs_f64();
            println!(
                "[checkpoint] rows={} cumulative_insert_time_ms={:.1} throughput_rows_per_sec={:.1}",
                cp,
                total_insert_wall.as_millis(),
                throughput
            );

            // Query phase: median latency over `num_queries` queries.
            let scanner_indexes = memtable
                .indexes_arc()
                .expect("indexes registered above");
            let bs = memtable.batch_store();
            let scanner_schema = memtable.schema().clone();

            let mut latencies = Vec::with_capacity(num_queries);
            for q in 0..num_queries {
                let q_fsl = make_query_fsl(dim, q as u64);
                let mut scanner = MemTableScanner::new(
                    bs.clone(),
                    scanner_indexes.clone(),
                    scanner_schema.clone(),
                );
                let q_arr: ArrayRef = Arc::new(q_fsl);
                scanner.nearest(VECTOR_COL, q_arr, 10);
                let t = Instant::now();
                let stream = scanner.try_into_stream().await?;
                let _: Vec<RecordBatch> = stream.try_collect().await?;
                latencies.push(t.elapsed());
            }
            latencies.sort();
            let median = latencies[latencies.len() / 2];
            let p99 = latencies[latencies.len() * 99 / 100];
            println!(
                "[checkpoint] rows={} query_median_us={} query_p99_us={} num_queries={}",
                cp,
                median.as_micros(),
                p99.as_micros(),
                num_queries
            );

            next_cp_idx += 1;
        }
    }

    println!(
        "write phase done: total_rows={} wall={:.2}s overall_throughput={:.0} rows/sec",
        total_inserted,
        total_insert_wall.as_secs_f64(),
        total_inserted as f64 / total_insert_wall.as_secs_f64()
    );

    // ---- Flush phase: time MemTableFlusher::flush_with_indexes ----
    // We can only flush the whole memtable once, so re-create it at each
    // checkpoint to measure flush time at that fill level.
    println!("flush phase:");
    for &cp in &checkpoints {
        let elapsed = measure_flush(cp, dim, batch_size, &index_configs).await?;
        println!(
            "[flush] rows={} flush_wall_ms={} throughput_rows_per_sec={:.0}",
            cp,
            elapsed.as_millis(),
            cp as f64 / elapsed.as_secs_f64()
        );
    }

    println!("=== DONE ===");
    Ok(())
}

async fn measure_flush(
    cp: usize,
    dim: usize,
    batch_size: usize,
    index_configs: &[MemIndexConfig],
) -> lance_core::Result<Duration> {
    let s = schema(dim);
    let mut memtable = MemTable::new(s.clone(), 1, vec![]).unwrap();
    let registry = IndexStore::from_configs(index_configs, cp).unwrap();
    memtable.set_indexes(registry);

    let total_batches = cp.div_ceil(batch_size);
    let mut wal_pos: u64 = 0;
    for i in 0..total_batches {
        let start = (i * batch_size) as i64;
        let rows = batch_size.min(cp - i * batch_size);
        let batch = make_batch(start, rows, dim);
        let frag_id = memtable.insert(batch).await?;
        memtable.mark_wal_flushed(&[frag_id], wal_pos + 1, &[i]);
        wal_pos += 1;
    }

    let temp_dir = tempfile::tempdir()
        .map_err(|e| lance_core::Error::io(format!("tempdir: {}", e)))?;
    let temp_path: PathBuf = temp_dir.path().to_path_buf();
    let uri = format!("file://{}", temp_path.display());
    let (store, base_path) = ObjectStore::from_uri(&uri).await?;
    let shard_id = Uuid::new_v4();
    let manifest_store = Arc::new(ShardManifestStore::new(store.clone(), &base_path, shard_id, 2));
    let (epoch, _) = manifest_store.claim_epoch(0).await?;
    let flusher = MemTableFlusher::new(store, base_path, uri, shard_id, manifest_store);

    let t = Instant::now();
    let _result = flusher
        .flush_with_indexes(&memtable, epoch, index_configs)
        .await?;
    Ok(t.elapsed())
}
