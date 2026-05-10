// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! HNSW recall benchmark for the in-memory MemTable index.
//!
//! - Downloads `KShivendu/dbpedia-entities-openai-1M` parquet shards (1536-dim
//!   OpenAI ada embeddings) from HF as needed to cover the requested
//!   checkpoints.
//! - For each checkpoint size N (default 100k / 500k / 1M):
//!     * Build a fresh MemTable with rows `[0..N)` via the production
//!       ShardWriter path.
//!     * Use 200 held-out queries from rows `[max_checkpoint..max_checkpoint+200)`
//!       (constant across checkpoints so recall comparisons are apples-to-apples).
//!     * For each query: brute-force top-k against the corpus (exact ground
//!       truth) vs HNSW top-k via `MemTableScanner::nearest`.
//!     * Recall@k = |brute ∩ hnsw| / k aggregated over queries.

#![recursion_limit = "256"]
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator,
    cast::AsArray, types::Float32Type,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::TryStreamExt;
use lance::dataset::mem_wal::write::{MemTableScanner, ShardWriterConfig};
use lance::dataset::mem_wal::{DatasetMemWalExt, MemWalConfig};
use lance::dataset::{Dataset, WriteParams};
use lance::index::DatasetIndexExt;
use lance::index::vector::VectorIndexParams;
use lance_index::IndexType;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::builder::PQBuildParams;
use lance_linalg::distance::{DistanceType, MetricType};
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use uuid::Uuid;

const VECTOR_COL: &str = "vector";
const VECTOR_INDEX_NAME: &str = "vector_idx";
const HF_API_LISTING: &str =
    "https://huggingface.co/api/datasets/KShivendu/dbpedia-entities-openai-1M/tree/main/data";
const HF_FILE_BASE: &str =
    "https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M/resolve/main/";
const DIM: usize = 1536;

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

#[derive(serde::Deserialize)]
struct HfTreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
}

async fn list_shard_paths() -> lance_core::Result<Vec<String>> {
    let entries: Vec<HfTreeEntry> = reqwest::get(HF_API_LISTING)
        .await
        .map_err(|e| lance_core::Error::io(format!("listing HTTP: {}", e)))?
        .json()
        .await
        .map_err(|e| lance_core::Error::io(format!("listing JSON: {}", e)))?;
    let mut shards: Vec<String> = entries
        .into_iter()
        .filter(|e| e.kind == "file" && e.path.ends_with(".parquet"))
        .map(|e| e.path)
        .collect();
    shards.sort();
    Ok(shards)
}

async fn download_shard(rel_path: &str, dest: &std::path::Path) -> lance_core::Result<()> {
    if dest.exists() {
        return Ok(());
    }
    let url = format!("{}{}", HF_FILE_BASE, rel_path);
    let max_attempts = 5;
    for attempt in 1..=max_attempts {
        println!(
            "downloading {} (attempt {}/{}) ...",
            rel_path, attempt, max_attempts
        );
        let result: lance_core::Result<bytes::Bytes> = async {
            let resp = reqwest::get(&url)
                .await
                .map_err(|e| lance_core::Error::io(format!("download HTTP: {}", e)))?;
            if !resp.status().is_success() {
                return Err(lance_core::Error::io(format!(
                    "download {} → status {}",
                    url,
                    resp.status()
                )));
            }
            resp.bytes()
                .await
                .map_err(|e| lance_core::Error::io(format!("read body: {}", e)))
        }
        .await;
        match result {
            Ok(bytes) => {
                std::fs::write(dest, &bytes)
                    .map_err(|e| lance_core::Error::io(format!("write: {}", e)))?;
                println!(
                    "  wrote {:.1} MB to {}",
                    bytes.len() as f64 / 1024.0 / 1024.0,
                    dest.display()
                );
                return Ok(());
            }
            Err(e) if attempt < max_attempts => {
                let backoff = Duration::from_secs(2u64.pow(attempt as u32));
                eprintln!(
                    "  attempt {} failed: {}; retrying in {:?}",
                    attempt, e, backoff
                );
                tokio::time::sleep(backoff).await;
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!()
}

/// Read up to `max_rows` rows from a single parquet file, appending to `buf`.
/// Returns the number of rows actually read from this file.
async fn read_shard_into_buf(
    path: &std::path::Path,
    buf: &mut Vec<f32>,
    max_rows: usize,
) -> lance_core::Result<usize> {
    let file = tokio::fs::File::open(path)
        .await
        .map_err(|e| lance_core::Error::io(format!("open parquet: {}", e)))?;
    let builder = ParquetRecordBatchStreamBuilder::new(file)
        .await
        .map_err(|e| lance_core::Error::io(format!("parquet builder: {}", e)))?;
    let mut stream = builder
        .build()
        .map_err(|e| lance_core::Error::io(format!("parquet stream: {}", e)))?;

    let cast_target = arrow_schema::DataType::FixedSizeList(
        Arc::new(Field::new("item", arrow_schema::DataType::Float32, true)),
        DIM as i32,
    );
    let mut rows_from_shard = 0usize;
    while rows_from_shard < max_rows {
        let Some(rb) = stream
            .try_next()
            .await
            .map_err(|e| lance_core::Error::io(format!("parquet read: {}", e)))?
        else {
            break;
        };
        let col = rb
            .column_by_name("openai")
            .ok_or_else(|| lance_core::Error::io("openai column missing".to_string()))?;
        let casted = if col.data_type() == &cast_target {
            col.clone()
        } else {
            arrow_cast::cast::cast(col.as_ref(), &cast_target).map_err(|e| {
                lance_core::Error::io(format!(
                    "cast openai column to FixedSizeList<Float32, {}>: {}",
                    DIM, e
                ))
            })?
        };
        let fsl = casted.as_fixed_size_list();
        let values = fsl
            .values()
            .as_primitive_opt::<Float32Type>()
            .ok_or_else(|| {
                lance_core::Error::io(format!(
                    "fsl values not Float32: {:?}",
                    fsl.values().data_type()
                ))
            })?
            .values();
        for i in 0..rb.num_rows() {
            if rows_from_shard >= max_rows {
                break;
            }
            let off = i * DIM;
            buf.extend_from_slice(&values[off..off + DIM]);
            rows_from_shard += 1;
        }
    }
    Ok(rows_from_shard)
}

async fn load_corpus(needed_rows: usize) -> lance_core::Result<Vec<f32>> {
    let shards = list_shard_paths().await?;
    println!("dataset has {} parquet shards", shards.len());

    let cache_dir = std::env::temp_dir().join("mem_wal_recall_hnsw_cache");
    std::fs::create_dir_all(&cache_dir)
        .map_err(|e| lance_core::Error::io(format!("mkdir cache: {}", e)))?;

    let mut buf: Vec<f32> = Vec::with_capacity(needed_rows * DIM);
    let mut total_rows: usize = 0;

    for rel_path in &shards {
        if total_rows >= needed_rows {
            break;
        }
        let local_name = rel_path.rsplit('/').next().unwrap_or(rel_path);
        let local = cache_dir.join(local_name);
        download_shard(rel_path, &local).await?;
        let want = needed_rows - total_rows;
        let got = read_shard_into_buf(&local, &mut buf, want).await?;
        total_rows += got;
        println!(
            "  shard {} → {} rows (cumulative {})",
            local_name, got, total_rows
        );
    }
    if total_rows < needed_rows {
        println!(
            "  note: dataset exhausted at {} rows (asked {}); caller will adjust",
            total_rows, needed_rows
        );
    }
    Ok(buf)
}

fn make_schema() -> Arc<ArrowSchema> {
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
                DIM as i32,
            ),
            false,
        ),
    ]))
}

fn make_batch(start_id: i64, vectors: &[f32], schema: Arc<ArrowSchema>) -> RecordBatch {
    let n = vectors.len() / DIM;
    let ids: Vec<i64> = (start_id..start_id + n as i64).collect();
    let id_arr = Arc::new(Int64Array::from(ids));
    let inner = Arc::new(Float32Array::from(vectors.to_vec()));
    let inner_field = Arc::new(Field::new("item", DataType::Float32, true));
    let fsl = Arc::new(FixedSizeListArray::try_new(inner_field, DIM as i32, inner, None).unwrap());
    RecordBatch::try_new(schema, vec![id_arr, fsl as ArrayRef]).unwrap()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    DistanceType::Cosine.func()(a, b)
}

fn brute_force_top_k(corpus: &[f32], n: usize, query: &[f32], k: usize) -> Vec<(f32, i64)> {
    let mut all: Vec<(f32, i64)> = (0..n)
        .map(|i| {
            let off = i * DIM;
            (cosine_distance(query, &corpus[off..off + DIM]), i as i64)
        })
        .collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    all.truncate(k);
    all
}

async fn build_base_dataset(uri: &str, schema: Arc<ArrowSchema>) -> lance_core::Result<()> {
    // Tiny base table — content is irrelevant for the recall test (we only
    // query the MemTable), but it can't be all-zero: IVF/PQ training rejects
    // zero-norm vectors under cosine. Use a deterministic non-zero pattern.
    let base_n = 1024usize;
    let mut base_vec = Vec::with_capacity(base_n * DIM);
    for i in 0..base_n {
        for d in 0..DIM {
            base_vec.push((((i * 31 + d) as f32) * 0.000_173_f32).sin());
        }
    }
    let base_batch = make_batch(0, &base_vec, schema.clone());
    let reader = RecordBatchIterator::new(std::iter::once(Ok(base_batch)), schema.clone());
    let mut dataset = Dataset::write(reader, uri, Some(WriteParams::default())).await?;
    let ivf = IvfBuildParams::new(16);
    let pq = PQBuildParams::new(16, 8);
    let params = VectorIndexParams::with_ivf_pq_params(MetricType::Cosine, ivf, pq);
    dataset
        .create_index(
            &[VECTOR_COL],
            IndexType::Vector,
            Some(VECTOR_INDEX_NAME.to_string()),
            &params,
            true,
        )
        .await?;
    dataset
        .initialize_mem_wal(MemWalConfig {
            shard_spec: None,
            maintained_indexes: vec![VECTOR_INDEX_NAME.to_string()],
        })
        .await?;
    Ok(())
}

struct CheckpointResult {
    rows: usize,
    write_wall: Duration,
    mean_recall: f64,
    min_recall: f64,
    median_query_us: u128,
    p99_query_us: u128,
    bf_total: Duration,
    hnsw_total: Duration,
}

async fn run_checkpoint(
    cp: usize,
    corpus: &[f32],
    queries: &[f32],
    num_queries: usize,
    k: usize,
    schema: Arc<ArrowSchema>,
) -> lance_core::Result<CheckpointResult> {
    println!("\n=== checkpoint rows={} ===", cp);
    let temp = tempfile::tempdir().map_err(|e| lance_core::Error::io(format!("tempdir: {}", e)))?;
    let uri = format!("file://{}/lsm", temp.path().display());
    build_base_dataset(&uri, schema.clone()).await?;
    let dataset = Arc::new(Dataset::open(&uri).await?);

    let shard_id = Uuid::new_v4();
    let row_size_estimate = DIM * 4 + 8;
    let total_batches_max = cp.div_ceil(1000);
    let writer_config = ShardWriterConfig {
        shard_id,
        shard_spec_id: 0,
        durable_write: false,
        sync_indexed_write: true,
        max_memtable_size: cp.saturating_mul(row_size_estimate).saturating_mul(4),
        max_memtable_rows: cp.saturating_mul(2),
        max_memtable_batches: total_batches_max.saturating_mul(2).max(8_000),
        max_wal_flush_interval: Some(Duration::from_millis(200)),
        max_unflushed_memtable_bytes: usize::MAX / 2,
        ..ShardWriterConfig::default()
    };
    let writer = dataset
        .as_ref()
        .mem_wal_writer(shard_id, writer_config)
        .await?;

    // Skip the base-table rows (1024) when assigning corpus IDs.
    let id_offset: i64 = 1024 + 1;
    let batch_size = 1000;
    let total_batches = cp.div_ceil(batch_size);
    let write_start = Instant::now();
    for i in 0..total_batches {
        let start_row = i * batch_size;
        let n = batch_size.min(cp - start_row);
        let batch_vec = &corpus[start_row * DIM..(start_row + n) * DIM];
        let batch = make_batch(id_offset + start_row as i64, batch_vec, schema.clone());
        writer.put(vec![batch]).await?;
    }

    let target_batch_pos = total_batches.saturating_sub(1);
    let mut spins = 0u64;
    loop {
        let active = writer.active_memtable_ref().await?;
        if active.index_store.max_indexed_batch_position() >= target_batch_pos {
            break;
        }
        drop(active);
        tokio::time::sleep(Duration::from_millis(50)).await;
        spins += 1;
        if spins.is_multiple_of(8) {
            let dummy_vec = vec![0.0f32; DIM];
            let dummy = make_batch(-1 - spins as i64, &dummy_vec, schema.clone());
            writer.put(vec![dummy]).await?;
        }
    }
    let write_wall = write_start.elapsed();
    println!(
        "  wrote {} rows in {:.2}s (incl. index catchup)",
        cp,
        write_wall.as_secs_f64()
    );
    use std::io::Write;
    std::io::stdout().flush().ok();

    let active = writer.active_memtable_ref().await?;
    let mut recall_sum: f64 = 0.0;
    let mut min_recall: f64 = 1.0;
    let mut bf_total = Duration::ZERO;
    let mut hnsw_total = Duration::ZERO;
    let mut latencies_us: Vec<u128> = Vec::with_capacity(num_queries);
    let recall_phase_start = Instant::now();

    for q in 0..num_queries {
        let q_off = q * DIM;
        let q_vec = &queries[q_off..q_off + DIM];

        let bf_t = Instant::now();
        let bf = brute_force_top_k(corpus, cp, q_vec, k);
        bf_total += bf_t.elapsed();

        let inner = Arc::new(Float32Array::from(q_vec.to_vec()));
        let inner_field = Arc::new(Field::new("item", DataType::Float32, true));
        let q_fsl = FixedSizeListArray::try_new(inner_field, DIM as i32, inner, None).unwrap();

        let mut scanner = MemTableScanner::new(
            active.batch_store.clone(),
            active.index_store.clone(),
            active.schema.clone(),
        );
        let q_arr: ArrayRef = Arc::new(q_fsl);
        scanner.nearest(VECTOR_COL, q_arr, k);
        let h_t = Instant::now();
        let stream = scanner.try_into_stream().await?;
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        let elapsed = h_t.elapsed();
        hnsw_total += elapsed;
        latencies_us.push(elapsed.as_micros());

        let mut hnsw_ids: Vec<i64> = Vec::with_capacity(k);
        for b in &batches {
            let id_col = b
                .column_by_name("id")
                .ok_or_else(|| lance_core::Error::invalid_input("id missing"))?
                .as_primitive::<arrow_array::types::Int64Type>();
            for i in 0..id_col.len() {
                hnsw_ids.push(id_col.value(i) - id_offset);
            }
        }

        let bf_set: std::collections::HashSet<i64> = bf.iter().map(|(_, id)| *id).collect();
        let hits = hnsw_ids.iter().filter(|id| bf_set.contains(id)).count();
        let recall = (hits as f64) / (k as f64);
        recall_sum += recall;
        if recall < min_recall {
            min_recall = recall;
        }
        if (q + 1) % 25 == 0 || q + 1 == num_queries {
            println!(
                "    progress: {}/{} queries (running mean recall = {:.4})",
                q + 1,
                num_queries,
                recall_sum / (q + 1) as f64
            );
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }

    let recall_phase_wall = recall_phase_start.elapsed();
    latencies_us.sort();
    let median_q = latencies_us[latencies_us.len() / 2];
    let p99_q = latencies_us[latencies_us.len() * 99 / 100];
    let mean_recall = recall_sum / num_queries as f64;
    println!(
        "  recall phase: cp={} mean_recall={:.4} min_recall={:.4} q_median_us={} q_p99_us={} bf_s={:.2} hnsw_s={:.2} wall={:.2}s",
        cp,
        mean_recall,
        min_recall,
        median_q,
        p99_q,
        bf_total.as_secs_f64(),
        hnsw_total.as_secs_f64(),
        recall_phase_wall.as_secs_f64()
    );
    std::io::stdout().flush().ok();

    drop(active);
    writer.close().await?;
    Ok(CheckpointResult {
        rows: cp,
        write_wall,
        mean_recall,
        min_recall,
        median_query_us: median_q,
        p99_query_us: p99_q,
        bf_total,
        hnsw_total,
    })
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> lance_core::Result<()> {
    let mut checkpoints = env_checkpoints();
    let num_queries = env_usize("BENCH_NUM_QUERIES", 200);
    let k = env_usize("BENCH_K", 10);
    let ef = env_usize("BENCH_EF", 64);

    // Probe how many rows the dataset actually contains by listing shards
    // and estimating from the file count, then read all of them. The DBpedia
    // dataset is exactly 1,000,000 rows across 26 shards; we want to leave
    // room for `num_queries` queries at the tail.
    println!(
        "=== mem_wal_recall_hnsw === checkpoints={:?} num_queries={} k={} ef={} dim={}",
        checkpoints, num_queries, k, ef, DIM
    );

    // Pull the entire dataset; queries come from the last `num_queries` rows
    // and corpus from `[0, total - num_queries)`. Any checkpoint larger than
    // the available corpus size is clamped down.
    let max_cp_requested = *checkpoints.iter().max().unwrap_or(&1_000_000);
    // load_corpus returns exactly the rows it could read, capped at max
    // available; ask for max_cp_requested + num_queries up front so the
    // common case (max_cp + queries <= dataset) avoids extra shards.
    let all_vectors = load_corpus(max_cp_requested + num_queries).await?;
    let total_rows_loaded = all_vectors.len() / DIM;
    let max_corpus = total_rows_loaded.saturating_sub(num_queries);
    println!(
        "loaded {} rows × {} dim ({:.2} GB); corpus available = {}, queries = last {} rows",
        total_rows_loaded,
        DIM,
        (total_rows_loaded * DIM * 4) as f64 / 1024.0 / 1024.0 / 1024.0,
        max_corpus,
        num_queries
    );
    // Clamp any checkpoint that exceeds available corpus rows.
    for cp in checkpoints.iter_mut() {
        if *cp > max_corpus {
            println!("  clamping checkpoint {} → {}", cp, max_corpus);
            *cp = max_corpus;
        }
    }
    let corpus = &all_vectors[..max_corpus * DIM];
    let queries = &all_vectors
        [max_corpus * DIM..(max_corpus + num_queries.min(total_rows_loaded - max_corpus)) * DIM];

    let schema = make_schema();
    let mut results: Vec<CheckpointResult> = Vec::with_capacity(checkpoints.len());
    for &cp in &checkpoints {
        let r = run_checkpoint(cp, corpus, queries, num_queries, k, schema.clone()).await?;
        results.push(r);
    }

    println!("\n=== RESULTS ===");
    println!(
        "{:<10} {:>14} {:>12} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "rows",
        "write_wall_s",
        "mean_recall",
        "min_recall",
        "q_median_us",
        "q_p99_us",
        "bf_s",
        "hnsw_s"
    );
    for r in &results {
        println!(
            "{:<10} {:>14.2} {:>12.4} {:>12.4} {:>12} {:>12} {:>10.2} {:>10.2}",
            r.rows,
            r.write_wall.as_secs_f64(),
            r.mean_recall,
            r.min_recall,
            r.median_query_us,
            r.p99_query_us,
            r.bf_total.as_secs_f64(),
            r.hnsw_total.as_secs_f64(),
        );
    }
    println!("=== DONE ===");
    Ok(())
}
