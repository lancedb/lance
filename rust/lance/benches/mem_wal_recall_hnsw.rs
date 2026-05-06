// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! HNSW recall benchmark for the in-memory MemTable index.
//!
//! - Downloads a single shard of `KShivendu/dbpedia-entities-openai-1M`
//!   (1536-dim, OpenAI ada embeddings) from the HF CDN.
//! - Splits into a corpus + a held-out query set.
//! - Loads the corpus into a MemTable via the production `ShardWriter` path,
//!   like `mem_wal_index_micro` does.
//! - For each query: brute-force top-k against the corpus (exact) vs HNSW
//!   top-k via `MemTableScanner::nearest`.
//! - Reports recall@k = |brute ∩ hnsw| / k aggregated across queries.

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
const DATASET_URL: &str = "https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M/resolve/main/data/train-00000-of-00026-3c7b99d1c7eda36e.parquet";
const DIM: usize = 1536;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

async fn download_parquet(local_path: &std::path::Path) -> lance_core::Result<()> {
    if local_path.exists() {
        println!("parquet already cached at {}", local_path.display());
        return Ok(());
    }
    println!("downloading {}", DATASET_URL);
    let resp = reqwest::get(DATASET_URL)
        .await
        .map_err(|e| lance_core::Error::io(format!("download failed: {}", e)))?;
    if !resp.status().is_success() {
        return Err(lance_core::Error::io(format!(
            "download status {}",
            resp.status()
        )));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| lance_core::Error::io(format!("read body: {}", e)))?;
    std::fs::write(local_path, &bytes)
        .map_err(|e| lance_core::Error::io(format!("write: {}", e)))?;
    println!(
        "downloaded {} bytes to {}",
        bytes.len(),
        local_path.display()
    );
    Ok(())
}

/// Read up to `max_rows` rows from the parquet, extracting the embedding
/// column as a contiguous Float32 buffer of length `rows * DIM` plus matching
/// row ids `[0..rows)`.
async fn read_embeddings(
    path: &std::path::Path,
    max_rows: usize,
) -> lance_core::Result<(Vec<f32>, usize)> {
    let file = tokio::fs::File::open(path)
        .await
        .map_err(|e| lance_core::Error::io(format!("open parquet: {}", e)))?;
    let builder = ParquetRecordBatchStreamBuilder::new(file)
        .await
        .map_err(|e| lance_core::Error::io(format!("parquet builder: {}", e)))?;
    let mut stream = builder
        .build()
        .map_err(|e| lance_core::Error::io(format!("parquet stream: {}", e)))?;

    let mut buf: Vec<f32> = Vec::with_capacity(max_rows * DIM);
    let mut rows: usize = 0;

    while rows < max_rows {
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
        let list = col.as_list::<i32>();
        let values = list.values().as_primitive::<Float32Type>();
        for i in 0..rb.num_rows() {
            if rows >= max_rows {
                break;
            }
            let off = list.value_offsets()[i] as usize;
            let len = (list.value_offsets()[i + 1] as usize) - off;
            if len != DIM {
                return Err(lance_core::Error::io(format!(
                    "expected {}-dim embedding, got {}",
                    DIM, len
                )));
            }
            buf.extend_from_slice(&values.values()[off..off + DIM]);
            rows += 1;
        }
    }
    println!("read {} rows × {} dims from parquet", rows, DIM);
    Ok((buf, rows))
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
    // Cohere/OpenAI ada embeddings are unit-normalized; cosine ≡ 1 - dot.
    // Lance uses cosine_distance = 1 - cos_similarity. Use the dispatched fn
    // so the brute force matches what the index reports.
    DistanceType::Cosine.func()(a, b)
}

fn brute_force_top_k(corpus: &[f32], n: usize, query: &[f32], k: usize) -> Vec<(f32, i64)> {
    let mut heap: Vec<(f32, i64)> = (0..n)
        .map(|i| {
            let off = i * DIM;
            (cosine_distance(query, &corpus[off..off + DIM]), i as i64)
        })
        .collect();
    heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    heap.truncate(k);
    heap
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> lance_core::Result<()> {
    let total_rows = env_usize("BENCH_ROWS", 30_000);
    let num_queries = env_usize("BENCH_NUM_QUERIES", 200);
    let k = env_usize("BENCH_K", 10);
    let ef = env_usize("BENCH_EF", 64);

    let total_to_read = total_rows + num_queries;
    println!(
        "=== mem_wal_recall_hnsw === corpus={} queries={} k={} ef={} dim={}",
        total_rows, num_queries, k, ef, DIM
    );

    let cache = std::env::temp_dir().join("mem_wal_recall_hnsw_dbpedia.parquet");
    download_parquet(&cache).await?;
    let (all_vectors, rows) = read_embeddings(&cache, total_to_read).await?;
    if rows < total_to_read {
        return Err(lance_core::Error::io(format!(
            "parquet shard only has {} rows, need {}",
            rows, total_to_read
        )));
    }

    let corpus = &all_vectors[..total_rows * DIM];
    let queries = &all_vectors[total_rows * DIM..(total_rows + num_queries) * DIM];

    // Build a base table + initialize MemWAL, mirroring mem_wal_index_micro.
    let temp = tempfile::tempdir().map_err(|e| lance_core::Error::io(format!("tempdir: {}", e)))?;
    let uri = format!("file://{}/lsm", temp.path().display());
    let schema = make_schema();
    {
        // Tiny base table with enough rows to train an IVF/PQ on the base
        // (the base index doesn't matter for this bench but MemWAL expects
        // the maintained index to exist on the dataset).
        let base_n = 1024usize;
        let base_vec = vec![0.0f32; base_n * DIM];
        let base_batch = make_batch(0, &base_vec, schema.clone());
        let reader = RecordBatchIterator::new(std::iter::once(Ok(base_batch)), schema.clone());
        let mut dataset = Dataset::write(reader, &uri, Some(WriteParams::default())).await?;
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
    }
    let dataset = Arc::new(Dataset::open(&uri).await?);

    let shard_id = Uuid::new_v4();
    let row_size_estimate = DIM * 4 + 8;
    let total_batches_max = total_rows.div_ceil(1000);
    let writer_config = ShardWriterConfig {
        shard_id,
        shard_spec_id: 0,
        durable_write: false,
        sync_indexed_write: true,
        max_memtable_size: total_rows
            .saturating_mul(row_size_estimate)
            .saturating_mul(4),
        max_memtable_rows: total_rows.saturating_mul(2),
        max_memtable_batches: total_batches_max.saturating_mul(2).max(8_000),
        max_wal_flush_interval: Some(Duration::from_millis(200)),
        max_unflushed_memtable_bytes: usize::MAX / 2,
        ..ShardWriterConfig::default()
    };
    let writer = dataset
        .as_ref()
        .mem_wal_writer(shard_id, writer_config)
        .await?;

    // Skip the base-table rows when assigning corpus IDs so the test can
    // distinguish them.
    let id_offset: i64 = 1024 + 1; // after base rows
    let batch_size = 1000;
    let total_batches = total_rows.div_ceil(batch_size);
    let write_start = Instant::now();
    for i in 0..total_batches {
        let start_row = i * batch_size;
        let n = batch_size.min(total_rows - start_row);
        let batch_vec = &corpus[start_row * DIM..(start_row + n) * DIM];
        let batch = make_batch(id_offset + start_row as i64, batch_vec, schema.clone());
        writer.put(vec![batch]).await?;
    }

    // Wait until the index is caught up. Heartbeat a tiny put every ~400ms
    // to drive the in-put time-trigger when the corpus stream is exhausted.
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
    println!(
        "wrote {} rows in {:.2}s (incl. index catchup)",
        total_rows,
        write_start.elapsed().as_secs_f64()
    );

    // Run recall comparison.
    let active = writer.active_memtable_ref().await?;
    let mut recall_sum: f64 = 0.0;
    let mut min_recall: f64 = 1.0;
    let mut bf_total = Duration::ZERO;
    let mut hnsw_total = Duration::ZERO;
    let mut latencies_us: Vec<u128> = Vec::with_capacity(num_queries);

    for q in 0..num_queries {
        let q_off = q * DIM;
        let q_vec = &queries[q_off..q_off + DIM];

        // Brute force.
        let bf_t = Instant::now();
        let bf = brute_force_top_k(corpus, total_rows, q_vec, k);
        bf_total += bf_t.elapsed();

        // HNSW via scanner.
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

        // Extract returned ids from the scanner output. The id column maps
        // back to corpus row index via `id - id_offset`.
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
    }

    latencies_us.sort();
    let median_lat = latencies_us[latencies_us.len() / 2];
    let p99_lat = latencies_us[latencies_us.len() * 99 / 100];
    let mean_recall = recall_sum / (num_queries as f64);

    println!();
    println!("=== RESULTS ===");
    println!(
        "recall@{}: mean={:.4} min={:.4} (over {} queries)",
        k, mean_recall, min_recall, num_queries
    );
    println!(
        "hnsw query latency: median={} us p99={} us total={:.2}s",
        median_lat,
        p99_lat,
        hnsw_total.as_secs_f64()
    );
    println!(
        "brute-force total: {:.2}s (sanity: per-query mean {} us)",
        bf_total.as_secs_f64(),
        bf_total.as_micros() / num_queries as u128
    );

    drop(active);
    writer.close().await?;
    Ok(())
}
