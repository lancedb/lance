// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
//
// TakeExec vs FilteredReadExec (mask mode) vs FilteredReadExec (take mode),
// apples-to-apples.
//
// All three materialize the SAME projection for the SAME row addresses per
// query.  A fixed list of queries (each = N random row addrs out of the
// dataset) is replayed identically against each exec.  We report QPS and
// P50/P90/P99 latency (a real latency harness, not criterion's mean).
//
// Arms:
//   - TakeExec:               the classic point-lookup take
//   - FilteredReadExec/mask:  row addrs serialized into an IndexExprResult
//                             batch (the `_rowid IN (...)` path)
//   - FilteredReadExec/take:  row addrs streamed as a `_rowaddr` record batch
//                             (the new take mode that Scanner::take now plans)
//
// Run:
//   LANCE_BENCH_DATASET=/path/to/dataset.lance \
//   LANCE_BENCH_COLUMN=full_content \
//   cargo run --profile release-with-debug --bench take_exec_compare
//
// Env:
//   LANCE_BENCH_DATASET      path to the .lance dataset (required)
//   LANCE_BENCH_COLUMN       column to take (default: full_content)
//   LANCE_BENCH_NQUERIES     number of queries in the fixed list (default: 1000)
//   LANCE_BENCH_ROWS         rows per take (default: 100)
//   LANCE_BENCH_WARMUP       warmup queries per exec (default: 100)
//   LANCE_BENCH_CREATE_ROWS  if set and the dataset does not exist, create a
//                            synthetic dataset with this many rows (columns:
//                            rating Int32, full_content ~1KiB Utf8), 100k rows
//                            per fragment

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::io::exec::TakeExec;
use lance::io::exec::filtered_read::{FilteredReadExec, FilteredReadOptions};
use lance_core::ROW_ADDR;
use lance_core::datatypes::{OnMissing, Projection};
use lance_datafusion::exec::{LanceExecutionOptions, OneShotExec, execute_plan};
use lance_select::result::IndexExprResultWireFormat;
use lance_select::{IndexExprResult, RowAddrMask, RowAddrTreeMap};
use lance_table::format::Fragment;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Build `n` queries, each a Vec of `rows_per` random row addresses
/// `(frag<<32)|offset` drawn size-weighted across the dataset's fragments.
async fn build_queries(dataset: &Dataset, n: usize, rows_per: usize, seed: u64) -> Vec<Vec<u64>> {
    let frags: Vec<&Fragment> = dataset.manifest().fragments.iter().collect();
    let mut frag_sizes: Vec<(u64, u64)> = Vec::with_capacity(frags.len());
    for f in &frags {
        let n = dataset
            .get_fragment(f.id as usize)
            .unwrap()
            .physical_rows()
            .await
            .unwrap_or(0) as u64;
        if n > 0 {
            frag_sizes.push((f.id, n));
        }
    }
    let total: u64 = frag_sizes.iter().map(|(_, n)| *n).sum();
    println!(
        "dataset: {} fragments, {} rows total",
        frag_sizes.len(),
        total
    );

    let mut rng = StdRng::seed_from_u64(seed);
    let mut queries = Vec::with_capacity(n);
    for _ in 0..n {
        let mut q = Vec::with_capacity(rows_per);
        for _ in 0..rows_per {
            let mut target = rng.random_range(0..total);
            for (fid, fsz) in &frag_sizes {
                if target < *fsz {
                    q.push((fid << 32) | target);
                    break;
                }
                target -= *fsz;
            }
        }
        queries.push(q);
    }
    queries
}

fn pctl(sorted_us: &[u128], p: f64) -> f64 {
    if sorted_us.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_us.len() as f64) * p) as usize;
    let idx = idx.min(sorted_us.len() - 1);
    sorted_us[idx] as f64 / 1000.0 // -> ms
}

fn projection(dataset: &Arc<Dataset>, column: &str) -> Projection {
    dataset
        .empty_projection()
        .union_column(column, OnMissing::Error)
        .unwrap()
}

/// An input plan emitting one `_rowaddr` batch (drives TakeExec and the
/// take-mode FilteredReadExec)
fn row_addr_input(addrs: Vec<u64>) -> Arc<OneShotExec> {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        ROW_ADDR,
        DataType::UInt64,
        true,
    )]));
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(UInt64Array::from(addrs))]).unwrap();
    let stream = futures::stream::once(async move { Ok(batch) });
    Arc::new(OneShotExec::new(Box::pin(RecordBatchStreamAdapter::new(
        schema, stream,
    ))))
}

/// An input plan emitting one serialized IndexExprResult mask batch (drives
/// the mask-mode FilteredReadExec, like `_rowid IN (...)`)
fn mask_input(dataset: &Arc<Dataset>, addrs: Vec<u64>) -> Arc<OneShotExec> {
    let fragments_covered: roaring::RoaringBitmap =
        dataset.fragments().iter().map(|f| f.id as u32).collect();
    let mask = RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(addrs));
    let batch = IndexExprResult::exact(mask)
        .serialize(&fragments_covered, IndexExprResultWireFormat::TwoMask)
        .unwrap();
    let schema = batch.schema();
    let stream = futures::stream::once(async move { Ok(batch) });
    Arc::new(OneShotExec::new(Box::pin(RecordBatchStreamAdapter::new(
        schema, stream,
    ))))
}

/// Replay all queries against one exec builder, timing each.
async fn run_bench<F>(
    name: &str,
    dataset: &Arc<Dataset>,
    queries: &[Vec<u64>],
    warmup: usize,
    build: F,
) where
    F: Fn(&Arc<Dataset>, Vec<u64>) -> Arc<dyn datafusion_physical_plan::ExecutionPlan>,
{
    // Warmup (populate OS/page cache, session caches).
    for q in queries.iter().take(warmup) {
        let plan = build(dataset, q.clone());
        let stream = execute_plan(plan, LanceExecutionOptions::default()).unwrap();
        let _ = stream.try_collect::<Vec<_>>().await.unwrap();
    }

    let mut lats = Vec::with_capacity(queries.len());
    let mut total_rows = 0usize;
    let wall = Instant::now();
    for q in queries {
        let plan = build(dataset, q.clone());
        let t = Instant::now();
        let stream = execute_plan(plan, LanceExecutionOptions::default()).unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        lats.push(t.elapsed().as_micros());
        total_rows += batches.iter().map(|b| b.num_rows()).sum::<usize>();
    }
    let wall = wall.elapsed().as_secs_f64();

    lats.sort_unstable();
    let qps = queries.len() as f64 / wall;
    println!(
        "\n=== {name} ===\n  queries={} rows/take~{} total_rows={} wall={:.2}s\n  QPS={:.1}\n  P50={:.2}ms  P90={:.2}ms  P99={:.2}ms  max={:.2}ms  mean={:.2}ms",
        queries.len(),
        queries.first().map(|q| q.len()).unwrap_or(0),
        total_rows,
        wall,
        qps,
        pctl(&lats, 0.50),
        pctl(&lats, 0.90),
        pctl(&lats, 0.99),
        pctl(&lats, 1.0),
        lats.iter().sum::<u128>() as f64 / lats.len() as f64 / 1000.0,
    );
}

async fn maybe_create_dataset(path: &str) {
    let Ok(rows) = std::env::var("LANCE_BENCH_CREATE_ROWS") else {
        return;
    };
    if std::path::Path::new(path).exists() {
        return;
    }
    let rows: u64 = rows
        .parse()
        .expect("LANCE_BENCH_CREATE_ROWS must be a number");
    println!("creating synthetic dataset at {path} with {rows} rows");
    use lance_datagen::{BatchCount, ByteCount, RowCount, array, gen_batch};
    let reader = gen_batch()
        .col("rating", array::step::<arrow_array::types::Int32Type>())
        .col(
            "full_content",
            array::rand_utf8(ByteCount::from(1024), false),
        )
        .into_reader_rows(
            RowCount::from(100_000),
            BatchCount::from((rows / 100_000).max(1) as u32),
        );
    let params = lance::dataset::WriteParams {
        max_rows_per_file: 100_000,
        ..Default::default()
    };
    Dataset::write(reader, path, Some(params)).await.unwrap();
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let path =
        std::env::var("LANCE_BENCH_DATASET").expect("set LANCE_BENCH_DATASET to the .lance path");
    let column = std::env::var("LANCE_BENCH_COLUMN").unwrap_or_else(|_| "full_content".into());
    let nqueries = env_usize("LANCE_BENCH_NQUERIES", 1000);
    let rows_per = env_usize("LANCE_BENCH_ROWS", 100);
    let warmup = env_usize("LANCE_BENCH_WARMUP", 100);

    maybe_create_dataset(&path).await;

    let cache = 8usize * 1024 * 1024 * 1024;
    let dataset = Arc::new(
        DatasetBuilder::from_uri(&path)
            .with_index_cache_size_bytes(cache)
            .with_metadata_cache_size_bytes(cache)
            .load()
            .await
            .expect("open dataset"),
    );
    println!(
        "opened {path}\n  column={column} nqueries={nqueries} rows/take={rows_per} warmup={warmup}"
    );

    let queries = build_queries(&dataset, nqueries, rows_per, 42).await;

    let col = column.clone();
    let build_take_exec = move |ds: &Arc<Dataset>, addrs: Vec<u64>| {
        Arc::new(
            TakeExec::try_new(ds.clone(), row_addr_input(addrs), projection(ds, &col))
                .unwrap()
                .unwrap(),
        ) as Arc<dyn datafusion_physical_plan::ExecutionPlan>
    };
    let col = column.clone();
    let build_mask_mode = move |ds: &Arc<Dataset>, addrs: Vec<u64>| {
        let input = mask_input(ds, addrs);
        Arc::new(
            FilteredReadExec::try_new(
                ds.clone(),
                FilteredReadOptions::new(projection(ds, &col)),
                Some(input),
            )
            .unwrap(),
        ) as Arc<dyn datafusion_physical_plan::ExecutionPlan>
    };
    let col = column.clone();
    let build_take_mode = move |ds: &Arc<Dataset>, addrs: Vec<u64>| {
        Arc::new(
            FilteredReadExec::try_new(
                ds.clone(),
                FilteredReadOptions::new(projection(ds, &col)),
                Some(row_addr_input(addrs)),
            )
            .unwrap(),
        ) as Arc<dyn datafusion_physical_plan::ExecutionPlan>
    };

    run_bench("TakeExec", &dataset, &queries, warmup, build_take_exec).await;
    run_bench(
        "FilteredReadExec (mask mode)",
        &dataset,
        &queries,
        warmup,
        build_mask_mode,
    )
    .await;
    run_bench(
        "FilteredReadExec (take mode)",
        &dataset,
        &queries,
        warmup,
        build_take_mode,
    )
    .await;
}
