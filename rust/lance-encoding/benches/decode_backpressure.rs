// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark comparing bounded vs unbounded decode channels.
//!
//! Measures:
//! - Throughput (rows/sec) at different parallelism levels
//! - Peak memory (bytes) at different parallelism levels
//!
//! Run with: cargo bench --bench decode_backpressure
//! Results are printed as CSV for easy plotting.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{Array, Float32Array};
use arrow_schema::{DataType, Field, Schema};
use futures::StreamExt;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::cache::LanceCache;
use lance_encoding::{
    decoder::{
        schedule_and_decode, DecoderConfig, DecoderPlugins, FilterExpression, RequestedRows,
        SchedulerDecoderConfig,
    },
    encoder::{default_encoding_strategy, encode_batch, EncodingOptions},
    version::LanceFileVersion,
    EncodingsIo,
};
use lance_linalg::distance::l2_distance_arrow_batch;
use peak_alloc::PeakAlloc;

#[global_allocator]
static PEAK_ALLOC: PeakAlloc = PeakAlloc;

// Benchmark parameters
const NUM_ROWS: u64 = 100_000;
const DIMS: i32 = 1024;
const BATCH_SIZE: u32 = 1024;
const PARALLELISM_LEVELS: &[usize] = &[4];
const WARMUP_ITERS: usize = 1;
const BENCH_ITERS: usize = 2;
// I/O latencies to simulate: local NVMe (0ms), various cloud latencies
const IO_LATENCIES_MS: &[u64] = &[0, 1, 5, 50];

/// A wrapper around BufferScheduler that clones data on each read
/// and simulates storage latency.
///
/// The latency model simulates parallel I/O: multiple concurrent requests
/// share the same latency window. We track the "last request time" and only
/// add latency if enough time has passed since the last batch of requests.
/// This models real cloud storage where parallel requests overlap.
#[derive(Debug)]
struct CloningIoScheduler {
    inner: lance_encoding::BufferScheduler,
    latency: Duration,
    request_count: AtomicUsize,
    range_count: AtomicUsize,
    total_bytes: AtomicUsize,
}

impl CloningIoScheduler {
    fn with_latency(data: bytes::Bytes, latency: Duration) -> Self {
        Self {
            inner: lance_encoding::BufferScheduler::new(data),
            latency,
            request_count: AtomicUsize::new(0),
            range_count: AtomicUsize::new(0),
            total_bytes: AtomicUsize::new(0),
        }
    }

    fn request_count(&self) -> usize {
        self.request_count.load(Ordering::Relaxed)
    }

    fn range_count(&self) -> usize {
        self.range_count.load(Ordering::Relaxed)
    }

    fn total_bytes(&self) -> usize {
        self.total_bytes.load(Ordering::Relaxed)
    }
}

impl EncodingsIo for CloningIoScheduler {
    fn submit_request(
        &self,
        ranges: Vec<std::ops::Range<u64>>,
        priority: u64,
    ) -> futures::future::BoxFuture<'static, lance_core::Result<Vec<bytes::Bytes>>> {
        self.request_count.fetch_add(1, Ordering::Relaxed);
        self.range_count.fetch_add(ranges.len(), Ordering::Relaxed);
        let bytes_requested: u64 = ranges.iter().map(|r| r.end - r.start).sum();
        self.total_bytes.fetch_add(bytes_requested as usize, Ordering::Relaxed);

        let fut = self.inner.submit_request(ranges, priority);
        let latency = self.latency;

        Box::pin(async move {
            // Simulate I/O latency. When multiple requests are awaited concurrently,
            // these sleeps overlap naturally (async parallelism).
            if !latency.is_zero() {
                tokio::time::sleep(latency).await;
            }
            let buffers = fut.await?;

            Ok(buffers
                .into_iter()
                .map(|b| bytes::Bytes::copy_from_slice(&b))
                .collect())
        })
    }
}

/// Compute L2 distance between a query vector and all vectors in a batch.
/// Uses SIMD-optimized implementation from lance-linalg.
fn compute_l2_distances(
    batch: &arrow_array::RecordBatch,
    query: &Float32Array,
) -> Arc<Float32Array> {
    let vector_col = batch
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::FixedSizeListArray>()
        .unwrap();

    l2_distance_arrow_batch(query, vector_col).unwrap()
}

/// Generate a batch of f32 vectors (similar to embedding vectors in KNN workloads)
fn generate_vector_batch(num_rows: usize, dims: i32) -> arrow_array::RecordBatch {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let values: Vec<f32> = (0..num_rows * dims as usize)
        .map(|_| rng.random::<f32>())
        .collect();

    let values_array = arrow_array::Float32Array::from(values);
    let list_array =
        arrow_array::FixedSizeListArray::try_new_from_values(values_array, dims).unwrap();

    let schema = Arc::new(Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dims),
        false,
    )]));

    arrow_array::RecordBatch::try_new(schema, vec![Arc::new(list_array)]).unwrap()
}

/// Encoded test data shared across benchmarks
struct EncodedData {
    encoded: Arc<lance_encoding::encoder::EncodedBatch>,
    query: Float32Array,
}

impl EncodedData {
    fn new(rt: &tokio::runtime::Runtime) -> Self {
        let data = generate_vector_batch(NUM_ROWS as usize, DIMS);
        let lance_schema =
            Arc::new(lance_core::datatypes::Schema::try_from(data.schema().as_ref()).unwrap());
        let encoding_strategy = default_encoding_strategy(LanceFileVersion::V2_1);

        let encoded = rt
            .block_on(encode_batch(
                &data,
                lance_schema,
                encoding_strategy.as_ref(),
                &EncodingOptions::default(),
            ))
            .unwrap();

        let query: Float32Array = {
            use rand::{Rng, SeedableRng};
            let mut rng = rand::rngs::StdRng::seed_from_u64(123);
            (0..DIMS).map(|_| Some(rng.random::<f32>())).collect()
        };

        Self {
            encoded: Arc::new(encoded),
            query,
        }
    }
}

/// Timing stats for diagnostics
#[derive(Default, Debug)]
struct DecodeStats {
    total_rows: usize,
    decode_wait_ns: u64,
    l2_compute_ns: u64,
    batch_count: usize,
    io_request_count: usize,
    io_range_count: usize,
    io_total_bytes: usize,
}

/// Run decode with specified configuration, return stats
async fn run_decode(
    encoded: &lance_encoding::encoder::EncodedBatch,
    query: &Float32Array,
    channel_capacity: Option<usize>,
    parallelism: usize,
    io_latency: Duration,
) -> DecodeStats {
    let io_scheduler = Arc::new(CloningIoScheduler::with_latency(encoded.data.clone(), io_latency));
    let io_scheduler_ref = io_scheduler.clone();
    let cache = Arc::new(LanceCache::no_cache());

    let config = SchedulerDecoderConfig {
        decoder_plugins: Arc::<DecoderPlugins>::default(),
        batch_size: BATCH_SIZE,
        io: io_scheduler,
        cache,
        decoder_config: DecoderConfig::default(),
        decode_channel_capacity: channel_capacity,
    };

    let decode_stream = schedule_and_decode(
        encoded.page_table.clone(),
        RequestedRows::Ranges(vec![0..encoded.num_rows]),
        FilterExpression::no_filter(),
        encoded.top_level_columns.clone(),
        encoded.schema.clone(),
        config,
    );

    let mut stats = DecodeStats::default();
    let mut all_distances: Vec<Arc<Float32Array>> = Vec::new();
    let mut stream = std::pin::pin!(decode_stream.map(|task| task.task).buffered(parallelism));

    while let Some(batch) = {
        let t0 = Instant::now();
        let b = stream.next().await;
        stats.decode_wait_ns += t0.elapsed().as_nanos() as u64;
        b
    } {
        let batch = batch.unwrap();
        stats.total_rows += batch.num_rows();
        stats.batch_count += 1;

        let t0 = Instant::now();
        let distances = compute_l2_distances(&batch, query);
        stats.l2_compute_ns += t0.elapsed().as_nanos() as u64;

        all_distances.push(distances);
    }

    std::hint::black_box(&all_distances);
    stats.io_request_count = io_scheduler_ref.request_count();
    stats.io_range_count = io_scheduler_ref.range_count();
    stats.io_total_bytes = io_scheduler_ref.total_bytes();
    stats
}

#[derive(Debug, Clone)]
struct BenchResult {
    mode: &'static str,
    parallelism: usize,
    io_latency_ms: u64,
    throughput_rows_per_sec: f64,
    peak_memory_bytes: usize,
    decode_wait_ms: f64,
    l2_compute_ms: f64,
    io_request_count: usize,
}

fn run_single_benchmark(
    rt: &tokio::runtime::Runtime,
    test_data: &EncodedData,
    mode: &'static str,
    channel_capacity: Option<usize>,
    parallelism: usize,
    io_latency: Duration,
) -> BenchResult {
    // Warmup
    for _ in 0..WARMUP_ITERS {
        PEAK_ALLOC.reset_peak_usage();
        rt.block_on(run_decode(
            &test_data.encoded,
            &test_data.query,
            channel_capacity,
            parallelism,
            io_latency,
        ));
    }

    // Benchmark iterations
    let mut total_duration = Duration::ZERO;
    let mut max_peak_memory = 0usize;
    let mut total_decode_wait_ns = 0u64;
    let mut total_l2_compute_ns = 0u64;
    let mut last_stats = DecodeStats::default();

    for _ in 0..BENCH_ITERS {
        PEAK_ALLOC.reset_peak_usage();
        let baseline = PEAK_ALLOC.current_usage();

        let start = Instant::now();
        let stats = rt.block_on(run_decode(
            &test_data.encoded,
            &test_data.query,
            channel_capacity,
            parallelism,
            io_latency,
        ));
        let duration = start.elapsed();

        let peak = PEAK_ALLOC.peak_usage();
        let peak_delta = peak.saturating_sub(baseline);

        total_duration += duration;
        max_peak_memory = max_peak_memory.max(peak_delta);
        total_decode_wait_ns += stats.decode_wait_ns;
        total_l2_compute_ns += stats.l2_compute_ns;
        last_stats = stats;

        assert_eq!(last_stats.total_rows, NUM_ROWS as usize);
    }

    let avg_duration = total_duration / BENCH_ITERS as u32;
    let throughput = NUM_ROWS as f64 / avg_duration.as_secs_f64();

    BenchResult {
        mode,
        parallelism,
        io_latency_ms: io_latency.as_millis() as u64,
        throughput_rows_per_sec: throughput,
        peak_memory_bytes: max_peak_memory,
        decode_wait_ms: (total_decode_wait_ns as f64 / BENCH_ITERS as f64) / 1_000_000.0,
        l2_compute_ms: (total_l2_compute_ns as f64 / BENCH_ITERS as f64) / 1_000_000.0,
        io_request_count: last_stats.io_request_count,
    }
}

fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();

    eprintln!("Generating test data ({} rows, {} dims)...", NUM_ROWS, DIMS);
    let test_data = EncodedData::new(&rt);
    eprintln!("Test data ready. Running benchmarks...\n");

    let mut results = Vec::new();

    // Channel capacities to test: unbounded, and various fixed sizes
    let channel_configs: &[(&str, Option<usize>)] = &[
        ("unbounded", None),
        ("bounded_c=2", Some(2)),
        ("bounded_c=4", Some(4)),
        ("bounded_c=8", Some(8)),
        ("bounded_c=16", Some(16)),
    ];

    for &latency_ms in IO_LATENCIES_MS {
        let latency = Duration::from_millis(latency_ms);
        eprintln!("=== I/O Latency: {}ms ===", latency_ms);

        for &p in PARALLELISM_LEVELS {
            for &(mode, capacity) in channel_configs {
                eprint!("  latency={:2}ms p={:2} {:14} ", latency_ms, p, mode);
                let result = run_single_benchmark(&rt, &test_data, mode, capacity, p, latency);
                eprintln!(
                    "{:>7.0} rows/s, {:>5.1} MB, io_reqs={}",
                    result.throughput_rows_per_sec,
                    result.peak_memory_bytes as f64 / (1024.0 * 1024.0),
                    result.io_request_count,
                );
                results.push(result);
            }
        }
        eprintln!();
    }

    // Print CSV
    eprintln!("--- CSV Output ---");
    println!("io_latency_ms,mode,parallelism,throughput_rows_per_sec,peak_memory_mb,decode_wait_ms,l2_compute_ms,io_request_count");
    for r in &results {
        println!(
            "{},{},{},{:.0},{:.1},{:.1},{:.1},{}",
            r.io_latency_ms,
            r.mode,
            r.parallelism,
            r.throughput_rows_per_sec,
            r.peak_memory_bytes as f64 / (1024.0 * 1024.0),
            r.decode_wait_ms,
            r.l2_compute_ms,
            r.io_request_count,
        );
    }
}
