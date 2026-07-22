use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::{StreamExt, future::try_join_all};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::{Dataset, WriteParams};
use lance::io::exec::filtered_read::{FilteredReadExec, FilteredReadOptions, FilteredReadPlan};
use lance_core::datatypes::{OnMissing, Projection};
use lance_core::utils::address::RowAddress;
use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
use lance_datafusion::exec::OneShotExec;
use lance_io::object_store::{ObjectStoreParams, WrappingObjectStore};
use lance_select::result::IndexExprResultWireFormat;
use lance_select::{IndexExprResult, RowAddrMask, RowAddrTreeMap};
use roaring::RoaringBitmap;
use serde_json::{Value, json};

type BenchResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Debug, Clone, Copy)]
enum Mode {
    Prepare,
    Direct,
    Staged,
    Precomputed,
}

impl Mode {
    fn from_env() -> BenchResult<Self> {
        match std::env::var("BENCH_MODE")
            .unwrap_or_else(|_| "direct".to_string())
            .as_str()
        {
            "prepare" => Ok(Self::Prepare),
            "direct" => Ok(Self::Direct),
            "staged" => Ok(Self::Staged),
            "precomputed" => Ok(Self::Precomputed),
            value => Err(format!(
                "BENCH_MODE must be prepare/direct/staged/precomputed, got {value}"
            )
            .into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Prepare => "prepare",
            Self::Direct => "direct",
            Self::Staged => "staged",
            Self::Precomputed => "precomputed",
        }
    }
}

#[derive(Debug, Clone)]
struct Config {
    label: String,
    dataset_uri: String,
    create_dataset: bool,
    mode: Mode,
    fragments: usize,
    rows_per_fragment: usize,
    selected_fragments: usize,
    concurrency: usize,
    warmups: usize,
    samples: usize,
}

impl Config {
    fn from_env() -> BenchResult<Self> {
        let config = Self {
            label: std::env::var("BENCH_LABEL").unwrap_or_else(|_| "unlabeled".to_string()),
            dataset_uri: std::env::var("BENCH_DATASET_URI")
                .map_err(|_| "BENCH_DATASET_URI is required")?,
            create_dataset: env_bool("BENCH_CREATE_DATASET", false)?,
            mode: Mode::from_env()?,
            fragments: env_usize("BENCH_FRAGMENTS", 100)?,
            rows_per_fragment: env_usize("BENCH_ROWS_PER_FRAGMENT", 1_000_000)?,
            selected_fragments: env_usize("BENCH_SELECTED_FRAGMENTS", 100)?,
            concurrency: env_usize("BENCH_CONCURRENCY", 1)?,
            warmups: env_usize("BENCH_WARMUPS", 5)?,
            samples: env_usize("BENCH_SAMPLES", 100)?,
        };
        if config.fragments == 0
            || config.rows_per_fragment == 0
            || config.selected_fragments == 0
            || config.concurrency == 0
            || config.samples == 0
        {
            return Err("all benchmark dimensions and samples must be non-zero".into());
        }
        if config.selected_fragments > config.fragments {
            return Err("BENCH_SELECTED_FRAGMENTS cannot exceed BENCH_FRAGMENTS".into());
        }
        let total_rows = config
            .fragments
            .checked_mul(config.rows_per_fragment)
            .ok_or("benchmark row count overflow")?;
        i32::try_from(total_rows)?;
        Ok(config)
    }
}

fn env_usize(name: &str, default: usize) -> BenchResult<usize> {
    Ok(std::env::var(name)
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(default))
}

fn env_bool(name: &str, default: bool) -> BenchResult<bool> {
    Ok(std::env::var(name)
        .ok()
        .map(|value| match value.as_str() {
            "1" | "true" | "yes" => Ok(true),
            "0" | "false" | "no" => Ok(false),
            _ => Err(format!("{name} must be one of 0/1/false/true/no/yes")),
        })
        .transpose()?
        .unwrap_or(default))
}

#[derive(Debug, Default)]
struct StoreCounters {
    data_ops: AtomicUsize,
    metadata_ops: AtomicUsize,
    heads: AtomicUsize,
}

impl StoreCounters {
    fn reset(&self) {
        self.data_ops.store(0, Ordering::Relaxed);
        self.metadata_ops.store(0, Ordering::Relaxed);
        self.heads.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> StoreSnapshot {
        StoreSnapshot {
            data_ops: self.data_ops.load(Ordering::Relaxed),
            metadata_ops: self.metadata_ops.load(Ordering::Relaxed),
            heads: self.heads.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct StoreSnapshot {
    data_ops: usize,
    metadata_ops: usize,
    heads: usize,
}

#[derive(Debug, Clone)]
struct TrackingWrapper {
    policy: Arc<Mutex<ProxyObjectStorePolicy>>,
}

impl TrackingWrapper {
    fn new() -> (Self, Arc<StoreCounters>) {
        let counters = Arc::new(StoreCounters::default());
        let policy = Arc::new(Mutex::new(ProxyObjectStorePolicy::new()));
        let before_counters = counters.clone();
        policy.lock().unwrap().set_before_policy(
            "count_store_ops",
            Arc::new(move |_method, path| {
                let path = path.as_ref();
                if path.contains("data/") && path.ends_with(".lance") {
                    before_counters.data_ops.fetch_add(1, Ordering::Relaxed);
                } else {
                    before_counters.metadata_ops.fetch_add(1, Ordering::Relaxed);
                }
                Ok(())
            }),
        );
        let head_counters = counters.clone();
        policy.lock().unwrap().set_obj_meta_policy(
            "count_heads",
            Arc::new(move |_method, meta| {
                head_counters.heads.fetch_add(1, Ordering::Relaxed);
                Ok(meta)
            }),
        );
        (Self { policy }, counters)
    }
}

impl WrappingObjectStore for TrackingWrapper {
    fn wrap(
        &self,
        _store_prefix: &str,
        target: Arc<dyn object_store::ObjectStore>,
    ) -> Arc<dyn object_store::ObjectStore> {
        Arc::new(ProxyObjectStore::new(target, self.policy.clone()))
    }
}

async fn create_dataset(
    dataset_uri: &str,
    config: &Config,
    store_params: ObjectStoreParams,
) -> BenchResult<Dataset> {
    let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
    let batch_schema = schema.clone();
    let rows_per_fragment = config.rows_per_fragment;
    let batches = (0..config.fragments).map(move |fragment_index| {
        let start = i32::try_from(fragment_index * rows_per_fragment).unwrap();
        let end = i32::try_from((fragment_index + 1) * rows_per_fragment).unwrap();
        RecordBatch::try_new(
            batch_schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(start..end))],
        )
    });
    let reader = RecordBatchIterator::new(batches, schema);
    Ok(Dataset::write(
        reader,
        dataset_uri,
        Some(WriteParams {
            max_rows_per_file: config.rows_per_fragment,
            max_rows_per_group: config.rows_per_fragment.min(4096),
            store_params: Some(store_params),
            ..Default::default()
        }),
    )
    .await?)
}

async fn open_dataset(dataset_uri: &str, store_params: ObjectStoreParams) -> BenchResult<Dataset> {
    Ok(DatasetBuilder::from_uri(dataset_uri)
        .with_store_params(store_params)
        .load()
        .await?)
}

#[derive(Debug, Clone)]
struct MaskFixture {
    batch: RecordBatch,
    expected_rows: usize,
    expected_checksum: i64,
}

fn make_mask(dataset: &Dataset, config: &Config) -> BenchResult<MaskFixture> {
    let descriptors = dataset.fragments();
    let covered: RoaringBitmap = descriptors
        .iter()
        .map(|fragment| u32::try_from(fragment.id))
        .collect::<Result<_, _>>()?;
    let mut addresses = Vec::with_capacity(config.selected_fragments);
    let mut checksum = 0_i64;
    for selected_index in 0..config.selected_fragments {
        let descriptor_index = selected_index * descriptors.len() / config.selected_fragments;
        let fragment_id = u32::try_from(descriptors[descriptor_index].id)?;
        let row_offset = (selected_index * 997) % config.rows_per_fragment;
        addresses.push(u64::from(RowAddress::new_from_parts(
            fragment_id,
            u32::try_from(row_offset)?,
        )));
        checksum += i64::try_from(descriptor_index * config.rows_per_fragment + row_offset)?;
    }
    let batch = IndexExprResult::exact(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(
        addresses,
    )))
    .serialize(&covered, IndexExprResultWireFormat::default())?;
    assert_eq!(
        batch.schema().fields(),
        IndexExprResultWireFormat::default().schema().fields(),
        "the benchmark must route through RowSelector::RowSet"
    );
    Ok(MaskFixture {
        batch,
        expected_rows: config.selected_fragments,
        expected_checksum: checksum,
    })
}

fn make_input(mask_batch: RecordBatch) -> Arc<dyn ExecutionPlan> {
    let schema = mask_batch.schema();
    let stream = futures::stream::once(async move { Ok(mask_batch) });
    Arc::new(OneShotExec::new(Box::pin(RecordBatchStreamAdapter::new(
        schema, stream,
    ))))
}

fn make_options(dataset: &Arc<Dataset>) -> BenchResult<FilteredReadOptions> {
    let projection: Projection = dataset
        .empty_projection()
        .union_columns(["i"], OnMissing::Error)?;
    Ok(FilteredReadOptions::new(projection).with_batch_size(4096))
}

fn make_exec(
    dataset: Arc<Dataset>,
    options: FilteredReadOptions,
    mask_batch: RecordBatch,
) -> BenchResult<FilteredReadExec> {
    Ok(FilteredReadExec::try_new(
        dataset,
        options,
        Some(make_input(mask_batch)),
    )?)
}

async fn consume_exec(
    exec: &FilteredReadExec,
    expected_rows: usize,
    expected_checksum: i64,
) -> BenchResult<(Duration, Duration)> {
    let started = Instant::now();
    let mut stream = exec.execute(0, Arc::new(TaskContext::default()))?;
    let first_batch = stream.next().await.ok_or("read returned no batches")??;
    let first_batch_elapsed = started.elapsed();
    let mut rows = first_batch.num_rows();
    let values = first_batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or("expected Int32 benchmark output")?;
    let mut checksum = values.iter().flatten().map(i64::from).sum::<i64>();
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        rows += batch.num_rows();
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or("expected Int32 benchmark output")?;
        checksum += values.iter().flatten().map(i64::from).sum::<i64>();
    }
    assert_eq!(rows, expected_rows);
    assert_eq!(checksum, expected_checksum);
    Ok((first_batch_elapsed, started.elapsed()))
}

#[derive(Debug)]
struct QueryTiming {
    total: Duration,
    plan: Option<Duration>,
    first_batch: Duration,
    execute: Duration,
}

async fn run_query(
    dataset: Arc<Dataset>,
    options: FilteredReadOptions,
    mask: MaskFixture,
    mode: Mode,
    precomputed_plan: Option<FilteredReadPlan>,
) -> BenchResult<QueryTiming> {
    let exec = make_exec(dataset, options, mask.batch)?;
    match mode {
        Mode::Prepare => unreachable!(),
        Mode::Direct => {
            let (first_batch, total) =
                consume_exec(&exec, mask.expected_rows, mask.expected_checksum).await?;
            Ok(QueryTiming {
                total,
                plan: None,
                first_batch,
                execute: total,
            })
        }
        Mode::Staged => {
            let plan_started = Instant::now();
            let _ = exec
                .get_or_create_plan(Arc::new(TaskContext::default()))
                .await?;
            let plan = plan_started.elapsed();
            let (first_batch, execute) =
                consume_exec(&exec, mask.expected_rows, mask.expected_checksum).await?;
            Ok(QueryTiming {
                total: plan + execute,
                plan: Some(plan),
                first_batch,
                execute,
            })
        }
        Mode::Precomputed => {
            let exec = exec
                .with_plan(precomputed_plan.ok_or("precomputed plan is required")?)
                .await?;
            let (first_batch, execute) =
                consume_exec(&exec, mask.expected_rows, mask.expected_checksum).await?;
            Ok(QueryTiming {
                total: execute,
                plan: None,
                first_batch,
                execute,
            })
        }
    }
}

#[derive(Debug)]
struct RoundResult {
    wall: Duration,
    queries: Vec<QueryTiming>,
    store: StoreSnapshot,
}

async fn run_round(
    dataset: Arc<Dataset>,
    options: FilteredReadOptions,
    mask: MaskFixture,
    config: Arc<Config>,
    precomputed_plan: Option<FilteredReadPlan>,
    counters: Arc<StoreCounters>,
) -> BenchResult<RoundResult> {
    counters.reset();
    let started = Instant::now();
    let queries = (0..config.concurrency).map(|_| {
        run_query(
            dataset.clone(),
            options.clone(),
            mask.clone(),
            config.mode,
            precomputed_plan.clone(),
        )
    });
    let queries = try_join_all(queries).await?;
    Ok(RoundResult {
        wall: started.elapsed(),
        queries,
        store: counters.snapshot(),
    })
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index]
}

fn stats(values: &[f64]) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let median = percentile(values, 0.50);
    let deviations = values
        .iter()
        .map(|value| (value - median).abs())
        .collect::<Vec<_>>();
    json!({
        "n": values.len(),
        "mean": mean,
        "stddev": variance.sqrt(),
        "p50": median,
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "mad": percentile(&deviations, 0.50),
        "raw": values,
    })
}

fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
#[ignore = "manual EC2 benchmark for PR 7792"]
async fn bench_filtered_read_metadata_reuse() -> BenchResult<()> {
    let config = Arc::new(Config::from_env()?);
    let (wrapper, counters) = TrackingWrapper::new();
    let store_params = ObjectStoreParams {
        object_store_wrapper: Some(Arc::new(wrapper)),
        ..Default::default()
    };
    let dataset = if config.create_dataset {
        create_dataset(&config.dataset_uri, &config, store_params).await?
    } else {
        open_dataset(&config.dataset_uri, store_params).await?
    };
    assert_eq!(dataset.fragments().len(), config.fragments);
    let known_file_sizes = dataset
        .fragments()
        .iter()
        .flat_map(|fragment| fragment.files.iter())
        .all(|file| file.file_size_bytes.get().is_some());
    assert!(
        known_file_sizes,
        "the positive benchmark requires modern descriptors with known file sizes"
    );
    let total_file_size_bytes = dataset
        .fragments()
        .iter()
        .flat_map(|fragment| fragment.files.iter())
        .filter_map(|file| file.file_size_bytes.get().map(|size| size.get()))
        .sum::<u64>();
    if matches!(config.mode, Mode::Prepare) {
        println!(
            "{}",
            json!({
                "event": "filtered_read_pr7792_dataset",
                "dataset_uri": config.dataset_uri,
                "fragments": config.fragments,
                "rows_per_fragment": config.rows_per_fragment,
                "total_rows": config.fragments * config.rows_per_fragment,
                "total_file_size_bytes": total_file_size_bytes,
                "known_file_sizes": known_file_sizes,
            })
        );
        return Ok(());
    }

    let dataset = Arc::new(dataset);
    let options = make_options(&dataset)?;
    let mask = make_mask(&dataset, &config)?;
    let precomputed_plan = if matches!(config.mode, Mode::Precomputed) {
        let exec = make_exec(dataset.clone(), options.clone(), mask.batch.clone())?;
        Some(
            exec.get_or_create_plan(Arc::new(TaskContext::default()))
                .await?,
        )
    } else {
        None
    };

    for _ in 0..config.warmups {
        let _ = run_round(
            dataset.clone(),
            options.clone(),
            mask.clone(),
            config.clone(),
            precomputed_plan.clone(),
            counters.clone(),
        )
        .await?;
    }

    let mut wall_ms = Vec::with_capacity(config.samples);
    let mut query_total_ms = Vec::with_capacity(config.samples * config.concurrency);
    let mut plan_ms = Vec::with_capacity(config.samples * config.concurrency);
    let mut first_batch_ms = Vec::with_capacity(config.samples * config.concurrency);
    let mut execute_ms = Vec::with_capacity(config.samples * config.concurrency);
    let mut data_ops_per_query = Vec::with_capacity(config.samples);
    let mut metadata_ops_per_query = Vec::with_capacity(config.samples);
    let mut heads_per_query = Vec::with_capacity(config.samples);
    for _ in 0..config.samples {
        let round = run_round(
            dataset.clone(),
            options.clone(),
            mask.clone(),
            config.clone(),
            precomputed_plan.clone(),
            counters.clone(),
        )
        .await?;
        wall_ms.push(millis(round.wall));
        for query in round.queries {
            query_total_ms.push(millis(query.total));
            if let Some(plan) = query.plan {
                plan_ms.push(millis(plan));
            }
            first_batch_ms.push(millis(query.first_batch));
            execute_ms.push(millis(query.execute));
        }
        data_ops_per_query.push(round.store.data_ops as f64 / config.concurrency as f64);
        metadata_ops_per_query.push(round.store.metadata_ops as f64 / config.concurrency as f64);
        heads_per_query.push(round.store.heads as f64 / config.concurrency as f64);
    }

    let wall_p50 = percentile(&wall_ms, 0.50);
    println!(
        "{}",
        json!({
            "event": "filtered_read_pr7792_bench",
            "label": config.label,
            "revision": option_env!("PR7792_BENCH_REVISION").unwrap_or("unknown"),
            "upstream_sha": option_env!("PR7792_UPSTREAM_SHA").unwrap_or("unknown"),
            "mode": config.mode.as_str(),
            "dataset_uri": config.dataset_uri,
            "fragments": config.fragments,
            "rows_per_fragment": config.rows_per_fragment,
            "total_rows": config.fragments * config.rows_per_fragment,
            "selected_fragments": config.selected_fragments,
            "selected_rows": mask.expected_rows,
            "expected_checksum": mask.expected_checksum,
            "concurrency": config.concurrency,
            "warmups": config.warmups,
            "samples": config.samples,
            "known_file_sizes": known_file_sizes,
            "total_file_size_bytes": total_file_size_bytes,
            "wall_ms": stats(&wall_ms),
            "query_total_ms": stats(&query_total_ms),
            "plan_ms": stats(&plan_ms),
            "first_batch_ms": stats(&first_batch_ms),
            "execute_ms": stats(&execute_ms),
            "data_store_ops_per_query": stats(&data_ops_per_query),
            "metadata_store_ops_per_query": stats(&metadata_ops_per_query),
            "heads_per_query": stats(&heads_per_query),
            "queries_per_second_at_p50_wall": config.concurrency as f64 / (wall_p50 / 1_000.0),
            "peak_rss_kib": peak_rss_kib(),
            "available_parallelism": std::thread::available_parallelism()?.get(),
        })
    );
    Ok(())
}
