use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::{TryStreamExt, future::try_join_all};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::{Dataset, WriteParams};
use lance::io::exec::filtered_read::{FilteredReadExec, FilteredReadOptions};
use lance_core::ROW_ADDR;
use lance_core::datatypes::{OnMissing, Projection};
use lance_core::utils::address::RowAddress;
use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
use lance_datafusion::exec::OneShotExec;
use lance_io::object_store::{ObjectStoreParams, WrappingObjectStore};
use lance_io::utils::CachedFileSize;
use serde_json::json;

type BenchResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Debug, Clone)]
struct Config {
    label: String,
    dataset_uri: Option<String>,
    create_dataset: bool,
    fragments: usize,
    rows_per_fragment: usize,
    selected_fragments: usize,
    input_batches: usize,
    concurrency: usize,
    warmups: usize,
    samples: usize,
    unknown_file_size: bool,
    head_delay: Duration,
}

impl Config {
    fn from_env() -> BenchResult<Self> {
        let dataset_uri = std::env::var("BENCH_DATASET_URI").ok();
        let config = Self {
            label: std::env::var("BENCH_LABEL").unwrap_or_else(|_| "unlabeled".to_string()),
            create_dataset: env_bool("BENCH_CREATE_DATASET", dataset_uri.is_none())?,
            dataset_uri,
            fragments: env_usize("BENCH_FRAGMENTS", 100)?,
            rows_per_fragment: env_usize("BENCH_ROWS_PER_FRAGMENT", 4096)?,
            selected_fragments: env_usize("BENCH_SELECTED_FRAGMENTS", 100)?,
            input_batches: env_usize("BENCH_INPUT_BATCHES", 10)?,
            concurrency: env_usize("BENCH_CONCURRENCY", 1)?,
            warmups: env_usize("BENCH_WARMUPS", 3)?,
            samples: env_usize("BENCH_SAMPLES", 15)?,
            unknown_file_size: env_bool("BENCH_UNKNOWN_FILE_SIZE", false)?,
            head_delay: Duration::from_micros(env_usize("BENCH_HEAD_DELAY_US", 0)? as u64),
        };
        if config.fragments == 0
            || config.rows_per_fragment == 0
            || config.selected_fragments == 0
            || config.input_batches == 0
            || config.concurrency == 0
            || config.samples == 0
        {
            return Err("all benchmark dimensions and samples must be non-zero".into());
        }
        if config.selected_fragments > config.fragments {
            return Err("BENCH_SELECTED_FRAGMENTS cannot exceed BENCH_FRAGMENTS".into());
        }
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

#[derive(Debug, Clone)]
struct HeadTrackingWrapper {
    policy: Arc<Mutex<ProxyObjectStorePolicy>>,
}

impl HeadTrackingWrapper {
    fn new(head_delay: Duration) -> (Self, Arc<AtomicUsize>) {
        let data_heads = Arc::new(AtomicUsize::new(0));
        let policy = Arc::new(Mutex::new(ProxyObjectStorePolicy::new()));
        let count = data_heads.clone();
        policy.lock().unwrap().set_obj_meta_policy(
            "count_data_heads",
            Arc::new(move |method, meta| {
                let path = meta.location.as_ref();
                if method == "head" && path.contains("data/") && path.ends_with(".lance") {
                    count.fetch_add(1, Ordering::SeqCst);
                    if !head_delay.is_zero() {
                        std::thread::sleep(head_delay);
                    }
                }
                Ok(meta)
            }),
        );
        (Self { policy }, data_heads)
    }
}

impl WrappingObjectStore for HeadTrackingWrapper {
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
    let total_rows = config.fragments * config.rows_per_fragment;
    let total_rows = i32::try_from(total_rows)?;
    let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..total_rows))],
    )?;
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let dataset = Dataset::write(
        reader,
        dataset_uri,
        Some(WriteParams {
            max_rows_per_file: config.rows_per_fragment,
            max_rows_per_group: config.rows_per_fragment,
            store_params: Some(store_params),
            ..Default::default()
        }),
    )
    .await?;
    Ok(dataset)
}

async fn open_dataset(dataset_uri: &str, store_params: ObjectStoreParams) -> BenchResult<Dataset> {
    Ok(DatasetBuilder::from_uri(dataset_uri)
        .with_store_params(store_params)
        .load()
        .await?)
}

fn make_input(
    input_schema: SchemaRef,
    selected_fragment_ids: &[u32],
    config: &Config,
) -> Arc<dyn ExecutionPlan> {
    let input_batches = (0..config.input_batches)
        .map(|batch_index| {
            let row_offset = (batch_index % config.rows_per_fragment) as u32;
            let addresses = selected_fragment_ids
                .iter()
                .map(|fragment_id| u64::from(RowAddress::new_from_parts(*fragment_id, row_offset)))
                .collect::<Vec<_>>();
            RecordBatch::try_new(
                input_schema.clone(),
                vec![Arc::new(UInt64Array::from(addresses))],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let stream = futures::stream::iter(input_batches.into_iter().map(Ok));
    let stream = Box::pin(RecordBatchStreamAdapter::new(input_schema, stream));
    Arc::new(OneShotExec::new(stream))
}

async fn run_query(
    dataset: Arc<Dataset>,
    fragments: Arc<Vec<lance_table::format::Fragment>>,
    projection: Projection,
    input_schema: SchemaRef,
    selected_fragment_ids: Arc<Vec<u32>>,
    config: Arc<Config>,
) -> BenchResult<Duration> {
    let input = make_input(input_schema, &selected_fragment_ids, &config);
    let options = FilteredReadOptions::new(projection)
        .with_fragments(fragments)
        .with_batch_size(config.selected_fragments as u32);
    let started = Instant::now();
    let plan = FilteredReadExec::try_new(dataset, options, Some(input))?;
    let output = plan
        .execute(0, Arc::new(TaskContext::default()))?
        .try_collect::<Vec<_>>()
        .await?;
    let elapsed = started.elapsed();
    let output_rows = output.iter().map(RecordBatch::num_rows).sum::<usize>();
    assert_eq!(
        output_rows,
        config.input_batches * config.selected_fragments
    );
    Ok(elapsed)
}

async fn run_round(
    dataset: Arc<Dataset>,
    fragments: Arc<Vec<lance_table::format::Fragment>>,
    projection: Projection,
    input_schema: SchemaRef,
    selected_fragment_ids: Arc<Vec<u32>>,
    config: Arc<Config>,
) -> BenchResult<(Duration, Vec<Duration>)> {
    let started = Instant::now();
    let queries = (0..config.concurrency).map(|_| {
        run_query(
            dataset.clone(),
            fragments.clone(),
            projection.clone(),
            input_schema.clone(),
            selected_fragment_ids.clone(),
            config.clone(),
        )
    });
    let query_times = try_join_all(queries).await?;
    Ok((started.elapsed(), query_times))
}

fn percentile(mut values: Vec<f64>, quantile: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * quantile).round() as usize;
    values[index]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
#[ignore = "manual EC2 benchmark for PR 7792"]
async fn bench_row_stream_fragment_reconstruction() -> BenchResult<()> {
    let config = Arc::new(Config::from_env()?);
    let (wrapper, data_heads) = HeadTrackingWrapper::new(config.head_delay);
    let store_params = ObjectStoreParams {
        object_store_wrapper: Some(Arc::new(wrapper)),
        ..Default::default()
    };
    let temp_dir = if config.dataset_uri.is_none() {
        Some(tempfile::tempdir()?)
    } else {
        None
    };
    let dataset_uri = config.dataset_uri.clone().unwrap_or_else(|| {
        temp_dir
            .as_ref()
            .unwrap()
            .path()
            .join("dataset")
            .to_string_lossy()
            .into_owned()
    });
    let dataset = if config.create_dataset {
        create_dataset(&dataset_uri, &config, store_params).await?
    } else {
        open_dataset(&dataset_uri, store_params).await?
    };
    assert_eq!(dataset.fragments().len(), config.fragments);
    let dataset = Arc::new(dataset);

    let mut fragment_descriptors = dataset.fragments().as_ref().clone();
    let known_file_sizes = fragment_descriptors
        .iter()
        .flat_map(|fragment| fragment.files.iter())
        .all(|file| file.file_size_bytes.get().is_some());
    assert!(
        known_file_sizes,
        "newly written benchmark data must include file sizes"
    );
    if config.unknown_file_size {
        for file in fragment_descriptors
            .iter_mut()
            .flat_map(|fragment| fragment.files.iter_mut())
        {
            file.file_size_bytes = CachedFileSize::unknown();
        }
    }
    let selected_fragment_ids = fragment_descriptors
        .iter()
        .take(config.selected_fragments)
        .map(|fragment| u32::try_from(fragment.id))
        .collect::<Result<Vec<_>, _>>()?;
    let selected_fragment_ids = Arc::new(selected_fragment_ids);
    let fragment_descriptors = Arc::new(fragment_descriptors);
    let projection = dataset
        .empty_projection()
        .union_columns(["i"], OnMissing::Error)?;
    let input_schema = Arc::new(Schema::new(vec![Field::new(
        ROW_ADDR,
        DataType::UInt64,
        false,
    )]));

    for _ in 0..config.warmups {
        run_round(
            dataset.clone(),
            fragment_descriptors.clone(),
            projection.clone(),
            input_schema.clone(),
            selected_fragment_ids.clone(),
            config.clone(),
        )
        .await?;
    }

    let mut wall_times = Vec::with_capacity(config.samples);
    let mut query_times = Vec::with_capacity(config.samples * config.concurrency);
    let mut head_counts = Vec::with_capacity(config.samples);
    for _ in 0..config.samples {
        data_heads.store(0, Ordering::SeqCst);
        let (wall_time, round_query_times) = run_round(
            dataset.clone(),
            fragment_descriptors.clone(),
            projection.clone(),
            input_schema.clone(),
            selected_fragment_ids.clone(),
            config.clone(),
        )
        .await?;
        wall_times.push(wall_time);
        query_times.extend(round_query_times);
        head_counts.push(data_heads.load(Ordering::SeqCst));
    }

    let wall_ms = wall_times
        .iter()
        .map(|duration| duration.as_secs_f64() * 1_000.0)
        .collect::<Vec<_>>();
    let query_ms = query_times
        .iter()
        .map(|duration| duration.as_secs_f64() * 1_000.0)
        .collect::<Vec<_>>();
    let wall_ms_p50 = percentile(wall_ms.clone(), 0.50);
    let heads_per_query = head_counts
        .iter()
        .map(|count| *count as f64 / config.concurrency as f64)
        .collect::<Vec<_>>();
    let result = json!({
        "event": "row_stream_pr7792_bench",
        "label": config.label,
        "dataset_uri": dataset_uri,
        "fragments": config.fragments,
        "rows_per_fragment": config.rows_per_fragment,
        "selected_fragments": config.selected_fragments,
        "input_batches": config.input_batches,
        "concurrency": config.concurrency,
        "warmups": config.warmups,
        "samples": config.samples,
        "unknown_file_size": config.unknown_file_size,
        "synthetic_head_delay_us": config.head_delay.as_micros(),
        "wall_ms_p50": wall_ms_p50,
        "wall_ms_p95": percentile(wall_ms, 0.95),
        "query_ms_p50": percentile(query_ms.clone(), 0.50),
        "query_ms_p95": percentile(query_ms, 0.95),
        "queries_per_second_at_p50_wall": config.concurrency as f64 / (wall_ms_p50 / 1_000.0),
        "data_heads_per_query_p50": percentile(heads_per_query.clone(), 0.50),
        "data_heads_per_query_p95": percentile(heads_per_query, 0.95),
        "available_parallelism": std::thread::available_parallelism()?.get(),
    });
    println!("{result}");
    Ok(())
}
