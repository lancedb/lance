// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Array, Int32Array, Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::{StreamExt, TryStreamExt, stream};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::optimize_columns::{ColumnGroup, OptimizeColumnsOptions};
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{CommitBuilder, NewColumnTransform};
use lance::{Dataset, Error, Result};
use lance_core::datatypes::Schema;
use serde::Serialize;

#[derive(Debug)]
struct Config {
    uri: String,
    fields: usize,
    rows: usize,
    fragments: usize,
    group_size: usize,
    iterations: usize,
    concurrency: usize,
}

#[derive(Debug, Clone)]
struct Workload {
    name: &'static str,
    fields: Vec<String>,
    iterations: usize,
}

#[derive(Debug, Serialize)]
struct LayoutSnapshot {
    version: u64,
    manifest_bytes: u64,
    fragment_count: usize,
    live_files: usize,
    file_descriptors: usize,
    overlays: usize,
    min_fields_per_live_file: usize,
    median_fields_per_live_file: usize,
    max_fields_per_live_file: usize,
}

#[derive(Debug, Default, Clone, Serialize)]
struct Sample {
    open_micros: u128,
    plan_micros: u128,
    time_to_first_batch_micros: u128,
    full_stream_micros: u128,
    open_read_iops: u64,
    open_read_bytes: u64,
    plan_read_iops: u64,
    plan_read_bytes: u64,
    first_batch_read_iops: u64,
    first_batch_read_bytes: u64,
    query_read_iops: u64,
    query_read_bytes: u64,
    checksum: i128,
}

#[derive(Debug, Serialize)]
struct Distribution {
    samples: Vec<u128>,
    median: u128,
    p95: u128,
}

#[derive(Debug, Serialize)]
struct PhaseSummary {
    open_micros: Distribution,
    plan_micros: Distribution,
    time_to_first_batch_micros: Distribution,
    full_stream_micros: Distribution,
    open_read_iops: Distribution,
    open_read_bytes: Distribution,
    plan_read_iops: Distribution,
    plan_read_bytes: Distribution,
    first_batch_read_iops: Distribution,
    first_batch_read_bytes: Distribution,
    query_read_iops: Distribution,
    query_read_bytes: Distribution,
    checksum: i128,
}

#[derive(Debug, Serialize)]
struct WorkloadResult {
    name: &'static str,
    projected_fields: usize,
    iterations: usize,
    application_cold_before: PhaseSummary,
    application_cold_after: PhaseSummary,
    warm_before: PhaseSummary,
    warm_after: PhaseSummary,
}

#[derive(Debug, Serialize)]
struct OptimizeCost {
    wall_millis: u128,
    read_iops: u64,
    read_bytes: u64,
    write_iops: u64,
    written_bytes: u64,
    files_added: usize,
    files_removed: usize,
    mixed_files_retained: usize,
}

#[derive(Debug, Serialize)]
struct BenchmarkResult {
    uri: String,
    fields: usize,
    rows: usize,
    fragments: usize,
    residual_group_size: usize,
    concurrency: usize,
    baseline_build_millis: u128,
    before_layout: LayoutSnapshot,
    after_layout: LayoutSnapshot,
    optimize: OptimizeCost,
    workloads: Vec<WorkloadResult>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = parse_args()?;
    writeln!(
        std::io::stderr().lock(),
        "creating one-file-per-feature baseline at {}",
        config.uri
    )?;
    let build_started = Instant::now();
    let mut dataset = create_baseline(&config).await?;
    let baseline_build_millis = build_started.elapsed().as_millis();
    let before_layout = layout_snapshot(&dataset);
    let before_version = dataset.version().version;

    let groups = affinity_groups(config.fields, config.group_size);
    let store = dataset.object_store(None).await?;
    let io_before = store.io_stats_snapshot();
    let optimize_started = Instant::now();
    let metrics = dataset
        .optimize_columns(OptimizeColumnsOptions {
            groups,
            fragment_ids: None,
            max_concurrency: Some(config.concurrency),
        })
        .await?;
    let optimize_wall_millis = optimize_started.elapsed().as_millis();
    let io_after = store.io_stats_snapshot();
    let after_layout = layout_snapshot(&dataset);
    let after_version = dataset.version().version;

    let workloads = workloads(&config);
    let mut results = Vec::with_capacity(workloads.len());
    for workload in workloads {
        writeln!(
            std::io::stderr().lock(),
            "measuring {} ({} fields, {} iterations)",
            workload.name,
            workload.fields.len(),
            workload.iterations
        )?;
        let (application_cold_before, application_cold_after) = measure_cold_pair(
            &config.uri,
            before_version,
            after_version,
            &workload.fields,
            workload.iterations,
        )
        .await?;
        let (warm_before, warm_after) = measure_warm_pair(
            &config.uri,
            before_version,
            after_version,
            &workload.fields,
            workload.iterations,
        )
        .await?;
        let application_cold_before = summarize(application_cold_before)?;
        let application_cold_after = summarize(application_cold_after)?;
        let warm_before = summarize(warm_before)?;
        let warm_after = summarize(warm_after)?;
        let checksums = [
            application_cold_before.checksum,
            application_cold_after.checksum,
            warm_before.checksum,
            warm_after.checksum,
        ];
        if checksums.iter().any(|checksum| *checksum != checksums[0]) {
            return Err(Error::internal(format!(
                "OptimizeColumns changed {} checksum across layouts or cache states: {checksums:?}",
                workload.name
            )));
        }
        results.push(WorkloadResult {
            name: workload.name,
            projected_fields: workload.fields.len(),
            iterations: workload.iterations,
            application_cold_before,
            application_cold_after,
            warm_before,
            warm_after,
        });
    }

    let result = BenchmarkResult {
        uri: config.uri,
        fields: config.fields,
        rows: config.rows,
        fragments: config.fragments,
        residual_group_size: config.group_size,
        concurrency: config.concurrency,
        baseline_build_millis,
        before_layout,
        after_layout,
        optimize: OptimizeCost {
            wall_millis: optimize_wall_millis,
            read_iops: io_after.read_iops.saturating_sub(io_before.read_iops),
            read_bytes: io_after.read_bytes.saturating_sub(io_before.read_bytes),
            write_iops: io_after.write_iops.saturating_sub(io_before.write_iops),
            written_bytes: io_after
                .written_bytes
                .saturating_sub(io_before.written_bytes),
            files_added: metrics.files_added,
            files_removed: metrics.files_removed,
            mixed_files_retained: metrics.mixed_files_retained,
        },
        workloads: results,
    };
    writeln!(
        std::io::stdout().lock(),
        "{}",
        serde_json::to_string_pretty(&result)?
    )?;
    Ok(())
}

fn parse_args() -> Result<Config> {
    let mut args = std::env::args().skip(1);
    let uri = args.next().ok_or_else(|| {
        Error::invalid_input(
            "usage: optimize_columns_layout_bench <uri> [fields] [rows] [fragments] [group_size] [iterations] [concurrency]",
        )
    })?;
    let parse = |value: Option<String>, default: usize, name: &str| -> Result<usize> {
        value
            .map(|value| {
                value.parse::<usize>().map_err(|error| {
                    Error::invalid_input(format!("invalid {name} value '{value}': {error}"))
                })
            })
            .unwrap_or(Ok(default))
    };
    let fields = parse(args.next(), 1_000, "fields")?;
    let rows = parse(args.next(), 100_000, "rows")?;
    let fragments = parse(args.next(), 64, "fragments")?;
    let group_size = parse(args.next(), 10, "group_size")?;
    let iterations = parse(args.next(), 30, "iterations")?;
    let concurrency = parse(args.next(), 16, "concurrency")?;
    if fields < 100
        || rows == 0
        || fragments == 0
        || group_size == 0
        || iterations < 3
        || concurrency == 0
    {
        return Err(Error::invalid_input(
            "fields must be at least 100, iterations at least 3, and all other numeric arguments must be positive",
        ));
    }
    Ok(Config {
        uri,
        fields,
        rows,
        fragments,
        group_size,
        iterations,
        concurrency,
    })
}

fn affinity_groups(fields: usize, group_size: usize) -> Vec<ColumnGroup> {
    let mut groups = vec![ColumnGroup {
        fields: (0..3).map(feature_name).collect(),
    }];
    groups.extend(
        (3..fields)
            .collect::<Vec<_>>()
            .chunks(group_size)
            .map(|chunk| ColumnGroup {
                fields: chunk.iter().copied().map(feature_name).collect(),
            }),
    );
    groups
}

fn workloads(config: &Config) -> Vec<Workload> {
    let cold_iterations = config.iterations;
    let wide_iterations = config.iterations.div_ceil(3).max(5);
    let full_iterations = config.iterations.div_ceil(10).max(3);
    vec![
        Workload {
            name: "co_located_narrow",
            fields: std::iter::once("id".to_string())
                .chain((0..3).map(feature_name))
                .collect(),
            iterations: cold_iterations,
        },
        Workload {
            name: "scattered_narrow",
            fields: vec![
                "id".to_string(),
                feature_name(0),
                feature_name(config.fields / 3),
                feature_name(config.fields * 2 / 3),
            ],
            iterations: cold_iterations,
        },
        Workload {
            name: "wide_100",
            fields: std::iter::once("id".to_string())
                .chain((0..100).map(feature_name))
                .collect(),
            iterations: wide_iterations,
        },
        Workload {
            name: "full_scan",
            fields: std::iter::once("id".to_string())
                .chain((0..config.fields).map(feature_name))
                .collect(),
            iterations: full_iterations,
        },
    ]
}

fn feature_name(field: usize) -> String {
    format!("feature_{field:04}")
}

fn layout_snapshot(dataset: &Dataset) -> LayoutSnapshot {
    let stats = dataset.column_layout_stats();
    let mut live_fields_per_file = stats
        .iter()
        .flat_map(|fragment| fragment.fields_per_file.iter().copied())
        .filter(|fields| *fields > 0)
        .collect::<Vec<_>>();
    live_fields_per_file.sort_unstable();
    LayoutSnapshot {
        version: dataset.version().version,
        manifest_bytes: dataset.manifest_location().size.unwrap_or_default(),
        fragment_count: stats.len(),
        live_files: stats.iter().map(|stats| stats.live_file_count).sum(),
        file_descriptors: dataset
            .manifest()
            .fragments
            .iter()
            .map(|fragment| fragment.files.len())
            .sum(),
        overlays: stats.iter().map(|stats| stats.overlay_count).sum(),
        min_fields_per_live_file: live_fields_per_file.first().copied().unwrap_or_default(),
        median_fields_per_live_file: live_fields_per_file
            .get(live_fields_per_file.len() / 2)
            .copied()
            .unwrap_or_default(),
        max_fields_per_live_file: live_fields_per_file.last().copied().unwrap_or_default(),
    }
}

async fn measure_cold_pair(
    uri: &str,
    before_version: u64,
    after_version: u64,
    fields: &[String],
    iterations: usize,
) -> Result<(Vec<Sample>, Vec<Sample>)> {
    let mut before = Vec::with_capacity(iterations);
    let mut after = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        if iteration % 2 == 0 {
            before.push(measure_fresh(uri, before_version, fields).await?);
            after.push(measure_fresh(uri, after_version, fields).await?);
        } else {
            after.push(measure_fresh(uri, after_version, fields).await?);
            before.push(measure_fresh(uri, before_version, fields).await?);
        }
    }
    Ok((before, after))
}

async fn measure_fresh(uri: &str, version: u64, fields: &[String]) -> Result<Sample> {
    let open_started = Instant::now();
    let dataset = DatasetBuilder::from_uri(uri)
        .with_version(version)
        .load()
        .await?;
    let open_micros = open_started.elapsed().as_micros();
    let store = dataset.object_store(None).await?;
    let after_open = store.io_stats_snapshot();
    let mut sample = measure_query(&dataset, fields).await?;
    sample.open_micros = open_micros;
    sample.open_read_iops = after_open.read_iops;
    sample.open_read_bytes = after_open.read_bytes;
    Ok(sample)
}

async fn measure_warm_pair(
    uri: &str,
    before_version: u64,
    after_version: u64,
    fields: &[String],
    iterations: usize,
) -> Result<(Vec<Sample>, Vec<Sample>)> {
    let before_dataset = DatasetBuilder::from_uri(uri)
        .with_version(before_version)
        .load()
        .await?;
    let after_dataset = DatasetBuilder::from_uri(uri)
        .with_version(after_version)
        .load()
        .await?;
    measure_query(&before_dataset, fields).await?;
    measure_query(&after_dataset, fields).await?;

    let mut before = Vec::with_capacity(iterations);
    let mut after = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        if iteration % 2 == 0 {
            before.push(measure_query(&before_dataset, fields).await?);
            after.push(measure_query(&after_dataset, fields).await?);
        } else {
            after.push(measure_query(&after_dataset, fields).await?);
            before.push(measure_query(&before_dataset, fields).await?);
        }
    }
    Ok((before, after))
}

async fn measure_query(dataset: &Dataset, fields: &[String]) -> Result<Sample> {
    let store = dataset.object_store(None).await?;
    let io_before = store.io_stats_snapshot();
    let query_started = Instant::now();
    let mut scanner = dataset.scan();
    let field_refs = fields.iter().map(String::as_str).collect::<Vec<_>>();
    scanner.project(&field_refs)?;
    let mut stream = scanner.try_into_stream().await?;
    let plan_micros = query_started.elapsed().as_micros();
    let io_after_plan = store.io_stats_snapshot();
    let first = stream.try_next().await?;
    let time_to_first_batch_micros = query_started.elapsed().as_micros();
    let io_after_first_batch = store.io_stats_snapshot();
    let mut checksum = first.as_ref().map(batch_checksum).unwrap_or_default();
    while let Some(batch) = stream.try_next().await? {
        checksum += batch_checksum(&batch);
    }
    let full_stream_micros = query_started.elapsed().as_micros();
    let io_after = store.io_stats_snapshot();
    Ok(Sample {
        open_micros: 0,
        plan_micros,
        time_to_first_batch_micros,
        full_stream_micros,
        open_read_iops: 0,
        open_read_bytes: 0,
        plan_read_iops: io_after_plan.read_iops.saturating_sub(io_before.read_iops),
        plan_read_bytes: io_after_plan
            .read_bytes
            .saturating_sub(io_before.read_bytes),
        first_batch_read_iops: io_after_first_batch
            .read_iops
            .saturating_sub(io_before.read_iops),
        first_batch_read_bytes: io_after_first_batch
            .read_bytes
            .saturating_sub(io_before.read_bytes),
        query_read_iops: io_after.read_iops.saturating_sub(io_before.read_iops),
        query_read_bytes: io_after.read_bytes.saturating_sub(io_before.read_bytes),
        checksum,
    })
}

fn batch_checksum(batch: &RecordBatch) -> i128 {
    batch
        .columns()
        .iter()
        .map(|column| {
            if let Some(values) = column.as_any().downcast_ref::<Int32Array>() {
                values.iter().flatten().map(i128::from).sum::<i128>()
            } else if let Some(values) = column.as_any().downcast_ref::<Int64Array>() {
                values.iter().flatten().map(i128::from).sum::<i128>()
            } else {
                0
            }
        })
        .sum()
}

fn summarize(samples: Vec<Sample>) -> Result<PhaseSummary> {
    let checksum = samples
        .first()
        .map(|sample| sample.checksum)
        .unwrap_or_default();
    if let Some(changed) = samples.iter().find(|sample| sample.checksum != checksum) {
        return Err(Error::internal(format!(
            "query checksum changed between iterations: {checksum} != {}",
            changed.checksum
        )));
    }
    Ok(PhaseSummary {
        open_micros: distribution(samples.iter().map(|sample| sample.open_micros)),
        plan_micros: distribution(samples.iter().map(|sample| sample.plan_micros)),
        time_to_first_batch_micros: distribution(
            samples
                .iter()
                .map(|sample| sample.time_to_first_batch_micros),
        ),
        full_stream_micros: distribution(samples.iter().map(|sample| sample.full_stream_micros)),
        open_read_iops: distribution(samples.iter().map(|sample| sample.open_read_iops as u128)),
        open_read_bytes: distribution(samples.iter().map(|sample| sample.open_read_bytes as u128)),
        plan_read_iops: distribution(samples.iter().map(|sample| sample.plan_read_iops as u128)),
        plan_read_bytes: distribution(samples.iter().map(|sample| sample.plan_read_bytes as u128)),
        first_batch_read_iops: distribution(
            samples
                .iter()
                .map(|sample| sample.first_batch_read_iops as u128),
        ),
        first_batch_read_bytes: distribution(
            samples
                .iter()
                .map(|sample| sample.first_batch_read_bytes as u128),
        ),
        query_read_iops: distribution(samples.iter().map(|sample| sample.query_read_iops as u128)),
        query_read_bytes: distribution(
            samples.iter().map(|sample| sample.query_read_bytes as u128),
        ),
        checksum,
    })
}

fn distribution(values: impl Iterator<Item = u128>) -> Distribution {
    let mut samples = values.collect::<Vec<_>>();
    samples.sort_unstable();
    let median = percentile(&samples, 0.5);
    let p95 = percentile(&samples, 0.95);
    Distribution {
        samples,
        median,
        p95,
    }
}

fn percentile(samples: &[u128], percentile: f64) -> u128 {
    let index = ((samples.len() - 1) as f64 * percentile).ceil() as usize;
    samples[index]
}

async fn create_baseline(config: &Config) -> Result<Dataset> {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    let id = Arc::new(Int64Array::from_iter_values(0..config.rows as i64));
    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![id])?;
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], arrow_schema),
        &config.uri,
        Some(lance::dataset::WriteParams {
            max_rows_per_file: config.rows.div_ceil(config.fragments),
            ..Default::default()
        }),
    )
    .await?;

    let feature_schema = Arc::new(ArrowSchema::new(
        (0..config.fields)
            .map(|field| Field::new(feature_name(field), DataType::Int32, true))
            .collect::<Vec<_>>(),
    ));
    dataset
        .add_columns(NewColumnTransform::AllNulls(feature_schema), None, None)
        .await?;

    for field_idx in 0..config.fields {
        let name = feature_name(field_idx);
        let field = dataset.schema().field(&name).ok_or_else(|| {
            Error::internal(format!("declared benchmark field '{name}' is missing"))
        })?;
        let schema = Schema {
            fields: vec![field.clone()],
            metadata: Default::default(),
        };
        let replacements = stream::iter(dataset.get_fragments().into_iter().map(|fragment| {
            let schema = schema.clone();
            let name = name.clone();
            async move {
                let rows = fragment.physical_rows().await?;
                let values = Int32Array::from_iter_values(
                    (0..rows).map(|row| field_idx as i32 * 1_000_003 + row as i32),
                );
                let batch_schema = Arc::new(ArrowSchema::new(vec![Field::new(
                    name,
                    DataType::Int32,
                    true,
                )]));
                let batch = RecordBatch::try_new(batch_schema, vec![Arc::new(values)])?;
                fragment
                    .write_column(stream::iter([Ok(batch)]), &schema)
                    .await
            }
        }))
        .buffer_unordered(config.concurrency)
        .try_collect::<Vec<_>>()
        .await?;
        let read_version = dataset.version().version;
        dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(Transaction::new(
                read_version,
                Operation::DataReplacement { replacements },
                None,
            ))
            .await?;
        if (field_idx + 1) % 50 == 0 || field_idx + 1 == config.fields {
            writeln!(
                std::io::stderr().lock(),
                "materialized {}/{} features",
                field_idx + 1,
                config.fields
            )?;
        }
    }
    Ok(dataset)
}
