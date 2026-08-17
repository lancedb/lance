// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{Array, Int32Array, Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use futures::{StreamExt, TryStreamExt, stream};
use lance::dataset::optimize_columns::{
    ColumnGroup, OptimizeColumnsMetrics, OptimizeColumnsOptions,
};
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

#[derive(Debug, Serialize)]
struct QuerySummary {
    samples_micros: Vec<u128>,
    median_micros: u128,
    p95_micros: u128,
    checksum: i128,
}

#[derive(Debug, Serialize)]
struct BenchmarkResult {
    uri: String,
    fields: usize,
    rows: usize,
    fragments: usize,
    group_size: usize,
    iterations: usize,
    files_before: usize,
    files_after: usize,
    manifest_version_before: u64,
    manifest_version_after: u64,
    before: QuerySummary,
    after: QuerySummary,
    optimize_wall_millis: u128,
    optimize_metrics: OptimizeColumnsMetrics,
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = parse_args()?;
    eprintln!("creating one-file-per-feature baseline at {}", config.uri);
    let mut dataset = create_baseline(&config).await?;
    let query_fields = ["id", "feature_0000", "feature_0001", "feature_0002"];
    let files_before = dataset
        .column_layout_stats()
        .iter()
        .map(|stats| stats.live_file_count)
        .sum();
    let manifest_version_before = dataset.version().version;
    let before = measure_queries(&dataset, &query_fields, config.iterations).await?;

    let groups = (0..config.fields)
        .collect::<Vec<_>>()
        .chunks(config.group_size)
        .map(|chunk| ColumnGroup {
            fields: chunk
                .iter()
                .map(|field| format!("feature_{field:04}"))
                .collect(),
        })
        .collect();
    let started = Instant::now();
    let optimize_metrics = dataset
        .optimize_columns(OptimizeColumnsOptions {
            groups,
            fragment_ids: None,
            max_concurrency: Some(config.concurrency),
        })
        .await?;
    let optimize_wall_millis = started.elapsed().as_millis();

    let after = measure_queries(&dataset, &query_fields, config.iterations).await?;
    if before.checksum != after.checksum {
        return Err(Error::internal(format!(
            "OptimizeColumns changed query checksum from {} to {}",
            before.checksum, after.checksum
        )));
    }
    let files_after = dataset
        .column_layout_stats()
        .iter()
        .map(|stats| stats.live_file_count)
        .sum();
    let result = BenchmarkResult {
        uri: config.uri,
        fields: config.fields,
        rows: config.rows,
        fragments: config.fragments,
        group_size: config.group_size,
        iterations: config.iterations,
        files_before,
        files_after,
        manifest_version_before,
        manifest_version_after: dataset.version().version,
        before,
        after,
        optimize_wall_millis,
        optimize_metrics,
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn parse_args() -> Result<Config> {
    let mut args = std::env::args().skip(1);
    let uri = args.next().ok_or_else(|| {
        Error::invalid_input(
            "usage: optimize_columns_bench <uri> [fields] [rows] [fragments] [group_size] [iterations] [concurrency]",
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
    let fields = parse(args.next(), 128, "fields")?;
    let rows = parse(args.next(), 1_000_000, "rows")?;
    let fragments = parse(args.next(), 8, "fragments")?;
    let group_size = parse(args.next(), 16, "group_size")?;
    let iterations = parse(args.next(), 15, "iterations")?;
    let concurrency = parse(args.next(), 8, "concurrency")?;
    if fields < 3
        || rows == 0
        || fragments == 0
        || group_size == 0
        || iterations == 0
        || concurrency == 0
    {
        return Err(Error::invalid_input(
            "fields must be at least 3 and all other numeric arguments must be positive",
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
            .map(|field| Field::new(format!("feature_{field:04}"), DataType::Int32, true))
            .collect::<Vec<_>>(),
    ));
    dataset
        .add_columns(NewColumnTransform::AllNulls(feature_schema), None, None)
        .await?;

    for field_idx in 0..config.fields {
        let name = format!("feature_{field_idx:04}");
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
        if (field_idx + 1) % 16 == 0 || field_idx + 1 == config.fields {
            eprintln!("materialized {}/{} features", field_idx + 1, config.fields);
        }
    }
    Ok(dataset)
}

async fn measure_queries(
    dataset: &Dataset,
    fields: &[&str],
    iterations: usize,
) -> Result<QuerySummary> {
    let mut samples = Vec::with_capacity(iterations);
    let mut checksum = None;
    for iteration in 0..iterations + 1 {
        let started = Instant::now();
        let mut scanner = dataset.scan();
        scanner.project(fields)?;
        let mut stream = scanner.try_into_stream().await?;
        let mut current_checksum = 0_i128;
        while let Some(batch) = stream.try_next().await? {
            for column in batch.columns() {
                if let Some(values) = column.as_any().downcast_ref::<Int32Array>() {
                    current_checksum += values.iter().flatten().map(i128::from).sum::<i128>();
                } else if let Some(values) = column.as_any().downcast_ref::<Int64Array>() {
                    current_checksum += values.iter().flatten().map(i128::from).sum::<i128>();
                }
            }
        }
        let elapsed = started.elapsed();
        if let Some(expected) = checksum {
            if expected != current_checksum {
                return Err(Error::internal(format!(
                    "query checksum changed between iterations: {expected} != {current_checksum}"
                )));
            }
        } else {
            checksum = Some(current_checksum);
        }
        if iteration > 0 {
            samples.push(elapsed.as_micros());
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    samples.sort_unstable();
    let median_micros = percentile(&samples, 0.5);
    let p95_micros = percentile(&samples, 0.95);
    Ok(QuerySummary {
        samples_micros: samples,
        median_micros,
        p95_micros,
        checksum: checksum.unwrap_or_default(),
    })
}

fn percentile(samples: &[u128], percentile: f64) -> u128 {
    let index = ((samples.len() - 1) as f64 * percentile).ceil() as usize;
    samples[index]
}
