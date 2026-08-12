// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end S3 + DynamoDB benchmark for external-manifest finalization.
//!
//! The happy path measures one helper materializing a reserved staging
//! manifest. The contention path releases many helpers against the same
//! reservation at once, exercising repeated overwrite, HEAD, external-index
//! publication, and staging cleanup. Setup is outside the timed interval.
//!
//! Required environment variables:
//! - `S3_BUCKET`: bucket used for benchmark objects
//! - `DDB_TABLE`: DynamoDB table with `(base_uri STRING, version NUMBER)` keys
//!
//! Optional environment variables:
//! - `AWS_REGION` (default `us-east-2`)
//! - `BENCH_PREFIX` (default `lance/external-manifest-finalization`)
//! - `HAPPY_SAMPLES` (default `100`)
//! - `CONTENTION_ROUNDS` (default `30`)
//! - `CONTENDERS` (default `32`)
//! - `MANIFEST_BYTES` (default `65536`)

#![allow(clippy::print_stderr, clippy::print_stdout)]

use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aws_config::{BehaviorVersion, Region};
use aws_sdk_dynamodb::Client;
use bytes::Bytes;
use lance_io::object_store::ObjectStore;
use lance_table::io::commit::dynamodb::DynamoDBExternalManifestStore;
use lance_table::io::commit::external_manifest::{
    ExternalManifestCommitHandler, ExternalManifestStore,
};
use lance_table::io::commit::{CommitHandler, ManifestNamingScheme};
use object_store::{ObjectStoreExt as _, PutPayload, path::Path};
use serde_json::json;
use tokio::sync::Barrier;
use uuid::Uuid;

type BenchResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const VERSION: u64 = 1;

struct Config {
    region: String,
    bucket: String,
    table: String,
    prefix: String,
    happy_samples: usize,
    contention_rounds: usize,
    contenders: usize,
    manifest_bytes: usize,
}

impl Config {
    fn from_env() -> BenchResult<Self> {
        Ok(Self {
            region: env_or("AWS_REGION", "us-east-2"),
            bucket: required_env("S3_BUCKET")?,
            table: required_env("DDB_TABLE")?,
            prefix: env_or("BENCH_PREFIX", "lance/external-manifest-finalization"),
            happy_samples: usize_env("HAPPY_SAMPLES", 100)?,
            contention_rounds: usize_env("CONTENTION_ROUNDS", 30)?,
            contenders: usize_env("CONTENDERS", 32)?,
            manifest_bytes: usize_env("MANIFEST_BYTES", 65_536)?,
        })
    }
}

fn required_env(name: &str) -> BenchResult<String> {
    std::env::var(name).map_err(|_| format!("{name} must be set").into())
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn usize_env(name: &str, default: usize) -> BenchResult<usize> {
    let value = std::env::var(name).unwrap_or_else(|_| default.to_string());
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {name}={value:?}: {error}"))?;
    if parsed == 0 {
        return Err(format!("{name} must be greater than zero").into());
    }
    Ok(parsed)
}

struct Fixture {
    base_path: Path,
}

async fn prepare_fixture(
    object_store: &ObjectStore,
    external_store: &dyn ExternalManifestStore,
    run_root: &Path,
    scenario: &str,
    payload: &Bytes,
) -> BenchResult<Fixture> {
    let base_path = run_root
        .clone()
        .join(scenario)
        .join(Uuid::new_v4().to_string());
    let final_path = ManifestNamingScheme::V2.manifest_path(&base_path, VERSION);
    let staging_path = Path::parse(format!("{final_path}-{}", Uuid::new_v4()))?;

    object_store
        .inner
        .put(&staging_path, PutPayload::from(payload.clone()))
        .await?;
    let staging_meta = object_store.inner.head(&staging_path).await?;
    external_store
        .put_if_not_exists(
            base_path.as_ref(),
            VERSION,
            staging_path.as_ref(),
            staging_meta.size,
            staging_meta.e_tag,
        )
        .await?;

    Ok(Fixture { base_path })
}

fn percentile_ms(samples: &[Duration], percentile: f64) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let mut nanos: Vec<u128> = samples.iter().map(Duration::as_nanos).collect();
    nanos.sort_unstable();
    let index = ((nanos.len() - 1) as f64 * percentile).round() as usize;
    nanos[index] as f64 / 1_000_000.0
}

fn mean_ms(samples: &[Duration]) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    samples.iter().map(Duration::as_secs_f64).sum::<f64>() * 1_000.0 / samples.len() as f64
}

fn summary(scenario: &str, durations: &[Duration], errors: usize) -> serde_json::Value {
    json!({
        "scenario": scenario,
        "operations": durations.len() + errors,
        "successes": durations.len(),
        "errors": errors,
        "mean_ms": mean_ms(durations),
        "p50_ms": percentile_ms(durations, 0.50),
        "p95_ms": percentile_ms(durations, 0.95),
        "p99_ms": percentile_ms(durations, 0.99),
    })
}

async fn run_happy_path(
    handler: Arc<ExternalManifestCommitHandler>,
    external_store: Arc<dyn ExternalManifestStore>,
    object_store: Arc<ObjectStore>,
    run_root: &Path,
    payload: &Bytes,
    samples: usize,
) -> BenchResult<serde_json::Value> {
    let mut durations = Vec::with_capacity(samples);
    let mut errors = 0;

    for _ in 0..samples {
        let fixture = prepare_fixture(
            object_store.as_ref(),
            external_store.as_ref(),
            run_root,
            "happy",
            payload,
        )
        .await?;
        let started = Instant::now();
        let result = handler
            .resolve_version_location(&fixture.base_path, VERSION, object_store.inner.as_ref())
            .await;
        let elapsed = started.elapsed();
        match result {
            Ok(_) => durations.push(elapsed),
            Err(error) => {
                errors += 1;
                eprintln!("happy-path finalization failed: {error}");
            }
        }
    }

    Ok(summary("happy_path", &durations, errors))
}

async fn run_contention(
    handler: Arc<ExternalManifestCommitHandler>,
    external_store: Arc<dyn ExternalManifestStore>,
    object_store: Arc<ObjectStore>,
    run_root: &Path,
    payload: &Bytes,
    rounds: usize,
    contenders: usize,
) -> BenchResult<serde_json::Value> {
    let mut task_durations = Vec::with_capacity(rounds * contenders);
    let mut burst_durations = Vec::with_capacity(rounds);
    let mut task_errors = 0;
    let mut final_read_errors = 0;

    for _ in 0..rounds {
        let fixture = prepare_fixture(
            object_store.as_ref(),
            external_store.as_ref(),
            run_root,
            "contention",
            payload,
        )
        .await?;
        let barrier = Arc::new(Barrier::new(contenders + 1));
        let mut tasks = Vec::with_capacity(contenders);

        for _ in 0..contenders {
            let task_handler = handler.clone();
            let task_store = object_store.clone();
            let task_base = fixture.base_path.clone();
            let task_barrier = barrier.clone();
            tasks.push(tokio::spawn(async move {
                task_barrier.wait().await;
                let started = Instant::now();
                let result = task_handler
                    .resolve_version_location(&task_base, VERSION, task_store.inner.as_ref())
                    .await;
                (started.elapsed(), result)
            }));
        }

        let burst_started = Instant::now();
        barrier.wait().await;
        for task in tasks {
            let (elapsed, result) = task.await?;
            match result {
                Ok(_) => task_durations.push(elapsed),
                Err(error) => {
                    task_errors += 1;
                    if task_errors <= 3 {
                        eprintln!("contended finalization failed: {error}");
                    }
                }
            }
        }
        burst_durations.push(burst_started.elapsed());

        // A burst can finish with a stale legacy ETag even when every copy
        // contained the same bytes. Track whether a fresh reader can consume
        // the final state; this is a correctness signal, outside the timing.
        if let Err(error) = handler
            .resolve_version_location(&fixture.base_path, VERSION, object_store.inner.as_ref())
            .await
        {
            final_read_errors += 1;
            if final_read_errors <= 3 {
                eprintln!("post-contention read failed: {error}");
            }
        }
    }

    let mut result = summary("high_contention_task", &task_durations, task_errors);
    let object = result
        .as_object_mut()
        .expect("the benchmark summary is always a JSON object");
    object.insert("rounds".to_string(), json!(rounds));
    object.insert("contenders".to_string(), json!(contenders));
    object.insert(
        "burst_mean_ms".to_string(),
        json!(mean_ms(&burst_durations)),
    );
    object.insert(
        "burst_p50_ms".to_string(),
        json!(percentile_ms(&burst_durations, 0.50)),
    );
    object.insert(
        "burst_p95_ms".to_string(),
        json!(percentile_ms(&burst_durations, 0.95)),
    );
    object.insert(
        "post_contention_read_errors".to_string(),
        json!(final_read_errors),
    );
    Ok(result)
}

#[tokio::main]
async fn main() -> BenchResult<()> {
    let config = Config::from_env()?;
    let sdk_config = aws_config::defaults(BehaviorVersion::latest())
        .region(Region::new(config.region.clone()))
        .load()
        .await;
    let external_store = DynamoDBExternalManifestStore::new_external_store(
        Arc::new(Client::new(&sdk_config)),
        &config.table,
        "external-manifest-finalization-bench",
    )
    .await?;
    let handler = Arc::new(ExternalManifestCommitHandler {
        external_manifest_store: external_store.clone(),
    });

    let run_id = Uuid::new_v4();
    let uri = format!(
        "s3://{}/{}/{}",
        config.bucket,
        config.prefix.trim_matches('/'),
        run_id
    );
    let (object_store, run_root) = ObjectStore::from_uri(&uri).await?;
    let payload = Bytes::from(vec![0x5a; config.manifest_bytes]);

    println!(
        "{}",
        json!({
            "scenario": "configuration",
            "revision": std::env::var("BENCH_REVISION").ok(),
            "region": config.region,
            "bucket": config.bucket,
            "table": config.table,
            "run_id": run_id,
            "happy_samples": config.happy_samples,
            "contention_rounds": config.contention_rounds,
            "contenders": config.contenders,
            "manifest_bytes": config.manifest_bytes,
        })
    );

    let happy = run_happy_path(
        handler.clone(),
        external_store.clone(),
        object_store.clone(),
        &run_root,
        &payload,
        config.happy_samples,
    )
    .await?;
    println!("{happy}");

    let contention = run_contention(
        handler,
        external_store,
        object_store,
        &run_root,
        &payload,
        config.contention_rounds,
        config.contenders,
    )
    .await?;
    println!("{contention}");

    Ok(())
}
