// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark comparing V2 vs V3 manifest naming scheme performance
//!
//! This benchmark compares performance between V2 and V3 manifest schemes for:
//! 1. checkout_latest operations
//! 2. concurrent commits with conflict resolution
//!
//! Storage backend can be controlled via LANCE_BENCH_STORAGE_PREFIX environment variable:
//! - If not set: uses a temporary directory (default)
//! - If set to a prefix: uses that prefix for the dataset path
//!   - Example: LANCE_BENCH_STORAGE_PREFIX=s3://my-bucket/path/to/
//!   - Example: LANCE_BENCH_STORAGE_PREFIX=/tmp/lance/
//!   - Example: LANCE_BENCH_STORAGE_PREFIX=memory://
//!
//! Usage:
//! ```
//! # Test with temporary directory (default)
//! cargo bench --bench manifest_scheme
//!
//! # Test with S3 storage
//! LANCE_BENCH_STORAGE_PREFIX=s3://my-bucket/benchmarks/ cargo bench --bench manifest_scheme
//!
//! # Test with memory storage
//! LANCE_BENCH_STORAGE_PREFIX=memory:// cargo bench --bench manifest_scheme
//! ```

#![allow(clippy::print_stdout)]

use std::sync::Arc;

use arrow_array::{Float32Array, Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{criterion_group, criterion_main, Criterion};
use lance::dataset::{Dataset, WriteParams};
use lance_table::io::commit::ManifestNamingScheme;
#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

/// Get storage URI prefix based on environment variable
fn get_storage_prefix() -> String {
    std::env::var("LANCE_BENCH_STORAGE_PREFIX").unwrap_or_else(|_| {
        // Use a temporary directory if no prefix is specified
        let temp_dir = std::env::temp_dir();
        let bench_dir = temp_dir.join("lance_bench");
        format!("{}/", bench_dir.display())
    })
}

/// Get storage type name for display
fn get_storage_type() -> String {
    let prefix = get_storage_prefix();
    if prefix.starts_with("s3://") {
        "s3".to_string()
    } else if prefix.starts_with("memory://") {
        "memory".to_string()
    } else if prefix.starts_with("gs://") {
        "gcs".to_string()
    } else if prefix.starts_with("az://") {
        "azure".to_string()
    } else {
        "local".to_string()
    }
}

/// Create a test dataset with specified number of versions and manifest scheme
/// Only creates the dataset if it doesn't exist, otherwise returns the existing one
async fn create_test_dataset(
    base_uri: &str,
    num_versions: u64,
    manifest_scheme: ManifestNamingScheme,
) -> Dataset {
    // Try to open existing dataset first
    if let Ok(mut existing_dataset) = Dataset::open(base_uri).await {
        // Check if it has the expected number of versions
        let current_version = existing_dataset.version().version;
        if current_version >= num_versions - 1 {
            // Dataset exists with enough versions, return it
            return existing_dataset;
        }
        // Add more versions if needed
        for i in (current_version + 1)..num_versions {
            existing_dataset
                .update_config([(format!("version_{}", i), i.to_string())])
                .await
                .unwrap();
        }
        return existing_dataset;
    }

    // Dataset doesn't exist, create it
    let write_params = WriteParams {
        enable_v2_manifest_paths: matches!(manifest_scheme, ManifestNamingScheme::V2),
        enable_v3_manifest_paths: matches!(manifest_scheme, ManifestNamingScheme::V3),
        max_rows_per_file: 100,
        max_rows_per_group: 50,
        ..Default::default()
    };

    // Create initial dataset using into_reader_rows and Dataset::write pattern
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Float32, false),
    ]));

    let batch_size = 50;
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..batch_size)),
            Arc::new(Float32Array::from_iter_values(
                (0..batch_size).map(|x| x as f32 * 0.1),
            )),
        ],
    )
    .unwrap();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let mut dataset = Dataset::write(reader, base_uri, Some(write_params))
        .await
        .unwrap();

    // Add additional versions using lightweight update_config operations
    for i in 1..num_versions {
        dataset
            .update_config([(format!("version_{}", i), i.to_string())])
            .await
            .unwrap();
    }

    dataset
}

fn bench_checkout_latest(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let storage_prefix = get_storage_prefix();
    let storage_type = get_storage_type();

    for num_versions in [10, 50, 100] {
        for scheme in [ManifestNamingScheme::V2, ManifestNamingScheme::V3] {
            let scheme_name = match scheme {
                ManifestNamingScheme::V2 => "V2",
                ManifestNamingScheme::V3 => "V3",
                _ => unreachable!(),
            };

            let dataset_uri = format!(
                "{}bench_dataset_{}_{}.lance",
                storage_prefix,
                scheme_name.to_lowercase(),
                num_versions
            );
            let dataset = rt.block_on(create_test_dataset(&dataset_uri, num_versions, scheme));
            let latest_version = dataset.version().version;
            // Start from an older version to test checkout_latest
            let start_version = if num_versions > 5 {
                latest_version - 5
            } else {
                0
            };

            c.bench_function(
                &format!(
                    "checkout_latest_{} ({} versions, {})",
                    scheme_name, num_versions, storage_type
                ),
                |b| {
                    b.to_async(&rt).iter(|| async {
                        // Open dataset at older version then checkout latest
                        let mut ds = Dataset::open(&dataset_uri).await.unwrap();
                        ds = ds.checkout_version(start_version).await.unwrap();
                        ds.checkout_latest().await.unwrap();
                        assert_eq!(ds.version().version, latest_version);
                    })
                },
            );
        }
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .significance_level(0.01)
        .sample_size(50)
        .warm_up_time(std::time::Duration::from_secs(5))
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_checkout_latest
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .significance_level(0.01)
        .sample_size(50)
        .warm_up_time(std::time::Duration::from_secs(5));
    targets = bench_checkout_latest
);

criterion_main!(benches);
