// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark comparing query performance across different index strategies:
//! - No index (full scan with filter)
//! - BTree index on first column only
//! - Compound index on multiple columns
//!
//! This demonstrates where compound indices provide value over single-column indices.

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use criterion::{criterion_group, criterion_main, Criterion};
use futures::TryStreamExt;
use lance::Dataset;
use lance_core::utils::tempfile::TempStrDir;
use lance_index::{scalar::ScalarIndexParams, DatasetIndexExt, IndexType};

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

/// Number of unique tenants
const NUM_TENANTS: i64 = 100;
/// Number of timestamps per tenant
const TIMESTAMPS_PER_TENANT: i64 = 10_000;
/// Total rows = NUM_TENANTS * TIMESTAMPS_PER_TENANT = 1,000,000

// Multi-fragment constants
/// Number of fragments for multi-fragment benchmark
const NUM_FRAGMENTS: i64 = 10;
/// Timestamps per fragment (each fragment covers this many timestamps for ALL tenants)
const TIMESTAMPS_PER_FRAGMENT: i64 = TIMESTAMPS_PER_TENANT / NUM_FRAGMENTS; // 1000

struct ComparisonFixture {
    _no_index_dir: TempStrDir,
    _btree_dir: TempStrDir,
    _compound_dir: TempStrDir,
    no_index_dataset: Arc<Dataset>,
    btree_dataset: Arc<Dataset>,
    compound_dataset: Arc<Dataset>,
}

fn generate_test_data() -> RecordBatch {
    // Generate tenant_ids: "tenant_00" through "tenant_99", repeated for each timestamp
    let tenant_ids: Vec<String> = (0..NUM_TENANTS)
        .flat_map(|t| {
            (0..TIMESTAMPS_PER_TENANT).map(move |_| format!("tenant_{:02}", t))
        })
        .collect();

    // Generate timestamps: 0 to TIMESTAMPS_PER_TENANT-1 for each tenant
    let timestamps: Vec<i64> = (0..NUM_TENANTS)
        .flat_map(|_| 0..TIMESTAMPS_PER_TENANT)
        .collect();

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("tenant_id", DataType::Utf8, false),
        ArrowField::new("timestamp", DataType::Int64, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(tenant_ids)),
            Arc::new(Int64Array::from(timestamps)),
        ],
    )
    .expect("Failed to create record batch")
}

impl ComparisonFixture {
    async fn open() -> Self {
        let batch = generate_test_data();
        let schema = batch.schema();

        // Create three separate directories for each dataset variant
        let no_index_dir = TempStrDir::default();
        let btree_dir = TempStrDir::default();
        let compound_dir = TempStrDir::default();

        let no_index_uri = format!("file://{}", no_index_dir.as_str());
        let btree_uri = format!("file://{}", btree_dir.as_str());
        let compound_uri = format!("file://{}", compound_dir.as_str());

        // Write the same data to all three locations
        let batches_no_index = RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone());
        let batches_btree = RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone());
        let batches_compound = RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone());

        let no_index_dataset = Arc::new(
            Dataset::write(batches_no_index, &no_index_uri, None)
                .await
                .expect("Failed to write no_index dataset"),
        );

        let mut btree_dataset = Dataset::write(batches_btree, &btree_uri, None)
            .await
            .expect("Failed to write btree dataset");

        let mut compound_dataset = Dataset::write(batches_compound, &compound_uri, None)
            .await
            .expect("Failed to write compound dataset");

        // Create BTree index on tenant_id only
        let params = ScalarIndexParams::default();
        btree_dataset
            .create_index(
                &["tenant_id"],
                IndexType::BTree,
                Some("btree_tenant_idx".to_string()),
                &params,
                false,
            )
            .await
            .expect("Failed to create btree index");

        // Create Compound index on (tenant_id, timestamp)
        compound_dataset
            .create_index(
                &["tenant_id", "timestamp"],
                IndexType::BTree,
                Some("compound_idx".to_string()),
                &params,
                false,
            )
            .await
            .expect("Failed to create compound index");

        // Re-open datasets to ensure indices are loaded
        let btree_dataset = Arc::new(
            Dataset::open(&btree_uri)
                .await
                .expect("Failed to reopen btree dataset"),
        );
        let compound_dataset = Arc::new(
            Dataset::open(&compound_uri)
                .await
                .expect("Failed to reopen compound dataset"),
        );

        Self {
            _no_index_dir: no_index_dir,
            _btree_dir: btree_dir,
            _compound_dir: compound_dir,
            no_index_dataset,
            btree_dataset,
            compound_dataset,
        }
    }
}

/// Helper to run a query and count rows
async fn run_query(dataset: &Dataset, filter: &str) -> usize {
    let mut stream = dataset
        .scan()
        .filter(filter)
        .expect("Failed to apply filter")
        .try_into_stream()
        .await
        .expect("Failed to create stream");

    let mut num_rows = 0;
    while let Some(batch) = stream.try_next().await.expect("Stream error") {
        num_rows += batch.num_rows();
    }
    num_rows
}

/// Query: WHERE tenant_id = 'tenant_50'
/// Expected: ~10,000 rows (all timestamps for that tenant)
/// NOTE: Compound index is NOT used for single-column queries (observed behavior)
async fn query_tenant_only(dataset: &Dataset) -> usize {
    run_query(dataset, "tenant_id = 'tenant_50'").await
}

/// Query: WHERE tenant_id = 'tenant_50' AND timestamp > 9900
/// Expected: ~99 rows (1% of tenant's timestamps)
/// This is where compound index should shine - both predicates use the index
async fn query_tenant_narrow_range(dataset: &Dataset) -> usize {
    run_query(dataset, "tenant_id = 'tenant_50' AND timestamp > 9900").await
}

/// Query: WHERE tenant_id = 'tenant_50' AND timestamp > 5000
/// Expected: ~4,999 rows (50% of tenant's timestamps)
/// Compound should still help but less dramatically
async fn query_tenant_wide_range(dataset: &Dataset) -> usize {
    run_query(dataset, "tenant_id = 'tenant_50' AND timestamp > 5000").await
}

/// Query: WHERE tenant_id = 'tenant_50' AND timestamp > 0
/// Expected: ~9,999 rows (nearly all of tenant's timestamps)
/// BTree and Compound should be similar - range doesn't filter much
async fn query_tenant_full_range(dataset: &Dataset) -> usize {
    run_query(dataset, "tenant_id = 'tenant_50' AND timestamp > 0").await
}

/// Query: WHERE timestamp > 9900
/// Expected: ~9,900 rows (99 rows per tenant * 100 tenants)
/// Neither index helps - violates leftmost prefix rule for compound
async fn query_timestamp_only(dataset: &Dataset) -> usize {
    run_query(dataset, "timestamp > 9900").await
}

fn bench_tenant_only(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let fixture = rt.block_on(ComparisonFixture::open());

    // Verify correctness first
    let expected_rows = TIMESTAMPS_PER_TENANT as usize; // 10,000 rows for tenant_50
    
    let no_index_rows = rt.block_on(query_tenant_only(&fixture.no_index_dataset));
    assert_eq!(no_index_rows, expected_rows, "no_index returned wrong row count");
    
    let btree_rows = rt.block_on(query_tenant_only(&fixture.btree_dataset));
    assert_eq!(btree_rows, expected_rows, "btree returned wrong row count");
    
    let compound_rows = rt.block_on(query_tenant_only(&fixture.compound_dataset));
    assert_eq!(compound_rows, expected_rows, "compound returned wrong row count");

    let mut group = c.benchmark_group("tenant_only");

    group.bench_function("no_index", |b| {
        b.iter(|| rt.block_on(query_tenant_only(&fixture.no_index_dataset)))
    });

    group.bench_function("btree_tenant", |b| {
        b.iter(|| rt.block_on(query_tenant_only(&fixture.btree_dataset)))
    });

    group.bench_function("compound", |b| {
        b.iter(|| rt.block_on(query_tenant_only(&fixture.compound_dataset)))
    });

    group.finish();
}

fn bench_tenant_narrow_range(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let fixture = rt.block_on(ComparisonFixture::open());

    // Verify correctness: timestamps 9901-9999 = 99 rows
    let expected_rows = 99;
    
    let no_index_rows = rt.block_on(query_tenant_narrow_range(&fixture.no_index_dataset));
    assert_eq!(no_index_rows, expected_rows, "no_index returned wrong row count");
    
    let btree_rows = rt.block_on(query_tenant_narrow_range(&fixture.btree_dataset));
    assert_eq!(btree_rows, expected_rows, "btree returned wrong row count");
    
    let compound_rows = rt.block_on(query_tenant_narrow_range(&fixture.compound_dataset));
    assert_eq!(compound_rows, expected_rows, "compound returned wrong row count");

    let mut group = c.benchmark_group("tenant_narrow_range");

    group.bench_function("no_index", |b| {
        b.iter(|| rt.block_on(query_tenant_narrow_range(&fixture.no_index_dataset)))
    });

    group.bench_function("btree_tenant", |b| {
        b.iter(|| rt.block_on(query_tenant_narrow_range(&fixture.btree_dataset)))
    });

    group.bench_function("compound", |b| {
        b.iter(|| rt.block_on(query_tenant_narrow_range(&fixture.compound_dataset)))
    });

    group.finish();
}

fn bench_tenant_wide_range(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let fixture = rt.block_on(ComparisonFixture::open());

    // Verify correctness: timestamps 5001-9999 = 4999 rows
    let expected_rows = 4999;
    
    let no_index_rows = rt.block_on(query_tenant_wide_range(&fixture.no_index_dataset));
    assert_eq!(no_index_rows, expected_rows, "no_index returned wrong row count");
    
    let btree_rows = rt.block_on(query_tenant_wide_range(&fixture.btree_dataset));
    assert_eq!(btree_rows, expected_rows, "btree returned wrong row count");
    
    let compound_rows = rt.block_on(query_tenant_wide_range(&fixture.compound_dataset));
    assert_eq!(compound_rows, expected_rows, "compound returned wrong row count");

    let mut group = c.benchmark_group("tenant_wide_range");

    group.bench_function("no_index", |b| {
        b.iter(|| rt.block_on(query_tenant_wide_range(&fixture.no_index_dataset)))
    });

    group.bench_function("btree_tenant", |b| {
        b.iter(|| rt.block_on(query_tenant_wide_range(&fixture.btree_dataset)))
    });

    group.bench_function("compound", |b| {
        b.iter(|| rt.block_on(query_tenant_wide_range(&fixture.compound_dataset)))
    });

    group.finish();
}

fn bench_tenant_full_range(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let fixture = rt.block_on(ComparisonFixture::open());

    // Verify correctness: timestamps 1-9999 = 9999 rows
    let expected_rows = 9999;
    
    let no_index_rows = rt.block_on(query_tenant_full_range(&fixture.no_index_dataset));
    assert_eq!(no_index_rows, expected_rows, "no_index returned wrong row count");
    
    let btree_rows = rt.block_on(query_tenant_full_range(&fixture.btree_dataset));
    assert_eq!(btree_rows, expected_rows, "btree returned wrong row count");
    
    let compound_rows = rt.block_on(query_tenant_full_range(&fixture.compound_dataset));
    assert_eq!(compound_rows, expected_rows, "compound returned wrong row count");

    let mut group = c.benchmark_group("tenant_full_range");

    group.bench_function("no_index", |b| {
        b.iter(|| rt.block_on(query_tenant_full_range(&fixture.no_index_dataset)))
    });

    group.bench_function("btree_tenant", |b| {
        b.iter(|| rt.block_on(query_tenant_full_range(&fixture.btree_dataset)))
    });

    group.bench_function("compound", |b| {
        b.iter(|| rt.block_on(query_tenant_full_range(&fixture.compound_dataset)))
    });

    group.finish();
}

fn bench_timestamp_only(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let fixture = rt.block_on(ComparisonFixture::open());

    // Verify correctness: 99 rows per tenant * 100 tenants = 9900 rows
    let expected_rows = 99 * NUM_TENANTS as usize;
    
    let no_index_rows = rt.block_on(query_timestamp_only(&fixture.no_index_dataset));
    assert_eq!(no_index_rows, expected_rows, "no_index returned wrong row count");
    
    let btree_rows = rt.block_on(query_timestamp_only(&fixture.btree_dataset));
    assert_eq!(btree_rows, expected_rows, "btree returned wrong row count");
    
    let compound_rows = rt.block_on(query_timestamp_only(&fixture.compound_dataset));
    assert_eq!(compound_rows, expected_rows, "compound returned wrong row count");

    let mut group = c.benchmark_group("timestamp_only");

    group.bench_function("no_index", |b| {
        b.iter(|| rt.block_on(query_timestamp_only(&fixture.no_index_dataset)))
    });

    group.bench_function("btree_tenant", |b| {
        b.iter(|| rt.block_on(query_timestamp_only(&fixture.btree_dataset)))
    });

    group.bench_function("compound", |b| {
        b.iter(|| rt.block_on(query_timestamp_only(&fixture.compound_dataset)))
    });

    group.finish();
}

// =============================================================================
// Multi-Fragment Benchmark
// =============================================================================
//
// This section benchmarks compound index performance when data is spread across
// multiple fragments, simulating a time-series ingestion pattern where data
// arrives over time and is written to separate fragments.
//
// Data Layout:
// - Fragment 0: timestamps 0-999 for ALL tenants (100K rows)
// - Fragment 1: timestamps 1000-1999 for ALL tenants (100K rows)
// - ...
// - Fragment 9: timestamps 9000-9999 for ALL tenants (100K rows)
//
// This layout means each tenant's data is spread across all 10 fragments.
// The compound index can use (tenant_id, timestamp) statistics to skip entire
// fragments, while a single-column btree on tenant_id must search all fragments.

/// Generate data for a single fragment (all tenants, specific timestamp range)
fn generate_fragment_data(start_timestamp: i64, end_timestamp: i64) -> RecordBatch {
    let timestamps_in_fragment = end_timestamp - start_timestamp;

    // Generate tenant_ids: all tenants for each timestamp in range
    let tenant_ids: Vec<String> = (start_timestamp..end_timestamp)
        .flat_map(|_| (0..NUM_TENANTS).map(|t| format!("tenant_{:02}", t)))
        .collect();

    // Generate timestamps: each timestamp repeated for all tenants
    let timestamps: Vec<i64> = (start_timestamp..end_timestamp)
        .flat_map(|ts| (0..NUM_TENANTS).map(move |_| ts))
        .collect();

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("tenant_id", DataType::Utf8, false),
        ArrowField::new("timestamp", DataType::Int64, false),
    ]));

    assert_eq!(
        tenant_ids.len(),
        (timestamps_in_fragment * NUM_TENANTS) as usize
    );

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(tenant_ids)),
            Arc::new(Int64Array::from(timestamps)),
        ],
    )
    .expect("Failed to create record batch")
}

struct MultiFragmentFixture {
    _no_index_dir: TempStrDir,
    _btree_dir: TempStrDir,
    _compound_dir: TempStrDir,
    no_index_dataset: Arc<Dataset>,
    btree_dataset: Arc<Dataset>,
    compound_dataset: Arc<Dataset>,
}

impl MultiFragmentFixture {
    async fn open() -> Self {
        use lance::dataset::WriteParams;

        // Create three separate directories for each dataset variant
        let no_index_dir = TempStrDir::default();
        let btree_dir = TempStrDir::default();
        let compound_dir = TempStrDir::default();

        let no_index_uri = format!("file://{}", no_index_dir.as_str());
        let btree_uri = format!("file://{}", btree_dir.as_str());
        let compound_uri = format!("file://{}", compound_dir.as_str());

        // Write first fragment to create datasets
        let first_batch = generate_fragment_data(0, TIMESTAMPS_PER_FRAGMENT);
        let schema = first_batch.schema();

        let mut no_index_dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(first_batch.clone())], schema.clone()),
            &no_index_uri,
            None,
        )
        .await
        .expect("Failed to write no_index dataset");

        let mut btree_dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(first_batch.clone())], schema.clone()),
            &btree_uri,
            None,
        )
        .await
        .expect("Failed to write btree dataset");

        let mut compound_dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(first_batch.clone())], schema.clone()),
            &compound_uri,
            None,
        )
        .await
        .expect("Failed to write compound dataset");

        // Append remaining fragments (fragments 1-9)
        for fragment_idx in 1..NUM_FRAGMENTS {
            let start_ts = fragment_idx * TIMESTAMPS_PER_FRAGMENT;
            let end_ts = start_ts + TIMESTAMPS_PER_FRAGMENT;
            let batch = generate_fragment_data(start_ts, end_ts);

            let write_params = WriteParams {
                mode: lance::dataset::WriteMode::Append,
                ..Default::default()
            };

            no_index_dataset = Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone()),
                &no_index_uri,
                Some(write_params.clone()),
            )
            .await
            .expect("Failed to append to no_index dataset");

            btree_dataset = Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone()),
                &btree_uri,
                Some(write_params.clone()),
            )
            .await
            .expect("Failed to append to btree dataset");

            compound_dataset = Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone()),
                &compound_uri,
                Some(write_params.clone()),
            )
            .await
            .expect("Failed to append to compound dataset");
        }

        // Verify we have the expected number of fragments
        assert_eq!(
            no_index_dataset.count_fragments(),
            NUM_FRAGMENTS as usize,
            "no_index should have {} fragments",
            NUM_FRAGMENTS
        );
        assert_eq!(
            btree_dataset.count_fragments(),
            NUM_FRAGMENTS as usize,
            "btree should have {} fragments",
            NUM_FRAGMENTS
        );
        assert_eq!(
            compound_dataset.count_fragments(),
            NUM_FRAGMENTS as usize,
            "compound should have {} fragments",
            NUM_FRAGMENTS
        );

        // Create BTree index on tenant_id only
        let params = ScalarIndexParams::default();
        btree_dataset
            .create_index(
                &["tenant_id"],
                IndexType::BTree,
                Some("btree_tenant_idx".to_string()),
                &params,
                false,
            )
            .await
            .expect("Failed to create btree index");

        // Create Compound index on (tenant_id, timestamp)
        compound_dataset
            .create_index(
                &["tenant_id", "timestamp"],
                IndexType::BTree,
                Some("compound_idx".to_string()),
                &params,
                false,
            )
            .await
            .expect("Failed to create compound index");

        // Re-open datasets to ensure indices are loaded
        let btree_dataset = Arc::new(
            Dataset::open(&btree_uri)
                .await
                .expect("Failed to reopen btree dataset"),
        );
        let compound_dataset = Arc::new(
            Dataset::open(&compound_uri)
                .await
                .expect("Failed to reopen compound dataset"),
        );

        Self {
            _no_index_dir: no_index_dir,
            _btree_dir: btree_dir,
            _compound_dir: compound_dir,
            no_index_dataset: Arc::new(no_index_dataset),
            btree_dataset,
            compound_dataset,
        }
    }
}

// Multi-fragment benchmark functions

fn bench_multi_fragment_tenant_only(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let fixture = rt.block_on(MultiFragmentFixture::open());

    // Verify correctness: 10,000 rows for tenant_50 (spread across all fragments)
    let expected_rows = TIMESTAMPS_PER_TENANT as usize;

    let no_index_rows = rt.block_on(query_tenant_only(&fixture.no_index_dataset));
    assert_eq!(
        no_index_rows, expected_rows,
        "multi_fragment no_index returned wrong row count"
    );

    let btree_rows = rt.block_on(query_tenant_only(&fixture.btree_dataset));
    assert_eq!(
        btree_rows, expected_rows,
        "multi_fragment btree returned wrong row count"
    );

    let compound_rows = rt.block_on(query_tenant_only(&fixture.compound_dataset));
    assert_eq!(
        compound_rows, expected_rows,
        "multi_fragment compound returned wrong row count"
    );

    let mut group = c.benchmark_group("multi_fragment_tenant_only");

    group.bench_function("no_index", |b| {
        b.iter(|| rt.block_on(query_tenant_only(&fixture.no_index_dataset)))
    });

    group.bench_function("btree_tenant", |b| {
        b.iter(|| rt.block_on(query_tenant_only(&fixture.btree_dataset)))
    });

    group.bench_function("compound", |b| {
        b.iter(|| rt.block_on(query_tenant_only(&fixture.compound_dataset)))
    });

    group.finish();
}

fn bench_multi_fragment_narrow_range(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let fixture = rt.block_on(MultiFragmentFixture::open());

    // Verify correctness: timestamps 9901-9999 = 99 rows
    // These are ALL in the last fragment (fragment 9: timestamps 9000-9999)
    let expected_rows = 99;

    let no_index_rows = rt.block_on(query_tenant_narrow_range(&fixture.no_index_dataset));
    assert_eq!(
        no_index_rows, expected_rows,
        "multi_fragment no_index returned wrong row count"
    );

    let btree_rows = rt.block_on(query_tenant_narrow_range(&fixture.btree_dataset));
    assert_eq!(
        btree_rows, expected_rows,
        "multi_fragment btree returned wrong row count"
    );

    let compound_rows = rt.block_on(query_tenant_narrow_range(&fixture.compound_dataset));
    assert_eq!(
        compound_rows, expected_rows,
        "multi_fragment compound returned wrong row count"
    );

    let mut group = c.benchmark_group("multi_fragment_narrow_range");

    group.bench_function("no_index", |b| {
        b.iter(|| rt.block_on(query_tenant_narrow_range(&fixture.no_index_dataset)))
    });

    group.bench_function("btree_tenant", |b| {
        b.iter(|| rt.block_on(query_tenant_narrow_range(&fixture.btree_dataset)))
    });

    group.bench_function("compound", |b| {
        b.iter(|| rt.block_on(query_tenant_narrow_range(&fixture.compound_dataset)))
    });

    group.finish();
}

fn bench_multi_fragment_medium_range(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let fixture = rt.block_on(MultiFragmentFixture::open());

    // Query: tenant_50 AND timestamp > 5000
    // Timestamps 5001-9999 = 4999 rows, spread across fragments 5-9
    // Compound index should skip fragments 0-4
    let expected_rows = 4999;

    let no_index_rows = rt.block_on(query_tenant_wide_range(&fixture.no_index_dataset));
    assert_eq!(
        no_index_rows, expected_rows,
        "multi_fragment no_index returned wrong row count"
    );

    let btree_rows = rt.block_on(query_tenant_wide_range(&fixture.btree_dataset));
    assert_eq!(
        btree_rows, expected_rows,
        "multi_fragment btree returned wrong row count"
    );

    let compound_rows = rt.block_on(query_tenant_wide_range(&fixture.compound_dataset));
    assert_eq!(
        compound_rows, expected_rows,
        "multi_fragment compound returned wrong row count"
    );

    let mut group = c.benchmark_group("multi_fragment_medium_range");

    group.bench_function("no_index", |b| {
        b.iter(|| rt.block_on(query_tenant_wide_range(&fixture.no_index_dataset)))
    });

    group.bench_function("btree_tenant", |b| {
        b.iter(|| rt.block_on(query_tenant_wide_range(&fixture.btree_dataset)))
    });

    group.bench_function("compound", |b| {
        b.iter(|| rt.block_on(query_tenant_wide_range(&fixture.compound_dataset)))
    });

    group.finish();
}

fn bench_multi_fragment_wide_range(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let fixture = rt.block_on(MultiFragmentFixture::open());

    // Query: tenant_50 AND timestamp > 9000
    // Timestamps 9001-9999 = 999 rows, all in fragment 9
    // Compound index should skip fragments 0-8
    let expected_rows = 999;

    let no_index_rows = rt.block_on(run_query(
        &fixture.no_index_dataset,
        "tenant_id = 'tenant_50' AND timestamp > 9000",
    ));
    assert_eq!(
        no_index_rows, expected_rows,
        "multi_fragment no_index returned wrong row count"
    );

    let btree_rows = rt.block_on(run_query(
        &fixture.btree_dataset,
        "tenant_id = 'tenant_50' AND timestamp > 9000",
    ));
    assert_eq!(
        btree_rows, expected_rows,
        "multi_fragment btree returned wrong row count"
    );

    let compound_rows = rt.block_on(run_query(
        &fixture.compound_dataset,
        "tenant_id = 'tenant_50' AND timestamp > 9000",
    ));
    assert_eq!(
        compound_rows, expected_rows,
        "multi_fragment compound returned wrong row count"
    );

    let mut group = c.benchmark_group("multi_fragment_wide_range");

    group.bench_function("no_index", |b| {
        b.iter(|| {
            rt.block_on(run_query(
                &fixture.no_index_dataset,
                "tenant_id = 'tenant_50' AND timestamp > 9000",
            ))
        })
    });

    group.bench_function("btree_tenant", |b| {
        b.iter(|| {
            rt.block_on(run_query(
                &fixture.btree_dataset,
                "tenant_id = 'tenant_50' AND timestamp > 9000",
            ))
        })
    });

    group.bench_function("compound", |b| {
        b.iter(|| {
            rt.block_on(run_query(
                &fixture.compound_dataset,
                "tenant_id = 'tenant_50' AND timestamp > 9000",
            ))
        })
    });

    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_tenant_only,
              bench_tenant_narrow_range,
              bench_tenant_wide_range,
              bench_tenant_full_range,
              bench_timestamp_only,
              bench_multi_fragment_tenant_only,
              bench_multi_fragment_narrow_range,
              bench_multi_fragment_medium_range,
              bench_multi_fragment_wide_range
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_tenant_only,
              bench_tenant_narrow_range,
              bench_tenant_wide_range,
              bench_tenant_full_range,
              bench_timestamp_only,
              bench_multi_fragment_tenant_only,
              bench_multi_fragment_narrow_range,
              bench_multi_fragment_medium_range,
              bench_multi_fragment_wide_range
);

criterion_main!(benches);
