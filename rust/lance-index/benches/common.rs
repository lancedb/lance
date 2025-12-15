// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Common utilities and data generation for scalar index benchmarks.
use std::sync::Arc;

use arrow::datatypes::{Int64Type, UInt64Type};
use arrow_array::{Int64Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion::physical_plan::SendableRecordBatchStream;
use lance_datafusion::datagen::DatafusionDatagenExt;
use lance_datagen::{array, gen_batch, BatchCount, RowCount};

/// Total number of rows in the dataset
pub const TOTAL_ROWS: u64 = 1_000_000;

/// Number of unique values for low cardinality tests
pub const LOW_CARDINALITY_COUNT: usize = 100;

/// Batch size for streaming data
pub const BATCH_SIZE: u64 = 10_000;

/// Number of batches in the dataset
pub const NUM_BATCHES: u64 = TOTAL_ROWS / BATCH_SIZE;

/// Generate a stream of int64 data with unique values (sequential)
pub fn generate_int_unique_stream() -> SendableRecordBatchStream {
    gen_batch()
        .col("value", array::step::<Int64Type>())
        .col("_rowid", array::step::<UInt64Type>())
        .into_df_stream(
            RowCount::from(BATCH_SIZE),
            BatchCount::from(NUM_BATCHES as u32),
        )
}

/// Generate sorted int64 data with low cardinality (100 unique values)
/// Each value appears 10,000 times consecutively
pub fn generate_int_low_cardinality_stream() -> SendableRecordBatchStream {
    let rows_per_value = TOTAL_ROWS / LOW_CARDINALITY_COUNT as u64;
    let mut batches = Vec::new();
    let mut current_row = 0u64;

    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Int64, false),
        Field::new("_rowid", DataType::UInt64, false),
    ]));

    for value_idx in 0..LOW_CARDINALITY_COUNT {
        let value = value_idx as i64;
        let value_end_row = current_row + rows_per_value;

        while current_row < value_end_row {
            let batch_end = (current_row + BATCH_SIZE).min(value_end_row);
            let batch_size = (batch_end - current_row) as usize;

            // Manually create arrays with proper row IDs
            let values = vec![value; batch_size];
            let row_ids: Vec<u64> = (current_row..batch_end).collect();

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(values)),
                    Arc::new(UInt64Array::from(row_ids)),
                ],
            )
            .unwrap();

            batches.push(Ok(batch));
            current_row = batch_end;
        }
    }

    let stream = futures::stream::iter(batches);
    Box::pin(datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream))
}

/// Generate a stream of string data with unique values
/// Strings are zero-padded to 10 digits for proper lexicographic sorting
pub fn generate_string_unique_stream() -> SendableRecordBatchStream {
    let mut batches = Vec::new();
    let mut current_row = 0u64;

    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Utf8, false),
        Field::new("_rowid", DataType::UInt64, false),
    ]));

    while current_row < TOTAL_ROWS {
        let batch_end = (current_row + BATCH_SIZE).min(TOTAL_ROWS);

        // Generate zero-padded strings for proper lexicographic sorting
        let values: Vec<String> = (current_row..batch_end)
            .map(|i| format!("string_{:010}", i))
            .collect();
        let row_ids: Vec<u64> = (current_row..batch_end).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(values)),
                Arc::new(UInt64Array::from(row_ids)),
            ],
        )
        .unwrap();

        batches.push(Ok(batch));
        current_row = batch_end;
    }

    let stream = futures::stream::iter(batches);
    Box::pin(datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream))
}

/// Generate sorted string data with low cardinality (100 unique values)
pub fn generate_string_low_cardinality_stream() -> SendableRecordBatchStream {
    let rows_per_value = TOTAL_ROWS / LOW_CARDINALITY_COUNT as u64;
    let mut batches = Vec::new();
    let mut current_row = 0u64;

    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Utf8, false),
        Field::new("_rowid", DataType::UInt64, false),
    ]));

    for value_idx in 0..LOW_CARDINALITY_COUNT {
        let value = format!("value_{:03}", value_idx);
        let value_end_row = current_row + rows_per_value;

        while current_row < value_end_row {
            let batch_end = (current_row + BATCH_SIZE).min(value_end_row);
            let batch_size = (batch_end - current_row) as usize;

            // Manually create arrays with proper row IDs
            let values = vec![value.as_str(); batch_size];
            let row_ids: Vec<u64> = (current_row..batch_end).collect();

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(values)),
                    Arc::new(UInt64Array::from(row_ids)),
                ],
            )
            .unwrap();

            batches.push(Ok(batch));
            current_row = batch_end;
        }
    }

    let stream = futures::stream::iter(batches);
    Box::pin(datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream))
}

// ============================================================================
// Compound Index Data Generators
// ============================================================================

/// Number of unique tenant IDs for compound index benchmarks
pub const COMPOUND_TENANT_COUNT: usize = 100;

/// Number of unique status values for compound index benchmarks
pub const COMPOUND_STATUS_COUNT: usize = 10;

/// Timestamps per tenant for 2-column compound index (100 tenants * 10000 = 1M rows)
pub const COMPOUND_TIMESTAMPS_PER_TENANT: usize = 10_000;

/// Timestamps per tenant/status combo for 3-column index (100 * 10 * 1000 = 1M rows)
pub const COMPOUND_TIMESTAMPS_PER_STATUS: usize = 1_000;

/// Generate a 2-column compound index stream with unique integer keys.
///
/// Schema: (key1: Int64, key2: Int64, _rowid: UInt64)
/// - key1: 0..1000 (1000 unique values)
/// - key2: 0..1000 per key1 (1000 unique per key1)
/// - Total: 1M rows, sorted by (key1, key2)
pub fn generate_compound_2col_int_unique_stream() -> SendableRecordBatchStream {
    let key1_count = 1000usize;
    let key2_per_key1 = 1000usize;
    let mut batches = Vec::new();
    let mut current_row = 0u64;

    let schema = Arc::new(Schema::new(vec![
        Field::new("key1", DataType::Int64, false),
        Field::new("key2", DataType::Int64, false),
        Field::new("_rowid", DataType::UInt64, false),
    ]));

    for key1 in 0..key1_count {
        for key2_batch_start in (0..key2_per_key1).step_by(BATCH_SIZE as usize) {
            let key2_batch_end = (key2_batch_start + BATCH_SIZE as usize).min(key2_per_key1);
            let batch_size = key2_batch_end - key2_batch_start;

            let key1_values = vec![key1 as i64; batch_size];
            let key2_values: Vec<i64> = (key2_batch_start..key2_batch_end)
                .map(|k| k as i64)
                .collect();
            let row_ids: Vec<u64> = (current_row..current_row + batch_size as u64).collect();

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(key1_values)),
                    Arc::new(Int64Array::from(key2_values)),
                    Arc::new(UInt64Array::from(row_ids)),
                ],
            )
            .unwrap();

            batches.push(Ok(batch));
            current_row += batch_size as u64;
        }
    }

    let stream = futures::stream::iter(batches);
    Box::pin(datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream))
}

/// Generate a 2-column compound index stream with string tenant + int timestamp.
///
/// Schema: (tenant: Utf8, timestamp: Int64, _rowid: UInt64)
/// - tenant: "tenant_000" .. "tenant_099" (100 unique)
/// - timestamp: 0..9999 per tenant (10000 per tenant)
/// - Total: 1M rows, sorted by (tenant, timestamp)
pub fn generate_compound_2col_tenant_stream() -> SendableRecordBatchStream {
    let mut batches = Vec::new();
    let mut current_row = 0u64;

    let schema = Arc::new(Schema::new(vec![
        Field::new("tenant", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("_rowid", DataType::UInt64, false),
    ]));

    for tenant_idx in 0..COMPOUND_TENANT_COUNT {
        let tenant = format!("tenant_{:03}", tenant_idx);

        for ts_batch_start in (0..COMPOUND_TIMESTAMPS_PER_TENANT).step_by(BATCH_SIZE as usize) {
            let ts_batch_end =
                (ts_batch_start + BATCH_SIZE as usize).min(COMPOUND_TIMESTAMPS_PER_TENANT);
            let batch_size = ts_batch_end - ts_batch_start;

            let tenant_values = vec![tenant.as_str(); batch_size];
            let timestamp_values: Vec<i64> =
                (ts_batch_start..ts_batch_end).map(|t| t as i64).collect();
            let row_ids: Vec<u64> = (current_row..current_row + batch_size as u64).collect();

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(tenant_values)),
                    Arc::new(Int64Array::from(timestamp_values)),
                    Arc::new(UInt64Array::from(row_ids)),
                ],
            )
            .unwrap();

            batches.push(Ok(batch));
            current_row += batch_size as u64;
        }
    }

    let stream = futures::stream::iter(batches);
    Box::pin(datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream))
}

/// Generate a 3-column compound index stream with tenant + status + timestamp.
///
/// Schema: (tenant: Utf8, status: Utf8, timestamp: Int64, _rowid: UInt64)
/// - tenant: "tenant_000" .. "tenant_099" (100 unique)
/// - status: "status_0" .. "status_9" (10 unique per tenant)
/// - timestamp: 0..999 per tenant/status combo (1000 per combo)
/// - Total: 1M rows, sorted by (tenant, status, timestamp)
pub fn generate_compound_3col_stream() -> SendableRecordBatchStream {
    let mut batches = Vec::new();
    let mut current_row = 0u64;

    let schema = Arc::new(Schema::new(vec![
        Field::new("tenant", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("timestamp", DataType::Int64, false),
        Field::new("_rowid", DataType::UInt64, false),
    ]));

    for tenant_idx in 0..COMPOUND_TENANT_COUNT {
        let tenant = format!("tenant_{:03}", tenant_idx);

        for status_idx in 0..COMPOUND_STATUS_COUNT {
            let status = format!("status_{}", status_idx);

            for ts_batch_start in (0..COMPOUND_TIMESTAMPS_PER_STATUS).step_by(BATCH_SIZE as usize) {
                let ts_batch_end =
                    (ts_batch_start + BATCH_SIZE as usize).min(COMPOUND_TIMESTAMPS_PER_STATUS);
                let batch_size = ts_batch_end - ts_batch_start;

                let tenant_values = vec![tenant.as_str(); batch_size];
                let status_values = vec![status.as_str(); batch_size];
                let timestamp_values: Vec<i64> =
                    (ts_batch_start..ts_batch_end).map(|t| t as i64).collect();
                let row_ids: Vec<u64> = (current_row..current_row + batch_size as u64).collect();

                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(StringArray::from(tenant_values)),
                        Arc::new(StringArray::from(status_values)),
                        Arc::new(Int64Array::from(timestamp_values)),
                        Arc::new(UInt64Array::from(row_ids)),
                    ],
                )
                .unwrap();

                batches.push(Ok(batch));
                current_row += batch_size as u64;
            }
        }
    }

    let stream = futures::stream::iter(batches);
    Box::pin(datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream))
}
