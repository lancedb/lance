// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Common utilities and data generation for scalar index benchmarks.

use std::sync::Arc;

use arrow::datatypes::{Float64Type, UInt64Type};
use arrow_array::RecordBatchReader;
use arrow_schema::{DataType, Field, Schema};
use datafusion::physical_plan::SendableRecordBatchStream;
use lance_datafusion::datagen::DatafusionDatagenExt;
use lance_datagen::{array, gen_batch, BatchCount, RowCount};

/// Total number of rows in the dataset
pub const TOTAL_ROWS: u64 = 50_000_000;

/// Number of unique values for low cardinality tests
pub const LOW_CARDINALITY_COUNT: usize = 100;

/// Batch size for streaming data
pub const BATCH_SIZE: u64 = 10_000;

/// Number of batches in the dataset
pub const NUM_BATCHES: u64 = TOTAL_ROWS / BATCH_SIZE;

/// Selectivity level for range queries
#[derive(Clone, Copy, Debug)]
pub enum Selectivity {
    Few,  // ~0.1% of rows
    Many, // ~10% of rows
    Most, // ~90% of rows
}

impl Selectivity {
    pub fn name(&self) -> &'static str {
        match self {
            Selectivity::Few => "few",
            Selectivity::Many => "many",
            Selectivity::Most => "most",
        }
    }

    /// Get the approximate percentage of rows that should match
    pub fn percentage(&self) -> f64 {
        match self {
            Selectivity::Few => 0.001,
            Selectivity::Many => 0.10,
            Selectivity::Most => 0.90,
        }
    }
}

/// Generate a stream of float data with unique values (sequential)
pub fn generate_float_unique_stream() -> SendableRecordBatchStream {
    gen_batch()
        .col("value", array::step::<Float64Type>())
        .col("_rowid", array::step::<UInt64Type>())
        .into_df_stream(
            RowCount::from(BATCH_SIZE),
            BatchCount::from(NUM_BATCHES as u32),
        )
}

/// Generate sorted float data with low cardinality (100 unique values)
/// Each value appears 500,000 times consecutively
pub fn generate_float_low_cardinality_stream() -> SendableRecordBatchStream {
    let rows_per_value = TOTAL_ROWS / LOW_CARDINALITY_COUNT as u64;
    let mut batches = Vec::new();
    let mut current_row = 0u64;

    for value_idx in 0..LOW_CARDINALITY_COUNT {
        let value = value_idx as f64;
        let value_end_row = current_row + rows_per_value;

        while current_row < value_end_row {
            let batch_end = (current_row + BATCH_SIZE).min(value_end_row);
            let batch_size = (batch_end - current_row) as usize;

            let mut reader = gen_batch()
                .col("value", array::fill::<Float64Type>(value))
                .col("_rowid", array::step::<UInt64Type>())
                .into_reader_rows(RowCount::from(batch_size as u64), BatchCount::from(1));

            let batch = reader.next().unwrap().unwrap();
            batches.push(Ok(batch));

            current_row = batch_end;
        }
    }

    let stream = futures::stream::iter(batches);
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Float64, false),
        Field::new("_rowid", DataType::UInt64, false),
    ]));
    Box::pin(datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream))
}

/// Generate a stream of string data with unique values
pub fn generate_string_unique_stream() -> SendableRecordBatchStream {
    gen_batch()
        .col("value", array::utf8_prefix_plus_counter("string_", false))
        .col("_rowid", array::step::<UInt64Type>())
        .into_df_stream(
            RowCount::from(BATCH_SIZE),
            BatchCount::from(NUM_BATCHES as u32),
        )
}

/// Generate sorted string data with low cardinality (100 unique values)
/// Each value appears 500,000 times consecutively
pub fn generate_string_low_cardinality_stream() -> SendableRecordBatchStream {
    let rows_per_value = TOTAL_ROWS / LOW_CARDINALITY_COUNT as u64;
    let mut batches = Vec::new();
    let mut current_row = 0u64;

    for value_idx in 0..LOW_CARDINALITY_COUNT {
        let value = format!("value_{:03}", value_idx);
        let value_end_row = current_row + rows_per_value;

        while current_row < value_end_row {
            let batch_end = (current_row + BATCH_SIZE).min(value_end_row);
            let batch_size = (batch_end - current_row) as usize;

            let mut reader = gen_batch()
                .col("value", array::fill_utf8(value.clone()))
                .col("_rowid", array::step::<UInt64Type>())
                .into_reader_rows(RowCount::from(batch_size as u64), BatchCount::from(1));

            let batch = reader.next().unwrap().unwrap();
            batches.push(Ok(batch));

            current_row = batch_end;
        }
    }

    let stream = futures::stream::iter(batches);
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Utf8, false),
        Field::new("_rowid", DataType::UInt64, false),
    ]));
    Box::pin(datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream))
}
