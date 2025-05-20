// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod write;

use std::sync::{Arc, LazyLock};

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion::physical_plan::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder};
pub use write::FullSchemaMergeInsertExec;

use super::MergeStats;

pub(crate) static MERGE_STATS_SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("num_inserted_rows", DataType::UInt64, false),
        Field::new("num_updated_rows", DataType::UInt64, false),
        Field::new("num_deleted_rows", DataType::UInt64, false),
    ]))
});

pub(super) struct MergeInsertMetrics {
    pub num_inserted_rows: Count,
    pub num_updated_rows: Count,
    pub num_deleted_rows: Count,
}

impl From<&RecordBatch> for MergeInsertMetrics {
    fn from(value: &RecordBatch) -> Self {
        todo!("Read columns out of batch")
    }
}

impl From<&MergeInsertMetrics> for RecordBatch {
    fn from(value: &MergeInsertMetrics) -> Self {
        let num_inserted_rows = UInt64Array::from(vec![value.num_inserted_rows.value() as u64]);
        let num_updated_rows = UInt64Array::from(vec![value.num_updated_rows.value() as u64]);
        let num_deleted_rows = UInt64Array::from(vec![value.num_deleted_rows.value() as u64]);
        RecordBatch::try_new(
            (*MERGE_STATS_SCHEMA).clone(),
            vec![
                Arc::new(num_inserted_rows),
                Arc::new(num_updated_rows),
                Arc::new(num_deleted_rows),
            ],
        )
        .expect("Failed to create merge insert statistics batch")
    }
}

impl From<&MergeInsertMetrics> for MergeStats {
    fn from(value: &MergeInsertMetrics) -> Self {
        Self {
            num_deleted_rows: value.num_deleted_rows.value() as u64,
            num_inserted_rows: value.num_inserted_rows.value() as u64,
            num_updated_rows: value.num_updated_rows.value() as u64,
            num_attempts: 1,
        }
    }
}

impl MergeInsertMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        let num_inserted_rows = MetricBuilder::new(metrics).counter("num_inserted_rows", partition);
        let num_updated_rows = MetricBuilder::new(metrics).counter("num_updated_rows", partition);
        let num_deleted_rows = MetricBuilder::new(metrics).counter("num_deleted_rows", partition);
        Self {
            num_inserted_rows,
            num_updated_rows,
            num_deleted_rows,
        }
    }
}
