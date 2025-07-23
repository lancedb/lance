// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod write;

use datafusion::physical_plan::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder};
pub use write::FullSchemaMergeInsertExec;

use super::MergeStats;

pub(super) struct MergeInsertMetrics {
    pub num_inserted_rows: Count,
    pub num_updated_rows: Count,
    pub num_deleted_rows: Count,
    pub bytes_written: Count,
    pub num_files_written: Count,
}

impl From<&MergeInsertMetrics> for MergeStats {
    fn from(value: &MergeInsertMetrics) -> Self {
        Self {
            num_deleted_rows: value.num_deleted_rows.value() as u64,
            num_inserted_rows: value.num_inserted_rows.value() as u64,
            num_updated_rows: value.num_updated_rows.value() as u64,
            bytes_written: value.bytes_written.value() as u64,
            num_files_written: value.num_files_written.value() as u64,
            num_attempts: 1,
        }
    }
}

impl MergeInsertMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        let num_inserted_rows = MetricBuilder::new(metrics).counter("num_inserted_rows", partition);
        let num_updated_rows = MetricBuilder::new(metrics).counter("num_updated_rows", partition);
        let num_deleted_rows = MetricBuilder::new(metrics).counter("num_deleted_rows", partition);
        let bytes_written = MetricBuilder::new(metrics).counter("bytes_written", partition);
        let num_files_written = MetricBuilder::new(metrics).counter("num_files_written", partition);
        Self {
            num_inserted_rows,
            num_updated_rows,
            num_deleted_rows,
            bytes_written,
            num_files_written,
        }
    }
}
