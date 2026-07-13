// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{BooleanArray, RecordBatch, cast::AsArray, types::UInt64Type};
use datafusion::common::stats::Precision;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::execution_plan::CardinalityEffect;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, Statistics,
};
use futures::StreamExt;
use lance_core::ROW_ID;
use lance_core::utils::address::LogicalRowAddress;
use lance_table::format::{LogicalIndexCoverage, LogicalRowAddressSelection};

#[derive(Debug, Clone)]
pub struct LogicalCoverageGroup {
    pub base: Vec<LogicalRowAddressSelection>,
    pub invalidated: Vec<LogicalRowAddressSelection>,
}

/// Filters a `_rowid` stream by exact v2.3 logical index coverage.
///
/// The indexed branch uses the coverage itself. The fallback branch uses its
/// complement, so the union has neither gaps nor duplicate live rows even when
/// one physical fragment packs several logical domains.
#[derive(Clone)]
pub struct LogicalCoverageFilterExec {
    input: Arc<dyn ExecutionPlan>,
    coverage_groups: Arc<Vec<LogicalCoverageGroup>>,
    include_covered: bool,
    row_id_position: usize,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl std::fmt::Debug for LogicalCoverageFilterExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogicalCoverageFilterExec")
            .field("input", &self.input)
            .field("coverage_group_count", &self.coverage_groups.len())
            .field("include_covered", &self.include_covered)
            .finish()
    }
}

impl LogicalCoverageFilterExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        coverages: impl IntoIterator<Item = LogicalIndexCoverage>,
        include_covered: bool,
    ) -> Result<Self> {
        let selections = coverages
            .into_iter()
            .flat_map(|coverage| coverage.shards)
            .map(|shard| {
                shard.selection.ok_or_else(|| {
                    DataFusionError::Internal(
                        "logical coverage detail was not resolved from its index anchor"
                            .to_string(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Self::try_new_groups(input, vec![selections], include_covered)
    }

    pub fn try_new_groups(
        input: Arc<dyn ExecutionPlan>,
        coverage_groups: Vec<Vec<LogicalRowAddressSelection>>,
        include_covered: bool,
    ) -> Result<Self> {
        Self::try_new_effective_groups(
            input,
            coverage_groups
                .into_iter()
                .map(|base| LogicalCoverageGroup {
                    base,
                    invalidated: Vec::new(),
                })
                .collect(),
            include_covered,
        )
    }

    pub fn try_new_effective_groups(
        input: Arc<dyn ExecutionPlan>,
        coverage_groups: Vec<LogicalCoverageGroup>,
        include_covered: bool,
    ) -> Result<Self> {
        let row_id_position = input
            .schema()
            .fields()
            .iter()
            .position(|field| field.name() == ROW_ID)
            .ok_or_else(|| {
                DataFusionError::Internal(
                    "LogicalCoverageFilterExec requires a _rowid column".to_string(),
                )
            })?;
        let properties = input.properties().clone();
        Ok(Self {
            input,
            coverage_groups: Arc::new(coverage_groups),
            include_covered,
            row_id_position,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    fn filter_batch(
        batch: &RecordBatch,
        coverage_groups: &[LogicalCoverageGroup],
        include_covered: bool,
        row_id_position: usize,
    ) -> Result<RecordBatch> {
        let row_ids = batch
            .column(row_id_position)
            .as_primitive_opt::<UInt64Type>()
            .ok_or_else(|| {
                DataFusionError::Internal(
                    "LogicalCoverageFilterExec _rowid column is not UInt64".to_string(),
                )
            })?;
        let mut keep = Vec::with_capacity(row_ids.len());
        for row_id in row_ids.iter() {
            let covered = if let Some(row_id) = row_id {
                let address = LogicalRowAddress::try_from(row_id)
                    .map_err(|error| DataFusionError::External(Box::new(error)))?;
                let mut covered = !coverage_groups.is_empty();
                for group in coverage_groups {
                    let mut in_group = false;
                    for selection in &group.base {
                        if selection
                            .contains(address)
                            .map_err(|error| DataFusionError::External(Box::new(error)))?
                        {
                            in_group = true;
                            break;
                        }
                    }
                    if !in_group {
                        covered = false;
                        break;
                    }
                    for invalidated in &group.invalidated {
                        if invalidated
                            .contains(address)
                            .map_err(|error| DataFusionError::External(Box::new(error)))?
                        {
                            covered = false;
                            break;
                        }
                    }
                    if !covered {
                        break;
                    }
                }
                covered
            } else {
                false
            };
            keep.push(covered == include_covered);
        }
        arrow_select::filter::filter_record_batch(batch, &BooleanArray::from(keep))
            .map_err(Into::into)
    }
}

impl DisplayAs for LogicalCoverageFilterExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "LogicalCoverageFilterExec: include_covered={}",
            self.include_covered
        )
    }
}

impl ExecutionPlan for LogicalCoverageFilterExec {
    fn name(&self) -> &str {
        "LogicalCoverageFilterExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "LogicalCoverageFilterExec requires one child, got {}",
                children.len()
            )));
        }
        let mut replacement = (*self).clone();
        replacement.input = children[0].clone();
        replacement.properties = children[0].properties().clone();
        Ok(Arc::new(replacement))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let coverage_groups = self.coverage_groups.clone();
        let include_covered = self.include_covered;
        let row_id_position = self.row_id_position;
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let stream = self.input.execute(partition, context)?.map(move |batch| {
            let _timer = baseline.elapsed_compute().timer();
            Self::filter_batch(
                &batch?,
                coverage_groups.as_ref(),
                include_covered,
                row_id_position,
            )
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.input.schema(),
            stream,
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Statistics> {
        Ok(Statistics {
            num_rows: Precision::Absent,
            total_byte_size: Precision::Absent,
            column_statistics: vec![],
        })
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::LowerEqual
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}
