// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Anti-join limit pushdown optimization
//!
//! This optimizer pushes down scan limits for anti-join patterns to reduce
//! the amount of data read from the probe side table.
//!
//! Pattern: Limit -> HashJoin(Anti) -> LanceScan (probe side)
//!
//! For anti-joins with limits, we only need to scan at most:
//! limit + build_side_rows from the probe side (worst case).

use std::sync::Arc;

use datafusion::common::Result as DFResult;
use datafusion::config::ConfigOptions;
use datafusion::logical_expr::JoinType;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion::physical_plan::ExecutionPlan;

use crate::io::exec::scan::LanceScanExec;

/// Optimizer that pushes down limits to LanceScan for anti-join patterns
///
/// This reduces I/O and memory usage by limiting how much data we read
/// from the probe side when we know we only need a limited number of results.
#[derive(Debug, Clone, Default)]
pub struct AntiJoinLimitPushdown {}

impl AntiJoinLimitPushdown {
    pub fn new() -> Self {
        Self {}
    }

    /// Push down limit to scan by using with_fetch or adding LocalLimitExec
    fn push_limit_to_scan(
        &self,
        probe_side: Arc<dyn ExecutionPlan>,
        limit: usize,
        build_rows: usize,
    ) -> Arc<dyn ExecutionPlan> {
        // Calculate maximum rows we need to scan
        // Worst case: first build_rows all match (excluded), then next limit rows don't match
        let max_probe_rows = limit + build_rows;

        // Try to push fetch directly to LanceScan
        if let Some(lance_scan) = probe_side.as_any().downcast_ref::<LanceScanExec>() {
            // Clone and create new scan with fetch limit
            let limited = LanceScanExec::new(
                lance_scan.dataset().clone(),
                lance_scan.fragments().clone(),
                lance_scan.range().clone(),
                lance_scan.projection().clone(),
                lance_scan.config().clone(),
            )
            .with_fetch(max_probe_rows);
            return Arc::new(limited);
        }

        // Fallback: add LocalLimitExec if we can't push to scan
        log::debug!(
            "AntiJoinLimitPushdown: Adding LocalLimitExec with {} rows (limit {} + build {})",
            max_probe_rows,
            limit,
            build_rows
        );
        Arc::new(LocalLimitExec::new(probe_side, max_probe_rows))
    }

    /// Optimize anti-join by pushing limit to probe side
    fn optimize_anti_join(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        limit: usize,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        use datafusion::common::stats::Precision;

        let join = plan.as_any().downcast_ref::<HashJoinExec>().unwrap();

        // Determine probe and build sides
        let (build_side, probe_side, probe_is_left) = match join.join_type() {
            JoinType::LeftAnti => (join.right(), join.left(), true),
            JoinType::RightAnti => (join.left(), join.right(), false),
            _ => return Ok(plan),
        };

        // Get build size - only optimize if we have exact statistics
        let build_rows = match build_side
            .partition_statistics(None)
            .ok()
            .and_then(|stats| match stats.num_rows {
                Precision::Exact(n) => Some(n),
                _ => None,
            }) {
            Some(rows) => rows,
            None => {
                log::debug!("AntiJoinLimitPushdown: Skip - no exact statistics for build side");
                return Ok(plan);
            }
        };

        // Push limit to probe side
        let limited_probe = self.push_limit_to_scan(probe_side.clone(), limit, build_rows);

        // Reconstruct join with limited probe, inheriting all original settings
        let new_join = if probe_is_left {
            Arc::new(HashJoinExec::try_new(
                limited_probe,
                join.right.clone(),
                join.on.clone(),
                join.filter.clone(),
                &join.join_type,
                join.projection.clone(),
                *join.partition_mode(),
                join.null_equals_null,
            )?)
        } else {
            Arc::new(HashJoinExec::try_new(
                join.left.clone(),
                limited_probe,
                join.on.clone(),
                join.filter.clone(),
                &join.join_type,
                join.projection.clone(),
                *join.partition_mode(),
                join.null_equals_null,
            )?)
        };

        Ok(new_join)
    }
}

impl PhysicalOptimizerRule for AntiJoinLimitPushdown {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Recursively optimize with limit context
        optimize_with_limit(plan, None, self)
    }

    fn name(&self) -> &str {
        "anti_join_limit_pushdown"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Recursively optimize plan, passing limit down to anti-joins
fn optimize_with_limit(
    plan: Arc<dyn ExecutionPlan>,
    parent_limit: Option<usize>,
    optimizer: &AntiJoinLimitPushdown,
) -> DFResult<Arc<dyn ExecutionPlan>> {
    // Check if this node has a fetch limit
    let current_limit = if let Some(global_limit) = plan.as_any().downcast_ref::<GlobalLimitExec>()
    {
        global_limit.fetch().map(|f| global_limit.skip() + f)
    } else {
        plan.fetch()
    }
    .or(parent_limit);

    // If this is an anti-join and we have a limit, optimize it
    if let Some(hash_join) = plan.as_any().downcast_ref::<HashJoinExec>() {
        if matches!(
            hash_join.join_type(),
            JoinType::LeftAnti | JoinType::RightAnti
        ) {
            if let Some(limit) = current_limit {
                return optimizer.optimize_anti_join(plan, limit);
            }
        }
    }

    // Recursively process children with the current limit
    let children = plan.children();
    if children.is_empty() {
        return Ok(plan);
    }

    let new_children: DFResult<Vec<_>> = children
        .into_iter()
        .map(|child| optimize_with_limit(Arc::clone(child), current_limit, optimizer))
        .collect();

    plan.with_new_children(new_children?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::RecordBatch;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::physical_plan::joins::PartitionMode;
    use datafusion_physical_plan::memory::MemoryExec;
    use datafusion_physical_expr::expressions::Column;
    use std::sync::Arc;

    /// Helper to create test data
    fn create_test_data() -> (Arc<dyn ExecutionPlan>, Arc<dyn ExecutionPlan>) {
        // Create left side data (probe side for LeftAnti)
        let left_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("data", DataType::Utf8, false),
        ]));

        let left_batch = RecordBatch::try_new(
            left_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])),
                Arc::new(StringArray::from(vec![
                    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
                ])),
            ],
        )
        .unwrap();

        let left = Arc::new(MemoryExec::try_new(&[vec![left_batch]], left_schema, None).unwrap())
            as Arc<dyn ExecutionPlan>;

        // Create right side data (build side for LeftAnti)
        let right_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

        let right_batch = RecordBatch::try_new(
            right_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![2, 4, 6]))],
        )
        .unwrap();

        let right = Arc::new(MemoryExec::try_new(&[vec![right_batch]], right_schema, None).unwrap())
            as Arc<dyn ExecutionPlan>;

        (left, right)
    }

    #[test]
    fn test_push_limit_into_left_anti_join() -> DFResult<()> {
        let (left, right) = create_test_data();

        // Create LeftAnti join
        let on = vec![(
            Arc::new(Column::new("id", 0)) as _,
            Arc::new(Column::new("id", 0)) as _,
        )];

        let anti_join = Arc::new(HashJoinExec::try_new(
            left,
            right,
            on,
            None,
            &JoinType::LeftAnti,
            None,
            PartitionMode::Partitioned,
            true,
        )?);

        // Add limit on top
        let limit = Arc::new(GlobalLimitExec::new(anti_join, 0, Some(3)));

        // Apply optimization
        let optimizer = AntiJoinLimitPushdown::new();
        let optimized = optimizer.optimize(limit, &ConfigOptions::default())?;

        // Verify structure
        assert!(optimized.as_any().is::<GlobalLimitExec>());

        let limit_node = optimized
            .as_any()
            .downcast_ref::<GlobalLimitExec>()
            .unwrap();
        assert_eq!(limit_node.fetch(), Some(3));

        let join_node = limit_node
            .input()
            .as_any()
            .downcast_ref::<HashJoinExec>()
            .unwrap();

        // Check that LocalLimit was added to the probe side (left for LeftAnti)
        assert!(join_node.left().as_any().is::<LocalLimitExec>());

        Ok(())
    }

    #[test]
    fn test_with_skip_and_fetch() -> DFResult<()> {
        let (left, right) = create_test_data();

        // Create LeftAnti join
        let on = vec![(
            Arc::new(Column::new("id", 0)) as _,
            Arc::new(Column::new("id", 0)) as _,
        )];

        let anti_join = Arc::new(HashJoinExec::try_new(
            left,
            right,
            on,
            None,
            &JoinType::LeftAnti,
            None,
            PartitionMode::Partitioned,
            true,
        )?);

        // Add limit with skip
        let limit = Arc::new(GlobalLimitExec::new(anti_join, 10, Some(90)));

        // Apply optimization
        let optimizer = AntiJoinLimitPushdown::new();
        let optimized = optimizer.optimize(limit, &ConfigOptions::default())?;

        // Verify the optimization happened
        let join_node = optimized
            .as_any()
            .downcast_ref::<GlobalLimitExec>()
            .unwrap()
            .input()
            .as_any()
            .downcast_ref::<HashJoinExec>()
            .unwrap();

        let local_limit = join_node
            .left()
            .as_any()
            .downcast_ref::<LocalLimitExec>()
            .unwrap();

        // Limit should be (skip + fetch) + build_rows = (10 + 90) + 3 = 103
        assert_eq!(local_limit.fetch(), 103);

        Ok(())
    }
}
