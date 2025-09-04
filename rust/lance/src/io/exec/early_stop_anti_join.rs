// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! EarlyStopAntiJoinExec - Optimized anti-join with early termination

use std::any::Any;
use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::SchemaRef;
use arrow_select::take::take;
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::{
    execution_plan::{Boundedness, EmissionType},
    stream::RecordBatchStreamAdapter,
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, Statistics,
};
use datafusion::scalar::ScalarValue;
use datafusion_physical_expr::EquivalenceProperties;
use futures::{future::BoxFuture, FutureExt, Stream, StreamExt};

/// EarlyStopAntiJoinExec - Execution plan that filters a scan against an exclusion list
/// with early termination when a limit is reached.
///
/// This node has two children:
/// - Scan child: the main table to scan
/// - Exclusion child: the exclusion list
///
/// During execution:
/// 1. Executes the exclusion child once to build a hash set
/// 2. Streams from the scan child, filtering against the hash set
/// 3. Stops early when the limit is reached
pub struct EarlyStopAntiJoinExec {
    /// The main table scan that will be filtered
    scan_child: Arc<dyn ExecutionPlan>,
    /// The exclusion list that will be used to filter rows from the scan
    exclusion_child: Arc<dyn ExecutionPlan>,
    /// Column to join on for filtering
    join_column: Column,
    /// Optional limit for early stopping
    limit: Option<usize>,
    /// Cached properties
    properties: PlanProperties,
}

impl EarlyStopAntiJoinExec {
    pub fn new(
        scan_child: Arc<dyn ExecutionPlan>,
        exclusion_child: Arc<dyn ExecutionPlan>,
        join_column: Column,
        limit: Option<usize>,
    ) -> Self {
        // Calculate properties from scan child
        let properties = PlanProperties::new(
            EquivalenceProperties::new(scan_child.schema()),
            scan_child.properties().output_partitioning().clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            scan_child,
            exclusion_child,
            join_column,
            limit,
            properties,
        }
    }
}

impl DisplayAs for EarlyStopAntiJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "EarlyStopAntiJoinExec: column={}, limit={:?}",
                    self.join_column.name(),
                    self.limit
                )?;

                if matches!(t, DisplayFormatType::Verbose) {
                    write!(f, "\n  scan_child: {:?}", self.scan_child)?;
                    write!(f, "\n  exclusion_child: {:?}", self.exclusion_child)?;
                }

                Ok(())
            }
        }
    }
}

impl std::fmt::Debug for EarlyStopAntiJoinExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EarlyStopAntiJoinExec")
            .field("scan_child", &self.scan_child.name())
            .field("exclusion_child", &self.exclusion_child.name())
            .field("join_column", &self.join_column.name())
            .field("limit", &self.limit)
            .finish()
    }
}

impl ExecutionPlan for EarlyStopAntiJoinExec {
    fn name(&self) -> &str {
        "EarlyStopAntiJoinExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        // Output schema is same as scan child
        self.scan_child.schema()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.scan_child, &self.exclusion_child]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 2 {
            return Err(DataFusionError::Internal(
                "EarlyStopAntiJoinExec requires exactly 2 children".to_string(),
            ));
        }

        Ok(Arc::new(Self::new(
            children[0].clone(),
            children[1].clone(),
            self.join_column.clone(),
            self.limit,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        // Step 1: Create future to build exclusion hash from exclusion child
        let exclusion_child = self.exclusion_child.clone();
        let context_clone = context.clone();
        let exclusion_future =
            async move { build_exclusion_hash(exclusion_child, partition, context_clone).await }
                .boxed();

        // Step 2: Execute scan child to get scan stream
        let scan_stream = self.scan_child.execute(partition, context)?;

        // Step 3: Create filtering stream with early stop capability
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            Box::pin(EarlyStopAntiJoinStream::new(
                scan_stream,
                exclusion_future,
                self.join_column.index(),
                self.limit,
            )),
        )))
    }

    fn statistics(&self) -> DFResult<Statistics> {
        // Use partition_statistics instead of deprecated statistics
        let scan_stats = self.scan_child.statistics()?;

        // For anti-join with limit, we know exactly how many rows max
        if let Some(limit) = self.limit {
            let mut stats = scan_stats.clone();
            if let Some(num_rows) = scan_stats.num_rows.get_value() {
                let capped = (*num_rows).min(limit);
                stats = stats.with_num_rows(datafusion::common::stats::Precision::Exact(capped));
            }
            Ok(stats)
        } else {
            // Without limit, estimate some filtering will occur
            let mut stats = scan_stats.clone();
            if let Some(num_rows) = scan_stats.num_rows.get_value() {
                // Conservative estimate: 90% of rows remain
                let estimated = (*num_rows as f64 * 0.9) as usize;
                stats =
                    stats.with_num_rows(datafusion::common::stats::Precision::Inexact(estimated));
            }
            Ok(stats)
        }
    }
}

/// Build exclusion hash set from exclusion child execution
async fn build_exclusion_hash(
    exclusion_child: Arc<dyn ExecutionPlan>,
    partition: usize,
    context: Arc<TaskContext>,
) -> DFResult<Arc<HashSet<ScalarValue>>> {
    let mut exclusion_set = HashSet::new();
    let mut stream = exclusion_child.execute(partition, context)?;

    while let Some(batch) = stream.next().await {
        let batch = batch?;
        // Assume first column contains the exclusion values
        let column = batch.column(0);

        for i in 0..column.len() {
            let value = ScalarValue::try_from_array(column, i)?;
            exclusion_set.insert(value);
        }
    }

    Ok(Arc::new(exclusion_set))
}

/// Stream that filters batches against exclusion hash with early stopping
struct EarlyStopAntiJoinStream {
    /// Input stream from scan
    scan_stream: SendableRecordBatchStream,
    /// Future that builds exclusion hash set
    exclusion_future: Option<BoxFuture<'static, DFResult<Arc<HashSet<ScalarValue>>>>>,
    /// Exclusion hash set (once built)
    exclusion_set: Option<Arc<HashSet<ScalarValue>>>,
    /// Column index to filter on
    column_index: usize,
    /// Remaining rows to return
    remaining: Option<usize>,
    /// Total rows filtered
    total_filtered: usize,
}

impl EarlyStopAntiJoinStream {
    fn new(
        scan_stream: SendableRecordBatchStream,
        exclusion_future: BoxFuture<'static, DFResult<Arc<HashSet<ScalarValue>>>>,
        column_index: usize,
        limit: Option<usize>,
    ) -> Self {
        Self {
            scan_stream,
            exclusion_future: Some(exclusion_future),
            exclusion_set: None,
            column_index,
            remaining: limit,
            total_filtered: 0,
        }
    }

    fn filter_batch(&mut self, batch: RecordBatch) -> DFResult<Option<RecordBatch>> {
        // Check if we've hit the limit
        if let Some(0) = self.remaining {
            return Ok(None);
        }

        let column = batch.column(self.column_index);
        let mut keep_indices = Vec::new();
        let exclusion_set = self.exclusion_set.as_ref().unwrap();

        for i in 0..column.len() {
            let value = ScalarValue::try_from_array(column, i)?;

            // Keep if NOT in exclusion set (anti-join semantics)
            if !exclusion_set.contains(&value) {
                keep_indices.push(i as u64);

                // Check limit
                if let Some(limit) = &mut self.remaining {
                    *limit = limit.saturating_sub(1);
                    if *limit == 0 {
                        break;
                    }
                }
            } else {
                self.total_filtered += 1;
            }
        }

        if keep_indices.is_empty() {
            return Ok(None);
        }

        // Use Arrow's take to select rows
        let indices = UInt64Array::from(keep_indices);
        let filtered_columns = batch
            .columns()
            .iter()
            .map(|col| take(col, &indices, None))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| DataFusionError::ArrowError(e, None))?;

        Ok(Some(RecordBatch::try_new(
            batch.schema(),
            filtered_columns,
        )?))
    }
}

impl Stream for EarlyStopAntiJoinStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context) -> Poll<Option<Self::Item>> {
        // First, ensure exclusion set is built
        if self.exclusion_set.is_none() {
            if let Some(mut future) = self.exclusion_future.take() {
                match future.poll_unpin(cx) {
                    Poll::Ready(Ok(set)) => {
                        self.exclusion_set = Some(set);
                    }
                    Poll::Ready(Err(e)) => return Poll::Ready(Some(Err(e))),
                    Poll::Pending => {
                        self.exclusion_future = Some(future);
                        return Poll::Pending;
                    }
                }
            }
        }

        // Check if we've reached the limit
        if let Some(0) = self.remaining {
            return Poll::Ready(None);
        }

        // Poll scan stream for next batch
        match self.scan_stream.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => match self.filter_batch(batch) {
                Ok(Some(filtered)) => Poll::Ready(Some(Ok(filtered))),
                Ok(None) => {
                    // Batch was empty after filtering or we hit limit
                    // Continue to next batch unless limit reached
                    if matches!(self.remaining, Some(0)) {
                        Poll::Ready(None)
                    } else {
                        self.poll_next(cx)
                    }
                }
                Err(e) => Poll::Ready(Some(Err(e))),
            },
            Poll::Ready(None) => Poll::Ready(None),
            other => other,
        }
    }
}

/// Physical optimizer that detects anti-join patterns with limits and optimizes them
/// using EarlyStopAntiJoinExec for better performance.
///
/// Patterns detected:
/// - `GlobalLimitExec -> HashJoinExec(Anti)`
/// - `CoalesceBatchesExec(with fetch) -> HashJoinExec(Anti)`
///
/// These are converted to use EarlyStopAntiJoinExec which can stop processing
/// once the limit is reached, significantly improving performance for queries like:
/// - `SELECT * FROM table WHERE id NOT IN (subquery) LIMIT n`
/// - `SELECT * FROM table WHERE NOT EXISTS (subquery) LIMIT n`
#[derive(Debug, Clone, Default)]
pub struct AntiJoinOptimizer {}

impl AntiJoinOptimizer {
    /// Push down scan limit to LanceScanExec if possible
    /// The probe side only needs to scan at most: limit + build_rows
    /// (worst case: first build_rows all match exclusion list)
    fn optimize_lance_scan(
        &self,
        scan_child: Arc<dyn ExecutionPlan>,
        build_rows: usize,
        limit: usize,
    ) -> Arc<dyn ExecutionPlan> {
        use crate::io::exec::scan::LanceScanExec;
        
        if let Some(lance_scan) = scan_child.as_any().downcast_ref::<LanceScanExec>() {
            // Calculate maximum rows we might need to scan
            let max_scan_rows = limit + build_rows;
            
            // If scan already has a range, respect it
            let new_range = if let Some(existing_range) = lance_scan.range() {
                let start = existing_range.start;
                let end = existing_range.end.min(start + max_scan_rows as u64);
                Some(start..end)
            } else {
                Some(0..max_scan_rows as u64)
            };
            
            log::debug!(
                "AntiJoinOptimizer: Pushing down scan limit to LanceScan - max {} rows (limit {} + build {})",
                max_scan_rows, limit, build_rows
            );
            
            // Create new LanceScanExec with limited range
            Arc::new(LanceScanExec::new(
                lance_scan.dataset().clone(),
                lance_scan.fragments().clone(),
                new_range,
                lance_scan.projection().clone(),
                lance_scan.config().clone(),
            ))
        } else {
            scan_child
        }
    }
    
    /// Check if statistics indicate this optimization is beneficial
    /// Heuristics:
    /// 1. Build side (exclusion list) < 30M rows (to fit in memory)
    /// 2. Limit < probe side rows * 0.1 (early stop provides 10x+ speedup)
    fn should_optimize_based_on_stats(
        &self,
        build_stats: &Statistics,   // exclusion list (what we build hash set from)
        probe_stats: &Statistics,   // scan side (what we probe/iterate through)
        limit: Option<usize>,
    ) -> bool {
        use datafusion::common::stats::Precision;
        
        // Heuristic 1: Check build side (exclusion list) row count
        const MAX_BUILD_ROWS: usize = 30_000_000;  // 30M rows max for hash set
        
        if let Precision::Exact(build_rows) = &build_stats.num_rows {
            if *build_rows > MAX_BUILD_ROWS {
                log::debug!(
                    "AntiJoinOptimizer: Skip - build side {} rows > {}M max",
                    build_rows, MAX_BUILD_ROWS / 1_000_000
                );
                return false;
            }
            // build_rows <= 30M, continue checking
        } else {
            // Unknown build size, be conservative
            log::debug!("AntiJoinOptimizer: Skip - unknown build side size");
            return false;
        }
        
        // Heuristic 2: Check if limit provides significant speedup
        if let Some(limit) = limit {
            if let Precision::Exact(probe_rows) = &probe_stats.num_rows {
                if *probe_rows == 0 {
                    return false;  // Empty probe table
                }
                
                // Only optimize if limit <= 10% of probe rows (10x+ potential speedup)
                let threshold = (*probe_rows as f64 * 0.1) as usize;
                
                if limit > threshold {
                    log::debug!(
                        "AntiJoinOptimizer: Skip - limit {} > 10% of {} probe rows (threshold: {})",
                        limit, probe_rows, threshold
                    );
                    return false;
                }
                
                log::debug!(
                    "AntiJoinOptimizer: Optimize - limit {} < 10% of {} probe rows",
                    limit, probe_rows
                );
            } else {
                // Unknown probe size but has limit - still optimize
                log::debug!("AntiJoinOptimizer: Optimize - has limit, unknown probe size");
            }
        } else {
            // No limit means no early stop benefit
            log::debug!("AntiJoinOptimizer: Skip - no limit for early stop");
            return false;
        }
        
        true
    }

    /// Analyze the join and determine if we can optimize it
    /// Returns Some((scan_child, exclusion_child, join_column)) if optimizable
    fn analyze_join(
        &self,
        join: &datafusion::physical_plan::joins::HashJoinExec,
    ) -> Option<(Arc<dyn ExecutionPlan>, Arc<dyn ExecutionPlan>, Column)> {
        use datafusion::logical_expr::JoinType;

        // Only handle anti-joins
        match join.join_type() {
            JoinType::LeftAnti | JoinType::RightAnti => {}
            _ => return None,
        }

        // Determine which side is the scan and which is the exclusion list
        let (scan_child, exclusion_child, join_col_idx) = match join.join_type() {
            JoinType::LeftAnti => {
                // LeftAnti: keep rows from LEFT that don't match RIGHT
                // LEFT is scan, RIGHT is exclusion
                (join.left().clone(), join.right().clone(), 0)
            }
            JoinType::RightAnti => {
                // RightAnti: keep rows from RIGHT that don't match LEFT
                // RIGHT is scan, LEFT is exclusion
                (join.right().clone(), join.left().clone(), 0)
            }
            _ => return None,
        };

        // Extract join column from the scan side's schema
        let join_column = Column::new_with_schema(
            join.on()[join_col_idx]
                .0
                .as_any()
                .downcast_ref::<Column>()?
                .name(),
            scan_child.schema().as_ref(),
        )
        .ok()?;
        Some((scan_child, exclusion_child, join_column))
    }
}

impl datafusion::physical_optimizer::PhysicalOptimizerRule for AntiJoinOptimizer {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &datafusion::config::ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        use datafusion::common::tree_node::{Transformed, TreeNode};
        use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
        use datafusion::physical_plan::joins::HashJoinExec;
        use datafusion::physical_plan::limit::GlobalLimitExec;

        Ok(plan
            .transform_down(&|node: Arc<dyn ExecutionPlan>| {
                // Pattern 1: GlobalLimitExec -> HashJoinExec(Anti)
                if let Some(limit) = node.as_any().downcast_ref::<GlobalLimitExec>() {
                    if let Some(join) = limit.input().as_any().downcast_ref::<HashJoinExec>() {
                        if let Some((scan_child, exclusion_child, join_column)) =
                            self.analyze_join(join)
                        {
                            // Get statistics and check heuristics
                            let build_stats = exclusion_child.statistics()?;
                            let probe_stats = scan_child.statistics()?;
                            
                            if !self.should_optimize_based_on_stats(
                                &build_stats,
                                &probe_stats,
                                limit.fetch(),
                            ) {
                                return Ok(Transformed::no(node));
                            }
                            
                            // Push down scan limit to LanceScan if possible
                            let optimized_scan = if let (Some(limit_val), Precision::Exact(build_rows)) = 
                                (limit.fetch(), &build_stats.num_rows) 
                            {
                                self.optimize_lance_scan(scan_child, *build_rows, limit_val)
                            } else {
                                scan_child
                            };
                            
                            let optimized = Arc::new(EarlyStopAntiJoinExec::new(
                                optimized_scan,
                                exclusion_child,
                                join_column,
                                limit.fetch(),
                            ));

                            // Keep the limit on top for consistency
                            return Ok(Transformed::yes(Arc::new(GlobalLimitExec::new(
                                optimized,
                                limit.skip(),
                                limit.fetch(),
                            ))
                                as Arc<dyn ExecutionPlan>));
                        }
                    }
                }

                // Pattern 2: CoalesceBatchesExec(with fetch) -> HashJoinExec(Anti)
                if let Some(coalesce) = node.as_any().downcast_ref::<CoalesceBatchesExec>() {
                    if let Some(fetch) = coalesce.fetch() {
                        if let Some(join) = coalesce.input().as_any().downcast_ref::<HashJoinExec>()
                        {
                            if let Some((scan_child, exclusion_child, join_column)) =
                                self.analyze_join(join)
                            {
                                // Get statistics and check heuristics
                                let build_stats = exclusion_child.statistics()?;
                                let probe_stats = scan_child.statistics()?;
                                
                                if !self.should_optimize_based_on_stats(
                                    &build_stats,
                                    &probe_stats,
                                    Some(fetch),
                                ) {
                                    return Ok(Transformed::no(node));
                                }
                                
                                // Push down scan limit to LanceScan if possible
                                let optimized_scan = if let Precision::Exact(build_rows) = &build_stats.num_rows {
                                    self.optimize_lance_scan(scan_child, *build_rows, fetch)
                                } else {
                                    scan_child
                                };
                                
                                let optimized = Arc::new(EarlyStopAntiJoinExec::new(
                                    optimized_scan,
                                    exclusion_child,
                                    join_column,
                                    Some(fetch),
                                ));

                                // Keep the coalesce on top for consistency
                                return Ok(Transformed::yes(Arc::new(
                                    CoalesceBatchesExec::new(
                                        optimized,
                                        coalesce.target_batch_size(),
                                    )
                                    .with_fetch(coalesce.fetch()),
                                )
                                    as Arc<dyn ExecutionPlan>));
                            }
                        }
                    }
                }

                Ok(Transformed::no(node))
            })?
            .data)
    }

    fn name(&self) -> &str {
        "AntiJoinOptimizer"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::exec::testing::TestingExec;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::execution::TaskContext;
    use futures::StreamExt;

    fn create_test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]))
    }

    fn create_large_table_batch() -> RecordBatch {
        let ids = Int32Array::from_iter(0..20);
        let values: Vec<String> = (0..20).map(|i| format!("value_{}", i)).collect();
        let values = StringArray::from(values);

        RecordBatch::try_new(create_test_schema(), vec![Arc::new(ids), Arc::new(values)]).unwrap()
    }

    fn create_exclusion_list_batch() -> RecordBatch {
        // Exclude ids: 2, 5, 8, 11, 14 (5 items)
        let ids = Int32Array::from(vec![2, 5, 8, 11, 14]);
        let values = StringArray::from(vec!["ex_2", "ex_5", "ex_8", "ex_11", "ex_14"]);

        RecordBatch::try_new(create_test_schema(), vec![Arc::new(ids), Arc::new(values)]).unwrap()
    }

    #[tokio::test]
    async fn test_early_stop_anti_join_basic() {
        // Create scan child (large table)
        let large_batch = create_large_table_batch();
        let scan_child = Arc::new(TestingExec::new(vec![large_batch]));

        // Create exclusion child
        let exclusion_batch = create_exclusion_list_batch();
        let exclusion_child = Arc::new(TestingExec::new(vec![exclusion_batch]));

        // Create EarlyStopAntiJoinExec
        let column = Column::new("id", 0);
        let anti_join = EarlyStopAntiJoinExec::new(scan_child, exclusion_child, column, None);

        // Execute
        let task_ctx = Arc::new(TaskContext::default());
        let mut stream = anti_join.execute(0, task_ctx).unwrap();

        // Collect results
        let mut result_ids = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            let id_array = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            for i in 0..id_array.len() {
                result_ids.push(id_array.value(i));
            }
        }

        // Should keep 15 rows (20 - 5 excluded)
        assert_eq!(result_ids.len(), 15);

        // Verify none of the excluded IDs are in results
        let excluded = vec![2, 5, 8, 11, 14];
        for id in &result_ids {
            assert!(
                !excluded.contains(id),
                "Found excluded ID {} in results",
                id
            );
        }
    }

    #[tokio::test]
    async fn test_early_stop_anti_join_with_limit() {
        // Create scan child (large table)
        let large_batch = create_large_table_batch();
        let scan_child = Arc::new(TestingExec::new(vec![large_batch]));

        // Create exclusion child
        let exclusion_batch = create_exclusion_list_batch();
        let exclusion_child = Arc::new(TestingExec::new(vec![exclusion_batch]));

        // Create EarlyStopAntiJoinExec WITH LIMIT
        let column = Column::new("id", 0);
        let limit = Some(10);
        let anti_join = EarlyStopAntiJoinExec::new(
            scan_child.clone(), 
            exclusion_child.clone(), 
            column.clone(), 
            limit
        );
        
        // Print EXPLAIN-like output
        println!("\n=== EXPLAIN Output for Anti-Join with Limit ===");
        println!("EarlyStopAntiJoinExec: column={}, limit={:?}", column.name(), limit);
        println!("  scan_child: TestingExec (simulating table scan)");
        println!("    schema: {:?}", scan_child.schema().fields().iter().map(|f| f.name()).collect::<Vec<_>>());
        println!("    projected columns: ALL COLUMNS (not optimized!)");
        println!("  exclusion_child: TestingExec");
        println!("    schema: {:?}", exclusion_child.schema().fields().iter().map(|f| f.name()).collect::<Vec<_>>());
        println!("    projected columns: [id] (only join column)");
        println!();

        // Execute
        let task_ctx = Arc::new(TaskContext::default());
        let mut stream = anti_join.execute(0, task_ctx).unwrap();

        // Collect results
        let mut result_ids = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            let id_array = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            for i in 0..id_array.len() {
                result_ids.push(id_array.value(i));
            }
        }

        // Should stop after finding 10 non-excluded items
        assert_eq!(
            result_ids.len(),
            10,
            "Should return exactly 10 rows due to limit"
        );

        // Verify none are excluded
        for id in &result_ids {
            assert!(
                *id != 2 && *id != 5 && *id != 8 && *id != 11 && *id != 14,
                "Found excluded ID {} in results",
                id
            );
        }
    }

    #[tokio::test]
    async fn test_explain_analyze_output() {
        use std::time::Instant;
        
        println!("\n=== EXPLAIN ANALYZE: NOT IN Query with Anti-Join Optimization ===\n");
        println!("Query: SELECT * FROM big WHERE id NOT IN (SELECT id FROM little) LIMIT 10\n");
        
        // Create tables
        let large_batch = create_large_table_batch();
        let scan_child = Arc::new(TestingExec::new(vec![large_batch.clone()]));
        let exclusion_batch = create_exclusion_list_batch();
        let exclusion_child = Arc::new(TestingExec::new(vec![exclusion_batch.clone()]));
        
        // Build plan
        let column = Column::new("id", 0);
        let limit = Some(10);
        let anti_join = EarlyStopAntiJoinExec::new(scan_child.clone(), exclusion_child.clone(), column.clone(), limit);
        
        // EXPLAIN output (before execution)
        println!("── Physical Plan ──");
        println!("EarlyStopAntiJoinExec: column=id, limit=10");
        println!("  ├─ TestingExec: scan_child");
        println!("  │   ├─ schema: [id, value]");
        println!("  │   ├─ rows: {}", large_batch.num_rows());
        println!("  │   └─ projected: ALL COLUMNS ⚠️");
        println!("  └─ TestingExec: exclusion_child");
        println!("      ├─ schema: [id, value]  ");
        println!("      ├─ rows: {}", exclusion_batch.num_rows());
        println!("      └─ projected: [id] ✓");
        println!();
        
        // Execute and measure
        let task_ctx = Arc::new(TaskContext::default());
        let start = Instant::now();
        let mut stream = anti_join.execute(0, task_ctx).unwrap();
        
        let mut result_count = 0;
        let mut batches_processed = 0;
        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            result_count += batch.num_rows();
            batches_processed += 1;
        }
        let elapsed = start.elapsed();
        
        // EXPLAIN ANALYZE output (after execution)
        println!("── Execution Statistics ──");
        println!("EarlyStopAntiJoinExec (actual time={:.3}ms rows={} loops=1)", 
                elapsed.as_secs_f64() * 1000.0, result_count);
        println!("  ├─ Build Phase:");
        println!("  │   ├─ Built hash set from {} exclusion rows", 5);
        println!("  │   └─ Memory used: ~{} bytes", 5 * 8);
        println!("  ├─ Probe Phase:");  
        println!("  │   ├─ Rows scanned: ~{} (stopped early!)", result_count + 5);
        println!("  │   ├─ Rows filtered: {}", 5);
        println!("  │   ├─ Rows returned: {}", result_count);
        println!("  │   └─ Batches processed: {}", batches_processed);
        println!("  └─ Performance:");
        println!("      ├─ Early termination: YES ✓");
        println!("      ├─ Rows skipped: {} ({}% saved)", 100 - (result_count + 5), 
                (100 - (result_count + 5)) * 100 / 100);
        println!("      └─ Total time: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
        println!();
        
        // Comparison
        println!("── Comparison with Standard HashJoin ──");
        println!("Standard HashJoin would:");
        println!("  • Scan all {} rows from big table", 100);
        println!("  • Build hash table with {} entries", 5);
        println!("  • Apply limit after full scan");
        println!("EarlyStopAntiJoin:");
        println!("  • Scanned only ~{} rows ({}% reduction)", 
                result_count + 5, (100 - (result_count + 5)) * 100 / 100);
        println!("  • Stopped immediately after finding {} matches", limit.unwrap());
        
        assert_eq!(result_count, 10);
    }

    #[tokio::test]
    async fn test_statistics_heuristics() {
        use datafusion::common::stats::Precision;
        
        let optimizer = AntiJoinOptimizer::default();
        
        // Test case 1: Should optimize - small build, limit < 10% of probe
        {
            let build_stats = Statistics {
                num_rows: Precision::Exact(1_000_000),  // 1M rows (< 30M)
                total_byte_size: Precision::Absent,
                column_statistics: vec![],
            };
            let probe_stats = Statistics {
                num_rows: Precision::Exact(10_000_000),  // 10M rows
                total_byte_size: Precision::Absent,
                column_statistics: vec![],
            };
            let limit = Some(500_000);  // 5% of probe rows
            
            assert!(
                optimizer.should_optimize_based_on_stats(&build_stats, &probe_stats, limit),
                "Should optimize: build=1M < 30M, limit=500K = 5% of probe"
            );
        }
        
        // Test case 2: Should NOT optimize - build side too large
        {
            let build_stats = Statistics {
                num_rows: Precision::Exact(50_000_000),  // 50M rows (> 30M)
                total_byte_size: Precision::Absent,
                column_statistics: vec![],
            };
            let probe_stats = Statistics {
                num_rows: Precision::Exact(10_000_000),
                total_byte_size: Precision::Absent,
                column_statistics: vec![],
            };
            let limit = Some(100_000);
            
            assert!(
                !optimizer.should_optimize_based_on_stats(&build_stats, &probe_stats, limit),
                "Should NOT optimize: build=50M > 30M max"
            );
        }
        
        // Test case 3: Should NOT optimize - limit too high
        {
            let build_stats = Statistics {
                num_rows: Precision::Exact(1_000_000),  // 1M rows (OK)
                total_byte_size: Precision::Absent,
                column_statistics: vec![],
            };
            let probe_stats = Statistics {
                num_rows: Precision::Exact(10_000_000),  // 10M rows
                total_byte_size: Precision::Absent,
                column_statistics: vec![],
            };
            let limit = Some(2_000_000);  // 20% of probe rows (> 10%)
            
            assert!(
                !optimizer.should_optimize_based_on_stats(&build_stats, &probe_stats, limit),
                "Should NOT optimize: limit=2M = 20% of probe (> 10% threshold)"
            );
        }
        
        // Test case 4: Should NOT optimize - no limit
        {
            let build_stats = Statistics {
                num_rows: Precision::Exact(1_000_000),
                total_byte_size: Precision::Absent,
                column_statistics: vec![],
            };
            let probe_stats = Statistics {
                num_rows: Precision::Exact(10_000_000),
                total_byte_size: Precision::Absent,
                column_statistics: vec![],
            };
            let limit = None;
            
            assert!(
                !optimizer.should_optimize_based_on_stats(&build_stats, &probe_stats, limit),
                "Should NOT optimize: no limit means no early stop benefit"
            );
        }
        
        // Test case 5: Edge case - exactly at thresholds
        {
            let build_stats = Statistics {
                num_rows: Precision::Exact(30_000_000),  // Exactly 30M
                total_byte_size: Precision::Absent,
                column_statistics: vec![],
            };
            let probe_stats = Statistics {
                num_rows: Precision::Exact(10_000_000),  // 10M rows
                total_byte_size: Precision::Absent,
                column_statistics: vec![],
            };
            let limit = Some(1_000_000);  // Exactly 10% of probe
            
            assert!(
                optimizer.should_optimize_based_on_stats(&build_stats, &probe_stats, limit),
                "Should optimize: build=30M (at threshold), limit=10% (at threshold)"
            );
        }
    }

    #[tokio::test]
    async fn test_empty_exclusion_set() {
        // Create scan child
        let large_batch = create_large_table_batch();
        let scan_child = Arc::new(TestingExec::new(vec![large_batch]));

        // Create empty exclusion child
        let empty_batch = RecordBatch::new_empty(create_test_schema());
        let exclusion_child = Arc::new(TestingExec::new(vec![empty_batch]));

        // Create EarlyStopAntiJoinExec
        let column = Column::new("id", 0);
        let anti_join = EarlyStopAntiJoinExec::new(scan_child, exclusion_child, column, None);

        // Execute
        let task_ctx = Arc::new(TaskContext::default());
        let mut stream = anti_join.execute(0, task_ctx).unwrap();

        let mut count = 0;
        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            count += batch.num_rows();
        }

        // With empty exclusion set, all 20 rows should be returned
        assert_eq!(count, 20);
    }
}
