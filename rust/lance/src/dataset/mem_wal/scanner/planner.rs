// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Query planner for LSM scanner.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef, SortOptions};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{limit::GlobalLimitExec, ExecutionPlan};
use datafusion::prelude::Expr;
use lance_core::Result;

use super::collector::LsmDataSourceCollector;
use super::data_source::LsmDataSource;
use super::exec::{DeduplicateExec, MemtableGenTagExec, MEMTABLE_GEN_COLUMN, ROW_ADDRESS_COLUMN};

/// Plans scan queries over LSM data.
pub struct LsmScanPlanner {
    /// Data source collector.
    collector: LsmDataSourceCollector,
    /// Primary key column names.
    pk_columns: Vec<String>,
    /// Schema of the base table.
    base_schema: SchemaRef,
}

impl LsmScanPlanner {
    /// Create a new planner.
    pub fn new(
        collector: LsmDataSourceCollector,
        pk_columns: Vec<String>,
        base_schema: SchemaRef,
    ) -> Self {
        Self {
            collector,
            pk_columns,
            base_schema,
        }
    }

    /// Create scan plan with deduplication.
    ///
    /// # Arguments
    ///
    /// * `projection` - Columns to include in output (None = all columns)
    /// * `filter` - Filter expression to apply
    /// * `limit` - Maximum rows to return
    /// * `offset` - Number of rows to skip
    /// * `with_memtable_gen` - Whether to include _memtable_gen in output
    /// * `keep_row_address` - Whether to include _rowaddr in output
    ///
    /// # Query Plan Optimization
    ///
    /// The planner uses an optimized execution strategy:
    /// 1. Each data source is scanned and locally sorted by (pk ASC, _rowaddr DESC)
    /// 2. Sources are ordered by _memtable_gen DESC (newest first) in the UnionExec
    /// 3. K pre-sorted streams are merged using SortPreservingMergeExec
    /// 4. DeduplicateExec performs streaming deduplication on the merged output
    ///
    /// Key insight: DataFusion's SortPreservingMergeExec uses stream index as a
    /// tiebreaker when sort keys are equal. By ordering inputs with highest _memtable_gen
    /// first (lowest stream index), the merge naturally prefers newer rows.
    ///
    /// This avoids needing a `_memtable_gen` column entirely - generation ordering is implicit
    /// in the stream ordering. The `_memtable_gen` column is only added (via MemtableGenTagExec)
    /// when `with_memtable_gen=true`.
    ///
    /// This is more efficient than the naive approach of Union + global Sort because:
    /// - Local sorts are smaller and can often fit in memory
    /// - SortPreservingMergeExec is O(N log K) where K is the number of sources
    /// - Memory usage is bounded by the sum of K sort buffers rather than all data
    /// - No extra column for _memtable_gen in the common case
    pub async fn plan_scan(
        &self,
        projection: Option<&[String]>,
        filter: Option<&Expr>,
        limit: Option<usize>,
        offset: Option<usize>,
        with_memtable_gen: bool,
        keep_row_address: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // 1. Collect all data sources
        let sources = self.collector.collect()?;

        if sources.is_empty() {
            // Return empty plan
            return self.empty_plan(projection, with_memtable_gen, keep_row_address);
        }

        // 2. Build scan plan for each source with local sorting
        // Order of operations: scan -> local sort -> (optional) tag with generation
        //
        // IMPORTANT: Sources are collected in generation order (base=0, then memtables 1,2,3...)
        // We reverse this to get _memtable_gen DESC order for the merge tiebreaker.
        let sources: Vec<_> = sources.into_iter().rev().collect();

        let mut sorted_plans = Vec::new();
        for source in sources {
            let scan = self.build_source_scan(&source, projection, filter).await?;

            // Sort locally by (pk ASC, _rowaddr DESC)
            let local_sort_exprs = self.build_local_sort_exprs(&scan)?;
            let lex_ordering =
                LexOrdering::new(local_sort_exprs).ok_or_else(|| lance_core::Error::Internal {
                    message: "Failed to create LexOrdering from sort expressions".to_string(),
                    location: snafu::location!(),
                })?;
            let sorted: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(lex_ordering, scan));

            // Only tag with generation if user wants _memtable_gen in output
            let plan: Arc<dyn ExecutionPlan> = if with_memtable_gen {
                Arc::new(MemtableGenTagExec::new(sorted, source.generation()))
            } else {
                sorted
            };

            sorted_plans.push(plan);
        }

        // 3. Merge pre-sorted streams
        // Merge using (pk ASC) only - NOT _rowaddr, because _rowaddr is different across tables
        // for the same pk, which would break the stream index tiebreaker.
        //
        // DataFusion's SortPreservingMergeExec uses stream index as a tiebreaker when
        // sort keys are equal (see merge.rs line 349: `ac.cmp(bc).then_with(|| a.cmp(&b))`).
        // By ordering inputs with highest _memtable_gen first (lowest stream index), the merge
        // naturally prefers newer rows when PKs are equal.
        //
        // Local sort uses (pk ASC, _rowaddr DESC) to order within each source, but the merge
        // only considers pk for comparison. This ensures:
        // 1. For the same pk, newer generation (lower stream index) comes first
        // 2. Within the same pk and generation, higher _rowaddr comes first
        let merged: Arc<dyn ExecutionPlan> = if sorted_plans.len() == 1 {
            sorted_plans.remove(0)
        } else {
            // Use SortPreservingMergeExec to merge K pre-sorted streams
            // IMPORTANT: Only merge by pk columns, not _rowaddr!
            let merge_sort_exprs = self.build_merge_sort_exprs(&sorted_plans[0])?;
            let lex_ordering =
                LexOrdering::new(merge_sort_exprs).ok_or_else(|| lance_core::Error::Internal {
                    message: "Failed to create LexOrdering from sort expressions".to_string(),
                    location: snafu::location!(),
                })?;

            // UnionExec to combine all partitions (ordered by _memtable_gen DESC)
            #[allow(deprecated)]
            let union = Arc::new(UnionExec::new(sorted_plans));

            // SortPreservingMergeExec merges pre-sorted partitions
            Arc::new(SortPreservingMergeExec::new(lex_ordering, union))
        };

        // 4. Add deduplication (input is already sorted by pk, newer rows first)
        let dedup = DeduplicateExec::new_sorted(
            merged,
            self.pk_columns.clone(),
            with_memtable_gen,
            keep_row_address,
        )?;
        let mut plan: Arc<dyn ExecutionPlan> = Arc::new(dedup);

        // 5. Add limit if specified
        if let Some(limit) = limit {
            plan = Arc::new(GlobalLimitExec::new(plan, offset.unwrap_or(0), Some(limit)));
        }

        Ok(plan)
    }

    /// Build sort expressions for local sorting within a single source.
    ///
    /// Sort order: (pk_columns ASC, _rowaddr DESC)
    /// Note: _memtable_gen is not included because it's constant within each source.
    fn build_local_sort_exprs(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Vec<PhysicalSortExpr>> {
        let schema = plan.schema();
        let mut sort_exprs = Vec::new();

        // Sort by PK columns (ASC) to group duplicates together
        for col in &self.pk_columns {
            let (idx, _) = schema.column_with_name(col).ok_or_else(|| {
                lance_core::Error::invalid_input(
                    format!("Column '{}' not found in schema", col),
                    snafu::location!(),
                )
            })?;
            sort_exprs.push(PhysicalSortExpr {
                expr: Arc::new(Column::new(col, idx)),
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            });
        }

        // Sort by _rowaddr DESC (higher address = newer within generation)
        let (addr_idx, _) = schema.column_with_name(ROW_ADDRESS_COLUMN).ok_or_else(|| {
            lance_core::Error::invalid_input(
                format!("Column '{}' not found in schema", ROW_ADDRESS_COLUMN),
                snafu::location!(),
            )
        })?;
        sort_exprs.push(PhysicalSortExpr {
            expr: Arc::new(Column::new(ROW_ADDRESS_COLUMN, addr_idx)),
            options: SortOptions {
                descending: true,
                nulls_first: false,
            },
        });

        Ok(sort_exprs)
    }

    /// Build sort expressions for merging streams.
    ///
    /// Sort order: (pk_columns ASC) only
    ///
    /// IMPORTANT: This does NOT include _rowaddr because _rowaddr values are different
    /// across different tables for the same pk. Including _rowaddr would break the
    /// stream index tiebreaker mechanism that ensures newer generations win.
    ///
    /// When pk is equal across streams, SortPreservingMergeExec uses stream index as
    /// tiebreaker (lower index wins). Since streams are ordered by generation DESC
    /// (newest first), this ensures newer rows come before older rows for the same pk.
    fn build_merge_sort_exprs(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Vec<PhysicalSortExpr>> {
        let schema = plan.schema();
        let mut sort_exprs = Vec::new();

        // Sort by PK columns (ASC) only - NOT _rowaddr!
        for col in &self.pk_columns {
            let (idx, _) = schema.column_with_name(col).ok_or_else(|| {
                lance_core::Error::invalid_input(
                    format!("Column '{}' not found in schema", col),
                    snafu::location!(),
                )
            })?;
            sort_exprs.push(PhysicalSortExpr {
                expr: Arc::new(Column::new(col, idx)),
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            });
        }

        Ok(sort_exprs)
    }

    /// Build scan plan for a single data source.
    async fn build_source_scan(
        &self,
        source: &LsmDataSource,
        projection: Option<&[String]>,
        filter: Option<&Expr>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match source {
            LsmDataSource::BaseTable { dataset } => {
                // Use Lance Scanner
                let mut scanner = dataset.scan();

                // Project columns + _rowaddr (needed for dedup)
                let cols = self.build_projection_with_rowaddr(projection);
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                scanner.with_row_address();

                // Apply filter - enables scalar index (BTree) optimization
                if let Some(expr) = filter {
                    scanner.filter_expr(expr.clone());
                }

                scanner.create_plan().await
            }
            LsmDataSource::FlushedMemTable { path, .. } => {
                // Open as Dataset and scan
                let dataset = crate::dataset::DatasetBuilder::from_uri(path)
                    .load()
                    .await?;
                let mut scanner = dataset.scan();

                let cols = self.build_projection_with_rowaddr(projection);
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                scanner.with_row_address();

                // Apply filter - enables scalar index (BTree) optimization
                if let Some(expr) = filter {
                    scanner.filter_expr(expr.clone());
                }

                scanner.create_plan().await
            }
            LsmDataSource::ActiveMemTable {
                batch_store,
                index_store,
                schema,
                ..
            } => {
                // Use MemTableScanner
                use crate::dataset::mem_wal::memtable::scanner::MemTableScanner;

                let mut scanner =
                    MemTableScanner::new(batch_store.clone(), index_store.clone(), schema.clone());

                // Project columns and add _rowaddr for dedup
                if let Some(cols) = projection {
                    scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>());
                }
                scanner.with_row_address();

                // Apply filter - enables BTree index optimization for MemTable
                if let Some(expr) = filter {
                    scanner.filter_expr(expr.clone());
                }

                scanner.create_plan().await
            }
        }
    }

    /// Build projection list ensuring all needed columns are included.
    fn build_projection_with_rowaddr(&self, projection: Option<&[String]>) -> Vec<String> {
        let mut cols: Vec<String> = if let Some(p) = projection {
            p.to_vec()
        } else {
            self.base_schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect()
        };

        // Ensure PK columns are included
        for pk in &self.pk_columns {
            if !cols.contains(pk) {
                cols.push(pk.clone());
            }
        }

        cols
    }

    /// Create an empty execution plan.
    fn empty_plan(
        &self,
        projection: Option<&[String]>,
        with_memtable_gen: bool,
        keep_row_address: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        use datafusion::physical_plan::empty::EmptyExec;

        let mut fields: Vec<Arc<Field>> = if let Some(cols) = projection {
            cols.iter()
                .filter_map(|name| {
                    self.base_schema
                        .field_with_name(name)
                        .ok()
                        .map(|f| Arc::new(f.clone()))
                })
                .collect()
        } else {
            self.base_schema.fields().iter().cloned().collect()
        };

        if with_memtable_gen {
            fields.push(Arc::new(Field::new(
                MEMTABLE_GEN_COLUMN,
                DataType::UInt64,
                false,
            )));
        }
        if keep_row_address {
            fields.push(Arc::new(Field::new(
                ROW_ADDRESS_COLUMN,
                DataType::UInt64,
                false,
            )));
        }

        let schema = Arc::new(Schema::new(fields));
        Ok(Arc::new(EmptyExec::new(schema)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::mem_wal::scanner::data_source::RegionSnapshot;

    fn create_test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("value", DataType::Float64, true),
        ]))
    }

    #[test]
    fn test_build_projection_with_rowaddr() {
        let schema = create_test_schema();

        // Create a mock collector (we can't easily create a real one without a dataset)
        // Instead, test the projection building logic directly

        // When projection is Some, should include specified cols + PK
        let pk_columns = vec!["id".to_string()];

        let mut cols: Vec<String> = vec!["name".to_string()];
        for pk in &pk_columns {
            if !cols.contains(pk) {
                cols.push(pk.clone());
            }
        }
        assert!(cols.contains(&"name".to_string()));
        assert!(cols.contains(&"id".to_string()));

        // When projection is None, should include all schema fields
        let cols_all: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
        assert_eq!(cols_all.len(), 3);
    }

    #[test]
    fn test_region_snapshot() {
        let region_id = uuid::Uuid::new_v4();
        let snapshot = RegionSnapshot::new(region_id)
            .with_current_generation(5)
            .with_flushed_generation(1, "gen_1".to_string())
            .with_flushed_generation(2, "gen_2".to_string());

        assert_eq!(snapshot.flushed_generations.len(), 2);
        assert_eq!(snapshot.current_generation, 5);
    }
}

/// Integration tests that verify LSM scanner behavior with real datasets.
///
/// These tests validate:
/// - Query plan structure for different configurations
/// - Deduplication correctness across multiple LSM levels
/// - Both with and without BTree index optimization
#[cfg(test)]
mod integration_tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use uuid::Uuid;

    use crate::dataset::mem_wal::scanner::collector::ActiveMemTableRef;
    use crate::dataset::mem_wal::scanner::data_source::RegionSnapshot;
    use crate::dataset::mem_wal::scanner::LsmScanner;
    use crate::dataset::mem_wal::write::{BatchStore, IndexStore};
    use crate::dataset::{Dataset, WriteParams};
    use crate::utils::test::assert_plan_node_equals;

    /// Create test schema with id as primary key.
    fn create_pk_schema() -> Arc<ArrowSchema> {
        let mut id_metadata = HashMap::new();
        id_metadata.insert(
            "lance-schema:unenforced-primary-key".to_string(),
            "true".to_string(),
        );
        let id_field = Field::new("id", DataType::Int32, false).with_metadata(id_metadata);

        Arc::new(ArrowSchema::new(vec![
            id_field,
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    /// Create a test batch with given ids and name prefix.
    fn create_test_batch(schema: &ArrowSchema, ids: &[i32], name_prefix: &str) -> RecordBatch {
        let names: Vec<String> = ids
            .iter()
            .map(|id| format!("{}_{}", name_prefix, id))
            .collect();
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap()
    }

    /// Create a dataset at the given URI with the provided batches.
    async fn create_dataset(uri: &str, batches: Vec<RecordBatch>) -> Dataset {
        let schema = batches[0].schema();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        Dataset::write(reader, uri, Some(WriteParams::default()))
            .await
            .unwrap()
    }

    /// Setup a multi-level LSM structure with:
    /// - Base table: ids 1-5 with "base" prefix
    /// - Flushed gen1: ids 3,4 (updates) with "gen1" prefix
    /// - Flushed gen2: ids 4,5 (updates) + id 6 (new) with "gen2" prefix
    /// - Active memtable: ids 5,6 (updates) + id 7 (new) with "active" prefix
    ///
    /// Expected deduplication results:
    /// - id=1: "base_1" (only in base)
    /// - id=2: "base_2" (only in base)
    /// - id=3: "gen1_3" (updated in gen1)
    /// - id=4: "gen2_4" (updated in gen1 then gen2, keep gen2)
    /// - id=5: "active_5" (updated in gen2 then active, keep active)
    /// - id=6: "active_6" (added in gen2 then updated in active, keep active)
    /// - id=7: "active_7" (added in active)
    async fn setup_multi_level_lsm() -> (
        Arc<Dataset>,
        Vec<RegionSnapshot>,
        Option<(Uuid, ActiveMemTableRef)>,
        Vec<String>,
        String, // temp_dir path for cleanup
    ) {
        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        // Create base table
        let base_uri = format!("{}/base", base_path);
        let base_batch = create_test_batch(&schema, &[1, 2, 3, 4, 5], "base");
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        // Create flushed gen1 as a separate dataset
        let region_id = Uuid::new_v4();
        let gen1_uri = format!("{}/_mem_wal/{}/gen_1", base_uri, region_id);
        let gen1_batch = create_test_batch(&schema, &[3, 4], "gen1");
        create_dataset(&gen1_uri, vec![gen1_batch]).await;

        // Create flushed gen2 as a separate dataset
        let gen2_uri = format!("{}/_mem_wal/{}/gen_2", base_uri, region_id);
        let gen2_batch = create_test_batch(&schema, &[4, 5, 6], "gen2");
        create_dataset(&gen2_uri, vec![gen2_batch]).await;

        // Build region snapshot
        let region_snapshot = RegionSnapshot::new(region_id)
            .with_current_generation(3)
            .with_flushed_generation(1, "gen_1".to_string())
            .with_flushed_generation(2, "gen_2".to_string());

        // Create active memtable
        let batch_store = Arc::new(BatchStore::with_capacity(100));
        let index_store = Arc::new(IndexStore::new());
        let active_batch = create_test_batch(&schema, &[5, 6, 7], "active");
        let _ = batch_store.append(active_batch);

        let active_memtable = ActiveMemTableRef {
            batch_store,
            index_store,
            schema: schema.clone(),
            generation: 3,
        };

        let pk_columns = vec!["id".to_string()];

        // Keep temp_dir alive by storing path
        let temp_path = temp_dir.keep().to_string_lossy().to_string();

        (
            base_dataset,
            vec![region_snapshot],
            Some((region_id, active_memtable)),
            pk_columns,
            temp_path,
        )
    }

    #[tokio::test]
    async fn test_lsm_scan_query_plan_without_memtable_gen() {
        let (base_dataset, region_snapshots, active_memtable, pk_columns, _temp_path) =
            setup_multi_level_lsm().await;

        // Create scanner without requesting _memtable_gen
        let mut scanner = LsmScanner::new(base_dataset, region_snapshots, pk_columns);
        if let Some((region_id, memtable)) = active_memtable {
            scanner = scanner.with_active_memtable(region_id, memtable);
        }

        let plan = scanner.create_plan().await.unwrap();

        // Verify plan structure showing all levels (gen DESC order: active -> gen2 -> gen1 -> base):
        // - DeduplicateExec at top (with_memtable_gen=false means no MemtableGenTagExec)
        // - SortPreservingMergeExec merging by pk only (enables stream index tiebreaker)
        // - UnionExec combining 4 sorted streams
        // - Each stream: SortExec -> MemTableScanExec or LanceRead
        assert_plan_node_equals(
            plan,
            "DeduplicateExec: pk=[id], with_memtable_gen=false, keep_addr=false, input_sorted=true
  SortPreservingMergeExec: [id@0 ASC NULLS LAST]
    UnionExec
      SortExec: expr=[id@0 ASC NULLS LAST, _rowaddr@2 DESC NULLS LAST]...
        MemTableScanExec: projection=[id, name, _rowaddr], with_row_id=false, with_row_address=true
      SortExec: expr=[id@0 ASC NULLS LAST, _rowaddr@2 DESC NULLS LAST]...
        LanceRead:...gen_2...
      SortExec: expr=[id@0 ASC NULLS LAST, _rowaddr@2 DESC NULLS LAST]...
        LanceRead:...gen_1...
      SortExec: expr=[id@0 ASC NULLS LAST, _rowaddr@2 DESC NULLS LAST]...
        LanceRead:...base/data...refine_filter=--",
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_lsm_scan_query_plan_with_memtable_gen() {
        let (base_dataset, region_snapshots, active_memtable, pk_columns, _temp_path) =
            setup_multi_level_lsm().await;

        // Create scanner requesting _memtable_gen
        let mut scanner =
            LsmScanner::new(base_dataset, region_snapshots, pk_columns).with_memtable_gen();
        if let Some((region_id, memtable)) = active_memtable {
            scanner = scanner.with_active_memtable(region_id, memtable);
        }

        let plan = scanner.create_plan().await.unwrap();

        // Verify plan structure with MemtableGenTagExec at each level (gen DESC order):
        // - DeduplicateExec at top (with_memtable_gen=true)
        // - SortPreservingMergeExec merging by pk only
        // - UnionExec combining 4 streams
        // - Each stream: MemtableGenTagExec -> SortExec -> data source
        //   - gen3 (active): MemtableGenTagExec: gen=gen3 -> MemTableScanExec
        //   - gen2 (flushed): MemtableGenTagExec: gen=gen2 -> LanceRead
        //   - gen1 (flushed): MemtableGenTagExec: gen=gen1 -> LanceRead
        //   - base: MemtableGenTagExec: gen=base -> LanceRead
        assert_plan_node_equals(
            plan,
            "DeduplicateExec: pk=[id], with_memtable_gen=true, keep_addr=false, input_sorted=true
  SortPreservingMergeExec: [id@0 ASC NULLS LAST]
    UnionExec
      MemtableGenTagExec: gen=gen3
        SortExec: expr=[id@0 ASC NULLS LAST, _rowaddr@2 DESC NULLS LAST]...
          MemTableScanExec: projection=[id, name, _rowaddr], with_row_id=false, with_row_address=true
      MemtableGenTagExec: gen=gen2
        SortExec: expr=[id@0 ASC NULLS LAST, _rowaddr@2 DESC NULLS LAST]...
          LanceRead:...gen_2...
      MemtableGenTagExec: gen=gen1
        SortExec: expr=[id@0 ASC NULLS LAST, _rowaddr@2 DESC NULLS LAST]...
          LanceRead:...gen_1...
      MemtableGenTagExec: gen=base
        SortExec: expr=[id@0 ASC NULLS LAST, _rowaddr@2 DESC NULLS LAST]...
          LanceRead:...base/data...refine_filter=--",
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_lsm_scan_deduplication_results() {
        let (base_dataset, region_snapshots, active_memtable, pk_columns, _temp_path) =
            setup_multi_level_lsm().await;

        // Create scanner
        let mut scanner = LsmScanner::new(base_dataset, region_snapshots, pk_columns);
        if let Some((region_id, memtable)) = active_memtable {
            scanner = scanner.with_active_memtable(region_id, memtable);
        }

        // Execute and collect results
        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        // Collect all results into a map for easy verification
        let mut results: HashMap<i32, String> = HashMap::new();
        for batch in batches {
            let ids = batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let names = batch
                .column_by_name("name")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();

            for i in 0..batch.num_rows() {
                results.insert(ids.value(i), names.value(i).to_string());
            }
        }

        // Verify deduplication kept the newest version of each row
        assert_eq!(results.len(), 7, "Should have 7 unique rows after dedup");

        // id=1: only in base
        assert_eq!(results.get(&1), Some(&"base_1".to_string()));
        // id=2: only in base
        assert_eq!(results.get(&2), Some(&"base_2".to_string()));
        // id=3: updated in gen1
        assert_eq!(results.get(&3), Some(&"gen1_3".to_string()));
        // id=4: updated in gen1, then gen2 -> keep gen2
        assert_eq!(results.get(&4), Some(&"gen2_4".to_string()));
        // id=5: updated in gen2, then active -> keep active
        assert_eq!(results.get(&5), Some(&"active_5".to_string()));
        // id=6: added in gen2, updated in active -> keep active
        assert_eq!(results.get(&6), Some(&"active_6".to_string()));
        // id=7: only in active
        assert_eq!(results.get(&7), Some(&"active_7".to_string()));
    }

    #[tokio::test]
    async fn test_lsm_scan_with_projection() {
        let (base_dataset, region_snapshots, active_memtable, pk_columns, _temp_path) =
            setup_multi_level_lsm().await;

        // Create scanner with projection (only id column)
        let mut scanner =
            LsmScanner::new(base_dataset, region_snapshots, pk_columns).project(&["id"]);
        if let Some((region_id, memtable)) = active_memtable {
            scanner = scanner.with_active_memtable(region_id, memtable);
        }

        // Execute and collect results
        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        // Verify schema only has "id" column
        let schema = batches[0].schema();
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "id");

        // Count total rows
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 7, "Should have 7 unique rows after dedup");
    }

    #[tokio::test]
    async fn test_lsm_scan_with_limit() {
        let (base_dataset, region_snapshots, active_memtable, pk_columns, _temp_path) =
            setup_multi_level_lsm().await;

        // Create scanner with limit
        let mut scanner =
            LsmScanner::new(base_dataset, region_snapshots, pk_columns).limit(3, None);
        if let Some((region_id, memtable)) = active_memtable {
            scanner = scanner.with_active_memtable(region_id, memtable);
        }

        // Execute and collect results
        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        // Count total rows
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 3, "Should have 3 rows due to limit");
    }

    #[tokio::test]
    async fn test_lsm_scan_base_only() {
        let (base_dataset, _, _, pk_columns, _temp_path) = setup_multi_level_lsm().await;

        // Create scanner with only base table (no region snapshots or active memtable)
        let scanner = LsmScanner::new(base_dataset, vec![], pk_columns);

        let plan = scanner.create_plan().await.unwrap();

        // With only one source, should skip UnionExec and SortPreservingMergeExec
        // Plan structure:
        // - DeduplicateExec at top
        // - SortExec (no merge needed)
        // - LanceRead for base table only
        assert_plan_node_equals(
            plan,
            "DeduplicateExec: pk=[id], with_memtable_gen=false, keep_addr=false, input_sorted=true
  SortExec: expr=[id@0 ASC NULLS LAST, _rowaddr@2 DESC NULLS LAST]...
    LanceRead:...base/data...refine_filter=--",
        )
        .await
        .unwrap();

        // Execute and verify all 5 base rows are returned
        let scanner = LsmScanner::new(
            Arc::new(
                Dataset::open(&format!("{}/base", _temp_path))
                    .await
                    .unwrap(),
            ),
            vec![],
            vec!["id".to_string()],
        );
        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 5, "Should have 5 rows from base table");
    }

    #[tokio::test]
    async fn test_lsm_scan_flushed_only_no_active() {
        let (base_dataset, region_snapshots, _, pk_columns, _temp_path) =
            setup_multi_level_lsm().await;

        // Create scanner with base + flushed (no active memtable)
        let scanner = LsmScanner::new(base_dataset, region_snapshots, pk_columns);

        // Execute and collect results
        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        // Collect all results into a map
        let mut results: HashMap<i32, String> = HashMap::new();
        for batch in batches {
            let ids = batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let names = batch
                .column_by_name("name")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();

            for i in 0..batch.num_rows() {
                results.insert(ids.value(i), names.value(i).to_string());
            }
        }

        // Verify results (without active memtable)
        assert_eq!(results.len(), 6, "Should have 6 unique rows (no id=7)");
        assert_eq!(results.get(&1), Some(&"base_1".to_string()));
        assert_eq!(results.get(&2), Some(&"base_2".to_string()));
        assert_eq!(results.get(&3), Some(&"gen1_3".to_string()));
        assert_eq!(results.get(&4), Some(&"gen2_4".to_string()));
        // Without active, gen2 is newest
        assert_eq!(results.get(&5), Some(&"gen2_5".to_string()));
        assert_eq!(results.get(&6), Some(&"gen2_6".to_string()));
        // id=7 doesn't exist without active memtable
        assert_eq!(results.get(&7), None);
    }

    #[tokio::test]
    async fn test_lsm_scan_with_row_address() {
        let (base_dataset, region_snapshots, active_memtable, pk_columns, _temp_path) =
            setup_multi_level_lsm().await;

        // Create scanner requesting _rowaddr
        let mut scanner =
            LsmScanner::new(base_dataset, region_snapshots, pk_columns).with_row_address();
        if let Some((region_id, memtable)) = active_memtable {
            scanner = scanner.with_active_memtable(region_id, memtable);
        }

        let plan = scanner.create_plan().await.unwrap();

        // Verify plan with keep_addr=true (no _memtable_gen, so no MemtableGenTagExec)
        assert_plan_node_equals(
            plan,
            "DeduplicateExec: pk=[id], with_memtable_gen=false, keep_addr=true, input_sorted=true
  SortPreservingMergeExec: [id@0 ASC NULLS LAST]
    UnionExec
      SortExec: expr=[id@0 ASC NULLS LAST, _rowaddr@2 DESC NULLS LAST]...
        MemTableScanExec: projection=[id, name, _rowaddr], with_row_id=false, with_row_address=true
      SortExec: expr=[id@0 ASC NULLS LAST, _rowaddr@2 DESC NULLS LAST]...
        LanceRead:...gen_2...
      SortExec: expr=[id@0 ASC NULLS LAST, _rowaddr@2 DESC NULLS LAST]...
        LanceRead:...gen_1...
      SortExec: expr=[id@0 ASC NULLS LAST, _rowaddr@2 DESC NULLS LAST]...
        LanceRead:...base/data...refine_filter=--",
        )
        .await
        .unwrap();

        // Execute and verify _rowaddr column is present
        let scanner = LsmScanner::new(
            Arc::new(
                Dataset::open(&format!("{}/base", _temp_path))
                    .await
                    .unwrap(),
            ),
            vec![],
            vec!["id".to_string()],
        )
        .with_row_address();

        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        // Verify schema includes _rowaddr
        let schema = batches[0].schema();
        assert!(
            schema.column_with_name("_rowaddr").is_some(),
            "Schema should include _rowaddr"
        );
    }

    #[tokio::test]
    async fn test_lsm_scan_with_both_memtable_gen_and_row_address() {
        let (base_dataset, region_snapshots, active_memtable, pk_columns, _temp_path) =
            setup_multi_level_lsm().await;

        // Create scanner requesting both _memtable_gen and _rowaddr
        let mut scanner = LsmScanner::new(base_dataset, region_snapshots, pk_columns)
            .with_memtable_gen()
            .with_row_address();
        if let Some((region_id, memtable)) = active_memtable {
            scanner = scanner.with_active_memtable(region_id, memtable);
        }

        let plan = scanner.create_plan().await.unwrap();

        // Verify plan with both with_memtable_gen=true and keep_addr=true
        // Full plan with all levels and MemtableGenTagExec at each
        assert_plan_node_equals(
            plan,
            "DeduplicateExec: pk=[id], with_memtable_gen=true, keep_addr=true, input_sorted=true
  SortPreservingMergeExec: [id@0 ASC NULLS LAST]
    UnionExec
      MemtableGenTagExec: gen=gen3
        SortExec: expr=[id@0 ASC NULLS LAST, _rowaddr@2 DESC NULLS LAST]...
          MemTableScanExec: projection=[id, name, _rowaddr], with_row_id=false, with_row_address=true
      MemtableGenTagExec: gen=gen2
        SortExec: expr=[id@0 ASC NULLS LAST, _rowaddr@2 DESC NULLS LAST]...
          LanceRead:...gen_2...
      MemtableGenTagExec: gen=gen1
        SortExec: expr=[id@0 ASC NULLS LAST, _rowaddr@2 DESC NULLS LAST]...
          LanceRead:...gen_1...
      MemtableGenTagExec: gen=base
        SortExec: expr=[id@0 ASC NULLS LAST, _rowaddr@2 DESC NULLS LAST]...
          LanceRead:...base/data...refine_filter=--",
        )
        .await
        .unwrap();
    }

    /// Setup LSM with BTree index on the primary key for filter optimization tests.
    ///
    /// Similar to setup_multi_level_lsm but:
    /// - Active memtable has a BTree index on the `id` column
    /// - Flushed datasets have BTree index created (enabling ScalarIndexQuery)
    async fn setup_multi_level_lsm_with_btree_index() -> (
        Arc<Dataset>,
        Vec<RegionSnapshot>,
        Option<(Uuid, ActiveMemTableRef)>,
        Vec<String>,
        String,
    ) {
        use crate::index::CreateIndexBuilder;
        use lance_index::scalar::ScalarIndexParams;
        use lance_index::IndexType;

        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        // Create base table with BTree index
        let base_uri = format!("{}/base", base_path);
        let base_batch = create_test_batch(&schema, &[1, 2, 3, 4, 5], "base");
        let mut base_dataset = create_dataset(&base_uri, vec![base_batch]).await;

        // Create BTree index on base table
        let params = ScalarIndexParams::default();
        CreateIndexBuilder::new(&mut base_dataset, &["id"], IndexType::BTree, &params)
            .await
            .unwrap();

        // Reload dataset to pick up the index
        let base_dataset = Arc::new(Dataset::open(&base_uri).await.unwrap());

        // Create flushed gen1 with BTree index
        let region_id = Uuid::new_v4();
        let gen1_uri = format!("{}/_mem_wal/{}/gen_1", base_uri, region_id);
        let gen1_batch = create_test_batch(&schema, &[3, 4], "gen1");
        let mut gen1_dataset = create_dataset(&gen1_uri, vec![gen1_batch]).await;
        CreateIndexBuilder::new(&mut gen1_dataset, &["id"], IndexType::BTree, &params)
            .await
            .unwrap();

        // Create flushed gen2 with BTree index
        let gen2_uri = format!("{}/_mem_wal/{}/gen_2", base_uri, region_id);
        let gen2_batch = create_test_batch(&schema, &[4, 5, 6], "gen2");
        let mut gen2_dataset = create_dataset(&gen2_uri, vec![gen2_batch]).await;
        CreateIndexBuilder::new(&mut gen2_dataset, &["id"], IndexType::BTree, &params)
            .await
            .unwrap();

        // Build region snapshot
        let region_snapshot = RegionSnapshot::new(region_id)
            .with_current_generation(3)
            .with_flushed_generation(1, "gen_1".to_string())
            .with_flushed_generation(2, "gen_2".to_string());

        // Create active memtable with BTree index
        let batch_store = Arc::new(BatchStore::with_capacity(100));
        let mut index_store = IndexStore::new();
        // Add BTree index on id column (field_id=0)
        index_store.add_btree("id_idx".to_string(), 0, "id".to_string());

        let active_batch = create_test_batch(&schema, &[5, 6, 7], "active");
        let _ = batch_store.append(active_batch.clone());

        // Index the batch with row offset 0 and batch position 0
        index_store
            .insert_with_batch_position(&active_batch, 0, Some(0))
            .unwrap();

        let index_store = Arc::new(index_store);

        let active_memtable = ActiveMemTableRef {
            batch_store,
            index_store,
            schema: schema.clone(),
            generation: 3,
        };

        let pk_columns = vec!["id".to_string()];
        let temp_path = temp_dir.keep().to_string_lossy().to_string();

        (
            base_dataset,
            vec![region_snapshot],
            Some((region_id, active_memtable)),
            pk_columns,
            temp_path,
        )
    }

    #[tokio::test]
    async fn test_lsm_scan_with_btree_index_filter() {
        let (base_dataset, region_snapshots, active_memtable, pk_columns, _temp_path) =
            setup_multi_level_lsm_with_btree_index().await;

        // Create scanner with filter on the indexed column
        let mut scanner = LsmScanner::new(base_dataset, region_snapshots, pk_columns)
            .filter("id = 5")
            .unwrap();
        if let Some((region_id, memtable)) = active_memtable {
            scanner = scanner.with_active_memtable(region_id, memtable);
        }

        let plan = scanner.create_plan().await.unwrap();

        // Verify plan structure with BTree index optimization.
        // Instead of complex pattern matching, verify key components directly:
        use datafusion::physical_plan::displayable;
        let plan_str = format!("{}", displayable(plan.as_ref()).indent(true));

        // 1. Verify overall structure
        assert!(
            plan_str.contains("DeduplicateExec: pk=[id]"),
            "Should have DeduplicateExec at top"
        );
        assert!(
            plan_str.contains("SortPreservingMergeExec"),
            "Should use SortPreservingMergeExec for merging"
        );
        assert!(plan_str.contains("UnionExec"), "Should have UnionExec");

        // 2. Verify BTree index optimization for active memtable
        assert!(
            plan_str.contains("BTreeIndexExec: predicate=Eq"),
            "Active memtable should use BTreeIndexExec instead of MemTableScanExec"
        );

        // 3. Verify filter pushdown to flushed and base datasets
        assert!(
            plan_str.contains("gen_2") && plan_str.contains("full_filter="),
            "gen_2 should have filter pushed down"
        );
        assert!(
            plan_str.contains("gen_1") && plan_str.contains("full_filter="),
            "gen_1 should have filter pushed down"
        );
        assert!(
            plan_str.contains("base/data") && plan_str.contains("full_filter="),
            "base table should have filter pushed down"
        );

        // Execute and verify result - should return only id=5 (from active, as it's newest)
        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        // Collect results
        let mut results: HashMap<i32, String> = HashMap::new();
        for batch in batches {
            let ids = batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let names = batch
                .column_by_name("name")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();

            for i in 0..batch.num_rows() {
                results.insert(ids.value(i), names.value(i).to_string());
            }
        }

        // Should only have id=5 with the active version (newest wins dedup)
        assert_eq!(results.len(), 1, "Filter should return only matching rows");
        assert_eq!(
            results.get(&5),
            Some(&"active_5".to_string()),
            "Should get newest version (active) for id=5"
        );
    }

    #[tokio::test]
    async fn test_lsm_scan_with_filter_no_index() {
        // Test that filter still works correctly even without BTree index
        let (base_dataset, region_snapshots, active_memtable, pk_columns, _temp_path) =
            setup_multi_level_lsm().await;

        // Create scanner with SQL filter
        // This tests that type coercion works correctly (Int64 literal -> Int32 column)
        let mut scanner = LsmScanner::new(base_dataset, region_snapshots, pk_columns)
            .filter("id = 3")
            .unwrap();
        if let Some((region_id, memtable)) = active_memtable {
            scanner = scanner.with_active_memtable(region_id, memtable);
        }

        // Execute and verify result
        let batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        let mut results: HashMap<i32, String> = HashMap::new();
        for batch in batches {
            let ids = batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let names = batch
                .column_by_name("name")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();

            for i in 0..batch.num_rows() {
                results.insert(ids.value(i), names.value(i).to_string());
            }
        }

        // id=3 should return gen1 version (base had 3, gen1 updated it)
        assert_eq!(results.len(), 1);
        assert_eq!(results.get(&3), Some(&"gen1_3".to_string()));
    }
}
