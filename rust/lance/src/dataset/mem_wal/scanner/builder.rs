// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! LSM Scanner builder.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::common::ToDFSchema;
use datafusion::physical_plan::{ExecutionPlan, SendableRecordBatchStream};
use datafusion::prelude::{Expr, SessionContext};
use futures::TryStreamExt;
use lance_core::{Error, Result};
use snafu::location;
use uuid::Uuid;

use super::collector::{ActiveMemTableRef, LsmDataSourceCollector};
use super::data_source::RegionSnapshot;
use super::planner::LsmScanPlanner;
use crate::dataset::Dataset;

/// Scanner for LSM tree data spanning base table, flushed MemTables, and active MemTable.
///
/// This scanner provides a unified interface for querying data across multiple
/// LSM tree levels:
/// - Base table (merged data, generation = 0)
/// - Flushed MemTables (persisted but not yet merged, generation = 1, 2, ...)
/// - Active MemTable (in-memory buffer, highest generation)
///
/// The scanner automatically handles deduplication by primary key, keeping
/// the newest version based on generation number and row address.
///
/// # Example
///
/// ```ignore
/// let scanner = LsmScanner::new(base_table, region_snapshots, vec!["pk".to_string()])
///     .project(&["id", "name"])
///     .filter("id > 10")?
///     .limit(100, None);
///
/// let results = scanner.try_into_batch().await?;
/// ```
pub struct LsmScanner {
    // Data sources
    base_table: Arc<Dataset>,
    region_snapshots: Vec<RegionSnapshot>,
    active_memtables: HashMap<Uuid, ActiveMemTableRef>,

    // Query configuration
    projection: Option<Vec<String>>,
    filter: Option<Expr>,
    limit: Option<usize>,
    offset: Option<usize>,

    // Internal columns
    with_row_address: bool,
    with_memtable_gen: bool,

    // Primary key columns (required for deduplication)
    pk_columns: Vec<String>,
}

impl LsmScanner {
    /// Create a new LSM scanner.
    ///
    /// # Arguments
    ///
    /// * `base_table` - The base Lance table (merged data)
    /// * `region_snapshots` - Snapshots of region states from MemWAL index
    /// * `pk_columns` - Primary key column names for deduplication
    pub fn new(
        base_table: Arc<Dataset>,
        region_snapshots: Vec<RegionSnapshot>,
        pk_columns: Vec<String>,
    ) -> Self {
        Self {
            base_table,
            region_snapshots,
            active_memtables: HashMap::new(),
            projection: None,
            filter: None,
            limit: None,
            offset: None,
            with_row_address: false,
            with_memtable_gen: false,
            pk_columns,
        }
    }

    /// Add an active MemTable for strong consistency reads.
    ///
    /// Active MemTables contain data that may not be persisted yet.
    /// Including them provides strong consistency at the cost of
    /// requiring coordination with the writer.
    pub fn with_active_memtable(mut self, region_id: Uuid, memtable: ActiveMemTableRef) -> Self {
        self.active_memtables.insert(region_id, memtable);
        self
    }

    /// Project specific columns.
    ///
    /// If not called, all columns from the base schema are included.
    /// Primary key columns are always included for deduplication.
    pub fn project(mut self, columns: &[&str]) -> Self {
        self.projection = Some(columns.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Set filter expression using SQL-like syntax.
    ///
    /// The filter is pushed down to each data source when possible.
    pub fn filter(mut self, filter_expr: &str) -> Result<Self> {
        let ctx = SessionContext::new();
        let lance_schema = self.base_table.schema();
        let arrow_schema: arrow_schema::Schema = lance_schema.into();
        let df_schema = arrow_schema.to_dfschema().map_err(|e| {
            Error::invalid_input(format!("Failed to create DFSchema: {}", e), location!())
        })?;
        let expr = ctx.parse_sql_expr(filter_expr, &df_schema).map_err(|e| {
            Error::invalid_input(
                format!("Failed to parse filter expression: {}", e),
                location!(),
            )
        })?;
        self.filter = Some(expr);
        Ok(self)
    }

    /// Set filter expression directly.
    pub fn filter_expr(mut self, expr: Expr) -> Self {
        self.filter = Some(expr);
        self
    }

    /// Limit the number of results.
    pub fn limit(mut self, limit: usize, offset: Option<usize>) -> Self {
        self.limit = Some(limit);
        self.offset = offset;
        self
    }

    /// Include `_rowaddr` column in output.
    ///
    /// The row address is used for ordering within a generation.
    pub fn with_row_address(mut self) -> Self {
        self.with_row_address = true;
        self
    }

    /// Include `_memtable_gen` column in output.
    ///
    /// The generation column shows which data source each row came from:
    /// - 0: Base table
    /// - 1, 2, ...: MemTable generations (higher = newer)
    pub fn with_memtable_gen(mut self) -> Self {
        self.with_memtable_gen = true;
        self
    }

    /// Get the output schema.
    pub fn schema(&self) -> SchemaRef {
        // For now, return base schema. Full implementation would compute
        // the projected schema with optional _gen/_rowaddr columns.
        let lance_schema = self.base_table.schema();
        let arrow_schema: arrow_schema::Schema = lance_schema.into();
        Arc::new(arrow_schema)
    }

    /// Create the execution plan.
    pub async fn create_plan(&self) -> Result<Arc<dyn ExecutionPlan>> {
        let collector = self.build_collector();
        let base_schema = self.schema();
        let planner = LsmScanPlanner::new(collector, self.pk_columns.clone(), base_schema);

        planner
            .plan_scan(
                self.projection.as_deref(),
                self.filter.as_ref(),
                self.limit,
                self.offset,
                self.with_memtable_gen,
                self.with_row_address,
            )
            .await
    }

    /// Execute the scan and return a stream of record batches.
    pub async fn try_into_stream(&self) -> Result<SendableRecordBatchStream> {
        let plan = self.create_plan().await?;
        let ctx = SessionContext::new();
        let task_ctx = ctx.task_ctx();
        plan.execute(0, task_ctx)
            .map_err(|e| Error::io(format!("Failed to execute plan: {}", e), location!()))
    }

    /// Execute the scan and collect all results into a single RecordBatch.
    pub async fn try_into_batch(&self) -> Result<RecordBatch> {
        let stream = self.try_into_stream().await?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| Error::io(format!("Failed to collect batches: {}", e), location!()))?;

        if batches.is_empty() {
            let schema = self.schema();
            return Ok(RecordBatch::new_empty(schema));
        }

        let schema = batches[0].schema();
        arrow_select::concat::concat_batches(&schema, &batches)
            .map_err(|e| Error::io(format!("Failed to concatenate batches: {}", e), location!()))
    }

    /// Count the number of rows that match the query.
    pub async fn count_rows(&self) -> Result<u64> {
        let stream = self.try_into_stream().await?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| Error::io(format!("Failed to count rows: {}", e), location!()))?;

        Ok(batches.iter().map(|b| b.num_rows() as u64).sum())
    }

    /// Build the data source collector.
    fn build_collector(&self) -> LsmDataSourceCollector {
        let mut collector =
            LsmDataSourceCollector::new(self.base_table.clone(), self.region_snapshots.clone());

        for (region_id, memtable) in &self.active_memtables {
            collector = collector.with_active_memtable(*region_id, memtable.clone());
        }

        collector
    }
}

impl std::fmt::Debug for LsmScanner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LsmScanner")
            .field("base_table", &self.base_table.uri())
            .field("num_regions", &self.region_snapshots.len())
            .field("num_active_memtables", &self.active_memtables.len())
            .field("projection", &self.projection)
            .field("limit", &self.limit)
            .field("offset", &self.offset)
            .field("pk_columns", &self.pk_columns)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsm_scanner_builder() {
        // Test that the builder pattern compiles and works
        // Full integration tests would require a real dataset

        let pk_columns = ["id".to_string()];
        let region_snapshots: Vec<RegionSnapshot> = vec![];

        // We can't easily create an Arc<Dataset> without I/O,
        // so just test the type construction
        assert_eq!(pk_columns.len(), 1);
        assert!(region_snapshots.is_empty());
    }

    #[test]
    fn test_region_snapshot_construction() {
        use super::super::data_source::RegionSnapshot;

        let region_id = Uuid::new_v4();
        let snapshot = RegionSnapshot::new(region_id)
            .with_spec_id(1)
            .with_current_generation(5)
            .with_flushed_generation(1, "path/gen_1".to_string())
            .with_flushed_generation(2, "path/gen_2".to_string());

        assert_eq!(snapshot.region_id, region_id);
        assert_eq!(snapshot.spec_id, 1);
        assert_eq!(snapshot.current_generation, 5);
        assert_eq!(snapshot.flushed_generations.len(), 2);
    }

    #[test]
    fn test_active_memtable_ref() {
        use crate::dataset::mem_wal::write::{BatchStore, IndexStore};

        let batch_store = Arc::new(BatchStore::with_capacity(100));
        let index_store = Arc::new(IndexStore::new());
        let schema = Arc::new(arrow_schema::Schema::empty());

        let memtable_ref = ActiveMemTableRef {
            batch_store,
            index_store,
            schema,
            generation: 10,
        };

        assert_eq!(memtable_ref.generation, 10);
    }
}
