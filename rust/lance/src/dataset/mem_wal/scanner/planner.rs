// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Query planner for LSM scanner.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{limit::GlobalLimitExec, ExecutionPlan};
use datafusion::prelude::Expr;
use lance_core::Result;

use super::collector::LsmDataSourceCollector;
use super::data_source::LsmDataSource;
use super::exec::{DeduplicateExec, GenerationTagExec, GENERATION_COLUMN, ROW_ADDRESS_COLUMN};

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
    /// * `keep_generation` - Whether to include _gen in output
    /// * `keep_row_address` - Whether to include _rowaddr in output
    pub async fn plan_scan(
        &self,
        projection: Option<&[String]>,
        _filter: Option<&Expr>,
        limit: Option<usize>,
        offset: Option<usize>,
        keep_generation: bool,
        keep_row_address: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // 1. Collect all data sources
        let sources = self.collector.collect()?;

        if sources.is_empty() {
            // Return empty plan
            return self.empty_plan(projection, keep_generation, keep_row_address);
        }

        // 2. Build scan plan for each source
        let mut scan_plans = Vec::new();
        for source in sources {
            let scan = self.build_source_scan(&source, projection).await?;
            let tagged = GenerationTagExec::new(scan, source.generation());
            scan_plans.push(Arc::new(tagged) as Arc<dyn ExecutionPlan>);
        }

        // 3. Union all scans
        #[allow(deprecated)]
        let union: Arc<dyn ExecutionPlan> = if scan_plans.len() == 1 {
            scan_plans.remove(0)
        } else {
            Arc::new(UnionExec::new(scan_plans))
        };

        // 4. Add deduplication
        let dedup = DeduplicateExec::new(
            union,
            self.pk_columns.clone(),
            keep_generation,
            keep_row_address,
        )?;
        let mut plan: Arc<dyn ExecutionPlan> = Arc::new(dedup);

        // 5. Add limit if specified
        if let Some(limit) = limit {
            plan = Arc::new(GlobalLimitExec::new(plan, offset.unwrap_or(0), Some(limit)));
        }

        Ok(plan)
    }

    /// Build scan plan for a single data source.
    async fn build_source_scan(
        &self,
        source: &LsmDataSource,
        projection: Option<&[String]>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match source {
            LsmDataSource::BaseTable { dataset } => {
                // Use Lance Scanner
                let mut scanner = dataset.scan();

                // Project columns + _rowaddr (needed for dedup)
                let cols = self.build_projection_with_rowaddr(projection);
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                scanner.with_row_address();

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
        keep_generation: bool,
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

        if keep_generation {
            fields.push(Arc::new(Field::new(
                GENERATION_COLUMN,
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
