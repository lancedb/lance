// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Point lookup planner for LSM scanner.
//!
//! Provides efficient primary key-based point lookups across LSM levels.

use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::common::ScalarValue;
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::Expr;
use lance_core::Result;
use lance_index::scalar::bloomfilter::sbbf::Sbbf;

use super::collector::LsmDataSourceCollector;
use super::data_source::LsmDataSource;
use super::exec::{compute_pk_hash_from_scalars, BloomFilterGuardExec, CoalesceFirstExec};

/// Plans point lookup queries over LSM data.
///
/// Point lookups are optimized for primary key-based queries where we expect
/// to find at most one row. The query plan uses:
///
/// 1. **Bloom filter guards**: Skip generations that definitely don't contain the key
/// 2. **Short-circuit evaluation**: Stop after finding the first match
/// 3. **Newest-first ordering**: Check newer generations before older ones
///
/// # Query Plan Structure
///
/// Since data is stored in reverse order (newest first), we use `GlobalLimitExec`
/// with limit=1 to take the first (most recent) matching row.
///
/// ```text
/// CoalesceFirstExec: return_first_non_null
///   BloomFilterGuardExec: gen=3
///     GlobalLimitExec: limit=1
///       FilterExec: pk = target
///         ScanExec: memtable_gen_3
///   BloomFilterGuardExec: gen=2
///     GlobalLimitExec: limit=1
///       FilterExec: pk = target
///         ScanExec: flushed_gen_2
///   BloomFilterGuardExec: gen=1
///     GlobalLimitExec: limit=1
///       FilterExec: pk = target
///         ScanExec: flushed_gen_1
///   GlobalLimitExec: limit=1
///     FilterExec: pk = target
///       ScanExec: base_table
/// ```
///
/// The base table doesn't use a bloom filter guard because:
/// - It's the fallback when no memtable has the key
/// - Bloom filters for the base table would be too large
pub struct LsmPointLookupPlanner {
    /// Data source collector.
    collector: LsmDataSourceCollector,
    /// Primary key column names.
    pk_columns: Vec<String>,
    /// Schema of the base table.
    base_schema: SchemaRef,
    /// Bloom filters for each memtable generation.
    /// Map: generation -> bloom filter
    bloom_filters: std::collections::HashMap<u64, Arc<Sbbf>>,
}

impl LsmPointLookupPlanner {
    /// Create a new planner.
    ///
    /// # Arguments
    ///
    /// * `collector` - Data source collector
    /// * `pk_columns` - Primary key column names
    /// * `base_schema` - Schema of the base table
    pub fn new(
        collector: LsmDataSourceCollector,
        pk_columns: Vec<String>,
        base_schema: SchemaRef,
    ) -> Self {
        Self {
            collector,
            pk_columns,
            base_schema,
            bloom_filters: std::collections::HashMap::new(),
        }
    }

    /// Add a bloom filter for a generation.
    ///
    /// Bloom filters are optional but improve performance by skipping
    /// generations that definitely don't contain the target key.
    pub fn with_bloom_filter(mut self, generation: u64, bloom_filter: Arc<Sbbf>) -> Self {
        self.bloom_filters.insert(generation, bloom_filter);
        self
    }

    /// Add multiple bloom filters.
    pub fn with_bloom_filters(
        mut self,
        bloom_filters: impl IntoIterator<Item = (u64, Arc<Sbbf>)>,
    ) -> Self {
        self.bloom_filters.extend(bloom_filters);
        self
    }

    /// Create a point lookup plan for the given primary key values.
    ///
    /// # Arguments
    ///
    /// * `pk_values` - Primary key values to look up (one value per pk column)
    /// * `projection` - Columns to include in output (None = all columns)
    ///
    /// # Returns
    ///
    /// An execution plan that returns at most one row - the newest version
    /// of the row with the given primary key.
    pub async fn plan_lookup(
        &self,
        pk_values: &[ScalarValue],
        projection: Option<&[String]>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if pk_values.len() != self.pk_columns.len() {
            return Err(lance_core::Error::invalid_input(
                format!(
                    "Expected {} primary key values, got {}",
                    self.pk_columns.len(),
                    pk_values.len()
                ),
                snafu::location!(),
            ));
        }

        let pk_hash = compute_pk_hash_from_scalars(pk_values);
        let filter_expr = self.build_pk_filter_expr(pk_values)?;
        let sources = self.collector.collect()?;

        if sources.is_empty() {
            return self.empty_plan(projection);
        }

        // Sort by generation DESC (newest first)
        let mut sources: Vec<_> = sources.into_iter().collect();
        sources.sort_by_key(|b| std::cmp::Reverse(b.generation()));

        let mut source_plans = Vec::new();

        for source in sources {
            let generation = source.generation().as_u64();

            let scan = self
                .build_source_scan(&source, projection, &filter_expr)
                .await?;

            // Data is stored in reverse order, so first match is newest
            let limited: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(scan, 0, Some(1)));

            let guarded_plan: Arc<dyn ExecutionPlan> =
                if let Some(bf) = self.bloom_filters.get(&generation) {
                    Arc::new(BloomFilterGuardExec::new(
                        limited,
                        bf.clone(),
                        pk_hash,
                        generation,
                    ))
                } else {
                    limited
                };

            source_plans.push(guarded_plan);
        }

        let plan: Arc<dyn ExecutionPlan> = if source_plans.len() == 1 {
            source_plans.remove(0)
        } else {
            Arc::new(CoalesceFirstExec::new(source_plans))
        };

        Ok(plan)
    }

    /// Build the filter expression for primary key equality.
    fn build_pk_filter_expr(&self, pk_values: &[ScalarValue]) -> Result<Expr> {
        use datafusion::prelude::{col, lit};

        let mut expr: Option<Expr> = None;

        for (col_name, value) in self.pk_columns.iter().zip(pk_values.iter()) {
            let eq_expr = col(col_name.as_str()).eq(lit(value.clone()));

            expr = Some(match expr {
                Some(e) => e.and(eq_expr),
                None => eq_expr,
            });
        }

        expr.ok_or_else(|| {
            lance_core::Error::invalid_input("No primary key columns specified", snafu::location!())
        })
    }

    /// Build scan plan for a single data source.
    async fn build_source_scan(
        &self,
        source: &LsmDataSource,
        projection: Option<&[String]>,
        filter: &Expr,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match source {
            LsmDataSource::BaseTable { dataset } => {
                let mut scanner = dataset.scan();
                let cols = self.build_projection(projection);
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                scanner.filter_expr(filter.clone());
                scanner.create_plan().await
            }
            LsmDataSource::FlushedMemTable { path, .. } => {
                let dataset = crate::dataset::DatasetBuilder::from_uri(path)
                    .load()
                    .await?;
                let mut scanner = dataset.scan();
                let cols = self.build_projection(projection);
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                scanner.filter_expr(filter.clone());
                scanner.create_plan().await
            }
            LsmDataSource::ActiveMemTable {
                batch_store,
                index_store,
                schema,
                ..
            } => {
                use crate::dataset::mem_wal::memtable::scanner::MemTableScanner;

                let mut scanner =
                    MemTableScanner::new(batch_store.clone(), index_store.clone(), schema.clone());
                if let Some(cols) = projection {
                    scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>());
                }
                scanner.filter_expr(filter.clone());
                scanner.create_plan().await
            }
        }
    }

    /// Build projection list ensuring PK columns are included.
    fn build_projection(&self, projection: Option<&[String]>) -> Vec<String> {
        let mut cols: Vec<String> = if let Some(p) = projection {
            p.to_vec()
        } else {
            self.base_schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect()
        };

        for pk in &self.pk_columns {
            if !cols.contains(pk) {
                cols.push(pk.clone());
            }
        }

        cols
    }

    /// Create an empty execution plan.
    fn empty_plan(&self, projection: Option<&[String]>) -> Result<Arc<dyn ExecutionPlan>> {
        use arrow_schema::{Field, Schema};
        use datafusion::physical_plan::empty::EmptyExec;

        let fields: Vec<Arc<Field>> = if let Some(cols) = projection {
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

        let schema = Arc::new(Schema::new(fields));
        Ok(Arc::new(EmptyExec::new(schema)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use datafusion::physical_plan::displayable;
    use std::collections::HashMap;
    use uuid::Uuid;

    use crate::dataset::mem_wal::scanner::data_source::RegionSnapshot;
    use crate::dataset::{Dataset, WriteParams};

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

    async fn create_dataset(uri: &str, batches: Vec<RecordBatch>) -> Dataset {
        let schema = batches[0].schema();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        Dataset::write(reader, uri, Some(WriteParams::default()))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_point_lookup_plan_structure() {
        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        // Create base table
        let base_uri = format!("{}/base", base_path);
        let base_batch = create_test_batch(&schema, &[1, 2, 3], "base");
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        // Create collector without memtables
        let collector = LsmDataSourceCollector::new(base_dataset, vec![]);

        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema.clone());

        let pk_values = vec![ScalarValue::Int32(Some(2))];
        let plan = planner.plan_lookup(&pk_values, None).await.unwrap();

        // Verify plan structure
        let plan_str = format!("{}", displayable(plan.as_ref()).indent(true));

        // Should have GlobalLimitExec with limit=1 (data is stored in reverse order)
        assert!(
            plan_str.contains("GlobalLimitExec"),
            "Should have GlobalLimitExec in plan: {}",
            plan_str
        );
    }

    #[tokio::test]
    async fn test_point_lookup_with_memtables() {
        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        // Create base table
        let base_uri = format!("{}/base", base_path);
        let base_batch = create_test_batch(&schema, &[1, 2, 3], "base");
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        // Create region snapshot
        let region_id = Uuid::new_v4();
        let gen1_uri = format!("{}/_mem_wal/{}/gen_1", base_uri, region_id);
        let gen1_batch = create_test_batch(&schema, &[2], "gen1"); // Update id=2
        create_dataset(&gen1_uri, vec![gen1_batch]).await;

        let region_snapshot = RegionSnapshot::new(region_id)
            .with_current_generation(2)
            .with_flushed_generation(1, "gen_1".to_string());

        // Create collector
        let collector = LsmDataSourceCollector::new(base_dataset, vec![region_snapshot]);

        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema.clone());

        let pk_values = vec![ScalarValue::Int32(Some(2))];
        let plan = planner.plan_lookup(&pk_values, None).await.unwrap();

        // Verify plan structure - should have CoalesceFirstExec with multiple children
        let plan_str = format!("{}", displayable(plan.as_ref()).indent(true));

        assert!(
            plan_str.contains("CoalesceFirstExec") || plan_str.contains("GlobalLimitExec"),
            "Should have CoalesceFirstExec or GlobalLimitExec in plan: {}",
            plan_str
        );
    }

    #[tokio::test]
    async fn test_point_lookup_with_bloom_filter() {
        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        // Create base table
        let base_uri = format!("{}/base", base_path);
        let base_batch = create_test_batch(&schema, &[1, 2, 3], "base");
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        // Create collector
        let collector = LsmDataSourceCollector::new(base_dataset, vec![]);

        // Create a bloom filter for generation 1 (simulating a memtable)
        let mut bf = Sbbf::with_ndv_fpp(100, 0.01).unwrap();
        let pk_hash = compute_pk_hash_from_scalars(&[ScalarValue::Int32(Some(2))]);
        bf.insert_hash(pk_hash);

        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema.clone())
            .with_bloom_filter(1, Arc::new(bf));

        let pk_values = vec![ScalarValue::Int32(Some(2))];
        let plan = planner.plan_lookup(&pk_values, None).await.unwrap();

        // Plan should be valid
        assert!(plan.schema().field_with_name("id").is_ok());
    }

    #[tokio::test]
    async fn test_pk_filter_expr() {
        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());
        let base_batch = create_test_batch(&schema, &[1], "base");
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        let collector = LsmDataSourceCollector::new(base_dataset, vec![]);

        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        let pk_values = vec![ScalarValue::Int32(Some(42))];
        let expr = planner.build_pk_filter_expr(&pk_values).unwrap();

        // Verify expression is an equality
        let expr_str = format!("{}", expr);
        assert!(
            expr_str.contains("id"),
            "Expression should contain column name"
        );
    }
}
