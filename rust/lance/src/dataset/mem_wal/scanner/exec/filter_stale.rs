// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! FilterStaleExec - Filters out rows that have newer versions in higher generations.
//!
//! Used in vector search and FTS queries to detect stale results across LSM levels.

use std::any::Any;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::SchemaRef;
use datafusion::error::Result as DFResult;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};
use futures::{Stream, StreamExt};
use lance_index::scalar::bloomfilter::sbbf::Sbbf;

use super::generation_tag::MEMTABLE_GEN_COLUMN;

/// Bloom filter for a specific generation.
#[derive(Clone)]
pub struct GenerationBloomFilter {
    /// Generation number (0 = base table, 1+ = memtables).
    pub generation: u64,
    /// The bloom filter.
    pub bloom_filter: Arc<Sbbf>,
}

impl std::fmt::Debug for GenerationBloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GenerationBloomFilter")
            .field("generation", &self.generation)
            .field(
                "bloom_filter_size",
                &self.bloom_filter.estimated_memory_size(),
            )
            .finish()
    }
}

/// Filters out rows that have a newer version in a higher generation.
///
/// For each candidate row with primary key `pk` from generation G, this node
/// checks bloom filters of all generations > G. If the bloom filter indicates
/// the key may exist in a newer generation, the candidate is filtered out.
///
/// # Bloom Filter Behavior
///
/// - False negatives: impossible (if key is in bloom filter, `check_hash` returns true)
/// - False positives: possible (may filter valid results that don't actually have newer versions)
///
/// This is acceptable for approximate search workloads (vector, FTS) where some
/// loss of recall is tolerable. The false positive rate is typically < 0.1%.
///
/// # Required Columns
///
/// The input must have:
/// - `_memtable_gen` (UInt64): Generation number for each row
/// - Primary key columns: Used for bloom filter hash computation
///
/// # Performance
///
/// - O(G) bloom filter checks per row, where G = number of newer generations
/// - Bloom filter checks are O(1)
/// - Overall: O(N * G) where N = input rows
#[derive(Debug)]
pub struct FilterStaleExec {
    /// Child execution plan.
    input: Arc<dyn ExecutionPlan>,
    /// Primary key column names (for hash computation).
    pk_columns: Vec<String>,
    /// Bloom filters for each generation, sorted by generation DESC.
    bloom_filters: Vec<GenerationBloomFilter>,
    /// Output schema.
    schema: SchemaRef,
    /// Plan properties.
    properties: PlanProperties,
}

impl FilterStaleExec {
    /// Create a new FilterStaleExec.
    ///
    /// # Arguments
    ///
    /// * `input` - Child plan producing rows with `_memtable_gen` column
    /// * `pk_columns` - Primary key column names for bloom filter hash
    /// * `bloom_filters` - Bloom filters for each generation (will be sorted by gen DESC)
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        pk_columns: Vec<String>,
        bloom_filters: Vec<GenerationBloomFilter>,
    ) -> Self {
        let schema = input.schema();

        // Sort bloom filters by generation DESC for efficient lookup
        let mut bloom_filters = bloom_filters;
        bloom_filters.sort_by(|a, b| b.generation.cmp(&a.generation));

        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            input.pipeline_behavior(),
            input.boundedness(),
        );

        Self {
            input,
            pk_columns,
            bloom_filters,
            schema,
            properties,
        }
    }

    /// Get the primary key columns.
    pub fn pk_columns(&self) -> &[String] {
        &self.pk_columns
    }

    /// Get the bloom filters.
    pub fn bloom_filters(&self) -> &[GenerationBloomFilter] {
        &self.bloom_filters
    }
}

impl DisplayAs for FilterStaleExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                let gens: Vec<String> = self
                    .bloom_filters
                    .iter()
                    .map(|bf| bf.generation.to_string())
                    .collect();
                write!(
                    f,
                    "FilterStaleExec: pk=[{}], generations=[{}]",
                    self.pk_columns.join(", "),
                    gens.join(", ")
                )
            }
        }
    }
}

impl ExecutionPlan for FilterStaleExec {
    fn name(&self) -> &str {
        "FilterStaleExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(datafusion::error::DataFusionError::Internal(
                "FilterStaleExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.pk_columns.clone(),
            self.bloom_filters.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;

        Ok(Box::pin(FilterStaleStream::new(
            input_stream,
            self.pk_columns.clone(),
            self.bloom_filters.clone(),
            self.schema.clone(),
        )))
    }
}

/// Stream that filters out stale rows.
struct FilterStaleStream {
    /// Input stream.
    input: SendableRecordBatchStream,
    /// Primary key column names.
    pk_columns: Vec<String>,
    /// Bloom filters sorted by generation DESC.
    bloom_filters: Vec<GenerationBloomFilter>,
    /// Output schema.
    schema: SchemaRef,
}

impl FilterStaleStream {
    fn new(
        input: SendableRecordBatchStream,
        pk_columns: Vec<String>,
        bloom_filters: Vec<GenerationBloomFilter>,
        schema: SchemaRef,
    ) -> Self {
        Self {
            input,
            pk_columns,
            bloom_filters,
            schema,
        }
    }

    /// Check if a row is stale (has newer version in higher generation).
    fn is_stale(&self, pk_hash: u64, row_generation: u64) -> bool {
        for bf in &self.bloom_filters {
            // Bloom filters are sorted DESC, so we can stop early
            if bf.generation <= row_generation {
                break;
            }
            if bf.bloom_filter.check_hash(pk_hash) {
                return true;
            }
        }
        false
    }

    /// Process a batch and filter out stale rows.
    fn filter_batch(&self, batch: RecordBatch) -> DFResult<RecordBatch> {
        if batch.num_rows() == 0 {
            return Ok(batch);
        }

        let gen_col = batch.column_by_name(MEMTABLE_GEN_COLUMN).ok_or_else(|| {
            datafusion::error::DataFusionError::Internal(format!(
                "Column '{}' not found in batch",
                MEMTABLE_GEN_COLUMN
            ))
        })?;
        let gen_array = gen_col
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!(
                    "Column '{}' is not UInt64",
                    MEMTABLE_GEN_COLUMN
                ))
            })?;

        let pk_indices: Vec<usize> = self
            .pk_columns
            .iter()
            .map(|col| {
                batch
                    .schema()
                    .column_with_name(col)
                    .map(|(idx, _)| idx)
                    .ok_or_else(|| {
                        datafusion::error::DataFusionError::Internal(format!(
                            "Primary key column '{}' not found",
                            col
                        ))
                    })
            })
            .collect::<DFResult<Vec<_>>>()?;

        let mut keep_indices: Vec<u32> = Vec::new();

        for row_idx in 0..batch.num_rows() {
            let row_generation = gen_array.value(row_idx);
            let pk_hash = compute_pk_hash(&batch, &pk_indices, row_idx);

            if !self.is_stale(pk_hash, row_generation) {
                keep_indices.push(row_idx as u32);
            }
        }

        if keep_indices.len() == batch.num_rows() {
            return Ok(batch);
        }

        if keep_indices.is_empty() {
            return Ok(RecordBatch::new_empty(self.schema.clone()));
        }

        let indices = arrow_array::UInt32Array::from(keep_indices);
        let columns: Vec<Arc<dyn Array>> = batch
            .columns()
            .iter()
            .map(|col| arrow_select::take::take(col.as_ref(), &indices, None))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))?;

        RecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
    }
}

/// Compute hash for a row's primary key.
fn compute_pk_hash(batch: &RecordBatch, pk_indices: &[usize], row_idx: usize) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    for &col_idx in pk_indices {
        let col = batch.column(col_idx);
        let is_null = col.is_null(row_idx);
        is_null.hash(&mut hasher);

        if !is_null {
            if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Int32Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Int64Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::BinaryArray>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::UInt32Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::UInt64Array>() {
                arr.value(row_idx).hash(&mut hasher);
            }
            // Add more types as needed
        }
    }

    hasher.finish()
}

impl Stream for FilterStaleStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.input.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let filtered = self.filter_batch(batch);
                Poll::Ready(Some(filtered))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl datafusion::physical_plan::RecordBatchStream for FilterStaleStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float32Array, Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::SessionContext;
    use datafusion_physical_plan::test::TestMemoryExec;
    use futures::TryStreamExt;

    fn create_test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("_distance", DataType::Float32, false),
            Field::new(MEMTABLE_GEN_COLUMN, DataType::UInt64, false),
        ]))
    }

    fn create_test_batch(schema: &Schema, ids: &[i32], gen: u64) -> RecordBatch {
        let names: Vec<String> = ids.iter().map(|id| format!("name_{}", id)).collect();
        let distances: Vec<f32> = ids.iter().map(|id| *id as f32 * 0.1).collect();
        let gens: Vec<u64> = vec![gen; ids.len()];

        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(StringArray::from(names)),
                Arc::new(Float32Array::from(distances)),
                Arc::new(UInt64Array::from(gens)),
            ],
        )
        .unwrap()
    }

    fn create_bloom_filter_with_keys(ids: &[i32]) -> Arc<Sbbf> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut bf = Sbbf::with_ndv_fpp(100, 0.01).unwrap();
        for id in ids {
            let mut hasher = DefaultHasher::new();
            false.hash(&mut hasher); // is_null = false
            id.hash(&mut hasher);
            let hash = hasher.finish();
            bf.insert_hash(hash);
        }
        Arc::new(bf)
    }

    #[tokio::test]
    async fn test_filter_stale_removes_rows_with_newer_versions() {
        let schema = create_test_schema();

        // Batch with rows from gen1: ids 1, 2, 3
        let batch = create_test_batch(&schema, &[1, 2, 3], 1);

        // Bloom filter for gen2 contains id=2
        let bf_gen2 = GenerationBloomFilter {
            generation: 2,
            bloom_filter: create_bloom_filter_with_keys(&[2]),
        };

        let input = TestMemoryExec::try_new_exec(&[vec![batch]], schema.clone(), None).unwrap();
        let filter = FilterStaleExec::new(input, vec!["id".to_string()], vec![bf_gen2]);

        let ctx = SessionContext::new();
        let stream = filter.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        // id=2 should be filtered (stale - exists in gen2)
        // id=1 and id=3 should remain
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);

        let ids: Vec<i32> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert!(ids.contains(&1));
        assert!(!ids.contains(&2)); // filtered
        assert!(ids.contains(&3));
    }

    #[tokio::test]
    async fn test_filter_stale_respects_generation_order() {
        let schema = create_test_schema();

        // Batch from gen2 with ids 1, 2
        let batch = create_test_batch(&schema, &[1, 2], 2);

        // Bloom filter for gen1 (older) contains id=1
        // This should NOT filter id=1 because gen1 < gen2
        let bf_gen1 = GenerationBloomFilter {
            generation: 1,
            bloom_filter: create_bloom_filter_with_keys(&[1]),
        };

        let input = TestMemoryExec::try_new_exec(&[vec![batch]], schema.clone(), None).unwrap();
        let filter = FilterStaleExec::new(input, vec!["id".to_string()], vec![bf_gen1]);

        let ctx = SessionContext::new();
        let stream = filter.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        // No rows should be filtered - gen1 bloom filter is for older gen
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);
    }

    #[tokio::test]
    async fn test_filter_stale_multiple_bloom_filters() {
        let schema = create_test_schema();

        // Batch from gen1 with ids 1, 2, 3, 4
        let batch = create_test_batch(&schema, &[1, 2, 3, 4], 1);

        // gen2 contains id=2, gen3 contains id=4
        let bf_gen2 = GenerationBloomFilter {
            generation: 2,
            bloom_filter: create_bloom_filter_with_keys(&[2]),
        };
        let bf_gen3 = GenerationBloomFilter {
            generation: 3,
            bloom_filter: create_bloom_filter_with_keys(&[4]),
        };

        let input = TestMemoryExec::try_new_exec(&[vec![batch]], schema.clone(), None).unwrap();
        let filter = FilterStaleExec::new(input, vec!["id".to_string()], vec![bf_gen2, bf_gen3]);

        let ctx = SessionContext::new();
        let stream = filter.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        // id=2 and id=4 should be filtered
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);

        let ids: Vec<i32> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
    }

    #[tokio::test]
    async fn test_filter_stale_no_bloom_filters() {
        let schema = create_test_schema();
        let batch = create_test_batch(&schema, &[1, 2, 3], 1);

        let input = TestMemoryExec::try_new_exec(&[vec![batch]], schema.clone(), None).unwrap();
        let filter = FilterStaleExec::new(input, vec!["id".to_string()], vec![]);

        let ctx = SessionContext::new();
        let stream = filter.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        // No bloom filters = nothing filtered
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 3);
    }

    #[tokio::test]
    async fn test_filter_stale_empty_batch() {
        let schema = create_test_schema();
        let batch = RecordBatch::new_empty(schema.clone());

        let bf = GenerationBloomFilter {
            generation: 2,
            bloom_filter: create_bloom_filter_with_keys(&[1]),
        };

        let input = TestMemoryExec::try_new_exec(&[vec![batch]], schema.clone(), None).unwrap();
        let filter = FilterStaleExec::new(input, vec!["id".to_string()], vec![bf]);

        let ctx = SessionContext::new();
        let stream = filter.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 0);
    }

    #[test]
    fn test_display() {
        let schema = create_test_schema();
        let batch = RecordBatch::new_empty(schema.clone());
        let input = TestMemoryExec::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let bf = GenerationBloomFilter {
            generation: 2,
            bloom_filter: create_bloom_filter_with_keys(&[1]),
        };

        let filter = FilterStaleExec::new(input, vec!["id".to_string()], vec![bf]);

        // Verify it doesn't panic
        let _ = format!("{:?}", filter);
    }
}
