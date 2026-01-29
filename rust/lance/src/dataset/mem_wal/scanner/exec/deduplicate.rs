// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Deduplication execution node for LSM merge reads.

use std::any::Any;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{Array, RecordBatch};
use arrow_schema::{Field, Schema, SchemaRef, SortOptions};
use datafusion::common::ScalarValue;
use datafusion::error::Result as DFResult;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{
    EquivalenceProperties, LexOrdering, Partitioning, PhysicalSortExpr,
};
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};
use futures::{Stream, StreamExt};
use lance_core::{Error, Result};
use snafu::location;

use super::generation_tag::MEMTABLE_GEN_COLUMN;

/// Column name for row address (used for ordering within generation).
pub const ROW_ADDRESS_COLUMN: &str = "_rowaddr";

/// Deduplicates rows by primary key, keeping the row with highest (_memtable_gen, _rowaddr).
///
/// # Algorithm
///
/// 1. Sort input by (pk_columns, _memtable_gen DESC, _rowaddr DESC) - if not already sorted
/// 2. Stream through sorted data, emit only first row per PK
///
/// After sorting, the first occurrence of each PK has the highest (_memtable_gen, _rowaddr),
/// so we can deduplicate in a single streaming pass.
///
/// # Pre-sorted Input Optimization
///
/// When `input_sorted` is true, the input is assumed to already be sorted by
/// (pk_columns ASC, _memtable_gen DESC, _rowaddr DESC). This allows skipping the internal
/// sort, which is useful when the input comes from SortPreservingMergeExec that
/// has already merged K pre-sorted streams.
///
/// # Memory Efficiency
///
/// Uses DataFusion's SortExec for external sort when data exceeds memory.
/// The streaming deduplication pass requires O(1) memory per partition.
#[derive(Debug)]
pub struct DeduplicateExec {
    /// Child plan (UnionExec of tagged scans).
    input: Arc<dyn ExecutionPlan>,
    /// Primary key column names.
    pk_columns: Vec<String>,
    /// Output schema.
    schema: SchemaRef,
    /// Whether to keep _memtable_gen in output.
    with_memtable_gen: bool,
    /// Whether to keep _rowaddr in output.
    keep_row_address: bool,
    /// Whether the input is already sorted by (pk, _memtable_gen DESC, _rowaddr DESC).
    input_sorted: bool,
    /// Plan properties.
    properties: PlanProperties,
}

impl DeduplicateExec {
    /// Create a new deduplication executor.
    ///
    /// # Arguments
    ///
    /// * `input` - Child plan producing tagged rows
    /// * `pk_columns` - Primary key column names for deduplication
    /// * `with_memtable_gen` - Whether to include _memtable_gen in output
    /// * `keep_row_address` - Whether to include _rowaddr in output
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        pk_columns: Vec<String>,
        with_memtable_gen: bool,
        keep_row_address: bool,
    ) -> Result<Self> {
        Self::new_with_sorted(
            input,
            pk_columns,
            with_memtable_gen,
            keep_row_address,
            false,
        )
    }

    /// Create a new deduplication executor with pre-sorted input.
    ///
    /// # Arguments
    ///
    /// * `input` - Child plan producing tagged rows
    /// * `pk_columns` - Primary key column names for deduplication
    /// * `with_memtable_gen` - Whether to include _memtable_gen in output
    /// * `keep_row_address` - Whether to include _rowaddr in output
    /// * `input_sorted` - Whether the input is already sorted by (pk, _memtable_gen DESC, _rowaddr DESC)
    pub fn new_with_sorted(
        input: Arc<dyn ExecutionPlan>,
        pk_columns: Vec<String>,
        with_memtable_gen: bool,
        keep_row_address: bool,
        input_sorted: bool,
    ) -> Result<Self> {
        let input_schema = input.schema();

        // Validate that required columns exist
        for col in &pk_columns {
            if input_schema.column_with_name(col).is_none() {
                return Err(Error::invalid_input(
                    format!("Primary key column '{}' not found in input schema", col),
                    location!(),
                ));
            }
        }

        if input_schema.column_with_name(MEMTABLE_GEN_COLUMN).is_none() {
            return Err(Error::invalid_input(
                format!(
                    "Generation column '{}' not found in input schema",
                    MEMTABLE_GEN_COLUMN
                ),
                location!(),
            ));
        }

        if input_schema.column_with_name(ROW_ADDRESS_COLUMN).is_none() {
            return Err(Error::invalid_input(
                format!(
                    "Row address column '{}' not found in input schema",
                    ROW_ADDRESS_COLUMN
                ),
                location!(),
            ));
        }

        // Build output schema (may exclude internal columns)
        let output_fields: Vec<Arc<Field>> = input_schema
            .fields()
            .iter()
            .filter(|f| {
                let name = f.name();
                if name == MEMTABLE_GEN_COLUMN && !with_memtable_gen {
                    return false;
                }
                if name == ROW_ADDRESS_COLUMN && !keep_row_address {
                    return false;
                }
                true
            })
            .cloned()
            .collect();
        let schema = Arc::new(Schema::new(output_fields));

        // Output is single partition after sort + dedup
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            input.pipeline_behavior(),
            input.boundedness(),
        );

        Ok(Self {
            input,
            pk_columns,
            schema,
            with_memtable_gen,
            keep_row_address,
            input_sorted,
            properties,
        })
    }

    /// Create a deduplication executor for pre-sorted input without _memtable_gen column.
    ///
    /// This is used when the input is already sorted by (pk ASC, _rowaddr DESC) with
    /// newer generations appearing first (via stream ordering). The _memtable_gen column is
    /// not required in the input schema unless `with_memtable_gen=true`.
    ///
    /// # Arguments
    ///
    /// * `input` - Child plan producing rows sorted by (pk ASC, _rowaddr DESC)
    /// * `pk_columns` - Primary key column names for deduplication
    /// * `with_memtable_gen` - Whether to include _memtable_gen in output (requires _memtable_gen in input)
    /// * `keep_row_address` - Whether to include _rowaddr in output
    pub fn new_sorted(
        input: Arc<dyn ExecutionPlan>,
        pk_columns: Vec<String>,
        with_memtable_gen: bool,
        keep_row_address: bool,
    ) -> Result<Self> {
        let input_schema = input.schema();

        // Validate that required columns exist
        for col in &pk_columns {
            if input_schema.column_with_name(col).is_none() {
                return Err(Error::invalid_input(
                    format!("Primary key column '{}' not found in input schema", col),
                    location!(),
                ));
            }
        }

        // _memtable_gen column is only required if with_memtable_gen=true
        if with_memtable_gen && input_schema.column_with_name(MEMTABLE_GEN_COLUMN).is_none() {
            return Err(Error::invalid_input(
                format!(
                    "Generation column '{}' not found in input schema (required when with_memtable_gen=true)",
                    MEMTABLE_GEN_COLUMN
                ),
                location!(),
            ));
        }

        if input_schema.column_with_name(ROW_ADDRESS_COLUMN).is_none() {
            return Err(Error::invalid_input(
                format!(
                    "Row address column '{}' not found in input schema",
                    ROW_ADDRESS_COLUMN
                ),
                location!(),
            ));
        }

        // Build output schema (may exclude internal columns)
        let output_fields: Vec<Arc<Field>> = input_schema
            .fields()
            .iter()
            .filter(|f| {
                let name = f.name();
                if name == MEMTABLE_GEN_COLUMN && !with_memtable_gen {
                    return false;
                }
                if name == ROW_ADDRESS_COLUMN && !keep_row_address {
                    return false;
                }
                true
            })
            .cloned()
            .collect();
        let schema = Arc::new(Schema::new(output_fields));

        // Output is single partition after dedup
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            input.pipeline_behavior(),
            input.boundedness(),
        );

        Ok(Self {
            input,
            pk_columns,
            schema,
            with_memtable_gen,
            keep_row_address,
            input_sorted: true,
            properties,
        })
    }

    /// Get the primary key columns.
    pub fn pk_columns(&self) -> &[String] {
        &self.pk_columns
    }

    /// Build sort expressions for deduplication ordering.
    fn build_sort_exprs(&self) -> DFResult<Vec<PhysicalSortExpr>> {
        let input_schema = self.input.schema();
        let mut sort_exprs = Vec::new();

        // Sort by PK columns (ASC) to group duplicates together
        for col in &self.pk_columns {
            let (idx, _) = input_schema.column_with_name(col).ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!("Column '{}' not found", col))
            })?;
            sort_exprs.push(PhysicalSortExpr {
                expr: Arc::new(Column::new(col, idx)),
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            });
        }

        // Sort by _memtable_gen DESC (higher generation = newer)
        let (gen_idx, _) = input_schema
            .column_with_name(MEMTABLE_GEN_COLUMN)
            .expect("_memtable_gen column validated in constructor");
        sort_exprs.push(PhysicalSortExpr {
            expr: Arc::new(Column::new(MEMTABLE_GEN_COLUMN, gen_idx)),
            options: SortOptions {
                descending: true,
                nulls_first: false,
            },
        });

        // Sort by _rowaddr DESC (higher address = newer within generation)
        let (addr_idx, _) = input_schema
            .column_with_name(ROW_ADDRESS_COLUMN)
            .expect("_rowaddr column validated in constructor");
        sort_exprs.push(PhysicalSortExpr {
            expr: Arc::new(Column::new(ROW_ADDRESS_COLUMN, addr_idx)),
            options: SortOptions {
                descending: true,
                nulls_first: false,
            },
        });

        Ok(sort_exprs)
    }

    /// Build the internal sorted execution plan.
    fn build_sorted_plan(&self) -> DFResult<Arc<dyn ExecutionPlan>> {
        let sort_exprs = self.build_sort_exprs()?;
        let lex_ordering = LexOrdering::new(sort_exprs).ok_or_else(|| {
            datafusion::error::DataFusionError::Internal(
                "Failed to create LexOrdering: empty sort expressions".to_string(),
            )
        })?;
        let sort_exec = SortExec::new(lex_ordering, self.input.clone());
        Ok(Arc::new(sort_exec))
    }

    /// Get column indices for PK comparison.
    fn pk_indices(&self) -> Vec<usize> {
        let schema = self.input.schema();
        self.pk_columns
            .iter()
            .map(|col| schema.column_with_name(col).unwrap().0)
            .collect()
    }

    /// Get column indices to keep in output.
    fn output_indices(&self) -> Vec<usize> {
        let input_schema = self.input.schema();
        input_schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                let name = f.name();
                if name == MEMTABLE_GEN_COLUMN && !self.with_memtable_gen {
                    return false;
                }
                if name == ROW_ADDRESS_COLUMN && !self.keep_row_address {
                    return false;
                }
                true
            })
            .map(|(i, _)| i)
            .collect()
    }
}

impl DisplayAs for DeduplicateExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "DeduplicateExec: pk=[{}], with_memtable_gen={}, keep_addr={}, input_sorted={}",
                    self.pk_columns.join(", "),
                    self.with_memtable_gen,
                    self.keep_row_address,
                    self.input_sorted
                )
            }
        }
    }
}

impl ExecutionPlan for DeduplicateExec {
    fn name(&self) -> &str {
        "DeduplicateExec"
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
                "DeduplicateExec requires exactly one child".to_string(),
            ));
        }
        Ok(Arc::new(
            Self::new_with_sorted(
                children[0].clone(),
                self.pk_columns.clone(),
                self.with_memtable_gen,
                self.keep_row_address,
                self.input_sorted,
            )
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?,
        ))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        // Either use input directly (if pre-sorted) or wrap in sort
        let sorted_stream = if self.input_sorted {
            // Input is already sorted, use directly
            self.input.execute(partition, context)?
        } else {
            // Build and execute the sorted plan
            let sorted_plan = self.build_sorted_plan()?;
            sorted_plan.execute(partition, context)?
        };

        Ok(Box::pin(DeduplicateStream::new(
            sorted_stream,
            self.pk_indices(),
            self.output_indices(),
            self.schema.clone(),
        )))
    }
}

/// Streaming deduplication on sorted input.
struct DeduplicateStream {
    input: SendableRecordBatchStream,
    pk_indices: Vec<usize>,
    output_indices: Vec<usize>,
    schema: SchemaRef,
    /// Last PK values seen (for comparison).
    last_pk: Option<Vec<Arc<dyn Array>>>,
}

impl DeduplicateStream {
    fn new(
        input: SendableRecordBatchStream,
        pk_indices: Vec<usize>,
        output_indices: Vec<usize>,
        schema: SchemaRef,
    ) -> Self {
        Self {
            input,
            pk_indices,
            output_indices,
            schema,
            last_pk: None,
        }
    }

    /// Process a batch and return deduplicated rows.
    fn process_batch(&mut self, batch: RecordBatch) -> DFResult<RecordBatch> {
        if batch.num_rows() == 0 {
            return Ok(RecordBatch::new_empty(self.schema.clone()));
        }

        let mut keep_indices = Vec::new();

        for row_idx in 0..batch.num_rows() {
            let current_pk: Vec<Arc<dyn Array>> = self
                .pk_indices
                .iter()
                .map(|&col_idx| batch.column(col_idx).slice(row_idx, 1))
                .collect();

            let is_new_pk = match &self.last_pk {
                None => true,
                Some(last) => !pk_equals(&current_pk, last),
            };

            if is_new_pk {
                // This is the first (newest) row for this PK
                keep_indices.push(row_idx);
                self.last_pk = Some(current_pk);
            }
            // Else: duplicate PK with lower gen/rowaddr, skip it
        }

        // Build output batch with only kept rows
        self.filter_batch(&batch, &keep_indices)
    }

    /// Filter batch to only include specified row indices.
    fn filter_batch(&self, batch: &RecordBatch, indices: &[usize]) -> DFResult<RecordBatch> {
        if indices.is_empty() {
            return Ok(RecordBatch::new_empty(self.schema.clone()));
        }

        let indices_array =
            arrow_array::UInt32Array::from(indices.iter().map(|&i| i as u32).collect::<Vec<_>>());

        // Select only output columns
        let columns: Vec<Arc<dyn Array>> = self
            .output_indices
            .iter()
            .map(|&col_idx| {
                let col = batch.column(col_idx);
                arrow_select::take::take(col.as_ref(), &indices_array, None)
                    .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
            })
            .collect::<DFResult<Vec<_>>>()?;

        RecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
    }
}

/// Compare two PK tuples for equality.
fn pk_equals(a: &[Arc<dyn Array>], b: &[Arc<dyn Array>]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for (col_a, col_b) in a.iter().zip(b.iter()) {
        // Each array has 1 element (single row) - convert to ScalarValue for comparison
        let val_a = ScalarValue::try_from_array(col_a.as_ref(), 0);
        let val_b = ScalarValue::try_from_array(col_b.as_ref(), 0);

        match (val_a, val_b) {
            (Ok(a), Ok(b)) => {
                if a != b {
                    return false;
                }
            }
            _ => return false,
        }
    }

    true
}

impl Stream for DeduplicateStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.input.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let result = self.process_batch(batch);
                Poll::Ready(Some(result))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl datafusion::physical_plan::RecordBatchStream for DeduplicateStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray, UInt64Array};
    use datafusion::prelude::SessionContext;
    use datafusion_physical_plan::test::TestMemoryExec;

    fn create_test_data() -> (SchemaRef, Vec<RecordBatch>) {
        // Schema: id (PK), name, _memtable_gen, _rowaddr
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", arrow_schema::DataType::Int32, false),
            Field::new("name", arrow_schema::DataType::Utf8, true),
            Field::new(MEMTABLE_GEN_COLUMN, arrow_schema::DataType::UInt64, false),
            Field::new(ROW_ADDRESS_COLUMN, arrow_schema::DataType::UInt64, false),
        ]));

        // Data with duplicates:
        // id=1: gen=0 (base), gen=2 (memtable) -> keep gen=2
        // id=2: gen=0 only -> keep gen=0
        // id=3: gen=1, gen=2 -> keep gen=2
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 1, 3, 3])),
                Arc::new(StringArray::from(vec![
                    "old_1", "only_2", "new_1", "old_3", "new_3",
                ])),
                Arc::new(UInt64Array::from(vec![0, 0, 2, 1, 2])),
                Arc::new(UInt64Array::from(vec![100, 200, 50, 10, 20])),
            ],
        )
        .unwrap();

        (schema, vec![batch])
    }

    #[tokio::test]
    async fn test_deduplicate_exec() {
        let (schema, batches) = create_test_data();

        let input = TestMemoryExec::try_new_exec(&[batches], schema, None).unwrap();

        let dedup = DeduplicateExec::new(
            input,
            vec!["id".to_string()],
            false, // don't keep _memtable_gen
            false, // don't keep _rowaddr
        )
        .unwrap();

        // Output schema should only have id, name
        assert_eq!(dedup.schema().fields().len(), 2);
        assert_eq!(dedup.schema().field(0).name(), "id");
        assert_eq!(dedup.schema().field(1).name(), "name");

        let ctx = SessionContext::new();
        let stream = dedup.execute(0, ctx.task_ctx()).unwrap();
        let result_batches: Vec<_> = stream.collect::<Vec<_>>().await;

        // Concatenate results
        let mut all_ids = Vec::new();
        let mut all_names = Vec::new();
        for batch_result in result_batches {
            let batch = batch_result.unwrap();
            if batch.num_rows() > 0 {
                let ids = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                let names = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                for i in 0..batch.num_rows() {
                    all_ids.push(ids.value(i));
                    all_names.push(names.value(i).to_string());
                }
            }
        }

        // Should have 3 unique rows
        assert_eq!(all_ids.len(), 3);

        // Find each id and verify the correct version was kept
        for (id, name) in all_ids.iter().zip(all_names.iter()) {
            match id {
                1 => assert_eq!(name, "new_1", "id=1 should keep gen=2 version"),
                2 => assert_eq!(name, "only_2", "id=2 has only one version"),
                3 => assert_eq!(name, "new_3", "id=3 should keep gen=2 version"),
                _ => panic!("Unexpected id: {}", id),
            }
        }
    }

    #[tokio::test]
    async fn test_deduplicate_with_memtable_gen() {
        let (schema, batches) = create_test_data();

        let input = TestMemoryExec::try_new_exec(&[batches], schema, None).unwrap();

        let dedup = DeduplicateExec::new(
            input,
            vec!["id".to_string()],
            true,  // keep _memtable_gen
            false, // don't keep _rowaddr
        )
        .unwrap();

        // Output schema should have id, name, _memtable_gen
        assert_eq!(dedup.schema().fields().len(), 3);
        assert_eq!(dedup.schema().field(2).name(), MEMTABLE_GEN_COLUMN);
    }

    #[test]
    fn test_deduplicate_missing_pk_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", arrow_schema::DataType::Int32, false),
            Field::new(MEMTABLE_GEN_COLUMN, arrow_schema::DataType::UInt64, false),
            Field::new(ROW_ADDRESS_COLUMN, arrow_schema::DataType::UInt64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(UInt64Array::from(vec![1])),
                Arc::new(UInt64Array::from(vec![1])),
            ],
        )
        .unwrap();

        let input = TestMemoryExec::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let result = DeduplicateExec::new(input, vec!["nonexistent".to_string()], false, false);

        assert!(result.is_err());
    }

    #[test]
    fn test_display() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", arrow_schema::DataType::Int32, false),
            Field::new("name", arrow_schema::DataType::Utf8, true),
            Field::new(MEMTABLE_GEN_COLUMN, arrow_schema::DataType::UInt64, false),
            Field::new(ROW_ADDRESS_COLUMN, arrow_schema::DataType::UInt64, false),
        ]));

        let batch = RecordBatch::new_empty(schema.clone());
        let input = TestMemoryExec::try_new_exec(&[vec![batch]], schema, None).unwrap();

        let dedup = DeduplicateExec::new(input, vec!["id".to_string()], true, false).unwrap();

        // Test Debug format
        let debug_str = format!("{:?}", dedup);
        assert!(debug_str.contains("DeduplicateExec"));
        assert!(debug_str.contains("pk_columns"));
    }
}
