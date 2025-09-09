// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, OnceLock};

use arrow_array::{Array, ArrayRef, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    common::{stats::Precision, Statistics},
    execution::{SendableRecordBatchStream, TaskContext},
    physical_plan::{
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
    },
    scalar::ScalarValue,
};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use futures::TryStreamExt;
use lance_core::{Error, Result, ROW_ID};
use lance_index::{
    metrics::NoOpMetricsCollector,
    scalar::{SargableQuery, ScalarIndex, SearchResult},
    DatasetIndexExt, ScalarIndexCriteria,
};
use snafu::location;
use tracing::instrument;

use crate::{index::DatasetIndexInternalExt, Dataset};

/// An execution node that performs a join between source data and target table using a scalar index.
///
/// This node takes source data and uses a scalar index on the target table to find matching rows
/// without scanning the entire target table. The output includes all source columns plus a `_rowid`
/// column containing the row IDs of matching target rows (null for non-matches in LEFT joins).
///
/// Supported join types:
/// - Inner: Only output rows with matches (non-null _rowid)
/// - Left: Output all source rows, null _rowid for non-matches
/// - Right: Output all source rows + any unmatched target rows (requires index scan)
#[derive(Debug)]
pub struct ScalarIndexJoinExec {
    /// Input stream of source data
    input: Arc<dyn ExecutionPlan>,

    /// Dataset to query index from
    dataset: Arc<Dataset>,

    /// Name of column to join on
    join_column: String,

    /// Name of scalar index to use
    index_name: String,

    /// Type of join (Inner, Left, Right)
    join_type: datafusion::logical_expr::JoinType,

    /// Cached scalar index (loaded lazily)
    scalar_index: OnceLock<Arc<dyn ScalarIndex>>,

    /// Output schema with source columns + _rowid
    output_schema: SchemaRef,

    /// Execution properties
    properties: PlanProperties,
    // Note: metrics field removed as it was unused
}

impl DisplayAs for ScalarIndexJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ScalarIndexJoinExec: on=[{}], index=[{}], type=[{:?}]",
                    self.join_column, self.index_name, self.join_type
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "ScalarIndexJoinExec\non=[{}], index=[{}], type=[{:?}]",
                    self.join_column, self.index_name, self.join_type
                )
            }
        }
    }
}

impl ScalarIndexJoinExec {
    /// Create a new ScalarIndexJoinExec
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        join_column: String,
        index_name: String,
        join_type: datafusion::logical_expr::JoinType,
    ) -> Result<Self> {
        // Validate join type - we support Inner, Left, and Right joins
        if !matches!(
            join_type,
            datafusion::logical_expr::JoinType::Inner
                | datafusion::logical_expr::JoinType::Left
                | datafusion::logical_expr::JoinType::Right
        ) {
            return Err(Error::NotSupported {
                source: format!(
                    "ScalarIndexJoinExec only supports Inner, Left, and Right joins, got {:?}",
                    join_type
                )
                .into(),
                location: location!(),
            });
        }

        // Validate that the join column exists in the input schema
        let input_schema = input.schema();
        if input_schema.field_with_name(&join_column).is_err() {
            return Err(Error::InvalidInput {
                source: format!("Join column '{}' not found in input schema", join_column).into(),
                location: location!(),
            });
        }

        // Build output schema: source columns + _rowid
        let mut fields = input_schema.fields().iter().cloned().collect::<Vec<_>>();
        let rowid_nullable = matches!(
            join_type,
            datafusion::logical_expr::JoinType::Left | datafusion::logical_expr::JoinType::Right
        );
        fields.push(Arc::new(Field::new(
            ROW_ID,
            DataType::UInt64,
            rowid_nullable,
        )));
        let output_schema = Arc::new(Schema::new_with_metadata(
            fields,
            input_schema.metadata().clone(),
        ));

        let properties = PlanProperties::new(
            EquivalenceProperties::new(output_schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Ok(Self {
            input,
            dataset,
            join_column,
            index_name,
            join_type,
            scalar_index: OnceLock::new(),
            output_schema,
            properties,
        })
    }

    /// Load the scalar index (cached after first call)
    async fn get_or_load_index(&self) -> Result<&Arc<dyn ScalarIndex>> {
        if let Some(index) = self.scalar_index.get() {
            return Ok(index);
        }

        // First load the index metadata by name to get the UUID
        let idx = self
            .dataset
            .load_scalar_index(ScalarIndexCriteria::default().with_name(&self.index_name))
            .await?
            .ok_or_else(|| Error::Index {
                message: format!("Index with name {} does not exist", self.index_name),
                location: location!(),
            })?;

        // Then open the index using the UUID
        let index = self
            .dataset
            .open_scalar_index(
                &self.join_column,
                &idx.uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await?;

        // Cache it
        let _ = self.scalar_index.set(index);
        // Return the cached index (either the one we just set or one set by another thread)
        Ok(self.scalar_index.get().unwrap())
    }

    /// Perform index lookups for a batch of keys
    #[instrument(skip_all, fields(num_keys = keys.len()))]
    async fn lookup_keys(&self, keys: &dyn Array) -> Result<Vec<Option<u64>>> {
        let index = self.get_or_load_index().await?;
        let mut results = Vec::with_capacity(keys.len());

        for i in 0..keys.len() {
            if keys.is_null(i) {
                // Null keys never match
                results.push(None);
                continue;
            }

            // Convert array value to ScalarValue
            let scalar_value = ScalarValue::try_from_array(keys, i)?;

            // Query the index
            let query = SargableQuery::Equals(scalar_value);
            let search_result = index.search(&query, &NoOpMetricsCollector).await?;

            // Extract the first matching row ID (if any)
            let row_id = match search_result {
                SearchResult::Exact(row_ids) => {
                    if row_ids.is_empty() {
                        None
                    } else {
                        // For merge insert, we expect at most one match per key
                        // If there are multiple, take the first one
                        row_ids
                            .row_ids()
                            .and_then(|mut iter| iter.next())
                            .map(u64::from)
                    }
                }
                _ => {
                    return Err(Error::Internal {
                        message: "ScalarIndexJoinExec requires exact search results".into(),
                        location: location!(),
                    });
                }
            };

            results.push(row_id);
        }

        Ok(results)
    }

    /// Process a single batch from the input
    async fn process_batch(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let join_col_idx = batch.schema().index_of(&self.join_column)?;
        let join_col_array = batch.column(join_col_idx);

        // Perform index lookups
        let row_ids = self.lookup_keys(join_col_array.as_ref()).await?;

        // Build the _rowid column
        let mut rowid_builder = arrow_array::builder::UInt64Builder::new();
        for row_id in &row_ids {
            match row_id {
                Some(id) => rowid_builder.append_value(*id),
                None => rowid_builder.append_null(),
            }
        }
        let rowid_array = Arc::new(rowid_builder.finish()) as ArrayRef;

        // Apply join type filtering
        let (filtered_columns, filtered_rowid) = match self.join_type {
            datafusion::logical_expr::JoinType::Inner => {
                // Only keep rows with non-null _rowid
                let keep_mask: Vec<bool> = row_ids.iter().map(|id| id.is_some()).collect();
                if keep_mask.iter().all(|&x| !x) {
                    // No matches, return empty batch with correct schema
                    let empty_columns: Vec<ArrayRef> = batch
                        .columns()
                        .iter()
                        .map(|col| {
                            arrow_select::take::take(
                                col,
                                &UInt64Array::new(vec![].into(), None),
                                None,
                            )
                            .unwrap()
                        })
                        .collect();
                    let empty_rowid = Arc::new(UInt64Array::new(vec![].into(), None)) as ArrayRef;

                    let mut result_columns = empty_columns;
                    result_columns.push(empty_rowid);

                    return Ok(RecordBatch::try_new(
                        self.output_schema.clone(),
                        result_columns,
                    )?);
                }

                // Create indices for rows to keep
                let keep_indices: Vec<u32> = keep_mask
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &keep)| if keep { Some(i as u32) } else { None })
                    .collect();

                if keep_indices.is_empty() {
                    // No matches, return empty batch
                    let empty_columns: Vec<ArrayRef> = batch
                        .columns()
                        .iter()
                        .map(|col| {
                            arrow_select::take::take(
                                col,
                                &UInt64Array::new(vec![].into(), None),
                                None,
                            )
                            .unwrap()
                        })
                        .collect();
                    let empty_rowid = Arc::new(UInt64Array::new(vec![].into(), None)) as ArrayRef;

                    let mut result_columns = empty_columns;
                    result_columns.push(empty_rowid);

                    return Ok(RecordBatch::try_new(
                        self.output_schema.clone(),
                        result_columns,
                    )?);
                }

                let indices_array = UInt64Array::from(
                    keep_indices
                        .iter()
                        .map(|&i| Some(i as u64))
                        .collect::<Vec<_>>(),
                );
                let filtered_columns: Result<Vec<ArrayRef>> = batch
                    .columns()
                    .iter()
                    .map(|col| {
                        arrow_select::take::take(col, &indices_array, None).map_err(Error::from)
                    })
                    .collect();

                let filtered_rowid = arrow_select::take::take(&rowid_array, &indices_array, None)?;
                (filtered_columns?, filtered_rowid)
            }
            datafusion::logical_expr::JoinType::Left => {
                // Keep all rows, _rowid can be null
                (batch.columns().to_vec(), rowid_array)
            }
            datafusion::logical_expr::JoinType::Right => {
                // Right join: Keep all source rows, _rowid can be null for unmatched rows
                // This is effectively the same as Left join in our case since we're
                // joining source (input) with target (index), and Right join means
                // keep all rows from the right side (source/input)
                (batch.columns().to_vec(), rowid_array)
            }
            _ => {
                return Err(Error::NotSupported {
                    source: format!("Join type {:?} not supported", self.join_type).into(),
                    location: location!(),
                });
            }
        };

        // Build result batch
        let mut result_columns = filtered_columns;
        result_columns.push(filtered_rowid);

        Ok(RecordBatch::try_new(
            self.output_schema.clone(),
            result_columns,
        )?)
    }
}

#[async_trait]
impl ExecutionPlan for ScalarIndexJoinExec {
    fn name(&self) -> &str {
        "ScalarIndexJoinExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
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
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(datafusion::error::DataFusionError::Internal(
                "ScalarIndexJoinExec requires exactly one child".to_string(),
            ));
        }

        Ok(Arc::new(Self::try_new(
            children.into_iter().next().unwrap(),
            self.dataset.clone(),
            self.join_column.clone(),
            self.index_name.clone(),
            self.join_type,
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        let exec = self.clone();

        let output_stream = input_stream
            .map_err(Error::from)
            .and_then(move |batch| {
                let exec = exec.clone();
                async move { exec.process_batch(batch).await }
            })
            .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)));

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.output_schema.clone(),
            Box::pin(output_stream),
        )))
    }

    fn statistics(&self) -> datafusion::error::Result<Statistics> {
        // For now, return unknown statistics
        // In the future, we could use index statistics to provide better estimates
        Ok(Statistics {
            num_rows: Precision::Absent,
            total_byte_size: Precision::Absent,
            column_statistics: self
                .output_schema
                .fields()
                .iter()
                .map(|_| datafusion::common::ColumnStatistics::new_unknown())
                .collect(),
        })
    }
}

impl Clone for ScalarIndexJoinExec {
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            dataset: self.dataset.clone(),
            join_column: self.join_column.clone(),
            index_name: self.index_name.clone(),
            join_type: self.join_type,
            scalar_index: OnceLock::new(), // Don't clone the cached index
            output_schema: self.output_schema.clone(),
            properties: self.properties.clone(),
        }
    }
}

// Tests will be added later once the core implementation is working
