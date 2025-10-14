// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Execution plan for adding version metadata columns to record batches

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, UInt64Array};
use arrow_schema::{Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::Statistics;
use futures::StreamExt;
use lance_core::{ROW_CREATED_AT_VERSION_FIELD, ROW_LAST_UPDATED_AT_VERSION_FIELD};
use lance_table::format::{Fragment, RowDatasetVersionSequence};

/// Add version metadata columns (`_row_last_updated_at_version` and `_row_created_at_version`)
/// to a stream of record batches.
///
/// This executor reads the version metadata from fragments and expands the RLE-encoded
/// version sequences into column arrays.
#[derive(Clone)]
pub struct AddVersionColumnsExec {
    input: Arc<dyn ExecutionPlan>,
    fragments: Arc<Vec<Fragment>>,
    /// Position in output schema for last_updated_at column
    last_updated_pos: usize,
    /// Position in output schema for created_at column
    created_at_pos: usize,
    output_schema: SchemaRef,
    properties: PlanProperties,
}

impl std::fmt::Debug for AddVersionColumnsExec {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("AddVersionColumnsExec")
            .field("input", &self.input)
            .field("last_updated_pos", &self.last_updated_pos)
            .field("created_at_pos", &self.created_at_pos)
            .field("output_schema", &self.output_schema)
            .field("properties", &self.properties)
            .finish()
    }
}

impl AddVersionColumnsExec {
    /// Create a new AddVersionColumnsExec node.
    ///
    /// This adds `_row_last_updated_at_version` and `_row_created_at_version` columns
    /// to the output stream.
    ///
    /// # Arguments
    /// * `input` - The input plan to add version columns to
    /// * `fragments` - Fragment metadata containing version sequences
    /// * `last_updated_pos` - Position in output schema for last_updated_at column
    /// * `created_at_pos` - Position in output schema for created_at column
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        fragments: Arc<Vec<Fragment>>,
        last_updated_pos: usize,
        created_at_pos: usize,
    ) -> Result<Self> {
        let input_schema = input.schema();

        let mut fields = input_schema.fields().iter().cloned().collect::<Vec<_>>();
        fields.insert(
            last_updated_pos,
            Arc::new(ROW_LAST_UPDATED_AT_VERSION_FIELD.clone()),
        );
        // Adjust created_at position if it comes after last_updated
        let adjusted_created_pos = if created_at_pos > last_updated_pos {
            created_at_pos + 1
        } else {
            created_at_pos
        };
        fields.insert(
            adjusted_created_pos,
            Arc::new(ROW_CREATED_AT_VERSION_FIELD.clone()),
        );

        let output_schema = Arc::new(Schema::new_with_metadata(
            fields,
            input_schema.metadata().clone(),
        ));

        let properties = input
            .properties()
            .clone()
            .with_eq_properties(EquivalenceProperties::new(output_schema.clone()));

        Ok(Self {
            input,
            fragments,
            last_updated_pos,
            created_at_pos: adjusted_created_pos,
            output_schema,
            properties,
        })
    }

    /// Build version column arrays from fragment metadata
    fn build_version_arrays(fragment: &Fragment, num_rows: usize) -> Result<(ArrayRef, ArrayRef)> {
        // Load last_updated_at sequence
        let last_updated_seq = if let Some(meta) = &fragment.last_updated_at_version_meta {
            meta.load_sequence().map_err(|e| {
                DataFusionError::Internal(format!(
                    "Failed to load last_updated_at version sequence: {}",
                    e
                ))
            })?
        } else {
            // Default: fragment predates version tracking, skip or use default
            return Ok((
                Arc::new(UInt64Array::new_null(num_rows)),
                Arc::new(UInt64Array::new_null(num_rows)),
            ));
        };

        // Load created_at sequence (default to version 1 if missing per user specification)
        let created_at_seq = if let Some(meta) = &fragment.created_at_version_meta {
            meta.load_sequence().map_err(|e| {
                DataFusionError::Internal(format!(
                    "Failed to load created_at version sequence: {}",
                    e
                ))
            })?
        } else {
            // Default: treat all rows as created at version 1
            RowDatasetVersionSequence::from_uniform_row_count(num_rows as u64, 1)
        };

        // Expand RLE sequences into arrays
        let last_updated_values: Vec<u64> = last_updated_seq.versions().collect();
        let created_at_values: Vec<u64> = created_at_seq.versions().collect();

        if last_updated_values.len() != num_rows || created_at_values.len() != num_rows {
            return Err(DataFusionError::Internal(format!(
                "Version sequence length mismatch: expected {} rows, got last_updated={}, created_at={}",
                num_rows,
                last_updated_values.len(),
                created_at_values.len()
            )));
        }

        Ok((
            Arc::new(UInt64Array::from(last_updated_values)),
            Arc::new(UInt64Array::from(created_at_values)),
        ))
    }
}

impl DisplayAs for AddVersionColumnsExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "AddVersionColumnsExec: last_updated_pos={}, created_at_pos={}",
                    self.last_updated_pos, self.created_at_pos
                )
            }
            DisplayFormatType::TreeRender => {
                write!(f, "AddVersionColumnsExec")
            }
        }
    }
}

impl ExecutionPlan for AddVersionColumnsExec {
    fn name(&self) -> &str {
        "AddVersionColumnsExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
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
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "AddVersionColumnsExec requires exactly one child".into(),
            ));
        }
        Ok(Arc::new(Self {
            input: children[0].clone(),
            fragments: self.fragments.clone(),
            last_updated_pos: self.last_updated_pos,
            created_at_pos: self.created_at_pos,
            output_schema: self.output_schema.clone(),
            properties: self.properties.clone(),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        let fragments = self.fragments.clone();
        let last_updated_pos = self.last_updated_pos;
        let created_at_pos = self.created_at_pos;
        let output_schema = self.output_schema.clone();
        let stream_schema = output_schema.clone();

        // Track which fragment we're currently reading from
        let mut current_fragment_idx = 0;
        let mut rows_read_in_fragment = 0;

        let stream = input_stream.map(move |batch| {
            let batch = batch?;
            let num_rows = batch.num_rows();

            // Determine which fragment this batch belongs to
            // For now, assume batches come in fragment order
            // TODO: This is a simplification - in reality we may need fragment ID in the batch
            if current_fragment_idx >= fragments.len() {
                return Err(DataFusionError::Internal(
                    "Batch read beyond available fragments".into(),
                ));
            }

            let fragment = &fragments[current_fragment_idx];
            let (last_updated_array, created_at_array) =
                Self::build_version_arrays(fragment, num_rows)?;

            // Insert columns at specified positions
            let mut columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns() + 2);
            let input_columns = batch.columns();

            // Build output with inserted columns
            let mut input_idx = 0;
            for out_idx in 0..(batch.num_columns() + 2) {
                if out_idx == last_updated_pos {
                    columns.push(last_updated_array.clone());
                } else if out_idx == created_at_pos {
                    columns.push(created_at_array.clone());
                } else {
                    columns.push(input_columns[input_idx].clone());
                    input_idx += 1;
                }
            }

            // Track progress through fragment
            rows_read_in_fragment += num_rows;
            if let Some(physical_rows) = fragment.physical_rows {
                if rows_read_in_fragment >= physical_rows {
                    current_fragment_idx += 1;
                    rows_read_in_fragment = 0;
                }
            }

            RecordBatch::try_new(stream_schema.clone(), columns).map_err(|e| {
                DataFusionError::Internal(format!("Failed to create output batch: {}", e))
            })
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            output_schema,
            stream,
        )))
    }

    fn statistics(&self) -> Result<Statistics> {
        #[allow(deprecated)]
        self.input.statistics()
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Statistics> {
        self.input.partition_statistics(partition)
    }
}
