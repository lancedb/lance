// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! VectorIndexExec - IVF-PQ vector search with MVCC visibility.

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arrow_array::{cast::AsArray, FixedSizeListArray, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::common::stats::Precision;
use datafusion::error::Result as DataFusionResult;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream, Statistics,
};
use datafusion_physical_expr::EquivalenceProperties;
use futures::stream::{self, StreamExt};
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use snafu::location;

use super::super::builder::VectorQuery;
use crate::dataset::mem_wal::write::{BatchStore, IndexStore};

/// Distance column name in output.
pub const DISTANCE_COLUMN: &str = "_distance";

/// ExecutionPlan node that queries IVF-PQ vector index with MVCC visibility.
pub struct VectorIndexExec {
    batch_store: Arc<BatchStore>,
    indexes: Arc<IndexStore>,
    query: VectorQuery,
    max_visible_batch_position: usize,
    projection: Option<Vec<usize>>,
    output_schema: SchemaRef,
    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
    /// Whether to include _rowid column (row position) in output.
    with_row_id: bool,
}

impl Debug for VectorIndexExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("VectorIndexExec");
        debug
            .field("column", &self.query.column)
            .field("k", &self.query.k)
            .field("nprobes", &self.query.nprobes);
        if let Some(max_nprobes) = self.query.maximum_nprobes {
            debug.field("maximum_nprobes", &max_nprobes);
        }
        if let Some(ef) = self.query.ef {
            debug.field("ef", &ef);
        }
        if let Some(refine) = self.query.refine_factor {
            debug.field("refine_factor", &refine);
        }
        if let Some(metric) = &self.query.distance_type {
            debug.field("distance_type", metric);
        }
        debug.field(
            "max_visible_batch_position",
            &self.max_visible_batch_position,
        );
        debug.field("with_row_id", &self.with_row_id);
        debug.finish()
    }
}

impl VectorIndexExec {
    /// Create a new VectorIndexExec.
    ///
    /// # Arguments
    ///
    /// * `batch_store` - Lock-free batch store containing data
    /// * `indexes` - Index registry with IVF-PQ indexes
    /// * `query` - Vector query parameters
    /// * `max_visible_batch_position` - MVCC visibility sequence number
    /// * `projection` - Optional column indices to project
    /// * `base_schema` - Schema after projection (will add _distance column, and _rowid if with_row_id)
    /// * `with_row_id` - Whether to include _rowid column (row position)
    pub fn new(
        batch_store: Arc<BatchStore>,
        indexes: Arc<IndexStore>,
        query: VectorQuery,
        max_visible_batch_position: usize,
        projection: Option<Vec<usize>>,
        base_schema: SchemaRef,
        with_row_id: bool,
    ) -> Result<Self> {
        // Verify the index exists for this column
        let column = &query.column;
        if indexes.get_ivf_pq_by_column(column).is_none() {
            return Err(Error::invalid_input(
                format!("No IVF-PQ index found for column '{}'", column),
                location!(),
            ));
        }

        // Build output schema: base fields + _distance + optional _rowid
        let mut fields: Vec<Field> = base_schema
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        fields.push(Field::new(DISTANCE_COLUMN, DataType::Float32, false));
        if with_row_id {
            fields.push(Field::new(lance_core::ROW_ID, DataType::UInt64, true));
        }
        let output_schema = Arc::new(Schema::new(fields));

        let properties = PlanProperties::new(
            EquivalenceProperties::new(output_schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Ok(Self {
            batch_store,
            indexes,
            query,
            max_visible_batch_position,
            projection,
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            with_row_id,
        })
    }

    /// Compute the maximum visible row position based on max_visible_batch_position.
    ///
    /// Returns the last row position that is visible at the given max_visible_batch_position,
    /// or None if no batches are visible.
    fn compute_max_visible_row(&self) -> Option<u64> {
        let mut max_visible_row_exclusive: u64 = 0;
        let mut current_row: u64 = 0;

        for (batch_position, stored_batch) in self.batch_store.iter().enumerate() {
            let batch_end = current_row + stored_batch.num_rows as u64;
            if batch_position <= self.max_visible_batch_position {
                max_visible_row_exclusive = batch_end;
            }
            current_row = batch_end;
        }

        if max_visible_row_exclusive > 0 {
            Some(max_visible_row_exclusive - 1)
        } else {
            None
        }
    }

    /// Query the index and return matching rows with distances.
    fn query_index(&self) -> Vec<(f32, u64)> {
        let Some(index) = self.indexes.get_ivf_pq_by_column(&self.query.column) else {
            return vec![];
        };

        // Compute max visible row for MVCC filtering
        let Some(max_visible_row) = self.compute_max_visible_row() else {
            return vec![];
        };

        // Convert query vector to FixedSizeListArray
        let query_array = self.query.query_vector.as_ref();

        // Try to interpret as FixedSizeList
        let fsl = if let Some(fsl) = query_array.as_fixed_size_list_opt() {
            fsl.clone()
        } else {
            // If it's a primitive array, wrap it in a FixedSizeList (single row)
            let values = self.query.query_vector.clone();
            let dim = values.len() as i32;
            let field = Arc::new(Field::new("item", values.data_type().clone(), true));
            match FixedSizeListArray::try_new(field, dim, values, None) {
                Ok(arr) => arr,
                Err(_) => return vec![],
            }
        };

        // Determine effective k: if refine_factor is set, fetch more candidates
        let effective_k = if let Some(factor) = self.query.refine_factor {
            self.query.k * factor as usize
        } else {
            self.query.k
        };

        // Search the index with visibility filtering
        let mut results = index
            .search(&fsl, effective_k, self.query.nprobes, max_visible_row)
            .unwrap_or_default();

        // Apply distance bounds filtering if specified
        if self.query.distance_lower_bound.is_some() || self.query.distance_upper_bound.is_some() {
            results.retain(|&(dist, _)| {
                let above_lower = self.query.distance_lower_bound.is_none_or(|lb| dist >= lb);
                let below_upper = self.query.distance_upper_bound.is_none_or(|ub| dist < ub);
                above_lower && below_upper
            });
        }

        // If refine_factor is set, compute exact distances and re-sort
        if self.query.refine_factor.is_some() && !results.is_empty() {
            let distance_type = self
                .query
                .distance_type
                .unwrap_or_else(|| index.distance_type());
            results = self.refine_with_exact_distances(results, distance_type);
        }

        // Truncate to requested k after filtering and refinement
        results.truncate(self.query.k);

        results
    }

    /// Refine results by computing exact distances using original vectors.
    ///
    /// Fetches the original vector data for each result row, computes the
    /// exact distance using the specified distance type, and returns results
    /// sorted by exact distance.
    fn refine_with_exact_distances(
        &self,
        results: Vec<(f32, u64)>,
        distance_type: DistanceType,
    ) -> Vec<(f32, u64)> {
        if results.is_empty() {
            return results;
        }

        // Find the vector column index in the schema
        let vector_col_idx = self.batch_store.iter().next().and_then(|stored| {
            stored
                .data
                .schema()
                .column_with_name(&self.query.column)
                .map(|(idx, _)| idx)
        });

        let Some(col_idx) = vector_col_idx else {
            // Vector column not found, return original results
            return results;
        };

        // Build batch ranges for row position lookup
        let mut batch_ranges = Vec::new();
        let mut current_row = 0usize;
        for stored_batch in self.batch_store.iter() {
            let batch_start = current_row;
            let batch_end = current_row + stored_batch.num_rows;
            batch_ranges.push((batch_start, batch_end));
            current_row = batch_end;
        }

        // Group rows by batch to minimize data fetching
        let mut batch_to_rows: std::collections::HashMap<usize, Vec<(usize, usize, u64)>> =
            std::collections::HashMap::new();

        for (result_idx, &(_, pos)) in results.iter().enumerate() {
            let pos_usize = pos as usize;
            for (batch_id, &(start, end)) in batch_ranges.iter().enumerate() {
                if pos_usize >= start && pos_usize < end {
                    batch_to_rows.entry(batch_id).or_default().push((
                        result_idx,
                        pos_usize - start,
                        pos,
                    ));
                    break;
                }
            }
        }

        // Compute exact distances
        let distance_func = distance_type.arrow_batch_func();
        let query_vec = &self.query.query_vector;

        let mut refined_results: Vec<(f32, u64)> = Vec::with_capacity(results.len());

        for (batch_id, rows) in batch_to_rows {
            let Some(stored) = self.batch_store.get(batch_id) else {
                // If batch not found, keep approximate distances for these rows
                for &(result_idx, _, pos) in &rows {
                    refined_results.push((results[result_idx].0, pos));
                }
                continue;
            };

            let vector_col = stored.data.column(col_idx);

            // For each row in this batch, compute exact distance
            for &(_, row_in_batch, pos) in &rows {
                // Extract the single vector at this row position
                let vector_arr = vector_col.as_fixed_size_list();
                let single_vector = vector_arr.value(row_in_batch);

                // Create a single-element FixedSizeList for distance computation
                let dim = vector_arr.value_length();
                let field = Arc::new(Field::new("item", single_vector.data_type().clone(), true));

                if let Ok(single_fsl) =
                    FixedSizeListArray::try_new(field, dim, single_vector.clone(), None)
                {
                    // Compute exact distance
                    if let Ok(distances) = distance_func(query_vec.as_ref(), &single_fsl) {
                        let exact_distance = distances.value(0);
                        refined_results.push((exact_distance, pos));
                        continue;
                    }
                }

                // Fallback: use approximate distance if exact computation fails
                if let Some((approx_dist, _)) = results.iter().find(|&&(_, p)| p == pos) {
                    refined_results.push((*approx_dist, pos));
                }
            }
        }

        // Sort by exact distance
        refined_results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        refined_results
    }

    /// Materialize rows from batch store with distance column.
    fn materialize_rows(&self, results: &[(f32, u64)]) -> DataFusionResult<Vec<RecordBatch>> {
        if results.is_empty() {
            return Ok(vec![]);
        }

        // Build batch ranges
        let mut batch_ranges = Vec::new();
        let mut current_row = 0usize;

        for stored_batch in self.batch_store.iter() {
            let batch_start = current_row;
            let batch_end = current_row + stored_batch.num_rows;
            batch_ranges.push((batch_start, batch_end));
            current_row = batch_end;
        }

        // Group rows by batch, tracking (row_in_batch, distance, row_position)
        let mut batches_data: std::collections::HashMap<usize, Vec<(usize, f32, u64)>> =
            std::collections::HashMap::new();

        for &(distance, pos) in results {
            let pos_usize = pos as usize;
            for (batch_id, &(start, end)) in batch_ranges.iter().enumerate() {
                if pos_usize >= start && pos_usize < end {
                    batches_data.entry(batch_id).or_default().push((
                        pos_usize - start,
                        distance,
                        pos,
                    ));
                    break;
                }
            }
        }

        let mut all_batches = Vec::new();

        for (batch_id, rows_with_dist) in batches_data {
            if let Some(stored) = self.batch_store.get(batch_id) {
                let rows: Vec<u32> = rows_with_dist.iter().map(|&(r, _, _)| r as u32).collect();
                let distances: Vec<f32> = rows_with_dist.iter().map(|&(_, d, _)| d).collect();
                let row_positions: Vec<u64> =
                    rows_with_dist.iter().map(|&(_, _, pos)| pos).collect();

                let indices = arrow_array::UInt32Array::from(rows);

                let mut columns: Vec<Arc<dyn arrow_array::Array>> = stored
                    .data
                    .columns()
                    .iter()
                    .map(|col| arrow_select::take::take(col.as_ref(), &indices, None).unwrap())
                    .collect();

                // Add distance column
                columns.push(Arc::new(Float32Array::from(distances)));

                // Apply projection if needed (excluding distance column which is always included)
                let mut final_columns = if let Some(ref proj_indices) = self.projection {
                    let mut projected: Vec<_> =
                        proj_indices.iter().map(|&i| columns[i].clone()).collect();
                    // Always include distance as last column
                    projected.push(columns.last().unwrap().clone());
                    projected
                } else {
                    columns
                };

                // Add _rowid column if requested
                if self.with_row_id {
                    final_columns.push(Arc::new(UInt64Array::from(row_positions)));
                }

                let batch = RecordBatch::try_new(self.output_schema.clone(), final_columns)?;
                all_batches.push(batch);
            }
        }

        Ok(all_batches)
    }
}

impl DisplayAs for VectorIndexExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "VectorIndexExec: column={}, k={}, nprobes={}",
                    self.query.column, self.query.k, self.query.nprobes
                )?;
                if let Some(ef) = self.query.ef {
                    write!(f, ", ef={}", ef)?;
                }
                if let Some(refine) = self.query.refine_factor {
                    write!(f, ", refine={}", refine)?;
                }
                write!(f, ", with_row_id={}", self.with_row_id)
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "VectorIndexExec\ncolumn={}\nk={}\nnprobes={}",
                    self.query.column, self.query.k, self.query.nprobes
                )?;
                if let Some(ef) = self.query.ef {
                    write!(f, "\nef={}", ef)?;
                }
                if let Some(refine) = self.query.refine_factor {
                    write!(f, "\nrefine={}", refine)?;
                }
                write!(f, "\nwith_row_id={}", self.with_row_id)
            }
        }
    }
}

impl ExecutionPlan for VectorIndexExec {
    fn name(&self) -> &str {
        "VectorIndexExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(datafusion::error::DataFusionError::Internal(
                "VectorIndexExec does not have children".to_string(),
            ));
        }
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        // Query the index (visibility filtering happens inside search)
        let results = self.query_index();

        // Materialize the rows
        let batches = self.materialize_rows(&results)?;

        let stream = stream::iter(batches.into_iter().map(Ok)).boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.output_schema.clone(),
            stream,
        )))
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> DataFusionResult<Statistics> {
        Ok(Statistics {
            num_rows: Precision::Exact(self.query.k),
            total_byte_size: Precision::Absent,
            column_statistics: vec![],
        })
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        true // Vector search naturally supports limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full tests for VectorIndexExec require setting up IVF-PQ index
    // with trained centroids and codebook, which is complex.
    // Basic structure tests are included here.

    #[test]
    fn test_distance_column_name() {
        assert_eq!(DISTANCE_COLUMN, "_distance");
    }
}
