// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Execution node for infini-gram (suffix array) search.
//!
//! Follows the same pattern as [`super::fts::MatchQueryExec`]: loads the
//! suffix array index, resolves matching byte positions to row IDs via
//! `doc_offsets`, and emits a two-column `RecordBatch` (row_id, _count).

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::{Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::common::DataFusionError;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use futures::stream;
use futures::StreamExt;
use lance_core::ROW_ID_FIELD;
use lance_core::utils::tracing::StreamTracingExt;
use lance_index::scalar::InfgramSearchQuery;
use lance_index::IndexCriteria;
use tracing::instrument;

use crate::dataset::Dataset;
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
use lance_index::metrics::NoOpMetricsCollector;

/// Column name for the occurrence count in infgram results.
pub const COUNT_COL: &str = "_count";

/// Schema for infgram search results: (ROW_ID, _count).
pub static INFGRAM_SCHEMA: std::sync::LazyLock<SchemaRef> = std::sync::LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new(COUNT_COL, DataType::Float32, true),
    ]))
});

/// Execution plan node for infini-gram search.
///
/// Loads the suffix array index for the target column, searches for the
/// query pattern, resolves byte positions to row IDs, and emits them as
/// a RecordBatch with row_id and _count columns.
#[derive(Debug)]
pub struct InfgramSearchExec {
    dataset: Arc<Dataset>,
    query: InfgramSearchQuery,
    properties: Arc<PlanProperties>,
}

impl InfgramSearchExec {
    pub fn new(dataset: Arc<Dataset>, query: InfgramSearchQuery) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(INFGRAM_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            dataset,
            query,
            properties,
        }
    }
}

impl DisplayAs for InfgramSearchExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "InfgramSearch: query={:?}, column={:?}, limit={:?}",
            self.query.query, self.query.column, self.query.limit
        )
    }
}

impl ExecutionPlan for InfgramSearchExec {
    fn name(&self) -> &str {
        "InfgramSearchExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        INFGRAM_SCHEMA.clone()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    #[instrument(name = "infgram_search_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        _partition: usize,
        _context: Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::common::Result<SendableRecordBatchStream> {
        let ds = self.dataset.clone();
        let query_text = self.query.query.clone();
        let query_tokens = self.query.query_tokens.clone();
        let column = self.query.column.clone();
        let limit = self.query.limit;

        let stream = stream::once(async move {
            // Find the suffix array index and determine the column name
            let (column, index_meta) = if let Some(col) = column {
                let meta = ds
                    .load_scalar_index(
                        IndexCriteria::default()
                            .for_column(&col)
                            .supports_suffix_array(),
                    )
                    .await
                    .map_err(|e| DataFusionError::Execution(format!(
                        "Failed to load SA index for column '{col}': {e}"
                    )))?
                    .ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "No suffix array index found for column '{col}'"
                        ))
                    })?;
                (col, meta)
            } else {
                // Find any suffix-array index and resolve its column
                let meta = ds
                    .load_scalar_index(
                        IndexCriteria::default().supports_suffix_array(),
                    )
                    .await
                    .map_err(|e| DataFusionError::Execution(format!(
                        "Failed to load SA index: {e}"
                    )))?
                    .ok_or_else(|| {
                        DataFusionError::Execution(
                            "No suffix array index found in dataset".to_string(),
                        )
                    })?;

                // Resolve field ID to column name
                let field_id = *meta.fields.first().ok_or_else(|| {
                    DataFusionError::Execution("SA index has no fields".to_string())
                })?;
                let field = ds.schema().field_by_id(field_id).ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "Field id {field_id} from SA index not found in schema"
                    ))
                })?;
                (field.name.clone(), meta)
            };

            // Load all segments for this logical index
            let indices = ds
                .load_indices_by_name(&index_meta.name)
                .await
                .map_err(|e| DataFusionError::Execution(format!(
                    "Failed to load SA index segments: {e}"
                )))?;

            if indices.is_empty() {
                let batch = RecordBatch::try_new(
                    INFGRAM_SCHEMA.clone(),
                    vec![
                        Arc::new(UInt64Array::from(Vec::<u64>::new())),
                        Arc::new(Float32Array::from(Vec::<f32>::new())),
                    ],
                )?;
                return Ok::<_, DataFusionError>(batch);
            }

            // Open and query each segment
            let mut all_row_ids = Vec::new();
            let max_results = limit.unwrap_or(usize::MAX);

            for meta in &indices {
                let uuid = meta.uuid.to_string();
                let index = ds
                    .open_generic_index(&column, &uuid, &NoOpMetricsCollector)
                    .await
                    .map_err(|e| {
                        DataFusionError::Execution(format!(
                            "Failed to open SA index segment {uuid}: {e}"
                        ))
                    })?;

                let sa_index = index
                    .as_any()
                    .downcast_ref::<lance_index::scalar::suffix_array::SuffixArrayIndex>()
                    .ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "Index segment {uuid} for column '{column}' is not a suffix array index"
                        ))
                    })?;

                // Build query bytes: either from token IDs or text string
                let query_bytes = if let Some(ref tokens) = query_tokens {
                    // Token-level query: serialize token IDs to bytes
                    let tw = sa_index.token_width() as usize;
                    let mut bytes = Vec::with_capacity(tokens.len() * tw);
                    for &tok in tokens {
                        match tw {
                            2 => bytes.extend_from_slice(&(tok as i16).to_le_bytes()),
                            4 => bytes.extend_from_slice(&(tok as i32).to_le_bytes()),
                            _ => bytes.extend_from_slice(&(tok as u8).to_le_bytes()),
                        }
                    }
                    bytes
                } else if sa_index.case_insensitive() {
                    query_text.to_lowercase().into_bytes()
                } else {
                    query_text.as_bytes().to_vec()
                };
                let remaining = max_results.saturating_sub(all_row_ids.len());
                if remaining == 0 {
                    break;
                }

                let row_ids = sa_index.search_rows(&query_bytes, remaining);
                all_row_ids.extend(row_ids);
            }

            // Deduplicate and sort
            all_row_ids.sort_unstable();
            all_row_ids.dedup();

            // Truncate to limit
            if let Some(lim) = limit {
                all_row_ids.truncate(lim);
            }

            // Score is 1.0 for matching rows (presence-based).
            // Future: compute per-document frequency from the suffix array.
            let num_results = all_row_ids.len();
            let scores = vec![1.0f32; num_results];

            let batch = RecordBatch::try_new(
                INFGRAM_SCHEMA.clone(),
                vec![
                    Arc::new(UInt64Array::from(all_row_ids)),
                    Arc::new(Float32Array::from(scores)),
                ],
            )?;
            Ok::<_, DataFusionError>(batch)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream.stream_in_current_span().boxed(),
        )))
    }
}
