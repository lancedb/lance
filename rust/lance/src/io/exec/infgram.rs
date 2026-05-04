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

use arrow_array::{UInt32Array, ListArray, RecordBatch, UInt64Array};
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

/// Column name for the byte positions of matches within each document.
pub const POSITIONS_COL: &str = "_positions";

/// Schema for infgram search results: (ROW_ID, _count, _positions).
pub static INFGRAM_SCHEMA: std::sync::LazyLock<SchemaRef> = std::sync::LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new(COUNT_COL, DataType::UInt32, true),
        Field::new(
            POSITIONS_COL,
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, false))),
            true,
        ),
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
        let query_clauses = self.query.clauses.clone();
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
                let empty_positions = {
                    let values = Arc::new(UInt64Array::from(Vec::<u64>::new()));
                    let offsets = arrow_buffer::OffsetBuffer::new(
                        arrow_buffer::ScalarBuffer::from(vec![0i32]),
                    );
                    let field = Arc::new(Field::new("item", DataType::UInt64, false));
                    ListArray::new(field, offsets, values, None)
                };
                let batch = RecordBatch::try_new(
                    INFGRAM_SCHEMA.clone(),
                    vec![
                        Arc::new(UInt64Array::from(Vec::<u64>::new())),
                        Arc::new(UInt32Array::from(Vec::<u32>::new())),
                        Arc::new(empty_positions),
                    ],
                )?;
                return Ok::<_, DataFusionError>(batch);
            }

            // Open and query each index version, collecting scored results with positions
            let max_results = limit.unwrap_or(usize::MAX);
            let mut all_scored: Vec<(u64, u32, Vec<u64>)> = Vec::new();

            // Check if we have boolean clauses or parse the query for AND/OR
            let parsed_clauses = query_clauses.or_else(|| {
                InfgramSearchQuery::parse_boolean_query(&query_text)
            });

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

                if let Some(ref clauses) = parsed_clauses {
                    // Boolean query: convert string clauses to byte clauses
                    let byte_clauses: Vec<Vec<Vec<u8>>> = clauses
                        .iter()
                        .map(|or_group| {
                            or_group
                                .iter()
                                .map(|term| {
                                    if sa_index.case_insensitive() {
                                        term.to_lowercase().into_bytes()
                                    } else {
                                        term.as_bytes().to_vec()
                                    }
                                })
                                .collect()
                        })
                        .collect();

                    let scored = sa_index
                        .search_boolean(&byte_clauses, max_results)
                        .await
                        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
                    all_scored.extend(scored);
                } else {
                    // Single-term query (backward compatible)
                    let query_bytes = if let Some(ref tokens) = query_tokens {
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

                    let scored = sa_index
                        .search_rows_scored(&query_bytes, max_results)
                        .await
                        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
                    all_scored.extend(scored);
                }
            }

            // If we collected from multiple index versions, re-sort and take top-K
            all_scored.sort_by(|a, b| b.1.cmp(&a.1));
            all_scored.dedup_by_key(|item| item.0); // dedup by row_id, keep highest score (first)
            if let Some(lim) = limit {
                all_scored.truncate(lim);
            }

            // Unzip into separate arrays
            let mut all_row_ids = Vec::with_capacity(all_scored.len());
            let mut counts = Vec::with_capacity(all_scored.len());
            let mut all_positions: Vec<Vec<u64>> = Vec::with_capacity(all_scored.len());
            for (row_id, count, positions) in all_scored {
                all_row_ids.push(row_id);
                counts.push(count);
                all_positions.push(positions);
            }

            // Build _positions ListArray
            let positions_array = {
                let mut offsets = Vec::with_capacity(all_positions.len() + 1);
                let mut values = Vec::new();
                offsets.push(0i32);
                for positions in &all_positions {
                    values.extend_from_slice(positions);
                    offsets.push(values.len() as i32);
                }
                let values_array = Arc::new(UInt64Array::from(values));
                let offsets_buffer = arrow_buffer::OffsetBuffer::new(
                    arrow_buffer::ScalarBuffer::from(offsets),
                );
                let field = Arc::new(Field::new("item", DataType::UInt64, false));
                ListArray::new(field, offsets_buffer, values_array, None)
            };

            let batch = RecordBatch::try_new(
                INFGRAM_SCHEMA.clone(),
                vec![
                    Arc::new(UInt64Array::from(all_row_ids)),
                    Arc::new(UInt32Array::from(counts)),
                    Arc::new(positions_array),
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
