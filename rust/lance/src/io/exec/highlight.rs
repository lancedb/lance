// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Final-hit highlighting. The child owns retrieval and snapshot-consistent source reads.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    RecordBatch,
    cast::AsArray,
    types::{UInt32Type, UInt64Type},
};
use arrow_schema::{Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::EquivalenceProperties;
use futures::TryStreamExt;
use lance_core::ROW_ADDR;
use lance_core::utils::{address::RowAddress, tokio::spawn_cpu};
use lance_index::scalar::inverted::DOC_INDEX_COL;
use lance_index::scalar::inverted::highlight::{
    HighlightDocument, PreparedHighlightQuery, highlights_array, highlights_field,
};
use roaring::RoaringBitmap;

use crate::index::scalar::inverted::ResolvedFtsField;

#[derive(Debug, Clone)]
pub(crate) struct HighlightSource {
    pub field: ResolvedFtsField,
    pub indexed_fragments: RoaringBitmap,
    pub stale_rows: HashMap<u32, RoaringBitmap>,
}

/// Appends `_highlights` to already selected, materialized search results.
#[derive(Debug)]
pub struct HighlightExec {
    input: Arc<dyn ExecutionPlan>,
    query: Arc<PreparedHighlightQuery>,
    sources: Arc<Vec<HighlightSource>>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl HighlightExec {
    pub(crate) fn new(
        input: Arc<dyn ExecutionPlan>,
        query: Arc<PreparedHighlightQuery>,
        sources: Arc<Vec<HighlightSource>>,
    ) -> Self {
        let mut fields = input.schema().fields().to_vec();
        fields.push(Arc::new(highlights_field()));
        let schema = Arc::new(Schema::new_with_metadata(
            fields,
            input.schema().metadata().clone(),
        ));
        let properties = Arc::new(
            input
                .properties()
                .as_ref()
                .clone()
                .with_eq_properties(EquivalenceProperties::new(schema.clone())),
        );
        Self {
            input,
            query,
            sources,
            schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for HighlightExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "HighlightExec: fields={}",
            self.sources
                .iter()
                .map(|source| source.field.canonical_path.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

impl ExecutionPlan for HighlightExec {
    fn name(&self) -> &str {
        "HighlightExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }
    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "HighlightExec requires one child".into(),
            ));
        }
        let input = children
            .pop()
            .ok_or_else(|| DataFusionError::Internal("HighlightExec child is missing".into()))?;
        Ok(Arc::new(Self::new(
            input,
            self.query.clone(),
            self.sources.clone(),
        )))
    }
    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let query = self.query.clone();
        let sources = self.sources.clone();
        let schema = self.schema.clone();
        let output_schema = schema.clone();
        let metrics = BaselineMetrics::new(&self.metrics, partition);
        let stream = self
            .input
            .execute(partition, context)?
            .and_then(move |batch| {
                let (query, sources, schema, metrics) = (
                    query.clone(),
                    sources.clone(),
                    schema.clone(),
                    metrics.clone(),
                );
                async move {
                    spawn_cpu(move || -> lance_core::Result<RecordBatch> {
                        let _timer = metrics.elapsed_compute().timer();
                        let addresses = batch
                            .column_by_name(ROW_ADDR)
                            .ok_or_else(|| {
                                lance_core::Error::internal(
                                    "highlight requires physical row addresses",
                                )
                            })?
                            .as_primitive::<UInt64Type>();
                        let mut rows = Vec::with_capacity(batch.num_rows());
                        for row in 0..batch.num_rows() {
                            let address = RowAddress::from(addresses.value(row));
                            let coordinate = batch
                                .column_by_name(DOC_INDEX_COL)
                                .map(|array| array.as_list::<i32>().value(row));
                            let doc_index = coordinate
                                .as_ref()
                                .map(|array| array.as_primitive::<UInt32Type>().values().as_ref())
                                .unwrap_or(&[]);
                            let documents = sources
                                .iter()
                                .map(|source| {
                                    Ok(HighlightDocument {
                                        field: source.field.canonical_path.clone(),
                                        values: source
                                            .field
                                            .highlight_values(&batch, row, doc_index)?,
                                        indexed: source
                                            .indexed_fragments
                                            .contains(address.fragment_id())
                                            && !source
                                                .stale_rows
                                                .get(&address.fragment_id())
                                                .is_some_and(|rows| {
                                                    rows.contains(address.row_offset())
                                                }),
                                    })
                                })
                                .collect::<lance_core::Result<Vec<_>>>()?;
                            rows.push(query.highlight(&documents).map_err(|error| {
                                let message = format!(
                                    "highlight row address {}: {error}",
                                    addresses.value(row)
                                );
                                match error {
                                    lance_core::Error::InvalidInput { .. } => {
                                        lance_core::Error::invalid_input(message)
                                    }
                                    _ => lance_core::Error::internal(message),
                                }
                            })?);
                        }
                        let mut columns = batch.columns().to_vec();
                        columns.push(highlights_array(&rows)?);
                        metrics.record_output(batch.num_rows());
                        Ok(RecordBatch::try_new(schema, columns)?)
                    })
                    .await
                    .map_err(DataFusionError::from)
                }
            });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            output_schema,
            stream,
        )))
    }
}
