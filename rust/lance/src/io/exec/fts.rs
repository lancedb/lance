// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{Schema, SchemaRef};
use datafusion::common::Statistics;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionMode, ExecutionPlan, PlanProperties,
};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use futures::stream::{self};
use futures::StreamExt;
use lance_core::ROW_ID_FIELD;
use lance_index::prefilter::{FilterLoader, PreFilter};
use lance_index::scalar::inverted::{InvertedIndex, SCORE_FIELD};
use lance_index::scalar::FullTextSearchQuery;
use lance_table::format::Index;
use tracing::instrument;

use crate::index::prefilter::DatasetPreFilter;
use crate::{index::DatasetIndexInternalExt, Dataset};

use super::utils::{FilteredRowIdsToPrefilter, SelectionVectorToPrefilter};
use super::PreFilterSource;

lazy_static::lazy_static! {
    pub static ref FTS_SCHEMA: SchemaRef = Arc::new(Schema::new(vec![ROW_ID_FIELD.clone(), SCORE_FIELD.clone()]));
}

/// An execution node that performs full text search
///
/// This node would perform full text search with inverted index on the dataset.
/// The result is a stream of record batches containing the row ids that match the search query,
/// and scores of the matched rows.
#[derive(Debug)]
pub struct FtsExec {
    dataset: Arc<Dataset>,
    indices: Vec<Index>,
    query: FullTextSearchQuery,
    /// Prefiltering input
    prefilter_source: PreFilterSource,
    properties: PlanProperties,
}

impl DisplayAs for FtsExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "Fts: query={}", self.query.query)
            }
        }
    }
}

impl FtsExec {
    pub fn new(
        dataset: Arc<Dataset>,
        indices: Vec<Index>,
        query: FullTextSearchQuery,
        prefilter_source: PreFilterSource,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            ExecutionMode::Bounded,
        );
        Self {
            dataset,
            indices,
            query,
            prefilter_source,
            properties,
        }
    }
}

impl ExecutionPlan for FtsExec {
    fn name(&self) -> &str {
        "FtsExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        FTS_SCHEMA.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.prefilter_source {
            PreFilterSource::None => vec![],
            PreFilterSource::FilteredRowIds(src) => vec![&src],
            PreFilterSource::ScalarIndexQuery(src) => vec![&src],
        }
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        todo!()
    }

    #[instrument(name = "fts_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let query = self.query.clone();
        let ds = self.dataset.clone();
        let prefilter_source = self.prefilter_source.clone();

        let stream = stream::iter(self.indices.clone())
            .map(move |index_meta| {
                let uuid = index_meta.uuid.to_string();
                let query = query.clone();
                let ds = ds.clone();
                let context = context.clone();
                let prefilter_source = prefilter_source.clone();

                async move {
                    let prefilter_loader = match &prefilter_source {
                        PreFilterSource::FilteredRowIds(src_node) => {
                            let stream = src_node.execute(partition, context.clone())?;
                            Some(Box::new(FilteredRowIdsToPrefilter(stream))
                                as Box<dyn FilterLoader>)
                        }
                        PreFilterSource::ScalarIndexQuery(src_node) => {
                            let stream = src_node.execute(partition, context.clone())?;
                            Some(Box::new(SelectionVectorToPrefilter(stream))
                                as Box<dyn FilterLoader>)
                        }
                        PreFilterSource::None => None,
                    };
                    let pre_filter = Arc::new(DatasetPreFilter::new(
                        ds.clone(),
                        &[index_meta],
                        prefilter_loader,
                    ));

                    let index = ds.open_generic_index(&query.columns[0], &uuid).await?;

                    let index =
                        index
                            .as_any()
                            .downcast_ref::<InvertedIndex>()
                            .ok_or_else(|| {
                                DataFusionError::Execution(format!(
                                    "Index {} is not an inverted index",
                                    uuid,
                                ))
                            })?;

                    pre_filter.wait_for_ready().await?;
                    let results = index.full_text_search(&query, pre_filter).await?;

                    let mut row_ids = Vec::new();
                    let mut scores = Vec::new();
                    for (row_id, score) in results {
                        row_ids.push(row_id);
                        scores.push(score);
                    }
                    let batch = RecordBatch::try_new(
                        FTS_SCHEMA.clone(),
                        vec![
                            Arc::new(UInt64Array::from(row_ids)),
                            Arc::new(Float32Array::from(scores)),
                        ],
                    )?;

                    Ok::<_, DataFusionError>(batch)
                }
            })
            .buffered(self.indices.len());
        let schema = self.schema();
        Ok(
            Box::pin(RecordBatchStreamAdapter::new(schema, stream.boxed()))
                as SendableRecordBatchStream,
        )
    }

    fn statistics(&self) -> DataFusionResult<datafusion::physical_plan::Statistics> {
        Ok(Statistics::new_unknown(&FTS_SCHEMA))
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
}
