// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Float32Array, RecordBatch, UInt64Array};
use arrow_schema::SchemaRef;
use datafusion::common::Statistics;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use futures::stream::{self};
use futures::{StreamExt, TryStreamExt};
use lance_index::prefilter::{FilterLoader, PreFilter};
use lance_index::scalar::inverted::{flat_bm25_search_stream, InvertedIndex, FTS_SCHEMA};
use lance_index::scalar::FullTextSearchQuery;
use lance_table::format::Index;
use tracing::instrument;

use crate::index::prefilter::DatasetPreFilter;
use crate::{index::DatasetIndexInternalExt, Dataset};

use super::utils::{
    FilteredRowIdsToPrefilter, InstrumentedRecordBatchStreamAdapter, SelectionVectorToPrefilter,
};
use super::PreFilterSource;

/// An execution node that performs full text search
///
/// This node would perform full text search with inverted index on the dataset.
/// The result is a stream of record batches containing the row ids that match the search query,
/// and scores of the matched rows.
#[derive(Debug)]
pub struct FtsExec {
    dataset: Arc<Dataset>,
    // column -> (indices, unindexed input stream)
    indices: HashMap<String, Vec<Index>>,
    query: FullTextSearchQuery,
    /// Prefiltering input
    prefilter_source: PreFilterSource,
    properties: PlanProperties,

    metrics: ExecutionPlanMetricsSet,
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
        indices: HashMap<String, Vec<Index>>,
        query: FullTextSearchQuery,
        prefilter_source: PreFilterSource,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            dataset,
            indices,
            query,
            prefilter_source,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
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
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let plan = match children.len() {
            0 => Self {
                dataset: self.dataset.clone(),
                indices: self.indices.clone(),
                query: self.query.clone(),
                prefilter_source: PreFilterSource::None,
                properties: self.properties.clone(),
                metrics: ExecutionPlanMetricsSet::new(),
            },
            1 => {
                let src = children.pop().unwrap();
                let prefilter_source = match &self.prefilter_source {
                    PreFilterSource::FilteredRowIds(_) => {
                        PreFilterSource::FilteredRowIds(src.clone())
                    }
                    PreFilterSource::ScalarIndexQuery(_) => {
                        PreFilterSource::ScalarIndexQuery(src.clone())
                    }
                    PreFilterSource::None => {
                        return Err(DataFusionError::Internal(
                            "Unexpected prefilter source".to_string(),
                        ));
                    }
                };
                Self {
                    dataset: self.dataset.clone(),
                    indices: self.indices.clone(),
                    query: self.query.clone(),
                    prefilter_source,
                    properties: self.properties.clone(),
                    metrics: ExecutionPlanMetricsSet::new(),
                }
            }
            _ => {
                return Err(DataFusionError::Internal(
                    "Unexpected number of children".to_string(),
                ));
            }
        };
        Ok(Arc::new(plan))
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
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);

        let indices = self.indices.clone();
        let stream = stream::iter(indices)
            .map(move |(column, indices)| {
                let index_meta = indices[0].clone();
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

                    let index = ds.open_generic_index(&column, &uuid).await?;
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

                    let (row_ids, scores): (Vec<u64>, Vec<f32>) = results.into_iter().unzip();
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
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            schema,
            stream.boxed(),
            baseline_metrics,
        )) as SendableRecordBatchStream)
    }

    fn statistics(&self) -> DataFusionResult<datafusion::physical_plan::Statistics> {
        Ok(Statistics::new_unknown(&FTS_SCHEMA))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
}

/// An execution node that performs flat full text search
///
/// This node would perform flat full text search on unindexed rows.
/// The result is a stream of record batches containing the row ids that match the search query,
/// and scores of the matched rows.
#[derive(Debug)]
pub struct FlatFtsExec {
    dataset: Arc<Dataset>,
    // (column, indices, unindexed input stream)
    column_inputs: Vec<(String, Vec<Index>, Arc<dyn ExecutionPlan>)>,
    query: FullTextSearchQuery,
    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for FlatFtsExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "FlatFts: query={}", self.query.query)
            }
        }
    }
}

impl FlatFtsExec {
    pub fn new(
        dataset: Arc<Dataset>,
        column_inputs: Vec<(String, Vec<Index>, Arc<dyn ExecutionPlan>)>,
        query: FullTextSearchQuery,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            dataset,
            column_inputs,
            query,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl ExecutionPlan for FlatFtsExec {
    fn name(&self) -> &str {
        "FlatFtsExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        FTS_SCHEMA.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.column_inputs
            .iter()
            .map(|(_, _, input)| input)
            .collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if self.column_inputs.len() != children.len() {
            return Err(DataFusionError::Internal(
                "Unexpected number of children".to_string(),
            ));
        }

        let column_inputs = self
            .column_inputs
            .iter()
            .zip(children)
            .map(|((column, indices, _), input)| (column.clone(), indices.clone(), input))
            .collect();
        Ok(Arc::new(Self {
            dataset: self.dataset.clone(),
            column_inputs,
            query: self.query.clone(),
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    #[instrument(name = "flat_fts_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let query = self.query.clone();
        let ds = self.dataset.clone();
        let column_inputs = self.column_inputs.clone();
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);

        let stream = stream::iter(column_inputs)
            .map(move |(column, indices, input)| {
                let index_meta = indices[0].clone();
                let uuid = index_meta.uuid.to_string();
                let query = query.clone();
                let ds = ds.clone();
                let context = context.clone();

                async move {
                    let index = ds.open_generic_index(&column, &uuid).await?;
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

                    let unindexed_stream = input.execute(partition, context)?;
                    let unindexed_result_stream =
                        flat_bm25_search_stream(unindexed_stream, column, query, index);

                    Ok::<_, DataFusionError>(unindexed_result_stream)
                }
            })
            .buffered(self.column_inputs.len())
            .try_flatten();
        let schema = self.schema();
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            schema,
            stream.boxed(),
            baseline_metrics,
        )) as SendableRecordBatchStream)
    }

    fn statistics(&self) -> DataFusionResult<datafusion::physical_plan::Statistics> {
        Ok(Statistics::new_unknown(&FTS_SCHEMA))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
}
