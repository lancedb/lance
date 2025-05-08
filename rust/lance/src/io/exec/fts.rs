// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{Float32Type, UInt64Type};
use arrow_array::{Float32Array, RecordBatch, UInt64Array};
use datafusion::common::Statistics;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::{Distribution, EquivalenceProperties, Partitioning};
use futures::stream::{self};
use futures::{FutureExt, StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_core::ROW_ID;
use lance_index::prefilter::PreFilter;
use lance_index::scalar::inverted::query::{
    collect_tokens, BoostQuery, FtsSearchParams, MatchQuery, PhraseQuery,
};
use lance_index::scalar::inverted::{
    flat_bm25_search_stream, InvertedIndex, FTS_SCHEMA, SCORE_COL,
};
use lance_index::DatasetIndexExt;
use tracing::instrument;

use crate::{index::DatasetIndexInternalExt, Dataset};

use super::utils::{build_prefilter, IndexMetrics, InstrumentedRecordBatchStreamAdapter};
use super::PreFilterSource;

#[derive(Debug)]
pub struct MatchQueryExec {
    dataset: Arc<Dataset>,
    query: MatchQuery,
    params: FtsSearchParams,
    prefilter_source: PreFilterSource,

    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for MatchQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "MatchQuery: query={}, limit={:?}",
                    self.query.terms, self.params.limit
                )
            }
        }
    }
}

impl MatchQueryExec {
    pub fn new(
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
        prefilter_source: PreFilterSource,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Self {
            dataset,
            query,
            params,
            prefilter_source,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl ExecutionPlan for MatchQueryExec {
    fn name(&self) -> &str {
        "MatchQueryExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.prefilter_source {
            PreFilterSource::None => vec![],
            PreFilterSource::FilteredRowIds(src) => vec![&src],
            PreFilterSource::ScalarIndexQuery(src) => vec![&src],
        }
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // Prefilter inputs must be a single partition
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let plan = match children.len() {
            0 => {
                if !matches!(self.prefilter_source, PreFilterSource::None) {
                    return Err(DataFusionError::Internal(
                        "Unexpected prefilter source".to_string(),
                    ));
                }

                Self {
                    dataset: self.dataset.clone(),
                    query: self.query.clone(),
                    params: self.params.clone(),
                    prefilter_source: PreFilterSource::None,
                    properties: self.properties.clone(),
                    metrics: ExecutionPlanMetricsSet::new(),
                }
            }
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
                    query: self.query.clone(),
                    params: self.params.clone(),
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

    #[instrument(name = "match_query_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let query = self.query.clone();
        let params = self.params.clone();
        let ds = self.dataset.clone();
        let prefilter_source = self.prefilter_source.clone();
        let metrics = Arc::new(IndexMetrics::new(&self.metrics, partition));
        let column = query.column.ok_or(DataFusionError::Execution(format!(
            "column not set for MatchQuery {}",
            query.terms
        )))?;

        let stream = stream::once(async move {
            let index_meta = ds.load_scalar_index_for_column(&column).await?.ok_or(
                DataFusionError::Execution(format!("No index found for column {}", column,)),
            )?;
            let uuid = index_meta.uuid.to_string();
            let index = ds
                .open_generic_index(&column, &uuid, metrics.as_ref())
                .await?;

            let pre_filter = build_prefilter(
                context.clone(),
                partition,
                &prefilter_source,
                ds,
                &[index_meta],
            )?;

            let inverted_idx = index
                .as_any()
                .downcast_ref::<InvertedIndex>()
                .ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "Index for column {} is not an inverted index",
                        column,
                    ))
                })?;

            let is_fuzzy = matches!(query.fuzziness, Some(n) if n != 0);
            let params = params
                .with_fuzziness(query.fuzziness)
                .with_max_expansions(query.max_expansions);
            let mut tokenizer = match is_fuzzy {
                false => inverted_idx.tokenizer(),
                true => tantivy::tokenizer::TextAnalyzer::from(
                    tantivy::tokenizer::SimpleTokenizer::default(),
                ),
            };
            let tokens = collect_tokens(&query.terms, &mut tokenizer, None);

            pre_filter.wait_for_ready().await?;
            let (doc_ids, mut scores) = inverted_idx
                .bm25_search(
                    tokens.into(),
                    params.into(),
                    query.operator,
                    false,
                    pre_filter,
                    metrics,
                )
                .boxed()
                .await?;
            scores.iter_mut().for_each(|s| {
                *s *= query.boost;
            });

            let batch = RecordBatch::try_new(
                FTS_SCHEMA.clone(),
                vec![
                    Arc::new(UInt64Array::from(doc_ids)),
                    Arc::new(Float32Array::from(scores)),
                ],
            )?;
            Ok::<_, DataFusionError>(batch)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream.boxed(),
        )))
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

/// Calculates the FTS score for each row in the input
#[derive(Debug)]
pub struct FlatMatchQueryExec {
    dataset: Arc<Dataset>,
    query: MatchQuery,
    params: FtsSearchParams,
    unindexed_input: Arc<dyn ExecutionPlan>,

    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for FlatMatchQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "FlatMatchQuery: query={}", self.query.terms)
            }
        }
    }
}

impl FlatMatchQueryExec {
    pub fn new(
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
        unindexed_input: Arc<dyn ExecutionPlan>,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            dataset,
            query,
            params,
            unindexed_input,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl ExecutionPlan for FlatMatchQueryExec {
    fn name(&self) -> &str {
        "FlatMatchQueryExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.unindexed_input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "Unexpected number of children".to_string(),
            ));
        }
        let unindexed_input = children.pop().unwrap();
        Ok(Arc::new(Self {
            dataset: self.dataset.clone(),
            query: self.query.clone(),
            params: self.params.clone(),
            unindexed_input,
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    #[instrument(name = "flat_match_query_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let query = self.query.clone();
        let ds = self.dataset.clone();
        let metrics = Arc::new(IndexMetrics::new(&self.metrics, partition));
        let unindexed_input = self.unindexed_input.execute(partition, context)?;

        let column = query.column.ok_or(DataFusionError::Execution(format!(
            "column not set for MatchQuery {}",
            query.terms
        )))?;

        let stream = stream::once(async move {
            let index_meta = ds.load_scalar_index_for_column(&column).await?.ok_or(
                DataFusionError::Execution(format!("No index found for column {}", column,)),
            )?;
            let uuid = index_meta.uuid.to_string();
            let index = ds
                .open_generic_index(&column, &uuid, metrics.as_ref())
                .await?;
            let inverted_idx = index
                .as_any()
                .downcast_ref::<InvertedIndex>()
                .ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "Index for column {} is not an inverted index",
                        column,
                    ))
                })?;
            Ok::<_, DataFusionError>(flat_bm25_search_stream(
                unindexed_input,
                column,
                query.terms,
                inverted_idx,
            ))
        })
        .try_flatten_unordered(None);
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            self.schema(),
            stream.boxed(),
            partition,
            &self.metrics,
        )))
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

#[derive(Debug)]
pub struct PhraseQueryExec {
    dataset: Arc<Dataset>,
    query: PhraseQuery,
    params: FtsSearchParams,
    prefilter_source: PreFilterSource,
    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for PhraseQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "PhraseQuery: query={}", self.query.terms)
            }
        }
    }
}

impl PhraseQueryExec {
    pub fn new(
        dataset: Arc<Dataset>,
        query: PhraseQuery,
        params: FtsSearchParams,
        prefilter_source: PreFilterSource,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Self {
            dataset,
            query,
            params,
            prefilter_source,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl ExecutionPlan for PhraseQueryExec {
    fn name(&self) -> &str {
        "PhraseQueryExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.prefilter_source {
            PreFilterSource::None => vec![],
            PreFilterSource::FilteredRowIds(src) => vec![&src],
            PreFilterSource::ScalarIndexQuery(src) => vec![&src],
        }
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // Prefilter inputs must be a single partition
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let plan = match children.len() {
            0 => Self {
                dataset: self.dataset.clone(),
                query: self.query.clone(),
                params: self.params.clone(),
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
                    query: self.query.clone(),
                    params: self.params.clone(),
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

    #[instrument(name = "phrase_query_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let query = self.query.clone();
        let params = self.params.clone();
        let ds = self.dataset.clone();
        let prefilter_source = self.prefilter_source.clone();
        let metrics = Arc::new(IndexMetrics::new(&self.metrics, partition));
        let stream = stream::once(async move {
            let column = query.column.ok_or(DataFusionError::Execution(format!(
                "column not set for PhraseQuery {}",
                query.terms
            )))?;
            let index_meta = ds.load_scalar_index_for_column(&column).await?.ok_or(
                DataFusionError::Execution(format!("No index found for column {}", column,)),
            )?;
            let uuid = index_meta.uuid.to_string();
            let index = ds
                .open_generic_index(&column, &uuid, metrics.as_ref())
                .await?;

            let pre_filter = build_prefilter(
                context.clone(),
                partition,
                &prefilter_source,
                ds,
                &[index_meta],
            )?;

            let index = index
                .as_any()
                .downcast_ref::<InvertedIndex>()
                .ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "Index for column {} is not an inverted index",
                        column,
                    ))
                })?;

            let mut tokenizer = index.tokenizer();
            let tokens = collect_tokens(&query.terms, &mut tokenizer, None);

            pre_filter.wait_for_ready().await?;
            let (doc_ids, scores) = index
                .bm25_search(
                    tokens.into(),
                    params.into(),
                    lance_index::scalar::inverted::query::Operator::And,
                    true,
                    pre_filter,
                    metrics,
                )
                .boxed()
                .await?;
            let batch = RecordBatch::try_new(
                FTS_SCHEMA.clone(),
                vec![
                    Arc::new(UInt64Array::from(doc_ids)),
                    Arc::new(Float32Array::from(scores)),
                ],
            )?;
            Ok::<_, DataFusionError>(batch)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream.boxed(),
        )))
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

#[derive(Debug)]
pub struct BoostQueryExec {
    query: BoostQuery,
    params: FtsSearchParams,
    positive: Arc<dyn ExecutionPlan>,
    negative: Arc<dyn ExecutionPlan>,

    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for BoostQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "BoostQuery: negative_boost={}",
                    self.query.negative_boost
                )
            }
        }
    }
}

impl BoostQueryExec {
    pub fn new(
        query: BoostQuery,
        params: FtsSearchParams,
        positive: Arc<dyn ExecutionPlan>,
        negative: Arc<dyn ExecutionPlan>,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Self {
            query,
            params,
            positive,
            negative,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl ExecutionPlan for BoostQueryExec {
    fn name(&self) -> &str {
        "BoostQueryExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.positive, &self.negative]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // This node fully consumes and re-orders the input rows.
        // It must be run on a single partition.
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 2 {
            return Err(DataFusionError::Internal(
                "Unexpected number of children".to_string(),
            ));
        }

        let negative = children.pop().unwrap();
        let positive = children.pop().unwrap();
        Ok(Arc::new(Self {
            query: self.query.clone(),
            params: self.params.clone(),
            positive,
            negative,
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    #[instrument(name = "boost_query_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let query = self.query.clone();
        let params = self.params.clone();
        let positive = self.positive.execute(partition, context.clone())?;
        let negative = self.negative.execute(partition, context)?;
        let stream = stream::once(async move {
            let positive = positive.try_collect::<Vec<_>>().await?;
            let negative = negative.try_collect::<Vec<_>>().await?;

            let mut res = HashMap::new();
            for batch in positive {
                let doc_ids = batch[ROW_ID].as_primitive::<UInt64Type>().values();
                let scores = batch[SCORE_COL].as_primitive::<Float32Type>().values();

                for (doc_id, score) in std::iter::zip(doc_ids, scores) {
                    res.insert(*doc_id, *score);
                }
            }
            for batch in negative {
                let doc_ids = batch[ROW_ID].as_primitive::<UInt64Type>().values();
                let scores = batch[SCORE_COL].as_primitive::<Float32Type>().values();

                for (doc_id, neg_score) in std::iter::zip(doc_ids, scores) {
                    if let Some(score) = res.get_mut(doc_id) {
                        *score -= query.negative_boost * neg_score;
                    }
                }
            }

            let (doc_ids, scores): (Vec<_>, Vec<_>) = res
                .into_iter()
                .sorted_unstable_by(|(_, a), (_, b)| b.total_cmp(a))
                .take(params.limit.unwrap_or(usize::MAX))
                .unzip();

            let batch = RecordBatch::try_new(
                FTS_SCHEMA.clone(),
                vec![
                    Arc::new(UInt64Array::from(doc_ids)),
                    Arc::new(Float32Array::from(scores)),
                ],
            )?;
            Ok::<_, DataFusionError>(batch)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream.boxed(),
        )))
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

#[cfg(test)]
pub mod tests {
    use std::sync::Arc;

    use datafusion::{execution::TaskContext, physical_plan::ExecutionPlan};
    use lance_datafusion::datagen::DatafusionDatagenExt;
    use lance_datagen::{BatchCount, ByteCount, RowCount};
    use lance_index::scalar::inverted::query::{
        BoostQuery, FtsQuery, FtsSearchParams, MatchQuery, PhraseQuery,
    };

    use crate::{io::exec::PreFilterSource, utils::test::NoContextTestFixture};

    use super::{BoostQueryExec, FlatMatchQueryExec, MatchQueryExec, PhraseQueryExec};

    #[test]
    fn execute_without_context() {
        // These tests ensure we can create nodes and call execute without a tokio Runtime
        // being active.  This is a requirement for proper implementation of a Datafusion foreign
        // table provider.
        let fixture = NoContextTestFixture::new();
        let match_query = MatchQueryExec::new(
            Arc::new(fixture.dataset.clone()),
            MatchQuery::new("blah".to_string()).with_column(Some("text".to_string())),
            FtsSearchParams::default(),
            PreFilterSource::None,
        );
        match_query
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();

        let flat_input = lance_datagen::gen()
            .col(
                "text",
                lance_datagen::array::rand_utf8(ByteCount::from(10), false),
            )
            .into_df_exec(RowCount::from(15), BatchCount::from(2));

        let flat_match_query = FlatMatchQueryExec::new(
            Arc::new(fixture.dataset.clone()),
            MatchQuery::new("blah".to_string()).with_column(Some("text".to_string())),
            FtsSearchParams::default(),
            flat_input,
        );
        flat_match_query
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();

        let phrase_query = PhraseQueryExec::new(
            Arc::new(fixture.dataset.clone()),
            PhraseQuery::new("blah".to_string()),
            FtsSearchParams::default(),
            PreFilterSource::None,
        );
        phrase_query
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();

        let boost_input_one = MatchQueryExec::new(
            Arc::new(fixture.dataset.clone()),
            MatchQuery::new("blah".to_string()).with_column(Some("text".to_string())),
            FtsSearchParams::default(),
            PreFilterSource::None,
        );

        let boost_input_two = MatchQueryExec::new(
            Arc::new(fixture.dataset),
            MatchQuery::new("blah".to_string()).with_column(Some("text".to_string())),
            FtsSearchParams::default(),
            PreFilterSource::None,
        );

        let boost_query = BoostQueryExec::new(
            BoostQuery::new(
                FtsQuery::Match(
                    MatchQuery::new("blah".to_string()).with_column(Some("text".to_string())),
                ),
                FtsQuery::Match(
                    MatchQuery::new("test".to_string()).with_column(Some("text".to_string())),
                ),
                Some(1.0),
            ),
            FtsSearchParams::default(),
            Arc::new(boost_input_one),
            Arc::new(boost_input_two),
        );
        boost_query
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();
    }
}
