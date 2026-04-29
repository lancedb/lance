// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{AsArray, BooleanBuilder};
use arrow::datatypes::{Float32Type, UInt64Type};
use arrow_array::{Array, BooleanArray, Float32Array, OffsetSizeTrait, RecordBatch, UInt64Array};
use arrow_schema::DataType;
use datafusion::common::Statistics;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::{Distribution, EquivalenceProperties, Partitioning};
use datafusion_physical_plan::metrics::{BaselineMetrics, Count};
use futures::future::try_join_all;
use futures::stream::{self};
use futures::{Stream, StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_core::{
    Error, ROW_ID, Result,
    utils::{tokio::get_num_compute_intensive_cpus, tracing::StreamTracingExt},
};
use lance_datafusion::utils::{ExecutionPlanMetricsSetExt, MetricsExt, PARTITIONS_SEARCHED_METRIC};
use lance_table::format::IndexMetadata;

use super::PreFilterSource;
use super::utils::{IndexMetrics, InstrumentedRecordBatchStreamAdapter, build_prefilter};
use crate::index::scalar::inverted::{load_segment_details, load_segments};
use crate::{Dataset, index::DatasetIndexInternalExt};
use lance_index::metrics::MetricsCollector;
use lance_index::scalar::inverted::builder::ScoredDoc;
use lance_index::scalar::inverted::builder::document_input;
use lance_index::scalar::inverted::document_tokenizer::{DocType, JsonTokenizer, LanceTokenizer};
use lance_index::scalar::inverted::query::{
    BoostQuery, FtsSearchParams, MatchQuery, PhraseQuery, Tokens, collect_query_tokens,
    has_query_token,
};
use lance_index::scalar::inverted::tokenizer::document_tokenizer::TextTokenizer;
use lance_index::scalar::inverted::{
    FTS_SCHEMA, InvertedIndex, MemBM25Scorer, SCORE_COL, flat_bm25_search_stream,
};
use lance_index::{prefilter::PreFilter, scalar::inverted::query::BooleanQuery};
use lance_tokenizer::{SimpleTokenizer, TextAnalyzer};
use tracing::instrument;

/// Open one FTS segment as an [`InvertedIndex`].
async fn open_fts_segment(
    dataset: &Dataset,
    column: &str,
    segment: &IndexMetadata,
    metrics: &IndexMetrics,
) -> Result<Arc<InvertedIndex>> {
    let uuid = segment.uuid.to_string();
    let index = dataset.open_generic_index(column, &uuid, metrics).await?;
    let inverted = index
        .as_any()
        .downcast_ref::<InvertedIndex>()
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "Index for column {} and segment {} is not an inverted index",
                column, uuid
            ))
        })?;
    Ok(Arc::new(inverted.clone()))
}

/// Open all committed FTS segments for a column.
///
/// Exact multi-segment BM25 still needs every segment's local corpus statistics, so the
/// current correctness-first path opens each committed segment before scoring.
async fn open_fts_segments(
    dataset: &Dataset,
    column: &str,
    segments: &[IndexMetadata],
    metrics: &IndexMetrics,
) -> Result<Vec<Arc<InvertedIndex>>> {
    try_join_all(
        segments
            .iter()
            .map(|segment| open_fts_segment(dataset, column, segment, metrics)),
    )
    .await
}

/// Collect the unique terms needed to build a shared BM25 scorer.
///
/// The scorer only needs corpus-level document frequencies, so we keep a deduplicated
/// term list here instead of constructing a full `Tokens` object with positions.
fn scorer_terms(
    indices: &[Arc<InvertedIndex>],
    query_tokens: &Tokens,
    params: &FtsSearchParams,
) -> Result<Vec<String>> {
    let mut terms = Vec::new();
    let mut seen = HashSet::new();

    if !matches!(params.fuzziness, Some(n) if n != 0) {
        for token in query_tokens {
            if seen.insert(token.to_string()) {
                terms.push(token.to_string());
            }
        }
        return Ok(terms);
    }

    for index in indices {
        let expanded = index.expand_fuzzy_tokens(query_tokens, params)?;
        for idx in 0..expanded.len() {
            let token = expanded.get_token(idx);
            if seen.insert(token.to_string()) {
                terms.push(token.to_string());
            }
        }
    }
    Ok(terms)
}

/// Build a shared BM25 scorer for a set of committed FTS segments.
fn build_global_bm25_scorer(
    indices: &[Arc<InvertedIndex>],
    query_tokens: &Tokens,
    params: &FtsSearchParams,
) -> Result<MemBM25Scorer> {
    let terms = scorer_terms(indices, query_tokens, params)?;
    let first_index = indices.first().ok_or_else(|| {
        Error::invalid_input("FTS index requires at least one segment".to_string())
    })?;
    let (mut total_tokens, mut num_docs, first_token_docs) =
        first_index.bm25_stats_for_terms(&terms);
    let mut token_docs = HashMap::with_capacity(terms.len());
    for (term, count) in terms.iter().cloned().zip(first_token_docs.into_iter()) {
        token_docs.insert(term, count);
    }

    for index in indices.iter().skip(1) {
        let (segment_total_tokens, segment_num_docs, segment_token_docs) =
            index.bm25_stats_for_terms(&terms);
        total_tokens += segment_total_tokens;
        num_docs += segment_num_docs;
        for (term, count) in terms.iter().zip(segment_token_docs.into_iter()) {
            *token_docs
                .get_mut(term)
                .expect("global scorer terms should already be initialized") += count;
        }
    }

    Ok(MemBM25Scorer::new(total_tokens, num_docs, token_docs))
}

async fn search_segments(
    indices: &[Arc<InvertedIndex>],
    tokens: Arc<Tokens>,
    params: Arc<FtsSearchParams>,
    operator: lance_index::scalar::inverted::query::Operator,
    pre_filter: Arc<dyn PreFilter>,
    metrics: Arc<FtsIndexMetrics>,
    base_scorer: Arc<MemBM25Scorer>,
) -> Result<(Vec<u64>, Vec<f32>)> {
    let limit = params.limit.unwrap_or(usize::MAX);
    let mut candidates = std::collections::BinaryHeap::new();
    let searches = indices
        .iter()
        .map(|index| {
            let index = Arc::clone(index);
            let tokens = tokens.clone();
            let params = params.clone();
            let pre_filter = pre_filter.clone();
            let metrics = metrics.clone();
            let base_scorer = base_scorer.clone();
            async move {
                index
                    .bm25_search(
                        tokens,
                        params,
                        operator,
                        pre_filter,
                        metrics,
                        Some(base_scorer.as_ref()),
                    )
                    .await
            }
        })
        .collect::<Vec<_>>();
    let searches = stream::iter(searches).buffer_unordered(get_num_compute_intensive_cpus());
    let mut searches = searches;

    while let Some((doc_ids, scores)) = searches.try_next().await? {
        for (row_id, score) in doc_ids.into_iter().zip(scores.into_iter()) {
            if candidates.len() < limit {
                candidates.push(std::cmp::Reverse(ScoredDoc::new(row_id, score)));
            } else if candidates.peek().unwrap().0.score.0 < score {
                candidates.pop();
                candidates.push(std::cmp::Reverse(ScoredDoc::new(row_id, score)));
            }
        }
    }

    Ok(candidates
        .into_sorted_vec()
        .into_iter()
        .map(|std::cmp::Reverse(doc)| (doc.row_id, doc.score.0))
        .unzip())
}

/// Fall back to the default simple tokenizer when no on-disk FTS segment exists.
fn default_text_tokenizer() -> Box<dyn LanceTokenizer> {
    Box::new(TextTokenizer::new(
        TextAnalyzer::builder(SimpleTokenizer::default()).build(),
    ))
}

pub struct FtsIndexMetrics {
    index_metrics: IndexMetrics,
    partitions_searched: Count,
    baseline_metrics: BaselineMetrics,
}

impl FtsIndexMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            index_metrics: IndexMetrics::new(metrics, partition),
            partitions_searched: metrics.new_count(PARTITIONS_SEARCHED_METRIC, partition),
            baseline_metrics: BaselineMetrics::new(metrics, partition),
        }
    }

    pub fn record_parts_searched(&self, num_parts: usize) {
        self.partitions_searched.add(num_parts);
    }
}

impl MetricsCollector for FtsIndexMetrics {
    fn record_parts_loaded(&self, num_parts: usize) {
        self.index_metrics.record_parts_loaded(num_parts);
    }

    fn record_index_loads(&self, num_indexes: usize) {
        self.index_metrics.record_index_loads(num_indexes);
    }

    fn record_comparisons(&self, num_comparisons: usize) {
        self.index_metrics.record_comparisons(num_comparisons);
    }
}

#[derive(Debug)]
pub struct MatchQueryExec {
    dataset: Arc<Dataset>,
    query: MatchQuery,
    params: FtsSearchParams,
    prefilter_source: PreFilterSource,

    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for MatchQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "MatchQuery: column={}, query={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "MatchQuery\ncolumn={}\nquery={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
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
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
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
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let column = query.column.ok_or(DataFusionError::Execution(format!(
            "column not set for MatchQuery {}",
            query.terms
        )))?;
        let stream = stream::once(async move {
            let _timer = metrics.baseline_metrics.elapsed_compute().timer();
            let segments = load_segments(&ds, &column)
                .await?
                .ok_or(DataFusionError::Execution(format!(
                    "No Inverted index found for column {}",
                    column,
                )))?;
            let _details = load_segment_details(&ds, &column, &segments).await?;
            let indices =
                open_fts_segments(&ds, &column, &segments, &metrics.index_metrics).await?;

            let mut pre_filter =
                build_prefilter(context.clone(), partition, &prefilter_source, ds, &segments)?;
            let deleted_fragments =
                indices
                    .iter()
                    .fold(roaring::RoaringBitmap::new(), |mut deleted, index| {
                        deleted |= index.deleted_fragments().clone();
                        deleted
                    });
            if !deleted_fragments.is_empty() {
                Arc::get_mut(&mut pre_filter)
                    .expect("prefilter just created")
                    .set_deleted_fragments(deleted_fragments);
            }
            metrics
                .record_parts_searched(indices.iter().map(|index| index.partition_count()).sum());

            let is_fuzzy = matches!(query.fuzziness, Some(n) if n != 0);
            let params = params
                .with_fuzziness(query.fuzziness)
                .with_max_expansions(query.max_expansions)
                .with_prefix_length(query.prefix_length);
            let first_index = indices.first().ok_or(DataFusionError::Execution(format!(
                "FTS index for column {} has no segments",
                column
            )))?;
            let mut tokenizer = match is_fuzzy {
                false => first_index.tokenizer(),
                true => {
                    let tokenizer = TextAnalyzer::from(SimpleTokenizer::default());
                    match first_index.tokenizer().doc_type() {
                        DocType::Text => {
                            Box::new(TextTokenizer::new(tokenizer)) as Box<dyn LanceTokenizer>
                        }
                        DocType::Json => {
                            Box::new(JsonTokenizer::new(tokenizer)) as Box<dyn LanceTokenizer>
                        }
                    }
                }
            };
            let tokens = collect_query_tokens(&query.terms, &mut tokenizer);
            let base_scorer = build_global_bm25_scorer(&indices, &tokens, &params)?;

            pre_filter.wait_for_ready().await?;
            let tokens = Arc::new(tokens);
            let params = Arc::new(params);
            let (doc_ids, mut scores) = search_segments(
                &indices,
                tokens,
                params,
                query.operator,
                pre_filter,
                metrics.clone(),
                Arc::new(base_scorer),
            )
            .await?;
            scores.iter_mut().for_each(|s| {
                *s *= query.boost;
            });
            metrics.baseline_metrics.record_output(doc_ids.len());

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
            stream.stream_in_current_span().boxed(),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

/// Filters the input, removing rows that do not share tokens with the query
#[derive(Debug)]
pub struct FlatMatchFilterExec {
    dataset: Arc<Dataset>,
    input: Arc<dyn ExecutionPlan>,
    query: MatchQuery,
    params: FtsSearchParams,

    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for FlatMatchFilterExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "FlatMatchFilter: column={}, query={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "FlatMatchFilter\ncolumn={}\nquery={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )
            }
        }
    }
}

impl FlatMatchFilterExec {
    async fn load_tokenizer(
        dataset: &Dataset,
        column: &str,
        metrics: &IndexMetrics,
    ) -> DataFusionResult<Box<dyn LanceTokenizer>> {
        if let Some(segments) = load_segments(dataset, column).await? {
            let index_meta = segments.first().ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "FTS index for column {} has no segments",
                    column
                ))
            })?;
            return Ok(open_fts_segment(dataset, column, index_meta, metrics)
                .await?
                .tokenizer());
        }
        Ok(default_text_tokenizer())
    }

    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
    ) -> Self {
        Self {
            dataset,
            input,
            query,
            params,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    fn find_matches<O: OffsetSizeTrait>(
        text_col: &dyn Array,
        tokenizer: &mut Box<dyn LanceTokenizer>,
        query_tokens: &Tokens,
    ) -> BooleanArray {
        let text_col = text_col.as_string::<O>();
        let mut predicate = BooleanBuilder::with_capacity(text_col.len());
        for idx in 0..text_col.len() {
            let value = text_col.value(idx);
            predicate.append_value(has_query_token(value, tokenizer, query_tokens));
        }
        predicate.finish()
    }

    async fn do_filter(
        input: SendableRecordBatchStream,
        dataset: Arc<Dataset>,
        query: MatchQuery,
        metrics: Arc<FtsIndexMetrics>,
    ) -> DataFusionResult<impl Stream<Item = DataFusionResult<RecordBatch>> + Send> {
        let column = query
            .column
            .as_ref()
            .ok_or(DataFusionError::Execution(format!(
                "column not set for MatchQuery {}",
                query.terms
            )))?;
        let mut tokenizer = Self::load_tokenizer(&dataset, column, &metrics.index_metrics).await?;
        let query_tokens = Arc::new(collect_query_tokens(&query.terms, &mut tokenizer));
        let column = column.clone();

        Ok(input.map(move |batch| -> DataFusionResult<_> {
            let batch = batch?;
            let text_column = batch.column_by_name(&column).ok_or_else(|| {
                DataFusionError::Execution(format!("Column {} not found in batch", column,))
            })?;
            let predicate = match text_column.data_type() {
                DataType::Utf8 => {
                    Self::find_matches::<i32>(text_column, &mut tokenizer, &query_tokens)
                }
                DataType::LargeUtf8 => {
                    Self::find_matches::<i64>(text_column, &mut tokenizer, &query_tokens)
                }
                _ => {
                    return Err(DataFusionError::Execution(format!(
                        "Column {} is not a string",
                        column,
                    )));
                }
            };
            DataFusionResult::Ok(arrow::compute::filter_record_batch(&batch, &predicate)?)
        }))
    }
}

impl ExecutionPlan for FlatMatchFilterExec {
    fn name(&self) -> &str {
        "FlatMatchFilterExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
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
        let input = children.pop().ok_or_else(|| {
            DataFusionError::Internal("Unexpected number of children".to_string())
        })?;

        Ok(Arc::new(Self {
            dataset: self.dataset.clone(),
            input,
            query: self.query.clone(),
            params: self.params.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    #[instrument(name = "flat_match_filter_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let query = self.query.clone();
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let metrics_clone = metrics.clone();

        let dataset = self.dataset.clone();
        let input = self.input.execute(partition, context)?;

        let stream =
            stream::once(async move { Self::do_filter(input, dataset, query, metrics).await })
                .try_flatten()
                .map(move |batch| {
                    if let Ok(batch) = &batch {
                        metrics_clone
                            .baseline_metrics
                            .record_output(batch.num_rows());
                    }
                    batch
                });
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            self.schema(),
            stream.stream_in_current_span().boxed(),
            partition,
            &self.metrics,
        )))
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Statistics> {
        self.input.partition_statistics(partition)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn supports_limit_pushdown(&self) -> bool {
        true
    }
}

/// Calculates the FTS score for each row in the input
#[derive(Debug)]
pub struct FlatMatchQueryExec {
    dataset: Arc<Dataset>,
    query: MatchQuery,
    params: FtsSearchParams,
    unindexed_input: Arc<dyn ExecutionPlan>,

    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for FlatMatchQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "FlatMatchQuery: column={}, query={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "FlatMatchQuery\ncolumn={}\nquery={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )
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
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
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
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let metrics_clone = metrics.clone();
        let target_batch_size = context.session_config().batch_size();

        let column = query.column.ok_or(DataFusionError::Execution(format!(
            "column not set for MatchQuery {}",
            query.terms
        )))?;
        let unindexed_input =
            document_input(self.unindexed_input.execute(partition, context)?, &column)?;

        let stream = stream::once(async move {
            let segments = load_segments(&ds, &column).await?;
            let (tokenizer, base_scorer) = match segments {
                Some(segments) => {
                    let _details = load_segment_details(&ds, &column, &segments).await?;
                    let indices =
                        open_fts_segments(&ds, &column, &segments, &metrics.index_metrics).await?;
                    metrics.record_parts_searched(
                        indices.iter().map(|index| index.partition_count()).sum(),
                    );
                    let first_index = indices.first().ok_or(DataFusionError::Execution(
                        format!("FTS index for column {} has no segments", column),
                    ))?;
                    let mut tokenizer = first_index.tokenizer();
                    let query_tokens = collect_query_tokens(&query.terms, &mut tokenizer);
                    let base_scorer =
                        build_global_bm25_scorer(&indices, &query_tokens, &FtsSearchParams::new())?;
                    (tokenizer, Some(base_scorer))
                }
                None => (default_text_tokenizer(), None),
            };

            flat_bm25_search_stream(
                unindexed_input,
                column,
                query.terms,
                tokenizer,
                base_scorer,
                target_batch_size,
            )
            .await
        })
        .try_flatten()
        .map(move |batch| {
            if let Ok(batch) = &batch {
                metrics_clone
                    .baseline_metrics
                    .record_output(batch.num_rows());
            }
            batch
        });
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            self.schema(),
            stream.stream_in_current_span().boxed(),
            partition,
            &self.metrics,
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct PhraseQueryExec {
    dataset: Arc<Dataset>,
    query: PhraseQuery,
    params: FtsSearchParams,
    prefilter_source: PreFilterSource,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for PhraseQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "PhraseQuery: column={}, query={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "PhraseQuery\ncolumn={}\nquery={}",
                    self.query.column.as_deref().unwrap_or_default(),
                    self.query.terms
                )
            }
        }
    }
}

impl PhraseQueryExec {
    pub fn new(
        dataset: Arc<Dataset>,
        query: PhraseQuery,
        mut params: FtsSearchParams,
        prefilter_source: PreFilterSource,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        params = params.with_phrase_slop(Some(query.slop));

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
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let stream = stream::once(async move {
            let _timer = metrics.baseline_metrics.elapsed_compute().timer();
            let column = query.column.ok_or(DataFusionError::Execution(format!(
                "column not set for PhraseQuery {}",
                query.terms
            )))?;
            let segments = load_segments(&ds, &column)
                .await?
                .ok_or(DataFusionError::Execution(format!(
                    "No Inverted index found for column {}",
                    column,
                )))?;
            let _details = load_segment_details(&ds, &column, &segments).await?;
            let indices =
                open_fts_segments(&ds, &column, &segments, &metrics.index_metrics).await?;

            let mut pre_filter =
                build_prefilter(context.clone(), partition, &prefilter_source, ds, &segments)?;
            let deleted_fragments =
                indices
                    .iter()
                    .fold(roaring::RoaringBitmap::new(), |mut deleted, index| {
                        deleted |= index.deleted_fragments().clone();
                        deleted
                    });
            if !deleted_fragments.is_empty() {
                Arc::get_mut(&mut pre_filter)
                    .expect("prefilter just created")
                    .set_deleted_fragments(deleted_fragments);
            }
            metrics
                .record_parts_searched(indices.iter().map(|index| index.partition_count()).sum());

            let first_index = indices.first().ok_or(DataFusionError::Execution(format!(
                "FTS index for column {} has no segments",
                column
            )))?;
            let mut tokenizer = first_index.tokenizer();
            let tokens = collect_query_tokens(&query.terms, &mut tokenizer);
            let base_scorer = build_global_bm25_scorer(&indices, &tokens, &params)?;

            pre_filter.wait_for_ready().await?;
            let tokens = Arc::new(tokens);
            let params = Arc::new(params);
            let (doc_ids, scores) = search_segments(
                &indices,
                tokens,
                params,
                lance_index::scalar::inverted::query::Operator::And,
                pre_filter,
                metrics.clone(),
                Arc::new(base_scorer),
            )
            .await?;
            metrics.baseline_metrics.record_output(doc_ids.len());
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
            stream.stream_in_current_span().boxed(),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct BoostQueryExec {
    query: BoostQuery,
    params: FtsSearchParams,
    positive: Arc<dyn ExecutionPlan>,
    negative: Arc<dyn ExecutionPlan>,

    properties: Arc<PlanProperties>,
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
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "BoostQuery\nnegative_boost={}",
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
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
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
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));
        let stream = stream::once(async move {
            let positive = positive.try_collect::<Vec<_>>().await?;
            let negative = negative.try_collect::<Vec<_>>().await?;

            let _timer = metrics.baseline_metrics.elapsed_compute().timer();
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
            metrics.baseline_metrics.record_output(doc_ids.len());

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
            stream.stream_in_current_span().boxed(),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct BooleanQueryExec {
    query: BooleanQuery,
    params: FtsSearchParams,
    should: Arc<dyn ExecutionPlan>,
    must: Option<Arc<dyn ExecutionPlan>>,
    must_not: Arc<dyn ExecutionPlan>,

    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for BooleanQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "BooleanQuery: should={:?}, must={:?}, must_not={:?}",
                    self.query.should, self.query.must, self.query.must_not,
                )
            }
            DisplayFormatType::TreeRender => {
                write!(f, "BooleanQuery")?;
                if !self.query.should.is_empty() {
                    write!(f, "\nshould={:?}", self.query.should)?;
                }
                if !self.query.must.is_empty() {
                    write!(f, "\nmust={:?}", self.query.must)?;
                }
                if !self.query.must_not.is_empty() {
                    write!(f, "\nmust_not={:?}", self.query.must_not)?;
                }
                std::fmt::Result::Ok(())
            }
        }
    }
}

impl BooleanQueryExec {
    pub fn new(
        query: BooleanQuery,
        params: FtsSearchParams,
        should: Arc<dyn ExecutionPlan>,
        must: Option<Arc<dyn ExecutionPlan>>,
        must_not: Arc<dyn ExecutionPlan>,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(FTS_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            query,
            params,
            must,
            should,
            must_not,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl ExecutionPlan for BooleanQueryExec {
    fn name(&self) -> &str {
        "BooleanQueryExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.must {
            Some(must) => vec![&self.should, &self.must_not, must],
            None => vec![&self.should, &self.must_not],
        }
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
        match children.len() {
            1 => {
                let should = children.pop().unwrap();
                Ok(Arc::new(Self {
                    query: self.query.clone(),
                    params: self.params.clone(),
                    should,
                    must: None,
                    must_not: self.must_not.clone(),
                    properties: self.properties.clone(),
                    metrics: ExecutionPlanMetricsSet::new(),
                }))
            }
            2 => {
                let must_not = children.pop().unwrap();
                let should = children.pop().unwrap();
                Ok(Arc::new(Self {
                    query: self.query.clone(),
                    params: self.params.clone(),
                    should,
                    must: None,
                    must_not,
                    properties: self.properties.clone(),
                    metrics: ExecutionPlanMetricsSet::new(),
                }))
            }
            3 => {
                let must = children.pop().unwrap();
                let must_not = children.pop().unwrap();
                let should = children.pop().unwrap();
                Ok(Arc::new(Self {
                    query: self.query.clone(),
                    params: self.params.clone(),
                    should,
                    must: Some(must),
                    must_not,
                    properties: self.properties.clone(),
                    metrics: ExecutionPlanMetricsSet::new(),
                }))
            }
            _ => Err(DataFusionError::Internal(
                "Unexpected number of children".to_string(),
            )),
        }
    }

    #[instrument(name = "bool_query_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let params = self.params.clone();
        let should_plan = self.should.clone();
        let must_plan = self.must.clone();
        let must_not_plan = self.must_not.clone();
        let must = self
            .must
            .as_ref()
            .map(|m| m.execute(partition, context.clone()))
            .transpose()?;
        let mut should = self.should.execute(partition, context.clone())?;
        let mut must_not = self.must_not.execute(partition, context)?;
        let metrics = Arc::new(FtsIndexMetrics::new(&self.metrics, partition));

        let stream = stream::once(async move {
            let elapsed_time = metrics.baseline_metrics.elapsed_compute();

            let mut res = HashMap::new();
            let has_must = must.is_some();
            if let Some(mut must) = must {
                while let Some(batch) = must.try_next().await? {
                    let _timer = elapsed_time.timer();
                    let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().values();
                    let scores = batch[SCORE_COL].as_primitive::<Float32Type>().values();
                    res.extend(std::iter::zip(
                        row_ids.iter().copied(),
                        scores.iter().copied(),
                    ));
                }
            }

            // add the scores from the should clause
            while let Some(batch) = should.try_next().await? {
                let _timer = elapsed_time.timer();
                let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().values();
                let scores = batch[SCORE_COL].as_primitive::<Float32Type>().values();

                for (row_id, score) in std::iter::zip(row_ids, scores) {
                    let entry = res.entry(*row_id).and_modify(|e| *e += score);
                    if !has_must {
                        entry.or_insert(*score);
                    }
                }
            }

            // remove the results from the must_not clause
            while let Some(batch) = must_not.try_next().await? {
                let _timer = elapsed_time.timer();
                let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().values();
                for row_id in row_ids {
                    res.remove(row_id);
                }
            }

            let mut partitions_searched = 0;
            for plan in [Some(&should_plan), must_plan.as_ref(), Some(&must_not_plan)] {
                let Some(plan) = plan else {
                    continue;
                };
                let Some(metrics) = plan.metrics() else {
                    continue;
                };
                for (metric_name, count) in metrics.iter_counts() {
                    if metric_name.as_ref() == PARTITIONS_SEARCHED_METRIC {
                        partitions_searched += count.value();
                    }
                }
            }
            metrics.record_parts_searched(partitions_searched);

            // sort the results and take the top k
            let _timer = elapsed_time.timer();
            let (row_ids, scores): (Vec<_>, Vec<_>) = res
                .into_iter()
                .sorted_unstable_by(|(_, a), (_, b)| b.total_cmp(a))
                .take(params.limit.unwrap_or(usize::MAX))
                .unzip();
            metrics.baseline_metrics.record_output(row_ids.len());
            let batch = RecordBatch::try_new(
                FTS_SCHEMA.clone(),
                vec![
                    Arc::new(UInt64Array::from(row_ids)),
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

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::index::DatasetIndexExt;
    use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
    use datafusion::{execution::TaskContext, physical_plan::ExecutionPlan};
    use lance_datafusion::datagen::DatafusionDatagenExt;
    use lance_datafusion::exec::{ExecutionStatsCallback, ExecutionSummaryCounts};
    use lance_datafusion::utils::PARTITIONS_SEARCHED_METRIC;
    use lance_datagen::{BatchCount, ByteCount, RowCount};
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::scalar::inverted::InvertedIndex;
    use lance_index::scalar::inverted::Language;
    use lance_index::scalar::inverted::query::{
        BooleanQuery, BoostQuery, FtsQuery, FtsSearchParams, MatchQuery, Occur, Operator,
        PhraseQuery, collect_query_tokens, has_query_token,
    };
    use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
    use lance_index::{IndexCriteria, IndexType};

    use crate::{
        dataset::transaction::{Operation, TransactionBuilder},
        index::DatasetIndexInternalExt,
        io::exec::PreFilterSource,
        utils::test::{DatagenExt, FragmentCount, FragmentRowCount, NoContextTestFixture},
    };

    use super::{
        BoostQueryExec, FlatMatchFilterExec, FlatMatchQueryExec, MatchQueryExec, PhraseQueryExec,
    };
    use crate::io::exec::utils::IndexMetrics;

    #[derive(Default)]
    struct StatsHolder {
        collected_stats: Arc<Mutex<Option<ExecutionSummaryCounts>>>,
    }

    impl StatsHolder {
        fn get_setter(&self) -> ExecutionStatsCallback {
            let collected_stats = self.collected_stats.clone();
            Arc::new(move |stats| {
                *collected_stats.lock().unwrap() = Some(stats.clone());
            })
        }

        fn consume(self) -> ExecutionSummaryCounts {
            self.collected_stats.lock().unwrap().take().unwrap()
        }
    }

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
        let metrics = match_query.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);

        let flat_input = lance_datagen::gen_batch()
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
        let metrics = flat_match_query.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);

        let phrase_query = PhraseQueryExec::new(
            Arc::new(fixture.dataset.clone()),
            PhraseQuery::new("blah".to_string()),
            FtsSearchParams::new().with_phrase_slop(Some(0)),
            PreFilterSource::None,
        );
        phrase_query
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();
        let metrics = phrase_query.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);

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
        let metrics = boost_query.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_flat_match_filter_load_tokenizer_uses_on_disk_params_when_details_missing() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "text",
                lance_datagen::array::cycle_utf8_literals(&["hello", "HELLO"]),
            )
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(2))
            .await
            .unwrap();

        let params = InvertedIndexParams::new("simple".to_string(), Language::English)
            .with_position(false)
            .lower_case(false)
            .stem(false)
            .remove_stop_words(false)
            .ascii_folding(false)
            .max_token_length(None);
        dataset
            .create_index(&["text"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();

        let index_meta = dataset
            .load_scalar_index(IndexCriteria::default().for_column("text").supports_fts())
            .await
            .unwrap()
            .unwrap();
        let mut legacy_index_meta = index_meta.clone();
        legacy_index_meta.index_details = None;
        let transaction = TransactionBuilder::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![legacy_index_meta],
                removed_indices: vec![index_meta],
            },
        )
        .build();
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();

        let metrics = IndexMetrics::new(&ExecutionPlanMetricsSet::new(), 0);
        let mut tokenizer = FlatMatchFilterExec::load_tokenizer(&dataset, "text", &metrics)
            .await
            .unwrap();
        let query_tokens = collect_query_tokens("hello", &mut tokenizer);

        let mut tokenizer = FlatMatchFilterExec::load_tokenizer(&dataset, "text", &metrics)
            .await
            .unwrap();
        assert!(has_query_token("hello", &mut tokenizer, &query_tokens));
        assert!(
            !has_query_token("HELLO", &mut tokenizer, &query_tokens),
            "legacy FTS indices should continue using on-disk tokenizer params"
        );
    }

    #[tokio::test]
    async fn test_parts_searched_metrics() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "text",
                lance_datagen::array::cycle_utf8_literals(&["hello", "lance", "search"]),
            )
            .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(5))
            .await
            .unwrap();

        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        let index_meta = dataset
            .load_scalar_index(IndexCriteria::default().for_column("text").supports_fts())
            .await
            .unwrap()
            .unwrap();
        let index = dataset
            .open_generic_index("text", &index_meta.uuid.to_string(), &NoOpMetricsCollector)
            .await
            .unwrap();
        let inverted_index = index.as_any().downcast_ref::<InvertedIndex>().unwrap();
        let expected_parts = inverted_index.partition_count();

        let stats_holder = StatsHolder::default();
        let mut scanner = dataset.scan();
        scanner
            .scan_stats_callback(stats_holder.get_setter())
            .project(&["text"])
            .unwrap()
            .with_row_id()
            .full_text_search(FullTextSearchQuery::new("hello".to_string()))
            .unwrap();
        let _ = scanner.try_into_batch().await.unwrap();
        let stats = stats_holder.consume();
        let parts_searched = stats
            .all_counts
            .get(PARTITIONS_SEARCHED_METRIC)
            .copied()
            .unwrap_or_default();
        assert_eq!(parts_searched, expected_parts);

        let mut analyze_scanner = dataset.scan();
        analyze_scanner
            .project(&["text"])
            .unwrap()
            .with_row_id()
            .full_text_search(FullTextSearchQuery::new("hello".to_string()))
            .unwrap();
        let analysis = analyze_scanner.analyze_plan().await.unwrap();
        assert!(analysis.contains(PARTITIONS_SEARCHED_METRIC));
    }

    #[tokio::test]
    async fn test_boolean_query_parts_searched_metrics() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "text",
                lance_datagen::array::cycle_utf8_literals(&["hello", "lance", "search"]),
            )
            .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(5))
            .await
            .unwrap();

        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        let index_meta = dataset
            .load_scalar_index(IndexCriteria::default().for_column("text").supports_fts())
            .await
            .unwrap()
            .unwrap();
        let index = dataset
            .open_generic_index("text", &index_meta.uuid.to_string(), &NoOpMetricsCollector)
            .await
            .unwrap();
        let inverted_index = index.as_any().downcast_ref::<InvertedIndex>().unwrap();
        let expected_parts = inverted_index.partition_count();

        let query = BooleanQuery::new([
            (
                Occur::Should,
                MatchQuery::new("hello".to_string())
                    .with_operator(Operator::And)
                    .into(),
            ),
            (
                Occur::Must,
                MatchQuery::new("lance".to_string())
                    .with_operator(Operator::And)
                    .into(),
            ),
        ]);
        let expected_total = expected_parts * 2;

        let mut scanner = dataset.scan();
        scanner
            .project(&["text"])
            .unwrap()
            .with_row_id()
            .full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap();
        let analysis = scanner.analyze_plan().await.unwrap();
        let boolean_line = analysis
            .lines()
            .find(|line| line.contains("BooleanQuery"))
            .unwrap();
        assert!(
            boolean_line.contains(&format!("{PARTITIONS_SEARCHED_METRIC}={expected_total}")),
            "BooleanQuery metrics missing partitions_searched: {boolean_line}"
        );
    }
}
