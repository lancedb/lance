// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{Float32Type, UInt64Type};
use arrow_array::{Float32Array, RecordBatch, UInt64Array};
use arrow_schema::SchemaRef;
use datafusion::common::Statistics;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use futures::stream::{self};
use futures::{StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::ROW_ID;
use lance_index::prefilter::{FilterLoader, PreFilter};
use lance_index::scalar::inverted::query::{
    collect_tokens, BoostQuery, FtsSearchParams, MatchQuery, MultiMatchQuery, PhraseQuery, Searcher,
};
use lance_index::scalar::inverted::{
    flat_bm25_search_stream, InvertedIndex, FTS_SCHEMA, SCORE_COL,
};
use lance_index::scalar::FullTextSearchQuery;
use lance_index::DatasetIndexExt;
use lance_table::format::Index;
use tracing::instrument;

use crate::index::prefilter::DatasetPreFilter;
use crate::{index::DatasetIndexInternalExt, Dataset};

use super::utils::{
    build_prefilter, FilteredRowIdsToPrefilter, IndexMetrics, InstrumentedRecordBatchStreamAdapter,
    SelectionVectorToPrefilter,
};
use super::PreFilterSource;

/// An execution node that performs full text search
///
/// This node would perform full text search with inverted index on the dataset.
/// The result is a stream of record batches containing the row ids that match the search query,
/// and scores of the matched rows.
// #[derive(Debug)]
// pub struct FtsExec {
//     dataset: Arc<Dataset>,
//     // column -> (indices, unindexed input stream)
//     indices: HashMap<String, Vec<Index>>,
//     query: FullTextSearchQuery,
//     /// Prefiltering input
//     prefilter_source: PreFilterSource,
//     properties: PlanProperties,

//     metrics: ExecutionPlanMetricsSet,
// }

// impl DisplayAs for FtsExec {
//     fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         match t {
//             DisplayFormatType::Default | DisplayFormatType::Verbose => {
//                 write!(f, "Fts: query={}", self.query.query)
//             }
//         }
//     }
// }

// impl FtsExec {
//     pub fn new(
//         dataset: Arc<Dataset>,
//         indices: HashMap<String, Vec<Index>>,
//         query: FullTextSearchQuery,
//         prefilter_source: PreFilterSource,
//     ) -> Self {
//         let properties = PlanProperties::new(
//             EquivalenceProperties::new(FTS_SCHEMA.clone()),
//             Partitioning::RoundRobinBatch(1),
//             EmissionType::Incremental,
//             Boundedness::Bounded,
//         );
//         Self {
//             dataset,
//             indices,
//             query,
//             prefilter_source,
//             properties,
//             metrics: ExecutionPlanMetricsSet::new(),
//         }
//     }
// }

// impl ExecutionPlan for FtsExec {
//     fn name(&self) -> &str {
//         "FtsExec"
//     }

//     fn as_any(&self) -> &dyn std::any::Any {
//         self
//     }

//     fn schema(&self) -> SchemaRef {
//         FTS_SCHEMA.clone()
//     }

//     fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
//         match &self.prefilter_source {
//             PreFilterSource::None => vec![],
//             PreFilterSource::FilteredRowIds(src) => vec![&src],
//             PreFilterSource::ScalarIndexQuery(src) => vec![&src],
//         }
//     }

//     fn with_new_children(
//         self: Arc<Self>,
//         mut children: Vec<Arc<dyn ExecutionPlan>>,
//     ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
//         let plan = match children.len() {
//             0 => Self {
//                 dataset: self.dataset.clone(),
//                 indices: self.indices.clone(),
//                 query: self.query.clone(),
//                 prefilter_source: PreFilterSource::None,
//                 properties: self.properties.clone(),
//                 metrics: ExecutionPlanMetricsSet::new(),
//             },
//             1 => {
//                 let src = children.pop().unwrap();
//                 let prefilter_source = match &self.prefilter_source {
//                     PreFilterSource::FilteredRowIds(_) => {
//                         PreFilterSource::FilteredRowIds(src.clone())
//                     }
//                     PreFilterSource::ScalarIndexQuery(_) => {
//                         PreFilterSource::ScalarIndexQuery(src.clone())
//                     }
//                     PreFilterSource::None => {
//                         return Err(DataFusionError::Internal(
//                             "Unexpected prefilter source".to_string(),
//                         ));
//                     }
//                 };
//                 Self {
//                     dataset: self.dataset.clone(),
//                     indices: self.indices.clone(),
//                     query: self.query.clone(),
//                     prefilter_source,
//                     properties: self.properties.clone(),
//                     metrics: ExecutionPlanMetricsSet::new(),
//                 }
//             }
//             _ => {
//                 return Err(DataFusionError::Internal(
//                     "Unexpected number of children".to_string(),
//                 ));
//             }
//         };
//         Ok(Arc::new(plan))
//     }

//     #[instrument(name = "fts_exec", level = "debug", skip_all)]
//     fn execute(
//         &self,
//         partition: usize,
//         context: Arc<datafusion::execution::context::TaskContext>,
//     ) -> DataFusionResult<SendableRecordBatchStream> {
//         let query = self.query.clone();
//         let ds = self.dataset.clone();
//         let prefilter_source = self.prefilter_source.clone();
//         let metrics = Arc::new(IndexMetrics::new(&self.metrics, partition));
//         let indices = self.indices.clone();
//         let stream = stream::iter(indices)
//             .map(move |(column, indices)| {
//                 let index_meta = indices[0].clone();
//                 let uuid = index_meta.uuid.to_string();
//                 let query = query.clone();
//                 let ds = ds.clone();
//                 let context = context.clone();
//                 let prefilter_source = prefilter_source.clone();
//                 let metrics = metrics.clone();

//                 async move {
//                     let prefilter_loader = match &prefilter_source {
//                         PreFilterSource::FilteredRowIds(src_node) => {
//                             let stream = src_node.execute(partition, context.clone())?;
//                             Some(Box::new(FilteredRowIdsToPrefilter(stream))
//                                 as Box<dyn FilterLoader>)
//                         }
//                         PreFilterSource::ScalarIndexQuery(src_node) => {
//                             let stream = src_node.execute(partition, context.clone())?;
//                             Some(Box::new(SelectionVectorToPrefilter(stream))
//                                 as Box<dyn FilterLoader>)
//                         }
//                         PreFilterSource::None => None,
//                     };
//                     let pre_filter = Arc::new(DatasetPreFilter::new(
//                         ds.clone(),
//                         &[index_meta],
//                         prefilter_loader,
//                     ));

//                     let index = ds
//                         .open_generic_index(&column, &uuid, metrics.as_ref())
//                         .await?;
//                     let index =
//                         index
//                             .as_any()
//                             .downcast_ref::<InvertedIndex>()
//                             .ok_or_else(|| {
//                                 DataFusionError::Execution(format!(
//                                     "Index {} is not an inverted index",
//                                     uuid,
//                                 ))
//                             })?;
//                     pre_filter.wait_for_ready().await?;
//                     // let results = index
//                     //     .full_text_search(&query, pre_filter, metrics.as_ref())
//                     //     .await?;

//                     let (row_ids, scores): (Vec<u64>, Vec<f32>) = results.into_iter().unzip();
//                     let batch = RecordBatch::try_new(
//                         FTS_SCHEMA.clone(),
//                         vec![
//                             Arc::new(UInt64Array::from(row_ids)),
//                             Arc::new(Float32Array::from(scores)),
//                         ],
//                     )?;
//                     Ok::<_, DataFusionError>(batch)
//                 }
//             })
//             .buffered(self.indices.len());
//         let schema = self.schema();
//         Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
//             schema,
//             stream.boxed(),
//             partition,
//             &self.metrics,
//         )) as SendableRecordBatchStream)
//     }

//     fn statistics(&self) -> DataFusionResult<datafusion::physical_plan::Statistics> {
//         Ok(Statistics::new_unknown(&FTS_SCHEMA))
//     }

//     fn metrics(&self) -> Option<MetricsSet> {
//         Some(self.metrics.clone_inner())
//     }

//     fn properties(&self) -> &PlanProperties {
//         &self.properties
//     }
// }

/// An execution node that performs flat full text search
///
/// This node would perform flat full text search on unindexed rows.
/// The result is a stream of record batches containing the row ids that match the search query,
/// and scores of the matched rows.
// #[derive(Debug)]
// pub struct FlatFtsExec {
//     dataset: Arc<Dataset>,
//     // (column, indices, unindexed input stream)
//     column_inputs: Vec<(String, Vec<Index>, Arc<dyn ExecutionPlan>)>,
//     query: FullTextSearchQuery,
//     properties: PlanProperties,
//     metrics: ExecutionPlanMetricsSet,
// }

// impl DisplayAs for FlatFtsExec {
//     fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         match t {
//             DisplayFormatType::Default | DisplayFormatType::Verbose => {
//                 write!(f, "FlatFts: query={}", self.query.query)
//             }
//         }
//     }
// }

// impl FlatFtsExec {
//     pub fn new(
//         dataset: Arc<Dataset>,
//         column_inputs: Vec<(String, Vec<Index>, Arc<dyn ExecutionPlan>)>,
//         query: FullTextSearchQuery,
//     ) -> Self {
//         let properties = PlanProperties::new(
//             EquivalenceProperties::new(FTS_SCHEMA.clone()),
//             Partitioning::RoundRobinBatch(1),
//             EmissionType::Incremental,
//             Boundedness::Bounded,
//         );
//         Self {
//             dataset,
//             column_inputs,
//             query,
//             properties,
//             metrics: ExecutionPlanMetricsSet::new(),
//         }
//     }
// }

// impl ExecutionPlan for FlatFtsExec {
//     fn name(&self) -> &str {
//         "FlatFtsExec"
//     }

//     fn as_any(&self) -> &dyn std::any::Any {
//         self
//     }

//     fn schema(&self) -> SchemaRef {
//         FTS_SCHEMA.clone()
//     }

//     fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
//         self.column_inputs
//             .iter()
//             .map(|(_, _, input)| input)
//             .collect()
//     }

//     fn with_new_children(
//         self: Arc<Self>,
//         children: Vec<Arc<dyn ExecutionPlan>>,
//     ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
//         if self.column_inputs.len() != children.len() {
//             return Err(DataFusionError::Internal(
//                 "Unexpected number of children".to_string(),
//             ));
//         }

//         let column_inputs = self
//             .column_inputs
//             .iter()
//             .zip(children)
//             .map(|((column, indices, _), input)| (column.clone(), indices.clone(), input))
//             .collect();
//         Ok(Arc::new(Self {
//             dataset: self.dataset.clone(),
//             column_inputs,
//             query: self.query.clone(),
//             properties: self.properties.clone(),
//             metrics: ExecutionPlanMetricsSet::new(),
//         }))
//     }

//     #[instrument(name = "flat_fts_exec", level = "debug", skip_all)]
//     fn execute(
//         &self,
//         partition: usize,
//         context: Arc<datafusion::execution::context::TaskContext>,
//     ) -> DataFusionResult<SendableRecordBatchStream> {
//         let query = self.query.clone();
//         let ds = self.dataset.clone();
//         let column_inputs = self.column_inputs.clone();
//         let metrics = Arc::new(IndexMetrics::new(&self.metrics, partition));

//         let stream = stream::iter(column_inputs)
//             .map(move |(column, indices, input)| {
//                 let index_meta = indices[0].clone();
//                 let uuid = index_meta.uuid.to_string();
//                 let query = query.clone();
//                 let ds = ds.clone();
//                 let context = context.clone();
//                 let metrics = metrics.clone();

//                 async move {
//                     let index = ds
//                         .open_generic_index(&column, &uuid, metrics.as_ref())
//                         .await?;

//                     let unindexed_stream = input.execute(partition, context)?;
//                     let unindexed_result_stream =
//                         flat_bm25_search_stream(unindexed_stream, column, query, index);

//                     Ok::<_, DataFusionError>(unindexed_result_stream)
//                 }
//             })
//             .buffered(self.column_inputs.len())
//             .try_flatten();
//         let schema = self.schema();
//         Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
//             schema,
//             stream.boxed(),
//             partition,
//             &self.metrics,
//         )) as SendableRecordBatchStream)
//     }

//     fn statistics(&self) -> DataFusionResult<datafusion::physical_plan::Statistics> {
//         Ok(Statistics::new_unknown(&FTS_SCHEMA))
//     }

//     fn metrics(&self) -> Option<MetricsSet> {
//         Some(self.metrics.clone_inner())
//     }

//     fn properties(&self) -> &PlanProperties {
//         &self.properties
//     }
// }

#[derive(Debug)]
pub struct MatchQueryExec {
    dataset: Arc<Dataset>,
    query: MatchQuery,
    params: FtsSearchParams,
    unindexed_input: Arc<dyn ExecutionPlan>,
    prefilter_source: PreFilterSource,
    is_flat_search: bool,

    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for MatchQueryExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "MatchQuery: query={:?}", self.query)
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
            unindexed_input: Arc::new(EmptyExec::new(FTS_SCHEMA.clone())),
            prefilter_source,
            is_flat_search: false,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    pub fn new_flat(
        dataset: Arc<Dataset>,
        query: MatchQuery,
        params: FtsSearchParams,
        unindexed_input: Arc<dyn ExecutionPlan>,
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
            query,
            params,
            unindexed_input,
            prefilter_source,
            is_flat_search: true,
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

    fn schema(&self) -> SchemaRef {
        FTS_SCHEMA.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.prefilter_source {
            PreFilterSource::None => vec![&self.unindexed_input],
            PreFilterSource::FilteredRowIds(src) => vec![&self.unindexed_input, &src],
            PreFilterSource::ScalarIndexQuery(src) => vec![&self.unindexed_input, &src],
        }
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let plan = match children.len() {
            0 => {
                return Err(DataFusionError::Internal(
                    "Unexpected number of children".to_string(),
                ));
            }
            1..=2 => {
                let unindexed_input = children.pop().unwrap();

                let mut prefilter_source = PreFilterSource::None;
                if let Some(src) = children.pop() {
                    prefilter_source = match &self.prefilter_source {
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
                }

                Self {
                    dataset: self.dataset.clone(),
                    query: self.query.clone(),
                    params: self.params.clone(),
                    unindexed_input,
                    prefilter_source,
                    is_flat_search: self.is_flat_search,
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
        let unindexed_input = self.unindexed_input.execute(partition, context.clone())?;

        let column = query.field.ok_or(DataFusionError::Execution(format!(
            "column not set for MatchQuery {}",
            query.terms
        )))?;
        if self.is_flat_search {
            let stream = stream::once(async move {
                let index_meta = ds.load_scalar_index_for_column(&column).await?.ok_or(
                    DataFusionError::Execution(format!("No index found for column {}", column,)),
                )?;
                let uuid = index_meta.uuid.to_string();
                let index = ds
                    .open_generic_index(&column, &uuid, metrics.as_ref())
                    .await?;
                let inverted_idx =
                    index
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
            .try_flatten_unordered(Some(get_num_compute_intensive_cpus()));
            Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
                self.schema(),
                stream.boxed(),
                partition,
                &self.metrics,
            )))
        } else {
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

                let inverted_idx =
                    index
                        .as_any()
                        .downcast_ref::<InvertedIndex>()
                        .ok_or_else(|| {
                            DataFusionError::Execution(format!(
                                "Index for column {} is not an inverted index",
                                column,
                            ))
                        })?;
                let mut tokenizer = match query.is_fuzzy {
                    true => tantivy::tokenizer::TextAnalyzer::from(
                        tantivy::tokenizer::SimpleTokenizer::default(),
                    ),
                    false => inverted_idx.tokenizer(),
                };
                let mut tokens = collect_tokens(&query.terms, &mut tokenizer, None);
                if query.is_fuzzy {
                    tokens = inverted_idx.expand_fuzzy(tokens, query.max_distance)?;
                }

                pre_filter.wait_for_ready().await?;
                let (doc_ids, mut scores) = inverted_idx
                    .bm25_search(&tokens, &params, false, pre_filter, metrics.as_ref())
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
                write!(f, "PhraseQuery: query={:?}", self.query)
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
            let column = query.field.ok_or(DataFusionError::Execution(format!(
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

            let searcher = index
                .as_any()
                .downcast_ref::<InvertedIndex>()
                .ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "Index for column {} is not an inverted index",
                        column,
                    ))
                })?;

            let mut tokenizer = searcher.tokenizer();
            let tokens = collect_tokens(&query.terms, &mut tokenizer, None);

            pre_filter.wait_for_ready().await?;
            let (doc_ids, scores) = searcher
                .bm25_search(&tokens, &params, true, pre_filter, metrics.as_ref())
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
                write!(f, "BoostQuery: {:?}", self.query)
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

    fn schema(&self) -> SchemaRef {
        FTS_SCHEMA.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.positive, &self.negative]
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
        let negative = self.negative.execute(partition, context.clone())?;
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
                .take(params.limit)
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
