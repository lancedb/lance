// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::datatypes::{Float32Type, UInt32Type, UInt64Type};
use arrow_array::{
    builder::{ListBuilder, UInt32Builder},
    cast::AsArray,
    ArrayRef, RecordBatch, StringArray,
};
use arrow_array::{Array, Float32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::PlanProperties;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, SendableRecordBatchStream,
    Statistics,
};
use datafusion::{
    common::stats::Precision,
    physical_plan::execution_plan::{Boundedness, EmissionType},
};
use datafusion::{common::ColumnStatistics, physical_plan::metrics::ExecutionPlanMetricsSet};
use datafusion::{
    error::{DataFusionError, Result as DataFusionResult},
    physical_plan::metrics::MetricsSet,
};
use datafusion_physical_expr::{Distribution, EquivalenceProperties};
use futures::stream::repeat_with;
use futures::{future, stream, StreamExt, TryFutureExt, TryStreamExt};
use itertools::Itertools;
use lance_core::ROW_ID;
use lance_core::{utils::tokio::get_num_compute_intensive_cpus, ROW_ID_FIELD};
use lance_index::vector::{
    flat::compute_distance, Query, DIST_COL, INDEX_UUID_COLUMN, PART_ID_COLUMN,
};
use lance_linalg::distance::DistanceType;
use lance_linalg::kernels::normalize_arrow;
use lance_table::format::Index;
use snafu::location;

use crate::dataset::Dataset;
use crate::index::prefilter::{DatasetPreFilter, FilterLoader};
use crate::index::vector::utils::get_vector_type;
use crate::index::DatasetIndexInternalExt;
use crate::{Error, Result};
use lance_arrow::*;

use super::utils::{
    FilteredRowIdsToPrefilter, IndexMetrics, InstrumentedRecordBatchStreamAdapter, PreFilterSource,
    SelectionVectorToPrefilter,
};

pub struct AnnMetrics {
    index_metrics: IndexMetrics,
}

impl AnnMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            index_metrics: IndexMetrics::new(metrics, partition),
        }
    }
}

/// [ExecutionPlan] compute vector distance from a query vector.
///
/// Preconditions:
/// - `input` schema must contains `query.column`,
/// - The column must be a vector column.
///
/// WARNING: Internal API with no stability guarantees.
#[derive(Debug)]
pub struct KNNVectorDistanceExec {
    /// Inner input node.
    pub input: Arc<dyn ExecutionPlan>,

    /// The vector query to execute.
    pub query: ArrayRef,
    column: String,
    distance_type: DistanceType,

    output_schema: SchemaRef,
    properties: PlanProperties,

    metrics: ExecutionPlanMetricsSet,
}

impl DisplayAs for KNNVectorDistanceExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "KNNVectorDistance: metric={}", self.distance_type,)
            }
            DisplayFormatType::TreeRender => {
                write!(f, "KNNVectorDistance\nmetric={}", self.distance_type,)
            }
        }
    }
}

impl KNNVectorDistanceExec {
    /// Create a new [KNNFlatExec] node.
    ///
    /// Returns an error if the preconditions are not met.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        column: &str,
        query: ArrayRef,
        distance_type: DistanceType,
    ) -> Result<Self> {
        let mut output_schema = input.schema().as_ref().clone();
        get_vector_type(&(&output_schema).try_into()?, column)?;

        // FlatExec appends a distance column to the input schema. The input
        // may already have a distance column (possibly in the wrong position), so
        // we need to remove it before adding a new one.
        if output_schema.column_with_name(DIST_COL).is_some() {
            output_schema = output_schema.without_column(DIST_COL);
        }
        let output_schema = Arc::new(output_schema.try_with_column(Field::new(
            DIST_COL,
            DataType::Float32,
            true,
        ))?);

        // This node has the same partitioning & boundedness as the input node
        // but it destroys any ordering.
        let properties = input
            .properties()
            .clone()
            .with_eq_properties(EquivalenceProperties::new(output_schema.clone()));

        Ok(Self {
            input,
            query,
            column: column.to_string(),
            distance_type,
            output_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl ExecutionPlan for KNNVectorDistanceExec {
    fn name(&self) -> &str {
        "KNNVectorDistanceExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Flat KNN inherits the schema from input node, and add one distance column.
    fn schema(&self) -> arrow_schema::SchemaRef {
        self.output_schema.clone()
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
                "KNNVectorDistanceExec node must have exactly one child".to_string(),
            ));
        }

        Ok(Arc::new(Self::try_new(
            children.pop().expect("length checked"),
            &self.column,
            self.query.clone(),
            self.distance_type,
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        let key = self.query.clone();
        let column = self.column.clone();
        let dt = self.distance_type;
        let stream = input_stream
            .try_filter(|batch| future::ready(batch.num_rows() > 0))
            .map(move |batch| {
                let key = key.clone();
                let column = column.clone();
                async move {
                    compute_distance(key, dt, &column, batch?)
                        .await
                        .map_err(|e| DataFusionError::Execution(e.to_string()))
                }
            })
            .buffer_unordered(get_num_compute_intensive_cpus());
        let schema = self.schema();
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            schema,
            stream.boxed(),
            partition,
            &self.metrics,
        )) as SendableRecordBatchStream)
    }

    fn statistics(&self) -> DataFusionResult<Statistics> {
        let inner_stats = self.input.statistics()?;
        let schema = self.input.schema();
        let dist_stats = inner_stats
            .column_statistics
            .iter()
            .zip(schema.fields())
            .find(|(_, field)| field.name() == &self.column)
            .map(|(stats, _)| ColumnStatistics {
                null_count: stats.null_count,
                ..Default::default()
            })
            .unwrap_or_default();
        let column_statistics = inner_stats
            .column_statistics
            .into_iter()
            .zip(schema.fields())
            .filter(|(_, field)| field.name() != DIST_COL)
            .map(|(stats, _)| stats)
            .chain(std::iter::once(dist_stats))
            .collect::<Vec<_>>();
        Ok(Statistics {
            num_rows: inner_stats.num_rows,
            column_statistics,
            ..Statistics::new_unknown(self.schema().as_ref())
        })
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
}

lazy_static::lazy_static! {
    pub static ref KNN_INDEX_SCHEMA: SchemaRef = Arc::new(Schema::new(vec![
        Field::new(DIST_COL, DataType::Float32, true),
        ROW_ID_FIELD.clone(),
    ]));

    pub static ref KNN_PARTITION_SCHEMA: SchemaRef = Arc::new(Schema::new(vec![
        Field::new(PART_ID_COLUMN, DataType::List(Field::new("item", DataType::UInt32, false).into()), false),
        Field::new(INDEX_UUID_COLUMN, DataType::Utf8, false),
    ]));
}

pub fn new_knn_exec(
    dataset: Arc<Dataset>,
    indices: &[Index],
    query: &Query,
    prefilter_source: PreFilterSource,
) -> Result<Arc<dyn ExecutionPlan>> {
    let ivf_node = ANNIvfPartitionExec::try_new(
        dataset.clone(),
        indices.iter().map(|idx| idx.uuid.to_string()).collect_vec(),
        query.clone(),
    )?;

    let sub_index = ANNIvfSubIndexExec::try_new(
        Arc::new(ivf_node),
        dataset,
        indices.to_vec(),
        query.clone(),
        prefilter_source,
    )?;

    Ok(Arc::new(sub_index))
}

/// [ExecutionPlan] to execute the find the closest IVF partitions.
///
/// It searches the partition IDs using the input query.
///
/// It allows to search multiple delta indices in parallel, and returns a
/// single RecordBatch, where each row contains the partition IDs and the delta index
/// `uuid`:
///
/// ```text
/// {
///    "__ivf_part_id": List<UInt32>,
///    "__index_uuid": String,
/// }
/// ```
#[derive(Debug)]
pub struct ANNIvfPartitionExec {
    pub dataset: Arc<Dataset>,

    /// The vector query to execute.
    pub query: Query,

    /// The UUIDs of the indices to search.
    pub index_uuids: Vec<String>,

    pub properties: PlanProperties,

    pub metrics: ExecutionPlanMetricsSet,
}

impl ANNIvfPartitionExec {
    pub fn try_new(dataset: Arc<Dataset>, index_uuids: Vec<String>, query: Query) -> Result<Self> {
        let dataset_schema = dataset.schema();
        get_vector_type(dataset_schema, &query.column)?;
        if index_uuids.is_empty() {
            return Err(Error::Execution {
                message: "ANNIVFPartitionExec node: no index found for query".to_string(),
                location: location!(),
            });
        }

        let schema = KNN_PARTITION_SCHEMA.clone();
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Ok(Self {
            dataset,
            query,
            index_uuids,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl DisplayAs for ANNIvfPartitionExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ANNIvfPartition: uuid={}, nprobes={}, deltas={}",
                    self.index_uuids[0],
                    self.query.nprobes,
                    self.index_uuids.len()
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "ANNIvfPartition\nuuid={}\nnprobes={}\ndeltas={}",
                    self.index_uuids[0],
                    self.query.nprobes,
                    self.index_uuids.len()
                )
            }
        }
    }
}

impl ExecutionPlan for ANNIvfPartitionExec {
    fn name(&self) -> &str {
        "ANNIVFPartitionExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        KNN_PARTITION_SCHEMA.clone()
    }

    fn statistics(&self) -> DataFusionResult<Statistics> {
        Ok(Statistics {
            num_rows: Precision::Exact(self.query.nprobes),
            ..Statistics::new_unknown(self.schema().as_ref())
        })
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            Err(DataFusionError::Internal(
                "ANNIVFPartitionExec node does not accept children".to_string(),
            ))
        } else {
            Ok(self)
        }
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let query = self.query.clone();
        let ds = self.dataset.clone();
        let metrics = Arc::new(AnnMetrics::new(&self.metrics, partition));
        let stream = stream::iter(self.index_uuids.clone())
            .map(move |uuid| {
                let query = query.clone();
                let ds = ds.clone();
                let metrics = metrics.clone();

                async move {
                    let index = ds
                        .open_vector_index(&query.column, &uuid, &metrics.index_metrics)
                        .await?;

                    let mut query = query.clone();
                    if index.metric_type() == DistanceType::Cosine {
                        let key = normalize_arrow(&query.key)?;
                        query.key = key;
                    };

                    let partitions = index.find_partitions(&query).map_err(|e| {
                        DataFusionError::Execution(format!("Failed to find partitions: {}", e))
                    })?;

                    let mut list_builder = ListBuilder::new(UInt32Builder::new())
                        .with_field(Field::new("item", DataType::UInt32, false));
                    list_builder.append_value(partitions.iter());
                    let partition_col = list_builder.finish();
                    let uuid_col = StringArray::from(vec![uuid.as_str()]);
                    let batch = RecordBatch::try_new(
                        KNN_PARTITION_SCHEMA.clone(),
                        vec![Arc::new(partition_col), Arc::new(uuid_col)],
                    )?;
                    Ok::<_, DataFusionError>(batch)
                }
            })
            .buffered(self.index_uuids.len());
        let schema = self.schema();
        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            schema,
            stream.boxed(),
            partition,
            &self.metrics,
        )) as SendableRecordBatchStream)
    }
}

/// Datafusion [ExecutionPlan] to run search on IVF partitions.
///
/// A IVF-{PQ/SQ/HNSW} query plan is:
///
/// ```text
/// AnnSubIndexExec: k=10
///   AnnPartitionExec: nprobes=20
/// ```
#[derive(Debug)]
pub struct ANNIvfSubIndexExec {
    /// Inner input source node.
    input: Arc<dyn ExecutionPlan>,

    dataset: Arc<Dataset>,

    indices: Vec<Index>,

    /// Vector Query.
    query: Query,

    /// Prefiltering input
    prefilter_source: PreFilterSource,

    /// Datafusion Plan Properties
    properties: PlanProperties,

    metrics: ExecutionPlanMetricsSet,
}

impl ANNIvfSubIndexExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        indices: Vec<Index>,
        query: Query,
        prefilter_source: PreFilterSource,
    ) -> Result<Self> {
        if input.schema().field_with_name(PART_ID_COLUMN).is_err() {
            return Err(Error::Index {
                message: format!(
                    "ANNSubIndexExec node: input schema does not have \"{}\" column",
                    PART_ID_COLUMN
                ),
                location: location!(),
            });
        }
        let properties = PlanProperties::new(
            EquivalenceProperties::new(KNN_INDEX_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Ok(Self {
            input,
            dataset,
            indices,
            query,
            prefilter_source,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl DisplayAs for ANNIvfSubIndexExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ANNSubIndex: name={}, k={}, deltas={}",
                    self.indices[0].name,
                    self.query.k * self.query.refine_factor.unwrap_or(1) as usize,
                    self.indices.len()
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "ANNSubIndex\nname={}\nk={}\ndeltas={}",
                    self.indices[0].name,
                    self.query.k * self.query.refine_factor.unwrap_or(1) as usize,
                    self.indices.len()
                )
            }
        }
    }
}

impl ExecutionPlan for ANNIvfSubIndexExec {
    fn name(&self) -> &str {
        "ANNSubIndexExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        KNN_INDEX_SCHEMA.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.prefilter_source {
            PreFilterSource::None => vec![&self.input],
            PreFilterSource::FilteredRowIds(src) => vec![&self.input, &src],
            PreFilterSource::ScalarIndexQuery(src) => vec![&self.input, &src],
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
        let plan = if children.len() == 1 || children.len() == 2 {
            let prefilter_source = if children.len() == 2 {
                let prefilter = children.pop().expect("length checked");
                match &self.prefilter_source {
                    PreFilterSource::None => PreFilterSource::None,
                    PreFilterSource::FilteredRowIds(_) => {
                        PreFilterSource::FilteredRowIds(prefilter)
                    }
                    PreFilterSource::ScalarIndexQuery(_) => {
                        PreFilterSource::ScalarIndexQuery(prefilter)
                    }
                }
            } else {
                self.prefilter_source.clone()
            };

            Self {
                input: children.pop().expect("length checked"),
                dataset: self.dataset.clone(),
                indices: self.indices.clone(),
                query: self.query.clone(),
                prefilter_source,
                properties: self.properties.clone(),
                metrics: ExecutionPlanMetricsSet::new(),
            }
        } else {
            return Err(DataFusionError::Internal(
                "ANNSubIndexExec node must have exactly one or two (prefilter) child".to_string(),
            ));
        };
        Ok(Arc::new(plan))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> DataFusionResult<datafusion::physical_plan::SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context.clone())?;
        let schema = self.schema();
        let query = self.query.clone();
        let ds = self.dataset.clone();
        let column = self.query.column.clone();
        let indices = self.indices.clone();
        let prefilter_source = self.prefilter_source.clone();
        let metrics = Arc::new(AnnMetrics::new(&self.metrics, partition));
        let metrics_clone = metrics.clone();
        // Per-delta-index stream:
        //   Stream<(parttitions, index uuid)>
        let per_index_stream = input_stream
            .and_then(move |batch| {
                let part_id_col = batch
                    .column_by_name(PART_ID_COLUMN)
                    .expect("ANNSubIndexExec: input missing part_id column");
                let part_id_arr = part_id_col.as_list::<i32>().clone();
                let index_uuid_col = batch
                    .column_by_name(INDEX_UUID_COLUMN)
                    .expect("ANNSubIndexExec: input missing index_uuid column");
                let index_uuid = index_uuid_col.as_string::<i32>().clone();

                let plan = part_id_arr
                    .iter()
                    .zip(index_uuid.iter())
                    .map(|(part_id, uuid)| {
                        // TODO: eliminate exceesive copying here to fight with lifetime.
                        let partitions = part_id
                            .unwrap()
                            .as_primitive::<UInt32Type>()
                            .values()
                            .to_vec();
                        let uuid = uuid.unwrap().to_string();
                        Ok((partitions, uuid))
                    })
                    .collect_vec();
                async move { Ok(stream::iter(plan)) }
            })
            .try_flatten();
        let prefilter_loader = match &prefilter_source {
            PreFilterSource::FilteredRowIds(src_node) => {
                let stream = src_node.execute(partition, context)?;
                Some(Box::new(FilteredRowIdsToPrefilter(stream)) as Box<dyn FilterLoader>)
            }
            PreFilterSource::ScalarIndexQuery(src_node) => {
                let stream = src_node.execute(partition, context)?;
                Some(Box::new(SelectionVectorToPrefilter(stream)) as Box<dyn FilterLoader>)
            }
            PreFilterSource::None => None,
        };

        let pre_filter = Arc::new(DatasetPreFilter::new(
            ds.clone(),
            &indices,
            prefilter_loader,
        ));

        Ok(Box::pin(InstrumentedRecordBatchStreamAdapter::new(
            schema,
            per_index_stream
                .and_then(move |(part_ids, index_uuid)| {
                    let ds = ds.clone();
                    let column = column.clone();
                    let metrics = metrics.clone();
                    let pre_filter = pre_filter.clone();

                    async move {
                        let raw_index = ds
                            .open_vector_index(&column, &index_uuid, &metrics.index_metrics)
                            .await?;

                        Ok::<_, DataFusionError>(
                            stream::iter(part_ids)
                                .zip(repeat_with(move || (raw_index.clone(), pre_filter.clone())))
                                .map(Ok::<_, DataFusionError>),
                        )
                    }
                })
                .try_flatten()
                .map(move |result| {
                    let query = query.clone();
                    let metrics = metrics_clone.clone();
                    async move {
                        let (part_id, (index, pre_filter)) = result?;

                        let mut query = query.clone();
                        if index.metric_type() == DistanceType::Cosine {
                            let key = normalize_arrow(&query.key)?;
                            query.key = key;
                        };

                        index
                            .search_in_partition(
                                part_id as usize,
                                &query,
                                pre_filter,
                                &metrics.index_metrics,
                            )
                            .map_err(|e| {
                                DataFusionError::Execution(format!(
                                    "Failed to calculate KNN: {}",
                                    e
                                ))
                            })
                            .await
                    }
                })
                .buffered(get_num_compute_intensive_cpus())
                .boxed(),
            partition,
            &self.metrics,
        )))
    }

    fn statistics(&self) -> DataFusionResult<datafusion::physical_plan::Statistics> {
        Ok(Statistics {
            num_rows: Precision::Exact(
                self.query.k
                    * self.query.refine_factor.unwrap_or(1) as usize
                    * self.input.statistics()?.num_rows.get_value().unwrap_or(&1),
            ),
            ..Statistics::new_unknown(self.schema().as_ref())
        })
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
}

#[derive(Debug)]
pub struct MultivectorScoringExec {
    // the inputs are sorted ANN search results
    inputs: Vec<Arc<dyn ExecutionPlan>>,
    query: Query,
    properties: PlanProperties,
}

impl MultivectorScoringExec {
    pub fn try_new(inputs: Vec<Arc<dyn ExecutionPlan>>, query: Query) -> Result<Self> {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(KNN_INDEX_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Ok(Self {
            inputs,
            query,
            properties,
        })
    }
}

impl DisplayAs for MultivectorScoringExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "MultivectorScoring: k={}", self.query.k)
            }
            DisplayFormatType::TreeRender => {
                write!(f, "MultivectorScoring\nk={}", self.query.k)
            }
        }
    }
}

impl ExecutionPlan for MultivectorScoringExec {
    fn name(&self) -> &str {
        "MultivectorScoringExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        KNN_INDEX_SCHEMA.clone()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inputs.iter().collect()
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        // This node fully consumes and re-orders the input rows.  It must be
        // run on a single partition.
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let plan = Self::try_new(children, self.query.clone())?;
        Ok(Arc::new(plan))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::context::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let inputs = self
            .inputs
            .iter()
            .map(|input| input.execute(partition, context.clone()))
            .collect::<DataFusionResult<Vec<_>>>()?;

        // collect the top k results from each stream,
        // and max-reduce for each query,
        // records the minimum distance for each query as estimation.
        let mut reduced_inputs = stream::select_all(inputs.into_iter().map(|stream| {
            stream.map(|batch| {
                let batch = batch?;
                let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
                let dists = batch[DIST_COL].as_primitive::<Float32Type>();
                debug_assert_eq!(dists.null_count(), 0);

                // max-reduce for the same row id
                let min_sim = dists
                    .values()
                    .last()
                    .map(|dist| 1.0 - *dist)
                    .unwrap_or_default();
                let mut new_row_ids = Vec::with_capacity(row_ids.len());
                let mut new_sims = Vec::with_capacity(row_ids.len());
                let mut visited_row_ids = HashSet::with_capacity(row_ids.len());

                for (row_id, dist) in row_ids.values().iter().zip(dists.values().iter()) {
                    // the results are sorted by distance, so we can skip if we have seen this row id before
                    if visited_row_ids.contains(row_id) {
                        continue;
                    }
                    visited_row_ids.insert(row_id);
                    new_row_ids.push(*row_id);
                    // it's cosine distance, so we need to convert it to similarity
                    new_sims.push(1.0 - *dist);
                }
                let new_row_ids = UInt64Array::from(new_row_ids);
                let new_dists = Float32Array::from(new_sims);

                let batch = RecordBatch::try_new(
                    KNN_INDEX_SCHEMA.clone(),
                    vec![Arc::new(new_dists), Arc::new(new_row_ids)],
                )?;

                Ok::<_, DataFusionError>((min_sim, batch))
            })
        }));

        let k = self.query.k;
        let refactor = self.query.refine_factor.unwrap_or(1) as usize;
        let num_queries = self.inputs.len() as f32;
        let stream = stream::once(async move {
            // at most, we will have k * refine_factor results for each query
            let mut results = HashMap::with_capacity(k * refactor);
            let mut missed_sim_sum = 0.0;
            while let Some((min_sim, batch)) = reduced_inputs.try_next().await? {
                let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
                let sims = batch[DIST_COL].as_primitive::<Float32Type>();

                let query_results = row_ids
                    .values()
                    .iter()
                    .copied()
                    .zip(sims.values().iter().copied())
                    .collect::<HashMap<_, _>>();

                // for a row `r`:
                // if `r` is in only `results``, then `results[r] += min_sim`
                // if `r` is in only `query_results`, then `results[r] = query_results[r] + missed_similarities`,
                // here `missed_similarities` is the sum of `min_sim` from previous iterations
                // if `r` is in both, then `results[r] += query_results[r]`
                results.iter_mut().for_each(|(row_id, sim)| {
                    if let Some(new_dist) = query_results.get(row_id) {
                        *sim += new_dist;
                    } else {
                        *sim += min_sim;
                    }
                });
                query_results.into_iter().for_each(|(row_id, sim)| {
                    results.entry(row_id).or_insert(sim + missed_sim_sum);
                });
                missed_sim_sum += min_sim;
            }

            let (row_ids, sims): (Vec<_>, Vec<_>) = results.into_iter().unzip();
            let dists = sims
                .into_iter()
                // it's similarity, so we need to convert it back to distance
                .map(|sim| num_queries - sim)
                .collect::<Vec<_>>();
            let row_ids = UInt64Array::from(row_ids);
            let dists = Float32Array::from(dists);
            let batch = RecordBatch::try_new(
                KNN_INDEX_SCHEMA.clone(),
                vec![Arc::new(dists), Arc::new(row_ids)],
            )?;
            Ok::<_, DataFusionError>(batch)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream.boxed(),
        )))
    }

    fn statistics(&self) -> DataFusionResult<Statistics> {
        Ok(Statistics {
            num_rows: Precision::Inexact(
                self.query.k * self.query.refine_factor.unwrap_or(1) as usize,
            ),
            ..Statistics::new_unknown(self.schema().as_ref())
        })
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::compute::{concat_batches, sort_to_indices, take_record_batch};
    use arrow::datatypes::Float32Type;
    use arrow_array::{FixedSizeListArray, Int32Array, RecordBatchIterator, StringArray};
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use lance_testing::datagen::generate_random_array;
    use tempfile::tempdir;

    use crate::dataset::WriteParams;
    use crate::io::exec::testing::TestingExec;

    #[tokio::test]
    async fn knn_flat_search() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("key", DataType::Int32, false),
            ArrowField::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    128,
                ),
                true,
            ),
            ArrowField::new("uri", DataType::Utf8, true),
        ]));

        let batches: Vec<RecordBatch> = (0..20)
            .map(|i| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                        Arc::new(
                            FixedSizeListArray::try_new_from_values(
                                generate_random_array(128 * 20),
                                128,
                            )
                            .unwrap(),
                        ),
                        Arc::new(StringArray::from_iter_values(
                            (i * 20..(i + 1) * 20).map(|i| format!("s3://bucket/file-{}", i)),
                        )),
                    ],
                )
                .unwrap()
            })
            .collect();

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            ..Default::default()
        };
        let vector_arr = batches[0].column_by_name("vector").unwrap();
        let q = as_fixed_size_list_array(&vector_arr).value(5);

        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let stream = dataset
            .scan()
            .nearest("vector", q.as_primitive::<Float32Type>(), 10)
            .unwrap()
            .try_into_stream()
            .await
            .unwrap();
        let results = stream.try_collect::<Vec<_>>().await.unwrap();

        assert!(results[0].schema().column_with_name(DIST_COL).is_some());

        assert_eq!(results.len(), 1);

        let stream = dataset.scan().try_into_stream().await.unwrap();
        let all_with_distances = stream
            .and_then(|batch| compute_distance(q.clone(), DistanceType::L2, "vector", batch))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let all_with_distances =
            concat_batches(&results[0].schema(), all_with_distances.iter()).unwrap();
        let dist_arr = all_with_distances.column_by_name(DIST_COL).unwrap();
        let distances = dist_arr.as_primitive::<Float32Type>();
        let indices = sort_to_indices(distances, None, Some(10)).unwrap();
        let expected = take_record_batch(&all_with_distances, &indices).unwrap();
        assert_eq!(expected, results[0]);
    }

    #[test]
    fn test_create_knn_flat() {
        let dim: usize = 128;
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("key", DataType::Int32, false),
            ArrowField::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                true,
            ),
            ArrowField::new("uri", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::new_empty(schema);

        let input: Arc<dyn ExecutionPlan> = Arc::new(TestingExec::new(vec![batch]));

        let idx = KNNVectorDistanceExec::try_new(
            input,
            "vector",
            Arc::new(generate_random_array(dim)),
            DistanceType::L2,
        )
        .unwrap();
        assert_eq!(
            idx.schema().as_ref(),
            &ArrowSchema::new(vec![
                ArrowField::new("key", DataType::Int32, false),
                ArrowField::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(ArrowField::new("item", DataType::Float32, true)),
                        dim as i32,
                    ),
                    true,
                ),
                ArrowField::new("uri", DataType::Utf8, true),
                ArrowField::new(DIST_COL, DataType::Float32, true),
            ])
        );
    }

    #[tokio::test]
    async fn test_multivector_score() {
        let query = Query {
            column: "vector".to_string(),
            key: Arc::new(generate_random_array(1)),
            k: 10,
            lower_bound: None,
            upper_bound: None,
            nprobes: 1,
            ef: None,
            refine_factor: None,
            metric_type: DistanceType::Cosine,
            use_index: true,
        };

        async fn multivector_scoring(
            inputs: Vec<Arc<dyn ExecutionPlan>>,
            query: Query,
        ) -> Result<HashMap<u64, f32>> {
            let ctx = Arc::new(datafusion::execution::context::TaskContext::default());
            let plan = MultivectorScoringExec::try_new(inputs, query.clone())?;
            let batches = plan
                .execute(0, ctx.clone())
                .unwrap()
                .try_collect::<Vec<_>>()
                .await?;
            let mut results = HashMap::new();
            for batch in batches {
                let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
                let dists = batch[DIST_COL].as_primitive::<Float32Type>();
                for (row_id, dist) in row_ids.values().iter().zip(dists.values().iter()) {
                    results.insert(*row_id, *dist);
                }
            }
            Ok(results)
        }

        let batches = (0..3)
            .map(|i| {
                RecordBatch::try_new(
                    KNN_INDEX_SCHEMA.clone(),
                    vec![
                        Arc::new(Float32Array::from(vec![i as f32 + 1.0, i as f32 + 2.0])),
                        Arc::new(UInt64Array::from(vec![i + 1, i + 2])),
                    ],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let mut res: Option<HashMap<_, _>> = None;
        for perm in batches.into_iter().permutations(3) {
            let inputs = perm
                .into_iter()
                .map(|batch| {
                    let input: Arc<dyn ExecutionPlan> = Arc::new(TestingExec::new(vec![batch]));
                    input
                })
                .collect::<Vec<_>>();
            let new_res = multivector_scoring(inputs, query.clone()).await.unwrap();
            assert_eq!(new_res.len(), 4);
            if let Some(res) = &res {
                for (row_id, dist) in new_res.iter() {
                    assert_eq!(res.get(row_id).unwrap(), dist)
                }
            } else {
                res = Some(new_res);
            }
        }
    }
}
