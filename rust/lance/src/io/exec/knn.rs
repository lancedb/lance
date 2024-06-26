// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::sync::Arc;

use arrow::datatypes::UInt32Type;
use arrow_array::{
    builder::{ListBuilder, UInt32Builder},
    cast::AsArray,
    ArrayRef, RecordBatch, StringArray, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::common::stats::Precision;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::physical_plan::{
    stream::RecordBatchStreamAdapter, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning,
    SendableRecordBatchStream, Statistics,
};
use datafusion::physical_plan::{ExecutionMode, PlanProperties};
use datafusion_physical_expr::EquivalenceProperties;
use futures::stream::repeat_with;
use futures::{future, stream, StreamExt, TryFutureExt, TryStreamExt};
use itertools::Itertools;
use lance_core::utils::mask::{RowIdMask, RowIdTreeMap};
use lance_core::{ROW_ID, ROW_ID_FIELD};
use lance_index::vector::{flat::flat_search, Query, DIST_COL, INDEX_UUID_COLUMN, PART_ID_COLUMN};
use lance_linalg::distance::DistanceType;
use lance_linalg::kernels::normalize_arrow;
use lance_table::format::Index;
use snafu::{location, Location};

use crate::dataset::Dataset;
use crate::index::prefilter::{DatasetPreFilter, FilterLoader};
use crate::index::DatasetIndexInternalExt;
use crate::{Error, Result};
use lance_arrow::*;

/// Check vector column exists and has the correct data type.
fn check_vector_column(schema: &Schema, column: &str) -> Result<()> {
    let field = schema.field_with_name(column).map_err(|_| {
        Error::io(
            format!("Query column '{}' not found in input schema", column),
            location!(),
        )
    })?;
    match field.data_type() {
        DataType::FixedSizeList(list_field, _)
            if matches!(
                list_field.data_type(),
                DataType::UInt8 | DataType::Float16 | DataType::Float32 | DataType::Float64
            ) => Ok(()),
        _ => {
           Err(Error::io(
                format!(
                    "KNNFlatExec node: query column {} is not a vector. Expect FixedSizeList<Float32>, got {}",
                    column, field.data_type()
                ),
                location!(),
            ))
        }
    }
}

/// [ExecutionPlan] for Flat KNN (bruteforce) search.
///
/// Preconditions:
/// - `input` schema must contains `query.column`,
/// - The column must be a vector.
/// - `input` schema does not have "_distance" column.
#[derive(Debug)]
pub struct KNNFlatExec {
    /// Inner input node.
    pub input: Arc<dyn ExecutionPlan>,

    /// The vector query to execute.
    pub query: ArrayRef,
    column: String,
    distance_type: DistanceType,

    output_schema: SchemaRef,
    properties: PlanProperties,
}

impl DisplayAs for KNNFlatExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "KNNFlat: metric={}", self.distance_type,)
            }
        }
    }
}

impl KNNFlatExec {
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
        check_vector_column(&output_schema, column)?;

        // FlatExec appends a distance column to the input schema. The input
        // may already have a distance column (possibly in the wrong position), so
        // we need to remove it before adding a new one.
        if output_schema.column_with_name(DIST_COL).is_some() {
            output_schema = output_schema.without_column(DIST_COL);
        }
        let output_schema = Arc::new(Schema::new(
            output_schema
                .try_with_column(Field::new(DIST_COL, DataType::Float32, true))
                .unwrap()
                .fields,
        ));

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
        })
    }
}

impl ExecutionPlan for KNNFlatExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Flat KNN inherits the schema from input node, and add one distance column.
    fn schema(&self) -> arrow_schema::SchemaRef {
        self.output_schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "KNNFlatExec node must have exactly one child".to_string(),
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
                    flat_search(key, dt, &column, batch?)
                        .await
                        .map_err(|e| DataFusionError::Execution(e.to_string()))
                }
            })
            .buffer_unordered(num_cpus::get());
        let schema = self.schema();
        Ok(
            Box::pin(RecordBatchStreamAdapter::new(schema, stream.boxed()))
                as SendableRecordBatchStream,
        )
    }

    fn statistics(&self) -> DataFusionResult<Statistics> {
        self.input.statistics()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }
}

// Utility to convert an input (containing row ids) into a prefilter
struct FilteredRowIdsToPrefilter(SendableRecordBatchStream);

#[async_trait]
impl FilterLoader for FilteredRowIdsToPrefilter {
    async fn load(mut self: Box<Self>) -> Result<RowIdMask> {
        let mut allow_list = RowIdTreeMap::new();
        while let Some(batch) = self.0.next().await {
            let batch = batch?;
            let row_ids = batch.column_by_name(ROW_ID).expect(
                "input batch missing row id column even though it is in the schema for the stream",
            );
            let row_ids = row_ids
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("row id column in input batch had incorrect type");
            allow_list.extend(row_ids.iter().flatten())
        }
        Ok(RowIdMask::from_allowed(allow_list))
    }
}

// Utility to convert a serialized selection vector into a prefilter
struct SelectionVectorToPrefilter(SendableRecordBatchStream);

#[async_trait]
impl FilterLoader for SelectionVectorToPrefilter {
    async fn load(mut self: Box<Self>) -> Result<RowIdMask> {
        let batch = self
            .0
            .try_next()
            .await?
            .ok_or_else(|| Error::Internal {
                message: "Selection vector source for prefilter did not yield any batches".into(),
                location: location!(),
            })
            .unwrap();
        RowIdMask::from_arrow(batch["result"].as_binary_opt::<i32>().ok_or_else(|| {
            Error::Internal {
                message: format!(
                    "Expected selection vector input to yield binary arrays but got {}",
                    batch["result"].data_type()
                ),
                location: location!(),
            }
        })?)
    }
}

#[derive(Debug, Clone)]
pub enum PreFilterSource {
    /// The prefilter input is an array of row ids that match the filter condition
    FilteredRowIds(Arc<dyn ExecutionPlan>),
    /// The prefilter input is a selection vector from an index query
    ScalarIndexQuery(Arc<dyn ExecutionPlan>),
    /// There is no prefilter
    None,
}

lazy_static::lazy_static! {
    pub static ref KNN_INDEX_SCHEMA: SchemaRef = Arc::new(Schema::new(vec![
        Field::new(DIST_COL, DataType::Float32, true),
        ROW_ID_FIELD.clone(),
    ]));

    static ref KNN_PARTITION_SCHEMA: SchemaRef = Arc::new(Schema::new(vec![
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
        Arc::new(indices.to_vec()),
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
    dataset: Arc<Dataset>,

    /// The vector query to execute.
    query: Query,

    /// The UUIDs of the indices to search.
    index_uuids: Vec<String>,

    properties: PlanProperties,
}

impl ANNIvfPartitionExec {
    pub fn try_new(dataset: Arc<Dataset>, index_uuids: Vec<String>, query: Query) -> Result<Self> {
        let dataset_schema = dataset.schema();
        check_vector_column(&dataset_schema.into(), &query.column)?;
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
            ExecutionMode::Bounded,
        );

        Ok(Self {
            dataset,
            query,
            index_uuids,
            properties,
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
        }
    }
}

impl ExecutionPlan for ANNIvfPartitionExec {
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

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Err(DataFusionError::Internal(
            "ANNIVFPartitionExec: with_new_children called, but no children to replace".to_string(),
        ))
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let query = self.query.clone();
        let ds = self.dataset.clone();

        let stream = stream::iter(self.index_uuids.clone())
            .map(move |uuid| {
                let query = query.clone();
                let ds = ds.clone();

                async move {
                    let index = ds.open_vector_index(&query.column, &uuid).await?;

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
        Ok(
            Box::pin(RecordBatchStreamAdapter::new(schema, stream.boxed()))
                as SendableRecordBatchStream,
        )
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

    indices: Arc<Vec<Index>>,

    /// Vector Query.
    query: Query,

    /// Prefiltering input
    prefilter_source: PreFilterSource,

    /// Datafusion Plan Properties
    properties: PlanProperties,
}

impl ANNIvfSubIndexExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        dataset: Arc<Dataset>,
        indices: Arc<Vec<Index>>,
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
            ExecutionMode::Bounded,
        );
        Ok(Self {
            input,
            dataset,
            indices,
            query,
            prefilter_source,
            properties,
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
        }
    }
}

impl ExecutionPlan for ANNIvfSubIndexExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        KNN_INDEX_SCHEMA.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        match &self.prefilter_source {
            PreFilterSource::None => vec![self.input.clone()],
            PreFilterSource::FilteredRowIds(src) => vec![self.input.clone(), src.clone()],
            PreFilterSource::ScalarIndexQuery(src) => vec![self.input.clone(), src.clone()],
        }
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "ANNSubIndexExec node must have exactly one child".to_string(),
            ));
        }

        let new_plan = Self {
            input: children.pop().expect("length checked"),
            dataset: self.dataset.clone(),
            indices: self.indices.clone(),
            query: self.query.clone(),
            prefilter_source: self.prefilter_source.clone(),
            properties: self.properties.clone(),
        };

        Ok(Arc::new(new_plan))
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

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            per_index_stream
                .and_then(move |(part_ids, index_uuid)| {
                    let ds = ds.clone();
                    let column = column.clone();
                    let indices = indices.clone();
                    let context = context.clone();
                    let prefilter_source = prefilter_source.clone();

                    let index_meta = indices
                        .iter()
                        .find(|idx| idx.uuid.to_string() == index_uuid)
                        .unwrap()
                        .clone();

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

                        let raw_index = ds.open_vector_index(&column, &index_uuid).await?;

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
                    async move {
                        let (part_id, (index, pre_filter)) = result?;

                        let mut query = query.clone();
                        if index.metric_type() == DistanceType::Cosine {
                            let key = normalize_arrow(&query.key)?;
                            query.key = key;
                        };

                        index
                            .search_in_partition(part_id as usize, &query, pre_filter)
                            .map_err(|e| {
                                DataFusionError::Execution(format!(
                                    "Failed to calculate KNN: {}",
                                    e
                                ))
                            })
                            .await
                    }
                })
                .buffered(num_cpus::get())
                .boxed(),
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
            .nearest("vector", q.as_primitive(), 10)
            .unwrap()
            .try_into_stream()
            .await
            .unwrap();
        let results = stream.try_collect::<Vec<_>>().await.unwrap();

        assert!(results[0].schema().column_with_name(DIST_COL).is_some());

        assert_eq!(results.len(), 1);

        let stream = dataset.scan().try_into_stream().await.unwrap();
        let all_with_distances = stream
            .and_then(|batch| flat_search(q.clone(), DistanceType::L2, "vector", batch))
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

        let idx = KNNFlatExec::try_new(
            input,
            "vector",
            Arc::new(generate_random_array(dim)),
            DistanceType::L2,
        )
        .unwrap();
        println!("{:?}", idx);
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
}
