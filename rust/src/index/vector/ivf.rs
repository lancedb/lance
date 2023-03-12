// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! IVF - Inverted File index.

use std::sync::Arc;

use arrow_arith::{
    aggregate::{max, min},
    arithmetic::subtract_dyn,
};
use arrow_array::{
    builder::Float32Builder,
    cast::{as_primitive_array, as_struct_array},
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, RecordBatch, StructArray,
    UInt32Array,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::{
    concat::{concat, concat_batches},
    filter::filter_record_batch,
    take::take,
};
use async_trait::async_trait;
use futures::{
    stream::{self, StreamExt},
    TryStreamExt,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use uuid::Uuid;

use super::{
    pq::{PQIndex, ProductQuantizer},
    MetricType, Query, Transformer, VectorIndex,
};
use crate::arrow::{linalg::MatrixView, *};
use crate::index::vector::opq::*;
use crate::io::{
    object_reader::{read_message, ObjectReader},
    read_message_from_buf, read_metadata_offset,
};
use crate::{
    dataset::{scanner::Scanner, Dataset, ROW_ID},
    index::{pb, pb::vector_index_stage::Stage, IndexBuilder, IndexType},
};
use crate::{Error, Result};

const INDEX_FILE_NAME: &str = "index.idx";
const PARTITION_ID_COLUMN: &str = "__ivf_part_id";
const RESIDUAL_COLUMN: &str = "__residual_vector";

/// IVF PQ Index.
pub struct IvfPQIndex<'a> {
    reader: Box<dyn ObjectReader + 'a>,

    /// Ivf file.
    ivf: Ivf,

    /// Number of bits used for product quantization centroids.
    pq: Arc<ProductQuantizer>,

    metric_type: MetricType,
}

impl<'a> IvfPQIndex<'a> {
    /// Open the IvfPQ index on dataset, specified by the index `name`.
    pub async fn new(dataset: &'a Dataset, uuid: &str) -> Result<IvfPQIndex<'a>> {
        let index_dir = dataset.indices_dir().child(uuid);
        let index_file = index_dir.child(INDEX_FILE_NAME);

        let object_store = dataset.object_store();
        let reader = object_store.open(&index_file).await?;

        let file_size = reader.size().await?;
        let prefetch_size = object_store.prefetch_size();
        let begin = if file_size < prefetch_size {
            0
        } else {
            file_size - prefetch_size
        };
        let tail_bytes = reader.get_range(begin..file_size).await?;
        let metadata_pos = read_metadata_offset(&tail_bytes)?;
        let proto: pb::Index = if metadata_pos < file_size - tail_bytes.len() {
            // We have not read the metadata bytes yet.
            read_message(reader.as_ref(), metadata_pos).await?
        } else {
            let offset = tail_bytes.len() - (file_size - metadata_pos);
            read_message_from_buf(&tail_bytes.slice(offset..))?
        };
        let index_metadata = IvfPQIndexMetadata::try_from(&proto)?;

        Ok(Self {
            reader,
            ivf: index_metadata.ivf,
            pq: index_metadata.pq,
            metric_type: index_metadata.metric_type,
        })
    }

    async fn search_in_partition(
        &self,
        partition_id: usize,
        key: &Float32Array,
        k: usize,
    ) -> Result<RecordBatch> {
        let offset = self.ivf.offsets[partition_id];
        let length = self.ivf.lengths[partition_id] as usize;
        let partition_centroids = self.ivf.centroids.value(partition_id);
        let residual_key = subtract_dyn(key, &partition_centroids)?;

        // TODO: Keep PQ index in LRU
        let pq_index = PQIndex::load(
            self.reader.as_ref(),
            self.pq.as_ref(),
            self.metric_type,
            offset,
            length,
        )
        .await?;
        pq_index.search(as_primitive_array(&residual_key), k)
    }
}

#[async_trait]
impl VectorIndex for IvfPQIndex<'_> {
    async fn search(&self, query: &Query) -> Result<RecordBatch> {
        let partition_ids = self
            .ivf
            .find_partitions(&query.key, query.nprobs, self.metric_type)?;
        let candidates = stream::iter(partition_ids.values())
            .then(|part_id| async move {
                self.search_in_partition(*part_id as usize, &query.key, query.k)
                    .await
            })
            .collect::<Vec<_>>()
            .await;
        let mut batches = vec![];
        for b in candidates {
            batches.push(b?);
        }
        let batch = concat_batches(&batches[0].schema(), &batches)?;

        let score_col = batch.column_by_name("score").ok_or_else(|| {
            Error::IO(format!(
                "score column does not exist in batch: {}",
                batch.schema()
            ))
        })?;
        let refined_index = sort_to_indices(score_col, None, Some(query.k))?;

        let struct_arr = StructArray::from(batch);
        let taken_scores = take(&struct_arr, &refined_index, None)?;
        Ok(as_struct_array(&taken_scores).into())
    }
}

/// Ivf PQ index metadata.
///
/// It contains the on-disk data for a IVF PQ index.
#[derive(Debug)]
pub struct IvfPQIndexMetadata {
    /// Index name
    name: String,

    /// The column to build the index for.
    column: String,

    /// Vector dimension.
    dimension: u32,

    /// The version of dataset where this index was built.
    dataset_version: u64,

    metric_type: MetricType,

    // Ivf related
    ivf: Ivf,

    /// Product Quantizer
    pq: Arc<ProductQuantizer>,
}

/// Convert a IvfPQIndex to protobuf payload
impl TryFrom<&IvfPQIndexMetadata> for pb::Index {
    type Error = Error;

    fn try_from(idx: &IvfPQIndexMetadata) -> std::result::Result<Self, Self::Error> {
        Ok(Self {
            name: idx.name.clone(),
            columns: vec![idx.column.clone()],
            dataset_version: idx.dataset_version,
            index_type: pb::IndexType::Vector.into(),
            implementation: Some(pb::index::Implementation::VectorIndex(pb::VectorIndex {
                spec_version: 1,
                dimension: idx.dimension,
                stages: vec![
                    pb::VectorIndexStage {
                        stage: Some(pb::vector_index_stage::Stage::Ivf(pb::Ivf::try_from(
                            &idx.ivf,
                        )?)),
                    },
                    pb::VectorIndexStage {
                        stage: Some(pb::vector_index_stage::Stage::Pq(idx.pq.as_ref().into())),
                    },
                ],
                metric_type: match idx.metric_type {
                    MetricType::L2 => pb::VectorMetricType::L2.into(),
                    MetricType::Cosine => pb::VectorMetricType::Cosine.into(),
                },
            })),
        })
    }
}

impl TryFrom<&pb::Index> for IvfPQIndexMetadata {
    type Error = Error;

    fn try_from(idx: &pb::Index) -> Result<Self> {
        if idx.columns.len() != 1 {
            return Err(Error::Schema("IVF_PQ only supports 1 column".to_string()));
        }
        assert_eq!(idx.index_type, pb::IndexType::Vector as i32);

        let metadata =
            if let Some(idx_impl) = idx.implementation.as_ref() {
                match idx_impl {
                    pb::index::Implementation::VectorIndex(vidx) => {
                        if vidx.stages.len() != 2 {
                            return Err(Error::IO("Only support IVF_PQ now".to_string()));
                        };
                        let stage0 = vidx.stages[0].stage.as_ref().ok_or_else(|| {
                            Error::IO("VectorIndex stage 0 is missing".to_string())
                        })?;
                        let ivf = match stage0 {
                            Stage::Ivf(ivf_pb) => Ok(Ivf::try_from(ivf_pb)?),
                            _ => Err(Error::IO("Stage 0 only supports IVF".to_string())),
                        }?;
                        let stage1 = vidx.stages[1].stage.as_ref().ok_or_else(|| {
                            Error::IO("VectorIndex stage 0 is missing".to_string())
                        })?;
                        let pq = match stage1 {
                            Stage::Pq(pq_proto) => Ok(Arc::new(pq_proto.into())),
                            _ => Err(Error::IO("Stage 1 only supports PQ".to_string())),
                        }?;

                        Ok::<Self, Error>(Self {
                            name: idx.name.clone(),
                            column: idx.columns[0].clone(),
                            dimension: vidx.dimension,
                            dataset_version: idx.dataset_version,
                            metric_type: pb::VectorMetricType::from_i32(vidx.metric_type)
                                .ok_or(Error::Index(format!(
                                    "Unsupported metric type value: {}",
                                    vidx.metric_type
                                )))?
                                .into(),
                            ivf,
                            pq,
                        })
                    }
                }?
            } else {
                return Err(Error::IO("Invalid protobuf".to_string()));
            };
        Ok(metadata)
    }
}

fn compute_residual(
    centroids: Arc<FixedSizeListArray>,
    vector_array: &FixedSizeListArray,
    partition_ids: &UInt32Array,
) -> Result<ArrayRef> {
    let mut residual_builder = Float32Builder::new();
    for i in 0..vector_array.len() {
        let vector = vector_array.value(i);
        let centroids = centroids.value(partition_ids.value(i) as usize);
        let residual_vector = subtract_dyn(vector.as_ref(), centroids.as_ref())?;
        let residual_float32: &Float32Array = as_primitive_array(residual_vector.as_ref());
        residual_builder.append_slice(residual_float32.values());
    }
    let values = residual_builder.finish();
    Ok(Arc::new(FixedSizeListArray::try_new(
        values,
        vector_array.value_length(),
    )?))
}

/// Ivf Model
#[derive(Debug)]
struct Ivf {
    /// Centroids of each partition.
    ///
    /// It is a 2-D `(num_partitions * dimension)` of float32 array, 64-bit aligned via Arrow
    /// memory allocator.
    centroids: Arc<FixedSizeListArray>,

    /// Offset of each partition in the file.
    offsets: Vec<usize>,

    /// Number of vectors in each partition.
    lengths: Vec<u32>,
}

impl Ivf {
    fn new(centroids: Arc<FixedSizeListArray>) -> Self {
        Self {
            centroids,
            offsets: vec![],
            lengths: vec![],
        }
    }

    /// Ivf model dimension.
    fn dimension(&self) -> usize {
        self.centroids.value_length() as usize
    }

    /// Use the query vector to find `nprobes` closest partitions.
    fn find_partitions(
        &self,
        query: &Float32Array,
        nprobes: usize,
        metric_type: MetricType,
    ) -> Result<UInt32Array> {
        if query.len() != self.dimension() {
            return Err(Error::IO(format!(
                "Ivf::find_partition: dimension mismatch: {} != {}",
                query.len(),
                self.dimension()
            )));
        }
        let dist_func = metric_type.func();
        let centroid_values = self.centroids.values();
        let distances = dist_func(
            query,
            as_primitive_array(centroid_values.as_ref()),
            self.dimension(),
        )? as ArrayRef;
        let top_k_partitions = sort_to_indices(&distances, None, Some(nprobes))?;
        Ok(top_k_partitions)
    }

    /// Add the offset and length of one partition.
    fn add_partition(&mut self, offset: usize, len: u32) {
        self.offsets.push(offset);
        self.lengths.push(len);
    }

    /// Scan the dataset and assign the partition ID for each row.
    ///
    /// Currently, it keeps batches in the memory.
    async fn partition(
        &self,
        scanner: &Scanner,
        metric_type: MetricType,
    ) -> Result<Vec<RecordBatch>> {
        let schema = scanner.schema()?;
        let column_name = schema.field(0).name();
        let batches_with_partition_id = scanner
            .try_into_stream()
            .await?
            .map(|b| async move {
                let batch = b?;
                let arr = batch.column_by_name(column_name).ok_or_else(|| {
                    Error::IO(format!("Dataset does not have column {column_name}"))
                })?;
                let vectors: MatrixView = as_fixed_size_list_array(arr).try_into()?;
                let centroids = self.centroids.as_ref().try_into()?;
                let partition_column = tokio::task::spawn_blocking(move || {
                    compute_residual_matrix(&vectors, &centroids, metric_type)
                })
                .await??;
                let batch_with_part_id = batch.try_with_column(
                    ArrowField::new(PARTITION_ID_COLUMN, DataType::UInt32, false),
                    partition_column,
                )?;
                Ok::<RecordBatch, Error>(batch_with_part_id)
            })
            .buffer_unordered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;

        // Compute the residual vectors for every RecordBatch.
        // let mut residual_batches = vec![];
        let residual_batches = stream::iter(batches_with_partition_id)
            .map(|batch| async move {
                let centorids = self.centroids.clone();
                let vector = batch.column_by_name(column_name).unwrap().clone();
                let partition_ids = batch.column_by_name(PARTITION_ID_COLUMN).unwrap().clone();
                let residual = tokio::task::spawn_blocking(move || {
                    compute_residual(
                        centorids.clone(),
                        as_fixed_size_list_array(vector.as_ref()),
                        as_primitive_array(partition_ids.as_ref()),
                    )
                })
                .await??;
                let residual_schema = Arc::new(ArrowSchema::new(vec![
                    ArrowField::new(RESIDUAL_COLUMN, residual.data_type().clone(), false),
                    ArrowField::new(PARTITION_ID_COLUMN, DataType::UInt32, false),
                    ArrowField::new(ROW_ID, DataType::UInt64, false),
                ]));
                let b = RecordBatch::try_new(
                    residual_schema,
                    vec![
                        residual,
                        batch.column_by_name(PARTITION_ID_COLUMN).unwrap().clone(),
                        batch.column_by_name(ROW_ID).unwrap().clone(),
                    ],
                )?;
                Ok::<RecordBatch, Error>(b)
            })
            .buffer_unordered(16)
            .try_collect::<Vec<_>>()
            .await?;
        Ok(residual_batches)
    }
}

/// Convert IvfModel to protobuf.
impl TryFrom<&Ivf> for pb::Ivf {
    type Error = Error;

    fn try_from(ivf: &Ivf) -> Result<Self> {
        if ivf.offsets.len() != ivf.centroids.len() {
            return Err(Error::IO("Ivf model has not been populated".to_string()));
        }
        let centroids_arr = ivf.centroids.values();
        let f32_centroids: &Float32Array = as_primitive_array(&centroids_arr);
        Ok(Self {
            centroids: f32_centroids.iter().map(|v| v.unwrap()).collect(),
            offsets: ivf.offsets.iter().map(|o| *o as u64).collect(),
            lengths: ivf.lengths.clone(),
        })
    }
}

/// Convert IvfModel to protobuf.
impl TryFrom<&pb::Ivf> for Ivf {
    type Error = Error;

    fn try_from(proto: &pb::Ivf) -> Result<Self> {
        let f32_centroids = Float32Array::from(proto.centroids.clone());
        let dimension = f32_centroids.len() / proto.offsets.len();
        let centroids = Arc::new(FixedSizeListArray::try_new(
            f32_centroids,
            dimension as i32,
        )?);
        Ok(Self {
            centroids,
            offsets: proto.offsets.iter().map(|o| *o as usize).collect(),
            lengths: proto.lengths.clone(),
        })
    }
}

pub struct IvfPqIndexBuilder {
    dataset: Arc<Dataset>,

    /// Unique id of the index.
    uuid: Uuid,

    /// Index name
    name: String,

    /// Vector column to search for.
    column: String,

    dimension: usize,

    /// Metric type.
    metric_type: MetricType,

    /// Number of IVF partitions.
    num_partitions: u32,

    // PQ parameters
    nbits: u32,

    num_sub_vectors: u32,

    /// Max iterations to train a k-mean model.
    kmeans_max_iters: u32,
}

impl IvfPqIndexBuilder {
    pub async fn try_new(
        dataset: Arc<Dataset>,
        uuid: Uuid,
        name: &str,
        column: &str,
        num_partitions: u32,
        num_sub_vectors: u32,
        metric_type: MetricType,
    ) -> Result<Self> {
        let field = dataset.schema().field(column).ok_or(Error::IO(format!(
            "Column {column} does not exist in the dataset"
        )))?;
        let DataType::FixedSizeList(_, d) = field.data_type() else {
            return Err(Error::IO(format!("Column {column} is not a vector type")));
        };
        let num_rows = dataset.count_rows().await?;
        Ok(Self {
            dataset,
            uuid,
            name: name.to_string(),
            column: column.to_string(),
            dimension: d as usize,
            metric_type,
            num_partitions,
            num_sub_vectors,
            nbits: 8,
            kmeans_max_iters: 100,
        })
    }

    /// Train IVF partitions using kmeans.
    async fn train_ivf_model(&self) -> Result<Ivf> {
        let mut scanner = self.dataset.scan();
        scanner.project(&[&self.column])?;

        let rng = SmallRng::from_entropy();
        Ok(Ivf::new(
            train_kmeans_model(
                &scanner,
                self.dimension,
                self.num_partitions as usize,
                self.kmeans_max_iters,
                rng.clone(),
                self.metric_type,
            )
            .await?,
        ))
    }

    /// A guess of the sample size to train IVF / PQ / OPQ.
    fn sample_size_hint(&self) -> usize {
        let n_clusters = std::cmp::max(
            self.num_partitions as usize,
            ProductQuantizer::num_centroids(self.nbits),
        );
        n_clusters * 256
    }
}

fn sanity_check(dataset: &Dataset, column: &str) -> Result<()> {
    let Some(field) = dataset.schema().field(column) else {
        return Err(Error::IO(format!(
            "Building index: column {} does not exist in dataset: {:?}",
            column, dataset
        )));
    };
    if let DataType::FixedSizeList(elem_type, _) = field.data_type() {
        if !matches!(elem_type.data_type(), DataType::Float32) {
            return Err(
        Error::Index(
            format!("VectorIndex requires the column data type to be fixed size list of float32s, got {}",
            elem_type.data_type())));
        }
    } else {
        return Err(Error::Index(format!(
            "VectorIndex requires the column data type to be fixed size list of float32s, got {}",
            field.data_type()
        )));
    }
    Ok(())
}

/// Parameters to build IVF partitions
pub struct IvfBuildParams {
    /// Number of partitions to build.
    pub num_partitions: usize,

    /// Metric type, L2 or Cosine.
    pub metric_type: MetricType,

    // ---- kmeans parameters
    /// Max number of iterations to train kmeans.
    pub max_iters: usize,
}

/// Parameters for building product quantization.
pub struct PQBuildParams {
    /// Number of subvectors to build PQ code
    pub num_sub_vectors: usize,

    /// The number of bits to present one PQ centroid.
    pub num_bits: usize,

    /// Metric type, L2 or Cosine.
    pub metric_type: MetricType,

    /// Train as optimized product quantization.
    pub use_opq: bool,

    /// The max number of iterations for kmeans training.
    pub max_iters: usize,
}

async fn maybe_sample_training_data(
    dataset: &Dataset,
    column: &str,
    sample_size_hint: usize,
) -> Result<MatrixView> {
    let num_rows = dataset.count_rows().await?;
    let projection = dataset.schema().project(&[&column])?;
    let batch = if num_rows > sample_size_hint {
        dataset.sample(sample_size_hint, &projection).await?
    } else {
        let mut scanner = dataset.scan();
        scanner.project(&[column])?;
        let batches = scanner
            .try_into_stream()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        concat_batches(&Arc::new(ArrowSchema::from(&projection)), &batches)?
    };
    let array = batch.column_by_name(column).ok_or(Error::Index(format!(
        "Sample training data: column {} does not exist in return",
        column
    )))?;
    let fixed_size_array = as_fixed_size_list_array(array);
    fixed_size_array.try_into()
}

/// Compute residual matrix.
///
/// Parameters
/// - *data*: input matrix to compute residual.
/// - *centroids*: the centroids to compute residual vectors.
/// - *metric_type*: the metric type to compute distance.
fn compute_residual_matrix(
    data: &MatrixView,
    centroids: &MatrixView,
    metric_type: MetricType,
) -> Result<Arc<Float32Array>> {
    assert_eq!(centroids.num_columns(), data.num_columns());
    let dist_func = metric_type.func();

    let dim = data.num_columns();
    let mut builder = Float32Builder::with_capacity(data.data().len());
    for i in 0..data.num_rows() {
        let row = data.row(i).unwrap();
        let part_id = argmin(
            dist_func(&row, centroids.data().as_ref(), dim)
                .unwrap()
                .as_ref(),
        )
        .unwrap();
        let centroid = centroids.row(part_id as usize).unwrap();
        let residual = subtract_dyn(&row, &centroid)?;
        let f32_residual_array: &Float32Array = as_primitive_array(&residual);
        builder.append_slice(f32_residual_array.values());
    }
    Ok(Arc::new(builder.finish()))
}

/// Build IVF(PQ) index
pub async fn build_ivf_pq_index(
    dataset: &Dataset,
    column: &str,
    uuid: &Uuid,
    ivf_params: &IvfBuildParams,
    pq_params: &PQBuildParams,
) -> Result<()> {
    println!(
        "Building vector index: IVF{},PQ{}, metric={}",
        ivf_params.num_partitions, pq_params.num_sub_vectors, ivf_params.metric_type,
    );

    sanity_check(dataset, column)?;

    // Maximum to train 256 vectors per centroids, see Faiss.
    let sample_size_hint = std::cmp::max(
        ivf_params.num_partitions,
        ProductQuantizer::num_centroids(pq_params.num_bits as u32),
    ) * 256;

    let training_data = maybe_sample_training_data(dataset, column, sample_size_hint).await?;

    // Train the OPQ rotation matrix.
    let opq = train_opq(&training_data, pq_params).await?;

    // Transform training data using OPQ matrix.
    let training_data = opq.transform(&training_data).await?;

    // Train IVF partitions.
    let mut ivf_model = train_ivf_model(&training_data, ivf_params).await?;

    // Compute the residual vector for training PQ
    let ivf_centroids = ivf_model.centroids.as_ref().try_into()?;
    let residual_data =
        compute_residual_matrix(&training_data, &ivf_centroids, ivf_params.metric_type)?;
    let pq_training_data =
        FixedSizeListArray::try_new(residual_data.as_ref(), training_data.num_columns() as i32)?;

    // Train PQ
    let mut pq = ProductQuantizer::new(
        pq_params.num_sub_vectors,
        pq_params.num_bits as u32,
        training_data.num_columns(),
    );
    pq.train(
        &pq_training_data,
        pq_params.metric_type,
        pq_params.max_iters,
    )
    .await?;

    todo!()
}

/// Train Optimized Product Quantization.
async fn train_opq(data: &MatrixView, params: &PQBuildParams) -> Result<OptimizedProductQuantizer> {
    let mut opq = OptimizedProductQuantizer::new(
        params.num_sub_vectors as usize,
        params.num_bits as u32,
        params.metric_type,
        params.max_iters,
    );

    opq.train(data).await?;

    Ok(opq)
}

/// Train IVF partitions using kmeans.
async fn train_ivf_model(data: &MatrixView, params: &IvfBuildParams) -> Result<Ivf> {
    let rng = SmallRng::from_entropy();

    let centroids = super::kmeans::train_kmeans(
        data.data().as_ref(),
        data.num_columns(),
        params.num_partitions,
        params.max_iters as u32,
        rng,
        params.metric_type,
    )
    .await?;
    Ok(Ivf::new(Arc::new(FixedSizeListArray::try_new(
        centroids,
        data.num_columns() as i32,
    )?)))
}

#[async_trait]
impl IndexBuilder for IvfPqIndexBuilder {
    fn index_type() -> IndexType {
        IndexType::Vector
    }

    /// Build the IVF_PQ index
    async fn build(&self) -> Result<()> {
        println!(
            "Building vector index: IVF{},PQ{}, metric={}",
            self.num_partitions, self.num_sub_vectors, self.metric_type,
        );

        // Step 1. Sanity check
        sanity_check(self.dataset.as_ref(), &self.column)?;

        // Make with row id to build inverted index, and sampling
        let sample_size = self.sample_size_hint();
        let dataset = self.dataset.clone();
        // let projection = dataset.schema().project(&[&self.column])?;
        // let training_data = dataset.sample(sample_size, &projection).await;
        // let training_data =
        //     sample_vector_column(self.dataset.clone(), &self.column, sample_size).await?;

        // First, scan the dataset to train IVF models.
        let mut ivf_model = self.train_ivf_model().await?;

        // A new scanner, with row id to build inverted index.
        let mut scanner = self.dataset.scan();
        scanner.project(&[&self.column])?;
        scanner.with_row_id();

        // Assign parition ID and compute residual vectors. cxc
        let partitioned_batches = ivf_model.partition(&scanner, self.metric_type).await?;

        // Train PQ
        let mut pq =
            ProductQuantizer::new(self.num_sub_vectors as usize, self.nbits, self.dimension);
        let batch = concat_batches(&partitioned_batches[0].schema(), &partitioned_batches)?;
        let residual_vector = batch.column_by_name(RESIDUAL_COLUMN).unwrap();

        let pq_code = pq
            .fit_transform(as_fixed_size_list_array(residual_vector), self.metric_type)
            .await?;

        const PQ_CODE_COLUMN: &str = "__pq_code";
        let pq_code_batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                ArrowField::new(PQ_CODE_COLUMN, pq_code.data_type().clone(), false),
                ArrowField::new(PARTITION_ID_COLUMN, DataType::UInt32, false),
                ArrowField::new(ROW_ID, DataType::UInt64, false),
            ])),
            vec![
                Arc::new(pq_code),
                batch.column_by_name(PARTITION_ID_COLUMN).unwrap().clone(),
                batch.column_by_name(ROW_ID).unwrap().clone(),
            ],
        )?;

        let object_store = self.dataset.object_store();
        let path = self
            .dataset
            .indices_dir()
            .child(self.uuid.to_string())
            .child(INDEX_FILE_NAME);
        let mut writer = object_store.create(&path).await?;

        // Write each partition to disk.
        let part_col = pq_code_batch
            .column_by_name(PARTITION_ID_COLUMN)
            .unwrap_or_else(|| panic!("{PARTITION_ID_COLUMN} does not exist"));
        let partition_ids: &UInt32Array = as_primitive_array(part_col);
        let min_id = min(partition_ids).unwrap_or(0);
        let max_id = max(partition_ids).unwrap_or(1024 * 1024);

        for part_id in min_id..max_id + 1 {
            let predicates = BooleanArray::from_unary(partition_ids, |x| x == part_id);
            let parted_batch = filter_record_batch(&pq_code_batch, &predicates)?;
            ivf_model.add_partition(writer.tell(), parted_batch.num_rows() as u32);
            if parted_batch.num_rows() > 0 {
                // Write one partition.
                let pq_code = &parted_batch[PQ_CODE_COLUMN];
                writer.write_plain_encoded_array(pq_code.as_ref()).await?;
                let row_ids = &parted_batch[ROW_ID];
                writer.write_plain_encoded_array(row_ids.as_ref()).await?;
            }
        }

        let metadata = IvfPQIndexMetadata {
            name: self.name.clone(),
            column: self.column.clone(),
            dimension: self.dimension as u32,
            dataset_version: self.dataset.version().version,
            ivf: ivf_model,
            pq: pq.into(),
            metric_type: self.metric_type,
        };

        let metadata = pb::Index::try_from(&metadata)?;
        let pos = writer.write_protobuf(&metadata).await?;
        writer.write_magics(pos).await?;
        writer.shutdown().await?;

        Ok(())
    }
}

async fn train_kmeans_model(
    scanner: &Scanner,
    dimension: usize,
    k: usize,
    max_iterations: u32,
    rng: impl Rng,
    metric_type: MetricType,
) -> Result<Arc<FixedSizeListArray>> {
    let schema = scanner.schema()?;
    assert_eq!(schema.fields.len(), 1);
    let column_name = schema.fields[0].name();
    // Copy all to memory for now, optimize later.
    let batches = scanner
        .try_into_stream()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let mut arr_list = vec![];
    for batch in batches {
        let arr = batch.column_by_name(column_name).unwrap();
        let list_arr = as_fixed_size_list_array(&arr);
        arr_list.push(list_arr.values().clone());
    }

    let arrays = arr_list.iter().map(|l| l.as_ref()).collect::<Vec<_>>();

    let all_vectors = concat(&arrays)?;
    let values: &Float32Array = as_primitive_array(&all_vectors);
    let centroids =
        super::kmeans::train_kmeans(values, dimension, k, max_iterations, rng, metric_type).await?;
    Ok(Arc::new(FixedSizeListArray::try_new(
        centroids,
        dimension as i32,
    )?))
}
