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

use arrow::array::UInt32Builder;
use arrow_arith::arithmetic::subtract_dyn;
use arrow_array::{
    builder::Float32Builder,
    cast::{as_primitive_array, as_struct_array},
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, RecordBatch, StructArray,
    UInt32Array,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::{concat::concat_batches, filter::filter_record_batch, take::take};
use async_trait::async_trait;
use futures::{
    stream::{self, StreamExt},
    TryStreamExt,
};
use rand::{rngs::SmallRng, SeedableRng};
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
    dataset::{Dataset, ROW_ID},
    index::{pb, pb::vector_index_stage::Stage},
};
use crate::{Error, Result};

const INDEX_FILE_NAME: &str = "index.idx";
const PARTITION_ID_COLUMN: &str = "__ivf_part_id";
const RESIDUAL_COLUMN: &str = "__residual_vector";
const PQ_CODE_COLUMN: &str = "__pq_code";

/// IVF PQ Index.
pub struct IvfPQIndex<'a> {
    reader: Box<dyn ObjectReader + 'a>,

    /// Ivf file.
    ivf: Ivf,

    /// Number of bits used for product quantization centroids.
    pq: Arc<ProductQuantizer>,

    metric_type: MetricType,

    /// Transform applys to each vector.
    transforms: Vec<Arc<dyn Transformer>>,
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

        let reader_ref = reader.as_ref();
        let transforms = stream::iter(index_metadata.transforms)
            .map(|tf| async move {
                Ok::<Arc<dyn Transformer>, Error>(Arc::new(
                    OptimizedProductQuantizer::load(
                        reader_ref,
                        tf.position as usize,
                        tf.shape
                            .iter()
                            .map(|s| *s as usize)
                            .collect::<Vec<_>>()
                            .as_slice(),
                    )
                    .await?,
                ))
            })
            .buffered(4)
            .try_collect::<Vec<Arc<dyn Transformer>>>()
            .await?;

        Ok(Self {
            reader,
            ivf: index_metadata.ivf,
            pq: index_metadata.pq,
            metric_type: index_metadata.metric_type,
            transforms,
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
        let mut mat = MatrixView::new(query.key.clone(), query.key.len());
        for transform in self.transforms.iter() {
            mat = transform.transform(&mat).await?;
        }
        let key = mat.data();
        let key_ref = key.as_ref();
        let partition_ids = self
            .ivf
            .find_partitions(key_ref, query.nprobs, self.metric_type)?;
        let candidates = stream::iter(partition_ids.values())
            .then(|part_id| async move {
                self.search_in_partition(*part_id as usize, key_ref, query.k)
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

    /// File position of transforms.
    transforms: Vec<pb::Transform>,
}

/// Convert a IvfPQIndex to protobuf payload
impl TryFrom<&IvfPQIndexMetadata> for pb::Index {
    type Error = Error;

    fn try_from(idx: &IvfPQIndexMetadata) -> Result<Self> {
        let mut stages: Vec<pb::VectorIndexStage> = idx
            .transforms
            .iter()
            .map(|tf| {
                Ok(pb::VectorIndexStage {
                    stage: Some(pb::vector_index_stage::Stage::Transform(tf.clone())),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        stages.extend_from_slice(&[
            pb::VectorIndexStage {
                stage: Some(pb::vector_index_stage::Stage::Ivf(pb::Ivf::try_from(
                    &idx.ivf,
                )?)),
            },
            pb::VectorIndexStage {
                stage: Some(pb::vector_index_stage::Stage::Pq(idx.pq.as_ref().into())),
            },
        ]);

        Ok(Self {
            name: idx.name.clone(),
            columns: vec![idx.column.clone()],
            dataset_version: idx.dataset_version,
            index_type: pb::IndexType::Vector.into(),
            implementation: Some(pb::index::Implementation::VectorIndex(pb::VectorIndex {
                spec_version: 1,
                dimension: idx.dimension,
                stages,
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
                        let num_stages = vidx.stages.len();
                        if num_stages != 2 || num_stages != 3 {
                            return Err(Error::IO("Only support IVF_(O)PQ now".to_string()));
                        };
                        let opq = match vidx.stages[0].stage.as_ref().unwrap() {
                            Stage::Transform(transform) => Ok(transform),
                            _ => Err(Error::IO("Stage 0 only supports OPQ".to_string())),
                        }?;
                        let stage0 = vidx.stages[1].stage.as_ref().ok_or_else(|| {
                            Error::IO("VectorIndex stage 0 is missing".to_string())
                        })?;
                        let ivf = match stage0 {
                            Stage::Ivf(ivf_pb) => Ok(Ivf::try_from(ivf_pb)?),
                            _ => Err(Error::IO("Stage 0 only supports IVF".to_string())),
                        }?;
                        let stage1 = vidx.stages[2].stage.as_ref().ok_or_else(|| {
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
                            transforms: vec![opq.clone()],
                        })
                    }
                }?
            } else {
                return Err(Error::IO("Invalid protobuf".to_string()));
            };
        Ok(metadata)
    }
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

    /// Number of IVF partitions.
    fn num_partitions(&self) -> usize {
        self.centroids.len()
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

    /// Compute the partition ID and residual vectors.
    ///
    /// Parameters
    /// - *data*: input matrix to compute residual.
    /// - *metric_type*: the metric type to compute distance.
    ///
    /// Returns a `RecordBatch` with schema `{__part_id: u32, __residual: FixedSizeList}`
    pub fn compute_partition_and_residual(
        &self,
        data: &MatrixView,
        metric_type: MetricType,
    ) -> Result<RecordBatch> {
        let mut part_id_builder = UInt32Builder::with_capacity(data.num_rows());
        let mut residual_builder =
            Float32Builder::with_capacity(data.num_columns() * data.num_rows());

        let dim = data.num_columns();
        let dist_func = metric_type.func();
        let centroids: MatrixView = self.centroids.as_ref().try_into()?;
        for i in 0..data.num_rows() {
            let vector = data.row(i).unwrap();
            let part_id = argmin(
                dist_func(&vector, centroids.data().as_ref(), dim)
                    .unwrap()
                    .as_ref(),
            )
            .unwrap();
            part_id_builder.append_value(part_id);
            let cent = centroids.row(part_id as usize).unwrap();
            let residual = subtract_dyn(&vector, &cent)?;
            let resi_arr: &Float32Array = as_primitive_array(&residual);
            residual_builder.append_slice(resi_arr.values());
        }

        let part_ids = part_id_builder.finish();
        let residuals = FixedSizeListArray::try_new(residual_builder.finish(), dim as i32)?;
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(PARTITION_ID_COLUMN, DataType::UInt32, false),
            ArrowField::new(
                RESIDUAL_COLUMN,
                DataType::FixedSizeList(
                    Box::new(ArrowField::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(part_ids), Arc::new(residuals)])?;
        Ok(batch)
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
    index_name: &str,
    uuid: &Uuid,
    ivf_params: &IvfBuildParams,
    pq_params: &PQBuildParams,
) -> Result<()> {
    println!(
        "Building vector index: IVF{},{}PQ{}, metric={}",
        ivf_params.num_partitions,
        if pq_params.use_opq { "O" } else { "" },
        pq_params.num_sub_vectors,
        ivf_params.metric_type,
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
    let ivf_model = train_ivf_model(&training_data, ivf_params).await?;

    // Compute the residual vector for training PQ
    let ivf_centroids = ivf_model.centroids.as_ref().try_into()?;
    let residual_data =
        compute_residual_matrix(&training_data, &ivf_centroids, ivf_params.metric_type)?;
    let pq_training_data =
        FixedSizeListArray::try_new(residual_data.as_ref(), training_data.num_columns() as i32)?;

    // The final train of PQ sub-vectors
    let pq = train_pq(&pq_training_data, pq_params).await?;

    // Transform data, compute residuals and sort by partition ids.
    let mut scanner = dataset.scan();
    scanner.project(&[column])?;
    scanner.with_row_id();

    let ivf = &ivf_model;
    let opq_ref = &opq;
    let pq_ref = &pq;
    let metric_type = pq_params.metric_type;

    // Scan the dataset and compute residual, pq with with partition ID.
    // For now, it loads all data into memory.
    let batches = scanner
        .try_into_stream()
        .await?
        .map(|b| async move {
            let batch = b?;
            let arr = batch
                .column_by_name(column)
                .ok_or_else(|| Error::IO(format!("Dataset does not have column {column}")))?;
            let vectors: MatrixView = as_fixed_size_list_array(arr).try_into()?;
            // Transform using OPQ matrix.
            let vectors = opq_ref.transform(&vectors).await?;
            let part_id_and_residual = ivf.compute_partition_and_residual(&vectors, metric_type)?;

            let residual_col = part_id_and_residual
                .column_by_name(RESIDUAL_COLUMN)
                .unwrap();
            let residual_data = as_fixed_size_list_array(&residual_col);
            let pq_code = pq_ref.transform(&residual_data, metric_type).await?;

            let row_ids = batch.column_by_name(ROW_ID).expect("Expect row id").clone();
            let part_ids = part_id_and_residual
                .column_by_name(PARTITION_ID_COLUMN)
                .expect("Expect partition ids column")
                .clone();

            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new(ROW_ID, DataType::UInt64, false),
                ArrowField::new(PARTITION_ID_COLUMN, DataType::UInt32, false),
                ArrowField::new(
                    PQ_CODE_COLUMN,
                    DataType::FixedSizeList(
                        Box::new(ArrowField::new("item", DataType::UInt8, true)),
                        pq_params.num_sub_vectors as i32,
                    ),
                    false,
                ),
            ]));
            RecordBatch::try_new(schema.clone(), vec![row_ids, part_ids, Arc::new(pq_code)])
        })
        .buffered(num_cpus::get())
        .try_collect::<Vec<_>>()
        .await?;

    write_index_file(
        dataset,
        column,
        index_name,
        uuid,
        ivf_model,
        opq,
        pq,
        ivf_params.metric_type,
        &batches,
    )
    .await
}

/// Write index into the file.
async fn write_index_file(
    dataset: &Dataset,
    column: &str,
    index_name: &str,
    uuid: &Uuid,
    mut ivf: Ivf,
    mut opq: OptimizedProductQuantizer,
    pq: ProductQuantizer,
    metric_type: MetricType,
    batches: &[RecordBatch],
) -> Result<()> {
    let object_store = dataset.object_store();
    let path = dataset
        .indices_dir()
        .child(uuid.to_string())
        .child(INDEX_FILE_NAME);
    let mut writer = object_store.create(&path).await?;

    // Write each partition to disk.
    for part_id in 0..ivf.num_partitions() as u32 {
        let mut batches_for_parq: Vec<RecordBatch> = vec![];
        for batch in batches.iter() {
            let part_col = batch
                .column_by_name(PARTITION_ID_COLUMN)
                .unwrap_or_else(|| panic!("{PARTITION_ID_COLUMN} does not exist"));
            let partition_ids: &UInt32Array = as_primitive_array(part_col);
            let predicates = BooleanArray::from_unary(partition_ids, |x| x == part_id);
            let parted_batch = filter_record_batch(&batch, &predicates)?;
            batches_for_parq.push(parted_batch);
        }
        let parted_batch = concat_batches(&batches_for_parq[0].schema(), &batches_for_parq)?;
        ivf.add_partition(writer.tell(), parted_batch.num_rows() as u32);
        if parted_batch.num_rows() > 0 {
            // Write one partition.
            let pq_code = &parted_batch[PQ_CODE_COLUMN];
            writer.write_plain_encoded_array(pq_code.as_ref()).await?;
            let row_ids = &parted_batch[ROW_ID];
            writer.write_plain_encoded_array(row_ids.as_ref()).await?;
        }
    }

    // Write OPQ matrix.
    let pos = writer
        .write_plain_encoded_array(opq.rotation.as_ref().unwrap().data().as_ref())
        .await?;
    opq.file_position = Some(pos);

    let metadata = IvfPQIndexMetadata {
        name: index_name.to_string(),
        column: column.to_string(),
        dimension: pq.dimension as u32,
        dataset_version: dataset.version().version,
        metric_type,
        ivf,
        pq: pq.into(),
        transforms: vec![opq.try_into_pb()?],
    };

    let metadata = pb::Index::try_from(&metadata)?;
    let pos = writer.write_protobuf(&metadata).await?;
    writer.write_magics(pos).await?;
    writer.shutdown().await?;

    Ok(())
}

/// Train product quantization over (OPQ-rotated) residual vectors.
async fn train_pq(data: &FixedSizeListArray, params: &PQBuildParams) -> Result<ProductQuantizer> {
    let mut pq = ProductQuantizer::new(
        params.num_sub_vectors,
        params.num_bits as u32,
        data.value_length() as usize,
    );
    pq.train(&data, params.metric_type, params.max_iters)
        .await?;
    Ok(pq)
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
