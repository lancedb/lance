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

use std::{any::Any, sync::Arc};

use arrow::datatypes::Float32Type;
use arrow_arith::arithmetic::subtract_dyn;
use arrow_array::{
    builder::{Float32Builder, UInt32Builder},
    cast::{as_primitive_array, as_struct_array, AsArray},
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
use lance_arrow::*;
use lance_linalg::{kernels::argmin, matrix::MatrixView};
use log::info;
use rand::{rngs::SmallRng, SeedableRng};
use roaring::RoaringBitmap;
use serde::Serialize;

#[cfg(feature = "opq")]
use super::opq::train_opq;
use super::{
    pq::{train_pq, PQBuildParams, ProductQuantizer},
    utils::maybe_sample_training_data,
    MetricType, Query, VectorIndex, INDEX_FILE_NAME,
};
use crate::{
    dataset::{Dataset, ROW_ID},
    datatypes::Field,
    index::{pb, prefilter::PreFilter, vector::Transformer, Index},
};
use crate::{
    io::{local::to_local_path, object_reader::ObjectReader},
    session::Session,
};
use crate::{Error, Result};

const PARTITION_ID_COLUMN: &str = "__ivf_part_id";
const RESIDUAL_COLUMN: &str = "__residual_vector";
const PQ_CODE_COLUMN: &str = "__pq_code";

/// IVF Index.
pub struct IVFIndex {
    uuid: String,

    /// Ivf model
    ivf: Ivf,

    reader: Arc<dyn ObjectReader>,

    /// Index in each partition.
    sub_index: Arc<dyn VectorIndex>,

    metric_type: MetricType,

    session: Arc<Session>,

    fragment_bitmap: Option<RoaringBitmap>,
}

impl IVFIndex {
    /// Create a new IVF index.
    pub(crate) fn try_new(
        session: Arc<Session>,
        uuid: &str,
        ivf: Ivf,
        reader: Arc<dyn ObjectReader>,
        sub_index: Arc<dyn VectorIndex>,
        metric_type: MetricType,
    ) -> Result<Self> {
        if !sub_index.is_loadable() {
            return Err(Error::Index {
                message: format!("IVF sub index must be loadable, got: {:?}", sub_index),
            });
        }
        Ok(Self {
            uuid: uuid.to_owned(),
            session,
            ivf,
            reader,
            sub_index,
            metric_type,
            fragment_bitmap: None,
        })
    }

    async fn search_in_partition(
        &self,
        partition_id: usize,
        query: &Query,
        pre_filter: &PreFilter,
    ) -> Result<RecordBatch> {
        let cache_key = format!("{}-ivf-{}", self.uuid, partition_id);
        let part_index = if let Some(part_idx) = self.session.index_cache.get(&cache_key) {
            part_idx
        } else {
            let offset = self.ivf.offsets[partition_id];
            let length = self.ivf.lengths[partition_id] as usize;
            let idx = self
                .sub_index
                .load(self.reader.as_ref(), offset, length)
                .await?;
            self.session.index_cache.insert(&cache_key, idx.clone());
            idx
        };

        let partition_centroids = self.ivf.centroids.value(partition_id);
        let residual_key = subtract_dyn(query.key.as_ref(), &partition_centroids)?;
        // Query in partition.
        let mut part_query = query.clone();
        part_query.key = as_primitive_array(&residual_key).clone().into();
        let batch = part_index.search(&part_query, pre_filter).await?;
        Ok(batch)
    }
}

impl std::fmt::Debug for IVFIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Ivf({}) -> {:?}", self.metric_type, self.sub_index)
    }
}

#[derive(Serialize)]
pub struct IvfIndexPartitionStatistics {
    index: usize,
    length: u32,
    offset: usize,
    centroid: Vec<f32>,
}

#[derive(Serialize)]
pub struct IvfIndexStatistics {
    index_type: String,
    uuid: String,
    uri: String,
    metric_type: String,
    num_partitions: usize,
    sub_index: serde_json::Value,
    partitions: Vec<IvfIndexPartitionStatistics>,
}

impl Index for IVFIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let partitions_statistics = self
            .ivf
            .lengths
            .iter()
            .enumerate()
            .map(|(i, &len)| {
                let centroid = self.ivf.centroids.value(i);
                let centroid_arr: &Float32Array = as_primitive_array(centroid.as_ref());
                IvfIndexPartitionStatistics {
                    index: i,
                    length: len,
                    offset: self.ivf.offsets[i],
                    centroid: centroid_arr.values().to_vec(),
                }
            })
            .collect::<Vec<_>>();

        Ok(serde_json::to_value(IvfIndexStatistics {
            index_type: "IVF".to_string(),
            uuid: self.uuid.clone(),
            uri: to_local_path(self.reader.path()),
            metric_type: self.metric_type.to_string(),
            num_partitions: self.ivf.num_partitions(),
            sub_index: self.sub_index.statistics()?,
            partitions: partitions_statistics,
        })?)
    }

    fn fragment_bitmap(&self) -> Option<&RoaringBitmap> {
        self.fragment_bitmap.as_ref()
    }
}

#[async_trait]
impl VectorIndex for IVFIndex {
    async fn search(&self, query: &Query, pre_filter: &PreFilter) -> Result<RecordBatch> {
        let partition_ids =
            self.ivf
                .find_partitions(&query.key, query.nprobes, self.metric_type)?;
        assert!(partition_ids.len() <= query.nprobes);
        let part_ids = partition_ids.values().to_vec();
        let batches = stream::iter(part_ids)
            .map(|part_id| async move {
                self.search_in_partition(part_id as usize, query, pre_filter)
                    .await
            })
            .buffer_unordered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;
        let batch = concat_batches(&batches[0].schema(), &batches)?;

        let dist_col = batch.column_by_name("_distance").ok_or_else(|| Error::IO {
            message: format!(
                "_distance column does not exist in batch: {}",
                batch.schema()
            ),
        })?;

        // TODO: Use a heap sort to get the top-k.
        let limit = query.k * query.refine_factor.unwrap_or(1) as usize;
        let selection = sort_to_indices(dist_col, None, Some(limit))?;
        let struct_arr = StructArray::from(batch);
        let taken_distances = take(&struct_arr, &selection, None)?;
        Ok(as_struct_array(&taken_distances).into())
    }

    fn is_loadable(&self) -> bool {
        false
    }

    async fn load(
        &self,
        _reader: &dyn ObjectReader,
        _offset: usize,
        _length: usize,
    ) -> Result<Arc<dyn VectorIndex>> {
        Err(Error::Index {
            message: "Flat index does not support load".to_string(),
        })
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

    /// Metric to compute distance
    pub(crate) metric_type: MetricType,

    /// IVF model
    pub(crate) ivf: Ivf,

    /// Product Quantizer
    pub(crate) pq: Arc<ProductQuantizer>,

    /// Transforms to be applied before search.
    transforms: Vec<pb::Transform>,

    /// The fragment coverage of this index.
    fragment_bitmap: RoaringBitmap,
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

        let mut fragment_bitmap: Vec<u8> = Vec::new();
        idx.fragment_bitmap.serialize_into(&mut fragment_bitmap)?;

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
                    MetricType::Dot => pb::VectorMetricType::Dot.into(),
                },
            })),
            fragment_bitmap,
        })
    }
}

/// Ivf Model
#[derive(Debug, Clone)]
pub(crate) struct Ivf {
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
            return Err(Error::IO {
                message: format!(
                    "Ivf::find_partition: dimension mismatch: {} != {}",
                    query.len(),
                    self.dimension()
                ),
            });
        }
        let dist_func = metric_type.batch_func();
        let centroid_values = self.centroids.values();
        let distances = dist_func(
            query.values(),
            centroid_values.as_primitive::<Float32Type>().values(),
            self.dimension(),
        ) as ArrayRef;
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
        let dist_func = metric_type.batch_func();
        let centroids: MatrixView = self.centroids.as_ref().try_into()?;
        for i in 0..data.num_rows() {
            let vector = data.row(i).unwrap();
            let part_id =
                argmin(dist_func(vector, centroids.data().values(), dim).as_ref()).unwrap();
            part_id_builder.append_value(part_id);
            let cent = centroids.row(part_id as usize).unwrap();
            if vector.len() != cent.len() {
                return Err(Error::IO {
                    message: format!(
                        "Ivf::compute_residual: dimension mismatch: {} != {}",
                        vector.len(),
                        cent.len()
                    ),
                });
            }
            unsafe {
                residual_builder
                    .append_trusted_len_iter(vector.iter().zip(cent.iter()).map(|(v, c)| v - c))
            }
        }

        let part_ids = part_id_builder.finish();
        let residuals =
            FixedSizeListArray::try_new_from_values(residual_builder.finish(), dim as i32)?;
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(PARTITION_ID_COLUMN, DataType::UInt32, false),
            ArrowField::new(
                RESIDUAL_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
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
            return Err(Error::IO {
                message: "Ivf model has not been populated".to_string(),
            });
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
        let centroids = Arc::new(FixedSizeListArray::try_new_from_values(
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

fn sanity_check<'a>(dataset: &'a Dataset, column: &str) -> Result<&'a Field> {
    let Some(field) = dataset.schema().field(column) else {
        return Err(Error::IO {
            message: format!(
                "Building index: column {} does not exist in dataset: {:?}",
                column, dataset
            ),
        });
    };
    if let DataType::FixedSizeList(elem_type, _) = field.data_type() {
        if !matches!(elem_type.data_type(), DataType::Float32) {
            return Err(
        Error::Index{message:
            format!("VectorIndex requires the column data type to be fixed size list of float32s, got {}",
            elem_type.data_type())});
        }
    } else {
        return Err(Error::Index {
            message: format!(
            "VectorIndex requires the column data type to be fixed size list of float32s, got {}",
            field.data_type()
        ),
        });
    }
    Ok(field)
}

/// Parameters to build IVF partitions
#[derive(Debug, Clone)]
pub struct IvfBuildParams {
    /// Number of partitions to build.
    pub num_partitions: usize,

    // ---- kmeans parameters
    /// Max number of iterations to train kmeans.
    pub max_iters: usize,

    /// Use provided IVF centroids.
    pub centroids: Option<Arc<FixedSizeListArray>>,
}

impl Default for IvfBuildParams {
    fn default() -> Self {
        Self {
            num_partitions: 32,
            max_iters: 50,
            centroids: None,
        }
    }
}

impl IvfBuildParams {
    /// Create a new instance of `IvfBuildParams`.
    pub fn new(num_partitions: usize) -> Self {
        Self {
            num_partitions,
            ..Default::default()
        }
    }

    /// Create a new instance of [`IvfBuildParams`] with centroids.
    pub fn try_with_centroids(
        num_partitions: usize,
        centroids: Arc<FixedSizeListArray>,
    ) -> Result<Self> {
        if num_partitions != centroids.len() {
            return Err(Error::Index {
                message: format!(
                    "IvfBuildParams::try_with_centroids: num_partitions {} != centroids.len() {}",
                    num_partitions,
                    centroids.len()
                ),
            });
        }
        Ok(Self {
            num_partitions,
            centroids: Some(centroids),
            ..Default::default()
        })
    }
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
    let dist_func = metric_type.batch_func();

    let dim = data.num_columns();
    let mut builder = Float32Builder::with_capacity(data.data().len());
    for i in 0..data.num_rows() {
        let row = data.row(i).unwrap();
        let dist_array = dist_func(row, centroids.data().values(), dim);
        let part_id = argmin(dist_array.as_ref()).ok_or_else(|| Error::Index {
            message: format!(
                "Ivf::compute_residual: argmin failed. Failed to find minimum of {:?}",
                dist_array
            ),
        })?;
        let centroid = centroids.row(part_id as usize).unwrap();
        if row.len() != centroid.len() {
            return Err(Error::IO {
                message: format!(
                    "Ivf::compute_residual: dimension mismatch: {} != {}",
                    row.len(),
                    centroid.len()
                ),
            });
        };
        unsafe {
            builder.append_trusted_len_iter(row.iter().zip(centroid.iter()).map(|(v, c)| v - c))
        }
    }
    Ok(Arc::new(builder.finish()))
}

/// Build IVF(PQ) index
pub async fn build_ivf_pq_index(
    dataset: &Dataset,
    column: &str,
    index_name: &str,
    uuid: &str,
    metric_type: MetricType,
    ivf_params: &IvfBuildParams,
    pq_params: &PQBuildParams,
) -> Result<()> {
    info!(
        "Building vector index: IVF{},{}PQ{}, metric={}",
        ivf_params.num_partitions,
        if pq_params.use_opq { "O" } else { "" },
        pq_params.num_sub_vectors,
        metric_type,
    );

    let field = sanity_check(dataset, column)?;
    let dim = if let DataType::FixedSizeList(_, d) = field.data_type() {
        d as usize
    } else {
        return Err(Error::Index {
            message: format!(
                "VectorIndex requires the column data type to be fixed size list of floats, got {}",
                field.data_type()
            ),
        });
    };

    // Maximum to train 256 vectors per centroids, see Faiss.
    let sample_size_hint = std::cmp::max(
        ivf_params.num_partitions,
        ProductQuantizer::num_centroids(pq_params.num_bits as u32),
    ) * 256;
    // TODO: only sample data if training is necessary.
    let mut training_data = maybe_sample_training_data(dataset, column, sample_size_hint).await?;
    #[cfg(feature = "opq")]
    let mut transforms: Vec<Box<dyn Transformer>> = vec![];
    #[cfg(not(feature = "opq"))]
    let transforms: Vec<Box<dyn Transformer>> = vec![];

    // Train IVF partitions.
    let ivf_model = if let Some(centroids) = &ivf_params.centroids {
        if centroids.values().len() != ivf_params.num_partitions * dim {
            return Err(Error::Index {
                message: format!(
                    "IVF centroids length mismatch: {} != {}",
                    centroids.len(),
                    ivf_params.num_partitions * dim,
                ),
            });
        }
        Ivf::new(centroids.clone())
    } else {
        // Pre-transforms
        if pq_params.use_opq {
            #[cfg(not(feature = "opq"))]
            return Err(Error::Index {
                message: "Feature 'opq' is not installed.".to_string(),
            });
            #[cfg(feature = "opq")]
            {
                let opq = train_opq(&training_data, pq_params).await?;
                transforms.push(Box::new(opq));
            }
        }

        // Transform training data if necessary.
        for transform in transforms.iter() {
            training_data = transform.transform(&training_data).await?;
        }

        train_ivf_model(&training_data, metric_type, ivf_params).await?
    };

    let pq = if let Some(codebook) = &pq_params.codebook {
        ProductQuantizer::new_with_codebook(
            pq_params.num_sub_vectors,
            pq_params.num_bits as u32,
            dim,
            codebook.clone(),
        )
    } else {
        // Compute the residual vector for training PQ
        let ivf_centroids = ivf_model.centroids.as_ref().try_into()?;
        let residual_data = compute_residual_matrix(&training_data, &ivf_centroids, metric_type)?;
        let pq_training_data = MatrixView::new(residual_data, training_data.num_columns());

        train_pq(&pq_training_data, pq_params).await?
    };

    // Transform data, compute residuals and sort by partition ids.
    let mut scanner = dataset.scan();
    scanner.project(&[column])?;
    scanner.with_row_id();

    let ivf = &ivf_model;
    let pq_ref = &pq;
    let metric_type = pq_params.metric_type;
    let transform_ref = &transforms;

    // Scan the dataset and compute residual, pq with with partition ID.
    // For now, it loads all data into memory.
    let batches = scanner
        .try_into_stream()
        .await?
        .map(|b| async move {
            let batch = b?;
            let arr = batch.column_by_name(column).ok_or_else(|| Error::IO {
                message: format!("Dataset does not have column {column}"),
            })?;
            let mut vectors: MatrixView = as_fixed_size_list_array(arr).try_into()?;

            // Transform the vectors if pre-transforms are used.
            for transform in transform_ref.iter() {
                vectors = transform.transform(&vectors).await?;
            }

            let i = ivf.clone();
            let part_id_and_residual = tokio::task::spawn_blocking(move || {
                i.compute_partition_and_residual(&vectors, metric_type)
            })
            .await??;

            let residual_col = part_id_and_residual
                .column_by_name(RESIDUAL_COLUMN)
                .unwrap();
            let residual_data = as_fixed_size_list_array(&residual_col);
            let pq_code = pq_ref
                .transform(&residual_data.try_into()?, metric_type)
                .await?;

            let row_ids = batch
                .column_by_name(ROW_ID)
                .expect("Expect row id column")
                .clone();
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
                        Arc::new(ArrowField::new("item", DataType::UInt8, true)),
                        pq_params.num_sub_vectors as i32,
                    ),
                    false,
                ),
            ]));
            Ok::<RecordBatch, Error>(RecordBatch::try_new(
                schema,
                vec![row_ids, part_ids, Arc::new(pq_code)],
            )?)
        })
        .buffered(num_cpus::get())
        .try_collect::<Vec<_>>()
        .await?;

    write_index_file(
        dataset,
        column,
        index_name,
        uuid,
        &transforms,
        ivf_model,
        pq,
        metric_type,
        &batches,
    )
    .await
}

/// Write the index to the index file.
///
#[allow(clippy::too_many_arguments)]
async fn write_index_file(
    dataset: &Dataset,
    column: &str,
    index_name: &str,
    uuid: &str,
    transformers: &[Box<dyn Transformer>],
    mut ivf: Ivf,
    pq: ProductQuantizer,
    metric_type: MetricType,
    batches: &[RecordBatch],
) -> Result<()> {
    let object_store = dataset.object_store();
    let path = dataset.indices_dir().child(uuid).child(INDEX_FILE_NAME);
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
            let parted_batch = filter_record_batch(batch, &predicates)?;
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

    // Convert [`Transformer`] to metadata.
    let mut transforms = vec![];
    for t in transformers {
        let t = t.save(&mut writer).await?;
        transforms.push(t);
    }

    let metadata = IvfPQIndexMetadata {
        name: index_name.to_string(),
        column: column.to_string(),
        dimension: pq.dimension as u32,
        dataset_version: dataset.version().version,
        metric_type,
        ivf,
        pq: pq.into(),
        transforms,
        fragment_bitmap: dataset
            .get_fragments()
            .iter()
            .map(|f| f.id() as u32)
            .collect(),
    };

    let metadata = pb::Index::try_from(&metadata)?;
    let pos = writer.write_protobuf(&metadata).await?;
    writer.write_magics(pos).await?;
    writer.shutdown().await?;

    Ok(())
}

/// Train IVF partitions using kmeans.
async fn train_ivf_model(
    data: &MatrixView,
    metric_type: MetricType,
    params: &IvfBuildParams,
) -> Result<Ivf> {
    let rng = SmallRng::from_entropy();
    const REDOS: usize = 1;
    let centroids = super::kmeans::train_kmeans(
        data.data().as_ref(),
        None,
        data.num_columns(),
        params.num_partitions,
        params.max_iters as u32,
        REDOS,
        rng,
        metric_type,
    )
    .await?;
    Ok(Ivf::new(Arc::new(FixedSizeListArray::try_new_from_values(
        centroids,
        data.num_columns() as i32,
    )?)))
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{cast::AsArray, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use lance_testing::datagen::generate_random_array;
    use tempfile::tempdir;

    use std::collections::HashMap;

    use crate::index::{vector::VectorIndexParams, DatasetIndexExt, IndexType};

    #[tokio::test]
    async fn test_create_ivf_pq_with_centroids() {
        const DIM: usize = 32;
        let vectors = generate_random_array(1000 * DIM);
        let metadata: HashMap<String, String> = vec![("test".to_string(), "ivf_pq".to_string())]
            .into_iter()
            .collect();

        let schema = Arc::new(
            Schema::new(vec![Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DIM as i32,
                ),
                true,
            )])
            .with_metadata(metadata),
        );
        let array = Arc::new(FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap());
        let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()]).unwrap();

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let centroids = generate_random_array(2 * DIM);
        let ivf_centroids = FixedSizeListArray::try_new_from_values(centroids, DIM as i32).unwrap();
        let ivf_params = IvfBuildParams::try_with_centroids(2, Arc::new(ivf_centroids)).unwrap();

        let codebook = Arc::new(generate_random_array(256 * DIM));
        let pq_params = PQBuildParams::with_codebook(4, 8, codebook);

        let params = VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf_params, pq_params);

        let dataset = dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let elem = array.value(10);
        let query = elem.as_primitive::<Float32Type>();
        let results = dataset
            .scan()
            .nearest("vector", query, 5)
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(1, results.len());
        assert_eq!(5, results[0].num_rows());
    }
}
