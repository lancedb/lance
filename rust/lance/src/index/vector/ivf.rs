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

use std::{any::Any, collections::HashMap, sync::Arc};

use arrow_arith::numeric::sub;
use arrow_array::{
    cast::{as_primitive_array, as_struct_array, AsArray},
    types::Float32Type,
    Array, FixedSizeListArray, Float32Array, RecordBatch, StructArray, UInt32Array,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::DataType;
use arrow_select::{concat::concat_batches, take::take};
use async_trait::async_trait;
use futures::{
    stream::{self, StreamExt},
    TryStreamExt,
};
use lance_arrow::*;
use lance_core::io::{
    local::to_local_path, ObjectWriter, Reader, RecordBatchStream, WriteExt, Writer,
};
use lance_core::{
    datatypes::Field, encodings::plain::PlainEncoder, format::Index as IndexMetadata, Error, Result,
};
use lance_index::vector::{
    pq::{PQBuildParams, ProductQuantizer},
    Query, DIST_COL, RESIDUAL_COLUMN,
};
use lance_linalg::matrix::MatrixView;
use log::{debug, info, warn};
use rand::{rngs::SmallRng, SeedableRng};
use serde::Serialize;
use snafu::{location, Location};
use tracing::{instrument, span, Level};
use uuid::Uuid;

#[cfg(feature = "opq")]
use super::opq::train_opq;
use super::{
    pq::{train_pq, PQIndex},
    utils::maybe_sample_training_data,
    MetricType, VectorIndex, INDEX_FILE_NAME,
};
use crate::{
    dataset::Dataset,
    index::{
        pb,
        prefilter::PreFilter,
        vector::{
            ivf::{builder::shuffle_dataset, io::write_index_partitions},
            Transformer,
        },
        Index,
    },
    session::Session,
};

mod builder;
mod io;
mod shuffler;

/// IVF Index.
pub struct IVFIndex {
    uuid: String,

    /// Ivf model
    ivf: Ivf,

    reader: Arc<dyn Reader>,

    /// Index in each partition.
    sub_index: Arc<dyn VectorIndex>,

    metric_type: MetricType,

    session: Arc<Session>,
}

impl IVFIndex {
    /// Create a new IVF index.
    pub(crate) fn try_new(
        session: Arc<Session>,
        uuid: &str,
        ivf: Ivf,
        reader: Arc<dyn Reader>,
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
        })
    }

    /// Load one partition of the IVF sub-index.
    ///
    /// Parameters
    /// ----------
    ///  - partition_id: partition ID.
    #[instrument(level = "debug", skip(self))]
    pub(crate) async fn load_partition(&self, partition_id: usize) -> Result<Arc<dyn VectorIndex>> {
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
            let idx: Arc<dyn VectorIndex> = idx.into();
            self.session.index_cache.insert(&cache_key, idx.clone());
            idx
        };
        Ok(part_index)
    }

    async fn search_in_partition(
        &self,
        partition_id: usize,
        query: &Query,
        pre_filter: Arc<PreFilter>,
    ) -> Result<RecordBatch> {
        let part_index = self.load_partition(partition_id).await?;

        let partition_centroids = self.ivf.centroids.value(partition_id);
        let residual_key = sub(query.key.as_ref(), &partition_centroids)?;
        // Query in partition.
        let mut part_query = query.clone();
        part_query.key = as_primitive_array(&residual_key).clone().into();
        let batch = part_index.search(&part_query, pre_filter).await?;
        Ok(batch)
    }

    pub(crate) async fn append(
        &self,
        dataset: &Dataset,
        data: impl RecordBatchStream + Unpin,
        metadata: &IndexMetadata,
        column: &str,
    ) -> Result<Uuid> {
        let new_uuid = Uuid::new_v4();
        let object_store = dataset.object_store();
        let index_file = dataset
            .indices_dir()
            .child(new_uuid.to_string())
            .child(INDEX_FILE_NAME);
        let mut writer = object_store.create(&index_file).await?;

        let pq_index = self
            .sub_index
            .as_any()
            .downcast_ref::<PQIndex>()
            .ok_or(Error::Index {
                message: "Only support append to IVF_PQ".to_string(),
            })?;

        // TODO: merge two IVF implementations.
        let ivf = Arc::new(lance_index::vector::ivf::Ivf::new_with_pq(
            self.ivf.centroids.as_ref().try_into()?,
            self.metric_type,
            column,
            pq_index.pq.clone(),
        ));
        let shuffler = shuffle_dataset(data, column, ivf, pq_index.pq.num_sub_vectors).await?;

        let mut ivf_mut = Ivf::new(self.ivf.centroids.clone());
        write_index_partitions(&mut writer, &mut ivf_mut, &shuffler, Some(self)).await?;
        let metadata = IvfPQIndexMetadata {
            name: metadata.name.clone(),
            column: column.to_string(),
            dimension: self.ivf.dimension() as u32,
            dataset_version: dataset.version().version,
            metric_type: self.metric_type,
            ivf: ivf_mut,
            pq: pq_index.pq.clone(),
            transforms: vec![],
        };

        let metadata = pb::Index::try_from(&metadata)?;
        let pos = writer.write_protobuf(&metadata).await?;
        writer.write_magics(pos).await?;
        writer.shutdown().await?;

        Ok(new_uuid)
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
}

#[async_trait]
impl VectorIndex for IVFIndex {
    #[instrument(level = "debug", skip_all, name = "IVFIndex::search")]
    async fn search(&self, query: &Query, pre_filter: Arc<PreFilter>) -> Result<RecordBatch> {
        let partition_ids =
            self.ivf
                .find_partitions(&query.key, query.nprobes, self.metric_type)?;
        assert!(partition_ids.len() <= query.nprobes);
        let part_ids = partition_ids.values().to_vec();
        let batches = stream::iter(part_ids)
            .map(|part_id| self.search_in_partition(part_id as usize, query, pre_filter.clone()))
            .buffer_unordered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;
        let batch = concat_batches(&batches[0].schema(), &batches)?;

        let dist_col = batch.column_by_name(DIST_COL).ok_or_else(|| Error::IO {
            message: format!(
                "_distance column does not exist in batch: {}",
                batch.schema()
            ),
            location: location!(),
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
        _reader: &dyn Reader,
        _offset: usize,
        _length: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        Err(Error::Index {
            message: "Flat index does not support load".to_string(),
        })
    }

    fn check_can_remap(&self) -> Result<()> {
        Ok(())
    }

    fn remap(&mut self, _mapping: &HashMap<u64, Option<u64>>) -> Result<()> {
        // This will be needed if we want to clean up IVF to allow more than just
        // one layer (e.g. IVF -> IVF -> PQ).  We need to pass on the call to
        // remap to the lower layers.

        // Currently, remapping for IVF is implemented in remap_index_file which
        // mirrors some of the other IVF routines like build_ivf_pq_index
        Err(Error::Index {
            message: "Remapping IVF in this way not supported".to_string(),
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
                    MetricType::Dot => pb::VectorMetricType::Dot.into(),
                },
            })),
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
        let ivf = lance_index::vector::ivf::Ivf::new(
            MatrixView::try_from(self.centroids.as_ref())?,
            metric_type,
            vec![],
        );
        ivf.find_partitions(query, nprobes)
    }

    /// Add the offset and length of one partition.
    fn add_partition(&mut self, offset: usize, len: u32) {
        self.offsets.push(offset);
        self.lengths.push(len);
    }

    /// Compute residual vector.
    ///
    /// A residual vector is `original vector - centroids`.
    ///
    /// Parameters:
    ///  - *original*: original vector.
    ///  - *partitions*: partition ID of each original vector.
    #[instrument(level = "debug", skip_all)]
    pub(super) fn compute_residual(
        &self,
        original: &MatrixView<Float32Type>,
        partitions: &UInt32Array,
    ) -> MatrixView<Float32Type> {
        let mut residual_arr: Vec<f32> =
            Vec::with_capacity(original.num_rows() * original.num_columns());
        original
            .iter()
            .zip(partitions.values().iter())
            .for_each(|(vector, &part_id)| {
                let values = self.centroids.value(part_id as usize);
                let centroid = values.as_primitive::<Float32Type>();
                residual_arr.extend(
                    vector
                        .iter()
                        .zip(centroid.values().iter())
                        .map(|(v, cent)| *v - *cent),
                );
            });
        MatrixView::new(Arc::new(residual_arr.into()), original.ndim())
    }
}

/// Convert IvfModel to protobuf.
impl TryFrom<&Ivf> for pb::Ivf {
    type Error = Error;

    fn try_from(ivf: &Ivf) -> Result<Self> {
        if ivf.offsets.len() != ivf.centroids.len() {
            return Err(Error::IO {
                message: "Ivf model has not been populated".to_string(),
                location: location!(),
            });
        }
        Ok(Self {
            centroids: vec![],
            offsets: ivf.offsets.iter().map(|o| *o as u64).collect(),
            lengths: ivf.lengths.clone(),
            centroids_tensor: Some(ivf.centroids.as_ref().try_into()?),
        })
    }
}

/// Convert IvfModel to protobuf.
impl TryFrom<&pb::Ivf> for Ivf {
    type Error = Error;

    fn try_from(proto: &pb::Ivf) -> Result<Self> {
        let centroids = if let Some(tensor) = proto.centroids_tensor.as_ref() {
            debug!("Ivf: loading IVF centroids from index format v2");
            Arc::new(FixedSizeListArray::try_from(tensor)?)
        } else {
            warn!("Ivf: loading IVF centroids from index format v1");
            // For backward-compatibility
            let f32_centroids = Float32Array::from(proto.centroids.clone());
            let dimension = f32_centroids.len() / proto.offsets.len();
            Arc::new(FixedSizeListArray::try_new_from_values(
                f32_centroids,
                dimension as i32,
            )?)
        };

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
            location: location!(),
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

    pub sample_rate: usize,
}

impl Default for IvfBuildParams {
    fn default() -> Self {
        Self {
            num_partitions: 32,
            max_iters: 50,
            centroids: None,
            sample_rate: 256, // See faiss
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

    let start = std::time::Instant::now();
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

        info!("Start to train IVF model");
        train_ivf_model(&training_data, metric_type, ivf_params).await?
    };
    info!(
        "Traied IVF model in {:02} seconds",
        start.elapsed().as_secs_f32()
    );

    let start = std::time::Instant::now();
    let pq = if let Some(codebook) = &pq_params.codebook {
        ProductQuantizer::new(
            pq_params.num_sub_vectors,
            pq_params.num_bits as u32,
            dim,
            codebook.clone(),
            metric_type,
        )
    } else {
        log::info!(
            "Start to train PQ code: PQ{}, bits={}",
            pq_params.num_sub_vectors,
            pq_params.num_bits
        );
        let expected_sample_size =
            ProductQuantizer::num_centroids(pq_params.num_bits as u32) * pq_params.sample_rate;
        let training_data = if training_data.num_rows() > expected_sample_size {
            training_data.sample(expected_sample_size)
        } else {
            training_data
        };

        // TODO: consolidate IVF models to `lance_index`.
        let ivf2 = lance_index::vector::ivf::Ivf::new(
            ivf_model.centroids.as_ref().try_into()?,
            metric_type,
            vec![],
        );
        // Compute the residual vector to train Product Quantizer.
        let part_ids = ivf2.compute_partitions(&training_data).await;

        let residuals = span!(Level::INFO, "compute residual for PQ training")
            .in_scope(|| ivf_model.compute_residual(&training_data, &part_ids));
        info!("Start train PQ: params={:#?}", pq_params);
        train_pq(&residuals, metric_type, pq_params).await?
    };
    info!("Trained PQ in: {} seconds", start.elapsed().as_secs_f32());

    // Transform data, compute residuals and sort by partition ids.
    let mut scanner = dataset.scan();
    scanner.batch_readahead(num_cpus::get() * 2);
    scanner.project(&[column])?;
    scanner.with_row_id();

    // Scan the dataset and compute residual, pq with with partition ID.
    // For now, it loads all data into memory.
    let stream = scanner.try_into_stream().await?;

    write_index_file(
        dataset,
        column,
        index_name,
        uuid,
        &transforms,
        ivf_model,
        pq.into(),
        metric_type,
        stream,
    )
    .await
}

struct RemapPageTask {
    offset: usize,
    length: u32,
    page: Option<Box<dyn VectorIndex>>,
}

impl RemapPageTask {
    fn new(offset: usize, length: u32) -> Self {
        Self {
            offset,
            length,
            page: None,
        }
    }
}

impl RemapPageTask {
    async fn load_and_remap(
        mut self,
        reader: &dyn Reader,
        index: &IVFIndex,
        mapping: &HashMap<u64, Option<u64>>,
    ) -> Result<Self> {
        let mut page = index
            .sub_index
            .load(reader, self.offset, self.length as usize)
            .await?;
        page.remap(mapping)?;
        self.page = Some(page);
        Ok(self)
    }

    async fn write(self, writer: &mut ObjectWriter, ivf: &mut Ivf) -> Result<()> {
        let page = self.page.as_ref().expect("Load was not called");
        let page: &PQIndex = page
            .as_any()
            .downcast_ref()
            .expect("Generic index writing not supported yet");
        ivf.offsets.push(writer.tell().await?);
        ivf.lengths
            .push(page.row_ids.as_ref().unwrap().len() as u32);
        PlainEncoder::write(writer, &[page.code.as_ref().unwrap().as_ref()]).await?;
        PlainEncoder::write(writer, &[page.row_ids.as_ref().unwrap().as_ref()]).await?;
        Ok(())
    }
}

fn generate_remap_tasks(offsets: &Vec<usize>, lengths: &[u32]) -> Result<Vec<RemapPageTask>> {
    let mut tasks: Vec<RemapPageTask> = Vec::with_capacity(offsets.len() * 2 + 1);

    for (offset, length) in offsets.iter().zip(lengths.iter()) {
        tasks.push(RemapPageTask::new(*offset, *length));
    }

    Ok(tasks)
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn remap_index_file(
    dataset: &Dataset,
    old_uuid: &str,
    new_uuid: &str,
    old_version: u64,
    index: &IVFIndex,
    mapping: &HashMap<u64, Option<u64>>,
    name: String,
    column: String,
    transforms: Vec<pb::Transform>,
) -> Result<()> {
    let object_store = dataset.object_store();
    let old_path = dataset.indices_dir().child(old_uuid).child(INDEX_FILE_NAME);
    let new_path = dataset.indices_dir().child(new_uuid).child(INDEX_FILE_NAME);

    let reader = object_store.open(&old_path).await?;
    let mut writer = object_store.create(&new_path).await?;

    let tasks = generate_remap_tasks(&index.ivf.offsets, &index.ivf.lengths)?;

    let mut task_stream = stream::iter(tasks.into_iter())
        .map(|task| task.load_and_remap(reader.as_ref(), index, mapping))
        .buffered(num_cpus::get());

    let mut ivf = Ivf {
        centroids: index.ivf.centroids.clone(),
        offsets: Vec::with_capacity(index.ivf.offsets.len()),
        lengths: Vec::with_capacity(index.ivf.lengths.len()),
    };
    while let Some(write_task) = task_stream.try_next().await? {
        write_task.write(&mut writer, &mut ivf).await?;
    }

    let pq_sub_index = index
        .sub_index
        .as_any()
        .downcast_ref::<PQIndex>()
        .ok_or_else(|| Error::NotSupported {
            source: "Remapping a non-pq sub-index".into(),
        })?;

    let metadata = IvfPQIndexMetadata {
        name,
        column,
        dimension: index.ivf.dimension() as u32,
        dataset_version: old_version,
        ivf,
        metric_type: index.metric_type,
        pq: pq_sub_index.pq.clone(),
        transforms,
    };

    let metadata = pb::Index::try_from(&metadata)?;
    let pos = writer.write_protobuf(&metadata).await?;
    writer.write_magics(pos).await?;
    writer.shutdown().await?;

    Ok(())
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
    pq: Arc<ProductQuantizer>,
    metric_type: MetricType,
    stream: impl RecordBatchStream + Unpin,
) -> Result<()> {
    let object_store = dataset.object_store();
    let path = dataset.indices_dir().child(uuid).child(INDEX_FILE_NAME);
    let mut writer = object_store.create(&path).await?;

    let start = std::time::Instant::now();
    let num_partitions = ivf.num_partitions() as u32;
    builder::build_partitions(
        &mut writer,
        stream,
        column,
        &mut ivf,
        pq.clone(),
        metric_type,
        0..num_partitions,
    )
    .await?;
    info!("Built IVF partitions: {}s", start.elapsed().as_secs_f32());

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
        pq,
        transforms,
    };

    let metadata = pb::Index::try_from(&metadata)?;
    let pos = writer.write_protobuf(&metadata).await?;
    writer.write_magics(pos).await?;
    writer.shutdown().await?;

    Ok(())
}

/// Train IVF partitions using kmeans.
async fn train_ivf_model(
    data: &MatrixView<Float32Type>,
    metric_type: MetricType,
    params: &IvfBuildParams,
) -> Result<Ivf> {
    let rng = SmallRng::from_entropy();
    const REDOS: usize = 1;
    let centroids = lance_index::vector::kmeans::train_kmeans(
        data.data().as_ref(),
        None,
        data.num_columns(),
        params.num_partitions,
        params.max_iters as u32,
        REDOS,
        rng,
        metric_type,
        params.sample_rate,
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

    use std::collections::HashMap;

    use arrow_array::{cast::AsArray, RecordBatchIterator, RecordBatchReader, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use lance_linalg::distance::l2_distance_batch;
    use lance_testing::datagen::{
        generate_random_array, generate_scaled_random_array, sample_without_replacement,
    };
    use rand::{seq::SliceRandom, thread_rng};
    use tempfile::tempdir;
    use uuid::Uuid;

    use crate::{
        format::RowAddress,
        index::{
            vector::{open_index, VectorIndexParams},
            DatasetIndexExt, IndexType,
        },
    };

    const DIM: usize = 32;

    /// This goal of this function is to generate data that behaves in a very deterministic way so that
    /// we can evaluate the correctness of an IVF_PQ implementation.  Currently it is restricted to the
    /// L2 distance metric.
    ///
    /// First, we generate a set of centroids.  These are generated randomly but we ensure that is
    /// sufficient distance between each of the centroids.
    ///
    /// Then, we generate 256 vectors per centroid.  Each vector is generated by making a line by
    /// adding / subtracting [1,1,1...,1] (with the centroid in the middle)
    ///
    /// The trained result should have our generated centroids (without these centroids actually being
    /// a part of the input data) and the PQ codes for every data point should be identical and, given
    /// any three data points a, b, and c that are in the same centroid then the distance between a and
    /// b should be different than the distance between a and c.
    struct WellKnownIvfPqData {
        dim: u32,
        num_centroids: u32,
        centroids: Option<Float32Array>,
        vectors: Option<Float32Array>,
    }

    impl WellKnownIvfPqData {
        // Right now we are assuming 8-bit codes
        const VALS_PER_CODE: u32 = 256;
        const COLUMN: &'static str = "vector";

        fn new(dim: u32, num_centroids: u32) -> Self {
            Self {
                dim,
                num_centroids,
                centroids: None,
                vectors: None,
            }
        }

        fn distance_between_points(&self) -> f32 {
            (self.dim as f32).sqrt()
        }

        fn generate_centroids(&self) -> Float32Array {
            const MAX_ATTEMPTS: u32 = 10;
            let distance_needed =
                self.distance_between_points() * Self::VALS_PER_CODE as f32 * 2_f32;
            let mut attempts_remaining = MAX_ATTEMPTS;
            let num_values = self.dim * self.num_centroids;
            while attempts_remaining > 0 {
                // Use some biggish numbers to ensure we get the distance we want but make them positive
                // and not too big for easier debugging.
                let centroids: Float32Array =
                    generate_scaled_random_array(num_values as usize, 0_f32, 1000_f32);
                let mut broken = false;
                for (index, centroid) in centroids
                    .values()
                    .chunks_exact(self.dim as usize)
                    .enumerate()
                {
                    let offset = (index + 1) * self.dim as usize;
                    let length = centroids.len() - offset;
                    if length == 0 {
                        // This will be true for the last item since we ignore comparison with self
                        continue;
                    }
                    let distances = l2_distance_batch(
                        centroid,
                        &centroids.values()[offset..offset + length],
                        self.dim as usize,
                    );
                    let min_distance = distances.min_by(|a, b| a.total_cmp(b)).unwrap();
                    // In theory we could just replace this one vector but, out of laziness, we just retry all of them
                    if min_distance < distance_needed {
                        broken = true;
                        break;
                    }
                }
                if !broken {
                    return centroids;
                }
                attempts_remaining -= 1;
            }
            panic!(
                "Unable to generate centroids with sufficient distance after {} attempts",
                MAX_ATTEMPTS
            );
        }

        fn get_centroids(&mut self) -> &Float32Array {
            if self.centroids.is_some() {
                return self.centroids.as_ref().unwrap();
            }
            self.centroids = Some(self.generate_centroids());
            self.centroids.as_ref().unwrap()
        }

        fn get_centroids_as_list_arr(&mut self) -> Arc<FixedSizeListArray> {
            Arc::new(
                FixedSizeListArray::try_new_from_values(
                    self.get_centroids().clone(),
                    self.dim as i32,
                )
                .unwrap(),
            )
        }

        fn generate_vectors(&mut self) -> Float32Array {
            let dim = self.dim as usize;
            let num_centroids = self.num_centroids;
            let centroids = self.get_centroids();
            let mut vectors: Vec<f32> =
                vec![0_f32; Self::VALS_PER_CODE as usize * dim * num_centroids as usize];
            for (centroid, dst_batch) in centroids
                .values()
                .chunks_exact(dim)
                .zip(vectors.chunks_exact_mut(dim * Self::VALS_PER_CODE as usize))
            {
                for (offset, dst) in (-128..0).chain(1..129).zip(dst_batch.chunks_exact_mut(dim)) {
                    for (cent_val, dst_val) in centroid.iter().zip(dst) {
                        *dst_val = *cent_val + offset as f32;
                    }
                }
            }
            Float32Array::from(vectors)
        }

        fn get_vectors(&mut self) -> &Float32Array {
            if self.vectors.is_some() {
                return self.vectors.as_ref().unwrap();
            }
            self.vectors = Some(self.generate_vectors());
            self.vectors.as_ref().unwrap()
        }

        fn get_vector(&mut self, idx: u32) -> Float32Array {
            let dim = self.dim as usize;
            let vectors = self.get_vectors();
            let start = idx as usize * dim;
            vectors.slice(start, dim)
        }

        fn generate_batches(&mut self) -> impl RecordBatchReader + Send + 'static {
            let dim = self.dim as usize;
            let vectors_array = self.get_vectors();

            let schema = Arc::new(Schema::new(vec![Field::new(
                Self::COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                true,
            )]));
            let array = Arc::new(
                FixedSizeListArray::try_new_from_values(vectors_array.clone(), dim as i32).unwrap(),
            );
            let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()]).unwrap();
            RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone())
        }

        async fn generate_dataset(&mut self, test_uri: &str) -> Result<Dataset> {
            let batches = self.generate_batches();
            Dataset::write(batches, test_uri, None).await
        }

        async fn check_index<F: Fn(u64) -> Option<u64>>(
            &mut self,
            index: &IVFIndex,
            prefilter: Arc<PreFilter>,
            ids_to_test: &Vec<u64>,
            row_id_map: F,
        ) {
            const ROWS_TO_TEST: u32 = 500;
            let num_vectors = ids_to_test.len() as u32 / self.dim;
            let num_tests = ROWS_TO_TEST.min(num_vectors);
            let row_ids_to_test = sample_without_replacement(ids_to_test, num_tests);
            for row_id in row_ids_to_test {
                let row = self.get_vector(row_id as u32);
                let query = Query {
                    column: Self::COLUMN.to_string(),
                    key: Arc::new(row),
                    k: 5,
                    nprobes: 1,
                    refine_factor: None,
                    metric_type: MetricType::L2,
                    use_index: true,
                };
                let search_result = index.search(&query, prefilter.clone()).await.unwrap();

                let found_ids = search_result.column(1);
                let found_ids = found_ids.as_any().downcast_ref::<UInt64Array>().unwrap();
                let expected_id = row_id_map(row_id);

                match expected_id {
                    // Original id was deleted, results can be anything, just make sure they don't
                    // include the original id
                    None => assert!(!found_ids.iter().any(|f_id| f_id.unwrap() == row_id)),
                    // Original id remains or was remapped, make sure expected id in results
                    Some(expected_id) => {
                        assert!(found_ids.iter().any(|f_id| f_id.unwrap() == expected_id))
                    }
                };
                // The invalid row id should never show up in results
                assert!(!found_ids
                    .iter()
                    .any(|f_id| f_id.unwrap() == RowAddress::TOMBSTONE_ROW));
            }
        }
    }

    async fn generate_test_dataset(test_uri: &str) -> (Dataset, Arc<FixedSizeListArray>) {
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

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write(batches, test_uri, None).await.unwrap();
        (dataset, array)
    }

    #[tokio::test]
    async fn test_create_ivf_pq_with_centroids() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let (mut dataset, vector_array) = generate_test_dataset(test_uri).await;

        let centroids = generate_random_array(2 * DIM);
        let ivf_centroids = FixedSizeListArray::try_new_from_values(centroids, DIM as i32).unwrap();
        let ivf_params = IvfBuildParams::try_with_centroids(2, Arc::new(ivf_centroids)).unwrap();

        let codebook = Arc::new(generate_random_array(256 * DIM));
        let pq_params = PQBuildParams::with_codebook(4, 8, codebook);

        let params = VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf_params, pq_params);

        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let sample_query = vector_array.value(10);
        let query = sample_query.as_primitive::<Float32Type>();
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

    fn partition_ids(mut ids: Vec<u64>, num_parts: u32) -> Vec<Vec<u64>> {
        if num_parts > ids.len() as u32 {
            panic!("Not enough ids to break into {num_parts} parts");
        }
        let mut rng = thread_rng();
        ids.shuffle(&mut rng);

        let values_per_part = ids.len() / num_parts as usize;
        let parts_with_one_extra = ids.len() % num_parts as usize;

        let mut parts = Vec::with_capacity(num_parts as usize);
        let mut offset = 0;
        for part_size in (0..num_parts).map(|part_idx| {
            if part_idx < parts_with_one_extra as u32 {
                values_per_part + 1
            } else {
                values_per_part
            }
        }) {
            parts.push(Vec::from_iter(
                ids[offset..(offset + part_size)].iter().copied(),
            ));
            offset += part_size;
        }

        parts
    }

    const BIG_OFFSET: u64 = 10000;

    fn build_mapping(
        row_ids_to_modify: &[u64],
        row_ids_to_remove: &[u64],
        max_id: u64,
    ) -> HashMap<u64, Option<u64>> {
        // Some big number we can add to row ids so they are remapped but don't intersect with anything
        if max_id > BIG_OFFSET {
            panic!("This logic will only work if the max row id is less than BIG_OFFSET");
        }
        row_ids_to_modify
            .iter()
            .copied()
            .map(|val| (val, Some(val + BIG_OFFSET)))
            .chain(row_ids_to_remove.iter().copied().map(|val| (val, None)))
            .collect()
    }

    #[tokio::test]
    async fn remap_ivf_pq_index() {
        // Use small numbers to keep runtime down
        const DIM: u32 = 8;
        const CENTROIDS: u32 = 2;
        const NUM_SUBVECTORS: u32 = 4;
        const NUM_BITS: u32 = 8;
        const INDEX_NAME: &str = "my_index";

        // In this test we create a sample dataset with reliable data, train an IVF PQ index
        // remap the rows, and then verify that we can still search the index and will get
        // back the remapped row ids.

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let mut test_data = WellKnownIvfPqData::new(DIM, CENTROIDS);

        let dataset = Arc::new(test_data.generate_dataset(test_uri).await.unwrap());
        let ivf_params = IvfBuildParams::try_with_centroids(
            CENTROIDS as usize,
            test_data.get_centroids_as_list_arr(),
        )
        .unwrap();
        let pq_params = PQBuildParams::new(NUM_SUBVECTORS as usize, NUM_BITS as usize);

        let uuid = Uuid::new_v4();
        let uuid_str = uuid.to_string();

        build_ivf_pq_index(
            &dataset,
            WellKnownIvfPqData::COLUMN,
            INDEX_NAME,
            &uuid_str,
            MetricType::L2,
            &ivf_params,
            &pq_params,
        )
        .await
        .unwrap();

        let index = open_index(dataset.clone(), WellKnownIvfPqData::COLUMN, &uuid_str)
            .await
            .unwrap();
        let ivf_index = index.as_any().downcast_ref::<IVFIndex>().unwrap();

        let index_meta = crate::format::Index {
            uuid,
            dataset_version: 0,
            fields: Vec::new(),
            name: INDEX_NAME.to_string(),
            fragment_bitmap: None,
        };

        let prefilter = Arc::new(PreFilter::new(dataset.clone(), index_meta, None));

        let is_not_remapped = Some;
        let is_remapped = |row_id| Some(row_id + BIG_OFFSET);
        let is_removed = |_| None;
        let max_id = test_data.get_vectors().len() as u64 / test_data.dim as u64;
        let row_ids = Vec::from_iter(0..max_id);

        // Sanity check to make sure the index we built is behaving correctly.  Any
        // input row, when used as a query, should be found in the results list with
        // the same id
        test_data
            .check_index(ivf_index, prefilter.clone(), &row_ids, is_not_remapped)
            .await;

        // When remapping we change the id of 1/3 of the rows, we remove another 1/3,
        // and we keep 1/3 as they are
        let partitioned_row_ids = partition_ids(row_ids, 3);
        let row_ids_to_modify = &partitioned_row_ids[0];
        let row_ids_to_remove = &partitioned_row_ids[1];
        let row_ids_to_remain = &partitioned_row_ids[2];

        let mapping = build_mapping(row_ids_to_modify, row_ids_to_remove, max_id);

        let new_uuid = Uuid::new_v4();
        let new_uuid_str = new_uuid.to_string();

        remap_index_file(
            &dataset,
            &uuid_str,
            &new_uuid_str,
            dataset.version().version,
            ivf_index,
            &mapping,
            INDEX_NAME.to_string(),
            WellKnownIvfPqData::COLUMN.to_string(),
            vec![],
        )
        .await
        .unwrap();

        let remapped = open_index(
            dataset.clone(),
            WellKnownIvfPqData::COLUMN,
            &new_uuid.to_string(),
        )
        .await
        .unwrap();
        let ivf_remapped = remapped.as_any().downcast_ref::<IVFIndex>().unwrap();

        // If the ids were remapped then make sure the new row id is in the results
        test_data
            .check_index(
                ivf_remapped,
                prefilter.clone(),
                row_ids_to_modify,
                is_remapped,
            )
            .await;
        // If the ids were removed then make sure the old row id isn't in the results
        test_data
            .check_index(
                ivf_remapped,
                prefilter.clone(),
                row_ids_to_remove,
                is_removed,
            )
            .await;
        // If the ids were not remapped then make sure they still return the old id
        test_data
            .check_index(
                ivf_remapped,
                prefilter.clone(),
                row_ids_to_remain,
                is_not_remapped,
            )
            .await;
    }

    #[tokio::test]
    async fn test_create_ivf_pq_cosine() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let (mut dataset, vector_array) = generate_test_dataset(test_uri).await;

        let centroids = generate_random_array(2 * DIM);
        let ivf_centroids = FixedSizeListArray::try_new_from_values(centroids, DIM as i32).unwrap();
        let ivf_params = IvfBuildParams::try_with_centroids(2, Arc::new(ivf_centroids)).unwrap();

        let codebook = Arc::new(generate_random_array(256 * DIM));
        let pq_params = PQBuildParams::with_codebook(4, 8, codebook);

        let params = VectorIndexParams::with_ivf_pq_params(MetricType::Cosine, ivf_params, pq_params);

        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let sample_query = vector_array.value(10);
        let query = sample_query.as_primitive::<Float32Type>();
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
        for batch in results.iter() {
            let dist = &batch["_distance"];
            assert!(dist.as_primitive::<Float32Type>().values().iter().all(|v| *v >= 0.0 && *v < 1.0));
        }
    }

}
