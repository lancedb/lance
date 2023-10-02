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

use arrow_arith::arithmetic::subtract_dyn;
use arrow_array::{
    cast::{as_primitive_array, as_struct_array, AsArray},
    types::Float32Type,
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StructArray, UInt32Array,
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
use lance_linalg::{distance::*, kernels::argmin, matrix::MatrixView};
use log::info;
use rand::{rngs::SmallRng, SeedableRng};
use serde::Serialize;
use snafu::{location, Location};
use tracing::{instrument, span, Level};

#[cfg(feature = "opq")]
use super::opq::train_opq;
use super::{
    pq::{train_pq, PQBuildParams, ProductQuantizer},
    utils::maybe_sample_training_data,
    MetricType, Query, VectorIndex, INDEX_FILE_NAME,
};
use crate::{
    dataset::Dataset,
    datatypes::Field,
    index::{pb, prefilter::PreFilter, vector::Transformer, Index},
    io::RecordBatchStream,
};
use crate::{
    io::{local::to_local_path, object_reader::ObjectReader},
    session::Session,
};
use crate::{Error, Result};

mod builder;
mod io;
mod shuffler;

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
        if query.len() != self.dimension() {
            return Err(Error::IO {
                message: format!(
                    "Ivf::find_partition: dimension mismatch: {} != {}",
                    query.len(),
                    self.dimension()
                ),
                location: location!(),
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

    /// Compute the partition for each row in the input Matrix.
    ///
    #[instrument(skip(data))]
    pub fn compute_partitions(
        &self,
        data: &MatrixView<Float32Type>,
        metric_type: MetricType,
    ) -> UInt32Array {
        let ndim = data.ndim();
        let centroids_arr: &Float32Array = self.centroids.values().as_primitive();
        let centroid_norms = centroids_arr
            .values()
            .chunks(ndim)
            .map(|centroid| centroid.norm_l2())
            .collect::<Vec<_>>();
        UInt32Array::from_iter_values(data.iter().map(|row| {
            argmin(
                centroids_arr
                    .values()
                    .chunks(ndim)
                    .zip(centroid_norms.iter())
                    .map(|(centroid, &norm)| match metric_type {
                        MetricType::L2 => row.l2(centroid),
                        MetricType::Cosine => centroid.cosine_fast(norm, row),
                        MetricType::Dot => row.dot(centroid),
                    }),
            )
            .expect("argmin should always return a value")
        }))
    }

    /// Compute residual vector.
    ///
    /// A residual vector is `original vector - centroids`.
    ///
    /// Parameters:
    ///  - *original*: original vector.
    ///  - *partitions*: partition ID of each original vector.
    #[instrument(skip_all)]
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
        ProductQuantizer::new_with_codebook(
            pq_params.num_sub_vectors,
            pq_params.num_bits as u32,
            dim,
            codebook.clone(),
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
        // Compute the residual vector to train Product Quantizer.
        let part_ids = span!(Level::INFO, "compute partition")
            .in_scope(|| ivf_model.compute_partitions(&training_data, metric_type));

        let residuals = span!(Level::INFO, "compute residual")
            .in_scope(|| ivf_model.compute_residual(&training_data, &part_ids));
        train_pq(&residuals, pq_params).await?
    };
    info!("Trained PQ in: {} seconds", start.elapsed().as_secs_f32());

    // Transform data, compute residuals and sort by partition ids.
    let mut scanner = dataset.scan();
    scanner.batch_readahead(num_cpus::get() * 2);
    scanner.project(&[column])?;
    scanner.with_row_id();

    let metric_type = pq_params.metric_type;

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
        pq,
        metric_type,
        stream,
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
        &pq,
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
        pq: pq.into(),
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
    let centroids = super::kmeans::train_kmeans(
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
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

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
