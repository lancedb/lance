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

//! Vector Index for Fast Approximate Nearest Neighbor (ANN) Search
//!

use std::any::Any;
use std::sync::Arc;

use arrow_array::Float32Array;

pub mod diskann;
pub mod flat;
#[allow(dead_code)]
mod graph;
pub mod ivf;
mod kmeans;
#[cfg(feature = "opq")]
pub mod opq;
pub mod pq;
mod traits;
mod utils;

use self::{
    ivf::{build_ivf_pq_index, IVFIndex, IvfBuildParams},
    pq::{PQBuildParams, PQIndex},
};

use super::{pb, IndexParams};
#[cfg(feature = "opq")]
use crate::index::vector::opq::{OPQIndex, OptimizedProductQuantizer};
use crate::{
    dataset::Dataset,
    index::{
        pb::vector_index_stage::Stage,
        vector::{
            diskann::{DiskANNIndex, DiskANNParams},
            ivf::Ivf,
            pq::ProductQuantizer,
        },
    },
    io::{
        object_reader::{read_message, ObjectReader},
        read_message_from_buf, read_metadata_offset,
    },
    linalg::{
        cosine::{cosine_distance, cosine_distance_batch},
        dot::{dot_distance, dot_distance_batch},
        l2::{l2_distance, l2_distance_batch},
    },
    Error, Result,
};
pub use traits::*;

pub(crate) const DIST_COL: &str = "_distance";
const INDEX_FILE_NAME: &str = "index.idx";

/// Query parameters for the vector indices
#[derive(Debug, Clone)]
pub struct Query {
    /// The column to be searched.
    pub column: String,

    /// The vector to be searched.
    pub key: Arc<Float32Array>,

    /// Top k results to return.
    pub k: usize,

    /// The number of probes to load and search.
    pub nprobes: usize,

    /// If presented, apply a refine step.
    /// TODO: should we support fraction / float number here?
    pub refine_factor: Option<u32>,

    /// Distance metric type
    pub metric_type: MetricType,

    /// Whether to use an ANN index if available
    pub use_index: bool,
}

/// Distance metrics type.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum MetricType {
    L2,
    Cosine,
    Dot, // Dot product
}

type DistanceFunc = dyn Fn(&[f32], &[f32]) -> f32 + Send + Sync + 'static;
type BatchDistanceFunc = dyn Fn(&[f32], &[f32], usize) -> Arc<Float32Array> + Send + Sync + 'static;

impl MetricType {
    /// Compute the distance from one vector to a batch of vectors.
    pub fn batch_func(&self) -> Arc<BatchDistanceFunc> {
        match self {
            Self::L2 => Arc::new(l2_distance_batch),
            Self::Cosine => Arc::new(cosine_distance_batch),
            Self::Dot => Arc::new(dot_distance_batch),
        }
    }

    /// Returns the distance function between two vectors.
    pub fn func(&self) -> Arc<DistanceFunc> {
        match self {
            Self::L2 => Arc::new(l2_distance),
            Self::Cosine => Arc::new(cosine_distance),
            Self::Dot => Arc::new(dot_distance),
        }
    }
}

impl std::fmt::Display for MetricType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::L2 => "l2",
                Self::Cosine => "cosine",
                Self::Dot => "dot",
            }
        )
    }
}

impl From<super::pb::VectorMetricType> for MetricType {
    fn from(proto: super::pb::VectorMetricType) -> Self {
        match proto {
            super::pb::VectorMetricType::L2 => Self::L2,
            super::pb::VectorMetricType::Cosine => Self::Cosine,
            super::pb::VectorMetricType::Dot => Self::Dot,
        }
    }
}

impl From<MetricType> for super::pb::VectorMetricType {
    fn from(mt: MetricType) -> Self {
        match mt {
            MetricType::L2 => Self::L2,
            MetricType::Cosine => Self::Cosine,
            MetricType::Dot => Self::Dot,
        }
    }
}

impl TryFrom<&str> for MetricType {
    type Error = Error;

    fn try_from(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "l2" | "euclidean" => Ok(Self::L2),
            "cosine" => Ok(Self::Cosine),
            "dot" => Ok(Self::Dot),
            _ => Err(Error::Index {
                message: format!("Metric type '{s}' is not supported"),
            }),
        }
    }
}

/// Parameters of each index stage.
#[derive(Debug)]
pub enum StageParams {
    Ivf(IvfBuildParams),

    PQ(PQBuildParams),

    DiskANN(DiskANNParams),
}

/// The parameters to build vector index.
pub struct VectorIndexParams {
    pub stages: Vec<StageParams>,

    /// Vector distance metrics type.
    pub metric_type: MetricType,
}

impl VectorIndexParams {
    /// Create index parameters for `IVF_PQ` index.
    ///
    /// Parameters
    ///
    ///  - `num_partitions`: the number of IVF partitions.
    ///  - `num_bits`: the number of bits to present the centroids used in PQ. Can only be `8` for now.
    ///  - `num_sub_vectors`: the number of sub vectors used in PQ.
    ///  - `metric_type`: how to compute distance, i.e., `L2` or `Cosine`.
    pub fn ivf_pq(
        num_partitions: usize,
        num_bits: u8,
        num_sub_vectors: usize,
        use_opq: bool,
        metric_type: MetricType,
        max_iterations: usize,
    ) -> Self {
        let mut stages: Vec<StageParams> = vec![];
        stages.push(StageParams::Ivf(IvfBuildParams::new(num_partitions)));

        let pq_params = PQBuildParams {
            num_bits: num_bits as usize,
            num_sub_vectors,
            use_opq,
            metric_type,
            max_iters: max_iterations,
            max_opq_iters: max_iterations,
            ..Default::default()
        };
        stages.push(StageParams::PQ(pq_params));

        Self {
            stages,
            metric_type,
        }
    }

    /// Create index parameters with `IVF` and `PQ` parameters, respectively.
    pub fn with_ivf_pq_params(
        metric_type: MetricType,
        ivf: IvfBuildParams,
        pq: PQBuildParams,
    ) -> Self {
        let stages = vec![StageParams::Ivf(ivf), StageParams::PQ(pq)];
        Self {
            stages,
            metric_type,
        }
    }

    pub fn with_diskann_params(metric_type: MetricType, diskann: DiskANNParams) -> Self {
        let stages = vec![StageParams::DiskANN(diskann)];
        Self {
            stages,
            metric_type,
        }
    }
}

impl IndexParams for VectorIndexParams {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn is_ivf_pq(stages: &[StageParams]) -> bool {
    if stages.len() < 2 {
        return false;
    }
    let len = stages.len();

    matches!(&stages[len - 1], StageParams::PQ(_))
        && matches!(&stages[len - 2], StageParams::Ivf(_))
}

fn is_diskann(stages: &[StageParams]) -> bool {
    if stages.is_empty() {
        return false;
    }
    let last = stages.last().unwrap();
    matches!(last, StageParams::DiskANN(_))
}

/// Build a Vector Index
pub(crate) async fn build_vector_index(
    dataset: &Dataset,
    column: &str,
    name: &str,
    uuid: &str,
    params: &VectorIndexParams,
) -> Result<()> {
    let stages = &params.stages;

    if stages.is_empty() {
        return Err(Error::Index {
            message: "Build Vector Index: must have at least 1 stage".to_string(),
        });
    };

    if is_ivf_pq(stages) {
        // This is a IVF PQ index.
        let len = stages.len();
        let StageParams::Ivf(ivf_params) = &stages[len - 2] else {
            return Err(Error::Index{message:
                format!("Build Vector Index: invalid stages: {:?}", stages),
            });
        };
        let StageParams::PQ(pq_params) = &stages[len - 1] else {
            return Err(Error::Index{message:
                format!("Build Vector Index: invalid stages: {:?}", stages),
            });
        };
        build_ivf_pq_index(
            dataset,
            column,
            name,
            uuid,
            params.metric_type,
            ivf_params,
            pq_params,
        )
        .await?
    } else if is_diskann(stages) {
        // This is DiskANN index.
        use self::diskann::build_diskann_index;
        let StageParams::DiskANN(params) = stages.last().unwrap() else {
            return Err(Error::Index{message:
                format!("Build Vector Index: invalid stages: {:?}", stages),
            });
        };
        build_diskann_index(dataset, column, name, uuid, params.clone()).await?;
    } else {
        return Err(Error::Index {
            message: format!("Build Vector Index: invalid stages: {:?}", stages),
        });
    }

    Ok(())
}

/// Open the Vector index on dataset, specified by the `uuid`.
pub(crate) async fn open_index(
    dataset: Arc<Dataset>,
    column: &str,
    uuid: &str,
) -> Result<Arc<dyn VectorIndex>> {
    if let Some(index) = dataset.session.index_cache.get(uuid) {
        return Ok(index);
    }

    let index_dir = dataset.indices_dir().child(uuid);
    let index_file = index_dir.child(INDEX_FILE_NAME);

    let object_store = dataset.object_store();
    let reader: Arc<dyn ObjectReader> = object_store.open(&index_file).await?.into();

    let file_size = reader.size().await?;
    let block_size = object_store.block_size();
    let begin = if file_size < block_size {
        0
    } else {
        file_size - block_size
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

    if proto.columns.len() != 1 {
        return Err(Error::Index {
            message: "VectorIndex only supports 1 column".to_string(),
        });
    }
    assert_eq!(proto.index_type, pb::IndexType::Vector as i32);

    let Some(idx_impl) = proto.implementation.as_ref() else {
        return Err(Error::Index{message:"Invalid protobuf for VectorIndex metadata".to_string()});
    };

    let pb::index::Implementation::VectorIndex(vec_idx) = idx_impl;

    let metric_type = pb::VectorMetricType::from_i32(vec_idx.metric_type)
        .ok_or(Error::Index {
            message: format!("Unsupported metric type value: {}", vec_idx.metric_type),
        })?
        .into();

    let mut last_stage: Option<Arc<dyn VectorIndex>> = None;

    for stg in vec_idx.stages.iter().rev() {
        match stg.stage.as_ref() {
            #[allow(unused_variables)]
            Some(Stage::Transform(tf)) => {
                if last_stage.is_none() {
                    return Err(Error::Index {
                        message: format!("Invalid vector index stages: {:?}", vec_idx.stages),
                    });
                }
                #[cfg(feature = "opq")]
                match tf.r#type() {
                    pb::TransformType::Opq => {
                        let opq = OptimizedProductQuantizer::load(
                            reader.as_ref(),
                            tf.position as usize,
                            tf.shape
                                .iter()
                                .map(|s| *s as usize)
                                .collect::<Vec<_>>()
                                .as_slice(),
                        )
                        .await?;
                        last_stage = Some(Arc::new(OPQIndex::new(
                            last_stage.as_ref().unwrap().clone(),
                            opq,
                        )));
                    }
                }
            }
            Some(Stage::Ivf(ivf_pb)) => {
                if last_stage.is_none() {
                    return Err(Error::Index {
                        message: format!("Invalid vector index stages: {:?}", vec_idx.stages),
                    });
                }
                let ivf = Ivf::try_from(ivf_pb)?;
                last_stage = Some(Arc::new(IVFIndex::try_new(
                    dataset.session.clone(),
                    uuid,
                    ivf,
                    reader.clone(),
                    last_stage.unwrap(),
                    metric_type,
                )?));
            }
            Some(Stage::Pq(pq_proto)) => {
                if last_stage.is_some() {
                    return Err(Error::Index {
                        message: format!("Invalid vector index stages: {:?}", vec_idx.stages),
                    });
                };
                let pq = Arc::new(ProductQuantizer::try_from(pq_proto).unwrap());
                last_stage = Some(Arc::new(PQIndex::new(pq, metric_type)));
            }
            Some(Stage::Diskann(diskann_proto)) => {
                if last_stage.is_some() {
                    return Err(Error::Index {
                        message: format!(
                            "DiskANN should be the only stage, but we got stages: {:?}",
                            vec_idx.stages
                        ),
                    });
                };
                let graph_path = index_dir.child(diskann_proto.filename.as_str());
                let diskann =
                    Arc::new(DiskANNIndex::try_new(dataset.clone(), column, &graph_path).await?);
                last_stage = Some(diskann);
            }
            _ => {}
        }
    }

    if last_stage.is_none() {
        return Err(Error::Index {
            message: format!("Invalid index stages: {:?}", vec_idx.stages),
        });
    }
    let idx = last_stage.unwrap();
    dataset.session.index_cache.insert(uuid, idx.clone());
    Ok(idx)
}
