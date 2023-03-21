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

pub mod flat;
pub mod ivf;
mod kmeans;
mod opq;
mod pq;
mod traits;

use self::{ivf::IVFIndex, pq::PQIndex};

use super::{pb, IndexParams};
use crate::{
    dataset::Dataset,
    index::{
        pb::vector_index_stage::Stage,
        vector::{ivf::Ivf, pq::ProductQuantizer},
    },
    io::{
        object_reader::{read_message, ObjectReader},
        read_message_from_buf, read_metadata_offset,
    },
    utils::distance::{cosine::cosine_distance, l2::l2_distance},
    Error, Result,
};
pub use traits::*;

const MAX_ITERATIONS: usize = 50;
/// Maximum number of iterations for OPQ.
/// See OPQ paper for details.
const MAX_OPQ_ITERATIONS: usize = 100;
const SCORE_COL: &str = "score";
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
}

impl MetricType {
    pub fn func(
        &self,
    ) -> Arc<dyn Fn(&Float32Array, &Float32Array, usize) -> Result<Arc<Float32Array>> + Send + Sync>
    {
        match self {
            Self::L2 => Arc::new(l2_distance),
            Self::Cosine => Arc::new(cosine_distance),
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
            }
        )
    }
}

impl From<super::pb::VectorMetricType> for MetricType {
    fn from(proto: super::pb::VectorMetricType) -> Self {
        match proto {
            super::pb::VectorMetricType::L2 => Self::L2,
            super::pb::VectorMetricType::Cosine => Self::Cosine,
        }
    }
}

impl From<MetricType> for super::pb::VectorMetricType {
    fn from(mt: MetricType) -> Self {
        match mt {
            MetricType::L2 => Self::L2,
            MetricType::Cosine => Self::Cosine,
        }
    }
}

impl TryFrom<&str> for MetricType {
    type Error = Error;

    fn try_from(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "l2" | "euclidean" => Ok(Self::L2),
            "cosine" => Ok(Self::Cosine),
            _ => Err(Error::Index(format!("Metric type '{s}' is not supported"))),
        }
    }
}

/// The parameters to build vector index.
pub struct VectorIndexParams {
    // This is hard coded for IVF_PQ for now. Can refactor later to support more.
    /// The number of IVF partitions
    pub num_partitions: u32,

    /// the number of bits to present the centroids used in PQ.
    pub nbits: u8,

    /// Use Optimized Product Quantizer.
    pub use_opq: bool,

    /// the number of sub vectors used in PQ.
    pub num_sub_vectors: u32,

    /// Vector distance metrics type.
    pub metric_type: MetricType,

    /// Max number of iterations to train a KMean model
    pub max_iterations: usize,

    /// Max number of iterations to train a OPQ model.
    pub max_opq_iterations: usize,
}

impl VectorIndexParams {
    /// Create index parameters for `IVF_PQ` index.
    ///
    /// Parameters
    ///
    ///  - `num_partitions`: the number of IVF partitions.
    ///  - `nbits`: the number of bits to present the centroids used in PQ. Can only be `8` for now.
    ///  - `num_sub_vectors`: the number of sub vectors used in PQ.
    ///  - `metric_type`: how to compute distance, i.e., `L2` or `Cosine`.
    pub fn ivf_pq(
        num_partitions: u32,
        nbits: u8,
        num_sub_vectors: u32,
        use_opq: bool,
        metric_type: MetricType,
        max_iterations: usize,
    ) -> Self {
        Self {
            num_partitions,
            nbits,
            num_sub_vectors,
            use_opq,
            metric_type,
            max_iterations,
            max_opq_iterations: max_iterations,
        }
    }
}

impl Default for VectorIndexParams {
    fn default() -> Self {
        Self {
            num_partitions: 32,
            nbits: 8,
            num_sub_vectors: 16,
            use_opq: true,
            metric_type: MetricType::L2,
            max_iterations: MAX_ITERATIONS, // Faiss
            max_opq_iterations: MAX_OPQ_ITERATIONS,
        }
    }
}

impl IndexParams for VectorIndexParams {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Open the Vector index on dataset, specified by the `uuid`.
pub async fn open_index(dataset: &Dataset, uuid: &str) -> Result<Arc<dyn VectorIndex>> {
    let index_dir = dataset.indices_dir().child(uuid);
    let index_file = index_dir.child(INDEX_FILE_NAME);

    let object_store = dataset.object_store();
    let reader: Arc<dyn ObjectReader> = object_store.open(&index_file).await?.into();

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

    if proto.columns.len() != 1 {
        return Err(Error::Index(
            "VectorIndex only supports 1 column".to_string(),
        ));
    }
    assert_eq!(proto.index_type, pb::IndexType::Vector as i32);

    let Some(idx_impl) = proto.implementation.as_ref() else {
        return Err(Error::Index("Invalid protobuf for VectorIndex metadata".to_string()));
    };

    let vec_idx = match idx_impl {
        pb::index::Implementation::VectorIndex(vi) => vi,
    };

    let num_stages = vec_idx.stages.len();
    if num_stages != 2 && num_stages != 3 {
        return Err(Error::IO("Only support IVF_(O)PQ now".to_string()));
    };

    let metric_type = pb::VectorMetricType::from_i32(vec_idx.metric_type)
        .ok_or(Error::Index(format!(
            "Unsupported metric type value: {}",
            vec_idx.metric_type
        )))?
        .into();

    let mut last_stage: Option<Arc<dyn VectorIndex>> = None;
    for stg in vec_idx.stages.iter().rev() {
        match stg.stage.as_ref() {
            Some(Stage::Transform(tf)) => {
                // stages.push(Arc::new(tf.clone()));
            }
            Some(Stage::Ivf(ivf_pb)) => {
                if last_stage.is_none() {
                    return Err(Error::Index(format!(
                        "Invalid vector index stages: {:?}",
                        vec_idx.stages
                    )));
                }
                let ivf = Ivf::try_from(ivf_pb)?;
                last_stage = Some(Arc::new(IVFIndex::new(
                    ivf,
                    reader.clone(),
                    last_stage.as_ref()
                        .unwrap()
                        .as_any()
                        .downcast_ref::<Arc<dyn LoadableVectorIndex>>()
                        .ok_or_else(|| {
                            Error::Index(format!(
                                "Expected a LoadableVectorIndex, got: {:?}",
                                last_stage
                            ))
                        })?
                        .clone(),
                    metric_type,
                )));

                // stages.push(Arc::new(Ivf::try_from(ivf_pb)?));
            }
            Some(Stage::Pq(pq_proto)) => {
                if last_stage.is_some() {
                    return Err(Error::Index(format!(
                        "Invalid vector index stages: {:?}",
                        vec_idx.stages
                    )));
                };
                let pq = Arc::new(ProductQuantizer::try_from(pq_proto).unwrap());
                last_stage = Some(Arc::new(PQIndex::new(pq, metric_type)));
            }
            _ => {}
        }
    }

    if last_stage.is_none() {
        return Err(Error::Index(format!(
            "Invalid index stages: {:?}",
            vec_idx.stages
        )));
    }
    Ok(last_stage.unwrap())
}
