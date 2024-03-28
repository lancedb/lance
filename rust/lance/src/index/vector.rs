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

use std::sync::Arc;
use std::{any::Any, collections::HashMap};

pub mod hnsw;
pub mod ivf;
pub mod pq;
mod traits;
mod utils;

#[cfg(test)]
mod fixture_test;

use lance_file::reader::FileReader;
use lance_index::vector::hnsw::HNSW;
use lance_index::vector::ivf::storage::IvfData;
use lance_index::vector::{hnsw::builder::HnswBuildParams, ivf::IvfBuildParams, pq::PQBuildParams};
use lance_index::{INDEX_AUXILIARY_FILE_NAME, INDEX_METADATA_SCHEMA_KEY};
use lance_io::traits::Reader;
use lance_linalg::distance::*;
use lance_table::format::Index as IndexMetadata;
use snafu::{location, Location};
use tracing::instrument;
use uuid::Uuid;

use self::hnsw::{HNSWIndex, HNSWIndexOptions};
use self::{
    ivf::{build_ivf_hnsw_index, build_ivf_pq_index, remap_index_file, IVFIndex},
    pq::PQIndex,
};

use super::{pb, DatasetIndexInternalExt, IndexParams};
use crate::{
    dataset::Dataset,
    index::{pb::vector_index_stage::Stage, vector::ivf::Ivf},
    Error, Result,
};
pub use traits::*;

/// Parameters of each index stage.
#[derive(Debug, Clone)]
pub enum StageParams {
    Ivf(IvfBuildParams),

    PQ(PQBuildParams),

    Hnsw(HnswBuildParams),
}

/// The parameters to build vector index.
#[derive(Debug, Clone)]
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

    /// Create index parameters with `IVF`, `PQ` and `HNSW` parameters, respectively.
    /// This is used for `IVF_HNSW` index.
    pub fn with_ivf_hnsw_pq_params(
        metric_type: MetricType,
        ivf: IvfBuildParams,
        hnsw: HnswBuildParams,
        pq: PQBuildParams,
    ) -> Self {
        let stages = vec![
            StageParams::Ivf(ivf),
            StageParams::Hnsw(hnsw),
            StageParams::PQ(pq),
        ];
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

fn is_ivf_hnsw_pq(stages: &[StageParams]) -> bool {
    if stages.len() < 3 {
        return false;
    }
    let len = stages.len();

    matches!(&stages[len - 1], StageParams::PQ(_))
        && matches!(&stages[len - 2], StageParams::Hnsw(_))
        && matches!(&stages[len - 3], StageParams::Ivf(_))
}

/// Build a Vector Index
#[instrument(level = "debug", skip(dataset))]
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
            location: location!(),
        });
    };

    if is_ivf_pq(stages) {
        // This is a IVF PQ index.
        let len = stages.len();
        let StageParams::Ivf(ivf_params) = &stages[len - 2] else {
            return Err(Error::Index {
                message: format!("Build Vector Index: invalid stages: {:?}", stages),
                location: location!(),
            });
        };
        let StageParams::PQ(pq_params) = &stages[len - 1] else {
            return Err(Error::Index {
                message: format!("Build Vector Index: invalid stages: {:?}", stages),
                location: location!(),
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
    } else if is_ivf_hnsw_pq(stages) {
        let len = stages.len();
        let StageParams::Ivf(ivf_params) = &stages[len - 3] else {
            return Err(Error::Index {
                message: format!("Build Vector Index: invalid stages: {:?}", stages),
                location: location!(),
            });
        };
        let StageParams::Hnsw(hnsw_params) = &stages[len - 2] else {
            return Err(Error::Index {
                message: format!("Build Vector Index: invalid stages: {:?}", stages),
                location: location!(),
            });
        };
        let StageParams::PQ(pq_params) = &stages[len - 1] else {
            return Err(Error::Index {
                message: format!("Build Vector Index: invalid stages: {:?}", stages),
                location: location!(),
            });
        };
        build_ivf_hnsw_index(
            dataset,
            column,
            name,
            uuid,
            params.metric_type,
            ivf_params,
            hnsw_params,
            pq_params,
        )
        .await?
    } else {
        return Err(Error::Index {
            message: format!("Build Vector Index: invalid stages: {:?}", stages),
            location: location!(),
        });
    }

    Ok(())
}

#[instrument(level = "debug", skip_all, fields(old_uuid = old_uuid.to_string(), new_uuid = new_uuid.to_string(), num_rows = mapping.len()))]
pub(crate) async fn remap_vector_index(
    dataset: Arc<Dataset>,
    column: &str,
    old_uuid: &Uuid,
    new_uuid: &Uuid,
    old_metadata: &IndexMetadata,
    mapping: &HashMap<u64, Option<u64>>,
) -> Result<()> {
    let old_index = dataset
        .open_vector_index(column, &old_uuid.to_string())
        .await?;
    old_index.check_can_remap()?;
    let ivf_index: &IVFIndex =
        old_index
            .as_any()
            .downcast_ref()
            .ok_or_else(|| Error::NotSupported {
                source: "Only IVF indexes can be remapped currently".into(),
                location: location!(),
            })?;

    remap_index_file(
        dataset.as_ref(),
        &old_uuid.to_string(),
        &new_uuid.to_string(),
        old_metadata.dataset_version,
        ivf_index,
        mapping,
        old_metadata.name.clone(),
        column.to_string(),
        // We can safely assume there are no transforms today.  We assert above that the
        // top stage is IVF and IVF does not support transforms between IVF and PQ.  This
        // will be fixed in the future.
        vec![],
    )
    .await?;
    Ok(())
}

/// Open the Vector index on dataset, specified by the `uuid`.
#[instrument(level = "debug", skip(dataset, vec_idx, reader))]
pub(crate) async fn open_vector_index(
    dataset: Arc<Dataset>,
    column: &str,
    uuid: &str,
    vec_idx: &lance_index::pb::VectorIndex,
    reader: Arc<dyn Reader>,
) -> Result<Arc<dyn VectorIndex>> {
    let metric_type = pb::VectorMetricType::try_from(vec_idx.metric_type)?.into();

    let mut last_stage: Option<Arc<dyn VectorIndex>> = None;

    for stg in vec_idx.stages.iter().rev() {
        match stg.stage.as_ref() {
            #[allow(unused_variables)]
            Some(Stage::Transform(tf)) => {
                if last_stage.is_none() {
                    return Err(Error::Index {
                        message: format!("Invalid vector index stages: {:?}", vec_idx.stages),
                        location: location!(),
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
                        location: location!(),
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
                        location: location!(),
                    });
                };
                let pq = lance_index::vector::pq::builder::from_proto(pq_proto, metric_type)?;
                last_stage = Some(Arc::new(PQIndex::new(pq, metric_type)));
            }
            Some(Stage::Diskann(_)) => {
                return Err(Error::Index {
                    message: "DiskANN support is removed from Lance.".to_string(),
                    location: location!(),
                });
            }
            _ => {}
        }
    }

    if last_stage.is_none() {
        return Err(Error::Index {
            message: format!("Invalid index stages: {:?}", vec_idx.stages),
            location: location!(),
        });
    }
    let idx = last_stage.unwrap();
    dataset.session.index_cache.insert_vector(uuid, idx.clone());
    Ok(idx)
}

#[instrument(level = "debug", skip(dataset, reader))]
pub(crate) async fn open_vector_index_v2(
    dataset: Arc<Dataset>,
    column: &str,
    uuid: &str,
    reader: FileReader,
) -> Result<Arc<dyn VectorIndex>> {
    let index_metadata = reader
        .schema()
        .metadata
        .get(INDEX_METADATA_SCHEMA_KEY)
        .ok_or(Error::Index {
            message: "Index Metadata not found".to_owned(),
            location: location!(),
        })?;
    let index_metadata: lance_index::IndexMetadata = serde_json::from_str(index_metadata)?;
    let distance_type = DistanceType::try_from(index_metadata.distance_type.as_str())?;

    let aux_path = dataset
        .indices_dir()
        .child(uuid)
        .child(INDEX_AUXILIARY_FILE_NAME);
    let aux_reader = dataset.object_store().open(&aux_path).await?;

    let index: Arc<dyn VectorIndex> = match index_metadata.index_type.as_str() {
        "IVF_HNSW" => {
            let ivf_data = IvfData::load(&reader).await?;
            let options = HNSWIndexOptions { use_residual: true };
            let hnsw = HNSWIndex::try_new(
                HNSW::empty(),
                reader.object_reader.clone(),
                aux_reader.into(),
                options,
            )
            .await?;
            let pb_ivf = pb::Ivf::try_from(&ivf_data)?;
            let ivf = Ivf::try_from(&pb_ivf)?;

            Arc::new(IVFIndex::try_new(
                dataset.session.clone(),
                uuid,
                ivf,
                reader.object_reader.clone(),
                Arc::new(hnsw),
                distance_type,
            )?)
        }

        _ => {
            return Err(Error::Index {
                message: format!("Unsupported index type: {}", index_metadata.index_type),
                location: location!(),
            })
        }
    };

    dataset
        .session
        .index_cache
        .insert_vector(uuid, index.clone());

    Ok(index)
}
