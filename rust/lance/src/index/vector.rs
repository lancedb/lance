// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Index for Fast Approximate Nearest Neighbor (ANN) Search
//!

use std::sync::Arc;
use std::{any::Any, collections::HashMap};

pub mod builder;
pub mod ivf;
pub mod pq;
mod traits;
mod utils;

#[cfg(test)]
mod fixture_test;

use arrow::datatypes::Float32Type;
use builder::IvfIndexBuilder;
use lance_file::reader::FileReader;
use lance_index::vector::flat::index::{FlatIndex, FlatQuantizer};
use lance_index::vector::hnsw::HNSW;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::pq::ProductQuantizerImpl;
use lance_index::vector::v3::shuffler::IvfShuffler;
use lance_index::vector::{
    hnsw::{
        builder::HnswBuildParams,
        index::{HNSWIndex, HNSWIndexOptions},
    },
    ivf::IvfBuildParams,
    pq::PQBuildParams,
    sq::{builder::SQBuildParams, ScalarQuantizer},
    VectorIndex,
};
use lance_index::{IndexType, INDEX_AUXILIARY_FILE_NAME, INDEX_METADATA_SCHEMA_KEY};
use lance_io::traits::Reader;
use lance_linalg::distance::*;
use lance_table::format::Index as IndexMetadata;
use snafu::{location, Location};
use tracing::instrument;
use uuid::Uuid;

use self::{ivf::*, pq::PQIndex};

use super::{pb, DatasetIndexInternalExt, IndexParams};
use crate::{dataset::Dataset, index::pb::vector_index_stage::Stage, Error, Result};
pub use traits::*;

pub const LANCE_VECTOR_INDEX: &str = "__lance_vector_index";

/// Parameters of each index stage.
#[derive(Debug, Clone)]
pub enum StageParams {
    Ivf(IvfBuildParams),
    Hnsw(HnswBuildParams),
    PQ(PQBuildParams),
    SQ(SQBuildParams),
}

/// The parameters to build vector index.
#[derive(Debug, Clone)]
pub struct VectorIndexParams {
    pub stages: Vec<StageParams>,

    /// Vector distance metrics type.
    pub metric_type: MetricType,
}

impl VectorIndexParams {
    pub fn ivf_flat(num_partitions: usize, metric_type: MetricType) -> Self {
        let ivf_params = IvfBuildParams::new(num_partitions);
        let stages = vec![StageParams::Ivf(ivf_params)];
        Self {
            stages,
            metric_type,
        }
    }

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
        metric_type: MetricType,
        max_iterations: usize,
    ) -> Self {
        let mut stages: Vec<StageParams> = vec![];
        stages.push(StageParams::Ivf(IvfBuildParams::new(num_partitions)));

        let pq_params = PQBuildParams {
            num_bits: num_bits as usize,
            num_sub_vectors,
            max_iters: max_iterations,
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
    /// This is used for `IVF_HNSW_PQ` index.
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

    /// Create index parameters with `IVF`, `HNSW` and `SQ` parameters, respectively.
    /// This is used for `IVF_HNSW_SQ` index.
    pub fn with_ivf_hnsw_sq_params(
        metric_type: MetricType,
        ivf: IvfBuildParams,
        hnsw: HnswBuildParams,
        sq: SQBuildParams,
    ) -> Self {
        let stages = vec![
            StageParams::Ivf(ivf),
            StageParams::Hnsw(hnsw),
            StageParams::SQ(sq),
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

    fn index_type(&self) -> IndexType {
        IndexType::Vector
    }

    fn index_name(&self) -> &str {
        LANCE_VECTOR_INDEX
    }
}

fn is_ivf_flat(stages: &[StageParams]) -> bool {
    if stages.len() != 1 {
        return false;
    }

    matches!(&stages[0], StageParams::Ivf(_))
}

fn is_ivf_pq(stages: &[StageParams]) -> bool {
    if stages.len() < 2 {
        return false;
    }
    let len = stages.len();

    matches!(&stages[len - 1], StageParams::PQ(_))
        && matches!(&stages[len - 2], StageParams::Ivf(_))
}

fn is_ivf_hnsw(stages: &[StageParams]) -> bool {
    if stages.len() < 2 {
        return false;
    }

    matches!(&stages[0], StageParams::Ivf(_)) && matches!(&stages[1], StageParams::Hnsw(_))
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

    let temp_dir = tempfile::tempdir()?;
    let path = temp_dir.path().to_str().unwrap().into();
    if is_ivf_flat(stages) {
        let StageParams::Ivf(ivf_params) = &stages[0] else {
            return Err(Error::Index {
                message: format!("Build Vector Index: invalid stages: {:?}", stages),
                location: location!(),
            });
        };

        let shuffler = IvfShuffler::new(path, ivf_params.num_partitions);
        IvfIndexBuilder::<FlatIndex, FlatQuantizer>::new(
            dataset.clone(),
            column.to_owned(),
            dataset.indices_dir().child(uuid),
            params.metric_type,
            Box::new(shuffler),
            Some(ivf_params.clone()),
            Some(()),
            (),
        )?
        .build()
        .await?;
    } else if is_ivf_pq(stages) {
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
    } else if is_ivf_hnsw(stages) {
        let len = stages.len();
        let StageParams::Ivf(ivf_params) = &stages[0] else {
            return Err(Error::Index {
                message: format!("Build Vector Index: invalid stages: {:?}", stages),
                location: location!(),
            });
        };
        let StageParams::Hnsw(hnsw_params) = &stages[1] else {
            return Err(Error::Index {
                message: format!("Build Vector Index: invalid stages: {:?}", stages),
                location: location!(),
            });
        };

        let shuffler = IvfShuffler::new(path, ivf_params.num_partitions);
        // with quantization
        if len > 2 {
            match stages.last().unwrap() {
                StageParams::PQ(pq_params) => {
                    build_ivf_hnsw_pq_index(
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
                }
                StageParams::SQ(sq_params) => {
                    IvfIndexBuilder::<HNSW, ScalarQuantizer>::new(
                        dataset.clone(),
                        column.to_owned(),
                        dataset.indices_dir().child(uuid),
                        params.metric_type,
                        Box::new(shuffler),
                        Some(ivf_params.clone()),
                        Some(sq_params.clone()),
                        hnsw_params.clone(),
                    )?
                    .build()
                    .await?;
                }
                _ => {
                    return Err(Error::Index {
                        message: format!("Build Vector Index: invalid stages: {:?}", stages),
                        location: location!(),
                    });
                }
            }
        }
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
            }
            Some(Stage::Ivf(ivf_pb)) => {
                if last_stage.is_none() {
                    return Err(Error::Index {
                        message: format!("Invalid vector index stages: {:?}", vec_idx.stages),
                        location: location!(),
                    });
                }
                let ivf = IvfModel::try_from(ivf_pb.to_owned())?;
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

    let index: Arc<dyn VectorIndex> = match index_metadata.index_type.as_str() {
        "IVF_HNSW_PQ" => {
            let aux_path = dataset
                .indices_dir()
                .child(uuid)
                .child(INDEX_AUXILIARY_FILE_NAME);
            let aux_reader = dataset.object_store().open(&aux_path).await?;

            let ivf_data = IvfModel::load(&reader).await?;
            let options = HNSWIndexOptions { use_residual: true };
            let hnsw = HNSWIndex::<ProductQuantizerImpl<Float32Type>>::try_new(
                reader.object_reader.clone(),
                aux_reader.into(),
                options,
            )
            .await?;
            let pb_ivf = pb::Ivf::try_from(&ivf_data)?;
            let ivf = IvfModel::try_from(pb_ivf)?;

            Arc::new(IVFIndex::try_new(
                dataset.session.clone(),
                uuid,
                ivf,
                reader.object_reader.clone(),
                Arc::new(hnsw),
                distance_type,
            )?)
        }

        "IVF_HNSW_SQ" => {
            let aux_path = dataset
                .indices_dir()
                .child(uuid)
                .child(INDEX_AUXILIARY_FILE_NAME);
            let aux_reader = dataset.object_store().open(&aux_path).await?;

            let ivf_data = IvfModel::load(&reader).await?;
            let options = HNSWIndexOptions {
                use_residual: false,
            };

            let hnsw = HNSWIndex::<ScalarQuantizer>::try_new(
                reader.object_reader.clone(),
                aux_reader.into(),
                options,
            )
            .await?;
            let pb_ivf = pb::Ivf::try_from(&ivf_data)?;
            let ivf = IvfModel::try_from(pb_ivf)?;

            Arc::new(IVFIndex::try_new(
                dataset.session.clone(),
                uuid,
                ivf,
                reader.object_reader.clone(),
                Arc::new(hnsw),
                distance_type,
            )?)
        }

        index_type => {
            if let Some(ext) = dataset
                .session
                .index_extensions
                .get(&(IndexType::Vector, index_type.to_string()))
            {
                ext.clone()
                    .to_vector()
                    .ok_or(Error::Internal {
                        message: "unable to cast index extension to vector".to_string(),
                        location: location!(),
                    })?
                    .load_index(dataset.clone(), column, uuid, reader)
                    .await?
            } else {
                return Err(Error::Index {
                    message: format!("Unsupported index type: {}", index_metadata.index_type),
                    location: location!(),
                });
            }
        }
    };

    Ok(index)
}
