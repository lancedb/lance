// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Index for Fast Approximate Nearest Neighbor (ANN) Search
//!

use std::sync::Arc;
use std::{any::Any, collections::HashMap};

pub mod builder;
pub mod ivf;
pub mod pq;
pub mod utils;

#[cfg(test)]
mod fixture_test;

use self::{ivf::*, pq::PQIndex};
use arrow_schema::DataType;
use builder::IvfIndexBuilder;
use lance_file::reader::FileReader;
use lance_index::frag_reuse::FragReuseIndex;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::vector::flat::index::{FlatBinQuantizer, FlatIndex, FlatQuantizer};
use lance_index::vector::hnsw::HNSW;
use lance_index::vector::ivf::builder::recommended_num_partitions;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::pq::ProductQuantizer;
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
use lance_index::{
    DatasetIndexExt, IndexType, INDEX_AUXILIARY_FILE_NAME, INDEX_METADATA_SCHEMA_KEY,
};
use lance_io::traits::Reader;
use lance_linalg::distance::*;
use lance_table::format::Index as IndexMetadata;
use object_store::path::Path;
use serde::Serialize;
use snafu::location;
use tempfile::tempdir;
use tracing::instrument;
use utils::get_vector_type;
use uuid::Uuid;

use super::{pb, DatasetIndexInternalExt, IndexParams};
use crate::{dataset::Dataset, index::pb::vector_index_stage::Stage, Error, Result};

pub const LANCE_VECTOR_INDEX: &str = "__lance_vector_index";

/// Parameters of each index stage.
#[derive(Debug, Clone)]
pub enum StageParams {
    Ivf(IvfBuildParams),
    Hnsw(HnswBuildParams),
    PQ(PQBuildParams),
    SQ(SQBuildParams),
}

// The version of the index file.
// `Legacy` is used for only IVF_PQ index, and is the default value.
// The other index types are using `V3`.
#[derive(Debug, Clone, Serialize)]
pub enum IndexFileVersion {
    Legacy,
    V3,
}

impl IndexFileVersion {
    pub fn try_from(version: &str) -> Result<Self> {
        match version.to_lowercase().as_str() {
            "legacy" => Ok(Self::Legacy),
            "v3" => Ok(Self::V3),
            _ => Err(Error::Index {
                message: format!("Invalid index file version: {}", version),
                location: location!(),
            }),
        }
    }
}

/// The parameters to build vector index.
#[derive(Debug, Clone)]
pub struct VectorIndexParams {
    pub stages: Vec<StageParams>,

    /// Vector distance metrics type.
    pub metric_type: MetricType,

    /// The version of the index file.
    pub version: IndexFileVersion,
}

impl VectorIndexParams {
    pub fn version(&mut self, version: IndexFileVersion) -> &mut Self {
        self.version = version;
        self
    }

    pub fn ivf_flat(num_partitions: usize, metric_type: MetricType) -> Self {
        let ivf_params = IvfBuildParams::new(num_partitions);
        let stages = vec![StageParams::Ivf(ivf_params)];
        Self {
            stages,
            metric_type,
            version: IndexFileVersion::V3,
        }
    }

    pub fn with_ivf_flat_params(metric_type: MetricType, ivf: IvfBuildParams) -> Self {
        let stages = vec![StageParams::Ivf(ivf)];
        Self {
            stages,
            metric_type,
            version: IndexFileVersion::V3,
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
            version: IndexFileVersion::V3,
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
            version: IndexFileVersion::V3,
        }
    }

    pub fn with_ivf_sq_params(
        metric_type: MetricType,
        ivf: IvfBuildParams,
        sq: SQBuildParams,
    ) -> Self {
        let stages = vec![StageParams::Ivf(ivf), StageParams::SQ(sq)];
        Self {
            stages,
            metric_type,
            version: IndexFileVersion::V3,
        }
    }

    pub fn ivf_hnsw(
        distance_type: DistanceType,
        ivf: IvfBuildParams,
        hnsw: HnswBuildParams,
    ) -> Self {
        let stages = vec![StageParams::Ivf(ivf), StageParams::Hnsw(hnsw)];
        Self {
            stages,
            metric_type: distance_type,
            version: IndexFileVersion::V3,
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
            version: IndexFileVersion::V3,
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
            version: IndexFileVersion::V3,
        }
    }

    pub fn index_type(&self) -> IndexType {
        let len = self.stages.len();
        match (len, self.stages.get(1), self.stages.last()) {
            (0, _, _) => IndexType::Vector,
            (1, _, Some(StageParams::Ivf(_))) => IndexType::IvfFlat,
            (2, _, Some(StageParams::PQ(_))) => IndexType::IvfPq,
            (2, _, Some(StageParams::SQ(_))) => IndexType::IvfSq,
            (2, _, Some(StageParams::Hnsw(_))) => IndexType::IvfHnswFlat,
            (3, Some(StageParams::Hnsw(_)), Some(StageParams::PQ(_))) => IndexType::IvfHnswPq,
            (3, Some(StageParams::Hnsw(_)), Some(StageParams::SQ(_))) => IndexType::IvfHnswSq,
            _ => IndexType::Vector,
        }
    }
}

impl IndexParams for VectorIndexParams {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index_name(&self) -> &str {
        LANCE_VECTOR_INDEX
    }
}

/// Build a Vector Index
#[instrument(level = "debug", skip(dataset))]
pub(crate) async fn build_vector_index(
    dataset: &Dataset,
    column: &str,
    name: &str,
    uuid: &str,
    params: &VectorIndexParams,
    frag_reuse_index: Option<Arc<FragReuseIndex>>,
) -> Result<()> {
    let stages = &params.stages;

    if stages.is_empty() {
        return Err(Error::Index {
            message: "Build Vector Index: must have at least 1 stage".to_string(),
            location: location!(),
        });
    };

    let StageParams::Ivf(ivf_params) = &stages[0] else {
        return Err(Error::Index {
            message: format!("Build Vector Index: invalid stages: {:?}", stages),
            location: location!(),
        });
    };

    let (vector_type, element_type) = get_vector_type(dataset.schema(), column)?;
    if let DataType::List(_) = vector_type {
        if params.metric_type != DistanceType::Cosine {
            return Err(Error::Index {
                message: "Build Vector Index: multivector type supports only cosine distance"
                    .to_string(),
                location: location!(),
            });
        }
    }

    let num_rows = dataset.count_rows(None).await?;
    let index_type = params.index_type();
    let num_partitions = ivf_params.num_partitions.unwrap_or_else(|| {
        recommended_num_partitions(
            num_rows,
            ivf_params
                .target_partition_size
                .unwrap_or(index_type.target_partition_size()),
        )
    });
    let mut ivf_params = ivf_params.clone();
    ivf_params.num_partitions = Some(num_partitions);

    let temp_dir = tempdir()?;
    let temp_dir_path = Path::from_filesystem_path(temp_dir.path())?;
    let shuffler = IvfShuffler::new(temp_dir_path, num_partitions);
    match index_type {
        IndexType::IvfFlat => match element_type {
            DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                IvfIndexBuilder::<FlatIndex, FlatQuantizer>::new(
                    dataset.clone(),
                    column.to_owned(),
                    dataset.indices_dir().child(uuid),
                    params.metric_type,
                    Box::new(shuffler),
                    Some(ivf_params),
                    Some(()),
                    (),
                    frag_reuse_index,
                )?
                .build()
                .await?;
            }
            DataType::UInt8 => {
                IvfIndexBuilder::<FlatIndex, FlatBinQuantizer>::new(
                    dataset.clone(),
                    column.to_owned(),
                    dataset.indices_dir().child(uuid),
                    params.metric_type,
                    Box::new(shuffler),
                    Some(ivf_params),
                    Some(()),
                    (),
                    frag_reuse_index,
                )?
                .build()
                .await?;
            }
            _ => {
                return Err(Error::Index {
                    message: format!("Build Vector Index: invalid data type: {:?}", element_type),
                    location: location!(),
                });
            }
        },
        IndexType::IvfPq => {
            let len = stages.len();
            let StageParams::PQ(pq_params) = &stages[len - 1] else {
                return Err(Error::Index {
                    message: format!("Build Vector Index: invalid stages: {:?}", stages),
                    location: location!(),
                });
            };

            match params.version {
                IndexFileVersion::Legacy => {
                    build_ivf_pq_index(
                        dataset,
                        column,
                        name,
                        uuid,
                        params.metric_type,
                        &ivf_params,
                        pq_params,
                    )
                    .await?;
                }
                IndexFileVersion::V3 => {
                    IvfIndexBuilder::<FlatIndex, ProductQuantizer>::new(
                        dataset.clone(),
                        column.to_owned(),
                        dataset.indices_dir().child(uuid),
                        params.metric_type,
                        Box::new(shuffler),
                        Some(ivf_params),
                        Some(pq_params.clone()),
                        (),
                        frag_reuse_index,
                    )?
                    .build()
                    .await?;
                }
            }
        }
        IndexType::IvfSq => {
            let StageParams::SQ(sq_params) = &stages[1] else {
                return Err(Error::Index {
                    message: format!("Build Vector Index: invalid stages: {:?}", stages),
                    location: location!(),
                });
            };

            IvfIndexBuilder::<FlatIndex, ScalarQuantizer>::new(
                dataset.clone(),
                column.to_owned(),
                dataset.indices_dir().child(uuid),
                params.metric_type,
                Box::new(shuffler),
                Some(ivf_params),
                Some(sq_params.clone()),
                (),
                frag_reuse_index,
            )?
            .build()
            .await?;
        }
        IndexType::IvfHnswFlat => {
            let StageParams::Hnsw(hnsw_params) = &stages[1] else {
                return Err(Error::Index {
                    message: format!("Build Vector Index: invalid stages: {:?}", stages),
                    location: location!(),
                });
            };
            IvfIndexBuilder::<HNSW, FlatQuantizer>::new(
                dataset.clone(),
                column.to_owned(),
                dataset.indices_dir().child(uuid),
                params.metric_type,
                Box::new(shuffler),
                Some(ivf_params),
                Some(()),
                hnsw_params.clone(),
                frag_reuse_index,
            )?
            .build()
            .await?;
        }
        IndexType::IvfHnswPq => {
            let StageParams::Hnsw(hnsw_params) = &stages[1] else {
                return Err(Error::Index {
                    message: format!("Build Vector Index: invalid stages: {:?}", stages),
                    location: location!(),
                });
            };
            let StageParams::PQ(pq_params) = &stages[2] else {
                return Err(Error::Index {
                    message: format!("Build Vector Index: invalid stages: {:?}", stages),
                    location: location!(),
                });
            };
            IvfIndexBuilder::<HNSW, ProductQuantizer>::new(
                dataset.clone(),
                column.to_owned(),
                dataset.indices_dir().child(uuid),
                params.metric_type,
                Box::new(shuffler),
                Some(ivf_params),
                Some(pq_params.clone()),
                hnsw_params.clone(),
                frag_reuse_index,
            )?
            .build()
            .await?;
        }
        IndexType::IvfHnswSq => {
            let StageParams::Hnsw(hnsw_params) = &stages[1] else {
                return Err(Error::Index {
                    message: format!("Build Vector Index: invalid stages: {:?}", stages),
                    location: location!(),
                });
            };
            let StageParams::SQ(sq_params) = &stages[2] else {
                return Err(Error::Index {
                    message: format!("Build Vector Index: invalid stages: {:?}", stages),
                    location: location!(),
                });
            };
            IvfIndexBuilder::<HNSW, ScalarQuantizer>::new(
                dataset.clone(),
                column.to_owned(),
                dataset.indices_dir().child(uuid),
                params.metric_type,
                Box::new(shuffler),
                Some(ivf_params),
                Some(sq_params.clone()),
                hnsw_params.clone(),
                frag_reuse_index,
            )?
            .build()
            .await?;
        }
        _ => {
            return Err(Error::Index {
                message: format!("Build Vector Index: invalid index type: {:?}", index_type),
                location: location!(),
            });
        }
    };
    Ok(())
}

/// Build an empty vector index without training on data
#[instrument(level = "debug", skip_all)]
pub(crate) async fn build_empty_vector_index(
    _dataset: &Dataset,
    column: &str,
    name: &str,
    _uuid: &str,
    _params: &VectorIndexParams,
) -> Result<()> {
    // For now, return a NotImplementedError to indicate this functionality
    // is still being developed
    Err(Error::NotSupported {
        source: format!(
            "Creating empty vector indices with train=False is not yet implemented. \
            Index '{}' for column '{}' cannot be created without training.",
            name, column
        )
        .into(),
        location: location!(),
    })
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
        .open_vector_index(column, &old_uuid.to_string(), &NoOpMetricsCollector)
        .await?;

    if let Some(ivf_index) = old_index.as_any().downcast_ref::<IVFIndex>() {
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
    } else {
        // it's v3 index
        remap_index_file_v3(
            dataset.as_ref(),
            &new_uuid.to_string(),
            old_index,
            mapping,
            column.to_string(),
        )
        .await?;
    }

    Ok(())
}

/// Open the Vector index on dataset, specified by the `uuid`.
#[instrument(level = "debug", skip(dataset, vec_idx, reader))]
pub(crate) async fn open_vector_index(
    dataset: Arc<Dataset>,
    uuid: &str,
    vec_idx: &lance_index::pb::VectorIndex,
    reader: Arc<dyn Reader>,
    frag_reuse_index: Option<Arc<FragReuseIndex>>,
) -> Result<Arc<dyn VectorIndex>> {
    let metric_type = pb::VectorMetricType::try_from(vec_idx.metric_type)?.into();

    let mut last_stage: Option<Arc<dyn VectorIndex>> = None;

    let frag_reuse_uuid = dataset.frag_reuse_index_uuid();

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
                    uuid,
                    ivf,
                    reader.clone(),
                    last_stage.unwrap(),
                    metric_type,
                    dataset
                        .index_cache
                        .for_index(uuid, frag_reuse_uuid.as_ref()),
                )?));
            }
            Some(Stage::Pq(pq_proto)) => {
                if last_stage.is_some() {
                    return Err(Error::Index {
                        message: format!("Invalid vector index stages: {:?}", vec_idx.stages),
                        location: location!(),
                    });
                };
                let pq = ProductQuantizer::from_proto(pq_proto, metric_type)?;
                last_stage = Some(Arc::new(PQIndex::new(
                    pq,
                    metric_type,
                    frag_reuse_index.clone(),
                )));
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
    frag_reuse_index: Option<Arc<FragReuseIndex>>,
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

    let frag_reuse_uuid = dataset.frag_reuse_index_uuid();
    // Load the index metadata to get the correct index directory
    let index_meta = dataset
        .load_index(uuid)
        .await?
        .ok_or_else(|| Error::Index {
            message: format!("Index with id {} does not exist", uuid),
            location: location!(),
        })?;
    let index_dir = dataset.indice_files_dir(&index_meta)?;

    let index: Arc<dyn VectorIndex> = match index_metadata.index_type.as_str() {
        "IVF_HNSW_PQ" => {
            let aux_path = index_dir.child(uuid).child(INDEX_AUXILIARY_FILE_NAME);
            let aux_reader = dataset.object_store().open(&aux_path).await?;

            let ivf_data = IvfModel::load(&reader).await?;
            let options = HNSWIndexOptions { use_residual: true };
            let hnsw = HNSWIndex::<ProductQuantizer>::try_new(
                reader.object_reader.clone(),
                aux_reader.into(),
                options,
            )
            .await?;
            let pb_ivf = pb::Ivf::try_from(&ivf_data)?;
            let ivf = IvfModel::try_from(pb_ivf)?;

            Arc::new(IVFIndex::try_new(
                uuid,
                ivf,
                reader.object_reader.clone(),
                Arc::new(hnsw),
                distance_type,
                dataset
                    .index_cache
                    .for_index(uuid, frag_reuse_uuid.as_ref()),
            )?)
        }

        "IVF_HNSW_SQ" => {
            let aux_path = index_dir.child(uuid).child(INDEX_AUXILIARY_FILE_NAME);
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
                uuid,
                ivf,
                reader.object_reader.clone(),
                Arc::new(hnsw),
                distance_type,
                dataset
                    .index_cache
                    .for_index(uuid, frag_reuse_uuid.as_ref()),
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
