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
use lance_core::utils::tempfile::TempStdDir;
use lance_file::reader::FileReader;
use lance_index::frag_reuse::FragReuseIndex;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::optimize::OptimizeOptions;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::RQBuildParams;
use lance_index::vector::flat::index::{FlatBinQuantizer, FlatIndex, FlatQuantizer};
use lance_index::vector::hnsw::HNSW;
use lance_index::vector::ivf::builder::recommended_num_partitions;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::quantizer::QuantizationType;
use lance_index::vector::v3::shuffler::IvfShuffler;
use lance_index::vector::v3::subindex::SubIndexType;
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
    VECTOR_INDEX_VERSION,
};
use lance_io::traits::Reader;
use lance_linalg::distance::*;
use lance_table::format::IndexMetadata;
use object_store::path::Path;
use serde::Serialize;
use snafu::location;
use tracing::instrument;
use utils::get_vector_type;
use uuid::Uuid;

use super::{pb, vector_index_details, DatasetIndexInternalExt, IndexParams};
use crate::dataset::transaction::{Operation, Transaction};
use crate::{dataset::Dataset, index::pb::vector_index_stage::Stage, Error, Result};

pub const LANCE_VECTOR_INDEX: &str = "__lance_vector_index";

/// Parameters of each index stage.
#[derive(Debug, Clone)]
pub enum StageParams {
    Ivf(IvfBuildParams),
    Hnsw(HnswBuildParams),
    PQ(PQBuildParams),
    SQ(SQBuildParams),
    RQ(RQBuildParams),
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

    pub fn ivf_rq(num_partitions: usize, num_bits: u8, distance_type: DistanceType) -> Self {
        let ivf = IvfBuildParams::new(num_partitions);
        let rq = RQBuildParams { num_bits };
        let stages = vec![StageParams::Ivf(ivf), StageParams::RQ(rq)];
        Self {
            stages,
            metric_type: distance_type,
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

    pub fn with_ivf_rq_params(
        metric_type: MetricType,
        ivf: IvfBuildParams,
        rq: RQBuildParams,
    ) -> Self {
        let stages = vec![StageParams::Ivf(ivf), StageParams::RQ(rq)];
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
            (2, _, Some(StageParams::RQ(_))) => IndexType::IvfRq,
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

    let temp_dir = TempStdDir::default();
    let temp_dir_path = Path::from_filesystem_path(&temp_dir)?;
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
        IndexType::IvfRq => {
            let StageParams::RQ(rq_params) = &stages[1] else {
                return Err(Error::Index {
                    message: format!("Build Vector Index: invalid stages: {:?}", stages),
                    location: location!(),
                });
            };

            IvfIndexBuilder::<FlatIndex, RabitQuantizer>::new(
                dataset.clone(),
                column.to_owned(),
                dataset.indices_dir().child(uuid),
                params.metric_type,
                Box::new(shuffler),
                Some(ivf_params),
                Some(rq_params.clone()),
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

/// Build a Vector Index incrementally using an existing index's IVF model and quantizer
/// This creates a delta index that shares centroids with the source index
#[instrument(level = "debug", skip(dataset, existing_index, frag_reuse_index))]
pub(crate) async fn build_vector_index_incremental(
    dataset: &Dataset,
    column: &str,
    uuid: &str,
    params: &VectorIndexParams,
    existing_index: Arc<dyn VectorIndex>,
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

    // Extract IVF model and quantizer from existing index
    let ivf_model = existing_index.ivf_model().clone();
    let quantizer = existing_index.quantizer();

    // Ensure the number of partitions matches
    let expected_partitions = ivf_params
        .num_partitions
        .unwrap_or(ivf_model.num_partitions());
    if ivf_model.num_partitions() != expected_partitions {
        return Err(Error::Index {
            message: format!(
                "Number of partitions mismatch: existing index has {} partitions, but params specify {}",
                ivf_model.num_partitions(),
                expected_partitions
            ),
            location: location!(),
        });
    }

    let temp_dir = TempStdDir::default();
    let temp_dir_path = Path::from_filesystem_path(&temp_dir)?;
    let shuffler = Box::new(IvfShuffler::new(temp_dir_path, ivf_model.num_partitions()));

    let index_dir = dataset.indices_dir().child(uuid);

    // Determine the index type and build incrementally
    let (sub_index_type, quantization_type) = existing_index.sub_index_type();

    match (sub_index_type, quantization_type) {
        // IVF_FLAT
        (SubIndexType::Flat, QuantizationType::Flat) => match element_type {
            DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                IvfIndexBuilder::<FlatIndex, FlatQuantizer>::new_incremental(
                    dataset.clone(),
                    column.to_owned(),
                    index_dir,
                    params.metric_type,
                    shuffler,
                    (),
                    frag_reuse_index,
                    OptimizeOptions::new(),
                )?
                .with_ivf(ivf_model)
                .with_quantizer(quantizer.try_into()?)
                .build()
                .await?;
            }
            DataType::UInt8 => {
                IvfIndexBuilder::<FlatIndex, FlatBinQuantizer>::new_incremental(
                    dataset.clone(),
                    column.to_owned(),
                    index_dir,
                    params.metric_type,
                    shuffler,
                    (),
                    frag_reuse_index,
                    OptimizeOptions::new(),
                )?
                .with_ivf(ivf_model)
                .with_quantizer(quantizer.try_into()?)
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
        // IVF_PQ
        (SubIndexType::Flat, QuantizationType::Product) => {
            IvfIndexBuilder::<FlatIndex, ProductQuantizer>::new_incremental(
                dataset.clone(),
                column.to_owned(),
                index_dir,
                params.metric_type,
                shuffler,
                (),
                frag_reuse_index,
                OptimizeOptions::new(),
            )?
            .with_ivf(ivf_model)
            .with_quantizer(quantizer.try_into()?)
            .build()
            .await?;
        }
        // IVF_SQ
        (SubIndexType::Flat, QuantizationType::Scalar) => {
            IvfIndexBuilder::<FlatIndex, ScalarQuantizer>::new_incremental(
                dataset.clone(),
                column.to_owned(),
                index_dir,
                params.metric_type,
                shuffler,
                (),
                frag_reuse_index,
                OptimizeOptions::new(),
            )?
            .with_ivf(ivf_model)
            .with_quantizer(quantizer.try_into()?)
            .build()
            .await?;
        }
        // IVF_RQ
        (SubIndexType::Flat, QuantizationType::Rabit) => {
            IvfIndexBuilder::<FlatIndex, RabitQuantizer>::new_incremental(
                dataset.clone(),
                column.to_owned(),
                index_dir,
                params.metric_type,
                shuffler,
                (),
                frag_reuse_index,
                OptimizeOptions::new(),
            )?
            .with_ivf(ivf_model)
            .with_quantizer(quantizer.try_into()?)
            .build()
            .await?;
        }
        // IVF_HNSW variants
        (SubIndexType::Hnsw, quantization_type) => {
            let StageParams::Hnsw(hnsw_params) = &stages[1] else {
                return Err(Error::Index {
                    message: format!(
                        "Build Vector Index: HNSW index missing HNSW params in stages: {:?}",
                        stages
                    ),
                    location: location!(),
                });
            };

            match quantization_type {
                QuantizationType::Flat => {
                    IvfIndexBuilder::<HNSW, FlatQuantizer>::new_incremental(
                        dataset.clone(),
                        column.to_owned(),
                        index_dir,
                        params.metric_type,
                        shuffler,
                        hnsw_params.clone(),
                        frag_reuse_index,
                        OptimizeOptions::new(),
                    )?
                    .with_ivf(ivf_model)
                    .with_quantizer(quantizer.try_into()?)
                    .build()
                    .await?;
                }
                QuantizationType::Product => {
                    IvfIndexBuilder::<HNSW, ProductQuantizer>::new_incremental(
                        dataset.clone(),
                        column.to_owned(),
                        index_dir,
                        params.metric_type,
                        shuffler,
                        hnsw_params.clone(),
                        frag_reuse_index,
                        OptimizeOptions::new(),
                    )?
                    .with_ivf(ivf_model)
                    .with_quantizer(quantizer.try_into()?)
                    .build()
                    .await?;
                }
                QuantizationType::Scalar => {
                    IvfIndexBuilder::<HNSW, ScalarQuantizer>::new_incremental(
                        dataset.clone(),
                        column.to_owned(),
                        index_dir,
                        params.metric_type,
                        shuffler,
                        hnsw_params.clone(),
                        frag_reuse_index,
                        OptimizeOptions::new(),
                    )?
                    .with_ivf(ivf_model)
                    .with_quantizer(quantizer.try_into()?)
                    .build()
                    .await?;
                }
                QuantizationType::Rabit => {
                    return Err(Error::Index {
                        message: "Rabit quantization is not supported for HNSW index".to_string(),
                        location: location!(),
                    });
                }
            }
        }
    }

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

    let frag_reuse_uuid = dataset.frag_reuse_index_uuid().await;

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

    let frag_reuse_uuid = dataset.frag_reuse_index_uuid().await;
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

/// Initialize a vector index from a source dataset
/// This will reuse the centroids from the source dataset,
/// making the new indices basically a "delta index" of the source dataset,
/// until the new dataset fully retains the index.
pub async fn initialize_vector_index(
    target_dataset: &mut Dataset,
    source_dataset: &Dataset,
    source_index: &IndexMetadata,
    field_names: &[&str],
) -> Result<()> {
    if field_names.is_empty() || field_names.len() > 1 {
        return Err(Error::Index {
            message: format!("Unsupported fields for vector index: {:?}", field_names),
            location: location!(),
        });
    }

    // Vector indices currently support only single fields, use the first one
    let column_name = field_names[0];

    let source_vector_index = source_dataset
        .open_vector_index(
            column_name,
            &source_index.uuid.to_string(),
            &NoOpMetricsCollector,
        )
        .await?;

    let metric_type = source_vector_index.metric_type();
    let ivf_model = source_vector_index.ivf_model();
    let quantizer = source_vector_index.quantizer();
    let (sub_index_type, quantization_type) = source_vector_index.sub_index_type();
    let ivf_params = derive_ivf_params(ivf_model);

    let params = match (sub_index_type, quantization_type) {
        (SubIndexType::Flat, QuantizationType::Flat) => {
            VectorIndexParams::with_ivf_flat_params(metric_type, ivf_params)
        }
        (SubIndexType::Flat, QuantizationType::Product) => {
            let pq_quantizer: ProductQuantizer = quantizer.try_into()?;
            let pq_params = derive_pq_params(&pq_quantizer);
            VectorIndexParams::with_ivf_pq_params(metric_type, ivf_params, pq_params)
        }
        (SubIndexType::Flat, QuantizationType::Scalar) => {
            let sq_quantizer: ScalarQuantizer = quantizer.try_into()?;
            let sq_params = derive_sq_params(&sq_quantizer);
            VectorIndexParams::with_ivf_sq_params(metric_type, ivf_params, sq_params)
        }
        (SubIndexType::Flat, QuantizationType::Rabit) => {
            let rabit_quantizer: RabitQuantizer = quantizer.try_into()?;
            let rabit_params = derive_rabit_params(&rabit_quantizer);
            VectorIndexParams::with_ivf_rq_params(metric_type, ivf_params, rabit_params)
        }
        (SubIndexType::Hnsw, quantization_type) => {
            let hnsw_params = derive_hnsw_params(source_vector_index.as_ref());
            match quantization_type {
                QuantizationType::Flat => {
                    VectorIndexParams::ivf_hnsw(metric_type, ivf_params, hnsw_params)
                }
                QuantizationType::Product => {
                    let pq_quantizer: ProductQuantizer = quantizer.try_into()?;
                    let pq_params = derive_pq_params(&pq_quantizer);
                    VectorIndexParams::with_ivf_hnsw_pq_params(
                        metric_type,
                        ivf_params,
                        hnsw_params,
                        pq_params,
                    )
                }
                QuantizationType::Scalar => {
                    let sq_quantizer: ScalarQuantizer = quantizer.try_into()?;
                    let sq_params = derive_sq_params(&sq_quantizer);
                    VectorIndexParams::with_ivf_hnsw_sq_params(
                        metric_type,
                        ivf_params,
                        hnsw_params,
                        sq_params,
                    )
                }
                QuantizationType::Rabit => {
                    return Err(Error::Index {
                        message: "Rabit quantization is not supported for HNSW index".to_string(),
                        location: location!(),
                    });
                }
            }
        }
    };

    let new_uuid = Uuid::new_v4();
    let frag_reuse_index = target_dataset
        .open_frag_reuse_index(&NoOpMetricsCollector)
        .await?;

    build_vector_index_incremental(
        target_dataset,
        column_name,
        &new_uuid.to_string(),
        &params,
        source_vector_index,
        frag_reuse_index,
    )
    .await?;

    let field = target_dataset
        .schema()
        .field(column_name)
        .ok_or_else(|| Error::Index {
            message: format!("Column '{}' not found in target dataset", column_name),
            location: location!(),
        })?;

    let fragment_bitmap = if target_dataset.get_fragments().is_empty() {
        Some(roaring::RoaringBitmap::new())
    } else {
        Some(
            target_dataset
                .get_fragments()
                .iter()
                .map(|f| f.id() as u32)
                .collect(),
        )
    };

    let new_idx = IndexMetadata {
        uuid: new_uuid,
        name: source_index.name.clone(),
        fields: vec![field.id],
        dataset_version: target_dataset.manifest.version,
        fragment_bitmap,
        index_details: Some(Arc::new(vector_index_details())),
        index_version: VECTOR_INDEX_VERSION as i32,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
    };

    let transaction = Transaction::new(
        target_dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![new_idx],
            removed_indices: vec![],
        },
        None,
    );

    target_dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await?;

    Ok(())
}

/// Create IVF build parameters for delta index creation from an existing IVF model
/// TODO: support deriving all the original parameters
fn derive_ivf_params(ivf_model: &IvfModel) -> IvfBuildParams {
    IvfBuildParams {
        num_partitions: Some(ivf_model.num_partitions()),
        target_partition_size: None,
        max_iters: 50, // Default
        centroids: ivf_model.centroids.clone().map(Arc::new),
        #[allow(deprecated)]
        retrain: false, // Don't retrain since we have centroids
        sample_rate: 256, // Default
        precomputed_partitions_file: None,
        precomputed_shuffle_buffers: None,
        shuffle_partition_batches: 1024 * 10, // Default
        shuffle_partition_concurrency: 2,     // Default
        storage_options: None,
    }
}

/// Create PQ build parameters from a ProductQuantizer
/// TODO: support consistently deriving all the original parameters
fn derive_pq_params(pq_quantizer: &ProductQuantizer) -> PQBuildParams {
    PQBuildParams {
        num_sub_vectors: pq_quantizer.num_sub_vectors,
        num_bits: pq_quantizer.num_bits as usize,
        max_iters: 50,   // Default
        kmeans_redos: 1, // Default
        codebook: Some(Arc::new(pq_quantizer.codebook.clone())),
        sample_rate: 256, // Default
    }
}

/// Create SQ build parameters from a ScalarQuantizer
/// TODO: support consistently deriving all the original parameters
fn derive_sq_params(sq_quantizer: &ScalarQuantizer) -> SQBuildParams {
    SQBuildParams {
        num_bits: sq_quantizer.num_bits(),
        sample_rate: 256, // Default
    }
}

/// Create Rabit build parameters from a RabitQuantizer
/// TODO: support consistently deriving all the original parameters
fn derive_rabit_params(rabit_quantizer: &RabitQuantizer) -> RQBuildParams {
    RQBuildParams {
        num_bits: rabit_quantizer.num_bits(),
    }
}

/// Extract HNSW build parameters from the source vector index statistics.
/// Returns default parameters if extraction fails.
/// TODO: support consistently deriving all the original parameters
fn derive_hnsw_params(source_index: &dyn VectorIndex) -> HnswBuildParams {
    let default_params = HnswBuildParams {
        max_level: 4,
        m: 20,
        ef_construction: 100,
        prefetch_distance: None,
    };

    let Ok(stats) = source_index.statistics() else {
        return default_params;
    };

    let Some(sub_index) = stats.get("sub_index") else {
        return default_params;
    };

    // Extract HNSW parameters from sub_index.params
    if let Some(params) = sub_index.get("params") {
        let max_level = params
            .get("max_level")
            .and_then(|v| v.as_u64())
            .map(|v| v as u16)
            .unwrap_or(4);
        let m = params
            .get("m")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(20);
        let ef_construction = params
            .get("ef_construction")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(100);

        return HnswBuildParams {
            max_level,
            m,
            ef_construction,
            prefetch_distance: None,
        };
    }

    default_params
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use arrow_array::types::{Float32Type, Int32Type};
    use arrow_array::Array;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::{array, BatchCount, RowCount};
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::DatasetIndexExt;
    use lance_linalg::distance::MetricType;

    #[tokio::test]
    async fn test_initialize_vector_index_ivf_pq() {
        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/source", test_dir.as_str());
        let target_uri = format!("{}/target", test_dir.as_str());

        // Create source dataset with vector column (need at least 256 rows for PQ training)
        let source_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create IVF_PQ index on source
        let params = VectorIndexParams::ivf_pq(10, 8, 16, MetricType::L2, 50);
        source_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_ivf_pq".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Reload to get updated metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();
        let source_indices = source_dataset.load_indices().await.unwrap();
        let source_index = source_indices
            .iter()
            .find(|idx| idx.name == "vector_ivf_pq")
            .unwrap();

        // Create target dataset with same schema
        let target_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize IVF_PQ index on target
        initialize_vector_index(
            &mut target_dataset,
            &source_dataset,
            source_index,
            &["vector"],
        )
        .await
        .unwrap();

        // Verify index was created
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Target should have 1 index");
        assert_eq!(
            target_indices[0].name, "vector_ivf_pq",
            "Index name should match"
        );
        assert_eq!(
            target_indices[0].fields,
            vec![1],
            "Index should be on field 1 (vector)"
        );

        // Verify the index type and parameters match
        let target_vector_index = target_dataset
            .open_vector_index(
                "vector",
                &target_indices[0].uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let stats = target_vector_index.statistics().unwrap();

        // Check basic index type
        assert_eq!(
            stats.get("index_type").and_then(|v| v.as_str()),
            Some("IVF_PQ"),
            "Index type should be IVF_PQ"
        );

        // Check metric type
        assert_eq!(
            stats.get("metric_type").and_then(|v| v.as_str()),
            Some("l2"),
            "Metric type should be L2"
        );

        // Check number of partitions
        assert_eq!(
            stats.get("num_partitions").and_then(|v| v.as_u64()),
            Some(10),
            "Should have 10 partitions"
        );

        // Verify centroids are shared between source and target indices
        let source_vector_index = source_dataset
            .open_vector_index(
                "vector",
                &source_index.uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();

        // Get IVF models from both indices to compare centroids
        let source_ivf_model = source_vector_index.ivf_model();
        let target_ivf_model = target_vector_index.ivf_model();

        // Verify they have the same number of partitions
        assert_eq!(
            source_ivf_model.num_partitions(),
            target_ivf_model.num_partitions(),
            "Source and target should have same number of partitions"
        );

        // Verify the centroids are exactly the same (key verification for delta indices)
        if let (Some(source_centroids), Some(target_centroids)) =
            (&source_ivf_model.centroids, &target_ivf_model.centroids)
        {
            assert_eq!(
                source_centroids.len(),
                target_centroids.len(),
                "Centroids arrays should have same length"
            );

            // Compare actual centroid values
            // Since value() returns Arc<dyn Array>, we need to compare the data directly
            for i in 0..source_centroids.len() {
                let source_centroid = source_centroids.value(i);
                let target_centroid = target_centroids.value(i);

                // Convert to the same type for comparison
                let source_data = source_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");
                let target_data = target_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");

                assert_eq!(
                    source_data.values(),
                    target_data.values(),
                    "Centroid {} values should be identical between source and target",
                    i
                );
            }
        } else {
            panic!("Both source and target should have centroids");
        }

        // Verify IVF parameters are correctly derived
        let source_ivf_params = derive_ivf_params(source_ivf_model);
        let target_ivf_params = derive_ivf_params(target_ivf_model);
        assert_eq!(
            source_ivf_params.num_partitions, target_ivf_params.num_partitions,
            "IVF num_partitions should match"
        );
        assert_eq!(
            target_ivf_params.num_partitions,
            Some(10),
            "Should have 10 partitions as configured"
        );

        // Verify PQ parameters are correctly derived
        let source_quantizer = source_vector_index.quantizer();
        let target_quantizer = target_vector_index.quantizer();
        let source_pq: ProductQuantizer = source_quantizer.try_into().unwrap();
        let target_pq: ProductQuantizer = target_quantizer.try_into().unwrap();

        let source_pq_params = derive_pq_params(&source_pq);
        let target_pq_params = derive_pq_params(&target_pq);

        assert_eq!(
            source_pq_params.num_sub_vectors, target_pq_params.num_sub_vectors,
            "PQ num_sub_vectors should match"
        );
        assert_eq!(
            source_pq_params.num_bits, target_pq_params.num_bits,
            "PQ num_bits should match"
        );
        assert_eq!(
            target_pq_params.num_sub_vectors, 16,
            "PQ should have 16 sub vectors"
        );
        assert_eq!(target_pq_params.num_bits, 8, "PQ should use 8 bits");

        // Verify the index is functional by performing a search
        let query_vector = lance_datagen::gen_batch()
            .anon_col(array::rand_vec::<Float32Type>(32.into()))
            .into_batch_rows(RowCount::from(1))
            .unwrap()
            .column(0)
            .clone();
        let query_vector = query_vector
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .unwrap();
        let results = target_dataset
            .scan()
            .nearest("vector", &query_vector.value(0), 10)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 10, "Should return 10 nearest neighbors");
    }

    #[tokio::test]
    async fn test_initialize_vector_index_ivf_flat() {
        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/source", test_dir.as_str());
        let target_uri = format!("{}/target", test_dir.as_str());

        // Create source dataset with vector column
        let source_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(64.into()))
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create IVF_FLAT index on source
        let params = VectorIndexParams::ivf_flat(8, MetricType::Cosine);
        source_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_ivf_flat".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Reload to get updated metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();
        let source_indices = source_dataset.load_indices().await.unwrap();
        let source_index = source_indices
            .iter()
            .find(|idx| idx.name == "vector_ivf_flat")
            .unwrap();

        // Create target dataset with same schema
        let target_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(64.into()))
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize IVF_FLAT index on target
        initialize_vector_index(
            &mut target_dataset,
            &source_dataset,
            source_index,
            &["vector"],
        )
        .await
        .unwrap();

        // Verify index was created
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Target should have 1 index");
        assert_eq!(
            target_indices[0].name, "vector_ivf_flat",
            "Index name should match"
        );
        assert_eq!(
            target_indices[0].fields,
            vec![1],
            "Index should be on field 1 (vector)"
        );

        // Verify the index type and parameters match
        let target_vector_index = target_dataset
            .open_vector_index(
                "vector",
                &target_indices[0].uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let stats = target_vector_index.statistics().unwrap();

        // Check basic index type
        assert_eq!(
            stats.get("index_type").and_then(|v| v.as_str()),
            Some("IVF_FLAT"),
            "Index type should be IVF_FLAT"
        );

        // Check metric type (Cosine might be stored as "cosine")
        let metric = stats
            .get("metric_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            metric == "cosine" || metric == "Cosine",
            "Metric type should be Cosine, got: {}",
            metric
        );

        // Check number of partitions
        assert_eq!(
            stats.get("num_partitions").and_then(|v| v.as_u64()),
            Some(8),
            "Should have 8 partitions"
        );

        // Verify centroids are shared between source and target indices
        let source_vector_index = source_dataset
            .open_vector_index(
                "vector",
                &source_index.uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();

        // Get IVF models from both indices to compare centroids
        let source_ivf_model = source_vector_index.ivf_model();
        let target_ivf_model = target_vector_index.ivf_model();

        // Verify they have the same number of partitions
        assert_eq!(
            source_ivf_model.num_partitions(),
            target_ivf_model.num_partitions(),
            "Source and target should have same number of partitions"
        );

        // Verify the centroids are exactly the same (key verification for delta indices)
        if let (Some(source_centroids), Some(target_centroids)) =
            (&source_ivf_model.centroids, &target_ivf_model.centroids)
        {
            assert_eq!(
                source_centroids.len(),
                target_centroids.len(),
                "Centroids arrays should have same length"
            );

            // Compare actual centroid values
            // Since value() returns Arc<dyn Array>, we need to compare the data directly
            for i in 0..source_centroids.len() {
                let source_centroid = source_centroids.value(i);
                let target_centroid = target_centroids.value(i);

                // Convert to the same type for comparison
                let source_data = source_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");
                let target_data = target_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");

                assert_eq!(
                    source_data.values(),
                    target_data.values(),
                    "Centroid {} values should be identical between source and target",
                    i
                );
            }
        } else {
            panic!("Both source and target should have centroids");
        }

        // Verify IVF parameters are correctly derived
        let source_ivf_params = derive_ivf_params(source_ivf_model);
        let target_ivf_params = derive_ivf_params(target_ivf_model);
        assert_eq!(
            source_ivf_params.num_partitions, target_ivf_params.num_partitions,
            "IVF num_partitions should match"
        );
        assert_eq!(
            target_ivf_params.num_partitions,
            Some(8),
            "Should have 8 partitions as configured"
        );

        // Verify the index is functional
        let query_vector = lance_datagen::gen_batch()
            .anon_col(array::rand_vec::<Float32Type>(64.into()))
            .into_batch_rows(RowCount::from(1))
            .unwrap()
            .column(0)
            .clone();
        let query_vector = query_vector
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .unwrap();
        let results = target_dataset
            .scan()
            .nearest("vector", &query_vector.value(0), 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 5, "Should return 5 nearest neighbors");
    }

    #[tokio::test]
    async fn test_initialize_vector_index_empty_dataset() {
        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/source", test_dir.as_str());
        let target_uri = format!("{}/target", test_dir.as_str());

        // Create source dataset with vector column
        let source_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create IVF_PQ index on source
        let params = VectorIndexParams::ivf_pq(10, 8, 16, MetricType::L2, 50);
        source_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_ivf_pq".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Reload to get updated metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();
        let source_indices = source_dataset.load_indices().await.unwrap();
        let source_index = source_indices
            .iter()
            .find(|idx| idx.name == "vector_ivf_pq")
            .unwrap();

        // Create EMPTY target dataset with same schema
        let empty_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(0), BatchCount::from(1)); // Empty dataset
        let mut target_dataset = Dataset::write(empty_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize IVF_PQ index on empty target
        initialize_vector_index(
            &mut target_dataset,
            &source_dataset,
            source_index,
            &["vector"],
        )
        .await
        .unwrap();

        // Verify index was created even though dataset is empty
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Empty target should have 1 index");
        assert_eq!(
            target_indices[0].name, "vector_ivf_pq",
            "Index name should match"
        );

        // Open both indices to compare centroids
        let source_vector_index = source_dataset
            .open_vector_index(
                "vector",
                &source_index.uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();

        let target_vector_index = target_dataset
            .open_vector_index(
                "vector",
                &target_indices[0].uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();

        // Get IVF models from both indices
        let source_ivf_model = source_vector_index.ivf_model();
        let target_ivf_model = target_vector_index.ivf_model();

        // Verify they have the same number of partitions
        assert_eq!(
            source_ivf_model.num_partitions(),
            target_ivf_model.num_partitions(),
            "Empty dataset should still have same number of partitions as source"
        );

        // Verify the centroids are exactly the same even for empty dataset
        if let (Some(source_centroids), Some(target_centroids)) =
            (&source_ivf_model.centroids, &target_ivf_model.centroids)
        {
            assert_eq!(
                source_centroids.len(),
                target_centroids.len(),
                "Centroids arrays should have same length even for empty dataset"
            );

            // Compare actual centroid values
            for i in 0..source_centroids.len() {
                let source_centroid = source_centroids.value(i);
                let target_centroid = target_centroids.value(i);

                let source_data = source_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");
                let target_data = target_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");

                assert_eq!(
                    source_data.values(),
                    target_data.values(),
                    "Empty dataset should have identical centroids from source"
                );
            }
        } else {
            panic!("Both source and empty target should have centroids");
        }

        // Now add data to the target dataset
        let new_data_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        target_dataset.append(new_data_reader, None).await.unwrap();

        // Run optimize_indices to index the newly added data and merge indices
        // We set num_indices_to_merge to a high value to force merging all indices into one
        use lance_index::optimize::OptimizeOptions;
        target_dataset
            .optimize_indices(&OptimizeOptions::merge(10))
            .await
            .unwrap();

        // Reload dataset to get updated index metadata
        let target_dataset = Dataset::open(&target_uri).await.unwrap();

        // Verify we have only one merged index after optimization
        let index_stats = target_dataset
            .index_statistics("vector_ivf_pq")
            .await
            .unwrap();
        let stats_json: serde_json::Value = serde_json::from_str(&index_stats).unwrap();
        assert_eq!(
            stats_json["num_indices"], 1,
            "Should have only 1 merged index after optimize with high num_indices_to_merge"
        );
        assert_eq!(
            stats_json["num_indexed_fragments"], 1,
            "Should have indexed the appended fragment (empty dataset has no fragments)"
        );
        assert_eq!(
            stats_json["num_unindexed_fragments"], 0,
            "All fragments should be indexed after optimization"
        );

        // The index should now work with the new data
        let query_vector = lance_datagen::gen_batch()
            .anon_col(array::rand_vec::<Float32Type>(32.into()))
            .into_batch_rows(RowCount::from(1))
            .unwrap()
            .column(0)
            .clone();
        let query_vector = query_vector
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .unwrap();

        let results = target_dataset
            .scan()
            .nearest("vector", &query_vector.value(0), 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            results.num_rows(),
            5,
            "Should return 5 nearest neighbors after optimizing index"
        );

        // Verify that the optimized index still shares centroids with the source
        let target_indices = target_dataset.load_indices().await.unwrap();
        let target_vector_index = target_dataset
            .open_vector_index(
                "vector",
                &target_indices[0].uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();

        let target_ivf_model = target_vector_index.ivf_model();

        // Verify centroids are still the same after optimization
        if let (Some(source_centroids), Some(target_centroids)) =
            (&source_ivf_model.centroids, &target_ivf_model.centroids)
        {
            for i in 0..source_centroids.len() {
                let source_centroid = source_centroids.value(i);
                let target_centroid = target_centroids.value(i);

                let source_data = source_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");
                let target_data = target_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");

                assert_eq!(
                    source_data.values(),
                    target_data.values(),
                    "Centroids should remain identical after optimize_indices"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_initialize_vector_index_ivf_sq() {
        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/source", test_dir.as_str());
        let target_uri = format!("{}/target", test_dir.as_str());

        // Create source dataset with vector column
        let source_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(400), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create IVF_SQ index on source
        use lance_index::vector::ivf::IvfBuildParams;
        use lance_index::vector::sq::builder::SQBuildParams;
        let ivf_params = IvfBuildParams::new(6);
        let sq_params = SQBuildParams::default();
        let params = VectorIndexParams::with_ivf_sq_params(MetricType::Dot, ivf_params, sq_params);
        source_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_ivf_sq".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Reload to get updated metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();
        let source_indices = source_dataset.load_indices().await.unwrap();
        let source_index = source_indices
            .iter()
            .find(|idx| idx.name == "vector_ivf_sq")
            .unwrap();

        // Create target dataset with same schema
        let target_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(400), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize IVF_SQ index on target
        initialize_vector_index(
            &mut target_dataset,
            &source_dataset,
            source_index,
            &["vector"],
        )
        .await
        .unwrap();

        // Verify index was created
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Target should have 1 index");
        assert_eq!(
            target_indices[0].name, "vector_ivf_sq",
            "Index name should match"
        );
        assert_eq!(
            target_indices[0].fields,
            vec![1],
            "Index should be on field 1 (vector)"
        );

        // Verify the index type and parameters match
        let target_vector_index = target_dataset
            .open_vector_index(
                "vector",
                &target_indices[0].uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let stats = target_vector_index.statistics().unwrap();

        // Check basic index type
        assert_eq!(
            stats.get("index_type").and_then(|v| v.as_str()),
            Some("IVF_SQ"),
            "Index type should be IVF_SQ"
        );

        // Check metric type (Dot might be stored as "dot")
        let metric = stats
            .get("metric_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            metric == "dot" || metric == "Dot",
            "Metric type should be Dot, got: {}",
            metric
        );

        // Check number of partitions
        assert_eq!(
            stats.get("num_partitions").and_then(|v| v.as_u64()),
            Some(6),
            "Should have 6 partitions"
        );

        // Verify centroids are shared between source and target indices
        let source_vector_index = source_dataset
            .open_vector_index(
                "vector",
                &source_index.uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();

        // Get IVF models from both indices to compare centroids
        let source_ivf_model = source_vector_index.ivf_model();
        let target_ivf_model = target_vector_index.ivf_model();

        // Verify they have the same number of partitions
        assert_eq!(
            source_ivf_model.num_partitions(),
            target_ivf_model.num_partitions(),
            "Source and target should have same number of partitions"
        );

        // Verify the centroids are exactly the same (key verification for delta indices)
        if let (Some(source_centroids), Some(target_centroids)) =
            (&source_ivf_model.centroids, &target_ivf_model.centroids)
        {
            assert_eq!(
                source_centroids.len(),
                target_centroids.len(),
                "Centroids arrays should have same length"
            );

            // Compare actual centroid values
            // Since value() returns Arc<dyn Array>, we need to compare the data directly
            for i in 0..source_centroids.len() {
                let source_centroid = source_centroids.value(i);
                let target_centroid = target_centroids.value(i);

                // Convert to the same type for comparison
                let source_data = source_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");
                let target_data = target_centroid
                    .as_any()
                    .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                    .expect("Centroid should be Float32Array");

                assert_eq!(
                    source_data.values(),
                    target_data.values(),
                    "Centroid {} values should be identical between source and target",
                    i
                );
            }
        } else {
            panic!("Both source and target should have centroids");
        }

        // Verify IVF parameters are correctly derived
        let source_ivf_params = derive_ivf_params(source_ivf_model);
        let target_ivf_params = derive_ivf_params(target_ivf_model);
        assert_eq!(
            source_ivf_params.num_partitions, target_ivf_params.num_partitions,
            "IVF num_partitions should match"
        );
        assert_eq!(
            target_ivf_params.num_partitions,
            Some(6),
            "Should have 6 partitions as configured"
        );

        // Verify SQ parameters are correctly derived
        let source_quantizer = source_vector_index.quantizer();
        let target_quantizer = target_vector_index.quantizer();
        let source_sq: ScalarQuantizer = source_quantizer.try_into().unwrap();
        let target_sq: ScalarQuantizer = target_quantizer.try_into().unwrap();

        let source_sq_params = derive_sq_params(&source_sq);
        let target_sq_params = derive_sq_params(&target_sq);

        assert_eq!(
            source_sq_params.num_bits, target_sq_params.num_bits,
            "SQ num_bits should match"
        );

        // Verify the index is functional
        let query_vector = lance_datagen::gen_batch()
            .anon_col(array::rand_vec::<Float32Type>(32.into()))
            .into_batch_rows(RowCount::from(1))
            .unwrap()
            .column(0)
            .clone();
        let query_vector = query_vector
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .unwrap();
        let results = target_dataset
            .scan()
            .nearest("vector", &query_vector.value(0), 15)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 15, "Should return 15 nearest neighbors");
    }

    #[tokio::test]
    async fn test_initialize_vector_index_ivf_hnsw_pq() {
        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/source", test_dir.as_str());
        let target_uri = format!("{}/target", test_dir.as_str());

        // Create source dataset with vector column (need at least 256 rows for PQ training)
        let source_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(400), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create IVF_HNSW_PQ index on source with custom HNSW parameters
        let ivf_params = IvfBuildParams {
            num_partitions: Some(8),
            ..Default::default()
        };
        let hnsw_params = HnswBuildParams {
            max_level: 6,
            m: 24,
            ef_construction: 120,
            prefetch_distance: None,
        };
        let pq_params = PQBuildParams {
            num_sub_vectors: 8,
            num_bits: 8,
            ..Default::default()
        };
        let params = VectorIndexParams::with_ivf_hnsw_pq_params(
            MetricType::L2,
            ivf_params,
            hnsw_params,
            pq_params,
        );

        source_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_ivf_hnsw_pq".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Reload to get updated metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();
        let source_indices = source_dataset.load_indices().await.unwrap();
        let source_index = source_indices
            .iter()
            .find(|idx| idx.name == "vector_ivf_hnsw_pq")
            .unwrap();

        // Create target dataset with same schema
        let target_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize IVF_HNSW_PQ index on target
        initialize_vector_index(
            &mut target_dataset,
            &source_dataset,
            source_index,
            &["vector"],
        )
        .await
        .unwrap();

        // Verify index was created
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Target should have 1 index");
        assert_eq!(
            target_indices[0].name, "vector_ivf_hnsw_pq",
            "Index name should match"
        );

        // Verify the index type and parameters match
        let target_vector_index = target_dataset
            .open_vector_index(
                "vector",
                &target_indices[0].uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let stats = target_vector_index.statistics().unwrap();

        // Check basic index type
        assert_eq!(
            stats.get("index_type").and_then(|v| v.as_str()),
            Some("IVF_HNSW_PQ"),
            "Index type should be IVF_HNSW_PQ"
        );

        // Check metric type
        assert_eq!(
            stats.get("metric_type").and_then(|v| v.as_str()),
            Some("l2"),
            "Metric type should be L2"
        );

        // Check number of partitions
        assert_eq!(
            stats.get("num_partitions").and_then(|v| v.as_u64()),
            Some(8),
            "Should have 8 partitions"
        );

        // Verify centroids are shared between source and target indices
        let source_vector_index = source_dataset
            .open_vector_index(
                "vector",
                &source_index.uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();

        // Get IVF models from both indices to compare centroids
        let source_ivf_model = source_vector_index.ivf_model();
        let target_ivf_model = target_vector_index.ivf_model();

        // Verify they have the same number of partitions
        assert_eq!(
            source_ivf_model.num_partitions(),
            target_ivf_model.num_partitions(),
            "Source and target should have same number of partitions"
        );

        // Verify the centroids are exactly the same (key verification for delta indices)
        if let (Some(source_centroids), Some(target_centroids)) =
            (&source_ivf_model.centroids, &target_ivf_model.centroids)
        {
            assert_eq!(
                source_centroids.len(),
                target_centroids.len(),
                "Centroids arrays should have same length"
            );

            // Compare first centroid to verify they're identical
            let source_centroid = source_centroids.value(0);
            let target_centroid = target_centroids.value(0);

            let source_data = source_centroid
                .as_any()
                .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                .expect("Centroid should be Float32Array");
            let target_data = target_centroid
                .as_any()
                .downcast_ref::<arrow_array::PrimitiveArray<arrow_array::types::Float32Type>>()
                .expect("Centroid should be Float32Array");

            assert_eq!(
                source_data.values(),
                target_data.values(),
                "Centroid values should be identical between source and target"
            );
        } else {
            panic!("Both source and target should have centroids");
        }

        // Check sub_index contains HNSW and PQ information
        let sub_index = stats
            .get("sub_index")
            .and_then(|v| v.as_object())
            .expect("IVF_HNSW_PQ index should have sub_index");

        // Verify PQ parameters
        assert_eq!(
            sub_index.get("nbits").and_then(|v| v.as_u64()),
            Some(8),
            "PQ should use 8 bits"
        );
        assert_eq!(
            sub_index.get("num_sub_vectors").and_then(|v| v.as_u64()),
            Some(8),
            "PQ should have 8 sub vectors"
        );

        // Verify IVF parameters are correctly derived
        let source_ivf_params = derive_ivf_params(source_ivf_model);
        let target_ivf_params = derive_ivf_params(target_ivf_model);
        assert_eq!(
            source_ivf_params.num_partitions, target_ivf_params.num_partitions,
            "IVF num_partitions should match"
        );
        assert_eq!(
            target_ivf_params.num_partitions,
            Some(8),
            "Should have 8 partitions as configured"
        );

        // Verify PQ parameters are correctly derived
        let source_quantizer = source_vector_index.quantizer();
        let target_quantizer = target_vector_index.quantizer();
        let source_pq: ProductQuantizer = source_quantizer.try_into().unwrap();
        let target_pq: ProductQuantizer = target_quantizer.try_into().unwrap();

        let source_pq_params = derive_pq_params(&source_pq);
        let target_pq_params = derive_pq_params(&target_pq);

        assert_eq!(
            source_pq_params.num_sub_vectors, target_pq_params.num_sub_vectors,
            "PQ num_sub_vectors should match"
        );
        assert_eq!(
            source_pq_params.num_bits, target_pq_params.num_bits,
            "PQ num_bits should match"
        );
        assert_eq!(
            target_pq_params.num_sub_vectors, 8,
            "PQ should have 8 sub vectors"
        );
        assert_eq!(target_pq_params.num_bits, 8, "PQ should use 8 bits");

        // Verify HNSW parameters are extracted and used correctly
        let derived_hnsw_params = derive_hnsw_params(target_vector_index.as_ref());
        assert_eq!(
            derived_hnsw_params.max_level, 6,
            "HNSW max_level should be extracted as 6 from source index"
        );
        assert_eq!(
            derived_hnsw_params.m, 24,
            "HNSW m should be extracted as 24 from source index"
        );
        assert_eq!(
            derived_hnsw_params.ef_construction, 120,
            "HNSW ef_construction should be extracted as 120 from source index"
        );

        // Verify the index is functional by performing a search
        let query_vector = lance_datagen::gen_batch()
            .anon_col(array::rand_vec::<Float32Type>(32.into()))
            .into_batch_rows(RowCount::from(1))
            .unwrap()
            .column(0)
            .clone();
        let query_vector = query_vector
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .unwrap();
        let results = target_dataset
            .scan()
            .nearest("vector", &query_vector.value(0), 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 5, "Should return 5 nearest neighbors");
    }

    #[tokio::test]
    async fn test_initialize_vector_index_ivf_hnsw_sq() {
        let test_dir = TempStrDir::default();
        let source_uri = format!("{}/source", test_dir.as_str());
        let target_uri = format!("{}/target", test_dir.as_str());

        // Create source dataset with vector column
        let source_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(300), BatchCount::from(1));
        let mut source_dataset = Dataset::write(source_reader, &source_uri, None)
            .await
            .unwrap();

        // Create IVF_HNSW_SQ index on source with custom HNSW parameters
        let ivf_params = IvfBuildParams {
            num_partitions: Some(6),
            ..Default::default()
        };
        let hnsw_params = HnswBuildParams {
            max_level: 5,
            m: 16,
            ef_construction: 80,
            prefetch_distance: None,
        };
        let sq_params = SQBuildParams {
            num_bits: 8,
            ..Default::default()
        };
        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            MetricType::Cosine,
            ivf_params,
            hnsw_params,
            sq_params,
        );

        source_dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_ivf_hnsw_sq".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();

        // Reload to get updated metadata
        let source_dataset = Dataset::open(&source_uri).await.unwrap();
        let source_indices = source_dataset.load_indices().await.unwrap();
        let source_index = source_indices
            .iter()
            .find(|idx| idx.name == "vector_ivf_hnsw_sq")
            .unwrap();

        // Create target dataset with same schema
        let target_reader = lance_datagen::gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vector", array::rand_vec::<Float32Type>(32.into()))
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let mut target_dataset = Dataset::write(target_reader, &target_uri, None)
            .await
            .unwrap();

        // Initialize IVF_HNSW_SQ index on target
        initialize_vector_index(
            &mut target_dataset,
            &source_dataset,
            source_index,
            &["vector"],
        )
        .await
        .unwrap();

        // Verify index was created
        let target_indices = target_dataset.load_indices().await.unwrap();
        assert_eq!(target_indices.len(), 1, "Target should have 1 index");
        assert_eq!(
            target_indices[0].name, "vector_ivf_hnsw_sq",
            "Index name should match"
        );

        // Verify the index type and parameters match
        let target_vector_index = target_dataset
            .open_vector_index(
                "vector",
                &target_indices[0].uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let stats = target_vector_index.statistics().unwrap();

        // Check basic index type
        assert_eq!(
            stats.get("index_type").and_then(|v| v.as_str()),
            Some("IVF_HNSW_SQ"),
            "Index type should be IVF_HNSW_SQ"
        );

        // Check metric type
        assert_eq!(
            stats.get("metric_type").and_then(|v| v.as_str()),
            Some("cosine"),
            "Metric type should be cosine"
        );

        // Check number of partitions
        assert_eq!(
            stats.get("num_partitions").and_then(|v| v.as_u64()),
            Some(6),
            "Should have 6 partitions"
        );

        // Verify centroids are shared between source and target indices
        let source_vector_index = source_dataset
            .open_vector_index(
                "vector",
                &source_index.uuid.to_string(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();

        // Get IVF models from both indices to compare centroids
        let source_ivf_model = source_vector_index.ivf_model();
        let target_ivf_model = target_vector_index.ivf_model();

        // Verify they have the same number of partitions
        assert_eq!(
            source_ivf_model.num_partitions(),
            target_ivf_model.num_partitions(),
            "Source and target should have same number of partitions"
        );

        // Check sub_index contains SQ information
        let sub_index = stats
            .get("sub_index")
            .and_then(|v| v.as_object())
            .expect("IVF_HNSW_SQ index should have sub_index");

        // Verify SQ parameters
        assert_eq!(
            sub_index.get("num_bits").and_then(|v| v.as_u64()),
            Some(8),
            "SQ should use 8 bits"
        );

        // Verify IVF parameters are correctly derived
        let source_ivf_params = derive_ivf_params(source_ivf_model);
        let target_ivf_params = derive_ivf_params(target_ivf_model);
        assert_eq!(
            source_ivf_params.num_partitions, target_ivf_params.num_partitions,
            "IVF num_partitions should match"
        );
        assert_eq!(
            target_ivf_params.num_partitions,
            Some(6),
            "Should have 6 partitions as configured"
        );

        // Verify SQ parameters are correctly derived
        let source_quantizer = source_vector_index.quantizer();
        let target_quantizer = target_vector_index.quantizer();
        let source_sq: ScalarQuantizer = source_quantizer.try_into().unwrap();
        let target_sq: ScalarQuantizer = target_quantizer.try_into().unwrap();

        let source_sq_params = derive_sq_params(&source_sq);
        let target_sq_params = derive_sq_params(&target_sq);

        assert_eq!(
            source_sq_params.num_bits, target_sq_params.num_bits,
            "SQ num_bits should match"
        );
        assert_eq!(target_sq_params.num_bits, 8, "SQ should use 8 bits");

        // Verify HNSW parameters are extracted and used correctly
        let derived_hnsw_params = derive_hnsw_params(target_vector_index.as_ref());
        assert_eq!(
            derived_hnsw_params.max_level, 5,
            "HNSW max_level should be extracted as 5 from source index"
        );
        assert_eq!(
            derived_hnsw_params.m, 16,
            "HNSW m should be extracted as 16 from source index"
        );
        assert_eq!(
            derived_hnsw_params.ef_construction, 80,
            "HNSW ef_construction should be extracted as 80 from source index"
        );

        // Verify the index is functional by performing a search
        let query_vector = lance_datagen::gen_batch()
            .anon_col(array::rand_vec::<Float32Type>(32.into()))
            .into_batch_rows(RowCount::from(1))
            .unwrap()
            .column(0)
            .clone();
        let query_vector = query_vector
            .as_any()
            .downcast_ref::<arrow_array::FixedSizeListArray>()
            .unwrap();
        let results = target_dataset
            .scan()
            .nearest("vector", &query_vector.value(0), 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 5, "Should return 5 nearest neighbors");
    }
}
