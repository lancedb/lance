// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF - Inverted File index.

use std::{
    any::Any,
    collections::HashMap,
    sync::{Arc, Weak},
};

use arrow_arith::numeric::sub;
use arrow_array::{
    cast::{as_struct_array, AsArray},
    types::{Float16Type, Float32Type, Float64Type},
    Array, FixedSizeListArray, RecordBatch, StructArray, UInt32Array,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Schema};
use arrow_select::{concat::concat_batches, take::take};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use futures::{
    stream::{self, StreamExt},
    TryStreamExt,
};
use io::write_hnsw_quantization_index_partitions;
use lance_arrow::*;
use lance_core::{datatypes::Field, traits::DatasetTakeRows, Error, Result, ROW_ID_FIELD};
use lance_file::{
    format::MAGIC,
    writer::{FileWriter, FileWriterOptions},
};
use lance_index::vector::flat::index::{FlatIndex, FlatQuantizer};
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::quantizer::QuantizationType;
use lance_index::vector::v3::shuffler::IvfShuffler;
use lance_index::vector::v3::subindex::{IvfSubIndex, SubIndexType};
use lance_index::{
    optimize::OptimizeOptions,
    vector::{
        hnsw::{builder::HnswBuildParams, HNSWIndex, HNSW},
        ivf::{
            builder::load_precomputed_partitions, shuffler::shuffle_dataset,
            storage::IVF_PARTITION_KEY, IvfBuildParams,
        },
        pq::{PQBuildParams, ProductQuantizer},
        quantizer::{Quantization, QuantizationMetadata, Quantizer},
        sq::ScalarQuantizer,
        Query, VectorIndex, DIST_COL,
    },
    Index, IndexMetadata, IndexType, INDEX_AUXILIARY_FILE_NAME, INDEX_METADATA_SCHEMA_KEY,
};
use lance_io::{
    encodings::plain::PlainEncoder,
    local::to_local_path,
    object_writer::ObjectWriter,
    stream::RecordBatchStream,
    traits::{Reader, WriteExt, Writer},
};
use lance_linalg::distance::{DistanceType, Dot, MetricType, L2};
use lance_linalg::{
    distance::Normalize,
    kernels::{normalize_arrow, normalize_fsl},
};
use log::info;
use object_store::path::Path;
use rand::{rngs::SmallRng, SeedableRng};
use roaring::RoaringBitmap;
use serde::Serialize;
use serde_json::json;
use snafu::{location, Location};
use tracing::instrument;
use uuid::Uuid;

use super::{builder::IvfIndexBuilder, utils::PartitionLoadLock};
use super::{
    pq::{build_pq_model, PQIndex},
    utils::maybe_sample_training_data,
};
use crate::dataset::builder::DatasetBuilder;
use crate::{
    dataset::Dataset,
    index::{
        pb,
        prefilter::PreFilter,
        vector::{ivf::io::write_pq_partitions, Transformer},
        INDEX_FILE_NAME,
    },
    session::Session,
};

pub mod builder;
mod io;
pub mod v2;

/// IVF Index.
pub struct IVFIndex {
    uuid: String,

    /// Ivf model
    ivf: IvfModel,

    reader: Arc<dyn Reader>,

    /// Index in each partition.
    sub_index: Arc<dyn VectorIndex>,

    partition_locks: PartitionLoadLock,

    metric_type: MetricType,

    // The session cache holds an Arc to this object so we need to
    // hold a weak pointer to avoid cycles
    /// The session cache, used when fetching pages
    session: Weak<Session>,
}

impl DeepSizeOf for IVFIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.uuid.deep_size_of_children(context)
            + self.reader.deep_size_of_children(context)
            + self.sub_index.deep_size_of_children(context)
        // Skipping session since it is a weak ref
    }
}

impl IVFIndex {
    /// Create a new IVF index.
    pub(crate) fn try_new(
        session: Arc<Session>,
        uuid: &str,
        ivf: IvfModel,
        reader: Arc<dyn Reader>,
        sub_index: Arc<dyn VectorIndex>,
        metric_type: MetricType,
    ) -> Result<Self> {
        if !sub_index.is_loadable() {
            return Err(Error::Index {
                message: format!("IVF sub index must be loadable, got: {:?}", sub_index),
                location: location!(),
            });
        }

        let num_partitions = ivf.num_partitions();
        Ok(Self {
            uuid: uuid.to_owned(),
            session: Arc::downgrade(&session),
            ivf,
            reader,
            sub_index,
            metric_type,
            partition_locks: PartitionLoadLock::new(num_partitions),
        })
    }

    /// Load one partition of the IVF sub-index.
    ///
    /// Internal API with no stability guarantees.
    ///
    /// Parameters
    /// ----------
    ///  - partition_id: partition ID.
    #[instrument(level = "debug", skip(self))]
    pub async fn load_partition(
        &self,
        partition_id: usize,
        write_cache: bool,
    ) -> Result<Arc<dyn VectorIndex>> {
        let cache_key = format!("{}-ivf-{}", self.uuid, partition_id);
        let session = self.session.upgrade().ok_or(Error::Internal {
            message: "attempt to use index after dataset was destroyed".into(),
            location: location!(),
        })?;
        let part_index = if let Some(part_idx) = session.index_cache.get_vector(&cache_key) {
            part_idx
        } else {
            let mtx = self.partition_locks.get_partition_mutex(partition_id);
            let _guard = mtx.lock().await;
            // check the cache again, as the partition may have been loaded by another
            // thread that held the lock on loading the partition
            if let Some(part_idx) = session.index_cache.get_vector(&cache_key) {
                part_idx
            } else {
                if partition_id >= self.ivf.num_partitions() {
                    return Err(Error::Index {
                        message: format!(
                            "partition id {} is out of range of {} partitions",
                            partition_id,
                            self.ivf.num_partitions()
                        ),
                        location: location!(),
                    });
                }

                let range = self.ivf.row_range(partition_id);
                let idx = self
                    .sub_index
                    .load_partition(
                        self.reader.clone(),
                        range.start,
                        range.end - range.start,
                        partition_id,
                    )
                    .await?;
                let idx: Arc<dyn VectorIndex> = idx.into();
                if write_cache {
                    session.index_cache.insert_vector(&cache_key, idx.clone());
                }
                idx
            }
        };
        Ok(part_index)
    }

    /// preprocess the query vector given the partition id.
    ///
    /// Internal API with no stability guarantees.
    pub fn preprocess_query(&self, partition_id: usize, query: &Query) -> Result<Query> {
        if self.sub_index.use_residual() {
            let partition_centroids = self.ivf.centroids.as_ref().unwrap().value(partition_id);
            let residual_key = sub(&query.key, &partition_centroids)?;
            let mut part_query = query.clone();
            part_query.key = residual_key;
            Ok(part_query)
        } else {
            Ok(query.clone())
        }
    }
}

impl std::fmt::Debug for IVFIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Ivf({}) -> {:?}", self.metric_type, self.sub_index)
    }
}

// TODO: move to `lance-index` crate.
///
/// Returns (new_uuid, num_indices_merged)
pub(crate) async fn optimize_vector_indices(
    dataset: Dataset,
    unindexed: Option<impl RecordBatchStream + Unpin + 'static>,
    vector_column: &str,
    existing_indices: &[Arc<dyn Index>],
    options: &OptimizeOptions,
) -> Result<(Uuid, usize)> {
    // Sanity check the indices
    if existing_indices.is_empty() {
        return Err(Error::Index {
            message: "optimizing vector index: no existing index found".to_string(),
            location: location!(),
        });
    }

    // try cast to v1 IVFIndex,
    // fallback to v2 IVFIndex if it's not v1 IVFIndex
    if !existing_indices[0].as_any().is::<IVFIndex>() {
        return optimize_vector_indices_v2(
            &dataset,
            unindexed,
            vector_column,
            existing_indices,
            options,
        )
        .await;
    }

    let new_uuid = Uuid::new_v4();
    let object_store = dataset.object_store();
    let index_file = dataset
        .indices_dir()
        .child(new_uuid.to_string())
        .child(INDEX_FILE_NAME);
    let writer = object_store.create(&index_file).await?;

    let first_idx = existing_indices[0]
        .as_any()
        .downcast_ref::<IVFIndex>()
        .ok_or(Error::Index {
            message: "optimizing vector index: the first index isn't IVF".to_string(),
            location: location!(),
        })?;

    let merged = if let Some(pq_index) = first_idx.sub_index.as_any().downcast_ref::<PQIndex>() {
        optimize_ivf_pq_indices(
            first_idx,
            pq_index,
            vector_column,
            unindexed,
            existing_indices,
            options,
            writer,
            dataset.version().version,
        )
        .await?
    } else if let Some(hnsw_sq) = first_idx
        .sub_index
        .as_any()
        .downcast_ref::<HNSWIndex<ScalarQuantizer>>()
    {
        let aux_file = dataset
            .indices_dir()
            .child(new_uuid.to_string())
            .child(INDEX_AUXILIARY_FILE_NAME);
        let aux_writer = object_store.create(&aux_file).await?;
        optimize_ivf_hnsw_indices(
            Arc::new(dataset),
            first_idx,
            hnsw_sq,
            vector_column,
            unindexed,
            existing_indices,
            options,
            writer,
            aux_writer,
        )
        .await?
    } else {
        return Err(Error::Index {
            message: "optimizing vector index: the sub index isn't PQ or HNSW".to_string(),
            location: location!(),
        });
    };

    Ok((new_uuid, merged))
}

pub(crate) async fn optimize_vector_indices_v2(
    dataset: &Dataset,
    unindexed: Option<impl RecordBatchStream + Unpin + 'static>,
    vector_column: &str,
    existing_indices: &[Arc<dyn Index>],
    options: &OptimizeOptions,
) -> Result<(Uuid, usize)> {
    // Sanity check the indices
    if existing_indices.is_empty() {
        return Err(Error::Index {
            message: "optimizing vector index: no existing index found".to_string(),
            location: location!(),
        });
    }
    let existing_indices = existing_indices
        .iter()
        .cloned()
        .map(|idx| idx.as_vector_index())
        .collect::<Result<Vec<_>>>()?;

    let new_uuid = Uuid::new_v4();
    let index_dir = dataset.indices_dir().child(new_uuid.to_string());
    let ivf_model = existing_indices[0].ivf_model();
    let quantizer = existing_indices[0].quantizer();
    let distance_type = existing_indices[0].metric_type();
    let num_partitions = ivf_model.num_partitions();
    let index_type = existing_indices[0].sub_index_type();

    let temp_dir = tempfile::tempdir()?;
    let temp_dir = temp_dir.path().to_str().unwrap().into();
    let shuffler = Box::new(IvfShuffler::new(temp_dir, num_partitions));
    let start_pos = if options.num_indices_to_merge > existing_indices.len() {
        0
    } else {
        existing_indices.len() - options.num_indices_to_merge
    };
    let indices_to_merge = existing_indices[start_pos..].to_vec();
    let merged_num = indices_to_merge.len();
    match index_type {
        (SubIndexType::Flat, QuantizationType::Flat) => {
            IvfIndexBuilder::<FlatIndex, FlatQuantizer>::new_incremental(
                dataset.clone(),
                vector_column.to_owned(),
                index_dir,
                distance_type,
                shuffler,
                (),
            )?
            .with_ivf(ivf_model)
            .with_quantizer(quantizer.try_into()?)
            .with_existing_indices(indices_to_merge)
            .shuffle_data(unindexed)
            .await?
            .build()
            .await?;
        }

        (SubIndexType::Hnsw, QuantizationType::Scalar) => {
            IvfIndexBuilder::<HNSW, ScalarQuantizer>::new(
                dataset.clone(),
                vector_column.to_owned(),
                index_dir,
                distance_type,
                shuffler,
                None,
                None,
                // TODO: get the HNSW parameters from the existing indices
                HnswBuildParams::default(),
            )?
            .with_ivf(ivf_model)
            .with_quantizer(quantizer.try_into()?)
            .with_existing_indices(indices_to_merge)
            .shuffle_data(unindexed)
            .await?
            .build()
            .await?;
        }

        (sub_index_type, quantizer_type) => {
            return Err(Error::Index {
                message: format!(
                    "optimizing vector index: unsupported index type IVF_{}_{}",
                    sub_index_type, quantizer_type
                ),
                location: location!(),
            });
        }
    }

    Ok((new_uuid, merged_num))
}

#[allow(clippy::too_many_arguments)]
async fn optimize_ivf_pq_indices(
    first_idx: &IVFIndex,
    pq_index: &PQIndex,
    vector_column: &str,
    unindexed: Option<impl RecordBatchStream + Unpin + 'static>,
    existing_indices: &[Arc<dyn Index>],
    options: &OptimizeOptions,
    mut writer: ObjectWriter,
    dataset_version: u64,
) -> Result<usize> {
    let metric_type = first_idx.metric_type;
    let dim = first_idx.ivf.dimension();

    // TODO: merge `lance::vector::ivf::IVF` and `lance-index::vector::ivf::Ivf`` implementations.
    let ivf = lance_index::vector::ivf::IvfTransformer::with_pq(
        first_idx.ivf.centroids.clone().unwrap(),
        metric_type,
        vector_column,
        pq_index.pq.clone(),
        None,
    );

    // Shuffled un-indexed data with partition.
    let shuffled = match unindexed {
        Some(unindexed) => Some(
            shuffle_dataset(
                unindexed,
                vector_column,
                ivf.into(),
                None,
                first_idx.ivf.num_partitions() as u32,
                10000,
                2,
                None,
            )
            .await?,
        ),
        None => None,
    };

    let mut ivf_mut = IvfModel::new(first_idx.ivf.centroids.clone().unwrap());

    let start_pos = if options.num_indices_to_merge > existing_indices.len() {
        0
    } else {
        existing_indices.len() - options.num_indices_to_merge
    };

    let indices_to_merge = existing_indices[start_pos..]
        .iter()
        .map(|idx| {
            idx.as_any().downcast_ref::<IVFIndex>().ok_or(Error::Index {
                message: "optimizing vector index: it is not a IVF index".to_string(),
                location: location!(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    write_pq_partitions(&mut writer, &mut ivf_mut, shuffled, Some(&indices_to_merge)).await?;
    let metadata = IvfPQIndexMetadata {
        name: format!("_{}_idx", vector_column),
        column: vector_column.to_string(),
        dimension: dim as u32,
        dataset_version,
        metric_type,
        ivf: ivf_mut,
        pq: pq_index.pq.clone(),
        transforms: vec![],
    };

    let metadata = pb::Index::try_from(&metadata)?;
    let pos = writer.write_protobuf(&metadata).await?;
    // TODO: for now the IVF_PQ index file format hasn't been updated, so keep the old version,
    // change it to latest version value after refactoring the IVF_PQ
    writer.write_magics(pos, 0, 1, MAGIC).await?;
    writer.shutdown().await?;

    Ok(existing_indices.len() - start_pos)
}

#[allow(clippy::too_many_arguments)]
async fn optimize_ivf_hnsw_indices<Q: Quantization>(
    dataset: Arc<dyn DatasetTakeRows>,
    first_idx: &IVFIndex,
    hnsw_index: &HNSWIndex<Q>,
    vector_column: &str,
    unindexed: Option<impl RecordBatchStream + Unpin + 'static>,
    existing_indices: &[Arc<dyn Index>],
    options: &OptimizeOptions,
    writer: ObjectWriter,
    aux_writer: ObjectWriter,
) -> Result<usize> {
    let distance_type = first_idx.metric_type;
    let quantizer = hnsw_index.quantizer().clone();
    let ivf = lance_index::vector::ivf::new_ivf_transformer_with_quantizer(
        first_idx.ivf.centroids.clone().unwrap(),
        distance_type,
        vector_column,
        quantizer.clone(),
        None,
    )?;

    // Shuffled un-indexed data with partition.
    let unindexed_data = match unindexed {
        Some(unindexed) => Some(
            shuffle_dataset(
                unindexed,
                vector_column,
                Arc::new(ivf),
                None,
                first_idx.ivf.num_partitions() as u32,
                10000,
                2,
                None,
            )
            .await?,
        ),
        None => None,
    };

    let mut ivf_mut = IvfModel::new(first_idx.ivf.centroids.clone().unwrap());

    let start_pos = if options.num_indices_to_merge > existing_indices.len() {
        0
    } else {
        existing_indices.len() - options.num_indices_to_merge
    };

    let indices_to_merge = existing_indices[start_pos..]
        .iter()
        .map(|idx| {
            idx.as_any().downcast_ref::<IVFIndex>().ok_or(Error::Index {
                message: "optimizing vector index: it is not a IVF index".to_string(),
                location: location!(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // Prepare the HNSW writer
    let schema = lance_core::datatypes::Schema::try_from(HNSW::schema().as_ref())?;
    let mut writer = FileWriter::with_object_writer(writer, schema, &FileWriterOptions::default())?;
    writer.add_metadata(
        INDEX_METADATA_SCHEMA_KEY,
        json!(IndexMetadata {
            index_type: format!("IVF_HNSW_{}", quantizer.quantization_type()),
            distance_type: distance_type.to_string(),
        })
        .to_string()
        .as_str(),
    );

    // Prepare the quantization storage writer
    let schema = Schema::new(vec![
        ROW_ID_FIELD.clone(),
        arrow_schema::Field::new(
            quantizer.column(),
            DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new("item", DataType::UInt8, true)),
                quantizer.code_dim() as i32,
            ),
            false,
        ),
    ]);
    let schema = lance_core::datatypes::Schema::try_from(&schema)?;
    let mut aux_writer =
        FileWriter::with_object_writer(aux_writer, schema, &FileWriterOptions::default())?;
    aux_writer.add_metadata(
        INDEX_METADATA_SCHEMA_KEY,
        json!(IndexMetadata {
            index_type: quantizer.quantization_type().to_string(),
            distance_type: distance_type.to_string(),
        })
        .to_string()
        .as_str(),
    );

    // Write the metadata of quantizer
    let quantization_metadata = match &quantizer {
        Quantizer::Flat(_) => None,
        Quantizer::Product(pq) => {
            let codebook_tensor = pb::Tensor::try_from(&pq.codebook)?;
            let codebook_pos = aux_writer.tell().await?;
            aux_writer
                .object_writer
                .write_protobuf(&codebook_tensor)
                .await?;

            Some(QuantizationMetadata {
                codebook_position: Some(codebook_pos),
                ..Default::default()
            })
        }
        Quantizer::Scalar(_) => None,
    };

    aux_writer.add_metadata(
        quantizer.metadata_key(),
        quantizer
            .metadata(quantization_metadata)?
            .to_string()
            .as_str(),
    );

    let hnsw_params = &hnsw_index.metadata().params;
    let (hnsw_metadata, aux_ivf) = write_hnsw_quantization_index_partitions(
        dataset,
        vector_column,
        distance_type,
        hnsw_params,
        &mut writer,
        Some(&mut aux_writer),
        &mut ivf_mut,
        quantizer,
        unindexed_data,
        Some(&indices_to_merge),
    )
    .await?;

    // Add the metadata of HNSW partitions
    let hnsw_metadata_json = json!(hnsw_metadata);
    writer.add_metadata(IVF_PARTITION_KEY, &hnsw_metadata_json.to_string());

    ivf_mut.write(&mut writer).await?;
    writer.finish().await?;

    // Write the aux file
    aux_ivf.write(&mut aux_writer).await?;
    aux_writer.finish().await?;

    Ok(existing_indices.len() - start_pos)
}

#[derive(Serialize)]
pub struct IvfIndexPartitionStatistics {
    size: u32,
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
    centroids: Vec<Vec<f32>>,
}

fn centroids_to_vectors(centroids: &FixedSizeListArray) -> Result<Vec<Vec<f32>>> {
    centroids
        .iter()
        .map(|v| {
            if let Some(row) = v {
                match row.data_type() {
                    DataType::Float16 => Ok(row
                        .as_primitive::<Float16Type>()
                        .values()
                        .iter()
                        .map(|v| v.to_f32())
                        .collect::<Vec<_>>()),
                    DataType::Float32 => Ok(row.as_primitive::<Float32Type>().values().to_vec()),
                    DataType::Float64 => Ok(row
                        .as_primitive::<Float64Type>()
                        .values()
                        .iter()
                        .map(|v| *v as f32)
                        .collect::<Vec<_>>()),
                    _ => Err(Error::Index {
                        message: format!(
                            "IVF centroids must be FixedSizeList of floating number, got: {}",
                            row.data_type()
                        ),
                        location: location!(),
                    }),
                }
            } else {
                Err(Error::Index {
                    message: "Invalid centroid".to_string(),
                    location: location!(),
                })
            }
        })
        .collect()
}

#[async_trait]
impl Index for IVFIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn VectorIndex>> {
        Ok(self)
    }

    fn index_type(&self) -> IndexType {
        IndexType::Vector
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let partitions_statistics = (0..self.ivf.num_partitions())
            .map(|part_id| IvfIndexPartitionStatistics {
                size: self.ivf.partition_size(part_id) as u32,
            })
            .collect::<Vec<_>>();

        let centroid_vecs = centroids_to_vectors(self.ivf.centroids.as_ref().unwrap())?;

        Ok(serde_json::to_value(IvfIndexStatistics {
            index_type: "IVF".to_string(),
            uuid: self.uuid.clone(),
            uri: to_local_path(self.reader.path()),
            metric_type: self.metric_type.to_string(),
            num_partitions: self.ivf.num_partitions(),
            sub_index: self.sub_index.statistics()?,
            partitions: partitions_statistics,
            centroids: centroid_vecs,
        })?)
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let mut frag_ids = RoaringBitmap::default();
        let part_ids = 0..self.ivf.num_partitions();
        for part_id in part_ids {
            let part = self.load_partition(part_id, false).await?;
            frag_ids |= part.calculate_included_frags().await?;
        }
        Ok(frag_ids)
    }
}

#[async_trait]
impl VectorIndex for IVFIndex {
    #[instrument(level = "debug", skip_all, name = "IVFIndex::search")]
    async fn search(&self, query: &Query, pre_filter: Arc<dyn PreFilter>) -> Result<RecordBatch> {
        let mut query = query.clone();
        if self.metric_type == MetricType::Cosine {
            let key = normalize_arrow(&query.key)?;
            query.key = key;
        };

        let partition_ids = self.find_partitions(&query)?;
        assert!(partition_ids.len() <= query.nprobes);
        let part_ids = partition_ids.values().to_vec();
        let batches = stream::iter(part_ids)
            .map(|part_id| self.search_in_partition(part_id as usize, &query, pre_filter.clone()))
            .buffer_unordered(num_cpus::get())
            .try_collect::<Vec<_>>()
            .await?;
        let batch = concat_batches(&batches[0].schema(), &batches)?;

        let dist_col = batch.column_by_name(DIST_COL).ok_or_else(|| {
            Error::io(
                format!(
                    "_distance column does not exist in batch: {}",
                    batch.schema()
                ),
                location!(),
            )
        })?;

        // TODO: Use a heap sort to get the top-k.
        let limit = query.k * query.refine_factor.unwrap_or(1) as usize;
        let selection = sort_to_indices(dist_col, None, Some(limit))?;
        let struct_arr = StructArray::from(batch);
        let taken_distances = take(&struct_arr, &selection, None)?;
        Ok(as_struct_array(&taken_distances).into())
    }

    /// find the IVF partitions ids given the query vector.
    ///
    /// Internal API with no stability guarantees.
    ///
    /// Assumes the query vector is normalized if the metric type is cosine.
    fn find_partitions(&self, query: &Query) -> Result<UInt32Array> {
        let mt = if self.metric_type == MetricType::Cosine {
            MetricType::L2
        } else {
            self.metric_type
        };

        self.ivf.find_partitions(&query.key, query.nprobes, mt)
    }

    async fn search_in_partition(
        &self,
        partition_id: usize,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
    ) -> Result<RecordBatch> {
        let part_index = self.load_partition(partition_id, true).await?;

        let query = self.preprocess_query(partition_id, query)?;
        let batch = part_index.search(&query, pre_filter).await?;
        Ok(batch)
    }

    fn is_loadable(&self) -> bool {
        false
    }

    fn use_residual(&self) -> bool {
        false
    }

    fn check_can_remap(&self) -> Result<()> {
        Ok(())
    }

    async fn load(
        &self,
        _reader: Arc<dyn Reader>,
        _offset: usize,
        _length: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        Err(Error::Index {
            message: "Flat index does not support load".to_string(),
            location: location!(),
        })
    }

    fn row_ids(&self) -> Box<dyn Iterator<Item = &u64>> {
        todo!("this method is for only IVF_HNSW_* index");
    }

    fn remap(&mut self, _mapping: &HashMap<u64, Option<u64>>) -> Result<()> {
        // This will be needed if we want to clean up IVF to allow more than just
        // one layer (e.g. IVF -> IVF -> PQ).  We need to pass on the call to
        // remap to the lower layers.

        // Currently, remapping for IVF is implemented in remap_index_file which
        // mirrors some of the other IVF routines like build_ivf_pq_index
        Err(Error::Index {
            message: "Remapping IVF in this way not supported".to_string(),
            location: location!(),
        })
    }

    fn ivf_model(&self) -> IvfModel {
        self.ivf.clone()
    }

    fn quantizer(&self) -> Quantizer {
        unimplemented!("only for v2 IVFIndex")
    }

    /// the index type of this vector index.
    fn sub_index_type(&self) -> (SubIndexType, QuantizationType) {
        unimplemented!("only for v2 IVFIndex")
    }

    fn metric_type(&self) -> MetricType {
        self.metric_type
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
    pub(crate) ivf: IvfModel,

    /// Product Quantizer
    pub(crate) pq: ProductQuantizer,

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
                stage: Some(pb::vector_index_stage::Stage::Pq(pb::Pq::try_from(
                    &idx.pq,
                )?)),
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
                    MetricType::Hamming => pb::VectorMetricType::Hamming.into(),
                },
            })),
        })
    }
}

fn sanity_check<'a>(dataset: &'a Dataset, column: &str) -> Result<&'a Field> {
    let Some(field) = dataset.schema().field(column) else {
        return Err(Error::io(
            format!(
                "Building index: column {} does not exist in dataset: {:?}",
                column, dataset
            ),
            location!(),
        ));
    };
    if let DataType::FixedSizeList(elem_type, _) = field.data_type() {
        if !elem_type.data_type().is_floating() {
            return Err(Error::Index{
                message:format!(
                    "VectorIndex requires the column data type to be fixed size list of f16/f32/f64, got {}",
                    elem_type.data_type()
                ),
                location: location!()
            });
        }
    } else {
        return Err(Error::Index {
            message: format!(
            "VectorIndex requires the column data type to be fixed size list of float32s, got {}",
            field.data_type()
        ),
            location: location!(),
        });
    }
    Ok(field)
}

fn sanity_check_ivf_params(ivf: &IvfBuildParams) -> Result<()> {
    if ivf.precomputed_partitons_file.is_some() && ivf.centroids.is_none() {
        return Err(Error::Index {
            message: "precomputed_partitions_file requires centroids to be set".to_string(),
            location: location!(),
        });
    }

    if ivf.precomputed_shuffle_buffers.is_some() && ivf.centroids.is_none() {
        return Err(Error::Index {
            message: "precomputed_shuffle_buffers requires centroids to be set".to_string(),
            location: location!(),
        });
    }

    if ivf.precomputed_shuffle_buffers.is_some() && ivf.precomputed_partitons_file.is_some() {
        return Err(Error::Index {
            message:
                "precomputed_shuffle_buffers and precomputed_partitons_file are mutually exclusive"
                    .to_string(),
            location: location!(),
        });
    }

    Ok(())
}

fn sanity_check_params(ivf: &IvfBuildParams, pq: &PQBuildParams) -> Result<()> {
    sanity_check_ivf_params(ivf)?;
    if ivf.precomputed_shuffle_buffers.is_some() && pq.codebook.is_none() {
        return Err(Error::Index {
            message: "precomputed_shuffle_buffers requires codebooks to be set".to_string(),
            location: location!(),
        });
    }

    Ok(())
}

/// Build IVF model from the dataset.
///
/// Parameters
/// ----------
/// - *dataset*: Dataset instance
/// - *column*: vector column.
/// - *dim*: vector dimension.
/// - *metric_type*: distance metric type.
/// - *params*: IVF build parameters.
///
/// Returns
/// -------
/// - IVF model.
///
/// Visibility: pub(super) for testing
#[instrument(level = "debug", skip_all, name = "build_ivf_model")]
pub async fn build_ivf_model(
    dataset: &Dataset,
    column: &str,
    dim: usize,
    metric_type: MetricType,
    params: &IvfBuildParams,
) -> Result<IvfModel> {
    if let Some(centroids) = params.centroids.as_ref() {
        info!("Pre-computed IVF centroids is provided, skip IVF training");
        if centroids.values().len() != params.num_partitions * dim {
            return Err(Error::Index {
                message: format!(
                    "IVF centroids length mismatch: {} != {}",
                    centroids.len(),
                    params.num_partitions * dim,
                ),
                location: location!(),
            });
        }
        return Ok(IvfModel::new(centroids.as_ref().clone()));
    }
    let sample_size_hint = params.num_partitions * params.sample_rate;

    let start = std::time::Instant::now();
    info!(
        "Loading training data for IVF. Sample size: {}",
        sample_size_hint
    );
    let training_data = maybe_sample_training_data(dataset, column, sample_size_hint).await?;
    info!(
        "Finished loading training data in {:02} seconds",
        start.elapsed().as_secs_f32()
    );

    // If metric type is cosine, normalize the training data, and after this point,
    // treat the metric type as L2.
    let (training_data, mt) = if metric_type == MetricType::Cosine {
        let training_data = normalize_fsl(&training_data)?;
        (training_data, MetricType::L2)
    } else {
        (training_data, metric_type)
    };

    info!("Start to train IVF model");
    let start = std::time::Instant::now();
    let ivf = train_ivf_model(&training_data, mt, params).await?;
    info!(
        "Trained IVF model in {:02} seconds",
        start.elapsed().as_secs_f32()
    );
    Ok(ivf)
}

async fn build_ivf_model_and_pq(
    dataset: &Dataset,
    column: &str,
    metric_type: MetricType,
    ivf_params: &IvfBuildParams,
    pq_params: &PQBuildParams,
) -> Result<(IvfModel, ProductQuantizer)> {
    sanity_check_params(ivf_params, pq_params)?;

    info!(
        "Building vector index: IVF{},PQ{}, metric={}",
        ivf_params.num_partitions, pq_params.num_sub_vectors, metric_type,
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
            location: location!(),
        });
    };

    let ivf_model = build_ivf_model(dataset, column, dim, metric_type, ivf_params).await?;

    let ivf_residual = if matches!(metric_type, MetricType::Cosine | MetricType::L2) {
        Some(&ivf_model)
    } else {
        None
    };

    let pq = build_pq_model(dataset, column, dim, metric_type, pq_params, ivf_residual).await?;

    Ok((ivf_model, pq))
}

async fn scan_index_field_stream(
    dataset: &Dataset,
    column: &str,
) -> Result<impl RecordBatchStream + Unpin + 'static> {
    let mut scanner = dataset.scan();
    scanner.batch_readahead(num_cpus::get() * 2);
    scanner.project(&[column])?;
    scanner.with_row_id();
    scanner.try_into_stream().await
}

async fn load_precomputed_partitions_if_available(
    ivf_params: &IvfBuildParams,
) -> Result<Option<HashMap<u64, u32>>> {
    match &ivf_params.precomputed_partitons_file {
        Some(file) => {
            info!("Loading precomputed partitions from file: {}", file);
            let ds = DatasetBuilder::from_uri(file).load().await?;
            let stream = ds.scan().try_into_stream().await?;
            Ok(Some(
                load_precomputed_partitions(stream, ds.count_rows(None).await?).await?,
            ))
        }
        None => Ok(None),
    }
}

pub async fn build_ivf_pq_index(
    dataset: &Dataset,
    column: &str,
    index_name: &str,
    uuid: &str,
    metric_type: MetricType,
    ivf_params: &IvfBuildParams,
    pq_params: &PQBuildParams,
) -> Result<()> {
    let (ivf_model, pq) =
        build_ivf_model_and_pq(dataset, column, metric_type, ivf_params, pq_params).await?;
    let stream = scan_index_field_stream(dataset, column).await?;
    let precomputed_partitions = load_precomputed_partitions_if_available(ivf_params).await?;

    write_ivf_pq_file(
        dataset,
        column,
        index_name,
        uuid,
        &[],
        ivf_model,
        pq,
        metric_type,
        stream,
        precomputed_partitions,
        ivf_params.shuffle_partition_batches,
        ivf_params.shuffle_partition_concurrency,
        ivf_params.precomputed_shuffle_buffers.clone(),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn build_ivf_hnsw_pq_index(
    dataset: &Dataset,
    column: &str,
    index_name: &str,
    uuid: &str,
    metric_type: MetricType,
    ivf_params: &IvfBuildParams,
    hnsw_params: &HnswBuildParams,
    pq_params: &PQBuildParams,
) -> Result<()> {
    let (ivf_model, pq) =
        build_ivf_model_and_pq(dataset, column, metric_type, ivf_params, pq_params).await?;
    let stream = scan_index_field_stream(dataset, column).await?;
    let precomputed_partitions = load_precomputed_partitions_if_available(ivf_params).await?;

    write_ivf_hnsw_file(
        dataset,
        column,
        index_name,
        uuid,
        &[],
        ivf_model,
        Quantizer::Product(pq),
        metric_type,
        hnsw_params,
        stream,
        precomputed_partitions,
        ivf_params.shuffle_partition_batches,
        ivf_params.shuffle_partition_concurrency,
        ivf_params.precomputed_shuffle_buffers.clone(),
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
        reader: Arc<dyn Reader>,
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

    async fn write(self, writer: &mut ObjectWriter, ivf: &mut IvfModel) -> Result<()> {
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

fn generate_remap_tasks(offsets: &[usize], lengths: &[u32]) -> Result<Vec<RemapPageTask>> {
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

    let reader: Arc<dyn Reader> = object_store.open(&old_path).await?.into();
    let mut writer = object_store.create(&new_path).await?;

    let tasks = generate_remap_tasks(&index.ivf.offsets, &index.ivf.lengths)?;

    let mut task_stream = stream::iter(tasks.into_iter())
        .map(|task| task.load_and_remap(reader.clone(), index, mapping))
        .buffered(num_cpus::get());

    let mut ivf = IvfModel {
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
            location: location!(),
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
    // TODO: for now the IVF_PQ index file format hasn't been updated, so keep the old version,
    // change it to latest version value after refactoring the IVF_PQ
    writer.write_magics(pos, 0, 1, MAGIC).await?;
    writer.shutdown().await?;

    Ok(())
}

/// Write the index to the index file.
///
#[allow(clippy::too_many_arguments)]
async fn write_ivf_pq_file(
    dataset: &Dataset,
    column: &str,
    index_name: &str,
    uuid: &str,
    transformers: &[Box<dyn Transformer>],
    mut ivf: IvfModel,
    pq: ProductQuantizer,
    metric_type: MetricType,
    stream: impl RecordBatchStream + Unpin + 'static,
    precomputed_partitons: Option<HashMap<u64, u32>>,
    shuffle_partition_batches: usize,
    shuffle_partition_concurrency: usize,
    precomputed_shuffle_buffers: Option<(Path, Vec<String>)>,
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
        precomputed_partitons,
        shuffle_partition_batches,
        shuffle_partition_concurrency,
        precomputed_shuffle_buffers,
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
    // TODO: for now the IVF_PQ index file format hasn't been updated, so keep the old version,
    // change it to latest version value after refactoring the IVF_PQ
    writer.write_magics(pos, 0, 1, MAGIC).await?;
    writer.shutdown().await?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn write_ivf_hnsw_file(
    dataset: &Dataset,
    column: &str,
    _index_name: &str,
    uuid: &str,
    _transformers: &[Box<dyn Transformer>],
    mut ivf: IvfModel,
    quantizer: Quantizer,
    distance_type: DistanceType,
    hnsw_params: &HnswBuildParams,
    stream: impl RecordBatchStream + Unpin + 'static,
    precomputed_partitons: Option<HashMap<u64, u32>>,
    shuffle_partition_batches: usize,
    shuffle_partition_concurrency: usize,
    precomputed_shuffle_buffers: Option<(Path, Vec<String>)>,
) -> Result<()> {
    let object_store = dataset.object_store();
    let path = dataset.indices_dir().child(uuid).child(INDEX_FILE_NAME);
    let writer = object_store.create(&path).await?;

    let schema = lance_core::datatypes::Schema::try_from(HNSW::schema().as_ref())?;
    let mut writer = FileWriter::with_object_writer(writer, schema, &FileWriterOptions::default())?;
    writer.add_metadata(
        INDEX_METADATA_SCHEMA_KEY,
        json!(IndexMetadata {
            index_type: format!("IVF_HNSW_{}", quantizer.quantization_type()),
            distance_type: distance_type.to_string(),
        })
        .to_string()
        .as_str(),
    );

    let aux_path = dataset
        .indices_dir()
        .child(uuid)
        .child(INDEX_AUXILIARY_FILE_NAME);
    let aux_writer = object_store.create(&aux_path).await?;
    let schema = Schema::new(vec![
        ROW_ID_FIELD.clone(),
        arrow_schema::Field::new(
            quantizer.column(),
            DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new("item", DataType::UInt8, true)),
                quantizer.code_dim() as i32,
            ),
            false,
        ),
    ]);
    let schema = lance_core::datatypes::Schema::try_from(&schema)?;
    let mut aux_writer =
        FileWriter::with_object_writer(aux_writer, schema, &FileWriterOptions::default())?;
    aux_writer.add_metadata(
        INDEX_METADATA_SCHEMA_KEY,
        json!(IndexMetadata {
            index_type: quantizer.quantization_type().to_string(),
            distance_type: distance_type.to_string(),
        })
        .to_string()
        .as_str(),
    );

    // For PQ, we need to store the codebook
    let quantization_metadata = match &quantizer {
        Quantizer::Flat(_) => None,
        Quantizer::Product(pq) => {
            let codebook_tensor = pb::Tensor::try_from(&pq.codebook)?;
            let codebook_pos = aux_writer.tell().await?;
            aux_writer
                .object_writer
                .write_protobuf(&codebook_tensor)
                .await?;

            Some(QuantizationMetadata {
                codebook_position: Some(codebook_pos),
                ..Default::default()
            })
        }
        Quantizer::Scalar(_) => None,
    };

    aux_writer.add_metadata(
        quantizer.metadata_key(),
        quantizer
            .metadata(quantization_metadata)?
            .to_string()
            .as_str(),
    );

    let start = std::time::Instant::now();
    let num_partitions = ivf.num_partitions() as u32;

    let (hnsw_metadata, aux_ivf) = builder::build_hnsw_partitions(
        Arc::new(dataset.clone()),
        &mut writer,
        Some(&mut aux_writer),
        stream,
        column,
        &mut ivf,
        quantizer,
        distance_type,
        hnsw_params,
        0..num_partitions,
        precomputed_partitons,
        shuffle_partition_batches,
        shuffle_partition_concurrency,
        precomputed_shuffle_buffers,
    )
    .await?;
    info!("Built IVF partitions: {}s", start.elapsed().as_secs_f32());

    // Add the metadata of HNSW partitions
    let hnsw_metadata_json = json!(hnsw_metadata);
    writer.add_metadata(IVF_PARTITION_KEY, &hnsw_metadata_json.to_string());

    ivf.write(&mut writer).await?;
    writer.finish().await?;

    // Write the aux file
    aux_ivf.write(&mut aux_writer).await?;
    aux_writer.finish().await?;
    Ok(())
}

async fn do_train_ivf_model<T: ArrowFloatType + 'static>(
    data: &T::ArrayType,
    dimension: usize,
    metric_type: MetricType,
    params: &IvfBuildParams,
) -> Result<IvfModel>
where
    T::Native: Dot + L2 + Normalize,
{
    let rng = SmallRng::from_entropy();
    const REDOS: usize = 1;
    let centroids = lance_index::vector::kmeans::train_kmeans::<T>(
        data,
        dimension,
        params.num_partitions,
        params.max_iters as u32,
        REDOS,
        rng,
        metric_type,
        params.sample_rate,
    )
    .await?;
    Ok(IvfModel::new(FixedSizeListArray::try_new_from_values(
        centroids,
        dimension as i32,
    )?))
}

/// Train IVF partitions using kmeans.
async fn train_ivf_model(
    data: &FixedSizeListArray,
    distance_type: DistanceType,
    params: &IvfBuildParams,
) -> Result<IvfModel> {
    assert!(
        distance_type != DistanceType::Cosine,
        "Cosine metric should be done by normalized L2 distance",
    );
    let values = data.values();
    let dim = data.value_length() as usize;
    match (values.data_type(), distance_type) {
        (DataType::Float16, _) => {
            do_train_ivf_model::<Float16Type>(values.as_primitive(), dim, distance_type, params)
                .await
        }
        (DataType::Float32, _) => {
            do_train_ivf_model::<Float32Type>(values.as_primitive(), dim, distance_type, params)
                .await
        }
        (DataType::Float64, _) => {
            do_train_ivf_model::<Float64Type>(values.as_primitive(), dim, distance_type, params)
                .await
        }
        // (DataType::UInt8, DistanceType::Hamming) => {
        //     do_train_ivf_model::<UInt8Type>(values.as_primitive(), dim, distance_type, params).await
        // }
        _ => Err(Error::Index {
            message: "Unsupported data type".to_string(),
            location: location!(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashSet;
    use std::iter::repeat;
    use std::ops::Range;

    use arrow_array::types::UInt64Type;
    use arrow_array::{Float32Array, RecordBatchIterator, RecordBatchReader, UInt64Array};
    use arrow_schema::Field;
    use itertools::Itertools;
    use lance_core::utils::address::RowAddress;
    use lance_core::ROW_ID;
    use lance_index::vector::sq::builder::SQBuildParams;
    use lance_linalg::distance::l2_distance_batch;
    use lance_linalg::MatrixView;
    use lance_testing::datagen::{
        generate_random_array, generate_random_array_with_range, generate_random_array_with_seed,
        generate_scaled_random_array, sample_without_replacement,
    };
    use rand::{seq::SliceRandom, thread_rng};
    use tempfile::tempdir;

    use crate::index::prefilter::DatasetPreFilter;
    use crate::index::{vector::VectorIndexParams, DatasetIndexExt, DatasetIndexInternalExt};

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
            prefilter: Arc<dyn PreFilter>,
            ids_to_test: &[u64],
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
                    ef: None,
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

    async fn generate_test_dataset(
        test_uri: &str,
        range: Range<f32>,
    ) -> (Dataset, Arc<FixedSizeListArray>) {
        let vectors = generate_random_array_with_range::<Float32Type>(1000 * DIM, range);
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

        let (mut dataset, vector_array) = generate_test_dataset(test_uri, 0.0..1.0).await;

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

        let index = dataset
            .open_vector_index(WellKnownIvfPqData::COLUMN, &uuid_str)
            .await
            .unwrap();
        let ivf_index = index.as_any().downcast_ref::<IVFIndex>().unwrap();

        let index_meta = lance_table::format::Index {
            uuid,
            dataset_version: 0,
            fields: Vec::new(),
            name: INDEX_NAME.to_string(),
            fragment_bitmap: None,
        };

        let prefilter = Arc::new(DatasetPreFilter::new(dataset.clone(), &[index_meta], None));

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

        let remapped = dataset
            .open_vector_index(WellKnownIvfPqData::COLUMN, &new_uuid.to_string())
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

        let (mut dataset, vector_array) = generate_test_dataset(test_uri, 0.0..1.0).await;

        let centroids = generate_random_array(2 * DIM);
        let ivf_centroids = FixedSizeListArray::try_new_from_values(centroids, DIM as i32).unwrap();
        let ivf_params = IvfBuildParams::try_with_centroids(2, Arc::new(ivf_centroids)).unwrap();

        let pq_params = PQBuildParams::new(4, 8);

        let params =
            VectorIndexParams::with_ivf_pq_params(MetricType::Cosine, ivf_params, pq_params);

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
            dist.as_primitive::<Float32Type>()
                .values()
                .iter()
                .for_each(|v| {
                    assert!(
                        (0.0..2.0).contains(v),
                        "Expect cosine value in range [0.0, 2.0], got: {}",
                        v
                    )
                });
        }
    }

    #[tokio::test]
    async fn test_build_ivf_model_l2() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let (dataset, _) = generate_test_dataset(test_uri, 1000.0..1100.0).await;

        let ivf_params = IvfBuildParams::new(2);
        let ivf_model = build_ivf_model(&dataset, "vector", DIM, MetricType::L2, &ivf_params)
            .await
            .unwrap();
        assert_eq!(2, ivf_model.centroids.as_ref().unwrap().len());
        assert_eq!(32, ivf_model.centroids.as_ref().unwrap().value_length());
        assert_eq!(2, ivf_model.num_partitions());

        // All centroids values should be in the range [1000, 1100]
        ivf_model
            .centroids
            .unwrap()
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .for_each(|v| {
                assert!((1000.0..1100.0).contains(v));
            });
    }

    #[tokio::test]
    async fn test_build_ivf_model_cosine() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let (dataset, _) = generate_test_dataset(test_uri, 1000.0..1100.0).await;

        let ivf_params = IvfBuildParams::new(2);
        let ivf_model = build_ivf_model(&dataset, "vector", DIM, MetricType::Cosine, &ivf_params)
            .await
            .unwrap();
        assert_eq!(2, ivf_model.centroids.as_ref().unwrap().len());
        assert_eq!(32, ivf_model.centroids.as_ref().unwrap().value_length());
        assert_eq!(2, ivf_model.num_partitions());

        // All centroids values should be in the range [1000, 1100]
        ivf_model
            .centroids
            .unwrap()
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .for_each(|v| {
                assert!(
                    (-1.0..1.0).contains(v),
                    "Expect cosine value in range [-1.0, 1.0], got: {}",
                    v
                );
            });
    }

    #[tokio::test]
    async fn test_create_ivf_pq_dot() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let (mut dataset, vector_array) = generate_test_dataset(test_uri, 0.0..1.0).await;

        let centroids = generate_random_array(2 * DIM);
        let ivf_centroids = FixedSizeListArray::try_new_from_values(centroids, DIM as i32).unwrap();
        let ivf_params = IvfBuildParams::try_with_centroids(2, Arc::new(ivf_centroids)).unwrap();

        let codebook = Arc::new(generate_random_array(256 * DIM));
        let pq_params = PQBuildParams::with_codebook(4, 8, codebook);

        let params = VectorIndexParams::with_ivf_pq_params(MetricType::Dot, ivf_params, pq_params);

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
            assert!(dist
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .all(|v| (-2.0 * DIM as f32..0.0).contains(v)));
        }
    }

    #[tokio::test]
    async fn test_create_ivf_pq_f16() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        const DIM: usize = 32;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float16, true)),
                DIM as i32,
            ),
            true,
        )]));

        let arr = generate_random_array_with_seed::<Float16Type>(1000 * DIM, [22; 32]);
        let fsl = FixedSizeListArray::try_new_from_values(arr, DIM as i32).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let params = VectorIndexParams::with_ivf_pq_params(
            MetricType::L2,
            IvfBuildParams::new(2),
            PQBuildParams::new(4, 8),
        );
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let results = dataset
            .scan()
            .nearest(
                "vector",
                &Float32Array::from_iter_values(repeat(0.5).take(DIM)),
                5,
            )
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_rows(), 5);
        let batch = &results[0];
        assert_eq!(
            batch.schema(),
            Arc::new(Schema::new(vec![
                Field::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float16, true)),
                        DIM as i32,
                    ),
                    true,
                ),
                Field::new("_distance", DataType::Float32, true)
            ]))
        );
    }

    #[tokio::test]
    async fn test_create_ivf_pq_f16_with_codebook() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        const DIM: usize = 32;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float16, true)),
                DIM as i32,
            ),
            true,
        )]));

        let arr = generate_random_array_with_seed::<Float16Type>(1000 * DIM, [22; 32]);
        let fsl = FixedSizeListArray::try_new_from_values(arr, DIM as i32).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let codebook = Arc::new(generate_random_array_with_seed::<Float16Type>(
            256 * DIM,
            [22; 32],
        ));
        let params = VectorIndexParams::with_ivf_pq_params(
            MetricType::L2,
            IvfBuildParams::new(2),
            PQBuildParams::with_codebook(4, 8, codebook),
        );
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let results = dataset
            .scan()
            .nearest(
                "vector",
                &Float32Array::from_iter_values(repeat(0.5).take(DIM)),
                5,
            )
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].num_rows(), 5);
        let batch = &results[0];
        assert_eq!(
            batch.schema(),
            Arc::new(Schema::new(vec![
                Field::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float16, true)),
                        DIM as i32,
                    ),
                    true,
                ),
                Field::new("_distance", DataType::Float32, true)
            ]))
        );
    }

    #[tokio::test]
    async fn test_create_ivf_pq_with_invalid_num_sub_vectors() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        const DIM: usize = 32;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            true,
        )]));

        let arr = generate_random_array_with_seed::<Float32Type>(1000 * DIM, [22; 32]);
        let fsl = FixedSizeListArray::try_new_from_values(arr, DIM as i32).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let params = VectorIndexParams::with_ivf_pq_params(
            MetricType::L2,
            IvfBuildParams::new(256),
            PQBuildParams::new(6, 8),
        );
        let res = dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await;
        match &res {
            Err(Error::InvalidInput { source, .. }) => {
                assert!(
                    source
                        .to_string()
                        .contains("num_sub_vectors must divide vector dimension"),
                    "{:?}",
                    res
                );
            }
            _ => panic!("Expected InvalidInput error: {:?}", res),
        }
    }

    fn ground_truth(
        mat: &MatrixView<Float32Type>,
        query: &[f32],
        k: usize,
        distance_type: DistanceType,
    ) -> Vec<(f32, u32)> {
        let mut dists = vec![];
        for i in 0..mat.num_rows() {
            let dist = distance_type.func()(query, mat.row_ref(i).unwrap());
            dists.push((dist, i as u32));
        }
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.truncate(k);
        dists
    }

    #[tokio::test]
    async fn test_create_ivf_hnsw_pq() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let nlist = 4;
        let (mut dataset, vector_array) = generate_test_dataset(test_uri, 0.0..1.0).await;

        let ivf_params = IvfBuildParams::new(nlist);
        let pq_params = PQBuildParams::default();
        let hnsw_params = HnswBuildParams::default();
        let params = VectorIndexParams::with_ivf_hnsw_pq_params(
            MetricType::L2,
            ivf_params,
            hnsw_params,
            pq_params,
        );

        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let mat = MatrixView::<Float32Type>::try_from(vector_array.as_ref()).unwrap();
        let query = vector_array.value(0);
        let query = query.as_primitive::<Float32Type>();
        let k = 100;
        let results = dataset
            .scan()
            .with_row_id()
            .nearest("vector", query, k)
            .unwrap()
            .nprobs(nlist)
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(1, results.len());
        assert_eq!(k, results[0].num_rows());

        let row_ids = results[0]
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap() as u32)
            .collect::<Vec<_>>();
        let dists = results[0]
            .column_by_name("_distance")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()
            .to_vec();

        let results = dists.into_iter().zip(row_ids.into_iter()).collect_vec();
        let gt = ground_truth(&mat, query.values(), k, DistanceType::L2);

        let results_set = results.iter().map(|r| r.1).collect::<HashSet<_>>();
        let gt_set = gt.iter().map(|r| r.1).collect::<HashSet<_>>();

        let recall = results_set.intersection(&gt_set).count() as f32 / k as f32;
        assert!(
            recall >= 0.9,
            "recall: {}\n results: {:?}\n\ngt: {:?}",
            recall,
            results,
            gt,
        );
    }

    #[tokio::test]
    async fn test_create_ivf_hnsw_with_empty_partition() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        // the generate_test_dataset function generates a dataset with 1000 vectors,
        // so 1001 partitions will have at least one empty partition
        let nlist = 1001;
        let (mut dataset, vector_array) = generate_test_dataset(test_uri, 0.0..1.0).await;

        let centroids = generate_random_array(nlist * DIM);
        let ivf_centroids = FixedSizeListArray::try_new_from_values(centroids, DIM as i32).unwrap();
        let ivf_params =
            IvfBuildParams::try_with_centroids(nlist, Arc::new(ivf_centroids)).unwrap();

        let distance_type = DistanceType::L2;
        let sq_params = SQBuildParams::default();
        let hnsw_params = HnswBuildParams::default();
        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            distance_type,
            ivf_params,
            hnsw_params,
            sq_params,
        );

        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();

        let mat = MatrixView::<Float32Type>::try_from(vector_array.as_ref()).unwrap();
        let query = vector_array.value(0);
        let query = query.as_primitive::<Float32Type>();
        let k = 100;
        let results = dataset
            .scan()
            .with_row_id()
            .nearest("vector", query, k)
            .unwrap()
            .nprobs(nlist)
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(1, results.len());
        assert_eq!(k, results[0].num_rows());

        let row_ids = results[0]
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap() as u32)
            .collect::<Vec<_>>();
        let dists = results[0]
            .column_by_name("_distance")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()
            .to_vec();

        let results = dists.into_iter().zip(row_ids.into_iter()).collect_vec();
        let gt = ground_truth(&mat, query.values(), k, distance_type);

        let results_set = results.iter().map(|r| r.1).collect::<HashSet<_>>();
        let gt_set = gt.iter().map(|r| r.1).collect::<HashSet<_>>();

        let recall = results_set.intersection(&gt_set).count() as f32 / k as f32;
        assert!(
            recall >= 0.9,
            "recall: {}\n results: {:?}\n\ngt: {:?}",
            recall,
            results,
            gt,
        );
    }

    #[tokio::test]
    async fn test_check_cosine_normalization() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        const DIM: usize = 32;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            true,
        )]));

        let arr = generate_random_array_with_range(1000 * DIM, 1000.0..1001.0);
        let fsl = FixedSizeListArray::try_new_from_values(arr.clone(), DIM as i32).unwrap();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).unwrap();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let params = VectorIndexParams::ivf_pq(2, 8, 4, MetricType::Cosine, 50);
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, false)
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let idx = dataset
            .open_generic_index("vector", indices[0].uuid.to_string().as_str())
            .await
            .unwrap();
        let ivf_idx = idx.as_any().downcast_ref::<IVFIndex>().unwrap();

        assert!(ivf_idx
            .ivf
            .centroids
            .as_ref()
            .unwrap()
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .all(|v| (0.0..=1.0).contains(v)));

        let pq_idx = ivf_idx
            .sub_index
            .as_any()
            .downcast_ref::<PQIndex>()
            .unwrap();

        // PQ code is on residual space
        pq_idx
            .pq
            .codebook
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .for_each(|v| assert!((-1.0..=1.0).contains(v), "Got {}", v));

        let dataset = Dataset::open(test_uri).await.unwrap();

        let mut correct_times = 0;
        for query_id in 0..10 {
            let query = &arr.slice(query_id * DIM, DIM);
            let results = dataset
                .scan()
                .with_row_id()
                .nearest("vector", query, 1)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            assert_eq!(results.num_rows(), 1);
            let row_id = results
                .column_by_name("_rowid")
                .unwrap()
                .as_primitive::<UInt64Type>()
                .value(0);
            println!("Row id: {} query_id: {}", row_id, query_id);
            if row_id == (query_id as u64) {
                correct_times += 1;
            }
        }

        assert!(correct_times >= 9, "correct: {}", correct_times);
    }
}
