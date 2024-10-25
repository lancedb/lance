// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF - Inverted File index.

use core::fmt;
use std::marker::PhantomData;
use std::{
    any::Any,
    collections::HashMap,
    sync::{Arc, Weak},
};

use arrow::{
    array::as_struct_array,
    compute::{concat_batches, sort_to_indices, take},
};
use arrow_arith::numeric::sub;
use arrow_array::{RecordBatch, StructArray, UInt32Array};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use futures::prelude::stream::{self, StreamExt, TryStreamExt};
use lance_arrow::RecordBatchExt;
use lance_core::cache::FileMetadataCache;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{Error, Result};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::v2::reader::{FileReader, FileReaderOptions};
use lance_index::vector::flat::index::{FlatIndex, FlatQuantizer};
use lance_index::vector::hnsw::HNSW;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::quantizer::{QuantizationType, Quantizer};
use lance_index::vector::sq::ScalarQuantizer;
use lance_index::vector::v3::subindex::SubIndexType;
use lance_index::{
    pb,
    vector::{
        ivf::storage::IVF_METADATA_KEY, quantizer::Quantization, storage::IvfQuantizationStorage,
        v3::subindex::IvfSubIndex, Query, DISTANCE_TYPE_KEY, DIST_COL,
    },
    Index, IndexType, INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME,
};
use lance_index::{IndexMetadata, INDEX_METADATA_SCHEMA_KEY};
use lance_io::scheduler::SchedulerConfig;
use lance_io::{
    object_store::ObjectStore, scheduler::ScanScheduler, traits::Reader, ReadBatchParams,
};
use lance_linalg::{distance::DistanceType, kernels::normalize_arrow};
use moka::sync::Cache;
use object_store::path::Path;
use prost::Message;
use roaring::RoaringBitmap;
use serde_json::json;
use snafu::{location, Location};
use tracing::instrument;

use crate::index::vector::builder::index_type_string;
use crate::{
    index::{
        vector::{utils::PartitionLoadLock, VectorIndex},
        PreFilter,
    },
    session::Session,
};

use super::{centroids_to_vectors, IvfIndexPartitionStatistics, IvfIndexStatistics};

#[derive(Debug)]
struct PartitionEntry<S: IvfSubIndex, Q: Quantization> {
    index: S,
    storage: Q::Storage,
}

/// IVF Index.
#[derive(Debug)]
pub struct IVFIndex<S: IvfSubIndex + 'static, Q: Quantization + 'static> {
    uuid: String,

    /// Ivf model
    ivf: IvfModel,

    reader: FileReader,
    sub_index_metadata: Vec<String>,
    storage: IvfQuantizationStorage,

    /// Index in each partition.
    partition_cache: Cache<String, Arc<PartitionEntry<S, Q>>>,

    partition_locks: PartitionLoadLock,

    distance_type: DistanceType,

    // The session cache holds an Arc to this object so we need to
    // hold a weak pointer to avoid cycles
    /// The session cache, used when fetching pages
    #[allow(dead_code)]
    session: Weak<Session>,

    _marker: PhantomData<Q>,
}

impl<S: IvfSubIndex, Q: Quantization> DeepSizeOf for IVFIndex<S, Q> {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.uuid.deep_size_of_children(context) + self.storage.deep_size_of_children(context)
        // Skipping session since it is a weak ref
    }
}

impl<S: IvfSubIndex + 'static, Q: Quantization> IVFIndex<S, Q> {
    /// Create a new IVF index.
    pub(crate) async fn try_new(
        object_store: Arc<ObjectStore>,
        index_dir: Path,
        uuid: String,
        session: Weak<Session>,
    ) -> Result<Self> {
        let scheduler_config = SchedulerConfig::max_bandwidth(&object_store);
        let scheduler = ScanScheduler::new(object_store, scheduler_config);

        let file_metadata_cache = session
            .upgrade()
            .map(|sess| sess.file_metadata_cache.clone())
            .unwrap_or_else(FileMetadataCache::no_cache);
        let index_cache_capacity = session.upgrade().unwrap().index_cache.capacity();
        let index_reader = FileReader::try_open(
            scheduler
                .open_file(&index_dir.child(uuid.as_str()).child(INDEX_FILE_NAME))
                .await?,
            None,
            Arc::<DecoderPlugins>::default(),
            &file_metadata_cache,
            FileReaderOptions::default(),
        )
        .await?;
        let index_metadata: IndexMetadata = serde_json::from_str(
            index_reader
                .schema()
                .metadata
                .get(INDEX_METADATA_SCHEMA_KEY)
                .ok_or(Error::Index {
                    message: format!("{} not found", DISTANCE_TYPE_KEY),
                    location: location!(),
                })?
                .as_str(),
        )?;
        let distance_type = DistanceType::try_from(index_metadata.distance_type.as_str())?;

        let ivf_pos = index_reader
            .schema()
            .metadata
            .get(IVF_METADATA_KEY)
            .ok_or(Error::Index {
                message: format!("{} not found", IVF_METADATA_KEY),
                location: location!(),
            })?
            .parse()
            .map_err(|e| Error::Index {
                message: format!("Failed to decode IVF position: {}", e),
                location: location!(),
            })?;
        let ivf_pb_bytes = index_reader.read_global_buffer(ivf_pos).await?;
        let ivf = IvfModel::try_from(pb::Ivf::decode(ivf_pb_bytes)?)?;

        let sub_index_metadata = index_reader
            .schema()
            .metadata
            .get(S::metadata_key())
            .ok_or(Error::Index {
                message: format!("{} not found", S::metadata_key()),
                location: location!(),
            })?;
        let sub_index_metadata: Vec<String> = serde_json::from_str(sub_index_metadata)?;

        let storage_reader = FileReader::try_open(
            scheduler
                .open_file(
                    &index_dir
                        .child(uuid.as_str())
                        .child(INDEX_AUXILIARY_FILE_NAME),
                )
                .await?,
            None,
            Arc::<DecoderPlugins>::default(),
            &file_metadata_cache,
            FileReaderOptions::default(),
        )
        .await?;
        let storage = IvfQuantizationStorage::try_new(storage_reader).await?;

        let num_partitions = ivf.num_partitions();
        Ok(Self {
            uuid,
            ivf,
            reader: index_reader,
            storage,
            partition_cache: Cache::new(index_cache_capacity),
            partition_locks: PartitionLoadLock::new(num_partitions),
            sub_index_metadata,
            distance_type,
            session,
            _marker: PhantomData,
        })
    }

    #[instrument(level = "debug", skip(self))]
    pub async fn load_partition(
        &self,
        partition_id: usize,
        write_cache: bool,
    ) -> Result<Arc<PartitionEntry<S, Q>>> {
        let cache_key = format!("{}-ivf-{}", self.uuid, partition_id);
        let part_entry = if let Some(part_idx) = self.partition_cache.get(&cache_key) {
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

            let mtx = self.partition_locks.get_partition_mutex(partition_id);
            let _guard = mtx.lock().await;

            // check the cache again, as the partition may have been loaded by another
            // thread that held the lock on loading the partition
            if let Some(part_idx) = self.partition_cache.get(&cache_key) {
                part_idx
            } else {
                let schema = Arc::new(self.reader.schema().as_ref().into());
                let batch = match self.reader.metadata().num_rows {
                    0 => RecordBatch::new_empty(schema),
                    _ => {
                        let row_range = self.ivf.row_range(partition_id);
                        if row_range.is_empty() {
                            RecordBatch::new_empty(schema)
                        } else {
                            let batches = self
                                .reader
                                .read_stream(
                                    ReadBatchParams::Range(row_range),
                                    u32::MAX,
                                    1,
                                    FilterExpression::no_filter(),
                                )?
                                .try_collect::<Vec<_>>()
                                .await?;
                            concat_batches(&schema, batches.iter())?
                        }
                    }
                };
                let batch = batch.add_metadata(
                    S::metadata_key().to_owned(),
                    self.sub_index_metadata[partition_id].clone(),
                )?;
                let idx = S::load(batch)?;
                let storage = self.load_partition_storage(partition_id).await?;
                let partition_entry = Arc::new(PartitionEntry {
                    index: idx,
                    storage,
                });
                if write_cache {
                    self.partition_cache
                        .insert(cache_key.clone(), partition_entry.clone());
                }

                partition_entry
            }
        };

        Ok(part_entry)
    }

    pub async fn load_partition_storage(&self, partition_id: usize) -> Result<Q::Storage> {
        self.storage.load_partition::<Q>(partition_id).await
    }

    /// preprocess the query vector given the partition id.
    ///
    /// Internal API with no stability guarantees.
    #[instrument(level = "debug", skip(self))]
    pub fn preprocess_query(&self, partition_id: usize, query: &Query) -> Result<Query> {
        if Q::use_residual(self.distance_type) {
            let partition_centroids =
                self.ivf
                    .centroid(partition_id)
                    .ok_or_else(|| Error::Index {
                        message: format!("partition centroid {} does not exist", partition_id),
                        location: location!(),
                    })?;
            let residual_key = sub(&query.key, &partition_centroids)?;
            let mut part_query = query.clone();
            part_query.key = residual_key;
            Ok(part_query)
        } else {
            Ok(query.clone())
        }
    }
}

#[async_trait]
impl<S: IvfSubIndex + 'static, Q: Quantization + 'static> Index for IVFIndex<S, Q> {
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
        match self.sub_index_type() {
            (SubIndexType::Flat, QuantizationType::Flat) => IndexType::IvfFlat,
            (SubIndexType::Flat, QuantizationType::Product) => IndexType::IvfPq,
            (SubIndexType::Flat, QuantizationType::Scalar) => IndexType::IvfSq,
            (SubIndexType::Hnsw, QuantizationType::Product) => IndexType::IvfHnswPq,
            (SubIndexType::Hnsw, QuantizationType::Scalar) => IndexType::IvfHnswSq,
            _ => IndexType::Vector,
        }
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let partitions_statistics = (0..self.ivf.num_partitions())
            .map(|part_id| IvfIndexPartitionStatistics {
                size: self.ivf.partition_size(part_id) as u32,
            })
            .collect::<Vec<_>>();

        let centroid_vecs = centroids_to_vectors(self.ivf.centroids.as_ref().unwrap())?;

        let (sub_index_type, quantization_type) = self.sub_index_type();
        let index_type = index_type_string(sub_index_type, quantization_type);
        let mut sub_index_stats: serde_json::Value =
            if let Some(metadata) = self.sub_index_metadata.iter().find(|m| !m.is_empty()) {
                serde_json::from_str(metadata)?
            } else {
                json!({})
            };
        sub_index_stats["index_type"] = S::name().into();
        Ok(serde_json::to_value(IvfIndexStatistics {
            index_type,
            uuid: self.uuid.clone(),
            uri: self.uuid.clone(),
            metric_type: self.distance_type.to_string(),
            num_partitions: self.ivf.num_partitions(),
            sub_index: sub_index_stats,
            partitions: partitions_statistics,
            centroids: centroid_vecs,
        })?)
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!(
            "this method is only needed for migrating older manifests, not for this new index"
        )
    }
}

#[async_trait]
impl<S: IvfSubIndex + fmt::Debug + 'static, Q: Quantization + fmt::Debug + 'static> VectorIndex
    for IVFIndex<S, Q>
{
    async fn search(&self, query: &Query, pre_filter: Arc<dyn PreFilter>) -> Result<RecordBatch> {
        let mut query = query.clone();
        if self.distance_type == DistanceType::Cosine {
            let key = normalize_arrow(&query.key)?;
            query.key = key;
        };

        let partition_ids = self.find_partitions(&query)?;
        assert!(partition_ids.len() <= query.nprobes);
        let part_ids = partition_ids.values().to_vec();
        let batches = stream::iter(part_ids)
            .map(|part_id| self.search_in_partition(part_id as usize, &query, pre_filter.clone()))
            .buffer_unordered(get_num_compute_intensive_cpus())
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

    fn find_partitions(&self, query: &Query) -> Result<UInt32Array> {
        let dt = if self.distance_type == DistanceType::Cosine {
            DistanceType::L2
        } else {
            self.distance_type
        };

        self.ivf.find_partitions(&query.key, query.nprobes, dt)
    }

    // async fn append(&self, batches: Vec<RecordBatch>) -> Result<()> {
    //     IvfIndexBuilder::new(
    //         dataset,
    //         column,
    //         index_dir,
    //         distance_type,
    //         shuffler,
    //         ivf_params,
    //         sub_index_params,
    //         quantizer_params,
    //     )
    // }

    #[instrument(level = "debug", skip(self, pre_filter))]
    async fn search_in_partition(
        &self,
        partition_id: usize,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
    ) -> Result<RecordBatch> {
        let part_entry = self.load_partition(partition_id, true).await?;
        pre_filter.wait_for_ready().await?;

        let query = self.preprocess_query(partition_id, query)?;
        let param = (&query).into();
        let refine_factor = query.refine_factor.unwrap_or(1) as usize;
        let k = query.k * refine_factor;
        part_entry
            .index
            .search(query.key, k, param, &part_entry.storage, pre_filter)
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

    fn row_ids(&self) -> Box<dyn Iterator<Item = &'_ u64> + '_> {
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
        self.storage.quantizer::<Q>().unwrap()
    }

    /// the index type of this vector index.
    fn sub_index_type(&self) -> (SubIndexType, QuantizationType) {
        (S::name().try_into().unwrap(), Q::quantization_type())
    }

    fn metric_type(&self) -> DistanceType {
        self.distance_type
    }
}

pub type IvfFlatIndex = IVFIndex<FlatIndex, FlatQuantizer>;
pub type IvfPq = IVFIndex<FlatIndex, ProductQuantizer>;
pub type IvfHnswSqIndex = IVFIndex<HNSW, ScalarQuantizer>;
pub type IvfHnswPqIndex = IVFIndex<HNSW, ProductQuantizer>;

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::{collections::HashMap, ops::Range, sync::Arc};

    use arrow::datatypes::UInt64Type;
    use arrow::{array::AsArray, datatypes::Float32Type};
    use arrow_array::{Array, FixedSizeListArray, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;

    use lance_core::ROW_ID;
    use lance_index::vector::hnsw::builder::HnswBuildParams;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::pq::PQBuildParams;
    use lance_index::vector::sq::builder::SQBuildParams;
    use lance_index::vector::DIST_COL;
    use lance_index::{DatasetIndexExt, IndexType};
    use lance_linalg::distance::DistanceType;
    use lance_testing::datagen::generate_random_array_with_range;
    use rstest::rstest;
    use tempfile::tempdir;

    use crate::{index::vector::VectorIndexParams, Dataset};

    const DIM: usize = 32;

    async fn generate_test_dataset(
        test_uri: &str,
        range: Range<f32>,
    ) -> (Dataset, Arc<FixedSizeListArray>) {
        let vectors = generate_random_array_with_range::<Float32Type>(1000 * DIM, range);
        let metadata: HashMap<String, String> = vec![("test".to_string(), "ivf_pq".to_string())]
            .into_iter()
            .collect();

        let schema: Arc<_> = Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            true,
        )])
        .with_metadata(metadata)
        .into();
        let fsl = FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap();
        let fsl = lance_linalg::kernels::normalize_fsl(&fsl).unwrap();
        let array = Arc::new(fsl);
        let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()]).unwrap();

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write(batches, test_uri, None).await.unwrap();
        (dataset, array)
    }

    #[allow(dead_code)]
    fn ground_truth(
        vectors: &FixedSizeListArray,
        query: &[f32],
        k: usize,
        distance_type: DistanceType,
    ) -> Vec<(f32, u64)> {
        let mut dists = vec![];
        for i in 0..vectors.len() {
            let dist = distance_type.func()(
                query,
                vectors.value(i).as_primitive::<Float32Type>().values(),
            );
            dists.push((dist, i as u64));
        }
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.truncate(k);
        dists
    }

    async fn test_index(params: VectorIndexParams, nlist: usize, recall_requirement: f32) {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let (mut dataset, vectors) = generate_test_dataset(test_uri, 0.0..1.0).await;

        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        let query = vectors.value(0);
        let k = 100;
        let result = dataset
            .scan()
            .nearest("vector", query.as_primitive::<Float32Type>(), k)
            .unwrap()
            .nprobs(nlist)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();

        let row_ids = result[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec();
        let dists = result[DIST_COL]
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        let results = dists
            .into_iter()
            .zip(row_ids.into_iter())
            .collect::<Vec<_>>();
        let row_ids = results.iter().map(|(_, id)| *id).collect::<HashSet<_>>();

        let gt = ground_truth(
            &vectors,
            query.as_primitive::<Float32Type>().values(),
            k,
            DistanceType::L2,
        );
        let gt_set = gt.iter().map(|r| r.1).collect::<HashSet<_>>();

        let recall = row_ids.intersection(&gt_set).count() as f32 / k as f32;
        assert!(
            recall >= recall_requirement,
            "recall: {}\n results: {:?}\n\ngt: {:?}",
            recall,
            results,
            gt,
        );
    }

    #[rstest]
    #[case(4, DistanceType::L2, 1.0)]
    #[case(4, DistanceType::Cosine, 1.0)]
    #[case(4, DistanceType::Dot, 1.0)]
    #[tokio::test]
    async fn test_build_ivf_flat(
        #[case] nlist: usize,
        #[case] distance_type: DistanceType,
        #[case] recall_requirement: f32,
    ) {
        let params = VectorIndexParams::ivf_flat(nlist, distance_type);
        test_index(params, nlist, recall_requirement).await;
    }

    #[rstest]
    #[case(4, DistanceType::L2, 0.9)]
    #[case(4, DistanceType::Cosine, 0.6)]
    #[case(4, DistanceType::Dot, 0.2)]
    #[tokio::test]
    async fn test_build_ivf_pq(
        #[case] nlist: usize,
        #[case] distance_type: DistanceType,
        #[case] recall_requirement: f32,
    ) {
        let ivf_params = IvfBuildParams::new(nlist);
        let pq_params = PQBuildParams::default();
        let params = VectorIndexParams::with_ivf_pq_params(distance_type, ivf_params, pq_params);
        test_index(params, nlist, recall_requirement).await;
    }

    #[rstest]
    #[case(4, DistanceType::L2, 0.9)]
    #[case(4, DistanceType::Cosine, 0.9)]
    #[case(4, DistanceType::Dot, 0.9)]
    #[tokio::test]
    async fn test_create_ivf_hnsw_sq(
        #[case] nlist: usize,
        #[case] distance_type: DistanceType,
        #[case] recall_requirement: f32,
    ) {
        let ivf_params = IvfBuildParams::new(nlist);
        let sq_params = SQBuildParams::default();
        let hnsw_params = HnswBuildParams::default();
        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            distance_type,
            ivf_params,
            hnsw_params,
            sq_params,
        );
        test_index(params, nlist, recall_requirement).await;
    }

    #[rstest]
    #[case(4, DistanceType::L2, 0.9)]
    #[case(4, DistanceType::Cosine, 0.6)]
    #[case(4, DistanceType::Dot, 0.2)]
    #[tokio::test]
    async fn test_create_ivf_hnsw_pq(
        #[case] nlist: usize,
        #[case] distance_type: DistanceType,
        #[case] recall_requirement: f32,
    ) {
        let ivf_params = IvfBuildParams::new(nlist);
        let pq_params = PQBuildParams::default();
        let hnsw_params = HnswBuildParams::default();
        let params = VectorIndexParams::with_ivf_hnsw_pq_params(
            distance_type,
            ivf_params,
            hnsw_params,
            pq_params,
        );
        test_index(params, nlist, recall_requirement).await;
    }

    #[tokio::test]
    async fn test_index_stats() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let nlist = 4;
        let (mut dataset, _) = generate_test_dataset(test_uri, 0.0..1.0).await;

        let ivf_params = IvfBuildParams::new(nlist);
        let sq_params = SQBuildParams::default();
        let hnsw_params = HnswBuildParams::default();
        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            DistanceType::L2,
            ivf_params,
            hnsw_params,
            sq_params,
        );

        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("test_index".to_owned()),
                &params,
                true,
            )
            .await
            .unwrap();

        let stats = dataset.index_statistics("test_index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(stats.as_str()).unwrap();

        assert_eq!(stats["index_type"].as_str().unwrap(), "IVF_HNSW_SQ");
        for index in stats["indices"].as_array().unwrap() {
            assert_eq!(index["index_type"].as_str().unwrap(), "IVF_HNSW_SQ");
            assert_eq!(
                index["num_partitions"].as_number().unwrap(),
                &serde_json::Number::from(nlist)
            );
            assert_eq!(index["sub_index"]["index_type"].as_str().unwrap(), "HNSW");
        }
    }

    #[tokio::test]
    async fn test_index_stats_empty_partition() {
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let nlist = 1000;
        let (mut dataset, _) = generate_test_dataset(test_uri, 0.0..1.0).await;

        let ivf_params = IvfBuildParams::new(nlist);
        let sq_params = SQBuildParams::default();
        let hnsw_params = HnswBuildParams::default();
        let params = VectorIndexParams::with_ivf_hnsw_sq_params(
            DistanceType::L2,
            ivf_params,
            hnsw_params,
            sq_params,
        );

        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("test_index".to_owned()),
                &params,
                true,
            )
            .await
            .unwrap();

        let stats = dataset.index_statistics("test_index").await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(stats.as_str()).unwrap();

        assert_eq!(stats["index_type"].as_str().unwrap(), "IVF_HNSW_SQ");
        for index in stats["indices"].as_array().unwrap() {
            assert_eq!(index["index_type"].as_str().unwrap(), "IVF_HNSW_SQ");
            assert_eq!(
                index["num_partitions"].as_number().unwrap(),
                &serde_json::Number::from(nlist)
            );
            assert_eq!(index["sub_index"]["index_type"].as_str().unwrap(), "HNSW");
        }
    }
}
