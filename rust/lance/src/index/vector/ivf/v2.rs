// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF - Inverted File index.

use std::marker::PhantomData;
use std::{any::Any, collections::HashMap, sync::Arc};

use crate::index::vector::{builder::index_type_string, IndexFileVersion};
use crate::index::{
    vector::{utils::PartitionLoadLock, VectorIndex},
    PreFilter,
};
use arrow::compute::concat_batches;
use arrow_arith::numeric::sub;
use arrow_array::{Float32Array, RecordBatch, UInt32Array};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use deepsize::DeepSizeOf;
use futures::prelude::stream::{self, TryStreamExt};
use lance_arrow::RecordBatchExt;
use lance_core::cache::{CacheKey, LanceCache, WeakLanceCache};
use lance_core::utils::tokio::spawn_cpu;
use lance_core::utils::tracing::{IO_TYPE_LOAD_VECTOR_PART, TRACE_IO_EVENTS};
use lance_core::{Error, Result, ROW_ID};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::v2::reader::{FileReader, FileReaderOptions};
use lance_index::frag_reuse::FragReuseIndex;
use lance_index::metrics::{LocalMetricsCollector, MetricsCollector};
use lance_index::vector::flat::index::{FlatIndex, FlatQuantizer};
use lance_index::vector::hnsw::HNSW;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::quantizer::{QuantizationType, Quantizer};
use lance_index::vector::sq::ScalarQuantizer;
use lance_index::vector::storage::VectorStore;
use lance_index::vector::v3::subindex::SubIndexType;
use lance_index::vector::VectorIndexCacheEntry;
use lance_index::{
    pb,
    vector::{
        ivf::storage::IVF_METADATA_KEY, quantizer::Quantization, storage::IvfQuantizationStorage,
        v3::subindex::IvfSubIndex, Query, DISTANCE_TYPE_KEY,
    },
    Index, IndexType, INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME,
};
use lance_index::{IndexMetadata, INDEX_METADATA_SCHEMA_KEY};
use lance_io::local::to_local_path;
use lance_io::scheduler::SchedulerConfig;
use lance_io::utils::CachedFileSize;
use lance_io::{
    object_store::ObjectStore, scheduler::ScanScheduler, traits::Reader, ReadBatchParams,
};
use lance_linalg::distance::DistanceType;
use object_store::path::Path;
use prost::Message;
use roaring::RoaringBitmap;
use snafu::location;
use tracing::{info, instrument};

use super::{centroids_to_vectors, IvfIndexPartitionStatistics, IvfIndexStatistics};

#[derive(Debug, DeepSizeOf)]
pub struct PartitionEntry<S: IvfSubIndex, Q: Quantization> {
    pub index: S,
    pub storage: Q::Storage,
}

impl<S: IvfSubIndex + 'static, Q: Quantization + 'static> VectorIndexCacheEntry
    for PartitionEntry<S, Q>
{
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Cache key for IVF partitions
#[derive(Debug, Clone)]
pub struct IVFPartitionKey<S: IvfSubIndex, Q: Quantization> {
    pub partition_id: usize,
    _marker: PhantomData<(S, Q)>,
}

impl<S: IvfSubIndex, Q: Quantization> IVFPartitionKey<S, Q> {
    pub fn new(partition_id: usize) -> Self {
        Self {
            partition_id,
            _marker: PhantomData,
        }
    }
}

impl<S: IvfSubIndex + 'static, Q: Quantization + 'static> CacheKey for IVFPartitionKey<S, Q> {
    type ValueType = PartitionEntry<S, Q>;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("ivf-{}", self.partition_id).into()
    }
}

/// IVF Index.
#[derive(Debug)]
pub struct IVFIndex<S: IvfSubIndex + 'static, Q: Quantization + 'static> {
    uri: String,
    uuid: String,

    /// Ivf model
    ivf: IvfModel,

    reader: FileReader,
    sub_index_metadata: Vec<String>,
    storage: IvfQuantizationStorage<Q>,

    partition_locks: PartitionLoadLock,

    distance_type: DistanceType,

    index_cache: WeakLanceCache,

    _marker: PhantomData<(S, Q)>,
}

impl<S: IvfSubIndex, Q: Quantization> DeepSizeOf for IVFIndex<S, Q> {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.uri.deep_size_of_children(context)
            + self.ivf.deep_size_of_children(context)
            + self.sub_index_metadata.deep_size_of_children(context)
            + self.uuid.deep_size_of_children(context)
            + self.storage.deep_size_of_children(context)
        // Skipping session since it is a weak ref
    }
}

impl<S: IvfSubIndex + 'static, Q: Quantization> IVFIndex<S, Q> {
    /// Create a new IVF index.
    pub(crate) async fn try_new(
        object_store: Arc<ObjectStore>,
        index_dir: Path,
        uuid: String,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        file_metadata_cache: &LanceCache,
        index_cache: LanceCache,
    ) -> Result<Self> {
        let scheduler_config = SchedulerConfig::max_bandwidth(&object_store);
        let scheduler = ScanScheduler::new(object_store, scheduler_config);

        let uri = index_dir.child(uuid.as_str()).child(INDEX_FILE_NAME);
        let index_reader = FileReader::try_open(
            scheduler
                .open_file(&uri, &CachedFileSize::unknown())
                .await?,
            None,
            Arc::<DecoderPlugins>::default(),
            file_metadata_cache,
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
                    &CachedFileSize::unknown(),
                )
                .await?,
            None,
            Arc::<DecoderPlugins>::default(),
            file_metadata_cache,
            FileReaderOptions::default(),
        )
        .await?;
        let storage =
            IvfQuantizationStorage::try_new(storage_reader, frag_reuse_index.clone()).await?;

        let num_partitions = ivf.num_partitions();
        Ok(Self {
            uri: to_local_path(&uri),
            uuid,
            ivf,
            reader: index_reader,
            storage,
            partition_locks: PartitionLoadLock::new(num_partitions),
            sub_index_metadata,
            distance_type,
            index_cache: WeakLanceCache::from(&index_cache),
            _marker: PhantomData,
        })
    }

    #[instrument(level = "debug", skip(self, metrics))]
    pub async fn load_partition(
        &self,
        partition_id: usize,
        write_cache: bool,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn VectorIndexCacheEntry>> {
        let cache_key = IVFPartitionKey::<S, Q>::new(partition_id);
        let part_entry = if let Some(part_idx) = self.index_cache.get_with_key(&cache_key).await {
            part_idx
        } else {
            info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_VECTOR_PART, index_type="ivf", part_id=cache_key.key().as_ref());
            metrics.record_part_load();
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
            if let Some(part_idx) = self.index_cache.get_with_key(&cache_key).await {
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
                let partition_entry = Arc::new(PartitionEntry::<S, Q> {
                    index: idx,
                    storage,
                });
                if write_cache {
                    self.index_cache
                        .insert_with_key(&cache_key, partition_entry.clone())
                        .await;
                }

                partition_entry
            }
        };

        Ok(part_entry)
    }

    pub async fn load_partition_storage(&self, partition_id: usize) -> Result<Q::Storage> {
        self.storage.load_partition(partition_id).await
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

    async fn prewarm(&self) -> Result<()> {
        // TODO: We should prewarm the IVF index by loading the partitions into memory
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        match self.sub_index_type() {
            (SubIndexType::Flat, QuantizationType::Flat) => IndexType::IvfFlat,
            (SubIndexType::Flat, QuantizationType::Product) => IndexType::IvfPq,
            (SubIndexType::Flat, QuantizationType::Scalar) => IndexType::IvfSq,
            (SubIndexType::Flat, QuantizationType::Rabit) => IndexType::IvfRq,
            (SubIndexType::Hnsw, QuantizationType::Product) => IndexType::IvfHnswPq,
            (SubIndexType::Hnsw, QuantizationType::Scalar) => IndexType::IvfHnswSq,
            (SubIndexType::Hnsw, QuantizationType::Flat) => IndexType::IvfHnswFlat,
            (sub_index_type, quantization_type) => {
                unimplemented!(
                    "unsupported index type: {}, {}",
                    sub_index_type,
                    quantization_type
                )
            }
        }
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let partitions_statistics = (0..self.ivf.num_partitions())
            .map(|part_id| IvfIndexPartitionStatistics {
                size: self.storage.partition_size(part_id) as u32,
            })
            .collect::<Vec<_>>();

        let centroid_vecs = centroids_to_vectors(self.ivf.centroids.as_ref().unwrap())?;

        let (sub_index_type, quantization_type) = self.sub_index_type();
        let index_type = index_type_string(sub_index_type, quantization_type);
        let mut sub_index_stats: serde_json::Map<String, serde_json::Value> =
            if let Some(metadata) = self.sub_index_metadata.iter().find(|m| !m.is_empty()) {
                serde_json::from_str(metadata)?
            } else {
                serde_json::map::Map::new()
            };
        let mut store_stats = serde_json::to_value(self.storage.metadata())?;
        let store_stats = store_stats.as_object_mut().ok_or(Error::Internal {
            message: "failed to get storage metadata".to_string(),
            location: location!(),
        })?;

        sub_index_stats.append(store_stats);
        if S::name() == "FLAT" {
            sub_index_stats.insert(
                "index_type".to_string(),
                Q::quantization_type().to_string().into(),
            );
        } else {
            sub_index_stats.insert("index_type".to_string(), S::name().into());
        }

        let sub_index_distance_type = if matches!(Q::quantization_type(), QuantizationType::Product)
            && self.distance_type == DistanceType::Cosine
        {
            DistanceType::L2
        } else {
            self.distance_type
        };
        sub_index_stats.insert(
            "metric_type".to_string(),
            sub_index_distance_type.to_string().into(),
        );

        // we need to drop some stats from the metadata
        sub_index_stats.remove("codebook_position");
        sub_index_stats.remove("codebook");
        sub_index_stats.remove("codebook_tensor");

        Ok(serde_json::to_value(IvfIndexStatistics {
            index_type,
            uuid: self.uuid.clone(),
            uri: self.uri.clone(),
            metric_type: self.distance_type.to_string(),
            num_partitions: self.ivf.num_partitions(),
            sub_index: serde_json::Value::Object(sub_index_stats),
            partitions: partitions_statistics,
            centroids: centroid_vecs,
            loss: self.ivf.loss(),
            index_file_version: IndexFileVersion::V3,
        })?)
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!(
            "this method is only needed for migrating older manifests, not for this new index"
        )
    }
}

#[async_trait]
impl<S: IvfSubIndex + 'static, Q: Quantization + 'static> VectorIndex for IVFIndex<S, Q> {
    async fn search(
        &self,
        _query: &Query,
        _pre_filter: Arc<dyn PreFilter>,
        _metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        unimplemented!("IVFIndex not currently used as sub-index and top-level indices do partition-aware search")
    }

    fn find_partitions(&self, query: &Query) -> Result<(UInt32Array, Float32Array)> {
        let dt = if self.distance_type == DistanceType::Cosine {
            DistanceType::L2
        } else {
            self.distance_type
        };

        let max_nprobes = query.maximum_nprobes.unwrap_or(self.ivf.num_partitions());

        self.ivf.find_partitions(&query.key, max_nprobes, dt)
    }

    fn total_partitions(&self) -> usize {
        self.ivf.num_partitions()
    }

    #[instrument(level = "debug", skip(self, pre_filter, metrics))]
    async fn search_in_partition(
        &self,
        partition_id: usize,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        let part_entry = self.load_partition(partition_id, true, metrics).await?;
        pre_filter.wait_for_ready().await?;

        let query = self.preprocess_query(partition_id, query)?;
        let (batch, local_metrics) = spawn_cpu(move || {
            let param = (&query).into();
            let refine_factor = query.refine_factor.unwrap_or(1) as usize;
            let k = query.k * refine_factor;
            let local_metrics = LocalMetricsCollector::default();
            let part = part_entry
                .as_any()
                .downcast_ref::<PartitionEntry<S, Q>>()
                .ok_or(Error::Internal {
                    message: "failed to downcast partition entry".to_string(),
                    location: location!(),
                })?;
            let batch = part.index.search(
                query.key,
                k,
                param,
                &part.storage,
                pre_filter,
                &local_metrics,
            )?;
            Ok((batch, local_metrics))
        })
        .await?;

        local_metrics.dump_into(metrics);

        Ok(batch)
    }

    fn is_loadable(&self) -> bool {
        false
    }

    fn use_residual(&self) -> bool {
        false
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

    async fn partition_reader(
        &self,
        partition_id: usize,
        with_vector: bool,
        metrics: &dyn MetricsCollector,
    ) -> Result<SendableRecordBatchStream> {
        let partition = self.load_partition(partition_id, false, metrics).await?;
        let partition = partition
            .as_any()
            .downcast_ref::<PartitionEntry<S, Q>>()
            .ok_or(Error::Internal {
                message: "failed to downcast partition entry".to_string(),
                location: location!(),
            })?;
        let store = &partition.storage;
        let schema = if with_vector {
            store.schema().clone()
        } else {
            let schema = store.schema();
            let row_id_idx = schema.index_of(ROW_ID)?;
            Arc::new(store.schema().project(&[row_id_idx])?)
        };

        let batches = store
            .to_batches()?
            .map(|b| {
                let batch = b.project_by_schema(&schema)?;
                Ok(batch)
            })
            .collect::<Vec<_>>();
        let stream = RecordBatchStreamAdapter::new(schema, stream::iter(batches));
        Ok(Box::pin(stream))
    }

    async fn to_batch_stream(&self, _with_vector: bool) -> Result<SendableRecordBatchStream> {
        unimplemented!("this method is for only sub index");
    }

    fn num_rows(&self) -> u64 {
        self.storage.num_rows()
    }

    fn row_ids(&self) -> Box<dyn Iterator<Item = &'_ u64> + '_> {
        todo!("this method is for only IVF_HNSW_* index");
    }

    async fn remap(&mut self, _mapping: &HashMap<u64, Option<u64>>) -> Result<()> {
        Err(Error::Index {
            message: "Remapping IVF in this way not supported".to_string(),
            location: location!(),
        })
    }

    fn ivf_model(&self) -> &IvfModel {
        &self.ivf
    }

    fn quantizer(&self) -> Quantizer {
        self.storage.quantizer().unwrap()
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
    use std::{ops::Range, sync::Arc};

    use all_asserts::{assert_ge, assert_lt};
    use arrow::datatypes::{Float64Type, UInt64Type, UInt8Type};
    use arrow::{array::AsArray, datatypes::Float32Type};
    use arrow_array::{
        Array, ArrayRef, ArrowPrimitiveType, FixedSizeListArray, Float32Array, Int64Array,
        ListArray, RecordBatch, RecordBatchIterator, UInt64Array,
    };
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use itertools::Itertools;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::bq::RQBuildParams;
    use lance_index::vector::storage::VectorStore;

    use crate::dataset::{InsertBuilder, UpdateBuilder, WriteMode, WriteParams};
    use crate::index::vector::ivf::v2::IvfPq;
    use crate::index::DatasetIndexInternalExt;
    use crate::utils::test::copy_test_data_to_tmp;
    use crate::{
        dataset::optimize::{compact_files, CompactionOptions},
        index::vector::IndexFileVersion,
    };
    use crate::{
        index::vector::{VectorIndex, VectorIndexParams},
        Dataset,
    };
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::{Result, ROW_ID};
    use lance_encoding::decoder::DecoderPlugins;
    use lance_file::v2::{
        reader::{FileReader, FileReaderOptions},
        writer::FileWriter,
    };
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::pq::PQBuildParams;
    use lance_index::vector::quantizer::QuantizerMetadata;
    use lance_index::vector::sq::builder::SQBuildParams;
    use lance_index::vector::DIST_COL;
    use lance_index::vector::{
        pq::storage::ProductQuantizationMetadata, storage::STORAGE_METADATA_KEY,
    };
    use lance_index::{metrics::NoOpMetricsCollector, INDEX_AUXILIARY_FILE_NAME};
    use lance_index::{optimize::OptimizeOptions, scalar::IndexReader};
    use lance_index::{scalar::IndexWriter, vector::hnsw::builder::HnswBuildParams};
    use lance_index::{DatasetIndexExt, IndexType};
    use lance_io::{
        object_store::ObjectStore,
        scheduler::{ScanScheduler, SchedulerConfig},
        utils::CachedFileSize,
    };
    use lance_linalg::distance::{multivec_distance, DistanceType};
    use lance_linalg::kernels::normalize_fsl;
    use lance_testing::datagen::{generate_random_array, generate_random_array_with_range};
    use object_store::path::Path;
    use rand::distr::uniform::SampleUniform;
    use rstest::rstest;

    const NUM_ROWS: usize = 512;
    const DIM: usize = 32;
    const PARTITION_SPLIT_APPEND_ROWS: usize = 50_000;

    async fn generate_test_dataset<T: ArrowPrimitiveType>(
        test_uri: &str,
        range: Range<T::Native>,
    ) -> (Dataset, Arc<FixedSizeListArray>)
    where
        T::Native: SampleUniform,
    {
        let (batch, schema) = generate_batch::<T>(NUM_ROWS, None, range, false);
        let vectors = batch.column_by_name("vector").unwrap().clone();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                mode: crate::dataset::WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        (dataset, Arc::new(vectors.as_fixed_size_list().clone()))
    }

    async fn generate_multivec_test_dataset<T: ArrowPrimitiveType>(
        test_uri: &str,
        range: Range<T::Native>,
    ) -> (Dataset, Arc<ListArray>)
    where
        T::Native: SampleUniform,
    {
        let (batch, schema) = generate_batch::<T>(NUM_ROWS, None, range, true);
        let vectors = batch.column_by_name("vector").unwrap().clone();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let dataset = Dataset::write(batches, test_uri, None).await.unwrap();
        (dataset, Arc::new(vectors.as_list::<i32>().clone()))
    }

    async fn append_dataset<T: ArrowPrimitiveType>(
        dataset: &mut Dataset,
        num_rows: usize,
        range: Range<T::Native>,
    ) -> ArrayRef
    where
        T::Native: SampleUniform,
    {
        let is_multivector = matches!(
            dataset.schema().field("vector").unwrap().data_type(),
            DataType::List(_)
        );
        let row_count = dataset.count_all_rows().await.unwrap();
        let (batch, schema) =
            generate_batch::<T>(num_rows, Some(row_count as u64), range, is_multivector);
        let vectors = batch["vector"].clone();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        dataset.append(batches, None).await.unwrap();
        vectors
    }

    async fn append_identical_vectors(dataset: &mut Dataset, num_rows: usize, vector: &[f32]) {
        assert_eq!(
            vector.len(),
            DIM,
            "vector length ({}) must match DIM ({})",
            vector.len(),
            DIM
        );
        let start_id = dataset.count_all_rows().await.unwrap() as u64;
        let ids: ArrayRef = Arc::new(UInt64Array::from_iter_values(
            start_id..start_id + num_rows as u64,
        ));
        let mut values = Vec::with_capacity(num_rows * DIM);
        for _ in 0..num_rows {
            values.extend_from_slice(vector);
        }
        let vectors: ArrayRef = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32)
                .unwrap(),
        );
        let schema = Arc::new(Schema::from(dataset.schema()));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, vectors]).unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        dataset.append(batches, None).await.unwrap();
    }

    fn generate_batch<T: ArrowPrimitiveType>(
        num_rows: usize,
        start_id: Option<u64>,
        range: Range<T::Native>,
        is_multivector: bool,
    ) -> (RecordBatch, SchemaRef)
    where
        T::Native: SampleUniform,
    {
        const VECTOR_NUM_PER_ROW: usize = 3;
        let start_id = start_id.unwrap_or(0);
        let ids = Arc::new(UInt64Array::from_iter_values(
            start_id..start_id + num_rows as u64,
        ));
        let total_floats = match is_multivector {
            true => num_rows * VECTOR_NUM_PER_ROW * DIM,
            false => num_rows * DIM,
        };
        let vectors = generate_random_array_with_range::<T>(total_floats, range);
        let data_type = vectors.data_type().clone();
        let mut fields = vec![Field::new("id", DataType::UInt64, false)];
        let mut arrays: Vec<ArrayRef> = vec![ids];
        let mut fsl = FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap();
        if fsl.value_type() != DataType::UInt8 {
            fsl = normalize_fsl(&fsl).unwrap();
        }
        if is_multivector {
            let vector_field = Arc::new(Field::new(
                "item",
                DataType::FixedSizeList(Arc::new(Field::new("item", data_type, true)), DIM as i32),
                true,
            ));
            fields.push(Field::new(
                "vector",
                DataType::List(vector_field.clone()),
                true,
            ));
            let array = Arc::new(ListArray::new(
                vector_field,
                OffsetBuffer::from_lengths(std::iter::repeat_n(VECTOR_NUM_PER_ROW, num_rows)),
                Arc::new(fsl),
                None,
            ));
            arrays.push(array);
        } else {
            fields.push(Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", data_type, true)), DIM as i32),
                true,
            ));
            let array = Arc::new(fsl);
            arrays.push(array);
        }
        let schema: Arc<_> = Schema::new(fields).into();
        let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
        (batch, schema)
    }

    struct VectorIndexTestContext {
        stats_json: String,
        stats: serde_json::Value,
        index: Arc<dyn VectorIndex>,
    }

    impl VectorIndexTestContext {
        fn stats(&self) -> &serde_json::Value {
            &self.stats
        }

        fn stats_json(&self) -> &str {
            &self.stats_json
        }

        fn num_partitions(&self) -> usize {
            self.stats()["indices"][0]["num_partitions"]
                .as_u64()
                .expect("num_partitions should be present") as usize
        }

        fn ivf(&self) -> &IvfPq {
            self.index
                .as_any()
                .downcast_ref::<IvfPq>()
                .expect("expected IvfPq index")
        }
    }

    async fn load_vector_index_context(
        dataset: &Dataset,
        column: &str,
        index_name: &str,
    ) -> VectorIndexTestContext {
        let stats_json = dataset.index_statistics(index_name).await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats_json).unwrap();
        let uuid = stats["indices"][0]["uuid"]
            .as_str()
            .expect("Index uuid should be present");
        let index = dataset
            .open_vector_index(column, uuid, &NoOpMetricsCollector)
            .await
            .unwrap();

        VectorIndexTestContext {
            stats_json,
            stats,
            index,
        }
    }

    async fn verify_partition_split_after_append(
        mut dataset: Dataset,
        test_uri: &str,
        params: VectorIndexParams,
        description: &str,
        append_override: Option<Vec<f32>>,
    ) {
        const INDEX_NAME: &str = "vector_idx";

        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let initial_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        assert_eq!(
            initial_ctx.num_partitions(),
            2,
            "Expected {} initial partitions to be 2 before append, got stats: {}",
            description,
            initial_ctx.stats_json()
        );

        // Append additional data to trigger a split.
        if let Some(vector) = append_override {
            append_identical_vectors(&mut dataset, PARTITION_SPLIT_APPEND_ROWS, &vector).await;
        } else {
            append_dataset::<Float32Type>(&mut dataset, PARTITION_SPLIT_APPEND_ROWS, 0.0..0.05)
                .await;
        }

        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let final_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        assert_eq!(
            final_ctx.num_partitions(),
            3,
            "Expected partition split to increase partitions from 2 to 3 for {}, got stats: {}",
            description,
            final_ctx.stats_json()
        );
    }

    async fn load_partition_row_ids(index: &IvfPq, partition_idx: usize) -> Vec<u64> {
        index
            .storage
            .load_partition(partition_idx)
            .await
            .unwrap()
            .row_ids()
            .copied()
            .collect()
    }

    async fn delete_ids(dataset: &mut Dataset, ids: &[u64]) {
        if ids.is_empty() {
            return;
        }
        let predicate = ids
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        dataset
            .delete(&format!("id in ({})", predicate))
            .await
            .unwrap();
    }

    async fn compact_after_deletions(dataset: &mut Dataset) {
        compact_files(
            dataset,
            CompactionOptions {
                materialize_deletions_threshold: 0.0,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
    }

    #[allow(dead_code)]
    async fn ground_truth(
        dataset: &Dataset,
        column: &str,
        query: &dyn Array,
        k: usize,
        distance_type: DistanceType,
    ) -> HashSet<u64> {
        let batch = dataset
            .scan()
            .with_row_id()
            .nearest(column, query, k)
            .unwrap()
            .distance_metric(distance_type)
            .use_index(false)
            .try_into_batch()
            .await
            .unwrap();
        batch[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect()
    }

    #[allow(dead_code)]
    fn multivec_ground_truth(
        vectors: &ListArray,
        query: &dyn Array,
        k: usize,
        distance_type: DistanceType,
    ) -> Vec<(f32, u64)> {
        let query = if let Some(list_array) = query.as_list_opt::<i32>() {
            list_array.values().clone()
        } else {
            query.as_fixed_size_list().values().clone()
        };
        multivec_distance(&query, vectors, distance_type)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(i, dist)| (dist, i as u64))
            .sorted_by(|a, b| a.0.total_cmp(&b.0))
            .take(k)
            .collect()
    }

    async fn test_index(
        params: VectorIndexParams,
        nlist: usize,
        recall_requirement: f32,
        dataset: Option<(Dataset, Arc<FixedSizeListArray>)>,
    ) {
        match params.metric_type {
            DistanceType::Hamming => {
                test_index_impl::<UInt8Type>(params, nlist, recall_requirement, 0..4, dataset)
                    .await;
            }
            _ => {
                test_index_impl::<Float32Type>(
                    params.clone(),
                    nlist,
                    recall_requirement,
                    0.0..1.0,
                    dataset.clone(),
                )
                .await;

                let index_type = params.index_type();
                // *_FLAT doesn't support float16/float64
                if !(index_type == IndexType::IvfFlat
                    || (index_type == IndexType::IvfHnswFlat && params.stages.len() == 2)) // IVF_HNSW_FLAT
                    && dataset.is_none()
                // if dataset is provided, it has been created, so the data type is already determined, no need to test float64
                {
                    test_index_impl::<Float64Type>(
                        params,
                        nlist,
                        recall_requirement,
                        0.0..1.0,
                        dataset,
                    )
                    .await;
                }
            }
        }
    }

    async fn test_index_impl<T: ArrowPrimitiveType>(
        params: VectorIndexParams,
        nlist: usize,
        recall_requirement: f32,
        range: Range<T::Native>,
        dataset: Option<(Dataset, Arc<FixedSizeListArray>)>,
    ) where
        T::Native: SampleUniform,
    {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = match dataset {
            Some((dataset, vectors)) => (dataset, vectors),
            None => generate_test_dataset::<T>(test_uri, range).await,
        };

        let vector_column = "vector";
        dataset
            .create_index(&[vector_column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        test_recall::<T>(
            params.clone(),
            nlist,
            recall_requirement,
            vector_column,
            &dataset,
            vectors.clone(),
        )
        .await;

        if params.stages.len() > 1
            && matches!(params.version, IndexFileVersion::V3)
            && params.index_type() == IndexType::IvfPq
        {
            let index = dataset.load_indices().await.unwrap();
            assert_eq!(index.len(), 1);
            let index_path = dataset.indices_dir().child(index[0].uuid.to_string());
            rewrite_pq_storage(index_path).await.unwrap();
            // do the test again
            test_recall::<T>(
                params,
                nlist,
                recall_requirement,
                vector_column,
                &dataset,
                vectors.clone(),
            )
            .await;
        }
    }

    async fn test_remap(params: VectorIndexParams, nlist: usize) {
        match params.metric_type {
            DistanceType::Hamming => {
                Box::pin(test_remap_impl::<UInt8Type>(params, nlist, 0..4)).await;
            }
            _ => {
                Box::pin(test_remap_impl::<Float32Type>(params, nlist, 0.0..1.0)).await;
            }
        }
    }

    async fn test_remap_impl<T: ArrowPrimitiveType>(
        params: VectorIndexParams,
        nlist: usize,
        range: Range<T::Native>,
    ) where
        T::Native: SampleUniform,
    {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<T>(test_uri, range.clone()).await;

        let vector_column = "vector";
        dataset
            .create_index(&[vector_column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        let query = vectors.value(0);
        // delete half rows to trigger compact
        let half_rows = NUM_ROWS / 2;
        dataset
            .delete(&format!("id < {}", half_rows))
            .await
            .unwrap();
        // update the other half rows
        let update_result = UpdateBuilder::new(Arc::new(dataset))
            .update_where(&format!("id >= {} and id<{}", half_rows, half_rows + 50))
            .unwrap()
            .set("id", &format!("{}+id", NUM_ROWS))
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let mut dataset = Dataset::open(update_result.new_dataset.uri())
            .await
            .unwrap();
        let num_rows = dataset.count_rows(None).await.unwrap();
        assert_eq!(num_rows, half_rows);
        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();
        // query again, the result should not include the deleted row
        let result = dataset.scan().try_into_batch().await.unwrap();
        let ids = result["id"].as_primitive::<UInt64Type>();
        assert_eq!(ids.len(), half_rows);
        ids.values().iter().for_each(|id| {
            assert!(*id >= half_rows as u64 + 50);
        });

        // make sure we can still hit the recall
        let gt = ground_truth(&dataset, vector_column, &query, 100, params.metric_type).await;
        let results = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), 100)
            .unwrap()
            .minimum_nprobes(nlist)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        let row_ids = results[ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let recall = row_ids.intersection(&gt).count() as f32 / 100.0;
        assert_ge!(recall, 0.7, "{}", recall);

        // delete so that only one row left, to trigger remap and there must be some empty partitions
        let (mut dataset, _) = generate_test_dataset::<T>(test_uri, range).await;
        dataset
            .create_index(&[vector_column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
        assert_eq!(dataset.load_indices().await.unwrap().len(), 1);
        dataset.delete("id > 0").await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 1);
        assert_eq!(dataset.load_indices().await.unwrap().len(), 1);
        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();
        let results = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), 100)
            .unwrap()
            .minimum_nprobes(nlist)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 1);
    }

    async fn test_delete_all_rows(params: VectorIndexParams) {
        match params.metric_type {
            DistanceType::Hamming => {
                test_delete_all_rows_impl::<UInt8Type>(params, 0..4).await;
            }
            _ => {
                test_delete_all_rows_impl::<Float32Type>(params, 0.0..1.0).await;
            }
        }
    }

    async fn test_delete_all_rows_impl<T: ArrowPrimitiveType>(
        params: VectorIndexParams,
        range: Range<T::Native>,
    ) where
        T::Native: SampleUniform,
    {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<T>(test_uri, range.clone()).await;

        let vector_column = "vector";
        dataset
            .create_index(&[vector_column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        dataset.delete("id >= 0").await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 0);

        // optimize after delete all rows
        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        let query = vectors.value(0);
        let results = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), 100)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 0);

        // compact after delete all rows
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, _) = generate_test_dataset::<T>(test_uri, range).await;

        let vector_column = "vector";
        dataset
            .create_index(&[vector_column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        dataset.delete("id >= 0").await.unwrap();
        assert_eq!(dataset.count_rows(None).await.unwrap(), 0);

        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();

        let results = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), 100)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 0);
    }

    #[tokio::test]
    async fn test_flat_knn() {
        test_distance_range(None, 4).await;
    }

    #[rstest]
    #[case(4, DistanceType::L2, 1.0)]
    #[case(4, DistanceType::Cosine, 1.0)]
    #[case(4, DistanceType::Dot, 1.0)]
    #[case(4, DistanceType::Hamming, 0.9)]
    #[tokio::test]
    async fn test_build_ivf_flat(
        #[case] nlist: usize,
        #[case] distance_type: DistanceType,
        #[case] recall_requirement: f32,
    ) {
        let params = VectorIndexParams::ivf_flat(nlist, distance_type);
        test_index(params.clone(), nlist, recall_requirement, None).await;
        if distance_type == DistanceType::Cosine {
            test_index_multivec(params.clone(), nlist, recall_requirement).await;
        }
        test_distance_range(Some(params.clone()), nlist).await;
        test_remap(params.clone(), nlist).await;
        test_delete_all_rows(params).await;
    }

    #[rstest]
    #[case(4, DistanceType::L2, 0.9)]
    #[case(4, DistanceType::Cosine, 0.9)]
    #[case(4, DistanceType::Dot, 0.85)]
    #[tokio::test]
    async fn test_build_ivf_pq(
        #[case] nlist: usize,
        #[case] distance_type: DistanceType,
        #[case] recall_requirement: f32,
    ) {
        let ivf_params = IvfBuildParams::new(nlist);
        let pq_params = PQBuildParams::default();
        let params = VectorIndexParams::with_ivf_pq_params(distance_type, ivf_params, pq_params)
            .version(crate::index::vector::IndexFileVersion::Legacy)
            .clone();
        test_index(params.clone(), nlist, recall_requirement, None).await;
        if distance_type == DistanceType::Cosine {
            test_index_multivec(params.clone(), nlist, recall_requirement).await;
        }
        test_distance_range(Some(params.clone()), nlist).await;
        test_remap(params, nlist).await;
    }

    #[rstest]
    #[case(1, DistanceType::L2, 0.9)]
    #[case(1, DistanceType::Cosine, 0.9)]
    #[case(1, DistanceType::Dot, 0.85)]
    #[case(4, DistanceType::L2, 0.9)]
    #[case(4, DistanceType::Cosine, 0.9)]
    #[case(4, DistanceType::Dot, 0.85)]
    #[tokio::test]
    async fn test_build_ivf_pq_v3(
        #[case] nlist: usize,
        #[case] distance_type: DistanceType,
        #[case] recall_requirement: f32,
    ) {
        let ivf_params = IvfBuildParams::new(nlist);
        let pq_params = PQBuildParams::default();
        let params = VectorIndexParams::with_ivf_pq_params(distance_type, ivf_params, pq_params);
        test_index(params.clone(), nlist, recall_requirement, None).await;
        if distance_type == DistanceType::Cosine {
            test_index_multivec(params.clone(), nlist, recall_requirement).await;
        }
        test_distance_range(Some(params.clone()), nlist).await;
        test_remap(params.clone(), nlist).await;
        test_delete_all_rows(params).await;
    }

    #[rstest]
    #[case(4, DistanceType::L2, 0.85)]
    #[case(4, DistanceType::Cosine, 0.85)]
    #[case(4, DistanceType::Dot, 0.75)]
    #[tokio::test]
    async fn test_build_ivf_pq_4bit(
        #[case] nlist: usize,
        #[case] distance_type: DistanceType,
        #[case] recall_requirement: f32,
    ) {
        let ivf_params = IvfBuildParams::new(nlist);
        let pq_params = PQBuildParams::new(32, 4);
        let params = VectorIndexParams::with_ivf_pq_params(distance_type, ivf_params, pq_params);
        test_index(params.clone(), nlist, recall_requirement, None).await;
        if distance_type == DistanceType::Cosine {
            test_index_multivec(params.clone(), nlist, recall_requirement).await;
        }
        test_remap(params, nlist).await;
    }

    #[rstest]
    #[case(4, DistanceType::L2, 0.85)]
    #[case(4, DistanceType::Cosine, 0.85)]
    #[case(4, DistanceType::Dot, 0.75)]
    #[tokio::test]
    async fn test_build_ivf_sq(
        #[case] nlist: usize,
        #[case] distance_type: DistanceType,
        #[case] recall_requirement: f32,
    ) {
        let ivf_params = IvfBuildParams::new(nlist);
        let sq_params = SQBuildParams::default();
        let params = VectorIndexParams::with_ivf_sq_params(distance_type, ivf_params, sq_params);
        test_index(params.clone(), nlist, recall_requirement, None).await;
        if distance_type == DistanceType::Cosine {
            test_index_multivec(params.clone(), nlist, recall_requirement).await;
        }
        test_remap(params, nlist).await;
    }

    // RQ doesn't perform well for random data
    // need to verify recall with real-world dataset (e.g. sift1m)
    #[rstest]
    #[case(1, DistanceType::L2, 0.5)]
    #[case(1, DistanceType::Cosine, 0.5)]
    #[case(1, DistanceType::Dot, 0.5)]
    #[case(4, DistanceType::L2, 0.5)]
    #[case(4, DistanceType::Cosine, 0.5)]
    #[case(4, DistanceType::Dot, 0.5)]
    #[tokio::test]
    // #[ignore = "Temporarily skipping flaky 4-bit IVF_RQ tests"]
    async fn test_build_ivf_rq(
        #[case] nlist: usize,
        #[case] distance_type: DistanceType,
        #[case] recall_requirement: f32,
    ) {
        let _ = env_logger::try_init();
        let ivf_params = IvfBuildParams::new(nlist);
        let rq_params = RQBuildParams::new(1);
        let params = VectorIndexParams::with_ivf_rq_params(distance_type, ivf_params, rq_params);
        test_index(params.clone(), nlist, recall_requirement, None).await;
        if distance_type == DistanceType::Cosine {
            test_index_multivec(params.clone(), nlist, recall_requirement).await;
        }
        test_remap(params.clone(), nlist).await;
    }

    #[rstest]
    #[case(4, DistanceType::L2, 0.9)]
    #[case(4, DistanceType::Cosine, 0.9)]
    #[case(4, DistanceType::Dot, 0.85)]
    #[tokio::test]
    async fn test_create_ivf_hnsw_flat(
        #[case] nlist: usize,
        #[case] distance_type: DistanceType,
        #[case] recall_requirement: f32,
    ) {
        let ivf_params = IvfBuildParams::new(nlist);
        let hnsw_params = HnswBuildParams::default();
        let params = VectorIndexParams::ivf_hnsw(distance_type, ivf_params, hnsw_params);
        test_index(params.clone(), nlist, recall_requirement, None).await;
        if distance_type == DistanceType::Cosine {
            test_index_multivec(params.clone(), nlist, recall_requirement).await;
        }
        test_remap(params, nlist).await;
    }

    #[rstest]
    #[case(4, DistanceType::L2, 0.9)]
    #[case(4, DistanceType::Cosine, 0.9)]
    #[case(4, DistanceType::Dot, 0.85)]
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
        test_index(params.clone(), nlist, recall_requirement, None).await;
        if distance_type == DistanceType::Cosine {
            test_index_multivec(params.clone(), nlist, recall_requirement).await;
        }
        test_distance_range(Some(params.clone()), nlist).await;
        test_delete_all_rows(params.clone()).await;
        test_remap(params, nlist).await;
    }

    #[rstest]
    #[case(4, DistanceType::L2, 0.9)]
    #[case(4, DistanceType::Cosine, 0.9)]
    #[case(4, DistanceType::Dot, 0.85)]
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
        test_index(params.clone(), nlist, recall_requirement, None).await;
        if distance_type == DistanceType::Cosine {
            test_index_multivec(params.clone(), nlist, recall_requirement).await;
        }
        test_remap(params, nlist).await;
    }

    #[rstest]
    #[case(4, DistanceType::L2, 0.85)]
    #[case(4, DistanceType::Cosine, 0.85)]
    #[case(4, DistanceType::Dot, 0.8)]
    #[tokio::test]
    async fn test_create_ivf_hnsw_pq_4bit(
        #[case] nlist: usize,
        #[case] distance_type: DistanceType,
        #[case] recall_requirement: f32,
    ) {
        let ivf_params = IvfBuildParams::new(nlist);
        let pq_params = PQBuildParams::new(32, 4);
        let hnsw_params = HnswBuildParams::default();
        let params = VectorIndexParams::with_ivf_hnsw_pq_params(
            distance_type,
            ivf_params,
            hnsw_params,
            pq_params,
        );
        test_index(params.clone(), nlist, recall_requirement, None).await;
        if distance_type == DistanceType::Cosine {
            test_index_multivec(params, nlist, recall_requirement).await;
        }
    }

    async fn test_index_multivec(params: VectorIndexParams, nlist: usize, recall_requirement: f32) {
        // we introduce XTR for performance, which would reduce the recall a little bit
        let recall_requirement = recall_requirement * 0.9;
        match params.metric_type {
            DistanceType::Hamming => {
                test_index_multivec_impl::<UInt8Type>(params, nlist, recall_requirement, 0..4)
                    .await;
            }
            _ => {
                test_index_multivec_impl::<Float32Type>(
                    params,
                    nlist,
                    recall_requirement,
                    0.0..1.0,
                )
                .await;
            }
        }
    }

    async fn test_index_multivec_impl<T: ArrowPrimitiveType>(
        params: VectorIndexParams,
        nlist: usize,
        recall_requirement: f32,
        range: Range<T::Native>,
    ) where
        T::Native: SampleUniform,
    {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (mut dataset, vectors) = generate_multivec_test_dataset::<T>(test_uri, range).await;

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

        let query = vectors.value(0);
        let k = 100;

        let result = dataset
            .scan()
            .nearest("vector", &query, k)
            .unwrap()
            .minimum_nprobes(nlist)
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
            .zip(row_ids.clone().into_iter())
            .collect::<Vec<_>>();
        let row_ids = row_ids.into_iter().collect::<HashSet<_>>();

        let gt = multivec_ground_truth(&vectors, &query, k, params.metric_type);
        let gt_set = gt.iter().map(|r| r.1).collect::<HashSet<_>>();

        let recall = row_ids.intersection(&gt_set).count() as f32 / 100.0;
        assert!(
            recall >= recall_requirement,
            "recall: {}\n results: {:?}\n\ngt: {:?}",
            recall,
            results,
            gt
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_migrate_v1_to_v3() {
        // only test the case of IVF_PQ
        // because only IVF_PQ is supported in v1
        let nlist = 4;
        let recall_requirement = 0.9;
        let ivf_params = IvfBuildParams::new(nlist);
        let pq_params = PQBuildParams::default();
        let v1_params =
            VectorIndexParams::with_ivf_pq_params(DistanceType::Cosine, ivf_params, pq_params)
                .version(crate::index::vector::IndexFileVersion::Legacy)
                .clone();

        let v3_params = v1_params
            .clone()
            .version(crate::index::vector::IndexFileVersion::V3)
            .clone();

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;
        test_index(
            v1_params,
            nlist,
            recall_requirement,
            Some((dataset.clone(), vectors.clone())),
        )
        .await;
        // retest with v3 params on the same dataset
        test_index(
            v3_params,
            nlist,
            recall_requirement,
            Some((dataset.clone(), vectors)),
        )
        .await;

        dataset.checkout_latest().await.unwrap();
        let indices = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(indices.len(), 1); // v1 index should be replaced by v3 index
        let index = dataset
            .open_vector_index(
                "vector",
                indices[0].uuid.to_string().as_str(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let v3_index = index.as_any().downcast_ref::<super::IvfPq>();
        assert!(v3_index.is_some());
    }

    #[rstest]
    #[tokio::test]
    async fn test_index_stats(
        #[values(
            (VectorIndexParams::ivf_flat(4, DistanceType::Hamming), IndexType::IvfFlat),
            (VectorIndexParams::ivf_pq(4, 8, 8, DistanceType::L2, 10), IndexType::IvfPq),
            (VectorIndexParams::with_ivf_hnsw_sq_params(
                DistanceType::Cosine,
                IvfBuildParams::new(4),
                Default::default(),
                Default::default()
            ), IndexType::IvfHnswSq),
        )]
        index: (VectorIndexParams, IndexType),
    ) {
        let (params, index_type) = index;
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let nlist = 4;
        let (mut dataset, _) = match params.metric_type {
            DistanceType::Hamming => generate_test_dataset::<UInt8Type>(test_uri, 0..2).await,
            _ => generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await,
        };
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

        assert_eq!(
            stats["index_type"].as_str().unwrap(),
            index_type.to_string()
        );
        for index in stats["indices"].as_array().unwrap() {
            assert_eq!(
                index["index_type"].as_str().unwrap(),
                index_type.to_string()
            );
            assert_eq!(
                index["num_partitions"].as_number().unwrap(),
                &serde_json::Number::from(nlist)
            );

            let sub_index = match index_type {
                IndexType::IvfHnswPq | IndexType::IvfHnswSq => "HNSW",
                IndexType::IvfPq => "PQ",
                _ => "FLAT",
            };
            assert_eq!(
                index["sub_index"]["index_type"].as_str().unwrap(),
                sub_index
            );
        }
    }

    #[tokio::test]
    async fn test_index_stats_empty_partition() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let nlist = 500;
        let (mut dataset, _) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

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

    async fn test_distance_range(params: Option<VectorIndexParams>, nlist: usize) {
        match params.as_ref().map_or(DistanceType::L2, |p| p.metric_type) {
            DistanceType::Hamming => {
                test_distance_range_impl::<UInt8Type>(params, nlist, 0..255).await;
            }
            _ => {
                test_distance_range_impl::<Float32Type>(params, nlist, 0.0..1.0).await;
            }
        }
    }

    async fn test_distance_range_impl<T: ArrowPrimitiveType>(
        params: Option<VectorIndexParams>,
        nlist: usize,
        range: Range<T::Native>,
    ) where
        T::Native: SampleUniform,
    {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, vectors) = generate_test_dataset::<T>(test_uri, range).await;

        let vector_column = "vector";
        let dist_type = params.as_ref().map_or(DistanceType::L2, |p| p.metric_type);
        if let Some(params) = params {
            dataset
                .create_index(&[vector_column], IndexType::Vector, None, &params, true)
                .await
                .unwrap();
        }

        let query = vectors.value(0);
        let k = 10;
        let result = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), k)
            .unwrap()
            .minimum_nprobes(nlist)
            .ef(100)
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(result.num_rows(), k);
        let row_ids = result[ROW_ID].as_primitive::<UInt64Type>().values();
        let dists = result[DIST_COL].as_primitive::<Float32Type>().values();

        let part_idx = k / 2;
        let part_dist = dists[part_idx];

        let left_res = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), part_idx)
            .unwrap()
            .minimum_nprobes(nlist)
            .ef(100)
            .with_row_id()
            .distance_range(None, Some(part_dist))
            .try_into_batch()
            .await
            .unwrap();
        let right_res = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), k - part_idx)
            .unwrap()
            .minimum_nprobes(nlist)
            .ef(100)
            .with_row_id()
            .distance_range(Some(part_dist), None)
            .try_into_batch()
            .await
            .unwrap();
        // don't verify the number of results and row ids for hamming distance,
        // because there are many vectors with the same distance
        if dist_type != DistanceType::Hamming {
            assert_eq!(left_res.num_rows(), part_idx);
            assert_eq!(right_res.num_rows(), k - part_idx);
            let left_row_ids = left_res[ROW_ID].as_primitive::<UInt64Type>().values();
            let right_row_ids = right_res[ROW_ID].as_primitive::<UInt64Type>().values();
            row_ids.iter().enumerate().for_each(|(i, id)| {
                if i < part_idx {
                    assert_eq!(left_row_ids[i], *id,);
                } else {
                    assert_eq!(right_row_ids[i - part_idx], *id,);
                }
            });
        }
        let left_dists = left_res[DIST_COL].as_primitive::<Float32Type>().values();
        let right_dists = right_res[DIST_COL].as_primitive::<Float32Type>().values();
        left_dists.iter().for_each(|d| {
            assert!(d < &part_dist);
        });
        right_dists.iter().for_each(|d| {
            assert!(d >= &part_dist);
        });

        let exclude_last_res = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), k)
            .unwrap()
            .minimum_nprobes(nlist)
            .ef(100)
            .with_row_id()
            .distance_range(dists.first().copied(), dists.last().copied())
            .try_into_batch()
            .await
            .unwrap();
        if dist_type != DistanceType::Hamming {
            let excluded_count = dists.iter().filter(|d| *d == dists.last().unwrap()).count();
            assert_eq!(exclude_last_res.num_rows(), k - excluded_count);
            let res_row_ids = exclude_last_res[ROW_ID]
                .as_primitive::<UInt64Type>()
                .values();
            row_ids.iter().enumerate().for_each(|(i, id)| {
                if i < k - excluded_count {
                    assert_eq!(res_row_ids[i], *id);
                }
            });
        }
        let res_dists = exclude_last_res[DIST_COL]
            .as_primitive::<Float32Type>()
            .values();
        res_dists.iter().for_each(|d| {
            assert_ge!(*d, dists[0]);
            assert_lt!(*d, dists[k - 1]);
        });
    }

    #[tokio::test]
    async fn test_index_with_zero_vectors() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (batch, schema) = generate_batch::<Float32Type>(256, None, 0.0..1.0, false);
        let vector_field = schema.field(1).clone();
        let zero_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![256])),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        Float32Array::from(vec![0.0; DIM]),
                        DIM as i32,
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![batch, zero_batch].into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                mode: crate::dataset::WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let vector_column = vector_field.name();
        let params = VectorIndexParams::ivf_pq(4, 8, DIM / 8, DistanceType::Cosine, 50);
        dataset
            .create_index(&[vector_column], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
    }

    async fn test_recall<T: ArrowPrimitiveType>(
        params: VectorIndexParams,
        nlist: usize,
        recall_requirement: f32,
        vector_column: &str,
        dataset: &Dataset,
        vectors: Arc<FixedSizeListArray>,
    ) {
        let query = vectors.value(0);
        let k = 100;
        let result = dataset
            .scan()
            .nearest(vector_column, query.as_primitive::<T>(), k)
            .unwrap()
            .nprobes(nlist)
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
        assert!(row_ids.len() == k);

        let gt = ground_truth(dataset, vector_column, &query, k, params.metric_type).await;

        let recall = row_ids.intersection(&gt).count() as f32 / k as f32;
        assert!(
            recall >= recall_requirement,
            "recall: {}\n results: {:?}\n\ngt: {:?}",
            recall,
            results,
            gt,
        );
    }

    async fn rewrite_pq_storage(dir: Path) -> Result<()> {
        let obj_store = Arc::new(ObjectStore::local());
        let store_path = dir.child(INDEX_AUXILIARY_FILE_NAME);
        let copied_path = dir.child(format!("{}.original", INDEX_AUXILIARY_FILE_NAME));
        obj_store.copy(&store_path, &copied_path).await?;
        obj_store.delete(&store_path).await?;
        let scheduler =
            ScanScheduler::new(obj_store.clone(), SchedulerConfig::default_for_testing());
        let reader = FileReader::try_open(
            scheduler
                .open_file(&copied_path, &CachedFileSize::unknown())
                .await?,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await?;

        let mut metadata = reader.schema().metadata.clone();
        let batch = reader
            .read_range(0..reader.num_rows() as usize, None)
            .await?;
        let mut writer = FileWriter::try_new(
            obj_store.create(&store_path).await?,
            batch.schema_ref().as_ref().try_into()?,
            Default::default(),
        )?;
        writer.write_batch(&batch).await?;
        // write the IVF
        writer
            .add_global_buffer(reader.read_global_buffer(1).await?)
            .await?;
        // rewrite the PQ to legacy format
        let codebook = reader.read_global_buffer(2).await?;
        let pq_metadata: Vec<String> = serde_json::from_str(&metadata[STORAGE_METADATA_KEY])?;
        let mut pq_metadata: ProductQuantizationMetadata = serde_json::from_str(&pq_metadata[0])?;
        pq_metadata.codebook_position = 0;
        pq_metadata.codebook_tensor = codebook.to_vec();
        let pq_metadata = serde_json::to_string(&pq_metadata)?;
        metadata.insert(
            STORAGE_METADATA_KEY.to_owned(),
            serde_json::to_string(&vec![pq_metadata])?,
        );
        writer.finish_with_metadata(metadata).await?;
        obj_store.delete(&copied_path).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_pq_storage_backwards_compat() {
        let test_dir = copy_test_data_to_tmp("v0.27.1/pq_in_schema").unwrap();
        let test_uri = test_dir.path_str();
        let test_uri = &test_uri;

        // Just make sure we can query the index.
        let dataset = Dataset::open(test_uri).await.unwrap();
        let query_vec = Float32Array::from(vec![0_f32; 32]);
        let search_result = dataset
            .scan()
            .nearest("vec", &query_vec, 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(search_result.num_rows(), 5);

        let obj_store = Arc::new(ObjectStore::local());
        let scheduler =
            ScanScheduler::new(obj_store.clone(), SchedulerConfig::default_for_testing());

        async fn get_pq_metadata(
            dataset: &Dataset,
            scheduler: Arc<ScanScheduler>,
        ) -> ProductQuantizationMetadata {
            let index = dataset.load_indices().await.unwrap();
            let index_path = dataset.indices_dir().child(index[0].uuid.to_string());
            let file_scheduler = scheduler
                .open_file(
                    &index_path.child(INDEX_AUXILIARY_FILE_NAME),
                    &CachedFileSize::unknown(),
                )
                .await
                .unwrap();
            let reader = FileReader::try_open(
                file_scheduler,
                None,
                Arc::<DecoderPlugins>::default(),
                &LanceCache::no_cache(),
                FileReaderOptions::default(),
            )
            .await
            .unwrap();
            let metadata = reader.schema().metadata.get(STORAGE_METADATA_KEY).unwrap();
            serde_json::from_str(&serde_json::from_str::<Vec<String>>(metadata).unwrap()[0])
                .unwrap()
        }
        let pq_meta: ProductQuantizationMetadata =
            get_pq_metadata(&dataset, scheduler.clone()).await;
        assert!(pq_meta.buffer_index().is_none());

        // If we add data and optimize indices, then we start using the global
        // buffer for the PQ index.
        let new_data = RecordBatch::try_new(
            Arc::new(Schema::from(dataset.schema())),
            vec![
                Arc::new(Int64Array::from(vec![0])),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; 32]), 32)
                        .unwrap(),
                ),
            ],
        )
        .unwrap();
        let mut dataset = InsertBuilder::new(Arc::new(dataset))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute(vec![new_data])
            .await
            .unwrap();
        dataset
            .optimize_indices(&OptimizeOptions::merge(1))
            .await
            .unwrap();

        let pq_meta: ProductQuantizationMetadata =
            get_pq_metadata(&dataset, scheduler.clone()).await;
        assert!(pq_meta.buffer_index().is_some());
    }

    #[tokio::test]
    async fn test_optimize_with_empty_partition() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, _) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        let num_rows = dataset.count_all_rows().await.unwrap();
        let nlist = num_rows + 2;
        let centroids = generate_random_array(nlist * DIM);
        let ivf_centroids = FixedSizeListArray::try_new_from_values(centroids, DIM as i32).unwrap();
        let ivf_params =
            IvfBuildParams::try_with_centroids(nlist, Arc::new(ivf_centroids)).unwrap();
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::Cosine,
            ivf_params,
            PQBuildParams::default(),
        );
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        append_dataset::<Float32Type>(&mut dataset, 1, 0.0..1.0).await;
        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_create_index_with_many_invalid_vectors() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        // we use 8192 batch size by default, so we need to generate 8192 * 3 vectors to get 3 batches
        // generate 3 batches, and the first batch's vectors are all with NaN
        let num_rows = 8192 * 3;
        let mut vectors = Vec::new();
        for i in 0..num_rows {
            if i < 8192 {
                vectors.extend(std::iter::repeat_n(f32::NAN, DIM));
            } else if i < 8192 * 2 {
                vectors.extend(std::iter::repeat_n(rand::random::<f32>(), DIM));
            } else {
                vectors.extend(std::iter::repeat_n(rand::random::<f32>() * 1e20, DIM));
            }
        }
        let schema = Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            true,
        )]);
        let schema = Arc::new(schema);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(
                FixedSizeListArray::try_new_from_values(Float32Array::from(vectors), DIM as i32)
                    .unwrap(),
            )],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
        let params = WriteParams {
            mode: WriteMode::Overwrite,
            ..Default::default()
        };
        let mut dataset = Dataset::write(batches, test_uri, Some(params))
            .await
            .unwrap();

        let params = VectorIndexParams::ivf_pq(4, 8, DIM / 8, DistanceType::Dot, 50);

        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_partition_split_on_append() {
        // This test verifies that when we append enough data to a partition
        // such that it exceeds MAX_PARTITION_SIZE_FACTOR * target_partition_size,
        // the partition will be split into 2 partitions.

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        // Build deterministic two-cluster data so centroids are predictable.
        const ROWS_PER_CLUSTER: usize = 2_048;
        let cluster_vectors = vec![
            {
                let mut v = vec![0.0; DIM];
                v[0] = 1.0;
                v
            },
            {
                let mut v = vec![0.0; DIM];
                v[1] = 1.0;
                v
            },
        ];
        let total_rows = ROWS_PER_CLUSTER * cluster_vectors.len();
        let mut ids = Vec::with_capacity(total_rows);
        let mut vector_values = Vec::with_capacity(total_rows * DIM);
        let mut current_id = 0u64;
        for cluster_vector in &cluster_vectors {
            for _ in 0..ROWS_PER_CLUSTER {
                ids.push(current_id);
                current_id += 1;
                vector_values.extend_from_slice(cluster_vector);
            }
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DIM as i32,
                ),
                false,
            ),
        ]));
        let ids_array: ArrayRef = Arc::new(UInt64Array::from(ids));
        let vectors_array: ArrayRef = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(vector_values), DIM as i32)
                .unwrap(),
        );
        let batch = RecordBatch::try_new(schema.clone(), vec![ids_array, vectors_array]).unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                mode: crate::dataset::WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Create an IVF-PQ index with 2 partitions
        // For IvfPq, target_partition_size = 8192
        // Split triggers when partition_size > 4 * 8192 = 32,768
        let params = VectorIndexParams::ivf_pq(2, 8, DIM / 8, DistanceType::L2, 50);
        verify_partition_split_after_append(
            dataset,
            test_uri,
            params,
            "scalar vector data",
            Some(cluster_vectors[0].clone()),
        )
        .await;
    }

    #[tokio::test]
    async fn test_join_partition_on_delete() {
        // This test verifies that partition join works correctly when partitions become
        // too small after deletions.
        //
        // The test uses deterministic data with predefined centroids to ensure:
        // 1. Predictable partition sizes
        // 2. Reliable triggering of partition join when size < threshold
        // 3. Correct handling of small partitions
        //
        // Join threshold: MIN_PARTITION_SIZE_PERCENT * target_partition_size / 100
        //               = 25 * 8192 / 100 = 2048 rows

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        // Create deterministic test data with 3 clusters
        // Cluster 0: 100 rows (will be deleted to trigger join)
        // Cluster 1: 3000 rows
        // Cluster 2: 3000 rows
        let nlist = 3;
        let cluster_sizes = [100, 3000, 3000];
        let total_rows: usize = cluster_sizes.iter().sum();

        // Generate 3 well-separated centroids in DIM-dimensional space
        let mut centroid_values = Vec::new();
        for i in 0..nlist {
            for j in 0..DIM {
                // Place centroids far apart to ensure clear cluster separation
                centroid_values.push(if j == 0 {
                    (i as f32) * 10.0 // Separate along first dimension
                } else {
                    0.0
                });
            }
        }
        let centroids = Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(centroid_values),
                DIM as i32,
            )
            .unwrap(),
        );

        // Generate vectors clustered around each centroid
        let mut ids = Vec::new();
        let mut vector_values = Vec::new();
        let mut current_id = 0u64;

        for (cluster_idx, &size) in cluster_sizes.iter().enumerate() {
            let centroid_base = (cluster_idx as f32) * 10.0;
            for _ in 0..size {
                ids.push(current_id);
                current_id += 1;

                // Generate vector close to centroid (within 0.5 radius)
                for j in 0..DIM {
                    vector_values.push(if j == 0 {
                        centroid_base + (current_id % 100) as f32 * 0.005 // Small variation
                    } else {
                        (current_id % 50) as f32 * 0.01
                    });
                }
            }
        }

        let ids_array = Arc::new(UInt64Array::from(ids.clone()));
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(vector_values), DIM as i32)
                .unwrap(),
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("vector", vectors.data_type().clone(), false),
        ]));

        let batch = RecordBatch::try_new(schema.clone(), vec![ids_array, vectors]).unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                mode: crate::dataset::WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Create IVF-PQ index with predefined centroids
        let ivf_params = IvfBuildParams::try_with_centroids(nlist, centroids).unwrap();
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            ivf_params,
            PQBuildParams::default(),
        );

        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Verify initial partition count
        let index_ctx = load_vector_index_context(&dataset, "vector", "vector_idx").await;
        assert_eq!(index_ctx.num_partitions(), nlist);

        // Verify partition sizes match expected (approximately, due to potential edge cases)
        let partitions = index_ctx.stats()["indices"][0]["partitions"]
            .as_array()
            .unwrap();
        let partition_0_size = partitions[0]["size"].as_u64().unwrap();
        assert!(
            (50..=150).contains(&partition_0_size),
            "Partition 0 should have ~100 rows, got {}",
            partition_0_size
        );

        // Delete most rows from partition 0, keeping only 1 row
        // This should bring it well below the 2048 threshold
        let row_ids = load_partition_row_ids(index_ctx.ivf(), 0).await;

        assert!(!row_ids.is_empty(), "Partition 0 should not be empty");

        let res = dataset
            .take_rows(&row_ids, dataset.schema().clone())
            .await
            .unwrap();
        let ids = res["id"].as_primitive::<UInt64Type>().values();
        let retained_id = ids[0]; // Save the ID of the row we're keeping
        let first_vector = res["vector"].as_fixed_size_list().value(0);

        // Delete all but the first row in partition 0
        delete_ids(&mut dataset, &ids[1..]).await;

        // Compact to trigger partition join
        compact_after_deletions(&mut dataset).await;

        // Verify partition was joined (should have nlist-1 partitions now)
        let final_ctx = load_vector_index_context(&dataset, "vector", "vector_idx").await;
        let final_num_partitions = final_ctx.num_partitions();
        assert_eq!(
            final_num_partitions,
            nlist - 1,
            "Expected partition join to decrease partitions from {} to {}, got stats: {}",
            nlist,
            nlist - 1,
            final_ctx.stats_json()
        );

        // Verify that vector search still works after partition join
        let result = dataset
            .scan()
            .nearest("vector", &first_vector, 10)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert!(
            result.num_rows() > 0,
            "Search should return results after partition join"
        );

        // Verify the retained row still exists in the dataset by filtering by ID
        let retained_row_result = dataset
            .scan()
            .filter(&format!("id = {}", retained_id))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            retained_row_result.num_rows(),
            1,
            "The retained row (id={}) should still exist in the dataset",
            retained_id
        );
        assert_eq!(
            retained_row_result["id"]
                .as_primitive::<UInt64Type>()
                .value(0),
            retained_id,
            "The filtered result should match the retained ID"
        );

        // Verify total row count
        let remaining_rows = dataset.count_all_rows().await.unwrap();
        let expected_rows = total_rows - cluster_sizes[0] + 1; // Deleted all but 1 from cluster 0
        assert_eq!(
            remaining_rows, expected_rows,
            "Should have {} rows remaining",
            expected_rows
        );

        // Verify the index is still functional by searching for different vectors
        for cluster_idx in 1..nlist {
            let test_vector_values: Vec<f32> = (0..DIM)
                .map(|j| {
                    if j == 0 {
                        (cluster_idx as f32) * 10.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let test_vector = Float32Array::from(test_vector_values);

            let result = dataset
                .scan()
                .nearest("vector", &test_vector, 5)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            assert!(
                result.num_rows() > 0,
                "Search for cluster {} should return results",
                cluster_idx
            );
        }
    }

    #[tokio::test]
    async fn test_partition_split_on_append_multivec() {
        // This test verifies that when we append enough multivector data to a partition
        // such that it exceeds MAX_PARTITION_SIZE_FACTOR * target_partition_size,
        // the partition will be split into 2 partitions.

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        // Create initial dataset with multivector data
        let (dataset, _) = generate_multivec_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        // Create an IVF-PQ index with 2 partitions
        // For IvfPq, target_partition_size = 8192
        // Split triggers when partition_size > 4 * 8192 = 32,768
        let params = VectorIndexParams::ivf_pq(2, 8, DIM / 8, DistanceType::Cosine, 50);
        verify_partition_split_after_append(dataset, test_uri, params, "multivector data", None)
            .await;
    }

    #[tokio::test]
    async fn test_join_partition_on_delete_multivec() {
        // This test verifies that IVF index with multivector data handles deletions
        // and compaction correctly, and that partition join works when applicable.
        //
        // Due to the complexity of multivector partition assignment, we use a more
        // flexible verification approach that doesn't require specific partition sizes.

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let num_rows = 5_000;
        let mut dataset = {
            let (batch, schema) = generate_batch::<Float32Type>(num_rows, None, 0.0..1.0, true);
            let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
            let dataset = Dataset::write(
                batches,
                test_uri,
                Some(WriteParams {
                    mode: crate::dataset::WriteMode::Overwrite,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
            dataset
        };

        // Create an IVF_PQ index with 10 partitions
        // More partitions increase likelihood of having small partitions
        let nlist = 10;
        let params = VectorIndexParams::ivf_pq(nlist, 8, DIM / 8, DistanceType::Cosine, 50);
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Verify initial partition count
        let index_ctx = load_vector_index_context(&dataset, "vector", "vector_idx").await;
        assert_eq!(index_ctx.num_partitions(), nlist);

        // Find the smallest partition and delete most of its rows
        let row_ids = {
            let ivf = index_ctx.ivf();
            let mut smallest: Option<Vec<u64>> = None;
            for i in 0..ivf.ivf.num_partitions() {
                let partition_row_ids = load_partition_row_ids(ivf, i).await;
                if partition_row_ids.is_empty() {
                    continue;
                }

                let is_better = smallest
                    .as_ref()
                    .map(|existing| partition_row_ids.len() < existing.len())
                    .unwrap_or(true);
                if is_better {
                    smallest = Some(partition_row_ids);
                }
            }
            smallest.unwrap_or_default()
        };

        if row_ids.is_empty() {
            // All partitions might be large - just verify basic functionality
            let (batch, _) = generate_batch::<Float32Type>(1, None, 0.0..1.0, true);
            let test_vector = batch["vector"].as_list::<i32>().value(0);
            let result = dataset
                .scan()
                .nearest("vector", &test_vector, 5)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            assert!(result.num_rows() > 0, "Multivector search should work");
            return;
        }

        // Keep only a few rows to make partition small
        let keep_count = 5.min(row_ids.len());
        let retained_ids: Vec<u64> = row_ids.iter().take(keep_count).copied().collect();

        // Delete all rows except the first keep_count rows
        delete_ids(&mut dataset, &row_ids[keep_count..]).await;

        // Compact to potentially trigger partition join
        compact_after_deletions(&mut dataset).await;

        // Verify partition count (may or may not have joined depending on sizes)
        let final_ctx = load_vector_index_context(&dataset, "vector", "vector_idx").await;
        let final_num_partitions = final_ctx.num_partitions();
        assert!(
            final_num_partitions <= nlist,
            "Partition count should not increase after deletions, was {}, now {}",
            nlist,
            final_num_partitions
        );

        // Verify that multivector search still works after compaction
        // Get a sample row by scanning and filtering
        let sample_id = retained_ids[0];
        let sample_row = dataset
            .scan()
            .filter(&format!("id = {}", sample_id))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        if sample_row.num_rows() > 0 {
            let test_vector = sample_row["vector"].as_list::<i32>().value(0);
            let result = dataset
                .scan()
                .nearest("vector", &test_vector, 10)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            assert!(
                result.num_rows() > 0,
                "Multivector search should return results after compaction"
            );
        }

        // Verify the dataset still has rows after deletions and compaction
        let remaining_rows = dataset.count_all_rows().await.unwrap();
        assert!(
            remaining_rows > 0,
            "Dataset should still have rows after deletions and compaction"
        );

        // Verify we can perform multivector search on remaining data
        let sample_batch = dataset
            .scan()
            .limit(Some(1), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        if sample_batch.num_rows() > 0 {
            let test_vector = sample_batch["vector"].as_list::<i32>().value(0);
            let search_result = dataset
                .scan()
                .nearest("vector", &test_vector, 10)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            assert!(
                search_result.num_rows() > 0,
                "Multivector search should return results with remaining data"
            );
        }
    }
}
