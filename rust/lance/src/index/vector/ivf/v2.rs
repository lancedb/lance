// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF - Inverted File index.

use std::io::Write as IoWrite;
use std::marker::PhantomData;
use std::{any::Any, collections::HashMap, sync::Arc};

use crate::index::vector::{IndexFileVersion, builder::index_type_string};
use crate::index::{PreFilter, vector::VectorIndex};
use arrow::compute::concat_batches;
use arrow_arith::numeric::sub;
use arrow_array::{Float32Array, RecordBatch, UInt32Array};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use deepsize::DeepSizeOf;
use futures::future::BoxFuture;
use futures::prelude::stream::{self, TryStreamExt};
use futures::{StreamExt, TryFutureExt};
use lance_arrow::RecordBatchExt;
use lance_arrow::ipc::write_len_prefixed_bytes;
use lance_core::cache::{CacheCodec, CacheCodecImpl, CacheKey, LanceCache, WeakLanceCache};
use lance_core::utils::tokio::spawn_cpu;
use lance_core::utils::tracing::{IO_TYPE_LOAD_VECTOR_PART, TRACE_IO_EVENTS};
use lance_core::{Error, ROW_ID, Result};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::LanceEncodingsIo;
use lance_file::reader::{CachedFileMetadata, FileReader, FileReaderOptions};
use lance_index::frag_reuse::FragReuseIndex;
use lance_index::metrics::{LocalMetricsCollector, MetricsCollector, NoOpMetricsCollector};
use lance_index::vector::VectorIndexCacheEntry;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::flat::index::{FlatBinQuantizer, FlatIndex, FlatQuantizer};
use lance_index::vector::hnsw::HNSW;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::quantizer::{
    QuantizationType, Quantizer, QuantizerMetadata, QuantizerStorage,
};
use lance_index::vector::sq::ScalarQuantizer;
use lance_index::vector::storage::VectorStore;
use lance_index::vector::v3::subindex::SubIndexType;
use lance_index::{
    INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME, Index, IndexType, pb,
    vector::{
        DISTANCE_TYPE_KEY, Query, ivf::storage::IVF_METADATA_KEY, quantizer::Quantization,
        storage::IvfQuantizationStorage, v3::subindex::IvfSubIndex,
    },
};
use lance_index::{INDEX_METADATA_SCHEMA_KEY, IndexMetadata};
use lance_io::local::to_local_path;
use lance_io::scheduler::SchedulerConfig;
use lance_io::utils::CachedFileSize;
use lance_io::{
    ReadBatchParams, object_store::ObjectStore, scheduler::ScanScheduler, traits::Reader,
};
use lance_linalg::distance::DistanceType;
use object_store::path::Path;
use prost::Message;
use roaring::RoaringBitmap;
use tracing::{info, instrument};

use super::{IvfIndexPartitionStatistics, IvfIndexStatistics, centroids_to_vectors};

/// Serializable state of an IVF index, sufficient to reconstruct the index
/// without re-reading global buffers from object storage.
///
/// Serializable, type-specific state of an IVF index.
///
/// Generic over `Q` so that the parsed quantizer metadata (`Q::Metadata`) can
/// be stored directly, avoiding repeated JSON round-trips on reconstruction.
/// Produced by [`IVFIndex::to_state_entry`] and wrapped in [`IvfStateEntryBox`]
/// for storage in the index cache.
#[derive(Debug, Clone)]
pub(crate) struct IvfIndexState<Q: Quantization> {
    pub(crate) index_file_path: String,
    pub(crate) uuid: String,
    pub(crate) ivf: IvfModel,
    /// IvfModel for the auxiliary/storage file (quantizer row layout).
    /// The index and aux files have independent row layouts, so we must store
    /// both to avoid using wrong row offsets during reconstruction.
    pub(crate) aux_ivf: IvfModel,
    pub(crate) distance_type: DistanceType,
    pub(crate) sub_index_metadata: Vec<String>,
    /// Parsed quantizer metadata — stored directly to avoid JSON re-parsing on
    /// every warm-path reconstruction.
    pub(crate) metadata: <Q::Storage as QuantizerStorage>::Metadata,
    pub(crate) sub_index_type: SubIndexType,
    pub(crate) quantization_type: QuantizationType,
    /// The cache key prefix used by the original index's WeakLanceCache.
    pub(crate) cache_key_prefix: String,
    /// File sizes for the index and auxiliary files, used to avoid HEAD requests
    /// when reconstructing from cache.
    pub(crate) index_file_size: u64,
    pub(crate) aux_file_size: u64,
}

impl<Q: Quantization> DeepSizeOf for IvfIndexState<Q> {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.index_file_path.deep_size_of_children(context)
            + self.uuid.deep_size_of_children(context)
            + self.ivf.deep_size_of_children(context)
            + self.aux_ivf.deep_size_of_children(context)
            + self.sub_index_metadata.deep_size_of_children(context)
            + self.metadata.deep_size_of_children(context)
            + self.cache_key_prefix.deep_size_of_children(context)
    }
}

/// Serialization header for the `IvfIndexState` wire format.
///
/// Kept as a flat, non-generic struct so the JSON header format is stable
/// regardless of `Q`. `quantizer_metadata_json` holds the serialized
/// `Q::Metadata`; large blobs (PQ codebook, RQ matrix) follow as raw bytes.
#[derive(serde::Serialize, serde::Deserialize)]
struct IvfIndexStateHeader {
    index_file_path: String,
    uuid: String,
    distance_type: String,
    sub_index_metadata: Vec<String>,
    sub_index_type: String,
    quantization_type: String,
    quantizer_metadata_json: String,
    #[serde(default)]
    cache_key_prefix: String,
    #[serde(default)]
    index_file_size: u64,
    #[serde(default)]
    aux_file_size: u64,
}

/// Object-safe interface for a type-erased `IvfIndexState<Q>`.
///
/// Stored as `Arc<dyn IvfStateEntry>` inside [`IvfStateEntryBox`], which is
/// the concrete type held in the index cache. Splitting the trait from the
/// wrapper lets the cache infrastructure work with a sized type while the
/// hot paths call `reconstruct` without knowing `Q`.
pub(crate) trait IvfStateEntry: DeepSizeOf + Send + Sync + 'static {
    fn serialize_state(&self, writer: &mut dyn IoWrite) -> Result<()>;

    fn reconstruct<'a>(
        &'a self,
        object_store: Arc<ObjectStore>,
        file_metadata_cache: &'a LanceCache,
        index_cache: LanceCache,
    ) -> BoxFuture<'a, Result<Arc<dyn VectorIndex>>>;
}

/// Sized wrapper around `Arc<dyn IvfStateEntry>` for use as a cache value.
///
/// `IvfStateEntryBox` is the `CacheKey::ValueType` for `IvfIndexStateCacheKey`.
/// `CacheCodecImpl` on this type holds the full deserialization dispatch
/// (matching on `quantization_type`) so callers never need to branch on
/// index type after a cache hit.
pub(crate) struct IvfStateEntryBox(pub(crate) Arc<dyn IvfStateEntry>);

impl DeepSizeOf for IvfStateEntryBox {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.0.deep_size_of_children(context)
    }
}

/// Wire format (unchanged from the non-generic `IvfIndexState`):
/// `[header_json_len: u64 LE][header JSON][ivf_pb_len: u64 LE][ivf protobuf]
///  [extra_len: u64 LE][extra bytes][aux_ivf_pb_len: u64 LE][aux_ivf protobuf]`
impl CacheCodecImpl for IvfStateEntryBox {
    fn serialize(&self, writer: &mut dyn IoWrite) -> Result<()> {
        self.0.serialize_state(writer)
    }

    fn deserialize(data: &bytes::Bytes) -> Result<Self> {
        use lance_arrow::ipc::read_len_prefixed_bytes_at;

        // Parse the common wire format, then dispatch on quantization_type to
        // construct the right IvfIndexState<Q>.
        let mut offset = 0;
        let header_bytes = read_len_prefixed_bytes_at(data, &mut offset)?;
        let header: IvfIndexStateHeader = serde_json::from_slice(&header_bytes)
            .map_err(|e| lance_core::Error::io(format!("IvfIndexState header: {e}")))?;

        let ivf_bytes = read_len_prefixed_bytes_at(data, &mut offset)?;
        let ivf = IvfModel::try_from(
            pb::Ivf::decode(ivf_bytes.as_ref())
                .map_err(|e| lance_core::Error::io(format!("IvfIndexState IVF decode: {e}")))?,
        )?;

        let extra_bytes = read_len_prefixed_bytes_at(data, &mut offset)?;

        // aux_ivf was added after initial deployment; fall back to ivf on
        // clean EOF (legacy format without the field).
        let aux_ivf = if offset + 8 <= data.len() {
            let aux_ivf_bytes = read_len_prefixed_bytes_at(data, &mut offset)?;
            IvfModel::try_from(pb::Ivf::decode(aux_ivf_bytes.as_ref()).map_err(|e| {
                lance_core::Error::io(format!("IvfIndexState aux IVF decode: {e}"))
            })?)?
        } else {
            ivf.clone()
        };

        let distance_type = DistanceType::try_from(header.distance_type.as_str())?;
        let sub_index_type = SubIndexType::try_from(header.sub_index_type.as_str())?;
        let quantization_type = header.quantization_type.parse::<QuantizationType>()?;

        // Helper: parse Q::Metadata from the JSON+extra_bytes in the header,
        // then build an IvfStateEntryBox wrapping IvfIndexState<Q>.
        fn make_entry<Q: Quantization + 'static>(
            header: IvfIndexStateHeader,
            ivf: IvfModel,
            aux_ivf: IvfModel,
            extra_bytes: bytes::Bytes,
            distance_type: DistanceType,
            sub_index_type: SubIndexType,
            quantization_type: QuantizationType,
        ) -> Result<IvfStateEntryBox>
        where
            <Q::Storage as QuantizerStorage>::Metadata:
                serde::de::DeserializeOwned + QuantizerMetadata,
        {
            let mut metadata: <Q::Storage as QuantizerStorage>::Metadata =
                serde_json::from_str(&header.quantizer_metadata_json)
                    .map_err(|e| lance_core::Error::io(format!("IvfIndexState metadata: {e}")))?;
            if !extra_bytes.is_empty() {
                metadata.parse_buffer(extra_bytes)?;
            }
            Ok(IvfStateEntryBox(Arc::new(IvfIndexState::<Q> {
                index_file_path: header.index_file_path,
                uuid: header.uuid,
                ivf,
                aux_ivf,
                distance_type,
                sub_index_metadata: header.sub_index_metadata,
                metadata,
                sub_index_type,
                quantization_type,
                cache_key_prefix: header.cache_key_prefix,
                index_file_size: header.index_file_size,
                aux_file_size: header.aux_file_size,
            })))
        }

        match quantization_type {
            QuantizationType::Flat => make_entry::<FlatQuantizer>(
                header,
                ivf,
                aux_ivf,
                extra_bytes,
                distance_type,
                sub_index_type,
                quantization_type,
            ),
            QuantizationType::FlatBin => make_entry::<FlatBinQuantizer>(
                header,
                ivf,
                aux_ivf,
                extra_bytes,
                distance_type,
                sub_index_type,
                quantization_type,
            ),
            QuantizationType::Product => make_entry::<ProductQuantizer>(
                header,
                ivf,
                aux_ivf,
                extra_bytes,
                distance_type,
                sub_index_type,
                quantization_type,
            ),
            QuantizationType::Scalar => make_entry::<ScalarQuantizer>(
                header,
                ivf,
                aux_ivf,
                extra_bytes,
                distance_type,
                sub_index_type,
                quantization_type,
            ),
            QuantizationType::Rabit => make_entry::<RabitQuantizer>(
                header,
                ivf,
                aux_ivf,
                extra_bytes,
                distance_type,
                sub_index_type,
                quantization_type,
            ),
        }
    }
}

impl<Q: Quantization + 'static> IvfStateEntry for IvfIndexState<Q> {
    fn serialize_state(&self, writer: &mut dyn IoWrite) -> Result<()> {
        let quantizer_metadata_json = serde_json::to_string(&self.metadata)
            .map_err(|e| lance_core::Error::io(format!("IvfIndexState metadata: {e}")))?;
        let extra = self.metadata.extra_metadata()?;
        let extra = extra.as_deref().unwrap_or(&[]);

        let header = IvfIndexStateHeader {
            index_file_path: self.index_file_path.clone(),
            uuid: self.uuid.clone(),
            distance_type: self.distance_type.to_string(),
            sub_index_metadata: self.sub_index_metadata.clone(),
            sub_index_type: self.sub_index_type.to_string(),
            quantization_type: self.quantization_type.to_string(),
            quantizer_metadata_json,
            cache_key_prefix: self.cache_key_prefix.clone(),
            index_file_size: self.index_file_size,
            aux_file_size: self.aux_file_size,
        };
        let header_json = serde_json::to_vec(&header)
            .map_err(|e| lance_core::Error::io(format!("IvfIndexState header: {e}")))?;
        let ivf_bytes = pb::Ivf::try_from(&self.ivf)?.encode_to_vec();
        let aux_ivf_bytes = pb::Ivf::try_from(&self.aux_ivf)?.encode_to_vec();

        write_len_prefixed_bytes(writer, &header_json)?;
        write_len_prefixed_bytes(writer, &ivf_bytes)?;
        write_len_prefixed_bytes(writer, extra)?;
        write_len_prefixed_bytes(writer, &aux_ivf_bytes)?;
        Ok(())
    }

    fn reconstruct<'a>(
        &'a self,
        object_store: Arc<ObjectStore>,
        file_metadata_cache: &'a LanceCache,
        index_cache: LanceCache,
    ) -> BoxFuture<'a, Result<Arc<dyn VectorIndex>>> {
        Box::pin(async move {
            match self.sub_index_type {
                SubIndexType::Flat => {
                    reconstruct_typed::<FlatIndex, Q>(
                        self,
                        object_store,
                        file_metadata_cache,
                        index_cache,
                    )
                    .await
                }
                SubIndexType::Hnsw => {
                    reconstruct_typed::<HNSW, Q>(
                        self,
                        object_store,
                        file_metadata_cache,
                        index_cache,
                    )
                    .await
                }
            }
        })
    }
}

struct FileMetadataCacheKey;

impl CacheKey for FileMetadataCacheKey {
    type ValueType = CachedFileMetadata;
    fn type_name() -> &'static str {
        "CachedFileMetadata"
    }
    fn key(&self) -> std::borrow::Cow<'_, str> {
        "".into()
    }
}

/// Cached open file readers for the index and aux files.
///
/// Stored in `file_metadata_cache` to avoid re-opening files on every reconstruction.
/// Not serializable (no codec); a cache miss just triggers a re-open.
struct CachedIndexReaders {
    index_reader: Arc<FileReader>,
    aux_reader: Arc<FileReader>,
}

impl deepsize::DeepSizeOf for CachedIndexReaders {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        // FileReader doesn't impl DeepSizeOf. We approximate by counting the
        // fixed struct size for each reader plus the Arc<CachedFileMetadata>
        // heap contents. The metadata Arcs are also held by FileMetadataCacheKey
        // entries, so this may over-count across cache entries, but
        // over-counting is safer than under-counting for eviction purposes.
        std::mem::size_of::<FileReader>() * 2
            + self.index_reader.metadata().deep_size_of_children(context)
            + self.aux_reader.metadata().deep_size_of_children(context)
    }
}

struct CachedIndexReadersKey {
    uuid: String,
}

impl CacheKey for CachedIndexReadersKey {
    type ValueType = CachedIndexReaders;
    fn type_name() -> &'static str {
        "CachedIndexReaders"
    }
    fn key(&self) -> std::borrow::Cow<'_, str> {
        self.uuid.as_str().into()
    }
    // No codec() override → in-memory only
}

/// Open a FileReader, reusing cached file metadata if available.
async fn open_reader_cached(
    scheduler: &Arc<ScanScheduler>,
    path: &Path,
    cache: &LanceCache,
    known_file_size: u64,
) -> Result<FileReader> {
    let file_cache = cache.with_key_prefix(path.as_ref());
    // CachedFileSize::new(0) == CachedFileSize::unknown(); passing the raw
    // hint directly is safe — the type already encodes 0 as "unknown".
    let cached_size = CachedFileSize::new(known_file_size);

    if let Some(cached_meta) = file_cache.get_with_key(&FileMetadataCacheKey).await {
        let file_scheduler = scheduler.open_file(path, &cached_size).await?;
        let encodings_io = Arc::new(LanceEncodingsIo::new(file_scheduler));
        FileReader::try_open_with_file_metadata(
            encodings_io,
            path.clone(),
            None,
            Arc::<DecoderPlugins>::default(),
            cached_meta,
            cache,
            FileReaderOptions::default(),
        )
        .await
    } else {
        let file_scheduler = scheduler.open_file(path, &cached_size).await?;
        FileReader::try_open(
            file_scheduler,
            None,
            Arc::<DecoderPlugins>::default(),
            cache,
            FileReaderOptions::default(),
        )
        .await
    }
}

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

    fn type_name() -> &'static str {
        // Using type_name is safe here: the impl is in the same crate as the
        // types, so the monomorphized pointer is consistent.
        std::any::type_name::<PartitionEntry<S, Q>>()
    }

    fn codec() -> Option<CacheCodec> {
        super::partition_serde::partition_entry_codec::<S, Q>()
    }
}

/// IVF Index.
#[derive(Debug)]
pub struct IVFIndex<S: IvfSubIndex + 'static, Q: Quantization + 'static> {
    /// Local display path (via `to_local_path`), used for statistics.
    uri: String,
    /// Object-store path to the index file (forward-slash separated).
    /// Used by `cacheable_state()` for cross-platform reconstruction.
    index_path: String,
    uuid: String,

    /// Ivf model
    ivf: IvfModel,

    reader: FileReader,
    sub_index_metadata: Vec<String>,
    storage: IvfQuantizationStorage<Q>,

    distance_type: DistanceType,

    index_cache: WeakLanceCache,

    io_parallelism: usize,

    _marker: PhantomData<(S, Q)>,
}

impl<S: IvfSubIndex, Q: Quantization> DeepSizeOf for IVFIndex<S, Q> {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.uri.deep_size_of_children(context)
            + self.index_path.deep_size_of_children(context)
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
        file_sizes: HashMap<String, u64>,
    ) -> Result<Self> {
        let io_parallelism = object_store.io_parallelism();
        let scheduler_config = SchedulerConfig::max_bandwidth(&object_store);
        let scheduler = ScanScheduler::new(object_store, scheduler_config);

        let uri = index_dir.child(uuid.as_str()).child(INDEX_FILE_NAME);
        let cached_size = file_sizes
            .get(INDEX_FILE_NAME)
            .map(|&size| CachedFileSize::new(size))
            .unwrap_or_else(CachedFileSize::unknown);
        let index_reader = FileReader::try_open(
            scheduler.open_file(&uri, &cached_size).await?,
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
                .ok_or(Error::index(format!("{} not found", DISTANCE_TYPE_KEY)))?
                .as_str(),
        )?;
        let distance_type = DistanceType::try_from(index_metadata.distance_type.as_str())?;

        let ivf_pos = index_reader
            .schema()
            .metadata
            .get(IVF_METADATA_KEY)
            .ok_or(Error::index(format!("{} not found", IVF_METADATA_KEY)))?
            .parse()
            .map_err(|e| Error::index(format!("Failed to decode IVF position: {}", e)))?;
        let ivf_pb_bytes = index_reader.read_global_buffer(ivf_pos).await?;
        let ivf = IvfModel::try_from(pb::Ivf::decode(ivf_pb_bytes)?)?;

        let sub_index_metadata = index_reader
            .schema()
            .metadata
            .get(S::metadata_key())
            .ok_or(Error::index(format!("{} not found", S::metadata_key())))?;
        let sub_index_metadata: Vec<String> = serde_json::from_str(sub_index_metadata)?;

        let aux_cached_size = file_sizes
            .get(INDEX_AUXILIARY_FILE_NAME)
            .map(|&size| CachedFileSize::new(size))
            .unwrap_or_else(CachedFileSize::unknown);
        let storage_reader = FileReader::try_open(
            scheduler
                .open_file(
                    &index_dir
                        .child(uuid.as_str())
                        .child(INDEX_AUXILIARY_FILE_NAME),
                    &aux_cached_size,
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

        // Cache file metadata so reconstructions from IvfIndexState can skip
        // footer reads.
        file_metadata_cache
            .with_key_prefix(uri.as_ref())
            .insert_with_key(&FileMetadataCacheKey, index_reader.metadata().clone())
            .await;
        let aux_path = index_dir
            .child(uuid.as_str())
            .child(INDEX_AUXILIARY_FILE_NAME);
        file_metadata_cache
            .with_key_prefix(aux_path.as_ref())
            .insert_with_key(&FileMetadataCacheKey, storage.reader().metadata().clone())
            .await;

        // Cache open readers so the first reconstruction also skips file opens.
        file_metadata_cache
            .insert_with_key(
                &CachedIndexReadersKey { uuid: uuid.clone() },
                Arc::new(CachedIndexReaders {
                    index_reader: Arc::new(index_reader.clone()),
                    aux_reader: Arc::new(storage.reader().clone()),
                }),
            )
            .await;

        Ok(Self {
            uri: to_local_path(&uri),
            index_path: uri.as_ref().to_string(),
            uuid,
            ivf,
            reader: index_reader,
            storage,
            sub_index_metadata,
            distance_type,
            index_cache: WeakLanceCache::from(&index_cache),
            io_parallelism,
            _marker: PhantomData,
        })
    }

    /// Reconstruct an IVFIndex from pre-parsed state without any I/O.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_cached_state(
        uri: String,
        index_path: String,
        uuid: String,
        ivf: IvfModel,
        reader: FileReader,
        storage: IvfQuantizationStorage<Q>,
        sub_index_metadata: Vec<String>,
        distance_type: DistanceType,
        index_cache: LanceCache,
        io_parallelism: usize,
    ) -> Self {
        Self {
            uri,
            index_path,
            uuid,
            ivf,
            reader,
            storage,
            sub_index_metadata,
            distance_type,
            index_cache: WeakLanceCache::from(&index_cache),
            io_parallelism,
            _marker: PhantomData,
        }
    }

    #[instrument(level = "debug", skip(self, metrics))]
    pub async fn load_partition(
        &self,
        partition_id: usize,
        write_cache: bool,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn VectorIndexCacheEntry>> {
        if partition_id >= self.ivf.num_partitions() {
            return Err(Error::index(format!(
                "partition id {} is out of range of {} partitions",
                partition_id,
                self.ivf.num_partitions()
            )));
        }

        let cache_key = IVFPartitionKey::<S, Q>::new(partition_id);

        if write_cache {
            let entry = self
                .index_cache
                .get_or_insert_with_key(cache_key, || async {
                    info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_VECTOR_PART, index_type="ivf", part_id=partition_id);
                    metrics.record_part_load();
                    self.load_partition_entry(partition_id).await
                })
                .await?;
            Ok(entry as Arc<dyn VectorIndexCacheEntry>)
        } else {
            if let Some(part_idx) = self.index_cache.get_with_key(&cache_key).await {
                return Ok(part_idx);
            }
            info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_VECTOR_PART, index_type="ivf", part_id=partition_id);
            metrics.record_part_load();
            Ok(Arc::new(self.load_partition_entry(partition_id).await?))
        }
    }

    async fn load_partition_entry(&self, partition_id: usize) -> Result<PartitionEntry<S, Q>> {
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
        Ok(PartitionEntry {
            index: idx,
            storage,
        })
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
            let partition_centroids = self.ivf.centroid(partition_id).ok_or_else(|| {
                Error::index(format!(
                    "partition centroid {} does not exist",
                    partition_id
                ))
            })?;
            let residual_key = sub(&query.key, &partition_centroids)?;
            let mut part_query = query.clone();
            part_query.key = residual_key;
            Ok(part_query)
        } else {
            Ok(query.clone())
        }
    }

    /// Export the index state needed for reconstruction from a disk cache.
    pub(crate) fn to_state_entry(&self) -> IvfStateEntryBox {
        let (sub_index_type, quantization_type) = self.sub_index_type();
        IvfStateEntryBox(Arc::new(IvfIndexState::<Q> {
            index_file_path: self.index_path.clone(),
            uuid: self.uuid.clone(),
            ivf: self.ivf.clone(),
            aux_ivf: self.storage.ivf().clone(),
            distance_type: self.distance_type,
            sub_index_metadata: self.sub_index_metadata.clone(),
            metadata: self.storage.metadata().clone(),
            sub_index_type,
            quantization_type,
            cache_key_prefix: self.index_cache.prefix().to_string(),
            index_file_size: self.reader.metadata().file_size(),
            aux_file_size: self.storage.reader().metadata().file_size(),
        }))
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
        futures::stream::iter(0..self.ivf.num_partitions())
            .map(Ok)
            .try_for_each_concurrent(Some(self.io_parallelism), |part_id| {
                self.load_partition(part_id, true, &NoOpMetricsCollector)
                    .map_ok(|_| ())
            })
            .await
    }

    fn index_type(&self) -> IndexType {
        match self.sub_index_type() {
            (SubIndexType::Flat, QuantizationType::Flat)
            | (SubIndexType::Flat, QuantizationType::FlatBin) => IndexType::IvfFlat,
            (SubIndexType::Flat, QuantizationType::Product) => IndexType::IvfPq,
            (SubIndexType::Flat, QuantizationType::Scalar) => IndexType::IvfSq,
            (SubIndexType::Flat, QuantizationType::Rabit) => IndexType::IvfRq,
            (SubIndexType::Hnsw, QuantizationType::Product) => IndexType::IvfHnswPq,
            (SubIndexType::Hnsw, QuantizationType::Scalar) => IndexType::IvfHnswSq,
            (SubIndexType::Hnsw, QuantizationType::Flat)
            | (SubIndexType::Hnsw, QuantizationType::FlatBin) => IndexType::IvfHnswFlat,
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
        let store_stats = store_stats.as_object_mut().ok_or(Error::internal(
            "failed to get storage metadata".to_string(),
        ))?;

        sub_index_stats.append(store_stats);
        if S::name() == "FLAT" {
            let qt_label = match Q::quantization_type() {
                // FlatBin is the Hamming variant of Flat; report as "FLAT".
                QuantizationType::FlatBin => "FLAT".to_string(),
                other => other.to_string(),
            };
            sub_index_stats.insert("index_type".to_string(), qt_label.into());
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
        unimplemented!(
            "IVFIndex not currently used as sub-index and top-level indices do partition-aware search"
        )
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
                .ok_or(Error::internal(
                    "failed to downcast partition entry".to_string(),
                ))?;
            let batch = part.index.search(
                query.key,
                k,
                param,
                &part.storage,
                pre_filter,
                &local_metrics,
            )?;
            Result::Ok((batch, local_metrics))
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
        Err(Error::index("Flat index does not support load".to_string()))
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
            .ok_or(Error::internal(
                "failed to downcast partition entry".to_string(),
            ))?;
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
        Err(Error::index(
            "Remapping IVF in this way not supported".to_string(),
        ))
    }

    fn ivf_model(&self) -> &IvfModel {
        &self.ivf
    }

    fn quantizer(&self) -> Quantizer {
        self.storage.quantizer().unwrap()
    }

    fn partition_size(&self, part_id: usize) -> usize {
        self.storage.partition_size(part_id)
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

async fn reconstruct_typed<S: IvfSubIndex + 'static, Q: Quantization + 'static>(
    state: &IvfIndexState<Q>,
    object_store: Arc<ObjectStore>,
    file_metadata_cache: &LanceCache,
    index_cache: LanceCache,
) -> Result<Arc<dyn VectorIndex>> {
    let io_parallelism = object_store.io_parallelism();

    let index_path = Path::parse(&state.index_file_path)
        .map_err(|e| Error::io(format!("invalid index path: {e}")))?;

    // Derive aux path from the index path's parent directory.
    let mut parts: Vec<_> = index_path.parts().collect();
    parts.pop();
    let dir: Path = parts.into_iter().collect();
    let aux_path = dir.child(INDEX_AUXILIARY_FILE_NAME);

    let readers_key = CachedIndexReadersKey {
        uuid: state.uuid.clone(),
    };

    let (index_reader, aux_reader) =
        if let Some(cached) = file_metadata_cache.get_with_key(&readers_key).await {
            // Warm path: reuse the cached readers directly, no file opens needed.
            ((*cached.index_reader).clone(), (*cached.aux_reader).clone())
        } else {
            // Cold path: open files, then cache the readers for future reconstructions.
            let scheduler_config = SchedulerConfig::max_bandwidth(&object_store);
            let scheduler = ScanScheduler::new(object_store, scheduler_config);
            let index_reader = open_reader_cached(
                &scheduler,
                &index_path,
                file_metadata_cache,
                state.index_file_size,
            )
            .await?;
            let aux_reader = open_reader_cached(
                &scheduler,
                &aux_path,
                file_metadata_cache,
                state.aux_file_size,
            )
            .await?;
            file_metadata_cache
                .insert_with_key(
                    &readers_key,
                    Arc::new(CachedIndexReaders {
                        index_reader: Arc::new(index_reader.clone()),
                        aux_reader: Arc::new(aux_reader.clone()),
                    }),
                )
                .await;
            (index_reader, aux_reader)
        };

    let storage = IvfQuantizationStorage::from_cached(
        aux_reader,
        state.aux_ivf.clone(),
        state.metadata.clone(),
        state.distance_type,
        None,
    );

    let index = IVFIndex::<S, Q>::from_cached_state(
        to_local_path(&index_path),
        index_path.to_string(),
        state.uuid.clone(),
        state.ivf.clone(),
        index_reader,
        storage,
        state.sub_index_metadata.clone(),
        state.distance_type,
        index_cache,
        io_parallelism,
    );
    Ok(Arc::new(index))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::iter::repeat_n;
    use std::{ops::Range, sync::Arc};

    use all_asserts::{assert_ge, assert_le, assert_lt};
    use arrow::datatypes::{Float64Type, UInt8Type, UInt64Type};
    use arrow::{array::AsArray, datatypes::Float32Type};
    use arrow_array::{
        Array, ArrayRef, ArrowPrimitiveType, FixedSizeListArray, Float32Array, Int64Array,
        ListArray, RecordBatch, RecordBatchIterator, UInt64Array,
    };
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use itertools::Itertools;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::bq::{
        RQBuildParams, RQRotationType, storage::RabitQuantizationMetadata,
    };
    use lance_index::vector::storage::VectorStore;

    use crate::dataset::{InsertBuilder, UpdateBuilder, WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::index::DatasetIndexInternalExt;
    use crate::index::vector::ivf::v2::IvfPq;
    use crate::utils::test::copy_test_data_to_tmp;
    use crate::{
        Dataset,
        index::vector::{VectorIndex, VectorIndexParams},
    };
    use crate::{
        dataset::optimize::{CompactionOptions, compact_files},
        index::vector::IndexFileVersion,
    };
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::{ROW_ID, Result};
    use lance_encoding::decoder::DecoderPlugins;
    use lance_file::reader::{FileReader, FileReaderOptions};
    use lance_file::writer::FileWriter;
    use lance_index::IndexType;
    use lance_index::progress::IndexBuildProgress;
    use lance_index::vector::DIST_COL;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::kmeans::{KMeansParams, train_kmeans};
    use lance_index::vector::pq::PQBuildParams;
    use lance_index::vector::quantizer::QuantizerMetadata;
    use lance_index::vector::sq::builder::SQBuildParams;
    use lance_index::vector::{
        pq::storage::ProductQuantizationMetadata,
        sq::storage::{SQ_METADATA_KEY, ScalarQuantizationMetadata},
        storage::STORAGE_METADATA_KEY,
    };
    use lance_index::{INDEX_AUXILIARY_FILE_NAME, metrics::NoOpMetricsCollector};
    use lance_index::{optimize::OptimizeOptions, scalar::IndexReader};
    use lance_index::{scalar::IndexWriter, vector::hnsw::builder::HnswBuildParams};
    use lance_io::{
        object_store::ObjectStore,
        scheduler::{ScanScheduler, SchedulerConfig},
        utils::CachedFileSize,
    };
    use lance_linalg::distance::{DistanceType, multivec_distance};
    use lance_linalg::kernels::normalize_fsl;
    use lance_table::format::IndexMetadata;
    use lance_testing::datagen::{generate_random_array, generate_random_array_with_range};
    use rand::distr::uniform::SampleUniform;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rstest::rstest;

    const NUM_ROWS: usize = 512;
    const DIM: usize = 32;

    lance_testing::define_stage_event_progress!(RecordingProgress, IndexBuildProgress, Result<()>);

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

    async fn get_rq_metadata(
        dataset: &Dataset,
        scheduler: Arc<ScanScheduler>,
        index_uuid: &str,
    ) -> RabitQuantizationMetadata {
        let index_path = dataset
            .indices_dir()
            .child(index_uuid)
            .child(INDEX_AUXILIARY_FILE_NAME);
        let file_scheduler = scheduler
            .open_file(&index_path, &CachedFileSize::unknown())
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
        let metadata_entries: Vec<String> = serde_json::from_str(metadata).unwrap();
        serde_json::from_str(&metadata_entries[0]).unwrap()
    }

    async fn get_sq_metadata(
        dataset: &Dataset,
        scheduler: Arc<ScanScheduler>,
        index_uuid: &str,
    ) -> ScalarQuantizationMetadata {
        let index_path = dataset
            .indices_dir()
            .child(index_uuid)
            .child(INDEX_AUXILIARY_FILE_NAME);
        let file_scheduler = scheduler
            .open_file(&index_path, &CachedFileSize::unknown())
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
        if let Some(metadata) = reader.schema().metadata.get(SQ_METADATA_KEY) {
            serde_json::from_str(metadata).unwrap()
        } else {
            let metadata = reader.schema().metadata.get(STORAGE_METADATA_KEY).unwrap();
            let metadata_entries: Vec<String> = serde_json::from_str(metadata).unwrap();
            serde_json::from_str(&metadata_entries[0]).unwrap()
        }
    }

    async fn assert_rq_rotation_type(dataset: &Dataset, expected: RQRotationType) {
        let obj_store = Arc::new(ObjectStore::local());
        let scheduler = ScanScheduler::new(obj_store, SchedulerConfig::default_for_testing());
        let indices = dataset.load_indices().await.unwrap();
        assert!(!indices.is_empty(), "Expected at least one vector index");
        for index in indices.iter() {
            let rq_meta =
                get_rq_metadata(dataset, scheduler.clone(), &index.uuid.to_string()).await;
            assert_eq!(
                rq_meta.rotation_type, expected,
                "RQ rotation type mismatch for index {}",
                index.uuid
            );
        }
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

    fn generate_clustered_batch(
        rows_per_partition: usize,
        offsets: [f32; 2],
    ) -> (RecordBatch, SchemaRef) {
        let num_partitions = offsets.len();
        let total_rows = rows_per_partition * num_partitions;
        let mut ids = Vec::with_capacity(total_rows);
        let mut values = Vec::with_capacity(total_rows * DIM);
        let mut rng = StdRng::seed_from_u64(42);
        for (cluster_idx, offset) in offsets.iter().enumerate() {
            for row in 0..rows_per_partition {
                ids.push((cluster_idx * rows_per_partition + row) as u64);
                for dim in 0..DIM {
                    let base = if dim == 0 { *offset } else { 0.0 };
                    let noise = (rng.random::<f32>() - 0.5) * 0.02;
                    values.push(base + noise);
                }
            }
        }
        let ids = Arc::new(UInt64Array::from(ids));
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32)
                .unwrap(),
        );
        let schema: Arc<_> = Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("vector", vectors.data_type().clone(), false),
        ])
        .into();
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, vectors]).unwrap();
        (batch, schema)
    }

    fn generate_clustered_multivec_batch(
        cluster_sizes: &[usize],
        offsets: &[f32],
        vectors_per_row: usize,
    ) -> (RecordBatch, SchemaRef) {
        assert_eq!(
            cluster_sizes.len(),
            offsets.len(),
            "cluster sizes and offsets must match"
        );
        const ITEM_FIELD_NAME: &str = "item";
        let total_rows: usize = cluster_sizes.iter().sum();
        let mut ids = Vec::with_capacity(total_rows);
        let mut values = Vec::with_capacity(total_rows * vectors_per_row * DIM);
        let mut rng = StdRng::seed_from_u64(12345);
        let mut current_id = 0u64;
        for (&rows, &offset) in cluster_sizes.iter().zip(offsets.iter()) {
            for _ in 0..rows {
                ids.push(current_id);
                current_id += 1;
                for _ in 0..vectors_per_row {
                    for dim in 0..DIM {
                        let base = if dim == 0 { offset } else { 0.0 };
                        let noise = (rng.random::<f32>() - 0.5) * 0.02;
                        values.push(base + noise);
                    }
                }
            }
        }
        let ids_array = Arc::new(UInt64Array::from(ids));
        let vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32)
                .unwrap();
        let vector_field = Arc::new(Field::new(
            ITEM_FIELD_NAME,
            DataType::FixedSizeList(
                Arc::new(Field::new(ITEM_FIELD_NAME, DataType::Float32, true)),
                DIM as i32,
            ),
            true,
        ));
        let offsets_buffer =
            OffsetBuffer::from_lengths(std::iter::repeat_n(vectors_per_row, total_rows));
        let list_array = Arc::new(ListArray::new(
            vector_field.clone(),
            offsets_buffer,
            Arc::new(vectors),
            None,
        ));
        let schema: Arc<_> = Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("vector", DataType::List(vector_field), false),
        ])
        .into();
        let batch = RecordBatch::try_new(schema.clone(), vec![ids_array, list_array]).unwrap();
        (batch, schema)
    }

    fn build_centroids_for_offsets(offsets: &[f32]) -> Arc<FixedSizeListArray> {
        let mut centroid_values = Vec::with_capacity(offsets.len() * DIM);
        for &offset in offsets {
            for dim in 0..DIM {
                centroid_values.push(if dim == 0 { offset } else { 0.0 });
            }
        }
        Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(centroid_values),
                DIM as i32,
            )
            .unwrap(),
        )
    }

    fn make_fragment_offset_batches(
        rows_per_fragment: usize,
        offsets: &[f32],
    ) -> (Arc<Schema>, Vec<RecordBatch>) {
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

        let mut next_id = 0_u64;
        let batches = offsets
            .iter()
            .map(|offset| {
                let ids = Arc::new(UInt64Array::from_iter_values(
                    next_id..next_id + rows_per_fragment as u64,
                ));
                next_id += rows_per_fragment as u64;

                let mut values = Vec::with_capacity(rows_per_fragment * DIM);
                for _ in 0..rows_per_fragment {
                    for dim in 0..DIM {
                        values.push(*offset + dim as f32);
                    }
                }

                let vectors = Arc::new(
                    FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32)
                        .unwrap(),
                );
                RecordBatch::try_new(schema.clone(), vec![ids, vectors]).unwrap()
            })
            .collect();

        (schema, batches)
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
    ) {
        const INDEX_NAME: &str = "vector_idx";
        const APPEND_ROWS: usize = 50_000;

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

        // Append tightly clustered vectors so data flows into the same partition.
        append_dataset::<Float32Type>(&mut dataset, APPEND_ROWS, 0.0..0.05).await;

        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let final_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        assert!(
            final_ctx.num_partitions() >= 3,
            "Expected partition split to increase partitions beyond 2 for {}, got stats: {}",
            description,
            final_ctx.stats_json()
        );
    }

    async fn shrink_smallest_partition(
        dataset: &mut Dataset,
        index_name: &str,
        expected_after_join: usize,
    ) -> (usize, usize, usize) {
        const ROWS_TO_APPEND_FOR_JOIN: usize = 32;
        let row_count_before = dataset.count_all_rows().await.unwrap();
        let index_ctx = load_vector_index_context(dataset, "vector", index_name).await;
        let partitions = index_ctx.stats()["indices"][0]["partitions"]
            .as_array()
            .expect("partitions should be present");
        let (partition_idx, _size) = partitions
            .iter()
            .enumerate()
            .filter_map(|(idx, part)| part["size"].as_u64().map(|size| (idx, size)))
            .filter(|(_, size)| *size > 1)
            .min_by_key(|(_, size)| *size)
            .expect("should have at least one partition with joinable rows");

        let row_ids = load_partition_row_ids(index_ctx.ivf(), partition_idx).await;
        assert!(
            row_ids.len() > 1,
            "Partition {} should have removable rows",
            partition_idx
        );

        let rows = dataset
            .take_rows(&row_ids, dataset.schema().clone())
            .await
            .unwrap();
        let ids = rows["id"].as_primitive::<UInt64Type>().values();
        let template_values = rows["vector"]
            .as_fixed_size_list()
            .value(0)
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();

        delete_ids(dataset, &ids[1..]).await;
        compact_after_deletions(dataset).await;

        append_constant_vector(dataset, ROWS_TO_APPEND_FOR_JOIN, &template_values).await;
        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        let post_ctx = load_vector_index_context(dataset, "vector", index_name).await;
        let post_partitions = post_ctx.num_partitions();
        assert_eq!(
            post_partitions,
            expected_after_join,
            "Expected partitions to be at most {} after join, got stats: {}",
            expected_after_join,
            post_ctx.stats_json()
        );

        let row_count_after = dataset.count_all_rows().await.unwrap();
        debug_assert!(
            row_count_before + ROWS_TO_APPEND_FOR_JOIN >= row_count_after,
            "row count should not increase after delete + append"
        );
        let deleted_rows = row_count_before + ROWS_TO_APPEND_FOR_JOIN - row_count_after;

        (deleted_rows, ROWS_TO_APPEND_FOR_JOIN, post_partitions)
    }

    async fn append_constant_vector(dataset: &mut Dataset, rows: usize, template: &[f32]) {
        append_constant_vector_with_params(dataset, rows, template, None).await;
    }

    async fn append_partition_templates(
        dataset: &mut Dataset,
        rows_per_template: usize,
        templates: &[Vec<f32>],
    ) {
        assert!(
            !templates.is_empty(),
            "at least one template is required for append"
        );
        for template in templates {
            assert_eq!(
                template.len(),
                DIM,
                "Template vector should have {} dimensions",
                DIM
            );
        }

        let start_id = dataset.count_all_rows().await.unwrap() as u64;
        let total_rows = rows_per_template * templates.len();
        let ids = Arc::new(UInt64Array::from_iter_values(
            start_id..start_id + total_rows as u64,
        ));
        let mut appended_values = Vec::with_capacity(total_rows * DIM);
        for template in templates {
            for _ in 0..rows_per_template {
                appended_values.extend_from_slice(template);
            }
        }
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(appended_values),
                DIM as i32,
            )
            .unwrap(),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("vector", vectors.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, vectors]).unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        dataset.append(batches, None).await.unwrap();
    }

    async fn append_constant_vector_with_params(
        dataset: &mut Dataset,
        rows: usize,
        template: &[f32],
        write_params: Option<WriteParams>,
    ) {
        assert_eq!(
            template.len(),
            DIM,
            "Template vector should have {} dimensions",
            DIM
        );

        let start_id = dataset.count_all_rows().await.unwrap() as u64;
        let ids = Arc::new(UInt64Array::from_iter_values(
            start_id..start_id + rows as u64,
        ));
        let mut appended_values = Vec::with_capacity(rows * DIM);
        for _ in 0..rows {
            appended_values.extend_from_slice(template);
        }
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(appended_values),
                DIM as i32,
            )
            .unwrap(),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("vector", vectors.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, vectors]).unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let params = write_params.map(|mut params| {
            params.mode = WriteMode::Append;
            params
        });
        dataset.append(batches, params).await.unwrap();
    }

    #[allow(clippy::too_many_arguments)]
    async fn append_and_verify_append_phase(
        dataset: &mut Dataset,
        index_name: &str,
        template: &[f32],
        rows_to_append: usize,
        expected_partitions: usize,
        expected_total_rows: usize,
        expected_index_count: usize,
        expect_split: bool,
    ) {
        append_constant_vector(dataset, rows_to_append, template).await;
        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        let stats_json = dataset.index_statistics(index_name).await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats_json).unwrap();

        let indices = stats["indices"]
            .as_array()
            .expect("indices array should exist");
        if expect_split {
            assert_eq!(
                indices.len(),
                expected_index_count,
                "Expected {} index entries after split, got {}, stats: {}",
                expected_index_count,
                indices.len(),
                stats
            );
        } else {
            assert!(
                indices.len() >= expected_index_count,
                "Expected at least {} index entries after append, got {}, stats: {}",
                expected_index_count,
                indices.len(),
                stats
            );
        }
        assert!(
            stats["num_indices"].as_u64().unwrap() as usize >= expected_index_count,
            "num_indices should be at least {}, stats: {}",
            expected_index_count,
            stats
        );
        assert_eq!(
            stats["num_indexed_rows"].as_u64().unwrap() as usize,
            expected_total_rows,
            "Total indexed rows mismatch after append"
        );

        let base_index = indices
            .iter()
            .max_by_key(|entry| entry["num_partitions"].as_u64().unwrap_or(0))
            .expect("at least one index entry should exist");
        assert_eq!(
            base_index["num_partitions"].as_u64().unwrap() as usize,
            expected_partitions,
            "Partition count mismatch after append"
        );

        if expected_index_count == 1 {
            let partitions = base_index["partitions"]
                .as_array()
                .expect("partitions should exist");
            assert_eq!(
                partitions.len(),
                expected_partitions,
                "Expected {} partitions, found {}",
                expected_partitions,
                partitions.len()
            );
            let partition_sizes: Vec<usize> = partitions
                .iter()
                .map(|part| part["size"].as_u64().unwrap() as usize)
                .collect();
            let total_partition_rows: usize = partition_sizes.iter().sum();
            assert_eq!(
                total_partition_rows, expected_total_rows,
                "Partition sizes should sum to total rows: {:?}",
                partition_sizes
            );
        } else {
            assert!(
                !expect_split,
                "Split should result in a single merged index"
            );
        }

        assert_eq!(
            dataset.count_all_rows().await.unwrap(),
            expected_total_rows,
            "Dataset row count mismatch after append"
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

    const TWO_FRAG_NUM_ROWS: usize = 2000;
    const TWO_FRAG_DIM: usize = 128;
    const TWO_FRAG_NUM_PARTITIONS: usize = 4;
    const TWO_FRAG_NUM_SUBVECTORS: usize = 16;
    const TWO_FRAG_NUM_BITS: usize = 8;
    const TWO_FRAG_SAMPLE_RATE: usize = 7;
    const TWO_FRAG_MAX_ITERS: u32 = 20;

    fn make_two_fragment_batches() -> (Arc<Schema>, Vec<RecordBatch>) {
        let ids = Arc::new(UInt64Array::from_iter_values(0..TWO_FRAG_NUM_ROWS as u64));

        let values = generate_random_array_with_range(TWO_FRAG_NUM_ROWS * TWO_FRAG_DIM, 0.0..1.0);
        let vectors = Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(values),
                TWO_FRAG_DIM as i32,
            )
            .unwrap(),
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("vector", vectors.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, vectors]).unwrap();

        (schema, vec![batch])
    }

    async fn write_dataset_from_batches(
        test_uri: &str,
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
    ) -> Dataset {
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);

        let write_params = WriteParams {
            max_rows_per_file: 500,
            mode: WriteMode::Overwrite,
            ..Default::default()
        };

        Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap()
    }

    async fn prepare_global_ivf_pq(
        dataset: &Dataset,
        vector_column: &str,
    ) -> (IvfBuildParams, PQBuildParams) {
        let batch = dataset
            .scan()
            .project(&[vector_column.to_string()])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let vectors = batch
            .column_by_name(vector_column)
            .expect("vector column should exist")
            .as_fixed_size_list();

        let dim = vectors.value_length() as usize;
        assert_eq!(dim, TWO_FRAG_DIM, "unexpected vector dimension");

        let values = vectors.values().as_primitive::<Float32Type>();

        let kmeans_params = KMeansParams::new(None, TWO_FRAG_MAX_ITERS, 1, DistanceType::L2);
        let kmeans = train_kmeans::<Float32Type>(
            values,
            kmeans_params,
            dim,
            TWO_FRAG_NUM_PARTITIONS,
            TWO_FRAG_SAMPLE_RATE,
        )
        .unwrap();

        let centroids_flat = kmeans.centroids.as_primitive::<Float32Type>().clone();
        let centroids_fsl =
            Arc::new(FixedSizeListArray::try_new_from_values(centroids_flat, dim as i32).unwrap());
        let mut ivf_params =
            IvfBuildParams::try_with_centroids(TWO_FRAG_NUM_PARTITIONS, centroids_fsl).unwrap();
        ivf_params.max_iters = TWO_FRAG_MAX_ITERS as usize;
        ivf_params.sample_rate = TWO_FRAG_SAMPLE_RATE;

        let mut pq_train_params = PQBuildParams::new(TWO_FRAG_NUM_SUBVECTORS, TWO_FRAG_NUM_BITS);
        pq_train_params.max_iters = TWO_FRAG_MAX_ITERS as usize;
        pq_train_params.sample_rate = TWO_FRAG_SAMPLE_RATE;

        let pq = pq_train_params.build(vectors, DistanceType::L2).unwrap();
        let codebook_flat = pq.codebook.values().as_primitive::<Float32Type>().clone();
        let pq_codebook: ArrayRef = Arc::new(codebook_flat);
        let mut pq_params =
            PQBuildParams::with_codebook(TWO_FRAG_NUM_SUBVECTORS, TWO_FRAG_NUM_BITS, pq_codebook);
        pq_params.max_iters = TWO_FRAG_MAX_ITERS as usize;
        pq_params.sample_rate = TWO_FRAG_SAMPLE_RATE;

        (ivf_params, pq_params)
    }

    async fn prepare_global_ivf(dataset: &Dataset, vector_column: &str) -> IvfBuildParams {
        let batch = dataset
            .scan()
            .project(&[vector_column.to_string()])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let vectors = batch
            .column_by_name(vector_column)
            .expect("vector column should exist")
            .as_fixed_size_list();

        let dim = vectors.value_length() as usize;
        assert_eq!(dim, TWO_FRAG_DIM, "unexpected vector dimension");

        let values = vectors.values().as_primitive::<Float32Type>();
        let kmeans_params = KMeansParams::new(None, TWO_FRAG_MAX_ITERS, 1, DistanceType::L2);
        let kmeans = train_kmeans::<Float32Type>(
            values,
            kmeans_params,
            dim,
            TWO_FRAG_NUM_PARTITIONS,
            TWO_FRAG_SAMPLE_RATE,
        )
        .unwrap();

        let centroids_flat = kmeans.centroids.as_primitive::<Float32Type>().clone();
        let centroids_fsl =
            Arc::new(FixedSizeListArray::try_new_from_values(centroids_flat, dim as i32).unwrap());
        let mut ivf_params =
            IvfBuildParams::try_with_centroids(TWO_FRAG_NUM_PARTITIONS, centroids_fsl).unwrap();
        ivf_params.max_iters = TWO_FRAG_MAX_ITERS as usize;
        ivf_params.sample_rate = TWO_FRAG_SAMPLE_RATE;
        ivf_params
    }

    async fn build_segments_for_fragment_groups(
        dataset: &mut Dataset,
        fragment_groups: Vec<Vec<u32>>, // each group is a set of fragment ids
        params: &VectorIndexParams,
        index_name: &str,
    ) -> Vec<IndexMetadata> {
        let mut segments = Vec::new();

        for fragments in fragment_groups {
            let mut builder = dataset.create_index_builder(&["vector"], IndexType::Vector, params);
            builder = builder.name(index_name.to_string()).fragments(fragments);
            segments.push(builder.execute_uncommitted().await.unwrap());
        }

        segments
    }

    async fn build_ivfpq_for_fragment_groups(
        dataset: &mut Dataset,
        fragment_groups: Vec<Vec<u32>>, // each group is a set of fragment ids
        ivf_params: &IvfBuildParams,
        pq_params: &PQBuildParams,
        index_name: &str,
    ) {
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            ivf_params.clone(),
            pq_params.clone(),
        );

        let segments =
            build_segments_for_fragment_groups(dataset, fragment_groups, &params, index_name).await;
        let committed_segments = build_distributed_segments(dataset, segments, index_name).await;
        assert!(!committed_segments.is_empty());
    }

    fn assert_centroids_equal(reference: &serde_json::Value, candidate: &serde_json::Value) {
        let centroids_a = reference["centroids"]
            .as_array()
            .expect("centroids should be an array");
        let centroids_b = candidate["centroids"]
            .as_array()
            .expect("centroids should be an array");
        assert_eq!(
            centroids_a.len(),
            centroids_b.len(),
            "num centroids mismatch",
        );
        for (row_a, row_b) in centroids_a.iter().zip(centroids_b.iter()) {
            let row_a = row_a
                .as_array()
                .unwrap_or_else(|| panic!("invalid centroid row: {:?}", row_a));
            let row_b = row_b
                .as_array()
                .unwrap_or_else(|| panic!("invalid centroid row: {:?}", row_b));
            assert_eq!(row_a.len(), row_b.len(), "centroid dim mismatch");
            for (va, vb) in row_a.iter().zip(row_b.iter()) {
                let fa = va.as_f64().expect("centroid must be numeric") as f32;
                let fb = vb.as_f64().expect("centroid must be numeric") as f32;
                assert!(
                    (fa - fb).abs() <= 1e-4,
                    "centroid mismatch: {} vs {}",
                    fa,
                    fb
                );
            }
        }
    }

    fn sum_partition_sizes(indices: &[serde_json::Value]) -> Vec<u64> {
        let mut totals = Vec::new();
        for index in indices {
            let partitions = index["partitions"]
                .as_array()
                .expect("partitions should be an array");
            if totals.is_empty() {
                totals.resize(partitions.len(), 0);
            } else {
                assert_eq!(totals.len(), partitions.len(), "num partitions mismatch");
            }
            for (total, partition) in totals.iter_mut().zip(partitions.iter()) {
                *total += partition["size"].as_u64().expect("partition size");
            }
        }
        totals
    }

    fn assert_ivf_layout_compatible(stats_a: &serde_json::Value, stats_b: &serde_json::Value) {
        let indices_a = stats_a["indices"]
            .as_array()
            .expect("indices should be an array");
        let indices_b = stats_b["indices"]
            .as_array()
            .expect("indices should be an array");
        assert!(
            !indices_a.is_empty() && !indices_b.is_empty(),
            "indices should not be empty",
        );

        let reference = &indices_a[0];
        for index in indices_a.iter().skip(1).chain(indices_b.iter()) {
            assert_centroids_equal(reference, index);
        }

        let sizes_a = sum_partition_sizes(indices_a);
        let sizes_b = sum_partition_sizes(indices_b);
        assert_eq!(sizes_a, sizes_b, "aggregated partition sizes mismatch");
    }

    /// Commit caller-defined segment groups as one logical index.
    async fn build_distributed_segments(
        dataset: &mut Dataset,
        segments: Vec<IndexMetadata>,
        index_name: &str,
    ) -> Vec<IndexMetadata> {
        dataset
            .commit_existing_index_segments(index_name, "vector", segments.clone())
            .await
            .unwrap();
        segments
    }

    #[tokio::test]
    async fn test_ivfpq_recall_performance_on_two_frags_single_vs_split() {
        const INDEX_NAME: &str = "vector_idx";

        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();

        let (schema, batches) = make_two_fragment_batches();

        let ds_single_uri = format!("{}/single", base_uri);
        let ds_split_uri = format!("{}/split", base_uri);

        let mut ds_single =
            write_dataset_from_batches(&ds_single_uri, schema.clone(), batches.clone()).await;
        let mut ds_split = write_dataset_from_batches(&ds_split_uri, schema, batches).await;

        let fragments_single = ds_single.get_fragments();
        assert!(
            fragments_single.len() >= 2,
            "expected at least 2 fragments in ds_single, got {}",
            fragments_single.len()
        );
        let fragments_split = ds_split.get_fragments();
        assert!(
            fragments_split.len() >= 2,
            "expected at least 2 fragments in ds_split, got {}",
            fragments_split.len()
        );

        let (ivf_params, pq_params) = prepare_global_ivf_pq(&ds_single, "vector").await;

        let group_single = vec![
            fragments_single[0].id() as u32,
            fragments_single[1].id() as u32,
        ];
        build_ivfpq_for_fragment_groups(
            &mut ds_single,
            vec![group_single],
            &ivf_params,
            &pq_params,
            INDEX_NAME,
        )
        .await;

        let group0 = vec![fragments_split[0].id() as u32];
        let group1 = vec![fragments_split[1].id() as u32];
        build_ivfpq_for_fragment_groups(
            &mut ds_split,
            vec![group0, group1],
            &ivf_params,
            &pq_params,
            INDEX_NAME,
        )
        .await;

        let stats_single_json = ds_single.index_statistics(INDEX_NAME).await.unwrap();
        let stats_split_json = ds_split.index_statistics(INDEX_NAME).await.unwrap();
        let stats_single: serde_json::Value = serde_json::from_str(&stats_single_json).unwrap();
        let stats_split: serde_json::Value = serde_json::from_str(&stats_split_json).unwrap();
        assert_ivf_layout_compatible(&stats_single, &stats_split);
        assert_eq!(
            stats_single["num_indexed_rows"],
            stats_split["num_indexed_rows"]
        );

        const K: usize = 10;
        const NUM_QUERIES: usize = 10;

        async fn collect_row_ids(ds: &Dataset, queries: &[Arc<dyn Array>]) -> Vec<Vec<u64>> {
            let mut ids_per_query = Vec::with_capacity(queries.len());
            for q in queries {
                let result = ds
                    .scan()
                    .with_row_id()
                    .project(&["_rowid"] as &[&str])
                    .unwrap()
                    .nearest("vector", q.as_ref(), K)
                    .unwrap()
                    .minimum_nprobes(TWO_FRAG_NUM_PARTITIONS)
                    .try_into_batch()
                    .await
                    .unwrap();

                let row_ids = result[ROW_ID]
                    .as_primitive::<UInt64Type>()
                    .values()
                    .iter()
                    .copied()
                    .collect::<Vec<u64>>();
                ids_per_query.push(row_ids);
            }
            ids_per_query
        }

        let query_batch = ds_single
            .scan()
            .project(&["vector"] as &[&str])
            .unwrap()
            .limit(Some(NUM_QUERIES as i64), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let vectors = query_batch["vector"].as_fixed_size_list();
        let queries: Vec<Arc<dyn Array>> = (0..vectors.len())
            .map(|i| vectors.value(i) as Arc<dyn Array>)
            .collect();

        let ids_single = collect_row_ids(&ds_single, &queries).await;
        let ids_split = collect_row_ids(&ds_split, &queries).await;

        assert_eq!(
            ids_single, ids_split,
            "single vs split index returned different Top-K row ids",
        );
    }

    #[rstest]
    #[case::ivf_flat(IndexType::IvfFlat)]
    #[case::ivf_pq(IndexType::IvfPq)]
    #[case::ivf_sq(IndexType::IvfSq)]
    #[case::ivf_rq(IndexType::IvfRq)]
    #[tokio::test]
    async fn test_distributed_vector_build_commits_multiple_segments_and_preserves_query_results(
        #[case] index_type: IndexType,
    ) {
        const INDEX_NAME: &str = "vector_idx";
        const K: usize = 10;
        const NUM_QUERIES: usize = 10;

        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();

        // Generate the data once, then write it twice to two independent dataset URIs.
        let (schema, batches) = make_two_fragment_batches();

        let ds_single_uri = format!("{}/single", base_uri);
        let ds_split_uri = format!("{}/split", base_uri);

        let mut ds_single =
            write_dataset_from_batches(&ds_single_uri, schema.clone(), batches.clone()).await;
        let mut ds_split = write_dataset_from_batches(&ds_split_uri, schema, batches).await;

        // Ensure we have at least 2 fragments.
        let fragments_single = ds_single.get_fragments();
        assert!(
            fragments_single.len() >= 2,
            "expected at least 2 fragments in ds_single, got {}",
            fragments_single.len()
        );
        let fragments_split = ds_split.get_fragments();
        assert!(
            fragments_split.len() >= 2,
            "expected at least 2 fragments in ds_split, got {}",
            fragments_split.len()
        );

        let distributed_params = match index_type {
            IndexType::IvfFlat => {
                let ivf_params = prepare_global_ivf(&ds_single, "vector").await;
                VectorIndexParams::with_ivf_flat_params(DistanceType::L2, ivf_params)
            }
            IndexType::IvfPq => {
                let (ivf_params, pq_params) = prepare_global_ivf_pq(&ds_single, "vector").await;
                VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params)
            }
            IndexType::IvfSq => {
                let ivf_params = prepare_global_ivf(&ds_single, "vector").await;
                VectorIndexParams::with_ivf_sq_params(
                    DistanceType::L2,
                    ivf_params,
                    SQBuildParams::default(),
                )
            }
            IndexType::IvfRq => {
                let ivf_params = prepare_global_ivf(&ds_single, "vector").await;
                VectorIndexParams::with_ivf_rq_params(
                    DistanceType::L2,
                    ivf_params,
                    RQBuildParams::with_rotation_type(1, RQRotationType::Fast),
                )
            }
            other => panic!("unsupported test index type: {}", other),
        };

        ds_single
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &distributed_params,
                true,
            )
            .await
            .unwrap();

        let fragment_groups = fragments_split
            .iter()
            .map(|fragment| vec![fragment.id() as u32])
            .collect::<Vec<_>>();
        let expected_segment_count = fragment_groups.len();
        let segments = build_segments_for_fragment_groups(
            &mut ds_split,
            fragment_groups,
            &distributed_params,
            INDEX_NAME,
        )
        .await;
        let segments = build_distributed_segments(&mut ds_split, segments, INDEX_NAME).await;
        assert_eq!(segments.len(), expected_segment_count);
        for segment in &segments {
            let segment_index = ds_split
                .indices_dir()
                .child(segment.uuid.to_string())
                .child(crate::index::INDEX_FILE_NAME);
            assert!(
                ds_split
                    .object_store()
                    .exists(&segment_index)
                    .await
                    .unwrap(),
                "segment file should exist at {}",
                segment_index
            );
        }

        let committed_segments = ds_split.load_indices_by_name(INDEX_NAME).await.unwrap();
        assert_eq!(committed_segments.len(), expected_segment_count);
        for committed in committed_segments {
            let covered_fragments = committed
                .fragment_bitmap
                .as_ref()
                .expect("distributed segment should have fragment coverage");
            assert_eq!(covered_fragments.len(), 1);
        }

        async fn collect_row_ids(ds: &Dataset, queries: &[Arc<dyn Array>]) -> Vec<Vec<u64>> {
            let mut ids_per_query = Vec::with_capacity(queries.len());
            for q in queries {
                let result = ds
                    .scan()
                    .with_row_id()
                    .project(&["_rowid"] as &[&str])
                    .unwrap()
                    .nearest("vector", q.as_ref(), K)
                    .unwrap()
                    .minimum_nprobes(TWO_FRAG_NUM_PARTITIONS)
                    .try_into_batch()
                    .await
                    .unwrap();

                let row_ids = result[ROW_ID]
                    .as_primitive::<UInt64Type>()
                    .values()
                    .iter()
                    .copied()
                    .collect::<Vec<u64>>();
                ids_per_query.push(row_ids);
            }
            ids_per_query
        }

        // Collect a deterministic query set from ds_single.
        let query_batch = ds_single
            .scan()
            .project(&["vector"] as &[&str])
            .unwrap()
            .limit(Some(NUM_QUERIES as i64), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let vectors = query_batch["vector"].as_fixed_size_list();
        let queries: Vec<Arc<dyn Array>> = (0..vectors.len())
            .map(|i| vectors.value(i) as Arc<dyn Array>)
            .collect();

        let ids_single = collect_row_ids(&ds_single, &queries).await;
        let ids_split = collect_row_ids(&ds_split, &queries).await;

        if index_type == IndexType::IvfRq {
            for row_ids in &ids_split {
                assert_eq!(
                    row_ids.len(),
                    K,
                    "distributed IVF_RQ query should still return exactly {K} row ids",
                );
            }
        } else {
            assert_eq!(
                ids_single, ids_split,
                "single vs segmented distributed index returned different Top-K row ids",
            );
        }
    }

    #[rstest]
    #[case::ivf_flat(IndexType::IvfFlat)]
    #[case::ivf_pq(IndexType::IvfPq)]
    #[case::ivf_sq(IndexType::IvfSq)]
    #[tokio::test]
    async fn test_distributed_vector_grouped_build_allows_concurrent_group_execution(
        #[case] index_type: IndexType,
    ) {
        const INDEX_NAME: &str = "grouped_idx";
        const K: usize = 10;
        const NUM_QUERIES: usize = 10;

        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();

        let (schema, batches) = make_two_fragment_batches();
        let ds_single_uri = format!("{}/grouped_single", base_uri);
        let ds_split_uri = format!("{}/grouped_split", base_uri);

        let mut ds_single =
            write_dataset_from_batches(&ds_single_uri, schema.clone(), batches.clone()).await;
        let mut ds_split = write_dataset_from_batches(&ds_split_uri, schema, batches).await;

        let distributed_params = match index_type {
            IndexType::IvfFlat => {
                let ivf_params = prepare_global_ivf(&ds_single, "vector").await;
                VectorIndexParams::with_ivf_flat_params(DistanceType::L2, ivf_params)
            }
            IndexType::IvfPq => {
                let (ivf_params, pq_params) = prepare_global_ivf_pq(&ds_single, "vector").await;
                VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params)
            }
            IndexType::IvfSq => {
                let ivf_params = prepare_global_ivf(&ds_single, "vector").await;
                VectorIndexParams::with_ivf_sq_params(
                    DistanceType::L2,
                    ivf_params,
                    SQBuildParams::default(),
                )
            }
            other => panic!("unsupported test index type: {}", other),
        };

        ds_single
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some(INDEX_NAME.to_string()),
                &distributed_params,
                true,
            )
            .await
            .unwrap();

        let fragment_groups = ds_split
            .get_fragments()
            .into_iter()
            .map(|fragment| vec![fragment.id() as u32])
            .collect::<Vec<_>>();
        let segments = build_segments_for_fragment_groups(
            &mut ds_split,
            fragment_groups,
            &distributed_params,
            INDEX_NAME,
        )
        .await;

        assert!(segments.len() >= 4);
        let grouped_inputs = segments
            .chunks(2)
            .map(|group| group.to_vec())
            .collect::<Vec<_>>();
        let mut expected_fragment_coverage = grouped_inputs
            .iter()
            .map(|group| {
                group
                    .iter()
                    .flat_map(|partial| {
                        partial
                            .fragment_bitmap
                            .as_ref()
                            .expect("partial shard should have fragment coverage")
                            .iter()
                    })
                    .sorted()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        expected_fragment_coverage.sort();

        let grouped_segments = futures::future::try_join_all(
            grouped_inputs
                .into_iter()
                .map(|group| ds_split.merge_existing_index_segments(group)),
        )
        .await
        .unwrap();
        let grouped_segments =
            build_distributed_segments(&mut ds_split, grouped_segments, INDEX_NAME).await;
        assert_eq!(grouped_segments.len(), expected_fragment_coverage.len());
        let mut actual_fragment_coverage = grouped_segments
            .iter()
            .map(|segment| {
                segment
                    .fragment_bitmap
                    .as_ref()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        actual_fragment_coverage.sort();
        assert_eq!(
            actual_fragment_coverage, expected_fragment_coverage,
            "built segment coverage should equal the union of its source partial shards",
        );

        async fn collect_row_ids(ds: &Dataset, queries: &[Arc<dyn Array>]) -> Vec<Vec<u64>> {
            let mut ids_per_query = Vec::with_capacity(queries.len());
            for q in queries {
                let result = ds
                    .scan()
                    .with_row_id()
                    .project(&["_rowid"] as &[&str])
                    .unwrap()
                    .nearest("vector", q.as_ref(), K)
                    .unwrap()
                    .minimum_nprobes(TWO_FRAG_NUM_PARTITIONS)
                    .try_into_batch()
                    .await
                    .unwrap();

                ids_per_query.push(
                    result[ROW_ID]
                        .as_primitive::<UInt64Type>()
                        .values()
                        .iter()
                        .copied()
                        .collect(),
                );
            }
            ids_per_query
        }

        let query_batch = ds_single
            .scan()
            .project(&["vector"] as &[&str])
            .unwrap()
            .limit(Some(NUM_QUERIES as i64), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let vectors = query_batch["vector"].as_fixed_size_list();
        let queries: Vec<Arc<dyn Array>> = (0..vectors.len())
            .map(|i| vectors.value(i) as Arc<dyn Array>)
            .collect();

        let ids_single = collect_row_ids(&ds_single, &queries).await;
        let ids_split = collect_row_ids(&ds_split, &queries).await;
        if matches!(index_type, IndexType::IvfSq) {
            for (single, split) in ids_single.iter().zip(ids_split.iter()) {
                assert_eq!(single.len(), split.len());
                let overlap = single
                    .iter()
                    .filter(|row_id| split.contains(row_id))
                    .count();
                assert!(
                    overlap >= K / 3,
                    "single vs segmented distributed SQ index returned too little top-k overlap",
                );
            }
        } else {
            assert_eq!(ids_single, ids_split);
        }
    }

    #[tokio::test]
    async fn test_distributed_vector_plan_rejects_overlapping_fragment_coverage() {
        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();
        let (schema, batches) = make_two_fragment_batches();
        let dataset_uri = format!("{}/overlap_fragments", base_uri);
        let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

        let fragment = dataset.get_fragments()[0].id() as u32;
        let params = VectorIndexParams::with_ivf_flat_params(
            DistanceType::L2,
            prepare_global_ivf(&dataset, "vector").await,
        );
        let mut segments = Vec::new();

        for _ in 0..2 {
            let segment = dataset
                .create_index_builder(&["vector"], IndexType::Vector, &params)
                .name("vector_idx".to_string())
                .fragments(vec![fragment])
                .execute_uncommitted()
                .await
                .unwrap();
            segments.push(segment);
        }

        let err = dataset
            .merge_existing_index_segments(segments)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("overlapping fragment coverage"));
    }

    #[tokio::test]
    async fn test_distributed_vector_build_supports_hnsw_variants() {
        let test_dir = TempStrDir::default();
        let base_uri = test_dir.as_str();
        let (schema, batches) = make_two_fragment_batches();
        let dataset_uri = format!("{}/distributed_hnsw_supported", base_uri);
        let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

        let fragments = dataset.get_fragments();
        assert!(fragments.len() >= 2);
        let params = VectorIndexParams::ivf_hnsw(
            DistanceType::L2,
            prepare_global_ivf(&dataset, "vector").await,
            HnswBuildParams::default(),
        );
        let mut segments = Vec::new();

        for fragment in fragments.iter().take(2) {
            let segment = dataset
                .create_index_builder(&["vector"], IndexType::Vector, &params)
                .name("vector_idx".to_string())
                .fragments(vec![fragment.id() as u32])
                .execute_uncommitted()
                .await
                .unwrap();
            segments.push(segment);
        }

        dataset
            .commit_existing_index_segments("vector_idx", "vector", segments)
            .await
            .unwrap();

        let query_batch = dataset
            .scan()
            .project(&["vector"] as &[&str])
            .unwrap()
            .limit(Some(4), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let q = query_batch["vector"].as_fixed_size_list().value(0);
        let result = dataset
            .scan()
            .project(&["_rowid"] as &[&str])
            .unwrap()
            .nearest("vector", q.as_ref(), 5)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert!(result.num_rows() > 0);
    }

    #[tokio::test]
    async fn test_merge_index_metadata_reports_progress() {
        const INDEX_NAME: &str = "vector_idx";

        let test_dir = TempStrDir::default();
        let dataset_uri = format!("{}/progress", test_dir.as_str());
        let (schema, batches) = make_two_fragment_batches();
        let mut dataset = write_dataset_from_batches(&dataset_uri, schema, batches).await;

        let fragments = dataset.get_fragments();
        assert!(
            fragments.len() >= 2,
            "expected at least 2 fragments, got {}",
            fragments.len()
        );
        let expected_rows = fragments[0].physical_rows().await.unwrap() as u64
            + fragments[1].physical_rows().await.unwrap() as u64;

        let (ivf_params, pq_params) = prepare_global_ivf_pq(&dataset, "vector").await;
        let params = VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);
        let mut segments = Vec::new();
        for fragment in fragments.iter().take(2) {
            segments.push(
                dataset
                    .create_index_builder(&["vector"], IndexType::Vector, &params)
                    .name(INDEX_NAME.to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }

        let progress = Arc::new(RecordingProgress::default());
        let merged_segment = crate::index::vector::ivf::merge_segments_with_progress(
            dataset.object_store(),
            &dataset.indices_dir(),
            segments,
            progress.clone(),
        )
        .await
        .unwrap();
        dataset
            .commit_existing_index_segments(INDEX_NAME, "vector", vec![merged_segment])
            .await
            .unwrap();

        let events = progress.recorded_events();
        let tags = events
            .iter()
            .map(|(kind, stage, _)| format!("{kind}:{stage}"))
            .collect::<Vec<_>>();
        let merge_total = events
            .iter()
            .find_map(|(kind, stage, value)| {
                if kind == "start" && stage == "merge_partitions" {
                    Some(*value)
                } else {
                    None
                }
            })
            .expect("missing merge_partitions start total");
        let merged_rows = events
            .iter()
            .filter_map(|(kind, stage, value)| {
                if kind == "progress" && stage == "merge_partitions" {
                    Some(*value)
                } else {
                    None
                }
            })
            .next_back()
            .unwrap_or_default();
        let read_start = tags
            .iter()
            .position(|e| e == "start:read_shard_metadata")
            .expect("missing read_shard_metadata start");
        let read_complete = tags
            .iter()
            .position(|e| e == "complete:read_shard_metadata")
            .expect("missing read_shard_metadata complete");
        let merge_start = tags
            .iter()
            .position(|e| e == "start:merge_partitions")
            .expect("missing merge_partitions start");
        let merge_complete = tags
            .iter()
            .position(|e| e == "complete:merge_partitions")
            .expect("missing merge_partitions complete");
        let aux_start = tags
            .iter()
            .position(|e| e == "start:write_auxiliary_index")
            .expect("missing write_auxiliary_index start");
        let aux_complete = tags
            .iter()
            .position(|e| e == "complete:write_auxiliary_index")
            .expect("missing write_auxiliary_index complete");
        let root_start = tags
            .iter()
            .position(|e| e == "start:write_root_index")
            .expect("missing write_root_index start");
        let root_complete = tags
            .iter()
            .position(|e| e == "complete:write_root_index")
            .expect("missing write_root_index complete");

        assert!(read_start < read_complete);
        assert!(read_complete < merge_start);
        assert!(merge_start < merge_complete);
        assert!(merge_complete < aux_start);
        assert!(aux_start < aux_complete);
        assert!(aux_complete < root_start);
        assert!(root_start < root_complete);
        assert_eq!(
            merge_total, expected_rows,
            "expected merge_partitions total rows to match dataset rows"
        );
        assert_eq!(
            merged_rows, expected_rows,
            "expected merge_partitions completed rows to match dataset rows"
        );
        assert!(
            tags.iter().any(|e| e == "progress:write_root_index"),
            "expected write_root_index progress callbacks"
        );
    }

    #[tokio::test]
    async fn test_distributed_ivf_sq_worker_training_respects_fragment_filter() {
        const ROWS_PER_FRAGMENT: usize = 64;
        const FRAGMENT_OFFSETS: [f32; 2] = [0.0, 1000.0];

        let test_dir = TempStrDir::default();
        let dataset_uri = format!("{}/distributed_sq_fragment_filter", test_dir.as_str());
        let (schema, batches) = make_fragment_offset_batches(ROWS_PER_FRAGMENT, &FRAGMENT_OFFSETS);
        let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        let mut dataset = Dataset::write(
            batches,
            &dataset_uri,
            Some(WriteParams {
                max_rows_per_file: ROWS_PER_FRAGMENT,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), FRAGMENT_OFFSETS.len());

        let ivf_params =
            IvfBuildParams::try_with_centroids(2, build_centroids_for_offsets(&FRAGMENT_OFFSETS))
                .unwrap();
        let params = VectorIndexParams::with_ivf_sq_params(
            DistanceType::L2,
            ivf_params,
            SQBuildParams::default(),
        );

        let segment = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name("sq_fragment_filter".to_string())
            .fragments(vec![fragments[0].id() as u32])
            .execute_uncommitted()
            .await
            .unwrap();

        let scheduler = ScanScheduler::new(
            Arc::new(dataset.object_store().clone()),
            SchedulerConfig::default_for_testing(),
        );
        let sq_meta = get_sq_metadata(&dataset, scheduler, &segment.uuid.to_string()).await;

        assert_eq!(sq_meta.bounds.start, 0.0);
        assert_eq!(sq_meta.bounds.end, (DIM - 1) as f64);
        assert_lt!(sq_meta.bounds.end, FRAGMENT_OFFSETS[1] as f64);
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
            let indices = dataset.load_indices().await.unwrap();
            assert_eq!(indices.len(), 1);
            let old_meta = indices[0].clone();
            rewrite_pq_storage(&mut dataset, &old_meta).await.unwrap();
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

    async fn test_remap(params: VectorIndexParams, nlist: usize, recall_requirement: f32) {
        match params.metric_type {
            DistanceType::Hamming => {
                Box::pin(test_remap_impl::<UInt8Type>(
                    params,
                    nlist,
                    recall_requirement,
                    0..4,
                ))
                .await;
            }
            _ => {
                Box::pin(test_remap_impl::<Float32Type>(
                    params,
                    nlist,
                    recall_requirement,
                    0.0..1.0,
                ))
                .await;
            }
        }
    }

    async fn test_remap_impl<T: ArrowPrimitiveType>(
        params: VectorIndexParams,
        nlist: usize,
        recall_requirement: f32,
        range: Range<T::Native>,
    ) where
        T::Native: SampleUniform,
    {
        // let recall_requirement = recall_requirement * 0.99;
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
        // 100 can't be exactly expressed as a float, so we need to use a tolerance
        assert_ge!(
            recall,
            recall_requirement - f32::EPSILON,
            "num_rows: {}, intersection: {}, recall: {}",
            row_ids.len(),
            row_ids.intersection(&gt).count(),
            recall
        );

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
        test_remap(params.clone(), nlist, recall_requirement).await;
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
        // PQ performs worse on farther vectors, so if we delete the many nearest vectors, the recall will be lower
        // lower the recall requirement in remap case for PQ, because it deletes half of the vectors
        test_remap(params, nlist, recall_requirement * 0.9).await;
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
        // PQ performs worse on farther vectors, so if we delete the many nearest vectors, the recall will be lower
        // lower the recall requirement in remap case for PQ, because it deletes half of the vectors
        test_remap(params.clone(), nlist, recall_requirement * 0.9).await;
        test_delete_all_rows(params).await;
    }

    #[rstest]
    // Temporarily disable recall checks for 4-bit PQ.
    #[case(4, DistanceType::L2, 0.0)]
    #[case(4, DistanceType::Cosine, 0.0)]
    #[case(4, DistanceType::Dot, 0.0)]
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
        // PQ performs worse on farther vectors, so if we delete the many nearest vectors, the recall will be lower
        // lower the recall requirement in remap case for PQ, because it deletes half of the vectors
        test_remap(params, nlist, recall_requirement * 0.9).await;
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
        test_remap(params, nlist, recall_requirement).await;
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
        #[values(RQRotationType::Fast, RQRotationType::Matrix)] rotation_type: RQRotationType,
    ) {
        let _ = env_logger::try_init();
        let ivf_params = IvfBuildParams::new(nlist);
        let rq_params = RQBuildParams::with_rotation_type(1, rotation_type);
        let params = VectorIndexParams::with_ivf_rq_params(distance_type, ivf_params, rq_params);
        test_index(params.clone(), nlist, recall_requirement, None).await;
        if distance_type == DistanceType::Cosine {
            test_index_multivec(params.clone(), nlist, recall_requirement).await;
        }
        test_remap(params.clone(), nlist, recall_requirement).await;
    }

    #[rstest]
    #[case::fast(RQRotationType::Fast)]
    #[case::matrix(RQRotationType::Matrix)]
    #[tokio::test]
    async fn test_ivf_rq_rotation_type_after_optimize(#[case] rotation_type: RQRotationType) {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, _) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        let ivf_params = IvfBuildParams::new(4);
        let rq_params = RQBuildParams::with_rotation_type(1, rotation_type);
        let params = VectorIndexParams::with_ivf_rq_params(DistanceType::L2, ivf_params, rq_params);
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        assert_rq_rotation_type(&dataset, rotation_type).await;

        append_dataset::<Float32Type>(&mut dataset, 64, 0.0..1.0).await;
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();

        let indices_after_append = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_append.len(),
            2,
            "Expected append optimize to create one delta index"
        );
        assert_rq_rotation_type(&dataset, rotation_type).await;

        dataset
            .optimize_indices(&OptimizeOptions::merge(10))
            .await
            .unwrap();
        let indices_after_merge = dataset.load_indices().await.unwrap();
        assert_eq!(
            indices_after_merge.len(),
            1,
            "Expected merge optimize to merge indices into one"
        );
        assert_rq_rotation_type(&dataset, rotation_type).await;
    }

    #[rstest]
    #[case(4, DistanceType::L2, 0.9)]
    #[case(4, DistanceType::Cosine, 0.9)]
    #[case(4, DistanceType::Dot, 0.85)]
    #[case(4, DistanceType::Hamming, 0.9)]
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
        test_remap(params, nlist, recall_requirement).await;
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
        test_remap(params, nlist, recall_requirement).await;
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
        // PQ performs worse on farther vectors, so if we delete the many nearest vectors, the recall will be lower
        // lower the recall requirement in remap case for PQ, because it deletes half of the vectors
        test_remap(params, nlist, recall_requirement * 0.9).await;
    }

    #[rstest]
    // Temporarily disable recall checks for 4-bit PQ.
    #[case(4, DistanceType::L2, 0.0)]
    #[case(4, DistanceType::Cosine, 0.0)]
    #[case(4, DistanceType::Dot, 0.0)]
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
        dataset.checkout_latest().await.unwrap();
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

    /// Rewrite the auxiliary storage file to the legacy PQ format (codebook
    /// embedded in schema metadata rather than stored as a global buffer), then
    /// commit a `CreateIndex` transaction so the manifest records the correct
    /// new file size.
    /// Rewrite the auxiliary PQ storage file with the codebook inlined into
    /// schema metadata (legacy format). Uses a new UUID to avoid cache key
    /// collisions with the original index.
    async fn rewrite_pq_storage(dataset: &mut Dataset, old_meta: &IndexMetadata) -> Result<()> {
        use crate::dataset::transaction::{Operation, Transaction};

        let obj_store = Arc::new(ObjectStore::local());
        let old_dir = dataset.indices_dir().child(old_meta.uuid.to_string());
        let new_uuid = uuid::Uuid::new_v4();
        let new_dir = dataset.indices_dir().child(new_uuid.to_string());

        // Copy the main index file to the new directory unchanged.
        obj_store
            .copy(
                &old_dir.child(super::INDEX_FILE_NAME),
                &new_dir.child(super::INDEX_FILE_NAME),
            )
            .await?;

        // Read the original auxiliary file.
        let old_aux_path = old_dir.child(INDEX_AUXILIARY_FILE_NAME);
        let scheduler =
            ScanScheduler::new(obj_store.clone(), SchedulerConfig::default_for_testing());
        let reader = FileReader::try_open(
            scheduler
                .open_file(&old_aux_path, &CachedFileSize::unknown())
                .await?,
            None,
            Arc::<DecoderPlugins>::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await?;

        // Rewrite auxiliary file with PQ codebook inlined into schema metadata.
        let mut metadata = reader.schema().metadata.clone();
        let batch = reader
            .read_range(0..reader.num_rows() as usize, None)
            .await?;
        let new_aux_path = new_dir.child(INDEX_AUXILIARY_FILE_NAME);
        let mut writer = FileWriter::try_new(
            obj_store.create(&new_aux_path).await?,
            batch.schema_ref().as_ref().try_into()?,
            Default::default(),
        )?;
        writer.write_batch(&batch).await?;
        writer
            .add_global_buffer(reader.read_global_buffer(1).await?)
            .await?;
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

        // Build new IndexMetadata with the new UUID and file sizes.
        let new_files =
            lance_table::format::list_index_files_with_sizes(&obj_store, &new_dir).await?;
        let mut new_meta = old_meta.clone();
        new_meta.uuid = new_uuid;
        new_meta.files = Some(new_files);

        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![new_meta],
                removed_indices: vec![old_meta.clone()],
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await?;

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
    async fn test_remap_join_on_second_delta() {
        const INDEX_NAME: &str = "vector_idx";
        const BASE_ROWS_PER_PARTITION: usize = 3_000;
        const SMALL_APPEND_ROWS: usize = 64;
        let offsets = [-50.0, 50.0];

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (batch, schema) = generate_clustered_batch(BASE_ROWS_PER_PARTITION, offsets);
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let centroids = build_centroids_for_offsets(&offsets);
        let ivf_params = IvfBuildParams::try_with_centroids(2, centroids).unwrap();
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            ivf_params,
            PQBuildParams::default(),
        );
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

        let template_batch = dataset
            .take_rows(&[0], dataset.schema().clone())
            .await
            .unwrap();
        let template_values = template_batch["vector"]
            .as_fixed_size_list()
            .value(0)
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        let mut append_params = WriteParams {
            max_rows_per_file: 32,
            max_rows_per_group: 32,
            ..Default::default()
        };
        append_params.mode = WriteMode::Append;
        append_constant_vector_with_params(
            &mut dataset,
            SMALL_APPEND_ROWS,
            &template_values,
            Some(append_params),
        )
        .await;

        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        let stats_before: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics(INDEX_NAME).await.unwrap()).unwrap();
        assert_eq!(stats_before["num_indices"].as_u64().unwrap(), 2);
        let partitions_before: Vec<usize> = stats_before["indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|idx| idx["num_partitions"].as_u64().unwrap() as usize)
            .collect();
        assert_eq!(partitions_before.len(), 2);
        let base_partition_count = partitions_before
            .iter()
            .copied()
            .max()
            .expect("expected at least one partition count");
        assert!(base_partition_count >= 2);
        assert!(
            partitions_before
                .iter()
                .all(|count| *count == base_partition_count)
        );

        let indices_meta = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
        assert_eq!(indices_meta.len(), 2);

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 5_000,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        let mut dataset = Dataset::open(test_uri).await.unwrap();
        let stats_after_compaction: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics(INDEX_NAME).await.unwrap()).unwrap();
        assert_eq!(stats_after_compaction["num_indices"].as_u64().unwrap(), 2);
        let mut partitions_after: Vec<usize> = stats_after_compaction["indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|idx| idx["num_partitions"].as_u64().unwrap() as usize)
            .collect();
        partitions_after.sort_unstable();
        assert_eq!(
            partitions_after,
            vec![base_partition_count, base_partition_count]
        );

        const LARGE_APPEND_ROWS: usize = 40_000;
        append_constant_vector(&mut dataset, LARGE_APPEND_ROWS, &template_values).await;
        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let stats_after_split: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics(INDEX_NAME).await.unwrap()).unwrap();
        assert_eq!(stats_after_split["num_indices"].as_u64().unwrap(), 1);
        let final_partition_count = stats_after_split["indices"][0]["num_partitions"]
            .as_u64()
            .unwrap() as usize;
        assert_eq!(
            final_partition_count,
            base_partition_count + 1,
            "expected split to increase partitions beyond {}, got {}",
            base_partition_count,
            final_partition_count
        );
    }

    #[tokio::test]
    async fn test_spfresh_join_split() {
        // Two join cycles followed by three append cycles:
        // 1. Each deletion shrinks the smallest partition and verifies the partition count.
        // 2. Append #1 (10k rows) creates a delta index without splitting.
        // 3. Append #2 and #3 (40k rows each) trigger splits, forcing merges and validating partition sizes.

        const INDEX_NAME: &str = "vector_idx";
        const NLIST: usize = 3;
        const FIRST_APPEND_ROWS: usize = 10_000;
        const SECOND_APPEND_ROWS: usize = 30_000;
        const THIRD_APPEND_ROWS: usize = 35_000;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        // Two small clusters (for joins) and two large clusters (for splits).
        let cluster_sizes = [100, 4_000, 4_000];
        let total_rows: usize = cluster_sizes.iter().sum();

        let mut centroid_values = Vec::new();
        for i in 0..NLIST {
            for j in 0..DIM {
                centroid_values.push(if j == 0 { (i as f32) * 10.0 } else { 0.0 });
            }
        }
        let centroids = Arc::new(
            FixedSizeListArray::try_new_from_values(
                Float32Array::from(centroid_values),
                DIM as i32,
            )
            .unwrap(),
        );

        let mut ids = Vec::new();
        let mut vector_values = Vec::new();
        let mut current_id = 0u64;
        for (cluster_idx, &size) in cluster_sizes.iter().enumerate() {
            let centroid_base = (cluster_idx as f32) * 10.0;
            for _ in 0..size {
                ids.push(current_id);
                current_id += 1;
                for j in 0..DIM {
                    vector_values.push(if j == 0 {
                        centroid_base + (current_id % 100) as f32 * 0.005
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

        let ivf_params = IvfBuildParams::try_with_centroids(NLIST, centroids).unwrap();
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            ivf_params,
            PQBuildParams::default(),
        );
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

        // Template vector from the first large cluster for deterministic appends.
        let template_id = (cluster_sizes[0] + cluster_sizes[1]) as u64;
        let template_batch = dataset
            .take_rows(&[template_id], dataset.schema().clone())
            .await
            .unwrap();
        let template_values = template_batch["vector"]
            .as_fixed_size_list()
            .value(0)
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        assert_eq!(
            template_values.len(),
            DIM,
            "Template vector should match DIM"
        );

        let mut expected_partitions = NLIST;
        let mut expected_rows = total_rows;

        // Two join cycles.
        for expected_after in [NLIST - 1, NLIST - 2] {
            let (deleted_rows, appended_rows, actual_partitions) =
                shrink_smallest_partition(&mut dataset, INDEX_NAME, expected_after).await;
            expected_rows = expected_rows - deleted_rows + appended_rows;
            assert_eq!(
                dataset.count_all_rows().await.unwrap(),
                expected_rows,
                "Row count mismatch after join"
            );
            expected_partitions = actual_partitions;
        }

        // Append #1: no split, expect a delta index.
        let rows = FIRST_APPEND_ROWS;
        append_and_verify_append_phase(
            &mut dataset,
            INDEX_NAME,
            &template_values,
            rows,
            expected_partitions,
            expected_rows + rows,
            2,
            false,
        )
        .await;
        expected_rows += rows;

        // Append #2: triggers split and merge.
        expected_partitions += 1;
        let rows = SECOND_APPEND_ROWS;
        append_and_verify_append_phase(
            &mut dataset,
            INDEX_NAME,
            &template_values,
            rows,
            expected_partitions,
            expected_rows + rows,
            1,
            true,
        )
        .await;
        expected_rows += rows;

        // Append #3: triggers another split, remains a single merged index.
        expected_partitions += 1;
        let rows = THIRD_APPEND_ROWS;
        append_and_verify_append_phase(
            &mut dataset,
            INDEX_NAME,
            &template_values,
            rows,
            expected_partitions,
            expected_rows + rows,
            1,
            true,
        )
        .await;
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
        verify_partition_split_after_append(dataset, test_uri, params, "multivector data").await;
    }

    #[tokio::test]
    async fn test_split_multiple_partitions_in_one_optimize() {
        const INDEX_NAME: &str = "vector_idx";
        const BASE_ROWS_PER_PARTITION: usize = 512;
        const APPEND_ROWS_PER_PARTITION: usize = 40_000;
        let offsets = [-50.0, 50.0];

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (batch, schema) = generate_clustered_batch(BASE_ROWS_PER_PARTITION, offsets);
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let centroids = build_centroids_for_offsets(&offsets);
        let ivf_params = IvfBuildParams::try_with_centroids(2, centroids).unwrap();
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            ivf_params,
            PQBuildParams::default(),
        );
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
        assert_eq!(initial_ctx.num_partitions(), 2);
        let mut templates = Vec::with_capacity(2);
        for partition_idx in 0..2 {
            let row_ids = load_partition_row_ids(initial_ctx.ivf(), partition_idx).await;
            let template_batch = dataset
                .take_rows(&[row_ids[0]], dataset.schema().clone())
                .await
                .unwrap();
            templates.push(
                template_batch["vector"]
                    .as_fixed_size_list()
                    .value(0)
                    .as_primitive::<Float32Type>()
                    .values()
                    .to_vec(),
            );
        }

        append_partition_templates(&mut dataset, APPEND_ROWS_PER_PARTITION, &templates).await;

        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();
        dataset.validate().await.unwrap();

        let final_ctx = load_vector_index_context(&dataset, "vector", INDEX_NAME).await;
        assert_eq!(
            final_ctx.num_partitions(),
            4,
            "Expected both original partitions to split in one optimize, stats: {}",
            final_ctx.stats_json()
        );

        let indices = final_ctx.stats()["indices"]
            .as_array()
            .expect("indices should be present");
        assert_eq!(
            indices.len(),
            1,
            "Expected split optimize to merge into one index, stats: {}",
            final_ctx.stats_json()
        );

        let partitions = indices[0]["partitions"]
            .as_array()
            .expect("partitions should be present");
        assert_eq!(partitions.len(), 4);
        let expected_rows = 2 * BASE_ROWS_PER_PARTITION + 2 * APPEND_ROWS_PER_PARTITION;
        let total_partition_rows = partitions
            .iter()
            .map(|part| part["size"].as_u64().unwrap() as usize)
            .sum::<usize>();
        assert_eq!(total_partition_rows, expected_rows);
        assert_eq!(dataset.count_all_rows().await.unwrap(), expected_rows);

        let nearest = dataset
            .scan()
            .with_row_id()
            .nearest("vector", &Float32Array::from(templates[0].clone()), 10)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let ids = nearest[ROW_ID].as_primitive::<UInt64Type>();
        let mut seen = HashSet::new();
        for row_id in ids.values() {
            assert!(seen.insert(*row_id), "Duplicate row id found: {}", row_id);
        }
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

        const MULTIVEC_PER_ROW: usize = 3;
        let cluster_sizes = [4000, 4000, 400];
        let offsets: Vec<f32> = vec![0.0, 10.0, 20.0];
        let nlist = offsets.len();
        let mut dataset = {
            let (batch, schema) =
                generate_clustered_multivec_batch(&cluster_sizes, &offsets, MULTIVEC_PER_ROW);
            let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);
            Dataset::write(
                batches,
                test_uri,
                Some(WriteParams {
                    mode: crate::dataset::WriteMode::Overwrite,
                    ..Default::default()
                }),
            )
            .await
            .unwrap()
        };

        const SMALL_APPEND_FOR_JOIN: usize = 32;
        let centroids = build_centroids_for_offsets(&offsets);
        let ivf_params = IvfBuildParams::try_with_centroids(nlist, centroids).unwrap();
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::Cosine,
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

        // Verify initial partition count and record it for later comparison.
        let index_ctx = load_vector_index_context(&dataset, "vector", "vector_idx").await;
        let initial_partitions = index_ctx.num_partitions();
        assert!(
            initial_partitions <= nlist && initial_partitions > 1,
            "Expected at most {} partitions, got {}",
            nlist,
            initial_partitions
        );

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

        // Append a tiny batch and optimize incrementally to trigger the join path.
        append_dataset::<Float32Type>(&mut dataset, SMALL_APPEND_FOR_JOIN, 0.0..0.01).await;
        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();
        dataset
            // A second pass ensures the incremental index sees the reduced
            // partition sizes and applies the join.
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        // Verify partition count decreased after join
        let final_ctx = load_vector_index_context(&dataset, "vector", "vector_idx").await;
        let final_num_partitions = final_ctx.num_partitions();
        assert_le!(
            final_num_partitions,
            initial_partitions,
            "Partition count should drop after join, was {}, now {}",
            initial_partitions,
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

    #[tokio::test]
    async fn test_prewarm_ivf_pq() {
        use lance_io::assert_io_eq;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();
        let (mut dataset, _) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;

        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            IvfBuildParams::new(4),
            PQBuildParams::default(),
        );
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("my_idx".to_owned()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Reset IO stats after index creation
        dataset.object_store().io_stats_incremental();

        // Prewarm should perform IO to load all partitions into cache
        dataset.prewarm_index("my_idx").await.unwrap();
        let stats = dataset.object_store().io_stats_incremental();
        assert!(
            stats.read_iops > 0,
            "prewarm should have read from disk, but read_iops was 0"
        );

        // Can query index without IO
        let q = Float32Array::from_iter_values(repeat_n(0.0, DIM));
        dataset
            .scan()
            .nearest("vector", &q, 10)
            .unwrap()
            .project(&["_rowid"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let stats = dataset.object_store().io_stats_incremental();
        assert_io_eq!(
            stats,
            read_iops,
            0,
            "query should not perform IO after prewarm"
        );

        // Second prewarm should not need IO (already cached)
        dataset.prewarm_index("my_idx").await.unwrap();
        let stats = dataset.object_store().io_stats_incremental();
        assert_io_eq!(stats, read_iops, 0, "second prewarm should not perform IO");
    }

    #[tokio::test]
    async fn test_prewarm_ivf_pq_multiple_deltas() {
        use lance_io::assert_io_eq;

        const INDEX_NAME: &str = "my_idx";
        const BASE_ROWS_PER_PARTITION: usize = 3_000;
        const SMALL_APPEND_ROWS: usize = 64;
        let offsets = [-50.0, 50.0];

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let (batch, schema) = generate_clustered_batch(BASE_ROWS_PER_PARTITION, offsets);
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let centroids = build_centroids_for_offsets(&offsets);
        let ivf_params = IvfBuildParams::try_with_centroids(2, centroids).unwrap();
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            ivf_params,
            PQBuildParams::default(),
        );
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

        let template_batch = dataset
            .take_rows(&[0], dataset.schema().clone())
            .await
            .unwrap();
        let template_values = template_batch["vector"]
            .as_fixed_size_list()
            .value(0)
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        let mut append_params = WriteParams {
            max_rows_per_file: 32,
            max_rows_per_group: 32,
            ..Default::default()
        };
        append_params.mode = WriteMode::Append;
        append_constant_vector_with_params(
            &mut dataset,
            SMALL_APPEND_ROWS,
            &template_values,
            Some(append_params),
        )
        .await;

        dataset
            .optimize_indices(&OptimizeOptions::new())
            .await
            .unwrap();

        // Reopen dataset to avoid carrying index state in-memory from index creation.
        let dataset = Dataset::open(test_uri).await.unwrap();
        let indices = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
        assert_eq!(indices.len(), 2, "expected two index deltas for my_idx");
        let unique_uuids: HashSet<_> = indices.iter().map(|meta| meta.uuid).collect();
        assert_eq!(unique_uuids.len(), 2, "expected two unique index UUIDs");

        // Reset IO stats after index creation
        dataset.object_store().io_stats_incremental();

        // Prewarm should perform IO to load all index deltas into cache
        dataset.prewarm_index(INDEX_NAME).await.unwrap();
        let stats = dataset.object_store().io_stats_incremental();
        assert!(
            stats.read_iops > 0,
            "prewarm should have read from disk, but read_iops was 0"
        );

        // Query should not perform IO after prewarm of all deltas
        let q = Float32Array::from(template_values.clone());
        dataset
            .scan()
            .nearest("vector", &q, 10)
            .unwrap()
            .project(&["_rowid"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let stats = dataset.object_store().io_stats_incremental();
        assert_io_eq!(
            stats,
            read_iops,
            0,
            "query should not perform IO after prewarm"
        );

        // Second prewarm should not need IO (already cached)
        dataset.prewarm_index(INDEX_NAME).await.unwrap();
        let stats = dataset.object_store().io_stats_incremental();
        assert_io_eq!(stats, read_iops, 0, "second prewarm should not perform IO");
    }

    type SerializedEntry = (Vec<u8>, lance_core::cache::CacheCodec, usize);

    #[derive(Debug)]
    struct SerializingBackend {
        /// Serialized entries: key -> (bytes, codec, size).
        serialized: tokio::sync::Mutex<
            std::collections::HashMap<lance_core::cache::InternalCacheKey, SerializedEntry>,
        >,
        /// Fallback for entries without a codec.
        passthrough: lance_core::cache::MokaCacheBackend,
    }

    impl SerializingBackend {
        fn new() -> Self {
            Self {
                serialized: tokio::sync::Mutex::new(std::collections::HashMap::new()),
                passthrough: lance_core::cache::MokaCacheBackend::with_capacity(256 * 1024 * 1024),
            }
        }

        async fn serialized_entry_count(&self) -> usize {
            self.serialized.lock().await.len()
        }

        async fn passthrough_entry_count(&self) -> usize {
            use lance_core::cache::CacheBackend;
            self.passthrough.num_entries().await
        }
    }

    #[async_trait::async_trait]
    impl lance_core::cache::CacheBackend for SerializingBackend {
        async fn get(
            &self,
            key: &lance_core::cache::InternalCacheKey,
            codec: Option<lance_core::cache::CacheCodec>,
        ) -> Option<lance_core::cache::CacheEntry> {
            // Try serialized store first
            let guard = self.serialized.lock().await;
            if let Some((bytes, stored_codec, _)) = guard.get(key) {
                return Some(
                    stored_codec
                        .deserialize(&bytes::Bytes::copy_from_slice(bytes))
                        .expect("deserialization should succeed"),
                );
            }
            drop(guard);
            // Fall through to passthrough
            self.passthrough.get(key, codec).await
        }

        async fn insert(
            &self,
            key: &lance_core::cache::InternalCacheKey,
            entry: lance_core::cache::CacheEntry,
            size_bytes: usize,
            codec: Option<lance_core::cache::CacheCodec>,
        ) {
            if let Some(codec) = codec {
                let mut bytes = Vec::new();
                codec
                    .serialize(&entry, &mut bytes)
                    .expect("serialization should succeed");
                self.serialized
                    .lock()
                    .await
                    .insert(key.clone(), (bytes, codec, size_bytes));
            } else {
                self.passthrough.insert(key, entry, size_bytes, None).await;
            }
        }

        async fn get_or_insert<'a>(
            &self,
            key: &lance_core::cache::InternalCacheKey,
            loader: std::pin::Pin<
                Box<
                    dyn futures::Future<Output = Result<(lance_core::cache::CacheEntry, usize)>>
                        + Send
                        + 'a,
                >,
            >,
            codec: Option<lance_core::cache::CacheCodec>,
        ) -> Result<(lance_core::cache::CacheEntry, bool)> {
            if let Some(entry) = self.get(key, codec).await {
                return Ok((entry, true));
            }
            let (entry, size) = loader.await?;
            self.insert(key, entry.clone(), size, codec).await;
            Ok((entry, false))
        }

        async fn invalidate_prefix(&self, prefix: &str) {
            self.serialized
                .lock()
                .await
                .retain(|k, _| !k.starts_with(prefix));
            self.passthrough.invalidate_prefix(prefix).await;
        }

        async fn clear(&self) {
            self.serialized.lock().await.clear();
            self.passthrough.clear().await;
        }

        async fn num_entries(&self) -> usize {
            self.serialized.lock().await.len() + self.passthrough.num_entries().await
        }

        async fn size_bytes(&self) -> usize {
            let serialized: usize = self
                .serialized
                .lock()
                .await
                .values()
                .map(|(_, _, s)| *s)
                .sum();
            serialized + self.passthrough.size_bytes().await
        }
    }

    /// Integration test: create a vector index, prewarm it through a
    /// serializing cache backend, then query. Verifies that entries are
    /// serialized to bytes and that queries produce correct results after
    /// deserialization.
    #[tokio::test]
    async fn test_prewarm_and_query_with_serializing_backend() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        // Create dataset with vector index using default cache
        let (mut dataset, _) = generate_test_dataset::<Float32Type>(test_uri, 0.0..1.0).await;
        let params = VectorIndexParams::with_ivf_pq_params(
            DistanceType::L2,
            IvfBuildParams::new(4),
            PQBuildParams::default(),
        );
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("serde_idx".to_owned()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Re-open with the serializing backend
        let backend = Arc::new(SerializingBackend::new());
        let session = Arc::new(crate::session::Session::with_index_cache_backend(
            backend.clone(),
            128 * 1024 * 1024,
            Arc::new(lance_io::object_store::ObjectStoreRegistry::default()),
        ));
        let dataset = crate::DatasetBuilder::from_uri(test_uri)
            .with_session(session)
            .load()
            .await
            .unwrap();

        // Prewarm — this should serialize entries into the backend
        dataset.prewarm_index("serde_idx").await.unwrap();
        let serialized = backend.serialized_entry_count().await;
        let passthrough = backend.passthrough_entry_count().await;
        assert!(
            serialized > 0,
            "prewarm should have serialized entries into the backend"
        );
        assert_eq!(
            passthrough, 0,
            "all index cache entries should have codecs (nothing in passthrough), \
             but found {passthrough} passthrough entries"
        );

        // Query — the backend will deserialize entries from bytes.
        // After prewarm, all entries are in serialized form, so every
        // cache hit involves a deserialization round-trip.
        let q = Float32Array::from_iter_values(repeat_n(0.5, DIM));
        let results = dataset
            .scan()
            .nearest("vector", &q, 10)
            .unwrap()
            .nprobes(4)
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 10, "should return 10 nearest neighbors");

        // Verify distances are sorted (ascending for L2)
        let distances: Vec<f32> = results
            .column_by_name("_distance")
            .unwrap()
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        for w in distances.windows(2) {
            assert!(w[1] >= w[0], "distances should be sorted ascending");
        }
    }
}
