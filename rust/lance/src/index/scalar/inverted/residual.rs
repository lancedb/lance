// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Query-time immutable FTS segments for append-only residual fragments.
//!
//! This mirrors Lucene's near-real-time model: newly appended fragments are
//! converted into the same immutable posting format as committed FTS segments.
//! The segment, rather than a query result, is cached and can therefore serve
//! arbitrary exact compound queries until the fragment or index configuration
//! changes.

use std::{
    borrow::Cow,
    fmt::Display,
    ops::Range,
    sync::{
        Arc, LazyLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use bytes::Bytes;
use futures::{StreamExt, TryStreamExt, stream::BoxStream};
use lance_core::{
    Result,
    cache::{CacheKey, CacheKeySchema, KeyBuilder, LanceCache},
    deepsize::{Context, DeepSizeOf},
};
use lance_index::{
    progress::noop_progress,
    scalar::{
        TrainingCriteria, TrainingOrdering,
        inverted::{InvertedIndex, InvertedIndexBuilder, InvertedIndexParams},
        lance_format::LanceIndexStore,
    },
};
use lance_io::object_store::ObjectStore as LanceObjectStore;
use lance_table::format::{Fragment, IndexMetadata};
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore as OsObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    RenameOptions, Result as OsResult, UploadPart, memory::InMemory, path::Path,
};
use tokio::sync::Semaphore;
use uuid::Uuid;

use super::ResolvedFtsField;
use crate::{Dataset, index::DatasetIndexInternalExt, index::scalar::load_fts_training_data};

/// Bound query-time indexing without serializing unrelated datasets behind one
/// long build. Same-key work is additionally coalesced by `DSIndexCache`.
static RESIDUAL_FTS_BUILD_SEMAPHORE: LazyLock<Semaphore> = LazyLock::new(|| {
    let permits = std::thread::available_parallelism()
        .map(|cpus| cpus.get().div_ceil(4))
        .unwrap_or(1)
        .clamp(1, 2);
    Semaphore::new(permits)
});

pub(crate) const MAX_RESIDUAL_FTS_FRAGMENTS: usize = 16;
pub(crate) const MAX_RESIDUAL_FTS_ROWS: usize = 1_000_000;
const MAX_RESIDUAL_FTS_BUILD_MEMORY_MB: u64 = 256;
const MAX_RESIDUAL_FTS_SERIALIZED_BYTES: usize = 1 << 30;
const MAX_RESIDUAL_FTS_GROUP_UPLOAD_BYTES: usize = 2 << 30;
const RESIDUAL_FTS_FRAGMENT_BUILD_CONCURRENCY: usize = 2;

#[derive(Debug)]
struct BudgetedMemoryStore {
    inner: InMemory,
    uploaded_bytes: Arc<AtomicUsize>,
    budget_exceeded: Arc<AtomicBool>,
    max_uploaded_bytes: usize,
}

impl BudgetedMemoryStore {
    #[cfg(test)]
    fn new(max_uploaded_bytes: usize) -> Self {
        Self::with_counter(
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicBool::new(false)),
            max_uploaded_bytes,
        )
    }

    fn with_counter(
        uploaded_bytes: Arc<AtomicUsize>,
        budget_exceeded: Arc<AtomicBool>,
        max_uploaded_bytes: usize,
    ) -> Self {
        Self {
            inner: InMemory::new(),
            uploaded_bytes,
            budget_exceeded,
            max_uploaded_bytes,
        }
    }

    fn reserve(&self, bytes: usize) -> OsResult<()> {
        reserve_upload_bytes(
            &self.uploaded_bytes,
            &self.budget_exceeded,
            self.max_uploaded_bytes,
            bytes,
        )
    }

    async fn reserve_source(&self, source: &Path) -> OsResult<()> {
        let source_size = self.inner.head(source).await?.size;
        let source_size =
            usize::try_from(source_size).map_err(|_| object_store::Error::Generic {
                store: "BudgetedMemoryStore",
                source: format!(
                    "source object {source} has size {source_size}, which does not fit usize"
                )
                .into(),
            })?;
        self.reserve(source_size)
    }
}

fn reserve_upload_bytes(
    uploaded_bytes: &AtomicUsize,
    budget_exceeded: &AtomicBool,
    max_uploaded_bytes: usize,
    bytes: usize,
) -> OsResult<()> {
    uploaded_bytes
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current
                .checked_add(bytes)
                .filter(|next| *next <= max_uploaded_bytes)
        })
        .map(|_| ())
        .map_err(|_| {
            budget_exceeded.store(true, Ordering::Relaxed);
            object_store::Error::Generic {
                store: "BudgetedMemoryStore",
                source: format!(
                    "residual FTS serialized output exceeded the {max_uploaded_bytes} byte build budget"
                )
                .into(),
            }
        })
}

impl Display for BudgetedMemoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BudgetedMemoryStore")
    }
}

#[async_trait]
impl OsObjectStore for BudgetedMemoryStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> OsResult<PutResult> {
        self.reserve(payload.content_length())?;
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OsResult<Box<dyn MultipartUpload>> {
        let inner = self.inner.put_multipart_opts(location, opts).await?;
        Ok(Box::new(BudgetedMultipartUpload {
            inner,
            uploaded_bytes: self.uploaded_bytes.clone(),
            budget_exceeded: self.budget_exceeded.clone(),
            max_uploaded_bytes: self.max_uploaded_bytes,
        }))
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OsResult<GetResult> {
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OsResult<Vec<Bytes>> {
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, OsResult<Path>>,
    ) -> BoxStream<'static, OsResult<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OsResult<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, OsResult<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OsResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OsResult<()> {
        // `InMemory::copy_opts` creates another logical object and is used by
        // `merge_index_files` to publish staged partitions. Charge the source
        // before the copy so this path cannot bypass the group-wide hard cap.
        self.reserve_source(from).await?;
        self.inner.copy_opts(from, to, opts).await
    }

    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> OsResult<()> {
        // `InMemory` inherits ObjectStore's copy-then-delete rename. Its Bytes
        // payload is shared, but the operation is not an atomic map move, so
        // conservatively charge the source just like copy.
        self.reserve_source(from).await?;
        self.inner.rename_opts(from, to, opts).await
    }
}

#[derive(Debug)]
struct BudgetedMultipartUpload {
    inner: Box<dyn MultipartUpload>,
    uploaded_bytes: Arc<AtomicUsize>,
    budget_exceeded: Arc<AtomicBool>,
    max_uploaded_bytes: usize,
}

#[async_trait]
impl MultipartUpload for BudgetedMultipartUpload {
    fn put_part(&mut self, payload: PutPayload) -> UploadPart {
        if let Err(error) = reserve_upload_bytes(
            &self.uploaded_bytes,
            &self.budget_exceeded,
            self.max_uploaded_bytes,
            payload.content_length(),
        ) {
            return Box::pin(async move { Err(error) });
        }
        self.inner.put_part(payload)
    }

    async fn complete(&mut self) -> OsResult<PutResult> {
        self.inner.complete().await
    }

    async fn abort(&mut self) -> OsResult<()> {
        self.inner.abort().await
    }
}

/// Stable identity for one query-time FTS segment.
///
/// The dataset URI is already a namespace of `DSIndexCache`. The remaining
/// fields deliberately omit dataset version so an unchanged fragment is reused
/// after another append. The complete serialized fragment metadata invalidates
/// data rewrites, overlays, deletions, and row-id metadata changes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResidualFtsFragmentKey {
    pub store_identity: Arc<str>,
    pub index_uuid: Uuid,
    pub index_version: i32,
    pub fragment_id: u64,
    pub fragment_fingerprint: Arc<[u8]>,
    pub field_id: i32,
    pub canonical_path: Arc<str>,
    pub field_type: Arc<str>,
    pub params_fingerprint: Arc<[u8]>,
}

impl CacheKey for ResidualFtsFragmentKey {
    type ValueType = CachedResidualFtsEntry;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "residual-fts/{}/{}/{}/{}",
            self.index_uuid, self.index_version, self.field_id, self.fragment_id
        ))
    }

    fn type_name() -> &'static str {
        "CachedResidualFtsEntry"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.fts-residual-fragment-key", 2)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(&self.store_identity);
        builder.write_fixed_bytes(self.index_uuid.as_bytes());
        builder.write_i32(self.index_version);
        builder.write_u64(self.fragment_id);
        builder.write_bytes(&self.fragment_fingerprint);
        builder.write_i32(self.field_id);
        builder.write_str(&self.canonical_path);
        builder.write_str(&self.field_type);
        builder.write_bytes(&self.params_fingerprint);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResidualFtsGroupKey {
    store_identity: Arc<str>,
    index_uuid: Uuid,
    index_version: i32,
    members: Arc<[ResidualFtsFragmentKey]>,
}

impl CacheKey for ResidualFtsGroupKey {
    type ValueType = ResidualFtsGroupState;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "residual-fts-group/{}/{}/{}",
            self.index_uuid,
            self.index_version,
            self.members.len()
        ))
    }

    fn type_name() -> &'static str {
        "ResidualFtsGroupState"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.fts-residual-group-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(&self.store_identity);
        builder.write_fixed_bytes(self.index_uuid.as_bytes());
        builder.write_i32(self.index_version);
        builder.write_u64(self.members.len() as u64);
        for member in self.members.iter() {
            member.write_key(builder);
        }
    }
}

#[derive(Debug)]
enum ResidualFtsGroupState {
    Seen,
    Rejected,
}

impl DeepSizeOf for ResidualFtsGroupState {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

#[derive(Clone, Debug)]
struct ResidualFtsRetentionProbeKey(ResidualFtsGroupKey);

impl CacheKey for ResidualFtsRetentionProbeKey {
    type ValueType = ResidualFtsRetentionProbe;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("retention-probe/{}", self.0.key()))
    }

    fn type_name() -> &'static str {
        "ResidualFtsRetentionProbe"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.fts-residual-retention-probe-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        self.0.write_key(builder);
    }
}

#[derive(Debug)]
struct ResidualFtsRetentionProbe;

impl DeepSizeOf for ResidualFtsRetentionProbe {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

#[derive(Debug)]
pub(crate) enum CachedResidualFtsEntry {
    Ready(CachedResidualFtsSegment),
}

impl DeepSizeOf for CachedResidualFtsEntry {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        match self {
            Self::Ready(segment) => segment.deep_size_of_children(context),
        }
    }
}

/// A standard immutable FTS segment plus the memory-store bytes backing it.
#[derive(Debug)]
pub(crate) struct CachedResidualFtsSegment {
    index: Arc<InvertedIndex>,
    rows: usize,
    documents: usize,
    serialized_bytes: usize,
}

impl CachedResidualFtsSegment {
    pub fn index(&self) -> Arc<InvertedIndex> {
        self.index.clone()
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn documents(&self) -> usize {
        self.documents
    }

    pub fn serialized_bytes(&self) -> usize {
        self.serialized_bytes
    }

    pub fn resident_bytes(&self) -> usize {
        let mut context = Context::default();
        self.deep_size_of_children(&mut context)
    }
}

impl DeepSizeOf for CachedResidualFtsSegment {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        // `InvertedIndex` intentionally does not charge its object store. The
        // store is private to this cache value, so account its serialized files
        // explicitly in addition to decoded partition state.
        self.index.deep_size_of_children(context) + self.serialized_bytes
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ResidualFtsCacheStats {
    pub loader_runs: usize,
    pub reuse_or_coalesced: usize,
    pub build_failures: usize,
    pub build_duration: Duration,
    pub rows: usize,
    pub documents: usize,
    pub serialized_bytes: usize,
    pub resident_bytes: usize,
}

#[derive(Debug)]
pub(crate) struct LoadedResidualFtsSegments {
    pub segments: Vec<Arc<InvertedIndex>>,
    pub fragment_bitmap: roaring::RoaringBitmap,
    pub stats: ResidualFtsCacheStats,
}

#[derive(Debug)]
pub(crate) enum ResidualFtsAdmission {
    Eligible(LoadedResidualFtsSegments),
    Deferred(&'static str),
    Rejected(&'static str),
}

/// Everything execution needs to probe or construct one exact residual group.
/// Construction is metadata-only; it never scans fragment data.
#[derive(Clone, Debug)]
pub(crate) struct ResidualFtsSpec {
    logical_index: IndexMetadata,
    column: Arc<str>,
    committed_segment_uuid: Uuid,
    fragments: Arc<[Fragment]>,
    resolved: ResolvedFtsField,
    params: InvertedIndexParams,
    keys: Arc<[ResidualFtsFragmentKey]>,
    group_key: ResidualFtsGroupKey,
    fragment_bitmap: roaring::RoaringBitmap,
}

fn fragment_fingerprint(fragment: &Fragment) -> Result<Arc<[u8]>> {
    Ok(Arc::from(serde_json::to_vec(fragment)?))
}

fn params_fingerprint(params: &InvertedIndexParams) -> Result<Arc<[u8]>> {
    Ok(Arc::from(serde_json::to_vec(&params.to_training_json()?)?))
}

fn residual_key(
    dataset: &Dataset,
    index: &IndexMetadata,
    fragment: &Fragment,
    resolved: &ResolvedFtsField,
    params_fingerprint: Arc<[u8]>,
) -> Result<ResidualFtsFragmentKey> {
    let field = dataset
        .schema()
        .field_by_id(resolved.final_field_id)
        .ok_or_else(|| {
            lance_core::Error::invalid_input(format!(
                "residual FTS field id {} is missing from the dataset schema",
                resolved.final_field_id
            ))
        })?;
    Ok(ResidualFtsFragmentKey {
        store_identity: Arc::from(dataset.object_store.store_prefix.as_str()),
        index_uuid: index.uuid,
        index_version: index.index_version,
        fragment_id: fragment.id,
        fragment_fingerprint: fragment_fingerprint(fragment)?,
        field_id: resolved.final_field_id,
        canonical_path: Arc::from(resolved.canonical_path.as_str()),
        field_type: Arc::from(field.data_type().to_string()),
        params_fingerprint,
    })
}

impl ResidualFtsSpec {
    pub(crate) fn try_new(
        dataset: &Dataset,
        logical_index: &IndexMetadata,
        column: &str,
        committed_segment_uuid: Uuid,
        fragments: &[Fragment],
        resolved: ResolvedFtsField,
        params: InvertedIndexParams,
    ) -> Result<std::result::Result<Self, &'static str>> {
        if params.get_document_granularity().is_list_element() {
            return Ok(Err("non-row document granularity"));
        }
        // The model tokenizers are opaque to DeepSizeOf. Until they expose an
        // exact retained size, do not let a residual Arc keep an uncharged
        // model alive after the committed index is evicted.
        if params.uses_external_language_model() {
            return Ok(Err(
                "external language-model tokenizer is not cache-accountable",
            ));
        }
        if fragments.is_empty() {
            return Ok(Err("empty residual fragment group"));
        }
        if fragments.len() > MAX_RESIDUAL_FTS_FRAGMENTS {
            return Ok(Err("too many residual fragments"));
        }
        if fragments
            .iter()
            .any(|fragment| !fragment.overlays.is_empty())
        {
            return Ok(Err("residual fragment has overlays"));
        }
        if fragments
            .iter()
            .any(|fragment| fragment.deletion_file.is_some())
        {
            return Ok(Err("residual fragment has deletions"));
        }
        let Some(total_rows) = fragments.iter().try_fold(0_usize, |total, fragment| {
            fragment
                .physical_rows
                .and_then(|rows| total.checked_add(rows))
        }) else {
            return Ok(Err("unknown or overflowing residual row count"));
        };
        if total_rows > MAX_RESIDUAL_FTS_ROWS {
            return Ok(Err("too many residual rows"));
        }
        let fragment_bitmap = fragments
            .iter()
            .map(|fragment| u32::try_from(fragment.id))
            .collect::<std::result::Result<roaring::RoaringBitmap, _>>()
            .map_err(|_| {
                lance_core::Error::invalid_input(
                    "residual FTS fragment id does not fit u32".to_string(),
                )
            })?;

        let params_fingerprint = params_fingerprint(&params)?;
        let keys = fragments
            .iter()
            .map(|fragment| {
                residual_key(
                    dataset,
                    logical_index,
                    fragment,
                    &resolved,
                    params_fingerprint.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let keys: Arc<[ResidualFtsFragmentKey]> = Arc::from(keys);
        let group_key = ResidualFtsGroupKey {
            store_identity: Arc::from(dataset.object_store.store_prefix.as_str()),
            index_uuid: logical_index.uuid,
            index_version: logical_index.index_version,
            members: keys.clone(),
        };
        Ok(Ok(Self {
            logical_index: logical_index.clone(),
            column: Arc::from(column),
            committed_segment_uuid,
            fragments: Arc::from(fragments.to_vec()),
            resolved,
            params,
            keys,
            group_key,
            fragment_bitmap,
        }))
    }

    pub(crate) fn fragment_bitmap(&self) -> roaring::RoaringBitmap {
        self.fragment_bitmap.clone()
    }
}

async fn serialized_store_size(object_store: &LanceObjectStore, index_dir: &Path) -> Result<usize> {
    let mut entries = object_store.read_dir_all(index_dir, None);
    let mut total = 0_u64;
    while let Some(entry) = entries.next().await {
        total = total.checked_add(entry?.size).ok_or_else(|| {
            lance_core::Error::io(format!(
                "residual FTS serialized byte count overflowed for {index_dir}"
            ))
        })?;
    }
    usize::try_from(total).map_err(|_| {
        lance_core::Error::io(format!(
            "residual FTS serialized size {total} does not fit usize for {index_dir}"
        ))
    })
}

async fn build_residual_segment(
    dataset: &Dataset,
    fragment: Fragment,
    resolved: &ResolvedFtsField,
    params: InvertedIndexParams,
    shared_tokenizer: Arc<dyn lance_index::scalar::inverted::document_tokenizer::LanceTokenizer>,
    group_uploaded_bytes: Arc<AtomicUsize>,
    group_budget_exceeded: Arc<AtomicBool>,
) -> Result<CachedResidualFtsSegment> {
    let fragment_id = u32::try_from(fragment.id).map_err(|_| {
        lance_core::Error::invalid_input(format!(
            "residual FTS fragment id {} does not fit u32",
            fragment.id
        ))
    })?;
    let rows = fragment.physical_rows.ok_or_else(|| {
        lance_core::Error::invalid_input(format!(
            "residual FTS fragment {} has unknown physical row count",
            fragment.id
        ))
    })?;
    let stream = load_fts_training_data(
        dataset,
        resolved,
        &TrainingCriteria::new(TrainingOrdering::None).with_row_id(),
        Some(vec![fragment]),
        true,
        None,
    )
    .await?;

    let mut object_store = LanceObjectStore::memory();
    object_store.inner = Arc::new(BudgetedMemoryStore::with_counter(
        group_uploaded_bytes,
        group_budget_exceeded.clone(),
        MAX_RESIDUAL_FTS_GROUP_UPLOAD_BYTES,
    ));
    let object_store = Arc::new(object_store);
    let index_dir = Path::from(format!("residual-{fragment_id}"));
    let private_cache = Arc::new(LanceCache::no_cache());
    let store = Arc::new(LanceIndexStore::new(
        object_store.clone(),
        index_dir.clone(),
        private_cache.clone(),
    ));
    // Bound the builder's working set in addition to the row admission limit.
    // The serialized store is checked before the value can enter the cache.
    let params = params
        .memory_limit_mb(MAX_RESIDUAL_FTS_BUILD_MEMORY_MB)
        .num_workers(1);
    let mut builder =
        InvertedIndexBuilder::new_with_fragment_mask(params, Some(u64::from(fragment_id) << 32));
    builder.update(stream, store.as_ref(), None).await?;
    lance_index::scalar::inverted::builder::merge_index_files(
        object_store.as_ref(),
        &index_dir,
        store.clone(),
        noop_progress(),
    )
    .await?;
    let serialized_bytes = serialized_store_size(object_store.as_ref(), &index_dir).await?;
    if serialized_bytes > MAX_RESIDUAL_FTS_SERIALIZED_BYTES {
        group_budget_exceeded.store(true, Ordering::Relaxed);
        return Err(lance_core::Error::io(format!(
            "residual FTS fragment {fragment_id} produced {serialized_bytes} serialized bytes, exceeding the {} byte build budget",
            MAX_RESIDUAL_FTS_SERIALIZED_BYTES
        )));
    }
    let index = InvertedIndex::load_with_shared_tokenizer(
        store,
        None,
        private_cache.as_ref(),
        shared_tokenizer,
    )
    .await?;
    // The private readers use a no-cache backend, so decoded posting groups
    // cannot accumulate outside this value's cache weight. Materialize the
    // lazy metadata and document state that the index itself does retain before
    // DSIndexCache computes its fixed admission weight.
    index.materialize_cache_weight().await?;
    let (_, documents, _) = index.bm25_stats_for_terms(&[], None).await?;
    Ok(CachedResidualFtsSegment {
        index,
        rows,
        documents,
        serialized_bytes,
    })
}

/// Load or build one immutable posting segment for each append-only fragment.
///
/// Deterministically unsupported shapes return `Rejected`; callers should keep
/// the existing exact flat fallback. I/O/build errors are returned so callers
/// can log them and fall back without poisoning the single-flight cache.
pub(crate) async fn load_residual_fts_segments(
    dataset: &Dataset,
    spec: &ResidualFtsSpec,
) -> Result<ResidualFtsAdmission> {
    let cache = dataset
        .index_cache
        .for_index(&spec.logical_index.uuid, None)
        .with_key_prefix("residual-fts");
    // Usage and negative-admission state must not compete with the posting
    // entries whose aggregate pressure it records.
    let group_cache = dataset
        .metadata_cache
        .with_key_prefix("residual-fts-groups");

    // A completely warm group can bypass the usage marker and build permit.
    let mut warm_entries = Vec::with_capacity(spec.keys.len());
    for key in spec.keys.iter() {
        let Some(entry) = cache.get_with_key(key).await else {
            warm_entries.clear();
            break;
        };
        warm_entries.push(entry);
    }
    if warm_entries.len() == spec.keys.len() {
        return Ok(ResidualFtsAdmission::Eligible(loaded_segments(
            warm_entries,
            spec.fragment_bitmap(),
            spec.keys.len(),
            0,
            Duration::default(),
        )));
    }

    match group_cache.get_with_key(&spec.group_key).await {
        Some(state) if matches!(state.as_ref(), ResidualFtsGroupState::Rejected) => {
            return Ok(ResidualFtsAdmission::Rejected(
                "residual working set exceeds cache admission capacity",
            ));
        }
        Some(_) => {}
        None => {
            group_cache
                .insert_with_key(&spec.group_key, Arc::new(ResidualFtsGroupState::Seen))
                .await;
            if group_cache.get_with_key(&spec.group_key).await.is_none() {
                return Ok(ResidualFtsAdmission::Rejected(
                    "residual metadata cache cannot retain the working-set marker",
                ));
            }
            return Ok(ResidualFtsAdmission::Deferred(
                "first use records residual working set",
            ));
        }
    }

    // Busy means exact fallback, not head-of-line blocking across datasets.
    let Ok(_permit) = RESIDUAL_FTS_BUILD_SEMAPHORE.try_acquire() else {
        return Ok(ResidualFtsAdmission::Deferred(
            "residual build resources are busy",
        ));
    };

    // The independent group state can be available while the index cache is
    // disabled. Probe posting-entry retention before scanning any fragment.
    let retention_probe_key = ResidualFtsRetentionProbeKey(spec.group_key.clone());
    cache
        .insert_with_key(&retention_probe_key, Arc::new(ResidualFtsRetentionProbe))
        .await;
    if cache.get_with_key(&retention_probe_key).await.is_none() {
        group_cache
            .insert_with_key(&spec.group_key, Arc::new(ResidualFtsGroupState::Rejected))
            .await;
        return Ok(ResidualFtsAdmission::Rejected(
            "residual index cache cannot retain posting entries",
        ));
    }

    let committed_index = dataset
        .open_scalar_index(
            spec.column.as_ref(),
            &spec.committed_segment_uuid,
            &lance_index::metrics::NoOpMetricsCollector,
        )
        .await?;
    let committed_index = committed_index
        .as_any()
        .downcast_ref::<InvertedIndex>()
        .ok_or_else(|| {
            lance_core::Error::internal(format!(
                "residual FTS source segment {} is not an inverted index",
                spec.committed_segment_uuid
            ))
        })?;
    let shared_tokenizer = committed_index.shared_tokenizer();
    let group_uploaded_bytes = Arc::new(AtomicUsize::new(0));
    let group_budget_exceeded = Arc::new(AtomicBool::new(false));

    let started = std::time::Instant::now();
    let entries = futures::stream::iter(
        spec.keys
            .iter()
            .cloned()
            .zip(spec.fragments.iter().cloned())
            .map(|(key, fragment)| {
                let cache = cache.clone();
                let resolved = spec.resolved.clone();
                let params = spec.params.clone();
                let shared_tokenizer = shared_tokenizer.clone();
                let group_uploaded_bytes = group_uploaded_bytes.clone();
                let group_budget_exceeded = group_budget_exceeded.clone();
                async move {
                    cache
                        .get_or_insert_with_key_hit(key.clone(), || async move {
                            build_residual_segment(
                                dataset,
                                fragment,
                                &resolved,
                                params,
                                shared_tokenizer,
                                group_uploaded_bytes,
                                group_budget_exceeded,
                            )
                            .await
                            .map(CachedResidualFtsEntry::Ready)
                        })
                        .await
                        .map(|(entry, reused)| (key, entry, reused))
                }
            }),
    )
    .buffered(RESIDUAL_FTS_FRAGMENT_BUILD_CONCURRENCY)
    .try_collect::<Vec<_>>()
    .await;
    let entries = match entries {
        Ok(entries) => entries,
        Err(_) if group_budget_exceeded.load(Ordering::Relaxed) => {
            group_cache
                .insert_with_key(&spec.group_key, Arc::new(ResidualFtsGroupState::Rejected))
                .await;
            return Ok(ResidualFtsAdmission::Rejected(
                "residual working set exceeds query-time build byte budget",
            ));
        }
        Err(error) => return Err(error),
    };

    // Verify the whole working set after all insertions. Per-entry success is
    // insufficient for a sharded cache: later siblings may already have
    // evicted an earlier one from the same shard.
    let mut retained = Vec::with_capacity(entries.len());
    for key in spec.keys.iter() {
        let Some(entry) = cache.get_with_key(key).await else {
            group_cache
                .insert_with_key(&spec.group_key, Arc::new(ResidualFtsGroupState::Rejected))
                .await;
            return Ok(ResidualFtsAdmission::Rejected(
                "residual working set exceeds cache admission capacity",
            ));
        };
        retained.push(entry);
    }
    if !matches!(
        group_cache.get_with_key(&spec.group_key).await.as_deref(),
        Some(ResidualFtsGroupState::Seen)
    ) {
        group_cache
            .insert_with_key(&spec.group_key, Arc::new(ResidualFtsGroupState::Rejected))
            .await;
        return Ok(ResidualFtsAdmission::Rejected(
            "residual metadata cache could not retain the working-set marker",
        ));
    }

    let reused = entries.iter().filter(|(_, _, reused)| *reused).count();
    let loader_runs = entries.len().saturating_sub(reused);
    Ok(ResidualFtsAdmission::Eligible(loaded_segments(
        retained,
        spec.fragment_bitmap(),
        reused,
        loader_runs,
        started.elapsed(),
    )))
}

fn loaded_segments(
    entries: Vec<Arc<CachedResidualFtsEntry>>,
    fragment_bitmap: roaring::RoaringBitmap,
    reuse_or_coalesced: usize,
    loader_runs: usize,
    build_duration: Duration,
) -> LoadedResidualFtsSegments {
    let mut loaded = LoadedResidualFtsSegments {
        segments: Vec::with_capacity(entries.len()),
        fragment_bitmap,
        stats: ResidualFtsCacheStats::default(),
    };
    loaded.stats.reuse_or_coalesced = reuse_or_coalesced;
    loaded.stats.loader_runs = loader_runs;
    if loader_runs != 0 {
        loaded.stats.build_duration = build_duration;
    }
    for entry in entries {
        let CachedResidualFtsEntry::Ready(segment) = entry.as_ref();
        loaded.stats.rows += segment.rows();
        loaded.stats.documents += segment.documents();
        loaded.stats.serialized_bytes += segment.serialized_bytes();
        loaded.stats.resident_bytes += segment.resident_bytes();
        loaded.segments.push(segment.index());
    }
    loaded
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use lance_core::cache::{CacheNamespace, KeyBuilder, recommended_cache_shards};

    use super::*;
    use crate::dataset::DEFAULT_INDEX_CACHE_SIZE;

    fn internal_key(key: &ResidualFtsFragmentKey) -> [u8; 16] {
        let mut builder = KeyBuilder::new(
            CacheNamespace::root(),
            ResidualFtsFragmentKey::stable_type_id(),
            ResidualFtsFragmentKey::schema(),
        );
        key.write_key(&mut builder);
        builder.finish().into_bytes()
    }

    fn key() -> ResidualFtsFragmentKey {
        ResidualFtsFragmentKey {
            store_identity: Arc::from("memory"),
            index_uuid: Uuid::nil(),
            index_version: 3,
            fragment_id: 7,
            fragment_fingerprint: Arc::from(&b"fragment-a"[..]),
            field_id: 2,
            canonical_path: Arc::from("body"),
            field_type: Arc::from("Utf8"),
            params_fingerprint: Arc::from(&b"params-a"[..]),
        }
    }

    fn group_key() -> ResidualFtsGroupKey {
        let member = key();
        ResidualFtsGroupKey {
            store_identity: member.store_identity.clone(),
            index_uuid: member.index_uuid,
            index_version: member.index_version,
            members: Arc::from([member]),
        }
    }

    struct PressureKey(&'static str);

    impl CacheKey for PressureKey {
        type ValueType = Vec<u8>;

        fn key(&self) -> Cow<'_, str> {
            Cow::Borrowed(self.0)
        }

        fn type_name() -> &'static str {
            "ResidualFtsPressureValue"
        }
    }

    #[test]
    fn residual_key_reuses_identical_fragment() {
        assert_eq!(internal_key(&key()), internal_key(&key()));
    }

    #[test]
    fn default_cache_admits_large_residual_fixture() {
        const FIXTURE_ROWS: usize = 100_000;
        const FIXTURE_SERIALIZED_BYTES: usize = 396 << 20;

        let shards = recommended_cache_shards(DEFAULT_INDEX_CACHE_SIZE);
        assert!(FIXTURE_ROWS <= MAX_RESIDUAL_FTS_ROWS);
        assert!(FIXTURE_SERIALIZED_BYTES < DEFAULT_INDEX_CACHE_SIZE / shards);
    }

    #[test]
    fn residual_key_invalidates_every_semantic_identity() {
        let original = key();
        let original_key = internal_key(&original);
        let variants = [
            ResidualFtsFragmentKey {
                store_identity: Arc::from("memory-other"),
                ..original.clone()
            },
            ResidualFtsFragmentKey {
                index_uuid: Uuid::from_u128(1),
                ..original.clone()
            },
            ResidualFtsFragmentKey {
                index_version: 4,
                ..original.clone()
            },
            ResidualFtsFragmentKey {
                fragment_id: 8,
                ..original.clone()
            },
            ResidualFtsFragmentKey {
                fragment_fingerprint: Arc::from(&b"fragment-b"[..]),
                ..original.clone()
            },
            ResidualFtsFragmentKey {
                field_id: 3,
                ..original.clone()
            },
            ResidualFtsFragmentKey {
                canonical_path: Arc::from("renamed"),
                ..original.clone()
            },
            ResidualFtsFragmentKey {
                field_type: Arc::from("LargeUtf8"),
                ..original.clone()
            },
            ResidualFtsFragmentKey {
                params_fingerprint: Arc::from(&b"params-b"[..]),
                ..original
            },
        ];
        for variant in variants {
            assert_ne!(original_key, internal_key(&variant));
        }
    }

    #[tokio::test]
    async fn budgeted_memory_store_rejects_growth_before_allocation() {
        use object_store::ObjectStoreExt;

        let store = BudgetedMemoryStore::new(4);
        let error = store
            .put(&Path::from("too-large"), PutPayload::from_static(b"12345"))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("4 byte build budget"));
        assert!(store.list(None).next().await.is_none());
    }

    #[tokio::test]
    async fn multi_fragment_builds_share_one_upload_budget() {
        use object_store::ObjectStoreExt;

        let uploaded = Arc::new(AtomicUsize::new(0));
        let exceeded = Arc::new(AtomicBool::new(false));
        let first = BudgetedMemoryStore::with_counter(uploaded.clone(), exceeded.clone(), 7);
        let second = BudgetedMemoryStore::with_counter(uploaded, exceeded.clone(), 7);
        first
            .put(&Path::from("first"), PutPayload::from_static(b"1234"))
            .await
            .unwrap();
        let error = second
            .put(&Path::from("second"), PutPayload::from_static(b"5678"))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("7 byte build budget"));
        assert!(exceeded.load(Ordering::Relaxed));
        assert!(second.list(None).next().await.is_none());
    }

    #[tokio::test]
    async fn object_copy_cannot_bypass_upload_budget() {
        use object_store::ObjectStoreExt;

        let store = BudgetedMemoryStore::new(7);
        let source = Path::from("source");
        let destination = Path::from("destination");
        store
            .put(&source, PutPayload::from_static(b"1234"))
            .await
            .unwrap();

        let error = store.copy(&source, &destination).await.unwrap_err();
        assert!(error.to_string().contains("7 byte build budget"));
        assert!(store.head(&source).await.is_ok());
        assert!(store.head(&destination).await.is_err());
    }

    #[tokio::test]
    async fn object_rename_is_conservatively_charged() {
        use object_store::ObjectStoreExt;

        let store = BudgetedMemoryStore::new(7);
        let source = Path::from("source");
        let destination = Path::from("destination");
        store
            .put(&source, PutPayload::from_static(b"1234"))
            .await
            .unwrap();

        let error = store.rename(&source, &destination).await.unwrap_err();
        assert!(error.to_string().contains("7 byte build budget"));
        assert!(store.head(&source).await.is_ok());
        assert!(store.head(&destination).await.is_err());
    }

    #[tokio::test]
    async fn aggregate_rejection_is_independent_of_posting_eviction() {
        let posting_cache = LanceCache::with_capacity(192);
        let group_cache = LanceCache::with_capacity(4096);
        let group_key = group_key();
        group_cache
            .insert_with_key(&group_key, Arc::new(ResidualFtsGroupState::Rejected))
            .await;

        // Two individually admissible values exceed aggregate capacity and
        // evict one another, but cannot evict the independent negative state.
        posting_cache
            .insert_with_key(&PressureKey("a"), Arc::new(vec![0; 128]))
            .await;
        posting_cache
            .insert_with_key(&PressureKey("b"), Arc::new(vec![0; 128]))
            .await;
        assert!(posting_cache.size_bytes().await <= 192);
        let loader_runs = AtomicUsize::new(0);
        if !matches!(
            group_cache.get_with_key(&group_key).await.as_deref(),
            Some(ResidualFtsGroupState::Rejected)
        ) {
            loader_runs.fetch_add(1, Ordering::Relaxed);
        }
        assert_eq!(loader_runs.load(Ordering::Relaxed), 0);
    }
}
