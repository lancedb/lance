// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Multi-segment SuffixArrayIndex implementing the ScalarIndex trait.
//!
//! A suffix array index may contain one or more segments. Each segment
//! covers a group of fragments and has its own text buffer and suffix
//! array. Query methods (count, prob, ntd, infgram_prob) transparently
//! aggregate results across all segments.
//!
//! Segments are loaded **lazily on first access** and cached via the
//! shared `LanceCache`. This keeps memory usage bounded by the cache
//! capacity rather than the total index size. For a 100 GB corpus
//! with 200×512 MB segments, only the actively queried segments stay
//! in memory; the rest are evicted by the LRU cache.
//!
//! This design is necessary because suffix arrays use u32 pointers,
//! limiting each segment to ~4 GB of text. For larger corpora the
//! plugin splits the data into multiple segments during build.

use std::any::Any;
use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use futures::stream::{self, StreamExt, TryStreamExt};
use lance_core::cache::{CacheKey, WeakLanceCache};
use lance_core::utils::mask::RowAddrTreeMap;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use tracing::{info, warn};

use crate::frag_reuse::FragReuseIndex;
use crate::metrics::MetricsCollector;
use crate::scalar::registry::{TrainingCriteria, TrainingOrdering};
use crate::scalar::{
    AnyQuery, CreatedIndex, IndexStore, OldIndexDataFilter, ScalarIndex, ScalarIndexParams,
    SearchResult, UpdateCriteria,
};
use crate::vector::VectorIndex;
use crate::{Index, IndexType};

use super::bloom::BloomFilter;
use super::query::{self, SuffixArrayQuery};

/// Maximum number of segments to query concurrently.
///
/// Controls I/O parallelism for S3 block-cache reads across segments.
/// Set `LANCE_SA_PARALLEL_QUERIES` env var to override (0 = all segments).
/// Default: 0 (all segments in parallel — optimal for I/O-bound S3 queries).
fn parallel_query_concurrency(num_segments: usize) -> usize {
    let val = std::env::var("LANCE_SA_PARALLEL_QUERIES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    if val == 0 {
        num_segments
    } else {
        val.min(num_segments)
    }
}

/// One segment of a suffix array index, loaded into memory.
///
/// Each segment holds a contiguous text buffer and its corresponding
/// suffix array. The text is at most ~2 GB so that u32 pointers suffice.
///
/// Implements `DeepSizeOf` so the LRU cache can track memory usage
/// and evict segments when the cache capacity is exceeded.
#[derive(Debug)]
pub struct SuffixArraySegment {
    /// The raw tokenized corpus data for this segment.
    /// `None` when using block-cached reads (raw file format).
    pub tokenized: Option<Bytes>,
    /// Block cache for tokenized data — always available.
    /// When data is loaded from a `.raw` file, reads go to S3.
    /// When data is loaded from `.bin` (legacy), the cache is
    /// pre-populated with in-memory blocks for zero-copy access.
    pub tokenized_cache: Option<super::block_cache::BlockCache>,
    /// The compacted suffix array (variable-width pointers).
    /// `None` when using block-cached reads (raw file format).
    pub suffix_array: Option<Bytes>,
    /// Block cache for suffix array — always available.
    /// Same pre-population logic as `tokenized_cache`.
    pub sa_cache: Option<super::block_cache::BlockCache>,
    /// Bytes per pointer in the suffix array.
    pub pointer_width: u8,
    /// Total number of entries in the suffix array.
    pub total_entries: u64,
    /// Cumulative byte offsets marking the end of each document.
    /// `doc_offsets[i]` is the byte offset where document `i` ends.
    /// Document `i` spans `[doc_offsets[i-1], doc_offsets[i])` (with doc_offsets[-1] = 0).
    /// Empty if document retrieval is not available (legacy indices).
    pub doc_offsets: Vec<u64>,
    /// Lance row IDs (`_rowid`) for each document.
    /// `row_ids[i]` is the fragment-encoded row address for document `i`.
    /// Used by `search_rows` to return proper Lance row addresses.
    /// Empty for legacy indices (pre-row-id support).
    pub row_ids: Vec<u64>,
}

impl DeepSizeOf for SuffixArraySegment {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.tokenized.as_ref().map_or(0, |t| t.len())
            + self.tokenized_cache.as_ref().map_or(0, |c: &super::block_cache::BlockCache| c.deep_size_of_children(context))
            + self.suffix_array.as_ref().map_or(0, |s| s.len())
            + self.sa_cache.as_ref().map_or(0, |c: &super::block_cache::BlockCache| c.deep_size_of_children(context))
            + self.doc_offsets.len() * std::mem::size_of::<u64>()
            + self.row_ids.len() * std::mem::size_of::<u64>()
    }
}

impl SuffixArraySegment {
    /// Count occurrences of a pattern in this segment.
    /// Only works when tokenized data is fully loaded (not block-cached).
    #[inline]
    pub fn count(&self, query: &[u8]) -> u64 {
        if let (Some(tokenized), Some(sa)) = (&self.tokenized, &self.suffix_array) {
            query::count(
                tokenized,
                sa,
                self.pointer_width as usize,
                self.total_entries,
                query,
            )
        } else {
            0 // block-cached path doesn't support sync count
        }
    }

    /// Get the tokenized data length (from either full load or block cache).
    pub fn tokenized_len(&self) -> usize {
        if let Some(ref tokenized) = self.tokenized {
            tokenized.len()
        } else if let Some(ref cache) = self.tokenized_cache {
            cache.len()
        } else {
            0
        }
    }

    /// Read bytes from the tokenized data at a given offset.
    ///
    /// For fully-loaded segments, this is a zero-copy slice.
    /// For block-cached segments, this fetches via range read.
    pub async fn read_tokenized(&self, offset: usize, len: usize) -> Result<Bytes> {
        if let Some(ref tokenized) = self.tokenized {
            let end = (offset + len).min(tokenized.len());
            Ok(tokenized.slice(offset..end))
        } else if let Some(ref cache) = self.tokenized_cache {
            cache.read(offset, len).await
        } else {
            Ok(Bytes::new())
        }
    }
}

/// Lightweight metadata for one segment (no data loaded).
/// Stored in `SuffixArrayIndex` to enable lazy loading.
#[derive(Debug, Clone)]
struct SegmentMeta {
    /// Index of this segment (0, 1, 2, ...).
    index: usize,
    /// File name for the tokenized data.
    tok_filename: String,
    /// File name for the suffix array.
    sa_filename: String,
    /// File name for the doc offsets (None for legacy v0 indices).
    offsets_filename: Option<String>,
    /// File name for the row IDs (None for legacy indices without row ID support).
    row_ids_filename: Option<String>,
    /// File name for the bloom filter (None for legacy indices).
    bloom_filename: Option<String>,
    /// Eagerly-loaded bloom filter for fast segment skipping.
    /// Loaded at index open time (~128 KB per segment) so that `search_rows`
    /// can skip segments without loading the full ~1 GB segment data.
    bloom_filter: Option<BloomFilter>,
    /// Bytes per pointer in the suffix array.
    pointer_width: u8,
    /// Total number of entries in the suffix array.
    total_entries: u64,
}

/// Cache key for per-segment lazy loading.
struct SegmentCacheKey {
    segment_index: usize,
}

impl CacheKey for SegmentCacheKey {
    type ValueType = SuffixArraySegment;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("sa-seg-{}", self.segment_index))
    }

    fn type_name() -> &'static str {
        "SuffixArraySegment"
    }
}

/// A suffix array index with lazy per-segment loading.
///
/// At load time, only segment metadata is stored. The actual segment
/// data (tokenized bytes + suffix array + doc offsets) is loaded on
/// first access and cached via the shared `LanceCache`. The LRU cache
/// evicts segments when the total cached size exceeds `index_cache_size`.
pub struct SuffixArrayIndex {
    /// Metadata for each segment (lightweight, always in memory).
    segment_metas: Vec<SegmentMeta>,
    /// The index store for loading segment files on demand.
    store: Arc<dyn IndexStore>,
    /// Weak reference to the shared LRU cache.
    cache: WeakLanceCache,
    /// Pre-loaded segments for the `from_segments()` / `new()` path.
    /// When set, `get_segment()` returns from here instead of the cache.
    /// This avoids needing an async runtime in test/build code.
    preloaded: Option<Vec<Arc<SuffixArraySegment>>>,
    /// When true, the corpus was lowercased at build time.
    /// Queries must be lowercased before searching.
    case_insensitive: bool,
    /// Bytes per token unit. 1 for byte-level (text), 2 for int16/uint16,
    /// 4 for int32/uint32. Queries on token-level indices are constrained
    /// to positions aligned to `token_width` boundaries.
    token_width: u8,
}

impl std::fmt::Debug for SuffixArrayIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SuffixArrayIndex")
            .field("num_segments", &self.segment_metas.len())
            .field("case_insensitive", &self.case_insensitive)
            .field("token_width", &self.token_width)
            .finish()
    }
}

impl DeepSizeOf for SuffixArrayIndex {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        // Count the metadata (always in memory) + bloom filters.
        // Segment data is in the cache and tracked there.
        self.segment_metas.iter().map(|m| {
            std::mem::size_of::<SegmentMeta>()
                + m.bloom_filter.as_ref().map_or(0, |bf| {
                    // Approximate: BloomFilter struct + bit array
                    std::mem::size_of::<BloomFilter>() + bf.to_bytes().len()
                })
        }).sum()
    }
}

#[async_trait]
impl Index for SuffixArrayIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn VectorIndex>> {
        Err(Error::invalid_input(
            "SuffixArrayIndex is not a vector index",
        ))
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "index_type": "SuffixArray",
            "num_segments": self.segment_metas.len(),
            "total_entries": self.segment_metas.iter().map(|m| m.total_entries).sum::<u64>(),
            "case_insensitive": self.case_insensitive,
            "token_width": self.token_width,
            "lazy_loading": true,
        }))
    }

    async fn prewarm(&self) -> Result<()> {
        // Load all segments into cache for predictable latency.
        for meta in &self.segment_metas {
            self.get_segment(meta).await?;
        }
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::SuffixArray
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        Ok(RoaringBitmap::new())
    }
}

#[async_trait]
impl ScalarIndex for SuffixArrayIndex {
    async fn search(
        &self,
        query_obj: &dyn AnyQuery,
        _metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let sa_query = query_obj
            .as_any()
            .downcast_ref::<SuffixArrayQuery>()
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "SuffixArrayIndex expected SuffixArrayQuery, got {:?}",
                    query_obj
                ))
            })?;

        // Helper: lowercase bytes if the index was built case-insensitively.
        // Only valid for UTF-8 text (not raw binary).
        let maybe_lower = |bytes: &[u8]| -> Vec<u8> {
            if self.case_insensitive {
                String::from_utf8_lossy(bytes).to_lowercase().into_bytes()
            } else {
                bytes.to_vec()
            }
        };

        match sa_query {
            SuffixArrayQuery::Count { query_bytes } => {
                let qb = maybe_lower(query_bytes);
                let n = self.total_count(&qb).await?;
                let mut map = RowAddrTreeMap::new();
                if n > 0 {
                    map.insert(n);
                }
                Ok(SearchResult::Exact(
                    lance_core::utils::mask::NullableRowAddrSet::new(map, Default::default()),
                ))
            }
            SuffixArrayQuery::Search {
                query_bytes,
                max_results,
            } => {
                let qb = maybe_lower(query_bytes);
                let row_ids = self.search_rows(&qb, *max_results).await?;
                let mut map = RowAddrTreeMap::new();
                for row_id in &row_ids {
                    map.insert(*row_id);
                }
                Ok(SearchResult::Exact(
                    lance_core::utils::mask::NullableRowAddrSet::new(map, Default::default()),
                ))
            }
            SuffixArrayQuery::Prob {
                prompt_bytes,
                continuation_bytes,
            } => {
                let pb = maybe_lower(prompt_bytes);
                let cb = maybe_lower(continuation_bytes);
                let result = self.compute_prob(&pb, &cb).await?;
                let mut map = RowAddrTreeMap::new();
                if result.cont_cnt > 0 {
                    map.insert(result.cont_cnt);
                }
                Ok(SearchResult::Exact(
                    lance_core::utils::mask::NullableRowAddrSet::new(map, Default::default()),
                ))
            }
            SuffixArrayQuery::NextByteDistribution {
                prompt_bytes,
                max_support,
            } => {
                let pb = maybe_lower(prompt_bytes);
                let result = self.compute_ntd(&pb, *max_support).await?;
                let mut map = RowAddrTreeMap::new();
                if result.prompt_cnt > 0 {
                    map.insert(result.prompt_cnt);
                }
                Ok(SearchResult::Exact(
                    lance_core::utils::mask::NullableRowAddrSet::new(map, Default::default()),
                ))
            }
            SuffixArrayQuery::InfgramProb {
                prompt_bytes,
                continuation_bytes,
            } => {
                let pb = maybe_lower(prompt_bytes);
                let cb = maybe_lower(continuation_bytes);
                let result = self.compute_infgram_prob(&pb, &cb).await?;
                let mut map = RowAddrTreeMap::new();
                if result.prob_result.cont_cnt > 0 {
                    map.insert(result.prob_result.cont_cnt);
                }
                Ok(SearchResult::Exact(
                    lance_core::utils::mask::NullableRowAddrSet::new(map, Default::default()),
                ))
            }
        }
    }

    fn can_remap(&self) -> bool {
        false
    }

    async fn remap(
        &self,
        _mapping: &HashMap<u64, Option<u64>>,
        _dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        Err(Error::not_supported(
            "Suffix array indices must be rebuilt, not remapped",
        ))
    }

    async fn update(
        &self,
        _new_data: SendableRecordBatchStream,
        _dest_store: &dyn IndexStore,
        _old_data_filter: Option<OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        Err(Error::not_supported(
            "Suffix array index update is not yet implemented; rebuild the index instead",
        ))
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::requires_old_data(TrainingCriteria::new(TrainingOrdering::None))
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        Ok(ScalarIndexParams::new("suffixarray".to_string()))
    }
}

impl SuffixArrayIndex {
    /// Create a single-segment SuffixArrayIndex from pre-loaded data.
    /// Used by tests that build segments in-memory.
    pub fn new(
        tokenized: Bytes,
        suffix_array: Bytes,
        pointer_width: u8,
        total_entries: u64,
    ) -> Self {
        Self::from_segments(vec![SuffixArraySegment {
            tokenized: Some(tokenized),
            tokenized_cache: None,
            suffix_array: Some(suffix_array),
            sa_cache: None,
            pointer_width,
            total_entries,
            doc_offsets: Vec::new(),
            row_ids: Vec::new(),
        }])
    }

    /// Whether this index was built with case-insensitive (lowercased) text.
    pub fn case_insensitive(&self) -> bool {
        self.case_insensitive
    }

    /// Token width (bytes per token unit). 1 = byte-level, 2 = int16, 4 = int32.
    pub fn token_width(&self) -> u8 {
        self.token_width
    }

    /// Create a SuffixArrayIndex from pre-loaded segments (tests + in-memory build).
    /// Segments are stored directly in the struct (no cache or store needed).
    pub fn from_segments(segments: Vec<SuffixArraySegment>) -> Self {
        let cache = lance_core::cache::LanceCache::with_capacity(0);
        let weak_cache = WeakLanceCache::from(&cache);

        let mut metas = Vec::with_capacity(segments.len());
        let mut preloaded = Vec::with_capacity(segments.len());
        for (i, seg) in segments.into_iter().enumerate() {
            metas.push(SegmentMeta {
                index: i,
                tok_filename: format!("segment_{i}_tokenized.bin"),
                sa_filename: format!("segment_{i}_suffix_array.bin"),
                offsets_filename: None,
                row_ids_filename: None,
                bloom_filename: None,
                bloom_filter: None,
                pointer_width: seg.pointer_width,
                total_entries: seg.total_entries,
            });
            preloaded.push(Arc::new(seg));
        }

        Self {
            segment_metas: metas,
            store: Arc::new(NoopIndexStore),
            cache: weak_cache,
            preloaded: Some(preloaded),
            case_insensitive: false,
            token_width: 1,
        }
    }

    /// Load a single-segment SuffixArrayIndex lazily (v0 format).
    pub async fn load(
        store: Arc<dyn IndexStore>,
        _frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: &lance_core::cache::LanceCache,
        pointer_width: u8,
        total_entries: u64,
        case_insensitive: bool,
        token_width: u8,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            segment_metas: vec![SegmentMeta {
                index: 0,
                tok_filename: "tokenized.bin".to_string(),
                sa_filename: "suffix_array.bin".to_string(),
                offsets_filename: None, // v0 has no offsets file
                row_ids_filename: None, // v0 has no row IDs file
                bloom_filename: None,   // v0 has no bloom filter
                bloom_filter: None,
                pointer_width,
                total_entries,
            }],
            store,
            cache: WeakLanceCache::from(cache),
            preloaded: None,
            case_insensitive,
            token_width,
        }))
    }

    /// Load a multi-segment SuffixArrayIndex lazily (v1+ format).
    /// Eagerly loads bloom filters (~128 KB each) for fast segment skipping.
    pub async fn load_multi(
        store: Arc<dyn IndexStore>,
        segment_infos: &[crate::pb::SuffixArraySegmentInfo],
        case_insensitive: bool,
        token_width: u8,
        cache: &lance_core::cache::LanceCache,
    ) -> Result<Arc<Self>> {
        let mut metas = Vec::with_capacity(segment_infos.len());
        let mut bloom_loaded = 0usize;
        let mut bloom_missing = 0usize;
        let mut bloom_parse_fail = 0usize;

        for (i, info) in segment_infos.iter().enumerate() {
            let bloom_filename = format!("segment_{i}_bloom.bin");

            // Eagerly load bloom filter (small: ~128 KB per segment)
            let bloom_filter = match store.open_index_file(&bloom_filename).await {
                Ok(reader) => {
                    match reader.read_record_batch(0, 1).await {
                        Ok(batch) => {
                            let col = batch
                                .column(0)
                                .as_any()
                                .downcast_ref::<arrow_array::LargeBinaryArray>();
                            match col.and_then(|c| BloomFilter::from_bytes(c.value(0))) {
                                Some(bf) => {
                                    bloom_loaded += 1;
                                    Some(bf)
                                }
                                None => {
                                    warn!(segment = i, "Bloom filter parse failed for {bloom_filename}");
                                    bloom_parse_fail += 1;
                                    None
                                }
                            }
                        }
                        Err(e) => {
                            warn!(segment = i, err = %e, "Failed to read bloom batch from {bloom_filename}");
                            bloom_parse_fail += 1;
                            None
                        }
                    }
                }
                Err(e) => {
                    if i == 0 {
                        // Only log once to avoid spam for legacy indices
                        info!(err = %e, "Bloom filter files not found (legacy index without bloom support)");
                    }
                    bloom_missing += 1;
                    None
                }
            };

            metas.push(SegmentMeta {
                index: i,
                tok_filename: format!("segment_{i}_tokenized.bin"),
                sa_filename: format!("segment_{i}_suffix_array.bin"),
                offsets_filename: Some(format!("segment_{i}_doc_offsets.bin")),
                row_ids_filename: Some(format!("segment_{i}_row_ids.bin")),
                bloom_filename: Some(bloom_filename),
                bloom_filter,
                pointer_width: info.pointer_width as u8,
                total_entries: info.total_entries,
            });
        }

        info!(
            total_segments = metas.len(),
            bloom_loaded,
            bloom_missing,
            bloom_parse_fail,
            "SA index loaded"
        );

        Ok(Arc::new(Self {
            segment_metas: metas,
            store,
            cache: WeakLanceCache::from(cache),
            preloaded: None,
            case_insensitive,
            token_width,
        }))
    }

    /// Load one segment's files from the store.
    ///
    /// Tries to load tokenized data as a raw file (block-cached) first.
    /// Falls back to Lance format (full load) if raw file doesn't exist.
    /// Suffix array is always fully loaded (~15 MB).
    async fn load_segment_from_store(
        store: &dyn IndexStore,
        meta: &SegmentMeta,
    ) -> Result<SuffixArraySegment> {
        let store_arc = store.clone_arc();

        // Load tokenized data — always produces a BlockCache.
        // If a .raw file exists, use lazy block reads from the store.
        // If only .bin exists, load into memory and wrap as a pre-populated BlockCache.
        let raw_tok_name = meta.tok_filename.replace(".bin", ".raw");
        let (tokenized, tokenized_cache) = match store.raw_file_size(&raw_tok_name).await {
            Ok(size) => {
                info!(
                    segment = meta.index,
                    raw_file = %raw_tok_name,
                    size_mb = size / (1024 * 1024),
                    "Using block-cached tokenized (raw file)"
                );
                let cache = super::block_cache::BlockCache::new(
                    store_arc.clone(),
                    raw_tok_name,
                    size,
                );
                (None, Some(cache))
            }
            Err(_) => {
                // No .raw file: load from .bin, wrap as block cache for uniform dispatch
                let tokenized_reader = store.open_index_file(&meta.tok_filename).await?;
                let tokenized_batch = tokenized_reader.read_record_batch(0, 1).await?;
                let tokenized_col = tokenized_batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<arrow_array::LargeBinaryArray>()
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "{} should contain a LargeBinary column",
                            meta.tok_filename
                        ))
                    })?;
                let tokenized = Bytes::copy_from_slice(tokenized_col.value(0));

                // Fire-and-forget: write .raw file so future queries use true block cache.
                // Non-blocking — the query proceeds immediately with from_bytes cache.
                {
                    let store_bg = store_arc.clone();
                    let raw_name = raw_tok_name.clone();
                    let data = tokenized.clone();
                    tokio::spawn(async move {
                        if let Err(e) = store_bg.write_raw_file(&raw_name, &data).await {
                            warn!(file = %raw_name, err = %e, "Failed to write tokenized .raw (non-fatal)");
                        }
                    });
                }

                let cache = super::block_cache::BlockCache::from_bytes(
                    store_arc.clone(),
                    meta.tok_filename.clone(),
                    tokenized.clone(),
                );
                (Some(tokenized), Some(cache))
            }
        };

        // Load suffix array — same strategy: always produces a BlockCache.
        // If a .raw file exists, use lazy block reads.
        // If only .bin exists, load into memory and wrap as a pre-populated BlockCache.
        // Also writes a .raw file on first load so future queries use true block cache.
        let raw_sa_name = meta.sa_filename.replace(".bin", ".raw");
        let (suffix_array, sa_cache) = match store.raw_file_size(&raw_sa_name).await {
            Ok(size) => {
                info!(
                    segment = meta.index,
                    raw_file = %raw_sa_name,
                    size_mb = size / (1024 * 1024),
                    "Using block-cached suffix array (raw file)"
                );
                let cache = super::block_cache::BlockCache::new(
                    store_arc.clone(),
                    raw_sa_name,
                    size,
                );
                (None, Some(cache))
            }
            Err(_) => {
                // Legacy index: no .raw file. Load from .bin (Lance IPC format),
                // wrap as pre-populated block cache for uniform dispatch.
                let sa_reader = store.open_index_file(&meta.sa_filename).await?;
                let sa_batch = sa_reader.read_record_batch(0, 1).await?;
                let sa_col = sa_batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<arrow_array::LargeBinaryArray>()
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "{} should contain a LargeBinary column",
                            meta.sa_filename
                        ))
                    })?;
                let sa = Bytes::copy_from_slice(sa_col.value(0));

                let cache = super::block_cache::BlockCache::from_bytes(
                    store_arc.clone(),
                    meta.sa_filename.clone(),
                    sa.clone(),
                );
                (Some(sa), Some(cache))
            }
        };

        // Try to load doc_offsets (optional — may not exist for legacy indices)
        let doc_offsets = if let Some(offsets_name) = &meta.offsets_filename {
            match store.open_index_file(offsets_name).await {
                Ok(offsets_reader) => {
                    let offsets_batch = offsets_reader.read_record_batch(0, 1).await?;
                    let offsets_col = offsets_batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<arrow_array::LargeBinaryArray>()
                        .ok_or_else(|| {
                            Error::invalid_input(format!(
                                "{offsets_name} should contain a LargeBinary column"
                            ))
                        })?;
                    let offsets_bytes = offsets_col.value(0);
                    // Parse packed u64 values from little-endian bytes
                    offsets_bytes
                        .chunks_exact(8)
                        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                        .collect()
                }
                Err(_) => Vec::new(), // File not found — legacy index
            }
        } else {
            Vec::new()
        };

        // Try to load row_ids (optional — may not exist for legacy indices)
        let row_ids = if let Some(row_ids_name) = &meta.row_ids_filename {
            match store.open_index_file(row_ids_name).await {
                Ok(row_ids_reader) => {
                    let row_ids_batch = row_ids_reader.read_record_batch(0, 1).await?;
                    let row_ids_col = row_ids_batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<arrow_array::LargeBinaryArray>()
                        .ok_or_else(|| {
                            Error::invalid_input(format!(
                                "{row_ids_name} should contain a LargeBinary column"
                            ))
                        })?;
                    let row_ids_bytes = row_ids_col.value(0);
                    row_ids_bytes
                        .chunks_exact(8)
                        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                        .collect()
                }
                Err(_) => Vec::new(), // File not found — legacy index
            }
        } else {
            Vec::new()
        };

        Ok(SuffixArraySegment {
            pointer_width: meta.pointer_width,
            total_entries: meta.total_entries,
            tokenized,
            tokenized_cache,
            suffix_array,
            sa_cache,
            doc_offsets,
            row_ids,
        })
    }

    /// Get a segment, loading from store on cache miss.
    /// If segments were pre-loaded (from_segments path), returns directly.
    async fn get_segment(&self, meta: &SegmentMeta) -> Result<Arc<SuffixArraySegment>> {
        // Fast path: pre-loaded segments (tests / in-memory build)
        if let Some(preloaded) = &self.preloaded {
            return Ok(preloaded[meta.index].clone());
        }

        // Lazy path: load from store via cache
        let key = SegmentCacheKey {
            segment_index: meta.index,
        };
        let store = self.store.clone();
        let meta_clone = meta.clone();
        self.cache
            .get_or_insert_with_key(key, || async move {
                Self::load_segment_from_store(store.as_ref(), &meta_clone).await
            })
            .await
    }

    // ─── Per-segment query helpers (named async fns for lifetime resolution) ──
    //
    // These exist as named methods instead of inline async blocks because
    // `stream::iter(iter.map(|x| async { ... })).buffer_unordered(n)` hits
    // Rust's "implementation of FnOnce is not general enough" error — the
    // closure's async block binds the iterator reference's lifetime, but
    // `buffer_unordered` needs FnOnce to be generic over any lifetime.
    //
    // Named async fns produce concrete futures with resolved lifetimes when
    // called eagerly: `.map(|m| self.helper(m, q)).collect::<Vec<_>>()`.

    /// Count matches in a single segment (byte-level, tw <= 1).
    async fn count_segment_byte(
        &self,
        meta: &SegmentMeta,
        query: &[u8],
    ) -> Result<u64> {
        let seg = self.get_segment(meta).await?;
        let pw = seg.pointer_width as usize;
        let n = seg.total_entries;
        if let (Some(tok_cache), Some(sa_cache)) = (&seg.tokenized_cache, &seg.sa_cache) {
            query::count_cached(tok_cache, sa_cache, pw, n, query).await
        } else if let (Some(tokenized), Some(sa)) = (&seg.tokenized, &seg.suffix_array) {
            Ok(query::count(tokenized, sa, pw, n, query))
        } else {
            Ok(0)
        }
    }

    /// Count token-aligned matches in a single segment (tw > 1).
    async fn count_segment_token(
        &self,
        meta: &SegmentMeta,
        query: &[u8],
        tw: usize,
    ) -> Result<u64> {
        let seg = self.get_segment(meta).await?;
        let pw = seg.pointer_width as usize;
        let n = seg.total_entries;
        let (lo, hi) = if let (Some(tok_cache), Some(sa_cache)) =
            (&seg.tokenized_cache, &seg.sa_cache)
        {
            query::sa_find_cached(tok_cache, sa_cache, pw, n, query).await?
        } else if let (Some(tokenized), Some(sa)) = (&seg.tokenized, &seg.suffix_array) {
            query::sa_find(tokenized, sa, pw, n, query)
        } else {
            (0, 0)
        };
        let ptrs_per_chunk: u64 = (64 * 1024 / pw) as u64;
        let mut seg_total = 0u64;
        let mut rank = lo;
        while rank < hi {
            let chunk_count = (hi - rank).min(ptrs_per_chunk);
            let pointers = if let Some(ref sa_cache) = seg.sa_cache {
                query::read_pointers_batch(sa_cache, rank, chunk_count, pw).await?
            } else if let Some(ref sa) = seg.suffix_array {
                query::read_pointers(sa, rank, chunk_count, pw)
            } else {
                break;
            };
            for byte_pos in pointers {
                if (byte_pos as usize) % tw == 0 {
                    seg_total += 1;
                }
            }
            rank += chunk_count;
        }
        Ok(seg_total)
    }

    /// Search a single segment and return per-document positions.
    async fn search_segment(
        &self,
        meta: &SegmentMeta,
        query: &[u8],
        tw: usize,
    ) -> Result<HashMap<u64, Vec<u64>>> {
        let seg = self.get_segment(meta).await?;
        let pw = seg.pointer_width as usize;
        let n = seg.total_entries;
        let (lo, hi) = if let (Some(tok_cache), Some(sa_cache)) =
            (&seg.tokenized_cache, &seg.sa_cache)
        {
            query::sa_find_cached(tok_cache, sa_cache, pw, n, query).await?
        } else if let (Some(tokenized), Some(sa)) = (&seg.tokenized, &seg.suffix_array) {
            query::sa_find(tokenized, sa, pw, n, query)
        } else {
            (0, 0)
        };

        let mut local_docs: HashMap<u64, Vec<u64>> = HashMap::new();

        if seg.doc_offsets.is_empty() || seg.row_ids.is_empty() {
            return Ok(local_docs);
        }

        let batch_count = hi - lo;
        let positions: Vec<u64> = if let Some(ref sa_cache) = seg.sa_cache {
            query::read_pointers_batch(sa_cache, lo, batch_count, pw).await?
        } else if let Some(ref sa) = seg.suffix_array {
            query::read_pointers(sa, lo, batch_count, pw)
        } else {
            vec![]
        };

        for byte_pos in positions {
            if tw > 1 && (byte_pos as usize) % tw != 0 {
                continue;
            }
            let local_doc = seg.doc_offsets.partition_point(|&o| o <= byte_pos);
            let row_id = if local_doc < seg.row_ids.len() {
                seg.row_ids[local_doc]
            } else {
                continue;
            };
            let doc_start = if local_doc == 0 {
                0
            } else {
                seg.doc_offsets[local_doc - 1]
            };
            let pos_in_doc = byte_pos - doc_start;
            let char_pos = if tw > 1 {
                pos_in_doc / tw as u64
            } else {
                pos_in_doc
            };
            local_docs.entry(row_id).or_default().push(char_pos);
        }
        Ok(local_docs)
    }

    /// Compute next-byte distribution for a single segment.
    async fn ntd_segment(
        &self,
        meta: &SegmentMeta,
        prompt: &[u8],
        max_support: Option<u64>,
    ) -> Result<query::NtdResult> {
        let seg = self.get_segment(meta).await?;
        let pw = seg.pointer_width as usize;
        let n = seg.total_entries;
        if let (Some(tok_cache), Some(sa_cache)) = (&seg.tokenized_cache, &seg.sa_cache) {
            query::next_byte_distribution_cached(tok_cache, sa_cache, pw, n, prompt, max_support)
                .await
        } else if let (Some(tokenized), Some(sa)) = (&seg.tokenized, &seg.suffix_array) {
            Ok(query::next_byte_distribution(
                tokenized,
                sa,
                pw,
                n,
                prompt,
                max_support,
            ))
        } else {
            Ok(query::NtdResult {
                prompt_cnt: 0,
                distribution: Vec::new(),
                approximate: false,
            })
        }
    }

    // ─── Multi-segment query aggregation ─────────────────────────────────────

    /// Count occurrences of a pattern across all segments.
    /// For token-level indices (`token_width > 1`), only counts matches
    /// at token-aligned positions.
    /// Uses bloom filters to skip segments that definitely don't contain the query.
    /// Dispatches to sync or async (block-cached) path based on segment format.
    pub async fn total_count(&self, query: &[u8]) -> Result<u64> {
        let tw = self.token_width as usize;
        let concurrency = parallel_query_concurrency(self.segment_metas.len());

        // Pre-filter segments using bloom filters (sync, no I/O)
        let active_metas: Vec<&SegmentMeta> = self.segment_metas.iter()
            .filter(|meta| {
                meta.bloom_filter.as_ref()
                    .map_or(true, |bf| bf.might_contain_substring_aligned(query, tw))
            })
            .collect();

        if tw <= 1 {
            // Byte-level: count = SA range size (no position iteration)
            let futures: Vec<_> = active_metas
                .iter()
                .map(|meta| self.count_segment_byte(meta, query))
                .collect();
            let counts: Vec<u64> = stream::iter(futures)
                .buffer_unordered(concurrency)
                .try_collect()
                .await?;
            Ok(counts.into_iter().sum())
        } else {
            // Token-level: must check alignment of each match
            let futures: Vec<_> = active_metas
                .iter()
                .map(|meta| self.count_segment_token(meta, query, tw))
                .collect();
            let counts: Vec<u64> = stream::iter(futures)
                .buffer_unordered(concurrency)
                .try_collect()
                .await?;
            Ok(counts.into_iter().sum())
        }
    }

    /// Search for byte positions matching a pattern and resolve them to
    /// Lance row IDs with per-document occurrence counts.
    ///
    /// Returns `Vec<(row_id, count)>` sorted by count descending (top-K).
    /// This matches the inverted index's behavior of returning the most
    /// relevant documents first.
    ///
    /// Uses bloom filters to skip segments that definitely don't contain
    /// the query, avoiding expensive full-segment loads from S3.
    ///
    /// For token-level indices (`token_width > 1`), only matches at
    /// token-aligned positions (`pos % token_width == 0`) are included.
    ///
    /// All SA positions in the matching range are scanned (no sampling),
    /// so ranking is independent of `limit` — matching inverted index behavior.
    pub async fn search_rows_scored(
        &self,
        query: &[u8],
        max_results: usize,
    ) -> Result<Vec<(u64, u32, Vec<u64>)>> {
        let tw = self.token_width as usize;
        let concurrency = parallel_query_concurrency(self.segment_metas.len());

        // Pre-filter segments using bloom filters (sync, no I/O)
        let mut segments_skipped = 0usize;
        let mut segments_no_bloom = 0usize;
        let active_metas: Vec<&SegmentMeta> = self.segment_metas.iter()
            .filter(|meta| {
                if let Some(ref bf) = meta.bloom_filter {
                    if !bf.might_contain_substring_aligned(query, tw) {
                        segments_skipped += 1;
                        return false;
                    }
                } else {
                    segments_no_bloom += 1;
                }
                true
            })
            .collect();
        let segments_loaded = active_metas.len();

        // Query all active segments concurrently — each returns its own doc_positions
        let futures: Vec<_> = active_metas
            .iter()
            .map(|meta| self.search_segment(meta, query, tw))
            .collect();
        let segment_results: Vec<HashMap<u64, Vec<u64>>> = stream::iter(futures)
            .buffer_unordered(concurrency)
            .try_collect()
            .await?;

        // Sequential merge of per-segment results
        let mut doc_positions: HashMap<u64, Vec<u64>> = HashMap::new();
        for seg_map in segment_results {
            for (row_id, positions) in seg_map {
                doc_positions.entry(row_id).or_default().extend(positions);
            }
        }

        info!(
            query_len = query.len(),
            segments_total = self.segment_metas.len(),
            segments_skipped,
            segments_loaded,
            segments_no_bloom,
            unique_docs = doc_positions.len(),
            "SA search_rows_scored complete"
        );

        // Sort positions ascending within each document
        for positions in doc_positions.values_mut() {
            positions.sort_unstable();
        }

        // Top-K by count using min-heap (same pattern as inverted index's BM25)
        let mut heap: BinaryHeap<Reverse<(u32, u64)>> = BinaryHeap::new();
        for (row_id, positions) in &doc_positions {
            let count = positions.len() as u32;
            if heap.len() < max_results {
                heap.push(Reverse((count, *row_id)));
            } else if let Some(&Reverse((min_count, _))) = heap.peek() {
                if count > min_count {
                    heap.pop();
                    heap.push(Reverse((count, *row_id)));
                }
            }
        }

        let top_k: Vec<(u64, u32)> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse((count, row_id))| (row_id, count))
            .collect();

        let mut results: Vec<(u64, u32, Vec<u64>)> = top_k
            .into_iter()
            .map(|(row_id, count)| {
                let positions = doc_positions.remove(&row_id).unwrap_or_default();
                (row_id, count, positions)
            })
            .collect();
        results.reverse();
        Ok(results)
    }

    /// Boolean search: execute CNF clauses (AND of OR groups).
    ///
    /// Each inner Vec is an OR group — a document matches the group if it
    /// contains ANY of the terms. The outer Vec is AND — a document must
    /// match ALL groups.
    ///
    /// Returns `Vec<(row_id, total_count, merged_positions)>` sorted by
    /// total count descending.
    pub async fn search_boolean(
        &self,
        clauses: &[Vec<Vec<u8>>],
        max_results: usize,
    ) -> Result<Vec<(u64, u32, Vec<u64>)>> {
        if clauses.is_empty() {
            return Ok(Vec::new());
        }

        // Search each AND group independently
        let mut group_maps: Vec<HashMap<u64, (u32, Vec<u64>)>> = Vec::new();

        for or_group in clauses {
            let mut group_map: HashMap<u64, (u32, Vec<u64>)> = HashMap::new();

            for term in or_group {
                // Search this single term (no limit — need all docs for intersection)
                let results = self
                    .search_rows_scored(term, usize::MAX)
                    .await?;

                // Union into group_map (OR semantics: accumulate counts + positions)
                for (row_id, count, positions) in results {
                    let entry = group_map.entry(row_id).or_insert((0, Vec::new()));
                    entry.0 += count;
                    entry.1.extend(positions);
                }
            }

            group_maps.push(group_map);
        }

        // Intersect across AND groups (document must appear in ALL groups)
        // Start with the smallest group for efficiency
        group_maps.sort_by_key(|m| m.len());
        let mut result_map = group_maps.remove(0);

        for group in &group_maps {
            // Keep only row_ids present in both
            result_map.retain(|row_id, _| group.contains_key(row_id));
            // Merge scores and positions from this group
            for (row_id, (count, positions)) in result_map.iter_mut() {
                if let Some((g_count, g_positions)) = group.get(row_id) {
                    *count += g_count;
                    positions.extend(g_positions);
                }
            }
        }

        // Top-K by count using min-heap
        let mut heap: BinaryHeap<Reverse<(u32, u64)>> = BinaryHeap::new();
        for (row_id, (count, _)) in &result_map {
            if heap.len() < max_results {
                heap.push(Reverse((*count, *row_id)));
            } else if let Some(&Reverse((min_count, _))) = heap.peek() {
                if *count > min_count {
                    heap.pop();
                    heap.push(Reverse((*count, *row_id)));
                }
            }
        }

        let top_k: Vec<(u64, u32)> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse((count, row_id))| (row_id, count))
            .collect();

        let mut results: Vec<(u64, u32, Vec<u64>)> = top_k
            .into_iter()
            .map(|(row_id, count)| {
                let (_, positions) = result_map.remove(&row_id).unwrap_or_default();
                (row_id, count, positions)
            })
            .collect();
        results.reverse();
        Ok(results)
    }

    /// Search for matching rows without scoring (legacy API, used by ScalarIndex::search).
    /// Returns deduplicated, sorted row IDs.
    pub async fn search_rows(
        &self,
        query: &[u8],
        max_results: usize,
    ) -> Result<Vec<u64>> {
        let scored = self.search_rows_scored(query, max_results).await?;
        Ok(scored.into_iter().map(|(row_id, _, _)| row_id).collect())
    }

    /// Compute conditional probability P(continuation | prompt)
    /// aggregated across all segments.
    pub async fn compute_prob(
        &self,
        prompt: &[u8],
        continuation: &[u8],
    ) -> Result<query::ProbResult> {
        let prompt_cnt = self.total_count(prompt).await?;
        if prompt_cnt == 0 {
            return Ok(query::ProbResult {
                prompt_cnt: 0,
                cont_cnt: 0,
                prob: 0.0,
            });
        }

        let mut full_query = Vec::with_capacity(prompt.len() + continuation.len());
        full_query.extend_from_slice(prompt);
        full_query.extend_from_slice(continuation);
        let cont_cnt = self.total_count(&full_query).await?;
        let prob = cont_cnt as f64 / prompt_cnt as f64;

        Ok(query::ProbResult {
            prompt_cnt,
            cont_cnt,
            prob,
        })
    }

    /// Compute next-byte distribution after a prompt, merged across
    /// all segments.
    pub async fn compute_ntd(
        &self,
        prompt: &[u8],
        max_support: Option<u64>,
    ) -> Result<query::NtdResult> {
        let concurrency = parallel_query_concurrency(self.segment_metas.len());

        // Query all segments concurrently
        let futures: Vec<_> = self
            .segment_metas
            .iter()
            .map(|meta| self.ntd_segment(meta, prompt, max_support))
            .collect();
        let segment_results: Vec<query::NtdResult> = stream::iter(futures)
            .buffer_unordered(concurrency)
            .try_collect()
            .await?;

        // Sequential merge
        let mut total_byte_counts: HashMap<u8, u64> = HashMap::new();
        let mut total_prompt_cnt: u64 = 0;
        let mut any_approximate = false;
        for result in segment_results {
            total_prompt_cnt += result.prompt_cnt;
            any_approximate |= result.approximate;
            for entry in &result.distribution {
                *total_byte_counts.entry(entry.byte_value).or_insert(0) += entry.count;
            }
        }

        let mut distribution: Vec<query::NtdEntry> = total_byte_counts
            .into_iter()
            .map(|(byte_value, count)| query::NtdEntry {
                byte_value,
                count,
                prob: if total_prompt_cnt > 0 {
                    count as f64 / total_prompt_cnt as f64
                } else {
                    0.0
                },
            })
            .collect();
        distribution.sort_by(|a, b| b.count.cmp(&a.count));

        Ok(query::NtdResult {
            prompt_cnt: total_prompt_cnt,
            distribution,
            approximate: any_approximate,
        })
    }

    /// Compute infinity-gram probability with backoff, using
    /// cross-segment total counts for the binary lifting.
    pub async fn compute_infgram_prob(
        &self,
        prompt: &[u8],
        continuation: &[u8],
    ) -> Result<query::InfgramProbResult> {
        let prompt_len = prompt.len();

        if prompt_len == 0 {
            let result = self.compute_prob(&[], continuation).await?;
            return Ok(query::InfgramProbResult {
                prob_result: result,
                effective_suffix_len: 0,
            });
        }

        // Phase 1: Binary lifting — find where total count drops to 0
        let mut good_len = 0usize;
        let mut bad_len = prompt_len + 1;
        let mut power = 1usize;

        while power <= prompt_len {
            let suffix_start = prompt_len.saturating_sub(power);
            let suffix = &prompt[suffix_start..];
            let cnt = self.total_count(suffix).await?;
            if cnt > 0 {
                good_len = power;
                power *= 2;
            } else {
                bad_len = power;
                break;
            }
        }

        // Check if the full prompt has nonzero count
        if good_len == prompt_len || power > prompt_len {
            if good_len == 0 {
                return Ok(query::InfgramProbResult {
                    prob_result: query::ProbResult {
                        prompt_cnt: 0,
                        cont_cnt: 0,
                        prob: 0.0,
                    },
                    effective_suffix_len: 0,
                });
            }
            let full_cnt = self.total_count(prompt).await?;
            if full_cnt > 0 {
                good_len = prompt_len;
            }
        }

        // Phase 2: Binary search between good_len and bad_len
        while good_len + 1 < bad_len && bad_len <= prompt_len {
            let mid = good_len + (bad_len - good_len) / 2;
            let suffix_start = prompt_len - mid;
            let suffix = &prompt[suffix_start..];
            let cnt = self.total_count(suffix).await?;
            if cnt > 0 {
                good_len = mid;
            } else {
                bad_len = mid;
            }
        }

        // Phase 3: Compute prob using the effective suffix
        if good_len == 0 {
            return Ok(query::InfgramProbResult {
                prob_result: query::ProbResult {
                    prompt_cnt: 0,
                    cont_cnt: 0,
                    prob: 0.0,
                },
                effective_suffix_len: 0,
            });
        }

        let suffix_start = prompt_len - good_len;
        let effective_suffix = &prompt[suffix_start..];
        let result = self.compute_prob(effective_suffix, continuation).await?;

        Ok(query::InfgramProbResult {
            prob_result: result,
            effective_suffix_len: good_len,
        })
    }
}

// ---------------------------------------------------------------------------
// NoopIndexStore — placeholder for from_segments() (test/in-memory path)
// ---------------------------------------------------------------------------

/// A no-op IndexStore used when segments are pre-loaded in memory.
/// All operations return errors because they should never be called.
#[derive(Debug, DeepSizeOf)]
struct NoopIndexStore;

#[async_trait]
impl IndexStore for NoopIndexStore {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_arc(&self) -> Arc<dyn IndexStore> {
        Arc::new(NoopIndexStore)
    }

    fn io_parallelism(&self) -> usize {
        1
    }

    async fn new_index_file(
        &self,
        _name: &str,
        _schema: Arc<arrow_schema::Schema>,
    ) -> Result<Box<dyn crate::scalar::IndexWriter>> {
        Err(Error::not_supported("NoopIndexStore: cannot create files"))
    }

    async fn open_index_file(
        &self,
        _name: &str,
    ) -> Result<Arc<dyn crate::scalar::IndexReader>> {
        Err(Error::not_supported(
            "NoopIndexStore: segments should be in cache",
        ))
    }

    async fn copy_index_file(&self, _name: &str, _dest: &dyn IndexStore) -> Result<()> {
        Err(Error::not_supported("NoopIndexStore: cannot copy files"))
    }

    async fn rename_index_file(&self, _name: &str, _new_name: &str) -> Result<()> {
        Err(Error::not_supported("NoopIndexStore: cannot rename files"))
    }

    async fn delete_index_file(&self, _name: &str) -> Result<()> {
        Err(Error::not_supported("NoopIndexStore: cannot delete files"))
    }

    async fn list_files_with_sizes(&self) -> Result<Vec<crate::scalar::IndexFile>> {
        Ok(Vec::new())
    }
}
