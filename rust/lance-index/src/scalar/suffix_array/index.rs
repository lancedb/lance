// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Multi-segment SuffixArrayIndex implementing the ScalarIndex trait.
//!
//! A suffix array index may contain one or more segments. Each segment
//! covers a group of fragments and has its own text buffer and suffix
//! array. Query methods (count, prob, ntd, infgram_prob) transparently
//! aggregate results across all segments.
//!
//! This design is necessary because suffix arrays use u32 pointers,
//! limiting each segment to ~4 GB of text. For larger corpora the
//! plugin splits the data into multiple segments during build.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use lance_core::utils::mask::RowAddrTreeMap;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;

use crate::frag_reuse::FragReuseIndex;
use crate::metrics::MetricsCollector;
use crate::scalar::registry::{TrainingCriteria, TrainingOrdering};
use crate::scalar::{
    AnyQuery, CreatedIndex, IndexStore, OldIndexDataFilter, ScalarIndex, ScalarIndexParams,
    SearchResult, UpdateCriteria,
};
use crate::vector::VectorIndex;
use crate::{Index, IndexType};

use super::query::{self, SuffixArrayQuery};

/// One segment of a suffix array index.
///
/// Each segment holds a contiguous text buffer and its corresponding
/// suffix array. The text is at most ~2 GB so that u32 pointers suffice.
#[derive(Debug)]
pub struct SuffixArraySegment {
    /// The raw tokenized corpus data for this segment.
    pub tokenized: Bytes,
    /// The compacted suffix array (variable-width pointers).
    pub suffix_array: Bytes,
    /// Bytes per pointer in the suffix array.
    pub pointer_width: u8,
    /// Total number of entries in the suffix array.
    pub total_entries: u64,
    /// Cumulative byte offsets marking the end of each document.
    /// `doc_offsets[i]` is the byte offset where document `i` ends.
    /// Document `i` spans `[doc_offsets[i-1], doc_offsets[i])` (with doc_offsets[-1] = 0).
    /// Empty if document retrieval is not available (legacy indices).
    pub doc_offsets: Vec<u64>,
}

impl SuffixArraySegment {
    /// Count occurrences of a pattern in this segment.
    #[inline]
    pub fn count(&self, query: &[u8]) -> u64 {
        query::count(
            &self.tokenized,
            &self.suffix_array,
            self.pointer_width as usize,
            self.total_entries,
            query,
        )
    }
}

/// A suffix array index loaded into memory for querying.
///
/// May contain one or more segments. All query methods aggregate
/// transparently across segments.
pub struct SuffixArrayIndex {
    segments: Vec<SuffixArraySegment>,
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
            .field("num_segments", &self.segments.len())
            .field(
                "total_bytes",
                &self
                    .segments
                    .iter()
                    .map(|s| s.tokenized.len())
                    .sum::<usize>(),
            )
            .finish()
    }
}

impl DeepSizeOf for SuffixArrayIndex {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.segments
            .iter()
            .map(|s| {
                s.tokenized.len()
                    + s.suffix_array.len()
                    + s.doc_offsets.len() * std::mem::size_of::<u64>()
            })
            .sum()
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
            "num_segments": self.segments.len(),
            "total_tokenized_bytes": self.segments.iter().map(|s| s.tokenized.len()).sum::<usize>(),
            "total_entries": self.segments.iter().map(|s| s.total_entries).sum::<u64>(),
            "case_insensitive": self.case_insensitive,
            "token_width": self.token_width,
        }))
    }

    async fn prewarm(&self) -> Result<()> {
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
                let n = self.total_count(&qb);
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
                let row_ids = self.search_rows(&qb, *max_results);
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
                let result = self.compute_prob(&pb, &cb);
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
                let result = self.compute_ntd(&pb, *max_support);
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
                let result = self.compute_infgram_prob(&pb, &cb);
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
    pub fn new(
        tokenized: Bytes,
        suffix_array: Bytes,
        pointer_width: u8,
        total_entries: u64,
    ) -> Self {
        Self {
            segments: vec![SuffixArraySegment {
                tokenized,
                suffix_array,
                pointer_width,
                total_entries,
                doc_offsets: Vec::new(),
            }],
            case_insensitive: false,
            token_width: 1,
        }
    }

    /// Whether this index was built with case-insensitive (lowercased) text.
    pub fn case_insensitive(&self) -> bool {
        self.case_insensitive
    }

    /// Token width (bytes per token unit). 1 = byte-level, 2 = int16, 4 = int32.
    pub fn token_width(&self) -> u8 {
        self.token_width
    }

    /// Create a multi-segment SuffixArrayIndex from pre-loaded segments.
    pub fn from_segments(segments: Vec<SuffixArraySegment>) -> Self {
        Self { segments, case_insensitive: false, token_width: 1 }
    }

    /// Load a single-segment SuffixArrayIndex from an IndexStore (v0 format).
    pub async fn load(
        store: Arc<dyn IndexStore>,
        _frag_reuse_index: Option<Arc<FragReuseIndex>>,
        _cache: &lance_core::cache::LanceCache,
        pointer_width: u8,
        total_entries: u64,
        case_insensitive: bool,
        token_width: u8,
    ) -> Result<Arc<Self>> {
        let seg = Self::load_segment(store.as_ref(), "tokenized.bin", "suffix_array.bin", None).await?;
        // Override metadata from protobuf (v0 format stores it at top level)
        Ok(Arc::new(Self {
            segments: vec![SuffixArraySegment {
                tokenized: seg.tokenized,
                suffix_array: seg.suffix_array,
                pointer_width,
                total_entries,
                doc_offsets: seg.doc_offsets,
            }],
            case_insensitive,
            token_width,
        }))
    }

    /// Load a multi-segment SuffixArrayIndex from an IndexStore (v1+ format).
    pub async fn load_multi(
        store: Arc<dyn IndexStore>,
        segment_infos: &[crate::pb::SuffixArraySegmentInfo],
        case_insensitive: bool,
        token_width: u8,
    ) -> Result<Arc<Self>> {
        let mut segments = Vec::with_capacity(segment_infos.len());
        for (i, info) in segment_infos.iter().enumerate() {
            let tok_name = format!("segment_{i}_tokenized.bin");
            let sa_name = format!("segment_{i}_suffix_array.bin");
            let offsets_name = format!("segment_{i}_doc_offsets.bin");
            let seg = Self::load_segment(
                store.as_ref(),
                &tok_name,
                &sa_name,
                Some(&offsets_name),
            ).await?;
            segments.push(SuffixArraySegment {
                tokenized: seg.tokenized,
                suffix_array: seg.suffix_array,
                pointer_width: info.pointer_width as u8,
                total_entries: info.total_entries,
                doc_offsets: seg.doc_offsets,
            });
        }
        Ok(Arc::new(Self { segments, case_insensitive, token_width }))
    }

    /// Load one segment's files from the store.
    async fn load_segment(
        store: &dyn IndexStore,
        tok_filename: &str,
        sa_filename: &str,
        offsets_filename: Option<&str>,
    ) -> Result<SuffixArraySegment> {
        let tokenized_reader = store.open_index_file(tok_filename).await?;
        let sa_reader = store.open_index_file(sa_filename).await?;

        let tokenized_batch = tokenized_reader.read_record_batch(0, 1).await?;
        let tokenized_col = tokenized_batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::LargeBinaryArray>()
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "{tok_filename} should contain a LargeBinary column"
                ))
            })?;
        let tokenized = Bytes::copy_from_slice(tokenized_col.value(0));

        let sa_batch = sa_reader.read_record_batch(0, 1).await?;
        let sa_col = sa_batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::LargeBinaryArray>()
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "{sa_filename} should contain a LargeBinary column"
                ))
            })?;
        let suffix_array = Bytes::copy_from_slice(sa_col.value(0));

        // Try to load doc_offsets (optional — may not exist for legacy indices)
        let doc_offsets = if let Some(offsets_name) = offsets_filename {
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

        // Derive metadata from loaded data (will be overridden by caller
        // with protobuf metadata for v0 format).
        Ok(SuffixArraySegment {
            pointer_width: 4,
            total_entries: 0,
            tokenized,
            suffix_array,
            doc_offsets,
        })
    }

    // ─── Multi-segment query aggregation ─────────────────────────────────────

    /// Count occurrences of a pattern across all segments.
    /// For token-level indices (`token_width > 1`), only counts matches
    /// at token-aligned positions.
    pub fn total_count(&self, query: &[u8]) -> u64 {
        if self.token_width <= 1 {
            // Byte-level: use fast count (SA range size)
            self.segments.iter().map(|seg| seg.count(query)).sum()
        } else {
            // Token-level: must check alignment of each match
            let tw = self.token_width as usize;
            let mut total = 0u64;
            for seg in &self.segments {
                let (lo, hi) = query::sa_find(
                    &seg.tokenized,
                    &seg.suffix_array,
                    seg.pointer_width as usize,
                    seg.total_entries,
                    query,
                );
                for rank in lo..hi {
                    let byte_pos = query::read_pointer(
                        &seg.suffix_array,
                        rank,
                        seg.pointer_width as usize,
                    );
                    if (byte_pos as usize) % tw == 0 {
                        total += 1;
                    }
                }
            }
            total
        }
    }

    /// Search for byte positions matching a pattern and resolve them to
    /// row (document) indices using doc_offsets. Returns deduplicated,
    /// sorted row indices.
    ///
    /// For token-level indices (`token_width > 1`), only matches at
    /// token-aligned positions (`pos % token_width == 0`) are included.
    pub fn search_rows(
        &self,
        query: &[u8],
        max_results: usize,
    ) -> Vec<u64> {
        let tw = self.token_width as usize;
        let mut row_ids = Vec::new();
        let mut remaining = max_results;
        let mut doc_base: u64 = 0; // cumulative doc count across segments

        for seg in &self.segments {
            if remaining == 0 {
                break;
            }
            let (lo, hi) = query::sa_find(
                &seg.tokenized,
                &seg.suffix_array,
                seg.pointer_width as usize,
                seg.total_entries,
                query,
            );

            if seg.doc_offsets.is_empty() {
                // No doc_offsets — can't resolve to rows
                let seg_results = ((hi - lo) as usize).min(remaining);
                remaining = remaining.saturating_sub(seg_results);
            } else {
                for rank in lo..hi {
                    if remaining == 0 {
                        break;
                    }
                    let byte_pos = query::read_pointer(
                        &seg.suffix_array,
                        rank,
                        seg.pointer_width as usize,
                    );
                    // For token-level SA, skip matches not aligned to token boundaries
                    if tw > 1 && (byte_pos as usize) % tw != 0 {
                        continue;
                    }
                    // Binary search: find first offset > byte_pos
                    let local_doc = seg.doc_offsets.partition_point(|&o| o <= byte_pos) as u64;
                    row_ids.push(doc_base + local_doc);
                    remaining -= 1;
                }
            }

            doc_base += seg.doc_offsets.len() as u64;
        }

        // Deduplicate and sort
        row_ids.sort_unstable();
        row_ids.dedup();
        row_ids
    }

    /// Compute conditional probability P(continuation | prompt)
    /// aggregated across all segments.
    pub fn compute_prob(
        &self,
        prompt: &[u8],
        continuation: &[u8],
    ) -> query::ProbResult {
        let prompt_cnt = self.total_count(prompt);
        if prompt_cnt == 0 {
            return query::ProbResult {
                prompt_cnt: 0,
                cont_cnt: 0,
                prob: 0.0,
            };
        }

        let mut full_query = Vec::with_capacity(prompt.len() + continuation.len());
        full_query.extend_from_slice(prompt);
        full_query.extend_from_slice(continuation);
        let cont_cnt = self.total_count(&full_query);
        let prob = cont_cnt as f64 / prompt_cnt as f64;

        query::ProbResult {
            prompt_cnt,
            cont_cnt,
            prob,
        }
    }

    /// Compute next-byte distribution after a prompt, merged across
    /// all segments.
    pub fn compute_ntd(
        &self,
        prompt: &[u8],
        max_support: Option<u64>,
    ) -> query::NtdResult {
        let mut total_byte_counts: HashMap<u8, u64> = HashMap::new();
        let mut total_prompt_cnt: u64 = 0;
        let mut any_approximate = false;

        for seg in &self.segments {
            let result = query::next_byte_distribution(
                &seg.tokenized,
                &seg.suffix_array,
                seg.pointer_width as usize,
                seg.total_entries,
                prompt,
                max_support,
            );
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

        query::NtdResult {
            prompt_cnt: total_prompt_cnt,
            distribution,
            approximate: any_approximate,
        }
    }

    /// Compute infinity-gram probability with backoff, using
    /// cross-segment total counts for the binary lifting.
    pub fn compute_infgram_prob(
        &self,
        prompt: &[u8],
        continuation: &[u8],
    ) -> query::InfgramProbResult {
        let prompt_len = prompt.len();

        if prompt_len == 0 {
            let result = self.compute_prob(&[], continuation);
            return query::InfgramProbResult {
                prob_result: result,
                effective_suffix_len: 0,
            };
        }

        // Phase 1: Binary lifting — find where total count drops to 0
        let mut good_len = 0usize;
        let mut bad_len = prompt_len + 1;
        let mut power = 1usize;

        while power <= prompt_len {
            let suffix_start = prompt_len.saturating_sub(power);
            let suffix = &prompt[suffix_start..];
            let cnt = self.total_count(suffix);
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
                return query::InfgramProbResult {
                    prob_result: query::ProbResult {
                        prompt_cnt: 0,
                        cont_cnt: 0,
                        prob: 0.0,
                    },
                    effective_suffix_len: 0,
                };
            }
            let full_cnt = self.total_count(prompt);
            if full_cnt > 0 {
                good_len = prompt_len;
            }
        }

        // Phase 2: Binary search between good_len and bad_len
        while good_len + 1 < bad_len && bad_len <= prompt_len {
            let mid = good_len + (bad_len - good_len) / 2;
            let suffix_start = prompt_len - mid;
            let suffix = &prompt[suffix_start..];
            let cnt = self.total_count(suffix);
            if cnt > 0 {
                good_len = mid;
            } else {
                bad_len = mid;
            }
        }

        // Phase 3: Compute prob using the effective suffix
        if good_len == 0 {
            return query::InfgramProbResult {
                prob_result: query::ProbResult {
                    prompt_cnt: 0,
                    cont_cnt: 0,
                    prob: 0.0,
                },
                effective_suffix_len: 0,
            };
        }

        let suffix_start = prompt_len - good_len;
        let effective_suffix = &prompt[suffix_start..];
        let result = self.compute_prob(effective_suffix, continuation);

        query::InfgramProbResult {
            prob_result: result,
            effective_suffix_len: good_len,
        }
    }
}
