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
            .map(|s| s.tokenized.len() + s.suffix_array.len())
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

        match sa_query {
            SuffixArrayQuery::Count { query_bytes } => {
                let n = self.total_count(query_bytes);
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
                let mut map = RowAddrTreeMap::new();
                let mut remaining = *max_results;
                let mut text_offset: u64 = 0;

                for seg in &self.segments {
                    if remaining == 0 {
                        break;
                    }
                    let (lo, hi) = query::sa_find(
                        &seg.tokenized,
                        &seg.suffix_array,
                        seg.pointer_width as usize,
                        seg.total_entries,
                        query_bytes,
                    );
                    let seg_results = ((hi - lo) as usize).min(remaining);
                    for rank in lo..lo + seg_results as u64 {
                        let pos = query::read_pointer(
                            &seg.suffix_array,
                            rank,
                            seg.pointer_width as usize,
                        );
                        map.insert(text_offset + pos);
                    }
                    remaining -= seg_results;
                    text_offset += seg.tokenized.len() as u64;
                }
                Ok(SearchResult::Exact(
                    lance_core::utils::mask::NullableRowAddrSet::new(map, Default::default()),
                ))
            }
            SuffixArrayQuery::Prob {
                prompt_bytes,
                continuation_bytes,
            } => {
                let result = self.compute_prob(prompt_bytes, continuation_bytes);
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
                let result = self.compute_ntd(prompt_bytes, *max_support);
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
                let result = self.compute_infgram_prob(prompt_bytes, continuation_bytes);
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
            }],
        }
    }

    /// Create a multi-segment SuffixArrayIndex from pre-loaded segments.
    pub fn from_segments(segments: Vec<SuffixArraySegment>) -> Self {
        Self { segments }
    }

    /// Load a single-segment SuffixArrayIndex from an IndexStore (v0 format).
    pub async fn load(
        store: Arc<dyn IndexStore>,
        _frag_reuse_index: Option<Arc<FragReuseIndex>>,
        _cache: &lance_core::cache::LanceCache,
        pointer_width: u8,
        total_entries: u64,
    ) -> Result<Arc<Self>> {
        let seg = Self::load_segment(store.as_ref(), "tokenized.bin", "suffix_array.bin").await?;
        // Override metadata from protobuf (v0 format stores it at top level)
        Ok(Arc::new(Self {
            segments: vec![SuffixArraySegment {
                tokenized: seg.tokenized,
                suffix_array: seg.suffix_array,
                pointer_width,
                total_entries,
            }],
        }))
    }

    /// Load a multi-segment SuffixArrayIndex from an IndexStore (v1+ format).
    pub async fn load_multi(
        store: Arc<dyn IndexStore>,
        segment_infos: &[crate::pb::SuffixArraySegmentInfo],
    ) -> Result<Arc<Self>> {
        let mut segments = Vec::with_capacity(segment_infos.len());
        for (i, info) in segment_infos.iter().enumerate() {
            let tok_name = format!("segment_{i}_tokenized.bin");
            let sa_name = format!("segment_{i}_suffix_array.bin");
            let seg = Self::load_segment(store.as_ref(), &tok_name, &sa_name).await?;
            segments.push(SuffixArraySegment {
                tokenized: seg.tokenized,
                suffix_array: seg.suffix_array,
                pointer_width: info.pointer_width as u8,
                total_entries: info.total_entries,
            });
        }
        Ok(Arc::new(Self { segments }))
    }

    /// Load one segment's files from the store.
    async fn load_segment(
        store: &dyn IndexStore,
        tok_filename: &str,
        sa_filename: &str,
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

        // Derive metadata from loaded data (will be overridden by caller
        // with protobuf metadata for v0 format).
        Ok(SuffixArraySegment {
            pointer_width: 4,
            total_entries: 0,
            tokenized,
            suffix_array,
        })
    }

    // ─── Multi-segment query aggregation ─────────────────────────────────────

    /// Count occurrences of a pattern across all segments.
    pub fn total_count(&self, query: &[u8]) -> u64 {
        self.segments.iter().map(|seg| seg.count(query)).sum()
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
