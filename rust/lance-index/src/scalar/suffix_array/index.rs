// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SuffixArrayIndex implementing the ScalarIndex trait.

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

/// A suffix array index loaded into memory for querying.
pub struct SuffixArrayIndex {
    /// The raw tokenized corpus data.
    tokenized: Bytes,
    /// The compacted suffix array (variable-width pointers).
    suffix_array: Bytes,
    /// Bytes per pointer in the suffix array.
    pointer_width: u8,
    /// Total number of entries in the suffix array.
    total_entries: u64,
}

impl std::fmt::Debug for SuffixArrayIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SuffixArrayIndex")
            .field("tokenized_bytes", &self.tokenized.len())
            .field("suffix_array_bytes", &self.suffix_array.len())
            .field("pointer_width", &self.pointer_width)
            .field("total_entries", &self.total_entries)
            .finish()
    }
}

impl DeepSizeOf for SuffixArrayIndex {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.tokenized.len() + self.suffix_array.len()
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
            "tokenized_bytes": self.tokenized.len(),
            "total_entries": self.total_entries,
            "pointer_width": self.pointer_width,
        }))
    }

    async fn prewarm(&self) -> Result<()> {
        // Data is already loaded in memory
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::SuffixArray
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        // Suffix array indexes do not track fragment membership in a way
        // that can be enumerated without the original row-to-fragment mapping.
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
                let n = query::count(
                    &self.tokenized,
                    &self.suffix_array,
                    self.pointer_width as usize,
                    self.total_entries,
                    query_bytes,
                );
                // Return the count encoded as a single row address.
                // The count value is stored directly as a "row address" for the caller
                // to interpret. This is a convention for count-only queries.
                let mut map = RowAddrTreeMap::new();
                if n > 0 {
                    // Store count as a synthetic row address (fragment 0, row n)
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
                let (lo, hi) = query::sa_find(
                    &self.tokenized,
                    &self.suffix_array,
                    self.pointer_width as usize,
                    self.total_entries,
                    query_bytes,
                );

                let num_results = ((hi - lo) as usize).min(*max_results);
                let mut map = RowAddrTreeMap::new();
                for rank in lo..lo + num_results as u64 {
                    let pos =
                        query::read_pointer(&self.suffix_array, rank, self.pointer_width as usize);
                    map.insert(pos);
                }
                Ok(SearchResult::Exact(
                    lance_core::utils::mask::NullableRowAddrSet::new(map, Default::default()),
                ))
            }
            SuffixArrayQuery::Prob {
                prompt_bytes,
                continuation_bytes,
            } => {
                let result = query::prob(
                    &self.tokenized,
                    &self.suffix_array,
                    self.pointer_width as usize,
                    self.total_entries,
                    prompt_bytes,
                    continuation_bytes,
                );
                // Encode prob result: store cont_cnt as row address.
                // The caller extracts the probability via the query API.
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
                let result = query::next_byte_distribution(
                    &self.tokenized,
                    &self.suffix_array,
                    self.pointer_width as usize,
                    self.total_entries,
                    prompt_bytes,
                    *max_support,
                );
                // Encode the prompt count as the row address.
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
                let result = query::infgram_prob(
                    &self.tokenized,
                    &self.suffix_array,
                    self.pointer_width as usize,
                    self.total_entries,
                    prompt_bytes,
                    continuation_bytes,
                );
                // Encode cont_cnt as row address (same convention as Prob).
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
        // Suffix arrays cannot be incrementally remapped; they must be rebuilt
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
        // Suffix arrays require a full rebuild with all data
        UpdateCriteria::requires_old_data(TrainingCriteria::new(TrainingOrdering::None))
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        Ok(ScalarIndexParams::new("suffixarray".to_string()))
    }
}

impl SuffixArrayIndex {
    /// Create a new SuffixArrayIndex from pre-loaded data.
    pub fn new(
        tokenized: Bytes,
        suffix_array: Bytes,
        pointer_width: u8,
        total_entries: u64,
    ) -> Self {
        Self {
            tokenized,
            suffix_array,
            pointer_width,
            total_entries,
        }
    }

    /// Load a SuffixArrayIndex from an IndexStore.
    pub async fn load(
        store: Arc<dyn IndexStore>,
        _frag_reuse_index: Option<Arc<FragReuseIndex>>,
        _cache: &lance_core::cache::LanceCache,
        pointer_width: u8,
        total_entries: u64,
    ) -> Result<Arc<Self>> {
        let tokenized_reader = store.open_index_file("tokenized.bin").await?;
        let sa_reader = store.open_index_file("suffix_array.bin").await?;

        // Read the binary data from single-row LargeBinary record batches
        let tokenized_batch = tokenized_reader.read_record_batch(0, 1).await?;
        let tokenized_col = tokenized_batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::LargeBinaryArray>()
            .ok_or_else(|| {
                Error::invalid_input("tokenized.bin should contain a LargeBinary column")
            })?;
        let tokenized = Bytes::copy_from_slice(tokenized_col.value(0));

        let sa_batch = sa_reader.read_record_batch(0, 1).await?;
        let sa_col = sa_batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::LargeBinaryArray>()
            .ok_or_else(|| {
                Error::invalid_input("suffix_array.bin should contain a LargeBinary column")
            })?;
        let suffix_array = Bytes::copy_from_slice(sa_col.value(0));

        Ok(Arc::new(Self {
            tokenized,
            suffix_array,
            pointer_width,
            total_entries,
        }))
    }

    /// Compute conditional probability P(continuation | prompt).
    pub fn compute_prob(
        &self,
        prompt: &[u8],
        continuation: &[u8],
    ) -> query::ProbResult {
        query::prob(
            &self.tokenized,
            &self.suffix_array,
            self.pointer_width as usize,
            self.total_entries,
            prompt,
            continuation,
        )
    }

    /// Compute next-byte distribution after a prompt.
    pub fn compute_ntd(
        &self,
        prompt: &[u8],
        max_support: Option<u64>,
    ) -> query::NtdResult {
        query::next_byte_distribution(
            &self.tokenized,
            &self.suffix_array,
            self.pointer_width as usize,
            self.total_entries,
            prompt,
            max_support,
        )
    }

    /// Compute infinity-gram probability with backoff.
    pub fn compute_infgram_prob(
        &self,
        prompt: &[u8],
        continuation: &[u8],
    ) -> query::InfgramProbResult {
        query::infgram_prob(
            &self.tokenized,
            &self.suffix_array,
            self.pointer_width as usize,
            self.total_entries,
            prompt,
            continuation,
        )
    }
}
