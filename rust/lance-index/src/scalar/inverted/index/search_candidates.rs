// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[derive(Debug)]
pub(in super::super) struct PartitionCandidates<C> {
    pub(super) tokens_by_position: Vec<String>,
    pub(super) grouped_expansions: Vec<GroupedExpansionTerms>,
    pub(super) candidates: Vec<DocCandidate<C>>,
}

pub(super) struct ModernSearchRequest<'a> {
    pub(super) tokens: Arc<Tokens>,
    pub(super) params: Arc<FtsSearchParams>,
    pub(super) operator: Operator,
    pub(super) mask: Arc<RowAddrMask>,
    pub(super) metrics: Arc<dyn MetricsCollector>,
    pub(super) scorer: &'a MemBM25Scorer,
    pub(super) impact_scorer: Arc<MemBM25Scorer>,
    pub(super) limit: usize,
}

/// Typed identity for one modern candidate after partition-local scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PartitionDocId {
    pub(super) partition_ordinal: u32,
    pub(super) doc_id: DocId,
}

impl PartitionDocId {
    pub(super) fn try_new(partition_ordinal: usize, doc_id: DocId) -> Result<Self> {
        Ok(Self {
            partition_ordinal: u32::try_from(partition_ordinal).map_err(|_| {
                Error::index(format!(
                    "FTS partition ordinal {partition_ordinal} exceeds candidate identity capacity"
                ))
            })?,
            doc_id,
        })
    }

    pub(super) fn partition_ordinal(self) -> usize {
        self.partition_ordinal as usize
    }
}

#[derive(Debug, Clone)]
pub(super) struct ScoredPartitionDoc {
    pub(super) document: PartitionDocId,
    pub(super) score: OrderedFloat,
}

impl ScoredPartitionDoc {
    fn new(document: PartitionDocId, score: f32) -> Self {
        Self {
            document,
            score: OrderedFloat(score),
        }
    }
}

impl PartialEq for ScoredPartitionDoc {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredPartitionDoc {}

impl PartialOrd for ScoredPartitionDoc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredPartitionDoc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.cmp(&other.score)
    }
}

pub(super) const MAX_CONCURRENT_ADDRESS_READ_BYTES: usize = 64 * 1024 * 1024;

pub(super) fn address_read_concurrency(io_parallelism: usize, largest_read_bytes: usize) -> usize {
    let io_parallelism = io_parallelism.max(1);
    if largest_read_bytes == 0 {
        return io_parallelism;
    }
    io_parallelism.min(
        MAX_CONCURRENT_ADDRESS_READ_BYTES
            .checked_div(largest_read_bytes)
            .unwrap_or(0)
            .max(1),
    )
}

/// Merge one legacy candidate into the global top-k.
///
/// Legacy candidates already carry row addresses, so the heap evicts on the full
/// `(score DESC, row_id ASC)` key and `into_sorted_vec` yields the final order
/// directly. A score tie with a lower row_id wins, a higher one loses, whatever
/// order candidates arrive in.
pub(super) fn push_scored_key(
    candidates: &mut BinaryHeap<Reverse<ScoredDoc>>,
    limit: usize,
    key: u64,
    score: f32,
) {
    // A NaN score is never competitive. It has to be rejected before the total
    // order sees it, because `total_cmp` ranks NaN above every real score.
    if score.is_nan() {
        return;
    }
    let candidate = ScoredDoc::new(key, score);
    if candidates.len() < limit {
        candidates.push(Reverse(candidate));
    } else if candidates.peek().is_some_and(|worst| worst.0 < candidate) {
        candidates.pop();
        candidates.push(Reverse(candidate));
    }
}

/// The global top-k of a modern search, as the merge can express it before row
/// addresses are known.
#[derive(Default)]
pub(super) struct ModernCandidates {
    /// The best `limit` candidates by score.
    heap: BinaryHeap<Reverse<ScoredPartitionDoc>>,
    /// Candidates tied at the current k-th score that the heap has no room for.
    /// Cleared whenever the k-th score rises, so it holds just the current tie
    /// band and stays empty for a query without ties.
    tie_band: Vec<ScoredPartitionDoc>,
}

impl ModernCandidates {
    /// Merge one candidate.
    ///
    /// Modern candidates carry dense DocIds whose row addresses are resolved only
    /// after selection, so a k-th-score tie cannot be ordered here. Every tied
    /// candidate is retained instead, and [`rank_resolved_documents`] picks the
    /// exact top-k once the addresses are known.
    fn push(&mut self, limit: usize, document: PartitionDocId, score: f32) {
        if score.is_nan() {
            return;
        }
        let candidate = ScoredPartitionDoc::new(document, score);
        if self.heap.len() < limit {
            self.heap.push(Reverse(candidate));
            return;
        }
        let Some(Reverse(worst)) = self.heap.peek() else {
            return;
        };
        let worst_score = worst.score;
        match candidate.score.cmp(&worst_score) {
            // Below the k-th score: never competitive.
            std::cmp::Ordering::Less => {}
            // Tied at the k-th score: a potential winner once addresses resolve.
            std::cmp::Ordering::Equal => self.tie_band.push(candidate),
            std::cmp::Ordering::Greater => {
                let Some(Reverse(displaced)) = self.heap.pop() else {
                    return;
                };
                self.heap.push(Reverse(candidate));
                let kth = self.heap.peek().map(|Reverse(entry)| entry.score);
                if kth.is_some_and(|kth| kth > worst_score) {
                    // The k-th score rose past the band: none of its members can win.
                    self.tie_band.clear();
                } else {
                    self.tie_band.push(displaced);
                }
            }
        }
    }

    /// Candidates held for the final selection: the top-k plus the tie band.
    pub(super) fn buffered(&self) -> usize {
        self.heap.len() + self.tie_band.len()
    }

    /// The candidates whose addresses have to be resolved: the top-k plus the tie
    /// band. Returning more than `limit` is safe, the final ranking truncates.
    pub(super) fn into_vec(self) -> Vec<ScoredPartitionDoc> {
        let mut candidates = self
            .heap
            .into_vec()
            .into_iter()
            .map(|Reverse(candidate)| candidate)
            .collect::<Vec<_>>();
        candidates.extend(self.tie_band);
        candidates
    }

    /// Rebuild from candidates already ordered best first, as the exact top-k with
    /// an empty tie band.
    pub(super) fn from_ranked(ordered: Vec<ScoredPartitionDoc>) -> Self {
        Self {
            heap: ordered.into_iter().map(Reverse).collect(),
            tie_band: Vec::new(),
        }
    }
}

/// Order resolved FTS documents by `(score DESC, row_id ASC)` and keep the exact
/// top-k.
///
/// This is where the deterministic tiebreak is finally settled for modern
/// indexes: the merge above hands over the top-k plus its k-th-score tie group,
/// and the row addresses resolved in between decide which tied documents win.
pub(super) fn rank_resolved_documents(
    mut documents: Vec<ScoredDoc>,
    limit: usize,
) -> Vec<ScoredDoc> {
    documents.sort_unstable_by(|left, right| right.cmp(left));
    documents.truncate(limit);
    documents
}

/// One modern partition prepared for scoring.
pub(super) struct LoadedModernPartition {
    partition_ordinal: usize,
    part: Arc<InvertedPartition>,
    lengths: Arc<DocLengths>,
    visibility: DocVisibility,
    postings: Vec<PostingIterator>,
    wand_scorer: Option<Arc<MemBM25Scorer>>,
    threshold: Arc<AtomicU32>,
    tokens_by_position: Vec<String>,
    grouped_expansions: Vec<GroupedExpansionTerms>,
    /// Set on the retry that follows a k-th-score tie overflow, so the walk orders
    /// ties by row address instead of deferring them to the global merge.
    pub(super) addresses: Option<ResidentAddressProjection>,
}

/// Candidates one modern partition contributed to the global merge.
pub(super) struct ScoredModernPartition {
    pub(super) partition_ordinal: usize,
    pub(super) part: Arc<InvertedPartition>,
    pub(super) candidates: PartitionCandidates<DocId>,
    /// See [`Wand::score_floor_overflow`]. When set, `candidates` is missing part
    /// of the partition's k-th-score tie band and must not be merged.
    pub(super) score_floor_overflow: bool,
}

/// Load everything the CPU scoring phase needs for one modern partition.
///
/// `Ok(None)` means the partition cannot contribute: no matching postings, or
/// nothing visible under the prefilter.
#[allow(clippy::too_many_arguments)]
pub(super) async fn load_modern_partition(
    partition_ordinal: usize,
    part: Arc<InvertedPartition>,
    tokens: Arc<Tokens>,
    params: Arc<FtsSearchParams>,
    operator: Operator,
    mask: Arc<RowAddrMask>,
    metrics: Arc<dyn MetricsCollector>,
    impact_scorer: Arc<MemBM25Scorer>,
    impact_shared_threshold: Arc<AtomicU32>,
) -> Result<Option<LoadedModernPartition>> {
    let LoadedPostings {
        postings,
        grouped_expansions,
        impact_safe,
        exact_scoring_required,
    } = part
        .load_posting_lists(
            tokens.as_ref(),
            params.as_ref(),
            operator,
            impact_scorer.as_ref(),
            metrics.as_ref(),
            false,
        )
        .await?;
    if postings.is_empty() {
        return Ok(None);
    }
    let documents = part
        .docs
        .modern()
        .cloned()
        .ok_or_else(|| Error::internal("modern index contains legacy partition documents"))?;
    let materialize_selected = operator == Operator::Or
        && mask.max_len().is_some_and(|selected| {
            u128::from(selected).saturating_mul(100)
                <= u128::from(*FLAT_SEARCH_PERCENT_THRESHOLD)
                    .saturating_mul(documents.len() as u128)
        });
    let visibility = match documents.immediate_visibility(mask.clone(), materialize_selected) {
        Some(visibility) => visibility,
        None => {
            documents
                .visibility(mask.clone(), materialize_selected)
                .await?
        }
    };
    if visibility.is_empty() {
        return Ok(None);
    }
    let lengths = match documents.cached_lengths() {
        Some(lengths) => lengths,
        None => documents.lengths().await?,
    };
    let max_position = postings
        .iter()
        .map(|posting| posting.term_index() as usize)
        .max()
        .unwrap_or_default();
    let mut tokens_by_position = vec![String::new(); max_position + 1];
    for posting in &postings {
        tokens_by_position[posting.term_index() as usize] = posting.token().to_owned();
    }
    let use_global_scorer = impact_safe || exact_scoring_required;
    let threshold = if use_global_scorer {
        impact_shared_threshold
    } else {
        Arc::new(AtomicU32::new(f32::NEG_INFINITY.to_bits()))
    };
    let wand_scorer = use_global_scorer.then_some(impact_scorer);
    Ok(Some(LoadedModernPartition {
        partition_ordinal,
        part,
        lengths,
        visibility,
        postings,
        wand_scorer,
        threshold,
        tokens_by_position,
        grouped_expansions,
        // A partition whose addresses are already in memory orders its k-th-score
        // ties during the walk. The rest defer them, and an oversized tie band
        // there is retried with the addresses loaded.
        addresses: documents.resident_address_projection(),
    }))
}

/// Run one modern partition's WAND search. Pure CPU work, so it can run on the
/// compute pool.
pub(super) fn score_modern_partition(
    partition: LoadedModernPartition,
    params: &FtsSearchParams,
    operator: Operator,
    metrics: &dyn MetricsCollector,
) -> Result<ScoredModernPartition> {
    let LoadedModernPartition {
        partition_ordinal,
        part,
        lengths,
        visibility,
        postings,
        wand_scorer,
        threshold,
        tokens_by_position,
        grouped_expansions,
        addresses,
    } = partition;
    let (candidates, score_floor_overflow) = part.bm25_search_modern(
        lengths.as_ref(),
        &visibility,
        addresses.as_ref(),
        params,
        operator,
        postings,
        wand_scorer,
        metrics,
        threshold,
    )?;
    Ok(ScoredModernPartition {
        partition_ordinal,
        part,
        candidates: PartitionCandidates {
            tokens_by_position,
            grouped_expansions,
            candidates,
        },
        score_floor_overflow,
    })
}

/// Rescore one partition's candidates with the corpus-wide statistics and merge
/// them into the global top-k.
pub(super) fn merge_modern_partition(
    ranked: &mut ModernCandidates,
    limit: usize,
    partition_ordinal: usize,
    candidates: PartitionCandidates<DocId>,
    scorer: &MemBM25Scorer,
    idf_cache: &mut HashMap<String, f32>,
) -> Result<()> {
    for (doc_id, score) in rescore_partition_candidates(candidates, scorer, idf_cache) {
        ranked.push(
            limit,
            PartitionDocId::try_new(partition_ordinal, doc_id)?,
            score,
        );
    }
    Ok(())
}

pub(super) fn rescore_partition_candidates<C>(
    partition: PartitionCandidates<C>,
    scorer: &MemBM25Scorer,
    idf_cache: &mut HashMap<String, f32>,
) -> Vec<(C, f32)> {
    let PartitionCandidates {
        tokens_by_position,
        grouped_expansions,
        candidates,
    } = partition;
    let idf_by_position = tokens_by_position
        .iter()
        .map(|token| {
            *idf_cache
                .entry(token.clone())
                .or_insert_with(|| scorer.query_weight(token))
        })
        .collect::<Vec<_>>();
    let grouped_positions = grouped_expansions
        .iter()
        .map(|group| group.position)
        .collect::<HashSet<_>>();

    candidates
        .into_iter()
        .map(
            |DocCandidate {
                 document,
                 posting_doc_id,
                 freqs,
                 doc_length,
             }| {
                let mut score = 0.0;
                for (term_index, freq) in freqs {
                    if grouped_positions.contains(&term_index) {
                        continue;
                    }
                    debug_assert!((term_index as usize) < idf_by_position.len());
                    score +=
                        idf_by_position[term_index as usize] * scorer.doc_weight(freq, doc_length);
                }
                for group in &grouped_expansions {
                    for term in group.terms.iter() {
                        let Some(freq) = term.frequency(posting_doc_id) else {
                            continue;
                        };
                        score += term.query_weight() * scorer.doc_weight(freq, doc_length);
                    }
                }
                (document, score)
            },
        )
        .collect()
}

#[derive(Debug)]
pub(in super::super) struct LoadedPostings {
    pub(in super::super) postings: Vec<PostingIterator>,
    pub(super) grouped_expansions: Vec<GroupedExpansionTerms>,
    pub(super) impact_safe: bool,
    pub(super) exact_scoring_required: bool,
}

pub(super) enum LoadedDocLengths {
    Legacy(Arc<DocSet>),
    Modern(Arc<DocLengths>),
}

impl LoadedDocLengths {
    pub(super) fn scoring_num_tokens(&self, doc_id: u32) -> u32 {
        match self {
            Self::Legacy(docs) => docs.scoring_num_tokens(doc_id),
            Self::Modern(lengths) => lengths.scoring(DocId::new(doc_id)),
        }
    }

    pub(super) fn num_tokens_by_row_id(&self, row_id: u64) -> u32 {
        match self {
            Self::Legacy(docs) => docs.num_tokens_by_row_id(row_id),
            Self::Modern(_) => unreachable!("modern posting lists use dense DocIds"),
        }
    }
}

impl LoadedPostings {
    pub(super) fn empty() -> Self {
        Self {
            postings: Vec::new(),
            grouped_expansions: Vec::new(),
            impact_safe: false,
            exact_scoring_required: false,
        }
    }
}

#[derive(Debug)]
pub(super) struct GroupedExpansionTerms {
    pub(super) position: u32,
    pub(super) terms: Arc<[GroupedTermScorer]>,
}
