// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-memory Full-Text Search (FTS) index with single-writer / multi-reader
//! semantics.
//!
//! # Concurrency model
//!
//! - **One writer** (`insert` / `insert_with_batch_position`) at a time per
//!   index. Callers are responsible for that invariant; this is consistent
//!   with `IndexStore`'s usage from `ShardWriter`.
//! - **Many readers** (`search*`, `expand_fuzzy`, `to_index_builder_reversed`)
//!   in parallel with the writer. Reads are lock-free aside from a brief
//!   tokenizer-pool checkout.
//! - **Per-batch monotonic visibility**: a reader either sees every row of
//!   batch `b` or none of them. A reader never sees a batch numbered above
//!   the published `visible_count`. BM25 statistics observed by a reader
//!   (`doc_count`, `total_tokens`, per-term `df`) are mutually consistent
//!   with the postings the reader walks.
//!
//! # Partition-structured layout
//!
//! The index is a set of immutable [`Partition`]s plus a bounded mutable
//! `tail` ([`TailIndex`]). Each partition holds a frozen slice of inserts as
//! on-disk-shaped posting lists and is queried with block-max WAND in
//! ≈O(matches). The tail accumulates recent inserts in a per-batch-chunk
//! layout and is searched in place; the writer freezes it into a new
//! partition once it crosses `freeze_threshold_rows`. The `{partitions, tail}`
//! pair lives behind one [`ArcSwap`] ([`IndexState`]) so a freeze publishes
//! atomically — a reader never sees a doc twice or misses one across a
//! freeze.
//!
//! Within the tail, visibility is published atomically by replacing a single
//! `Snapshot` via `ArcSwap`. The writer first installs all term chunks into
//! the per-term `ArcSwap<TermSlice>` slots, then atomically swaps in a new
//! `Snapshot` whose `visible_count` covers the new batch. Readers load the
//! `Snapshot` first and filter every term chunk by `batch_position <
//! snapshot.visible_count`. Frozen partitions are fully visible by
//! construction.
//!
//! # On-disk format
//!
//! At flush time we hand off to `lance_index::scalar::inverted::builder::InnerBuilder`
//! via `to_index_builder_reversed`, which merges every partition and the tail
//! into one builder. The on-disk format is unchanged from Lance's existing
//! inverted index.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use arc_swap::ArcSwap;
use arrow_array::builder::{Int32Builder, ListBuilder};
use arrow_array::{Array, Int32Array, LargeStringArray, RecordBatch, StringArray, StringViewArray};
use arrow_buffer::ScalarBuffer;
use arrow_schema::DataType;
use crossbeam_skiplist::SkipMap;
use lance_core::{Error, Result};
use lance_index::scalar::InvertedIndexParams;
use lance_index::scalar::inverted::query::{FtsSearchParams, Operator};
use lance_index::scalar::inverted::tokenizer::document_tokenizer::LanceTokenizer;
use lance_index::scalar::inverted::{
    DocSet, MemBM25Scorer, PlainPostingList, PostingList, Scorer, TokenSet, WandTerm, wand_search,
};
use lance_tokenizer::TokenStream;

use super::RowPosition;

// ============================================================================
// Public types preserved from previous API
// ============================================================================

/// In-memory FTS index entry returned from search.
#[derive(Debug, Clone)]
pub struct FtsEntry {
    /// Row position in MemTable.
    pub row_position: RowPosition,
    /// BM25 score for this document.
    pub score: f32,
}

/// Full-text search query expression for composable queries.
#[derive(Debug, Clone)]
pub enum FtsQueryExpr {
    /// Simple term match query.
    Match {
        /// The search query string.
        query: String,
        /// Boost factor applied to the score (default 1.0).
        boost: f32,
    },
    /// Phrase query with optional slop.
    Phrase {
        /// The phrase to search for.
        query: String,
        /// Maximum allowed distance between consecutive tokens.
        slop: u32,
        /// Boost factor applied to the score (default 1.0).
        boost: f32,
    },
    /// Fuzzy match query with typo tolerance.
    Fuzzy {
        /// The search query string.
        query: String,
        /// Maximum edit distance (Levenshtein distance).
        /// `None` means auto-fuzziness based on token length.
        fuzziness: Option<u32>,
        /// Maximum number of terms to expand to.
        max_expansions: usize,
        /// Boost factor applied to the score (default 1.0).
        boost: f32,
    },
    /// Boolean combination of queries.
    Boolean {
        /// All MUST clauses must match for a document to be included.
        must: Vec<Self>,
        /// At least one SHOULD clause should match (adds to score).
        should: Vec<Self>,
        /// No MUST_NOT clause may match (excludes documents).
        must_not: Vec<Self>,
    },
    /// Boosting query with a positive and an optional negative component.
    Boost {
        /// Documents must match this query.
        positive: Box<Self>,
        /// Optional query whose matches are demoted.
        negative: Option<Box<Self>>,
        /// Multiplier applied to documents matching `negative` (typically
        /// `< 1.0` to demote).
        negative_boost: f32,
    },
}

/// Default maximum number of fuzzy expansions.
pub const DEFAULT_MAX_EXPANSIONS: usize = 50;

/// Default WAND factor for full recall (no early termination).
pub const DEFAULT_WAND_FACTOR: f32 = 1.0;

/// Search options for controlling performance/recall tradeoffs.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// WAND factor for early termination (0.0 to 1.0).
    /// 1.0 = full recall (default).
    pub wand_factor: f32,
    /// Maximum number of results to return. None means unlimited.
    pub limit: Option<usize>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            wand_factor: DEFAULT_WAND_FACTOR,
            limit: None,
        }
    }
}

impl SearchOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the WAND factor for early termination.
    pub fn with_wand_factor(mut self, wand_factor: f32) -> Self {
        self.wand_factor = wand_factor.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum number of results to return.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

impl FtsQueryExpr {
    pub fn match_query(query: impl Into<String>) -> Self {
        Self::Match {
            query: query.into(),
            boost: 1.0,
        }
    }

    pub fn phrase(query: impl Into<String>) -> Self {
        Self::Phrase {
            query: query.into(),
            slop: 0,
            boost: 1.0,
        }
    }

    pub fn phrase_with_slop(query: impl Into<String>, slop: u32) -> Self {
        Self::Phrase {
            query: query.into(),
            slop,
            boost: 1.0,
        }
    }

    pub fn fuzzy(query: impl Into<String>) -> Self {
        Self::Fuzzy {
            query: query.into(),
            fuzziness: None,
            max_expansions: DEFAULT_MAX_EXPANSIONS,
            boost: 1.0,
        }
    }

    pub fn fuzzy_with_distance(query: impl Into<String>, fuzziness: u32) -> Self {
        Self::Fuzzy {
            query: query.into(),
            fuzziness: Some(fuzziness),
            max_expansions: DEFAULT_MAX_EXPANSIONS,
            boost: 1.0,
        }
    }

    pub fn fuzzy_with_options(
        query: impl Into<String>,
        fuzziness: Option<u32>,
        max_expansions: usize,
    ) -> Self {
        Self::Fuzzy {
            query: query.into(),
            fuzziness,
            max_expansions,
            boost: 1.0,
        }
    }

    pub fn boolean() -> BooleanQueryBuilder {
        BooleanQueryBuilder::new()
    }

    pub fn boosting(positive: Self) -> Self {
        Self::Boost {
            positive: Box::new(positive),
            negative: None,
            negative_boost: 1.0,
        }
    }

    pub fn boosting_with_negative(positive: Self, negative: Self, negative_boost: f32) -> Self {
        Self::Boost {
            positive: Box::new(positive),
            negative: Some(Box::new(negative)),
            negative_boost,
        }
    }

    pub fn with_boost(self, boost: f32) -> Self {
        match self {
            Self::Match { query, .. } => Self::Match { query, boost },
            Self::Phrase { query, slop, .. } => Self::Phrase { query, slop, boost },
            Self::Fuzzy {
                query,
                fuzziness,
                max_expansions,
                ..
            } => Self::Fuzzy {
                query,
                fuzziness,
                max_expansions,
                boost,
            },
            // Boolean and Boost don't carry a top-level boost field today.
            // Preserved as-is to keep behavior identical to the previous impl.
            other @ (Self::Boolean { .. } | Self::Boost { .. }) => other,
        }
    }
}

/// Auto-fuzziness based on token length:
/// 0–2 chars → 0, 3–5 chars → 1, 6+ chars → 2.
pub fn auto_fuzziness(token: &str) -> u32 {
    match token.chars().count() {
        0..=2 => 0,
        3..=5 => 1,
        _ => 2,
    }
}

/// Levenshtein distance using two-row dynamic programming.
pub fn levenshtein_distance(a: &str, b: &str) -> u32 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n as u32;
    }
    if n == 0 {
        return m as u32;
    }

    let mut prev_row: Vec<u32> = (0..=n as u32).collect();
    let mut curr_row: Vec<u32> = vec![0; n + 1];

    for (i, a_char) in a_chars.iter().enumerate() {
        curr_row[0] = (i + 1) as u32;
        for (j, b_char) in b_chars.iter().enumerate() {
            let cost = if a_char == b_char { 0 } else { 1 };
            curr_row[j + 1] = (prev_row[j + 1] + 1)
                .min(curr_row[j] + 1)
                .min(prev_row[j] + cost);
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[n]
}

/// Builder for constructing Boolean queries.
#[derive(Debug, Clone, Default)]
pub struct BooleanQueryBuilder {
    must: Vec<FtsQueryExpr>,
    should: Vec<FtsQueryExpr>,
    must_not: Vec<FtsQueryExpr>,
}

impl BooleanQueryBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn must(mut self, query: FtsQueryExpr) -> Self {
        self.must.push(query);
        self
    }

    pub fn should(mut self, query: FtsQueryExpr) -> Self {
        self.should.push(query);
        self
    }

    pub fn must_not(mut self, query: FtsQueryExpr) -> Self {
        self.must_not.push(query);
        self
    }

    pub fn build(self) -> FtsQueryExpr {
        FtsQueryExpr::Boolean {
            must: self.must,
            should: self.should,
            must_not: self.must_not,
        }
    }
}

// ============================================================================
// Internal types
// ============================================================================

/// Compressed sparse row (CSR) layout for per-document positions.
///
/// `offsets[i]..offsets[i+1]` is the slice of `data` belonging to the i-th
/// document in this `TermChunk`. `offsets.len() == row_positions.len() + 1`.
#[derive(Debug)]
struct Positions {
    offsets: Vec<u32>,
    data: Vec<u32>,
}

impl Positions {
    fn empty() -> Self {
        Self {
            offsets: vec![0],
            data: Vec::new(),
        }
    }

    fn push_doc(&mut self, positions: &[u32]) {
        self.data.extend_from_slice(positions);
        self.offsets.push(self.data.len() as u32);
    }

    fn doc_positions(&self, doc_idx: usize) -> &[u32] {
        let start = self.offsets[doc_idx] as usize;
        let end = self.offsets[doc_idx + 1] as usize;
        &self.data[start..end]
    }

    fn memory_size(&self) -> usize {
        self.offsets.capacity() * std::mem::size_of::<u32>()
            + self.data.capacity() * std::mem::size_of::<u32>()
    }
}

/// Postings produced by a single batch insert for a single term.
///
/// SoA layout for cache locality and trivial handoff to
/// `PostingListBuilder`. `row_positions` is sorted ascending.
#[derive(Debug)]
struct TermChunk {
    batch_position: usize,
    row_positions: Vec<u64>,
    frequencies: Vec<u32>,
    /// Per-doc token positions. Always `Some` in the current
    /// implementation (the writer tracks positions unconditionally so
    /// in-memory phrase queries work even when `params.has_positions()`
    /// is false). Kept as `Option` for forward compatibility with a
    /// future "no-positions" mode that drops the allocation.
    positions: Option<Positions>,
}

impl TermChunk {
    fn doc_count(&self) -> usize {
        self.row_positions.len()
    }

    fn memory_size(&self) -> usize {
        let base = std::mem::size_of::<Self>()
            + self.row_positions.capacity() * std::mem::size_of::<u64>()
            + self.frequencies.capacity() * std::mem::size_of::<u32>();
        base + self.positions.as_ref().map_or(0, Positions::memory_size)
    }
}

/// Append-only list of `TermChunk`s for a single term, replaced atomically
/// via `ArcSwap`.
#[derive(Debug, Default)]
struct TermSlice {
    chunks: Vec<Arc<TermChunk>>,
}

impl TermSlice {
    fn empty() -> Arc<Self> {
        Arc::new(Self { chunks: Vec::new() })
    }

    /// Returns a fresh `Arc<TermSlice>` containing all current chunks plus the
    /// new one. The previous slice is left unchanged so any reader holding
    /// it continues to see a consistent state.
    fn with_chunk_appended(&self, chunk: Arc<TermChunk>) -> Arc<Self> {
        let mut chunks = Vec::with_capacity(self.chunks.len() + 1);
        chunks.extend(self.chunks.iter().cloned());
        chunks.push(chunk);
        Arc::new(Self { chunks })
    }

    fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.chunks.capacity() * std::mem::size_of::<Arc<TermChunk>>()
            + self.chunks.iter().map(|c| c.memory_size()).sum::<usize>()
    }
}

/// Per-batch row metadata.
#[derive(Debug)]
struct BatchMeta {
    batch_position: usize,
    row_offset: u64,
    /// `doc_lengths[i]` is the token count of the row at `row_offset + i`.
    doc_lengths: Vec<u32>,
    rows: u32,
}

impl BatchMeta {
    fn dl(&self, row_position: u64) -> Option<u32> {
        if row_position < self.row_offset {
            return None;
        }
        let idx = (row_position - self.row_offset) as usize;
        self.doc_lengths.get(idx).copied()
    }

    fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>() + self.doc_lengths.capacity() * std::mem::size_of::<u32>()
    }
}

/// Atomic snapshot of the visible state. Replaced via `ArcSwap` after each
/// batch is fully linked.
#[derive(Debug)]
struct Snapshot {
    /// Number of batches visible to readers. `0` means empty index.
    visible_count: usize,
    /// Visible-batch metadata, written to in publish order. Always
    /// `batches.len() == visible_count` for any snapshot the writer has
    /// stored (each `publish_batch` appends a single entry and bumps
    /// `visible_count` by one). The slice is kept exactly the visible
    /// length so readers can iterate it directly without re-bounding.
    batches: Arc<[Arc<BatchMeta>]>,
    /// `Σ batches[i].rows` for `i < visible_count`.
    cumulative_doc_count: u64,
    /// `Σ batches[i].doc_lengths.iter().sum()` for `i < visible_count`.
    cumulative_total_tokens: u64,
}

impl Snapshot {
    fn empty() -> Arc<Self> {
        Arc::new(Self {
            visible_count: 0,
            batches: Arc::from(Vec::<Arc<BatchMeta>>::new().into_boxed_slice()),
            cumulative_doc_count: 0,
            cumulative_total_tokens: 0,
        })
    }

    fn batch_for(&self, batch_position: usize) -> Option<&Arc<BatchMeta>> {
        // The fast path assumes batch_position equals the index in
        // `batches` (true when callers use the no-arg `insert()` and let
        // the index assign sequential positions). When callers pass
        // explicit positions to `insert_with_batch_position`, those
        // positions can be sparse / out of order, so we fall back to a
        // linear search through visible batches.
        self.batches
            .get(batch_position)
            .filter(|m| m.batch_position == batch_position)
            .or_else(|| {
                self.batches[..self.visible_count]
                    .iter()
                    .find(|m| m.batch_position == batch_position)
            })
    }
}

/// Bounded pool of reader tokenizers with a writer-dedicated slot.
///
/// `LanceTokenizer::token_stream_for_*` takes `&mut self`, so each concurrent
/// caller needs its own tokenizer instance. Builds are cheap for English but
/// can load dictionaries for CJK tokenizers, so we amortize via the pool.
struct TokenizerPool {
    template: Box<dyn LanceTokenizer>,
    free: Mutex<Vec<Box<dyn LanceTokenizer>>>,
    cap: usize,
}

impl std::fmt::Debug for TokenizerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenizerPool")
            .field("cap", &self.cap)
            .finish()
    }
}

impl TokenizerPool {
    fn new(params: &InvertedIndexParams, cap: usize) -> Result<Self> {
        let template = params.build()?;
        Ok(Self {
            template,
            free: Mutex::new(Vec::new()),
            cap: cap.max(1),
        })
    }

    /// Acquire a tokenizer. Pops from the free list, otherwise clones the
    /// template.
    fn acquire(&self) -> Box<dyn LanceTokenizer> {
        if let Some(t) = self.free.lock().ok().and_then(|mut g| g.pop()) {
            return t;
        }
        self.template.box_clone()
    }

    /// Return a tokenizer to the pool, dropping it if the pool is at cap.
    fn release(&self, tokenizer: Box<dyn LanceTokenizer>) {
        if let Ok(mut g) = self.free.lock()
            && g.len() < self.cap
        {
            g.push(tokenizer);
        }
    }
}

/// RAII guard that returns the tokenizer to the pool on drop.
struct PooledTokenizer<'a> {
    pool: &'a TokenizerPool,
    inner: Option<Box<dyn LanceTokenizer>>,
}

impl<'a> PooledTokenizer<'a> {
    fn new(pool: &'a TokenizerPool) -> Self {
        Self {
            pool,
            inner: Some(pool.acquire()),
        }
    }

    fn get_mut(&mut self) -> &mut dyn LanceTokenizer {
        self.inner.as_mut().expect("tokenizer in scope").as_mut()
    }
}

impl<'a> Drop for PooledTokenizer<'a> {
    fn drop(&mut self) {
        if let Some(t) = self.inner.take() {
            self.pool.release(t);
        }
    }
}

// ============================================================================
// FtsMemIndex
// ============================================================================

/// In-memory full-text search index. See module docs for the concurrency
/// model and visibility contract.
/// The bounded mutable accumulator: recent inserts since the last freeze, in
/// the per-batch-chunk layout. Searched in place; frozen into a [`Partition`]
/// once it crosses the freeze threshold.
struct TailIndex {
    /// Per-term posting slices. `Arc<str>` interns the term so a single
    /// allocation backs every chunk that mentions it.
    terms: SkipMap<Arc<str>, ArcSwap<TermSlice>>,
    /// Atomically-swapped visibility snapshot.
    snapshot: ArcSwap<Snapshot>,
    /// Strictly-monotonic, dense, 0-based batch position counter. Dense
    /// positions are required by the `batch_position < visible_count`
    /// visibility filter; the tail therefore assigns its own.
    next_batch_position: AtomicUsize,
}

impl TailIndex {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            terms: SkipMap::new(),
            snapshot: ArcSwap::from(Snapshot::empty()),
            next_batch_position: AtomicUsize::new(0),
        })
    }

    fn snapshot(&self) -> Arc<Snapshot> {
        self.snapshot.load_full()
    }

    fn doc_count(&self) -> u64 {
        self.snapshot.load().cumulative_doc_count
    }

    fn visible_count(&self) -> usize {
        self.snapshot.load().visible_count
    }

    /// Next dense batch position for an incoming batch.
    fn next_position(&self) -> usize {
        self.next_batch_position.fetch_add(1, Ordering::Relaxed)
    }

    /// Install one batch's term chunks then publish a new visibility snapshot.
    fn append_batch(
        &self,
        batch_position: usize,
        row_offset: u64,
        rows: u32,
        doc_lengths: Vec<u32>,
        total_tokens: u64,
        term_builders: HashMap<Arc<str>, BatchTermBuilder>,
    ) {
        for (term, builder) in term_builders {
            let chunk = builder.build(batch_position);
            let entry = self
                .terms
                .get_or_insert_with(term, TermSlice::empty_arc_swap);
            let cur = entry.value().load();
            entry.value().store(cur.with_chunk_appended(chunk));
        }
        let new_meta = Arc::new(BatchMeta {
            batch_position,
            row_offset,
            doc_lengths,
            rows,
        });
        let cur = self.snapshot.load();
        let mut batches: Vec<Arc<BatchMeta>> = Vec::with_capacity(cur.batches.len() + 1);
        batches.extend(cur.batches.iter().cloned());
        batches.push(new_meta);
        self.snapshot.store(Arc::new(Snapshot {
            visible_count: cur.visible_count + 1,
            batches: Arc::from(batches.into_boxed_slice()),
            cumulative_doc_count: cur.cumulative_doc_count + rows as u64,
            cumulative_total_tokens: cur.cumulative_total_tokens + total_tokens,
        }));
    }

    fn memory_size(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();
        for entry in self.terms.iter() {
            let term: &Arc<str> = entry.key();
            total += std::mem::size_of::<Arc<str>>() + term.len() + 32;
            total += entry.value().load().memory_size();
        }
        total += self
            .snapshot
            .load()
            .batches
            .iter()
            .map(|b| b.memory_size())
            .sum::<usize>();
        total
    }
}

/// Atomically-published index state: the immutable partitions plus the live
/// mutable tail. The pair lives behind one `ArcSwap` so a freeze (which both
/// appends a partition and resets the tail) is observed atomically.
struct IndexState {
    partitions: Arc<[Arc<Partition>]>,
    tail: Arc<TailIndex>,
}

impl IndexState {
    fn empty() -> Arc<Self> {
        Arc::new(Self {
            partitions: Arc::from(Vec::<Arc<Partition>>::new().into_boxed_slice()),
            tail: TailIndex::new(),
        })
    }
}

/// In-memory full-text search index. See module docs for the concurrency
/// model and visibility contract.
pub struct FtsMemIndex {
    field_id: i32,
    column_name: String,
    params: InvertedIndexParams,

    tokenizer_pool: Arc<TokenizerPool>,
    /// Writer-only tokenizer slot. Held under a Mutex purely so `insert`
    /// can take `&self`. Single-writer assumption means this is uncontested.
    writer_tokenizer: Mutex<Box<dyn LanceTokenizer>>,

    /// `{partitions, tail}` published atomically. The tail mutates in place
    /// between freezes; the whole state is swapped on freeze.
    state: ArcSwap<IndexState>,

    /// The tail freezes into a partition once it reaches this many docs.
    freeze_threshold_rows: usize,
}

impl std::fmt::Debug for FtsMemIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = self.state.load();
        f.debug_struct("FtsMemIndex")
            .field("field_id", &self.field_id)
            .field("column_name", &self.column_name)
            .field("doc_count", &self.doc_count())
            .field("partitions", &st.partitions.len())
            .field("params", &self.params)
            .finish()
    }
}

impl FtsMemIndex {
    /// Default reader-tokenizer pool capacity. Small but enough to absorb
    /// short bursts of concurrent search calls without thrashing.
    const DEFAULT_TOKENIZER_POOL_CAP: usize = 8;

    /// Default tail freeze threshold (docs). Analogous to Lucene's
    /// `ramBufferSizeMB`: large enough that the per-query tail scan stays
    /// cheap, small enough to bound it independent of corpus size.
    const DEFAULT_FREEZE_THRESHOLD_ROWS: usize = 50_000;

    /// Hard cap on live partitions: when a freeze would exceed it, all
    /// partitions are merged into one. With the default freeze threshold this
    /// only triggers past ~1.6M docs.
    const MAX_PARTITIONS: usize = 32;

    /// Create a new FTS index for the given field with default parameters.
    pub fn new(field_id: i32, column_name: String) -> Self {
        Self::with_params(field_id, column_name, InvertedIndexParams::default())
    }

    /// Create a new FTS index with custom tokenizer parameters.
    pub fn with_params(field_id: i32, column_name: String, params: InvertedIndexParams) -> Self {
        let pool = TokenizerPool::new(&params, Self::DEFAULT_TOKENIZER_POOL_CAP)
            .expect("Failed to build tokenizer");
        let writer_tokenizer = pool.template.box_clone();
        Self {
            field_id,
            column_name,
            params,
            tokenizer_pool: Arc::new(pool),
            writer_tokenizer: Mutex::new(writer_tokenizer),
            state: ArcSwap::from(IndexState::empty()),
            freeze_threshold_rows: Self::DEFAULT_FREEZE_THRESHOLD_ROWS,
        }
    }

    /// Override the tail freeze threshold. Used by tests to exercise the
    /// multi-partition path with small inputs.
    #[cfg(test)]
    fn with_freeze_threshold_rows(mut self, rows: usize) -> Self {
        self.freeze_threshold_rows = rows.max(1);
        self
    }

    pub fn field_id(&self) -> i32 {
        self.field_id
    }

    pub fn column_name(&self) -> &str {
        &self.column_name
    }

    pub fn params(&self) -> &InvertedIndexParams {
        &self.params
    }

    /// Number of visible documents across all partitions and the tail.
    pub fn doc_count(&self) -> usize {
        let st = self.state.load();
        st.partitions.iter().map(|p| p.doc_count()).sum::<usize>() + st.tail.doc_count() as usize
    }

    /// Whether there are any visible documents.
    pub fn is_empty(&self) -> bool {
        let st = self.state.load();
        st.partitions.is_empty() && st.tail.visible_count() == 0
    }

    /// Total number of visible (term, doc) postings.
    pub fn entry_count(&self) -> usize {
        let st = self.state.load_full();
        let part: usize = st.partitions.iter().map(|p| p.entry_count()).sum();
        let visible = st.tail.visible_count();
        let tail: usize = st
            .tail
            .terms
            .iter()
            .map(|e| {
                e.value()
                    .load()
                    .chunks
                    .iter()
                    .filter(|c| c.batch_position < visible)
                    .map(|c| c.doc_count())
                    .sum::<usize>()
            })
            .sum();
        part + tail
    }

    /// Estimated bytes of heap memory held by this index.
    pub fn memory_usage(&self) -> usize {
        let st = self.state.load_full();
        let mut total = std::mem::size_of::<Self>();
        total += st.partitions.iter().map(|p| p.memory_size()).sum::<usize>();
        total += st.tail.memory_size();
        total
    }

    // ------------------------------------------------------------------
    // Insert
    // ------------------------------------------------------------------

    /// Insert a batch into the mutable tail.
    pub fn insert(&self, batch: &RecordBatch, row_offset: u64) -> Result<()> {
        self.insert_batch(batch, row_offset)
    }

    /// Insert a batch. The explicit `batch_position` is accepted for API
    /// compatibility but not used for placement: the tail assigns its own
    /// dense, 0-based positions (required by the visibility filter).
    pub fn insert_with_batch_position(
        &self,
        batch: &RecordBatch,
        row_offset: u64,
        _batch_position: usize,
    ) -> Result<()> {
        self.insert_batch(batch, row_offset)
    }

    fn insert_batch(&self, batch: &RecordBatch, row_offset: u64) -> Result<()> {
        let st = self.state.load_full();
        let batch_position = st.tail.next_position();

        let Some(col_idx) = batch
            .schema()
            .column_with_name(&self.column_name)
            .map(|(idx, _)| idx)
        else {
            // Column missing: nothing to index, but publish an empty batch so
            // the tail's visibility counters keep up with the writer.
            st.tail.append_batch(
                batch_position,
                row_offset,
                batch.num_rows() as u32,
                vec![0; batch.num_rows()],
                0,
                HashMap::new(),
            );
            return Ok(());
        };

        let column = batch.column(col_idx);
        let texts = extract_texts(column.as_ref())?;
        debug_assert_eq!(texts.len(), batch.num_rows());

        let mut tok_guard = self
            .writer_tokenizer
            .lock()
            .expect("writer tokenizer poisoned — single-writer invariant violated");
        let tokenizer: &mut dyn LanceTokenizer = tok_guard.as_mut();

        // Per-term builders (frequency + per-doc positions, indexed by
        // local doc index in this batch).
        let mut term_builders: HashMap<Arc<str>, BatchTermBuilder> = HashMap::new();
        let mut doc_lengths: Vec<u32> = Vec::with_capacity(batch.num_rows());
        let mut total_tokens: u64 = 0;

        for (local_doc_idx, text_opt) in texts.iter().enumerate() {
            // Track each doc's position even for null/missing rows so the
            // dense `doc_lengths` array stays aligned with `row_offset + i`.
            let mut doc_token_count: u32 = 0;
            // term -> (frequency, positions). Keyed by `String` so we only
            // pay the `Arc<str>` allocation once per unique term per doc,
            // when transferring to `term_builders` below.
            let mut per_doc: HashMap<String, (u32, Vec<u32>)> = HashMap::new();

            if let Some(text) = text_opt {
                let mut stream = tokenizer.token_stream_for_doc(text);
                let mut position: u32 = 0;
                while let Some(tok) = stream.next() {
                    let entry = per_doc
                        .entry(tok.text.clone())
                        .or_insert_with(|| (0, Vec::new()));
                    entry.0 += 1;
                    entry.1.push(position);
                    position += 1;
                    doc_token_count += 1;
                }
            }

            doc_lengths.push(doc_token_count);
            total_tokens += doc_token_count as u64;

            let row_position = row_offset + local_doc_idx as u64;
            for (term, (freq, positions)) in per_doc {
                // Reuse an interned `Arc<str>` if we've already seen the
                // term in this batch. The first occurrence pays the
                // allocation; subsequent occurrences clone the Arc.
                let term_arc: Arc<str> =
                    if let Some((existing, _)) = term_builders.get_key_value(term.as_str()) {
                        Arc::clone(existing)
                    } else {
                        Arc::<str>::from(term.as_str())
                    };
                let builder = term_builders
                    .entry(term_arc)
                    .or_insert_with(BatchTermBuilder::new);
                builder.push_doc(row_position, freq, positions);
            }
        }

        // Drop the tokenizer guard before publishing so we don't hold it
        // across the snapshot install.
        drop(tok_guard);

        st.tail.append_batch(
            batch_position,
            row_offset,
            batch.num_rows() as u32,
            doc_lengths,
            total_tokens,
            term_builders,
        );

        if st.tail.doc_count() >= self.freeze_threshold_rows as u64 {
            self.freeze(&st);
        }
        Ok(())
    }

    /// Freeze the current tail into a new immutable partition and publish a
    /// fresh empty tail. Only the writer calls this; readers snapshotting the
    /// old `IndexState` keep a consistent view across the freeze.
    fn freeze(&self, st: &IndexState) {
        let Some(partition) = Partition::from_tail(&st.tail) else {
            return;
        };
        let mut partitions: Vec<Arc<Partition>> = st.partitions.iter().cloned().collect();
        partitions.push(Arc::new(partition));
        if partitions.len() > Self::MAX_PARTITIONS {
            partitions = vec![Arc::new(Partition::merge(&partitions))];
        }
        self.state.store(Arc::new(IndexState {
            partitions: Arc::from(partitions.into_boxed_slice()),
            tail: TailIndex::new(),
        }));
    }

    // ------------------------------------------------------------------
    // Read path
    // ------------------------------------------------------------------

    /// Search for documents containing a term.
    ///
    /// The term is tokenized using the configured tokenizer. Returns all
    /// matching documents with BM25 scores. Result order is unspecified;
    /// use `search_with_options` for sorted/limited output.
    pub fn search(&self, term: &str) -> Vec<FtsEntry> {
        let st = self.state.load_full();
        let tokens = self.tokenize_for_search(term);
        self.search_match(&st, &tokens)
    }

    /// Search for documents containing an exact phrase, optionally allowing
    /// `slop` intervening tokens between consecutive query tokens.
    pub fn search_phrase(&self, phrase: &str, slop: u32) -> Vec<FtsEntry> {
        let st = self.state.load_full();
        let tokens = self.tokenize_for_search(phrase);
        self.search_phrase_tokens(&st, &tokens, slop)
    }

    /// Expand a term to fuzzy matches within the specified edit distance.
    /// Returns `(matched_term, distance)` pairs sorted by distance, capped
    /// at `max_expansions`.
    pub fn expand_fuzzy(
        &self,
        term: &str,
        max_distance: u32,
        max_expansions: usize,
    ) -> Vec<(String, u32)> {
        let st = self.state.load_full();
        self.expand_fuzzy_term(&st, term, max_distance, max_expansions)
    }

    /// Search for documents using fuzzy matching on each query token.
    pub fn search_fuzzy(
        &self,
        query: &str,
        fuzziness: Option<u32>,
        max_expansions: usize,
    ) -> Vec<FtsEntry> {
        let st = self.state.load_full();
        let tokens = self.tokenize_for_search(query);
        self.search_fuzzy_tokens(&st, &tokens, fuzziness, max_expansions)
    }

    /// BM25 OR-search over the query tokens: WAND each immutable partition,
    /// scan the tail, all scored with one corpus-wide [`MemBM25Scorer`], and
    /// the results merged. Each doc lives in exactly one partition or the
    /// tail, so the merge is a plain concatenation.
    fn search_match(&self, st: &IndexState, tokens: &[String]) -> Vec<FtsEntry> {
        if tokens.is_empty() {
            return Vec::new();
        }
        // Snapshot the tail once so the scorer's stats and the scanned tail
        // postings are from the same visibility point.
        let tail_snap = st.tail.snapshot();
        let scorer = build_scorer(st, &tail_snap, tokens);
        if scorer.num_docs() == 0 {
            return Vec::new();
        }
        let mut results = Vec::new();
        for p in st.partitions.iter() {
            match p.search_match(tokens, Operator::Or, &scorer) {
                Ok(hits) => results.extend(hits),
                Err(e) => log::warn!("FTS partition match search failed: {e}"),
            }
        }
        if tail_snap.visible_count > 0 {
            results.extend(score_terms(&tail_snap, &st.tail.terms, tokens, &scorer));
        }
        results
    }

    fn search_phrase_tokens(&self, st: &IndexState, tokens: &[String], slop: u32) -> Vec<FtsEntry> {
        if tokens.is_empty() {
            return Vec::new();
        }
        if tokens.len() == 1 {
            // A single-token phrase reduces to a regular term search.
            return self.search_match(st, tokens);
        }
        let tail_snap = st.tail.snapshot();
        let scorer = build_scorer(st, &tail_snap, tokens);
        if scorer.num_docs() == 0 {
            return Vec::new();
        }
        let mut results = Vec::new();
        for p in st.partitions.iter() {
            match p.search_phrase(tokens, slop, &scorer) {
                Ok(hits) => results.extend(hits),
                Err(e) => log::warn!("FTS partition phrase search failed: {e}"),
            }
        }
        if tail_snap.visible_count > 0 {
            results.extend(phrase_search_tail(
                &tail_snap,
                &st.tail.terms,
                tokens,
                slop,
                &scorer,
            ));
        }
        results
    }

    fn search_fuzzy_tokens(
        &self,
        st: &IndexState,
        tokens: &[String],
        fuzziness: Option<u32>,
        max_expansions: usize,
    ) -> Vec<FtsEntry> {
        if tokens.is_empty() {
            return Vec::new();
        }
        let mut expanded: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for tok in tokens {
            let max_dist = fuzziness.unwrap_or_else(|| auto_fuzziness(tok));
            for (matched, _) in self.expand_fuzzy_term(st, tok, max_dist, max_expansions) {
                if seen.insert(matched.clone()) {
                    expanded.push(matched);
                }
            }
        }
        if expanded.is_empty() {
            return Vec::new();
        }
        self.search_match(st, &expanded)
    }

    /// Expand `term` against the term dictionaries of every partition and the
    /// visible tail.
    fn expand_fuzzy_term(
        &self,
        st: &IndexState,
        term: &str,
        max_distance: u32,
        max_expansions: usize,
    ) -> Vec<(String, u32)> {
        let tail_snap = st.tail.snapshot();
        if max_distance == 0 {
            let in_tail = st
                .tail
                .terms
                .get(term)
                .map(|e| has_visible_chunk(&e.value().load(), tail_snap.visible_count))
                .unwrap_or(false);
            let in_partition = st.partitions.iter().any(|p| p.contains_token(term));
            return if in_tail || in_partition {
                vec![(term.to_string(), 0)]
            } else {
                Vec::new()
            };
        }
        let mut matches: Vec<(String, u32)> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for entry in st.tail.terms.iter() {
            if !has_visible_chunk(&entry.value().load(), tail_snap.visible_count) {
                continue;
            }
            let key: &Arc<str> = entry.key();
            let dist = levenshtein_distance(term, key);
            if dist <= max_distance && seen.insert(key.to_string()) {
                matches.push((key.to_string(), dist));
            }
        }
        for p in st.partitions.iter() {
            for key in p.tokens() {
                let dist = levenshtein_distance(term, key);
                if dist <= max_distance && seen.insert(key.to_string()) {
                    matches.push((key.to_string(), dist));
                }
            }
        }
        matches.sort_by_key(|(_, d)| *d);
        matches.truncate(max_expansions);
        matches
    }

    /// Execute a query expression and return matching documents with scores.
    ///
    /// Snapshots the index state once at entry so the entire compound
    /// query — including every leaf invoked recursively from `Boolean` /
    /// `Boost` — sees the same `{partitions, tail}` view. This preserves the
    /// per-batch monotonic visibility contract for compound queries.
    pub fn search_query(&self, query: &FtsQueryExpr) -> Vec<FtsEntry> {
        let st = self.state.load_full();
        self.search_query_with_state(query, &st)
    }

    fn search_query_with_state(&self, query: &FtsQueryExpr, st: &IndexState) -> Vec<FtsEntry> {
        match query {
            FtsQueryExpr::Match { query, boost } => {
                let tokens = self.tokenize_for_search(query);
                let mut results = self.search_match(st, &tokens);
                apply_boost(&mut results, *boost);
                results
            }
            FtsQueryExpr::Phrase { query, slop, boost } => {
                let tokens = self.tokenize_for_search(query);
                let mut results = self.search_phrase_tokens(st, &tokens, *slop);
                apply_boost(&mut results, *boost);
                results
            }
            FtsQueryExpr::Fuzzy {
                query,
                fuzziness,
                max_expansions,
                boost,
            } => {
                let tokens = self.tokenize_for_search(query);
                let mut results =
                    self.search_fuzzy_tokens(st, &tokens, *fuzziness, *max_expansions);
                apply_boost(&mut results, *boost);
                results
            }
            FtsQueryExpr::Boolean {
                must,
                should,
                must_not,
            } => self.search_boolean(must, should, must_not, st),
            FtsQueryExpr::Boost {
                positive,
                negative,
                negative_boost,
            } => self.search_boost(positive, negative.as_deref(), *negative_boost, st),
        }
    }

    /// Execute a query with options (sort + WAND prune + limit).
    pub fn search_with_options(
        &self,
        query: &FtsQueryExpr,
        options: SearchOptions,
    ) -> Vec<FtsEntry> {
        let st = self.state.load_full();
        let mut results = self.search_query_with_state(query, &st);
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if options.wand_factor < 1.0 {
            if let Some(limit) = options.limit {
                if results.len() > limit {
                    let top_k_score = results[limit - 1].score;
                    let threshold = top_k_score * options.wand_factor;
                    results.retain(|e| e.score >= threshold);
                }
            } else if let Some(max_entry) = results.first() {
                let threshold = max_entry.score * options.wand_factor;
                results.retain(|e| e.score >= threshold);
            }
        }
        if let Some(limit) = options.limit {
            results.truncate(limit);
        }
        results
    }

    fn search_boost(
        &self,
        positive: &FtsQueryExpr,
        negative: Option<&FtsQueryExpr>,
        negative_boost: f32,
        st: &IndexState,
    ) -> Vec<FtsEntry> {
        let mut results = self.search_query_with_state(positive, st);
        let Some(neg) = negative else {
            return results;
        };
        let negative_results = self.search_query_with_state(neg, st);
        let negative_set: HashSet<RowPosition> = negative_results
            .into_iter()
            .map(|e| e.row_position)
            .collect();
        for entry in &mut results {
            if negative_set.contains(&entry.row_position) {
                entry.score *= negative_boost;
            }
        }
        results
    }

    fn search_boolean(
        &self,
        must: &[FtsQueryExpr],
        should: &[FtsQueryExpr],
        must_not: &[FtsQueryExpr],
        st: &IndexState,
    ) -> Vec<FtsEntry> {
        let excluded: HashSet<RowPosition> = must_not
            .iter()
            .flat_map(|q| self.search_query_with_state(q, st))
            .map(|e| e.row_position)
            .collect();

        let mut result_map: HashMap<RowPosition, f32> = if must.is_empty() {
            let mut map: HashMap<RowPosition, f32> = HashMap::new();
            for q in should {
                for entry in self.search_query_with_state(q, st) {
                    *map.entry(entry.row_position).or_default() += entry.score;
                }
            }
            map
        } else {
            let first_results = self.search_query_with_state(&must[0], st);
            let mut map: HashMap<RowPosition, f32> = first_results
                .into_iter()
                .map(|e| (e.row_position, e.score))
                .collect();
            for q in must.iter().skip(1) {
                let results = self.search_query_with_state(q, st);
                let result_set: HashMap<RowPosition, f32> = results
                    .into_iter()
                    .map(|e| (e.row_position, e.score))
                    .collect();
                map = map
                    .into_iter()
                    .filter_map(|(pos, score)| result_set.get(&pos).map(|s| (pos, score + s)))
                    .collect();
            }
            for q in should {
                for entry in self.search_query_with_state(q, st) {
                    if let Some(score) = map.get_mut(&entry.row_position) {
                        *score += entry.score;
                    }
                }
            }
            map
        };

        for pos in &excluded {
            result_map.remove(pos);
        }

        result_map
            .into_iter()
            .map(|(row_position, score)| FtsEntry {
                row_position,
                score,
            })
            .collect()
    }

    fn tokenize_for_search(&self, text: &str) -> Vec<String> {
        let mut tok = PooledTokenizer::new(&self.tokenizer_pool);
        let mut stream = tok.get_mut().token_stream_for_search(text);
        let mut out = Vec::new();
        while let Some(t) = stream.next() {
            out.push(t.text.clone());
        }
        out
    }

    // ------------------------------------------------------------------
    // Flush to Lance inverted index format
    // ------------------------------------------------------------------

    /// Export the in-memory FTS index to an `InnerBuilder` ready to be
    /// written to disk.
    ///
    /// `total_rows` is the total number of rows in the MemTable being
    /// flushed; row positions are reversed (`reversed = total_rows - pos -
    /// 1`) to match the LSM-friendly newest-first flush order used by the
    /// rest of MemWAL.
    pub fn to_index_builder_reversed(
        &self,
        partition_id: u64,
        total_rows: usize,
    ) -> Result<lance_index::scalar::inverted::builder::InnerBuilder> {
        use lance_index::scalar::inverted::PostingListBuilder;
        use lance_index::scalar::inverted::builder::{InnerBuilder, PositionRecorder};

        let st = self.state.load_full();
        let with_position = self.params.has_positions();
        let total_rows_u64 = total_rows as u64;

        // Step 1: collect (original_pos, num_tokens) for every doc across all
        // immutable partitions and the visible tail.
        let mut all_docs: Vec<(u64, u32)> = Vec::new();
        for p in st.partitions.iter() {
            for (row_pos, num_tokens) in p.docs.iter() {
                all_docs.push((*row_pos, *num_tokens));
            }
        }
        let tail_snap = st.tail.snapshot();
        for batch in tail_snap.batches.iter().take(tail_snap.visible_count) {
            for i in 0..batch.rows as usize {
                all_docs.push((batch.row_offset + i as u64, batch.doc_lengths[i]));
            }
        }
        if all_docs.is_empty() {
            return Ok(InnerBuilder::new(
                partition_id,
                with_position,
                Default::default(),
            ));
        }

        // Step 2: assign doc_ids in ascending reversed-position order, so the
        // on-disk layout matches MemWAL's newest-first flush convention.
        let mut entries: Vec<(u64, u64, u32)> = Vec::with_capacity(all_docs.len());
        for (original, num_tokens) in &all_docs {
            if *original >= total_rows_u64 {
                return Err(Error::io(format!(
                    "FTS flush: row position {} >= total_rows {}",
                    original, total_rows
                )));
            }
            entries.push((total_rows_u64 - original - 1, *original, *num_tokens));
        }
        entries.sort_by_key(|(rev, _, _)| *rev);
        let mut docs = DocSet::default();
        let mut original_to_doc_id: HashMap<u64, u32> = HashMap::with_capacity(entries.len());
        for (rev, original, num_tokens) in &entries {
            let doc_id = docs.append(*rev, *num_tokens);
            original_to_doc_id.insert(*original, doc_id);
        }

        // Step 3: merge per-term postings across every partition and the tail.
        let mut term_postings: HashMap<String, Vec<(u32, u32, Option<Vec<u32>>)>> = HashMap::new();
        for p in st.partitions.iter() {
            for (token, posting) in p.tokens_with_postings() {
                let plain = posting_as_plain(posting);
                let bucket = term_postings.entry(token.to_string()).or_default();
                for i in 0..plain.len() {
                    let row_pos = p.docs.row_id(plain.row_ids[i] as u32);
                    let Some(&doc_id) = original_to_doc_id.get(&row_pos) else {
                        continue;
                    };
                    let pos = if with_position {
                        Some(read_positions(plain, i))
                    } else {
                        None
                    };
                    bucket.push((doc_id, plain.frequencies[i] as u32, pos));
                }
            }
        }
        for entry in st.tail.terms.iter() {
            let token: &Arc<str> = entry.key();
            let slice = entry.value().load();
            let bucket = term_postings.entry(token.to_string()).or_default();
            for chunk in &slice.chunks {
                if chunk.batch_position >= tail_snap.visible_count {
                    continue;
                }
                for (i, row_position) in chunk.row_positions.iter().enumerate() {
                    let Some(&doc_id) = original_to_doc_id.get(row_position) else {
                        continue;
                    };
                    let pos = if with_position {
                        Some(
                            chunk
                                .positions
                                .as_ref()
                                .map(|p| p.doc_positions(i).to_vec())
                                .unwrap_or_default(),
                        )
                    } else {
                        None
                    };
                    bucket.push((doc_id, chunk.frequencies[i], pos));
                }
            }
        }

        // Step 4: emit posting lists; tokens in sorted order for determinism.
        let mut sorted_tokens: Vec<String> = term_postings.keys().cloned().collect();
        sorted_tokens.sort_unstable();
        let mut tokens = TokenSet::default();
        let mut posting_lists: Vec<PostingListBuilder> = Vec::with_capacity(sorted_tokens.len());
        for token in sorted_tokens {
            let mut docs_for_term = term_postings.remove(&token).unwrap_or_default();
            if docs_for_term.is_empty() {
                continue;
            }
            docs_for_term.sort_by_key(|(doc_id, _, _)| *doc_id);
            let token_id = tokens.add(token) as usize;
            debug_assert_eq!(token_id, posting_lists.len());
            posting_lists.push(PostingListBuilder::new(with_position));
            let plb = &mut posting_lists[token_id];
            for (doc_id, freq, pos) in docs_for_term {
                let recorder = if with_position {
                    PositionRecorder::Position(pos.unwrap_or_default().into())
                } else {
                    PositionRecorder::Count(freq)
                };
                plb.add(doc_id, recorder);
            }
        }

        let mut builder = InnerBuilder::new(partition_id, with_position, Default::default());
        builder.set_tokens(tokens);
        builder.set_docs(docs);
        builder.set_posting_lists(posting_lists);
        Ok(builder)
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

impl TermSlice {
    fn empty_arc_swap() -> ArcSwap<Self> {
        ArcSwap::from(Self::empty())
    }
}

/// Helper used during insert to accumulate a single batch's contribution to
/// one term.
struct BatchTermBuilder {
    row_positions: Vec<u64>,
    frequencies: Vec<u32>,
    positions: Vec<Vec<u32>>,
}

impl BatchTermBuilder {
    fn new() -> Self {
        Self {
            row_positions: Vec::new(),
            frequencies: Vec::new(),
            positions: Vec::new(),
        }
    }

    fn push_doc(&mut self, row_position: u64, frequency: u32, positions: Vec<u32>) {
        self.row_positions.push(row_position);
        self.frequencies.push(frequency);
        self.positions.push(positions);
    }

    /// Always materializes the per-doc positions: in-memory phrase queries
    /// rely on them regardless of `params.has_positions()`. The flush path
    /// consults `params.has_positions()` and emits a `Count` recorder
    /// instead of `Position` when positions should not be persisted.
    fn build(self, batch_position: usize) -> Arc<TermChunk> {
        let mut p = Positions::empty();
        for doc in &self.positions {
            p.push_doc(doc);
        }
        Arc::new(TermChunk {
            batch_position,
            row_positions: self.row_positions,
            frequencies: self.frequencies,
            positions: Some(p),
        })
    }
}

/// Borrowed text for a row, or `None` for null/missing.
type TextOpt<'a> = Option<&'a str>;

fn extract_texts(column: &dyn Array) -> Result<Vec<TextOpt<'_>>> {
    match column.data_type() {
        DataType::Utf8 => {
            let array = column
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Utf8 array");
            Ok((0..array.len())
                .map(|i| (!array.is_null(i)).then(|| array.value(i)))
                .collect())
        }
        DataType::LargeUtf8 => {
            let array = column
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("LargeUtf8 array");
            Ok((0..array.len())
                .map(|i| (!array.is_null(i)).then(|| array.value(i)))
                .collect())
        }
        DataType::Utf8View => {
            let array = column
                .as_any()
                .downcast_ref::<StringViewArray>()
                .expect("Utf8View array");
            Ok((0..array.len())
                .map(|i| (!array.is_null(i)).then(|| array.value(i)))
                .collect())
        }
        other => Err(Error::invalid_input(format!(
            "FTS index only supports Utf8, LargeUtf8, and Utf8View columns; got {other:?}"
        ))),
    }
}

fn has_visible_chunk(slice: &TermSlice, visible_count: usize) -> bool {
    slice
        .chunks
        .iter()
        .any(|c| c.batch_position < visible_count)
}

fn lookup_dl(snap: &Snapshot, row_position: u64) -> Option<u32> {
    snap.batches[..snap.visible_count]
        .iter()
        .find_map(|b| b.dl(row_position))
}

fn find_doc_in_chunks(
    chunks: &[Arc<TermChunk>],
    row_position: u64,
) -> Option<(&Arc<TermChunk>, usize)> {
    for chunk in chunks {
        if let Ok(idx) = chunk.row_positions.binary_search(&row_position) {
            return Some((chunk, idx));
        }
    }
    None
}

/// Build a corpus-wide BM25 scorer over every partition and the tail, for the
/// (deduplicated) set of query tokens. A single scorer makes partition-WAND
/// scores and tail-scan scores directly comparable. `tail_snap` must be the
/// *same* tail snapshot the caller scans, so the scorer's stats and the
/// scanned postings stay mutually consistent.
fn build_scorer(st: &IndexState, tail_snap: &Snapshot, tokens: &[String]) -> MemBM25Scorer {
    let mut total_tokens = tail_snap.cumulative_total_tokens;
    let mut num_docs = tail_snap.cumulative_doc_count as usize;
    for p in st.partitions.iter() {
        total_tokens += p.total_tokens();
        num_docs += p.doc_count();
    }
    let mut token_docs: HashMap<String, usize> = HashMap::new();
    for token in tokens {
        if token_docs.contains_key(token) {
            continue;
        }
        let mut df = tail_token_df(&st.tail.terms, token, tail_snap.visible_count);
        for p in st.partitions.iter() {
            df += p.token_df(token);
        }
        token_docs.insert(token.clone(), df);
    }
    MemBM25Scorer::new(total_tokens, num_docs, token_docs)
}

/// Number of visible tail docs containing `token`.
fn tail_token_df(
    terms: &SkipMap<Arc<str>, ArcSwap<TermSlice>>,
    token: &str,
    visible_count: usize,
) -> usize {
    match terms.get(token) {
        Some(e) => e
            .value()
            .load()
            .chunks
            .iter()
            .filter(|c| c.batch_position < visible_count)
            .map(|c| c.doc_count())
            .sum(),
        None => 0,
    }
}

/// OR-score `tokens` against the visible tail, summing each token's BM25
/// contribution per document. Uses the shared corpus-wide `scorer`.
fn score_terms(
    snap: &Snapshot,
    terms: &SkipMap<Arc<str>, ArcSwap<TermSlice>>,
    tokens: &[String],
    scorer: &MemBM25Scorer,
) -> Vec<FtsEntry> {
    let mut doc_scores: HashMap<RowPosition, f32> = HashMap::new();
    for token in tokens {
        let Some(entry) = terms.get(token.as_str()) else {
            continue;
        };
        let qw = scorer.query_weight(token);
        if qw == 0.0 {
            continue;
        }
        let slice = entry.value().load_full();
        for chunk in &slice.chunks {
            if chunk.batch_position >= snap.visible_count {
                continue;
            }
            let Some(meta) = snap.batch_for(chunk.batch_position) else {
                continue;
            };
            for (i, &row_position) in chunk.row_positions.iter().enumerate() {
                let dl = meta.dl(row_position).unwrap_or(1);
                let score = qw * scorer.doc_weight(chunk.frequencies[i], dl);
                *doc_scores.entry(row_position).or_default() += score;
            }
        }
    }
    doc_scores
        .into_iter()
        .map(|(row_position, score)| FtsEntry {
            row_position,
            score,
        })
        .collect()
}

/// Phrase-search the visible tail. Callers shortcut the single-token case, so
/// `tokens.len() >= 2` here. Scored with the shared corpus-wide `scorer`.
fn phrase_search_tail(
    snap: &Snapshot,
    terms: &SkipMap<Arc<str>, ArcSwap<TermSlice>>,
    tokens: &[String],
    slop: u32,
    scorer: &MemBM25Scorer,
) -> Vec<FtsEntry> {
    // Gather visible chunks per query token; any missing token => no match.
    let mut per_token_chunks: Vec<Vec<Arc<TermChunk>>> = Vec::with_capacity(tokens.len());
    for tok in tokens {
        match terms.get(tok.as_str()) {
            Some(entry) => {
                let slice = entry.value().load_full();
                let visible: Vec<Arc<TermChunk>> = slice
                    .chunks
                    .iter()
                    .filter(|c| c.batch_position < snap.visible_count)
                    .cloned()
                    .collect();
                if visible.is_empty() {
                    return Vec::new();
                }
                per_token_chunks.push(visible);
            }
            None => return Vec::new(),
        }
    }
    // Drive the intersection from the rarest token to bound work.
    let smallest_idx = (0..per_token_chunks.len())
        .min_by_key(|&i| {
            per_token_chunks[i]
                .iter()
                .map(|c| c.doc_count())
                .sum::<usize>()
        })
        .unwrap();

    let mut results = Vec::new();
    for chunk in &per_token_chunks[smallest_idx] {
        for (doc_idx, &row_position) in chunk.row_positions.iter().enumerate() {
            let Some(pos) = chunk
                .positions
                .as_ref()
                .map(|p| p.doc_positions(doc_idx).to_vec())
            else {
                continue;
            };
            let mut all_positions: Vec<Vec<u32>> = vec![Vec::new(); tokens.len()];
            all_positions[smallest_idx] = pos;
            let mut frequencies = vec![0u32; tokens.len()];
            frequencies[smallest_idx] = chunk.frequencies[doc_idx];
            let mut all_present = true;
            for (ti, chunks) in per_token_chunks.iter().enumerate() {
                if ti == smallest_idx {
                    continue;
                }
                match find_doc_in_chunks(chunks, row_position) {
                    Some((c, other_idx)) => {
                        frequencies[ti] = c.frequencies[other_idx];
                        all_positions[ti] = c
                            .positions
                            .as_ref()
                            .map(|p| p.doc_positions(other_idx).to_vec())
                            .unwrap_or_default();
                    }
                    None => {
                        all_present = false;
                        break;
                    }
                }
            }
            if !all_present || !phrase_matches(&all_positions, slop) {
                continue;
            }
            let dl = lookup_dl(snap, row_position).unwrap_or(1);
            let score: f32 = tokens
                .iter()
                .enumerate()
                .map(|(ti, tok)| scorer.query_weight(tok) * scorer.doc_weight(frequencies[ti], dl))
                .sum();
            results.push(FtsEntry {
                row_position,
                score,
            });
        }
    }
    results
}

fn phrase_matches(positions: &[Vec<u32>], slop: u32) -> bool {
    if positions.is_empty() {
        return false;
    }
    for &first_pos in &positions[0] {
        if phrase_from_position(positions, first_pos, slop) {
            return true;
        }
    }
    false
}

fn phrase_from_position(positions: &[Vec<u32>], first_pos: u32, slop: u32) -> bool {
    let mut expected = first_pos;
    for token_positions in positions.iter().skip(1) {
        let min = expected.saturating_add(1);
        let max = expected.saturating_add(1 + slop);
        match token_positions
            .iter()
            .filter(|&&p| p >= min && p <= max)
            .min()
        {
            Some(&p) => expected = p,
            None => return false,
        }
    }
    true
}

fn apply_boost(results: &mut [FtsEntry], boost: f32) {
    if boost == 1.0 {
        return;
    }
    for r in results.iter_mut() {
        r.score *= boost;
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a Full-Text Search index.
#[derive(Debug, Clone)]
pub struct FtsIndexConfig {
    pub name: String,
    pub field_id: i32,
    pub column: String,
    pub params: InvertedIndexParams,
}

impl FtsIndexConfig {
    pub fn new(name: String, field_id: i32, column: String) -> Self {
        Self {
            name,
            field_id,
            column,
            params: InvertedIndexParams::default(),
        }
    }

    pub fn with_params(
        name: String,
        field_id: i32,
        column: String,
        params: InvertedIndexParams,
    ) -> Self {
        Self {
            name,
            field_id,
            column,
            params,
        }
    }
}

// ============================================================================
// Immutable partition
// ============================================================================

/// An immutable, frozen FTS partition: a slice of the MemTable's inserts held
/// as on-disk-shaped posting lists and queried with block-max WAND in
/// ≈O(matches). Built by freezing the tail; flushed 1:1 into the Lance FTS
/// on-disk format. See `DESIGN.md` in the redesign analysis directory.
struct Partition {
    /// token text -> local token id (dense, 0-based; indexes `postings`).
    token_ids: HashMap<Arc<str>, u32>,
    /// posting list per local token id. In-memory partitions are always
    /// `PostingList::Plain` and always carry positions.
    postings: Vec<PostingList>,
    /// local doc id -> (MemTable row position, token count).
    docs: DocSet,
}

impl Partition {
    fn doc_count(&self) -> usize {
        self.docs.len()
    }

    fn total_tokens(&self) -> u64 {
        self.docs.total_tokens_num()
    }

    fn entry_count(&self) -> usize {
        self.postings
            .iter()
            .map(|p| posting_as_plain(p).len())
            .sum()
    }

    fn contains_token(&self, token: &str) -> bool {
        self.token_ids.contains_key(token)
    }

    fn tokens(&self) -> impl Iterator<Item = &Arc<str>> {
        self.token_ids.keys()
    }

    fn tokens_with_postings(&self) -> impl Iterator<Item = (&Arc<str>, &PostingList)> {
        self.token_ids
            .iter()
            .map(move |(t, &id)| (t, &self.postings[id as usize]))
    }

    /// Number of docs in this partition containing `token`.
    fn token_df(&self, token: &str) -> usize {
        self.token_ids
            .get(token)
            .map(|&id| posting_as_plain(&self.postings[id as usize]).len())
            .unwrap_or(0)
    }

    fn memory_size(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();
        for t in self.token_ids.keys() {
            total += std::mem::size_of::<Arc<str>>() + t.len() + std::mem::size_of::<u32>() + 16;
        }
        for p in &self.postings {
            let pl = posting_as_plain(p);
            total += pl.row_ids.len() * std::mem::size_of::<u64>()
                + pl.frequencies.len() * std::mem::size_of::<f32>()
                + pl.positions
                    .as_ref()
                    .map(|a| a.get_array_memory_size())
                    .unwrap_or(0);
        }
        total += self.docs.len() * (std::mem::size_of::<u64>() + std::mem::size_of::<u32>());
        total
    }

    /// Freeze the visible contents of `tail` into a new partition. Returns
    /// `None` if the tail has no visible docs.
    fn from_tail(tail: &TailIndex) -> Option<Self> {
        let snap = tail.snapshot();
        if snap.visible_count == 0 {
            return None;
        }
        // Assign dense local doc ids in row-position order.
        let mut docs = DocSet::default();
        let mut pos_to_doc: HashMap<u64, u32> = HashMap::new();
        for batch in snap.batches.iter().take(snap.visible_count) {
            for i in 0..batch.rows as usize {
                let rp = batch.row_offset + i as u64;
                let doc_id = docs.append(rp, batch.doc_lengths[i]);
                pos_to_doc.insert(rp, doc_id);
            }
        }
        let mut token_ids: HashMap<Arc<str>, u32> = HashMap::new();
        let mut raw: Vec<Vec<(u32, u32, Vec<u32>)>> = Vec::new();
        for entry in tail.terms.iter() {
            let term: &Arc<str> = entry.key();
            let slice = entry.value().load();
            let mut docs_for_term: Vec<(u32, u32, Vec<u32>)> = Vec::new();
            for chunk in &slice.chunks {
                if chunk.batch_position >= snap.visible_count {
                    continue;
                }
                for (i, rp) in chunk.row_positions.iter().enumerate() {
                    let Some(&doc_id) = pos_to_doc.get(rp) else {
                        continue;
                    };
                    let pos = chunk
                        .positions
                        .as_ref()
                        .map(|p| p.doc_positions(i).to_vec())
                        .unwrap_or_default();
                    docs_for_term.push((doc_id, chunk.frequencies[i], pos));
                }
            }
            if docs_for_term.is_empty() {
                continue;
            }
            docs_for_term.sort_by_key(|(d, _, _)| *d);
            let id = raw.len() as u32;
            token_ids.insert(term.clone(), id);
            raw.push(docs_for_term);
        }
        let postings = raw.into_iter().map(freeze_postings_one).collect();
        Some(Self {
            token_ids,
            postings,
            docs,
        })
    }

    /// Merge several partitions into one. Local doc ids are reassigned by
    /// concatenation, which keeps each merged per-token posting list sorted.
    fn merge(parts: &[Arc<Self>]) -> Self {
        let mut token_ids: HashMap<Arc<str>, u32> = HashMap::new();
        let mut raw: Vec<Vec<(u32, u32, Vec<u32>)>> = Vec::new();
        let mut docs = DocSet::default();
        let mut doc_offset: u32 = 0;
        for p in parts {
            for (rp, nt) in p.docs.iter() {
                docs.append(*rp, *nt);
            }
            for (token, &local_id) in &p.token_ids {
                let merged_id = *token_ids.entry(token.clone()).or_insert_with(|| {
                    let id = raw.len() as u32;
                    raw.push(Vec::new());
                    id
                });
                let plain = posting_as_plain(&p.postings[local_id as usize]);
                for i in 0..plain.len() {
                    raw[merged_id as usize].push((
                        plain.row_ids[i] as u32 + doc_offset,
                        plain.frequencies[i] as u32,
                        read_positions(plain, i),
                    ));
                }
            }
            doc_offset += p.docs.len() as u32;
        }
        let postings = raw
            .into_iter()
            .map(|mut v| {
                v.sort_by_key(|(d, _, _)| *d);
                freeze_postings_one(v)
            })
            .collect();
        Self {
            token_ids,
            postings,
            docs,
        }
    }

    /// WAND OR/AND-search; candidates scored with `scorer` and reported as
    /// MemTable row positions.
    /// WAND runs unbounded (no `limit`): block-max top-k pruning needs
    /// per-term `max_score` upper bounds, which in-memory posting lists do
    /// not carry, so a bounded WAND would drop valid top-k docs. The scan is
    /// still O(matches); the caller truncates to the real limit after merge.
    fn search_match(
        &self,
        tokens: &[String],
        operator: Operator,
        scorer: &MemBM25Scorer,
    ) -> Result<Vec<FtsEntry>> {
        let mut wand_terms: Vec<WandTerm> = Vec::new();
        let mut tok_order: Vec<String> = Vec::new();
        for (qpos, t) in tokens.iter().enumerate() {
            match self.token_ids.get(t.as_str()) {
                Some(&id) => {
                    wand_terms.push((
                        t.clone(),
                        id,
                        qpos as u32,
                        1.0,
                        self.postings[id as usize].clone(),
                    ));
                    tok_order.push(t.clone());
                }
                // Term absent: an AND query cannot match here; OR drops it.
                None if operator == Operator::And => return Ok(Vec::new()),
                None => {}
            }
        }
        if wand_terms.is_empty() {
            return Ok(Vec::new());
        }
        let params = FtsSearchParams::new();
        let cands = wand_search(operator, wand_terms, &self.docs, scorer.clone(), &params)?;
        Ok(cands
            .into_iter()
            .map(|c| {
                let score: f32 = c
                    .freqs
                    .iter()
                    .map(|(ti, f)| {
                        scorer.query_weight(&tok_order[*ti as usize])
                            * scorer.doc_weight(*f, c.doc_length)
                    })
                    .sum();
                FtsEntry {
                    row_position: self.docs.row_id(c.row_id as u32),
                    score,
                }
            })
            .collect())
    }

    /// Phrase-search: WAND-And finds docs containing every token, then token
    /// positions are verified. `tokens.len() >= 2`.
    fn search_phrase(
        &self,
        tokens: &[String],
        slop: u32,
        scorer: &MemBM25Scorer,
    ) -> Result<Vec<FtsEntry>> {
        let mut wand_terms: Vec<WandTerm> = Vec::new();
        let mut token_id_order: Vec<u32> = Vec::new();
        for (qpos, t) in tokens.iter().enumerate() {
            match self.token_ids.get(t.as_str()) {
                Some(&id) => {
                    wand_terms.push((
                        t.clone(),
                        id,
                        qpos as u32,
                        1.0,
                        self.postings[id as usize].clone(),
                    ));
                    token_id_order.push(id);
                }
                // A phrase needs every token present in this partition.
                None => return Ok(Vec::new()),
            }
        }
        let params = FtsSearchParams::new();
        let cands = wand_search(
            Operator::And,
            wand_terms,
            &self.docs,
            scorer.clone(),
            &params,
        )?;
        let mut results = Vec::new();
        for c in cands {
            let local_doc = c.row_id; // partition-local doc id
            let mut all_positions: Vec<Vec<u32>> = Vec::with_capacity(tokens.len());
            let mut present = true;
            for &tid in &token_id_order {
                let plain = posting_as_plain(&self.postings[tid as usize]);
                match plain.row_ids.binary_search(&local_doc) {
                    Ok(idx) => all_positions.push(read_positions(plain, idx)),
                    Err(_) => {
                        present = false;
                        break;
                    }
                }
            }
            if !present || !phrase_matches(&all_positions, slop) {
                continue;
            }
            let score: f32 = c
                .freqs
                .iter()
                .map(|(ti, f)| {
                    scorer.query_weight(&tokens[*ti as usize]) * scorer.doc_weight(*f, c.doc_length)
                })
                .sum();
            results.push(FtsEntry {
                row_position: self.docs.row_id(local_doc as u32),
                score,
            });
        }
        Ok(results)
    }
}

/// Convert per-token `(doc_id, freq, positions)` postings — sorted by doc id —
/// into an on-disk-shaped `PlainPostingList`. `max_score` is left `None`;
/// WAND derives per-term bounds from the posting list at query time.
fn freeze_postings_one(docs: Vec<(u32, u32, Vec<u32>)>) -> PostingList {
    let row_ids: Vec<u64> = docs.iter().map(|(d, _, _)| *d as u64).collect();
    let freqs: Vec<f32> = docs.iter().map(|(_, f, _)| *f as f32).collect();
    let mut positions = ListBuilder::new(Int32Builder::new());
    for (_, _, doc_positions) in &docs {
        for p in doc_positions {
            positions.values().append_value(*p as i32);
        }
        positions.append(true);
    }
    PostingList::Plain(PlainPostingList::new(
        ScalarBuffer::from(row_ids),
        ScalarBuffer::from(freqs),
        None,
        Some(positions.finish()),
    ))
}

/// In-memory FTS partitions only ever build `PostingList::Plain`.
fn posting_as_plain(p: &PostingList) -> &PlainPostingList {
    match p {
        PostingList::Plain(pl) => pl,
        PostingList::Compressed(_) => {
            unreachable!("in-memory FTS partitions only build plain posting lists")
        }
    }
}

/// Read the position list for the `idx`-th posting as a `Vec<u32>`.
fn read_positions(plain: &PlainPostingList, idx: usize) -> Vec<u32> {
    match plain.positions(idx) {
        Some(arr) => arr
            .as_any()
            .downcast_ref::<Int32Array>()
            .map(|a| a.values().iter().map(|&v| v as u32).collect())
            .unwrap_or_default(),
        None => Vec::new(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    fn create_test_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("description", DataType::Utf8, true),
        ]))
    }

    fn create_test_batch(schema: &ArrowSchema) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![0, 1, 2])),
                Arc::new(StringArray::from(vec![
                    "hello world",
                    "goodbye world",
                    "hello again",
                ])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_fts_index_insert_and_search() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        assert_eq!(index.doc_count(), 3);

        let entries = index.search("hello");
        assert_eq!(entries.len(), 2);

        let entries = index.search("world");
        assert_eq!(entries.len(), 2);

        let entries = index.search("goodbye");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_position, 1);

        let entries = index.search("nonexistent");
        assert!(entries.is_empty());
    }

    fn create_phrase_test_batch(schema: &ArrowSchema) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4])),
                Arc::new(StringArray::from(vec![
                    "alpha beta gamma",
                    "beta alpha gamma",
                    "alpha delta beta gamma",
                    "alpha gamma",
                    "alpha delta epsilon beta gamma",
                ])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_phrase_search_exact_match() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_phrase_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let entries = index.search_phrase("alpha beta", 0);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_position, 0);

        let batch2 = create_test_batch(&schema);
        let index2 = FtsMemIndex::new(1, "description".to_string());
        index2.insert(&batch2, 0).unwrap();

        let entries = index2.search_phrase("hello world", 0);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_position, 0);

        let entries = index2.search_phrase("goodbye world", 0);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_position, 1);
    }

    #[test]
    fn test_phrase_search_with_slop() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_phrase_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let entries = index.search_phrase("alpha beta", 0);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_position, 0);

        let entries = index.search_phrase("alpha beta", 1);
        assert_eq!(entries.len(), 2);
        let positions: Vec<_> = entries.iter().map(|e| e.row_position).collect();
        assert!(positions.contains(&0));
        assert!(positions.contains(&2));

        let entries = index.search_phrase("alpha beta", 2);
        assert_eq!(entries.len(), 3);

        let entries = index.search_phrase("alpha gamma", 0);
        assert_eq!(entries.len(), 2);

        let entries = index.search_phrase("alpha gamma", 1);
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn test_phrase_search_no_match() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_phrase_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let entries = index.search_phrase("beta alpha", 0);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_position, 1);

        let entries = index.search_phrase("nonexistent phrase", 0);
        assert!(entries.is_empty());

        let entries = index.search_phrase("alpha hello", 0);
        assert!(entries.is_empty());

        let entries = index.search_phrase("gamma alpha", 0);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_phrase_search_single_token() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_phrase_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let phrase_entries = index.search_phrase("alpha", 0);
        let search_entries = index.search("alpha");
        assert_eq!(phrase_entries.len(), search_entries.len());
    }

    #[test]
    fn test_phrase_search_empty() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let entries = index.search_phrase("", 0);
        assert!(entries.is_empty());
    }

    fn create_boolean_test_batch(schema: &ArrowSchema) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4])),
                Arc::new(StringArray::from(vec![
                    "rust programming language",
                    "python programming language",
                    "rust web server",
                    "python web framework",
                    "javascript programming",
                ])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_boolean_must_only() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boolean_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::boolean()
            .must(FtsQueryExpr::match_query("rust"))
            .must(FtsQueryExpr::match_query("programming"))
            .build();

        let entries = index.search_query(&query);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_position, 0);
    }

    #[test]
    fn test_boolean_should_only() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boolean_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::boolean()
            .should(FtsQueryExpr::match_query("rust"))
            .should(FtsQueryExpr::match_query("python"))
            .build();

        let entries = index.search_query(&query);
        assert_eq!(entries.len(), 4);
        let positions: Vec<_> = entries.iter().map(|e| e.row_position).collect();
        assert!(positions.contains(&0));
        assert!(positions.contains(&1));
        assert!(positions.contains(&2));
        assert!(positions.contains(&3));
    }

    #[test]
    fn test_boolean_must_not_only() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boolean_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::boolean()
            .must_not(FtsQueryExpr::match_query("rust"))
            .build();

        let entries = index.search_query(&query);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_boolean_must_with_should() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boolean_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::boolean()
            .must(FtsQueryExpr::match_query("programming"))
            .should(FtsQueryExpr::match_query("rust"))
            .build();

        let entries = index.search_query(&query);
        assert_eq!(entries.len(), 3);

        let doc0 = entries.iter().find(|e| e.row_position == 0).unwrap();
        let doc1 = entries.iter().find(|e| e.row_position == 1).unwrap();
        assert!(doc0.score > doc1.score);
    }

    #[test]
    fn test_boolean_must_with_must_not() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boolean_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::boolean()
            .must(FtsQueryExpr::match_query("programming"))
            .must_not(FtsQueryExpr::match_query("python"))
            .build();

        let entries = index.search_query(&query);
        assert_eq!(entries.len(), 2);

        let positions: Vec<_> = entries.iter().map(|e| e.row_position).collect();
        assert!(positions.contains(&0));
        assert!(positions.contains(&4));
        assert!(!positions.contains(&1));
    }

    #[test]
    fn test_boolean_combined() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boolean_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::boolean()
            .must(FtsQueryExpr::match_query("web"))
            .should(FtsQueryExpr::match_query("rust"))
            .must_not(FtsQueryExpr::match_query("framework"))
            .build();

        let entries = index.search_query(&query);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_position, 2);
    }

    #[test]
    fn test_boolean_nested_phrase() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boolean_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::boolean()
            .must(FtsQueryExpr::phrase("programming language"))
            .build();

        let entries = index.search_query(&query);
        assert_eq!(entries.len(), 2);
        let positions: Vec<_> = entries.iter().map(|e| e.row_position).collect();
        assert!(positions.contains(&0));
        assert!(positions.contains(&1));
    }

    #[test]
    fn test_search_query_match() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::match_query("hello");
        let entries = index.search_query(&query);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_search_query_phrase() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::phrase("hello world");
        let entries = index.search_query(&query);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_position, 0);
    }

    #[test]
    fn test_search_query_with_boost() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query_no_boost = FtsQueryExpr::match_query("hello");
        let query_with_boost = FtsQueryExpr::match_query("hello").with_boost(2.0);

        let entries_no_boost = index.search_query(&query_no_boost);
        let entries_with_boost = index.search_query(&query_with_boost);

        assert_eq!(entries_no_boost.len(), entries_with_boost.len());
        for e1 in &entries_no_boost {
            let e2 = entries_with_boost
                .iter()
                .find(|e| e.row_position == e1.row_position)
                .unwrap();
            let expected = e1.score * 2.0;
            assert!((e2.score - expected).abs() < 0.001);
        }
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        assert_eq!(levenshtein_distance("hello", "hallo"), 1);
        assert_eq!(levenshtein_distance("hello", "hell"), 1);
        assert_eq!(levenshtein_distance("hello", "helloo"), 1);
        assert_eq!(levenshtein_distance("hello", "hxllo"), 1);
        assert_eq!(levenshtein_distance("hello", "hxxlo"), 2);
        assert_eq!(levenshtein_distance("abc", "xyz"), 3);
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("hello", ""), 5);
        assert_eq!(levenshtein_distance("", "hello"), 5);
        assert_eq!(levenshtein_distance("Hello", "hello"), 1);
    }

    #[test]
    fn test_auto_fuzziness() {
        assert_eq!(auto_fuzziness(""), 0);
        assert_eq!(auto_fuzziness("a"), 0);
        assert_eq!(auto_fuzziness("ab"), 0);
        assert_eq!(auto_fuzziness("abc"), 1);
        assert_eq!(auto_fuzziness("abcd"), 1);
        assert_eq!(auto_fuzziness("abcde"), 1);
        assert_eq!(auto_fuzziness("abcdef"), 2);
        assert_eq!(auto_fuzziness("programming"), 2);
    }

    fn create_fuzzy_test_batch(schema: &ArrowSchema) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4])),
                Arc::new(StringArray::from(vec![
                    "alpha beta gamma",
                    "alpho beta delta",
                    "alpha delta epsilon",
                    "omega zeta",
                    "alphax gamma",
                ])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_expand_fuzzy_exact_match() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let matches = index.expand_fuzzy("alpha", 0, 50);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].0, "alpha");
        assert_eq!(matches[0].1, 0);

        let matches = index.expand_fuzzy("nonexistent", 0, 50);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_expand_fuzzy_single_edit() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let matches = index.expand_fuzzy("alpho", 1, 50);
        assert!(matches.iter().any(|(t, d)| t == "alpha" && *d == 1));
        assert!(matches.iter().any(|(t, _)| t == "alpho"));
    }

    #[test]
    fn test_expand_fuzzy_max_expansions() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let matches = index.expand_fuzzy("a", 10, 3);
        assert!(matches.len() <= 3);
    }

    #[test]
    fn test_search_fuzzy_basic() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let entries = index.search_fuzzy("alpho", Some(1), 50);
        assert!(!entries.is_empty());
    }

    #[test]
    fn test_search_fuzzy_auto_fuzziness() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let entries = index.search_fuzzy("alpho", None, 50);
        assert!(!entries.is_empty());
    }

    #[test]
    fn test_search_fuzzy_no_match() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let entries = index.search_fuzzy("xyz", Some(0), 50);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_search_query_fuzzy() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::fuzzy("alpho");
        let entries = index.search_query(&query);
        assert!(!entries.is_empty());
    }

    #[test]
    fn test_search_query_fuzzy_with_distance() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::fuzzy_with_distance("alpho", 1);
        let entries = index.search_query(&query);
        assert!(!entries.is_empty());
    }

    #[test]
    fn test_search_query_fuzzy_with_boost() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query_no_boost = FtsQueryExpr::fuzzy("alpho");
        let query_with_boost = FtsQueryExpr::fuzzy("alpho").with_boost(2.0);

        let entries_no_boost = index.search_query(&query_no_boost);
        let entries_with_boost = index.search_query(&query_with_boost);
        assert_eq!(entries_no_boost.len(), entries_with_boost.len());

        for e1 in &entries_no_boost {
            let e2 = entries_with_boost
                .iter()
                .find(|e| e.row_position == e1.row_position)
                .unwrap();
            let expected = e1.score * 2.0;
            assert!((e2.score - expected).abs() < 0.001);
        }
    }

    #[test]
    fn test_boolean_with_fuzzy() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::boolean()
            .must(FtsQueryExpr::fuzzy_with_distance("alpho", 1))
            .must_not(FtsQueryExpr::match_query("delta"))
            .build();

        let entries = index.search_query(&query);
        let positions: Vec<_> = entries.iter().map(|e| e.row_position).collect();
        assert!(!positions.contains(&1));
        assert!(!positions.contains(&2));
        assert!(positions.contains(&0));
    }

    fn create_boost_test_batch(schema: &ArrowSchema) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4])),
                Arc::new(StringArray::from(vec![
                    "rust programming language",
                    "python programming language",
                    "rust web server",
                    "python web framework",
                    "javascript programming",
                ])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_boost_query_positive_only() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boost_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::boosting(FtsQueryExpr::match_query("programming"));
        let entries = index.search_query(&query);
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn test_boost_query_with_negative() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boost_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::boosting_with_negative(
            FtsQueryExpr::match_query("programming"),
            FtsQueryExpr::match_query("python"),
            0.5,
        );
        let entries = index.search_query(&query);
        assert_eq!(entries.len(), 3);

        let doc0 = entries.iter().find(|e| e.row_position == 0).unwrap();
        let doc1 = entries.iter().find(|e| e.row_position == 1).unwrap();
        let doc4 = entries.iter().find(|e| e.row_position == 4).unwrap();
        assert!(doc1.score < doc0.score);
        assert!(doc1.score < doc4.score);
    }

    #[test]
    fn test_boost_query_negative_boost_factor() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boost_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query_no_demote = FtsQueryExpr::boosting_with_negative(
            FtsQueryExpr::match_query("programming"),
            FtsQueryExpr::match_query("python"),
            1.0,
        );
        let query_half_demote = FtsQueryExpr::boosting_with_negative(
            FtsQueryExpr::match_query("programming"),
            FtsQueryExpr::match_query("python"),
            0.5,
        );
        let query_zero_demote = FtsQueryExpr::boosting_with_negative(
            FtsQueryExpr::match_query("programming"),
            FtsQueryExpr::match_query("python"),
            0.0,
        );

        let r_no = index.search_query(&query_no_demote);
        let r_half = index.search_query(&query_half_demote);
        let r_zero = index.search_query(&query_zero_demote);

        let s_no = r_no.iter().find(|e| e.row_position == 1).unwrap().score;
        let s_half = r_half.iter().find(|e| e.row_position == 1).unwrap().score;
        let s_zero = r_zero.iter().find(|e| e.row_position == 1).unwrap().score;

        assert!((s_half - s_no * 0.5).abs() < 0.001);
        assert!(s_zero.abs() < 0.001);
    }

    #[test]
    fn test_boost_query_no_negative_match() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boost_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::boosting_with_negative(
            FtsQueryExpr::match_query("rust"),
            FtsQueryExpr::match_query("python"),
            0.1,
        );
        let entries = index.search_query(&query);
        assert_eq!(entries.len(), 2);

        let baseline = index.search_query(&FtsQueryExpr::match_query("rust"));
        for entry in &entries {
            let b = baseline
                .iter()
                .find(|e| e.row_position == entry.row_position)
                .unwrap();
            assert!((entry.score - b.score).abs() < 0.001);
        }
    }

    #[test]
    fn test_search_options_default() {
        let options = SearchOptions::default();
        assert_eq!(options.wand_factor, 1.0);
        assert!(options.limit.is_none());
    }

    #[test]
    fn test_search_options_builder() {
        let options = SearchOptions::new().with_wand_factor(0.5).with_limit(10);
        assert_eq!(options.wand_factor, 0.5);
        assert_eq!(options.limit, Some(10));
    }

    #[test]
    fn test_search_options_wand_factor_clamped() {
        let options = SearchOptions::new().with_wand_factor(2.0);
        assert_eq!(options.wand_factor, 1.0);
        let options = SearchOptions::new().with_wand_factor(-0.5);
        assert_eq!(options.wand_factor, 0.0);
    }

    fn create_wand_test_batch(schema: &ArrowSchema) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4])),
                Arc::new(StringArray::from(vec![
                    "alpha alpha alpha beta",
                    "alpha beta gamma",
                    "beta gamma delta",
                    "alpha alpha",
                    "alpha",
                ])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_search_with_options_full_recall() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_wand_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::match_query("alpha");
        let results = index.search_with_options(&query, SearchOptions::default());
        assert_eq!(results.len(), 4);
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }

    #[test]
    fn test_search_with_options_with_limit() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_wand_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::match_query("alpha");
        let options = SearchOptions::new().with_limit(2);
        let results = index.search_with_options(&query, options);
        assert_eq!(results.len(), 2);

        let mut full = index.search_query(&query);
        full.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        assert_eq!(results[0].row_position, full[0].row_position);
        assert_eq!(results[1].row_position, full[1].row_position);
    }

    #[test]
    fn test_search_with_options_wand_factor_pruning() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_wand_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::match_query("alpha");
        let mut full = index.search_query(&query);
        full.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        let options = SearchOptions::new().with_wand_factor(0.5);
        let results = index.search_with_options(&query, options);

        if !results.is_empty() {
            let max_score = full[0].score;
            let threshold = max_score * 0.5;
            for result in &results {
                assert!(result.score >= threshold - 0.001);
            }
        }
    }

    #[test]
    fn test_search_with_options_wand_factor_with_limit() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_wand_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::match_query("alpha");
        let options = SearchOptions::new().with_limit(2).with_wand_factor(0.5);
        let results = index.search_with_options(&query, options);
        assert!(results.len() <= 2);
        if results.len() > 1 {
            assert!(results[0].score >= results[1].score);
        }
    }

    #[test]
    fn test_search_with_options_empty_results() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_wand_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::match_query("nonexistent");
        let options = SearchOptions::new().with_limit(10).with_wand_factor(0.5);
        let results = index.search_with_options(&query, options);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_with_options_boolean_query() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_wand_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::boolean()
            .should(FtsQueryExpr::match_query("alpha"))
            .should(FtsQueryExpr::match_query("beta"))
            .build();

        let options = SearchOptions::new().with_limit(3);
        let results = index.search_with_options(&query, options);
        assert!(results.len() <= 3);
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }

    // ====== New tests for SWMR semantics, Utf8View, memory accounting ======

    #[test]
    fn test_utf8view_insert_and_search() {
        // Mirror of test_fts_index_insert_and_search but with a Utf8View column.
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("description", DataType::Utf8View, true),
        ]));
        let index = FtsMemIndex::new(1, "description".to_string());

        let view =
            arrow_array::StringViewArray::from(vec!["hello world", "goodbye world", "hello again"]);
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![0, 1, 2])), Arc::new(view)],
        )
        .unwrap();
        index.insert(&batch, 0).unwrap();

        assert_eq!(index.doc_count(), 3);
        assert_eq!(index.search("hello").len(), 2);
        assert_eq!(index.search("world").len(), 2);
        assert_eq!(index.search("goodbye").len(), 1);
    }

    #[test]
    fn test_memory_usage_grows_with_inserts() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let empty = index.memory_usage();
        index.insert(&create_test_batch(&schema), 0).unwrap();
        let after_one = index.memory_usage();
        index
            .insert(&create_phrase_test_batch(&schema), 100)
            .unwrap();
        let after_two = index.memory_usage();

        assert!(after_one > empty, "memory should grow after first insert");
        assert!(
            after_two > after_one,
            "memory should grow after second insert"
        );
    }

    #[test]
    fn test_partial_doc_never_visible_phrase() {
        // A phrase query inside a single document must either match fully
        // (both phrase tokens present) or not match at all — readers must
        // never observe a half-inserted doc that contains only one of the
        // phrase tokens. Since `search_phrase` only returns rows where
        // every token's position constraint holds, any returned entry
        // implicitly proves both tokens were visible together.
        let schema = create_test_schema();
        let index = Arc::new(FtsMemIndex::new(1, "description".to_string()));

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut readers = Vec::new();
        for _ in 0..4 {
            let idx = index.clone();
            let stop = stop.clone();
            readers.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let entries = idx.search_phrase("hello world", 0);
                    for e in &entries {
                        // "hello world" appears as the doc at offset 0 of
                        // every inserted batch, whose row_offset is a
                        // multiple of 100 — so any matched row_position
                        // must be a multiple of 100 (the row_offset itself,
                        // not offset+1 or offset+2).
                        assert_eq!(
                            e.row_position % 100,
                            0,
                            "phrase 'hello world' should only match the row_offset row of each batch, but got row_position {}",
                            e.row_position
                        );
                        assert!(e.score.is_finite() && e.score >= 0.0);
                    }
                }
            }));
        }

        for i in 0..50 {
            let batch = create_test_batch(&schema);
            index.insert(&batch, (i * 100) as u64).unwrap();
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        stop.store(true, Ordering::Relaxed);

        for r in readers {
            r.join().unwrap();
        }
    }

    #[test]
    fn test_swmr_visibility_torture() {
        let schema = create_test_schema();
        let index = Arc::new(FtsMemIndex::new(1, "description".to_string()));
        let stop = Arc::new(AtomicBool::new(false));

        // 8 reader threads issuing match queries.
        let mut readers = Vec::new();
        for _ in 0..8 {
            let idx = index.clone();
            let stop = stop.clone();
            readers.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let r = idx.search("hello");
                    // BM25 scores must be finite and non-negative.
                    for e in &r {
                        assert!(e.score.is_finite());
                        assert!(e.score >= 0.0);
                    }
                }
            }));
        }

        // One writer thread.
        let writer = {
            let idx = index.clone();
            std::thread::spawn(move || {
                for i in 0..200 {
                    let batch = create_test_batch(&schema);
                    idx.insert(&batch, (i * 100) as u64).unwrap();
                }
            })
        };

        writer.join().unwrap();
        stop.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }

        // After all inserts, doc_count must be 200 batches × 3 rows.
        assert_eq!(index.doc_count(), 600);
    }

    #[test]
    fn test_to_index_builder_reversed_smoke() {
        // Ensure flush works on a minimal input.
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());
        let batch = create_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let builder = index.to_index_builder_reversed(42, 3).unwrap();
        // The builder can be consumed by callers; we just check it built.
        assert!(builder.id() > 0 || builder.id() == 42);
    }

    #[test]
    fn test_unsupported_column_type_errors() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("description", DataType::Int32, true),
        ]));
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![0, 1])),
                Arc::new(Int32Array::from(vec![1, 2])),
            ],
        )
        .unwrap();

        let err = index.insert(&batch, 0).unwrap_err();
        assert!(err.to_string().contains("only supports"), "{err}");
    }

    // ===== Partition-structured redesign =====

    /// Sorted row positions of a result set.
    fn rows(mut entries: Vec<FtsEntry>) -> Vec<u64> {
        entries.sort_by_key(|e| e.row_position);
        entries.into_iter().map(|e| e.row_position).collect()
    }

    /// An index whose 4 docs (row positions 100..104) have all been frozen
    /// into a single partition (`freeze_threshold_rows = 1`).
    fn build_test_partition_index() -> FtsMemIndex {
        let index = FtsMemIndex::new(1, "description".to_string()).with_freeze_threshold_rows(1);
        let batch = RecordBatch::try_new(
            create_test_schema(),
            vec![
                Arc::new(Int32Array::from(vec![0, 1, 2, 3])),
                Arc::new(StringArray::from(vec![
                    "apple banana",
                    "apple cherry",
                    "banana",
                    "apple apple date",
                ])),
            ],
        )
        .unwrap();
        index.insert(&batch, 100).unwrap();
        index
    }

    #[test]
    fn test_partition_build_and_search() {
        let index = build_test_partition_index();
        let st = index.state.load_full();
        assert_eq!(st.partitions.len(), 1, "the batch should have frozen");
        assert_eq!(st.tail.visible_count(), 0, "the tail should be empty");

        let p = &st.partitions[0];
        assert_eq!(p.doc_count(), 4);
        assert_eq!(p.total_tokens(), 8); // 2 + 2 + 1 + 3
        // Term dictionary stores the *tokenized* form, so probe via the
        // configured tokenizer rather than the raw word.
        let tok = |w: &str| index.tokenize_for_search(w).pop().unwrap();
        assert_eq!(p.token_df(&tok("apple")), 3);
        assert_eq!(p.token_df(&tok("banana")), 2);
        assert_eq!(p.token_df("definitely_missing"), 0);

        // Searching the index must return the partition's row positions.
        assert_eq!(rows(index.search("apple")), vec![100, 101, 103]);
        assert_eq!(rows(index.search("banana")), vec![100, 102]);
        assert_eq!(rows(index.search("date")), vec![103]);
        assert!(index.search("definitely_missing").is_empty());
    }

    #[test]
    fn test_partition_and_query_short_circuits_missing_term() {
        let index = build_test_partition_index();
        let st = index.state.load_full();
        let p = &st.partitions[0];
        let tail_snap = st.tail.snapshot();
        let apple = index.tokenize_for_search("apple").pop().unwrap();
        // "apple" is present -> an OR search over it matches.
        let or_scorer = build_scorer(&st, &tail_snap, std::slice::from_ref(&apple));
        let or_hits = p
            .search_match(std::slice::from_ref(&apple), Operator::Or, &or_scorer)
            .unwrap();
        assert_eq!(or_hits.len(), 3);
        // Adding an absent term to an AND query short-circuits to nothing.
        let and_tokens = vec![apple, "definitely_missing".to_string()];
        let and_scorer = build_scorer(&st, &tail_snap, &and_tokens);
        let and_hits = p
            .search_match(&and_tokens, Operator::And, &and_scorer)
            .unwrap();
        assert!(and_hits.is_empty());
    }

    #[test]
    fn test_freeze_produces_multiple_partitions() {
        // freeze_threshold_rows = 3 with 3-row batches => one partition per
        // batch insert.
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string()).with_freeze_threshold_rows(3);
        for i in 0..5 {
            index
                .insert(&create_test_batch(&schema), (i * 100) as u64)
                .unwrap();
        }
        let st = index.state.load_full();
        assert_eq!(st.partitions.len(), 5, "every batch should have frozen");
        assert_eq!(st.tail.visible_count(), 0);
        assert_eq!(index.doc_count(), 15);

        // "hello" appears in docs 0 and 2 of every batch; "world" in 0 and 1.
        assert_eq!(
            rows(index.search("hello")),
            vec![0, 2, 100, 102, 200, 202, 300, 302, 400, 402]
        );
        assert_eq!(
            rows(index.search("world")),
            vec![0, 1, 100, 101, 200, 201, 300, 301, 400, 401]
        );
        assert!(index.search("definitely_missing").is_empty());
        // Every score must be finite and positive.
        for e in index.search("world") {
            assert!(e.score.is_finite() && e.score > 0.0);
        }
    }

    #[test]
    fn test_search_spans_partitions_and_tail() {
        // freeze_threshold_rows = 5, 3-row batches: batch 1 -> tail(3);
        // batch 2 -> tail(6) freezes -> partition(6); batch 3 -> tail(3).
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string()).with_freeze_threshold_rows(5);
        index.insert(&create_test_batch(&schema), 0).unwrap();
        index.insert(&create_test_batch(&schema), 100).unwrap();
        index.insert(&create_test_batch(&schema), 200).unwrap();
        let st = index.state.load_full();
        assert_eq!(st.partitions.len(), 1, "the first two batches froze");
        assert_eq!(st.tail.visible_count(), 1, "the third batch is in the tail");
        assert_eq!(index.doc_count(), 9);

        // Results must merge the partition (rows 0..200) and the tail (200..).
        assert_eq!(rows(index.search("hello")), vec![0, 2, 100, 102, 200, 202]);
        assert_eq!(rows(index.search("goodbye")), vec![1, 101, 201]);
    }

    #[test]
    fn test_flush_merges_partitions_and_tail() {
        // A frozen partition plus a live tail must all reach the builder.
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string()).with_freeze_threshold_rows(5);
        index.insert(&create_test_batch(&schema), 0).unwrap();
        index.insert(&create_test_batch(&schema), 100).unwrap();
        index.insert(&create_test_batch(&schema), 200).unwrap();
        let st = index.state.load_full();
        assert_eq!(st.partitions.len(), 1);
        assert_eq!(st.tail.visible_count(), 1);

        let builder = index.to_index_builder_reversed(7, 300).unwrap();
        assert_eq!(builder.id(), 7);
        // A non-empty builder proves the partition and the tail both reached
        // the flush; end-to-end flush correctness is covered by the MemTable
        // flush integration tests.
        assert!(
            !builder.is_empty(),
            "9 docs across partition + tail must flush"
        );
    }

    #[test]
    fn test_phrase_across_partitions() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string()).with_freeze_threshold_rows(3);
        // Each batch has "hello world" at its row_offset row.
        for i in 0..4 {
            index
                .insert(&create_test_batch(&schema), (i * 100) as u64)
                .unwrap();
        }
        let st = index.state.load_full();
        assert_eq!(st.partitions.len(), 4);
        // The phrase must be found in every partition.
        assert_eq!(
            rows(index.search_phrase("hello world", 0)),
            vec![0, 100, 200, 300]
        );
        // A phrase not present anywhere returns nothing.
        assert!(index.search_phrase("world hello", 0).is_empty());
    }

    #[test]
    fn test_merge_caps_partition_count() {
        // freeze_threshold_rows = 1 freezes every 3-row batch; past
        // MAX_PARTITIONS (32) the partitions are merged into one.
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string()).with_freeze_threshold_rows(1);
        for i in 0..40 {
            index
                .insert(&create_test_batch(&schema), (i * 100) as u64)
                .unwrap();
        }
        let st = index.state.load_full();
        assert!(
            st.partitions.len() <= FtsMemIndex::MAX_PARTITIONS,
            "partition count {} should be capped",
            st.partitions.len()
        );
        assert_eq!(index.doc_count(), 120);
        // Merge must preserve every posting: "hello" hits 2 docs per batch.
        assert_eq!(index.search("hello").len(), 80);
        for e in index.search("hello") {
            assert!(e.score.is_finite() && e.score > 0.0);
        }
    }

    #[test]
    fn test_freeze_during_concurrent_search() {
        // Searches running concurrently with freezes must never observe a
        // doc twice or miss one: every result row is valid and scored.
        let schema = create_test_schema();
        let index =
            Arc::new(FtsMemIndex::new(1, "description".to_string()).with_freeze_threshold_rows(6));
        let stop = Arc::new(AtomicBool::new(false));
        let mut readers = Vec::new();
        for _ in 0..4 {
            let idx = index.clone();
            let stop = stop.clone();
            readers.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    for e in idx.search("world") {
                        // "world" only ever appears in docs 0 and 1 of a
                        // batch, whose row_offset is a multiple of 100.
                        assert!(e.row_position % 100 < 2);
                        assert!(e.score.is_finite() && e.score >= 0.0);
                    }
                }
            }));
        }
        for i in 0..60 {
            index
                .insert(&create_test_batch(&schema), (i * 100) as u64)
                .unwrap();
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        stop.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }
        assert_eq!(index.doc_count(), 180);
        assert_eq!(index.search("world").len(), 120);
    }

    #[test]
    fn test_limited_search_returns_exact_top_k_across_partitions() {
        // Regression: a limited `search_with_options` over many partitions
        // must still return the *exact* global top-k. Per-partition WAND must
        // not prune by a limit it cannot bound (in-memory posting lists carry
        // no `max_score`), or valid top-k docs are silently dropped.
        let index = FtsMemIndex::new(1, "description".to_string()).with_freeze_threshold_rows(2);
        // 8 docs (4 batches of 2) -> 4 partitions. "lion" appears with tf
        // 1..=6, so the 6 matching docs have strictly distinct BM25 scores.
        let texts = [
            ["lion", "lion lion"],
            ["lion lion lion", "lion lion lion lion"],
            ["cat", "lion lion lion lion lion"],
            ["lion lion lion lion lion lion", "dog"],
        ];
        for (b, pair) in texts.iter().enumerate() {
            let batch = RecordBatch::try_new(
                create_test_schema(),
                vec![
                    Arc::new(Int32Array::from(vec![0, 1])),
                    Arc::new(StringArray::from(pair.to_vec())),
                ],
            )
            .unwrap();
            index.insert(&batch, (b * 100) as u64).unwrap();
        }
        assert_eq!(index.state.load_full().partitions.len(), 4);

        // Exact full ranking via the unbounded search path.
        let mut full = index.search("lion");
        full.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        assert_eq!(full.len(), 6);

        // The limited top-3 must equal the first 3 of the full ranking.
        let limited = index.search_with_options(
            &FtsQueryExpr::match_query("lion"),
            SearchOptions::new().with_limit(3),
        );
        assert_eq!(limited.len(), 3);
        let got: Vec<u64> = limited.iter().map(|e| e.row_position).collect();
        let expected: Vec<u64> = full.iter().take(3).map(|e| e.row_position).collect();
        assert_eq!(got, expected, "limited search must return the exact top-3");
    }
}
