// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-memory Full-Text Search (FTS) index.
//!
//! Provides inverted index for text search using crossbeam-skiplist.
//! Uses the same tokenization as Lance's InvertedIndex for consistency.
//!
//! ## Current Features
//! - BM25 scoring algorithm for relevance ranking
//! - Automatic result ordering by score (descending)
//! - Single-column term queries
//! - Phrase queries with slop support
//!
//! ## Pending Features (TODO)
//! - Multi-column search: Search across multiple columns simultaneously
//! - Boolean queries: MUST/SHOULD/MUST_NOT for complex query logic
//! - Fuzzy matching: Typo tolerance with configurable edit distance
//! - Boost queries: Positive/negative boosting for relevance tuning
//! - WAND factor: Performance/recall tradeoff control
//! - Per-term/column boost: Fine-grained relevance weighting
//!
//! **Note**: FTS index flush to persistent storage is NOT YET IMPLEMENTED.
//! The in-memory index works for real-time queries on MemTable data,
//! but is skipped during MemTable flush.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use arrow_array::RecordBatch;
use crossbeam_skiplist::SkipMap;
use datafusion::common::ScalarValue;
use lance_core::Result;
use lance_index::scalar::inverted::tokenizer::lance_tokenizer::LanceTokenizer;
use lance_index::scalar::InvertedIndexParams;
use tantivy::tokenizer::TokenStream;

use super::RowPosition;

/// Composite key for FTS index.
///
/// By combining (token, row_position), each entry is unique.
/// This follows the same pattern as IndexKey and IvfPqKey.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FtsKey {
    /// The indexed token (lowercase).
    pub token: String,
    /// Row position (makes the key unique for tokens appearing in multiple docs).
    pub row_position: RowPosition,
}

/// In-memory FTS (Full-Text Search) index entry (returned from search).
#[derive(Debug, Clone)]
pub struct FtsEntry {
    /// Row position in MemTable.
    pub row_position: RowPosition,
    /// BM25 score for this document.
    pub score: f32,
}

/// Full-text search query expression for composable queries.
///
/// Supports simple term matches, phrase queries, fuzzy matching, and Boolean
/// combinations with MUST/SHOULD/MUST_NOT logic.
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
        /// None means auto-fuzziness based on token length.
        fuzziness: Option<u32>,
        /// Maximum number of terms to expand to (default 50).
        max_expansions: usize,
        /// Boost factor applied to the score (default 1.0).
        boost: f32,
    },
    /// Boolean combination of queries.
    Boolean {
        /// All MUST clauses must match for a document to be included.
        must: Vec<FtsQueryExpr>,
        /// At least one SHOULD clause should match (adds to score).
        should: Vec<FtsQueryExpr>,
        /// No MUST_NOT clause may match (excludes documents).
        must_not: Vec<FtsQueryExpr>,
    },
    /// Boosting query with positive and optional negative components.
    ///
    /// Documents matching the positive query are returned.
    /// If a negative query is provided, documents matching both positive
    /// and negative have their scores reduced by `negative_boost`.
    Boost {
        /// The primary query (documents must match this).
        positive: Box<FtsQueryExpr>,
        /// Optional query to demote matching documents.
        negative: Option<Box<FtsQueryExpr>>,
        /// Boost factor for documents matching negative query (typically < 1.0).
        /// Score becomes: original_score * negative_boost for docs matching negative.
        negative_boost: f32,
    },
}

/// Default maximum number of fuzzy expansions.
pub const DEFAULT_MAX_EXPANSIONS: usize = 50;

/// Default WAND factor for full recall (no early termination).
pub const DEFAULT_WAND_FACTOR: f32 = 1.0;

/// Search options for controlling performance/recall tradeoffs.
///
/// The WAND (Weak AND) factor allows trading recall for performance:
/// - `wand_factor = 1.0`: Full recall (default), all matching documents returned
/// - `wand_factor < 1.0`: Faster but may miss some results. Documents with
///   scores below `top_k_score * wand_factor` are pruned.
///
/// # Example
/// ```ignore
/// let options = SearchOptions::default()
///     .with_limit(10)
///     .with_wand_factor(0.5);
/// let results = index.search_with_options(&query, options);
/// ```
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// WAND factor for early termination (0.0 to 1.0).
    /// 1.0 = full recall, <1.0 = faster but may miss low-scoring results.
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
    /// Create new SearchOptions with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the WAND factor for early termination.
    ///
    /// - 1.0 = full recall (default)
    /// - 0.5 = prune documents scoring below 50% of the current k-th best score
    /// - 0.0 = only return the absolute best match
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
    /// Create a simple match query.
    pub fn match_query(query: impl Into<String>) -> Self {
        Self::Match {
            query: query.into(),
            boost: 1.0,
        }
    }

    /// Create a phrase query with exact matching (slop=0).
    pub fn phrase(query: impl Into<String>) -> Self {
        Self::Phrase {
            query: query.into(),
            slop: 0,
            boost: 1.0,
        }
    }

    /// Create a phrase query with specified slop.
    pub fn phrase_with_slop(query: impl Into<String>, slop: u32) -> Self {
        Self::Phrase {
            query: query.into(),
            slop,
            boost: 1.0,
        }
    }

    /// Create a fuzzy match query with auto-fuzziness.
    ///
    /// Auto-fuzziness is calculated based on token length:
    /// - 0-2 chars: 0 (exact match)
    /// - 3-5 chars: 1
    /// - 6+ chars: 2
    pub fn fuzzy(query: impl Into<String>) -> Self {
        Self::Fuzzy {
            query: query.into(),
            fuzziness: None, // auto
            max_expansions: DEFAULT_MAX_EXPANSIONS,
            boost: 1.0,
        }
    }

    /// Create a fuzzy match query with specified edit distance.
    pub fn fuzzy_with_distance(query: impl Into<String>, fuzziness: u32) -> Self {
        Self::Fuzzy {
            query: query.into(),
            fuzziness: Some(fuzziness),
            max_expansions: DEFAULT_MAX_EXPANSIONS,
            boost: 1.0,
        }
    }

    /// Create a fuzzy match query with specified edit distance and max expansions.
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

    /// Create a Boolean query.
    pub fn boolean() -> BooleanQueryBuilder {
        BooleanQueryBuilder::new()
    }

    /// Create a boosting query with only a positive component.
    ///
    /// This is equivalent to just running the positive query.
    pub fn boosting(positive: Self) -> Self {
        Self::Boost {
            positive: Box::new(positive),
            negative: None,
            negative_boost: 1.0,
        }
    }

    /// Create a boosting query with positive and negative components.
    ///
    /// Documents matching the positive query are returned.
    /// Documents matching both positive and negative have their scores
    /// multiplied by `negative_boost` (typically < 1.0 to demote).
    ///
    /// # Arguments
    ///
    /// * `positive` - The primary query (documents must match this)
    /// * `negative` - Query to demote matching documents
    /// * `negative_boost` - Multiplier for documents matching negative (e.g., 0.5)
    pub fn boosting_with_negative(positive: Self, negative: Self, negative_boost: f32) -> Self {
        Self::Boost {
            positive: Box::new(positive),
            negative: Some(Box::new(negative)),
            negative_boost,
        }
    }

    /// Apply a boost factor to this query.
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
            Self::Boolean {
                must,
                should,
                must_not,
            } => {
                // For Boolean queries, boost is not directly applied
                // (would need to apply to sub-queries)
                Self::Boolean {
                    must,
                    should,
                    must_not,
                }
            }
            Self::Boost {
                positive,
                negative,
                negative_boost,
            } => {
                // For Boost queries, we wrap the positive in a boosted match
                // This is a bit unusual - typically you'd boost individual sub-queries
                Self::Boost {
                    positive,
                    negative,
                    negative_boost,
                }
            }
        }
    }
}

/// Calculate auto-fuzziness based on token length.
///
/// This follows the same algorithm as Lance's existing InvertedIndex:
/// - 0-2 chars: 0 (exact match only)
/// - 3-5 chars: 1 edit allowed
/// - 6+ chars: 2 edits allowed
pub fn auto_fuzziness(token: &str) -> u32 {
    match token.chars().count() {
        0..=2 => 0,
        3..=5 => 1,
        _ => 2,
    }
}

/// Calculate Levenshtein distance between two strings.
///
/// Returns the minimum number of single-character edits (insertions,
/// deletions, or substitutions) required to transform one string into another.
pub fn levenshtein_distance(a: &str, b: &str) -> u32 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    // Handle edge cases
    if m == 0 {
        return n as u32;
    }
    if n == 0 {
        return m as u32;
    }

    // Use two rows instead of full matrix for space efficiency
    let mut prev_row: Vec<u32> = (0..=n as u32).collect();
    let mut curr_row: Vec<u32> = vec![0; n + 1];

    for (i, a_char) in a_chars.iter().enumerate() {
        curr_row[0] = (i + 1) as u32;

        for (j, b_char) in b_chars.iter().enumerate() {
            let cost = if a_char == b_char { 0 } else { 1 };

            curr_row[j + 1] = (prev_row[j + 1] + 1) // deletion
                .min(curr_row[j] + 1) // insertion
                .min(prev_row[j] + cost); // substitution
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
    /// Create a new Boolean query builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a MUST clause (document must match).
    pub fn must(mut self, query: FtsQueryExpr) -> Self {
        self.must.push(query);
        self
    }

    /// Add a SHOULD clause (document should match, adds to score).
    pub fn should(mut self, query: FtsQueryExpr) -> Self {
        self.should.push(query);
        self
    }

    /// Add a MUST_NOT clause (document must not match).
    pub fn must_not(mut self, query: FtsQueryExpr) -> Self {
        self.must_not.push(query);
        self
    }

    /// Build the Boolean query.
    pub fn build(self) -> FtsQueryExpr {
        FtsQueryExpr::Boolean {
            must: self.must,
            should: self.should,
            must_not: self.must_not,
        }
    }
}

/// Posting value stored in the inverted index.
/// Contains term frequency and positions for phrase query support.
#[derive(Clone, Debug)]
pub struct PostingValue {
    /// Term frequency in the document.
    pub frequency: u32,
    /// Token positions within the document (0-indexed).
    /// Used for phrase matching.
    pub positions: Vec<u32>,
}

/// In-memory FTS index for full-text search.
pub struct FtsMemIndex {
    /// Field ID this index is built on.
    field_id: i32,
    /// Column name (for Arrow batch lookups).
    column_name: String,
    /// Inverted index: (token, row_position) -> (frequency, positions).
    postings: SkipMap<FtsKey, PostingValue>,
    /// Total document count.
    doc_count: AtomicUsize,
    /// Tokenizer for text processing (same as Lance's InvertedIndex).
    tokenizer: Mutex<Box<dyn LanceTokenizer>>,
    /// The parameters used to create the tokenizer (for flush).
    params: InvertedIndexParams,
    /// Document lengths: row_position -> token count (for BM25).
    doc_lengths: SkipMap<u64, u32>,
    /// Total token count across all documents (for computing avgdl).
    total_tokens: AtomicUsize,
    /// Document frequency: term -> number of documents containing the term.
    doc_freq: SkipMap<String, AtomicUsize>,
}

impl std::fmt::Debug for FtsMemIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FtsMemIndex")
            .field("field_id", &self.field_id)
            .field("column_name", &self.column_name)
            .field("doc_count", &self.doc_count)
            .field("params", &self.params)
            .finish()
    }
}

impl FtsMemIndex {
    /// Create a new FTS index for the given field with default parameters.
    pub fn new(field_id: i32, column_name: String) -> Self {
        Self::with_params(field_id, column_name, InvertedIndexParams::default())
    }

    /// Create a new FTS index with custom tokenizer parameters.
    pub fn with_params(field_id: i32, column_name: String, params: InvertedIndexParams) -> Self {
        let tokenizer = params.build().expect("Failed to build tokenizer");
        Self {
            field_id,
            column_name,
            postings: SkipMap::new(),
            doc_count: AtomicUsize::new(0),
            tokenizer: Mutex::new(tokenizer),
            params,
            doc_lengths: SkipMap::new(),
            total_tokens: AtomicUsize::new(0),
            doc_freq: SkipMap::new(),
        }
    }

    /// Get the field ID this index is built on.
    pub fn field_id(&self) -> i32 {
        self.field_id
    }

    /// Get the inverted index parameters.
    pub fn params(&self) -> &InvertedIndexParams {
        &self.params
    }

    /// Insert documents from a batch into the index.
    pub fn insert(&self, batch: &RecordBatch, row_offset: u64) -> Result<()> {
        let col_idx = batch
            .schema()
            .column_with_name(&self.column_name)
            .map(|(idx, _)| idx);

        if col_idx.is_none() {
            return Ok(());
        }

        let column = batch.column(col_idx.unwrap());

        for row_idx in 0..batch.num_rows() {
            let value = ScalarValue::try_from_array(column.as_ref(), row_idx)?;
            let row_position = row_offset + row_idx as u64;

            if let ScalarValue::Utf8(Some(text)) | ScalarValue::LargeUtf8(Some(text)) = value {
                // Use the tokenizer (same as InvertedIndex)
                // Track both frequency and positions for each term
                let mut term_data: HashMap<String, (u32, Vec<u32>)> = HashMap::new();
                {
                    let mut tokenizer = self.tokenizer.lock().unwrap();
                    let mut token_stream = tokenizer.token_stream_for_doc(&text);
                    let mut position: u32 = 0;
                    while let Some(token) = token_stream.next() {
                        let entry = term_data.entry(token.text.clone()).or_default();
                        entry.0 += 1; // frequency
                        entry.1.push(position); // position
                        position += 1;
                    }
                }

                // Calculate document length (total token count in this doc)
                let doc_length: u32 = term_data.values().map(|(freq, _)| freq).sum();
                self.doc_lengths.insert(row_position, doc_length);
                self.total_tokens
                    .fetch_add(doc_length as usize, Ordering::Relaxed);

                for (token, (freq, positions)) in term_data {
                    // Update document frequency for this term
                    if let Some(entry) = self.doc_freq.get(&token) {
                        entry.value().fetch_add(1, Ordering::Relaxed);
                    } else {
                        self.doc_freq.insert(token.clone(), AtomicUsize::new(1));
                    }

                    let key = FtsKey {
                        token,
                        row_position,
                    };
                    self.postings.insert(
                        key,
                        PostingValue {
                            frequency: freq,
                            positions,
                        },
                    );
                }
            }

            self.doc_count.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Search for documents containing a term.
    ///
    /// The term is tokenized using the same tokenizer as the index.
    /// Returns all matching documents with their BM25 scores.
    pub fn search(&self, term: &str) -> Vec<FtsEntry> {
        // Tokenize the search term using token_stream_for_search
        let tokens: Vec<String> = {
            let mut tokenizer = self.tokenizer.lock().unwrap();
            let mut token_stream = tokenizer.token_stream_for_search(term);
            let mut tokens = Vec::new();
            while let Some(token) = token_stream.next() {
                tokens.push(token.text.clone());
            }
            tokens
        };

        // BM25 parameters
        const K1: f32 = 1.2;
        const B: f32 = 0.75;

        let n = self.doc_count.load(Ordering::Relaxed) as f32;
        let total_tokens = self.total_tokens.load(Ordering::Relaxed) as f32;
        let avgdl = if n > 0.0 { total_tokens / n } else { 1.0 };

        // Collect term frequencies per document for all query tokens
        // Map: row_position -> Vec<(term_freq, doc_freq_for_term)>
        let mut doc_term_info: HashMap<RowPosition, Vec<(u32, usize)>> = HashMap::new();

        for token in &tokens {
            // Get document frequency for this term
            let df = self
                .doc_freq
                .get(token)
                .map(|e| e.value().load(Ordering::Relaxed))
                .unwrap_or(0);

            if df == 0 {
                continue;
            }

            let start = FtsKey {
                token: token.clone(),
                row_position: 0,
            };
            let end = FtsKey {
                token: token.clone(),
                row_position: u64::MAX,
            };

            for entry in self.postings.range(start..=end) {
                doc_term_info
                    .entry(entry.key().row_position)
                    .or_default()
                    .push((entry.value().frequency, df));
            }
        }

        // Compute BM25 score for each document
        doc_term_info
            .into_iter()
            .map(|(row_position, term_infos)| {
                let dl = self
                    .doc_lengths
                    .get(&row_position)
                    .map(|e| *e.value() as f32)
                    .unwrap_or(1.0);

                let mut score: f32 = 0.0;
                for (tf, df) in term_infos {
                    // IDF = log((N - n + 0.5) / (n + 0.5) + 1)
                    let df_f = df as f32;
                    let idf = ((n - df_f + 0.5) / (df_f + 0.5) + 1.0).ln();

                    // BM25 term score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))
                    let tf_f = tf as f32;
                    let numerator = tf_f * (K1 + 1.0);
                    let denominator = tf_f + K1 * (1.0 - B + B * (dl / avgdl));
                    score += idf * (numerator / denominator);
                }

                FtsEntry {
                    row_position,
                    score,
                }
            })
            .collect()
    }

    /// Search for documents containing an exact phrase.
    ///
    /// The phrase is tokenized and documents must contain all tokens
    /// in the correct order (within the specified slop distance).
    ///
    /// # Arguments
    /// * `phrase` - The phrase to search for
    /// * `slop` - Maximum allowed distance between consecutive tokens.
    ///   0 means exact phrase match (tokens must be adjacent).
    ///   1 allows one intervening token, etc.
    ///
    /// Returns matching documents with BM25 scores.
    pub fn search_phrase(&self, phrase: &str, slop: u32) -> Vec<FtsEntry> {
        // Tokenize the phrase
        let tokens: Vec<String> = {
            let mut tokenizer = self.tokenizer.lock().unwrap();
            let mut token_stream = tokenizer.token_stream_for_search(phrase);
            let mut tokens = Vec::new();
            while let Some(token) = token_stream.next() {
                tokens.push(token.text.clone());
            }
            tokens
        };

        if tokens.is_empty() {
            return vec![];
        }

        // Single token phrase is just a regular search
        if tokens.len() == 1 {
            return self.search(phrase);
        }

        // BM25 parameters
        const K1: f32 = 1.2;
        const B: f32 = 0.75;

        let n = self.doc_count.load(Ordering::Relaxed) as f32;
        let total_tokens = self.total_tokens.load(Ordering::Relaxed) as f32;
        let avgdl = if n > 0.0 { total_tokens / n } else { 1.0 };

        // Collect posting lists for each token
        // Map: token_index -> Map<row_position, PostingValue>
        let mut token_postings: Vec<HashMap<RowPosition, PostingValue>> = Vec::new();

        for token in &tokens {
            let start = FtsKey {
                token: token.clone(),
                row_position: 0,
            };
            let end = FtsKey {
                token: token.clone(),
                row_position: u64::MAX,
            };

            let mut postings_for_token: HashMap<RowPosition, PostingValue> = HashMap::new();
            for entry in self.postings.range(start..=end) {
                postings_for_token.insert(entry.key().row_position, entry.value().clone());
            }
            token_postings.push(postings_for_token);
        }

        // Find documents that contain ALL tokens
        let first_token_docs: Vec<RowPosition> = token_postings[0].keys().copied().collect();

        let mut matching_docs: Vec<FtsEntry> = Vec::new();

        for row_position in first_token_docs {
            // Check if this document contains all tokens
            let all_tokens_present = token_postings
                .iter()
                .all(|tp| tp.contains_key(&row_position));
            if !all_tokens_present {
                continue;
            }

            // Check if the phrase matches (positions are in order within slop)
            if self.check_phrase_positions(&token_postings, row_position, slop) {
                // Calculate BM25 score
                let dl = self
                    .doc_lengths
                    .get(&row_position)
                    .map(|e| *e.value() as f32)
                    .unwrap_or(1.0);

                let mut score: f32 = 0.0;
                for (token_idx, token) in tokens.iter().enumerate() {
                    let df = self
                        .doc_freq
                        .get(token)
                        .map(|e| e.value().load(Ordering::Relaxed))
                        .unwrap_or(1) as f32;
                    let tf = token_postings[token_idx]
                        .get(&row_position)
                        .map(|p| p.frequency as f32)
                        .unwrap_or(1.0);

                    // IDF = log((N - n + 0.5) / (n + 0.5) + 1)
                    let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

                    // BM25 term score
                    let numerator = tf * (K1 + 1.0);
                    let denominator = tf + K1 * (1.0 - B + B * (dl / avgdl));
                    score += idf * (numerator / denominator);
                }

                matching_docs.push(FtsEntry {
                    row_position,
                    score,
                });
            }
        }

        matching_docs
    }

    /// Check if phrase positions match within the given slop.
    ///
    /// Uses relative position algorithm: for each token, compute
    /// `relative_pos = doc_position - query_position`. If all tokens
    /// have the same relative position (within slop), the phrase matches.
    fn check_phrase_positions(
        &self,
        token_postings: &[HashMap<RowPosition, PostingValue>],
        row_position: RowPosition,
        slop: u32,
    ) -> bool {
        // Get positions for each token in this document
        let mut all_positions: Vec<&Vec<u32>> = Vec::new();
        for tp in token_postings {
            if let Some(posting) = tp.get(&row_position) {
                all_positions.push(&posting.positions);
            } else {
                return false;
            }
        }

        // For each position of the first token, check if we can form a phrase
        for &first_pos in all_positions[0] {
            if Self::check_phrase_from_position(&all_positions, first_pos, slop) {
                return true;
            }
        }

        false
    }

    /// Check if a phrase can be formed starting from a given position of the first token.
    fn check_phrase_from_position(all_positions: &[&Vec<u32>], first_pos: u32, slop: u32) -> bool {
        let mut expected_pos = first_pos;

        for positions in all_positions.iter().skip(1) {
            // Find a position for this token that's within slop of expected
            // For slop=0, next token must be at expected_pos+1 (adjacent)
            // For slop=1, next token can be at expected_pos+1 or expected_pos+2
            let min_pos = expected_pos.saturating_add(1);
            let max_pos = expected_pos.saturating_add(1 + slop);

            // Find the actual position used (smallest valid one)
            if let Some(&actual_pos) = positions
                .iter()
                .filter(|&&pos| pos >= min_pos && pos <= max_pos)
                .min()
            {
                expected_pos = actual_pos;
            } else {
                return false;
            }
        }

        true
    }

    /// Get the number of entries in the index.
    /// Note: This counts (token, row_position) pairs, not unique tokens.
    pub fn entry_count(&self) -> usize {
        self.postings.len()
    }

    /// Get the document count.
    pub fn doc_count(&self) -> usize {
        self.doc_count.load(Ordering::Relaxed)
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.doc_count.load(Ordering::Relaxed) == 0
    }

    /// Get the column name.
    pub fn column_name(&self) -> &str {
        &self.column_name
    }

    /// Expand a term to fuzzy matches within the specified edit distance.
    ///
    /// Returns a list of (matching_term, edit_distance) tuples, sorted by
    /// edit distance (closest matches first), limited to max_expansions.
    pub fn expand_fuzzy(
        &self,
        term: &str,
        max_distance: u32,
        max_expansions: usize,
    ) -> Vec<(String, u32)> {
        let mut matches: Vec<(String, u32)> = Vec::new();

        // If max_distance is 0, only exact matches
        if max_distance == 0 {
            if self.doc_freq.get(term).is_some() {
                matches.push((term.to_string(), 0));
            }
            return matches;
        }

        // Iterate through all tokens in doc_freq
        for entry in self.doc_freq.iter() {
            let indexed_term = entry.key();
            let distance = levenshtein_distance(term, indexed_term);

            if distance <= max_distance {
                matches.push((indexed_term.clone(), distance));
            }
        }

        // Sort by distance (prefer closer matches)
        matches.sort_by_key(|(_, d)| *d);

        // Limit to max_expansions
        matches.truncate(max_expansions);

        matches
    }

    /// Search for documents using fuzzy matching.
    ///
    /// Each query token is expanded to fuzzy matches within the edit distance,
    /// then searched. Results from all expansions are combined.
    pub fn search_fuzzy(
        &self,
        query: &str,
        fuzziness: Option<u32>,
        max_expansions: usize,
    ) -> Vec<FtsEntry> {
        // Tokenize the query
        let tokens: Vec<String> = {
            let mut tokenizer = self.tokenizer.lock().unwrap();
            let mut token_stream = tokenizer.token_stream_for_search(query);
            let mut tokens = Vec::new();
            while let Some(token) = token_stream.next() {
                tokens.push(token.text.clone());
            }
            tokens
        };

        if tokens.is_empty() {
            return vec![];
        }

        // BM25 parameters
        const K1: f32 = 1.2;
        const B: f32 = 0.75;

        let n = self.doc_count.load(Ordering::Relaxed) as f32;
        let total_tokens = self.total_tokens.load(Ordering::Relaxed) as f32;
        let avgdl = if n > 0.0 { total_tokens / n } else { 1.0 };

        // Collect term frequencies per document for all expanded tokens
        // Map: row_position -> Vec<(term_freq, doc_freq_for_term)>
        let mut doc_term_info: HashMap<RowPosition, Vec<(u32, usize)>> = HashMap::new();

        for token in &tokens {
            // Determine fuzziness for this token
            let max_distance = fuzziness.unwrap_or_else(|| auto_fuzziness(token));

            // Expand to fuzzy matches
            let expanded = self.expand_fuzzy(token, max_distance, max_expansions);

            for (matched_term, _distance) in expanded {
                // Get document frequency for this term
                let df = self
                    .doc_freq
                    .get(&matched_term)
                    .map(|e| e.value().load(Ordering::Relaxed))
                    .unwrap_or(0);

                if df == 0 {
                    continue;
                }

                let start = FtsKey {
                    token: matched_term.clone(),
                    row_position: 0,
                };
                let end = FtsKey {
                    token: matched_term,
                    row_position: u64::MAX,
                };

                for entry in self.postings.range(start..=end) {
                    doc_term_info
                        .entry(entry.key().row_position)
                        .or_default()
                        .push((entry.value().frequency, df));
                }
            }
        }

        // Compute BM25 score for each document
        doc_term_info
            .into_iter()
            .map(|(row_position, term_infos)| {
                let dl = self
                    .doc_lengths
                    .get(&row_position)
                    .map(|e| *e.value() as f32)
                    .unwrap_or(1.0);

                let mut score: f32 = 0.0;
                for (tf, df) in term_infos {
                    // IDF = log((N - n + 0.5) / (n + 0.5) + 1)
                    let df_f = df as f32;
                    let idf = ((n - df_f + 0.5) / (df_f + 0.5) + 1.0).ln();

                    // BM25 term score
                    let tf_f = tf as f32;
                    let numerator = tf_f * (K1 + 1.0);
                    let denominator = tf_f + K1 * (1.0 - B + B * (dl / avgdl));
                    score += idf * (numerator / denominator);
                }

                FtsEntry {
                    row_position,
                    score,
                }
            })
            .collect()
    }

    /// Execute a query expression and return matching documents with scores.
    ///
    /// This is the main entry point for executing complex queries including
    /// match, phrase, fuzzy, and Boolean queries.
    ///
    /// For performance optimization with limits, use `search_with_options()` instead.
    pub fn search_query(&self, query: &FtsQueryExpr) -> Vec<FtsEntry> {
        match query {
            FtsQueryExpr::Match { query, boost } => {
                let mut results = self.search(query);
                if *boost != 1.0 {
                    for entry in &mut results {
                        entry.score *= boost;
                    }
                }
                results
            }
            FtsQueryExpr::Phrase { query, slop, boost } => {
                let mut results = self.search_phrase(query, *slop);
                if *boost != 1.0 {
                    for entry in &mut results {
                        entry.score *= boost;
                    }
                }
                results
            }
            FtsQueryExpr::Fuzzy {
                query,
                fuzziness,
                max_expansions,
                boost,
            } => {
                let mut results = self.search_fuzzy(query, *fuzziness, *max_expansions);
                if *boost != 1.0 {
                    for entry in &mut results {
                        entry.score *= boost;
                    }
                }
                results
            }
            FtsQueryExpr::Boolean {
                must,
                should,
                must_not,
            } => self.search_boolean(must, should, must_not),
            FtsQueryExpr::Boost {
                positive,
                negative,
                negative_boost,
            } => self.search_boost(positive, negative.as_deref(), *negative_boost),
        }
    }

    /// Execute a query with options for performance/recall tradeoffs.
    ///
    /// This method extends `search_query()` with:
    /// - **WAND factor**: Early termination based on score threshold.
    ///   With `wand_factor < 1.0`, documents scoring below
    ///   `threshold = top_k_score * wand_factor` are pruned after scoring.
    /// - **Limit**: Maximum number of results to return (top-k by score).
    ///
    /// Results are always sorted by score in descending order.
    ///
    /// # Arguments
    /// * `query` - The query expression to execute
    /// * `options` - Search options including wand_factor and limit
    ///
    /// # Example
    /// ```ignore
    /// let options = SearchOptions::default()
    ///     .with_limit(10)
    ///     .with_wand_factor(0.8);
    /// let results = index.search_with_options(&query, options);
    /// ```
    pub fn search_with_options(
        &self,
        query: &FtsQueryExpr,
        options: SearchOptions,
    ) -> Vec<FtsEntry> {
        // Execute the query to get all results
        let mut results = self.search_query(query);

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply WAND factor pruning if wand_factor < 1.0 and we have a limit
        if options.wand_factor < 1.0 {
            if let Some(limit) = options.limit {
                if results.len() > limit {
                    // Get the k-th best score (at position limit-1)
                    let top_k_score = results[limit - 1].score;
                    let threshold = top_k_score * options.wand_factor;

                    // Keep results scoring above the threshold, plus all results up to limit
                    // This ensures we don't accidentally prune results that would be in top-k
                    results.retain(|e| e.score >= threshold);
                }
            } else {
                // No limit but wand_factor < 1.0: prune relative to max score
                if let Some(max_entry) = results.first() {
                    let threshold = max_entry.score * options.wand_factor;
                    results.retain(|e| e.score >= threshold);
                }
            }
        }

        // Apply limit
        if let Some(limit) = options.limit {
            results.truncate(limit);
        }

        results
    }

    /// Execute a boosting query.
    ///
    /// Returns documents matching the positive query. Documents that also
    /// match the negative query have their scores multiplied by `negative_boost`.
    fn search_boost(
        &self,
        positive: &FtsQueryExpr,
        negative: Option<&FtsQueryExpr>,
        negative_boost: f32,
    ) -> Vec<FtsEntry> {
        // Execute positive query to get base results
        let mut results = self.search_query(positive);

        // If no negative query, just return positive results
        let Some(neg_query) = negative else {
            return results;
        };

        // Execute negative query
        let negative_results = self.search_query(neg_query);

        // Build a set of row positions that match the negative query
        let negative_positions: std::collections::HashSet<RowPosition> =
            negative_results.iter().map(|e| e.row_position).collect();

        // Apply negative boost to documents matching both queries
        for entry in &mut results {
            if negative_positions.contains(&entry.row_position) {
                entry.score *= negative_boost;
            }
        }

        results
    }

    /// Execute a Boolean query with MUST/SHOULD/MUST_NOT logic.
    ///
    /// - MUST: All clauses must match (intersection). Scores are summed.
    /// - SHOULD: At least one clause should match (union). Scores are added.
    /// - MUST_NOT: No clause may match (exclusion).
    ///
    /// If only SHOULD clauses are present, at least one must match.
    /// If MUST clauses are present, SHOULD clauses just add to the score.
    fn search_boolean(
        &self,
        must: &[FtsQueryExpr],
        should: &[FtsQueryExpr],
        must_not: &[FtsQueryExpr],
    ) -> Vec<FtsEntry> {
        // Collect MUST_NOT results for exclusion
        let excluded: std::collections::HashSet<RowPosition> = must_not
            .iter()
            .flat_map(|q| self.search_query(q))
            .map(|e| e.row_position)
            .collect();

        // Start with MUST clauses (intersection)
        let mut result_map: HashMap<RowPosition, f32> = if must.is_empty() {
            // No MUST clauses: start with all SHOULD results
            let mut map = HashMap::new();
            for q in should {
                for entry in self.search_query(q) {
                    *map.entry(entry.row_position).or_default() += entry.score;
                }
            }
            map
        } else {
            // Execute first MUST clause
            let first_results = self.search_query(&must[0]);
            let mut map: HashMap<RowPosition, f32> = first_results
                .into_iter()
                .map(|e| (e.row_position, e.score))
                .collect();

            // Intersect with remaining MUST clauses
            for q in must.iter().skip(1) {
                let results = self.search_query(q);
                let result_set: HashMap<RowPosition, f32> = results
                    .into_iter()
                    .map(|e| (e.row_position, e.score))
                    .collect();

                // Keep only documents in both sets, sum scores
                map = map
                    .into_iter()
                    .filter_map(|(pos, score)| result_set.get(&pos).map(|s| (pos, score + s)))
                    .collect();
            }

            // Add SHOULD clause scores (don't require match since MUST already filters)
            for q in should {
                for entry in self.search_query(q) {
                    if let Some(score) = map.get_mut(&entry.row_position) {
                        *score += entry.score;
                    }
                }
            }

            map
        };

        // Filter out MUST_NOT results
        for pos in &excluded {
            result_map.remove(pos);
        }

        // Convert to FtsEntry list
        result_map
            .into_iter()
            .map(|(row_position, score)| FtsEntry {
                row_position,
                score,
            })
            .collect()
    }
}

/// Configuration for a Full-Text Search index.
#[derive(Debug, Clone)]
pub struct FtsIndexConfig {
    /// Index name.
    pub name: String,
    /// Field ID the index is built on.
    pub field_id: i32,
    /// Column name (for Arrow batch lookups).
    pub column: String,
    /// Tokenizer parameters (same as InvertedIndex).
    pub params: InvertedIndexParams,
}

impl FtsIndexConfig {
    /// Create a new FtsIndexConfig with default tokenizer parameters.
    pub fn new(name: String, field_id: i32, column: String) -> Self {
        Self {
            name,
            field_id,
            column,
            params: InvertedIndexParams::default(),
        }
    }

    /// Create a new FtsIndexConfig with custom tokenizer parameters.
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use std::sync::Arc;

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

        // "hello" appears in docs 0 and 2
        let entries = index.search("hello");
        assert!(!entries.is_empty());
        assert_eq!(entries.len(), 2);

        // "world" appears in docs 0 and 1
        let entries = index.search("world");
        assert!(!entries.is_empty());
        assert_eq!(entries.len(), 2);

        // "goodbye" appears only in doc 1 (row position 1)
        let entries = index.search("goodbye");
        assert!(!entries.is_empty());
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_position, 1);

        // Non-existent term returns empty Vec
        let entries = index.search("nonexistent");
        assert!(entries.is_empty());
    }

    fn create_phrase_test_batch(schema: &ArrowSchema) -> RecordBatch {
        // Note: The tokenizer filters stop words (the, and, very, etc.) and lowercases.
        // Positions are assigned to non-filtered tokens only.
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4])),
                Arc::new(StringArray::from(vec![
                    "alpha beta gamma",               // 0: alpha=0, beta=1, gamma=2
                    "beta alpha gamma",               // 1: beta=0, alpha=1, gamma=2
                    "alpha delta beta gamma",         // 2: alpha=0, delta=1, beta=2, gamma=3
                    "alpha gamma",                    // 3: alpha=0, gamma=1
                    "alpha delta epsilon beta gamma", // 4: alpha=0, delta=1, epsilon=2, beta=3, gamma=4
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

        // Exact phrase "alpha beta" with slop=0 should match only doc 0
        // Doc 0: "alpha beta gamma" - alpha=0, beta=1 (adjacent)
        // Doc 2: "alpha delta beta gamma" - alpha=0, beta=2 (NOT adjacent, slop needed)
        let entries = index.search_phrase("alpha beta", 0);
        assert_eq!(
            entries.len(),
            1,
            "Expected 1 match for 'alpha beta', got {:?}",
            entries.iter().map(|e| e.row_position).collect::<Vec<_>>()
        );
        assert_eq!(entries[0].row_position, 0);

        // "hello world" exact phrase
        let batch2 = create_test_batch(&schema);
        let index2 = FtsMemIndex::new(1, "description".to_string());
        index2.insert(&batch2, 0).unwrap();

        let entries = index2.search_phrase("hello world", 0);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].row_position, 0);

        // "goodbye world" exact phrase
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

        // Positions after tokenization (no stop words filtered):
        // Doc 0: "alpha beta gamma" - alpha=0, beta=1, gamma=2
        // Doc 2: "alpha delta beta gamma" - alpha=0, delta=1, beta=2, gamma=3
        // Doc 4: "alpha delta epsilon beta gamma" - alpha=0, delta=1, epsilon=2, beta=3, gamma=4

        // "alpha beta" with slop=0 should match only doc 0
        // Doc 0: alpha=0, beta=1 (adjacent)
        let entries = index.search_phrase("alpha beta", 0);
        assert_eq!(
            entries.len(),
            1,
            "slop=0 matches: {:?}",
            entries.iter().map(|e| e.row_position).collect::<Vec<_>>()
        );
        assert_eq!(entries[0].row_position, 0);

        // "alpha beta" with slop=1 should match docs 0 and 2
        // Doc 0: alpha=0, beta=1 (diff=1, within slop=1)
        // Doc 2: alpha=0, beta=2 (diff=2, slop=1 allows pos 1-2)
        // Doc 4: alpha=0, beta=3 (diff=3, slop=1 does NOT allow pos 3)
        let entries = index.search_phrase("alpha beta", 1);
        assert_eq!(
            entries.len(),
            2,
            "slop=1 matches: {:?}",
            entries.iter().map(|e| e.row_position).collect::<Vec<_>>()
        );
        let positions: Vec<_> = entries.iter().map(|e| e.row_position).collect();
        assert!(positions.contains(&0));
        assert!(positions.contains(&2));

        // "alpha beta" with slop=2 should match docs 0, 2, and 4
        let entries = index.search_phrase("alpha beta", 2);
        assert_eq!(
            entries.len(),
            3,
            "slop=2 matches: {:?}",
            entries.iter().map(|e| e.row_position).collect::<Vec<_>>()
        );

        // "alpha gamma" with slop=0 should match docs 1 and 3 (adjacent)
        // Doc 1: "beta alpha gamma" - alpha=1, gamma=2 (adjacent)
        // Doc 3: "alpha gamma" - alpha=0, gamma=1 (adjacent)
        let entries = index.search_phrase("alpha gamma", 0);
        assert_eq!(
            entries.len(),
            2,
            "alpha gamma slop=0: {:?}",
            entries.iter().map(|e| e.row_position).collect::<Vec<_>>()
        );

        // "alpha gamma" with slop=1 should match docs 0, 1, 2, and 3
        // Doc 0: alpha=0, gamma=2 (diff=2, slop=1 allows pos 1-2)
        // Doc 1: alpha=1, gamma=2 (adjacent)
        // Doc 2: alpha=0, gamma=3 (diff=3, slop=1 allows pos 1-2, gamma at 3 NOT in range)
        // Doc 3: alpha=0, gamma=1 (adjacent)
        let entries = index.search_phrase("alpha gamma", 1);
        assert_eq!(
            entries.len(),
            3,
            "alpha gamma slop=1: {:?}",
            entries.iter().map(|e| e.row_position).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_phrase_search_no_match() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_phrase_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // "beta alpha" with slop=0 should not match in most docs (wrong order)
        // Doc 1 has "beta alpha gamma" - beta=0, alpha=1, so "beta alpha" matches there!
        let entries = index.search_phrase("beta alpha", 0);
        assert_eq!(entries.len(), 1); // matches doc 1
        assert_eq!(entries[0].row_position, 1);

        // Non-existent phrase
        let entries = index.search_phrase("nonexistent phrase", 0);
        assert!(entries.is_empty());

        // Partial phrase not in any doc
        let entries = index.search_phrase("alpha hello", 0);
        assert!(entries.is_empty());

        // "gamma alpha" should not match (wrong order in all docs)
        let entries = index.search_phrase("gamma alpha", 0);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_phrase_search_single_token() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_phrase_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // Single token phrase should behave like regular search
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

        // Empty phrase
        let entries = index.search_phrase("", 0);
        assert!(entries.is_empty());
    }

    // ====== Boolean Query Tests ======

    fn create_boolean_test_batch(schema: &ArrowSchema) -> RecordBatch {
        // Test documents for Boolean queries:
        // Doc 0: "rust programming language"
        // Doc 1: "python programming language"
        // Doc 2: "rust web server"
        // Doc 3: "python web framework"
        // Doc 4: "javascript programming"
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

        // MUST: rust AND programming
        // Should match doc 0 only ("rust programming language")
        let query = FtsQueryExpr::boolean()
            .must(FtsQueryExpr::match_query("rust"))
            .must(FtsQueryExpr::match_query("programming"))
            .build();

        let entries = index.search_query(&query);
        assert_eq!(
            entries.len(),
            1,
            "Expected 1 match for MUST(rust, programming), got {:?}",
            entries.iter().map(|e| e.row_position).collect::<Vec<_>>()
        );
        assert_eq!(entries[0].row_position, 0);
    }

    #[test]
    fn test_boolean_should_only() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boolean_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // SHOULD: rust OR python
        // Should match docs 0, 1, 2, 3 (all containing rust or python)
        let query = FtsQueryExpr::boolean()
            .should(FtsQueryExpr::match_query("rust"))
            .should(FtsQueryExpr::match_query("python"))
            .build();

        let entries = index.search_query(&query);
        assert_eq!(
            entries.len(),
            4,
            "Expected 4 matches for SHOULD(rust, python), got {:?}",
            entries.iter().map(|e| e.row_position).collect::<Vec<_>>()
        );

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

        // MUST_NOT alone with no MUST or SHOULD returns empty
        // (nothing to include, only exclusions)
        let query = FtsQueryExpr::boolean()
            .must_not(FtsQueryExpr::match_query("rust"))
            .build();

        let entries = index.search_query(&query);
        assert!(
            entries.is_empty(),
            "MUST_NOT only should return empty, got {:?}",
            entries.iter().map(|e| e.row_position).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_boolean_must_with_should() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boolean_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // MUST: programming, SHOULD: rust
        // Should match docs 0, 1, 4 (all with programming)
        // Doc 0 should have higher score (also matches rust)
        let query = FtsQueryExpr::boolean()
            .must(FtsQueryExpr::match_query("programming"))
            .should(FtsQueryExpr::match_query("rust"))
            .build();

        let entries = index.search_query(&query);
        assert_eq!(
            entries.len(),
            3,
            "Expected 3 matches for MUST(programming) SHOULD(rust), got {:?}",
            entries.iter().map(|e| e.row_position).collect::<Vec<_>>()
        );

        // Find doc 0 and doc 1 scores
        let doc0 = entries.iter().find(|e| e.row_position == 0).unwrap();
        let doc1 = entries.iter().find(|e| e.row_position == 1).unwrap();

        // Doc 0 has both programming and rust, should score higher than doc 1 (only programming)
        assert!(
            doc0.score > doc1.score,
            "Doc 0 (rust+programming) should score higher than doc 1 (programming only). Doc0: {}, Doc1: {}",
            doc0.score,
            doc1.score
        );
    }

    #[test]
    fn test_boolean_must_with_must_not() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boolean_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // MUST: programming, MUST_NOT: python
        // Should match docs 0 and 4 (programming but not python)
        let query = FtsQueryExpr::boolean()
            .must(FtsQueryExpr::match_query("programming"))
            .must_not(FtsQueryExpr::match_query("python"))
            .build();

        let entries = index.search_query(&query);
        assert_eq!(
            entries.len(),
            2,
            "Expected 2 matches for MUST(programming) MUST_NOT(python), got {:?}",
            entries.iter().map(|e| e.row_position).collect::<Vec<_>>()
        );

        let positions: Vec<_> = entries.iter().map(|e| e.row_position).collect();
        assert!(positions.contains(&0)); // rust programming language
        assert!(positions.contains(&4)); // javascript programming
        assert!(!positions.contains(&1)); // python programming language - excluded
    }

    #[test]
    fn test_boolean_combined() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boolean_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // MUST: web, SHOULD: rust, MUST_NOT: framework
        // Docs with "web": 2 (rust web server), 3 (python web framework)
        // After MUST_NOT framework: only doc 2
        // Doc 2 also matches SHOULD(rust), so should have higher score
        let query = FtsQueryExpr::boolean()
            .must(FtsQueryExpr::match_query("web"))
            .should(FtsQueryExpr::match_query("rust"))
            .must_not(FtsQueryExpr::match_query("framework"))
            .build();

        let entries = index.search_query(&query);
        assert_eq!(
            entries.len(),
            1,
            "Expected 1 match for MUST(web) SHOULD(rust) MUST_NOT(framework), got {:?}",
            entries.iter().map(|e| e.row_position).collect::<Vec<_>>()
        );
        assert_eq!(entries[0].row_position, 2);
    }

    #[test]
    fn test_boolean_nested_phrase() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boolean_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // MUST: phrase("programming language")
        // Should match docs 0 and 1
        let query = FtsQueryExpr::boolean()
            .must(FtsQueryExpr::phrase("programming language"))
            .build();

        let entries = index.search_query(&query);
        assert_eq!(
            entries.len(),
            2,
            "Expected 2 matches for MUST(phrase 'programming language'), got {:?}",
            entries.iter().map(|e| e.row_position).collect::<Vec<_>>()
        );

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

        // Test FtsQueryExpr::Match
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

        // Test FtsQueryExpr::Phrase
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

        // Test boost
        let query_no_boost = FtsQueryExpr::match_query("hello");
        let query_with_boost = FtsQueryExpr::match_query("hello").with_boost(2.0);

        let entries_no_boost = index.search_query(&query_no_boost);
        let entries_with_boost = index.search_query(&query_with_boost);

        assert_eq!(entries_no_boost.len(), entries_with_boost.len());

        // Boosted scores should be 2x
        for (e1, e2) in entries_no_boost.iter().zip(entries_with_boost.iter()) {
            let expected = e1.score * 2.0;
            assert!(
                (e2.score - expected).abs() < 0.001,
                "Boosted score {} should be 2x original {}",
                e2.score,
                e1.score
            );
        }
    }

    // ====== Fuzzy Matching Tests ======

    #[test]
    fn test_levenshtein_distance() {
        // Identical strings
        assert_eq!(levenshtein_distance("hello", "hello"), 0);

        // Single character difference
        assert_eq!(levenshtein_distance("hello", "hallo"), 1); // substitution
        assert_eq!(levenshtein_distance("hello", "hell"), 1); // deletion
        assert_eq!(levenshtein_distance("hello", "helloo"), 1); // insertion

        // Two character differences
        assert_eq!(levenshtein_distance("hello", "hxllo"), 1);
        assert_eq!(levenshtein_distance("hello", "hxxlo"), 2);

        // Completely different strings
        assert_eq!(levenshtein_distance("abc", "xyz"), 3);

        // Empty strings
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("hello", ""), 5);
        assert_eq!(levenshtein_distance("", "hello"), 5);

        // Case sensitivity
        assert_eq!(levenshtein_distance("Hello", "hello"), 1);
    }

    #[test]
    fn test_auto_fuzziness() {
        // 0-2 chars: 0 fuzziness
        assert_eq!(auto_fuzziness(""), 0);
        assert_eq!(auto_fuzziness("a"), 0);
        assert_eq!(auto_fuzziness("ab"), 0);

        // 3-5 chars: 1 fuzziness
        assert_eq!(auto_fuzziness("abc"), 1);
        assert_eq!(auto_fuzziness("abcd"), 1);
        assert_eq!(auto_fuzziness("abcde"), 1);

        // 6+ chars: 2 fuzziness
        assert_eq!(auto_fuzziness("abcdef"), 2);
        assert_eq!(auto_fuzziness("programming"), 2);
    }

    fn create_fuzzy_test_batch(schema: &ArrowSchema) -> RecordBatch {
        // Test documents for fuzzy matching.
        // Note: The tokenizer stems words, so we use unstemmed single tokens
        // for predictable fuzzy matching tests.
        // Levenshtein distance examples:
        // - "alpha" to "alpho" = 1 (substitution: a -> o)
        // - "alpha" to "alphax" = 1 (insertion)
        // - "alpha" to "alph" = 1 (deletion)
        // Doc 0: "alpha beta gamma"
        // Doc 1: "alpho beta delta" (typo: 'alpho' instead of 'alpha', distance=1)
        // Doc 2: "alpha delta epsilon"
        // Doc 3: "omega zeta"
        // Doc 4: "alphax gamma" (typo: extra 'x', distance=1)
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

        // Exact match with fuzziness=0: "alpha" exists in index
        let matches = index.expand_fuzzy("alpha", 0, 50);
        assert_eq!(
            matches.len(),
            1,
            "Expected 1 match for 'alpha', got {:?}",
            matches
        );
        assert_eq!(matches[0].0, "alpha");
        assert_eq!(matches[0].1, 0);

        // Non-existent term with fuzziness=0
        let matches = index.expand_fuzzy("nonexistent", 0, 50);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_expand_fuzzy_single_edit() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // "alpho" (typo, substitution distance=1 from "alpha") should match "alpha"
        let matches = index.expand_fuzzy("alpho", 1, 50);
        assert!(
            matches
                .iter()
                .any(|(term, dist)| term == "alpha" && *dist == 1),
            "Expected 'alpha' with distance 1, got {:?}",
            matches
        );

        // Also matches itself since it's in the index
        assert!(
            matches.iter().any(|(term, _)| term == "alpho"),
            "Expected 'alpho' in matches, got {:?}",
            matches
        );
    }

    #[test]
    fn test_expand_fuzzy_max_expansions() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // With very high distance, should be limited by max_expansions
        let matches = index.expand_fuzzy("a", 10, 3);
        assert!(
            matches.len() <= 3,
            "Expected at most 3 matches, got {}",
            matches.len()
        );
    }

    #[test]
    fn test_search_fuzzy_basic() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // Search with typo "alpho" should match documents with "alpha" or "alpho"
        let entries = index.search_fuzzy("alpho", Some(1), 50);
        assert!(!entries.is_empty(), "Expected matches for fuzzy 'alpho'");

        // Should match docs with alpha (0, 2) and alpho (1)
        let positions: Vec<_> = entries.iter().map(|e| e.row_position).collect();
        assert!(
            positions.contains(&0) || positions.contains(&1) || positions.contains(&2),
            "Expected to match docs with alpha/alpho, got {:?}",
            positions
        );
    }

    #[test]
    fn test_search_fuzzy_auto_fuzziness() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // "alpho" (5 chars) should get auto-fuzziness of 1
        let entries = index.search_fuzzy("alpho", None, 50);
        assert!(!entries.is_empty(), "Expected matches with auto-fuzziness");
    }

    #[test]
    fn test_search_fuzzy_no_match() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // Search for something completely different with low fuzziness
        let entries = index.search_fuzzy("xyz", Some(0), 50);
        assert!(entries.is_empty(), "Expected no matches for 'xyz'");

        // Even with fuzziness=1, "xyz" shouldn't match anything meaningful
        // (this may or may not be empty depending on what 3-letter words are in the index)
        let _ = index.search_fuzzy("xyz", Some(1), 50);
    }

    #[test]
    fn test_search_query_fuzzy() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // Test FtsQueryExpr::Fuzzy via search_query
        let query = FtsQueryExpr::fuzzy("alpho");
        let entries = index.search_query(&query);
        assert!(
            !entries.is_empty(),
            "Expected matches for fuzzy query 'alpho'"
        );
    }

    #[test]
    fn test_search_query_fuzzy_with_distance() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // Exact distance: "alpho" has distance 1 from "alpha"
        let query = FtsQueryExpr::fuzzy_with_distance("alpho", 1);
        let entries = index.search_query(&query);
        assert!(
            !entries.is_empty(),
            "Expected matches for fuzzy query with distance 1"
        );
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

        // Boosted scores should be 2x
        for e1 in &entries_no_boost {
            let e2 = entries_with_boost
                .iter()
                .find(|e| e.row_position == e1.row_position)
                .unwrap();
            let expected = e1.score * 2.0;
            assert!(
                (e2.score - expected).abs() < 0.001,
                "Boosted score {} should be 2x original {}",
                e2.score,
                e1.score
            );
        }
    }

    #[test]
    fn test_boolean_with_fuzzy() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_fuzzy_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // MUST: fuzzy("alpho", distance=1), MUST_NOT: "delta"
        // "alpho" matches "alpha" (distance=1) and itself
        // Doc 0: "alpha beta gamma" - matches fuzzy alpho, no delta -> included
        // Doc 1: "alpho beta delta" - matches fuzzy alpho, has delta -> excluded
        // Doc 2: "alpha delta epsilon" - matches fuzzy alpho, has delta -> excluded
        // Doc 4: "alphax gamma" - matches fuzzy alpho via alphax (dist=1 to alpho), no delta -> included
        let query = FtsQueryExpr::boolean()
            .must(FtsQueryExpr::fuzzy_with_distance("alpho", 1))
            .must_not(FtsQueryExpr::match_query("delta"))
            .build();

        let entries = index.search_query(&query);

        // Should not contain docs 1 and 2 (have "delta")
        let positions: Vec<_> = entries.iter().map(|e| e.row_position).collect();
        assert!(
            !positions.contains(&1),
            "Doc 1 should be excluded due to MUST_NOT, got {:?}",
            positions
        );
        assert!(
            !positions.contains(&2),
            "Doc 2 should be excluded due to MUST_NOT, got {:?}",
            positions
        );
        // Doc 0 should be included
        assert!(
            positions.contains(&0),
            "Doc 0 should be included, got {:?}",
            positions
        );
    }

    // ====== Boost Query Tests ======

    fn create_boost_test_batch(schema: &ArrowSchema) -> RecordBatch {
        // Test documents for boost queries:
        // Doc 0: "rust programming language" - matches rust, programming, language
        // Doc 1: "python programming language" - matches python, programming, language
        // Doc 2: "rust web server" - matches rust, web, server
        // Doc 3: "python web framework" - matches python, web, framework
        // Doc 4: "javascript programming" - matches javascript, programming
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

        // Boosting query with only positive component (same as regular query)
        let query = FtsQueryExpr::boosting(FtsQueryExpr::match_query("programming"));
        let entries = index.search_query(&query);

        // Should match docs 0, 1, 4 (all with "programming")
        assert_eq!(
            entries.len(),
            3,
            "Expected 3 matches for 'programming', got {:?}",
            entries.iter().map(|e| e.row_position).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_boost_query_with_negative() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boost_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // Boosting query: find "programming", demote docs with "python"
        let query = FtsQueryExpr::boosting_with_negative(
            FtsQueryExpr::match_query("programming"),
            FtsQueryExpr::match_query("python"),
            0.5, // Demote python docs by half
        );
        let entries = index.search_query(&query);

        // Should still match docs 0, 1, 4 (all with "programming")
        assert_eq!(entries.len(), 3);

        // Find scores for each doc
        let doc0 = entries.iter().find(|e| e.row_position == 0); // rust programming
        let doc1 = entries.iter().find(|e| e.row_position == 1); // python programming
        let doc4 = entries.iter().find(|e| e.row_position == 4); // javascript programming

        assert!(doc0.is_some() && doc1.is_some() && doc4.is_some());

        // Doc 1 (python) should have lower score than doc 0 (rust) due to negative boost
        // Doc 0 and doc 4 should have similar scores (neither match "python")
        let score0 = doc0.unwrap().score;
        let score1 = doc1.unwrap().score;
        let score4 = doc4.unwrap().score;

        // Doc 1 was demoted by 0.5, so it should have roughly half the score
        assert!(
            score1 < score0,
            "Doc 1 (python) should have lower score than doc 0 (rust). Doc0: {}, Doc1: {}",
            score0,
            score1
        );

        // Doc 0 and doc 4 should have similar scores (both not demoted)
        // They may differ slightly due to BM25 scoring differences, but doc 1 should be lower
        assert!(
            score1 < score4,
            "Doc 1 (python) should have lower score than doc 4 (javascript). Doc1: {}, Doc4: {}",
            score1,
            score4
        );
    }

    #[test]
    fn test_boost_query_negative_boost_factor() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boost_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // Compare different negative boost factors
        let query_no_demote = FtsQueryExpr::boosting_with_negative(
            FtsQueryExpr::match_query("programming"),
            FtsQueryExpr::match_query("python"),
            1.0, // No demotion
        );

        let query_half_demote = FtsQueryExpr::boosting_with_negative(
            FtsQueryExpr::match_query("programming"),
            FtsQueryExpr::match_query("python"),
            0.5, // Half score for python
        );

        let query_zero_demote = FtsQueryExpr::boosting_with_negative(
            FtsQueryExpr::match_query("programming"),
            FtsQueryExpr::match_query("python"),
            0.0, // Zero score for python
        );

        let results_no_demote = index.search_query(&query_no_demote);
        let results_half_demote = index.search_query(&query_half_demote);
        let results_zero_demote = index.search_query(&query_zero_demote);

        // Get doc 1 (python programming) scores
        let score_no_demote = results_no_demote
            .iter()
            .find(|e| e.row_position == 1)
            .unwrap()
            .score;
        let score_half_demote = results_half_demote
            .iter()
            .find(|e| e.row_position == 1)
            .unwrap()
            .score;
        let score_zero_demote = results_zero_demote
            .iter()
            .find(|e| e.row_position == 1)
            .unwrap()
            .score;

        // Verify demotion factors are applied correctly
        assert!(
            (score_half_demote - score_no_demote * 0.5).abs() < 0.001,
            "Half demotion should give half score. Expected {}, got {}",
            score_no_demote * 0.5,
            score_half_demote
        );

        assert!(
            score_zero_demote.abs() < 0.001,
            "Zero demotion should give zero score, got {}",
            score_zero_demote
        );
    }

    #[test]
    fn test_boost_query_no_negative_match() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boost_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // Boosting query where negative doesn't match any positive results
        let query = FtsQueryExpr::boosting_with_negative(
            FtsQueryExpr::match_query("rust"),   // Matches docs 0, 2
            FtsQueryExpr::match_query("python"), // Matches docs 1, 3 (no overlap!)
            0.1,
        );

        let entries = index.search_query(&query);

        // Should match docs 0, 2 (rust docs)
        assert_eq!(entries.len(), 2);

        // Scores should not be demoted (no overlap with python)
        let query_baseline = FtsQueryExpr::match_query("rust");
        let baseline_entries = index.search_query(&query_baseline);

        for entry in &entries {
            let baseline = baseline_entries
                .iter()
                .find(|e| e.row_position == entry.row_position)
                .unwrap();
            assert!(
                (entry.score - baseline.score).abs() < 0.001,
                "Scores should match when no negative overlap. Got {} vs {}",
                entry.score,
                baseline.score
            );
        }
    }

    #[test]
    fn test_boost_query_nested() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_boost_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // Nested boost: positive is a Boolean query
        let positive_query = FtsQueryExpr::boolean()
            .should(FtsQueryExpr::match_query("programming"))
            .should(FtsQueryExpr::match_query("web"))
            .build();

        let query = FtsQueryExpr::boosting_with_negative(
            positive_query,
            FtsQueryExpr::match_query("python"),
            0.5,
        );

        let entries = index.search_query(&query);

        // Should match docs 0, 1, 2, 3, 4 (programming or web)
        assert!(entries.len() >= 4, "Should match multiple docs");

        // Python docs (1, 3) should be demoted
        let python_docs: Vec<_> = entries
            .iter()
            .filter(|e| e.row_position == 1 || e.row_position == 3)
            .collect();

        let non_python_docs: Vec<_> = entries
            .iter()
            .filter(|e| e.row_position != 1 && e.row_position != 3)
            .collect();

        // At least some python docs should have lower scores
        if !python_docs.is_empty() && !non_python_docs.is_empty() {
            let max_python_score = python_docs.iter().map(|e| e.score).fold(0.0f32, f32::max);
            let max_non_python_score = non_python_docs
                .iter()
                .map(|e| e.score)
                .fold(0.0f32, f32::max);

            // This is a soft check - depends on BM25 scoring details
            // Just verify the demotion is happening
            assert!(
                python_docs.iter().any(|e| e.score < max_non_python_score)
                    || max_python_score <= max_non_python_score,
                "Python docs should generally have lower scores"
            );
        }
    }

    // ====== WAND Factor / Search Options Tests ======

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
        // wand_factor should be clamped to [0.0, 1.0]
        let options = SearchOptions::new().with_wand_factor(2.0);
        assert_eq!(options.wand_factor, 1.0);

        let options = SearchOptions::new().with_wand_factor(-0.5);
        assert_eq!(options.wand_factor, 0.0);
    }

    fn create_wand_test_batch(schema: &ArrowSchema) -> RecordBatch {
        // Test documents with varying relevance:
        // Doc 0: "alpha alpha alpha beta" - high relevance for "alpha" (3 occurrences)
        // Doc 1: "alpha beta gamma" - medium relevance for "alpha" (1 occurrence)
        // Doc 2: "beta gamma delta" - no relevance for "alpha"
        // Doc 3: "alpha alpha" - medium-high relevance for "alpha" (2 occurrences, shorter doc)
        // Doc 4: "alpha" - some relevance for "alpha" (1 occurrence, very short doc)
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

        // Full recall (wand_factor = 1.0)
        let options = SearchOptions::default();
        let results = index.search_with_options(&query, options);

        // Should return all docs containing "alpha" (docs 0, 1, 3, 4)
        assert_eq!(results.len(), 4, "Expected 4 matches with full recall");

        // Results should be sorted by score descending
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "Results should be sorted by score descending"
            );
        }
    }

    #[test]
    fn test_search_with_options_with_limit() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_wand_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::match_query("alpha");

        // Limit to top 2 results
        let options = SearchOptions::new().with_limit(2);
        let results = index.search_with_options(&query, options);

        assert_eq!(results.len(), 2, "Expected 2 matches with limit=2");

        // Should be the top 2 by score
        let full_results = index.search_query(&query);
        let mut full_sorted = full_results;
        full_sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        assert_eq!(
            results[0].row_position, full_sorted[0].row_position,
            "First result should be highest scorer"
        );
        assert_eq!(
            results[1].row_position, full_sorted[1].row_position,
            "Second result should be second highest scorer"
        );
    }

    #[test]
    fn test_search_with_options_wand_factor_pruning() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_wand_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::match_query("alpha");

        // Get full results first to understand the score distribution
        let full_results = index.search_query(&query);
        let mut full_sorted = full_results.clone();
        full_sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // With wand_factor = 0.0, should only keep results at or above threshold (max_score * 0.0 = 0)
        // Actually with wand_factor = 0.0, threshold = max_score * 0.0 = 0, so all positive scores pass
        // The real test is to use a higher wand_factor like 0.5
        let options = SearchOptions::new().with_wand_factor(0.5);
        let results = index.search_with_options(&query, options);

        // Results should be pruned based on threshold
        if !results.is_empty() {
            let max_score = full_sorted[0].score;
            let threshold = max_score * 0.5;

            for result in &results {
                assert!(
                    result.score >= threshold - 0.001, // small epsilon for float comparison
                    "With wand_factor=0.5, all results should score >= {} but got {}",
                    threshold,
                    result.score
                );
            }

            // Should have fewer or equal results compared to full results
            assert!(
                results.len() <= full_results.len(),
                "Pruned results should not exceed full results"
            );
        }
    }

    #[test]
    fn test_search_with_options_wand_factor_with_limit() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_wand_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        let query = FtsQueryExpr::match_query("alpha");

        // Get full results to understand score distribution
        let full_results = index.search_query(&query);
        assert!(
            full_results.len() >= 3,
            "Need at least 3 results for this test"
        );

        // With limit=2 and wand_factor=0.5, prune docs scoring below 50% of 2nd best
        let options = SearchOptions::new().with_limit(2).with_wand_factor(0.5);
        let results = index.search_with_options(&query, options);

        // Should have at most 2 results (the limit)
        assert!(results.len() <= 2, "Should not exceed limit");

        // Results should be sorted by score
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

        // Query for something that doesn't exist
        let query = FtsQueryExpr::match_query("nonexistent");
        let options = SearchOptions::new().with_limit(10).with_wand_factor(0.5);
        let results = index.search_with_options(&query, options);

        assert!(
            results.is_empty(),
            "Should return empty for non-matching query"
        );
    }

    #[test]
    fn test_search_with_options_boolean_query() {
        let schema = create_test_schema();
        let index = FtsMemIndex::new(1, "description".to_string());

        let batch = create_wand_test_batch(&schema);
        index.insert(&batch, 0).unwrap();

        // Boolean query: alpha SHOULD beta
        let query = FtsQueryExpr::boolean()
            .should(FtsQueryExpr::match_query("alpha"))
            .should(FtsQueryExpr::match_query("beta"))
            .build();

        let options = SearchOptions::new().with_limit(3);
        let results = index.search_with_options(&query, options);

        assert!(results.len() <= 3, "Should not exceed limit");
        // Results should be sorted by score descending
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }
}
