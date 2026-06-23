// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::collections::BTreeMap;
use std::iter::once;
use std::time::Instant;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
};

use super::lance_format::LanceIndexStore;
use super::{
    AnyQuery, BuiltinIndexType, IndexFile, IndexReader, IndexStore, IndexWriter, MetricsCollector,
    ScalarIndex, ScalarIndexParams, SearchResult, TextQuery,
};
use crate::frag_reuse::FragReuseIndex;
use crate::metrics::NoOpMetricsCollector;
use crate::pbold;
use crate::scalar::expression::{ScalarQueryParser, TextQueryParser};
use crate::scalar::registry::{
    ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest, VALUE_COLUMN_NAME,
};
use crate::scalar::{CreatedIndex, UpdateCriteria};
use crate::vector::VectorIndex;
use crate::{Index, IndexType};
use arrow::array::{AsArray, UInt32Builder};
use arrow::datatypes::{UInt32Type, UInt64Type};
use arrow_array::{BinaryArray, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use futures::{FutureExt, Stream, StreamExt, TryStreamExt, stream};
use lance_arrow::iter_str_array;
use lance_core::cache::{CacheKey, LanceCache, WeakLanceCache};
use lance_core::deepsize::DeepSizeOf;
use lance_core::error::LanceOptionExt;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tempfile::TempDir;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::utils::tracing::{IO_TYPE_LOAD_SCALAR_PART, TRACE_IO_EVENTS};
use lance_core::{Error, ROW_ID, Result};
use lance_io::object_store::ObjectStore;
use lance_select::RowAddrTreeMap;
use lance_tokenizer::{
    AlphaNumOnlyFilter, AsciiFoldingFilter, LowerCaser, NgramTokenizer, RawTokenizer, TextAnalyzer,
};
use log::info;
use roaring::{RoaringBitmap, RoaringTreemap};
use serde::{Deserialize, Serialize};
use tracing::instrument;

mod ngram_regex;
pub(crate) use ngram_regex::regex_can_use_index;

const TOKENS_COL: &str = "tokens";
const POSTING_LIST_COL: &str = "posting_list";
const POSTINGS_FILENAME: &str = "ngram_postings.lance";
const NGRAM_TRIGRAM_INDEX_VERSION: u32 = 0;
const NGRAM_SPARSE_INDEX_VERSION: u32 = 1;

// The Index format version is checked before loading. The postings metadata is used
// after loading to pick the query tokenization without changing protobuf index
// details.
const NGRAM_TOKENIZATION_METADATA_KEY: &str = "lance.ngram.tokenization";

// FNV-1a constants used to map variable-length sparse n-grams into the existing
// u32 token column.
const SPARSE_HASH_OFFSET: u32 = 0x811c9dc5;
const SPARSE_HASH_PRIME: u32 = 0x01000193;
const SPARSE_MIN_NGRAM_LEN: usize = 3;
// Bound sparse token generation so one very long URL/base64/id run cannot
// materialize every substring during index build or query planning.
const SPARSE_MAX_NGRAM_LEN: usize = 128;

use std::sync::LazyLock;

pub static TOKENS_FIELD: LazyLock<Field> =
    LazyLock::new(|| Field::new(TOKENS_COL, DataType::UInt32, true));
pub static POSTINGS_FIELD: LazyLock<Field> =
    LazyLock::new(|| Field::new(POSTING_LIST_COL, DataType::Binary, false));
pub static POSTINGS_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        TOKENS_FIELD.clone(),
        POSTINGS_FIELD.clone(),
    ]))
});
pub static TEXT_PREPPER: LazyLock<TextAnalyzer> = LazyLock::new(|| {
    TextAnalyzer::builder(RawTokenizer::default())
        .filter(LowerCaser)
        .filter(AsciiFoldingFilter)
        .build()
});
/// Currently we ALWAYS use trigrams with ascii folding and lower casing.  We may want to make this configurable in the future.
pub static NGRAM_TOKENIZER: LazyLock<TextAnalyzer> = LazyLock::new(|| {
    TextAnalyzer::builder(NgramTokenizer::all_ngrams(3, 3).unwrap())
        .filter(AlphaNumOnlyFilter)
        .build()
});

// Helper function to apply a function to each token in a text
fn tokenize_visitor(tokenizer: &TextAnalyzer, text: &str, mut visitor: impl FnMut(&String)) {
    // The token_stream method is mutable.  As far as I can tell this is to enforce exclusivity and not
    // true mutability.  For example, the object returned by `token_stream` has thread-local state but
    // it is reset each time `token_stream` is called.
    //
    // However, I don't see this documented anywhere and I'm not sure about relying on it.  For now, we
    // make a clone as that seems to be the safer option.  All the tokenizers we use here should be trivially
    // cloneable (although it requires a heap allocation so may be worth investigating in the future)
    let mut prepper = TEXT_PREPPER.clone();
    let mut tokenizer = tokenizer.clone();
    let mut raw_stream = prepper.token_stream(text);
    while raw_stream.advance() {
        let mut token_stream = tokenizer.token_stream(&raw_stream.token().text);
        while token_stream.advance() {
            visitor(&token_stream.token().text);
        }
    }
}

const ALPHA_SPAN: usize = 37;
const MAX_TOKEN: usize = ALPHA_SPAN.pow(2) + ALPHA_SPAN;
const MIN_TOKEN: usize = 0;
const NGRAM_N: usize = 3;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NGramTokenization {
    #[default]
    Trigram,
    Sparse,
}

impl NGramTokenization {
    fn as_metadata_value(self) -> &'static str {
        match self {
            Self::Trigram => "trigram",
            Self::Sparse => "sparse",
        }
    }

    fn from_metadata_value(value: Option<&String>) -> Result<Self> {
        match value.map(String::as_str) {
            None | Some("trigram") => Ok(Self::Trigram),
            Some("sparse") => Ok(Self::Sparse),
            Some(other) => Err(Error::invalid_input_source(
                format!("Unknown ngram tokenization mode: {other}").into(),
            )),
        }
    }

    fn index_version(self) -> u32 {
        match self {
            Self::Trigram => NGRAM_TRIGRAM_INDEX_VERSION,
            Self::Sparse => NGRAM_SPARSE_INDEX_VERSION,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NGramIndexParams {
    #[serde(default)]
    pub tokenization: NGramTokenization,
}

// Convert an ngram (string) to a token (u32).  This helps avoid heap allocations
// and it makes it easier to partition the tokens for shuffling
//
// There are 36 alphanumeric values and we add 1 for the NULL token giving us 37^3
// potential tokens.
//
// "" => 0
// "?" => 37^2 * ?
// "?$" => 37^2 * ? + 37 * $
// "?$#" => 37^2 * ? + 37 * $ + #
// ...
//
// The ?,$,# represent the position in the alphabet (+1 to distinguish from NULL)
//
// Small strings get the larger multipliers because those ngrams are
// less likely to be unique and will have larger bitmaps.  We want to
// spread those out.
//
// NOTE: Today we hard-code trigrams and we do not include 1-grams or 2-grams so this
// function is more general than it needs to be...just in case.
fn ngram_to_token(ngram: &str, ngram_length: usize) -> u32 {
    let mut token = 0;
    // Empty string will get 0
    for (idx, byte) in ngram.bytes().enumerate() {
        let pos = if byte <= b'9' {
            byte - b'0'
        } else if byte <= b'z' {
            byte - b'a' + 10
        } else {
            unreachable!()
        } + 1;
        debug_assert!(pos < ALPHA_SPAN as u8);
        let mult = ALPHA_SPAN.pow(ngram_length as u32 - idx as u32 - 1) as u32;
        token += pos as u32 * mult;
    }
    token
}

// Stable FNV-1a hash for persisted sparse token ids. Collisions only add false
// positives because NGram search results are rechecked by the scan.
fn stable_sparse_hash(bytes: impl IntoIterator<Item = u8>) -> u32 {
    let mut hash = SPARSE_HASH_OFFSET;
    for byte in bytes {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(SPARSE_HASH_PRIME);
    }
    // Token 0 is reserved for NULL rows. Hash collisions are acceptable because
    // this index always returns candidates that are rechecked by the scan.
    if hash == 0 { 1 } else { hash }
}

// Prefix sparse n-gram bytes before hashing so future token families can share
// the u32 postings column without reusing the same hash namespace.
fn sparse_ngram_to_token(ngram: &[u8]) -> u32 {
    stable_sparse_hash(
        b"lance_sparse_ngram_v1"
            .iter()
            .copied()
            .chain(ngram.iter().copied()),
    )
}

fn sparse_pair_weight(left: u8, right: u8) -> u32 {
    stable_sparse_hash([left, right])
}

// Apply the same text normalization as the trigram tokenizer, then expose only
// ASCII-alphanumeric runs so byte offsets remain valid sparse n-gram boundaries.
fn normalized_alnum_runs(text: &str, mut visitor: impl FnMut(&str)) {
    let mut prepper = TEXT_PREPPER.clone();
    let mut raw_stream = prepper.token_stream(text);
    while raw_stream.advance() {
        let mut run = String::new();
        for ch in raw_stream.token().text.chars() {
            if ch.is_ascii_alphanumeric() {
                run.push(ch);
            } else if !run.is_empty() {
                visitor(&run);
                run.clear();
            }
        }
        if !run.is_empty() {
            visitor(&run);
        }
    }
}

// A sparse token plus the byte range it covers within a normalized run. The
// range is used by query covering; the index build path only needs `token`.
#[derive(Debug, Clone, Copy)]
struct SparseTokenSpan {
    token: u32,
    #[cfg(test)]
    start: usize,
    #[cfg(test)]
    end: usize,
}

#[derive(Debug, Clone, Copy)]
struct SparsePair {
    // A sparse n-gram boundary is represented by the hash weight of an
    // adjacent byte pair. `pos` is the byte offset of the pair's first byte.
    weight: u32,
    pos: usize,
}

fn sparse_span(bytes: &[u8], start: usize, end: usize) -> SparseTokenSpan {
    SparseTokenSpan {
        token: sparse_ngram_to_token(&bytes[start..end]),
        #[cfg(test)]
        start,
        #[cfg(test)]
        end,
    }
}

fn push_sparse_span(spans: &mut Vec<SparseTokenSpan>, bytes: &[u8], start: usize, end: usize) {
    if end - start <= SPARSE_MAX_NGRAM_LEN {
        spans.push(sparse_span(bytes, start, end));
    }
}

// Generate the sparse n-grams stored in the index with the standard monotonic
// stack construction. A bounded span is emitted when its boundary pair weights
// dominate the internal pair weights.
fn sparse_spans_for_run(run: &str) -> Vec<SparseTokenSpan> {
    let bytes = run.as_bytes();
    if bytes.len() < SPARSE_MIN_NGRAM_LEN {
        return Vec::new();
    }

    let mut spans = Vec::new();
    // Monotonic stack of candidate boundary pairs, ordered by non-increasing
    // pair weight from bottom to top.
    let mut stack: Vec<SparsePair> = Vec::new();
    for pair_pos in 0..bytes.len() - 1 {
        let pair = SparsePair {
            weight: sparse_pair_weight(bytes[pair_pos], bytes[pair_pos + 1]),
            pos: pair_pos,
        };
        while stack.last().is_some_and(|last| pair.weight > last.weight) {
            let start = stack.last().unwrap().pos;
            // The current pair becomes the right boundary for the span whose
            // left boundary is the pending tail pair.
            push_sparse_span(&mut spans, bytes, start, pair_pos + 2);
            // Equal-weight suffix pairs form a plateau. Sparse n-gram
            // boundaries require strict domination over internal pairs, so the
            // plateau can be collapsed before removing the dominated boundary.
            while stack.len() > 1 && stack.last().unwrap().weight == stack[stack.len() - 2].weight {
                stack.pop();
            }
            stack.pop();
        }
        if let Some(last) = stack.last() {
            // The stack tail and the current pair form a valid sparse span. For
            // adjacent pairs this emits the minimum-length token with no
            // internal pair.
            push_sparse_span(&mut spans, bytes, last.pos, pair_pos + 2);
        }
        stack.push(pair);
    }
    spans
}

// Index construction stores every sparse token for each normalized run. Query
// planning can then choose a smaller covering subset without missing matches.
fn sparse_index_token_visitor(text: &str, mut visitor: impl FnMut(u32)) {
    normalized_alnum_runs(text, |run| {
        for span in sparse_spans_for_run(run) {
            visitor(span.token);
        }
    });
}

// Query planning uses the covering sparse n-gram stack construction so it can
// read fewer, longer posting lists than the fixed-trigram path.
fn sparse_covering_tokens(text: &str) -> Vec<u32> {
    let mut tokens = Vec::new();
    normalized_alnum_runs(text, |run| {
        let bytes = run.as_bytes();
        if bytes.len() < SPARSE_MIN_NGRAM_LEN {
            return;
        }

        // Keep boundary candidates in a deque ordered by byte position. The
        // front is the oldest boundary still available for the current
        // covering window, and the back is the newest boundary being compared
        // against the next adjacent pair. It is also monotonic by weight from
        // front to back, with non-increasing pair weights.
        let mut stack: VecDeque<SparsePair> = VecDeque::new();
        for pair_pos in 0..bytes.len() - 1 {
            let pair = SparsePair {
                weight: sparse_pair_weight(bytes[pair_pos], bytes[pair_pos + 1]),
                pos: pair_pos,
            };

            if stack.len() > 1 {
                let covering_len = pair_pos + 2 - stack.front().unwrap().pos;
                if covering_len > SPARSE_MAX_NGRAM_LEN {
                    // The first two entries are adjacent surviving boundaries.
                    // The build path emits the same bounded span when the
                    // second boundary is pushed, so this token is safe to
                    // require during query filtering.
                    let start = stack.front().unwrap().pos;
                    let end = stack[1].pos + 2;
                    tokens.push(sparse_ngram_to_token(&bytes[start..end]));
                    stack.pop_front();
                }
            }

            while stack.back().is_some_and(|last| pair.weight > last.weight) {
                // Query covering intentionally emits fewer tokens than index
                // build. The index stores all valid sparse n-grams, but a
                // query only needs a covering subset. Emit here only when the
                // stack must be closed, such as for an equal-weight suffix.
                if stack.front().unwrap().weight == stack.back().unwrap().weight {
                    let start = stack.back().unwrap().pos;
                    tokens.push(sparse_ngram_to_token(&bytes[start..pair_pos + 2]));
                    while stack.len() > 1 {
                        let end = stack.back().unwrap().pos + 2;
                        stack.pop_back();
                        let start = stack.back().unwrap().pos;
                        tokens.push(sparse_ngram_to_token(&bytes[start..end]));
                    }
                }
                stack.pop_back();
            }

            stack.push_back(pair);
        }

        // The run ended before these pending boundaries were closed by a
        // higher-weight pair. Emit adjacent spans to cover the remaining
        // monotonic suffix; adjacent spans have no internal pair, so they are
        // valid sparse n-grams.
        while stack.len() > 1 {
            let end = stack.back().unwrap().pos + 2;
            stack.pop_back();
            let start = stack.back().unwrap().pos;
            tokens.push(sparse_ngram_to_token(&bytes[start..end]));
        }
    });
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

// Convert a query literal into the token ids required by the selected
// tokenization. Regex analysis uses this for literal fragments as well.
fn query_tokens_for_text(tokenization: NGramTokenization, text: &str) -> Vec<u32> {
    match tokenization {
        NGramTokenization::Trigram => {
            let mut tokens = Vec::new();
            tokenize_visitor(&NGRAM_TOKENIZER, text, |ngram| {
                tokens.push(ngram_to_token(ngram, NGRAM_N));
            });
            tokens
        }
        NGramTokenization::Sparse => sparse_covering_tokens(text),
    }
}

// Persist the tokenization mode in the postings file so an index can recover
// query tokenization after it is loaded.
fn ngram_metadata(tokenization: NGramTokenization) -> HashMap<String, String> {
    HashMap::from([(
        NGRAM_TOKENIZATION_METADATA_KEY.to_string(),
        tokenization.as_metadata_value().to_string(),
    )])
}

/// Basic stats about an ngram index
#[derive(Serialize)]
struct NGramStatistics {
    num_ngrams: usize,
    tokenization: NGramTokenization,
}

/// The row ids that contain a given ngram
#[derive(Debug)]
pub struct NGramPostingList {
    bitmap: RoaringTreemap,
}

impl DeepSizeOf for NGramPostingList {
    fn deep_size_of_children(&self, _: &mut lance_core::deepsize::Context) -> usize {
        self.bitmap.serialized_size()
    }
}

// Cache key implementation for type-safe cache access
#[derive(Debug, Clone)]
pub struct NGramPostingListKey {
    pub row_offset: u32,
}

impl CacheKey for NGramPostingListKey {
    type ValueType = NGramPostingList;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("posting-list-{}", self.row_offset).into()
    }

    fn type_name() -> &'static str {
        "NGramPostingList"
    }
}

impl NGramPostingList {
    fn try_from_batch(
        batch: RecordBatch,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let bitmap_bytes = batch.column(0).as_binary::<i32>().value(0);
        let mut bitmap = RoaringTreemap::deserialize_from(bitmap_bytes)
            .map_err(|e| Error::internal(format!("Error deserializing ngram list: {}", e)))?;
        if let Some(frag_reuse_index_ref) = frag_reuse_index.as_ref() {
            bitmap = frag_reuse_index_ref.remap_row_ids_roaring_tree_map(&bitmap);
        }
        Ok(Self { bitmap })
    }

    fn intersect<'a>(lists: impl IntoIterator<Item = &'a Self>) -> RoaringTreemap {
        let mut iter = lists.into_iter();
        let mut result = iter
            .next()
            .map(|list| list.bitmap.clone())
            .unwrap_or_default();
        for list in iter {
            result &= &list.bitmap;
        }
        result
    }
}

/// Reads on-demand ngram posting lists from storage (and stores them in a cache)
struct NGramPostingListReader {
    reader: Arc<dyn IndexReader>,
    frag_reuse_index: Option<Arc<FragReuseIndex>>,
    index_cache: WeakLanceCache,
}

impl DeepSizeOf for NGramPostingListReader {
    fn deep_size_of_children(&self, _: &mut lance_core::deepsize::Context) -> usize {
        0
    }
}

impl std::fmt::Debug for NGramPostingListReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NGramListReader").finish()
    }
}

impl NGramPostingListReader {
    #[instrument(level = "debug", skip(self, metrics))]
    pub async fn ngram_list(
        &self,
        row_offset: u32,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<NGramPostingList>> {
        self.index_cache.get_or_insert_with_key(NGramPostingListKey { row_offset }, || async move {
            metrics.record_part_load();
                tracing::info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_SCALAR_PART, index_type="ngram", part_id=row_offset);
                let batch = self
                    .reader
                    .read_range(
                        row_offset as usize..row_offset as usize + 1,
                        Some(&[POSTING_LIST_COL]),
                    )
                    .await?;
                NGramPostingList::try_from_batch(batch, self.frag_reuse_index.clone())
        }).await
    }
}

/// An ngram index
///
/// At a high level this is an inverted index that maps ngrams (small fixed size substrings) to the
/// row ids that contain them.
///
/// As a simple example consider a 1-gram index.  It would basically be a mapping from
/// each letter to the row ids that contain that letter.  Then, if the user searches for
/// "cat", the index would look up the row ids for "c", "a", and "t", and return the intersection
/// of those row ids because only rows have at least one c, a, and t could possible contain "cat".
///
/// This is an in-exact index, similar to a bloom filter.  It can return false positives and a
/// recheck step is needed to confirm the results.
///
/// Note that it cannot return false negatives.
pub struct NGramIndex {
    /// The mapping from tokens to row offsets
    tokens: HashMap<u32, u32>,
    /// Tokenization mode used to build this index and to derive query tokens.
    tokenization: NGramTokenization,
    /// The reader for the posting lists
    list_reader: Arc<NGramPostingListReader>,
    io_parallelism: usize,
    /// The store that owns the index
    store: Arc<dyn IndexStore>,
}

impl std::fmt::Debug for NGramIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NGramIndex")
            .field("tokens", &self.tokens)
            .field("list_reader", &self.list_reader)
            .finish()
    }
}

impl DeepSizeOf for NGramIndex {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.tokens.deep_size_of_children(context)
    }
}

impl NGramIndex {
    async fn from_store(
        store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        index_cache: &LanceCache,
    ) -> Result<Self> {
        let tokens_reader = store.open_index_file(POSTINGS_FILENAME).await?;
        let tokenization = NGramTokenization::from_metadata_value(
            tokens_reader
                .schema()
                .metadata
                .get(NGRAM_TOKENIZATION_METADATA_KEY),
        )?;
        let tokens = tokens_reader
            .read_range(0..tokens_reader.num_rows(), Some(&[TOKENS_COL]))
            .await?;

        let tokens_map = HashMap::from_iter(
            tokens
                .column(0)
                .as_primitive::<UInt32Type>()
                .values()
                .iter()
                .copied()
                .enumerate()
                .map(|(idx, token)| (token, idx as u32)),
        );
        let posting_reader = Arc::new(NGramPostingListReader {
            reader: store.open_index_file(POSTINGS_FILENAME).await?,
            frag_reuse_index,
            index_cache: WeakLanceCache::from(index_cache),
        });

        Ok(Self {
            io_parallelism: store.io_parallelism(),
            tokens: tokens_map,
            tokenization,
            list_reader: posting_reader,
            store,
        })
    }

    fn remap_batch(
        &self,
        batch: RecordBatch,
        mapping: &HashMap<u64, Option<u64>>,
    ) -> Result<RecordBatch> {
        let posting_lists_array = batch
            .column_by_name(POSTING_LIST_COL)
            .expect_ok()?
            .as_binary::<i32>();

        let new_posting_lists = posting_lists_array
            .iter()
            .map(|posting_list| {
                let posting_list = posting_list.unwrap();
                let posting_list = RoaringTreemap::deserialize_from(posting_list)?;
                let new_posting_list =
                    RoaringTreemap::from_iter(posting_list.into_iter().filter_map(|row_id| {
                        match mapping.get(&row_id) {
                            Some(Some(new_row_id)) => Some(*new_row_id),
                            Some(None) => None,
                            None => Some(row_id),
                        }
                    }));
                let mut buf = Vec::with_capacity(new_posting_list.serialized_size());
                new_posting_list.serialize_into(&mut buf)?;
                Ok(buf)
            })
            .collect::<Result<Vec<_>>>()?;

        let new_posting_lists_array = BinaryArray::from_iter_values(new_posting_lists);

        Ok(RecordBatch::try_new(
            POSTINGS_SCHEMA.clone(),
            vec![
                batch.column_by_name(TOKENS_COL).expect_ok()?.clone(),
                Arc::new(new_posting_lists_array),
            ],
        )?)
    }

    async fn load(
        store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        index_cache: &LanceCache,
    ) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        Ok(Arc::new(
            Self::from_store(store, frag_reuse_index, index_cache).await?,
        ))
    }

    fn query_tokens_for_text(&self, text: &str) -> Vec<u32> {
        query_tokens_for_text(self.tokenization, text)
    }
}

#[async_trait]
impl Index for NGramIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn VectorIndex>> {
        Err(Error::invalid_input_source(
            "NGramIndex is not a vector index".into(),
        ))
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let ngram_stats = NGramStatistics {
            num_ngrams: self.tokens.len(),
            tokenization: self.tokenization,
        };
        serde_json::to_value(ngram_stats)
            .map_err(|e| Error::internal(format!("Error serializing statistics: {}", e)))
    }

    async fn prewarm(&self) -> Result<()> {
        // TODO: NGram index can pre-warm by loading all posting lists into memory
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::NGram
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let mut frag_ids = RoaringBitmap::new();
        for row_offset in self.tokens.values() {
            let list = self
                .list_reader
                .ngram_list(*row_offset, &NoOpMetricsCollector)
                .await?;
            frag_ids.extend(
                list.bitmap
                    .iter()
                    .map(|row_addr| RowAddress::from(row_addr).fragment_id()),
            );
        }
        Ok(frag_ids)
    }
}

#[async_trait]
impl ScalarIndex for NGramIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query
            .as_any()
            .downcast_ref::<TextQuery>()
            .ok_or_else(|| Error::invalid_input_source("Query is not a TextQuery".into()))?;
        match query {
            TextQuery::StringContains(substr) => {
                if substr.len() < NGRAM_N {
                    // We know nothing on short searches, need to recheck all
                    return Ok(SearchResult::at_least(RowAddrTreeMap::new()));
                }

                let mut row_offsets = Vec::with_capacity(substr.len() * 3);
                let mut missing = false;
                let query_tokens = self.query_tokens_for_text(substr);
                if query_tokens.is_empty() {
                    return Ok(SearchResult::at_least(RowAddrTreeMap::new()));
                }
                for token in query_tokens {
                    if let Some(row_offset) = self.tokens.get(&token) {
                        row_offsets.push(*row_offset);
                    } else {
                        missing = true;
                    }
                }
                // At least one token was missing, so we know there are zero results
                if missing {
                    return Ok(SearchResult::exact(RowAddrTreeMap::new()));
                }
                let posting_lists = futures::stream::iter(
                    row_offsets
                        .into_iter()
                        .map(|row_offset| self.list_reader.ngram_list(row_offset, metrics)),
                )
                .buffer_unordered(self.io_parallelism)
                .try_collect::<Vec<_>>()
                .await?;
                metrics.record_comparisons(posting_lists.len());
                let list_refs = posting_lists.iter().map(|list| list.as_ref());
                let row_ids = NGramPostingList::intersect(list_refs);
                Ok(SearchResult::at_most(RowAddrTreeMap::from(row_ids)))
            }
            TextQuery::Regex(pattern) => {
                let ngram_query = ngram_regex::regex_to_ngram_query(pattern, self.tokenization);
                match &ngram_query {
                    // No usable ngram structure (e.g. `a.b`, `.*`): the index
                    // cannot prune, so every row must be rechecked.
                    ngram_regex::NGramQuery::All => {
                        Ok(SearchResult::at_least(RowAddrTreeMap::new()))
                    }
                    // The pattern is provably unsatisfiable.
                    ngram_regex::NGramQuery::None => Ok(SearchResult::exact(RowAddrTreeMap::new())),
                    _ => {
                        let mut tokens = HashSet::new();
                        ngram_regex::collect_tokens(&ngram_query, &mut tokens);
                        // Fetch the posting list for every token the condition
                        // references; a token absent from the index contributes
                        // an empty list, which `eval_ngram_query` handles.
                        let present = tokens.into_iter().filter_map(|token| {
                            self.tokens.get(&token).map(|offset| (token, *offset))
                        });
                        let lists = futures::stream::iter(present.map(|(token, offset)| {
                            self.list_reader
                                .ngram_list(offset, metrics)
                                .map(move |result| result.map(|list| (token, list)))
                        }))
                        .buffer_unordered(self.io_parallelism)
                        .try_collect::<Vec<(u32, Arc<NGramPostingList>)>>()
                        .await?;
                        metrics.record_comparisons(lists.len());
                        let bitmaps: HashMap<u32, RoaringTreemap> = lists
                            .into_iter()
                            .map(|(token, list)| (token, list.bitmap.clone()))
                            .collect();
                        let row_ids = ngram_regex::eval_ngram_query(&ngram_query, &bitmaps);
                        Ok(SearchResult::at_most(RowAddrTreeMap::from(row_ids)))
                    }
                }
            }
        }
    }

    fn can_remap(&self) -> bool {
        true
    }

    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        let reader = self.store.open_index_file(POSTINGS_FILENAME).await?;
        let mut writer = dest_store
            .new_index_file(POSTINGS_FILENAME, POSTINGS_SCHEMA.clone())
            .await?;

        let mut offset = 0;
        let num_rows = reader.num_rows();
        const BATCH_SIZE: usize = 64;
        while offset < num_rows {
            let batch_size = BATCH_SIZE.min(num_rows - offset);
            let batch = reader.read_range(offset..offset + batch_size, None).await?;
            let batch = self.remap_batch(batch, mapping)?;
            writer.write_record_batch(batch).await?;
            offset += BATCH_SIZE;
        }

        let file = writer
            .finish_with_metadata(ngram_metadata(self.tokenization))
            .await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::NGramIndexDetails::default())
                .unwrap(),
            index_version: self.tokenization.index_version(),
            files: vec![file],
        })
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        _old_data_filter: Option<super::OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        let mut builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions {
            tokenization: self.tokenization,
            ..NGramIndexBuilderOptions::default()
        })?;
        let spill_files = builder.train(new_data).await?;

        let file = builder
            .write_index(dest_store, spill_files, Some(self.store.clone()))
            .await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::NGramIndexDetails::default())
                .unwrap(),
            index_version: self.tokenization.index_version(),
            files: vec![file],
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(TrainingCriteria::new(TrainingOrdering::None).with_row_id())
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::NGram);
        if self.tokenization == NGramTokenization::Trigram {
            Ok(params)
        } else {
            Ok(params.with_params(&NGramIndexParams {
                tokenization: self.tokenization,
            }))
        }
    }
}

#[derive(Debug, Clone)]
pub struct NGramIndexBuilderOptions {
    tokens_per_spill: usize,
    tokenization: NGramTokenization,
}

// A higher value will use more RAM.  A lower value will have to do more spilling
static DEFAULT_TOKENS_PER_SPILL: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("LANCE_NGRAM_TOKENS_PER_SPILL")
        .unwrap_or_else(|_| "1000000000".to_string())
        .parse()
        .expect("failed to parse LANCE_NGRAM_TOKENS_PER_SPILL")
});
// How many partitions to use for shuffling out the work.  We slightly
// over-allocate this since the amount of work per-partition is not uniform.
//
// Increasing this may increase the performance but it could increase RAM (since we will spill less often)
// and could hurt performance (since there will be more files at the end for the final spill)
static DEFAULT_NUM_PARTITIONS: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("LANCE_NGRAM_NUM_PARTITIONS")
        .map(|s| s.parse().expect("failed to parse LANCE_NGRAM_PARALLELISM"))
        .unwrap_or((get_num_compute_intensive_cpus() * 4).max(128))
});
// Just enough so that tokenizing is faster than I/O
static DEFAULT_TOKENIZE_PARALLELISM: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("LANCE_NGRAM_TOKENIZE_PARALLELISM")
        .map(|s| {
            s.parse()
                .expect("failed to parse LANCE_NGRAM_TOKENIZE_PARALLELISM")
        })
        .unwrap_or(8)
});

impl Default for NGramIndexBuilderOptions {
    fn default() -> Self {
        Self {
            tokens_per_spill: *DEFAULT_TOKENS_PER_SPILL,
            tokenization: NGramTokenization::Trigram,
        }
    }
}

// An ordered list of tokens and bitmaps
//
// The `tokens` list is ordered by token value.  This makes it easier to merge spill files.
struct NGramIndexSpillState {
    tokens: UInt32Array,
    bitmaps: Vec<RoaringTreemap>,
}

impl NGramIndexSpillState {
    fn try_from_batch(batch: RecordBatch) -> Result<Self> {
        let tokens = batch
            .column_by_name(TOKENS_COL)
            .expect_ok()?
            .as_primitive::<UInt32Type>()
            .clone();
        let postings = batch
            .column_by_name(POSTING_LIST_COL)
            .expect_ok()?
            .as_binary::<i32>();

        let bitmaps = postings
            .into_iter()
            .map(|bytes| {
                RoaringTreemap::deserialize_from(bytes.expect_ok()?)
                    .map_err(|e| Error::internal(format!("Error deserializing ngram list: {}", e)))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { tokens, bitmaps })
    }

    fn try_into_batch(self) -> Result<RecordBatch> {
        let bitmap_array = BinaryArray::from_iter_values(self.bitmaps.into_iter().map(|bitmap| {
            let mut buf = Vec::with_capacity(bitmap.serialized_size());
            bitmap.serialize_into(&mut buf).unwrap();
            buf
        }));
        Ok(RecordBatch::try_new(
            POSTINGS_SCHEMA.clone(),
            vec![Arc::new(self.tokens), Arc::new(bitmap_array)],
        )?)
    }
}

// As we're building we create a map from ngram to row ids.  When this map gets too large
// we spill it to disk.
struct NGramIndexBuildState {
    tokens_map: BTreeMap<u32, RoaringTreemap>,
}

impl NGramIndexBuildState {
    fn starting() -> Self {
        Self {
            tokens_map: BTreeMap::new(),
        }
    }

    fn take(&mut self) -> Self {
        let mut taken = Self::starting();
        std::mem::swap(&mut self.tokens_map, &mut taken.tokens_map);
        taken
    }

    fn into_spill(self) -> NGramIndexSpillState {
        // We can rely on these being in token order because of BTreeMap
        let tokens = UInt32Array::from_iter_values(self.tokens_map.keys().copied());
        let bitmaps = Vec::from_iter(self.tokens_map.into_values());

        NGramIndexSpillState { bitmaps, tokens }
    }
}

/// A builder for an ngram index
///
/// The builder is a small pipeline.  First, we read in the data and tokenize it.  This
/// stage uses fan-out parallelism to tokenize the data because tokenization may be a little
/// slower than I/O.
///
/// The second stage fans out much wider.  It partitions the tokens into a number of partitions.
/// Each partition has a BTreemap that maps tokens to row ids.  The partitions then build up
/// roaring treemaps.  When a partition gets too full it will spill to disk.
///
/// Once all the data is processed we spill all the parititons to disk and then we merge the
/// spill files into a single index file.
pub struct NGramIndexBuilder {
    tokenizer: TextAnalyzer,
    options: NGramIndexBuilderOptions,
    tmpdir: Arc<TempDir>,
    spill_store: Arc<dyn IndexStore>,

    tokens_seen: usize,
    worker_number: usize,
    has_flushed: bool,

    state: NGramIndexBuildState,
}

impl NGramIndexBuilder {
    pub fn try_new(options: NGramIndexBuilderOptions) -> Result<Self> {
        Self::from_state(NGramIndexBuildState::starting(), options)
    }

    fn clone_worker(&self, worker_number: usize) -> Self {
        let mut bitmaps = Vec::with_capacity(36 * 36 * 36 + 1);
        // Token 0 is always the NULL bitmap
        bitmaps.push(RoaringTreemap::new());
        Self {
            tokenizer: self.tokenizer.clone(),
            state: NGramIndexBuildState::starting(),
            tmpdir: self.tmpdir.clone(),
            spill_store: self.spill_store.clone(),
            options: self.options.clone(),
            tokens_seen: 0,
            worker_number,
            has_flushed: false,
        }
    }

    fn from_state(state: NGramIndexBuildState, options: NGramIndexBuilderOptions) -> Result<Self> {
        let tokenizer = NGRAM_TOKENIZER.clone();

        let tmpdir = Arc::new(TempDir::default());
        let spill_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));

        Ok(Self {
            tokenizer,
            state,
            tmpdir,
            spill_store,
            options,
            tokens_seen: 0,
            worker_number: 0,
            has_flushed: false,
        })
    }

    fn validate_schema(schema: &Schema) -> Result<()> {
        if schema.fields().len() != 2 {
            return Err(Error::invalid_input_source(
                "Ngram index schema must have exactly two fields".into(),
            ));
        }
        let values_field = schema.field_with_name(VALUE_COLUMN_NAME)?;
        if *values_field.data_type() != DataType::Utf8
            && *values_field.data_type() != DataType::LargeUtf8
        {
            return Err(Error::invalid_input_source(
                "First field in ngram index schema must be of type Utf8/LargeUtf8".into(),
            ));
        }
        let row_id_field = schema.field_with_name(ROW_ID)?;
        if *row_id_field.data_type() != DataType::UInt64 {
            return Err(Error::invalid_input_source(
                "Second field in ngram index schema must be of type UInt64".into(),
            ));
        }
        Ok(())
    }

    async fn process_batch(&mut self, tokens_and_ids: Vec<(u32, u64)>) -> Result<()> {
        let mut tokens_seen = 0;
        for (token, row_id) in tokens_and_ids {
            tokens_seen += 1;
            // This would be a bit simpler with entry API but, at scale, the vast majority
            // of cases will be a hit and we want to avoid cloning the string if we can.  So
            // for now we do the double-hash.  We can simplify in the future with raw_entry
            // when it stabilizes.
            self.state
                .tokens_map
                .entry(token)
                .or_default()
                .insert(row_id);
        }
        self.tokens_seen += tokens_seen;
        if self.tokens_seen >= self.options.tokens_per_spill {
            let state = self.state.take();
            self.flush(state).await?;
        }
        Ok(())
    }

    fn spill_filename(id: usize) -> String {
        format!("spill-{}.lance", id)
    }

    fn tmp_spill_filename(id: usize) -> String {
        format!("spill-{}.lance.tmp", id)
    }

    async fn flush(&mut self, state: NGramIndexBuildState) -> Result<bool> {
        if self.tokens_seen == 0 {
            assert!(state.tokens_map.is_empty());
            return Ok(self.has_flushed);
        }
        self.tokens_seen = 0;
        let spill_state = state.into_spill();
        let flush_start = Instant::now();
        // The primary builder should never flush
        debug_assert_ne!(self.worker_number, 0);
        if self.has_flushed {
            info!("Merging flush for worker {}", self.worker_number);
            // If we have flushed before then we need to merge with the spill file
            let mut writer = self
                .spill_store
                .new_index_file(
                    &Self::tmp_spill_filename(self.worker_number),
                    POSTINGS_SCHEMA.clone(),
                )
                .await?;

            let left_stream = stream::once(std::future::ready(Ok(spill_state)));
            let right_stream =
                Self::stream_spill(self.spill_store.clone(), self.worker_number).await?;
            Self::merge_spill_streams(left_stream, right_stream, writer.as_mut()).await?;
            drop(writer);
            self.spill_store
                .rename_index_file(
                    &Self::tmp_spill_filename(self.worker_number),
                    &Self::spill_filename(self.worker_number),
                )
                .await?;
        } else {
            // If we haven't flushed before we can just write to the spill file
            info!("Initial flush for worker {}", self.worker_number);
            self.has_flushed = true;
            let writer = self
                .spill_store
                .new_index_file(
                    &Self::spill_filename(self.worker_number),
                    POSTINGS_SCHEMA.clone(),
                )
                .await?;
            self.write(writer, spill_state).await?;
        }
        let flush_time = flush_start.elapsed();
        info!(
            "Flushed worker {} in {}ms",
            self.worker_number,
            flush_time.as_millis()
        );
        Ok(true)
    }

    fn tokenize_and_partition(
        tokenizer: &TextAnalyzer,
        tokenization: NGramTokenization,
        batch: RecordBatch,
        num_workers: usize,
    ) -> Result<Vec<Vec<(u32, u64)>>> {
        let text_iter = iter_str_array(batch.column_by_name(VALUE_COLUMN_NAME).expect_ok()?);
        let row_id_col = batch
            .column_by_name(ROW_ID)
            .expect_ok()?
            .as_primitive::<UInt64Type>();
        // Guessing 1000 tokens per row to at least avoid some of the earlier allocations
        let mut partitions = vec![Vec::with_capacity(batch.num_rows() * 1000); num_workers];
        let divisor = (MAX_TOKEN - MIN_TOKEN) / num_workers;
        for (text, row_id) in text_iter.zip(row_id_col.values()) {
            if let Some(text) = text {
                match tokenization {
                    NGramTokenization::Trigram => tokenize_visitor(tokenizer, text, |token| {
                        let token = ngram_to_token(token, NGRAM_N);
                        let partition_id = (token as usize).saturating_sub(MIN_TOKEN) / divisor;
                        partitions[partition_id % num_workers].push((token, *row_id));
                    }),
                    NGramTokenization::Sparse => sparse_index_token_visitor(text, |token| {
                        let partition_id = token as usize % num_workers;
                        partitions[partition_id].push((token, *row_id));
                    }),
                }
            } else {
                partitions[0].push((0, *row_id));
            }
        }
        Ok(partitions)
    }

    fn write_metadata(&self) -> HashMap<String, String> {
        ngram_metadata(self.options.tokenization)
    }

    pub async fn train(&mut self, data: SendableRecordBatchStream) -> Result<Vec<usize>> {
        let schema = data.schema();
        Self::validate_schema(schema.as_ref())?;

        let num_workers = *DEFAULT_NUM_PARTITIONS;
        let mut senders = Vec::with_capacity(num_workers);
        let mut builders = Vec::with_capacity(num_workers);
        for worker_idx in 0..num_workers {
            let (send, mut recv) = tokio::sync::mpsc::channel(2);
            senders.push(send);

            let mut builder = self.clone_worker(worker_idx + 1);
            let future = tokio::spawn(async move {
                while let Some(partition) = recv.recv().await {
                    builder.process_batch(partition).await?;
                }
                Result::Ok(builder)
            });
            builders.push(future);
        }

        let mut partitions_stream = data
            .and_then(|batch| {
                let tokenizer = self.tokenizer.clone();
                let tokenization = self.options.tokenization;
                std::future::ready(Ok(tokio::task::spawn(async move {
                    Ok(Self::tokenize_and_partition(
                        &tokenizer,
                        tokenization,
                        batch,
                        num_workers,
                    )?)
                })
                .map(|res| res.unwrap())))
            })
            .try_buffer_unordered(*DEFAULT_TOKENIZE_PARALLELISM);

        while let Some(partitions) = partitions_stream.try_next().await? {
            for (part_idx, partition) in partitions.into_iter().enumerate() {
                senders[part_idx].send(partition).await.unwrap();
            }
        }

        std::mem::drop(senders);
        let builders = futures::future::try_join_all(builders).await?;

        // Final flush is serialized.  If we kick this off in parallel it can
        // use a lot of memory.

        let mut to_spill = Vec::with_capacity(builders.len());

        for builder in builders {
            let mut builder = builder?;
            let state = builder.state.take();
            if builder.flush(state).await? {
                to_spill.push(builder.worker_number);
            }
        }

        Ok(to_spill)
    }

    async fn write(
        &mut self,
        mut writer: Box<dyn IndexWriter>,
        state: NGramIndexSpillState,
    ) -> Result<()> {
        writer.write_record_batch(state.try_into_batch()?).await?;
        writer.finish().await?;

        Ok(())
    }

    async fn stream_spill_reader(
        reader: Arc<dyn IndexReader>,
    ) -> Result<impl Stream<Item = Result<NGramIndexSpillState>>> {
        let num_rows = reader.num_rows();

        Ok(stream::try_unfold(0, move |offset| {
            let reader = reader.clone();
            async move {
                // These are small batches but, in the worst case scenario, each row could
                // be massive (up to 128MB per row at 1B rows) and we end up breaking memory
                let batch_size = std::cmp::min(num_rows - offset, 64);
                if batch_size == 0 {
                    return Ok(None);
                }
                let batch = reader.read_range(offset..offset + batch_size, None).await?;
                let state = NGramIndexSpillState::try_from_batch(batch)?;
                let new_offset = offset + batch_size;
                Ok(Some((state, new_offset)))
            }
            .boxed()
        }))
    }

    async fn stream_spill(
        spill_store: Arc<dyn IndexStore>,
        id: usize,
    ) -> Result<impl Stream<Item = Result<NGramIndexSpillState>>> {
        let reader = spill_store
            .open_index_file(&Self::spill_filename(id))
            .await?;
        Self::stream_spill_reader(reader).await
    }

    fn merge_spill_states(
        left_opt: &mut Option<NGramIndexSpillState>,
        right_opt: &mut Option<NGramIndexSpillState>,
    ) -> NGramIndexSpillState {
        let left = left_opt.take().unwrap();
        let right = right_opt.take().unwrap();

        let item_capacity = left.tokens.len() + right.tokens.len();
        let mut merged_tokens = UInt32Builder::with_capacity(item_capacity);
        let mut merged_bitmaps = Vec::with_capacity(left.bitmaps.len() + right.bitmaps.len());

        let mut left_tokens = left.tokens.values().iter().copied();
        let mut left_bitmaps = left.bitmaps.into_iter();
        let mut right_tokens = right.tokens.values().iter().copied();
        let mut right_bitmaps = right.bitmaps.into_iter();

        let mut left_token = left_tokens.next();
        let mut left_bitmap = left_bitmaps.next();
        let mut right_token = right_tokens.next();
        let mut right_bitmap = right_bitmaps.next();

        while left_token.is_some() && right_token.is_some() {
            let left_token_val = left_token.unwrap();
            let right_token_val = right_token.unwrap();
            match left_token_val.cmp(&right_token_val) {
                std::cmp::Ordering::Less => {
                    merged_tokens.append_value(left_token_val);
                    merged_bitmaps.push(left_bitmap.unwrap());
                    left_token = left_tokens.next();
                    left_bitmap = left_bitmaps.next();
                }
                std::cmp::Ordering::Greater => {
                    merged_tokens.append_value(right_token_val);
                    merged_bitmaps.push(right_bitmap.unwrap());
                    right_token = right_tokens.next();
                    right_bitmap = right_bitmaps.next();
                }
                std::cmp::Ordering::Equal => {
                    merged_tokens.append_value(left_token_val);
                    merged_bitmaps.push(left_bitmap.unwrap() | &right_bitmap.unwrap());
                    left_token = left_tokens.next();
                    left_bitmap = left_bitmaps.next();
                    right_token = right_tokens.next();
                    right_bitmap = right_bitmaps.next();
                }
            }
        }

        let collect_remaining = |cur_token, tokens, cur_bitmap, bitmaps| {
            let tokens = UInt32Array::from_iter_values(once(cur_token).chain(tokens));
            let bitmaps = once(cur_bitmap).chain(bitmaps).collect::<Vec<_>>();
            NGramIndexSpillState { tokens, bitmaps }
        };

        if let Some(left_token) = left_token {
            *left_opt = Some(collect_remaining(
                left_token,
                left_tokens,
                left_bitmap.unwrap(),
                left_bitmaps,
            ));
        } else {
            *left_opt = None;
        }
        if let Some(right_token) = right_token {
            *right_opt = Some(collect_remaining(
                right_token,
                right_tokens,
                right_bitmap.unwrap(),
                right_bitmaps,
            ));
        } else {
            *right_opt = None;
        }

        NGramIndexSpillState {
            tokens: merged_tokens.finish(),
            bitmaps: merged_bitmaps,
        }
    }

    async fn merge_spill_streams(
        mut left_stream: impl Stream<Item = Result<NGramIndexSpillState>> + Unpin,
        mut right_stream: impl Stream<Item = Result<NGramIndexSpillState>> + Unpin,
        writer: &mut dyn IndexWriter,
    ) -> Result<IndexFile> {
        let mut left_state = left_stream.try_next().await?;
        let mut right_state = right_stream.try_next().await?;

        while left_state.is_some() || right_state.is_some() {
            if left_state.is_none() {
                // Left is done, full drain right
                let state = right_state.take().expect_ok()?;
                writer.write_record_batch(state.try_into_batch()?).await?;
                while let Some(state) = right_stream.try_next().await? {
                    writer.write_record_batch(state.try_into_batch()?).await?;
                }
            } else if right_state.is_none() {
                // Right is done, full drain left
                let state = left_state.take().expect_ok()?;
                writer.write_record_batch(state.try_into_batch()?).await?;
                while let Some(state) = left_stream.try_next().await? {
                    writer.write_record_batch(state.try_into_batch()?).await?;
                }
            } else {
                // There is a batch from both left and right.  Need to merge them
                let merged = Self::merge_spill_states(&mut left_state, &mut right_state);
                writer.write_record_batch(merged.try_into_batch()?).await?;
                if left_state.is_none() {
                    left_state = left_stream.try_next().await?;
                }
                if right_state.is_none() {
                    right_state = right_stream.try_next().await?;
                }
            }
        }

        writer.finish().await
    }

    async fn merge_spill_files(
        spill_store: Arc<dyn IndexStore>,
        index_of_left: usize,
        index_of_right: usize,
        output_index: usize,
    ) -> Result<()> {
        // We fully load the small file into memory and then stream the large file
        info!(
            "Merge spill files {} and {} into {}",
            index_of_left, index_of_right, output_index
        );

        let mut writer = spill_store
            .new_index_file(&Self::spill_filename(output_index), POSTINGS_SCHEMA.clone())
            .await?;

        let (left_stream, right_stream) = futures::try_join!(
            Self::stream_spill(spill_store.clone(), index_of_left),
            Self::stream_spill(spill_store.clone(), index_of_right)
        )?;

        Self::merge_spill_streams(left_stream, right_stream, writer.as_mut()).await?;

        spill_store
            .delete_index_file(&Self::spill_filename(index_of_left))
            .await?;
        spill_store
            .delete_index_file(&Self::spill_filename(index_of_right))
            .await?;

        Ok(())
    }

    // Can potentially parallelize in the future if this step becomes a bottleneck
    //
    // We can also merge in a more balanced fashion (e.g. binary tree) to reduce the size of
    // intermediate files
    //
    // Note: worker indices start at 1 and not 0 (hence all the +1's)
    async fn merge_spills(&mut self, mut spill_files: Vec<usize>) -> Result<usize> {
        info!(
            "Merging {} index files into one combined index",
            spill_files.len()
        );

        let mut spill_counter = spill_files.iter().max().expect_ok()? + 1;
        while spill_files.len() > 1 {
            let mut new_spills = Vec::with_capacity(spill_files.len() / 2);
            while spill_files.len() >= 2 {
                let left = spill_files.pop().expect_ok()?;
                let right = spill_files.pop().expect_ok()?;
                new_spills.push(tokio::spawn(Self::merge_spill_files(
                    self.spill_store.clone(),
                    left,
                    right,
                    spill_counter + new_spills.len(),
                )));
            }
            for i in 0..new_spills.len() {
                spill_files.push(spill_counter + i);
            }
            spill_counter += new_spills.len();
            futures::future::try_join_all(new_spills).await?;
        }

        spill_files.pop().expect_ok()
    }

    async fn merge_old_index(
        &mut self,
        new_data_num: usize,
        old_index: Arc<dyn IndexStore>,
    ) -> Result<usize> {
        info!("Merging old index into new index");
        let final_num = new_data_num + 1;

        let mut writer = self
            .spill_store
            .new_index_file(&Self::spill_filename(final_num), POSTINGS_SCHEMA.clone())
            .await?;

        let left_stream = Self::stream_spill(self.spill_store.clone(), new_data_num).await?;
        let old_reader = old_index.open_index_file(POSTINGS_FILENAME).await?;
        let right_stream = Self::stream_spill_reader(old_reader).await?;

        Self::merge_spill_streams(left_stream, right_stream, writer.as_mut()).await?;

        self.spill_store
            .delete_index_file(&Self::spill_filename(new_data_num))
            .await?;

        Ok(final_num)
    }

    pub async fn write_index(
        mut self,
        store: &dyn IndexStore,
        spill_files: Vec<usize>,
        old_index: Option<Arc<dyn IndexStore>>,
    ) -> Result<IndexFile> {
        let mut writer = store
            .new_index_file(POSTINGS_FILENAME, POSTINGS_SCHEMA.clone())
            .await?;

        if spill_files.is_empty() {
            if let Some(old_index) = old_index {
                // An update with no new data, just copy the old index to the new store
                return old_index.copy_index_file(POSTINGS_FILENAME, store).await;
            } else {
                // Training an index with no data, make an empty index
                let mut writer = store
                    .new_index_file(POSTINGS_FILENAME, POSTINGS_SCHEMA.clone())
                    .await?;
                return writer.finish_with_metadata(self.write_metadata()).await;
            }
        }

        let mut index_to_copy = self.merge_spills(spill_files).await?;

        if let Some(old_index) = old_index {
            index_to_copy = self.merge_old_index(index_to_copy, old_index).await?;
        }

        let reader = self
            .spill_store
            .open_index_file(&Self::spill_filename(index_to_copy))
            .await?;

        let num_rows = reader.num_rows();
        let mut offset = 0;

        while offset < num_rows {
            let batch_size = std::cmp::min(num_rows - offset, 64);
            let batch = reader.read_range(offset..offset + batch_size, None).await?;
            writer.write_record_batch(batch).await?;
            offset += batch_size;
        }

        writer.finish_with_metadata(self.write_metadata()).await
    }
}

#[derive(Debug, Default)]
pub struct NGramIndexPlugin;

// Carries parsed NGram index parameters from request parsing into training.
struct NGramTrainingRequest {
    criteria: TrainingCriteria,
    params: NGramIndexParams,
}

impl NGramTrainingRequest {
    fn new(params: NGramIndexParams) -> Self {
        Self {
            criteria: TrainingCriteria::new(TrainingOrdering::None).with_row_id(),
            params,
        }
    }
}

impl TrainingRequest for NGramTrainingRequest {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

impl NGramIndexPlugin {
    pub async fn train_ngram_index(
        batches_source: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
    ) -> Result<IndexFile> {
        let mut builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions::default())?;

        let spill_files = builder.train(batches_source).await?;

        builder.write_index(index_store, spill_files, None).await
    }
}

#[async_trait]
impl ScalarIndexPlugin for NGramIndexPlugin {
    fn name(&self) -> &str {
        "NGram"
    }

    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        if !matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
            return Err(Error::invalid_input_source(format!(
                "A ngram index can only be created on a Utf8 or LargeUtf8 field.  Column has type {:?}",
                field.data_type()
            )
            .into()));
        }
        let parsed_params = if params.trim().is_empty() {
            NGramIndexParams::default()
        } else {
            serde_json::from_str(params).map_err(|e| {
                Error::invalid_input_source(format!("Invalid NGram index params: {e}").into())
            })?
        };
        Ok(Box::new(NGramTrainingRequest::new(parsed_params)))
    }

    fn provides_exact_answer(&self) -> bool {
        false
    }

    fn version(&self) -> u32 {
        NGRAM_SPARSE_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        Some(Box::new(TextQueryParser::new(
            index_name,
            self.name().to_string(),
            // needs_recheck: ngram results are an inexact candidate superset.
            true,
            // supports_regex: the ngram index can answer regex queries.
            true,
        )))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        fragment_ids: Option<Vec<u32>>,
        _progress: Arc<dyn crate::progress::IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        if fragment_ids.is_some() {
            return Err(Error::invalid_input_source(
                "NGram index does not support fragment training".into(),
            ));
        }

        let request = request
            .as_any()
            .downcast_ref::<NGramTrainingRequest>()
            .ok_or_else(|| Error::internal("Invalid NGram training request type"))?;
        let mut builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions {
            tokenization: request.params.tokenization,
            ..NGramIndexBuilderOptions::default()
        })?;

        let spill_files = builder.train(data).await?;
        let file = builder.write_index(index_store, spill_files, None).await?;
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::NGramIndexDetails::default())
                .unwrap(),
            index_version: request.params.tokenization.index_version(),
            files: vec![file],
        })
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(NGramIndex::load(index_store, frag_reuse_index, cache).await? as Arc<dyn ScalarIndex>)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        sync::Arc,
    };

    use arrow::datatypes::UInt64Type;
    use arrow_array::{Array, BinaryArray, RecordBatch, StringArray, UInt32Array, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        execution::SendableRecordBatchStream, physical_plan::stream::RecordBatchStreamAdapter,
    };
    use datafusion_common::DataFusionError;
    use futures::{TryStreamExt, stream};
    use itertools::Itertools;
    use lance_core::{ROW_ID, cache::LanceCache, utils::tempfile::TempDir};
    use lance_datagen::{BatchCount, ByteCount, RowCount};
    use lance_io::object_store::ObjectStore;
    use lance_select::RowAddrTreeMap;
    use lance_tokenizer::TextAnalyzer;

    use crate::progress::NoopIndexBuildProgress;
    use crate::scalar::{
        IndexStore, ScalarIndex, SearchResult, TextQuery,
        lance_format::LanceIndexStore,
        ngram::{NGramIndex, NGramIndexBuilder, NGramIndexBuilderOptions, NGramIndexPlugin},
    };
    use crate::{
        metrics::NoOpMetricsCollector,
        scalar::registry::{ScalarIndexPlugin, VALUE_COLUMN_NAME},
    };

    use super::{
        NGRAM_N, NGRAM_SPARSE_INDEX_VERSION, NGRAM_TOKENIZER, NGRAM_TRIGRAM_INDEX_VERSION,
        NGramTokenization, POSTING_LIST_COL, POSTINGS_FILENAME, SPARSE_MAX_NGRAM_LEN,
        SPARSE_MIN_NGRAM_LEN, TOKENS_COL, ngram_to_token, query_tokens_for_text,
        sparse_index_token_visitor, sparse_ngram_to_token, sparse_spans_for_run, tokenize_visitor,
    };
    use roaring::RoaringTreemap;

    fn collect_tokens(analyzer: &TextAnalyzer, text: &str) -> Vec<String> {
        let mut tokens = Vec::with_capacity(text.len() * 3);
        tokenize_visitor(analyzer, text, |token| tokens.push(token.to_owned()));
        tokens
    }

    #[test]
    fn test_tokenizer() {
        let tokenizer = NGRAM_TOKENIZER.clone();

        // ASCII folding
        let tokens = collect_tokens(&tokenizer, "café");
        assert_eq!(
            tokens,
            vec!["caf", "afe"] // spellchecker:disable-line
        );

        // Allow numbers
        let tokens = collect_tokens(&tokenizer, "a1b2");
        assert_eq!(tokens, vec!["a1b", "1b2"]);

        // Remove symbols and UTF-8 that doesn't map to characters
        let tokens = collect_tokens(&tokenizer, "abc👍b!c24");

        assert_eq!(tokens, vec!["abc", "c24"]);

        let tokens = collect_tokens(&tokenizer, "anstoß");

        assert_eq!(tokens, vec!["ans", "nst", "sto", "tos", "oss"]);

        // Lower casing
        let tokens = collect_tokens(&tokenizer, "ABC");
        assert_eq!(tokens, vec!["abc"]);

        // Duplicate tokens
        let tokens = collect_tokens(&tokenizer, "ababab");
        // Confirming that the tokenizer doesn't deduplicate tokens (this can be taken into consideration
        // when training the index)
        assert_eq!(
            tokens,
            vec!["aba", "bab", "aba", "bab"] // spellchecker:disable-line
        );
    }

    #[test]
    fn test_sparse_span_generation_is_bounded() {
        let run = "abcdefghijklmnopqrstuvwxyz0123456789".repeat(20);
        let spans = sparse_spans_for_run(&run);

        assert!(!spans.is_empty());
        assert!(spans.iter().all(|span| {
            let len = span.end - span.start;
            (SPARSE_MIN_NGRAM_LEN..=SPARSE_MAX_NGRAM_LEN).contains(&len)
        }));
        assert!(spans.len() <= 2 * run.len().saturating_sub(1));
    }

    #[test]
    fn test_sparse_covering_tokens_for_long_query_is_bounded() {
        let literal = "abcdefghijklmnopqrstuvwxyz0123456789".repeat(256);
        let spans = sparse_spans_for_run(&literal);
        let sparse_tokens = query_tokens_for_text(NGramTokenization::Sparse, &literal);
        let mut index_tokens = HashSet::new();
        sparse_index_token_visitor(&literal, |token| {
            index_tokens.insert(token);
        });

        assert!(!sparse_tokens.is_empty());
        assert!(sparse_tokens.len() <= spans.len());
        assert!(sparse_tokens.len() <= literal.len() / SPARSE_MIN_NGRAM_LEN + 1);
        assert!(
            sparse_tokens
                .iter()
                .all(|token| index_tokens.contains(token))
        );
    }

    #[test]
    fn test_sparse_covering_tokens_matches_stack_builder() {
        let literal = "sparsemarkerabcdefghijklmnopqrstuvwx";
        let sparse_tokens = query_tokens_for_text(NGramTokenization::Sparse, literal);
        let expected = [
            "spa",
            "par",
            "arsema",
            "mark",
            "rkerabcdefg",
            "fghi",
            "hijk",
            "jklmno",
            "nopq",
            "pqr",
            "qrstuvw",
            "vwx",
        ]
        .into_iter()
        .map(|ngram| sparse_ngram_to_token(ngram.as_bytes()))
        .collect::<HashSet<_>>();

        assert_eq!(sparse_tokens.into_iter().collect::<HashSet<_>>(), expected);
    }

    #[test]
    fn test_ngram_rejects_unknown_params() {
        let plugin = NGramIndexPlugin;
        let field = Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false);
        let Err(err) = plugin.new_training_request(r#"{"tokenisation":"sparse"}"#, &field) else {
            panic!("expected unknown ngram params to fail");
        };

        assert!(
            err.to_string().contains("Invalid NGram index params"),
            "{err}"
        );
    }

    async fn do_train(
        mut builder: NGramIndexBuilder,
        data: SendableRecordBatchStream,
    ) -> (NGramIndex, Arc<TempDir>) {
        let spill_files = builder.train(data).await.unwrap();

        let tmpdir = Arc::new(TempDir::default());
        let test_store = LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        );

        builder
            .write_index(&test_store, spill_files, None)
            .await
            .unwrap();

        (
            NGramIndex::from_store(Arc::new(test_store), None, &LanceCache::no_cache())
                .await
                .unwrap(),
            tmpdir,
        )
    }

    async fn get_posting_list_for_trigram(index: &NGramIndex, trigram: &str) -> Vec<u64> {
        let token = ngram_to_token(trigram, 3);
        let row_offset = index.tokens[&token];
        let list = index
            .list_reader
            .ngram_list(row_offset, &NoOpMetricsCollector)
            .await
            .unwrap();
        list.bitmap.iter().sorted().collect()
    }

    async fn get_null_posting_list(index: &NGramIndex) -> Vec<u64> {
        let row_offset = index.tokens[&0];
        let list = index
            .list_reader
            .ngram_list(row_offset, &NoOpMetricsCollector)
            .await
            .unwrap();
        list.bitmap.iter().sorted().collect()
    }

    #[test_log::test(tokio::test)]
    async fn test_basic_ngram_index() {
        let data = StringArray::from_iter_values([
            "cat",
            "dog",
            "cat dog",
            "dog cat",
            "elephant",
            "mouse",
            "rhino",
            "giraffe",
            "rhinos nose",
        ]);
        let row_ids = UInt64Array::from_iter_values((0..data.len()).map(|i| i as u64));
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();
        let data = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        let builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions::default()).unwrap();

        let (index, _tmpdir) = do_train(builder, data).await;
        assert_eq!(index.tokens.len(), 21);

        // Basic search
        let res = index
            .search(
                &TextQuery::StringContains("cat".to_string()),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();

        let expected = SearchResult::at_most(RowAddrTreeMap::from_iter([0, 2, 3]));

        assert_eq!(expected, res);

        // Whitespace in query
        let res = index
            .search(
                &TextQuery::StringContains("nos nos".to_string()),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let expected = SearchResult::at_most(RowAddrTreeMap::from_iter([8]));
        assert_eq!(expected, res);

        // No matches
        let res = index
            .search(
                &TextQuery::StringContains("tdo".to_string()),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let expected = SearchResult::exact(RowAddrTreeMap::new());
        assert_eq!(expected, res);

        // False positive
        let res = index
            .search(
                &TextQuery::StringContains("inose".to_string()),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let expected = SearchResult::at_most(RowAddrTreeMap::from_iter([8]));
        assert_eq!(expected, res);

        // Too short, don't know anything
        let res = index
            .search(
                &TextQuery::StringContains("ab".to_string()),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let expected = SearchResult::at_least(RowAddrTreeMap::new());
        assert_eq!(expected, res);

        // One short string but we still get at least one trigram, this is ok
        let res = index
            .search(
                &TextQuery::StringContains("no nos".to_string()),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let expected = SearchResult::at_most(RowAddrTreeMap::from_iter([8]));
        assert_eq!(expected, res);
    }

    #[test_log::test(tokio::test)]
    async fn test_ngram_regex_search() {
        // Same corpus as test_basic_ngram_index.
        let data = StringArray::from_iter_values([
            "cat",         // 0
            "dog",         // 1
            "cat dog",     // 2
            "dog cat",     // 3
            "elephant",    // 4
            "mouse",       // 5
            "rhino",       // 6
            "giraffe",     // 7
            "rhinos nose", // 8
        ]);
        let row_ids = UInt64Array::from_iter_values((0..data.len()).map(|i| i as u64));
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();
        let data = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        let builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions::default()).unwrap();
        let (index, _tmpdir) = do_train(builder, data).await;

        async fn search(index: &NGramIndex, pattern: &str) -> SearchResult {
            index
                .search(
                    &TextQuery::Regex(pattern.to_string()),
                    &NoOpMetricsCollector,
                )
                .await
                .unwrap()
        }

        // A plain literal yields the same candidates as contains("cat").
        assert_eq!(
            search(&index, "cat").await,
            SearchResult::at_most(RowAddrTreeMap::from_iter([0, 2, 3]))
        );

        // Alternation -> union of each branch's rows.
        assert_eq!(
            search(&index, "(cat|dog)").await,
            SearchResult::at_most(RowAddrTreeMap::from_iter([0, 1, 2, 3]))
        );

        // AND across `.*`: must contain both the `rhino` and `nose` trigrams, so
        // row 6 ("rhino") is correctly excluded and only row 8 survives.
        assert_eq!(
            search(&index, "rhino.*nose").await,
            SearchResult::at_most(RowAddrTreeMap::from_iter([8]))
        );

        // No derivable trigram -> recheck everything.
        assert_eq!(
            search(&index, "a.b").await,
            SearchResult::at_least(RowAddrTreeMap::new())
        );

        // A trigram that is absent from the index -> empty candidate set.
        assert_eq!(
            search(&index, "zzz").await,
            SearchResult::at_most(RowAddrTreeMap::new())
        );
    }

    #[test_log::test(tokio::test)]
    async fn test_ngram_regex_search_nulls() {
        // Rows: cat(0), dog(1), NULL(2), NULL(3), cat dog(4).
        let data = simple_data_with_nulls();
        let builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions::default()).unwrap();
        let (index, _tmpdir) = do_train(builder, data).await;

        // The NULL rows (2, 3) must never appear in the candidate set.
        let res = index
            .search(&TextQuery::Regex("cat".to_string()), &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            res,
            SearchResult::at_most(RowAddrTreeMap::from_iter([0, 4]))
        );

        let res = index
            .search(
                &TextQuery::Regex("(cat|dog)".to_string()),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        assert_eq!(
            res,
            SearchResult::at_most(RowAddrTreeMap::from_iter([0, 1, 4]))
        );
    }

    #[test]
    fn test_sparse_covering_tokens_are_index_tokens() {
        let literal = "commonmarkerabcdefghijklmnopqrstuvwx";
        let trigram_tokens = query_tokens_for_text(NGramTokenization::Trigram, literal);
        let sparse_tokens = query_tokens_for_text(NGramTokenization::Sparse, literal);
        let mut index_tokens = HashSet::new();
        sparse_index_token_visitor(literal, |token| {
            index_tokens.insert(token);
        });

        assert!(!sparse_tokens.is_empty());
        assert!(sparse_tokens.len() <= trigram_tokens.len());
        assert!(
            sparse_tokens
                .iter()
                .all(|token| index_tokens.contains(token))
        );
    }

    #[test_log::test(tokio::test)]
    async fn test_sparse_ngram_search_and_metadata() {
        let data = StringArray::from_iter_values([
            "alpha commonmarkerabcdefghijklmnopqrstuvwx omega",
            "alpha commonmarkerabcdefghijklmno omega",
            "totally unrelated row",
        ]);
        let row_ids = UInt64Array::from_iter_values((0..data.len()).map(|i| i as u64));
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();
        let data = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        let builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions {
            tokenization: NGramTokenization::Sparse,
            ..NGramIndexBuilderOptions::default()
        })
        .unwrap();
        let (index, _tmpdir) = do_train(builder, data).await;

        assert_eq!(index.tokenization, NGramTokenization::Sparse);
        assert_eq!(
            index
                .search(
                    &TextQuery::StringContains("commonmarkerabcdefghijklmnopqrstuvwx".to_string()),
                    &NoOpMetricsCollector,
                )
                .await
                .unwrap(),
            SearchResult::at_most(RowAddrTreeMap::from_iter([0]))
        );
        assert_eq!(
            index
                .search(
                    &TextQuery::Regex("commonmarkerabcdefghijklmnopqrstuvwx".to_string()),
                    &NoOpMetricsCollector,
                )
                .await
                .unwrap(),
            SearchResult::at_most(RowAddrTreeMap::from_iter([0]))
        );
        assert_eq!(
            index
                .search(
                    &TextQuery::Regex("commonmarker.*omega".to_string()),
                    &NoOpMetricsCollector,
                )
                .await
                .unwrap(),
            SearchResult::at_most(RowAddrTreeMap::from_iter([0, 1]))
        );
    }

    #[test_log::test(tokio::test)]
    async fn test_ngram_loads_postings_without_tokenization_metadata() {
        let tmpdir = Arc::new(TempDir::default());
        let test_store = LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        );
        let old_schema = Arc::new(Schema::new(vec![
            Field::new(TOKENS_COL, DataType::UInt32, true),
            Field::new(POSTING_LIST_COL, DataType::Binary, false),
        ]));
        let token = ngram_to_token("cat", NGRAM_N);
        let bitmap = RoaringTreemap::from_iter([42]);
        let mut bitmap_bytes = Vec::new();
        bitmap.serialize_into(&mut bitmap_bytes).unwrap();
        let batch = RecordBatch::try_new(
            old_schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values([token])),
                Arc::new(BinaryArray::from_iter_values([bitmap_bytes])),
            ],
        )
        .unwrap();
        let mut writer = test_store
            .new_index_file(POSTINGS_FILENAME, old_schema)
            .await
            .unwrap();
        writer.write_record_batch(batch).await.unwrap();
        writer.finish().await.unwrap();

        let index = NGramIndex::from_store(Arc::new(test_store), None, &LanceCache::no_cache())
            .await
            .unwrap();
        assert_eq!(index.tokenization, NGramTokenization::Trigram);
        assert_eq!(
            index
                .search(&TextQuery::Regex("cat".to_string()), &NoOpMetricsCollector)
                .await
                .unwrap(),
            SearchResult::at_most(RowAddrTreeMap::from_iter([42]))
        );
    }

    async fn train_created_index(params: &str) -> u32 {
        let data = StringArray::from_iter_values(["alpha sparsemarkerabcdefgh"]);
        let row_ids = UInt64Array::from_iter_values([0]);
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();
        let data = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        let plugin = NGramIndexPlugin;
        let field = Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false);
        let request = plugin.new_training_request(params, &field).unwrap();
        let tmpdir = Arc::new(TempDir::default());
        let test_store = LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        );

        plugin
            .train_index(
                data,
                &test_store,
                request,
                None,
                Arc::new(NoopIndexBuildProgress),
            )
            .await
            .unwrap()
            .index_version
    }

    #[test_log::test(tokio::test)]
    async fn test_ngram_created_index_versions() {
        assert_eq!(train_created_index("").await, NGRAM_TRIGRAM_INDEX_VERSION);
        assert_eq!(
            train_created_index(r#"{"tokenization":"sparse"}"#).await,
            NGRAM_SPARSE_INDEX_VERSION
        );
    }

    fn test_data_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Utf8, true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]))
    }

    fn simple_data_with_nulls() -> SendableRecordBatchStream {
        let data = StringArray::from_iter(&[Some("cat"), Some("dog"), None, None, Some("cat dog")]);
        let row_ids = UInt64Array::from_iter_values((0..data.len()).map(|i| i as u64));
        let schema = test_data_schema();
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ))
    }

    fn data_from_values(values: &[Option<&str>], first_row_id: u64) -> SendableRecordBatchStream {
        let data = StringArray::from_iter(values.iter().copied());
        let row_ids =
            UInt64Array::from_iter_values((0..data.len()).map(|i| first_row_id + i as u64));
        let schema = test_data_schema();
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ))
    }

    #[test_log::test(tokio::test)]
    async fn test_ngram_nulls() {
        let data = simple_data_with_nulls();

        let builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions::default()).unwrap();

        let (index, _tmpdir) = do_train(builder, data).await;
        assert_eq!(index.tokens.len(), 3);

        let res = index
            .search(
                &TextQuery::StringContains("cat".to_string()),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let expected = SearchResult::at_most(RowAddrTreeMap::from_iter([0, 4]));
        assert_eq!(expected, res);

        let null_posting_list = get_null_posting_list(&index).await;
        assert_eq!(null_posting_list, vec![2, 3]);

        // TODO: Support IS NULL queries
    }

    fn empty_data() -> SendableRecordBatchStream {
        Box::pin(RecordBatchStreamAdapter::new(
            test_data_schema(),
            stream::empty::<lance_core::error::DataFusionResult<RecordBatch>>(),
        ))
    }

    #[test_log::test(tokio::test)]
    async fn test_train_empty() {
        let builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions::default()).unwrap();

        let (index, _tmpdir) = do_train(builder, empty_data()).await;
        assert_eq!(index.tokens.len(), 0);
    }

    #[test_log::test(tokio::test)]
    async fn test_update_empty() {
        let data = simple_data_with_nulls();

        let builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions::default()).unwrap();
        let (index, _tmpdir) = do_train(builder, empty_data()).await;

        let new_tmpdir = Arc::new(TempDir::default());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            new_tmpdir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));

        index.update(data, test_store.as_ref(), None).await.unwrap();

        let index = NGramIndex::from_store(test_store, None, &LanceCache::no_cache())
            .await
            .unwrap();
        assert_eq!(index.tokens.len(), 3);
    }

    #[test_log::test(tokio::test)]
    async fn test_sparse_update_preserves_tokenization_and_version() {
        let literal = "commonmarkerabcdefghijklmnopqrstuvwx";
        let data = data_from_values(
            &[
                Some("alpha commonmarkerabcdefghijklmnopqrstuvwx omega"),
                Some("totally unrelated row"),
            ],
            0,
        );
        let builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions {
            tokenization: NGramTokenization::Sparse,
            ..NGramIndexBuilderOptions::default()
        })
        .unwrap();
        let (index, _tmpdir) = do_train(builder, data).await;

        let new_tmpdir = Arc::new(TempDir::default());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            new_tmpdir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));
        let new_data = data_from_values(
            &[Some("updated commonmarkerabcdefghijklmnopqrstuvwx row")],
            100,
        );

        let created = index
            .update(new_data, test_store.as_ref(), None)
            .await
            .unwrap();
        assert_eq!(created.index_version, NGRAM_SPARSE_INDEX_VERSION);

        let index = NGramIndex::from_store(test_store, None, &LanceCache::no_cache())
            .await
            .unwrap();
        assert_eq!(index.tokenization, NGramTokenization::Sparse);
        assert_eq!(
            index
                .search(
                    &TextQuery::StringContains(literal.to_string()),
                    &NoOpMetricsCollector,
                )
                .await
                .unwrap(),
            SearchResult::at_most(RowAddrTreeMap::from_iter([0, 100]))
        );
    }

    async fn row_ids_in_index(index: &NGramIndex) -> Vec<u64> {
        let mut row_ids = HashSet::new();
        for row_offset in index.tokens.values() {
            let list = index
                .list_reader
                .ngram_list(*row_offset, &NoOpMetricsCollector)
                .await
                .unwrap();
            row_ids.extend(list.bitmap.iter());
        }
        row_ids.into_iter().sorted().collect()
    }

    #[test_log::test(tokio::test)]
    async fn test_ngram_index_remap() {
        let data = simple_data_with_nulls();
        let builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions::default()).unwrap();
        let (index, _tmpdir) = do_train(builder, data).await;

        let row_ids = row_ids_in_index(&index).await;
        assert_eq!(row_ids, vec![0, 1, 2, 3, 4]);

        let new_tmpdir = Arc::new(TempDir::default());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            new_tmpdir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));

        let remapping = HashMap::from([(2, Some(100)), (3, None), (4, Some(101))]);
        index.remap(&remapping, test_store.as_ref()).await.unwrap();

        let index = NGramIndex::from_store(test_store, None, &LanceCache::no_cache())
            .await
            .unwrap();
        let row_ids = row_ids_in_index(&index).await;
        assert_eq!(row_ids, vec![0, 1, 100, 101]);

        let null_posting_list = get_null_posting_list(&index).await;
        assert_eq!(null_posting_list, vec![100]);
    }

    #[test_log::test(tokio::test)]
    async fn test_sparse_remap_preserves_tokenization_and_version() {
        let literal = "commonmarkerabcdefghijklmnopqrstuvwx";
        let data = data_from_values(
            &[
                Some("alpha commonmarkerabcdefghijklmnopqrstuvwx omega"),
                Some("totally unrelated row"),
            ],
            0,
        );
        let builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions {
            tokenization: NGramTokenization::Sparse,
            ..NGramIndexBuilderOptions::default()
        })
        .unwrap();
        let (index, _tmpdir) = do_train(builder, data).await;

        let new_tmpdir = Arc::new(TempDir::default());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            new_tmpdir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));

        let remapping = HashMap::from([(0, Some(10)), (1, None)]);
        let created = index.remap(&remapping, test_store.as_ref()).await.unwrap();
        assert_eq!(created.index_version, NGRAM_SPARSE_INDEX_VERSION);

        let index = NGramIndex::from_store(test_store, None, &LanceCache::no_cache())
            .await
            .unwrap();
        assert_eq!(index.tokenization, NGramTokenization::Sparse);
        assert_eq!(
            index
                .search(
                    &TextQuery::Regex(literal.to_string()),
                    &NoOpMetricsCollector,
                )
                .await
                .unwrap(),
            SearchResult::at_most(RowAddrTreeMap::from_iter([10]))
        );
    }

    #[test_log::test(tokio::test)]
    async fn test_ngram_index_merge() {
        let data = simple_data_with_nulls();
        let builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions::default()).unwrap();
        let (index, _tmpdir) = do_train(builder, data).await;

        let data = StringArray::from_iter(&[Some("giraffe"), Some("cat"), None]);
        let row_ids = UInt64Array::from_iter_values((0..data.len()).map(|i| i as u64 + 100));
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Utf8, true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();
        let data = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        let posting_list = get_posting_list_for_trigram(&index, "cat").await;
        assert_eq!(posting_list, vec![0, 4]);

        let new_tmpdir = Arc::new(TempDir::default());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            new_tmpdir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));

        index.update(data, test_store.as_ref(), None).await.unwrap();

        let index = NGramIndex::from_store(test_store, None, &LanceCache::no_cache())
            .await
            .unwrap();
        let row_ids = row_ids_in_index(&index).await;
        assert_eq!(row_ids, vec![0, 1, 2, 3, 4, 100, 101, 102]);

        let posting_list = get_posting_list_for_trigram(&index, "cat").await;
        assert_eq!(posting_list, vec![0, 4, 101]);

        let posting_list = get_posting_list_for_trigram(&index, "ffe").await;
        assert_eq!(posting_list, vec![100]);

        let posting_list = get_null_posting_list(&index).await;
        assert_eq!(posting_list, vec![2, 3, 102]);
    }

    #[test_log::test(tokio::test)]
    async fn test_ngram_index_with_spill() {
        let (data, schema) = lance_datagen::gen_batch()
            .col(
                VALUE_COLUMN_NAME,
                lance_datagen::array::rand_utf8(ByteCount::from(50), false),
            )
            .col(ROW_ID, lance_datagen::array::step::<UInt64Type>())
            .into_reader_stream(RowCount::from(128), BatchCount::from(32));

        let data = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            data.map_err(|arrow_err| DataFusionError::ArrowError(Box::new(arrow_err), None)),
        ));

        let builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions {
            tokens_per_spill: 100,
            ..NGramIndexBuilderOptions::default()
        })
        .unwrap();

        let (index, _tmpdir) = do_train(builder, data).await;

        assert_eq!(index.tokens.len(), 29012);
    }
}
