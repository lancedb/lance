// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::collections::BTreeMap;
use std::iter::once;
use std::time::Instant;
use std::{collections::HashMap, sync::Arc};

use super::btree::TrainingSource;
use super::lance_format::LanceIndexStore;
use super::{
    AnyQuery, IndexReader, IndexStore, IndexWriter, MetricsCollector, ScalarIndex, SearchResult,
    TextQuery,
};
use crate::frag_reuse::FragReuseIndex;
use crate::metrics::NoOpMetricsCollector;
use crate::scalar::inverted::CACHE_SIZE;
use crate::vector::VectorIndex;
use crate::{Index, IndexType};
use arrow::array::{AsArray, UInt32Builder};
use arrow::datatypes::{UInt32Type, UInt64Type};
use arrow_array::{BinaryArray, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use futures::{stream, FutureExt, Stream, StreamExt, TryStreamExt};
use lance_arrow::iter_str_array;
use lance_core::cache::LanceCache;
use lance_core::error::LanceOptionExt;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::utils::tracing::{IO_TYPE_LOAD_SCALAR_PART, TRACE_IO_EVENTS};
use lance_core::Result;
use lance_core::{utils::mask::RowIdTreeMap, Error};
use lance_io::object_store::ObjectStore;
use log::info;
use moka::future::Cache;
use object_store::path::Path;
use roaring::{RoaringBitmap, RoaringTreemap};
use serde::Serialize;
use snafu::location;
use tantivy::tokenizer::TextAnalyzer;
use tempfile::{tempdir, TempDir};
use tracing::instrument;

const TOKENS_COL: &str = "tokens";
const POSTING_LIST_COL: &str = "posting_list";
const POSTINGS_FILENAME: &str = "ngram_postings.lance";

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
    TextAnalyzer::builder(tantivy::tokenizer::RawTokenizer::default())
        .filter(tantivy::tokenizer::LowerCaser)
        .filter(tantivy::tokenizer::AsciiFoldingFilter)
        .build()
});
/// Currently we ALWAYS use trigrams with ascii folding and lower casing.  We may want to make this configurable in the future.
pub static NGRAM_TOKENIZER: LazyLock<TextAnalyzer> = LazyLock::new(|| {
    TextAnalyzer::builder(tantivy::tokenizer::NgramTokenizer::all_ngrams(3, 3).unwrap())
        .filter(tantivy::tokenizer::AlphaNumOnlyFilter)
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

/// Basic stats about an ngram index
#[derive(Serialize)]
struct NGramStatistics {
    num_ngrams: usize,
}

/// The row ids that contain a given ngram
#[derive(Debug)]
struct NGramPostingList {
    bitmap: RoaringTreemap,
}

impl DeepSizeOf for NGramPostingList {
    fn deep_size_of_children(&self, _: &mut deepsize::Context) -> usize {
        self.bitmap.serialized_size()
    }
}

impl NGramPostingList {
    fn try_from_batch(batch: RecordBatch, fri: Option<Arc<FragReuseIndex>>) -> Result<Self> {
        let bitmap_bytes = batch.column(0).as_binary::<i32>().value(0);
        let mut bitmap =
            RoaringTreemap::deserialize_from(bitmap_bytes).map_err(|e| Error::Internal {
                message: format!("Error deserializing ngram list: {}", e),
                location: location!(),
            })?;
        if let Some(fri_ref) = fri.as_ref() {
            bitmap = fri_ref.remap_row_ids_roaring_tree_map(&bitmap);
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
    /// The cache key is the row_offset
    cache: Cache<u32, Arc<NGramPostingList>>,
    fri: Option<Arc<FragReuseIndex>>,
}

impl DeepSizeOf for NGramPostingListReader {
    fn deep_size_of_children(&self, _: &mut deepsize::Context) -> usize {
        self.cache.weighted_size() as usize
    }
}

impl std::fmt::Debug for NGramPostingListReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NGramListReader")
            .field("cache_entry_count", &self.cache.entry_count())
            .finish()
    }
}

impl NGramPostingListReader {
    #[instrument(level = "debug", skip(self, metrics))]
    pub async fn ngram_list(
        &self,
        row_offset: u32,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<NGramPostingList>> {
        self.cache
            .try_get_with(row_offset, async move {
                metrics.record_part_load();
                tracing::info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_SCALAR_PART, index_type="ngram", part_id=row_offset);
                let batch = self
                    .reader
                    .read_range(
                        row_offset as usize..row_offset as usize + 1,
                        Some(&[POSTING_LIST_COL]),
                    )
                    .await?;
                Result::Ok(Arc::new(NGramPostingList::try_from_batch(batch, self.fri.clone())?))
            })
            .await
            .map_err(|e| Error::io(e.to_string(), location!()))
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
    /// The reader for the posting lists
    list_reader: Arc<NGramPostingListReader>,
    /// The tokenizer used to tokenize text.  Note: not all tokenizers can be used with this index.  For
    /// example, a stemming tokenizer would not work well because "dozing" would stem to "doze" and if the
    /// search term is "zing" it would not match.  As a result, this tokenizer is not as configurable as the
    /// tokenizers used in an inverted index.
    tokenizer: TextAnalyzer,
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
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.tokens.deep_size_of_children(context) + self.list_reader.deep_size_of_children(context)
    }
}

impl NGramIndex {
    async fn from_store(
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let tokens = store.open_index_file(POSTINGS_FILENAME).await?;
        let tokens = tokens
            .read_range(0..tokens.num_rows(), Some(&[TOKENS_COL]))
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
            cache: Cache::builder()
                .max_capacity(*CACHE_SIZE as u64)
                .weigher(|_, posting: &Arc<NGramPostingList>| posting.deep_size_of() as u32)
                .build(),
            fri,
        });

        Ok(Self {
            io_parallelism: store.io_parallelism(),
            tokens: tokens_map,
            list_reader: posting_reader,
            tokenizer: NGRAM_TOKENIZER.clone(),
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
        Err(Error::InvalidInput {
            source: "NGramIndex is not a vector index".into(),
            location: location!(),
        })
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let ngram_stats = NGramStatistics {
            num_ngrams: self.tokens.len(),
        };
        serde_json::to_value(ngram_stats).map_err(|e| Error::Internal {
            message: format!("Error serializing statistics: {}", e),
            location: location!(),
        })
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
        let query =
            query
                .as_any()
                .downcast_ref::<TextQuery>()
                .ok_or_else(|| Error::InvalidInput {
                    source: "Query is not a TextQuery".into(),
                    location: location!(),
                })?;
        match query {
            TextQuery::StringContains(substr) => {
                if substr.len() < NGRAM_N {
                    // We know nothing on short searches, need to recheck all
                    return Ok(SearchResult::AtLeast(RowIdTreeMap::new()));
                }

                let mut row_offsets = Vec::with_capacity(substr.len() * 3);
                let mut missing = false;
                tokenize_visitor(&self.tokenizer, substr, |ngram| {
                    let token = ngram_to_token(ngram, NGRAM_N);
                    if let Some(row_offset) = self.tokens.get(&token) {
                        row_offsets.push(*row_offset);
                    } else {
                        missing = true;
                    }
                });
                // At least one token was missing, so we know there are zero results
                if missing {
                    return Ok(SearchResult::Exact(RowIdTreeMap::new()));
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
                Ok(SearchResult::AtMost(RowIdTreeMap::from(row_ids)))
            }
        }
    }

    fn can_answer_exact(&self, _: &dyn AnyQuery) -> bool {
        false
    }

    async fn load(store: Arc<dyn IndexStore>, fri: Option<Arc<FragReuseIndex>>) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        Ok(Arc::new(Self::from_store(store, fri).await?))
    }

    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
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

        writer.finish().await
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        let mut builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions::default())?;
        let spill_files = builder.train(new_data).await?;

        builder
            .write_index(dest_store, spill_files, Some(self.store.clone()))
            .await?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct NGramIndexBuilderOptions {
    tokens_per_spill: usize,
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
                RoaringTreemap::deserialize_from(bytes.expect_ok()?).map_err(|e| Error::Internal {
                    message: format!("Error deserializing ngram list: {}", e),
                    location: location!(),
                })
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

        let tmpdir = Arc::new(tempdir()?);
        let spill_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path())?,
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
            return Err(Error::InvalidInput {
                source: "Ngram index schema must have exactly two fields".into(),
                location: location!(),
            });
        }
        if *schema.field(0).data_type() != DataType::Utf8
            && *schema.field(0).data_type() != DataType::LargeUtf8
        {
            return Err(Error::InvalidInput {
                source: "First field in ngram index schema must be of type Utf8/LargeUtf8".into(),
                location: location!(),
            });
        }
        if *schema.field(1).data_type() != DataType::UInt64 {
            return Err(Error::InvalidInput {
                source: "Second field in ngram index schema must be of type UInt64".into(),
                location: location!(),
            });
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
        batch: RecordBatch,
        num_workers: usize,
    ) -> Vec<Vec<(u32, u64)>> {
        let text_iter = iter_str_array(batch.column(0));
        let row_id_col = batch.column(1).as_primitive::<UInt64Type>();
        // Guessing 1000 tokens per row to at least avoid some of the earlier allocations
        let mut partitions = vec![Vec::with_capacity(batch.num_rows() * 1000); num_workers];
        let divisor = (MAX_TOKEN - MIN_TOKEN) / num_workers;
        for (text, row_id) in text_iter.zip(row_id_col.values()) {
            if let Some(text) = text {
                tokenize_visitor(tokenizer, text, |token| {
                    let token = ngram_to_token(token, NGRAM_N);
                    let partition_id = (token as usize).saturating_sub(MIN_TOKEN) / divisor;
                    partitions[partition_id % num_workers].push((token, *row_id));
                });
            } else {
                partitions[0].push((0, *row_id));
            }
        }
        partitions
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
                std::future::ready(Ok(tokio::task::spawn(async move {
                    Ok(Self::tokenize_and_partition(&tokenizer, batch, num_workers))
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

        if left_token.is_some() {
            *left_opt = Some(collect_remaining(
                left_token.unwrap(),
                left_tokens,
                left_bitmap.unwrap(),
                left_bitmaps,
            ));
        } else {
            *left_opt = None;
        }
        if right_token.is_some() {
            *right_opt = Some(collect_remaining(
                right_token.unwrap(),
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
    ) -> Result<()> {
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
    ) -> Result<()> {
        let mut writer = store
            .new_index_file(POSTINGS_FILENAME, POSTINGS_SCHEMA.clone())
            .await?;

        if spill_files.is_empty() {
            if let Some(old_index) = old_index {
                // An update with no new data, just copy the old index to the new store
                old_index.copy_index_file(POSTINGS_FILENAME, store).await?;
            } else {
                // Training an index with no data, make an empty index
                let mut writer = store
                    .new_index_file(POSTINGS_FILENAME, POSTINGS_SCHEMA.clone())
                    .await?;
                writer.finish().await?;
            }
            return Ok(());
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

        writer.finish().await
    }
}

pub async fn train_ngram_index(
    data_source: Box<dyn TrainingSource + Send>,
    index_store: &dyn IndexStore,
) -> Result<()> {
    let batches_source = data_source.scan_unordered_chunks(4096).await?;
    let mut builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions::default())?;

    let spill_files = builder.train(batches_source).await?;

    builder.write_index(index_store, spill_files, None).await
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        sync::Arc,
    };

    use arrow::datatypes::UInt64Type;
    use arrow_array::{Array, RecordBatch, StringArray, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        execution::SendableRecordBatchStream, physical_plan::stream::RecordBatchStreamAdapter,
    };
    use datafusion_common::DataFusionError;
    use futures::{stream, TryStreamExt};
    use itertools::Itertools;
    use lance_core::{cache::LanceCache, utils::mask::RowIdTreeMap};
    use lance_datagen::{BatchCount, ByteCount, RowCount};
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;
    use tantivy::tokenizer::TextAnalyzer;
    use tempfile::{tempdir, TempDir};

    use crate::metrics::NoOpMetricsCollector;
    use crate::scalar::{
        lance_format::LanceIndexStore,
        ngram::{NGramIndex, NGramIndexBuilder, NGramIndexBuilderOptions},
        ScalarIndex, SearchResult, TextQuery,
    };

    use super::{ngram_to_token, tokenize_visitor, NGRAM_TOKENIZER};

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

    async fn do_train(
        mut builder: NGramIndexBuilder,
        data: SendableRecordBatchStream,
    ) -> (NGramIndex, Arc<TempDir>) {
        let spill_files = builder.train(data).await.unwrap();

        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        );

        builder
            .write_index(&test_store, spill_files, None)
            .await
            .unwrap();

        (
            NGramIndex::from_store(Arc::new(test_store), None)
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
            Field::new("values", DataType::Utf8, false),
            Field::new("row_ids", DataType::UInt64, false),
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

        let expected = SearchResult::AtMost(RowIdTreeMap::from_iter([0, 2, 3]));

        assert_eq!(expected, res);

        // Whitespace in query
        let res = index
            .search(
                &TextQuery::StringContains("nos nos".to_string()),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let expected = SearchResult::AtMost(RowIdTreeMap::from_iter([8]));
        assert_eq!(expected, res);

        // No matches
        let res = index
            .search(
                &TextQuery::StringContains("tdo".to_string()),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let expected = SearchResult::Exact(RowIdTreeMap::new());
        assert_eq!(expected, res);

        // False positive
        let res = index
            .search(
                &TextQuery::StringContains("inose".to_string()),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let expected = SearchResult::AtMost(RowIdTreeMap::from_iter([8]));
        assert_eq!(expected, res);

        // Too short, don't know anything
        let res = index
            .search(
                &TextQuery::StringContains("ab".to_string()),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let expected = SearchResult::AtLeast(RowIdTreeMap::new());
        assert_eq!(expected, res);

        // One short string but we still get at least one trigram, this is ok
        let res = index
            .search(
                &TextQuery::StringContains("no nos".to_string()),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let expected = SearchResult::AtMost(RowIdTreeMap::from_iter([8]));
        assert_eq!(expected, res);
    }

    fn test_data_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("values", DataType::Utf8, true),
            Field::new("row_ids", DataType::UInt64, false),
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
        let expected = SearchResult::AtMost(RowIdTreeMap::from_iter([0, 4]));
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

        let new_tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(new_tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        index.update(data, test_store.as_ref()).await.unwrap();

        let index = NGramIndex::from_store(test_store, None).await.unwrap();
        assert_eq!(index.tokens.len(), 3);
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

        let new_tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(new_tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        let remapping = HashMap::from([(2, Some(100)), (3, None), (4, Some(101))]);
        index.remap(&remapping, test_store.as_ref()).await.unwrap();

        let index = NGramIndex::from_store(test_store, None).await.unwrap();
        let row_ids = row_ids_in_index(&index).await;
        assert_eq!(row_ids, vec![0, 1, 100, 101]);

        let null_posting_list = get_null_posting_list(&index).await;
        assert_eq!(null_posting_list, vec![100]);
    }

    #[test_log::test(tokio::test)]
    async fn test_ngram_index_merge() {
        let data = simple_data_with_nulls();
        let builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions::default()).unwrap();
        let (index, _tmpdir) = do_train(builder, data).await;

        let data = StringArray::from_iter(&[Some("giraffe"), Some("cat"), None]);
        let row_ids = UInt64Array::from_iter_values((0..data.len()).map(|i| i as u64 + 100));
        let schema = Arc::new(Schema::new(vec![
            Field::new("values", DataType::Utf8, true),
            Field::new("row_ids", DataType::UInt64, false),
        ]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();
        let data = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        let posting_list = get_posting_list_for_trigram(&index, "cat").await;
        assert_eq!(posting_list, vec![0, 4]);

        let new_tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(new_tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        index.update(data, test_store.as_ref()).await.unwrap();

        let index = NGramIndex::from_store(test_store, None).await.unwrap();
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
        let (data, schema) = lance_datagen::gen()
            .col(
                "values",
                lance_datagen::array::rand_utf8(ByteCount::from(50), false),
            )
            .col("row_ids", lance_datagen::array::step::<UInt64Type>())
            .into_reader_stream(RowCount::from(128), BatchCount::from(32));

        let data = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            data.map_err(|arrow_err| DataFusionError::ArrowError(arrow_err, None)),
        ));

        let builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions {
            tokens_per_spill: 100,
        })
        .unwrap();

        let (index, _tmpdir) = do_train(builder, data).await;

        assert_eq!(index.tokens.len(), 29012);
    }
}
