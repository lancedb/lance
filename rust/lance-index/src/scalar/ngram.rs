// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::collections::{BTreeMap, VecDeque};
use std::iter::once;
use std::time::Instant;
use std::{collections::HashMap, sync::Arc};

use arrow::array::{AsArray, UInt32Builder};
use arrow::datatypes::{UInt32Type, UInt64Type};
use arrow_array::{BinaryArray, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use futures::{stream, FutureExt, Stream, StreamExt, TryStreamExt};
use lance_core::cache::FileMetadataCache;
use lance_core::error::LanceOptionExt;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
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

use crate::scalar::inverted::CACHE_SIZE;
use crate::vector::VectorIndex;
use crate::{Index, IndexType};

use super::btree::TrainingSource;
use super::lance_format::LanceIndexStore;
use super::{AnyQuery, IndexReader, IndexStore, IndexWriter, ScalarIndex, SearchResult, TextQuery};

const TOKENS_COL: &str = "tokens";
const POSTING_LIST_COL: &str = "posting_list";
const POSTINGS_FILENAME: &str = "ngram_postings.lance";

lazy_static::lazy_static! {
    pub static ref TOKENS_FIELD: Field = Field::new(TOKENS_COL, DataType::UInt32, true);
    pub static ref POSTINGS_FIELD: Field = Field::new(POSTING_LIST_COL, DataType::Binary, false);
    pub static ref POSTINGS_SCHEMA: SchemaRef = Arc::new(Schema::new(vec![TOKENS_FIELD.clone(), POSTINGS_FIELD.clone()]));
    pub static ref TEXT_PREPPER: TextAnalyzer = TextAnalyzer::builder(tantivy::tokenizer::RawTokenizer::default())
        .filter(tantivy::tokenizer::LowerCaser)
        .filter(tantivy::tokenizer::AsciiFoldingFilter)
        .build();
    /// Currently we ALWAYS use trigrams with ascii folding and lower casing.  We may want to make this configurable in the future.
    pub static ref NGRAM_TOKENIZER: TextAnalyzer = TextAnalyzer::builder(tantivy::tokenizer::NgramTokenizer::all_ngrams(3, 3).unwrap())
        .filter(tantivy::tokenizer::AlphaNumOnlyFilter)
        .build();
}

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

const ALPHA_SPAN: usize = b'z' as usize - b'0' as usize + 2;
const MAX_TOKEN: usize = ALPHA_SPAN.pow(2) + ALPHA_SPAN;
const MIN_TOKEN: usize = 0;
const NGRAM_N: usize = 3;

// There are 75 values between '0' (48) and 'z' (122) (inclusive)
// We add 1 for the NULL token
//
// "" => 0
// "?" => 76 * 2 * ?
// "?$" => 76 * 2 * ? + 76 * $
// "?$#" => 76 * 2 * ? + 76 * $ + #
//
// The ?,$,# represent the position in the alphabet (+1 to distinguish from NULL)
//
// Small strings get the larger multipliers because those ngrams are
// less likely to be unique and will have larger bitmaps.  We want to
// spread those out.
fn ngram_to_token(ngram: &str, ngram_length: usize) -> u32 {
    let mut token = 0;
    // Empty string will get 0
    for (idx, byte) in ngram.bytes().enumerate() {
        let pos = (byte - b'0') as u32 + 1;
        let mult = ALPHA_SPAN.pow(ngram_length as u32 - idx as u32 - 1) as u32;
        token += pos * mult;
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
    fn try_from_batch(batch: RecordBatch) -> Result<Self> {
        let bitmap_bytes = batch.column(0).as_binary::<i32>().value(0);
        let bitmap =
            RoaringTreemap::deserialize_from(bitmap_bytes).map_err(|e| Error::Internal {
                message: format!("Error deserializing ngram list: {}", e),
                location: location!(),
            })?;
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
    #[instrument(level = "debug", skip(self))]
    pub async fn ngram_list(&self, row_offset: u32) -> Result<Arc<NGramPostingList>> {
        self.cache
            .try_get_with(row_offset, async move {
                let batch = self
                    .reader
                    .read_range(
                        row_offset as usize..row_offset as usize + 1,
                        Some(&[POSTING_LIST_COL]),
                    )
                    .await?;
                Result::Ok(Arc::new(NGramPostingList::try_from_batch(batch)?))
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
    async fn from_store(store: &dyn IndexStore) -> Result<Self> {
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
        });

        Ok(Self {
            io_parallelism: store.io_parallelism(),
            tokens: tokens_map,
            list_reader: posting_reader,
            tokenizer: NGRAM_TOKENIZER.clone(),
        })
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

    fn index_type(&self) -> IndexType {
        IndexType::NGram
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let mut frag_ids = RoaringBitmap::new();
        for row_offset in self.tokens.values() {
            let list = self.list_reader.ngram_list(*row_offset).await?;
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
    async fn search(&self, query: &dyn AnyQuery) -> Result<SearchResult> {
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
                    let token = ngram_to_token(&ngram, NGRAM_N);
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
                        .map(|row_offset| self.list_reader.ngram_list(row_offset)),
                )
                .buffer_unordered(self.io_parallelism)
                .try_collect::<Vec<_>>()
                .await?;
                let list_refs = posting_lists.iter().map(|list| list.as_ref());
                let row_ids = NGramPostingList::intersect(list_refs);
                Ok(SearchResult::AtMost(RowIdTreeMap::from(row_ids)))
            }
        }
    }

    fn can_answer_exact(&self, _: &dyn AnyQuery) -> bool {
        false
    }

    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        Ok(Arc::new(Self::from_store(store.as_ref()).await?))
    }

    async fn remap(
        &self,
        _mapping: &HashMap<u64, Option<u64>>,
        _dest_store: &dyn IndexStore,
    ) -> Result<()> {
        todo!()
        // let mut builder = self.to_builder().await?;
        // let lists = std::mem::take(&mut builder.state.bitmaps);
        // let remapped_lists = lists
        //     .into_iter()
        //     .map(|list| {
        //         RoaringTreemap::from_iter(list.iter().filter_map(|row_id| {
        //             if let Some(mapped) = mapping.get(&row_id) {
        //                 // Mapped to either new value or None (delete)
        //                 *mapped
        //             } else {
        //                 // Not mapped to new value, keep original value
        //                 Some(row_id)
        //             }
        //         }))
        //     })
        //     .collect::<Vec<_>>();
        // builder.state.bitmaps = remapped_lists;
        // builder.write_index(dest_store).await
    }

    async fn update(
        &self,
        _new_data: SendableRecordBatchStream,
        _dest_store: &dyn IndexStore,
    ) -> Result<()> {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct NGramIndexBuilderOptions {
    tokens_per_spill: usize,
}

lazy_static::lazy_static! {
    // A higher value will use more RAM.  A lower value will have to do more spilling
    static ref DEFAULT_TOKENS_PER_SPILL: usize = std::env::var("LANCE_NGRAM_TOKENS_PER_SPILL")
        .unwrap_or_else(|_| "1000000000".to_string())
        .parse()
        .expect("failed to parse LANCE_NGRAM_TOKENS_PER_SPILL");

    static ref DEFAULT_PARALLELISM: usize = std::env::var("LANCE_NGRAM_PARALLELISM").map(|s| s.parse().expect("failed to parse LANCE_NGRAM_PARALLELISM")).unwrap_or(get_num_compute_intensive_cpus());
    // Just enough to so that tokenizing is faster than I/O
    static ref DEFAULT_TOKENIZE_PARALLELISM: usize = std::env::var("LANCE_NGRAM_TOKENIZE_PARALLELISM").map(|s| s.parse().expect("failed to parse LANCE_NGRAM_TOKENIZE_PARALLELISM")).unwrap_or(8);
}

impl Default for NGramIndexBuilderOptions {
    fn default() -> Self {
        Self {
            tokens_per_spill: *DEFAULT_TOKENS_PER_SPILL,
        }
    }
}

struct NGramIndexSpillState {
    tokens: UInt32Array,
    bitmaps: Vec<RoaringTreemap>,
}

impl NGramIndexSpillState {
    fn try_from_batch(batch: RecordBatch) -> Result<Self> {
        let tokens = batch
            .column_by_name(TOKENS_COL)
            .ok_or_lance()?
            .as_primitive::<UInt32Type>()
            .clone();
        let postings = batch
            .column_by_name(POSTING_LIST_COL)
            .unwrap()
            .as_binary::<i32>();

        let bitmaps = postings
            .into_iter()
            .map(|bytes| {
                RoaringTreemap::deserialize_from(bytes.ok_or_lance()?).map_err(|e| {
                    Error::Internal {
                        message: format!("Error deserializing ngram list: {}", e),
                        location: location!(),
                    }
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
        let tokens = UInt32Array::from_iter_values(self.tokens_map.keys().copied());
        let bitmaps = Vec::from_iter(self.tokens_map.into_values());

        NGramIndexSpillState { bitmaps, tokens }
    }
}

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
            ObjectStore::local(),
            Path::from_filesystem_path(tmpdir.path())?,
            FileMetadataCache::no_cache(),
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
        if *schema.field(0).data_type() != DataType::Utf8 {
            return Err(Error::InvalidInput {
                source: "First field in ngram index schema must be of type Utf8".into(),
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

    async fn flush(&mut self, state: NGramIndexBuildState) -> Result<()> {
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
            let right_stream = self.stream_spill(self.worker_number).await?;
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
        Ok(())
    }

    fn tokenize_and_partition(
        tokenizer: &TextAnalyzer,
        batch: RecordBatch,
        num_workers: usize,
    ) -> Vec<Vec<(u32, u64)>> {
        let text_col = batch.column(0).as_string::<i32>();
        let row_id_col = batch.column(1).as_primitive::<UInt64Type>();
        // Guessing 1000 tokens per row to at least avoid some of the earlier allocations
        let mut partitions = vec![Vec::with_capacity(batch.num_rows() * 1000); num_workers];
        let divisor = (MAX_TOKEN - MIN_TOKEN) / num_workers;
        for (text, row_id) in text_col.iter().zip(row_id_col.values()) {
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

    pub async fn train(&mut self, data: SendableRecordBatchStream) -> Result<usize> {
        let schema = data.schema();
        Self::validate_schema(schema.as_ref())?;

        let num_workers = *DEFAULT_PARALLELISM;
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
                let state = builder.state.take();
                builder.flush(state).await?;
                Result::Ok(())
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
        futures::future::try_join_all(builders).await?;

        Ok(num_workers)
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

    async fn stream_spill(
        &self,
        id: usize,
    ) -> Result<impl Stream<Item = Result<NGramIndexSpillState>>> {
        let reader = self
            .spill_store
            .open_index_file(&Self::spill_filename(id))
            .await?;

        let num_rows = reader.num_rows();

        Ok(stream::try_unfold(0, move |offset| {
            let reader = reader.clone();
            async move {
                // These are rediculously small batches but, in the worst case scenario, each row could
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
                let state = right_state.take().unwrap();
                writer.write_record_batch(state.try_into_batch()?).await?;
                while let Some(state) = right_stream.try_next().await? {
                    writer.write_record_batch(state.try_into_batch()?).await?;
                }
            } else if right_state.is_none() {
                // Right is done, full drain left
                let state = left_state.take().unwrap();
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
        &mut self,
        index_of_left: usize,
        index_of_right: usize,
        output_index: usize,
    ) -> Result<()> {
        // We fully load the small file into memory and then stream the large file
        info!(
            "Merge spill files {} and {} into {}",
            index_of_left, index_of_right, output_index
        );

        let mut writer = self
            .spill_store
            .new_index_file(&Self::spill_filename(output_index), POSTINGS_SCHEMA.clone())
            .await?;

        let left_stream = self.stream_spill(index_of_left).await?;
        let right_stream = self.stream_spill(index_of_right).await?;

        Self::merge_spill_streams(left_stream, right_stream, writer.as_mut()).await?;

        self.spill_store
            .delete_index_file(&Self::spill_filename(index_of_left))
            .await?;
        self.spill_store
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
    async fn merge_spills(&mut self, num_spills: usize) -> Result<usize> {
        info!("Merging {} index files into one combined index", num_spills);

        let mut spills_remaining = VecDeque::from_iter(1..=num_spills);
        let mut spill_counter = num_spills + 1;
        while spills_remaining.len() > 1 {
            let left = spills_remaining.pop_front().unwrap();
            let right = spills_remaining.pop_front().unwrap();
            self.merge_spill_files(left, right, spill_counter).await?;
            spills_remaining.push_back(spill_counter);
            spill_counter += 1;
        }

        return Ok(spills_remaining.pop_front().unwrap());
    }

    pub async fn write_index(
        mut self,
        store: &dyn IndexStore,
        num_spill_files: usize,
    ) -> Result<()> {
        let mut writer = store
            .new_index_file(POSTINGS_FILENAME, POSTINGS_SCHEMA.clone())
            .await?;

        let index_to_copy = self.merge_spills(num_spill_files).await?;

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

    let num_spill_files = builder.train(batches_source).await?;

    builder.write_index(index_store, num_spill_files).await
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::UInt64Type;
    use arrow_array::{Array, RecordBatch, StringArray, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        execution::SendableRecordBatchStream, physical_plan::stream::RecordBatchStreamAdapter,
    };
    use datafusion_common::DataFusionError;
    use futures::{stream, TryStreamExt};
    use lance_core::{cache::FileMetadataCache, utils::mask::RowIdTreeMap};
    use lance_datagen::{BatchCount, ByteCount, RowCount};
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;
    use tantivy::tokenizer::TextAnalyzer;
    use tempfile::tempdir;

    use crate::scalar::{
        lance_format::LanceIndexStore,
        ngram::{NGramIndex, NGramIndexBuilder, NGramIndexBuilderOptions},
        ScalarIndex, SearchResult, TextQuery,
    };

    use super::{tokenize_visitor, NGRAM_TOKENIZER};

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
    ) -> NGramIndex {
        let num_spill_files = builder.train(data).await.unwrap();

        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = LanceIndexStore::new(
            ObjectStore::local(),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            FileMetadataCache::no_cache(),
        );

        builder
            .write_index(&test_store, num_spill_files)
            .await
            .unwrap();

        NGramIndex::from_store(&test_store).await.unwrap()
    }

    #[test_log::test(tokio::test)]
    async fn test_basic_ngram_index() {
        let data = StringArray::from_iter_values(&[
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

        let index = do_train(builder, data).await;
        assert_eq!(index.tokens.len(), 21);

        // Basic search
        let res = index
            .search(&TextQuery::StringContains("cat".to_string()))
            .await
            .unwrap();

        let expected = SearchResult::AtMost(RowIdTreeMap::from_iter([0, 2, 3]));

        assert_eq!(expected, res);

        // Whitespace in query
        let res = index
            .search(&TextQuery::StringContains("nos nos".to_string()))
            .await
            .unwrap();
        let expected = SearchResult::AtMost(RowIdTreeMap::from_iter([8]));
        assert_eq!(expected, res);

        // No matches
        let res = index
            .search(&TextQuery::StringContains("tdo".to_string()))
            .await
            .unwrap();
        let expected = SearchResult::Exact(RowIdTreeMap::new());
        assert_eq!(expected, res);

        // False positive
        let res = index
            .search(&TextQuery::StringContains("inose".to_string()))
            .await
            .unwrap();
        let expected = SearchResult::AtMost(RowIdTreeMap::from_iter([8]));
        assert_eq!(expected, res);

        // Too short, don't know anything
        let res = index
            .search(&TextQuery::StringContains("ab".to_string()))
            .await
            .unwrap();
        let expected = SearchResult::AtLeast(RowIdTreeMap::new());
        assert_eq!(expected, res);

        // One short string but we still get at least one trigram, this is ok
        let res = index
            .search(&TextQuery::StringContains("no nos".to_string()))
            .await
            .unwrap();
        let expected = SearchResult::AtMost(RowIdTreeMap::from_iter([8]));
        assert_eq!(expected, res);
    }

    #[test_log::test(tokio::test)]
    async fn test_ngram_nulls() {
        let data = StringArray::from_iter(&[Some("cat"), Some("dog"), None, None, Some("cat dog")]);
        let row_ids = UInt64Array::from_iter_values((0..data.len()).map(|i| i as u64));
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

        let builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions::default()).unwrap();

        let index = do_train(builder, data).await;
        assert_eq!(index.tokens.len(), 3);

        let res = index
            .search(&TextQuery::StringContains("cat".to_string()))
            .await
            .unwrap();
        let expected = SearchResult::AtMost(RowIdTreeMap::from_iter([0, 4]));
        assert_eq!(expected, res);

        // TODO: Support IS NULL queries
    }

    #[test_log::test(tokio::test)]
    async fn test_ngram_index_with_spill() {
        let data = lance_datagen::gen()
            .col(
                "values",
                lance_datagen::array::rand_utf8(ByteCount::from(50), false),
            )
            .col("row_ids", lance_datagen::array::step::<UInt64Type>())
            .into_reader_stream(RowCount::from(128), BatchCount::from(32));

        let schema = Arc::new(Schema::new(vec![
            Field::new("values", DataType::Utf8, false),
            Field::new("row_ids", DataType::UInt64, false),
        ]));
        let data = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            data.map_err(|arrow_err| DataFusionError::ArrowError(arrow_err, None)),
        ));

        let builder = NGramIndexBuilder::try_new(NGramIndexBuilderOptions {
            tokens_per_spill: 100,
        })
        .unwrap();

        let index = do_train(builder, data).await;

        assert_eq!(index.tokens.len(), 29012);
    }
}
