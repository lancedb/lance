use std::any::Any;
use std::{collections::HashMap, sync::Arc};

use arrow::array::AsArray;
use arrow::datatypes::UInt64Type;
use arrow_array::{BinaryArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use futures::{StreamExt, TryStreamExt};
use lance_core::utils::address::RowAddress;
use lance_core::Result;
use lance_core::{utils::mask::RowIdTreeMap, Error};
use moka::future::Cache;
use roaring::{RoaringBitmap, RoaringTreemap};
use serde::Serialize;
use snafu::location;
use tantivy::tokenizer::TextAnalyzer;
use tracing::instrument;

use crate::scalar::inverted::CACHE_SIZE;
use crate::vector::VectorIndex;
use crate::{Index, IndexType};

use super::btree::TrainingSource;
use super::inverted::TokenSet;
use super::{AnyQuery, IndexReader, IndexStore, ScalarIndex, SearchResult, TextQuery};

const TOKENS_COL: &str = "tokens";
const POSTING_LIST_COL: &str = "posting_list";
const POSTINGS_FILENAME: &str = "ngram_postings.lance";

lazy_static::lazy_static! {
    pub static ref TOKENS_FIELD: Field = Field::new(TOKENS_COL, DataType::Utf8, false);
    pub static ref POSTINGS_FIELD: Field = Field::new(POSTING_LIST_COL, DataType::Binary, false);
    pub static ref POSTINGS_SCHEMA: SchemaRef = Arc::new(Schema::new(vec![TOKENS_FIELD.clone(), POSTINGS_FIELD.clone()]));
    /// Currently we ALWAYS use trigrams with ascii folding and lower casing.  We may want to make this configurable in the future.
    pub static ref NGRAM_TOKENIZER: TextAnalyzer = TextAnalyzer::builder(tantivy::tokenizer::NgramTokenizer::all_ngrams(1, 3).unwrap())
    .filter(tantivy::tokenizer::LowerCaser)
    .filter(tantivy::tokenizer::AlphaNumOnlyFilter)
    .filter(tantivy::tokenizer::AsciiFoldingFilter)
    .build();
}
trait TextAnalyzerExt {
    fn tokenize(self, text: &str) -> Vec<String>;
}

impl TextAnalyzerExt for TextAnalyzer {
    fn tokenize(mut self, text: &str) -> Vec<String> {
        let mut stream = self.token_stream(text);
        let mut tokens = Vec::new();
        while stream.advance() {
            let token = stream.token();
            tokens.push(token.text.clone());
        }
        tokens
    }
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
        self.bitmap.serialized_size() as usize
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
    pub async fn ngram_list(&self, token_id: u32) -> Result<Arc<NGramPostingList>> {
        Ok(self
            .cache
            .try_get_with(token_id, async move {
                let batch = self
                    .reader
                    .read_range(
                        token_id as usize..token_id as usize + 1,
                        Some(&[POSTING_LIST_COL]),
                    )
                    .await?;
                Result::Ok(Arc::new(NGramPostingList::try_from_batch(batch)?))
            })
            .await
            .map_err(|e| Error::io(e.to_string(), location!()))?)
    }
}

/// An ngram index
///
/// This index stores a mapping from ngrams to the row ids that contain them.
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
    /// The mapping from ngrams to token ids
    tokens: TokenSet,
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

        let mut tokens_map = HashMap::with_capacity(tokens.num_rows());
        tokens_map.extend(
            tokens
                .column(0)
                .as_string::<i32>()
                .iter()
                .enumerate()
                .map(|(i, token)| (token.unwrap().to_owned(), i as u32)),
        );
        let tokens = TokenSet::new(tokens_map);

        let posting_reader = Arc::new(NGramPostingListReader {
            reader: store.open_index_file(POSTINGS_FILENAME).await?,
            cache: Cache::builder()
                .max_capacity(*CACHE_SIZE as u64)
                .weigher(|_, posting: &Arc<NGramPostingList>| posting.deep_size_of() as u32)
                .build(),
        });

        Ok(Self {
            io_parallelism: store.io_parallelism(),
            tokens,
            list_reader: posting_reader,
            tokenizer: NGRAM_TOKENIZER.clone(),
        })
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokenizer = self.tokenizer.clone();
        let mut stream = tokenizer.token_stream(text);
        let mut tokens = Vec::new();
        while stream.advance() {
            let token = stream.token();
            tokens.push(token.text.clone());
        }
        tokens
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
            num_ngrams: self.tokens.num_tokens(),
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
        for token in self.tokens.all_tokens() {
            let list = self.list_reader.ngram_list(token).await?;
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
                let tokens = self.tokenize(substr);
                let token_ids = tokens
                    .into_iter()
                    .map(|t| self.tokens.get(&t))
                    .collect::<Option<Vec<_>>>();
                let Some(token_ids) = token_ids else {
                    return Ok(SearchResult::Exact(RowIdTreeMap::new()));
                };
                let posting_lists = futures::stream::iter(
                    token_ids
                        .into_iter()
                        .map(|token_id| self.list_reader.ngram_list(token_id)),
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

    /// Load the scalar index from storage
    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        Ok(Arc::new(Self::from_store(store.as_ref()).await?))
    }

    /// Remap the row ids, creating a new remapped version of this index in `dest_store`
    async fn remap(
        &self,
        _mapping: &HashMap<u64, Option<u64>>,
        _dest_store: &dyn IndexStore,
    ) -> Result<()> {
        todo!()
    }

    /// Add the new data into the index, creating an updated version of the index in `dest_store`
    async fn update(
        &self,
        _new_data: SendableRecordBatchStream,
        _dest_store: &dyn IndexStore,
    ) -> Result<()> {
        todo!()
    }
}

pub struct NGramIndexBuilder {
    tokenizer: TextAnalyzer,
    tokens_map: HashMap<String, u32>,
    tokens_counter: u32,
    bitmaps: Vec<RoaringTreemap>,
}

impl Default for NGramIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl NGramIndexBuilder {
    pub fn new() -> Self {
        let tokenizer = NGRAM_TOKENIZER.clone();
        Self {
            tokenizer,
            // Default capacity loosely based on case insensitive ascii ngrams with punctuation stripped
            tokens_map: HashMap::with_capacity(4 * 26 * 26 * 26),
            bitmaps: Vec::with_capacity(4 * 26 * 26 * 26),
            tokens_counter: 0,
        }
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

    fn process_batch(&mut self, batch: &RecordBatch) {
        let text_col = batch.column(0).as_string::<i32>();
        let row_id_col = batch.column(1).as_primitive::<UInt64Type>();
        for (text, row_id) in text_col.iter().zip(row_id_col.values()) {
            let text = text.unwrap();
            let tokens = self.tokenizer.clone().tokenize(text);
            for token in tokens {
                let token_id = *self.tokens_map.entry(token).or_insert_with(|| {
                    let token_id = self.tokens_counter;
                    self.tokens_counter += 1;
                    self.bitmaps.push(RoaringTreemap::new());
                    token_id
                });
                self.bitmaps[token_id as usize].insert(*row_id);
            }
        }
    }

    pub async fn train(&mut self, mut data: SendableRecordBatchStream) -> Result<()> {
        let schema = data.schema();
        Self::validate_schema(schema.as_ref())?;

        while let Some(batch) = data.try_next().await? {
            self.process_batch(&batch);
        }
        Ok(())
    }

    pub async fn write(self, store: &dyn IndexStore) -> Result<()> {
        let mut ordered_tokens = self.tokens_map.into_iter().collect::<Vec<_>>();
        ordered_tokens.sort_by_key(|(_, id)| *id);
        let tokens_array =
            StringArray::from_iter_values(ordered_tokens.into_iter().map(|(t, _)| t));

        let bitmap_array = BinaryArray::from_iter_values(self.bitmaps.into_iter().map(|bitmap| {
            let mut buf = Vec::with_capacity(bitmap.serialized_size());
            bitmap.serialize_into(&mut buf).unwrap();
            buf
        }));
        let postings_batch = RecordBatch::try_new(
            POSTINGS_SCHEMA.clone(),
            vec![Arc::new(tokens_array), Arc::new(bitmap_array)],
        )?;

        let mut postings_writer = store
            .new_index_file(POSTINGS_FILENAME, POSTINGS_SCHEMA.clone())
            .await?;
        postings_writer.write_record_batch(postings_batch).await?;
        postings_writer.finish().await?;

        Ok(())
    }
}

pub async fn train_ngram_index(
    data_source: Box<dyn TrainingSource + Send>,
    index_store: &dyn IndexStore,
) -> Result<()> {
    let batches_source = data_source.scan_unordered_chunks(4096).await?;
    let mut builder = NGramIndexBuilder::new();

    builder.train(batches_source).await?;

    builder.write(index_store).await
}
