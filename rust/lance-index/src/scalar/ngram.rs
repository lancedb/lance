// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

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
    pub static ref TOKENS_FIELD: Field = Field::new(TOKENS_COL, DataType::Utf8, true);
    pub static ref POSTINGS_FIELD: Field = Field::new(POSTING_LIST_COL, DataType::Binary, false);
    pub static ref POSTINGS_SCHEMA: SchemaRef = Arc::new(Schema::new(vec![TOKENS_FIELD.clone(), POSTINGS_FIELD.clone()]));
    /// Currently we ALWAYS use trigrams with ascii folding and lower casing.  We may want to make this configurable in the future.
    pub static ref NGRAM_TOKENIZER: TextAnalyzer = TextAnalyzer::builder(tantivy::tokenizer::NgramTokenizer::all_ngrams(1, 3).unwrap())
    .filter(tantivy::tokenizer::LowerCaser)
    .filter(tantivy::tokenizer::AsciiFoldingFilter)
    .filter(tantivy::tokenizer::AlphaNumOnlyFilter)
    .build();
}

// Helper function to apply a function to each token in a text
fn tokenize_visitor(analyzer: &TextAnalyzer, text: &str, mut visitor: impl FnMut(&String)) {
    // The token_stream method is mutable.  As far as I can tell this is to enforce exclusivity and not
    // true mutability.  For example, the object returned by `token_stream` has thread-local state but
    // it is reset each time `token_stream` is called.
    //
    // However, I don't see this documented anywhere and I'm not sure about relying on it.  For now, we
    // make a clone as that seems to be the safer option.  All the tokenizers we use here should be trivially
    // cloneable (although it requires a heap allocation so may be worth investigating in the future)
    let mut this = analyzer.clone();
    let mut stream = this.token_stream(text);
    while stream.advance() {
        visitor(&stream.token().text);
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
        self.cache
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
            .map_err(|e| Error::io(e.to_string(), location!()))
    }

    pub async fn load_all_lists(&self) -> Result<Vec<RoaringTreemap>> {
        let num_rows = self.reader.num_rows();
        let batch = self
            .reader
            .read_range(0..num_rows, Some(&[POSTING_LIST_COL]))
            .await?;
        let arr = batch.column(0).as_binary::<i32>();
        arr.iter()
            .map(|bytes| {
                RoaringTreemap::deserialize_from(
                    bytes.expect("should not be any null values in ngram posting lists"),
                )
                .map_err(|e| Error::Internal {
                    message: format!("Error deserializing ngram list: {}", e),
                    location: location!(),
                })
            })
            .collect()
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
                // Filter out the null token
                .filter_map(|(i, token)| token.map(|token| (token.to_owned(), i as u32))),
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

    async fn to_builder(&self) -> Result<NGramIndexBuilder> {
        let tokens_map = self.tokens.tokens.clone();
        let tokenizer = self.tokenizer.clone();
        let bitmaps = self.list_reader.load_all_lists().await?;
        Ok(NGramIndexBuilder {
            tokens_map,
            tokenizer,
            bitmaps,
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
                let mut token_ids = Vec::with_capacity(substr.len() * 3);
                let mut missing = false;
                tokenize_visitor(&self.tokenizer, substr, |token| {
                    if let Some(token_id) = self.tokens.get(token) {
                        token_ids.push(token_id);
                    } else {
                        missing = true;
                    }
                });
                // At least one token was missing, so we know there are zero results
                if missing {
                    return Ok(SearchResult::Exact(RowIdTreeMap::new()));
                }
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

    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        Ok(Arc::new(Self::from_store(store.as_ref()).await?))
    }

    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        let mut builder = self.to_builder().await?;
        let lists = std::mem::take(&mut builder.bitmaps);
        let remapped_lists = lists
            .into_iter()
            .map(|list| {
                RoaringTreemap::from_iter(list.iter().filter_map(|row_id| {
                    if let Some(mapped) = mapping.get(&row_id) {
                        // Mapped to either new value or None (delete)
                        *mapped
                    } else {
                        // Not mapped to new value, keep original value
                        Some(row_id)
                    }
                }))
            })
            .collect::<Vec<_>>();
        builder.bitmaps = remapped_lists;
        builder.write(dest_store).await
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        let mut builder = self.to_builder().await?;
        builder.train(new_data).await?;
        builder.write(dest_store).await
    }
}

pub struct NGramIndexBuilder {
    tokenizer: TextAnalyzer,
    tokens_map: HashMap<String, u32>,
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
        let mut bitmaps = Vec::with_capacity(36 * 36 * 36 + 1);
        // Token 0 is always the NULL bitmap
        bitmaps.push(RoaringTreemap::new());
        Self {
            tokenizer,
            // Default capacity loosely based on case insensitive ascii trigrams with punctuation stripped
            tokens_map: HashMap::with_capacity(36 * 36 * 36),
            bitmaps,
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
            if let Some(text) = text {
                tokenize_visitor(&self.tokenizer, text, |token| {
                    // This would be a bit simpler with entry API but, at scale, the vast majority
                    // of cases will be a hit and we want to avoid cloning the string if we can.  So
                    // for now we do the double-hash.  We can simplify in the future with raw_entry
                    // when it stabilizes.
                    let tokens_list = self.tokens_map.get(token);
                    if let Some(token_id) = tokens_list {
                        self.bitmaps[*token_id as usize].insert(*row_id);
                        return;
                    }

                    let mut new_map = RoaringTreemap::new();
                    let token_id = self.bitmaps.len() as u32;
                    self.tokens_map.insert(token.to_owned(), token_id);
                    new_map.insert(*row_id);
                    self.bitmaps.push(new_map);
                });
            } else {
                self.bitmaps[0].insert(*row_id);
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
        // Prepend NULL token
        let tokens_array = StringArray::from_iter(
            std::iter::once(None).chain(ordered_tokens.into_iter().map(|(t, _)| Some(t))),
        );

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

#[cfg(test)]
mod tests {
    use tantivy::tokenizer::TextAnalyzer;

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
            vec!["c", "ca", "caf", "a", "af", "afe", "f", "fe", "e"] // spellchecker:disable-line
        );

        // Allow numbers
        let tokens = collect_tokens(&tokenizer, "a1b2");
        assert_eq!(
            tokens,
            vec!["a", "a1", "a1b", "1", "1b", "1b2", "b", "b2", "2"]
        );

        // Remove symbols and UTF-8 that doesn't map to characters
        let tokens = collect_tokens(&tokenizer, "a👍b!c");

        assert_eq!(tokens, vec!["a", "b", "c"]);

        // Lower casing
        let tokens = collect_tokens(&tokenizer, "ABC");
        assert_eq!(tokens, vec!["a", "ab", "abc", "b", "bc", "c"]);

        // Duplicate tokens
        let tokens = collect_tokens(&tokenizer, "abab");
        // Confirming that the tokenizer doesn't deduplicate tokens (this can be taken into consideration
        // when training the index)
        assert_eq!(
            tokens,
            vec!["a", "ab", "aba", "b", "ba", "bab", "a", "ab", "b"] // spellchecker:disable-line
        );
    }
}
