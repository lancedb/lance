// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{AsArray, Int32Builder, ListBuilder};
use arrow::buffer::ScalarBuffer;
use arrow::datatypes::{self, Float32Type, Int32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, Float32Array, ListArray, OffsetSizeTrait, PrimitiveArray, RecordBatch,
    StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use futures::stream::repeat_with;
use futures::{stream, StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_core::utils::mask::RowIdTreeMap;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{Error, Result, ROW_ID};
use lazy_static::lazy_static;
use moka::future::Cache;
use roaring::RoaringBitmap;
use snafu::{location, Location};
use tantivy::tokenizer::Language;
use tracing::instrument;

use crate::prefilter::{NoFilter, PreFilter};
use crate::scalar::{
    AnyQuery, FullTextSearchQuery, IndexReader, IndexStore, SargableQuery, ScalarIndex,
};
use crate::Index;

use super::builder::InvertedList;
use super::{wand::*, InvertedIndexBuilder};

pub const TOKENS_FILE: &str = "tokens.lance";
pub const INVERT_LIST_FILE: &str = "invert.lance";
pub const DOCS_FILE: &str = "docs.lance";

pub const TOKEN_COL: &str = "_token";
pub const TOKEN_ID_COL: &str = "_token_id";
pub const FREQUENCY_COL: &str = "_frequency";
pub const POSITION_COL: &str = "_position";
pub const NUM_TOKEN_COL: &str = "_num_tokens";
pub const SCORE_COL: &str = "_score";
lazy_static! {
    pub static ref SCORE_FIELD: Field = Field::new(SCORE_COL, DataType::Float32, true);
}

// BM25 parameters
pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;

lazy_static! {
    pub static ref TOKENIZER: tantivy::tokenizer::TextAnalyzer = {
        tantivy::tokenizer::TextAnalyzer::builder(tantivy::tokenizer::SimpleTokenizer::default())
            .filter(tantivy::tokenizer::RemoveLongFilter::limit(40))
            .filter(tantivy::tokenizer::StopWordFilter::new(Language::English).unwrap())
            .filter(tantivy::tokenizer::LowerCaser)
            .filter(tantivy::tokenizer::Stemmer::new(Language::English))
            .build()
    };
    static ref CACHE_SIZE: usize = std::env::var("LANCE_INVERTED_CACHE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512 * 1024 * 1024);
}

#[derive(Debug, Clone)]
pub struct InvertedIndex {
    tokens: TokenSet,
    inverted_list: Arc<InvertedListReader>,
    docs: DocSet,
}

impl DeepSizeOf for InvertedIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.tokens.deep_size_of_children(context)
            + self.inverted_list.deep_size_of_children(context)
            + self.docs.deep_size_of_children(context)
    }
}

impl InvertedIndex {
    // map tokens to token ids
    // ignore tokens that are not in the index cause they won't contribute to the search
    #[instrument(level = "debug", skip_all)]
    fn map(&self, texts: &[String]) -> Vec<u32> {
        texts
            .iter()
            .filter_map(|text| self.tokens.get(text))
            .collect()
    }

    #[instrument(level = "debug", skip_all)]
    pub async fn full_text_search(
        &self,
        query: &FullTextSearchQuery,
        prefilter: Arc<dyn PreFilter>,
    ) -> Result<Vec<(u64, f32)>> {
        let tokens = collect_tokens(&query.query);
        let token_ids = self.map(&tokens).into_iter();
        let token_ids = if !is_phrase_query(&query.query) {
            token_ids.sorted_unstable().dedup().collect()
        } else {
            token_ids.collect()
        };
        self.bm25_search(token_ids, query, prefilter).await
    }

    // search the documents that contain the query
    // return the row ids of the documents sorted by bm25 score
    // ref: https://en.wikipedia.org/wiki/Okapi_BM25
    #[instrument(level = "debug", skip_all)]
    async fn bm25_search(
        &self,
        token_ids: Vec<u32>,
        query: &FullTextSearchQuery,
        prefilter: Arc<dyn PreFilter>,
    ) -> Result<Vec<(u64, f32)>> {
        let limit = query
            .limit
            .map(|limit| limit as usize)
            .unwrap_or(usize::MAX);
        let wand_factor = query.wand_factor.unwrap_or(1.0);

        let mask = prefilter.mask();
        let is_phrase_query = is_phrase_query(&query.query);
        let postings = stream::iter(token_ids)
            .enumerate()
            .zip(repeat_with(|| (self.inverted_list.clone(), mask.clone())))
            .map(|((position, token_id), (inverted_list, mask))| async move {
                let posting = inverted_list
                    .posting_list(token_id, is_phrase_query)
                    .await?;
                Result::Ok(PostingIterator::new(
                    token_id,
                    position as i32,
                    posting,
                    self.docs.len(),
                    mask.clone(),
                ))
            })
            // Use compute count since data hopefully cached
            .buffered(get_num_compute_intensive_cpus())
            .try_collect::<Vec<_>>()
            .await?;

        let mut wand = Wand::new(self.docs.len(), postings.into_iter());
        wand.search(is_phrase_query, limit, wand_factor, |doc, freq| {
            let doc_norm =
                K1 * (1.0 - B + B * self.docs.num_tokens(doc) as f32 / self.docs.average_length());
            freq / (freq + doc_norm)
        })
        .await
    }

    async fn to_builder(&self) -> Result<InvertedIndexBuilder> {
        let tokens = self.tokens.clone();
        let invert_list = InvertedList::load(self.inverted_list.reader.clone()).await?;
        let docs = self.docs.clone();
        Ok(InvertedIndexBuilder {
            tokens,
            invert_list,
            docs,
        })
    }
}

#[async_trait]
impl Index for InvertedIndex {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::invalid_input(
            "inverted index cannot be cast to vector index",
            location!(),
        ))
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "num_tokens": self.tokens.tokens.len(),
            "num_docs": self.docs.token_count.len(),
        }))
    }

    fn index_type(&self) -> crate::IndexType {
        crate::IndexType::Inverted
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}

#[async_trait]
impl ScalarIndex for InvertedIndex {
    // return the row ids of the documents that contain the query
    #[instrument(level = "debug", skip_all)]
    async fn search(&self, query: &dyn AnyQuery) -> Result<RowIdTreeMap> {
        let query = query.as_any().downcast_ref::<SargableQuery>().unwrap();
        let row_ids = match query {
            SargableQuery::FullTextSearch(query) => self
                .full_text_search(query, Arc::new(NoFilter))
                .await?
                .into_iter()
                .map(|(row_id, _)| row_id),
            query => {
                return Err(Error::invalid_input(
                    format!("unsupported query {:?} for inverted index", query),
                    location!(),
                ))
            }
        };

        Ok(RowIdTreeMap::from_iter(row_ids))
    }

    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        let token_reader = store.open_index_file(TOKENS_FILE).await?;
        let invert_list_reader = store.open_index_file(INVERT_LIST_FILE).await?;
        let docs_reader = store.open_index_file(DOCS_FILE).await?;

        let tokens = TokenSet::load(token_reader).await?;
        let inverted_list = InvertedListReader::new(invert_list_reader)?;
        let docs = DocSet::load(docs_reader).await?;

        Ok(Arc::new(Self {
            tokens,
            inverted_list: Arc::new(inverted_list),
            docs,
        }))
    }

    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        self.to_builder().await?.remap(mapping, dest_store).await
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        self.to_builder().await?.update(new_data, dest_store).await
    }
}

// TokenSet is a mapping from tokens to token ids
// it also records the frequency of each token
#[derive(Debug, Clone, Default, DeepSizeOf)]
pub struct TokenSet {
    // token -> (token_id, frequency)
    tokens: HashMap<String, u32>,
    next_id: u32,
}

impl TokenSet {
    pub fn to_batch(&self) -> Result<RecordBatch> {
        let mut tokens = Vec::with_capacity(self.tokens.len());
        let mut token_ids = Vec::with_capacity(self.tokens.len());
        self.tokens.iter().for_each(|(token, token_id)| {
            tokens.push(token.clone());
            token_ids.push(*token_id);
        });
        let mut indices = (0..self.tokens.len()).collect_vec();
        indices.sort_unstable_by_key(|&i| tokens[i].as_str());
        let token_col = StringArray::from_iter_values(indices.iter().map(|&i| tokens[i].as_str()));
        let token_id_col = UInt32Array::from_iter_values(indices.iter().map(|&i| token_ids[i]));

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(TOKEN_COL, DataType::Utf8, false),
            arrow_schema::Field::new(TOKEN_ID_COL, DataType::UInt32, false),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(token_col) as ArrayRef,
                Arc::new(token_id_col) as ArrayRef,
            ],
        )?;
        Ok(batch)
    }

    pub async fn load(reader: Arc<dyn IndexReader>) -> Result<Self> {
        let mut tokens = HashMap::new();
        let mut next_id = 0;

        let batch = reader.read_range(0..reader.num_rows(), None).await?;
        let token_col = batch[TOKEN_COL].as_string::<i32>();
        let token_id_col = batch[TOKEN_ID_COL].as_primitive::<datatypes::UInt32Type>();

        for (token, &token_id) in token_col.iter().zip(token_id_col.values().iter()) {
            let token = token.unwrap();
            tokens.insert(token.to_owned(), token_id);
            next_id = next_id.max(token_id + 1);
        }

        Ok(Self { tokens, next_id })
    }

    pub fn add(&mut self, token: String) -> u32 {
        let next_id = self.next_id();
        let token_id = *self.tokens.entry(token).or_insert(next_id);

        // add token if it doesn't exist
        if token_id == next_id {
            self.next_id += 1;
        }

        token_id
    }

    pub fn get(&self, token: &str) -> Option<u32> {
        self.tokens.get(token).copied()
    }

    pub fn next_id(&self) -> u32 {
        self.next_id
    }
}

struct InvertedListReader {
    reader: Arc<dyn IndexReader>,
    offsets: Vec<usize>,
    max_scores: Option<Vec<f32>>,

    // cache
    posting_cache: Cache<u32, PostingList>,
    position_cache: Cache<u32, ListArray>,
}

impl std::fmt::Debug for InvertedListReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedListReader")
            .field("offsets", &self.offsets)
            .finish()
    }
}

impl DeepSizeOf for InvertedListReader {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.offsets.deep_size_of_children(context)
            // + self.lengths.deep_size_of_children(context)
            + self.posting_cache.weighted_size() as usize
    }
}

impl InvertedListReader {
    pub(crate) fn new(reader: Arc<dyn IndexReader>) -> Result<Self> {
        let offsets = reader
            .schema()
            .metadata
            .get("offsets")
            .ok_or_else(|| Error::io("offsets not found".to_string(), location!()))?;
        let offsets: Vec<usize> = serde_json::from_str(offsets)?;

        let max_scores = match reader.schema().metadata.get("max_scores") {
            Some(max_scores) => serde_json::from_str(max_scores)?,
            None => None,
        };

        let posting_cache = Cache::builder()
            .max_capacity(*CACHE_SIZE as u64)
            .weigher(|_, posting: &PostingList| posting.deep_size_of() as u32)
            .build();
        let position_cache = Cache::builder()
            .max_capacity(*CACHE_SIZE as u64)
            .weigher(|_, positions: &ListArray| positions.get_array_memory_size() as u32)
            .build();
        Ok(Self {
            reader,
            offsets,
            max_scores,
            posting_cache,
            position_cache,
        })
    }

    pub(crate) fn posting_len(&self, token_id: u32) -> usize {
        let token_id = token_id as usize;
        let next_offset = self
            .offsets
            .get(token_id + 1)
            .copied()
            .unwrap_or(self.reader.num_rows());
        next_offset - self.offsets[token_id]
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) async fn posting_list(
        &self,
        token_id: u32,
        is_phrase_query: bool,
    ) -> Result<PostingList> {
        let mut posting = self
            .posting_cache
            .try_get_with(token_id, async move {
                let length = self.posting_len(token_id);
                let token_id = token_id as usize;
                let offset = self.offsets[token_id];
                let batch = self
                    .reader
                    .read_range(offset..offset + length, Some(&[ROW_ID, FREQUENCY_COL]))
                    .await?;
                let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().clone();
                let frequencies = batch[FREQUENCY_COL].as_primitive::<Float32Type>().clone();
                Result::Ok(PostingList::new(
                    row_ids.values().clone(),
                    frequencies.values().clone(),
                    self.max_scores
                        .as_ref()
                        .map(|max_scores| max_scores[token_id]),
                ))
            })
            .await
            .map_err(|e| Error::io(e.to_string(), location!()))?;

        if is_phrase_query {
            // hit the cache and when the cache was populated, the positions column was not loaded
            let positions = self.read_positions(token_id).await?;
            posting.positions = Some(positions);
        }

        Ok(posting)
    }

    async fn read_positions(&self, token_id: u32) -> Result<ListArray> {
        self.position_cache.try_get_with(token_id, async move {
            let length = self.posting_len(token_id);
            let token_id = token_id as usize;
            let offset = self.offsets[token_id];
            let batch = self
                .reader
                .read_range(offset..offset + length, Some(&[POSITION_COL]))
                .await?;
            Result::Ok(batch
                .column_by_name(POSITION_COL)
                .ok_or(Error::Index { message: "the index was built with old version which doesn't support phrase query, please re-create the index".to_owned(), location: location!() })?
                .as_list::<i32>()
                .clone())
        }).await.map_err(|e| Error::io(e.to_string(), location!()))
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct PostingList {
    pub row_ids: ScalarBuffer<u64>,
    pub frequencies: ScalarBuffer<f32>,
    pub max_score: Option<f32>,
    pub positions: Option<ListArray>,
}

impl DeepSizeOf for PostingList {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.row_ids.len() * std::mem::size_of::<u64>()
            + self.frequencies.len() * std::mem::size_of::<f32>()
    }
}

impl PostingList {
    pub fn new(
        row_ids: ScalarBuffer<u64>,
        frequencies: ScalarBuffer<f32>,
        max_score: Option<f32>,
    ) -> Self {
        Self {
            row_ids,
            frequencies,
            max_score,
            positions: None,
        }
    }

    pub fn len(&self) -> usize {
        self.row_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn doc(&self, i: usize) -> DocInfo {
        DocInfo::new(self.row_ids[i], self.frequencies[i])
    }

    pub fn positions(&self, row_id: u64) -> Option<PrimitiveArray<Int32Type>> {
        let pos = self.row_ids.binary_search(&row_id).ok()?;
        self.positions
            .as_ref()
            .map(|positions| positions.value(pos).as_primitive::<Int32Type>().clone())
    }

    pub fn max_score(&self) -> Option<f32> {
        self.max_score
    }

    pub fn row_id(&self, i: usize) -> u64 {
        self.row_ids[i]
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub struct PostingListBuilder {
    pub row_ids: Vec<u64>,
    pub frequencies: Vec<f32>,
    pub positions: Option<Vec<Vec<i32>>>,
}

impl Default for PostingListBuilder {
    fn default() -> Self {
        Self {
            row_ids: Vec::new(),
            frequencies: Vec::new(),
            positions: Some(Vec::new()),
        }
    }
}

impl PostingListBuilder {
    pub fn new(row_ids: Vec<u64>, frequencies: Vec<f32>, positions: Option<Vec<Vec<i32>>>) -> Self {
        Self {
            row_ids,
            frequencies,
            positions,
        }
    }

    pub fn len(&self) -> usize {
        self.row_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn calculate_max_score(&self, docs: &DocSet) -> f32 {
        // score(q, D) = IDF(q) * (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b * |D| / avgdl))
        let num_docs = docs.len();
        let avgdl = docs.average_length();
        let mut max_score = 0.0;
        for (&row_id, &freq) in self.row_ids.iter().zip(self.frequencies.iter()) {
            let doc_norm = K1 * (1.0 - B + B * docs.num_tokens(row_id) as f32 / avgdl);
            let score = idf(self.len(), num_docs) * (K1 + 1.0) * freq / (freq + doc_norm);
            if score > max_score {
                max_score = score;
            }
        }
        max_score
    }

    pub fn to_batch(&self) -> Result<RecordBatch> {
        let indices = (0..self.row_ids.len())
            .sorted_unstable_by_key(|&i| self.row_ids[i])
            .collect_vec();
        let row_id_col = UInt64Array::from_iter_values(indices.iter().map(|&i| self.row_ids[i]));
        let frequency_col =
            Float32Array::from_iter_values(indices.iter().map(|&i| self.frequencies[i]));
        let position_col = self.positions.as_ref().map(|positions| {
            let mut position_col_builder =
                ListBuilder::with_capacity(Int32Builder::new(), self.len());
            for i in indices {
                let term_positions = &positions[i];
                let position_builder = position_col_builder.values();
                position_builder.append_slice(term_positions);
                position_col_builder.append(true);
            }
            position_col_builder.finish()
        });

        let mut fields = vec![
            arrow_schema::Field::new(ROW_ID, DataType::UInt64, false),
            arrow_schema::Field::new(FREQUENCY_COL, DataType::Float32, false),
        ];
        let mut columns = vec![
            Arc::new(row_id_col) as ArrayRef,
            Arc::new(frequency_col) as ArrayRef,
        ];
        if let Some(position_col) = position_col {
            fields.push(arrow_schema::Field::new(
                POSITION_COL,
                DataType::List(Field::new("item", DataType::Int32, true).into()),
                false,
            ));
            columns.push(Arc::new(position_col));
        }
        let schema = arrow_schema::Schema::new(fields);
        let batch = RecordBatch::try_new(Arc::new(schema), columns)?;
        Ok(batch)
    }
}

#[derive(Debug, Clone, Default, DeepSizeOf)]
pub struct DocInfo {
    pub row_id: u64,
    pub frequency: f32,
}

impl DocInfo {
    pub fn new(row_id: u64, frequency: f32) -> Self {
        Self { row_id, frequency }
    }
}

impl Eq for DocInfo {}

impl PartialEq for DocInfo {
    fn eq(&self, other: &Self) -> bool {
        self.row_id == other.row_id
    }
}

impl PartialOrd for DocInfo {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DocInfo {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.row_id.cmp(&other.row_id)
    }
}

// DocSet is a mapping from row ids to the number of tokens in the document
// It's used to sort the documents by the bm25 score
#[derive(Debug, Clone, Default, DeepSizeOf)]
pub struct DocSet {
    // row id -> (num tokens, norm_len)
    token_count: HashMap<u64, u32>,
    total_tokens: u64,
}

impl DocSet {
    #[inline]
    pub fn len(&self) -> usize {
        self.token_count.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn average_length(&self) -> f32 {
        self.total_tokens as f32 / self.token_count.len() as f32
    }

    pub fn to_batch(&self) -> Result<RecordBatch> {
        let row_id_col = UInt64Array::from_iter_values(self.token_count.keys().cloned());
        let num_tokens_col = UInt32Array::from_iter_values(self.token_count.values().cloned());

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(ROW_ID, DataType::UInt64, false),
            arrow_schema::Field::new(NUM_TOKEN_COL, DataType::UInt32, false),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(row_id_col) as ArrayRef,
                Arc::new(num_tokens_col) as ArrayRef,
            ],
        )?;
        Ok(batch)
    }

    pub async fn load(reader: Arc<dyn IndexReader>) -> Result<Self> {
        let mut token_count = HashMap::new();
        let mut total_tokens = 0;

        let batch = reader.read_range(0..reader.num_rows(), None).await?;
        let row_id_col = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();
        let num_tokens_col = batch[NUM_TOKEN_COL].as_primitive::<datatypes::UInt32Type>();
        for (&row_id, &num_tokens) in row_id_col
            .values()
            .iter()
            .zip(num_tokens_col.values().iter())
        {
            token_count.insert(row_id, num_tokens);
            total_tokens += num_tokens as u64;
        }

        Ok(Self {
            token_count,
            total_tokens,
        })
    }

    pub fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) {
        for (old_row_id, new_row_id) in mapping {
            match new_row_id {
                Some(new_row_id) => {
                    if let Some(num_tokens) = self.token_count.remove(old_row_id) {
                        self.token_count.insert(*new_row_id, num_tokens);
                    }
                }
                None => {
                    self.token_count.remove(old_row_id);
                }
            }
        }
    }

    #[inline]
    pub fn num_tokens(&self, row_id: u64) -> u32 {
        self.token_count.get(&row_id).cloned().unwrap_or_default()
    }

    pub fn add(&mut self, row_id: u64, num_tokens: u32) {
        self.token_count.insert(row_id, num_tokens);
        self.total_tokens += num_tokens as u64;
    }
}

#[inline]
pub fn idf(nq: usize, num_docs: usize) -> f32 {
    let num_docs = num_docs as f32;
    ((num_docs - nq as f32 + 0.5) / (nq as f32 + 0.5) + 1.0).ln()
}

#[instrument(level = "debug", skip(batches))]
pub fn flat_full_text_search(
    batches: &[&RecordBatch],
    doc_col: &str,
    query: &str,
) -> Result<Vec<u64>> {
    if batches.is_empty() {
        return Ok(Vec::new());
    }

    match batches[0][doc_col].data_type() {
        DataType::Utf8 => do_flat_full_text_search::<i32>(batches, doc_col, query),
        DataType::LargeUtf8 => do_flat_full_text_search::<i64>(batches, doc_col, query),
        data_type => Err(Error::invalid_input(
            format!("unsupported data type {} for inverted index", data_type),
            location!(),
        )),
    }
}

fn do_flat_full_text_search<Offset: OffsetSizeTrait>(
    batches: &[&RecordBatch],
    doc_col: &str,
    query: &str,
) -> Result<Vec<u64>> {
    let mut results = Vec::new();
    let query_tokens = collect_tokens(query).into_iter().collect::<HashSet<_>>();
    for batch in batches {
        let row_id_array = batch[ROW_ID].as_primitive::<UInt64Type>();
        let doc_array = batch[doc_col].as_string::<Offset>();
        for i in 0..row_id_array.len() {
            let doc = doc_array.value(i);
            let doc_tokens = collect_tokens(doc);
            if doc_tokens.iter().any(|token| query_tokens.contains(token)) {
                results.push(row_id_array.value(i));
                assert!(doc.contains(query));
            }
        }
    }

    Ok(results)
}

pub fn collect_tokens(text: &str) -> Vec<String> {
    let mut tokenizer = TOKENIZER.clone();
    let mut stream = tokenizer.token_stream(text);
    let mut tokens = Vec::new();
    while let Some(token) = stream.next() {
        tokens.push(token.text.to_owned());
    }
    tokens
}

pub fn is_phrase_query(query: &str) -> bool {
    query.starts_with("\"") && query.ends_with("\"")
}
