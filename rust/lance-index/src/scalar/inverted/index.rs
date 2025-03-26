// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::Arc;
use std::usize;

use arrow::array::{
    AsArray, Float32Builder, Int32Builder, ListBuilder, StringBuilder, UInt32Builder, UInt64Builder,
};
use arrow::buffer::ScalarBuffer;
use arrow::datatypes::{self, Float32Type, Int32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, ListArray, OffsetSizeTrait, PrimitiveArray,
    RecordBatch, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_common::DataFusionError;
use deepsize::DeepSizeOf;
use fst::{IntoStreamer, Streamer};
use futures::stream::repeat_with;
use futures::{stream, StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_arrow::{iter_str_array, RecordBatchExt};
use lance_core::utils::mask::RowIdTreeMap;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::utils::tracing::{IO_TYPE_LOAD_SCALAR_PART, TRACE_IO_EVENTS};
use lance_core::{Error, Result, ROW_ID, ROW_ID_FIELD};
use lazy_static::lazy_static;
use moka::future::Cache;
use roaring::RoaringBitmap;
use snafu::location;
use tracing::{info, instrument};

use super::builder::inverted_list_schema;
use super::query::*;
use super::{wand::*, InvertedIndexBuilder, TokenizerConfig};
use crate::prefilter::{NoFilter, PreFilter};
use crate::scalar::{
    AnyQuery, FullTextSearchQuery, IndexReader, IndexStore, InvertedIndexParams, MetricsCollector,
    SargableQuery, ScalarIndex, SearchResult,
};
use crate::Index;

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
    pub static ref FTS_SCHEMA: SchemaRef =
        Arc::new(Schema::new(vec![ROW_ID_FIELD.clone(), SCORE_FIELD.clone()]));
}

// BM25 parameters
pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;

lazy_static! {
    pub static ref CACHE_SIZE: usize = std::env::var("LANCE_INVERTED_CACHE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512 * 1024 * 1024);
}

#[derive(Clone)]
pub struct InvertedIndex {
    params: InvertedIndexParams,
    tokenizer: tantivy::tokenizer::TextAnalyzer,
    tokens: TokenSet,
    inverted_list: Arc<InvertedListReader>,
    docs: DocSet,
}

impl Debug for InvertedIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedIndex")
            .field("tokens", &self.tokens)
            .field("inverted_list", &self.inverted_list)
            .field("docs", &self.docs)
            .finish()
    }
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
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<(u64, f32)>> {
        let params = query.into();
        process_compound_query(&query.query, self, &params, prefilter, metrics).await
    }

    fn to_builder(&self) -> InvertedIndexBuilder {
        let tokens = self.tokens.clone().into_mut();
        let inverted_list = self.inverted_list.clone();
        let docs = self.docs.clone();
        InvertedIndexBuilder::from_existing_index(self.params.clone(), tokens, inverted_list, docs)
    }
}

impl Searcher for InvertedIndex {
    fn tokenizer(&self) -> tantivy::tokenizer::TextAnalyzer {
        self.tokenizer.clone()
    }

    fn expand_fuzzy(&self, tokens: Vec<String>, max_distance: Option<u32>) -> Result<Vec<String>> {
        let mut new_tokens = Vec::with_capacity(tokens.len());
        for token in tokens {
            let max_dist = match max_distance {
                Some(max_dist) => max_dist,
                None => MatchQuery::auto_dist(&token),
            };
            let lev =
                fst::automaton::Levenshtein::new(&token, max_dist).map_err(|e| Error::Index {
                    message: format!("failed to construct the fuzzy query: {}", e),
                    location: location!(),
                })?;
            if let TokenMap::Fst(ref map) = self.tokens.tokens {
                let mut stream = map.search(lev).into_stream();
                while let Some((token, _)) = stream.next() {
                    new_tokens.push(String::from_utf8_lossy(token).into_owned());
                }
            } else {
                return Err(Error::Index {
                    message: "tokens is not fst, which is not expected".to_owned(),
                    location: location!(),
                });
            }
        }
        Ok(new_tokens)
    }

    // search the documents that contain the query
    // return the row ids of the documents sorted by bm25 score
    // ref: https://en.wikipedia.org/wiki/Okapi_BM25
    #[instrument(level = "debug", skip_all)]
    async fn bm25_search(
        &self,
        tokens: Vec<String>,
        params: &FtsSearchParams,
        is_phrase_query: bool,
        prefilter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<(u64, f32)>> {
        metrics.record_comparisons(tokens.len());

        let mask = prefilter.mask();
        let token_ids = self.map(&tokens);
        let postings = stream::iter(token_ids)
            .enumerate()
            .zip(repeat_with(|| (self.inverted_list.clone(), mask.clone())))
            .map(|((position, token_id), (inverted_list, mask))| async move {
                let posting = inverted_list
                    .posting_list(token_id, is_phrase_query, metrics)
                    .await?;
                Result::Ok(PostingIterator::new(
                    token_id,
                    position as i32,
                    posting,
                    self.docs.len(),
                    mask,
                ))
            })
            // Use compute count since data hopefully cached
            .buffered(get_num_compute_intensive_cpus())
            .try_collect::<Vec<_>>()
            .await?;

        let mut wand = Wand::new(self.docs.len(), postings.into_iter());
        wand.search(
            is_phrase_query,
            params.limit,
            params.wand_factor,
            |doc, freq| {
                let doc_norm = K1
                    * (1.0 - B + B * self.docs.num_tokens(doc) as f32 / self.docs.average_length());
                freq / (freq + doc_norm)
            },
        )
        .await
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
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query.as_any().downcast_ref::<SargableQuery>().unwrap();
        let row_ids = match query {
            SargableQuery::FullTextSearch(query) => self
                .full_text_search(query, Arc::new(NoFilter), metrics)
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

        Ok(SearchResult::Exact(RowIdTreeMap::from_iter(row_ids)))
    }

    fn can_answer_exact(&self, _: &dyn AnyQuery) -> bool {
        true
    }

    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        let tokens_fut = tokio::spawn({
            let store = store.clone();
            async move {
                let token_reader = store.open_index_file(TOKENS_FILE).await?;
                let tokenizer = token_reader
                    .schema()
                    .metadata
                    .get("tokenizer")
                    .map(|s| serde_json::from_str::<TokenizerConfig>(s))
                    .transpose()?
                    .unwrap_or_default();
                let tokens = TokenSet::load(token_reader).await?;
                Result::Ok((tokenizer, tokens))
            }
        });
        let invert_list_fut = tokio::spawn({
            let store = store.clone();
            async move {
                let invert_list_reader = store.open_index_file(INVERT_LIST_FILE).await?;
                let invert_list = InvertedListReader::new(invert_list_reader)?;
                Result::Ok(Arc::new(invert_list))
            }
        });
        let docs_fut = tokio::spawn({
            let store = store.clone();
            async move {
                let docs_reader = store.open_index_file(DOCS_FILE).await?;
                let docs = DocSet::load(docs_reader).await?;
                Result::Ok(docs)
            }
        });

        let (tokenizer_config, tokens) = tokens_fut.await??;
        let inverted_list = invert_list_fut.await??;
        let docs = docs_fut.await??;

        let tokenizer = tokenizer_config.build()?;
        let params = InvertedIndexParams {
            with_position: inverted_list.has_positions(),
            tokenizer_config,
        };
        Ok(Arc::new(Self {
            params,
            tokenizer,
            tokens,
            inverted_list,
            docs,
        }))
    }

    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        self.to_builder().remap(mapping, dest_store).await
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        self.to_builder().update(new_data, dest_store).await
    }
}

// at indexing, we use HashMap because we need it to be mutable,
// at searching, we use fst::Map because it's more efficient
#[derive(Debug, Clone)]
pub enum TokenMap {
    HashMap(HashMap<String, u32>),
    Fst(fst::Map<Vec<u8>>),
}

impl Default for TokenMap {
    fn default() -> Self {
        Self::HashMap(HashMap::new())
    }
}

impl DeepSizeOf for TokenMap {
    fn deep_size_of_children(&self, ctx: &mut deepsize::Context) -> usize {
        match self {
            Self::HashMap(map) => map.deep_size_of_children(ctx),
            Self::Fst(map) => map.as_fst().size(),
        }
    }
}

impl TokenMap {
    pub fn len(&self) -> usize {
        match self {
            Self::HashMap(map) => map.len(),
            Self::Fst(map) => map.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// TokenSet is a mapping from tokens to token ids
#[derive(Debug, Clone, Default, DeepSizeOf)]
pub struct TokenSet {
    // token -> token_id
    pub(crate) tokens: TokenMap,
    pub(crate) next_id: u32,
    total_length: usize,
}

impl TokenSet {
    pub fn into_mut(self) -> Self {
        let tokens = match self.tokens {
            TokenMap::HashMap(map) => map,
            TokenMap::Fst(map) => {
                let mut new_map = HashMap::with_capacity(map.len());
                let mut stream = map.into_stream();
                while let Some((token, token_id)) = stream.next() {
                    new_map.insert(String::from_utf8_lossy(token).into_owned(), token_id as u32);
                }

                new_map
            }
        };

        Self {
            tokens: TokenMap::HashMap(tokens),
            next_id: self.next_id,
            total_length: self.total_length,
        }
    }

    pub fn num_tokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn to_batch(self) -> Result<RecordBatch> {
        let mut token_builder = StringBuilder::with_capacity(self.tokens.len(), self.total_length);
        let mut token_id_builder = UInt32Builder::with_capacity(self.tokens.len());

        if let TokenMap::HashMap(map) = self.tokens {
            for (token, token_id) in map.into_iter().sorted_unstable() {
                token_builder.append_value(&token);
                token_id_builder.append_value(token_id);
            }
        } else {
            return Err(Error::Index {
                message: "tokens is not a HashMap".to_owned(),
                location: location!(),
            });
        }

        let token_col = token_builder.finish();
        let token_id_col = token_id_builder.finish();

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
        let mut next_id = 0;
        let mut total_length = 0;
        let mut tokens = fst::MapBuilder::memory();

        let batch = reader.read_range(0..reader.num_rows(), None).await?;
        let token_col = batch[TOKEN_COL].as_string::<i32>();
        let token_id_col = batch[TOKEN_ID_COL].as_primitive::<datatypes::UInt32Type>();

        for (token, &token_id) in token_col.iter().zip(token_id_col.values().iter()) {
            let token = token.ok_or(Error::Index {
                message: "found null token in token set".to_owned(),
                location: location!(),
            })?;
            next_id = next_id.max(token_id + 1);
            total_length += token.len();
            tokens
                .insert(token, token_id as u64)
                .map_err(|e| Error::Index {
                    message: format!("failed to insert token {}: {}", token, e),
                    location: location!(),
                })?;
        }

        Ok(Self {
            tokens: TokenMap::Fst(tokens.into_map()),
            next_id,
            total_length,
        })
    }

    pub fn add(&mut self, token: String) -> u32 {
        let next_id = self.next_id();
        let len = token.len();
        let token_id = match self.tokens {
            TokenMap::HashMap(ref mut map) => *map.entry(token).or_insert(next_id),
            _ => unreachable!("tokens must be HashMap while indexing"),
        };

        // add token if it doesn't exist
        if token_id == next_id {
            self.next_id += 1;
            self.total_length += len;
        }

        token_id
    }

    pub fn get(&self, token: &str) -> Option<u32> {
        match self.tokens {
            TokenMap::HashMap(ref map) => map.get(token).copied(),
            TokenMap::Fst(ref map) => map.get(token).map(|id| id as u32),
        }
    }

    pub fn next_id(&self) -> u32 {
        self.next_id
    }
}

pub struct InvertedListReader {
    reader: Arc<dyn IndexReader>,
    offsets: Vec<usize>,
    max_scores: Option<Vec<f32>>,

    has_position: bool,

    // cache
    posting_cache: Cache<u32, PostingList>,
    position_cache: Cache<u32, ListArray>,
}

impl std::fmt::Debug for InvertedListReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedListReader")
            .field("offsets", &self.offsets)
            .field("max_scores", &self.max_scores)
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

        let has_position = reader.schema().field(POSITION_COL).is_some();

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
            has_position,
            posting_cache,
            position_cache,
        })
    }

    pub(crate) fn has_positions(&self) -> bool {
        self.has_position
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

    pub(crate) async fn posting_batch(
        &self,
        token_id: u32,
        with_position: bool,
    ) -> Result<RecordBatch> {
        let mut columns = vec![ROW_ID, FREQUENCY_COL];
        if with_position {
            columns.push(POSITION_COL);
        }

        let length = self.posting_len(token_id);
        let token_id = token_id as usize;
        let offset = self.offsets[token_id];
        let batch = self
            .reader
            .read_range(offset..offset + length, Some(&columns))
            .await?;
        Ok(batch)
    }

    #[instrument(level = "debug", skip(self, metrics))]
    pub(crate) async fn posting_list(
        &self,
        token_id: u32,
        is_phrase_query: bool,
        metrics: &dyn MetricsCollector,
    ) -> Result<PostingList> {
        let mut posting = self
            .posting_cache
            .try_get_with(token_id, async move {
                metrics.record_part_load();
                info!(target: TRACE_IO_EVENTS, type=IO_TYPE_LOAD_SCALAR_PART, index_type="inverted", part_id=token_id);
                let batch = self.posting_batch(token_id, false).await?;
                let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().clone();
                let frequencies = batch[FREQUENCY_COL].as_primitive::<Float32Type>().clone();
                Result::Ok(PostingList::new(
                    row_ids.values().clone(),
                    frequencies.values().clone(),
                    self.max_scores
                        .as_ref()
                        .map(|max_scores| max_scores[token_id as usize]),
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
                .ok_or(Error::Index { message: "position is not found but required for phrase queries, try recreating the index with position".to_owned(), location: location!() })?
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
    pub positions: Option<PositionBuilder>,
}

impl PostingListBuilder {
    pub fn new(row_ids: Vec<u64>, frequencies: Vec<f32>, positions: Option<Vec<Vec<i32>>>) -> Self {
        Self {
            row_ids,
            frequencies,
            positions: positions.map(PositionBuilder::from),
        }
    }

    pub fn size(&self) -> usize {
        std::mem::size_of::<u64>() * self.row_ids.len()
            + std::mem::size_of::<f32>() * self.frequencies.len()
            + self
                .positions
                .as_ref()
                .map(|positions| positions.size())
                .unwrap_or(0)
    }

    pub fn from_batches(batches: &[RecordBatch]) -> Self {
        let row_ids = batches
            .iter()
            .flat_map(|batch| batch[ROW_ID].as_primitive::<UInt64Type>().values().iter())
            .cloned()
            .collect();
        let frequencies = batches
            .iter()
            .flat_map(|batch| {
                batch[FREQUENCY_COL]
                    .as_primitive::<Float32Type>()
                    .values()
                    .iter()
            })
            .cloned()
            .collect();
        let mut positions = None;
        if batches[0].column_by_name(POSITION_COL).is_some() {
            let mut position_builder = PositionBuilder::new();
            batches.iter().for_each(|batch| {
                let positions = batch[POSITION_COL].as_list::<i32>();
                for i in 0..positions.len() {
                    let pos = positions.value(i);
                    position_builder.push(pos.as_primitive::<Int32Type>().values().to_vec());
                }
            });
            positions = Some(position_builder);
        }
        Self {
            row_ids,
            frequencies,
            positions,
        }
    }

    pub fn empty(with_position: bool) -> Self {
        Self {
            row_ids: Vec::new(),
            frequencies: Vec::new(),
            positions: with_position.then(PositionBuilder::new),
        }
    }

    pub fn len(&self) -> usize {
        self.row_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn add(&mut self, row_id: u64, term_positions: Vec<i32>) {
        self.row_ids.push(row_id);
        self.frequencies.push(term_positions.len() as f32);
        if let Some(positions) = self.positions.as_mut() {
            positions.push(term_positions);
        }
    }

    pub fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) {
        let mut new_row_ids = Vec::with_capacity(self.len());
        let mut new_freqs = Vec::with_capacity(self.len());
        let mut new_positions = self.positions.as_mut().map(|_| PositionBuilder::new());

        for i in 0..self.len() {
            let row_id = self.row_ids[i];
            let freq = self.frequencies[i];
            let positions = self
                .positions
                .as_ref()
                .map(|positions| positions.get(i).to_vec());

            match mapping.get(&row_id) {
                Some(Some(new_row_id)) => {
                    new_row_ids.push(*new_row_id);
                    new_freqs.push(freq);
                    if let Some(new_positions) = new_positions.as_mut() {
                        new_positions.push(positions.unwrap());
                    }
                }
                Some(None) => {
                    // remove the row_id
                    // do nothing
                }
                None => {
                    new_row_ids.push(row_id);
                    new_freqs.push(freq);
                    if let Some(new_positions) = new_positions.as_mut() {
                        new_positions.push(positions.unwrap());
                    }
                }
            }
        }

        self.row_ids = new_row_ids;
        self.frequencies = new_freqs;
        self.positions = new_positions;
    }

    // convert the posting list to a record batch
    // with docs, it would calculate the max score to accelerate the search
    pub fn to_batch(mut self, docs: Option<Arc<DocSet>>) -> Result<(RecordBatch, f32)> {
        let length = self.len();
        let num_docs = docs.as_ref().map(|docs| docs.len()).unwrap_or(0);
        let avgdl = docs
            .as_ref()
            .map(|docs| docs.average_length())
            .unwrap_or(0.0);
        let mut max_score = 0.0;

        let mut row_id_builder = UInt64Builder::with_capacity(length);
        let mut freq_builder = Float32Builder::with_capacity(length);
        let mut position_builder = self.positions.as_mut().map(|positions| {
            ListBuilder::with_capacity(Int32Builder::with_capacity(positions.total_len()), length)
        });
        for index in (0..length).sorted_unstable_by_key(|&i| self.row_ids[i]) {
            let (row_id, freq) = (self.row_ids[index], self.frequencies[index]);
            // reorder the posting list by row id
            row_id_builder.append_value(row_id);
            freq_builder.append_value(freq);
            if let Some(position_builder) = position_builder.as_mut() {
                let inner_builder = position_builder.values();
                inner_builder.append_slice(self.positions.as_ref().unwrap().get(index));
                position_builder.append(true);
            }
            // calculate the max score
            if let Some(docs) = &docs {
                let doc_norm = K1 * (1.0 - B + B * docs.num_tokens(row_id) as f32 / avgdl);
                let score = freq / (freq + doc_norm);
                if score > max_score {
                    max_score = score;
                }
            }
        }
        max_score *= idf(self.len(), num_docs) * (K1 + 1.0);

        let row_id_col = row_id_builder.finish();
        let freq_col = freq_builder.finish();

        let mut columns = vec![
            Arc::new(row_id_col) as ArrayRef,
            Arc::new(freq_col) as ArrayRef,
        ];
        let schema = inverted_list_schema(position_builder.is_some());
        if let Some(mut position_builder) = position_builder {
            let position_col = position_builder.finish();
            columns.push(Arc::new(position_col));
        }
        let batch = RecordBatch::try_new(schema, columns)?;
        Ok((batch, max_score))
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub struct PositionBuilder {
    positions: Vec<i32>,
    offsets: Vec<usize>,
}

impl Default for PositionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionBuilder {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            offsets: vec![0],
        }
    }

    pub fn size(&self) -> usize {
        std::mem::size_of::<i32>() * self.positions.len()
            + std::mem::size_of::<usize>() * self.offsets.len()
    }

    pub fn total_len(&self) -> usize {
        self.positions.len()
    }

    pub fn push(&mut self, positions: Vec<i32>) {
        self.positions.extend(positions);
        self.offsets.push(self.positions.len());
    }

    pub fn get(&self, i: usize) -> &[i32] {
        let start = self.offsets[i];
        let end = self.offsets[i + 1];
        &self.positions[start..end]
    }
}

impl From<Vec<Vec<i32>>> for PositionBuilder {
    fn from(positions: Vec<Vec<i32>>) -> Self {
        let mut builder = Self::new();
        builder.offsets.reserve(positions.len());
        for pos in positions {
            builder.push(pos);
        }
        builder
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
        self.total_tokens as f32 / self.len() as f32
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

struct FlatIndex<'a> {
    doc_col: &'a str,
    existing_index: &'a InvertedIndex,
    batch: &'a RecordBatch,
}

impl<'a> FlatIndex<'a> {
    pub fn new(
        doc_col: &'a str,
        existing_index: &'a InvertedIndex,
        batch: &'a RecordBatch,
    ) -> Self {
        Self {
            doc_col,
            existing_index,
            batch,
        }
    }
}

impl<'a> Searcher for FlatIndex<'a> {
    fn tokenizer(&self) -> tantivy::tokenizer::TextAnalyzer {
        self.existing_index.tokenizer()
    }

    fn expand_fuzzy(&self, tokens: Vec<String>, max_distance: Option<u32>) -> Result<Vec<String>> {
        self.existing_index.expand_fuzzy(tokens, max_distance)
    }

    async fn bm25_search(
        &self,
        tokens: Vec<String>,
        params: &FtsSearchParams,
        is_phrase_query: bool,
        prefilter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<(u64, f32)>> {
        let avgdl = self.existing_index.docs.average_length();
        let num_docs = self.existing_index.docs.len();

        let row_ids = self.batch[ROW_ID].as_primitive::<UInt64Type>().values();
        let doc_iter = iter_str_array(&self.batch[self.doc_col]);
        let mut res = Vec::with_capacity(self.batch.num_rows());
        for (doc_id, doc) in std::iter::zip(row_ids, doc_iter) {
            let Some(doc) = doc else {
                continue;
            };

            let query_tokens = tokens.iter().cloned().collect();
            let mut tokenizer = self.tokenizer();
            let doc_tokens = collect_tokens(doc, &mut tokenizer, Some(&query_tokens));
            let doc_norm = K1 * (1.0 - B + B * doc_tokens.len() as f32 / avgdl);
            let mut doc_token_count = HashMap::new();
            for token in doc_tokens {
                doc_token_count
                    .entry(token)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
            let mut score = 0.0;
            for token in tokens.iter() {
                let freq = doc_token_count.get(token).copied().unwrap_or_default() as f32;

                let idf = if let Some(token_id) = self.existing_index.tokens.get(token) {
                    // for known token, we just use the index's metadata to calculate the score
                    // it's not accurate but it's good enough for ranking
                    idf(
                        self.existing_index.inverted_list.posting_len(token_id),
                        num_docs,
                    )
                } else {
                    // for unknown token, we set the idf to a very high value
                    // so that the new token will significantly effect the score
                    idf(1, num_docs)
                };
                score += idf * (freq * (K1 + 1.0) / (freq + doc_norm));
            }

            if score > 0.0 {
                res.push((*doc_id, score));
            }
        }

        Ok(res)
    }
}

pub fn flat_full_text_search(
    batches: &[&RecordBatch],
    doc_col: &str,
    query: &str,
    tokenizer: Option<tantivy::tokenizer::TextAnalyzer>,
) -> Result<Vec<u64>> {
    if batches.is_empty() {
        return Ok(vec![]);
    }

    if is_phrase_query(query) {
        return Err(Error::invalid_input(
            "phrase query is not supported for flat full text search, try using FTS index",
            location!(),
        ));
    }

    match batches[0][doc_col].data_type() {
        DataType::Utf8 => do_flat_full_text_search::<i32>(batches, doc_col, query, tokenizer),
        DataType::LargeUtf8 => do_flat_full_text_search::<i64>(batches, doc_col, query, tokenizer),
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
    tokenizer: Option<tantivy::tokenizer::TextAnalyzer>,
) -> Result<Vec<u64>> {
    let mut results = Vec::new();
    let mut tokenizer = tokenizer.unwrap_or_else(|| TokenizerConfig::default().build().unwrap());
    let query_tokens = collect_tokens(query, &mut tokenizer, None)
        .into_iter()
        .collect::<HashSet<_>>();

    for batch in batches {
        let row_id_array = batch[ROW_ID].as_primitive::<UInt64Type>();
        let doc_array = batch[doc_col].as_string::<Offset>();
        for i in 0..row_id_array.len() {
            let doc = doc_array.value(i);
            let doc_tokens = collect_tokens(doc, &mut tokenizer, Some(&query_tokens));
            if !doc_tokens.is_empty() {
                results.push(row_id_array.value(i));
                assert!(doc.contains(query));
            }
        }
    }

    Ok(results)
}

// #[allow(clippy::too_many_arguments)]
// pub fn flat_bm25_search(
//     batch: RecordBatch,
//     doc_col: &str,
//     inverted_list: &InvertedListReader,
//     query_tokens: &Vec<String>,
//     query_token_ids: &HashMap<String, Option<u32>>,
//     tokenizer: &mut tantivy::tokenizer::TextAnalyzer,
//     avgdl: f32,
//     num_docs: usize,
// ) -> std::result::Result<RecordBatch, DataFusionError> {
//     let doc_iter = iter_str_array(&batch[doc_col]);
//     let mut scores = Vec::with_capacity(batch.num_rows());
//     for doc in doc_iter {
//         let Some(doc) = doc else {
//             scores.push(0.0);
//             continue;
//         };

//         let doc_tokens = collect_tokens(doc, tokenizer, Some(query_tokens));
//         let doc_norm = K1 * (1.0 - B + B * doc_tokens.len() as f32 / avgdl);
//         let mut doc_token_count = HashMap::new();
//         for token in doc_tokens {
//             doc_token_count
//                 .entry(token)
//                 .and_modify(|count| *count += 1)
//                 .or_insert(1);
//         }
//         let mut score = 0.0;
//         for (token, token_id) in query_token_ids.iter() {
//             let freq = doc_token_count.get(token).copied().unwrap_or_default() as f32;

//             let idf = if let Some(token_id) = token_id {
//                 // for known token, we just use the index's metadata to calculate the score
//                 // it's not accurate but it's good enough for ranking
//                 idf(inverted_list.posting_len(*token_id), num_docs)
//             } else {
//                 // for unknown token, we set the idf to a very high value
//                 // so that the new token will significantly effect the score
//                 idf(1, num_docs)
//             };
//             score += idf * (freq * (K1 + 1.0) / (freq + doc_norm));
//         }
//         scores.push(score);
//     }

//     let score_col = Arc::new(Float32Array::from(scores)) as ArrayRef;
//     let batch = batch
//         .try_with_column(SCORE_FIELD.clone(), score_col)?
//         .project_by_schema(&FTS_SCHEMA)?; // the scan node would probably scan some extra columns for prefilter, drop them here
//     Ok(batch)
// }

pub fn flat_bm25_search_stream(
    input: SendableRecordBatchStream,
    doc_col: String,
    query: FullTextSearchQuery,
    index: &InvertedIndex,
    metrics: &dyn MetricsCollector,
) -> SendableRecordBatchStream {
    let stream = input
        .map(move |batch| {
            let doc_col = doc_col.clone();
            let params = (&query).into();
            let query = query.query.clone();
            async move {
                let batch = batch?;
                let flat_index = FlatIndex::new(&doc_col, index, &batch);
                let res = process_compound_query(
                    &query,
                    &flat_index,
                    &params,
                    Arc::new(NoFilter),
                    metrics,
                )
                .await?;

                let batch = RecordBatch::try_new(
                    FTS_SCHEMA.clone(),
                    vec![
                        Arc::new(UInt64Array::from_iter_values(
                            res.iter().map(|(row_id, _)| *row_id),
                        )) as ArrayRef,
                        Arc::new(Float32Array::from_iter_values(
                            res.iter().map(|(_, score)| *score),
                        )) as ArrayRef,
                    ],
                )?;
                Ok(batch)
            }
        })
        .buffer_unordered(get_num_compute_intensive_cpus());

    Box::pin(RecordBatchStreamAdapter::new(
        FTS_SCHEMA.clone(),
        stream.boxed(),
    )) as SendableRecordBatchStream
}

pub fn is_phrase_query(query: &str) -> bool {
    query.starts_with('\"') && query.ends_with('\"')
}
