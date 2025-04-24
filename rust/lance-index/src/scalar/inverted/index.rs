// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::Arc;

use arrow::datatypes::{self, Float32Type, Int32Type, UInt64Type};
use arrow::{
    array::{AsArray, ListBuilder, StringBuilder, UInt32Builder},
    buffer::OffsetBuffer,
};
use arrow::{buffer::ScalarBuffer, datatypes::UInt32Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Int32Array, LargeBinaryArray, ListArray,
    OffsetSizeTrait, PrimitiveArray, RecordBatch, UInt32Array, UInt64Array,
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
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::utils::tracing::{IO_TYPE_LOAD_SCALAR_PART, TRACE_IO_EVENTS};
use lance_core::{Error, Result, ROW_ID, ROW_ID_FIELD};
use lazy_static::lazy_static;
use moka::future::Cache;
use roaring::RoaringBitmap;
use snafu::location;
use tracing::{info, instrument};

use super::query::*;
use super::{
    builder::{legacy_inverted_list_schema, PositionRecorder},
    encoding::CompressedPostingListHeader,
};
use super::{wand::*, InvertedIndexBuilder, TokenizerConfig};
use crate::prefilter::PreFilter;
use crate::scalar::{
    AnyQuery, IndexReader, IndexStore, MetricsCollector, SargableQuery, ScalarIndex, SearchResult,
};
use crate::Index;

pub const TOKENS_FILE: &str = "tokens.lance";
pub const INVERT_LIST_FILE: &str = "invert.lance";
pub const DOCS_FILE: &str = "docs.lance";
pub const METADATA_FILE: &str = "metadata.json";

pub const TOKEN_COL: &str = "_token";
pub const TOKEN_ID_COL: &str = "_token_id";
pub const FREQUENCY_COL: &str = "_frequency";
pub const POSITION_COL: &str = "_position";
pub const COMPRESSED_POSITION_COL: &str = "_compressed_position";
pub const POSTING_COL: &str = "_posting";
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
    params: TokenizerConfig,
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

    fn to_builder(&self) -> InvertedIndexBuilder {
        let tokens = self.tokens.clone().into_mut();
        let inverted_list = self.inverted_list.clone();
        let docs = self.docs.clone();
        InvertedIndexBuilder::from_existing_index(self.params.clone(), tokens, inverted_list, docs)
    }

    pub fn tokenizer(&self) -> tantivy::tokenizer::TextAnalyzer {
        self.tokenizer.clone()
    }

    pub fn expand_fuzzy(
        &self,
        tokens: Vec<String>,
        fuzziness: Option<u32>,
        max_expansions: usize,
    ) -> Result<Vec<String>> {
        let mut new_tokens = Vec::with_capacity(min(tokens.len(), max_expansions));
        for token in tokens {
            let fuzziness = match fuzziness {
                Some(fuzziness) => fuzziness,
                None => MatchQuery::auto_fuzziness(&token),
            };
            let lev =
                fst::automaton::Levenshtein::new(&token, fuzziness).map_err(|e| Error::Index {
                    message: format!("failed to construct the fuzzy query: {}", e),
                    location: location!(),
                })?;
            if let TokenMap::Fst(ref map) = self.tokens.tokens {
                let mut stream = map.search(lev).into_stream();
                while let Some((token, _)) = stream.next() {
                    new_tokens.push(String::from_utf8_lossy(token).into_owned());
                    if new_tokens.len() >= max_expansions {
                        break;
                    }
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
    pub async fn bm25_search(
        &self,
        tokens: &[String],
        params: &FtsSearchParams,
        is_phrase_query: bool,
        prefilter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        metrics.record_comparisons(tokens.len());

        let mask = prefilter.mask();
        let token_ids = self.map(tokens);
        if token_ids.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        if is_phrase_query && token_ids.len() != tokens.len() {
            return Ok((Vec::new(), Vec::new()));
        }

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
            params.limit.unwrap_or(usize::MAX),
            params.wand_factor,
            |doc, freq| {
                let doc_norm = K1
                    * (1.0 - B
                        + B * self.docs.num_tokens(&doc) as f32 / self.docs.average_length());
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
            "params": self.params,
            "num_tokens": self.tokens.tokens.len(),
            "num_docs": self.docs.len(),
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
        _metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query.as_any().downcast_ref::<SargableQuery>().unwrap();
        return Err(Error::invalid_input(
            format!("unsupported query {:?} for inverted index", query),
            location!(),
        ));
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
        Ok(Arc::new(Self {
            params: tokenizer_config,
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

    pub fn len(&self) -> usize {
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
                let (row_ids, frequencies) = Self::extract_columns(&batch);
                Result::Ok(PostingList::Plain(PlainPostingList::new(
                    row_ids,
                    frequencies,
                    self.max_scores
                        .as_ref()
                        .map(|max_scores| max_scores[token_id as usize]),
                )))
            })
            .await
            .map_err(|e| Error::io(e.to_string(), location!()))?;

        if is_phrase_query {
            // hit the cache and when the cache was populated, the positions column was not loaded
            let positions = self.read_positions(token_id).await?;
            posting.set_positions(positions);
        }

        Ok(posting)
    }

    fn extract_columns(batch: &RecordBatch) -> (ScalarBuffer<u64>, ScalarBuffer<f32>) {
        let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
        let frequencies = batch[FREQUENCY_COL].as_primitive::<Float32Type>();
        (row_ids.values().clone(), frequencies.values().clone())
    }

    async fn read_positions(&self, token_id: u32) -> Result<ListArray> {
        self.position_cache.try_get_with(token_id, async move {
            let length = self.posting_len(token_id);
            let token_id = token_id as usize;
            let offset = self.offsets[token_id];
            let batch = self
                .reader
                .read_range(offset..offset + length, Some(&[POSITION_COL]))
                .await.map_err(|e| {
                    match e {
                        Error::Schema { .. } => Error::Index {
                            message: "position is not found but required for phrase queries, try recreating the index with position".to_owned(), 
                            location: location!(),
                        },
                        e => e
                    }
                })?;
            Result::Ok(batch[POSITION_COL]
                .as_list::<i32>()
                .clone())
        }).await.map_err(|e| Error::io(e.to_string(), location!()))
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub enum PostingList {
    Plain(PlainPostingList),
    Compressed(CompressedPostingList),
}

impl PostingList {
    pub fn try_from_batch(batch: &RecordBatch) -> Result<Self> {
        match batch.column_by_name(POSTING_COL) {
            Some(_) => {
                unimplemented!("TODO")
            }
            None => {
                let posting = PlainPostingList::from_batch(batch)?;
                Ok(Self::Plain(posting))
            }
        }
    }

    pub fn set_positions(&mut self, positions: ListArray) {
        match self {
            PostingList::Plain(posting) => posting.positions = Some(positions),
            PostingList::Compressed(posting) => {
                posting.positions = Some(positions);
            }
        }
    }

    pub fn row_id(&self, i: usize) -> u64 {
        match self {
            PostingList::Plain(posting) => posting.row_id(i),
            PostingList::Compressed(posting) => {
                unimplemented!("compressed posting list does not support row_id")
            }
        }
    }

    pub fn doc(&self, i: usize) -> DocInfo {
        match self {
            PostingList::Plain(posting) => posting.doc(i),
            PostingList::Compressed(posting) => {
                unimplemented!("compressed posting list does not support doc")
            }
        }
    }

    pub fn positions(&self, row_id: u64) -> Option<PrimitiveArray<Int32Type>> {
        match self {
            PostingList::Plain(posting) => posting.positions(row_id),
            PostingList::Compressed(posting) => {
                unimplemented!("compressed posting list does not support positions")
            }
        }
    }

    pub fn max_score(&self) -> Option<f32> {
        match self {
            PostingList::Plain(posting) => posting.max_score,
            PostingList::Compressed(posting) => Some(posting.header.max_score),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            PostingList::Plain(posting) => posting.len(),
            PostingList::Compressed(posting) => posting.header.num_docs as usize,
        }
    }
}

impl Into<PostingListBuilder> for PostingList {
    fn into(self) -> PostingListBuilder {
        match self {
            PostingList::Plain(posting) => unimplemented!(),
            PostingList::Compressed(_) => unimplemented!(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct PlainPostingList {
    pub row_ids: ScalarBuffer<u64>,
    pub frequencies: ScalarBuffer<f32>,
    pub max_score: Option<f32>,
    pub positions: Option<ListArray>,
}

impl DeepSizeOf for PlainPostingList {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.row_ids.len() * std::mem::size_of::<u64>()
            + self.frequencies.len() * std::mem::size_of::<u32>()
            + self
                .positions
                .as_ref()
                .map(|positions| positions.get_array_memory_size())
                .unwrap_or(0)
    }
}

impl PlainPostingList {
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

    pub fn from_batch(batch: &RecordBatch) -> Result<Self> {
        let max_score = batch
            .metadata()
            .get("max_score")
            .map(|s| serde_json::from_str(s))
            .transpose()?;

        let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().values().clone();
        let frequencies = batch[FREQUENCY_COL]
            .as_primitive::<Float32Type>()
            .values()
            .clone();
        let positions = batch
            .column_by_name(POSITION_COL)
            .map(|col| col.as_list::<i32>().clone());

        Ok(Self {
            row_ids,
            frequencies,
            max_score,
            positions,
        })
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

#[derive(Debug, PartialEq, Clone)]
pub struct CompressedPostingList {
    // the first binary is the header
    // after that, each binary is a block of compressed data
    // that contains `BLOCK_SIZE` doc ids and then `BLOCK_SIZE` frequencies
    pub blocks: LargeBinaryArray,
    pub header: CompressedPostingListHeader,
    pub positions: Option<ListArray>,
}

impl DeepSizeOf for CompressedPostingList {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.blocks.get_array_memory_size()
            + self
                .positions
                .as_ref()
                .map(|positions| positions.get_array_memory_size())
                .unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct PostingListBuilder {
    pub doc_ids: Vec<u32>,
    pub frequencies: Vec<u32>,
    pub positions: Option<PositionBuilder>,
}

impl PostingListBuilder {
    pub fn size(&self) -> usize {
        std::mem::size_of::<u32>() * self.doc_ids.len()
            + std::mem::size_of::<f32>() * self.frequencies.len()
            + self
                .positions
                .as_ref()
                .map(|positions| positions.size())
                .unwrap_or(0)
    }

    pub fn try_from_batches(batches: &[RecordBatch]) -> Result<Self> {
        for batch in batches {
            let posting_list = PostingList::try_from_batch(batch)?;
        }
        unimplemented!()
        // Self {
        //     doc_ids: row_ids,
        //     frequencies,
        //     positions,
        // }
    }

    pub fn has_positions(&self) -> bool {
        self.positions.is_some()
    }

    pub fn empty(with_position: bool) -> Self {
        Self {
            doc_ids: Vec::new(),
            frequencies: Vec::new(),
            positions: with_position.then(PositionBuilder::new),
        }
    }

    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn add(&mut self, doc_id: u32, term_positions: PositionRecorder) {
        self.doc_ids.push(doc_id);
        self.frequencies.push(term_positions.len() as u32);
        if let Some(positions) = self.positions.as_mut() {
            positions.push(term_positions.into_vec());
        }
    }

    pub fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>, docs: &DocSet) {
        let mut new_doc_ids = Vec::with_capacity(self.len());
        let mut new_freqs = Vec::with_capacity(self.len());
        let mut new_positions = self.positions.as_mut().map(|_| PositionBuilder::new());

        for i in 0..self.len() {
            let doc_id = self.doc_ids[i];
            let row_id = docs.row_id(doc_id);
            let freq = self.frequencies[i];
            let positions = self
                .positions
                .as_ref()
                .map(|positions| positions.get(i).to_vec());

            match mapping.get(&row_id) {
                Some(Some(_)) => {
                    // the doc_id -> row_id mapping will be done in `DocSet`
                    new_doc_ids.push(doc_id);
                }
                Some(None) => {
                    // remove this doc
                    // do nothing
                }
                None => {
                    new_doc_ids.push(doc_id);
                    new_freqs.push(freq);
                    if let Some(new_positions) = new_positions.as_mut() {
                        new_positions.push(positions.unwrap());
                    }
                }
            }
        }

        self.doc_ids = new_doc_ids;
        self.frequencies = new_freqs;
        self.positions = new_positions;
    }

    // convert the posting list to a record batch
    // with docs, it would calculate the max score to accelerate the search
    pub fn to_batch(mut self, docs: Option<&DocSet>) -> Result<RecordBatch> {
        let length = self.len();
        let num_docs = docs.as_ref().map(|docs| docs.len()).unwrap_or(0);
        let avgdl = docs.map(|docs| docs.average_length()).unwrap_or(0.0);
        let mut max_score = 0.0;

        let mut doc_id_builder = UInt32Builder::with_capacity(length);
        let mut freq_builder = UInt32Builder::with_capacity(length);
        let mut position_builder = self.positions.as_mut().map(|positions| {
            ListBuilder::with_capacity(UInt32Builder::with_capacity(positions.total_len()), length)
        });
        for index in (0..length).sorted_unstable_by_key(|&i| self.doc_ids[i]) {
            let (doc_id, freq) = (self.doc_ids[index], self.frequencies[index]);
            // reorder the posting list by row id
            doc_id_builder.append_value(doc_id);
            freq_builder.append_value(freq);
            if let Some(position_builder) = position_builder.as_mut() {
                let inner_builder = position_builder.values();
                inner_builder.append_slice(self.positions.as_ref().unwrap().get(index));
                position_builder.append(true);
            }
            // calculate the max score
            if let Some(docs) = &docs {
                let doc_norm = K1 * (1.0 - B + B * docs.num_tokens(doc_id) as f32 / avgdl);
                let freq = freq as f32;
                let score = freq / (freq + doc_norm);
                if score > max_score {
                    max_score = score;
                }
            }
        }
        max_score *= idf(self.len(), num_docs) * (K1 + 1.0);

        let row_id_col = doc_id_builder.finish();
        let freq_col = freq_builder.finish();

        let mut columns = vec![
            Arc::new(row_id_col) as ArrayRef,
            Arc::new(freq_col) as ArrayRef,
        ];
        let schema = legacy_inverted_list_schema(position_builder.is_some());
        if let Some(mut position_builder) = position_builder {
            let position_col = position_builder.finish();
            columns.push(Arc::new(position_col));
        }
        let batch = RecordBatch::try_new(schema, columns)?;
        Ok(batch)
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub struct PositionBuilder {
    positions: Vec<u32>,
    offsets: Vec<i32>,
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
        std::mem::size_of::<u32>() * self.positions.len()
            + std::mem::size_of::<i32>() * (self.offsets.len() - 1)
    }

    pub fn total_len(&self) -> usize {
        self.positions.len()
    }

    pub fn push(&mut self, positions: Vec<u32>) {
        self.positions.extend(positions);
        self.offsets.push(self.positions.len() as i32);
    }

    pub fn get(&self, i: usize) -> &[u32] {
        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;
        &self.positions[start..end]
    }
}

impl From<Vec<Vec<u32>>> for PositionBuilder {
    fn from(positions: Vec<Vec<u32>>) -> Self {
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
    row_ids: Vec<u64>,
    num_tokens: Vec<u32>,
    total_tokens: u64,
}

impl DocSet {
    #[inline]
    pub fn len(&self) -> usize {
        self.row_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn row_id(&self, doc_id: u32) -> u64 {
        self.row_ids[doc_id as usize]
    }

    pub fn total_tokens_num(&self) -> u64 {
        self.total_tokens
    }

    #[inline]
    pub fn average_length(&self) -> f32 {
        self.total_tokens as f32 / self.len() as f32
    }

    pub fn to_batch(&self) -> Result<RecordBatch> {
        let row_id_col = UInt64Array::from(self.row_ids.clone());
        let num_tokens_col = UInt32Array::from(self.num_tokens.clone());

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
        let mut total_tokens = 0;

        let batch = reader.read_range(0..reader.num_rows(), None).await?;
        let row_id_col = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();
        let num_tokens_col = batch[NUM_TOKEN_COL].as_primitive::<datatypes::UInt32Type>();
        for &num_tokens in num_tokens_col.values() {
            total_tokens += num_tokens as u64;
        }

        Ok(Self {
            row_ids: row_id_col.values().to_vec(),
            num_tokens: num_tokens_col.values().to_vec(),
            total_tokens,
        })
    }

    pub fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) {
        for row_id in self.row_ids.iter_mut() {
            match mapping.get(row_id) {
                Some(Some(new_row_id)) => *row_id = *new_row_id,
                Some(None) => {
                    unimplemented!("TODO: handle the deletion case")
                }
                None => {}
            }
        }
    }

    #[inline]
    pub fn num_tokens(&self, doc_id: u32) -> u32 {
        self.num_tokens[doc_id as usize]
    }

    // append a new document to the doc set
    // returns the doc_id (the number of documents before appending)
    pub fn append(&mut self, row_id: u64, num_tokens: u32) -> u32 {
        self.row_ids.push(row_id);
        self.num_tokens.push(num_tokens);
        self.total_tokens += num_tokens as u64;
        self.row_ids.len() as u32 - 1
    }
}

#[inline]
pub fn idf(nq: usize, num_docs: usize) -> f32 {
    let num_docs = num_docs as f32;
    ((num_docs - nq as f32 + 0.5) / (nq as f32 + 0.5) + 1.0).ln()
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

#[allow(clippy::too_many_arguments)]
pub fn flat_bm25_search(
    batch: RecordBatch,
    doc_col: &str,
    inverted_list: &InvertedListReader,
    query_tokens: &HashSet<String>,
    query_token_ids: &HashMap<String, Option<u32>>,
    tokenizer: &mut tantivy::tokenizer::TextAnalyzer,
    avgdl: f32,
    num_docs: usize,
) -> std::result::Result<RecordBatch, DataFusionError> {
    let doc_iter = iter_str_array(&batch[doc_col]);
    let mut scores = Vec::with_capacity(batch.num_rows());
    for doc in doc_iter {
        let Some(doc) = doc else {
            scores.push(0.0);
            continue;
        };

        let doc_tokens = collect_tokens(doc, tokenizer, Some(query_tokens));
        let doc_norm = K1 * (1.0 - B + B * doc_tokens.len() as f32 / avgdl);
        let mut doc_token_count = HashMap::new();
        for token in doc_tokens {
            doc_token_count
                .entry(token)
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }
        let mut score = 0.0;
        for (token, token_id) in query_token_ids.iter() {
            let freq = doc_token_count.get(token).copied().unwrap_or_default() as f32;

            let idf = if let Some(token_id) = token_id {
                // for known token, we just use the index's metadata to calculate the score
                // it's not accurate but it's good enough for ranking
                idf(inverted_list.posting_len(*token_id), num_docs)
            } else {
                // for unknown token, we set the idf to a very high value
                // so that the new token will significantly effect the score
                idf(1, num_docs)
            };
            score += idf * (freq * (K1 + 1.0) / (freq + doc_norm));
        }
        scores.push(score);
    }

    let score_col = Arc::new(Float32Array::from(scores)) as ArrayRef;
    let batch = batch
        .try_with_column(SCORE_FIELD.clone(), score_col)?
        .project_by_schema(&FTS_SCHEMA)?; // the scan node would probably scan some extra columns for prefilter, drop them here
    Ok(batch)
}

pub fn flat_bm25_search_stream(
    input: SendableRecordBatchStream,
    doc_col: String,
    query: String,
    index: &InvertedIndex,
) -> SendableRecordBatchStream {
    let mut tokenizer = index.tokenizer.clone();
    let query_token_ids = collect_tokens(&query, &mut tokenizer, None)
        .into_iter()
        .dedup()
        .map(|token| {
            let token_id = index.tokens.get(&token);
            (token, token_id)
        })
        .collect::<HashMap<_, _>>();
    let query_tokens = query_token_ids.keys().cloned().collect::<HashSet<_>>();
    let inverted_list = index.inverted_list.clone();
    let num_docs = index.docs.len();
    let avgdl = index.docs.average_length();

    let stream = input.map(move |batch| {
        let batch = batch?;
        let scored_batch = flat_bm25_search(
            batch,
            &doc_col,
            inverted_list.as_ref(),
            &query_tokens,
            &query_token_ids,
            &mut tokenizer,
            avgdl,
            num_docs,
        )?;

        // filter out rows with score 0
        let score_col = scored_batch[SCORE_COL].as_primitive::<Float32Type>();
        let mask = score_col
            .iter()
            .map(|score| score.is_some_and(|score| score > 0.0))
            .collect::<Vec<_>>();
        let mask = BooleanArray::from(mask);
        let batch = arrow::compute::filter_record_batch(&scored_batch, &mask)?;
        Ok(batch)
    });

    Box::pin(RecordBatchStreamAdapter::new(FTS_SCHEMA.clone(), stream)) as SendableRecordBatchStream
}

pub fn is_phrase_query(query: &str) -> bool {
    query.starts_with('\"') && query.ends_with('\"')
}
