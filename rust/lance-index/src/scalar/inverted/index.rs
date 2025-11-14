// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt::{Debug, Display};
use std::sync::Arc;
use std::{
    cmp::{min, Reverse},
    collections::BinaryHeap,
};
use std::{collections::HashMap, ops::Range};

use crate::metrics::NoOpMetricsCollector;
use crate::prefilter::NoFilter;
use crate::scalar::registry::{TrainingCriteria, TrainingOrdering};
use arrow::datatypes::{self, Float32Type, Int32Type, UInt64Type};
use arrow::{
    array::{
        AsArray, LargeBinaryBuilder, ListBuilder, StringBuilder, UInt32Builder, UInt64Builder,
    },
    buffer::OffsetBuffer,
};
use arrow::{buffer::ScalarBuffer, datatypes::UInt32Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, LargeBinaryArray, ListArray, OffsetSizeTrait,
    RecordBatch, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_common::DataFusionError;
use deepsize::DeepSizeOf;
use fst::{Automaton, IntoStreamer, Streamer};
use futures::{stream, FutureExt, StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_arrow::{iter_str_array, RecordBatchExt};
use lance_core::cache::{CacheKey, LanceCache, WeakLanceCache};
use lance_core::utils::mask::RowIdTreeMap;
use lance_core::utils::{
    mask::RowIdMask,
    tracing::{IO_TYPE_LOAD_SCALAR_PART, TRACE_IO_EVENTS},
};
use lance_core::{container::list::ExpLinkedList, utils::tokio::get_num_compute_intensive_cpus};
use lance_core::{Error, Result, ROW_ID, ROW_ID_FIELD};
use roaring::RoaringBitmap;
use snafu::location;
use std::sync::LazyLock;
use tokio::task::spawn_blocking;
use tracing::{info, instrument};

use super::{
    builder::{
        doc_file_path, inverted_list_schema, posting_file_path, token_file_path, ScoredDoc,
        BLOCK_SIZE,
    },
    iter::PlainPostingListIterator,
    query::*,
    scorer::{idf, IndexBM25Scorer, Scorer, B, K1},
};
use super::{
    builder::{InnerBuilder, PositionRecorder},
    encoding::compress_posting_list,
    iter::CompressedPostingListIterator,
};
use super::{
    encoding::compress_positions,
    iter::{PostingListIterator, TokenIterator, TokenSource},
};
use super::{wand::*, InvertedIndexBuilder, InvertedIndexParams};
use crate::frag_reuse::FragReuseIndex;
use crate::pbold;
use crate::scalar::inverted::lance_tokenizer::TextTokenizer;
use crate::scalar::inverted::scorer::MemBM25Scorer;
use crate::scalar::inverted::tokenizer::lance_tokenizer::LanceTokenizer;
use crate::scalar::{
    AnyQuery, BuiltinIndexType, CreatedIndex, IndexReader, IndexStore, MetricsCollector,
    ScalarIndex, ScalarIndexParams, SearchResult, TokenQuery, UpdateCriteria,
};
use crate::Index;
use crate::{prefilter::PreFilter, scalar::inverted::iter::take_fst_keys};
use std::str::FromStr;

// Version 0: Arrow TokenSetFormat (legacy)
// Version 1: Fst TokenSetFormat (new default, incompatible clients < 0.38)
pub const INVERTED_INDEX_VERSION: u32 = 1;
pub const TOKENS_FILE: &str = "tokens.lance";
pub const INVERT_LIST_FILE: &str = "invert.lance";
pub const DOCS_FILE: &str = "docs.lance";
pub const METADATA_FILE: &str = "metadata.lance";

pub const TOKEN_COL: &str = "_token";
pub const TOKEN_ID_COL: &str = "_token_id";
pub const TOKEN_FST_BYTES_COL: &str = "_token_fst_bytes";
pub const TOKEN_NEXT_ID_COL: &str = "_token_next_id";
pub const TOKEN_TOTAL_LENGTH_COL: &str = "_token_total_length";
pub const FREQUENCY_COL: &str = "_frequency";
pub const POSITION_COL: &str = "_position";
pub const COMPRESSED_POSITION_COL: &str = "_compressed_position";
pub const POSTING_COL: &str = "_posting";
pub const MAX_SCORE_COL: &str = "_max_score";
pub const LENGTH_COL: &str = "_length";
pub const BLOCK_MAX_SCORE_COL: &str = "_block_max_score";
pub const NUM_TOKEN_COL: &str = "_num_tokens";
pub const SCORE_COL: &str = "_score";
pub const TOKEN_SET_FORMAT_KEY: &str = "token_set_format";

pub static SCORE_FIELD: LazyLock<Field> =
    LazyLock::new(|| Field::new(SCORE_COL, DataType::Float32, true));
pub static FTS_SCHEMA: LazyLock<SchemaRef> =
    LazyLock::new(|| Arc::new(Schema::new(vec![ROW_ID_FIELD.clone(), SCORE_FIELD.clone()])));
static ROW_ID_SCHEMA: LazyLock<SchemaRef> =
    LazyLock::new(|| Arc::new(Schema::new(vec![ROW_ID_FIELD.clone()])));

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub enum TokenSetFormat {
    Arrow,
    #[default]
    Fst,
}

impl Display for TokenSetFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Arrow => f.write_str("arrow"),
            Self::Fst => f.write_str("fst"),
        }
    }
}

impl FromStr for TokenSetFormat {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim() {
            "" => Ok(Self::Arrow),
            "arrow" => Ok(Self::Arrow),
            "fst" => Ok(Self::Fst),
            other => Err(Error::Index {
                message: format!("unsupported token set format {}", other),
                location: location!(),
            }),
        }
    }
}

impl DeepSizeOf for TokenSetFormat {
    fn deep_size_of_children(&self, _: &mut deepsize::Context) -> usize {
        0
    }
}

#[derive(Clone)]
pub struct InvertedIndex {
    params: InvertedIndexParams,
    store: Arc<dyn IndexStore>,
    tokenizer: Box<dyn LanceTokenizer>,
    token_set_format: TokenSetFormat,
    pub(crate) partitions: Vec<Arc<InvertedPartition>>,
}

impl Debug for InvertedIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedIndex")
            .field("params", &self.params)
            .field("token_set_format", &self.token_set_format)
            .field("partitions", &self.partitions)
            .finish()
    }
}

impl DeepSizeOf for InvertedIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.partitions.deep_size_of_children(context)
    }
}

impl InvertedIndex {
    fn to_builder(&self) -> InvertedIndexBuilder {
        self.to_builder_with_offset(None)
    }

    fn to_builder_with_offset(&self, fragment_mask: Option<u64>) -> InvertedIndexBuilder {
        if self.is_legacy() {
            // for legacy format, we re-create the index in the new format
            InvertedIndexBuilder::from_existing_index(
                self.params.clone(),
                None,
                Vec::new(),
                self.token_set_format,
                fragment_mask,
            )
        } else {
            let partitions = match fragment_mask {
                Some(fragment_mask) => self
                    .partitions
                    .iter()
                    // Filter partitions that belong to the specified fragment
                    // The mask contains fragment_id in high 32 bits, we check if partition's
                    // fragment_id matches by comparing the masked result with the original mask
                    .filter(|part| part.belongs_to_fragment(fragment_mask))
                    .map(|part| part.id())
                    .collect(),
                None => self.partitions.iter().map(|part| part.id()).collect(),
            };

            InvertedIndexBuilder::from_existing_index(
                self.params.clone(),
                Some(self.store.clone()),
                partitions,
                self.token_set_format,
                fragment_mask,
            )
        }
    }

    pub fn tokenizer(&self) -> Box<dyn LanceTokenizer> {
        self.tokenizer.clone()
    }

    pub fn params(&self) -> &InvertedIndexParams {
        &self.params
    }

    // search the documents that contain the query
    // return the row ids of the documents sorted by bm25 score
    // ref: https://en.wikipedia.org/wiki/Okapi_BM25
    // we first calculate in-partition BM25 scores,
    // then re-calculate the scores for the top k documents across all partitions
    #[instrument(level = "debug", skip_all)]
    pub async fn bm25_search(
        &self,
        tokens: Arc<Tokens>,
        params: Arc<FtsSearchParams>,
        operator: Operator,
        prefilter: Arc<dyn PreFilter>,
        metrics: Arc<dyn MetricsCollector>,
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        let limit = params.limit.unwrap_or(usize::MAX);
        if limit == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let mask = prefilter.mask();

        let mut candidates = BinaryHeap::new();
        let parts = self
            .partitions
            .iter()
            .map(|part| {
                let part = part.clone();
                let tokens = tokens.clone();
                let params = params.clone();
                let mask = mask.clone();
                let metrics = metrics.clone();
                tokio::spawn(async move {
                    part.bm25_search(
                        tokens.as_ref(),
                        params.as_ref(),
                        operator,
                        mask,
                        metrics.as_ref(),
                    )
                    .await
                })
            })
            .collect::<Vec<_>>();
        let mut parts = stream::iter(parts).buffer_unordered(get_num_compute_intensive_cpus());
        let scorer = IndexBM25Scorer::new(self.partitions.iter().map(|part| part.as_ref()));
        while let Some(res) = parts.try_next().await? {
            for DocCandidate {
                row_id,
                freqs,
                doc_length,
            } in res?
            {
                let mut score = 0.0;
                for (token, freq) in freqs.into_iter() {
                    score += scorer.score(token.as_str(), freq, doc_length);
                }
                if candidates.len() < limit {
                    candidates.push(Reverse(ScoredDoc::new(row_id, score)));
                } else if candidates.peek().unwrap().0.score.0 < score {
                    candidates.pop();
                    candidates.push(Reverse(ScoredDoc::new(row_id, score)));
                }
            }
        }

        Ok(candidates
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse(doc)| (doc.row_id, doc.score.0))
            .unzip())
    }

    async fn load_legacy_index(
        store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        index_cache: &LanceCache,
    ) -> Result<Arc<Self>> {
        log::warn!("loading legacy FTS index");
        let tokens_fut = tokio::spawn({
            let store = store.clone();
            async move {
                let token_reader = store.open_index_file(TOKENS_FILE).await?;
                let tokenizer = token_reader
                    .schema()
                    .metadata
                    .get("tokenizer")
                    .map(|s| serde_json::from_str::<InvertedIndexParams>(s))
                    .transpose()?
                    .unwrap_or_default();
                let tokens = TokenSet::load(token_reader, TokenSetFormat::Arrow).await?;
                Result::Ok((tokenizer, tokens))
            }
        });
        let invert_list_fut = tokio::spawn({
            let store = store.clone();
            let index_cache_clone = index_cache.clone();
            async move {
                let invert_list_reader = store.open_index_file(INVERT_LIST_FILE).await?;
                let invert_list =
                    PostingListReader::try_new(invert_list_reader, &index_cache_clone).await?;
                Result::Ok(Arc::new(invert_list))
            }
        });
        let docs_fut = tokio::spawn({
            let store = store.clone();
            async move {
                let docs_reader = store.open_index_file(DOCS_FILE).await?;
                let docs = DocSet::load(docs_reader, true, frag_reuse_index).await?;
                Result::Ok(docs)
            }
        });

        let (tokenizer_config, tokens) = tokens_fut.await??;
        let inverted_list = invert_list_fut.await??;
        let docs = docs_fut.await??;

        let tokenizer = tokenizer_config.build()?;

        Ok(Arc::new(Self {
            params: tokenizer_config,
            store: store.clone(),
            tokenizer,
            token_set_format: TokenSetFormat::Arrow,
            partitions: vec![Arc::new(InvertedPartition {
                id: 0,
                store,
                tokens,
                inverted_list,
                docs,
                token_set_format: TokenSetFormat::Arrow,
            })],
        }))
    }

    pub fn is_legacy(&self) -> bool {
        self.partitions.len() == 1 && self.partitions[0].is_legacy()
    }

    pub async fn load(
        store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        index_cache: &LanceCache,
    ) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        // for new index format, there is a metadata file and multiple partitions,
        // each partition is a separate index containing tokens, inverted list and docs.
        // for old index format, there is no metadata file, and it's just like a single partition

        match store.open_index_file(METADATA_FILE).await {
            Ok(reader) => {
                let params = reader.schema().metadata.get("params").ok_or(Error::Index {
                    message: "params not found in metadata".to_owned(),
                    location: location!(),
                })?;
                let params = serde_json::from_str::<InvertedIndexParams>(params)?;
                let partitions =
                    reader
                        .schema()
                        .metadata
                        .get("partitions")
                        .ok_or(Error::Index {
                            message: "partitions not found in metadata".to_owned(),
                            location: location!(),
                        })?;
                let partitions: Vec<u64> = serde_json::from_str(partitions)?;
                let token_set_format = reader
                    .schema()
                    .metadata
                    .get(TOKEN_SET_FORMAT_KEY)
                    .map(|name| TokenSetFormat::from_str(name))
                    .transpose()?
                    .unwrap_or(TokenSetFormat::Arrow);

                let format = token_set_format;
                let partitions = partitions.into_iter().map(|id| {
                    let store = store.clone();
                    let frag_reuse_index_clone = frag_reuse_index.clone();
                    let index_cache_for_part =
                        index_cache.with_key_prefix(format!("part-{}", id).as_str());
                    let token_set_format = format;
                    async move {
                        Result::Ok(Arc::new(
                            InvertedPartition::load(
                                store,
                                id,
                                frag_reuse_index_clone,
                                &index_cache_for_part,
                                token_set_format,
                            )
                            .await?,
                        ))
                    }
                });
                let partitions = stream::iter(partitions)
                    .buffer_unordered(store.io_parallelism())
                    .try_collect::<Vec<_>>()
                    .await?;

                let tokenizer = params.build()?;
                Ok(Arc::new(Self {
                    params,
                    store,
                    tokenizer,
                    token_set_format,
                    partitions,
                }))
            }
            Err(_) => {
                // old index format
                Self::load_legacy_index(store, frag_reuse_index, index_cache).await
            }
        }
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
        let num_tokens = self
            .partitions
            .iter()
            .map(|part| part.tokens.len())
            .sum::<usize>();
        let num_docs = self
            .partitions
            .iter()
            .map(|part| part.docs.len())
            .sum::<usize>();
        Ok(serde_json::json!({
            "params": self.params,
            "num_tokens": num_tokens,
            "num_docs": num_docs,
        }))
    }

    async fn prewarm(&self) -> Result<()> {
        for part in &self.partitions {
            part.inverted_list.prewarm().await?;
        }
        Ok(())
    }

    fn index_type(&self) -> crate::IndexType {
        crate::IndexType::Inverted
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}

impl InvertedIndex {
    /// Search docs match the input text.
    async fn do_search(&self, text: &str) -> Result<RecordBatch> {
        let params = FtsSearchParams::new();
        let mut tokenizer = self.tokenizer.clone();
        let tokens = collect_query_tokens(text, &mut tokenizer, None);

        let (doc_ids, _) = self
            .bm25_search(
                Arc::new(tokens),
                params.into(),
                Operator::And,
                Arc::new(NoFilter),
                Arc::new(NoOpMetricsCollector),
            )
            .boxed()
            .await?;

        Ok(RecordBatch::try_new(
            ROW_ID_SCHEMA.clone(),
            vec![Arc::new(UInt64Array::from(doc_ids))],
        )?)
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
        let query = query.as_any().downcast_ref::<TokenQuery>().unwrap();

        match query {
            TokenQuery::TokensContains(text) => {
                let records = self.do_search(text).await?;
                let row_ids = records
                    .column(0)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                let row_ids = row_ids.iter().flatten().collect_vec();
                Ok(SearchResult::AtMost(RowIdTreeMap::from_iter(row_ids)))
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
        self.to_builder()
            .remap(mapping, self.store.clone(), dest_store)
            .await?;

        let details = pbold::InvertedIndexDetails::try_from(&self.params)?;

        // Use version 0 for Arrow format (legacy), version 1 for Fst format (new)
        let index_version = match self.token_set_format {
            TokenSetFormat::Arrow => 0,
            TokenSetFormat::Fst => INVERTED_INDEX_VERSION,
        };

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&details).unwrap(),
            index_version,
        })
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        self.to_builder().update(new_data, dest_store).await?;

        let details = pbold::InvertedIndexDetails::try_from(&self.params)?;

        // Use version 0 for Arrow format (legacy), version 1 for Fst format (new)
        let index_version = match self.token_set_format {
            TokenSetFormat::Arrow => 0,
            TokenSetFormat::Fst => INVERTED_INDEX_VERSION,
        };

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&details).unwrap(),
            index_version,
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        let criteria = TrainingCriteria::new(TrainingOrdering::None).with_row_id();
        if self.is_legacy() {
            UpdateCriteria::requires_old_data(criteria)
        } else {
            UpdateCriteria::only_new_data(criteria)
        }
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        let mut params = self.params.clone();
        if params.base_tokenizer.is_empty() {
            params.base_tokenizer = "simple".to_string();
        }

        let params_json = serde_json::to_string(&params)?;

        Ok(ScalarIndexParams {
            index_type: BuiltinIndexType::Inverted.as_str().to_string(),
            params: Some(params_json),
        })
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub struct InvertedPartition {
    // 0 for legacy format
    id: u64,
    store: Arc<dyn IndexStore>,
    pub(crate) tokens: TokenSet,
    pub(crate) inverted_list: Arc<PostingListReader>,
    pub(crate) docs: DocSet,
    token_set_format: TokenSetFormat,
}

impl InvertedPartition {
    /// Check if this partition belongs to the specified fragment.
    ///
    /// This method encapsulates the bit manipulation logic for fragment filtering
    /// in distributed indexing scenarios.
    ///
    /// # Arguments
    /// * `fragment_mask` - A mask with fragment_id in high 32 bits
    ///
    /// # Returns
    /// * `true` if the partition belongs to the fragment, `false` otherwise
    pub fn belongs_to_fragment(&self, fragment_mask: u64) -> bool {
        (self.id() & fragment_mask) == fragment_mask
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn store(&self) -> &dyn IndexStore {
        self.store.as_ref()
    }

    pub fn is_legacy(&self) -> bool {
        self.inverted_list.lengths.is_none()
    }

    pub async fn load(
        store: Arc<dyn IndexStore>,
        id: u64,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        index_cache: &LanceCache,
        token_set_format: TokenSetFormat,
    ) -> Result<Self> {
        let token_file = store.open_index_file(&token_file_path(id)).await?;
        let tokens = TokenSet::load(token_file, token_set_format).await?;
        let invert_list_file = store.open_index_file(&posting_file_path(id)).await?;
        let inverted_list = PostingListReader::try_new(invert_list_file, index_cache).await?;
        let docs_file = store.open_index_file(&doc_file_path(id)).await?;
        let docs = DocSet::load(docs_file, false, frag_reuse_index).await?;

        Ok(Self {
            id,
            store,
            tokens,
            inverted_list: Arc::new(inverted_list),
            docs,
            token_set_format,
        })
    }

    fn map(&self, token: &str) -> Option<u32> {
        self.tokens.get(token)
    }

    pub fn expand_fuzzy(&self, tokens: &Tokens, params: &FtsSearchParams) -> Result<Tokens> {
        let mut new_tokens = Vec::with_capacity(min(tokens.len(), params.max_expansions));
        for token in tokens {
            let fuzziness = match params.fuzziness {
                Some(fuzziness) => fuzziness,
                None => MatchQuery::auto_fuzziness(token),
            };
            let lev =
                fst::automaton::Levenshtein::new(token, fuzziness).map_err(|e| Error::Index {
                    message: format!("failed to construct the fuzzy query: {}", e),
                    location: location!(),
                })?;

            let base_len = tokens.token_type().prefix_len(token) as u32;
            if let TokenMap::Fst(ref map) = self.tokens.tokens {
                match base_len + params.prefix_length {
                    0 => take_fst_keys(map.search(lev), &mut new_tokens, params.max_expansions),
                    prefix_length => {
                        let prefix = &token[..min(prefix_length as usize, token.len())];
                        let prefix = fst::automaton::Str::new(prefix).starts_with();
                        take_fst_keys(
                            map.search(lev.intersection(prefix)),
                            &mut new_tokens,
                            params.max_expansions,
                        )
                    }
                }
            } else {
                return Err(Error::Index {
                    message: "tokens is not fst, which is not expected".to_owned(),
                    location: location!(),
                });
            }
        }
        Ok(Tokens::new(new_tokens, tokens.token_type().clone()))
    }

    // search the documents that contain the query
    // return the doc info and the doc length
    // ref: https://en.wikipedia.org/wiki/Okapi_BM25
    #[instrument(level = "debug", skip_all)]
    pub async fn bm25_search(
        &self,
        tokens: &Tokens,
        params: &FtsSearchParams,
        operator: Operator,
        mask: Arc<RowIdMask>,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<DocCandidate>> {
        let is_fuzzy = matches!(params.fuzziness, Some(n) if n != 0);
        let is_phrase_query = params.phrase_slop.is_some();
        let tokens = match is_fuzzy {
            true => self.expand_fuzzy(tokens, params)?,
            false => tokens.clone(),
        };
        let mut token_ids = Vec::with_capacity(tokens.len());
        for token in tokens {
            let token_id = self.map(&token);
            if let Some(token_id) = token_id {
                token_ids.push((token_id, token));
            } else if is_phrase_query {
                // if the token is not found, we can't do phrase query
                return Ok(Vec::new());
            }
        }
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        if !is_phrase_query {
            // remove duplicates
            token_ids.sort_unstable_by_key(|(token_id, _)| *token_id);
            token_ids.dedup_by_key(|(token_id, _)| *token_id);
        }

        let num_docs = self.docs.len();
        let postings = stream::iter(token_ids)
            .enumerate()
            .map(|(position, (token_id, token))| async move {
                let posting = self
                    .inverted_list
                    .posting_list(token_id, is_phrase_query, metrics)
                    .await?;

                Result::Ok(PostingIterator::new(
                    token,
                    token_id,
                    position as u32,
                    posting,
                    num_docs,
                ))
            })
            .buffered(self.store.io_parallelism())
            .try_collect::<Vec<_>>()
            .await?;
        let scorer = IndexBM25Scorer::new(std::iter::once(self));
        let mut wand = Wand::new(operator, postings.into_iter(), &self.docs, scorer);
        wand.search(params, mask, metrics)
    }

    pub async fn into_builder(self) -> Result<InnerBuilder> {
        let mut builder = InnerBuilder::new(
            self.id,
            self.inverted_list.has_positions(),
            self.token_set_format,
        );
        builder.tokens = self.tokens;
        builder.docs = self.docs;

        builder
            .posting_lists
            .reserve_exact(self.inverted_list.len());
        for posting_list in self
            .inverted_list
            .read_all(self.inverted_list.has_positions())
            .await?
        {
            let posting_list = posting_list?;
            builder
                .posting_lists
                .push(posting_list.into_builder(&builder.docs));
        }
        Ok(builder)
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

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn iter(&self) -> TokenIterator<'_> {
        TokenIterator::new(match &self.tokens {
            TokenMap::HashMap(map) => TokenSource::HashMap(map.iter()),
            TokenMap::Fst(map) => TokenSource::Fst(map.stream()),
        })
    }

    pub fn to_batch(self, format: TokenSetFormat) -> Result<RecordBatch> {
        match format {
            TokenSetFormat::Arrow => self.into_arrow_batch(),
            TokenSetFormat::Fst => self.into_fst_batch(),
        }
    }

    fn into_arrow_batch(self) -> Result<RecordBatch> {
        let mut token_builder = StringBuilder::with_capacity(self.tokens.len(), self.total_length);
        let mut token_id_builder = UInt32Builder::with_capacity(self.tokens.len());

        match self.tokens {
            TokenMap::Fst(map) => {
                let mut stream = map.stream();
                while let Some((token, token_id)) = stream.next() {
                    token_builder.append_value(String::from_utf8_lossy(token));
                    token_id_builder.append_value(token_id as u32);
                }
            }
            TokenMap::HashMap(map) => {
                for (token, token_id) in map.into_iter().sorted_unstable() {
                    token_builder.append_value(token);
                    token_id_builder.append_value(token_id);
                }
            }
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

    fn into_fst_batch(mut self) -> Result<RecordBatch> {
        let fst_map = match std::mem::take(&mut self.tokens) {
            TokenMap::Fst(map) => map,
            TokenMap::HashMap(map) => Self::build_fst_from_map(map)?,
        };
        let bytes = fst_map.into_fst().into_inner();

        let mut fst_builder = LargeBinaryBuilder::with_capacity(1, bytes.len());
        fst_builder.append_value(bytes);
        let fst_col = fst_builder.finish();

        let mut next_id_builder = UInt32Builder::with_capacity(1);
        next_id_builder.append_value(self.next_id);
        let next_id_col = next_id_builder.finish();

        let mut total_length_builder = UInt64Builder::with_capacity(1);
        total_length_builder.append_value(self.total_length as u64);
        let total_length_col = total_length_builder.finish();

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(TOKEN_FST_BYTES_COL, DataType::LargeBinary, false),
            arrow_schema::Field::new(TOKEN_NEXT_ID_COL, DataType::UInt32, false),
            arrow_schema::Field::new(TOKEN_TOTAL_LENGTH_COL, DataType::UInt64, false),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(fst_col) as ArrayRef,
                Arc::new(next_id_col) as ArrayRef,
                Arc::new(total_length_col) as ArrayRef,
            ],
        )?;
        Ok(batch)
    }

    fn build_fst_from_map(map: HashMap<String, u32>) -> Result<fst::Map<Vec<u8>>> {
        let mut entries: Vec<_> = map.into_iter().collect();
        entries.sort_unstable_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
        let mut builder = fst::MapBuilder::memory();
        for (token, token_id) in entries {
            builder
                .insert(&token, token_id as u64)
                .map_err(|e| Error::Index {
                    message: format!("failed to insert token {}: {}", token, e),
                    location: location!(),
                })?;
        }
        Ok(builder.into_map())
    }

    pub async fn load(reader: Arc<dyn IndexReader>, format: TokenSetFormat) -> Result<Self> {
        match format {
            TokenSetFormat::Arrow => Self::load_arrow(reader).await,
            TokenSetFormat::Fst => Self::load_fst(reader).await,
        }
    }

    async fn load_arrow(reader: Arc<dyn IndexReader>) -> Result<Self> {
        let batch = reader.read_range(0..reader.num_rows(), None).await?;

        let (tokens, next_id, total_length) = spawn_blocking(move || {
            let mut next_id = 0;
            let mut total_length = 0;
            let mut tokens = fst::MapBuilder::memory();

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

            Ok::<_, Error>((tokens.into_map(), next_id, total_length))
        })
        .await
        .map_err(|err| Error::Execution {
            message: format!("failed to spawn blocking task: {}", err),
            location: location!(),
        })??;

        Ok(Self {
            tokens: TokenMap::Fst(tokens),
            next_id,
            total_length,
        })
    }

    async fn load_fst(reader: Arc<dyn IndexReader>) -> Result<Self> {
        let batch = reader.read_range(0..reader.num_rows(), None).await?;
        if batch.num_rows() == 0 {
            return Err(Error::Index {
                message: "token set batch is empty".to_owned(),
                location: location!(),
            });
        }

        let fst_col = batch[TOKEN_FST_BYTES_COL].as_binary::<i64>();
        let bytes = fst_col.value(0);
        let map = fst::Map::new(bytes.to_vec()).map_err(|e| Error::Index {
            message: format!("failed to load fst tokens: {}", e),
            location: location!(),
        })?;

        let next_id_col = batch[TOKEN_NEXT_ID_COL].as_primitive::<datatypes::UInt32Type>();
        let total_length_col =
            batch[TOKEN_TOTAL_LENGTH_COL].as_primitive::<datatypes::UInt64Type>();

        let next_id = next_id_col.values().first().copied().ok_or(Error::Index {
            message: "token next id column is empty".to_owned(),
            location: location!(),
        })?;

        let total_length = total_length_col
            .values()
            .first()
            .copied()
            .ok_or(Error::Index {
                message: "token total length column is empty".to_owned(),
                location: location!(),
            })?;

        Ok(Self {
            tokens: TokenMap::Fst(map),
            next_id,
            total_length: usize::try_from(total_length).map_err(|_| Error::Index {
                message: format!("token total length {} overflows usize", total_length),
                location: location!(),
            })?,
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

    // the `removed_token_ids` must be sorted
    pub fn remap(&mut self, removed_token_ids: &[u32]) {
        if removed_token_ids.is_empty() {
            return;
        }

        let mut map = match std::mem::take(&mut self.tokens) {
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

        map.retain(
            |_, token_id| match removed_token_ids.binary_search(token_id) {
                Ok(_) => false,
                Err(index) => {
                    *token_id -= index as u32;
                    true
                }
            },
        );

        self.tokens = TokenMap::HashMap(map);
    }

    pub fn next_id(&self) -> u32 {
        self.next_id
    }
}

pub struct PostingListReader {
    reader: Arc<dyn IndexReader>,

    // legacy format only
    offsets: Option<Vec<usize>>,

    // from metadata for legacy format
    // from column for new format
    max_scores: Option<Vec<f32>>,

    // new format only
    lengths: Option<Vec<u32>>,

    has_position: bool,

    index_cache: WeakLanceCache,
}

impl std::fmt::Debug for PostingListReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedListReader")
            .field("offsets", &self.offsets)
            .field("max_scores", &self.max_scores)
            .finish()
    }
}

impl DeepSizeOf for PostingListReader {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.offsets.deep_size_of_children(context)
            + self.max_scores.deep_size_of_children(context)
            + self.lengths.deep_size_of_children(context)
    }
}

impl PostingListReader {
    pub(crate) async fn try_new(
        reader: Arc<dyn IndexReader>,
        index_cache: &LanceCache,
    ) -> Result<Self> {
        let has_position = reader.schema().field(POSITION_COL).is_some();
        let (offsets, max_scores, lengths) = if reader.schema().field(POSTING_COL).is_none() {
            let (offsets, max_scores) = Self::load_metadata(reader.schema())?;
            (Some(offsets), max_scores, None)
        } else {
            let metadata = reader
                .read_range(0..reader.num_rows(), Some(&[MAX_SCORE_COL, LENGTH_COL]))
                .await?;
            let max_scores = metadata[MAX_SCORE_COL]
                .as_primitive::<Float32Type>()
                .values()
                .to_vec();
            let lengths = metadata[LENGTH_COL]
                .as_primitive::<UInt32Type>()
                .values()
                .to_vec();
            (None, Some(max_scores), Some(lengths))
        };

        Ok(Self {
            reader,
            offsets,
            max_scores,
            lengths,
            has_position,
            index_cache: WeakLanceCache::from(index_cache),
        })
    }

    // for legacy format
    // returns the offsets and max scores
    fn load_metadata(
        schema: &lance_core::datatypes::Schema,
    ) -> Result<(Vec<usize>, Option<Vec<f32>>)> {
        let offsets = schema.metadata.get("offsets").ok_or(Error::Index {
            message: "offsets not found in metadata".to_owned(),
            location: location!(),
        })?;
        let offsets = serde_json::from_str(offsets)?;

        let max_scores = schema
            .metadata
            .get("max_scores")
            .map(|max_scores| serde_json::from_str(max_scores))
            .transpose()?;
        Ok((offsets, max_scores))
    }

    // the number of posting lists
    pub fn len(&self) -> usize {
        match self.offsets {
            Some(ref offsets) => offsets.len(),
            None => self.reader.num_rows(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn has_positions(&self) -> bool {
        self.has_position
    }

    pub(crate) fn posting_len(&self, token_id: u32) -> usize {
        let token_id = token_id as usize;

        match self.offsets {
            Some(ref offsets) => {
                let next_offset = offsets
                    .get(token_id + 1)
                    .copied()
                    .unwrap_or(self.reader.num_rows());
                next_offset - offsets[token_id]
            }
            None => {
                if let Some(lengths) = &self.lengths {
                    lengths[token_id] as usize
                } else {
                    panic!("posting list reader is not initialized")
                }
            }
        }
    }

    pub(crate) async fn posting_batch(
        &self,
        token_id: u32,
        with_position: bool,
    ) -> Result<RecordBatch> {
        if self.offsets.is_some() {
            self.posting_batch_legacy(token_id, with_position).await
        } else {
            let token_id = token_id as usize;
            let columns = if with_position {
                vec![POSTING_COL, POSITION_COL]
            } else {
                vec![POSTING_COL]
            };
            let batch = self
                .reader
                .read_range(token_id..token_id + 1, Some(&columns))
                .await?;
            Ok(batch)
        }
    }

    async fn posting_batch_legacy(
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
        let offset = self.offsets.as_ref().unwrap()[token_id];
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
        let cache_key = PostingListKey { token_id };
        let mut posting = self
            .index_cache
            .get_or_insert_with_key(cache_key, || async move {
                metrics.record_part_load();
                info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_SCALAR_PART, index_type="inverted", part_id=token_id);
                let batch = self.posting_batch(token_id, false).await?;
                self.posting_list_from_batch(&batch, token_id)
            })
            .await
            .map_err(|e| Error::io(e.to_string(), location!()))?
            .as_ref()
            .clone();

        if is_phrase_query {
            // hit the cache and when the cache was populated, the positions column was not loaded
            let positions = self.read_positions(token_id).await?;
            posting.set_positions(positions);
        }

        Ok(posting)
    }

    pub(crate) fn posting_list_from_batch(
        &self,
        batch: &RecordBatch,
        token_id: u32,
    ) -> Result<PostingList> {
        let posting_list = PostingList::from_batch(
            batch,
            self.max_scores
                .as_ref()
                .map(|max_scores| max_scores[token_id as usize]),
            self.lengths
                .as_ref()
                .map(|lengths| lengths[token_id as usize]),
        )?;
        Ok(posting_list)
    }

    async fn prewarm(&self) -> Result<()> {
        let batch = self.read_batch(false).await?;
        for token_id in 0..self.len() {
            let posting_range = self.posting_list_range(token_id as u32);
            let batch = batch.slice(posting_range.start, posting_range.end - posting_range.start);
            // Apply shrink_to_fit to create a deep copy with compacted buffers
            // This ensures each cached entry has its own memory, not shared references
            let batch = batch.shrink_to_fit()?;
            let posting_list = self.posting_list_from_batch(&batch, token_id as u32)?;
            let inserted = self
                .index_cache
                .insert_with_key(
                    &PostingListKey {
                        token_id: token_id as u32,
                    },
                    Arc::new(posting_list),
                )
                .await;

            if !inserted {
                return Err(Error::Internal {
                    message: "Failed to prewarm index: cache is no longer available".to_string(),
                    location: location!(),
                });
            }
        }

        Ok(())
    }

    pub(crate) async fn read_batch(&self, with_position: bool) -> Result<RecordBatch> {
        let columns = self.posting_columns(with_position);
        let batch = self
            .reader
            .read_range(0..self.reader.num_rows(), Some(&columns))
            .await?;
        Ok(batch)
    }

    pub(crate) async fn read_all(
        &self,
        with_position: bool,
    ) -> Result<impl Iterator<Item = Result<PostingList>> + '_> {
        let batch = self.read_batch(with_position).await?;
        Ok((0..self.len()).map(move |i| {
            let token_id = i as u32;
            let range = self.posting_list_range(token_id);
            let batch = batch.slice(i, range.end - range.start);
            self.posting_list_from_batch(&batch, token_id)
        }))
    }

    async fn read_positions(&self, token_id: u32) -> Result<ListArray> {
        let positions = self.index_cache.get_or_insert_with_key(PositionKey { token_id }, || async move {
            let batch = self
                .reader
                .read_range(self.posting_list_range(token_id), Some(&[POSITION_COL]))
                .await.map_err(|e| {
                    match e {
                        Error::Schema { .. } => Error::invalid_input(
                            "position is not found but required for phrase queries, try recreating the index with position".to_owned(),
                            location!(),
                        ),
                        e => e
                    }
                })?;
            Result::Ok(Positions(batch[POSITION_COL]
                .as_list::<i32>()
                .clone()))
        }).await?;
        Ok(positions.0.clone())
    }

    fn posting_list_range(&self, token_id: u32) -> Range<usize> {
        match self.offsets {
            Some(ref offsets) => {
                let offset = offsets[token_id as usize];
                let posting_len = self.posting_len(token_id);
                offset..offset + posting_len
            }
            None => {
                let token_id = token_id as usize;
                token_id..token_id + 1
            }
        }
    }

    fn posting_columns(&self, with_position: bool) -> Vec<&'static str> {
        let mut base_columns = match self.offsets {
            Some(_) => vec![ROW_ID, FREQUENCY_COL],
            None => vec![POSTING_COL],
        };
        if with_position {
            base_columns.push(POSITION_COL);
        }
        base_columns
    }
}

/// New type just to allow Positions implement DeepSizeOf so it can be put
/// in the cache.
#[derive(Clone)]
pub struct Positions(ListArray);

impl DeepSizeOf for Positions {
    fn deep_size_of(&self) -> usize {
        self.0.get_array_memory_size()
    }

    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.0.get_buffer_memory_size()
    }
}

// Cache key implementations for type-safe cache access
#[derive(Debug, Clone)]
pub struct PostingListKey {
    pub token_id: u32,
}

impl CacheKey for PostingListKey {
    type ValueType = PostingList;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("postings-{}", self.token_id).into()
    }
}

#[derive(Debug, Clone)]
pub struct PositionKey {
    pub token_id: u32,
}

impl CacheKey for PositionKey {
    type ValueType = Positions;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("positions-{}", self.token_id).into()
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub enum PostingList {
    Plain(PlainPostingList),
    Compressed(CompressedPostingList),
}

impl PostingList {
    pub fn from_batch(
        batch: &RecordBatch,
        max_score: Option<f32>,
        length: Option<u32>,
    ) -> Result<Self> {
        match batch.column_by_name(POSTING_COL) {
            Some(_) => {
                debug_assert!(max_score.is_some() && length.is_some());
                let posting =
                    CompressedPostingList::from_batch(batch, max_score.unwrap(), length.unwrap());
                Ok(Self::Compressed(posting))
            }
            None => {
                let posting = PlainPostingList::from_batch(batch, max_score);
                Ok(Self::Plain(posting))
            }
        }
    }

    pub fn iter(&self) -> PostingListIterator<'_> {
        PostingListIterator::new(self)
    }

    pub fn has_position(&self) -> bool {
        match self {
            Self::Plain(posting) => posting.positions.is_some(),
            Self::Compressed(posting) => posting.positions.is_some(),
        }
    }

    pub fn set_positions(&mut self, positions: ListArray) {
        match self {
            Self::Plain(posting) => posting.positions = Some(positions),
            Self::Compressed(posting) => {
                posting.positions = Some(positions.value(0).as_list::<i32>().clone());
            }
        }
    }

    pub fn max_score(&self) -> Option<f32> {
        match self {
            Self::Plain(posting) => posting.max_score,
            Self::Compressed(posting) => Some(posting.max_score),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Plain(posting) => posting.len(),
            Self::Compressed(posting) => posting.length as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn into_builder(self, docs: &DocSet) -> PostingListBuilder {
        let mut builder = PostingListBuilder::new(self.has_position());
        match self {
            // legacy format
            Self::Plain(posting) => {
                // convert the posting list to the new format:
                // 1. map row ids to doc ids
                // 2. sort the posting list by doc ids
                struct Item {
                    doc_id: u32,
                    positions: PositionRecorder,
                }
                let doc_ids = docs
                    .row_ids
                    .iter()
                    .enumerate()
                    .map(|(doc_id, row_id)| (*row_id, doc_id as u32))
                    .collect::<HashMap<_, _>>();
                let mut items = Vec::with_capacity(posting.len());
                for (row_id, freq, positions) in posting.iter() {
                    let freq = freq as u32;
                    let positions = match positions {
                        Some(positions) => {
                            PositionRecorder::Position(positions.collect::<Vec<_>>())
                        }
                        None => PositionRecorder::Count(freq),
                    };
                    items.push(Item {
                        doc_id: doc_ids[&row_id],
                        positions,
                    });
                }
                items.sort_unstable_by_key(|item| item.doc_id);
                for item in items {
                    builder.add(item.doc_id, item.positions);
                }
            }
            Self::Compressed(posting) => {
                posting.iter().for_each(|(doc_id, freq, positions)| {
                    let positions = match positions {
                        Some(positions) => {
                            PositionRecorder::Position(positions.collect::<Vec<_>>())
                        }
                        None => PositionRecorder::Count(freq),
                    };
                    builder.add(doc_id, positions);
                });
            }
        }
        builder
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct PlainPostingList {
    pub row_ids: ScalarBuffer<u64>,
    pub frequencies: ScalarBuffer<f32>,
    pub max_score: Option<f32>,
    pub positions: Option<ListArray>, // List of Int32
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
        positions: Option<ListArray>,
    ) -> Self {
        Self {
            row_ids,
            frequencies,
            max_score,
            positions,
        }
    }

    pub fn from_batch(batch: &RecordBatch, max_score: Option<f32>) -> Self {
        let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().values().clone();
        let frequencies = batch[FREQUENCY_COL]
            .as_primitive::<Float32Type>()
            .values()
            .clone();
        let positions = batch
            .column_by_name(POSITION_COL)
            .map(|col| col.as_list::<i32>().clone());

        Self::new(row_ids, frequencies, max_score, positions)
    }

    pub fn len(&self) -> usize {
        self.row_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> PlainPostingListIterator<'_> {
        Box::new(
            self.row_ids
                .iter()
                .zip(self.frequencies.iter())
                .enumerate()
                .map(|(idx, (doc_id, freq))| {
                    (
                        *doc_id,
                        *freq,
                        self.positions.as_ref().map(|p| {
                            let start = p.value_offsets()[idx] as usize;
                            let end = p.value_offsets()[idx + 1] as usize;
                            Box::new(
                                p.values().as_primitive::<Int32Type>().values()[start..end]
                                    .iter()
                                    .map(|pos| *pos as u32),
                            ) as _
                        }),
                    )
                }),
        )
    }

    #[inline]
    pub fn doc(&self, i: usize) -> LocatedDocInfo {
        LocatedDocInfo::new(self.row_ids[i], self.frequencies[i])
    }

    pub fn positions(&self, index: usize) -> Option<Arc<dyn Array>> {
        self.positions
            .as_ref()
            .map(|positions| positions.value(index))
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
    pub max_score: f32,
    pub length: u32,
    // each binary is a block of compressed data
    // that contains `BLOCK_SIZE` doc ids and then `BLOCK_SIZE` frequencies
    pub blocks: LargeBinaryArray,
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

impl CompressedPostingList {
    pub fn new(
        blocks: LargeBinaryArray,
        max_score: f32,
        length: u32,
        positions: Option<ListArray>,
    ) -> Self {
        Self {
            max_score,
            length,
            blocks,
            positions,
        }
    }

    pub fn from_batch(batch: &RecordBatch, max_score: f32, length: u32) -> Self {
        debug_assert_eq!(batch.num_rows(), 1);
        let blocks = batch[POSTING_COL]
            .as_list::<i32>()
            .value(0)
            .as_binary::<i64>()
            .clone();
        let positions = batch
            .column_by_name(POSITION_COL)
            .map(|col| col.as_list::<i32>().value(0).as_list::<i32>().clone());

        Self {
            max_score,
            length,
            blocks,
            positions,
        }
    }

    pub fn iter(&self) -> CompressedPostingListIterator {
        CompressedPostingListIterator::new(
            self.length as usize,
            self.blocks.clone(),
            self.positions.clone(),
        )
    }

    pub fn block_max_score(&self, block_idx: usize) -> f32 {
        let block = self.blocks.value(block_idx);
        block[0..4].try_into().map(f32::from_le_bytes).unwrap()
    }

    pub fn block_least_doc_id(&self, block_idx: usize) -> u32 {
        let block = self.blocks.value(block_idx);
        block[4..8].try_into().map(u32::from_le_bytes).unwrap()
    }
}

#[derive(Debug)]
pub struct PostingListBuilder {
    pub doc_ids: ExpLinkedList<u32>,
    pub frequencies: ExpLinkedList<u32>,
    pub positions: Option<PositionBuilder>,
}

impl PostingListBuilder {
    pub fn size(&self) -> u64 {
        (std::mem::size_of::<u32>() * self.doc_ids.len()
            + std::mem::size_of::<u32>() * self.frequencies.len()
            + self
                .positions
                .as_ref()
                .map(|positions| positions.size())
                .unwrap_or(0)) as u64
    }

    pub fn has_positions(&self) -> bool {
        self.positions.is_some()
    }

    pub fn new(with_position: bool) -> Self {
        Self {
            doc_ids: ExpLinkedList::new().with_capacity_limit(128),
            frequencies: ExpLinkedList::new().with_capacity_limit(128),
            positions: with_position.then(PositionBuilder::new),
        }
    }

    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = (&u32, &u32, Option<&[u32]>)> {
        self.doc_ids
            .iter()
            .zip(self.frequencies.iter())
            .enumerate()
            .map(|(idx, (doc_id, freq))| {
                let positions = self.positions.as_ref().map(|positions| positions.get(idx));
                (doc_id, freq, positions)
            })
    }

    pub fn add(&mut self, doc_id: u32, term_positions: PositionRecorder) {
        self.doc_ids.push(doc_id);
        self.frequencies.push(term_positions.len());
        if let Some(positions) = self.positions.as_mut() {
            positions.push(term_positions.into_vec());
        }
    }

    // assume the posting list is sorted by doc id
    pub fn to_batch(self, block_max_scores: Vec<f32>) -> Result<RecordBatch> {
        let length = self.len();
        let max_score = block_max_scores.iter().copied().fold(f32::MIN, f32::max);

        let schema = inverted_list_schema(self.has_positions());
        let compressed = compress_posting_list(
            self.doc_ids.len(),
            self.doc_ids.iter(),
            self.frequencies.iter(),
            block_max_scores.into_iter(),
        )?;
        let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0, compressed.len() as i32]));
        let mut columns = vec![
            Arc::new(ListArray::try_new(
                Arc::new(Field::new("item", datatypes::DataType::LargeBinary, true)),
                offsets,
                Arc::new(compressed),
                None,
            )?) as ArrayRef,
            Arc::new(Float32Array::from_iter_values(std::iter::once(max_score))) as ArrayRef,
            Arc::new(UInt32Array::from_iter_values(std::iter::once(
                self.len() as u32
            ))) as ArrayRef,
        ];

        if let Some(positions) = self.positions.as_ref() {
            let mut position_builder = ListBuilder::new(ListBuilder::with_capacity(
                LargeBinaryBuilder::new(),
                length,
            ));
            for index in 0..length {
                let positions_in_doc = positions.get(index);
                let compressed = compress_positions(positions_in_doc)?;
                let inner_builder = position_builder.values();
                inner_builder.append_value(compressed.into_iter());
            }
            position_builder.append(true);
            let position_col = position_builder.finish();
            columns.push(Arc::new(position_col));
        }

        let batch = RecordBatch::try_new(schema, columns)?;
        Ok(batch)
    }

    pub fn remap(&mut self, removed: &[u32]) {
        let mut cursor = 0;
        let mut new_doc_ids = ExpLinkedList::with_capacity(self.len());
        let mut new_frequencies = ExpLinkedList::with_capacity(self.len());
        let mut new_positions = self.positions.as_mut().map(|_| PositionBuilder::new());
        for (&doc_id, &freq, positions) in self.iter() {
            while cursor < removed.len() && removed[cursor] < doc_id {
                cursor += 1;
            }
            if cursor < removed.len() && removed[cursor] == doc_id {
                // this doc is removed
                continue;
            }
            // there are cursor removed docs before this doc
            // so we need to shift the doc id
            new_doc_ids.push(doc_id - cursor as u32);
            new_frequencies.push(freq);
            if let Some(new_positions) = new_positions.as_mut() {
                new_positions.push(positions.unwrap().to_vec());
            }
        }

        self.doc_ids = new_doc_ids;
        self.frequencies = new_frequencies;
        self.positions = new_positions;
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
            + std::mem::size_of::<i32>() * self.offsets.len()
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

#[derive(Debug, Clone, DeepSizeOf, Copy)]
pub enum DocInfo {
    Located(LocatedDocInfo),
    Raw(RawDocInfo),
}

impl DocInfo {
    pub fn doc_id(&self) -> u64 {
        match self {
            Self::Raw(info) => info.doc_id as u64,
            Self::Located(info) => info.row_id,
        }
    }

    pub fn frequency(&self) -> u32 {
        match self {
            Self::Raw(info) => info.frequency,
            Self::Located(info) => info.frequency as u32,
        }
    }
}

impl Eq for DocInfo {}

impl PartialEq for DocInfo {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id() == other.doc_id()
    }
}

impl PartialOrd for DocInfo {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DocInfo {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.doc_id().cmp(&other.doc_id())
    }
}

#[derive(Debug, Clone, Default, DeepSizeOf, Copy)]
pub struct LocatedDocInfo {
    pub row_id: u64,
    pub frequency: f32,
}

impl LocatedDocInfo {
    pub fn new(row_id: u64, frequency: f32) -> Self {
        Self { row_id, frequency }
    }
}

impl Eq for LocatedDocInfo {}

impl PartialEq for LocatedDocInfo {
    fn eq(&self, other: &Self) -> bool {
        self.row_id == other.row_id
    }
}

impl PartialOrd for LocatedDocInfo {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LocatedDocInfo {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.row_id.cmp(&other.row_id)
    }
}

#[derive(Debug, Clone, Default, DeepSizeOf, Copy)]
pub struct RawDocInfo {
    pub doc_id: u32,
    pub frequency: u32,
}

impl RawDocInfo {
    pub fn new(doc_id: u32, frequency: u32) -> Self {
        Self { doc_id, frequency }
    }
}

impl Eq for RawDocInfo {}

impl PartialEq for RawDocInfo {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id == other.doc_id
    }
}

impl PartialOrd for RawDocInfo {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RawDocInfo {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.doc_id.cmp(&other.doc_id)
    }
}

// DocSet is a mapping from row ids to the number of tokens in the document
// It's used to sort the documents by the bm25 score
#[derive(Debug, Clone, Default, DeepSizeOf)]
pub struct DocSet {
    row_ids: Vec<u64>,
    num_tokens: Vec<u32>,
    // (row_id, doc_id) pairs sorted by row_id
    inv: Vec<(u64, u32)>,

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

    pub fn iter(&self) -> impl Iterator<Item = (&u64, &u32)> {
        self.row_ids.iter().zip(self.num_tokens.iter())
    }

    pub fn row_id(&self, doc_id: u32) -> u64 {
        self.row_ids[doc_id as usize]
    }

    pub fn doc_id(&self, row_id: u64) -> Option<u64> {
        if self.inv.is_empty() {
            // in legacy format, the row id is doc id
            match self.row_ids.binary_search(&row_id) {
                Ok(_) => Some(row_id),
                Err(_) => None,
            }
        } else {
            match self.inv.binary_search_by_key(&row_id, |x| x.0) {
                Ok(idx) => Some(self.inv[idx].1 as u64),
                Err(_) => None,
            }
        }
    }
    pub fn total_tokens_num(&self) -> u64 {
        self.total_tokens
    }

    #[inline]
    pub fn average_length(&self) -> f32 {
        self.total_tokens as f32 / self.len() as f32
    }

    pub fn calculate_block_max_scores<'a>(
        &self,
        doc_ids: impl Iterator<Item = &'a u32>,
        freqs: impl Iterator<Item = &'a u32>,
    ) -> Vec<f32> {
        let avgdl = self.average_length();
        let length = doc_ids.size_hint().0;
        let mut block_max_scores = Vec::with_capacity(length);
        let mut max_score = f32::MIN;
        for (i, (doc_id, freq)) in doc_ids.zip(freqs).enumerate() {
            let doc_norm = K1 * (1.0 - B + B * self.num_tokens(*doc_id) as f32 / avgdl);
            let freq = *freq as f32;
            let score = freq / (freq + doc_norm);
            if score > max_score {
                max_score = score;
            }
            if (i + 1) % BLOCK_SIZE == 0 {
                max_score *= idf(length, self.len()) * (K1 + 1.0);
                block_max_scores.push(max_score);
                max_score = f32::MIN;
            }
        }
        if length % BLOCK_SIZE > 0 {
            max_score *= idf(length, self.len()) * (K1 + 1.0);
            block_max_scores.push(max_score);
        }
        block_max_scores
    }

    pub fn to_batch(&self) -> Result<RecordBatch> {
        let row_id_col = UInt64Array::from_iter_values(self.row_ids.iter().cloned());
        let num_tokens_col = UInt32Array::from_iter_values(self.num_tokens.iter().cloned());

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

    pub async fn load(
        reader: Arc<dyn IndexReader>,
        is_legacy: bool,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let batch = reader.read_range(0..reader.num_rows(), None).await?;
        let row_id_col = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();
        let num_tokens_col = batch[NUM_TOKEN_COL].as_primitive::<datatypes::UInt32Type>();

        // for legacy format, the row id is doc id; sorting keeps binary search viable
        if is_legacy {
            let (row_ids, num_tokens): (Vec<_>, Vec<_>) = row_id_col
                .values()
                .iter()
                .filter_map(|id| {
                    if let Some(frag_reuse_index_ref) = frag_reuse_index.as_ref() {
                        frag_reuse_index_ref.remap_row_id(*id)
                    } else {
                        Some(*id)
                    }
                })
                .zip(num_tokens_col.values().iter())
                .sorted_unstable_by_key(|x| x.0)
                .unzip();

            let total_tokens = num_tokens.iter().map(|&x| x as u64).sum();
            return Ok(Self {
                row_ids,
                num_tokens,
                inv: Vec::new(),
                total_tokens,
            });
        }

        // if frag reuse happened, we'll need to remap the row_ids. And after row_ids been
        // remapped, we'll need resort to make sure binary_search works.
        if let Some(frag_reuse_index_ref) = frag_reuse_index.as_ref() {
            let mut row_ids = Vec::with_capacity(row_id_col.len());
            let mut num_tokens = Vec::with_capacity(num_tokens_col.len());
            for (row_id, num_token) in row_id_col.values().iter().zip(num_tokens_col.values()) {
                if let Some(new_row_id) = frag_reuse_index_ref.remap_row_id(*row_id) {
                    row_ids.push(new_row_id);
                    num_tokens.push(*num_token);
                }
            }

            let mut inv: Vec<(u64, u32)> = row_ids
                .iter()
                .enumerate()
                .map(|(doc_id, row_id)| (*row_id, doc_id as u32))
                .collect();
            inv.sort_unstable_by_key(|entry| entry.0);

            let total_tokens = num_tokens.iter().map(|&x| x as u64).sum();
            return Ok(Self {
                row_ids,
                num_tokens,
                inv,
                total_tokens,
            });
        }

        let row_ids = row_id_col.values().to_vec();
        let num_tokens = num_tokens_col.values().to_vec();
        let mut inv: Vec<(u64, u32)> = row_ids
            .iter()
            .enumerate()
            .map(|(doc_id, row_id)| (*row_id, doc_id as u32))
            .collect();
        if !row_ids.is_sorted() {
            inv.sort_unstable_by_key(|entry| entry.0);
        }
        let total_tokens = num_tokens.iter().map(|&x| x as u64).sum();
        Ok(Self {
            row_ids,
            num_tokens,
            inv,
            total_tokens,
        })
    }

    // remap the row ids to the new row ids
    // returns the removed doc ids
    pub fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) -> Vec<u32> {
        let mut removed = Vec::new();
        let len = self.len();
        let row_ids = std::mem::replace(&mut self.row_ids, Vec::with_capacity(len));
        let num_tokens = std::mem::replace(&mut self.num_tokens, Vec::with_capacity(len));
        for (doc_id, (row_id, num_token)) in std::iter::zip(row_ids, num_tokens).enumerate() {
            match mapping.get(&row_id) {
                Some(Some(new_row_id)) => {
                    self.row_ids.push(*new_row_id);
                    self.num_tokens.push(num_token);
                }
                Some(None) => {
                    removed.push(doc_id as u32);
                }
                None => {
                    self.row_ids.push(row_id);
                    self.num_tokens.push(num_token);
                }
            }
        }
        removed
    }

    #[inline]
    pub fn num_tokens(&self, doc_id: u32) -> u32 {
        self.num_tokens[doc_id as usize]
    }

    // this can be used only if it's a legacy format,
    // which store the sorted row ids so that we can use binary search
    #[inline]
    pub fn num_tokens_by_row_id(&self, row_id: u64) -> u32 {
        self.row_ids
            .binary_search(&row_id)
            .map(|idx| self.num_tokens[idx])
            .unwrap_or(0)
    }

    // append a document to the doc set
    // returns the doc_id (the number of documents before appending)
    pub fn append(&mut self, row_id: u64, num_tokens: u32) -> u32 {
        self.row_ids.push(row_id);
        self.num_tokens.push(num_tokens);
        self.total_tokens += num_tokens as u64;
        self.row_ids.len() as u32 - 1
    }
}

pub fn flat_full_text_search(
    batches: &[&RecordBatch],
    doc_col: &str,
    query: &str,
    tokenizer: Option<Box<dyn LanceTokenizer>>,
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
    tokenizer: Option<Box<dyn LanceTokenizer>>,
) -> Result<Vec<u64>> {
    let mut results = Vec::new();
    let mut tokenizer =
        tokenizer.unwrap_or_else(|| InvertedIndexParams::default().build().unwrap());
    let query_tokens = collect_query_tokens(query, &mut tokenizer, None);

    for batch in batches {
        let row_id_array = batch[ROW_ID].as_primitive::<UInt64Type>();
        let doc_array = batch[doc_col].as_string::<Offset>();
        for i in 0..row_id_array.len() {
            let doc = doc_array.value(i);
            let doc_tokens = collect_doc_tokens(doc, &mut tokenizer, Some(&query_tokens));
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
    query_tokens: &Tokens,
    tokenizer: &mut Box<dyn LanceTokenizer>,
    scorer: &mut MemBM25Scorer,
) -> std::result::Result<RecordBatch, DataFusionError> {
    let doc_iter = iter_str_array(&batch[doc_col]);
    let mut scores = Vec::with_capacity(batch.num_rows());
    for doc in doc_iter {
        let Some(doc) = doc else {
            scores.push(0.0);
            continue;
        };

        let doc_tokens = collect_doc_tokens(doc, tokenizer, None);
        scorer.update(&doc_tokens);
        let doc_tokens = doc_tokens
            .into_iter()
            .filter(|t| query_tokens.contains(t))
            .collect::<Vec<_>>();

        let doc_norm = K1 * (1.0 - B + B * doc_tokens.len() as f32 / scorer.avg_doc_length());
        let mut doc_token_count = HashMap::new();
        for token in doc_tokens {
            doc_token_count
                .entry(token)
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }
        let mut score = 0.0;
        for token in query_tokens {
            let freq = doc_token_count.get(token).copied().unwrap_or_default() as f32;

            let idf = idf(scorer.num_docs_containing_token(token), scorer.num_docs());
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
    index: &Option<InvertedIndex>,
) -> SendableRecordBatchStream {
    let mut tokenizer = match index {
        Some(index) => index.tokenizer(),
        None => Box::new(TextTokenizer::new(
            tantivy::tokenizer::TextAnalyzer::builder(
                tantivy::tokenizer::SimpleTokenizer::default(),
            )
            .build(),
        )),
    };
    let tokens = collect_query_tokens(&query, &mut tokenizer, None);

    let mut bm25_scorer = match index {
        Some(index) => {
            let index_bm25_scorer =
                IndexBM25Scorer::new(index.partitions.iter().map(|p| p.as_ref()));
            if index_bm25_scorer.num_docs() == 0 {
                MemBM25Scorer::new(0, 0, HashMap::new())
            } else {
                let mut token_docs = HashMap::with_capacity(tokens.len());
                for token in &tokens {
                    let token_nq = index_bm25_scorer.num_docs_containing_token(token).max(1);
                    token_docs.insert(token.clone(), token_nq);
                }
                MemBM25Scorer::new(
                    index_bm25_scorer.avg_doc_length() as u64 * index_bm25_scorer.num_docs() as u64,
                    index_bm25_scorer.num_docs(),
                    token_docs,
                )
            }
        }
        None => MemBM25Scorer::new(0, 0, HashMap::new()),
    };

    let stream = input.map(move |batch| {
        let batch = batch?;

        let batch = flat_bm25_search(batch, &doc_col, &tokens, &mut tokenizer, &mut bm25_scorer)?;

        // filter out rows with score 0
        let score_col = batch[SCORE_COL].as_primitive::<Float32Type>();
        let mask = score_col
            .iter()
            .map(|score| score.is_some_and(|score| score > 0.0))
            .collect::<Vec<_>>();
        let mask = BooleanArray::from(mask);
        let batch = arrow::compute::filter_record_batch(&batch, &mask)?;
        debug_assert!(batch[ROW_ID].null_count() == 0, "flat FTS produces nulls");
        Ok(batch)
    });

    Box::pin(RecordBatchStreamAdapter::new(FTS_SCHEMA.clone(), stream)) as SendableRecordBatchStream
}

pub fn is_phrase_query(query: &str) -> bool {
    query.starts_with('\"') && query.ends_with('\"')
}

#[cfg(test)]
mod tests {
    use crate::scalar::inverted::lance_tokenizer::DocType;
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_io::object_store::ObjectStore;

    use crate::metrics::NoOpMetricsCollector;
    use crate::prefilter::NoFilter;
    use crate::scalar::inverted::builder::{InnerBuilder, PositionRecorder};
    use crate::scalar::inverted::encoding::decompress_posting_list;
    use crate::scalar::inverted::query::{FtsSearchParams, Operator};
    use crate::scalar::lance_format::LanceIndexStore;

    use super::*;

    #[tokio::test]
    async fn test_posting_builder_remap() {
        let mut builder = PostingListBuilder::new(false);
        let n = BLOCK_SIZE + 3;
        for i in 0..n {
            builder.add(i as u32, PositionRecorder::Count(1));
        }
        let removed = vec![5, 7];
        builder.remap(&removed);

        let mut expected = PostingListBuilder::new(false);
        for i in 0..n - removed.len() {
            expected.add(i as u32, PositionRecorder::Count(1));
        }
        assert_eq!(builder.doc_ids, expected.doc_ids);
        assert_eq!(builder.frequencies, expected.frequencies);

        // BLOCK_SIZE + 3 elements should be reduced to BLOCK_SIZE + 1,
        // there are still 2 blocks.
        let batch = builder.to_batch(vec![1.0, 2.0]).unwrap();
        let (doc_ids, freqs) = decompress_posting_list(
            (n - removed.len()) as u32,
            batch[POSTING_COL]
                .as_list::<i32>()
                .value(0)
                .as_binary::<i64>(),
        )
        .unwrap();
        assert!(doc_ids
            .iter()
            .zip(expected.doc_ids.iter())
            .all(|(a, b)| a == b));
        assert!(freqs
            .iter()
            .zip(expected.frequencies.iter())
            .all(|(a, b)| a == b));
    }

    #[tokio::test]
    async fn test_remap_to_empty_posting_list() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());

        // index of docs:
        // 0: lance
        // 1: lake lake
        // 2: lake lake lake
        builder.tokens.add("lance".to_owned());
        builder.tokens.add("lake".to_owned());
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder.posting_lists[1].add(1, PositionRecorder::Count(2));
        builder.posting_lists[1].add(2, PositionRecorder::Count(3));
        builder.docs.append(0, 1);
        builder.docs.append(1, 1);
        builder.docs.append(2, 1);
        builder.write(store.as_ref()).await.unwrap();

        let index = InvertedPartition::load(
            store.clone(),
            0,
            None,
            &LanceCache::no_cache(),
            TokenSetFormat::default(),
        )
        .await
        .unwrap();
        let mut builder = index.into_builder().await.unwrap();

        let mapping = HashMap::from([(0, None), (2, Some(3))]);
        builder.remap(&mapping).await.unwrap();

        // after remap, the doc 0 is removed, and the doc 2 is updated to 3
        assert_eq!(builder.tokens.len(), 1);
        assert_eq!(builder.tokens.get("lake"), Some(0));
        assert_eq!(builder.posting_lists.len(), 1);
        assert_eq!(builder.posting_lists[0].len(), 2);
        assert_eq!(builder.docs.len(), 2);
        assert_eq!(builder.docs.row_id(0), 1);
        assert_eq!(builder.docs.row_id(1), 3);

        builder.write(store.as_ref()).await.unwrap();

        // remap to delete all docs
        let mapping = HashMap::from([(1, None), (3, None)]);
        builder.remap(&mapping).await.unwrap();

        assert_eq!(builder.tokens.len(), 0);
        assert_eq!(builder.posting_lists.len(), 0);
        assert_eq!(builder.docs.len(), 0);

        builder.write(store.as_ref()).await.unwrap();
    }

    #[tokio::test]
    async fn test_posting_cache_conflict_across_partitions() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create first partition with one token and posting list length 1
        let mut builder1 = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder1.tokens.add("test".to_owned());
        builder1.posting_lists.push(PostingListBuilder::new(false));
        builder1.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder1.docs.append(100, 1); // row_id=100, num_tokens=1
        builder1.write(store.as_ref()).await.unwrap();

        // Create second partition with one token and posting list length 4
        let mut builder2 = InnerBuilder::new(1, false, TokenSetFormat::default());
        builder2.tokens.add("test".to_owned()); // Use same token to test cache prefix fix
        builder2.posting_lists.push(PostingListBuilder::new(false));
        builder2.posting_lists[0].add(0, PositionRecorder::Count(2));
        builder2.posting_lists[0].add(1, PositionRecorder::Count(1));
        builder2.posting_lists[0].add(2, PositionRecorder::Count(3));
        builder2.posting_lists[0].add(3, PositionRecorder::Count(1));
        builder2.docs.append(200, 2); // row_id=200, num_tokens=2
        builder2.docs.append(201, 1); // row_id=201, num_tokens=1
        builder2.docs.append(202, 3); // row_id=202, num_tokens=3
        builder2.docs.append(203, 1); // row_id=203, num_tokens=1
        builder2.write(store.as_ref()).await.unwrap();

        // Create metadata file with both partitions
        let metadata = std::collections::HashMap::from_iter(vec![
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0u64, 1u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default()).unwrap(),
            ),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
        ]);
        let mut writer = store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();

        // Load the inverted index
        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();

        // Verify the index structure
        assert_eq!(index.partitions.len(), 2);
        assert_eq!(index.partitions[0].tokens.len(), 1);
        assert_eq!(index.partitions[1].tokens.len(), 1);

        // Verify the partitions were loaded correctly

        // Verify posting list lengths (note: partition order may differ from creation order)
        // Verify based on actual loading order
        if index.partitions[0].id() == 0 {
            // If partition[0] is ID=0, then it should have 1 document
            assert_eq!(index.partitions[0].inverted_list.posting_len(0), 1);
            assert_eq!(index.partitions[1].inverted_list.posting_len(0), 4);
            assert_eq!(index.partitions[0].docs.len(), 1);
            assert_eq!(index.partitions[1].docs.len(), 4);
        } else {
            // If partition[0] is ID=1, then it should have 4 documents
            assert_eq!(index.partitions[0].inverted_list.posting_len(0), 4);
            assert_eq!(index.partitions[1].inverted_list.posting_len(0), 1);
            assert_eq!(index.partitions[0].docs.len(), 4);
            assert_eq!(index.partitions[1].docs.len(), 1);
        }

        // Prewarm the inverted index (this loads posting lists into cache)
        index.prewarm().await.unwrap();

        let tokens = Arc::new(Tokens::new(vec!["test".to_string()], DocType::Text));
        let params = Arc::new(FtsSearchParams::new().with_limit(Some(10)));
        let prefilter = Arc::new(NoFilter);
        let metrics = Arc::new(NoOpMetricsCollector);

        let (row_ids, scores) = index
            .bm25_search(tokens, params, Operator::Or, prefilter, metrics)
            .await
            .unwrap();

        // Verify that we got search results
        // Expected to find 5 documents: 1 from first partition, 4 from second partition
        assert_eq!(row_ids.len(), 5, "row_ids: {:?}", row_ids);
        assert!(!row_ids.is_empty(), "Should find at least some documents");
        assert_eq!(row_ids.len(), scores.len());

        // All scores should be positive since all documents contain the search token
        for &score in &scores {
            assert!(score > 0.0, "All scores should be positive");
        }

        // Check that we got results from both partitions
        assert!(
            row_ids.contains(&100),
            "Should contain row_id from partition 0"
        );
        assert!(
            row_ids.iter().any(|&id| id >= 200),
            "Should contain row_id from partition 1"
        );
    }
}
