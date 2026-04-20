// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt::{Debug, Display};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::{
    cmp::{Reverse, min},
    collections::BinaryHeap,
};
use std::{
    collections::{HashMap, HashSet},
    ops::Range,
    time::Instant,
};

use crate::metrics::NoOpMetricsCollector;
use crate::prefilter::NoFilter;
use crate::scalar::registry::{TrainingCriteria, TrainingOrdering};
use arrow::array::{FixedSizeListBuilder, Float32Builder};
use arrow::datatypes::{self, Float32Type, Int32Type, UInt64Type};
use arrow::{
    array::{
        AsArray, LargeBinaryBuilder, ListBuilder, StringBuilder, UInt32Builder, UInt64Builder,
    },
    buffer::{Buffer, OffsetBuffer},
};
use arrow::{buffer::ScalarBuffer, datatypes::UInt32Type};
use arrow_array::{
    Array, ArrayRef, Float32Array, LargeBinaryArray, ListArray, OffsetSizeTrait, RecordBatch,
    UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use deepsize::DeepSizeOf;
use fst::{Automaton, IntoStreamer, Streamer};
use futures::{FutureExt, Stream, StreamExt, TryStreamExt, stream};
use itertools::Itertools;
use lance_arrow::{RecordBatchExt, iter_str_array};
use lance_core::cache::{CacheKey, LanceCache, WeakLanceCache};
use lance_core::error::{DataFusionResult, LanceOptionExt};
use lance_core::utils::mask::{RowAddrMask, RowAddrTreeMap};
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::utils::tracing::{IO_TYPE_LOAD_SCALAR_PART, TRACE_IO_EVENTS};
use lance_core::{Error, ROW_ID, ROW_ID_FIELD, Result};
use roaring::RoaringBitmap;
use std::sync::LazyLock;
use tokio::task::spawn_blocking;
use tracing::{info, instrument};

use super::encoding::PositionBlockBuilder;
use super::iter::PostingListIterator;
use super::{InvertedIndexBuilder, InvertedIndexParams, wand::*};
use super::{
    builder::{
        BLOCK_SIZE, ScoredDoc, doc_file_path, inverted_list_schema_for_version, posting_file_path,
        token_file_path,
    },
    iter::PlainPostingListIterator,
    query::*,
    scorer::{B, IndexBM25Scorer, K1, Scorer, idf},
};
use super::{
    builder::{InnerBuilder, PositionRecorder},
    iter::CompressedPostingListIterator,
};
use crate::frag_reuse::FragReuseIndex;
use crate::pbold;
use crate::scalar::inverted::scorer::MemBM25Scorer;
use crate::scalar::inverted::tokenizer::document_tokenizer::LanceTokenizer;
use crate::scalar::{
    AnyQuery, BuiltinIndexType, CreatedIndex, IndexReader, IndexStore, MetricsCollector,
    ScalarIndex, ScalarIndexParams, SearchResult, TokenQuery, UpdateCriteria,
};
use crate::{FtsPrewarmOptions, Index};
use crate::{prefilter::PreFilter, scalar::inverted::iter::take_fst_keys};
use std::str::FromStr;

// Version 0: Arrow TokenSetFormat (legacy)
// Version 1: Fst TokenSetFormat with per-doc compressed positions
// Version 2: Fst TokenSetFormat with shared posting-list position streams.
pub const INVERTED_INDEX_VERSION_V1: u32 = 1;
pub const INVERTED_INDEX_VERSION_V2: u32 = 2;
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
pub const POSITION_BLOCK_OFFSET_COL: &str = "_position_block_offset";
pub const POSTING_COL: &str = "_posting";
pub const MAX_SCORE_COL: &str = "_max_score";
pub const LENGTH_COL: &str = "_length";
pub const BLOCK_MAX_SCORE_COL: &str = "_block_max_score";
pub const NUM_TOKEN_COL: &str = "_num_tokens";
pub const SCORE_COL: &str = "_score";
pub const TOKEN_SET_FORMAT_KEY: &str = "token_set_format";
pub const POSTING_TAIL_CODEC_KEY: &str = "posting_tail_codec";
pub const POSITIONS_LAYOUT_KEY: &str = "positions_layout";
pub const POSITIONS_CODEC_KEY: &str = "positions_codec";
pub const POSTING_TAIL_CODEC_FIXED32_V1: &str = "fixed32_v1";
pub const POSTING_TAIL_CODEC_VARINT_DELTA_V1: &str = "varint_delta_v1";
pub const POSITIONS_LAYOUT_SHARED_STREAM_V2: &str = "shared_stream_v2";
pub const POSITIONS_CODEC_VARINT_DOC_DELTA_V2: &str = "varint_doc_delta_v2";
pub const POSITIONS_CODEC_PACKED_DELTA_V1: &str = "packed_delta_v1";
pub const DELETED_FRAGMENTS_COL: &str = "deleted_fragments";

// Just a heuristic when we need to pre-allocate memory for tokens
pub const ESTIMATED_MAX_TOKENS_PER_ROW: usize = 4 * 1024;

pub static SCORE_FIELD: LazyLock<Field> =
    LazyLock::new(|| Field::new(SCORE_COL, DataType::Float32, true));
pub static FTS_SCHEMA: LazyLock<SchemaRef> =
    LazyLock::new(|| Arc::new(Schema::new(vec![ROW_ID_FIELD.clone(), SCORE_FIELD.clone()])));
static ROW_ID_SCHEMA: LazyLock<SchemaRef> =
    LazyLock::new(|| Arc::new(Schema::new(vec![ROW_ID_FIELD.clone()])));

fn resolve_fts_format_version(
    value: Option<&str>,
) -> std::result::Result<InvertedListFormatVersion, Error> {
    value.unwrap_or("1").parse()
}

pub fn current_fts_format_version() -> InvertedListFormatVersion {
    resolve_fts_format_version(std::env::var("LANCE_FTS_FORMAT_VERSION").ok().as_deref())
        .expect("failed to parse LANCE_FTS_FORMAT_VERSION")
}

pub fn max_supported_fts_format_version() -> InvertedListFormatVersion {
    InvertedListFormatVersion::V2
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum InvertedListFormatVersion {
    #[default]
    V1,
    V2,
}

impl InvertedListFormatVersion {
    pub fn from_posting_tail_codec(codec: PostingTailCodec) -> Self {
        match codec {
            PostingTailCodec::Fixed32 => Self::V1,
            PostingTailCodec::VarintDelta => Self::V2,
        }
    }

    pub fn index_version(self) -> u32 {
        match self {
            Self::V1 => INVERTED_INDEX_VERSION_V1,
            Self::V2 => INVERTED_INDEX_VERSION_V2,
        }
    }

    pub fn posting_tail_codec(self) -> PostingTailCodec {
        match self {
            Self::V1 => PostingTailCodec::Fixed32,
            Self::V2 => PostingTailCodec::VarintDelta,
        }
    }

    pub fn position_codec(self) -> Option<PositionStreamCodec> {
        match self {
            Self::V1 => None,
            Self::V2 => Some(PositionStreamCodec::PackedDelta),
        }
    }

    pub fn uses_shared_position_stream(self) -> bool {
        matches!(self, Self::V2)
    }
}

impl FromStr for InvertedListFormatVersion {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim() {
            "1" | "v1" | "V1" => Ok(Self::V1),
            "2" | "v2" | "V2" => Ok(Self::V2),
            other => Err(Error::index(format!(
                "unsupported FTS format version {}, expected 1 or 2",
                other
            ))),
        }
    }
}

#[derive(Debug)]
struct PartitionCandidates {
    tokens_by_position: Vec<String>,
    candidates: Vec<DocCandidate>,
}

impl PartitionCandidates {
    fn empty() -> Self {
        Self {
            tokens_by_position: Vec::new(),
            candidates: Vec::new(),
        }
    }
}

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
            other => Err(Error::index(format!(
                "unsupported token set format {}",
                other
            ))),
        }
    }
}

impl DeepSizeOf for TokenSetFormat {
    fn deep_size_of_children(&self, _: &mut deepsize::Context) -> usize {
        0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PositionStreamCodec {
    VarintDocDelta,
    #[default]
    PackedDelta,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PostingTailCodec {
    Fixed32,
    #[default]
    VarintDelta,
}

impl PostingTailCodec {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fixed32 => POSTING_TAIL_CODEC_FIXED32_V1,
            Self::VarintDelta => POSTING_TAIL_CODEC_VARINT_DELTA_V1,
        }
    }

    fn from_metadata_value(value: &str) -> Result<Self> {
        match value.trim() {
            POSTING_TAIL_CODEC_FIXED32_V1 => Ok(Self::Fixed32),
            POSTING_TAIL_CODEC_VARINT_DELTA_V1 => Ok(Self::VarintDelta),
            other => Err(Error::index(format!(
                "unsupported posting tail codec {}",
                other
            ))),
        }
    }
}

pub(super) fn parse_posting_tail_codec(
    metadata: &HashMap<String, String>,
) -> Result<PostingTailCodec> {
    Ok(metadata
        .get(POSTING_TAIL_CODEC_KEY)
        .map(|codec| PostingTailCodec::from_metadata_value(codec))
        .transpose()?
        .unwrap_or(PostingTailCodec::Fixed32))
}

impl PositionStreamCodec {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::VarintDocDelta => POSITIONS_CODEC_VARINT_DOC_DELTA_V2,
            Self::PackedDelta => POSITIONS_CODEC_PACKED_DELTA_V1,
        }
    }

    fn from_metadata_value(value: &str) -> Result<Self> {
        match value.trim() {
            POSITIONS_CODEC_VARINT_DOC_DELTA_V2 => Ok(Self::VarintDocDelta),
            POSITIONS_CODEC_PACKED_DELTA_V1 => Ok(Self::PackedDelta),
            other => Err(Error::index(format!(
                "unsupported positions codec {}",
                other
            ))),
        }
    }
}

fn parse_shared_position_codec(metadata: &HashMap<String, String>) -> Result<PositionStreamCodec> {
    if let Some(codec) = metadata.get(POSITIONS_CODEC_KEY) {
        return PositionStreamCodec::from_metadata_value(codec);
    }

    match metadata
        .get(POSITIONS_LAYOUT_KEY)
        .map(|layout| layout.as_str())
    {
        Some(POSITIONS_LAYOUT_SHARED_STREAM_V2) => Ok(PositionStreamCodec::VarintDocDelta),
        _ => Ok(PositionStreamCodec::VarintDocDelta),
    }
}

pub(super) fn parse_format_version_from_metadata(
    metadata: &HashMap<String, String>,
) -> Result<InvertedListFormatVersion> {
    if metadata.contains_key(POSITIONS_CODEC_KEY) || metadata.contains_key(POSITIONS_LAYOUT_KEY) {
        return Ok(InvertedListFormatVersion::V2);
    }
    if parse_posting_tail_codec(metadata)? == PostingTailCodec::VarintDelta {
        Ok(InvertedListFormatVersion::V2)
    } else {
        Ok(InvertedListFormatVersion::V1)
    }
}

#[derive(Clone)]
pub struct InvertedIndex {
    params: InvertedIndexParams,
    store: Arc<dyn IndexStore>,
    tokenizer: Box<dyn LanceTokenizer>,
    token_set_format: TokenSetFormat,
    pub(crate) partitions: Vec<Arc<InvertedPartition>>,
    // Fragments which are contained in the index, but no longer in the dataset.
    // These should be pruned at search time since we don't prune them at update time.
    deleted_fragments: RoaringBitmap,
}

impl Debug for InvertedIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedIndex")
            .field("params", &self.params)
            .field("token_set_format", &self.token_set_format)
            .field("partitions", &self.partitions)
            .field("deleted_fragments", &self.deleted_fragments)
            .finish()
    }
}

impl DeepSizeOf for InvertedIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.partitions.deep_size_of_children(context)
    }
}

impl InvertedIndex {
    fn format_version(&self) -> InvertedListFormatVersion {
        self.partitions
            .first()
            .map(|partition| {
                InvertedListFormatVersion::from_posting_tail_codec(
                    partition.inverted_list.posting_tail_codec(),
                )
            })
            .unwrap_or_else(current_fts_format_version)
    }

    fn index_version(&self) -> u32 {
        match self.token_set_format {
            TokenSetFormat::Arrow => 0,
            TokenSetFormat::Fst => self.format_version().index_version(),
        }
    }

    fn posting_tail_codec(&self) -> PostingTailCodec {
        self.partitions
            .first()
            .map(|partition| partition.inverted_list.posting_tail_codec())
            .unwrap_or_default()
    }

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
                self.deleted_fragments.clone(),
            )
            .with_posting_tail_codec(self.posting_tail_codec())
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
                self.deleted_fragments.clone(),
            )
            .with_format_version(self.format_version())
        }
    }

    pub fn tokenizer(&self) -> Box<dyn LanceTokenizer> {
        self.tokenizer.clone()
    }

    pub fn params(&self) -> &InvertedIndexParams {
        &self.params
    }

    /// Returns the number of partitions in this inverted index.
    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }
    /// Returns the set of fragments which are contained in the index, but no longer in the dataset.
    ///
    /// Most other indices remove data from deleted fragments when the index updates (copy-on-write).
    /// However, this would require an expensive copy of the FTS index.  Instead, we track the deleted
    /// fragments and prune them at search time (merge-on-read).
    pub fn deleted_fragments(&self) -> &RoaringBitmap {
        &self.deleted_fragments
    }

    pub fn bm25_base_scorer(&self, query_tokens: &Tokens) -> MemBM25Scorer {
        let scorer = IndexBM25Scorer::new(self.partitions.iter().map(|part| part.as_ref()));
        let token_docs = query_tokens
            .into_iter()
            .map(|token| (token.to_string(), scorer.num_docs_containing_token(token)))
            .collect::<HashMap<_, _>>();
        MemBM25Scorer::new(scorer.total_tokens(), scorer.num_docs(), token_docs)
    }

    pub fn bm25_stats_for_terms(&self, terms: &[String]) -> (u64, usize, Vec<usize>) {
        let scorer = IndexBM25Scorer::new(self.partitions.iter().map(|part| part.as_ref()));
        let token_docs = terms
            .iter()
            .map(|term| scorer.num_docs_containing_token(term))
            .collect();
        (scorer.total_tokens(), scorer.num_docs(), token_docs)
    }

    /// Expand fuzzy query tokens against all partitions in this segment.
    pub fn expand_fuzzy_tokens(&self, tokens: &Tokens, params: &FtsSearchParams) -> Result<Tokens> {
        let mut expanded_tokens = Vec::new();
        let mut expanded_positions = Vec::new();
        let mut seen = HashSet::new();
        for partition in &self.partitions {
            let expanded = partition.expand_fuzzy(tokens, params)?;
            for idx in 0..expanded.len() {
                let token = expanded.get_token(idx);
                if seen.insert(token.to_string()) {
                    expanded_tokens.push(token.to_string());
                    expanded_positions.push(expanded.position(idx));
                }
            }
        }
        Ok(Tokens::with_positions(
            expanded_tokens,
            expanded_positions,
            tokens.token_type().clone(),
        ))
    }

    /// Search documents that match the query and return row ids sorted by BM25 score.
    ///
    /// When `base_scorer` is provided, search uses those corpus-level BM25 statistics
    /// instead of deriving them from this segment alone.
    #[instrument(level = "debug", skip_all)]
    pub async fn bm25_search(
        &self,
        tokens: Arc<Tokens>,
        params: Arc<FtsSearchParams>,
        operator: Operator,
        prefilter: Arc<dyn PreFilter>,
        metrics: Arc<dyn MetricsCollector>,
        base_scorer: Option<&MemBM25Scorer>,
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        let local_scorer;
        let scorer: &dyn Scorer = if let Some(base_scorer) = base_scorer {
            base_scorer
        } else {
            local_scorer = IndexBM25Scorer::new(self.partitions.iter().map(|part| part.as_ref()));
            &local_scorer
        };

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
                async move {
                    let postings = part
                        .load_posting_lists(tokens.as_ref(), params.as_ref(), metrics.as_ref())
                        .await?;
                    if postings.is_empty() {
                        return Result::Ok(PartitionCandidates::empty());
                    }
                    let max_position = postings
                        .iter()
                        .map(|posting| posting.term_index() as usize)
                        .max()
                        .unwrap_or_default();
                    let mut tokens_by_position = vec![String::new(); max_position + 1];
                    for posting in &postings {
                        let idx = posting.term_index() as usize;
                        tokens_by_position[idx] = posting.token().to_owned();
                    }
                    let params = params.clone();
                    let mask = mask.clone();
                    let metrics = metrics.clone();
                    spawn_cpu(move || {
                        let candidates = part.bm25_search(
                            params.as_ref(),
                            operator,
                            mask,
                            postings,
                            metrics.as_ref(),
                        )?;
                        Ok(PartitionCandidates {
                            tokens_by_position,
                            candidates,
                        })
                    })
                    .await
                }
            })
            .collect::<Vec<_>>();
        let mut parts = stream::iter(parts).buffer_unordered(get_num_compute_intensive_cpus());
        let mut idf_cache: HashMap<String, f32> = HashMap::new();
        while let Some(res) = parts.try_next().await? {
            if res.candidates.is_empty() {
                continue;
            }
            let mut idf_by_position = Vec::with_capacity(res.tokens_by_position.len());
            for token in &res.tokens_by_position {
                let idf_weight = match idf_cache.get(token) {
                    Some(weight) => *weight,
                    None => {
                        let weight = scorer.query_weight(token);
                        idf_cache.insert(token.clone(), weight);
                        weight
                    }
                };
                idf_by_position.push(idf_weight);
            }
            for DocCandidate {
                row_id,
                freqs,
                doc_length,
            } in res.candidates
            {
                let mut score = 0.0;
                for (term_index, freq) in freqs.into_iter() {
                    debug_assert!((term_index as usize) < idf_by_position.len());
                    score +=
                        idf_by_position[term_index as usize] * scorer.doc_weight(freq, doc_length);
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
            deleted_fragments: RoaringBitmap::new(),
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
                let params = reader
                    .schema()
                    .metadata
                    .get("params")
                    .ok_or(Error::index("params not found in metadata".to_owned()))?;
                let params = serde_json::from_str::<InvertedIndexParams>(params)?;
                let partitions = reader
                    .schema()
                    .metadata
                    .get("partitions")
                    .ok_or(Error::index("partitions not found in metadata".to_owned()))?;
                let partitions: Vec<u64> = serde_json::from_str(partitions)?;
                let token_set_format = reader
                    .schema()
                    .metadata
                    .get(TOKEN_SET_FORMAT_KEY)
                    .map(|name| TokenSetFormat::from_str(name))
                    .transpose()?
                    .unwrap_or(TokenSetFormat::Arrow);

                // Load deleted_fragments if present (optional for backward compatibility)
                let deleted_fragments = if reader.num_rows() > 0 {
                    let metadata_batch = reader.read_range(0..1, None).await?;
                    if let Some(col) = metadata_batch.column_by_name(DELETED_FRAGMENTS_COL) {
                        let arr = col.as_binary_opt::<i32>().expect_ok()?;
                        RoaringBitmap::deserialize_from(arr.value(0))?
                    } else {
                        RoaringBitmap::new()
                    }
                } else {
                    RoaringBitmap::new()
                };

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
                    deleted_fragments,
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
        self.prewarm_with_options(&FtsPrewarmOptions::default())
            .await
    }

    fn index_type(&self) -> crate::IndexType {
        crate::IndexType::Inverted
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}

impl InvertedIndex {
    pub async fn prewarm_with_options(&self, options: &FtsPrewarmOptions) -> Result<()> {
        let with_position = options.with_position;
        let io_parallelism = self.store.io_parallelism();
        let prewarm_futures = self
            .partitions
            .iter()
            .map(Arc::clone)
            .map(|part| async move {
                part.inverted_list
                    .prewarm_posting_lists(with_position)
                    .await?;
                Result::Ok(())
            });
        stream::iter(prewarm_futures)
            .buffer_unordered(io_parallelism)
            .try_collect::<Vec<_>>()
            .await?;
        Ok(())
    }
    /// Search docs match the input text.
    async fn do_search(&self, text: &str) -> Result<RecordBatch> {
        let params = FtsSearchParams::new();
        let mut tokenizer = self.tokenizer.clone();
        let tokens = collect_query_tokens(text, &mut tokenizer);

        let (doc_ids, _) = self
            .bm25_search(
                Arc::new(tokens),
                params.into(),
                Operator::And,
                Arc::new(NoFilter),
                Arc::new(NoOpMetricsCollector),
                None,
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
                Ok(SearchResult::at_most(RowAddrTreeMap::from_iter(row_ids)))
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

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&details).unwrap(),
            index_version: self.index_version(),
            files: Some(dest_store.list_files_with_sizes().await?),
        })
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        old_data_filter: Option<crate::scalar::OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        self.to_builder()
            .update(new_data, dest_store, old_data_filter)
            .await?;

        let details = pbold::InvertedIndexDetails::try_from(&self.params)?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&details).unwrap(),
            index_version: self.index_version(),
            files: Some(dest_store.list_files_with_sizes().await?),
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
            let lev = fst::automaton::Levenshtein::new(token, fuzziness)
                .map_err(|e| Error::index(format!("failed to construct the fuzzy query: {}", e)))?;

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
                return Err(Error::index(
                    "tokens is not fst, which is not expected".to_owned(),
                ));
            }
        }
        Ok(Tokens::new(new_tokens, tokens.token_type().clone()))
    }

    // search the documents that contain the query
    // return the doc info and the doc length
    // ref: https://en.wikipedia.org/wiki/Okapi_BM25
    #[instrument(level = "debug", skip_all)]
    pub async fn load_posting_lists(
        &self,
        tokens: &Tokens,
        params: &FtsSearchParams,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<PostingIterator>> {
        let is_fuzzy = matches!(params.fuzziness, Some(n) if n != 0);
        let is_phrase_query = params.phrase_slop.is_some();
        let tokens = match is_fuzzy {
            true => self.expand_fuzzy(tokens, params)?,
            false => tokens.clone(),
        };
        let token_positions = (0..tokens.len())
            .map(|index| tokens.position(index))
            .collect::<Vec<_>>();
        let mut token_ids = Vec::with_capacity(tokens.len());
        for (index, token) in tokens.into_iter().enumerate() {
            let token_id = self.map(&token);
            if let Some(token_id) = token_id {
                token_ids.push((token_id, token, token_positions[index]));
            } else if is_phrase_query {
                // if the token is not found, we can't do phrase query
                return Ok(Vec::new());
            }
        }
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        if !is_phrase_query {
            token_ids.sort_unstable_by_key(|(token_id, _, _)| *token_id);
            token_ids.dedup_by_key(|(token_id, _, _)| *token_id);
        }

        let num_docs = self.docs.len();
        stream::iter(token_ids)
            .map(|(token_id, token, position)| async move {
                let posting = self
                    .inverted_list
                    .posting_list(token_id, is_phrase_query, metrics)
                    .await?;

                let query_weight = idf(posting.len(), num_docs);

                Result::Ok(PostingIterator::with_query_weight(
                    token,
                    token_id,
                    position,
                    query_weight,
                    posting,
                    num_docs,
                ))
            })
            .buffered(self.store.io_parallelism())
            .try_collect::<Vec<_>>()
            .await
    }

    #[instrument(level = "debug", skip_all)]
    pub fn bm25_search(
        &self,
        params: &FtsSearchParams,
        operator: Operator,
        mask: Arc<RowAddrMask>,
        postings: Vec<PostingIterator>,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<DocCandidate>> {
        if postings.is_empty() {
            return Ok(Vec::new());
        }

        // let local_metrics = LocalMetricsCollector::default();
        let scorer = IndexBM25Scorer::new(std::iter::once(self));
        let mut wand = Wand::new(operator, postings.into_iter(), &self.docs, scorer);
        let hits = wand.search(params, mask, metrics)?;
        // local_metrics.dump_into(metrics);
        Ok(hits)
    }

    pub async fn into_builder(self) -> Result<InnerBuilder> {
        let mut builder = InnerBuilder::new_with_posting_tail_codec(
            self.id,
            self.inverted_list.has_positions(),
            self.token_set_format,
            self.inverted_list.posting_tail_codec(),
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
                .map_err(|e| Error::index(format!("failed to insert token {}: {}", token, e)))?;
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
                let token =
                    token.ok_or(Error::index("found null token in token set".to_owned()))?;
                next_id = next_id.max(token_id + 1);
                total_length += token.len();
                tokens.insert(token, token_id as u64).map_err(|e| {
                    Error::index(format!("failed to insert token {}: {}", token, e))
                })?;
            }

            Ok::<_, Error>((tokens.into_map(), next_id, total_length))
        })
        .await
        .map_err(|err| Error::execution(format!("failed to spawn blocking task: {}", err)))??;

        Ok(Self {
            tokens: TokenMap::Fst(tokens),
            next_id,
            total_length,
        })
    }

    async fn load_fst(reader: Arc<dyn IndexReader>) -> Result<Self> {
        let batch = reader.read_range(0..reader.num_rows(), None).await?;
        if batch.num_rows() == 0 {
            return Err(Error::index("token set batch is empty".to_owned()));
        }

        let fst_col = batch[TOKEN_FST_BYTES_COL].as_binary::<i64>();
        let bytes = fst_col.value(0);
        let map = fst::Map::new(bytes.to_vec())
            .map_err(|e| Error::index(format!("failed to load fst tokens: {}", e)))?;

        let next_id_col = batch[TOKEN_NEXT_ID_COL].as_primitive::<datatypes::UInt32Type>();
        let total_length_col =
            batch[TOKEN_TOTAL_LENGTH_COL].as_primitive::<datatypes::UInt64Type>();

        let next_id = next_id_col
            .values()
            .first()
            .copied()
            .ok_or(Error::index("token next id column is empty".to_owned()))?;

        let total_length = total_length_col
            .values()
            .first()
            .copied()
            .ok_or(Error::index(
                "token total length column is empty".to_owned(),
            ))?;

        Ok(Self {
            tokens: TokenMap::Fst(map),
            next_id,
            total_length: usize::try_from(total_length).map_err(|_| {
                Error::index(format!(
                    "token total length {} overflows usize",
                    total_length
                ))
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

    pub(crate) fn get_or_add(&mut self, token: &str) -> u32 {
        let next_id = self.next_id;
        match self.tokens {
            TokenMap::HashMap(ref mut map) => {
                if let Some(&token_id) = map.get(token) {
                    return token_id;
                }

                map.insert(token.to_owned(), next_id);
            }
            _ => unreachable!("tokens must be HashMap while indexing"),
        }

        self.next_id += 1;
        self.total_length += token.len();
        next_id
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

    pub(crate) fn memory_size(&self) -> usize {
        match &self.tokens {
            TokenMap::HashMap(map) => {
                self.total_length
                    + map.capacity()
                        * (std::mem::size_of::<String>()
                            + std::mem::size_of::<u32>()
                            + std::mem::size_of::<usize>())
            }
            TokenMap::Fst(map) => map.as_fst().size(),
        }
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
    posting_tail_codec: PostingTailCodec,
    positions_layout: PositionsLayout,

    index_cache: WeakLanceCache,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PositionsLayout {
    None,
    LegacyPerDoc,
    SharedStream(PositionStreamCodec),
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
        let positions_layout = if reader.schema().field(COMPRESSED_POSITION_COL).is_some() {
            PositionsLayout::SharedStream(parse_shared_position_codec(&reader.schema().metadata)?)
        } else if reader.schema().field(POSITION_COL).is_some() {
            PositionsLayout::LegacyPerDoc
        } else {
            PositionsLayout::None
        };
        let posting_tail_codec = parse_posting_tail_codec(&reader.schema().metadata)?;
        let has_position = positions_layout != PositionsLayout::None;
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
            posting_tail_codec,
            positions_layout,
            index_cache: WeakLanceCache::from(index_cache),
        })
    }

    // for legacy format
    // returns the offsets and max scores
    fn load_metadata(
        schema: &lance_core::datatypes::Schema,
    ) -> Result<(Vec<usize>, Option<Vec<f32>>)> {
        let offsets = schema
            .metadata
            .get("offsets")
            .ok_or(Error::index("offsets not found in metadata".to_owned()))?;
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

    pub(crate) fn posting_tail_codec(&self) -> PostingTailCodec {
        self.posting_tail_codec
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
                match self.positions_layout {
                    PositionsLayout::SharedStream(_) => {
                        vec![
                            POSTING_COL,
                            COMPRESSED_POSITION_COL,
                            POSITION_BLOCK_OFFSET_COL,
                        ]
                    }
                    PositionsLayout::LegacyPerDoc => vec![POSTING_COL, POSITION_COL],
                    PositionsLayout::None => vec![POSTING_COL],
                }
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
            .await?
            .as_ref()
            .clone();

        if is_phrase_query && !posting.has_position() {
            // hit the cache and when the cache was populated, the positions column was not loaded
            let positions = self.read_positions(token_id).await?;
            posting.set_positions(positions);
        }

        Ok(posting)
    }

    fn posting_list_from_batch_parts(
        batch: &RecordBatch,
        max_score: Option<f32>,
        length: Option<u32>,
        posting_tail_codec: PostingTailCodec,
        positions_layout: PositionsLayout,
    ) -> Result<PostingList> {
        let posting_list = PostingList::from_batch_with_tail_codec_and_positions_layout(
            batch,
            max_score,
            length,
            posting_tail_codec,
            positions_layout,
        )?;
        Ok(posting_list)
    }

    pub(crate) fn posting_list_from_batch(
        &self,
        batch: &RecordBatch,
        token_id: u32,
    ) -> Result<PostingList> {
        Self::posting_list_from_batch_parts(
            batch,
            self.max_scores
                .as_ref()
                .map(|max_scores| max_scores[token_id as usize]),
            self.lengths
                .as_ref()
                .map(|lengths| lengths[token_id as usize]),
            self.posting_tail_codec,
            self.positions_layout,
        )
    }

    fn build_prewarm_posting_lists(
        batch: RecordBatch,
        offsets: Option<Vec<usize>>,
        max_scores: Option<Vec<f32>>,
        lengths: Option<Vec<u32>>,
        posting_tail_codec: PostingTailCodec,
        positions_layout: PositionsLayout,
    ) -> Result<Vec<(u32, PostingList)>> {
        let token_count = if let Some(offsets) = offsets.as_ref() {
            offsets.len()
        } else if let Some(lengths) = lengths.as_ref() {
            lengths.len()
        } else {
            batch.num_rows()
        };

        let mut posting_lists = Vec::with_capacity(token_count);
        for token_id in 0..token_count {
            let batch = if let Some(offsets) = offsets.as_ref() {
                let start = offsets[token_id];
                let end = if token_id + 1 < offsets.len() {
                    offsets[token_id + 1]
                } else {
                    batch.num_rows()
                };
                batch.slice(start, end - start)
            } else {
                batch.slice(token_id, 1)
            };
            let batch = batch.shrink_to_fit()?;
            let posting_list = Self::posting_list_from_batch_parts(
                &batch,
                max_scores.as_ref().map(|scores| scores[token_id]),
                lengths.as_ref().map(|lengths| lengths[token_id]),
                posting_tail_codec,
                positions_layout,
            )?;
            posting_lists.push((token_id as u32, posting_list));
        }

        Ok(posting_lists)
    }

    async fn prewarm_posting_lists(&self, with_position: bool) -> Result<()> {
        if with_position && !self.has_positions() {
            return Err(Error::invalid_input(
                "cannot prewarm positions for an inverted index that was built without positions; recreate the index with with_position=true".to_owned(),
            ));
        }

        let read_batch_start = Instant::now();
        let batch = self.read_batch(with_position).await?;
        let read_batch_elapsed = read_batch_start.elapsed();

        let legacy_layout = self.offsets.is_some();
        let offsets = self.offsets.clone();
        let max_scores = self.max_scores.clone();
        let lengths = self.lengths.clone();
        let posting_tail_codec = self.posting_tail_codec;
        let positions_layout = self.positions_layout;
        let populate_start = Instant::now();
        let posting_lists = spawn_blocking(move || {
            Self::build_prewarm_posting_lists(
                batch,
                offsets,
                max_scores,
                lengths,
                posting_tail_codec,
                positions_layout,
            )
        })
        .await
        .map_err(|err| {
            Error::internal(format!(
                "Failed to build prewarm posting lists in blocking task: {err}"
            ))
        })??;
        for (token_id, mut posting_list) in posting_lists {
            if with_position && let Some(positions) = posting_list.take_positions() {
                self.index_cache
                    .insert_with_key(&PositionKey { token_id }, Arc::new(Positions(positions)))
                    .await;
            }
            self.index_cache
                .insert_with_key(&PostingListKey { token_id }, Arc::new(posting_list))
                .await;
        }
        let populate_elapsed = populate_start.elapsed();

        info!(
            legacy_layout,
            with_position,
            token_count = self.len(),
            read_batch_ms = read_batch_elapsed.as_secs_f64() * 1000.0,
            post_read_loop_ms = populate_elapsed.as_secs_f64() * 1000.0,
            "posting list prewarm timing"
        );

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

    async fn read_positions(&self, token_id: u32) -> Result<CompressedPositionStorage> {
        let positions = self.index_cache.get_or_insert_with_key(PositionKey { token_id }, || async move {
            let positions = match self.positions_layout {
                PositionsLayout::None => {
                    return Err(Error::invalid_input(
                        "position is not found but required for phrase queries, try recreating the index with position".to_owned(),
                    ));
                }
                PositionsLayout::LegacyPerDoc => {
                    let batch = self
                        .reader
                        .read_range(self.posting_list_range(token_id), Some(&[POSITION_COL]))
                        .await
                        .map_err(|e| match e {
                            Error::Schema { .. } => Error::invalid_input("position is not found but required for phrase queries, try recreating the index with position".to_owned()),
                            e => e,
                        })?;
                    CompressedPositionStorage::LegacyPerDoc(
                        batch[POSITION_COL].as_list::<i32>().value(0).as_list::<i32>().clone(),
                    )
                }
                PositionsLayout::SharedStream(codec) => {
                    let batch = self
                        .reader
                        .read_range(
                            self.posting_list_range(token_id),
                            Some(&[COMPRESSED_POSITION_COL, POSITION_BLOCK_OFFSET_COL]),
                        )
                        .await
                        .map_err(|e| match e {
                            Error::Schema { .. } => Error::invalid_input("position is not found but required for phrase queries, try recreating the index with position".to_owned()),
                            e => e,
                        })?;
                    let bytes = batch[COMPRESSED_POSITION_COL]
                        .as_binary::<i64>()
                        .value(0)
                        .to_vec();
                    let block_offsets = batch[POSITION_BLOCK_OFFSET_COL]
                        .as_list::<i32>()
                        .value(0)
                        .as_primitive::<UInt32Type>()
                        .values()
                        .to_vec();
                    CompressedPositionStorage::SharedStream(SharedPositionStream::new(
                        codec,
                        block_offsets,
                        bytes,
                    ))
                }
            };
            Result::Ok(Positions(positions))
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
            match self.positions_layout {
                PositionsLayout::None => {}
                PositionsLayout::LegacyPerDoc => base_columns.push(POSITION_COL),
                PositionsLayout::SharedStream(_) => {
                    base_columns.push(COMPRESSED_POSITION_COL);
                    base_columns.push(POSITION_BLOCK_OFFSET_COL);
                }
            }
        }
        base_columns
    }
}

/// New type just to allow Positions implement DeepSizeOf so it can be put
/// in the cache.
#[derive(Clone)]
pub struct Positions(CompressedPositionStorage);

impl DeepSizeOf for Positions {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        match &self.0 {
            CompressedPositionStorage::LegacyPerDoc(positions) => {
                positions.get_buffer_memory_size()
            }
            CompressedPositionStorage::SharedStream(stream) => stream.size(),
        }
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

    fn type_name() -> &'static str {
        "PostingList"
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

    fn type_name() -> &'static str {
        "Position"
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompressedPositionStorage {
    LegacyPerDoc(ListArray),
    SharedStream(SharedPositionStream),
}

impl DeepSizeOf for CompressedPositionStorage {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        match self {
            Self::LegacyPerDoc(positions) => positions.get_buffer_memory_size(),
            Self::SharedStream(stream) => stream.size(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SharedPositionStream {
    codec: PositionStreamCodec,
    block_offsets: Vec<u32>,
    bytes: Vec<u8>,
}

impl SharedPositionStream {
    pub fn new(codec: PositionStreamCodec, block_offsets: Vec<u32>, bytes: Vec<u8>) -> Self {
        Self {
            codec,
            block_offsets,
            bytes,
        }
    }

    pub fn codec(&self) -> PositionStreamCodec {
        self.codec
    }

    pub fn block_count(&self) -> usize {
        self.block_offsets.len()
    }

    pub fn block_range(&self, index: usize) -> Range<usize> {
        let start = self.block_offsets[index] as usize;
        let end = self
            .block_offsets
            .get(index + 1)
            .map(|offset| *offset as usize)
            .unwrap_or(self.bytes.len());
        start..end
    }

    pub fn block(&self, index: usize) -> &[u8] {
        let range = self.block_range(index);
        &self.bytes[range]
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn block_offsets(&self) -> &[u32] {
        &self.block_offsets
    }

    pub fn size(&self) -> usize {
        self.block_offsets.capacity() * std::mem::size_of::<u32>() + self.bytes.capacity()
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
        let posting_tail_codec = parse_posting_tail_codec(batch.schema_ref().metadata())?;
        Self::from_batch_with_tail_codec(batch, max_score, length, posting_tail_codec)
    }

    pub fn from_batch_with_tail_codec(
        batch: &RecordBatch,
        max_score: Option<f32>,
        length: Option<u32>,
        posting_tail_codec: PostingTailCodec,
    ) -> Result<Self> {
        let positions_layout = if batch.column_by_name(COMPRESSED_POSITION_COL).is_some() {
            PositionsLayout::SharedStream(parse_shared_position_codec(
                batch.schema_ref().metadata(),
            )?)
        } else if batch.column_by_name(POSITION_COL).is_some() {
            PositionsLayout::LegacyPerDoc
        } else {
            PositionsLayout::None
        };
        Self::from_batch_with_tail_codec_and_positions_layout(
            batch,
            max_score,
            length,
            posting_tail_codec,
            positions_layout,
        )
    }

    fn from_batch_with_tail_codec_and_positions_layout(
        batch: &RecordBatch,
        max_score: Option<f32>,
        length: Option<u32>,
        posting_tail_codec: PostingTailCodec,
        positions_layout: PositionsLayout,
    ) -> Result<Self> {
        match batch.column_by_name(POSTING_COL) {
            Some(_) => {
                debug_assert!(max_score.is_some() && length.is_some());
                let shared_position_codec = match positions_layout {
                    PositionsLayout::SharedStream(codec) => Some(codec),
                    _ => None,
                };
                let posting = CompressedPostingList::from_batch(
                    batch,
                    max_score.unwrap(),
                    length.unwrap(),
                    posting_tail_codec,
                    shared_position_codec,
                );
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

    pub fn set_positions(&mut self, positions: CompressedPositionStorage) {
        match self {
            Self::Plain(posting) => match positions {
                CompressedPositionStorage::LegacyPerDoc(positions) => {
                    posting.positions = Some(positions)
                }
                CompressedPositionStorage::SharedStream(_) => {
                    unreachable!("shared position stream is not supported for plain postings")
                }
            },
            Self::Compressed(posting) => {
                posting.positions = Some(positions);
            }
        }
    }

    pub fn take_positions(&mut self) -> Option<CompressedPositionStorage> {
        match self {
            Self::Plain(posting) => posting
                .positions
                .take()
                .map(CompressedPositionStorage::LegacyPerDoc),
            Self::Compressed(posting) => posting.positions.take(),
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
        let posting_tail_codec = match &self {
            Self::Plain(_) => PostingTailCodec::Fixed32,
            Self::Compressed(posting) => posting.posting_tail_codec,
        };
        let mut builder = PostingListBuilder::new_with_posting_tail_codec(
            self.has_position(),
            posting_tail_codec,
        );
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
                            PositionRecorder::Position(positions.collect::<Vec<_>>().into())
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
                            PositionRecorder::Position(positions.collect::<Vec<_>>().into())
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
                .map(Array::get_buffer_memory_size)
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
    pub posting_tail_codec: PostingTailCodec,
    pub positions: Option<CompressedPositionStorage>,
}

impl DeepSizeOf for CompressedPostingList {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.blocks.get_buffer_memory_size()
            + self
                .positions
                .as_ref()
                .map(|positions| match positions {
                    CompressedPositionStorage::LegacyPerDoc(positions) => {
                        positions.get_buffer_memory_size()
                    }
                    CompressedPositionStorage::SharedStream(stream) => stream.size(),
                })
                .unwrap_or(0)
    }
}

impl CompressedPostingList {
    pub fn new(
        blocks: LargeBinaryArray,
        max_score: f32,
        length: u32,
        posting_tail_codec: PostingTailCodec,
        positions: Option<CompressedPositionStorage>,
    ) -> Self {
        Self {
            max_score,
            length,
            blocks,
            posting_tail_codec,
            positions,
        }
    }

    pub fn from_batch(
        batch: &RecordBatch,
        max_score: f32,
        length: u32,
        posting_tail_codec: PostingTailCodec,
        shared_position_codec: Option<PositionStreamCodec>,
    ) -> Self {
        debug_assert_eq!(batch.num_rows(), 1);
        let blocks = batch[POSTING_COL]
            .as_list::<i32>()
            .value(0)
            .as_binary::<i64>()
            .clone();
        let positions = if let Some(col) = batch.column_by_name(COMPRESSED_POSITION_COL) {
            let bytes = col.as_binary::<i64>().value(0).to_vec();
            let block_offsets = batch[POSITION_BLOCK_OFFSET_COL]
                .as_list::<i32>()
                .value(0)
                .as_primitive::<UInt32Type>()
                .values()
                .to_vec();
            let codec = shared_position_codec.unwrap_or_else(|| {
                parse_shared_position_codec(batch.schema_ref().metadata())
                    .expect("shared position stream codec metadata should be valid")
            });
            Some(CompressedPositionStorage::SharedStream(
                SharedPositionStream::new(codec, block_offsets, bytes),
            ))
        } else {
            batch.column_by_name(POSITION_COL).map(|col| {
                CompressedPositionStorage::LegacyPerDoc(
                    col.as_list::<i32>().value(0).as_list::<i32>().clone(),
                )
            })
        };

        Self {
            max_score,
            length,
            blocks,
            posting_tail_codec,
            positions,
        }
    }

    pub fn iter(&self) -> CompressedPostingListIterator {
        CompressedPostingListIterator::new(
            self.length as usize,
            self.blocks.clone(),
            self.posting_tail_codec,
            self.positions.clone(),
        )
    }

    pub fn block_max_score(&self, block_idx: usize) -> f32 {
        let block = self.blocks.value(block_idx);
        block[0..4].try_into().map(f32::from_le_bytes).unwrap()
    }

    pub fn block_least_doc_id(&self, block_idx: usize) -> u32 {
        let block = self.blocks.value(block_idx);
        let remainder = self.length as usize % BLOCK_SIZE;
        let is_remainder_block = remainder > 0 && block_idx + 1 == self.blocks.len();
        if is_remainder_block {
            super::encoding::read_posting_tail_first_doc(block, self.posting_tail_codec)
        } else {
            block[4..8].try_into().map(u32::from_le_bytes).unwrap()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct EncodedBlocks {
    offsets: Vec<u32>,
    bytes: Vec<u8>,
}

impl EncodedBlocks {
    fn len(&self) -> usize {
        self.offsets.len()
    }

    fn size(&self) -> usize {
        self.offsets.capacity() * std::mem::size_of::<u32>() + self.bytes.capacity()
    }

    fn push_full_block(&mut self, doc_ids: &[u32], frequencies: &[u32]) -> Result<usize> {
        let start = self.bytes.len();
        self.offsets.push(start as u32);
        super::encoding::encode_full_posting_block_into(doc_ids, frequencies, &mut self.bytes)?;
        Ok(self.bytes.len() - start)
    }

    fn block(&self, index: usize) -> &[u8] {
        let (start, end) = self.block_range(index);
        &self.bytes[start..end]
    }

    fn block_range(&self, index: usize) -> (usize, usize) {
        let start = self.offsets[index] as usize;
        let end = self
            .offsets
            .get(index + 1)
            .map(|offset| *offset as usize)
            .unwrap_or(self.bytes.len());
        (start, end)
    }

    fn set_block_score(&mut self, index: usize, score: f32) {
        let (start, _) = self.block_range(index);
        self.bytes[start..start + 4].copy_from_slice(&score.to_le_bytes());
    }

    fn append_remainder_block_with_codec(
        &mut self,
        doc_ids: &[u32],
        frequencies: &[u32],
        codec: PostingTailCodec,
    ) -> Result<()> {
        self.offsets.push(self.bytes.len() as u32);
        super::encoding::encode_remainder_posting_block_into(
            doc_ids,
            frequencies,
            codec,
            &mut self.bytes,
        )
    }

    fn into_array(mut self) -> LargeBinaryArray {
        let mut offsets = Vec::with_capacity(self.offsets.len() + 1);
        offsets.extend(self.offsets.into_iter().map(i64::from));
        offsets.push(self.bytes.len() as i64);
        LargeBinaryArray::new(
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            Buffer::from_vec(std::mem::take(&mut self.bytes)),
            None,
        )
    }

    fn iter(&self) -> impl Iterator<Item = &[u8]> {
        (0..self.len()).map(|index| self.block(index))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct EncodedPositionBlocks {
    offsets: Vec<u32>,
    bytes: Vec<u8>,
}

impl EncodedPositionBlocks {
    fn size(&self) -> usize {
        self.offsets.capacity() * std::mem::size_of::<u32>() + self.bytes.capacity()
    }

    fn block(&self, index: usize) -> &[u8] {
        let start = self.offsets[index] as usize;
        let end = self
            .offsets
            .get(index + 1)
            .map(|offset| *offset as usize)
            .unwrap_or(self.bytes.len());
        &self.bytes[start..end]
    }

    fn push_encoded_block(&mut self, block: &[u8]) -> usize {
        let start = self.bytes.len();
        self.offsets.push(start as u32);
        self.bytes.extend_from_slice(block);
        self.bytes.len() - start
    }

    fn into_stream(self) -> SharedPositionStream {
        SharedPositionStream::new(PositionStreamCodec::PackedDelta, self.offsets, self.bytes)
    }
}

#[derive(Debug)]
pub struct PostingListBuilder {
    with_positions: bool,
    posting_tail_codec: PostingTailCodec,
    encoded_blocks: Option<Box<EncodedBlocks>>,
    encoded_position_blocks: Option<Box<EncodedPositionBlocks>>,
    tail_entries: Vec<RawDocInfo>,
    tail_positions: PositionBlockBuilder,
    open_doc_id: Option<u32>,
    open_doc_frequency: u32,
    open_doc_last_position: Option<u32>,
    memory_size_bytes: u32,
    len: u32,
}

pub(super) struct PostingListBatchBuilder {
    schema: SchemaRef,
    postings: ListBuilder<LargeBinaryBuilder>,
    max_scores: Float32Builder,
    lengths: UInt32Builder,
    positions: BatchPositionsBuilder,
    len: usize,
}

enum BatchPositionsBuilder {
    None,
    Legacy(ListBuilder<ListBuilder<LargeBinaryBuilder>>),
    Shared {
        bytes: LargeBinaryBuilder,
        block_offsets: ListBuilder<UInt32Builder>,
    },
}

struct PostingListParts<'a> {
    with_positions: bool,
    posting_tail_codec: PostingTailCodec,
    length: usize,
    encoded_blocks: EncodedBlocks,
    encoded_position_blocks: EncodedPositionBlocks,
    tail_entries: &'a [RawDocInfo],
    tail_position_block: Option<Vec<u8>>,
}

impl PostingListBatchBuilder {
    pub fn new(
        schema: SchemaRef,
        with_positions: bool,
        format_version: InvertedListFormatVersion,
        capacity: usize,
    ) -> Self {
        let positions = if !with_positions {
            BatchPositionsBuilder::None
        } else if format_version.uses_shared_position_stream() {
            BatchPositionsBuilder::Shared {
                bytes: LargeBinaryBuilder::with_capacity(capacity, 0),
                block_offsets: ListBuilder::with_capacity(UInt32Builder::new(), capacity),
            }
        } else {
            BatchPositionsBuilder::Legacy(ListBuilder::with_capacity(
                ListBuilder::new(LargeBinaryBuilder::new()),
                capacity,
            ))
        };
        Self {
            schema,
            postings: ListBuilder::with_capacity(LargeBinaryBuilder::new(), capacity),
            max_scores: Float32Builder::with_capacity(capacity),
            lengths: UInt32Builder::with_capacity(capacity),
            positions,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn append(
        &mut self,
        compressed: LargeBinaryArray,
        max_score: f32,
        length: u32,
        positions: Option<&CompressedPositionStorage>,
    ) -> Result<()> {
        {
            let values = self.postings.values();
            for index in 0..compressed.len() {
                values.append_value(compressed.value(index));
            }
        }
        self.postings.append(true);
        self.max_scores.append_value(max_score);
        self.lengths.append_value(length);

        match &mut self.positions {
            BatchPositionsBuilder::None => {}
            BatchPositionsBuilder::Shared {
                bytes,
                block_offsets,
            } => {
                let positions = positions.ok_or_else(|| {
                    Error::index(format!(
                        "positions builder missing position data for posting length {}",
                        length
                    ))
                })?;
                let CompressedPositionStorage::SharedStream(positions) = positions else {
                    return Err(Error::index(
                        "shared positions builder received legacy positions".to_owned(),
                    ));
                };
                bytes.append_value(positions.bytes());
                let offsets_builder = block_offsets.values();
                for &offset in positions.block_offsets() {
                    offsets_builder.append_value(offset);
                }
                block_offsets.append(true);
            }
            BatchPositionsBuilder::Legacy(position_lists) => {
                let positions = positions.ok_or_else(|| {
                    Error::index(format!(
                        "positions builder missing position data for posting length {}",
                        length
                    ))
                })?;
                let CompressedPositionStorage::LegacyPerDoc(positions) = positions else {
                    return Err(Error::index(
                        "legacy positions builder received shared position stream".to_owned(),
                    ));
                };
                let docs_builder = position_lists.values();
                for doc_idx in 0..positions.len() {
                    let doc_positions = positions.value(doc_idx);
                    let compressed_positions = doc_positions.as_binary::<i64>();
                    for block_idx in 0..compressed_positions.len() {
                        docs_builder
                            .values()
                            .append_value(compressed_positions.value(block_idx));
                    }
                    docs_builder.append(true);
                }
                position_lists.append(true);
            }
        }

        self.len += 1;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<RecordBatch> {
        let mut columns = vec![
            Arc::new(self.postings.finish()) as ArrayRef,
            Arc::new(self.max_scores.finish()) as ArrayRef,
            Arc::new(self.lengths.finish()) as ArrayRef,
        ];
        match &mut self.positions {
            BatchPositionsBuilder::None => {}
            BatchPositionsBuilder::Legacy(position_lists) => {
                columns.push(Arc::new(position_lists.finish()) as ArrayRef);
            }
            BatchPositionsBuilder::Shared {
                bytes,
                block_offsets,
            } => {
                columns.push(Arc::new(bytes.finish()) as ArrayRef);
                columns.push(Arc::new(block_offsets.finish()) as ArrayRef);
            }
        }
        self.len = 0;
        RecordBatch::try_new(self.schema.clone(), columns).map_err(Error::from)
    }
}

impl PostingListBuilder {
    pub fn size(&self) -> u64 {
        self.memory_size_bytes as u64
    }

    pub fn has_positions(&self) -> bool {
        self.with_positions
    }

    pub fn new(with_position: bool) -> Self {
        Self::new_with_posting_tail_codec(
            with_position,
            current_fts_format_version().posting_tail_codec(),
        )
    }

    pub fn new_with_posting_tail_codec(
        with_position: bool,
        posting_tail_codec: PostingTailCodec,
    ) -> Self {
        Self {
            with_positions: with_position,
            posting_tail_codec,
            encoded_blocks: None,
            encoded_position_blocks: None,
            tail_entries: Vec::new(),
            tail_positions: PositionBlockBuilder::default(),
            open_doc_id: None,
            open_doc_frequency: 0,
            open_doc_last_position: None,
            len: 0,
            memory_size_bytes: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> std::vec::IntoIter<(u32, u32, Option<Vec<u32>>)> {
        self.collect_entries().into_iter()
    }

    pub fn for_each_entry<E>(
        &self,
        mut visit: impl FnMut(u32, u32, Option<Vec<u32>>) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        let mut doc_ids = Vec::with_capacity(BLOCK_SIZE);
        let mut frequencies = Vec::with_capacity(BLOCK_SIZE);
        let mut decoded_positions = Vec::new();
        let mut position_block_index = 0usize;

        if let Some(encoded_blocks) = self.encoded_blocks.as_deref() {
            for block in encoded_blocks.iter() {
                doc_ids.clear();
                frequencies.clear();
                super::encoding::decode_full_posting_block(block, &mut doc_ids, &mut frequencies);
                decoded_positions.clear();
                if self.with_positions {
                    let position_blocks = self
                        .encoded_position_blocks
                        .as_deref()
                        .expect("positions must exist for posting list");
                    super::encoding::decode_position_stream_block(
                        position_blocks.block(position_block_index),
                        &frequencies,
                        PositionStreamCodec::PackedDelta,
                        &mut decoded_positions,
                    )
                    .expect("position stream decoding should succeed");
                    position_block_index += 1;
                }
                let mut offset = 0usize;
                for (doc_id, frequency) in doc_ids.iter().copied().zip(frequencies.iter().copied())
                {
                    let positions = self.with_positions.then(|| {
                        let end = offset + frequency as usize;
                        let doc_positions = decoded_positions[offset..end].to_vec();
                        offset = end;
                        doc_positions
                    });
                    visit(doc_id, frequency, positions)?;
                }
            }
        }

        let mut decoded_tail_positions = Vec::new();
        if self.with_positions && !self.tail_entries.is_empty() {
            let tail_frequencies = self
                .tail_entries
                .iter()
                .map(|entry| entry.frequency)
                .collect::<Vec<_>>();
            self.tail_positions
                .decode_into(tail_frequencies.as_slice(), &mut decoded_tail_positions)
                .expect("tail position stream decoding should succeed");
        }
        let mut tail_offset = 0usize;
        for entry in &self.tail_entries {
            let positions = self.with_positions.then(|| {
                let end = tail_offset + entry.frequency as usize;
                let doc_positions = decoded_tail_positions[tail_offset..end].to_vec();
                tail_offset = end;
                doc_positions
            });
            visit(entry.doc_id, entry.frequency, positions)?;
        }

        Ok(())
    }

    pub fn add(&mut self, doc_id: u32, term_positions: PositionRecorder) {
        debug_assert!(
            self.open_doc_id.is_none(),
            "cannot add closed doc while a positions doc is still open"
        );
        let tail_entries_capacity_before = self.tail_entries.capacity();
        self.tail_entries
            .push(RawDocInfo::new(doc_id, term_positions.len()));
        let tail_entries_capacity_after = self.tail_entries.capacity();
        if tail_entries_capacity_after > tail_entries_capacity_before {
            self.add_memory_bytes(
                (tail_entries_capacity_after - tail_entries_capacity_before)
                    * std::mem::size_of::<RawDocInfo>(),
            );
        }
        if let PositionRecorder::Position(positions_in_doc) = term_positions {
            debug_assert!(self.with_positions);
            let old_size = self.tail_positions.size();
            self.tail_positions
                .append_doc_positions(positions_in_doc.as_slice())
                .expect("position stream encoding should succeed");
            self.adjust_tail_positions_size(old_size);
        }
        self.len += 1;

        if self.tail_entries.len() == BLOCK_SIZE {
            self.flush_tail_block()
                .expect("posting list block compression should succeed");
        }
    }

    pub fn add_occurrence(&mut self, doc_id: u32, position: u32) -> Result<bool> {
        if !self.with_positions {
            return Err(Error::index(
                "cannot append streamed positions to a posting list without positions".to_owned(),
            ));
        }

        match self.open_doc_id {
            Some(open_doc_id) if open_doc_id == doc_id => {
                let old_size = self.tail_positions.size();
                self.tail_positions
                    .append_position(position, self.open_doc_last_position)?;
                self.adjust_tail_positions_size(old_size);
                self.open_doc_frequency += 1;
                self.open_doc_last_position = Some(position);
                Ok(false)
            }
            Some(open_doc_id) => Err(Error::index(format!(
                "posting list received doc {} before finishing open doc {}",
                doc_id, open_doc_id
            ))),
            None => {
                let old_size = self.tail_positions.size();
                self.tail_positions.append_position(position, None)?;
                self.adjust_tail_positions_size(old_size);
                self.open_doc_id = Some(doc_id);
                self.open_doc_frequency = 1;
                self.open_doc_last_position = Some(position);
                self.len += 1;
                Ok(true)
            }
        }
    }

    pub fn finish_open_doc(&mut self, doc_id: u32) -> Result<()> {
        if !self.with_positions {
            return Ok(());
        }
        match self.open_doc_id {
            Some(open_doc_id) if open_doc_id == doc_id => {
                let tail_entries_capacity_before = self.tail_entries.capacity();
                self.tail_entries
                    .push(RawDocInfo::new(doc_id, self.open_doc_frequency));
                let tail_entries_capacity_after = self.tail_entries.capacity();
                if tail_entries_capacity_after > tail_entries_capacity_before {
                    self.add_memory_bytes(
                        (tail_entries_capacity_after - tail_entries_capacity_before)
                            * std::mem::size_of::<RawDocInfo>(),
                    );
                }
                self.open_doc_id = None;
                self.open_doc_frequency = 0;
                self.open_doc_last_position = None;
                if self.tail_entries.len() == BLOCK_SIZE {
                    self.flush_tail_block()?;
                }
                Ok(())
            }
            Some(open_doc_id) => Err(Error::index(format!(
                "attempted to finish doc {} while doc {} is still open",
                doc_id, open_doc_id
            ))),
            None => Ok(()),
        }
    }

    fn collect_entries(&self) -> Vec<(u32, u32, Option<Vec<u32>>)> {
        let mut entries = Vec::with_capacity(self.len());
        self.for_each_entry(|doc_id, frequency, positions| {
            entries.push((doc_id, frequency, positions));
            Ok::<(), ()>(())
        })
        .expect("collecting posting list entries should not fail");
        entries
    }

    fn encoded_blocks_mut(&mut self) -> &mut EncodedBlocks {
        if self.encoded_blocks.is_none() {
            self.encoded_blocks = Some(Box::default());
            self.add_memory_bytes(std::mem::size_of::<EncodedBlocks>());
        }
        self.encoded_blocks
            .as_deref_mut()
            .expect("encoded blocks must exist")
    }

    fn encoded_position_blocks_mut(&mut self) -> &mut EncodedPositionBlocks {
        if self.encoded_position_blocks.is_none() {
            self.encoded_position_blocks = Some(Box::default());
            self.add_memory_bytes(std::mem::size_of::<EncodedPositionBlocks>());
        }
        self.encoded_position_blocks
            .as_deref_mut()
            .expect("encoded position blocks must exist")
    }

    fn flush_tail_block(&mut self) -> Result<()> {
        if self.tail_entries.is_empty() {
            return Ok(());
        }
        debug_assert!(
            self.open_doc_id.is_none(),
            "cannot flush a posting block while a document is still open"
        );
        debug_assert_eq!(self.tail_entries.len(), BLOCK_SIZE);
        let mut doc_ids = [0u32; BLOCK_SIZE];
        let mut frequencies = [0u32; BLOCK_SIZE];
        for (index, entry) in self.tail_entries.iter().enumerate() {
            doc_ids[index] = entry.doc_id;
            frequencies[index] = entry.frequency;
        }
        let encoded_blocks_size_before = self
            .encoded_blocks
            .as_ref()
            .map(|encoded_blocks| encoded_blocks.size())
            .unwrap_or(0usize);
        self.encoded_blocks_mut()
            .push_full_block(&doc_ids, &frequencies)?;
        let encoded_blocks_size_after = self
            .encoded_blocks
            .as_ref()
            .map(|encoded_blocks| encoded_blocks.size())
            .unwrap_or(0usize);
        if encoded_blocks_size_after > encoded_blocks_size_before {
            self.add_memory_bytes(encoded_blocks_size_after - encoded_blocks_size_before);
        }
        if self.with_positions {
            let encoded_positions_size_before = self
                .encoded_position_blocks
                .as_ref()
                .map(|encoded| encoded.size())
                .unwrap_or(0usize);
            let released_tail_positions_bytes = self.tail_positions.size();
            let tail_position_block = std::mem::take(&mut self.tail_positions).finish();
            self.encoded_position_blocks_mut()
                .push_encoded_block(tail_position_block.as_slice());
            let encoded_positions_size_after = self
                .encoded_position_blocks
                .as_ref()
                .map(|encoded| encoded.size())
                .unwrap_or(0usize);
            if released_tail_positions_bytes > 0 {
                self.subtract_memory_bytes(released_tail_positions_bytes);
            }
            if encoded_positions_size_after > encoded_positions_size_before {
                self.add_memory_bytes(encoded_positions_size_after - encoded_positions_size_before);
            }
        }
        self.tail_entries.clear();
        Ok(())
    }

    fn adjust_tail_positions_size(&mut self, old_size: usize) {
        let new_size = self.tail_positions.size();
        if new_size > old_size {
            self.add_memory_bytes(new_size - old_size);
        } else if old_size > new_size {
            self.subtract_memory_bytes(old_size - new_size);
        }
    }

    fn add_memory_bytes(&mut self, bytes: usize) {
        self.memory_size_bytes = self
            .memory_size_bytes
            .checked_add(
                u32::try_from(bytes).expect("posting list memory size delta overflowed u32"),
            )
            .expect("posting list memory size overflowed u32");
    }

    fn subtract_memory_bytes(&mut self, bytes: usize) {
        self.memory_size_bytes = self
            .memory_size_bytes
            .checked_sub(
                u32::try_from(bytes).expect("posting list memory size delta overflowed u32"),
            )
            .expect("posting list memory size underflowed u32");
    }

    fn build_position_columns(
        positions: Option<CompressedPositionStorage>,
    ) -> Result<Vec<ArrayRef>> {
        let Some(positions) = positions else {
            return Ok(Vec::new());
        };
        match positions {
            CompressedPositionStorage::LegacyPerDoc(positions) => {
                Ok(vec![Arc::new(ListArray::try_new(
                    Arc::new(Field::new("item", positions.data_type().clone(), true)),
                    OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, positions.len() as i32])),
                    Arc::new(positions) as ArrayRef,
                    None,
                )?) as ArrayRef])
            }
            CompressedPositionStorage::SharedStream(positions) => {
                let mut columns = Vec::with_capacity(2);
                columns.push(
                    Arc::new(LargeBinaryArray::from(vec![Some(positions.bytes())])) as ArrayRef,
                );

                let mut offsets_builder = ListBuilder::new(UInt32Builder::new());
                for &offset in positions.block_offsets() {
                    offsets_builder.values().append_value(offset);
                }
                offsets_builder.append(true);
                columns.push(Arc::new(offsets_builder.finish()) as ArrayRef);
                Ok(columns)
            }
        }
    }

    fn build_batch(
        self,
        compressed: LargeBinaryArray,
        max_score: f32,
        schema: SchemaRef,
        positions: Option<CompressedPositionStorage>,
    ) -> Result<RecordBatch> {
        let length = self.len();
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
                length as u32,
            ))) as ArrayRef,
        ];
        columns.extend(Self::build_position_columns(positions)?);

        let batch = RecordBatch::try_new(schema, columns)?;
        Ok(batch)
    }

    fn build_legacy_positions(&self) -> Result<ListArray> {
        let mut positions_builder = ListBuilder::new(LargeBinaryBuilder::new());
        self.for_each_entry(|_doc_id, frequency, positions| {
            let positions = positions.ok_or_else(|| {
                Error::index(format!(
                    "legacy position writer missing positions for frequency {}",
                    frequency
                ))
            })?;
            let compressed = super::encoding::compress_positions(positions.as_slice())?;
            for block_idx in 0..compressed.len() {
                positions_builder
                    .values()
                    .append_value(compressed.value(block_idx));
            }
            positions_builder.append(true);
            Ok::<(), Error>(())
        })?;
        Ok(positions_builder.finish())
    }

    pub(super) fn append_to_batch_with_docs(
        self,
        docs: &DocSet,
        batch_builder: &mut PostingListBatchBuilder,
        format_version: InvertedListFormatVersion,
    ) -> Result<()> {
        let legacy_positions =
            if self.with_positions && !format_version.uses_shared_position_stream() {
                Some(self.build_legacy_positions()?)
            } else {
                None
            };
        let Self {
            with_positions,
            posting_tail_codec,
            encoded_blocks,
            encoded_position_blocks,
            tail_entries,
            tail_positions,
            open_doc_id,
            open_doc_frequency,
            open_doc_last_position,
            len,
            ..
        } = self;
        debug_assert!(open_doc_id.is_none());
        debug_assert_eq!(open_doc_frequency, 0);
        debug_assert!(open_doc_last_position.is_none());
        let parts = PostingListParts {
            with_positions,
            posting_tail_codec,
            length: len as usize,
            encoded_blocks: encoded_blocks
                .map(|encoded_blocks| *encoded_blocks)
                .unwrap_or_default(),
            encoded_position_blocks: encoded_position_blocks
                .map(|encoded_positions| *encoded_positions)
                .unwrap_or_default(),
            tail_entries: tail_entries.as_slice(),
            tail_position_block: with_positions.then(|| tail_positions.finish()),
        };
        let (compressed, shared_positions, max_score) =
            Self::build_compressed_with_scores_from_parts(parts, docs)?;
        let positions = match legacy_positions {
            Some(positions) => Some(CompressedPositionStorage::LegacyPerDoc(positions)),
            None => shared_positions.map(CompressedPositionStorage::SharedStream),
        };
        batch_builder.append(compressed, max_score, len, positions.as_ref())
    }

    fn extend_tail_components(
        tail_entries: &[RawDocInfo],
        doc_ids: &mut Vec<u32>,
        frequencies: &mut Vec<u32>,
    ) {
        doc_ids.clear();
        frequencies.clear();
        doc_ids.extend(tail_entries.iter().map(|entry| entry.doc_id));
        frequencies.extend(tail_entries.iter().map(|entry| entry.frequency));
    }

    fn build_compressed_with_scores_from_parts(
        parts: PostingListParts<'_>,
        docs: &DocSet,
    ) -> Result<(LargeBinaryArray, Option<SharedPositionStream>, f32)> {
        let PostingListParts {
            with_positions,
            posting_tail_codec,
            length,
            mut encoded_blocks,
            mut encoded_position_blocks,
            tail_entries,
            tail_position_block,
        } = parts;
        let avgdl = docs.average_length();
        let idf_scale = idf(length, docs.len()) * (K1 + 1.0);
        let mut max_score = f32::MIN;
        let mut doc_ids = Vec::with_capacity(BLOCK_SIZE);
        let mut frequencies = Vec::with_capacity(BLOCK_SIZE);

        for index in 0..encoded_blocks.len() {
            let block = encoded_blocks.block(index);
            doc_ids.clear();
            frequencies.clear();
            super::encoding::decode_full_posting_block(block, &mut doc_ids, &mut frequencies);
            let block_score = compute_block_score(
                docs,
                avgdl,
                idf_scale,
                doc_ids.iter().copied(),
                frequencies.iter().copied(),
            );
            max_score = max_score.max(block_score);
            encoded_blocks.set_block_score(index, block_score);
        }

        if !tail_entries.is_empty() {
            Self::extend_tail_components(tail_entries, &mut doc_ids, &mut frequencies);
            let block_score = compute_block_score(
                docs,
                avgdl,
                idf_scale,
                doc_ids.iter().copied(),
                frequencies.iter().copied(),
            );
            max_score = max_score.max(block_score);
            encoded_blocks.append_remainder_block_with_codec(
                doc_ids.as_slice(),
                frequencies.as_slice(),
                posting_tail_codec,
            )?;
            encoded_blocks.set_block_score(encoded_blocks.len() - 1, block_score);
            if with_positions {
                encoded_position_blocks.push_encoded_block(
                    tail_position_block
                        .as_deref()
                        .expect("tail position block must exist for postings with positions"),
                );
            }
        }

        Ok((
            encoded_blocks.into_array(),
            with_positions.then(|| encoded_position_blocks.into_stream()),
            max_score,
        ))
    }

    fn build_compressed_with_block_scores_from_parts(
        with_positions: bool,
        posting_tail_codec: PostingTailCodec,
        mut encoded_blocks: EncodedBlocks,
        mut encoded_position_blocks: EncodedPositionBlocks,
        tail_entries: &[RawDocInfo],
        tail_position_block: Option<Vec<u8>>,
        mut block_max_scores: impl Iterator<Item = f32>,
    ) -> Result<(LargeBinaryArray, Option<SharedPositionStream>, f32)> {
        let mut max_score = f32::MIN;
        let mut doc_ids = Vec::with_capacity(BLOCK_SIZE);
        let mut frequencies = Vec::with_capacity(BLOCK_SIZE);

        for index in 0..encoded_blocks.len() {
            let block_score = block_max_scores
                .next()
                .ok_or_else(|| Error::index("missing block max score".to_owned()))?;
            max_score = max_score.max(block_score);
            encoded_blocks.set_block_score(index, block_score);
        }

        if !tail_entries.is_empty() {
            let block_score = block_max_scores
                .next()
                .ok_or_else(|| Error::index("missing tail block max score".to_owned()))?;
            max_score = max_score.max(block_score);
            Self::extend_tail_components(tail_entries, &mut doc_ids, &mut frequencies);
            encoded_blocks.append_remainder_block_with_codec(
                doc_ids.as_slice(),
                frequencies.as_slice(),
                posting_tail_codec,
            )?;
            encoded_blocks.set_block_score(encoded_blocks.len() - 1, block_score);
            if with_positions {
                encoded_position_blocks.push_encoded_block(
                    tail_position_block
                        .as_deref()
                        .expect("tail position block must exist for postings with positions"),
                );
            }
        }

        Ok((
            encoded_blocks.into_array(),
            with_positions.then(|| encoded_position_blocks.into_stream()),
            max_score,
        ))
    }

    pub fn to_batch(self, block_max_scores: Vec<f32>) -> Result<RecordBatch> {
        let format_version = if self.posting_tail_codec == PostingTailCodec::Fixed32 {
            InvertedListFormatVersion::V1
        } else {
            InvertedListFormatVersion::V2
        };
        let schema = inverted_list_schema_for_version(self.has_positions(), format_version);
        let legacy_positions =
            if self.with_positions && !format_version.uses_shared_position_stream() {
                Some(self.build_legacy_positions()?)
            } else {
                None
            };
        let Self {
            with_positions,
            posting_tail_codec,
            encoded_blocks,
            encoded_position_blocks,
            tail_entries,
            tail_positions,
            open_doc_id,
            open_doc_frequency,
            open_doc_last_position,
            len,
            ..
        } = self;
        debug_assert!(open_doc_id.is_none());
        debug_assert_eq!(open_doc_frequency, 0);
        debug_assert!(open_doc_last_position.is_none());
        let (compressed, shared_positions, max_score) =
            Self::build_compressed_with_block_scores_from_parts(
                with_positions,
                posting_tail_codec,
                encoded_blocks
                    .map(|encoded_blocks| *encoded_blocks)
                    .unwrap_or_default(),
                encoded_position_blocks
                    .map(|encoded_positions| *encoded_positions)
                    .unwrap_or_default(),
                tail_entries.as_slice(),
                with_positions.then(|| tail_positions.finish()),
                block_max_scores.into_iter(),
            )?;
        let builder = Self {
            with_positions,
            posting_tail_codec,
            encoded_blocks: None,
            encoded_position_blocks: None,
            tail_entries: Vec::new(),
            tail_positions: PositionBlockBuilder::default(),
            open_doc_id: None,
            open_doc_frequency: 0,
            open_doc_last_position: None,
            memory_size_bytes: 0,
            len,
        };
        let positions = match legacy_positions {
            Some(positions) => Some(CompressedPositionStorage::LegacyPerDoc(positions)),
            None => shared_positions.map(CompressedPositionStorage::SharedStream),
        };
        builder.build_batch(compressed, max_score, schema, positions)
    }

    pub fn to_batch_with_docs(self, docs: &DocSet, schema: SchemaRef) -> Result<RecordBatch> {
        let format_version = if schema.column_with_name(POSITION_COL).is_some()
            && schema.column_with_name(COMPRESSED_POSITION_COL).is_none()
        {
            InvertedListFormatVersion::V1
        } else {
            InvertedListFormatVersion::V2
        };
        let legacy_positions =
            if self.with_positions && !format_version.uses_shared_position_stream() {
                Some(self.build_legacy_positions()?)
            } else {
                None
            };
        let Self {
            with_positions,
            posting_tail_codec,
            encoded_blocks,
            encoded_position_blocks,
            tail_entries,
            tail_positions,
            open_doc_id,
            open_doc_frequency,
            open_doc_last_position,
            len,
            ..
        } = self;
        debug_assert!(open_doc_id.is_none());
        debug_assert_eq!(open_doc_frequency, 0);
        debug_assert!(open_doc_last_position.is_none());
        let parts = PostingListParts {
            with_positions,
            posting_tail_codec,
            length: len as usize,
            encoded_blocks: encoded_blocks
                .map(|encoded_blocks| *encoded_blocks)
                .unwrap_or_default(),
            encoded_position_blocks: encoded_position_blocks
                .map(|encoded_positions| *encoded_positions)
                .unwrap_or_default(),
            tail_entries: tail_entries.as_slice(),
            tail_position_block: with_positions.then(|| tail_positions.finish()),
        };
        let (compressed, shared_positions, max_score) =
            Self::build_compressed_with_scores_from_parts(parts, docs)?;
        let builder = Self {
            with_positions,
            posting_tail_codec,
            encoded_blocks: None,
            encoded_position_blocks: None,
            tail_entries: Vec::new(),
            tail_positions: PositionBlockBuilder::default(),
            open_doc_id: None,
            open_doc_frequency: 0,
            open_doc_last_position: None,
            memory_size_bytes: 0,
            len,
        };
        let positions = match legacy_positions {
            Some(positions) => Some(CompressedPositionStorage::LegacyPerDoc(positions)),
            None => shared_positions.map(CompressedPositionStorage::SharedStream),
        };
        builder.build_batch(compressed, max_score, schema, positions)
    }

    pub fn remap(&mut self, removed: &[u32]) {
        let mut cursor = 0;
        let mut new_builder =
            Self::new_with_posting_tail_codec(self.has_positions(), self.posting_tail_codec);
        for (doc_id, freq, positions) in self.iter() {
            while cursor < removed.len() && removed[cursor] < doc_id {
                cursor += 1;
            }
            if cursor < removed.len() && removed[cursor] == doc_id {
                continue;
            }
            let positions = match positions {
                Some(positions) => PositionRecorder::Position(positions.into()),
                None => PositionRecorder::Count(freq),
            };
            new_builder.add(doc_id - cursor as u32, positions);
        }

        *self = new_builder;
    }
}

fn compute_block_score(
    docs: &DocSet,
    avgdl: f32,
    idf_scale: f32,
    doc_ids: impl Iterator<Item = u32>,
    frequencies: impl Iterator<Item = u32>,
) -> f32 {
    let mut block_max_score = f32::MIN;
    for (doc_id, freq) in doc_ids.zip(frequencies) {
        let doc_norm = K1 * (1.0 - B + B * docs.num_tokens(doc_id) as f32 / avgdl);
        let freq = freq as f32;
        let score = freq / (freq + doc_norm);
        block_max_score = block_max_score.max(score);
    }
    block_max_score * idf_scale
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
        let num_blocks = length.div_ceil(BLOCK_SIZE);
        let mut block_max_scores = Vec::with_capacity(num_blocks);
        let idf_scale = idf(length, self.len()) * (K1 + 1.0);
        let mut max_score = f32::MIN;
        for (i, (doc_id, freq)) in doc_ids.zip(freqs).enumerate() {
            let doc_norm = K1 * (1.0 - B + B * self.num_tokens(*doc_id) as f32 / avgdl);
            let freq = *freq as f32;
            let score = freq / (freq + doc_norm);
            if score > max_score {
                max_score = score;
            }
            if (i + 1) % BLOCK_SIZE == 0 {
                max_score *= idf_scale;
                block_max_scores.push(max_score);
                max_score = f32::MIN;
            }
        }
        if !length.is_multiple_of(BLOCK_SIZE) {
            max_score *= idf_scale;
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

    pub(crate) fn memory_size(&self) -> usize {
        self.row_ids.capacity() * std::mem::size_of::<u64>()
            + self.num_tokens.capacity() * std::mem::size_of::<u32>()
            + self.inv.capacity() * std::mem::size_of::<(u64, u32)>()
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
        ));
    }

    match batches[0][doc_col].data_type() {
        DataType::Utf8 => do_flat_full_text_search::<i32>(batches, doc_col, query, tokenizer),
        DataType::LargeUtf8 => do_flat_full_text_search::<i64>(batches, doc_col, query, tokenizer),
        data_type => Err(Error::invalid_input(format!(
            "unsupported data type {} for inverted index",
            data_type
        ))),
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
    let query_tokens = collect_query_tokens(query, &mut tokenizer);

    for batch in batches {
        let row_id_array = batch[ROW_ID].as_primitive::<UInt64Type>();
        let doc_array = batch[doc_col].as_string::<Offset>();
        for i in 0..row_id_array.len() {
            let doc = doc_array.value(i);
            if has_query_token(doc, &mut tokenizer, &query_tokens) {
                results.push(row_id_array.value(i));
                // What is this assertion for?  Why would doc contain query?  Don't we reach
                // here only if they share at least one token?  Why is it not debug_assert?
                assert!(doc.contains(query));
            }
        }
    }

    Ok(results)
}

const FLAT_ROW_ID_COL_IDX: usize = 0;
const FLAT_ALL_TOKENS_COL_IDX: usize = 1;
const FLAT_QUERY_TOKEN_COUNTS_COL_IDX: usize = 2;

/// If we accumulate this many bytes we warn the user they probably want to use an FTS index instead.
const BYTES_ACCUMULATED_WARNING_THRESHOLD: u64 = 1024 * 1024 * 1024; // 1GB

/// Consumes a stream of record batches and produces token counts
///
/// The resulting batch will have three columns:
/// - row_id: the row id of the document
/// - all_tokens: the total number of tokens in the document
/// - query_token_counts: a fixed size list of the count of each query token in the document
///
/// This is an unbounded accumulation, however, for most queries, the per-row
/// growth will be fairly small.  As a result we can process millions of tokens
/// with fairly modest memory usage.
///
/// However, it is unwise to do a flat search across billions of rows.  An FTS
/// index should be created instead.
async fn tokenize_and_count(
    input: impl Stream<Item = DataFusionResult<RecordBatch>> + Send,
    tokenizer: Box<dyn LanceTokenizer>,
    query_tokens: Arc<Tokens>,
    doc_col_idx: usize,
) -> DataFusionResult<RecordBatch> {
    let output_schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new("all_tokens", DataType::UInt64, false),
        Field::new(
            "query_token_counts",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt64, true)),
                query_tokens.len() as i32,
            ),
            false,
        ),
    ]));
    let output_schema_clone = output_schema.clone();
    let bytes_accumulated = Arc::new(AtomicU64::new(0));
    let bytes_warning_emitted = Arc::new(AtomicBool::new(false));

    let batches = input
        .map(move |batch| {
            let mut tokenizer = tokenizer.box_clone();
            let output_schema = output_schema.clone();
            let query_tokens = query_tokens.clone();
            let bytes_accumulated = bytes_accumulated.clone();
            let bytes_warning_emitted = bytes_warning_emitted.clone();
            spawn_cpu(move || {
                let batch = batch?;
                let mut all_token_counts = UInt64Builder::with_capacity(batch.num_rows());
                let mut query_token_counts = FixedSizeListBuilder::with_capacity(
                    UInt64Builder::with_capacity(batch.num_rows() * query_tokens.len()),
                    query_tokens.len() as i32,
                    batch.num_rows(),
                );
                let mut temp_query_token_counts = Vec::with_capacity(query_tokens.len());
                let doc_iter = iter_str_array(batch.column(doc_col_idx));
                for doc in doc_iter {
                    let Some(doc) = doc else {
                        all_token_counts.append_value(0);
                        query_token_counts
                            .values()
                            .append_value_n(0, query_tokens.len());
                        query_token_counts.append(true);
                        continue;
                    };

                    temp_query_token_counts.clear();
                    temp_query_token_counts.extend(std::iter::repeat_n(0, query_tokens.len()));

                    let mut stream = tokenizer.token_stream_for_doc(doc);
                    let mut all_tokens = 0;
                    while let Some(token) = stream.next() {
                        all_tokens += 1;
                        if let Some(token_index) = query_tokens.token_index(&token.text) {
                            temp_query_token_counts[token_index] += 1;
                        }
                    }
                    all_token_counts.append_value(all_tokens);
                    for count in temp_query_token_counts.iter().copied() {
                        query_token_counts.values().append_value(count);
                    }
                    query_token_counts.append(true);
                }
                let row_ids = batch[ROW_ID].clone();
                let all_token_counts = all_token_counts.finish();
                let query_token_counts = query_token_counts.finish();
                let result_batch = RecordBatch::try_new(

                    output_schema,
                    vec![
                        row_ids,
                        Arc::new(all_token_counts) as ArrayRef,
                        Arc::new(query_token_counts) as ArrayRef,
                    ],
                )?;
                let bytes_accumulated = bytes_accumulated.fetch_add(result_batch.get_array_memory_size() as u64, Ordering::Relaxed);
                if bytes_accumulated > BYTES_ACCUMULATED_WARNING_THRESHOLD && !bytes_warning_emitted.swap(true, Ordering::Relaxed) {
                    tracing::warn!("Flat full text search is accumulating a large number of bytes.  Consider using an FTS index instead.");
                }

                DataFusionResult::Ok(result_batch)
            })
        })
        .buffered(get_num_compute_intensive_cpus())
        .try_collect::<Vec<_>>()
        .await?;

    Ok(arrow::compute::concat_batches(
        &output_schema_clone,
        &batches,
    )?)
}

/// Initialize the BM25 scorer
///
/// In order to calculate BM25 scores we need to know token counts for the entire corpus.  We extract these from the
/// counted input of the flat search combined with any counts recorded for the indexed portion.
fn initialize_scorer(
    base_scorer: Option<&MemBM25Scorer>,
    query_tokens: &Tokens,
    counted_input: &RecordBatch,
) -> MemBM25Scorer {
    let mut total_tokens = 0;
    let mut num_docs = 0;
    let mut all_token_counts = vec![0; query_tokens.len()];

    if let Some(base_scorer) = base_scorer {
        total_tokens += base_scorer.total_tokens;
        num_docs += base_scorer.num_docs;
        for (token_index, token) in query_tokens.into_iter().enumerate() {
            all_token_counts[token_index] = base_scorer.num_docs_containing_token(token) as u64;
        }
    }

    num_docs += counted_input.num_rows();
    total_tokens += arrow::compute::sum(
        counted_input
            .column(FLAT_ALL_TOKENS_COL_IDX)
            .as_primitive::<UInt64Type>(),
    )
    .unwrap_or_default();

    let mut input_token_counters = counted_input
        .column(FLAT_QUERY_TOKEN_COUNTS_COL_IDX)
        .as_fixed_size_list()
        .values()
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied();

    for _ in 0..counted_input.num_rows() {
        for token_count in all_token_counts.iter_mut() {
            *token_count += input_token_counters.next().unwrap_or_default();
        }
    }

    let token_counts_map = all_token_counts
        .into_iter()
        .enumerate()
        .map(|(token_index, count)| {
            (
                query_tokens.get_token(token_index).to_string(),
                count as usize,
            )
        })
        .collect::<HashMap<String, usize>>();
    MemBM25Scorer::new(total_tokens, num_docs, token_counts_map)
}

fn flat_bm25_score(
    query_tokens: &Tokens,
    counted_input: &RecordBatch,
    scorer: &MemBM25Scorer,
) -> Result<RecordBatch> {
    let mut row_ids_builder = UInt64Builder::with_capacity(counted_input.num_rows());
    let mut scores_builder = Float32Builder::with_capacity(counted_input.num_rows());

    let mut row_ids_iter = counted_input
        .column(FLAT_ROW_ID_COL_IDX)
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied();
    let mut all_token_counts_iter = counted_input
        .column(FLAT_ALL_TOKENS_COL_IDX)
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied();
    let mut query_token_counts_iter = counted_input
        .column(FLAT_QUERY_TOKEN_COUNTS_COL_IDX)
        .as_fixed_size_list()
        .values()
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied();
    for _ in 0..counted_input.num_rows() {
        let num_tokens_in_doc = all_token_counts_iter.next().expect_ok()?;
        let row_id = row_ids_iter.next().expect_ok()?;
        if num_tokens_in_doc == 0 {
            for _ in query_tokens {
                query_token_counts_iter.next().expect_ok()?;
            }
            continue;
        }
        let doc_norm = K1 * (1.0 - B + B * num_tokens_in_doc as f32 / scorer.avg_doc_length());
        let mut score = 0.0;
        for token in query_tokens {
            let freq = query_token_counts_iter.next().expect_ok()? as f32;
            let idf = idf(scorer.num_docs_containing_token(token), scorer.num_docs());
            score += idf * (freq * (K1 + 1.0) / (freq + doc_norm));
        }
        if score > 0.0 {
            row_ids_builder.append_value(row_id);
            scores_builder.append_value(score);
        }
    }

    let row_ids = row_ids_builder.finish();
    let scores = scores_builder.finish();
    let batch = RecordBatch::try_new(
        FTS_SCHEMA.clone(),
        vec![Arc::new(row_ids) as ArrayRef, Arc::new(scores) as ArrayRef],
    )?;
    Ok(batch)
}

pub async fn flat_bm25_search_stream(
    input: SendableRecordBatchStream,
    doc_col: String,
    query: String,
    tokenizer: Box<dyn LanceTokenizer>,
    base_scorer: Option<MemBM25Scorer>,
    target_batch_size: usize,
) -> DataFusionResult<SendableRecordBatchStream> {
    let mut tokenizer = tokenizer;
    let query_tokens = Arc::new(collect_query_tokens(&query, &mut tokenizer));

    let input_schema = input.schema();
    let doc_col_idx = input_schema.index_of(&doc_col)?;

    // Accumulate small batches until this threshold before dispatching a task.
    const ACCUMULATE_BYTES: usize = 256 * 1024;
    // Slice oversized batches down to roughly this size.
    const SLICE_BYTES: usize = 512 * 1024;

    // Phase 1 - rechunk the input stream into appropriately sized chunks.  Tokenization is
    // fairly CPU-intensive, and we don't need too much data to justify a new thread task.
    let chunked = lance_arrow::stream::rechunk_stream_by_size(
        input,
        input_schema,
        ACCUMULATE_BYTES,
        SLICE_BYTES,
    );

    // Phase 2 - For each row we need to know the total number of tokens and the count of each
    // of the query tokens.  For example, if the query is "book" and the row is "the book shop"
    // and we are tokenizing with a whitespace tokenizer, we need to know that there are 3 tokens
    // and the token book appears once.
    let counted_input =
        tokenize_and_count(chunked, tokenizer, query_tokens.clone(), doc_col_idx).await?;

    // Phase 3 - Calculate final scores (this is fairly cheap, probably don't need to parallelize)
    let scorer = initialize_scorer(base_scorer.as_ref(), query_tokens.as_ref(), &counted_input);
    let scores = flat_bm25_score(query_tokens.as_ref(), &counted_input, &scorer)?;

    // Finally we emit batches according to the target batch size
    let num_out_batches = scores.num_rows().div_ceil(target_batch_size);
    let mut batches = Vec::with_capacity(num_out_batches);
    for i in 0..num_out_batches {
        let start = i * target_batch_size;
        let len = (scores.num_rows() - start).min(target_batch_size);
        batches.push(Ok(scores.slice(start, len)));
    }
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        FTS_SCHEMA.clone(),
        stream::iter(batches),
    )))
}

pub fn is_phrase_query(query: &str) -> bool {
    query.starts_with('\"') && query.ends_with('\"')
}

#[cfg(test)]
mod tests {
    use crate::scalar::inverted::document_tokenizer::DocType;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_io::object_store::ObjectStore;

    use crate::metrics::NoOpMetricsCollector;
    use crate::prefilter::NoFilter;
    use crate::scalar::ScalarIndex;
    use crate::scalar::inverted::builder::{InnerBuilder, PositionRecorder, inverted_list_schema};
    use crate::scalar::inverted::encoding::{
        compress_positions, compress_posting_list_with_tail_codec,
        decompress_posting_list_with_tail_codec, encode_position_stream_block_into,
    };
    use crate::scalar::inverted::query::{FtsSearchParams, Operator};
    use crate::scalar::lance_format::LanceIndexStore;
    use arrow::array::{AsArray, LargeBinaryBuilder, ListBuilder, UInt32Builder};
    use arrow::datatypes::{Float32Type, UInt32Type};
    use arrow_array::{ArrayRef, Float32Array, RecordBatch, StringArray, UInt32Array, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::*;

    #[tokio::test]
    async fn test_posting_builder_remap() {
        let posting_tail_codec = PostingTailCodec::Fixed32;
        let mut builder =
            PostingListBuilder::new_with_posting_tail_codec(false, posting_tail_codec);
        let n = BLOCK_SIZE + 3;
        for i in 0..n {
            builder.add(i as u32, PositionRecorder::Count(1));
        }
        let removed = vec![5, 7];
        builder.remap(&removed);

        let mut expected =
            PostingListBuilder::new_with_posting_tail_codec(false, posting_tail_codec);
        for i in 0..n - removed.len() {
            expected.add(i as u32, PositionRecorder::Count(1));
        }
        let expected_entries = expected.iter().collect::<Vec<_>>();
        let actual_entries = builder.iter().collect::<Vec<_>>();
        assert_eq!(actual_entries, expected_entries);

        // BLOCK_SIZE + 3 elements should be reduced to BLOCK_SIZE + 1,
        // there are still 2 blocks.
        let batch = builder.to_batch(vec![1.0, 2.0]).unwrap();
        let (doc_ids, freqs) = decompress_posting_list_with_tail_codec(
            (n - removed.len()) as u32,
            batch[POSTING_COL]
                .as_list::<i32>()
                .value(0)
                .as_binary::<i64>(),
            posting_tail_codec,
        )
        .unwrap();
        assert!(
            doc_ids
                .iter()
                .zip(expected_entries.iter().map(|(doc_id, _, _)| doc_id))
                .all(|(a, b)| a == b)
        );
        assert!(
            freqs
                .iter()
                .zip(expected_entries.iter().map(|(_, freq, _)| freq))
                .all(|(a, b)| a == b)
        );
    }

    #[test]
    fn test_posting_builder_size_tracking_matches_structure() {
        fn tracked_memory_size(builder: &PostingListBuilder) -> u64 {
            let encoded_blocks_size = builder
                .encoded_blocks
                .iter()
                .map(|encoded_blocks| std::mem::size_of::<EncodedBlocks>() + encoded_blocks.size())
                .sum::<usize>();
            let encoded_positions_size = builder
                .encoded_position_blocks
                .as_ref()
                .map(|positions| std::mem::size_of::<EncodedPositionBlocks>() + positions.size())
                .unwrap_or(0usize);
            (encoded_blocks_size
                + builder.tail_entries.capacity() * std::mem::size_of::<RawDocInfo>()
                + builder.tail_positions.size()
                + encoded_positions_size) as u64
        }

        let mut builder = PostingListBuilder::new(true);
        for doc_id in 0..(BLOCK_SIZE + 5) as u32 {
            builder.add(
                doc_id,
                PositionRecorder::Position(smallvec::smallvec![1, 3, 5]),
            );
        }

        assert_eq!(builder.size(), tracked_memory_size(&builder));
    }

    #[test]
    fn test_posting_builder_flush_releases_tail_position_capacity() {
        let mut builder = PostingListBuilder::new(true);
        let positions = smallvec::SmallVec::<[u32; 2]>::from_vec((0..1024).collect());
        for doc_id in 0..BLOCK_SIZE as u32 {
            builder.add(doc_id, PositionRecorder::Position(positions.clone()));
        }

        assert_eq!(builder.tail_positions.size(), 0);
        assert_eq!(builder.size(), {
            let encoded_blocks_size = builder
                .encoded_blocks
                .iter()
                .map(|encoded_blocks| std::mem::size_of::<EncodedBlocks>() + encoded_blocks.size())
                .sum::<usize>();
            let encoded_positions_size = builder
                .encoded_position_blocks
                .as_ref()
                .map(|positions| std::mem::size_of::<EncodedPositionBlocks>() + positions.size())
                .unwrap_or(0usize);
            (encoded_blocks_size
                + builder.tail_entries.capacity() * std::mem::size_of::<RawDocInfo>()
                + builder.tail_positions.size()
                + encoded_positions_size) as u64
        });
    }

    #[test]
    fn test_posting_builder_streamed_positions_roundtrip() {
        let mut builder = PostingListBuilder::new(true);
        assert!(builder.add_occurrence(0, 1).unwrap());
        assert!(!builder.add_occurrence(0, 4).unwrap());
        assert!(!builder.add_occurrence(0, 9).unwrap());
        builder.finish_open_doc(0).unwrap();

        assert!(builder.add_occurrence(2, 3).unwrap());
        builder.finish_open_doc(2).unwrap();

        let entries = builder.iter().collect::<Vec<_>>();
        assert_eq!(
            entries,
            vec![
                (0_u32, 3_u32, Some(vec![1_u32, 4_u32, 9_u32])),
                (2_u32, 1_u32, Some(vec![3_u32])),
            ]
        );
    }

    #[test]
    fn test_posting_builder_roundtrip_shared_positions() {
        let entries = vec![
            (0_u32, vec![1_u32, 5]),
            (2, vec![0, 4, 9]),
            (4, vec![7]),
            (8, vec![3, 10]),
            (13, vec![2, 11, 30]),
        ];
        let mut builder =
            PostingListBuilder::new_with_posting_tail_codec(true, PostingTailCodec::VarintDelta);
        for (doc_id, positions) in &entries {
            builder.add(
                *doc_id,
                PositionRecorder::Position(positions.clone().into()),
            );
        }

        let batch = builder.to_batch(vec![1.0]).unwrap();
        assert!(batch.column_by_name(COMPRESSED_POSITION_COL).is_some());
        assert!(batch.column_by_name(POSITION_COL).is_none());
        assert_eq!(
            batch.schema_ref().metadata().get(POSTING_TAIL_CODEC_KEY),
            Some(&PostingTailCodec::VarintDelta.as_str().to_owned())
        );
        assert_eq!(
            batch.schema_ref().metadata().get(POSITIONS_LAYOUT_KEY),
            Some(&POSITIONS_LAYOUT_SHARED_STREAM_V2.to_owned())
        );
        assert_eq!(
            batch.schema_ref().metadata().get(POSITIONS_CODEC_KEY),
            Some(&PositionStreamCodec::PackedDelta.as_str().to_owned())
        );

        let posting =
            PostingList::from_batch(&batch, Some(1.0), Some(entries.len() as u32)).unwrap();
        let actual = posting
            .iter()
            .map(|(doc_id, freq, positions)| {
                (doc_id as u32, freq, positions.unwrap().collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();
        let expected = entries
            .iter()
            .map(|(doc_id, positions)| (*doc_id, positions.len() as u32, positions.clone()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_posting_builder_roundtrip_legacy_positions() {
        let entries = vec![(0_u32, vec![1_u32, 5]), (2, vec![0, 4, 9]), (4, vec![7])];
        let mut builder =
            PostingListBuilder::new_with_posting_tail_codec(true, PostingTailCodec::Fixed32);
        for (doc_id, positions) in &entries {
            builder.add(
                *doc_id,
                PositionRecorder::Position(positions.clone().into()),
            );
        }

        let batch = builder.to_batch(vec![1.0]).unwrap();
        assert!(batch.column_by_name(POSITION_COL).is_some());
        assert!(batch.column_by_name(COMPRESSED_POSITION_COL).is_none());
        assert_eq!(
            batch.schema_ref().metadata().get(POSTING_TAIL_CODEC_KEY),
            None
        );
        assert_eq!(
            batch.schema_ref().metadata().get(POSITIONS_LAYOUT_KEY),
            None
        );
        assert_eq!(batch.schema_ref().metadata().get(POSITIONS_CODEC_KEY), None);

        let posting =
            PostingList::from_batch(&batch, Some(1.0), Some(entries.len() as u32)).unwrap();
        let actual = posting
            .iter()
            .map(|(doc_id, freq, positions)| {
                (doc_id as u32, freq, positions.unwrap().collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();
        let expected = entries
            .iter()
            .map(|(doc_id, positions)| (*doc_id, positions.len() as u32, positions.clone()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_resolve_fts_format_version_defaults_to_v1() {
        assert_eq!(
            resolve_fts_format_version(None).unwrap(),
            InvertedListFormatVersion::V1
        );
        assert_eq!(
            resolve_fts_format_version(Some("2")).unwrap(),
            InvertedListFormatVersion::V2
        );
    }

    #[test]
    fn test_legacy_compressed_positions_still_readable() {
        let doc_ids = [1_u32, 3_u32];
        let frequencies = [2_u32, 3_u32];
        let posting = compress_posting_list_with_tail_codec(
            doc_ids.len(),
            doc_ids.iter(),
            frequencies.iter(),
            std::iter::once(1.0_f32),
            PostingTailCodec::Fixed32,
        )
        .unwrap();

        let mut posting_builder = ListBuilder::new(LargeBinaryBuilder::new());
        for idx in 0..posting.len() {
            posting_builder.values().append_value(posting.value(idx));
        }
        posting_builder.append(true);

        let mut positions_builder = ListBuilder::new(ListBuilder::new(LargeBinaryBuilder::new()));
        for positions in [vec![1_u32, 5_u32], vec![0_u32, 4_u32, 9_u32]] {
            let compressed = compress_positions(&positions).unwrap();
            let doc_builder = positions_builder.values();
            for idx in 0..compressed.len() {
                doc_builder.values().append_value(compressed.value(idx));
            }
            doc_builder.append(true);
        }
        positions_builder.append(true);

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                POSTING_COL,
                DataType::List(Arc::new(Field::new("item", DataType::LargeBinary, true))),
                false,
            ),
            Field::new(MAX_SCORE_COL, DataType::Float32, false),
            Field::new(LENGTH_COL, DataType::UInt32, false),
            Field::new(
                POSITION_COL,
                DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::List(Arc::new(Field::new("item", DataType::LargeBinary, true))),
                    true,
                ))),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(posting_builder.finish()) as ArrayRef,
                Arc::new(Float32Array::from(vec![1.0])) as ArrayRef,
                Arc::new(UInt32Array::from(vec![doc_ids.len() as u32])) as ArrayRef,
                Arc::new(positions_builder.finish()) as ArrayRef,
            ],
        )
        .unwrap();

        let posting =
            PostingList::from_batch(&batch, Some(1.0), Some(doc_ids.len() as u32)).unwrap();
        let actual = posting
            .iter()
            .map(|(doc_id, freq, positions)| {
                (doc_id as u32, freq, positions.unwrap().collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, vec![(1, 2, vec![1, 5]), (3, 3, vec![0, 4, 9]),]);
    }

    #[test]
    fn test_shared_stream_v2_without_codec_still_readable() {
        let doc_ids = [1_u32, 3_u32];
        let frequencies = [2_u32, 3_u32];
        let posting = compress_posting_list_with_tail_codec(
            doc_ids.len(),
            doc_ids.iter(),
            frequencies.iter(),
            std::iter::once(1.0_f32),
            PostingTailCodec::Fixed32,
        )
        .unwrap();

        let mut posting_builder = ListBuilder::new(LargeBinaryBuilder::new());
        for idx in 0..posting.len() {
            posting_builder.values().append_value(posting.value(idx));
        }
        posting_builder.append(true);

        let positions = vec![1_u32, 5_u32, 0_u32, 4_u32, 9_u32];
        let mut encoded_positions = Vec::new();
        encode_position_stream_block_into(
            &positions,
            &frequencies,
            PositionStreamCodec::VarintDocDelta,
            &mut encoded_positions,
        )
        .unwrap();

        let mut position_offsets = ListBuilder::new(UInt32Builder::new());
        position_offsets.values().append_value(0);
        position_offsets.append(true);

        let schema = Arc::new(Schema::new_with_metadata(
            vec![
                Field::new(
                    POSTING_COL,
                    DataType::List(Arc::new(Field::new("item", DataType::LargeBinary, true))),
                    false,
                ),
                Field::new(MAX_SCORE_COL, DataType::Float32, false),
                Field::new(LENGTH_COL, DataType::UInt32, false),
                Field::new(COMPRESSED_POSITION_COL, DataType::LargeBinary, false),
                Field::new(
                    POSITION_BLOCK_OFFSET_COL,
                    DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
                    false,
                ),
            ],
            HashMap::from([(
                POSITIONS_LAYOUT_KEY.to_owned(),
                POSITIONS_LAYOUT_SHARED_STREAM_V2.to_owned(),
            )]),
        ));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(posting_builder.finish()) as ArrayRef,
                Arc::new(Float32Array::from(vec![1.0])) as ArrayRef,
                Arc::new(UInt32Array::from(vec![doc_ids.len() as u32])) as ArrayRef,
                Arc::new(arrow_array::LargeBinaryArray::from(vec![Some(
                    encoded_positions.as_slice(),
                )])) as ArrayRef,
                Arc::new(position_offsets.finish()) as ArrayRef,
            ],
        )
        .unwrap();

        let posting =
            PostingList::from_batch(&batch, Some(1.0), Some(doc_ids.len() as u32)).unwrap();
        let actual = posting
            .iter()
            .map(|(doc_id, freq, positions)| {
                (doc_id as u32, freq, positions.unwrap().collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, vec![(1, 2, vec![1, 5]), (3, 3, vec![0, 4, 9]),]);
    }

    #[test]
    fn test_shared_position_stream_is_smaller_for_sparse_positions() {
        let mut builder =
            PostingListBuilder::new_with_posting_tail_codec(true, PostingTailCodec::VarintDelta);
        let mut legacy_positions = Vec::with_capacity(BLOCK_SIZE * 4);
        for doc_id in 0..(BLOCK_SIZE * 4) as u32 {
            let mut positions = vec![doc_id * 3 + 1];
            if doc_id % 8 == 0 {
                positions.push(doc_id * 3 + 2);
            }
            builder.add(doc_id, PositionRecorder::Position(positions.clone().into()));
            legacy_positions.push(positions);
        }

        let batch = builder.to_batch(vec![1.0; 4]).unwrap();
        let shared_positions_size = batch[COMPRESSED_POSITION_COL].get_buffer_memory_size()
            + batch[POSITION_BLOCK_OFFSET_COL].get_buffer_memory_size();

        let mut positions_builder = ListBuilder::new(ListBuilder::new(LargeBinaryBuilder::new()));
        for positions in legacy_positions {
            let compressed = compress_positions(&positions).unwrap();
            let doc_builder = positions_builder.values();
            for idx in 0..compressed.len() {
                doc_builder.values().append_value(compressed.value(idx));
            }
            doc_builder.append(true);
        }
        positions_builder.append(true);
        let legacy_positions_size = positions_builder.finish().get_buffer_memory_size();

        assert!(
            shared_positions_size < legacy_positions_size,
            "expected shared position stream to be smaller than legacy per-doc storage, shared={shared_positions_size}, legacy={legacy_positions_size}",
        );
    }

    #[test]
    fn test_posting_list_batch_matches_docset_scoring() {
        let mut docs = DocSet::default();
        let num_docs = BLOCK_SIZE + 3;
        for doc_id in 0..num_docs as u32 {
            docs.append(doc_id as u64, doc_id % 7 + 1);
        }

        let doc_ids = (0..num_docs as u32).collect::<Vec<_>>();
        let freqs = doc_ids
            .iter()
            .map(|doc_id| doc_id % 5 + 1)
            .collect::<Vec<_>>();

        let mut builder_scores = PostingListBuilder::new(false);
        let mut builder_docs = PostingListBuilder::new(false);
        for (&doc_id, &freq) in doc_ids.iter().zip(freqs.iter()) {
            builder_scores.add(doc_id, PositionRecorder::Count(freq));
            builder_docs.add(doc_id, PositionRecorder::Count(freq));
        }

        let block_max_scores = docs.calculate_block_max_scores(doc_ids.iter(), freqs.iter());
        let batch_scores = builder_scores.to_batch(block_max_scores).unwrap();
        let batch_docs = builder_docs
            .to_batch_with_docs(&docs, inverted_list_schema(false))
            .unwrap();

        let scores_posting = batch_scores[POSTING_COL].as_list::<i32>().value(0);
        let scores_posting = scores_posting.as_binary::<i64>();
        let docs_posting = batch_docs[POSTING_COL].as_list::<i32>().value(0);
        let docs_posting = docs_posting.as_binary::<i64>();
        assert_eq!(scores_posting, docs_posting);

        let score_left = batch_scores[MAX_SCORE_COL]
            .as_primitive::<Float32Type>()
            .value(0);
        let score_right = batch_docs[MAX_SCORE_COL]
            .as_primitive::<Float32Type>()
            .value(0);
        assert!((score_left - score_right).abs() < 1e-6);

        let len_left = batch_scores[LENGTH_COL]
            .as_primitive::<UInt32Type>()
            .value(0);
        let len_right = batch_docs[LENGTH_COL].as_primitive::<UInt32Type>().value(0);
        assert_eq!(len_left, len_right);
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
            .bm25_search(tokens, params, Operator::Or, prefilter, metrics, None)
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

    #[tokio::test]
    async fn test_modern_prewarm_shrinks_cached_posting_buffers() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder.tokens.add("alpha".to_owned());
        builder.tokens.add("beta".to_owned());
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder.posting_lists[0].add(1, PositionRecorder::Count(2));
        builder.posting_lists[1].add(2, PositionRecorder::Count(3));
        builder.posting_lists[1].add(3, PositionRecorder::Count(4));
        builder.docs.append(100, 1);
        builder.docs.append(101, 2);
        builder.docs.append(102, 3);
        builder.docs.append(103, 4);
        builder.write(store.as_ref()).await.unwrap();

        let metadata = std::collections::HashMap::from_iter(vec![
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0u64]).unwrap(),
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

        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();
        let inverted_list = &index.partitions[0].inverted_list;
        assert!(
            inverted_list.offsets.is_none(),
            "test should use modern posting layout"
        );

        inverted_list.prewarm_posting_lists(false).await.unwrap();

        let alpha = inverted_list
            .index_cache
            .get_with_key(&PostingListKey { token_id: 0 })
            .await
            .unwrap();
        let beta = inverted_list
            .index_cache
            .get_with_key(&PostingListKey { token_id: 1 })
            .await
            .unwrap();

        let PostingList::Compressed(alpha) = alpha.as_ref() else {
            panic!("expected compressed posting list for token 0");
        };
        let PostingList::Compressed(beta) = beta.as_ref() else {
            panic!("expected compressed posting list for token 1");
        };

        assert_ne!(
            alpha.blocks.values().as_ptr(),
            beta.blocks.values().as_ptr(),
            "prewarm should not leave cached posting lists sharing the same values buffer"
        );
    }

    #[tokio::test]
    async fn test_prewarm_with_positions_populates_separate_position_cache() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new_with_format_version(
            0,
            true,
            TokenSetFormat::default(),
            InvertedListFormatVersion::V1,
        );
        builder.tokens.add("hello".to_owned());
        builder.tokens.add("world".to_owned());
        builder
            .posting_lists
            .push(PostingListBuilder::new_with_posting_tail_codec(
                true,
                PostingTailCodec::Fixed32,
            ));
        builder
            .posting_lists
            .push(PostingListBuilder::new_with_posting_tail_codec(
                true,
                PostingTailCodec::Fixed32,
            ));
        builder.posting_lists[0].add(0, PositionRecorder::Position(vec![0].into()));
        builder.posting_lists[1].add(0, PositionRecorder::Position(vec![1].into()));
        builder.posting_lists[0].add(1, PositionRecorder::Position(vec![0].into()));
        builder.posting_lists[1].add(1, PositionRecorder::Position(vec![2].into()));
        builder.docs.append(100, 2);
        builder.docs.append(101, 2);
        builder.write(store.as_ref()).await.unwrap();

        let metadata = std::collections::HashMap::from_iter(vec![
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0_u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default().with_position(true)).unwrap(),
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

        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();

        index
            .prewarm_with_options(&FtsPrewarmOptions::new().with_position(true))
            .await
            .unwrap();

        let inverted_list = &index.partitions[0].inverted_list;
        let posting = inverted_list
            .index_cache
            .get_with_key(&PostingListKey { token_id: 0 })
            .await
            .unwrap();
        assert!(
            !posting.has_position(),
            "posting cache should remain positions-free after prewarm"
        );

        let positions = inverted_list
            .index_cache
            .get_with_key(&PositionKey { token_id: 0 })
            .await
            .unwrap();
        assert!(
            matches!(
                positions.as_ref().0,
                CompressedPositionStorage::LegacyPerDoc(_)
            ),
            "positions should be stored in the dedicated position cache"
        );
    }

    #[tokio::test]
    async fn test_prewarm_with_v2_positions_preserves_shared_stream_codec() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let format_version = InvertedListFormatVersion::V2;
        let posting_tail_codec = format_version.posting_tail_codec();
        let mut builder = InnerBuilder::new_with_format_version(
            0,
            true,
            TokenSetFormat::default(),
            format_version,
        );
        builder.tokens.add("body".to_owned());

        let mut posting_list =
            PostingListBuilder::new_with_posting_tail_codec(true, posting_tail_codec);
        let expected = (0..(BLOCK_SIZE + 5) as u32)
            .map(|doc_id| {
                let positions = vec![doc_id % 3, doc_id % 3 + 2, doc_id % 3 + 5];
                posting_list.add(doc_id, PositionRecorder::Position(positions.clone().into()));
                builder.docs.append(30_000 + doc_id as u64, 20 + doc_id % 7);
                (doc_id, positions.len() as u32, positions)
            })
            .collect::<Vec<_>>();
        builder.posting_lists.push(posting_list);
        builder.write(store.as_ref()).await.unwrap();

        let metadata = HashMap::from([
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0_u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default().with_position(true)).unwrap(),
            ),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
            (
                POSTING_TAIL_CODEC_KEY.to_owned(),
                posting_tail_codec.as_str().to_owned(),
            ),
            (
                POSITIONS_LAYOUT_KEY.to_owned(),
                POSITIONS_LAYOUT_SHARED_STREAM_V2.to_owned(),
            ),
            (
                POSITIONS_CODEC_KEY.to_owned(),
                PositionStreamCodec::PackedDelta.as_str().to_owned(),
            ),
        ]);
        let mut writer = store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();

        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store, None, cache.as_ref())
            .await
            .unwrap();
        index
            .prewarm_with_options(&FtsPrewarmOptions::new().with_position(true))
            .await
            .unwrap();

        let actual = index.partitions[0]
            .inverted_list
            .posting_list(0, true, &NoOpMetricsCollector)
            .await
            .unwrap()
            .iter()
            .map(|(doc_id, freq, positions)| {
                (doc_id as u32, freq, positions.unwrap().collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_block_max_scores_capacity_matches_block_count() {
        let mut docs = DocSet::default();
        let num_docs = BLOCK_SIZE * 3 + 7;
        let doc_ids = (0..num_docs as u32).collect::<Vec<_>>();
        for doc_id in &doc_ids {
            docs.append(*doc_id as u64, 1);
        }

        let freqs = vec![1_u32; doc_ids.len()];
        let block_max_scores = docs.calculate_block_max_scores(doc_ids.iter(), freqs.iter());
        let expected_blocks = doc_ids.len().div_ceil(BLOCK_SIZE);

        assert_eq!(block_max_scores.len(), expected_blocks);
        assert_eq!(block_max_scores.capacity(), expected_blocks);
    }

    #[tokio::test]
    async fn test_bm25_search_uses_global_idf() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Partition 0: 3 docs, only one contains "alpha".
        let mut builder0 = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder0.tokens.add("alpha".to_owned());
        builder0.tokens.add("beta".to_owned());
        builder0.posting_lists.push(PostingListBuilder::new(false));
        builder0.posting_lists.push(PostingListBuilder::new(false));
        builder0.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder0.posting_lists[1].add(1, PositionRecorder::Count(1));
        builder0.posting_lists[1].add(2, PositionRecorder::Count(1));
        builder0.docs.append(100, 1);
        builder0.docs.append(101, 1);
        builder0.docs.append(102, 1);
        builder0.write(store.as_ref()).await.unwrap();

        // Partition 1: 1 doc, contains "alpha".
        let mut builder1 = InnerBuilder::new(1, false, TokenSetFormat::default());
        builder1.tokens.add("alpha".to_owned());
        builder1.posting_lists.push(PostingListBuilder::new(false));
        builder1.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder1.docs.append(200, 1);
        builder1.write(store.as_ref()).await.unwrap();

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

        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();

        let tokens = Arc::new(Tokens::new(vec!["alpha".to_string()], DocType::Text));
        let params = Arc::new(FtsSearchParams::new().with_limit(Some(10)));
        let prefilter = Arc::new(NoFilter);
        let metrics = Arc::new(NoOpMetricsCollector);

        let (row_ids, scores) = index
            .bm25_search(tokens, params, Operator::Or, prefilter, metrics, None)
            .await
            .unwrap();

        assert_eq!(row_ids.len(), 2);
        assert!(row_ids.contains(&100));
        assert!(row_ids.contains(&200));
        assert_eq!(row_ids.len(), scores.len());

        let expected_idf = idf(2, 4);
        for score in scores {
            assert!(
                (score - expected_idf).abs() < 1e-6,
                "score: {}, expected: {}",
                score,
                expected_idf
            );
        }
    }

    #[tokio::test]
    async fn test_phrase_query_reads_legacy_per_doc_positions() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new_with_format_version(
            0,
            true,
            TokenSetFormat::default(),
            InvertedListFormatVersion::V1,
        );
        builder.tokens.add("hello".to_owned());
        builder.tokens.add("world".to_owned());
        builder
            .posting_lists
            .push(PostingListBuilder::new_with_posting_tail_codec(
                true,
                PostingTailCodec::Fixed32,
            ));
        builder
            .posting_lists
            .push(PostingListBuilder::new_with_posting_tail_codec(
                true,
                PostingTailCodec::Fixed32,
            ));
        builder.posting_lists[0].add(0, PositionRecorder::Position(vec![0].into()));
        builder.posting_lists[1].add(0, PositionRecorder::Position(vec![1].into()));
        builder.posting_lists[0].add(1, PositionRecorder::Position(vec![0].into()));
        builder.posting_lists[1].add(1, PositionRecorder::Position(vec![2].into()));
        builder.docs.append(100, 2);
        builder.docs.append(101, 2);
        builder.write(store.as_ref()).await.unwrap();

        let metadata = std::collections::HashMap::from_iter(vec![
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0_u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default().with_position(true)).unwrap(),
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

        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();

        let tokens = Arc::new(Tokens::new(
            vec!["hello".to_owned(), "world".to_owned()],
            DocType::Text,
        ));
        let params = Arc::new(
            FtsSearchParams::new()
                .with_limit(Some(10))
                .with_phrase_slop(Some(0)),
        );
        let prefilter = Arc::new(NoFilter);
        let metrics = Arc::new(NoOpMetricsCollector);

        let (row_ids, _scores) = index
            .bm25_search(tokens, params, Operator::And, prefilter, metrics, None)
            .await
            .unwrap();

        assert_eq!(row_ids, vec![100]);
    }

    #[tokio::test]
    async fn test_update_preserves_loaded_v2_format_version() -> Result<()> {
        let src_dir = TempObjDir::default();
        let dest_dir = TempObjDir::default();
        let src_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            src_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        let dest_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            dest_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let format_version = InvertedListFormatVersion::V2;
        let posting_tail_codec = format_version.posting_tail_codec();
        let mut partition = InnerBuilder::new_with_format_version(
            0,
            false,
            TokenSetFormat::default(),
            format_version,
        );
        partition.tokens.add("hello".to_owned());
        let mut posting_list =
            PostingListBuilder::new_with_posting_tail_codec(false, posting_tail_codec);
        posting_list.add(0, PositionRecorder::Count(1));
        partition.posting_lists.push(posting_list);
        partition.docs.append(100, 1);
        partition.write(src_store.as_ref()).await?;

        let metadata = HashMap::from([
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0_u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default()).unwrap(),
            ),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
            (
                POSTING_TAIL_CODEC_KEY.to_owned(),
                posting_tail_codec.as_str().to_owned(),
            ),
        ]);
        let mut writer = src_store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();

        let index = InvertedIndex::load(src_store, None, &LanceCache::no_cache()).await?;
        assert_eq!(index.index_version(), format_version.index_version());

        let schema = Arc::new(Schema::new(vec![
            Field::new("doc", DataType::Utf8, true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let docs = Arc::new(StringArray::from(vec![Some("hello again")]));
        let row_ids = Arc::new(UInt64Array::from(vec![101u64]));
        let batch = RecordBatch::try_new(schema.clone(), vec![docs, row_ids])?;
        let stream = RecordBatchStreamAdapter::new(schema, stream::iter(vec![Ok(batch)]));
        let created = index
            .update(Box::pin(stream), dest_store.as_ref(), None)
            .await?;

        assert_eq!(created.index_version, format_version.index_version());

        let updated = InvertedIndex::load(dest_store, None, &LanceCache::no_cache()).await?;
        assert_eq!(updated.index_version(), format_version.index_version());
        assert_eq!(updated.partitions.len(), 2);
        for partition in &updated.partitions {
            assert_eq!(
                partition.inverted_list.posting_tail_codec(),
                posting_tail_codec
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_modern_index_without_deleted_col_has_empty_bitmap() {
        // An index created before the deleted_fragments feature was added
        // will have a metadata file with num_rows=0 (no record batch data).
        // The load path should gracefully handle this with an empty bitmap.
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder.tokens.add("test".to_owned());
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder.docs.append(100, 1);
        builder.write(store.as_ref()).await.unwrap();

        // Write a metadata file WITHOUT the deleted_fragments column
        // (simulates an older index version)
        let metadata = std::collections::HashMap::from_iter(vec![
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0u64]).unwrap(),
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

        let index = InvertedIndex::load(store, None, &LanceCache::no_cache())
            .await
            .unwrap();
        assert!(
            index.deleted_fragments().is_empty(),
            "index without deleted_fragments column should have empty bitmap"
        );
    }
}
