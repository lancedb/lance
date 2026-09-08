// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MinHash LSH index for near-duplicate detection over text columns.
//!
//! The index answers "which rows have the highest estimated Jaccard similarity
//! (over token shingles) to this query text?" and is the building block for
//! fuzzy deduplication of training corpora, web crawls, and RAG ingestion.
//!
//! Every indexed row gets a signature of `num_hashes` MinHash values, computed
//! from the row's token shingles. The signature is split into `num_bands`
//! bands; rows that agree on at least one band become candidates and are then
//! refined by comparing their full signatures against the query signature.
//!
//! A segment stores two files:
//!
//! ```text
//! segment (_indices/{uuid}/)
//! ├── signatures.lance   one row per indexed document, row number = doc id
//! │     ├── _rowid     UInt64
//! │     └── signature  FixedSizeList<UInt16, num_hashes>   (full-zip, no compression)
//! └── bands.lance        one row per (band key, doc id), ascending
//!       ├── band_key   UInt64   band id (high 8 bits) | hash of band values (low 56 bits)
//!       └── doc_id     UInt32   row number in signatures.lance
//!       (schema metadata: params json, page rows; global buffer: page table)
//! ```
//!
//! A bucket is the run of rows sharing a band key; it may span pages. Lookups
//! binary-search a resident page table (max band key per page), read the
//! candidate pages with one scattered read, binary-search each page for the
//! band key, collect the bucket's doc ids into a candidate bitmap, and read the
//! candidate signatures with a second scattered read (or a sequential scan when
//! the candidate set is dense).

use std::any::Any;
use std::collections::{BinaryHeap, HashMap};
use std::ops::Range;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};

use arrow_array::cast::AsArray;
use arrow_array::types::{UInt16Type, UInt32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, RecordBatch, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use bytes::Bytes;
use datafusion::execution::SendableRecordBatchStream;
use futures::future::try_join_all;
use futures::stream::FuturesOrdered;
use futures::{Stream, StreamExt, TryStreamExt};
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder, LanceCache, WeakLanceCache};
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::utils::tracing::{IO_TYPE_LOAD_SCALAR_PART, TRACE_IO_EVENTS};
use lance_core::{Error, ROW_ID, Result};
use lance_encoding::constants::{
    COMPRESSION_META_KEY, STRUCTURAL_ENCODING_FULLZIP, STRUCTURAL_ENCODING_META_KEY,
};
use lance_select::RowAddrMask;
use lance_tokenizer::TokenStream;
use rayon::slice::ParallelSliceMut;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use twox_hash::XxHash64;

use crate::scalar::expression::ScalarQueryParser;
use crate::scalar::inverted::tokenizer::InvertedIndexParams;
use crate::scalar::inverted::tokenizer::document_tokenizer::LanceTokenizer;
use crate::scalar::registry::{
    BasicTrainer, ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
    VALUE_COLUMN_NAME,
};
use crate::scalar::{
    AnyQuery, BuiltinIndexType, CreatedIndex, IndexFile, IndexReader, IndexStore, IndexWriter,
    MetricsCollector, OldIndexDataFilter, RowIdRemapper, ScalarIndex, ScalarIndexParams,
    SearchResult, UpdateCriteria,
};
use crate::{Index, IndexType};
use crate::{pb, pbold};

/// On-disk format version of the index files. The format is unstable; bump
/// on any layout change instead of adding compatibility paths.
pub const MINHASH_LSH_INDEX_VERSION: u32 = 0;
/// Version of the signature generation behavior (tokenizer pipeline, shingle
/// hashing, permutation family and the 16-bit truncation); bumped whenever a
/// change would make old and new signatures incomparable.
pub const SIGNATURE_VERSION: u32 = 0;

/// Seed of the shingle hash and of the generator that derives the permutation
/// and compression coefficients. Part of the signature scheme: changing it
/// changes every signature and therefore [`SIGNATURE_VERSION`].
const SIGNATURE_SEED: u64 = 42;

/// A stored MinHash value: the 64-bit permuted minimum compressed to 16 bits
/// by multiply-shift hashing (b-bit MinHash). A minimum is not uniformly
/// distributed (it shrinks as the set grows), so its bits cannot be stored
/// directly; multiply-shift with a random odd multiplier is 2-universal, so
/// two unrelated documents agree on a stored value with probability at most
/// 2^-15 whatever their length, far below the 1/num_hashes resolution of the
/// Jaccard estimate.
pub type SignatureValue = u16;

pub const SIGNATURES_FILENAME: &str = "signatures.lance";
pub const BANDS_FILENAME: &str = "bands.lance";
pub const SIGNATURE_COL: &str = "signature";
pub const BAND_KEY_COL: &str = "band_key";
pub const DOC_ID_COL: &str = "doc_id";

const PARAMS_META_KEY: &str = "minhash_lsh_params";
const INDEX_VERSION_META_KEY: &str = "minhash_lsh_index_version";
const PAGE_ROWS_META_KEY: &str = "minhash_lsh_page_rows";
const PAGE_TABLE_BUFFER_META_KEY: &str = "minhash_lsh_page_table_buffer";
const NUM_DOCS_META_KEY: &str = "minhash_lsh_num_docs";

/// Number of band-key rows per logical page of `bands.lance`. The page table
/// keeps one u64 per page in memory, so 4096 rows per page keeps the table at
/// ~0.01% of the bands file while one page read stays small (tens of KiB for
/// typical postings).
const DEFAULT_PAGE_ROWS: usize = 4096;
/// Byte inserted between the tokens of one shingle so that different token
/// splits of the same characters hash differently.
const SHINGLE_SEPARATOR: u8 = 0x1F;
/// A candidate set covering more than this percentage of a segment is refined
/// with a sequential scan of the signature file instead of a scattered read.
const SPARSE_REFINE_READ_PERCENT: u64 = 10;
/// Candidate rows per scattered read while refining: a candidate set just
/// under the dense threshold is still a tenth of the segment.
const REFINE_READ_ROWS: usize = 1 << 16;
/// Documents per resident chunk of the signature table (136 MB at 64 hashes).
/// Chunks are separate cache entries, so a prewarm keeps as many as the
/// cache holds and a query scores candidates in resident chunks from memory.
const SIGNATURE_CHUNK_DOCS: usize = 1 << 20;
/// Rows per read while scanning the signature table end to end (16 MiB at
/// the default width); `IndexReader::read_range_stream` reads 4096-row
/// batches only two deep, which leaves the scheduler mostly idle.
const SIGNATURE_SCAN_ROWS: usize = 32 * 1024;
/// Reads in flight while scanning an index file end to end.
const SCAN_READ_CONCURRENCY: usize = 4;

/// Memory budget for one in-memory sort run of (band key, doc id) records.
/// Runs beyond this are sorted and spilled to temporary index files that are
/// merged when the bands file is written.
const DEFAULT_SORT_RUN_BYTES: usize = 256 * 1024 * 1024;
/// Runs sorted and written concurrently with signing. Each holds a full run
/// in memory, so this bounds the backlog when signing outpaces the disk.
const MAX_INFLIGHT_SPILLS: usize = 4;
/// Signed batches queued between the driver that polls the signing stream
/// and the loop that consumes them.
const SIGNED_BATCH_QUEUE: usize = 16;
/// Rows per batch handed to the signature file writer. Signed batches follow
/// the scan's 8192-row batches; the writer's per-batch cost makes larger
/// batches noticeably faster (about 15% on the signing stage at 10^8 rows).
const SIGNATURE_WRITE_BATCH_ROWS: usize = 1 << 16;
/// Rows per batch when writing sort runs.
const SPILL_BATCH_ROWS: usize = 1 << 20;
/// Every sorted run records the row at which each key-range partition
/// starts, so the merge can gather one partition from every run, sort it in
/// memory and write it, partitions processed independently and in key order.
/// A band gets `SPILL_PARTITIONS / num_bands` partitions split on the hash
/// bits below the band id, so a partition holds `num_bands / SPILL_PARTITIONS`
/// of all records: 1.5 MB per 10^9 rows at eight bands, 47 MB at 256.
const SPILL_PARTITIONS: usize = 1 << 16;
/// Records of adjacent partitions merged as one group (about 32 MiB), each
/// group sorted in memory on the CPU pool.
const MERGE_GROUP_RECORDS: usize = (32 * 1024 * 1024) / std::mem::size_of::<(u64, u32)>();
/// Upper bound on groups gathered and sorted concurrently; the actual count
/// is a quarter of the CPU pool, so the in-flight reads (two per group) stay
/// at half the runtime workers, each of which a read can park while its
/// decode waits for IO.
const MERGE_GROUPS_IN_FLIGHT: usize = 16;
/// Reads in flight while a group gathers its records from the runs.
const MERGE_GROUP_READ_CONCURRENCY: usize = 2;
/// Records per batch written to the bands file from a resident run.
const BANDS_WRITE_BATCH_RECORDS: usize = 1 << 20;
/// Highest band count representable in the 8-bit band id prefix of a band key.
const MAX_NUM_BANDS: u32 = 256;

/// `bands.lance`: (band key, doc id) records in ascending order. Both columns
/// are fixed width and stored raw, so a page is one exact ranged read that
/// decodes without copying.
static BANDS_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new(BAND_KEY_COL, DataType::UInt64, false).with_metadata(HashMap::from([(
            COMPRESSION_META_KEY.to_string(),
            "none".to_string(),
        )])),
        Field::new(DOC_ID_COL, DataType::UInt32, false).with_metadata(HashMap::from([(
            COMPRESSION_META_KEY.to_string(),
            "none".to_string(),
        )])),
    ]))
});

/// Sort runs hold the same columns with the default lightweight encodings:
/// they are written once and read back in large sequential ranges.
static SPILL_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new(BAND_KEY_COL, DataType::UInt64, false),
        Field::new(DOC_ID_COL, DataType::UInt32, false),
    ]))
});

fn signatures_schema(num_hashes: i32) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(ROW_ID, DataType::UInt64, false),
        Field::new(
            SIGNATURE_COL,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt16, false)),
                num_hashes,
            ),
            false,
        )
        .with_metadata(HashMap::from([
            (
                STRUCTURAL_ENCODING_META_KEY.to_string(),
                STRUCTURAL_ENCODING_FULLZIP.to_string(),
            ),
            (COMPRESSION_META_KEY.to_string(), "none".to_string()),
        ])),
    ]))
}

/// A MinHash similarity search: the rows of `column` whose token shingles have
/// the highest estimated Jaccard similarity to `text`.
///
/// The number of rows comes from the scan limit and the output carries a
/// `_distance` column equal to `1 - estimated Jaccard similarity`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinHashQuery {
    /// Column with a MinHash LSH index.
    pub column: String,
    /// Query text, tokenized and shingled exactly like the indexed rows.
    pub text: String,
}

impl MinHashQuery {
    pub fn new(text: impl Into<String>, column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            text: text.into(),
        }
    }
}

/// Parameters of a MinHash LSH index.
///
/// These are the only inputs to signature generation, so every segment of a
/// logical index must be built from identical parameters. They are persisted
/// in the index details and re-read on the query side.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MinHashLshIndexParams {
    /// Number of MinHash values per signature (k). Must be a positive
    /// multiple of `num_bands`. The Jaccard estimate has error ~1/sqrt(k).
    pub num_hashes: u32,
    /// Number of LSH bands (b), in `[1, 256]`. Two rows become candidates when
    /// all `num_hashes / num_bands` values of at least one band agree.
    pub num_bands: u32,
    /// Number of consecutive tokens joined into one shingle. Documents shorter
    /// than this produce a single shingle of all their tokens.
    pub shingle_size: u32,
    /// Tokenizer configuration, using the same keys as the inverted (full
    /// text search) index. Defaults to lower-casing without stemming or stop
    /// word removal: aggressive normalization inflates Jaccard similarity and
    /// turns "related" into "duplicate".
    pub tokenizer: InvertedIndexParams,
}

impl Default for MinHashLshIndexParams {
    fn default() -> Self {
        Self {
            num_hashes: 128,
            num_bands: 16,
            shingle_size: 3,
            tokenizer: InvertedIndexParams::default()
                .stem(false)
                .remove_stop_words(false),
        }
    }
}

/// User-facing JSON shape: every key optional, tokenizer keys layered over the
/// MinHash tokenizer defaults rather than the full text search defaults.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMinHashLshIndexParams {
    num_hashes: Option<u32>,
    num_bands: Option<u32>,
    shingle_size: Option<u32>,
    tokenizer: Option<serde_json::Map<String, serde_json::Value>>,
}

impl MinHashLshIndexParams {
    /// Parse user parameters from JSON, filling omitted keys with defaults and
    /// validating the result.
    pub fn from_json(params: &str) -> Result<Self> {
        let params = if params.trim().is_empty() {
            "{}"
        } else {
            params
        };
        let raw: RawMinHashLshIndexParams = serde_json::from_str(params).map_err(|err| {
            Error::invalid_input(format!(
                "invalid MinHash LSH index params {params:?}: {err}"
            ))
        })?;
        let mut resolved = Self::default();
        if let Some(num_hashes) = raw.num_hashes {
            resolved.num_hashes = num_hashes;
        }
        if let Some(num_bands) = raw.num_bands {
            resolved.num_bands = num_bands;
        }
        if let Some(shingle_size) = raw.shingle_size {
            resolved.shingle_size = shingle_size;
        }
        if let Some(tokenizer) = raw.tokenizer {
            // Presets (`analyzer`) and full text search defaults resolve inside
            // InvertedIndexParams; the dedup defaults are layered back for the
            // keys the user did not set.
            let has_stem = tokenizer.contains_key("stem");
            let has_stop_words = tokenizer.contains_key("remove_stop_words");
            let mut parsed: InvertedIndexParams =
                serde_json::from_value(serde_json::Value::Object(tokenizer)).map_err(|err| {
                    Error::invalid_input(format!("invalid MinHash LSH tokenizer params: {err}"))
                })?;
            if !has_stem {
                parsed.stem = false;
            }
            if !has_stop_words {
                parsed.remove_stop_words = false;
            }
            resolved.tokenizer = parsed;
        }
        resolved.validate()?;
        // Keep only what the index details can record, so parameters compare
        // equal wherever they were read from.
        resolved.tokenizer = InvertedIndexParams::try_from(
            &pbold::InvertedIndexDetails::try_from(&resolved.tokenizer)?,
        )?;
        Ok(resolved)
    }

    fn validate(&self) -> Result<()> {
        if self.num_hashes == 0 {
            return Err(Error::invalid_input(
                "MinHash LSH index requires num_hashes > 0".to_string(),
            ));
        }
        if self.num_bands == 0 || self.num_bands > MAX_NUM_BANDS {
            return Err(Error::invalid_input(format!(
                "MinHash LSH index requires 1 <= num_bands <= {MAX_NUM_BANDS}, got num_bands={}",
                self.num_bands
            )));
        }
        if !self.num_hashes.is_multiple_of(self.num_bands) {
            return Err(Error::invalid_input(format!(
                "MinHash LSH index requires num_hashes to be a multiple of num_bands, got num_hashes={} num_bands={}",
                self.num_hashes, self.num_bands
            )));
        }
        if i32::try_from(self.num_hashes).is_err() {
            return Err(Error::invalid_input(format!(
                "MinHash LSH index num_hashes={} exceeds the maximum signature width {}",
                self.num_hashes,
                i32::MAX
            )));
        }
        if self.shingle_size == 0 {
            return Err(Error::invalid_input(
                "MinHash LSH index requires shingle_size > 0".to_string(),
            ));
        }
        // The index details record the tokenizer the way a full text search
        // index does, which has no room for these; refusing them beats
        // silently tokenizing queries differently from the index.
        if self.tokenizer.custom_stop_words.is_some() {
            return Err(Error::invalid_input(
                "MinHash LSH index does not support custom_stop_words: they cannot be recorded in the index details".to_string(),
            ));
        }
        if self.tokenizer.lance_tokenizer.is_some() {
            return Err(Error::invalid_input(
                "MinHash LSH index does not support lance_tokenizer: it cannot be recorded in the index details".to_string(),
            ));
        }
        // Surface tokenizer configuration errors at index creation instead of
        // on the first batch.
        self.tokenizer.build()?;
        Ok(())
    }

    fn to_details(&self) -> Result<pb::MinHashLshIndexDetails> {
        Ok(pb::MinHashLshIndexDetails {
            num_hashes: self.num_hashes,
            num_bands: self.num_bands,
            shingle_size: self.shingle_size,
            tokenizer: Some(pbold::InvertedIndexDetails::try_from(&self.tokenizer)?),
            signature_version: SIGNATURE_VERSION,
        })
    }

    /// Parse the parameters stored in an index's details.
    pub fn from_index_details(details: &prost_types::Any) -> Result<Self> {
        Self::from_details(&details.to_msg::<pb::MinHashLshIndexDetails>()?)
    }

    fn from_details(details: &pb::MinHashLshIndexDetails) -> Result<Self> {
        if details.signature_version != SIGNATURE_VERSION {
            return Err(Error::not_supported(format!(
                "MinHash LSH index was built with signature_version {} but this version of Lance produces signature_version {}; rebuild the index",
                details.signature_version, SIGNATURE_VERSION
            )));
        }
        let tokenizer = details.tokenizer.as_ref().ok_or_else(|| {
            Error::invalid_input("MinHash LSH index details carry no tokenizer".to_string())
        })?;
        let tokenizer = InvertedIndexParams::try_from(tokenizer)?;
        let params = Self {
            num_hashes: details.num_hashes,
            num_bands: details.num_bands,
            shingle_size: details.shingle_size,
            tokenizer,
        };
        params.validate()?;
        Ok(params)
    }

    fn details_any(&self) -> Result<prost_types::Any> {
        Ok(prost_types::Any::from_msg(&self.to_details()?)?)
    }
}

/// Deterministic 64-bit generator (SplitMix64) used to derive the permutation
/// coefficients from the seed. Implemented inline so the sequence is a fixed
/// part of the signature contract rather than a dependency's implementation
/// detail.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Computes signatures and band keys. Cloned per batch during builds and per
/// query during searches; the build and query side must share this code path
/// so signatures stay comparable.
#[derive(Clone)]
pub struct SignatureGenerator {
    tokenizer: Box<dyn LanceTokenizer>,
    shingle_size: usize,
    num_bands: usize,
    /// Odd multipliers of the `num_hashes` permutations `h(x) = a * x + b`.
    multipliers: Vec<u64>,
    /// Increments of the permutations.
    increments: Vec<u64>,
    /// Odd multipliers compressing each 64-bit minimum to its stored 16 bits.
    compressors: Vec<u64>,
    /// Token bytes of the current document, separated by [`SHINGLE_SEPARATOR`],
    /// and the end offset of each token; a shingle is one contiguous slice.
    token_text: Vec<u8>,
    token_ends: Vec<usize>,
    /// 64-bit permuted minima of the current document, mixed into the signature.
    mins: Vec<u64>,
}

impl std::fmt::Debug for SignatureGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignatureGenerator")
            .field("num_hashes", &self.multipliers.len())
            .field("num_bands", &self.num_bands)
            .field("shingle_size", &self.shingle_size)
            .finish()
    }
}

impl SignatureGenerator {
    pub fn try_new(params: &MinHashLshIndexParams) -> Result<Self> {
        params.validate()?;
        let num_hashes = params.num_hashes as usize;
        let mut state = SIGNATURE_SEED;
        let mut multipliers = Vec::with_capacity(num_hashes);
        let mut increments = Vec::with_capacity(num_hashes);
        for _ in 0..num_hashes {
            multipliers.push(splitmix64(&mut state) | 1);
            increments.push(splitmix64(&mut state));
        }
        let compressors = (0..num_hashes)
            .map(|_| splitmix64(&mut state) | 1)
            .collect();
        Ok(Self {
            tokenizer: params.tokenizer.build()?,
            shingle_size: params.shingle_size as usize,
            num_bands: params.num_bands as usize,
            multipliers,
            increments,
            compressors,
            token_text: Vec::new(),
            token_ends: Vec::new(),
            mins: vec![u64::MAX; num_hashes],
        })
    }

    pub fn num_hashes(&self) -> usize {
        self.multipliers.len()
    }

    pub fn num_bands(&self) -> usize {
        self.num_bands
    }

    /// Fill `signature` (length `num_hashes`) with the MinHash signature of
    /// `text`. Returns false, leaving `signature` unspecified, when the text
    /// has no tokens and therefore no signature.
    pub fn signature(&mut self, text: &str, signature: &mut [SignatureValue]) -> bool {
        debug_assert_eq!(
            signature.len(),
            self.multipliers.len(),
            "signature buffer must hold num_hashes values"
        );
        self.token_text.clear();
        self.token_ends.clear();
        {
            let mut stream = self.tokenizer.token_stream_for_doc(text);
            while stream.advance() {
                if !self.token_ends.is_empty() {
                    self.token_text.push(SHINGLE_SEPARATOR);
                }
                self.token_text
                    .extend_from_slice(stream.token().text.as_bytes());
                self.token_ends.push(self.token_text.len());
            }
        }
        let num_tokens = self.token_ends.len();
        if num_tokens == 0 {
            return false;
        }
        self.mins.fill(u64::MAX);
        let window = self.shingle_size.min(num_tokens);
        for start in 0..=(num_tokens - window) {
            // Tokens are stored back to back with separators, so the shingle
            // is the byte range from this token's start to the window's end.
            let from = if start == 0 {
                0
            } else {
                self.token_ends[start - 1] + 1
            };
            let to = self.token_ends[start + window - 1];
            let base = XxHash64::oneshot(SIGNATURE_SEED, &self.token_text[from..to]);
            for ((value, multiplier), increment) in self
                .mins
                .iter_mut()
                .zip(&self.multipliers)
                .zip(&self.increments)
            {
                // Wrapping arithmetic is the hash function itself, not an
                // overflowing counter. The full 64-bit value is compared: the
                // ordering is decided by its high bits, so the weak low bits
                // of `a * x + b` only break ties. Branch-free min keeps this
                // loop vectorizable.
                let permuted = multiplier.wrapping_mul(base).wrapping_add(*increment);
                *value = (*value).min(permuted);
            }
        }
        // A minimum shrinks with the number of shingles, so its raw bits are
        // not comparable across documents of different lengths. Multiply-shift
        // with a random odd multiplier is 2-universal for any input
        // distribution: distinct minima collide with probability <= 2^-15.
        for ((value, min), compressor) in
            signature.iter_mut().zip(&self.mins).zip(&self.compressors)
        {
            *value = (compressor.wrapping_mul(*min) >> 48) as SignatureValue;
        }
        true
    }

    /// Append the `num_bands` band keys of `signature` to `keys`.
    ///
    /// A band key is the band id in the high 8 bits and the low 56 bits of
    /// the hash of the band's signature values, so keys of one band are
    /// contiguous in the sorted bands file.
    pub fn band_keys(&self, signature: &[SignatureValue], keys: &mut Vec<u64>) {
        let band_width = signature.len() / self.num_bands;
        let mut band_bytes = Vec::with_capacity(band_width * std::mem::size_of::<SignatureValue>());
        for (band_id, band) in signature.chunks_exact(band_width).enumerate() {
            band_bytes.clear();
            for value in band {
                band_bytes.extend_from_slice(&value.to_le_bytes());
            }
            let hash = XxHash64::oneshot(SIGNATURE_SEED, &band_bytes);
            keys.push(((band_id as u64) << 56) | (hash & 0x00FF_FFFF_FFFF_FFFF));
        }
    }
}

/// Estimated Jaccard similarity of two signatures: the fraction of positions
/// whose MinHash values agree.
pub fn estimate_jaccard(a: &[SignatureValue], b: &[SignatureValue]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "signatures must have the same width");
    let matches = a.iter().zip(b).filter(|(x, y)| x == y).count();
    matches as f32 / a.len() as f32
}

/// Convert a document index into a segment-local doc id.
///
/// This is the only place where a document count becomes a `u32`; a segment
/// that overflows it must be split rather than silently wrapped.
fn checked_doc_id(doc_index: u64) -> Result<u32> {
    u32::try_from(doc_index).map_err(|_| {
        Error::invalid_input(format!(
            "MinHash LSH index segment cannot hold more than {} rows (row index {} does not fit a u32 doc id); \
             build the index in pieces by calling create_index_uncommitted on disjoint fragment subsets \
             and committing the resulting segments together with commit_existing_index_segments",
            u32::MAX as u64 + 1,
            doc_index
        ))
    })
}

/// One logical page of `bands.lance`, cached per page.
#[derive(Debug, DeepSizeOf)]
pub struct BandPage {
    keys: Vec<u64>,
    doc_ids: Vec<u32>,
}

impl BandPage {
    fn try_from_batch(batch: &RecordBatch) -> Result<Self> {
        let (keys, doc_ids) = band_columns(batch, BANDS_FILENAME)?;
        // Copy out of the batch so a cached page neither pins nor is charged
        // for the buffers of every other page read in the same request.
        Ok(Self {
            keys: keys.values().to_vec(),
            doc_ids: doc_ids.values().to_vec(),
        })
    }

    /// Doc ids of the bucket `band_key` within this page.
    fn members(&self, band_key: u64) -> &[u32] {
        let start = self.keys.partition_point(|&key| key < band_key);
        let len = self.keys[start..]
            .iter()
            .take_while(|&&key| key == band_key)
            .count();
        &self.doc_ids[start..start + len]
    }

    /// Whether the bucket `band_key` reaches the end of this page and may
    /// therefore continue on the next one.
    fn continues_after(&self, band_key: u64) -> bool {
        self.keys.last() == Some(&band_key)
    }
}

/// The two non-null columns of a bands or spill batch.
fn band_columns<'a>(
    batch: &'a RecordBatch,
    file: &str,
) -> Result<(&'a UInt64Array, &'a UInt32Array)> {
    let column = |name: &str| {
        batch
            .column_by_name(name)
            .ok_or_else(|| Error::corrupt_file_named(file, format!("missing column {name}")))
    };
    let keys = column(BAND_KEY_COL)?
        .as_primitive_opt::<UInt64Type>()
        .filter(|keys| keys.null_count() == 0)
        .ok_or_else(|| {
            Error::corrupt_file_named(
                file,
                format!("{BAND_KEY_COL} is not a non-null UInt64 column"),
            )
        })?;
    let doc_ids = column(DOC_ID_COL)?
        .as_primitive_opt::<UInt32Type>()
        .filter(|doc_ids| doc_ids.null_count() == 0)
        .ok_or_else(|| {
            Error::corrupt_file_named(
                file,
                format!("{DOC_ID_COL} is not a non-null UInt32 column"),
            )
        })?;
    Ok((keys, doc_ids))
}

#[derive(Debug, Clone)]
struct BandPageKey {
    page: u32,
}

/// `SIGNATURE_CHUNK_DOCS` consecutive documents of the signature table,
/// resident after a prewarm.
#[derive(Debug, DeepSizeOf)]
pub struct SignatureChunk {
    row_ids: Vec<u64>,
    /// `row_ids.len() * num_hashes` values.
    signatures: Vec<SignatureValue>,
}

#[derive(Debug, Clone)]
struct SignatureChunkKey {
    chunk: u32,
}

impl CacheKey for SignatureChunkKey {
    type ValueType = SignatureChunk;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("signature-chunk-{}", self.chunk).into()
    }

    fn type_name() -> &'static str {
        "MinHashLshSignatureChunk"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.scalar.minhashlsh.signature-chunk-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u32(self.chunk);
    }
}

/// Bytes read per prewarm chunk of band pages; bounds the transient memory of
/// a prewarm to roughly one chunk.
const PREWARM_CHUNK_BYTES: usize = 128 * 1024 * 1024;

impl CacheKey for BandPageKey {
    type ValueType = BandPage;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("band-page-{}", self.page).into()
    }

    fn type_name() -> &'static str {
        "MinHashLshBandPage"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.scalar.minhashlsh.band-page-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u32(self.page);
    }
}

/// A query's signature and band keys, computed once and searched against
/// every segment of a logical index.
#[derive(Debug, Clone)]
pub struct QuerySignature {
    signature: Vec<SignatureValue>,
    band_keys: Vec<u64>,
}

impl QuerySignature {
    /// Compute the query signature with `generator`, or `None` when the text
    /// has no tokens.
    pub fn compute(generator: &mut SignatureGenerator, text: &str) -> Option<Self> {
        let mut signature = vec![SignatureValue::MAX; generator.num_hashes()];
        if !generator.signature(text, &mut signature) {
            return None;
        }
        let mut band_keys = Vec::with_capacity(generator.num_bands());
        generator.band_keys(&signature, &mut band_keys);
        Some(Self {
            signature,
            band_keys,
        })
    }

    pub fn signature(&self) -> &[SignatureValue] {
        &self.signature
    }

    /// Whether a row signature shares at least one band with the query: the
    /// candidate test the index applies, so rows scored without the index
    /// (unindexed fragments) follow the same rule.
    pub fn shares_band(
        &self,
        generator: &SignatureGenerator,
        signature: &[SignatureValue],
        scratch: &mut Vec<u64>,
    ) -> bool {
        scratch.clear();
        generator.band_keys(signature, scratch);
        scratch
            .iter()
            .zip(&self.band_keys)
            .any(|(row, query)| row == query)
    }
}

/// One search hit: a row id and its Jaccard distance (`1 - estimated Jaccard`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinHashHit {
    pub row_id: u64,
    pub distance: f32,
}

impl Eq for MinHashHit {}

impl PartialOrd for MinHashHit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinHashHit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.row_id.cmp(&other.row_id))
    }
}

/// Bounded collection of the `limit` best (smallest distance) hits.
pub struct TopHits {
    limit: usize,
    heap: BinaryHeap<MinHashHit>,
}

impl TopHits {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            heap: BinaryHeap::with_capacity(limit.min(1024)),
        }
    }

    pub fn push(&mut self, hit: MinHashHit) {
        if self.limit == 0 {
            return;
        }
        if self.heap.len() < self.limit {
            self.heap.push(hit);
        } else if self.heap.peek().is_some_and(|worst| hit < *worst) {
            self.heap.pop();
            self.heap.push(hit);
        }
    }

    /// Hits ordered by ascending distance, ties broken by row id.
    pub fn into_sorted(self) -> Vec<MinHashHit> {
        self.heap.into_sorted_vec()
    }
}

impl Extend<MinHashHit> for TopHits {
    fn extend<I: IntoIterator<Item = MinHashHit>>(&mut self, hits: I) {
        for hit in hits {
            self.push(hit);
        }
    }
}

/// A single segment of a MinHash LSH index.
pub struct MinHashLshIndex {
    params: MinHashLshIndexParams,
    generator: SignatureGenerator,
    bands: Arc<dyn IndexReader>,
    signatures: Arc<dyn IndexReader>,
    page_rows: usize,
    /// Largest band key of each logical page of `bands.lance`.
    page_max_keys: Vec<u64>,
    num_docs: usize,
    /// Documents per resident signature chunk; see [`SIGNATURE_CHUNK_DOCS`].
    signature_chunk_docs: usize,
    cache: WeakLanceCache,
    frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
}

impl std::fmt::Debug for MinHashLshIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MinHashLshIndex")
            .field("params", &self.params)
            .field("num_docs", &self.num_docs)
            .field("num_buckets", &self.bands.num_rows())
            .field("num_pages", &self.page_max_keys.len())
            .finish()
    }
}

impl DeepSizeOf for MinHashLshIndex {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        // Pages and signature chunks live in the index cache, which sizes them
        // itself; the parameters are a few strings, not worth counting.
        self.page_max_keys.deep_size_of_children(context)
    }
}

fn metadata_value<'a>(metadata: &'a HashMap<String, String>, key: &str) -> Result<&'a String> {
    metadata.get(key).ok_or_else(|| {
        Error::corrupt_file_named(BANDS_FILENAME, format!("missing schema metadata key {key}"))
    })
}

impl MinHashLshIndex {
    pub async fn load(
        store: Arc<dyn IndexStore>,
        details: &prost_types::Any,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Arc<Self>> {
        let details = details.to_msg::<pb::MinHashLshIndexDetails>()?;
        let params = MinHashLshIndexParams::from_details(&details)?;
        let (bands, signatures) = futures::try_join!(
            store.open_index_file(BANDS_FILENAME),
            store.open_index_file(SIGNATURES_FILENAME)
        )?;

        let metadata = &bands.schema().metadata;
        let corrupt = |message: String| Error::corrupt_file_named(BANDS_FILENAME, message);
        let index_version: u32 = metadata_value(metadata, INDEX_VERSION_META_KEY)?
            .parse()
            .map_err(|err| corrupt(format!("invalid {INDEX_VERSION_META_KEY}: {err}")))?;
        if index_version > MINHASH_LSH_INDEX_VERSION {
            return Err(Error::not_supported(format!(
                "MinHash LSH index version {index_version} is newer than the supported version {MINHASH_LSH_INDEX_VERSION}"
            )));
        }
        let file_params =
            MinHashLshIndexParams::from_json(metadata_value(metadata, PARAMS_META_KEY)?)?;
        if file_params != params {
            return Err(corrupt(format!(
                "index file params {file_params:?} do not match index details {params:?}"
            )));
        }
        let page_rows: usize = metadata_value(metadata, PAGE_ROWS_META_KEY)?
            .parse()
            .map_err(|err| corrupt(format!("invalid {PAGE_ROWS_META_KEY}: {err}")))?;
        if page_rows == 0 {
            return Err(corrupt(format!("{PAGE_ROWS_META_KEY} must be positive")));
        }
        let num_docs: usize = metadata_value(metadata, NUM_DOCS_META_KEY)?
            .parse()
            .map_err(|err| corrupt(format!("invalid {NUM_DOCS_META_KEY}: {err}")))?;
        if num_docs != signatures.num_rows() {
            return Err(corrupt(format!(
                "bands file records {num_docs} documents but {SIGNATURES_FILENAME} has {} rows",
                signatures.num_rows()
            )));
        }
        let page_table_buffer: u32 = metadata_value(metadata, PAGE_TABLE_BUFFER_META_KEY)?
            .parse()
            .map_err(|err| corrupt(format!("invalid {PAGE_TABLE_BUFFER_META_KEY}: {err}")))?;
        let page_table = bands.read_global_buffer(page_table_buffer).await?;
        if page_table.len() % 8 != 0 {
            return Err(corrupt(format!(
                "page table buffer length {} is not a multiple of 8",
                page_table.len()
            )));
        }
        let page_max_keys: Vec<u64> = page_table
            .chunks_exact(8)
            .map(|bytes| {
                u64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ])
            })
            .collect();
        let expected_pages = bands.num_rows().div_ceil(page_rows);
        if page_max_keys.len() != expected_pages {
            return Err(corrupt(format!(
                "page table has {} entries but {} rows at {page_rows} rows per page need {expected_pages}",
                page_max_keys.len(),
                bands.num_rows()
            )));
        }

        let generator = SignatureGenerator::try_new(&params)?;
        Ok(Arc::new(Self {
            params,
            generator,
            bands,
            signatures,
            page_rows,
            page_max_keys,
            num_docs,
            signature_chunk_docs: SIGNATURE_CHUNK_DOCS,
            cache: WeakLanceCache::from(cache),
            frag_reuse_index,
        }))
    }

    #[cfg(test)]
    fn with_signature_chunk_docs(mut self, signature_chunk_docs: usize) -> Self {
        self.signature_chunk_docs = signature_chunk_docs.max(1);
        self
    }

    pub fn params(&self) -> &MinHashLshIndexParams {
        &self.params
    }

    /// Number of documents (rows with at least one token) in this segment.
    pub fn num_docs(&self) -> usize {
        self.num_docs
    }

    fn signature_source<'a>(&self, transform: RowIdTransform<'a>) -> SignatureSource<'a> {
        SignatureSource {
            reader: self.signatures.clone(),
            num_docs: self.num_docs,
            frag_reuse_index: self.frag_reuse_index.clone(),
            transform,
        }
    }

    fn created_index(&self, files: Vec<IndexFile>) -> Result<CreatedIndex> {
        Ok(CreatedIndex {
            index_details: self.params.details_any()?,
            index_version: MINHASH_LSH_INDEX_VERSION,
            files,
        })
    }

    fn page_range(&self, page: u32) -> Range<usize> {
        let start = page as usize * self.page_rows;
        start..(start + self.page_rows).min(self.bands.num_rows())
    }

    /// Signature and band keys of `text`, or `None` when the text has no
    /// tokens and therefore matches nothing.
    pub fn query_signature(&self, text: &str) -> Option<QuerySignature> {
        QuerySignature::compute(&mut self.generator.clone(), text)
    }

    /// Return the `limit` rows with the smallest Jaccard distance to `text`
    /// among rows selected by `mask`, ordered by ascending distance.
    pub async fn search_text(
        &self,
        text: &str,
        limit: usize,
        mask: &RowAddrMask,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<MinHashHit>> {
        match self.query_signature(text) {
            Some(query) => self.search_signature(&query, limit, mask, metrics).await,
            None => Ok(Vec::new()),
        }
    }

    /// Like [`Self::search_text`] for a signature computed by
    /// [`Self::query_signature`] on any segment with the same parameters.
    pub async fn search_signature(
        &self,
        query: &QuerySignature,
        limit: usize,
        mask: &RowAddrMask,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<MinHashHit>> {
        if query.signature.len() != self.params.num_hashes as usize {
            return Err(Error::invalid_input(format!(
                "query signature has {} values but the index uses num_hashes={}",
                query.signature.len(),
                self.params.num_hashes
            )));
        }
        if limit == 0 || self.num_docs == 0 {
            return Ok(Vec::new());
        }
        let candidates = self.collect_candidates(&query.band_keys, metrics).await?;
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        self.refine(&candidates, &query.signature, limit, mask, metrics)
            .await
    }

    /// Union the postings of every band key present in the bands file.
    async fn collect_candidates(
        &self,
        band_keys: &[u64],
        metrics: &dyn MetricsCollector,
    ) -> Result<RoaringBitmap> {
        let mut lookups: Vec<(u32, u64)> = band_keys
            .iter()
            .filter_map(|&key| {
                let page = self.page_max_keys.partition_point(|&max_key| max_key < key);
                // `page == len` means the key is larger than every stored key.
                (page < self.page_max_keys.len()).then_some((page as u32, key))
            })
            .collect();
        metrics.record_comparisons(band_keys.len());
        let mut candidates = RoaringBitmap::new();
        if lookups.is_empty() {
            return Ok(candidates);
        }
        lookups.sort_unstable();
        let mut pages: Vec<u32> = lookups.iter().map(|(page, _)| *page).collect();
        pages.dedup();
        let mut pages = self.load_pages(&pages, metrics).await?;
        let num_pages = self.page_max_keys.len();
        for (first_page, key) in lookups {
            let mut page_id = first_page;
            loop {
                if !pages.contains_key(&page_id) {
                    pages.extend(self.load_pages(&[page_id], metrics).await?);
                }
                let Some(page) = pages.get(&page_id) else {
                    return Err(Error::internal(format!(
                        "band page {page_id} was requested but not loaded"
                    )));
                };
                candidates.extend(page.members(key).iter().copied());
                // A bucket larger than the rest of its page continues on the next one
                if !page.continues_after(key) || page_id as usize + 1 >= num_pages {
                    break;
                }
                page_id += 1;
            }
        }
        if let Some(max_doc_id) = candidates.max()
            && max_doc_id as usize >= self.num_docs
        {
            return Err(Error::corrupt_file_named(
                BANDS_FILENAME,
                format!(
                    "posting references doc id {max_doc_id} but the segment has {} documents",
                    self.num_docs
                ),
            ));
        }
        Ok(candidates)
    }

    /// Fetch pages through the cache; every page missing from the cache is
    /// read with one scattered read.
    async fn load_pages(
        &self,
        pages: &[u32],
        metrics: &dyn MetricsCollector,
    ) -> Result<HashMap<u32, Arc<BandPage>>> {
        let mut loaded = HashMap::with_capacity(pages.len());
        let mut missing = Vec::new();
        for &page in pages {
            match self.cache.get_with_key(&BandPageKey { page }).await {
                Some(cached) => {
                    metrics.record_index_cache_hit();
                    loaded.insert(page, cached);
                }
                None => missing.push(page),
            }
        }
        if missing.is_empty() {
            return Ok(loaded);
        }
        metrics.record_index_cache_misses(missing.len());
        metrics.record_parts_loaded(missing.len());
        tracing::info!(
            target: TRACE_IO_EVENTS,
            r#type = IO_TYPE_LOAD_SCALAR_PART,
            index_type = "minhashlsh",
            num_parts = missing.len(),
        );
        let ranges: Vec<Range<usize>> = missing.iter().map(|&page| self.page_range(page)).collect();
        let batch = self.bands.read_ranges(&ranges, None).await?;
        let mut offset = 0;
        for (page, range) in missing.into_iter().zip(&ranges) {
            let page_batch = batch.slice(offset, range.len());
            offset += range.len();
            let band_page = Arc::new(BandPage::try_from_batch(&page_batch)?);
            self.cache
                .insert_with_key(&BandPageKey { page }, band_page.clone())
                .await;
            loaded.insert(page, band_page);
        }
        Ok(loaded)
    }

    async fn load_signature_chunk(&self, chunk: usize) -> Result<SignatureChunk> {
        let num_hashes = self.params.num_hashes as usize;
        let start = chunk * self.signature_chunk_docs;
        let end = (start + self.signature_chunk_docs).min(self.num_docs);
        let batch = self
            .signatures
            .read_range(start..end, Some(&[ROW_ID, SIGNATURE_COL]))
            .await?;
        let (row_ids, signatures) = signature_columns(&batch, num_hashes)?;
        if row_ids.len() != end - start {
            return Err(Error::corrupt_file_named(
                SIGNATURES_FILENAME,
                format!(
                    "read of docs {start}..{end} returned {} rows",
                    row_ids.len()
                ),
            ));
        }
        Ok(SignatureChunk {
            row_ids: row_ids.values().to_vec(),
            signatures: signatures.to_vec(),
        })
    }

    /// Score every candidate against the query signature and keep the best
    /// `limit` rows selected by `mask`.
    async fn refine(
        &self,
        candidates: &RoaringBitmap,
        query: &[SignatureValue],
        limit: usize,
        mask: &RowAddrMask,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<MinHashHit>> {
        let num_hashes = query.len();
        let mut hits = TopHits::new(limit);
        let mut score = |row_id: u64, signature: &[SignatureValue]| {
            let row_id = match &self.frag_reuse_index {
                Some(remapper) => match remapper.remap_row_id(row_id) {
                    Some(row_id) => row_id,
                    None => return,
                },
                None => row_id,
            };
            if !mask.selected(row_id) {
                return;
            }
            hits.push(MinHashHit {
                row_id,
                distance: 1.0 - estimate_jaccard(query, signature),
            });
        };
        metrics.record_comparisons(candidates.len() as usize);
        // Look up each signature chunk holding candidates once; candidates in
        // resident chunks are scored from memory, the rest are read.
        let chunk_docs = self.signature_chunk_docs;
        let mut resident: Vec<(usize, Arc<SignatureChunk>)> = Vec::new();
        let mut doc_ids = candidates.iter();
        let mut next = doc_ids.next();
        while let Some(doc_id) = next {
            let chunk = doc_id as usize / chunk_docs;
            // A chunk that is not resident is read, not loaded, so only a
            // hit is a cache event.
            if let Some(data) = self
                .cache
                .get_with_key(&SignatureChunkKey {
                    chunk: chunk as u32,
                })
                .await
            {
                metrics.record_index_cache_hit();
                resident.push((chunk, data));
            }
            let chunk_end = ((chunk + 1) * chunk_docs) as u64;
            next = doc_ids.find(|&doc_id| doc_id as u64 >= chunk_end);
        }
        // Score the candidates of resident chunks and drop them from the set
        // that still needs reading; both are per-chunk range operations.
        let mut unresolved;
        let candidates = if resident.is_empty() {
            candidates
        } else {
            unresolved = candidates.clone();
            for (chunk, data) in &resident {
                let first_doc = chunk * chunk_docs;
                let last_doc = (first_doc + chunk_docs - 1).min(u32::MAX as usize);
                let chunk_range = first_doc as u32..=last_doc as u32;
                for doc_id in candidates.range(chunk_range.clone()) {
                    let offset = doc_id as usize - first_doc;
                    let Some(row_id) = data.row_ids.get(offset) else {
                        return Err(Error::corrupt_file_named(
                            SIGNATURES_FILENAME,
                            format!("resident signature chunk {chunk} has no doc {doc_id}"),
                        ));
                    };
                    score(
                        *row_id,
                        &data.signatures[offset * num_hashes..(offset + 1) * num_hashes],
                    );
                }
                unresolved.remove_range(chunk_range);
            }
            &unresolved
        };
        if candidates.is_empty() {
            return Ok(hits.into_sorted());
        }
        let mut score_batch = |batch: &RecordBatch, keep: &mut dyn FnMut(usize) -> bool| {
            let (row_ids, signatures) = signature_columns(batch, num_hashes)?;
            for (index, (row_id, signature)) in row_ids
                .values()
                .iter()
                .zip(signatures.chunks_exact(num_hashes))
                .enumerate()
            {
                if keep(index) {
                    score(*row_id, signature);
                }
            }
            Ok::<_, Error>(())
        };
        metrics.record_part_load();
        tracing::info!(
            target: TRACE_IO_EVENTS,
            r#type = IO_TYPE_LOAD_SCALAR_PART,
            index_type = "minhashlsh",
            part_id = "signatures",
        );
        let projection = [ROW_ID, SIGNATURE_COL];
        if candidates.len().saturating_mul(100)
            <= (self.num_docs as u64).saturating_mul(SPARSE_REFINE_READ_PERCENT)
        {
            let mut doc_ids = candidates.iter();
            loop {
                let ranges = doc_id_ranges(doc_ids.by_ref().take(REFINE_READ_ROWS));
                if ranges.is_empty() {
                    break;
                }
                let expected_rows: usize = ranges.iter().map(|range| range.len()).sum();
                let batch = self
                    .signatures
                    .read_ranges(&ranges, Some(&projection))
                    .await?;
                if batch.num_rows() != expected_rows {
                    return Err(Error::corrupt_file_named(
                        SIGNATURES_FILENAME,
                        format!(
                            "scattered read returned {} rows for {expected_rows} candidates",
                            batch.num_rows()
                        ),
                    ));
                }
                score_batch(&batch, &mut |_| true)?;
            }
        } else {
            let mut stream = std::pin::pin!(scan_rows(
                self.signatures.clone(),
                self.num_docs,
                Some(projection.iter().map(|column| column.to_string()).collect()),
                SIGNATURE_SCAN_ROWS,
            ));
            // The scan and the candidate set are both ascending, so walk them
            // in lockstep instead of probing the bitmap for every row.
            let mut pending = candidates.iter().peekable();
            let mut first_doc = 0usize;
            while let Some(batch) = stream.try_next().await? {
                score_batch(&batch, &mut |index| {
                    let Ok(doc_id) = u32::try_from(first_doc + index) else {
                        return false;
                    };
                    while pending
                        .next_if(|&pending_doc| pending_doc < doc_id)
                        .is_some()
                    {}
                    pending.next_if_eq(&doc_id).is_some()
                })?;
                first_doc += batch.num_rows();
            }
        }
        Ok(hits.into_sorted())
    }
}

/// Unwrap a finished spill task, surfacing a panicked or cancelled task as an error.
fn joined_spill(
    spilled: std::result::Result<Result<SpilledRun>, tokio::task::JoinError>,
) -> Result<SpilledRun> {
    spilled.map_err(|err| Error::internal(format!("spill task failed: {err}")))?
}

/// Coalesce ascending doc ids into contiguous row ranges.
fn doc_id_ranges(doc_ids: impl Iterator<Item = u32>) -> Vec<Range<usize>> {
    let mut ranges: Vec<Range<usize>> = Vec::new();
    for doc_id in doc_ids {
        let doc_id = doc_id as usize;
        match ranges.last_mut() {
            Some(last) if last.end == doc_id => last.end += 1,
            _ => ranges.push(doc_id..doc_id + 1),
        }
    }
    ranges
}

/// Extract the row id column and the flattened signature values of a batch
/// read from `signatures.lance`.
fn signature_columns(
    batch: &RecordBatch,
    num_hashes: usize,
) -> Result<(&UInt64Array, &[SignatureValue])> {
    let corrupt = |message: String| Error::corrupt_file_named(SIGNATURES_FILENAME, message);
    let row_ids = batch
        .column_by_name(ROW_ID)
        .and_then(|column| column.as_primitive_opt::<UInt64Type>())
        .ok_or_else(|| corrupt(format!("missing UInt64 column {ROW_ID}")))?;
    let signatures = batch
        .column_by_name(SIGNATURE_COL)
        .and_then(|column| column.as_fixed_size_list_opt())
        .ok_or_else(|| corrupt(format!("missing FixedSizeList column {SIGNATURE_COL}")))?;
    if signatures.value_length() as usize != num_hashes {
        return Err(corrupt(format!(
            "signature width {} does not match num_hashes {num_hashes}",
            signatures.value_length()
        )));
    }
    let values = signatures
        .values()
        .as_primitive_opt::<UInt16Type>()
        .ok_or_else(|| corrupt(format!("{SIGNATURE_COL} values are not UInt16")))?;
    if values.len() != signatures.len() * num_hashes {
        return Err(corrupt(format!(
            "{SIGNATURE_COL} has {} values for {} rows of width {num_hashes}",
            values.len(),
            signatures.len()
        )));
    }
    Ok((row_ids, values.values()))
}

/// Stream rows `0..num_rows` of `reader` as `rows_per_read` batches with a
/// few reads in flight.
fn scan_rows(
    reader: Arc<dyn IndexReader>,
    num_rows: usize,
    projection: Option<Vec<String>>,
    rows_per_read: usize,
) -> impl Stream<Item = Result<RecordBatch>> + Send {
    let ranges: Vec<Range<usize>> = (0..num_rows)
        .step_by(rows_per_read.max(1))
        .map(|start| start..(start + rows_per_read).min(num_rows))
        .collect();
    futures::stream::iter(ranges)
        .map(move |range| {
            let reader = reader.clone();
            let projection = projection.clone();
            async move {
                let columns: Option<Vec<&str>> = projection
                    .as_ref()
                    .map(|columns| columns.iter().map(String::as_str).collect());
                reader.read_range(range, columns.as_deref()).await
            }
        })
        .buffered(SCAN_READ_CONCURRENCY)
}

#[async_trait]
impl Index for MinHashLshIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    async fn prewarm(&self) -> Result<()> {
        // Load what the cache can hold, signature chunks first: a scattered
        // signature read costs a query far more than a page read, and a chunk
        // is one allocation of `SIGNATURE_CHUNK_DOCS * (8 + 2 * num_hashes)`
        // bytes. A cache without capacity has nothing to warm.
        let capacity = self.cache.capacity_bytes();
        if capacity == Some(0) {
            return Ok(());
        }
        let mut budget = capacity;
        let mut fits = |bytes: usize, count: usize| -> usize {
            match budget {
                None => count,
                Some(remaining) => {
                    let affordable = (remaining / bytes.max(1)).min(count);
                    budget = Some(remaining - affordable * bytes);
                    affordable
                }
            }
        };

        let doc_bytes = std::mem::size_of::<u64>()
            + self.params.num_hashes as usize * std::mem::size_of::<SignatureValue>();
        let num_chunks = self.num_docs.div_ceil(self.signature_chunk_docs);
        let mut chunks_to_load = 0;
        while chunks_to_load < num_chunks {
            let start = chunks_to_load * self.signature_chunk_docs;
            let docs = self.signature_chunk_docs.min(self.num_docs - start);
            if fits(docs * doc_bytes, 1) == 0 {
                break;
            }
            self.cache
                .get_or_insert_with_key(
                    SignatureChunkKey {
                        chunk: chunks_to_load as u32,
                    },
                    || self.load_signature_chunk(chunks_to_load),
                )
                .await?;
            chunks_to_load += 1;
        }

        let num_rows = self.bands.num_rows();
        let num_pages = self.page_max_keys.len();
        let bytes_per_row = self
            .bands
            .file_size_bytes()
            .map(|bytes| (bytes as usize / num_rows.max(1)).max(1))
            .unwrap_or(64);
        let pages_to_load = fits(bytes_per_row * self.page_rows, num_pages);
        let pages_per_read = (PREWARM_CHUNK_BYTES / (bytes_per_row * self.page_rows)).max(1);
        for first_page in (0..pages_to_load).step_by(pages_per_read) {
            let last_page = (first_page + pages_per_read).min(pages_to_load);
            let rows = first_page * self.page_rows..(last_page * self.page_rows).min(num_rows);
            let batch = self.bands.read_range(rows, None).await?;
            for page in first_page..last_page {
                let range = self.page_range(page as u32);
                let page_batch =
                    batch.slice(range.start - first_page * self.page_rows, range.len());
                let band_page = Arc::new(BandPage::try_from_batch(&page_batch)?);
                self.cache
                    .insert_with_key(&BandPageKey { page: page as u32 }, band_page)
                    .await;
            }
        }

        if chunks_to_load < num_chunks || pages_to_load < num_pages {
            log::warn!(
                "MinHash LSH prewarm kept {chunks_to_load} of {num_chunks} signature chunks and {pages_to_load} of {num_pages} band pages: the index cache capacity is {} bytes",
                capacity.unwrap_or(0)
            );
        }
        Ok(())
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "type": "MinHashLsh",
            "num_docs": self.num_docs,
            "num_pages": self.page_max_keys.len(),
            "num_hashes": self.params.num_hashes,
            "num_bands": self.params.num_bands,
            "shingle_size": self.params.shingle_size,
            "signature_version": SIGNATURE_VERSION,
        }))
    }

    fn index_type(&self) -> IndexType {
        IndexType::MinHashLsh
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        // The signature table stores row ids, which no longer identify a
        // fragment once stable row ids are enabled; coverage is recorded in
        // the index metadata when a segment is committed.
        Err(Error::not_supported(
            "MinHash LSH indices do not recalculate fragment coverage from their files; the fragment bitmap of the index metadata is authoritative".to_string(),
        ))
    }
}

#[async_trait]
impl ScalarIndex for MinHashLshIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        _metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        Err(Error::not_supported(format!(
            "MinHash LSH index cannot evaluate scalar filter {query:?}; query it with a MinHash similarity search instead"
        )))
    }

    fn can_remap(&self) -> bool {
        true
    }

    async fn remap(
        &self,
        mapping: &RowAddrRemap,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        let files = MinHashLshIndexBuilder::try_new(self.params.clone())?
            .rebuild_from(
                vec![self.signature_source(RowIdTransform::Remap(mapping))],
                None,
                dest_store,
            )
            .await?;
        self.created_index(files)
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        old_data_filter: Option<OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        let transform = match &old_data_filter {
            Some(filter) => RowIdTransform::Filter(filter),
            None => RowIdTransform::Keep,
        };
        let files = MinHashLshIndexBuilder::try_new(self.params.clone())?
            .rebuild_from(
                vec![self.signature_source(transform)],
                Some(new_data),
                dest_store,
            )
            .await?;
        self.created_index(files)
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(TrainingCriteria::new(TrainingOrdering::None).with_row_id())
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        Ok(ScalarIndexParams::for_builtin(BuiltinIndexType::MinHashLsh).with_params(&self.params))
    }
}

/// Signatures and band keys of one input batch, for rows that have tokens.
struct SignedBatch {
    row_ids: Vec<u64>,
    /// `row_ids.len() * num_hashes` values.
    signatures: Vec<SignatureValue>,
    /// `row_ids.len() * num_bands` keys.
    band_keys: Vec<u64>,
}

fn sign_strings<'a>(
    generator: &mut SignatureGenerator,
    row_ids: &UInt64Array,
    values: impl Iterator<Item = Option<&'a str>>,
    signed: &mut SignedBatch,
) {
    let mut signature = vec![SignatureValue::MAX; generator.num_hashes()];
    for (row, value) in values.enumerate() {
        let Some(text) = value else {
            continue;
        };
        if generator.signature(text, &mut signature) {
            signed.row_ids.push(row_ids.value(row));
            signed.signatures.extend_from_slice(&signature);
            generator.band_keys(&signature, &mut signed.band_keys);
        }
    }
}

fn sign_batch(mut generator: SignatureGenerator, batch: RecordBatch) -> Result<SignedBatch> {
    let row_ids = batch
        .column_by_name(ROW_ID)
        .and_then(|column| column.as_primitive_opt::<UInt64Type>())
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "MinHash LSH training data must contain a non-null UInt64 column {ROW_ID}"
            ))
        })?;
    let values = batch.column_by_name(VALUE_COLUMN_NAME).ok_or_else(|| {
        Error::invalid_input(format!(
            "MinHash LSH training data must contain a column {VALUE_COLUMN_NAME}"
        ))
    })?;
    let num_rows = batch.num_rows();
    let mut signed = SignedBatch {
        row_ids: Vec::with_capacity(num_rows),
        signatures: Vec::with_capacity(num_rows * generator.num_hashes()),
        band_keys: Vec::with_capacity(num_rows * generator.num_bands()),
    };
    match values.data_type() {
        DataType::Utf8 => sign_strings(
            &mut generator,
            row_ids,
            values.as_string::<i32>().iter(),
            &mut signed,
        ),
        DataType::LargeUtf8 => sign_strings(
            &mut generator,
            row_ids,
            values.as_string::<i64>().iter(),
            &mut signed,
        ),
        DataType::Utf8View => sign_strings(
            &mut generator,
            row_ids,
            values.as_string_view().iter(),
            &mut signed,
        ),
        other => {
            return Err(Error::invalid_input(format!(
                "MinHash LSH index supports Utf8, LargeUtf8 and Utf8View columns, got {other}"
            )));
        }
    }
    Ok(signed)
}

/// Signed batches computed from text batches, `num_cpus` batches in flight.
fn text_signed_batches(
    data: SendableRecordBatchStream,
    generator: SignatureGenerator,
) -> impl Stream<Item = Result<SignedBatch>> + Send {
    data.map(move |batch| {
        let generator = generator.clone();
        async move {
            let batch = batch?;
            spawn_cpu(move || sign_batch(generator, batch)).await
        }
    })
    .buffered(get_num_compute_intensive_cpus())
}

/// How the row ids of an existing signature table are carried into a
/// rebuilt segment.
pub enum RowIdTransform<'a> {
    /// Keep every row with its row id.
    Keep,
    /// Keep only the rows the filter selects.
    Filter(&'a OldIndexDataFilter),
    /// Rewrite row ids through the mapping, dropping rows it deletes.
    Remap(&'a RowAddrRemap),
}

impl RowIdTransform<'_> {
    /// The row id each input row keeps, or `None` for rows to drop.
    ///
    /// Stored row ids predate any deferred compaction the segment was opened
    /// with, so `frag_reuse_index` first brings them into the current address
    /// space, which is the space the filter or mapping is expressed in.
    fn apply(
        &self,
        row_ids: &UInt64Array,
        frag_reuse_index: Option<&dyn RowIdRemapper>,
    ) -> Vec<Option<u64>> {
        let mut row_ids: Vec<Option<u64>> = row_ids
            .values()
            .iter()
            .map(|&row_id| match frag_reuse_index {
                Some(remapper) => remapper.remap_row_id(row_id),
                None => Some(row_id),
            })
            .collect();
        match self {
            Self::Keep => {}
            Self::Filter(filter) => {
                let keep = filter.filter_row_ids(&UInt64Array::from(row_ids.clone()));
                for (row_id, keep) in row_ids.iter_mut().zip(keep.iter()) {
                    if !keep.unwrap_or(false) {
                        *row_id = None;
                    }
                }
            }
            Self::Remap(mapping) => mapping.remap_in_place(&mut row_ids),
        }
        row_ids
    }
}

/// An existing signature table whose surviving rows are carried into a
/// rebuilt segment.
pub struct SignatureSource<'a> {
    pub reader: Arc<dyn IndexReader>,
    pub num_docs: usize,
    /// The deferred compactions the segment was opened with; see
    /// [`RowIdTransform::apply`].
    pub frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    pub transform: RowIdTransform<'a>,
}

type SignedBatchStream<'a> = Pin<Box<dyn Stream<Item = Result<SignedBatch>> + Send + 'a>>;

impl SignatureSource<'_> {
    async fn signed_batches(&self, generator: SignatureGenerator) -> Result<SignedBatchStream<'_>> {
        let num_hashes = generator.num_hashes();
        let batches = scan_rows(
            self.reader.clone(),
            self.num_docs,
            None,
            SIGNATURE_SCAN_ROWS,
        );
        let transform = &self.transform;
        let frag_reuse_index = self.frag_reuse_index.as_deref();
        let stream = batches
            .map(move |batch| {
                let generator = generator.clone();
                // Row id filtering needs the transform, which cannot move into
                // the CPU task, so it runs here; band keys are computed there.
                let prepared = batch.and_then(|batch| {
                    let (row_ids, _) = signature_columns(&batch, num_hashes)?;
                    let kept = transform.apply(row_ids, frag_reuse_index);
                    Ok((batch, kept))
                });
                async move {
                    let (batch, kept) = prepared?;
                    spawn_cpu(move || resign_batch(generator, batch, kept)).await
                }
            })
            .buffered(get_num_compute_intensive_cpus());
        Ok(Box::pin(stream))
    }
}

/// Build a signed batch from stored signatures, keeping the rows whose entry
/// in `row_ids` is `Some` (the row id to store).
fn resign_batch(
    generator: SignatureGenerator,
    batch: RecordBatch,
    row_ids: Vec<Option<u64>>,
) -> Result<SignedBatch> {
    let num_hashes = generator.num_hashes();
    let (_, signatures) = signature_columns(&batch, num_hashes)?;
    let kept = row_ids.iter().filter(|row_id| row_id.is_some()).count();
    let mut signed = SignedBatch {
        row_ids: Vec::with_capacity(kept),
        signatures: Vec::with_capacity(kept * num_hashes),
        band_keys: Vec::with_capacity(kept * generator.num_bands()),
    };
    for (row_id, signature) in row_ids.iter().zip(signatures.chunks_exact(num_hashes)) {
        let Some(row_id) = row_id else {
            continue;
        };
        signed.row_ids.push(*row_id);
        signed.signatures.extend_from_slice(signature);
        generator.band_keys(signature, &mut signed.band_keys);
    }
    Ok(signed)
}

fn signatures_batch(
    schema: &SchemaRef,
    row_ids: Vec<u64>,
    signatures: Vec<SignatureValue>,
    num_hashes: i32,
) -> Result<RecordBatch> {
    let values: ArrayRef = Arc::new(UInt16Array::from(signatures));
    let signatures = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::UInt16, false)),
        num_hashes,
        values,
        None,
    )?;
    Ok(RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(row_ids)) as ArrayRef,
            Arc::new(signatures) as ArrayRef,
        ],
    )?)
}

/// Streams ascending (band key, doc id) rows into `bands.lance` and records
/// the page table as rows are written.
struct BandsWriter {
    writer: Box<dyn IndexWriter>,
    page_rows: usize,
    rows_written: usize,
    /// Key of the most recently written row, for the trailing partial page.
    last_key: Option<u64>,
    page_max_keys: Vec<u64>,
}

impl BandsWriter {
    /// Write one batch of ascending records and extend the page table.
    async fn write_batch(&mut self, keys: Vec<u64>, doc_ids: Vec<u32>) -> Result<()> {
        // Only the rows that end a page matter; this runs on the serial
        // write path, so step to them instead of testing every row.
        let mut page_end = self.page_rows - self.rows_written % self.page_rows;
        while page_end <= keys.len() {
            self.page_max_keys.push(keys[page_end - 1]);
            page_end += self.page_rows;
        }
        self.rows_written += keys.len();
        if let Some(last) = keys.last() {
            self.last_key = Some(*last);
        }
        let batch = RecordBatch::try_new(
            BANDS_SCHEMA.clone(),
            vec![
                Arc::new(UInt64Array::from(keys)) as ArrayRef,
                Arc::new(UInt32Array::from(doc_ids)) as ArrayRef,
            ],
        )?;
        self.writer.write_record_batch(batch).await?;
        Ok(())
    }

    async fn finish(mut self, params: &MinHashLshIndexParams, num_docs: u64) -> Result<IndexFile> {
        if !self.rows_written.is_multiple_of(self.page_rows)
            && let Some(last_key) = self.last_key
        {
            self.page_max_keys.push(last_key);
        }
        // Always write at least one batch so the file carries the schema.
        if self.rows_written == 0 {
            self.write_batch(Vec::new(), Vec::new()).await?;
        }
        let mut page_table = Vec::with_capacity(self.page_max_keys.len() * 8);
        for key in &self.page_max_keys {
            page_table.extend_from_slice(&key.to_le_bytes());
        }
        let page_table_buffer = self
            .writer
            .add_global_buffer(Bytes::from(page_table))
            .await?;
        self.writer
            .finish_with_metadata(HashMap::from([
                (PARAMS_META_KEY.to_string(), serde_json::to_string(params)?),
                (
                    INDEX_VERSION_META_KEY.to_string(),
                    MINHASH_LSH_INDEX_VERSION.to_string(),
                ),
                (PAGE_ROWS_META_KEY.to_string(), self.page_rows.to_string()),
                (
                    PAGE_TABLE_BUFFER_META_KEY.to_string(),
                    page_table_buffer.to_string(),
                ),
                (NUM_DOCS_META_KEY.to_string(), num_docs.to_string()),
            ]))
            .await
    }
}

/// A sorted run of (band key, doc id) records in a temporary index file, with
/// the row at which each key-range partition starts (`SPILL_PARTITIONS + 1`
/// entries, the last one being the row count).
struct SpilledRun {
    name: String,
    partition_starts: Vec<usize>,
}

impl SpilledRun {
    /// Rows of this run that hold `partitions`.
    fn rows(&self, partitions: &Range<usize>) -> Range<usize> {
        self.partition_starts[partitions.start]..self.partition_starts[partitions.end]
    }
}

/// Key-range partition of a record: its band id, then the high bits of the
/// band hash, so partitions are contiguous in key order and every band's
/// records spread evenly over `partitions_per_band` of them.
fn spill_partition(key: u64, partitions_per_band: usize) -> usize {
    let band = (key >> 56) as usize;
    let hash = key & 0x00FF_FFFF_FFFF_FFFF;
    band * partitions_per_band + ((hash as u128 * partitions_per_band as u128) >> 56) as usize
}

/// Row at which each partition starts in a sorted run (`SPILL_PARTITIONS + 1`
/// entries, the last one being the row count).
fn partition_starts(records: &[(u64, u32)], partitions_per_band: usize) -> Vec<usize> {
    (0..=SPILL_PARTITIONS)
        .map(|partition| {
            records.partition_point(|record| {
                spill_partition(record.0, partitions_per_band) < partition
            })
        })
        .collect()
}

/// Split the partitions into contiguous groups of about `group_records`
/// records each (at least one partition per group).
fn merge_groups(runs: &[SpilledRun], group_records: usize) -> Vec<Range<usize>> {
    let mut groups = Vec::new();
    let mut start = 0;
    let mut records = 0usize;
    for partition in 0..SPILL_PARTITIONS {
        records += runs
            .iter()
            .map(|run| run.rows(&(partition..partition + 1)).len())
            .sum::<usize>();
        if records >= group_records {
            groups.push(start..partition + 1);
            start = partition + 1;
            records = 0;
        }
    }
    if start < SPILL_PARTITIONS {
        groups.push(start..SPILL_PARTITIONS);
    }
    groups
}

/// Gather the records of `partitions` from every run, sort them and split
/// them into the key and doc id columns.
async fn merge_group(
    runs: &[SpilledRun],
    readers: &[Arc<dyn IndexReader>],
    partitions: Range<usize>,
) -> Result<(Vec<u64>, Vec<u32>)> {
    let mut records: Vec<(u64, u32)> =
        Vec::with_capacity(runs.iter().map(|run| run.rows(&partitions).len()).sum());
    let reads: Vec<_> = runs
        .iter()
        .zip(readers)
        .filter_map(|(run, reader)| {
            let rows = run.rows(&partitions);
            if rows.is_empty() {
                return None;
            }
            let reader = reader.clone();
            Some(async move { reader.read_range(rows, None).await })
        })
        .collect();
    let mut batches = futures::stream::iter(reads).buffered(MERGE_GROUP_READ_CONCURRENCY);
    while let Some(batch) = batches.try_next().await? {
        let (keys, doc_ids) = band_columns(&batch, "bands spill run")?;
        records.extend(
            keys.values()
                .iter()
                .copied()
                .zip(doc_ids.values().iter().copied()),
        );
    }
    spawn_cpu(move || {
        records.sort_unstable();
        Ok::<_, Error>(records.into_iter().unzip())
    })
    .await
}

/// Builds the two index files from a stream of `value` (text) and `_rowid`
/// batches.
pub struct MinHashLshIndexBuilder {
    params: MinHashLshIndexParams,
    page_rows: usize,
    /// Maximum records held in memory before a sort run is spilled.
    sort_run_records: usize,
}

impl MinHashLshIndexBuilder {
    pub fn try_new(params: MinHashLshIndexParams) -> Result<Self> {
        params.validate()?;
        Ok(Self {
            params,
            page_rows: DEFAULT_PAGE_ROWS,
            sort_run_records: DEFAULT_SORT_RUN_BYTES / std::mem::size_of::<(u64, u32)>(),
        })
    }

    /// Rows per logical page of `bands.lance`; exposed so tests can force
    /// multi-page layouts with small inputs.
    pub fn with_page_rows(mut self, page_rows: usize) -> Result<Self> {
        if page_rows == 0 {
            return Err(Error::invalid_input(
                "MinHash LSH page_rows must be positive".to_string(),
            ));
        }
        self.page_rows = page_rows;
        Ok(self)
    }

    /// Records per in-memory sort run; exposed so tests can force spilling.
    pub fn with_sort_run_records(mut self, sort_run_records: usize) -> Result<Self> {
        if sort_run_records == 0 {
            return Err(Error::invalid_input(
                "MinHash LSH sort_run_records must be positive".to_string(),
            ));
        }
        self.sort_run_records = sort_run_records;
        Ok(self)
    }

    fn spill_filename(run: usize) -> String {
        format!("bands-spill-{run}.lance")
    }

    /// Sort one run and write it to a temporary index file, returning its name.
    async fn spill_run(
        store: Arc<dyn IndexStore>,
        run: usize,
        records: Vec<(u64, u32)>,
        partitions_per_band: usize,
    ) -> Result<SpilledRun> {
        let (records, partition_starts) = spawn_cpu(move || {
            let mut records = records;
            records.par_sort_unstable();
            let partition_starts = partition_starts(&records, partitions_per_band);
            Ok::<_, Error>((records, partition_starts))
        })
        .await?;
        let name = Self::spill_filename(run);
        let mut writer = store.new_index_file(&name, SPILL_SCHEMA.clone()).await?;
        for chunk in records.chunks(SPILL_BATCH_ROWS) {
            let keys = UInt64Array::from_iter_values(chunk.iter().map(|(key, _)| *key));
            let doc_ids = UInt32Array::from_iter_values(chunk.iter().map(|(_, doc_id)| *doc_id));
            let batch = RecordBatch::try_new(
                SPILL_SCHEMA.clone(),
                vec![Arc::new(keys) as ArrayRef, Arc::new(doc_ids) as ArrayRef],
            )?;
            writer.write_record_batch(batch).await?;
        }
        writer.finish().await?;
        Ok(SpilledRun {
            name,
            partition_starts,
        })
    }

    /// Build the index files from a stream of `value` (text) and `_rowid`
    /// batches.
    pub async fn train(
        &self,
        data: SendableRecordBatchStream,
        store: &dyn IndexStore,
    ) -> Result<Vec<IndexFile>> {
        let generator = SignatureGenerator::try_new(&self.params)?;
        self.train_signed(text_signed_batches(data, generator), store)
            .await
    }

    /// Build the index files from the surviving rows of existing signature
    /// tables plus optional new text batches, without tokenizing the existing
    /// rows again. Used by segment merges, updates and row id remaps.
    pub async fn rebuild_from(
        &self,
        sources: Vec<SignatureSource<'_>>,
        new_data: Option<SendableRecordBatchStream>,
        store: &dyn IndexStore,
    ) -> Result<Vec<IndexFile>> {
        let generator = SignatureGenerator::try_new(&self.params)?;
        let mut streams: Vec<SignedBatchStream<'_>> = Vec::with_capacity(sources.len() + 1);
        for source in &sources {
            streams.push(source.signed_batches(generator.clone()).await?);
        }
        if let Some(new_data) = new_data {
            streams.push(Box::pin(text_signed_batches(new_data, generator)));
        }
        self.train_signed(futures::stream::iter(streams).flatten(), store)
            .await
    }

    /// Write `signatures.lance` and `bands.lance` from signed batches,
    /// assigning doc ids in arrival order.
    async fn train_signed(
        &self,
        signed_batches: impl Stream<Item = Result<SignedBatch>> + Send,
        store: &dyn IndexStore,
    ) -> Result<Vec<IndexFile>> {
        let num_hashes = self.params.num_hashes as usize;
        let num_bands = self.params.num_bands as usize;
        let num_hashes_i32 = i32::try_from(num_hashes).map_err(|_| {
            Error::invalid_input(format!("num_hashes {num_hashes} exceeds i32::MAX"))
        })?;
        let signatures_schema = signatures_schema(num_hashes_i32);
        let mut signatures_writer = store
            .new_index_file(SIGNATURES_FILENAME, signatures_schema.clone())
            .await?;
        let mut signed_batches = std::pin::pin!(signed_batches);
        // The signature file is written by its own task so that awaiting a
        // write never stops the consumer loop below.
        let (batch_tx, mut batch_rx) = tokio::sync::mpsc::channel::<RecordBatch>(4);
        let writer_task = tokio::spawn(async move {
            while let Some(batch) = batch_rx.recv().await {
                signatures_writer.write_record_batch(batch).await?;
            }
            Ok::<_, Error>(signatures_writer)
        });

        // The signing stream (CPU-pool tasks behind `buffered`) only makes
        // progress while it is polled, and the consumer loop also waits on the
        // writer, on spills and on its own bookkeeping. A driver joined with
        // the loop keeps polling the stream whenever the loop is blocked, so
        // the CPU pool stays busy; the queue between them is bounded.
        let (signed_tx, mut signed_rx) =
            tokio::sync::mpsc::channel::<Result<SignedBatch>>(SIGNED_BATCH_QUEUE);
        let driver = async move {
            while let Some(signed) = signed_batches.next().await
                && signed_tx.send(signed).await.is_ok()
            {}
        };
        let started = std::time::Instant::now();
        let schema = &signatures_schema;
        let partitions_per_band = SPILL_PARTITIONS / num_bands;
        let consumer = async move {
            let mut run: Vec<(u64, u32)> = Vec::new();
            // Spills run as tasks so sorting and writing a run overlaps with
            // signing the next batches; at most MAX_INFLIGHT_SPILLS are in flight.
            let mut spills: FuturesOrdered<tokio::task::JoinHandle<Result<SpilledRun>>> =
                FuturesOrdered::new();
            let mut spilled_runs: Vec<SpilledRun> = Vec::new();
            // The run state is returned on failure too so spill files can be
            // cleaned up.
            let outcome = async {
                let mut num_docs: u64 = 0;
                let mut wrote_signatures = false;
                // Signed batches are coalesced into larger write batches.
                let mut pending_row_ids: Vec<u64> = Vec::new();
                let mut pending_signatures: Vec<SignatureValue> = Vec::new();
                while let Some(signed) = signed_rx.recv().await {
                    let signed = signed?;
                    if signed.row_ids.is_empty() {
                        continue;
                    }
                    for (doc_offset, keys) in signed.band_keys.chunks_exact(num_bands).enumerate() {
                        let doc_id = checked_doc_id(num_docs + doc_offset as u64)?;
                        run.extend(keys.iter().map(|&key| (key, doc_id)));
                        if run.len() >= self.sort_run_records {
                            let records = std::mem::replace(
                                &mut run,
                                Vec::with_capacity(self.sort_run_records),
                            );
                            let store = store.clone_arc();
                            let run_index = spilled_runs.len() + spills.len();
                            spills.push_back(tokio::spawn(Self::spill_run(
                                store,
                                run_index,
                                records,
                                partitions_per_band,
                            )));
                            if spills.len() >= MAX_INFLIGHT_SPILLS
                                && let Some(spilled) = spills.next().await
                            {
                                spilled_runs.push(joined_spill(spilled)?);
                            }
                        }
                    }
                    num_docs += signed.row_ids.len() as u64;
                    pending_row_ids.extend_from_slice(&signed.row_ids);
                    pending_signatures.extend_from_slice(&signed.signatures);
                    if pending_row_ids.len() < SIGNATURE_WRITE_BATCH_ROWS {
                        continue;
                    }
                    let batch = signatures_batch(
                        schema,
                        std::mem::take(&mut pending_row_ids),
                        std::mem::take(&mut pending_signatures),
                        num_hashes_i32,
                    )?;
                    batch_tx.send(batch).await.map_err(|_| {
                        Error::internal("signature writer task stopped before the build finished")
                    })?;
                    wrote_signatures = true;
                }
                if !pending_row_ids.is_empty() {
                    let batch = signatures_batch(
                        schema,
                        pending_row_ids,
                        pending_signatures,
                        num_hashes_i32,
                    )?;
                    batch_tx.send(batch).await.map_err(|_| {
                        Error::internal("signature writer task stopped before the build finished")
                    })?;
                    wrote_signatures = true;
                }
                Ok::<_, Error>((num_docs, wrote_signatures))
            }
            .await;
            // Dropping the sender ends the writer task; dropping the receiver
            // (with this block) ends the driver.
            drop(batch_tx);
            (outcome, run, spills, spilled_runs)
        };
        let ((), (outcome, mut run, mut spills, mut spilled_runs)) =
            futures::join!(driver, consumer);
        // The writer's own error explains a consumer that failed to send.
        let writer_result = writer_task
            .await
            .map_err(|err| Error::internal(format!("signature writer task failed: {err}")))
            .and_then(|result| result);
        let (mut signatures_writer, (num_docs, wrote_signatures)) = match (writer_result, outcome) {
            (Ok(writer), Ok(counts)) => (writer, counts),
            (Err(err), _) | (_, Err(err)) => {
                abandon_spills(store, spills, spilled_runs).await;
                return Err(err);
            }
        };
        log::debug!(
            "MinHash LSH build: signed {num_docs} docs and wrote {SIGNATURES_FILENAME} in {:?}",
            started.elapsed()
        );
        if !wrote_signatures {
            signatures_writer
                .write_record_batch(RecordBatch::new_empty(signatures_schema.clone()))
                .await?;
        }
        let signatures_file = signatures_writer
            .finish_with_metadata(HashMap::from([
                (
                    PARAMS_META_KEY.to_string(),
                    serde_json::to_string(&self.params)?,
                ),
                (
                    INDEX_VERSION_META_KEY.to_string(),
                    MINHASH_LSH_INDEX_VERSION.to_string(),
                ),
            ]))
            .await?;

        let runs = if spills.is_empty() && spilled_runs.is_empty() {
            let sorted = spawn_cpu(move || {
                run.par_sort_unstable();
                Ok::<_, Error>(run)
            })
            .await?;
            SortedRuns::Resident(sorted)
        } else {
            if !run.is_empty() {
                let run_index = spilled_runs.len() + spills.len();
                spills.push_back(tokio::spawn(Self::spill_run(
                    store.clone_arc(),
                    run_index,
                    run,
                    partitions_per_band,
                )));
            }
            while let Some(spilled) = spills.next().await {
                match joined_spill(spilled) {
                    Ok(run) => spilled_runs.push(run),
                    Err(err) => {
                        abandon_spills(store, spills, spilled_runs).await;
                        return Err(err);
                    }
                }
            }
            log::debug!(
                "MinHash LSH build: {} spill runs complete at {:?}",
                spilled_runs.len(),
                started.elapsed()
            );
            SortedRuns::Spilled(spilled_runs.into())
        };

        let bands_file = self.write_bands(store, &runs, num_docs).await;
        if let SortedRuns::Spilled(spilled_runs) = &runs {
            delete_spill_files(store, spilled_runs).await;
        }
        Ok(vec![signatures_file, bands_file?])
    }

    /// Write the sorted records to `bands.lance`; spilled runs are merged one
    /// key-range group at a time, several groups in flight as independent
    /// tasks, written in order.
    async fn write_bands(
        &self,
        store: &dyn IndexStore,
        runs: &SortedRuns,
        num_docs: u64,
    ) -> Result<IndexFile> {
        let mut bands = BandsWriter {
            writer: store
                .new_index_file(BANDS_FILENAME, BANDS_SCHEMA.clone())
                .await?,
            page_rows: self.page_rows,
            rows_written: 0,
            last_key: None,
            page_max_keys: Vec::new(),
        };
        match runs {
            SortedRuns::Resident(records) => {
                for chunk in records.chunks(BANDS_WRITE_BATCH_RECORDS) {
                    let (keys, doc_ids) = chunk.iter().copied().unzip();
                    bands.write_batch(keys, doc_ids).await?;
                }
            }
            SortedRuns::Spilled(runs) => {
                let readers = Arc::new(
                    try_join_all(runs.iter().map(|run| store.open_index_file(&run.name))).await?,
                );
                // Each group is its own task so gathering and sorting proceed
                // while this loop waits on the writer.
                let mut merged = futures::stream::iter(merge_groups(runs, MERGE_GROUP_RECORDS))
                    .map(|group| {
                        let runs = runs.clone();
                        let readers = readers.clone();
                        tokio::spawn(async move { merge_group(&runs, &readers, group).await })
                    })
                    .buffered(
                        (get_num_compute_intensive_cpus() / 4).clamp(1, MERGE_GROUPS_IN_FLIGHT),
                    );
                while let Some(joined) = merged.next().await {
                    let (keys, doc_ids) = joined
                        .map_err(|err| Error::internal(format!("merge task failed: {err}")))??;
                    if !keys.is_empty() {
                        bands.write_batch(keys, doc_ids).await?;
                    }
                }
            }
        }
        bands.finish(&self.params, num_docs).await
    }
}

/// Sorted (band key, doc id) records ready to be written.
enum SortedRuns {
    /// Everything fit in one in-memory run.
    Resident(Vec<(u64, u32)>),
    /// Several runs spilled to temporary files, merged while writing.
    Spilled(Arc<[SpilledRun]>),
}

/// Delete spill files, best effort: a leftover only wastes space.
async fn delete_spill_files(store: &dyn IndexStore, runs: &[SpilledRun]) {
    for run in runs {
        if let Err(err) = store.delete_index_file(&run.name).await {
            log::warn!(
                "failed to delete MinHash LSH spill file {}: {err}",
                run.name
            );
        }
    }
}

/// Wait for the in-flight spills of a failed build and delete every spill
/// file it wrote.
async fn abandon_spills(
    store: &dyn IndexStore,
    mut spills: FuturesOrdered<tokio::task::JoinHandle<Result<SpilledRun>>>,
    mut spilled_runs: Vec<SpilledRun>,
) {
    while let Some(spilled) = spills.next().await {
        if let Ok(Ok(run)) = spilled {
            spilled_runs.push(run);
        }
    }
    delete_spill_files(store, &spilled_runs).await;
}

/// Total documents of a merge, rejected when a segment could not address
/// them all with u32 doc ids.
fn checked_merged_doc_count(counts: impl IntoIterator<Item = usize>) -> Result<u64> {
    let capacity = u32::MAX as u64 + 1;
    let mut total: u64 = 0;
    for count in counts {
        total = total.checked_add(count as u64).ok_or_else(|| {
            Error::invalid_input("MinHash LSH segment document count overflowed".to_string())
        })?;
    }
    if total > capacity {
        return Err(Error::invalid_input(format!(
            "merging these MinHash LSH segments would produce {total} documents but a segment holds at most {capacity}; keep the segments separate"
        )));
    }
    Ok(total)
}

/// Merge segments into one new segment in `dest_store`, keeping only the rows
/// each segment's filter selects. Every segment must have been built with the
/// same parameters.
pub async fn merge_minhash_indices(
    sources: &[(&MinHashLshIndex, Option<&OldIndexDataFilter>)],
    dest_store: &dyn IndexStore,
) -> Result<CreatedIndex> {
    let Some((first, _)) = sources.first() else {
        return Err(Error::invalid_input(
            "merging MinHash LSH segments requires at least one segment".to_string(),
        ));
    };
    for (index, _) in sources {
        if index.params != first.params {
            return Err(Error::invalid_input(format!(
                "MinHash LSH segments were built with different parameters and cannot be merged: {:?} vs {:?}; rebuild the index with replace=true instead",
                first.params, index.params
            )));
        }
    }
    checked_merged_doc_count(sources.iter().map(|(index, _)| index.num_docs))?;
    let signature_sources = sources
        .iter()
        .map(|(index, filter)| {
            index.signature_source(match filter {
                Some(filter) => RowIdTransform::Filter(filter),
                None => RowIdTransform::Keep,
            })
        })
        .collect();
    let files = MinHashLshIndexBuilder::try_new(first.params.clone())?
        .rebuild_from(signature_sources, None, dest_store)
        .await?;
    first.created_index(files)
}

#[derive(Debug, Default)]
pub struct MinHashLshIndexPlugin;

#[derive(Debug)]
pub struct MinHashLshTrainingRequest {
    pub params: MinHashLshIndexParams,
    criteria: TrainingCriteria,
}

impl MinHashLshTrainingRequest {
    pub fn new(params: MinHashLshIndexParams) -> Self {
        Self {
            params,
            criteria: TrainingCriteria::new(TrainingOrdering::None).with_row_id(),
        }
    }
}

impl TrainingRequest for MinHashLshTrainingRequest {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

#[async_trait]
impl BasicTrainer for MinHashLshIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {}
            other => {
                return Err(Error::invalid_input(format!(
                    "MinHash LSH index can only be created on a string column (Utf8, LargeUtf8, Utf8View), field {} has type {other}",
                    field.name()
                )));
            }
        }
        let params = MinHashLshIndexParams::from_json(params)?;
        Ok(Box::new(MinHashLshTrainingRequest::new(params)))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        _fragment_ids: Option<Vec<u32>>,
        _progress: Arc<dyn crate::progress::IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        let request = (request as Box<dyn Any>)
            .downcast::<MinHashLshTrainingRequest>()
            .map_err(|_| {
                Error::invalid_input(
                    "must provide training request created by new_training_request".to_string(),
                )
            })?;
        let builder = MinHashLshIndexBuilder::try_new(request.params.clone())?;
        let files = builder.train(data, index_store).await?;
        Ok(CreatedIndex {
            index_details: request.params.details_any()?,
            index_version: MINHASH_LSH_INDEX_VERSION,
            files,
        })
    }
}

#[async_trait]
impl ScalarIndexPlugin for MinHashLshIndexPlugin {
    fn basic_trainer(&self) -> Option<&dyn BasicTrainer> {
        Some(self)
    }

    fn name(&self) -> &str {
        "MinHashLsh"
    }

    fn provides_exact_answer(&self) -> bool {
        false
    }

    fn version(&self) -> u32 {
        MINHASH_LSH_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        _index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        // Similarity search is planned by the MinHash search node, not by
        // filter expression rewriting.
        None
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(
            MinHashLshIndex::load(index_store, index_details, frag_reuse_index, cache).await?
                as Arc<dyn ScalarIndex>,
        )
    }

    fn validate_new_segments_against_existing(
        &self,
        existing: &[&prost_types::Any],
        incoming: &[&prost_types::Any],
    ) -> Result<()> {
        // Compare parsed parameters rather than bytes: a newer Lance may
        // serialize additional tokenizer keys with their default values, which
        // must not split a logical index. Byte-identical details are equal
        // without parsing, which keeps the per-query check on the search path
        // cheap.
        let mut all = existing.iter().chain(incoming);
        let Some(first) = all.next() else {
            return Ok(());
        };
        let mut reference = None;
        for details in all {
            if details.type_url == first.type_url && details.value == first.value {
                continue;
            }
            let reference = match &reference {
                Some(reference) => reference,
                None => reference.insert(MinHashLshIndexParams::from_index_details(first)?),
            };
            let params = MinHashLshIndexParams::from_index_details(details)?;
            if params != *reference {
                return Err(Error::invalid_input(format!(
                    "MinHash LSH index segments must share identical parameters (signatures are only comparable under the same hash functions and tokenizer); found {reference:?} and {params:?}"
                )));
            }
        }
        Ok(())
    }

    fn details_as_json(&self, details: &prost_types::Any) -> Result<serde_json::Value> {
        let details = details.to_msg::<pb::MinHashLshIndexDetails>()?;
        let params = MinHashLshIndexParams::from_details(&details)?;
        let mut value = serde_json::to_value(&params)?;
        if let Some(object) = value.as_object_mut() {
            object.insert(
                "signature_version".to_string(),
                serde_json::Value::from(details.signature_version),
            );
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use arrow_array::StringArray;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_io::object_store::ObjectStore;
    use lance_select::RowAddrTreeMap;
    use rstest::rstest;

    use super::*;
    use crate::metrics::{LocalMetricsCollector, NoOpMetricsCollector};
    use crate::scalar::lance_format::LanceIndexStore;

    fn test_store() -> (TempObjDir, Arc<LanceIndexStore>) {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        (tmpdir, store)
    }

    fn training_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Utf8, true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]))
    }

    /// Stream of `(text, row id)` rows split into batches of `batch_rows`.
    fn text_stream(rows: &[(Option<&str>, u64)], batch_rows: usize) -> SendableRecordBatchStream {
        let schema = training_schema();
        let batches: Vec<datafusion::error::Result<RecordBatch>> = rows
            .chunks(batch_rows.max(1))
            .map(|chunk| {
                let texts =
                    StringArray::from(chunk.iter().map(|(text, _)| *text).collect::<Vec<_>>());
                let row_ids =
                    UInt64Array::from(chunk.iter().map(|(_, id)| *id).collect::<Vec<_>>());
                Ok(RecordBatch::try_new(
                    training_schema(),
                    vec![Arc::new(texts) as ArrayRef, Arc::new(row_ids) as ArrayRef],
                )?)
            })
            .collect();
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(batches),
        ))
    }

    fn rows_from<'a>(texts: &[&'a str]) -> Vec<(Option<&'a str>, u64)> {
        texts
            .iter()
            .enumerate()
            .map(|(i, text)| (Some(*text), i as u64))
            .collect()
    }

    /// Build an index and load it against `cache`; the caller keeps the cache
    /// alive because the index only holds a weak reference to it.
    async fn build_and_load(
        store: &Arc<LanceIndexStore>,
        builder: MinHashLshIndexBuilder,
        rows: &[(Option<&str>, u64)],
        batch_rows: usize,
        cache: &LanceCache,
    ) -> Arc<MinHashLshIndex> {
        let params = builder.params.clone();
        builder
            .train(text_stream(rows, batch_rows), store.as_ref())
            .await
            .unwrap();
        MinHashLshIndex::load(store.clone(), &params.details_any().unwrap(), None, cache)
            .await
            .unwrap()
    }

    fn default_builder() -> MinHashLshIndexBuilder {
        MinHashLshIndexBuilder::try_new(MinHashLshIndexParams::default()).unwrap()
    }

    async fn search(index: &MinHashLshIndex, text: &str, limit: usize) -> Vec<MinHashHit> {
        index
            .search_text(text, limit, &RowAddrMask::all_rows(), &NoOpMetricsCollector)
            .await
            .unwrap()
    }

    /// Tiny deterministic generator for synthetic corpora.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }

        fn below(&mut self, bound: usize) -> usize {
            (self.next() % bound as u64) as usize
        }
    }

    fn random_words(rng: &mut Lcg, count: usize) -> Vec<String> {
        (0..count).map(|_| format!("w{}", rng.below(500))).collect()
    }

    /// Exact Jaccard similarity of the 3-shingle sets of two whitespace
    /// tokenized documents, independent of the index code under test.
    fn shingle_jaccard(a: &str, b: &str) -> f64 {
        let shingles = |text: &str| -> HashSet<String> {
            let tokens: Vec<&str> = text.split_whitespace().collect();
            tokens.windows(3).map(|w| w.join(" ")).collect()
        };
        let (a, b) = (shingles(a), shingles(b));
        a.intersection(&b).count() as f64 / a.union(&b).count() as f64
    }

    /// Base documents plus one near duplicate of each, with two replaced words.
    fn near_duplicate_corpus(num_docs: usize, words_per_doc: usize) -> (Vec<String>, Vec<String>) {
        let mut rng = Lcg(7);
        let mut bases = Vec::with_capacity(num_docs);
        let mut duplicates = Vec::with_capacity(num_docs);
        for _ in 0..num_docs {
            let words = random_words(&mut rng, words_per_doc);
            let mut edited = words.clone();
            for _ in 0..2 {
                let position = rng.below(words_per_doc);
                edited[position] = format!("x{}", rng.below(1000));
            }
            bases.push(words.join(" "));
            duplicates.push(edited.join(" "));
        }
        (bases, duplicates)
    }

    #[test]
    fn test_signature_is_deterministic_and_estimates_jaccard() {
        let params = MinHashLshIndexParams::default();
        let mut generator = SignatureGenerator::try_new(&params).unwrap();
        let (bases, duplicates) = near_duplicate_corpus(3, 80);
        let mut a = vec![0 as SignatureValue; params.num_hashes as usize];
        let mut b = vec![0 as SignatureValue; params.num_hashes as usize];
        assert!(generator.signature(&bases[0], &mut a));
        assert!(generator.signature(&bases[0], &mut b));
        assert_eq!(a, b, "same text must yield the same signature");
        assert_eq!(estimate_jaccard(&a, &b), 1.0);

        assert!(generator.signature(&duplicates[0], &mut b));
        let estimate = estimate_jaccard(&a, &b) as f64;
        let exact = shingle_jaccard(&bases[0], &duplicates[0]);
        assert!(
            (estimate - exact).abs() < 0.15,
            "estimate {estimate} too far from exact {exact}"
        );

        assert!(generator.signature(&bases[1], &mut b));
        assert!(
            estimate_jaccard(&a, &b) < 0.2,
            "unrelated documents must not look similar"
        );

        let mut keys = Vec::new();
        generator.band_keys(&a, &mut keys);
        assert_eq!(keys.len(), params.num_bands as usize);
        for (band, key) in keys.iter().enumerate() {
            assert_eq!(
                key >> 56,
                band as u64,
                "band id is the high byte of the key"
            );
        }
    }

    #[rstest]
    #[case::ten_thousand(10_000)]
    #[case::hundred_thousand(100_000)]
    fn test_disjoint_long_texts_stay_dissimilar(#[case] num_tokens: usize) {
        // The minimum over n shingles is a small number: a signature storing
        // its raw high bits had them all zero with probability e^(-n / 65536),
        // so two unrelated 100k-token texts estimated a Jaccard near 0.8 and
        // million-token texts looked identical.
        let params = MinHashLshIndexParams::default();
        let mut generator = SignatureGenerator::try_new(&params).unwrap();
        let mut rng = Lcg(5);
        let mut text = |prefix: char| {
            (0..num_tokens)
                .map(|_| format!("{prefix}{}", rng.next() % 1_000_000))
                .collect::<Vec<_>>()
                .join(" ")
        };
        let (a, b) = (text('a'), text('b'));
        let mut signature_a = vec![0 as SignatureValue; params.num_hashes as usize];
        let mut signature_b = signature_a.clone();
        assert!(generator.signature(&a, &mut signature_a));
        assert!(generator.signature(&b, &mut signature_b));
        let estimate = estimate_jaccard(&signature_a, &signature_b);
        assert!(estimate < 0.05, "disjoint texts estimated at {estimate}");
        let query = QuerySignature::compute(&mut generator, &a).unwrap();
        assert!(!query.shares_band(&generator, &signature_b, &mut Vec::new()));
        assert!(
            signature_a.iter().filter(|&&value| value == 0).count() < 4,
            "signature collapsed towards zero: {signature_a:?}"
        );
    }

    #[test]
    fn test_params_json_defaults_and_details_roundtrip() {
        let params = MinHashLshIndexParams::default();
        assert!(params.tokenizer.lower_case);
        assert!(!params.tokenizer.stem);
        assert!(!params.tokenizer.remove_stop_words);

        // Partial tokenizer objects layer over the MinHash defaults, not the
        // full text search defaults.
        let params = MinHashLshIndexParams::from_json(
            r#"{"num_hashes": 64, "num_bands": 8, "tokenizer": {"base_tokenizer": "whitespace"}}"#,
        )
        .unwrap();
        assert_eq!(params.num_hashes, 64);
        assert_eq!(params.tokenizer.base_tokenizer, "whitespace");
        assert!(!params.tokenizer.stem);
        assert!(!params.tokenizer.remove_stop_words);
        // Presets are accepted and explicit keys win over the dedup defaults
        let code =
            MinHashLshIndexParams::from_json(r#"{"tokenizer": {"analyzer": "code"}}"#).unwrap();
        assert_eq!(code.tokenizer.base_tokenizer, "code");
        assert!(!code.tokenizer.stem);
        let stemmed = MinHashLshIndexParams::from_json(r#"{"tokenizer": {"stem": true}}"#).unwrap();
        assert!(stemmed.tokenizer.stem);
        assert!(!stemmed.tokenizer.remove_stop_words);
        let Err(err) = MinHashLshIndexBuilder::try_new(MinHashLshIndexParams::default())
            .unwrap()
            .with_page_rows(0)
        else {
            panic!("page_rows=0 must be rejected");
        };
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");

        let details = params.to_details().unwrap();
        assert_eq!(details.signature_version, SIGNATURE_VERSION);
        assert_eq!(
            MinHashLshIndexParams::from_details(&details).unwrap(),
            params
        );

        let stale = pb::MinHashLshIndexDetails {
            signature_version: SIGNATURE_VERSION + 1,
            ..details
        };
        let err = MinHashLshIndexParams::from_details(&stale).unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }), "{err}");
        assert!(err.to_string().contains("signature_version"), "{err}");

        let json = MinHashLshIndexPlugin
            .details_as_json(&params.details_any().unwrap())
            .unwrap();
        assert_eq!(json["num_hashes"], 64);
        assert_eq!(json["signature_version"], SIGNATURE_VERSION);
        assert_eq!(json["tokenizer"]["base_tokenizer"], "whitespace");
    }

    #[rstest]
    #[case::not_divisible(r#"{"num_hashes": 100, "num_bands": 16}"#, "multiple of num_bands")]
    #[case::zero_hashes(r#"{"num_hashes": 0}"#, "num_hashes > 0")]
    #[case::zero_bands(r#"{"num_bands": 0}"#, "num_bands")]
    #[case::too_many_bands(r#"{"num_hashes": 512, "num_bands": 512}"#, "num_bands <= 256")]
    #[case::zero_shingle(r#"{"shingle_size": 0}"#, "shingle_size > 0")]
    #[case::custom_stop_words(
        r#"{"tokenizer": {"custom_stop_words": ["the"]}}"#,
        "custom_stop_words"
    )]
    #[case::lance_tokenizer(r#"{"tokenizer": {"lance_tokenizer": "json"}}"#, "lance_tokenizer")]
    #[case::unknown_key(r#"{"num_hash": 128}"#, "unknown field")]
    #[case::bad_tokenizer(
        r#"{"tokenizer": {"base_tokenizer": "nope"}}"#,
        "unknown base tokenizer"
    )]
    fn test_params_validation(#[case] json: &str, #[case] message: &str) {
        let err = MinHashLshIndexParams::from_json(json).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        assert!(err.to_string().contains(message), "{err}");
    }

    #[test]
    fn test_doc_id_limits_reject_u32_overflow() {
        assert_eq!(checked_doc_id(0).unwrap(), 0);
        assert_eq!(checked_doc_id(u32::MAX as u64).unwrap(), u32::MAX);
        let err = checked_doc_id(u32::MAX as u64 + 1).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        assert!(
            err.to_string().contains("create_index_uncommitted"),
            "{err}"
        );
        assert!(
            err.to_string().contains("commit_existing_index_segments"),
            "{err}"
        );

        assert_eq!(checked_merged_doc_count([3, 4]).unwrap(), 7);
        assert_eq!(
            checked_merged_doc_count([u32::MAX as usize, 1]).unwrap(),
            u32::MAX as u64 + 1
        );
        let err = checked_merged_doc_count([u32::MAX as usize, 2]).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        assert!(
            err.to_string().contains("keep the segments separate"),
            "{err}"
        );
    }

    #[test]
    fn test_spill_partitions_are_monotonic_and_bound_merge_groups() {
        let num_bands = 8usize;
        let partitions_per_band = SPILL_PARTITIONS / num_bands;
        let mut state = 3u64;
        // Three sorted runs of random 56-bit hashes over eight bands
        let runs: Vec<SpilledRun> = (0..3)
            .map(|run| {
                let mut records: Vec<(u64, u32)> = (0..100_000u32)
                    .map(|doc| {
                        let band = splitmix64(&mut state) % num_bands as u64;
                        let hash = splitmix64(&mut state) & 0x00FF_FFFF_FFFF_FFFF;
                        ((band << 56) | hash, doc)
                    })
                    .collect();
                records.sort_unstable();
                let partitions: Vec<usize> = records
                    .iter()
                    .map(|record| spill_partition(record.0, partitions_per_band))
                    .collect();
                assert!(partitions.is_sorted(), "partitions must follow key order");
                assert!(partitions.iter().all(|&p| p < SPILL_PARTITIONS));
                SpilledRun {
                    name: format!("run-{run}"),
                    partition_starts: partition_starts(&records, partitions_per_band),
                }
            })
            .collect();
        let partition_sizes: Vec<usize> = (0..SPILL_PARTITIONS)
            .map(|p| {
                runs.iter()
                    .map(|run| run.rows(&(p..p + 1)).len())
                    .sum::<usize>()
            })
            .collect();
        let largest_partition = *partition_sizes.iter().max().unwrap();
        // Records spread over (nearly) all partitions of every band: about
        // 4.6 per partition here, not 300_000 / (num_bands * 16).
        let non_empty = partition_sizes.iter().filter(|&&size| size > 0).count();
        assert!(
            non_empty > SPILL_PARTITIONS * 9 / 10,
            "{non_empty} partitions used"
        );
        assert!(
            largest_partition < 64,
            "largest partition {largest_partition}"
        );
        let group_records = 5_000;
        let groups = merge_groups(&runs, group_records);
        assert!(groups.len() > 10);
        assert_eq!(groups[0].start, 0);
        assert_eq!(groups.last().unwrap().end, SPILL_PARTITIONS);
        let mut total = 0;
        for (i, group) in groups.iter().enumerate() {
            if i > 0 {
                assert_eq!(group.start, groups[i - 1].end);
            }
            let records: usize = runs.iter().map(|run| run.rows(group).len()).sum();
            assert!(
                records < group_records + largest_partition,
                "group of {records}"
            );
            total += records;
        }
        assert_eq!(total, 300_000);
    }

    #[tokio::test]
    async fn test_bucket_spanning_pages_is_fully_collected() {
        // Ten identical texts share every bucket; with four rows per page each
        // bucket spans several pages and every member must still be found.
        let (bases, _) = near_duplicate_corpus(6, 40);
        let mut texts: Vec<&str> = vec![bases[0].as_str(); 10];
        texts.extend(bases[1..].iter().map(String::as_str));
        let rows = rows_from(&texts);
        let (_tmpdir, store) = test_store();
        let index = build_and_load(
            &store,
            default_builder().with_page_rows(4).unwrap(),
            &rows,
            64,
            &LanceCache::no_cache(),
        )
        .await;
        let hits = search(&index, &bases[0], 10).await;
        let mut row_ids: Vec<u64> = hits.iter().map(|hit| hit.row_id).collect();
        row_ids.sort_unstable();
        assert_eq!(row_ids, (0..10).collect::<Vec<u64>>());
        assert!(hits.iter().all(|hit| hit.distance == 0.0));
    }

    #[tokio::test]
    async fn test_build_load_search_roundtrip() {
        let (_tmpdir, store) = test_store();
        let texts = [
            "the quick brown fox jumps over the lazy dog near the river bank",
            "an entirely different sentence about databases and columnar storage formats",
            "the quick brown fox jumps over the lazy dog near the river bank today",
            "short",
            "the quick brown fox jumps over the lazy dog near the river bank",
        ];
        let rows: Vec<(Option<&str>, u64)> = texts
            .iter()
            .enumerate()
            .map(|(i, text)| (Some(*text), (i as u64 % 2) << 32 | i as u64))
            .collect();
        let index =
            build_and_load(&store, default_builder(), &rows, 2, &LanceCache::no_cache()).await;
        assert_eq!(index.num_docs(), 5);
        let err = index.calculate_included_frags().await.unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }), "{err}");
        let stats = index.statistics().unwrap();
        assert_eq!(stats["type"], "MinHashLsh");
        assert_eq!(stats["num_docs"], 5);
        assert_eq!(stats["num_bands"], 16);

        let hits = search(&index, texts[0], 10).await;
        assert!(hits.len() >= 3, "{hits:?}");
        assert_eq!(hits[0].row_id, rows[0].1);
        assert_eq!(hits[0].distance, 0.0);
        assert_eq!(hits[1].row_id, rows[4].1);
        assert_eq!(hits[1].distance, 0.0);
        assert_eq!(hits[2].row_id, rows[2].1);
        assert!(hits[2].distance > 0.0 && hits[2].distance < 0.4, "{hits:?}");
        assert!(
            hits.iter()
                .all(|hit| hit.row_id != rows[1].1 && hit.row_id != rows[3].1),
            "unrelated rows must not be candidates: {hits:?}"
        );

        assert_eq!(search(&index, texts[0], 1).await.len(), 1);
        assert!(search(&index, texts[0], 0).await.is_empty());
        assert!(
            search(&index, "nothing in common with the corpus at all", 10)
                .await
                .is_empty()
        );

        // Documents shorter than the shingle size are indexed as one shingle.
        let hits = search(&index, "short", 10).await;
        assert_eq!(
            hits,
            vec![MinHashHit {
                row_id: rows[3].1,
                distance: 0.0
            }]
        );

        let err = index
            .search(
                &crate::scalar::SargableQuery::IsNull(),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }), "{err}");
    }

    #[tokio::test]
    async fn test_recall_on_near_duplicates() {
        let (_tmpdir, store) = test_store();
        let (bases, duplicates) = near_duplicate_corpus(60, 80);
        let mut texts: Vec<&str> = bases.iter().map(String::as_str).collect();
        texts.extend(duplicates.iter().map(String::as_str));
        let rows = rows_from(&texts);
        let index = build_and_load(
            &store,
            default_builder(),
            &rows,
            17,
            &LanceCache::no_cache(),
        )
        .await;
        assert_eq!(index.num_docs(), 120);

        let mut pairs = 0;
        let mut found = 0;
        for (i, duplicate) in duplicates.iter().enumerate() {
            let exact = shingle_jaccard(&bases[i], duplicate);
            if exact < 0.8 {
                continue;
            }
            pairs += 1;
            let hits = search(&index, duplicate, 2).await;
            assert_eq!(hits[0].row_id, (bases.len() + i) as u64, "{hits:?}");
            assert_eq!(hits[0].distance, 0.0);
            if hits.iter().any(|hit| hit.row_id == i as u64) {
                found += 1;
            }
            for hit in &hits {
                assert!(
                    hit.row_id == i as u64 || hit.row_id == (bases.len() + i) as u64,
                    "unrelated row {} surfaced with distance {}",
                    hit.row_id,
                    hit.distance
                );
            }
        }
        assert!(
            pairs >= 50,
            "corpus should mostly contain pairs above 0.8, got {pairs}"
        );
        let recall = found as f64 / pairs as f64;
        assert!(
            recall >= 0.9,
            "recall {recall} below threshold ({found}/{pairs})"
        );
    }

    #[tokio::test]
    async fn test_null_and_empty_texts_are_not_indexed() {
        let (_tmpdir, store) = test_store();
        let rows = vec![
            (Some("alpha beta gamma delta"), 10u64),
            (None, 11),
            (Some(""), 12),
            (Some("   \n\t "), 13),
            (Some("alpha beta gamma delta"), 14),
        ];
        let index =
            build_and_load(&store, default_builder(), &rows, 2, &LanceCache::no_cache()).await;
        assert_eq!(index.num_docs(), 2);
        let hits = search(&index, "alpha beta gamma delta", 10).await;
        assert_eq!(
            hits,
            vec![
                MinHashHit {
                    row_id: 10,
                    distance: 0.0
                },
                MinHashHit {
                    row_id: 14,
                    distance: 0.0
                },
            ]
        );
        assert!(search(&index, "", 10).await.is_empty());
        assert!(search(&index, "   ", 10).await.is_empty());
    }

    #[rstest]
    #[case::no_rows(vec![])]
    #[case::only_empty_texts(vec![(None, 0), (Some(""), 1)])]
    #[tokio::test]
    async fn test_empty_index(#[case] rows: Vec<(Option<&str>, u64)>) {
        let (_tmpdir, store) = test_store();
        let index =
            build_and_load(&store, default_builder(), &rows, 4, &LanceCache::no_cache()).await;
        assert_eq!(index.num_docs(), 0);
        assert!(search(&index, "anything at all here", 5).await.is_empty());
        assert_eq!(index.statistics().unwrap()["num_pages"], 0);
    }

    #[tokio::test]
    async fn test_multi_page_and_spilled_build_matches_single_run() {
        let (bases, duplicates) = near_duplicate_corpus(20, 40);
        let mut texts: Vec<&str> = bases.iter().map(String::as_str).collect();
        texts.extend(duplicates.iter().map(String::as_str));
        let rows = rows_from(&texts);

        let (_single_dir, single_store) = test_store();
        let single = build_and_load(
            &single_store,
            default_builder(),
            &rows,
            64,
            &LanceCache::no_cache(),
        )
        .await;
        assert_eq!(single.statistics().unwrap()["num_pages"], 1);

        let (_split_dir, split_store) = test_store();
        let split = build_and_load(
            &split_store,
            default_builder()
                .with_page_rows(7)
                .unwrap()
                .with_sort_run_records(50)
                .unwrap(),
            &rows,
            3,
            &LanceCache::no_cache(),
        )
        .await;
        assert!(split.statistics().unwrap()["num_pages"].as_u64().unwrap() > 10);
        let files: Vec<String> = split_store
            .list_files_with_sizes()
            .await
            .unwrap()
            .into_iter()
            .map(|file| file.path)
            .collect();
        assert_eq!(files.len(), 2, "spill files must be deleted: {files:?}");
        assert!(files.contains(&BANDS_FILENAME.to_string()));
        assert!(files.contains(&SIGNATURES_FILENAME.to_string()));

        for text in texts.iter().step_by(5) {
            assert_eq!(
                search(&single, text, 5).await,
                search(&split, text, 5).await
            );
        }
    }

    #[test]
    fn test_band_page_copies_only_its_rows() {
        let keys = UInt64Array::from_iter_values((0..1000u64).map(|i| i / 2));
        let doc_ids = UInt32Array::from_iter_values(0..1000u32);
        let batch = RecordBatch::try_new(
            BANDS_SCHEMA.clone(),
            vec![Arc::new(keys) as ArrayRef, Arc::new(doc_ids) as ArrayRef],
        )
        .unwrap();
        let page = BandPage::try_from_batch(&batch.slice(10, 4)).unwrap();
        assert_eq!(page.keys, vec![5, 5, 6, 6]);
        assert_eq!(page.members(5), &[10, 11]);
        assert_eq!(page.members(6), &[12, 13]);
        assert!(page.members(7).is_empty());
        assert!(page.continues_after(6));
        assert!(!page.continues_after(5));
        // Four rows of twelve bytes, not the 1000-row batch
        assert!(page.deep_size_of() < 256, "{}", page.deep_size_of());
    }

    #[tokio::test]
    async fn test_band_pages_are_cached_and_mask_is_applied() {
        let (_tmpdir, store) = test_store();
        let (bases, duplicates) = near_duplicate_corpus(10, 40);
        let mut texts: Vec<&str> = bases.iter().map(String::as_str).collect();
        texts.extend(duplicates.iter().map(String::as_str));
        let rows = rows_from(&texts);
        let cache = LanceCache::with_capacity(64 * 1024 * 1024);
        let index = build_and_load(
            &store,
            default_builder().with_page_rows(16).unwrap(),
            &rows,
            8,
            &cache,
        )
        .await;

        let cold = LocalMetricsCollector::default();
        let hits = index
            .search_text(&duplicates[3], 3, &RowAddrMask::all_rows(), &cold)
            .await
            .unwrap();
        assert_eq!(hits[0].row_id, 13);
        assert!(cold.parts_loaded.load(std::sync::atomic::Ordering::Relaxed) > 1);
        assert!(cold.index_cache_misses() > 0);
        assert_eq!(cold.index_cache_hits(), 0);

        let warm = LocalMetricsCollector::default();
        let warm_hits = index
            .search_text(&duplicates[3], 3, &RowAddrMask::all_rows(), &warm)
            .await
            .unwrap();
        assert_eq!(warm_hits, hits);
        assert_eq!(warm.index_cache_misses(), 0);
        assert!(warm.index_cache_hits() > 0);
        // Only the signature refine read remains once the pages are cached.
        assert_eq!(
            warm.parts_loaded.load(std::sync::atomic::Ordering::Relaxed),
            1
        );

        let blocked = RowAddrMask::from_block(RowAddrTreeMap::from_iter([13u64]));
        let hits = index
            .search_text(&duplicates[3], 3, &blocked, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert!(hits.iter().all(|hit| hit.row_id != 13), "{hits:?}");
        assert_eq!(hits[0].row_id, 3);

        let allowed = RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([3u64]));
        let hits = index
            .search_text(&duplicates[3], 3, &allowed, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].row_id, 3);

        let nothing = RowAddrMask::allow_nothing();
        assert!(
            index
                .search_text(&duplicates[3], 3, &nothing, &NoOpMetricsCollector)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn test_dense_candidate_set_uses_sequential_refine() {
        let (_tmpdir, store) = test_store();
        let text = "every row in this segment carries exactly the same text content";
        let mut rows: Vec<(Option<&str>, u64)> = (0..40).map(|i| (Some(text), i)).collect();
        rows.push((Some("one row that is completely unrelated to the rest"), 40));
        let index =
            build_and_load(&store, default_builder(), &rows, 9, &LanceCache::no_cache()).await;

        let hits = search(&index, text, 5).await;
        assert_eq!(hits.len(), 5);
        assert!(hits.iter().all(|hit| hit.distance == 0.0));
        assert_eq!(
            hits.iter().map(|hit| hit.row_id).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4],
            "ties are broken by row id"
        );

        let hits = search(&index, text, 100).await;
        assert_eq!(hits.len(), 40);
        assert!(hits.iter().all(|hit| hit.row_id < 40));

        let mask = RowAddrMask::from_block(RowAddrTreeMap::from_iter(0..20u64));
        let hits = index
            .search_text(text, 100, &mask, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(hits.len(), 20);
        assert!(hits.iter().all(|hit| (20..40).contains(&hit.row_id)));
    }

    #[tokio::test]
    async fn test_load_rejects_details_that_do_not_match_the_files() {
        let (_tmpdir, store) = test_store();
        let rows = rows_from(&["some text to index for this test"]);
        default_builder()
            .train(text_stream(&rows, 1), store.as_ref())
            .await
            .unwrap();
        let other = MinHashLshIndexParams::from_json(r#"{"shingle_size": 4}"#).unwrap();
        let err = MinHashLshIndex::load(
            store.clone(),
            &other.details_any().unwrap(),
            None,
            &LanceCache::no_cache(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }), "{err}");
        assert!(
            err.to_string().contains("do not match index details"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn test_update_keeps_filtered_rows_and_adds_new_rows() {
        let (_tmpdir, store) = test_store();
        let (bases, duplicates) = near_duplicate_corpus(6, 120);
        let rows: Vec<(Option<&str>, u64)> = bases
            .iter()
            .enumerate()
            .map(|(i, text)| (Some(text.as_str()), i as u64))
            .collect();
        let index =
            build_and_load(&store, default_builder(), &rows, 4, &LanceCache::no_cache()).await;

        // Drop base rows 0 and 1, append the near duplicates
        let filter = OldIndexDataFilter::RowIds(RowAddrTreeMap::from_iter(2..6u64));
        let new_rows: Vec<(Option<&str>, u64)> = duplicates
            .iter()
            .enumerate()
            .map(|(i, text)| (Some(text.as_str()), 100 + i as u64))
            .collect();
        let (_dest_dir, dest_store) = test_store();
        let created = index
            .update(text_stream(&new_rows, 3), dest_store.as_ref(), Some(filter))
            .await
            .unwrap();
        assert_eq!(created.index_version, MINHASH_LSH_INDEX_VERSION);
        assert_eq!(created.files.len(), 2);
        let updated = MinHashLshIndex::load(
            dest_store.clone(),
            &created.index_details,
            None,
            &LanceCache::no_cache(),
        )
        .await
        .unwrap();
        assert_eq!(updated.num_docs(), 4 + 6);

        // Base row 0 is gone; its duplicate (100) is the best hit
        let hits = search(&updated, &bases[0], 3).await;
        assert_eq!(hits[0].row_id, 100);
        assert!(hits.iter().all(|hit| hit.row_id != 0 && hit.row_id != 1));
        // Base row 3 survived and still pairs with its duplicate
        let hits = search(&updated, &duplicates[3], 2).await;
        assert_eq!(
            hits[0],
            MinHashHit {
                row_id: 103,
                distance: 0.0
            }
        );
        assert_eq!(hits[1].row_id, 3);
    }

    /// Row id remapper of a segment opened after a deferred compaction.
    #[derive(Debug)]
    struct TestRemapper(HashMap<u64, Option<u64>>);

    impl RowIdRemapper for TestRemapper {
        fn remap_row_id(&self, row_id: u64) -> Option<u64> {
            self.0.get(&row_id).copied().unwrap_or(Some(row_id))
        }

        fn remap_row_addrs_tree_map(&self, _row_addrs: &RowAddrTreeMap) -> RowAddrTreeMap {
            unreachable!()
        }

        fn remap_row_ids_roaring_tree_map(
            &self,
            _row_ids: &roaring::RoaringTreemap,
        ) -> roaring::RoaringTreemap {
            unreachable!()
        }

        fn remap_row_ids_record_batch(
            &self,
            _batch: RecordBatch,
            _row_id_idx: usize,
        ) -> Result<RecordBatch> {
            unreachable!()
        }
    }

    async fn load_created(
        store: &Arc<LanceIndexStore>,
        created: &CreatedIndex,
    ) -> Arc<MinHashLshIndex> {
        MinHashLshIndex::load(
            store.clone(),
            &created.index_details,
            None,
            &LanceCache::no_cache(),
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn test_rebuilds_remap_rows_of_a_deferred_compaction() {
        // The segment stores fragment 0 addresses; a deferred compaction moved
        // its rows to fragment 1 and deleted row 2. Filters and mappings are
        // expressed in the compacted address space, so a rebuild applying
        // them to the stored addresses dropped every old row.
        let (_tmpdir, store) = test_store();
        let (bases, duplicates) = near_duplicate_corpus(4, 40);
        let rows = rows_from(&bases.iter().map(String::as_str).collect::<Vec<_>>());
        build_and_load(&store, default_builder(), &rows, 4, &LanceCache::no_cache()).await;
        let compacted = |row: u64| (1u64 << 32) | row;
        let remapper = TestRemapper(HashMap::from([
            (0u64, Some(compacted(0))),
            (1u64, Some(compacted(1))),
            (2u64, None),
            (3u64, Some(compacted(3))),
        ]));
        let index = MinHashLshIndex::load(
            store.clone(),
            &MinHashLshIndexParams::default().details_any().unwrap(),
            Some(Arc::new(remapper)),
            &LanceCache::no_cache(),
        )
        .await
        .unwrap();
        assert_eq!(search(&index, &bases[0], 1).await[0].row_id, compacted(0));

        // Update: keep the compacted fragment, add a duplicate of row 3
        let filter = OldIndexDataFilter::Fragments {
            to_keep: RoaringBitmap::from_iter([1u32]),
            to_remove: RoaringBitmap::from_iter([0u32]),
        };
        let new_rows = [(Some(duplicates[3].as_str()), 2u64 << 32)];
        let (_dir, dest_store) = test_store();
        let created = index
            .update(
                text_stream(&new_rows, 1),
                dest_store.as_ref(),
                Some(filter.clone()),
            )
            .await
            .unwrap();
        let updated = load_created(&dest_store, &created).await;
        assert_eq!(updated.num_docs(), 3 + 1);
        assert_eq!(search(&updated, &bases[0], 1).await[0].row_id, compacted(0));
        assert!(search(&updated, &bases[2], 5).await.is_empty());
        let hits = search(&updated, &duplicates[3], 2).await;
        assert_eq!(hits[0].row_id, 2u64 << 32);
        assert_eq!(hits[1].row_id, compacted(3));

        // Merge with the same filter
        let (_dir, dest_store) = test_store();
        let created = merge_minhash_indices(&[(&index, Some(&filter))], dest_store.as_ref())
            .await
            .unwrap();
        let merged = load_created(&dest_store, &created).await;
        assert_eq!(merged.num_docs(), 3);
        assert_eq!(search(&merged, &bases[3], 1).await[0].row_id, compacted(3));
        assert!(search(&merged, &bases[2], 5).await.is_empty());

        // An eager remap after the deferred one composes both
        let mapping = RowAddrRemap::direct(HashMap::from([
            (compacted(0), Some(9u64)),
            (compacted(1), None),
        ]));
        let (_dir, dest_store) = test_store();
        let created = index.remap(&mapping, dest_store.as_ref()).await.unwrap();
        let remapped = load_created(&dest_store, &created).await;
        assert_eq!(remapped.num_docs(), 2);
        assert_eq!(search(&remapped, &bases[0], 1).await[0].row_id, 9);
        assert!(search(&remapped, &bases[1], 5).await.is_empty());
        assert_eq!(
            search(&remapped, &bases[3], 1).await[0].row_id,
            compacted(3)
        );
    }

    #[tokio::test]
    async fn test_remap_rewrites_and_drops_row_ids() {
        let (_tmpdir, store) = test_store();
        let (bases, _) = near_duplicate_corpus(4, 40);
        let rows = rows_from(&bases.iter().map(String::as_str).collect::<Vec<_>>());
        let index =
            build_and_load(&store, default_builder(), &rows, 4, &LanceCache::no_cache()).await;
        assert!(index.can_remap());

        let mapping = RowAddrRemap::direct(HashMap::from([
            (0u64, Some(7u64)),
            (1u64, None),
            // row 2 is absent from the mapping and keeps its id
            (3u64, Some(9u64)),
        ]));
        let (_dest_dir, dest_store) = test_store();
        let created = index.remap(&mapping, dest_store.as_ref()).await.unwrap();
        let remapped = MinHashLshIndex::load(
            dest_store.clone(),
            &created.index_details,
            None,
            &LanceCache::no_cache(),
        )
        .await
        .unwrap();
        assert_eq!(remapped.num_docs(), 3);
        assert_eq!(
            search(&remapped, &bases[0], 1).await,
            vec![MinHashHit {
                row_id: 7,
                distance: 0.0
            }]
        );
        assert!(search(&remapped, &bases[1], 5).await.is_empty());
        assert_eq!(
            search(&remapped, &bases[2], 1).await,
            vec![MinHashHit {
                row_id: 2,
                distance: 0.0
            }]
        );
        assert_eq!(
            search(&remapped, &bases[3], 1).await,
            vec![MinHashHit {
                row_id: 9,
                distance: 0.0
            }]
        );
    }

    #[tokio::test]
    async fn test_merge_segments() {
        let (bases, duplicates) = near_duplicate_corpus(8, 120);
        let (_dir_a, store_a) = test_store();
        let rows_a: Vec<(Option<&str>, u64)> = bases
            .iter()
            .enumerate()
            .map(|(i, text)| (Some(text.as_str()), i as u64))
            .collect();
        let segment_a = build_and_load(
            &store_a,
            default_builder(),
            &rows_a,
            3,
            &LanceCache::no_cache(),
        )
        .await;
        let (_dir_b, store_b) = test_store();
        let rows_b: Vec<(Option<&str>, u64)> = duplicates
            .iter()
            .enumerate()
            .map(|(i, text)| (Some(text.as_str()), (1u64 << 32) | i as u64))
            .collect();
        let segment_b = build_and_load(
            &store_b,
            default_builder(),
            &rows_b,
            3,
            &LanceCache::no_cache(),
        )
        .await;

        // Segment b keeps only its first four rows
        let filter_b = OldIndexDataFilter::RowIds(RowAddrTreeMap::from_iter(
            (0..4u64).map(|i| (1u64 << 32) | i),
        ));
        let (_dir_merged, merged_store) = test_store();
        let created = merge_minhash_indices(
            &[(&segment_a, None), (&segment_b, Some(&filter_b))],
            merged_store.as_ref(),
        )
        .await
        .unwrap();
        let merged = MinHashLshIndex::load(
            merged_store.clone(),
            &created.index_details,
            None,
            &LanceCache::no_cache(),
        )
        .await
        .unwrap();
        assert_eq!(merged.num_docs(), 8 + 4);
        for (i, duplicate) in duplicates.iter().enumerate() {
            let hits = search(&merged, duplicate, 2).await;
            if i < 4 {
                assert_eq!(
                    hits[0],
                    MinHashHit {
                        row_id: (1u64 << 32) | i as u64,
                        distance: 0.0
                    }
                );
                assert_eq!(hits[1].row_id, i as u64);
            } else {
                assert_eq!(hits[0].row_id, i as u64, "{hits:?}");
                assert!(hits.iter().all(|hit| hit.row_id >> 32 == 0), "{hits:?}");
            }
        }

        // Segments built with different parameters cannot be merged
        let (_dir_c, store_c) = test_store();
        let other = MinHashLshIndexBuilder::try_new(
            MinHashLshIndexParams::from_json(r#"{"shingle_size": 4}"#).unwrap(),
        )
        .unwrap();
        let segment_c = build_and_load(&store_c, other, &rows_a, 3, &LanceCache::no_cache()).await;
        let Err(err) = merge_minhash_indices(
            &[(&segment_a, None), (&segment_c, None)],
            merged_store.as_ref(),
        )
        .await
        else {
            panic!("segments with different parameters must not merge");
        };
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        assert!(err.to_string().contains("different parameters"), "{err}");
    }

    #[test]
    fn test_plugin_validates_segment_parameter_drift() {
        let plugin = MinHashLshIndexPlugin;
        let a = MinHashLshIndexParams::default().details_any().unwrap();
        let b = MinHashLshIndexParams::from_json(r#"{"shingle_size": 4}"#)
            .unwrap()
            .details_any()
            .unwrap();
        plugin
            .validate_new_segments_against_existing(&[&a], &[&a, &a])
            .unwrap();
        plugin
            .validate_new_segments_against_existing(&[], &[])
            .unwrap();
        let err = plugin
            .validate_new_segments_against_existing(&[&a], &[&b])
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        assert!(err.to_string().contains("identical parameters"), "{err}");
        let err = plugin
            .validate_new_segments_against_existing(&[], &[&a, &b])
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    }

    #[tokio::test]
    async fn test_prewarm_makes_search_io_free() {
        let (_tmpdir, store) = test_store();
        let (bases, duplicates) = near_duplicate_corpus(10, 120);
        let mut texts: Vec<&str> = bases.iter().map(String::as_str).collect();
        texts.extend(duplicates.iter().map(String::as_str));
        let rows = rows_from(&texts);
        let cache = LanceCache::with_capacity(64 * 1024 * 1024);
        let index = build_and_load(
            &store,
            default_builder().with_page_rows(16).unwrap(),
            &rows,
            8,
            &cache,
        )
        .await;
        index.prewarm().await.unwrap();

        let metrics = LocalMetricsCollector::default();
        let hits = index
            .search_text(&duplicates[5], 2, &RowAddrMask::all_rows(), &metrics)
            .await
            .unwrap();
        assert_eq!(hits[0].row_id, 15);
        assert_eq!(hits[1].row_id, 5);
        assert_eq!(
            metrics
                .parts_loaded
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert_eq!(metrics.index_cache_misses(), 0);
        assert!(metrics.index_cache_hits() > 0);

        // A blocked row is still excluded on the resident path
        let blocked = RowAddrMask::from_block(RowAddrTreeMap::from_iter([15u64]));
        let hits = index
            .search_text(&duplicates[5], 2, &blocked, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(hits[0].row_id, 5);
        assert!(hits.iter().all(|hit| hit.row_id != 15));
    }

    #[tokio::test]
    async fn test_prewarm_keeps_the_signature_chunks_that_fit() {
        let (_tmpdir, store) = test_store();
        let (bases, duplicates) = near_duplicate_corpus(10, 120);
        let mut texts: Vec<&str> = bases.iter().map(String::as_str).collect();
        texts.extend(duplicates.iter().map(String::as_str));
        let rows = rows_from(&texts);
        // Chunks of 8 docs: docs 0..8, 8..16 and 16..20. Room for two chunks
        // (8 docs x (8 + 2 x 128) bytes each, plus cache key overhead).
        let chunk_bytes = 8 * (8 + 2 * 128);
        let cache = LanceCache::with_capacity(2 * chunk_bytes + 256);
        let index = build_and_load(
            &store,
            default_builder().with_page_rows(16).unwrap(),
            &rows,
            8,
            &cache,
        )
        .await;
        let index = Arc::try_unwrap(index)
            .expect("sole owner")
            .with_signature_chunk_docs(8);
        index.prewarm().await.unwrap();
        assert!(
            cache
                .get_with_key(&SignatureChunkKey { chunk: 1 })
                .await
                .is_some()
        );
        assert!(
            cache
                .get_with_key(&SignatureChunkKey { chunk: 2 })
                .await
                .is_none()
        );

        // Refine directly so page loads do not count: docs 2 and 12 sit in
        // resident chunks and are scored from memory; doc 17 is in the chunk
        // that did not fit and is read.
        let expected = search(&index, &duplicates[2], 2).await;
        assert_eq!(expected[0].row_id, 12);
        assert_eq!(expected[1].row_id, 2);
        let query = index.query_signature(&duplicates[2]).unwrap();
        let metrics = LocalMetricsCollector::default();
        let hits = index
            .refine(
                &RoaringBitmap::from_iter([2u32, 12]),
                query.signature(),
                2,
                &RowAddrMask::all_rows(),
                &metrics,
            )
            .await
            .unwrap();
        assert_eq!(hits, expected);
        assert_eq!(
            metrics
                .parts_loaded
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert_eq!(metrics.index_cache_hits(), 2);

        let expected = search(&index, &duplicates[7], 2).await;
        assert_eq!(expected[0].row_id, 17);
        assert_eq!(expected[1].row_id, 7);
        let query = index.query_signature(&duplicates[7]).unwrap();
        let metrics = LocalMetricsCollector::default();
        let hits = index
            .refine(
                &RoaringBitmap::from_iter([7u32, 17]),
                query.signature(),
                2,
                &RowAddrMask::all_rows(),
                &metrics,
            )
            .await
            .unwrap();
        assert_eq!(hits, expected);
        assert_eq!(
            metrics
                .parts_loaded
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(metrics.index_cache_hits(), 1);
    }

    #[test]
    fn test_plugin_rejects_non_string_fields() {
        let plugin = MinHashLshIndexPlugin;
        let Err(err) = plugin.new_training_request("{}", &Field::new("id", DataType::Int64, false))
        else {
            panic!("non-string field must be rejected");
        };
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        assert!(err.to_string().contains("string column"), "{err}");
        assert!(
            plugin
                .new_training_request("", &Field::new("text", DataType::LargeUtf8, true))
                .is_ok()
        );
        assert!(!plugin.provides_exact_answer());
        assert_eq!(plugin.name(), "MinHashLsh");
    }
}
