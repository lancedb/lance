// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Abstract scalar index traits and types

use arrow_array::{BooleanArray, RecordBatch, UInt64Array};
use arrow_schema::Schema;
use async_trait::async_trait;
use bytes::Bytes;
use datafusion::physical_plan::SendableRecordBatchStream;
use deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use lance_io::stream::{RecordBatchStream, RecordBatchStreamAdapter};
use lance_select::{NullableRowAddrSet, RowAddrTreeMap, RowSetOps};
use roaring::RoaringBitmap;
use serde::Serialize;
use std::collections::HashMap;
use std::pin::Pin;
use std::{any::Any, sync::Arc};

/// Metadata about a single file within an index.
#[derive(Debug, Clone, PartialEq, deepsize::DeepSizeOf)]
pub struct IndexFile {
    /// Path relative to the index directory
    pub path: String,
    /// Size of the file in bytes
    pub size_bytes: u64,
}

use crate::index::{Index, IndexParams};
use crate::metrics::MetricsCollector;
use crate::scalar::registry::TrainingCriteria;

pub mod expression;
pub mod registry;

pub const LANCE_SCALAR_INDEX: &str = "__lance_scalar_index";

/// Summary of a completed index file write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexWriteSummary {
    /// The final size of the index file in bytes.
    pub size_bytes: u64,
}

/// Builtin index types supported by the Lance library
///
/// This is primarily for convenience to avoid a bunch of string
/// constants and provide some auto-complete.  This type should not
/// be used in the manifest as plugins cannot add new entries.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub enum BuiltinIndexType {
    BTree,
    Bitmap,
    LabelList,
    NGram,
    ZoneMap,
    BloomFilter,
    RTree,
    Inverted,
    FMIndex,
}

impl BuiltinIndexType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::BTree => "btree",
            Self::Bitmap => "bitmap",
            Self::LabelList => "labellist",
            Self::NGram => "ngram",
            Self::ZoneMap => "zonemap",
            Self::Inverted => "inverted",
            Self::BloomFilter => "bloomfilter",
            Self::RTree => "rtree",
            Self::FMIndex => "fmindex",
        }
    }
}

use crate::index::IndexType;

impl TryFrom<IndexType> for BuiltinIndexType {
    type Error = Error;

    fn try_from(value: IndexType) -> Result<Self> {
        match value {
            IndexType::BTree => Ok(Self::BTree),
            IndexType::Bitmap => Ok(Self::Bitmap),
            IndexType::LabelList => Ok(Self::LabelList),
            IndexType::NGram => Ok(Self::NGram),
            IndexType::ZoneMap => Ok(Self::ZoneMap),
            IndexType::Inverted => Ok(Self::Inverted),
            IndexType::BloomFilter => Ok(Self::BloomFilter),
            IndexType::RTree => Ok(Self::RTree),
            IndexType::FMIndex => Ok(Self::FMIndex),
            _ => Err(Error::index("Invalid index type".to_string())),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScalarIndexParams {
    /// The type of index to create
    ///
    /// Plugins may add additional index types.  Index type lookup is case-insensitive.
    pub index_type: String,
    /// The parameters to train the index
    ///
    /// This should be a JSON string.  The contents of the JSON string will be specific to the
    /// index type.  If not set, then default parameters will be used for the index type.
    pub params: Option<String>,
}

impl Default for ScalarIndexParams {
    fn default() -> Self {
        Self {
            index_type: BuiltinIndexType::BTree.as_str().to_string(),
            params: None,
        }
    }
}

impl ScalarIndexParams {
    /// Creates a new ScalarIndexParams from one of the builtin index types
    pub fn for_builtin(index_type: BuiltinIndexType) -> Self {
        Self {
            index_type: index_type.as_str().to_string(),
            params: None,
        }
    }

    /// Create a new ScalarIndexParams with the given index type
    pub fn new(index_type: String) -> Self {
        Self {
            index_type,
            params: None,
        }
    }

    /// Set the parameters for the index
    pub fn with_params<ParamsType: Serialize>(mut self, params: &ParamsType) -> Self {
        self.params = Some(serde_json::to_string(params).unwrap());
        self
    }
}

impl IndexParams for ScalarIndexParams {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn index_name(&self) -> &str {
        LANCE_SCALAR_INDEX
    }
}

/// Trait for storing an index (or parts of an index) into storage
#[async_trait]
pub trait IndexWriter: Send {
    /// Writes a record batch into the file, returning the 0-based index of the batch in the file
    ///
    /// E.g. if this is the third time this is called this method will return 2
    async fn write_record_batch(&mut self, batch: RecordBatch) -> Result<u64>;
    /// Adds a global buffer and returns its index.
    async fn add_global_buffer(&mut self, _data: Bytes) -> Result<u32> {
        Err(Error::not_supported(
            "global buffers are not supported by this index writer",
        ))
    }
    /// Finishes writing the file and closes the file
    async fn finish(&mut self) -> Result<IndexFile>;
    /// Finishes writing the file and closes the file with additional metadata
    async fn finish_with_metadata(
        &mut self,
        metadata: HashMap<String, String>,
    ) -> Result<IndexFile>;
}

/// Trait for reading an index (or parts of an index) from storage
#[async_trait]
pub trait IndexReader: Send + Sync {
    /// Read the n-th record batch from the file
    async fn read_record_batch(&self, n: u64, batch_size: u64) -> Result<RecordBatch>;
    /// Reads a global buffer by index.
    async fn read_global_buffer(&self, _index: u32) -> Result<Bytes> {
        Err(Error::not_supported(
            "global buffers are not supported by this index reader",
        ))
    }
    /// Read the range of rows from the file.
    /// If projection is Some, only return the columns in the projection,
    /// nested columns like Some(&["x.y"]) are not supported.
    /// If projection is None, return all columns.
    async fn read_range(
        &self,
        range: std::ops::Range<usize>,
        projection: Option<&[&str]>,
    ) -> Result<RecordBatch>;
    /// Read multiple ranges and concatenate into a single batch.
    /// Default impl runs `read_range`s in parallel via `try_join_all`.
    async fn read_ranges(
        &self,
        ranges: &[std::ops::Range<usize>],
        projection: Option<&[&str]>,
    ) -> Result<RecordBatch> {
        if ranges.is_empty() {
            return self.read_range(0..0, projection).await;
        }
        let futures = ranges
            .iter()
            .map(|r| self.read_range(r.clone(), projection));
        let batches = futures::future::try_join_all(futures).await?;
        let schema = batches[0].schema();
        Ok(arrow_select::concat::concat_batches(&schema, &batches)?)
    }
    /// Read a range of rows as a stream of record batches.
    ///
    /// This allows the caller to process rows incrementally without loading the
    /// entire range into memory at once.
    ///
    /// The default implementation falls back to [`Self::read_range`] and wraps
    /// the result in a single-item stream.
    async fn read_range_stream(
        &self,
        range: std::ops::Range<usize>,
        projection: Option<&[&str]>,
    ) -> Result<Pin<Box<dyn RecordBatchStream>>> {
        let batch = self.read_range(range, projection).await?;
        let schema = batch.schema();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::once(async move { Ok(batch) }),
        )))
    }
    /// Return the number of batches in the file
    async fn num_batches(&self, batch_size: u64) -> u32;
    /// Return the number of rows in the file
    fn num_rows(&self) -> usize;
    /// Return the metadata of the file
    fn schema(&self) -> &lance_core::datatypes::Schema;
}

/// Trait abstracting I/O away from index logic
///
/// Scalar indices are currently serialized as indexable arrow record batches stored in
/// named "files".  The index store is responsible for serializing and deserializing
/// these batches into file data (e.g. as .lance files or .parquet files, etc.)
#[async_trait]
pub trait IndexStore: std::fmt::Debug + Send + Sync + DeepSizeOf {
    fn as_any(&self) -> &dyn Any;
    fn clone_arc(&self) -> Arc<dyn IndexStore>;

    /// Suggested I/O parallelism for the store
    fn io_parallelism(&self) -> usize;

    /// Create a new file and return a writer to store data in the file
    async fn new_index_file(&self, name: &str, schema: Arc<Schema>)
    -> Result<Box<dyn IndexWriter>>;

    /// Open an existing file for retrieval
    async fn open_index_file(&self, name: &str) -> Result<Arc<dyn IndexReader>>;

    /// Copy a range of batches from an index file from this store to another
    ///
    /// This is often useful when remapping or updating
    async fn copy_index_file(&self, name: &str, dest_store: &dyn IndexStore) -> Result<IndexFile>;

    /// Copy an index file from this store to a new name in another store, leaving the source intact
    async fn copy_index_file_to(
        &self,
        name: &str,
        new_name: &str,
        dest_store: &dyn IndexStore,
    ) -> Result<IndexFile> {
        if name == new_name {
            self.copy_index_file(name, dest_store).await
        } else {
            Err(Error::not_supported(format!(
                "copying index file {name} to {new_name} is not supported by this index store"
            )))
        }
    }

    /// Rename an index file
    async fn rename_index_file(&self, name: &str, new_name: &str) -> Result<IndexFile>;

    /// Delete an index file (used in the tmp spill store to keep tmp size down)
    async fn delete_index_file(&self, name: &str) -> Result<()>;

    /// List all files in the index directory with their sizes.
    ///
    /// Returns a list of (relative_path, size_bytes) tuples.
    /// Used to capture file metadata after index creation/modification.
    async fn list_files_with_sizes(&self) -> Result<Vec<IndexFile>>;
}

/// Different scalar indices may support different kinds of queries
///
/// For example, a btree index can support a wide range of queries (e.g. x > 7)
/// while an index based on FTS only supports queries like "x LIKE 'foo'"
///
/// This trait is used when we need an object that can represent any kind of query
///
/// Note: if you are implementing this trait for a query type then you probably also
/// need to implement the [crate::scalar::expression::ScalarQueryParser] trait to
/// create instances of your query at parse time.
pub trait AnyQuery: std::fmt::Debug + Any + Send + Sync {
    /// Cast the query as Any to allow for downcasting
    fn as_any(&self) -> &dyn Any;
    /// Format the query as a string for display purposes
    fn format(&self, col: &str) -> String;
    /// Convert the query to a datafusion expression
    fn to_expr(&self, col: String) -> datafusion_expr::Expr;
    /// Compare this query to another query
    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool;
}

impl PartialEq for dyn AnyQuery {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other)
    }
}

/// The result of a search operation against a scalar index
#[derive(Debug, PartialEq)]
pub enum SearchResult {
    /// The exact row ids that satisfy the query
    Exact(NullableRowAddrSet),
    /// Any row id satisfying the query will be in this set but not every
    /// row id in this set will satisfy the query, a further recheck step
    /// is needed
    AtMost(NullableRowAddrSet),
    /// All of the given row ids satisfy the query but there may be more
    ///
    /// No scalar index actually returns this today but it can arise from
    /// boolean operations (e.g. NOT(AtMost(x)) == AtLeast(NOT(x)))
    AtLeast(NullableRowAddrSet),
}

impl SearchResult {
    pub fn exact(row_ids: impl Into<RowAddrTreeMap>) -> Self {
        Self::Exact(NullableRowAddrSet::new(row_ids.into(), Default::default()))
    }

    pub fn at_most(row_ids: impl Into<RowAddrTreeMap>) -> Self {
        Self::AtMost(NullableRowAddrSet::new(row_ids.into(), Default::default()))
    }

    pub fn at_least(row_ids: impl Into<RowAddrTreeMap>) -> Self {
        Self::AtLeast(NullableRowAddrSet::new(row_ids.into(), Default::default()))
    }

    pub fn with_nulls(self, nulls: impl Into<RowAddrTreeMap>) -> Self {
        match self {
            Self::Exact(row_ids) => Self::Exact(row_ids.with_nulls(nulls.into())),
            Self::AtMost(row_ids) => Self::AtMost(row_ids.with_nulls(nulls.into())),
            Self::AtLeast(row_ids) => Self::AtLeast(row_ids.with_nulls(nulls.into())),
        }
    }

    pub fn row_addrs(&self) -> &NullableRowAddrSet {
        match self {
            Self::Exact(row_addrs) => row_addrs,
            Self::AtMost(row_addrs) => row_addrs,
            Self::AtLeast(row_addrs) => row_addrs,
        }
    }

    pub fn is_exact(&self) -> bool {
        matches!(self, Self::Exact(_))
    }
}

/// Brief information about an index that was created
pub struct CreatedIndex {
    /// The details of the index that was created
    ///
    /// These should be stored somewhere as they will be needed to
    /// load the index later.
    pub index_details: prost_types::Any,
    /// The version of the index that was created
    ///
    /// This can be used to determine if a reader is able to load the index.
    pub index_version: u32,
    /// List of files and their sizes for this index
    ///
    /// This enables skipping HEAD calls when opening indices and provides
    /// visibility into index storage size via describe_indices().
    pub files: Option<Vec<IndexFile>>,
}

/// The criteria that specifies how to update an index
pub struct UpdateCriteria {
    /// If true, then we need to read the old data to update the index
    ///
    /// This should be avoided if possible but is left in for some legacy paths
    pub requires_old_data: bool,
    /// The criteria required for data (both old and new)
    pub data_criteria: TrainingCriteria,
}

/// Filter used when merging existing scalar-index rows during update.
///
/// The caller must pick a filter mode that matches the row-id semantics of the
/// dataset:
/// - address-style row IDs: fragment filtering is valid
/// - stable row IDs: use exact row-id membership instead
#[derive(Debug, Clone)]
pub enum OldIndexDataFilter {
    /// Keeps track of which fragments are still valid and which are no longer valid.
    ///
    /// This is valid for address-style row IDs.
    Fragments {
        to_keep: RoaringBitmap,
        to_remove: RoaringBitmap,
    },
    /// Keep old rows whose row IDs are in this exact allow-list.
    ///
    /// This is required for stable row IDs, where row IDs are opaque and
    /// should not be interpreted as encoded row addresses.
    RowIds(RowAddrTreeMap),
}

impl OldIndexDataFilter {
    /// Build a boolean mask that keeps only row IDs selected by this filter.
    pub fn filter_row_ids(&self, row_ids: &UInt64Array) -> BooleanArray {
        match self {
            Self::Fragments { to_keep, .. } => row_ids
                .iter()
                .map(|id| id.map(|id| to_keep.contains((id >> 32) as u32)))
                .collect(),
            Self::RowIds(valid_row_ids) => row_ids
                .iter()
                .map(|id| id.map(|id| valid_row_ids.contains(id)))
                .collect(),
        }
    }
}

impl UpdateCriteria {
    pub fn requires_old_data(data_criteria: TrainingCriteria) -> Self {
        Self {
            requires_old_data: true,
            data_criteria,
        }
    }

    pub fn only_new_data(data_criteria: TrainingCriteria) -> Self {
        Self {
            requires_old_data: false,
            data_criteria,
        }
    }
}

/// Compute the lexicographically next prefix by incrementing the last character's code point.
/// Returns None if no valid upper bound exists.
///
/// This is used for LIKE prefix queries to convert `LIKE 'foo%'` to range `[foo, fop)`.
///
/// # UTF-8 and Unicode Handling
///
/// This function operates on Unicode code points (characters), not bytes. Since UTF-8
/// byte ordering is identical to Unicode code point ordering, incrementing a character's
/// code point produces the correct lexicographic successor for byte-wise string comparison.
///
/// If incrementing the last character would overflow or land in the surrogate range
/// (U+D800-U+DFFF), we try incrementing the previous character, and so on.
///
/// Examples:
/// - `"foo"` → `Some("fop")`
/// - `"café"` → `Some("cafê")`  (é U+00E9 → ê U+00EA)
/// - `"abc中"` → `Some("abc丮")` (中 U+4E2D → 丮 U+4E2E)
/// - `"cafÿ"` → `Some("cafĀ")` (ÿ U+00FF → Ā U+0100)
pub fn compute_next_prefix(prefix: &str) -> Option<String> {
    if prefix.is_empty() {
        return None;
    }

    let chars: Vec<char> = prefix.chars().collect();

    // Try incrementing characters from right to left
    for i in (0..chars.len()).rev() {
        if let Some(next_char) = next_unicode_char(chars[i]) {
            let mut result: String = chars[..i].iter().collect();
            result.push(next_char);
            return Some(result);
        }
        // This character cannot be incremented (e.g., U+10FFFF), try previous
    }

    // All characters were at maximum value
    None
}

/// Get the next valid Unicode scalar value after the given character.
/// Skips the surrogate range (U+D800-U+DFFF) which is not valid in UTF-8.
fn next_unicode_char(c: char) -> Option<char> {
    let cp = c as u32;
    let next_cp = cp.checked_add(1)?;

    // Skip surrogate range (U+D800-U+DFFF)
    let next_cp = if (0xD800..=0xDFFF).contains(&next_cp) {
        0xE000
    } else {
        next_cp
    };

    char::from_u32(next_cp)
}

/// A trait for a scalar index, a structure that can determine row ids that satisfy scalar queries
#[async_trait]
pub trait ScalarIndex: Send + Sync + std::fmt::Debug + Index + DeepSizeOf {
    /// Search the scalar index
    ///
    /// Returns all row ids that satisfy the query, these row ids are not necessarily ordered
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult>;

    /// Returns true if the remap operation is supported
    fn can_remap(&self) -> bool;

    /// Remap the row ids, creating a new remapped version of this index in `dest_store`
    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex>;

    /// Add the new data into the index, creating an updated version of the index in `dest_store`
    ///
    /// If `old_data_filter` is provided, old index data will be filtered before
    /// merge according to the chosen filter mode.
    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        old_data_filter: Option<OldIndexDataFilter>,
    ) -> Result<CreatedIndex>;

    /// Returns the criteria that will be used to update the index
    fn update_criteria(&self) -> UpdateCriteria;

    /// Derive the index parameters from the current index
    ///
    /// This returns a ScalarIndexParams that can be used to recreate an index
    /// with the same configuration on another dataset.
    fn derive_index_params(&self) -> Result<ScalarIndexParams>;
}
