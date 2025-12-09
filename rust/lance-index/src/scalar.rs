// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Scalar indices for metadata search & filtering

use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use arrow_array::{ListArray, RecordBatch};
use arrow_schema::{Field, Schema};
use async_trait::async_trait;
use datafusion::functions::string::contains::ContainsFunc;
use datafusion::functions_nested::array_has;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion_common::{scalar::ScalarValue, Column};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::{any::Any, ops::Bound, sync::Arc};

use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::Expr;
use deepsize::DeepSizeOf;
use inverted::query::{fill_fts_query_column, FtsQuery, FtsQueryNode, FtsSearchParams, MatchQuery};
use lance_core::utils::mask::RowAddrTreeMap;
use lance_core::{Error, Result};
use serde::Serialize;
use snafu::location;

use crate::metrics::MetricsCollector;
use crate::scalar::registry::TrainingCriteria;
use crate::{Index, IndexParams, IndexType};

pub mod bitmap;
pub mod bloomfilter;
pub mod btree;
pub mod expression;
pub mod flat;
pub mod inverted;
pub mod json;
pub mod label_list;
pub mod lance_format;
pub mod ngram;
pub mod registry;
pub mod zoned;
pub mod zonemap;

use crate::frag_reuse::FragReuseIndex;
pub use inverted::tokenizer::InvertedIndexParams;
use lance_datafusion::udf::CONTAINS_TOKENS_UDF;

pub const LANCE_SCALAR_INDEX: &str = "__lance_scalar_index";

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
    Inverted,
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
        }
    }
}

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
            _ => Err(Error::Index {
                message: "Invalid index type".to_string(),
                location: location!(),
            }),
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

impl IndexParams for InvertedIndexParams {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn index_name(&self) -> &str {
        "INVERTED"
    }
}

/// Trait for storing an index (or parts of an index) into storage
#[async_trait]
pub trait IndexWriter: Send {
    /// Writes a record batch into the file, returning the 0-based index of the batch in the file
    ///
    /// E.g. if this is the third time this is called this method will return 2
    async fn write_record_batch(&mut self, batch: RecordBatch) -> Result<u64>;
    /// Finishes writing the file and closes the file
    async fn finish(&mut self) -> Result<()>;
    /// Finishes writing the file and closes the file with additional metadata
    async fn finish_with_metadata(&mut self, metadata: HashMap<String, String>) -> Result<()>;
}

/// Trait for reading an index (or parts of an index) from storage
#[async_trait]
pub trait IndexReader: Send + Sync {
    /// Read the n-th record batch from the file
    async fn read_record_batch(&self, n: u64, batch_size: u64) -> Result<RecordBatch>;
    /// Read the range of rows from the file.
    /// If projection is Some, only return the columns in the projection,
    /// nested columns like Some(&["x.y"]) are not supported.
    /// If projection is None, return all columns.
    async fn read_range(
        &self,
        range: std::ops::Range<usize>,
        projection: Option<&[&str]>,
    ) -> Result<RecordBatch>;
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
    async fn copy_index_file(&self, name: &str, dest_store: &dyn IndexStore) -> Result<()>;

    /// Rename an index file
    async fn rename_index_file(&self, name: &str, new_name: &str) -> Result<()>;

    /// Delete an index file (used in the tmp spill store to keep tmp size down)
    async fn delete_index_file(&self, name: &str) -> Result<()>;
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
    fn to_expr(&self, col: String) -> Expr;
    /// Compare this query to another query
    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool;
}

impl PartialEq for dyn AnyQuery {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other)
    }
}
/// A full text search query
#[derive(Debug, Clone, PartialEq)]
pub struct FullTextSearchQuery {
    pub query: FtsQuery,

    /// The maximum number of results to return
    pub limit: Option<i64>,

    /// The wand factor to use for ranking
    /// if None, use the default value of 1.0
    /// Increasing this value will reduce the recall and improve the performance
    /// 1.0 is the value that would give the best performance without recall loss
    pub wand_factor: Option<f32>,
}

impl FullTextSearchQuery {
    /// Create a new terms query
    pub fn new(query: String) -> Self {
        let query = MatchQuery::new(query).into();
        Self {
            query,
            limit: None,
            wand_factor: None,
        }
    }

    /// Create a new fuzzy query
    pub fn new_fuzzy(term: String, max_distance: Option<u32>) -> Self {
        let query = MatchQuery::new(term).with_fuzziness(max_distance).into();
        Self {
            query,
            limit: None,
            wand_factor: None,
        }
    }

    /// Create a new compound query
    pub fn new_query(query: FtsQuery) -> Self {
        Self {
            query,
            limit: None,
            wand_factor: None,
        }
    }

    /// Set the column to search over
    /// This is available for only MatchQuery and PhraseQuery
    pub fn with_column(mut self, column: String) -> Result<Self> {
        self.query = fill_fts_query_column(&self.query, &[column], true)?;
        Ok(self)
    }

    /// Set the column to search over
    /// This is available for only MatchQuery
    pub fn with_columns(mut self, columns: &[String]) -> Result<Self> {
        self.query = fill_fts_query_column(&self.query, columns, true)?;
        Ok(self)
    }

    /// limit the number of results to return
    /// if None, return all results
    pub fn limit(mut self, limit: Option<i64>) -> Self {
        self.limit = limit;
        self
    }

    pub fn wand_factor(mut self, wand_factor: Option<f32>) -> Self {
        self.wand_factor = wand_factor;
        self
    }

    pub fn columns(&self) -> HashSet<String> {
        self.query.columns()
    }

    pub fn params(&self) -> FtsSearchParams {
        FtsSearchParams::new()
            .with_limit(self.limit.map(|limit| limit as usize))
            .with_wand_factor(self.wand_factor.unwrap_or(1.0))
    }
}

/// A query that a basic scalar index (e.g. btree / bitmap) can satisfy
///
/// This is a subset of expression operators that is often referred to as the
/// "sargable" operators
///
/// Note that negation is not included.  Negation should be applied later.  For
/// example, to invert an equality query (e.g. all rows where the value is not 7)
/// you can grab all rows where the value = 7 and then do an inverted take (or use
/// a block list instead of an allow list for prefiltering)
#[derive(Debug, Clone, PartialEq)]
pub enum SargableQuery {
    /// Retrieve all row ids where the value is in the given [min, max) range
    Range(Bound<ScalarValue>, Bound<ScalarValue>),
    /// Retrieve all row ids where the value is in the given set of values
    IsIn(Vec<ScalarValue>),
    /// Retrieve all row ids where the value is exactly the given value
    Equals(ScalarValue),
    /// Retrieve all row ids where the value matches the given full text search query
    FullTextSearch(FullTextSearchQuery),
    /// Retrieve all row ids where the value is null
    IsNull(),
}

impl AnyQuery for SargableQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn format(&self, col: &str) -> String {
        match self {
            Self::Range(lower, upper) => match (lower, upper) {
                (Bound::Unbounded, Bound::Unbounded) => "true".to_string(),
                (Bound::Unbounded, Bound::Included(rhs)) => format!("{} <= {}", col, rhs),
                (Bound::Unbounded, Bound::Excluded(rhs)) => format!("{} < {}", col, rhs),
                (Bound::Included(lhs), Bound::Unbounded) => format!("{} >= {}", col, lhs),
                (Bound::Included(lhs), Bound::Included(rhs)) => {
                    format!("{} >= {} && {} <= {}", col, lhs, col, rhs)
                }
                (Bound::Included(lhs), Bound::Excluded(rhs)) => {
                    format!("{} >= {} && {} < {}", col, lhs, col, rhs)
                }
                (Bound::Excluded(lhs), Bound::Unbounded) => format!("{} > {}", col, lhs),
                (Bound::Excluded(lhs), Bound::Included(rhs)) => {
                    format!("{} > {} && {} <= {}", col, lhs, col, rhs)
                }
                (Bound::Excluded(lhs), Bound::Excluded(rhs)) => {
                    format!("{} > {} && {} < {}", col, lhs, col, rhs)
                }
            },
            Self::IsIn(values) => {
                format!(
                    "{} IN [{}]",
                    col,
                    values
                        .iter()
                        .map(|val| val.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
            Self::FullTextSearch(query) => {
                format!("fts({})", query.query)
            }
            Self::IsNull() => {
                format!("{} IS NULL", col)
            }
            Self::Equals(val) => {
                format!("{} = {}", col, val)
            }
        }
    }

    fn to_expr(&self, col: String) -> Expr {
        let col_expr = Expr::Column(Column::new_unqualified(col));
        match self {
            Self::Range(lower, upper) => match (lower, upper) {
                (Bound::Unbounded, Bound::Unbounded) => {
                    Expr::Literal(ScalarValue::Boolean(Some(true)), None)
                }
                (Bound::Unbounded, Bound::Included(rhs)) => {
                    col_expr.lt_eq(Expr::Literal(rhs.clone(), None))
                }
                (Bound::Unbounded, Bound::Excluded(rhs)) => {
                    col_expr.lt(Expr::Literal(rhs.clone(), None))
                }
                (Bound::Included(lhs), Bound::Unbounded) => {
                    col_expr.gt_eq(Expr::Literal(lhs.clone(), None))
                }
                (Bound::Included(lhs), Bound::Included(rhs)) => col_expr.between(
                    Expr::Literal(lhs.clone(), None),
                    Expr::Literal(rhs.clone(), None),
                ),
                (Bound::Included(lhs), Bound::Excluded(rhs)) => col_expr
                    .clone()
                    .gt_eq(Expr::Literal(lhs.clone(), None))
                    .and(col_expr.lt(Expr::Literal(rhs.clone(), None))),
                (Bound::Excluded(lhs), Bound::Unbounded) => {
                    col_expr.gt(Expr::Literal(lhs.clone(), None))
                }
                (Bound::Excluded(lhs), Bound::Included(rhs)) => col_expr
                    .clone()
                    .gt(Expr::Literal(lhs.clone(), None))
                    .and(col_expr.lt_eq(Expr::Literal(rhs.clone(), None))),
                (Bound::Excluded(lhs), Bound::Excluded(rhs)) => col_expr
                    .clone()
                    .gt(Expr::Literal(lhs.clone(), None))
                    .and(col_expr.lt(Expr::Literal(rhs.clone(), None))),
            },
            Self::IsIn(values) => col_expr.in_list(
                values
                    .iter()
                    .map(|val| Expr::Literal(val.clone(), None))
                    .collect::<Vec<_>>(),
                false,
            ),
            Self::FullTextSearch(query) => col_expr.like(Expr::Literal(
                ScalarValue::Utf8(Some(query.query.to_string())),
                None,
            )),
            Self::IsNull() => col_expr.is_null(),
            Self::Equals(value) => col_expr.eq(Expr::Literal(value.clone(), None)),
        }
    }

    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
        match other.as_any().downcast_ref::<Self>() {
            Some(o) => self == o,
            None => false,
        }
    }
}

/// A query that a LabelListIndex can satisfy
#[derive(Debug, Clone, PartialEq)]
pub enum LabelListQuery {
    /// Retrieve all row ids where every label is in the list of values for the row
    HasAllLabels(Vec<ScalarValue>),
    /// Retrieve all row ids where at least one of the given labels is in the list of values for the row
    HasAnyLabel(Vec<ScalarValue>),
}

impl AnyQuery for LabelListQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn format(&self, col: &str) -> String {
        format!("{}", self.to_expr(col.to_string()))
    }

    fn to_expr(&self, col: String) -> Expr {
        match self {
            Self::HasAllLabels(labels) => {
                let labels_arr = ScalarValue::iter_to_array(labels.iter().cloned()).unwrap();
                let offsets_buffer =
                    OffsetBuffer::new(ScalarBuffer::<i32>::from(vec![0, labels_arr.len() as i32]));
                let labels_list = ListArray::try_new(
                    Arc::new(Field::new("item", labels_arr.data_type().clone(), false)),
                    offsets_buffer,
                    labels_arr,
                    None,
                )
                .unwrap();
                let labels_arr = Arc::new(labels_list);
                Expr::ScalarFunction(ScalarFunction {
                    func: Arc::new(array_has::ArrayHasAll::new().into()),
                    args: vec![
                        Expr::Column(Column::new_unqualified(col)),
                        Expr::Literal(ScalarValue::List(labels_arr), None),
                    ],
                })
            }
            Self::HasAnyLabel(labels) => {
                let labels_arr = ScalarValue::iter_to_array(labels.iter().cloned()).unwrap();
                let offsets_buffer =
                    OffsetBuffer::new(ScalarBuffer::<i32>::from(vec![0, labels_arr.len() as i32]));
                let labels_list = ListArray::try_new(
                    Arc::new(Field::new("item", labels_arr.data_type().clone(), false)),
                    offsets_buffer,
                    labels_arr,
                    None,
                )
                .unwrap();
                let labels_arr = Arc::new(labels_list);
                Expr::ScalarFunction(ScalarFunction {
                    func: Arc::new(array_has::ArrayHasAny::new().into()),
                    args: vec![
                        Expr::Column(Column::new_unqualified(col)),
                        Expr::Literal(ScalarValue::List(labels_arr), None),
                    ],
                })
            }
        }
    }

    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
        match other.as_any().downcast_ref::<Self>() {
            Some(o) => self == o,
            None => false,
        }
    }
}

/// A query that a NGramIndex can satisfy
#[derive(Debug, Clone, PartialEq)]
pub enum TextQuery {
    /// Retrieve all row ids where the text contains the given string
    StringContains(String),
    // TODO: In the future we should be able to do string-insensitive contains
    // as well as partial matches (e.g. LIKE 'foo%') and potentially even
    // some regular expressions
}

impl AnyQuery for TextQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn format(&self, col: &str) -> String {
        format!("{}", self.to_expr(col.to_string()))
    }

    fn to_expr(&self, col: String) -> Expr {
        match self {
            Self::StringContains(substr) => Expr::ScalarFunction(ScalarFunction {
                func: Arc::new(ContainsFunc::new().into()),
                args: vec![
                    Expr::Column(Column::new_unqualified(col)),
                    Expr::Literal(ScalarValue::Utf8(Some(substr.clone())), None),
                ],
            }),
        }
    }

    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
        match other.as_any().downcast_ref::<Self>() {
            Some(o) => self == o,
            None => false,
        }
    }
}

/// A query that a InvertedIndex can satisfy
#[derive(Debug, Clone, PartialEq)]
pub enum TokenQuery {
    /// Retrieve all row ids where the text contains all tokens parsed from given string. The tokens
    /// are separated by punctuations and white spaces.
    TokensContains(String),
}

/// A query that a BloomFilter index can satisfy
///
/// This is a subset of SargableQuery that only includes operations that bloom filters
/// can efficiently handle: equals, is_null, and is_in queries.
#[derive(Debug, Clone, PartialEq)]
pub enum BloomFilterQuery {
    /// Retrieve all row ids where the value is exactly the given value
    Equals(ScalarValue),
    /// Retrieve all row ids where the value is null
    IsNull(),
    /// Retrieve all row ids where the value is in the given set of values
    IsIn(Vec<ScalarValue>),
}

impl AnyQuery for BloomFilterQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn format(&self, col: &str) -> String {
        match self {
            Self::Equals(val) => {
                format!("{} = {}", col, val)
            }
            Self::IsNull() => {
                format!("{} IS NULL", col)
            }
            Self::IsIn(values) => {
                format!(
                    "{} IN [{}]",
                    col,
                    values
                        .iter()
                        .map(|val| val.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
        }
    }

    fn to_expr(&self, col: String) -> Expr {
        let col_expr = Expr::Column(Column::new_unqualified(col));
        match self {
            Self::Equals(value) => col_expr.eq(Expr::Literal(value.clone(), None)),
            Self::IsNull() => col_expr.is_null(),
            Self::IsIn(values) => col_expr.in_list(
                values
                    .iter()
                    .map(|val| Expr::Literal(val.clone(), None))
                    .collect::<Vec<_>>(),
                false,
            ),
        }
    }

    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
        match other.as_any().downcast_ref::<Self>() {
            Some(o) => self == o,
            None => false,
        }
    }
}

impl AnyQuery for TokenQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn format(&self, col: &str) -> String {
        format!("{}", self.to_expr(col.to_string()))
    }

    fn to_expr(&self, col: String) -> Expr {
        match self {
            Self::TokensContains(substr) => Expr::ScalarFunction(ScalarFunction {
                func: Arc::new(CONTAINS_TOKENS_UDF.clone()),
                args: vec![
                    Expr::Column(Column::new_unqualified(col)),
                    Expr::Literal(ScalarValue::Utf8(Some(substr.clone())), None),
                ],
            }),
        }
    }

    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
        match other.as_any().downcast_ref::<Self>() {
            Some(o) => self == o,
            None => false,
        }
    }
}

/// The result of a search operation against a scalar index
#[derive(Debug, PartialEq)]
pub enum SearchResult {
    /// The exact row ids that satisfy the query
    Exact(RowAddrTreeMap),
    /// Any row id satisfying the query will be in this set but not every
    /// row id in this set will satisfy the query, a further recheck step
    /// is needed
    AtMost(RowAddrTreeMap),
    /// All of the given row ids satisfy the query but there may be more
    ///
    /// No scalar index actually returns this today but it can arise from
    /// boolean operations (e.g. NOT(AtMost(x)) == AtLeast(NOT(x)))
    AtLeast(RowAddrTreeMap),
}

impl SearchResult {
    pub fn row_addrs(&self) -> &RowAddrTreeMap {
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
    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex>;

    /// Returns the criteria that will be used to update the index
    fn update_criteria(&self) -> UpdateCriteria;

    /// Derive the index parameters from the current index
    ///
    /// This returns a ScalarIndexParams that can be used to recreate an index
    /// with the same configuration on another dataset.
    fn derive_index_params(&self) -> Result<ScalarIndexParams>;
}
