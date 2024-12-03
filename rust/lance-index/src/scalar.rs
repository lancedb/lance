// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Scalar indices for metadata search & filtering

use std::collections::HashMap;
use std::fmt::Debug;
use std::{any::Any, ops::Bound, sync::Arc};

use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use arrow_array::{ListArray, RecordBatch};
use arrow_schema::{Field, Schema};
use async_trait::async_trait;
use datafusion::functions_array::array_has;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion_common::{scalar::ScalarValue, Column};

use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::Expr;
use deepsize::DeepSizeOf;
use inverted::TokenizerConfig;
use lance_core::utils::mask::RowIdTreeMap;
use lance_core::{Error, Result};
use snafu::{location, Location};

use crate::{Index, IndexParams, IndexType};

pub mod bitmap;
pub mod btree;
pub mod expression;
pub mod flat;
pub mod inverted;
pub mod label_list;
pub mod lance_format;

pub const LANCE_SCALAR_INDEX: &str = "__lance_scalar_index";

#[derive(Debug, Copy, Clone)]
pub enum ScalarIndexType {
    BTree,
    Bitmap,
    LabelList,
    Inverted,
}

impl TryFrom<IndexType> for ScalarIndexType {
    type Error = Error;

    fn try_from(value: IndexType) -> Result<Self> {
        match value {
            IndexType::BTree | IndexType::Scalar => Ok(Self::BTree),
            IndexType::Bitmap => Ok(Self::Bitmap),
            IndexType::LabelList => Ok(Self::LabelList),
            IndexType::Inverted => Ok(Self::Inverted),
            _ => Err(Error::InvalidInput {
                source: format!("Index type {:?} is not a scalar index", value).into(),
                location: location!(),
            }),
        }
    }
}

#[derive(Default)]
pub struct ScalarIndexParams {
    /// If set then always use the given index type and skip auto-detection
    pub force_index_type: Option<ScalarIndexType>,
}

impl ScalarIndexParams {
    pub fn new(index_type: ScalarIndexType) -> Self {
        Self {
            force_index_type: Some(index_type),
        }
    }
}

impl IndexParams for ScalarIndexParams {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn index_type(&self) -> IndexType {
        match self.force_index_type {
            Some(ScalarIndexType::BTree) | None => IndexType::BTree,
            Some(ScalarIndexType::Bitmap) => IndexType::Bitmap,
            Some(ScalarIndexType::LabelList) => IndexType::LabelList,
            Some(ScalarIndexType::Inverted) => IndexType::Inverted,
        }
    }

    fn index_name(&self) -> &str {
        LANCE_SCALAR_INDEX
    }
}

#[derive(Clone)]
pub struct InvertedIndexParams {
    /// If true, store the position of the term in the document
    /// This can significantly increase the size of the index
    /// If false, only store the frequency of the term in the document
    /// Default is true
    pub with_position: bool,

    pub tokenizer_config: TokenizerConfig,
}

impl Debug for InvertedIndexParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedIndexParams")
            .field("with_position", &self.with_position)
            .finish()
    }
}

impl DeepSizeOf for InvertedIndexParams {
    fn deep_size_of_children(&self, _: &mut deepsize::Context) -> usize {
        0
    }
}

impl Default for InvertedIndexParams {
    fn default() -> Self {
        Self {
            with_position: true,
            tokenizer_config: TokenizerConfig::default(),
        }
    }
}

impl InvertedIndexParams {
    pub fn with_position(mut self, with_position: bool) -> Self {
        self.with_position = with_position;
        self
    }
}

impl IndexParams for InvertedIndexParams {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn index_type(&self) -> IndexType {
        IndexType::Inverted
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
    async fn read_record_batch(&self, n: u32) -> Result<RecordBatch>;
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
    async fn num_batches(&self) -> u32;
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
    /// Format the query as a string
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
    /// The columns to search,
    /// if empty, search all indexed columns
    pub columns: Vec<String>,
    /// The full text search query
    pub query: String,
    /// The maximum number of results to return
    pub limit: Option<i64>,
    /// The wand factor to use for ranking
    /// if None, use the default value of 1.0
    /// Increasing this value will reduce the recall and improve the performance
    /// 1.0 is the value that would give the best performance without recall loss
    pub wand_factor: Option<f32>,
}

impl FullTextSearchQuery {
    pub fn new(query: String) -> Self {
        Self {
            query,
            limit: None,
            columns: vec![],
            wand_factor: None,
        }
    }

    pub fn columns(mut self, columns: Option<Vec<String>>) -> Self {
        if let Some(columns) = columns {
            self.columns = columns;
        }
        self
    }

    pub fn limit(mut self, limit: Option<i64>) -> Self {
        self.limit = limit;
        self
    }

    pub fn wand_factor(mut self, wand_factor: Option<f32>) -> Self {
        self.wand_factor = wand_factor;
        self
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
                    Expr::Literal(ScalarValue::Boolean(Some(true)))
                }
                (Bound::Unbounded, Bound::Included(rhs)) => {
                    col_expr.lt_eq(Expr::Literal(rhs.clone()))
                }
                (Bound::Unbounded, Bound::Excluded(rhs)) => col_expr.lt(Expr::Literal(rhs.clone())),
                (Bound::Included(lhs), Bound::Unbounded) => {
                    col_expr.gt_eq(Expr::Literal(lhs.clone()))
                }
                (Bound::Included(lhs), Bound::Included(rhs)) => {
                    col_expr.between(Expr::Literal(lhs.clone()), Expr::Literal(rhs.clone()))
                }
                (Bound::Included(lhs), Bound::Excluded(rhs)) => col_expr
                    .clone()
                    .gt_eq(Expr::Literal(lhs.clone()))
                    .and(col_expr.lt(Expr::Literal(rhs.clone()))),
                (Bound::Excluded(lhs), Bound::Unbounded) => col_expr.gt(Expr::Literal(lhs.clone())),
                (Bound::Excluded(lhs), Bound::Included(rhs)) => col_expr
                    .clone()
                    .gt(Expr::Literal(lhs.clone()))
                    .and(col_expr.lt_eq(Expr::Literal(rhs.clone()))),
                (Bound::Excluded(lhs), Bound::Excluded(rhs)) => col_expr
                    .clone()
                    .gt(Expr::Literal(lhs.clone()))
                    .and(col_expr.lt(Expr::Literal(rhs.clone()))),
            },
            Self::IsIn(values) => col_expr.in_list(
                values
                    .iter()
                    .map(|val| Expr::Literal(val.clone()))
                    .collect::<Vec<_>>(),
                false,
            ),
            Self::FullTextSearch(query) => {
                col_expr.like(Expr::Literal(ScalarValue::Utf8(Some(query.query.clone()))))
            }
            Self::IsNull() => col_expr.is_null(),
            Self::Equals(value) => col_expr.eq(Expr::Literal(value.clone())),
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
                        Expr::Literal(ScalarValue::List(labels_arr)),
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
                        Expr::Literal(ScalarValue::List(labels_arr)),
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

/// A trait for a scalar index, a structure that can determine row ids that satisfy scalar queries
#[async_trait]
pub trait ScalarIndex: Send + Sync + std::fmt::Debug + Index + DeepSizeOf {
    /// Search the scalar index
    ///
    /// Returns all row ids that satisfy the query, these row ids are not necessarily ordered
    async fn search(&self, query: &dyn AnyQuery) -> Result<RowIdTreeMap>;

    /// Load the scalar index from storage
    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>>
    where
        Self: Sized;

    /// Remap the row ids, creating a new remapped version of this index in `dest_store`
    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<()>;

    /// Add the new data into the index, creating an updated version of the index in `dest_store`
    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()>;
}
