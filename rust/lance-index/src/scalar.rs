// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Scalar indices for metadata search & filtering

use std::collections::HashMap;
use std::{any::Any, ops::Bound, sync::Arc};

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::Schema;
use async_trait::async_trait;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion_common::{scalar::ScalarValue, Column};

use datafusion_expr::Expr;
use deepsize::DeepSizeOf;
use lance_core::Result;

use crate::Index;

pub mod btree;
pub mod expression;
pub mod flat;
pub mod invert;
pub mod lance_format;

/// Trait for storing an index (or parts of an index) into storage
#[async_trait]
pub trait IndexWriter: Send {
    /// Writes a record batch into the file, returning the 0-based index of the batch in the file
    ///
    /// E.g. if this is the third time this is called this method will return 2
    async fn write_record_batch(&mut self, batch: RecordBatch) -> Result<u64>;
    /// Finishes writing the file and closes the file
    async fn finish(&mut self) -> Result<()>;
}

/// Trait for reading an index (or parts of an index) from storage
#[async_trait]
pub trait IndexReader: Send + Sync {
    /// Read the n-th record batch from the file
    async fn read_record_batch(&self, n: u32) -> Result<RecordBatch>;
    /// Return the number of batches in the file
    async fn num_batches(&self) -> u32;
}

/// Trait abstracting I/O away from index logic
///
/// Scalar indices are currently serialized as indexable arrow record batches stored in
/// named "files".  The index store is responsible for serializing and deserializing
/// these batches into file data (e.g. as .lance files or .parquet files, etc.)
#[async_trait]
pub trait IndexStore: std::fmt::Debug + Send + Sync + DeepSizeOf {
    fn as_any(&self) -> &dyn Any;

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

/// A query that a scalar index can satisfy
///
/// This is a subset of expression operators that is often referred to as the
/// "sargable" operators
///
/// Note that negation is not included.  Negation should be applied later.  For
/// example, to invert an equality query (e.g. all rows where the value is not 7)
/// you can grab all rows where the value = 7 and then do an inverted take (or use
/// a block list instead of an allow list for prefiltering)
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarQuery {
    /// Retrieve all row ids where the value is in the given [min, max) range
    Range(Bound<ScalarValue>, Bound<ScalarValue>),
    /// Retrieve all row ids where the value is in the given set of values
    IsIn(Vec<ScalarValue>),
    /// Retrieve all row ids where the value is exactly the given value
    Equals(ScalarValue),
    /// Retrieve all row ids where the value matches the given full text search query
    FullTextSearch(Vec<String>),
    /// Retrieve all row ids where the value is null
    IsNull(),
}

impl ScalarQuery {
    pub fn to_expr(&self, col: String) -> Expr {
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
            Self::FullTextSearch(values) => {
                col_expr.like(Expr::Literal(ScalarValue::Utf8(Some(values.join("|")))))
            }
            Self::IsNull() => col_expr.is_null(),
            Self::Equals(value) => col_expr.eq(Expr::Literal(value.clone())),
        }
    }

    pub fn fmt_with_col(&self, col: &str) -> String {
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
            Self::FullTextSearch(values) => {
                format!("{} LIKE '{}'", col, values.join("|"))
            }
            Self::IsNull() => {
                format!("{} IS NULL", col)
            }
            Self::Equals(val) => {
                format!("{} = {}", col, val)
            }
        }
    }
}

/// A trait for a scalar index, a structure that can determine row ids that satisfy scalar queries
#[async_trait]
pub trait ScalarIndex: Send + Sync + std::fmt::Debug + Index + DeepSizeOf {
    /// Search the scalar index
    ///
    /// Returns all row ids that satisfy the query, these row ids are not neccesarily ordered
    async fn search(&self, query: &ScalarQuery) -> Result<UInt64Array>;

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
