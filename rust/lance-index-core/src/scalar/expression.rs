// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Core scalar index expression types shared between lance-index-core and lance-index.

use std::{ops::Bound, sync::Arc};

use arrow_schema::DataType;
use async_recursion::async_recursion;
use async_trait::async_trait;
use datafusion_common::ScalarValue;
use datafusion_expr::{Expr, Like, Operator, ScalarUDF};
use lance_core::Result;
use lance_select::{IndexExprResult, NullableIndexExprResult, NullableRowAddrMask};
use roaring::RoaringBitmap;
use futures::try_join;
use tracing::instrument;

use crate::metrics::MetricsCollector;
use crate::scalar::{AnyQuery, ScalarIndex, SearchResult};

/// An indexed expression consists of a scalar index query with a post-scan filter
///
/// When a user wants to filter the data returned by a scan we may be able to use
/// one or more scalar indices to reduce the amount of data we load from the disk.
///
/// For example, if a user provides the filter "x = 7", and we have a scalar index
/// on x, then we can possibly identify the exact row that the user desires with our
/// index.  A full-table scan can then turn into a take operation fetching the rows
/// desired.  This would create an IndexedExpression with a scalar_query but no
/// refine.
///
/// If the user asked for "type = 'dog' && z = 3" and we had a scalar index on the
/// "type" column then we could convert this to an indexed scan for "type='dog'"
/// followed by an in-memory filter for z=3.  This would create an IndexedExpression
/// with both a scalar_query AND a refine.
///
/// Finally, if the user asked for "z = 3" and we do not have a scalar index on the
/// "z" column then we must fallback to an IndexedExpression with no scalar_query and
/// only a refine.
///
/// Two IndexedExpressions can be AND'd together.  Each part is AND'd together.
/// Two IndexedExpressions cannot be OR'd together unless both are scalar_query only
///   or both are refine only
/// An IndexedExpression cannot be negated if it has both a refine and a scalar_query
///
/// When an operation cannot be performed we fallback to the original expression-only
/// representation
#[derive(Debug, PartialEq)]
pub struct IndexedExpression {
    /// The portion of the query that can be satisfied by scalar indices
    pub scalar_query: Option<ScalarIndexExpr>,
    /// The portion of the query that cannot be satisfied by scalar indices
    pub refine_expr: Option<Expr>,
}

impl IndexedExpression {
    /// Create an expression that only does refine
    pub fn refine_only(refine_expr: Expr) -> Self {
        Self {
            scalar_query: None,
            refine_expr: Some(refine_expr),
        }
    }

    /// Create an expression that is only an index query
    pub fn index_query(
        column: String,
        index_name: String,
        index_type: String,
        query: Arc<dyn AnyQuery>,
    ) -> Self {
        Self {
            scalar_query: Some(ScalarIndexExpr::Query(ScalarIndexSearch {
                column,
                index_name,
                index_type,
                query,
                needs_recheck: false,
                fragment_bitmap: None,
            })),
            refine_expr: None,
        }
    }

    /// Create an expression that is only an index query with explicit needs_recheck
    pub fn index_query_with_recheck(
        column: String,
        index_name: String,
        index_type: String,
        query: Arc<dyn AnyQuery>,
        needs_recheck: bool,
    ) -> Self {
        Self {
            scalar_query: Some(ScalarIndexExpr::Query(ScalarIndexSearch {
                column,
                index_name,
                index_type,
                query,
                needs_recheck,
                fragment_bitmap: None,
            })),
            refine_expr: None,
        }
    }

    /// Try and negate the expression
    ///
    /// If the expression contains both an index query and a refine expression then it
    /// cannot be negated today and None will be returned (we give up trying to use indices)
    pub fn maybe_not(self) -> Option<Self> {
        match (self.scalar_query, self.refine_expr) {
            (Some(_), Some(_)) => None,
            (Some(scalar_query), None) => {
                if scalar_query.needs_recheck() {
                    return None;
                }
                Some(Self {
                    scalar_query: Some(ScalarIndexExpr::Not(Box::new(scalar_query))),
                    refine_expr: None,
                })
            }
            (None, Some(refine_expr)) => Some(Self {
                scalar_query: None,
                refine_expr: Some(Expr::Not(Box::new(refine_expr))),
            }),
            (None, None) => panic!("Empty node should not occur"),
        }
    }

    /// Perform a logical AND of two indexed expressions
    ///
    /// This is straightforward because we can just AND the individual parts
    /// because (A && B) && (C && D) == (A && C) && (B && D)
    pub fn and(self, other: Self) -> Self {
        let scalar_query = match (self.scalar_query, other.scalar_query) {
            (Some(scalar_query), Some(other_scalar_query)) => Some(ScalarIndexExpr::And(
                Box::new(scalar_query),
                Box::new(other_scalar_query),
            )),
            (Some(scalar_query), None) => Some(scalar_query),
            (None, Some(scalar_query)) => Some(scalar_query),
            (None, None) => None,
        };
        let refine_expr = match (self.refine_expr, other.refine_expr) {
            (Some(refine_expr), Some(other_refine_expr)) => {
                Some(refine_expr.and(other_refine_expr))
            }
            (Some(refine_expr), None) => Some(refine_expr),
            (None, Some(refine_expr)) => Some(refine_expr),
            (None, None) => None,
        };
        Self {
            scalar_query,
            refine_expr,
        }
    }

    /// Try and perform a logical OR of two indexed expressions
    ///
    /// This is a bit tricky because something like:
    ///   (color == 'blue' AND size < 20) OR (color == 'green' AND size < 50)
    /// is not equivalent to:
    ///   (color == 'blue' OR color == 'green') AND (size < 20 OR size < 50)
    pub fn maybe_or(self, other: Self) -> Option<Self> {
        // If either expression is missing a scalar_query then we need to load all rows from
        // the database and so we short-circuit and return None
        let scalar_query = self.scalar_query?;
        let other_scalar_query = other.scalar_query?;
        let scalar_query = Some(ScalarIndexExpr::Or(
            Box::new(scalar_query),
            Box::new(other_scalar_query),
        ));

        let refine_expr = match (self.refine_expr, other.refine_expr) {
            (Some(_), Some(_)) => {
                return None;
            }
            (Some(_), None) => {
                return None;
            }
            (None, Some(_)) => {
                return None;
            }
            (None, None) => None,
        };
        Some(Self {
            scalar_query,
            refine_expr,
        })
    }

    /// Add a refine expression to this IndexedExpression
    pub fn refine(self, expr: Expr) -> Self {
        match self.refine_expr {
            Some(refine_expr) => Self {
                scalar_query: self.scalar_query,
                refine_expr: Some(refine_expr.and(expr)),
            },
            None => Self {
                scalar_query: self.scalar_query,
                refine_expr: Some(expr),
            },
        }
    }
}

/// A trait for scalar index query parsers.
///
/// Implementors visit different expression types and return [`IndexedExpression`]
/// instances when the index can accelerate the given predicate.
pub trait ScalarQueryParser: std::fmt::Debug + Send + Sync {
    /// Visit a between expression
    ///
    /// Returns an IndexedExpression if the index can accelerate between expressions
    fn visit_between(
        &self,
        column: &str,
        low: &Bound<ScalarValue>,
        high: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression>;
    /// Visit an in list expression
    ///
    /// Returns an IndexedExpression if the index can accelerate in list expressions
    fn visit_in_list(&self, column: &str, in_list: &[ScalarValue]) -> Option<IndexedExpression>;
    /// Visit an is bool expression
    ///
    /// Returns an IndexedExpression if the index can accelerate is bool expressions
    fn visit_is_bool(&self, column: &str, value: bool) -> Option<IndexedExpression>;
    /// Visit an is null expression
    ///
    /// Returns an IndexedExpression if the index can accelerate is null expressions
    fn visit_is_null(&self, column: &str) -> Option<IndexedExpression>;
    /// Visit a comparison expression
    ///
    /// Returns an IndexedExpression if the index can accelerate comparison expressions
    fn visit_comparison(
        &self,
        column: &str,
        value: &ScalarValue,
        op: &Operator,
    ) -> Option<IndexedExpression>;
    /// Visit a scalar function expression
    ///
    /// Returns an IndexedExpression if the index can accelerate the given scalar function.
    /// For example, an ngram index can accelerate the contains function.
    fn visit_scalar_function(
        &self,
        column: &str,
        data_type: &DataType,
        func: &ScalarUDF,
        args: &[Expr],
    ) -> Option<IndexedExpression>;

    /// Visit a LIKE expression
    ///
    /// Returns an IndexedExpression if the index can accelerate LIKE expressions.
    fn visit_like(
        &self,
        _column: &str,
        _like: &Like,
        _pattern: &ScalarValue,
    ) -> Option<IndexedExpression> {
        None
    }

    /// Visits a potential reference to a column
    ///
    /// This function is used to test if a potential column reference is a reference
    /// the index handles. Most indexes are designed to run on references to the indexed
    /// column, but some indexes handle projections (e.g. JSON indexes).
    ///
    /// The default implementation matches column references but this can be overridden
    /// by indexes that handle projections.
    ///
    /// Returns the data type of the reference if it is valid, `None` otherwise.
    fn is_valid_reference(&self, func: &Expr, data_type: &DataType) -> Option<DataType> {
        match func {
            Expr::Column(_) => Some(data_type.clone()),
            _ => None,
        }
    }
}

/// A trait implemented by anything that can load indices by name
///
/// This is used during the evaluation of an index expression
#[async_trait]
pub trait ScalarIndexLoader: Send + Sync {
    /// Load the index with the given name
    async fn load_index(
        &self,
        column: &str,
        index_name: &str,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>>;

    /// Translate an address-domain index result into the row-id domain
    ///
    /// Address-domain indices (see [`ScalarIndex::results_are_row_addresses`])
    /// report matches as physical row addresses. The default returns `result`
    /// unchanged, which is correct when addresses and row ids coincide (no
    /// stable row ids). A dataset with stable row ids overrides this to remap
    /// addresses to stable row ids via its per-fragment row-id sequences.
    async fn row_addr_result_to_row_ids(
        &self,
        result: NullableIndexExprResult,
    ) -> Result<NullableIndexExprResult> {
        Ok(result)
    }
}

/// This represents a search into a scalar index
#[derive(Debug, Clone)]
pub struct ScalarIndexSearch {
    /// The column to search (redundant, used for debugging messages)
    pub column: String,
    /// The name of the index to search
    pub index_name: String,
    /// The type of the index being searched (e.g. "BTree", "Bitmap"), used for display purposes
    pub index_type: String,
    /// The query to search for
    pub query: Arc<dyn AnyQuery>,
    /// If true, the query results are inexact and will need a recheck
    pub needs_recheck: bool,
    /// The fragments the underlying index has entries for.
    ///
    /// `None` means coverage is unknown (e.g. constructed outside of scanner
    /// planning, or from a legacy code path). Optimizer rules that need to
    /// decide whether the index covers the dataset must treat `None` as
    /// "refuse to use" — the bitmap is the only way to safely answer that
    /// question synchronously without an async metadata load.
    pub fragment_bitmap: Option<RoaringBitmap>,
}

impl PartialEq for ScalarIndexSearch {
    fn eq(&self, other: &Self) -> bool {
        // `fragment_bitmap` is metadata derived from the dataset state, not
        // part of the query identity, so it intentionally does not participate
        // in equality.
        self.column == other.column
            && self.index_name == other.index_name
            && self.query.as_ref().eq(other.query.as_ref())
    }
}

/// This represents a lookup into one or more scalar indices
///
/// This is a tree of operations because we may need to logically combine or
/// modify the results of scalar lookups
#[derive(Debug, Clone)]
pub enum ScalarIndexExpr {
    Not(Box<Self>),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Query(ScalarIndexSearch),
}

impl PartialEq for ScalarIndexExpr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Not(l0), Self::Not(r0)) => l0 == r0,
            (Self::And(l0, l1), Self::And(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Or(l0, l1), Self::Or(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Query(l_search), Self::Query(r_search)) => l_search == r_search,
            _ => false,
        }
    }
}

fn search_result_to_nullable(result: SearchResult) -> NullableIndexExprResult {
    match result {
        SearchResult::Exact(mask) => {
            NullableIndexExprResult::exact(NullableRowAddrMask::AllowList(mask))
        }
        SearchResult::AtMost(mask) => {
            NullableIndexExprResult::at_most(NullableRowAddrMask::AllowList(mask))
        }
        SearchResult::AtLeast(mask) => {
            NullableIndexExprResult::at_least(NullableRowAddrMask::AllowList(mask))
        }
    }
}

impl ScalarIndexExpr {
    /// Evaluates the scalar index expression
    ///
    /// This will result in loading one or more scalar indices and searching them
    #[async_recursion]
    pub async fn evaluate_nullable(
        &self,
        index_loader: &dyn ScalarIndexLoader,
        metrics: &dyn MetricsCollector,
    ) -> Result<NullableIndexExprResult> {
        match self {
            Self::Not(inner) => {
                let result = inner.evaluate_nullable(index_loader, metrics).await?;
                Ok(!result)
            }
            Self::And(lhs, rhs) => {
                let lhs_result = lhs.evaluate_nullable(index_loader, metrics);
                let rhs_result = rhs.evaluate_nullable(index_loader, metrics);
                let (lhs_result, rhs_result) = try_join!(lhs_result, rhs_result)?;
                Ok(lhs_result & rhs_result)
            }
            Self::Or(lhs, rhs) => {
                let lhs_result = lhs.evaluate_nullable(index_loader, metrics);
                let rhs_result = rhs.evaluate_nullable(index_loader, metrics);
                let (lhs_result, rhs_result) = try_join!(lhs_result, rhs_result)?;
                Ok(lhs_result | rhs_result)
            }
            Self::Query(search) => {
                let index = index_loader
                    .load_index(&search.column, &search.index_name, metrics)
                    .await?;
                let search_result = index.search(search.query.as_ref(), metrics).await?;
                let result = search_result_to_nullable(search_result);
                if index.results_are_row_addresses() {
                    index_loader.row_addr_result_to_row_ids(result).await
                } else {
                    Ok(result)
                }
            }
        }
    }

    /// Evaluates the scalar index expression, dropping nullable results
    #[instrument(level = "debug", skip_all)]
    pub async fn evaluate(
        &self,
        index_loader: &dyn ScalarIndexLoader,
        metrics: &dyn MetricsCollector,
    ) -> Result<IndexExprResult> {
        Ok(self
            .evaluate_nullable(index_loader, metrics)
            .await?
            .drop_nulls())
    }

    /// Convert this expression back to a DataFusion [`Expr`]
    pub fn to_expr(&self) -> Expr {
        match self {
            Self::Not(inner) => Expr::Not(inner.to_expr().into()),
            Self::And(lhs, rhs) => {
                let lhs = lhs.to_expr();
                let rhs = rhs.to_expr();
                lhs.and(rhs)
            }
            Self::Or(lhs, rhs) => {
                let lhs = lhs.to_expr();
                let rhs = rhs.to_expr();
                lhs.or(rhs)
            }
            Self::Query(search) => search.query.to_expr(search.column.clone()),
        }
    }

    /// Returns true if this expression requires a post-search recheck
    pub fn needs_recheck(&self) -> bool {
        match self {
            Self::Not(inner) => inner.needs_recheck(),
            Self::And(lhs, rhs) | Self::Or(lhs, rhs) => lhs.needs_recheck() || rhs.needs_recheck(),
            Self::Query(search) => search.needs_recheck,
        }
    }
}

impl std::fmt::Display for ScalarIndexExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Not(inner) => write!(f, "NOT({})", inner),
            Self::And(lhs, rhs) => write!(f, "AND({},{})", lhs, rhs),
            Self::Or(lhs, rhs) => write!(f, "OR({},{})", lhs, rhs),
            Self::Query(search) => write!(
                f,
                "[{}]@{}({})",
                search.query.format(&search.column),
                search.index_name,
                search.index_type
            ),
        }
    }
}
