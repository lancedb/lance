// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{ops::Bound, sync::Arc};

use arrow_schema::{DataType, Field};
use async_recursion::async_recursion;
use async_trait::async_trait;
use datafusion_common::ScalarValue;
use datafusion_expr::{Expr, Operator, ReturnFieldArgs, ScalarUDF, expr::Like};
use tokio::try_join;

use lance_core::{Error, Result};
use lance_select::{IndexExprResult, NullableIndexExprResult, NullableRowAddrMask};
use roaring::RoaringBitmap;
use tracing::instrument;

use crate::metrics::MetricsCollector;
use crate::scalar::{AnyQuery, ScalarIndex, SearchResult};

const MAX_DEPTH: usize = 500;

/// An indexed expression consists of a scalar index query with a post-scan filter
///
/// When a user wants to filter the data returned by a scan we may be able to use
/// one or more scalar indices to reduce the amount of data we load from the disk.
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
    pub fn maybe_or(self, other: Self) -> Option<Self> {
        let scalar_query = self.scalar_query?;
        let other_scalar_query = other.scalar_query?;
        let scalar_query = Some(ScalarIndexExpr::Or(
            Box::new(scalar_query),
            Box::new(other_scalar_query),
        ));

        let refine_expr = match (self.refine_expr, other.refine_expr) {
            (Some(_), Some(_)) | (Some(_), None) | (None, Some(_)) => {
                return None;
            }
            (None, None) => None,
        };
        Some(Self {
            scalar_query,
            refine_expr,
        })
    }

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

pub trait ScalarQueryParser: std::fmt::Debug + Send + Sync {
    /// Visit a between expression
    fn visit_between(
        &self,
        column: &str,
        low: &Bound<ScalarValue>,
        high: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression>;
    /// Visit an in list expression
    fn visit_in_list(&self, column: &str, in_list: &[ScalarValue]) -> Option<IndexedExpression>;
    /// Visit an is bool expression
    fn visit_is_bool(&self, column: &str, value: bool) -> Option<IndexedExpression>;
    /// Visit an is null expression
    fn visit_is_null(&self, column: &str) -> Option<IndexedExpression>;
    /// Visit a comparison expression
    fn visit_comparison(
        &self,
        column: &str,
        value: &ScalarValue,
        op: &Operator,
    ) -> Option<IndexedExpression>;
    /// Visit a scalar function expression
    fn visit_scalar_function(
        &self,
        column: &str,
        data_type: &DataType,
        func: &ScalarUDF,
        args: &[Expr],
    ) -> Option<IndexedExpression>;

    /// Visit a LIKE expression
    fn visit_like(
        &self,
        _column: &str,
        _like: &Like,
        _pattern: &ScalarValue,
    ) -> Option<IndexedExpression> {
        None
    }

    /// Visits a potential reference to a column
    fn is_valid_reference(&self, func: &Expr, data_type: &DataType) -> Option<DataType> {
        match func {
            Expr::Column(_) => Some(data_type.clone()),
            _ => None,
        }
    }
}

/// A generic parser that wraps multiple scalar query parsers
///
/// It will search each parser in order and return the first non-None result
#[derive(Debug)]
pub struct MultiQueryParser {
    parsers: Vec<Box<dyn ScalarQueryParser>>,
}

impl MultiQueryParser {
    /// Create a new MultiQueryParser with a single parser
    pub fn single(parser: Box<dyn ScalarQueryParser>) -> Self {
        Self {
            parsers: vec![parser],
        }
    }

    /// Add a new parser to the MultiQueryParser
    pub fn add(&mut self, other: Box<dyn ScalarQueryParser>) {
        self.parsers.push(other);
    }
}

impl ScalarQueryParser for MultiQueryParser {
    fn visit_between(
        &self,
        column: &str,
        low: &Bound<ScalarValue>,
        high: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression> {
        self.parsers
            .iter()
            .find_map(|parser| parser.visit_between(column, low, high))
    }
    fn visit_in_list(&self, column: &str, in_list: &[ScalarValue]) -> Option<IndexedExpression> {
        self.parsers
            .iter()
            .find_map(|parser| parser.visit_in_list(column, in_list))
    }
    fn visit_is_bool(&self, column: &str, value: bool) -> Option<IndexedExpression> {
        self.parsers
            .iter()
            .find_map(|parser| parser.visit_is_bool(column, value))
    }
    fn visit_is_null(&self, column: &str) -> Option<IndexedExpression> {
        self.parsers
            .iter()
            .find_map(|parser| parser.visit_is_null(column))
    }
    fn visit_comparison(
        &self,
        column: &str,
        value: &ScalarValue,
        op: &Operator,
    ) -> Option<IndexedExpression> {
        self.parsers
            .iter()
            .find_map(|parser| parser.visit_comparison(column, value, op))
    }
    fn visit_scalar_function(
        &self,
        column: &str,
        data_type: &DataType,
        func: &ScalarUDF,
        args: &[Expr],
    ) -> Option<IndexedExpression> {
        self.parsers
            .iter()
            .find_map(|parser| parser.visit_scalar_function(column, data_type, func, args))
    }
    fn visit_like(
        &self,
        column: &str,
        like: &Like,
        pattern: &ScalarValue,
    ) -> Option<IndexedExpression> {
        self.parsers
            .iter()
            .find_map(|parser| parser.visit_like(column, like, pattern))
    }
    fn is_valid_reference(&self, func: &Expr, data_type: &DataType) -> Option<DataType> {
        self.parsers
            .iter()
            .find_map(|parser| parser.is_valid_reference(func, data_type))
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

impl From<SearchResult> for NullableIndexExprResult {
    fn from(result: SearchResult) -> Self {
        match result {
            SearchResult::Exact(mask) => Self::exact(NullableRowAddrMask::AllowList(mask)),
            SearchResult::AtMost(mask) => Self::at_most(NullableRowAddrMask::AllowList(mask)),
            SearchResult::AtLeast(mask) => Self::at_least(NullableRowAddrMask::AllowList(mask)),
        }
    }
}

impl ScalarIndexExpr {
    /// Evaluates the scalar index expression
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
                Ok(search_result.into())
            }
        }
    }

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

    pub fn needs_recheck(&self) -> bool {
        match self {
            Self::Not(inner) => inner.needs_recheck(),
            Self::And(lhs, rhs) | Self::Or(lhs, rhs) => lhs.needs_recheck() || rhs.needs_recheck(),
            Self::Query(search) => search.needs_recheck,
        }
    }
}

/// A trait to be used in `apply_scalar_indices` to inform the function which columns are indexed
pub trait IndexInformationProvider {
    /// Check if an index exists for `col` and, if so, return the data type of col
    /// as well as a query parser that can parse queries for that column
    fn get_index(&self, col: &str) -> Option<(&DataType, &dyn ScalarQueryParser)>;

    /// The set of fragments covered by `(column, index_name)`.
    ///
    /// Returns `None` when the provider doesn't know — callers must treat
    /// that as "coverage unknown" rather than "covers everything". The
    /// default implementation always returns `None`, so providers that
    /// haven't been updated cannot accidentally claim full coverage.
    fn fragment_bitmap(&self, _column: &str, _index_name: &str) -> Option<RoaringBitmap> {
        None
    }
}

/// Attempt to split a filter expression into a search of scalar indexes and an
///   optional post-search refinement query
pub fn apply_scalar_indices(
    expr: Expr,
    index_info: &dyn IndexInformationProvider,
) -> Result<IndexedExpression> {
    let mut result =
        visit_node(&expr, index_info, 0)?.unwrap_or(IndexedExpression::refine_only(expr));
    if let Some(query) = result.scalar_query.as_mut() {
        populate_fragment_bitmaps(query, index_info);
    }
    Ok(result)
}

fn populate_fragment_bitmaps(
    expr: &mut ScalarIndexExpr,
    index_info: &dyn IndexInformationProvider,
) {
    match expr {
        ScalarIndexExpr::Not(inner) => populate_fragment_bitmaps(inner, index_info),
        ScalarIndexExpr::And(lhs, rhs) | ScalarIndexExpr::Or(lhs, rhs) => {
            populate_fragment_bitmaps(lhs, index_info);
            populate_fragment_bitmaps(rhs, index_info);
        }
        ScalarIndexExpr::Query(search) => {
            search.fragment_bitmap = index_info.fragment_bitmap(&search.column, &search.index_name);
        }
    }
}

fn visit_node(
    expr: &Expr,
    index_info: &dyn IndexInformationProvider,
    depth: usize,
) -> Result<Option<IndexedExpression>> {
    if depth >= MAX_DEPTH {
        return Err(Error::invalid_input(format!(
            "the filter expression is too long, lance limit the max number of conditions to {}",
            MAX_DEPTH
        )));
    }
    match expr {
        Expr::Between(between) => Ok(visit_between_expr(between, index_info)),
        Expr::Alias(alias) => visit_node(alias.expr.as_ref(), index_info, depth),
        Expr::Column(_) => Ok(visit_column(expr, index_info)),
        Expr::InList(in_list) => Ok(visit_in_list_expr(in_list, index_info)),
        Expr::IsFalse(expr) => Ok(visit_is_bool(expr.as_ref(), index_info, false)),
        Expr::IsTrue(expr) => Ok(visit_is_bool(expr.as_ref(), index_info, true)),
        Expr::IsNull(expr) => Ok(visit_is_null(expr.as_ref(), index_info, false)),
        Expr::IsNotNull(expr) => Ok(visit_is_null(expr.as_ref(), index_info, true)),
        Expr::Not(expr) => visit_not(expr.as_ref(), index_info, depth),
        Expr::BinaryExpr(binary_expr) => visit_binary_expr(binary_expr, index_info, depth),
        Expr::ScalarFunction(scalar_fn) => Ok(visit_scalar_fn(scalar_fn, index_info)),
        Expr::Like(like) => {
            if like.negated {
                Ok(None)
            } else {
                Ok(visit_like_expr(like, index_info))
            }
        }
        _ => Ok(None),
    }
}

fn maybe_indexed_column<'b>(
    expr: &Expr,
    index_info: &'b dyn IndexInformationProvider,
) -> Option<(String, DataType, &'b dyn ScalarQueryParser)> {
    match expr {
        Expr::Column(col) => {
            let col = col.name.as_str();
            let (data_type, parser) = index_info.get_index(col)?;
            if let Some(data_type) = parser.is_valid_reference(expr, data_type) {
                Some((col.to_string(), data_type, parser))
            } else {
                None
            }
        }
        Expr::ScalarFunction(udf) => {
            if udf.args.is_empty() {
                return None;
            }
            let col = match &udf.args[0] {
                Expr::Column(col) => col.name.as_str(),
                _ => return None,
            };
            let (data_type, parser) = index_info.get_index(col)?;
            if let Some(data_type) = parser.is_valid_reference(expr, data_type) {
                Some((col.to_string(), data_type, parser))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn visit_between_expr(
    between: &datafusion_expr::Between,
    index_info: &dyn IndexInformationProvider,
) -> Option<IndexedExpression> {
    use std::ops::Bound;
    let (column, col_type, query_parser) = maybe_indexed_column(&between.expr, index_info)?;
    let low = maybe_scalar(&between.low, &col_type)?;
    let high = maybe_scalar(&between.high, &col_type)?;

    let indexed_expr =
        query_parser.visit_between(&column, &Bound::Included(low), &Bound::Included(high))?;

    if between.negated {
        indexed_expr.maybe_not()
    } else {
        Some(indexed_expr)
    }
}

fn visit_in_list_expr(
    in_list: &datafusion_expr::expr::InList,
    index_info: &dyn IndexInformationProvider,
) -> Option<IndexedExpression> {
    let (column, col_type, query_parser) = maybe_indexed_column(&in_list.expr, index_info)?;
    let values = maybe_scalar_list(&in_list.list, &col_type)?;

    let indexed_expr = query_parser.visit_in_list(&column, &values)?;

    if in_list.negated {
        indexed_expr.maybe_not()
    } else {
        Some(indexed_expr)
    }
}

fn visit_is_bool(
    expr: &Expr,
    index_info: &dyn IndexInformationProvider,
    value: bool,
) -> Option<IndexedExpression> {
    let (column, col_type, query_parser) = maybe_indexed_column(expr, index_info)?;
    if col_type != DataType::Boolean {
        None
    } else {
        query_parser.visit_is_bool(&column, value)
    }
}

fn visit_column(
    col: &Expr,
    index_info: &dyn IndexInformationProvider,
) -> Option<IndexedExpression> {
    let (column, col_type, query_parser) = maybe_indexed_column(col, index_info)?;
    if col_type != DataType::Boolean {
        None
    } else {
        query_parser.visit_is_bool(&column, true)
    }
}

fn visit_is_null(
    expr: &Expr,
    index_info: &dyn IndexInformationProvider,
    negated: bool,
) -> Option<IndexedExpression> {
    let (column, _, query_parser) = maybe_indexed_column(expr, index_info)?;
    let indexed_expr = query_parser.visit_is_null(&column)?;
    if negated {
        indexed_expr.maybe_not()
    } else {
        Some(indexed_expr)
    }
}

fn visit_not(
    expr: &Expr,
    index_info: &dyn IndexInformationProvider,
    depth: usize,
) -> Result<Option<IndexedExpression>> {
    let node = visit_node(expr, index_info, depth + 1)?;
    Ok(node.and_then(|node| node.maybe_not()))
}

fn visit_comparison(
    expr: &datafusion_expr::BinaryExpr,
    index_info: &dyn IndexInformationProvider,
) -> Option<IndexedExpression> {
    let left_col = maybe_indexed_column(&expr.left, index_info);
    if let Some((column, col_type, query_parser)) = left_col {
        let scalar = maybe_scalar(&expr.right, &col_type)?;
        query_parser.visit_comparison(&column, &scalar, &expr.op)
    } else {
        None
    }
}

fn maybe_range(
    expr: &datafusion_expr::BinaryExpr,
    index_info: &dyn IndexInformationProvider,
) -> Option<IndexedExpression> {
    let left_expr = match expr.left.as_ref() {
        Expr::BinaryExpr(binary_expr) => Some(binary_expr),
        _ => None,
    }?;
    let right_expr = match expr.right.as_ref() {
        Expr::BinaryExpr(binary_expr) => Some(binary_expr),
        _ => None,
    }?;

    let (left_col, dt, parser) = maybe_indexed_column(&left_expr.left, index_info)?;
    let right_col = match &right_expr.left.as_ref() {
        Expr::Column(col) => col.name.as_str().to_string(),
        _ => return None,
    };

    if left_col != right_col {
        return None;
    }

    let left_value = maybe_scalar(&left_expr.right, &dt)?;
    let right_value = maybe_scalar(&right_expr.right, &dt)?;

    let (low, high) = match (left_expr.op, right_expr.op) {
        (Operator::GtEq, Operator::LtEq) => {
            (Bound::Included(left_value), Bound::Included(right_value))
        }
        (Operator::GtEq, Operator::Lt) => {
            (Bound::Included(left_value), Bound::Excluded(right_value))
        }
        (Operator::Gt, Operator::LtEq) => {
            (Bound::Excluded(left_value), Bound::Included(right_value))
        }
        (Operator::Gt, Operator::Lt) => (Bound::Excluded(left_value), Bound::Excluded(right_value)),
        (Operator::LtEq, Operator::GtEq) => {
            (Bound::Included(right_value), Bound::Included(left_value))
        }
        (Operator::LtEq, Operator::Gt) => {
            (Bound::Excluded(right_value), Bound::Included(left_value))
        }
        (Operator::Lt, Operator::GtEq) => {
            (Bound::Included(right_value), Bound::Excluded(left_value))
        }
        (Operator::Lt, Operator::Gt) => (Bound::Excluded(right_value), Bound::Excluded(left_value)),
        _ => return None,
    };

    parser.visit_between(&left_col, &low, &high)
}

fn visit_and(
    expr: &datafusion_expr::BinaryExpr,
    index_info: &dyn IndexInformationProvider,
    depth: usize,
) -> Result<Option<IndexedExpression>> {
    if let Some(range_expr) = maybe_range(expr, index_info) {
        return Ok(Some(range_expr));
    }

    let left = visit_node(&expr.left, index_info, depth + 1)?;
    let right = visit_node(&expr.right, index_info, depth + 1)?;
    Ok(match (left, right) {
        (Some(left), Some(right)) => Some(left.and(right)),
        (Some(left), None) => Some(left.refine((*expr.right).clone())),
        (None, Some(right)) => Some(right.refine((*expr.left).clone())),
        (None, None) => None,
    })
}

fn visit_or(
    expr: &datafusion_expr::BinaryExpr,
    index_info: &dyn IndexInformationProvider,
    depth: usize,
) -> Result<Option<IndexedExpression>> {
    let left = visit_node(&expr.left, index_info, depth + 1)?;
    let right = visit_node(&expr.right, index_info, depth + 1)?;
    Ok(match (left, right) {
        (Some(left), Some(right)) => left.maybe_or(right),
        (Some(_), None) | (None, Some(_)) => None,
        (None, None) => None,
    })
}

fn visit_binary_expr(
    expr: &datafusion_expr::BinaryExpr,
    index_info: &dyn IndexInformationProvider,
    depth: usize,
) -> Result<Option<IndexedExpression>> {
    match &expr.op {
        Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq | Operator::Eq => {
            Ok(visit_comparison(expr, index_info))
        }
        Operator::NotEq => Ok(visit_comparison(expr, index_info).and_then(|node| node.maybe_not())),
        Operator::And => visit_and(expr, index_info, depth),
        Operator::Or => visit_or(expr, index_info, depth),
        _ => Ok(None),
    }
}

fn visit_scalar_fn(
    scalar_fn: &datafusion_expr::expr::ScalarFunction,
    index_info: &dyn IndexInformationProvider,
) -> Option<IndexedExpression> {
    if scalar_fn.args.is_empty() {
        return None;
    }
    let (col, data_type, query_parser) = maybe_indexed_column(&scalar_fn.args[0], index_info)?;
    query_parser.visit_scalar_function(&col, &data_type, &scalar_fn.func, &scalar_fn.args)
}

fn visit_like_expr(
    like: &Like,
    index_info: &dyn IndexInformationProvider,
) -> Option<IndexedExpression> {
    let (column, _, query_parser) = maybe_indexed_column(&like.expr, index_info)?;

    let pattern = match like.pattern.as_ref() {
        Expr::Literal(scalar, _) => scalar.clone(),
        _ => return None,
    };

    query_parser.visit_like(&column, like, &pattern)
}

fn maybe_scalar(expr: &Expr, expected_type: &DataType) -> Option<ScalarValue> {
    match expr {
        Expr::Literal(value, _) => coerce_scalar(value, expected_type),
        Expr::Cast(cast) => match cast.expr.as_ref() {
            Expr::Literal(value, _) => {
                let casted = value.cast_to(&cast.data_type).ok()?;
                coerce_scalar(&casted, expected_type)
            }
            _ => None,
        },
        // arrow_cast(value, 'type') is represented as a ScalarFunction, not a Cast.
        // This commonly arises for types not expressible in SQL literals (e.g. fixed-size-binary).
        Expr::ScalarFunction(scalar_function) => {
            if scalar_function.name() == "arrow_cast" && scalar_function.args.len() == 2 {
                match (&scalar_function.args[0], &scalar_function.args[1]) {
                    (Expr::Literal(value, _), Expr::Literal(cast_type, _)) => {
                        let target_field = scalar_function
                            .func
                            .return_field_from_args(ReturnFieldArgs {
                                arg_fields: &[
                                    Arc::new(Field::new("expression", value.data_type(), false)),
                                    Arc::new(Field::new("datatype", cast_type.data_type(), false)),
                                ],
                                scalar_arguments: &[Some(value), Some(cast_type)],
                            })
                            .ok()?;
                        let casted = value.cast_to(target_field.data_type()).ok()?;
                        coerce_scalar(&casted, expected_type)
                    }
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

fn coerce_scalar(value: &ScalarValue, expected_type: &DataType) -> Option<ScalarValue> {
    if value.data_type() == *expected_type {
        return Some(value.clone());
    }
    value.cast_to(expected_type).ok()
}

fn maybe_scalar_list(exprs: &[Expr], expected_type: &DataType) -> Option<Vec<ScalarValue>> {
    let mut scalar_values = Vec::with_capacity(exprs.len());
    for expr in exprs {
        match maybe_scalar(expr, expected_type) {
            Some(scalar_val) => {
                scalar_values.push(scalar_val);
            }
            None => {
                return None;
            }
        }
    }
    Some(scalar_values)
}
