// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{ops::Bound, sync::Arc};

use arrow_array::Array;
use arrow_schema::{DataType, Field};
use async_recursion::async_recursion;
use async_trait::async_trait;
use datafusion_common::ScalarValue;
use datafusion_expr::{
    expr::{InList, ScalarFunction},
    Between, BinaryExpr, Expr, Operator, ReturnFieldArgs, ScalarUDF,
};

use futures::join;
use lance_core::{utils::mask::RowIdMask, Result};
use lance_datafusion::{expr::safe_coerce_scalar, planner::Planner};
use tracing::instrument;

use super::{
    AnyQuery, LabelListQuery, MetricsCollector, SargableQuery, ScalarIndex, SearchResult, TextQuery,
};

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

pub trait ScalarQueryParser: std::fmt::Debug + Send + Sync {
    fn visit_between(
        &self,
        column: &str,
        low: &Bound<ScalarValue>,
        high: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression>;
    fn visit_in_list(&self, column: &str, in_list: &[ScalarValue]) -> Option<IndexedExpression>;
    fn visit_is_bool(&self, column: &str, value: bool) -> Option<IndexedExpression>;
    fn visit_is_null(&self, column: &str) -> Option<IndexedExpression>;
    fn visit_comparison(
        &self,
        column: &str,
        value: &ScalarValue,
        op: &Operator,
    ) -> Option<IndexedExpression>;
    fn visit_scalar_function(
        &self,
        column: &str,
        data_type: &DataType,
        func: &ScalarUDF,
        args: &[Expr],
    ) -> Option<IndexedExpression>;
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
}

/// A parser for indices that handle SARGable queries
#[derive(Debug)]
pub struct SargableQueryParser {
    index_name: String,
}

impl SargableQueryParser {
    pub fn new(index_name: String) -> Self {
        Self { index_name }
    }
}

impl ScalarQueryParser for SargableQueryParser {
    fn visit_between(
        &self,
        column: &str,
        low: &Bound<ScalarValue>,
        high: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression> {
        let query = SargableQuery::Range(low.clone(), high.clone());
        Some(IndexedExpression::index_query(
            column.to_string(),
            self.index_name.clone(),
            Arc::new(query),
        ))
    }

    fn visit_in_list(&self, column: &str, in_list: &[ScalarValue]) -> Option<IndexedExpression> {
        let query = SargableQuery::IsIn(in_list.to_vec());
        Some(IndexedExpression::index_query(
            column.to_string(),
            self.index_name.clone(),
            Arc::new(query),
        ))
    }

    fn visit_is_bool(&self, column: &str, value: bool) -> Option<IndexedExpression> {
        Some(IndexedExpression::index_query(
            column.to_string(),
            self.index_name.clone(),
            Arc::new(SargableQuery::Equals(ScalarValue::Boolean(Some(value)))),
        ))
    }

    fn visit_is_null(&self, column: &str) -> Option<IndexedExpression> {
        Some(IndexedExpression::index_query(
            column.to_string(),
            self.index_name.clone(),
            Arc::new(SargableQuery::IsNull()),
        ))
    }

    fn visit_comparison(
        &self,
        column: &str,
        value: &ScalarValue,
        op: &Operator,
    ) -> Option<IndexedExpression> {
        let query = match op {
            Operator::Lt => SargableQuery::Range(Bound::Unbounded, Bound::Excluded(value.clone())),
            Operator::LtEq => {
                SargableQuery::Range(Bound::Unbounded, Bound::Included(value.clone()))
            }
            Operator::Gt => SargableQuery::Range(Bound::Excluded(value.clone()), Bound::Unbounded),
            Operator::GtEq => {
                SargableQuery::Range(Bound::Included(value.clone()), Bound::Unbounded)
            }
            Operator::Eq => SargableQuery::Equals(value.clone()),
            // This will be negated by the caller
            Operator::NotEq => SargableQuery::Equals(value.clone()),
            _ => unreachable!(),
        };
        Some(IndexedExpression::index_query(
            column.to_string(),
            self.index_name.clone(),
            Arc::new(query),
        ))
    }

    fn visit_scalar_function(
        &self,
        _: &str,
        _: &DataType,
        _: &ScalarUDF,
        _: &[Expr],
    ) -> Option<IndexedExpression> {
        None
    }
}

/// A parser for indices that handle label list queries
#[derive(Debug)]
pub struct LabelListQueryParser {
    index_name: String,
}

impl LabelListQueryParser {
    pub fn new(index_name: String) -> Self {
        Self { index_name }
    }
}

impl ScalarQueryParser for LabelListQueryParser {
    fn visit_between(
        &self,
        _: &str,
        _: &Bound<ScalarValue>,
        _: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_in_list(&self, _: &str, _: &[ScalarValue]) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_bool(&self, _: &str, _: bool) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_null(&self, _: &str) -> Option<IndexedExpression> {
        None
    }

    fn visit_comparison(
        &self,
        _: &str,
        _: &ScalarValue,
        _: &Operator,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_scalar_function(
        &self,
        column: &str,
        data_type: &DataType,
        func: &ScalarUDF,
        args: &[Expr],
    ) -> Option<IndexedExpression> {
        if args.len() != 2 {
            return None;
        }
        let label_list = maybe_scalar(&args[1], data_type)?;
        if let ScalarValue::List(list_arr) = label_list {
            let list_values = list_arr.values();
            let mut scalars = Vec::with_capacity(list_values.len());
            for idx in 0..list_values.len() {
                scalars.push(ScalarValue::try_from_array(list_values.as_ref(), idx).ok()?);
            }
            if func.name() == "array_has_all" {
                let query = LabelListQuery::HasAllLabels(scalars);
                Some(IndexedExpression::index_query(
                    column.to_string(),
                    self.index_name.clone(),
                    Arc::new(query),
                ))
            } else if func.name() == "array_has_any" {
                let query = LabelListQuery::HasAnyLabel(scalars);
                Some(IndexedExpression::index_query(
                    column.to_string(),
                    self.index_name.clone(),
                    Arc::new(query),
                ))
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// A parser for indices that handle string contains queries
#[derive(Debug, Clone)]
pub struct TextQueryParser {
    index_name: String,
}

impl TextQueryParser {
    pub fn new(index_name: String) -> Self {
        Self { index_name }
    }
}

impl ScalarQueryParser for TextQueryParser {
    fn visit_between(
        &self,
        _: &str,
        _: &Bound<ScalarValue>,
        _: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_in_list(&self, _: &str, _: &[ScalarValue]) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_bool(&self, _: &str, _: bool) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_null(&self, _: &str) -> Option<IndexedExpression> {
        None
    }

    fn visit_comparison(
        &self,
        _: &str,
        _: &ScalarValue,
        _: &Operator,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_scalar_function(
        &self,
        column: &str,
        data_type: &DataType,
        func: &ScalarUDF,
        args: &[Expr],
    ) -> Option<IndexedExpression> {
        if args.len() != 2 {
            return None;
        }
        let scalar = maybe_scalar(&args[1], data_type)?;
        match scalar {
            ScalarValue::Utf8(Some(scalar_str)) | ScalarValue::LargeUtf8(Some(scalar_str)) => {
                if func.name() == "contains" {
                    let query = TextQuery::StringContains(scalar_str);
                    Some(IndexedExpression::index_query(
                        column.to_string(),
                        self.index_name.clone(),
                        Arc::new(query),
                    ))
                } else {
                    None
                }
            }
            _ => {
                // If the scalar is not a string, we cannot handle it
                None
            }
        }
    }
}

/// A parser for indices that handle queries with the contains_tokens function
#[derive(Debug, Clone)]
pub struct FtsQueryParser {
    index_name: String,
}

impl FtsQueryParser {
    pub fn new(index_name: String) -> Self {
        Self { index_name }
    }
}

impl ScalarQueryParser for FtsQueryParser {
    fn visit_between(
        &self,
        _: &str,
        _: &Bound<ScalarValue>,
        _: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_in_list(&self, _: &str, _: &[ScalarValue]) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_bool(&self, _: &str, _: bool) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_null(&self, _: &str) -> Option<IndexedExpression> {
        None
    }

    fn visit_comparison(
        &self,
        _: &str,
        _: &ScalarValue,
        _: &Operator,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_scalar_function(
        &self,
        column: &str,
        data_type: &DataType,
        func: &ScalarUDF,
        args: &[Expr],
    ) -> Option<IndexedExpression> {
        if args.len() != 2 {
            return None;
        }
        let scalar = maybe_scalar(&args[1], data_type)?;
        if let ScalarValue::Utf8(Some(scalar_str)) = scalar {
            // TODO(https://github.com/lancedb/lance/issues/3855):
            //
            // Create the contains_tokens UDF
            if func.name() == "contains_tokens" {
                let query = TextQuery::StringContains(scalar_str);
                Some(IndexedExpression::index_query(
                    column.to_string(),
                    self.index_name.clone(),
                    Arc::new(query),
                ))
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl IndexedExpression {
    /// Create an expression that only does refine
    fn refine_only(refine_expr: Expr) -> Self {
        Self {
            scalar_query: None,
            refine_expr: Some(refine_expr),
        }
    }

    /// Create an expression that is only an index query
    fn index_query(column: String, index_name: String, query: Arc<dyn AnyQuery>) -> Self {
        Self {
            scalar_query: Some(ScalarIndexExpr::Query(ScalarIndexSearch {
                column,
                index_name,
                query,
            })),
            refine_expr: None,
        }
    }

    /// Try and negate the expression
    ///
    /// If the expression contains both an index query and a refine expression then it
    /// cannot be negated today and None will be returned (we give up trying to use indices)
    fn maybe_not(self) -> Option<Self> {
        match (self.scalar_query, self.refine_expr) {
            (Some(_), Some(_)) => None,
            (Some(scalar_query), None) => Some(Self {
                scalar_query: Some(ScalarIndexExpr::Not(Box::new(scalar_query))),
                refine_expr: None,
            }),
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
    fn and(self, other: Self) -> Self {
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
    fn maybe_or(self, other: Self) -> Option<Self> {
        // If either expression is missing a scalar_query then we need to load all rows from
        // the database and so we short-circuit and return None
        let scalar_query = self.scalar_query?;
        let other_scalar_query = other.scalar_query?;
        let scalar_query = Some(ScalarIndexExpr::Or(
            Box::new(scalar_query),
            Box::new(other_scalar_query),
        ));

        let refine_expr = match (self.refine_expr, other.refine_expr) {
            // TODO
            //
            // To handle these cases we need a way of going back from a scalar expression query to a logical DF expression (perhaps
            // we can store the expression that led to the creation of the query)
            //
            // For example, imagine we have something like "(color == 'blue' AND size < 20) OR (color == 'green' AND size < 50)"
            //
            // We can do an indexed load of all rows matching "color == 'blue' OR color == 'green'" but then we need to
            // refine that load with the full original expression which, at the moment, we no longer have.
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

    fn refine(self, expr: Expr) -> Self {
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
    /// The query to search for
    pub query: Arc<dyn AnyQuery>,
}

impl PartialEq for ScalarIndexSearch {
    fn eq(&self, other: &Self) -> bool {
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
    Not(Box<ScalarIndexExpr>),
    And(Box<ScalarIndexExpr>, Box<ScalarIndexExpr>),
    Or(Box<ScalarIndexExpr>, Box<ScalarIndexExpr>),
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
                "[{}]@{}",
                search.query.format(&search.column),
                search.index_name
            ),
        }
    }
}

pub enum IndexExprResult {
    // The answer is exactly the rows in the allow list minus the rows in the block list
    Exact(RowIdMask),
    // The answer is at most the rows in the allow list minus the rows in the block list
    // Some of the rows in the allow list may not be in the result and will need to be filtered
    // by a recheck.  Every row in the block list is definitely not in the result.
    AtMost(RowIdMask),
    // The answer is at least the rows in the allow list minus the rows in the block list
    // Some of the rows in the block list might be in the result.  Every row in the allow list is
    // definitely in the result.
    AtLeast(RowIdMask),
}

impl ScalarIndexExpr {
    /// Evaluates the scalar index expression
    ///
    /// This will result in loading one or more scalar indices and searching them
    ///
    /// TODO: We could potentially try and be smarter about reusing loaded indices for
    /// any situations where the session cache has been disabled.
    #[async_recursion]
    #[instrument(level = "debug", skip_all)]
    pub async fn evaluate(
        &self,
        index_loader: &dyn ScalarIndexLoader,
        metrics: &dyn MetricsCollector,
    ) -> Result<IndexExprResult> {
        match self {
            Self::Not(inner) => {
                let result = inner.evaluate(index_loader, metrics).await?;
                match result {
                    IndexExprResult::Exact(mask) => Ok(IndexExprResult::Exact(!mask)),
                    IndexExprResult::AtMost(mask) => Ok(IndexExprResult::AtLeast(!mask)),
                    IndexExprResult::AtLeast(mask) => Ok(IndexExprResult::AtMost(!mask)),
                }
            }
            Self::And(lhs, rhs) => {
                let lhs_result = lhs.evaluate(index_loader, metrics);
                let rhs_result = rhs.evaluate(index_loader, metrics);
                let (lhs_result, rhs_result) = join!(lhs_result, rhs_result);
                match (lhs_result?, rhs_result?) {
                    (IndexExprResult::Exact(lhs), IndexExprResult::Exact(rhs)) => {
                        Ok(IndexExprResult::Exact(lhs & rhs))
                    }
                    (IndexExprResult::Exact(lhs), IndexExprResult::AtMost(rhs))
                    | (IndexExprResult::AtMost(lhs), IndexExprResult::Exact(rhs)) => {
                        Ok(IndexExprResult::AtMost(lhs & rhs))
                    }
                    (IndexExprResult::Exact(lhs), IndexExprResult::AtLeast(_)) => {
                        // We could do better here, elements in both lhs and rhs are known
                        // to be true and don't require a recheck.  We only need to recheck
                        // elements in lhs that are not in rhs
                        Ok(IndexExprResult::AtMost(lhs))
                    }
                    (IndexExprResult::AtLeast(_), IndexExprResult::Exact(rhs)) => {
                        // We could do better here (see above)
                        Ok(IndexExprResult::AtMost(rhs))
                    }
                    (IndexExprResult::AtMost(lhs), IndexExprResult::AtMost(rhs)) => {
                        Ok(IndexExprResult::AtMost(lhs & rhs))
                    }
                    (IndexExprResult::AtLeast(lhs), IndexExprResult::AtLeast(rhs)) => {
                        Ok(IndexExprResult::AtLeast(lhs & rhs))
                    }
                    (IndexExprResult::AtLeast(_), IndexExprResult::AtMost(rhs)) => {
                        Ok(IndexExprResult::AtMost(rhs))
                    }
                    (IndexExprResult::AtMost(lhs), IndexExprResult::AtLeast(_)) => {
                        Ok(IndexExprResult::AtMost(lhs))
                    }
                }
            }
            Self::Or(lhs, rhs) => {
                let lhs_result = lhs.evaluate(index_loader, metrics);
                let rhs_result = rhs.evaluate(index_loader, metrics);
                let (lhs_result, rhs_result) = join!(lhs_result, rhs_result);
                match (lhs_result?, rhs_result?) {
                    (IndexExprResult::Exact(lhs), IndexExprResult::Exact(rhs)) => {
                        Ok(IndexExprResult::Exact(lhs | rhs))
                    }
                    (IndexExprResult::Exact(lhs), IndexExprResult::AtMost(rhs))
                    | (IndexExprResult::AtMost(lhs), IndexExprResult::Exact(rhs)) => {
                        // We could do better here.  Elements in the exact side don't need
                        // re-check.  We only need to recheck elements exclusively in the
                        // at-most side
                        Ok(IndexExprResult::AtMost(lhs | rhs))
                    }
                    (IndexExprResult::Exact(lhs), IndexExprResult::AtLeast(rhs)) => {
                        Ok(IndexExprResult::AtLeast(lhs | rhs))
                    }
                    (IndexExprResult::AtLeast(lhs), IndexExprResult::Exact(rhs)) => {
                        Ok(IndexExprResult::AtLeast(lhs | rhs))
                    }
                    (IndexExprResult::AtMost(lhs), IndexExprResult::AtMost(rhs)) => {
                        Ok(IndexExprResult::AtMost(lhs | rhs))
                    }
                    (IndexExprResult::AtLeast(lhs), IndexExprResult::AtLeast(rhs)) => {
                        Ok(IndexExprResult::AtLeast(lhs | rhs))
                    }
                    (IndexExprResult::AtLeast(lhs), IndexExprResult::AtMost(_)) => {
                        Ok(IndexExprResult::AtLeast(lhs))
                    }
                    (IndexExprResult::AtMost(_), IndexExprResult::AtLeast(rhs)) => {
                        Ok(IndexExprResult::AtLeast(rhs))
                    }
                }
            }
            Self::Query(search) => {
                let index = index_loader
                    .load_index(&search.column, &search.index_name, metrics)
                    .await?;
                let search_result = index.search(search.query.as_ref(), metrics).await?;
                match search_result {
                    SearchResult::Exact(matching_row_ids) => {
                        Ok(IndexExprResult::Exact(RowIdMask {
                            block_list: None,
                            allow_list: Some(matching_row_ids),
                        }))
                    }
                    SearchResult::AtMost(row_ids) => Ok(IndexExprResult::AtMost(RowIdMask {
                        block_list: None,
                        allow_list: Some(row_ids),
                    })),
                    SearchResult::AtLeast(row_ids) => Ok(IndexExprResult::AtLeast(RowIdMask {
                        block_list: None,
                        allow_list: Some(row_ids),
                    })),
                }
            }
        }
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
            Self::Query(search) => search.query.needs_recheck(),
        }
    }
}

// Extract a column from the expression, if it is a column, or None
fn maybe_column(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(col) => Some(&col.name),
        _ => None,
    }
}

// Extract a column from the expression, if it is a column, and we have an index for that column, or None
fn maybe_indexed_column<'a, 'b>(
    expr: &'a Expr,
    index_info: &'b dyn IndexInformationProvider,
) -> Option<(&'a str, &'b DataType, &'b dyn ScalarQueryParser)> {
    let col = maybe_column(expr)?;
    let data_type = index_info.get_index(col);
    data_type.map(|(ty, parser)| (col, ty, parser))
}

// Extract a literal scalar value from an expression, if it is a literal, or None
fn maybe_scalar(expr: &Expr, expected_type: &DataType) -> Option<ScalarValue> {
    match expr {
        Expr::Literal(value, _) => safe_coerce_scalar(value, expected_type),
        // Some literals can't be expressed in datafusion's SQL and can only be expressed with
        // a cast.  For example, there is no way to express a fixed-size-binary literal (which is
        // commonly used for UUID).  As a result the expression could look like...
        //
        // col = arrow_cast(value, 'fixed_size_binary(16)')
        //
        // In this case we need to extract the value, apply the cast, and then test the casted value
        Expr::Cast(cast) => match cast.expr.as_ref() {
            Expr::Literal(value, _) => {
                let casted = value.cast_to(&cast.data_type).ok()?;
                safe_coerce_scalar(&casted, expected_type)
            }
            _ => None,
        },
        Expr::ScalarFunction(scalar_function) => {
            if scalar_function.name() == "arrow_cast" {
                if scalar_function.args.len() != 2 {
                    return None;
                }
                match (&scalar_function.args[0], &scalar_function.args[1]) {
                    (Expr::Literal(value, _), Expr::Literal(cast_type, _)) => {
                        let target_type = scalar_function
                            .func
                            .return_field_from_args(ReturnFieldArgs {
                                arg_fields: &[
                                    Arc::new(Field::new("expression", value.data_type(), false)),
                                    Arc::new(Field::new("datatype", cast_type.data_type(), false)),
                                ],
                                scalar_arguments: &[Some(value), Some(cast_type)],
                            })
                            .ok()?;
                        let casted = value.cast_to(target_type.data_type()).ok()?;
                        safe_coerce_scalar(&casted, expected_type)
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

// Extract a list of scalar values from an expression, if it is a list of scalar values, or None
fn maybe_scalar_list(exprs: &Vec<Expr>, expected_type: &DataType) -> Option<Vec<ScalarValue>> {
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

fn visit_between(
    between: &Between,
    index_info: &dyn IndexInformationProvider,
) -> Option<IndexedExpression> {
    let (column, col_type, query_parser) = maybe_indexed_column(&between.expr, index_info)?;
    let low = maybe_scalar(&between.low, col_type)?;
    let high = maybe_scalar(&between.high, col_type)?;

    let indexed_expr =
        query_parser.visit_between(column, &Bound::Included(low), &Bound::Included(high))?;

    if between.negated {
        indexed_expr.maybe_not()
    } else {
        Some(indexed_expr)
    }
}

fn visit_in_list(
    in_list: &InList,
    index_info: &dyn IndexInformationProvider,
) -> Option<IndexedExpression> {
    let (column, col_type, query_parser) = maybe_indexed_column(&in_list.expr, index_info)?;
    let values = maybe_scalar_list(&in_list.list, col_type)?;

    let indexed_expr = query_parser.visit_in_list(column, &values)?;

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
    if *col_type != DataType::Boolean {
        None
    } else {
        query_parser.visit_is_bool(column, value)
    }
}

// A column can be a valid indexed expression if the column is boolean (e.g. 'WHERE on_sale')
fn visit_column(
    col: &Expr,
    index_info: &dyn IndexInformationProvider,
) -> Option<IndexedExpression> {
    let (column, col_type, query_parser) = maybe_indexed_column(col, index_info)?;
    if *col_type != DataType::Boolean {
        None
    } else {
        query_parser.visit_is_bool(column, true)
    }
}

fn visit_is_null(
    expr: &Expr,
    index_info: &dyn IndexInformationProvider,
    negated: bool,
) -> Option<IndexedExpression> {
    let (column, _, query_parser) = maybe_indexed_column(expr, index_info)?;
    let indexed_expr = query_parser.visit_is_null(column)?;
    if negated {
        indexed_expr.maybe_not()
    } else {
        Some(indexed_expr)
    }
}

fn visit_not(expr: &Expr, index_info: &dyn IndexInformationProvider) -> Option<IndexedExpression> {
    let node = visit_node(expr, index_info)?;
    node.maybe_not()
}

fn visit_comparison(
    expr: &BinaryExpr,
    index_info: &dyn IndexInformationProvider,
) -> Option<IndexedExpression> {
    let left_col = maybe_indexed_column(&expr.left, index_info);
    if let Some((column, col_type, query_parser)) = left_col {
        let scalar = maybe_scalar(&expr.right, col_type)?;
        query_parser.visit_comparison(column, &scalar, &expr.op)
    } else {
        // Datafusion's query simplifier will canonicalize expressions and so we shouldn't reach this case.  If, for some reason, we
        // do reach this case we can handle it in the future by inverting expr.op and swapping the left and right sides
        None
    }
}

fn maybe_range(
    expr: &BinaryExpr,
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
    let right_col = maybe_column(&right_expr.left)?;

    if left_col != right_col {
        return None;
    }

    let left_value = maybe_scalar(&left_expr.right, dt)?;
    let right_value = maybe_scalar(&right_expr.right, dt)?;

    let (low, high) = match (left_expr.op, right_expr.op) {
        // x >= a && x <= b
        (Operator::GtEq, Operator::LtEq) => {
            (Bound::Included(left_value), Bound::Included(right_value))
        }
        // x >= a && x < b
        (Operator::GtEq, Operator::Lt) => {
            (Bound::Included(left_value), Bound::Excluded(right_value))
        }
        // x > a && x <= b
        (Operator::Gt, Operator::LtEq) => {
            (Bound::Excluded(left_value), Bound::Included(right_value))
        }
        // x > a && x < b
        (Operator::Gt, Operator::Lt) => (Bound::Excluded(left_value), Bound::Excluded(right_value)),
        // x <= a && x >= b
        (Operator::LtEq, Operator::GtEq) => {
            (Bound::Included(right_value), Bound::Included(left_value))
        }
        // x <= a && x > b
        (Operator::LtEq, Operator::Gt) => {
            (Bound::Included(right_value), Bound::Excluded(left_value))
        }
        // x < a && x >= b
        (Operator::Lt, Operator::GtEq) => {
            (Bound::Excluded(right_value), Bound::Included(left_value))
        }
        // x < a && x > b
        (Operator::Lt, Operator::Gt) => (Bound::Excluded(right_value), Bound::Excluded(left_value)),
        _ => return None,
    };

    parser.visit_between(left_col, &low, &high)
}

fn visit_and(
    expr: &BinaryExpr,
    index_info: &dyn IndexInformationProvider,
) -> Option<IndexedExpression> {
    // Many scalar indices can efficiently handle a BETWEEN query as a single search and this
    // can be much more efficient than two separate range queries.  As an optimization we check
    // to see if this is a between query and, if so, we handle it as a single query
    //
    // Note: We can't rely on users writing the SQL BETWEEN operator because:
    //   * Some users won't realize it's an option or a good idea
    //   * Datafusion's simplifier will rewrite the BETWEEN operator into two separate range queries
    if let Some(range_expr) = maybe_range(expr, index_info) {
        return Some(range_expr);
    }

    let left = visit_node(&expr.left, index_info);
    let right = visit_node(&expr.right, index_info);
    match (left, right) {
        (Some(left), Some(right)) => Some(left.and(right)),
        (Some(left), None) => Some(left.refine((*expr.right).clone())),
        (None, Some(right)) => Some(right.refine((*expr.left).clone())),
        (None, None) => None,
    }
}

fn visit_or(
    expr: &BinaryExpr,
    index_info: &dyn IndexInformationProvider,
) -> Option<IndexedExpression> {
    let left = visit_node(&expr.left, index_info);
    let right = visit_node(&expr.right, index_info);
    match (left, right) {
        (Some(left), Some(right)) => left.maybe_or(right),
        // If one side can use an index and the other side cannot then
        // we must abandon the entire thing.  For example, consider the
        // query "color == 'blue' or size > 10" where color is indexed but
        // size is not.  It's entirely possible that size > 10 matches every
        // row in our database.  There is nothing we can do except a full scan
        (Some(_), None) => None,
        (None, Some(_)) => None,
        (None, None) => None,
    }
}

fn visit_binary_expr(
    expr: &BinaryExpr,
    index_info: &dyn IndexInformationProvider,
) -> Option<IndexedExpression> {
    match &expr.op {
        Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq | Operator::Eq => {
            visit_comparison(expr, index_info)
        }
        // visit_comparison will maybe create an Eq query which we negate
        Operator::NotEq => visit_comparison(expr, index_info).and_then(|node| node.maybe_not()),
        Operator::And => visit_and(expr, index_info),
        Operator::Or => visit_or(expr, index_info),
        _ => None,
    }
}

fn visit_scalar_fn(
    scalar_fn: &ScalarFunction,
    index_info: &dyn IndexInformationProvider,
) -> Option<IndexedExpression> {
    if scalar_fn.args.is_empty() {
        return None;
    }
    let (col, data_type, query_parser) = maybe_indexed_column(&scalar_fn.args[0], index_info)?;
    query_parser.visit_scalar_function(col, data_type, &scalar_fn.func, &scalar_fn.args)
}

fn visit_node(expr: &Expr, index_info: &dyn IndexInformationProvider) -> Option<IndexedExpression> {
    match expr {
        Expr::Between(between) => visit_between(between, index_info),
        Expr::Column(_) => visit_column(expr, index_info),
        Expr::InList(in_list) => visit_in_list(in_list, index_info),
        Expr::IsFalse(expr) => visit_is_bool(expr.as_ref(), index_info, false),
        Expr::IsTrue(expr) => visit_is_bool(expr.as_ref(), index_info, true),
        Expr::IsNull(expr) => visit_is_null(expr.as_ref(), index_info, false),
        Expr::IsNotNull(expr) => visit_is_null(expr.as_ref(), index_info, true),
        Expr::Not(expr) => visit_not(expr.as_ref(), index_info),
        Expr::BinaryExpr(binary_expr) => visit_binary_expr(binary_expr, index_info),
        Expr::ScalarFunction(scalar_fn) => visit_scalar_fn(scalar_fn, index_info),
        _ => None,
    }
}

/// A trait to be used in `apply_scalar_indices` to inform the function which columns are indexeds
pub trait IndexInformationProvider {
    /// Check if an index exists for `col` and, if so, return the data type of col
    /// as well as a query parser that can parse queries for that column
    fn get_index(&self, col: &str) -> Option<(&DataType, &dyn ScalarQueryParser)>;
}

/// Attempt to split a filter expression into a search of scalar indexes and an
///   optional post-search refinement query
pub fn apply_scalar_indices(
    expr: Expr,
    index_info: &dyn IndexInformationProvider,
) -> IndexedExpression {
    visit_node(&expr, index_info).unwrap_or(IndexedExpression::refine_only(expr))
}

#[derive(Clone, Default, Debug)]
pub struct FilterPlan {
    pub index_query: Option<ScalarIndexExpr>,
    /// True if the index query is guaranteed to return exact results
    pub skip_recheck: bool,
    pub refine_expr: Option<Expr>,
    pub full_expr: Option<Expr>,
}

impl FilterPlan {
    pub fn empty() -> Self {
        Self {
            index_query: None,
            skip_recheck: true,
            refine_expr: None,
            full_expr: None,
        }
    }

    pub fn refine_columns(&self) -> Vec<String> {
        self.refine_expr
            .as_ref()
            .map(Planner::column_names_in_expr)
            .unwrap_or_default()
    }

    /// Return true if this has a refine step, regardless of the status of prefilter
    pub fn has_refine(&self) -> bool {
        self.refine_expr.is_some()
    }

    pub fn has_any_filter(&self) -> bool {
        self.refine_expr.is_some() || self.index_query.is_some()
    }

    pub fn make_refine_only(&mut self) {
        self.index_query = None;
        self.refine_expr = self.full_expr.clone();
    }
}

pub trait PlannerIndexExt {
    /// Determine how to apply a provided filter
    ///
    /// We parse the filter into a logical expression.  We then
    /// split the logical expression into a portion that can be
    /// satisfied by an index search (of one or more indices) and
    /// a refine portion that must be applied after the index search
    fn create_filter_plan(
        &self,
        filter: Expr,
        index_info: &dyn IndexInformationProvider,
        use_scalar_index: bool,
    ) -> Result<FilterPlan>;
}

impl PlannerIndexExt for Planner {
    fn create_filter_plan(
        &self,
        filter: Expr,
        index_info: &dyn IndexInformationProvider,
        use_scalar_index: bool,
    ) -> Result<FilterPlan> {
        let logical_expr = self.optimize_expr(filter)?;
        if use_scalar_index {
            let indexed_expr = apply_scalar_indices(logical_expr.clone(), index_info);
            let mut skip_recheck = false;
            if let Some(scalar_query) = indexed_expr.scalar_query.as_ref() {
                skip_recheck = !scalar_query.needs_recheck();
            }
            Ok(FilterPlan {
                index_query: indexed_expr.scalar_query,
                refine_expr: indexed_expr.refine_expr,
                full_expr: Some(logical_expr),
                skip_recheck,
            })
        } else {
            Ok(FilterPlan {
                index_query: None,
                skip_recheck: true,
                refine_expr: Some(logical_expr.clone()),
                full_expr: Some(logical_expr),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow_schema::{Field, Schema};
    use datafusion::prelude::SessionContext;
    use datafusion_common::{Column, DFSchema};
    use datafusion_expr::execution_props::ExecutionProps;
    use datafusion_expr::simplify::SimplifyContext;

    use super::*;

    struct ColInfo {
        data_type: DataType,
        parser: Box<dyn ScalarQueryParser>,
    }

    impl ColInfo {
        fn new(data_type: DataType, parser: Box<dyn ScalarQueryParser>) -> Self {
            Self { data_type, parser }
        }
    }

    struct MockIndexInfoProvider {
        indexed_columns: HashMap<String, ColInfo>,
    }

    impl MockIndexInfoProvider {
        fn new(indexed_columns: Vec<(&str, ColInfo)>) -> Self {
            Self {
                indexed_columns: HashMap::from_iter(
                    indexed_columns
                        .into_iter()
                        .map(|(s, ty)| (s.to_string(), ty)),
                ),
            }
        }
    }

    impl IndexInformationProvider for MockIndexInfoProvider {
        fn get_index(&self, col: &str) -> Option<(&DataType, &dyn ScalarQueryParser)> {
            self.indexed_columns
                .get(col)
                .map(|col_info| (&col_info.data_type, col_info.parser.as_ref()))
        }
    }

    fn check(
        index_info: &dyn IndexInformationProvider,
        expr: &str,
        expected: Option<IndexedExpression>,
        optimize: bool,
    ) {
        let schema = Schema::new(vec![
            Field::new("color", DataType::Utf8, false),
            Field::new("size", DataType::Float32, false),
            Field::new("aisle", DataType::UInt32, false),
            Field::new("on_sale", DataType::Boolean, false),
            Field::new("price", DataType::Float32, false),
        ]);
        let df_schema: DFSchema = schema.try_into().unwrap();

        let ctx = SessionContext::default();
        let state = ctx.state();
        let mut expr = state.create_logical_expr(expr, &df_schema).unwrap();
        if optimize {
            let props = ExecutionProps::default();
            let simplify_context = SimplifyContext::new(&props).with_schema(Arc::new(df_schema));
            let simplifier =
                datafusion::optimizer::simplify_expressions::ExprSimplifier::new(simplify_context);
            expr = simplifier.simplify(expr).unwrap();
        }

        let actual = apply_scalar_indices(expr.clone(), index_info);
        if let Some(expected) = expected {
            assert_eq!(actual, expected);
        } else {
            assert!(actual.scalar_query.is_none());
            assert_eq!(actual.refine_expr.unwrap(), expr);
        }
    }

    fn check_no_index(index_info: &dyn IndexInformationProvider, expr: &str) {
        check(index_info, expr, None, false)
    }

    fn check_simple(
        index_info: &dyn IndexInformationProvider,
        expr: &str,
        col: &str,
        query: SargableQuery,
    ) {
        check(
            index_info,
            expr,
            Some(IndexedExpression::index_query(
                col.to_string(),
                format!("{}_idx", col),
                Arc::new(query),
            )),
            false,
        )
    }

    fn check_range(
        index_info: &dyn IndexInformationProvider,
        expr: &str,
        col: &str,
        query: SargableQuery,
    ) {
        check(
            index_info,
            expr,
            Some(IndexedExpression::index_query(
                col.to_string(),
                format!("{}_idx", col),
                Arc::new(query),
            )),
            true,
        )
    }

    fn check_simple_negated(
        index_info: &dyn IndexInformationProvider,
        expr: &str,
        col: &str,
        query: SargableQuery,
    ) {
        check(
            index_info,
            expr,
            Some(
                IndexedExpression::index_query(
                    col.to_string(),
                    format!("{}_idx", col),
                    Arc::new(query),
                )
                .maybe_not()
                .unwrap(),
            ),
            false,
        )
    }

    #[test]
    fn test_expressions() {
        let index_info = MockIndexInfoProvider::new(vec![
            (
                "color",
                ColInfo::new(
                    DataType::Utf8,
                    Box::new(SargableQueryParser::new("color_idx".to_string())),
                ),
            ),
            (
                "aisle",
                ColInfo::new(
                    DataType::UInt32,
                    Box::new(SargableQueryParser::new("aisle_idx".to_string())),
                ),
            ),
            (
                "on_sale",
                ColInfo::new(
                    DataType::Boolean,
                    Box::new(SargableQueryParser::new("on_sale_idx".to_string())),
                ),
            ),
            (
                "price",
                ColInfo::new(
                    DataType::Float32,
                    Box::new(SargableQueryParser::new("price_idx".to_string())),
                ),
            ),
        ]);

        check_no_index(&index_info, "size BETWEEN 5 AND 10");
        // Cast case.  We will cast 5 (an int64) to Int16 and then coerce to UInt32
        check_simple(
            &index_info,
            "aisle = arrow_cast(5, 'Int16')",
            "aisle",
            SargableQuery::Equals(ScalarValue::UInt32(Some(5))),
        );
        // 5 different ways of writing BETWEEN (all should be recognized)
        check_range(
            &index_info,
            "aisle BETWEEN 5 AND 10",
            "aisle",
            SargableQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(5))),
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );
        check_range(
            &index_info,
            "aisle >= 5 AND aisle <= 10",
            "aisle",
            SargableQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(5))),
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );

        check_range(
            &index_info,
            "aisle <= 10 AND aisle >= 5",
            "aisle",
            SargableQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(5))),
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );

        check_range(
            &index_info,
            "5 <= aisle AND 10 >= aisle",
            "aisle",
            SargableQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(5))),
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );

        check_range(
            &index_info,
            "10 >= aisle AND 5 <= aisle",
            "aisle",
            SargableQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(5))),
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );
        check_simple(
            &index_info,
            "on_sale IS TRUE",
            "on_sale",
            SargableQuery::Equals(ScalarValue::Boolean(Some(true))),
        );
        check_simple(
            &index_info,
            "on_sale",
            "on_sale",
            SargableQuery::Equals(ScalarValue::Boolean(Some(true))),
        );
        check_simple_negated(
            &index_info,
            "NOT on_sale",
            "on_sale",
            SargableQuery::Equals(ScalarValue::Boolean(Some(true))),
        );
        check_simple(
            &index_info,
            "on_sale IS FALSE",
            "on_sale",
            SargableQuery::Equals(ScalarValue::Boolean(Some(false))),
        );
        check_simple_negated(
            &index_info,
            "aisle NOT BETWEEN 5 AND 10",
            "aisle",
            SargableQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(5))),
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );
        check_simple(
            &index_info,
            "aisle IN (5, 6, 7)",
            "aisle",
            SargableQuery::IsIn(vec![
                ScalarValue::UInt32(Some(5)),
                ScalarValue::UInt32(Some(6)),
                ScalarValue::UInt32(Some(7)),
            ]),
        );
        check_simple_negated(
            &index_info,
            "NOT aisle IN (5, 6, 7)",
            "aisle",
            SargableQuery::IsIn(vec![
                ScalarValue::UInt32(Some(5)),
                ScalarValue::UInt32(Some(6)),
                ScalarValue::UInt32(Some(7)),
            ]),
        );
        check_simple_negated(
            &index_info,
            "aisle NOT IN (5, 6, 7)",
            "aisle",
            SargableQuery::IsIn(vec![
                ScalarValue::UInt32(Some(5)),
                ScalarValue::UInt32(Some(6)),
                ScalarValue::UInt32(Some(7)),
            ]),
        );
        check_simple(
            &index_info,
            "on_sale is false",
            "on_sale",
            SargableQuery::Equals(ScalarValue::Boolean(Some(false))),
        );
        check_simple(
            &index_info,
            "on_sale is true",
            "on_sale",
            SargableQuery::Equals(ScalarValue::Boolean(Some(true))),
        );
        check_simple(
            &index_info,
            "aisle < 10",
            "aisle",
            SargableQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::UInt32(Some(10))),
            ),
        );
        check_simple(
            &index_info,
            "aisle <= 10",
            "aisle",
            SargableQuery::Range(
                Bound::Unbounded,
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );
        check_simple(
            &index_info,
            "aisle > 10",
            "aisle",
            SargableQuery::Range(
                Bound::Excluded(ScalarValue::UInt32(Some(10))),
                Bound::Unbounded,
            ),
        );
        // In the future we can handle this case if we need to.  For
        // now let's make sure we don't accidentally do the wrong thing
        // (we were getting this backwards in the past)
        check_no_index(&index_info, "10 > aisle");
        check_simple(
            &index_info,
            "aisle >= 10",
            "aisle",
            SargableQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(10))),
                Bound::Unbounded,
            ),
        );
        check_simple(
            &index_info,
            "aisle = 10",
            "aisle",
            SargableQuery::Equals(ScalarValue::UInt32(Some(10))),
        );
        check_simple_negated(
            &index_info,
            "aisle <> 10",
            "aisle",
            SargableQuery::Equals(ScalarValue::UInt32(Some(10))),
        );
        // // Common compound case, AND'd clauses
        let left = Box::new(ScalarIndexExpr::Query(ScalarIndexSearch {
            column: "aisle".to_string(),
            index_name: "aisle_idx".to_string(),
            query: Arc::new(SargableQuery::Equals(ScalarValue::UInt32(Some(10)))),
        }));
        let right = Box::new(ScalarIndexExpr::Query(ScalarIndexSearch {
            column: "color".to_string(),
            index_name: "color_idx".to_string(),
            query: Arc::new(SargableQuery::Equals(ScalarValue::Utf8(Some(
                "blue".to_string(),
            )))),
        }));
        check(
            &index_info,
            "aisle = 10 AND color = 'blue'",
            Some(IndexedExpression {
                scalar_query: Some(ScalarIndexExpr::And(left.clone(), right.clone())),
                refine_expr: None,
            }),
            false,
        );
        // Compound AND's and not all of them are indexed columns
        let refine = Expr::Column(Column::new_unqualified("size")).gt(datafusion_expr::lit(30_i64));
        check(
            &index_info,
            "aisle = 10 AND color = 'blue' AND size > 30",
            Some(IndexedExpression {
                scalar_query: Some(ScalarIndexExpr::And(left.clone(), right.clone())),
                refine_expr: Some(refine.clone()),
            }),
            false,
        );
        // Compounded OR's where ALL columns are indexed
        check(
            &index_info,
            "aisle = 10 OR color = 'blue'",
            Some(IndexedExpression {
                scalar_query: Some(ScalarIndexExpr::Or(left.clone(), right.clone())),
                refine_expr: None,
            }),
            false,
        );
        // Compounded OR's with one or more unindexed columns
        check_no_index(&index_info, "aisle = 10 OR color = 'blue' OR size > 30");
        // AND'd group of OR
        check(
            &index_info,
            "(aisle = 10 OR color = 'blue') AND size > 30",
            Some(IndexedExpression {
                scalar_query: Some(ScalarIndexExpr::Or(left, right)),
                refine_expr: Some(refine),
            }),
            false,
        );
        // Examples of things that are not yet supported but should be supportable someday

        // OR'd group of refined index searches (see IndexedExpression::or for details)
        check_no_index(
            &index_info,
            "(aisle = 10 AND size > 30) OR (color = 'blue' AND size > 20)",
        );

        // Non-normalized arithmetic (can use expression simplification)
        check_no_index(&index_info, "aisle + 3 < 10")
    }
}
