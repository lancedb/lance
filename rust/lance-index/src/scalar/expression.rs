// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{ops::Bound, sync::Arc};

use arrow_array::Array;
use arrow_schema::DataType;
use async_recursion::async_recursion;
use async_trait::async_trait;
use datafusion_common::ScalarValue;
use datafusion_expr::{
    expr::{InList, ScalarFunction},
    Between, BinaryExpr, Expr, Operator, ScalarUDF,
};

use futures::join;
use lance_core::{utils::mask::RowIdMask, Result};
use lance_datafusion::{expr::safe_coerce_scalar, planner::Planner};
use tracing::instrument;

use super::{AnyQuery, LabelListQuery, SargableQuery, ScalarIndex};

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
        low: ScalarValue,
        high: ScalarValue,
    ) -> Option<IndexedExpression>;
    fn visit_in_list(&self, column: &str, in_list: Vec<ScalarValue>) -> Option<IndexedExpression>;
    fn visit_is_bool(&self, column: &str, value: bool) -> Option<IndexedExpression>;
    fn visit_is_null(&self, column: &str) -> Option<IndexedExpression>;
    fn visit_comparison(
        &self,
        column: &str,
        value: ScalarValue,
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

#[derive(Debug, Default)]
pub struct SargableQueryParser {}

impl ScalarQueryParser for SargableQueryParser {
    fn visit_between(
        &self,
        column: &str,
        low: ScalarValue,
        high: ScalarValue,
    ) -> Option<IndexedExpression> {
        let query = SargableQuery::Range(Bound::Included(low), Bound::Included(high));
        Some(IndexedExpression::index_query(
            column.to_string(),
            Arc::new(query),
        ))
    }

    fn visit_in_list(&self, column: &str, in_list: Vec<ScalarValue>) -> Option<IndexedExpression> {
        let query = SargableQuery::IsIn(in_list);
        Some(IndexedExpression::index_query(
            column.to_string(),
            Arc::new(query),
        ))
    }

    fn visit_is_bool(&self, column: &str, value: bool) -> Option<IndexedExpression> {
        Some(IndexedExpression::index_query(
            column.to_string(),
            Arc::new(SargableQuery::Equals(ScalarValue::Boolean(Some(value)))),
        ))
    }

    fn visit_is_null(&self, column: &str) -> Option<IndexedExpression> {
        Some(IndexedExpression::index_query(
            column.to_string(),
            Arc::new(SargableQuery::IsNull()),
        ))
    }

    fn visit_comparison(
        &self,
        column: &str,
        value: ScalarValue,
        op: &Operator,
    ) -> Option<IndexedExpression> {
        let query = match op {
            Operator::Lt => SargableQuery::Range(Bound::Unbounded, Bound::Excluded(value)),
            Operator::LtEq => SargableQuery::Range(Bound::Unbounded, Bound::Included(value)),
            Operator::Gt => SargableQuery::Range(Bound::Excluded(value), Bound::Unbounded),
            Operator::GtEq => SargableQuery::Range(Bound::Included(value), Bound::Unbounded),
            Operator::Eq => SargableQuery::Equals(value),
            // This will be negated by the caller
            Operator::NotEq => SargableQuery::Equals(value),
            _ => unreachable!(),
        };
        Some(IndexedExpression::index_query(
            column.to_string(),
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

#[derive(Debug, Default)]
pub struct LabelListQueryParser {}

impl ScalarQueryParser for LabelListQueryParser {
    fn visit_between(&self, _: &str, _: ScalarValue, _: ScalarValue) -> Option<IndexedExpression> {
        None
    }

    fn visit_in_list(&self, _: &str, _: Vec<ScalarValue>) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_bool(&self, _: &str, _: bool) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_null(&self, _: &str) -> Option<IndexedExpression> {
        None
    }

    fn visit_comparison(&self, _: &str, _: ScalarValue, _: &Operator) -> Option<IndexedExpression> {
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
                    Arc::new(query),
                ))
            } else if func.name() == "array_has_any" {
                let query = LabelListQuery::HasAnyLabel(scalars);
                Some(IndexedExpression::index_query(
                    column.to_string(),
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
    fn index_query(column: String, query: Arc<dyn AnyQuery>) -> Self {
        Self {
            scalar_query: Some(ScalarIndexExpr::Query(column, query)),
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
    async fn load_index(&self, name: &str) -> Result<Arc<dyn ScalarIndex>>;
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
    Query(String, Arc<dyn AnyQuery>),
}

impl PartialEq for ScalarIndexExpr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Not(l0), Self::Not(r0)) => l0 == r0,
            (Self::And(l0, l1), Self::And(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Or(l0, l1), Self::Or(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Query(l0, l1), Self::Query(r0, r1)) => l0 == r0 && l1 == r1,
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
            Self::Query(column, query) => write!(f, "{}", query.format(column)),
        }
    }
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
    pub async fn evaluate(&self, index_loader: &dyn ScalarIndexLoader) -> Result<RowIdMask> {
        match self {
            Self::Not(inner) => {
                let result = inner.evaluate(index_loader).await?;
                Ok(!result)
            }
            Self::And(lhs, rhs) => {
                let lhs_result = lhs.evaluate(index_loader);
                let rhs_result = rhs.evaluate(index_loader);
                let (lhs_result, rhs_result) = join!(lhs_result, rhs_result);
                Ok(lhs_result? & rhs_result?)
            }
            Self::Or(lhs, rhs) => {
                let lhs_result = lhs.evaluate(index_loader);
                let rhs_result = rhs.evaluate(index_loader);
                let (lhs_result, rhs_result) = join!(lhs_result, rhs_result);
                Ok(lhs_result? | rhs_result?)
            }
            Self::Query(column, query) => {
                let index = index_loader.load_index(column).await?;
                let matching_row_ids = index.search(query.as_ref()).await?;
                Ok(RowIdMask {
                    block_list: None,
                    allow_list: Some(matching_row_ids),
                })
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
            Self::Query(column, query) => query.to_expr(column.clone()),
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
        Expr::Literal(value) => safe_coerce_scalar(value, expected_type),
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

    let indexed_expr = query_parser.visit_between(column, low, high)?;

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

    let indexed_expr = query_parser.visit_in_list(column, values)?;

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
        query_parser.visit_comparison(column, scalar, &expr.op)
    } else {
        let (column, col_type, query_parser) = maybe_indexed_column(&expr.right, index_info)?;
        let scalar = maybe_scalar(&expr.left, col_type)?;
        query_parser.visit_comparison(column, scalar, &expr.op)
    }
}

fn visit_and(
    expr: &BinaryExpr,
    index_info: &dyn IndexInformationProvider,
) -> Option<IndexedExpression> {
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

#[derive(Default, Debug)]
pub struct FilterPlan {
    pub index_query: Option<ScalarIndexExpr>,
    pub refine_expr: Option<Expr>,
    pub full_expr: Option<Expr>,
}

impl FilterPlan {
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
            Ok(FilterPlan {
                index_query: indexed_expr.scalar_query,
                refine_expr: indexed_expr.refine_expr,
                full_expr: Some(logical_expr),
            })
        } else {
            Ok(FilterPlan {
                index_query: None,
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
    use datafusion::error::Result as DFResult;
    use datafusion_common::{config::ConfigOptions, TableReference};
    use datafusion_common::{Column, DFSchema};
    use datafusion_expr::{AggregateUDF, ScalarUDF, TableSource, WindowUDF};
    use datafusion_sql::planner::{ContextProvider, PlannerContext, SqlToRel};
    use datafusion_sql::sqlparser::{dialect::PostgreSqlDialect, parser::Parser};

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

    struct MockContextProvider {}

    // We're just compiling simple expressions (not entire statements) and so this is unused
    impl ContextProvider for MockContextProvider {
        fn get_table_source(&self, _name: TableReference) -> DFResult<Arc<dyn TableSource>> {
            todo!()
        }

        fn get_function_meta(&self, _: &str) -> Option<std::sync::Arc<ScalarUDF>> {
            todo!()
        }

        fn get_aggregate_meta(&self, _: &str) -> Option<std::sync::Arc<AggregateUDF>> {
            todo!()
        }

        fn get_window_meta(&self, _: &str) -> Option<std::sync::Arc<WindowUDF>> {
            todo!()
        }

        fn get_variable_type(&self, _: &[String]) -> Option<DataType> {
            todo!()
        }

        fn options(&self) -> &ConfigOptions {
            todo!()
        }

        fn udf_names(&self) -> Vec<String> {
            todo!()
        }

        fn udaf_names(&self) -> Vec<String> {
            todo!()
        }

        fn udwf_names(&self) -> Vec<String> {
            todo!()
        }
    }

    fn check(
        index_info: &dyn IndexInformationProvider,
        expr: &str,
        expected: Option<IndexedExpression>,
    ) {
        let schema = Schema::new(vec![
            Field::new("color", DataType::Utf8, false),
            Field::new("size", DataType::Float32, false),
            Field::new("aisle", DataType::UInt32, false),
            Field::new("on_sale", DataType::Boolean, false),
            Field::new("price", DataType::Float32, false),
        ]);
        let dialect = PostgreSqlDialect {};
        let mut parser = Parser::new(&dialect).try_with_sql(expr).unwrap();
        let expr = parser.parse_expr().unwrap();
        let context_provider = MockContextProvider {};
        let planner = SqlToRel::new(&context_provider);
        let df_schema: DFSchema = schema.try_into().unwrap();
        let mut planner_context = PlannerContext::new();
        let expr = planner
            .sql_to_expr(expr, &df_schema, &mut planner_context)
            .unwrap();

        let actual = apply_scalar_indices(expr.clone(), index_info);
        if let Some(expected) = expected {
            assert_eq!(actual, expected);
        } else {
            assert!(actual.scalar_query.is_none());
            assert_eq!(actual.refine_expr.unwrap(), expr);
        }
    }

    fn check_no_index(index_info: &dyn IndexInformationProvider, expr: &str) {
        check(index_info, expr, None)
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
                Arc::new(query),
            )),
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
                IndexedExpression::index_query(col.to_string(), Arc::new(query))
                    .maybe_not()
                    .unwrap(),
            ),
        )
    }

    #[test]
    fn test_expressions() {
        let index_info = MockIndexInfoProvider::new(vec![
            (
                "color",
                ColInfo::new(DataType::Utf8, Box::<SargableQueryParser>::default()),
            ),
            (
                "aisle",
                ColInfo::new(DataType::UInt32, Box::<SargableQueryParser>::default()),
            ),
            (
                "on_sale",
                ColInfo::new(DataType::Boolean, Box::<SargableQueryParser>::default()),
            ),
            (
                "price",
                ColInfo::new(DataType::Float32, Box::<SargableQueryParser>::default()),
            ),
        ]);

        check_no_index(&index_info, "size BETWEEN 5 AND 10");
        check_simple(
            &index_info,
            "aisle BETWEEN 5 AND 10",
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
        let left = Box::new(ScalarIndexExpr::Query(
            "aisle".to_string(),
            Arc::new(SargableQuery::Equals(ScalarValue::UInt32(Some(10)))),
        ));
        let right = Box::new(ScalarIndexExpr::Query(
            "color".to_string(),
            Arc::new(SargableQuery::Equals(ScalarValue::Utf8(Some(
                "blue".to_string(),
            )))),
        ));
        check(
            &index_info,
            "aisle = 10 AND color = 'blue'",
            Some(IndexedExpression {
                scalar_query: Some(ScalarIndexExpr::And(left.clone(), right.clone())),
                refine_expr: None,
            }),
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
        );
        // Compounded OR's where ALL columns are indexed
        check(
            &index_info,
            "aisle = 10 OR color = 'blue'",
            Some(IndexedExpression {
                scalar_query: Some(ScalarIndexExpr::Or(left.clone(), right.clone())),
                refine_expr: None,
            }),
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
