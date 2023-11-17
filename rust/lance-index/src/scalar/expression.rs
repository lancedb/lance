// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{ops::Bound, sync::Arc};

use arrow_schema::DataType;
use async_recursion::async_recursion;
use async_trait::async_trait;
use datafusion_common::ScalarValue;
use datafusion_expr::{expr::InList, Between, BinaryExpr, Expr, Operator};

use futures::join;
use lance_core::{utils::mask::RowIdMask, Result};
use lance_datafusion::expr::safe_coerce_scalar;
use roaring::RoaringTreemap;

use super::{ScalarIndex, ScalarQuery};

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
    fn refine_only(refine_expr: Expr) -> Self {
        Self {
            scalar_query: None,
            refine_expr: Some(refine_expr),
        }
    }

    /// Create an expression that is only an index query
    fn index_query(column: String, query: ScalarQuery) -> Self {
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
/// This is a tree of operations beacause we may need to logically combine or
/// modify the results of scalar lookups
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarIndexExpr {
    Not(Box<ScalarIndexExpr>),
    And(Box<ScalarIndexExpr>, Box<ScalarIndexExpr>),
    Or(Box<ScalarIndexExpr>, Box<ScalarIndexExpr>),
    Query(String, ScalarQuery),
}

impl std::fmt::Display for ScalarIndexExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Not(inner) => write!(f, "NOT({})", inner),
            Self::And(lhs, rhs) => write!(f, "AND({},{})", lhs, rhs),
            Self::Or(lhs, rhs) => write!(f, "OR({},{})", lhs, rhs),
            Self::Query(column, query) => write!(f, "{}", query.fmt_with_col(column)),
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
                let allow_list = index.search(query).await?;
                let allow_list = RoaringTreemap::from_iter(allow_list.values().iter());
                Ok(RowIdMask {
                    block_list: None,
                    allow_list: Some(allow_list),
                })
            }
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
) -> Option<(&'a str, &'b DataType)> {
    let col = maybe_column(expr)?;
    let data_type = index_info.get_index(col);
    data_type.map(|ty| (col, ty))
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
    let (column, col_type) = maybe_indexed_column(&between.expr, index_info)?;
    let low = maybe_scalar(&between.low, col_type)?;
    let high = maybe_scalar(&between.high, col_type)?;

    let query = ScalarQuery::Range(Bound::Included(low.clone()), Bound::Included(high.clone()));
    let indexed_expr = IndexedExpression::index_query(column.to_string(), query);
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
    let (column, col_type) = maybe_indexed_column(&in_list.expr, index_info)?;
    let values = maybe_scalar_list(&in_list.list, col_type)?;

    let query = ScalarQuery::IsIn(values);
    let indexed_expr = IndexedExpression::index_query(column.to_string(), query);
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
    let (column, col_type) = maybe_indexed_column(expr, index_info)?;
    if *col_type != DataType::Boolean {
        None
    } else {
        Some(IndexedExpression::index_query(
            column.to_string(),
            ScalarQuery::Equals(ScalarValue::Boolean(Some(value))),
        ))
    }
}

fn visit_is_null(
    expr: &Expr,
    index_info: &dyn IndexInformationProvider,
    negated: bool,
) -> Option<IndexedExpression> {
    let (column, _) = maybe_indexed_column(expr, index_info)?;
    let indexed_expr = IndexedExpression::index_query(column.to_string(), ScalarQuery::IsNull());
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

fn visit_comparison_normalized(scalar: ScalarValue, op: &Operator) -> ScalarQuery {
    match op {
        Operator::Lt => ScalarQuery::Range(Bound::Unbounded, Bound::Excluded(scalar)),
        Operator::LtEq => ScalarQuery::Range(Bound::Unbounded, Bound::Included(scalar)),
        Operator::Gt => ScalarQuery::Range(Bound::Excluded(scalar), Bound::Unbounded),
        Operator::GtEq => ScalarQuery::Range(Bound::Included(scalar), Bound::Unbounded),
        Operator::Eq => ScalarQuery::Equals(scalar),
        // This will be negated by the caller
        Operator::NotEq => ScalarQuery::Equals(scalar),
        _ => unreachable!(),
    }
}

fn visit_comparison(
    expr: &BinaryExpr,
    index_info: &dyn IndexInformationProvider,
) -> Option<IndexedExpression> {
    let left_col = maybe_indexed_column(&expr.left, index_info);
    if let Some((column, col_type)) = left_col {
        let scalar = maybe_scalar(&expr.right, col_type)?;
        Some(IndexedExpression::index_query(
            column.to_string(),
            visit_comparison_normalized(scalar, &expr.op),
        ))
    } else {
        let (column, col_type) = maybe_indexed_column(&expr.right, index_info)?;
        let scalar = maybe_scalar(&expr.left, col_type)?;
        Some(IndexedExpression::index_query(
            column.to_string(),
            visit_comparison_normalized(scalar, &expr.op),
        ))
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

fn visit_node(expr: &Expr, index_info: &dyn IndexInformationProvider) -> Option<IndexedExpression> {
    match expr {
        Expr::Between(between) => visit_between(between, index_info),
        Expr::InList(in_list) => visit_in_list(in_list, index_info),
        Expr::IsFalse(expr) => visit_is_bool(expr.as_ref(), index_info, false),
        Expr::IsTrue(expr) => visit_is_bool(expr.as_ref(), index_info, true),
        Expr::IsNull(expr) => visit_is_null(expr.as_ref(), index_info, false),
        Expr::IsNotNull(expr) => visit_is_null(expr.as_ref(), index_info, true),
        Expr::Not(expr) => visit_not(expr.as_ref(), index_info),
        Expr::BinaryExpr(binary_expr) => visit_binary_expr(binary_expr, index_info),
        _ => None,
    }
}

/// A trait to be used in `apply_scalar_indices` to inform the function which columns are indexeds
pub trait IndexInformationProvider {
    /// Check if an index exists for `col` and, if so, return the data type of col
    fn get_index(&self, col: &str) -> Option<&DataType>;
}

/// Attempt to split a filter expression into a search of scalar indexes and an
///   optional post-search refinement query
pub fn apply_scalar_indices(
    expr: Expr,
    index_info: &dyn IndexInformationProvider,
) -> IndexedExpression {
    visit_node(&expr, index_info).unwrap_or(IndexedExpression::refine_only(expr))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ops::Bound;

    use arrow_schema::{DataType, Field, Schema};
    use datafusion_common::{config::ConfigOptions, TableReference};
    use datafusion_common::{Column, DFSchema, ScalarValue};
    use datafusion_expr::{AggregateUDF, Expr, ScalarUDF, TableSource, WindowUDF};
    use datafusion_sql::planner::{ContextProvider, PlannerContext, SqlToRel};
    use datafusion_sql::sqlparser::{dialect::PostgreSqlDialect, parser::Parser};

    use crate::scalar::expression::apply_scalar_indices;
    use crate::scalar::ScalarQuery;

    use super::*;

    struct MockIndexInfoProvider {
        indexed_columns: HashMap<String, DataType>,
    }

    impl MockIndexInfoProvider {
        fn new(indexed_columns: Vec<(&str, DataType)>) -> Self {
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
        fn get_index(&self, col: &str) -> Option<&DataType> {
            self.indexed_columns.get(col)
        }
    }

    struct MockContextProvider {}

    // We're just compiling simple expressions (not entire statements) and so this is unused
    impl ContextProvider for MockContextProvider {
        fn get_table_provider(
            &self,
            _: TableReference,
        ) -> datafusion_common::Result<std::sync::Arc<dyn TableSource>> {
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
        query: ScalarQuery,
    ) {
        check(
            index_info,
            expr,
            Some(IndexedExpression::index_query(col.to_string(), query)),
        )
    }

    fn check_simple_negated(
        index_info: &dyn IndexInformationProvider,
        expr: &str,
        col: &str,
        query: ScalarQuery,
    ) {
        check(
            index_info,
            expr,
            Some(
                IndexedExpression::index_query(col.to_string(), query)
                    .maybe_not()
                    .unwrap(),
            ),
        )
    }

    #[test]
    fn test_expressions() {
        let index_info = MockIndexInfoProvider::new(vec![
            ("color", DataType::Utf8),
            ("aisle", DataType::UInt32),
            ("on_sale", DataType::Boolean),
            ("price", DataType::Float32),
        ]);

        check_no_index(&index_info, "size BETWEEN 5 AND 10");
        check_simple(
            &index_info,
            "aisle BETWEEN 5 AND 10",
            "aisle",
            ScalarQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(5))),
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );
        check_simple_negated(
            &index_info,
            "aisle NOT BETWEEN 5 AND 10",
            "aisle",
            ScalarQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(5))),
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );
        check_simple(
            &index_info,
            "aisle IN (5, 6, 7)",
            "aisle",
            ScalarQuery::IsIn(vec![
                ScalarValue::UInt32(Some(5)),
                ScalarValue::UInt32(Some(6)),
                ScalarValue::UInt32(Some(7)),
            ]),
        );
        check_simple_negated(
            &index_info,
            "NOT aisle IN (5, 6, 7)",
            "aisle",
            ScalarQuery::IsIn(vec![
                ScalarValue::UInt32(Some(5)),
                ScalarValue::UInt32(Some(6)),
                ScalarValue::UInt32(Some(7)),
            ]),
        );
        check_simple_negated(
            &index_info,
            "aisle NOT IN (5, 6, 7)",
            "aisle",
            ScalarQuery::IsIn(vec![
                ScalarValue::UInt32(Some(5)),
                ScalarValue::UInt32(Some(6)),
                ScalarValue::UInt32(Some(7)),
            ]),
        );
        check_simple(
            &index_info,
            "on_sale is false",
            "on_sale",
            ScalarQuery::Equals(ScalarValue::Boolean(Some(false))),
        );
        check_simple(
            &index_info,
            "on_sale is true",
            "on_sale",
            ScalarQuery::Equals(ScalarValue::Boolean(Some(true))),
        );
        check_simple(
            &index_info,
            "aisle < 10",
            "aisle",
            ScalarQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::UInt32(Some(10))),
            ),
        );
        check_simple(
            &index_info,
            "aisle <= 10",
            "aisle",
            ScalarQuery::Range(
                Bound::Unbounded,
                Bound::Included(ScalarValue::UInt32(Some(10))),
            ),
        );
        check_simple(
            &index_info,
            "aisle > 10",
            "aisle",
            ScalarQuery::Range(
                Bound::Excluded(ScalarValue::UInt32(Some(10))),
                Bound::Unbounded,
            ),
        );
        check_simple(
            &index_info,
            "aisle >= 10",
            "aisle",
            ScalarQuery::Range(
                Bound::Included(ScalarValue::UInt32(Some(10))),
                Bound::Unbounded,
            ),
        );
        check_simple(
            &index_info,
            "aisle = 10",
            "aisle",
            ScalarQuery::Equals(ScalarValue::UInt32(Some(10))),
        );
        check_simple_negated(
            &index_info,
            "aisle <> 10",
            "aisle",
            ScalarQuery::Equals(ScalarValue::UInt32(Some(10))),
        );
        // // Common compound case, AND'd clauses
        let left = Box::new(ScalarIndexExpr::Query(
            "aisle".to_string(),
            ScalarQuery::Equals(ScalarValue::UInt32(Some(10))),
        ));
        let right = Box::new(ScalarIndexExpr::Query(
            "color".to_string(),
            ScalarQuery::Equals(ScalarValue::Utf8(Some("blue".to_string()))),
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
                scalar_query: Some(ScalarIndexExpr::Or(left.clone(), right.clone())),
                refine_expr: Some(refine.clone()),
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
