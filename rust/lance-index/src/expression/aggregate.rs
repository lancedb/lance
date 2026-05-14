// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Aggregate query parsing — the plan-time half of aggregate pushdown.
//!
//! Parallels [`crate::scalar::expression`]: where the scalar side splits a
//! `WHERE` clause into an index search and a post-filter refine, the aggregate
//! side splits a `SELECT` element (typically containing an aggregate) into an
//! [`AggregateIndexSearch`] and a post-projection [`Expr`] that runs on top of
//! the search's output.

use std::any::Any;
use std::sync::Arc;

use arrow_schema::Field;
use datafusion_expr::Expr;

use datafusion_expr::expr::AggregateFunction;
use lance_core::Result;

/// A parsed aggregate query, ready to be evaluated against a scalar index.
///
/// Implementations carry whatever state the parser produced (e.g. the
/// distinct/approximate flags on [`CountQuery`]); the execute side downcasts
/// via [`AnyAggregateQuery::as_any`] to drive the right computation. The
/// counterpart for `WHERE` filters is [`crate::scalar::AnyQuery`].
pub trait AnyAggregateQuery: std::fmt::Debug + std::fmt::Display + Any + Send + Sync {
    /// Cast the query as `Any` to allow for downcasting to the concrete query type.
    fn as_any(&self) -> &dyn Any;
    /// The expected schema of the output of this aggregate query.
    ///
    /// This should be the "partial aggregate" representation.  For example, if the query
    /// is an AVG aggregate then it should be a struct field with two fields: `sum` and `count`.
    fn output(&self) -> &Field;
}

/// A parser that decides whether an expression (containing one
/// aggregate) can be served by indices
///
/// For example, in the query SELECT MAX(score), MIN(error) FROM t WHERE category = '7'
/// this would be called twice, once for MAX(score) and once for MIN(error).  The query
/// should indicate whether it can handle a filter or not.  If so, and the filter can
/// be satisfied by scalar index search, then the filter will be provided as a bitmap
/// to the aggregate search.  If not, then the aggregate query will not be executed
/// if a filter is present.
///
/// In most cases this will only return when the aggregate function is an approximate
/// function as exact acceleration of aggregates is difficult.
pub trait AggregateQueryParser: std::fmt::Debug + Send + Sync {
    /// Parse the given aggregate, returning a query that can be evaluated.
    ///
    /// This method should not load or search the index.  It is expected we can
    /// do the parsing purely from the dataset metadata and the index details.
    ///
    /// Returns `Some(query)` if the parser recognizes the aggregate and can
    /// produce a query for it. Returns `None` if not — the caller is expected
    /// to fall back to a normal scan in that case.
    fn parse_aggregate(
        &self,
        aggregate: &AggregateFunction,
    ) -> Result<Option<Arc<dyn AnyAggregateQuery>>>;
}

/// A single aggregate-index search — leaf carrying the parsed query.
///
/// Parallels [`crate::scalar::expression::ScalarIndexSearch`]. There is no
/// tree variant (no `AggregateIndexExpr`) because v1 only emits a single
/// search per parsed aggregate. If we later need to combine multiple
/// aggregate searches logically we'll introduce one.
#[derive(Debug, Clone)]
pub struct AggregateIndexSearch {
    /// The index accelerating the aggregate search
    ///
    /// Will be None for nilary aggregates such as COUNT(*)
    pub index_name: Option<String>,
    /// The query that the exec node will evaluate
    pub query: Arc<dyn AnyAggregateQuery>,
    /// Filter to be applied only to this aggregate
    pub filter: Option<Expr>,
    /// The original expression that was parsed into this aggregate search
    pub original_expr: Expr,
}

impl std::fmt::Display for AggregateIndexSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}@{:?}",
            self.query,
            self.index_name.as_deref().unwrap_or("*")
        )
    }
}

/// A query for `COUNT`-shaped aggregates — covers `COUNT(*)`, `COUNT(col)`,
/// `COUNT(DISTINCT col)`, and their approximate variants.
///
/// The combination of `is_approximate` and `is_distinct` selects between the
/// four standard SQL shapes, with the matching constructors below.
#[derive(Debug, Clone, PartialEq)]
pub struct CountQuery {
    is_approximate: bool,
    is_distinct: bool,
    output_field: Field,
}

impl CountQuery {
    /// Exact non-distinct count — `COUNT(*)` or `COUNT(col)`.
    pub fn basic() -> Self {
        Self {
            is_approximate: false,
            is_distinct: false,
            output_field: Field::new("count", arrow_schema::DataType::UInt64, false),
        }
    }

    /// Approximate non-distinct count — used when the underlying index can
    /// only produce an estimate (e.g. via a sketch).
    pub fn approx() -> Self {
        Self {
            is_approximate: true,
            is_distinct: false,
            output_field: Field::new("count", arrow_schema::DataType::UInt64, false),
        }
    }

    /// Exact distinct count — `COUNT(DISTINCT col)`.
    pub fn distinct() -> Self {
        Self {
            is_approximate: false,
            is_distinct: true,
            output_field: Field::new("count_distinct", arrow_schema::DataType::UInt64, false),
        }
    }

    /// Approximate distinct count — `APPROX_COUNT_DISTINCT(col)` / HLL-style.
    pub fn approx_distinct() -> Self {
        Self {
            is_approximate: true,
            is_distinct: true,
            output_field: Field::new(
                "approx_count_distinct",
                arrow_schema::DataType::UInt64,
                false,
            ),
        }
    }

    /// `true` if the result is an approximation rather than an exact count.
    pub fn is_approximate(&self) -> bool {
        self.is_approximate
    }

    /// `true` if the count is over distinct values rather than rows.
    pub fn is_distinct(&self) -> bool {
        self.is_distinct
    }
}

impl std::fmt::Display for CountQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.output_field.name().fmt(f)
    }
}

impl AnyAggregateQuery for CountQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn output(&self) -> &Field {
        &self.output_field
    }
}

/// Parser for [`CountQuery`]
#[derive(Debug, Default)]
pub struct CountQueryParser {
    #[allow(dead_code)]
    index_name: Option<String>,
}

impl CountQueryParser {
    /// Create a parser. `index_name`, when set, identifies a count-supporting
    /// scalar index that the produced [`AggregateIndexSearch`] should bind to.
    pub fn new(index_name: Option<String>) -> Self {
        Self { index_name }
    }
}

impl AggregateQueryParser for CountQueryParser {
    fn parse_aggregate(
        &self,
        agg: &AggregateFunction,
    ) -> Result<Option<Arc<dyn AnyAggregateQuery>>> {
        if agg.func.name() != "count" {
            return Ok(None);
        }
        let query = if agg.params.distinct {
            CountQuery::distinct()
        } else {
            CountQuery::basic()
        };
        Ok(Some(Arc::new(query)))
    }
}
