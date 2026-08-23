// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage 1: `Scanner` state -> an unoptimized [`LogicalPlan`].
//!
//! This stage is deliberately synchronous and does no I/O. It emits the *naive* plan — the shape
//! the user's query literally describes — with no index awareness and no late-materialization
//! split. Every Lance-specific decision is left to the rules in
//! [`super::rules`], which run after [`super::context::ScanPlanningContext`] has done the
//! prefetching they need.

use std::sync::Arc;

use datafusion::common::Column;
use datafusion::datasource::provider_as_source;
use datafusion::functions::core::expr_fn::get_field;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, SortExpr};
use lance_core::{ROW_ADDR, ROW_ID};

use super::source::{LanceScanSource, ScanSourceOptions};
use crate::dataset::scanner::ColumnOrdering;
use crate::dataset::{Dataset, Scanner};
use crate::{Error, Result};

/// Relation name for the scan leaf. Column references in filters and projections are
/// unqualified, so this only ever shows up in `EXPLAIN` output.
pub const TABLE_NAME: &str = "lance";

pub fn build(scanner: &Scanner) -> Result<LogicalPlan> {
    let scan = scan_leaf(scanner)?;
    let filter = scanner.get_expr_filter()?;

    let mut builder = LogicalPlanBuilder::new(scan);
    if let Some(filter) = filter {
        builder = builder.filter(filter)?;
    }

    // An aggregate replaces the output projection rather than sitting above it, and `validate_options`
    // has already rejected a limit, offset or ordering alongside one. The columns it reads are found
    // by projection pushdown, which is what the imperative path's `agg_projection` does by hand.
    if let Some(aggregate) = &scanner.aggregate {
        let aggregate = builder.aggregate(
            aggregate
                .group_by
                .iter()
                .map(unqualified)
                .collect::<Vec<_>>(),
            aggregate
                .aggregates
                .iter()
                .map(unqualified)
                .collect::<Vec<_>>(),
        )?;
        return Ok(aggregate.build()?);
    }

    // Sort below limit/offset, matching the imperative path: the limit takes the first rows of the
    // ordering, not an arbitrary subset that is then ordered.
    if let Some(ordering) = &scanner.ordering {
        builder = builder.sort(ordering_exprs(ordering, &scanner.dataset)?)?;
    }

    if scanner.limit.unwrap_or(0) > 0 || scanner.offset.is_some() {
        builder = builder.limit(
            scanner.offset.unwrap_or(0) as usize,
            scanner.limit.map(|l| l as usize),
        )?;
    }

    builder = builder.project(output_exprs(scanner)?)?;
    Ok(builder.build()?)
}

/// Pin an aggregate expression's output name to the unqualified one.
///
/// Resolving `sum(i)` against the scan relation would otherwise name the result `sum(lance.i)`,
/// leaking this module's relation name into the user's output schema. The imperative path names
/// these from the expression as written, so this keeps the two agreeing.
fn unqualified(expr: &Expr) -> Expr {
    let name = expr.schema_name().to_string();
    expr.clone().alias(name)
}

fn scan_leaf(scanner: &Scanner) -> Result<LogicalPlan> {
    let source = LanceScanSource::new(scanner.dataset.clone(), source_options(scanner))?;
    Ok(
        LogicalPlanBuilder::scan(TABLE_NAME, provider_as_source(Arc::new(source)), None)?
            .build()?,
    )
}

/// The scanner's `order_by` as logical sort expressions.
///
/// A nested ordering column ("outer.inner") becomes a chain of `get_field` calls rather than a
/// column reference, which is also what makes projection pushdown keep the outer struct.
fn ordering_exprs(ordering: &[ColumnOrdering], dataset: &Dataset) -> Result<Vec<SortExpr>> {
    ordering
        .iter()
        .map(|column| {
            let path = dataset
                .schema()
                .resolve_case_insensitive(&column.column_name)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Field '{}' not found in schema",
                        column.column_name
                    ))
                })?;
            let mut expr = Expr::Column(Column::new_unqualified(&path[0].name));
            for nested in &path[1..] {
                expr = get_field(expr, nested.name.clone());
            }
            Ok(expr.sort(column.ascending, column.nulls_first))
        })
        .collect()
}

/// The user's requested output columns, as logical expressions.
///
/// `ProjectionPlan` has already parsed and coerced these against the full schema, so they can be
/// used as-is. An alias is only added when the output name differs from what the expression
/// would be named anyway — a redundant `s AS s` survives optimization and shows up in `EXPLAIN`.
fn output_exprs(scanner: &Scanner) -> Result<Vec<Expr>> {
    let mut exprs = scanner
        .projection_plan
        .requested_output_expr
        .iter()
        .map(|output| {
            if output.expr.schema_name().to_string() == output.name {
                output.expr.clone()
            } else {
                output.expr.clone().alias(&output.name)
            }
        })
        .collect::<Vec<_>>();

    // `with_row_id`/`with_row_address` promise their column is *last*.
    if scanner.legacy_with_row_id {
        move_to_end(&mut exprs, ROW_ID);
    }
    if scanner.legacy_with_row_addr {
        move_to_end(&mut exprs, ROW_ADDR);
    }
    Ok(exprs)
}

fn move_to_end(exprs: &mut Vec<Expr>, name: &str) {
    if let Some(position) = exprs
        .iter()
        .position(|expr| expr.schema_name().to_string() == name)
    {
        let expr = exprs.remove(position);
        exprs.push(expr);
    }
}

pub(super) fn source_options(scanner: &Scanner) -> ScanSourceOptions {
    ScanSourceOptions {
        batch_size: scanner.batch_size,
        batch_readahead: scanner.batch_readahead,
        fragment_readahead: scanner.fragment_readahead,
        io_buffer_size: scanner.io_buffer_size,
        file_reader_options: scanner.resolved_file_reader_options(),
        fragments: scanner.fragments.clone().map(Arc::new),
        index_expr_result_format: scanner.index_expr_result_format(),
        use_scalar_index: scanner.use_scalar_index,
        materialization_style: scanner.materialization_style.clone(),
        blob_handling: scanner.blob_handling.clone(),
        fast_search: scanner.fast_search,
        index_segments: scanner.index_segments.clone().map(Arc::new),
        include_deleted_rows: scanner.include_deleted_rows,
        rows: None,
        filter_plan: None,
        legacy_scanner: super::source::v1::is_legacy(&scanner.dataset)
            .then(|| Arc::new(scanner.clone())),
    }
}
