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

use datafusion::datasource::provider_as_source;
use datafusion::logical_expr::{
    Expr, Extension, LogicalPlan, LogicalPlanBuilder, UserDefinedLogicalNodeCore,
};

use datafusion::prelude::col;
use lance_index::vector::DIST_COL;

use super::nodes::{LanceTakeNode, VectorSearchNode};
use super::source::{LanceScanSource, ScanSourceOptions};
use crate::Result;
use crate::dataset::Scanner;

/// Relation name for the scan leaf. Column references in filters and projections are
/// unqualified, so this only ever shows up in `EXPLAIN` output.
pub const TABLE_NAME: &str = "lance";

pub fn build(scanner: &Scanner) -> Result<LogicalPlan> {
    let source = LanceScanSource::new(scanner.dataset.clone(), source_options(scanner))?;
    let mut builder =
        LogicalPlanBuilder::scan(TABLE_NAME, provider_as_source(Arc::new(source)), None)?;

    let filter = scanner.get_expr_filter()?;

    // Prefilter and postfilter are the same predicate on opposite sides of the search: restrict
    // the candidates, or trim the results. They are genuinely different queries under
    // approximation, which is why this is operator ordering and not a flag on the search node.
    let prefilter = scanner.prefilter && scanner.nearest.is_some();
    if prefilter && let Some(filter) = filter.clone() {
        builder = builder.filter(filter)?;
    }

    if let Some(query) = &scanner.nearest {
        let search = VectorSearchNode::try_new(
            builder.plan().clone(),
            scanner.dataset.clone(),
            query.clone(),
        )?;
        builder = LogicalPlanBuilder::new(extension(search));
        builder = with_take(builder, scanner)?;
    }

    if !prefilter && let Some(filter) = filter {
        builder = builder.filter(filter)?;
    }

    if scanner.limit.unwrap_or(0) > 0 || scanner.offset.is_some() {
        builder = builder.limit(
            scanner.offset.unwrap_or(0) as usize,
            scanner.limit.map(|l| l as usize),
        )?;
    }

    builder = builder.project(output_exprs(scanner))?;
    Ok(builder.build()?)
}

/// Insert late materialization above a search, unless the search already produced everything.
///
/// A search emits only `[_rowid, _distance]`; the user's columns have to be fetched by row id
/// afterwards. Plain scans need no equivalent — DataFusion's projection pushdown puts the column
/// list into the leaf directly.
fn with_take(builder: LogicalPlanBuilder, scanner: &Scanner) -> Result<LogicalPlanBuilder> {
    let projection = scanner.projection_plan.physical_projection.clone();
    let input = builder.plan().clone();
    if LanceTakeNode::is_noop(&input, &projection)? {
        return Ok(builder);
    }
    let take = LanceTakeNode::try_new(input, scanner.dataset.clone(), projection)?;
    Ok(LogicalPlanBuilder::new(extension(take)))
}

fn extension(node: impl UserDefinedLogicalNodeCore) -> LogicalPlan {
    LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    })
}

/// The user's requested output columns, as logical expressions.
///
/// `ProjectionPlan` has already parsed and coerced these against the full schema, so they can be
/// used as-is. An alias is only added when the output name differs from what the expression
/// would be named anyway — a redundant `s AS s` survives optimization and shows up in `EXPLAIN`.
fn output_exprs(scanner: &Scanner) -> Vec<Expr> {
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

    // `_distance` is appended even when the user did not ask for it. That is legacy behavior the
    // imperative path implements in `calculate_final_projection`; replicating it here is what
    // lets the two paths be compared row for row.
    let has_distance = exprs
        .iter()
        .any(|expr| expr.schema_name().to_string() == DIST_COL);
    if scanner.autoproject_scoring_columns && scanner.nearest.is_some() && !has_distance {
        exprs.push(col(DIST_COL));
    }
    exprs
}

fn source_options(scanner: &Scanner) -> ScanSourceOptions {
    ScanSourceOptions {
        batch_size: scanner.batch_size,
        batch_readahead: scanner.batch_readahead,
        fragment_readahead: scanner.fragment_readahead,
        io_buffer_size: scanner.io_buffer_size,
        file_reader_options: scanner.resolved_file_reader_options(),
        fragments: scanner.fragments.clone().map(Arc::new),
        index_expr_result_format: scanner.index_expr_result_format(),
        use_scalar_index: scanner.use_scalar_index,
    }
}
