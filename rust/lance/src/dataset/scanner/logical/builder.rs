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
use datafusion::logical_expr::{
    Expr, Extension, LogicalPlan, LogicalPlanBuilder, SortExpr, UserDefinedLogicalNodeCore,
};

use datafusion::prelude::col;
use lance_core::{ROW_ADDR, ROW_ID, datatypes::OnMissing};
use lance_index::scalar::inverted::{DOC_INDEX_COL, SCORE_COL};
use lance_index::vector::DIST_COL;

use super::context::take_settings;
use super::fts;
use super::prepare::PreparedQueries;
use super::source::{LanceScanSource, ScanSourceOptions};
use super::{LanceTakeNode, VectorAccessPath, VectorRerankNode, VectorSearchNode};
use crate::dataset::scanner::ColumnOrdering;
use crate::dataset::{Dataset, Scanner};
use crate::{Error, Result};

/// Relation name for the scan leaf. Column references in filters and projections are
/// unqualified, so this only ever shows up in `EXPLAIN` output.
pub const TABLE_NAME: &str = "lance";

pub fn build(scanner: &Scanner, prepared: &PreparedQueries) -> Result<LogicalPlan> {
    let scan = scan_leaf(scanner)?;
    let filter = scanner.get_expr_filter()?;

    // Prefilter and postfilter are the same predicate on opposite sides of the search: restrict
    // the candidates, or trim the results. They are genuinely different queries under
    // approximation, which is why this is operator ordering and not a flag on the search node.
    let has_search = scanner.nearest.is_some() || prepared.full_text.is_some();
    let prefilter = scanner.prefilter && has_search;

    let mut source = scan.clone();
    if prefilter && let Some(filter) = filter.clone() {
        source = LogicalPlanBuilder::new(source).filter(filter)?.build()?;
    }

    // A postfilter reads columns the user may not have projected, and above a search the only way
    // to get a column is to take it. So the take has to cover the filter's columns too, and the
    // final projection trims them back off. The imperative path splits this across two takes
    // (`pre_filter_projection` then the output projection); one take plus a projection is the same
    // set of reads.
    // A sort sits above any search, so its columns have to survive late materialization too.
    let mut postfilter_columns: Vec<String> = scanner
        .ordering
        .iter()
        .flatten()
        .map(|column| column.column_name.clone())
        .collect();
    if !prefilter {
        if let Some(filter) = &filter {
            postfilter_columns.extend(filter.column_refs().iter().map(|c| c.name.clone()));
        }
        // A vector `query_filter` above the search scores the rows it is given, so the vector
        // column has to be in flight by then.
        if let Some(query) = &prepared.vector_filter {
            postfilter_columns.push(query.column.clone());
        }
    }

    // A `query_filter` obeys the same prefilter/postfilter switch as an expression filter: below
    // the search it produces the candidates, above it trims the results.
    let mut builder = if let Some(query) = &prepared.full_text {
        let searched = match prepared.vector_filter.as_ref().filter(|_| prefilter) {
            // Vector search first, then re-rank the survivors by BM25.
            Some(vector_query) => {
                let vector = extension(VectorSearchNode::try_new(
                    source,
                    scanner.dataset.clone(),
                    vector_query.clone(),
                )?);
                fts::build_rerank(
                    vector,
                    scan,
                    &scanner.dataset,
                    query,
                    search_limit(scanner),
                    &take_settings(scanner),
                )?
            }
            None => fts::build_source(source, &scanner.dataset, query, search_limit(scanner))?,
        };
        with_take(
            LogicalPlanBuilder::new(searched),
            scanner,
            &postfilter_columns,
        )?
    } else if let Some(query) = &scanner.nearest {
        let searched = match prepared.fts_filter.as_ref().filter(|_| prefilter) {
            // An FTS query used as a filter produces the candidate rows, whose vectors are then
            // fetched and scored exactly. `with_resolution(Flat)` is what says "these candidates
            // are the search space", so no rule may swap in the index.
            Some(fts_query) => {
                let mut candidates = fts::build_source(source, &scanner.dataset, fts_query, None)?;
                if candidates
                    .schema()
                    .has_column_with_unqualified_name(DOC_INDEX_COL)
                {
                    candidates = fts::dedupe_rows(candidates)?;
                }
                let candidates = fts::take_column(
                    candidates,
                    &scanner.dataset,
                    &query.column,
                    &take_settings(scanner),
                )?;
                extension(
                    VectorSearchNode::try_new(candidates, scanner.dataset.clone(), query.clone())?
                        .with_resolution(VectorAccessPath::Flat),
                )
            }
            None => extension(VectorSearchNode::try_new(
                source,
                scanner.dataset.clone(),
                query.clone(),
            )?),
        };
        let taken = with_take(
            LogicalPlanBuilder::new(searched),
            scanner,
            &postfilter_columns,
        )?;
        // A vector search promises rows in distance order, and the take between the search and
        // here does not preserve it: `FilteredReadExec` reports no output ordering, so a
        // multi-partition input gets a plain coalesce and the order is lost. Stating the ordering
        // in the plan is the only way to hold the contract — inheriting it from whichever physical
        // operator happened to sort is what made the imperative path's version of this fragile.
        taken.sort([
            col(DIST_COL).sort(true, false),
            col(ROW_ID).sort(true, false),
        ])?
    } else {
        LogicalPlanBuilder::new(source)
    };

    // Postfilters, innermost first: an FTS `query_filter` runs before the expression filter,
    // matching `FilterPlan::refine_filter`.
    if !prefilter {
        if let Some(query) = &prepared.fts_filter {
            builder = LogicalPlanBuilder::new(fts::build_match_filter(
                builder.plan().clone(),
                &scanner.dataset,
                query,
                &take_settings(scanner),
            )?);
        }
        if let Some(query) = &prepared.vector_filter {
            builder = LogicalPlanBuilder::new(extension(VectorRerankNode::try_new(
                builder.plan().clone(),
                &scanner.dataset,
                query.clone(),
            )?));
        }
        if let Some(filter) = filter {
            builder = builder.filter(filter)?;
        }
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

    builder = builder.project(output_exprs(scanner, prepared)?)?;
    Ok(builder.build()?)
}

fn scan_leaf(scanner: &Scanner) -> Result<LogicalPlan> {
    let source = LanceScanSource::new(scanner.dataset.clone(), source_options(scanner))?;
    Ok(
        LogicalPlanBuilder::scan(TABLE_NAME, provider_as_source(Arc::new(source)), None)?
            .build()?,
    )
}

/// The number of results the search itself should produce, before any post-filtering.
///
/// An offset means the search has to return `limit + offset` so the limit node has something to
/// skip past; an offset with no limit means it cannot be bounded at all.
fn search_limit(scanner: &Scanner) -> Option<usize> {
    match (scanner.limit, scanner.offset) {
        (Some(limit), Some(offset)) => Some((limit + offset) as usize),
        (Some(limit), None) => Some(limit as usize),
        (None, _) => None,
    }
}

/// Insert late materialization above a search, unless the search already produced everything.
///
/// A search emits only its scoring columns; the user's columns have to be fetched by row id
/// afterwards. Plain scans need no equivalent — DataFusion's projection pushdown puts the column
/// list into the leaf directly.
fn with_take(
    builder: LogicalPlanBuilder,
    scanner: &Scanner,
    extra_columns: &[String],
) -> Result<LogicalPlanBuilder> {
    let projection = scanner
        .projection_plan
        .physical_projection
        .clone()
        .union_columns(extra_columns, OnMissing::Ignore)?;
    let input = builder.plan().clone();
    if LanceTakeNode::is_noop(&input, &projection)? {
        return Ok(builder);
    }
    let take = LanceTakeNode::try_new(
        input,
        scanner.dataset.clone(),
        projection,
        take_settings(scanner),
    )?;
    Ok(LogicalPlanBuilder::new(extension(take)))
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
fn output_exprs(scanner: &Scanner, prepared: &PreparedQueries) -> Result<Vec<Expr>> {
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

    let named = |exprs: &[Expr], name: &str| {
        exprs
            .iter()
            .any(|expr| expr.schema_name().to_string() == name)
    };

    // The scoring columns are appended even when the user did not ask for them. That is legacy
    // behavior the imperative path implements in `calculate_final_projection`; replicating it
    // here is what lets the two paths be compared row for row.
    if prepared.full_text.is_some() && prepared.element_granularity && !named(&exprs, DOC_INDEX_COL)
    {
        exprs.push(col(DOC_INDEX_COL));
    }
    if scanner.autoproject_scoring_columns {
        if scanner.nearest.is_some() && !named(&exprs, DIST_COL) {
            exprs.push(col(DIST_COL));
        }
        if prepared.full_text.is_some() && !named(&exprs, SCORE_COL) {
            exprs.push(col(SCORE_COL));
        }
    }

    // `with_row_id`/`with_row_address` promise their column is *last*, so the scoring columns that
    // were just appended have to move ahead of it.
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
        fast_search: scanner.fast_search,
        rows: None,
        filter_plan: None,
        overlay_block: None,
        legacy_scanner: super::source::v1::is_legacy(&scanner.dataset)
            .then(|| Arc::new(scanner.clone())),
    }
}
