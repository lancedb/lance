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
use lance_core::{
    ROW_ADDR, ROW_ID,
    datatypes::{OnMissing, Projection},
};
use lance_index::vector::DIST_COL;

use super::prepare::PreparedQueries;
use super::source::{LanceScanSource, ScanSourceOptions};
use super::{LanceTakeNode, TakeSettings, VectorRerankNode, VectorSearchNode};
use crate::dataset::scanner::{ColumnOrdering, MaterializationStyle};
use crate::dataset::{Dataset, Scanner};
use crate::io::exec::knn::QUERY_INDEX_COL;
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
    let prefilter = scanner.prefilter && scanner.nearest.is_some();

    let mut source = scan;
    if prefilter && let Some(filter) = filter.clone() {
        source = LogicalPlanBuilder::new(source).filter(filter)?.build()?;
    }

    // An aggregate replaces the output projection, so late materialization above a search has
    // nothing to materialize on its behalf — only whatever the aggregate itself reads. `COUNT(*)`
    // reads nothing, and the take then drops out entirely rather than reading every column and
    // projecting it away.
    let take_projection = match &scanner.aggregate {
        Some(aggregate) => {
            let columns = aggregate
                .group_by
                .iter()
                .chain(aggregate.aggregates.iter())
                .flat_map(|expr| {
                    expr.column_refs()
                        .into_iter()
                        .map(|column| column.name.clone())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            scanner
                .dataset
                .empty_projection()
                .union_columns(&columns, OnMissing::Ignore)?
        }
        None => scanner.projection_plan.physical_projection.clone(),
    };

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
    let mut builder = if let Some(query) = &scanner.nearest {
        let searched = {
            let search = VectorSearchNode::try_new(source, scanner.dataset.clone(), query.clone())?;
            extension(if scanner.is_batch_nearest {
                search.with_batch_queries(scanner.nearest_query_count)?
            } else {
                search
            })
        };
        let taken = with_take(
            LogicalPlanBuilder::new(searched),
            scanner,
            &take_projection,
            &postfilter_columns,
        )?;
        // A vector search promises rows in distance order, and the take between the search and
        // here does not preserve it: `FilteredReadExec` reports no output ordering, so a
        // multi-partition input gets a plain coalesce and the order is lost. Stating the ordering
        // in the plan is the only way to hold the contract — inheriting it from whichever physical
        // operator happened to sort is what made the imperative path's version of this fragile.
        //
        // A batch search interleaves every query's results, so they are grouped by query first.
        let mut ordering = Vec::with_capacity(3);
        if scanner.is_batch_nearest {
            ordering.push(col(QUERY_INDEX_COL).sort(true, false));
        }
        ordering.push(col(DIST_COL).sort(true, false));
        ordering.push(col(ROW_ID).sort(true, false));
        taken.sort(ordering)?
    } else {
        LogicalPlanBuilder::new(source)
    };

    // Postfilters, innermost first, matching `FilterPlan::refine_filter`.
    if !prefilter {
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

/// Read settings a take must honor, lifted off the `Scanner`. `None` fragments means "all of
/// them", which is a different plan from an explicit list of every fragment.
fn take_settings(scanner: &Scanner) -> TakeSettings {
    TakeSettings {
        fragments: scanner.fragments.clone().map(Arc::new),
        batch_size: scanner.batch_size.map(|size| size as u32),
    }
}

fn scan_leaf(scanner: &Scanner) -> Result<LogicalPlan> {
    let source = LanceScanSource::new(scanner.dataset.clone(), source_options(scanner))?;
    Ok(
        LogicalPlanBuilder::scan(TABLE_NAME, provider_as_source(Arc::new(source)), None)?
            .build()?,
    )
}

/// Insert late materialization above a search, unless the search already produced everything.
///
/// A search emits only its scoring columns; the user's columns have to be fetched by row id
/// afterwards. Plain scans need no equivalent — DataFusion's projection pushdown puts the column
/// list into the leaf directly.
fn with_take(
    builder: LogicalPlanBuilder,
    scanner: &Scanner,
    projection: &Projection,
    extra_columns: &[String],
) -> Result<LogicalPlanBuilder> {
    let projection = projection
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

pub(super) fn extension(node: impl UserDefinedLogicalNodeCore) -> LogicalPlan {
    LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    })
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

    let named = |exprs: &[Expr], name: &str| {
        exprs
            .iter()
            .any(|expr| expr.schema_name().to_string() == name)
    };

    // `_distance` is appended even when the user did not ask for it. That is legacy behavior the
    // imperative path implements in `calculate_final_projection`; replicating it here is what lets
    // the two paths be compared row for row.
    if scanner.autoproject_scoring_columns && scanner.nearest.is_some() && !named(&exprs, DIST_COL)
    {
        exprs.push(col(DIST_COL));
    }

    // Batch nearest exposes the query discriminator as the *first* output column, which is what
    // LanceDB's batch vector search reads it from.
    if scanner.is_batch_nearest {
        if !named(&exprs, QUERY_INDEX_COL) {
            exprs.push(col(QUERY_INDEX_COL));
        }
        move_to_front(&mut exprs, QUERY_INDEX_COL);
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

fn move_to_front(exprs: &mut Vec<Expr>, name: &str) {
    if let Some(position) = exprs
        .iter()
        .position(|expr| expr.schema_name().to_string() == name)
    {
        let expr = exprs.remove(position);
        exprs.insert(0, expr);
    }
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
        // `MaterializationStyle` is documented as affecting plain scans only, and a search plan
        // already takes its columns above the search. Splitting the read below it as well would
        // add a take that fetches the same columns for strictly more rows.
        materialization_style: match scanner.nearest.is_some() {
            true => MaterializationStyle::AllEarly,
            false => scanner.materialization_style.clone(),
        },
        blob_handling: scanner.blob_handling.clone(),
        fast_search: scanner.fast_search,
        index_segments: scanner.index_segments.clone().map(Arc::new),
        include_deleted_rows: scanner.include_deleted_rows,
        rows: None,
        filter_plan: None,
        overlay_block: None,
        legacy_scanner: super::source::v1::is_legacy(&scanner.dataset)
            .then(|| Arc::new(scanner.clone())),
    }
}
