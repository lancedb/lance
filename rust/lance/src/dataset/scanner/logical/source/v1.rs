// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The scan leaf on legacy (v1) storage.
//!
//! V1 is the only place the read path branches on file version, and it branches in exactly one
//! spot: a full scan with a predicate. Everything else the planner builds — takes, scalar index
//! lookups, every search node — is version-agnostic, because `FilteredReadExec` implements
//! `take_all_tasks` for v1 fragments and only refuses `read_ranges_tasks`
//! (`dataset/fragment.rs`). So this module covers the scan-mode case and nothing else.
//!
//! It does not reimplement the legacy scan; it calls it.
//! [`legacy_filtered_read`](crate::dataset::Scanner::legacy_filtered_read) is a frozen
//! compatibility surface, and the way to honor that is to keep using it verbatim. What that costs
//! is a captured `Scanner`: the legacy builder reads a dozen fields straight off `&Scanner`, and
//! changing its signature to take them individually would be a legacy refactor for no read-path
//! benefit.

use std::ops::Range;
use std::sync::Arc;

use arrow_schema::Schema as ArrowSchema;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::ExecutionPlan;
#[allow(deprecated)]
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::projection::ProjectionExec;
use lance_arrow::SchemaExt as ArrowSchemaExt;
use lance_core::datatypes::{OnMissing, Projection};
use lance_file::version::ConcreteFileVersion;
use lance_table::format::Fragment;

use crate::dataset::Scanner;
use crate::dataset::scanner::{BATCH_SIZE_FALLBACK, get_default_batch_size};
use crate::dataset::versions;
use crate::io::exec::{FilterPlan as ExprFilterPlan, LanceFilterExec, Planner, TakeExec};
use crate::{Error, Result};

/// Whether the scan leaf must go through the legacy builder for this dataset.
pub fn is_legacy(dataset: &crate::Dataset) -> bool {
    dataset.manifest().data_storage_format.lance_file_format() == ConcreteFileVersion::V1
}

/// Whether the legacy builder's statistics-pushdown branch is the one it will take.
///
/// Mirrors the branch condition in
/// [`legacy_filtered_read`](crate::dataset::Scanner::legacy_filtered_read), with `is_prefilter`
/// pinned to `false` the way [`scan`] calls it. Says nothing about whether that branch answers the
/// right question — see [`pushdown_covers`].
fn takes_statistics_pushdown(scanner: &Scanner, filter_plan: &ExprFilterPlan) -> bool {
    !filter_plan.has_index_query()
        && filter_plan.has_refine()
        && scanner.batch_size.is_none()
        && scanner.use_stats
        && !scanner.filter_references_version_columns(filter_plan)
}

/// Whether `pushdown_scan` would read what this scan asked for.
///
/// It is the one legacy branch that ignores both arguments the leaf cares about: it takes its
/// columns and its fragment list straight off the `Scanner`. So it only answers the right question
/// where the two agree — the plan's root scan, reading what the caller configured. Every narrowing
/// the rules introduce afterwards is invisible to it: a coverage split's fragment subset, an
/// identity column a search node needs, a late-materialized projection. Where they disagree,
/// [`scan`] turns the branch off rather than let it answer for a different read.
fn pushdown_covers(
    scanner: &Scanner,
    projection: &Projection,
    fragments: Option<&Arc<Vec<Fragment>>>,
) -> Result<bool> {
    if let Some(fragments) = fragments {
        let configured = scanner
            .fragments
            .as_deref()
            .unwrap_or_else(|| scanner.dataset.fragments());
        if !fragments
            .iter()
            .map(|f| f.id)
            .eq(configured.iter().map(|f| f.id))
        {
            return Ok(false);
        }
    }
    // `ScanConfig` carries no version-column flags, so the branch cannot emit those however the
    // scanner is projected.
    let mut emitted = scanner.projection_plan.physical_projection.clone();
    emitted.with_row_last_updated_at_version = false;
    emitted.with_row_created_at_version = false;

    let missing = projection
        .clone()
        .subtract_arrow_schema(&ArrowSchema::from(&emitted.to_schema()), OnMissing::Ignore)?;
    Ok(!missing.has_data_fields()
        && !missing.with_row_id
        && !missing.with_row_addr
        && !missing.with_row_last_updated_at_version
        && !missing.with_row_created_at_version)
}

/// Whether a read of `projection` will go through the legacy statistics-pushdown branch.
///
/// Late materialization has to ask, because that branch reads the `Scanner`'s columns whatever it
/// is handed: narrowing the projection under it would save nothing and cost a take that finds every
/// column already there.
pub fn uses_statistics_pushdown(
    scanner: &Scanner,
    filter_plan: &ExprFilterPlan,
    projection: &Projection,
    fragments: Option<&Arc<Vec<Fragment>>>,
) -> Result<bool> {
    Ok(takes_statistics_pushdown(scanner, filter_plan)
        && pushdown_covers(scanner, projection, fragments)?)
}

/// Fetch `projection`'s columns for the rows `input` has already identified.
///
/// `FilteredReadExec` refuses a row-stream read on legacy files, so every take on v1 comes here
/// instead — the scan leaf's late materialization as well as the plan's own take nodes. Mirrors
/// `Scanner::take_legacy`, including the coalesce: `TakeExec` issues one read per batch, so small
/// batches out of a search would each become their own round trip.
#[allow(deprecated)]
pub fn take(
    dataset: &Arc<crate::Dataset>,
    input: Arc<dyn ExecutionPlan>,
    projection: Projection,
    batch_size: Option<u32>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let batch_size = get_default_batch_size().unwrap_or_else(|| {
        batch_size.map(|size| size as usize).unwrap_or_else(|| {
            std::cmp::max(
                dataset.object_store.as_ref().block_size() / 4,
                BATCH_SIZE_FALLBACK,
            )
        })
    });
    let coalesced = Arc::new(CoalesceBatchesExec::new(input.clone(), batch_size));
    match TakeExec::try_new(dataset.clone(), coalesced, projection)? {
        Some(take) => Ok(Arc::new(take)),
        None => Ok(input),
    }
}

/// Build a v1 scan through the frozen legacy path.
///
/// The plan this returns satisfies the same contract as the v2 leaf: it applies the whole
/// predicate, and its schema is exactly `projection`. Neither is free here — the legacy builder
/// picks one of three branches and only two of them apply the predicate, and none of them promise
/// an exact output schema — so both are re-established on the way out.
///
/// `scan_range` is only ever `Some` when there is no filter, matching what the imperative path
/// asks of the legacy builder.
pub async fn scan(
    scanner: &Scanner,
    filter_plan: &ExprFilterPlan,
    projection: Projection,
    include_deleted_rows: bool,
    fragments: Option<Arc<Vec<Fragment>>>,
    scan_range: Option<Range<u64>>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let output_schema: ArrowSchema = (&projection.to_schema()).into();

    // Nothing in the legacy builder's signature turns the pushdown branch off, so when it would
    // answer for a different read than the one asked for, take away the field its condition reads.
    // The fragment scan it falls back to honors both arguments. Asked before the union below,
    // because that union is for the branch this one is not.
    let restated;
    let scanner = if takes_statistics_pushdown(scanner, filter_plan)
        && !pushdown_covers(scanner, &projection, fragments.as_ref())?
    {
        let mut without_stats = scanner.clone();
        without_stats.use_stats = false;
        restated = without_stats;
        &restated
    } else {
        scanner
    };

    // The branch that does not apply the refine predicate also does not read its columns, so ask
    // for them up front. Where a branch was going to read them anyway this is a no-op, and where
    // it was not they are trimmed off again by `align_to_projection`.
    let read_projection = match filter_plan.refine_expr.as_ref() {
        Some(refine_expr) => projection.union_columns(
            Planner::column_names_in_expr(refine_expr),
            OnMissing::Ignore,
        )?,
        None => projection,
    };

    let planned = versions::filtered_read(
        ConcreteFileVersion::V1,
        scanner,
        filter_plan,
        read_projection,
        include_deleted_rows,
        fragments,
        scan_range,
        // Prefilter-ness is operator ordering in this path, so the leaf has nothing to say about
        // it. `false` is the value that leaves the legacy builder's statistics-pushdown branch
        // reachable, which is worth keeping: it is the only page pruning v1 has.
        false,
    )
    .await?;

    let mut plan = planned.plan;
    // Two of the three legacy branches apply the predicate themselves: the statistics pushdown
    // says so with `filter_pushed_down`, and the scalar-indexed scan applies it as a post-take
    // filter without saying so. The third — the plain fragment scan — leaves it to the caller.
    let already_filtered = planned.filter_pushed_down || filter_plan.has_index_query();
    if let Some(refine_expr) = filter_plan.refine_expr.as_ref()
        && !already_filtered
    {
        plan = Arc::new(LanceFilterExec::try_new(refine_expr.clone(), plan)?);
    }

    align_to_projection(plan, &output_schema)
}

/// Trim the legacy plan's output down to exactly the columns the projection asked for.
///
/// The legacy builder reads whatever its own branches need — the filter's columns as well as the
/// projected ones — and orders system columns to suit itself. `TableProvider::scan` promises a
/// plan whose schema *is* the requested projection, so anything extra or out of order has to be
/// corrected here. On a full scan the schemas already match and no node is added.
fn align_to_projection(
    plan: Arc<dyn ExecutionPlan>,
    output_schema: &ArrowSchema,
) -> Result<Arc<dyn ExecutionPlan>> {
    let input_schema = plan.schema();
    if input_schema.fields() == output_schema.fields() {
        return Ok(plan);
    }
    let mut exprs = Vec::with_capacity(output_schema.fields().len());
    for field in output_schema.fields() {
        let column =
            Column::new_with_schema(field.name(), input_schema.as_ref()).map_err(|_| {
                Error::internal(format!(
                    "the legacy scan did not produce the projected column {}; it produced {:?}",
                    field.name(),
                    input_schema.field_names(),
                ))
            })?;
        exprs.push((
            Arc::new(column) as Arc<dyn PhysicalExpr>,
            field.name().clone(),
        ));
    }
    Ok(Arc::new(ProjectionExec::try_new(exprs, plan)?))
}
