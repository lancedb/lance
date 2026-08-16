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
use datafusion::physical_plan::projection::ProjectionExec;
use lance_arrow::SchemaExt as ArrowSchemaExt;
use lance_core::datatypes::{OnMissing, Projection};
use lance_file::version::ConcreteFileVersion;
use lance_table::format::Fragment;

use crate::dataset::Scanner;
use crate::dataset::versions;
use crate::io::exec::{FilterPlan as ExprFilterPlan, LanceFilterExec, Planner};
use crate::{Error, Result};

/// Whether the scan leaf must go through the legacy builder for this dataset.
pub fn is_legacy(dataset: &crate::Dataset) -> bool {
    dataset.manifest().data_storage_format.lance_file_format() == ConcreteFileVersion::V1
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
    fragments: Option<Arc<Vec<Fragment>>>,
    scan_range: Option<Range<u64>>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let output_schema: ArrowSchema = (&projection.to_schema()).into();

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
        // The logical path never asks the leaf for tombstones; `include_deleted_rows` is a
        // whole-query option and the builder handles it above the leaf.
        false,
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
