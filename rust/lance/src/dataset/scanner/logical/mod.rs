// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! **Spike.** A logical-plan read path for the scanner.
//!
//! [`Scanner::create_plan`](crate::dataset::Scanner::create_plan) builds physical plans directly.
//! This module is a prototype of the alternative: assemble the query as a DataFusion
//! [`LogicalPlan`], run a curated rule set over it, and lower it with an `ExtensionPlanner`.
//!
//! Planning is staged so that the parts which must be synchronous can be:
//!
//! 1. **Build** the naive logical plan from `Scanner` state. Sync, no I/O.
//! 2. **Collect** a [`ScanPlanningContext`](context::ScanPlanningContext) by walking that plan and
//!    prefetching everything the later stages need — index metadata, plus the manifest's fragment
//!    metadata, which is already in memory. This is the only stage that does I/O.
//! 3. **Derive** the Lance-owned optimizer rules from the context. Each rule holds an
//!    `Arc<ScanPlanningContext>`, which is how a synchronous `OptimizerRule` gets at information
//!    that took I/O to obtain.
//! 4. **Optimize and lower**, logical rules then physical.
//!
//! This path is off by default. See [`is_enabled`].

pub(super) mod builder;
pub(super) mod context;
pub(super) mod fts;
pub(super) mod nodes;
pub(super) mod planner;
pub(super) mod prepare;
pub(super) mod rules;
pub(super) mod source;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::optimizer::OptimizerRule;
use datafusion::optimizer::optimize_projections::OptimizeProjections;
use datafusion::optimizer::push_down_filter::PushDownFilter;
use datafusion::optimizer::push_down_limit::PushDownLimit;
use datafusion::optimizer::simplify_expressions::SimplifyExpressions;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_optimizer::join_selection::JoinSelection;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
use datafusion::prelude::SessionConfig;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_file::version::ConcreteFileVersion;

use self::context::ScanPlanningContext;
use crate::dataset::Scanner;
use crate::io::exec::get_physical_optimizer;
use crate::{Error, Result};

/// Environment switch for the prototype path. Off unless explicitly set to `1`.
///
/// An env var (rather than a `Scanner` field) keeps the spike from touching the public builder
/// API; where the switch belongs long-term is an open question.
pub fn is_enabled() -> bool {
    std::env::var("LANCE_LOGICAL_SCAN_PLANNER").is_ok_and(|value| value == "1")
}

/// Plan a scan through the logical path.
///
/// Returns [`Error::NotSupported`] for any query shape the spike does not cover, rather than
/// silently falling back — a quiet fallback would make the equivalence tests meaningless.
pub async fn create_plan(scanner: &Scanner) -> Result<Arc<dyn ExecutionPlan>> {
    scanner.validate_options()?;
    ensure_supported(scanner)?;

    let prepared = prepare::PreparedQueries::resolve(scanner).await?;
    let logical_plan = builder::build(scanner, &prepared)?;
    let context = Arc::new(ScanPlanningContext::collect(scanner, &logical_plan).await?);

    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(session_config(scanner))
        .with_optimizer_rules(optimizer_rules(&context))
        .with_physical_optimizer_rules(physical_optimizer_rules())
        .build();

    let optimized = state.optimize(&logical_plan)?;

    let plan = DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(
        planner::LanceExtensionPlanner,
    )])
    .create_physical_plan(&optimized, &state)
    .await?;
    Ok(plan)
}

/// The curated logical rule set: a pinned subset of DataFusion's rules plus the Lance-owned
/// ones derived from the planning context.
///
/// Pinned rather than inherited from `with_default_features()` so that a DataFusion upgrade
/// cannot silently change what runs — the same discipline `get_physical_optimizer` already
/// applies on the physical side. Anything that could move a predicate or limit *across* a search
/// node is deliberately absent.
fn optimizer_rules(
    context: &Arc<ScanPlanningContext>,
) -> Vec<Arc<dyn OptimizerRule + Send + Sync>> {
    let mut rules: Vec<Arc<dyn OptimizerRule + Send + Sync>> = vec![Arc::new(
        rules::ResolveVectorAccessPath::new(context.clone()),
    )];
    // The FTS rules are contributed as a block by the module that owns the FTS nodes, which is
    // the closest the spike gets to the doc's "each index plugin provides its own rules".
    rules.extend(fts::rules(context));
    rules.extend::<Vec<Arc<dyn OptimizerRule + Send + Sync>>>(vec![
        // Before PushDownFilter, so a prefilter predicate is still a `Filter` node that gets
        // duplicated onto both branches; each branch's scan then absorbs its own copy.
        Arc::new(rules::SplitOnIndexCoverage::new(context.clone())),
        // After the split, so the refine lands on the *indexed branch* of a partially-covered
        // search rather than above the union — the nesting the imperative path produces.
        Arc::new(rules::ExpandVectorRefine::new(context.clone())),
        Arc::new(SimplifyExpressions::new()),
        Arc::new(PushDownFilter::new()),
        Arc::new(PushDownLimit::new()),
        // After PushDownFilter, so the predicate has reached its final position, and after the
        // access-path rules, whose choice decides what each child must produce.
        Arc::new(rules::ResolvePrefilterSource),
        Arc::new(OptimizeProjections::new()),
    ]);
    rules
}

/// The physical rule set: Lance's own rules, plus the stock DataFusion rules that hand-built
/// physical plans never needed.
///
/// `get_physical_optimizer` is tuned for the imperative path, where every node is constructed with
/// its final configuration already chosen. A plan lowered from stock logical nodes is not like
/// that: `DefaultPhysicalPlanner` emits a `HashJoinExec` with `PartitionMode::Auto`, which panics
/// at `execute()` unless `JoinSelection` resolves it. It runs first so the Lance rules see a
/// fully-resolved plan, the same way they do today.
fn physical_optimizer_rules() -> Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>> {
    let mut rules: Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>> =
        vec![Arc::new(JoinSelection::new())];
    rules.extend(get_physical_optimizer().rules);
    rules
}

fn session_config(scanner: &Scanner) -> SessionConfig {
    SessionConfig::new().with_target_partitions(
        scanner
            .target_parallelism
            .unwrap_or_else(get_num_compute_intensive_cpus),
    )
}

/// Reject every query shape the spike has not implemented yet.
///
/// Kept as one list so the coverage boundary is readable in a single place, and so each entry can
/// be deleted as the corresponding stage lands.
fn ensure_supported(scanner: &Scanner) -> Result<()> {
    let unsupported = |what: &str| -> Result<()> {
        Err(Error::not_supported_source(
            format!("logical scan planner (spike): {what}").into(),
        ))
    };

    if scanner
        .dataset
        .manifest()
        .data_storage_format
        .lance_file_format()
        == ConcreteFileVersion::V1
    {
        // The legacy read path is a frozen compatibility surface; the spike targets current
        // storage only.
        return unsupported("legacy (v1) storage format");
    }
    if scanner.is_batch_nearest {
        return unsupported("batch vector search");
    }
    if scanner.aggregate.is_some() {
        return unsupported("aggregates");
    }
    if scanner.include_deleted_rows {
        return unsupported("include_deleted_rows");
    }
    if scanner.projection_plan.must_add_row_offset {
        return unsupported("_rowoffset projection");
    }
    if scanner.strict_batch_size {
        return unsupported("strict_batch_size");
    }
    if scanner.index_segments.is_some() {
        return unsupported("index segment selection");
    }
    Ok(())
}
