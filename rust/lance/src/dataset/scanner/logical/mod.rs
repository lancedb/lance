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
//! 3. **Derive** the Lance-owned rules from the context. Each rule holds an
//!    `Arc<ScanPlanningContext>`, which is how a synchronous rule gets at information that took
//!    I/O to obtain. Rewrites that must happen for the plan to be *correct* are `AnalyzerRule`s;
//!    only rewrites that are optional are `OptimizerRule`s.
//! 4. **Optimize and lower**, analyzer then logical rules then physical.
//!
//! This path is off by default. See [`is_enabled`].
//!
//! # Layout
//!
//! The framework is split by stage; each index type keeps its own contribution together.
//!
//! ```text
//! builder      stage 1: Scanner -> LogicalPlan
//! prepare      stage 0: the async work that must precede the builder
//! context      stage 2: the one prefetch every later stage reads from
//! source       the scan leaf, as a TableProvider
//! rules        rule plumbing shared by every index type
//! coverage     splitting a search across indexed and unindexed fragments
//! scan_index   recording each scan's scalar index query on its source
//! planner      stage 4: dispatch to each node's lowering
//! take/        late materialization
//! vector/      node, rerank, rules, planner   <- five entry points
//! fts/         node, rules, planner, prefetch <- the same five
//! ```
//!
//! `vector/` and `fts/` are deliberately symmetrical. The design doc proposes that an index type
//! could one day ship its own planning support; two index types reaching the framework through the
//! same five entry points is what makes that a claim rather than a guess. Rule *ordering* stays
//! here, in [`analyzer_rules`] and [`optimizer_rules`], because it is a whole-plan property that no
//! single index can decide.

pub(super) mod builder;
pub(super) mod context;
pub(super) mod coverage;
pub(super) mod fts;
pub(super) mod planner;
pub(super) mod prepare;
pub(super) mod rules;
pub(super) mod scan_index;
pub(super) mod source;
pub(super) mod take;
#[cfg(test)]
mod tests;
pub(super) mod vector;

pub use coverage::*;
pub use rules::*;
pub use scan_index::*;
pub use take::*;
pub use vector::*;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use datafusion::execution::session_state::{SessionState, SessionStateBuilder};
use datafusion::optimizer::optimize_projections::OptimizeProjections;
use datafusion::optimizer::push_down_filter::PushDownFilter;
use datafusion::optimizer::push_down_limit::PushDownLimit;
use datafusion::optimizer::simplify_expressions::SimplifyExpressions;
use datafusion::optimizer::{Analyzer, AnalyzerRule, Optimizer, OptimizerRule};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_optimizer::join_selection::JoinSelection;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
use lance_core::ROW_OFFSET;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_datafusion::exec::get_session_context;
use lance_file::version::ConcreteFileVersion;

use self::context::ScanPlanningContext;
use crate::dataset::Scanner;
use crate::dataset::scanner::{ExprFilter, MaterializationStyle};
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

    if scanner.fragments.as_ref().is_some_and(Vec::is_empty) {
        // An explicit empty fragment list means the scan reads nothing, whatever the query on top
        // of it is. Stating that once here rather than per query kind also skips stage 2: there is
        // no point loading index metadata to plan a scan over no data.
        let schema = Arc::new(logical_plan.schema().as_arrow().clone());
        return Ok(Arc::new(EmptyExec::new(schema)));
    }

    let context = Arc::new(ScanPlanningContext::collect(scanner, &logical_plan).await?);

    let state = planning_state(scanner);

    // Run the two logical stages directly rather than through `SessionState::optimize`, because the
    // rule lists are the one part of planning that genuinely varies per query — each Lance rule
    // holds the `ScanPlanningContext` above — and registering them on a `SessionState` would drag
    // the rest of the state into varying with them. See [`planning_state`].
    let analyzed = Analyzer::with_rules(analyzer_rules(&context)).execute_and_check(
        logical_plan,
        state.config_options(),
        |_, _| {},
    )?;
    let optimized = Optimizer::with_rules(optimizer_rules(&context)).optimize(
        analyzed,
        state.as_ref(),
        |_, _| {},
    )?;

    let plan = DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(
        planner::LanceExtensionPlanner,
    )])
    .create_physical_plan(&optimized, state.as_ref())
    .await?;
    Ok(plan)
}

/// The query-independent half of planning state, built once and shared.
///
/// Built from the same session `execute_plan` will run this plan on, so that lowering sees the
/// runtime it will actually execute against. `DefaultPhysicalPlanner` consults the config when it
/// builds sorts and joins — `sort_spill_reservation_bytes` in particular — and a bare
/// `SessionConfig::new()` here made the resulting plans hold more concurrently in the shared
/// `FairSpillPool` than the imperative path's, which showed up as intermittent
/// `ResourcesExhausted` under parallel test load.
///
/// Cached because building it is not cheap and nothing in it depends on the query:
/// `with_default_features()` re-populates DataFusion's entire catalog of scalar, aggregate, window,
/// and table functions, plus file formats and expression planners, none of which a scan consults.
/// Measured at roughly 37 µs — against an imperative plan-build of 27 µs for the same query, so
/// paying it per query doubled the cost of planning a trivial scan.
///
/// Only the *rule lists* vary per query, and they are a handful of `Arc`s; [`create_plan`] applies
/// them with a standalone [`Analyzer`] and [`Optimizer`] instead of registering them here.
fn planning_state(scanner: &Scanner) -> Arc<SessionState> {
    /// Keyed by the session this state derives from and the parallelism it was built for.
    type StateCache = Mutex<HashMap<(String, usize), Arc<SessionState>>>;

    let session = get_session_context(&scanner.execution_options());
    let target_partitions = scanner
        .target_parallelism
        .unwrap_or_else(get_num_compute_intensive_cpus);
    // The session id identifies the cached `SessionContext` this state derives from, so it stands in
    // for every execution option without this module having to know what they are.
    let key = (session.session_id(), target_partitions);

    static CACHE: OnceLock<StateCache> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    if let Some(state) = cache.get(&key) {
        return state.clone();
    }

    let state = Arc::new(
        SessionStateBuilder::new()
            .with_default_features()
            .with_config(
                session
                    .copied_config()
                    .with_target_partitions(target_partitions),
            )
            .with_runtime_env(session.runtime_env())
            .with_physical_optimizer_rules(physical_optimizer_rules())
            .build(),
    );

    // The key space is (execution options) × (requested parallelism), both of which take a handful
    // of distinct values in practice — `get_session_context` caps its own side at 4. Clearing on
    // overflow rather than evicting an entry keeps this to one line; a workload that actually
    // thrashed it would be better served by a real LRU.
    const MAX_ENTRIES: usize = 8;
    if cache.len() >= MAX_ENTRIES {
        cache.clear();
    }
    cache.insert(key, state.clone());
    state
}

/// The rewrites that must happen for the plan to be *correct*, in the stage that guarantees each
/// runs exactly once and then checks `InvariantLevel::Executable`.
///
/// Unlike [`optimizer_rules`], this list is *not* pinned: it starts from whatever
/// `Analyzer::new()` provides, because `TypeCoercion` in particular has to run over the predicates
/// the builder produced before anything duplicates them onto two branches, and reimplementing that
/// to pin it would be worse than inheriting it. The cost is that a DataFusion upgrade can change
/// this stage's behavior without the change being visible here.
fn analyzer_rules(context: &Arc<ScanPlanningContext>) -> Vec<Arc<dyn AnalyzerRule + Send + Sync>> {
    let mut rules = Analyzer::new().rules;
    rules.push(Arc::new(ResolveVectorAccessPath::new(context.clone())));
    // The FTS rules are contributed as a block by the module that owns the FTS nodes, which is
    // the closest the spike gets to the doc's "each index plugin provides its own rules".
    rules.extend(fts::analyzer_rules(context));
    rules.extend::<Vec<Arc<dyn AnalyzerRule + Send + Sync>>>(vec![
        Arc::new(SplitOnIndexCoverage::searches(context.clone())),
        // After the split, so the refine lands on the *indexed branch* of a partially-covered
        // search rather than above the union — the nesting the imperative path produces.
        Arc::new(ExpandVectorRefine::new(context.clone())),
    ]);
    rules
}

/// The curated logical rule set: a pinned subset of DataFusion's rules plus the Lance-owned ones
/// derived from the planning context.
///
/// Pinned rather than inherited from `with_default_features()` so that a DataFusion upgrade cannot
/// silently change what runs — the same discipline `get_physical_optimizer` already applies on the
/// physical side. Anything that could move a predicate or limit *across* a search node is
/// deliberately absent.
fn optimizer_rules(
    context: &Arc<ScanPlanningContext>,
) -> Vec<Arc<dyn OptimizerRule + Send + Sync>> {
    let mut rules = fts::optimizer_rules(context);
    rules.extend::<Vec<Arc<dyn OptimizerRule + Send + Sync>>>(vec![
        Arc::new(SimplifyExpressions::new()),
        Arc::new(PushDownFilter::new()),
        // The rest of this list is mandatory work that could not run in the analyzer, because each
        // rule reads something `PushDownFilter` is what settles.
        //
        // Which predicates reached the leaf decides the scalar index query, and the index query
        // decides the scan's coverage — so the split runs here for scans and in the analyzer for
        // searches. Before `PushDownLimit`, so a limit is pushed into a union of branches rather
        // than duplicated onto each of them.
        Arc::new(ResolveScalarIndexQuery::new(context.clone())),
        Arc::new(SplitOnIndexCoverage::scans(context.clone())),
        Arc::new(PushDownLimit::new()),
        // Whether a predicate sits below the search is what makes it a prefilter, and pushdown is
        // what moves it there.
        Arc::new(ResolvePrefilterSource),
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

/// Reject every query shape the spike has not implemented yet.
///
/// `Scanner` is destructured **exhaustively** — no `..` — so that adding a field to it fails to
/// compile until someone decides what this path does with it. That is the point of the function:
/// an option this path silently ignores is worse than one it rejects, because it returns different
/// rows with no signal that anything was dropped. Every binding below is either read somewhere in
/// this module, rejected here, or carries a comment saying why it cannot affect the plan.
fn ensure_supported(scanner: &Scanner) -> Result<()> {
    let Scanner {
        dataset,
        projection_plan,
        materialization_style,
        is_batch_nearest,
        aggregate,
        include_deleted_rows,
        strict_batch_size,
        index_segments,

        // Read by the builder, the source, the context, or the session config.
        prefilter: _,
        filter,
        full_text_query: _,
        batch_size: _,
        batch_readahead: _,
        fragment_readahead: _,
        io_buffer_size: _,
        limit: _,
        offset: _,
        ordering: _,
        nearest: _,
        use_scalar_index: _,
        fragments: _,
        fast_search: _,
        file_reader_options: _,
        target_parallelism: _,
        legacy_with_row_id: _,
        legacy_with_row_addr: _,
        autoproject_scoring_columns: _,

        // Folded into something this path does read, at the moment the caller sets it:
        // `blob_handling` into `projection_plan.physical_projection` by `apply_blob_handling`,
        // `batch_size_bytes` into `resolved_file_reader_options`, and
        // `relational_algebra_version` into `index_expr_result_format`.
        blob_handling: _,
        batch_size_bytes: _,
        relational_algebra_version: _,

        // Read only by `legacy_filtered_read` and `scan_fragments`, which are the V1 path this
        // function rejects below.
        use_stats: _,
        ordered: _,

        // Not plan-affecting. The callback is applied by `execute_plan` on the finished plan, and
        // `explicit_projection` only gates a deprecation warning. `nearest_query_count` is
        // meaningful only alongside `is_batch_nearest`, which is rejected.
        scan_stats_callback: _,
        explicit_projection: _,
        nearest_query_count: _,
    } = scanner;

    let unsupported = |what: &str| -> Result<()> {
        Err(Error::not_supported_source(
            format!("logical scan planner (spike): {what}").into(),
        ))
    };

    if dataset.manifest().data_storage_format.lance_file_format() == ConcreteFileVersion::V1 {
        // The legacy read path is a frozen compatibility surface; the spike targets current
        // storage only.
    }
    if *is_batch_nearest {
        return unsupported("batch vector search");
    }
    if aggregate.is_some() {
        return unsupported("aggregates");
    }
    if *include_deleted_rows {
        return unsupported("include_deleted_rows");
    }
    // Two ways to ask for `_rowoffset`, and only one of them is a projection: the imperative path
    // also accepts it in a predicate (`_rowoffset IN (5, 9)`), where it reaches the builder as a
    // column the scan schema does not have and fails as `FieldNotFound` — which reads as a bug
    // rather than a gap. Matched textually because parsing the predicate needs a schema this path
    // has not built yet; over-rejecting a column merely *named* like this one is the safe
    // direction for a guard.
    let filters_on_row_offset = match &filter.expr_filter {
        Some(ExprFilter::Sql(sql)) => sql.contains(ROW_OFFSET),
        Some(ExprFilter::Datafusion(expr)) => expr.to_string().contains(ROW_OFFSET),
        // Substrait conversion runs against the dataset schema, which has no metadata columns.
        Some(ExprFilter::Substrait(_)) | None => false,
    };
    if projection_plan.must_add_row_offset || filters_on_row_offset {
        return unsupported("_rowoffset");
    }
    if *strict_batch_size {
        return unsupported("strict_batch_size");
    }
    if index_segments.is_some() {
        return unsupported("index segment selection");
    }
    if projection_plan.has_output_cols() && projection_plan.physical_projection.is_empty() {
        // `SELECT 1 AS foo` — output columns that read nothing. The imperative path rejects this at
        // `scanner.rs:2846`; this path would otherwise fall into the leaf's "reading nothing is not
        // a thing the reader can do" branch and quietly return row addresses instead.
        return unsupported("a projection of only dynamic expressions");
    }
    if !matches!(materialization_style, MaterializationStyle::Heuristic) {
        // The builder decides take placement itself and never consults `is_early_field`, so an
        // explicit materialization request would be dropped rather than honored. Note that the
        // default is approximated rather than implemented: under `Heuristic` the imperative path
        // reads narrow columns eagerly and takes wide ones after a refine filter, and this path
        // reads everything in one pass. That is a performance gap on filtered scans over wide
        // columns, not a correctness one — see the findings doc.
        return unsupported("explicit materialization style");
    }
    Ok(())
}
