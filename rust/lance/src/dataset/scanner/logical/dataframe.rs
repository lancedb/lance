// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance search as DataFusion `DataFrame` operators.
//!
//! The scanner exposes one fixed query shape: filter, search, sort, limit, project, in that order.
//! Everything the logical path added to make that shape planable — a scan leaf that is a real
//! `TableProvider`, search nodes that are real logical nodes, and a lowering stage that reads only
//! the plan — also makes the shape unnecessary. A `DataFrame` can put a search anywhere, and join,
//! aggregate, or window the result with anything DataFusion can express.
//!
//! ```
//! # use std::sync::Arc;
//! # use datafusion::prelude::{SessionContext, col, lit};
//! # use lance::Result;
//! # use lance::dataset::Dataset;
//! # use lance::datafusion::{LanceContextExt, LanceDataFrameExt};
//! # use lance_index::scalar::FullTextSearchQuery;
//! # use lance_index::vector::Query;
//! # async fn hybrid(dataset: Arc<Dataset>, text: FullTextSearchQuery, vector: Query) -> Result<()> {
//! let ctx = SessionContext::new();
//! let plan = ctx
//!     .read_lance_dataset(dataset)?
//!     .filter(col("category").eq(lit("news")))?
//!     .full_text_search(text)
//!     .await?
//!     .nearest(vector)?
//!     .limit(0, Some(10))?
//!     .lance_plan()
//!     .await?;
//! # let _ = plan;
//! # Ok(())
//! # }
//! ```
//!
//! The filter is pushed into the Lance scan leaf, the text search runs against the inverted index,
//! and the vector search re-scores its matches — one plan, planned once. Note that this is *not*
//! gated by [`is_enabled`](super::is_enabled): there is no imperative equivalent to fall back to.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::catalog::TableProvider;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::logical_expr::{LogicalPlan, Projection as DfProjection};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{DataFrame, SessionContext, col};
use lance_core::ROW_ID;
use lance_core::datatypes::{OnMissing, Projection};
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::{DOC_INDEX_COL, SCORE_COL};
use lance_index::vector::{DIST_COL, Query};

use super::builder::{extension, source_options};
use super::fts::{self, FtsCompoundNode, FtsLeafNode};
use super::source::LanceScanSource;
use super::{LanceTakeNode, TakeSettings, VectorAccessPath, VectorSearchNode, with_lance_source};
use crate::dataset::Dataset;
use crate::{Error, Result};

/// Read a Lance dataset as a `DataFrame` that the Lance scan planner will lower.
pub trait LanceContextExt {
    /// A `DataFrame` over `dataset`, backed by the logical path's own scan leaf.
    ///
    /// Unlike [`SessionContextExt::read_lance`](crate::datafusion::SessionContextExt::read_lance),
    /// this does not go through `Dataset::scan()` — the returned frame's leaf is the same
    /// `TableProvider` the scanner's logical plans are built on, so filters and projections land in
    /// the scan itself and [`LanceDataFrameExt`]'s operators can be stacked on top.
    fn read_lance_dataset(&self, dataset: Arc<Dataset>) -> Result<DataFrame>;
}

impl LanceContextExt for SessionContext {
    fn read_lance_dataset(&self, dataset: Arc<Dataset>) -> Result<DataFrame> {
        let defaults = dataset.scan();
        let source = LanceScanSource::new(dataset, source_options(&defaults))?;
        Ok(self.read_table(Arc::new(source) as Arc<dyn TableProvider>)?)
    }
}

/// Lance search operators for a `DataFrame` whose leaf is a Lance dataset.
#[async_trait]
pub trait LanceDataFrameExt: Sized {
    /// Nearest-neighbour search over this frame's rows, ordered by `_distance`.
    ///
    /// The result carries the frame's existing columns plus `_distance`; the columns are re-read by
    /// row id, so a search never has to carry them through.
    ///
    /// Whether the search uses a vector index is left to the planner — *unless* this frame is
    /// already the result of a search, in which case its rows are the search space and the scoring
    /// is exact. That is the same rule the scanner applies to a vector search over a full-text
    /// filter.
    fn nearest(self, query: Query) -> Result<DataFrame>;

    /// Full-text search over this frame's rows, ordered by descending `_score`.
    ///
    /// Async because a query that names neither a column nor a document granularity is completed
    /// from the dataset's inverted indices, which is I/O.
    async fn full_text_search(self, query: FullTextSearchQuery) -> Result<DataFrame>;

    /// Lower this frame through the Lance scan planner.
    ///
    /// Use this instead of `DataFrame::create_physical_plan`: the Lance nodes need Lance's own
    /// analyzer, optimizer, and physical rules, and the index metadata they read is prefetched
    /// here. The frame's session config and runtime are kept.
    async fn lance_plan(self) -> Result<Arc<dyn ExecutionPlan>>;
}

#[async_trait]
impl LanceDataFrameExt for DataFrame {
    fn nearest(self, query: Query) -> Result<Self> {
        let (state, plan) = self.into_parts();
        let dataset = lance_dataset(&plan)?;
        let plan = carrying_row_ids(plan)?;

        let search = if is_search_result(&plan) {
            // These rows are the search space, so they are scored exactly — and scoring them means
            // reading their vectors, which a search's output does not carry.
            let candidates = fts::take_column(
                plan.clone(),
                &dataset,
                &query.column,
                &TakeSettings::default(),
            )?;
            VectorSearchNode::try_new(candidates, dataset.clone(), query)?
                .with_resolution(VectorAccessPath::Flat)
        } else {
            VectorSearchNode::try_new(plan.clone(), dataset.clone(), query)?
        };
        let searched = with_take(extension(search), &dataset, &plan)?;

        Ok(Self::new(state, searched).sort(vec![col(DIST_COL).sort(true, false)])?)
    }

    async fn full_text_search(self, query: FullTextSearchQuery) -> Result<Self> {
        let (state, plan) = self.into_parts();
        let dataset = lance_dataset(&plan)?;
        let plan = carrying_row_ids(plan)?;

        let resolved = dataset
            .scan()
            .resolve_full_text_search_query(&query)
            .await?;
        let mut searched = fts::build_source(plan.clone(), &dataset, &resolved, None)?;
        // A list-element query scores each matching element, so one row can come back several
        // times. Rows are what a frame's later operators expect.
        if searched
            .schema()
            .has_column_with_unqualified_name(DOC_INDEX_COL)
        {
            searched = fts::dedupe_rows(searched)?;
        }
        let searched = with_take(searched, &dataset, &plan)?;

        Ok(Self::new(state, searched).sort(vec![col(SCORE_COL).sort(false, false)])?)
    }

    async fn lance_plan(self) -> Result<Arc<dyn ExecutionPlan>> {
        let (state, plan) = self.into_parts();
        // The frame's own state has DataFusion's physical rules, not Lance's, and lowering a Lance
        // node depends on them — `EnforceDistribution` in particular decides how a search fans out.
        let state = SessionStateBuilder::new_from_existing(state)
            .with_physical_optimizer_rules(super::physical_optimizer_rules())
            .build();
        super::lower(plan, Arc::new(state)).await
    }
}

/// The dataset a frame reads, recovered from its scan leaf.
fn lance_dataset(plan: &LogicalPlan) -> Result<Arc<Dataset>> {
    let mut dataset = None;
    plan.apply(|node| {
        dataset = with_lance_source(node, |source| source.dataset().clone());
        Ok(match dataset {
            Some(_) => TreeNodeRecursion::Stop,
            None => TreeNodeRecursion::Continue,
        })
    })?;
    dataset.ok_or_else(|| {
        Error::invalid_input(
            "a Lance search can only be added to a DataFrame read with read_lance_dataset"
                .to_string(),
        )
    })
}

/// The same plan, emitting `_rowid`.
///
/// A search identifies its results by row id — that is how a prefilter reaches the index, and how
/// the take above the search reads the columns back. A user's own projection has no reason to keep
/// the column, so it is put back here rather than made a rule of the API.
fn carrying_row_ids(plan: LogicalPlan) -> Result<LogicalPlan> {
    if plan.schema().has_column_with_unqualified_name(ROW_ID) {
        return Ok(plan);
    }
    let plan = plan
        .transform_up(|node| {
            let LogicalPlan::Projection(projection) = &node else {
                return Ok(Transformed::no(node));
            };
            if !projection
                .input
                .schema()
                .has_column_with_unqualified_name(ROW_ID)
            {
                return Ok(Transformed::no(node));
            }
            let mut exprs = projection.expr.clone();
            exprs.push(col(ROW_ID));
            Ok(Transformed::yes(LogicalPlan::Projection(
                DfProjection::try_new(exprs, projection.input.clone())?,
            )))
        })?
        .data;

    if !plan.schema().has_column_with_unqualified_name(ROW_ID) {
        return Err(Error::invalid_input(format!(
            "a Lance search needs {ROW_ID}, and nothing below this point in the plan produces it",
        )));
    }
    Ok(plan)
}

/// Whether this plan already scored its rows, and so is a candidate set rather than a table.
fn is_search_result(plan: &LogicalPlan) -> bool {
    let mut found = false;
    let _ = plan.apply(|node| {
        let LogicalPlan::Extension(extension) = node else {
            return Ok(TreeNodeRecursion::Continue);
        };
        let node = extension.node.as_any();
        found = node.is::<VectorSearchNode>()
            || node.is::<FtsLeafNode>()
            || node.is::<FtsCompoundNode>();
        Ok(match found {
            true => TreeNodeRecursion::Stop,
            false => TreeNodeRecursion::Continue,
        })
    });
    found
}

/// Re-read the columns `before` carried, which a search's output does not.
fn with_take(
    searched: LogicalPlan,
    dataset: &Arc<Dataset>,
    before: &LogicalPlan,
) -> Result<LogicalPlan> {
    let columns = before
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect::<Vec<_>>();
    let projection =
        Projection::empty(dataset.clone() as Arc<dyn lance_core::datatypes::Projectable>)
            .union_columns(&columns, OnMissing::Ignore)?;
    if LanceTakeNode::is_noop(&searched, &projection)? {
        return Ok(searched);
    }
    Ok(extension(LanceTakeNode::try_new(
        searched,
        dataset.clone(),
        projection,
        TakeSettings::default(),
    )?))
}
