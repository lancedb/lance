// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage 1: building the FTS subtree of the logical plan.

use std::sync::Arc;

use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder};
use lance_core::ROW_ID;
use lance_core::datatypes::OnMissing;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::DocumentGranularity;
use lance_index::scalar::inverted::query::{FtsQuery, FtsSearchParams};

use super::super::{LanceTakeNode, TakeSettings};
use super::*;
use crate::dataset::Dataset;
use crate::index::scalar::inverted::resolve_fts_field;
use crate::{Error, Result};

/// Build the subtree for a full-text search source over `input`.
///
/// `limit` is the scanner's limit, which the imperative path folds into the root node's params;
/// compound children get `None` so intermediate results stay complete.
pub fn build_source(
    input: LogicalPlan,
    dataset: &Arc<Dataset>,
    query: &FullTextSearchQuery,
    limit: Option<usize>,
) -> Result<LogicalPlan> {
    let mut params = query.params();
    if params.limit.is_none() {
        params = params.with_limit(limit);
    }
    build_query(input, dataset, &query.query, &params)
}

fn build_query(
    input: LogicalPlan,
    dataset: &Arc<Dataset>,
    query: &FtsQuery,
    params: &FtsSearchParams,
) -> Result<LogicalPlan> {
    let granularity = granularity_of(query)?;
    match query {
        FtsQuery::Match(_) | FtsQuery::Phrase(_) => Ok(extension(FtsLeafNode::try_new(
            input,
            dataset.clone(),
            query.clone(),
            params.clone(),
        )?)),
        // Compound parents require complete child results, so the limit stops here. This is the
        // "recursive planning contract" `FtsSearchParams::limit` documents.
        FtsQuery::Boost(boost) => {
            let unlimited = params.clone().with_limit(None);
            let children = vec![
                build_query(input.clone(), dataset, &boost.positive, &unlimited)?,
                build_query(input, dataset, &boost.negative, &unlimited)?,
            ];
            Ok(extension(FtsCompoundNode::try_new(
                children,
                query.clone(),
                params.clone(),
                FtsCompoundKind::Boost,
                granularity,
            )?))
        }
        FtsQuery::MultiMatch(multi) => {
            let children = multi
                .match_queries
                .iter()
                .map(|child| {
                    build_query(
                        input.clone(),
                        dataset,
                        &FtsQuery::Match(child.clone()),
                        params,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(extension(FtsCompoundNode::try_new(
                children,
                query.clone(),
                params.clone(),
                FtsCompoundKind::MultiMatch,
                granularity,
            )?))
        }
        FtsQuery::Boolean(boolean) => {
            let unlimited = params.clone().with_limit(None);
            let mut children = Vec::with_capacity(
                boolean.should.len() + boolean.must.len() + boolean.must_not.len(),
            );
            for child in boolean
                .should
                .iter()
                .chain(&boolean.must)
                .chain(&boolean.must_not)
            {
                children.push(build_query(input.clone(), dataset, child, &unlimited)?);
            }
            Ok(extension(FtsCompoundNode::try_new(
                children,
                query.clone(),
                params.clone(),
                FtsCompoundKind::Boolean {
                    should: boolean.should.len(),
                    must: boolean.must.len(),
                    must_not: boolean.must_not.len(),
                },
                granularity,
            )?))
        }
    }
}

fn granularity_of(query: &FtsQuery) -> Result<DocumentGranularity> {
    let missing = || Error::internal("FTS query document granularity was not resolved".to_string());
    match query {
        FtsQuery::Match(q) => q.document_granularity.ok_or_else(missing),
        FtsQuery::Phrase(q) => q.document_granularity.ok_or_else(missing),
        FtsQuery::Boost(q) => granularity_of(&q.positive),
        FtsQuery::MultiMatch(q) => q
            .match_queries
            .first()
            .and_then(|child| child.document_granularity)
            .ok_or_else(missing),
        FtsQuery::Boolean(q) => q
            .should
            .iter()
            .chain(&q.must)
            .chain(&q.must_not)
            .next()
            .ok_or_else(|| {
                Error::invalid_input(
                    "boolean query must have at least one should/must query".to_string(),
                )
            })
            .and_then(granularity_of),
    }
}

/// Re-rank an already-scored input by BM25, for `full_text_search` combined with a vector
/// `query_filter`.
///
/// A `Match` re-rank is just a flat leaf over the input, which is why it needs no node of its
/// own. Anything else is a join against an independently planned FTS tree — and that join is a
/// stock DataFusion node, not a Lance one.
pub fn build_rerank(
    input: LogicalPlan,
    scan: LogicalPlan,
    dataset: &Arc<Dataset>,
    query: &FullTextSearchQuery,
    limit: Option<usize>,
    settings: &TakeSettings,
) -> Result<LogicalPlan> {
    match &query.query {
        // The imperative `fts_rerank` uses the query's own params here, without folding in the
        // scanner limit: the input is already bounded by the upstream search.
        FtsQuery::Match(match_query) => {
            let params = query.params();
            let column = match_query.column.clone().ok_or_else(|| {
                Error::invalid_input("the column must be specified in the query".to_string())
            })?;
            let granularity = match_query.document_granularity.ok_or_else(|| {
                Error::internal("FTS Match query granularity was not resolved".to_string())
            })?;
            let field = resolve_fts_field(dataset.schema(), &column, granularity)?;
            let input = take_column(input, dataset, &field.root_column, settings)?;
            Ok(extension(
                FtsLeafNode::try_new(
                    input,
                    dataset.clone(),
                    FtsQuery::Match(match_query.clone()),
                    params,
                )?
                .with_resolution(FtsAccessPath::Flat)
                .retaining_input_order(),
            ))
        }
        other => {
            let mut params = query.params();
            if params.limit.is_none() {
                params = params.with_limit(limit);
            }
            let fts = build_query(scan, dataset, other, &params)?;
            join_on_row_id(input, fts)
        }
    }
}

/// Insert a [`LanceTakeNode`] for `column` unless the input already carries it.
pub fn take_column(
    input: LogicalPlan,
    dataset: &Arc<Dataset>,
    column: &str,
    settings: &TakeSettings,
) -> Result<LogicalPlan> {
    let projection = dataset
        .empty_projection()
        .union_column(column, OnMissing::Error)?;
    if LanceTakeNode::is_noop(&input, &projection)? {
        return Ok(input);
    }
    Ok(extension(LanceTakeNode::try_new(
        input,
        dataset.clone(),
        projection,
        settings.clone(),
    )?))
}

/// Inner-join two scored plans on `_rowid`, keeping one copy of each column.
fn join_on_row_id(left: LogicalPlan, right: LogicalPlan) -> Result<LogicalPlan> {
    // Both sides carry an unqualified `_rowid`, so the join key would be ambiguous without
    // relation aliases.
    let left = LogicalPlanBuilder::new(left).alias("search")?;
    let right = LogicalPlanBuilder::new(right).alias("fts")?.build()?;
    let key = |relation: &str| {
        datafusion::common::Column::new(
            Some(datafusion::common::TableReference::bare(relation)),
            ROW_ID,
        )
    };
    // `join_on` with an equality predicate lowers to a `NestedLoopJoinExec`; naming the keys is
    // what makes it a hash join, and a hash join also emits in probe-side (FTS score) order,
    // which is the ordering the imperative path's `HashJoinExec` produces.
    let joined = left.join(
        right,
        datafusion::logical_expr::JoinType::Inner,
        (vec![key("search")], vec![key("fts")]),
        None,
    )?;

    // Drop the right side's duplicate `_rowid`, matching the projection the imperative path
    // builds by hand over its `HashJoinExec`.
    let mut exprs = Vec::new();
    let mut seen_row_id = false;
    for (qualifier, field) in joined.schema().iter() {
        if field.name() == ROW_ID {
            if seen_row_id {
                continue;
            }
            seen_row_id = true;
        }
        exprs.push(Expr::Column(datafusion::common::Column::from((
            qualifier,
            field.as_ref(),
        ))));
    }
    Ok(joined.project(exprs)?.build()?)
}

/// Apply an FTS `query_filter` above `input`, as a postfilter.
pub fn build_match_filter(
    input: LogicalPlan,
    dataset: &Arc<Dataset>,
    query: &FullTextSearchQuery,
    settings: &TakeSettings,
) -> Result<LogicalPlan> {
    let FtsQuery::Match(match_query) = &query.query else {
        return Err(Error::not_supported(
            "Only Match queries are supported currently when using FTS as a post-filter",
        ));
    };
    let granularity = match_query.document_granularity.ok_or_else(|| {
        Error::internal("FTS Match query granularity was not resolved".to_string())
    })?;
    let column = match_query.column.clone().ok_or_else(|| {
        Error::invalid_input("the column must be specified in the query".to_string())
    })?;
    let field = resolve_fts_field(dataset.schema(), &column, granularity)?;
    let input = take_column(input, dataset, &field.root_column, settings)?;
    Ok(extension(FtsMatchFilterNode::try_new(
        input,
        dataset.clone(),
        match_query.clone(),
        query.params(),
    )?))
}

/// Deduplicate an FTS filter's element-level hits down to one row per `_rowid`.
///
/// Stock `Aggregate` with no aggregate expressions — the imperative path builds the same thing
/// out of `RepartitionExec` + `AggregateExec` by hand.
pub fn dedupe_rows(input: LogicalPlan) -> Result<LogicalPlan> {
    Ok(LogicalPlanBuilder::new(input)
        .aggregate(vec![datafusion::prelude::col(ROW_ID)], Vec::<Expr>::new())?
        .build()?)
}
