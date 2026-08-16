// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage 4: lowering FTS logical nodes to execution plans.

use std::sync::Arc;

use arrow_schema::SortOptions;
use datafusion::functions_aggregate;
use datafusion::logical_expr::UserDefinedLogicalNode;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{ExecutionPlan, Partitioning, expressions};
use datafusion_physical_expr::{PhysicalExpr, PhysicalSortExpr};
use lance_core::ROW_ID;
use lance_index::scalar::inverted::query::{FtsQuery, MatchQuery, Operator};
use lance_index::scalar::inverted::{SCORE_COL, fts_schema};
use lance_index::scalar::registry::VALUE_COLUMN_NAME;
use lance_select::mask::RowAddrMask;
use lance_table::format::Fragment;

use super::super::nodes::PrefilterSourceKind;
use crate::dataset::{Dataset, Scanner};
use crate::io::exec::PreFilterSource;
use crate::io::exec::fts::{
    BoolSlot, BooleanQueryExec, BoostQueryExec, CompoundQueryExec, FlatMatchFilterExec,
    FlatMatchQueryExec, FtsDocumentExec, MatchQueryExec, PhraseQueryExec,
    build_boolean_query_children_with_schema,
};
use crate::{Error, Result};

use super::*;

// ---------------------------------------------------------------------------------------------
// Stage 4: lowering
// ---------------------------------------------------------------------------------------------

/// Lower any FTS node. Returns `None` for nodes this module does not own.
pub fn plan_extension(
    node: &dyn UserDefinedLogicalNode,
    inputs: &[Arc<dyn ExecutionPlan>],
) -> Option<Result<Arc<dyn ExecutionPlan>>> {
    if let Some(leaf) = node.as_any().downcast_ref::<FtsLeafNode>() {
        return Some(plan_leaf(leaf, inputs.first().cloned()?));
    }
    if let Some(compound) = node.as_any().downcast_ref::<FtsCompoundNode>() {
        return Some(plan_compound(compound, inputs));
    }
    if let Some(scorer) = node.as_any().downcast_ref::<FtsCompoundScorerNode>() {
        return Some(plan_compound_scorer(scorer, inputs.first().cloned()?));
    }
    if let Some(filter) = node.as_any().downcast_ref::<FtsMatchFilterNode>() {
        return Some(plan_match_filter(filter, inputs.first().cloned()?));
    }
    None
}

fn prefilter_source(kind: PrefilterSourceKind, input: Arc<dyn ExecutionPlan>) -> PreFilterSource {
    match kind {
        PrefilterSourceKind::None => PreFilterSource::None,
        PrefilterSourceKind::ChildRowIds => PreFilterSource::FilteredRowIds(input),
    }
}

fn plan_leaf(node: &FtsLeafNode, input: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    match &node.resolution {
        Some(FtsAccessPath::Index { segments }) => {
            let prefilter = prefilter_source(node.prefilter, input);
            let block = node
                .overlay_block
                .as_ref()
                .map(|rows| RowAddrMask::from_block(rows.as_ref().clone()));
            let plan: Arc<dyn ExecutionPlan> = match &node.query {
                FtsQuery::Match(query) => {
                    let mut exec = MatchQueryExec::new_with_segments_and_document_granularity(
                        node.dataset.clone(),
                        query.clone(),
                        node.params.clone(),
                        prefilter,
                        segments.clone(),
                        node.granularity,
                    );
                    if let Some(block) = block {
                        exec = exec.with_overlay_block(block);
                    }
                    Arc::new(exec)
                }
                FtsQuery::Phrase(query) => {
                    let mut exec = PhraseQueryExec::new_with_segments_and_document_granularity(
                        node.dataset.clone(),
                        query.clone(),
                        node.params.clone(),
                        prefilter,
                        segments.clone(),
                        node.granularity,
                    );
                    if let Some(block) = block {
                        exec = exec.with_overlay_block(block);
                    }
                    Arc::new(exec)
                }
                other => {
                    return Err(Error::internal(format!(
                        "FtsLeaf holds a compound query: {other}"
                    )));
                }
            };
            Ok(plan)
        }
        // An unresolved leaf means the rule did not run; a flat scan is always correct.
        Some(FtsAccessPath::Flat) | None => plan_flat_leaf(node, input),
    }
}

/// The brute-force path: feed the input's text to `FlatMatchQueryExec`.
///
/// A phrase query becomes an `And` match with a slop parameter, which is how the imperative path
/// scores phrases over unindexed rows too.
fn plan_flat_leaf(
    node: &FtsLeafNode,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let (query, params) = match &node.query {
        FtsQuery::Match(query) => (query.clone(), node.params.clone()),
        FtsQuery::Phrase(phrase) => (
            MatchQuery::new(phrase.terms.clone())
                .with_column(phrase.column.clone())
                .with_operator(Operator::And)
                .with_document_granularity(node.granularity),
            node.params.clone().with_phrase_slop(Some(phrase.slop)),
        ),
        other => {
            return Err(Error::internal(format!(
                "FtsLeaf holds a compound query: {other}"
            )));
        }
    };

    let document_column = if node.field.has_lists() {
        VALUE_COLUMN_NAME.to_string()
    } else {
        node.field.canonical_path.clone()
    };
    let input = if node.field.has_lists() {
        Arc::new(FtsDocumentExec::new(input, node.field.clone())) as Arc<dyn ExecutionPlan>
    } else {
        ensure_column_alias(input, &node.dataset, &document_column)?
    };

    let scored = Arc::new(FlatMatchQueryExec::new_with_document_granularity(
        node.dataset.clone(),
        query,
        params,
        input,
        node.granularity,
        document_column,
    )) as Arc<dyn ExecutionPlan>;

    // `combine_fts_leaf_plans` sorts a flat-only leaf when the search is bounded; the index path
    // gets its top-k from the posting lists instead.
    match node.params.limit {
        Some(limit) if !node.retains_input_order => Ok(sort_by_score(scored, Some(limit))?),
        _ => Ok(scored),
    }
}

fn plan_compound(
    node: &FtsCompoundNode,
    inputs: &[Arc<dyn ExecutionPlan>],
) -> Result<Arc<dyn ExecutionPlan>> {
    match &node.kind {
        FtsCompoundKind::Boost => {
            let FtsQuery::Boost(query) = &node.query else {
                return Err(Error::internal(
                    "FtsCompound{Boost} holds a non-boost query",
                ));
            };
            let [positive, negative] = inputs else {
                return Err(Error::internal("boost query requires exactly two children"));
            };
            Ok(Arc::new(BoostQueryExec::new(
                query.clone(),
                node.params.clone(),
                positive.clone(),
                negative.clone(),
            )))
        }
        FtsCompoundKind::MultiMatch => plan_multi_match(node, inputs),
        FtsCompoundKind::Boolean {
            should,
            must,
            must_not,
        } => {
            let FtsQuery::Boolean(query) = &node.query else {
                return Err(Error::internal(
                    "FtsCompound{Boolean} holds a non-boolean query",
                ));
            };
            if inputs.len() != should + must + must_not {
                return Err(Error::internal("boolean query child arity changed"));
            }
            let schema = fts_schema(node.granularity);
            let (should_children, rest) = inputs.split_at(*should);
            let (must_children, must_not_children) = rest.split_at(*must);

            let should_plan = build_boolean_query_children_with_schema(
                BoolSlot::Should,
                should_children.to_vec(),
                schema.clone(),
            )?
            .ok_or_else(|| {
                Error::internal("boolean should planning returned no execution plan".to_string())
            })?;
            let must_plan = build_boolean_query_children_with_schema(
                BoolSlot::Must,
                must_children.to_vec(),
                schema.clone(),
            )?;
            let must_not_plan = build_boolean_query_children_with_schema(
                BoolSlot::MustNot,
                must_not_children.to_vec(),
                schema,
            )?
            .ok_or_else(|| {
                Error::internal("boolean must-not planning returned no execution plan".to_string())
            })?;

            if *should == 0 && must_plan.is_none() {
                return Err(Error::invalid_input(
                    "boolean query must have at least one should/must query".to_string(),
                ));
            }
            Ok(Arc::new(BooleanQueryExec::new(
                query.clone(),
                node.params.clone(),
                should_plan,
                must_plan,
                must_not_plan,
            )))
        }
    }
}

/// Union the sub-matches, keep the best score per row, and take the top k.
///
/// Everything here is stock relational algebra — union, group-by-max, sort-with-fetch — which is
/// a finding in itself: `MultiMatch` needs no Lance-specific execution at all.
fn plan_multi_match(
    node: &FtsCompoundNode,
    inputs: &[Arc<dyn ExecutionPlan>],
) -> Result<Arc<dyn ExecutionPlan>> {
    let unioned = UnionExec::try_new(inputs.to_vec())?;
    let schema = unioned.schema();
    let single = Arc::new(RepartitionExec::try_new(
        unioned,
        Partitioning::RoundRobinBatch(1),
    )?);
    let deduped = Arc::new(AggregateExec::try_new(
        AggregateMode::Single,
        PhysicalGroupBy::new_single(vec![(
            expressions::col(ROW_ID, schema.as_ref())?,
            ROW_ID.to_string(),
        )]),
        vec![Arc::new(
            datafusion_physical_expr::aggregate::AggregateExprBuilder::new(
                functions_aggregate::min_max::max_udaf(),
                vec![expressions::col(SCORE_COL, &schema)?],
            )
            .schema(schema.clone())
            .alias(SCORE_COL)
            .build()?,
        )],
        vec![None],
        single,
        schema,
    )?);
    sort_by_score_and_row_id(deduped, node.params.limit)
}

fn plan_compound_scorer(
    node: &FtsCompoundScorerNode,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    Ok(Arc::new(CompoundQueryExec::new_with_segments(
        node.dataset.clone(),
        node.query.clone(),
        node.params.clone(),
        prefilter_source(node.prefilter, input),
        node.segments.clone(),
    )))
}

fn plan_match_filter(
    node: &FtsMatchFilterNode,
    input: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let input = if node.field.has_lists() {
        input
    } else {
        ensure_column_alias(input, &node.dataset, &node.field.canonical_path)?
    };
    Ok(Arc::new(FlatMatchFilterExec::new_with_resolved_field(
        input,
        node.dataset.clone(),
        node.query.clone(),
        node.params.clone(),
        node.field.clone(),
    )))
}

fn sort_by_score(
    plan: Arc<dyn ExecutionPlan>,
    fetch: Option<usize>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let expr = PhysicalSortExpr {
        expr: expressions::col(SCORE_COL, plan.schema().as_ref())?,
        options: SortOptions {
            descending: true,
            nulls_first: false,
        },
    };
    Ok(Arc::new(
        SortExec::new([expr].into(), plan).with_fetch(fetch),
    ))
}

fn sort_by_score_and_row_id(
    plan: Arc<dyn ExecutionPlan>,
    fetch: Option<usize>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let schema = plan.schema();
    let exprs = [
        PhysicalSortExpr {
            expr: expressions::col(SCORE_COL, schema.as_ref())?,
            options: SortOptions {
                descending: true,
                nulls_first: false,
            },
        },
        PhysicalSortExpr {
            expr: expressions::col(ROW_ID, schema.as_ref())?,
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        },
    ];
    Ok(Arc::new(
        SortExec::new(exprs.into(), plan).with_fetch(fetch),
    ))
}

/// Expose a (possibly nested) field path as a top-level column, as `Scanner::ensure_column_alias`
/// does: the reader produces the containing struct, but the FTS executor wants one named column.
fn ensure_column_alias(
    input: Arc<dyn ExecutionPlan>,
    dataset: &Arc<Dataset>,
    column: &str,
) -> Result<Arc<dyn ExecutionPlan>> {
    let schema = input.schema();
    if schema.column_with_name(column).is_some() {
        return Ok(input);
    }
    let mut exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = schema
        .fields()
        .iter()
        .map(|field| {
            expressions::col(field.name(), schema.as_ref()).map(|expr| (expr, field.name().clone()))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    exprs.push((
        Scanner::create_column_expr(column, dataset.as_ref(), schema.as_ref())?,
        column.to_string(),
    ));
    Ok(Arc::new(ProjectionExec::try_new(exprs, input)?))
}

/// Fragments an FTS index does or does not cover, for the split rule's two branches.
pub fn partition_fragments(
    info: &FtsIndexInfo,
    fragments: &[Fragment],
    covered_side: bool,
) -> Option<Vec<Fragment>> {
    let covered = info.covered_fragments()?;
    Some(
        fragments
            .iter()
            .filter(|fragment| covered.contains(fragment.id as u32) == covered_side)
            .cloned()
            .collect(),
    )
}
