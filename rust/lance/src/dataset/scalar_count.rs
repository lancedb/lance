// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::Schema as ArrowSchema;
use futures::stream::{self, StreamExt, TryStreamExt};
use lance_core::utils::mask::{RowAddrMask, RowAddrSelection, RowAddrTreeMap, RowSetOps};
use lance_datafusion::planner::Planner;
use lance_index::{metrics::NoOpMetricsCollector, scalar::expression::PlannerIndexExt};
use lance_table::format::Fragment;

use crate::dataset::{Dataset, fragment::FileFragment, rowids, scanner::ExprFilter};
use crate::index::prefilter::DatasetPreFilter;
use crate::index::{DatasetIndexInternalExt, coverage::fragments_covered_by_scalar_index_query};
use crate::{Error, Result};

pub(super) async fn try_count_rows_with_exact_scalar_index(
    dataset: &Dataset,
    filter: &str,
) -> Result<Option<usize>> {
    if dataset
        .manifest
        .fragments
        .iter()
        .any(|fragment| fragment.physical_rows.is_none())
    {
        return Ok(None);
    }

    let schema = dataset.schema();
    let expr = ExprFilter::Sql(filter.to_owned()).to_datafusion(schema, schema)?;
    let planner = Planner::new(Arc::new(ArrowSchema::from(schema)));
    let index_info = dataset.scalar_index_info().await?;
    let filter_plan = planner.create_filter_plan(expr, &index_info, true)?;

    if !filter_plan.is_exact_index_search() {
        return Ok(None);
    }

    let index_query = filter_plan
        .index_query
        .expect("exact scalar index search should always have an index query");
    let covered_fragments = fragments_covered_by_scalar_index_query(dataset, &index_query).await?;

    if dataset
        .manifest
        .fragments
        .iter()
        .any(|fragment| !covered_fragments.contains(fragment.id as u32))
    {
        return Ok(None);
    }

    let selected_fragments = dataset
        .manifest
        .fragments
        .iter()
        .filter(|fragment| covered_fragments.contains(fragment.id as u32))
        .cloned()
        .collect::<Vec<_>>();

    let mask = match index_query.evaluate(dataset, &NoOpMetricsCollector).await? {
        lance_index::scalar::expression::IndexExprResult::Exact(mask) => mask,
        lance_index::scalar::expression::IndexExprResult::AtMost(_)
        | lance_index::scalar::expression::IndexExprResult::AtLeast(_) => return Ok(None),
    };

    let deletion_mask = if let Some(deletion_mask_fut) =
        DatasetPreFilter::create_deletion_mask(Arc::new(dataset.clone()), covered_fragments)
    {
        Some((*deletion_mask_fut.await?).clone())
    } else {
        None
    };

    Ok(Some(
        count_rows_for_exact_mask(dataset, mask, deletion_mask.as_ref(), &selected_fragments)
            .await?,
    ))
}

async fn count_rows_for_exact_mask(
    dataset: &Dataset,
    mask: RowAddrMask,
    deletion_mask: Option<&RowAddrMask>,
    fragments: &[Fragment],
) -> Result<usize> {
    match mask {
        RowAddrMask::AllowList(mut allow_list) => {
            if let Some(deletion_mask) = deletion_mask {
                match deletion_mask {
                    RowAddrMask::AllowList(live_rows) => allow_list &= live_rows,
                    RowAddrMask::BlockList(deleted_rows) => allow_list -= deleted_rows,
                }
            }
            if dataset.manifest.uses_stable_row_ids() {
                let fragment_ids = rowids::load_row_id_sequences(dataset, fragments)
                    .map_ok(|(_, sequence)| RowAddrTreeMap::from(sequence.as_ref()))
                    .try_fold(RowAddrTreeMap::new(), |mut acc, tree| async move {
                        acc |= tree;
                        Ok(acc)
                    })
                    .await?;
                allow_list &= &fragment_ids;
            } else {
                allow_list.retain_fragments(fragments.iter().map(|fragment| fragment.id as u32));
            }
            count_selected_rows(dataset, &allow_list, fragments).await
        }
        RowAddrMask::BlockList(block_list) if block_list.is_empty() => match deletion_mask {
            Some(RowAddrMask::AllowList(live_rows)) => {
                count_selected_rows(dataset, live_rows, fragments).await
            }
            _ => count_rows_in_fragments(dataset, fragments).await,
        },
        RowAddrMask::BlockList(mut block_list) => {
            if let Some(RowAddrMask::BlockList(deleted_rows)) = deletion_mask {
                block_list -= deleted_rows;
            }
            if dataset.manifest.uses_stable_row_ids() {
                let live_rows = match deletion_mask {
                    Some(RowAddrMask::AllowList(live_rows)) => Some(live_rows),
                    _ => None,
                };
                let sequences = rowids::load_row_id_sequences(dataset, fragments)
                    .map_ok(|(_, sequence)| sequence)
                    .try_collect::<Vec<_>>()
                    .await?;
                Ok(sequences
                    .into_iter()
                    .map(|sequence| {
                        sequence
                            .iter()
                            .filter(|row_id| {
                                live_rows.is_none_or(|live_rows| live_rows.contains(*row_id))
                            })
                            .filter(|row_id| !block_list.contains(*row_id))
                            .count()
                    })
                    .sum())
            } else {
                let total_rows = count_rows_in_fragments(dataset, fragments).await?;
                let blocked_rows = count_selected_rows(dataset, &block_list, fragments).await?;
                Ok(total_rows - blocked_rows)
            }
        }
    }
}

async fn count_rows_in_fragments(dataset: &Dataset, fragments: &[Fragment]) -> Result<usize> {
    let dataset = Arc::new(dataset.clone());
    stream::iter(fragments.iter().cloned())
        .map(move |fragment| {
            let dataset = dataset.clone();
            async move { FileFragment::new(dataset, fragment).count_rows(None).await }
        })
        .buffer_unordered(16)
        .try_fold(0_usize, |acc, count| async move { Ok(acc + count) })
        .await
}

async fn count_selected_rows(
    dataset: &Dataset,
    row_addrs: &RowAddrTreeMap,
    fragments: &[Fragment],
) -> Result<usize> {
    let fragment_map = fragments
        .iter()
        .cloned()
        .map(|fragment| (fragment.id as u32, fragment))
        .collect::<HashMap<_, _>>();
    let full_fragment_ids = row_addrs
        .iter()
        .filter_map(|(fragment_id, selection)| match selection {
            RowAddrSelection::Full => Some(*fragment_id),
            RowAddrSelection::Partial(_) => None,
        })
        .collect::<Vec<_>>();
    let dataset = Arc::new(dataset.clone());
    let full_fragment_counts = stream::iter(full_fragment_ids)
        .map(move |fragment_id| {
            let dataset = dataset.clone();
            let fragment = fragment_map.get(&fragment_id).cloned();
            async move {
                let fragment = fragment.ok_or_else(|| {
                    Error::internal(format!(
                        "Scalar index count referenced unknown fragment id {}",
                        fragment_id
                    ))
                })?;
                FileFragment::new(dataset, fragment)
                    .count_rows(None)
                    .await
                    .map(|count| (fragment_id, count))
            }
        })
        .buffer_unordered(16)
        .try_collect::<HashMap<_, _>>()
        .await?;

    Ok(row_addrs
        .iter()
        .map(|(fragment_id, selection)| match selection {
            RowAddrSelection::Full => full_fragment_counts[fragment_id],
            RowAddrSelection::Partial(bitmap) => bitmap.len() as usize,
        })
        .sum())
}
