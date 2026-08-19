// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage 2: the bridge between async planning and synchronous rules.
//!
//! An `OptimizerRule` is synchronous, so a rule cannot load index metadata. This struct is how a
//! rule gets it anyway: everything is fetched once, up front, while we are still in an async
//! context, and the rules hold an `Arc` to the result.
//!
//! The prefetch set is deliberately small — **all index metadata, plus the manifest's fragment
//! metadata**. The latter is already in memory (`Manifest::fragments`, carrying `physical_rows`,
//! `deletion_file`, and `overlays`), which is what makes so many of the imperative path's `await`
//! points collapse into synchronous derivations.
//!
//! [`ScalarIndexInfo`](crate::index::ScalarIndexInfo) is the existing instance of this pattern —
//! its doc comment describes the same contract for scalar indices. Later index types generalize it
//! rather than inventing a second mechanism.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::LogicalPlan;
use lance_core::utils::address::RowAddress;
use lance_select::mask::RowAddrTreeMap;

use super::scan_index::with_lance_source;
use crate::Result;
use crate::dataset::rowids::live_row_addrs_to_row_ids;
use crate::dataset::scanner::TakeOperation;
use crate::dataset::{Dataset, row_offsets_to_row_addresses};
use crate::index::{DatasetIndexInternalExt, ScalarIndexInfo};

/// Index and fragment state, captured once per planning invocation.
#[derive(Debug)]
pub struct ScanPlanningContext {
    /// What the filter parser needs to turn a predicate into a scalar index query. Already the
    /// canonical instance of this whole pattern — see the module docs.
    scalar_indices: Arc<ScalarIndexInfo>,
    /// Row selections for take-shaped predicates whose resolution needed I/O, keyed by the
    /// predicate's values rather than its expression: the rules see it after `PushDownFilter` has
    /// moved it, and the values are what survive that unchanged.
    take_rows: HashMap<String, Arc<RowAddrTreeMap>>,
}

impl ScanPlanningContext {
    /// Walk the unoptimized plan for what it will need, then fetch all of it.
    ///
    /// This is the only `async fn` in stages 2 through 4 that is allowed to do I/O.
    ///
    /// Everything comes from the plan — the dataset and the scan-wide settings from its leaf.
    /// Nothing reads the `Scanner`, so any plan built over a
    /// [`LanceScanSource`](super::source::LanceScanSource) can be planned this way, not just one
    /// the scanner's builder produced.
    pub async fn collect(plan: &LogicalPlan) -> Result<Self> {
        let mut leaf = None;
        let mut takes: Vec<TakeOperation> = Vec::new();
        plan.apply(|node| {
            takes.extend(take_operations(node));
            if leaf.is_none() {
                leaf = with_lance_source(node, |source| source.dataset().clone());
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        let Some(dataset) = leaf else {
            return Err(crate::Error::internal(
                "a Lance scan plan reached stage 2 without a Lance scan leaf".to_string(),
            ));
        };

        Ok(Self {
            scalar_indices: Arc::new(dataset.scalar_index_info().await?),
            take_rows: resolve_takes(&dataset, takes).await?,
        })
    }

    pub fn scalar_indices(&self) -> &ScalarIndexInfo {
        &self.scalar_indices
    }

    /// The rows a take-shaped predicate selects, if resolving it needed I/O and stage 2 did it.
    pub fn take_rows(&self, take: &TakeOperation) -> Option<&Arc<RowAddrTreeMap>> {
        self.take_rows.get(&take_key(take))
    }
}

/// The take-shaped predicates a plan node carries, if any.
///
/// Both positions are checked because stage 2 runs before `PushDownFilter`: a predicate written by
/// the builder is a `Filter`, while one the DataFrame API pushed is already on the scan.
fn take_operations(node: &LogicalPlan) -> Vec<TakeOperation> {
    let predicates: Vec<_> = match node {
        LogicalPlan::Filter(filter) => vec![filter.predicate.clone()],
        LogicalPlan::TableScan(scan) => scan.filters.clone(),
        _ => vec![],
    };
    predicates
        .iter()
        .filter_map(|predicate| Some(TakeOperation::try_from_expr(predicate)?.0))
        .collect()
}

/// Identifies a take by what it selects, so the same take found at two points in planning agrees.
fn take_key(take: &TakeOperation) -> String {
    format!("{take:?}")
}

/// Turn `_rowaddr` and `_rowoffset` takes into the row ids a read can be restricted to.
///
/// `TakeOperation::RowIds` is absent because it needs no translation — the rule handles it inline.
async fn resolve_takes(
    dataset: &Arc<Dataset>,
    takes: Vec<TakeOperation>,
) -> Result<HashMap<String, Arc<RowAddrTreeMap>>> {
    let mut resolved = HashMap::new();
    for take in takes {
        let key = take_key(&take);
        if resolved.contains_key(&key) {
            continue;
        }
        let addrs = match &take {
            TakeOperation::RowIds(_) => continue,
            TakeOperation::RowAddrs(addrs) => addrs.clone(),
            TakeOperation::RowOffsets(offsets) => {
                let mut addrs =
                    row_offsets_to_row_addresses(&dataset.get_fragments(), offsets).await?;
                // An offset can only name a live row, so a tombstone means the offset ran past
                // the end of the dataset. Mirrors `Scanner::take_source`.
                addrs.retain(|addr| *addr != RowAddress::TOMBSTONE_ROW);
                addrs
            }
        };
        let row_ids = live_row_addrs_to_row_ids(dataset, addrs.into_iter().map(Some)).await?;
        resolved.insert(
            key,
            Arc::new(RowAddrTreeMap::from_iter(row_ids.into_iter().flatten())),
        );
    }
    Ok(resolved)
}
