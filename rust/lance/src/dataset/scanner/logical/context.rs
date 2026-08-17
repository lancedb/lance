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

use std::sync::Arc;

use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::LogicalPlan;

use super::scan_index::with_lance_source;
use crate::Result;
use crate::index::{DatasetIndexInternalExt, ScalarIndexInfo};

/// Index and fragment state, captured once per planning invocation.
#[derive(Debug)]
pub struct ScanPlanningContext {
    /// What the filter parser needs to turn a predicate into a scalar index query. Already the
    /// canonical instance of this whole pattern — see the module docs.
    scalar_indices: Arc<ScalarIndexInfo>,
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
        plan.apply(|node| {
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
        })
    }

    pub fn scalar_indices(&self) -> &ScalarIndexInfo {
        &self.scalar_indices
    }
}
