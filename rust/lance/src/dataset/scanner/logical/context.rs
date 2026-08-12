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
//! points collapse into the synchronous derivations below.
//!
//! [`ScalarIndexInfo`](crate::index::ScalarIndexInfo) is the existing instance of this pattern —
//! its doc comment describes the same contract for scalar indices. This generalizes it to vector
//! indices rather than inventing a second mechanism.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::LogicalPlan;
use lance_index::metrics::NoOpMetricsCollector;
use lance_linalg::distance::DistanceType;
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;

use lance_index::scalar::inverted::DocumentGranularity;

use super::fts::{self, FtsIndexInfo};
use super::nodes::{TakeSettings, VectorSearchNode};
use crate::Result;
use crate::dataset::{Dataset, Scanner};
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};

/// Everything known about the vector index covering one column.
#[derive(Debug, Clone)]
pub struct VectorIndexInfo {
    /// The index's delta segments; a search fans out over all of them.
    pub segments: Vec<IndexMetadata>,
    /// The metric the index was built with. Resolving this is the one prefetch that may have to
    /// open the index itself: indices written before `details` carried the metric have nowhere
    /// else to record it, and "user asked for a different metric, so fall back to brute force"
    /// is a planning decision that cannot wait until execution.
    pub metric: DistanceType,
}

impl VectorIndexInfo {
    /// Fragments covered by at least one of this index's segments.
    ///
    /// A segment with no fragment bitmap predates the field, so nothing can be assumed about its
    /// coverage; treating it as covering nothing would wrongly route indexed rows to a flat scan,
    /// so callers get `None` and should fall back.
    fn indexed_fragments(&self) -> Option<RoaringBitmap> {
        let mut covered = RoaringBitmap::new();
        for segment in &self.segments {
            covered |= segment.fragment_bitmap.as_ref()?;
        }
        Some(covered)
    }
}

/// Index and fragment state, captured once per planning invocation.
#[derive(Debug)]
pub struct ScanPlanningContext {
    /// From the manifest, so this costs nothing — but holding it here is what lets the
    /// derivations below be honest about doing no I/O.
    fragments: Arc<Vec<Fragment>>,
    vector: HashMap<String, VectorIndexInfo>,
    fts: HashMap<(String, DocumentGranularity), FtsIndexInfo>,
    /// `Scanner::fast_search`: skip rows no index covers. A rule needs it to know whether to
    /// build the flat branch at all, and it is not derivable from the plan.
    fast_search: bool,
    /// Read settings for any take a rule introduces, so a rewrite cannot lose the scanner's
    /// fragment restriction.
    take_settings: TakeSettings,
}

impl ScanPlanningContext {
    /// Walk the unoptimized plan for what it will need, then fetch all of it.
    ///
    /// This is the only `async fn` in stages 2 through 4 that is allowed to do I/O.
    pub async fn collect(scanner: &Scanner, plan: &LogicalPlan) -> Result<Self> {
        let dataset = scanner.dataset.clone();
        let mut searched_columns = Vec::new();
        plan.apply(|node| {
            if let LogicalPlan::Extension(extension) = node
                && let Some(search) = extension.node.as_any().downcast_ref::<VectorSearchNode>()
            {
                searched_columns.push(search.query().column.clone());
            }
            Ok(TreeNodeRecursion::Continue)
        })?;

        let mut vector = HashMap::with_capacity(searched_columns.len());
        if !searched_columns.is_empty() {
            let indices = dataset.load_indices().await?;
            for column in searched_columns {
                if vector.contains_key(&column) {
                    continue;
                }
                if let Some(info) = load_vector_index(&dataset, &indices, &column).await? {
                    vector.insert(column, info);
                }
            }
        }

        let fragments = scanner
            .fragments
            .clone()
            .map(Arc::new)
            .unwrap_or_else(|| dataset.fragments().clone());
        let target_fragments =
            RoaringBitmap::from_iter(fragments.iter().map(|fragment| fragment.id as u32));
        let fts = fts::prefetch(&dataset, &fts::collect_requests(plan), &target_fragments).await?;

        Ok(Self {
            fragments,
            vector,
            fts,
            fast_search: scanner.fast_search,
            take_settings: take_settings(scanner),
        })
    }

    pub fn vector_index(&self, column: &str) -> Option<&VectorIndexInfo> {
        self.vector.get(column)
    }

    pub fn fast_search(&self) -> bool {
        self.fast_search
    }

    pub fn take_settings(&self) -> &TakeSettings {
        &self.take_settings
    }

    /// The subset of `segments` that could contribute a row this scan is allowed to return.
    ///
    /// A segment with no fragment bitmap predates the field, so its coverage is unknown and it has
    /// to be kept.
    pub fn reachable_segments(&self, segments: &[IndexMetadata]) -> Vec<IndexMetadata> {
        let target = RoaringBitmap::from_iter(self.fragments.iter().map(|f| f.id as u32));
        segments
            .iter()
            .filter(|segment| {
                segment
                    .fragment_bitmap
                    .as_ref()
                    .is_none_or(|covered| !covered.is_disjoint(&target))
            })
            .cloned()
            .collect()
    }

    pub fn fts_index(
        &self,
        column: &str,
        granularity: DocumentGranularity,
    ) -> Option<&FtsIndexInfo> {
        self.fts.get(&(column.to_string(), granularity))
    }

    /// Fragments the FTS index on `column` does not cover — the same synchronous bitmap
    /// arithmetic the vector path uses, over the same manifest fragment list.
    pub fn fts_unindexed_fragments(
        &self,
        column: &str,
        granularity: DocumentGranularity,
    ) -> Option<Vec<Fragment>> {
        fts::partition_fragments(self.fts_index(column, granularity)?, &self.fragments, false)
    }

    pub fn fts_indexed_fragments(
        &self,
        column: &str,
        granularity: DocumentGranularity,
    ) -> Option<Vec<Fragment>> {
        fts::partition_fragments(self.fts_index(column, granularity)?, &self.fragments, true)
    }

    /// Fragments the given vector index does not cover — synchronous, because it is just the
    /// index's fragment bitmap subtracted from the manifest's fragment list.
    ///
    /// This is the derivation that decides between a single-branch ANN plan and the two-branch
    /// combined plan, and in the imperative path it is an `await`.
    pub fn unindexed_fragments(&self, column: &str) -> Option<Vec<Fragment>> {
        self.partition_fragments(column, false)
    }

    /// The complement of [`Self::unindexed_fragments`].
    pub fn indexed_fragments(&self, column: &str) -> Option<Vec<Fragment>> {
        self.partition_fragments(column, true)
    }

    fn partition_fragments(&self, column: &str, covered_side: bool) -> Option<Vec<Fragment>> {
        let covered = self.vector_index(column)?.indexed_fragments()?;
        Some(
            self.fragments
                .iter()
                .filter(|fragment| covered.contains(fragment.id as u32) == covered_side)
                .cloned()
                .collect(),
        )
    }
}

async fn load_vector_index(
    dataset: &Arc<Dataset>,
    indices: &[IndexMetadata],
    column: &str,
) -> Result<Option<VectorIndexInfo>> {
    let Ok(field_id) = dataset.schema().field_id(column) else {
        return Ok(None);
    };
    let Some(index) = indices
        .iter()
        .find(|index| index.fields.contains(&field_id))
    else {
        return Ok(None);
    };

    let metric = match crate::index::vector::details::metric_type_from_index_metadata(index) {
        Some(metric) => metric,
        None => dataset
            .open_vector_index(column, &index.uuid, &NoOpMetricsCollector)
            .await?
            .metric_type(),
    };

    Ok(Some(VectorIndexInfo {
        segments: dataset.load_indices_by_name(&index.name).await?.to_vec(),
        metric,
    }))
}

/// Read settings a take must honor, lifted off the `Scanner`. `None` fragments means "all of
/// them", which is a different plan from an explicit list of every fragment.
pub fn take_settings(scanner: &Scanner) -> TakeSettings {
    TakeSettings {
        fragments: scanner.fragments.clone().map(Arc::new),
        batch_size: scanner.batch_size.map(|size| size as u32),
    }
}
