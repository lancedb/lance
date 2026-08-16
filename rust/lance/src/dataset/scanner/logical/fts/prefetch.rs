// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage 2 prefetch for full-text search: what one FTS index contributes to the
//! planning context.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::LogicalPlan;
use lance_index::scalar::inverted::DocumentGranularity;
use lance_index::scalar::inverted::query::FtsQuery;
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;

use super::super::context::{OpaqueSegments, OverlayStaleness};
use super::*;
use crate::dataset::Dataset;
use crate::index::scalar::inverted::load_segment_details;
use crate::{Error, Result};

// ---------------------------------------------------------------------------------------------
// Stage 2: prefetch
// ---------------------------------------------------------------------------------------------

/// Everything known about one FTS index, captured once per planning invocation.
#[derive(Debug, Clone)]
pub struct FtsIndexInfo {
    /// Every committed segment of the index; a search fans out over all of them.
    pub segments: Vec<IndexMetadata>,
    /// Whether the index stores token positions. Only phrase queries need it, and finding out
    /// requires reading the index details, so it is resolved here rather than at lowering time.
    pub with_position: bool,
    /// Row-level coverage: which of this index's entries a data overlay has invalidated.
    pub staleness: OverlayStaleness,
}

impl FtsIndexInfo {
    pub(super) fn covered_fragments(&self) -> Option<RoaringBitmap> {
        let mut covered = RoaringBitmap::new();
        for segment in &self.segments {
            covered |= segment.fragment_bitmap.as_ref()?;
        }
        Some(covered)
    }
}

/// Which FTS index a plan needs metadata for. One per (column, granularity) pair, since a column
/// may carry both a row-level and a list-element index.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FtsIndexRequest {
    pub column: String,
    pub granularity: DocumentGranularity,
    /// Whether any leaf on this column is a phrase query, which is what makes `with_position`
    /// worth the extra read.
    pub needs_position: bool,
}

/// Walk the unoptimized plan for the FTS indices it will need.
pub fn collect_requests(plan: &LogicalPlan) -> Vec<FtsIndexRequest> {
    let mut requests: Vec<FtsIndexRequest> = Vec::new();
    let _ = plan.apply(|node| {
        if let LogicalPlan::Extension(extension) = node
            && let Some(leaf) = extension.node.as_any().downcast_ref::<FtsLeafNode>()
        {
            let needs_position = matches!(leaf.query, FtsQuery::Phrase(_));
            match requests.iter_mut().find(|r| {
                r.column == leaf.field.canonical_path && r.granularity == leaf.granularity
            }) {
                Some(existing) => existing.needs_position |= needs_position,
                None => requests.push(FtsIndexRequest {
                    column: leaf.field.canonical_path.clone(),
                    granularity: leaf.granularity,
                    needs_position,
                }),
            }
        }
        Ok(TreeNodeRecursion::Continue)
    });
    requests
}

/// Load the index metadata for each request. The one stage that is allowed to do I/O.
///
/// Input validation that depends on this metadata happens here too, not in the rules: an error
/// raised inside an `OptimizerRule` comes back wrapped as a DataFusion `External` error, which
/// loses the [`Error`] variant callers match on.
pub async fn prefetch(
    dataset: &Arc<Dataset>,
    requests: &[FtsIndexRequest],
    target_fragments: &RoaringBitmap,
    fragments: &[Fragment],
) -> Result<HashMap<(String, DocumentGranularity), FtsIndexInfo>> {
    let mut loaded = HashMap::with_capacity(requests.len());
    for request in requests {
        let Some(segments) = crate::index::scalar::inverted::load_segments(
            dataset,
            &request.column,
            request.granularity,
        )
        .await?
        else {
            continue;
        };
        let with_position = if request.needs_position {
            load_segment_details(dataset, &request.column, &segments)
                .await?
                .with_position
        } else {
            false
        };
        // `Opaque`, not `Covering`: a legacy segment cannot say which rows it indexed, and a BM25
        // score depends on the whole indexed document set, so a relevant overlay invalidates it
        // wholesale rather than row by row.
        let staleness = super::super::context::overlay_staleness(
            dataset,
            &segments,
            fragments,
            OpaqueSegments::Opaque,
        )
        .await?;
        let info = FtsIndexInfo {
            segments,
            with_position,
            staleness,
        };
        // A phrase query only needs positions from an index it will actually read. When the index
        // covers none of the target fragments the query is answered by a flat scan, which computes
        // positions from the text itself.
        let index_is_reachable = info
            .covered_fragments()
            .is_none_or(|covered| !(&covered & target_fragments).is_empty());
        if request.needs_position && !with_position && index_is_reachable {
            return Err(Error::invalid_input(
                "position is not found but required for phrase queries, try recreating the index with position"
                    .to_string(),
            ));
        }
        loaded.insert((request.column.clone(), request.granularity), info);
    }
    Ok(loaded)
}
