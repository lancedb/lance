// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Custom scalar index maintenance through the ordinary optimize entry point.
//!
//! A session can associate an optimizer with a scalar index's protobuf details
//! type. The optimizer prepares files and returns their actual coverage; Lance
//! validates and publishes the changes in the optimize transaction. Missing
//! coverage is allowed. Neither an unchanged result nor an error triggers the
//! default column-scan fallback.
//!
//! This first interface runs on the coordinator through
//! [`DatasetIndexExt::optimize_indices`](super::DatasetIndexExt::optimize_indices).
//! Distributed optimize planning explicitly rejects these handlers rather than
//! serializing a callback or silently running the builtin algorithm.

use std::collections::HashSet;

use async_trait::async_trait;
use lance_core::deepsize::DeepSizeOf;
use lance_index::optimize::OptimizeOptions;
use lance_table::format::IndexMetadata;
use uuid::Uuid;

use super::optimize::IndexSegmentResult;
use super::{
    IndexSegment, build_index_metadata_from_segments, validate_segment_index_details,
    validate_segment_metadata, validate_segment_params_compatible,
};
use crate::{Dataset, Error, Result};

/// The immutable snapshot and complete logical index group being optimized.
///
/// Read records through their current coverage and check overlay staleness before
/// using them. Copied records retain their source version; writing a new file
/// does not make old statistics current. Do not commit from inside the callback.
#[derive(Clone, Copy)]
pub struct ScalarIndexOptimizeContext<'a> {
    /// Snapshot anchoring both reads and the eventual optimize transaction.
    pub dataset: &'a Dataset,
    /// All committed segments of one logical index, in manifest order.
    pub segments: &'a [&'a IndexMetadata],
}

/// Files prepared by a custom optimizer, not yet published in the manifest.
#[derive(Debug)]
pub enum ScalarIndexOptimization {
    /// No change. This is final, not a request for default optimization.
    Unchanged,
    /// Publish one or more new segments and remove the named input segments.
    ///
    /// New segments must use fresh UUIDs, the same field/type declaration, and
    /// explicit coverage of current fragments. Replacing an input may reduce
    /// coverage; fragments not covered by any remaining segment are unindexed.
    /// Empty new coverage can represent a definition-only index, but
    /// `new_segments` itself must not be empty.
    Updated {
        /// New files with their true coverage and physical build provenance.
        new_segments: Vec<IndexSegment>,
        /// UUIDs from the supplied group; unlisted segments remain unchanged.
        replaced_segments: Vec<Uuid>,
    },
}

/// Prepare a logical scalar index's maintenance without committing it.
///
/// Register with [`crate::session::Session::register_scalar_index_optimizer`].
/// Lance calls this before preparing a training scan. A successful round need
/// not cover every fragment; consumers must retain unindexed data. The callback
/// must honor explicit append/merge requests or return an error. Retraining and
/// distributed row bounds are not supported by this first interface.
///
/// ```
/// # use async_trait::async_trait;
/// # use lance::{Result, index::optimizer::{ScalarIndexOptimizer,
/// #     ScalarIndexOptimizeContext, ScalarIndexOptimization}};
/// # use lance_core::deepsize::DeepSizeOf;
/// # use lance_index::optimize::OptimizeOptions;
/// #[derive(Debug, DeepSizeOf)]
/// struct NoWork;
/// #[async_trait]
/// impl ScalarIndexOptimizer for NoWork {
///     async fn optimize(&self, context: ScalarIndexOptimizeContext<'_>,
///                       options: &OptimizeOptions) -> Result<ScalarIndexOptimization> {
///         // Inspect metadata or existing summaries without scanning column values.
///         Ok(ScalarIndexOptimization::Unchanged)
///     }
/// }
/// ```
#[async_trait]
pub trait ScalarIndexOptimizer: std::fmt::Debug + DeepSizeOf + Send + Sync {
    /// Write uncommitted output files or leave this group unchanged.
    async fn optimize(
        &self,
        context: ScalarIndexOptimizeContext<'_>,
        options: &OptimizeOptions,
    ) -> Result<ScalarIndexOptimization>;
}

pub(crate) async fn prepare_optimization(
    context: ScalarIndexOptimizeContext<'_>,
    options: &OptimizeOptions,
    result: ScalarIndexOptimization,
    committed: &[IndexMetadata],
    plan_id: Uuid,
) -> Result<Vec<IndexSegmentResult>> {
    let ScalarIndexOptimization::Updated {
        new_segments,
        replaced_segments,
    } = result
    else {
        return Ok(Vec::new());
    };
    let dataset = context.dataset;
    let reference = context.segments[0];
    let invalid = |reason: &str| {
        Error::invalid_input(format!(
            "custom optimize of index '{}': {reason}",
            reference.name
        ))
    };
    if new_segments.is_empty() {
        return Err(invalid("an update must contain at least one new segment"));
    }
    let replaced: HashSet<Uuid> = replaced_segments.iter().copied().collect();
    if replaced.len() != replaced_segments.len() {
        return Err(invalid("replacement UUIDs must be unique"));
    }
    if replaced
        .iter()
        .any(|uuid| !context.segments.iter().any(|old| old.uuid == *uuid))
    {
        return Err(invalid("replacement UUID does not belong to this index"));
    }
    let live = dataset.fragment_bitmap();
    if let Some(limit) = options.num_indices_to_merge {
        let eligible: HashSet<_> = context
            .segments
            .iter()
            .rev()
            .take(limit)
            .map(|s| s.uuid)
            .collect();
        if context.segments.iter().any(|old| {
            replaced.contains(&old.uuid)
                && !eligible.contains(&old.uuid)
                && old
                    .fragment_bitmap
                    .as_ref()
                    .is_none_or(|bitmap| !bitmap.is_disjoint(live))
        }) {
            return Err(invalid(
                "replacement exceeds num_indices_to_merge (append only replaces empty inputs)",
            ));
        }
    }
    for segment in &new_segments {
        if committed.iter().any(|old| old.uuid == segment.uuid()) {
            return Err(invalid("new segments must use fresh UUIDs"));
        }
        if segment.fields() != reference.fields
            || segment.covering_fields() != reference.covering_fields
        {
            return Err(invalid(
                "new segments must preserve the index field declaration",
            ));
        }
        if reference
            .index_details
            .as_ref()
            .is_none_or(|details| details.type_url != segment.index_details().type_url)
        {
            return Err(invalid("new segments must preserve the index details type"));
        }
        if !segment.fragment_bitmap().is_subset(live) {
            return Err(invalid("new coverage must contain only current fragments"));
        }
    }
    let field_id = reference
        .keyed_field()
        .ok_or_else(|| invalid("expected one keyed field"))?;
    let incoming =
        build_index_metadata_from_segments(dataset, &reference.name, field_id, new_segments)
            .await?;
    validate_segment_index_details(&reference.name, &incoming)?;
    for segment in &incoming {
        if super::unsupported_index_version(segment).is_some() {
            return Err(invalid(
                "output index version is not supported by the registered reader",
            ));
        }
    }
    let mut retained = Vec::new();
    for old in context.segments {
        let bitmap = old
            .fragment_bitmap
            .as_ref()
            .ok_or_else(|| invalid("input coverage is missing"))?
            & live.as_ref();
        if !replaced.contains(&old.uuid) {
            let mut metadata = (**old).clone();
            metadata.fragment_bitmap = Some(bitmap);
            retained.push(metadata);
        }
    }
    let coexisting: Vec<_> = incoming.iter().chain(&retained).cloned().collect();
    validate_segment_metadata(&reference.name, &coexisting)?;
    validate_segment_params_compatible(&retained, &incoming)?;
    Ok(incoming
        .into_iter()
        .enumerate()
        .map(|(i, segment)| IndexSegmentResult {
            plan_id,
            read_version: dataset.manifest.version,
            // These are final outputs passed straight to commit, not inputs
            // dispatched to a distributed merge by group number.
            group: 0,
            segment,
            replaces: if i == 0 {
                replaced_segments.clone()
            } else {
                Vec::new()
            },
        })
        .collect())
}
