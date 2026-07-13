// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use lance::Dataset;
use lance::index::DatasetIndexExt;
use lance_core::{Error, Result};
use lance_index::is_system_index;
use lance_index::optimize::OptimizeOptions;
use lance_io::utils::tracking_store::IoOperation;
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;

pub fn aggregate_control_path_category(
    operation: IoOperation,
    path: &str,
    num_bytes: u64,
) -> Option<&'static str> {
    (operation == IoOperation::Delete && path.is_empty() && num_bytes == 0).then_some("metadata")
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LegacyIndexGroup {
    fields: Vec<i32>,
    segments: usize,
    current_fragment_coverage: RoaringBitmap,
}

fn current_fragment_bitmap(fragments: &[Fragment]) -> Result<RoaringBitmap> {
    fragments
        .iter()
        .map(|fragment| {
            u32::try_from(fragment.id).map_err(|_| {
                Error::internal(format!(
                    "benchmark index coverage cannot represent fragment {}",
                    fragment.id
                ))
            })
        })
        .collect()
}

fn legacy_user_index_catalog(
    indices: &[IndexMetadata],
    current_fragments: &RoaringBitmap,
) -> Result<BTreeMap<String, LegacyIndexGroup>> {
    let mut catalog = BTreeMap::new();
    for index in indices.iter().filter(|index| !is_system_index(index)) {
        let coverage = index
            .effective_fragment_bitmap(current_fragments)
            .ok_or_else(|| {
                Error::internal(format!(
                    "benchmark legacy index segment {} for '{}' has no exact fragment coverage",
                    index.uuid, index.name
                ))
            })?;
        let group = catalog
            .entry(index.name.clone())
            .or_insert_with(|| LegacyIndexGroup {
                fields: index.fields.clone(),
                segments: 0,
                current_fragment_coverage: RoaringBitmap::new(),
            });
        if group.fields != index.fields {
            return Err(Error::internal(format!(
                "benchmark legacy index '{}' has inconsistent fields {:?} and {:?}",
                index.name, group.fields, index.fields
            )));
        }
        group.segments += 1;
        group.current_fragment_coverage |= coverage;
    }
    Ok(catalog)
}

/// Consolidate all user-index segments before replaying one contiguous legacy
/// v2.2 compaction group.
///
/// The physical maintenance plan is validated by the caller before this
/// index-only commit. This function rejects any data-fragment or current index
/// coverage change so the caller can safely bind that plan to the new manifest
/// version without replanning its physical topology.
pub async fn consolidate_legacy_index_segments(
    dataset: &mut Dataset,
    indices_before: &[IndexMetadata],
) -> Result<Option<Duration>> {
    let mut fragments_before = dataset.manifest().fragments.as_ref().clone();
    fragments_before.sort_by(policy_fragment_order);
    let current_fragments = current_fragment_bitmap(&fragments_before)?;
    let catalog_before = legacy_user_index_catalog(indices_before, &current_fragments)?;
    for (name, group) in &catalog_before {
        if group.current_fragment_coverage != current_fragments {
            return Err(Error::internal(format!(
                "benchmark index '{name}' covers {} of {} current fragments before maintenance",
                group.current_fragment_coverage.len(),
                current_fragments.len()
            )));
        }
    }
    let index_names = catalog_before
        .iter()
        .filter(|(_, group)| group.segments > 1)
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    if index_names.is_empty() {
        return Ok(None);
    }

    let source_version = dataset.version().version;
    let started = Instant::now();
    dataset
        .optimize_indices(&OptimizeOptions::merge(usize::MAX).index_names(index_names))
        .await?;
    let elapsed = started.elapsed();
    let expected_version = source_version
        .checked_add(1)
        .ok_or_else(|| Error::internal("benchmark dataset version overflow"))?;
    if dataset.version().version != expected_version {
        return Err(Error::internal(format!(
            "index consolidation expected dataset version {expected_version}, found {}",
            dataset.version().version
        )));
    }

    let mut fragments_after = dataset.manifest().fragments.as_ref().clone();
    fragments_after.sort_by(policy_fragment_order);
    if fragments_after != fragments_before {
        return Err(Error::internal(
            "index consolidation changed the frozen physical maintenance source",
        ));
    }
    let indices_after = dataset.load_indices().await?;
    let catalog_after = legacy_user_index_catalog(indices_after.as_ref(), &current_fragments)?;
    if catalog_after.keys().ne(catalog_before.keys()) {
        return Err(Error::internal(format!(
            "index consolidation changed user index names: before={:?}, after={:?}",
            catalog_before.keys().collect::<Vec<_>>(),
            catalog_after.keys().collect::<Vec<_>>()
        )));
    }
    for (name, before) in &catalog_before {
        let after = &catalog_after[name];
        if after.fields != before.fields
            || after.current_fragment_coverage != before.current_fragment_coverage
            || after.segments != 1
        {
            return Err(Error::internal(format!(
                "index consolidation did not preserve one complete segment for '{name}': before={before:?}, after={after:?}"
            )));
        }
    }
    Ok(Some(elapsed))
}

pub fn policy_fragment_order(left: &Fragment, right: &Fragment) -> Ordering {
    let left_file = left.files.first();
    let right_file = right.files.first();
    (
        left_file.and_then(|file| file.base_id),
        left.id,
        left_file.map(|file| file.path.as_str()).unwrap_or_default(),
    )
        .cmp(&(
            right_file.and_then(|file| file.base_id),
            right.id,
            right_file
                .map(|file| file.path.as_str())
                .unwrap_or_default(),
        ))
}

pub fn fragments_for_native_compaction(
    sources: &[Fragment],
    requires_physical_address_order: bool,
) -> Vec<Fragment> {
    let mut fragments = sources.to_vec();
    if requires_physical_address_order {
        fragments.sort_by_key(|fragment| fragment.id);
    }
    fragments
}
