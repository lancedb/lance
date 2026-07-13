// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod delete;
mod write;

use std::collections::BTreeMap;
use std::sync::Arc;

use datafusion::physical_plan::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder};
use futures::StreamExt;
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, Result};
use lance_table::format::{DeletionFile, DeletionFileType, Fragment};
use lance_table::io::deletion::{relative_deletion_file_path, write_deletion_file_with_metadata};
use roaring::{RoaringBitmap, RoaringTreemap};
use uuid::Uuid;

pub use delete::DeleteOnlyMergeInsertExec;
pub use write::FullSchemaMergeInsertExec;
pub(in crate::dataset::write) use write::preflight_v2_3_transaction;
pub(super) use write::{preflight_v2_3_merge, spool_for_replay};

use super::MergeStats;
use crate::Dataset;

#[derive(Debug, Clone)]
pub(super) struct PlannedDeletionFile {
    fragment_id: u64,
    metadata: DeletionFile,
    vector: DeletionVector,
}

#[derive(Debug, Default)]
pub(in crate::dataset::write) struct DeletionPlan {
    pub(in crate::dataset::write) updated_fragments: Vec<Fragment>,
    pub(in crate::dataset::write) removed_fragment_ids: Vec<u64>,
    pub(in crate::dataset::write) current_vectors: BTreeMap<u32, RoaringBitmap>,
    pub(in crate::dataset::write) successor_vectors: BTreeMap<u32, RoaringBitmap>,
    files: Vec<PlannedDeletionFile>,
}

pub(in crate::dataset::write) async fn plan_deletions(
    dataset: &Dataset,
    removed_row_addrs: &RoaringTreemap,
) -> Result<DeletionPlan> {
    let additions = removed_row_addrs
        .bitmaps()
        .map(|(fragment_id, bitmap)| (fragment_id, bitmap.clone()))
        .collect::<BTreeMap<u32, RoaringBitmap>>();
    let mut plan = DeletionPlan::default();
    let fragment_ids = additions.keys().copied().collect::<Vec<_>>();
    let fragments = dataset.get_frags_from_ordered_ids(&fragment_ids);
    if fragments.len() != fragment_ids.len() {
        return Err(Error::invalid_input(
            "merge deletion references a fragment absent from the source snapshot",
        ));
    }
    for ((fragment_id, addition), fragment) in additions.iter().zip(fragments) {
        let fragment = fragment.ok_or_else(|| {
            Error::invalid_input(format!(
                "merge deletion references fragment {fragment_id} absent from the source snapshot"
            ))
        })?;
        let current = fragment
            .get_deletion_vector()
            .await?
            .map(|vector| vector.as_ref().clone())
            .unwrap_or_default();
        if !current.is_empty() {
            plan.current_vectors
                .insert(*fragment_id, RoaringBitmap::from(&current));
        }
        let mut successor = current;
        successor.extend(addition.iter());
        let physical_rows = fragment.metadata.physical_rows.ok_or_else(|| {
            Error::internal(format!(
                "storage-version-2.3 fragment {} is missing physical_rows",
                fragment.id()
            ))
        })?;
        let physical_rows = u32::try_from(physical_rows)
            .map_err(|_| Error::invalid_input("fragment rows exceed deletion-vector capacity"))?;
        if successor.iter().any(|offset| offset >= physical_rows) {
            return Err(Error::invalid_input(format!(
                "merge deletion vector for fragment {} contains offsets outside {} physical rows",
                fragment.id(),
                physical_rows
            )));
        }
        if successor.len() == physical_rows as usize {
            plan.removed_fragment_ids.push(fragment.id() as u64);
            continue;
        }

        let file_type = match &successor {
            DeletionVector::NoDeletions => continue,
            DeletionVector::Set(_) => DeletionFileType::Array,
            DeletionVector::Bitmap(_) => DeletionFileType::Bitmap,
        };
        let metadata = DeletionFile {
            read_version: dataset.manifest.version,
            id: Uuid::new_v4().as_u128() as u64,
            file_type,
            num_deleted_rows: Some(successor.len()),
            base_id: None,
        };
        let mut updated = fragment.metadata.clone();
        updated.deletion_file = Some(metadata.clone());
        plan.updated_fragments.push(updated);
        plan.successor_vectors
            .insert(*fragment_id, RoaringBitmap::from(&successor));
        plan.files.push(PlannedDeletionFile {
            fragment_id: fragment.id() as u64,
            metadata,
            vector: successor,
        });
    }
    plan.updated_fragments.sort_by_key(|fragment| fragment.id);
    plan.removed_fragment_ids.sort_unstable();
    Ok(plan)
}

pub(in crate::dataset::write) async fn plan_full_fragment_deletions(
    dataset: &Dataset,
    removed_fragment_ids: &[u64],
) -> Result<DeletionPlan> {
    let fragment_ids = removed_fragment_ids
        .iter()
        .map(|fragment_id| {
            u32::try_from(*fragment_id).map_err(|_| {
                Error::invalid_input(format!(
                    "fragment id {fragment_id} exceeds deletion-vector capacity"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let fragments = dataset.get_frags_from_ordered_ids(&fragment_ids);
    if fragments.len() != fragment_ids.len() {
        return Err(Error::invalid_input(
            "full-fragment deletion references a fragment absent from the source snapshot",
        ));
    }

    let mut plan = DeletionPlan::default();
    for (fragment_id, fragment) in fragment_ids.into_iter().zip(fragments) {
        let fragment = fragment.ok_or_else(|| {
            Error::invalid_input(format!(
                "full-fragment deletion references fragment {fragment_id} absent from the source snapshot"
            ))
        })?;
        if let Some(current) = fragment.get_deletion_vector().await? {
            let current = RoaringBitmap::from(current.as_ref());
            if !current.is_empty() {
                plan.current_vectors.insert(fragment_id, current);
            }
        }
        plan.removed_fragment_ids.push(fragment.id() as u64);
    }
    plan.removed_fragment_ids.sort_unstable();
    plan.removed_fragment_ids.dedup();
    Ok(plan)
}

pub(in crate::dataset::write) async fn cleanup_planned_deletions(
    dataset: &Dataset,
    plan: &DeletionPlan,
) {
    for file in &plan.files {
        let path = dataset.base.clone().join(relative_deletion_file_path(
            file.fragment_id,
            &file.metadata,
        ));
        let _ = dataset.object_store.delete(&path).await;
    }
}

pub(in crate::dataset::write) async fn write_planned_deletions(
    dataset: &Dataset,
    plan: &DeletionPlan,
) -> Result<()> {
    for file in &plan.files {
        if let Err(error) = write_deletion_file_with_metadata(
            &dataset.base,
            file.fragment_id,
            &file.metadata,
            &file.vector,
            dataset.object_store.as_ref(),
        )
        .await
        {
            cleanup_planned_deletions(dataset, plan).await;
            return Err(error);
        }
    }
    Ok(())
}

pub(super) struct MergeInsertMetrics {
    pub num_inserted_rows: Count,
    pub num_updated_rows: Count,
    pub num_deleted_rows: Count,
    pub bytes_written: Count,
    pub num_files_written: Count,
    pub num_skipped_duplicates: Count,
}

impl From<&MergeInsertMetrics> for MergeStats {
    fn from(value: &MergeInsertMetrics) -> Self {
        Self {
            num_deleted_rows: value.num_deleted_rows.value() as u64,
            num_inserted_rows: value.num_inserted_rows.value() as u64,
            num_updated_rows: value.num_updated_rows.value() as u64,
            bytes_written: value.bytes_written.value() as u64,
            num_files_written: value.num_files_written.value() as u64,
            num_skipped_duplicates: value.num_skipped_duplicates.value() as u64,
            num_attempts: 1,
        }
    }
}

impl MergeInsertMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        let num_inserted_rows = MetricBuilder::new(metrics).counter("num_inserted_rows", partition);
        let num_updated_rows = MetricBuilder::new(metrics).counter("num_updated_rows", partition);
        let num_deleted_rows = MetricBuilder::new(metrics).counter("num_deleted_rows", partition);
        let bytes_written = MetricBuilder::new(metrics).counter("bytes_written", partition);
        let num_files_written = MetricBuilder::new(metrics).counter("num_files_written", partition);
        let num_skipped_duplicates =
            MetricBuilder::new(metrics).counter("num_skipped_duplicates", partition);
        Self {
            num_inserted_rows,
            num_updated_rows,
            num_deleted_rows,
            bytes_written,
            num_files_written,
            num_skipped_duplicates,
        }
    }
}

pub(super) async fn apply_deletions(
    dataset: &Dataset,
    removed_row_addrs: &RoaringTreemap,
) -> crate::Result<(Vec<Fragment>, Vec<u64>)> {
    let bitmaps = Arc::new(removed_row_addrs.bitmaps().collect::<BTreeMap<_, _>>());

    enum FragmentChange {
        Unchanged,
        Modified(Box<Fragment>),
        Removed(u64),
    }

    let mut updated_fragments = Vec::new();
    let mut removed_fragments = Vec::new();

    let mut stream = futures::stream::iter(dataset.get_fragments())
        .map(move |fragment| {
            let bitmaps_ref = bitmaps.clone();
            async move {
                let fragment_id = fragment.id();
                if let Some(bitmap) = bitmaps_ref.get(&(fragment_id as u32)) {
                    match fragment.extend_deletions(*bitmap).await {
                        Ok(Some(new_fragment)) => {
                            Ok(FragmentChange::Modified(Box::new(new_fragment.metadata)))
                        }
                        Ok(None) => Ok(FragmentChange::Removed(fragment_id as u64)),
                        Err(e) => Err(e),
                    }
                } else {
                    Ok(FragmentChange::Unchanged)
                }
            }
        })
        .buffer_unordered(dataset.object_store.io_parallelism());

    while let Some(res) = stream.next().await.transpose()? {
        match res {
            FragmentChange::Unchanged => {}
            FragmentChange::Modified(fragment) => updated_fragments.push(*fragment),
            FragmentChange::Removed(fragment_id) => removed_fragments.push(fragment_id),
        }
    }

    Ok((updated_fragments, removed_fragments))
}
