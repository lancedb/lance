// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use futures::{FutureExt, TryStreamExt};
use lance_core::{Error, Result};
use lance_index::{
    INDEX_FILE_NAME, IndexType,
    metrics::NoOpMetricsCollector,
    optimize::OptimizeOptions,
    progress::NoopIndexBuildProgress,
    scalar::{
        CreatedIndex, OldIndexDataFilter, ScalarIndex, inverted::InvertedIndex,
        lance_format::LanceIndexStore,
    },
};
use lance_select::{RowAddrTreeMap, RowSetOps};
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;
use uuid::Uuid;

use super::DatasetIndexInternalExt;
use super::vector::LogicalVectorIndex;
use super::vector::ivf::{optimize_vector_indices, select_segment_for_single_rebalance};
use crate::dataset::Dataset;
use crate::dataset::index::LanceIndexStoreExt;
use crate::dataset::rowids::load_row_id_sequences;
use crate::index::scalar::load_training_data;
use crate::index::vector_index_details_default;

#[derive(Debug, Clone)]
pub struct IndexMergeResults<'a> {
    pub new_uuid: Uuid,
    pub removed_indices: Vec<&'a IndexMetadata>,
    pub new_fragment_bitmap: RoaringBitmap,
    pub new_index_version: i32,
    pub new_index_details: prost_types::Any,
    /// List of files and their sizes for the merged index
    pub files: Vec<lance_table::format::IndexFile>,
}

async fn build_stable_row_id_filter(
    dataset: &Dataset,
    effective_old_frags: &RoaringBitmap,
) -> Result<RowAddrTreeMap> {
    // For stable row IDs we cannot derive fragment ownership from row_id bits.
    // Instead, we:
    // 1) keep only fragments still considered "effective" for the old index, and
    // 2) load their persisted row-id sequences from dataset metadata, then
    // 3) build one exact allow-list used to retain only still-valid old rows.
    let retained_frags = dataset
        .manifest
        .fragments
        .iter()
        .filter(|frag| effective_old_frags.contains(frag.id as u32))
        .cloned()
        .collect::<Vec<_>>();

    if retained_frags.is_empty() {
        return Ok(RowAddrTreeMap::new());
    }

    let row_id_sequences = load_row_id_sequences(dataset, &retained_frags)
        .try_collect::<Vec<_>>()
        .await?;

    let frag_by_id: std::collections::HashMap<u32, _> = dataset
        .get_fragments()
        .into_iter()
        .map(|f| (f.id() as u32, f))
        .collect();

    let mut row_id_maps = Vec::with_capacity(row_id_sequences.len());
    for (frag_id, seq) in &row_id_sequences {
        row_id_maps.push(live_row_ids(frag_by_id.get(frag_id), seq).await?);
    }
    let row_id_map_refs = row_id_maps.iter().collect::<Vec<_>>();

    // Merge all fragment-local row-id sets into one exact membership structure.
    Ok(<RowAddrTreeMap as RowSetOps>::union_all(&row_id_map_refs))
}

/// The fragment's live row ids: its persisted row-id sequence minus the rows
/// its deletion vector marks gone. A persisted sequence covers every row the
/// fragment ever held, so a row whose old copy was deleted (e.g. rewritten by an
/// update under the same stable row id) would otherwise be retained as a stale
/// old-index entry.
async fn live_row_ids(
    fragment: Option<&crate::dataset::fragment::FileFragment>,
    seq: &lance_table::rowids::RowIdSequence,
) -> Result<RowAddrTreeMap> {
    // Propagate a deletion-vector read failure rather than swallowing it: a
    // swallowed error would fall through to the "no deletions" branch below,
    // putting the deleted rows back into the allow-list as stale entries.
    let deletion_vector = match fragment {
        Some(f) if f.metadata().deletion_file.is_some() => f.get_deletion_vector().await?,
        _ => None,
    };
    Ok(match deletion_vector {
        Some(dv) => seq
            .iter()
            .enumerate()
            .filter(|(offset, _)| !dv.contains(*offset as u32))
            .map(|(_, row_id)| row_id)
            .collect(),
        None => RowAddrTreeMap::from(seq),
    })
}

/// Build the [`OldIndexDataFilter`] that must be applied to existing index
/// rows when their owning fragments have been pruned by compaction or
/// deletions.
pub async fn build_old_data_filter(
    dataset: &Dataset,
    effective_old_frags: &RoaringBitmap,
    deleted_old_frags: &RoaringBitmap,
) -> Result<Option<OldIndexDataFilter>> {
    if dataset.manifest.uses_stable_row_ids() {
        let valid_old_row_ids = build_stable_row_id_filter(dataset, effective_old_frags).await?;
        Ok(Some(OldIndexDataFilter::RowIds(valid_old_row_ids)))
    } else {
        Ok(Some(OldIndexDataFilter::Fragments {
            to_keep: effective_old_frags.clone(),
            to_remove: deleted_old_frags.clone(),
        }))
    }
}

/// Split the stored fragment coverage of `segments` into fragments still live in
/// `dataset` (`effective`) and fragments that compaction or deletion has already
/// retired (`deleted`).
pub fn split_segment_coverage<'a>(
    dataset: &Dataset,
    segments: impl IntoIterator<Item = &'a IndexMetadata>,
) -> (RoaringBitmap, RoaringBitmap) {
    let mut effective = RoaringBitmap::new();
    let mut deleted = RoaringBitmap::new();
    for segment in segments {
        if let Some(eff) = segment.effective_fragment_bitmap(&dataset.fragment_bitmap) {
            effective |= eff;
        }
        if let Some(del) = segment.deleted_fragment_bitmap(&dataset.fragment_bitmap) {
            deleted |= del;
        }
    }
    (effective, deleted)
}

/// Build one [`OldIndexDataFilter`] per segment, each derived from that segment's
/// *own* effective (still-live) and retired fragment coverage, plus the union of
/// every segment's still-live coverage.
pub async fn build_per_segment_filters(
    dataset: &Dataset,
    segments: &[&IndexMetadata],
) -> Result<(RoaringBitmap, Vec<Option<OldIndexDataFilter>>)> {
    let mut effective_union = RoaringBitmap::new();
    let mut filters = Vec::with_capacity(segments.len());
    for segment in segments {
        if segment.fragment_bitmap.is_none() {
            return Err(Error::invalid_input(format!(
                "CreateIndex: segment {} is missing fragment coverage",
                segment.uuid
            )));
        }
        let effective = segment
            .effective_fragment_bitmap(&dataset.fragment_bitmap)
            .unwrap_or_default();
        let deleted = segment
            .deleted_fragment_bitmap(&dataset.fragment_bitmap)
            .unwrap_or_default();
        effective_union |= &effective;
        filters.push(build_old_data_filter(dataset, &effective, &deleted).await?);
    }
    Ok((effective_union, filters))
}

async fn load_unindexed_training_data(
    dataset: &Dataset,
    field_path: &str,
    update_criteria: &lance_index::scalar::UpdateCriteria,
    unindexed: &[Fragment],
) -> Result<datafusion::execution::SendableRecordBatchStream> {
    let fragments = if update_criteria.requires_old_data {
        None
    } else {
        Some(unindexed.to_vec())
    };
    load_training_data(
        dataset,
        field_path,
        &update_criteria.data_criteria,
        fragments,
        true,
        None,
    )
    .await
}

/// Build a fresh, canonical (non-sharded) scalar index over `fragment_ids`,
/// reusing `reference_index`'s params and training criteria.
async fn rebuild_scalar_segment(
    dataset: &Dataset,
    reference_index: &Arc<dyn ScalarIndex>,
    field_path: &str,
    column_name: &str,
    uuid: Uuid,
    fragment_ids: Vec<u32>,
) -> Result<CreatedIndex> {
    let params = reference_index.derive_index_params()?;
    let update_criteria = reference_index.update_criteria();
    let training_data = load_training_data(
        dataset,
        field_path,
        &update_criteria.data_criteria,
        None,
        true,
        Some(fragment_ids),
    )
    .await?;
    super::scalar::build_scalar_index(
        dataset,
        column_name,
        uuid,
        &params,
        true,
        None,
        Some(training_data),
        Arc::new(NoopIndexBuildProgress),
    )
    .await
}

/// The index segments to rewrite in this optimize pass.
///
/// Normally the trailing `num_indices_to_merge` segments. Under stable row ids,
/// any *older* segment that still covers a fragment carrying deletions is added
/// too: an update deletes a row's old copy (leaving a deletion vector) and
/// rewrites it under the same row id, so its stale old-value postings survive
/// until that segment is rewritten and filtered. Only the segments that actually
/// cover a deleted-from fragment are pulled in -- clean segments in between are
/// left untouched -- so an edit to old data does not force a full reindex.
///
/// The deletion check is conservative (any current deletion vector on a covered
/// fragment), so a segment built after those deletions may be rewritten as a
/// harmless no-op; it never leaves a stale segment behind (PR #7359).
fn select_segments_to_merge<'a>(
    dataset: &Dataset,
    old_indices: &[&'a IndexMetadata],
    options: &OptimizeOptions,
) -> Vec<&'a IndexMetadata> {
    let num_to_merge = options
        .num_indices_to_merge
        .unwrap_or(1)
        .min(old_indices.len());
    let tail_start = old_indices.len() - num_to_merge;

    // Address-style row ids mask stale postings at search time, and append mode
    // (num_to_merge == 0) defers cleanup to a real merge; both keep the plain tail.
    if num_to_merge == 0 || !dataset.manifest.uses_stable_row_ids() {
        return old_indices[tail_start..].to_vec();
    }

    let deleted_frags: RoaringBitmap = dataset
        .get_fragments()
        .iter()
        .filter(|f| f.metadata().deletion_file.is_some())
        .map(|f| f.id() as u32)
        .collect();
    if deleted_frags.is_empty() {
        return old_indices[tail_start..].to_vec();
    }

    let mut selected = Vec::new();
    for (i, idx) in old_indices.iter().enumerate() {
        let covers_deleted = idx
            .effective_fragment_bitmap(&dataset.fragment_bitmap)
            .is_some_and(|eff| !eff.is_disjoint(&deleted_frags));
        if i >= tail_start || covers_deleted {
            selected.push(*idx);
        }
    }
    selected
}

#[allow(clippy::too_many_arguments)]
async fn merge_scalar_indices<'a>(
    dataset: Arc<Dataset>,
    old_indices: &[&'a IndexMetadata],
    unindexed: &[Fragment],
    options: &OptimizeOptions,
    index_type: IndexType,
    field_path: &str,
    column_name: &str,
    base_unindexed_bitmap: RoaringBitmap,
) -> Result<Option<(Uuid, Vec<&'a IndexMetadata>, RoaringBitmap, CreatedIndex)>> {
    if old_indices.is_empty() {
        return Err(Error::index(
            "merge_scalar_indices: no previous index found".to_string(),
        ));
    }

    let selected_old_indices = select_segments_to_merge(dataset.as_ref(), old_indices, options);

    // No new data + ≤1 old selected = rewriting one segment to itself.
    if unindexed.is_empty() && selected_old_indices.len() <= 1 {
        return Ok(None);
    }

    // For the delta case (`selected` empty) the reference is purely
    // for reading params; fall back to the last old index then.
    let reference_idx = selected_old_indices
        .first()
        .copied()
        .unwrap_or(old_indices[old_indices.len() - 1]);
    let reference_index = dataset
        .open_scalar_index(field_path, &reference_idx.uuid, &NoOpMetricsCollector)
        .await?;
    let update_criteria = reference_index.update_criteria();

    // Effective = bitmap ∩ live fragments; deleted = bitmap \ live fragments.
    let (effective_old_frags, deleted_old_frags) =
        split_segment_coverage(dataset.as_ref(), selected_old_indices.iter().copied());

    let mut frag_bitmap = base_unindexed_bitmap.clone();
    frag_bitmap |= &effective_old_frags;
    let new_uuid = Uuid::new_v4();

    // Scalar Index that expos an N:1 segment-merge primitive reachable without
    // rescanning the dataset
    let has_segment_merge_primitive = matches!(index_type, IndexType::BTree);

    // Merge new data into the existing segment(s) without rebuilding from
    // scratch, when all hold:
    //   - `effective_old_frags`: the selected segments' coverage intersected
    //     with live fragments is non-empty, i.e. there is old data worth keeping.
    //   - `update_criteria` only requires the newly appended data. Indexes that
    //     need old data must rebuild over `frag_bitmap` so the scanned rows
    //     exactly match the segment coverage being committed.
    //   - `has_segment_merge_primitive` (Indices supports N:1 segments merge) OR
    //     `selected_old_indices.len() == 1` (any scalar type can `update` one).
    // Otherwise (e.g. ≥2 selected segments of a type without an N:1 merge
    // primitive) the index is rebuilt from scratch over `frag_bitmap`.
    let can_merge_segments = !effective_old_frags.is_empty()
        && !update_criteria.requires_old_data
        && (has_segment_merge_primitive || selected_old_indices.len() == 1);

    let created_index = if !can_merge_segments {
        rebuild_scalar_segment(
            dataset.as_ref(),
            &reference_index,
            field_path,
            column_name,
            new_uuid,
            frag_bitmap.iter().collect(),
        )
        .await?
    } else {
        let new_data_stream =
            load_unindexed_training_data(dataset.as_ref(), field_path, &update_criteria, unindexed)
                .await?;
        let new_store = LanceIndexStore::from_dataset_for_new(&dataset, &new_uuid)?;

        match index_type {
            IndexType::BTree => {
                let (_, old_data_filters) =
                    build_per_segment_filters(dataset.as_ref(), &selected_old_indices).await?;
                crate::index::scalar::btree::open_and_merge_segments(
                    dataset.as_ref(),
                    field_path,
                    &selected_old_indices,
                    new_data_stream,
                    &new_store,
                    &old_data_filters,
                )
                .await?
            }
            // NOTE: IndexType::Inverted never reaches here -- it is handled by the
            // dedicated arm in merge_indices_with_unindexed_frags before this
            // function is called.
            _ => {
                let old_data_filter = build_old_data_filter(
                    dataset.as_ref(),
                    &effective_old_frags,
                    &deleted_old_frags,
                )
                .await?;
                reference_index
                    .update(new_data_stream, &new_store, old_data_filter)
                    .await?
            }
        }
    };

    Ok(Some((
        new_uuid,
        selected_old_indices.to_vec(),
        frag_bitmap,
        created_index,
    )))
}

async fn metadata_is_vector_index(dataset: &Dataset, index: &IndexMetadata) -> Result<bool> {
    if let Some(files) = &index.files {
        return Ok(files.iter().any(|file| file.path == INDEX_FILE_NAME));
    }

    let index_dir = dataset.indice_files_dir(index)?;
    let index_file = index_dir
        .clone()
        .join(index.uuid.to_string())
        .join(INDEX_FILE_NAME);
    let object_store = dataset.object_store_for_index(index).await?;
    object_store.exists(&index_file).await
}

/// Merge in-inflight unindexed data, with a specific number of previous indices
/// into a new index, to improve the query performance.
///
/// The merge behavior is controlled by [`OptimizeOptions::num_indices_to_merge].
///
/// Returns
/// -------
/// - the UUID of the new index
/// - merged indices,
/// - Bitmap of the fragments that covered in the newly created index.
pub async fn merge_indices<'a>(
    dataset: Arc<Dataset>,
    old_indices: &[&'a IndexMetadata],
    options: &OptimizeOptions,
) -> Result<Option<IndexMergeResults<'a>>> {
    if old_indices.is_empty() {
        return Err(Error::index(
            "Append index: no previous index found".to_string(),
        ));
    };

    let unindexed = dataset.unindexed_fragments(&old_indices[0].name).await?;
    Box::pin(merge_indices_with_unindexed_frags(
        dataset,
        old_indices,
        &unindexed,
        options,
    ))
    .await
}

/// Merge a list of provided unindexed data, with a specific number of previous indices
/// into a new index, to improve the query performance.
pub async fn merge_indices_with_unindexed_frags<'a>(
    dataset: Arc<Dataset>,
    old_indices: &[&'a IndexMetadata],
    unindexed: &[Fragment],
    options: &OptimizeOptions,
) -> Result<Option<IndexMergeResults<'a>>> {
    if old_indices.is_empty() {
        return Err(Error::index(
            "Append index: no previous index found".to_string(),
        ));
    };

    let column = dataset
        .schema()
        .field_by_id(old_indices[0].fields[0])
        .ok_or(Error::index(format!(
            "Append index: column {} does not exist",
            old_indices[0].fields[0]
        )))?;

    let field_path = dataset.schema().field_path(old_indices[0].fields[0])?;
    let first_is_vector_index = metadata_is_vector_index(dataset.as_ref(), old_indices[0]).await?;
    for idx in old_indices.iter().skip(1) {
        let is_vector_index = metadata_is_vector_index(dataset.as_ref(), idx).await?;
        if is_vector_index != first_is_vector_index {
            return Err(Error::index(format!(
                "Append index: invalid mixed index deltas: {:?}",
                old_indices
            )));
        }
    }

    let mut base_unindexed_bitmap = RoaringBitmap::new();
    unindexed.iter().for_each(|frag| {
        base_unindexed_bitmap.insert(frag.id as u32);
    });

    let (new_uuid, removed_indices, new_fragment_bitmap, created_index) = if first_is_vector_index {
        let full_logical_index = dataset
            .open_logical_vector_index(&field_path, &old_indices[0].name)
            .await?;
        let mut opened_indices_by_uuid = full_logical_index
            .iter()
            .map(|(metadata, index)| (metadata.uuid, (metadata.clone(), index.clone())))
            .collect::<std::collections::HashMap<_, _>>();
        let mut selected_metadatas = Vec::with_capacity(old_indices.len());
        let mut selected_indices = Vec::with_capacity(old_indices.len());
        for metadata in old_indices {
            let (selected_metadata, selected_index) = opened_indices_by_uuid.remove(&metadata.uuid).ok_or_else(|| {
                Error::index(format!(
                    "Append index: logical vector index '{}' does not contain requested segment {}",
                    old_indices[0].name, metadata.uuid
                ))
            })?;
            selected_metadatas.push(selected_metadata);
            selected_indices.push(selected_index);
        }
        let logical_index = LogicalVectorIndex::try_new(
            old_indices[0].name.clone(),
            field_path.clone(),
            selected_metadatas
                .into_iter()
                .zip(selected_indices)
                .collect(),
        )?;
        let ivf_view = logical_index.as_ivf()?;

        // Specialized vector no-op: when there is no new data and the caller
        // hasn't asked for retrain or an explicit delta merge, the only useful
        // work is rebalancing. Bail when no segment needs rebalancing so
        // repeated optimize calls don't keep rewriting the same index. This
        // matches the scalar gate in `Dataset::optimize_indices`, which also
        // treats `OptimizeOptions::append()` (num_indices_to_merge=Some(0))
        // as "no explicit merge requested".
        if unindexed.is_empty()
            && !options.retrain
            && options.num_indices_to_merge.is_none_or(|n| n == 0)
            && select_segment_for_single_rebalance(&ivf_view)?.is_none()
        {
            return Ok(None);
        }

        let use_single_segment_rebalance = logical_index.num_segments() > 1
            && options.num_indices_to_merge.is_none_or(|n| n == 0)
            && !options.retrain
            && unindexed.is_empty();

        if use_single_segment_rebalance {
            let Some(selected_segment_id) = select_segment_for_single_rebalance(&ivf_view)? else {
                return Ok(None);
            };
            let removed_segment = old_indices
                .iter()
                .copied()
                .find(|metadata| metadata.uuid == selected_segment_id)
                .ok_or_else(|| {
                    Error::index(format!(
                        "Append index: logical vector index '{}' does not contain selected segment {}",
                        old_indices[0].name, selected_segment_id
                    ))
                })?;
            let (selected_metadata, selected_index) = logical_index
                .iter()
                .find(|(metadata, _)| metadata.uuid == selected_segment_id)
                .map(|(metadata, index)| (metadata.clone(), index.clone()))
                .ok_or_else(|| {
                    Error::index(format!(
                        "Append index: failed to materialize selected segment {} from logical vector index '{}'",
                        selected_segment_id, old_indices[0].name
                    ))
                })?;
            let selected_logical_index = LogicalVectorIndex::try_new(
                old_indices[0].name.clone(),
                field_path.clone(),
                vec![(selected_metadata, selected_index)],
            )?;
            let selected_ivf_view = selected_logical_index.as_ivf()?;
            let (new_uuid, indices_merged, files) = Box::pin(optimize_vector_indices(
                dataset.as_ref().clone(),
                Option::<
                    lance_io::stream::RecordBatchStreamAdapter<
                        futures::stream::Empty<lance_core::Result<arrow_array::RecordBatch>>,
                    >,
                >::None,
                &field_path,
                &selected_ivf_view,
                options,
            ))
            .await?;
            if indices_merged == 0 {
                return Ok(None);
            }

            let new_fragment_bitmap = removed_segment
                .effective_fragment_bitmap(&dataset.fragment_bitmap)
                .or_else(|| removed_segment.fragment_bitmap.clone())
                .unwrap_or_default();

            Ok((
                new_uuid,
                vec![removed_segment],
                new_fragment_bitmap,
                CreatedIndex {
                    index_details: vector_index_details_default(),
                    index_version: lance_index::IndexType::Vector.version() as u32,
                    files,
                },
            ))
        } else {
            let mut frag_bitmap = base_unindexed_bitmap.clone();

            let new_data_stream = if unindexed.is_empty() {
                None
            } else {
                let mut scanner = dataset.scan();
                scanner
                    .with_fragments(unindexed.to_vec())
                    .with_row_id()
                    .project(&[&field_path])?;
                if column.nullable {
                    let column_expr =
                        lance_datafusion::logical_expr::field_path_to_expr(&field_path)?;
                    scanner.filter_expr(column_expr.is_not_null());
                }
                Some(scanner.try_into_stream().await?)
            };

            let (new_uuid, indices_merged, files) = optimize_vector_indices(
                dataset.as_ref().clone(),
                new_data_stream,
                &field_path,
                &ivf_view,
                options,
            )
            .boxed()
            .await?;

            let removed_indices = old_indices[old_indices.len() - indices_merged..].to_vec();
            removed_indices.iter().for_each(|idx| {
                frag_bitmap.extend(idx.fragment_bitmap.as_ref().unwrap().iter());
            });
            for removed in removed_indices.iter() {
                if let Some(effective) = removed.effective_fragment_bitmap(&dataset.fragment_bitmap)
                {
                    frag_bitmap |= &effective;
                }
            }

            // Carry forward existing index details, preferring the first segment
            // that has populated (non-empty) details.
            let index_details = old_indices
                .iter()
                .rev()
                .filter_map(|idx| idx.index_details.as_ref())
                .find(|d| !d.value.is_empty())
                .map(|d| d.as_ref().clone())
                .unwrap_or_else(vector_index_details_default);

            Ok((
                new_uuid,
                removed_indices,
                frag_bitmap,
                CreatedIndex {
                    index_details,
                    // retain_supported_indices guarantees all old_indices have
                    // index_version <= our max supported version, so we can safely
                    // write the current library's version for this index type.
                    index_version: lance_index::IndexType::Vector.version() as u32,
                    files,
                },
            ))
        }
    } else {
        let mut indices = Vec::with_capacity(old_indices.len());
        for idx in old_indices {
            match dataset
                .open_generic_index(&field_path, &idx.uuid, &NoOpMetricsCollector)
                .await
            {
                Ok(index) => indices.push(index),
                Err(e) => {
                    log::warn!(
                        "Cannot open index on column '{}': {}. \
                         Skipping index merge for this column.",
                        field_path,
                        e
                    );
                    return Ok(None);
                }
            }
        }

        if indices
            .windows(2)
            .any(|w| w[0].index_type() != w[1].index_type())
        {
            return Err(Error::index(format!(
                "Append index: invalid index deltas: {:?}",
                old_indices
            )));
        }

        let index_type = indices[0].index_type();
        match index_type {
            IndexType::Inverted => {
                let selected_old_indices =
                    select_segments_to_merge(dataset.as_ref(), old_indices, options);
                if unindexed.is_empty() && selected_old_indices.len() <= 1 {
                    return Ok(None);
                }
                let reference_idx = selected_old_indices
                    .first()
                    .copied()
                    .unwrap_or(old_indices[old_indices.len() - 1]);
                let reference_index = dataset
                    .open_scalar_index(&field_path, &reference_idx.uuid, &NoOpMetricsCollector)
                    .await?;
                let update_criteria = reference_index.update_criteria();
                if update_criteria.requires_old_data {
                    let params = reference_index.derive_index_params()?;
                    let new_data_stream = load_training_data(
                        dataset.as_ref(),
                        &field_path,
                        &update_criteria.data_criteria,
                        None,
                        true,
                        None,
                    )
                    .await?;
                    let new_uuid = Uuid::new_v4();
                    let created_index = super::scalar::build_scalar_index(
                        dataset.as_ref(),
                        column.name.as_str(),
                        new_uuid,
                        &params,
                        true,
                        None,
                        Some(new_data_stream),
                        Arc::new(NoopIndexBuildProgress),
                    )
                    .await?;
                    return Ok(Some(IndexMergeResults {
                        new_uuid,
                        removed_indices: old_indices.to_vec(),
                        new_fragment_bitmap: dataset.fragment_bitmap.as_ref().clone(),
                        new_index_version: created_index.index_version as i32,
                        new_index_details: created_index.index_details,
                        files: created_index.files,
                    }));
                }

                let fragments = Some(unindexed.to_vec());
                let new_data_stream = load_training_data(
                    dataset.as_ref(),
                    &field_path,
                    &update_criteria.data_criteria,
                    fragments,
                    true,
                    None,
                )
                .await?;

                let mut frag_bitmap = base_unindexed_bitmap;
                let mut effective_old_frags = RoaringBitmap::new();
                let mut selected_indices = Vec::with_capacity(selected_old_indices.len());
                for idx in &selected_old_indices {
                    if let Some(effective) = idx.effective_fragment_bitmap(&dataset.fragment_bitmap)
                    {
                        frag_bitmap |= &effective;
                        effective_old_frags |= &effective;
                    }
                    let scalar_index = dataset
                        .open_scalar_index(&field_path, &idx.uuid, &NoOpMetricsCollector)
                        .await?;
                    let inverted_index = scalar_index
                        .as_any()
                        .downcast_ref::<InvertedIndex>()
                        .ok_or_else(|| {
                            Error::index(format!(
                                "Append index: expected inverted index segment {}, got {:?}",
                                idx.uuid,
                                scalar_index.index_type()
                            ))
                        })?;
                    selected_indices.push(Arc::new(inverted_index.clone()));
                }

                let old_data_filter = if selected_indices.is_empty() {
                    None
                } else if dataset.manifest.uses_stable_row_ids() {
                    let valid_old_row_ids =
                        build_stable_row_id_filter(dataset.as_ref(), &effective_old_frags).await?;
                    Some(OldIndexDataFilter::RowIds(valid_old_row_ids))
                } else {
                    Some(OldIndexDataFilter::Fragments {
                        to_keep: effective_old_frags,
                        to_remove: RoaringBitmap::new(),
                    })
                };

                let new_uuid = Uuid::new_v4();
                let new_store = LanceIndexStore::from_dataset_for_new(&dataset, &new_uuid)?;
                let created_index = if selected_indices.is_empty() {
                    let params = reference_index.derive_index_params()?;
                    super::scalar::build_scalar_index(
                        dataset.as_ref(),
                        column.name.as_str(),
                        new_uuid,
                        &params,
                        true,
                        None,
                        Some(new_data_stream),
                        Arc::new(NoopIndexBuildProgress),
                    )
                    .await?
                } else {
                    InvertedIndex::merge_segments(
                        &selected_indices,
                        new_data_stream,
                        &new_store,
                        old_data_filter,
                        options.progress.clone(),
                    )
                    .await?
                };

                Ok((
                    new_uuid,
                    selected_old_indices.to_vec(),
                    frag_bitmap,
                    created_index,
                ))
            }
            it if it.is_scalar() => {
                let Some(result) = merge_scalar_indices(
                    dataset.clone(),
                    old_indices,
                    unindexed,
                    options,
                    it,
                    &field_path,
                    column.name.as_str(),
                    base_unindexed_bitmap,
                )
                .await?
                else {
                    return Ok(None);
                };
                Ok(result)
            }
            _ => Err(Error::index(format!(
                "Append index: invalid index type: {:?}",
                indices[0].index_type()
            ))),
        }
    }?;

    Ok(Some(IndexMergeResults {
        new_uuid,
        removed_indices,
        new_fragment_bitmap,
        new_index_version: created_index.index_version as i32,
        new_index_details: created_index.index_details,
        files: created_index.files,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::index::DatasetIndexExt;
    use crate::index::DatasetIndexInternalExt;
    use arrow::datatypes::{Float32Type, UInt32Type};
    use arrow_array::cast::AsArray;
    use arrow_array::{
        FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array,
    };
    use arrow_schema::{DataType, Field, Schema};
    use futures::TryStreamExt;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datafusion::utils::reader_to_stream;
    use lance_datagen::{Dimension, RowCount, array};
    use lance_index::vector::hnsw::builder::HnswBuildParams;
    use lance_index::vector::sq::builder::SQBuildParams;
    use lance_index::{
        IndexType,
        scalar::{BuiltinIndexType, ScalarIndexParams, SearchResult, TextQuery},
        vector::{ivf::IvfBuildParams, pq::PQBuildParams},
    };
    use lance_linalg::distance::MetricType;
    use lance_testing::datagen::generate_random_array;
    use rstest::rstest;

    use crate::dataset::builder::DatasetBuilder;
    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::{MergeInsertBuilder, WhenMatched, WhenNotMatched, WriteMode, WriteParams};
    use crate::index::vector::VectorIndexParams;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

    #[tokio::test]
    async fn test_append_index() {
        const DIM: usize = 64;
        const IVF_PARTITIONS: usize = 2;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let vectors = generate_random_array(1000 * DIM);

        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            true,
        )]));
        let array = Arc::new(FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap());
        let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()]).unwrap();

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        let ivf_params = IvfBuildParams::new(IVF_PARTITIONS);
        let pq_params = PQBuildParams {
            num_sub_vectors: 2,
            ..Default::default()
        };
        let params = VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf_params, pq_params);

        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        let vectors = generate_random_array(1000 * DIM);
        let array = Arc::new(FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap());
        let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()]).unwrap();

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        dataset.append(batches, None).await.unwrap();

        let index = &dataset.load_indices().await.unwrap()[0];
        assert!(
            !dataset
                .unindexed_fragments(&index.name)
                .await
                .unwrap()
                .is_empty()
        );

        let q = array.value(5);
        let mut scanner = dataset.scan();
        scanner
            .nearest("vector", q.as_primitive::<Float32Type>(), 10)
            .unwrap();
        let results = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(results[0].num_rows(), 10); // Flat search.

        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        let indices = dataset.load_indices().await.unwrap();

        assert!(
            dataset
                .unindexed_fragments(&index.name)
                .await
                .unwrap()
                .is_empty()
        );

        // There should be two indices directories existed.
        let object_store = dataset.object_store.as_ref();
        let index_dirs = object_store.read_dir(dataset.indices_dir()).await.unwrap();
        assert_eq!(index_dirs.len(), 2);

        let mut scanner = dataset.scan();
        scanner
            .nearest("vector", q.as_primitive::<Float32Type>(), 10)
            .unwrap();
        let results = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let vectors = &results[0]["vector"];
        // Second batch of vectors should be in the index.
        let contained = vectors.as_fixed_size_list().iter().any(|v| {
            let vec = v.as_ref().unwrap();
            array.iter().any(|a| a.as_ref().unwrap() == vec)
        });
        assert!(contained);

        // Check that the index has all 2000 rows.
        let mut num_rows = 0;
        for index in indices.iter() {
            let index = dataset
                .open_vector_index("vector", &index.uuid, &NoOpMetricsCollector)
                .await
                .unwrap();
            num_rows += index.num_rows();
        }
        assert_eq!(num_rows, 2000);
    }

    /// Regression: a second `OptimizeOptions::append()` call on a steady-state
    /// vector index used to fall through to `optimize_vector_indices` and write
    /// a new UUID directory + manifest even though nothing had changed. The
    /// no-op gate inside `merge_indices_with_unindexed_frags` should bail out
    /// once `select_segment_for_single_rebalance` returns `None`.
    #[tokio::test]
    async fn test_optimize_indices_append_is_noop_on_steady_state() {
        const DIM: usize = 64;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = Arc::new(Schema::new(vec![Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            true,
        )]));
        let make_batch = || {
            let arr = Arc::new(
                FixedSizeListArray::try_new_from_values(
                    generate_random_array(1000 * DIM),
                    DIM as i32,
                )
                .unwrap(),
            );
            RecordBatch::try_new(schema.clone(), vec![arr]).unwrap()
        };

        let batches =
            RecordBatchIterator::new(vec![make_batch()].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();

        // num_partitions = 1 so the auto-rebalance heuristic has no join/split
        // candidate after the initial build — keeps this test focused on the
        // append-no-op behavior rather than the rebalance path.
        let params = VectorIndexParams::with_ivf_pq_params(
            MetricType::L2,
            IvfBuildParams::new(1),
            PQBuildParams {
                num_sub_vectors: 2,
                ..Default::default()
            },
        );
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        let batches =
            RecordBatchIterator::new(vec![make_batch()].into_iter().map(Ok), schema.clone());
        dataset.append(batches, None).await.unwrap();

        // First append: folds the new fragment into a fresh delta segment.
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        let version_before = dataset.version().version;
        let object_store = dataset.object_store.as_ref();
        let dirs_before = object_store
            .read_dir(dataset.indices_dir())
            .await
            .unwrap()
            .into_iter()
            .collect::<std::collections::HashSet<_>>();

        // Second append: nothing changed since the previous call. Must not
        // bump the manifest version or add a new UUID directory.
        let mut dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        let dirs_after = object_store
            .read_dir(dataset.indices_dir())
            .await
            .unwrap()
            .into_iter()
            .collect::<std::collections::HashSet<_>>();

        assert_eq!(
            dataset.version().version,
            version_before,
            "second optimize_indices(append()) bumped the dataset version"
        );
        assert_eq!(
            dirs_after, dirs_before,
            "second optimize_indices(append()) created a new index directory"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_query_delta_indices(
        #[values(
            VectorIndexParams::ivf_pq(2, 8, 4, MetricType::L2, 2),
            VectorIndexParams::with_ivf_hnsw_sq_params(
                MetricType::L2,
                IvfBuildParams::new(2),
                HnswBuildParams::default(),
                SQBuildParams::default()
            )
        )]
        index_params: VectorIndexParams,
    ) {
        const DIM: usize = 64;
        const TOTAL: usize = 1000;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let vectors = generate_random_array(TOTAL * DIM);

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DIM as i32,
                ),
                true,
            ),
            Field::new("id", DataType::UInt32, false),
        ]));
        let array = Arc::new(FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                array.clone(),
                Arc::new(UInt32Array::from_iter_values(0..TOTAL as u32)),
            ],
        )
        .unwrap();

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &index_params, true)
            .await
            .unwrap();
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vector_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_indices"], 1);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_unindexed_fragments"], 0);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                array.clone(),
                Arc::new(UInt32Array::from_iter_values(
                    TOTAL as u32..(TOTAL * 2) as u32,
                )),
            ],
        )
        .unwrap();

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        dataset.append(batches, None).await.unwrap();
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vector_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_indices"], 1);
        assert_eq!(stats["num_indexed_fragments"], 1);
        assert_eq!(stats["num_unindexed_fragments"], 1);

        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vector_idx").await.unwrap()).unwrap();
        assert_eq!(stats["num_indices"], 2);
        assert_eq!(stats["num_indexed_fragments"], 2);
        assert_eq!(stats["num_unindexed_fragments"], 0);
        let logical_index = dataset
            .open_logical_vector_index("vector", "vector_idx")
            .await
            .unwrap();
        assert_eq!(logical_index.num_segments(), 2);
        assert_eq!(
            logical_index
                .num_rows_per_segment()
                .into_iter()
                .map(|(_, num_rows)| num_rows)
                .sum::<u64>(),
            2000
        );

        let results = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .nearest("vector", array.value(0).as_primitive::<Float32Type>(), 2)
            .unwrap()
            .nprobes(2)
            .refine(1)
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(results.num_rows(), 2);
        let mut id_arr = results["id"].as_primitive::<UInt32Type>().values().to_vec();
        id_arr.sort();
        assert_eq!(id_arr, vec![0, 1000]);
    }

    #[tokio::test]
    async fn test_merge_indices_after_merge_insert() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        // Create initial dataset using lance_datagen
        let mut dataset = lance_datagen::gen_batch()
            .col("id", array::step::<UInt32Type>())
            .col("value", array::cycle_utf8_literals(&["a", "b", "c"]))
            .col(
                "vector",
                array::rand_vec::<Float32Type>(Dimension::from(64)),
            )
            .into_dataset_with_params(
                test_uri,
                FragmentCount(1),
                FragmentRowCount(1000),
                Some(WriteParams {
                    max_rows_per_file: 1000,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        // Create initial index
        let ivf_params = IvfBuildParams::new(2);
        let pq_params = PQBuildParams {
            num_sub_vectors: 2,
            ..Default::default()
        };
        let params = VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf_params, pq_params);

        dataset
            .create_index(&["vector"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();

        // Load initial index metadata
        let initial_indices = dataset.load_indices().await.unwrap();
        assert_eq!(initial_indices.len(), 1);
        let index_name = initial_indices[0].name.clone();

        // Prepare new data for merge insert (updates to existing rows)
        let new_batch = lance_datagen::gen_batch()
            .col("id", array::step_custom::<UInt32Type>(500, 1)) // IDs 500-999
            .col("value", array::cycle_utf8_literals(&["d", "e", "f"])) // Different values
            .col(
                "vector",
                array::rand_vec::<Float32Type>(Dimension::from(64)),
            )
            .into_batch_rows(RowCount::from(500))
            .unwrap();

        // Record the maximum fragment ID before merge insert
        let max_fragment_id_before = dataset.manifest.max_fragment_id().unwrap_or(0);

        // Execute merge insert operation
        let merge_job =
            MergeInsertBuilder::try_new(Arc::new(dataset.clone()), vec!["id".to_string()])
                .unwrap()
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::InsertAll)
                .try_build()
                .unwrap();

        let schema = new_batch.schema();
        let new_reader = Box::new(RecordBatchIterator::new([Ok(new_batch)], schema.clone()));
        let new_stream = reader_to_stream(new_reader);
        let (updated_dataset, merge_stats) = merge_job.execute(new_stream).await.unwrap();

        // Check merge stats
        assert_eq!(merge_stats.num_updated_rows, 500); // Updates for rows 500-999
        assert_eq!(merge_stats.num_inserted_rows, 0); // No new inserts in this case

        // Get the newly added fragments by comparing fragment IDs
        let unindexed_fragments: Vec<Fragment> = updated_dataset
            .get_fragments()
            .into_iter()
            .filter(|f| f.id() as u64 > max_fragment_id_before)
            .map(|f| f.metadata().clone())
            .collect();

        // Now run merge with known unindexed fragments
        let old_indices = updated_dataset
            .load_indices_by_name(&index_name)
            .await
            .unwrap();
        let old_indices_refs: Vec<&IndexMetadata> = old_indices.iter().collect();

        let merge_result = merge_indices_with_unindexed_frags(
            updated_dataset.clone(),
            &old_indices_refs,
            &unindexed_fragments,
            &OptimizeOptions::merge(old_indices.len()),
        )
        .await
        .unwrap();

        assert!(merge_result.is_some());
        let merge_result = merge_result.unwrap();

        // Verify that the new index covers all fragments
        let new_fragment_bitmap = &merge_result.new_fragment_bitmap;

        // Check that unindexed fragments are now included
        for fragment in &unindexed_fragments {
            assert!(new_fragment_bitmap.contains(fragment.id as u32));
        }

        // Check that old indexed fragments are still included
        // All fragments with ID <= max_fragment_id_before should be included
        for frag_id in 0..=max_fragment_id_before as u32 {
            assert!(new_fragment_bitmap.contains(frag_id));
        }

        // Verify the index can be used for search
        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        let indices = dataset.load_indices().await.unwrap();

        // There should still be indices (old one might be kept plus new one)
        assert!(!indices.is_empty());

        // Test that search works by querying for nearest neighbors
        let query_batch = lance_datagen::gen_batch()
            .col("query", array::rand_vec::<Float32Type>(Dimension::from(64)))
            .into_batch_rows(RowCount::from(1))
            .unwrap();

        let q = query_batch.column(0).as_fixed_size_list();
        let mut scanner = dataset.scan();
        scanner
            .nearest("vector", q.value(0).as_primitive::<Float32Type>(), 10)
            .unwrap();
        let results = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(results[0].num_rows(), 10);
    }

    #[tokio::test]
    async fn test_merge_indices_with_unindexed_frags_vector_subset() {
        const DIM: usize = 64;
        const TOTAL: usize = 1000;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let vectors = generate_random_array(TOTAL * DIM);
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    DIM as i32,
                ),
                true,
            ),
            Field::new("id", DataType::UInt32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(FixedSizeListArray::try_new_from_values(vectors, DIM as i32).unwrap()),
                Arc::new(UInt32Array::from_iter_values(0..TOTAL as u32)),
            ],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
        let mut dataset = Dataset::write(batches, test_uri, None).await.unwrap();
        let index_params = VectorIndexParams::ivf_pq(2, 8, 4, MetricType::L2, 2);
        dataset
            .create_index(&["vector"], IndexType::Vector, None, &index_params, true)
            .await
            .unwrap();

        let next_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        generate_random_array(TOTAL * DIM),
                        DIM as i32,
                    )
                    .unwrap(),
                ),
                Arc::new(UInt32Array::from_iter_values(
                    TOTAL as u32..(TOTAL * 2) as u32,
                )),
            ],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![next_batch].into_iter().map(Ok), schema);
        dataset.append(batches, None).await.unwrap();
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();

        let indices = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert_eq!(indices.len(), 2);
        let subset = vec![&indices[1]];
        let merge_result = merge_indices_with_unindexed_frags(
            Arc::new(dataset),
            &subset,
            &[],
            &OptimizeOptions::merge(1),
        )
        .await
        .unwrap();

        assert!(
            merge_result.is_some(),
            "subset merges should respect the caller-provided indices"
        );
    }

    #[tokio::test]
    async fn test_optimize_btree_multi_segment_optimize_default() {
        async fn query_id_count(dataset: &Dataset, id: &str) -> usize {
            dataset
                .scan()
                .filter(&format!("id = '{}'", id))
                .unwrap()
                .project(&["id"])
                .unwrap()
                .try_into_batch()
                .await
                .unwrap()
                .num_rows()
        }

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
        let make_batch = |start: i32, end: i32| {
            let ids = StringArray::from_iter_values((start..end).map(|i| format!("song-{i}")));
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids)]).unwrap()
        };

        // Three fragments of 64 rows each; each commits as its own BTree
        // segment so optimize sees a multi-segment scalar logical index.
        let reader = RecordBatchIterator::new(
            vec![
                Ok(make_batch(0, 64)),
                Ok(make_batch(64, 128)),
                Ok(make_batch(128, 192)),
            ],
            schema.clone(),
        );
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 64,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 3);

        let mut staged_segments = Vec::new();
        for fragment in &fragments {
            let segment = crate::index::create::CreateIndexBuilder::new(
                &mut dataset,
                &["id"],
                IndexType::BTree,
                &params,
            )
            .name("id_idx".into())
            .fragments(vec![fragment.id() as u32])
            .execute_uncommitted()
            .await
            .unwrap();
            staged_segments.push(segment);
        }
        dataset
            .commit_existing_index_segments("id_idx", "id", staged_segments)
            .await
            .unwrap();
        assert_eq!(
            dataset.load_indices_by_name("id_idx").await.unwrap().len(),
            3
        );

        let appended = RecordBatchIterator::new(vec![Ok(make_batch(192, 256))], schema.clone());
        let mut dataset = Dataset::write(
            appended,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 64,
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 4);

        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();

        // Reload from disk to ensure we're reading committed manifest state.
        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();

        // Each of these IDs lives in a distinct old segment / fragment.
        // song-10 lives in fragment 0, song-80 in fragment 1, song-160 in
        // fragment 2, and song-200 in the appended fragment. After optimize
        // every row must still be reachable through the logical index,
        // regardless of which segment absorbed the new data.
        for id in ["song-10", "song-80", "song-160", "song-200"] {
            assert_eq!(
                query_id_count(&dataset, id).await,
                1,
                "expected exactly one row for {id} after multi-segment optimize"
            );
        }

        // `OptimizeOptions::default()` (= num_indices_to_merge: None) merges
        // the newest segment with the unindexed fragment, like the
        // inverted/vector default. The three old segments minus the merged one
        // plus the new delta means three segments remain, and together they
        // must still cover every dataset fragment without overlap.
        let segments_after = dataset.load_indices_by_name("id_idx").await.unwrap();
        assert_eq!(
            segments_after.len(),
            3,
            "default optimize must merge one delta, not all segments, got {segments_after:?}"
        );
        let mut covered = RoaringBitmap::new();
        for segment in &segments_after {
            let bitmap = segment
                .fragment_bitmap
                .as_ref()
                .expect("each segment should carry fragment coverage");
            assert!(
                covered.is_disjoint(bitmap),
                "post-optimize segments must not overlap, got {segments_after:?}"
            );
            covered |= bitmap;
        }
        let mut expected = RoaringBitmap::new();
        for frag in dataset.get_fragments() {
            expected.insert(frag.id() as u32);
        }
        assert_eq!(
            covered, expected,
            "post-optimize segments should cover every dataset fragment"
        );
    }

    #[tokio::test]
    async fn test_optimize_fmindex_default_rebuilds_old_and_new_rows() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let make_batch = |values: &[&str]| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(StringArray::from_iter_values(
                    values.iter().copied(),
                ))],
            )
            .unwrap()
        };

        let reader = RecordBatchIterator::new(
            vec![Ok(make_batch(&["old alpha needle", "old beta"]))],
            schema.clone(),
        );
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::Fm);
        dataset
            .create_index(
                &["text"],
                IndexType::Fm,
                Some("text_fmindex".to_string()),
                &params,
                true,
            )
            .await
            .unwrap();

        let appended = RecordBatchIterator::new(
            vec![Ok(make_batch(&["new gamma needle", "new delta"]))],
            schema.clone(),
        );
        dataset.append(appended, None).await.unwrap();

        assert!(
            !dataset
                .unindexed_fragments("text_fmindex")
                .await
                .unwrap()
                .is_empty()
        );

        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();

        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        assert!(
            dataset
                .unindexed_fragments("text_fmindex")
                .await
                .unwrap()
                .is_empty()
        );

        let committed = dataset.load_indices_by_name("text_fmindex").await.unwrap();
        assert_eq!(committed.len(), 1);
        assert_eq!(
            committed[0]
                .fragment_bitmap
                .as_ref()
                .expect("FMIndex segment should carry fragment coverage")
                .len(),
            2
        );

        let logical = crate::index::scalar_logical::open_named_scalar_index(
            &dataset,
            "text",
            "text_fmindex",
            &NoOpMetricsCollector,
        )
        .await
        .unwrap();

        for (pattern, expected) in [("old alpha", 1), ("new gamma", 1), ("needle", 2)] {
            let query = TextQuery::StringContains(pattern.to_string());
            let result = logical.search(&query, &NoOpMetricsCollector).await.unwrap();
            let row_addrs = match result {
                SearchResult::Exact(row_addrs) => row_addrs,
                other => panic!("expected exact result for {pattern}, got {other:?}"),
            };
            let count = row_addrs.true_rows().row_addrs().unwrap().count();
            assert_eq!(
                count, expected,
                "expected {expected} matches for {pattern}, got {count}"
            );
        }
    }

    #[tokio::test]
    async fn test_optimize_btree_optimize_append() {
        async fn query_id_count(dataset: &Dataset, id: &str) -> usize {
            dataset
                .scan()
                .filter(&format!("id = '{}'", id))
                .unwrap()
                .project(&["id"])
                .unwrap()
                .try_into_batch()
                .await
                .unwrap()
                .num_rows()
        }

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
        let make_batch = |start: i32, end: i32| {
            let ids = StringArray::from_iter_values((start..end).map(|i| format!("song-{i}")));
            RecordBatch::try_new(schema.clone(), vec![Arc::new(ids)]).unwrap()
        };

        // Start with two fragments + two committed BTree segments.
        let reader = RecordBatchIterator::new(
            vec![Ok(make_batch(0, 64)), Ok(make_batch(64, 128))],
            schema.clone(),
        );
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 64,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::BTree);
        let original_segment_uuids: Vec<_> = {
            let mut staged = Vec::new();
            for fragment in dataset.get_fragments() {
                let segment = crate::index::create::CreateIndexBuilder::new(
                    &mut dataset,
                    &["id"],
                    IndexType::BTree,
                    &params,
                )
                .name("id_idx".into())
                .fragments(vec![fragment.id() as u32])
                .execute_uncommitted()
                .await
                .unwrap();
                staged.push(segment);
            }
            let uuids = staged.iter().map(|s| s.uuid).collect::<Vec<_>>();
            dataset
                .commit_existing_index_segments("id_idx", "id", staged)
                .await
                .unwrap();
            uuids
        };
        assert_eq!(original_segment_uuids.len(), 2);

        // Append a third fragment, leave it unindexed, then run append-mode optimize.
        let appended = RecordBatchIterator::new(vec![Ok(make_batch(128, 192))], schema.clone());
        let mut dataset = Dataset::write(
            appended,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 64,
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();

        // Read fresh from disk to make sure we're inspecting committed state.
        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();

        // append() must preserve every original old segment unchanged and add
        // exactly one new segment covering only the newly appended fragments.
        let committed = dataset.load_indices_by_name("id_idx").await.unwrap();
        let committed_uuids: std::collections::HashSet<_> =
            committed.iter().map(|idx| idx.uuid).collect();
        for original in &original_segment_uuids {
            assert!(
                committed_uuids.contains(original),
                "append() must not remove pre-existing segment {original}, \
                 but the committed UUIDs are {committed_uuids:?}"
            );
        }
        assert_eq!(
            committed.len(),
            original_segment_uuids.len() + 1,
            "append() should add exactly one new delta segment, got {committed:?}"
        );
        let new_segment = committed
            .iter()
            .find(|idx| !original_segment_uuids.contains(&idx.uuid))
            .expect("append() must add a new delta segment");
        let new_segment_frags: Vec<_> = new_segment
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .iter()
            .collect();
        // The appended fragment should be the only one covered by the new delta;
        // old segments retain their own coverage.
        assert_eq!(new_segment_frags.len(), 1);

        // Sanity check: queries across all fragments still return their rows.
        for id in ["song-10", "song-100", "song-160"] {
            assert_eq!(query_id_count(&dataset, id).await, 1, "missing row {id}");
        }
    }

    #[tokio::test]
    async fn test_optimize_bitmap_index_append() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = Arc::new(Schema::new(vec![Field::new(
            "category",
            DataType::Utf8,
            false,
        )]));
        let make_batch = |labels: &[&str]| {
            let arr = StringArray::from_iter_values(labels.iter().copied());
            RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).unwrap()
        };

        // One fragment + one Bitmap segment.
        let reader =
            RecordBatchIterator::new(vec![Ok(make_batch(&["a", "b", "a", "c"]))], schema.clone());
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 4,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = ScalarIndexParams::for_builtin(lance_index::scalar::BuiltinIndexType::Bitmap);
        dataset
            .create_index(
                &["category"],
                IndexType::Bitmap,
                Some("cat_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();
        let original_uuid = {
            let committed = dataset.load_indices_by_name("cat_idx").await.unwrap();
            assert_eq!(committed.len(), 1);
            committed[0].uuid
        };

        // Append a second fragment, leave it unindexed, then optimize with
        // `append()` (= num_indices_to_merge: Some(0)).
        let appended =
            RecordBatchIterator::new(vec![Ok(make_batch(&["b", "d", "d", "a"]))], schema.clone());
        let mut dataset = Dataset::write(
            appended,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 4,
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();

        // append() (= num_indices_to_merge: Some(0)) is now honored uniformly:
        // Bitmap, like BTree, must keep the original segment untouched and add
        // exactly one delta segment covering only the appended fragment.
        let committed = dataset.load_indices_by_name("cat_idx").await.unwrap();
        assert_eq!(
            committed.len(),
            2,
            "Bitmap optimize append() must add a delta segment, not merge, got {committed:?}"
        );
        assert!(
            committed.iter().any(|idx| idx.uuid == original_uuid),
            "append() must preserve the pre-existing segment {original_uuid}, got {committed:?}"
        );
        let new_segment = committed
            .iter()
            .find(|idx| idx.uuid != original_uuid)
            .expect("append() must add a new delta segment");
        let new_segment_frags: std::collections::BTreeSet<u32> = new_segment
            .fragment_bitmap
            .as_ref()
            .expect("delta Bitmap should carry fragment coverage")
            .iter()
            .collect();
        assert_eq!(
            new_segment_frags,
            [1u32].into_iter().collect(),
            "the delta segment must cover only the appended fragment"
        );

        // Data correctness: a value that lives only in the appended fragment
        // is queryable through the (now multi-segment) index.
        let rows = dataset
            .scan()
            .filter("category = 'd'")
            .unwrap()
            .project(&["category"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows();
        assert_eq!(rows, 2, "value 'd' lives in appended fragment");
    }

    #[tokio::test]
    async fn test_optimize_btree_keeps_rows_with_stable_row_ids_after_compaction() {
        async fn query_id_count(dataset: &Dataset, id: &str) -> usize {
            dataset
                .scan()
                .filter(&format!("id = '{}'", id))
                .unwrap()
                .project(&["id"])
                .unwrap()
                .try_into_batch()
                .await
                .unwrap()
                .num_rows()
        }

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
        let ids = StringArray::from_iter_values((0..256).map(|i| format!("song-{i}")));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids)]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 64,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".into()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        assert_eq!(query_id_count(&dataset, "song-42").await, 1);

        compact_files(
            &mut dataset,
            crate::dataset::optimize::CompactionOptions {
                target_rows_per_fragment: 512,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        let frags = dataset.get_fragments();
        assert!(!frags.is_empty());
        assert!(frags.iter().all(|frag| frag.id() > 0));
        assert!(
            dataset
                .unindexed_fragments("id_idx")
                .await
                .unwrap()
                .is_empty()
        );

        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();

        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        assert_eq!(query_id_count(&dataset, "song-42").await, 1);
    }

    /// Under stable row ids, updating an indexed column and then calling
    /// `optimize_indices` must not leave stale entries (old value -> updated row)
    /// in the scalar index. An update deletes the old copy of each row and
    /// rewrites it under the same stable row id, so the old index entry is stale
    /// and must be dropped on merge. Covers BTree, Bitmap, and Inverted (FTS),
    /// which take three different merge paths.
    #[tokio::test]
    async fn test_optimize_scalar_index_drops_stale_rows_after_update() {
        use crate::dataset::UpdateBuilder;
        use arrow_array::Int32Array;
        use lance_index::scalar::FullTextSearchQuery;
        use lance_index::scalar::inverted::InvertedIndexParams;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        // 100 rows: num == id; cat = "A" for id<50 else "B"; body = "alpha" for
        // id<50 else "beta".
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("num", DataType::Int32, false),
            Field::new("cat", DataType::Utf8, false),
            Field::new("body", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..100)),
                Arc::new(Int32Array::from_iter_values(0..100)),
                Arc::new(StringArray::from_iter_values(
                    (0..100).map(|i| if i < 50 { "A" } else { "B" }),
                )),
                Arc::new(StringArray::from_iter_values(
                    (0..100).map(|i| if i < 50 { "alpha" } else { "beta" }),
                )),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        dataset
            .create_index(
                &["num"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &["cat"],
                IndexType::Bitmap,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        dataset
            .create_index(
                &["body"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Update the first 25 rows (id < 25): num -> -1, cat -> 'B', body -> 'beta'.
        let res = UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("id < 25")
            .unwrap()
            .set("num", "-1")
            .unwrap()
            .set("cat", "'B'")
            .unwrap()
            .set("body", "'beta'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset = res.new_dataset.as_ref().clone();

        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();

        // BTree: `num >= 0` matches ids 25..99 (75 rows); the 25 updated rows
        // hold num = -1 and must not appear.
        let btree_count = dataset
            .scan()
            .filter("num >= 0")
            .unwrap()
            .count_rows()
            .await
            .unwrap();
        assert_eq!(btree_count, 75, "btree returned stale/incorrect rows");

        // Bitmap: only the 25 rows (ids 25..49) that still carry cat = 'A' match;
        // the 25 rows updated to 'B' must not.
        let bitmap_count = dataset
            .scan()
            .filter("cat = 'A'")
            .unwrap()
            .count_rows()
            .await
            .unwrap();
        assert_eq!(bitmap_count, 25, "bitmap returned stale rows");

        // FTS: only the 25 rows (ids 25..49) whose body still reads "alpha" match;
        // the 25 rows updated to "beta" must not.
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new("alpha".to_owned()))
            .unwrap();
        let fts_count = scan.count_rows().await.unwrap();
        assert_eq!(fts_count, 25, "FTS index returned stale rows");
    }

    /// Multi-segment variant (Jack Ye's repro, PR #7359): with one BTree segment
    /// per fragment, default optimize merges only the tail segment. A stable-row-id
    /// update to a row in an older segment's fragment must still drop that
    /// segment's stale postings -- the merge has to reach back to cover it.
    #[tokio::test]
    async fn test_optimize_btree_drops_stale_rows_across_segments_after_update() {
        use crate::dataset::UpdateBuilder;
        use crate::index::CreateIndexBuilder;
        use arrow_array::Int32Array;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("num", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..100)),
                Arc::new(Int32Array::from_iter_values(0..100)),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        // Two fragments (0..49, 50..99) -> one BTree segment each.
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: 50,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        let fragments = dataset.get_fragments();
        let mut segments = Vec::new();
        for fragment in &fragments {
            segments.push(
                CreateIndexBuilder::new(&mut dataset, &["num"], IndexType::BTree, &params)
                    .name("num_idx".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }
        dataset
            .commit_existing_index_segments("num_idx", "num", segments)
            .await
            .unwrap();

        // Update the first 25 rows (in the first/older segment's fragment).
        let res = UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("id < 25")
            .unwrap()
            .set("num", "-1")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset = res.new_dataset.as_ref().clone();

        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();

        assert_eq!(
            dataset
                .scan()
                .filter("num = 0")
                .unwrap()
                .count_rows()
                .await
                .unwrap(),
            0,
            "stale entry leaked from the older, unmerged segment"
        );
        assert_eq!(
            dataset
                .scan()
                .filter("num >= 0")
                .unwrap()
                .count_rows()
                .await
                .unwrap(),
            75
        );
    }

    /// Same multi-segment gap for FTS, which takes the separate Inverted dispatch
    /// path. One Inverted segment per fragment; an update to the older segment's
    /// fragment must not leave its old-token postings searchable.
    #[tokio::test]
    async fn test_optimize_fts_drops_stale_rows_across_segments_after_update() {
        use crate::dataset::UpdateBuilder;
        use crate::index::CreateIndexBuilder;
        use arrow_array::Int32Array;
        use lance_index::scalar::FullTextSearchQuery;
        use lance_index::scalar::inverted::InvertedIndexParams;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("body", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..100)),
                Arc::new(StringArray::from_iter_values(
                    (0..100).map(|i| if i < 50 { "alpha" } else { "beta" }),
                )),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: 50,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = InvertedIndexParams::default();
        let fragments = dataset.get_fragments();
        let mut segments = Vec::new();
        for fragment in &fragments {
            segments.push(
                CreateIndexBuilder::new(&mut dataset, &["body"], IndexType::Inverted, &params)
                    .name("body_idx".to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }
        dataset
            .commit_existing_index_segments("body_idx", "body", segments)
            .await
            .unwrap();

        // Update the first 25 rows (older segment's fragment): body -> "beta".
        let res = UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("id < 25")
            .unwrap()
            .set("body", "'beta'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset = res.new_dataset.as_ref().clone();

        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();

        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new("alpha".to_owned()))
            .unwrap();
        assert_eq!(
            scan.count_rows().await.unwrap(),
            25,
            "FTS stale rows leaked from the older, unmerged segment"
        );
    }

    /// `optimize_indices` builds the stable-row-id allow-list by subtracting each
    /// fragment's deletion vector. If a deletion vector cannot be read, the merge
    /// must fail loudly: swallowing the error (treating the load as "no
    /// deletions") would put every deleted row back into the allow-list and
    /// silently reintroduce the stale entries this fix removes. Simulate an
    /// unreadable deletion vector by deleting the file the manifest still
    /// references, then assert optimize errors instead of succeeding.
    #[tokio::test]
    async fn test_optimize_errors_when_deletion_vector_unreadable() {
        use crate::dataset::UpdateBuilder;
        use arrow_array::Int32Array;
        use lance_table::io::deletion::deletion_file_path;

        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("num", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..100)),
                Arc::new(Int32Array::from_iter_values(0..100)),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        dataset
            .create_index(
                &["num"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Update rewrites the first 25 rows under the same stable row ids,
        // leaving a deletion vector on the original fragment.
        UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("id < 25")
            .unwrap()
            .set("num", "-1")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        // Reload cold (nothing has cached the deletion vector), then remove the
        // deletion file the manifest still references so the next read fails.
        let mut dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        let mut removed = 0;
        for fragment in dataset.get_fragments() {
            if let Some(deletion_file) = fragment.metadata().deletion_file.clone() {
                let path =
                    deletion_file_path(&dataset.base, fragment.metadata().id, &deletion_file);
                dataset.object_store.delete(&path).await.unwrap();
                removed += 1;
            }
        }
        assert_eq!(
            removed, 1,
            "update should have left exactly one deletion file"
        );

        let result = dataset.optimize_indices(&OptimizeOptions::default()).await;
        assert!(
            result.is_err(),
            "optimize must fail when a deletion vector cannot be read, not \
             silently keep the deleted rows in the index"
        );
    }

    #[tokio::test]
    async fn test_optimize_scalar_no_unindexed_fragments() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
        let ids = StringArray::from_iter_values((0..32).map(|i| format!("song-{i}")));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids)]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let mut dataset = Dataset::write(reader, test_uri, None).await.unwrap();

        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".into()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        let before = dataset.load_indices_by_name("id_idx").await.unwrap();
        assert_eq!(before.len(), 1);
        let original_uuid = before[0].uuid;
        let original_version = dataset.manifest.version;

        // `merge(1)` would historically rebuild the single existing segment
        // (steady state, nothing unindexed) and replace its UUID; with the
        // short-circuit it must skip work entirely.
        dataset
            .optimize_indices(&OptimizeOptions::merge(1))
            .await
            .unwrap();

        let after = dataset.load_indices_by_name("id_idx").await.unwrap();
        assert_eq!(after.len(), 1, "no new segment should be produced");
        assert_eq!(
            after[0].uuid, original_uuid,
            "no-op optimize must not churn the index UUID"
        );
        assert_eq!(
            dataset.manifest.version, original_version,
            "no-op optimize must not advance the dataset version"
        );

        // The default options also short-circuit (num_to_merge defaults to 1
        // when there is a single old segment).
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
        let after_default = dataset.load_indices_by_name("id_idx").await.unwrap();
        assert_eq!(after_default[0].uuid, original_uuid);
        assert_eq!(dataset.manifest.version, original_version);
    }

    #[rstest]
    #[case::address_row_ids(false)]
    #[case::stable_row_ids(true)]
    #[tokio::test]
    async fn test_optimize_btree_no_duplicate_row_addr(#[case] use_stable_row_ids: bool) {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("payload", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int32Array::from(vec![10])),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        let write_params = WriteParams {
            enable_stable_row_ids: use_stable_row_ids,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Reordered source columns (payload, id) force the partial-schema
        // RewriteColumns path instead of a full row rewrite.
        let source_schema = Arc::new(Schema::new(vec![
            Field::new("payload", DataType::Int32, false),
            Field::new("id", DataType::Int32, false),
        ]));
        let source_batch = RecordBatch::try_new(
            source_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![100])),
                Arc::new(Int32Array::from(vec![1])),
            ],
        )
        .unwrap();
        let merge_job =
            MergeInsertBuilder::try_new(Arc::new(dataset.clone()), vec!["id".to_string()])
                .unwrap()
                .when_matched(WhenMatched::UpdateAll)
                .try_build()
                .unwrap();
        let source_reader = Box::new(RecordBatchIterator::new(
            [Ok(source_batch)],
            source_schema.clone(),
        ));
        merge_job
            .execute(reader_to_stream(source_reader))
            .await
            .unwrap();

        // Build a delta BTree segment over the now-unindexed fragment.
        let mut dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        assert_eq!(
            dataset.load_indices_by_name("id_idx").await.unwrap().len(),
            2,
            "append must create a delta segment over the rewritten fragment"
        );

        // Force the old segment + delta segment to merge.
        dataset
            .optimize_indices(&OptimizeOptions::merge(2))
            .await
            .unwrap();

        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        let rows = dataset
            .scan()
            .filter("id = 1")
            .unwrap()
            .project(&["id"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows();
        assert_eq!(rows, 1, "id = 1 must return exactly one row after merge");
    }

    #[tokio::test]
    async fn test_optimize_btree_merge_remaps_deferred_compaction() {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let make = |range: std::ops::Range<i32>| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from_iter_values(range))],
            )
            .unwrap()
        };

        // Two fragments: [0, 50) and [50, 100).
        let reader =
            RecordBatchIterator::new(vec![Ok(make(0..50)), Ok(make(50..100))], schema.clone());
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 50,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);

        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();

        // Deferred-remap compaction fuses the two fragments into one and leaves a
        // pending FragReuseIndex; the index segment is not eagerly remapped.
        compact_files(
            &mut dataset,
            CompactionOptions {
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        // Append a third fragment, left unindexed.
        let mut dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(make(100..150))], schema.clone()),
                None,
            )
            .await
            .unwrap();

        // Merge the deferred-remapped old segment with the new delta.
        dataset
            .optimize_indices(&OptimizeOptions::merge(2))
            .await
            .unwrap();

        let dataset = DatasetBuilder::from_uri(test_uri).load().await.unwrap();
        // A value from the compacted fragments must still be found via the index.
        let hit = dataset
            .scan()
            .filter("id = 25")
            .unwrap()
            .project(&["id"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows();
        assert_eq!(
            hit, 1,
            "compacted-then-merged row must remain queryable via the index"
        );
        let total = dataset
            .scan()
            .filter("id >= 0")
            .unwrap()
            .project(&["id"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows();
        assert_eq!(total, 150, "no rows may be lost across compaction + merge");
    }
}
