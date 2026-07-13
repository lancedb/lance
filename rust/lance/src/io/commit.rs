// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Trait for commit implementations.
//!
//! In Lance, a transaction is committed by writing the next manifest file.
//! However, care should be taken to ensure that the manifest file is written
//! only once, even if there are concurrent writers. Different stores have
//! different abilities to handle concurrent writes, so a trait is provided
//! to allow for different implementations.
//!
//! The trait [`CommitHandler`] can be implemented to provide different commit
//! strategies. The default implementation for most object stores is
//! `ConditionalPutCommitHandler`, which writes the manifest to a temporary path, then
//! renames the temporary path to the final path if no object already exists
//! at the final path.
//!
//! When providing your own commit handler, most often you are implementing in
//! terms of a lock. The trait `CommitLock` can be implemented as a simpler
//! alternative to [`CommitHandler`].

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::num::NonZero;
use std::sync::Arc;
use std::time::Instant;

use conflict_resolver::TransactionRebase;
use lance_core::utils::address::RowAddress;
use lance_core::utils::backoff::{Backoff, SlotBackoff};
use lance_file::version::LanceFileVersion;
use lance_index::metrics::NoOpMetricsCollector;
use lance_io::utils::CachedFileSize;
use lance_select::RowAddrTreeMap;
use lance_table::format::{
    DETACHED_VERSION_MASK, DataStorageFormat, DeletionFile, Fragment, IndexMetadata,
    LogicalRowAddressSelection, Manifest, RowAddressFieldChange, RowAddressLayoutDelta,
    RowAddressPlacementDelta, RowAddressPlacementKind, RowAddressSourceFloor,
    RowAddressTargetFragment, RowAlignedRewriteProof, WriterVersion, fingerprint_deleted_offsets,
    is_detached_version, list_index_files_with_sizes, pb,
};
use lance_table::io::commit::{
    CommitConfig, CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme,
};
use rand::{Rng, rng};

use super::ObjectStore;
use crate::Dataset;
use crate::dataset::cleanup::auto_cleanup_hook;
use crate::dataset::fragment::FileFragment;
use crate::dataset::transaction::{
    Operation, RowAddressManifestApplyContext, Transaction, data_replacement_successor_fragment,
    logical_index_is_monotonic_coverage_restriction, merge_rewritten_existing_field_ids,
    row_aligned_data_file_identity_eq,
};
use crate::dataset::{
    ManifestWriteConfig, NewTransactionResult, TRANSACTIONS_DIR, load_new_transactions,
    write_manifest_file,
};
use crate::index::DatasetIndexInternalExt;
use crate::index::vector::details::infer_missing_vector_details;
use crate::index::{
    DatasetIndexExt, load_authoritative_logical_index_coverage,
    resolve_logical_index_coverage_from_authoritative, resolve_logical_index_metadata,
    validate_index_contract, validate_resolved_logical_index_overlaps,
};
use crate::io::deletion::read_dataset_deletion_file;
use crate::session::Session;
use crate::session::caches::DSMetadataCache;
use crate::session::index_caches::IndexMetadataKey;
use futures::future::Either;
use futures::{StreamExt, TryFutureExt, TryStreamExt};
use lance_core::{Error, Result};
use lance_index::is_system_index;
use lance_io::object_store::ObjectStoreRegistry;
use log;
use object_store::ObjectStoreExt;
use object_store::path::Path;
use prost::Message;
use roaring::{RoaringBitmap, RoaringTreemap};
use uuid::Uuid;

pub mod conflict_resolver;
#[cfg(all(feature = "dynamodb_tests", test))]
mod dynamodb;
#[cfg(test)]
mod external_manifest;
pub mod namespace_manifest;
#[cfg(all(feature = "dynamodb_tests", test))]
mod s3_test;

/// Read the transaction data from a transaction file.
pub(crate) async fn read_transaction_file(
    object_store: &ObjectStore,
    base_path: &Path,
    transaction_file: &str,
) -> Result<Transaction> {
    let path = base_path
        .clone()
        .join(TRANSACTIONS_DIR)
        .join(transaction_file);
    let result = object_store.inner.get(&path).await?;
    let data = result.bytes().await?;
    let transaction = pb::Transaction::decode(data)?;
    transaction.try_into()
}

/// Best-effort delete of a transaction file that is no longer needed.
///
/// Logs a warning on failure rather than propagating the error, since the
/// primary operation has already failed and the orphaned file will eventually
/// be removed by GC.
async fn cleanup_transaction_file(
    object_store: &ObjectStore,
    base_path: &Path,
    transaction_file: &str,
) {
    if transaction_file.is_empty() {
        return;
    }
    let path = base_path
        .clone()
        .join(TRANSACTIONS_DIR)
        .join(transaction_file);
    if let Err(e) = object_store.delete(&path).await {
        log::warn!(
            "Failed to clean up orphaned transaction file '{}': {}",
            transaction_file,
            e
        );
    }
}

fn transaction_file_name(transaction: &Transaction) -> String {
    format!("{}-{}.txn", transaction.read_version, transaction.uuid)
}

fn check_protobuf_capacity(message: &impl Message, section: &str) -> Result<()> {
    let encoded_len = message.encoded_len();
    u32::try_from(encoded_len).map_err(|_| {
        Error::format_capacity_exceeded(format!(
            "{section} protobuf is {encoded_len} bytes but the Lance framing limit is {} bytes",
            u32::MAX
        ))
    })?;
    Ok(())
}

/// Validate every length-delimited protobuf before publishing any commit object.
///
/// Section offsets are populated by the manifest writer.  Use their largest
/// wire representation here so the final manifest cannot cross the format
/// boundary merely because those offsets were added after preflight.
fn validate_manifest_write_capacity(
    manifest: &Manifest,
    indices: &[IndexMetadata],
    transaction: &Transaction,
) -> Result<()> {
    if !indices.is_empty() {
        check_protobuf_capacity(
            &pb::IndexSection {
                indices: indices.iter().map(Into::into).collect(),
            },
            "index section",
        )?;
    }
    check_protobuf_capacity(&pb::Transaction::from(transaction), "transaction section")?;

    let mut projected_manifest = manifest.clone();
    projected_manifest.index_section = (!indices.is_empty()).then_some(usize::MAX);
    projected_manifest.transaction_section = Some(usize::MAX);
    check_protobuf_capacity(&pb::Manifest::from(&projected_manifest), "manifest section")
}

fn transaction_requires_row_address_context(transaction: &Transaction) -> bool {
    transaction.row_address_layout_delta.is_some()
}

async fn load_row_address_deletion_bitmap(
    dataset: &Dataset,
    fragment: &Fragment,
) -> Result<Option<(u32, RoaringBitmap)>> {
    let Some(deletion_file) = fragment.deletion_file.as_ref() else {
        return Ok(None);
    };
    let fragment_id = u32::try_from(fragment.id).map_err(|_| {
        Error::invalid_input(format!(
            "storage-version-2.3 fragment id {} exceeds row-address capacity",
            fragment.id
        ))
    })?;
    let deletion_vector = read_dataset_deletion_file(dataset, fragment.id, deletion_file).await?;
    Ok(Some((
        fragment_id,
        RoaringBitmap::from(deletion_vector.as_ref()),
    )))
}

async fn prepare_row_address_manifest_context(
    dataset: &Dataset,
    transaction: &Transaction,
) -> Result<Option<RowAddressManifestApplyContext>> {
    if !dataset.manifest.uses_stable_logical_row_addresses()
        || !transaction_requires_row_address_context(transaction)
    {
        return Ok(None);
    }
    let delta = transaction
        .row_address_layout_delta
        .as_ref()
        .expect("row-address context predicate requires a delta");
    // A pure ExplicitMap replacement carries the exact complement of its live
    // output as a replacement mask.  That complement legitimately includes
    // rows retired by an earlier delete; all other delta kinds must still
    // resolve every newly retired source row in the current snapshot.
    let allows_already_retired = delta.is_pure_explicit_rewrite();
    let mut current_fragment_ids = HashSet::<u32>::new();
    let mut pending = Vec::<u64>::with_capacity(4096);
    for selection in &delta.retired_selections {
        for logical_address in selection.iter() {
            pending.push(logical_address?.raw());
            if pending.len() == 4096 {
                for physical_address in dataset.resolve_logical_row_ids_async(&pending).await? {
                    match physical_address {
                        Some(physical_address) => {
                            current_fragment_ids.insert(physical_address.fragment_id());
                        }
                        None if allows_already_retired => {}
                        None => {
                            return Err(Error::invalid_input(
                                "retired logical selection contains an unmapped source address",
                            ));
                        }
                    }
                }
                pending.clear();
            }
        }
    }
    if !pending.is_empty() {
        for physical_address in dataset.resolve_logical_row_ids_async(&pending).await? {
            match physical_address {
                Some(physical_address) => {
                    current_fragment_ids.insert(physical_address.fragment_id());
                }
                None if allows_already_retired => {}
                None => {
                    return Err(Error::invalid_input(
                        "retired logical selection contains an unmapped source address",
                    ));
                }
            }
        }
    }
    current_fragment_ids.extend(
        delta
            .row_aligned_rewrite_proofs
            .iter()
            .map(|proof| proof.physical_fragment_id),
    );

    let mut context = RowAddressManifestApplyContext {
        explicit_map_placements: delta.explicit_map_placements.clone(),
        ..Default::default()
    };
    let mut record_fully_deleted_fragment = |fragment_id: u64| -> Result<()> {
        context
            .newly_fully_deleted_source_fragments
            .insert(u32::try_from(fragment_id).map_err(|_| {
                Error::invalid_input(format!(
                    "fully deleted fragment id {fragment_id} exceeds row-address capacity"
                ))
            })?);
        Ok(())
    };
    match &transaction.operation {
        Operation::Delete {
            deleted_fragment_ids,
            ..
        } => {
            for fragment_id in deleted_fragment_ids {
                record_fully_deleted_fragment(*fragment_id)?;
            }
        }
        Operation::Update {
            removed_fragment_ids,
            ..
        } => {
            for fragment_id in removed_fragment_ids {
                record_fully_deleted_fragment(*fragment_id)?;
            }
        }
        Operation::Rewrite { groups, .. } => {
            // Rewrite removes each source fragment as a physical object.  The
            // format validator separately rejects treating a reused fragment
            // ID as fully deleted when it is still present in the successor.
            for fragment in groups.iter().flat_map(|group| &group.old_fragments) {
                record_fully_deleted_fragment(fragment.id)?;
            }
        }
        _ => {}
    }
    for fragment_id in current_fragment_ids {
        if context
            .newly_fully_deleted_source_fragments
            .contains(&fragment_id)
        {
            continue;
        }
        let fragment = dataset
            .manifest
            .fragments
            .iter()
            .find(|fragment| fragment.id == fragment_id as u64)
            .ok_or_else(|| {
                Error::internal(format!(
                    "retired row resolves to missing source fragment {fragment_id}"
                ))
            })?;
        if let Some((fragment_id, bitmap)) =
            load_row_address_deletion_bitmap(dataset, fragment).await?
        {
            context.current_deletion_vectors.insert(fragment_id, bitmap);
        }
    }

    let successor_overrides: Vec<(Fragment, bool)> = match &transaction.operation {
        Operation::Delete {
            updated_fragments, ..
        } => updated_fragments
            .iter()
            .cloned()
            .map(|fragment| (fragment, false))
            .collect(),
        Operation::Update {
            updated_fragments,
            new_fragments,
            ..
        } => updated_fragments
            .iter()
            .cloned()
            .map(|fragment| (fragment, false))
            .chain(
                new_fragments
                    .iter()
                    .cloned()
                    .map(|fragment| (fragment, true)),
            )
            .collect(),
        Operation::Rewrite { groups, .. } => groups
            .iter()
            .flat_map(|group| group.new_fragments.iter().cloned())
            .map(|fragment| (fragment, true))
            .collect(),
        Operation::Merge { fragments, .. } => fragments
            .iter()
            .cloned()
            .map(|fragment| (fragment, false))
            .collect(),
        Operation::Overwrite { fragments, .. } | Operation::Append { fragments } => fragments
            .iter()
            .cloned()
            .map(|fragment| (fragment, true))
            .collect(),
        _ => Vec::new(),
    };
    for (fragment, speculative) in &successor_overrides {
        if *speculative && fragment.id == 0 {
            if fragment.deletion_file.is_some() {
                return Err(Error::invalid_input(
                    "a speculative storage-version-2.3 fragment cannot carry a deletion file",
                ));
            }
            continue;
        }
        if let Some((fragment_id, bitmap)) =
            load_row_address_deletion_bitmap(dataset, fragment).await?
        {
            context
                .successor_deletion_vectors
                .insert(fragment_id, bitmap);
        }
    }
    Ok(Some(context))
}

async fn validate_v23_incoming_index_artifacts(
    dataset: &Dataset,
    transaction: &Transaction,
) -> Result<()> {
    if !dataset.manifest.uses_stable_logical_row_addresses() {
        return Ok(());
    }
    let Operation::CreateIndex {
        new_indices,
        removed_indices,
    } = &transaction.operation
    else {
        return Ok(());
    };
    let incoming = new_indices
        .iter()
        .filter(|index| {
            index.row_reference_domain
                == Some(lance_table::format::RowReferenceDomain::StableLogicalRowAddress)
        })
        .collect::<Vec<_>>();
    let current_indices = dataset.load_indices().await?;
    let mut removed_uuids = HashSet::with_capacity(removed_indices.len());
    for removed in removed_indices {
        if !removed_uuids.insert(removed.uuid) {
            return Err(Error::invalid_input(format!(
                "CreateIndex removes index segment {} more than once",
                removed.uuid
            )));
        }
        if current_indices
            .iter()
            .find(|current| current.uuid == removed.uuid)
            != Some(removed)
        {
            return Err(Error::invalid_input(format!(
                "CreateIndex removed segment {} does not match its exact current metadata",
                removed.uuid
            )));
        }
    }
    let mut incoming_uuids = HashSet::with_capacity(new_indices.len());
    for index in new_indices {
        if !incoming_uuids.insert(index.uuid) {
            return Err(Error::invalid_input(format!(
                "CreateIndex publishes index segment {} more than once",
                index.uuid
            )));
        }
        if current_indices
            .iter()
            .any(|current| current.uuid == index.uuid)
            && !removed_uuids.contains(&index.uuid)
        {
            return Err(Error::invalid_input(format!(
                "CreateIndex replaces current segment {} without removing its exact metadata",
                index.uuid
            )));
        }
    }
    let mut final_uuids = current_indices
        .iter()
        .filter(|index| {
            !removed_uuids.contains(&index.uuid) && !incoming_uuids.contains(&index.uuid)
        })
        .map(|index| index.uuid)
        .collect::<HashSet<_>>();
    for index in new_indices {
        if !final_uuids.insert(index.uuid) {
            return Err(Error::invalid_input(format!(
                "CreateIndex final catalog contains duplicate segment UUID {}",
                index.uuid
            )));
        }
    }
    if incoming.is_empty() {
        return Ok(());
    }

    let mut resolved_incoming = Vec::with_capacity(incoming.len());
    for replacement in incoming {
        let has_exclusions = replacement
            .logical_coverage
            .as_ref()
            .is_some_and(|coverage| {
                coverage
                    .shards
                    .iter()
                    .any(|shard| shard.excluded_selection.is_some())
            });
        if !has_exclusions {
            let mut resolved = replacement.clone();
            resolved.logical_coverage =
                Some(load_authoritative_logical_index_coverage(dataset, replacement).await?);
            resolved_incoming.push(resolved);
            continue;
        }

        let current = current_indices
            .iter()
            .find(|index| index.uuid == replacement.uuid)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "v2.3 index segment {} cannot add ownership exclusions without replacing a current segment",
                    replacement.uuid
                ))
            })?;
        let removed = removed_indices
            .iter()
            .find(|index| index.uuid == replacement.uuid)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "v2.3 index segment {} cannot add ownership exclusions without removing its exact current metadata",
                    replacement.uuid
                ))
            })?;
        if current != removed
            || !logical_index_is_monotonic_coverage_restriction(current, replacement)?
        {
            return Err(Error::invalid_input(format!(
                "v2.3 index segment {} ownership restriction does not replace its exact current metadata",
                replacement.uuid
            )));
        }

        let authoritative =
            resolve_logical_index_coverage_from_authoritative(dataset, replacement).await?;
        for (shard_index, shard) in authoritative.shards.iter().enumerate() {
            let Some(excluded) = shard.excluded_selection.as_ref() else {
                continue;
            };
            let raw_selection = shard.selection.as_ref().ok_or_else(|| {
                Error::internal(format!(
                    "v2.3 index segment {} authoritative shard {} remained external",
                    replacement.uuid, shard_index
                ))
            })?;
            if !excluded.is_subset_of(raw_selection)? {
                return Err(Error::invalid_input(format!(
                    "v2.3 index segment {} shard {} ownership exclusion is not a subset of its authoritative coverage",
                    replacement.uuid, shard_index
                )));
            }
        }
        let mut resolved = replacement.clone();
        resolved.logical_coverage = Some(authoritative);
        resolved_incoming.push(resolved);
    }

    let touched_names = resolved_incoming
        .iter()
        .map(|index| index.name.as_str())
        .collect::<HashSet<_>>();
    let retained = current_indices
        .iter()
        .filter(|index| {
            touched_names.contains(index.name.as_str())
                && !removed_uuids.contains(&index.uuid)
                && !incoming_uuids.contains(&index.uuid)
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut final_touched = resolve_logical_index_metadata(dataset, &retained).await?;
    final_touched.extend(resolved_incoming);
    validate_resolved_logical_index_overlaps(&final_touched)?;
    Ok(())
}

#[derive(Debug)]
struct RowAlignedRewriteTarget {
    current: Fragment,
    successor: Fragment,
    field_ids: Vec<i32>,
    rewritten_files: Vec<lance_table::format::DataFile>,
}

fn row_aligned_data_file_object_identity_eq(
    left: &lance_table::format::DataFile,
    right: &lance_table::format::DataFile,
) -> bool {
    left.path == right.path
        && left.file_major_version == right.file_major_version
        && left.file_minor_version == right.file_minor_version
        && left.base_id == right.base_id
}

fn rewritten_data_files(
    current: &Fragment,
    successor: &Fragment,
) -> Vec<lance_table::format::DataFile> {
    successor
        .files
        .iter()
        .filter(|successor_file| {
            !current.files.iter().any(|current_file| {
                row_aligned_data_file_object_identity_eq(current_file, successor_file)
            })
        })
        .cloned()
        .collect()
}

fn collect_row_aligned_rewrite_targets(
    dataset: &Dataset,
    transaction: &Transaction,
) -> Result<Vec<RowAlignedRewriteTarget>> {
    let current_by_id = dataset
        .manifest
        .fragments
        .iter()
        .map(|fragment| (fragment.id, fragment))
        .collect::<HashMap<_, _>>();
    match &transaction.operation {
        Operation::DataReplacement { replacements } => {
            let mut seen = HashSet::new();
            replacements
                .iter()
                .map(|replacement| {
                    if !seen.insert(replacement.0) {
                        return Err(Error::invalid_input(format!(
                            "DataReplacement contains duplicate fragment id {}",
                            replacement.0
                        )));
                    }
                    let current = current_by_id.get(&replacement.0).ok_or_else(|| {
                        Error::commit_conflict_source(
                            dataset.manifest.version,
                            format!(
                                "DataReplacement target fragment {} is absent from the current manifest",
                                replacement.0
                            )
                            .into(),
                        )
                    })?;
                    let successor =
                        data_replacement_successor_fragment(current, &replacement.1)?;
                    let mut field_ids = replacement
                        .1
                        .fields
                        .iter()
                        .copied()
                        .filter(|field_id| {
                            *field_id >= 0
                                && dataset.manifest.schema.field_by_id(*field_id).is_some()
                        })
                        .collect::<Vec<_>>();
                    field_ids.sort_unstable();
                    field_ids.dedup();
                    let rewritten_files = rewritten_data_files(current, &successor);
                    Ok(RowAlignedRewriteTarget {
                        current: (*current).clone(),
                        successor,
                        field_ids,
                        rewritten_files,
                    })
                })
                .collect()
        }
        Operation::Merge { fragments, .. } => {
            let mut seen = HashSet::new();
            let mut targets = Vec::new();
            for successor in fragments {
                if !seen.insert(successor.id) {
                    return Err(Error::invalid_input(format!(
                        "Merge contains duplicate fragment id {}",
                        successor.id
                    )));
                }
                let Some(current) = current_by_id.get(&successor.id) else {
                    continue;
                };
                let field_ids = merge_rewritten_existing_field_ids(
                    &dataset.manifest.schema,
                    current,
                    successor,
                );
                let is_physically_rewritten =
                    current.files.len() != successor.files.len()
                        || current.files.iter().zip(&successor.files).any(
                            |(current, successor)| {
                                !row_aligned_data_file_identity_eq(current, successor)
                            },
                        );
                if is_physically_rewritten {
                    let rewritten_files = rewritten_data_files(current, successor);
                    targets.push(RowAlignedRewriteTarget {
                        current: (*current).clone(),
                        successor: successor.clone(),
                        field_ids,
                        rewritten_files,
                    });
                }
            }
            Ok(targets)
        }
        _ => Ok(Vec::new()),
    }
}

fn v2_3_merge_direct_placements(
    dataset: &Dataset,
    transaction: &Transaction,
) -> Result<(Vec<Fragment>, Vec<RowAddressPlacementDelta>)> {
    let Operation::Merge { fragments, .. } = &transaction.operation else {
        return Ok((Vec::new(), Vec::new()));
    };
    let current_ids = dataset
        .manifest
        .fragments
        .iter()
        .map(|fragment| fragment.id)
        .collect::<HashSet<_>>();
    let new_fragments = fragments
        .iter()
        .filter(|fragment| !current_ids.contains(&fragment.id))
        .cloned()
        .collect::<Vec<_>>();
    let placements = transaction
        .row_address_layout_delta
        .as_ref()
        .map(|delta| delta.placements.clone())
        .unwrap_or_default();
    if new_fragments.is_empty() {
        if !placements.is_empty() {
            return Err(Error::invalid_input(
                "generic Merge has Direct provenance but no new physical fragment",
            ));
        }
        return Ok((new_fragments, placements));
    }
    let maximum_fragment_id = dataset.manifest.max_fragment_id();
    for fragment in &new_fragments {
        if maximum_fragment_id.is_some_and(|maximum| fragment.id <= maximum)
            || fragment.id >= u32::MAX as u64
            || fragment.deletion_file.is_some()
            || fragment.row_id_meta.is_some()
            || fragment.native_logical_domain.is_some()
        {
            return Err(Error::invalid_input(format!(
                "storage-version-2.3 generic Merge new fragment {} must advance the physical allocator and carry no preassigned row identity or deletion metadata",
                fragment.id
            )));
        }
    }
    let mut by_ordinal = vec![None; new_fragments.len()];
    for placement in &placements {
        if placement.placement_kind != RowAddressPlacementKind::Direct
            || !placement.source_selections.is_empty()
            || !placement.output_row_sequence_fingerprint.is_empty()
        {
            return Err(Error::invalid_input(
                "storage-version-2.3 generic Merge new fragments require source-free Direct provenance",
            ));
        }
        let RowAddressTargetFragment::NewFragmentOrdinal(ordinal) = placement.target.fragment
        else {
            return Err(Error::invalid_input(
                "generic Merge Direct provenance must target a new-fragment ordinal",
            ));
        };
        let fragment = new_fragments.get(ordinal as usize).ok_or_else(|| {
            Error::invalid_input(format!(
                "generic Merge Direct provenance references missing new-fragment ordinal {ordinal}"
            ))
        })?;
        let physical_rows = u32::try_from(fragment.physical_rows.ok_or_else(|| {
            Error::invalid_input(format!(
                "generic Merge new fragment {} is missing physical_rows",
                fragment.id
            ))
        })?)
        .map_err(|_| {
            Error::format_capacity_exceeded("generic Merge new fragment rows exceed u32")
        })?;
        if physical_rows == 0
            || placement.target.start_offset != 0
            || placement.target.end_offset != physical_rows
            || placement.output_cardinality != physical_rows as u64
            || by_ordinal[ordinal as usize].replace(()).is_some()
        {
            return Err(Error::invalid_input(format!(
                "generic Merge Direct provenance does not exactly cover new-fragment ordinal {ordinal}"
            )));
        }
    }
    if by_ordinal.iter().any(Option::is_none) {
        return Err(Error::invalid_input(
            "storage-version-2.3 generic Merge has a new fragment without Direct row-address provenance",
        ));
    }
    Ok((new_fragments, placements))
}

fn validate_row_aligned_rewrite_target(target: &RowAlignedRewriteTarget) -> Result<()> {
    if target.current.physical_rows.is_none()
        || target.current.physical_rows == Some(0)
        || target.successor.id != target.current.id
        || target.successor.physical_rows != target.current.physical_rows
        || target.successor.deletion_file != target.current.deletion_file
        || target.successor.row_id_meta != target.current.row_id_meta
        || target.successor.native_logical_domain != target.current.native_logical_domain
    {
        return Err(Error::invalid_input(format!(
            "storage-version-2.3 existing-field rewrite changed row alignment or ownership metadata for fragment {}",
            target.current.id
        )));
    }
    Ok(())
}

async fn validate_row_aligned_footer_files(
    dataset: Arc<Dataset>,
    successor: &Fragment,
    rewritten_files: &[lance_table::format::DataFile],
) -> Result<()> {
    FileFragment::new(dataset.clone(), successor.clone()).validate_data_file_metadata()?;
    if rewritten_files.is_empty() {
        return Ok(());
    }

    let mut footer_fragment = successor.clone();
    footer_fragment.files = rewritten_files.to_vec();
    // The source deletion vector is verified once while constructing the row-
    // aligned proof. Footer preflight must not reread it for every column file.
    footer_fragment.deletion_file = None;
    FileFragment::new(dataset, footer_fragment).validate().await
}

fn row_aligned_source_floor(
    layout: &lance_table::format::RowAddressLayout,
    field_id: i32,
) -> Result<u64> {
    layout
        .index_commit_floors
        .iter()
        .find(|floor| floor.field_id == field_id)
        .or_else(|| {
            layout
                .field_default_generations
                .iter()
                .find(|generation| generation.field_id == field_id)
        })
        .map(|generation| generation.generation)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "storage-version-2.3 field {field_id} is missing its source generation floor"
            ))
        })
}

async fn row_aligned_rewrite_proof(
    dataset: &Dataset,
    target: &RowAlignedRewriteTarget,
    field_change_index: usize,
    source_floor_indices: Vec<usize>,
) -> Result<(RowAddressFieldChange, RowAlignedRewriteProof)> {
    let fragment_id = u32::try_from(target.current.id).map_err(|_| {
        Error::invalid_input(format!(
            "storage-version-2.3 fragment id {} exceeds row-address capacity",
            target.current.id
        ))
    })?;
    let physical_rows = u32::try_from(target.current.physical_rows.ok_or_else(|| {
        Error::invalid_input(format!(
            "storage-version-2.3 fragment {} is missing physical_rows",
            target.current.id
        ))
    })?)
    .map_err(|_| {
        Error::format_capacity_exceeded(format!(
            "storage-version-2.3 fragment {} exceeds row-address slot capacity",
            target.current.id
        ))
    })?;
    validate_row_aligned_rewrite_target(target)?;

    let deleted_offsets = match target.current.deletion_file.as_ref() {
        Some(deletion_file) => RoaringBitmap::from(
            read_dataset_deletion_file(dataset, target.current.id, deletion_file)
                .await?
                .as_ref(),
        ),
        None => RoaringBitmap::new(),
    };
    if deleted_offsets
        .max()
        .is_some_and(|offset| offset >= physical_rows)
        || deleted_offsets.len() >= physical_rows as u64
    {
        return Err(Error::invalid_input(format!(
            "storage-version-2.3 row-aligned rewrite fragment {} has no valid live physical row set",
            target.current.id
        )));
    }
    let live_rows = physical_rows as u64 - deleted_offsets.len();
    let layout = dataset
        .manifest
        .row_address_layout
        .as_ref()
        .ok_or_else(|| {
            Error::invalid_input("storage-version-2.3 manifest is missing RowAddressLayout")
        })?;
    let compact_source = layout.row_aligned_rewrite_source(&target.current)?;
    let (selection, mapped_offsets_fingerprint) =
        if let Some((mapped, fingerprint)) = compact_source {
            if deleted_offsets.is_empty() {
                (mapped, fingerprint)
            } else {
                let mut deleted_logical = RoaringTreemap::new();
                let mut physical = Vec::with_capacity(4096);
                for offset in &deleted_offsets {
                    physical.push(RowAddress::new_from_parts(fragment_id, offset));
                    if physical.len() == physical.capacity() {
                        for logical in dataset
                            .resolve_physical_row_ids_async(&physical)
                            .await?
                            .into_iter()
                            .flatten()
                        {
                            deleted_logical.insert(logical.raw());
                        }
                        physical.clear();
                    }
                }
                if !physical.is_empty() {
                    for logical in dataset
                        .resolve_physical_row_ids_async(&physical)
                        .await?
                        .into_iter()
                        .flatten()
                    {
                        deleted_logical.insert(logical.raw());
                    }
                }
                let deleted = LogicalRowAddressSelection::from_bitmap(deleted_logical)?;
                (mapped.difference(&deleted)?, fingerprint)
            }
        } else {
            let summary = layout
                .physical_row_ownership
                .binary_search_by_key(&fragment_id, |summary| summary.physical_fragment_id)
                .ok()
                .map(|index| &layout.physical_row_ownership[index])
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "ExplicitMap fragment {fragment_id} is missing physical ownership metadata"
                    ))
                })?;
            let mut live_logical = RoaringTreemap::new();
            let mut physical = Vec::with_capacity(4096);
            for offset in 0..physical_rows {
                if deleted_offsets.contains(offset) {
                    continue;
                }
                physical.push(RowAddress::new_from_parts(fragment_id, offset));
                if physical.len() == physical.capacity() {
                    for logical in dataset.resolve_physical_row_ids_async(&physical).await? {
                        let logical = logical.ok_or_else(|| {
                            Error::invalid_input(format!(
                                "live physical row in fragment {} has no current logical owner",
                                target.current.id
                            ))
                        })?;
                        if !live_logical.insert(logical.raw()) {
                            return Err(Error::invalid_input(
                                "row-aligned rewrite source contains duplicate logical ownership",
                            ));
                        }
                    }
                    physical.clear();
                }
            }
            if !physical.is_empty() {
                for logical in dataset.resolve_physical_row_ids_async(&physical).await? {
                    let logical = logical.ok_or_else(|| {
                        Error::invalid_input(format!(
                            "live physical row in fragment {} has no current logical owner",
                            target.current.id
                        ))
                    })?;
                    if !live_logical.insert(logical.raw()) {
                        return Err(Error::invalid_input(
                            "row-aligned rewrite source contains duplicate logical ownership",
                        ));
                    }
                }
            }
            (
                LogicalRowAddressSelection::from_bitmap(live_logical)?,
                summary.mapped_offsets_fingerprint.clone(),
            )
        };
    if selection.cardinality() != live_rows {
        return Err(Error::invalid_input(
            "row-aligned rewrite logical ownership cardinality changed during preflight",
        ));
    }
    Ok((
        RowAddressFieldChange {
            selection,
            field_ids: target.field_ids.clone(),
        },
        RowAlignedRewriteProof {
            physical_fragment_id: fragment_id,
            physical_rows,
            mapped_offsets_fingerprint,
            deletion_offsets_fingerprint: target
                .current
                .deletion_file
                .as_ref()
                .map(|_| fingerprint_deleted_offsets(&deleted_offsets)),
            field_change_index,
            source_floor_indices,
        },
    ))
}

async fn enrich_v2_3_row_aligned_rewrite(
    dataset: &Dataset,
    transaction: &mut Transaction,
) -> Result<()> {
    if !dataset.manifest.uses_stable_logical_row_addresses()
        || !matches!(
            transaction.operation,
            Operation::DataReplacement { .. } | Operation::Merge { .. }
        )
    {
        return Ok(());
    }
    let (new_fragments, direct_placements) = v2_3_merge_direct_placements(dataset, transaction)?;
    let mut targets = collect_row_aligned_rewrite_targets(dataset, transaction)?;
    targets.sort_by_key(|target| target.current.id);
    let validation_schema = match &transaction.operation {
        Operation::Merge { schema, .. } => Some(schema.clone()),
        _ => None,
    };
    let mut validation_dataset = dataset.clone();
    if let Some(schema) = validation_schema {
        Arc::make_mut(&mut validation_dataset.manifest).schema = schema;
    }
    let validation_dataset = Arc::new(validation_dataset);
    for target in &targets {
        validate_row_aligned_rewrite_target(target)?;
        validate_row_aligned_footer_files(
            validation_dataset.clone(),
            &target.successor,
            &target.rewritten_files,
        )
        .await?;
    }
    for fragment in &new_fragments {
        validate_row_aligned_footer_files(validation_dataset.clone(), fragment, &fragment.files)
            .await?;
    }
    targets.retain(|target| !target.field_ids.is_empty());
    if targets.is_empty() {
        if transaction
            .row_address_layout_delta
            .as_ref()
            .is_some_and(|delta| !delta.row_aligned_rewrite_proofs.is_empty())
        {
            return Err(Error::invalid_input(
                "row-aligned rewrite proof does not correspond to an existing-field rewrite",
            ));
        }
        if direct_placements.is_empty() {
            return Ok(());
        }
    }

    let layout = dataset
        .manifest
        .row_address_layout
        .as_ref()
        .ok_or_else(|| {
            Error::invalid_input("storage-version-2.3 manifest is missing RowAddressLayout")
        })?;
    let field_ids = targets
        .iter()
        .flat_map(|target| target.field_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    let source_floors = field_ids
        .iter()
        .map(|field_id| {
            Ok(RowAddressSourceFloor {
                field_id: *field_id,
                generation: row_aligned_source_floor(layout, *field_id)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let source_floor_by_field = source_floors
        .iter()
        .enumerate()
        .map(|(index, floor)| (floor.field_id, index))
        .collect::<BTreeMap<_, _>>();
    let mut canonical = RowAddressLayoutDelta {
        placements: direct_placements,
        source_floors,
        expected_layout_fingerprint: layout.fingerprint.clone(),
        ..RowAddressLayoutDelta::default()
    };
    for (field_change_index, target) in targets.iter().enumerate() {
        let source_floor_indices = target
            .field_ids
            .iter()
            .map(|field_id| source_floor_by_field[field_id])
            .collect();
        let (field_change, proof) =
            row_aligned_rewrite_proof(dataset, target, field_change_index, source_floor_indices)
                .await?;
        canonical.field_changes.push(field_change);
        canonical.row_aligned_rewrite_proofs.push(proof);
    }
    canonical.validate_row_aligned_rewrite_proofs()?;

    if let Some(previous) = transaction.row_address_layout_delta.as_ref()
        && !previous.row_aligned_rewrite_proofs.is_empty()
        && (previous.row_aligned_rewrite_proofs != canonical.row_aligned_rewrite_proofs
            || previous.field_changes != canonical.field_changes)
    {
        return Err(Error::retryable_commit_conflict_source(
            dataset.manifest.version,
            "row-aligned rewrite source ownership or deletion state changed during commit retry"
                .into(),
        ));
    }
    transaction.row_address_layout_delta = Some(canonical);
    transaction.read_version = dataset.manifest.version;
    Ok(())
}

fn apply_clone_row_address_namespace(
    manifest: &mut Manifest,
    transaction: &Transaction,
) -> Result<()> {
    let is_v2_3 =
        manifest.data_storage_format.lance_file_version()?.resolve() == LanceFileVersion::V2_3;

    if !is_v2_3 {
        if transaction.row_address_layout_delta.is_some() {
            return Err(Error::invalid_input(
                "clone transactions for storage version 2.2 and earlier must not contain a row-address layout delta",
            ));
        }
        return Ok(());
    }

    let source_namespace = manifest
        .row_address_layout
        .as_ref()
        .ok_or_else(|| {
            Error::invalid_input(
                "storage-version-2.3 clone source is missing its row-address layout",
            )
        })?
        .namespace_uuid;
    let delta = transaction
        .row_address_layout_delta
        .as_ref()
        .ok_or_else(|| {
            Error::invalid_input(
                "storage-version-2.3 clone requires a transaction-persisted namespace UUID",
            )
        })?;
    let target_namespace = delta.create_namespace_uuid.ok_or_else(|| {
        Error::invalid_input("storage-version-2.3 clone delta is missing its create_namespace_uuid")
    })?;
    if target_namespace.is_nil() || target_namespace == source_namespace {
        return Err(Error::invalid_input(
            "storage-version-2.3 clone must create a non-nil namespace distinct from its source",
        ));
    }
    if !delta.source_domains.is_empty()
        || !delta.placements.is_empty()
        || !delta.retired_selections.is_empty()
        || !delta.field_changes.is_empty()
        || !delta.source_floors.is_empty()
        || !delta.expected_layout_fingerprint.is_empty()
        || !delta.replaced_generations.is_empty()
        || !delta.row_aligned_rewrite_proofs.is_empty()
        || !delta.explicit_map_placements.is_empty()
    {
        return Err(Error::invalid_input(
            "storage-version-2.3 clone delta may only contain create_namespace_uuid",
        ));
    }

    Arc::make_mut(manifest.row_address_layout.as_mut().unwrap()).namespace_uuid = target_namespace;
    manifest.refresh_row_address_fingerprint()?;
    manifest.validate_row_address_contract()?;
    Ok(())
}

/// Write a transaction to a file and return the relative path.
pub(crate) async fn write_transaction_file(
    object_store: &ObjectStore,
    base_path: &Path,
    transaction: &Transaction,
) -> Result<String> {
    let file_name = transaction_file_name(transaction);
    let path = base_path
        .clone()
        .join(TRANSACTIONS_DIR)
        .join(file_name.as_str());

    let message = pb::Transaction::from(transaction);
    check_protobuf_capacity(&message, "external transaction")?;
    let buf = message.encode_to_vec();
    object_store.put(&path, &buf).await?;

    Ok(file_name)
}

async fn reconcile_new_dataset_commit(
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    base_path: &Path,
    expected_version: u64,
    requested: &Transaction,
) -> Result<Option<(Manifest, ManifestLocation)>> {
    let location = match commit_handler
        .resolve_version_location(base_path, expected_version, &object_store.inner)
        .await
    {
        Ok(location) => location,
        Err(Error::DatasetNotFound { .. } | Error::NotFound { .. }) => return Ok(None),
        Err(error) => return Err(error),
    };
    let committed_manifest = Dataset::load_manifest(
        object_store,
        &location,
        base_path.to_string().as_str(),
        &Session::default(),
    )
    .await?;
    let Some(transaction_section) = committed_manifest.transaction_section else {
        return Ok(None);
    };
    let reader = object_store.open(&location.path).await?;
    let committed: pb::Transaction =
        lance_io::utils::read_message(reader.as_ref(), transaction_section).await?;
    let committed = Transaction::try_from(committed)?;
    if committed.uuid != requested.uuid {
        return Ok(None);
    }
    if !same_transaction_request(requested, &committed) {
        return Err(Error::invalid_input(format!(
            "transaction UUID {} was already used to create this dataset with a different request",
            requested.uuid
        )));
    }
    Ok(Some((committed_manifest, location)))
}

#[allow(clippy::too_many_arguments)]
async fn do_commit_new_dataset(
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    base_path: &Path,
    transaction: &Transaction,
    write_config: &ManifestWriteConfig,
    manifest_naming_scheme: ManifestNamingScheme,
    metadata_cache: &DSMetadataCache,
    store_registry: Arc<ObjectStoreRegistry>,
) -> Result<(Manifest, ManifestLocation)> {
    let transaction_file = if !write_config.disable_transaction_file() {
        transaction_file_name(transaction)
    } else {
        String::new()
    };

    let (mut manifest, mut indices) = if let Operation::Clone {
        is_shallow,
        ref_name,
        ref_version,
        ref_path,
        branch_name,
        ..
    } = &transaction.operation
    {
        let source_base_path =
            ObjectStore::extract_path_from_uri(store_registry, ref_path.as_str())?;
        let source_manifest_location = commit_handler
            .resolve_version_location(&source_base_path, *ref_version, &object_store.inner)
            .await?;
        let source_manifest = Dataset::load_manifest(
            object_store,
            &source_manifest_location,
            base_path.to_string().as_str(),
            &Session::default(),
        )
        .await?;
        source_manifest.validate_row_address_contract()?;

        if *is_shallow {
            let new_base_id = source_manifest
                .base_paths
                .keys()
                .max()
                .map(|id| *id + 1)
                .unwrap_or(0);
            let new_manifest = source_manifest.shallow_clone(
                ref_name.clone(),
                ref_path.clone(),
                new_base_id,
                branch_name.clone(),
                transaction_file.clone(),
            );

            let updated_indices = if let Some(index_section_pos) = source_manifest.index_section {
                let reader = object_store.open(&source_manifest_location.path).await?;
                let section: pb::IndexSection =
                    lance_io::utils::read_message(reader.as_ref(), index_section_pos).await?;
                section
                    .indices
                    .into_iter()
                    .map(|index_pb| {
                        let mut index = IndexMetadata::try_from(index_pb)?;
                        // Preserve an already-external index base across clone
                        // chains. Only index files local to the immediate
                        // source dataset need to be rebound to the new source
                        // base.
                        if index.base_id.is_none() {
                            index.base_id = Some(new_base_id);
                        }
                        Ok(index)
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                vec![]
            };
            (new_manifest, updated_indices)
        } else {
            // Deep clone: build a manifest that references local files (no external bases)
            let mut new_manifest = source_manifest.clone();
            new_manifest.base_paths.clear();
            new_manifest.branch = None;
            new_manifest.tag = None;
            new_manifest.index_section = None; // will be rewritten below
            new_manifest.transaction_file = Some(transaction_file.clone());
            new_manifest.transaction_section = None; // will be rewritten below
            let mut new_frags = new_manifest.fragments.as_ref().clone();
            for f in &mut new_frags {
                for df in &mut f.files {
                    df.base_id = None;
                }
                if let Some(d) = f.deletion_file.as_mut() {
                    d.base_id = None;
                }
            }
            new_manifest.fragments = Arc::new(new_frags);
            if let Some(layout) = new_manifest.row_address_layout.as_mut() {
                for placement in &mut Arc::make_mut(layout).placements {
                    if let lance_table::format::RowAddressPlacement::ExplicitMap(explicit) =
                        placement
                    {
                        explicit.base_id = None;
                    }
                }
            }

            // Indices: keep metadata but normalize base to local
            let mut updated_indices = Vec::new();
            if let Some(index_section_pos) = source_manifest.index_section {
                let reader = object_store.open(&source_manifest_location.path).await?;
                let section: pb::IndexSection =
                    lance_io::utils::read_message(reader.as_ref(), index_section_pos).await?;
                updated_indices = section
                    .indices
                    .into_iter()
                    .map(|index_pb| {
                        let mut index = IndexMetadata::try_from(index_pb)?;
                        index.base_id = None;
                        Ok(index)
                    })
                    .collect::<Result<Vec<_>>>()?;
            }
            (new_manifest, updated_indices)
        }
    } else {
        let (manifest, indices) =
            transaction.build_manifest(None, vec![], &transaction_file, write_config)?;
        (manifest, indices)
    };

    if matches!(&transaction.operation, Operation::Clone { .. }) {
        let source_namespace_uuid = manifest
            .row_address_layout
            .as_ref()
            .map(|layout| layout.namespace_uuid);
        let source_manifest_version = manifest.version;
        let clone_is_shallow = matches!(
            &transaction.operation,
            Operation::Clone {
                is_shallow: true,
                ..
            }
        );
        apply_clone_row_address_namespace(&mut manifest, transaction)?;
        if let Some(source_namespace_uuid) = source_namespace_uuid {
            let target_namespace_uuid = manifest
                .row_address_layout
                .as_ref()
                .ok_or_else(|| {
                    Error::internal("storage-version-2.3 clone lost its target row-address layout")
                })?
                .namespace_uuid;
            let transaction_uuid = Uuid::parse_str(&transaction.uuid).map_err(|error| {
                Error::invalid_input(format!(
                    "clone transaction UUID {} is invalid: {error}",
                    transaction.uuid
                ))
            })?;
            for index in &mut indices {
                if is_system_index(index) {
                    continue;
                }
                let files = index.files.as_ref().ok_or_else(|| {
                    Error::invalid_input(format!(
                        "storage-version-2.3 cloned index segment {} is missing files",
                        index.uuid
                    ))
                })?;
                let coverage = index.logical_coverage.as_mut().ok_or_else(|| {
                    Error::invalid_input(format!(
                        "storage-version-2.3 cloned index segment {} is missing logical coverage",
                        index.uuid
                    ))
                })?;
                coverage.rebind_for_clone(
                    source_namespace_uuid,
                    target_namespace_uuid,
                    index.uuid,
                    files,
                    transaction_uuid,
                    clone_is_shallow,
                    source_manifest_version,
                )?;
            }
        }
        transaction.finalize_row_address_metadata_debt(
            None,
            &mut manifest,
            &indices,
            write_config,
            false,
        )?;
    }

    validate_manifest_write_capacity(&manifest, &indices, transaction)?;
    if !write_config.disable_transaction_file() {
        let written = write_transaction_file(object_store, base_path, transaction).await?;
        debug_assert_eq!(written, transaction_file);
    }

    let result = write_manifest_file(
        object_store,
        commit_handler,
        base_path,
        &mut manifest,
        if indices.is_empty() {
            None
        } else {
            Some(indices.clone())
        },
        write_config,
        manifest_naming_scheme,
        Some(transaction),
    )
    .await;

    // TODO: Allow Append or Overwrite mode to retry using `commit_transaction`
    // if there is a conflict.
    match result {
        Ok(manifest_location) => {
            let tx_key = crate::session::caches::TransactionKey {
                version: manifest.version,
            };
            metadata_cache
                .insert_with_key(&tx_key, Arc::new(transaction.clone()))
                .await;

            let manifest_key = crate::session::caches::ManifestKey {
                version: manifest_location.version,
                e_tag: manifest_location.e_tag.as_deref(),
            };
            metadata_cache
                .insert_with_key(&manifest_key, Arc::new(manifest.clone()))
                .await;
            Ok((manifest, manifest_location))
        }
        Err(CommitError::CommitConflict) => {
            match reconcile_new_dataset_commit(
                object_store,
                commit_handler,
                base_path,
                manifest.version,
                transaction,
            )
            .await
            {
                Ok(Some(committed)) => Ok(committed),
                Ok(None) => {
                    cleanup_transaction_file(object_store, base_path, &transaction_file).await;
                    Err(crate::Error::dataset_already_exists(base_path.to_string()))
                }
                Err(error) => Err(error),
            }
        }
        Err(CommitError::OtherError(err)) => {
            match reconcile_new_dataset_commit(
                object_store,
                commit_handler,
                base_path,
                manifest.version,
                transaction,
            )
            .await
            {
                Ok(Some(committed)) => Ok(committed),
                Ok(None) => {
                    cleanup_transaction_file(object_store, base_path, &transaction_file).await;
                    Err(err)
                }
                Err(error) => Err(error),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn commit_new_dataset(
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    base_path: &Path,
    transaction: &Transaction,
    write_config: &ManifestWriteConfig,
    manifest_naming_scheme: ManifestNamingScheme,
    metadata_cache: &crate::session::caches::DSMetadataCache,
    store_registry: Arc<ObjectStoreRegistry>,
) -> Result<(Manifest, ManifestLocation)> {
    do_commit_new_dataset(
        object_store,
        commit_handler,
        base_path,
        transaction,
        write_config,
        manifest_naming_scheme,
        metadata_cache,
        store_registry,
    )
    .await
}

/// Internal function to check if a manifest could use some migration.
///
/// Manifest migrations happen on each write, but sometimes we need to run them
/// before certain new operations. An easy way to force a migration is to run
/// `dataset.delete(false)`, which won't modify data but will cause a migration.
/// However, you don't want to always have to do this, so we provide this method
/// to check if a migration is needed.
pub fn manifest_needs_migration(manifest: &Manifest, indices: &[IndexMetadata]) -> bool {
    manifest.writer_version.is_none()
        || manifest.fragments.iter().any(|f| {
            f.physical_rows.is_none()
                || (f
                    .deletion_file
                    .as_ref()
                    .map(|d| d.num_deleted_rows.is_none())
                    .unwrap_or(false))
        })
        || indices
            .iter()
            .any(|i| must_recalculate_fragment_bitmap(i, manifest.writer_version.as_ref()))
}

/// Update manifest with new metadata fields.
///
/// Fields such as `physical_rows` and `num_deleted_rows` may not have been
/// in older datasets. To bring these old manifests up-to-date, we add them here.
async fn migrate_manifest(
    dataset: &Dataset,
    manifest: &mut Manifest,
    recompute_stats: bool,
) -> Result<()> {
    if !recompute_stats
        && manifest.fragments.iter().all(|f| {
            f.num_rows().map(|n| n > 0).unwrap_or(false)
                && f.files.iter().all(|f| f.file_size_bytes.get().is_some())
        })
    {
        return Ok(());
    }

    manifest.fragments =
        Arc::new(migrate_fragments(dataset, &manifest.fragments, recompute_stats).await?);

    Ok(())
}

fn check_storage_version(manifest: &mut Manifest) -> Result<()> {
    let data_storage_version = manifest.data_storage_format.lance_file_version()?;
    if manifest.data_storage_format.lance_file_version()? == LanceFileVersion::Legacy {
        // Due to bugs in 0.16 it is possible the dataset's data storage version does not
        // match the file version.  As a result, we need to check and see if they are out
        // of sync.
        if let Some(actual_file_version) =
            Fragment::try_infer_version(&manifest.fragments).map_err(|e| Error::internal(format!(
                "The dataset contains a mixture of file versions.  You will need to rollback to an earlier version: {}",
                e
            )))?
                && actual_file_version > data_storage_version {
                    log::warn!(
                        "Data storage version {} is less than the actual file version {}.  This has been automatically updated.",
                        data_storage_version,
                        actual_file_version
                    );
                    manifest.data_storage_format = DataStorageFormat::new(actual_file_version);
                }
    } else {
        // Otherwise, if we are on 2.0 or greater, we should ensure that the file versions
        // match the data storage version.  This is a sanity assertion to prevent data corruption.
        if let Some(actual_file_version) = Fragment::try_infer_version(&manifest.fragments)?
            && actual_file_version != data_storage_version
        {
            return Err(Error::internal(format!(
                "The operation added files with version {}.  However, the data storage version is {}.",
                actual_file_version, data_storage_version
            )));
        }
    }
    Ok(())
}

fn check_column_indices(manifest: &Manifest) -> Result<()> {
    let data_storage_version = manifest.data_storage_format.lance_file_version()?;
    if data_storage_version < LanceFileVersion::V2_1 {
        return Ok(());
    }

    for fragment in manifest.fragments.iter() {
        for data_file in &fragment.files {
            if data_file.is_legacy_file() || data_file.column_indices.is_empty() {
                continue;
            }
            if data_file.fields.len() != data_file.column_indices.len() {
                return Err(Error::invalid_input(format!(
                    "Data file '{}' (fragment {}) has {} field ids but {} column indices. \
                     These must be the same length.",
                    data_file.path,
                    fragment.id,
                    data_file.fields.len(),
                    data_file.column_indices.len()
                )));
            }
            let file_version = LanceFileVersion::try_from_major_minor(
                data_file.file_major_version,
                data_file.file_minor_version,
            )?;
            if file_version < LanceFileVersion::V2_1 {
                continue;
            }
            for (field_id, column_index) in
                data_file.fields.iter().zip(data_file.column_indices.iter())
            {
                // Field ids may not exist in the current schema after schema
                // evolution (e.g. cast/drop column). Skip those.
                let Some(field) = manifest.schema.field_by_id(*field_id) else {
                    continue;
                };
                let needs_column = field.is_leaf() || field.is_packed_struct() || field.is_blob();
                if needs_column && *column_index == -1 {
                    return Err(Error::invalid_input(format!(
                        "Field '{}' (id={}) in data file '{}' (fragment {}) \
                         has column_index=-1, but leaf fields, packed structs, \
                         and blob fields must have a valid column index in \
                         file format 2.1+.",
                        field.name, field_id, data_file.path, fragment.id
                    )));
                }
                if !needs_column && *column_index != -1 {
                    return Err(Error::invalid_input(format!(
                        "Non-leaf field '{}' (id={}) in data file '{}' (fragment {}) \
                         has column_index={}, but non-leaf fields should have \
                         column_index=-1 in file format 2.1+. Only leaf fields, \
                         packed structs, and blob fields should have column indices.",
                        field.name, field_id, data_file.path, fragment.id, column_index
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Fix schema in case of duplicate field ids.
///
/// See test dataset v0.10.5/corrupt_schema
fn fix_schema(manifest: &mut Manifest) -> Result<()> {
    // We can short-circuit if there is only one file per fragment or no fragments.
    if manifest.fragments.iter().all(|f| f.files.len() <= 1) {
        return Ok(());
    }

    // First, see which, if any fields have duplicate ids, within any fragment.
    let mut fields_with_duplicate_ids = HashSet::new();
    let mut seen_fields = HashSet::new();
    for fragment in manifest.fragments.iter() {
        for file in fragment.files.iter() {
            for field_id in file.fields.iter() {
                if *field_id >= 0 && !seen_fields.insert(*field_id) {
                    fields_with_duplicate_ids.insert(*field_id);
                }
            }
        }
        seen_fields.clear();
    }
    if fields_with_duplicate_ids.is_empty() {
        return Ok(());
    }

    // Now, we need to remap the field ids to be unique.
    let mut old_field_id_mapping: HashMap<i32, i32> = HashMap::new();
    let mut fields_with_duplicate_ids = fields_with_duplicate_ids.into_iter().collect::<Vec<_>>();
    fields_with_duplicate_ids.sort_unstable();
    for (field_id_seed, field_id) in (manifest.max_field_id() + 1..).zip(fields_with_duplicate_ids)
    {
        old_field_id_mapping.insert(field_id, field_id_seed);
    }

    let mut fragments = manifest.fragments.as_ref().clone();

    // Apply mapping to fragment files list
    // We iterate over files in reverse order so that we only map the last field id
    seen_fields.clear();
    for fragment in fragments.iter_mut() {
        for file in fragment.files.iter_mut().rev() {
            let new_fields: Arc<[i32]> = file
                .fields
                .iter()
                .map(|field_id| {
                    if let Some(new_field_id) = old_field_id_mapping.get(field_id)
                        && seen_fields.insert(*field_id)
                    {
                        *new_field_id
                    } else {
                        *field_id
                    }
                })
                .collect::<Vec<_>>()
                .into();
            file.fields = new_fields;
        }
        seen_fields.clear();
    }

    // Apply mapping to the schema
    for (old_field_id, new_field_id) in &old_field_id_mapping {
        let field = manifest.schema.mut_field_by_id(*old_field_id).unwrap();
        field.id = *new_field_id;
    }

    // Drop data files that are no longer in use.
    let remaining_field_ids = manifest
        .schema
        .fields_pre_order()
        .map(|f| f.id)
        .collect::<HashSet<_>>();
    for fragment in fragments.iter_mut() {
        fragment.files.retain(|file| {
            file.fields
                .iter()
                .any(|field_id| remaining_field_ids.contains(field_id))
        });
    }

    manifest.fragments = Arc::new(fragments);

    Ok(())
}

/// Get updated vector of fragments that has `physical_rows` and `num_deleted_rows`
/// filled in. This is no-op for newer tables, but may do IO for tables written
/// with older versions of Lance.
pub(crate) async fn migrate_fragments(
    dataset: &Dataset,
    fragments: &[Fragment],
    recompute_stats: bool,
) -> Result<Vec<Fragment>> {
    let dataset = Arc::new(dataset.clone());
    let new_fragments = futures::stream::iter(fragments)
        .map(|fragment| async {
            let physical_rows = if recompute_stats {
                None
            } else {
                fragment.physical_rows
            };
            let physical_rows = if let Some(physical_rows) = physical_rows {
                Either::Right(futures::future::ready(Ok(physical_rows)))
            } else {
                let file_fragment = FileFragment::new(dataset.clone(), fragment.clone());
                Either::Left(async move { file_fragment.physical_rows().await })
            };
            let num_deleted_rows = match &fragment.deletion_file {
                None => Either::Left(futures::future::ready(Ok(None))),
                Some(DeletionFile {
                    num_deleted_rows: Some(deleted_rows),
                    ..
                }) if !recompute_stats => {
                    Either::Left(futures::future::ready(Ok(Some(*deleted_rows))))
                }
                Some(deletion_file) => Either::Right(async {
                    let deletion_vector =
                        read_dataset_deletion_file(dataset.as_ref(), fragment.id, deletion_file)
                            .await?;
                    Ok(Some(deletion_vector.len()))
                }),
            };

            let (physical_rows, num_deleted_rows) =
                futures::future::try_join(physical_rows, num_deleted_rows).await?;

            let mut data_files = fragment.files.clone();

            // For each of the data files in the fragment, we need to get the file size.
            // Resolve each file against its own storage base: multi-base datasets
            // keep data files outside the dataset root (DataFile.base_id).
            let get_sizes = data_files
                .iter()
                .map(|file| {
                    if let Some(size) = file.file_size_bytes.get() {
                        Either::Left(futures::future::ready(Ok(size)))
                    } else {
                        let dataset = dataset.clone();
                        Either::Right(async move {
                            let object_store = dataset.object_store_for_data_file(file).await?;
                            let data_dir = dataset.data_file_dir_for_base(file.base_id)?;
                            object_store
                                .size(&data_dir.join(file.path.clone()))
                                .map_ok(|size| {
                                    NonZero::new(size).ok_or_else(|| {
                                        Error::internal(format!("File {} has size 0", file.path))
                                    })
                                })
                                .await?
                        })
                    }
                })
                .collect::<Vec<_>>();
            let sizes = futures::future::try_join_all(get_sizes).await?;
            data_files.iter_mut().zip(sizes).for_each(|(file, size)| {
                file.file_size_bytes = CachedFileSize::new(size.into());
            });

            let deletion_file = fragment
                .deletion_file
                .as_ref()
                .map(|deletion_file| DeletionFile {
                    num_deleted_rows,
                    ..deletion_file.clone()
                });

            Ok::<_, Error>(Fragment {
                physical_rows: Some(physical_rows),
                deletion_file,
                files: data_files,
                ..fragment.clone()
            })
        })
        .buffered(dataset.object_store.io_parallelism())
        // Filter out empty fragments
        .try_filter(|frag| futures::future::ready(frag.num_rows().map(|n| n > 0).unwrap_or(false)))
        .boxed();

    new_fragments.try_collect().await
}

fn must_recalculate_fragment_bitmap(
    index: &IndexMetadata,
    version: Option<&WriterVersion>,
) -> bool {
    if index.fragment_bitmap.is_none() {
        return true;
    }
    // If the fragment bitmap was written by an old version of lance then we need to recalculate
    // it because it could be corrupt due to a bug in versions < 0.8.15
    if let Some(version) = version {
        if version.library != "lance" {
            // We assume a different library is not affected by the bug.
            return false;
        }

        let cutoff = semver::Version::new(0, 8, 15);
        version
            .lance_lib_version()
            .map(|lance_lib_version| lance_lib_version < cutoff)
            .unwrap_or(true)
    } else {
        // Older versions of Lance library didn't record writer version at all.
        true
    }
}

/// Update indices with new fields.
///
/// Indices might be missing `fragment_bitmap`, so this function will add it.
/// Indices might also be missing `files` (file sizes), so this function will collect them.
async fn migrate_indices(dataset: &Dataset, indices: &mut [IndexMetadata]) -> Result<()> {
    infer_missing_vector_details(dataset, indices).await;
    if dataset.manifest.uses_stable_logical_row_addresses() {
        // v2.3 index coverage is a logical selection. Treating an intentionally
        // absent physical fragment_bitmap as legacy metadata would try to open
        // a newly-built index through the old manifest before it is committed.
        validate_index_contract(dataset, indices)?;
        return Ok(());
    }
    let needs_recalculating = match detect_overlapping_fragments(indices) {
        Ok(()) => vec![],
        Err(BadFragmentBitmapError { bad_indices }) => {
            bad_indices.into_iter().map(|(name, _)| name).collect()
        }
    };
    for index in indices.iter_mut() {
        if needs_recalculating.contains(&index.name)
            || must_recalculate_fragment_bitmap(index, dataset.manifest.writer_version.as_ref())
                && !is_system_index(index)
        {
            debug_assert_eq!(index.fields.len(), 1);
            let idx_field = dataset.schema().field_by_id(index.fields[0]).ok_or_else(|| Error::internal(format!("Index with uuid {} referred to field with id {} which did not exist in dataset", index.uuid, index.fields[0])))?;
            // We need to calculate the fragments covered by the index
            let idx = dataset
                .open_generic_index(&idx_field.name, &index.uuid, &NoOpMetricsCollector)
                .await?;
            index.fragment_bitmap = Some(idx.calculate_included_frags().await?);
        }
        // We can't reliably recalculate the index type for label_list and bitmap indices and so we can't migrate this field.
        // However, we still log for visibility and to help potentially diagnose issues in the future if we grow to rely on the field.
        if index.index_details.is_none() {
            log::debug!(
                "the index with uuid {} is missing index metadata.  This probably means it was written with Lance version <= 0.19.2.  This is not a problem.",
                index.uuid
            );
        }

        // Migrate file sizes for indices that don't have them.
        // Use indice_files_dir to handle shallow-cloned indices with base_id.
        if index.files.is_none() && !is_system_index(index) {
            let result = async {
                let index_dir = dataset
                    .indice_files_dir(index)?
                    .join(index.uuid.to_string());
                let object_store = dataset.object_store_for_index(index).await?;
                list_index_files_with_sizes(&object_store, &index_dir).await
            }
            .await;
            match result {
                Ok(files) => {
                    log::debug!(
                        "Migrated file sizes for index {} (uuid: {}): {} files",
                        index.name,
                        index.uuid,
                        files.len()
                    );
                    index.files = Some(files);
                }
                Err(e) => {
                    // Log but don't fail - file sizes are optional
                    log::debug!(
                        "Could not collect file sizes for index {} (uuid: {}): {}",
                        index.name,
                        index.uuid,
                        e
                    );
                }
            }
        }
    }

    Ok(())
}

pub(crate) struct BadFragmentBitmapError {
    pub bad_indices: Vec<(String, Vec<u32>)>,
}

/// Detect whether a given index has overlapping fragment bitmaps in its index
/// segments.
pub(crate) fn detect_overlapping_fragments(
    indices: &[IndexMetadata],
) -> std::result::Result<(), BadFragmentBitmapError> {
    let index_names: HashSet<&str> = indices.iter().map(|i| i.name.as_str()).collect();
    let mut bad_indices = Vec::new(); // (index_name, overlapping_fragments)
    for name in index_names {
        let mut seen_fragment_ids = HashSet::new();
        let mut overlap = Vec::new();
        for index in indices.iter().filter(|i| i.name == name) {
            if let Some(fragment_bitmap) = index.fragment_bitmap.as_ref() {
                for fragment in fragment_bitmap {
                    if !seen_fragment_ids.insert(fragment) {
                        overlap.push(fragment);
                    }
                }
            }
        }
        if !overlap.is_empty() {
            bad_indices.push((name.to_string(), overlap));
        }
    }
    if bad_indices.is_empty() {
        Ok(())
    } else {
        Err(BadFragmentBitmapError { bad_indices })
    }
}

pub(crate) async fn do_commit_detached_transaction(
    dataset: &Dataset,
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    transaction: &Transaction,
    write_config: &ManifestWriteConfig,
    commit_config: &CommitConfig,
) -> Result<(Manifest, ManifestLocation)> {
    let mut transaction = transaction.clone();
    transaction.refresh_original_request_fingerprint();
    enrich_v2_3_row_aligned_rewrite(dataset, &mut transaction).await?;
    validate_v23_incoming_index_artifacts(dataset, &transaction).await?;
    let transaction_file = if !write_config.disable_transaction_file() {
        transaction_file_name(&transaction)
    } else {
        String::new()
    };
    let mut transaction_file_written = false;

    // We still do a loop since we may have conflicts in the random version we pick
    let mut backoff = Backoff::default();
    while backoff.attempt() < commit_config.num_retries {
        // Pick a random u64 with the highest bit set to indicate it is detached
        let random_version = rng().random::<u64>() | DETACHED_VERSION_MASK;

        let row_address_context =
            prepare_row_address_manifest_context(dataset, &transaction).await?;
        let (mut manifest, mut indices) = match transaction.operation {
            Operation::Restore { version } => {
                Transaction::restore_old_manifest(
                    object_store,
                    commit_handler,
                    &dataset.base,
                    version,
                    write_config,
                    &transaction_file,
                    &dataset.manifest,
                )
                .await?
            }
            _ => transaction.build_manifest_with_row_address_context(
                Some(dataset.manifest.as_ref()),
                dataset.load_indices().await?.as_ref().clone(),
                &transaction_file,
                write_config,
                row_address_context.as_ref(),
            )?,
        };

        manifest.version = random_version;

        // recompute_stats is always false so far because detached manifests are newer than
        // the old stats bug.
        migrate_manifest(dataset, &mut manifest, /*recompute_stats=*/ false).await?;
        // fix_schema and check_storage_version are just for sanity-checking and consistency
        fix_schema(&mut manifest)?;
        check_storage_version(&mut manifest)?;
        check_column_indices(&manifest)?;
        migrate_indices(dataset, &mut indices).await?;

        if matches!(&transaction.operation, Operation::Restore { .. }) {
            transaction.finalize_row_address_metadata_debt(
                Some(dataset.manifest.as_ref()),
                &mut manifest,
                &indices,
                write_config,
                false,
            )?;
        }

        validate_manifest_write_capacity(&manifest, &indices, &transaction)?;
        if !write_config.disable_transaction_file() && !transaction_file_written {
            let written = write_transaction_file(object_store, &dataset.base, &transaction).await?;
            debug_assert_eq!(written, transaction_file);
            transaction_file_written = true;
        }

        // Try to commit the manifest
        let result = write_manifest_file(
            object_store,
            commit_handler,
            &dataset.base,
            &mut manifest,
            if indices.is_empty() {
                None
            } else {
                Some(indices.clone())
            },
            write_config,
            ManifestNamingScheme::V2,
            Some(&transaction),
        )
        .await;

        match result {
            Ok(location) => {
                return Ok((manifest, location));
            }
            Err(CommitError::CommitConflict) => {
                // We pick a random u64 for the version, so it's possible (though extremely unlikely)
                // that we have a conflict. In that case, we just try again.
                tokio::time::sleep(backoff.next_backoff()).await;
            }
            Err(CommitError::OtherError(err)) => {
                // The detached manifest may have committed even though its response was lost.
                // There is no authoritative reconciliation for a random detached version, so
                // retain the transaction file for readers and version-aware GC.
                if transaction_file_written {
                    log::warn!(
                        "Detached manifest commit failed ambiguously; retaining transaction file '{}'",
                        transaction_file
                    );
                }
                return Err(err);
            }
        }
    }

    // This should be extremely unlikely.  There should not be *that* many detached commits.  If
    // this happens then it seems more likely there is a bug in our random u64 generation.
    // A concurrent submitter can share this UUID-derived transaction-file key.
    // Conflicts do not prove the object is unreachable; version-aware GC does.
    Err(crate::Error::commit_conflict_source(
        0,
        format!(
            "Failed find unused random u64 after {} retries.",
            commit_config.num_retries
        )
        .into(),
    ))
}

pub(crate) async fn commit_detached_transaction(
    dataset: &Dataset,
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    transaction: &Transaction,
    write_config: &ManifestWriteConfig,
    commit_config: &CommitConfig,
) -> Result<(Manifest, ManifestLocation)> {
    do_commit_detached_transaction(
        dataset,
        object_store,
        commit_handler,
        transaction,
        write_config,
        commit_config,
    )
    .await
}

/// Load new transactions and sort them by version in ascending order (oldest to newest)
async fn load_and_sort_new_transactions(
    dataset: &Dataset,
) -> Result<(Dataset, Vec<(u64, Arc<Transaction>)>)> {
    let NewTransactionResult {
        dataset: new_ds,
        new_transactions,
    } = load_new_transactions(dataset);
    let new_transactions = new_transactions.try_collect::<Vec<_>>();
    let (new_ds, mut txns) = futures::future::try_join(new_ds, new_transactions).await?;
    txns.sort_by_key(|(version, _)| *version);
    Ok((new_ds, txns))
}

fn source_free_direct_enrichment_matches(
    requested: &Transaction,
    committed_delta: &RowAddressLayoutDelta,
) -> bool {
    let (Operation::Append { fragments } | Operation::Overwrite { fragments, .. }) =
        &requested.operation
    else {
        return false;
    };
    if requested.row_address_layout_delta.is_some()
        || !committed_delta.source_domains.is_empty()
        || !committed_delta.retired_selections.is_empty()
        || !committed_delta.field_changes.is_empty()
        || !committed_delta.source_floors.is_empty()
        || !committed_delta.replaced_generations.is_empty()
        || !committed_delta.row_aligned_rewrite_proofs.is_empty()
        || !committed_delta.explicit_map_placements.is_empty()
        || committed_delta.placements.len() != fragments.len()
    {
        return false;
    }
    let create_metadata_is_valid = if matches!(requested.operation, Operation::Overwrite { .. })
        && requested.read_version == 0
    {
        committed_delta.create_namespace_uuid.is_some()
            == committed_delta.expected_layout_fingerprint.is_empty()
    } else {
        committed_delta.create_namespace_uuid.is_none()
            && !committed_delta.expected_layout_fingerprint.is_empty()
    };
    create_metadata_is_valid
        && fragments
            .iter()
            .zip(&committed_delta.placements)
            .enumerate()
            .all(|(ordinal, (fragment, placement))| {
                let Some(physical_rows) = fragment
                    .physical_rows
                    .and_then(|rows| u32::try_from(rows).ok())
                else {
                    return false;
                };
                let Ok(ordinal) = u32::try_from(ordinal) else {
                    return false;
                };
                placement.placement_kind == RowAddressPlacementKind::Direct
                    && placement.source_selections.is_empty()
                    && placement.output_row_sequence_fingerprint.is_empty()
                    && placement.target.fragment
                        == RowAddressTargetFragment::NewFragmentOrdinal(ordinal)
                    && placement.target.start_offset == 0
                    && placement.target.end_offset == physical_rows
                    && placement.output_cardinality == u64::from(physical_rows)
            })
}

pub(crate) fn same_transaction_request(requested: &Transaction, committed: &Transaction) -> bool {
    if let Some(committed_fingerprint) = committed.original_request_fingerprint.as_deref() {
        return requested
            .canonical_original_request_fingerprint()
            .is_some_and(|requested_fingerprint| {
                requested_fingerprint.as_slice() == committed_fingerprint
            });
    }
    let same_operation = match (&requested.operation, &committed.operation) {
        (
            Operation::Append {
                fragments: requested,
            },
            Operation::Append {
                fragments: committed,
            },
        ) => requested == committed,
        (requested, committed) => requested == committed,
    };
    let allows_core_row_aligned_enrichment = matches!(
        requested.operation,
        Operation::DataReplacement { .. } | Operation::Merge { .. }
    ) && committed
        .row_address_layout_delta
        .as_ref()
        .is_some_and(|committed_delta| {
            !committed_delta.row_aligned_rewrite_proofs.is_empty()
                && requested.row_address_layout_delta.as_ref().map_or_else(
                    || committed_delta.placements.is_empty(),
                    |requested_delta| {
                        requested_delta.row_aligned_rewrite_proofs.is_empty()
                            && requested_delta.source_domains.is_empty()
                            && requested_delta.retired_selections.is_empty()
                            && requested_delta.field_changes.is_empty()
                            && requested_delta.source_floors.is_empty()
                            && requested_delta.replaced_generations.is_empty()
                            && requested_delta.create_namespace_uuid.is_none()
                            && requested_delta.explicit_map_placements.is_empty()
                            && requested_delta.placements == committed_delta.placements
                    },
                )
        });
    let allows_core_direct_enrichment = committed
        .row_address_layout_delta
        .as_ref()
        .is_some_and(|delta| source_free_direct_enrichment_matches(requested, delta));
    (requested.read_version == committed.read_version || allows_core_row_aligned_enrichment)
        && requested.tag == committed.tag
        && requested.transaction_properties == committed.transaction_properties
        && (requested.row_address_layout_delta == committed.row_address_layout_delta
            || allows_core_row_aligned_enrichment
            || allows_core_direct_enrichment)
        && same_operation
}

fn committed_transaction_version(
    requested: &Transaction,
    transactions: &[(u64, Arc<Transaction>)],
) -> Result<Option<u64>> {
    let mut matches = transactions
        .iter()
        .filter(|(_, committed)| committed.uuid == requested.uuid);
    let Some((version, committed)) = matches.next() else {
        return Ok(None);
    };
    if matches.next().is_some() {
        return Err(Error::internal(format!(
            "transaction UUID {} was committed at multiple versions",
            requested.uuid
        )));
    }
    if !same_transaction_request(requested, committed) {
        return Err(Error::invalid_input(format!(
            "transaction UUID {} was already committed at version {} with a different request",
            requested.uuid, version
        )));
    }
    Ok(Some(*version))
}

async fn committed_transaction_result(
    dataset: &Dataset,
    version: u64,
) -> Result<(Manifest, ManifestLocation)> {
    let committed = dataset.checkout_version(version).await?;
    Ok((
        committed.manifest.as_ref().clone(),
        committed.manifest_location,
    ))
}

/// Attempt to commit a transaction, with retries and conflict resolution.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn commit_transaction(
    dataset: &Dataset,
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    transaction: &Transaction,
    write_config: &ManifestWriteConfig,
    commit_config: &CommitConfig,
    manifest_naming_scheme: ManifestNamingScheme,
    affected_rows: Option<&RowAddrTreeMap>,
) -> Result<(Manifest, ManifestLocation)> {
    // Note: object_store has been configured with WriteParams, but dataset.object_store.as_ref()
    // has not necessarily. So for anything involving writing, use `object_store`.
    let read_version = transaction.read_version;
    let mut target_version = read_version + 1;
    let original_dataset = dataset.clone();

    // read_version sometimes defaults to zero for overwrite.
    // If num_retries is zero, we are in "strict overwrite" mode.
    // Strict overwrites are not subject to any sort of automatic conflict resolution.
    let strict_overwrite = matches!(transaction.operation, Operation::Overwrite { .. })
        && commit_config.num_retries == 0;
    let mut dataset =
        if dataset.manifest.version != read_version && (read_version != 0 || strict_overwrite) {
            // If the dataset version is not the same as the read version, we need to
            // checkout the read version.
            dataset.checkout_version(read_version).await?
        } else {
            // If the dataset version is the same as the read version, we can use it directly.
            dataset.clone()
        };

    let mut requested_transaction = transaction.clone();
    requested_transaction.refresh_original_request_fingerprint();
    let mut transaction = requested_transaction.clone();
    let transaction_base_dataset = dataset.clone();

    let num_attempts = std::cmp::max(commit_config.num_retries, 1);
    let mut backoff = SlotBackoff::default();
    let start = Instant::now();

    // Other transactions that may have been committed since the read_version.
    // We keep pair of (version, transaction). No other transactions to check initially
    let mut other_transactions: Vec<(u64, Arc<Transaction>)>;
    while backoff.attempt() < num_attempts {
        // We are pessimistic here and assume there may be other transactions
        // we need to check for. We could be optimistic here and blindly
        // attempt to commit, giving faster performance for sequence writes and
        // slower performance for concurrent writes. But that makes the fast path
        // faster and the slow path slower, which makes performance less predictable
        // for users. So we always check for other transactions.
        // We skip this for strict overwrites, because strict overwrites can't be rebased.
        if !strict_overwrite {
            (dataset, other_transactions) = load_and_sort_new_transactions(&dataset).await?;

            if let Some(version) =
                committed_transaction_version(&requested_transaction, &other_transactions)?
            {
                return committed_transaction_result(&dataset, version).await;
            }

            // See if we can retry the commit. Try to account for all
            // transactions that have been committed since the read_version.
            // Use small amount of backoff to handle transactions that all
            // started at exact same time better.

            let mut rebase =
                TransactionRebase::try_new(&original_dataset, transaction, affected_rows).await?;

            for (other_version, other_transaction) in other_transactions.iter() {
                rebase.check_txn(other_transaction, *other_version)?;
            }

            transaction = rebase.finish(&dataset).await?;
        }

        enrich_v2_3_row_aligned_rewrite(&dataset, &mut transaction).await?;
        validate_v23_incoming_index_artifacts(&dataset, &transaction).await?;

        let current_transaction_file = if !write_config.disable_transaction_file() {
            transaction_file_name(&transaction)
        } else {
            String::new()
        };
        let transaction_file = current_transaction_file.as_str();

        target_version = dataset.manifest.version + 1;
        if is_detached_version(target_version) {
            return Err(Error::internal(
                "more than 2^65 versions have been created and so regular version numbers are appearing as 'detached' versions.",
            ));
        }
        // Build an up-to-date manifest from the transaction and current manifest
        let row_address_context =
            prepare_row_address_manifest_context(&dataset, &transaction).await?;
        let (mut manifest, mut indices) = match transaction.operation {
            Operation::Restore { version } => {
                Transaction::restore_old_manifest(
                    object_store,
                    commit_handler,
                    &dataset.base,
                    version,
                    write_config,
                    transaction_file,
                    &dataset.manifest,
                )
                .await?
            }
            _ => transaction.build_manifest_with_row_address_context(
                Some(dataset.manifest.as_ref()),
                dataset.load_indices().await?.as_ref().clone(),
                transaction_file,
                write_config,
                row_address_context.as_ref(),
            )?,
        };

        manifest.version = target_version;

        let previous_writer_version = &dataset.manifest.writer_version;
        // The versions of Lance prior to when we started writing the writer version
        // sometimes wrote incorrect `Fragment.physical_rows` values, so we should
        // make sure to recompute them.
        // See: https://github.com/lance-format/lance/issues/1531
        let recompute_stats = previous_writer_version.is_none();

        migrate_manifest(&dataset, &mut manifest, recompute_stats).await?;

        fix_schema(&mut manifest)?;

        check_storage_version(&mut manifest)?;
        check_column_indices(&manifest)?;

        migrate_indices(&dataset, &mut indices).await?;

        if matches!(&transaction.operation, Operation::Restore { .. }) {
            transaction.finalize_row_address_metadata_debt(
                Some(dataset.manifest.as_ref()),
                &mut manifest,
                &indices,
                write_config,
                false,
            )?;
        }

        validate_manifest_write_capacity(&manifest, &indices, &transaction)?;
        if !write_config.disable_transaction_file() {
            let written = write_transaction_file(object_store, &dataset.base, &transaction).await?;
            debug_assert_eq!(written, current_transaction_file);
        }

        // Try to commit the manifest
        let result = write_manifest_file(
            object_store,
            commit_handler,
            &dataset.base,
            &mut manifest,
            if indices.is_empty() {
                None
            } else {
                Some(indices.clone())
            },
            write_config,
            manifest_naming_scheme,
            Some(&transaction),
        )
        .await;

        match result {
            Ok(manifest_location) => {
                // Cache both the transaction file and manifest
                let tx_key = crate::session::caches::TransactionKey {
                    version: target_version,
                };
                dataset
                    .metadata_cache
                    .insert_with_key(&tx_key, Arc::new(transaction.clone()))
                    .await;

                let manifest_key = crate::session::caches::ManifestKey {
                    version: manifest_location.version,
                    e_tag: manifest_location.e_tag.as_deref(),
                };
                dataset
                    .metadata_cache
                    .insert_with_key(&manifest_key, Arc::new(manifest.clone()))
                    .await;
                if !indices.is_empty() {
                    let key = IndexMetadataKey::for_manifest(target_version, &manifest);
                    dataset
                        .index_cache
                        .insert_with_key(&key, Arc::new(indices))
                        .await;
                }

                if !commit_config.skip_auto_cleanup {
                    // Note: We're using the old dataset here (before the new manifest is committed).
                    // This means cleanup runs based on the previous version's state, which may affect
                    // which versions are available for cleanup.
                    match auto_cleanup_hook(&dataset, &manifest).await {
                        Ok(Some(stats)) => log::info!("Auto cleanup triggered: {:?}", stats),
                        Err(e) => log::error!("Error encountered during auto_cleanup_hook: {}", e),
                        _ => {}
                    };
                }
                return Ok((manifest, manifest_location));
            }
            Err(CommitError::CommitConflict) => {
                let next_attempt_i = backoff.attempt() + 1;

                if backoff.attempt() == 0 {
                    // We add 10% buffer here, to allow concurrent writes to complete.
                    // We pass the first attempt's time to the backoff so it's used
                    // as the unit for backoff time slots.
                    // See SlotBackoff implementation for more details on how this works.
                    backoff = backoff.with_unit((start.elapsed().as_millis() * 11 / 10) as u32);
                }

                if next_attempt_i < num_attempts {
                    // Another submitter may be committing the same transaction UUID and
                    // sharing this transaction-file key. Leave conflict cleanup to GC.
                    tokio::time::sleep(backoff.next_backoff()).await;
                    continue;
                } else {
                    break;
                }
            }
            Err(CommitError::OtherError(err)) => {
                match load_and_sort_new_transactions(&transaction_base_dataset).await {
                    Ok((latest, transactions)) => {
                        if let Some(version) =
                            committed_transaction_version(&requested_transaction, &transactions)?
                        {
                            return committed_transaction_result(&latest, version).await;
                        }
                        cleanup_transaction_file(
                            object_store,
                            &dataset.base,
                            &current_transaction_file,
                        )
                        .await;
                    }
                    Err(reconciliation_error) => {
                        // The manifest may have committed even though its response was lost.
                        // Without a successful reconciliation, the transaction file can be
                        // referenced by that manifest and must remain available for readers.
                        log::warn!(
                            "Failed to reconcile an ambiguous manifest commit; retaining transaction file '{}': {}",
                            current_transaction_file,
                            reconciliation_error
                        );
                    }
                }
                return Err(err);
            }
        }
    }

    Err(crate::Error::commit_conflict_source(
        target_version,
        format!(
            "Failed to commit the transaction after {} retries.",
            commit_config.num_retries
        )
        .into(),
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use arrow_array::types::Int32Type;
    use arrow_array::{Int32Array, Int64Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::future::join_all;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::datatypes::{Field, Schema};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
    use lance_datagen::{BatchCount, RowCount, array, gen_batch};
    use lance_file::writer::{FileWriter, FileWriterOptions};
    use lance_index::IndexType;
    use lance_io::object_store::{ObjectStoreParams, WrappingObjectStore};
    use lance_linalg::distance::MetricType;
    use lance_table::format::{
        DataFile, DataStorageFormat, NativeLogicalDomain, RowAddressLayout, RowAddressLayoutDelta,
        RowAddressTargetRange,
    };
    use lance_table::io::commit::{
        CommitLease, CommitLock, ManifestWriter, RenameCommitHandler, UnsafeCommitHandler,
    };
    use lance_testing::datagen::generate_random_array;
    use uuid::Uuid;

    use super::*;

    use crate::Dataset;
    use crate::dataset::transaction::{DataReplacementGroup, TransactionBuilder};
    use crate::dataset::{CommitBuilder, DeleteBuilder, InsertBuilder, WriteMode, WriteParams};
    use crate::index::vector::VectorIndexParams;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

    async fn test_commit_handler(handler: Arc<dyn CommitHandler>, should_succeed: bool) {
        // Create a dataset, passing handler as commit handler
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            DataType::Int64,
            false,
        )]));
        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(data)], schema);

        let options = WriteParams {
            commit_handler: Some(handler),
            ..Default::default()
        };
        let dataset = Dataset::write(reader, "memory://test", Some(options))
            .await
            .unwrap();

        // Create 10 concurrent tasks to write into the table
        // Record how many succeed and how many fail
        let tasks = (0..10).map(|_| {
            let mut dataset = dataset.clone();
            tokio::task::spawn(async move {
                dataset
                    .delete("x = 2")
                    .await
                    .map(|_| dataset.manifest.version)
            })
        });

        let task_results: Vec<Option<u64>> = join_all(tasks)
            .await
            .iter()
            .map(|res| match res {
                Ok(Ok(version)) => Some(*version),
                _ => None,
            })
            .collect();

        let num_successes = task_results.iter().filter(|x| x.is_some()).count();
        let distinct_results: HashSet<_> = task_results.iter().filter_map(|x| x.as_ref()).collect();

        if should_succeed {
            assert_eq!(
                num_successes,
                distinct_results.len(),
                "Expected no two tasks to succeed for the same version. Got {:?}",
                task_results
            );
        } else {
            // All we can promise here is at least one tasks succeeds, but multiple
            // could in theory.
            assert!(num_successes >= distinct_results.len(),);
        }
    }

    #[tokio::test]
    async fn test_rename_commit_handler() {
        // Rename is default for memory
        let handler = Arc::new(RenameCommitHandler);
        test_commit_handler(handler, true).await;
    }

    #[tokio::test]
    async fn test_custom_commit() {
        #[derive(Debug)]
        struct CustomCommitHandler {
            locked_version: Arc<Mutex<Option<u64>>>,
        }

        struct CustomCommitLease {
            version: u64,
            locked_version: Arc<Mutex<Option<u64>>>,
        }

        #[async_trait::async_trait]
        impl CommitLock for CustomCommitHandler {
            type Lease = CustomCommitLease;

            async fn lock(&self, version: u64) -> std::result::Result<Self::Lease, CommitError> {
                let mut locked_version = self.locked_version.lock().unwrap();
                if locked_version.is_some() {
                    // Already locked
                    return Err(CommitError::CommitConflict);
                }

                // Lock the version
                *locked_version = Some(version);

                Ok(CustomCommitLease {
                    version,
                    locked_version: self.locked_version.clone(),
                })
            }
        }

        #[async_trait::async_trait]
        impl CommitLease for CustomCommitLease {
            async fn release(&self, _success: bool) -> std::result::Result<(), CommitError> {
                let mut locked_version = self.locked_version.lock().unwrap();
                if *locked_version != Some(self.version) {
                    // Already released
                    return Err(CommitError::CommitConflict);
                }

                // Release the version
                *locked_version = None;

                Ok(())
            }
        }

        let locked_version = Arc::new(Mutex::new(None));
        let handler = Arc::new(CustomCommitHandler { locked_version });
        test_commit_handler(handler, true).await;
    }

    #[tokio::test]
    async fn test_unsafe_commit_handler() {
        let handler = Arc::new(UnsafeCommitHandler);
        test_commit_handler(handler, false).await;
    }

    #[tokio::test]
    async fn test_roundtrip_transaction_file() {
        let object_store = ObjectStore::memory();
        let base_path = Path::from("test");
        let transaction = Transaction::new(
            42,
            Operation::Append { fragments: vec![] },
            Some("hello world".to_string()),
        );

        let file_name = write_transaction_file(&object_store, &base_path, &transaction)
            .await
            .unwrap();
        let read_transaction = read_transaction_file(&object_store, &base_path, &file_name)
            .await
            .unwrap();

        assert_eq!(transaction.read_version, read_transaction.read_version);
        assert_eq!(transaction.uuid, read_transaction.uuid);
        assert!(matches!(
            read_transaction.operation,
            Operation::Append { .. }
        ));
        assert_eq!(transaction.tag, read_transaction.tag);
    }

    #[rstest::rstest]
    #[case::v2_2_no_stable(LanceFileVersion::V2_2, false)]
    #[case::v2_2_stable(LanceFileVersion::V2_2, true)]
    #[case::v2_3(LanceFileVersion::V2_3, false)]
    #[tokio::test]
    async fn ambiguous_append_retry_returns_committed_version(
        #[case] version: LanceFileVersion,
        #[case] enable_stable_row_ids: bool,
    ) {
        let uri = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let initial =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
                .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(initial)], schema.clone()),
            uri.as_str(),
            Some(WriteParams {
                data_storage_version: Some(version),
                enable_stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let appended =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![2]))]).unwrap();
        let append_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let transaction = InsertBuilder::new(Arc::new(dataset.clone()))
            .with_params(&append_params)
            .execute_uncommitted(vec![appended])
            .await
            .unwrap();

        let first = CommitBuilder::new(Arc::new(dataset.clone()))
            .execute(transaction.clone())
            .await
            .unwrap();
        let second = CommitBuilder::new(Arc::new(dataset))
            .execute(transaction)
            .await
            .unwrap();

        assert_eq!(first.version_id(), 2);
        assert_eq!(second.version_id(), first.version_id());
        assert_eq!(second.count_rows(None).await.unwrap(), 2);
        assert_eq!(
            second.manifest.max_fragment_id,
            first.manifest.max_fragment_id
        );
        assert_eq!(
            second.manifest.max_logical_fragment_id,
            first.manifest.max_logical_fragment_id
        );
    }

    #[test]
    fn same_uuid_rejects_different_append_order() {
        let first = Fragment::new(0).with_physical_rows(1);
        let second = Fragment::new(0).with_physical_rows(2);
        let requested = TransactionBuilder::new(
            1,
            Operation::Append {
                fragments: vec![first.clone(), second.clone()],
            },
        )
        .uuid("fixed-uuid".to_owned())
        .build();
        let committed = TransactionBuilder::new(
            1,
            Operation::Append {
                fragments: vec![second, first],
            },
        )
        .uuid(requested.uuid.clone())
        .build();

        let error =
            committed_transaction_version(&requested, &[(2, Arc::new(committed))]).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("different request"));
    }

    #[tokio::test]
    async fn test_concurrent_create_index() {
        // Create a table with two vector columns
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let dimension = 16;
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(
                "vector1",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    dimension,
                ),
                false,
            ),
            ArrowField::new(
                "vector2",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Float32, true)),
                    dimension,
                ),
                false,
            ),
        ]));
        let float_arr = generate_random_array(512 * dimension as usize);
        let vectors = Arc::new(
            <arrow_array::FixedSizeListArray as FixedSizeListArrayExt>::try_new_from_values(
                float_arr, dimension,
            )
            .unwrap(),
        );
        let batches = vec![
            RecordBatch::try_new(schema.clone(), vec![vectors.clone(), vectors.clone()]).unwrap(),
        ];

        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let dataset = Dataset::write(reader, test_uri, None).await.unwrap();
        dataset.validate().await.unwrap();

        // From initial version, concurrently call create index 3 times,
        // two of which will be for the same column.
        let params = VectorIndexParams::ivf_pq(10, 8, 2, MetricType::L2, 50);
        let futures: Vec<_> = ["vector1", "vector1", "vector2"]
            .iter()
            .map(|col_name| {
                let mut dataset = dataset.clone();
                let params = params.clone();
                tokio::spawn(async move {
                    dataset
                        .create_index(&[col_name], IndexType::Vector, None, &params, true)
                        .await
                })
            })
            .collect();

        let results = join_all(futures).await;
        let success_count = results
            .iter()
            .filter(|result| matches!(result, Ok(Ok(_))))
            .count();
        let retryable_count = results
            .iter()
            .filter(|result| matches!(result, Ok(Err(Error::RetryableCommitConflict { .. }))))
            .count();
        assert_eq!(success_count, 2, "{results:?}");
        assert_eq!(retryable_count, 1, "{results:?}");

        // Validate that each version has the anticipated number of indexes
        let dataset = dataset.checkout_version(1).await.unwrap();
        assert!(dataset.load_indices().await.unwrap().is_empty());

        let dataset = dataset.checkout_version(2).await.unwrap();
        assert_eq!(dataset.load_indices().await.unwrap().len(), 1);

        let dataset = dataset.checkout_version(3).await.unwrap();
        let indices = dataset.load_indices().await.unwrap();
        assert!(!indices.is_empty() && indices.len() <= 2);

        // At this point, we have created two indices. If they are both for the same column,
        // it must be vector1 and not vector2.
        if indices.len() == 2 {
            let mut fields: Vec<i32> = indices.iter().flat_map(|i| i.fields.clone()).collect();
            fields.sort();
            assert_eq!(fields, vec![0, 1]);
        } else {
            assert_eq!(indices[0].fields, vec![0]);
        }

        assert!(dataset.checkout_version(4).await.is_err());
    }

    #[tokio::test]
    async fn test_load_and_sort_new_transactions() {
        // Create a dataset
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(10))
            .await
            .unwrap();

        // Create 100 small UpdateConfig transactions
        for i in 0..100 {
            dataset
                .update_config(vec![(format!("key_{}", i), format!("value_{}", i))])
                .await
                .unwrap();
        }

        // Now load the dataset at version 1 and check that load_and_sort_new_transactions
        // returns transactions in order
        let dataset_v1 = dataset.checkout_version(1).await.unwrap();
        let (_, transactions) = load_and_sort_new_transactions(&dataset_v1).await.unwrap();

        // Verify transactions are sorted by version
        let versions: Vec<u64> = transactions.iter().map(|(v, _)| *v).collect();
        for i in 1..versions.len() {
            assert!(
                versions[i] > versions[i - 1],
                "Transactions not in order: version {} came after version {}",
                versions[i],
                versions[i - 1]
            );
        }

        // Also verify we have exactly 100 transactions (versions 2-101)
        assert_eq!(transactions.len(), 100);
        assert_eq!(versions.first(), Some(&2));
        assert_eq!(versions.last(), Some(&101));
    }

    #[tokio::test]
    async fn test_concurrent_writes() {
        // Test concurrent appends - all should succeed
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));

        let dataset = Dataset::write(
            RecordBatchIterator::new(vec![].into_iter().map(Ok), schema.clone()),
            test_uri,
            None,
        )
        .await
        .unwrap();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();

        let futures: Vec<_> = (0..5)
            .map(|_| {
                let batch = batch.clone();
                let schema = schema.clone();
                let uri = test_uri.to_string();
                tokio::spawn(async move {
                    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
                    Dataset::write(
                        reader,
                        &uri,
                        Some(WriteParams {
                            mode: WriteMode::Append,
                            ..Default::default()
                        }),
                    )
                    .await
                })
            })
            .collect();
        let results = join_all(futures).await;

        for result in results {
            assert!(matches!(result, Ok(Ok(_))), "{:?}", result);
        }

        let dataset = dataset.checkout_version(6).await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 5);
        dataset.validate().await.unwrap()
    }

    #[tokio::test]
    async fn test_restore_does_not_decrease_max_fragment_id() {
        let reader = gen_batch()
            .col("i", array::step::<Int32Type>())
            .into_reader_rows(RowCount::from(3), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        // Append a few times to advance max_fragment_id and create newer versions.
        for _ in 0..2 {
            let reader = gen_batch()
                .col("i", array::step::<Int32Type>())
                .into_reader_rows(RowCount::from(3), BatchCount::from(1));
            dataset.append(reader, None).await.unwrap();
        }

        let latest_max = dataset.manifest.max_fragment_id().unwrap_or(0);

        // Restore an earlier version (version 1) as the latest.
        let mut dataset_v1 = dataset.checkout_version(1).await.unwrap();
        dataset_v1.restore().await.unwrap();

        // After restore, max_fragment_id should not decrease compared to the latest value before restore.
        let restored_max = dataset_v1.manifest.max_fragment_id().unwrap_or(0);
        assert!(
            restored_max >= latest_max,
            "max_fragment_id should not decrease on restore: before={}, after={}",
            latest_max,
            restored_max
        );
    }

    async fn get_empty_dataset() -> (TempStrDir, Dataset) {
        let test_dir = TempStrDir::default();
        let test_uri = test_dir.as_str();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));

        let ds = Dataset::write(
            RecordBatchIterator::new(vec![].into_iter().map(Ok), schema.clone()),
            test_uri,
            None,
        )
        .await
        .unwrap();
        (test_dir, ds)
    }

    #[tokio::test]
    async fn test_good_concurrent_config_writes() {
        let (_tmpdir, dataset) = get_empty_dataset().await;
        let original_num_config_keys = dataset.manifest.config.len();

        // Test successful concurrent insert config operations
        let futures: Vec<_> = ["key1", "key2", "key3", "key4", "key5"]
            .iter()
            .map(|key| {
                let mut dataset = dataset.clone();
                tokio::spawn(async move {
                    dataset
                        .update_config(HashMap::from([(
                            key.to_string(),
                            Some("value".to_string()),
                        )]))
                        .await
                })
            })
            .collect();
        let results = join_all(futures).await;

        // Assert all succeeded
        for result in results {
            assert!(matches!(result, Ok(Ok(_))), "{:?}", result);
        }

        let dataset = dataset.checkout_version(6).await.unwrap();
        assert_eq!(dataset.manifest.config.len(), 5 + original_num_config_keys);

        dataset.validate().await.unwrap();

        // Test successful concurrent delete operations. If multiple delete
        // operations attempt to delete the same key, they are all successful.
        let futures: Vec<_> = ["key1", "key1", "key1", "key2", "key2"]
            .iter()
            .map(|key| {
                let mut dataset = dataset.clone();
                tokio::spawn(async move {
                    dataset
                        .update_config(HashMap::from([(key.to_string(), None)]))
                        .await
                })
            })
            .collect();
        let results = join_all(futures).await;

        // Assert all succeeded
        for result in results {
            assert!(matches!(result, Ok(Ok(_))), "{:?}", result);
        }

        let dataset = dataset.checkout_version(11).await.unwrap();

        // There are now two fewer keys
        assert_eq!(dataset.manifest.config.len(), 3 + original_num_config_keys);

        dataset.validate().await.unwrap()
    }

    #[tokio::test]
    async fn test_bad_concurrent_config_writes() {
        // If two concurrent insert config operations occur for the same key, a
        // `CommitConflict` should be returned
        let (_tmpdir, dataset) = get_empty_dataset().await;

        let futures: Vec<_> = ["key1", "key1", "key2", "key3", "key4"]
            .iter()
            .map(|key| {
                let mut dataset = dataset.clone();
                tokio::spawn(async move {
                    dataset
                        .update_config(HashMap::from([(
                            key.to_string(),
                            Some("value".to_string()),
                        )]))
                        .await
                })
            })
            .collect();

        let results = join_all(futures).await;

        // Assert that either the first or the second operation fails
        let mut first_operation_failed = false;
        for (i, result) in results.into_iter().enumerate() {
            let result = result.unwrap();
            match i {
                0 => {
                    if result.is_err() {
                        first_operation_failed = true;
                        assert!(
                            matches!(&result, &Err(Error::IncompatibleTransaction { .. })),
                            "{:?}",
                            result,
                        );
                    }
                }
                1 => match first_operation_failed {
                    true => assert!(result.is_ok(), "{:?}", result),
                    false => {
                        assert!(
                            matches!(&result, &Err(Error::IncompatibleTransaction { .. })),
                            "{:?}",
                            result,
                        );
                    }
                },
                _ => assert!(result.is_ok(), "{:?}", result),
            }
        }
    }

    #[test]
    fn test_fix_schema() {
        // Manifest has a fragment with no fields in use
        // Manifest has a duplicate field id in one fragment but not others.
        let mut field0 =
            Field::try_from(ArrowField::new("a", arrow_schema::DataType::Int64, false)).unwrap();
        field0.set_id(-1, &mut 0);
        let mut field2 =
            Field::try_from(ArrowField::new("b", arrow_schema::DataType::Int64, false)).unwrap();
        field2.set_id(-1, &mut 2);

        let schema = Schema {
            fields: vec![field0.clone(), field2.clone()],
            metadata: Default::default(),
        };
        let fragments = vec![
            Fragment {
                id: 0,
                files: vec![
                    DataFile::new_legacy_from_fields("path1", vec![0, 1, 2], None),
                    DataFile::new_legacy_from_fields("unused", vec![9], None),
                ],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                native_logical_domain: None,
            },
            Fragment {
                id: 1,
                files: vec![
                    DataFile::new_legacy_from_fields("path2", vec![0, 1, 2], None),
                    DataFile::new_legacy_from_fields("path3", vec![2], None),
                ],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                native_logical_domain: None,
            },
        ];

        let mut manifest = Manifest::new(
            schema,
            Arc::new(fragments),
            DataStorageFormat::default(),
            HashMap::new(),
        );

        fix_schema(&mut manifest).unwrap();

        // Because of the duplicate field id, the field id of field2 should have been changed to 10
        field2.id = 10;
        let expected_schema = Schema {
            fields: vec![field0, field2],
            metadata: Default::default(),
        };
        assert_eq!(manifest.schema, expected_schema);

        // The fragment with just field 9 should have been removed, since it's
        // not used in the current schema.
        // The field 2 should have been changed to 10, except in the first
        // file of the second fragment.
        let expected_fragments = vec![
            Fragment {
                id: 0,
                files: vec![DataFile::new_legacy_from_fields(
                    "path1",
                    vec![0, 1, 10],
                    None,
                )],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                native_logical_domain: None,
            },
            Fragment {
                id: 1,
                files: vec![
                    DataFile::new_legacy_from_fields("path2", vec![0, 1, 2], None),
                    DataFile::new_legacy_from_fields("path3", vec![10], None),
                ],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                native_logical_domain: None,
            },
        ];
        assert_eq!(manifest.fragments.as_ref(), &expected_fragments);
    }

    /// A CommitHandler that always fails with OtherError, used to simulate
    /// a manifest write failure so we can verify orphaned transaction files
    /// are cleaned up.
    #[derive(Debug)]
    struct FailingCommitHandler;

    #[async_trait::async_trait]
    impl CommitHandler for FailingCommitHandler {
        async fn commit(
            &self,
            _manifest: &mut Manifest,
            _indices: Option<Vec<IndexMetadata>>,
            _base_path: &Path,
            _object_store: &ObjectStore,
            _manifest_writer: ManifestWriter,
            _naming_scheme: ManifestNamingScheme,
            _transaction: Option<lance_table::format::Transaction>,
        ) -> std::result::Result<ManifestLocation, CommitError> {
            Err(CommitError::OtherError(lance_core::Error::io(
                "simulated commit failure",
            )))
        }
    }

    #[derive(Debug)]
    struct CommitThenErrorHandler {
        fail_once: std::sync::atomic::AtomicBool,
    }

    #[async_trait::async_trait]
    impl CommitHandler for CommitThenErrorHandler {
        async fn commit(
            &self,
            manifest: &mut Manifest,
            indices: Option<Vec<IndexMetadata>>,
            base_path: &Path,
            object_store: &ObjectStore,
            manifest_writer: ManifestWriter,
            naming_scheme: ManifestNamingScheme,
            transaction: Option<lance_table::format::Transaction>,
        ) -> std::result::Result<ManifestLocation, CommitError> {
            let location = RenameCommitHandler
                .commit(
                    manifest,
                    indices,
                    base_path,
                    object_store,
                    manifest_writer,
                    naming_scheme,
                    transaction,
                )
                .await?;
            if self
                .fail_once
                .swap(false, std::sync::atomic::Ordering::SeqCst)
            {
                Err(CommitError::OtherError(Error::io(format!(
                    "simulated response loss after committing {}",
                    location.version
                ))))
            } else {
                Ok(location)
            }
        }
    }

    #[derive(Debug)]
    struct ToggleReadFailureWrapper {
        enabled: Arc<std::sync::atomic::AtomicBool>,
        failed_reads: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl WrappingObjectStore for ToggleReadFailureWrapper {
        fn wrap(
            &self,
            _storage_prefix: &str,
            original: Arc<dyn object_store::ObjectStore>,
        ) -> Arc<dyn object_store::ObjectStore> {
            let enabled = self.enabled.clone();
            let failed_reads = self.failed_reads.clone();
            let mut policy = ProxyObjectStorePolicy::new();
            policy.set_before_policy(
                "fail_reads_after_ambiguous_commit",
                Arc::new(move |method, _path| {
                    if method == "get_opts" && enabled.load(std::sync::atomic::Ordering::SeqCst) {
                        failed_reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        Err(Error::io("simulated reconciliation read failure"))
                    } else {
                        Ok(())
                    }
                }),
            );
            Arc::new(ProxyObjectStore::new(
                original,
                Arc::new(Mutex::new(policy)),
            ))
        }
    }

    #[derive(Debug)]
    struct CommitThenErrorAndBlockReadsHandler {
        block_reads: Arc<std::sync::atomic::AtomicBool>,
        committed_location: Arc<Mutex<Option<ManifestLocation>>>,
    }

    #[async_trait::async_trait]
    impl CommitHandler for CommitThenErrorAndBlockReadsHandler {
        async fn commit(
            &self,
            manifest: &mut Manifest,
            indices: Option<Vec<IndexMetadata>>,
            base_path: &Path,
            object_store: &ObjectStore,
            manifest_writer: ManifestWriter,
            naming_scheme: ManifestNamingScheme,
            transaction: Option<lance_table::format::Transaction>,
        ) -> std::result::Result<ManifestLocation, CommitError> {
            let location = RenameCommitHandler
                .commit(
                    manifest,
                    indices,
                    base_path,
                    object_store,
                    manifest_writer,
                    naming_scheme,
                    transaction,
                )
                .await?;
            *self.committed_location.lock().unwrap() = Some(location.clone());
            self.block_reads
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Err(CommitError::OtherError(Error::io(format!(
                "simulated response loss after committing {}",
                location.version
            ))))
        }
    }

    #[derive(Debug)]
    struct FirstAttemptBarrierCommitHandler {
        barrier: Arc<tokio::sync::Barrier>,
        calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl CommitHandler for FirstAttemptBarrierCommitHandler {
        async fn commit(
            &self,
            manifest: &mut Manifest,
            indices: Option<Vec<IndexMetadata>>,
            base_path: &Path,
            object_store: &ObjectStore,
            manifest_writer: ManifestWriter,
            naming_scheme: ManifestNamingScheme,
            transaction: Option<lance_table::format::Transaction>,
        ) -> std::result::Result<ManifestLocation, CommitError> {
            if self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < 2 {
                self.barrier.wait().await;
            }
            RenameCommitHandler
                .commit(
                    manifest,
                    indices,
                    base_path,
                    object_store,
                    manifest_writer,
                    naming_scheme,
                    transaction,
                )
                .await
        }
    }

    #[derive(Debug)]
    struct AlwaysCommitConflictHandler;

    #[async_trait::async_trait]
    impl CommitHandler for AlwaysCommitConflictHandler {
        async fn commit(
            &self,
            _manifest: &mut Manifest,
            _indices: Option<Vec<IndexMetadata>>,
            _base_path: &Path,
            _object_store: &ObjectStore,
            _manifest_writer: ManifestWriter,
            _naming_scheme: ManifestNamingScheme,
            _transaction: Option<lance_table::format::Transaction>,
        ) -> std::result::Result<ManifestLocation, CommitError> {
            Err(CommitError::CommitConflict)
        }
    }

    #[tokio::test]
    async fn post_commit_other_error_is_reconciled_by_transaction_uuid() {
        let uri = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let initial =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
                .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(initial)], schema.clone()),
            uri.as_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let appended =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![2]))]).unwrap();
        let append_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let transaction = InsertBuilder::new(Arc::new(dataset.clone()))
            .with_params(&append_params)
            .execute_uncommitted(vec![appended])
            .await
            .unwrap();
        let handler = Arc::new(CommitThenErrorHandler {
            fail_once: std::sync::atomic::AtomicBool::new(true),
        });

        let committed = CommitBuilder::new(Arc::new(dataset))
            .with_commit_handler(handler)
            .execute(transaction)
            .await
            .unwrap();
        assert_eq!(committed.version_id(), 2);
        assert_eq!(committed.count_rows(None).await.unwrap(), 2);
    }

    #[tokio::test]
    async fn ambiguous_post_commit_error_retains_transaction_for_later_reconciliation() {
        let uri = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let initial =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
                .unwrap();
        let block_reads = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let failed_reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let wrapper = Arc::new(ToggleReadFailureWrapper {
            enabled: block_reads.clone(),
            failed_reads: failed_reads.clone(),
        });
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(initial)], schema.clone()),
            uri.as_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                store_params: Some(ObjectStoreParams {
                    object_store_wrapper: Some(wrapper),
                    ..ObjectStoreParams::default()
                }),
                ..WriteParams::default()
            }),
        )
        .await
        .unwrap();
        let appended =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![2]))]).unwrap();
        let transaction = InsertBuilder::new(Arc::new(dataset.clone()))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..WriteParams::default()
            })
            .execute_uncommitted(vec![appended])
            .await
            .unwrap();
        let handler = Arc::new(CommitThenErrorAndBlockReadsHandler {
            block_reads: block_reads.clone(),
            committed_location: Arc::new(Mutex::new(None)),
        });

        let error = CommitBuilder::new(Arc::new(dataset.clone()))
            .with_commit_handler(handler)
            .execute(transaction.clone())
            .await
            .expect_err("failed reconciliation must preserve the ambiguous error");
        assert!(error.to_string().contains("simulated response loss"));
        assert!(failed_reads.load(std::sync::atomic::Ordering::SeqCst) > 0);

        block_reads.store(false, std::sync::atomic::Ordering::SeqCst);
        let committed = Dataset::open(uri.as_str()).await.unwrap();
        assert_eq!(committed.version_id(), 2);
        assert_eq!(committed.count_rows(None).await.unwrap(), 2);
        assert_eq!(
            committed.read_transaction().await.unwrap().unwrap().uuid,
            transaction.uuid
        );
        let transaction_path = committed
            .base
            .clone()
            .join(TRANSACTIONS_DIR)
            .join(committed.manifest.transaction_file.as_deref().unwrap());
        committed
            .object_store
            .inner
            .head(&transaction_path)
            .await
            .unwrap();

        let reconciled = CommitBuilder::new(Arc::new(dataset))
            .execute(transaction)
            .await
            .unwrap();
        assert_eq!(reconciled.version_id(), committed.version_id());
    }

    #[tokio::test]
    async fn ambiguous_detached_commit_retains_transaction_file() {
        let uri = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let initial =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
                .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(initial)], schema.clone()),
            uri.as_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..WriteParams::default()
            }),
        )
        .await
        .unwrap();
        let appended =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![2]))]).unwrap();
        let transaction = InsertBuilder::new(Arc::new(dataset.clone()))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..WriteParams::default()
            })
            .execute_uncommitted(vec![appended])
            .await
            .unwrap();
        let committed_location = Arc::new(Mutex::new(None));
        let handler = CommitThenErrorAndBlockReadsHandler {
            block_reads: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            committed_location: committed_location.clone(),
        };

        let error = do_commit_detached_transaction(
            &dataset,
            dataset.object_store.as_ref(),
            &handler,
            &transaction,
            &ManifestWriteConfig::default(),
            &CommitConfig::default(),
        )
        .await
        .expect_err("detached response loss must remain caller-visible");
        assert!(error.to_string().contains("simulated response loss"));

        let location = committed_location.lock().unwrap().clone().unwrap();
        assert!(is_detached_version(location.version));
        let committed = dataset.checkout_version(location.version).await.unwrap();
        assert_eq!(committed.count_rows(None).await.unwrap(), 2);
        let transaction_path = committed
            .base
            .clone()
            .join(TRANSACTIONS_DIR)
            .join(transaction_file_name(&transaction));
        committed
            .object_store
            .inner
            .head(&transaction_path)
            .await
            .unwrap();
        assert_eq!(
            committed.read_transaction().await.unwrap().unwrap().uuid,
            transaction.uuid
        );
    }

    #[rstest::rstest]
    #[case::terminal_conflict(1)]
    #[case::retry_after_conflict(2)]
    #[tokio::test]
    async fn concurrent_same_transaction_conflict_keeps_shared_transaction_file(
        #[case] max_retries: u32,
    ) {
        let uri = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let initial =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
                .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(initial)], schema.clone()),
            uri.as_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..WriteParams::default()
            }),
        )
        .await
        .unwrap();
        let appended =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![2]))]).unwrap();
        let transaction = InsertBuilder::new(Arc::new(dataset.clone()))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..WriteParams::default()
            })
            .execute_uncommitted(vec![appended])
            .await
            .unwrap();
        let handler = Arc::new(FirstAttemptBarrierCommitHandler {
            barrier: Arc::new(tokio::sync::Barrier::new(2)),
            calls: std::sync::atomic::AtomicUsize::new(0),
        });

        let first = CommitBuilder::new(Arc::new(dataset.clone()))
            .with_commit_handler(handler.clone())
            .with_max_retries(max_retries)
            .execute(transaction.clone());
        let second = CommitBuilder::new(Arc::new(dataset))
            .with_commit_handler(handler)
            .with_max_retries(max_retries)
            .execute(transaction.clone());
        let (first, second) = tokio::join!(first, second);
        let successes = [&first, &second]
            .into_iter()
            .filter(|result| result.is_ok())
            .count();
        if max_retries == 1 {
            assert_eq!(successes, 1);
            let error = [first.as_ref().err(), second.as_ref().err()]
                .into_iter()
                .flatten()
                .next()
                .unwrap();
            assert!(matches!(error, Error::CommitConflict { .. }));
        } else {
            assert_eq!(successes, 2);
            assert_eq!(first.as_ref().unwrap().version_id(), 2);
            assert_eq!(second.as_ref().unwrap().version_id(), 2);
        }

        let committed = Dataset::open(uri.as_str()).await.unwrap();
        assert_eq!(committed.version_id(), 2);
        let transaction_path = committed
            .base
            .clone()
            .join(TRANSACTIONS_DIR)
            .join(committed.manifest.transaction_file.as_deref().unwrap());
        committed
            .object_store
            .inner
            .head(&transaction_path)
            .await
            .unwrap();
        assert_eq!(
            committed.read_transaction().await.unwrap().unwrap().uuid,
            transaction.uuid
        );
    }

    #[tokio::test]
    async fn detached_conflict_keeps_shared_transaction_file() {
        let uri = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let initial =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
                .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(initial)], schema.clone()),
            uri.as_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..WriteParams::default()
            }),
        )
        .await
        .unwrap();
        let appended =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![2]))]).unwrap();
        let transaction = InsertBuilder::new(Arc::new(dataset.clone()))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..WriteParams::default()
            })
            .execute_uncommitted(vec![appended])
            .await
            .unwrap();
        let (_, committed_location) = do_commit_detached_transaction(
            &dataset,
            dataset.object_store.as_ref(),
            &RenameCommitHandler,
            &transaction,
            &ManifestWriteConfig::default(),
            &CommitConfig::default(),
        )
        .await
        .unwrap();

        let error = do_commit_detached_transaction(
            &dataset,
            dataset.object_store.as_ref(),
            &AlwaysCommitConflictHandler,
            &transaction,
            &ManifestWriteConfig::default(),
            &CommitConfig {
                num_retries: 1,
                ..CommitConfig::default()
            },
        )
        .await
        .expect_err("the forced detached conflict must exhaust its retry budget");
        assert!(matches!(error, Error::CommitConflict { .. }));

        let committed = dataset
            .checkout_version(committed_location.version)
            .await
            .unwrap();
        let transaction_path = committed
            .base
            .clone()
            .join(TRANSACTIONS_DIR)
            .join(transaction_file_name(&transaction));
        committed
            .object_store
            .inner
            .head(&transaction_path)
            .await
            .unwrap();
        assert_eq!(
            committed.read_transaction().await.unwrap().unwrap().uuid,
            transaction.uuid
        );
    }

    #[tokio::test]
    async fn post_commit_other_error_is_reconciled_after_v2_3_rebase() {
        let uri = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let initial = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(initial)], schema.clone()),
            uri.as_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let staged = DeleteBuilder::new(Arc::new(dataset.clone()), "value = 1")
            .execute_uncommitted()
            .await
            .unwrap();
        let appended =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![4]))]).unwrap();
        let append_params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let appended_dataset = InsertBuilder::new(Arc::new(dataset.clone()))
            .with_params(&append_params)
            .execute(vec![appended])
            .await
            .unwrap();
        assert_eq!(appended_dataset.version_id(), 2);

        let transaction = staged.transaction;
        let affected_rows = staged.affected_rows.unwrap();
        let handler = Arc::new(CommitThenErrorHandler {
            fail_once: std::sync::atomic::AtomicBool::new(true),
        });
        let committed = CommitBuilder::new(Arc::new(dataset.clone()))
            .with_affected_rows(affected_rows.clone())
            .with_commit_handler(handler)
            .execute(transaction.clone())
            .await
            .unwrap();
        assert_eq!(committed.version_id(), 3);
        let committed_transaction = committed.read_transaction().await.unwrap().unwrap();
        assert_eq!(committed_transaction.read_version, 2);
        assert_eq!(
            committed_transaction
                .original_request_fingerprint
                .as_ref()
                .unwrap()
                .len(),
            crate::dataset::transaction::ORIGINAL_REQUEST_FINGERPRINT_SIZE
        );

        let mut different_request = transaction.clone();
        let Operation::Delete { predicate, .. } = &mut different_request.operation else {
            unreachable!()
        };
        *predicate = "value = 2".to_owned();
        different_request.original_request_fingerprint =
            committed_transaction.original_request_fingerprint.clone();
        let retried = CommitBuilder::new(Arc::new(dataset))
            .with_affected_rows(affected_rows)
            .execute(transaction)
            .await
            .unwrap();
        assert_eq!(retried.version_id(), committed.version_id());
        assert_eq!(retried.count_rows(None).await.unwrap(), 3);

        let error = CommitBuilder::new(Arc::new(retried))
            .execute(different_request)
            .await
            .expect_err("same UUID with a different predicate must be rejected");
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("different request"));
    }

    #[tokio::test]
    async fn post_commit_other_error_is_reconciled_for_source_free_create() {
        let uri = TempStrDir::default();
        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("value", DataType::Int32, false)]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let (major, minor) = LanceFileVersion::V2_3.to_numbers();
        let mut fragment = Fragment::new(0).with_physical_rows(1);
        fragment.files.push(DataFile::new(
            "source-free-create.lance",
            vec![0],
            vec![0],
            major,
            minor,
            None,
            None,
        ));
        let transaction = TransactionBuilder::new(
            0,
            Operation::Overwrite {
                fragments: vec![fragment],
                schema,
                config_upsert_values: None,
                initial_bases: None,
            },
        )
        .build();
        let handler = Arc::new(CommitThenErrorHandler {
            fail_once: std::sync::atomic::AtomicBool::new(true),
        });

        let committed = CommitBuilder::new(uri.as_str())
            .with_storage_format(LanceFileVersion::V2_3)
            .with_commit_handler(handler)
            .execute(transaction.clone())
            .await
            .unwrap();
        let retry = CommitBuilder::new(uri.as_str())
            .with_storage_format(LanceFileVersion::V2_3)
            .execute(transaction)
            .await
            .unwrap();

        assert_eq!(committed.version_id(), 1);
        assert_eq!(retry.version_id(), committed.version_id());
        assert_eq!(
            retry
                .manifest
                .row_address_layout
                .as_ref()
                .unwrap()
                .namespace_uuid,
            committed
                .manifest
                .row_address_layout
                .as_ref()
                .unwrap()
                .namespace_uuid
        );
    }

    #[tokio::test]
    async fn v2_3_row_aligned_retry_rejects_changed_deletion_authority() {
        let uri = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let initial = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![10, 20, 30]))],
        )
        .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(initial)], schema.clone()),
            uri.as_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let replacement = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![40, 50, 60]))],
        )
        .unwrap();
        let object_writer = dataset
            .object_store
            .create(&dataset.data_dir().join("retry-replacement.lance"))
            .await
            .unwrap();
        let mut writer = FileWriter::try_new(
            object_writer,
            schema.as_ref().try_into().unwrap(),
            FileWriterOptions {
                format_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            },
        )
        .unwrap();
        writer.write_batch(&replacement).await.unwrap();
        let summary = writer.finish().await.unwrap();
        let mut replacement_file = dataset.manifest.fragments[0].files[0].clone();
        replacement_file.path = "retry-replacement.lance".to_owned();
        replacement_file.file_size_bytes = CachedFileSize::new(summary.size_bytes);
        let mut transaction = TransactionBuilder::new(
            dataset.version_id(),
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, replacement_file)],
            },
        )
        .build();
        enrich_v2_3_row_aligned_rewrite(&dataset, &mut transaction)
            .await
            .unwrap();
        let first_proof = transaction
            .row_address_layout_delta
            .as_ref()
            .unwrap()
            .row_aligned_rewrite_proofs[0]
            .clone();

        let mut advanced = dataset;
        advanced.delete("value = 20").await.unwrap();
        let error = enrich_v2_3_row_aligned_rewrite(&advanced, &mut transaction)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::RetryableCommitConflict { .. }));
        assert_eq!(first_proof.deletion_offsets_fingerprint, None);
        assert!(error.to_string().contains("deletion state changed"));
    }

    #[tokio::test]
    async fn v2_3_generic_merge_new_fragment_requires_and_preserves_direct_provenance() {
        let uri = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let initial = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![10, 20, 30]))],
        )
        .unwrap();
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(initial)], schema.clone()),
            uri.as_str(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let appended =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![40]))])
                .unwrap();
        let object_writer = dataset
            .object_store
            .create(&dataset.data_dir().join("merge-direct-new.lance"))
            .await
            .unwrap();
        let mut writer = FileWriter::try_new(
            object_writer,
            schema.as_ref().try_into().unwrap(),
            FileWriterOptions {
                format_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            },
        )
        .unwrap();
        writer.write_batch(&appended).await.unwrap();
        let summary = writer.finish().await.unwrap();
        let mut new_file = dataset.manifest.fragments[0].files[0].clone();
        new_file.path = "merge-direct-new.lance".to_owned();
        new_file.file_size_bytes = CachedFileSize::new(summary.size_bytes);
        let mut new_fragment = Fragment::new(1).with_physical_rows(1);
        new_fragment.files.push(new_file);
        let rewritten = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![10, 20, 30]))],
        )
        .unwrap();
        let object_writer = dataset
            .object_store
            .create(&dataset.data_dir().join("merge-direct-existing.lance"))
            .await
            .unwrap();
        let mut writer = FileWriter::try_new(
            object_writer,
            schema.as_ref().try_into().unwrap(),
            FileWriterOptions {
                format_version: Some(LanceFileVersion::V2_3),
                ..Default::default()
            },
        )
        .unwrap();
        writer.write_batch(&rewritten).await.unwrap();
        let summary = writer.finish().await.unwrap();
        let mut rewritten_fragment = dataset.manifest.fragments[0].clone();
        rewritten_fragment.files[0].path = "merge-direct-existing.lance".to_owned();
        rewritten_fragment.files[0].file_size_bytes = CachedFileSize::new(summary.size_bytes);
        let operation = Operation::Merge {
            fragments: vec![rewritten_fragment, new_fragment],
            schema: dataset.schema().clone(),
        };

        let mut missing = TransactionBuilder::new(dataset.version_id(), operation.clone()).build();
        let error = enrich_v2_3_row_aligned_rewrite(&dataset, &mut missing)
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("without Direct row-address provenance")
        );

        let layout = dataset.manifest.row_address_layout.as_ref().unwrap();
        let transaction = TransactionBuilder::new(dataset.version_id(), operation)
            .row_address_layout_delta(Some(RowAddressLayoutDelta {
                placements: vec![RowAddressPlacementDelta {
                    source_selections: Vec::new(),
                    target: RowAddressTargetRange {
                        fragment: RowAddressTargetFragment::NewFragmentOrdinal(0),
                        start_offset: 0,
                        end_offset: 1,
                    },
                    placement_kind: RowAddressPlacementKind::Direct,
                    output_cardinality: 1,
                    output_row_sequence_fingerprint: Vec::new(),
                }],
                expected_layout_fingerprint: layout.fingerprint.clone(),
                ..RowAddressLayoutDelta::default()
            }))
            .build();
        let mut enriched = transaction.clone();
        enrich_v2_3_row_aligned_rewrite(&dataset, &mut enriched)
            .await
            .unwrap();
        assert!(same_transaction_request(&transaction, &enriched));
        let mut missing_direct = transaction.clone();
        missing_direct.row_address_layout_delta = None;
        assert!(!same_transaction_request(&missing_direct, &enriched));

        let committed = CommitBuilder::new(Arc::new(dataset))
            .execute(transaction)
            .await
            .unwrap();
        assert_eq!(committed.count_rows(None).await.unwrap(), 4);
        assert_eq!(committed.manifest.fragments.len(), 2);
        assert!(
            committed.manifest.fragments[1]
                .native_logical_domain
                .is_some()
        );
        let committed_transaction = read_transaction_file(
            committed.object_store.as_ref(),
            &committed.base,
            committed.manifest.transaction_file.as_deref().unwrap(),
        )
        .await
        .unwrap();
        let delta = committed_transaction.row_address_layout_delta.unwrap();
        assert_eq!(delta.placements.len(), 1);
        assert_eq!(delta.row_aligned_rewrite_proofs.len(), 1);
    }

    fn count_txn_files(uri: &str) -> usize {
        let tx_dir = std::path::Path::new(uri).join("_transactions");
        std::fs::read_dir(&tx_dir)
            .map(|rd| rd.filter_map(|e| e.ok()).count())
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn test_transaction_file_cleanup_on_commit_failure() {
        let tmp = TempStrDir::default();
        let uri = tmp.as_str();

        // Create initial dataset with a normal commit handler.
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone());
        Dataset::write(reader, uri, None).await.unwrap();

        let txn_files_before = count_txn_files(uri);

        // Attempt to append with a commit handler that always fails.
        let params = WriteParams {
            mode: WriteMode::Append,
            commit_handler: Some(Arc::new(FailingCommitHandler)),
            ..Default::default()
        };
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let result = Dataset::write(reader, uri, Some(params)).await;
        assert!(result.is_err(), "expected commit to fail");

        // The failed commit must not leave any extra transaction files behind.
        let txn_files_after = count_txn_files(uri);
        assert_eq!(
            txn_files_after,
            txn_files_before,
            "failed commit left {extra} orphaned transaction file(s)",
            extra = txn_files_after.saturating_sub(txn_files_before),
        );
    }
    /// Helper to build a simple manifest for check_column_indices tests.
    fn make_manifest_with_file(
        schema: Schema,
        data_file: DataFile,
        data_storage_version: LanceFileVersion,
    ) -> Manifest {
        let fragment = Fragment {
            id: 0,
            files: vec![data_file],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(100),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            native_logical_domain: None,
        };
        Manifest::new(
            schema,
            Arc::new(vec![fragment]),
            DataStorageFormat::new(data_storage_version),
            HashMap::new(),
        )
    }

    fn make_direct_v23_manifest(namespace_uuid: Uuid) -> Manifest {
        let mut field = Field::try_from(ArrowField::new("x", DataType::Int32, false)).unwrap();
        field.set_id(-1, &mut 0);
        let schema = Schema {
            fields: vec![field],
            metadata: Default::default(),
        };
        let data_file = DataFile::new("data.lance", vec![0], vec![0], 2, 1, None, None);
        let mut manifest = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_3);
        Arc::make_mut(&mut manifest.fragments)[0].native_logical_domain =
            Some(NativeLogicalDomain::new(0, 1).unwrap());
        manifest.max_logical_fragment_id = Some(0);
        let mut layout = RowAddressLayout::new(namespace_uuid);
        let field_generation = lance_table::format::FieldGeneration {
            field_id: 0,
            generation: 1,
        };
        layout.field_default_generations = vec![field_generation];
        layout.index_commit_floors = vec![field_generation];
        manifest.row_address_layout = Some(Arc::new(layout));
        manifest.refresh_row_address_fingerprint().unwrap();
        manifest.validate_row_address_contract().unwrap();
        manifest
    }

    fn clone_transaction(delta: Option<RowAddressLayoutDelta>) -> Transaction {
        crate::dataset::transaction::TransactionBuilder::new(
            1,
            Operation::Clone {
                is_shallow: true,
                ref_name: None,
                ref_version: 1,
                ref_path: "memory://source".to_string(),
                branch_name: None,
            },
        )
        .row_address_layout_delta(delta)
        .build()
    }

    #[test]
    fn test_clone_namespace_retry_is_deterministic() {
        let source = make_direct_v23_manifest(Uuid::new_v4());
        let target_namespace = Uuid::new_v4();
        let transaction =
            clone_transaction(Some(RowAddressLayoutDelta::for_create(target_namespace)));
        let mut first_attempt = source.clone();
        let mut retried_attempt = source.clone();

        apply_clone_row_address_namespace(&mut first_attempt, &transaction).unwrap();
        apply_clone_row_address_namespace(&mut retried_attempt, &transaction).unwrap();

        assert_eq!(
            first_attempt
                .row_address_layout
                .as_ref()
                .unwrap()
                .namespace_uuid,
            target_namespace
        );
        assert_eq!(
            first_attempt.row_address_layout,
            retried_attempt.row_address_layout
        );
        assert_eq!(first_attempt.fragments, source.fragments);
        assert_eq!(
            first_attempt.max_logical_fragment_id,
            source.max_logical_fragment_id
        );
    }

    #[test]
    fn test_clone_namespace_delta_is_storage_version_specific() {
        let source_namespace = Uuid::new_v4();
        let source = make_direct_v23_manifest(source_namespace);
        let mut missing = source.clone();
        assert!(apply_clone_row_address_namespace(&mut missing, &clone_transaction(None)).is_err());

        let mut reused = source;
        assert!(
            apply_clone_row_address_namespace(
                &mut reused,
                &clone_transaction(Some(RowAddressLayoutDelta::for_create(source_namespace))),
            )
            .is_err()
        );

        let schema = Schema::try_from(&ArrowSchema::new(vec![ArrowField::new(
            "x",
            DataType::Int32,
            false,
        )]))
        .unwrap();
        let data_file = DataFile::new("data.lance", vec![0], vec![0], 2, 1, None, None);
        let mut v2_2 = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_2);
        let unchanged = v2_2.clone();
        apply_clone_row_address_namespace(&mut v2_2, &clone_transaction(None)).unwrap();
        assert_eq!(v2_2, unchanged);

        assert!(
            apply_clone_row_address_namespace(
                &mut v2_2,
                &clone_transaction(Some(RowAddressLayoutDelta::for_create(Uuid::new_v4()))),
            )
            .is_err()
        );
    }

    #[test]
    fn test_clone_finalization_recomputes_row_address_debt() {
        let source = make_direct_v23_manifest(Uuid::new_v4());
        let transaction =
            clone_transaction(Some(RowAddressLayoutDelta::for_create(Uuid::new_v4())));
        let mut cloned = source;
        {
            let debt = &mut Arc::make_mut(cloned.row_address_layout.as_mut().unwrap()).debt_summary;
            debt.canonical_layout_bytes = 999_001;
            debt.fast_delta_bytes = 999_002;
            debt.metadata_bytes_written_since_maintenance = 999_003;
            debt.generation_delta_bytes = 999_004;
            debt.generation_metadata_bytes_written_since_maintenance = 999_005;
        }
        apply_clone_row_address_namespace(&mut cloned, &transaction).unwrap();

        transaction
            .finalize_row_address_metadata_debt(
                None,
                &mut cloned,
                &[],
                &ManifestWriteConfig::default(),
                false,
            )
            .unwrap();
        let first = cloned
            .row_address_layout
            .as_ref()
            .unwrap()
            .debt_summary
            .clone();
        assert_ne!(first.canonical_layout_bytes, 999_001);
        assert_ne!(first.fast_delta_bytes, 999_002);
        assert_ne!(first.metadata_bytes_written_since_maintenance, 999_003);
        assert!(first.fast_delta_bytes > 0);

        transaction
            .finalize_row_address_metadata_debt(
                None,
                &mut cloned,
                &[],
                &ManifestWriteConfig::default(),
                false,
            )
            .unwrap();
        assert_eq!(
            cloned.row_address_layout.as_ref().unwrap().debt_summary,
            first
        );
    }

    #[test]
    fn test_restore_finalization_uses_current_epoch_and_restored_snapshot() {
        let mut current = make_direct_v23_manifest(Uuid::new_v4());
        current.version = 7;
        {
            let debt =
                &mut Arc::make_mut(current.row_address_layout.as_mut().unwrap()).debt_summary;
            debt.metadata_bytes_written_since_maintenance = 1_000;
            debt.generation_metadata_bytes_written_since_maintenance = 100;
        }
        let mut restored = current.clone();
        restored.version = 8;
        {
            let debt =
                &mut Arc::make_mut(restored.row_address_layout.as_mut().unwrap()).debt_summary;
            debt.fast_delta_bytes = 888_001;
            debt.metadata_bytes_written_since_maintenance = 888_002;
            debt.generation_delta_bytes = 888_003;
            debt.generation_metadata_bytes_written_since_maintenance = 888_004;
        }
        let transaction = TransactionBuilder::new(7, Operation::Restore { version: 1 }).build();

        transaction
            .finalize_row_address_metadata_debt(
                Some(&current),
                &mut restored,
                &[],
                &ManifestWriteConfig::default(),
                false,
            )
            .unwrap();
        let debt = &restored.row_address_layout.as_ref().unwrap().debt_summary;
        assert_ne!(debt.fast_delta_bytes, 888_001);
        assert_ne!(debt.metadata_bytes_written_since_maintenance, 888_002);
        assert!(debt.metadata_bytes_written_since_maintenance > 1_000);
        assert!(debt.generation_metadata_bytes_written_since_maintenance >= 100);
        assert!(debt.generation_delta_bytes <= debt.fast_delta_bytes);
    }

    #[test]
    fn test_check_column_indices_rejects_struct_with_column() {
        // Struct (non-leaf) field with column_index=0 in v2.1 should be rejected.
        let mut struct_field = Field::try_from(ArrowField::new(
            "s",
            DataType::Struct(vec![ArrowField::new("x", DataType::Int32, false)].into()),
            false,
        ))
        .unwrap();
        struct_field.set_id(-1, &mut 0);

        let schema = Schema {
            fields: vec![struct_field],
            metadata: Default::default(),
        };

        // field ids: struct=0, leaf=1; give struct a real column_index (wrong)
        let data_file = DataFile::new("data.lance", vec![0, 1], vec![0, 1], 2, 1, None, None);
        let manifest = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_1);
        let result = check_column_indices(&manifest);
        assert!(
            result.is_err(),
            "Expected error for struct with column_index=0"
        );
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Non-leaf field"), "{msg}");
    }

    #[test]
    fn test_check_column_indices_rejects_list_with_column() {
        // List (non-leaf) field with column_index=0 in v2.1 should be rejected.
        let mut list_field = Field::try_from(ArrowField::new(
            "l",
            DataType::List(Arc::new(ArrowField::new("item", DataType::Int32, true))),
            false,
        ))
        .unwrap();
        list_field.set_id(-1, &mut 0);

        let schema = Schema {
            fields: vec![list_field],
            metadata: Default::default(),
        };

        // field ids: list=0, item=1; give list a real column_index (wrong)
        let data_file = DataFile::new("data.lance", vec![0, 1], vec![0, 1], 2, 1, None, None);
        let manifest = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_1);
        let result = check_column_indices(&manifest);
        assert!(
            result.is_err(),
            "Expected error for list with column_index=0"
        );
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Non-leaf field"), "{msg}");
    }

    #[test]
    fn test_check_column_indices_allows_correct_v21() {
        // Non-leaf with column_index=-1 and leaf with column_index>=0 should pass.
        let mut struct_field = Field::try_from(ArrowField::new(
            "s",
            DataType::Struct(vec![ArrowField::new("x", DataType::Int32, false)].into()),
            false,
        ))
        .unwrap();
        struct_field.set_id(-1, &mut 0);

        let schema = Schema {
            fields: vec![struct_field],
            metadata: Default::default(),
        };

        // struct=-1 (correct), leaf=0 (correct)
        let data_file = DataFile::new("data.lance", vec![0, 1], vec![-1, 0], 2, 1, None, None);
        let manifest = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_1);
        assert!(check_column_indices(&manifest).is_ok());
    }

    #[test]
    fn test_check_column_indices_allows_packed_struct() {
        // Packed struct with a real column_index in v2.1 should be allowed.
        let mut struct_field = Field::try_from(ArrowField::new(
            "s",
            DataType::Struct(vec![ArrowField::new("x", DataType::Int32, false)].into()),
            false,
        ))
        .unwrap();
        struct_field.set_id(-1, &mut 0);
        struct_field
            .metadata
            .insert("lance-encoding:packed".to_string(), "true".to_string());

        let schema = Schema {
            fields: vec![struct_field],
            metadata: Default::default(),
        };

        // packed struct=0 (allowed), leaf=1
        let data_file = DataFile::new("data.lance", vec![0, 1], vec![0, 1], 2, 1, None, None);
        let manifest = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_1);
        assert!(check_column_indices(&manifest).is_ok());
    }

    #[test]
    fn test_check_column_indices_skips_v20() {
        // Non-leaf with column_index>=0 in v2.0 should be allowed (no validation).
        let mut struct_field = Field::try_from(ArrowField::new(
            "s",
            DataType::Struct(vec![ArrowField::new("x", DataType::Int32, false)].into()),
            false,
        ))
        .unwrap();
        struct_field.set_id(-1, &mut 0);

        let schema = Schema {
            fields: vec![struct_field],
            metadata: Default::default(),
        };

        let data_file = DataFile::new("data.lance", vec![0, 1], vec![0, 1], 2, 0, None, None);
        let manifest = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_0);
        assert!(check_column_indices(&manifest).is_ok());
    }

    #[test]
    fn test_check_column_indices_rejects_mismatched_lengths() {
        // fields and column_indices must have the same length.
        let mut leaf_field = Field::try_from(ArrowField::new("x", DataType::Int32, false)).unwrap();
        leaf_field.set_id(-1, &mut 0);

        let schema = Schema {
            fields: vec![leaf_field],
            metadata: Default::default(),
        };

        // 1 field id but 2 column indices
        let data_file = DataFile::new("data.lance", vec![0], vec![0, 1], 2, 1, None, None);
        let manifest = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_1);
        let result = check_column_indices(&manifest);
        assert!(result.is_err(), "Expected error for mismatched lengths");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("1 field ids but 2 column indices"), "{msg}");
    }

    #[test]
    fn test_check_column_indices_skips_unknown_field_id() {
        // A field id not present in the schema is skipped (schema evolution).
        let mut leaf_field = Field::try_from(ArrowField::new("x", DataType::Int32, false)).unwrap();
        leaf_field.set_id(-1, &mut 0);

        let schema = Schema {
            fields: vec![leaf_field],
            metadata: Default::default(),
        };

        // field id 99 does not exist in the schema — should be skipped
        let data_file = DataFile::new("data.lance", vec![0, 99], vec![0, 1], 2, 1, None, None);
        let manifest = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_1);
        assert!(check_column_indices(&manifest).is_ok());
    }

    #[test]
    fn test_check_column_indices_rejects_leaf_with_negative_one() {
        // A leaf field with column_index=-1 in v2.1 should be rejected.
        let mut struct_field = Field::try_from(ArrowField::new(
            "s",
            DataType::Struct(vec![ArrowField::new("x", DataType::Int32, false)].into()),
            false,
        ))
        .unwrap();
        struct_field.set_id(-1, &mut 0);

        let schema = Schema {
            fields: vec![struct_field],
            metadata: Default::default(),
        };

        // struct=-1 (correct), but leaf=-1 (wrong — leaf must have a real column)
        let data_file = DataFile::new("data.lance", vec![0, 1], vec![-1, -1], 2, 1, None, None);
        let manifest = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_1);
        let result = check_column_indices(&manifest);
        assert!(
            result.is_err(),
            "Expected error for leaf with column_index=-1"
        );
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("must have a valid column index"), "{msg}");
    }
}
