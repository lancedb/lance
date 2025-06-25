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
//! The trait [CommitHandler] can be implemented to provide different commit
//! strategies. The default implementation for most object stores is
//! [ConditionalPutCommitHandler], which writes the manifest to a temporary path, then
//! renames the temporary path to the final path if no object already exists
//! at the final path.
//!
//! When providing your own commit handler, most often you are implementing in
//! terms of a lock. The trait [CommitLock] can be implemented as a simpler
//! alternative to [CommitHandler].

use std::collections::{HashMap, HashSet};
use std::num::NonZero;
use std::sync::Arc;
use std::time::Instant;

use conflict_resolver::TransactionRebase;
use lance_core::cache::LanceCache;
use lance_core::utils::backoff::{Backoff, SlotBackoff};
use lance_core::utils::mask::RowIdTreeMap;
use lance_file::version::LanceFileVersion;
use lance_index::metrics::NoOpMetricsCollector;
use lance_io::utils::CachedFileSize;
use lance_table::format::{
    is_detached_version, pb, DataStorageFormat, DeletionFile, Fragment, Index, Manifest,
    WriterVersion, DETACHED_VERSION_MASK,
};
use lance_table::io::commit::{
    CommitConfig, CommitError, CommitHandler, ManifestLocation, ManifestNamingScheme,
};
use rand::{thread_rng, Rng};
use snafu::location;

use futures::future::Either;
use futures::{StreamExt, TryFutureExt, TryStreamExt};
use lance_core::{Error, Result};
use lance_index::DatasetIndexExt;
use log;
use object_store::path::Path;
use prost::Message;

use super::ObjectStore;
use crate::dataset::cleanup::auto_cleanup_hook;
use crate::dataset::fragment::FileFragment;
use crate::dataset::transaction::{Operation, Transaction};
use crate::dataset::{
    load_new_transactions, write_manifest_file, ManifestWriteConfig, NewTransactionResult, BLOB_DIR,
};
use crate::index::DatasetIndexInternalExt;
use crate::io::deletion::read_dataset_deletion_file;
use crate::Dataset;

mod conflict_resolver;
#[cfg(all(feature = "dynamodb_tests", test))]
mod dynamodb;
#[cfg(test)]
mod external_manifest;
#[cfg(all(feature = "dynamodb_tests", test))]
mod s3_test;

/// Read the transaction data from a transaction file.
pub(crate) async fn read_transaction_file(
    object_store: &ObjectStore,
    base_path: &Path,
    transaction_file: &str,
) -> Result<Transaction> {
    let path = base_path.child("_transactions").child(transaction_file);
    let result = object_store.inner.get(&path).await?;
    let data = result.bytes().await?;
    let transaction = pb::Transaction::decode(data)?;
    transaction.try_into()
}

pub(crate) fn transaction_file_cache_key(version: u64) -> String {
    format!("txn/{version}")
}

pub(crate) fn manifest_cache_key(location: &ManifestLocation) -> String {
    if let Some(e_tag) = &location.e_tag {
        format!("manifest/{}/{}", location.version, e_tag)
    } else {
        format!("manifest/{}", location.version)
    }
}

pub(crate) fn deletion_file_cache_key(fragment_id: u64, deletion_file: &DeletionFile) -> String {
    format!(
        "deletion/{}/{}/{}/{}",
        fragment_id,
        deletion_file.read_version,
        deletion_file.id,
        deletion_file.file_type.suffix()
    )
}

/// Write a transaction to a file and return the relative path.
async fn write_transaction_file(
    object_store: &ObjectStore,
    base_path: &Path,
    transaction: &Transaction,
) -> Result<String> {
    let file_name = format!("{}-{}.txn", transaction.read_version, transaction.uuid);
    let path = base_path.child("_transactions").child(file_name.as_str());

    let message = pb::Transaction::from(transaction);
    let buf = message.encode_to_vec();
    object_store.inner.put(&path, buf.into()).await?;

    Ok(file_name)
}

#[allow(clippy::too_many_arguments)]
async fn do_commit_new_dataset(
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    base_path: &Path,
    transaction: &Transaction,
    write_config: &ManifestWriteConfig,
    manifest_naming_scheme: ManifestNamingScheme,
    blob_version: Option<u64>,
    metadata_cache: &LanceCache,
) -> Result<(Manifest, ManifestLocation)> {
    let transaction_file = write_transaction_file(object_store, base_path, transaction).await?;

    let (mut manifest, indices) =
        transaction.build_manifest(None, vec![], &transaction_file, write_config, blob_version)?;

    manifest.blob_dataset_version = blob_version;

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
    )
    .await;

    // TODO: Allow Append or Overwrite mode to retry using `commit_transaction`
    // if there is a conflict.
    match result {
        Ok(manifest_location) => {
            metadata_cache.insert(
                &transaction_file_cache_key(manifest.version),
                Arc::new(transaction.clone()),
            );
            metadata_cache.insert(
                &manifest_cache_key(&manifest_location),
                Arc::new(manifest.clone()),
            );
            Ok((manifest, manifest_location))
        }
        Err(CommitError::CommitConflict) => Err(crate::Error::DatasetAlreadyExists {
            uri: base_path.to_string(),
            location: location!(),
        }),
        Err(CommitError::OtherError(err)) => Err(err),
    }
}

pub(crate) async fn commit_new_dataset(
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    base_path: &Path,
    transaction: &Transaction,
    write_config: &ManifestWriteConfig,
    manifest_naming_scheme: ManifestNamingScheme,
    metadata_cache: &LanceCache,
) -> Result<(Manifest, ManifestLocation)> {
    let blob_version = if let Some(blob_op) = transaction.blobs_op.as_ref() {
        let blob_path = base_path.child(BLOB_DIR);
        let blob_tx = Transaction::new(0, blob_op.clone(), None, None);
        let (blob_manifest, _) = do_commit_new_dataset(
            object_store,
            commit_handler,
            &blob_path,
            &blob_tx,
            write_config,
            manifest_naming_scheme,
            None,
            metadata_cache,
        )
        .await?;
        Some(blob_manifest.version)
    } else {
        None
    };

    do_commit_new_dataset(
        object_store,
        commit_handler,
        base_path,
        transaction,
        write_config,
        manifest_naming_scheme,
        blob_version,
        metadata_cache,
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
pub fn manifest_needs_migration(manifest: &Manifest, indices: &[Index]) -> bool {
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
            Fragment::try_infer_version(&manifest.fragments).map_err(|e| Error::Internal {
                message: format!(
                    "The dataset contains a mixture of file versions.  You will need to rollback to an earlier version: {}",
                    e
                ),
                location: location!(),
            })? {
                if actual_file_version > data_storage_version {
                    log::warn!(
                        "Data storage version {} is less than the actual file version {}.  This has been automatically updated.",
                        data_storage_version,
                        actual_file_version
                    );
                    manifest.data_storage_format = DataStorageFormat::new(actual_file_version);
                }
            }
    } else {
        // Otherwise, if we are on 2.0 or greater, we should ensure that the file versions
        // match the data storage version.  This is a sanity assertion to prevent data corruption.
        if let Some(actual_file_version) = Fragment::try_infer_version(&manifest.fragments)? {
            if actual_file_version != data_storage_version {
                return Err(Error::Internal {
                    message: format!(
                        "The operation added files with version {}.  However, the data storage version is {}.",
                        actual_file_version,
                        data_storage_version
                    ),
                    location: location!(),
                });
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
    let mut field_id_seed = manifest.max_field_id() + 1;
    let mut old_field_id_mapping: HashMap<i32, i32> = HashMap::new();
    let mut fields_with_duplicate_ids = fields_with_duplicate_ids.into_iter().collect::<Vec<_>>();
    fields_with_duplicate_ids.sort_unstable();
    for field_id in fields_with_duplicate_ids {
        old_field_id_mapping.insert(field_id, field_id_seed);
        field_id_seed += 1;
    }

    let mut fragments = manifest.fragments.as_ref().clone();

    // Apply mapping to fragment files list
    // We iterate over files in reverse order so that we only map the last field id
    seen_fields.clear();
    for fragment in fragments.iter_mut() {
        for field_id in fragment
            .files
            .iter_mut()
            .rev()
            .flat_map(|file| file.fields.iter_mut())
        {
            if let Some(new_field_id) = old_field_id_mapping.get(field_id) {
                if seen_fields.insert(*field_id) {
                    *field_id = *new_field_id;
                }
            }
        }
        seen_fields.clear();
    }

    // Apply mapping to the schema
    for (old_field_id, new_field_id) in &old_field_id_mapping {
        let field = manifest.schema.mut_field_by_id(*old_field_id).unwrap();
        field.id = *new_field_id;

        if let Some(local_field) = manifest.local_schema.mut_field_by_id(*old_field_id) {
            local_field.id = *new_field_id;
        }
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

            // For each of the data files in the fragment, we need to get the file size
            let object_store = dataset.object_store();
            let get_sizes = data_files
                .iter()
                .map(|file| {
                    if let Some(size) = file.file_size_bytes.get() {
                        Either::Left(futures::future::ready(Ok(size)))
                    } else {
                        Either::Right(async {
                            object_store
                                .size(&dataset.base.child("data").child(file.path.clone()))
                                .map_ok(|size| {
                                    NonZero::new(size).ok_or_else(|| Error::Internal {
                                        message: format!("File {} has size 0", file.path),
                                        location: location!(),
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

fn must_recalculate_fragment_bitmap(index: &Index, version: Option<&WriterVersion>) -> bool {
    // If the fragment bitmap was written by an old version of lance then we need to recalculate
    // it because it could be corrupt due to a bug in versions < 0.8.15
    index.fragment_bitmap.is_none() || version.map(|v| v.older_than(0, 8, 15)).unwrap_or(true)
}

/// Update indices with new fields.
///
/// Indices might be missing `fragment_bitmap`, so this function will add it.
async fn migrate_indices(dataset: &Dataset, indices: &mut [Index]) -> Result<()> {
    let needs_recalculating = match detect_overlapping_fragments(indices) {
        Ok(()) => vec![],
        Err(BadFragmentBitmapError { bad_indices }) => {
            bad_indices.into_iter().map(|(name, _)| name).collect()
        }
    };
    for index in indices {
        if needs_recalculating.contains(&index.name)
            || must_recalculate_fragment_bitmap(index, dataset.manifest.writer_version.as_ref())
        {
            debug_assert_eq!(index.fields.len(), 1);
            let idx_field = dataset.schema().field_by_id(index.fields[0]).ok_or_else(|| Error::Internal { message: format!("Index with uuid {} referred to field with id {} which did not exist in dataset", index.uuid, index.fields[0]), location: location!() })?;
            // We need to calculate the fragments covered by the index
            let idx = dataset
                .open_generic_index(
                    &idx_field.name,
                    &index.uuid.to_string(),
                    &NoOpMetricsCollector,
                )
                .await?;
            index.fragment_bitmap = Some(idx.calculate_included_frags().await?);
        }
        // We can't reliably recalculate the index type for label_list and bitmap indices and so we can't migrate this field.
        // However, we still log for visibility and to help potentially diagnose issues in the future if we grow to rely on the field.
        if index.index_details.is_none() {
            log::debug!("the index with uuid {} is missing index metadata.  This probably means it was written with Lance version <= 0.19.2.  This is not a problem.", index.uuid);
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
    indices: &[Index],
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
    new_blob_version: Option<u64>,
) -> Result<(Manifest, ManifestLocation)> {
    // We don't strictly need a transaction file but we go ahead and create one for
    // record-keeping if nothing else.
    let transaction_file = write_transaction_file(object_store, &dataset.base, transaction).await?;

    // We still do a loop since we may have conflicts in the random version we pick
    let mut backoff = Backoff::default();
    while backoff.attempt() < commit_config.num_retries {
        // Pick a random u64 with the highest bit set to indicate it is detached
        let random_version = thread_rng().gen::<u64>() | DETACHED_VERSION_MASK;

        let (mut manifest, mut indices) = match transaction.operation {
            Operation::Restore { version } => {
                Transaction::restore_old_manifest(
                    object_store,
                    commit_handler,
                    &dataset.base,
                    version,
                    write_config,
                    &transaction_file,
                )
                .await?
            }
            _ => transaction.build_manifest(
                Some(dataset.manifest.as_ref()),
                dataset.load_indices().await?.as_ref().clone(),
                &transaction_file,
                write_config,
                new_blob_version,
            )?,
        };

        manifest.version = random_version;

        // recompute_stats is always false so far because detached manifests are newer than
        // the old stats bug.
        migrate_manifest(dataset, &mut manifest, /*recompute_stats=*/ false).await?;
        // fix_schema and check_storage_version are just for sanity-checking and consistency
        fix_schema(&mut manifest)?;
        check_storage_version(&mut manifest)?;
        migrate_indices(dataset, &mut indices).await?;

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
                // If other error, return
                return Err(err);
            }
        }
    }

    // This should be extremely unlikely.  There should not be *that* many detached commits.  If
    // this happens then it seems more likely there is a bug in our random u64 generation.
    Err(crate::Error::CommitConflict {
        version: 0,
        source: format!(
            "Failed find unused random u64 after {} retries.",
            commit_config.num_retries
        )
        .into(),
        location: location!(),
    })
}

pub(crate) async fn commit_detached_transaction(
    dataset: &Dataset,
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    transaction: &Transaction,
    write_config: &ManifestWriteConfig,
    commit_config: &CommitConfig,
) -> Result<(Manifest, ManifestLocation)> {
    let new_blob_version = if let Some(blob_op) = transaction.blobs_op.as_ref() {
        let blobs_dataset = dataset.blobs_dataset().await?.unwrap();
        let blobs_tx =
            Transaction::new(blobs_dataset.version().version, blob_op.clone(), None, None);
        let (blobs_manifest, _) = do_commit_detached_transaction(
            blobs_dataset.as_ref(),
            object_store,
            commit_handler,
            &blobs_tx,
            write_config,
            commit_config,
            None,
        )
        .await?;
        Some(blobs_manifest.version)
    } else {
        None
    };

    do_commit_detached_transaction(
        dataset,
        object_store,
        commit_handler,
        transaction,
        write_config,
        commit_config,
        new_blob_version,
    )
    .await
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
    affected_rows: Option<&RowIdTreeMap>,
) -> Result<(Manifest, ManifestLocation)> {
    let new_blob_version = if let Some(blob_op) = transaction.blobs_op.as_ref() {
        let blobs_dataset = dataset.blobs_dataset().await?.unwrap();
        let blobs_tx =
            Transaction::new(blobs_dataset.version().version, blob_op.clone(), None, None);
        let (blobs_manifest, _) = do_commit_detached_transaction(
            blobs_dataset.as_ref(),
            object_store,
            commit_handler,
            &blobs_tx,
            write_config,
            commit_config,
            None,
        )
        .await?;
        Some(blobs_manifest.version)
    } else {
        None
    };

    // Note: object_store has been configured with WriteParams, but dataset.object_store()
    // has not necessarily. So for anything involving writing, use `object_store`.
    let read_version = transaction.read_version;
    let mut target_version = read_version + 1;
    let original_dataset = dataset.clone();

    // read_version sometimes defaults to zero for overwrite.
    // If num_retries is zero, we are in "strict overwrite" mode.
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

    let mut transaction = transaction.clone();

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
        (dataset, other_transactions) = {
            // Load new dataset and other transactions concurrently
            let NewTransactionResult {
                dataset: new_ds,
                new_transactions,
            } = load_new_transactions(&dataset);
            let new_transactions = new_transactions.try_collect::<Vec<_>>();
            futures::future::try_join(new_ds, new_transactions).await?
        };

        // See if we can retry the commit. Try to account for all
        // transactions that have been committed since the read_version.
        // Use small amount of backoff to handle transactions that all
        // started at exact same time better.

        let mut rebase =
            TransactionRebase::try_new(&original_dataset, transaction, affected_rows).await?;

        // Check against committed transactions from oldest to latest
        for (other_version, other_transaction) in other_transactions.iter().rev() {
            rebase.check_txn(other_transaction, *other_version)?;
        }

        transaction = rebase.finish(&dataset).await?;

        let transaction_file =
            write_transaction_file(object_store, &dataset.base, &transaction).await?;

        target_version = dataset.manifest.version + 1;
        if is_detached_version(target_version) {
            return Err(Error::Internal { message: "more than 2^65 versions have been created and so regular version numbers are appearing as 'detached' versions.".into(), location: location!() });
        }
        // Build an up-to-date manifest from the transaction and current manifest
        let (mut manifest, mut indices) = match transaction.operation {
            Operation::Restore { version } => {
                Transaction::restore_old_manifest(
                    object_store,
                    commit_handler,
                    &dataset.base,
                    version,
                    write_config,
                    &transaction_file,
                )
                .await?
            }
            _ => transaction.build_manifest(
                Some(dataset.manifest.as_ref()),
                dataset.load_indices().await?.as_ref().clone(),
                &transaction_file,
                write_config,
                new_blob_version,
            )?,
        };

        manifest.version = target_version;

        let previous_writer_version = &dataset.manifest.writer_version;
        // The versions of Lance prior to when we started writing the writer version
        // sometimes wrote incorrect `Fragment.physical_rows` values, so we should
        // make sure to recompute them.
        // See: https://github.com/lancedb/lance/issues/1531
        let recompute_stats = previous_writer_version.is_none();

        migrate_manifest(&dataset, &mut manifest, recompute_stats).await?;

        fix_schema(&mut manifest)?;

        check_storage_version(&mut manifest)?;

        migrate_indices(&dataset, &mut indices).await?;

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
        )
        .await;

        match result {
            Ok(manifest_location) => {
                // Cache both the transaction file and manifest
                let cache_key = transaction_file_cache_key(target_version);
                dataset
                    .metadata_cache
                    .insert(&cache_key, Arc::new(transaction.clone()));
                dataset.metadata_cache.insert(
                    &manifest_cache_key(&manifest_location),
                    Arc::new(manifest.clone()),
                );
                if !indices.is_empty() {
                    dataset.session().index_cache.insert_metadata(
                        dataset.base.as_ref(),
                        target_version,
                        Arc::new(indices),
                    );
                }

                match auto_cleanup_hook(&dataset, &manifest).await {
                    Ok(Some(stats)) => log::info!("Auto cleanup triggered: {:?}", stats),
                    Err(e) => log::error!("Error encountered during auto_cleanup_hook: {}", e),
                    _ => {}
                };
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
                    tokio::time::sleep(backoff.next_backoff()).await;
                    continue;
                } else {
                    break;
                }
            }
            Err(CommitError::OtherError(err)) => {
                // If other error, return
                return Err(err);
            }
        }
    }

    Err(crate::Error::CommitConflict {
        version: target_version,
        source: format!(
            "Failed to commit the transaction after {} retries.",
            commit_config.num_retries
        )
        .into(),
        location: location!(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use arrow_array::{Int32Array, Int64Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::future::join_all;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::datatypes::{Field, Schema};
    use lance_index::IndexType;
    use lance_linalg::distance::MetricType;
    use lance_table::format::{DataFile, DataStorageFormat};
    use lance_table::io::commit::{
        CommitLease, CommitLock, RenameCommitHandler, UnsafeCommitHandler,
    };
    use lance_testing::datagen::generate_random_array;

    use super::*;

    use crate::dataset::{WriteMode, WriteParams};
    use crate::index::vector::VectorIndexParams;
    use crate::Dataset;

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
            /*blobs_op= */ None,
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

    #[tokio::test]
    async fn test_concurrent_create_index() {
        // Create a table with two vector columns
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

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
        let batches =
            vec![
                RecordBatch::try_new(schema.clone(), vec![vectors.clone(), vectors.clone()])
                    .unwrap(),
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
        for result in results {
            assert!(matches!(result, Ok(Ok(_))), "{:?}", result);
        }

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

        let dataset = dataset.checkout_version(4).await.unwrap();
        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 2);
        let mut fields: Vec<i32> = indices.iter().flat_map(|i| i.fields.clone()).collect();
        fields.sort();
        assert_eq!(fields, vec![0, 1]);
    }

    #[tokio::test]
    async fn test_concurrent_writes() {
        for write_mode in [WriteMode::Append, WriteMode::Overwrite] {
            // Create an empty table
            let test_dir = tempfile::tempdir().unwrap();
            let test_uri = test_dir.path().to_str().unwrap();

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

            // Make some sample data
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
            )
            .unwrap();

            // Write data concurrently in 5 tasks
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
                                mode: write_mode,
                                ..Default::default()
                            }),
                        )
                        .await
                    })
                })
                .collect();
            let results = join_all(futures).await;

            // Assert all succeeded
            for result in results {
                assert!(matches!(result, Ok(Ok(_))), "{:?}", result);
            }

            // Assert final fragments and versions expected
            let dataset = dataset.checkout_version(6).await.unwrap();

            match write_mode {
                WriteMode::Append => {
                    assert_eq!(dataset.get_fragments().len(), 5);
                }
                WriteMode::Overwrite => {
                    assert_eq!(dataset.get_fragments().len(), 1);
                }
                _ => unreachable!(),
            }

            dataset.validate().await.unwrap()
        }
    }

    async fn get_empty_dataset() -> (tempfile::TempDir, Dataset) {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

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
                        .update_config(vec![(key.to_string(), "value".to_string())])
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
                tokio::spawn(async move { dataset.delete_config_keys(&[key]).await })
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
                        .update_config(vec![(key.to_string(), "value".to_string())])
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
                            matches!(&result, &Err(Error::CommitConflict { .. })),
                            "{:?}",
                            result,
                        );
                    }
                }
                1 => match first_operation_failed {
                    true => assert!(result.is_ok(), "{:?}", result),
                    false => {
                        assert!(
                            matches!(&result, &Err(Error::CommitConflict { .. })),
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
                    DataFile::new_legacy_from_fields("path1", vec![0, 1, 2]),
                    DataFile::new_legacy_from_fields("unused", vec![9]),
                ],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: None,
            },
            Fragment {
                id: 1,
                files: vec![
                    DataFile::new_legacy_from_fields("path2", vec![0, 1, 2]),
                    DataFile::new_legacy_from_fields("path3", vec![2]),
                ],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: None,
            },
        ];

        let mut manifest = Manifest::new(
            schema,
            Arc::new(fragments),
            DataStorageFormat::default(),
            /*blob_dataset_version=*/ None,
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
                files: vec![DataFile::new_legacy_from_fields("path1", vec![0, 1, 10])],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: None,
            },
            Fragment {
                id: 1,
                files: vec![
                    DataFile::new_legacy_from_fields("path2", vec![0, 1, 2]),
                    DataFile::new_legacy_from_fields("path3", vec![10]),
                ],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: None,
            },
        ];
        assert_eq!(manifest.fragments.as_ref(), &expected_fragments);
    }
}
