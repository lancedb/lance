// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
//! [RenameCommitHandler], which writes the manifest to a temporary path, then
//! renames the temporary path to the final path if no object already exists
//! at the final path. This is an atomic operation in most object stores, but
//! not in AWS S3. So for AWS S3, the default commit handler is
//! [UnsafeCommitHandler], which writes the manifest to the final path without
//! any checks.
//!
//! When providing your own commit handler, most often you are implementing in
//! terms of a lock. The trait [CommitLock] can be implemented as a simpler
//! alternative to [CommitHandler].

use std::{fmt::Debug, sync::atomic::AtomicBool};

use crate::dataset::transaction::Transaction;
use crate::dataset::{write_manifest_file, ManifestWriteConfig};
use crate::Dataset;
use crate::{format::Index, format::Manifest};
use futures::future::BoxFuture;
use object_store::path::Path;
use object_store::Error as ObjectStoreError;

use super::ObjectStore;

/// Function that writes the manifest to the object store.
pub type ManifestWriter = for<'a> fn(
    object_store: &'a ObjectStore,
    manifest: &'a mut Manifest,
    indices: Option<Vec<Index>>,
    path: &'a Path,
) -> BoxFuture<'a, crate::Result<()>>;

/// Handle commits that prevent conflicting writes.
///
/// Commit implementations ensure that if there are multiple concurrent writers
/// attempting to write the next version of a table, only one will win. In order
/// to work, all writers must use the same commit handler type.
#[async_trait::async_trait]
pub trait CommitHandler: Debug + Send + Sync {
    /// Commit a manifest to a path.
    ///
    /// This function should return an [CommitError::CommitConflict] if another
    /// transaction has already been committed to the path.
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<Index>>,
        path: &Path,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
    ) -> Result<(), CommitError>;
}

/// Errors that can occur when committing a manifest.
pub enum CommitError {
    /// Another transaction has already been written to the path
    CommitConflict,
    /// Something else went wrong
    OtherError(crate::Error),
}

impl From<crate::Error> for CommitError {
    fn from(e: crate::Error) -> Self {
        Self::OtherError(e)
    }
}

impl From<CommitError> for crate::Error {
    fn from(e: CommitError) -> Self {
        match e {
            CommitError::CommitConflict => Self::Internal {
                message: "Commit conflict".to_string(),
            },
            CommitError::OtherError(e) => e,
        }
    }
}

/// Whether we have issued a warning about using the unsafe commit handler.
static WARNED_ON_UNSAFE_COMMIT: AtomicBool = AtomicBool::new(false);

/// A naive commit implementation that does not prevent conflicting writes.
///
/// This will log a warning the first time it is used.
pub struct UnsafeCommitHandler;

#[async_trait::async_trait]
impl CommitHandler for UnsafeCommitHandler {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<Index>>,
        path: &Path,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
    ) -> Result<(), CommitError> {
        // Log a one-time warning
        if !WARNED_ON_UNSAFE_COMMIT.load(std::sync::atomic::Ordering::Relaxed) {
            WARNED_ON_UNSAFE_COMMIT.store(true, std::sync::atomic::Ordering::Relaxed);
            log::warn!(
                "Using unsafe commit handler. Concurrent writes may result in data loss. \
                 Consider providing a commit handler that prevents conflicting writes."
            );
        }

        // Write the manifest naively
        manifest_writer(object_store, manifest, indices, path).await?;

        Ok(())
    }
}

impl Debug for UnsafeCommitHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnsafeCommitHandler").finish()
    }
}

/// A commit implementation that uses a temporary path and renames the object.
///
/// This only works for object stores that support atomic rename if not exist.
pub struct RenameCommitHandler;

#[async_trait::async_trait]
impl CommitHandler for RenameCommitHandler {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<Index>>,
        path: &Path,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
    ) -> Result<(), CommitError> {
        // Create a temporary object, then use `rename_if_not_exists` to commit.
        // If failed, clean up the temporary object.

        // Add .tmp_ prefix to the path
        let mut parts: Vec<_> = path.parts().collect();
        // Add a UUID to the end of the filename to avoid conflicts
        let uuid = uuid::Uuid::new_v4();
        let new_name = format!(
            ".tmp_{}_{}",
            parts.last().unwrap().as_ref(),
            uuid.as_hyphenated()
        );
        let _ = std::mem::replace(parts.last_mut().unwrap(), new_name.into());
        let tmp_path: Path = parts.into_iter().collect();

        // Write the manifest to the temporary path
        manifest_writer(object_store, manifest, indices, &tmp_path).await?;

        match object_store
            .inner
            .rename_if_not_exists(&tmp_path, path)
            .await
        {
            Ok(_) => Ok(()),
            Err(ObjectStoreError::AlreadyExists { .. }) => {
                // Another transaction has already been committed
                // Attempt to clean up temporary object, but ignore errors if we can't
                let _ = object_store.inner.delete(&tmp_path).await;

                return Err(CommitError::CommitConflict);
            }
            Err(e) => {
                // Something else went wrong
                return Err(CommitError::OtherError(e.into()));
            }
        }
    }
}

impl Debug for RenameCommitHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenameCommitHandler").finish()
    }
}

/// A commit implementation that uses a lock to prevent conflicting writes.
#[async_trait::async_trait]
pub trait CommitLock {
    type Lease: CommitLease;

    /// Attempt to lock the table for the given version.
    ///
    /// If it is already locked by another transaction, wait until it is unlocked.
    /// Once it is unlocked, return [CommitError::CommitConflict] if the version
    /// has already been committed. Otherwise, return the lock.
    ///
    /// To prevent poisoned locks, it's recommended to set a timeout on the lock
    /// of at least 30 seconds.
    ///
    /// It is not required that the lock tracks the version. It is provided in
    /// case the locking is handled by a catalog service that needs to know the
    /// current version of the table.
    async fn lock(&self, version: u64) -> Result<Self::Lease, CommitError>;
}

#[async_trait::async_trait]
pub trait CommitLease: Send + Sync {
    /// Return the lease, indicating whether the commit was successful.
    async fn release(&self, success: bool) -> Result<(), CommitError>;
}

#[async_trait::async_trait]
impl<T: CommitLock + Send + Sync + Debug> CommitHandler for T {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<Index>>,
        path: &Path,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
    ) -> Result<(), CommitError> {
        // NOTE: once we have the lease we cannot use ? to return errors, since
        // we must release the lease before returning.
        let lease = self.lock(manifest.version).await?;

        // Head the location and make sure it's not already committed
        match object_store.inner.head(path).await {
            Ok(_) => {
                // The path already exists, so it's already committed
                // Release the lock
                lease.release(false).await?;

                return Err(CommitError::CommitConflict);
            }
            Err(ObjectStoreError::NotFound { .. }) => {}
            Err(e) => {
                // Something else went wrong
                // Release the lock
                lease.release(false).await?;

                return Err(CommitError::OtherError(e.into()));
            }
        }
        let res = manifest_writer(object_store, manifest, indices, path).await;

        // Release the lock
        lease.release(res.is_ok()).await?;

        res.map_err(|err| err.into())
    }
}

#[derive(Debug, Clone)]
pub struct CommitConfig {
    pub num_retries: u32,
    // TODO: add isolation_level
}

impl Default for CommitConfig {
    fn default() -> Self {
        Self { num_retries: 3 }
    }
}

pub(crate) fn check_transaction(
    transaction: &Transaction,
    other_version: u64,
    other_transaction: &Option<Transaction>,
) -> crate::Result<()> {
    if other_transaction.is_none() {
        return Err(crate::Error::Internal {
            message: format!(
                "There was a conflicting transaction at version {}, \
                and it was missing transaction metadata.",
                other_version
            ),
        });
    }

    if transaction.conflicts_with(other_transaction.as_ref().unwrap()) {
        return Err(crate::Error::CommitConflict {
            version: other_version,
            source: format!(
                "There was a concurrent commit that conflicts with this one and it \
                cannot be automatically resolve. Please rerun the operation off the latest version \
                of the table.\n Transaction: {:?}\n Conflicting Transaction: {:?}",
                transaction, other_transaction
            )
            .into(),
        });
    }

    Ok(())
}

/// Attempt to commit a transaction, with retries and conflict resolution.
pub(crate) async fn commit_transaction(
    object_store: &ObjectStore,
    base_path: &Path,
    transaction: &Transaction,
    indices: Option<Vec<Index>>,
    write_config: &ManifestWriteConfig,
    commit_config: &CommitConfig,
) -> crate::Result<Manifest> {
    let mut current_manifest = Dataset::checkout(base_path.as_ref(), transaction.read_version)
        .await?
        .manifest;
    // First, get all transactions since read_version
    let mut other_transactions = Vec::new();
    let mut version = transaction.read_version;
    loop {
        version += 1;
        match Dataset::checkout(base_path.as_ref(), version).await {
            Ok(dataset) => {
                other_transactions.push(dataset.manifest.transaction()?);
                current_manifest = dataset.manifest;
            }
            Err(crate::Error::NotFound { .. }) => {
                break;
            }
            Err(e) => {
                return Err(e);
            }
        }
    }

    let mut target_version = version + 1;

    // If any of them conflict with the transaction, return an error
    for (version_offset, other_transaction) in other_transactions.iter().enumerate() {
        let other_version = transaction.read_version + version_offset as u64 + 1;
        check_transaction(transaction, other_version, other_transaction)?;
    }

    for _ in 0..commit_config.num_retries {
        // Build an up-to-date manifest from the transaction and current manifest
        let mut manifest = transaction.build_manifest(current_manifest.as_ref(), write_config)?;

        // Try to commit the manifest
        let result = write_manifest_file(
            object_store,
            base_path,
            &mut manifest,
            indices.clone(),
            write_config,
        )
        .await;

        match result {
            Ok(()) => return Ok(manifest),
            Err(crate::Error::CommitConflict { .. }) => {
                // See if we can retry the commit
                current_manifest = Dataset::checkout(base_path.as_ref(), target_version)
                    .await?
                    .manifest;
                check_transaction(
                    transaction,
                    target_version,
                    &current_manifest.transaction()?,
                )?;
                target_version += 1;
            }
            Err(err) => {
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
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};

    use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use futures::future::join_all;

    use super::*;

    use crate::dataset::WriteParams;
    use crate::io::object_store::ObjectStoreParams;
    use crate::Dataset;

    async fn test_commit_handler(handler: Arc<dyn CommitHandler>, should_succeed: bool) {
        // Create a dataset, passing handler as commit handler
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
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
            store_params: Some(ObjectStoreParams {
                commit_handler: Some(handler),
                ..Default::default()
            }),
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
            assert!(
                num_successes > distinct_results.len(),
                "Expected some conflicts. Got {:?}",
                task_results
            );
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

            async fn lock(&self, version: u64) -> Result<Self::Lease, CommitError> {
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
            async fn release(&self, _success: bool) -> Result<(), CommitError> {
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
}
