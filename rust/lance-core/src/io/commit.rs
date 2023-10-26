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

use std::fmt::Debug;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use futures::{
    future::{self, BoxFuture},
    stream::BoxStream,
    StreamExt, TryStreamExt,
};
use object_store::{path::Path, Error as ObjectStoreError, ObjectStore};
use snafu::{location, Location};

#[cfg(feature = "dynamodb")]
pub mod dynamodb;
pub mod external_manifest;

use crate::format::{Index, Manifest};
use crate::io::object_store::ObjectStoreExt;
use crate::{Error, Result};

const LATEST_MANIFEST_NAME: &str = "_latest.manifest";
const VERSIONS_DIR: &str = "_versions";
const MANIFEST_EXTENSION: &str = "manifest";

/// Function that writes the manifest to the object store.
pub type ManifestWriter = for<'a> fn(
    object_store: &'a dyn ObjectStore,
    manifest: &'a mut Manifest,
    indices: Option<Vec<Index>>,
    path: &'a Path,
) -> BoxFuture<'a, Result<()>>;

/// Get the manifest file path for a version.
fn manifest_path(base: &Path, version: u64) -> Path {
    base.child(VERSIONS_DIR)
        .child(format!("{version}.{MANIFEST_EXTENSION}"))
}

pub fn latest_manifest_path(base: &Path) -> Path {
    base.child(LATEST_MANIFEST_NAME)
}

/// Get the latest manifest path
async fn current_manifest_path(object_store: &dyn ObjectStore, base: &Path) -> Result<Path> {
    // TODO: list gives us the size, so we could also return the size of the manifest.
    // That avoids a HEAD request later.

    // We use `list_with_delimiter` to avoid listing the contents of child directories.
    let manifest_files = object_store
        .list_with_delimiter(Some(&base.child(VERSIONS_DIR)))
        .await?;

    let current = manifest_files
        .objects
        .into_iter()
        .map(|meta| meta.location)
        .filter(|path| {
            path.filename().is_some() && path.filename().unwrap().ends_with(MANIFEST_EXTENSION)
        })
        .filter_map(|path| {
            let version = path
                .filename()
                .unwrap()
                .split_once('.')
                .and_then(|(version_str, _)| version_str.parse::<u64>().ok())?;
            Some((version, path))
        })
        .max_by_key(|(version, _)| *version)
        .map(|(_, path)| path);

    if let Some(path) = current {
        Ok(path)
    } else {
        Err(Error::NotFound {
            uri: manifest_path(base, 1).to_string(),
            location: location!(),
        })
    }
}

async fn list_manifests<'a>(
    base_path: &Path,
    object_store: &'a dyn ObjectStore,
) -> Result<BoxStream<'a, Result<Path>>> {
    let base_path = base_path.clone();
    Ok(object_store
        .read_dir_all(&base_path.child(VERSIONS_DIR), None)
        .await?
        .try_filter_map(|obj_meta| {
            if obj_meta.location.extension() == Some("manifest") {
                future::ready(Ok(Some(obj_meta.location)))
            } else {
                future::ready(Ok(None))
            }
        })
        .boxed())
}

pub fn parse_version_from_path(path: &Path) -> Result<u64> {
    path.filename()
        .and_then(|name| name.split_once('.'))
        .filter(|(_, extension)| *extension == "manifest")
        .and_then(|(version, _)| version.parse::<u64>().ok())
        .ok_or(crate::Error::Internal {
            message: format!("Expected manifest file, but found {}", path),
        })
}

fn make_staging_manifest_path(base: &Path) -> Result<Path> {
    let id = uuid::Uuid::new_v4().to_string();
    Path::parse(format!("{base}-{id}")).map_err(|e| crate::Error::IO {
        message: format!("failed to parse path: {}", e),
        location: location!(),
    })
}

async fn write_latest_manifest(
    from_path: &Path,
    base_path: &Path,
    object_store: &dyn ObjectStore,
) -> Result<()> {
    let latest_path = latest_manifest_path(base_path);
    let staging_path = make_staging_manifest_path(from_path)?;
    object_store
        .copy(from_path, &staging_path)
        .await
        .map_err(|err| CommitError::OtherError(err.into()))?;
    object_store.rename(&staging_path, &latest_path).await?;
    Ok(())
}

/// Handle commits that prevent conflicting writes.
///
/// Commit implementations ensure that if there are multiple concurrent writers
/// attempting to write the next version of a table, only one will win. In order
/// to work, all writers must use the same commit handler type.
/// This trait is also responsible for resolving where the manifests live.
///
// TODO: pub(crate)
#[async_trait::async_trait]
pub trait CommitHandler: Debug + Send + Sync {
    /// Get the path to the latest version manifest of a dataset at the base_path
    async fn resolve_latest_version(
        &self,
        base_path: &Path,
        object_store: &dyn ObjectStore,
    ) -> std::result::Result<Path, crate::Error> {
        // TODO: we need to pade 0's to the version number on the manifest file path
        Ok(current_manifest_path(object_store, base_path).await?)
    }

    // for default implementation, parse the version from the path
    async fn resolve_latest_version_id(
        &self,
        base_path: &Path,
        object_store: &dyn ObjectStore,
    ) -> Result<u64> {
        let path = self.resolve_latest_version(base_path, object_store).await?;

        parse_version_from_path(&path)
    }

    /// Get the path to a specific versioned manifest of a dataset at the base_path
    async fn resolve_version(
        &self,
        base_path: &Path,
        version: u64,
        _object_store: &dyn ObjectStore,
    ) -> std::result::Result<Path, crate::Error> {
        Ok(manifest_path(base_path, version))
    }

    /// List manifests that are available for a dataset at the base_path
    async fn list_manifests<'a>(
        &self,
        base_path: &Path,
        object_store: &'a dyn ObjectStore,
    ) -> Result<BoxStream<'a, Result<Path>>> {
        list_manifests(base_path, object_store).await
    }

    /// Commit a manifest.
    ///
    /// This function should return an [CommitError::CommitConflict] if another
    /// transaction has already been committed to the path.
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<Index>>,
        base_path: &Path,
        object_store: &dyn ObjectStore,
        manifest_writer: ManifestWriter,
    ) -> std::result::Result<(), CommitError>;
}

/// Errors that can occur when committing a manifest.
#[derive(Debug)]
pub enum CommitError {
    /// Another transaction has already been written to the path
    CommitConflict,
    /// Something else went wrong
    OtherError(Error),
}

impl From<Error> for CommitError {
    fn from(e: crate::Error) -> Self {
        Self::OtherError(e)
    }
}

impl From<CommitError> for Error {
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
        base_path: &Path,
        object_store: &dyn ObjectStore,
        manifest_writer: ManifestWriter,
    ) -> std::result::Result<(), CommitError> {
        // Log a one-time warning
        if !WARNED_ON_UNSAFE_COMMIT.load(std::sync::atomic::Ordering::Relaxed) {
            WARNED_ON_UNSAFE_COMMIT.store(true, std::sync::atomic::Ordering::Relaxed);
            log::warn!(
                "Using unsafe commit handler. Concurrent writes may result in data loss. \
                 Consider providing a commit handler that prevents conflicting writes."
            );
        }

        let version_path = self
            .resolve_version(base_path, manifest.version, object_store)
            .await?;
        // Write the manifest naively
        manifest_writer(object_store, manifest, indices, &version_path).await?;

        write_latest_manifest(&version_path, base_path, object_store).await?;

        Ok(())
    }
}

impl Debug for UnsafeCommitHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnsafeCommitHandler").finish()
    }
}

/// A commit implementation that uses a lock to prevent conflicting writes.
#[async_trait::async_trait]
pub trait CommitLock: Debug {
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
    async fn lock(&self, version: u64) -> std::result::Result<Self::Lease, CommitError>;
}

#[async_trait::async_trait]
pub trait CommitLease: Send + Sync {
    /// Return the lease, indicating whether the commit was successful.
    async fn release(&self, success: bool) -> std::result::Result<(), CommitError>;
}

#[async_trait::async_trait]
impl<T: CommitLock + Send + Sync> CommitHandler for T {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<Index>>,
        base_path: &Path,
        object_store: &dyn ObjectStore,
        manifest_writer: ManifestWriter,
    ) -> std::result::Result<(), CommitError> {
        let path = self
            .resolve_version(base_path, manifest.version, object_store)
            .await?;
        // NOTE: once we have the lease we cannot use ? to return errors, since
        // we must release the lease before returning.
        let lease = self.lock(manifest.version).await?;

        // Head the location and make sure it's not already committed
        match object_store.head(&path).await {
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
        let res = manifest_writer(object_store, manifest, indices, &path).await;

        write_latest_manifest(&path, base_path, object_store).await?;

        // Release the lock
        lease.release(res.is_ok()).await?;

        res.map_err(|err| err.into())
    }
}

#[async_trait::async_trait]
impl<T: CommitLock + Send + Sync> CommitHandler for Arc<T> {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<Index>>,
        base_path: &Path,
        object_store: &dyn ObjectStore,
        manifest_writer: ManifestWriter,
    ) -> std::result::Result<(), CommitError> {
        self.as_ref()
            .commit(manifest, indices, base_path, object_store, manifest_writer)
            .await
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
        base_path: &Path,
        object_store: &dyn ObjectStore,
        manifest_writer: ManifestWriter,
    ) -> std::result::Result<(), CommitError> {
        // Create a temporary object, then use `rename_if_not_exists` to commit.
        // If failed, clean up the temporary object.

        let path = self
            .resolve_version(base_path, manifest.version, object_store)
            .await?;

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

        let res = match object_store.rename_if_not_exists(&tmp_path, &path).await {
            Ok(_) => Ok(()),
            Err(ObjectStoreError::AlreadyExists { .. }) => {
                // Another transaction has already been committed
                // Attempt to clean up temporary object, but ignore errors if we can't
                let _ = object_store.delete(&tmp_path).await;

                return Err(CommitError::CommitConflict);
            }
            Err(e) => {
                // Something else went wrong
                return Err(CommitError::OtherError(e.into()));
            }
        };

        write_latest_manifest(&path, base_path, object_store).await?;

        res
    }
}

impl Debug for RenameCommitHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenameCommitHandler").finish()
    }
}

#[derive(Debug, Clone)]
pub struct CommitConfig {
    pub num_retries: u32,
    // TODO: add isolation_level
}

impl Default for CommitConfig {
    fn default() -> Self {
        Self { num_retries: 5 }
    }
}
