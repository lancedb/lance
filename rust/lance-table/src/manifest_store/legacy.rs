// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
//! # Legacy Manifest Store
//!
//! This module contains the legacy manifest store implementation.
use std::io;

use futures::{stream::BoxStream, StreamExt, TryStreamExt};
use lance_core::{Error, Result};
use lance_io::{object_store::ObjectStore, traits::Reader};
use object_store::path::Path;
use snafu::{location, Location};
use std::{fmt::Debug, fs::DirEntry};

use crate::{format::{Index, Manifest}, io::manifest::write_manifest_file_to_path};

use super::{CommitError, ManifestStore, ManifestVersion, MANIFEST_EXTENSION};

const VERSIONS_DIR: &str = "_versions";

mod commit;

pub use commit::{
    CommitHandler, CommitLease, CommitLock, ManifestWriter, RenameCommitHandler,
    UnsafeCommitHandler,
};

/// A manifest store that stores manifests in the `_versions` directory and
/// uses the `{version}.manifest` file name format.
///
/// It uses a pluggable commit handler to allow for different commit strategies.
pub struct LegacyManifestStore<'a> {
    base: &'a Path,
    object_store: &'a ObjectStore,
    commit_handler: &'a dyn CommitHandler,
}

impl<'a> LegacyManifestStore<'a> {
    pub fn new(
        base: &'a Path,
        object_store: &'a ObjectStore,
        commit_handler: &'a dyn CommitHandler,
    ) -> Self {
        Self {
            base,
            object_store,
            commit_handler,
        }
    }
}

impl<'a> ManifestStore for LegacyManifestStore<'a> {
    async fn latest_version(&self) -> Result<ManifestVersion> {
        let current_path = current_manifest_path(&self.object_store, &self.base).await?;
        Ok(current_path.into())
    }

    async fn open_latest_manifest(&self) -> Result<Box<dyn Reader>> {
        let location = current_manifest_path(&self.object_store, &self.base).await?;
        // By re-using the size from the list operation, we avoid an extra HEAD request.
        if let Some(size) = location.size {
            self.object_store
                .open_with_size(&location.path, size as usize)
                .await
        } else {
            self.object_store.open(&location.path).await
        }
    }

    async fn open_manifest(&self, version: impl Into<ManifestVersion>) -> Result<Box<dyn Reader>> {
        let version = version.into();

        let path = manifest_path(&Path::default(), version.version);
        if let Some(size) = version.known_size {
            self.object_store.open_with_size(&path, size as usize).await
        } else {
            self.object_store.open(&path).await
        }
    }

    fn list_versions(&self) -> BoxStream<Result<ManifestVersion>> {
        // Because of lack of order guarantees, this won't be a true stream.
        // We have to collect all the versions, and then sort.
        let future = async {
            let mut versions = self
                .object_store
                .read_dir_all(&self.base.child(VERSIONS_DIR), None)
                .await?
                .try_filter_map(|obj_meta| {
                    if obj_meta.location.extension() == Some(MANIFEST_EXTENSION) {
                        let version = obj_meta
                            .location
                            .filename()
                            .and_then(|filename| filename.split_once('.'))
                            .and_then(|(version_str, _)| version_str.parse::<u64>().ok());
                        if let Some(version) = version {
                            let version = ManifestVersion {
                                version,
                                known_size: Some(obj_meta.size as u64),
                            };
                            futures::future::ready(Ok(Some(version)))
                        } else {
                            futures::future::ready(Ok(None))
                        }
                    } else {
                        futures::future::ready(Ok(None))
                    }
                })
                .try_collect::<Vec<_>>()
                .await?;
            versions.sort_by_key(|location| location.version);

            Ok(versions.into_iter().rev())
        };

        futures::stream::once(future)
            .flat_map(|res| match res {
                Ok(versions) => futures::stream::iter(versions).map(Ok).boxed(),
                Err(err) => futures::stream::once(futures::future::ready(Err(err))).boxed(),
            })
            .boxed()
    }

    async fn try_commit(
        &self,
        manifest: &mut Manifest,
        indices: &[Index],
    ) -> std::result::Result<(), CommitError> {
        self.commit_handler
            .commit(
                manifest,
                indices,
                self.base,
                self.object_store,
                write_manifest_file_to_path,
            )
            .await
    }
}

/// Get the manifest file path for a version.
pub fn manifest_path(base: &Path, version: u64) -> Path {
    base.child(VERSIONS_DIR)
        .child(format!("{version}.{MANIFEST_EXTENSION}"))
}

// This is an optimized function that searches for the latest manifest. In
// object_store, list operations lookup metadata for each file listed. This
// method only gets the metadata for the found latest manifest.
fn current_manifest_local(base: &Path) -> std::io::Result<Option<ManifestLocation>> {
    let path = lance_io::local::to_local_path(&base.child(VERSIONS_DIR));
    let entries = std::fs::read_dir(path)?;

    let mut latest_entry: Option<(u64, DirEntry)> = None;

    for entry in entries {
        let entry = entry?;
        let filename_raw = entry.file_name();
        let filename = filename_raw.to_string_lossy();
        if !filename.ends_with(MANIFEST_EXTENSION) {
            // Need to ignore temporary files, such as
            // .tmp_7.manifest_9c100374-3298-4537-afc6-f5ee7913666d
            continue;
        }
        let Some(version) = filename
            .split_once('.')
            .and_then(|(version_str, _)| version_str.parse::<u64>().ok())
        else {
            continue;
        };

        if let Some((latest_version, _)) = &latest_entry {
            if version > *latest_version {
                latest_entry = Some((version, entry));
            }
        } else {
            latest_entry = Some((version, entry));
        }
    }

    if let Some((version, entry)) = latest_entry {
        let path = Path::from_filesystem_path(entry.path())
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))?;
        Ok(Some(ManifestLocation {
            version,
            path,
            size: Some(entry.metadata()?.len()),
        }))
    } else {
        Ok(None)
    }
}

#[derive(Debug)]
pub struct ManifestLocation {
    /// The version the manifest corresponds to.
    pub version: u64,
    /// Path of the manifest file, relative to the table root.
    pub path: Path,
    /// Size, in bytes, of the manifest file. If it is not known, this field should be `None`.
    pub size: Option<u64>,
}

impl From<ManifestLocation> for ManifestVersion {
    fn from(location: ManifestLocation) -> Self {
        Self {
            version: location.version,
            known_size: location.size,
        }
    }
}

/// Get the latest manifest path
async fn current_manifest_path(
    object_store: &ObjectStore,
    base: &Path,
) -> Result<ManifestLocation> {
    if object_store.is_local() {
        if let Ok(Some(location)) = current_manifest_local(base) {
            return Ok(location);
        }
    }

    // We use `list_with_delimiter` to avoid listing the contents of child directories.
    let manifest_files = object_store
        .inner
        .list_with_delimiter(Some(&base.child(VERSIONS_DIR)))
        .await?;

    let current = manifest_files
        .objects
        .into_iter()
        .filter(|meta| {
            meta.location.filename().is_some()
                && meta
                    .location
                    .filename()
                    .unwrap()
                    .ends_with(MANIFEST_EXTENSION)
        })
        .filter_map(|meta| {
            let version = meta
                .location
                .filename()
                .unwrap()
                .split_once('.')
                .and_then(|(version_str, _)| version_str.parse::<u64>().ok())?;
            Some((version, meta))
        })
        .max_by_key(|(version, _)| *version);

    if let Some((version, meta)) = current {
        Ok(ManifestLocation {
            version,
            path: meta.location,
            size: Some(meta.size as u64),
        })
    } else {
        Err(Error::NotFound {
            uri: manifest_path(base, 1).to_string(),
            location: location!(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: tests
}
