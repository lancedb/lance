// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! [ManifestStore] encapsulates the logic for reading, writing and managing
//! manifest files. Different implementations may be used, depending on the
//! capabilities of the underlying storage system.

use futures::future::Future;

use futures::stream::BoxStream;
use lance_core::{Error, Result};
use lance_io::traits::Reader;

use crate::format::{Index, Manifest};

pub mod legacy;
#[cfg(feature = "dynamodb")]
pub mod dynamodb;

const MANIFEST_EXTENSION: &str = "manifest";

/// A store of manifests. This provides fast access to the latest version
/// of the dataset and allows for listing and opening older versions.
pub trait ManifestStore {
    /// Get the latest version of the dataset.
    fn latest_version(&self) -> impl Future<Output = Result<ManifestVersion>>;
    
    /// Open the latest manifest file.
    fn open_latest_manifest(&self) -> impl Future<Output = Result<Box<dyn Reader>>>;

    /// Open the manifest file for the given version.
    ///
    /// Should use the provided size if available to avoid an extra HEAD request.
    fn open_manifest(&self, version: impl Into<ManifestVersion>) -> impl Future<Output = Result<Box<dyn Reader>>>;

    /// List all the versions of the dataset.
    ///
    /// This should return them in descending order.
    fn list_versions(&self) -> BoxStream<Result<ManifestVersion>>;

    /// Try to commit the given manifest as the given version.
    ///
    /// If the version already exists, this should return an error, even if
    /// the version was created by a concurrent process.
    ///
    /// Any temporary files created during the commit should be cleaned up
    /// if the commit fails.
    /// 
    /// The `manifest` is mutable because the offsets to certain internal
    /// structures are updated during the writing process.
    fn try_commit(
        &self,
        manifest: &mut Manifest,
        indices: &[Index],
    ) -> impl Future<Output = std::result::Result<(), CommitError>>;

    // TODO: what about cleanup?
}

pub struct ManifestVersion {
    version: u64,
    known_size: Option<u64>,
}

impl From<u64> for ManifestVersion {
    fn from(version: u64) -> Self {
        Self {
            version,
            known_size: None,
        }
    }
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
    fn from(e: Error) -> Self {
        Self::OtherError(e)
    }
}

// Goal: make the paths opaque, so that the store implementation can choose how 
// the paths are represented.

// Goal 2: separate idea of commit handler (what happens when we write the manifest)
// from the idea of the store (how we read the manifests). Allow customizing both.

// This is really just a cleaned up version of CommitHandler. We can provide
// an adapter for now.