// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Trait for external manifest handler.
//!
//! This trait abstracts an external storage with put_if_not_exists semantics.

use std::sync::Arc;

use async_trait::async_trait;
use lance_core::{Error, Result};
use lance_io::object_store::{ObjectStore, ObjectStoreExt};
use log::warn;
use object_store::{path::Path, Error as ObjectStoreError, ObjectStore as OSObjectStore};
use snafu::{location, Location};

use super::{
    current_manifest_path, default_resolve_version, make_staging_manifest_path, ManifestLocation,
    ManifestNamingScheme, MANIFEST_EXTENSION,
};
use crate::format::{Index, Manifest};
use crate::io::commit::{CommitError, CommitHandler, ManifestWriter};

/// External manifest store
///
/// This trait abstracts an external storage for source of truth for manifests.
/// The storage is expected to remember (uri, version) -> manifest_path
/// and able to run transactions on the manifest_path.
///
/// This trait is called an **External** manifest store because the store is
/// expected to work in tandem with the object store. We are only leveraging
/// the external store for concurrent commit. Any manifest committed thru this
/// trait should ultimately be materialized in the object store.
/// For a visual explanation of the commit loop see
/// https://github.com/lancedb/lance/assets/12615154/b0822312-0826-432a-b554-3965f8d48d04
#[async_trait]
pub trait ExternalManifestStore: std::fmt::Debug + Send + Sync {
    /// Get the manifest path for a given base_uri and version
    async fn get(&self, base_uri: &str, version: u64) -> Result<String>;

    /// Get the latest version of a dataset at the base_uri, and the path to the manifest.
    /// The path is provided as an optimization. The path is deterministic based on
    /// the version and the store should not customize it.
    async fn get_latest_version(&self, base_uri: &str) -> Result<Option<(u64, String)>>;

    /// Get the latest manifest location for a given base_uri.
    ///
    /// By default, this calls get_latest_version.  Impls should
    /// override this method if they store both the location and size
    /// of the latest manifest.
    async fn get_latest_manifest_location(
        &self,
        base_uri: &str,
    ) -> Result<Option<ManifestLocation>> {
        self.get_latest_version(base_uri).await.and_then(|res| {
            res.map(|(version, uri)| {
                let path = Path::from(uri);
                let naming_scheme = detect_naming_scheme_from_path(&path)?;
                Ok(ManifestLocation {
                    version,
                    path,
                    size: None,
                    naming_scheme,
                })
            })
            .transpose()
        })
    }

    /// Put the manifest path for a given base_uri and version, should fail if the version already exists
    async fn put_if_not_exists(&self, base_uri: &str, version: u64, path: &str) -> Result<()>;

    /// Put the manifest path for a given base_uri and version, should fail if the version **does not** already exist
    async fn put_if_exists(&self, base_uri: &str, version: u64, path: &str) -> Result<()>;

    /// Delete the manifest information for given base_uri from the store
    async fn delete(&self, base_uri: &str) -> Result<()>;
}

fn detect_naming_scheme_from_path(path: &Path) -> Result<ManifestNamingScheme> {
    path.filename()
        .and_then(ManifestNamingScheme::detect_scheme)
        .ok_or_else(|| {
            Error::corrupt_file(
                path.clone(),
                "Path does not follow known manifest naming convention.",
                location!(),
            )
        })
}

/// External manifest commit handler
/// This handler is used to commit a manifest to an external store
/// for detailed design, see https://github.com/lancedb/lance/issues/1183
#[derive(Debug)]
pub struct ExternalManifestCommitHandler {
    pub external_manifest_store: Arc<dyn ExternalManifestStore>,
}

impl ExternalManifestCommitHandler {
    /// The manifest is considered committed once the staging manifest is written
    /// to object store and that path is committed to the external store.
    ///
    /// However, to fully complete this, the staging manifest should be materialized
    /// into the final path, the final path should be committed to the external store
    /// and the staging manifest should be deleted. These steps may be completed
    /// by any number of readers or writers, so care should be taken to ensure
    /// that the manifest is not lost nor any errors occur due to duplicate
    /// operations.
    async fn finalize_manifest(
        &self,
        base_path: &Path,
        staging_manifest_path: &Path,
        version: u64,
        store: &dyn OSObjectStore,
        naming_scheme: ManifestNamingScheme,
    ) -> std::result::Result<Path, Error> {
        // step 1: copy the manifest to the final location
        let final_manifest_path = naming_scheme.manifest_path(base_path, version);
        match store
            .copy(staging_manifest_path, &final_manifest_path)
            .await
        {
            Ok(_) => {}
            Err(ObjectStoreError::NotFound { .. }) => return Ok(final_manifest_path), // Another writer beat us to it.
            Err(e) => return Err(e.into()),
        };

        // step 2: flip the external store to point to the final location
        self.external_manifest_store
            .put_if_exists(base_path.as_ref(), version, final_manifest_path.as_ref())
            .await?;

        // step 3: delete the staging manifest
        match store.delete(staging_manifest_path).await {
            Ok(_) => {}
            Err(ObjectStoreError::NotFound { .. }) => {}
            Err(e) => return Err(e.into()),
        }

        Ok(final_manifest_path)
    }
}

#[async_trait]
impl CommitHandler for ExternalManifestCommitHandler {
    async fn resolve_latest_location(
        &self,
        base_path: &Path,
        object_store: &ObjectStore,
    ) -> std::result::Result<ManifestLocation, Error> {
        let path = self.resolve_latest_version(base_path, object_store).await?;
        let naming_scheme = detect_naming_scheme_from_path(&path)?;
        Ok(ManifestLocation {
            version: self
                .resolve_latest_version_id(base_path, object_store)
                .await?,
            path,
            size: None,
            naming_scheme,
        })
    }

    /// Get the latest version of a dataset at the path
    async fn resolve_latest_version(
        &self,
        base_path: &Path,
        object_store: &ObjectStore,
    ) -> std::result::Result<Path, Error> {
        let version = self
            .external_manifest_store
            .get_latest_version(base_path.as_ref())
            .await?;

        match version {
            Some((version, path)) => {
                // The path is finalized, no need to check object store
                if path.ends_with(&format!(".{MANIFEST_EXTENSION}")) {
                    return Ok(Path::parse(path)?);
                }

                // Detect naming scheme based on presence of zero padding.
                let staged_path = Path::parse(&path)?;
                let naming_scheme =
                    ManifestNamingScheme::detect_scheme_staging(staged_path.filename().unwrap());

                self.finalize_manifest(
                    base_path,
                    &staged_path,
                    version,
                    &object_store.inner,
                    naming_scheme,
                )
                .await
            }
            // Dataset not found in the external store, this could be because the dataset did not
            // use external store for commit before. In this case, we search for the latest manifest
            None => Ok(current_manifest_path(object_store, base_path).await?.path),
        }
    }

    async fn resolve_latest_version_id(
        &self,
        base_path: &Path,
        object_store: &ObjectStore,
    ) -> std::result::Result<u64, Error> {
        let version = self
            .external_manifest_store
            .get_latest_version(base_path.as_ref())
            .await?;

        match version {
            Some((version, _)) => Ok(version),
            None => Ok(current_manifest_path(object_store, base_path)
                .await?
                .version),
        }
    }

    async fn resolve_version(
        &self,
        base_path: &Path,
        version: u64,
        object_store: &dyn OSObjectStore,
    ) -> std::result::Result<Path, Error> {
        let path_res = self
            .external_manifest_store
            .get(base_path.as_ref(), version)
            .await;

        let path = match path_res {
            Ok(p) => p,
            // not board external manifest yet, direct to object store
            Err(Error::NotFound { .. }) => {
                let path = default_resolve_version(base_path, version, object_store)
                    .await
                    .map_err(|_| Error::NotFound {
                        uri: format!("{}@{}", base_path, version),
                        location: location!(),
                    })?
                    .path;
                if object_store.exists(&path).await? {
                    // best effort put, if it fails, it's okay
                    match self
                        .external_manifest_store
                        .put_if_not_exists(base_path.as_ref(), version, path.as_ref())
                        .await
                    {
                        Ok(_) => {}
                        Err(e) => {
                            warn!(
                            "could not update external manifest store during load, with error: {}",
                            e
                        );
                        }
                    }
                    return Ok(path);
                } else {
                    return Err(Error::NotFound {
                        uri: path.to_string(),
                        location: location!(),
                    });
                }
            }
            Err(e) => return Err(e),
        };

        // finalized path, just return
        let current_path = Path::parse(path)?;
        if current_path.extension() == Some(MANIFEST_EXTENSION) {
            return Ok(current_path);
        }

        let naming_scheme =
            ManifestNamingScheme::detect_scheme_staging(current_path.filename().unwrap());

        self.finalize_manifest(
            base_path,
            &Path::parse(&current_path)?,
            version,
            object_store,
            naming_scheme,
        )
        .await
    }

    async fn resolve_version_location(
        &self,
        base_path: &Path,
        version: u64,
        object_store: &dyn OSObjectStore,
    ) -> std::result::Result<ManifestLocation, Error> {
        let path = self
            .resolve_version(base_path, version, object_store)
            .await?;
        let naming_scheme = detect_naming_scheme_from_path(&path)?;
        Ok(ManifestLocation {
            version,
            path,
            size: None,
            naming_scheme,
        })
    }

    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<Index>>,
        base_path: &Path,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
        naming_scheme: ManifestNamingScheme,
    ) -> std::result::Result<(), CommitError> {
        // path we get here is the path to the manifest we want to write
        // use object_store.base_path.as_ref() for getting the root of the dataset

        // step 1: Write the manifest we want to commit to object store with a temporary name
        let path = naming_scheme.manifest_path(base_path, manifest.version);
        let staging_path = make_staging_manifest_path(&path)?;
        manifest_writer(object_store, manifest, indices, &staging_path).await?;

        // step 2 & 3: Try to commit this version to external store, return err on failure
        let res = self
            .external_manifest_store
            .put_if_not_exists(base_path.as_ref(), manifest.version, staging_path.as_ref())
            .await
            .map_err(|_| CommitError::CommitConflict {});

        if res.is_err() {
            // delete the staging manifest
            match object_store.inner.delete(&staging_path).await {
                Ok(_) => {}
                Err(ObjectStoreError::NotFound { .. }) => {}
                Err(e) => return Err(CommitError::OtherError(e.into())),
            }
            return res;
        }

        let scheme = detect_naming_scheme_from_path(&path)?;

        self.finalize_manifest(
            base_path,
            &staging_path,
            manifest.version,
            &object_store.inner,
            scheme,
        )
        .await?;

        Ok(())
    }

    async fn delete(&self, base_path: &Path) -> Result<()> {
        self.external_manifest_store.delete(base_path.as_ref()).await
    }
}
