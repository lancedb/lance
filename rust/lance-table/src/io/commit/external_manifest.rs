// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Trait for external manifest handler.
//!
//! This trait abstracts an external storage with put_if_not_exists semantics.

use std::sync::Arc;

use async_trait::async_trait;
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStoreExt;
use log::warn;
use object_store::{path::Path, ObjectStore};
use snafu::{location, Location};

use super::{
    current_manifest_path, make_staging_manifest_path, manifest_path, write_latest_manifest,
    MANIFEST_EXTENSION,
};
use crate::format::{Index, Manifest};
use crate::io::commit::{parse_version_from_path, CommitError, CommitHandler, ManifestWriter};

/// External manifest store
///
/// This trait abstracts an external storage for source of truth for manifests.
/// The storge is expected to remember (uri, version) -> manifest_path
/// and able to run transactions on the manifest_path.
///
/// This trait is called an **External** manifest store because the store is
/// expected to work in tandem with the object store. We are only leveraging
/// the external store for concurrent commit. Any manifest committed thru this
/// trait should ultimately be materialized in the object store.
/// For a visual explaination of the commit loop see
/// https://github.com/lancedb/lance/assets/12615154/b0822312-0826-432a-b554-3965f8d48d04
#[async_trait]
pub trait ExternalManifestStore: std::fmt::Debug + Send + Sync {
    /// Get the manifest path for a given base_uri and version
    async fn get(&self, base_uri: &str, version: u64) -> Result<String>;

    /// Get the latest version of a dataset at the base_uri, and the path to the manifest.
    /// The path is provided as an optimization. The path is deterministic based on
    /// the version and the store should not customize it.
    async fn get_latest_version(&self, base_uri: &str) -> Result<Option<(u64, String)>>;

    /// Put the manifest path for a given base_uri and version, should fail if the version already exists
    async fn put_if_not_exists(&self, base_uri: &str, version: u64, path: &str) -> Result<()>;

    /// Put the manifest path for a given base_uri and version, should fail if the version **does not** already exist
    async fn put_if_exists(&self, base_uri: &str, version: u64, path: &str) -> Result<()>;
}

/// External manifest commit handler
/// This handler is used to commit a manifest to an external store
/// for detailed design, see https://github.com/lancedb/lance/issues/1183
#[derive(Debug)]
pub struct ExternalManifestCommitHandler {
    pub external_manifest_store: Arc<dyn ExternalManifestStore>,
}

#[async_trait]
impl CommitHandler for ExternalManifestCommitHandler {
    /// Get the latest version of a dataset at the path
    async fn resolve_latest_version(
        &self,
        base_path: &Path,
        object_store: &dyn ObjectStore,
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
                // path is not finalized yet, we should try to finalize the path before loading
                // if sync/finalize fails, return error
                //
                // step 1: copy path -> object_store_manifest_path
                let object_store_manifest_path = manifest_path(base_path, version);
                let manifest_path = Path::parse(path)?;
                let staging = make_staging_manifest_path(&manifest_path)?;
                // TODO: remove copy-rename once we upgrade object_store crate
                object_store.copy(&manifest_path, &staging).await?;
                object_store
                    .rename(&staging, &object_store_manifest_path)
                    .await?;

                // step 2: write _latest.manifest
                write_latest_manifest(&manifest_path, base_path, object_store).await?;

                // step 3: update external store to finalize path
                self.external_manifest_store
                    .put_if_exists(
                        base_path.as_ref(),
                        version,
                        object_store_manifest_path.as_ref(),
                    )
                    .await?;

                Ok(object_store_manifest_path)
            }
            // Dataset not found in the external store, this could be because the dataset did not
            // use external store for commit before. In this case, we search for the latest manifest
            None => current_manifest_path(object_store, base_path).await,
        }
    }

    async fn resolve_latest_version_id(
        &self,
        base_path: &Path,
        object_store: &dyn ObjectStore,
    ) -> std::result::Result<u64, Error> {
        let version = self
            .external_manifest_store
            .get_latest_version(base_path.as_ref())
            .await?;

        match version {
            Some((version, _)) => Ok(version),
            None => parse_version_from_path(&current_manifest_path(object_store, base_path).await?),
        }
    }

    async fn resolve_version(
        &self,
        base_path: &Path,
        version: u64,
        object_store: &dyn ObjectStore,
    ) -> std::result::Result<Path, Error> {
        let path_res = self
            .external_manifest_store
            .get(base_path.as_ref(), version)
            .await;

        let path = match path_res {
            Ok(p) => p,
            // not board external manifest yet, direct to object store
            Err(Error::NotFound { .. }) => {
                let path = manifest_path(base_path, version);
                // if exist update external manifest store
                if object_store.exists(&path).await? {
                    // best effort put, if it fails, it's okay
                    match self
                        .external_manifest_store
                        .put_if_not_exists(base_path.as_ref(), version, path.as_ref())
                        .await
                    {
                        Ok(_) => {}
                        Err(e) => {
                            warn!("could up update external manifest store during load, with error: {}", e);
                        }
                    }
                } else {
                    return Err(Error::NotFound {
                        uri: path.to_string(),
                        location: location!(),
                    });
                }
                return Ok(manifest_path(base_path, version));
            }
            Err(e) => return Err(e),
        };

        // finalized path, just return
        if path.ends_with(&format!(".{MANIFEST_EXTENSION}")) {
            return Ok(Path::parse(path)?);
        }

        let manifest_path = manifest_path(base_path, version);
        let staging_path = make_staging_manifest_path(&manifest_path)?;

        // step1: try to materialize the manifest from external store to object store
        // multiple writers could try to copy at the same time, this is okay
        // as the content is immutable and copy is atomic
        // We can't use `copy_if_not_exists` here because not all store supports it
        object_store
            .copy(&Path::parse(path)?, &staging_path)
            .await?;
        object_store.rename(&staging_path, &manifest_path).await?;

        // finalize the external store
        self.external_manifest_store
            .put_if_exists(base_path.as_ref(), version, manifest_path.as_ref())
            .await?;

        Ok(manifest_path)
    }

    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<Index>>,
        base_path: &Path,
        object_store: &dyn object_store::ObjectStore,
        manifest_writer: ManifestWriter,
    ) -> std::result::Result<(), CommitError> {
        // path we get here is the path to the manifest we want to write
        // use object_store.base_path.as_ref() for getting the root of the dataset

        // step 1: Write the manifest we want to commit to object store with a temporary name
        let path = manifest_path(base_path, manifest.version);
        let staging_path = make_staging_manifest_path(&path)?;
        manifest_writer(object_store, manifest, indices, &staging_path).await?;

        // step 2 & 3: Try to commit this version to external store, return err on failure
        // TODO: add logic to clean up orphaned staged manifests, the ones that failed to commit to external store
        // https://github.com/lancedb/lance/issues/1201
        self.external_manifest_store
            .put_if_not_exists(base_path.as_ref(), manifest.version, staging_path.as_ref())
            .await
            .map_err(|_| CommitError::CommitConflict {})?;

        // step 4: copy the manifest to the final location
        object_store.copy(
            &staging_path,
            &path,
        ).await.map_err(|e| CommitError::OtherError(
            Error::io(
                format!("commit to external store is successful, but could not copy manifest to object store, with error: {}.", e),
                location!(),
            )
        ))?;

        // update the _latest.manifest pointer
        write_latest_manifest(&path, base_path, object_store).await?;

        // step 5: flip the external store to point to the final location
        self.external_manifest_store
            .put_if_exists(base_path.as_ref(), manifest.version, path.as_ref())
            .await?;

        Ok(())
    }
}
