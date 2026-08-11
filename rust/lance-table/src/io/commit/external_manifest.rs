// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Trait for external manifest handler.
//!
//! This trait abstracts an external storage with put_if_not_exists semantics.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lance_core::utils::tracing::{
    AUDIT_MODE_CREATE, AUDIT_MODE_DELETE, AUDIT_TYPE_MANIFEST, TRACE_FILE_AUDIT,
};
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use log::warn;
use object_store::ObjectMeta;
use object_store::ObjectStoreExt;
use object_store::{Error as ObjectStoreError, ObjectStore as OSObjectStore, path::Path};
use tracing::info;

use super::{
    MANIFEST_EXTENSION, ManifestLocation, ManifestNamingScheme, current_manifest_path,
    default_resolve_version, make_staging_manifest_path, write_version_hint,
};
use crate::format::{IndexMetadata, Manifest, Transaction};
use crate::io::commit::{CommitError, CommitHandler};

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
/// <https://github.com/lance-format/lance/assets/12615154/b0822312-0826-432a-b554-3965f8d48d04>
#[async_trait]
pub trait ExternalManifestStore: std::fmt::Debug + Send + Sync {
    /// Whether the paired object store provides an atomic, synchronous
    /// `copy_if_not_exists` for supported manifest sizes.
    fn use_create_only_manifest_copy(&self) -> bool {
        false
    }

    /// Get the manifest path for a given base_uri and version
    async fn get(&self, base_uri: &str, version: u64) -> Result<String>;

    async fn get_manifest_location(
        &self,
        base_uri: &str,
        version: u64,
    ) -> Result<ManifestLocation> {
        let path = self.get(base_uri, version).await?;
        let path = Path::parse(&path).map_err(|e| Error::invalid_input(e.to_string()))?;
        let naming_scheme = detect_naming_scheme_from_path(&path)?;
        Ok(ManifestLocation {
            version,
            path,
            size: None,
            naming_scheme,
            e_tag: None,
        })
    }

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
                let path = Path::parse(&uri).map_err(|e| Error::invalid_input(e.to_string()))?;
                let naming_scheme = detect_naming_scheme_from_path(&path)?;
                Ok(ManifestLocation {
                    version,
                    path,
                    size: None,
                    naming_scheme,
                    e_tag: None,
                })
            })
            .transpose()
        })
    }

    /// Put the manifest to the external store.
    ///
    /// The staging manifest has been written to `staging_path` on the object store.
    /// This method should atomically claim the version and return the final manifest location.
    ///
    /// The default implementation uses put_if_not_exists and put_if_exists to
    /// implement a staging-based workflow. Implementations that can write directly
    /// (e.g., namespace-backed stores) should override this method.
    #[allow(clippy::too_many_arguments)]
    async fn put(
        &self,
        base_path: &Path,
        version: u64,
        staging_path: &Path,
        size: u64,
        e_tag: Option<String>,
        object_store: &dyn OSObjectStore,
        naming_scheme: ManifestNamingScheme,
    ) -> Result<ManifestLocation> {
        // Default implementation: staging-based workflow

        // Step 1: Record staging path atomically
        self.put_if_not_exists(
            base_path.as_ref(),
            version,
            staging_path.as_ref(),
            size,
            e_tag.clone(),
        )
        .await?;

        // Step 2: Materialize staging at the final path. Stores that opt in use
        // create-only copy for supported manifest sizes.
        let final_path = naming_scheme.manifest_path(base_path, version);
        let materialized = materialize_manifest_create(
            object_store,
            staging_path,
            &final_path,
            size,
            self.use_create_only_manifest_copy(),
        )
        .await?;

        complete_manifest_finalization(
            self,
            base_path,
            version,
            staging_path,
            final_path,
            naming_scheme,
            object_store,
            materialized,
        )
        .await
    }

    /// Put the manifest path for a given base_uri and version, should fail if the version already exists
    async fn put_if_not_exists(
        &self,
        base_uri: &str,
        version: u64,
        path: &str,
        size: u64,
        e_tag: Option<String>,
    ) -> Result<()>;

    /// Put the manifest path for a given base_uri and version, should fail if the version **does not** already exist
    async fn put_if_exists(
        &self,
        base_uri: &str,
        version: u64,
        path: &str,
        size: u64,
        e_tag: Option<String>,
    ) -> Result<()>;

    /// Publish `final_path` only while the row still points at `staging_path`.
    /// Conditional stores should override this with an atomic compare-and-swap.
    #[allow(clippy::too_many_arguments)]
    async fn finalize_staging_manifest(
        &self,
        base_uri: &str,
        version: u64,
        _staging_path: &str,
        final_path: &str,
        size: u64,
        e_tag: Option<String>,
    ) -> Result<()> {
        self.put_if_exists(base_uri, version, final_path, size, e_tag)
            .await
    }

    /// Delete the manifest information for given base_uri from the store
    async fn delete(&self, _base_uri: &str) -> Result<()> {
        Ok(())
    }
}

pub(crate) fn detect_naming_scheme_from_path(path: &Path) -> Result<ManifestNamingScheme> {
    path.filename()
        .and_then(|name| {
            ManifestNamingScheme::detect_scheme(name)
                .or_else(|| Some(ManifestNamingScheme::detect_scheme_staging(name)))
        })
        .ok_or_else(|| {
            Error::corrupt_file(
                path.clone(),
                "Path does not follow known manifest naming convention.",
            )
        })
}

/// The most conservative server-side-copy size limit across the object
/// stores we support. This is not S3-specific: S3's `CopyObject` and GCS's
/// single-shot `Objects: copy` both reject sources above ~5 GiB, so we use
/// 5 GiB as a backend-agnostic threshold. Above it we stream the source
/// through the client and re-upload via multipart instead of relying on a
/// server-side copy. Stores that have no such cap (e.g. local filesystem)
/// also take the fallback above this size — correctness is preserved; only
/// the rare >5 GiB copy is slower than a native copy would be.
const MAX_SERVER_SIDE_COPY_BYTES: u64 = 5 * 1024 * 1024 * 1024;

/// Part size for the read+rewrite fallback. Multipart-capable stores
/// (S3, GCS) require every part except the last to be ≥5 MB and allow up to
/// 10,000 parts. 100 MB sits comfortably inside both bounds and keeps the
/// part count low (~140 parts for a 14 GB manifest) without large per-part
/// RAM.
const COPY_REWRITE_PART_SIZE: usize = 100 * 1024 * 1024;

/// Copy `from` to `to`, falling back to a multipart-equivalent read+rewrite
/// when the source exceeds the server-side-copy size limit
/// (`MAX_SERVER_SIDE_COPY_BYTES`).
///
/// For sources below the limit, this is the same fast server-side
/// `store.copy()` as before. For larger sources, the source is streamed
/// through the client and re-uploaded as a multipart upload at `to`. This
/// doubles bytes-on-the-wire for the rare large case while preserving the
/// cheap fast path for the common small case.
///
/// `size` is the known source size. It is required: the only caller already
/// has it, and the alternative (an extra `head(from)` round-trip) is work
/// the caller can avoid by passing what it already knows.
///
/// `NotFound` errors on `from` propagate unchanged so callers can keep
/// existing `Err(NotFound { .. })` arms.
///
/// This is a workaround for the missing `UploadPartCopy` primitive in the
/// upstream `object_store` crate. Once that lands, this helper can be
/// deleted and the call sites can go back to plain `store.copy()`.
async fn copy_size_aware(
    store: &dyn OSObjectStore,
    from: &Path,
    to: &Path,
    size: u64,
) -> std::result::Result<(), ObjectStoreError> {
    if size < MAX_SERVER_SIDE_COPY_BYTES {
        store.copy(from, to).await
    } else {
        copy_via_read_rewrite(store, from, to).await
    }
}

// NOTE: parts are uploaded sequentially. This could be parallelized (a
// bounded JoinSet, like lance-io/src/object_writer.rs's
// LANCE_UPLOAD_CONCURRENCY) or sidestepped entirely by switching to
// `object_store::WriteMultipart` (which also handles abort-on-drop). Left
// sequential here: this is a cold path (only >5 GiB manifests) and the
// helper is itself a stopgap until `object_store` exposes UploadPartCopy.
async fn copy_via_read_rewrite(
    store: &dyn OSObjectStore,
    from: &Path,
    to: &Path,
) -> std::result::Result<(), ObjectStoreError> {
    // NotFound here propagates upward unchanged.
    let mut stream = store.get(from).await?.into_stream();

    // From here on, errors must `abort()` the upload to avoid leaving an
    // orphan multipart upload on stores that support them (e.g. S3, GCS),
    // which would otherwise incur storage charges until the bucket's
    // lifecycle policy cleans it up.
    //
    // Note: this does NOT cover task cancellation — `MultipartUpload`'s
    // upstream Drop is documented as a no-op for S3/GCS. Callers that
    // need cancellation cleanliness should run this with a guard or
    // switch to `object_store::WriteMultipart` (planned follow-up).
    let mut upload = store.put_multipart(to).await?;
    let mut part_buf: Vec<u8> = Vec::with_capacity(COPY_REWRITE_PART_SIZE);

    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(b) => b,
            Err(e) => {
                let _ = upload.abort().await;
                return Err(e);
            }
        };
        // Append the chunk in COPY_REWRITE_PART_SIZE-bounded slices so a
        // single oversized chunk (e.g., LocalFileSystem returning a whole
        // file) cannot push part_buf past the backend's per-part size limit
        // (5 GiB on S3/GCS). COPY_REWRITE_PART_SIZE is well under every
        // backend's cap, so each flushed part is always valid.
        let mut offset = 0;
        while offset < chunk.len() {
            let want = COPY_REWRITE_PART_SIZE - part_buf.len();
            let take = want.min(chunk.len() - offset);
            part_buf.extend_from_slice(&chunk[offset..offset + take]);
            offset += take;

            if part_buf.len() >= COPY_REWRITE_PART_SIZE {
                let payload =
                    std::mem::replace(&mut part_buf, Vec::with_capacity(COPY_REWRITE_PART_SIZE));
                if let Err(e) = upload.put_part(Bytes::from(payload).into()).await {
                    let _ = upload.abort().await;
                    return Err(e);
                }
            }
        }
    }

    // Flush the final (possibly-short) part. The last part of a multipart
    // upload is exempt from the per-part minimum on S3/GCS.
    if !part_buf.is_empty()
        && let Err(e) = upload.put_part(Bytes::from(part_buf).into()).await
    {
        let _ = upload.abort().await;
        return Err(e);
    }

    if let Err(e) = upload.complete().await {
        let _ = upload.abort().await;
        return Err(e);
    }
    Ok(())
}

/// Range size used to validate a create-only winner without buffering a whole
/// manifest. This is only paid by finalizers that lose a destination create.
const MANIFEST_COMPARE_RANGE_BYTES: u64 = 8 * 1024 * 1024;

enum MaterializeManifestResult {
    Available(ObjectMeta),
    SourceMissing(u64),
}

enum ManifestComparison {
    Match,
    Different,
    SourceMissing,
}

fn validate_materialized_size(
    final_meta: ObjectMeta,
    final_path: &Path,
    staging_path: &Path,
    staging_size: u64,
) -> Result<ObjectMeta> {
    if final_meta.size != staging_size {
        return Err(Error::corrupt_file(
            final_path.clone(),
            format!(
                "Finalized manifest has size {}, expected {} from staging manifest '{}'",
                final_meta.size, staging_size, staging_path
            ),
        ));
    }
    Ok(final_meta)
}

async fn manifests_match(
    store: &dyn OSObjectStore,
    staging_path: &Path,
    final_path: &Path,
    size: u64,
) -> std::result::Result<ManifestComparison, ObjectStoreError> {
    let mut start = 0;
    while start < size {
        let end = (start + MANIFEST_COMPARE_RANGE_BYTES).min(size);
        let staging = match store.get_range(staging_path, start..end).await {
            Ok(staging) => staging,
            Err(ObjectStoreError::NotFound { .. }) => {
                return Ok(ManifestComparison::SourceMissing);
            }
            Err(e) => return Err(e),
        };
        let final_bytes = store.get_range(final_path, start..end).await?;
        if staging != final_bytes {
            return Ok(ManifestComparison::Different);
        }
        start = end;
    }
    Ok(ManifestComparison::Match)
}

async fn resolve_existing_manifest(
    store: &dyn OSObjectStore,
    staging_path: &Path,
    final_path: &Path,
    staging_size: u64,
) -> Result<MaterializeManifestResult> {
    let final_meta = validate_materialized_size(
        store.head(final_path).await?,
        final_path,
        staging_path,
        staging_size,
    )?;

    match manifests_match(store, staging_path, final_path, staging_size).await {
        Ok(ManifestComparison::Match) => Ok(MaterializeManifestResult::Available(final_meta)),
        Ok(ManifestComparison::Different) => Err(Error::corrupt_file(
            final_path.clone(),
            format!(
                "Existing finalized manifest does not match staging manifest '{}'",
                staging_path
            ),
        )),
        Ok(ManifestComparison::SourceMissing) => {
            Ok(MaterializeManifestResult::SourceMissing(staging_size))
        }
        Err(e) => Err(e.into()),
    }
}

async fn resolve_create_error(
    store: &dyn OSObjectStore,
    staging_path: &Path,
    final_path: &Path,
    staging_size: u64,
    create_error: ObjectStoreError,
) -> Result<MaterializeManifestResult> {
    match store.head(final_path).await {
        Ok(_) => resolve_existing_manifest(store, staging_path, final_path, staging_size).await,
        Err(_) => Err(create_error.into()),
    }
}

/// Materialize `staging_path` at `final_path`.
///
/// Stores that are known to support atomic create-only copy use it for
/// manifests within the server-side copy limit. Other stores, and larger
/// manifests, retain the compatible overwrite/read-and-rewrite path and its
/// strict destination ETag validation.
async fn materialize_manifest_create(
    store: &dyn OSObjectStore,
    staging_path: &Path,
    final_path: &Path,
    size: u64,
    use_create_only_copy: bool,
) -> Result<MaterializeManifestResult> {
    if use_create_only_copy && size < MAX_SERVER_SIDE_COPY_BYTES {
        match store.copy_if_not_exists(staging_path, final_path).await {
            Ok(()) => {
                info!(target: TRACE_FILE_AUDIT, mode=AUDIT_MODE_CREATE, r#type=AUDIT_TYPE_MANIFEST, path = final_path.as_ref());
                let final_meta = validate_materialized_size(
                    store.head(final_path).await?,
                    final_path,
                    staging_path,
                    size,
                )?;
                return Ok(MaterializeManifestResult::Available(final_meta));
            }
            Err(ObjectStoreError::AlreadyExists { .. })
            | Err(ObjectStoreError::Precondition { .. }) => {
                return resolve_existing_manifest(store, staging_path, final_path, size).await;
            }
            Err(error @ ObjectStoreError::NotImplemented { .. })
            | Err(error @ ObjectStoreError::NotSupported { .. }) => {
                return Err(error.into());
            }
            Err(error) => {
                return resolve_create_error(store, staging_path, final_path, size, error).await;
            }
        }
    }

    match copy_size_aware(store, staging_path, final_path, size).await {
        Ok(()) => {
            info!(target: TRACE_FILE_AUDIT, mode=AUDIT_MODE_CREATE, r#type=AUDIT_TYPE_MANIFEST, path = final_path.as_ref());
            let final_meta = validate_materialized_size(
                store.head(final_path).await?,
                final_path,
                staging_path,
                size,
            )?;
            Ok(MaterializeManifestResult::Available(final_meta))
        }
        Err(ObjectStoreError::NotFound { .. }) => {
            Ok(MaterializeManifestResult::SourceMissing(size))
        }
        Err(error) => Err(error.into()),
    }
}

#[allow(clippy::too_many_arguments)]
async fn complete_manifest_finalization<S: ExternalManifestStore + ?Sized>(
    external_store: &S,
    base_path: &Path,
    version: u64,
    staging_path: &Path,
    final_path: Path,
    naming_scheme: ManifestNamingScheme,
    object_store: &dyn OSObjectStore,
    materialized: MaterializeManifestResult,
) -> Result<ManifestLocation> {
    let final_meta = match materialized {
        MaterializeManifestResult::Available(meta) => meta,
        MaterializeManifestResult::SourceMissing(staging_size) => {
            let current_meta = validate_materialized_size(
                object_store.head(&final_path).await?,
                &final_path,
                staging_path,
                staging_size,
            )?;
            let published_location = external_store
                .get_manifest_location(base_path.as_ref(), version)
                .await?;
            if published_location.path != final_path
                || published_location.size != Some(current_meta.size)
                || published_location.e_tag != current_meta.e_tag
            {
                return Err(Error::corrupt_file(
                    final_path.clone(),
                    format!(
                        "Staging manifest '{}' is missing, but external metadata for version {} \
                         does not match the finalized object (path '{}', size {:?}, ETag {:?})",
                        staging_path,
                        version,
                        published_location.path,
                        published_location.size,
                        published_location.e_tag,
                    ),
                ));
            }
            return Ok(ManifestLocation {
                version,
                path: final_path,
                size: Some(current_meta.size),
                naming_scheme,
                e_tag: current_meta.e_tag,
            });
        }
    };

    let location = ManifestLocation {
        version,
        path: final_path,
        size: Some(final_meta.size),
        naming_scheme,
        e_tag: final_meta.e_tag.clone(),
    };

    external_store
        .finalize_staging_manifest(
            base_path.as_ref(),
            version,
            staging_path.as_ref(),
            location.path.as_ref(),
            final_meta.size,
            location.e_tag.clone(),
        )
        .await?;

    match object_store.delete(staging_path).await {
        Ok(()) | Err(ObjectStoreError::NotFound { .. }) => {}
        Err(e) => return Err(e.into()),
    }
    info!(target: TRACE_FILE_AUDIT, mode=AUDIT_MODE_DELETE, r#type=AUDIT_TYPE_MANIFEST, path = staging_path.as_ref());

    Ok(location)
}

/// External manifest commit handler
/// This handler is used to commit a manifest to an external store
/// for detailed design, see <https://github.com/lance-format/lance/issues/1183>
#[derive(Debug)]
pub struct ExternalManifestCommitHandler {
    pub external_manifest_store: Arc<dyn ExternalManifestStore>,
}

impl ExternalManifestCommitHandler {
    async fn verify_finalized_manifest_location(
        &self,
        base_path: &Path,
        location: ManifestLocation,
        object_store: &dyn OSObjectStore,
    ) -> std::result::Result<ManifestLocation, Error> {
        match object_store.head(&location.path).await {
            Ok(ObjectMeta { size, e_tag, .. }) => {
                let ManifestLocation {
                    version,
                    path,
                    size: expected_size,
                    naming_scheme,
                    e_tag: expected_e_tag,
                } = location;

                let size = match expected_size {
                    Some(expected_size) if expected_size != size => {
                        return Err(Error::corrupt_file(
                            path,
                            format!(
                                "Manifest size mismatch for version {}: external store expected {}, object store returned {}",
                                version, expected_size, size
                            ),
                        ));
                    }
                    Some(expected_size) => Some(expected_size),
                    None => Some(size),
                };

                let e_tag = match expected_e_tag {
                    Some(expected_e_tag) => {
                        if e_tag.as_ref() != Some(&expected_e_tag) {
                            return Err(Error::corrupt_file(
                                path,
                                format!(
                                    "Manifest e_tag mismatch for version {}: external store expected {:?}, object store returned {:?}",
                                    version, expected_e_tag, e_tag
                                ),
                            ));
                        }
                        Some(expected_e_tag)
                    }
                    None => e_tag,
                };

                Ok(ManifestLocation {
                    version,
                    path,
                    size,
                    naming_scheme,
                    e_tag,
                })
            }
            Err(ObjectStoreError::NotFound { .. }) => {
                // The external store may hold a stale finalized V2 path while
                // the object store still has the manifest at the V1 location.
                default_resolve_version(base_path, location.version, object_store).await
            }
            Err(e) => Err(e.into()),
        }
    }

    /// The manifest is considered committed once the staging manifest is written
    /// to object store and that path is committed to the external store.
    ///
    /// However, to fully complete this, the staging manifest should be materialized
    /// into the final path, the final path should be committed to the external store
    /// and the staging manifest should be deleted. These steps may be completed
    /// by any number of readers or writers, so care should be taken to ensure
    /// that the manifest is not lost nor any errors occur due to duplicate
    /// operations.
    #[allow(clippy::too_many_arguments)]
    async fn finalize_manifest(
        &self,
        base_path: &Path,
        staging_manifest_path: &Path,
        version: u64,
        size: u64,
        store: &dyn OSObjectStore,
        naming_scheme: ManifestNamingScheme,
    ) -> std::result::Result<ManifestLocation, Error> {
        let final_manifest_path = naming_scheme.manifest_path(base_path, version);
        let materialized = materialize_manifest_create(
            store,
            staging_manifest_path,
            &final_manifest_path,
            size,
            self.external_manifest_store.use_create_only_manifest_copy(),
        )
        .await?;

        complete_manifest_finalization(
            self.external_manifest_store.as_ref(),
            base_path,
            version,
            staging_manifest_path,
            final_manifest_path,
            naming_scheme,
            store,
            materialized,
        )
        .await
    }
}

#[async_trait]
impl CommitHandler for ExternalManifestCommitHandler {
    async fn resolve_latest_location(
        &self,
        base_path: &Path,
        object_store: &ObjectStore,
    ) -> std::result::Result<ManifestLocation, Error> {
        let location = self
            .external_manifest_store
            .get_latest_manifest_location(base_path.as_ref())
            .await?;

        match location {
            Some(location) => {
                if location.path.extension() == Some(MANIFEST_EXTENSION) {
                    return self
                        .verify_finalized_manifest_location(
                            base_path,
                            location,
                            object_store.inner.as_ref(),
                        )
                        .await;
                }

                let ManifestLocation {
                    version,
                    path,
                    size,
                    naming_scheme,
                    e_tag: _,
                } = location;

                let size = if let Some(size) = size {
                    size
                } else {
                    match object_store.inner.head(&path).await {
                        Ok(meta) => meta.size,
                        Err(ObjectStoreError::NotFound { .. }) => {
                            // there may be other threads that have finished executing finalize_manifest.
                            let new_location = self
                                .external_manifest_store
                                .get_manifest_location(base_path.as_ref(), version)
                                .await?;
                            return Ok(new_location);
                        }
                        Err(e) => return Err(e.into()),
                    }
                };

                let final_location = self
                    .finalize_manifest(
                        base_path,
                        &path,
                        version,
                        size,
                        &object_store.inner,
                        naming_scheme,
                    )
                    .await?;

                Ok(final_location)
            }
            // Dataset not found in the external store, this could be because the dataset did not
            // use external store for commit before. In this case, we search for the latest manifest
            None => current_manifest_path(object_store, base_path).await,
        }
    }

    async fn resolve_version_location(
        &self,
        base_path: &Path,
        version: u64,
        object_store: &dyn OSObjectStore,
    ) -> std::result::Result<ManifestLocation, Error> {
        let location_res = self
            .external_manifest_store
            .get_manifest_location(base_path.as_ref(), version)
            .await;

        let location = match location_res {
            Ok(p) => p,
            // not board external manifest yet, direct to object store
            Err(Error::NotFound { .. }) => {
                let path = default_resolve_version(base_path, version, object_store)
                    .await
                    .map_err(|_| Error::not_found(format!("{}@{}", base_path, version)))?
                    .path;
                match object_store.head(&path).await {
                    Ok(ObjectMeta { size, e_tag, .. }) => {
                        let res = self
                            .external_manifest_store
                            .put_if_not_exists(
                                base_path.as_ref(),
                                version,
                                path.as_ref(),
                                size,
                                e_tag.clone(),
                            )
                            .await;
                        if let Err(e) = res {
                            warn!(
                                "could not update external manifest store during load, with error: {}",
                                e
                            );
                        }
                        let naming_scheme =
                            ManifestNamingScheme::detect_scheme_staging(path.filename().unwrap());
                        return Ok(ManifestLocation {
                            version,
                            path,
                            size: Some(size),
                            naming_scheme,
                            e_tag,
                        });
                    }
                    Err(ObjectStoreError::NotFound { .. }) => {
                        return Err(Error::not_found(path.to_string()));
                    }
                    Err(e) => return Err(e.into()),
                }
            }
            Err(e) => return Err(e),
        };

        if location.path.extension() == Some(MANIFEST_EXTENSION) {
            return self
                .verify_finalized_manifest_location(base_path, location, object_store)
                .await;
        }

        let naming_scheme =
            ManifestNamingScheme::detect_scheme_staging(location.path.filename().unwrap());

        let size = if let Some(size) = location.size {
            size
        } else {
            let meta = object_store.head(&location.path).await?;
            meta.size
        };

        self.finalize_manifest(
            base_path,
            &location.path,
            version,
            size,
            object_store,
            naming_scheme,
        )
        .await
    }

    async fn version_exists(
        &self,
        base_path: &Path,
        version: u64,
        object_store: &dyn OSObjectStore,
        naming_scheme: ManifestNamingScheme,
    ) -> Result<bool> {
        match self
            .external_manifest_store
            .get_manifest_location(base_path.as_ref(), version)
            .await
        {
            Ok(_) => Ok(true),
            Err(Error::NotFound { .. }) => {
                let path = naming_scheme.manifest_path(base_path, version);
                match object_store.head(&path).await {
                    Ok(_) => Ok(true),
                    Err(ObjectStoreError::NotFound { .. }) => Ok(false),
                    Err(e) => Err(e.into()),
                }
            }
            Err(e) => Err(e),
        }
    }

    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base_path: &Path,
        object_store: &ObjectStore,
        manifest_writer: super::ManifestWriter,
        naming_scheme: ManifestNamingScheme,
        transaction: Option<Transaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        // path we get here is the path to the manifest we want to write
        // use object_store.base_path.as_ref() for getting the root of the dataset

        // step 1: Write the manifest we want to commit to object store with a temporary name
        let path = naming_scheme.manifest_path(base_path, manifest.version);
        let staging_path = make_staging_manifest_path(&path)?;
        let write_res =
            manifest_writer(object_store, manifest, indices, &staging_path, transaction).await?;

        // step 2 & 3: Put the manifest to external store
        let result = self
            .external_manifest_store
            .put(
                base_path,
                manifest.version,
                &staging_path,
                write_res.size as u64,
                write_res.e_tag,
                &object_store.inner,
                naming_scheme,
            )
            .await;

        match result {
            Ok(location) => {
                write_version_hint(object_store, base_path, manifest.version).await;
                Ok(location)
            }
            Err(error) => {
                // A different recorded path proves this staging manifest lost
                // the version and is safe to remove. Otherwise, the external
                // store may have recorded our staging path before its response
                // was lost, so retain it for outcome verification/finalization.
                let recorded_location = self
                    .external_manifest_store
                    .get_manifest_location(base_path.as_ref(), manifest.version)
                    .await;
                if matches!(
                    &recorded_location,
                    Ok(location) if location.path != staging_path
                ) {
                    match object_store.inner.delete(&staging_path).await {
                        Ok(()) => {
                            info!(target: TRACE_FILE_AUDIT, mode=AUDIT_MODE_DELETE, r#type=AUDIT_TYPE_MANIFEST, path = staging_path.as_ref());
                        }
                        Err(ObjectStoreError::NotFound { .. }) => {}
                        Err(delete_error) => {
                            warn!(
                                "Failed to delete losing staging manifest '{}': {}",
                                staging_path, delete_error
                            );
                        }
                    }
                    return Err(CommitError::CommitConflict);
                }
                warn!(
                    "External manifest commit for version {} failed; retaining staging manifest \
                     '{}' until the commit outcome is resolved: {}",
                    manifest.version, staging_path, error
                );
                Err(CommitError::CommitConflict)
            }
        }
    }

    async fn delete(&self, base_path: &Path) -> Result<()> {
        self.external_manifest_store
            .delete(base_path.as_ref())
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::stream::BoxStream;
    use lance_core::datatypes::Schema;
    use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
    use lance_file::version::LanceFileVersion;
    use object_store::{
        CopyMode, CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload,
        PutMultipartOptions, PutOptions, PutPayload, PutResult,
    };
    use tokio::sync::Notify;

    use super::*;
    use crate::format::DataStorageFormat;
    use crate::io::commit::write_manifest_file_to_path;

    #[derive(Debug, Clone)]
    struct StoredManifest {
        path: String,
        size: u64,
        e_tag: Option<String>,
    }

    #[derive(Debug)]
    struct UnsupportedCreateCopyStore {
        target: Arc<dyn OSObjectStore>,
        overwrite_copy_calls: AtomicUsize,
    }

    impl std::fmt::Display for UnsupportedCreateCopyStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "UnsupportedCreateCopyStore({})", self.target)
        }
    }

    #[async_trait]
    impl OSObjectStore for UnsupportedCreateCopyStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            options: PutOptions,
        ) -> object_store::Result<PutResult> {
            self.target.put_opts(location, payload, options).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            options: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            self.target.put_multipart_opts(location, options).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<GetResult> {
            self.target.get_opts(location, options).await
        }

        async fn get_ranges(
            &self,
            location: &Path,
            ranges: &[std::ops::Range<u64>],
        ) -> object_store::Result<Vec<Bytes>> {
            self.target.get_ranges(location, ranges).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            self.target.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.target.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<ListResult> {
            self.target.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: CopyOptions,
        ) -> object_store::Result<()> {
            if options.mode == CopyMode::Create {
                return Err(ObjectStoreError::NotSupported {
                    source: "create-only copy unavailable".into(),
                });
            }
            self.overwrite_copy_calls.fetch_add(1, Ordering::SeqCst);
            self.target.copy_opts(from, to, options).await
        }
    }

    #[derive(Debug)]
    struct TestExternalManifestStore {
        manifests: Mutex<HashMap<(String, u64), StoredManifest>>,
        fail_next_put_response: AtomicBool,
        block_first_final_publish: bool,
        final_publish_calls: AtomicUsize,
        first_final_publish_started: Notify,
        release_first_final_publish: Notify,
    }

    impl TestExternalManifestStore {
        fn new(fail_next_put_response: bool) -> Self {
            Self {
                manifests: Mutex::new(HashMap::new()),
                fail_next_put_response: AtomicBool::new(fail_next_put_response),
                block_first_final_publish: false,
                final_publish_calls: AtomicUsize::new(0),
                first_final_publish_started: Notify::new(),
                release_first_final_publish: Notify::new(),
            }
        }

        fn blocking_first_final_publish() -> Self {
            Self {
                block_first_final_publish: true,
                ..Self::new(false)
            }
        }
    }

    #[async_trait]
    impl ExternalManifestStore for TestExternalManifestStore {
        fn use_create_only_manifest_copy(&self) -> bool {
            self.block_first_final_publish
        }

        async fn get(&self, base_uri: &str, version: u64) -> Result<String> {
            self.manifests
                .lock()
                .unwrap()
                .get(&(base_uri.to_string(), version))
                .map(|manifest| manifest.path.clone())
                .ok_or_else(|| Error::not_found(format!("{base_uri}@{version}")))
        }

        async fn get_manifest_location(
            &self,
            base_uri: &str,
            version: u64,
        ) -> Result<ManifestLocation> {
            let stored = self
                .manifests
                .lock()
                .unwrap()
                .get(&(base_uri.to_string(), version))
                .cloned()
                .ok_or_else(|| Error::not_found(format!("{base_uri}@{version}")))?;
            let path = Path::from(stored.path);
            Ok(ManifestLocation {
                version,
                naming_scheme: detect_naming_scheme_from_path(&path)?,
                path,
                size: Some(stored.size),
                e_tag: stored.e_tag,
            })
        }

        async fn get_latest_version(&self, base_uri: &str) -> Result<Option<(u64, String)>> {
            Ok(self
                .manifests
                .lock()
                .unwrap()
                .iter()
                .filter(|((stored_base, _), _)| stored_base == base_uri)
                .max_by_key(|((_, version), _)| *version)
                .map(|((_, version), manifest)| (*version, manifest.path.clone())))
        }

        async fn put_if_not_exists(
            &self,
            base_uri: &str,
            version: u64,
            path: &str,
            size: u64,
            e_tag: Option<String>,
        ) -> Result<()> {
            let key = (base_uri.to_string(), version);
            let mut manifests = self.manifests.lock().unwrap();
            if manifests.contains_key(&key) {
                return Err(Error::commit_conflict_source(
                    version,
                    "manifest already exists".to_string().into(),
                ));
            }
            manifests.insert(
                key,
                StoredManifest {
                    path: path.to_string(),
                    size,
                    e_tag,
                },
            );
            drop(manifests);
            if self.fail_next_put_response.swap(false, Ordering::SeqCst) {
                Err(Error::io("simulated lost external-store response"))
            } else {
                Ok(())
            }
        }

        async fn put_if_exists(
            &self,
            base_uri: &str,
            version: u64,
            path: &str,
            size: u64,
            e_tag: Option<String>,
        ) -> Result<()> {
            if self.block_first_final_publish
                && self.final_publish_calls.fetch_add(1, Ordering::SeqCst) == 0
            {
                self.first_final_publish_started.notify_one();
                self.release_first_final_publish.notified().await;
            }
            let key = (base_uri.to_string(), version);
            let mut manifests = self.manifests.lock().unwrap();
            let manifest = manifests
                .get_mut(&key)
                .ok_or_else(|| Error::not_found(format!("{base_uri}@{version}")))?;
            *manifest = StoredManifest {
                path: path.to_string(),
                size,
                e_tag,
            };
            Ok(())
        }
    }

    fn test_manifest() -> Manifest {
        let arrow_schema = ArrowSchema::new(vec![ArrowField::new("id", DataType::Int32, false)]);
        Manifest::new(
            Schema::try_from(&arrow_schema).unwrap(),
            Arc::new(vec![]),
            DataStorageFormat::new(LanceFileVersion::Stable.resolve()),
            HashMap::new(),
        )
    }

    #[tokio::test]
    async fn test_lost_external_store_response_retains_staging_manifest() {
        let external_store = Arc::new(TestExternalManifestStore::new(true));
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: external_store.clone(),
        };
        let object_store = ObjectStore::memory();
        let base_path = Path::from("dataset");
        let mut manifest = test_manifest();

        let commit_error = handler
            .commit(
                &mut manifest,
                None,
                &base_path,
                &object_store,
                write_manifest_file_to_path,
                ManifestNamingScheme::V2,
                None,
            )
            .await
            .expect_err("the simulated response loss must be surfaced");
        assert!(matches!(commit_error, CommitError::CommitConflict));

        let staging_path = Path::from(external_store.get("dataset", 1).await.unwrap());
        object_store.inner.head(&staging_path).await.unwrap();

        let resolved = handler
            .resolve_version_location(&base_path, 1, object_store.inner.as_ref())
            .await
            .expect("the retained staging manifest must allow finalization");
        assert_eq!(
            resolved.path,
            ManifestNamingScheme::V2.manifest_path(&base_path, 1)
        );
        object_store.inner.head(&resolved.path).await.unwrap();
    }

    #[tokio::test]
    async fn test_existing_manifest_validation() {
        let object_store = ObjectStore::memory();
        let staging_path = Path::from("dataset/_versions/1.manifest-staging-test");
        let final_path = Path::from("dataset/_versions/1.manifest");
        object_store
            .inner
            .put(
                &staging_path,
                object_store::PutPayload::from_static(b"staging"),
            )
            .await
            .unwrap();

        let missing_final =
            manifests_match(object_store.inner.as_ref(), &staging_path, &final_path, 7).await;
        assert!(matches!(
            missing_final,
            Err(ObjectStoreError::NotFound { .. })
        ));

        object_store
            .inner
            .put(
                &final_path,
                object_store::PutPayload::from_static(b"foreign"),
            )
            .await
            .unwrap();

        let result = materialize_manifest_create(
            object_store.inner.as_ref(),
            &staging_path,
            &final_path,
            7,
            true,
        )
        .await;
        let Err(error) = result else {
            panic!("a different existing destination must not be accepted");
        };
        assert!(
            error
                .to_string()
                .contains("does not match staging manifest"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn test_create_only_copy_does_not_fallback_to_overwrite() {
        let target = ObjectStore::memory().inner;
        let store = UnsupportedCreateCopyStore {
            target: target.clone(),
            overwrite_copy_calls: AtomicUsize::new(0),
        };
        let staging_path = Path::from("dataset/_versions/1.manifest-staging-test");
        let final_path = Path::from("dataset/_versions/1.manifest");
        target
            .put(&staging_path, PutPayload::from_static(b"manifest"))
            .await
            .unwrap();

        let result = materialize_manifest_create(
            &store,
            &staging_path,
            &final_path,
            b"manifest".len() as u64,
            true,
        )
        .await;
        let Err(error) = result else {
            panic!("an opted-in store must not downgrade create-only copy to overwrite");
        };
        assert!(
            matches!(&error, Error::IO { .. }),
            "unexpected error variant: {error:?}"
        );
        assert!(
            error.to_string().contains("Operation not supported"),
            "unexpected error: {error}"
        );
        assert_eq!(store.overwrite_copy_calls.load(Ordering::SeqCst), 0);
        let final_error = target
            .head(&final_path)
            .await
            .expect_err("the destination must remain absent");
        assert!(matches!(final_error, ObjectStoreError::NotFound { .. }));
    }

    #[tokio::test]
    async fn test_source_missing_requires_published_final_location() {
        let external_store = TestExternalManifestStore::new(false);
        let object_store = ObjectStore::memory();
        let base_path = Path::from("dataset");
        let version = 1;
        let staging_path = Path::from("dataset/_versions/1.manifest-staging-test");
        let final_path = ManifestNamingScheme::V2.manifest_path(&base_path, version);
        object_store
            .inner
            .put(&final_path, PutPayload::from_static(b"foreign"))
            .await
            .unwrap();
        let final_meta = object_store.inner.head(&final_path).await.unwrap();
        external_store
            .put_if_not_exists(
                base_path.as_ref(),
                version,
                staging_path.as_ref(),
                final_meta.size,
                Some("staging-etag".to_string()),
            )
            .await
            .unwrap();

        let materialized = materialize_manifest_create(
            object_store.inner.as_ref(),
            &staging_path,
            &final_path,
            final_meta.size,
            true,
        )
        .await
        .unwrap();
        let result = complete_manifest_finalization(
            &external_store,
            &base_path,
            version,
            &staging_path,
            final_path,
            ManifestNamingScheme::V2,
            object_store.inner.as_ref(),
            materialized,
        )
        .await;
        let Err(error) = result else {
            panic!("a foreign final object must not be accepted while metadata is staging");
        };
        assert!(
            matches!(&error, Error::CorruptFile { .. }),
            "unexpected error variant: {error:?}"
        );
        assert!(
            error
                .to_string()
                .contains("does not match the finalized object"),
            "unexpected error: {error}"
        );
        let recorded = external_store
            .get_manifest_location(base_path.as_ref(), version)
            .await
            .unwrap();
        assert_eq!(recorded.path, staging_path);
    }

    #[rstest::rstest]
    #[case(ManifestNamingScheme::V1)]
    #[case(ManifestNamingScheme::V2)]
    #[tokio::test]
    async fn test_concurrent_direct_and_reader_finalization_preserves_winner_metadata(
        #[case] naming_scheme: ManifestNamingScheme,
    ) {
        let external_store = Arc::new(TestExternalManifestStore::blocking_first_final_publish());
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: external_store.clone(),
        };
        let object_store = ObjectStore::memory();
        let base_path = Path::from("dataset");
        let version = 1;
        let final_path = naming_scheme.manifest_path(&base_path, version);
        let staging_path = make_staging_manifest_path(&final_path).unwrap();
        object_store
            .inner
            .put(
                &staging_path,
                object_store::PutPayload::from_static(b"manifest body"),
            )
            .await
            .unwrap();
        let staging_meta = object_store.inner.head(&staging_path).await.unwrap();

        let writer_store = object_store.inner.clone();
        let writer_external_store = external_store.clone();
        let writer_base_path = base_path.clone();
        let writer_staging_path = staging_path.clone();
        let writer_e_tag = staging_meta.e_tag.clone();
        let writer = tokio::spawn(async move {
            writer_external_store
                .put(
                    &writer_base_path,
                    version,
                    &writer_staging_path,
                    staging_meta.size,
                    writer_e_tag,
                    writer_store.as_ref(),
                    naming_scheme,
                )
                .await
        });

        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            external_store.first_final_publish_started.notified(),
        )
        .await
        .expect("direct finalizer should reach metadata publication");

        // The direct writer has copied the destination and read its metadata,
        // but has not published it yet. A reader now observes the staging row
        // and helps finalize the same manifest.
        let reader_result = handler
            .resolve_version_location(&base_path, version, object_store.inner.as_ref())
            .await;

        external_store.release_first_final_publish.notify_one();
        let reader_location = reader_result.unwrap();
        let writer_location = writer.await.unwrap().unwrap();

        let final_meta = object_store.inner.head(&final_path).await.unwrap();
        let stored_location = external_store
            .get_manifest_location(base_path.as_ref(), version)
            .await
            .unwrap();

        for location in [&stored_location, &writer_location, &reader_location] {
            assert_eq!(location.path, final_path);
            assert_eq!(location.size, Some(final_meta.size));
            assert_eq!(
                location.e_tag, final_meta.e_tag,
                "all finalizers must describe the materialized winner"
            );
        }

        handler
            .resolve_version_location(&base_path, version, object_store.inner.as_ref())
            .await
            .expect("a strict reader should accept the finalized manifest");
    }

    #[tokio::test]
    async fn test_copy_failure_after_external_store_commit_retains_staging_manifest() {
        let external_store = Arc::new(TestExternalManifestStore::new(false));
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: external_store.clone(),
        };

        let mut object_store = ObjectStore::memory();
        let fail_next_copy = Arc::new(AtomicBool::new(true));
        let failed_copy_source = Arc::new(Mutex::new(None));
        let mut policy = ProxyObjectStorePolicy::new();
        let policy_fail_next_copy = fail_next_copy.clone();
        let policy_failed_copy_source = failed_copy_source.clone();
        policy.set_before_policy(
            "fail-copy-once",
            Arc::new(move |method, location| {
                if method == "copy" && policy_fail_next_copy.swap(false, Ordering::SeqCst) {
                    *policy_failed_copy_source.lock().unwrap() = Some(location.clone());
                    return Err(Error::io("simulated copy failure"));
                }
                Ok(())
            }),
        );
        let policy = Arc::new(Mutex::new(policy));
        object_store.inner = Arc::new(ProxyObjectStore::new(
            object_store.inner.clone(),
            policy.clone(),
        ));

        let base_path = Path::from("dataset");
        let mut manifest = test_manifest();
        let version = manifest.version;
        let canonical_path = ManifestNamingScheme::V2.manifest_path(&base_path, version);

        let commit_error = handler
            .commit(
                &mut manifest,
                None,
                &base_path,
                &object_store,
                write_manifest_file_to_path,
                ManifestNamingScheme::V2,
                None,
            )
            .await
            .expect_err("the simulated copy failure must be surfaced");
        assert!(matches!(commit_error, CommitError::CommitConflict));
        assert!(
            !fail_next_copy.load(Ordering::SeqCst),
            "the one-shot copy failure must be consumed"
        );

        let recorded_location = external_store
            .get_manifest_location(base_path.as_ref(), version)
            .await
            .expect("the external store must retain the committed staging location");
        let staging_path = failed_copy_source
            .lock()
            .unwrap()
            .clone()
            .expect("the failure must be injected at copy(staging, canonical)");
        assert_eq!(recorded_location.path, staging_path);
        object_store
            .inner
            .head(&staging_path)
            .await
            .expect("the winning staging manifest must be retained");

        let canonical_error = object_store
            .inner
            .head(&canonical_path)
            .await
            .expect_err("copy failed before creating the canonical manifest");
        assert!(
            matches!(canonical_error, ObjectStoreError::NotFound { .. }),
            "unexpected canonical manifest error: {canonical_error}"
        );

        policy.lock().unwrap().clear_before_policy("fail-copy-once");
        let resolved = handler
            .resolve_version_location(&base_path, version, object_store.inner.as_ref())
            .await
            .expect("the retained staging manifest must allow finalization");
        assert_eq!(resolved.path, canonical_path);

        let finalized_location = external_store
            .get_manifest_location(base_path.as_ref(), version)
            .await
            .expect("the external store must publish the canonical location");
        assert_eq!(finalized_location.path, canonical_path);
        object_store
            .inner
            .head(&canonical_path)
            .await
            .expect("the canonical manifest must exist after finalization");

        let staging_error = object_store
            .inner
            .head(&staging_path)
            .await
            .expect_err("successful finalization must clean up the staging manifest");
        assert!(
            matches!(staging_error, ObjectStoreError::NotFound { .. }),
            "unexpected staging manifest error: {staging_error}"
        );
    }
}
