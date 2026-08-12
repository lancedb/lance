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
/// This trait abstracts a concurrency coordinator and lookup index for
/// manifests. The store is expected to remember
/// `(uri, version) -> manifest_path` and to atomically select one staging path
/// for each version. The manifest bytes in object storage remain authoritative.
///
/// This trait is called an **External** manifest store because the store is
/// expected to work in tandem with the object store. We are only leveraging
/// the external store for concurrent commit. Any manifest committed thru this
/// trait should ultimately be materialized in the object store.
///
/// # Correctness model
///
/// 1. Writers first upload immutable manifests to unique staging paths.
/// 2. `put_if_not_exists` linearizes `(dataset, version)` and records exactly
///    one winning staging path. A writer that loses this operation must never
///    materialize its own staging object at the final path.
/// 3. The winner, or any helping reader, copies the recorded staging object to
///    the deterministic final path. Successful final-path materialization is
///    the durable commit point. Repeating this step is content-idempotent
///    because every helper reads the same immutable source selected in step 2.
/// 4. The external row is then compacted from staging to final path and staging
///    is deleted. These are repair and garbage-collection operations: failures
///    leave enough information for another helper and cannot undo step 3.
///
/// Object-store overwrites can assign a new ETag to identical bytes. Therefore
/// the finalized external row contains stable logical metadata (path and size)
/// but omits the destination ETag. Readers get the current opaque generation
/// token from object storage for cache scoping. This protocol assumes one
/// dataset incarnation owns the physical prefix; safe prefix reuse requires a
/// separate incarnation identity rather than interpreting an ETag as one.
/// For a visual explanation of the commit loop see
/// <https://github.com/lance-format/lance/assets/12615154/b0822312-0826-432a-b554-3965f8d48d04>
#[async_trait]
pub trait ExternalManifestStore: std::fmt::Debug + Send + Sync {
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

        // Step 2: Copy staging to final path
        let final_path = naming_scheme.manifest_path(base_path, version);
        let copied = match copy_size_aware(object_store, staging_path, &final_path, size).await {
            Ok(_) => true,
            Err(ObjectStoreError::NotFound { .. }) => false,
            Err(e) => return Err(e.into()),
        };
        if copied {
            info!(target: TRACE_FILE_AUDIT, mode=AUDIT_MODE_CREATE, r#type=AUDIT_TYPE_MANIFEST, path = final_path.as_ref());
        }

        let (final_size, final_e_tag) =
            observe_final_manifest(object_store, &final_path, version, size, copied).await?;

        let location = ManifestLocation {
            version,
            path: final_path.clone(),
            size: Some(final_size),
            naming_scheme,
            e_tag: final_e_tag,
        };

        // Step 3: Update the external index to the final path.
        //
        // Do not persist the destination ETag. The external store has already
        // linearized the logical commit by selecting exactly one immutable
        // staging path in step 1. Every correct finalizer therefore copies the
        // same bytes to the deterministic final path. A repeated overwrite may
        // still create a new physical object generation (and, for S3 Express,
        // a new opaque random ETag), so publishing an ETag would let the DDB
        // update race independently of the content it indexes:
        //
        //   A: COPY -> HEAD(E1)             -> publish E1
        //   B:          COPY -> HEAD(E2) -> publish E2
        //
        // The final bytes are identical in either order, but DDB can retain E1
        // while object storage exposes E2. Omitting the physical generation
        // makes finalization idempotent: every finalizer publishes the same
        // `(path, size, no-etag)` tuple. Readers obtain the current ETag from
        // object storage and use it only to scope caches.
        let published = self
            .put_if_exists(
                base_path.as_ref(),
                version,
                final_path.as_ref(),
                final_size,
                None,
            )
            .await;

        if let Err(error) = published {
            // The canonical object is already durable and is the commit point.
            // Keep staging so an old or new reader that still observes the
            // reservation can retry this cache/index update. A DDB failure must
            // not turn an S3-committed transaction into a reported conflict.
            warn!(
                "Final manifest '{}' is committed, but the external manifest index could not be updated; retaining staging manifest '{}' for repair: {}",
                final_path, staging_path, error
            );
            return Ok(location);
        }

        // Step 4: Delete staging manifest
        match object_store.delete(staging_path).await {
            Ok(_) => {}
            Err(ObjectStoreError::NotFound { .. }) => {}
            Err(error) => {
                // Staging is no longer authoritative after the canonical
                // object and final index entry exist. Its deletion is garbage
                // collection and cannot roll back the commit.
                warn!(
                    "Failed to delete finalized staging manifest '{}': {}",
                    staging_path, error
                );
                return Ok(location);
            }
        }
        info!(target: TRACE_FILE_AUDIT, mode=AUDIT_MODE_DELETE, r#type=AUDIT_TYPE_MANIFEST, path = staging_path.as_ref());

        Ok(location)
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

/// Observe metadata for a canonical manifest after a finalization attempt.
///
/// A successful whole-object copy already proves that the selected immutable
/// bytes were materialized at `final_path`. The following HEAD is useful for a
/// defensive size check and for obtaining a cache-scoping generation, but a
/// transient HEAD failure cannot undo that completed copy. In that case the
/// caller continues with the selected size and no cache generation.
///
/// When `copied` is false, the source disappeared before this helper could copy
/// it. Another helper is only *suspected* to have completed finalization, so a
/// successful HEAD and exact size match are required before accepting the
/// canonical object.
async fn observe_final_manifest(
    object_store: &dyn OSObjectStore,
    final_path: &Path,
    version: u64,
    selected_size: u64,
    copied: bool,
) -> Result<(u64, Option<String>)> {
    match object_store.head(final_path).await {
        Ok(final_meta) if final_meta.size == selected_size => {
            Ok((final_meta.size, final_meta.e_tag))
        }
        Ok(final_meta) => Err(Error::corrupt_file(
            final_path.clone(),
            format!(
                "Manifest size mismatch for version {}: selected staging manifest had {}, object store returned {}",
                version, selected_size, final_meta.size
            ),
        )),
        Err(error) if copied => {
            warn!(
                "Final manifest '{}' was copied successfully, but its metadata could not be read; continuing without a cache generation: {}",
                final_path, error
            );
            Ok((selected_size, None))
        }
        Err(error) => Err(error.into()),
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
                    // An external-store ETag is deliberately ignored. It can
                    // only describe a physical generation observed by an old
                    // finalizer and has no bearing on the logical manifest.
                    e_tag: _,
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

                // Always return the current object-store token solely for
                // cache scoping. Size remains the stable cross-store check;
                // content integrity comes from manifest decoding or an
                // explicit checksum, never from an opaque ETag comparison.
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

    /// Recording the staging path in the external store reserves the version
    /// for one immutable manifest. The commit becomes authoritative when those
    /// bytes are materialized at the deterministic final object-store path.
    /// Updating the external row to that final path and deleting staging are
    /// repair and garbage-collection steps. They may be completed by any number
    /// of readers or writers and must not roll back an already materialized
    /// canonical manifest.
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
        // step 1: copy the manifest to the final location
        let final_manifest_path = naming_scheme.manifest_path(base_path, version);

        let copied =
            match copy_size_aware(store, staging_manifest_path, &final_manifest_path, size).await {
                Ok(_) => true,
                Err(ObjectStoreError::NotFound { .. }) => false, // Another writer beat us to it.
                Err(e) => return Err(e.into()),
            };
        if copied {
            info!(target: TRACE_FILE_AUDIT, mode=AUDIT_MODE_CREATE, r#type=AUDIT_TYPE_MANIFEST, path = final_manifest_path.as_ref());
        }

        let (final_size, final_e_tag) =
            observe_final_manifest(store, &final_manifest_path, version, size, copied).await?;

        let location = ManifestLocation {
            version,
            path: final_manifest_path,
            size: Some(final_size),
            naming_scheme,
            e_tag: final_e_tag,
        };

        // Step 2: point the external index at the final location. As in
        // `ExternalManifestStore::put`, intentionally omit the destination
        // ETag. DDB selected `staging_manifest_path` before any finalizer got
        // here, so concurrent finalizers all copy the same immutable bytes.
        // Path and size are stable logical metadata; an ETag belongs to one
        // physical overwrite and is allowed to change independently.
        let published = self
            .external_manifest_store
            .put_if_exists(
                base_path.as_ref(),
                version,
                location.path.as_ref(),
                final_size,
                None,
            )
            .await;

        if let Err(error) = published {
            // The canonical object is the data authority. Retaining staging
            // lets another helper repair the external index without making
            // this successfully materialized commit appear to have failed.
            warn!(
                "Final manifest '{}' is committed, but the external manifest index could not be updated; retaining staging manifest '{}' for repair: {}",
                location.path, staging_manifest_path, error
            );
            return Ok(location);
        }

        // step 3: delete the staging manifest
        match store.delete(staging_manifest_path).await {
            Ok(_) => {}
            Err(ObjectStoreError::NotFound { .. }) => {}
            Err(error) => {
                warn!(
                    "Failed to delete finalized staging manifest '{}': {}",
                    staging_manifest_path, error
                );
                return Ok(location);
            }
        }
        info!(target: TRACE_FILE_AUDIT, mode=AUDIT_MODE_DELETE, r#type=AUDIT_TYPE_MANIFEST, path = staging_manifest_path.as_ref());

        Ok(location)
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
                                None,
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
    use lance_core::datatypes::Schema;
    use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
    use lance_file::version::LanceFileVersion;
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
    struct TestExternalManifestStore {
        manifests: Mutex<HashMap<(String, u64), StoredManifest>>,
        fail_next_put_response: AtomicBool,
        fail_next_final_publish: AtomicBool,
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
                fail_next_final_publish: AtomicBool::new(false),
                block_first_final_publish: false,
                final_publish_calls: AtomicUsize::new(0),
                first_final_publish_started: Notify::new(),
                release_first_final_publish: Notify::new(),
            }
        }

        fn failing_final_publish_once() -> Self {
            Self {
                fail_next_final_publish: AtomicBool::new(true),
                ..Self::new(false)
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
            if self.fail_next_final_publish.swap(false, Ordering::SeqCst) {
                return Err(Error::io("simulated final index update failure"));
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
    async fn test_finalized_manifest_ignores_external_store_etag() {
        let external_store = Arc::new(TestExternalManifestStore::new(false));
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: external_store.clone(),
        };
        let object_store = ObjectStore::memory();
        let base_path = Path::from("dataset");
        let final_path = ManifestNamingScheme::V2.manifest_path(&base_path, 1);

        object_store
            .inner
            .put(
                &final_path,
                object_store::PutPayload::from_static(b"manifest"),
            )
            .await
            .unwrap();
        let final_meta = object_store.inner.head(&final_path).await.unwrap();

        // Rows written by older Lance versions can contain the ETag observed
        // by a finalizer before a concurrent identical overwrite. The physical
        // token may be stale even though DDB still identifies the right logical
        // version and S3 contains the right manifest bytes.
        external_store
            .put_if_not_exists(
                base_path.as_ref(),
                1,
                final_path.as_ref(),
                final_meta.size,
                Some("legacy-stale-etag".to_string()),
            )
            .await
            .unwrap();

        let resolved = handler
            .resolve_version_location(&base_path, 1, object_store.inner.as_ref())
            .await
            .expect("an opaque ETag mismatch must not make valid manifest bytes unreadable");
        assert_eq!(resolved.path, final_path);
        assert_eq!(resolved.size, Some(final_meta.size));
        assert_eq!(
            resolved.e_tag, final_meta.e_tag,
            "cache identity must use the current object-store generation"
        );
    }

    #[tokio::test]
    async fn test_finalized_manifest_size_mismatch_remains_corruption() {
        let external_store = Arc::new(TestExternalManifestStore::new(false));
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: external_store.clone(),
        };
        let object_store = ObjectStore::memory();
        let base_path = Path::from("dataset");
        let final_path = ManifestNamingScheme::V2.manifest_path(&base_path, 1);

        object_store
            .inner
            .put(
                &final_path,
                object_store::PutPayload::from_static(b"manifest"),
            )
            .await
            .unwrap();
        let final_meta = object_store.inner.head(&final_path).await.unwrap();
        external_store
            .put_if_not_exists(
                base_path.as_ref(),
                1,
                final_path.as_ref(),
                final_meta.size + 1,
                None,
            )
            .await
            .unwrap();

        let error = handler
            .resolve_version_location(&base_path, 1, object_store.inner.as_ref())
            .await
            .expect_err("copies of the selected staging object must preserve its size");
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("Manifest size mismatch"));
    }

    #[tokio::test]
    async fn test_canonical_manifest_commits_before_index_repair() {
        let external_store = Arc::new(TestExternalManifestStore::failing_final_publish_once());
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: external_store.clone(),
        };
        let object_store = ObjectStore::memory();
        let base_path = Path::from("dataset");
        let mut manifest = test_manifest();
        let version = manifest.version;
        let final_path = ManifestNamingScheme::V2.manifest_path(&base_path, version);

        let committed = handler
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
            .expect("a failed index update must not overturn a canonical S3 commit");
        assert_eq!(committed.path, final_path);
        object_store
            .inner
            .head(&final_path)
            .await
            .expect("the canonical manifest is the durable commit point");

        let pending = external_store
            .get_manifest_location(base_path.as_ref(), version)
            .await
            .unwrap();
        assert_ne!(pending.path, final_path);
        object_store
            .inner
            .head(&pending.path)
            .await
            .expect("staging must remain until the external index is repaired");

        let repaired = handler
            .resolve_version_location(&base_path, version, object_store.inner.as_ref())
            .await
            .expect("a reader must be able to repair the pending external index");
        assert_eq!(repaired.path, final_path);
        let indexed = external_store
            .get_manifest_location(base_path.as_ref(), version)
            .await
            .unwrap();
        assert_eq!(indexed.path, final_path);
        assert_eq!(indexed.size, repaired.size);
        assert_eq!(
            indexed.e_tag, None,
            "the repaired index must not retain a physical object generation"
        );
        let staging_error = object_store
            .inner
            .head(&pending.path)
            .await
            .expect_err("repair should garbage-collect the retained staging object");
        assert!(matches!(staging_error, ObjectStoreError::NotFound { .. }));
    }

    #[tokio::test]
    async fn test_concurrent_finalizers_publish_generation_independent_metadata() {
        let external_store = Arc::new(TestExternalManifestStore::blocking_first_final_publish());
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: external_store.clone(),
        };
        let object_store = ObjectStore::memory();
        let base_path = Path::from("dataset");
        let version = 1;
        let final_path = ManifestNamingScheme::V2.manifest_path(&base_path, version);
        let staging_path = make_staging_manifest_path(&final_path).unwrap();
        let manifest_bytes = Bytes::from_static(b"immutable manifest bytes");

        object_store
            .inner
            .put(&staging_path, manifest_bytes.clone().into())
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
                    ManifestNamingScheme::V2,
                )
                .await
        });

        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            external_store.first_final_publish_started.notified(),
        )
        .await
        .expect("the direct finalizer should pause after COPY");

        let first_generation = object_store.inner.head(&final_path).await.unwrap();

        // The writer created generation E1. While its final index update is
        // paused, a reader observes the DDB-selected staging path and performs
        // the same immutable copy, producing generation E2. Each helper returns
        // the generation it observed for local cache scoping, but neither
        // publishes that token. Both copies have exactly the same bytes; only
        // their physical object generations differ.
        let reader_location = handler
            .resolve_version_location(&base_path, version, object_store.inner.as_ref())
            .await
            .unwrap();

        external_store.release_first_final_publish.notify_one();
        let writer_location = writer.await.unwrap().unwrap();
        let final_meta = object_store.inner.head(&final_path).await.unwrap();
        let final_bytes = object_store
            .inner
            .get(&final_path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let indexed = external_store
            .get_manifest_location(base_path.as_ref(), version)
            .await
            .unwrap();

        assert_eq!(final_bytes, manifest_bytes);
        assert_ne!(
            first_generation.e_tag, final_meta.e_tag,
            "the deterministic race must create a new physical generation"
        );
        assert_eq!(writer_location.e_tag, first_generation.e_tag);
        assert_eq!(reader_location.e_tag, final_meta.e_tag);
        assert_eq!(indexed.path, final_path);
        assert_eq!(indexed.size, Some(final_meta.size));
        assert_eq!(
            indexed.e_tag, None,
            "all finalizers must publish the same generation-independent tuple"
        );

        let resolved = handler
            .resolve_version_location(&base_path, version, object_store.inner.as_ref())
            .await
            .expect("the finalized manifest must remain readable after the race");
        assert_eq!(resolved.e_tag, final_meta.e_tag);
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
    async fn test_successful_copy_commits_when_cache_head_fails() {
        let external_store = Arc::new(TestExternalManifestStore::new(false));
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: external_store.clone(),
        };
        let mut object_store = ObjectStore::memory();
        let base_path = Path::from("dataset");
        let mut manifest = test_manifest();
        let version = manifest.version;
        let final_path = ManifestNamingScheme::V2.manifest_path(&base_path, version);

        let fail_final_head = Arc::new(AtomicBool::new(true));
        let policy_fail_final_head = fail_final_head.clone();
        let policy_final_path = final_path.clone();
        let mut policy = ProxyObjectStorePolicy::new();
        policy.set_obj_meta_policy(
            "fail-final-head-once",
            Arc::new(move |method, meta| {
                if method == "head"
                    && meta.location == policy_final_path
                    && policy_fail_final_head.swap(false, Ordering::SeqCst)
                {
                    return Err(Error::io("simulated final HEAD failure"));
                }
                Ok(meta)
            }),
        );
        object_store.inner = Arc::new(ProxyObjectStore::new(
            object_store.inner.clone(),
            Arc::new(Mutex::new(policy)),
        ));

        let committed = handler
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
            .expect("a cache-metadata failure must not overturn a successful canonical copy");
        assert_eq!(committed.path, final_path);
        assert_eq!(
            committed.e_tag, None,
            "the caller must not invent a cache generation when HEAD failed"
        );
        assert!(
            !fail_final_head.load(Ordering::SeqCst),
            "the one-shot final HEAD failure must be consumed"
        );

        let indexed = external_store
            .get_manifest_location(base_path.as_ref(), version)
            .await
            .expect("the external index must advance after the canonical copy");
        assert_eq!(indexed.path, final_path);
        assert_eq!(indexed.e_tag, None);
        object_store
            .inner
            .head(&final_path)
            .await
            .expect("the canonical copy must remain readable");
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
