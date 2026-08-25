// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Trait for external manifest handler.
//!
//! This trait abstracts an external storage with put_if_not_exists semantics.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use futures::{StreamExt, TryFutureExt, TryStreamExt, future, stream};
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
    CANDIDATES_DIR, MANIFEST_EXTENSION, ManifestLocation, ManifestNamingScheme,
    PredecessorIdentity, current_manifest_path, default_resolve_version, list_manifests,
    make_staging_manifest_path, write_version_hint,
};
use crate::format::{IndexMetadata, Manifest, Transaction};
use crate::io::commit::{CommitError, CommitHandler};

/// A store record as `(version, path, size)`.
pub type VersionRecord = (u64, String, Option<u64>);

/// Where a predecessor-conditioned commit publishes its manifest: under
/// [`CANDIDATES_DIR`] in a directory unique to this attempt, with the
/// canonical file name so its naming scheme still parses. No other attempt
/// or incarnation can name the object, and generic `_versions` listings
/// never see it; the store's record is the only way to reach it.
fn candidate_manifest_path(
    base: &Path,
    naming_scheme: ManifestNamingScheme,
    version: u64,
) -> Result<Path> {
    let canonical = naming_scheme.manifest_path(base, version);
    let filename = canonical
        .filename()
        .ok_or_else(|| Error::io("canonical manifest path has no file name"))?;
    let attempt = uuid::Uuid::new_v4().to_string();
    Ok(base
        .clone()
        .join(CANDIDATES_DIR)
        .join(attempt.as_str())
        .join(filename))
}

/// A store that enumerates its records is the history; one that cannot is
/// listed from the object store as any dataset is.
async fn history<'a>(
    base_path: &Path,
    object_store: &'a ObjectStore,
    records: Option<(Option<String>, Vec<VersionRecord>)>,
    sorted_descending: bool,
) -> Result<BoxStream<'a, Result<ManifestLocation>>> {
    let mut locations = match records {
        Some((generation, records)) => records
            .into_iter()
            .map(|(version, path, size)| {
                location_from_record(version, &path, size, generation.clone())
            })
            .collect::<Result<Vec<_>>>()?,
        None => {
            list_manifests(base_path, &object_store.inner)
                .try_collect::<Vec<_>>()
                .await?
        }
    };
    if sorted_descending {
        locations.sort_by_key(|location| std::cmp::Reverse(location.version));
    }
    Ok(stream::iter(locations.into_iter().map(Ok)).boxed())
}

/// Evaluate `$observe` under `$store`'s generation: the generation before
/// and after must agree, so the observation belongs to it. Yields
/// `(generation, observed)`; a store that keeps none yields `None`.
macro_rules! under_generation {
    ($store:expr, $base_uri:expr, $observe:expr) => {{
        let mut attempts = 0;
        loop {
            let before = $store.generation($base_uri).await?;
            let observed = $observe.await?;
            let after = $store.generation($base_uri).await?;
            if before == after {
                break (before, observed);
            }
            attempts += 1;
            if attempts == 8 {
                return Err(Error::io(format!(
                    "the dataset at '{}' was recreated repeatedly while it was being read",
                    $base_uri
                ))
                .into());
            }
        }
    }};
}

/// A store that owns publication holds the only path to its manifests, so
/// its records are the history and nothing else may stand in for them.
fn enumerated(
    store: &dyn ExternalManifestStore,
    base_uri: &str,
    records: Option<Vec<VersionRecord>>,
) -> Result<Option<Vec<VersionRecord>>> {
    if records.is_none() && store.supports_predecessor_condition() {
        return Err(Error::not_supported(format!(
            "the external manifest store owns publication for '{base_uri}' but cannot enumerate \
             its records; there is no other history to fall back on"
        )));
    }
    Ok(records)
}

/// A store record as a [`ManifestLocation`].
fn location_from_record(
    version: u64,
    path: &str,
    size: Option<u64>,
    generation: Option<String>,
) -> Result<ManifestLocation> {
    let path = Path::parse(path).map_err(|e| Error::invalid_input(e.to_string()))?;
    let naming_scheme = detect_naming_scheme_from_path(&path)?;
    Ok(ManifestLocation {
        version,
        path,
        size,
        naming_scheme,
        e_tag: None,
        generation,
    })
}

/// The outcome of [`ExternalManifestStore::put_if_predecessor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reservation {
    /// The version is reserved for the candidate; its minted identity.
    Reserved { identity: String },
    /// Nothing was written: the predecessor is no longer the manifest at
    /// its version.
    Refused { reason: String },
    /// Nothing was written: the version already has an entry.
    Taken,
}

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
/// Object-store overwrites can assign a new ETag to identical bytes. An ETag is
/// therefore neither logical manifest identity nor dataset-incarnation identity.
/// The generic protocol never persists or validates ETags in the external index:
/// a finalizer can observe generation E1, another finalizer can replace it with
/// the same selected bytes as E2, and then the first finalizer can publish after
/// the second. Persisting E1 would make a correct canonical object look corrupt.
///
/// A canonical HEAD still returns the generation observed by the current caller
/// in [`ManifestLocation`]. That ephemeral token keeps runtime caches from
/// treating a newly materialized object as the same observation as an older
/// object at the same `(uri, version)`, without turning the external index into
/// a second authority for physical object generations. The generic external
/// index stores only stable `(path, size)` metadata and readers ignore any legacy
/// stored ETag. This protocol assumes one dataset incarnation owns the physical
/// prefix; a separate incarnation identity is required to make arbitrary prefix
/// reuse unconditionally safe.
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
            generation: None,
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
                    generation: None,
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
    /// Whether [`Self::put_if_predecessor`] is available. A store that
    /// supports it owns publication: every manifest, conditioned or not, is
    /// published at a candidate path and reached only through its record,
    /// so it must also enumerate ([`Self::list_versions`]) and keep a
    /// generation ([`Self::generation`]).
    fn supports_predecessor_condition(&self) -> bool {
        false
    }

    /// The identity recorded for `version`, unique to that physical manifest.
    async fn get_identity(&self, _base_uri: &str, _version: u64) -> Result<Option<String>> {
        Ok(None)
    }

    /// Reserve `version` only if the entry at `predecessor.version` still
    /// carries `predecessor.identity` and `version` is above the no-reuse
    /// floor ([`Self::reuse_floor`]), as one atomic decision. The `Ok`
    /// outcomes are definitive: [`Reservation::Refused`] and
    /// [`Reservation::Taken`] promise no entry was written. Anything the
    /// store cannot vouch for -- a lost response, an internal retry that
    /// saw its own write -- must be an `Err`, which the handler reads back.
    async fn put_if_predecessor(
        &self,
        _base_uri: &str,
        _version: u64,
        _path: &str,
        _size: u64,
        _predecessor: &PredecessorIdentity,
    ) -> Result<Reservation> {
        Err(Error::not_supported(
            "this external manifest store cannot condition a reservation on its predecessor",
        ))
    }

    /// Every record for `base_uri` as `(version, path, size)`, in any order,
    /// or `None` where the store cannot enumerate. A store that enumerates
    /// is the dataset's whole history -- conditioned commits exist only as
    /// records -- so it must hold a record for every live version, including
    /// any that existed before it started recording.
    async fn list_versions(&self, _base_uri: &str) -> Result<Option<Vec<VersionRecord>>> {
        Ok(None)
    }

    /// [`Self::list_versions`] bounded to `version > since`, the conflict
    /// scan every commit attempt makes.
    async fn list_versions_since(
        &self,
        base_uri: &str,
        since: u64,
    ) -> Result<Option<Vec<VersionRecord>>> {
        Ok(self.list_versions(base_uri).await?.map(|records| {
            records
                .into_iter()
                .filter(|(version, _, _)| *version > since)
                .collect()
        }))
    }

    /// Forget the record for `version` once its manifest is gone, as one
    /// atomic decision: only while the record still names `path` and
    /// `generation` is still the current one. Any other record at that
    /// version belongs to a later writer or incarnation and is kept.
    /// Forgetting also raises the dataset's no-reuse floor
    /// ([`Self::reuse_floor`]) to `version`, durably: a forgotten version is
    /// never reserved again, so a reservation delayed past cleanup cannot
    /// reopen it.
    async fn forget_version(
        &self,
        _base_uri: &str,
        _version: u64,
        _path: &str,
        _generation: Option<&str>,
    ) -> Result<()> {
        Ok(())
    }

    /// The highest version forgotten in the current generation:
    /// [`Self::put_if_predecessor`] and [`Self::put_if_not_exists`] refuse
    /// any version at or below it, decided atomically with the reservation.
    /// It ends with the generation ([`Self::delete`]), so a recreation
    /// starts at version 1 again. `None` where the store keeps none; cleanup
    /// then never reclaims a candidate manifest.
    async fn reuse_floor(&self, _base_uri: &str) -> Result<Option<u64>> {
        Ok(None)
    }

    /// The dataset generation the records belong to: minted when the first
    /// record is written, ended by [`Self::delete`], which must end it before
    /// removing any record and must never remove a record of a newer
    /// generation, so a reservation racing a drop is refused rather than
    /// inherited by the recreation. A store that supports the predecessor
    /// condition must keep one; without it nothing fences a drop, and
    /// cleanup refuses to act.
    async fn generation(&self, _base_uri: &str) -> Result<Option<String>> {
        Ok(None)
    }

    /// The entry for `version` as `(path, identity)`, read as one
    /// observation: a path and an identity read separately can belong to two
    /// incarnations.
    async fn get_reservation(
        &self,
        _base_uri: &str,
        _version: u64,
    ) -> Result<Option<(String, String)>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    async fn put(
        &self,
        base_path: &Path,
        version: u64,
        staging_path: &Path,
        size: u64,
        _e_tag: Option<String>,
        object_store: &dyn OSObjectStore,
        naming_scheme: ManifestNamingScheme,
    ) -> Result<ManifestLocation> {
        // Default implementation: staging-based workflow

        // Step 1: Record staging path atomically
        // The external index owns version reservation, not object identity.
        // Staging paths are immutable and unique, so path and size are enough
        // to identify the selected source. Keeping ETags out of every generic
        // write also makes rolling upgrades converge naturally: new readers
        // ignore legacy values and every new publication removes them.
        self.put_if_not_exists(
            base_path.as_ref(),
            version,
            staging_path.as_ref(),
            size,
            None,
        )
        .await?;

        // Step 2: Copy staging to final path
        let final_path = naming_scheme.manifest_path(base_path, version);
        let final_e_tag =
            copy_or_verify_final_manifest(object_store, staging_path, &final_path, version, size)
                .await?;

        let location = ManifestLocation {
            version,
            path: final_path.clone(),
            size: Some(size),
            naming_scheme,
            e_tag: final_e_tag,
            generation: None,
        };

        // Step 3: Update the external index to the final path.
        //
        // Publish only generation-independent metadata. COPY and this update
        // are not one atomic operation, so an ETag observed above can already
        // be stale when this call linearizes. `location` still carries that
        // observation to the current caller for cache separation.
        let published = self
            .put_if_exists(base_path.as_ref(), version, final_path.as_ref(), size, None)
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

    /// Put the manifest path for a given base_uri and version, should fail if the version already exists.
    ///
    /// The generic staging workflow always passes `None` for `e_tag`. The
    /// parameter remains part of the trait for compatibility with stores that
    /// override the full [`Self::put`] protocol. Generic implementations must
    /// not retain a previous ETag when `None` is supplied.
    async fn put_if_not_exists(
        &self,
        base_uri: &str,
        version: u64,
        path: &str,
        size: u64,
        e_tag: Option<String>,
    ) -> Result<()>;

    /// Put the manifest path for a given base_uri and version, should fail if the version **does not** already exist.
    ///
    /// See [`Self::put_if_not_exists`] for the `e_tag` contract.
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

/// Copy the selected staging manifest to its canonical path.
///
/// A successful copy is the object store's acknowledgement that the known
/// immutable bytes were materialized. We then HEAD the destination for two
/// separate reasons: validate that the materialized size matches the selected
/// staging object, and return the physical-generation token observed by this
/// caller. The token is not content identity, but downstream caches currently
/// use it to avoid reusing an older object at the same `(uri, version)`.
///
/// `NotFound` is different: the selected staging object may have disappeared
/// because another helper finalized and deleted it, or because the commit is
/// unrecoverable. Only in that ambiguous recovery path do we HEAD the canonical
/// object and require its size to match the external-store-selected staging
/// manifest. Any ETag returned by that required HEAD is merely the current
/// object's opaque generation metadata.
async fn copy_or_verify_final_manifest(
    object_store: &dyn OSObjectStore,
    staging_path: &Path,
    final_path: &Path,
    version: u64,
    selected_size: u64,
) -> Result<Option<String>> {
    match copy_size_aware(object_store, staging_path, final_path, selected_size).await {
        Ok(()) => {
            info!(target: TRACE_FILE_AUDIT, mode=AUDIT_MODE_CREATE, r#type=AUDIT_TYPE_MANIFEST, path = final_path.as_ref());
            let final_meta = object_store.head(final_path).await?;
            if final_meta.size != selected_size {
                return Err(Error::corrupt_file(
                    final_path.clone(),
                    format!(
                        "Manifest size mismatch for version {}: selected staging manifest had {}, object store returned {}",
                        version, selected_size, final_meta.size
                    ),
                ));
            }
            Ok(final_meta.e_tag)
        }
        Err(ObjectStoreError::NotFound { .. }) => match object_store.head(final_path).await {
            Ok(final_meta) if final_meta.size == selected_size => Ok(final_meta.e_tag),
            Ok(final_meta) => Err(Error::corrupt_file(
                final_path.clone(),
                format!(
                    "Manifest size mismatch for version {}: selected staging manifest had {}, object store returned {}",
                    version, selected_size, final_meta.size
                ),
            )),
            Err(error) => Err(error.into()),
        },
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
    /// An unconditioned commit through a store that owns publication. Once
    /// the dataset has a record, the reservation is bound to the latest
    /// identity and its generation exactly as a conditioned commit's is, so
    /// a writer paused across a drop cannot join the recreation; only the
    /// first record is reserved unconditionally. The manifest is published
    /// at a candidate path, reached only through its record, so no object
    /// path is ever shared with another incarnation (see [`CANDIDATES_DIR`]).
    #[allow(clippy::too_many_arguments)]
    async fn commit_owned(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base_path: &Path,
        object_store: &ObjectStore,
        manifest_writer: super::ManifestWriter,
        naming_scheme: ManifestNamingScheme,
        transaction: Option<Transaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        let store = self.external_manifest_store.as_ref();
        let base_uri = base_path.as_ref();
        let version = manifest.version;
        if let Some(predecessor) = self
            .resolve_latest_identity(base_path, object_store)
            .await?
        {
            if predecessor.version + 1 != version {
                return Err(CommitError::CommitConflict);
            }
            return self
                .commit_after(
                    manifest,
                    indices,
                    base_path,
                    object_store,
                    manifest_writer,
                    naming_scheme,
                    transaction,
                    &predecessor,
                )
                .await;
        }
        let candidate = candidate_manifest_path(base_path, naming_scheme, version)?;
        let write_res =
            manifest_writer(object_store, manifest, indices, &candidate, transaction).await?;
        let size = write_res.size as u64;

        if let Err(error) = store
            .put_if_not_exists(base_uri, version, candidate.as_ref(), size, None)
            .await
        {
            // As in `commit`: only a record naming another path proves loss.
            let recorded = store.get_reservation(base_uri, version).await;
            if matches!(&recorded, Ok(Some((path, _))) if path != candidate.as_ref()) {
                match object_store.inner.delete(&candidate).await {
                    Ok(()) | Err(ObjectStoreError::NotFound { .. }) => {}
                    Err(delete_error) => warn!(
                        "Failed to delete losing manifest '{}': {}",
                        candidate, delete_error
                    ),
                }
                return Err(CommitError::CommitConflict);
            }
            warn!(
                "Reservation of version {version} at '{base_uri}' failed; retaining manifest '{}' \
                 until the outcome is resolved: {error}",
                candidate
            );
            return Err(CommitError::CommitConflict);
        }
        // The record is read back under the generation so the one returned
        // is the one it lives in -- a first record mints it.
        let (generation, recorded) =
            under_generation!(store, base_uri, store.get_reservation(base_uri, version));
        let generation = match recorded {
            Some((path, _)) if path == candidate.as_ref() => generation,
            _ => None,
        };
        write_version_hint(object_store, base_path, version).await;
        Ok(ManifestLocation {
            version,
            path: candidate,
            size: Some(size),
            naming_scheme,
            e_tag: write_res.e_tag,
            generation,
        })
    }

    async fn recorded_latest_location(
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
                    generation: _,
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
    async fn recorded_version_location(
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
                            generation: None,
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
                    e_tag: _,
                    generation: _,
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

                // Ignore any ETag returned by the external index. It may be a
                // legacy value published after a later equivalent COPY and is
                // therefore neither a safe generation fence nor content proof.
                // The HEAD result is the canonical object's current generation
                // and is returned only as an ephemeral cache discriminator.

                Ok(ManifestLocation {
                    version,
                    path,
                    size,
                    naming_scheme,
                    e_tag,
                    generation: None,
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

        let final_e_tag = copy_or_verify_final_manifest(
            store,
            staging_manifest_path,
            &final_manifest_path,
            version,
            size,
        )
        .await?;

        let location = ManifestLocation {
            version,
            path: final_manifest_path,
            size: Some(size),
            naming_scheme,
            e_tag: final_e_tag,
            generation: None,
        };

        // Step 2: point the external index at the final location without an
        // ETag. A direct writer and any number of helping readers can perform
        // the same immutable COPY concurrently. Since COPY and index update
        // are not atomic, persisting a helper's observed generation would let
        // an older helper overwrite a newer token. `location` retains the
        // current helper's observation for runtime cache separation only.
        let published = self
            .external_manifest_store
            .put_if_exists(
                base_path.as_ref(),
                version,
                location.path.as_ref(),
                size,
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
        let (generation, mut location) = under_generation!(
            self.external_manifest_store,
            base_path.as_ref(),
            self.recorded_latest_location(base_path, object_store)
        );
        location.generation = generation;
        Ok(location)
    }

    async fn resolve_version_location(
        &self,
        base_path: &Path,
        version: u64,
        object_store: &dyn OSObjectStore,
    ) -> std::result::Result<ManifestLocation, Error> {
        let (generation, mut location) = under_generation!(
            self.external_manifest_store,
            base_path.as_ref(),
            self.recorded_version_location(base_path, version, object_store)
        );
        location.generation = generation;
        Ok(location)
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

    /// A store that enumerates its records is the dataset's history: the
    /// manifests of conditioned commits exist only as records, and the
    /// commit-time scan is the store's bounded `version > since` range. A
    /// store that cannot enumerate is listed from the object store.
    fn list_manifest_locations<'a>(
        &self,
        base_path: &Path,
        object_store: &'a ObjectStore,
        sorted_descending: bool,
    ) -> BoxStream<'a, std::result::Result<ManifestLocation, Error>> {
        let store = self.external_manifest_store.clone();
        let base_path = base_path.clone();
        async move {
            let (generation, records) = under_generation!(
                store,
                base_path.as_ref(),
                store.list_versions(base_path.as_ref())
            );
            let records = enumerated(store.as_ref(), base_path.as_ref(), records)?
                .map(|records| (generation, records));
            history(&base_path, object_store, records, sorted_descending).await
        }
        .try_flatten_stream()
        .boxed()
    }

    fn list_manifest_locations_since<'a>(
        &self,
        base_path: &Path,
        object_store: &'a ObjectStore,
        since_version: u64,
    ) -> BoxStream<'a, std::result::Result<ManifestLocation, Error>> {
        let store = self.external_manifest_store.clone();
        let base_path = base_path.clone();
        async move {
            let (generation, records) = under_generation!(
                store,
                base_path.as_ref(),
                store.list_versions_since(base_path.as_ref(), since_version)
            );
            let records = enumerated(store.as_ref(), base_path.as_ref(), records)?
                .map(|records| (generation, records));
            history(&base_path, object_store, records, true).await
        }
        .try_flatten_stream()
        .try_filter(move |location| future::ready(location.version > since_version))
        .boxed()
    }

    async fn forget_version(
        &self,
        base_path: &Path,
        version: u64,
        path: &Path,
        generation: Option<&str>,
    ) -> Result<()> {
        self.external_manifest_store
            .forget_version(base_path.as_ref(), version, path.as_ref(), generation)
            .await
    }

    async fn may_reclaim(
        &self,
        base_path: &Path,
        version: u64,
        path: &Path,
        generation: Option<&str>,
    ) -> Result<bool> {
        let store = self.external_manifest_store.as_ref();
        let base_uri = base_path.as_ref();
        // One observation: the floor and the record at `version`, under the
        // generation they belong to.
        let (current, (floor, recorded)) = under_generation!(store, base_uri, async {
            Ok::<_, Error>((
                store.reuse_floor(base_uri).await?,
                store.get_reservation(base_uri, version).await?,
            ))
        });
        Ok(current.is_some()
            && current.as_deref() == generation
            && floor.is_some_and(|floor| version <= floor)
            && recorded.is_none_or(|(recorded, _)| recorded != path.as_ref()))
    }

    fn supports_predecessor_condition(&self) -> bool {
        self.external_manifest_store
            .supports_predecessor_condition()
    }

    async fn resolve_latest_identity(
        &self,
        base_path: &Path,
        _object_store: &ObjectStore,
    ) -> std::result::Result<Option<PredecessorIdentity>, Error> {
        let Some((version, _)) = self
            .external_manifest_store
            .get_latest_version(base_path.as_ref())
            .await?
        else {
            return Ok(None);
        };
        let generation = self
            .external_manifest_store
            .generation(base_path.as_ref())
            .await?;
        Ok(self
            .external_manifest_store
            .get_identity(base_path.as_ref(), version)
            .await?
            .map(|identity| PredecessorIdentity {
                version,
                identity,
                generation,
            }))
    }

    async fn commit_after(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base_path: &Path,
        object_store: &ObjectStore,
        manifest_writer: super::ManifestWriter,
        naming_scheme: ManifestNamingScheme,
        transaction: Option<Transaction>,
        predecessor: &PredecessorIdentity,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        let store = self.external_manifest_store.as_ref();
        let base_uri = base_path.as_ref();
        let version = manifest.version;
        // The manifest is published where it is written: a path no other
        // writer or incarnation can name, so nothing is ever copied onto the
        // canonical path shared with a recreated dataset. The name keeps the
        // canonical form so listings and cleanup parse its version.
        let candidate = candidate_manifest_path(base_path, naming_scheme, version)?;
        let write_res =
            manifest_writer(object_store, manifest, indices, &candidate, transaction).await?;
        let size = write_res.size as u64;
        let delete_candidate = || async {
            match object_store.inner.delete(&candidate).await {
                Ok(()) | Err(ObjectStoreError::NotFound { .. }) => {}
                Err(error) => warn!(
                    "Failed to delete unreserved manifest '{}': {}",
                    candidate, error
                ),
            }
        };

        // The reservation is the commit point and the only atomic step; the
        // store decides the predecessor condition together with it.
        match store
            .put_if_predecessor(base_uri, version, candidate.as_ref(), size, predecessor)
            .await
        {
            Ok(Reservation::Reserved { .. }) => {}
            // Only the store's definitive outcomes prove nothing was
            // written; only then is the candidate safe to delete.
            Ok(Reservation::Refused { reason }) => {
                delete_candidate().await;
                return Err(CommitError::OtherError(
                    lance_core::error::PrerequisiteFailedSnafu { message: reason }.build(),
                ));
            }
            Ok(Reservation::Taken) => {
                delete_candidate().await;
                return Err(CommitError::CommitConflict);
            }
            // Any error may follow a reservation the store applied and lost
            // the response to (its own retry then sees the version taken or
            // the predecessor changed). An entry naming this candidate
            // proves it landed; an entry naming another path proves it
            // lost. An absent entry proves nothing -- the write may still be
            // in flight -- so the candidate is retained and the outcome
            // reported unknown, since a late entry pointing at deleted bytes
            // would corrupt the version.
            Err(error) => match store.get_reservation(base_uri, version).await {
                Ok(Some((path, _))) if path == candidate.as_ref() => {}
                Ok(Some(_)) => {
                    delete_candidate().await;
                    return Err(CommitError::CommitConflict);
                }
                Ok(None) | Err(Error::NotFound { .. }) => {
                    warn!(
                        "Reservation of version {version} at '{base_uri}' has an unknown outcome; \
                         retaining manifest '{}': {}",
                        candidate, error
                    );
                    return Err(CommitError::OtherError(
                        Error::commit_status_unknown_source(version, Box::new(error)),
                    ));
                }
                Err(readback) => {
                    warn!(
                        "Reservation of version {version} at '{base_uri}' has an unknown outcome; \
                         retaining manifest '{}': {} (readback: {})",
                        candidate, error, readback
                    );
                    return Err(CommitError::OtherError(
                        Error::commit_status_unknown_source(version, Box::new(error)),
                    ));
                }
            },
        }
        write_version_hint(object_store, base_path, version).await;
        Ok(ManifestLocation {
            version,
            path: candidate,
            size: Some(size),
            naming_scheme,
            e_tag: write_res.e_tag,
            generation: predecessor.generation.clone(),
        })
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
        if self
            .external_manifest_store
            .supports_predecessor_condition()
        {
            return self
                .commit_owned(
                    manifest,
                    indices,
                    base_path,
                    object_store,
                    manifest_writer,
                    naming_scheme,
                    transaction,
                )
                .await;
        }

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
    use crate::io::commit::ConditionalPutCommitHandler;
    use crate::io::commit::write_manifest_file_to_path;

    #[derive(Debug, Clone)]
    struct StoredManifest {
        path: String,
        size: u64,
        e_tag: Option<String>,
        identity: String,
    }

    #[derive(Debug)]
    struct TestExternalManifestStore {
        manifests: Mutex<HashMap<(String, u64), StoredManifest>>,
        generation: Mutex<Option<String>>,
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
                generation: Mutex::new(None),
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
                generation: None,
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
            let identity = format!("identity-{}", manifests.len());
            manifests.insert(
                key,
                StoredManifest {
                    path: path.to_string(),
                    size,
                    e_tag,
                    identity,
                },
            );
            drop(manifests);
            if self.fail_next_put_response.swap(false, Ordering::SeqCst) {
                Err(Error::io("simulated lost external-store response"))
            } else {
                Ok(())
            }
        }

        async fn get_identity(&self, base_uri: &str, version: u64) -> Result<Option<String>> {
            Ok(self
                .manifests
                .lock()
                .unwrap()
                .get(&(base_uri.to_string(), version))
                .map(|manifest| manifest.identity.clone()))
        }

        async fn get_reservation(
            &self,
            base_uri: &str,
            version: u64,
        ) -> Result<Option<(String, String)>> {
            Ok(self
                .manifests
                .lock()
                .unwrap()
                .get(&(base_uri.to_string(), version))
                .map(|manifest| (manifest.path.clone(), manifest.identity.clone())))
        }

        async fn list_versions(&self, base_uri: &str) -> Result<Option<Vec<VersionRecord>>> {
            Ok(Some(
                self.manifests
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|((stored_base, _), _)| stored_base == base_uri)
                    .map(|((_, version), manifest)| {
                        (*version, manifest.path.clone(), Some(manifest.size))
                    })
                    .collect(),
            ))
        }

        async fn forget_version(
            &self,
            base_uri: &str,
            version: u64,
            path: &str,
            generation: Option<&str>,
        ) -> Result<()> {
            if self.generation.lock().unwrap().as_deref() != generation {
                return Ok(());
            }
            let mut manifests = self.manifests.lock().unwrap();
            let key = (base_uri.to_string(), version);
            if manifests.get(&key).is_some_and(|m| m.path == path) {
                manifests.remove(&key);
            }
            Ok(())
        }

        async fn generation(&self, _base_uri: &str) -> Result<Option<String>> {
            Ok(self.generation.lock().unwrap().clone())
        }

        async fn delete(&self, base_uri: &str) -> Result<()> {
            *self.generation.lock().unwrap() = None;
            self.manifests
                .lock()
                .unwrap()
                .retain(|(stored_base, _), _| stored_base != base_uri);
            Ok(())
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
            manifest.path = path.to_string();
            manifest.size = size;
            manifest.e_tag = e_tag;
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

    /// A store that enumerates is the whole history: a canonical manifest it
    /// holds no record for is not history, and cleanup forgetting a record
    /// removes the version.
    #[tokio::test]
    async fn test_history_is_the_store_records() {
        let external_store = Arc::new(TestExternalManifestStore::new(false));
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: external_store.clone(),
        };
        let object_store = ObjectStore::memory();
        let base_path = Path::from("dataset");
        let scheme = ManifestNamingScheme::V2;
        let v1 = scheme.manifest_path(&base_path, 1);
        object_store.put(&v1, b"one").await.unwrap();
        let v2 = candidate_manifest_path(&base_path, scheme, 2).unwrap();
        object_store.put(&v2, b"two").await.unwrap();
        external_store
            .put_if_not_exists(base_path.as_ref(), 2, v2.as_ref(), 3, None)
            .await
            .unwrap();

        let versions = |locations: Vec<ManifestLocation>| {
            locations.iter().map(|m| m.version).collect::<Vec<_>>()
        };
        let listed = handler
            .list_manifest_locations(&base_path, &object_store, true)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let generation = listed[0].generation.clone();
        assert_eq!(versions(listed), vec![2]);
        let newer = handler
            .list_manifest_locations_since(&base_path, &object_store, 1)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(versions(newer), vec![2]);

        handler
            .forget_version(&base_path, 2, &v2, generation.as_deref())
            .await
            .unwrap();
        let listed = handler
            .list_manifest_locations(&base_path, &object_store, true)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert!(listed.is_empty());
    }

    /// Candidates live outside `_versions`, so the generic listing never
    /// reports them; the handler's own listing comes from the store's
    /// records, ordered by version regardless of key order.
    #[tokio::test]
    async fn test_history_is_enumerated_from_the_store_records() {
        let external_store = Arc::new(TestExternalManifestStore::new(false));
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: external_store.clone(),
        };
        let object_store = ObjectStore::memory();
        let base_path = Path::from("dataset");
        let scheme = ManifestNamingScheme::V2;
        let v1 = scheme.manifest_path(&base_path, 1);
        let v2 = candidate_manifest_path(&base_path, scheme, 2).unwrap();
        assert!(v2.as_ref().starts_with("dataset/_candidates/"));
        for (path, version) in [(&v1, 1), (&v2, 2)] {
            object_store.put(path, b"m").await.unwrap();
            external_store
                .put_if_not_exists(base_path.as_ref(), version, path.as_ref(), 1, None)
                .await
                .unwrap();
        }

        let generic = ConditionalPutCommitHandler
            .list_manifest_locations(&base_path, &object_store, true)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(
            generic.iter().map(|m| m.version).collect::<Vec<_>>(),
            vec![1]
        );

        let listed = handler
            .list_manifest_locations(&base_path, &object_store, true)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(
            listed.iter().map(|m| m.version).collect::<Vec<_>>(),
            vec![2, 1]
        );
        assert_eq!(listed[0].path, v2);
        let newer = handler
            .list_manifest_locations_since(&base_path, &object_store, 1)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(newer.iter().map(|m| m.version).collect::<Vec<_>>(), vec![2]);
    }

    #[tokio::test]
    async fn test_finalized_manifest_ignores_legacy_external_store_etag() {
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
                final_meta.size,
                Some("expected-generation".to_string()),
            )
            .await
            .unwrap();

        let resolved = handler
            .resolve_version_location(&base_path, 1, object_store.inner.as_ref())
            .await
            .expect("a legacy external-store ETag must not override object storage");
        assert_eq!(resolved.path, final_path);
        assert_eq!(resolved.size, Some(final_meta.size));
        assert_eq!(resolved.e_tag, final_meta.e_tag);
    }

    #[tokio::test]
    async fn test_finalized_manifest_without_external_store_etag_uses_current_etag() {
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
                final_meta.size,
                None,
            )
            .await
            .unwrap();

        let resolved = handler
            .resolve_version_location(&base_path, 1, object_store.inner.as_ref())
            .await
            .expect("an absent external-store ETag must opt out of comparison");
        assert_eq!(resolved.path, final_path);
        assert_eq!(resolved.size, Some(final_meta.size));
        assert_eq!(resolved.e_tag, final_meta.e_tag);
    }

    #[tokio::test]
    async fn test_default_store_returns_but_does_not_persist_etag() {
        let external_store = Arc::new(TestExternalManifestStore::new(false));
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: external_store.clone(),
        };
        let object_store = ObjectStore::memory();
        let base_path = Path::from("dataset");
        let mut manifest = test_manifest();

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
            .expect("the default store should finalize the selected manifest");
        let original = object_store.inner.head(&committed.path).await.unwrap();
        assert_eq!(committed.e_tag, original.e_tag);

        let indexed = external_store
            .get_manifest_location(base_path.as_ref(), committed.version)
            .await
            .unwrap();
        assert_eq!(indexed.e_tag, None);

        object_store
            .inner
            .put(
                &committed.path,
                object_store::PutPayload::from(vec![0_u8; original.size as usize]),
            )
            .await
            .unwrap();

        let replacement = object_store.inner.head(&committed.path).await.unwrap();
        assert_ne!(replacement.e_tag, original.e_tag);

        let resolved = handler
            .resolve_version_location(&base_path, committed.version, object_store.inner.as_ref())
            .await
            .expect("the external index must not reject a new physical generation");
        assert_eq!(resolved.e_tag, replacement.e_tag);
    }

    #[tokio::test]
    async fn test_helping_finalizer_returns_but_does_not_persist_etag() {
        let external_store = Arc::new(TestExternalManifestStore::new(false));
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
        external_store
            .put_if_not_exists(
                base_path.as_ref(),
                version,
                staging_path.as_ref(),
                staging_meta.size,
                staging_meta.e_tag,
            )
            .await
            .unwrap();

        let finalized = handler
            .resolve_version_location(&base_path, version, object_store.inner.as_ref())
            .await
            .expect("a reader should finalize the selected staging manifest");
        let final_meta = object_store.inner.head(&final_path).await.unwrap();
        assert_eq!(finalized.e_tag, final_meta.e_tag);

        let indexed = external_store
            .get_manifest_location(base_path.as_ref(), version)
            .await
            .unwrap();
        assert_eq!(indexed.e_tag, None);
    }

    #[tokio::test]
    async fn test_onboarding_returns_but_does_not_persist_etag() {
        let external_store = Arc::new(TestExternalManifestStore::new(false));
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: external_store.clone(),
        };
        let object_store = ObjectStore::memory();
        let base_path = Path::from("dataset");
        let version = 1;
        let final_path = ManifestNamingScheme::V2.manifest_path(&base_path, version);

        object_store
            .inner
            .put(
                &final_path,
                object_store::PutPayload::from_static(b"manifest"),
            )
            .await
            .unwrap();
        let final_meta = object_store.inner.head(&final_path).await.unwrap();

        let resolved = handler
            .resolve_version_location(&base_path, version, object_store.inner.as_ref())
            .await
            .expect("an existing manifest should be indexed during onboarding");
        assert_eq!(resolved.e_tag, final_meta.e_tag);

        let indexed = external_store
            .get_manifest_location(base_path.as_ref(), version)
            .await
            .unwrap();
        assert_eq!(indexed.e_tag, None);
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
        assert!(
            committed.e_tag.is_some(),
            "the caller must retain the canonical generation even when index repair fails"
        );
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
        assert!(
            repaired.e_tag.is_some(),
            "a helping reader must receive the generation it observed"
        );
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
    async fn test_concurrent_finalizers_return_but_do_not_persist_generations() {
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
        let reservation = external_store
            .get_manifest_location(base_path.as_ref(), version)
            .await
            .unwrap();
        assert_eq!(reservation.path, staging_path);
        assert_eq!(reservation.e_tag, None);

        // The writer created generation E1. While its final index update is
        // paused, a reader observes the DDB-selected staging path and performs
        // the same immutable copy, producing generation E2. Each helper HEADs
        // the canonical object after its copy and returns the generation it
        // observed, but neither persists that race-prone token in the external
        // index. Both copies have exactly the same bytes; only their physical
        // object generations differ.
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
    async fn test_finalization_returns_etag_without_persisting_it() {
        let external_store = Arc::new(TestExternalManifestStore::new(false));
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
            .expect("the generic workflow should commit the canonical manifest");
        assert_eq!(committed.path, final_path);
        let final_meta = object_store.inner.head(&final_path).await.unwrap();
        assert_eq!(
            committed.e_tag, final_meta.e_tag,
            "the freshly committed Dataset needs the observed generation for cache separation"
        );

        let indexed = external_store
            .get_manifest_location(base_path.as_ref(), version)
            .await
            .expect("the external index must advance after the canonical copy");
        assert_eq!(indexed.path, final_path);
        assert_eq!(
            indexed.e_tag, None,
            "the external index must remain independent of physical generations"
        );
    }

    #[tokio::test]
    async fn test_missing_staging_verifies_existing_final_manifest() {
        let object_store = ObjectStore::memory();
        let staging_path = Path::from("dataset/_versions/1.manifest-missing");
        let final_path = Path::from("dataset/_versions/1.manifest");
        let manifest_bytes = Bytes::from_static(b"immutable manifest bytes");
        object_store
            .inner
            .put(&final_path, manifest_bytes.clone().into())
            .await
            .unwrap();
        let final_meta = object_store.inner.head(&final_path).await.unwrap();

        let recovered_e_tag = copy_or_verify_final_manifest(
            object_store.inner.as_ref(),
            &staging_path,
            &final_path,
            1,
            manifest_bytes.len() as u64,
        )
        .await
        .expect("an existing canonical manifest should prove another helper finalized it");

        assert_eq!(recovered_e_tag, final_meta.e_tag);
    }

    #[tokio::test]
    async fn test_missing_staging_rejects_missing_final_manifest() {
        let object_store = ObjectStore::memory();
        let staging_path = Path::from("dataset/_versions/1.manifest-missing");
        let final_path = Path::from("dataset/_versions/1.manifest");

        let error = copy_or_verify_final_manifest(
            object_store.inner.as_ref(),
            &staging_path,
            &final_path,
            1,
            42,
        )
        .await
        .expect_err("missing staging and canonical objects cannot establish a commit");

        assert!(matches!(error, Error::NotFound { .. }), "{error:?}");
        assert!(error.to_string().contains(final_path.as_ref()), "{error}");
    }

    #[tokio::test]
    async fn test_missing_staging_rejects_wrong_final_size() {
        let object_store = ObjectStore::memory();
        let staging_path = Path::from("dataset/_versions/1.manifest-missing");
        let final_path = Path::from("dataset/_versions/1.manifest");
        object_store
            .inner
            .put(&final_path, Bytes::from_static(b"wrong size").into())
            .await
            .unwrap();

        let error = copy_or_verify_final_manifest(
            object_store.inner.as_ref(),
            &staging_path,
            &final_path,
            1,
            42,
        )
        .await
        .expect_err("a same-path object with the wrong size is not the selected manifest");

        assert!(matches!(error, Error::CorruptFile { .. }), "{error:?}");
        assert!(
            error.to_string().contains("Manifest size mismatch"),
            "{error}"
        );
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
