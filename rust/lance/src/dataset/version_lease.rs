// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Advisory leases that protect dataset versions from cleanup.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, LazyLock},
    time::Duration,
};

use bytes::Bytes;
use chrono::{DateTime, TimeDelta, Utc};
use dashmap::DashSet;
use futures::{StreamExt, TryStreamExt, stream};
use lance_io::object_store::ObjectStore;
use object_store::{ObjectMeta, ObjectStoreExt, PutMode, PutOptions, path::Path};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{
    Dataset,
    refs::{MAIN_BRANCH, Refs},
};
use crate::{Error, Result};

const LEASES_DIR: &str = "_refs/version_leases";
const LEASE_GC_MARKERS_DIR: &str = "_refs/version_lease_gc";
const LEASE_FILE_SUFFIX: &str = ".lease";
const DRAINING_MARKER_SUFFIX: &str = ".draining";
const SEALED_MARKER_SUFFIX: &str = ".sealed";
const COMMITTED_MARKER_SUFFIX: &str = ".committed";
const STORAGE_CLOCK_PATH: &str = "_clock";

/// HTTP `Last-Modified`, used by supported cloud stores, has whole-second precision.
/// Adding one interval guarantees a lease is never shortened by timestamp truncation.
const STORAGE_TIMESTAMP_PRECISION: Duration = Duration::from_secs(1);

/// Draining does no deletion and must complete within this ownership window.
/// A cleaner that exceeds the window fails before sealing; another actor may then
/// ignore and remove the abandoned drain without racing a live deletion.
const DRAINING_OWNERSHIP_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// Drains whose in-process owner was dropped before sealing. The operation UUID
/// makes each path unique across stores, while process-wide scope lets a fresh
/// dataset handle immediately disregard a drain that can no longer delete.
static LOCALLY_ABANDONED_DRAINS: LazyLock<DashSet<Path>> = LazyLock::new(DashSet::new);

/// A renewable advisory lease that protects one dataset version from cleanup.
///
/// The lease must be renewed before [`Self::expires_at`]. Once it expires,
/// [`Dataset::cleanup_old_versions`](Dataset::cleanup_old_versions) may delete
/// the protected version. Dropping a lease releases it on a best-effort basis;
/// if a process exits or cannot release the lease, it stops protecting the
/// version when its TTL expires.
#[derive(Debug)]
pub struct VersionLease {
    store: VersionLeaseStore,
    path: Path,
    version: u64,
    expires_at: DateTime<Utc>,
    released: bool,
}

impl VersionLease {
    /// The dataset version protected by this lease.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// The time after which cleanup may remove the protected version.
    pub fn expires_at(&self) -> DateTime<Utc> {
        self.expires_at
    }

    /// Renew this lease for `ttl` from the current time.
    ///
    /// Renewal must complete before the current expiration. A lease admitted
    /// before cleanup starts draining its version may renew while draining;
    /// renewal fails once cleanup seals the version for deletion.
    pub async fn renew(&mut self, ttl: Duration) -> Result<()> {
        if !self.store.object_store.exists(&self.path).await? {
            return Err(expired_lease_error(self.version, self.expires_at));
        }
        self.store.ensure_not_sealed(self.version).await?;

        // Publish the replacement before removing the old file so cleanup never
        // observes a gap between a timely renewal and its predecessor.
        let lease_file = self.store.create_lease_file(self.version, ttl).await?;
        if lease_file.created_at >= self.expires_at {
            let _ = self.store.object_store.delete(&lease_file.path).await;
            return Err(expired_lease_error(self.version, self.expires_at));
        }
        if let Err(error) = self.store.ensure_not_sealed(self.version).await {
            let _ = self.store.object_store.delete(&lease_file.path).await;
            return Err(error);
        }

        if let Err(error) = self.store.object_store.delete(&self.path).await
            && !error.is_not_found()
        {
            tracing::warn!(
                path = %self.path,
                error = %error,
                "Failed to remove superseded version lease"
            );
        }
        self.path = lease_file.path;
        self.expires_at = lease_file.expires_at;
        Ok(())
    }

    /// Release this lease before its TTL expires.
    pub async fn release(mut self) -> Result<()> {
        match self.store.object_store.delete(&self.path).await {
            Ok(()) => {
                self.released = true;
                Ok(())
            }
            Err(error) if error.is_not_found() => {
                self.released = true;
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
}

impl Drop for VersionLease {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        let object_store = Arc::clone(&self.store.object_store);
        let path = self.path.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let _ = object_store.delete(&path).await;
            });
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct VersionLeaseStore {
    object_store: Arc<ObjectStore>,
    leases_path: Path,
    markers_path: Path,
    manifest_path: Option<Path>,
}

#[derive(Debug)]
struct LeaseFile {
    path: Path,
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
}

#[derive(Debug)]
struct RetirementFence {
    draining_path: Option<Path>,
    sealed_path: Option<Path>,
    committed_path: Option<Path>,
    observed_at: DateTime<Utc>,
    manifest_paths: Vec<Path>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RetirementMarker {
    manifest_paths: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RetirementState {
    Draining,
    Sealed,
    Committed,
}

#[derive(Debug)]
struct RetirementMarkerMetadata {
    version: u64,
    operation_id: String,
    state: RetirementState,
    metadata: ObjectMeta,
}

/// Per-cleanup retirement markers for versions selected for deletion.
///
/// Draining markers reject new leases while allowing already-admitted leases
/// to renew. Sealed markers reject both acquisition and renewal while cleanup
/// performs its final lease and reference checks. Committed markers are the
/// durable, irreversible deletion boundary shared with reference admission.
/// Unique operation markers keep cancellation and finalization safe when
/// multiple cleanups overlap.
#[derive(Debug)]
pub(super) struct RetirementGuard {
    store: VersionLeaseStore,
    fences: HashMap<u64, RetirementFence>,
}

impl VersionLeaseStore {
    pub(super) async fn for_dataset(dataset: &Dataset) -> Result<Self> {
        let branch = dataset.manifest.branch.as_deref();
        Self::for_refs(
            &dataset.refs,
            branch,
            Some(dataset.manifest_location.path.clone()),
        )
        .await
    }

    async fn for_refs(
        refs: &Refs,
        branch: Option<&str>,
        manifest_path: Option<Path>,
    ) -> Result<Self> {
        let root = refs.root()?;
        let branch_identifier = refs.branches().get_identifier(branch).await?;
        let namespace = if branch.is_none() {
            MAIN_BRANCH
        } else {
            branch_identifier
                .version_mapping
                .last()
                .map(|(_, id)| id.as_str())
                .ok_or_else(|| {
                    Error::internal(format!(
                        "branch {} has no branch identifier",
                        branch.unwrap_or_default()
                    ))
                })?
        };

        Ok(Self {
            object_store: Arc::clone(&refs.object_store),
            leases_path: root.path.clone().join(LEASES_DIR).join(namespace),
            markers_path: root.path.join(LEASE_GC_MARKERS_DIR).join(namespace),
            manifest_path,
        })
    }

    async fn acquire(&self, version: u64, ttl: Duration) -> Result<VersionLease> {
        self.ensure_version_available(version).await?;
        let lease_file = self.create_lease_file(version, ttl).await?;
        if let Err(error) = self.ensure_version_available(version).await {
            let _ = self.object_store.delete(&lease_file.path).await;
            return Err(error);
        }
        Ok(VersionLease {
            store: self.clone(),
            path: lease_file.path,
            version,
            expires_at: lease_file.expires_at,
            released: false,
        })
    }

    async fn create_lease_file(&self, version: u64, ttl: Duration) -> Result<LeaseFile> {
        let ttl_micros = ttl_micros(ttl)?;
        let path = self.leases_path.clone().join(format!(
            "{}-{}-{}{}",
            version,
            ttl_micros,
            Uuid::new_v4().simple(),
            LEASE_FILE_SUFFIX
        ));
        self.object_store
            .inner
            .put_opts(
                &path,
                Bytes::new().into(),
                PutOptions {
                    mode: PutMode::Create,
                    ..Default::default()
                },
            )
            .await?;
        let metadata = match self.object_store.inner.head(&path).await {
            Ok(metadata) => metadata,
            Err(error) => {
                let _ = self.object_store.delete(&path).await;
                return Err(error.into());
            }
        };
        let expires_at = expiration_from_ttl(metadata.last_modified, ttl)?;
        Ok(LeaseFile {
            path,
            created_at: metadata.last_modified,
            expires_at,
        })
    }

    pub(super) async fn all_lease_versions(&self) -> Result<HashSet<u64>> {
        self.lease_metadata()
            .await?
            .into_iter()
            .map(|metadata| parse_lease_metadata(&metadata).map(|(version, _)| version))
            .collect()
    }

    pub(super) async fn active_versions_at(
        &self,
        observed_at: &HashMap<u64, DateTime<Utc>>,
        remove_expired: bool,
    ) -> Result<HashSet<u64>> {
        let mut active_versions = HashSet::new();
        let mut expired_paths = Vec::new();

        for metadata in self.lease_metadata().await? {
            let (version, ttl) = parse_lease_metadata(&metadata)?;
            let Some(reference_time) = observed_at.get(&version) else {
                continue;
            };
            let expires_at = expiration_from_ttl(metadata.last_modified, ttl)?;
            if expires_at > *reference_time {
                active_versions.insert(version);
            } else if remove_expired {
                expired_paths.push(metadata.location);
            }
        }

        if remove_expired {
            stream::iter(expired_paths)
                .map(Ok)
                .try_for_each_concurrent(self.object_store.io_parallelism(), |path| async move {
                    match self.object_store.delete(&path).await {
                        Ok(()) => Ok(()),
                        Err(error) if error.is_not_found() => Ok(()),
                        Err(error) => Err(error),
                    }
                })
                .await?;
        }
        Ok(active_versions)
    }

    pub(super) async fn fence_versions(
        &self,
        manifest_paths: &HashMap<u64, Vec<Path>>,
    ) -> Result<RetirementGuard> {
        let operation_id = Uuid::new_v4().simple().to_string();
        let version_manifests = manifest_paths
            .iter()
            .map(|(version, paths)| (*version, paths.clone()))
            .collect::<Vec<_>>();
        let results = stream::iter(version_manifests)
            .map(|(version, manifest_paths)| {
                let path = self
                    .marker_version_path(version)
                    .join(format!("{operation_id}{DRAINING_MARKER_SUFFIX}"));
                async move {
                    let payload = retirement_marker_payload(&manifest_paths)?;
                    let metadata = self.create_marker(&path, payload).await?;
                    Ok::<_, Error>((version, path, metadata.last_modified, manifest_paths))
                }
            })
            .buffer_unordered(self.object_store.io_parallelism())
            .collect::<Vec<_>>()
            .await;

        let mut fences = HashMap::with_capacity(results.len());
        let mut first_error = None;
        for result in results {
            match result {
                Ok((version, path, observed_at, manifest_paths)) => {
                    fences.insert(
                        version,
                        RetirementFence {
                            draining_path: Some(path),
                            sealed_path: None,
                            committed_path: None,
                            observed_at,
                            manifest_paths,
                        },
                    );
                }
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            }
        }
        let mut guard = RetirementGuard {
            store: self.clone(),
            fences,
        };
        if let Some(error) = first_error {
            guard.cancel_all().await?;
            return Err(error);
        }
        Ok(guard)
    }

    async fn ensure_version_available(&self, version: u64) -> Result<()> {
        if let Some(manifest_path) = &self.manifest_path
            && !self.object_store.exists(manifest_path).await?
        {
            Err(Error::VersionNotFound {
                message: format!("version {version} no longer exists and cannot be leased"),
            })
        } else if self.has_active_retirement_marker(version).await? {
            Err(retiring_version_error(version))
        } else {
            Ok(())
        }
    }

    async fn ensure_not_sealed(&self, version: u64) -> Result<()> {
        if self.has_sealed_marker(version).await? {
            Err(retiring_version_error(version))
        } else {
            Ok(())
        }
    }

    async fn has_sealed_marker(&self, version: u64) -> Result<bool> {
        Ok(self
            .version_marker_metadata(version)
            .await?
            .iter()
            .any(|marker| marker.state != RetirementState::Draining))
    }

    async fn has_active_retirement_marker(&self, version: u64) -> Result<bool> {
        let markers = self.version_marker_metadata(version).await?;
        if markers
            .iter()
            .any(|marker| marker.state != RetirementState::Draining)
        {
            return Ok(true);
        }

        let mut draining_markers = Vec::new();
        for marker in markers {
            if LOCALLY_ABANDONED_DRAINS.contains(&marker.metadata.location) {
                match self.object_store.delete(&marker.metadata.location).await {
                    Ok(()) => {
                        LOCALLY_ABANDONED_DRAINS.remove(&marker.metadata.location);
                    }
                    Err(error) if error.is_not_found() => {
                        LOCALLY_ABANDONED_DRAINS.remove(&marker.metadata.location);
                    }
                    Err(error) => {
                        tracing::warn!(
                            path = %marker.metadata.location,
                            error = %error,
                            "Failed to remove locally abandoned version retirement drain"
                        );
                    }
                }
            } else {
                draining_markers.push(marker);
            }
        }
        if draining_markers.is_empty() {
            return Ok(false);
        }

        let observed_at = self.storage_observed_at().await?;
        let mut stale_paths = Vec::new();
        for marker in draining_markers {
            if draining_owner_is_active(marker.metadata.last_modified, observed_at)? {
                return Ok(true);
            }
            stale_paths.push(marker.metadata.location);
        }
        self.delete_paths(stale_paths).await?;
        Ok(false)
    }

    async fn version_marker_metadata(&self, version: u64) -> Result<Vec<RetirementMarkerMetadata>> {
        let metadata = self
            .object_store
            .list(Some(self.marker_version_path(version)))
            .try_collect::<Vec<_>>()
            .await?;
        metadata
            .into_iter()
            .map(|metadata| self.parse_retirement_marker_metadata(metadata))
            .collect()
    }

    async fn all_marker_metadata(&self) -> Result<Vec<RetirementMarkerMetadata>> {
        let metadata = self
            .object_store
            .list(Some(self.markers_path.clone()))
            .try_collect::<Vec<_>>()
            .await?;
        metadata
            .into_iter()
            .filter(|metadata| metadata.location != self.storage_clock_path())
            .map(|metadata| self.parse_retirement_marker_metadata(metadata))
            .collect()
    }

    fn parse_retirement_marker_metadata(
        &self,
        metadata: ObjectMeta,
    ) -> Result<RetirementMarkerMetadata> {
        let relative_parts = metadata
            .location
            .prefix_match(&self.markers_path)
            .ok_or_else(|| {
                Error::corrupt_file(
                    metadata.location.clone(),
                    "retirement marker is outside its namespace",
                )
            })?
            .map(|part| part.as_ref().to_string())
            .collect::<Vec<_>>();
        if relative_parts.len() != 2 {
            return Err(Error::corrupt_file(
                metadata.location,
                "retirement marker path must contain a version and operation filename",
            ));
        }
        let version = relative_parts[0].parse::<u64>().map_err(|error| {
            Error::corrupt_file(
                metadata.location.clone(),
                format!("retirement marker has invalid version: {error}"),
            )
        })?;
        let file_name = &relative_parts[1];
        let (operation_id, state) =
            if let Some(operation_id) = file_name.strip_suffix(DRAINING_MARKER_SUFFIX) {
                (operation_id, RetirementState::Draining)
            } else if let Some(operation_id) = file_name.strip_suffix(SEALED_MARKER_SUFFIX) {
                (operation_id, RetirementState::Sealed)
            } else if let Some(operation_id) = file_name.strip_suffix(COMMITTED_MARKER_SUFFIX) {
                (operation_id, RetirementState::Committed)
            } else {
                return Err(Error::corrupt_file(
                    metadata.location,
                    "retirement marker filename has an unknown state suffix",
                ));
            };
        Uuid::parse_str(operation_id).map_err(|error| {
            Error::corrupt_file(
                metadata.location.clone(),
                format!("retirement marker has invalid operation id: {error}"),
            )
        })?;
        Ok(RetirementMarkerMetadata {
            version,
            operation_id: operation_id.to_string(),
            state,
            metadata,
        })
    }

    async fn storage_observed_at(&self) -> Result<DateTime<Utc>> {
        let path = self.storage_clock_path();
        self.object_store
            .inner
            .put(&path, Bytes::new().into())
            .await?;
        Ok(self.object_store.inner.head(&path).await?.last_modified)
    }

    fn storage_clock_path(&self) -> Path {
        self.markers_path.clone().join(STORAGE_CLOCK_PATH)
    }

    pub(super) async fn recover_retirements(self) -> Result<HashSet<u64>> {
        let markers = self.all_marker_metadata().await?;
        if markers.is_empty() {
            return Ok(HashSet::new());
        }

        let sealed_operations: HashSet<_> = markers
            .iter()
            .filter(|marker| marker.state != RetirementState::Draining)
            .map(|marker| (marker.version, marker.operation_id.clone()))
            .collect();
        let observed_at = self.storage_observed_at().await?;
        let sealed_observation = markers
            .iter()
            .filter(|marker| marker.state != RetirementState::Draining)
            .map(|marker| (marker.version, observed_at))
            .collect::<HashMap<_, _>>();
        let actively_leased_versions = self.active_versions_at(&sealed_observation, true).await?;
        let mut stale_drains = Vec::new();
        let mut sealed_manifest_paths = HashMap::<u64, HashSet<Path>>::new();
        for marker in markers {
            if marker.state != RetirementState::Draining {
                let manifest_paths = self.read_retirement_marker(&marker.metadata).await?;
                sealed_manifest_paths
                    .entry(marker.version)
                    .or_default()
                    .extend(manifest_paths);
            } else if sealed_operations.contains(&(marker.version, marker.operation_id))
                || LOCALLY_ABANDONED_DRAINS.contains(&marker.metadata.location)
                || !draining_owner_is_active(marker.metadata.last_modified, observed_at)?
            {
                stale_drains.push(marker.metadata.location);
            }
        }
        self.delete_paths(stale_drains.clone()).await?;
        for path in stale_drains {
            LOCALLY_ABANDONED_DRAINS.remove(&path);
        }

        let mut terminal_manifests = HashMap::new();
        let mut versions_to_resume = HashSet::new();
        for (version, manifest_paths) in sealed_manifest_paths {
            let manifest_paths = manifest_paths.into_iter().collect::<Vec<_>>();
            let mut has_existing_manifest = false;
            for path in &manifest_paths {
                has_existing_manifest |= self.object_store.exists(path).await?;
            }
            if has_existing_manifest {
                // A seal is the renewal cutoff. If its owner disappeared before
                // confirming the final lease scan, wait out every lease that was
                // still entitled to its published TTL before resuming deletion.
                if !actively_leased_versions.contains(&version) {
                    versions_to_resume.insert(version);
                }
            } else {
                terminal_manifests.insert(version, manifest_paths);
            }
        }
        self.finalize_versions(&terminal_manifests).await?;
        Ok(versions_to_resume)
    }

    async fn read_retirement_marker(&self, metadata: &ObjectMeta) -> Result<Vec<Path>> {
        let bytes = self
            .object_store
            .inner
            .get(&metadata.location)
            .await?
            .bytes()
            .await?;
        let marker: RetirementMarker = serde_json::from_slice(&bytes)
            .map_err(|error| Error::corrupt_file(metadata.location.clone(), error.to_string()))?;
        if marker.manifest_paths.is_empty() {
            return Err(Error::corrupt_file(
                metadata.location.clone(),
                "retirement marker has no manifest identity",
            ));
        }
        marker
            .manifest_paths
            .into_iter()
            .map(Path::parse)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::from)
    }

    async fn lease_metadata(&self) -> Result<Vec<ObjectMeta>> {
        self.object_store
            .list(Some(self.leases_path.clone()))
            .try_collect()
            .await
    }

    fn marker_version_path(&self, version: u64) -> Path {
        self.markers_path.clone().join(version.to_string())
    }

    async fn create_marker(&self, path: &Path, payload: Bytes) -> Result<ObjectMeta> {
        self.object_store
            .inner
            .put_opts(
                path,
                payload.into(),
                PutOptions {
                    mode: PutMode::Create,
                    ..Default::default()
                },
            )
            .await?;
        match self.object_store.inner.head(path).await {
            Ok(metadata) => Ok(metadata),
            Err(error) => {
                let _ = self.object_store.delete(path).await;
                Err(error.into())
            }
        }
    }
}

impl RetirementGuard {
    pub(super) fn observed_at(&self) -> HashMap<u64, DateTime<Utc>> {
        self.fences
            .iter()
            .map(|(version, fence)| (*version, fence.observed_at))
            .collect()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.fences.is_empty()
    }

    pub(super) async fn seal_versions(&mut self, versions: &HashSet<u64>) -> Result<()> {
        let mut marker_paths = Vec::with_capacity(versions.len());
        for version in versions {
            let fence = self.fences.get(version).ok_or_else(|| {
                Error::internal(format!("missing draining fence for version {version}"))
            })?;
            let draining_path = fence.draining_path.as_ref().ok_or_else(|| {
                Error::internal(format!("version {version} has no draining marker"))
            })?;
            let file_name = draining_path.filename().ok_or_else(|| {
                Error::internal(format!("draining marker {draining_path} has no filename"))
            })?;
            let operation_id = file_name
                .strip_suffix(DRAINING_MARKER_SUFFIX)
                .ok_or_else(|| {
                    Error::internal(format!(
                        "draining marker {draining_path} has an invalid suffix"
                    ))
                })?;
            marker_paths.push((
                *version,
                self.store
                    .marker_version_path(*version)
                    .join(format!("{operation_id}{SEALED_MARKER_SUFFIX}")),
                retirement_marker_payload(&fence.manifest_paths)?,
                fence.observed_at,
            ));
        }

        let store = self.store.clone();
        let results = stream::iter(marker_paths)
            .map(move |(version, path, payload, draining_observed_at)| {
                let store = store.clone();
                async move {
                    let metadata = store.create_marker(&path, payload).await?;
                    if !draining_owner_is_active(draining_observed_at, metadata.last_modified)? {
                        let _ = store.object_store.delete(&path).await;
                        return Err(Error::internal(format!(
                            "version {version} retirement ownership expired before sealing"
                        )));
                    }
                    Ok::<_, Error>((version, path, metadata.last_modified))
                }
            })
            .buffer_unordered(self.store.object_store.io_parallelism())
            .collect::<Vec<_>>()
            .await;

        let mut first_error = None;
        for result in results {
            match result {
                Ok((version, sealed_path, observed_at)) => {
                    let fence = self.fences.get_mut(&version).ok_or_else(|| {
                        Error::internal(format!("missing draining fence for version {version}"))
                    })?;
                    fence.sealed_path = Some(sealed_path);
                    fence.observed_at = observed_at;
                }
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }

        let draining_paths = versions
            .iter()
            .filter_map(|version| {
                self.fences
                    .get(version)
                    .and_then(|fence| fence.draining_path.clone())
            })
            .collect::<Vec<_>>();
        self.store.delete_paths(draining_paths).await?;
        for version in versions {
            if let Some(fence) = self.fences.get_mut(version) {
                fence.draining_path = None;
            }
        }
        Ok(())
    }

    /// Publish the irreversible retirement boundary after the final lease and
    /// durable-reference scans. Reference admission checks this marker before
    /// and after publishing a tag or branch.
    pub(super) async fn commit_versions(&mut self, versions: &HashSet<u64>) -> Result<()> {
        let mut marker_paths = Vec::with_capacity(versions.len());
        for version in versions {
            let fence = self.fences.get(version).ok_or_else(|| {
                Error::internal(format!("missing sealed fence for version {version}"))
            })?;
            let sealed_path = fence.sealed_path.as_ref().ok_or_else(|| {
                Error::internal(format!("version {version} has no sealed marker"))
            })?;
            let file_name = sealed_path.filename().ok_or_else(|| {
                Error::internal(format!("sealed marker {sealed_path} has no filename"))
            })?;
            let operation_id = file_name
                .strip_suffix(SEALED_MARKER_SUFFIX)
                .ok_or_else(|| {
                    Error::internal(format!("sealed marker {sealed_path} has an invalid suffix"))
                })?;
            marker_paths.push((
                *version,
                self.store
                    .marker_version_path(*version)
                    .join(format!("{operation_id}{COMMITTED_MARKER_SUFFIX}")),
                retirement_marker_payload(&fence.manifest_paths)?,
            ));
        }

        let store = self.store.clone();
        let results = stream::iter(marker_paths)
            .map(move |(version, path, payload)| {
                let store = store.clone();
                async move {
                    let metadata = store.create_marker(&path, payload).await?;
                    Ok::<_, Error>((version, path, metadata.last_modified))
                }
            })
            .buffer_unordered(self.store.object_store.io_parallelism())
            .collect::<Vec<_>>()
            .await;

        let mut first_error = None;
        for result in results {
            match result {
                Ok((version, committed_path, observed_at)) => {
                    let fence = self.fences.get_mut(&version).ok_or_else(|| {
                        Error::internal(format!("missing sealed fence for version {version}"))
                    })?;
                    fence.committed_path = Some(committed_path);
                    fence.observed_at = observed_at;
                }
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }

        let sealed_paths = versions
            .iter()
            .filter_map(|version| {
                self.fences
                    .get(version)
                    .and_then(|fence| fence.sealed_path.clone())
            })
            .collect::<Vec<_>>();
        if let Err(error) = self.store.delete_paths(sealed_paths).await {
            tracing::warn!(
                error = %error,
                "Failed to remove superseded sealed retirement markers"
            );
        } else {
            for version in versions {
                if let Some(fence) = self.fences.get_mut(version) {
                    fence.sealed_path = None;
                }
            }
        }
        Ok(())
    }

    pub(super) async fn cancel_versions(&mut self, versions: &HashSet<u64>) -> Result<()> {
        let cancellable_versions = versions
            .iter()
            .filter(|version| {
                self.fences
                    .get(version)
                    .is_some_and(|fence| fence.committed_path.is_none())
            })
            .copied()
            .collect::<HashSet<_>>();
        let paths = cancellable_versions
            .iter()
            .filter_map(|version| self.fences.get(version))
            .flat_map(|fence| {
                [fence.draining_path.clone(), fence.sealed_path.clone()]
                    .into_iter()
                    .flatten()
            })
            .collect::<Vec<_>>();
        self.store.delete_paths(paths).await?;
        self.fences
            .retain(|version, _| !cancellable_versions.contains(version));
        Ok(())
    }

    pub(super) async fn cancel_all(&mut self) -> Result<()> {
        let versions = self.fences.keys().copied().collect();
        self.cancel_versions(&versions).await
    }

    pub(super) async fn finalize(
        &mut self,
        manifest_paths: &HashMap<u64, Vec<Path>>,
    ) -> Result<()> {
        let guarded_manifest_paths = self
            .fences
            .keys()
            .map(|version| {
                manifest_paths
                    .get(version)
                    .map(|paths| (*version, paths.clone()))
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "missing manifest identity for retiring version {version}"
                        ))
                    })
            })
            .collect::<Result<HashMap<_, _>>>()?;
        self.store
            .finalize_versions(&guarded_manifest_paths)
            .await?;
        self.fences.clear();
        Ok(())
    }
}

impl Drop for RetirementGuard {
    fn drop(&mut self) {
        let draining_paths = self
            .fences
            .values()
            .filter_map(|fence| fence.draining_path.clone())
            .collect::<Vec<_>>();
        if draining_paths.is_empty() {
            return;
        }
        for path in &draining_paths {
            LOCALLY_ABANDONED_DRAINS.insert(path.clone());
        }
        let store = self.store.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                match store.delete_paths(draining_paths.clone()).await {
                    Ok(()) => {
                        for path in draining_paths {
                            LOCALLY_ABANDONED_DRAINS.remove(&path);
                        }
                    }
                    Err(error) => {
                        tracing::warn!(
                            error = %error,
                            "Failed to remove abandoned version retirement drains"
                        );
                    }
                }
            });
        }
    }
}

impl VersionLeaseStore {
    async fn finalize_versions(&self, manifest_paths: &HashMap<u64, Vec<Path>>) -> Result<()> {
        if manifest_paths.is_empty() {
            return Ok(());
        }
        for (version, paths) in manifest_paths {
            if paths.is_empty() {
                return Err(Error::internal(format!(
                    "missing manifest identity for retiring version {version}"
                )));
            }
            for path in paths {
                if self.object_store.exists(path).await? {
                    return Err(Error::internal(format!(
                        "cannot finalize retirement for version {version}: manifest {path} still exists"
                    )));
                }
            }
        }

        let versions: HashSet<_> = manifest_paths.keys().copied().collect();
        let marker_prefixes = versions
            .iter()
            .map(|version| self.marker_version_path(*version))
            .collect::<Vec<_>>();
        let marker_streams = marker_prefixes
            .into_iter()
            .map(|prefix| self.object_store.list(Some(prefix)));
        let marker_metadata = stream::iter(marker_streams)
            .flatten()
            .try_collect::<Vec<_>>()
            .await?;
        let parsed_markers = marker_metadata
            .into_iter()
            .map(|metadata| self.parse_retirement_marker_metadata(metadata))
            .collect::<Result<Vec<_>>>()?;

        let mut dependent_marker_paths = Vec::new();
        let mut anchor_paths = Vec::new();
        for version in &versions {
            let version_markers = parsed_markers
                .iter()
                .filter(|marker| marker.version == *version)
                .collect::<Vec<_>>();
            let has_committed = version_markers
                .iter()
                .any(|marker| marker.state == RetirementState::Committed);
            if !has_committed
                && !version_markers
                    .iter()
                    .any(|marker| marker.state == RetirementState::Sealed)
            {
                return Err(Error::internal(format!(
                    "cannot finalize retirement for version {version} without a durable marker"
                )));
            }
            for marker in version_markers {
                let is_anchor = marker.state == RetirementState::Committed
                    || (!has_committed && marker.state == RetirementState::Sealed);
                if is_anchor {
                    anchor_paths.push(marker.metadata.location.clone());
                } else {
                    dependent_marker_paths.push(marker.metadata.location.clone());
                }
            }
        }

        let mut lease_paths = Vec::new();
        for metadata in self.lease_metadata().await? {
            let (version, _) = parse_lease_metadata(&metadata)?;
            if versions.contains(&version) {
                lease_paths.push(metadata.location);
            }
        }
        // Leases and superseded marker states are dependent metadata. Keep at
        // least one terminal marker as the retry anchor until they are gone.
        self.delete_paths(lease_paths).await?;
        self.delete_paths(dependent_marker_paths).await?;
        self.delete_paths(anchor_paths).await
    }

    async fn delete_paths(&self, paths: Vec<Path>) -> Result<()> {
        stream::iter(paths)
            .map(Ok)
            .try_for_each_concurrent(self.object_store.io_parallelism(), |path| async move {
                match self.object_store.delete(&path).await {
                    Ok(()) => Ok(()),
                    Err(error) if error.is_not_found() => Ok(()),
                    Err(error) => Err(error),
                }
            })
            .await
    }
}

/// Verify that a durable tag or branch can be admitted for a version.
///
/// Callers check both before and after publishing their reference. Cleanup
/// publishes a draining marker before its final reference scan, so either the
/// reference is visible to that scan or the second admission check rejects and
/// rolls it back.
pub(super) async fn ensure_reference_admissible(
    refs: &Refs,
    branch: Option<&str>,
    version: u64,
) -> Result<()> {
    let branch_location = refs.base_location.find_branch(branch)?;
    let manifest = refs
        .commit_handler
        .resolve_version_location(&branch_location.path, version, &refs.object_store.inner)
        .await?;
    if !refs.object_store.exists(&manifest.path).await? {
        return Err(Error::VersionNotFound {
            message: format!("version {version} no longer exists and cannot be referenced"),
        });
    }

    let store = VersionLeaseStore::for_refs(refs, branch, None).await?;
    if store.has_active_retirement_marker(version).await? {
        return Err(Error::RefConflict {
            message: format!(
                "version {version} is retiring and cannot accept a new durable reference"
            ),
        });
    }
    if !refs.object_store.exists(&manifest.path).await? {
        return Err(Error::VersionNotFound {
            message: format!("version {version} no longer exists and cannot be referenced"),
        });
    }
    Ok(())
}

pub(super) async fn remove_branch_state(
    object_store: &ObjectStore,
    root_path: &Path,
    namespace: &str,
) -> Result<()> {
    for path in [
        root_path
            .clone()
            .join(LEASES_DIR)
            .join(namespace.to_string()),
        root_path
            .clone()
            .join(LEASE_GC_MARKERS_DIR)
            .join(namespace.to_string()),
    ] {
        match object_store.remove_dir_all(path).await {
            Ok(()) => {}
            Err(error) if error.is_not_found() => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

impl Dataset {
    /// Acquire a renewable advisory lease for this dataset version.
    ///
    /// Cleanup retains the version until the lease expires. Acquire the lease
    /// before starting a long-running read and renew it before expiration.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use lance::{Dataset, Result};
    /// # async fn read_historical(dataset: &Dataset) -> Result<()> {
    /// let historical = dataset.checkout_version(1).await?;
    /// let lease = historical
    ///     .acquire_version_lease(Duration::from_secs(60))
    ///     .await?;
    /// let _batch = historical.scan().try_into_batch().await?;
    /// lease.release().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn acquire_version_lease(&self, ttl: Duration) -> Result<VersionLease> {
        VersionLeaseStore::for_dataset(self)
            .await?
            .acquire(self.version().version, ttl)
            .await
    }
}

fn ttl_micros(ttl: Duration) -> Result<i64> {
    if ttl.is_zero() {
        return Err(Error::invalid_input(
            "version lease TTL must be greater than zero",
        ));
    }
    let ttl = TimeDelta::from_std(ttl).map_err(|error| {
        Error::invalid_input(format!(
            "version lease TTL {ttl:?} is out of range: {error}"
        ))
    })?;
    ttl.num_microseconds().ok_or_else(|| {
        Error::invalid_input(format!(
            "version lease TTL {ttl:?} cannot be represented in microseconds"
        ))
    })
}

fn expiration_from_ttl(observed_at: DateTime<Utc>, ttl: Duration) -> Result<DateTime<Utc>> {
    let conservative_ttl = ttl
        .checked_add(STORAGE_TIMESTAMP_PRECISION)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "version lease TTL {ttl:?} overflows the storage timestamp precision interval"
            ))
        })?;
    let ttl_delta = TimeDelta::from_std(conservative_ttl).map_err(|error| {
        Error::invalid_input(format!(
            "version lease TTL {ttl:?} is out of range: {error}"
        ))
    })?;
    observed_at.checked_add_signed(ttl_delta).ok_or_else(|| {
        Error::invalid_input(format!(
            "version lease TTL {ttl:?} overflows its expiration"
        ))
    })
}

fn draining_owner_is_active(
    draining_observed_at: DateTime<Utc>,
    current_observed_at: DateTime<Utc>,
) -> Result<bool> {
    Ok(
        expiration_from_ttl(draining_observed_at, DRAINING_OWNERSHIP_TIMEOUT)?
            > current_observed_at,
    )
}

fn retirement_marker_payload(manifest_paths: &[Path]) -> Result<Bytes> {
    if manifest_paths.is_empty() {
        return Err(Error::internal(
            "cannot create a retirement marker without a manifest identity",
        ));
    }
    let marker = RetirementMarker {
        manifest_paths: manifest_paths.iter().map(ToString::to_string).collect(),
    };
    serde_json::to_vec(&marker)
        .map(Bytes::from)
        .map_err(|error| Error::internal(format!("failed to serialize retirement marker: {error}")))
}

fn parse_lease_metadata(metadata: &ObjectMeta) -> Result<(u64, Duration)> {
    let file_name = metadata.location.filename().ok_or_else(|| {
        Error::corrupt_file(
            metadata.location.clone(),
            "version lease path has no filename",
        )
    })?;
    parse_lease_file_name(file_name)
        .map_err(|error| Error::corrupt_file(metadata.location.clone(), error))
}

fn parse_lease_file_name(file_name: &str) -> std::result::Result<(u64, Duration), String> {
    let stem = file_name.strip_suffix(LEASE_FILE_SUFFIX).ok_or_else(|| {
        format!("version lease file name '{file_name}' must end with {LEASE_FILE_SUFFIX}")
    })?;
    let mut parts = stem.splitn(3, '-');
    let version = parts
        .next()
        .ok_or_else(|| format!("version lease file name '{file_name}' has no version"))?
        .parse::<u64>()
        .map_err(|error| {
            format!("version lease file name '{file_name}' has invalid version: {error}")
        })?;
    let ttl_micros = parts
        .next()
        .ok_or_else(|| format!("version lease file name '{file_name}' has no TTL"))?
        .parse::<u64>()
        .map_err(|error| {
            format!("version lease file name '{file_name}' has invalid TTL: {error}")
        })?;
    if ttl_micros == 0 {
        return Err(format!(
            "version lease file name '{file_name}' has a zero TTL"
        ));
    }
    let lease_id = parts
        .next()
        .ok_or_else(|| format!("version lease file name '{file_name}' has no lease id"))?;
    Uuid::parse_str(lease_id).map_err(|error| {
        format!("version lease file name '{file_name}' has invalid lease id: {error}")
    })?;
    Ok((version, Duration::from_micros(ttl_micros)))
}

fn expired_lease_error(version: u64, expires_at: DateTime<Utc>) -> Error {
    Error::invalid_input(format!(
        "cannot renew expired version lease for version {version}: expired at {expires_at}"
    ))
}

fn retiring_version_error(version: u64) -> Error {
    Error::VersionNotFound {
        message: format!("version {version} is retiring and cannot accept a new lease"),
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::test::FailingProxyStore;
    use lance_io::object_store::WrappingObjectStore;
    use mock_instant::thread_local::MockClock;

    use super::*;

    fn memory_store() -> VersionLeaseStore {
        VersionLeaseStore {
            object_store: Arc::new(ObjectStore::memory()),
            leases_path: Path::from("leases"),
            markers_path: Path::from("markers"),
            manifest_path: None,
        }
    }

    fn manifest_paths(version: u64) -> HashMap<u64, Vec<Path>> {
        HashMap::from([(
            version,
            vec![Path::from(format!("manifests/{version}.manifest"))],
        )])
    }

    fn assert_send<T: Send>(_: T) {}

    #[test]
    fn retirement_futures_are_send() {
        let store = memory_store();
        let manifests = manifest_paths(42);
        assert_send(store.clone().recover_retirements());
        assert_send(store.fence_versions(&manifests));
    }

    #[test]
    fn parses_lease_file_name() {
        let lease_id = Uuid::nil().simple();
        assert_eq!(
            parse_lease_file_name(&format!("42-123-{lease_id}.lease")).unwrap(),
            (42, Duration::from_micros(123))
        );
    }

    #[tokio::test]
    async fn draining_lease_remains_renewable() {
        let store = memory_store();
        let mut lease = store.acquire(42, Duration::from_secs(60)).await.unwrap();
        let mut guard = store.fence_versions(&manifest_paths(42)).await.unwrap();

        assert!(
            store
                .active_versions_at(&guard.observed_at(), false)
                .await
                .unwrap()
                .contains(&42)
        );
        lease.renew(Duration::from_secs(60)).await.unwrap();
        guard.cancel_all().await.unwrap();
    }

    #[tokio::test]
    async fn lease_expiry_uses_storage_clock() {
        MockClock::set_system_time(Duration::from_secs(100));
        let store = memory_store();
        let _lease = store.acquire(42, Duration::from_secs(60)).await.unwrap();

        // Moving the cleanup host clock ahead does not affect the storage
        // timestamps used for lease liveness.
        MockClock::set_system_time(Duration::from_secs(160));
        let mut guard = store.fence_versions(&manifest_paths(42)).await.unwrap();
        assert!(
            store
                .active_versions_at(&guard.observed_at(), false)
                .await
                .unwrap()
                .contains(&42)
        );
        guard.cancel_all().await.unwrap();
    }

    #[tokio::test]
    async fn sealed_version_rejects_renewal() {
        let store = memory_store();
        let mut lease = store.acquire(42, Duration::from_secs(60)).await.unwrap();
        let mut guard = store.fence_versions(&manifest_paths(42)).await.unwrap();
        guard.seal_versions(&HashSet::from([42])).await.unwrap();

        let error = lease.renew(Duration::from_secs(60)).await.unwrap_err();
        assert!(matches!(error, Error::VersionNotFound { .. }));
        assert!(error.to_string().contains("retiring"), "{error}");
        guard.cancel_all().await.unwrap();
    }

    #[tokio::test]
    async fn committed_retirement_cannot_be_cancelled() {
        let store = memory_store();
        let mut guard = store.fence_versions(&manifest_paths(42)).await.unwrap();
        guard.seal_versions(&HashSet::from([42])).await.unwrap();
        guard.commit_versions(&HashSet::from([42])).await.unwrap();
        let committed_path = guard.fences[&42].committed_path.clone().unwrap();

        guard.cancel_all().await.unwrap();

        assert!(!guard.is_empty());
        assert!(store.object_store.exists(&committed_path).await.unwrap());
        let error = store
            .acquire(42, Duration::from_secs(60))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("retiring"), "{error}");
    }

    #[tokio::test]
    async fn terminal_marker_survives_lease_deletion_failure() {
        let failing_store = Arc::new(FailingProxyStore::new());
        let mut object_store = ObjectStore::memory();
        object_store.inner = failing_store.wrap("memory", Arc::clone(&object_store.inner));
        let store = VersionLeaseStore {
            object_store: Arc::new(object_store),
            leases_path: Path::from("leases"),
            markers_path: Path::from("markers"),
            manifest_path: None,
        };
        let lease = store.acquire(42, Duration::from_secs(60)).await.unwrap();
        let manifests = manifest_paths(42);
        let mut guard = store.fence_versions(&manifests).await.unwrap();
        guard.seal_versions(&HashSet::from([42])).await.unwrap();
        guard.commit_versions(&HashSet::from([42])).await.unwrap();
        let committed_path = guard.fences[&42].committed_path.clone().unwrap();
        failing_store.fail_when(
            "delete",
            LEASE_FILE_SUFFIX,
            "injected lease deletion failure",
        );

        let error = guard.finalize(&manifests).await.unwrap_err();

        assert!(
            error
                .to_string()
                .contains("injected lease deletion failure"),
            "{error}"
        );
        assert!(store.object_store.exists(&committed_path).await.unwrap());
        assert!(store.object_store.exists(&lease.path).await.unwrap());

        failing_store.clear_fail_when("delete", LEASE_FILE_SUFFIX);
        assert!(
            store
                .clone()
                .recover_retirements()
                .await
                .unwrap()
                .is_empty()
        );
        assert!(!store.object_store.exists(&committed_path).await.unwrap());
        assert!(!store.object_store.exists(&lease.path).await.unwrap());
    }

    #[test]
    fn lease_ttl_survives_coarse_storage_timestamps() {
        let storage_second = DateTime::from_timestamp(100, 0).unwrap();
        let acquired_at = storage_second + TimeDelta::try_milliseconds(900).unwrap();
        let cleanup_started_at = storage_second + TimeDelta::try_milliseconds(1_001).unwrap();
        let ttl = Duration::from_millis(900);

        assert!(
            cleanup_started_at < acquired_at + TimeDelta::from_std(ttl).unwrap(),
            "the requested TTL is still active"
        );
        let marker_last_modified = storage_second + TimeDelta::try_seconds(1).unwrap();
        assert!(
            expiration_from_ttl(storage_second, ttl).unwrap() > marker_last_modified,
            "coarse Last-Modified timestamps must not expire the lease early"
        );
    }

    #[tokio::test]
    async fn abandoned_drain_does_not_block_future_acquire() {
        let store = memory_store();
        let guard = store.fence_versions(&manifest_paths(42)).await.unwrap();
        let draining_path = guard.fences[&42].draining_path.clone().unwrap();

        // Dropping the owner proves this in-process drain cannot proceed to deletion.
        drop(guard);
        assert!(LOCALLY_ABANDONED_DRAINS.contains(&draining_path));

        store.acquire(42, Duration::from_secs(60)).await.unwrap();
    }

    #[test]
    fn draining_ownership_is_bounded() {
        let started_at = DateTime::from_timestamp(100, 0).unwrap();
        let ownership_expired_at =
            expiration_from_ttl(started_at, DRAINING_OWNERSHIP_TIMEOUT).unwrap();

        assert!(!draining_owner_is_active(started_at, ownership_expired_at).unwrap());
    }

    #[tokio::test]
    async fn sealed_retirement_is_recovered() {
        let store = memory_store();
        let manifest_paths = manifest_paths(42);
        let manifest_path = manifest_paths[&42][0].clone();
        store.object_store.put(&manifest_path, &[]).await.unwrap();
        let mut guard = store.fence_versions(&manifest_paths).await.unwrap();
        guard.seal_versions(&HashSet::from([42])).await.unwrap();
        drop(guard);

        let versions_to_resume = store.clone().recover_retirements().await.unwrap();
        assert_eq!(versions_to_resume, HashSet::from([42]));

        store.object_store.delete(&manifest_path).await.unwrap();
        assert!(
            store
                .clone()
                .recover_retirements()
                .await
                .unwrap()
                .is_empty()
        );
        assert!(store.version_marker_metadata(42).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn sealed_recovery_waits_for_active_lease() {
        let store = memory_store();
        let manifest_paths = manifest_paths(42);
        let manifest_path = manifest_paths[&42][0].clone();
        store.object_store.put(&manifest_path, &[]).await.unwrap();
        let lease = store.acquire(42, Duration::from_secs(60)).await.unwrap();
        let mut guard = store.fence_versions(&manifest_paths).await.unwrap();
        guard.seal_versions(&HashSet::from([42])).await.unwrap();
        drop(guard);

        assert!(
            store
                .clone()
                .recover_retirements()
                .await
                .unwrap()
                .is_empty()
        );
        assert!(store.object_store.exists(&manifest_path).await.unwrap());

        lease.release().await.unwrap();
        assert_eq!(
            store.clone().recover_retirements().await.unwrap(),
            HashSet::from([42])
        );
    }

    #[tokio::test]
    async fn branch_deletion_removes_incarnation_state() {
        let store = memory_store();
        let root = Path::from("dataset");
        let namespace = "branch-id";
        let lease_path = root.clone().join(LEASES_DIR).join(namespace).join("lease");
        let marker_path = root
            .clone()
            .join(LEASE_GC_MARKERS_DIR)
            .join(namespace)
            .join("marker");
        store.object_store.put(&lease_path, &[]).await.unwrap();
        store.object_store.put(&marker_path, &[]).await.unwrap();

        remove_branch_state(&store.object_store, &root, namespace)
            .await
            .unwrap();

        assert!(!store.object_store.exists(&lease_path).await.unwrap());
        assert!(!store.object_store.exists(&marker_path).await.unwrap());
    }
}
