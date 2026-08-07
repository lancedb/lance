// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Advisory leases that protect dataset versions from cleanup.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use bytes::Bytes;
use chrono::{DateTime, TimeDelta, Utc};
use futures::{StreamExt, TryStreamExt, stream};
use lance_io::object_store::ObjectStore;
use object_store::{ObjectMeta, ObjectStoreExt, PutMode, PutOptions, path::Path};
use uuid::Uuid;

use super::{Dataset, refs::MAIN_BRANCH};
use crate::{Error, Result};

const LEASES_DIR: &str = "_refs/version_leases";
const LEASE_GC_MARKERS_DIR: &str = "_refs/version_lease_gc";
const LEASE_FILE_SUFFIX: &str = ".lease";
const DRAINING_MARKER_SUFFIX: &str = ".draining";
const SEALED_MARKER_SUFFIX: &str = ".sealed";

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
    observed_at: DateTime<Utc>,
}

/// Per-cleanup retirement markers for versions selected for deletion.
///
/// Draining markers reject new leases while allowing already-admitted leases
/// to renew. Sealed markers reject both acquisition and renewal immediately
/// before deletion. Unique operation markers keep cancellation and finalization
/// safe when multiple cleanups overlap.
#[derive(Debug)]
pub(super) struct RetirementGuard {
    store: VersionLeaseStore,
    fences: HashMap<u64, RetirementFence>,
}

impl VersionLeaseStore {
    pub(super) async fn for_dataset(dataset: &Dataset) -> Result<Self> {
        let root = dataset.refs.root()?;
        let branch = dataset.manifest.branch.as_deref();
        let branch_identifier = dataset.branches().get_identifier(branch).await?;
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
            object_store: Arc::clone(&dataset.object_store),
            leases_path: root.path.clone().join(LEASES_DIR).join(namespace),
            markers_path: root.path.join(LEASE_GC_MARKERS_DIR).join(namespace),
            manifest_path: Some(dataset.manifest_location.path.clone()),
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

    pub(super) async fn fence_versions(&self, versions: &HashSet<u64>) -> Result<RetirementGuard> {
        let operation_id = Uuid::new_v4().simple().to_string();
        let results = stream::iter(versions.iter().copied())
            .map(|version| {
                let path = self
                    .marker_version_path(version)
                    .join(format!("{operation_id}{DRAINING_MARKER_SUFFIX}"));
                async move {
                    let metadata = self.create_marker(&path).await?;
                    Ok::<_, Error>((version, path, metadata.last_modified))
                }
            })
            .buffer_unordered(self.object_store.io_parallelism())
            .collect::<Vec<_>>()
            .await;

        let mut fences = HashMap::with_capacity(results.len());
        let mut first_error = None;
        for result in results {
            match result {
                Ok((version, path, observed_at)) => {
                    fences.insert(
                        version,
                        RetirementFence {
                            draining_path: Some(path),
                            sealed_path: None,
                            observed_at,
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
        } else if self.has_marker(version, None).await? {
            Err(retiring_version_error(version))
        } else {
            Ok(())
        }
    }

    async fn ensure_not_sealed(&self, version: u64) -> Result<()> {
        if self.has_marker(version, Some(SEALED_MARKER_SUFFIX)).await? {
            Err(retiring_version_error(version))
        } else {
            Ok(())
        }
    }

    async fn has_marker(&self, version: u64, suffix: Option<&str>) -> Result<bool> {
        let mut markers = self
            .object_store
            .list(Some(self.marker_version_path(version)));
        while let Some(metadata) = markers.try_next().await? {
            if suffix.is_none_or(|suffix| {
                metadata
                    .location
                    .filename()
                    .is_some_and(|name| name.ends_with(suffix))
            }) {
                return Ok(true);
            }
        }
        Ok(false)
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

    async fn create_marker(&self, path: &Path) -> Result<ObjectMeta> {
        self.object_store
            .inner
            .put_opts(
                path,
                Bytes::new().into(),
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
        let marker_paths = versions
            .iter()
            .filter_map(|version| {
                self.fences.get(version).and_then(|fence| {
                    fence.draining_path.as_ref().and_then(|draining_path| {
                        let file_name = draining_path.filename()?;
                        let operation_id = file_name.strip_suffix(DRAINING_MARKER_SUFFIX)?;
                        Some((
                            *version,
                            self.store
                                .marker_version_path(*version)
                                .join(format!("{operation_id}{SEALED_MARKER_SUFFIX}")),
                        ))
                    })
                })
            })
            .collect::<Vec<_>>();

        let store = self.store.clone();
        let results = stream::iter(marker_paths)
            .map(move |(version, path)| {
                let store = store.clone();
                async move {
                    let metadata = store.create_marker(&path).await?;
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

    pub(super) async fn cancel_versions(&mut self, versions: &HashSet<u64>) -> Result<()> {
        let paths = versions
            .iter()
            .filter_map(|version| self.fences.get(version))
            .flat_map(|fence| {
                [fence.draining_path.clone(), fence.sealed_path.clone()]
                    .into_iter()
                    .flatten()
            })
            .collect::<Vec<_>>();
        self.store.delete_paths(paths).await?;
        self.fences.retain(|version, _| !versions.contains(version));
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
        for version in self.fences.keys() {
            let paths = manifest_paths.get(version).ok_or_else(|| {
                Error::internal(format!(
                    "missing manifest identity for retiring version {version}"
                ))
            })?;
            for path in paths {
                if self.store.object_store.exists(path).await? {
                    return Err(Error::internal(format!(
                        "cannot finalize retirement for version {version}: manifest {path} still exists"
                    )));
                }
            }
        }

        let versions: HashSet<_> = self.fences.keys().copied().collect();
        let marker_prefixes = versions
            .iter()
            .map(|version| self.store.marker_version_path(*version))
            .collect::<Vec<_>>();
        let marker_streams = marker_prefixes
            .into_iter()
            .map(|prefix| self.store.object_store.list(Some(prefix)));
        let mut paths = stream::iter(marker_streams)
            .flatten()
            .map_ok(|metadata| metadata.location)
            .try_collect::<Vec<_>>()
            .await?;
        for metadata in self.store.lease_metadata().await? {
            let (version, _) = parse_lease_metadata(&metadata)?;
            if versions.contains(&version) {
                paths.push(metadata.location);
            }
        }
        self.store.delete_paths(paths).await?;
        self.fences.clear();
        Ok(())
    }
}

impl VersionLeaseStore {
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
    let ttl = TimeDelta::from_std(ttl).map_err(|error| {
        Error::invalid_input(format!(
            "version lease TTL {ttl:?} is out of range: {error}"
        ))
    })?;
    observed_at.checked_add_signed(ttl).ok_or_else(|| {
        Error::invalid_input(format!(
            "version lease TTL {ttl:?} overflows its expiration"
        ))
    })
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
        let mut guard = store.fence_versions(&HashSet::from([42])).await.unwrap();

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
        let mut guard = store.fence_versions(&HashSet::from([42])).await.unwrap();
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
        let mut guard = store.fence_versions(&HashSet::from([42])).await.unwrap();
        guard.seal_versions(&HashSet::from([42])).await.unwrap();

        let error = lease.renew(Duration::from_secs(60)).await.unwrap_err();
        assert!(matches!(error, Error::VersionNotFound { .. }));
        assert!(error.to_string().contains("retiring"), "{error}");
        guard.cancel_all().await.unwrap();
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
