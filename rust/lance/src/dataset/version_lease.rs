// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Advisory leases that protect dataset versions from cleanup.

use std::{collections::HashSet, sync::Arc, time::Duration};

use bytes::Bytes;
use chrono::{DateTime, TimeDelta, Utc};
use futures::{StreamExt, TryStreamExt, stream};
use lance_io::object_store::ObjectStore;
use object_store::{PutMode, PutOptions, path::Path};
use uuid::Uuid;

use super::{Dataset, refs::MAIN_BRANCH};
use crate::{Error, Result, utils::temporal::utc_now};

const LEASES_DIR: &str = "_refs/version_leases";
const LEASE_GC_MARKERS_DIR: &str = "_refs/version_lease_gc";
const LEASE_FILE_SUFFIX: &str = ".lease";

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
    /// Renewal must complete before the current expiration. It fails if the
    /// lease has expired or cleanup has fenced the version for deletion.
    pub async fn renew(&mut self, ttl: Duration) -> Result<()> {
        let now = utc_now();
        if now >= self.expires_at {
            return Err(Error::invalid_input(format!(
                "cannot renew expired version lease for version {}: expired at {}",
                self.version, self.expires_at
            )));
        }
        let expires_at = expiration_from_ttl(now, ttl)?;
        self.store.ensure_not_fenced(self.version).await?;

        // Publish the replacement before removing the old file so cleanup never
        // observes a gap between a timely renewal and its predecessor.
        let path = self
            .store
            .create_lease_file(self.version, expires_at)
            .await?;
        if let Err(error) = self.store.ensure_not_fenced(self.version).await {
            let _ = self.store.object_store.delete(&path).await;
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
        self.path = path;
        self.expires_at = expires_at;
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
        })
    }

    async fn acquire(&self, version: u64, ttl: Duration) -> Result<VersionLease> {
        self.ensure_not_fenced(version).await?;
        let expires_at = expiration_from_ttl(utc_now(), ttl)?;
        let path = self.create_lease_file(version, expires_at).await?;
        if let Err(error) = self.ensure_not_fenced(version).await {
            let _ = self.object_store.delete(&path).await;
            return Err(error);
        }
        Ok(VersionLease {
            store: self.clone(),
            path,
            version,
            expires_at,
            released: false,
        })
    }

    async fn create_lease_file(&self, version: u64, expires_at: DateTime<Utc>) -> Result<Path> {
        let path = self.leases_path.clone().join(format!(
            "{}-{}-{}{}",
            version,
            expires_at.timestamp_micros(),
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
        Ok(path)
    }

    pub(super) async fn active_versions(&self, remove_expired: bool) -> Result<HashSet<u64>> {
        let lease_files = match self.object_store.read_dir(self.leases_path.clone()).await {
            Ok(files) => files,
            Err(error) if error.is_not_found() => return Ok(HashSet::new()),
            Err(error) => return Err(error),
        };
        let now_micros = utc_now().timestamp_micros();
        let mut active_versions = HashSet::new();
        let mut expired_paths = Vec::new();

        for file_name in lease_files {
            let (version, expires_at_micros) =
                parse_lease_file_name(&file_name).map_err(|error| {
                    Error::corrupt_file(self.leases_path.clone().join(file_name.as_str()), error)
                })?;
            if expires_at_micros > now_micros {
                active_versions.insert(version);
            } else if remove_expired {
                expired_paths.push(self.leases_path.clone().join(file_name));
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

    pub(super) async fn fence_versions(&self, versions: &HashSet<u64>) -> Result<()> {
        stream::iter(versions.iter().copied())
            .map(Ok)
            .try_for_each_concurrent(self.object_store.io_parallelism(), |version| async move {
                let path = self.markers_path.clone().join(version.to_string());
                match self
                    .object_store
                    .inner
                    .put_opts(
                        &path,
                        Bytes::new().into(),
                        PutOptions {
                            mode: PutMode::Create,
                            ..Default::default()
                        },
                    )
                    .await
                {
                    Ok(_) => Ok(()),
                    Err(object_store::Error::AlreadyExists { .. })
                    | Err(object_store::Error::Precondition { .. }) => Ok(()),
                    Err(error) => Err(Error::from(error)),
                }
            })
            .await
    }

    async fn ensure_not_fenced(&self, version: u64) -> Result<()> {
        let marker = self.markers_path.clone().join(version.to_string());
        if self.object_store.exists(&marker).await? {
            Err(Error::VersionNotFound {
                message: format!(
                    "version {} has been fenced for cleanup and cannot be leased",
                    version
                ),
            })
        } else {
            Ok(())
        }
    }
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

fn expiration_from_ttl(now: DateTime<Utc>, ttl: Duration) -> Result<DateTime<Utc>> {
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
    now.checked_add_signed(ttl).ok_or_else(|| {
        Error::invalid_input(format!(
            "version lease TTL {ttl:?} overflows its expiration"
        ))
    })
}

fn parse_lease_file_name(file_name: &str) -> std::result::Result<(u64, i64), String> {
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
    let expires_at_micros = parts
        .next()
        .ok_or_else(|| format!("version lease file name '{file_name}' has no expiration"))?
        .parse::<i64>()
        .map_err(|error| {
            format!("version lease file name '{file_name}' has invalid expiration: {error}")
        })?;
    let lease_id = parts
        .next()
        .ok_or_else(|| format!("version lease file name '{file_name}' has no lease id"))?;
    Uuid::parse_str(lease_id).map_err(|error| {
        format!("version lease file name '{file_name}' has invalid lease id: {error}")
    })?;
    Ok((version, expires_at_micros))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lease_file_name() {
        let lease_id = Uuid::nil().simple();
        assert_eq!(
            parse_lease_file_name(&format!("42-123-{lease_id}.lease")).unwrap(),
            (42, 123)
        );
    }
}
