// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::future::Future;
use std::time::Duration;

use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::future::BoxFuture;
use itertools::Itertools;
use lance_io::object_store::ObjectStore;
use lance_table::io::commit::CommitHandler;
use lance_table::io::manifest::read_manifest;
use object_store::{Error as ObjectStoreError, ObjectStoreExt, PutMode, PutOptions, path::Path};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::dataset::branch_location::{BRANCH_GENERATIONS_DIR, BranchLocation};
use crate::dataset::refs::Ref::{Tag, Version, VersionNumber};
use crate::utils::temporal::utc_now;
use crate::{Error, Result};
use serde::de::DeserializeOwned;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Formatter;
use uuid::Uuid;

pub const MAIN_BRANCH: &str = "main";

const REF_MUTATION_LOCK_FILE: &str = "mutation.json";
const REF_MUTATION_LEASES_DIR: &str = "mutation_leases";
const TAG_MUTATION_INTENTS_DIR: &str = "tag_mutations";
const REF_CATALOG_DIR: &str = "catalog";
const LEASE_FILE: &str = "lease.json";
const LEASE_HEARTBEATS_DIR: &str = "heartbeats";
const LEASE_RELEASED_FILE: &str = "released.json";
const LEASE_PUBLICATION_FILE: &str = "publication.json";
const LEASE_RECONCILED_FILE: &str = "reconciled.json";
const REF_MUTATION_LOCK_TIMEOUT: Duration = Duration::from_secs(30);
const REF_MUTATION_LOCK_RETRY_DELAY: Duration = Duration::from_millis(10);
const REF_MUTATION_LEASE_DURATION_MILLIS: i64 = 30_000;
const REF_MUTATION_LEASE_RENEW_INTERVAL: Duration = Duration::from_secs(10);

/// Lance Ref
#[derive(Debug, Clone)]
pub enum Ref {
    // Version number points of the current branch
    VersionNumber(u64),
    // This is a global version identifier present as (branch_name, version_number)
    // if branch_name is None, it points to the main branch
    // if version_number is None, it points to the latest version
    Version(Option<String>, Option<u64>),
    // Tag name points to the global version identifier, could be considered as an alias of specific global version
    Tag(String),
}

impl From<u64> for Ref {
    fn from(reference: u64) -> Self {
        VersionNumber(reference)
    }
}

impl From<&str> for Ref {
    fn from(reference: &str) -> Self {
        Tag(reference.to_string())
    }
}

impl From<(&str, u64)> for Ref {
    fn from(reference: (&str, u64)) -> Self {
        Version(standardize_branch(reference.0), Some(reference.1))
    }
}

impl From<(Option<&str>, Option<u64>)> for Ref {
    fn from(reference: (Option<&str>, Option<u64>)) -> Self {
        Version(reference.0.and_then(standardize_branch), reference.1)
    }
}

impl From<(&str, Option<u64>)> for Ref {
    fn from(reference: (&str, Option<u64>)) -> Self {
        Version(standardize_branch(reference.0), reference.1)
    }
}

impl fmt::Display for Ref {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Version(branch, version_number) => {
                let version_str = version_number
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "latest".to_string());
                write!(f, "{}:{}", normalize_branch(branch.as_deref()), version_str)
            }
            VersionNumber(version_number) => write!(f, "{}", version_number),
            Tag(tag_name) => write!(f, "{}", tag_name),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Refs {
    pub(crate) object_store: Arc<ObjectStore>,
    pub(crate) commit_handler: Arc<dyn CommitHandler>,
    pub(crate) base_location: BranchLocation,
}

impl Refs {
    pub fn new(
        object_store: Arc<ObjectStore>,
        commit_handler: Arc<dyn CommitHandler>,
        base_location: BranchLocation,
    ) -> Self {
        Self {
            object_store,
            commit_handler,
            base_location,
        }
    }

    pub fn tags(&self) -> Tags<'_> {
        Tags { refs: self }
    }

    pub fn branches(&self) -> Branches<'_> {
        Branches { refs: self }
    }

    pub fn base(&self) -> &Path {
        &self.base_location.path
    }

    pub fn root(&self) -> Result<BranchLocation> {
        self.base_location.find_main()
    }

    pub(super) fn run_mutation<'a, T, F, Fut>(&'a self, mutation: F) -> BoxFuture<'a, Result<T>>
    where
        T: Send + 'a,
        F: FnOnce(DurableLeaseFence) -> Fut + Send + 'a,
        Fut: Future<Output = Result<T>> + Send + 'a,
    {
        Box::pin(async move {
            let mut lease = RefMutationLease::acquire(self).await?;
            let fence = lease.handle.fence()?;
            let mutation_result = Self::drive_with_lease(&mut lease.handle, mutation(fence)).await;
            Self::finish_mutation(&mut lease, mutation_result).await
        })
    }

    fn run_tag_mutation<'a, T, F, Fut>(&'a self, mutation: F) -> BoxFuture<'a, Result<T>>
    where
        T: Send + 'a,
        F: FnOnce(DurableLeaseFence) -> Fut + Send + 'a,
        Fut: Future<Output = Result<T>> + Send + 'a,
    {
        Box::pin(async move {
            let mut intent = TagMutationIntent::acquire(self).await?;
            let mutation_result =
                Self::drive_with_lease(&mut intent.handle, self.run_mutation(mutation)).await;
            if let Err(error) = intent.release().await {
                log::warn!("Failed to release tag mutation intent: {}", error);
            }
            mutation_result
        })
    }

    fn run_branch_deletion<'a, T, F, Fut>(&'a self, mutation: F) -> BoxFuture<'a, Result<T>>
    where
        T: Send + 'a,
        F: FnOnce(DurableLeaseFence) -> Fut + Send + 'a,
        Fut: Future<Output = Result<T>> + Send + 'a,
    {
        Box::pin(async move {
            // Give already-scheduled tag mutations a chance to publish their durable intent before
            // deletion takes the exclusive mutation lock. This gives durable references priority
            // when both operations begin together.
            tokio::task::yield_now().await;
            let deadline = tokio::time::Instant::now() + REF_MUTATION_LOCK_TIMEOUT;
            let mut lease = loop {
                while self.has_active_tag_mutation_intents().await? {
                    if tokio::time::Instant::now() >= deadline {
                        return Err(Error::RefConflict {
                            message: format!(
                                "a tag mutation did not finish within {} seconds",
                                REF_MUTATION_LOCK_TIMEOUT.as_secs()
                            ),
                        });
                    }
                    tokio::time::sleep(REF_MUTATION_LOCK_RETRY_DELAY).await;
                }

                let mut lease = RefMutationLease::acquire(self).await?;
                if !self.has_active_tag_mutation_intents().await? {
                    break lease;
                }
                lease.release().await?;
                tokio::time::sleep(REF_MUTATION_LOCK_RETRY_DELAY).await;
            };

            let fence = lease.handle.fence()?;
            let mutation_result = Self::drive_with_lease(&mut lease.handle, mutation(fence)).await;
            Self::finish_mutation(&mut lease, mutation_result).await
        })
    }

    async fn drive_with_lease<T, F>(handle: &mut DurableLeaseHandle, mutation: F) -> Result<T>
    where
        T: Send,
        F: Future<Output = Result<T>> + Send,
    {
        // Lease acquisition can perform synchronous cleanup. Renew and revalidate after that
        // work so an expired owner never polls the protected mutation.
        handle.renew().await?;
        tokio::pin!(mutation);
        loop {
            tokio::select! {
                biased;
                _ = tokio::time::sleep(REF_MUTATION_LEASE_RENEW_INTERVAL) => {
                    handle.renew().await?;
                }
                result = &mut mutation => return result,
            }
        }
    }

    async fn has_active_tag_mutation_intents(&self) -> Result<bool> {
        let base_path = base_tag_mutation_intents_path(&self.root()?.path);
        for file_name in self.object_store.read_dir(base_path.clone()).await? {
            let path = base_path.clone().join(file_name);
            if path.extension() == Some("json") {
                // The previous protocol wrote empty intent objects. They carry no
                // recoverable ownership and must not permanently block deletion.
                log::warn!("Ignoring stale legacy tag mutation intent {}", path);
                continue;
            }
            if DurableLeaseHandle::is_active(&self.object_store, &path).await? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    async fn finish_mutation<T>(
        lease: &mut RefMutationLease,
        mutation_result: Result<T>,
    ) -> Result<T> {
        if let Err(error) = lease.release().await {
            log::warn!("Failed to release reference mutation lease: {}", error);
        }
        mutation_result
    }
}

struct TagMutationIntent {
    handle: DurableLeaseHandle,
}

impl TagMutationIntent {
    async fn acquire(refs: &Refs) -> Result<Self> {
        let path = base_tag_mutation_intents_path(&refs.root()?.path)
            .join(Uuid::new_v4().simple().to_string());
        let state = DurableLeaseState::acquired(Uuid::new_v4().simple().to_string(), 1)?;
        if !create_lease_file(&refs.object_store, &path, &state).await? {
            return Err(Error::RefConflict {
                message: "tag mutation intent identifier already exists".to_string(),
            });
        }
        Ok(Self {
            handle: DurableLeaseHandle::new(refs.object_store.clone(), path, state, None),
        })
    }

    async fn release(&mut self) -> Result<()> {
        if !self.handle.is_held {
            return Ok(());
        }
        let path = self.handle.path.clone();
        let object_store = self.handle.object_store.clone();
        self.handle.release().await?;
        if let Err(error) = object_store.remove_dir_all(path).await
            && !matches!(error, Error::NotFound { .. })
        {
            return Err(error);
        }
        Ok(())
    }
}

// Lease epochs and heartbeats are immutable because not every supported object store implements
// conditional updates. Atomic create elects one owner for each increasing epoch; renewal checks
// that epoch is still the newest fence, and an expired owner cannot resume after a takeover.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DurableLeaseState {
    owner: String,
    #[serde(default)]
    epoch: u64,
    #[serde(default)]
    expires_at_millis: i64,
}

impl DurableLeaseState {
    fn acquired(owner: String, epoch: u64) -> Result<Self> {
        Ok(Self {
            owner,
            epoch,
            expires_at_millis: lease_expiry_millis()?,
        })
    }

    fn renewed(&self) -> Result<Self> {
        if !self.is_active() {
            return Err(Error::RefConflict {
                message: format!(
                    "reference mutation lease epoch {} expired before renewal",
                    self.epoch
                ),
            });
        }
        Ok(Self {
            expires_at_millis: lease_expiry_millis()?,
            ..self.clone()
        })
    }

    fn released(&self) -> Self {
        Self {
            expires_at_millis: 0,
            ..self.clone()
        }
    }

    fn is_active(&self) -> bool {
        self.expires_at_millis > utc_now().timestamp_millis()
    }

    fn serialize(&self) -> Result<Bytes> {
        Ok(Bytes::from(serde_json::to_vec(self)?))
    }
}

fn lease_expiry_millis() -> Result<i64> {
    utc_now()
        .timestamp_millis()
        .checked_add(REF_MUTATION_LEASE_DURATION_MILLIS)
        .ok_or_else(|| Error::internal("reference mutation lease expiry overflow"))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DurableRefPublication {
    epoch: u64,
    path: String,
    body: String,
}

impl DurableRefPublication {
    fn catalog(&self) -> Result<(Path, Path, RefCatalog)> {
        let path = Path::parse(&self.path)?;
        let catalog: RefCatalog = serde_json::from_str(&self.body)?;
        if catalog.mutation_epoch != self.epoch {
            return Err(Error::internal(format!(
                "reference catalog epoch {} does not match publication epoch {}",
                catalog.mutation_epoch, self.epoch
            )));
        }
        let root = ref_catalog_root_from_version_path(&path)?;
        let expected_path = ref_catalog_version_path(&root, self.epoch);
        if path != expected_path {
            return Err(Error::internal(format!(
                "reference catalog publication path {} does not match {}",
                path, expected_path
            )));
        }
        Ok((path, root, catalog))
    }

    async fn is_committed(&self, object_store: &ObjectStore) -> Result<bool> {
        let (path, _, expected) = self.catalog()?;
        let result = match object_store.inner.get(&path).await {
            Ok(result) => result,
            Err(ObjectStoreError::NotFound { .. }) => return Ok(false),
            Err(error) => return Err(error.into()),
        };
        let actual: RefCatalog = serde_json::from_slice(&result.bytes().await?)?;
        Ok(actual.mutation_epoch == expected.mutation_epoch
            && actual.tags == expected.tags
            && actual.branches == expected.branches)
    }

    async fn apply(&self, object_store: &ObjectStore) -> Result<()> {
        let (path, root, mut catalog) = self.catalog()?;

        // Snapshot released-format refs immediately before the atomic create. A legacy mutation
        // that completed before this publication becomes part of the boundary and cannot override
        // the newer catalog; a later legacy mutation differs from this baseline and is reconciled.
        // A concurrent legacy mutation overlaps this publication and may be ordered either way.
        catalog.legacy_baseline = read_legacy_ref_state(object_store, &root).await?;

        // A complete catalog snapshot is the atomically discoverable state for point reads and
        // enumeration. Delayed lower epochs are harmless because readers select the greatest
        // epoch, and the next successful publication compacts them. Everything after this create
        // is best-effort cleanup: a committed mutation must never report failure.
        create_ref_catalog(object_store, &path, &catalog).await?;
        if let Err(error) = compact_ref_catalog(object_store, &root, self.epoch).await {
            log::warn!("Failed to compact superseded reference catalogs: {}", error);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RefCatalog {
    #[serde(rename = "_mutationEpoch")]
    mutation_epoch: u64,
    #[serde(default)]
    tags: HashMap<String, serde_json::Value>,
    #[serde(default)]
    branches: HashMap<String, serde_json::Value>,
    // Legacy files remain as a durable migration baseline. Current readers compare the live
    // files with this snapshot so a released writer cannot successfully update a flat ref and
    // then have that update silently ignored by the catalog.
    #[serde(rename = "_legacyBaseline", default)]
    legacy_baseline: LegacyRefState,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LegacyRefState {
    #[serde(default)]
    tags: HashMap<String, serde_json::Value>,
    #[serde(default)]
    branches: HashMap<String, serde_json::Value>,
}

impl RefCatalog {
    fn reconcile_legacy_changes(&mut self, legacy: LegacyRefState) {
        reconcile_legacy_entries(&mut self.tags, &self.legacy_baseline.tags, &legacy.tags);
        reconcile_legacy_entries(
            &mut self.branches,
            &self.legacy_baseline.branches,
            &legacy.branches,
        );
        self.legacy_baseline = legacy;
    }
}

fn reconcile_legacy_entries(
    current: &mut HashMap<String, serde_json::Value>,
    baseline: &HashMap<String, serde_json::Value>,
    legacy: &HashMap<String, serde_json::Value>,
) {
    for name in baseline.keys() {
        if !legacy.contains_key(name) {
            current.remove(name);
        }
    }
    for (name, value) in legacy {
        if baseline.get(name) != Some(value) {
            current.insert(name.clone(), value.clone());
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LeaseReconciliationState {
    through_epoch: u64,
}

#[derive(Clone)]
pub(super) struct DurableLeaseFence {
    object_store: Arc<ObjectStore>,
    path: Path,
    fence_path: Path,
    epoch: u64,
}

impl DurableLeaseFence {
    async fn ensure_current(&self) -> Result<()> {
        if latest_lease_epoch(&self.object_store, &self.fence_path).await? != Some(self.epoch)
            || !DurableLeaseHandle::is_active(&self.object_store, &self.path).await?
        {
            return Err(Error::RefConflict {
                message: format!(
                    "reference mutation lease epoch {} lost its fence",
                    self.epoch
                ),
            });
        }
        Ok(())
    }

    async fn publish<T>(&self, path: &Path, contents: Option<&T>) -> Result<()>
    where
        T: Serialize,
    {
        let address = RefAddress::from_path(path)?;
        let mut catalog = read_ref_catalog(&self.object_store, &address.root).await?;
        catalog.mutation_epoch = self.epoch;
        let entries = match address.kind {
            RefKind::Tag => &mut catalog.tags,
            RefKind::Branch => &mut catalog.branches,
        };
        match contents {
            Some(contents) => {
                entries.insert(address.name, serde_json::to_value(contents)?);
            }
            None => {
                entries.remove(&address.name);
            }
        }
        let publication_path = ref_catalog_version_path(&address.root, self.epoch);
        let publication = DurableRefPublication {
            epoch: self.epoch,
            path: publication_path.to_string(),
            body: serde_json::to_string_pretty(&catalog)?,
        };
        let intent_path = self.path.clone().join(LEASE_PUBLICATION_FILE);
        create_serialized_file(&self.object_store, &intent_path, &publication).await?;
        self.apply_committed_publication(&publication).await
    }

    async fn apply_committed_publication(&self, publication: &DurableRefPublication) -> Result<()> {
        let had_current_fence = match self.ensure_current().await {
            Ok(()) => true,
            Err(error) if matches!(&error, Error::RefConflict { .. }) => {
                if publication.is_committed(&self.object_store).await? {
                    log::warn!(
                        "Reference mutation lease lost its fence after a successor committed its publication: {}",
                        error
                    );
                    return Ok(());
                }
                // publication.json is the durable commit decision. A successor will replay it,
                // so the original caller must complete the same outcome instead of reporting a
                // conflict for an operation that can still become visible.
                log::warn!(
                    "Reference mutation lease lost its fence after recording a commit-ready publication; completing it: {}",
                    error
                );
                false
            }
            Err(error) => return Err(error),
        };

        publication.apply(&self.object_store).await?;
        if had_current_fence && let Err(error) = self.ensure_current().await {
            log::warn!(
                "Reference mutation lease lost its fence after catalog publication: {}",
                error
            );
        }
        Ok(())
    }

    async fn reconcile_prior_publications(&self) -> Result<()> {
        let mut epochs = self
            .object_store
            .read_dir(self.fence_path.clone())
            .await?
            .into_iter()
            .filter_map(|entry| entry.parse::<u64>().ok())
            .filter(|epoch| *epoch < self.epoch)
            .collect_vec();
        epochs.sort_unstable();

        let mut reconciled_through = 0;
        for epoch in epochs.iter().rev() {
            let path = lease_epoch_path(&self.fence_path, *epoch).join(LEASE_RECONCILED_FILE);
            if let Some(state) =
                read_serialized_file::<LeaseReconciliationState>(&self.object_store, &path).await?
            {
                reconciled_through = state.through_epoch;
                break;
            }
        }

        for epoch in epochs {
            if epoch <= reconciled_through {
                continue;
            }
            let path = lease_epoch_path(&self.fence_path, epoch).join(LEASE_PUBLICATION_FILE);
            if let Some(publication) =
                read_serialized_file::<DurableRefPublication>(&self.object_store, &path).await?
            {
                publication.apply(&self.object_store).await?;
            }
        }

        let state = LeaseReconciliationState {
            through_epoch: self.epoch.checked_sub(1).ok_or_else(|| {
                Error::internal("reference mutation lease epoch must be greater than zero")
            })?,
        };
        create_serialized_file(
            &self.object_store,
            &self.path.clone().join(LEASE_RECONCILED_FILE),
            &state,
        )
        .await?;
        self.ensure_current().await
    }
}

struct DurableLeaseHandle {
    object_store: Arc<ObjectStore>,
    path: Path,
    state: DurableLeaseState,
    fence_path: Option<Path>,
    is_held: bool,
}

impl DurableLeaseHandle {
    fn new(
        object_store: Arc<ObjectStore>,
        path: Path,
        state: DurableLeaseState,
        fence_path: Option<Path>,
    ) -> Self {
        Self {
            object_store,
            path,
            state,
            fence_path,
            is_held: true,
        }
    }

    fn fence(&self) -> Result<DurableLeaseFence> {
        let fence_path = self.fence_path.clone().ok_or_else(|| {
            Error::internal("reference publication requested from an unfenced lease")
        })?;
        Ok(DurableLeaseFence {
            object_store: self.object_store.clone(),
            path: self.path.clone(),
            fence_path,
            epoch: self.state.epoch,
        })
    }

    async fn renew(&mut self) -> Result<()> {
        let next_state = self.state.renewed()?;
        if let Some(fence_path) = self.fence_path.as_ref()
            && latest_lease_epoch(&self.object_store, fence_path).await? != Some(self.state.epoch)
        {
            return Err(Error::RefConflict {
                message: format!(
                    "reference mutation lease epoch {} lost its fence",
                    self.state.epoch
                ),
            });
        }
        let heartbeat_path = self
            .path
            .clone()
            .join(LEASE_HEARTBEATS_DIR)
            .join(format!("{}.json", Uuid::new_v4().simple()));
        self.object_store
            .inner
            .put_opts(
                &heartbeat_path,
                next_state.serialize()?.into(),
                PutOptions {
                    mode: PutMode::Create,
                    ..Default::default()
                },
            )
            .await?;
        if let Some(fence_path) = self.fence_path.as_ref()
            && latest_lease_epoch(&self.object_store, fence_path).await? != Some(self.state.epoch)
        {
            return Err(Error::RefConflict {
                message: format!(
                    "reference mutation lease epoch {} lost its fence",
                    self.state.epoch
                ),
            });
        }
        if !next_state.is_active() {
            return Err(Error::RefConflict {
                message: format!(
                    "reference mutation lease epoch {} expired during renewal",
                    self.state.epoch
                ),
            });
        }
        self.state = next_state;
        Ok(())
    }

    async fn release_owned(
        object_store: &ObjectStore,
        path: &Path,
        state: &DurableLeaseState,
    ) -> Result<()> {
        let released_path = path.clone().join(LEASE_RELEASED_FILE);
        match object_store
            .inner
            .put_opts(
                &released_path,
                state.released().serialize()?.into(),
                PutOptions {
                    mode: PutMode::Create,
                    ..Default::default()
                },
            )
            .await
        {
            Ok(_)
            | Err(ObjectStoreError::AlreadyExists { .. } | ObjectStoreError::Precondition { .. }) => {
                Ok(())
            }
            Err(error) => Err(error.into()),
        }
    }

    async fn is_active(object_store: &ObjectStore, path: &Path) -> Result<bool> {
        if object_store
            .exists(&path.clone().join(LEASE_RELEASED_FILE))
            .await?
        {
            return Ok(false);
        }

        if read_lease_state(object_store, &path.clone().join(LEASE_FILE))
            .await?
            .is_some_and(|state| state.is_active())
        {
            return Ok(true);
        }
        let heartbeats_path = path.clone().join(LEASE_HEARTBEATS_DIR);
        for heartbeat in object_store.read_dir(heartbeats_path.clone()).await? {
            if read_lease_state(object_store, &heartbeats_path.clone().join(heartbeat))
                .await?
                .is_some_and(|state| state.is_active())
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    async fn release(&mut self) -> Result<()> {
        if !self.is_held {
            return Ok(());
        }
        Self::release_owned(&self.object_store, &self.path, &self.state).await?;
        self.is_held = false;
        Ok(())
    }
}

impl Drop for DurableLeaseHandle {
    fn drop(&mut self) {
        if !self.is_held {
            return;
        }
        self.is_held = false;
        let object_store = self.object_store.clone();
        let path = self.path.clone();
        let state = self.state.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                if let Err(error) = Self::release_owned(&object_store, &path, &state).await {
                    log::warn!("Failed to release cancelled reference mutation: {}", error);
                }
            });
        }
    }
}

struct RefMutationLease {
    handle: DurableLeaseHandle,
}

impl RefMutationLease {
    async fn acquire(refs: &Refs) -> Result<Self> {
        let root_path = refs.root()?.path;
        discard_legacy_mutation_lock(&refs.object_store, &root_path).await?;
        let fence_path = base_ref_mutation_leases_path(&root_path);
        let owner = Uuid::new_v4().simple().to_string();
        let deadline = tokio::time::Instant::now() + REF_MUTATION_LOCK_TIMEOUT;

        loop {
            if tokio::time::Instant::now() >= deadline {
                return Err(Error::RefConflict {
                    message: format!(
                        "another reference mutation did not finish within {} seconds",
                        REF_MUTATION_LOCK_TIMEOUT.as_secs()
                    ),
                });
            }
            let latest_epoch = latest_lease_epoch(&refs.object_store, &fence_path).await?;
            if let Some(epoch) = latest_epoch {
                let latest_path = lease_epoch_path(&fence_path, epoch);
                if DurableLeaseHandle::is_active(&refs.object_store, &latest_path).await? {
                    tokio::time::sleep(REF_MUTATION_LOCK_RETRY_DELAY).await;
                    continue;
                }
            }

            let next_epoch = latest_epoch
                .unwrap_or(0)
                .checked_add(1)
                .ok_or_else(|| Error::internal("reference mutation lease epoch overflow"))?;
            let state = DurableLeaseState::acquired(owner.clone(), next_epoch)?;
            let path = lease_epoch_path(&fence_path, next_epoch);
            if create_lease_file(&refs.object_store, &path, &state).await? {
                let mut lease = Self {
                    handle: DurableLeaseHandle::new(
                        refs.object_store.clone(),
                        path,
                        state,
                        Some(fence_path.clone()),
                    ),
                };
                let fence = lease.handle.fence()?;
                if let Err(error) = fence.reconcile_prior_publications().await {
                    if let Err(release_error) = lease.release().await {
                        log::warn!(
                            "Failed to release reference mutation lease after reconciliation error: {}",
                            release_error
                        );
                    }
                    return Err(error);
                }
                let cleanup_object_store = refs.object_store.clone();
                let cleanup_path = fence_path.clone();
                // Complete local directory removal before returning the lease. On Windows, a
                // detached cleanup task can otherwise remove an epoch directory while the next
                // mutation is walking it, which makes the local object-store listing fail with
                // AccessDenied.
                cleanup_old_lease_epochs(&cleanup_object_store, &cleanup_path, next_epoch).await;
                return Ok(lease);
            }
        }
    }

    async fn release(&mut self) -> Result<()> {
        self.handle.release().await
    }
}

async fn create_lease_file(
    object_store: &ObjectStore,
    path: &Path,
    state: &DurableLeaseState,
) -> Result<bool> {
    match object_store
        .inner
        .put_opts(
            &path.clone().join(LEASE_FILE),
            state.serialize()?.into(),
            PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            },
        )
        .await
    {
        Ok(_) => Ok(true),
        Err(ObjectStoreError::AlreadyExists { .. } | ObjectStoreError::Precondition { .. }) => {
            Ok(false)
        }
        Err(error) => Err(error.into()),
    }
}

async fn create_serialized_file<T>(object_store: &ObjectStore, path: &Path, value: &T) -> Result<()>
where
    T: Serialize + DeserializeOwned + PartialEq,
{
    let body = serde_json::to_vec(value)?;
    match object_store
        .inner
        .put_opts(
            path,
            Bytes::from(body).into(),
            PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            },
        )
        .await
    {
        Ok(_) => Ok(()),
        Err(ObjectStoreError::AlreadyExists { .. } | ObjectStoreError::Precondition { .. }) => {
            if read_serialized_file(object_store, path).await?.as_ref() == Some(value) {
                Ok(())
            } else {
                Err(Error::RefConflict {
                    message: format!("coordination record already exists at {}", path),
                })
            }
        }
        Err(error) => Err(error.into()),
    }
}

async fn create_ref_catalog(
    object_store: &ObjectStore,
    path: &Path,
    catalog: &RefCatalog,
) -> Result<()> {
    let body = serde_json::to_vec_pretty(catalog)?;
    match object_store
        .inner
        .put_opts(
            path,
            Bytes::from(body).into(),
            PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            },
        )
        .await
    {
        Ok(_) => Ok(()),
        Err(ObjectStoreError::AlreadyExists { .. } | ObjectStoreError::Precondition { .. }) => {
            let current = object_store.inner.get(path).await?.bytes().await?;
            let current: RefCatalog = serde_json::from_slice(&current)?;
            if current.mutation_epoch == catalog.mutation_epoch
                && current.tags == catalog.tags
                && current.branches == catalog.branches
            {
                // Recovery may replay an intent after the first attempt committed with an earlier
                // publication-time legacy baseline. The immutable committed baseline wins.
                Ok(())
            } else {
                Err(Error::RefConflict {
                    message: format!("reference catalog already exists at {}", path),
                })
            }
        }
        Err(error) => Err(error.into()),
    }
}

async fn read_serialized_file<T>(object_store: &ObjectStore, path: &Path) -> Result<Option<T>>
where
    T: DeserializeOwned,
{
    let current = match object_store.inner.get(path).await {
        Ok(current) => current,
        Err(ObjectStoreError::NotFound { .. }) => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    Ok(Some(serde_json::from_slice(&current.bytes().await?)?))
}

async fn read_lease_state(
    object_store: &ObjectStore,
    path: &Path,
) -> Result<Option<DurableLeaseState>> {
    let current = match object_store.inner.get(path).await {
        Ok(current) => current,
        Err(ObjectStoreError::NotFound { .. }) => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    Ok(Some(serde_json::from_slice(&current.bytes().await?)?))
}

async fn latest_lease_epoch(object_store: &ObjectStore, path: &Path) -> Result<Option<u64>> {
    Ok(object_store
        .read_dir(path.clone())
        .await?
        .into_iter()
        .filter_map(|name| name.parse::<u64>().ok())
        .max())
}

fn lease_epoch_path(base_path: &Path, epoch: u64) -> Path {
    base_path.clone().join(format!("{epoch:020}"))
}

async fn cleanup_old_lease_epochs(
    object_store: &ObjectStore,
    base_path: &Path,
    current_epoch: u64,
) {
    let Ok(entries) = object_store.read_dir(base_path.clone()).await else {
        return;
    };
    for entry in entries {
        let Ok(epoch) = entry.parse::<u64>() else {
            continue;
        };
        if epoch >= current_epoch {
            continue;
        }
        let path = base_path.clone().join(entry);
        if let Err(error) = object_store.remove_dir_all(path).await
            && !matches!(error, Error::NotFound { .. })
        {
            log::warn!(
                "Failed to clean up an old reference mutation lease: {}",
                error
            );
        }
    }
}

async fn discard_legacy_mutation_lock(object_store: &ObjectStore, root_path: &Path) -> Result<()> {
    let path = ref_mutation_lock_path(root_path);
    if let Err(error) = object_store.delete(&path).await
        && !matches!(error, Error::NotFound { .. })
    {
        return Err(error);
    }
    Ok(())
}

/// Tags operation
#[derive(Debug, Clone)]
pub struct Tags<'a> {
    refs: &'a Refs,
}

/// Branches operation
#[derive(Debug, Clone)]
pub struct Branches<'a> {
    refs: &'a Refs,
}

impl Tags<'_> {
    fn object_store(&self) -> &ObjectStore {
        &self.refs.object_store
    }
}

impl Branches<'_> {
    fn object_store(&self) -> &ObjectStore {
        &self.refs.object_store
    }
}

impl Tags<'_> {
    pub async fn fetch_tags(&self) -> Result<Vec<(String, TagContents)>> {
        let root_location = self.refs.root()?;
        let catalog = read_ref_catalog(self.object_store(), &root_location.path).await?;
        catalog
            .tags
            .into_iter()
            .map(|(name, value)| {
                serde_json::from_value(value)
                    .map(|contents| (name, contents))
                    .map_err(Into::into)
            })
            .collect()
    }

    pub async fn list(&self) -> Result<HashMap<String, TagContents>> {
        self.fetch_tags()
            .await
            .map(|tags| tags.into_iter().collect())
    }

    pub async fn list_tags_ordered(
        &self,
        order: Option<Ordering>,
    ) -> Result<Vec<(String, TagContents)>> {
        let mut tags = self.fetch_tags().await?;
        tags.sort_by(|a, b| {
            let desired_ordering = order.unwrap_or(Ordering::Greater);
            let version_ordering = a.1.version.cmp(&b.1.version);
            let version_result = match desired_ordering {
                Ordering::Less => version_ordering,
                _ => version_ordering.reverse(),
            };
            version_result.then_with(|| a.0.cmp(&b.0))
        });
        Ok(tags)
    }

    pub async fn get_version(&self, tag: &str) -> Result<u64> {
        self.get(tag).await.map(|tag| tag.version)
    }

    pub async fn get(&self, tag: &str) -> Result<TagContents> {
        check_valid_tag(tag)?;

        let root_location = self.refs.root()?;
        let tag_file = tag_path(&root_location.path, tag);

        read_stored_ref(&tag_file, self.object_store())
            .await?
            .and_then(|stored| stored.contents)
            .ok_or_else(|| Error::RefNotFound {
                message: format!("tag {} does not exist", tag),
            })
    }

    pub async fn create(&self, tag: &str, reference: impl Into<Ref>) -> Result<()> {
        check_valid_tag(tag)?;
        let reference = reference.into();
        self.refs
            .run_tag_mutation(|fence| async move {
                let root_location = self.refs.root()?;
                let tag_file = tag_path(&root_location.path, tag);

                let stored = read_stored_ref::<TagContents>(&tag_file, self.object_store()).await?;
                if stored
                    .as_ref()
                    .is_some_and(|stored| stored.contents.is_some())
                {
                    return Err(Error::RefConflict {
                        message: format!("tag {} already exists", tag),
                    });
                }
                let now = utc_now();
                let tag_contents = self
                    .build_tag_content_by_ref(reference, Some(now), Some(now))
                    .await?;

                fence.publish(&tag_file, Some(&tag_contents)).await
            })
            .await
    }

    pub async fn delete(&self, tag: &str) -> Result<()> {
        check_valid_tag(tag)?;
        self.refs
            .run_tag_mutation(|fence| async move {
                let root_location = self.refs.root()?;
                let tag_file = tag_path(&root_location.path, tag);

                let stored = read_stored_ref::<TagContents>(&tag_file, self.object_store()).await?;
                if !stored
                    .as_ref()
                    .is_some_and(|stored| stored.contents.is_some())
                {
                    return Err(Error::RefNotFound {
                        message: format!("tag {} does not exist", tag),
                    });
                }

                fence.publish::<TagContents>(&tag_file, None).await
            })
            .await
    }

    pub async fn update(&self, tag: &str, reference: impl Into<Ref>) -> Result<()> {
        check_valid_tag(tag)?;
        let reference = reference.into();
        self.refs
            .run_tag_mutation(|fence| async move {
                let root_location = self.refs.root()?;
                let tag_file = tag_path(&root_location.path, tag);
                let stored = read_stored_ref::<TagContents>(&tag_file, self.object_store()).await?;
                let Some(stored) = stored.filter(|stored| stored.contents.is_some()) else {
                    return Err(Error::RefNotFound {
                        message: format!("tag {} does not exist", tag),
                    });
                };
                let mut tag_contents = stored.contents.ok_or_else(|| {
                    Error::internal("live tag reference lost its contents during update")
                })?;
                let updated_reference = self
                    .build_tag_content_by_ref(reference, tag_contents.created_at, Some(utc_now()))
                    .await?;
                tag_contents.branch = updated_reference.branch;
                tag_contents.version = updated_reference.version;
                tag_contents.created_at = updated_reference.created_at;
                tag_contents.updated_at = updated_reference.updated_at;
                tag_contents.manifest_size = updated_reference.manifest_size;

                fence.publish(&tag_file, Some(&tag_contents)).await
            })
            .await
    }

    pub async fn replace_metadata(
        &self,
        tag: &str,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        check_valid_tag(tag)?;
        self.refs
            .run_tag_mutation(|fence| async move {
                let root_location = self.refs.root()?;
                let tag_file = tag_path(&root_location.path, tag);
                let stored = read_stored_ref::<TagContents>(&tag_file, self.object_store()).await?;
                let Some(stored) = stored.filter(|stored| stored.contents.is_some()) else {
                    return Err(Error::RefNotFound {
                        message: format!("tag {} does not exist", tag),
                    });
                };
                let mut tag_contents = stored.contents.ok_or_else(|| {
                    Error::internal("live tag reference lost its contents during metadata update")
                })?;
                tag_contents.metadata = metadata;

                fence.publish(&tag_file, Some(&tag_contents)).await
            })
            .await
    }

    async fn build_tag_content_by_ref(
        &self,
        reference: impl Into<Ref>,
        created_at: Option<DateTime<Utc>>,
        updated_at: Option<DateTime<Utc>>,
    ) -> Result<TagContents> {
        let reference = reference.into();
        let (branch, version_number) = match reference {
            Version(branch, version_number) => (branch, version_number),
            VersionNumber(version_number) => {
                (self.refs.base_location.branch.clone(), Some(version_number))
            }
            Tag(tag_name) => {
                let tag_content = self.get(tag_name.as_str()).await?;
                (tag_content.branch, Some(tag_content.version))
            }
        };

        let branch_location = self
            .refs
            .branches()
            .resolve_location(branch.as_deref())
            .await?;
        let manifest_file = if let Some(version_number) = version_number {
            self.refs
                .commit_handler
                .resolve_version_location(
                    &branch_location.path,
                    version_number,
                    &self.refs.object_store.inner,
                )
                .await?
        } else {
            self.refs
                .commit_handler
                .resolve_latest_location(&branch_location.path, &self.refs.object_store)
                .await?
        };

        if !self.object_store().exists(&manifest_file.path).await? {
            return Err(Error::VersionNotFound {
                message: format!("version {} does not exist", Version(branch, version_number)),
            });
        }

        let manifest_size = if let Some(size) = manifest_file.size {
            size as usize
        } else {
            self.object_store().size(&manifest_file.path).await? as usize
        };

        let tag_contents = TagContents {
            branch,
            version: manifest_file.version,
            created_at,
            updated_at,
            manifest_size,
            metadata: HashMap::new(),
        };
        Ok(tag_contents)
    }
}

impl Branches<'_> {
    pub(crate) fn is_main_branch(branch: Option<&str>) -> bool {
        branch == Some(MAIN_BRANCH)
    }

    pub async fn fetch(&self) -> Result<Vec<(String, BranchContents)>> {
        let root_location = self.refs.root()?;
        let catalog = read_ref_catalog(self.object_store(), &root_location.path).await?;
        catalog
            .branches
            .into_iter()
            .map(|(name, value)| {
                let mut contents: BranchContents = serde_json::from_value(value)?;
                contents.hydrate_legacy_identifier(&name);
                Ok((name, contents))
            })
            .collect()
    }

    pub async fn list(&self) -> Result<HashMap<String, BranchContents>> {
        self.fetch()
            .await
            .map(|branches| branches.into_iter().collect())
    }

    pub async fn get(&self, branch: &str) -> Result<BranchContents> {
        check_valid_branch(branch)?;

        let root_location = self.refs.root()?;
        let branch_file = branch_contents_path(&root_location.path, branch);

        let mut contents = read_stored_ref::<BranchContents>(&branch_file, self.object_store())
            .await?
            .and_then(|stored| stored.contents)
            .ok_or_else(|| Error::RefNotFound {
                message: format!("branch {} does not exist", branch),
            })?;
        contents.hydrate_legacy_identifier(branch);
        Ok(contents)
    }

    pub async fn get_identifier(&self, branch: Option<&str>) -> Result<BranchIdentifier> {
        if let Some(branch_name) = branch {
            let branch_contents = self.get(branch_name).await?;
            Ok(branch_contents.identifier)
        } else {
            Ok(BranchIdentifier::main())
        }
    }

    pub(crate) async fn resolve_location(&self, branch: Option<&str>) -> Result<BranchLocation> {
        let Some(branch_name) = branch.and_then(standardize_branch) else {
            let location = self.refs.base_location.find_branch(None)?;
            self.refs
                .commit_handler
                .register_branch_path(&location.path, None);
            return Ok(location);
        };
        let location = match self.get(&branch_name).await {
            Ok(contents) => self.resolve_contents_location(&branch_name, &contents)?,
            // Metadata is the source of truth for branch CRUD, but readers retain the legacy
            // name-derived fallback for branch datasets that predate metadata or are being
            // recovered after an interrupted create.
            Err(Error::RefNotFound { .. }) => {
                self.refs.base_location.find_branch(Some(&branch_name))?
            }
            Err(error) => return Err(error),
        };
        self.refs
            .commit_handler
            .register_branch_path(&location.path, Some(&branch_name));
        Ok(location)
    }

    pub(crate) async fn resolve_path_location(
        &self,
        path_branch: &str,
    ) -> Result<Option<BranchLocation>> {
        // This is only a probe for a logical alias. An invalid branch name means the path suffix
        // belongs to the dataset root (for example, a main dataset rooted at `tree/main`).
        if check_valid_branch(path_branch).is_err() {
            return Ok(None);
        }
        match self.get(path_branch).await {
            Ok(_) => self.resolve_location(Some(path_branch)).await.map(Some),
            Err(Error::RefNotFound { .. }) => Ok(None),
            Err(error) => Err(error),
        }
    }

    fn resolve_contents_location(
        &self,
        branch_name: &str,
        contents: &BranchContents,
    ) -> Result<BranchLocation> {
        let Some(storage) = contents.storage.as_ref() else {
            return self.refs.base_location.find_branch(Some(branch_name));
        };
        Uuid::parse_str(&storage.generation).map_err(|error| Error::InvalidRef {
            message: format!(
                "Invalid branch storage generation '{}': {}",
                storage.generation, error
            ),
        })?;
        self.refs
            .base_location
            .find_branch_generation(branch_name, &storage.generation)
    }

    pub(crate) async fn prepare_create(
        &self,
        branch_name: &str,
        version_number: u64,
        source_branch: Option<&str>,
    ) -> Result<BranchContents> {
        check_valid_branch(branch_name)?;

        let source_branch = source_branch.and_then(standardize_branch);
        let root_location = self.refs.root()?;
        let branch_file = branch_contents_path(&root_location.path, branch_name);
        if read_stored_ref::<BranchContents>(&branch_file, self.object_store())
            .await?
            .is_some_and(|stored| stored.contents.is_some())
        {
            return Err(Error::RefConflict {
                message: format!("branch {} already exists", branch_name),
            });
        }

        let branch_location = self.resolve_location(source_branch.as_deref()).await?;
        // Verify the source version exists
        let manifest_file = self
            .refs
            .commit_handler
            .resolve_version_location(
                &branch_location.path,
                version_number,
                &self.refs.object_store.inner,
            )
            .await?;

        if !self.object_store().exists(&manifest_file.path).await? {
            return Err(Error::VersionNotFound {
                message: format!("Manifest file {} does not exist", manifest_file.path),
            });
        };

        let parent_branch_id = if let Some(ref parent_branch) = source_branch {
            self.get(parent_branch).await?.identifier
        } else {
            BranchIdentifier::main()
        };

        let identifier = BranchIdentifier::new(&parent_branch_id, version_number);
        let generation = identifier.storage_id().ok_or_else(|| {
            Error::internal(format!(
                "new branch '{}' is missing its physical storage generation",
                branch_name
            ))
        })?;
        Ok(BranchContents {
            parent_branch: source_branch,
            storage: Some(BranchStorage {
                layout: BranchStorageLayout::Detached,
                generation: generation.to_string(),
            }),
            identifier,
            parent_version: version_number,
            create_at: chrono::Utc::now().timestamp() as u64,
            manifest_size: if let Some(size) = manifest_file.size {
                size as usize
            } else {
                self.object_store().size(&manifest_file.path).await? as usize
            },
            metadata: HashMap::new(),
        })
    }

    // Only create branch metadata. The caller holds the reference mutation lease across the
    // physical clone and this metadata publication.
    pub(super) async fn create_unlocked(
        &self,
        fence: &DurableLeaseFence,
        branch_name: &str,
        branch_contents: BranchContents,
    ) -> Result<()> {
        let root_location = self.refs.root()?;
        let branch_file = branch_contents_path(&root_location.path, branch_name);
        let stored = read_stored_ref::<BranchContents>(&branch_file, self.object_store()).await?;
        if stored
            .as_ref()
            .is_some_and(|stored| stored.contents.is_some())
        {
            return Err(Error::RefConflict {
                message: format!("branch {} already exists", branch_name),
            });
        }

        fence.publish(&branch_file, Some(&branch_contents)).await
    }

    pub async fn replace_metadata(
        &self,
        branch: &str,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        check_valid_branch(branch)?;
        self.refs
            .run_mutation(|fence| async move {
                let root_location = self.refs.root()?;
                let branch_file = branch_contents_path(&root_location.path, branch);
                let stored =
                    read_stored_ref::<BranchContents>(&branch_file, self.object_store()).await?;
                let Some(stored) = stored.filter(|stored| stored.contents.is_some()) else {
                    return Err(Error::RefNotFound {
                        message: format!("branch {} does not exist", branch),
                    });
                };
                let mut branch_contents = stored.contents.ok_or_else(|| {
                    Error::internal(
                        "live branch reference lost its contents during metadata update",
                    )
                })?;
                branch_contents.hydrate_legacy_identifier(branch);
                branch_contents.metadata = metadata;

                fence.publish(&branch_file, Some(&branch_contents)).await
            })
            .await
    }

    #[cfg(test)]
    pub(crate) async fn publish_contents_for_test(
        &self,
        branch: &str,
        contents: Option<BranchContents>,
    ) -> Result<()> {
        let branch = branch.to_string();
        self.refs
            .run_mutation(|fence| async move {
                let root_location = self.refs.root()?;
                let branch_file = branch_contents_path(&root_location.path, &branch);
                fence.publish(&branch_file, contents.as_ref()).await
            })
            .await
    }

    /// Delete a branch
    ///
    /// If the `BranchContents` does not exist, it will return an error directly unless `force` is true.
    /// If `force` is true, it will try to delete the branch directories no matter `BranchContents` exists or not.
    pub async fn delete(&self, branch: &str, force: bool) -> Result<()> {
        check_valid_branch(branch)?;

        self.refs
            .run_branch_deletion(|fence| self.delete_unlocked(fence, branch, force))
            .await
    }

    async fn delete_unlocked(
        &self,
        fence: DurableLeaseFence,
        branch: &str,
        force: bool,
    ) -> Result<()> {
        let mut referencing_tags = self
            .refs
            .tags()
            .fetch_tags()
            .await?
            .into_iter()
            .filter_map(|(tag_name, contents)| {
                (contents.branch.as_deref() == Some(branch)).then_some(tag_name)
            })
            .collect_vec();
        referencing_tags.sort();
        if !referencing_tags.is_empty() {
            return Err(Error::RefConflict {
                message: format!(
                    "Branch {} is referenced by tags {:?} and cannot be deleted",
                    branch, referencing_tags
                ),
            });
        }

        let root_location = self.refs.root()?;
        let branch_file = branch_contents_path(&root_location.path, branch);
        let stored = read_stored_ref::<BranchContents>(&branch_file, self.object_store()).await?;
        let mut branch_contents = stored.as_ref().and_then(|stored| stored.contents.clone());
        if let Some(contents) = branch_contents.as_mut() {
            contents.hydrate_legacy_identifier(branch);
        }
        let all_branches = self.list().await?;
        if branch_contents.is_none() && !force {
            return Err(Error::RefNotFound {
                message: format!("Branch {} does not exist", branch),
            });
        } else if branch_contents.is_none() {
            log::warn!("BranchContents of {} does not exist", branch);
        }

        if let Some(contents) = branch_contents.as_ref()
            && contents.storage.is_none()
        {
            let referenced_versions = contents
                .identifier
                .collect_referenced_versions(&all_branches);
            if !referenced_versions.is_empty() {
                return Err(Error::RefConflict {
                    message: format!(
                        "Legacy branch {} is referenced by {:?} versions and cannot be deleted",
                        branch, referenced_versions
                    ),
                });
            }
        }

        if stored
            .as_ref()
            .is_some_and(|stored| stored.contents.is_some())
        {
            fence.publish::<BranchContents>(&branch_file, None).await?;
            let branch_contents = branch_contents.ok_or_else(|| {
                Error::internal("live branch reference lost its contents during deletion")
            })?;
            if let Err(error) = self
                .cleanup_committed_branch_storage(branch, branch_contents)
                .await
            {
                log::warn!(
                    "Failed to clean up storage after deleting branch '{}': {}",
                    branch,
                    error
                );
            }
            return Ok(());
        }

        self.cleanup_branch_directories(branch).await?;
        self.cleanup_generation_directories(branch).await
    }

    async fn cleanup_committed_branch_storage(
        &self,
        branch: &str,
        branch_contents: BranchContents,
    ) -> Result<()> {
        if branch_contents.storage.is_none() {
            return self.cleanup_branch_directories(branch).await;
        }

        // A deleted branch may still provide files to descendants. UUID-backed directories are
        // reclaimed only after their identifiers disappear from every remaining lineage.
        let remaining_branches = self.list().await?;
        let referenced_storage_ids = Self::collect_referenced_storage_ids(&remaining_branches);
        for storage_id in branch_contents
            .identifier
            .version_mapping
            .iter()
            .map(|(_, storage_id)| storage_id)
        {
            Uuid::parse_str(storage_id).map_err(|error| Error::InvalidRef {
                message: format!("Invalid branch storage id '{}': {}", storage_id, error),
            })?;
            if referenced_storage_ids.contains(storage_id.as_str()) {
                continue;
            }
            let location = self
                .refs
                .base_location
                .find_branch_generation(branch, storage_id)?;
            if let Err(error) = self.refs.object_store.remove_dir_all(location.path).await
                && !matches!(error, Error::NotFound { .. })
            {
                return Err(error);
            }
        }
        Ok(())
    }

    async fn cleanup_generation_directories(&self, branch: &str) -> Result<()> {
        let root_location = self.refs.root()?;
        let generations_path = root_location.path.clone().join(BRANCH_GENERATIONS_DIR);
        let branches = self.list().await?;
        let referenced_storage_ids = Self::collect_referenced_storage_ids(&branches);
        for storage_id in self.object_store().read_dir(generations_path).await? {
            if Uuid::parse_str(&storage_id).is_err() || referenced_storage_ids.contains(&storage_id)
            {
                continue;
            }
            let location = root_location.find_branch_generation(branch, &storage_id)?;
            self.refs
                .commit_handler
                .register_branch_path(&location.path, Some(branch));
            let manifest_location = match self
                .refs
                .commit_handler
                .resolve_latest_location(&location.path, self.object_store())
                .await
            {
                Ok(location) => location,
                Err(Error::NotFound { .. } | Error::DatasetNotFound { .. }) => continue,
                Err(error) => return Err(error),
            };
            let manifest = read_manifest(
                self.object_store(),
                &manifest_location.path,
                manifest_location.size,
            )
            .await?;
            if manifest.branch.as_deref() == Some(branch) {
                self.refs.object_store.remove_dir_all(location.path).await?;
            }
        }
        Ok(())
    }

    fn collect_referenced_storage_ids(
        branches: &HashMap<String, BranchContents>,
    ) -> HashSet<String> {
        branches
            .values()
            .flat_map(|contents| contents.identifier.version_mapping.iter())
            .map(|(_, storage_id)| storage_id.clone())
            .collect()
    }

    pub async fn list_ordered(
        &self,
        order: Option<Ordering>,
    ) -> Result<Vec<(String, BranchContents)>> {
        let mut branches = self.fetch().await?;
        branches.sort_by(|a, b| {
            let desired_ordering = order.unwrap_or(Ordering::Greater);
            let version_ordering = a.1.parent_version.cmp(&b.1.parent_version);
            let version_result = match desired_ordering {
                Ordering::Less => version_ordering,
                _ => version_ordering.reverse(),
            };
            version_result.then_with(|| a.0.cmp(&b.0))
        });
        Ok(branches)
    }

    /// Clean up empty parent directories
    async fn cleanup_branch_directories(&self, branch: &str) -> Result<()> {
        let branches = self.list().await?;
        let remaining_branches: Vec<&str> = branches.keys().map(|k| k.as_str()).collect();

        if let Some(delete_path) =
            Self::get_cleanup_path(branch, &remaining_branches, &self.refs.base_location)?
            && let Err(e) = self.refs.object_store.remove_dir_all(delete_path).await
        {
            match &e {
                Error::NotFound { .. } => {
                    log::debug!("Branch directory already deleted");
                }
                _ => return Err(e),
            }
        }
        Ok(())
    }

    fn get_cleanup_path(
        branch: &str,
        remaining_branches: &[&str],
        base_location: &BranchLocation,
    ) -> Result<Option<Path>> {
        let deleted_branch = BranchRelativePath::new(branch);
        let mut related_branches = Vec::new();
        let mut relative_dir = branch.to_string();
        for branch in remaining_branches {
            let branch = BranchRelativePath::new(branch);
            if branch.is_parent(&deleted_branch) || branch.is_child(&deleted_branch) {
                related_branches.push(branch);
            } else if let Some(common_prefix) = deleted_branch.find_common_prefix(&branch) {
                related_branches.push(common_prefix);
            }
        }

        related_branches.sort_by(|a, b| a.segments.len().cmp(&b.segments.len()).reverse());
        if let Some(branch) = related_branches.first() {
            if branch.is_child(&deleted_branch) || branch == &deleted_branch {
                // There are children of the deleted branch, we can't delete any directory for now
                // Example: deleted_branch = "a/b/c", remaining_branches = ["a/b/c/d"], we need to delete nothing
                return Ok(None);
            } else {
                // We pick the longest common directory between the deleted branch and the remaining branches
                // Then delete the first child of this common directory
                // Example: deleted_branch = "a/b/c", remaining_branches = ["a"], we need to delete "a/b"
                relative_dir = format!(
                    "{}/{}",
                    branch.segments.join("/"),
                    deleted_branch.segments[branch.segments.len()]
                );
            }
        } else if !deleted_branch.segments.is_empty() {
            // There are no common directories between the deleted branch and the remaining branches
            // We need to delete the entire directory
            // Example: deleted_branch = "a/b/c", remaining_branches = [], we need to delete "a"
            relative_dir = deleted_branch.segments[0].to_string();
        }

        let absolute_dir = base_location.find_branch(Some(relative_dir.as_str()))?;
        Ok(Some(absolute_dir.path))
    }
}

#[derive(Debug, PartialEq)]
struct BranchRelativePath<'a> {
    segments: Vec<&'a str>,
}

impl<'a> BranchRelativePath<'a> {
    fn new(branch_name: &'a str) -> Self {
        let segments = branch_name.split('/').collect_vec();
        Self { segments }
    }

    fn find_common_prefix(&self, other: &Self) -> Option<Self> {
        let mut common_segments = Vec::new();
        for (i, segment) in self.segments.iter().enumerate() {
            if i >= other.segments.len() || other.segments[i] != *segment {
                break;
            }
            common_segments.push(*segment);
        }
        if !common_segments.is_empty() {
            Some(BranchRelativePath {
                segments: common_segments,
            })
        } else {
            None
        }
    }

    fn is_parent(&self, other: &Self) -> bool {
        if other.segments.len() <= self.segments.len() {
            false
        } else {
            for (i, segment) in self.segments.iter().enumerate() {
                if other.segments[i] != *segment {
                    return false;
                }
            }
            true
        }
    }

    fn is_child(&self, other: &Self) -> bool {
        if other.segments.len() >= self.segments.len() {
            false
        } else {
            for (i, segment) in other.segments.iter().enumerate() {
                if self.segments[i] != *segment {
                    return false;
                }
            }
            true
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TagContents {
    pub branch: Option<String>,
    pub version: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
    pub manifest_size: usize,
    /// Metadata associated with this tag.
    ///
    /// Missing metadata is deserialized as an empty map.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchContents {
    pub parent_branch: Option<String>,
    /// Physical storage for this branch. Absence identifies the legacy name-backed layout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage: Option<BranchStorage>,
    #[serde(default = "BranchIdentifier::missing_identifier_sentinel")]
    pub identifier: BranchIdentifier,
    pub parent_version: u64,
    pub create_at: u64, // unix timestamp
    pub manifest_size: usize,
    /// Metadata associated with this branch.
    ///
    /// Missing metadata is deserialized as an empty map.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Explicit physical storage mapping for a branch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BranchStorage {
    /// The physical layout used by this branch generation.
    pub layout: BranchStorageLayout,
    /// Opaque UUID identifying the physical branch generation.
    pub generation: String,
}

/// Physical layout used for branch-specific files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BranchStorageLayout {
    /// Store files outside the logical alias namespace under `_branch_generations/<generation>`.
    Detached,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BranchIdentifier {
    pub version_mapping: Vec<(u64, String)>,
}

impl BranchIdentifier {
    pub fn new(parent: &Self, parent_version: u64) -> Self {
        let mut version_mapping = parent.version_mapping.clone();
        version_mapping.push((parent_version, Uuid::new_v4().simple().to_string()));
        Self { version_mapping }
    }

    /// Creates a sentinel identifier for legacy branch metadata that lacks an explicit
    /// identifier.
    ///
    /// `BranchContents::from_path` replaces this value with a deterministic synthetic
    /// identifier. Keeping this sentinel stable lets us distinguish missing identifiers from
    /// persisted identifiers without changing this field to `Option<BranchIdentifier>`.
    pub fn missing_identifier_sentinel() -> Self {
        Self {
            version_mapping: vec![(0, Uuid::nil().simple().to_string())],
        }
    }

    fn synthetic_identifier(
        branch_name: &str,
        parent_branch: Option<&str>,
        parent_version: u64,
        create_at: u64,
    ) -> Self {
        let identifier_input = format!(
            "branch_name={branch_name}\nparent_branch={}\nparent_version={parent_version}\ncreate_at={create_at}",
            parent_branch.unwrap_or("")
        );
        Self {
            version_mapping: vec![(
                0,
                Uuid::from_bytes(Self::synthetic_identifier_bytes(
                    identifier_input.as_bytes(),
                ))
                .simple()
                .to_string(),
            )],
        }
    }

    fn synthetic_identifier_bytes(input: &[u8]) -> [u8; 16] {
        // Use fixed, local hashing so legacy fallback identifiers stay deterministic without
        // enabling extra UUID generation features.
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        fn hash_with_seed(input: &[u8], seed: u64) -> u64 {
            input.iter().fold(seed, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
            })
        }

        let first = hash_with_seed(input, FNV_OFFSET);
        let second = hash_with_seed(input, FNV_OFFSET ^ 0x9e3779b97f4a7c15);
        let mut bytes = [0; 16];
        bytes[..8].copy_from_slice(&first.to_be_bytes());
        bytes[8..].copy_from_slice(&second.to_be_bytes());
        bytes
    }

    pub fn main() -> Self {
        Self {
            version_mapping: vec![],
        }
    }

    pub(crate) fn storage_id(&self) -> Option<&str> {
        self.version_mapping.last().map(|(_, uuid)| uuid.as_str())
    }

    pub fn parse(identifier: &str) -> Result<Self> {
        let parts: Vec<&str> = identifier.split(':').collect();
        if !parts.len().is_multiple_of(2) {
            return Err(Error::InvalidRef {
                message: format!(
                    "Invalid branch identifier: {}, format should be 'ver1:uuid1:ver2:uuid2:...:final_uuid'",
                    parts.len()
                ),
            });
        }

        let version_mapping = parts
            .chunks_exact(2)
            .map(|chunk| {
                let version = chunk[0].parse::<u64>().map_err(|e| Error::InvalidRef {
                    message: format!("Invalid version number '{}': {}", chunk[0], e),
                })?;
                let uuid = chunk[1].to_string();
                Ok((version, uuid))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { version_mapping })
    }

    pub fn find_referenced_version(&self, referenced_branch: &Self) -> Option<u64> {
        let ref_mapping = &referenced_branch.version_mapping;
        let next_idx = ref_mapping.len();

        (self.version_mapping.len() > next_idx && self.version_mapping[..next_idx] == *ref_mapping)
            .then(|| self.version_mapping[next_idx].0)
            .filter(|&version| version > 0)
    }

    /// Collects all branches that reference this branch, returning (branch_name, version) tuples.
    /// Results are in post-order traversal (deepest branches first).
    pub fn collect_referenced_versions(
        &self,
        branches: &HashMap<String, BranchContents>,
    ) -> Vec<(String, u64)> {
        let mut branch_ids = branches
            .iter()
            .map(|(name, branch)| (branch.identifier.clone(), name.clone()))
            .collect::<Vec<_>>();
        // Sort by BranchIdentifier desc to implement post-order traversal.
        branch_ids.sort_by(|a, b| b.cmp(a));
        branch_ids
            .into_iter()
            .filter_map(|(branch_id, name)| {
                branch_id
                    .find_referenced_version(self)
                    .map(|version| (name, version))
            })
            .collect()
    }
}

pub fn base_tags_path(base_path: &Path) -> Path {
    base_path.clone().join("_refs").join("tags")
}

pub fn base_branches_contents_path(base_path: &Path) -> Path {
    base_path.clone().join("_refs").join("branches")
}

fn ref_mutation_lock_path(base_path: &Path) -> Path {
    base_path.clone().join("_refs").join(REF_MUTATION_LOCK_FILE)
}

fn base_ref_mutation_leases_path(base_path: &Path) -> Path {
    base_path
        .clone()
        .join("_refs")
        .join(REF_MUTATION_LEASES_DIR)
}

fn base_tag_mutation_intents_path(base_path: &Path) -> Path {
    base_path
        .clone()
        .join("_refs")
        .join(TAG_MUTATION_INTENTS_DIR)
}

pub fn tag_path(base_path: &Path, branch: &str) -> Path {
    base_tags_path(base_path).join(format!("{}.json", branch))
}

// Note: child will encode '/' to '%2F'
pub fn branch_contents_path(base_path: &Path, branch: &str) -> Path {
    base_branches_contents_path(base_path).join(format!("{}.json", branch))
}

fn base_ref_catalog_path(base_path: &Path) -> Path {
    base_path.clone().join("_refs").join(REF_CATALOG_DIR)
}

fn ref_catalog_version_path(base_path: &Path, epoch: u64) -> Path {
    base_ref_catalog_path(base_path).join(format!("{epoch:020}.json"))
}

fn ref_catalog_root_from_version_path(path: &Path) -> Result<Path> {
    let catalog_path = path
        .parent()
        .ok_or_else(|| Error::internal(format!("reference catalog path {} has no parent", path)))?;
    if catalog_path.filename() != Some(REF_CATALOG_DIR) {
        return Err(Error::internal(format!(
            "reference catalog path {} is not below {}",
            path, REF_CATALOG_DIR
        )));
    }
    let refs_path = catalog_path.parent().ok_or_else(|| {
        Error::internal(format!(
            "reference catalog path {} is not below _refs",
            path
        ))
    })?;
    if refs_path.filename() != Some("_refs") {
        return Err(Error::internal(format!(
            "reference catalog path {} is not below _refs",
            path
        )));
    }
    refs_path.parent().ok_or_else(|| {
        Error::internal(format!(
            "reference catalog path {} has no dataset root",
            path
        ))
    })
}

#[derive(Clone, Copy)]
enum RefKind {
    Tag,
    Branch,
}

struct RefAddress {
    root: Path,
    kind: RefKind,
    name: String,
}

impl RefAddress {
    fn from_path(path: &Path) -> Result<Self> {
        let category_path = path
            .parent()
            .ok_or_else(|| Error::internal(format!("reference path {} has no parent", path)))?;
        let refs_path = category_path.parent().ok_or_else(|| {
            Error::internal(format!("reference path {} is not below _refs", path))
        })?;
        if refs_path.filename() != Some("_refs") {
            return Err(Error::internal(format!(
                "reference path {} is not below _refs",
                path
            )));
        }
        let root = refs_path.parent().ok_or_else(|| {
            Error::internal(format!("reference path {} has no dataset root", path))
        })?;
        let file_name = path
            .filename()
            .ok_or_else(|| Error::internal(format!("reference path {} has no file name", path)))?;
        let encoded_name = file_name.strip_suffix(".json").ok_or_else(|| {
            Error::internal(format!("reference path {} is not a JSON file", path))
        })?;
        let (kind, name) = match category_path.filename() {
            Some("tags") => (RefKind::Tag, encoded_name.to_string()),
            Some("branches") => {
                let name = Path::from_url_path(encoded_name)
                    .map_err(|error| Error::InvalidRef {
                        message: format!(
                            "Failed to decode branch name: {} due to exception {}",
                            encoded_name, error
                        ),
                    })?
                    .to_string();
                (RefKind::Branch, name)
            }
            _ => {
                return Err(Error::internal(format!(
                    "reference path {} has an unknown category",
                    path
                )));
            }
        };
        Ok(Self { root, kind, name })
    }
}

async fn read_latest_ref_catalog(
    object_store: &ObjectStore,
    root: &Path,
) -> Result<Option<RefCatalog>> {
    let catalog_path = base_ref_catalog_path(root);
    loop {
        let entries = match object_store.read_dir(catalog_path.clone()).await {
            Ok(entries) => entries,
            Err(error) => return Err(error),
        };
        let Some((epoch, file_name)) = entries
            .into_iter()
            .filter_map(|file_name| {
                file_name
                    .strip_suffix(".json")
                    .and_then(|epoch| epoch.parse::<u64>().ok())
                    .map(|epoch| (epoch, file_name))
            })
            .max_by_key(|(epoch, _)| *epoch)
        else {
            return Ok(None);
        };
        let path = catalog_path.clone().join(file_name);
        let result = match object_store.inner.get(&path).await {
            Ok(result) => result,
            Err(ObjectStoreError::NotFound { .. }) => {
                tokio::task::yield_now().await;
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let catalog: RefCatalog = serde_json::from_slice(&result.bytes().await?)?;
        if catalog.mutation_epoch != epoch {
            return Err(Error::corrupt_file(
                path,
                format!(
                    "reference catalog epoch {} does not match path epoch {}",
                    catalog.mutation_epoch, epoch
                ),
            ));
        }
        return Ok(Some(catalog));
    }
}

async fn read_legacy_ref_entries(
    object_store: &ObjectStore,
    base_path: &Path,
    is_branch: bool,
) -> Result<HashMap<String, serde_json::Value>> {
    let mut entries = HashMap::new();
    for file_name in object_store.read_dir(base_path.clone()).await? {
        let Some(encoded_name) = file_name.strip_suffix(".json") else {
            continue;
        };
        let encoded_name = encoded_name.to_string();
        let path = base_path.clone().join(file_name);
        let Some(value) = read_legacy_ref_entry(object_store, &path).await? else {
            continue;
        };
        let name = if is_branch {
            Path::from_url_path(&encoded_name)
                .map_err(|error| Error::InvalidRef {
                    message: format!(
                        "Failed to decode branch name: {} due to exception {}",
                        encoded_name, error
                    ),
                })?
                .to_string()
        } else {
            encoded_name
        };
        entries.insert(name, value);
    }
    Ok(entries)
}

async fn read_legacy_ref_entry(
    object_store: &ObjectStore,
    path: &Path,
) -> Result<Option<serde_json::Value>> {
    let result = match object_store.inner.get(path).await {
        Ok(result) => result,
        Err(ObjectStoreError::NotFound { .. }) => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let value: serde_json::Value = serde_json::from_slice(&result.bytes().await?)?;
    if value
        .get("_deleted")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        Ok(None)
    } else {
        Ok(Some(value))
    }
}

async fn read_legacy_ref_state(object_store: &ObjectStore, root: &Path) -> Result<LegacyRefState> {
    Ok(LegacyRefState {
        tags: read_legacy_ref_entries(object_store, &base_tags_path(root), false).await?,
        branches: read_legacy_ref_entries(object_store, &base_branches_contents_path(root), true)
            .await?,
    })
}

async fn read_ref_catalog(object_store: &ObjectStore, root: &Path) -> Result<RefCatalog> {
    let latest = read_latest_ref_catalog(object_store, root).await?;
    let legacy = read_legacy_ref_state(object_store, root).await?;
    if let Some(mut catalog) = latest {
        catalog.reconcile_legacy_changes(legacy);
        return Ok(catalog);
    }

    // A catalog may be published while the legacy snapshot is being read. Prefer that complete
    // snapshot and reconcile changes observed at the old paths instead of returning a partial
    // legacy view.
    if let Some(mut catalog) = read_latest_ref_catalog(object_store, root).await? {
        catalog.reconcile_legacy_changes(legacy);
        return Ok(catalog);
    }

    Ok(RefCatalog {
        mutation_epoch: 0,
        tags: legacy.tags.clone(),
        branches: legacy.branches.clone(),
        legacy_baseline: legacy,
    })
}

async fn compact_ref_catalog(
    object_store: &ObjectStore,
    root: &Path,
    current_epoch: u64,
) -> Result<()> {
    let catalog_path = base_ref_catalog_path(root);
    let mut prior_epochs = object_store
        .read_dir(catalog_path.clone())
        .await?
        .into_iter()
        .filter_map(|file_name| {
            file_name
                .strip_suffix(".json")
                .and_then(|epoch| epoch.parse::<u64>().ok())
                .filter(|epoch| *epoch < current_epoch)
                .map(|epoch| (epoch, file_name))
        })
        .collect_vec();
    prior_epochs.sort_unstable_by_key(|(epoch, _)| *epoch);
    // Keep the immediate predecessor so a reader that listed it before this publication can
    // still fetch it. Raced readers also restart until a listed catalog is fetched.
    prior_epochs.pop();
    for (_, file_name) in prior_epochs {
        let path = catalog_path.clone().join(file_name);
        if let Err(error) = object_store.delete(&path).await
            && !matches!(error, Error::NotFound { .. })
        {
            return Err(error);
        }
    }
    Ok(())
}

pub(crate) fn normalize_branch(branch: Option<&str>) -> String {
    match branch {
        None => MAIN_BRANCH.to_string(),
        Some(name) => name.to_string(),
    }
}

pub(crate) fn standardize_branch(branch: &str) -> Option<String> {
    match branch {
        MAIN_BRANCH => None,
        name => Some(name.to_string()),
    }
}

struct StoredRef<T> {
    contents: Option<T>,
}

async fn read_stored_ref<T>(path: &Path, object_store: &ObjectStore) -> Result<Option<StoredRef<T>>>
where
    T: DeserializeOwned,
{
    let address = RefAddress::from_path(path)?;
    let latest = read_latest_ref_catalog(object_store, &address.root).await?;
    let legacy = read_legacy_ref_entry(object_store, path).await?;
    let catalog = match latest {
        Some(catalog) => Some(catalog),
        None => {
            // A catalog may be published while the flat entry is being read. Recheck once so
            // migration cannot make a point lookup return an older partial view.
            read_latest_ref_catalog(object_store, &address.root).await?
        }
    };
    let value = if let Some(catalog) = catalog {
        let (current, baseline) = match address.kind {
            RefKind::Tag => (
                catalog.tags.get(&address.name),
                catalog.legacy_baseline.tags.get(&address.name),
            ),
            RefKind::Branch => (
                catalog.branches.get(&address.name),
                catalog.legacy_baseline.branches.get(&address.name),
            ),
        };
        if baseline != legacy.as_ref() {
            legacy
        } else {
            current.cloned()
        }
    } else {
        legacy
    };
    value
        .map(serde_json::from_value)
        .transpose()
        .map(|contents| {
            contents.map(|contents| StoredRef {
                contents: Some(contents),
            })
        })
        .map_err(Into::into)
}

impl TagContents {
    pub async fn from_path(path: &Path, object_store: &ObjectStore) -> Result<Self> {
        read_stored_ref(path, object_store)
            .await?
            .and_then(|stored| stored.contents)
            .ok_or_else(|| Error::RefNotFound {
                message: format!("tag metadata does not exist at {}", path),
            })
    }
}

impl BranchContents {
    fn hydrate_legacy_identifier(&mut self, branch_name: &str) {
        if self.identifier == BranchIdentifier::missing_identifier_sentinel() {
            self.identifier = BranchIdentifier::synthetic_identifier(
                branch_name,
                self.parent_branch.as_deref(),
                self.parent_version,
                self.create_at,
            );
        }
    }

    pub async fn from_path(
        path: &Path,
        object_store: &ObjectStore,
        branch_name: &str,
    ) -> Result<Self> {
        let mut contents: Self = read_stored_ref(path, object_store)
            .await?
            .and_then(|stored| stored.contents)
            .ok_or_else(|| Error::RefNotFound {
                message: format!("branch {} does not exist", branch_name),
            })?;
        // Legacy branch files do not store an identifier. Derive a deterministic fallback from
        // stable branch metadata so repeated reads expose the same public branch_identifier.
        contents.hydrate_legacy_identifier(branch_name);
        Ok(contents)
    }
}

pub fn check_valid_branch(branch_name: &str) -> Result<()> {
    if branch_name.is_empty() {
        return Err(Error::InvalidRef {
            message: "Branch name cannot be empty".to_string(),
        });
    }

    // Validate if the branch name starts or ends with a '/'
    if branch_name.starts_with('/') || branch_name.ends_with('/') {
        return Err(Error::InvalidRef {
            message: "Branch name cannot start or end with a '/'".to_string(),
        });
    }

    // Validate if there are any consecutive '/' in the branch name
    if branch_name.contains("//") {
        return Err(Error::InvalidRef {
            message: "Branch name cannot contain consecutive '/'".to_string(),
        });
    }

    // Validate if there are any dangerous characters in the branch name
    if branch_name.contains("..") || branch_name.contains('\\') {
        return Err(Error::InvalidRef {
            message: "Branch name cannot contain '..' or '\\'".to_string(),
        });
    }

    for segment in branch_name.split('/') {
        if segment.is_empty() {
            return Err(Error::InvalidRef {
                message: "Branch name cannot have empty segments between '/'".to_string(),
            });
        }
        if !segment
            .chars()
            .all(|c| c.is_alphanumeric() || c == '.' || c == '-' || c == '_')
        {
            return Err(Error::InvalidRef {
                message: format!(
                    "Branch segment '{}' contains invalid characters. Only alphanumeric, '.', '-', '_' are allowed.",
                    segment
                ),
            });
        }
    }

    if branch_name.ends_with(".lock") {
        return Err(Error::InvalidRef {
            message: "Branch name cannot end with '.lock'".to_string(),
        });
    }

    if branch_name.eq("main") {
        return Err(Error::InvalidRef {
            message: "Branch name cannot be 'main'".to_string(),
        });
    }
    Ok(())
}

pub fn check_valid_tag(s: &str) -> Result<()> {
    if s.is_empty() {
        return Err(Error::InvalidRef {
            message: "Ref cannot be empty".to_string(),
        });
    }

    if !s
        .chars()
        .all(|c| c.is_alphanumeric() || c == '.' || c == '-' || c == '_')
    {
        return Err(Error::InvalidRef {
            message: "Ref characters must be either alphanumeric, '.', '-' or '_'".to_string(),
        });
    }

    if s.starts_with('.') {
        return Err(Error::InvalidRef {
            message: "Ref cannot begin with a dot".to_string(),
        });
    }

    if s.ends_with('.') {
        return Err(Error::InvalidRef {
            message: "Ref cannot end with a dot".to_string(),
        });
    }

    if s.ends_with(".lock") {
        return Err(Error::InvalidRef {
            message: "Ref cannot end with .lock".to_string(),
        });
    }

    if s.contains("..") {
        return Err(Error::InvalidRef {
            message: "Ref cannot have two consecutive dots".to_string(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::test::FailingProxyStore;
    use datafusion::common::assert_contains;
    use futures::stream::BoxStream;
    use lance_io::object_store::WrappingObjectStore;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
        PutMultipartOptions, PutPayload, PutResult, RenameOptions, Result as ObjectStoreResult,
    };

    use rstest::rstest;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};

    #[derive(Debug)]
    struct RefTestStore {
        target: Arc<dyn object_store::ObjectStore>,
        heartbeat_started: Arc<tokio::sync::Notify>,
        release_heartbeat: Arc<tokio::sync::Notify>,
        remaining_catalog_get_failures: AtomicUsize,
        get_count: Arc<AtomicUsize>,
    }

    impl fmt::Display for RefTestStore {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
            write!(formatter, "RefTestStore({})", self.target)
        }
    }

    #[async_trait::async_trait]
    #[deny(clippy::missing_trait_methods)]
    #[cfg_attr(coverage, coverage(off))]
    impl object_store::ObjectStore for RefTestStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            options: PutOptions,
        ) -> ObjectStoreResult<PutResult> {
            if location.as_ref().contains("/heartbeats/") {
                self.heartbeat_started.notify_one();
                self.release_heartbeat.notified().await;
            }
            self.target.put_opts(location, payload, options).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            options: PutMultipartOptions,
        ) -> ObjectStoreResult<Box<dyn MultipartUpload>> {
            self.target.put_multipart_opts(location, options).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> ObjectStoreResult<GetResult> {
            self.get_count.fetch_add(1, AtomicOrdering::SeqCst);
            if location.as_ref().contains("/_refs/catalog/") && {
                let mut remaining = self
                    .remaining_catalog_get_failures
                    .load(AtomicOrdering::SeqCst);
                loop {
                    let Some(next) = remaining.checked_sub(1) else {
                        break false;
                    };
                    match self.remaining_catalog_get_failures.compare_exchange_weak(
                        remaining,
                        next,
                        AtomicOrdering::SeqCst,
                        AtomicOrdering::SeqCst,
                    ) {
                        Ok(_) => break true,
                        Err(actual) => remaining = actual,
                    }
                }
            } {
                return Err(ObjectStoreError::NotFound {
                    path: location.to_string(),
                    source: "catalog was compacted before GET".into(),
                });
            }
            self.target.get_opts(location, options).await
        }

        async fn get_ranges(
            &self,
            location: &Path,
            ranges: &[std::ops::Range<u64>],
        ) -> ObjectStoreResult<Vec<Bytes>> {
            self.target.get_ranges(location, ranges).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, ObjectStoreResult<Path>>,
        ) -> BoxStream<'static, ObjectStoreResult<Path>> {
            self.target.delete_stream(locations)
        }

        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, ObjectStoreResult<ObjectMeta>> {
            self.target.list(prefix)
        }

        fn list_with_offset(
            &self,
            prefix: Option<&Path>,
            offset: &Path,
        ) -> BoxStream<'static, ObjectStoreResult<ObjectMeta>> {
            self.target.list_with_offset(prefix, offset)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> ObjectStoreResult<ListResult> {
            self.target.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: CopyOptions,
        ) -> ObjectStoreResult<()> {
            self.target.copy_opts(from, to, options).await
        }

        async fn rename_opts(
            &self,
            from: &Path,
            to: &Path,
            options: RenameOptions,
        ) -> ObjectStoreResult<()> {
            self.target.rename_opts(from, to, options).await
        }
    }

    #[rstest]
    fn test_ok_ref(
        #[values(
            "ref",
            "ref-with-dashes",
            "ref.extension",
            "ref_with_underscores",
            "v1.2.3-rc4"
        )]
        r: &str,
    ) {
        check_valid_tag(r).unwrap();
    }

    #[rstest]
    fn test_err_ref(
        #[values(
            "",
            "../ref",
            ".ref",
            "/ref",
            "@",
            "deeply/nested/ref",
            "nested//ref",
            "nested/ref",
            "nested\\ref",
            "ref*",
            "ref.lock",
            "ref/",
            "ref?",
            "ref@{ref",
            "ref[",
            "ref^",
            "~/ref",
            "ref.",
            "ref..ref"
        )]
        r: &str,
    ) {
        assert_contains!(
            check_valid_tag(r).err().unwrap().to_string(),
            "Ref is invalid: Ref"
        );
    }

    #[rstest]
    fn test_valid_branch_names(
        #[values(
            "feature/login",
            "bugfix/issue-123",
            "release/v1.2.3",
            "user/someone/my-feature",
            "normal",
            "with-dash",
            "with_underscore",
            "with.dot"
        )]
        branch_name: &str,
    ) {
        assert!(
            check_valid_branch(branch_name).is_ok(),
            "Branch name '{}' should be valid",
            branch_name
        );
    }

    #[rstest]
    fn test_invalid_branch_names(
        #[values(
            "",
            "/start-with-slash",
            "end-with-slash/",
            "have//consecutive-slash",
            "have..dot-dot",
            "have\\backslash",
            "segment/",
            "/segment",
            "segment//empty",
            "name.lock",
            "bad@character",
            "bad segment"
        )]
        branch_name: &str,
    ) {
        assert!(
            check_valid_branch(branch_name).is_err(),
            "Branch name '{}' should be invalid",
            branch_name
        );
    }

    #[test]
    fn test_path_functions() {
        let base_path = Path::from("dataset");

        // Test base_tags_path
        let tags_path = base_tags_path(&base_path);
        assert_eq!(tags_path, Path::from("dataset/_refs/tags"));

        // Test base_branches_path
        let branches_path = base_branches_contents_path(&base_path);
        assert_eq!(branches_path, Path::from("dataset/_refs/branches"));

        // Test tag_path
        let tag_file_path = tag_path(&base_path, "v1.0.0");
        assert_eq!(tag_file_path, Path::from("dataset/_refs/tags/v1.0.0.json"));

        // Test branch_path
        let branch_file_path = branch_contents_path(&base_path, "feature");
        assert_eq!(
            branch_file_path,
            Path::from("dataset/_refs/branches/feature.json")
        );
    }

    #[tokio::test]
    async fn test_delayed_heartbeat_cannot_revive_fenced_epoch() {
        let heartbeat_started = Arc::new(tokio::sync::Notify::new());
        let release_heartbeat = Arc::new(tokio::sync::Notify::new());
        let target: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());
        let mut object_store = ObjectStore::memory();
        object_store.inner = Arc::new(RefTestStore {
            target,
            heartbeat_started: heartbeat_started.clone(),
            release_heartbeat: release_heartbeat.clone(),
            remaining_catalog_get_failures: AtomicUsize::new(0),
            get_count: Arc::new(AtomicUsize::new(0)),
        });
        let object_store = Arc::new(object_store);
        let fence_path = Path::from("dataset/_refs/mutation_leases");

        let first_state = DurableLeaseState::acquired("first".to_string(), 1).unwrap();
        let first_path = lease_epoch_path(&fence_path, 1);
        assert!(
            create_lease_file(&object_store, &first_path, &first_state)
                .await
                .unwrap()
        );
        let mut first_handle = DurableLeaseHandle::new(
            object_store.clone(),
            first_path,
            first_state,
            Some(fence_path.clone()),
        );
        let renew_task = tokio::spawn(async move { first_handle.renew().await });
        heartbeat_started.notified().await;

        let second_state = DurableLeaseState::acquired("second".to_string(), 2).unwrap();
        assert!(
            create_lease_file(
                &object_store,
                &lease_epoch_path(&fence_path, 2),
                &second_state,
            )
            .await
            .unwrap()
        );
        release_heartbeat.notify_one();

        let error = renew_task.await.unwrap().unwrap_err();
        assert!(matches!(error, Error::RefConflict { .. }));
        assert!(error.to_string().contains("lost its fence"));
    }

    #[tokio::test]
    async fn test_expired_lease_does_not_start_mutation() {
        let object_store = Arc::new(ObjectStore::memory());
        let fence_path = Path::from("dataset/_refs/mutation_leases");
        let state = DurableLeaseState {
            owner: "expired".to_string(),
            epoch: 1,
            expires_at_millis: 0,
        };
        let mut handle = DurableLeaseHandle::new(
            object_store,
            lease_epoch_path(&fence_path, 1),
            state,
            Some(fence_path),
        );
        let was_polled = Arc::new(AtomicBool::new(false));
        let mutation_was_polled = was_polled.clone();
        let mutation = std::future::poll_fn(move |_| {
            mutation_was_polled.store(true, AtomicOrdering::SeqCst);
            std::task::Poll::Ready(Ok::<_, Error>(()))
        });

        let error = Refs::drive_with_lease(&mut handle, mutation)
            .await
            .unwrap_err();
        handle.is_held = false;

        assert!(matches!(error, Error::RefConflict { .. }));
        assert!(error.to_string().contains("expired before renewal"));
        assert!(!was_polled.load(AtomicOrdering::SeqCst));
    }

    #[tokio::test]
    async fn test_reference_publication_ignores_delayed_older_epoch() {
        let object_store = ObjectStore::memory();
        let root = Path::from("dataset");
        let path = tag_path(&root, "release");
        let older_contents = TagContents {
            branch: None,
            version: 1,
            created_at: None,
            updated_at: None,
            manifest_size: 1,
            metadata: HashMap::new(),
        };
        let newer_contents = TagContents {
            version: 2,
            ..older_contents.clone()
        };
        let older = DurableRefPublication {
            epoch: 1,
            path: ref_catalog_version_path(&root, 1).to_string(),
            body: serde_json::to_string_pretty(&RefCatalog {
                mutation_epoch: 1,
                tags: HashMap::from([(
                    "release".to_string(),
                    serde_json::to_value(&older_contents).unwrap(),
                )]),
                branches: HashMap::new(),
                legacy_baseline: LegacyRefState::default(),
            })
            .unwrap(),
        };
        let newer = DurableRefPublication {
            epoch: 2,
            path: ref_catalog_version_path(&root, 2).to_string(),
            body: serde_json::to_string_pretty(&RefCatalog {
                mutation_epoch: 2,
                tags: HashMap::from([(
                    "release".to_string(),
                    serde_json::to_value(&newer_contents).unwrap(),
                )]),
                branches: HashMap::new(),
                legacy_baseline: LegacyRefState::default(),
            })
            .unwrap(),
        };
        let deleted = DurableRefPublication {
            epoch: 3,
            path: ref_catalog_version_path(&root, 3).to_string(),
            body: serde_json::to_string_pretty(&RefCatalog {
                mutation_epoch: 3,
                tags: HashMap::new(),
                branches: HashMap::new(),
                legacy_baseline: LegacyRefState::default(),
            })
            .unwrap(),
        };

        older.apply(&object_store).await.unwrap();
        newer.apply(&object_store).await.unwrap();
        older.apply(&object_store).await.unwrap();
        assert_eq!(
            read_stored_ref::<TagContents>(&path, &object_store)
                .await
                .unwrap()
                .unwrap()
                .contents
                .unwrap()
                .version,
            2
        );

        deleted.apply(&object_store).await.unwrap();
        newer.apply(&object_store).await.unwrap();
        assert!(
            read_stored_ref::<TagContents>(&path, &object_store)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_successor_reconciles_prior_reference_publication() {
        let object_store = Arc::new(ObjectStore::memory());
        let fence_path = Path::from("dataset/_refs/mutation_leases");
        let tag_file = tag_path(&Path::from("dataset"), "pending");
        let tag_contents = TagContents {
            branch: None,
            version: 7,
            created_at: None,
            updated_at: None,
            manifest_size: 1,
            metadata: HashMap::new(),
        };

        let first_path = lease_epoch_path(&fence_path, 1);
        let first_state = DurableLeaseState {
            owner: "expired".to_string(),
            epoch: 1,
            expires_at_millis: 0,
        };
        assert!(
            create_lease_file(&object_store, &first_path, &first_state)
                .await
                .unwrap()
        );
        let publication = DurableRefPublication {
            epoch: 1,
            path: ref_catalog_version_path(&Path::from("dataset"), 1).to_string(),
            body: serde_json::to_string_pretty(&RefCatalog {
                mutation_epoch: 1,
                tags: HashMap::from([(
                    "pending".to_string(),
                    serde_json::to_value(&tag_contents).unwrap(),
                )]),
                branches: HashMap::new(),
                legacy_baseline: LegacyRefState::default(),
            })
            .unwrap(),
        };
        create_serialized_file(
            &object_store,
            &first_path.clone().join(LEASE_PUBLICATION_FILE),
            &publication,
        )
        .await
        .unwrap();
        let first_fence = DurableLeaseFence {
            object_store: object_store.clone(),
            path: first_path,
            fence_path: fence_path.clone(),
            epoch: 1,
        };

        let second_path = lease_epoch_path(&fence_path, 2);
        let second_state = DurableLeaseState::acquired("successor".to_string(), 2).unwrap();
        assert!(
            create_lease_file(&object_store, &second_path, &second_state)
                .await
                .unwrap()
        );
        DurableLeaseFence {
            object_store: object_store.clone(),
            path: second_path,
            fence_path: fence_path.clone(),
            epoch: 2,
        }
        .reconcile_prior_publications()
        .await
        .unwrap();

        first_fence
            .apply_committed_publication(&publication)
            .await
            .expect("a successor-committed publication must not report failure");

        assert_eq!(
            read_stored_ref::<TagContents>(&tag_file, &object_store)
                .await
                .unwrap()
                .unwrap()
                .contents
                .unwrap()
                .version,
            7
        );
    }

    #[tokio::test]
    async fn test_partial_publication_is_discoverable_by_list() {
        let object_store = ObjectStore::memory();
        let root = Path::from("dataset");
        let contents = TagContents {
            branch: None,
            version: 7,
            created_at: None,
            updated_at: None,
            manifest_size: 1,
            metadata: HashMap::new(),
        };
        let publication = DurableRefPublication {
            epoch: 1,
            path: ref_catalog_version_path(&root, 1).to_string(),
            body: serde_json::to_string_pretty(&RefCatalog {
                mutation_epoch: 1,
                tags: HashMap::from([(
                    "pending".to_string(),
                    serde_json::to_value(&contents).unwrap(),
                )]),
                branches: HashMap::new(),
                legacy_baseline: LegacyRefState::default(),
            })
            .unwrap(),
        };

        publication.apply(&object_store).await.unwrap();

        let catalog = read_ref_catalog(&object_store, &root).await.unwrap();
        assert!(catalog.tags.contains_key("pending"));
        assert_eq!(
            read_stored_ref::<TagContents>(&tag_path(&root, "pending"), &object_store)
                .await
                .unwrap()
                .unwrap()
                .contents
                .unwrap()
                .version,
            7
        );
    }

    #[tokio::test]
    async fn test_catalog_migration_preserves_legacy_baseline() {
        let object_store = ObjectStore::memory();
        let root = Path::from("dataset");
        let legacy_path = tag_path(&root, "legacy");
        let contents = TagContents {
            branch: None,
            version: 1,
            created_at: None,
            updated_at: None,
            manifest_size: 1,
            metadata: HashMap::new(),
        };
        object_store
            .put(
                &legacy_path,
                serde_json::to_vec(&contents).unwrap().as_slice(),
            )
            .await
            .unwrap();
        let publication = DurableRefPublication {
            epoch: 1,
            path: ref_catalog_version_path(&root, 1).to_string(),
            body: serde_json::to_string_pretty(&RefCatalog {
                mutation_epoch: 1,
                tags: HashMap::from([(
                    "legacy".to_string(),
                    serde_json::to_value(&contents).unwrap(),
                )]),
                branches: HashMap::new(),
                legacy_baseline: LegacyRefState {
                    tags: HashMap::from([(
                        "legacy".to_string(),
                        serde_json::to_value(&contents).unwrap(),
                    )]),
                    branches: HashMap::new(),
                },
            })
            .unwrap(),
        };

        publication.apply(&object_store).await.unwrap();

        assert!(object_store.exists(&legacy_path).await.unwrap());
        assert_eq!(
            read_stored_ref::<TagContents>(&legacy_path, &object_store)
                .await
                .unwrap()
                .unwrap()
                .contents
                .unwrap()
                .version,
            1
        );
    }

    #[tokio::test]
    async fn test_point_read_reconciles_only_addressed_legacy_ref() {
        let target: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());
        let mut writer = ObjectStore::memory();
        writer.inner = target.clone();
        let root = Path::from("dataset");
        for version in 1..=100_u64 {
            let contents = TagContents {
                branch: None,
                version,
                created_at: None,
                updated_at: None,
                manifest_size: 1,
                metadata: HashMap::new(),
            };
            writer
                .put(
                    &tag_path(&root, &format!("legacy-{version}")),
                    serde_json::to_vec(&contents).unwrap().as_slice(),
                )
                .await
                .unwrap();
        }

        let mut catalog = read_ref_catalog(&writer, &root).await.unwrap();
        catalog.mutation_epoch = 1;
        DurableRefPublication {
            epoch: 1,
            path: ref_catalog_version_path(&root, 1).to_string(),
            body: serde_json::to_string_pretty(&catalog).unwrap(),
        }
        .apply(&writer)
        .await
        .unwrap();

        let get_count = Arc::new(AtomicUsize::new(0));
        let mut reader = ObjectStore::memory();
        reader.inner = Arc::new(RefTestStore {
            target,
            heartbeat_started: Arc::new(tokio::sync::Notify::new()),
            release_heartbeat: Arc::new(tokio::sync::Notify::new()),
            remaining_catalog_get_failures: AtomicUsize::new(0),
            get_count: get_count.clone(),
        });

        assert_eq!(
            read_stored_ref::<TagContents>(&tag_path(&root, "legacy-100"), &reader)
                .await
                .unwrap()
                .unwrap()
                .contents
                .unwrap()
                .version,
            100
        );
        assert_eq!(get_count.load(AtomicOrdering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_legacy_writer_is_not_ignored_after_catalog_migration() {
        let object_store = ObjectStore::memory();
        let root = Path::from("dataset");
        let release_path = tag_path(&root, "release");
        let contents = TagContents {
            branch: None,
            version: 1,
            created_at: None,
            updated_at: None,
            manifest_size: 1,
            metadata: HashMap::new(),
        };
        DurableRefPublication {
            epoch: 1,
            path: ref_catalog_version_path(&root, 1).to_string(),
            body: serde_json::to_string_pretty(&RefCatalog {
                mutation_epoch: 1,
                tags: HashMap::from([(
                    "release".to_string(),
                    serde_json::to_value(&contents).unwrap(),
                )]),
                branches: HashMap::new(),
                legacy_baseline: LegacyRefState::default(),
            })
            .unwrap(),
        }
        .apply(&object_store)
        .await
        .unwrap();

        let legacy_contents = TagContents {
            version: 99,
            ..contents
        };
        object_store
            .put(
                &release_path,
                serde_json::to_vec(&legacy_contents).unwrap().as_slice(),
            )
            .await
            .unwrap();

        assert_eq!(
            read_stored_ref::<TagContents>(&release_path, &object_store)
                .await
                .unwrap()
                .unwrap()
                .contents
                .unwrap()
                .version,
            99
        );
    }

    #[tokio::test]
    async fn test_later_catalog_commit_wins_over_earlier_legacy_write() {
        let object_store = ObjectStore::memory();
        let root = Path::from("dataset");
        let release_path = tag_path(&root, "release");
        let initial_contents = TagContents {
            branch: None,
            version: 1,
            created_at: None,
            updated_at: None,
            manifest_size: 1,
            metadata: HashMap::new(),
        };
        object_store
            .put(
                &release_path,
                serde_json::to_vec(&initial_contents).unwrap().as_slice(),
            )
            .await
            .unwrap();

        let catalog_contents = TagContents {
            version: 2,
            ..initial_contents.clone()
        };
        let publication = DurableRefPublication {
            epoch: 1,
            path: ref_catalog_version_path(&root, 1).to_string(),
            body: serde_json::to_string_pretty(&RefCatalog {
                mutation_epoch: 1,
                tags: HashMap::from([(
                    "release".to_string(),
                    serde_json::to_value(&catalog_contents).unwrap(),
                )]),
                branches: HashMap::new(),
                legacy_baseline: LegacyRefState {
                    tags: HashMap::from([(
                        "release".to_string(),
                        serde_json::to_value(&initial_contents).unwrap(),
                    )]),
                    branches: HashMap::new(),
                },
            })
            .unwrap(),
        };

        let earlier_legacy_contents = TagContents {
            version: 99,
            ..initial_contents
        };
        object_store
            .put(
                &release_path,
                serde_json::to_vec(&earlier_legacy_contents)
                    .unwrap()
                    .as_slice(),
            )
            .await
            .unwrap();
        publication.apply(&object_store).await.unwrap();

        assert_eq!(
            read_stored_ref::<TagContents>(&release_path, &object_store)
                .await
                .unwrap()
                .unwrap()
                .contents
                .unwrap()
                .version,
            2
        );
    }

    #[tokio::test]
    async fn test_reference_catalog_compacts_history() {
        let object_store = ObjectStore::memory();
        let root = Path::from("dataset");
        for epoch in 1..=100 {
            let contents = TagContents {
                branch: None,
                version: epoch,
                created_at: None,
                updated_at: None,
                manifest_size: 1,
                metadata: HashMap::new(),
            };
            DurableRefPublication {
                epoch,
                path: ref_catalog_version_path(&root, epoch).to_string(),
                body: serde_json::to_string_pretty(&RefCatalog {
                    mutation_epoch: epoch,
                    tags: HashMap::from([(
                        "release".to_string(),
                        serde_json::to_value(contents).unwrap(),
                    )]),
                    branches: HashMap::new(),
                    legacy_baseline: LegacyRefState::default(),
                })
                .unwrap(),
            }
            .apply(&object_store)
            .await
            .unwrap();
        }

        assert_eq!(
            object_store
                .read_dir(base_ref_catalog_path(&root))
                .await
                .unwrap(),
            vec!["00000000000000000099.json", "00000000000000000100.json"]
        );
        assert_eq!(
            read_stored_ref::<TagContents>(&tag_path(&root, "release"), &object_store)
                .await
                .unwrap()
                .unwrap()
                .contents
                .unwrap()
                .version,
            100
        );
    }

    #[tokio::test]
    async fn test_catalog_reader_survives_compaction_churn() {
        let target: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());
        let mut writer = ObjectStore::memory();
        writer.inner = target.clone();
        let root = Path::from("dataset");
        DurableRefPublication {
            epoch: 7,
            path: ref_catalog_version_path(&root, 7).to_string(),
            body: serde_json::to_string_pretty(&RefCatalog {
                mutation_epoch: 7,
                tags: HashMap::new(),
                branches: HashMap::new(),
                legacy_baseline: LegacyRefState::default(),
            })
            .unwrap(),
        }
        .apply(&writer)
        .await
        .unwrap();

        let mut reader = ObjectStore::memory();
        reader.inner = Arc::new(RefTestStore {
            target,
            heartbeat_started: Arc::new(tokio::sync::Notify::new()),
            release_heartbeat: Arc::new(tokio::sync::Notify::new()),
            remaining_catalog_get_failures: AtomicUsize::new(4),
            get_count: Arc::new(AtomicUsize::new(0)),
        });

        assert_eq!(
            read_latest_ref_catalog(&reader, &root)
                .await
                .unwrap()
                .unwrap()
                .mutation_epoch,
            7
        );
    }

    #[tokio::test]
    async fn test_catalog_cleanup_failure_does_not_fail_committed_mutation() {
        let target: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());
        let failing = Arc::new(FailingProxyStore::new());
        let mut object_store = ObjectStore::memory();
        object_store.inner = failing.wrap("", target);
        let root = Path::from("dataset");

        for epoch in 1..=2 {
            DurableRefPublication {
                epoch,
                path: ref_catalog_version_path(&root, epoch).to_string(),
                body: serde_json::to_string_pretty(&RefCatalog {
                    mutation_epoch: epoch,
                    tags: HashMap::new(),
                    branches: HashMap::new(),
                    legacy_baseline: LegacyRefState::default(),
                })
                .unwrap(),
            }
            .apply(&object_store)
            .await
            .unwrap();
        }
        failing.fail_when(
            "delete",
            "_refs/catalog",
            "injected catalog cleanup failure",
        );

        DurableRefPublication {
            epoch: 3,
            path: ref_catalog_version_path(&root, 3).to_string(),
            body: serde_json::to_string_pretty(&RefCatalog {
                mutation_epoch: 3,
                tags: HashMap::new(),
                branches: HashMap::new(),
                legacy_baseline: LegacyRefState::default(),
            })
            .unwrap(),
        }
        .apply(&object_store)
        .await
        .unwrap();

        assert_eq!(
            read_latest_ref_catalog(&object_store, &root)
                .await
                .unwrap()
                .unwrap()
                .mutation_epoch,
            3
        );
    }

    #[tokio::test]
    async fn test_release_failure_does_not_fail_committed_mutation() {
        let target: Arc<dyn object_store::ObjectStore> =
            Arc::new(object_store::memory::InMemory::new());
        let failing = Arc::new(FailingProxyStore::new());
        let mut object_store = ObjectStore::memory();
        object_store.inner = failing.wrap("", target);
        let object_store = Arc::new(object_store);
        let root = Path::from("dataset");
        let fence_path = base_ref_mutation_leases_path(&root);
        let lease_path = lease_epoch_path(&fence_path, 1);
        let state = DurableLeaseState::acquired("owner".to_string(), 1).unwrap();
        assert!(
            create_lease_file(&object_store, &lease_path, &state)
                .await
                .unwrap()
        );
        let mut lease = RefMutationLease {
            handle: DurableLeaseHandle::new(
                object_store.clone(),
                lease_path,
                state,
                Some(fence_path),
            ),
        };

        DurableRefPublication {
            epoch: 1,
            path: ref_catalog_version_path(&root, 1).to_string(),
            body: serde_json::to_string_pretty(&RefCatalog {
                mutation_epoch: 1,
                ..Default::default()
            })
            .unwrap(),
        }
        .apply(&object_store)
        .await
        .unwrap();
        failing.fail_when("put", LEASE_RELEASED_FILE, "injected release failure");

        Refs::finish_mutation(&mut lease, Ok(())).await.unwrap();
        assert_eq!(
            read_latest_ref_catalog(&object_store, &root)
                .await
                .unwrap()
                .unwrap()
                .mutation_epoch,
            1
        );

        failing.clear_fail_when("put", LEASE_RELEASED_FILE);
        lease.release().await.unwrap();
    }

    #[tokio::test]
    async fn test_refs_from_traits() {
        // Test From<u64> for Ref
        let version_ref: Ref = 42u64.into();
        match version_ref {
            VersionNumber(version_number) => {
                assert_eq!(version_number, 42);
            }
            _ => panic!("Expected Version variant"),
        }

        // Test From<&str> for Ref
        let tag_ref: Ref = "test_tag".into();
        match tag_ref {
            Tag(name) => assert_eq!(name, "test_tag"),
            _ => panic!("Expected Tag variant"),
        }

        // Test From<(&str, u64)> for Ref
        let branch_ref: Ref = ("test_branch", 10u64).into();
        match branch_ref {
            Version(name, version) => {
                assert_eq!(name.unwrap(), "test_branch");
                assert_eq!(version, Some(10));
            }
            _ => panic!("Expected Branch variant"),
        }
    }

    #[tokio::test]
    async fn test_branch_contents_serialization() {
        let storage_id = "34e6c4b343a84a7ca40295852ed4d5d8";
        let branch_contents = BranchContents {
            parent_branch: Some("main".to_string()),
            storage: Some(BranchStorage {
                layout: BranchStorageLayout::Detached,
                generation: storage_id.to_string(),
            }),
            identifier: BranchIdentifier {
                version_mapping: vec![(42, storage_id.to_string())],
            },
            parent_version: 42,
            create_at: 1234567890,
            manifest_size: 1024,
            metadata: HashMap::from([("description".to_string(), "production branch".to_string())]),
        };

        // Test serialization
        let json = serde_json::to_string(&branch_contents).unwrap();
        assert!(json.contains("parentBranch"));
        assert!(json.contains("storage"));
        assert!(json.contains("parentVersion"));
        assert!(json.contains("createAt"));
        assert!(json.contains("manifestSize"));
        assert!(json.contains("metadata"));

        // Test deserialization
        let deserialized: BranchContents = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.parent_branch, branch_contents.parent_branch);
        assert_eq!(deserialized.storage, branch_contents.storage);
        assert_eq!(deserialized.parent_version, branch_contents.parent_version);
        assert_eq!(deserialized.create_at, branch_contents.create_at);
        assert_eq!(deserialized.manifest_size, branch_contents.manifest_size);
        assert_eq!(deserialized.metadata, branch_contents.metadata);

        // Backward compatibility: older serialized content does not include metadata.
        let legacy_json = r#"{"parentBranch":"main","parentVersion":42,"createAt":1234567890,"manifestSize":1024}"#;
        let legacy_deserialized: BranchContents = serde_json::from_str(legacy_json).unwrap();
        assert!(legacy_deserialized.storage.is_none());
        assert!(legacy_deserialized.metadata.is_empty());
    }

    #[tokio::test]
    async fn test_branch_synthetic_uuid_is_stable() {
        let legacy_json = r#"{"parentBranch":"main","parentVersion":42,"createAt":1234567890,"manifestSize":1024}"#;
        let store = ObjectStore::memory();
        let base_path = Path::from("dataset");
        let first_path = branch_contents_path(&base_path, "legacy_branch");
        store
            .put(&first_path, legacy_json.as_bytes())
            .await
            .unwrap();
        let second_path = branch_contents_path(&base_path, "legacy_branch_other");
        store
            .put(&second_path, legacy_json.as_bytes())
            .await
            .unwrap();

        let first = BranchContents::from_path(&first_path, &store, "legacy_branch")
            .await
            .unwrap();
        let second = BranchContents::from_path(&first_path, &store, "legacy_branch")
            .await
            .unwrap();
        assert_eq!(first.identifier, second.identifier);
        assert_ne!(
            first.identifier,
            BranchIdentifier::missing_identifier_sentinel()
        );
        assert_eq!(first.identifier.version_mapping[0].1.len(), 32);
        assert!(
            first.identifier.version_mapping[0]
                .1
                .chars()
                .all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase())
        );

        let other = BranchContents::from_path(&second_path, &store, "legacy_branch_other")
            .await
            .unwrap();
        assert_ne!(first.identifier, other.identifier);
    }

    #[tokio::test]
    async fn test_tag_contents_serialization() {
        let tag_contents = TagContents {
            branch: Some("feature".to_string()),
            version: 10,
            created_at: Some(chrono::DateTime::from_timestamp(1_234_567_000, 456_000_000).unwrap()),
            updated_at: Some(chrono::DateTime::from_timestamp(1_234_567_890, 123_000_000).unwrap()),
            manifest_size: 2048,
            metadata: HashMap::from([("channel".to_string(), "release".to_string())]),
        };

        // Test serialization
        let json = serde_json::to_string(&tag_contents).unwrap();
        assert!(json.contains("branch"));
        assert!(json.contains("version"));
        assert!(json.contains("createdAt"));
        assert!(json.contains("updatedAt"));
        assert!(json.contains("manifestSize"));
        assert!(json.contains("metadata"));

        // Test deserialization
        let deserialized: TagContents = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.branch, tag_contents.branch);
        assert_eq!(deserialized.version, tag_contents.version);
        assert_eq!(deserialized.created_at, tag_contents.created_at);
        assert_eq!(deserialized.updated_at, tag_contents.updated_at);
        assert_eq!(deserialized.manifest_size, tag_contents.manifest_size);
        assert_eq!(deserialized.metadata, tag_contents.metadata);

        let tag_contents_without_created_at = TagContents {
            branch: Some("feature".to_string()),
            version: 10,
            created_at: None,
            updated_at: Some(chrono::DateTime::from_timestamp(1_234_567_890, 123_000_000).unwrap()),
            manifest_size: 2048,
            metadata: HashMap::new(),
        };
        let json_without_created_at =
            serde_json::to_string(&tag_contents_without_created_at).unwrap();
        assert!(!json_without_created_at.contains("createdAt"));
        assert!(json_without_created_at.contains("updatedAt"));

        // Backward compatibility: older serialized content does not include timestamps or metadata.
        let legacy_json = r#"{"branch":"feature","version":10,"manifestSize":2048}"#;
        let legacy_deserialized: TagContents = serde_json::from_str(legacy_json).unwrap();
        assert_eq!(legacy_deserialized.created_at, None);
        assert_eq!(legacy_deserialized.updated_at, None);
        assert!(legacy_deserialized.metadata.is_empty());

        let legacy_updated_only_json = r#"{"branch":"feature","version":10,"updatedAt":"2009-02-13T23:31:30.123Z","manifestSize":2048}"#;
        let legacy_updated_only_deserialized: TagContents =
            serde_json::from_str(legacy_updated_only_json).unwrap();
        assert_eq!(legacy_updated_only_deserialized.created_at, None);
        assert_eq!(
            legacy_updated_only_deserialized.updated_at,
            Some(chrono::DateTime::from_timestamp(1_234_567_890, 123_000_000).unwrap())
        );
        assert!(legacy_updated_only_deserialized.metadata.is_empty());
    }

    #[rstest]
    #[case("feature/auth", &["feature/auth/sub"], None)]
    #[case("feature", &["feature/sub1", "feature/sub2"], None)]
    #[case("a/b", &["a/b/c", "b/c/d"], None)]
    #[case("main", &[], Some("main"))]
    #[case("a", &["a"], None)]
    #[case("feature/auth", &["feature/login", "feature/signup"], Some("feature/auth"))]
    #[case("feature/sub", &["feature", "other"], Some("feature/sub"))]
    #[case("very/long/common/prefix/branch1", &["very/long/common/prefix/branch2"], Some("very/long/common/prefix/branch1"))]
    #[case("feature/auth/module", &["feature/other"], Some("feature/auth"))]
    #[case("feature/dev", &["bugfix", "hotfix"], Some("feature"))]
    #[case("branch1", &["dev/branch2", "feature/nathan/branch3", "branch4"], Some("branch1"))]
    fn test_get_cleanup_path(
        #[case] branch_to_delete: &str,
        #[case] remaining_branches: &[&str],
        #[case] expected_relative_cleanup_path: Option<&str>,
    ) {
        let dataset_root_dir = "file:///var/balabala/dataset1".to_string();
        let base_location = BranchLocation {
            path: Path::from(format!("{}/tree/random_branch", dataset_root_dir.as_str())),
            uri: format!("{}/tree/random_branch", dataset_root_dir.as_str()),
            branch: Some("random_branch".to_string()),
        };

        let result =
            Branches::get_cleanup_path(branch_to_delete, remaining_branches, &base_location)
                .unwrap();

        match expected_relative_cleanup_path {
            Some(expected_relative) => {
                assert!(
                    result.is_some(),
                    "Expected cleanup path but got None for branch: {}",
                    branch_to_delete
                );
                let expected_full_path = base_location
                    .find_branch(Some(expected_relative))
                    .unwrap()
                    .path;
                assert_eq!(result.unwrap().as_ref(), expected_full_path.as_ref());
            }
            None => {
                assert!(
                    result.is_none(),
                    "Expected no cleanup but got: {:?} for branch: {}",
                    result,
                    branch_to_delete
                );
            }
        }
    }

    /// Build a reusable mocked BranchContents map mirroring cleanup::lineage_tests::build_lineage_datasets.
    ///
    /// Structure:
    ///    main:v1 ──▶ branch1:v1 ──▶ dev/branch2:v2 ──▶ feature/nathan/branch3:v3
    ///        │
    ///    (main:v2) ──▶ branch4:v2
    ///
    /// Notes:
    /// - The "main" root is virtual (no BranchContents entry).
    /// - Version numbers are representative and monotonically increasing along the chain.
    /// - Tests reuse this builder to ensure consistent lineage and deterministic assertions.
    fn build_mock_branch_contents() -> HashMap<String, BranchContents> {
        fn build(
            parent_name: Option<&str>,
            parent_branch: Option<&BranchContents>,
            parent_ver: u64,
        ) -> BranchContents {
            let parent_branch_id = if let Some(parent_branch) = parent_branch {
                parent_branch.identifier.clone()
            } else {
                BranchIdentifier::main()
            };
            BranchContents {
                parent_branch: parent_name.map(String::from),
                storage: None,
                identifier: BranchIdentifier::new(&parent_branch_id, parent_ver),
                parent_version: parent_ver,
                create_at: 0,
                manifest_size: 1,
                metadata: HashMap::new(),
            }
        }
        let mut contents = HashMap::new();
        contents.insert("branch1".to_string(), build(None, None, 1));
        contents.insert(
            "dev/branch2".to_string(),
            build(Some("branch1"), contents.get("branch1"), 2),
        );
        contents.insert(
            "feature/nathan/branch3".to_string(),
            build(Some("dev/branch2"), contents.get("dev/branch2"), 3),
        );
        contents.insert("branch4".to_string(), build(None, None, 5));
        contents
    }

    #[test]
    fn test_collect_children_for_branch3() {
        let all_branches = build_mock_branch_contents();
        let root_id = all_branches
            .get("feature/nathan/branch3")
            .unwrap()
            .identifier
            .clone();
        assert!(
            root_id
                .collect_referenced_versions(&all_branches)
                .is_empty()
        );
    }

    #[test]
    fn test_collect_children_for_branch2() {
        let all_branches = build_mock_branch_contents();
        let root_id = all_branches.get("dev/branch2").unwrap().identifier.clone();
        let children = root_id.collect_referenced_versions(&all_branches);

        assert_eq!(children.len(), 1);
        assert_eq!(children[0].0.as_str(), "feature/nathan/branch3");
        assert_eq!(children[0].1, 3);
    }

    #[test]
    fn test_collect_children_for_branch1() {
        let all_branches = build_mock_branch_contents();
        let root_id = all_branches.get("branch1").unwrap().identifier.clone();
        let children = root_id.collect_referenced_versions(&all_branches);

        assert_eq!(children.len(), 2);
        assert_eq!(children[0].0.as_str(), "feature/nathan/branch3");
        assert_eq!(children[1].0.as_str(), "dev/branch2");
        assert_eq!(children[0].1, 2);
        assert_eq!(children[1].1, 2);
    }

    #[test]
    fn test_collect_children_for_main() {
        let all_branches = build_mock_branch_contents();
        let root_id = BranchIdentifier::main();
        let children = root_id.collect_referenced_versions(&all_branches);

        assert_eq!(children.len(), 4);
        assert_eq!(children[0].0.as_str(), "branch4");
        assert_eq!(children[1].0.as_str(), "feature/nathan/branch3");
        assert_eq!(children[2].0.as_str(), "dev/branch2");
        assert_eq!(children[3].0.as_str(), "branch1");
        assert_eq!(children[0].1, 5);
        assert_eq!(children[1].1, 1);
        assert_eq!(children[2].1, 1);
        assert_eq!(children[3].1, 1);
    }
}
