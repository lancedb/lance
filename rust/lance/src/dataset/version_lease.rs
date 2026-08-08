// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Advisory leases that protect dataset versions from cleanup.

use std::{
    collections::{HashMap, HashSet},
    fs::{File, OpenOptions},
    path::PathBuf,
    sync::{Arc, LazyLock},
    time::Duration,
};

use bytes::Bytes;
use chrono::{DateTime, TimeDelta, Utc};
use dashmap::DashSet;
use futures::{StreamExt, TryStreamExt, stream};
use lance_io::object_store::{ConditionalDeleteResult, ObjectStore};
use object_store::{ObjectMeta, ObjectStoreExt, PutMode, PutOptions, UpdateVersion, path::Path};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{
    Dataset,
    refs::{BranchIdentifier, MAIN_BRANCH, Refs},
};
use crate::{Error, Result};

const LEASES_DIR: &str = "_refs/version_leases";
const LEASE_GC_MARKERS_DIR: &str = "_refs/version_lease_gc";
const REFERENCE_INTENTS_DIR: &str = "_refs/version_reference_intents";
const REFERENCE_STATES_DIR: &str = "_refs/version_reference_states";
const BRANCH_TERMINATIONS_DIR: &str = "_refs/version_branch_terminations";
const LEASE_FILE_SUFFIX: &str = ".lease";
const REFERENCE_INTENT_SUFFIX: &str = ".intent";
const DRAINING_MARKER_SUFFIX: &str = ".draining";
const SEALED_MARKER_SUFFIX: &str = ".sealed";
const COMMITTED_MARKER_SUFFIX: &str = ".committed";
const STORAGE_CLOCK_PATH: &str = "_clock";

const REFERENCE_GENERATION_FIELD: &str = "_lanceReferenceGeneration";

/// HTTP `Last-Modified`, used by supported cloud stores, has whole-second precision.
/// Adding one interval guarantees a lease is never shortened by timestamp truncation.
const STORAGE_TIMESTAMP_PRECISION: Duration = Duration::from_secs(1);

/// Draining does no deletion and must complete within this ownership window.
/// A cleaner that exceeds the window fails before sealing; another actor may then
/// ignore and remove the abandoned drain without racing a live deletion.
const DRAINING_OWNERSHIP_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// Reference publication is expected to be short-lived. An abandoned intent
/// protects its target for this ownership window and is then recoverable by
/// cleanup, preventing a crashed writer from retaining a version forever.
const REFERENCE_ADMISSION_TIMEOUT: Duration = Duration::from_secs(15 * 60);

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
    root_path: Path,
    namespace: String,
    leases_path: Path,
    markers_path: Path,
    reference_intents_path: Path,
    manifest_path: Option<Path>,
    canonical_references: Option<CanonicalReferenceContext>,
}

#[derive(Clone, Debug)]
struct CanonicalReferenceContext {
    refs: Refs,
    branch: Option<String>,
    branch_identifier: BranchIdentifier,
}

/// Durable ownership of one in-flight tag or branch publication.
///
/// This type intentionally has no `Drop` cleanup. Cancelling a future while a
/// remote conditional write is in flight must leave the intent durable until
/// cleanup can safely expire it.
#[derive(Debug)]
pub(super) struct ReferenceAdmission {
    store: VersionLeaseStore,
    path: Path,
    manifest_path: Path,
    version: u64,
    created_at: DateTime<Utc>,
    operation_id: String,
    mutation: ReferenceMutation,
}

/// An operating-system advisory lock for one local canonical reference.
///
/// The lock file is intentionally retained after unlock. Reusing one inode
/// avoids a delete/recreate race between local processes, while the OS releases
/// the held lock automatically if its owner exits.
#[derive(Debug)]
pub(super) struct LocalReferenceLock {
    _file: File,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub(super) enum ReferenceMutation {
    Create {
        path: String,
        payload: Vec<u8>,
    },
    Update {
        path: String,
        expected_payload: Vec<u8>,
        expected_etag: Option<String>,
        expected_version: Option<String>,
        payload: Vec<u8>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ReferenceIntent {
    manifest_path: String,
    mutation: ReferenceMutation,
    #[serde(default)]
    operation_id: String,
    #[serde(default)]
    state: ReferenceIntentState,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ReferenceTarget {
    namespace: String,
    version: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", tag = "state")]
enum ReferenceLifecycleState {
    Pending {
        canonical_path: String,
        operation_id: String,
        target: ReferenceTarget,
        previous: Option<ReferenceLiveState>,
        previous_was_legacy: bool,
    },
    Live {
        canonical_path: String,
        live: ReferenceLiveState,
    },
    Legacy {
        canonical_path: String,
    },
    Vacant {
        canonical_path: String,
    },
    Revoking {
        canonical_path: String,
        operation_id: String,
        target: ReferenceTarget,
        previous: Option<ReferenceLiveState>,
        previous_was_legacy: bool,
        mutation: ReferenceMutation,
    },
    Deleting {
        canonical_path: String,
        operation_id: String,
        previous: Option<ReferenceLiveState>,
        previous_was_legacy: bool,
        expected_payload: Vec<u8>,
        expected_etag: Option<String>,
        expected_version: Option<String>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ReferenceLiveState {
    generation: String,
    target: ReferenceTarget,
}

#[derive(Debug)]
struct ReferenceLifecycleSnapshot {
    metadata: ObjectMeta,
    payload: Bytes,
    state: ReferenceLifecycleState,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
enum ReferenceIntentState {
    #[default]
    Pending,
    Completed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompletedReferenceIntentHandling {
    DeferToCanonicalCensus,
    RetainForCurrentScan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReferenceMutationOutcome {
    Published,
    Conflict,
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

#[derive(Debug)]
struct ReferenceIntentMetadata {
    version: u64,
    metadata: ObjectMeta,
    intent: ReferenceIntent,
}

#[derive(Debug)]
pub(super) struct CanonicalReferenceCensus {
    pub(super) versions: HashSet<u64>,
    pub(super) completed_intent_paths: HashSet<Path>,
    pub(super) lifecycle_generations: HashMap<Path, String>,
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
            root_path: root.path.clone(),
            namespace: namespace.to_string(),
            leases_path: root.path.clone().join(LEASES_DIR).join(namespace),
            markers_path: root.path.clone().join(LEASE_GC_MARKERS_DIR).join(namespace),
            reference_intents_path: root
                .path
                .clone()
                .join(REFERENCE_INTENTS_DIR)
                .join(namespace),
            manifest_path,
            canonical_references: Some(CanonicalReferenceContext {
                refs: refs.clone(),
                branch: branch.map(ToOwned::to_owned),
                branch_identifier,
            }),
        })
    }

    fn lifecycle_only(object_store: Arc<ObjectStore>, root_path: Path) -> Self {
        Self {
            object_store,
            root_path: root_path.clone(),
            namespace: String::new(),
            leases_path: root_path.clone().join(LEASES_DIR),
            markers_path: root_path.clone().join(LEASE_GC_MARKERS_DIR),
            reference_intents_path: root_path.join(REFERENCE_INTENTS_DIR),
            manifest_path: None,
            canonical_references: None,
        }
    }

    fn reference_state_path(&self, canonical_path: &Path) -> Path {
        self.root_path
            .clone()
            .join(REFERENCE_STATES_DIR)
            .join(format!(
                "{}.state",
                stable_reference_path_id(canonical_path.as_ref())
            ))
    }

    async fn reference_lifecycle_snapshot(
        &self,
        canonical_path: &Path,
    ) -> Result<Option<ReferenceLifecycleSnapshot>> {
        let state_path = self.reference_state_path(canonical_path);
        let result = match self.object_store.inner.get(&state_path).await {
            Ok(result) => result,
            Err(object_store::Error::NotFound { .. }) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let metadata = result.meta.clone();
        let payload = result.bytes().await?;
        let state: ReferenceLifecycleState = serde_json::from_slice(&payload)
            .map_err(|error| Error::corrupt_file(state_path, error.to_string()))?;
        let recorded_path = match &state {
            ReferenceLifecycleState::Pending { canonical_path, .. }
            | ReferenceLifecycleState::Live { canonical_path, .. }
            | ReferenceLifecycleState::Legacy { canonical_path }
            | ReferenceLifecycleState::Vacant { canonical_path }
            | ReferenceLifecycleState::Revoking { canonical_path, .. }
            | ReferenceLifecycleState::Deleting { canonical_path, .. } => canonical_path,
        };
        if recorded_path != canonical_path.as_ref() {
            return Err(Error::corrupt_file(
                metadata.location.clone(),
                format!(
                    "reference lifecycle state belongs to {recorded_path}, not {canonical_path}"
                ),
            ));
        }
        Ok(Some(ReferenceLifecycleSnapshot {
            metadata,
            payload,
            state,
        }))
    }

    async fn put_reference_lifecycle(
        &self,
        canonical_path: &Path,
        expected: Option<&ReferenceLifecycleSnapshot>,
        state: &ReferenceLifecycleState,
        has_local_lock: bool,
    ) -> Result<Option<ReferenceLifecycleSnapshot>> {
        let state_path = self.reference_state_path(canonical_path);
        let payload = Bytes::from(serde_json::to_vec(state)?);
        let mode = expected.map_or(PutMode::Create, |snapshot| {
            PutMode::Update(UpdateVersion {
                e_tag: snapshot.metadata.e_tag.clone(),
                version: snapshot.metadata.version.clone(),
            })
        });
        let result = self
            .object_store
            .inner
            .put_opts(
                &state_path,
                payload.clone().into(),
                PutOptions {
                    mode,
                    ..Default::default()
                },
            )
            .await;
        let result = match result {
            Err(
                object_store::Error::NotSupported { .. }
                | object_store::Error::NotImplemented { .. },
            ) if has_local_lock => {
                let current = match self.object_store.inner.get(&state_path).await {
                    Ok(result) => Some(result.bytes().await?),
                    Err(object_store::Error::NotFound { .. }) => None,
                    Err(error) => return Err(error.into()),
                };
                let matches_expected = match (expected, current.as_deref()) {
                    (None, None) => true,
                    (Some(expected), Some(current)) => current == expected.payload.as_ref(),
                    _ => false,
                };
                if !matches_expected {
                    return Ok(None);
                }
                self.object_store
                    .inner
                    .put(&state_path, payload.into())
                    .await
            }
            Err(
                object_store::Error::NotSupported { .. }
                | object_store::Error::NotImplemented { .. },
            ) => {
                return Err(Error::not_supported(format!(
                    "object store {} does not support atomic reference lifecycle updates for {canonical_path}",
                    self.object_store.scheme()
                )));
            }
            result => result,
        };
        match result {
            Ok(_) => self.reference_lifecycle_snapshot(canonical_path).await,
            Err(
                object_store::Error::AlreadyExists { .. }
                | object_store::Error::Precondition { .. }
                | object_store::Error::NotFound { .. },
            ) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    async fn put_terminal_reference_lifecycle(
        &self,
        canonical_path: &Path,
        expected: &ReferenceLifecycleSnapshot,
        terminal: &ReferenceLifecycleState,
        has_local_lock: bool,
    ) -> Result<bool> {
        let Some(snapshot) = self
            .put_reference_lifecycle(canonical_path, Some(expected), terminal, has_local_lock)
            .await?
        else {
            return Ok(false);
        };
        if matches!(terminal, ReferenceLifecycleState::Vacant { .. })
            && matches!(snapshot.state, ReferenceLifecycleState::Vacant { .. })
        {
            let expected = UpdateVersion {
                e_tag: snapshot.metadata.e_tag.clone(),
                version: snapshot.metadata.version.clone(),
            };
            match self
                .object_store
                .delete_if_matches(&snapshot.metadata.location, &expected)
                .await?
            {
                ConditionalDeleteResult::Deleted
                | ConditionalDeleteResult::NotFound
                | ConditionalDeleteResult::IdentityMismatch => {}
            }
        }
        Ok(true)
    }

    async fn claim_reference_lifecycle(
        &self,
        canonical_path: &Path,
        operation_id: &str,
        version: u64,
        previous_was_legacy: bool,
    ) -> Result<()> {
        let local_lock = lock_local_reference(&self.object_store, canonical_path).await?;
        loop {
            let current = self.reference_lifecycle_snapshot(canonical_path).await?;
            let (previous, previous_was_legacy) = match current
                .as_ref()
                .map(|snapshot| &snapshot.state)
            {
                None | Some(ReferenceLifecycleState::Vacant { .. }) => (None, previous_was_legacy),
                Some(ReferenceLifecycleState::Live { live, .. }) => (Some(live.clone()), false),
                Some(ReferenceLifecycleState::Legacy { .. }) => (None, true),
                Some(ReferenceLifecycleState::Pending { .. }) => {
                    return Err(Error::RefConflict {
                        message: format!("reference {canonical_path} has an in-flight mutation"),
                    });
                }
                Some(ReferenceLifecycleState::Revoking { .. }) => {
                    let current = current.as_ref().ok_or_else(|| {
                        Error::internal(format!(
                            "reference {canonical_path} lost its revoking state"
                        ))
                    })?;
                    if !self
                        .reconcile_revoking_reference(canonical_path, current, local_lock.is_some())
                        .await?
                    {
                        return Err(Error::RefConflict {
                            message: format!(
                                "reference {canonical_path} is settling a revoked mutation"
                            ),
                        });
                    }
                    continue;
                }
                Some(ReferenceLifecycleState::Deleting { .. }) => {
                    let current = current.as_ref().ok_or_else(|| {
                        Error::internal(format!(
                            "reference {canonical_path} lost its deleting state"
                        ))
                    })?;
                    self.reconcile_deleting_reference(
                        canonical_path,
                        current,
                        local_lock.is_some(),
                    )
                    .await?;
                    continue;
                }
            };
            let pending = ReferenceLifecycleState::Pending {
                canonical_path: canonical_path.to_string(),
                operation_id: operation_id.to_string(),
                target: ReferenceTarget {
                    namespace: self.namespace.clone(),
                    version,
                },
                previous,
                previous_was_legacy,
            };
            if self
                .put_reference_lifecycle(
                    canonical_path,
                    current.as_ref(),
                    &pending,
                    local_lock.is_some(),
                )
                .await?
                .is_some()
            {
                return Ok(());
            }
            return Err(Error::RefConflict {
                message: format!("reference {canonical_path} lifecycle changed during admission"),
            });
        }
    }

    async fn complete_reference_lifecycle(
        &self,
        canonical_path: &Path,
        operation_id: &str,
        version: u64,
        has_local_lock: bool,
    ) -> Result<bool> {
        let Some(current) = self.reference_lifecycle_snapshot(canonical_path).await? else {
            return Ok(false);
        };
        if !matches!(
            &current.state,
            ReferenceLifecycleState::Pending {
                operation_id: current_operation,
                ..
            } if current_operation == operation_id
        ) {
            return Ok(false);
        }
        let canonical = match self.object_store.inner.get(canonical_path).await {
            Ok(result) => result.bytes().await?,
            Err(object_store::Error::NotFound { .. }) => return Ok(false),
            Err(error) => return Err(error.into()),
        };
        if payload_reference_generation(&canonical)?.as_deref() != Some(operation_id) {
            return Ok(false);
        }
        let live = ReferenceLifecycleState::Live {
            canonical_path: canonical_path.to_string(),
            live: ReferenceLiveState {
                generation: operation_id.to_string(),
                target: ReferenceTarget {
                    namespace: self.namespace.clone(),
                    version,
                },
            },
        };
        Ok(self
            .put_reference_lifecycle(canonical_path, Some(&current), &live, has_local_lock)
            .await?
            .is_some())
    }

    async fn cancel_reference_lifecycle(
        &self,
        canonical_path: &Path,
        operation_id: &str,
        has_local_lock: bool,
    ) -> Result<bool> {
        let Some(current) = self.reference_lifecycle_snapshot(canonical_path).await? else {
            return Ok(false);
        };
        let previous = match &current.state {
            ReferenceLifecycleState::Pending {
                operation_id: current_operation,
                previous,
                ..
            } if current_operation == operation_id => previous.clone(),
            _ => return Ok(false),
        };
        if let Some(previous) = previous {
            let live = ReferenceLifecycleState::Live {
                canonical_path: canonical_path.to_string(),
                live: previous,
            };
            return Ok(self
                .put_reference_lifecycle(canonical_path, Some(&current), &live, has_local_lock)
                .await?
                .is_some());
        }
        if matches!(
            &current.state,
            ReferenceLifecycleState::Pending {
                previous_was_legacy: true,
                ..
            }
        ) {
            let legacy = ReferenceLifecycleState::Legacy {
                canonical_path: canonical_path.to_string(),
            };
            return Ok(self
                .put_reference_lifecycle(canonical_path, Some(&current), &legacy, has_local_lock)
                .await?
                .is_some());
        }
        let vacant = ReferenceLifecycleState::Vacant {
            canonical_path: canonical_path.to_string(),
        };
        self.put_terminal_reference_lifecycle(canonical_path, &current, &vacant, has_local_lock)
            .await
    }

    fn branch_termination_path(&self) -> Path {
        self.root_path
            .clone()
            .join(BRANCH_TERMINATIONS_DIR)
            .join(format!("{}.deleted", self.namespace))
    }

    async fn ensure_branch_incarnation_active(&self) -> Result<()> {
        if self.namespace != MAIN_BRANCH
            && self
                .object_store
                .exists(&self.branch_termination_path())
                .await?
        {
            return Err(Error::RefConflict {
                message: "parent branch incarnation is being deleted".to_string(),
            });
        }
        Ok(())
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

    pub(super) async fn active_reference_versions(&self) -> Result<HashSet<u64>> {
        self.reconcile_reference_lifecycles().await?;
        let mut versions = self
            .reference_versions(
                CompletedReferenceIntentHandling::RetainForCurrentScan,
                &HashSet::new(),
            )
            .await?
            .versions;
        versions.extend(self.lifecycle_reference_versions().await?);
        versions.extend(self.canonical_reference_versions().await?);
        Ok(versions)
    }

    async fn reconcile_reference_lifecycles(&self) -> Result<()> {
        let state_path = self.root_path.clone().join(REFERENCE_STATES_DIR);
        let metadata = self
            .object_store
            .list(Some(state_path))
            .try_collect::<Vec<_>>()
            .await?;
        for listed in metadata {
            let result = match self.object_store.inner.get(&listed.location).await {
                Ok(result) => result,
                Err(object_store::Error::NotFound { .. }) => continue,
                Err(error) => return Err(error.into()),
            };
            let current_metadata = result.meta.clone();
            let payload = result.bytes().await?;
            let state: ReferenceLifecycleState = serde_json::from_slice(&payload)
                .map_err(|error| Error::corrupt_file(listed.location.clone(), error.to_string()))?;
            let canonical_path = match &state {
                ReferenceLifecycleState::Revoking { canonical_path, .. }
                | ReferenceLifecycleState::Deleting { canonical_path, .. } => {
                    Some(Path::parse(canonical_path)?)
                }
                ReferenceLifecycleState::Vacant { .. } => {
                    if self.object_store.supports_conditional_delete() {
                        let expected = UpdateVersion {
                            e_tag: current_metadata.e_tag,
                            version: current_metadata.version,
                        };
                        self.object_store
                            .delete_if_matches(&listed.location, &expected)
                            .await?;
                    }
                    None
                }
                _ => None,
            };
            let Some(canonical_path) = canonical_path else {
                continue;
            };
            let local_lock = lock_local_reference(&self.object_store, &canonical_path).await?;
            let Some(snapshot) = self.reference_lifecycle_snapshot(&canonical_path).await? else {
                continue;
            };
            match &snapshot.state {
                ReferenceLifecycleState::Revoking { .. } => {
                    self.reconcile_revoking_reference(
                        &canonical_path,
                        &snapshot,
                        local_lock.is_some(),
                    )
                    .await?;
                }
                ReferenceLifecycleState::Deleting { .. } => {
                    self.reconcile_deleting_reference(
                        &canonical_path,
                        &snapshot,
                        local_lock.is_some(),
                    )
                    .await?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    async fn lifecycle_reference_versions(&self) -> Result<HashSet<u64>> {
        let state_path = self.root_path.clone().join(REFERENCE_STATES_DIR);
        let metadata = self
            .object_store
            .list(Some(state_path))
            .try_collect::<Vec<_>>()
            .await?;
        stream::iter(metadata)
            .map(|metadata| async move {
                let payload = self
                    .object_store
                    .inner
                    .get(&metadata.location)
                    .await?
                    .bytes()
                    .await?;
                let state: ReferenceLifecycleState =
                    serde_json::from_slice(&payload).map_err(|error| {
                        Error::corrupt_file(metadata.location.clone(), error.to_string())
                    })?;
                let target = self.retained_lifecycle_target(&state).await?;
                Ok::<_, Error>(target.and_then(|target| {
                    (target.namespace == self.namespace).then_some(target.version)
                }))
            })
            .buffer_unordered(self.object_store.io_parallelism())
            .try_collect::<Vec<_>>()
            .await
            .map(|versions| versions.into_iter().flatten().collect())
    }

    async fn lifecycle_generation_snapshot(&self) -> Result<HashMap<Path, String>> {
        let state_path = self.root_path.clone().join(REFERENCE_STATES_DIR);
        let metadata = self
            .object_store
            .list(Some(state_path))
            .try_collect::<Vec<_>>()
            .await?;
        stream::iter(metadata)
            .map(|metadata| async move {
                let payload = self
                    .object_store
                    .inner
                    .get(&metadata.location)
                    .await?
                    .bytes()
                    .await?;
                let state: ReferenceLifecycleState =
                    serde_json::from_slice(&payload).map_err(|error| {
                        Error::corrupt_file(metadata.location.clone(), error.to_string())
                    })?;
                let generation = match state {
                    ReferenceLifecycleState::Pending { operation_id, .. } => Some(operation_id),
                    ReferenceLifecycleState::Live { live, .. } => Some(live.generation),
                    ReferenceLifecycleState::Revoking { operation_id, .. }
                    | ReferenceLifecycleState::Deleting { operation_id, .. } => Some(operation_id),
                    ReferenceLifecycleState::Legacy { .. } => Some("legacy".to_string()),
                    ReferenceLifecycleState::Vacant { .. } => Some("vacant".to_string()),
                };
                Ok::<_, Error>(generation.map(|generation| (metadata.location, generation)))
            })
            .buffer_unordered(self.object_store.io_parallelism())
            .try_collect::<Vec<_>>()
            .await
            .map(|generations| generations.into_iter().flatten().collect())
    }

    async fn lifecycle_reference_versions_since(
        &self,
        observed_generations: &HashMap<Path, String>,
    ) -> Result<HashSet<u64>> {
        let state_path = self.root_path.clone().join(REFERENCE_STATES_DIR);
        let metadata = self
            .object_store
            .list(Some(state_path))
            .try_collect::<Vec<_>>()
            .await?;
        stream::iter(metadata)
            .map(|metadata| async move {
                let payload = self
                    .object_store
                    .inner
                    .get(&metadata.location)
                    .await?
                    .bytes()
                    .await?;
                let state: ReferenceLifecycleState =
                    serde_json::from_slice(&payload).map_err(|error| {
                        Error::corrupt_file(metadata.location.clone(), error.to_string())
                    })?;
                let generation = match &state {
                    ReferenceLifecycleState::Pending { operation_id, .. } => operation_id.clone(),
                    ReferenceLifecycleState::Live { live, .. } => live.generation.clone(),
                    ReferenceLifecycleState::Revoking { operation_id, .. }
                    | ReferenceLifecycleState::Deleting { operation_id, .. } => {
                        operation_id.clone()
                    }
                    ReferenceLifecycleState::Legacy { .. } => "legacy".to_string(),
                    ReferenceLifecycleState::Vacant { .. } => "vacant".to_string(),
                };
                let target = self.retained_lifecycle_target(&state).await?;
                let changed = observed_generations.get(&metadata.location) != Some(&generation);
                Ok::<_, Error>(target.and_then(|target| {
                    (changed && target.namespace == self.namespace).then_some(target.version)
                }))
            })
            .buffer_unordered(self.object_store.io_parallelism())
            .try_collect::<Vec<_>>()
            .await
            .map(|versions| versions.into_iter().flatten().collect())
    }

    async fn retained_lifecycle_target(
        &self,
        state: &ReferenceLifecycleState,
    ) -> Result<Option<ReferenceTarget>> {
        let (canonical_path, expected) = match state {
            ReferenceLifecycleState::Pending { target, .. } => return Ok(Some(target.clone())),
            ReferenceLifecycleState::Live {
                canonical_path,
                live,
            } => (canonical_path, Some(live)),
            ReferenceLifecycleState::Revoking {
                canonical_path,
                previous,
                ..
            }
            | ReferenceLifecycleState::Deleting {
                canonical_path,
                previous,
                ..
            } => (canonical_path, previous.as_ref()),
            ReferenceLifecycleState::Legacy { .. } | ReferenceLifecycleState::Vacant { .. } => {
                return Ok(None);
            }
        };
        let canonical_path = Path::parse(canonical_path)?;
        let payload = match self.object_store.inner.get(&canonical_path).await {
            Ok(result) => result.bytes().await?,
            Err(object_store::Error::NotFound { .. }) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let generation = payload_reference_generation(&payload)?;
        Ok(expected.and_then(|live| {
            (generation.as_deref() == Some(live.generation.as_str())).then(|| live.target.clone())
        }))
    }

    async fn canonical_reference_versions(&self) -> Result<HashSet<u64>> {
        let Some(context) = &self.canonical_references else {
            return Ok(HashSet::new());
        };
        let tags = context.refs.tags().list().await?;
        let branches = context.refs.branches().list().await?;
        let mut versions = tags
            .values()
            .filter(|tag| tag.branch == context.branch)
            .map(|tag| tag.version)
            .collect::<HashSet<_>>();
        versions.extend(
            context
                .branch_identifier
                .collect_referenced_versions(&branches)
                .into_iter()
                .map(|(_, version)| version),
        );
        Ok(versions)
    }

    /// Fence in-flight publication before the caller scans canonical tags and
    /// branches. Completed intents can be removed without retaining their
    /// target because the following canonical census observes their result.
    pub(super) async fn reference_versions_before_canonical_census(
        &self,
    ) -> Result<CanonicalReferenceCensus> {
        self.reconcile_reference_lifecycles().await?;
        let mut census = self
            .reference_versions(
                CompletedReferenceIntentHandling::DeferToCanonicalCensus,
                &HashSet::new(),
            )
            .await?;
        census.lifecycle_generations = self.lifecycle_generation_snapshot().await?;
        Ok(census)
    }

    async fn reference_versions(
        &self,
        completed_handling: CompletedReferenceIntentHandling,
        completed_intents_observed_before_census: &HashSet<Path>,
    ) -> Result<CanonicalReferenceCensus> {
        let observed_at = self.storage_observed_at().await?;
        let mut active_versions = HashSet::new();
        let mut completed_intent_paths = HashSet::new();
        let mut stale_paths = Vec::new();
        for intent in self.reference_intent_metadata().await? {
            match intent.intent.state {
                ReferenceIntentState::Completed => {
                    if completed_intents_observed_before_census.contains(&intent.metadata.location)
                    {
                        stale_paths.push(intent.metadata.location);
                        continue;
                    }
                    // The completed intent is the durable handoff from an
                    // in-flight publication to its canonical object. A caller
                    // that already completed its canonical census retains a
                    // matching payload for this scan; a caller about to census
                    // canonical state can defer to that newer observation.
                    if completed_handling == CompletedReferenceIntentHandling::RetainForCurrentScan
                    {
                        let (path, payload) = match &intent.intent.mutation {
                            ReferenceMutation::Create { path, payload }
                            | ReferenceMutation::Update { path, payload, .. } => {
                                (Path::parse(path)?, payload)
                            }
                        };
                        if self
                            .reference_mutation_conflict_outcome(&path, payload)
                            .await?
                            == ReferenceMutationOutcome::Published
                        {
                            active_versions.insert(intent.version);
                        }
                    }
                    // The pre-census scan must not consume the only durable
                    // handoff while a pre-commit seal exists. If cleanup is
                    // interrupted before its canonical census cancels that
                    // seal, recovery still needs the completed intent to
                    // recognize that the canonical reference won.
                    if completed_handling
                        == CompletedReferenceIntentHandling::DeferToCanonicalCensus
                        && self.has_precommit_sealed_marker(intent.version).await?
                    {
                        completed_intent_paths.insert(intent.metadata.location);
                    } else {
                        stale_paths.push(intent.metadata.location);
                    }
                }
                ReferenceIntentState::Pending
                    if reference_owner_is_active(intent.metadata.last_modified, observed_at)? =>
                {
                    active_versions.insert(intent.version);
                }
                ReferenceIntentState::Pending => {
                    let manifest_path = Path::parse(&intent.intent.manifest_path)?;
                    let allow_publish = self.object_store.exists(&manifest_path).await?
                        && !self.has_committed_marker(intent.version).await?;
                    if self
                        .expired_reference_mutation_outcome(
                            &intent.intent.mutation,
                            &intent.intent.operation_id,
                            intent.version,
                            allow_publish,
                        )
                        .await?
                        == ReferenceMutationOutcome::Published
                    {
                        active_versions.insert(intent.version);
                    }
                    stale_paths.push(intent.metadata.location);
                }
            }
        }
        self.delete_paths(stale_paths).await?;
        Ok(CanonicalReferenceCensus {
            versions: active_versions,
            completed_intent_paths,
            lifecycle_generations: HashMap::new(),
        })
    }

    async fn expired_reference_mutation_outcome(
        &self,
        mutation: &ReferenceMutation,
        operation_id: &str,
        version: u64,
        allow_publish: bool,
    ) -> Result<ReferenceMutationOutcome> {
        let (path, intended_payload) = match mutation {
            ReferenceMutation::Create { path, payload }
            | ReferenceMutation::Update { path, payload, .. } => {
                (Path::parse(path)?, payload.as_slice())
            }
        };
        let local_lock = lock_local_reference(&self.object_store, &path).await?;
        if operation_id.is_empty() {
            return self
                .reference_mutation_conflict_outcome(&path, intended_payload)
                .await;
        }

        let lifecycle = self.reference_lifecycle_snapshot(&path).await?;
        match lifecycle.as_ref().map(|snapshot| &snapshot.state) {
            Some(ReferenceLifecycleState::Live { live, .. }) if live.generation == operation_id => {
                return Ok(ReferenceMutationOutcome::Published);
            }
            Some(ReferenceLifecycleState::Pending {
                operation_id: current_operation,
                ..
            }) if current_operation == operation_id => {}
            Some(ReferenceLifecycleState::Revoking {
                operation_id: current_operation,
                ..
            }) if current_operation == operation_id => {
                let lifecycle = lifecycle.as_ref().ok_or_else(|| {
                    Error::internal(format!(
                        "reference {path} lost its revoking lifecycle state"
                    ))
                })?;
                self.reconcile_revoking_reference(&path, lifecycle, local_lock.is_some())
                    .await?;
                return Ok(ReferenceMutationOutcome::Conflict);
            }
            _ => return Ok(ReferenceMutationOutcome::Conflict),
        }

        let canonical = match self.object_store.inner.get(&path).await {
            Ok(result) => {
                let metadata = result.meta.clone();
                Some((metadata, result.bytes().await?))
            }
            Err(object_store::Error::NotFound { .. }) => None,
            Err(error) => return Err(error.into()),
        };
        let canonical_is_intended = canonical
            .as_ref()
            .is_some_and(|(_, payload)| payload.as_ref() == intended_payload);
        if canonical_is_intended && allow_publish {
            if self
                .complete_reference_lifecycle(&path, operation_id, version, local_lock.is_some())
                .await?
            {
                return Ok(ReferenceMutationOutcome::Published);
            }
            return Ok(ReferenceMutationOutcome::Conflict);
        }
        let Some(revoking) = self
            .revoke_reference_lifecycle(&path, operation_id, mutation, local_lock.is_some())
            .await?
        else {
            return Ok(ReferenceMutationOutcome::Conflict);
        };
        self.reconcile_revoking_reference(&path, &revoking, local_lock.is_some())
            .await?;
        Ok(ReferenceMutationOutcome::Conflict)
    }

    async fn revoke_reference_lifecycle(
        &self,
        canonical_path: &Path,
        operation_id: &str,
        mutation: &ReferenceMutation,
        has_local_lock: bool,
    ) -> Result<Option<ReferenceLifecycleSnapshot>> {
        let Some(current) = self.reference_lifecycle_snapshot(canonical_path).await? else {
            return Ok(None);
        };
        if let ReferenceLifecycleState::Revoking {
            operation_id: current_operation,
            ..
        } = &current.state
            && current_operation == operation_id
        {
            return Ok(Some(current));
        }
        let ReferenceLifecycleState::Pending {
            operation_id: current_operation,
            target,
            previous,
            previous_was_legacy,
            ..
        } = &current.state
        else {
            return Ok(None);
        };
        if current_operation != operation_id {
            return Ok(None);
        }
        let revoking = ReferenceLifecycleState::Revoking {
            canonical_path: canonical_path.to_string(),
            operation_id: operation_id.to_string(),
            target: target.clone(),
            previous: previous.clone(),
            previous_was_legacy: *previous_was_legacy,
            mutation: mutation.clone(),
        };
        self.put_reference_lifecycle(canonical_path, Some(&current), &revoking, has_local_lock)
            .await
    }

    async fn reconcile_revoking_reference(
        &self,
        canonical_path: &Path,
        revoking: &ReferenceLifecycleSnapshot,
        has_local_lock: bool,
    ) -> Result<bool> {
        let ReferenceLifecycleState::Revoking {
            previous, mutation, ..
        } = &revoking.state
        else {
            return Ok(false);
        };
        let current = match self.object_store.inner.get(canonical_path).await {
            Ok(result) => {
                let metadata = result.meta.clone();
                Some((metadata, result.bytes().await?))
            }
            Err(object_store::Error::NotFound { .. }) => None,
            Err(error) => return Err(error.into()),
        };
        if matches!(mutation, ReferenceMutation::Create { .. })
            && current.is_none()
            && reference_owner_is_active(
                revoking.metadata.last_modified,
                self.storage_observed_at().await?,
            )?
        {
            // A conditional create may already be in flight when its owner is
            // revoked. Keep the generation until the request timeout window is
            // closed so a late exact-incarnation write remains recoverable.
            return Ok(false);
        }
        match (mutation, current.as_ref()) {
            (
                ReferenceMutation::Update {
                    expected_payload,
                    payload,
                    ..
                },
                Some((metadata, current_payload)),
            ) if current_payload.as_ref() == payload => {
                self.rewrite_reference_payload(
                    canonical_path,
                    metadata,
                    current_payload,
                    expected_payload,
                    has_local_lock,
                )
                .await?;
            }
            (
                ReferenceMutation::Update {
                    expected_payload, ..
                },
                Some((metadata, current_payload)),
            ) if current_payload.as_ref() == expected_payload => {
                let fenced_payload = fenced_reference_payload(current_payload);
                self.rewrite_reference_payload(
                    canonical_path,
                    metadata,
                    current_payload,
                    &fenced_payload,
                    has_local_lock,
                )
                .await?;
            }
            (ReferenceMutation::Create { .. }, None) => {}
            (ReferenceMutation::Create { payload, .. }, Some((metadata, current_payload)))
                if current_payload.as_ref() == payload =>
            {
                let expected = UpdateVersion {
                    e_tag: metadata.e_tag.clone(),
                    version: metadata.version.clone(),
                };
                self.object_store
                    .delete_if_matches(canonical_path, &expected)
                    .await?;
            }
            _ => {}
        }
        let terminal = self
            .terminal_reference_lifecycle(canonical_path, previous.as_ref())
            .await?;
        self.put_terminal_reference_lifecycle(canonical_path, revoking, &terminal, has_local_lock)
            .await
    }

    async fn reconcile_deleting_reference(
        &self,
        canonical_path: &Path,
        deleting: &ReferenceLifecycleSnapshot,
        has_local_lock: bool,
    ) -> Result<bool> {
        let ReferenceLifecycleState::Deleting {
            previous,
            expected_etag,
            expected_version,
            ..
        } = &deleting.state
        else {
            return Ok(false);
        };
        let expected = UpdateVersion {
            e_tag: expected_etag.clone(),
            version: expected_version.clone(),
        };
        self.object_store
            .delete_if_matches(canonical_path, &expected)
            .await?;
        let terminal = self
            .terminal_reference_lifecycle(canonical_path, previous.as_ref())
            .await?;
        let is_deleted = matches!(terminal, ReferenceLifecycleState::Vacant { .. });
        if !self
            .put_terminal_reference_lifecycle(canonical_path, deleting, &terminal, has_local_lock)
            .await?
        {
            return Err(Error::RefConflict {
                message: format!(
                    "reference {canonical_path} lifecycle changed during reconciliation"
                ),
            });
        }
        Ok(is_deleted)
    }

    async fn terminal_reference_lifecycle(
        &self,
        canonical_path: &Path,
        previous: Option<&ReferenceLiveState>,
    ) -> Result<ReferenceLifecycleState> {
        let payload = match self.object_store.inner.get(canonical_path).await {
            Ok(result) => Some(result.bytes().await?),
            Err(object_store::Error::NotFound { .. }) => None,
            Err(error) => return Err(error.into()),
        };
        let Some(payload) = payload else {
            return Ok(ReferenceLifecycleState::Vacant {
                canonical_path: canonical_path.to_string(),
            });
        };
        let generation = payload_reference_generation(&payload)?;
        if let Some(previous) = previous
            && generation.as_deref() == Some(previous.generation.as_str())
        {
            return Ok(ReferenceLifecycleState::Live {
                canonical_path: canonical_path.to_string(),
                live: previous.clone(),
            });
        }
        if generation.is_none() {
            return Ok(ReferenceLifecycleState::Legacy {
                canonical_path: canonical_path.to_string(),
            });
        }
        Err(Error::RefConflict {
            message: format!(
                "reference {canonical_path} canonical generation is not reconciled with lifecycle state"
            ),
        })
    }

    async fn rewrite_reference_payload(
        &self,
        path: &Path,
        metadata: &ObjectMeta,
        expected_payload: &[u8],
        replacement_payload: &[u8],
        has_local_lock: bool,
    ) -> Result<()> {
        let result = self
            .object_store
            .inner
            .put_opts(
                path,
                Bytes::copy_from_slice(replacement_payload).into(),
                PutOptions {
                    mode: PutMode::Update(UpdateVersion {
                        e_tag: metadata.e_tag.clone(),
                        version: metadata.version.clone(),
                    }),
                    ..Default::default()
                },
            )
            .await;
        match result {
            Ok(_) => Ok(()),
            Err(
                object_store::Error::NotSupported { .. }
                | object_store::Error::NotImplemented { .. },
            ) if has_local_lock => {
                let current = self.object_store.inner.get(path).await?.bytes().await?;
                if current.as_ref() != expected_payload {
                    return Err(Error::RefConflict {
                        message: format!("reference {path} changed while fencing an expired owner"),
                    });
                }
                self.object_store
                    .inner
                    .put(path, Bytes::copy_from_slice(replacement_payload).into())
                    .await?;
                Ok(())
            }
            Err(
                object_store::Error::Precondition { .. } | object_store::Error::NotFound { .. },
            ) => Err(Error::RefConflict {
                message: format!("reference {path} changed while fencing an expired owner"),
            }),
            Err(error) => Err(error.into()),
        }
    }

    async fn create_reference_intent(
        &self,
        version: u64,
        intent: &ReferenceIntent,
    ) -> Result<(Path, DateTime<Utc>)> {
        let path = self
            .reference_intents_path
            .clone()
            .join(version.to_string())
            .join(format!(
                "{}{}",
                Uuid::new_v4().simple(),
                REFERENCE_INTENT_SUFFIX
            ));
        self.object_store
            .inner
            .put_opts(
                &path,
                Bytes::from(serde_json::to_vec(intent)?).into(),
                PutOptions {
                    mode: PutMode::Create,
                    ..Default::default()
                },
            )
            .await?;
        // A failed HEAD is ambiguous after a successful create. Leave the
        // intent for bounded recovery instead of deleting a write that may be
        // protecting an in-flight canonical publication.
        let metadata = self.object_store.inner.head(&path).await?;
        Ok((path, metadata.last_modified))
    }

    async fn apply_reference_mutation_inner(
        &self,
        mutation: &ReferenceMutation,
        has_local_lock: bool,
        expected_operation_id: Option<&str>,
    ) -> Result<ReferenceMutationOutcome> {
        let pending_version = if let Some(expected_operation_id) = expected_operation_id {
            self.ensure_branch_incarnation_active().await?;
            let canonical_path = match mutation {
                ReferenceMutation::Create { path, .. } | ReferenceMutation::Update { path, .. } => {
                    Path::parse(path)?
                }
            };
            let lifecycle = self.reference_lifecycle_snapshot(&canonical_path).await?;
            match lifecycle.as_ref().map(|snapshot| &snapshot.state) {
                Some(ReferenceLifecycleState::Pending {
                    operation_id,
                    target,
                    ..
                }) if operation_id == expected_operation_id => Some(target.version),
                _ => return Ok(ReferenceMutationOutcome::Conflict),
            }
        } else {
            None
        };
        let (path, payload, result) = match mutation {
            ReferenceMutation::Create { path, payload } => {
                let path = Path::parse(path)?;
                let result = self
                    .object_store
                    .inner
                    .put_opts(
                        &path,
                        Bytes::from(payload.clone()).into(),
                        PutOptions {
                            mode: PutMode::Create,
                            ..Default::default()
                        },
                    )
                    .await;
                (path, payload, result)
            }
            ReferenceMutation::Update {
                path,
                expected_payload,
                expected_etag,
                expected_version,
                payload,
            } => {
                let path = Path::parse(path)?;
                let result = self
                    .object_store
                    .inner
                    .put_opts(
                        &path,
                        Bytes::from(payload.clone()).into(),
                        PutOptions {
                            mode: PutMode::Update(UpdateVersion {
                                e_tag: expected_etag.clone(),
                                version: expected_version.clone(),
                            }),
                            ..Default::default()
                        },
                    )
                    .await;
                let result = match result {
                    Err(
                        object_store::Error::NotSupported { .. }
                        | object_store::Error::NotImplemented { .. },
                    ) if has_local_lock => {
                        let current = self.object_store.inner.get(&path).await?.bytes().await?;
                        if current.as_ref() != expected_payload {
                            return Ok(ReferenceMutationOutcome::Conflict);
                        }
                        self.object_store
                            .inner
                            .put(&path, Bytes::from(payload.clone()).into())
                            .await
                    }
                    Err(
                        object_store::Error::NotSupported { .. }
                        | object_store::Error::NotImplemented { .. },
                    ) => {
                        return Err(Error::not_supported(format!(
                            "object store {} does not support atomic conditional reference updates for {path}",
                            self.object_store.scheme()
                        )));
                    }
                    result => result,
                };
                (path, payload, result)
            }
        };

        let outcome = match result {
            Ok(_) => ReferenceMutationOutcome::Published,
            Err(
                object_store::Error::AlreadyExists { .. }
                | object_store::Error::Precondition { .. },
            ) => {
                self.reference_mutation_conflict_outcome(&path, payload)
                    .await?
            }
            Err(object_store::Error::NotFound { .. }) => ReferenceMutationOutcome::Conflict,
            Err(error) => return Err(error.into()),
        };
        if outcome == ReferenceMutationOutcome::Published
            && let (Some(operation_id), Some(version)) = (expected_operation_id, pending_version)
        {
            self.finish_reference_mutation(mutation, operation_id, version, has_local_lock)
                .await
        } else {
            Ok(outcome)
        }
    }

    async fn finish_reference_mutation(
        &self,
        mutation: &ReferenceMutation,
        operation_id: &str,
        version: u64,
        has_local_lock: bool,
    ) -> Result<ReferenceMutationOutcome> {
        let canonical_path = match mutation {
            ReferenceMutation::Create { path, .. } | ReferenceMutation::Update { path, .. } => {
                Path::parse(path)?
            }
        };
        let branch_is_active = match self.ensure_branch_incarnation_active().await {
            Ok(()) => true,
            Err(Error::RefConflict { .. }) => false,
            Err(error) => return Err(error),
        };
        let lifecycle_completed = branch_is_active
            && self
                .complete_reference_lifecycle(
                    &canonical_path,
                    operation_id,
                    version,
                    has_local_lock,
                )
                .await?;
        if lifecycle_completed {
            return Ok(ReferenceMutationOutcome::Published);
        }
        let Some(revoking) = self
            .revoke_reference_lifecycle(&canonical_path, operation_id, mutation, has_local_lock)
            .await?
        else {
            return Ok(ReferenceMutationOutcome::Conflict);
        };
        self.reconcile_revoking_reference(&canonical_path, &revoking, has_local_lock)
            .await?;
        Ok(ReferenceMutationOutcome::Conflict)
    }

    async fn reference_mutation_conflict_outcome(
        &self,
        path: &Path,
        intended_payload: &[u8],
    ) -> Result<ReferenceMutationOutcome> {
        match self.object_store.inner.get(path).await {
            Ok(result) => {
                let current = result.bytes().await?;
                Ok(if current.as_ref() == intended_payload {
                    ReferenceMutationOutcome::Published
                } else {
                    ReferenceMutationOutcome::Conflict
                })
            }
            Err(object_store::Error::NotFound { .. }) => Ok(ReferenceMutationOutcome::Conflict),
            Err(error) => Err(error.into()),
        }
    }

    async fn reference_intent_metadata(&self) -> Result<Vec<ReferenceIntentMetadata>> {
        let metadata = self
            .object_store
            .list(Some(self.reference_intents_path.clone()))
            .try_collect::<Vec<_>>()
            .await?;
        stream::iter(metadata)
            .map(|metadata| async move {
                let result = match self.object_store.inner.get(&metadata.location).await {
                    Ok(result) => result,
                    Err(object_store::Error::NotFound { .. }) => return Ok(None),
                    Err(error) => return Err(Error::from(error)),
                };
                let bytes = result.bytes().await?;
                let intent = serde_json::from_slice(&bytes).map_err(|error| {
                    Error::corrupt_file(metadata.location.clone(), error.to_string())
                })?;
                self.parse_reference_intent_metadata(metadata, intent)
                    .map(Some)
            })
            .buffer_unordered(self.object_store.io_parallelism())
            .try_collect::<Vec<_>>()
            .await
            .map(|intents| intents.into_iter().flatten().collect())
    }

    fn parse_reference_intent_metadata(
        &self,
        metadata: ObjectMeta,
        intent: ReferenceIntent,
    ) -> Result<ReferenceIntentMetadata> {
        let relative_parts = metadata
            .location
            .prefix_match(&self.reference_intents_path)
            .ok_or_else(|| {
                Error::corrupt_file(
                    metadata.location.clone(),
                    "reference intent is outside its namespace",
                )
            })?
            .map(|part| part.as_ref().to_string())
            .collect::<Vec<_>>();
        if relative_parts.len() != 2 {
            return Err(Error::corrupt_file(
                metadata.location,
                "reference intent path must contain a version and operation filename",
            ));
        }
        let version = relative_parts[0].parse::<u64>().map_err(|error| {
            Error::corrupt_file(
                metadata.location.clone(),
                format!("reference intent has invalid version: {error}"),
            )
        })?;
        let operation_id = relative_parts[1]
            .strip_suffix(REFERENCE_INTENT_SUFFIX)
            .ok_or_else(|| {
                Error::corrupt_file(
                    metadata.location.clone(),
                    "reference intent filename has an invalid suffix",
                )
            })?;
        Uuid::parse_str(operation_id).map_err(|error| {
            Error::corrupt_file(
                metadata.location.clone(),
                format!("reference intent has invalid operation id: {error}"),
            )
        })?;
        Ok(ReferenceIntentMetadata {
            version,
            metadata,
            intent,
        })
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

    async fn has_precommit_sealed_marker(&self, version: u64) -> Result<bool> {
        Ok(self
            .version_marker_metadata(version)
            .await?
            .iter()
            .any(|marker| marker.state == RetirementState::Sealed))
    }

    async fn has_committed_marker(&self, version: u64) -> Result<bool> {
        Ok(self
            .version_marker_metadata(version)
            .await?
            .iter()
            .any(|marker| marker.state == RetirementState::Committed))
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
        let actively_referenced_versions = self.active_reference_versions().await?;
        let mut stale_drains = Vec::new();
        let mut sealed_manifest_paths = HashMap::<u64, HashSet<Path>>::new();
        let mut sealed_marker_paths = HashMap::<u64, Vec<Path>>::new();
        let mut committed_versions = HashSet::new();
        for marker in markers {
            if marker.state != RetirementState::Draining {
                let manifest_paths = self.read_retirement_marker(&marker.metadata).await?;
                sealed_manifest_paths
                    .entry(marker.version)
                    .or_default()
                    .extend(manifest_paths);
                if marker.state == RetirementState::Committed {
                    committed_versions.insert(marker.version);
                } else {
                    sealed_marker_paths
                        .entry(marker.version)
                        .or_default()
                        .push(marker.metadata.location);
                }
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
        let mut cancelled_seals = Vec::new();
        for (version, manifest_paths) in sealed_manifest_paths {
            let manifest_paths = manifest_paths.into_iter().collect::<Vec<_>>();
            let mut has_existing_manifest = false;
            for path in &manifest_paths {
                has_existing_manifest |= self.object_store.exists(path).await?;
            }
            if has_existing_manifest {
                let is_committed = committed_versions.contains(&version);
                let is_retained = actively_leased_versions.contains(&version)
                    || actively_referenced_versions.contains(&version);
                if is_committed || !is_retained {
                    versions_to_resume.insert(version);
                } else {
                    // Recovery may cancel a pre-commit seal when a lease or
                    // durable reference won the final census. Removing the
                    // seal restores renewal; committed retirement remains
                    // irreversible and is always resumed above.
                    cancelled_seals
                        .extend(sealed_marker_paths.remove(&version).unwrap_or_default());
                }
            } else {
                terminal_manifests.insert(version, manifest_paths);
            }
        }
        self.delete_paths(cancelled_seals).await?;
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
    /// durable-reference scans. A final intent scan keeps any publication that
    /// crossed the caller's census on the cancellable side of the boundary.
    pub(super) async fn commit_versions(
        &mut self,
        versions: &HashSet<u64>,
        completed_intents_observed_before_census: &HashSet<Path>,
        lifecycle_generations_observed_before_census: &HashMap<Path, String>,
    ) -> Result<HashSet<u64>> {
        // This scan runs inside the commit operation, after the caller's final
        // reference census. An admission that won that narrow race remains
        // cancellable and cannot cross the irreversible committed boundary.
        let mut active_reference_versions = self
            .store
            .reference_versions(
                CompletedReferenceIntentHandling::RetainForCurrentScan,
                completed_intents_observed_before_census,
            )
            .await?
            .versions;
        active_reference_versions.extend(
            self.store
                .lifecycle_reference_versions_since(lifecycle_generations_observed_before_census)
                .await?,
        );
        let retained_versions = versions
            .intersection(&active_reference_versions)
            .copied()
            .collect::<HashSet<_>>();
        let versions = versions
            .difference(&retained_versions)
            .copied()
            .collect::<HashSet<_>>();
        let mut marker_paths = Vec::with_capacity(versions.len());
        for version in &versions {
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
            for version in &versions {
                if let Some(fence) = self.fences.get_mut(version) {
                    fence.sealed_path = None;
                }
            }
        }
        Ok(retained_versions)
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
        let mut intent_paths = Vec::new();
        for intent in self.reference_intent_metadata().await? {
            if versions.contains(&intent.version) {
                intent_paths.push(intent.metadata.location);
            }
        }
        // Leases and superseded marker states are dependent metadata. Keep at
        // least one terminal marker as the retry anchor until they are gone.
        self.delete_paths(lease_paths).await?;
        self.delete_paths(intent_paths).await?;
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

/// Begin a durable tag or branch admission before canonical publication.
pub(super) async fn begin_reference_admission(
    refs: &Refs,
    branch: Option<&str>,
    version: u64,
    mut mutation: ReferenceMutation,
) -> Result<ReferenceAdmission> {
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
    ensure_reference_available(&store, &manifest.path, version).await?;
    store.ensure_branch_incarnation_active().await?;
    let operation_id = Uuid::new_v4().simple().to_string();
    let (canonical_path, previous_was_legacy) = match &mut mutation {
        ReferenceMutation::Create { path, payload } => {
            if !store.object_store.supports_conditional_delete() {
                return Err(Error::not_supported(format!(
                    "object store {} cannot safely roll back reference creation for {path} because atomic conditional delete is unavailable",
                    store.object_store.scheme()
                )));
            }
            *payload = set_payload_reference_generation(payload, &operation_id)?;
            (Path::parse(path)?, false)
        }
        ReferenceMutation::Update {
            path,
            expected_payload,
            payload,
            ..
        } => {
            let previous_was_legacy = payload_reference_generation(expected_payload)?.is_none();
            *payload = set_payload_reference_generation(payload, &operation_id)?;
            (Path::parse(path)?, previous_was_legacy)
        }
    };
    let intent = ReferenceIntent {
        manifest_path: manifest.path.to_string(),
        mutation: mutation.clone(),
        operation_id: operation_id.clone(),
        state: ReferenceIntentState::Pending,
    };
    let (path, created_at) = store.create_reference_intent(version, &intent).await?;
    if let Err(error) = store
        .claim_reference_lifecycle(&canonical_path, &operation_id, version, previous_was_legacy)
        .await
    {
        let claim_is_definitively_absent =
            match store.reference_lifecycle_snapshot(&canonical_path).await {
                Ok(Some(ReferenceLifecycleSnapshot {
                    state:
                        ReferenceLifecycleState::Pending {
                            operation_id: current_operation,
                            ..
                        }
                        | ReferenceLifecycleState::Revoking {
                            operation_id: current_operation,
                            ..
                        },
                    ..
                })) => current_operation != operation_id,
                Ok(_) => true,
                Err(_) => false,
            };
        if claim_is_definitively_absent {
            let _ = store.object_store.delete(&path).await;
        }
        return Err(error);
    }
    let admission = ReferenceAdmission {
        store,
        path,
        manifest_path: manifest.path,
        version,
        created_at,
        operation_id,
        mutation,
    };
    if let Err(error) = admission.ensure_owned().await {
        admission.cancel_before_publish().await;
        return Err(error);
    }
    Ok(admission)
}

impl ReferenceAdmission {
    /// Recheck ownership immediately before the canonical conditional write.
    pub(super) async fn ensure_owned(&self) -> Result<()> {
        let metadata = self.store.object_store.inner.head(&self.path).await?;
        let observed_at = self.store.storage_observed_at().await?;
        if metadata.last_modified != self.created_at
            || !reference_owner_is_active(metadata.last_modified, observed_at)?
        {
            return Err(Error::RefConflict {
                message: format!(
                    "reference admission for version {} no longer owns its publication intent",
                    self.version
                ),
            });
        }
        ensure_reference_available(&self.store, &self.manifest_path, self.version).await?;
        self.store.ensure_branch_incarnation_active().await?;
        let canonical_path = self.canonical_path()?;
        let lifecycle = self
            .store
            .reference_lifecycle_snapshot(&canonical_path)
            .await?;
        if !matches!(
            lifecycle.as_ref().map(|snapshot| &snapshot.state),
            Some(ReferenceLifecycleState::Pending {
                operation_id,
                ..
            }) if operation_id == &self.operation_id
        ) {
            return Err(Error::RefConflict {
                message: format!(
                    "reference admission for version {} lost its lifecycle generation",
                    self.version
                ),
            });
        }
        Ok(())
    }

    fn canonical_path(&self) -> Result<Path> {
        match &self.mutation {
            ReferenceMutation::Create { path, .. } | ReferenceMutation::Update { path, .. } => {
                Ok(Path::parse(path)?)
            }
        }
    }

    pub(super) async fn publish(self, conflict_message: String) -> Result<()> {
        let canonical_path = self.canonical_path()?;
        let local_lock = lock_local_reference(&self.store.object_store, &canonical_path).await?;
        if let Err(error) = self.ensure_owned().await {
            self.cancel_before_publish_inner(local_lock.is_some()).await;
            return Err(error);
        }
        match self
            .store
            .apply_reference_mutation_inner(
                &self.mutation,
                local_lock.is_some(),
                Some(&self.operation_id),
            )
            .await
        {
            Ok(ReferenceMutationOutcome::Published) => {
                self.complete().await;
                Ok(())
            }
            Ok(ReferenceMutationOutcome::Conflict) => {
                self.cancel_before_publish_inner(local_lock.is_some()).await;
                Err(Error::RefConflict {
                    message: conflict_message,
                })
            }
            // A transport failure can be ambiguous after the server accepted
            // the conditional mutation. Recovery retains the intent and
            // verifies canonical state without replaying the operation.
            Err(error) => Err(error),
        }
    }

    /// Remove this operation's uniquely owned intent after a definitive result
    /// that did not publish the canonical reference.
    pub(super) async fn cancel_before_publish(&self) {
        let local_lock = match self.canonical_path() {
            Ok(canonical_path) => {
                match lock_local_reference(&self.store.object_store, &canonical_path).await {
                    Ok(local_lock) => local_lock,
                    Err(error) => {
                        tracing::warn!(
                            path = %canonical_path,
                            error = %error,
                            "Failed to lock cancelled reference admission"
                        );
                        None
                    }
                }
            }
            Err(_) => None,
        };
        self.cancel_before_publish_inner(local_lock.is_some()).await;
    }

    async fn cancel_before_publish_inner(&self, has_local_lock: bool) {
        let Ok(canonical_path) = self.canonical_path() else {
            return;
        };
        if let Err(error) = self
            .store
            .cancel_reference_lifecycle(&canonical_path, &self.operation_id, has_local_lock)
            .await
        {
            tracing::warn!(
                path = %canonical_path,
                error = %error,
                "Failed to cancel reference lifecycle admission"
            );
            return;
        }
        let matching_pending_is_absent = match self
            .store
            .reference_lifecycle_snapshot(&canonical_path)
            .await
        {
            Ok(Some(ReferenceLifecycleSnapshot {
                state:
                    ReferenceLifecycleState::Pending {
                        operation_id: current_operation,
                        ..
                    },
                ..
            })) => current_operation != self.operation_id,
            Ok(_) => true,
            Err(error) => {
                tracing::warn!(
                    path = %canonical_path,
                    error = %error,
                    "Failed to confirm cancelled reference lifecycle admission"
                );
                false
            }
        };
        if matching_pending_is_absent
            && let Err(error) = self.store.object_store.delete(&self.path).await
            && !error.is_not_found()
        {
            tracing::warn!(
                path = %self.path,
                error = %error,
                "Failed to remove cancelled reference admission intent"
            );
        }
    }

    /// Mark the intent as the durable handoff to the canonical reference.
    /// Cleanup verifies the canonical payload, retains it for the current scan,
    /// and removes the completed intent without waiting for the ownership timeout.
    async fn complete(&self) {
        if let Err(error) = self.store.object_store.delete(&self.path).await
            && !error.is_not_found()
        {
            // The live lifecycle generation is already the durable handoff.
            tracing::warn!(
                path = %self.path,
                error = %error,
                "Failed to remove completed reference admission intent"
            );
        }
    }
}

pub(super) async fn lock_local_reference(
    object_store: &ObjectStore,
    path: &Path,
) -> Result<Option<LocalReferenceLock>> {
    if !object_store.is_local() && object_store.scheme() != "file-object-store" {
        return Ok(None);
    }

    let path = PathBuf::from(path.as_ref());
    let path = if path.is_absolute() {
        path
    } else {
        PathBuf::from(std::path::MAIN_SEPARATOR_STR).join(path)
    };
    let lock_path = PathBuf::from(format!("{}.lance-reference.lock", path.display()));
    tokio::task::spawn_blocking(move || -> Result<LocalReferenceLock> {
        if let Some(parent) = lock_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(lock_path)?;
        file.lock()?;
        Ok(LocalReferenceLock { _file: file })
    })
    .await
    .map_err(|error| Error::internal(format!("failed to acquire local reference lock: {error}")))?
    .map(Some)
}

fn stable_reference_path_id(path: &str) -> String {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    fn hash(path: &[u8], seed: u64) -> u64 {
        path.iter().fold(seed, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
        })
    }

    format!(
        "{:016x}{:016x}",
        hash(path.as_bytes(), FNV_OFFSET),
        hash(path.as_bytes(), FNV_OFFSET ^ 0x9e3779b97f4a7c15)
    )
}

fn payload_reference_generation(payload: &[u8]) -> Result<Option<String>> {
    let value: serde_json::Value = serde_json::from_slice(payload)?;
    Ok(value
        .as_object()
        .and_then(|object| object.get(REFERENCE_GENERATION_FIELD))
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned))
}

fn set_payload_reference_generation(payload: &[u8], generation: &str) -> Result<Vec<u8>> {
    let mut value: serde_json::Value = serde_json::from_slice(payload)?;
    let object = value.as_object_mut().ok_or_else(|| {
        Error::internal("canonical reference payload must be a JSON object".to_string())
    })?;
    object.insert(
        REFERENCE_GENERATION_FIELD.to_string(),
        serde_json::Value::String(generation.to_string()),
    );
    Ok(serde_json::to_vec_pretty(&value)?)
}

fn fenced_reference_payload(payload: &[u8]) -> Vec<u8> {
    let mut fenced = Vec::with_capacity(payload.len() + 1);
    fenced.extend_from_slice(payload);
    fenced.push(b'\n');
    fenced
}

pub(super) async fn canonical_reference_is_visible(
    object_store: Arc<ObjectStore>,
    root_path: &Path,
    canonical_path: &Path,
    payload: &[u8],
) -> Result<bool> {
    let generation = payload_reference_generation(payload)?;
    // Released writers do not preserve fields they do not know about. Their
    // rewrites remain authoritative canonical references and must stay visible
    // while the sidecar is reconciled by a later current-client mutation.
    if generation.is_none() {
        return Ok(true);
    }
    let store = VersionLeaseStore::lifecycle_only(object_store, root_path.clone());
    let Some(snapshot) = store.reference_lifecycle_snapshot(canonical_path).await? else {
        return Ok(false);
    };
    Ok(match snapshot.state {
        ReferenceLifecycleState::Live { live, .. } => {
            generation.as_deref() == Some(live.generation.as_str())
        }
        ReferenceLifecycleState::Legacy { .. } => false,
        ReferenceLifecycleState::Vacant { .. } => false,
        ReferenceLifecycleState::Pending { previous, .. } => match previous {
            Some(previous) => generation.as_deref() == Some(previous.generation.as_str()),
            None => false,
        },
        ReferenceLifecycleState::Revoking { previous, .. }
        | ReferenceLifecycleState::Deleting { previous, .. } => previous
            .is_some_and(|previous| generation.as_deref() == Some(previous.generation.as_str())),
    })
}

pub(super) async fn delete_canonical_reference(
    object_store: Arc<ObjectStore>,
    root_path: &Path,
    canonical_path: &Path,
    expected_metadata: &ObjectMeta,
    expected_payload: &[u8],
) -> Result<()> {
    if !object_store.supports_conditional_delete()
        || (expected_metadata.e_tag.is_none() && expected_metadata.version.is_none())
    {
        return Err(Error::not_supported(format!(
            "object store {} cannot safely delete reference {canonical_path} because atomic conditional delete is unavailable",
            object_store.scheme()
        )));
    }
    let store = VersionLeaseStore::lifecycle_only(Arc::clone(&object_store), root_path.clone());
    let local_lock = lock_local_reference(&object_store, canonical_path).await?;
    let current = object_store
        .inner
        .get(canonical_path)
        .await
        .map_err(|error| {
            if matches!(error, object_store::Error::NotFound { .. }) {
                Error::RefConflict {
                    message: format!(
                        "reference {canonical_path} changed during conditional deletion"
                    ),
                }
            } else {
                error.into()
            }
        })?;
    let current_metadata = current.meta.clone();
    let current_payload = current.bytes().await?;
    if current_payload.as_ref() != expected_payload
        || current_metadata.e_tag != expected_metadata.e_tag
        || current_metadata.version != expected_metadata.version
    {
        return Err(Error::RefConflict {
            message: format!("reference {canonical_path} changed during conditional deletion"),
        });
    }

    let lifecycle = store.reference_lifecycle_snapshot(canonical_path).await?;
    let canonical_generation = payload_reference_generation(&current_payload)?;
    match lifecycle.as_ref().map(|snapshot| &snapshot.state) {
        Some(
            ReferenceLifecycleState::Pending { .. } | ReferenceLifecycleState::Revoking { .. },
        ) => {
            return Err(Error::RefConflict {
                message: format!("reference {canonical_path} has an in-flight mutation"),
            });
        }
        Some(ReferenceLifecycleState::Live { live, .. })
            if canonical_generation.as_deref().is_some()
                && canonical_generation.as_deref() != Some(live.generation.as_str()) =>
        {
            return Err(Error::RefConflict {
                message: format!("reference {canonical_path} lifecycle changed during deletion"),
            });
        }
        Some(ReferenceLifecycleState::Legacy { .. }) if canonical_generation.is_some() => {
            return Err(Error::RefConflict {
                message: format!("reference {canonical_path} lifecycle changed during deletion"),
            });
        }
        Some(ReferenceLifecycleState::Vacant { .. }) if canonical_generation.is_some() => {
            return Err(Error::RefConflict {
                message: format!("reference {canonical_path} lifecycle changed during deletion"),
            });
        }
        _ => {}
    }

    let is_retry = lifecycle
        .as_ref()
        .is_some_and(|snapshot| matches!(snapshot.state, ReferenceLifecycleState::Deleting { .. }));
    let deleting_snapshot = if is_retry {
        let Some(snapshot) = lifecycle else {
            return Err(Error::internal(format!(
                "reference {canonical_path} lost its deleting lifecycle state"
            )));
        };
        snapshot
    } else {
        let previous = lifecycle
            .as_ref()
            .and_then(|snapshot| match &snapshot.state {
                ReferenceLifecycleState::Live { live, .. } => Some(live.clone()),
                _ => None,
            });
        let deleting = ReferenceLifecycleState::Deleting {
            canonical_path: canonical_path.to_string(),
            operation_id: Uuid::new_v4().simple().to_string(),
            previous,
            previous_was_legacy: canonical_generation.is_none(),
            expected_payload: current_payload.to_vec(),
            expected_etag: current_metadata.e_tag,
            expected_version: current_metadata.version,
        };
        let Some(snapshot) = store
            .put_reference_lifecycle(
                canonical_path,
                lifecycle.as_ref(),
                &deleting,
                local_lock.is_some(),
            )
            .await?
        else {
            return Err(Error::RefConflict {
                message: format!("reference {canonical_path} lifecycle changed during deletion"),
            });
        };
        snapshot
    };
    if !store
        .reconcile_deleting_reference(canonical_path, &deleting_snapshot, local_lock.is_some())
        .await?
    {
        return Err(Error::RefConflict {
            message: format!("reference {canonical_path} changed during conditional deletion"),
        });
    }
    Ok(())
}

async fn ensure_reference_available(
    store: &VersionLeaseStore,
    manifest_path: &Path,
    version: u64,
) -> Result<()> {
    if store.has_active_retirement_marker(version).await? {
        return Err(Error::RefConflict {
            message: format!(
                "version {version} is retiring and cannot accept a new durable reference"
            ),
        });
    }
    if !store.object_store.exists(manifest_path).await? {
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
    begin_branch_state_removal(object_store, root_path, namespace).await?;
    let lifecycle_store =
        VersionLeaseStore::lifecycle_only(Arc::new(object_store.clone()), root_path.clone());
    let intent_prefix = root_path
        .clone()
        .join(REFERENCE_INTENTS_DIR)
        .join(namespace.to_string());
    let intent_metadata = object_store
        .list(Some(intent_prefix))
        .try_collect::<Vec<_>>()
        .await?;
    for metadata in &intent_metadata {
        let result = match object_store.inner.get(&metadata.location).await {
            Ok(result) => result,
            Err(object_store::Error::NotFound { .. }) => continue,
            Err(error) => return Err(error.into()),
        };
        let intent: ReferenceIntent = serde_json::from_slice(&result.bytes().await?)
            .map_err(|error| Error::corrupt_file(metadata.location.clone(), error.to_string()))?;
        if intent.operation_id.is_empty() {
            continue;
        }
        let canonical_path = match &intent.mutation {
            ReferenceMutation::Create { path, .. } | ReferenceMutation::Update { path, .. } => {
                Path::parse(path)?
            }
        };
        let local_lock = lock_local_reference(object_store, &canonical_path).await?;
        if let Some(revoking) = lifecycle_store
            .revoke_reference_lifecycle(
                &canonical_path,
                &intent.operation_id,
                &intent.mutation,
                local_lock.is_some(),
            )
            .await?
        {
            lifecycle_store
                .reconcile_revoking_reference(&canonical_path, &revoking, local_lock.is_some())
                .await?;
        }
    }
    for path in [
        root_path
            .clone()
            .join(LEASES_DIR)
            .join(namespace.to_string()),
        root_path
            .clone()
            .join(LEASE_GC_MARKERS_DIR)
            .join(namespace.to_string()),
        root_path
            .clone()
            .join(REFERENCE_INTENTS_DIR)
            .join(namespace.to_string()),
    ] {
        match object_store.remove_dir_all(path).await {
            Ok(()) => {}
            Err(error) if error.is_not_found() => {}
            Err(error) => return Err(error),
        }
    }
    cancel_branch_state_removal(object_store, root_path, namespace).await
}

pub(super) async fn begin_branch_state_removal(
    object_store: &ObjectStore,
    root_path: &Path,
    namespace: &str,
) -> Result<()> {
    let path = root_path
        .clone()
        .join(BRANCH_TERMINATIONS_DIR)
        .join(format!("{namespace}.deleted"));
    match object_store
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
        Ok(_) | Err(object_store::Error::AlreadyExists { .. }) => Ok(()),
        Err(error) => Err(error.into()),
    }
}

pub(super) async fn cancel_branch_state_removal(
    object_store: &ObjectStore,
    root_path: &Path,
    namespace: &str,
) -> Result<()> {
    let path = root_path
        .clone()
        .join(BRANCH_TERMINATIONS_DIR)
        .join(format!("{namespace}.deleted"));
    match object_store.delete(&path).await {
        Ok(()) => Ok(()),
        Err(error) if error.is_not_found() => Ok(()),
        Err(error) => Err(error),
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

fn reference_owner_is_active(
    intent_observed_at: DateTime<Utc>,
    current_observed_at: DateTime<Utc>,
) -> Result<bool> {
    Ok(expiration_from_ttl(intent_observed_at, REFERENCE_ADMISSION_TIMEOUT)? > current_observed_at)
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
    use crate::dataset::refs::{BranchContents, TagContents};
    use crate::utils::test::FailingProxyStore;
    use lance_io::object_store::WrappingObjectStore;
    use mock_instant::thread_local::MockClock;

    use super::*;

    fn memory_store() -> VersionLeaseStore {
        VersionLeaseStore {
            object_store: Arc::new(ObjectStore::memory()),
            root_path: Path::from(""),
            namespace: MAIN_BRANCH.to_string(),
            leases_path: Path::from("leases"),
            markers_path: Path::from("markers"),
            reference_intents_path: Path::from("reference_intents"),
            manifest_path: None,
            canonical_references: None,
        }
    }

    fn manifest_paths(version: u64) -> HashMap<u64, Vec<Path>> {
        HashMap::from([(
            version,
            vec![Path::from(format!("manifests/{version}.manifest"))],
        )])
    }

    fn reference_intent(version: u64) -> ReferenceIntent {
        ReferenceIntent {
            manifest_path: format!("manifests/{version}.manifest"),
            mutation: ReferenceMutation::Create {
                path: format!("tags/version-{version}.json"),
                payload: version.to_string().into_bytes(),
            },
            operation_id: String::new(),
            state: ReferenceIntentState::Pending,
        }
    }

    async fn create_test_reference_admission(
        store: &VersionLeaseStore,
        version: u64,
        manifest_path: &Path,
        canonical_path: &Path,
        payload: &[u8],
    ) -> ReferenceAdmission {
        let operation_id = Uuid::new_v4().simple().to_string();
        let mutation = ReferenceMutation::Create {
            path: canonical_path.to_string(),
            payload: set_payload_reference_generation(payload, &operation_id).unwrap(),
        };
        let intent = ReferenceIntent {
            manifest_path: manifest_path.to_string(),
            mutation: mutation.clone(),
            operation_id: operation_id.clone(),
            state: ReferenceIntentState::Pending,
        };
        let (path, created_at) = store
            .create_reference_intent(version, &intent)
            .await
            .unwrap();
        store
            .claim_reference_lifecycle(canonical_path, &operation_id, version, false)
            .await
            .unwrap();
        ReferenceAdmission {
            store: store.clone(),
            path,
            manifest_path: manifest_path.clone(),
            version,
            created_at,
            operation_id,
            mutation,
        }
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
    fn generated_reference_payloads_remain_readable_by_released_clients() {
        let tag = set_payload_reference_generation(
            br#"{"branch":null,"version":42,"manifestSize":0,"metadata":{}}"#,
            "generation",
        )
        .unwrap();
        let branch = set_payload_reference_generation(
            br#"{"parentBranch":null,"identifier":{"version_mapping":[]},"parentVersion":42,"createAt":0,"manifestSize":0,"metadata":{}}"#,
            "generation",
        )
        .unwrap();

        assert!(serde_json::from_slice::<TagContents>(&tag).is_ok());
        assert!(serde_json::from_slice::<BranchContents>(&branch).is_ok());
    }

    #[tokio::test]
    async fn released_client_rewrite_keeps_generated_references_visible() {
        let store = memory_store();
        let manifest_path = Path::from("manifests/42.manifest");
        store.object_store.put(&manifest_path, &[]).await.unwrap();

        let tag_path = Path::from("tags/released.json");
        create_test_reference_admission(
            &store,
            42,
            &manifest_path,
            &tag_path,
            br#"{"branch":null,"version":42,"manifestSize":0,"metadata":{}}"#,
        )
        .await
        .publish("conflict".to_string())
        .await
        .unwrap();
        let generated_tag = store
            .object_store
            .inner
            .get(&tag_path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let released_tag = serde_json::to_vec_pretty(
            &serde_json::from_slice::<TagContents>(&generated_tag).unwrap(),
        )
        .unwrap();
        assert!(
            payload_reference_generation(&released_tag)
                .unwrap()
                .is_none()
        );
        store
            .object_store
            .put(&tag_path, &released_tag)
            .await
            .unwrap();
        assert!(
            canonical_reference_is_visible(
                Arc::clone(&store.object_store),
                &store.root_path,
                &tag_path,
                &released_tag,
            )
            .await
            .unwrap()
        );

        let branch_path = Path::from("branches/released.json");
        create_test_reference_admission(
            &store,
            42,
            &manifest_path,
            &branch_path,
            br#"{"parentBranch":null,"identifier":{"version_mapping":[]},"parentVersion":42,"createAt":0,"manifestSize":0,"metadata":{}}"#,
        )
        .await
        .publish("conflict".to_string())
        .await
        .unwrap();
        let generated_branch = store
            .object_store
            .inner
            .get(&branch_path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let released_branch = serde_json::to_vec_pretty(
            &serde_json::from_slice::<BranchContents>(&generated_branch).unwrap(),
        )
        .unwrap();
        assert!(
            payload_reference_generation(&released_branch)
                .unwrap()
                .is_none()
        );
        store
            .object_store
            .put(&branch_path, &released_branch)
            .await
            .unwrap();
        assert!(
            canonical_reference_is_visible(
                Arc::clone(&store.object_store),
                &store.root_path,
                &branch_path,
                &released_branch,
            )
            .await
            .unwrap()
        );
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
        assert!(
            guard
                .commit_versions(&HashSet::from([42]), &HashSet::new(), &HashMap::new())
                .await
                .unwrap()
                .is_empty()
        );
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
    async fn reference_intent_blocks_retirement_commit() {
        let store = memory_store();
        let (intent_path, _) = store
            .create_reference_intent(42, &reference_intent(42))
            .await
            .unwrap();
        let mut guard = store.fence_versions(&manifest_paths(42)).await.unwrap();
        guard.seal_versions(&HashSet::from([42])).await.unwrap();

        let retained_versions = guard
            .commit_versions(&HashSet::from([42]), &HashSet::new(), &HashMap::new())
            .await
            .unwrap();

        assert_eq!(retained_versions, HashSet::from([42]));
        assert!(guard.fences[&42].committed_path.is_none());
        assert!(guard.fences[&42].sealed_path.is_some());
        store.object_store.delete(&intent_path).await.unwrap();
        guard.cancel_all().await.unwrap();
    }

    #[tokio::test]
    async fn terminal_marker_survives_lease_deletion_failure() {
        let failing_store = Arc::new(FailingProxyStore::new());
        let mut object_store = ObjectStore::memory();
        object_store.inner = failing_store.wrap("memory", Arc::clone(&object_store.inner));
        let store = VersionLeaseStore {
            object_store: Arc::new(object_store),
            root_path: Path::from(""),
            namespace: MAIN_BRANCH.to_string(),
            leases_path: Path::from("leases"),
            markers_path: Path::from("markers"),
            reference_intents_path: Path::from("reference_intents"),
            manifest_path: None,
            canonical_references: None,
        };
        let lease = store.acquire(42, Duration::from_secs(60)).await.unwrap();
        let manifests = manifest_paths(42);
        let mut guard = store.fence_versions(&manifests).await.unwrap();
        guard.seal_versions(&HashSet::from([42])).await.unwrap();
        assert!(
            guard
                .commit_versions(&HashSet::from([42]), &HashSet::new(), &HashMap::new())
                .await
                .unwrap()
                .is_empty()
        );
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
        let mut lease = store.acquire(42, Duration::from_secs(60)).await.unwrap();
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
        assert!(store.version_marker_metadata(42).await.unwrap().is_empty());
        lease.renew(Duration::from_secs(60)).await.unwrap();

        lease.release().await.unwrap();
        assert!(
            store
                .clone()
                .recover_retirements()
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn sealed_recovery_cancels_for_active_reference() {
        let store = memory_store();
        let manifest_paths = manifest_paths(42);
        let manifest_path = manifest_paths[&42][0].clone();
        store.object_store.put(&manifest_path, &[]).await.unwrap();
        let (intent_path, _) = store
            .create_reference_intent(42, &reference_intent(42))
            .await
            .unwrap();
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
        assert!(store.version_marker_metadata(42).await.unwrap().is_empty());

        store.object_store.delete(&intent_path).await.unwrap();
    }

    #[tokio::test]
    async fn canonical_reference_handoff_blocks_retirement_commit() {
        let store = memory_store();
        let manifest_path = Path::from("manifests/42.manifest");
        let canonical_path = Path::from("tags/racing.json");
        store.object_store.put(&manifest_path, &[]).await.unwrap();
        let admission = create_test_reference_admission(
            &store,
            42,
            &manifest_path,
            &canonical_path,
            br#"{"version":42}"#,
        )
        .await;
        let intent_path = admission.path.clone();
        admission.ensure_owned().await.unwrap();

        let manifests = HashMap::from([(42, vec![manifest_path])]);
        let mut guard = store.fence_versions(&manifests).await.unwrap();
        guard.seal_versions(&HashSet::from([42])).await.unwrap();
        assert_eq!(
            store
                .apply_reference_mutation_inner(
                    &admission.mutation,
                    false,
                    Some(&admission.operation_id),
                )
                .await
                .unwrap(),
            ReferenceMutationOutcome::Published
        );
        admission
            .store
            .object_store
            .delete(&admission.path)
            .await
            .unwrap();

        assert_eq!(
            guard
                .commit_versions(&HashSet::from([42]), &HashSet::new(), &HashMap::new())
                .await
                .unwrap(),
            HashSet::from([42])
        );
        assert!(!store.object_store.exists(&intent_path).await.unwrap());
        guard.cancel_all().await.unwrap();
    }

    #[tokio::test]
    async fn completed_intent_does_not_retain_deleted_reference() {
        let store = memory_store();
        let manifest_path = Path::from("manifests/42.manifest");
        let canonical_path = Path::from("tags/removed.json");
        store.object_store.put(&manifest_path, &[]).await.unwrap();
        let admission = create_test_reference_admission(
            &store,
            42,
            &manifest_path,
            &canonical_path,
            br#"{"version":42}"#,
        )
        .await;
        let intent_path = admission.path.clone();
        admission.publish("conflict".to_string()).await.unwrap();
        let result = store.object_store.inner.get(&canonical_path).await.unwrap();
        let metadata = result.meta.clone();
        let payload = result.bytes().await.unwrap();
        delete_canonical_reference(
            Arc::clone(&store.object_store),
            &store.root_path,
            &canonical_path,
            &metadata,
            &payload,
        )
        .await
        .unwrap();

        assert!(store.active_reference_versions().await.unwrap().is_empty());
        assert!(!store.object_store.exists(&intent_path).await.unwrap());
    }

    #[tokio::test]
    async fn completed_intent_defers_to_following_canonical_census() {
        let store = memory_store();
        let manifest_path = Path::from("manifests/42.manifest");
        let canonical_path = Path::from("branches/child.json");
        store.object_store.put(&manifest_path, &[]).await.unwrap();
        let admission = create_test_reference_admission(
            &store,
            42,
            &manifest_path,
            &canonical_path,
            br#"{"parentVersion":42}"#,
        )
        .await;
        let intent_path = admission.path.clone();
        admission.publish("conflict".to_string()).await.unwrap();

        let census = store
            .reference_versions_before_canonical_census()
            .await
            .unwrap();
        assert!(census.versions.is_empty());
        assert!(census.completed_intent_paths.is_empty());
        assert!(store.object_store.exists(&canonical_path).await.unwrap());
        assert!(!store.object_store.exists(&intent_path).await.unwrap());
    }

    #[tokio::test]
    async fn conditional_reference_update_does_not_resurrect_deleted_reference() {
        let store = memory_store();
        let canonical_path = Path::from("branches/child.json");
        let original = store
            .object_store
            .inner
            .put(&canonical_path, Bytes::from_static(b"original").into())
            .await
            .unwrap();
        let mutation = ReferenceMutation::Update {
            path: canonical_path.to_string(),
            expected_payload: b"original".to_vec(),
            expected_etag: original.e_tag,
            expected_version: original.version,
            payload: b"updated".to_vec(),
        };
        store.object_store.delete(&canonical_path).await.unwrap();

        assert_eq!(
            store
                .apply_reference_mutation_inner(&mutation, false, None)
                .await
                .unwrap(),
            ReferenceMutationOutcome::Conflict
        );
        assert!(!store.object_store.exists(&canonical_path).await.unwrap());
    }

    #[tokio::test]
    async fn expired_pending_create_intent_does_not_resurrect_deleted_reference() {
        let store = memory_store();
        let manifest_path = Path::from("manifests/42.manifest");
        let canonical_path = Path::from("tags/deleted.json");
        store.object_store.put(&manifest_path, &[]).await.unwrap();
        let admission = create_test_reference_admission(
            &store,
            42,
            &manifest_path,
            &canonical_path,
            br#"{"version":42}"#,
        )
        .await;
        assert_eq!(
            store
                .apply_reference_mutation_inner(&admission.mutation, false, None,)
                .await
                .unwrap(),
            ReferenceMutationOutcome::Published
        );
        store.object_store.delete(&canonical_path).await.unwrap();

        assert_eq!(
            store
                .expired_reference_mutation_outcome(
                    &admission.mutation,
                    &admission.operation_id,
                    42,
                    false,
                )
                .await
                .unwrap(),
            ReferenceMutationOutcome::Conflict
        );
        assert_eq!(
            store
                .apply_reference_mutation_inner(
                    &admission.mutation,
                    false,
                    Some(&admission.operation_id),
                )
                .await
                .unwrap(),
            ReferenceMutationOutcome::Conflict
        );
        assert!(!store.object_store.exists(&canonical_path).await.unwrap());
        assert!(matches!(
            store
                .reference_lifecycle_snapshot(&canonical_path)
                .await
                .unwrap()
                .map(|snapshot| snapshot.state),
            Some(ReferenceLifecycleState::Revoking { .. })
        ));
    }

    #[tokio::test]
    async fn expired_update_owner_cannot_publish_after_retirement_commit() {
        let store = memory_store();
        let canonical_path = Path::from("tags/updated.json");
        let original_payload = br#"{"version":1}"#;
        let original = store
            .object_store
            .inner
            .put(&canonical_path, Bytes::from_static(original_payload).into())
            .await
            .unwrap();
        let operation_id = Uuid::new_v4().simple().to_string();
        let mutation = ReferenceMutation::Update {
            path: canonical_path.to_string(),
            expected_payload: original_payload.to_vec(),
            expected_etag: original.e_tag,
            expected_version: original.version,
            payload: set_payload_reference_generation(br#"{"version":42}"#, &operation_id).unwrap(),
        };
        store
            .claim_reference_lifecycle(&canonical_path, &operation_id, 42, true)
            .await
            .unwrap();

        assert_eq!(
            store
                .expired_reference_mutation_outcome(&mutation, &operation_id, 42, false)
                .await
                .unwrap(),
            ReferenceMutationOutcome::Conflict
        );
        assert_eq!(
            store
                .apply_reference_mutation_inner(&mutation, false, Some(&operation_id))
                .await
                .unwrap(),
            ReferenceMutationOutcome::Conflict
        );
        let fenced_payload = store
            .object_store
            .inner
            .get(&canonical_path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_ne!(fenced_payload.as_ref(), original_payload);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&fenced_payload).unwrap(),
            serde_json::from_slice::<serde_json::Value>(original_payload).unwrap()
        );
    }

    #[tokio::test]
    async fn unsupported_conditional_reference_update_is_rejected() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", temp_dir.path().display());
        let (object_store, base_path) = ObjectStore::from_uri(&uri).await.unwrap();
        let store = VersionLeaseStore {
            object_store,
            root_path: base_path.clone(),
            namespace: MAIN_BRANCH.to_string(),
            leases_path: base_path.clone().join("leases"),
            markers_path: base_path.clone().join("markers"),
            reference_intents_path: base_path.clone().join("reference_intents"),
            manifest_path: None,
            canonical_references: None,
        };
        let canonical_path = base_path.join("branches/child.json");
        let original = store
            .object_store
            .inner
            .put(&canonical_path, Bytes::from_static(b"original").into())
            .await
            .unwrap();
        let mutation = ReferenceMutation::Update {
            path: canonical_path.to_string(),
            expected_payload: b"original".to_vec(),
            expected_etag: original.e_tag,
            expected_version: original.version,
            payload: b"updated".to_vec(),
        };

        let error = store
            .apply_reference_mutation_inner(&mutation, false, None)
            .await
            .unwrap_err();

        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("atomic conditional"), "{error}");
        assert_eq!(
            store
                .object_store
                .inner
                .get(&canonical_path)
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap()
                .as_ref(),
            b"original"
        );
    }

    #[tokio::test]
    async fn canonical_only_recovery_cancels_seal_after_completed_handoff() {
        let store = memory_store();
        let manifest_path = Path::from("manifests/42.manifest");
        let canonical_path = Path::from("tags/recovery.json");
        store.object_store.put(&manifest_path, &[]).await.unwrap();
        let admission = create_test_reference_admission(
            &store,
            42,
            &manifest_path,
            &canonical_path,
            br#"{"version":42}"#,
        )
        .await;
        let intent_path = admission.path.clone();
        admission.publish("conflict".to_string()).await.unwrap();
        assert!(!store.object_store.exists(&intent_path).await.unwrap());
        let pre_seal_census = store
            .reference_versions_before_canonical_census()
            .await
            .unwrap();
        assert!(pre_seal_census.versions.is_empty());
        assert!(pre_seal_census.completed_intent_paths.is_empty());

        let manifests = HashMap::from([(42, vec![manifest_path])]);
        let mut guard = store.fence_versions(&manifests).await.unwrap();
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
        assert!(store.version_marker_metadata(42).await.unwrap().is_empty());
        store.acquire(42, Duration::from_secs(60)).await.unwrap();
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

    #[tokio::test]
    async fn deleting_parent_branch_state_fences_admitted_child_publication() {
        let root = Path::from("dataset");
        let namespace = "branch-id";
        let store = VersionLeaseStore {
            object_store: Arc::new(ObjectStore::memory()),
            root_path: root.clone(),
            namespace: namespace.to_string(),
            leases_path: root.clone().join(LEASES_DIR).join(namespace),
            markers_path: root.clone().join(LEASE_GC_MARKERS_DIR).join(namespace),
            reference_intents_path: root.clone().join(REFERENCE_INTENTS_DIR).join(namespace),
            manifest_path: None,
            canonical_references: None,
        };
        let manifest_path = Path::from("dataset/branches/parent/versions/42.manifest");
        let canonical_path = Path::from("dataset/_refs/branches/child.json");
        store.object_store.put(&manifest_path, &[]).await.unwrap();
        let admission = create_test_reference_admission(
            &store,
            42,
            &manifest_path,
            &canonical_path,
            br#"{"parentVersion":42}"#,
        )
        .await;
        admission.ensure_owned().await.unwrap();

        remove_branch_state(&store.object_store, &root, namespace)
            .await
            .unwrap();

        assert_eq!(
            store
                .apply_reference_mutation_inner(
                    &admission.mutation,
                    false,
                    Some(&admission.operation_id),
                )
                .await
                .unwrap(),
            ReferenceMutationOutcome::Conflict
        );
        assert!(!store.object_store.exists(&canonical_path).await.unwrap());
    }

    #[tokio::test]
    async fn checked_child_create_is_rolled_back_after_parent_state_removal() {
        let root = Path::from("dataset");
        let namespace = "branch-id";
        let store = VersionLeaseStore {
            object_store: Arc::new(ObjectStore::memory()),
            root_path: root.clone(),
            namespace: namespace.to_string(),
            leases_path: root.clone().join(LEASES_DIR).join(namespace),
            markers_path: root.clone().join(LEASE_GC_MARKERS_DIR).join(namespace),
            reference_intents_path: root.clone().join(REFERENCE_INTENTS_DIR).join(namespace),
            manifest_path: None,
            canonical_references: None,
        };
        let manifest_path = Path::from("dataset/branches/parent/versions/42.manifest");
        let canonical_path = Path::from("dataset/_refs/branches/child.json");
        store.object_store.put(&manifest_path, &[]).await.unwrap();
        let admission = create_test_reference_admission(
            &store,
            42,
            &manifest_path,
            &canonical_path,
            br#"{"parentVersion":42}"#,
        )
        .await;
        admission.ensure_owned().await.unwrap();

        remove_branch_state(&store.object_store, &root, namespace)
            .await
            .unwrap();
        assert_eq!(
            store
                .apply_reference_mutation_inner(&admission.mutation, false, None)
                .await
                .unwrap(),
            ReferenceMutationOutcome::Published
        );
        assert_eq!(
            store
                .finish_reference_mutation(&admission.mutation, &admission.operation_id, 42, false,)
                .await
                .unwrap(),
            ReferenceMutationOutcome::Conflict
        );
        assert!(!store.object_store.exists(&canonical_path).await.unwrap());
    }

    #[tokio::test]
    async fn released_client_delete_does_not_leave_stale_lifecycle_retention() {
        let store = memory_store();
        let manifest_path = Path::from("manifests/42.manifest");
        let canonical_path = Path::from("tags/released-delete.json");
        store.object_store.put(&manifest_path, &[]).await.unwrap();
        create_test_reference_admission(
            &store,
            42,
            &manifest_path,
            &canonical_path,
            br#"{"version":42}"#,
        )
        .await
        .publish("conflict".to_string())
        .await
        .unwrap();
        assert_eq!(
            store.active_reference_versions().await.unwrap(),
            HashSet::from([42])
        );

        store.object_store.delete(&canonical_path).await.unwrap();

        assert!(store.active_reference_versions().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn expired_update_completion_cannot_win_after_canonical_rollback() {
        let store = memory_store();
        let canonical_path = Path::from("tags/expired-update.json");
        let original_payload = br#"{"version":1}"#;
        let original = store
            .object_store
            .inner
            .put(&canonical_path, Bytes::from_static(original_payload).into())
            .await
            .unwrap();
        let operation_id = Uuid::new_v4().simple().to_string();
        let mutation = ReferenceMutation::Update {
            path: canonical_path.to_string(),
            expected_payload: original_payload.to_vec(),
            expected_etag: original.e_tag,
            expected_version: original.version,
            payload: set_payload_reference_generation(br#"{"version":42}"#, &operation_id).unwrap(),
        };
        store
            .claim_reference_lifecycle(&canonical_path, &operation_id, 42, true)
            .await
            .unwrap();
        assert_eq!(
            store
                .apply_reference_mutation_inner(&mutation, false, None)
                .await
                .unwrap(),
            ReferenceMutationOutcome::Published
        );
        let stale_pending = store
            .reference_lifecycle_snapshot(&canonical_path)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(
            store
                .expired_reference_mutation_outcome(&mutation, &operation_id, 42, false)
                .await
                .unwrap(),
            ReferenceMutationOutcome::Conflict
        );
        let stale_live = ReferenceLifecycleState::Live {
            canonical_path: canonical_path.to_string(),
            live: ReferenceLiveState {
                generation: operation_id,
                target: ReferenceTarget {
                    namespace: MAIN_BRANCH.to_string(),
                    version: 42,
                },
            },
        };
        assert!(
            store
                .put_reference_lifecycle(&canonical_path, Some(&stale_pending), &stale_live, false,)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn rollback_create_failure_keeps_recoverable_revocation() {
        let failing_store = Arc::new(FailingProxyStore::new());
        let mut object_store = ObjectStore::memory();
        object_store.inner = failing_store.wrap("memory", Arc::clone(&object_store.inner));
        let root = Path::from("dataset");
        let namespace = "branch-id";
        let store = VersionLeaseStore {
            object_store: Arc::new(object_store),
            root_path: root.clone(),
            namespace: namespace.to_string(),
            leases_path: root.clone().join(LEASES_DIR).join(namespace),
            markers_path: root.clone().join(LEASE_GC_MARKERS_DIR).join(namespace),
            reference_intents_path: root.clone().join(REFERENCE_INTENTS_DIR).join(namespace),
            manifest_path: None,
            canonical_references: None,
        };
        let manifest_path = Path::from("dataset/branches/parent/versions/42.manifest");
        let canonical_path = Path::from("dataset/_refs/branches/rollback-crash.json");
        store.object_store.put(&manifest_path, &[]).await.unwrap();
        let admission = create_test_reference_admission(
            &store,
            42,
            &manifest_path,
            &canonical_path,
            br#"{"parentVersion":42}"#,
        )
        .await;
        assert_eq!(
            store
                .apply_reference_mutation_inner(&admission.mutation, false, None)
                .await
                .unwrap(),
            ReferenceMutationOutcome::Published
        );
        failing_store.fail_when(
            "put",
            "rollback-crash.json",
            "injected conditional rollback delete failure",
        );

        store
            .expired_reference_mutation_outcome(
                &admission.mutation,
                &admission.operation_id,
                42,
                false,
            )
            .await
            .unwrap_err();
        assert!(matches!(
            store
                .reference_lifecycle_snapshot(&canonical_path)
                .await
                .unwrap()
                .map(|snapshot| snapshot.state),
            Some(ReferenceLifecycleState::Revoking { .. })
        ));
        assert!(store.object_store.exists(&admission.path).await.unwrap());

        failing_store.clear_fail_when("put", "rollback-crash.json");
        assert_eq!(
            store
                .expired_reference_mutation_outcome(
                    &admission.mutation,
                    &admission.operation_id,
                    42,
                    false,
                )
                .await
                .unwrap(),
            ReferenceMutationOutcome::Conflict
        );
        assert!(!store.object_store.exists(&canonical_path).await.unwrap());
    }

    #[tokio::test]
    async fn ambiguous_lifecycle_readback_keeps_recovery_intent() {
        let failing_store = Arc::new(FailingProxyStore::new());
        let mut object_store = ObjectStore::memory();
        object_store.inner = failing_store.wrap("memory", Arc::clone(&object_store.inner));
        let store = VersionLeaseStore {
            object_store: Arc::new(object_store),
            root_path: Path::from(""),
            namespace: MAIN_BRANCH.to_string(),
            leases_path: Path::from("leases"),
            markers_path: Path::from("markers"),
            reference_intents_path: Path::from("reference_intents"),
            manifest_path: None,
            canonical_references: None,
        };
        let canonical_path = Path::from("tags/ambiguous-state.json");
        let operation_id = Uuid::new_v4().simple().to_string();
        let intent = ReferenceIntent {
            manifest_path: "manifests/42.manifest".to_string(),
            mutation: ReferenceMutation::Create {
                path: canonical_path.to_string(),
                payload: br#"{"version":42}"#.to_vec(),
            },
            operation_id: operation_id.clone(),
            state: ReferenceIntentState::Pending,
        };
        let (intent_path, _) = store.create_reference_intent(42, &intent).await.unwrap();
        failing_store.fail_after_n(
            "get_opts",
            "version_reference_states",
            1,
            "injected lifecycle readback failure",
        );

        store
            .claim_reference_lifecycle(&canonical_path, &operation_id, 42, false)
            .await
            .unwrap_err();

        assert!(store.object_store.exists(&intent_path).await.unwrap());
    }

    #[tokio::test]
    async fn cancelled_admission_lifecycle_failure_keeps_recovery_anchor() {
        let failing_store = Arc::new(FailingProxyStore::new());
        let mut object_store = ObjectStore::memory();
        object_store.inner = failing_store.wrap("memory", Arc::clone(&object_store.inner));
        let store = VersionLeaseStore {
            object_store: Arc::new(object_store),
            root_path: Path::from(""),
            namespace: MAIN_BRANCH.to_string(),
            leases_path: Path::from("leases"),
            markers_path: Path::from("markers"),
            reference_intents_path: Path::from("reference_intents"),
            manifest_path: None,
            canonical_references: None,
        };
        let manifest_path = Path::from("manifests/42.manifest");
        let canonical_path = Path::from("tags/cancel-failure.json");
        store.object_store.put(&manifest_path, &[]).await.unwrap();
        let admission = create_test_reference_admission(
            &store,
            42,
            &manifest_path,
            &canonical_path,
            br#"{"version":42}"#,
        )
        .await;
        failing_store.fail_when(
            "put",
            "version_reference_states",
            "injected lifecycle cancellation failure",
        );

        admission.cancel_before_publish().await;

        assert!(store.object_store.exists(&admission.path).await.unwrap());
        assert!(matches!(
            store
                .reference_lifecycle_snapshot(&canonical_path)
                .await
                .unwrap()
                .map(|snapshot| snapshot.state),
            Some(ReferenceLifecycleState::Pending { operation_id, .. })
                if operation_id == admission.operation_id
        ));
        failing_store.clear_fail_when("put", "version_reference_states");
        admission.cancel_before_publish().await;
        assert!(!store.object_store.exists(&admission.path).await.unwrap());
        assert!(
            store
                .reference_lifecycle_snapshot(&canonical_path)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn conditional_delete_is_absent_to_released_tag_clients() {
        let store = memory_store();
        let manifest_path = Path::from("manifests/42.manifest");
        let canonical_path = Path::from("tags/released-delete.json");
        let released_payload = br#"{"branch":null,"version":42,"manifestSize":0,"metadata":{}}"#;
        store.object_store.put(&manifest_path, &[]).await.unwrap();
        create_test_reference_admission(
            &store,
            42,
            &manifest_path,
            &canonical_path,
            released_payload,
        )
        .await
        .publish("conflict".to_string())
        .await
        .unwrap();
        let result = store.object_store.inner.get(&canonical_path).await.unwrap();
        let metadata = result.meta.clone();
        let payload = result.bytes().await.unwrap();
        serde_json::from_slice::<TagContents>(&payload).unwrap();

        delete_canonical_reference(
            Arc::clone(&store.object_store),
            &store.root_path,
            &canonical_path,
            &metadata,
            &payload,
        )
        .await
        .unwrap();

        assert!(!store.object_store.exists(&canonical_path).await.unwrap());
        assert!(
            !store
                .object_store
                .read_dir(Path::from("tags"))
                .await
                .unwrap()
                .iter()
                .any(|name| name == "released-delete.json")
        );
        store
            .object_store
            .inner
            .put_opts(
                &canonical_path,
                Bytes::from_static(released_payload).into(),
                PutOptions {
                    mode: PutMode::Create,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn released_writer_update_during_delete_is_not_erased() {
        use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};

        let mut object_store = ObjectStore::memory();
        let underlying = Arc::clone(&object_store.inner);
        let policy = Arc::new(std::sync::Mutex::new(ProxyObjectStorePolicy::new()));
        object_store.inner = Arc::new(ProxyObjectStore::new(
            Arc::clone(&underlying),
            Arc::clone(&policy),
        ));
        let store = VersionLeaseStore {
            object_store: Arc::new(object_store),
            root_path: Path::from(""),
            namespace: MAIN_BRANCH.to_string(),
            leases_path: Path::from("leases"),
            markers_path: Path::from("markers"),
            reference_intents_path: Path::from("reference_intents"),
            manifest_path: None,
            canonical_references: None,
        };
        let manifest_path = Path::from("manifests/42.manifest");
        let canonical_path = Path::from("tags/mixed-client-delete.json");
        store.object_store.put(&manifest_path, &[]).await.unwrap();
        create_test_reference_admission(
            &store,
            42,
            &manifest_path,
            &canonical_path,
            br#"{"branch":null,"version":42,"manifestSize":0,"metadata":{}}"#,
        )
        .await
        .publish("conflict".to_string())
        .await
        .unwrap();
        let result = store.object_store.inner.get(&canonical_path).await.unwrap();
        let metadata = result.meta.clone();
        let payload = result.bytes().await.unwrap();

        let (put_entered_tx, put_entered_rx) = tokio::sync::oneshot::channel();
        let put_entered_tx = Arc::new(std::sync::Mutex::new(Some(put_entered_tx)));
        let (resume_put_tx, resume_put_rx) = std::sync::mpsc::channel();
        let resume_put_rx = Arc::new(std::sync::Mutex::new(resume_put_rx));
        let canonical_path_string = canonical_path.to_string();
        policy.lock().unwrap().set_before_policy(
            "pause_conditional_delete",
            Arc::new(move |method, path| {
                if method == "put"
                    && path.as_ref() == canonical_path_string
                    && let Some(sender) = put_entered_tx.lock().unwrap().take()
                {
                    sender.send(()).unwrap();
                    resume_put_rx.lock().unwrap().recv().unwrap();
                }
                Ok(())
            }),
        );

        let delete_store = Arc::clone(&store.object_store);
        let delete_root = store.root_path.clone();
        let delete_path = canonical_path.clone();
        let delete_task = tokio::spawn(async move {
            delete_canonical_reference(
                delete_store,
                &delete_root,
                &delete_path,
                &metadata,
                &payload,
            )
            .await
        });
        put_entered_rx.await.unwrap();

        let released_payload =
            Bytes::from_static(br#"{"branch":null,"version":99,"manifestSize":0,"metadata":{}}"#);
        underlying
            .put(&canonical_path, released_payload.clone().into())
            .await
            .unwrap();
        resume_put_tx.send(()).unwrap();
        assert!(matches!(
            delete_task.await.unwrap(),
            Err(Error::RefConflict { .. })
        ));

        let remaining = underlying
            .get(&canonical_path)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(remaining, released_payload);
    }

    #[tokio::test]
    async fn state_completion_failure_does_not_retain_deleted_reference_version() {
        let failing_store = Arc::new(FailingProxyStore::new());
        let mut object_store = ObjectStore::memory();
        object_store.inner = failing_store.wrap("memory", Arc::clone(&object_store.inner));
        let store = VersionLeaseStore {
            object_store: Arc::new(object_store),
            root_path: Path::from(""),
            namespace: MAIN_BRANCH.to_string(),
            leases_path: Path::from("leases"),
            markers_path: Path::from("markers"),
            reference_intents_path: Path::from("reference_intents"),
            manifest_path: None,
            canonical_references: None,
        };
        let manifest_path = Path::from("manifests/42.manifest");
        let canonical_path = Path::from("tags/state-completion-failure.json");
        store.object_store.put(&manifest_path, &[]).await.unwrap();
        create_test_reference_admission(
            &store,
            42,
            &manifest_path,
            &canonical_path,
            br#"{"branch":null,"version":42,"manifestSize":0,"metadata":{}}"#,
        )
        .await
        .publish("conflict".to_string())
        .await
        .unwrap();
        let result = store.object_store.inner.get(&canonical_path).await.unwrap();
        let metadata = result.meta.clone();
        let payload = result.bytes().await.unwrap();
        failing_store.fail_after_n(
            "put",
            "version_reference_states",
            1,
            "injected lifecycle completion failure",
        );

        delete_canonical_reference(
            Arc::clone(&store.object_store),
            &store.root_path,
            &canonical_path,
            &metadata,
            &payload,
        )
        .await
        .unwrap_err();

        assert!(!store.object_store.exists(&canonical_path).await.unwrap());
        let snapshot = store
            .reference_lifecycle_snapshot(&canonical_path)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            &snapshot.state,
            ReferenceLifecycleState::Deleting { .. }
        ));
        assert!(
            store
                .retained_lifecycle_target(&snapshot.state)
                .await
                .unwrap()
                .is_none(),
            "an absent canonical reference must not retain its deleted version"
        );
    }

    #[tokio::test]
    async fn deletion_failure_restores_reference_lifecycle() {
        let failing_store = Arc::new(FailingProxyStore::new());
        let mut object_store = ObjectStore::memory();
        object_store.inner = failing_store.wrap("memory", Arc::clone(&object_store.inner));
        let store = VersionLeaseStore {
            object_store: Arc::new(object_store),
            root_path: Path::from(""),
            namespace: MAIN_BRANCH.to_string(),
            leases_path: Path::from("leases"),
            markers_path: Path::from("markers"),
            reference_intents_path: Path::from("reference_intents"),
            manifest_path: None,
            canonical_references: None,
        };
        let manifest_path = Path::from("manifests/42.manifest");
        let canonical_path = Path::from("tags/deletion-failure.json");
        store.object_store.put(&manifest_path, &[]).await.unwrap();
        create_test_reference_admission(
            &store,
            42,
            &manifest_path,
            &canonical_path,
            br#"{"branch":null,"version":42,"manifestSize":0,"metadata":{}}"#,
        )
        .await
        .publish("conflict".to_string())
        .await
        .unwrap();
        let result = store.object_store.inner.get(&canonical_path).await.unwrap();
        let metadata = result.meta.clone();
        let payload = result.bytes().await.unwrap();
        failing_store.fail_when(
            "put",
            "deletion-failure.json",
            "injected conditional delete failure",
        );

        let error = delete_canonical_reference(
            Arc::clone(&store.object_store),
            &store.root_path,
            &canonical_path,
            &metadata,
            &payload,
        )
        .await
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("injected conditional delete failure"),
            "{error}"
        );
        failing_store.clear_fail_when("put", "deletion-failure.json");

        let remaining = store.object_store.inner.get(&canonical_path).await.unwrap();
        let remaining_metadata = remaining.meta.clone();
        let remaining_payload = remaining.bytes().await.unwrap();
        assert!(
            canonical_reference_is_visible(
                Arc::clone(&store.object_store),
                &store.root_path,
                &canonical_path,
                &remaining_payload,
            )
            .await
            .unwrap(),
            "a failed delete must not hide the still-present reference"
        );
        delete_canonical_reference(
            Arc::clone(&store.object_store),
            &store.root_path,
            &canonical_path,
            &remaining_metadata,
            &remaining_payload,
        )
        .await
        .unwrap();
        assert!(!store.object_store.exists(&canonical_path).await.unwrap());
        assert!(
            store
                .reference_lifecycle_snapshot(&canonical_path)
                .await
                .unwrap()
                .is_none()
        );
    }
}
