// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shard manifest storage with bit-reversed versioned naming.
//!
//! Shard manifests are stored as versioned protobuf files using bit-reversed
//! naming scheme to distribute files across object store keyspace.
//!
//! ## File Layout
//!
//! ```text
//! _mem_wal/{shard_id}/manifest/
//!   ├── {bit_reversed_version}.binpb  # Versioned manifest files
//!   └── version_hint.json             # Best-effort version hint
//! ```
//!
//! ## Write Protocol
//!
//! 1. Compute next version number
//! 2. Write manifest to `{bit_reversed_version}.binpb` using PUT-IF-NOT-EXISTS
//! 3. Best-effort update `version_hint.json` (failure is acceptable)
//!
//! ## Read Protocol
//!
//! 1. Read `version_hint.json` for starting version (default: 1 if not found)
//! 2. Use HEAD requests to check existence of subsequent versions
//! 3. Continue until a version is not found
//! 4. Return the last found version

use object_store::ObjectStoreExt;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use bytes::Bytes;
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use lance_core::{Error, Result};
use lance_index::mem_wal::{ShardManifest, ShardStatus};
use lance_io::object_store::ObjectStore;
use lance_table::format::pb;
use log::{info, warn};
use object_store::path::Path;
use prost::Message;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use uuid::Uuid;

use super::util::{manifest_filename, parse_bit_reversed_filename, shard_manifest_path};

/// Version hint file structure.
#[derive(Debug, Serialize, Deserialize)]
struct VersionHint {
    version: u64,
}

/// Store for reading and writing shard manifests.
///
/// Handles versioned manifest files with bit-reversed naming scheme
/// and PUT-IF-NOT-EXISTS atomicity.
#[derive(Debug)]
pub struct ShardManifestStore {
    object_store: Arc<ObjectStore>,
    shard_id: Uuid,
    manifest_dir: Path,
    manifest_scan_batch_size: usize,
    /// The last manifest this store durably wrote, served to [`Self::read_latest`].
    ///
    /// Populated *only* by [`Self::write`], never by a read. Authority comes
    /// from having written the value: `claim_epoch` plus the WAL fence sentinel
    /// make the epoch holder the only writer permitted to commit, so what we
    /// wrote is the latest until we lose the claim — which surfaces as a
    /// PUT-IF-NOT-EXISTS collision and clears this. A store that has never
    /// written has no such claim and always reads through, so a reader-only
    /// handle (a drop-reconcile probe, a WAL tailer) still observes the writer.
    ///
    /// Callers that must see a *peer's* commit even on the writer's own
    /// handle — [`Self::check_fenced`], [`Self::claim_epoch`] — use
    /// [`Self::read_latest_uncached`].
    latest: RwLock<Option<ShardManifest>>,
}

impl ShardManifestStore {
    /// Create a new manifest store for the given shard.
    ///
    /// # Arguments
    ///
    /// * `object_store` - Object store for reading/writing manifests
    /// * `base_path` - Base path within the object store (from ObjectStore::from_uri)
    /// * `shard_id` - Shard UUID
    /// * `manifest_scan_batch_size` - Batch size for parallel HEAD requests when scanning versions
    pub fn new(
        object_store: Arc<ObjectStore>,
        base_path: &Path,
        shard_id: Uuid,
        manifest_scan_batch_size: usize,
    ) -> Self {
        let manifest_dir = shard_manifest_path(base_path, &shard_id);
        Self {
            object_store,
            shard_id,
            manifest_dir,
            manifest_scan_batch_size,
            latest: RwLock::new(None),
        }
    }

    /// The cached manifest, if this store has written one.
    fn cached(&self) -> Option<ShardManifest> {
        self.latest.read().expect("manifest cache lock").clone()
    }

    /// Publish `manifest` as the latest. Only ever called after a durable write.
    fn cache(&self, manifest: &ShardManifest) {
        *self.latest.write().expect("manifest cache lock") = Some(manifest.clone());
    }

    /// Drop the cached manifest so the next read goes to storage. Called when a
    /// write collides, which is the one signal that another writer may have moved
    /// the shard past us.
    fn invalidate(&self) {
        *self.latest.write().expect("manifest cache lock") = None;
    }

    /// Read the latest manifest version, serving this store's own last commit
    /// from memory when it has one.
    ///
    /// Every scan of the version space costs `2 + manifest_scan_batch_size`
    /// object-store requests — the terminating batch of HEADs is all misses by
    /// construction — so the read path must not pay for it per request. A miss
    /// deliberately does *not* populate the cache: only a write earns the right
    /// to be cached (see `latest`). Callers that need to observe a *peer's*
    /// commit want [`Self::read_latest_uncached`].
    ///
    /// Returns `None` if no manifest exists (new shard).
    pub async fn read_latest(&self) -> Result<Option<ShardManifest>> {
        match self.cached() {
            Some(cached) => Ok(Some(cached)),
            None => self.read_latest_uncached().await,
        }
    }

    /// Read the latest manifest version from storage, ignoring the cache.
    ///
    /// Returns `None` if no manifest exists (new shard).
    #[instrument(name = "manifest_read_latest", level = "debug", skip_all, fields(shard_id = %self.shard_id))]
    pub async fn read_latest_uncached(&self) -> Result<Option<ShardManifest>> {
        let version = self.find_latest_version().await?;
        if version == 0 {
            return Ok(None);
        }

        self.read_version(version).await.map(Some)
    }

    /// Read a specific manifest version.
    pub async fn read_version(&self, version: u64) -> Result<ShardManifest> {
        let filename = manifest_filename(version);
        let path = self.manifest_dir.clone().join(filename.as_str());

        let data = self.object_store.inner.get(&path).await.map_err(|e| {
            Error::io(format!(
                "Failed to read manifest version {} for shard {}: {}",
                version, self.shard_id, e
            ))
        })?;

        let bytes = data
            .bytes()
            .await
            .map_err(|e| Error::io(format!("Failed to read manifest bytes: {}", e)))?;

        let pb_manifest = pb::ShardManifest::decode(bytes)
            .map_err(|e| Error::io(format!("Failed to decode manifest protobuf: {}", e)))?;

        ShardManifest::try_from(pb_manifest)
    }

    /// Write an initial manifest for a newly-created shard.
    ///
    /// `shard_field_values` maps field_id to raw Arrow scalar bytes.
    /// Initial manifests use writer epoch 0. A writer that claims the shard
    /// will write a new manifest with epoch 1 before appending WAL entries.
    pub async fn initialize_shard(
        &self,
        shard_spec_id: u32,
        shard_field_values: HashMap<String, Vec<u8>>,
    ) -> Result<ShardManifest> {
        let manifest = ShardManifest {
            shard_id: self.shard_id,
            version: 1,
            shard_spec_id,
            shard_field_values,
            writer_epoch: 0,
            replay_after_wal_entry_position: 0,
            wal_entry_position_last_seen: 0,
            current_generation: 1,
            sstables: vec![],
            status: ShardStatus::Active,
        };

        match self.write(&manifest).await {
            Ok(_) => Ok(manifest),
            Err(error) => match self.read_latest_uncached().await? {
                Some(existing)
                    if existing.shard_spec_id == manifest.shard_spec_id
                        && existing.shard_field_values == manifest.shard_field_values =>
                {
                    Ok(existing)
                }
                _ => Err(error),
            },
        }
    }

    /// Write a new manifest version atomically.
    ///
    /// Uses storage-appropriate strategy:
    /// - Local: Write to temp file + atomic rename for fencing
    /// - Cloud: PUT-IF-NOT-EXISTS (S3 conditional write)
    ///
    /// Returns the version that was written.
    ///
    /// # Errors
    ///
    /// Returns `Error::AlreadyExists` if another writer already wrote this version.
    #[instrument(name = "manifest_write", level = "debug", skip_all, fields(shard_id = %self.shard_id, version = manifest.version, epoch = manifest.writer_epoch))]
    pub async fn write(&self, manifest: &ShardManifest) -> Result<u64> {
        let version = manifest.version;
        let filename = manifest_filename(version);
        let path = self.manifest_dir.clone().join(filename.as_str());

        let pb_manifest = pb::ShardManifest::from(manifest);
        let bytes = pb_manifest.encode_to_vec();

        self.object_store
            .put_if_absent(&path, Bytes::from(bytes).into())
            .await
            .inspect_err(|_| self.invalidate())
            .map_err(|error| {
                if matches!(
                    error,
                    object_store::Error::AlreadyExists { .. }
                        | object_store::Error::Precondition { .. }
                ) {
                    Error::io(format!(
                        "Manifest version {} already exists for shard {}",
                        version, self.shard_id
                    ))
                } else {
                    Error::io(format!(
                        "Failed to write manifest version {} for shard {}: {}",
                        version, self.shard_id, error
                    ))
                }
            })?;

        // The write landed, so this is now the latest and every later
        // `read_latest` can be served from memory.
        self.cache(manifest);

        // Best-effort update version hint (failures are logged as warnings)
        self.write_version_hint(version).await;

        Ok(version)
    }

    /// Find the latest manifest version.
    ///
    /// Uses HEAD requests starting from version hint, scanning forward
    /// until a version is not found.
    async fn find_latest_version(&self) -> Result<u64> {
        // Start from version hint or 1
        let hint = self.read_version_hint().await.unwrap_or(1);

        // Scan forward from hint using HEAD requests
        let mut latest_found = 0u64;

        // First, check if hint version exists
        if hint > 0 && self.version_exists(hint).await? {
            latest_found = hint;
        } else if hint > 1 {
            // Hint might be stale, scan from beginning
            if self.version_exists(1).await? {
                latest_found = 1;
            }
        }

        // Parallel scan forward with batches of HEAD requests
        let batch_size = self.manifest_scan_batch_size;
        loop {
            let mut futures = FuturesUnordered::new();
            for offset in 0..batch_size {
                let version = latest_found + 1 + offset as u64;
                futures.push(async move { (version, self.version_exists(version).await) });
            }

            let mut found_any = false;
            while let Some((version, result)) = futures.next().await {
                if let Ok(true) = result
                    && version > latest_found
                {
                    latest_found = version;
                    found_any = true;
                }
            }

            if !found_any {
                break;
            }
        }

        Ok(latest_found)
    }

    /// Check if a manifest version exists using HEAD request.
    async fn version_exists(&self, version: u64) -> Result<bool> {
        let filename = manifest_filename(version);
        let path = self.manifest_dir.clone().join(filename.as_str());

        match self.object_store.inner.head(&path).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { .. }) => Ok(false),
            Err(e) => Err(Error::io(format!(
                "HEAD request failed for version {}: {}",
                version, e
            ))),
        }
    }

    /// Read the version hint file.
    async fn read_version_hint(&self) -> Option<u64> {
        let path = self.manifest_dir.clone().join("version_hint.json");

        let data = self.object_store.inner.get(&path).await.ok()?;
        let bytes = data.bytes().await.ok()?;
        let hint: VersionHint = serde_json::from_slice(&bytes).ok()?;

        Some(hint.version)
    }

    /// Write the version hint file (best-effort, failures logged but ignored).
    async fn write_version_hint(&self, version: u64) {
        let path = self.manifest_dir.clone().join("version_hint.json");
        let hint = VersionHint { version };

        match serde_json::to_vec(&hint) {
            Ok(bytes) => {
                if let Err(e) = self
                    .object_store
                    .inner
                    .put(&path, Bytes::from(bytes).into())
                    .await
                {
                    warn!(
                        "Failed to write version hint for shard {}: {}",
                        self.shard_id, e
                    );
                }
            }
            Err(e) => {
                warn!("Failed to serialize version hint: {}", e);
            }
        }
    }

    /// List all manifest versions (for garbage collection or debugging).
    pub async fn list_versions(&self) -> Result<Vec<u64>> {
        let mut versions = Vec::new();

        let list_result = self
            .object_store
            .inner
            .list(Some(&self.manifest_dir))
            .collect::<Vec<_>>()
            .await;

        for item in list_result {
            match item {
                Ok(meta) => {
                    if let Some(filename) = meta.location.filename()
                        && filename.ends_with(".binpb")
                        && let Some(version) = parse_bit_reversed_filename(filename)
                    {
                        versions.push(version);
                    }
                }
                Err(e) => {
                    warn!("Error listing manifest directory: {}", e);
                }
            }
        }

        versions.sort_unstable();
        Ok(versions)
    }

    /// Get the shard ID.
    pub fn shard_id(&self) -> Uuid {
        self.shard_id
    }

    // ========================================================================
    // Epoch-based Writer Fencing
    // ========================================================================

    /// Claim a shard by incrementing its writer epoch.
    ///
    /// This establishes single-writer semantics by:
    /// 1. Loading the current manifest (or creating initial state)
    /// 2. Incrementing the writer epoch
    /// 3. Atomically writing the new manifest
    ///
    /// On version conflict, re-reads the manifest and only retries when
    /// the latest writer_epoch is strictly less than the epoch we were
    /// targeting — meaning the version was bumped by something other than
    /// a real claim (a tailer cursor update or a concurrent
    /// `initialize_shard` writing epoch 0). If the latest writer_epoch
    /// is equal to or greater than our target, the target epoch is
    /// already claimed and this call fails. This preserves the
    /// no-epoch-war guarantee for real claimants while tolerating benign
    /// version bumps.
    ///
    /// # Returns
    ///
    /// A tuple of `(epoch, ShardManifest)` where the manifest is the
    /// claimed state (may be freshly created or loaded and epoch-bumped).
    ///
    /// # Errors
    ///
    /// Returns an error if another writer claimed an equal-or-higher
    /// epoch than our target, or if the manifest stays contended past
    /// the retry budget.
    #[instrument(name = "manifest_claim_epoch", level = "info", skip_all, fields(shard_id = %self.shard_id, shard_spec_id))]
    pub async fn claim_epoch(&self, shard_spec_id: u32) -> Result<(u64, ShardManifest)> {
        const MAX_CLAIM_RETRIES: usize = 16;
        let mut last_write_err: Option<Error> = None;
        for _ in 0..MAX_CLAIM_RETRIES {
            // Uncached throughout: a claim exists to discover another writer's
            // epoch, and on retry the cache would replay the version we just
            // lost the race on.
            let current = self.read_latest_uncached().await?;

            // A sealed shard is mid-drop (drop-table 2PC). Refuse the claim
            // with a distinguishable error rather than minting a new epoch,
            // so a caller that skips its own status check still cannot
            // resurrect a shard being dropped. Sophon's reconcile keys on
            // the "sealed" marker in this message to tell it apart from an
            // ordinary epoch fence.
            if let Some(m) = &current
                && m.status == ShardStatus::Sealed
            {
                return Err(Error::invalid_input(format!(
                    "shard {} is sealed; refusing claim (drop in flight)",
                    self.shard_id
                )));
            }

            let (next_version, next_epoch, base_manifest) = match current {
                Some(m) => (m.version + 1, m.writer_epoch + 1, Some(m)),
                None => (1, 1, None),
            };

            let new_manifest = if let Some(base) = base_manifest {
                ShardManifest {
                    version: next_version,
                    writer_epoch: next_epoch,
                    ..base
                }
            } else {
                ShardManifest {
                    shard_id: self.shard_id,
                    version: next_version,
                    shard_spec_id,
                    shard_field_values: HashMap::new(),
                    writer_epoch: next_epoch,
                    replay_after_wal_entry_position: 0,
                    wal_entry_position_last_seen: 0,
                    current_generation: 1,
                    sstables: vec![],
                    status: ShardStatus::Active,
                }
            };

            match self.write(&new_manifest).await {
                Ok(_) => {
                    info!(
                        "Claimed shard {} with epoch {} (version {})",
                        self.shard_id, next_epoch, next_version
                    );
                    return Ok((next_epoch, new_manifest));
                }
                Err(write_err) => {
                    let latest_epoch = self
                        .read_latest_uncached()
                        .await?
                        .map(|m| m.writer_epoch)
                        .unwrap_or(0);
                    if latest_epoch >= next_epoch {
                        return Err(Error::io(format!(
                            "Failed to claim shard {} (version {}): another writer claimed epoch {} (>= our target {}): {}",
                            self.shard_id, next_version, latest_epoch, next_epoch, write_err
                        )));
                    }
                    last_write_err = Some(write_err);
                }
            }
        }

        Err(Error::io(format!(
            "Failed to claim shard {} after {} retries due to manifest contention: {}",
            self.shard_id,
            MAX_CLAIM_RETRIES,
            last_write_err
                .map(|e| e.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        )))
    }

    /// Check if the given epoch has been fenced by a newer writer.
    ///
    /// Loads the current manifest and compares epochs. If the stored epoch
    /// is higher than the local epoch, the writer has been fenced.
    #[instrument(name = "manifest_check_fenced", level = "debug", skip_all, fields(shard_id = %self.shard_id, local_epoch))]
    pub async fn check_fenced(&self, local_epoch: u64) -> Result<()> {
        // Uncached on purpose: a fence is by definition another process's write,
        // which our own cache can never show us.
        let current = self.read_latest_uncached().await?;
        Self::check_fenced_against(&current, local_epoch, self.shard_id)
    }

    /// Check fencing against a pre-read manifest (avoids redundant read).
    fn check_fenced_against(
        manifest: &Option<ShardManifest>,
        local_epoch: u64,
        shard_id: Uuid,
    ) -> Result<()> {
        match manifest {
            Some(m) if m.writer_epoch > local_epoch => Err(Error::fenced_by_peer(format!(
                "local epoch {} < stored epoch {} for shard {}",
                local_epoch, m.writer_epoch, shard_id
            ))),
            _ => Ok(()),
        }
    }

    /// Update the manifest with retry on version conflict.
    ///
    /// This method:
    /// 1. Reads the latest manifest
    /// 2. Checks if fenced (fails immediately if so)
    /// 3. Calls `prepare_fn` to create the new manifest
    /// 4. Attempts to write
    /// 5. On version conflict, retries from step 1
    ///
    /// # Arguments
    ///
    /// * `local_epoch` - The writer's epoch (for fencing check)
    /// * `prepare_fn` - Function that takes current manifest and returns new manifest
    ///
    /// # Returns
    ///
    /// The successfully written manifest.
    #[instrument(name = "manifest_commit_update", level = "debug", skip_all, fields(shard_id = %self.shard_id, local_epoch))]
    pub async fn commit_update<F>(&self, local_epoch: u64, prepare_fn: F) -> Result<ShardManifest>
    where
        F: Fn(&ShardManifest) -> ShardManifest,
    {
        const MAX_RETRIES: usize = 10;

        for attempt in 0..MAX_RETRIES {
            // Step 1: Read latest
            let current = self
                .read_latest()
                .await?
                .ok_or_else(|| Error::io("Shard manifest not found"))?;

            // Step 2: Check fencing
            Self::check_fenced_against(&Some(current.clone()), local_epoch, self.shard_id)?;

            // Step 3: Prepare new manifest
            let new_manifest = prepare_fn(&current);

            // Validate epoch matches
            if new_manifest.writer_epoch != local_epoch {
                return Err(Error::invalid_input(format!(
                    "Manifest epoch {} doesn't match local epoch {}",
                    new_manifest.writer_epoch, local_epoch
                )));
            }

            // Step 4: Try to commit
            match self.write(&new_manifest).await {
                Ok(_) => {
                    return Ok(new_manifest);
                }
                Err(e) => {
                    // Check if it's a version conflict (can retry) vs other error
                    let is_version_conflict = e.to_string().contains("already exists");

                    if is_version_conflict && attempt < MAX_RETRIES - 1 {
                        continue;
                    }

                    return Err(e);
                }
            }
        }

        Err(Error::io(format!(
            "Failed to update manifest for shard {} after {} attempts",
            self.shard_id, MAX_RETRIES
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_local_store() -> (Arc<ObjectStore>, Path, TempDir) {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("file://{}", temp_dir.path().display());
        let (store, path) = ObjectStore::from_uri(&uri).await.unwrap();
        (store, path, temp_dir)
    }

    fn create_test_manifest(shard_id: Uuid, version: u64, epoch: u64) -> ShardManifest {
        ShardManifest {
            shard_id,
            version,
            shard_spec_id: 0,
            shard_field_values: HashMap::new(),
            writer_epoch: epoch,
            replay_after_wal_entry_position: 0,
            wal_entry_position_last_seen: 0,
            current_generation: 1,
            sstables: vec![],
            status: ShardStatus::Active,
        }
    }

    /// A peer's claim must be visible to `check_fenced` even though our own
    /// write left a manifest cached. This is the property the cache is allowed
    /// to break and must not.
    #[tokio::test]
    async fn check_fenced_sees_a_peer_through_a_warm_cache() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let incumbent = ShardManifestStore::new(store.clone(), &base_path, shard_id, 2);
        let successor = ShardManifestStore::new(store, &base_path, shard_id, 2);

        // Claiming writes a manifest, which is what warms the cache.
        let (epoch, _) = incumbent.claim_epoch(0).await.unwrap();
        assert!(incumbent.cached().is_some(), "the claim write must cache");

        successor.claim_epoch(0).await.unwrap();

        assert!(
            incumbent.check_fenced(epoch).await.is_err(),
            "a cached manifest must not hide a successor's epoch"
        );
    }

    /// A handle that has only ever read must not pin what it read. Only a write
    /// earns a cache entry — otherwise a reader (a WAL tailer polling for a
    /// cursor update, a drop-reconcile probe) would never observe the writer.
    #[tokio::test]
    async fn a_reader_only_store_never_caches() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let writer = ShardManifestStore::new(store.clone(), &base_path, shard_id, 2);
        let reader = ShardManifestStore::new(store, &base_path, shard_id, 2);

        let (epoch, _) = writer.claim_epoch(0).await.unwrap();
        assert_eq!(
            reader
                .read_latest()
                .await
                .unwrap()
                .unwrap()
                .current_generation,
            1
        );
        assert!(
            reader.cached().is_none(),
            "a read must not populate the cache"
        );

        writer
            .commit_update(epoch, |c| ShardManifest {
                version: c.version + 1,
                current_generation: 5,
                ..c.clone()
            })
            .await
            .unwrap();

        assert_eq!(
            reader
                .read_latest()
                .await
                .unwrap()
                .unwrap()
                .current_generation,
            5,
            "a reader must see the writer's later commits"
        );
    }

    /// A losing `commit_update` re-reads from storage rather than replaying the
    /// version it just lost on, so it converges instead of spinning.
    #[tokio::test]
    async fn commit_update_recovers_from_a_stale_cache() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let ours = ShardManifestStore::new(store.clone(), &base_path, shard_id, 2);
        let peer = ShardManifestStore::new(store, &base_path, shard_id, 2);

        let (epoch, _) = ours.claim_epoch(0).await.unwrap();
        ours.read_latest().await.unwrap().unwrap();

        // A same-epoch commit through another handle: our cache now points at a
        // version that is no longer the latest.
        peer.commit_update(epoch, |c| ShardManifest {
            version: c.version + 1,
            current_generation: 7,
            ..c.clone()
        })
        .await
        .unwrap();

        let updated = ours
            .commit_update(epoch, |c| ShardManifest {
                version: c.version + 1,
                wal_entry_position_last_seen: 42,
                ..c.clone()
            })
            .await
            .unwrap();

        // Built on the peer's version, not on the stale cached one.
        assert_eq!(updated.current_generation, 7);
        assert_eq!(updated.wal_entry_position_last_seen, 42);
        assert_eq!(
            ours.read_latest_uncached().await.unwrap().unwrap().version,
            updated.version
        );
    }

    /// The cached read and the storage read agree after a write.
    #[tokio::test]
    async fn read_latest_serves_the_written_manifest() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, shard_id, 2);

        let mut manifest = create_test_manifest(shard_id, 1, 1);
        manifest_store.write(&manifest).await.unwrap();
        manifest.version = 2;
        manifest.current_generation = 9;
        manifest_store.write(&manifest).await.unwrap();

        let cached = manifest_store.read_latest().await.unwrap().unwrap();
        let durable = manifest_store
            .read_latest_uncached()
            .await
            .unwrap()
            .unwrap();
        assert_eq!(cached.version, 2);
        assert_eq!(cached.current_generation, 9);
        assert_eq!(cached, durable);
    }

    #[tokio::test]
    async fn test_read_latest_empty() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, shard_id, 2);

        let result = manifest_store.read_latest().await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_write_and_read_manifest() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, shard_id, 2);

        let manifest = create_test_manifest(shard_id, 1, 1);
        manifest_store.write(&manifest).await.unwrap();

        let loaded = manifest_store.read_latest().await.unwrap().unwrap();
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.writer_epoch, 1);
        assert_eq!(loaded.shard_id, shard_id);
    }

    #[tokio::test]
    async fn test_multiple_versions() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, shard_id, 2);

        // Write multiple versions
        for version in 1..=5 {
            let manifest = create_test_manifest(shard_id, version, version);
            manifest_store.write(&manifest).await.unwrap();
        }

        // Should find latest
        let loaded = manifest_store.read_latest().await.unwrap().unwrap();
        assert_eq!(loaded.version, 5);
        assert_eq!(loaded.writer_epoch, 5);

        // List should return all versions
        let versions = manifest_store.list_versions().await.unwrap();
        assert_eq!(versions, vec![1, 2, 3, 4, 5]);
    }

    #[tokio::test]
    async fn test_read_specific_version() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, shard_id, 2);

        for version in 1..=3 {
            let manifest = create_test_manifest(shard_id, version, version * 10);
            manifest_store.write(&manifest).await.unwrap();
        }

        let v2 = manifest_store.read_version(2).await.unwrap();
        assert_eq!(v2.version, 2);
        assert_eq!(v2.writer_epoch, 20);
    }

    #[tokio::test]
    async fn test_put_if_not_exists() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, shard_id, 2);

        let manifest1 = create_test_manifest(shard_id, 1, 1);
        manifest_store.write(&manifest1).await.unwrap();

        // Second write to same version should fail
        let manifest2 = create_test_manifest(shard_id, 1, 2);
        let result = manifest_store.write(&manifest2).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_initialize_shard_writes_v1_with_epoch_zero() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, shard_id, 2);

        let mut field_values = HashMap::new();
        field_values.insert("user_bucket".to_string(), 7i32.to_le_bytes().to_vec());

        let manifest = manifest_store
            .initialize_shard(3, field_values.clone())
            .await
            .unwrap();
        assert_eq!(manifest.shard_id, shard_id);
        assert_eq!(manifest.version, 1);
        assert_eq!(manifest.writer_epoch, 0);
        assert_eq!(manifest.shard_spec_id, 3);
        assert_eq!(manifest.shard_field_values, field_values);

        let loaded = manifest_store.read_latest().await.unwrap().unwrap();
        assert_eq!(loaded, manifest);
    }

    #[tokio::test]
    async fn test_initialize_shard_idempotent_on_match() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, shard_id, 2);

        let mut field_values = HashMap::new();
        field_values.insert("k".to_string(), b"v".to_vec());

        let first = manifest_store
            .initialize_shard(1, field_values.clone())
            .await
            .unwrap();
        let second = manifest_store
            .initialize_shard(1, field_values)
            .await
            .unwrap();
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn test_claim_epoch_after_cursor_update() {
        // After a tailer cursor update bumps the manifest version without
        // claiming an epoch, the next claim_epoch should observe the new
        // state and produce the next epoch — this guards against treating
        // a cursor update as a real claimant.
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, shard_id, 2);

        let (first_epoch, first) = manifest_store.claim_epoch(0).await.unwrap();
        assert_eq!(first_epoch, 1);
        assert_eq!(first.version, 1);

        let mut cursor_update = first.clone();
        cursor_update.version += 1;
        cursor_update.wal_entry_position_last_seen = 42;
        manifest_store.write(&cursor_update).await.unwrap();

        let (second_epoch, second) = manifest_store.claim_epoch(0).await.unwrap();
        assert_eq!(second_epoch, 2);
        assert_eq!(second.version, 3);
        assert_eq!(second.wal_entry_position_last_seen, 42);
    }

    #[tokio::test]
    async fn test_claim_epoch_refuses_sealed_manifest() {
        // A `Sealed` manifest is the drop-table 2PC in-doubt marker:
        // `claim_epoch` must refuse it with a distinguishable error rather
        // than mint a new epoch, so a sealed shard can't be resurrected even
        // by a caller that skips its own status check. Rolling the status
        // back to `Active` makes the shard claimable again (reversible).
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, shard_id, 2);

        let (epoch, claimed) = manifest_store.claim_epoch(0).await.unwrap();
        assert_eq!(claimed.status, ShardStatus::Active);

        // Seal it (drop-table prepare).
        let sealed = ShardManifest {
            version: claimed.version + 1,
            status: ShardStatus::Sealed,
            ..claimed
        };
        manifest_store.write(&sealed).await.unwrap();

        // The claim is refused with the distinguishable "sealed" error —
        // and the manifest is left untouched (no new epoch minted).
        let err = manifest_store.claim_epoch(0).await.unwrap_err();
        assert!(
            err.to_string().contains("sealed"),
            "expected a distinguishable sealed-refusal error, got: {err}"
        );
        let after = manifest_store.read_latest().await.unwrap().unwrap();
        assert_eq!(after.writer_epoch, sealed.writer_epoch, "no epoch minted");
        assert_eq!(after.status, ShardStatus::Sealed);

        // Roll back to Active (drop-table abort) → re-claimable.
        let active = ShardManifest {
            version: sealed.version + 1,
            status: ShardStatus::Active,
            ..sealed
        };
        manifest_store.write(&active).await.unwrap();
        let (next_epoch, reclaimed) = manifest_store.claim_epoch(0).await.unwrap();
        assert!(next_epoch > epoch, "rolled-back shard mints the next epoch");
        assert_eq!(reclaimed.status, ShardStatus::Active);
    }

    #[tokio::test]
    async fn test_initialize_shard_rejects_conflict_with_mismatch() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let manifest_store = ShardManifestStore::new(store, &base_path, shard_id, 2);

        manifest_store
            .initialize_shard(1, HashMap::new())
            .await
            .unwrap();

        let mut other = HashMap::new();
        other.insert("k".to_string(), b"v".to_vec());
        let result = manifest_store.initialize_shard(1, other).await;
        assert!(
            result.is_err(),
            "second initialize_shard with different fields must fail"
        );
    }
}
