// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Region manifest storage with bit-reversed versioned naming.
//!
//! Region manifests are stored as versioned protobuf files using bit-reversed
//! naming scheme to distribute files across object store keyspace.
//!
//! ## File Layout
//!
//! ```text
//! _mem_wal/{region_id}/manifest/
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

use std::sync::Arc;

use bytes::Bytes;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use lance_core::{Error, Result};
use lance_index::mem_wal::RegionManifest;
use lance_io::object_store::ObjectStore;
use lance_table::format::pb;
use log::{info, warn};
use object_store::path::Path;
use object_store::PutMode;
use object_store::PutOptions;
use prost::Message;
use serde::{Deserialize, Serialize};
use snafu::location;
use uuid::Uuid;

use super::util::{manifest_filename, parse_bit_reversed_filename, region_manifest_path};

/// Version hint file structure.
#[derive(Debug, Serialize, Deserialize)]
struct VersionHint {
    version: u64,
}

/// Store for reading and writing region manifests.
///
/// Handles versioned manifest files with bit-reversed naming scheme
/// and PUT-IF-NOT-EXISTS atomicity.
#[derive(Debug)]
pub struct RegionManifestStore {
    object_store: Arc<ObjectStore>,
    region_id: Uuid,
    manifest_dir: Path,
    manifest_scan_batch_size: usize,
}

impl RegionManifestStore {
    /// Create a new manifest store for the given region.
    ///
    /// # Arguments
    ///
    /// * `object_store` - Object store for reading/writing manifests
    /// * `base_path` - Base path within the object store (from ObjectStore::from_uri)
    /// * `region_id` - Region UUID
    /// * `manifest_scan_batch_size` - Batch size for parallel HEAD requests when scanning versions
    pub fn new(
        object_store: Arc<ObjectStore>,
        base_path: &Path,
        region_id: Uuid,
        manifest_scan_batch_size: usize,
    ) -> Self {
        let manifest_dir = region_manifest_path(base_path, &region_id);
        Self {
            object_store,
            region_id,
            manifest_dir,
            manifest_scan_batch_size,
        }
    }

    /// Read the latest manifest version.
    ///
    /// Returns `None` if no manifest exists (new region).
    pub async fn read_latest(&self) -> Result<Option<RegionManifest>> {
        let version = self.find_latest_version().await?;
        if version == 0 {
            return Ok(None);
        }

        self.read_version(version).await.map(Some)
    }

    /// Read a specific manifest version.
    pub async fn read_version(&self, version: u64) -> Result<RegionManifest> {
        let filename = manifest_filename(version);
        let path = self.manifest_dir.child(filename.as_str());

        let data = self.object_store.inner.get(&path).await.map_err(|e| {
            Error::io(
                format!(
                    "Failed to read manifest version {} for region {}: {}",
                    version, self.region_id, e
                ),
                location!(),
            )
        })?;

        let bytes = data
            .bytes()
            .await
            .map_err(|e| Error::io(format!("Failed to read manifest bytes: {}", e), location!()))?;

        let pb_manifest = pb::RegionManifest::decode(bytes).map_err(|e| {
            Error::io(
                format!("Failed to decode manifest protobuf: {}", e),
                location!(),
            )
        })?;

        RegionManifest::try_from(pb_manifest)
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
    pub async fn write(&self, manifest: &RegionManifest) -> Result<u64> {
        let version = manifest.version;
        let filename = manifest_filename(version);
        let path = self.manifest_dir.child(filename.as_str());

        let pb_manifest = pb::RegionManifest::from(manifest);
        let bytes = pb_manifest.encode_to_vec();

        if self.object_store.is_local() {
            // Local storage: Use temp file + atomic rename for fencing
            let temp_filename = format!("{}.tmp.{}", filename, uuid::Uuid::new_v4());
            let temp_path = self.manifest_dir.child(temp_filename.as_str());

            // Write to temp file
            self.object_store
                .inner
                .put(&temp_path, Bytes::from(bytes).into())
                .await
                .map_err(|e| {
                    Error::io(format!("Failed to write temp manifest: {}", e), location!())
                })?;

            // Atomically rename to final path
            match self
                .object_store
                .inner
                .rename_if_not_exists(&temp_path, &path)
                .await
            {
                Ok(()) => {}
                Err(object_store::Error::AlreadyExists { .. }) => {
                    // Clean up temp file
                    let _ = self.object_store.delete(&temp_path).await;
                    return Err(Error::io(
                        format!(
                            "Manifest version {} already exists for region {}",
                            version, self.region_id
                        ),
                        location!(),
                    ));
                }
                Err(e) => {
                    // Clean up temp file
                    let _ = self.object_store.delete(&temp_path).await;
                    return Err(Error::io(
                        format!(
                            "Failed to write manifest version {} for region {}: {}",
                            version, self.region_id, e
                        ),
                        location!(),
                    ));
                }
            }
        } else {
            // Cloud storage: Use PUT-IF-NOT-EXISTS
            let put_opts = PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            };

            self.object_store
                .inner
                .put_opts(&path, Bytes::from(bytes).into(), put_opts)
                .await
                .map_err(|e| {
                    if matches!(e, object_store::Error::AlreadyExists { .. }) {
                        Error::io(
                            format!(
                                "Manifest version {} already exists for region {}",
                                version, self.region_id
                            ),
                            location!(),
                        )
                    } else {
                        Error::io(
                            format!(
                                "Failed to write manifest version {} for region {}: {}",
                                version, self.region_id, e
                            ),
                            location!(),
                        )
                    }
                })?;
        }

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
                if let Ok(true) = result {
                    if version > latest_found {
                        latest_found = version;
                        found_any = true;
                    }
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
        let path = self.manifest_dir.child(filename.as_str());

        match self.object_store.inner.head(&path).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { .. }) => Ok(false),
            Err(e) => Err(Error::io(
                format!("HEAD request failed for version {}: {}", version, e),
                location!(),
            )),
        }
    }

    /// Read the version hint file.
    async fn read_version_hint(&self) -> Option<u64> {
        let path = self.manifest_dir.child("version_hint.json");

        let data = self.object_store.inner.get(&path).await.ok()?;
        let bytes = data.bytes().await.ok()?;
        let hint: VersionHint = serde_json::from_slice(&bytes).ok()?;

        Some(hint.version)
    }

    /// Write the version hint file (best-effort, failures logged but ignored).
    async fn write_version_hint(&self, version: u64) {
        let path = self.manifest_dir.child("version_hint.json");
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
                        "Failed to write version hint for region {}: {}",
                        self.region_id, e
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
                    if let Some(filename) = meta.location.filename() {
                        if filename.ends_with(".binpb") {
                            if let Some(version) = parse_bit_reversed_filename(filename) {
                                versions.push(version);
                            }
                        }
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

    /// Get the region ID.
    pub fn region_id(&self) -> Uuid {
        self.region_id
    }

    // ========================================================================
    // Epoch-based Writer Fencing
    // ========================================================================

    /// Claim a region by incrementing its writer epoch.
    ///
    /// This establishes single-writer semantics by:
    /// 1. Loading the current manifest (or creating initial state)
    /// 2. Incrementing the writer epoch
    /// 3. Atomically writing the new manifest
    ///
    /// If another writer has already claimed the region (version conflict),
    /// this fails immediately rather than retrying. This prevents "epoch wars"
    /// where multiple writers keep fencing each other.
    ///
    /// # Returns
    ///
    /// A tuple of `(epoch, RegionManifest)` where the manifest is the
    /// claimed state (may be freshly created or loaded and epoch-bumped).
    ///
    /// # Errors
    ///
    /// Returns an error if another writer already claimed the region.
    pub async fn claim_epoch(&self, region_spec_id: u32) -> Result<(u64, RegionManifest)> {
        let current = self.read_latest().await?;

        let (next_version, next_epoch, base_manifest) = match current {
            Some(m) => (m.version + 1, m.writer_epoch + 1, Some(m)),
            None => (1, 1, None),
        };

        let new_manifest = if let Some(base) = base_manifest {
            RegionManifest {
                version: next_version,
                writer_epoch: next_epoch,
                ..base
            }
        } else {
            RegionManifest {
                region_id: self.region_id,
                version: next_version,
                region_spec_id,
                writer_epoch: next_epoch,
                replay_after_wal_entry_position: 0,
                wal_entry_position_last_seen: 0,
                current_generation: 1,
                flushed_generations: vec![],
            }
        };

        self.write(&new_manifest).await.map_err(|e| {
            Error::io(
                format!(
                    "Failed to claim region {} (version {}): another writer may have claimed it: {}",
                    self.region_id, next_version, e
                ),
                location!(),
            )
        })?;

        info!(
            "Claimed region {} with epoch {} (version {})",
            self.region_id, next_epoch, next_version
        );

        Ok((next_epoch, new_manifest))
    }

    /// Check if the given epoch has been fenced by a newer writer.
    ///
    /// Loads the current manifest and compares epochs. If the stored epoch
    /// is higher than the local epoch, the writer has been fenced.
    pub async fn check_fenced(&self, local_epoch: u64) -> Result<()> {
        let current = self.read_latest().await?;
        Self::check_fenced_against(&current, local_epoch, self.region_id)
    }

    /// Check fencing against a pre-read manifest (avoids redundant read).
    fn check_fenced_against(
        manifest: &Option<RegionManifest>,
        local_epoch: u64,
        region_id: Uuid,
    ) -> Result<()> {
        match manifest {
            Some(m) if m.writer_epoch > local_epoch => Err(Error::io(
                format!(
                    "Writer fenced: local epoch {} < stored epoch {} for region {}",
                    local_epoch, m.writer_epoch, region_id
                ),
                location!(),
            )),
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
    pub async fn commit_update<F>(&self, local_epoch: u64, prepare_fn: F) -> Result<RegionManifest>
    where
        F: Fn(&RegionManifest) -> RegionManifest,
    {
        const MAX_RETRIES: usize = 10;

        for attempt in 0..MAX_RETRIES {
            // Step 1: Read latest
            let current = self
                .read_latest()
                .await?
                .ok_or_else(|| Error::io("Region manifest not found", location!()))?;

            // Step 2: Check fencing
            Self::check_fenced_against(&Some(current.clone()), local_epoch, self.region_id)?;

            // Step 3: Prepare new manifest
            let new_manifest = prepare_fn(&current);

            // Validate epoch matches
            if new_manifest.writer_epoch != local_epoch {
                return Err(Error::invalid_input(
                    format!(
                        "Manifest epoch {} doesn't match local epoch {}",
                        new_manifest.writer_epoch, local_epoch
                    ),
                    location!(),
                ));
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

        Err(Error::io(
            format!(
                "Failed to update manifest for region {} after {} attempts",
                self.region_id, MAX_RETRIES
            ),
            location!(),
        ))
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

    fn create_test_manifest(region_id: Uuid, version: u64, epoch: u64) -> RegionManifest {
        RegionManifest {
            region_id,
            version,
            region_spec_id: 0,
            writer_epoch: epoch,
            replay_after_wal_entry_position: 0,
            wal_entry_position_last_seen: 0,
            current_generation: 1,
            flushed_generations: vec![],
        }
    }

    #[tokio::test]
    async fn test_read_latest_empty() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let region_id = Uuid::new_v4();
        let manifest_store = RegionManifestStore::new(store, &base_path, region_id, 2);

        let result = manifest_store.read_latest().await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_write_and_read_manifest() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let region_id = Uuid::new_v4();
        let manifest_store = RegionManifestStore::new(store, &base_path, region_id, 2);

        let manifest = create_test_manifest(region_id, 1, 1);
        manifest_store.write(&manifest).await.unwrap();

        let loaded = manifest_store.read_latest().await.unwrap().unwrap();
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.writer_epoch, 1);
        assert_eq!(loaded.region_id, region_id);
    }

    #[tokio::test]
    async fn test_multiple_versions() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let region_id = Uuid::new_v4();
        let manifest_store = RegionManifestStore::new(store, &base_path, region_id, 2);

        // Write multiple versions
        for version in 1..=5 {
            let manifest = create_test_manifest(region_id, version, version);
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
        let region_id = Uuid::new_v4();
        let manifest_store = RegionManifestStore::new(store, &base_path, region_id, 2);

        for version in 1..=3 {
            let manifest = create_test_manifest(region_id, version, version * 10);
            manifest_store.write(&manifest).await.unwrap();
        }

        let v2 = manifest_store.read_version(2).await.unwrap();
        assert_eq!(v2.version, 2);
        assert_eq!(v2.writer_epoch, 20);
    }

    #[tokio::test]
    async fn test_put_if_not_exists() {
        let (store, base_path, _temp_dir) = create_local_store().await;
        let region_id = Uuid::new_v4();
        let manifest_store = RegionManifestStore::new(store, &base_path, region_id, 2);

        let manifest1 = create_test_manifest(region_id, 1, 1);
        manifest_store.write(&manifest1).await.unwrap();

        // Second write to same version should fail
        let manifest2 = create_test_manifest(region_id, 1, 2);
        let result = manifest_store.write(&manifest2).await;
        assert!(result.is_err());
    }
}
