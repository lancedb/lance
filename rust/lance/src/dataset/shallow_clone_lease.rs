// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;

use futures::TryStreamExt;
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{Error, Result};

const LEASE_FORMAT_VERSION: u32 = 2;
const SHALLOW_CLONE_LEASES_DIR: &str = "shallow_clones";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShallowCloneLease {
    format_version: u32,
    /// Exact source namespace for a direct lease; nil only for a branch-wide
    /// transitive lease.
    pub source_namespace_uuid: Uuid,
    pub source_branch: Option<String>,
    pub source_version: u64,
    pub source_manifest_path: String,
    /// A transitive lease pins the entire source branch. Clone manifests do
    /// not retain the exact ancestor version that introduced every inherited
    /// base path, so narrowing this to one version would be unsafe.
    #[serde(default)]
    pub pin_all_versions: bool,
    pub target_uri: String,
    /// Object-store identity for `target_base_path`. If an ancestor cleanup
    /// cannot prove it is addressing the same store, the lease remains pinned.
    pub target_store_prefix: String,
    /// Filled after the target manifest commit. A missing value is a pending
    /// lease and must be retained conservatively after a client crash.
    pub target_manifest_path: Option<String>,
    /// Target dataset root in the same object store. The source lease is
    /// released only after this entire prefix disappears, so successor target
    /// manifests that inherit source references remain protected.
    pub target_base_path: Option<String>,
}

impl ShallowCloneLease {
    pub(crate) fn pending(
        source_namespace_uuid: Uuid,
        source_branch: Option<String>,
        source_version: u64,
        source_manifest_path: String,
        target_uri: String,
        target_store_prefix: String,
    ) -> Self {
        Self {
            format_version: LEASE_FORMAT_VERSION,
            source_namespace_uuid,
            source_branch,
            source_version,
            source_manifest_path,
            pin_all_versions: false,
            target_uri,
            target_store_prefix,
            target_manifest_path: None,
            target_base_path: None,
        }
    }

    pub(crate) fn transitive_pending(
        source_branch: Option<String>,
        source_root: String,
        target_uri: String,
        target_store_prefix: String,
    ) -> Self {
        Self {
            format_version: LEASE_FORMAT_VERSION,
            source_namespace_uuid: Uuid::nil(),
            source_branch,
            source_version: 0,
            source_manifest_path: source_root,
            pin_all_versions: true,
            target_uri,
            target_store_prefix,
            target_manifest_path: None,
            target_base_path: None,
        }
    }

    pub(crate) fn complete(&mut self, target_manifest_path: String, target_base_path: String) {
        self.target_manifest_path = Some(target_manifest_path);
        self.target_base_path = Some(target_base_path);
    }

    fn validate(&self) -> Result<()> {
        if self.format_version != LEASE_FORMAT_VERSION
            || (self.pin_all_versions
                && (!self.source_namespace_uuid.is_nil() || self.source_version != 0))
            || (!self.pin_all_versions
                && (self.source_namespace_uuid.is_nil() || self.source_version == 0))
            || self.source_manifest_path.is_empty()
            || self.target_uri.is_empty()
            || self
                .target_manifest_path
                .as_ref()
                .is_some_and(String::is_empty)
            || self.target_base_path.as_ref().is_some_and(String::is_empty)
            || self.target_manifest_path.is_some() != self.target_base_path.is_some()
        {
            return Err(Error::invalid_input(
                "invalid storage-version-2.3 shallow-clone lease",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ActiveShallowClonePins {
    versions: HashSet<u64>,
    pin_all_versions: bool,
}

impl ActiveShallowClonePins {
    pub(crate) fn contains(&self, version: u64) -> bool {
        self.pin_all_versions || self.versions.contains(&version)
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        !self.pin_all_versions && self.versions.is_empty()
    }
}

fn lease_dir(root: &Path) -> Path {
    root.clone().join("_refs").join(SHALLOW_CLONE_LEASES_DIR)
}

fn lease_path(root: &Path, lease_id: &str) -> Path {
    lease_dir(root).join(format!("{lease_id}.json"))
}

pub async fn write_shallow_clone_lease(
    object_store: &ObjectStore,
    root: &Path,
    lease_id: &str,
    lease: &ShallowCloneLease,
) -> Result<()> {
    lease.validate()?;
    let bytes = serde_json::to_vec(lease)?;
    object_store
        .put(&lease_path(root, lease_id), &bytes)
        .await
        .map(|_| ())
}

/// Return source snapshots pinned by independent shallow clones.
///
/// A completed lease remains active while the target dataset prefix exists.
/// This deliberately includes successor manifests that inherit the clone's
/// source references. Pending leases are always retained: a target commit may
/// have become visible before the client managed to finalize the lease.
/// Transitive leases conservatively pin the entire ancestor branch because an
/// inherited base path does not encode the exact source version that introduced
/// each referenced object. A target in a different object store remains pinned
/// until an external lease-release mechanism can prove its deletion; source
/// cleanup cannot safely infer cross-store liveness.
/// Corrupt or unreadable leases fail cleanup instead of risking source-data
/// deletion.
pub async fn active_shallow_clone_pins(
    object_store: &ObjectStore,
    root: &Path,
    source_namespace_uuid: Uuid,
    source_branch: Option<&str>,
    prune_stale_leases: bool,
) -> Result<ActiveShallowClonePins> {
    let directory = lease_dir(root);
    let lease_files = object_store.read_dir(directory.clone()).await?;
    let mut active = ActiveShallowClonePins::default();
    for file_name in lease_files
        .into_iter()
        .filter(|name| name.ends_with(".json"))
    {
        let path = directory.clone().join(file_name.as_str());
        let bytes = object_store.read_one_all(&path).await?;
        let lease: ShallowCloneLease = serde_json::from_slice(&bytes).map_err(|error| {
            Error::invalid_input(format!(
                "failed to decode shallow-clone lease {path}: {error}"
            ))
        })?;
        lease.validate()?;
        if lease.source_branch.as_deref() != source_branch
            || (!lease.pin_all_versions && lease.source_namespace_uuid != source_namespace_uuid)
        {
            continue;
        }
        let target_exists = match lease.target_base_path.as_deref() {
            Some(target_base_path) if lease.target_store_prefix == object_store.store_prefix => {
                object_store
                    .list(Some(Path::parse(target_base_path)?))
                    .try_next()
                    .await?
                    .is_some()
            }
            // Cross-store liveness cannot be proven from an ancestor cleanup
            // context. Retain rather than risk deleting a live descendant's
            // immutable objects.
            Some(_) => true,
            None => true,
        };
        if target_exists {
            if lease.pin_all_versions {
                active.pin_all_versions = true;
            } else {
                active.versions.insert(lease.source_version);
            }
        } else if prune_stale_leases {
            object_store.delete(&path).await?;
        }
    }
    Ok(active)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn pending_and_completed_leases_pin_until_target_manifest_is_removed() {
        let object_store = ObjectStore::memory();
        let root = Path::from("source");
        let mut lease = ShallowCloneLease::pending(
            Uuid::new_v4(),
            None,
            7,
            "source/_versions/7.manifest".to_string(),
            "memory://target".to_string(),
            object_store.store_prefix.clone(),
        );
        write_shallow_clone_lease(&object_store, &root, "lease", &lease)
            .await
            .unwrap();
        assert_eq!(
            active_shallow_clone_pins(
                &object_store,
                &root,
                lease.source_namespace_uuid,
                None,
                false,
            )
            .await
            .unwrap(),
            ActiveShallowClonePins {
                versions: HashSet::from([7]),
                pin_all_versions: false,
            }
        );

        let target_manifest = Path::from("target/_versions/1.manifest");
        object_store
            .put(&target_manifest, b"manifest")
            .await
            .unwrap();
        lease.complete(target_manifest.to_string(), "target".to_string());
        write_shallow_clone_lease(&object_store, &root, "lease", &lease)
            .await
            .unwrap();
        assert_eq!(
            active_shallow_clone_pins(
                &object_store,
                &root,
                lease.source_namespace_uuid,
                None,
                false,
            )
            .await
            .unwrap(),
            ActiveShallowClonePins {
                versions: HashSet::from([7]),
                pin_all_versions: false,
            }
        );

        object_store.delete(&target_manifest).await.unwrap();
        assert!(
            active_shallow_clone_pins(
                &object_store,
                &root,
                lease.source_namespace_uuid,
                None,
                true,
            )
            .await
            .unwrap()
            .is_empty()
        );
        assert!(
            object_store
                .read_dir(lease_dir(&root))
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn transitive_lease_pins_every_version_until_descendant_is_removed() {
        let object_store = ObjectStore::memory();
        let root = Path::from("source");
        let mut lease = ShallowCloneLease::transitive_pending(
            None,
            root.to_string(),
            "memory://descendant".to_string(),
            object_store.store_prefix.clone(),
        );
        let descendant_manifest = Path::from("descendant/_versions/1.manifest");
        object_store
            .put(&descendant_manifest, b"manifest")
            .await
            .unwrap();
        lease.complete(descendant_manifest.to_string(), "descendant".to_string());
        write_shallow_clone_lease(&object_store, &root, "transitive", &lease)
            .await
            .unwrap();

        let pins = active_shallow_clone_pins(&object_store, &root, Uuid::new_v4(), None, false)
            .await
            .unwrap();
        assert!(pins.contains(1));
        assert!(pins.contains(u64::MAX));

        object_store.delete(&descendant_manifest).await.unwrap();
        assert!(
            active_shallow_clone_pins(&object_store, &root, Uuid::new_v4(), None, true,)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn cross_store_target_is_retained_when_liveness_cannot_be_proven() {
        let object_store = ObjectStore::memory();
        let root = Path::from("source");
        let mut lease = ShallowCloneLease::transitive_pending(
            None,
            root.to_string(),
            "s3://other/descendant".to_string(),
            "different-object-store".to_string(),
        );
        lease.complete(
            "descendant/_versions/1.manifest".to_string(),
            "descendant".to_string(),
        );
        write_shallow_clone_lease(&object_store, &root, "cross-store", &lease)
            .await
            .unwrap();

        let pins = active_shallow_clone_pins(&object_store, &root, Uuid::new_v4(), None, true)
            .await
            .unwrap();
        assert!(pins.contains(u64::MAX));
        assert_eq!(
            object_store.read_dir(lease_dir(&root)).await.unwrap(),
            vec!["cross-store.json"]
        );
    }
}
