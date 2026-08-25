// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Honest flat-manifest baseline for the Bε-tree benchmark.
//!
//! This is *today's* behavior: the whole (growing) fragment list lives inline in
//! one protobuf manifest, and every commit rewrites the entire file via the real
//! [`write_manifest_file_to_path`] path — so byte counts and latency are
//! authentic. A backfill commit adds a `DataFile` to a subset of fragments and
//! still pays the full-manifest write.

use std::collections::HashMap;
use std::sync::Arc;

use object_store::path::Path;

use crate::format::{DataFile, DataStorageFormat, Fragment, Manifest};
use crate::io::commit::write_manifest_file_to_path;
use crate::io::manifest::read_manifest;
use lance_core::Result;
use lance_core::datatypes::Schema;
use lance_io::object_store::ObjectStore;

pub fn manifest_path(base: &Path, version: u64) -> Path {
    base.clone()
        .join("_versions")
        .join(format!("{version}.manifest"))
}

/// A writer session over a flat manifest. Holds the full fragment list in memory
/// (inherent to flat — it rewrites the whole list every commit).
pub struct FlatBaseline {
    object_store: Arc<ObjectStore>,
    base: Path,
    manifest: Manifest,
    /// fragment id -> position in the manifest fragment list.
    index: HashMap<u64, usize>,
}

impl FlatBaseline {
    pub fn new(
        object_store: Arc<ObjectStore>,
        base: Path,
        schema: Schema,
        fragments: Vec<Fragment>,
    ) -> Self {
        let index = fragments
            .iter()
            .enumerate()
            .map(|(i, f)| (f.id, i))
            .collect();
        let mut manifest = Manifest::new(
            schema,
            Arc::new(fragments),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        manifest.version = 1;
        Self {
            object_store,
            base,
            manifest,
            index,
        }
    }

    pub fn version(&self) -> u64 {
        self.manifest.version
    }

    /// Write the current manifest at its version; returns bytes written.
    pub async fn write(&mut self) -> Result<u64> {
        let path = manifest_path(&self.base, self.manifest.version);
        let res =
            write_manifest_file_to_path(&self.object_store, &mut self.manifest, None, &path, None)
                .await?;
        Ok(res.size as u64)
    }

    /// Apply the backfill adds to their fragments, bump the version, and rewrite
    /// the full manifest. Returns bytes written.
    pub async fn commit_add_data_files(&mut self, adds: &[(u64, DataFile)]) -> Result<u64> {
        {
            let frags = Arc::make_mut(&mut self.manifest.fragments);
            for (fid, file) in adds {
                if let Some(&pos) = self.index.get(fid) {
                    frags[pos].files.push(file.clone());
                }
            }
        }
        self.manifest.version += 1;
        self.write().await
    }

    /// Cold open: read a manifest version back from storage.
    pub async fn cold_open(
        object_store: &ObjectStore,
        base: &Path,
        version: u64,
    ) -> Result<Manifest> {
        read_manifest(object_store, &manifest_path(base, version), None).await
    }
}
