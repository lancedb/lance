// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Temporary scratch storage for memory-budgeted builders.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use lance_core::Result;
use lance_core::utils::tempfile::TempDir;
use object_store::path::Path;

use crate::local::to_local_path;
use crate::object_store::ObjectStore;
use crate::object_writer::DiskQuota;
use crate::traits::{Reader, Writer};

/// Session-scoped scratch store for reclaimable intermediate files.
pub trait SpillStore: Send + Sync + std::fmt::Debug + 'static {
    /// Object store backing the scratch directory.
    fn object_store(&self) -> Arc<ObjectStore>;

    /// Root directory for files owned by this scratch store.
    fn root_path(&self) -> Path;

    /// Create one empty scratch file. The file is deleted when the returned
    /// handle is dropped.
    fn create_spill_file(&self) -> Result<Box<dyn SpillFile>>;
}

/// A single reclaimable spill file.
#[async_trait]
pub trait SpillFile: Send + Sync + std::fmt::Debug {
    /// Open a writer for this spill file.
    async fn writer(&self) -> Result<Box<dyn Writer>>;

    /// Open a reader for this spill file.
    async fn reader(&self) -> Result<Box<dyn Reader>>;

    /// Return the object-store path for diagnostics and tests.
    fn path(&self) -> &Path;
}

/// Local-disk implementation of [`SpillStore`].
#[derive(Debug)]
pub struct LocalSpillStore {
    temp_dir: Arc<TempDir>,
    object_store: Arc<ObjectStore>,
    disk_quota: DiskQuota,
    next_file_id: AtomicU64,
}

impl LocalSpillStore {
    /// Create a local spill store capped at `cap_bytes` live scratch bytes.
    pub fn try_new(cap_bytes: u64) -> Result<Self> {
        let temp_dir = Arc::new(TempDir::try_new()?);
        let disk_quota = DiskQuota::new(cap_bytes);
        Ok(Self {
            temp_dir,
            object_store: Arc::new(ObjectStore::local_with_disk_quota(disk_quota.clone())),
            disk_quota,
            next_file_id: AtomicU64::new(0),
        })
    }

    /// Create a local spill store in the system temporary directory.
    pub fn new(cap_bytes: u64) -> Self {
        Self::try_new(cap_bytes).expect("failed to create local spill store")
    }

    /// Return the configured cap in bytes.
    pub fn cap_bytes(&self) -> u64 {
        self.disk_quota.cap_bytes()
    }

    /// Return the currently reserved spill bytes.
    pub fn used_bytes(&self) -> u64 {
        self.disk_quota.used_bytes()
    }
}

impl SpillStore for LocalSpillStore {
    fn object_store(&self) -> Arc<ObjectStore> {
        self.object_store.clone()
    }

    fn root_path(&self) -> Path {
        self.temp_dir.obj_path()
    }

    fn create_spill_file(&self) -> Result<Box<dyn SpillFile>> {
        let file_id = self.next_file_id.fetch_add(1, Ordering::Relaxed);
        let path = self
            .temp_dir
            .std_path()
            .join(format!("spill-{file_id}.bin"));
        let path = Path::from_absolute_path(path)?;
        Ok(Box::new(LocalSpillFile {
            _temp_dir: self.temp_dir.clone(),
            object_store: self.object_store.clone(),
            disk_quota: self.disk_quota.clone(),
            path,
        }))
    }
}

#[derive(Debug)]
struct LocalSpillFile {
    _temp_dir: Arc<TempDir>,
    object_store: Arc<ObjectStore>,
    disk_quota: DiskQuota,
    path: Path,
}

#[async_trait]
impl SpillFile for LocalSpillFile {
    async fn writer(&self) -> Result<Box<dyn Writer>> {
        self.object_store.create(&self.path).await
    }

    async fn reader(&self) -> Result<Box<dyn Reader>> {
        self.object_store.open(&self.path).await
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for LocalSpillFile {
    fn drop(&mut self) {
        let local_path = to_local_path(&self.path);
        if let Ok(metadata) = std::fs::metadata(&local_path) {
            self.disk_quota.release(metadata.len());
        }
        if let Err(err) = std::fs::remove_file(&local_path)
            && err.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(path = %self.path, error = %err, "failed to remove spill file");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use lance_core::Error;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn local_spill_file_cleans_up_and_releases_quota() {
        let store = LocalSpillStore::try_new(1024).unwrap();
        let spill = store.create_spill_file().unwrap();
        let path = spill.path().clone();

        let mut writer = spill.writer().await.unwrap();
        writer.write_all(b"hello spill").await.unwrap();
        Writer::shutdown(&mut writer).await.unwrap();

        assert_eq!(store.used_bytes(), 11);
        assert!(std::path::Path::new(&to_local_path(&path)).exists());

        let reader = spill.reader().await.unwrap();
        assert_eq!(
            reader.get_all().await.unwrap(),
            Bytes::from_static(b"hello spill")
        );

        drop(spill);

        assert_eq!(store.used_bytes(), 0);
        assert!(!std::path::Path::new(&to_local_path(&path)).exists());
    }

    #[tokio::test]
    async fn local_spill_file_returns_typed_error_on_cap_exhaustion() {
        let store = LocalSpillStore::try_new(4).unwrap();
        let spill = store.create_spill_file().unwrap();
        let mut writer = spill.writer().await.unwrap();

        writer.write_all(b"abcd").await.unwrap();
        let err = writer.write_all(b"e").await.unwrap_err();
        let err = Error::from(err);

        assert!(matches!(err, Error::DiskCapExceeded { .. }));
    }
}
