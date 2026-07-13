// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reclaimable scratch storage.
//!
//! A [`SpillStore`] hands out scratch space for temporary state that is too
//! large to keep in memory and is read back later in the same process (for
//! example, posting lists or shuffle runs accumulated while building an index).
//! The backing storage is reclaimed automatically when the handle is dropped.
//!
//! [`SpillStore::new_spill`] returns a [`Writer`] paired with a [`Spill`]
//! handle: the writer is the byte sink (feed it to `FileWriter::try_new`, or
//! write to it directly); the [`Spill`] reads the bytes back (via
//! [`crate::scheduler::ScanScheduler::open_reader`] for a v2 `FileReader`) and
//! owns the file's lifetime.
//!
//! # Lifecycle
//!
//! - **Write-once.** The only way to obtain a writer is `new_spill`, and each
//!   call allocates a fresh unit of storage, so a single spill cannot be
//!   written twice — there is no second-writer path to guard against.
//! - **Write-before-read.** [`Spill::reader`] fails until the writer has been
//!   shut down, so partially written bytes are never read back.
//! - **RAII.** Dropping a completed [`Spill`] deletes the file and releases its
//!   bytes back to the store's disk budget. Successfully aborting its writer
//!   releases the reservation immediately. The store's temp directory is the
//!   backstop for anything leaked if a handle is forgotten.
//!
//! # Disk cap
//!
//! [`LocalSpillStore::with_cap`] enforces a byte budget shared across all live
//! handles, returning a typed [`lance_core::Error::DiskCapExceeded`] rather than
//! silently filling the disk. Accounting tracks the bytes accepted for each
//! spill: a completed spill releases them on drop, while a successful abort
//! releases them immediately. Dropping an unfinished writer without a
//! successful abort can retain its reservation until the store is dropped.

use std::io;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use async_trait::async_trait;
use object_store::path::Path;
use tokio::io::AsyncWrite;

use lance_core::{Error, Result};

use crate::object_store::ObjectStore;
use crate::object_writer::WriteResult;
use crate::traits::{Reader, Writer};

/// A factory for scratch storage.
///
/// The trait is object-safe and `Send + Sync` so it can be held behind an
/// `Arc<dyn SpillStore>` (e.g. inside a `Session`). Implementations need not be
/// backed by local files (e.g. in-memory buffers, remote object stores).
#[async_trait]
pub trait SpillStore: Send + Sync + 'static {
    /// Allocate a unit of scratch storage.
    ///
    /// Returns the byte sink to write it with and a [`Spill`] handle to read it
    /// back. For a capped store, writes that would exceed the cap fail with
    /// [`lance_core::Error::DiskCapExceeded`]. The storage is reclaimed when the
    /// [`Spill`] is dropped.
    async fn new_spill(&self) -> Result<(Box<dyn Writer>, Box<dyn Spill>)>;
}

/// The readable half of a spill, and the owner of its backing storage.
///
/// Dropping it reclaims the storage. The trait is object-safe so it can be
/// returned as `Box<dyn Spill>` from [`SpillStore::new_spill`].
#[async_trait]
pub trait Spill: Send + Sync {
    /// Open a reader over the spilled bytes.
    ///
    /// Fails until the paired writer has been shut down, since the bytes are not
    /// complete before then.
    async fn reader(&self) -> Result<Box<dyn Reader>>;
}

/// A shared, cloneable byte budget.
///
/// Cloning produces another handle to the *same* underlying counter, so a quota
/// shared across many writers enforces a single combined cap.
#[derive(Debug, Clone)]
struct DiskQuota {
    cap_bytes: u64,
    used: Arc<Mutex<u64>>,
}

impl DiskQuota {
    fn new(cap_bytes: u64) -> Self {
        Self {
            cap_bytes,
            used: Arc::new(Mutex::new(0)),
        }
    }

    /// Try to reserve `n` bytes, failing with [`Error::DiskCapExceeded`] if the
    /// reservation would push total usage past the cap.
    fn try_reserve(&self, n: u64) -> Result<()> {
        // The lock is held only for a couple of arithmetic ops and never across
        // an `.await`, so a std `Mutex` is the simplest correct choice.
        let mut used = self.used.lock().unwrap();
        let Some(next) = used.checked_add(n) else {
            return Err(Error::disk_cap_exceeded(self.cap_bytes, *used));
        };
        if next > self.cap_bytes {
            return Err(Error::disk_cap_exceeded(self.cap_bytes, *used));
        }
        *used = next;
        Ok(())
    }

    /// Release `n` previously reserved bytes back to the budget.
    fn release(&self, n: u64) {
        // Saturating sub keeps a stray double-release from underflowing.
        let mut used = self.used.lock().unwrap();
        *used = used.saturating_sub(n);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpillWriterPhase {
    Writing,
    ShuttingDown,
    Finished,
    Failed,
    Aborted,
}

#[derive(Debug)]
struct SpillLifecycleState {
    phase: SpillWriterPhase,
    has_spill_owner: bool,
    reserved_bytes: u64,
}

/// Coordinates the writer, readable owner, and per-spill quota reservation.
///
/// The mutex linearizes owner drop against shutdown completion. Whichever side
/// observes both "owner gone" and "writer finished" takes the reservation and
/// removes the final path, so neither side can miss cleanup or release twice.
#[derive(Debug)]
struct SpillLifecycle {
    quota: Option<DiskQuota>,
    fs_path: PathBuf,
    state: Mutex<SpillLifecycleState>,
}

impl SpillLifecycle {
    fn new(quota: Option<DiskQuota>, fs_path: PathBuf) -> Self {
        Self {
            quota,
            fs_path,
            state: Mutex::new(SpillLifecycleState {
                phase: SpillWriterPhase::Writing,
                has_spill_owner: true,
                reserved_bytes: 0,
            }),
        }
    }

    fn writer_error(&self, action: &str, reason: &str) -> io::Error {
        io::Error::other(format!(
            "cannot {action} spill writer for '{}': {reason}",
            self.fs_path.display()
        ))
    }

    fn ensure_writable(&self, action: &str) -> io::Result<()> {
        let state = self.state.lock().unwrap();
        if !state.has_spill_owner {
            return Err(self.writer_error(action, "the paired Spill handle was dropped"));
        }
        match state.phase {
            SpillWriterPhase::Writing => Ok(()),
            SpillWriterPhase::ShuttingDown => {
                Err(self.writer_error(action, "shutdown has already started"))
            }
            SpillWriterPhase::Finished => {
                Err(self.writer_error(action, "shutdown has already completed"))
            }
            SpillWriterPhase::Failed => {
                Err(self.writer_error(action, "a previous shutdown failed"))
            }
            SpillWriterPhase::Aborted => Err(self.writer_error(action, "the writer was aborted")),
        }
    }

    fn try_reserve(&self, bytes: u64) -> Result<()> {
        if let Some(quota) = &self.quota {
            quota.try_reserve(bytes)?;
        }
        Ok(())
    }

    fn release_unaccepted(&self, bytes: u64) {
        if let Some(quota) = &self.quota {
            quota.release(bytes);
        }
    }

    fn record_accepted(&self, bytes: u64) -> io::Result<()> {
        if self.quota.is_none() {
            return Ok(());
        }
        let mut state = self.state.lock().unwrap();
        let Some(reserved_bytes) = state.reserved_bytes.checked_add(bytes) else {
            state.phase = SpillWriterPhase::Failed;
            return Err(self.writer_error(
                "account for bytes in",
                &format!(
                    "reservation overflowed: reserved_bytes={}, accepted_bytes={bytes}",
                    state.reserved_bytes
                ),
            ));
        };
        state.reserved_bytes = reserved_bytes;
        Ok(())
    }

    /// Start trait-based shutdown. `false` means the spill owner is already
    /// gone, so this path should abort instead of publishing the file.
    fn begin_shutdown(&self) -> Result<bool> {
        let mut state = self.state.lock().unwrap();
        match state.phase {
            SpillWriterPhase::Writing | SpillWriterPhase::ShuttingDown => {
                if !state.has_spill_owner {
                    return Ok(false);
                }
                state.phase = SpillWriterPhase::ShuttingDown;
                Ok(true)
            }
            // Preserve the inner Writer's idempotent shutdown behavior. This
            // also lets callers drive AsyncWrite::poll_shutdown first and then
            // obtain the WriteResult through Writer::shutdown.
            SpillWriterPhase::Finished if state.has_spill_owner => Ok(true),
            SpillWriterPhase::Finished => Err(self
                .writer_error("shut down", "the paired Spill handle was dropped")
                .into()),
            SpillWriterPhase::Failed => Err(self
                .writer_error("shut down", "a previous shutdown failed")
                .into()),
            SpillWriterPhase::Aborted => Err(self
                .writer_error("shut down", "the writer was aborted")
                .into()),
        }
    }

    /// Start or resume poll-based shutdown. `false` means shutdown had already
    /// completed successfully while the spill owner was still present.
    fn begin_poll_shutdown(&self) -> io::Result<bool> {
        let mut state = self.state.lock().unwrap();
        match state.phase {
            SpillWriterPhase::Writing => {
                state.phase = SpillWriterPhase::ShuttingDown;
                Ok(true)
            }
            SpillWriterPhase::ShuttingDown => Ok(true),
            SpillWriterPhase::Finished if state.has_spill_owner => Ok(false),
            SpillWriterPhase::Finished => {
                Err(self.writer_error("shut down", "the paired Spill handle was dropped"))
            }
            SpillWriterPhase::Failed => {
                Err(self.writer_error("shut down", "a previous shutdown failed"))
            }
            SpillWriterPhase::Aborted => {
                Err(self.writer_error("shut down", "the writer was aborted"))
            }
        }
    }

    fn shutdown_failed(&self) {
        self.state.lock().unwrap().phase = SpillWriterPhase::Failed;
    }

    /// Mark shutdown complete and clean up if the owner disappeared while the
    /// inner writer was finishing. Returns whether ownership was lost.
    fn shutdown_succeeded(&self) -> bool {
        let reserved_bytes = {
            let mut state = self.state.lock().unwrap();
            state.phase = SpillWriterPhase::Finished;
            if state.has_spill_owner {
                return false;
            }
            std::mem::take(&mut state.reserved_bytes)
        };
        self.release_reserved(reserved_bytes);
        self.remove_file();
        true
    }

    fn abort_succeeded(&self) {
        let reserved_bytes = {
            let mut state = self.state.lock().unwrap();
            state.phase = SpillWriterPhase::Aborted;
            std::mem::take(&mut state.reserved_bytes)
        };
        self.release_reserved(reserved_bytes);
        self.remove_file();
    }

    fn spill_dropped(&self) {
        let reserved_bytes = {
            let mut state = self.state.lock().unwrap();
            state.has_spill_owner = false;
            if state.phase == SpillWriterPhase::Finished {
                std::mem::take(&mut state.reserved_bytes)
            } else {
                0
            }
        };
        self.release_reserved(reserved_bytes);
        self.remove_file();
    }

    fn is_finished(&self) -> bool {
        self.state.lock().unwrap().phase == SpillWriterPhase::Finished
    }

    fn release_reserved(&self, bytes: u64) {
        if let Some(quota) = &self.quota {
            quota.release(bytes);
        }
    }

    fn remove_file(&self) {
        if let Err(error) = std::fs::remove_file(&self.fs_path)
            && error.kind() != io::ErrorKind::NotFound
        {
            log::warn!(
                "Failed to remove spill file '{}': {}",
                self.fs_path.display(),
                error
            );
        }
    }
}

/// The byte sink handed out by [`SpillStore::new_spill`].
///
/// It optionally reserves a [`DiskQuota`] as bytes are written (keeping cap
/// enforcement inside the spill store rather than in [`ObjectStore`], and
/// working for any backend the store opens), and shares a lifecycle with the
/// paired [`Spill`] so shutdown and owner drop cannot miss one another.
struct SpillWriter {
    inner: Box<dyn Writer>,
    lifecycle: Arc<SpillLifecycle>,
}

impl AsyncWrite for SpillWriter {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        if let Err(error) = this.lifecycle.ensure_writable("write to") {
            return Poll::Ready(Err(error));
        }
        // Reserve up-front for the bytes we intend to write, then release the
        // remainder the inner writer did not accept so the reservation tracks
        // bytes actually buffered (and, for a write-once file, the file size).
        if let Err(e) = this.lifecycle.try_reserve(buf.len() as u64) {
            return Poll::Ready(Err(io::Error::other(e)));
        }
        let poll = Pin::new(this.inner.as_mut()).poll_write(cx, buf);
        match &poll {
            Poll::Ready(Ok(n)) => {
                this.lifecycle.release_unaccepted((buf.len() - *n) as u64);
                if let Err(error) = this.lifecycle.record_accepted(*n as u64) {
                    return Poll::Ready(Err(error));
                }
            }
            _ => this.lifecycle.release_unaccepted(buf.len() as u64),
        }
        poll
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if let Err(error) = this.lifecycle.ensure_writable("flush") {
            return Poll::Ready(Err(error));
        }
        Pin::new(this.inner.as_mut()).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        match this.lifecycle.begin_poll_shutdown() {
            Ok(true) => {}
            Ok(false) => return Poll::Ready(Ok(())),
            Err(error) => return Poll::Ready(Err(error)),
        }
        match Pin::new(this.inner.as_mut()).poll_shutdown(cx) {
            Poll::Ready(Ok(())) => {
                if this.lifecycle.shutdown_succeeded() {
                    Poll::Ready(Err(this.lifecycle.writer_error(
                        "shut down",
                        "the paired Spill handle was dropped during shutdown",
                    )))
                } else {
                    Poll::Ready(Ok(()))
                }
            }
            Poll::Ready(Err(error)) => {
                this.lifecycle.shutdown_failed();
                Poll::Ready(Err(error))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[async_trait]
impl Writer for SpillWriter {
    async fn tell(&mut self) -> Result<usize> {
        self.inner.tell().await
    }

    async fn shutdown(&mut self) -> Result<WriteResult> {
        if !self.lifecycle.begin_shutdown()? {
            return match self.inner.abort().await {
                Ok(()) => {
                    self.lifecycle.abort_succeeded();
                    Err(self
                        .lifecycle
                        .writer_error(
                            "shut down",
                            "the paired Spill handle was dropped before shutdown",
                        )
                        .into())
                }
                Err(abort_error) => Err(Error::io(format!(
                    "cannot shut down spill writer for '{}': the paired Spill handle was dropped; additionally failed to abort the inner writer: {abort_error}",
                    self.lifecycle.fs_path.display()
                ))),
            };
        }

        let result = match self.inner.shutdown().await {
            Ok(result) => result,
            Err(error) => {
                self.lifecycle.shutdown_failed();
                return Err(error);
            }
        };
        if self.lifecycle.shutdown_succeeded() {
            return Err(self
                .lifecycle
                .writer_error(
                    "shut down",
                    "the paired Spill handle was dropped during shutdown",
                )
                .into());
        }
        Ok(result)
    }

    async fn abort(&mut self) -> Result<()> {
        self.inner.abort().await?;
        self.lifecycle.abort_succeeded();
        Ok(())
    }
}

/// A [`SpillStore`] that writes temporary files to a local temp directory.
///
/// By default there is no disk cap. Use [`LocalSpillStore::with_cap`] to
/// configure one shared across every handle this store produces.
///
/// The temp directory is deleted when the store is dropped, cleaning up any
/// files whose handles have already been dropped.
pub struct LocalSpillStore {
    store: Arc<ObjectStore>,
    /// Backstop cleanup: removes the whole scratch directory on drop.
    temp_dir: Arc<tempfile::TempDir>,
    file_counter: Arc<AtomicU64>,
    /// Byte budget shared across every handle, enforced while writing.
    quota: Option<DiskQuota>,
}

impl LocalSpillStore {
    /// Create a store with no disk cap.
    pub fn new() -> Result<Self> {
        Ok(Self {
            store: Arc::new(ObjectStore::local()),
            temp_dir: Arc::new(tempfile::tempdir()?),
            file_counter: Arc::new(AtomicU64::new(0)),
            quota: None,
        })
    }

    /// Create a store that returns [`lance_core::Error::DiskCapExceeded`] once
    /// total bytes written across all live handles would exceed `cap_bytes`.
    pub fn with_cap(cap_bytes: u64) -> Result<Self> {
        Ok(Self {
            store: Arc::new(ObjectStore::local()),
            temp_dir: Arc::new(tempfile::tempdir()?),
            file_counter: Arc::new(AtomicU64::new(0)),
            quota: Some(DiskQuota::new(cap_bytes)),
        })
    }
}

impl Default for LocalSpillStore {
    fn default() -> Self {
        Self::new().expect("failed to create temp directory for LocalSpillStore")
    }
}

#[async_trait]
impl SpillStore for LocalSpillStore {
    async fn new_spill(&self) -> Result<(Box<dyn Writer>, Box<dyn Spill>)> {
        let idx = self.file_counter.fetch_add(1, Ordering::Relaxed);
        let fs_path = self.temp_dir.path().join(format!("spill_{idx:06}.bin"));
        let os_path = Path::from_absolute_path(&fs_path)?;
        let lifecycle = Arc::new(SpillLifecycle::new(self.quota.clone(), fs_path));

        let writer = Box::new(SpillWriter {
            inner: self.store.create(&os_path).await?,
            lifecycle: lifecycle.clone(),
        });
        let spill = Box::new(LocalSpill {
            store: self.store.clone(),
            os_path,
            lifecycle,
            _temp_dir: self.temp_dir.clone(),
        });
        Ok((writer, spill))
    }
}

/// The readable half of a [`LocalSpillStore`] spill; reclaims the file on drop.
struct LocalSpill {
    store: Arc<ObjectStore>,
    os_path: Path,
    lifecycle: Arc<SpillLifecycle>,
    /// Keep the store's temp directory alive for at least this file's lifetime.
    _temp_dir: Arc<tempfile::TempDir>,
}

#[async_trait]
impl Spill for LocalSpill {
    async fn reader(&self) -> Result<Box<dyn Reader>> {
        if !self.lifecycle.is_finished() {
            return Err(Error::invalid_input(
                "spill reader requested before the writer was shut down",
            ));
        }
        self.store.open(&self.os_path).await
    }
}

impl Drop for LocalSpill {
    fn drop(&mut self) {
        self.lifecycle.spill_dropped();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    /// Write `data` to a fresh writer and shut it down.
    async fn finish_writer(mut writer: Box<dyn Writer>, data: &[u8]) -> Result<()> {
        writer.write_all(data).await?;
        Writer::shutdown(writer.as_mut()).await?;
        Ok(())
    }

    #[test]
    fn test_disk_quota_reserve_release() {
        let quota = DiskQuota::new(100);
        quota.try_reserve(60).unwrap();
        assert!(quota.try_reserve(60).is_err());
        quota.release(60);
        quota.try_reserve(60).unwrap();
        // Reserving exactly up to the cap succeeds; one byte past it fails.
        quota.try_reserve(40).unwrap();
        assert!(quota.try_reserve(1).is_err());

        let max_quota = DiskQuota::new(u64::MAX);
        max_quota.try_reserve(u64::MAX).unwrap();
        let error = max_quota.try_reserve(1).unwrap_err();
        assert!(matches!(
            error,
            Error::DiskCapExceeded {
                cap_bytes: u64::MAX,
                used_bytes: u64::MAX,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_write_then_read() {
        let store = LocalSpillStore::new().unwrap();
        let (writer, spill) = store.new_spill().await.unwrap();

        let data = b"hello spill world";
        finish_writer(writer, data).await.unwrap();

        let reader = spill.reader().await.unwrap();
        let read_back = reader.get_all().await.unwrap();
        assert_eq!(read_back.as_ref(), data);
    }

    #[tokio::test]
    async fn test_reader_requires_finished_writer() {
        let store = LocalSpillStore::new().unwrap();
        let (mut writer, spill) = store.new_spill().await.unwrap();
        writer.write_all(b"partial").await.unwrap();

        // Reading before the writer is shut down is rejected.
        let Err(err) = spill.reader().await else {
            panic!("reader before shutdown should be rejected");
        };
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got {err:?}"
        );

        // After shutdown the reader sees the bytes.
        Writer::shutdown(writer.as_mut()).await.unwrap();
        let reader = spill.reader().await.unwrap();
        assert_eq!(reader.get_all().await.unwrap().as_ref(), b"partial");
    }

    #[tokio::test]
    async fn test_reader_ready_after_async_shutdown() {
        // Shutting down through the `AsyncWrite` surface (not the `Writer`
        // trait) must also mark the spill readable — covers poll_shutdown's
        // flag set, the path the `Writer::shutdown` tests don't reach.
        let store = LocalSpillStore::new().unwrap();
        let (mut writer, spill) = store.new_spill().await.unwrap();
        writer.write_all(b"async").await.unwrap();
        AsyncWriteExt::shutdown(&mut writer).await.unwrap();
        let result = Writer::shutdown(writer.as_mut()).await.unwrap();
        assert_eq!(result.size, b"async".len());

        let reader = spill.reader().await.unwrap();
        assert_eq!(reader.get_all().await.unwrap().as_ref(), b"async");
    }

    #[tokio::test]
    async fn test_reader_rejected_after_writer_abort() {
        let store = LocalSpillStore::new().unwrap();
        let (mut writer, spill) = store.new_spill().await.unwrap();
        writer.write_all(b"partial").await.unwrap();
        Writer::abort(writer.as_mut()).await.unwrap();

        let Err(err) = spill.reader().await else {
            panic!("reader after abort should be rejected");
        };
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got {err:?}"
        );
        assert!(
            err.to_string().contains("before the writer was shut down"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_empty_spill() {
        // A spill written with no bytes round-trips empty, and the capped path
        // handles the zero-byte reserve/stat without error.
        let store = LocalSpillStore::with_cap(100).unwrap();
        let (writer, spill) = store.new_spill().await.unwrap();
        finish_writer(writer, b"").await.unwrap();

        let reader = spill.reader().await.unwrap();
        assert!(reader.get_all().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_raii_cleanup() {
        let store = LocalSpillStore::new().unwrap();
        let (writer, spill) = store.new_spill().await.unwrap();
        finish_writer(writer, b"some bytes").await.unwrap();

        // The first spill gets a deterministic name under the store's temp dir.
        let path = store.temp_dir.path().join("spill_000000.bin");
        assert!(path.exists());
        drop(spill);
        assert!(!path.exists(), "spill file should be deleted on drop");
    }

    #[tokio::test]
    async fn test_cap_exceeded() {
        let store = LocalSpillStore::with_cap(100).unwrap();
        let (writer, _spill) = store.new_spill().await.unwrap();
        let err = finish_writer(writer, &[0u8; 101]).await.unwrap_err();
        assert!(
            matches!(err, Error::DiskCapExceeded { cap_bytes: 100, .. }),
            "expected DiskCapExceeded, got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_cap_shared_across_files() {
        let store = LocalSpillStore::with_cap(100).unwrap();
        let (writer_a, _spill_a) = store.new_spill().await.unwrap();
        let (writer_b, _spill_b) = store.new_spill().await.unwrap();

        finish_writer(writer_a, &[0u8; 60]).await.unwrap();
        // 60 already reserved by `a`; writing 60 more would reach 120 > 100.
        let err = finish_writer(writer_b, &[0u8; 60]).await.unwrap_err();
        assert!(
            matches!(err, Error::DiskCapExceeded { cap_bytes: 100, .. }),
            "expected DiskCapExceeded, got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_cap_freed_on_drop() {
        let store = LocalSpillStore::with_cap(100).unwrap();

        {
            let (writer, spill) = store.new_spill().await.unwrap();
            finish_writer(writer, &[0u8; 80]).await.unwrap();
            // `spill` drops at the end of this block, releasing its 80 bytes.
            drop(spill);
        }

        let (writer, _spill) = store.new_spill().await.unwrap();
        // Succeeds because the cap is no longer under pressure.
        finish_writer(writer, &[0u8; 80]).await.unwrap();
    }

    #[tokio::test]
    async fn test_cap_freed_after_abort() {
        let store = LocalSpillStore::with_cap(100).unwrap();
        let (mut writer, spill) = store.new_spill().await.unwrap();
        writer.write_all(&[0u8; 80]).await.unwrap();

        Writer::abort(writer.as_mut()).await.unwrap();
        drop(writer);
        drop(spill);

        let (writer, _spill) = store.new_spill().await.unwrap();
        finish_writer(writer, &[0u8; 80]).await.unwrap();
    }

    #[tokio::test]
    async fn test_drop_spill_before_shutdown_aborts_and_releases_cap() {
        let store = LocalSpillStore::with_cap(100).unwrap();
        let path = store.temp_dir.path().join("spill_000000.bin");
        let (mut writer, spill) = store.new_spill().await.unwrap();
        writer.write_all(&[0u8; 80]).await.unwrap();

        drop(spill);
        let write_error = writer.write_all(b"more").await.unwrap_err();
        assert!(
            write_error
                .to_string()
                .contains("paired Spill handle was dropped"),
            "unexpected error: {write_error}"
        );
        let shutdown_error = Writer::shutdown(writer.as_mut()).await.unwrap_err();
        assert!(
            matches!(shutdown_error, Error::IO { .. }),
            "expected IO error, got {shutdown_error:?}"
        );
        assert!(
            shutdown_error
                .to_string()
                .contains("paired Spill handle was dropped before shutdown"),
            "unexpected error: {shutdown_error}"
        );
        assert!(!path.exists(), "ownerless spill must not be published");

        let (replacement_writer, _replacement_spill) = store.new_spill().await.unwrap();
        finish_writer(replacement_writer, &[0u8; 80]).await.unwrap();
    }

    #[tokio::test]
    async fn test_abort_releases_only_its_own_reservation() {
        let store = LocalSpillStore::with_cap(100).unwrap();

        let (live_writer, _live_spill) = store.new_spill().await.unwrap();
        finish_writer(live_writer, &[0u8; 20]).await.unwrap();

        let (mut aborted_writer, aborted_spill) = store.new_spill().await.unwrap();
        aborted_writer.write_all(&[0u8; 80]).await.unwrap();
        Writer::abort(aborted_writer.as_mut()).await.unwrap();
        drop(aborted_writer);
        drop(aborted_spill);

        let (replacement_writer, _replacement_spill) = store.new_spill().await.unwrap();
        finish_writer(replacement_writer, &[0u8; 80]).await.unwrap();

        let (extra_writer, _extra_spill) = store.new_spill().await.unwrap();
        let error = finish_writer(extra_writer, &[0u8; 1]).await.unwrap_err();
        assert!(
            matches!(error, Error::DiskCapExceeded { cap_bytes: 100, .. }),
            "expected DiskCapExceeded, got {error:?}"
        );
    }

    #[tokio::test]
    async fn test_custom_implementation() {
        // A custom store can satisfy the traits without a local file.
        struct MemStore;
        struct MemSpill;

        #[async_trait]
        impl Spill for MemSpill {
            async fn reader(&self) -> Result<Box<dyn Reader>> {
                ObjectStore::memory().open(&Path::from("/mem")).await
            }
        }

        #[async_trait]
        impl SpillStore for MemStore {
            async fn new_spill(&self) -> Result<(Box<dyn Writer>, Box<dyn Spill>)> {
                let writer = ObjectStore::memory().create(&Path::from("/mem")).await?;
                Ok((writer, Box::new(MemSpill)))
            }
        }

        let store = MemStore;
        // Exercise the factory + trait objects; the in-memory store is a fresh
        // instance per call so we don't round-trip data here.
        let (_writer, _spill) = store.new_spill().await.unwrap();
    }

    /// A [`Writer`] whose `poll_write` accepts a fixed number of bytes per call,
    /// or fails, so we can drive the [`SpillWriter`] release arms that the local
    /// backend (which accepts every write in full) never hits.
    struct ControlledWriter {
        outcome: Poll<io::Result<usize>>,
        remaining_abort_failures: usize,
    }

    fn test_lifecycle(quota: DiskQuota) -> (Arc<SpillLifecycle>, tempfile::TempDir) {
        let temp_dir = tempfile::tempdir().unwrap();
        let lifecycle = Arc::new(SpillLifecycle::new(
            Some(quota),
            temp_dir.path().join("controlled-spill.bin"),
        ));
        (lifecycle, temp_dir)
    }

    impl AsyncWrite for ControlledWriter {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            match &self.outcome {
                Poll::Ready(Ok(n)) => Poll::Ready(Ok((*n).min(buf.len()))),
                Poll::Ready(Err(e)) => Poll::Ready(Err(io::Error::new(e.kind(), e.to_string()))),
                Poll::Pending => Poll::Pending,
            }
        }
        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    #[async_trait]
    impl Writer for ControlledWriter {
        async fn tell(&mut self) -> Result<usize> {
            Ok(0)
        }
        async fn shutdown(&mut self) -> Result<WriteResult> {
            Ok(WriteResult::default())
        }

        async fn abort(&mut self) -> Result<()> {
            if self.remaining_abort_failures > 0 {
                self.remaining_abort_failures -= 1;
                Err(Error::io("controlled abort failure"))
            } else {
                Ok(())
            }
        }
    }

    /// Publishes a file only after the test releases its shutdown gate.
    struct GatedShutdownWriter {
        shutdown_started: Option<tokio::sync::oneshot::Sender<()>>,
        resume_shutdown: Option<tokio::sync::oneshot::Receiver<()>>,
        publish_path: PathBuf,
        size: usize,
    }

    impl AsyncWrite for GatedShutdownWriter {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            self.size += buf.len();
            Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    #[async_trait]
    impl Writer for GatedShutdownWriter {
        async fn tell(&mut self) -> Result<usize> {
            Ok(self.size)
        }

        async fn shutdown(&mut self) -> Result<WriteResult> {
            if let Some(shutdown_started) = self.shutdown_started.take() {
                shutdown_started
                    .send(())
                    .map_err(|_| Error::io("shutdown-start receiver was dropped"))?;
            }
            if let Some(resume_shutdown) = self.resume_shutdown.take() {
                resume_shutdown
                    .await
                    .map_err(|_| Error::io("shutdown-resume sender was dropped"))?;
            }
            std::fs::write(&self.publish_path, vec![0u8; self.size])?;
            Ok(WriteResult {
                size: self.size,
                e_tag: None,
            })
        }

        async fn abort(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_drop_spill_during_shutdown_cleans_publication_and_releases_cap() {
        let quota = DiskQuota::new(100);
        let temp_dir = Arc::new(tempfile::tempdir().unwrap());
        let fs_path = temp_dir.path().join("concurrent-spill.bin");
        let os_path = Path::from_absolute_path(&fs_path).unwrap();
        let lifecycle = Arc::new(SpillLifecycle::new(Some(quota.clone()), fs_path.clone()));
        let (shutdown_started_tx, shutdown_started_rx) = tokio::sync::oneshot::channel();
        let (resume_shutdown_tx, resume_shutdown_rx) = tokio::sync::oneshot::channel();
        let mut writer = SpillWriter {
            inner: Box::new(GatedShutdownWriter {
                shutdown_started: Some(shutdown_started_tx),
                resume_shutdown: Some(resume_shutdown_rx),
                publish_path: fs_path.clone(),
                size: 0,
            }),
            lifecycle: lifecycle.clone(),
        };
        let spill = LocalSpill {
            store: Arc::new(ObjectStore::local()),
            os_path,
            lifecycle,
            _temp_dir: temp_dir.clone(),
        };
        writer.write_all(&[0u8; 80]).await.unwrap();

        let shutdown = tokio::spawn(async move {
            let result = Writer::shutdown(&mut writer).await;
            (result, writer)
        });
        shutdown_started_rx.await.unwrap();
        drop(spill);
        resume_shutdown_tx.send(()).unwrap();

        let (result, _writer) = shutdown.await.unwrap();
        let error = result.unwrap_err();
        assert!(
            matches!(error, Error::IO { .. }),
            "expected IO error, got {error:?}"
        );
        assert!(
            error
                .to_string()
                .contains("paired Spill handle was dropped during shutdown"),
            "unexpected error: {error}"
        );
        assert!(
            !fs_path.exists(),
            "a concurrent owner drop must remove a file published by shutdown"
        );
        quota.try_reserve(80).unwrap();
    }

    #[tokio::test]
    async fn test_spill_writer_releases_unaccepted_bytes() {
        // Short write: the inner writer accepts only 10 of the 40 reserved bytes,
        // so the 30-byte remainder must be returned to the budget.
        let quota = DiskQuota::new(100);
        let (lifecycle, _temp_dir) = test_lifecycle(quota.clone());
        let mut writer = SpillWriter {
            inner: Box::new(ControlledWriter {
                outcome: Poll::Ready(Ok(10)),
                remaining_abort_failures: 0,
            }),
            lifecycle,
        };
        let n = writer.write(&[0u8; 40]).await.unwrap();
        assert_eq!(n, 10);
        assert_eq!(
            *quota.used.lock().unwrap(),
            10,
            "only the accepted bytes should remain reserved"
        );

        // Failed write: the full reservation must be released.
        let quota = DiskQuota::new(100);
        let (lifecycle, _temp_dir) = test_lifecycle(quota.clone());
        let mut writer = SpillWriter {
            inner: Box::new(ControlledWriter {
                outcome: Poll::Ready(Err(io::Error::other("boom"))),
                remaining_abort_failures: 0,
            }),
            lifecycle,
        };
        writer.write(&[0u8; 40]).await.unwrap_err();
        assert_eq!(
            *quota.used.lock().unwrap(),
            0,
            "a failed write should release its entire reservation"
        );
    }

    #[tokio::test]
    async fn test_spill_writer_retains_reservation_when_abort_fails() {
        let quota = DiskQuota::new(100);
        let (lifecycle, _temp_dir) = test_lifecycle(quota.clone());
        let mut writer = SpillWriter {
            inner: Box::new(ControlledWriter {
                outcome: Poll::Ready(Ok(40)),
                remaining_abort_failures: 1,
            }),
            lifecycle,
        };
        writer.write_all(&[0u8; 40]).await.unwrap();

        let error = Writer::abort(&mut writer).await.unwrap_err();
        assert!(
            matches!(error, Error::IO { .. }),
            "expected IO error, got {error:?}"
        );
        assert!(
            error.to_string().contains("controlled abort failure"),
            "unexpected error: {error}"
        );
        assert_eq!(
            *quota.used.lock().unwrap(),
            40,
            "failed abort must retain the reservation for a later cleanup attempt"
        );

        Writer::abort(&mut writer).await.unwrap();
        assert_eq!(
            *quota.used.lock().unwrap(),
            0,
            "a successful retry should release the retained reservation"
        );
    }
}
