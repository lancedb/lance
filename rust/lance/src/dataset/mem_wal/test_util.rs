// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Test-only object store that injects WAL-write failures (for the WAL
//! persistence-failure fencing path) and records the paths it serves (for
//! asserting which opens actually resolved through a given `ObjectStoreParams`).

use std::fmt::{Debug, Display, Formatter};
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use bytes::Bytes;
use futures::stream::BoxStream;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore as OSObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    RenameOptions, Result as OSResult,
};

use lance_io::object_store::{
    ObjectStore, ObjectStoreParams, ObjectStoreRegistry, WrappingObjectStore,
};

/// Knobs controlling injected WAL-entry write failures, shared with the test.
#[derive(Debug, Default)]
pub struct FailControls {
    /// Remaining WAL-entry `put_opts` calls to fail; saturates at 0. Set high to
    /// "always fail", set to 0 to "recover".
    wal_put_failures: AtomicUsize,
    /// Remaining SSTable (generation directory) writes to fail; saturates at 0.
    /// Drives the L0-flush failure path, as `wal_put_failures` does the WAL.
    sstable_put_failures: AtomicUsize,
    /// When failing, write to the inner store anyway before reporting the error,
    /// simulating a lost acknowledgement (the PUT actually landed).
    simulate_lost_ack: AtomicBool,
    /// WAL-entry `put_opts` attempts observed, for assertions.
    wal_put_attempts: AtomicUsize,
    /// Every location written through this store.
    put_paths: StdMutex<Vec<String>>,
    /// Every location read through this store.
    get_paths: StdMutex<Vec<String>>,
}

impl FailControls {
    pub fn fail_wal_puts(&self, n: usize) {
        self.wal_put_failures.store(n, Ordering::SeqCst);
    }
    pub fn fail_sstable_puts(&self, n: usize) {
        self.sstable_put_failures.store(n, Ordering::SeqCst);
    }
    /// Storage heals: every injected failure mode stops firing.
    pub fn recover(&self) {
        self.wal_put_failures.store(0, Ordering::SeqCst);
        self.sstable_put_failures.store(0, Ordering::SeqCst);
    }
    pub fn set_lost_ack(&self, v: bool) {
        self.simulate_lost_ack.store(v, Ordering::SeqCst);
    }
    pub fn attempts(&self) -> usize {
        self.wal_put_attempts.load(Ordering::SeqCst)
    }

    /// Did any write land on a path containing `needle`? An open that resolved
    /// its store from other params never reaches this store, so a `false` here
    /// means the params under test did not reach that open.
    pub fn wrote_under(&self, needle: &str) -> bool {
        self.put_paths
            .lock()
            .unwrap()
            .iter()
            .any(|p| p.contains(needle))
    }

    /// Distinct `{hash}_gen_{n}` directory segments written through this store.
    /// A generation whose flush retries must reuse one: a fresh directory per
    /// attempt strands everything the previous attempt wrote.
    pub fn generation_dirs(&self) -> std::collections::BTreeSet<String> {
        self.put_paths
            .lock()
            .unwrap()
            .iter()
            .flat_map(|p| p.split('/'))
            .filter(|segment| segment.contains("_gen_"))
            .map(str::to_string)
            .collect()
    }

    /// Did any read land on a path containing `needle`? See [`Self::wrote_under`].
    pub fn read_under(&self, needle: &str) -> bool {
        self.get_paths
            .lock()
            .unwrap()
            .iter()
            .any(|p| p.contains(needle))
    }
}

/// Wraps the inner store with [`FailingObjectStore`] at construction.
#[derive(Debug)]
struct FailingWrapper {
    controls: Arc<FailControls>,
}

impl WrappingObjectStore for FailingWrapper {
    fn wrap(&self, _prefix: &str, original: Arc<dyn OSObjectStore>) -> Arc<dyn OSObjectStore> {
        Arc::new(FailingObjectStore {
            inner: original,
            controls: self.controls.clone(),
        })
    }
}

/// Delegates everything to `inner`, failing WAL-entry and SSTable writes per
/// [`FailControls`].
#[derive(Debug)]
struct FailingObjectStore {
    inner: Arc<dyn OSObjectStore>,
    controls: Arc<FailControls>,
}

impl FailingObjectStore {
    /// WAL entries live under `.../wal/`; manifests (`.../manifest/`) are never
    /// failed so `ShardWriter::open`/`claim_epoch` still work.
    fn is_wal_entry(location: &Path) -> bool {
        location.as_ref().contains("/wal/")
    }

    /// An L0 flush writes everything for a generation under its
    /// `{hash}_gen_{n}` directory (see `sstable_path`): data file, dataset
    /// manifest, bloom filter, indexes.
    fn is_sstable_write(location: &Path) -> bool {
        location.as_ref().contains("_gen_")
    }

    fn injected_error(what: &'static str) -> object_store::Error {
        object_store::Error::Generic {
            store: "failing-test-store",
            source: format!("injected transient {what} put failure").into(),
        }
    }

    /// Consume one armed SSTable failure, if this write is one and any remain.
    /// Both entry points call it: the data file goes out multipart, the
    /// manifest and bloom filter as plain PUTs.
    fn take_sstable_failure(&self, location: &Path) -> Option<object_store::Error> {
        if !Self::is_sstable_write(location)
            || self.controls.sstable_put_failures.load(Ordering::SeqCst) == 0
        {
            return None;
        }
        self.controls
            .sstable_put_failures
            .fetch_sub(1, Ordering::SeqCst);
        Some(Self::injected_error("SSTable"))
    }
}

impl Display for FailingObjectStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "FailingObjectStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl OSObjectStore for FailingObjectStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        self.controls
            .put_paths
            .lock()
            .unwrap()
            .push(location.to_string());
        if Self::is_wal_entry(location) {
            self.controls
                .wal_put_attempts
                .fetch_add(1, Ordering::SeqCst);
            if self.controls.wal_put_failures.load(Ordering::SeqCst) > 0 {
                self.controls
                    .wal_put_failures
                    .fetch_sub(1, Ordering::SeqCst);
                if self.controls.simulate_lost_ack.load(Ordering::SeqCst) {
                    // The write lands but we still report failure (lost ack).
                    let _ = self.inner.put_opts(location, payload, opts).await;
                }
                return Err(Self::injected_error("WAL"));
            }
        }
        if let Some(e) = self.take_sstable_failure(location) {
            return Err(e);
        }
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        // Data files (`*.lance`) are written multipart, not via `put_opts`.
        self.controls
            .put_paths
            .lock()
            .unwrap()
            .push(location.to_string());
        if let Some(e) = self.take_sstable_failure(location) {
            return Err(e);
        }
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        self.controls
            .get_paths
            .lock()
            .unwrap()
            .push(location.to_string());
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
        self.controls
            .get_paths
            .lock()
            .unwrap()
            .push(location.to_string());
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, OSResult<Path>>,
    ) -> BoxStream<'static, OSResult<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
        self.inner.copy_opts(from, to, opts).await
    }

    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> OSResult<()> {
        self.inner.rename_opts(from, to, opts).await
    }
}

/// `ObjectStoreParams` carrying the observable store wrapper, plus the controls
/// to drive and inspect it. Open a dataset with these and every store resolved
/// *from these params* — the base and any derived URI they are threaded to —
/// reports its traffic back through the controls.
pub fn observable_store_params() -> (ObjectStoreParams, Arc<FailControls>) {
    let controls = Arc::new(FailControls::default());
    let params = ObjectStoreParams {
        object_store_wrapper: Some(Arc::new(FailingWrapper {
            controls: controls.clone(),
        })),
        ..Default::default()
    };
    (params, controls)
}

/// Build an in-memory `ObjectStore` whose WAL-entry writes can be failed on
/// demand. Returns the store, its base path, and the shared controls.
pub async fn failing_memory_store() -> (Arc<ObjectStore>, Path, Arc<FailControls>) {
    let (params, controls) = observable_store_params();
    let (store, base) = ObjectStore::from_uri_and_params(
        Arc::new(ObjectStoreRegistry::default()),
        "memory:///",
        &params,
    )
    .await
    .unwrap();
    (store, base, controls)
}
