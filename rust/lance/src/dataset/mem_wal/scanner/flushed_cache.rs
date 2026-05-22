// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cache of opened flushed-generation datasets for the LSM scanner.
//!
//! Flushed generations are written exactly once to a globally-unique,
//! content-addressed path (see `memtable/flush.rs`): a fresh random hash per
//! flush invocation means the same path always maps to the same immutable
//! bytes. A cached `Arc<Dataset>` therefore can never go stale and needs no
//! TTL or invalidation for correctness — pruning entries is a pure memory
//! optimization driven by the consumer at compaction time.
//!
//! ```text
//! query ──> open_flushed_dataset(path, session, cache)
//!                                  │
//!            cache.is_some() ──────┤────── cache.is_none()
//!                  │                              │
//!     FlushedMemTableCache::get_or_open      DatasetBuilder::from_uri
//!     (single-flight, shared Arc)           (cold open every call)
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use lance_core::{Error, Result};

use crate::dataset::{Dataset, DatasetBuilder};
use crate::session::Session;

/// Cache of opened flushed-generation datasets, keyed by resolved path.
///
/// Flushed generations live at a globally-unique, immutable path, so cached
/// entries are never stale and require no TTL. Intended to be held by a
/// long-lived owner (one per process or per table) and injected into
/// per-request scanners via [`crate::dataset::mem_wal::scanner::LsmScanner`]
/// (and the point-lookup / vector-search planners).
///
/// The key is the resolved absolute flushed path
/// (`{base}/_mem_wal/{shard}/{folder}`), which is globally unique, so a single
/// cache can safely span multiple tables.
pub struct FlushedMemTableCache {
    // `moka`'s async cache gives a bounded size plus single-flight
    // `try_get_with`, so concurrent first-queries on a just-flushed
    // generation open the dataset exactly once.
    inner: moka::future::Cache<String, Arc<Dataset>>,
    // Per-generation set of PK hashes for the vector-search block-list, keyed by
    // the same immutable flushed path. Built lazily on the first query that needs
    // it (single-flight) so repeated searches skip re-scanning the PK column.
    pk_hashes: moka::future::Cache<String, Arc<HashSet<u64>>>,
}

impl FlushedMemTableCache {
    /// Create a cache holding at most `max_entries` opened datasets.
    ///
    /// Eviction is size-only (no TTL): an evicted-then-re-requested generation
    /// simply re-opens, which is always correct because the path is immutable.
    pub fn new(max_entries: u64) -> Self {
        Self {
            inner: moka::future::Cache::builder()
                .max_capacity(max_entries)
                // Required for `retain_paths`: moka silently ignores
                // `invalidate_entries_if` unless closure support is opted
                // into at build time.
                .support_invalidation_closures()
                .build(),
            pk_hashes: moka::future::Cache::builder()
                .max_capacity(max_entries)
                .support_invalidation_closures()
                .build(),
        }
    }

    /// Get the dataset for `path`, opening it (exactly once) on a miss.
    ///
    /// `session` is threaded into the open so the first open populates the
    /// shared index / file-metadata caches; subsequent hits are a pure
    /// `Arc::clone` with zero object-store I/O. Concurrent callers for the
    /// same path share a single open via `moka`'s single-flight
    /// `try_get_with`.
    pub async fn get_or_open(
        &self,
        path: &str,
        session: Option<Arc<Session>>,
    ) -> Result<Arc<Dataset>> {
        self.inner
            .try_get_with(path.to_string(), async move {
                let mut builder = DatasetBuilder::from_uri(path);
                if let Some(session) = session {
                    builder = builder.with_session(session);
                }
                builder.load().await.map(Arc::new)
            })
            .await
            // `try_get_with` hands losing racers an `Arc<Error>`; the original
            // error keeps full context, clones collapse to `Error::Cloned`.
            .map_err(|e: Arc<Error>| Error::cloned(e.to_string()))
    }

    /// Get the cached set of PK hashes for `path`, building it (exactly once) on
    /// a miss via `build`. The flushed path is immutable, so a cached set is
    /// never stale; concurrent first-queries share one build via `moka`'s
    /// single-flight `try_get_with`.
    pub async fn get_or_build_pk_hashes(
        &self,
        path: &str,
        build: impl std::future::Future<Output = Result<HashSet<u64>>>,
    ) -> Result<Arc<HashSet<u64>>> {
        self.pk_hashes
            .try_get_with(path.to_string(), async move { build.await.map(Arc::new) })
            .await
            .map_err(|e: Arc<Error>| Error::cloned(e.to_string()))
    }

    /// Drop cached entries whose path is not in `live_paths`.
    ///
    /// Called by the consumer after compaction retires generations. Purely a
    /// memory optimization: stale entries are unobservable because a retired
    /// generation's path never reappears in a shard snapshot, so correctness
    /// never depends on this running. Invalidation is applied lazily by
    /// `moka` during its next maintenance cycle.
    pub fn retain_paths(&self, live_paths: &HashSet<String>) {
        let live = live_paths.clone();
        // The only error is exceeding moka's registered-predicate cap, which
        // would just defer reclamation — never a correctness issue.
        let _ = self
            .inner
            .invalidate_entries_if(move |path, _| !live.contains(path));
        let live = live_paths.clone();
        let _ = self
            .pk_hashes
            .invalidate_entries_if(move |path, _| !live.contains(path));
    }
}

impl std::fmt::Debug for FlushedMemTableCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlushedMemTableCache")
            .field("entry_count", &self.inner.entry_count())
            .finish()
    }
}

/// Open a flushed-generation dataset, shared by all three LSM open sites
/// (scan, point lookup, vector search).
///
/// - `cache` present: route through [`FlushedMemTableCache`] (single-flight,
///   shared `Arc`, manifest read amortized across queries).
/// - `cache` absent: cold open via [`DatasetBuilder`]. Passing `session`
///   still reuses the shared index / metadata caches; `None`/`None`
///   reproduces the original per-query cold-open behavior exactly.
pub(super) async fn open_flushed_dataset(
    path: &str,
    session: Option<&Arc<Session>>,
    cache: Option<&Arc<FlushedMemTableCache>>,
) -> Result<Arc<Dataset>> {
    match cache {
        Some(cache) => cache.get_or_open(path, session.cloned()).await,
        None => {
            let mut builder = DatasetBuilder::from_uri(path);
            if let Some(session) = session {
                builder = builder.with_session(session.clone());
            }
            Ok(Arc::new(builder.load().await?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};

    use crate::dataset::WriteParams;

    async fn write_dataset(uri: &str, ids: &[i32]) {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(ids.to_vec()))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        Dataset::write(reader, uri, Some(WriteParams::default()))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_hit_returns_same_arc() {
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("{}/gen_1", temp_dir.path().to_str().unwrap());
        write_dataset(&uri, &[1, 2, 3]).await;

        let cache = FlushedMemTableCache::new(8);
        let first = cache.get_or_open(&uri, None).await.unwrap();
        let second = cache.get_or_open(&uri, None).await.unwrap();

        assert!(
            Arc::ptr_eq(&first, &second),
            "a cache hit must return the same Arc<Dataset>, not re-open"
        );
        assert_eq!(cache.inner.entry_count(), 0); // not yet flushed to count
        cache.inner.run_pending_tasks().await;
        assert_eq!(cache.inner.entry_count(), 1);
    }

    #[tokio::test]
    async fn test_concurrent_get_or_open_single_flight() {
        // moka's `try_get_with` must collapse concurrent first-queries on the
        // same path into exactly one open. We can't count opens through the
        // public API, so wrap the cache call: every task that observes the
        // same returned Arc proves they shared one open.
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("{}/gen_1", temp_dir.path().to_str().unwrap());
        write_dataset(&uri, &[1, 2, 3]).await;

        let cache = Arc::new(FlushedMemTableCache::new(8));
        let calls = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..16 {
            let cache = cache.clone();
            let uri = uri.clone();
            let calls = calls.clone();
            handles.push(tokio::spawn(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                cache.get_or_open(&uri, None).await.unwrap()
            }));
        }

        let datasets: Vec<Arc<Dataset>> = futures::future::try_join_all(handles).await.unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 16, "all tasks ran");
        let first = &datasets[0];
        for ds in &datasets {
            assert!(
                Arc::ptr_eq(first, ds),
                "all concurrent callers must share one opened dataset"
            );
        }
        cache.inner.run_pending_tasks().await;
        assert_eq!(cache.inner.entry_count(), 1, "exactly one entry cached");
    }

    #[tokio::test]
    async fn pk_hashes_cached_reuses_first_build() {
        // The PK-hash set is keyed by the immutable flushed path: a hit returns
        // the first-built set and never runs the second build closure.
        let cache = FlushedMemTableCache::new(8);
        let path = "memory://shard/gen_1";
        let first = cache
            .get_or_build_pk_hashes(path, async { Ok(HashSet::from([1u64, 2])) })
            .await
            .unwrap();
        let second = cache
            .get_or_build_pk_hashes(path, async {
                // Different contents; must be ignored because the path is cached.
                Ok(HashSet::from([9u64]))
            })
            .await
            .unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "a PK-hash cache hit must reuse the first-built set"
        );
        assert_eq!(
            second.len(),
            2,
            "cached set keeps the first build's contents"
        );
    }

    #[tokio::test]
    async fn test_retain_paths_drops_unreferenced() {
        let temp_dir = tempfile::tempdir().unwrap();
        let base = temp_dir.path().to_str().unwrap();
        let keep_uri = format!("{}/gen_1", base);
        let drop_uri = format!("{}/gen_2", base);
        write_dataset(&keep_uri, &[1]).await;
        write_dataset(&drop_uri, &[2]).await;

        let cache = FlushedMemTableCache::new(8);
        cache.get_or_open(&keep_uri, None).await.unwrap();
        cache.get_or_open(&drop_uri, None).await.unwrap();
        cache.inner.run_pending_tasks().await;
        assert_eq!(cache.inner.entry_count(), 2);

        let live: HashSet<String> = [keep_uri.clone()].into_iter().collect();
        cache.retain_paths(&live);
        cache.inner.run_pending_tasks().await;

        assert_eq!(cache.inner.entry_count(), 1, "only live path retained");
        assert!(cache.inner.contains_key(&keep_uri));
        assert!(!cache.inner.contains_key(&drop_uri));
    }

    #[tokio::test]
    async fn test_open_flushed_dataset_no_cache_matches_direct_open() {
        // The `None`/`None` path must reproduce a plain cold open: same data,
        // independent Arc per call (no caching).
        let temp_dir = tempfile::tempdir().unwrap();
        let uri = format!("{}/gen_1", temp_dir.path().to_str().unwrap());
        write_dataset(&uri, &[7, 8, 9]).await;

        let a = open_flushed_dataset(&uri, None, None).await.unwrap();
        let b = open_flushed_dataset(&uri, None, None).await.unwrap();
        assert!(
            !Arc::ptr_eq(&a, &b),
            "no-cache path must cold-open each call"
        );
        assert_eq!(a.count_rows(None).await.unwrap(), 3);

        // With a cache, the second call is a shared clone.
        let cache = Arc::new(FlushedMemTableCache::new(8));
        let c = open_flushed_dataset(&uri, None, Some(&cache))
            .await
            .unwrap();
        let d = open_flushed_dataset(&uri, None, Some(&cache))
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&c, &d), "cached path must reuse the Arc");
    }
}
