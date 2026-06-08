// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Helpers for converting between [`Dataset`] and [`TableIdentifier`](pb::TableIdentifier) proto.

use std::sync::{Arc, LazyLock};
use std::time::Duration;

use lance_core::{Error, Result};
use lance_datafusion::pb;
use lance_io::object_store::StorageOptions;
use prost::Message;

use crate::Dataset;
use crate::dataset::builder::DatasetBuilder;

/// Cache key for a dataset opened from a serialized proto. A pinned dataset
/// version is an immutable snapshot, uniquely identified by
/// `(uri, version, manifest_etag)`.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct DatasetCacheKey {
    uri: String,
    version: u64,
    etag: Option<String>,
}

/// Max distinct `(uri, version, etag)` datasets cached per process. Override
/// via `LANCE_PROTO_DATASET_CACHE_SIZE`; set to `0` to disable caching and
/// restore cold-open-per-plan_run behavior.
static PROTO_DATASET_CACHE_CAPACITY: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("LANCE_PROTO_DATASET_CACHE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64)
});

/// Idle eviction window. Re-opening an evicted snapshot is always correct (the
/// version is immutable); this just bounds memory and how long the storage
/// credentials captured at first open are retained.
const PROTO_DATASET_CACHE_TTI: Duration = Duration::from_secs(300);

/// Process-global cache of datasets opened from serialized protos.
///
/// Serialized `FilteredReadExec` / ANN protos cannot carry a live
/// `Arc<Dataset>`, so without this every distributed plan_run cold-opens the
/// dataset from scratch (full `ObjectStore` init + manifest GET). On a
/// many-fragment table that is thousands of redundant opens; the cache collapses
/// them to one per `(uri, version, etag)` per worker process, and moka's
/// single-flight `try_get_with` coalesces concurrent first-misses into one open.
///
/// Correctness: reusing a cached `Arc<Dataset>` for a pinned version is always
/// safe — different versions/manifests get distinct keys, and a new write
/// produces a new version (and etag) rather than mutating a cached entry, so the
/// cache never serves stale data, breaks version pinning, or affects time-travel
/// reads. Storage options are intentionally excluded from the key: credentials
/// may be re-vended per plan_run, but the bytes (identified by etag) are the
/// same, so a cached entry reuses the `ObjectStore` configured at first open.
static PROTO_DATASET_CACHE: LazyLock<moka::future::Cache<DatasetCacheKey, Arc<Dataset>>> =
    LazyLock::new(|| {
        moka::future::Cache::builder()
            .max_capacity(*PROTO_DATASET_CACHE_CAPACITY)
            .time_to_idle(PROTO_DATASET_CACHE_TTI)
            .build()
    });

/// Build a [`TableIdentifier`] from a [`Dataset`].
///
/// Default: lightweight mode (uri + version + etag only, no serialized manifest).
/// Includes the dataset's latest storage options (if any) so the remote executor
/// can open or cache the dataset with the correct storage configuration.
pub async fn table_identifier_from_dataset(dataset: &Dataset) -> Result<pb::TableIdentifier> {
    Ok(pb::TableIdentifier {
        uri: dataset.uri().to_string(),
        version: dataset.manifest.version,
        manifest_etag: dataset.manifest_location.e_tag.clone(),
        serialized_manifest: None,
        storage_options: dataset
            .latest_storage_options()
            .await?
            .map(|StorageOptions(m)| m)
            .unwrap_or_default(),
    })
}

/// Build a [`TableIdentifier`] with serialized manifest bytes included.
///
/// Fast path: remote executor skips manifest read from storage.
pub async fn table_identifier_from_dataset_with_manifest(
    dataset: &Dataset,
) -> Result<pb::TableIdentifier> {
    let manifest_proto = lance_table::format::pb::Manifest::from(dataset.manifest.as_ref());
    Ok(pb::TableIdentifier {
        uri: dataset.uri().to_string(),
        version: dataset.manifest.version,
        manifest_etag: dataset.manifest_location.e_tag.clone(),
        serialized_manifest: Some(manifest_proto.encode_to_vec()),
        storage_options: dataset
            .latest_storage_options()
            .await?
            .map(|StorageOptions(m)| m)
            .unwrap_or_default(),
    })
}

/// Open a dataset from a table identifier proto.
pub async fn open_dataset_from_table_identifier(
    table_id: &pb::TableIdentifier,
) -> Result<Arc<Dataset>> {
    let mut builder = DatasetBuilder::from_uri(&table_id.uri).with_version(table_id.version);
    if let Some(manifest_bytes) = &table_id.serialized_manifest {
        builder = builder.with_serialized_manifest(manifest_bytes)?;
    }
    if !table_id.storage_options.is_empty() {
        builder = builder.with_storage_options(table_id.storage_options.clone());
    }
    Ok(Arc::new(builder.load().await?))
}

/// Resolve a dataset from an optional pre-loaded instance or from a table identifier.
///
/// If `dataset` is `Some`, returns it directly. Otherwise opens the dataset from
/// the table identifier proto, sharing one `Arc<Dataset>` per
/// `(uri, version, etag)` across the worker process via [`PROTO_DATASET_CACHE`]
/// (unless caching is disabled with `LANCE_PROTO_DATASET_CACHE_SIZE=0`).
pub async fn resolve_dataset(
    dataset: Option<Arc<Dataset>>,
    table_id: Option<&pb::TableIdentifier>,
) -> Result<Arc<Dataset>> {
    match dataset {
        Some(ds) => Ok(ds),
        None => {
            let table_id = table_id.ok_or_else(|| {
                Error::invalid_input_source("Missing TableIdentifier in proto".into())
            })?;
            if *PROTO_DATASET_CACHE_CAPACITY == 0 {
                // Caching disabled: preserve cold-open-per-call behavior.
                return open_dataset_from_table_identifier(table_id).await;
            }
            resolve_dataset_cached(&PROTO_DATASET_CACHE, table_id).await
        }
    }
}

/// Open a dataset for `table_id` through `cache`, sharing one open per
/// `(uri, version, etag)`. Split out so tests can inject an isolated cache.
async fn resolve_dataset_cached(
    cache: &moka::future::Cache<DatasetCacheKey, Arc<Dataset>>,
    table_id: &pb::TableIdentifier,
) -> Result<Arc<Dataset>> {
    resolve_dataset_cached_with(cache, table_id, |tid| async move {
        open_dataset_from_table_identifier(&tid).await
    })
    .await
}

/// [`resolve_dataset_cached`] with an injectable `open` for the cache-miss
/// path. The production opener is [`open_dataset_from_table_identifier`]; tests
/// substitute a counting opener to assert how many times a serialized proto
/// actually re-opens the dataset (each open == one manifest read == one
/// `lance::events dataset_events event="loading"`).
async fn resolve_dataset_cached_with<Open, Fut>(
    cache: &moka::future::Cache<DatasetCacheKey, Arc<Dataset>>,
    table_id: &pb::TableIdentifier,
    open: Open,
) -> Result<Arc<Dataset>>
where
    Open: FnOnce(pb::TableIdentifier) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<Arc<Dataset>>> + Send + 'static,
{
    let key = DatasetCacheKey {
        uri: table_id.uri.clone(),
        version: table_id.version,
        etag: table_id.manifest_etag.clone(),
    };
    // Own the proto so the single-flight init future is self-contained.
    let table_id = table_id.clone();
    cache
        .try_get_with(key, async move {
            // Logged at the open (cache-miss) site so a heavy distributed plan
            // is greppable without enabling `lance::events` tracing.
            log::info!(
                "resolve_dataset: opening dataset (cache miss) uri={} version={}",
                table_id.uri,
                table_id.version
            );
            open(table_id).await
        })
        .await
        // `try_get_with` hands losing racers an `Arc<Error>`; collapse it back
        // to an owned `Error` (full context on the winner, `Error::Cloned`
        // otherwise).
        .map_err(|e: Arc<Error>| Error::cloned(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::RecordBatchIterator;
    use arrow_array::types::UInt32Type;
    use lance_datagen::{array, gen_batch};
    use std::collections::HashMap;

    async fn make_test_dataset() -> (Arc<Dataset>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let batch = gen_batch()
            .col("x", array::step::<UInt32Type>())
            .col("y", array::step::<UInt32Type>())
            .into_batch_rows(lance_datagen::RowCount::from(100))
            .unwrap();
        let path = dir.path().join("test.lance");
        let ds = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            path.to_str().unwrap(),
            None,
        )
        .await
        .unwrap();
        (Arc::new(ds), dir)
    }

    #[test]
    fn test_table_identifier_proto_roundtrip() {
        let id = pb::TableIdentifier {
            uri: "s3://bucket/table.lance".to_string(),
            version: 42,
            manifest_etag: Some("etag123".to_string()),
            serialized_manifest: None,
            storage_options: HashMap::new(),
        };
        let bytes = id.encode_to_vec();
        let back = pb::TableIdentifier::decode(bytes.as_slice()).unwrap();
        assert_eq!(id.uri, back.uri);
        assert_eq!(id.version, back.version);
        assert_eq!(id.manifest_etag, back.manifest_etag);
        assert!(back.serialized_manifest.is_none());
    }

    #[test]
    fn test_table_identifier_proto_with_storage_options() {
        let mut opts = HashMap::new();
        opts.insert("region".to_string(), "us-east-1".to_string());
        opts.insert("endpoint".to_string(), "https://s3.example.com".to_string());

        let id = pb::TableIdentifier {
            uri: "s3://bucket/table.lance".to_string(),
            version: 7,
            manifest_etag: None,
            serialized_manifest: None,
            storage_options: opts.clone(),
        };
        let bytes = id.encode_to_vec();
        let back = pb::TableIdentifier::decode(bytes.as_slice()).unwrap();
        assert_eq!(back.storage_options, opts);
    }

    #[tokio::test]
    async fn test_table_identifier_from_dataset_roundtrip() {
        let (dataset, _dir) = make_test_dataset().await;

        let id = table_identifier_from_dataset(&dataset).await.unwrap();
        assert_eq!(id.uri, dataset.uri());
        assert_eq!(id.version, dataset.manifest.version);
        assert!(id.serialized_manifest.is_none());

        // Roundtrip: open the dataset back from the identifier
        let back = open_dataset_from_table_identifier(&id).await.unwrap();
        assert_eq!(back.uri(), dataset.uri());
        assert_eq!(back.manifest.version, dataset.manifest.version);
    }

    #[tokio::test]
    async fn test_table_identifier_with_manifest_roundtrip() {
        let (dataset, _dir) = make_test_dataset().await;

        let id = table_identifier_from_dataset_with_manifest(&dataset)
            .await
            .unwrap();
        assert_eq!(id.uri, dataset.uri());
        assert_eq!(id.version, dataset.manifest.version);
        assert!(id.serialized_manifest.is_some());

        // Verify the serialized manifest bytes decode
        let manifest_bytes = id.serialized_manifest.as_ref().unwrap();
        let _manifest_proto =
            lance_table::format::pb::Manifest::decode(manifest_bytes.as_slice()).unwrap();

        // Roundtrip: open the dataset back from the identifier (with manifest)
        let back = open_dataset_from_table_identifier(&id).await.unwrap();
        assert_eq!(back.uri(), dataset.uri());
        assert_eq!(back.manifest.version, dataset.manifest.version);
    }

    fn test_cache() -> moka::future::Cache<DatasetCacheKey, Arc<Dataset>> {
        moka::future::Cache::builder().max_capacity(16).build()
    }

    #[tokio::test]
    async fn test_resolve_dataset_cached_returns_same_arc() {
        let (dataset, _dir) = make_test_dataset().await;
        let id = table_identifier_from_dataset(&dataset).await.unwrap();

        let cache = test_cache();
        let a = resolve_dataset_cached(&cache, &id).await.unwrap();
        let b = resolve_dataset_cached(&cache, &id).await.unwrap();

        assert!(
            Arc::ptr_eq(&a, &b),
            "a cache hit must reuse the Arc<Dataset>, not re-open"
        );
        assert_eq!(a.manifest.version, dataset.manifest.version);

        cache.run_pending_tasks().await;
        assert_eq!(cache.entry_count(), 1);
    }

    #[tokio::test]
    async fn test_resolve_dataset_cached_single_flight() {
        // moka's `try_get_with` must collapse concurrent first-misses on the
        // same (uri, version, etag) into exactly one open.
        let (dataset, _dir) = make_test_dataset().await;
        let id = table_identifier_from_dataset(&dataset).await.unwrap();

        let cache = Arc::new(test_cache());
        let mut handles = Vec::new();
        for _ in 0..16 {
            let cache = cache.clone();
            let id = id.clone();
            handles.push(tokio::spawn(async move {
                resolve_dataset_cached(&cache, &id).await.unwrap()
            }));
        }
        let datasets: Vec<Arc<Dataset>> = futures::future::try_join_all(handles).await.unwrap();

        let first = &datasets[0];
        for ds in &datasets {
            assert!(
                Arc::ptr_eq(first, ds),
                "all concurrent callers must share one opened dataset"
            );
        }
        cache.run_pending_tasks().await;
        assert_eq!(cache.entry_count(), 1, "exactly one entry cached");
    }

    #[tokio::test]
    async fn test_resolve_dataset_cached_distinct_versions() {
        use crate::dataset::{WriteMode, WriteParams};

        let (v1, _dir) = make_test_dataset().await;
        let uri = v1.uri().to_string();

        // Append to create version 2 at the same uri.
        let batch = gen_batch()
            .col("x", array::step::<UInt32Type>())
            .col("y", array::step::<UInt32Type>())
            .into_batch_rows(lance_datagen::RowCount::from(50))
            .unwrap();
        let v2 = Arc::new(
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
                &uri,
                Some(WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        assert!(v2.manifest.version > v1.manifest.version);

        let id1 = table_identifier_from_dataset(&v1).await.unwrap();
        let id2 = table_identifier_from_dataset(&v2).await.unwrap();

        let cache = test_cache();
        let a = resolve_dataset_cached(&cache, &id1).await.unwrap();
        let b = resolve_dataset_cached(&cache, &id2).await.unwrap();

        // Version pinning is preserved: different versions never alias.
        assert!(!Arc::ptr_eq(&a, &b));
        assert_eq!(a.manifest.version, v1.manifest.version);
        assert_eq!(b.manifest.version, v2.manifest.version);

        cache.run_pending_tasks().await;
        assert_eq!(cache.entry_count(), 2);
    }

    #[tokio::test]
    async fn test_resolve_dataset_some_returns_passed_arc() {
        // The `Some` arm bypasses the cache and returns the supplied handle.
        let (dataset, _dir) = make_test_dataset().await;
        let out = resolve_dataset(Some(dataset.clone()), None).await.unwrap();
        assert!(Arc::ptr_eq(&out, &dataset));
    }

    /// Deterministic, non-cloud repro of the per-plan re-open amplification.
    ///
    /// A serialized `FilteredReadExec`/ANN proto carries no live `Arc<Dataset>`,
    /// so each distributed plan_run deserialized on a worker calls
    /// `DatasetBuilder::load()` from scratch — one `ObjectStore` init + manifest
    /// read per plan_run (one `dataset_events event="loading"` each). On a
    /// many-fragment table that is thousands of redundant opens. We count opens
    /// by instrumenting the cache-miss opener: one opener call == one such load.
    /// With the cache, N plan_runs collapse to a single open per
    /// `(uri, version, etag)`.
    #[tokio::test]
    async fn test_resolve_dataset_opens_once_across_plan_runs() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        const PLAN_RUNS: usize = 50;

        let (dataset, _dir) = make_test_dataset().await;
        let id = table_identifier_from_dataset(&dataset).await.unwrap();

        // --- Buggy baseline: no cache => one cold open per plan_run. ---
        let cold_opens = Arc::new(AtomicUsize::new(0));
        for _ in 0..PLAN_RUNS {
            cold_opens.fetch_add(1, Ordering::SeqCst);
            let ds = open_dataset_from_table_identifier(&id).await.unwrap();
            assert_eq!(ds.manifest.version, dataset.manifest.version);
        }
        assert_eq!(
            cold_opens.load(Ordering::SeqCst),
            PLAN_RUNS,
            "without the cache, every plan_run cold-opens (the bug)"
        );

        // --- Fixed: the cache collapses N plan_runs to a single open. ---
        let cache = test_cache();
        let opens = Arc::new(AtomicUsize::new(0));
        for _ in 0..PLAN_RUNS {
            let opens = opens.clone();
            let ds = resolve_dataset_cached_with(&cache, &id, move |tid| async move {
                opens.fetch_add(1, Ordering::SeqCst);
                open_dataset_from_table_identifier(&tid).await
            })
            .await
            .unwrap();
            assert_eq!(ds.manifest.version, dataset.manifest.version);
        }
        assert_eq!(
            opens.load(Ordering::SeqCst),
            1,
            "with the cache, {PLAN_RUNS} plan_runs open the dataset exactly once"
        );

        // Concurrent first-misses must also collapse to one open (single-flight).
        let cache = Arc::new(test_cache());
        let concurrent_opens = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..PLAN_RUNS {
            let cache = cache.clone();
            let id = id.clone();
            let concurrent_opens = concurrent_opens.clone();
            handles.push(tokio::spawn(async move {
                resolve_dataset_cached_with(&cache, &id, move |tid| async move {
                    concurrent_opens.fetch_add(1, Ordering::SeqCst);
                    open_dataset_from_table_identifier(&tid).await
                })
                .await
                .unwrap()
            }));
        }
        futures::future::try_join_all(handles).await.unwrap();
        assert_eq!(
            concurrent_opens.load(Ordering::SeqCst),
            1,
            "concurrent plan_runs single-flight to one open"
        );
    }
}
