// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Host-side adapter for dynamic cache backend plugins.

use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use futures::Future;
use lance_cache_abi::{CacheKey128, DynamicCacheBackendHandle};

use crate::Result;
use crate::error::Error;

use super::backend::{CacheBackend, CacheEntry};
use super::{CacheCodec, CacheDecode, InternalCacheKey, MokaCacheBackend};

/// Adapter from Lance's typed [`CacheBackend`] trait to a dynamic byte backend.
///
/// Dynamic plugins only store serialized bytes. This adapter keeps Lance-side
/// cache semantics in the host: entries without a [`CacheCodec`] stay in the
/// in-memory fallback, serialized entries are decoded before returning to
/// callers, and `get_or_insert` still deduplicates concurrent loads through
/// the fallback cache.
pub struct DynamicCacheBackendAdapter {
    backend: DynamicCacheBackendHandle,
    fallback: MokaCacheBackend,
    serialized_entries: AtomicUsize,
    serialized_size_bytes: AtomicUsize,
}

impl std::fmt::Debug for DynamicCacheBackendAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicCacheBackendAdapter")
            .field(
                "serialized_entries",
                &self.serialized_entries.load(Ordering::Relaxed),
            )
            .field(
                "serialized_size_bytes",
                &self.serialized_size_bytes.load(Ordering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

impl DynamicCacheBackendAdapter {
    /// Create a dynamic backend adapter with a host-side memory fallback.
    ///
    /// `fallback_capacity_bytes` bounds the entries that cannot be serialized
    /// yet (`codec == None`) and the in-process singleflight records used by
    /// [`CacheBackend::get_or_insert`].
    pub fn new(backend: DynamicCacheBackendHandle, fallback_capacity_bytes: usize) -> Self {
        Self {
            backend,
            fallback: MokaCacheBackend::with_capacity(fallback_capacity_bytes),
            serialized_entries: AtomicUsize::new(0),
            serialized_size_bytes: AtomicUsize::new(0),
        }
    }

    /// Load a named dynamic cache backend from a trusted native library.
    ///
    /// # Safety
    ///
    /// Loading native code is unsafe. The caller must trust the library at
    /// `path` and ensure it follows the xabi and Lance cache contracts.
    pub unsafe fn load_named(
        path: impl AsRef<Path>,
        name: &str,
        fallback_capacity_bytes: usize,
    ) -> Result<Self> {
        let backend =
            unsafe { lance_cache_abi::load_backend_named(path, name) }.map_err(|err| {
                Error::invalid_input(format!(
                    "failed to load dynamic cache backend {name:?}: {err}"
                ))
            })?;
        Ok(Self::new(backend, fallback_capacity_bytes))
    }

    fn plugin_key(key: &InternalCacheKey) -> CacheKey128 {
        CacheKey128::from_bytes(*key.as_bytes())
    }

    async fn get_serialized(
        &self,
        key: &InternalCacheKey,
        codec: CacheCodec,
    ) -> Option<CacheEntry> {
        let plugin_key = Self::plugin_key(key);
        let lookup = match self.backend.get(plugin_key).await {
            Ok(lookup) => lookup?,
            Err(error) => {
                log::warn!("dynamic cache backend get failed: {error}");
                return None;
            }
        };

        let bytes = Bytes::from(lookup.bytes);
        match codec.deserialize(&bytes) {
            CacheDecode::Hit(entry) => Some(entry),
            CacheDecode::Miss(reason) => {
                log::debug!("dynamic cache entry rejected for key {:?}: {reason:?}", key);
                None
            }
        }
    }

    async fn insert_serialized(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        codec: CacheCodec,
    ) -> bool {
        let mut bytes = Vec::new();
        if let Err(error) = codec.serialize(&entry, &mut bytes) {
            log::warn!(
                "dynamic cache entry serialization failed for key {:?}: {error}",
                key
            );
            return false;
        }

        let plugin_key = Self::plugin_key(key);
        if let Err(error) = self.backend.insert(plugin_key, &bytes, size_bytes).await {
            log::warn!("dynamic cache backend insert failed: {error}");
            return false;
        }

        self.serialized_entries.fetch_add(1, Ordering::Relaxed);
        self.serialized_size_bytes
            .fetch_add(size_bytes, Ordering::Relaxed);
        true
    }
}

#[async_trait]
impl CacheBackend for DynamicCacheBackendAdapter {
    async fn get(&self, key: &InternalCacheKey, codec: Option<CacheCodec>) -> Option<CacheEntry> {
        match codec {
            Some(codec) => self.get_serialized(key, codec).await,
            None => self.fallback.get(key, None).await,
        }
    }

    async fn insert(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        codec: Option<CacheCodec>,
    ) {
        match codec {
            Some(codec) => {
                self.insert_serialized(key, entry, size_bytes, codec).await;
            }
            None => self.fallback.insert(key, entry, size_bytes, None).await,
        }
    }

    async fn get_or_insert<'a>(
        &self,
        key: &InternalCacheKey,
        loader: Pin<Box<dyn Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>,
        codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)> {
        if let Some(entry) = self.get(key, codec).await {
            return Ok((entry, true));
        }

        let (entry, size_bytes, was_cached) =
            self.fallback.get_or_insert_record(key, loader).await?;

        if !was_cached
            && let Some(codec) = codec
            && self
                .insert_serialized(key, entry.clone(), size_bytes, codec)
                .await
        {
            self.fallback.invalidate_key(key).await;
        }

        Ok((entry, was_cached))
    }

    async fn clear(&self) {
        self.fallback.clear().await;
        if let Err(error) = self.backend.clear().await {
            log::warn!("dynamic cache backend clear failed: {error}");
        }
        self.serialized_entries.store(0, Ordering::Relaxed);
        self.serialized_size_bytes.store(0, Ordering::Relaxed);
    }

    async fn num_entries(&self) -> usize {
        match self.backend.measure().await {
            Ok(measure) => {
                self.serialized_entries
                    .store(measure.entries, Ordering::Relaxed);
                self.fallback
                    .num_entries()
                    .await
                    .saturating_add(measure.entries)
            }
            Err(error) => {
                log::warn!("dynamic cache backend measure failed: {error}");
                self.fallback.num_entries().await
            }
        }
    }

    async fn size_bytes(&self) -> usize {
        match self.backend.measure().await {
            Ok(measure) => {
                self.serialized_size_bytes
                    .store(measure.size_bytes, Ordering::Relaxed);
                self.fallback
                    .size_bytes()
                    .await
                    .saturating_add(measure.size_bytes)
            }
            Err(error) => {
                log::warn!("dynamic cache backend measure failed: {error}");
                self.fallback.size_bytes().await
            }
        }
    }

    fn approx_num_entries(&self) -> usize {
        self.fallback
            .approx_num_entries()
            .saturating_add(self.serialized_entries.load(Ordering::Relaxed))
    }

    fn approx_size_bytes(&self) -> usize {
        self.fallback
            .approx_size_bytes()
            .saturating_add(self.serialized_size_bytes.load(Ordering::Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::Arc;
    use std::sync::OnceLock;

    use super::*;
    use crate::cache::{CacheCodecImpl, CacheEntryReader, CacheEntryWriter};
    use crate::error::Error;

    static FIXTURE_LIBRARY: OnceLock<PathBuf> = OnceLock::new();

    #[derive(Debug, PartialEq)]
    struct Widget {
        n: u32,
    }

    impl CacheCodecImpl for Widget {
        const TYPE_ID: &'static str = "test.dynamic.Widget";
        const CURRENT_VERSION: u32 = 1;

        fn serialize(&self, writer: &mut CacheEntryWriter<'_>) -> Result<()> {
            writer.write_raw(&self.n.to_le_bytes())
        }

        fn deserialize(reader: &mut CacheEntryReader<'_>) -> Result<Self> {
            let bytes = reader.read_raw()?;
            let n = u32::from_le_bytes(
                bytes
                    .as_ref()
                    .try_into()
                    .map_err(|_| Error::io("invalid widget payload"))?,
            );
            Ok(Self { n })
        }
    }

    #[derive(Debug, PartialEq)]
    struct RuntimeOnly {
        label: String,
    }

    fn adapter() -> DynamicCacheBackendAdapter {
        unsafe {
            DynamicCacheBackendAdapter::load_named(fixture_library_path(), "memory", 1024 * 1024)
                .expect("fixture backend loads through xabi")
        }
    }

    fn key(discriminant: u8) -> InternalCacheKey {
        InternalCacheKey::from_bytes([discriminant; 16])
    }

    #[tokio::test]
    async fn codec_entries_round_trip_through_dynamic_backend() {
        let backend = adapter();
        let key = key(1);
        let codec = CacheCodec::from_impl::<Widget>();
        let entry: CacheEntry = Arc::new(Widget { n: 42 });

        backend.insert(&key, entry, 4, Some(codec)).await;

        let got = backend
            .get(&key, Some(codec))
            .await
            .expect("entry should hit")
            .downcast::<Widget>()
            .expect("entry should decode as Widget");
        assert_eq!(*got, Widget { n: 42 });
        assert_eq!(backend.num_entries().await, 1);
        assert_eq!(backend.size_bytes().await, 4);
    }

    #[tokio::test]
    async fn entries_without_codec_use_host_fallback() {
        let backend = adapter();
        let key = key(2);
        let entry: CacheEntry = Arc::new(RuntimeOnly {
            label: "host-only".to_string(),
        });

        backend.insert(&key, entry, 9, None).await;

        let got = backend
            .get(&key, None)
            .await
            .expect("fallback entry should hit")
            .downcast::<RuntimeOnly>()
            .expect("entry should stay typed in fallback");
        assert_eq!(
            *got,
            RuntimeOnly {
                label: "host-only".to_string()
            }
        );
        assert_eq!(backend.num_entries().await, 1);
    }

    #[tokio::test]
    async fn get_or_insert_reports_dynamic_hit_as_cached() {
        let backend = adapter();
        let key = key(3);
        let codec = CacheCodec::from_impl::<Widget>();
        let entry: CacheEntry = Arc::new(Widget { n: 7 });

        backend.insert(&key, entry, 4, Some(codec)).await;

        let (got, was_cached) = backend
            .get_or_insert(
                &key,
                Box::pin(async {
                    let entry: CacheEntry = Arc::new(Widget { n: 99 });
                    Ok((entry, 4))
                }),
                Some(codec),
            )
            .await
            .expect("get_or_insert succeeds");

        assert!(was_cached);
        assert_eq!(*got.downcast::<Widget>().unwrap(), Widget { n: 7 });
    }

    fn fixture_library_path() -> &'static Path {
        FIXTURE_LIBRARY.get_or_init(build_fixture_library).as_path()
    }

    fn build_fixture_library() -> PathBuf {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_dir = manifest_dir
            .ancestors()
            .nth(2)
            .expect("lance-core lives under rust/")
            .to_path_buf();
        let target_dir = workspace_dir.join("target").join("xabi-fixtures");
        let status = Command::new("cargo")
            .args(["build", "-p", "lance-cache-xabi-fixture", "--target-dir"])
            .arg(&target_dir)
            .args(["--message-format", "short"])
            .current_dir(&workspace_dir)
            .env_remove("RUSTC_WRAPPER")
            .env_remove("CARGO_TARGET_DIR")
            .status()
            .expect("cargo build can be launched");
        assert!(status.success(), "fixture cdylib build failed");

        let profile_dir = target_dir.join("debug");
        let library_path = profile_dir.join(dynamic_library_name("lance_cache_xabi_fixture"));
        assert!(
            library_path.exists(),
            "fixture cdylib was not built at {}",
            library_path.display()
        );
        library_path
    }

    fn dynamic_library_name(stem: &str) -> String {
        if cfg!(target_os = "macos") {
            format!("lib{stem}.dylib")
        } else if cfg!(target_os = "windows") {
            format!("{stem}.dll")
        } else {
            format!("lib{stem}.so")
        }
    }
}
