// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Experimental xabi contract for dynamically loaded Lance cache backends.
//!
//! This prototype keeps the host-side cache policy out of the low-level ABI:
//! Lance remains responsible for typed cache entries, `codec == None` fallback,
//! singleflight, metrics, registration, and lifecycle policy. Dynamic backends
//! only receive an async Rust-shaped contract for serialized cache bytes.

use std::path::Path;

/// Stable xabi contract identifier for Lance cache backends.
pub const DYNAMIC_CACHE_BACKEND_TRAIT_ID: &str = "org.lance.cache.DynamicCacheBackend";

/// Prototype contract version.
///
/// Keep this at `0` while the key model and backend lifecycle are still being
/// validated. The generated xabi type names use an `XabiV1` prefix for xabi's
/// own wire-generation format; that is separate from this Lance contract
/// version.
pub const DYNAMIC_CACHE_BACKEND_CONTRACT_VERSION: u32 = 0;

/// A cache key represented as Lance's canonical opaque 16-byte value.
///
/// The byte representation is the contract. Dynamic backends should persist
/// and compare these bytes directly instead of interpreting them as host-native
/// integers or reconstructing logical key components.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CacheKey128 {
    bytes: CacheKey128Wire,
}

/// Wire representation of a cache key across the xabi boundary.
pub type CacheKey128Wire = [u8; 16];

impl CacheKey128 {
    /// Build a key from its canonical bytes.
    pub const fn from_bytes(bytes: CacheKey128Wire) -> Self {
        Self { bytes }
    }

    /// Borrow the canonical bytes.
    pub const fn as_bytes(&self) -> &CacheKey128Wire {
        &self.bytes
    }

    /// Consume the key and return its canonical bytes.
    pub const fn into_bytes(self) -> CacheKey128Wire {
        self.bytes
    }
}

impl xabi::XabiType for CacheKey128 {
    type Wire = CacheKey128Wire;
    const WIRE_TYPE_NAME: &'static str = "CacheKey128Wire";

    fn into_wire(self) -> Self::Wire {
        self.bytes
    }

    unsafe fn from_wire(wire: *const Self::Wire) -> xabi::Result<Self> {
        let bytes = unsafe {
            wire.as_ref()
                .copied()
                .ok_or(xabi::Error::NullPointer("CacheKey128Wire pointer"))?
        };
        Ok(Self::from_bytes(bytes))
    }

    fn collect_xabi_layout(collector: &mut dyn xabi::XabiLayoutCollector) {
        const FIELDS: &[xabi::XabiFieldLayout] =
            &[xabi::XabiFieldLayout::new("bytes", 0, "[u8; 16]")];
        collector.push(xabi::XabiLayoutItem::Type(xabi::XabiTypeLayout::new(
            concat!(module_path!(), "::CacheKey128Wire"),
            xabi::XabiLayoutStability::Fixed,
            std::mem::size_of::<CacheKey128Wire>(),
            std::mem::align_of::<CacheKey128Wire>(),
            FIELDS,
        )));
    }
}

/// Serialized cache entry bytes returned by a dynamic backend.
#[xabi::data]
#[derive(Debug, Eq, PartialEq)]
pub struct CacheLookup {
    /// Serialized payload bytes.
    pub bytes: Vec<u8>,
    /// Host-side cache weight originally associated with this entry.
    pub size_bytes: usize,
}

/// Approximate backend size metrics.
#[xabi::data]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheMeasure {
    /// Approximate number of stored entries.
    pub entries: usize,
    /// Approximate weighted size in bytes.
    pub size_bytes: usize,
}

/// Typed error payload returned by cache backend implementations.
#[xabi::data]
#[derive(Debug, Eq, PartialEq)]
pub struct CacheAbiError {
    /// Human-readable error message.
    pub message: String,
}

impl From<xabi::Error> for CacheAbiError {
    fn from(value: xabi::Error) -> Self {
        Self::new(value.to_string())
    }
}

impl From<xabi::XabiCallError<Self>> for CacheAbiError {
    fn from(value: xabi::XabiCallError<Self>) -> Self {
        match value {
            xabi::XabiCallError::Runtime(err) => Self::from(err),
            xabi::XabiCallError::Export(err) => err,
        }
    }
}

impl std::fmt::Display for CacheAbiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CacheAbiError {}

/// Async cache backend contract generated through xabi.
///
/// This deliberately omits `invalidate_prefix`; prefix lifecycle remains a host
/// adapter concern while the cache key contract settles on opaque bytes.
#[xabi::xabi(
    id = DYNAMIC_CACHE_BACKEND_TRAIT_ID,
    version = DYNAMIC_CACHE_BACKEND_CONTRACT_VERSION
)]
pub trait DynamicCacheBackend {
    /// Human-readable backend implementation name.
    fn name(&self) -> String;

    /// Look up serialized bytes by key.
    async fn get(
        &self,
        key: CacheKey128,
    ) -> std::result::Result<Option<CacheLookup>, CacheAbiError>;

    /// Insert serialized bytes by key.
    async fn insert(
        &self,
        key: CacheKey128,
        value: &[u8],
        size_bytes: usize,
    ) -> std::result::Result<(), CacheAbiError>;

    /// Clear all entries owned by this backend instance.
    async fn clear(&self) -> std::result::Result<(), CacheAbiError>;

    /// Return approximate backend size metrics.
    async fn measure(&self) -> std::result::Result<CacheMeasure, CacheAbiError>;
}

/// Host-side handle for a dynamically loaded cache backend.
pub type DynamicCacheBackendHandle = XabiV1HandleTraitDynamicCacheBackend;

/// Owned in-process backend handle, useful for tests and host-side adapters.
pub type OwnedDynamicCacheBackend = XabiV1OwnedTraitDynamicCacheBackend;

/// Borrowed in-process backend handle.
pub type BorrowedDynamicCacheBackend = XabiV1BorrowedTraitDynamicCacheBackend;

/// Generated xabi contract descriptor.
pub type DynamicCacheBackendAbi = XabiV1AbiTraitDynamicCacheBackend;

/// Load the first matching cache backend export from a trusted dynamic library.
///
/// # Safety
///
/// Loading native code is unsafe. The caller must trust the library at `path`
/// and ensure the loaded module follows the xabi and Lance cache contracts.
pub unsafe fn load_backend(path: impl AsRef<Path>) -> xabi::Result<DynamicCacheBackendHandle> {
    let module = unsafe { xabi::load(path) }?;
    DynamicCacheBackendHandle::xabi_load(&module)
}

/// Load a named cache backend export from a trusted dynamic library.
///
/// # Safety
///
/// Loading native code is unsafe. The caller must trust the library at `path`
/// and ensure the loaded module follows the xabi and Lance cache contracts.
pub unsafe fn load_backend_named(
    path: impl AsRef<Path>,
    name: &str,
) -> xabi::Result<DynamicCacheBackendHandle> {
    let module = unsafe { xabi::load(path) }?;
    DynamicCacheBackendHandle::xabi_load_named(&module, name)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use super::*;

    #[derive(Default)]
    struct InMemoryBackend {
        entries: Mutex<HashMap<CacheKey128, (Vec<u8>, usize)>>,
    }

    impl DynamicCacheBackend for InMemoryBackend {
        fn name(&self) -> String {
            "in-memory-test".to_string()
        }

        async fn get(
            &self,
            key: CacheKey128,
        ) -> std::result::Result<Option<CacheLookup>, CacheAbiError> {
            let entries = self.entries.lock().map_err(|err| {
                CacheAbiError::new(format!("cache backend mutex poisoned: {err}"))
            })?;
            Ok(entries
                .get(&key)
                .map(|(bytes, size_bytes)| CacheLookup::new(bytes.clone(), *size_bytes)))
        }

        async fn insert(
            &self,
            key: CacheKey128,
            value: &[u8],
            size_bytes: usize,
        ) -> std::result::Result<(), CacheAbiError> {
            let mut entries = self.entries.lock().map_err(|err| {
                CacheAbiError::new(format!("cache backend mutex poisoned: {err}"))
            })?;
            entries.insert(key, (value.to_vec(), size_bytes));
            Ok(())
        }

        async fn clear(&self) -> std::result::Result<(), CacheAbiError> {
            let mut entries = self.entries.lock().map_err(|err| {
                CacheAbiError::new(format!("cache backend mutex poisoned: {err}"))
            })?;
            entries.clear();
            Ok(())
        }

        async fn measure(&self) -> std::result::Result<CacheMeasure, CacheAbiError> {
            let entries = self.entries.lock().map_err(|err| {
                CacheAbiError::new(format!("cache backend mutex poisoned: {err}"))
            })?;
            Ok(CacheMeasure::new(
                entries.len(),
                entries.values().map(|(_, size_bytes)| *size_bytes).sum(),
            ))
        }
    }

    #[test]
    fn key128_round_trips_canonical_bytes() {
        let bytes = [
            0x12, 0x34, 0x56, 0x78, 0x90, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x09, 0x87, 0x65,
            0x43, 0x21,
        ];
        let key = CacheKey128::from_bytes(bytes);

        assert_eq!(key.as_bytes(), &bytes);
        assert_eq!(key.into_bytes(), bytes);
    }

    #[test]
    fn owned_xabi_backend_round_trips_bytes() {
        futures::executor::block_on(async {
            let backend = OwnedDynamicCacheBackend::new(InMemoryBackend::default());
            let backend = backend.xabi_borrow();
            let key = CacheKey128::from_bytes([42; 16]);

            backend
                .insert(key, b"payload", 64)
                .await
                .expect("insert succeeds");

            let hit = backend
                .get(key)
                .await
                .expect("get succeeds")
                .expect("entry is cached");
            assert_eq!(hit.bytes, b"payload");
            assert_eq!(hit.size_bytes, 64);

            let measure = backend.measure().await.expect("measure succeeds");
            assert_eq!(measure, CacheMeasure::new(1, 64));

            backend.clear().await.expect("clear succeeds");
            assert_eq!(backend.get(key).await.expect("get succeeds"), None);
        });
    }

    #[test]
    fn abi_layout_is_snapshotted() {
        xabi_assert::assert_abi!(DynamicCacheBackendAbi);
    }
}
