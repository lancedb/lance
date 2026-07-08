// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Pluggable cache-backend registry.
//!
//! A [`BackendConfig`] identifies which backend to build (`kind`) and carries
//! backend-specific string options. Backends are constructed through a
//! [`BackendBuildFn`] registered under a unique `kind`. Third-party crates
//! integrate by calling [`register_backend`] once at application startup;
//! [`build_from_config`] then locates the constructor and hands it the
//! config.
//!
//! The registry uses `HashMap<String, String>` for options so it can be
//! represented naturally across FFI (Python `dict[str, str]`, Java
//! `Map<String, String>`, etc.).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use super::backend::CacheBackend;
use super::moka::{MOKA_BACKEND_KIND, build_moka};
use crate::{Error, Result};

/// Backend-independent configuration passed to a [`BackendBuildFn`].
///
/// `kind` selects which registered backend to construct; `options` carries
/// backend-specific key/value settings (e.g. `capacity`, `path`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendConfig {
    /// Registered backend identifier, e.g. `"moka"`.
    pub kind: String,
    /// Backend-specific string options.
    pub options: HashMap<String, String>,
}

impl BackendConfig {
    /// Build a config with no options.
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            options: HashMap::new(),
        }
    }

    /// Insert a single option and return `self`, enabling chaining.
    pub fn with_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }
}

/// Constructor signature for a cache backend.
///
/// Constructors are synchronous. Backends that need async initialization
/// should surface a `try_new_blocking` shim (or equivalent) and call it here.
pub type BackendBuildFn = fn(&BackendConfig) -> Result<Arc<dyn CacheBackend>>;

fn registry() -> &'static Mutex<HashMap<String, BackendBuildFn>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, BackendBuildFn>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn registry_lock() -> Result<MutexGuard<'static, HashMap<String, BackendBuildFn>>> {
    registry()
        .lock()
        .map_err(|_| Error::internal("cache backend registry mutex is poisoned"))
}

#[cfg(test)]
fn registry_lock_for_test() -> MutexGuard<'static, HashMap<String, BackendBuildFn>> {
    registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Register a constructor for a cache backend under `kind`.
///
/// Returns `Err` if `kind` is already registered or reserved by a built-in
/// backend. This is intentional: duplicate registration (e.g. two crates or a
/// dynamic plugin claiming the same identifier) is a configuration bug that
/// should surface immediately rather than silently overriding an existing
/// implementation.
///
/// Typical usage from a backend crate:
///
/// ```ignore
/// pub fn register() -> lance_core::Result<()> {
///     lance_core::cache::register_backend("my_backend", build_my_backend)
/// }
/// ```
pub fn register_backend(kind: &str, build: BackendBuildFn) -> Result<()> {
    if builtin_backend(kind).is_some() {
        return Err(Error::invalid_input(format!(
            "cache backend kind {:?} is reserved for a built-in backend",
            kind
        )));
    }
    insert_backend(kind, build)
}

fn insert_backend(kind: &str, build: BackendBuildFn) -> Result<()> {
    let mut map = registry_lock()?;
    if map.contains_key(kind) {
        return Err(Error::invalid_input(format!(
            "cache backend {:?} is already registered",
            kind
        )));
    }
    map.insert(kind.to_string(), build);
    Ok(())
}

fn builtin_backend(kind: &str) -> Option<BackendBuildFn> {
    match kind {
        MOKA_BACKEND_KIND => Some(build_moka),
        _ => None,
    }
}

/// Look up the constructor for `config.kind` and build a backend.
///
/// Returns `Err` if no backend has been registered under that identifier.
pub fn build_from_config(config: &BackendConfig) -> Result<Arc<dyn CacheBackend>> {
    ensure_builtin_backends()?;
    let build = {
        let map = registry_lock()?;
        map.get(&config.kind).copied()
    };
    match build {
        Some(build) => build(config),
        None => Err(Error::invalid_input(format!(
            "unknown cache backend kind: {:?}",
            config.kind
        ))),
    }
}

/// Idempotently register the backends that ship with `lance-core`.
///
/// Called by [`build_from_config`] (and, transitively, by
/// [`build_from_uri`](super::backend_uri::build_from_uri)) so a bare Lance
/// installation can build a Moka backend without the caller having to
/// register it. Third-party backends still have to opt in with their own
/// `register()` call.
///
/// The check is against the current registry contents rather than a
/// process-once flag so that `#[cfg(test)]` helpers which snapshot and
/// restore the registry still see the built-in backend after they take
/// ownership.
fn ensure_builtin_backends() -> Result<()> {
    let mut map = registry_lock()?;
    if !map.contains_key(MOKA_BACKEND_KIND)
        && let Some(build) = builtin_backend(MOKA_BACKEND_KIND)
    {
        map.insert(MOKA_BACKEND_KIND.to_string(), build);
    }
    Ok(())
}

/// Test-only helper: replace the registry with an empty map so tests can
/// exercise duplicate-registration logic without polluting the global one.
#[cfg(test)]
pub(super) fn take_registry_for_test() -> HashMap<String, BackendBuildFn> {
    let mut map = registry_lock_for_test();
    std::mem::take(&mut *map)
}

/// Test-only helper: restore a previously captured registry state.
#[cfg(test)]
pub(super) fn restore_registry_for_test(saved: HashMap<String, BackendBuildFn>) {
    let mut map = registry_lock_for_test();
    *map = saved;
}

#[cfg(test)]
pub(super) fn registry_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static M: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
    M.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::pin::Pin;

    use crate::cache::InternalCacheKey;
    use crate::cache::backend::CacheEntry;
    use crate::cache::codec::CacheCodec;
    use futures::Future;

    // A trivial no-op backend so tests do not depend on Moka or any other
    // real backend. Every method returns "empty" / does nothing.
    #[derive(Debug, Default)]
    struct NullBackend;

    #[async_trait]
    impl CacheBackend for NullBackend {
        async fn get(
            &self,
            _key: &InternalCacheKey,
            _codec: Option<CacheCodec>,
        ) -> Option<CacheEntry> {
            None
        }

        async fn insert(
            &self,
            _key: &InternalCacheKey,
            _entry: CacheEntry,
            _size_bytes: usize,
            _codec: Option<CacheCodec>,
        ) {
        }

        async fn get_or_insert<'a>(
            &self,
            _key: &InternalCacheKey,
            loader: Pin<Box<dyn Future<Output = crate::Result<(CacheEntry, usize)>> + Send + 'a>>,
            _codec: Option<CacheCodec>,
        ) -> crate::Result<(CacheEntry, bool)> {
            let (entry, _size) = loader.await?;
            Ok((entry, false))
        }

        async fn clear(&self) {}
        async fn num_entries(&self) -> usize {
            0
        }
        async fn size_bytes(&self) -> usize {
            0
        }
    }

    fn build_null(_cfg: &BackendConfig) -> Result<Arc<dyn CacheBackend>> {
        Ok(Arc::new(NullBackend))
    }

    struct RegistryGuard {
        // Hold the serialization lock for the full test.
        _lock: std::sync::MutexGuard<'static, ()>,
        saved: HashMap<String, BackendBuildFn>,
    }
    impl RegistryGuard {
        fn new() -> Self {
            Self {
                _lock: registry_test_lock(),
                saved: take_registry_for_test(),
            }
        }
    }
    impl Drop for RegistryGuard {
        fn drop(&mut self) {
            restore_registry_for_test(std::mem::take(&mut self.saved));
        }
    }

    #[test]
    fn test_register_and_build() {
        let _guard = RegistryGuard::new();
        register_backend("null", build_null).unwrap();
        let backend = build_from_config(&BackendConfig::new("null")).unwrap();
        // Backend is opaque; we just check that the constructor ran and
        // gave us an Arc<dyn CacheBackend>.
        assert_eq!(Arc::strong_count(&backend), 1);
    }

    #[test]
    fn test_duplicate_registration_errors() {
        let _guard = RegistryGuard::new();
        register_backend("dup", build_null).unwrap();
        let err = register_backend("dup", build_null).unwrap_err();
        assert!(err.to_string().contains("already registered"));
    }

    #[test]
    fn test_builtin_kind_is_reserved() {
        let _guard = RegistryGuard::new();
        let err = register_backend("moka", build_null).unwrap_err();
        assert!(err.to_string().contains("reserved"));
    }

    #[test]
    fn test_unknown_kind_errors() {
        let _guard = RegistryGuard::new();
        let err = build_from_config(&BackendConfig::new("missing")).unwrap_err();
        assert!(err.to_string().contains("unknown cache backend kind"));
    }

    #[test]
    fn test_options_are_passed_through() {
        let _guard = RegistryGuard::new();
        fn build_echo(cfg: &BackendConfig) -> Result<Arc<dyn CacheBackend>> {
            assert_eq!(cfg.options.get("capacity").map(String::as_str), Some("42"));
            Ok(Arc::new(NullBackend))
        }
        register_backend("echo", build_echo).unwrap();
        let cfg = BackendConfig::new("echo").with_option("capacity", "42");
        build_from_config(&cfg).unwrap();
    }
}
