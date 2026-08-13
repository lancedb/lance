// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Runtime registration hook for external `ObjectStoreProvider` implementations.
//!
//! This module exposes two pyclasses:
//!
//! - `PyObjectStoreRegistry` wraps [`lance_io::object_store::ObjectStoreRegistry`]
//!   and lets Python code register additional `ObjectStoreProvider`s under new
//!   URL schemes. A registry constructed here can be passed to `Session` so
//!   `lance.dataset("myscheme://...")` dispatches through the new provider.
//! - `PyObjectStoreProvider` is a bridge that adapts either a built-in Rust
//!   provider (currently just `MemoryStoreProvider`) or a Python object that
//!   implements the `new_store` protocol to `Arc<dyn ObjectStoreProvider>`.
//!
//! The Python-callable path is intentionally stubbed for this first cut: the
//! full Python-to-Rust `ObjectStore` bridge (i.e. wrapping a Python-returned
//! object as an `object_store::ObjectStore`) is a follow-up. The smoke test
//! against the built-in memory provider proves the registration + dispatch
//! plumbing works end to end.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use lance_io::object_store::{
    ObjectStore, ObjectStoreParams, ObjectStoreProvider, ObjectStoreRegistry,
};
use lance_io::object_store::providers::memory::MemoryStoreProvider;

/// Bridge between a Python object and the Rust `ObjectStoreProvider` trait.
///
/// For the memory variant we short-circuit to a real Rust provider so the
/// smoke test can prove the scheme-dispatch plumbing works. For the Python
/// callable variant we hold a `Py<PyAny>` and (in a follow-up) will call
/// `new_store(base_path, storage_options)` on it via the GIL.
#[derive(Debug)]
enum PyProviderBridge {
    /// Built-in `MemoryStoreProvider`, wrapped directly.
    Memory(MemoryStoreProvider),
    /// A Python object implementing the `new_store(base_path, storage_options)`
    /// protocol. Not yet dispatchable end-to-end (see module docstring).
    ///
    /// The wrapped `Py<PyAny>` is intentionally held even though we do not
    /// invoke it yet: keeping the Python object alive here means once the
    /// bridge lands, we can dispatch without changing the enum shape.
    #[allow(dead_code)]
    PyCallable(Py<PyAny>),
}

#[async_trait::async_trait]
impl ObjectStoreProvider for PyProviderBridge {
    async fn new_store(
        &self,
        base_path: url::Url,
        params: &ObjectStoreParams,
    ) -> lance_core::Result<ObjectStore> {
        match self {
            Self::Memory(inner) => inner.new_store(base_path, params).await,
            Self::PyCallable(_) => Err(lance_core::Error::not_supported(
                "PyObjectStoreProvider: the Python-callable bridge is not yet \
                 implemented. Use PyObjectStoreProvider.memory() for the current cut.",
            )),
        }
    }
}

/// Python-facing wrapper around `Arc<dyn ObjectStoreProvider>`.
///
/// Two ways to construct one from Python:
/// - `ObjectStoreProvider(py_obj)` — hold a Python object implementing
///   `new_store(base_path, storage_options)`. Registration succeeds; dispatch
///   currently raises because the full Python-to-Rust `ObjectStore` bridge
///   is a follow-up.
/// - `ObjectStoreProvider.memory()` — wrap the built-in `MemoryStoreProvider`.
///   Registrable under any scheme and fully functional for read/write.
#[pyclass(name = "_ObjectStoreProvider", module = "_lib")]
#[derive(Clone)]
pub struct PyObjectStoreProvider {
    pub(crate) inner: Arc<dyn ObjectStoreProvider>,
}

#[pymethods]
impl PyObjectStoreProvider {
    /// Wrap a Python object implementing the `new_store(base_path, storage_options)`
    /// protocol. Registration will succeed, but scheme-dispatch will raise until
    /// the full Python-to-Rust `ObjectStore` bridge is implemented.
    #[new]
    fn new(py_object: Py<PyAny>) -> Self {
        Self {
            inner: Arc::new(PyProviderBridge::PyCallable(py_object)),
        }
    }

    /// Return a provider backed by the built-in `MemoryStoreProvider`. Every
    /// call to `new_store` allocates a fresh in-memory `object_store::InMemory`;
    /// the enclosing `ObjectStoreRegistry` caches the resulting `ObjectStore`
    /// so writers and readers using the same scheme share storage as long as
    /// something holds a strong reference.
    #[staticmethod]
    fn memory() -> Self {
        Self {
            inner: Arc::new(PyProviderBridge::Memory(MemoryStoreProvider)),
        }
    }

    fn __repr__(&self) -> String {
        format!("_ObjectStoreProvider({:?})", self.inner)
    }
}

/// Python-facing wrapper around `Arc<ObjectStoreRegistry>`.
///
/// A new instance starts from `ObjectStoreRegistry::default()`, so all
/// built-in schemes (memory, file, and any of s3/az/gs/oss/... enabled at
/// build time) are already registered. Additional providers can be inserted
/// under new (or overridden) schemes via `register_provider`.
///
/// Pass an instance as the `store_registry` argument of `Session(...)` to
/// make its schemes visible to `lance.dataset(uri, session=...)` and
/// `lance.write_dataset(..., uri, session=...)`.
#[pyclass(name = "_ObjectStoreRegistry", module = "_lib")]
#[derive(Clone)]
pub struct PyObjectStoreRegistry {
    pub(crate) inner: Arc<ObjectStoreRegistry>,
}

#[pymethods]
impl PyObjectStoreRegistry {
    /// Create a new registry pre-populated with the built-in schemes.
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(ObjectStoreRegistry::default()),
        }
    }

    /// Register a provider under a scheme. Idempotent: registering the same
    /// scheme again replaces the previous provider. Registering under a
    /// built-in scheme (e.g. `"memory"`) overrides that built-in.
    fn register_provider(&self, scheme: &str, provider: &PyObjectStoreProvider) -> PyResult<()> {
        if scheme.is_empty() {
            return Err(PyValueError::new_err("scheme must be a non-empty string"));
        }
        self.inner.insert(scheme, provider.inner.clone());
        Ok(())
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "_ObjectStoreRegistry(active_stores={}, hits={}, misses={})",
            stats.active_stores, stats.hits, stats.misses,
        )
    }
}
