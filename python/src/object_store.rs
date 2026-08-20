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
//! - `PyObjectStoreProvider` is a bridge that adapts a built-in Rust provider
//!   (currently just `MemoryStoreProvider`), a Python object that implements
//!   the `new_store` protocol, or an `Arc<dyn ObjectStoreProvider>` produced
//!   by a *separate* wheel and handed across a `PyCapsule`, to
//!   `Arc<dyn ObjectStoreProvider>`.
//!
//! The Python-callable path is intentionally stubbed for this first cut: the
//! full Python-to-Rust `ObjectStore` bridge (i.e. wrapping a Python-returned
//! object as an `object_store::ObjectStore`) is a follow-up. The smoke test
//! against the built-in memory provider proves the registration + dispatch
//! plumbing works end to end.
//!
//! # Out-of-tree providers via `PyCapsule`
//!
//! [`PyObjectStoreProvider::from_capsule`] lets an external wheel register a
//! Rust `ObjectStoreProvider` it compiled itself. The external wheel builds an
//! `Arc<dyn ObjectStoreProvider>`, wraps it in a `PyCapsule` named
//! [`PROVIDER_CAPSULE_NAME`], and passes that capsule here. Because Rust has
//! no stable ABI, this is sound **only when both wheels are built in lockstep**:
//! identical `rustc`, identical `lance-io` / `object_store` source, and
//! identical resolved dependency versions, so the trait object's vtable and the
//! types in `new_store`'s signature have the same layout on both sides. In
//! Phase I both wheels are built locally from the same branch and toolchain, so
//! the constraint holds; distributing pre-built wheels that must interoperate
//! is deferred (a packaging-phase concern).

use std::ffi::CStr;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};

use lance_io::object_store::providers::memory::MemoryStoreProvider;
use lance_io::object_store::{
    ObjectStore, ObjectStoreParams, ObjectStoreProvider, ObjectStoreRegistry,
};

/// Name that every capsule passed to [`PyObjectStoreProvider::from_capsule`]
/// must carry. External wheels create their capsule with this exact name so a
/// capsule holding some unrelated pointer cannot be mistaken for a provider.
pub const PROVIDER_CAPSULE_NAME: &CStr = c"lance_object_store_provider";

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
/// Three ways to construct one from Python:
/// - `ObjectStoreProvider(py_obj)` — hold a Python object implementing
///   `new_store(base_path, storage_options)`. Registration succeeds; dispatch
///   currently raises because the full Python-to-Rust `ObjectStore` bridge
///   is a follow-up.
/// - `ObjectStoreProvider.memory()` — wrap the built-in `MemoryStoreProvider`.
///   Registrable under any scheme and fully functional for read/write.
/// - `ObjectStoreProvider.from_capsule(capsule)` — adopt an
///   `Arc<dyn ObjectStoreProvider>` produced by a separate, ABI-compatible
///   wheel (see the module docs on the `PyCapsule` handoff and its lockstep
///   build requirement). This is how an out-of-tree Rust provider registers
///   itself without living in the Lance source tree.
#[pyclass(name = "_ObjectStoreProvider", module = "_lib", from_py_object)]
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

    /// Adopt an `Arc<dyn ObjectStoreProvider>` carried in a `PyCapsule` created
    /// by a separate wheel. The capsule must be named [`PROVIDER_CAPSULE_NAME`]
    /// and hold exactly an `Arc<dyn ObjectStoreProvider>`.
    ///
    /// See the module docstring for the ABI-lockstep requirement: the calling
    /// wheel must be built against the identical `lance-io` / `object_store`
    /// source and toolchain as this one.
    #[staticmethod]
    fn from_capsule(capsule: &Bound<'_, PyCapsule>) -> PyResult<Self> {
        // pyo3 0.28's `PyCapsule::name()` yields a `CapsuleName`; `as_cstr` is
        // unsafe only because the name pointer's lifetime is not tied to the
        // capsule. Our capsule names are statically allocated, and we use the
        // borrow immediately (compare, or copy into an owned String), so the
        // pointer is valid for the duration of each use.
        match capsule.name()? {
            Some(name) if unsafe { name.as_cstr() } == PROVIDER_CAPSULE_NAME => {}
            other => {
                return Err(PyValueError::new_err(format!(
                    "expected a PyCapsule named {:?}, got {:?}",
                    PROVIDER_CAPSULE_NAME.to_string_lossy(),
                    other.map(|c| unsafe { c.as_cstr() }.to_string_lossy().into_owned()),
                )));
            }
        }

        // SAFETY: by the capsule-name contract above, the capsule carries an
        // `Arc<dyn ObjectStoreProvider>` built against the identical lance-io /
        // object_store types (same source, rustc, and resolved dependency
        // versions). The name was validated above, so we retrieve the pointer
        // (`pointer_checked` re-verifies the capsule is non-null and valid),
        // dereference it only long enough to clone the `Arc` (bumping the
        // strong count); the capsule keeps its own reference and drops it on GC.
        let ptr = capsule.pointer_checked(None)?;
        let provider = unsafe { ptr.cast::<Arc<dyn ObjectStoreProvider>>().as_ref() };
        Ok(Self {
            inner: provider.clone(),
        })
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
#[pyclass(name = "_ObjectStoreRegistry", module = "_lib", from_py_object)]
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
