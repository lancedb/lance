// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyString};
use pyo3::{Bound, PyAny, PyResult, pyclass, pymethods};

use lance::dataset::{DEFAULT_INDEX_CACHE_SIZE, DEFAULT_METADATA_CACHE_SIZE};
use lance::session::Session as LanceSession;
use lance_core::cache::{BackendConfig, CacheBackend, build_from_config, build_from_uri};

use crate::rt;

/// The Session holds stateful information for a dataset.
///
/// The session contains caches for opened indices and file metadata.
///
/// Parameters
/// ----------
/// index_cache_size_bytes : int, optional
///     Capacity of the default index cache in bytes.
/// metadata_cache_size_bytes : int, optional
///     Capacity of the default metadata cache in bytes.
/// index_cache_backend : str or dict, optional
///     Custom index cache backend. Strings are backend URIs such as
///     ``"moka://?capacity=1048576"``. Dicts must contain ``"kind"`` and may
///     contain ``"options"``, for example
///     ``{"kind": "moka", "options": {"capacity": "1048576"}}``.
/// metadata_cache_backend : str or dict, optional
///     Custom metadata cache backend with the same format as
///     ``index_cache_backend``.
///
/// ``index_cache_backend`` is mutually exclusive with
/// ``index_cache_size_bytes``. ``metadata_cache_backend`` is mutually
/// exclusive with ``metadata_cache_size_bytes``.
#[pyclass(name = "_Session", module = "_lib", from_py_object)]
#[derive(Clone)]
pub struct Session {
    pub inner: Arc<LanceSession>,
}

impl Session {
    pub fn new(inner: Arc<LanceSession>) -> Self {
        Self { inner }
    }
}

/// Turn a Python-supplied backend descriptor into an `Arc<dyn CacheBackend>`,
/// or return `Ok(None)` when the caller did not pass one.
///
/// Accepts:
///   * `str` — treated as a URI (`moka://?capacity=...`) and passed to
///     [`build_from_uri`].
///   * `dict` — must have string keys `kind` (required) and `options`
///     (optional `dict[str, str]`) matching [`BackendConfig`]; passed to
///     [`build_from_config`].
///
/// Any other Python type is rejected with a clear `TypeError`-style
/// `PyValueError`.
///
/// If `size_field_set` is `true` and `backend` is `Some`, both a size and a
/// backend were provided for the same cache. Rather than silently letting
/// one override the other (Proposal §7), this is rejected up-front so the
/// operator gets an actionable error.
fn resolve_backend(
    backend_field: &str,
    backend: Option<&Bound<'_, PyAny>>,
    size_field: &str,
    size_field_set: bool,
) -> PyResult<Option<Arc<dyn CacheBackend>>> {
    let Some(value) = backend else {
        return Ok(None);
    };
    if size_field_set {
        return Err(PyValueError::new_err(format!(
            "{} and {} are mutually exclusive; set one or the other",
            size_field, backend_field,
        )));
    }

    if value.cast::<PyString>().is_ok() {
        let uri: String = value.extract()?;
        return build_from_uri(&uri)
            .map(Some)
            .map_err(|e| PyValueError::new_err(format!("{}: {}", backend_field, e)));
    }

    if let Ok(dict) = value.cast::<PyDict>() {
        let cfg = backend_config_from_dict(backend_field, dict)?;
        return build_from_config(&cfg)
            .map(Some)
            .map_err(|e| PyValueError::new_err(format!("{}: {}", backend_field, e)));
    }

    let type_name: String = value.get_type().getattr("__name__")?.extract()?;
    Err(PyValueError::new_err(format!(
        "{}: expected str (URI) or dict with 'kind'/'options' keys, got {}",
        backend_field, type_name,
    )))
}

fn backend_config_from_dict(field: &str, dict: &Bound<'_, PyDict>) -> PyResult<BackendConfig> {
    for (key, _) in dict.iter() {
        if key.cast::<PyString>().is_err() {
            return Err(PyValueError::new_err(format!(
                "{}: dict keys must be strings",
                field
            )));
        }
        let key: String = key.extract()?;
        if key != "kind" && key != "options" {
            return Err(PyValueError::new_err(format!(
                "{}: unknown dict key {:?}; expected 'kind' or 'options'",
                field, key
            )));
        }
    }

    let kind_obj = dict.get_item("kind")?.ok_or_else(|| {
        PyValueError::new_err(format!("{}: dict must contain a 'kind' key", field))
    })?;
    if kind_obj.cast::<PyString>().is_err() {
        return Err(PyValueError::new_err(format!(
            "{}: 'kind' must be a string",
            field
        )));
    }
    let kind: String = kind_obj.extract()?;

    let mut options: HashMap<String, String> = HashMap::new();
    if let Some(options_obj) = dict.get_item("options")? {
        let options_dict = options_obj.cast::<PyDict>().map_err(|_| {
            PyValueError::new_err(format!("{}: 'options' must be a dict[str, str]", field))
        })?;
        for (k, v) in options_dict.iter() {
            if k.cast::<PyString>().is_err() {
                return Err(PyValueError::new_err(format!(
                    "{}: 'options' keys must be strings",
                    field
                )));
            }
            if v.cast::<PyString>().is_err() {
                return Err(PyValueError::new_err(format!(
                    "{}: 'options' values must be strings",
                    field
                )));
            }
            let key: String = k.extract()?;
            let value: String = v.extract()?;
            options.insert(key, value);
        }
    }

    Ok(BackendConfig { kind, options })
}

#[pymethods]
impl Session {
    #[new]
    #[pyo3(signature=(
        index_cache_size_bytes=None,
        metadata_cache_size_bytes=None,
        index_cache_backend=None,
        metadata_cache_backend=None,
    ))]
    fn create(
        index_cache_size_bytes: Option<usize>,
        metadata_cache_size_bytes: Option<usize>,
        index_cache_backend: Option<Bound<'_, PyAny>>,
        metadata_cache_backend: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let index_backend = resolve_backend(
            "index_cache_backend",
            index_cache_backend.as_ref(),
            "index_cache_size_bytes",
            index_cache_size_bytes.is_some(),
        )?;
        let metadata_backend = resolve_backend(
            "metadata_cache_backend",
            metadata_cache_backend.as_ref(),
            "metadata_cache_size_bytes",
            metadata_cache_size_bytes.is_some(),
        )?;

        let session = LanceSession::with_cache_backends(
            index_backend,
            metadata_backend,
            index_cache_size_bytes.unwrap_or(DEFAULT_INDEX_CACHE_SIZE),
            metadata_cache_size_bytes.unwrap_or(DEFAULT_METADATA_CACHE_SIZE),
            Default::default(),
        );
        Ok(Self {
            inner: Arc::new(session),
        })
    }

    fn __repr__(&self) -> String {
        let (index_cache_size, meta_cache_size) = rt()
            .block_on(None, async move {
                (
                    self.inner.index_cache_stats().await.size_bytes,
                    self.inner.metadata_cache_stats().await.size_bytes,
                )
            })
            .unwrap_or((0, 0));
        format!(
            "Session(index_cache_size_bytes={}, metadata_cache_size_bytes={})",
            index_cache_size, meta_cache_size
        )
    }

    /// Return the current size of the session in bytes
    pub fn size_bytes(&self) -> u64 {
        self.inner.size_bytes()
    }

    /// Return the current size of the index cache in bytes.
    pub fn index_cache_size_bytes(&self) -> PyResult<u64> {
        rt().block_on(None, async move {
            self.inner.index_cache_stats().await.size_bytes as u64
        })
    }

    /// Return whether the other session is the same as this one.
    pub fn is_same_as(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}
