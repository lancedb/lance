// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use pyo3::{pyclass, pymethods};

use lance::dataset::{DEFAULT_INDEX_CACHE_SIZE, DEFAULT_METADATA_CACHE_SIZE};
use lance::session::Session as LanceSession;

use crate::RT;

/// The Session holds stateful information for a dataset.
///
/// The session contains caches for opened indices and file metadata.
#[pyclass(name = "_Session", module = "_lib")]
#[derive(Clone)]
pub struct Session {
    pub inner: Arc<LanceSession>,
}

impl Session {
    pub fn new(inner: Arc<LanceSession>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl Session {
    #[new]
    #[pyo3(signature=(index_cache_size_bytes=None, metadata_cache_size_bytes=None))]
    fn create(
        index_cache_size_bytes: Option<usize>,
        metadata_cache_size_bytes: Option<usize>,
    ) -> Self {
        let session = LanceSession::new(
            index_cache_size_bytes.unwrap_or(DEFAULT_INDEX_CACHE_SIZE),
            metadata_cache_size_bytes.unwrap_or(DEFAULT_METADATA_CACHE_SIZE),
            Default::default(),
        );
        Session {
            inner: Arc::new(session),
        }
    }

    fn __repr__(&self) -> String {
        let (index_cache_size, meta_cache_size) = RT
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

    /// Return whether the other session is the same as this one.
    pub fn is_same_as(&self, other: &Session) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}
