// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use pyo3::{pyclass, pymethods, prelude::*, types::PyDict};

use lance::dataset::{DEFAULT_INDEX_CACHE_SIZE, DEFAULT_METADATA_CACHE_SIZE};
use lance::session::Session as LanceSession;

use crate::rt;

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
        Self {
            inner: Arc::new(session),
        }
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

    /// Return whether the other session is the same as this one.
    pub fn is_same_as(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Return a snapshot of the data cache statistics, or ``None`` if no data
    /// cache is configured.
    ///
    /// Keys
    /// ----
    /// memory_hits : int
    ///     Byte ranges served directly from the in-memory (L1) cache.
    /// memory_misses : int
    ///     Byte ranges not found in memory (went to SSD or object store).
    /// memory_evictions : int
    ///     Entries evicted from memory (may have been written to SSD).
    /// memory_current_bytes : int
    ///     Bytes currently held in the memory tier.
    /// memory_stale_evictions : int
    ///     Memory entries evicted because cached size < requested length.
    /// ssd_hits : int
    ///     Memory misses that were served from the SSD (L2) tier.
    /// ssd_bytes_written : int
    ///     Total bytes written to the SSD tier via memory eviction.
    /// ssd_stale_misses : int
    ///     SSD entries skipped because cached size < requested length.
    pub fn cache_stats<'py>(&self, py: Python<'py>) -> pyo3::PyResult<Option<pyo3::Bound<'py, PyDict>>> {
        let Some(cache) = self.inner.data_cache() else {
            return Ok(None);
        };
        let s = cache.cache_stats();
        let d = PyDict::new(py);
        d.set_item("memory_hits", s.memory_hits)?;
        d.set_item("memory_misses", s.memory_misses)?;
        d.set_item("memory_evictions", s.memory_evictions)?;
        d.set_item("memory_current_bytes", s.memory_current_bytes)?;
        d.set_item("memory_stale_evictions", s.memory_stale_evictions)?;
        d.set_item("ssd_hits", s.ssd_hits)?;
        d.set_item("ssd_bytes_written", s.ssd_bytes_written)?;
        d.set_item("ssd_stale_misses", s.ssd_stale_misses)?;
        Ok(Some(d))
    }
}
