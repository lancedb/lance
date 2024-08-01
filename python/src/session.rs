// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use pyo3::{pyclass, pymethods};

use lance::session::Session as LanceSession;

/// The Session holds stateful information for a dataset.
///
/// The session contains caches for opened indices and file metadata.
#[pyclass(name = "_Session", module = "_lib")]
#[derive(Clone)]
pub struct Session {
    inner: Arc<LanceSession>,
}

impl Session {
    pub fn new(inner: Arc<LanceSession>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl Session {
    /// Return the current size of the session in bytes
    pub fn size_bytes(&self) -> u64 {
        self.inner.size_bytes()
    }
}
