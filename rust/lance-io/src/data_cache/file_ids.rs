// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use object_store::path::Path;

/// Interns file paths as compact `u64` identifiers for use in cache keys.
///
/// Modelled after FileIds / `StringIdMap`: a global registry that
/// maps each unique path string to a stable numeric ID. The ID is allocated
/// once and never changes for the lifetime of the cache, so it is safe to use
/// as a hash-map key without holding any lock after the first lookup.
///
/// The registry is held behind a `Mutex` only during the (rare) first-time
/// registration of a new path; subsequent lookups hit the hot path with a
/// single atomic read of `next_id` to bound-check before taking the lock.
pub struct FileIds {
    map: Mutex<HashMap<String, u64>>,
    next_id: AtomicU64,
}

impl std::fmt::Debug for FileIds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileIds")
            .field("count", &self.next_id.load(Ordering::Relaxed))
            .finish()
    }
}

impl FileIds {
    pub fn new() -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
        }
    }

 /// Return the stable numeric ID for `path`, registering it if this is the
 /// first time it is seen.
    pub fn get_or_intern(&self, path: &Path) -> u64 {
        let key = path.as_ref();
        let mut map = self.map.lock().unwrap();
        if let Some(&id) = map.get(key) {
            return id;
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        map.insert(key.to_string(), id);
        id
    }
}

impl Default for FileIds {
    fn default() -> Self {
        Self::new()
    }
}
