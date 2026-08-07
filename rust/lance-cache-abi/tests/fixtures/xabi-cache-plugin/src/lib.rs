// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Mutex;

use lance_cache_abi::{CacheAbiError, CacheKey128, CacheLookup, CacheMeasure, DynamicCacheBackend};

type EntryMap = HashMap<CacheKey128, (Vec<u8>, usize)>;
type EntryGuard<'a> = std::sync::MutexGuard<'a, EntryMap>;

#[derive(Default)]
struct MemoryBackend {
    entries: Mutex<EntryMap>,
}

impl MemoryBackend {
    fn lock_entries(&self) -> std::result::Result<EntryGuard<'_>, CacheAbiError> {
        self.entries
            .lock()
            .map_err(|err| CacheAbiError::new(format!("cache backend mutex poisoned: {err}")))
    }

    fn reject_error_key(key: CacheKey128) -> std::result::Result<(), CacheAbiError> {
        if key.as_bytes() == &[13; 16] {
            return Err(CacheAbiError::new("fixture rejected key 13"));
        }
        Ok(())
    }
}

#[xabi::module]
mod exports {
    use super::*;

    #[xabi::xabi(name = "memory", version = 1)]
    impl DynamicCacheBackend for MemoryBackend {
        fn name(&self) -> String {
            "memory-fixture".to_string()
        }

        async fn get(
            &self,
            key: CacheKey128,
        ) -> std::result::Result<Option<CacheLookup>, CacheAbiError> {
            Self::reject_error_key(key)?;
            let entries = self.lock_entries()?;
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
            Self::reject_error_key(key)?;
            let mut entries = self.lock_entries()?;
            entries.insert(key, (value.to_vec(), size_bytes));
            Ok(())
        }

        async fn clear(&self) -> std::result::Result<(), CacheAbiError> {
            self.lock_entries()?.clear();
            Ok(())
        }

        async fn measure(&self) -> std::result::Result<CacheMeasure, CacheAbiError> {
            let entries = self.lock_entries()?;
            Ok(CacheMeasure::new(
                entries.len(),
                entries.values().map(|(_, size_bytes)| *size_bytes).sum(),
            ))
        }
    }
}
