// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::Future;
use tokio::sync::Notify;

use crate::{Error, Result};

use super::CacheCodec;
use super::backend::{
    CacheBackend, CacheBatchEntry, CacheBatchLoader, CacheEntry, CacheKeyIterator,
    InternalCacheKey, validate_loaded_entries, validate_unique_keys,
};

/// Internal record stored in the moka cache.
#[derive(Clone, Debug)]
struct MokaCacheEntry {
    entry: CacheEntry,
    size_bytes: usize,
}

/// Default [`CacheBackend`] backed by a [moka](https://crates.io/crates/moka) cache.
///
/// Provides weighted-capacity eviction and concurrent-load deduplication
/// via a shared in-flight registry. Single-key and batch get-or-insert calls
/// use the same registry, so mixed callers still coalesce on the same key.
/// Hot hits return from the moka cache before touching the registry; cold
/// misses allocate short-lived per-key flight state until the owner completes.
/// This in-flight state is not counted against the weighted cache capacity; it
/// is bounded by currently live cold misses and slow loaders.
pub struct MokaCacheBackend {
    cache: moka::future::Cache<InternalCacheKey, MokaCacheEntry>,
    flights: FlightRegistry,
}

type FlightRegistry = Arc<Mutex<HashMap<InternalCacheKey, Arc<CacheFlight>>>>;

struct CacheFlight {
    state: Mutex<FlightState>,
    notify: Notify,
}

#[derive(Clone)]
enum FlightState {
    Loading,
    Ready(MokaCacheEntry),
    Retry,
}

enum FlightClaim {
    Owner(FlightOwner),
    Waiter(Arc<CacheFlight>),
}

struct FlightOwner {
    key: InternalCacheKey,
    flight: Arc<CacheFlight>,
    flights: FlightRegistry,
    completed: bool,
}

impl CacheFlight {
    fn new() -> Self {
        Self {
            state: Mutex::new(FlightState::Loading),
            notify: Notify::new(),
        }
    }

    async fn wait(&self) -> Result<FlightState> {
        loop {
            let notified = self.notify.notified();
            {
                let state = self.state.lock().map_err(|_| {
                    Error::internal("cache flight state mutex poisoned while waiting")
                })?;
                match &*state {
                    FlightState::Loading => {}
                    FlightState::Ready(entry) => return Ok(FlightState::Ready(entry.clone())),
                    FlightState::Retry => return Ok(FlightState::Retry),
                }
            }
            notified.await;
        }
    }
}

impl FlightOwner {
    fn complete(&mut self, entry: MokaCacheEntry) -> Result<()> {
        self.set_state(FlightState::Ready(entry))
    }

    fn retry(&mut self) -> Result<()> {
        self.set_state(FlightState::Retry)
    }

    fn set_state(&mut self, state: FlightState) -> Result<()> {
        {
            let mut guard = self.flight.state.lock().map_err(|_| {
                Error::internal("cache flight state mutex poisoned while completing owner")
            })?;
            *guard = state;
        }
        let remove_result = self.remove_from_registry();
        self.completed = true;
        self.flight.notify.notify_waiters();
        remove_result
    }

    fn remove_from_registry(&self) -> Result<()> {
        let mut flights = self.flights.lock().map_err(|_| {
            Error::internal("cache flight registry mutex poisoned while removing owner")
        })?;
        if flights
            .get(&self.key)
            .is_some_and(|flight| Arc::ptr_eq(flight, &self.flight))
        {
            flights.remove(&self.key);
        }
        Ok(())
    }
}

impl Drop for FlightOwner {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.retry();
        }
    }
}

impl std::fmt::Debug for MokaCacheBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let in_flight_entries = self.flights.lock().map(|flights| flights.len()).ok();
        f.debug_struct("MokaCacheBackend")
            .field("entry_count", &self.cache.entry_count())
            .field("in_flight_entries", &in_flight_entries)
            .finish()
    }
}

impl MokaCacheBackend {
    pub fn with_capacity(capacity: usize) -> Self {
        let cache = moka::future::Cache::builder()
            .max_capacity(capacity as u64)
            .weigher(|_, v: &MokaCacheEntry| v.size_bytes.try_into().unwrap_or(u32::MAX))
            .support_invalidation_closures()
            .build();
        Self {
            cache,
            flights: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn no_cache() -> Self {
        Self {
            cache: moka::future::Cache::new(0),
            flights: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn claim_flight(&self, key: InternalCacheKey) -> Result<FlightClaim> {
        let mut flights = self.flights.lock().map_err(|_| {
            Error::internal("cache flight registry mutex poisoned while claiming owner")
        })?;
        // Do not inspect flight.state while holding the registry lock. Owners
        // complete with state -> registry order, so claim must only take the
        // registry lock to avoid lock-order inversions.
        if let Some(flight) = flights.get(&key).cloned() {
            return Ok(FlightClaim::Waiter(flight));
        }

        let flight = Arc::new(CacheFlight::new());
        flights.insert(key.clone(), flight.clone());
        Ok(FlightClaim::Owner(FlightOwner {
            key,
            flight,
            flights: self.flights.clone(),
            completed: false,
        }))
    }
}

#[async_trait]
impl CacheBackend for MokaCacheBackend {
    async fn get(&self, key: &InternalCacheKey, _codec: Option<CacheCodec>) -> Option<CacheEntry> {
        self.cache.get(key).await.map(|r| r.entry)
    }

    async fn insert(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size_bytes: usize,
        _codec: Option<CacheCodec>,
    ) {
        self.cache
            .insert(key.clone(), MokaCacheEntry { entry, size_bytes })
            .await;
    }

    async fn get_or_insert<'a>(
        &self,
        key: &InternalCacheKey,
        loader: Pin<Box<dyn Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>,
        _codec: Option<CacheCodec>,
    ) -> Result<(CacheEntry, bool)> {
        let mut loader = Some(loader);

        loop {
            if let Some(record) = self.cache.get(key).await {
                return Ok((record.entry, true));
            }

            match self.claim_flight(key.clone())? {
                FlightClaim::Owner(mut owner) => {
                    if let Some(record) = self.cache.get(key).await {
                        owner.complete(record.clone())?;
                        return Ok((record.entry, true));
                    }

                    let Some(loader) = loader.take() else {
                        owner.retry()?;
                        return Err(crate::Error::internal(
                            "single-key cache loader already consumed",
                        ));
                    };
                    let (entry, size_bytes) = match loader.await {
                        Ok(loaded) => loaded,
                        Err(err) => {
                            owner.retry()?;
                            return Err(err);
                        }
                    };
                    let record = MokaCacheEntry { entry, size_bytes };
                    self.cache.insert(key.clone(), record.clone()).await;
                    owner.complete(record.clone())?;
                    return Ok((record.entry, false));
                }
                FlightClaim::Waiter(flight) => match flight.wait().await? {
                    FlightState::Ready(record) => return Ok((record.entry, true)),
                    FlightState::Retry => {}
                    FlightState::Loading => {
                        return Err(Error::internal(format!(
                            "cache flight wait returned loading state for key: prefix='{}', key='{}', type='{}'",
                            key.prefix(),
                            key.key(),
                            key.type_name()
                        )));
                    }
                },
            }
        }
    }

    async fn get_or_insert_many<'a>(
        &self,
        keys: Vec<InternalCacheKey>,
        loader: CacheBatchLoader<'a>,
        _codec: Option<CacheCodec>,
    ) -> Result<Vec<CacheBatchEntry>> {
        validate_unique_keys(&keys)?;

        // This override keeps the loader batched for owned missing keys while
        // still coordinating per-key single-flight with single-key callers. The
        // HashMaps/Vectors below are per-request state bounded by the input
        // batch and current in-flight overlap; they are not retained in the
        // cache after this call completes and are not part of cache capacity
        // accounting.
        let mut remaining = keys.clone();
        let mut results = HashMap::with_capacity(keys.len());

        while !remaining.is_empty() {
            let remaining_len = remaining.len();
            let mut owners = HashMap::with_capacity(remaining_len);
            let mut owner_keys = Vec::with_capacity(remaining_len);
            let mut waiters = Vec::with_capacity(remaining_len);

            for key in remaining {
                if let Some(record) = self.cache.get(&key).await {
                    results.insert(
                        key.clone(),
                        CacheBatchEntry {
                            key,
                            entry: record.entry,
                            was_cached: true,
                        },
                    );
                    continue;
                }

                match self.claim_flight(key.clone())? {
                    FlightClaim::Owner(owner) => {
                        owner_keys.push(key.clone());
                        owners.insert(key, owner);
                    }
                    FlightClaim::Waiter(flight) => {
                        waiters.push((key, flight));
                    }
                }
            }

            let mut loader_keys = Vec::with_capacity(owner_keys.len());
            for key in owner_keys {
                if let Some(record) = self.cache.get(&key).await {
                    let mut owner = owners.remove(&key).ok_or_else(|| {
                        Error::internal(format!(
                            "owned cache flight missing for cache-hit owner key: prefix='{}', key='{}', type='{}'",
                            key.prefix(),
                            key.key(),
                            key.type_name()
                        ))
                    })?;
                    owner.complete(record.clone())?;
                    results.insert(
                        key.clone(),
                        CacheBatchEntry {
                            key,
                            entry: record.entry,
                            was_cached: true,
                        },
                    );
                } else {
                    loader_keys.push(key);
                }
            }

            if !loader_keys.is_empty() {
                match loader(loader_keys.clone()).await {
                    Ok(loaded) => {
                        let mut loaded = validate_loaded_entries(&loader_keys, loaded)?;
                        for key in loader_keys {
                            let loaded = loaded.remove(&key).ok_or_else(|| {
                                Error::internal(format!(
                                    "validated batch loader result missing owner key: prefix='{}', key='{}', type='{}'",
                                    key.prefix(),
                                    key.key(),
                                    key.type_name()
                                ))
                            })?;
                            let record = MokaCacheEntry {
                                entry: loaded.entry,
                                size_bytes: loaded.size_bytes,
                            };
                            self.cache.insert(key.clone(), record.clone()).await;
                            let mut owner = owners.remove(&key).ok_or_else(|| {
                                Error::internal(format!(
                                    "owned cache flight missing for loaded key: prefix='{}', key='{}', type='{}'",
                                    key.prefix(),
                                    key.key(),
                                    key.type_name()
                                ))
                            })?;
                            owner.complete(record.clone())?;
                            results.insert(
                                key.clone(),
                                CacheBatchEntry {
                                    key,
                                    entry: record.entry,
                                    was_cached: false,
                                },
                            );
                        }
                    }
                    Err(err) => {
                        for (_, mut owner) in owners {
                            owner.retry()?;
                        }
                        return Err(err);
                    }
                }
            }

            let mut retry_keys = Vec::new();
            for (key, flight) in waiters {
                match flight.wait().await? {
                    FlightState::Ready(record) => {
                        results.insert(
                            key.clone(),
                            CacheBatchEntry {
                                key,
                                entry: record.entry,
                                was_cached: true,
                            },
                        );
                    }
                    FlightState::Retry => retry_keys.push(key),
                    FlightState::Loading => {
                        return Err(Error::internal(format!(
                            "cache flight wait returned loading state for key: prefix='{}', key='{}', type='{}'",
                            key.prefix(),
                            key.key(),
                            key.type_name()
                        )));
                    }
                }
            }

            remaining = retry_keys;
        }

        keys.into_iter()
            .map(|key| {
                results.remove(&key).ok_or_else(|| {
                    Error::internal(format!(
                        "batch cache result missing for input key: prefix='{}', key='{}', type='{}'",
                        key.prefix(),
                        key.key(),
                        key.type_name()
                    ))
                })
            })
            .collect()
    }

    async fn invalidate_prefix(&self, prefix: &str) {
        let prefix = prefix.to_owned();
        self.cache
            .invalidate_entries_if(move |key, _value| key.starts_with(&prefix))
            .expect("Cache configured correctly");
    }

    async fn clear(&self) {
        self.cache.invalidate_all();
        self.cache.run_pending_tasks().await;
    }

    async fn keys(&self) -> Option<CacheKeyIterator<'_>> {
        self.cache.run_pending_tasks().await;
        Some(Box::new(
            self.cache.iter().map(|(key, _)| key.as_ref().clone()),
        ))
    }

    async fn num_entries(&self) -> usize {
        self.cache.run_pending_tasks().await;
        self.cache.entry_count() as usize
    }

    async fn size_bytes(&self) -> usize {
        self.cache.run_pending_tasks().await;
        self.cache.weighted_size() as usize
    }

    fn approx_num_entries(&self) -> usize {
        self.cache.entry_count() as usize
    }

    fn approx_size_bytes(&self) -> usize {
        // Iterate rather than using `weighted_size()` because moka's
        // weighted_size can be stale without `run_pending_tasks()`, which
        // is async and can't be called from this synchronous context.
        self.cache.iter().map(|(_, v)| v.size_bytes).sum()
    }
}
