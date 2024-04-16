// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Testing utilities

use crate::Result;
use async_trait::async_trait;
use bytes::Bytes;
use chrono::{Duration, TimeDelta};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use object_store::path::Path;
use object_store::{
    Error as OSError, GetOptions, GetResult, ListResult, MultipartId, ObjectMeta, ObjectStore,
    PutOptions, PutResult, Result as OSResult,
};
use std::collections::HashMap;
use std::fmt::Debug;
use std::future;
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::io::AsyncWrite;

// A policy function takes in the name of the operation (e.g. "put") and the location
// that is being accessed / modified and returns an optional error.
pub trait PolicyFnT: Fn(&str, &Path) -> Result<()> + Send + Sync {}
impl<F> PolicyFnT for F where F: Fn(&str, &Path) -> Result<()> + Send + Sync {}
impl Debug for dyn PolicyFnT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PolicyFn")
    }
}
type PolicyFn = Arc<dyn PolicyFnT>;

// These policy functions receive (and optionally transform) an ObjectMeta
// They apply to functions that list file info
pub trait ObjectMetaPolicyFnT: Fn(&str, ObjectMeta) -> Result<ObjectMeta> + Send + Sync {}
impl<F> ObjectMetaPolicyFnT for F where F: Fn(&str, ObjectMeta) -> Result<ObjectMeta> + Send + Sync {}
impl Debug for dyn ObjectMetaPolicyFnT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PolicyFn")
    }
}
type ObjectMetaPolicyFn = Arc<dyn ObjectMetaPolicyFnT>;

/// A policy container, meant to be shared between test code and the proxy object store.
///
/// This container allows you to configure policies that should apply to the proxied calls.
///
/// Typically, you would use this to simulate I/O errors or mock out data.
///
/// Currently, for simplicity, we only proxy calls that involve some kind of path.  Calls
/// to copy functions, which have a src and dst, will provide the source to the policy
#[derive(Debug, Default)]
pub struct ProxyObjectStorePolicy {
    /// Policies which run before a method is invoked.  If the policy returns
    /// an error then the target method will not be invoked and the error will
    /// be returned instead.
    before_policies: HashMap<String, PolicyFn>,
    /// Policies which run after calls that return ObjectMeta.  The policy can
    /// tranform the returned ObjectMeta to mock out file listing results.
    object_meta_policies: HashMap<String, ObjectMetaPolicyFn>,
}

impl ProxyObjectStorePolicy {
    pub fn new() -> Self {
        Default::default()
    }

    /// Set a new policy with the given name
    ///
    /// The name can be used to later remove this policy
    pub fn set_before_policy(&mut self, name: &str, policy: PolicyFn) {
        self.before_policies.insert(name.to_string(), policy);
    }

    pub fn clear_before_policy(&mut self, name: &str) {
        self.before_policies.remove(name);
    }

    pub fn set_obj_meta_policy(&mut self, name: &str, policy: ObjectMetaPolicyFn) {
        self.object_meta_policies.insert(name.to_string(), policy);
    }
}

/// A proxy object store
///
/// This store wraps another object store and applies the given policy to all calls
/// made to the underlying store.  This can be used to simulate failures or, perhaps
/// in the future, to mock out results or provide other fine-grained control.
#[derive(Debug)]
pub struct ProxyObjectStore {
    target: Arc<dyn ObjectStore>,
    policy: Arc<Mutex<ProxyObjectStorePolicy>>,
}

impl ProxyObjectStore {
    pub fn new(target: Arc<dyn ObjectStore>, policy: Arc<Mutex<ProxyObjectStorePolicy>>) -> Self {
        Self { target, policy }
    }

    fn before_method(&self, method: &str, location: &Path) -> OSResult<()> {
        let policy = self.policy.lock().unwrap();
        for policy in policy.before_policies.values() {
            policy(method, location).map_err(OSError::from)?;
        }
        Ok(())
    }

    fn transform_meta(&self, method: &str, meta: ObjectMeta) -> OSResult<ObjectMeta> {
        let policy = self.policy.lock().unwrap();
        let mut meta = meta;
        for policy in policy.object_meta_policies.values() {
            meta = policy(method, meta).map_err(OSError::from)?;
        }
        Ok(meta)
    }
}

impl std::fmt::Display for ProxyObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ProxyObjectStore({})", self.target)
    }
}

#[async_trait]
impl ObjectStore for ProxyObjectStore {
    async fn put(&self, location: &Path, bytes: Bytes) -> OSResult<PutResult> {
        self.before_method("put", location)?;
        self.target.put(location, bytes).await
    }

    async fn put_opts(
        &self,
        location: &Path,
        bytes: Bytes,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        self.before_method("put", location)?;
        self.target.put_opts(location, bytes, opts).await
    }

    async fn put_multipart(
        &self,
        location: &Path,
    ) -> OSResult<(MultipartId, Box<dyn AsyncWrite + Unpin + Send>)> {
        self.before_method("put_multipart", location)?;
        self.target.put_multipart(location).await
    }

    async fn abort_multipart(&self, location: &Path, multipart_id: &MultipartId) -> OSResult<()> {
        self.before_method("abort_multipart", location)?;
        self.target.abort_multipart(location, multipart_id).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        self.before_method("get_opts", location)?;
        self.target.get_opts(location, options).await
    }

    async fn get_range(&self, location: &Path, range: Range<usize>) -> OSResult<Bytes> {
        self.before_method("get_range", location)?;
        self.target.get_range(location, range).await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<usize>]) -> OSResult<Vec<Bytes>> {
        self.before_method("get_ranges", location)?;
        self.target.get_ranges(location, ranges).await
    }

    async fn head(&self, location: &Path) -> OSResult<ObjectMeta> {
        self.before_method("head", location)?;
        let meta = self.target.head(location).await?;
        self.transform_meta("head", meta)
    }

    async fn delete(&self, location: &Path) -> OSResult<()> {
        self.before_method("delete", location)?;
        self.target.delete(location).await
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'_, OSResult<ObjectMeta>> {
        self.target
            .list(prefix)
            .and_then(|meta| future::ready(self.transform_meta("list", meta)))
            .boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.target.list_with_delimiter(prefix).await
    }

    async fn copy(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.before_method("copy", from)?;
        self.target.copy(from, to).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.before_method("rename", from)?;
        self.target.rename(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.before_method("copy_if_not_exists", from)?;
        self.target.copy_if_not_exists(from, to).await
    }
}

// Regrettably, the system clock is a process-wide global. That means that tests running
// in parallel can interfere with each other if they both want to adjust the system clock.
//
// By using MockClock below (which wraps mock_instant::MockClock), we can prevent this from
// happening, though there is a test time cost as this will prevent some potential test
// parallelism in a rather negative way (blocking).
//
// It also means that if one clock-dependent test fails then all future clock-dependent
// tests will fail because of mutex poisoning.
static CLOCK_MUTEX: Mutex<()> = Mutex::new(());
pub struct MockClock<'a> {
    _guard: MutexGuard<'a, ()>,
}

impl Default for MockClock<'_> {
    fn default() -> Self {
        Self {
            _guard: CLOCK_MUTEX.lock().unwrap(),
        }
    }
}

impl<'a> MockClock<'a> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn set_system_time(&self, time: Duration) {
        mock_instant::MockClock::set_system_time(time.to_std().unwrap());
    }
}

impl<'a> Drop for MockClock<'a> {
    fn drop(&mut self) {
        // Reset the clock to the epoch
        mock_instant::MockClock::set_system_time(TimeDelta::try_days(0).unwrap().to_std().unwrap());
    }
}

/// A guard that sets an environment variable to a given value and restores the original value when dropped.
pub struct EnvVarGuard {
    name: String,
    old_value: Option<String>,
}

impl EnvVarGuard {
    pub fn new(name: &str, value: &str) -> Self {
        let old_value = std::env::var(name).ok();
        std::env::set_var(name, value);
        Self {
            name: name.to_string(),
            old_value,
        }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(value) = &self.old_value {
            std::env::set_var(&self.name, value);
        } else {
            std::env::remove_var(&self.name);
        }
    }
}
