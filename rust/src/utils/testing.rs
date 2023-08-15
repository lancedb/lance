// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Testing utilities

use crate::Result;
use async_trait::async_trait;
use bytes::Bytes;
use chrono::Duration;
use futures::stream::BoxStream;
use num_traits::real::Real;
use num_traits::FromPrimitive;
use object_store::path::Path;
use object_store::{
    Error as OSError, GetOptions, GetResult, ListResult, MultipartId, ObjectMeta, ObjectStore,
    Result as OSResult,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::repeat_with;
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::io::AsyncWrite;

use arrow_array::{ArrowNumericType, Float32Array, NativeAdapter, PrimitiveArray};

/// Create a random float32 array.
pub fn generate_random_array_with_seed<T: ArrowNumericType>(
    n: usize,
    seed: [u8; 32],
) -> PrimitiveArray<T>
where
    T::Native: Real + FromPrimitive,
    NativeAdapter<T>: From<T::Native>,
{
    let mut rng = StdRng::from_seed(seed);

    PrimitiveArray::<T>::from_iter(repeat_with(|| T::Native::from_f32(rng.gen::<f32>())).take(n))
}

/// Create a random float32 array.
pub fn generate_random_array(n: usize) -> Float32Array {
    let mut rng = rand::thread_rng();
    Float32Array::from(
        repeat_with(|| rng.gen::<f32>())
            .take(n)
            .collect::<Vec<f32>>(),
    )
}

macro_rules! assert_err_containing {
    ($expr: expr, $message: expr) => {
        match $expr {
            Ok(_) => panic!("expected an error"),
            Err(e) => {
                let err_msg = e.to_string();
                if !err_msg.contains($message) {
                    panic!(
                        "unexpected error message: '{}' but was expecting '{}'",
                        err_msg, $message
                    );
                }
            }
        }
    };
}

pub(crate) use assert_err_containing;

#[derive(Debug)]
pub(crate) struct ProxyObjectStorePolicy {
    before_policies: HashMap<String, fn(&str, &Path) -> Result<()>>,
    after_policies: HashMap<String, fn(&str, &Path) -> Result<()>>,
}

impl ProxyObjectStorePolicy {
    pub fn new() -> Self {
        Self {
            before_policies: HashMap::new(),
            after_policies: HashMap::new(),
        }
    }

    pub fn set_before_policy(&mut self, name: &str, policy: fn(&str, &Path) -> Result<()>) {
        self.before_policies.insert(name.to_string(), policy);
    }

    pub fn set_after_policy(&mut self, name: &str, policy: fn(&str, &Path) -> Result<()>) {
        self.after_policies.insert(name.to_string(), policy);
    }

    pub fn clear_before_policy(&mut self, name: &str) {
        self.before_policies.remove(name);
    }

    pub fn clear_after_policy(&mut self, name: &str) {
        self.after_policies.remove(name);
    }
}

#[derive(Debug)]
pub(crate) struct ProxyObjectStore {
    target: Arc<dyn ObjectStore>,
    policy: Arc<Mutex<ProxyObjectStorePolicy>>,
}

impl ProxyObjectStore {
    pub(crate) fn new(
        target: Arc<dyn ObjectStore>,
        policy: Arc<Mutex<ProxyObjectStorePolicy>>,
    ) -> Self {
        Self { target, policy }
    }

    fn before_method(&self, method: &str, location: &Path) -> OSResult<()> {
        let policy = self.policy.lock().unwrap();
        for policy in policy.before_policies.values() {
            policy(method, location).map_err(|err| OSError::Generic {
                store: "ProxyObjectStore::before",
                source: Box::new(err),
            })?;
        }
        Ok(())
    }

    fn after_method(&self, method: &str, location: &Path) -> OSResult<()> {
        let policy = self.policy.lock().unwrap();
        for policy in policy.after_policies.values() {
            policy(method, location).map_err(|err| OSError::Generic {
                store: "ProxyObjectStore::after",
                source: Box::new(err),
            })?;
        }
        Ok(())
    }
}

impl std::fmt::Display for ProxyObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ProxyObjectStore({})", self.target)
    }
}

#[async_trait]
impl ObjectStore for ProxyObjectStore {
    async fn put(&self, location: &Path, bytes: Bytes) -> OSResult<()> {
        self.before_method("put", location)?;
        let result = self.target.put(location, bytes).await;
        self.after_method("put", location)?;
        result
    }

    async fn put_multipart(
        &self,
        location: &Path,
    ) -> OSResult<(MultipartId, Box<dyn AsyncWrite + Unpin + Send>)> {
        self.before_method("put_multipart", location)?;
        let result = self.target.put_multipart(location).await;
        self.after_method("put_multipart", location)?;
        result
    }

    async fn abort_multipart(&self, location: &Path, multipart_id: &MultipartId) -> OSResult<()> {
        self.before_method("abort_multipart", location)?;
        let result = self.target.abort_multipart(location, multipart_id).await;
        self.after_method("abort_multipart", location)?;
        result
    }

    async fn append(&self, location: &Path) -> OSResult<Box<dyn AsyncWrite + Unpin + Send>> {
        self.before_method("append", location)?;
        let result = self.target.append(location).await;
        self.after_method("append", location)?;
        result
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        self.before_method("get_opts", location)?;
        let result = self.target.get_opts(location, options).await;
        self.after_method("get_opts", location)?;
        result
    }

    async fn get_range(&self, location: &Path, range: Range<usize>) -> OSResult<Bytes> {
        self.before_method("get_range", location)?;
        let result = self.target.get_range(location, range).await;
        self.after_method("get_range", location)?;
        result
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<usize>]) -> OSResult<Vec<Bytes>> {
        self.before_method("get_ranges", location)?;
        let result = self.target.get_ranges(location, ranges).await;
        self.after_method("get_ranges", location)?;
        result
    }

    async fn head(&self, location: &Path) -> OSResult<ObjectMeta> {
        self.before_method("head", location)?;
        let result = self.target.head(location).await;
        self.after_method("head", location)?;
        result
    }

    async fn delete(&self, location: &Path) -> OSResult<()> {
        self.before_method("delete", location)?;
        let result = self.target.delete(location).await;
        self.after_method("delete", location)?;
        result
    }

    async fn list(&self, prefix: Option<&Path>) -> OSResult<BoxStream<'_, OSResult<ObjectMeta>>> {
        self.target.list(prefix).await
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.target.list_with_delimiter(prefix).await
    }

    async fn copy(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.before_method("copy", from)?;
        let result = self.target.copy(from, to).await;
        self.after_method("copy", from)?;
        result
    }

    async fn rename(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.before_method("rename", from)?;
        let result = self.target.rename(from, to).await;
        self.after_method("rename", from)?;
        result
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()> {
        self.before_method("copy_if_not_exists", from)?;
        let result = self.target.copy_if_not_exists(from, to).await;
        self.after_method("copy_if_not_exists", from)?;
        result
    }
}

// Regrettably, the system clock is a process-wide global. That means that tests running
// in parallel can interfere with each other if they both want to adjust the system clock.
//
// By using MockClock below (which wraps mock_instant::MockClock), we can prevent this from
// happening, though there is a performance hit as this will prevent some potential test
// parallelism.
static CLOCK_MUTEX: Mutex<()> = Mutex::new(());
pub struct MockClock<'a> {
    _guard: MutexGuard<'a, ()>,
}

impl<'a> MockClock<'a> {
    pub fn new() -> Self {
        Self {
            _guard: CLOCK_MUTEX.lock().unwrap(),
        }
    }

    pub fn set_system_time(&self, time: Duration) {
        mock_instant::MockClock::set_system_time(time.to_std().unwrap());
    }
}

impl<'a> Drop for MockClock<'a> {
    fn drop(&mut self) {
        // Reset the clock to the epoch
        mock_instant::MockClock::set_system_time(Duration::days(0).to_std().unwrap());
    }
}
