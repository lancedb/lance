// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

use snafu::location;
use url::Url;

use super::{ObjectStore, ObjectStoreParams};
use lance_core::error::{Error, Result};

#[cfg(feature = "aws")]
pub mod aws;
#[cfg(feature = "azure")]
pub mod azure;
#[cfg(feature = "gcp")]
pub mod gcp;
pub mod local;
pub mod memory;

#[async_trait::async_trait]
pub trait ObjectStoreProvider: std::fmt::Debug + Sync + Send {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore>;
}

#[derive(Debug)]
pub struct ObjectStoreRegistry {
    providers: HashMap<String, Arc<dyn ObjectStoreProvider>>,
    // TODO: should the registry itself have a cache?
    // Cache should probable have weak references?
}

impl ObjectStoreRegistry {
    pub fn empty() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    pub fn get_provider(&self, scheme: &str) -> Option<Arc<dyn ObjectStoreProvider>> {
        self.providers.get(scheme).cloned()
    }

    pub async fn get_store(
        &self,
        base_path: Url,
        params: &ObjectStoreParams,
    ) -> Result<ObjectStore> {
        // TODO: caching
        let scheme = base_path.scheme();
        let provider = self.get_provider(scheme).ok_or_else(|| {
            Error::invalid_input(
                format!("No object store provider found for scheme: {}", scheme),
                location!(),
            )
        })?;
        provider.new_store(base_path, params).await
    }
}

impl Default for ObjectStoreRegistry {
    fn default() -> Self {
        let mut registry = Self {
            providers: HashMap::new(),
        };
        registry.insert("memory", Arc::new(memory::MemoryStoreProvider));
        registry.insert("file", Arc::new(local::FileStoreProvider));
        registry.insert("file-object-store", Arc::new(local::FileStoreProvider));
        #[cfg(feature = "aws")]
        {
            let aws = Arc::new(aws::AwsStoreProvider);
            registry.insert("s3", aws.clone());
            registry.insert("s3+ddb", aws);
        }
        #[cfg(feature = "azure")]
        registry.insert("az", Arc::new(azure::AzureBlobStoreProvider));
        #[cfg(feature = "gcp")]
        registry.insert("gs", Arc::new(gcp::GcsStoreProvider));
        registry
    }
}

impl ObjectStoreRegistry {
    pub fn insert(&mut self, scheme: &str, provider: Arc<dyn ObjectStoreProvider>) {
        self.providers.insert(scheme.into(), provider);
    }
}
