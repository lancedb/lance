// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Storage options provider for dynamic credential fetching
//!
//! This module provides a trait for fetching storage options from various sources
//! (namespace servers, secret managers, etc.) with support for expiration tracking
//! and automatic refresh.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::{Error, Result};
use async_trait::async_trait;
use lance_namespace::models::DescribeTableRequest;
use lance_namespace::LanceNamespace;
use snafu::location;

/// Key for the expiration timestamp in storage options HashMap
pub const EXPIRES_AT_MILLIS_KEY: &str = "expires_at_millis";

/// Trait for providing storage options with expiration tracking
///
/// Implementations can fetch storage options from various sources (namespace servers,
/// secret managers, etc.) and are usable from Python/Java.
///
/// # Current Use Cases
///
/// - **Temporary Credentials**: Fetch short-lived AWS temporary credentials that expire
///   after a set time period, with automatic refresh before expiration
///
/// # Future Possible Use Cases
///
/// - **Dynamic Storage Location Resolution**: Resolve logical names to actual storage
///   locations (bucket aliases, S3 Access Points, region-specific endpoints) that may
///   change based on region, tier, data migration, or failover scenarios
/// - **Runtime S3 Tags Assignment**: Inject cost allocation tags, security labels, or
///   compliance metadata into S3 requests based on the current execution context (user,
///   application, workspace, etc.)
/// - **Dynamic Endpoint Configuration**: Update storage endpoints for disaster recovery,
///   A/B testing, or gradual migration scenarios
/// - **Just-in-time Permission Elevation**: Request elevated permissions only when needed
///   for sensitive operations, then immediately revoke them
/// - **Secret Manager Integration**: Fetch encryption keys from AWS Secrets Manager,
///   Azure Key Vault, or Google Secret Manager with automatic rotation
/// - **OIDC/SAML Federation**: Integrate with identity providers to obtain storage
///   credentials based on user identity and group membership
///
/// # Equality and Hashing
///
/// Implementations must provide `provider_id()` which returns a unique identifier for
/// equality and hashing purposes. Two providers with the same ID are considered equal
/// and will share the same cached ObjectStore in the registry.
#[async_trait]
pub trait StorageOptionsProvider: Send + Sync + fmt::Debug {
    /// Fetch fresh storage options
    ///
    /// Returns None if no storage options are available, or Some(HashMap) with the options.
    /// If the [`EXPIRES_AT_MILLIS_KEY`] key is present in the HashMap, it should contain the
    /// epoch time in milliseconds when the options expire, and credentials will automatically
    /// refresh before expiration.
    /// If [`EXPIRES_AT_MILLIS_KEY`] is not provided, the options are considered to never expire.
    async fn fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>>;

    /// Return a human-readable unique identifier for this provider instance
    ///
    /// This is used for equality comparison and hashing in the object store registry.
    /// Two providers with the same ID will be treated as equal and share the same cached
    /// ObjectStore.
    ///
    /// The ID should be human-readable for debugging and logging purposes.
    /// For example: `"namespace[dir(root=/data)],table[db$schema$table1]"`
    ///
    /// The ID should uniquely identify the provider's configuration.
    fn provider_id(&self) -> String;
}

/// StorageOptionsProvider implementation that fetches options from a LanceNamespace
pub struct LanceNamespaceStorageOptionsProvider {
    namespace: Arc<dyn LanceNamespace>,
    table_id: Vec<String>,
}

impl fmt::Debug for LanceNamespaceStorageOptionsProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.provider_id())
    }
}

impl fmt::Display for LanceNamespaceStorageOptionsProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.provider_id())
    }
}

impl LanceNamespaceStorageOptionsProvider {
    /// Create a new LanceNamespaceStorageOptionsProvider
    ///
    /// # Arguments
    /// * `namespace` - The namespace implementation to fetch storage options from
    /// * `table_id` - The table identifier
    pub fn new(namespace: Arc<dyn LanceNamespace>, table_id: Vec<String>) -> Self {
        Self {
            namespace,
            table_id,
        }
    }
}

#[async_trait]
impl StorageOptionsProvider for LanceNamespaceStorageOptionsProvider {
    async fn fetch_storage_options(&self) -> Result<Option<HashMap<String, String>>> {
        let request = DescribeTableRequest {
            id: Some(self.table_id.clone()),
            version: None,
        };

        let response = self
            .namespace
            .describe_table(request)
            .await
            .map_err(|e| Error::IO {
                source: Box::new(std::io::Error::other(format!(
                    "Failed to fetch storage options: {}",
                    e
                ))),
                location: location!(),
            })?;

        Ok(response.storage_options)
    }

    fn provider_id(&self) -> String {
        format!(
            "LanceNamespaceStorageOptionsProvider {{ namespace: {}, table_id: {:?} }}",
            self.namespace.namespace_id(),
            self.table_id
        )
    }
}
