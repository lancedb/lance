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
}

/// StorageOptionsProvider implementation that fetches options from a LanceNamespace
pub struct LanceNamespaceStorageOptionsProvider {
    namespace: Arc<dyn LanceNamespace>,
    table_id: Vec<String>,
}

impl fmt::Debug for LanceNamespaceStorageOptionsProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LanceNamespaceStorageOptionsProvider")
            .field("namespace", &self.namespace)
            .field("table_id", &self.table_id)
            .finish()
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
}
