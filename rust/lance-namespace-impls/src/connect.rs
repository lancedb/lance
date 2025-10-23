// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Connect functionality for Lance Namespace implementations.

use std::collections::HashMap;
use std::sync::Arc;

use lance::session::Session;
use lance_core::{Error, Result};
use lance_namespace::LanceNamespace;

/// Builder for creating Lance namespace connections.
///
/// This builder provides a fluent API for configuring and establishing
/// connections to Lance namespace implementations.
///
/// # Examples
///
/// ```no_run
/// # use lance_namespace_impls::ConnectBuilder;
/// # use std::collections::HashMap;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Connect to directory implementation
/// let namespace = ConnectBuilder::new("dir")
///     .property("root", "/path/to/data")
///     .property("storage.region", "us-west-2")
///     .connect()
///     .await?;
/// # Ok(())
/// # }
/// ```
///
/// ```no_run
/// # use lance_namespace_impls::ConnectBuilder;
/// # use lance::session::Session;
/// # use std::sync::Arc;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Connect with a shared session
/// let session = Arc::new(Session::default());
/// let namespace = ConnectBuilder::new("dir")
///     .property("root", "/path/to/data")
///     .session(session)
///     .connect()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ConnectBuilder {
    impl_name: String,
    properties: HashMap<String, String>,
    session: Option<Arc<Session>>,
}

impl ConnectBuilder {
    /// Create a new ConnectBuilder for the specified implementation.
    ///
    /// # Arguments
    ///
    /// * `impl_name` - Implementation identifier ("dir", "rest", etc.)
    pub fn new(impl_name: impl Into<String>) -> Self {
        Self {
            impl_name: impl_name.into(),
            properties: HashMap::new(),
            session: None,
        }
    }

    /// Add a configuration property.
    ///
    /// # Arguments
    ///
    /// * `key` - Property key
    /// * `value` - Property value
    pub fn property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Add multiple configuration properties.
    ///
    /// # Arguments
    ///
    /// * `properties` - HashMap of properties to add
    pub fn properties(mut self, properties: HashMap<String, String>) -> Self {
        self.properties.extend(properties);
        self
    }

    /// Set the Lance session to use for this connection.
    ///
    /// When a session is provided, the namespace will reuse the session's
    /// object store registry, allowing multiple namespaces and datasets
    /// to share the same underlying storage connections.
    ///
    /// # Arguments
    ///
    /// * `session` - Arc-wrapped Lance session
    pub fn session(mut self, session: Arc<Session>) -> Self {
        self.session = Some(session);
        self
    }

    /// Build and establish the connection to the namespace.
    ///
    /// # Returns
    ///
    /// Returns a trait object implementing `LanceNamespace`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The implementation type is not supported
    /// - Required configuration properties are missing
    /// - Connection to the backend fails
    pub async fn connect(self) -> Result<Arc<dyn LanceNamespace>> {
        match self.impl_name.as_str() {
            #[cfg(feature = "rest")]
            "rest" => {
                // Create REST implementation (REST doesn't use session)
                crate::rest::RestNamespaceBuilder::from_properties(self.properties)
                    .map(|builder| Arc::new(builder.build()) as Arc<dyn LanceNamespace>)
            }
            #[cfg(not(feature = "rest"))]
            "rest" => Err(Error::Namespace {
                source: "REST namespace implementation requires 'rest' feature to be enabled"
                    .into(),
                location: snafu::location!(),
            }),
            "dir" => {
                // Create directory implementation (always available)
                crate::dir::DirectoryNamespaceBuilder::from_properties(
                    self.properties,
                    self.session,
                )?
                .build()
                .await
                .map(|ns| Arc::new(ns) as Arc<dyn LanceNamespace>)
            }
            _ => Err(Error::Namespace {
                source: format!(
                    "Implementation '{}' is not available. Supported: dir{}",
                    self.impl_name,
                    if cfg!(feature = "rest") { ", rest" } else { "" }
                )
                .into(),
                location: snafu::location!(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_core::utils::tempfile::TempStdDir;
    use lance_namespace::models::ListTablesRequest;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_connect_builder_basic() {
        let temp_dir = TempStdDir::default();

        let namespace = ConnectBuilder::new("dir")
            .property("root", temp_dir.to_str().unwrap())
            .connect()
            .await
            .unwrap();

        // Verify we can use the namespace
        let request = ListTablesRequest::new();
        let response = namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0);
    }

    #[tokio::test]
    async fn test_connect_builder_with_properties() {
        let temp_dir = TempStdDir::default();
        let mut props = HashMap::new();
        props.insert("storage.option1".to_string(), "value1".to_string());

        let namespace = ConnectBuilder::new("dir")
            .property("root", temp_dir.to_str().unwrap())
            .properties(props)
            .connect()
            .await
            .unwrap();

        // Verify we can use the namespace
        let request = ListTablesRequest::new();
        let response = namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0);
    }

    #[tokio::test]
    async fn test_connect_builder_with_session() {
        let temp_dir = TempStdDir::default();
        let session = Arc::new(Session::default());

        let namespace = ConnectBuilder::new("dir")
            .property("root", temp_dir.to_str().unwrap())
            .session(session.clone())
            .connect()
            .await
            .unwrap();

        // Verify we can use the namespace
        let request = ListTablesRequest::new();
        let response = namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0);
    }

    #[tokio::test]
    async fn test_connect_builder_invalid_impl() {
        let result = ConnectBuilder::new("invalid")
            .property("root", "/tmp")
            .connect()
            .await;

        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("not available"));
    }
}
