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
        connect(&self.impl_name, self.properties, self.session).await
    }
}

/// Connect to a Lance namespace implementation.
///
/// **Note:** Consider using [`ConnectBuilder`] for a more ergonomic API.
///
/// This function creates a connection to a Lance namespace backend based on
/// the specified implementation type and configuration properties.
///
/// # Arguments
///
/// * `impl_name` - Implementation identifier. Supported values:
///   - "rest": REST API implementation (requires "rest" feature)
///   - "dir": Directory-based implementation (always available)
///
/// * `properties` - Configuration properties specific to the implementation.
///   Common properties:
///   - For REST: "uri" (base URL), "delimiter", "header.*" (custom headers)
///   - For DIR: "root" (directory path), "storage.*" (storage options)
///
/// * `session` - Optional Lance session to reuse object store registry
///
/// # Returns
///
/// Returns a boxed trait object implementing the `LanceNamespace` trait.
///
/// # Examples
///
/// ```no_run
/// use lance_namespace_impls::connect;
/// use std::collections::HashMap;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Connect to REST implementation
/// let mut props = HashMap::new();
/// props.insert("uri".to_string(), "http://localhost:8080".to_string());
/// let namespace = connect("rest", props, None).await?;
/// # Ok(())
/// # }
/// ```
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use lance_namespace_impls::connect;
/// use std::collections::HashMap;
///
/// // Connect to directory implementation
/// let mut props = HashMap::new();
/// props.insert("root".to_string(), "/path/to/data".to_string());
/// let namespace = connect("dir", props, None).await?;
/// # Ok(())
/// # }
/// ```
pub async fn connect(
    impl_name: &str,
    properties: HashMap<String, String>,
    session: Option<Arc<Session>>,
) -> Result<Arc<dyn LanceNamespace>> {
    match impl_name {
        #[cfg(feature = "rest")]
        "rest" => {
            // Create REST implementation (REST doesn't use session)
            Ok(Arc::new(crate::rest::RestNamespace::new(properties)))
        }
        #[cfg(not(feature = "rest"))]
        "rest" => Err(Error::Namespace {
            source: "REST namespace implementation requires 'rest' feature to be enabled".into(),
            location: snafu::location!(),
        }),
        "dir" => {
            // Create directory implementation (always available)
            crate::dir::connect_dir(properties, session).await
        }
        _ => Err(Error::Namespace {
            source: format!(
                "Implementation '{}' is not available. Supported: dir{}",
                impl_name,
                if cfg!(feature = "rest") { ", rest" } else { "" }
            )
            .into(),
            location: snafu::location!(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_namespace::models::ListTablesRequest;
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_connect_builder_basic() {
        let temp_dir = TempDir::new().unwrap();

        let namespace = ConnectBuilder::new("dir")
            .property("root", temp_dir.path().to_string_lossy().to_string())
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
        let temp_dir = TempDir::new().unwrap();
        let mut props = HashMap::new();
        props.insert("storage.option1".to_string(), "value1".to_string());

        let namespace = ConnectBuilder::new("dir")
            .property("root", temp_dir.path().to_string_lossy().to_string())
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
        let temp_dir = TempDir::new().unwrap();
        let session = Arc::new(Session::default());

        let namespace = ConnectBuilder::new("dir")
            .property("root", temp_dir.path().to_string_lossy().to_string())
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

    #[tokio::test]
    async fn test_connect_function_backwards_compat() {
        let temp_dir = TempDir::new().unwrap();
        let mut props = HashMap::new();
        props.insert(
            "root".to_string(),
            temp_dir.path().to_string_lossy().to_string(),
        );

        let namespace = connect("dir", props, None).await.unwrap();

        // Verify we can use the namespace
        let request = ListTablesRequest::new();
        let response = namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0);
    }
}
