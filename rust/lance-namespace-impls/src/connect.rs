// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Connect functionality for Lance Namespace implementations.

use std::collections::HashMap;
use std::sync::Arc;

use lance_core::{Error, Result};
use lance_namespace::LanceNamespace;

#[cfg(feature = "dir")]
use lance::session::Session;

/// Builder for connecting to a Lance namespace implementation.
///
/// This builder allows configuring connection parameters including an optional
/// shared `Session` for object store management.
///
/// # Examples
///
/// ```no_run
/// use lance_namespace_impls::connect;
/// use std::collections::HashMap;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Simple connection without session
/// let mut props = HashMap::new();
/// props.insert("uri".to_string(), "http://localhost:8080".to_string());
/// let namespace = connect("rest", props).connect().await?;
/// # Ok(())
/// # }
/// ```
///
/// ```no_run
/// # #[cfg(feature = "dir")]
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use lance_namespace_impls::connect;
/// use lance::session::Session;
/// use std::collections::HashMap;
/// use std::sync::Arc;
///
/// // Connection with shared session
/// let session = Arc::new(Session::default());
/// let mut props = HashMap::new();
/// props.insert("root".to_string(), "/path/to/data".to_string());
/// let namespace = connect("dir", props)
///     .with_session(session)
///     .connect()
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct ConnectBuilder {
    impl_name: String,
    properties: HashMap<String, String>,
    #[cfg(feature = "dir")]
    session: Option<Arc<Session>>,
}

impl ConnectBuilder {
    /// Create a new ConnectBuilder.
    ///
    /// # Arguments
    ///
    /// * `impl_name` - Implementation identifier ("rest", "dir", etc.)
    /// * `properties` - Configuration properties for the implementation
    pub fn new(impl_name: impl Into<String>, properties: HashMap<String, String>) -> Self {
        Self {
            impl_name: impl_name.into(),
            properties,
            #[cfg(feature = "dir")]
            session: None,
        }
    }

    /// Set a shared Session for object store management.
    ///
    /// This allows reusing an existing Session across multiple namespace connections,
    /// which can improve performance by sharing object store connections and caches.
    ///
    /// Currently only supported for the "dir" implementation.
    #[cfg(feature = "dir")]
    pub fn with_session(mut self, session: Arc<Session>) -> Self {
        self.session = Some(session);
        self
    }

    /// Connect to the namespace implementation.
    ///
    /// # Returns
    ///
    /// Returns a boxed trait object implementing the `LanceNamespace` trait.
    pub async fn connect(self) -> Result<Arc<dyn LanceNamespace>> {
        match self.impl_name.as_str() {
            #[cfg(feature = "rest")]
            "rest" => {
                // Create REST implementation
                Ok(Arc::new(crate::rest::RestNamespace::new(self.properties)))
            }
            #[cfg(not(feature = "rest"))]
            "rest" => Err(Error::Namespace {
                source: "REST namespace implementation requires 'rest' feature to be enabled"
                    .into(),
                location: snafu::location!(),
            }),
            #[cfg(feature = "dir")]
            "dir" => {
                // Create directory implementation with optional session
                crate::dir::connect_dir_with_session(self.properties, self.session).await
            }
            #[cfg(not(feature = "dir"))]
            "dir" => Err(Error::Namespace {
                source:
                    "Directory namespace implementation requires 'dir' feature to be enabled"
                        .into(),
                location: snafu::location!(),
            }),
            _ => Err(Error::Namespace {
                source: format!(
                    "Implementation '{}' is not available. Supported: {}{}",
                    self.impl_name,
                    if cfg!(feature = "rest") { "rest" } else { "" },
                    if cfg!(feature = "dir") {
                        if cfg!(feature = "rest") {
                            ", dir"
                        } else {
                            "dir"
                        }
                    } else {
                        ""
                    }
                )
                .into(),
                location: snafu::location!(),
            }),
        }
    }
}

/// Connect to a Lance namespace implementation.
///
/// This function creates a `ConnectBuilder` that can be configured and then used
/// to establish a connection to a Lance namespace backend.
///
/// # Arguments
///
/// * `impl_name` - Implementation identifier. Supported values:
///   - "rest": REST API implementation (requires "rest" feature)
///   - "dir": Directory-based implementation (requires "dir" feature)
///
/// * `properties` - Configuration properties specific to the implementation.
///   Common properties:
///   - For REST: "uri" (base URL), "delimiter", "header.*" (custom headers)
///   - For DIR: "root" (directory path), "storage.*" (storage options)
///
/// # Returns
///
/// Returns a `ConnectBuilder` that can be further configured (e.g., with a shared Session)
/// before calling `.connect()` to establish the connection.
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
/// let namespace = connect("rest", props).connect().await?;
/// # Ok(())
/// # }
/// ```
///
/// ```no_run
/// # #[cfg(feature = "dir")]
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use lance_namespace_impls::connect;
/// use std::collections::HashMap;
///
/// // Connect to directory implementation (requires "dir" feature)
/// let mut props = HashMap::new();
/// props.insert("root".to_string(), "/path/to/data".to_string());
/// let namespace = connect("dir", props).connect().await?;
/// # Ok(())
/// # }
/// ```
pub fn connect(impl_name: &str, properties: HashMap<String, String>) -> ConnectBuilder {
    ConnectBuilder::new(impl_name, properties)
}
