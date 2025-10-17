// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Connect functionality for Lance Namespace implementations.

use std::collections::HashMap;
use std::sync::Arc;

use lance_core::{Error, Result};
use lance_namespace::LanceNamespace;

/// Connect to a Lance namespace implementation.
///
/// This function creates a connection to a Lance namespace backend based on
/// the specified implementation type and configuration properties.
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
/// let namespace = connect("rest", props).await?;
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
/// let namespace = connect("dir", props).await?;
/// # Ok(())
/// # }
/// ```
pub async fn connect(
    impl_name: &str,
    #[allow(unused)] properties: HashMap<String, String>,
) -> Result<Arc<dyn LanceNamespace>> {
    match impl_name {
        #[cfg(feature = "rest")]
        "rest" => {
            // Create REST implementation
            Ok(Arc::new(crate::rest::RestNamespace::new(properties)))
        }
        #[cfg(not(feature = "rest"))]
        "rest" => Err(Error::Namespace {
            source: "REST namespace implementation requires 'rest' feature to be enabled".into(),
            location: snafu::location!(),
        }),
        #[cfg(feature = "dir")]
        "dir" => {
            // Create directory implementation
            crate::dir::connect_dir(properties).await
        }
        #[cfg(not(feature = "dir"))]
        "dir" => Err(Error::Namespace {
            source: "Directory namespace implementation requires 'dir' feature to be enabled"
                .into(),
            location: snafu::location!(),
        }),
        _ => Err(Error::Namespace {
            source: format!(
                "Implementation '{}' is not available. Supported: {}{}",
                impl_name,
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
