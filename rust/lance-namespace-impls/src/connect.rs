// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Connect functionality for Lance Namespace implementations.

use std::collections::HashMap;
use std::sync::Arc;

use lance_core::{Error, Result};
use lance_namespace::{LanceNamespace, RestNamespace};
use snafu::Location;

/// Connect to a Lance namespace implementation.
///
/// This function creates a connection to a Lance namespace backend based on
/// the specified implementation type and configuration properties.
///
/// # Arguments
///
/// * `impl_name` - Implementation identifier. Supported values:
///   - "rest": REST API implementation
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
    properties: HashMap<String, String>,
) -> Result<Arc<dyn LanceNamespace>> {
    match impl_name {
        "rest" => {
            // Create REST implementation
            Ok(Arc::new(RestNamespace::new(properties)))
        }
        #[cfg(feature = "dir")]
        "dir" => {
            // Create directory implementation
            crate::dir::connect_dir(properties).await
        }
        #[cfg(not(feature = "dir"))]
        "dir" => Err(Error::Namespace {
            source: "Directory namespace implementation requires 'dir' feature to be enabled".into(),
            location: Location::new(file!(), line!(), column!()),
        }),
        _ => Err(Error::Namespace {
            source: format!(
                "Implementation '{}' is not available. Supported: rest{}",
                impl_name,
                if cfg!(feature = "dir") { ", dir" } else { "" }
            ).into(),
            location: Location::new(file!(), line!(), column!()),
        }),
    }
}
