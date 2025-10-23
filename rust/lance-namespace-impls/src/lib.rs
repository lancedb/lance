// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Namespace implementations.
//!
//! This crate provides various implementations of the Lance Namespace trait.
//!
//! ## Features
//!
//! - `rest`: REST API-based namespace implementation
//! - `dir-aws`, `dir-azure`, `dir-gcp`, `dir-oss`: Cloud storage backend support for directory namespace (via lance-io)
//!
//! ## Implementations
//!
//! - `DirectoryNamespace`: Directory-based implementation (always available)
//! - `RestNamespace`: REST API-based implementation (requires `rest` feature)
//!
//! ## Usage
//!
//! The recommended way to connect to a namespace is using [`ConnectBuilder`]:
//!
//! ```no_run
//! # use lance_namespace_impls::ConnectBuilder;
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let namespace = ConnectBuilder::new("dir")
//!     .property("root", "/path/to/data")
//!     .connect()
//!     .await?;
//! # Ok(())
//! # }
//! ```

pub mod connect;
pub mod dir;

#[cfg(feature = "rest")]
pub mod rest;

// Re-export connect function and builder
pub use connect::{connect, ConnectBuilder};
pub use dir::{connect_dir, DirectoryNamespace, DirectoryNamespaceConfig};

#[cfg(feature = "rest")]
pub use rest::RestNamespace;
