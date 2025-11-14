// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Namespace implementations.
//!
//! This crate provides various implementations of the Lance Namespace trait.
//!
//! ## Features
//!
//! - `rest`: REST API-based namespace implementation
//! - `rest-adapter`: REST server adapter that exposes any namespace via HTTP
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

#[cfg(feature = "rest-adapter")]
pub mod rest_adapter;

// Re-export connect builder
pub use connect::ConnectBuilder;
pub use dir::{manifest::ManifestNamespace, DirectoryNamespace, DirectoryNamespaceBuilder};

#[cfg(feature = "rest")]
pub use rest::{RestNamespace, RestNamespaceBuilder};

#[cfg(feature = "rest-adapter")]
pub use rest_adapter::{RestAdapter, RestAdapterConfig};
