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

pub mod connect;
pub mod dir;

#[cfg(feature = "rest")]
pub mod rest;

// Re-export connect function
pub use connect::connect;
pub use dir::{connect_dir, DirectoryNamespace, DirectoryNamespaceConfig};

#[cfg(feature = "rest")]
pub use rest::RestNamespace;
