// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Namespace implementations.
//!
//! This crate provides various implementations of the Lance Namespace trait.
//!
//! ## Features
//!
//! - `dir`: Directory-based namespace implementation that stores tables as Lance datasets

pub mod connect;

#[cfg(feature = "dir")]
pub mod dir;

// Re-export connect function and error type
pub use connect::{connect, ConnectError};

// Re-export RestNamespace from lance-namespace for convenience
pub use lance_namespace::RestNamespace;

#[cfg(feature = "dir")]
pub use dir::{connect_dir, DirectoryNamespace, DirectoryNamespaceConfig};
