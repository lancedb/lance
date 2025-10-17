// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Namespace implementations.
//!
//! This crate provides various implementations of the Lance Namespace trait.
//!
//! ## Features
//!
//! - `dir`: Directory-based namespace implementation that stores tables as Lance datasets
//! - `rest`: REST client implementation that connects to a remote Lance REST server
//! - `rest-server`: REST server adapter that exposes any namespace via HTTP

pub mod connect;

#[cfg(feature = "rest")]
pub mod rest;

#[cfg(feature = "dir")]
pub mod dir;

#[cfg(feature = "rest-server")]
pub mod rest_adapter;

// Re-export connect function
pub use connect::connect;

#[cfg(feature = "rest")]
pub use rest::RestNamespace;

#[cfg(feature = "dir")]
pub use dir::{connect_dir, DirectoryNamespace, DirectoryNamespaceConfig};

#[cfg(feature = "rest-server")]
pub use rest_adapter::{RestServer, RestServerConfig};
