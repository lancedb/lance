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

#[cfg(feature = "rest")]
pub mod rest;

#[cfg(feature = "dir")]
pub mod dir;

// Re-export connect function
pub use connect::connect;

#[cfg(feature = "rest")]
pub use rest::RestNamespace;

#[cfg(feature = "dir")]
pub use dir::{connect_dir, DirectoryNamespace, DirectoryNamespaceConfig};
