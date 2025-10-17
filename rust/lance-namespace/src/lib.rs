// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Namespace Rust Client
//!
//! A Rust client for the Lance Namespace API that provides a unified interface
//! for managing namespaces and tables across different backend implementations.

pub mod namespace;
pub mod schema;

// Re-export the trait at the crate root
pub use lance_core::{Error, Result};
pub use namespace::LanceNamespace;

// Re-export reqwest client for convenience
pub use lance_namespace_reqwest_client as reqwest_client;

// Re-export commonly used models from the reqwest client
pub mod models {
    pub use lance_namespace_reqwest_client::models::*;
}

// Re-export APIs from the reqwest client
pub mod apis {
    pub use lance_namespace_reqwest_client::apis::*;
}
