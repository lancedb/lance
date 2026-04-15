// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cached table context from a namespace client.
//!
//! [`NamespaceClientTableContext`] holds the information that was returned by a
//! prior `describe_table` or `declare_table` call (location, storage options,
//! managed-versioning flag).
//!
//! Passing this struct avoids repeating the same parameters across every API
//! entry-point and makes it explicit that the values are cached from a prior
//! namespace call rather than user-provided overrides.  The namespace client
//! and table ID are **not** part of this struct — they are still passed
//! separately.

use std::collections::HashMap;

use lance_namespace_reqwest_client::models::{DeclareTableResponse, DescribeTableResponse};

/// Cached context from a namespace client's `describe_table` or `declare_table`
/// response.
///
/// Contains only the resolved table metadata (location, storage options,
/// managed-versioning flag).  The namespace client and table ID remain
/// separate parameters.
#[derive(Debug, Clone)]
pub struct NamespaceClientTableContext {
    /// The table's storage location (URI).
    pub location: String,
    /// Storage options returned by the namespace (e.g. temporary credentials).
    pub storage_options: Option<HashMap<String, String>>,
    /// Whether commits should go through the namespace's version API.
    pub managed_versioning: bool,
}

impl NamespaceClientTableContext {
    /// Build a context from a `DescribeTableResponse`.
    ///
    /// Returns an error if the response does not contain a `location`.
    pub fn from_describe_table_response(response: DescribeTableResponse) -> Result<Self, String> {
        let location = response
            .location
            .ok_or("DescribeTableResponse missing location")?;
        Ok(Self {
            location,
            storage_options: response.storage_options,
            managed_versioning: response.managed_versioning == Some(true),
        })
    }

    /// Build a context from a `DeclareTableResponse`.
    ///
    /// Returns an error if the response does not contain a `location`.
    pub fn from_declare_table_response(response: DeclareTableResponse) -> Result<Self, String> {
        let location = response
            .location
            .ok_or("DeclareTableResponse missing location")?;
        Ok(Self {
            location,
            storage_options: response.storage_options,
            managed_versioning: response.managed_versioning == Some(true),
        })
    }
}
