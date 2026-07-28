// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Minimal registry trait for scalar index plugins.
//!
//! This trait allows scalar index plugins to look up sibling plugins without
//! depending on the concrete [`IndexPluginRegistry`] type in `lance-index`.

use lance_core::Result;

/// A trait for looking up scalar index plugins.
///
/// Implemented by [`IndexPluginRegistry`](lance_index::registry::IndexPluginRegistry) in
/// `lance-index`. Plugins that need to delegate to other plugins (e.g. [`JsonIndexPlugin`])
/// depend on this trait rather than the concrete registry type, which would create a
/// circular dependency.
pub trait IndexRegistry: Send + Sync {
    /// Look up a plugin by its short name (case-insensitive).
    fn get_plugin_by_name(
        &self,
        name: &str,
    ) -> Result<&dyn crate::scalar::registry::ScalarIndexPlugin>;

    /// Look up a plugin by its index details protobuf message.
    fn get_plugin_by_details(
        &self,
        details: &prost_types::Any,
    ) -> Result<&dyn crate::scalar::registry::ScalarIndexPlugin>;
}
