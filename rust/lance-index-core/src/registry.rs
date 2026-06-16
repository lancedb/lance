// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;

use lance_core::{Error, Result};

use crate::scalar::registry::ScalarIndexPlugin;

/// Abstract interface for a registry of index plugins.
///
/// This trait is used by [`ScalarIndexPlugin::attach_registry`] to give plugins
/// a handle to the registry so they can look up other plugins (e.g., to delegate
/// loading to another plugin).
pub trait PluginRegistry: Send + Sync {
    fn get_plugin_by_name(&self, name: &str) -> Result<&dyn ScalarIndexPlugin>;
    fn get_plugin_by_details(&self, details: &prost_types::Any) -> Result<&dyn ScalarIndexPlugin>;
}

/// A registry of index plugins
pub struct IndexPluginRegistry {
    plugins: HashMap<String, Box<dyn ScalarIndexPlugin>>,
}

impl IndexPluginRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    fn normalize_plugin_name(name: &str) -> String {
        name.to_lowercase()
    }

    fn get_plugin_name_from_details_name(&self, details_name: &str) -> String {
        let details_name = Self::normalize_plugin_name(details_name);
        if details_name.ends_with("indexdetails") {
            details_name.replace("indexdetails", "")
        } else {
            details_name
        }
    }

    /// Adds a plugin to the registry, using the name of the details message to determine
    /// the plugin name.
    ///
    /// The plugin name will be the lowercased name of the details message with any trailing
    /// "indexdetails" removed.
    ///
    /// For example, if the details message is `BTreeIndexDetails`, the plugin name will be
    /// `btree`.
    pub fn add_plugin<
        DetailsType: prost::Message + prost::Name,
        PluginType: ScalarIndexPlugin + std::default::Default + 'static,
    >(
        &mut self,
    ) {
        let plugin_name = self.get_plugin_name_from_details_name(DetailsType::NAME);
        self.plugins
            .insert(plugin_name, Box::new(PluginType::default()));
    }

    /// Get an index plugin suitable for training an index with the given parameters
    pub fn get_plugin_by_name(&self, name: &str) -> Result<&dyn ScalarIndexPlugin> {
        let plugin_name = Self::normalize_plugin_name(name);
        self.plugins
            .get(&plugin_name)
            .map(|plugin| plugin.as_ref())
            .ok_or_else(|| {
                let hint = if plugin_name == "rtree" {
                    ". The 'rtree' index requires the `geo` feature. \
                     Rebuild with `--features geo` to enable geospatial support"
                } else {
                    ""
                };
                Error::invalid_input_source(
                    format!("No scalar index plugin found for name '{name}'{hint}").into(),
                )
            })
    }

    pub fn get_plugin_by_details(
        &self,
        details: &prost_types::Any,
    ) -> Result<&dyn ScalarIndexPlugin> {
        let details_name = details.type_url.split('.').next_back().unwrap();
        let plugin_name = self.get_plugin_name_from_details_name(details_name);
        self.get_plugin_by_name(&plugin_name)
    }

    /// Call a closure for each registered plugin
    pub fn for_each_plugin(&self, mut f: impl FnMut(&dyn ScalarIndexPlugin)) {
        for plugin in self.plugins.values() {
            f(plugin.as_ref());
        }
    }
}

impl Default for IndexPluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRegistry for IndexPluginRegistry {
    fn get_plugin_by_name(&self, name: &str) -> Result<&dyn ScalarIndexPlugin> {
        Self::get_plugin_by_name(self, name)
    }

    fn get_plugin_by_details(&self, details: &prost_types::Any) -> Result<&dyn ScalarIndexPlugin> {
        Self::get_plugin_by_details(self, details)
    }
}
