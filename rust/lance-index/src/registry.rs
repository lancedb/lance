// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::{collections::HashMap, sync::Arc};

use lance_core::{Error, Result};
use snafu::location;

use crate::{
    pb, pbold,
    scalar::{
        bitmap::BitmapIndexPlugin, bloomfilter::BloomFilterIndexPlugin, btree::BTreeIndexPlugin,
        inverted::InvertedIndexPlugin, json::JsonIndexPlugin, label_list::LabelListIndexPlugin,
        ngram::NGramIndexPlugin, registry::ScalarIndexPlugin, rtree::RTreeIndexPlugin,
        zonemap::ZoneMapIndexPlugin,
    },
};

/// A registry of index plugins
pub struct IndexPluginRegistry {
    plugins: HashMap<String, Box<dyn ScalarIndexPlugin>>,
}

impl IndexPluginRegistry {
    fn get_plugin_name_from_details_name(&self, details_name: &str) -> String {
        let details_name = details_name.to_lowercase();
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

    /// Create a registry with the default plugins
    pub fn with_default_plugins() -> Arc<Self> {
        let mut registry = Self {
            plugins: HashMap::new(),
        };
        registry.add_plugin::<pbold::BTreeIndexDetails, BTreeIndexPlugin>();
        registry.add_plugin::<pbold::BitmapIndexDetails, BitmapIndexPlugin>();
        registry.add_plugin::<pbold::LabelListIndexDetails, LabelListIndexPlugin>();
        registry.add_plugin::<pbold::NGramIndexDetails, NGramIndexPlugin>();
        registry.add_plugin::<pbold::ZoneMapIndexDetails, ZoneMapIndexPlugin>();
        registry.add_plugin::<pb::BloomFilterIndexDetails, BloomFilterIndexPlugin>();
        registry.add_plugin::<pbold::InvertedIndexDetails, InvertedIndexPlugin>();
        registry.add_plugin::<pb::JsonIndexDetails, JsonIndexPlugin>();
        registry.add_plugin::<pb::RTreeIndexDetails, RTreeIndexPlugin>();

        let registry = Arc::new(registry);
        for plugin in registry.plugins.values() {
            plugin.attach_registry(registry.clone());
        }

        registry
    }

    /// Get an index plugin suitable for training an index with the given parameters
    pub fn get_plugin_by_name(&self, name: &str) -> Result<&dyn ScalarIndexPlugin> {
        self.plugins
            .get(name)
            .map(|plugin| plugin.as_ref())
            .ok_or_else(|| Error::InvalidInput {
                source: format!("No scalar index plugin found for name {}", name).into(),
                location: location!(),
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
}
