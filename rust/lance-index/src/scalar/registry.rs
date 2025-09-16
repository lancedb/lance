// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

use arrow_schema::Field;
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use lance_core::{cache::LanceCache, Error, Result};
use snafu::location;

use crate::{
    frag_reuse::FragReuseIndex,
    pb,
    scalar::{
        bitmap::BitmapIndexPlugin, bloomfilter::BloomFilterIndexPlugin, btree::BTreeIndexPlugin,
        expression::ScalarQueryParser, inverted::InvertedIndexPlugin, json::JsonIndexPlugin,
        label_list::LabelListIndexPlugin, ngram::NGramIndexPlugin, zonemap::ZoneMapIndexPlugin,
        CreatedIndex, IndexStore, ScalarIndex,
    },
};

pub const VALUE_COLUMN_NAME: &str = "value";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingOrdering {
    /// The input will arrive sorted by the value column in ascending order
    Values,
    /// The input will arrive sorted by the address column in ascending order
    Addresses,
    /// The input will arrive in an arbitrary order
    None,
}

#[derive(Debug, Clone)]
pub struct TrainingCriteria {
    pub ordering: TrainingOrdering,
    pub needs_row_ids: bool,
    pub needs_row_addrs: bool,
}

impl TrainingCriteria {
    pub fn new(ordering: TrainingOrdering) -> Self {
        Self {
            ordering,
            needs_row_ids: false,
            needs_row_addrs: false,
        }
    }

    pub fn with_row_id(mut self) -> Self {
        self.needs_row_ids = true;
        self
    }

    pub fn with_row_addr(mut self) -> Self {
        self.needs_row_addrs = true;
        self
    }
}

/// A trait that describes what criteria is needed to train an index
///
/// The training process has two steps.  First, the parameters are given to the
/// plugin and it creates a TrainingRequest.  Then, the caller prepares the training
/// data and calls train_index.
///
/// The call to train_index will include the training request.  This allows the plugin
/// to stash any deserialized parameter info in the request and fetch it later during
/// training by downcasting to the appropriate type.
pub trait TrainingRequest: std::any::Any + Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
    fn criteria(&self) -> &TrainingCriteria;
}

/// A default training request impl for indexes that don't need any parameters
pub(crate) struct DefaultTrainingRequest {
    criteria: TrainingCriteria,
}

impl DefaultTrainingRequest {
    pub fn new(criteria: TrainingCriteria) -> Self {
        Self { criteria }
    }
}

impl TrainingRequest for DefaultTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

/// A trait for scalar index plugins
#[async_trait]
pub trait ScalarIndexPlugin: Send + Sync + std::fmt::Debug {
    /// Creates a new training request from the given parameters
    ///
    /// This training request specifies the criteria that the data must satisfy to train the index.
    /// For example, does the index require the input data to be sorted?
    fn new_training_request(&self, params: &str, field: &Field)
        -> Result<Box<dyn TrainingRequest>>;

    /// Train a new index
    ///
    /// The provided data must fulfill all the criteria returned by `training_criteria`.
    /// It is the caller's responsibility to ensure this.
    ///
    /// Returns index details that describe the index.  These details can potentially be
    /// useful for planning (although this will currently require inside information on
    /// the index type) and they will need to be provided when loading the index.
    ///
    /// It is the caller's responsibility to store these details somewhere.
    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        fragment_ids: Option<Vec<u32>>,
    ) -> Result<CreatedIndex>;

    /// Returns true if the index returns an exact answer (e.g. not AtMost)
    fn provides_exact_answer(&self) -> bool;

    /// The version of the index plugin
    ///
    /// We assume that indexes are not forwards compatible.  If an index was written with a
    /// newer version than this, it cannot be read
    fn version(&self) -> u32;

    /// Returns a new query parser for the index
    ///
    /// Can return None if this index cannot participate in query optimization
    fn new_query_parser(
        &self,
        index_name: String,
        index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>>;

    /// Load an index from storage
    ///
    /// The index details should match the details that were returned when the index was
    /// originally trained.
    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>>;

    /// Optional hook that plugins can use if they need to be aware of the registry
    fn attach_registry(&self, _registry: Arc<ScalarIndexPluginRegistry>) {}
}

/// A registry of scalar index plugins
pub struct ScalarIndexPluginRegistry {
    plugins: HashMap<String, Box<dyn ScalarIndexPlugin>>,
}

impl ScalarIndexPluginRegistry {
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
        registry.add_plugin::<pb::BTreeIndexDetails, BTreeIndexPlugin>();
        registry.add_plugin::<pb::BitmapIndexDetails, BitmapIndexPlugin>();
        registry.add_plugin::<pb::LabelListIndexDetails, LabelListIndexPlugin>();
        registry.add_plugin::<pb::NGramIndexDetails, NGramIndexPlugin>();
        registry.add_plugin::<pb::ZoneMapIndexDetails, ZoneMapIndexPlugin>();
        registry.add_plugin::<pb::BloomFilterIndexDetails, BloomFilterIndexPlugin>();
        registry.add_plugin::<pb::InvertedIndexDetails, InvertedIndexPlugin>();
        registry.add_plugin::<pb::JsonIndexDetails, JsonIndexPlugin>();

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
