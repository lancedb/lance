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
        bitmap::BitmapIndexPlugin, btree::BTreeIndexPlugin, expression::ScalarQueryParser,
        inverted::InvertedIndexPlugin, label_list::LabelListIndexPlugin, ngram::NGramIndexPlugin,
        zonemap::ZoneMapIndexPlugin, CreatedIndex, IndexStore, ScalarIndex,
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

/// A trait for scalar index plugins
#[async_trait]
pub trait ScalarIndexPlugin: Send + Sync + std::fmt::Debug {
    /// Return the criteria data needs to satisfy to train this index
    ///
    /// For example, does the index require the input data to be sorted?
    fn training_criteria(&self, params: &prost_types::Any) -> Result<TrainingCriteria>;

    /// Checks if the index can be trained on the given field
    ///
    /// The caller should call this to validate the index is applicable before calling
    /// train_index.
    ///
    /// Returns an error with more details if it cannot
    fn check_can_train(&self, field: &Field) -> Result<()>;

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
        params: &prost_types::Any,
    ) -> Result<CreatedIndex>;

    /// Load an index from storage
    ///
    /// The index details should match the details that were returned when the index was
    /// originally trained.
    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>>;
}

/// A registry of scalar index plugins
pub struct ScalarIndexPluginRegistry {
    params_url_to_plugin: HashMap<String, Box<dyn ScalarIndexPlugin>>,
    details_url_to_plugin: HashMap<String, Box<dyn ScalarIndexPlugin>>,
}

impl Default for ScalarIndexPluginRegistry {
    fn default() -> Self {
        Self::with_default_plugins()
    }
}

impl ScalarIndexPluginRegistry {
    pub fn add_plugin<
        T: prost::Message + prost::Name,
        L: prost::Message + prost::Name,
        P: ScalarIndexPlugin + std::default::Default + 'static,
    >(
        &mut self,
    ) {
        self.params_url_to_plugin
            .insert(T::type_url(), Box::new(P::default()));
        self.details_url_to_plugin
            .insert(L::type_url(), Box::new(P::default()));
    }

    pub fn add_plugin_with_alias<P: ScalarIndexPlugin + std::default::Default + 'static>(
        &mut self,
        params_url: impl Into<String>,
        details_url: impl Into<String>,
    ) {
        self.params_url_to_plugin
            .insert(params_url.into(), Box::new(P::default()));
        self.details_url_to_plugin
            .insert(details_url.into(), Box::new(P::default()));
    }

    /// Create a registry with the default plugins
    pub fn with_default_plugins() -> Self {
        let mut registry = Self {
            params_url_to_plugin: HashMap::new(),
            details_url_to_plugin: HashMap::new(),
        };
        registry.add_plugin::<pb::BTreeIndexParams, pb::BTreeIndexDetails, BTreeIndexPlugin>();
        registry.add_plugin::<pb::BitmapIndexParams, pb::BitmapIndexDetails, BitmapIndexPlugin>();
        registry.add_plugin::<pb::LabelListIndexParams, pb::LabelListIndexDetails, LabelListIndexPlugin>();
        registry.add_plugin::<pb::NGramIndexParams, pb::NGramIndexDetails, NGramIndexPlugin>();
        registry
            .add_plugin::<pb::ZoneMapIndexParams, pb::ZoneMapIndexDetails, ZoneMapIndexPlugin>();
        registry
            .add_plugin::<pb::InvertedIndexParams, pb::InvertedIndexDetails, InvertedIndexPlugin>();

        // Older versions of lance had the index parameters / details in the table format package
        //
        // No need to update this list for new plugins
        registry.add_plugin_with_alias::<InvertedIndexPlugin>(
            "/lance.table.InvertedIndexParams",
            "/lance.table.InvertedIndexDetails",
        );
        registry.add_plugin_with_alias::<BitmapIndexPlugin>(
            "/lance.table.BitmapIndexParams",
            "/lance.table.BitmapIndexDetails",
        );
        registry.add_plugin_with_alias::<BTreeIndexPlugin>(
            "/lance.table.BTreeIndexParams",
            "/lance.table.BTreeIndexDetails",
        );
        registry.add_plugin_with_alias::<NGramIndexPlugin>(
            "/lance.table.NGramIndexParams",
            "/lance.table.NGramIndexDetails",
        );
        registry.add_plugin_with_alias::<ZoneMapIndexPlugin>(
            "/lance.table.ZoneMapIndexParams",
            "/lance.table.ZoneMapIndexDetails",
        );
        registry.add_plugin_with_alias::<LabelListIndexPlugin>(
            "/lance.table.LabelListIndexParams",
            "/lance.table.LabelListIndexDetails",
        );

        registry
    }

    /// Get an index plugin suitable for training an index with the given parameters
    pub fn get_plugin_by_training_params(
        &self,
        params: &prost_types::Any,
    ) -> Result<&dyn ScalarIndexPlugin> {
        self.params_url_to_plugin
            .get(params.type_url.as_str())
            .map(|plugin| plugin.as_ref())
            .ok_or_else(|| Error::InvalidInput {
                source: format!("No scalar index plugin found for type {}", params.type_url).into(),
                location: location!(),
            })
    }

    pub fn get_plugin_by_loading_details(
        &self,
        details: &prost_types::Any,
    ) -> Result<&dyn ScalarIndexPlugin> {
        self.details_url_to_plugin
            .get(details.type_url.as_str())
            .map(|plugin| plugin.as_ref())
            .ok_or_else(|| Error::InvalidInput {
                source: format!("No scalar index plugin found for type {}", details.type_url)
                    .into(),
                location: location!(),
            })
    }
}
