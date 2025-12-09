// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_schema::Field;
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use lance_core::{cache::LanceCache, Result};

use crate::registry::IndexPluginRegistry;
use crate::{
    frag_reuse::FragReuseIndex,
    scalar::{expression::ScalarQueryParser, CreatedIndex, IndexStore, ScalarIndex},
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

    /// A short name for the index
    ///
    /// This is a friendly name for display purposes and also can be used as an alias for
    /// the index type URL.  If multiple plugins have the same name, then the first one
    /// found will be used.
    ///
    /// By convention this is MixedCase with no spaces.  When used as an alias, it will be
    /// compared case-insensitively.
    fn name(&self) -> &str;

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

    /// Optional hook allowing a plugin to provide statistics without loading the index.
    async fn load_statistics(
        &self,
        _index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
    ) -> Result<Option<serde_json::Value>> {
        Ok(None)
    }

    /// Optional hook that plugins can use if they need to be aware of the registry
    fn attach_registry(&self, _registry: Arc<IndexPluginRegistry>) {}

    /// Returns a JSON string representation of the provided index details
    ///
    /// These details will be user-visible and should be considered part of the public
    /// API.  As a result, efforts should be made to ensure the information is backwards
    /// compatible and avoid breaking changes.
    fn details_as_json(&self, _details: &prost_types::Any) -> Result<serde_json::Value> {
        // Return an empty JSON object as the default implementation
        Ok(serde_json::json!({}))
    }
}
