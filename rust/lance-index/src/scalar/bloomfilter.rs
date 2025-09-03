// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bloom Filter Index
//!
//! Bloom Filter is a probabilistic data structure that allows for fast membership testing.
//! It is a space-efficient data structure that can be used to test whether an element is a member of a set.
//! It's an inexact filter - they may include false positives that require rechecking.

use crate::scalar::expression::{SargableQueryParser, ScalarQueryParser};
use crate::scalar::registry::{
    ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
};
use crate::scalar::{CreatedIndex, UpdateCriteria};
use crate::{pb, Any};
use serde::{Deserialize, Serialize};

use arrow_schema::Field;
use datafusion::execution::SendableRecordBatchStream;
use std::{collections::HashMap, sync::Arc};

use super::{AnyQuery, IndexStore, MetricsCollector, ScalarIndex, SearchResult};
use crate::scalar::FragReuseIndex;
use crate::vector::VectorIndex;
use crate::{Index, IndexType};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::cache::LanceCache;
use lance_core::Error;
use lance_core::Result;
use roaring::RoaringBitmap;
use snafu::location;

pub struct BloomFilterIndex {}

impl std::fmt::Debug for BloomFilterIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BloomFilterIndex").finish()
    }
}

impl DeepSizeOf for BloomFilterIndex {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        0
    }
}

impl BloomFilterIndex {
    async fn load(
        _store: Arc<dyn IndexStore>,
        _fri: Option<Arc<FragReuseIndex>>,
        _index_cache: LanceCache,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {}))
    }
}

#[async_trait]
impl Index for BloomFilterIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn VectorIndex>> {
        Err(Error::InvalidInput {
            source: "BloomFilter is not a vector index".into(),
            location: location!(),
        })
    }

    async fn prewarm(&self) -> Result<()> {
        // Not much to prewarm
        Ok(())
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "type": "BloomFilter",
        }))
    }

    fn index_type(&self) -> IndexType {
        IndexType::BloomFilter
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        // TODO: fix me
        Ok(RoaringBitmap::new())
    }
}

#[async_trait]
impl ScalarIndex for BloomFilterIndex {
    async fn search(
        &self,
        _query: &dyn AnyQuery,
        _metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        Ok(SearchResult::AtMost(Default::default()))
    }

    fn can_remap(&self) -> bool {
        false
    }

    /// Remap the row ids, creating a new remapped version of this index in `dest_store`
    async fn remap(
        &self,
        _mapping: &HashMap<u64, Option<u64>>,
        _dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        Err(Error::InvalidInput {
            source: "BloomFilter does not support remap".into(),
            location: location!(),
        })
    }

    /// Add the new data , creating an updated version of the index in `dest_store`
    async fn update(
        &self,
        _new_data: SendableRecordBatchStream,
        _dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        // TODO: Support me
        Err(Error::InvalidInput {
            source: "BloomFilter update not yet implemented".into(),
            location: location!(),
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        // TODO: What do i need???
        UpdateCriteria::only_new_data(TrainingCriteria::new(TrainingOrdering::None))
    }
}
#[derive(Debug, Default)]
pub struct BloomFilterIndexPlugin;

#[async_trait]
impl ScalarIndexPlugin for BloomFilterIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        _field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        let params = if params.is_empty() {
            BloomFilterIndexBuilderParams::default()
        } else {
            serde_json::from_str::<BloomFilterIndexBuilderParams>(params)?
        };

        Ok(Box::new(BloomFilterIndexTrainingRequest {
            params,
            criteria: TrainingCriteria::new(TrainingOrdering::None),
        }))
    }

    async fn train_index(
        &self,
        _data: SendableRecordBatchStream,
        _index_store: &dyn IndexStore,
        _request: Box<dyn TrainingRequest>,
    ) -> Result<CreatedIndex> {
        // Dummy implementation for now
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pb::BloomFilterIndexDetails::default())
                .unwrap(),
            index_version: 0,
        })
    }

    fn provides_exact_answer(&self) -> bool {
        false
    }

    fn version(&self) -> u32 {
        0
    }

    fn new_query_parser(
        &self,
        index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        Some(Box::new(SargableQueryParser::new(index_name, false)))
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(BloomFilterIndex::load(index_store, frag_reuse_index, cache).await?)
    }
}

#[derive(Debug)]
pub struct BloomFilterIndexTrainingRequest {
    pub params: BloomFilterIndexBuilderParams,
    pub criteria: TrainingCriteria,
}

impl TrainingRequest for BloomFilterIndexTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BloomFilterIndexBuilderParams {
    // Add fields as needed for real implementation
}
