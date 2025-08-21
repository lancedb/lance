// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//!Bloom Filter Index
//!
//! Bloom Filter is a probabilistic data structure that allows for fast membership testing.
//! It is a space-efficient data structure that can be used to test whether an element is a member of a set.
//! It's an inexact filter - they may include false positives that require rechecking.

use crate::scalar::expression::{SargableQueryParser, ScalarQueryParser};
use crate::scalar::registry::{
    ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
};
use crate::scalar::{CreatedIndex, SargableQuery, UpdateCriteria};
use crate::{pb, Any};
use datafusion::functions_aggregate::min_max::{MaxAccumulator, MinAccumulator};
use datafusion_expr::Accumulator;
use futures::TryStreamExt;
use lance_core::cache::LanceCache;
use lance_core::ROW_ADDR;
use lance_datafusion::chunker::chunk_concat_stream;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

use arrow_array::{new_empty_array, ArrayRef, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field};
use datafusion::execution::SendableRecordBatchStream;
use datafusion_common::ScalarValue;
use std::{collections::HashMap, sync::Arc};

use super::{AnyQuery, IndexStore, MetricsCollector, ScalarIndex, SearchResult};
use crate::scalar::FragReuseIndex;
use crate::vector::VectorIndex;
use crate::{Index, IndexType};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::Result;
use lance_core::{utils::mask::RowIdTreeMap, Error};
use roaring::RoaringBitmap;
use snafu::location;

pub struct BloomFilterIndex {}

impl std::fmt::Debug for BloomFilterIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BloomFilterIndex")
    }
}

impl DeepSizeOf for BloomFilterIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        0
    }
}

impl BloomFilterIndex {
    async fn load(
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<FragReuseIndex>>,
        index_cache: LanceCache,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {}))
    }

    fn try_from_serialized(
        data: RecordBatch,
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<FragReuseIndex>>,
        index_cache: LanceCache,
        max_zonemap_size: u64,
    ) -> Result<Self> {
        Ok(Self {})
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
        Ok(())
    }
}

#[async_trait]
impl ScalarIndex for BloomFilterIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
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
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        // TODO: Support me
        Ok(())
    }

    fn update_criteria(&self) -> UpdateCriteria {
        // TODO: What do i need???
        UpdateCriteria::only_new_data(TrainingCriteria::new(TrainingOrdering::None))
    }
}
#[derive(Debug, Default)]
pub struct BloomFilterIndexPlugin;
