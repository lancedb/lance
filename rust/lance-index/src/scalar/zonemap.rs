// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Zone Map Index
//!
//! Zone maps are a columnar database technique for predicate pushdown and scan pruning.
//! They break data into fixed-size chunks called "zones" and maintain summary statistics
//! (min, max, null count) for each zone. This enables efficient filtering by eliminating
//! zones that cannot contain matching values.
//!
//! Zone maps are "inexact" filters - they can definitively exclude zones but may include
//! false positives that require rechecking.
//!
//!
use super::btree::TrainingSource;
use crate::Any;
use std::env::temp_dir;
use std::sync::LazyLock;
use tempfile::{tempdir, TempDir};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::execution::SendableRecordBatchStream;
use datafusion_common::ScalarValue;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use super::{AnyQuery, IndexReader, IndexStore, MetricsCollector, ScalarIndex, SearchResult};
use crate::scalar::FragReuseIndex;
use crate::vector::VectorIndex;
use crate::{Index, IndexType};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::Result;
use lance_core::{utils::mask::RowIdTreeMap, Error};
use roaring::RoaringBitmap;
use snafu::location;
const DEFAULT_ZONE_SIZE: u32 = 4096;

/// Basic stats about zonemap index
#[derive(Debug)]
struct ZoneMapStatistics {
    zone_id: u64,
    min: ScalarValue,
    max: ScalarValue,
    null_count: u64,
    pub row_ids: RowIdTreeMap,
}

impl DeepSizeOf for ZoneMapStatistics {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        // Estimate sizes for ScalarValue
        let min_size = match &self.min {
            ScalarValue::Int32(_) => 4,
            ScalarValue::Int64(_) => 8,
            ScalarValue::Float32(_) => 4,
            ScalarValue::Float64(_) => 8,
            ScalarValue::Utf8(Some(s)) => s.len(),
            ScalarValue::Utf8(None) => 0,
            ScalarValue::LargeUtf8(None) => 0,
            _ => 16, // Default estimate
        };

        let max_size = match &self.max {
            ScalarValue::Int32(_) => 4,
            ScalarValue::Int64(_) => 8,
            ScalarValue::Float32(_) => 4,
            ScalarValue::Float64(_) => 8,
            ScalarValue::Utf8(Some(s)) => s.len(),
            ScalarValue::Utf8(None) => 0,
            ScalarValue::LargeUtf8(None) => 0,
            _ => 16, // Default estimate
        };

        min_size + max_size + 8 + self.row_ids.serialized_size()
    }
}

/// ZoneMap index
/// At high level it's a columnar database technique for predicate push down and scan pruning.
/// It breaks data into fixed-size chunks called `zones` and store summary statistics(min, max, null_count) for each zone. It enables efficient filtering by skipping zones that do not contain matching values
///
/// This is an inexact filter, similar to a bloom filter. It can return false positives that require rechecking.
///
/// Note that it cannot return false negatives.
pub struct ZoneMapIndex {
    zones: Vec<ZoneMapStatistics>,
    data_type: DataType,
}

impl std::fmt::Debug for ZoneMapIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZoneMapIndex")
            .field("data_type", &self.data_type)
            .finish()
    }
}

impl DeepSizeOf for ZoneMapIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.zones.deep_size_of_children(context)
    }
}

impl ZoneMapIndex {
    async fn hello_world() -> u32 {
        42
    }
}

#[async_trait]
impl Index for ZoneMapIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn VectorIndex>> {
        Err(Error::InvalidInput {
            source: "ZoneMapIndex is not a vector index".into(),
            location: location!(),
        })
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "type": "ZoneMap",
            "num_zones": self.zones.len()
        }))
    }

    async fn prewarm(&self) -> Result<()> {
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::ZoneMap
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let frag_ids = RoaringBitmap::new();
        Ok(frag_ids)
    }
}

#[async_trait]
impl ScalarIndex for ZoneMapIndex {
    async fn search(
        &self,
        _query: &dyn AnyQuery,
        _metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        // TODO: Implement actual search logic
        Ok(SearchResult::AtMost(RowIdTreeMap::new()))
    }

    fn can_answer_exact(&self, _: &dyn AnyQuery) -> bool {
        false
    }

    /// Load the scalar index from storage
    async fn load(
        _store: Arc<dyn IndexStore>,
        _fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        // TODO: Implement actual loading logic
        // For now, return a placeholder implementation
        Err(Error::InvalidInput {
            source: "ZoneMapIndex::load not yet implemented".into(),
            location: location!(),
        })
    }

    /// Remap the row ids, creating a new remapped version of this index in `dest_store`
    async fn remap(
        &self,
        _mapping: &HashMap<u64, Option<u64>>,
        _dest_store: &dyn IndexStore,
    ) -> Result<()> {
        // TODO: Implement remap logic
        Ok(())
    }

    /// Add the new data into the index, creating an updated version of the index in `dest_store`
    async fn update(
        &self,
        _new_data: SendableRecordBatchStream,
        _dest_store: &dyn IndexStore,
    ) -> Result<()> {
        // TODO: Implement update logic
        Ok(())
    }
}
#[derive(Debug, Clone)]
pub struct ZoneMapIndexBuilderOptions {
    rows_per_zone: usize,
}

/// TODO: Is 10,000 a good default value?
static DEFAULT_ROWS_PER_ZONE: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("LANCE_ZONEMAP_DEFAULT_ROWS_PER_ZONE")
        .unwrap_or_else(|_| "10000".to_string())
        .parse()
        .expect("failed to parse Lance_ZONEMAP_DEFAULT_ROWS_PER_ZONE")
});

impl Default for ZoneMapIndexBuilderOptions {
    fn default() -> Self {
        Self {
            rows_per_zone: *DEFAULT_ROWS_PER_ZONE,
        }
    }
}

// A builder for zonemap index
pub struct ZoneMapIndexBuilder {
    options: ZoneMapIndexBuilderOptions,
    tmpdir: Arc<TempDir>,
    has_flushed: bool,
}

impl ZoneMapIndexBuilder {
    fn try_new(options: ZoneMapIndexBuilderOptions) -> Result<Self> {
        let tmpdir = Arc::new(tempdir()?);
        Ok(Self {
            options,
            tmpdir,
            has_flushed: false,
        })
    }

    pub async fn train(&mut self, data: SendableRecordBatchStream) -> Result<Vec<usize>> {
        let mut to_spill = Vec::with_capacity(10);
        Ok(to_spill)
    }

    pub async fn write_index(
        mut self,
        store: &dyn IndexStore,
        spill_files: Vec<usize>,
    ) -> Result<()> {
        Ok(())
    }
}

pub async fn train_zonemap_index(
    data_source: Box<dyn TrainingSource + Send>,
    index_store: &dyn IndexStore,
) -> Result<()> {
    // TODO: Implement actual training logic
    let batches_source = data_source.scan_unordered_chunks(DEFAULT_ZONE_SIZE).await?;

    let mut builder = ZoneMapIndexBuilder::try_new(ZoneMapIndexBuilderOptions::default())?;

    // Calculate the output
    let split_files = builder.train(batches_source).await?;

    //  Write it
    builder.write_index(index_store, split_files).await
}
