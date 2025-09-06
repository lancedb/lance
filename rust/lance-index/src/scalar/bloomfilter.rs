// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bloom Filter Index
//!
//! Bloom Filter is a probabilistic data structure that allows for fast membership testing.
//! It is a space-efficient data structure that can be used to test whether an element is a member of a set.
//! It's an inexact filter - they may include false positives that require rechecking.

use crate::scalar::bloomfilter::sbbf::{Sbbf, SbbfBuilder};
use crate::scalar::expression::{SargableQueryParser, ScalarQueryParser};
use crate::scalar::registry::{
    ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
};
use crate::scalar::{CreatedIndex, SargableQuery, UpdateCriteria};
use crate::{pb, Any};
use arrow_array::{Array, UInt64Array};
use lance_core::utils::mask::RowIdTreeMap;
use lance_core::ROW_ADDR;
use lance_datafusion::chunker::chunk_concat_stream;
mod sbbf;
use arrow_schema::{DataType, Field};
use futures::TryStreamExt;
use serde::{Deserialize, Serialize};

use std::sync::LazyLock;

use datafusion::execution::SendableRecordBatchStream;
use parquet::data_type::AsBytes;
use std::{collections::HashMap, sync::Arc};

use crate::scalar::FragReuseIndex;
use crate::scalar::{AnyQuery, IndexStore, MetricsCollector, ScalarIndex, SearchResult};
use crate::vector::VectorIndex;
use crate::{Index, IndexType};
use arrow_array::{ArrayRef, RecordBatch};
use async_trait::async_trait;
use deepsize::DeepSizeOf;
use lance_core::cache::LanceCache;
use lance_core::Error;
use lance_core::Result;
use roaring::RoaringBitmap;
use snafu::location;

const BLOOMFILTER_FILENAME: &str = "bloomfilter.lance";
const BLOOMFILTER_ITEM_META_KEY: &str = "bloomfilter_item";
const BLOOMFILTER_BITS_META_KEY: &str = "bloomfilter_bits";
const BLOOMFILTER_INDEX_VERSION: u32 = 0;

#[derive(Debug, PartialEq, Clone, DeepSizeOf)]
struct BloomFilterStatistics {
    fragment_id: u64,
    // block_start is the start row of the block in the fragment, also known
    // as local row offset
    block_start: u64,
    block_size: usize,
    // Serialized bloom filter data (SBBF bytes)
    bloom_filter_data: Vec<u8>,
}

pub struct BloomFilterIndex {
    blocks: Vec<BloomFilterStatistics>,
    number_of_items: u64,
    number_of_bytes: usize,

    store: Arc<dyn IndexStore>,
    fri: Option<Arc<FragReuseIndex>>,
    index_cache: LanceCache,
}

impl std::fmt::Debug for BloomFilterIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BloomFilterIndex")
            .field("blocks", &self.blocks)
            .field("number_of_items", &self.number_of_items)
            .field("number_of_bytes", &self.number_of_bytes)
            .field("store", &self.store)
            .field("fri", &self.fri)
            .field("index_cache", &self.index_cache)
            .finish()
    }
}

impl DeepSizeOf for BloomFilterIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.blocks.deep_size_of_children(context)
    }
}

impl BloomFilterIndex {
    async fn load(
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<FragReuseIndex>>,
        index_cache: LanceCache,
    ) -> Result<Arc<Self>> {
        let index_file = store.open_index_file(BLOOMFILTER_FILENAME).await?;
        let bloom_data = index_file
            .read_range(0..index_file.num_rows(), None)
            .await?;
        let file_schema = index_file.schema();

        let number_of_items: u64 = file_schema
            .metadata
            .get(BLOOMFILTER_ITEM_META_KEY)
            .and_then(|bs| bs.parse().ok())
            .unwrap_or(*DEFAULT_NUMBER_OF_ITEMS);

        let number_of_bytes: usize = file_schema
            .metadata
            .get(BLOOMFILTER_BITS_META_KEY)
            .and_then(|bs| bs.parse().ok())
            .unwrap_or(*DEFAULT_NUMBER_OF_BYTES);

        Ok(Arc::new(Self::try_from_serialized(
            bloom_data,
            store,
            fri,
            index_cache,
            number_of_items,
            number_of_bytes,
        )?))
    }

    fn try_from_serialized(
        data: RecordBatch,
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<FragReuseIndex>>,
        index_cache: LanceCache,
        number_of_items: u64,
        number_of_bytes: usize,
    ) -> Result<Self> {
        if data.num_rows() == 0 {
            // Return empty index for empty data
            return Ok(Self {
                blocks: Vec::new(),
                number_of_items,
                number_of_bytes,
                store,
                fri,
                index_cache,
            });
        }

        let fragment_id_col = data
            .column_by_name("fragment_id")
            .ok_or_else(|| {
                Error::invalid_input(
                    "BloomFilterIndex: missing 'fragment_id' column",
                    location!(),
                )
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::invalid_input(
                    "BloomFilterIndex: 'fragment_id' column is not UInt64",
                    location!(),
                )
            })?;

        let block_start_col = data
            .column_by_name("block_start")
            .ok_or_else(|| {
                Error::invalid_input(
                    "BloomFilterIndex: missing 'block_start' column",
                    location!(),
                )
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::invalid_input(
                    "BloomFilterIndex: 'block_start' column is not UInt64",
                    location!(),
                )
            })?;

        let block_size_col = data
            .column_by_name("block_size")
            .ok_or_else(|| {
                Error::invalid_input("BloomFilterIndex: missing 'block_size' column", location!())
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::invalid_input(
                    "BloomFilterIndex: 'block_size' column is not UInt64",
                    location!(),
                )
            })?;

        let bloom_filter_data_col = data
            .column_by_name("bloom_filter_data")
            .ok_or_else(|| {
                Error::invalid_input(
                    "BloomFilterIndex: missing 'bloom_filter_data' column",
                    location!(),
                )
            })?
            .as_any()
            .downcast_ref::<arrow_array::BinaryArray>()
            .ok_or_else(|| {
                Error::invalid_input(
                    "BloomFilterIndex: 'bloom_filter_data' column is not Binary",
                    location!(),
                )
            })?;

        let num_blocks = data.num_rows();
        let mut blocks = Vec::with_capacity(num_blocks);

        for i in 0..num_blocks {
            let bloom_filter_data = if bloom_filter_data_col.is_valid(i) {
                bloom_filter_data_col.value(i).to_vec()
            } else {
                Vec::new()
            };

            blocks.push(BloomFilterStatistics {
                fragment_id: fragment_id_col.value(i),
                block_start: block_start_col.value(i),
                block_size: block_size_col.value(i) as usize,
                bloom_filter_data,
            });
        }

        Ok(Self {
            blocks,
            number_of_items,
            number_of_bytes,
            store,
            fri,
            index_cache,
        })
    }

    fn evaluate_block_against_query(
        &self,
        block: &BloomFilterStatistics,
        query: &SargableQuery,
    ) -> Result<bool> {
        // Try to reconstruct the bloom filter from the stored data
        let sbbf = match Sbbf::new(&block.bloom_filter_data) {
            Ok(sbbf) => sbbf,
            Err(_) => {
                // If we can't reconstruct the bloom filter, be conservative
                return Ok(true);
            }
        };

        match query {
            SargableQuery::IsNull() => {
                // For null values, we could use a special marker in the bloom filter
                // For now, conservatively assume blocks might contain nulls
                Ok(true)
            }
            SargableQuery::Equals(target) => {
                if target.is_null() {
                    // Handle null values - conservatively return true for now
                    return Ok(true);
                }

                // Check the bloom filter for the target value
                match target {
                    datafusion_common::ScalarValue::Int32(Some(val)) => Ok(sbbf.check(val)),
                    datafusion_common::ScalarValue::Int64(Some(val)) => Ok(sbbf.check(val)),
                    datafusion_common::ScalarValue::Float32(Some(val)) => Ok(sbbf.check(val)),
                    datafusion_common::ScalarValue::Float64(Some(val)) => Ok(sbbf.check(val)),
                    datafusion_common::ScalarValue::Utf8(Some(val)) => Ok(sbbf.check(val.as_str())),
                    datafusion_common::ScalarValue::LargeUtf8(Some(val)) => {
                        Ok(sbbf.check(val.as_str()))
                    }
                    datafusion_common::ScalarValue::Binary(Some(val)) => {
                        Ok(sbbf.check(val.as_slice()))
                    }
                    datafusion_common::ScalarValue::LargeBinary(Some(val)) => {
                        Ok(sbbf.check(val.as_slice()))
                    }
                    _ => {
                        // For unsupported types, be conservative
                        Ok(true)
                    }
                }
            }
            SargableQuery::Range(_, _) => {
                // Bloom filters are not great for range queries
                // but we can conservatively return true
                Ok(true)
            }
            SargableQuery::IsIn(values) => {
                // Check if any value in the set is in the bloom filter
                for value in values {
                    if value.is_null() {
                        // Handle null values - conservatively assume they might exist
                        return Ok(true);
                    }

                    let found = match value {
                        datafusion_common::ScalarValue::Int32(Some(val)) => sbbf.check(val),
                        datafusion_common::ScalarValue::Int64(Some(val)) => sbbf.check(val),
                        datafusion_common::ScalarValue::Float32(Some(val)) => sbbf.check(val),
                        datafusion_common::ScalarValue::Float64(Some(val)) => sbbf.check(val),
                        datafusion_common::ScalarValue::Utf8(Some(val)) => sbbf.check(val.as_str()),
                        datafusion_common::ScalarValue::LargeUtf8(Some(val)) => {
                            sbbf.check(val.as_str())
                        }
                        datafusion_common::ScalarValue::Binary(Some(val)) => {
                            sbbf.check(val.as_slice())
                        }
                        datafusion_common::ScalarValue::LargeBinary(Some(val)) => {
                            sbbf.check(val.as_slice())
                        }
                        _ => true, // Conservative for unsupported types
                    };

                    if found {
                        return Ok(true);
                    }
                }
                Ok(false) // None of the values were found
            }
            SargableQuery::FullTextSearch(_) => Err(Error::NotSupported {
                source: "full text search is not supported for bloom filter indexes".into(),
                location: location!(),
            }),
        }
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
        Ok(())
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "number_of_items": self.number_of_items,
            "number_of_bytes": self.number_of_bytes,
        }))
    }

    fn index_type(&self) -> IndexType {
        IndexType::BloomFilter
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let mut frag_ids = RoaringBitmap::new();

        // Loop through blocks and add unique fragment IDs to the bitmap
        for block in &self.blocks {
            frag_ids.insert(block.fragment_id as u32);
        }

        Ok(frag_ids)
    }
}

#[async_trait]
impl ScalarIndex for BloomFilterIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        metrics.record_comparisons(self.blocks.len());
        let query = query.as_any().downcast_ref::<SargableQuery>().unwrap();

        let mut row_id_tree_map = RowIdTreeMap::new();

        // For each block, check if it might contain the queried value
        for block in self.blocks.iter() {
            // In a real bloom filter implementation, this would check the actual bloom filter
            // For now, we conservatively assume all blocks might contain the value
            // This is safe but not optimal - it means we never have false negatives
            if self.evaluate_block_against_query(block, query)? {
                // Calculate the range of row addresses for this block
                // Row addresses are: (fragment_id << 32) + block_start
                let block_start_addr = (block.fragment_id << 32) + block.block_start;
                let block_end_addr = block_start_addr + (block.block_size as u64);

                // Add all row addresses in this block to the result
                row_id_tree_map.insert_range(block_start_addr..block_end_addr);
            }
        }

        Ok(SearchResult::AtMost(row_id_tree_map))
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
        // 1. Prepare the builder for new bloom filters
        let batches_source = new_data;

        let mut builder = BloomFilterIndexBuilder::try_new(BloomFilterIndexBuilderParams {
            number_of_items: self.number_of_items,
            number_of_bytes: self.number_of_bytes,
        })?;

        builder.train(batches_source).await?;

        // Get the new blocks from the builder
        let new_blocks = builder.blocks;

        // Combine existing blocks with new blocks
        let mut all_blocks = self.blocks.clone();
        all_blocks.extend(new_blocks);

        // Create a new builder with all blocks to write them out
        let mut combined_builder =
            BloomFilterIndexBuilder::try_new(BloomFilterIndexBuilderParams {
                number_of_items: self.number_of_items,
                number_of_bytes: self.number_of_bytes,
            })?;
        combined_builder.blocks = all_blocks;

        // Write the updated index to dest_store
        combined_builder.write_index(dest_store).await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pb::BloomFilterIndexDetails::default())
                .unwrap(),
            index_version: BLOOMFILTER_INDEX_VERSION,
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(
            TrainingCriteria::new(TrainingOrdering::Addresses).with_row_addr(),
        )
    }
}

fn default_number_of_items() -> u64 {
    *DEFAULT_NUMBER_OF_ITEMS
}

fn default_number_of_bytes() -> usize {
    *DEFAULT_NUMBER_OF_BYTES
}

// NumberOfItems: 8192 + NumberOfBytes: 16384(16KiB) + 8 SALT values -> Probability of false positive: 0.00057(1 in 1754)
// reference: https://hur.st/bloomfilter/?n=8192&p=&m=16KiB&k=8
static DEFAULT_NUMBER_OF_ITEMS: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("LANCE_BLOOMFILTER_DEFAULT_NUMBER_OF_ITEMS")
        .unwrap_or_else(|_| "8192".to_string())
        .parse()
        .expect("failed to parse Lance_BLOOMFILTER_DEFAULT_NUMBER_OF_ITEMS")
});

static DEFAULT_NUMBER_OF_BYTES: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("LANCE_BLOOMFILTER_DEFAULT_NUMBER_OF_BYTES")
        // 16384 = 16KiB
        .unwrap_or_else(|_| "16384".to_string())
        .parse()
        .expect("failed to parse Lance_BLOOMFILTER_DEFAULT_NUMBER_OF_BYTES")
});

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilterIndexBuilderParams {
    #[serde(default = "default_number_of_items")]
    number_of_items: u64,
    #[serde(default = "default_number_of_bytes")]
    number_of_bytes: usize,
}

impl Default for BloomFilterIndexBuilderParams {
    fn default() -> Self {
        Self {
            number_of_items: *DEFAULT_NUMBER_OF_ITEMS,
            number_of_bytes: *DEFAULT_NUMBER_OF_BYTES,
        }
    }
}

impl BloomFilterIndexBuilderParams {
    fn new(number_of_items: u64, number_of_bytes: usize) -> Self {
        Self {
            number_of_items,
            number_of_bytes,
        }
    }
}

pub struct BloomFilterIndexBuilder {
    params: BloomFilterIndexBuilderParams,
    blocks: Vec<BloomFilterStatistics>,
    // The local offset within the current blocks
    cur_block_offset: usize,
    cur_fragment_id: u64,
    sbbf: Option<Sbbf>,
}

impl BloomFilterIndexBuilder {
    pub fn try_new(params: BloomFilterIndexBuilderParams) -> Result<Self> {
        let sbbf = SbbfBuilder::new()
            .expected_items(params.number_of_items)
            .num_bytes(params.number_of_bytes)
            .build()
            .map_err(|e| Error::InvalidInput {
                source: format!("Failed to build SBBF: {:?}", e).into(),
                location: location!(),
            })?;

        Ok(Self {
            params,
            blocks: Vec::new(),
            cur_block_offset: 0,
            cur_fragment_id: 0,
            sbbf: Some(sbbf),
        })
    }

    fn process_primitive_array<T>(sbbf: &mut Sbbf, array: &arrow_array::PrimitiveArray<T>)
    where
        T: arrow_array::ArrowPrimitiveType,
        T::Native: parquet::data_type::AsBytes,
    {
        for i in 0..array.len() {
            if array.is_valid(i) {
                sbbf.insert(&array.value(i));
            }
        }
    }

    fn process_string_array(sbbf: &mut Sbbf, array: &arrow_array::StringArray) {
        for i in 0..array.len() {
            if array.is_valid(i) {
                sbbf.insert(array.value(i));
            }
        }
    }

    fn process_large_string_array(sbbf: &mut Sbbf, array: &arrow_array::LargeStringArray) {
        for i in 0..array.len() {
            if array.is_valid(i) {
                sbbf.insert(array.value(i));
            }
        }
    }

    fn process_binary_array(sbbf: &mut Sbbf, array: &arrow_array::BinaryArray) {
        for i in 0..array.len() {
            if array.is_valid(i) {
                sbbf.insert(array.value(i));
            }
        }
    }

    fn process_large_binary_array(sbbf: &mut Sbbf, array: &arrow_array::LargeBinaryArray) {
        for i in 0..array.len() {
            if array.is_valid(i) {
                sbbf.insert(array.value(i));
            }
        }
    }

    fn update_stats(&mut self, array: &ArrayRef) -> Result<()> {
        if let Some(ref mut sbbf) = self.sbbf {
            // TODO: is there a more efficient way to do this?
            match array.data_type() {
                DataType::Int32 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::Int32Array>()
                        .unwrap();
                    Self::process_primitive_array(sbbf, typed_array);
                }
                DataType::Int64 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::Int64Array>()
                        .unwrap();
                    Self::process_primitive_array(sbbf, typed_array);
                }
                DataType::Float32 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::Float32Array>()
                        .unwrap();
                    Self::process_primitive_array(sbbf, typed_array);
                }
                DataType::Float64 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::Float64Array>()
                        .unwrap();
                    Self::process_primitive_array(sbbf, typed_array);
                }
                DataType::Utf8 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::StringArray>()
                        .unwrap();
                    Self::process_string_array(sbbf, typed_array);
                }
                DataType::LargeUtf8 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::LargeStringArray>()
                        .unwrap();
                    Self::process_large_string_array(sbbf, typed_array);
                }
                DataType::Binary => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::BinaryArray>()
                        .unwrap();
                    Self::process_binary_array(sbbf, typed_array);
                }
                DataType::LargeBinary => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::LargeBinaryArray>()
                        .unwrap();
                    Self::process_large_binary_array(sbbf, typed_array);
                }
                _ => {
                    return Err(Error::InvalidInput {
                        source: format!(
                            "Bloom filter does not support data type: {:?}",
                            array.data_type()
                        )
                        .into(),
                        location: location!(),
                    });
                }
            }
        }

        Ok(())
    }

    fn new_block(&mut self, fragment_id: u64) -> Result<()> {
        // Calculate block_start based on existing blocks in the same fragment
        let block_start = self
            .blocks
            .iter()
            .filter(|block| block.fragment_id == fragment_id)
            .map(|block| block.block_size as u64)
            .sum::<u64>();

        // Store the current bloom filter data
        let bloom_filter_data = if let Some(ref sbbf) = self.sbbf {
            sbbf.to_bytes()
        } else {
            Vec::new()
        };

        let new_block = BloomFilterStatistics {
            fragment_id,
            block_start,
            block_size: self.cur_block_offset,
            bloom_filter_data,
        };

        self.blocks.push(new_block);
        self.cur_block_offset = 0;

        // Reset sbbf for the next block
        self.sbbf = Some(
            SbbfBuilder::new()
                .expected_items(self.params.number_of_items)
                .num_bytes(self.params.number_of_bytes)
                .build()
                .map_err(|e| Error::InvalidInput {
                    source: format!("Failed to build SBBF: {:?}", e).into(),
                    location: location!(),
                })?,
        );

        Ok(())
    }

    pub async fn train(&mut self, batches_source: SendableRecordBatchStream) -> Result<()> {
        assert!(batches_source.schema().field_with_name(ROW_ADDR).is_ok());

        let mut batches_source =
            chunk_concat_stream(batches_source, self.params.number_of_items as usize);

        while let Some(batch) = batches_source.try_next().await? {
            if batch.num_rows() == 0 {
                continue;
            }

            let data_array: &arrow_array::ArrayRef = batch.column(0);
            let row_addrs_array = batch
                .column_by_name(ROW_ADDR)
                .unwrap()
                .as_any()
                .downcast_ref::<arrow_array::UInt64Array>()
                .unwrap();

            let mut remaining = batch.num_rows();
            let mut array_offset: usize = 0;

            // Initialize cur_fragment_id from the first row address if this is the first batch
            if self.blocks.is_empty() && self.cur_block_offset == 0 {
                let first_row_addr = row_addrs_array.value(0);
                self.cur_fragment_id = first_row_addr >> 32;
            }

            while remaining > 0 {
                // Find the next fragment boundary in this batch
                let next_fragment_index = (array_offset..row_addrs_array.len()).find(|&i| {
                    let row_addr = row_addrs_array.value(i);
                    let fragment_id = row_addr >> 32;
                    fragment_id == self.cur_fragment_id + 1
                });
                let empty_rows_left_in_cur_zone: usize =
                    (self.params.number_of_items - self.cur_block_offset as u64) as usize;

                // Check if there is enough data from the current fragment to fill the current zone
                let desired = if let Some(idx) = next_fragment_index {
                    self.cur_fragment_id = row_addrs_array.value(idx) >> 32;
                    // Take the minimum between distance to boundary and space left in zone
                    // to ensure we don't exceed the zone size limit
                    std::cmp::min(idx - array_offset, empty_rows_left_in_cur_zone)
                } else {
                    empty_rows_left_in_cur_zone
                };

                if desired > remaining {
                    // Not enough data to fill a map, just increment counts
                    self.update_stats(&data_array.slice(array_offset, remaining))?;
                    self.cur_block_offset += remaining;
                    break;
                } else if desired > 0 {
                    // There is enough data, create a new block
                    self.update_stats(&data_array.slice(array_offset, desired))?;
                    self.cur_block_offset += desired;
                    self.new_block(row_addrs_array.value(array_offset) >> 32)?;
                } else if desired == 0 {
                    // The new batch starts with a new fragment. Flush the current zone if it's not empty
                    if self.cur_block_offset > 0 {
                        self.new_block(self.cur_fragment_id - 1)?;
                    }
                    // Let the loop run again
                    // to find the next fragment boundary
                    continue;
                }
                array_offset += desired;
                remaining = remaining.saturating_sub(desired);
            }
        }
        // Create the final block
        if self.cur_block_offset > 0 {
            self.new_block(self.cur_fragment_id)?;
        }

        Ok(())
    }

    fn bloomfilter_stats_as_batch(&self) -> Result<RecordBatch> {
        let fragment_ids =
            UInt64Array::from_iter_values(self.blocks.iter().map(|block| block.fragment_id));

        let block_starts =
            UInt64Array::from_iter_values(self.blocks.iter().map(|block| block.block_start));

        let block_sizes =
            UInt64Array::from_iter_values(self.blocks.iter().map(|block| block.block_size as u64));

        // Store bloom filter data as binary array
        let bloom_filter_data = if self.blocks.is_empty() {
            Arc::new(arrow_array::BinaryArray::new_null(0)) as ArrayRef
        } else {
            let binary_data: Vec<Option<&[u8]>> = self
                .blocks
                .iter()
                .map(|block| Some(block.bloom_filter_data.as_slice()))
                .collect();
            Arc::new(arrow_array::BinaryArray::from_opt_vec(binary_data)) as ArrayRef
        };

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new("fragment_id", DataType::UInt64, false),
            Field::new("block_start", DataType::UInt64, false),
            Field::new("block_size", DataType::UInt64, false),
            Field::new("bloom_filter_data", DataType::Binary, false),
        ]));

        let columns: Vec<ArrayRef> = vec![
            Arc::new(fragment_ids) as ArrayRef,
            Arc::new(block_starts) as ArrayRef,
            Arc::new(block_sizes) as ArrayRef,
            bloom_filter_data,
        ];

        Ok(RecordBatch::try_new(schema, columns)?)
    }

    pub async fn write_index(self, index_store: &dyn IndexStore) -> Result<()> {
        let record_batch = self.bloomfilter_stats_as_batch()?;

        let mut file_schema = record_batch.schema().as_ref().clone();
        file_schema.metadata.insert(
            BLOOMFILTER_ITEM_META_KEY.to_string(),
            self.params.number_of_items.to_string(),
        );

        file_schema.metadata.insert(
            BLOOMFILTER_BITS_META_KEY.to_string(),
            self.params.number_of_bytes.to_string(),
        );

        let mut index_file = index_store
            .new_index_file(BLOOMFILTER_FILENAME, Arc::new(file_schema))
            .await?;
        index_file.write_record_batch(record_batch).await?;
        index_file.finish().await?;
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct BloomFilterIndexPlugin;

impl BloomFilterIndexPlugin {
    async fn train_bloomfilter_index(
        batches_source: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        options: Option<BloomFilterIndexBuilderParams>,
    ) -> Result<()> {
        let mut builder = BloomFilterIndexBuilder::try_new(options.unwrap_or_default())?;

        builder.train(batches_source).await?;

        builder.write_index(index_store).await?;
        Ok(())
    }
}

#[async_trait]
impl ScalarIndexPlugin for BloomFilterIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        if field.data_type().is_nested() {
            return Err(Error::InvalidInput {
                source: "A bloom filter index can only be created on a non-nested field.".into(),
                location: location!(),
            });
        }

        let params = serde_json::from_str::<BloomFilterIndexBuilderParams>(params)?;

        Ok(Box::new(BloomFilterIndexTrainingRequest::new(params)))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
    ) -> Result<CreatedIndex> {
        let request = (request as Box<dyn std::any::Any>)
            .downcast::<BloomFilterIndexTrainingRequest>()
            .map_err(|_| Error::InvalidInput {
                source: "must provide training request created by new_training_request".into(),
                location: location!(),
            })?;
        Self::train_bloomfilter_index(data, index_store, Some(request.params)).await?;
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pb::BloomFilterIndexDetails::default())
                .unwrap(),
            index_version: BLOOMFILTER_INDEX_VERSION,
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
        // TODO: Should it be sargable query?
        Some(Box::new(SargableQueryParser::new(index_name, true)))
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(
            BloomFilterIndex::load(index_store, frag_reuse_index, cache).await?
                as Arc<dyn ScalarIndex>,
        )
    }
}

#[derive(Debug)]
pub struct BloomFilterIndexTrainingRequest {
    pub params: BloomFilterIndexBuilderParams,
    pub criteria: TrainingCriteria,
}

impl BloomFilterIndexTrainingRequest {
    pub fn new(params: BloomFilterIndexBuilderParams) -> Self {
        Self {
            params,
            criteria: TrainingCriteria::new(TrainingOrdering::Addresses).with_row_addr(),
        }
    }
}

impl TrainingRequest for BloomFilterIndexTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

#[cfg(test)]
mod tests {
    use crate::scalar::registry::VALUE_COLUMN_NAME;
    use std::sync::Arc;

    use crate::scalar::bloomfilter::{BloomFilterIndexPlugin, BloomFilterStatistics};
    use arrow::datatypes::Float32Type;
    use arrow_array::{Array, RecordBatch, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::execution::SendableRecordBatchStream;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion_common::ScalarValue;
    use futures::{stream, StreamExt, TryStreamExt};
    use lance_core::{cache::LanceCache, utils::mask::RowIdTreeMap, ROW_ADDR};
    use lance_datafusion::datagen::DatafusionDatagenExt;
    use lance_datagen::ArrayGeneratorExt;
    use lance_datagen::{array, BatchCount, RowCount};
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;
    use tempfile::tempdir;

    use crate::scalar::{
        bloomfilter::{
            BloomFilterIndex, BloomFilterIndexBuilderParams, BLOOMFILTER_BITS_META_KEY,
            BLOOMFILTER_FILENAME, BLOOMFILTER_ITEM_META_KEY,
        },
        lance_format::LanceIndexStore,
        SargableQuery, ScalarIndex, SearchResult,
    };

    use crate::metrics::NoOpMetricsCollector;
    use crate::Index; // Import Index trait to access calculate_included_frags
    use roaring::RoaringBitmap; // Import RoaringBitmap for the test
    use std::collections::Bound;

    // Adds a _rowaddr column emulating each batch as a new fragment
    fn add_row_addr(stream: SendableRecordBatchStream) -> SendableRecordBatchStream {
        let schema = stream.schema();
        let schema_with_row_addr = Arc::new(Schema::new(vec![
            schema.field(0).clone(),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let schema = schema_with_row_addr.clone();
        let stream = stream.enumerate().map(move |(frag_id, batch)| {
            let batch = batch.unwrap();
            let row_addr = Arc::new(UInt64Array::from_iter_values(
                (0..batch.num_rows() as u64).map(|off| off + ((frag_id as u64) << 32)),
            ));
            Ok(RecordBatch::try_new(
                schema_with_row_addr.clone(),
                vec![batch.column(0).clone(), row_addr],
            )?)
        });
        Box::pin(RecordBatchStreamAdapter::new(schema, stream))
    }

    #[tokio::test]
    async fn test_empty_bloomfilter_index() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        let data = arrow_array::Int32Array::from(Vec::<i32>::new());
        let row_ids = arrow_array::UInt64Array::from(Vec::<u64>::new());
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Int32, false),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();

        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        BloomFilterIndexPlugin::train_bloomfilter_index(data_stream, test_store.as_ref(), None)
            .await
            .unwrap();

        log::debug!("Successfully wrote the index file");

        // Read the index file back and check its contents
        let index = BloomFilterIndex::load(test_store.clone(), None, LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");
        assert_eq!(index.blocks.len(), 0);
        assert_eq!(index.number_of_items, 8192);
        assert_eq!(index.number_of_bytes, 16384); // 16kib

        // Equals query: null (should match nothing, as there are no nulls in empty index)
        let query = SargableQuery::Equals(ScalarValue::Int32(None));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::AtMost(RowIdTreeMap::new()));
    }

    #[tokio::test]
    async fn test_basic_bloomfilter_index() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        let data = arrow_array::Int32Array::from_iter_values(0..100);
        let row_ids = UInt64Array::from_iter_values((0..data.len()).map(|i| i as u64));
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Int32, false),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(100, 1024)),
        )
        .await
        .unwrap();

        log::debug!("Successfully wrote the index file");

        // Read the index file back and check its contents
        let index = BloomFilterIndex::load(test_store.clone(), None, LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");

        assert_eq!(index.blocks.len(), 1);
        assert_eq!(index.number_of_items, 100);
        assert_eq!(index.number_of_bytes, 1024);

        // Check that we have one block (since 100 items fit exactly in one block of size 100)
        assert_eq!(index.blocks[0].fragment_id, 0);
        assert_eq!(index.blocks[0].block_start, 0);
        assert_eq!(index.blocks[0].block_size, 100);

        // Test search functionality
        // The bloom filter should work correctly and find the value
        let query = SargableQuery::Equals(ScalarValue::Int32(Some(50)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match the block since value 50 is in the range [0, 100)
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..100);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value that shouldn't exist
        let query = SargableQuery::Equals(ScalarValue::Int32(Some(500))); // Value not in [0, 100)
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should return empty result since bloom filter correctly filters out this value
        assert_eq!(result, SearchResult::AtMost(RowIdTreeMap::new()));

        // Test calculate_included_frags
        assert_eq!(
            index.calculate_included_frags().await.unwrap(),
            RoaringBitmap::from_iter(0..1)
        );
    }

    #[tokio::test]
    async fn test_multiple_fragments_bloomfilter() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Int64, false),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));

        // Create multiple fragments with data
        // Fragment 0: values 0-99
        let fragment0_data = arrow_array::Int64Array::from_iter_values(0..100);
        let fragment0_row_ids = UInt64Array::from_iter_values(0..100);
        let fragment0_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(fragment0_data), Arc::new(fragment0_row_ids)],
        )
        .unwrap();

        // Fragment 1: values 100-199
        let fragment1_data = arrow_array::Int64Array::from_iter_values(100..200);
        let fragment1_row_ids = UInt64Array::from_iter_values((0..100).map(|i| i + (1 << 32)));
        let fragment1_batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(fragment1_data), Arc::new(fragment1_row_ids)],
        )
        .unwrap();

        // Create a stream with multiple batches (fragments)
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::iter(vec![
                Ok(fragment0_batch.clone()),
                Ok(fragment1_batch.clone()),
            ]),
        ));

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(50, 1024)),
        )
        .await
        .unwrap();

        // Read the index file back and check its contents
        let index = BloomFilterIndex::load(test_store.clone(), None, LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");

        // Should have 4 blocks total (2 blocks per fragment)
        assert_eq!(index.blocks.len(), 4);

        // Check fragment 0 blocks
        assert_eq!(index.blocks[0].fragment_id, 0);
        assert_eq!(index.blocks[0].block_start, 0);
        assert_eq!(index.blocks[0].block_size, 50);

        assert_eq!(index.blocks[1].fragment_id, 0);
        assert_eq!(index.blocks[1].block_start, 50);
        assert_eq!(index.blocks[1].block_size, 50);

        // Check fragment 1 blocks
        assert_eq!(index.blocks[2].fragment_id, 1);
        assert_eq!(index.blocks[2].block_start, 0);
        assert_eq!(index.blocks[2].block_size, 50);

        assert_eq!(index.blocks[3].fragment_id, 1);
        assert_eq!(index.blocks[3].block_start, 50);
        assert_eq!(index.blocks[3].block_size, 50);

        // Test search functionality
        let query = SargableQuery::Equals(ScalarValue::Int64(Some(150)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should only match fragment 1 blocks since bloom filter correctly filters
        // Value 150 is only in fragment 1 (values 100-199), not in fragment 0 (values 0-99)
        let mut expected = RowIdTreeMap::new();
        expected.insert_range((1u64 << 32) + 50..((1u64 << 32) + 100)); // Only the block containing 150
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test calculate_included_frags
        assert_eq!(
            index.calculate_included_frags().await.unwrap(),
            RoaringBitmap::from_iter(0..2)
        );
    }

    #[tokio::test]
    async fn test_nan_bloomfilter_index() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create deterministic data with NaN values
        // Pattern: [1.0, 2.0, NaN, 3.0, 4.0, 5.0, NaN, 6.0, 7.0, 8.0, ...]
        let mut values = Vec::new();
        for i in 0..500 {
            if i % 5 == 2 {
                values.push(f32::NAN);
            } else {
                values.push(i as f32);
            }
        }

        let float_data = arrow_array::Float32Array::from(values);
        let row_ids = UInt64Array::from_iter_values((0..float_data.len()).map(|i| i as u64));
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Float32, true),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(float_data.clone()), Arc::new(row_ids)],
        )
        .unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(100, 1024)),
        )
        .await
        .unwrap();

        // Load the index
        let index = BloomFilterIndex::load(test_store.clone(), None, LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");

        // Should have 5 blocks since we have 500 rows and block size is 100
        assert_eq!(index.blocks.len(), 5);

        // Test search for NaN values using Equals with NaN
        let query = SargableQuery::Equals(ScalarValue::Float32(Some(f32::NAN)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match all blocks since they all contain NaN values
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..500); // All rows since NaN is in every block
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a specific finite value that exists in the data
        let query = SargableQuery::Equals(ScalarValue::Float32(Some(5.0)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match only the first block since 5.0 only exists in rows 0-99
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..100);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value that doesn't exist but is within expected range
        let query = SargableQuery::Equals(ScalarValue::Float32(Some(250.0)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match the third block since 250.0 would be in that range if it existed
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(200..300);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value way outside the range
        let query = SargableQuery::Equals(ScalarValue::Float32(Some(10000.0)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should return empty since bloom filter correctly filters out this value
        assert_eq!(result, SearchResult::AtMost(RowIdTreeMap::new()));

        // Test IsIn query with NaN and finite values
        let query = SargableQuery::IsIn(vec![
            ScalarValue::Float32(Some(f32::NAN)),
            ScalarValue::Float32(Some(5.0)),
            ScalarValue::Float32(Some(150.0)), // This value exists in the second block
        ]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match all blocks since they all contain NaN values
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..500);
        assert_eq!(result, SearchResult::AtMost(expected));
    }

    #[tokio::test]
    async fn test_complex_bloomfilter_index() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create data that will produce multiple blocks
        let data_size = 10000;
        let data = arrow_array::Int64Array::from_iter_values(0..data_size as i64);
        let row_ids = UInt64Array::from_iter_values((0..data.len()).map(|i| i as u64));
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Int64, false),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(1000, 2048)), // 10 blocks total
        )
        .await
        .unwrap();

        // Load the index
        let index = BloomFilterIndex::load(test_store.clone(), None, LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");

        // Should have 10 blocks since we have 10000 rows and block size is 1000
        assert_eq!(index.blocks.len(), 10);
        assert_eq!(index.number_of_items, 1000);
        assert_eq!(index.number_of_bytes, 2048);

        // Verify block structure
        for (i, block) in index.blocks.iter().enumerate() {
            assert_eq!(block.fragment_id, 0);
            assert_eq!(block.block_start, (i * 1000) as u64);
            assert_eq!(block.block_size, 1000);
            assert!(!block.bloom_filter_data.is_empty());
        }

        // Test search for a value in a specific block
        let query = SargableQuery::Equals(ScalarValue::Int64(Some(2500))); // In block 2 (2000-2999)
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match block 2
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(2000..3000);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value way outside the range
        let query = SargableQuery::Equals(ScalarValue::Int64(Some(50000)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should return empty since bloom filter correctly filters out this value
        assert_eq!(result, SearchResult::AtMost(RowIdTreeMap::new()));

        // Test IsIn query with values from different blocks
        let query = SargableQuery::IsIn(vec![
            ScalarValue::Int64(Some(500)),   // Block 0 (0-999)
            ScalarValue::Int64(Some(2500)),  // Block 2 (2000-2999)
            ScalarValue::Int64(Some(7500)),  // Block 7 (7000-7999)
            ScalarValue::Int64(Some(50000)), // Not in any block
        ]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match blocks 0, 2, and 7
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..1000); // Block 0
        expected.insert_range(2000..3000); // Block 2
        expected.insert_range(7000..8000); // Block 7
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test calculate_included_frags
        assert_eq!(
            index.calculate_included_frags().await.unwrap(),
            RoaringBitmap::from_iter(0..1)
        );
    }

    #[tokio::test]
    async fn test_string_bloomfilter_index() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create string data
        let string_values: Vec<String> = (0..200).map(|i| format!("value_{:03}", i)).collect();
        let string_data = arrow_array::StringArray::from_iter_values(string_values.iter());
        let row_ids = UInt64Array::from_iter_values((0..string_data.len()).map(|i| i as u64));
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(string_data), Arc::new(row_ids)],
        )
        .unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(100, 1024)),
        )
        .await
        .unwrap();

        // Load the index
        let index = BloomFilterIndex::load(test_store.clone(), None, LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");

        // Should have 2 blocks since we have 200 rows and block size is 100
        assert_eq!(index.blocks.len(), 2);

        // Test search for a value in the first block
        let query = SargableQuery::Equals(ScalarValue::Utf8(Some("value_050".to_string())));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match the first block
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..100);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value in the second block
        let query = SargableQuery::Equals(ScalarValue::Utf8(Some("value_150".to_string())));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match the second block
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(100..200);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value that doesn't exist
        let query = SargableQuery::Equals(ScalarValue::Utf8(Some("nonexistent_value".to_string())));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should return empty since bloom filter correctly filters out this value
        assert_eq!(result, SearchResult::AtMost(RowIdTreeMap::new()));

        // Test IsIn query with string values
        let query = SargableQuery::IsIn(vec![
            ScalarValue::Utf8(Some("value_025".to_string())), // First block
            ScalarValue::Utf8(Some("value_175".to_string())), // Second block
            ScalarValue::Utf8(Some("nonexistent".to_string())), // Not present
        ]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match both blocks
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..200);
        assert_eq!(result, SearchResult::AtMost(expected));
    }

    #[tokio::test]
    async fn test_binary_bloomfilter_index() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create binary data
        let binary_values: Vec<Vec<u8>> = (0..100)
            .map(|i| vec![i as u8, (i + 1) as u8, (i + 2) as u8])
            .collect();
        let binary_data = arrow_array::BinaryArray::from_iter_values(binary_values.iter());
        let row_ids = UInt64Array::from_iter_values((0..binary_data.len()).map(|i| i as u64));
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Binary, false),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(binary_data), Arc::new(row_ids)],
        )
        .unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(50, 1024)),
        )
        .await
        .unwrap();

        // Load the index
        let index = BloomFilterIndex::load(test_store.clone(), None, LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");

        // Should have 2 blocks since we have 100 rows and block size is 50
        assert_eq!(index.blocks.len(), 2);

        // Test search for a value in the first block
        let query = SargableQuery::Equals(ScalarValue::Binary(Some(vec![25, 26, 27])));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match the first block
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..50);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value in the second block
        let query = SargableQuery::Equals(ScalarValue::Binary(Some(vec![75, 76, 77])));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match the second block
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(50..100);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value that doesn't exist
        let query = SargableQuery::Equals(ScalarValue::Binary(Some(vec![255, 254, 253])));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should return empty since bloom filter correctly filters out this value
        assert_eq!(result, SearchResult::AtMost(RowIdTreeMap::new()));
    }

    #[tokio::test]
    async fn test_large_data_types_bloomfilter_index() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Test LargeUtf8 data type
        let large_string_values: Vec<String> =
            (0..100).map(|i| format!("large_value_{:05}", i)).collect();
        let large_string_data =
            arrow_array::LargeStringArray::from_iter_values(large_string_values.iter());
        let row_ids = UInt64Array::from_iter_values((0..large_string_data.len()).map(|i| i as u64));
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::LargeUtf8, false),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(large_string_data), Arc::new(row_ids)],
        )
        .unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(50, 1024)),
        )
        .await
        .unwrap();

        // Load the index
        let index = BloomFilterIndex::load(test_store.clone(), None, LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");

        assert_eq!(index.blocks.len(), 2);

        // Test search functionality
        let query = SargableQuery::Equals(ScalarValue::LargeUtf8(Some(
            "large_value_00025".to_string(),
        )));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match the first block
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..50);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value that doesn't exist
        let query = SargableQuery::Equals(ScalarValue::LargeUtf8(Some(
            "nonexistent_large_value".to_string(),
        )));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should return empty since bloom filter correctly filters out this value
        assert_eq!(result, SearchResult::AtMost(RowIdTreeMap::new()));
    }

    #[tokio::test]
    async fn test_bloomfilter_range_queries() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        let data = arrow_array::Int32Array::from_iter_values(0..1000);
        let row_ids = UInt64Array::from_iter_values((0..data.len()).map(|i| i as u64));
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Int32, false),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(data), Arc::new(row_ids)]).unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(250, 1024)), // 4 blocks total
        )
        .await
        .unwrap();

        // Load the index
        let index = BloomFilterIndex::load(test_store.clone(), None, LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");

        assert_eq!(index.blocks.len(), 4);

        // Test range query - bloom filters are conservative for ranges, should return all blocks
        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(100))),
            Bound::Included(ScalarValue::Int32(Some(300))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Bloom filters are conservative for range queries - should include all blocks
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..1000);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test range query with unbounded start
        let query = SargableQuery::Range(
            Bound::Unbounded,
            Bound::Included(ScalarValue::Int32(Some(300))),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should include all blocks (conservative)
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..1000);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test range query with unbounded end
        let query = SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(500))),
            Bound::Unbounded,
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should include all blocks (conservative)
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..1000);
        assert_eq!(result, SearchResult::AtMost(expected));
    }
}
