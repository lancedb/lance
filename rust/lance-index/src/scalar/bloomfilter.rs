// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bloom Filter Index
//!
//! Bloom Filter is a probabilistic data structure that allows for fast membership testing.
//! It is a space-efficient data structure that can be used to test whether an element is a member of a set.
//! It's an inexact filter - they may include false positives that require rechecking.

use crate::scalar::bloomfilter::sbbf::{Sbbf, SbbfBuilder};
use crate::scalar::expression::{BloomFilterQueryParser, ScalarQueryParser};
use crate::scalar::registry::{
    ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
};
use crate::scalar::{
    BloomFilterQuery, BuiltinIndexType, CreatedIndex, ScalarIndexParams, UpdateCriteria,
};
use crate::{pb, Any};
use arrow_array::{Array, UInt64Array};
use lance_core::utils::mask::RowIdTreeMap;
use lance_core::ROW_ADDR;
use lance_datafusion::chunker::chunk_concat_stream;
mod as_bytes;
mod sbbf;
use arrow_schema::{DataType, Field};
use futures::TryStreamExt;
use serde::{Deserialize, Serialize};

use std::sync::LazyLock;

use datafusion::execution::SendableRecordBatchStream;
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
const BLOOMFILTER_PROBABILITY_META_KEY: &str = "bloomfilter_probability";
const BLOOMFILTER_INDEX_VERSION: u32 = 0;

#[derive(Debug, Clone)]
struct BloomFilterStatistics {
    fragment_id: u64,
    // zone_start is the start row of the zone in the fragment, also known
    // as local row offset
    zone_start: u64,
    zone_length: usize,
    // Whether this zone contains any null values
    has_null: bool,
    // The actual bloom filter (SBBF) for efficient querying
    bloom_filter: Sbbf,
}

impl DeepSizeOf for BloomFilterStatistics {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        // Estimate the size of the bloom filter
        // We could try to get the actual size from the Sbbf if it has a method for that,
        // but for now we'll estimate based on the number of bytes it serializes to
        self.bloom_filter.to_bytes().len()
    }
}

#[derive(Debug, Clone)]
pub struct BloomFilterIndex {
    zones: Vec<BloomFilterStatistics>,
    // Number of items in the filter
    number_of_items: u64,
    // Probability of false positives, fraction between 0 and 1
    probability: f64,
}

impl DeepSizeOf for BloomFilterIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.zones.deep_size_of_children(context)
    }
}

impl BloomFilterIndex {
    async fn load(
        store: Arc<dyn IndexStore>,
        _fri: Option<Arc<FragReuseIndex>>,
        _index_cache: &LanceCache,
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

        let probability: f64 = file_schema
            .metadata
            .get(BLOOMFILTER_PROBABILITY_META_KEY)
            .and_then(|bs| bs.parse().ok())
            .unwrap_or(*DEFAULT_PROBABILITY);

        Ok(Arc::new(Self::try_from_serialized(
            bloom_data,
            number_of_items,
            probability,
        )?))
    }

    fn try_from_serialized(
        data: RecordBatch,
        number_of_items: u64,
        probability: f64,
    ) -> Result<Self> {
        if data.num_rows() == 0 {
            // Return empty index for empty data
            return Ok(Self {
                zones: Vec::new(),
                number_of_items,
                probability,
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

        let zone_start_col = data
            .column_by_name("zone_start")
            .ok_or_else(|| {
                Error::invalid_input("BloomFilterIndex: missing 'zone_start' column", location!())
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::invalid_input(
                    "BloomFilterIndex: 'zone_start' column is not UInt64",
                    location!(),
                )
            })?;

        let zone_length_col = data
            .column_by_name("zone_length")
            .ok_or_else(|| {
                Error::invalid_input(
                    "BloomFilterIndex: missing 'zone_length' column",
                    location!(),
                )
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::invalid_input(
                    "BloomFilterIndex: 'zone_length' column is not UInt64",
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

        let has_null_col = data
            .column_by_name("has_null")
            .ok_or_else(|| {
                Error::invalid_input("BloomFilterIndex: missing 'has_null' column", location!())
            })?
            .as_any()
            .downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| {
                Error::invalid_input(
                    "BloomFilterIndex: 'has_null' column is not Boolean",
                    location!(),
                )
            })?;

        let num_blocks = data.num_rows();
        let mut blocks = Vec::with_capacity(num_blocks);

        for i in 0..num_blocks {
            let bloom_filter_bytes = if bloom_filter_data_col.is_valid(i) {
                bloom_filter_data_col.value(i).to_vec()
            } else {
                Vec::new()
            };

            // Convert bytes back to Sbbf
            let bloom_filter = Sbbf::new(&bloom_filter_bytes).map_err(|e| {
                Error::invalid_input(
                    format!("Failed to deserialize bloom filter: {:?}", e),
                    location!(),
                )
            })?;

            blocks.push(BloomFilterStatistics {
                fragment_id: fragment_id_col.value(i),
                zone_start: zone_start_col.value(i),
                zone_length: zone_length_col.value(i) as usize,
                has_null: has_null_col.value(i),
                bloom_filter,
            });
        }

        Ok(Self {
            zones: blocks,
            number_of_items,
            probability,
        })
    }

    fn evaluate_block_against_query(
        &self,
        block: &BloomFilterStatistics,
        query: &BloomFilterQuery,
    ) -> Result<bool> {
        let sbbf = &block.bloom_filter;

        match query {
            BloomFilterQuery::IsNull() => {
                // Use the has_null information to determine if this block contains nulls
                Ok(block.has_null)
            }
            BloomFilterQuery::Equals(target) => {
                if target.is_null() {
                    // Handle null values using has_null information
                    return Ok(block.has_null);
                }

                // Check the bloom filter for the target value
                match target {
                    // Signed integers
                    datafusion_common::ScalarValue::Int8(Some(val)) => Ok(sbbf.check(val)),
                    datafusion_common::ScalarValue::Int16(Some(val)) => Ok(sbbf.check(val)),
                    datafusion_common::ScalarValue::Int32(Some(val)) => Ok(sbbf.check(val)),
                    datafusion_common::ScalarValue::Int64(Some(val)) => Ok(sbbf.check(val)),
                    // Unsigned integers
                    datafusion_common::ScalarValue::UInt8(Some(val)) => Ok(sbbf.check(val)),
                    datafusion_common::ScalarValue::UInt16(Some(val)) => Ok(sbbf.check(val)),
                    datafusion_common::ScalarValue::UInt32(Some(val)) => Ok(sbbf.check(val)),
                    datafusion_common::ScalarValue::UInt64(Some(val)) => Ok(sbbf.check(val)),
                    // Floating point
                    datafusion_common::ScalarValue::Float32(Some(val)) => Ok(sbbf.check(val)),
                    datafusion_common::ScalarValue::Float64(Some(val)) => Ok(sbbf.check(val)),
                    // String types
                    datafusion_common::ScalarValue::Utf8(Some(val)) => Ok(sbbf.check(val.as_str())),
                    datafusion_common::ScalarValue::LargeUtf8(Some(val)) => {
                        Ok(sbbf.check(val.as_str()))
                    }
                    // Binary types
                    datafusion_common::ScalarValue::Binary(Some(val)) => {
                        Ok(sbbf.check(val.as_slice()))
                    }
                    datafusion_common::ScalarValue::LargeBinary(Some(val)) => {
                        Ok(sbbf.check(val.as_slice()))
                    }
                    // Date and time types
                    datafusion_common::ScalarValue::Date32(Some(val)) => Ok(sbbf.check(val)),
                    datafusion_common::ScalarValue::Date64(Some(val)) => Ok(sbbf.check(val)),
                    datafusion_common::ScalarValue::Time32Second(Some(val)) => Ok(sbbf.check(val)),
                    datafusion_common::ScalarValue::Time32Millisecond(Some(val)) => {
                        Ok(sbbf.check(val))
                    }
                    datafusion_common::ScalarValue::Time64Microsecond(Some(val)) => {
                        Ok(sbbf.check(val))
                    }
                    datafusion_common::ScalarValue::Time64Nanosecond(Some(val)) => {
                        Ok(sbbf.check(val))
                    }
                    datafusion_common::ScalarValue::TimestampSecond(Some(val), _) => {
                        Ok(sbbf.check(val))
                    }
                    datafusion_common::ScalarValue::TimestampMillisecond(Some(val), _) => {
                        Ok(sbbf.check(val))
                    }
                    datafusion_common::ScalarValue::TimestampMicrosecond(Some(val), _) => {
                        Ok(sbbf.check(val))
                    }
                    datafusion_common::ScalarValue::TimestampNanosecond(Some(val), _) => {
                        Ok(sbbf.check(val))
                    }
                    _ => Err(Error::InvalidInput {
                        source: format!(
                            "Unsupported data type in bloom filter query: {:?}",
                            target
                        )
                        .into(),
                        location: location!(),
                    }),
                }
            }
            BloomFilterQuery::IsIn(values) => {
                // Check if any value in the set is in the bloom filter
                for value in values {
                    if value.is_null() {
                        // Handle null values using has_null information
                        if block.has_null {
                            return Ok(true);
                        }
                        continue;
                    }

                    let found = match value {
                        // Signed integers
                        datafusion_common::ScalarValue::Int8(Some(val)) => sbbf.check(val),
                        datafusion_common::ScalarValue::Int16(Some(val)) => sbbf.check(val),
                        datafusion_common::ScalarValue::Int32(Some(val)) => sbbf.check(val),
                        datafusion_common::ScalarValue::Int64(Some(val)) => sbbf.check(val),
                        // Unsigned integers
                        datafusion_common::ScalarValue::UInt8(Some(val)) => sbbf.check(val),
                        datafusion_common::ScalarValue::UInt16(Some(val)) => sbbf.check(val),
                        datafusion_common::ScalarValue::UInt32(Some(val)) => sbbf.check(val),
                        datafusion_common::ScalarValue::UInt64(Some(val)) => sbbf.check(val),
                        // Floating point
                        datafusion_common::ScalarValue::Float32(Some(val)) => sbbf.check(val),
                        datafusion_common::ScalarValue::Float64(Some(val)) => sbbf.check(val),
                        // String types
                        datafusion_common::ScalarValue::Utf8(Some(val)) => sbbf.check(val.as_str()),
                        datafusion_common::ScalarValue::LargeUtf8(Some(val)) => {
                            sbbf.check(val.as_str())
                        }
                        // Binary types
                        datafusion_common::ScalarValue::Binary(Some(val)) => {
                            sbbf.check(val.as_slice())
                        }
                        datafusion_common::ScalarValue::LargeBinary(Some(val)) => {
                            sbbf.check(val.as_slice())
                        }
                        // Date and time types
                        datafusion_common::ScalarValue::Date32(Some(val)) => sbbf.check(val),
                        datafusion_common::ScalarValue::Date64(Some(val)) => sbbf.check(val),
                        datafusion_common::ScalarValue::Time32Second(Some(val)) => sbbf.check(val),
                        datafusion_common::ScalarValue::Time32Millisecond(Some(val)) => {
                            sbbf.check(val)
                        }
                        datafusion_common::ScalarValue::Time64Microsecond(Some(val)) => {
                            sbbf.check(val)
                        }
                        datafusion_common::ScalarValue::Time64Nanosecond(Some(val)) => {
                            sbbf.check(val)
                        }
                        datafusion_common::ScalarValue::TimestampSecond(Some(val), _) => {
                            sbbf.check(val)
                        }
                        datafusion_common::ScalarValue::TimestampMillisecond(Some(val), _) => {
                            sbbf.check(val)
                        }
                        datafusion_common::ScalarValue::TimestampMicrosecond(Some(val), _) => {
                            sbbf.check(val)
                        }
                        datafusion_common::ScalarValue::TimestampNanosecond(Some(val), _) => {
                            sbbf.check(val)
                        }
                        _ => {
                            return Err(Error::InvalidInput {
                                source: format!(
                                    "Unsupported data type in bloom filter query: {:?}",
                                    value
                                )
                                .into(),
                                location: location!(),
                            });
                        }
                    };

                    if found {
                        return Ok(true);
                    }
                }
                Ok(false) // None of the values were found
            }
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
            "type": "BloomFilter",
            "num_blocks": self.zones.len(),
            "number_of_items": self.number_of_items,
            "probability": self.probability,
        }))
    }

    fn index_type(&self) -> IndexType {
        IndexType::BloomFilter
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let mut frag_ids = RoaringBitmap::new();

        // Loop through zones and add unique fragment IDs to the bitmap
        for block in &self.zones {
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
        metrics.record_comparisons(self.zones.len());
        let query = query.as_any().downcast_ref::<BloomFilterQuery>().unwrap();

        let mut row_id_tree_map = RowIdTreeMap::new();

        // For each zone, check if it might contain the queried value
        for block in self.zones.iter() {
            if self.evaluate_block_against_query(block, query)? {
                // Calculate the range of row addresses for this zone
                // Row addresses are: (fragment_id << 32) + zone_start
                let zone_start_addr = (block.fragment_id << 32) + block.zone_start;
                let zone_end_addr = zone_start_addr + (block.zone_length as u64);

                // Add all row addresses in this zone to the result
                row_id_tree_map.insert_range(zone_start_addr..zone_end_addr);
            }
        }

        Ok(SearchResult::AtMost(row_id_tree_map))
    }

    fn can_remap(&self) -> bool {
        false
    }

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

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        // 1. Prepare the builder for new bloom filters
        let batches_source = new_data;

        let mut builder = BloomFilterIndexBuilder::try_new(BloomFilterIndexBuilderParams {
            number_of_items: self.number_of_items,
            probability: self.probability,
        })?;

        builder.train(batches_source).await?;

        // Get the new blocks from the builder
        let new_blocks = builder.blocks;

        // Combine existing zones with new zones
        let mut all_blocks = self.zones.clone();
        all_blocks.extend(new_blocks);

        // Create a new builder with all blocks to write them out
        let mut combined_builder =
            BloomFilterIndexBuilder::try_new(BloomFilterIndexBuilderParams {
                number_of_items: self.number_of_items,
                probability: self.probability,
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

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        let params = serde_json::to_value(BloomFilterIndexBuilderParams {
            number_of_items: self.number_of_items,
            probability: self.probability,
        })?;
        Ok(ScalarIndexParams::for_builtin(BuiltinIndexType::BloomFilter).with_params(params))
    }
}

fn default_number_of_items() -> u64 {
    *DEFAULT_NUMBER_OF_ITEMS
}

fn default_probability() -> f64 {
    *DEFAULT_PROBABILITY
}

// NumberOfItems: 8192 + Probability: 0.00057(1 in 1754) -> NumberOfBytes: 16384(16KiB) + 8 SALT values
// reference: https://hur.st/bloomfilter/?n=8192&p=&m=16KiB&k=8
static DEFAULT_NUMBER_OF_ITEMS: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("LANCE_BLOOMFILTER_DEFAULT_NUMBER_OF_ITEMS")
        .unwrap_or_else(|_| "8192".to_string())
        .parse()
        .expect("failed to parse Lance_BLOOMFILTER_DEFAULT_NUMBER_OF_ITEMS")
});

#[allow(clippy::manual_inspect)]
static DEFAULT_PROBABILITY: LazyLock<f64> = LazyLock::new(|| {
    std::env::var("LANCE_BLOOMFILTER_DEFAULT_PROBABILITY")
        // 0.00057 ≈ 1 in 1754 false positive rate
        .unwrap_or_else(|_| "0.00057".to_string())
        .parse()
        .map(|prob: f64| {
            assert!(
                (0.0..=1.0).contains(&prob),
                "LANCE_BLOOMFILTER_DEFAULT_PROBABILITY must be between 0 and 1, got {}",
                prob
            );
            prob
        })
        .expect("failed to parse LANCE_BLOOMFILTER_DEFAULT_PROBABILITY")
});

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilterIndexBuilderParams {
    #[serde(default = "default_number_of_items")]
    number_of_items: u64,
    #[serde(default = "default_probability")]
    probability: f64,
}

impl Default for BloomFilterIndexBuilderParams {
    fn default() -> Self {
        Self {
            number_of_items: *DEFAULT_NUMBER_OF_ITEMS,
            probability: *DEFAULT_PROBABILITY,
        }
    }
}

impl BloomFilterIndexBuilderParams {
    #[cfg(test)]
    fn new(number_of_items: u64, probability: f64) -> Self {
        Self {
            number_of_items,
            probability,
        }
    }
}

pub struct BloomFilterIndexBuilder {
    params: BloomFilterIndexBuilderParams,
    blocks: Vec<BloomFilterStatistics>,
    // The local offset within the current zones
    cur_zone_offset: usize,
    cur_fragment_id: u64,
    cur_zone_has_null: bool,
    sbbf: Option<Sbbf>,
}

impl BloomFilterIndexBuilder {
    pub fn try_new(params: BloomFilterIndexBuilderParams) -> Result<Self> {
        let sbbf = SbbfBuilder::new()
            .expected_items(params.number_of_items)
            .false_positive_probability(params.probability)
            .build()
            .map_err(|e| Error::InvalidInput {
                source: format!("Failed to build SBBF: {:?}", e).into(),
                location: location!(),
            })?;

        Ok(Self {
            params,
            blocks: Vec::new(),
            cur_zone_offset: 0,
            cur_fragment_id: 0,
            cur_zone_has_null: false,
            sbbf: Some(sbbf),
        })
    }

    fn process_primitive_array<T>(sbbf: &mut Sbbf, array: &arrow_array::PrimitiveArray<T>) -> bool
    where
        T: arrow_array::ArrowPrimitiveType,
        T::Native: as_bytes::AsBytes,
    {
        let mut has_null = false;
        for i in 0..array.len() {
            if array.is_valid(i) {
                sbbf.insert(&array.value(i));
            } else {
                has_null = true;
            }
        }
        has_null
    }

    fn process_string_array(sbbf: &mut Sbbf, array: &arrow_array::StringArray) -> bool {
        let mut has_null = false;
        for i in 0..array.len() {
            if array.is_valid(i) {
                sbbf.insert(array.value(i));
            } else {
                has_null = true;
            }
        }
        has_null
    }

    fn process_large_string_array(sbbf: &mut Sbbf, array: &arrow_array::LargeStringArray) -> bool {
        let mut has_null = false;
        for i in 0..array.len() {
            if array.is_valid(i) {
                sbbf.insert(array.value(i));
            } else {
                has_null = true;
            }
        }
        has_null
    }

    fn process_binary_array(sbbf: &mut Sbbf, array: &arrow_array::BinaryArray) -> bool {
        let mut has_null = false;
        for i in 0..array.len() {
            if array.is_valid(i) {
                sbbf.insert(array.value(i));
            } else {
                has_null = true;
            }
        }
        has_null
    }

    fn process_large_binary_array(sbbf: &mut Sbbf, array: &arrow_array::LargeBinaryArray) -> bool {
        let mut has_null = false;
        for i in 0..array.len() {
            if array.is_valid(i) {
                sbbf.insert(array.value(i));
            } else {
                has_null = true;
            }
        }
        has_null
    }

    fn update_stats(&mut self, array: &ArrayRef) -> Result<()> {
        if let Some(ref mut sbbf) = self.sbbf {
            let has_null = match array.data_type() {
                // Signed integers
                DataType::Int8 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::Int8Array>()
                        .unwrap();
                    Self::process_primitive_array(sbbf, typed_array)
                }
                DataType::Int16 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::Int16Array>()
                        .unwrap();
                    Self::process_primitive_array(sbbf, typed_array)
                }
                DataType::Int32 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::Int32Array>()
                        .unwrap();
                    Self::process_primitive_array(sbbf, typed_array)
                }
                DataType::Int64 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::Int64Array>()
                        .unwrap();
                    Self::process_primitive_array(sbbf, typed_array)
                }
                // Unsigned integers
                DataType::UInt8 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt8Array>()
                        .unwrap();
                    Self::process_primitive_array(sbbf, typed_array)
                }
                DataType::UInt16 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt16Array>()
                        .unwrap();
                    Self::process_primitive_array(sbbf, typed_array)
                }
                DataType::UInt32 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt32Array>()
                        .unwrap();
                    Self::process_primitive_array(sbbf, typed_array)
                }
                DataType::UInt64 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::UInt64Array>()
                        .unwrap();
                    Self::process_primitive_array(sbbf, typed_array)
                }
                // Floating point numbers
                DataType::Float32 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::Float32Array>()
                        .unwrap();
                    Self::process_primitive_array(sbbf, typed_array)
                }
                DataType::Float64 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::Float64Array>()
                        .unwrap();
                    Self::process_primitive_array(sbbf, typed_array)
                }
                // Date and time types (stored as i32 internally)
                DataType::Date32 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::Date32Array>()
                        .unwrap();
                    Self::process_primitive_array(sbbf, typed_array)
                }
                DataType::Time32(time_unit) => match time_unit {
                    arrow_schema::TimeUnit::Second => {
                        let typed_array = array
                            .as_any()
                            .downcast_ref::<arrow_array::Time32SecondArray>()
                            .unwrap();
                        Self::process_primitive_array(sbbf, typed_array)
                    }
                    arrow_schema::TimeUnit::Millisecond => {
                        let typed_array = array
                            .as_any()
                            .downcast_ref::<arrow_array::Time32MillisecondArray>()
                            .unwrap();
                        Self::process_primitive_array(sbbf, typed_array)
                    }
                    _ => {
                        return Err(Error::InvalidInput {
                            source: format!("Unsupported Time32 unit: {:?}", time_unit).into(),
                            location: location!(),
                        });
                    }
                },
                // Date and time types (stored as i64 internally)
                DataType::Date64 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::Date64Array>()
                        .unwrap();
                    Self::process_primitive_array(sbbf, typed_array)
                }
                DataType::Time64(time_unit) => match time_unit {
                    arrow_schema::TimeUnit::Microsecond => {
                        let typed_array = array
                            .as_any()
                            .downcast_ref::<arrow_array::Time64MicrosecondArray>()
                            .unwrap();
                        Self::process_primitive_array(sbbf, typed_array)
                    }
                    arrow_schema::TimeUnit::Nanosecond => {
                        let typed_array = array
                            .as_any()
                            .downcast_ref::<arrow_array::Time64NanosecondArray>()
                            .unwrap();
                        Self::process_primitive_array(sbbf, typed_array)
                    }
                    _ => {
                        return Err(Error::InvalidInput {
                            source: format!("Unsupported Time64 unit: {:?}", time_unit).into(),
                            location: location!(),
                        });
                    }
                },
                DataType::Timestamp(time_unit, _) => match time_unit {
                    arrow_schema::TimeUnit::Second => {
                        let typed_array = array
                            .as_any()
                            .downcast_ref::<arrow_array::TimestampSecondArray>()
                            .unwrap();
                        Self::process_primitive_array(sbbf, typed_array)
                    }
                    arrow_schema::TimeUnit::Millisecond => {
                        let typed_array = array
                            .as_any()
                            .downcast_ref::<arrow_array::TimestampMillisecondArray>()
                            .unwrap();
                        Self::process_primitive_array(sbbf, typed_array)
                    }
                    arrow_schema::TimeUnit::Microsecond => {
                        let typed_array = array
                            .as_any()
                            .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
                            .unwrap();
                        Self::process_primitive_array(sbbf, typed_array)
                    }
                    arrow_schema::TimeUnit::Nanosecond => {
                        let typed_array = array
                            .as_any()
                            .downcast_ref::<arrow_array::TimestampNanosecondArray>()
                            .unwrap();
                        Self::process_primitive_array(sbbf, typed_array)
                    }
                },
                DataType::Utf8 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::StringArray>()
                        .unwrap();
                    Self::process_string_array(sbbf, typed_array)
                }
                DataType::LargeUtf8 => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::LargeStringArray>()
                        .unwrap();
                    Self::process_large_string_array(sbbf, typed_array)
                }
                DataType::Binary => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::BinaryArray>()
                        .unwrap();
                    Self::process_binary_array(sbbf, typed_array)
                }
                DataType::LargeBinary => {
                    let typed_array = array
                        .as_any()
                        .downcast_ref::<arrow_array::LargeBinaryArray>()
                        .unwrap();
                    Self::process_large_binary_array(sbbf, typed_array)
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
            };

            // Update the current zone's null tracking
            self.cur_zone_has_null = self.cur_zone_has_null || has_null;
        }

        Ok(())
    }

    fn new_block(&mut self, fragment_id: u64) -> Result<()> {
        // Calculate zone_start based on existing zones in the same fragment
        let zone_start = self
            .blocks
            .iter()
            .filter(|block| block.fragment_id == fragment_id)
            .map(|block| block.zone_length as u64)
            .sum::<u64>();

        // Store the current bloom filter directly
        let bloom_filter = if let Some(ref sbbf) = self.sbbf {
            sbbf.clone()
        } else {
            // Create a default empty bloom filter
            SbbfBuilder::new()
                .expected_items(self.params.number_of_items)
                .false_positive_probability(self.params.probability)
                .build()
                .map_err(|e| Error::InvalidInput {
                    source: format!("Failed to build default SBBF: {:?}", e).into(),
                    location: location!(),
                })?
        };

        let new_block = BloomFilterStatistics {
            fragment_id,
            zone_start,
            zone_length: self.cur_zone_offset,
            has_null: self.cur_zone_has_null,
            bloom_filter,
        };

        self.blocks.push(new_block);
        self.cur_zone_offset = 0;
        self.cur_zone_has_null = false;

        // Reset sbbf for the next block
        self.sbbf = Some(
            SbbfBuilder::new()
                .expected_items(self.params.number_of_items)
                .false_positive_probability(self.params.probability)
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
            if self.blocks.is_empty() && self.cur_zone_offset == 0 {
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
                    (self.params.number_of_items - self.cur_zone_offset as u64) as usize;

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
                    self.cur_zone_offset += remaining;
                    break;
                } else if desired > 0 {
                    // There is enough data, create a new zone
                    self.update_stats(&data_array.slice(array_offset, desired))?;
                    self.cur_zone_offset += desired;
                    self.new_block(row_addrs_array.value(array_offset) >> 32)?;
                } else if desired == 0 {
                    // The new batch starts with a new fragment. Flush the current zone if it's not empty
                    if self.cur_zone_offset > 0 {
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
        // Create the final zone
        if self.cur_zone_offset > 0 {
            self.new_block(self.cur_fragment_id)?;
        }

        Ok(())
    }

    fn bloomfilter_stats_as_batch(&self) -> Result<RecordBatch> {
        let fragment_ids =
            UInt64Array::from_iter_values(self.blocks.iter().map(|block| block.fragment_id));

        let zone_starts =
            UInt64Array::from_iter_values(self.blocks.iter().map(|block| block.zone_start));

        let zone_lengths =
            UInt64Array::from_iter_values(self.blocks.iter().map(|block| block.zone_length as u64));

        let has_nulls = arrow_array::BooleanArray::from(
            self.blocks
                .iter()
                .map(|block| block.has_null)
                .collect::<Vec<bool>>(),
        );

        // Convert bloom filters to binary data for serialization
        let bloom_filter_data = if self.blocks.is_empty() {
            Arc::new(arrow_array::BinaryArray::new_null(0)) as ArrayRef
        } else {
            let binary_data: Vec<Vec<u8>> = self
                .blocks
                .iter()
                .map(|block| block.bloom_filter.to_bytes())
                .collect();
            let binary_refs: Vec<Option<&[u8]>> = binary_data
                .iter()
                .map(|bytes| Some(bytes.as_slice()))
                .collect();
            Arc::new(arrow_array::BinaryArray::from_opt_vec(binary_refs)) as ArrayRef
        };

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new("fragment_id", DataType::UInt64, false),
            Field::new("zone_start", DataType::UInt64, false),
            Field::new("zone_length", DataType::UInt64, false),
            Field::new("has_null", DataType::Boolean, false),
            Field::new("bloom_filter_data", DataType::Binary, false),
        ]));

        let columns: Vec<ArrayRef> = vec![
            Arc::new(fragment_ids) as ArrayRef,
            Arc::new(zone_starts) as ArrayRef,
            Arc::new(zone_lengths) as ArrayRef,
            Arc::new(has_nulls) as ArrayRef,
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
            BLOOMFILTER_PROBABILITY_META_KEY.to_string(),
            self.params.probability.to_string(),
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

        // Check if the data type is supported by bloom filter
        match field.data_type() {
            // Signed integers
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            // Unsigned integers
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            // Floating point
            | DataType::Float32
            | DataType::Float64
            // String types
            | DataType::Utf8
            | DataType::LargeUtf8
            // Binary types
            | DataType::Binary
            | DataType::LargeBinary
            // Date and time types
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _) => {
                // Type is supported, continue
            }
            _ => {
                return Err(Error::InvalidInput {
                    source: format!(
                        "Bloom filter index does not support data type: {:?}. Supported types: Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64, Utf8, LargeUtf8, Binary, LargeBinary, Date32, Date64, Time32, Time64, Timestamp",
                        field.data_type()
                    ).into(),
                    location: location!(),
                });
            }
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
        BLOOMFILTER_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        Some(Box::new(BloomFilterQueryParser::new(index_name, true)))
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: &LanceCache,
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

    use crate::scalar::bloomfilter::BloomFilterIndexPlugin;
    use arrow_array::{RecordBatch, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::execution::SendableRecordBatchStream;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion_common::ScalarValue;
    use futures::{stream, StreamExt};
    use lance_core::{cache::LanceCache, utils::mask::RowIdTreeMap, ROW_ADDR};
    use lance_io::object_store::ObjectStore;
    use object_store::path::Path;
    use tempfile::tempdir;

    use crate::scalar::{
        bloomfilter::{BloomFilterIndex, BloomFilterIndexBuilderParams},
        lance_format::LanceIndexStore,
        BloomFilterQuery, ScalarIndex, SearchResult,
    };

    use crate::metrics::NoOpMetricsCollector;
    use crate::Index; // Import Index trait to access calculate_included_frags
    use roaring::RoaringBitmap; // Import RoaringBitmap for the test

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
        let schema = Arc::new(Schema::new(vec![Field::new(
            VALUE_COLUMN_NAME,
            DataType::Int32,
            false,
        )]));
        let data = RecordBatch::try_new(schema.clone(), vec![Arc::new(data)]).unwrap();

        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));
        let data_stream = add_row_addr(data_stream);

        BloomFilterIndexPlugin::train_bloomfilter_index(data_stream, test_store.as_ref(), None)
            .await
            .unwrap();

        log::debug!("Successfully wrote the index file");

        // Read the index file back and check its contents
        let index = BloomFilterIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");
        assert_eq!(index.zones.len(), 0);
        assert_eq!(index.number_of_items, 8192);
        assert_eq!(index.probability, 0.00057); // Default probability

        // Equals query: null (should match nothing, as there are no nulls in empty index)
        let query = BloomFilterQuery::Equals(ScalarValue::Int32(None));
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            VALUE_COLUMN_NAME,
            DataType::Int32,
            false,
        )]));
        let data = RecordBatch::try_new(schema.clone(), vec![Arc::new(data)]).unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));
        let data_stream = add_row_addr(data_stream);

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(100, 0.01)), // ~1% false positive rate
        )
        .await
        .unwrap();

        log::debug!("Successfully wrote the index file");

        // Read the index file back and check its contents
        let index = BloomFilterIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");

        assert_eq!(index.zones.len(), 1);
        assert_eq!(index.number_of_items, 100);
        assert_eq!(index.probability, 0.01);

        // Check that we have one zone (since 100 items fit exactly in one zone of size 100)
        assert_eq!(index.zones[0].fragment_id, 0);
        assert_eq!(index.zones[0].zone_start, 0);
        assert_eq!(index.zones[0].zone_length, 100);

        // Test search functionality
        // The bloom filter should work correctly and find the value
        let query = BloomFilterQuery::Equals(ScalarValue::Int32(Some(50)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match the block since value 50 is in the range [0, 100)
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..100);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value that shouldn't exist
        let query = BloomFilterQuery::Equals(ScalarValue::Int32(Some(500))); // Value not in [0, 100)
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

        let schema = Arc::new(Schema::new(vec![Field::new(
            VALUE_COLUMN_NAME,
            DataType::Int64,
            false,
        )]));

        // Create multiple fragments with data
        // Fragment 0: values 0-99
        let fragment0_data = arrow_array::Int64Array::from_iter_values(0..100);
        let fragment0_batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(fragment0_data)]).unwrap();

        // Fragment 1: values 100-199
        let fragment1_data = arrow_array::Int64Array::from_iter_values(100..200);
        let fragment1_batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(fragment1_data)]).unwrap();

        // Create a stream with multiple batches (fragments)
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::iter(vec![
                Ok(fragment0_batch.clone()),
                Ok(fragment1_batch.clone()),
            ]),
        ));
        let data_stream = add_row_addr(data_stream);

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(50, 0.05)), // ~5% false positive rate
        )
        .await
        .unwrap();

        // Read the index file back and check its contents
        let index = BloomFilterIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");

        // Should have 4 zones total (2 zones per fragment)
        assert_eq!(index.zones.len(), 4);

        // Check fragment 0 zones
        assert_eq!(index.zones[0].fragment_id, 0);
        assert_eq!(index.zones[0].zone_start, 0);
        assert_eq!(index.zones[0].zone_length, 50);

        assert_eq!(index.zones[1].fragment_id, 0);
        assert_eq!(index.zones[1].zone_start, 50);
        assert_eq!(index.zones[1].zone_length, 50);

        // Check fragment 1 zones
        assert_eq!(index.zones[2].fragment_id, 1);
        assert_eq!(index.zones[2].zone_start, 0);
        assert_eq!(index.zones[2].zone_length, 50);

        assert_eq!(index.zones[3].fragment_id, 1);
        assert_eq!(index.zones[3].zone_start, 50);
        assert_eq!(index.zones[3].zone_length, 50);

        // Test search functionality
        let query = BloomFilterQuery::Equals(ScalarValue::Int64(Some(150)));
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            VALUE_COLUMN_NAME,
            DataType::Float32,
            true,
        )]));
        let data =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(float_data.clone())]).unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));
        let data_stream = add_row_addr(data_stream);

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(100, 0.01)), // ~1% false positive rate
        )
        .await
        .unwrap();

        // Load the index
        let index = BloomFilterIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");

        // Should have 5 zones since we have 500 rows and zone size is 100
        assert_eq!(index.zones.len(), 5);

        // Test search for NaN values using Equals with NaN
        let query = BloomFilterQuery::Equals(ScalarValue::Float32(Some(f32::NAN)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match all blocks since they all contain NaN values
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..500); // All rows since NaN is in every block
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a specific finite value that exists in the data
        let query = BloomFilterQuery::Equals(ScalarValue::Float32(Some(5.0)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match only the first block since 5.0 only exists in rows 0-99
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..100);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value that doesn't exist but is within expected range
        let query = BloomFilterQuery::Equals(ScalarValue::Float32(Some(250.0)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match the third block since 250.0 would be in that range if it existed
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(200..300);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value way outside the range
        let query = BloomFilterQuery::Equals(ScalarValue::Float32(Some(10000.0)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should return empty since bloom filter correctly filters out this value
        assert_eq!(result, SearchResult::AtMost(RowIdTreeMap::new()));

        // Test IsIn query with NaN and finite values
        let query = BloomFilterQuery::IsIn(vec![
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            VALUE_COLUMN_NAME,
            DataType::Int64,
            false,
        )]));
        let data = RecordBatch::try_new(schema.clone(), vec![Arc::new(data)]).unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));
        let data_stream = add_row_addr(data_stream);

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(1000, 0.001)), // 10 blocks total
        )
        .await
        .unwrap();

        // Load the index
        let index = BloomFilterIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");

        // Should have 10 zones since we have 10000 rows and zone size is 1000
        assert_eq!(index.zones.len(), 10);
        assert_eq!(index.number_of_items, 1000);
        assert_eq!(index.probability, 0.001);

        // Verify zone structure
        for (i, block) in index.zones.iter().enumerate() {
            assert_eq!(block.fragment_id, 0);
            assert_eq!(block.zone_start, (i * 1000) as u64);
            assert_eq!(block.zone_length, 1000);
            // Check that the bloom filter has some data (non-zero bytes when serialized)
            assert!(!block.bloom_filter.to_bytes().is_empty());
        }

        // Test search for a value in a specific zone
        let query = BloomFilterQuery::Equals(ScalarValue::Int64(Some(2500))); // In zone 2 (2000-2999)
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match zone 2
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(2000..3000);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value way outside the range
        let query = BloomFilterQuery::Equals(ScalarValue::Int64(Some(50000)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should return empty since bloom filter correctly filters out this value
        assert_eq!(result, SearchResult::AtMost(RowIdTreeMap::new()));

        // Test IsIn query with values from different zones
        let query = BloomFilterQuery::IsIn(vec![
            ScalarValue::Int64(Some(500)),   // Zone 0 (0-999)
            ScalarValue::Int64(Some(2500)),  // Zone 2 (2000-2999)
            ScalarValue::Int64(Some(7500)),  // Zone 7 (7000-7999)
            ScalarValue::Int64(Some(50000)), // Not in any zone
        ]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match zones 0, 2, and 7
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..1000); // Zone 0
        expected.insert_range(2000..3000); // Zone 2
        expected.insert_range(7000..8000); // Zone 7
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            VALUE_COLUMN_NAME,
            DataType::Utf8,
            false,
        )]));
        let data = RecordBatch::try_new(schema.clone(), vec![Arc::new(string_data)]).unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));
        let data_stream = add_row_addr(data_stream);

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(100, 0.01)), // ~1% false positive rate
        )
        .await
        .unwrap();

        // Load the index
        let index = BloomFilterIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");

        // Should have 2 zones since we have 200 rows and zone size is 100
        assert_eq!(index.zones.len(), 2);

        // Test search for a value in the first zone
        let query = BloomFilterQuery::Equals(ScalarValue::Utf8(Some("value_050".to_string())));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match the first zone
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..100);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value in the second zone
        let query = BloomFilterQuery::Equals(ScalarValue::Utf8(Some("value_150".to_string())));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match the second zone
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(100..200);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value that doesn't exist
        let query =
            BloomFilterQuery::Equals(ScalarValue::Utf8(Some("nonexistent_value".to_string())));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should return empty since bloom filter correctly filters out this value
        assert_eq!(result, SearchResult::AtMost(RowIdTreeMap::new()));

        // Test IsIn query with string values
        let query = BloomFilterQuery::IsIn(vec![
            ScalarValue::Utf8(Some("value_025".to_string())), // First zone
            ScalarValue::Utf8(Some("value_175".to_string())), // Second zone
            ScalarValue::Utf8(Some("nonexistent".to_string())), // Not present
        ]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match both zones
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            VALUE_COLUMN_NAME,
            DataType::Binary,
            false,
        )]));
        let data = RecordBatch::try_new(schema.clone(), vec![Arc::new(binary_data)]).unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));
        let data_stream = add_row_addr(data_stream);

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(50, 0.05)),
        )
        .await
        .unwrap();

        // Load the index
        let index = BloomFilterIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");

        // Should have 2 zones since we have 100 rows and zone size is 50
        assert_eq!(index.zones.len(), 2);

        // Test search for a value in the first zone
        let query = BloomFilterQuery::Equals(ScalarValue::Binary(Some(vec![25, 26, 27])));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match the first zone
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..50);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value in the second zone
        let query = BloomFilterQuery::Equals(ScalarValue::Binary(Some(vec![75, 76, 77])));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match the second zone
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(50..100);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value that doesn't exist
        let query = BloomFilterQuery::Equals(ScalarValue::Binary(Some(vec![255, 254, 253])));
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
        let schema = Arc::new(Schema::new(vec![Field::new(
            VALUE_COLUMN_NAME,
            DataType::LargeUtf8,
            false,
        )]));
        let data = RecordBatch::try_new(schema.clone(), vec![Arc::new(large_string_data)]).unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));
        let data_stream = add_row_addr(data_stream);

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(50, 0.05)),
        )
        .await
        .unwrap();

        // Load the index
        let index = BloomFilterIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");

        assert_eq!(index.zones.len(), 2);

        // Test search functionality
        let query = BloomFilterQuery::Equals(ScalarValue::LargeUtf8(Some(
            "large_value_00025".to_string(),
        )));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should match the first zone
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..50);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for a value that doesn't exist
        let query = BloomFilterQuery::Equals(ScalarValue::LargeUtf8(Some(
            "nonexistent_large_value".to_string(),
        )));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();

        // Should return empty since bloom filter correctly filters out this value
        assert_eq!(result, SearchResult::AtMost(RowIdTreeMap::new()));
    }

    #[tokio::test]
    async fn test_timestamp_bloomfilter_index() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Test Date32 (days since Unix epoch)
        let date32_values: Vec<i32> = (0..100).collect(); // Days since Unix epoch
        let date32_data = arrow_array::Date32Array::from(date32_values.clone());
        let schema = Arc::new(Schema::new(vec![Field::new(
            VALUE_COLUMN_NAME,
            DataType::Date32,
            false,
        )]));
        let data = RecordBatch::try_new(schema.clone(), vec![Arc::new(date32_data)]).unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));
        let data_stream = add_row_addr(data_stream);

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(50, 0.01)),
        )
        .await
        .unwrap();

        // Load the Date32 index
        let index = BloomFilterIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load Date32 BloomFilterIndex");

        assert_eq!(index.zones.len(), 2); // 100 rows, zone size 50

        // Test search for Date32 value in first zone
        let query = BloomFilterQuery::Equals(ScalarValue::Date32(Some(25)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..50);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for Date32 value in second zone
        let query = BloomFilterQuery::Equals(ScalarValue::Date32(Some(75)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(50..100);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for Date32 value that doesn't exist
        let query = BloomFilterQuery::Equals(ScalarValue::Date32(Some(500)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::AtMost(RowIdTreeMap::new()));
    }

    #[tokio::test]
    async fn test_timestamp_types_bloomfilter_index() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Test Timestamp with nanosecond precision - use simple incrementing values
        let timestamp_values: Vec<i64> = (0..100).map(|i| 1_000_000_000i64 + (i as i64)).collect();

        let timestamp_data = arrow_array::TimestampNanosecondArray::from(timestamp_values.clone());
        let schema = Arc::new(Schema::new(vec![Field::new(
            VALUE_COLUMN_NAME,
            DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
            false,
        )]));
        let data = RecordBatch::try_new(schema.clone(), vec![Arc::new(timestamp_data)]).unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));
        let data_stream = add_row_addr(data_stream);

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(50, 0.01)),
        )
        .await
        .unwrap();

        // Load the Timestamp index
        let index = BloomFilterIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load Timestamp BloomFilterIndex");

        assert_eq!(index.zones.len(), 2); // 100 rows, zone size 50

        // Test search for Timestamp value in first zone
        let first_timestamp = timestamp_values[25];
        let query = BloomFilterQuery::Equals(ScalarValue::TimestampNanosecond(
            Some(first_timestamp),
            None,
        ));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..50);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for Timestamp value in second zone
        let second_timestamp = timestamp_values[75];
        let query = BloomFilterQuery::Equals(ScalarValue::TimestampNanosecond(
            Some(second_timestamp),
            None,
        ));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(50..100);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for Timestamp value that doesn't exist
        let query =
            BloomFilterQuery::Equals(ScalarValue::TimestampNanosecond(Some(999_999_999i64), None));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::AtMost(RowIdTreeMap::new()));

        // Test IsIn query with multiple timestamp values
        let query = BloomFilterQuery::IsIn(vec![
            ScalarValue::TimestampNanosecond(Some(timestamp_values[10]), None), // First zone
            ScalarValue::TimestampNanosecond(Some(timestamp_values[85]), None), // Second zone
            ScalarValue::TimestampNanosecond(Some(999_999_999i64), None),       // Not present
        ]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..100); // Should match both zones
        assert_eq!(result, SearchResult::AtMost(expected));
    }

    #[tokio::test]
    async fn test_time_types_bloomfilter_index() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Test Time64 with microsecond precision (stored as i64)
        let time_values: Vec<i64> = (0..100)
            .map(|i| (i as i64) * 3_600_000_000) // Hours in microseconds
            .collect();

        let time_data = arrow_array::Time64MicrosecondArray::from(time_values.clone());
        let schema = Arc::new(Schema::new(vec![Field::new(
            VALUE_COLUMN_NAME,
            DataType::Time64(arrow_schema::TimeUnit::Microsecond),
            false,
        )]));
        let data = RecordBatch::try_new(schema.clone(), vec![Arc::new(time_data)]).unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));
        let data_stream = add_row_addr(data_stream);

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(25, 0.05)),
        )
        .await
        .unwrap();

        // Load the Time64 index
        let index = BloomFilterIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load Time64 BloomFilterIndex");

        assert_eq!(index.zones.len(), 4); // 100 rows, zone size 25

        // Test search for Time64 value in first zone
        let first_time = time_values[10];
        let query = BloomFilterQuery::Equals(ScalarValue::Time64Microsecond(Some(first_time)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..25);
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test search for Time64 value that doesn't exist
        let query = BloomFilterQuery::Equals(ScalarValue::Time64Microsecond(Some(999_999_999i64)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::AtMost(RowIdTreeMap::new()));
    }

    #[tokio::test]
    async fn test_bloomfilter_supported_operations() {
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            Arc::new(LanceCache::no_cache()),
        ));

        let data = arrow_array::Int32Array::from_iter_values(0..1000);
        let schema = Arc::new(Schema::new(vec![Field::new(
            VALUE_COLUMN_NAME,
            DataType::Int32,
            false,
        )]));
        let data = RecordBatch::try_new(schema.clone(), vec![Arc::new(data)]).unwrap();
        let data_stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(data))),
        ));
        let data_stream = add_row_addr(data_stream);

        BloomFilterIndexPlugin::train_bloomfilter_index(
            data_stream,
            test_store.as_ref(),
            Some(BloomFilterIndexBuilderParams::new(250, 0.01)), // 4 zones total
        )
        .await
        .unwrap();

        // Load the index
        let index = BloomFilterIndex::load(test_store.clone(), None, &LanceCache::no_cache())
            .await
            .expect("Failed to load BloomFilterIndex");

        assert_eq!(index.zones.len(), 4);

        // Test that bloom filters support the operations they are designed for
        // Test a specific equality query
        let query = BloomFilterQuery::Equals(ScalarValue::Int32(Some(500)));
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(500..750); // Should match the zone containing 500
        assert_eq!(result, SearchResult::AtMost(expected));

        // Test IsNull query
        let query = BloomFilterQuery::IsNull();
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(result, SearchResult::AtMost(RowIdTreeMap::new())); // No nulls in the data

        // Test IsIn query
        let query = BloomFilterQuery::IsIn(vec![
            ScalarValue::Int32(Some(100)),
            ScalarValue::Int32(Some(600)),
        ]);
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        let mut expected = RowIdTreeMap::new();
        expected.insert_range(0..250); // Zone containing 100
        expected.insert_range(500..750); // Zone containing 600
        assert_eq!(result, SearchResult::AtMost(expected));
    }
}
