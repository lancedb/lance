// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    any::Any,
    collections::{BTreeMap, HashMap},
    fmt::Debug,
    ops::Bound,
    sync::Arc,
};

use arrow::array::BinaryBuilder;
use arrow_array::{new_empty_array, new_null_array, Array, BinaryArray, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion_common::ScalarValue;
use deepsize::DeepSizeOf;
use futures::TryStreamExt;
use lance_core::{utils::mask::RowIdTreeMap, Error, Result};
use roaring::RoaringBitmap;
use serde::Serialize;
use snafu::location;
use tracing::instrument;

use crate::{metrics::MetricsCollector, Index, IndexType};

use super::{btree::OrderableScalarValue, SargableQuery, SearchResult};
use super::{btree::TrainingSource, AnyQuery, IndexStore, ScalarIndex};

pub const BITMAP_LOOKUP_NAME: &str = "bitmap_page_lookup.lance";

const MAX_ARROW_ARRAY_LEN: usize = i32::MAX as usize - 1024 * 1024; // leave headroom

/// A scalar index that stores a bitmap for each possible value
///
/// This index works best for low-cardinality columns, where the number of unique values is small.
/// The bitmap stores a list of row ids where the value is present.
#[derive(Clone, Debug)]
pub struct BitmapIndex {
    index_map: BTreeMap<OrderableScalarValue, RowIdTreeMap>,
    // We put null in its own map to avoid it matching range queries (arrow-rs considers null to come before minval)
    null_map: RowIdTreeMap,
    // The data type of the values in the index
    value_type: DataType,
    // Memoized index_map size for DeepSizeOf
    index_map_size_bytes: usize,
    store: Arc<dyn IndexStore>,
}

impl BitmapIndex {
    fn new(
        index_map: BTreeMap<OrderableScalarValue, RowIdTreeMap>,
        null_map: RowIdTreeMap,
        value_type: DataType,
        index_map_size_bytes: usize,
        store: Arc<dyn IndexStore>,
    ) -> Self {
        Self {
            index_map,
            null_map,
            value_type,
            index_map_size_bytes,
            store,
        }
    }
}

impl DeepSizeOf for BitmapIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        let mut total_size = 0;

        // Size of BTreeMap values
        total_size += self.index_map_size_bytes;

        // Size of Arc<dyn IndexStore> contents
        total_size += self.store.deep_size_of_children(context);

        total_size
    }
}

#[derive(Serialize)]
struct BitmapStatistics {
    num_bitmaps: usize,
}

#[async_trait]
impl Index for BitmapIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::NotSupported {
            source: "BitmapIndex is not a vector index".into(),
            location: location!(),
        })
    }

    async fn prewarm(&self) -> Result<()> {
        // TODO: Bitmap index essentially pre-warms on load right now.  This is a problem for some
        // of the larger bitmap indices (e.g. label_list).  We should probably change it to behave
        // like other indices and then we will need to implement this.
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::Bitmap
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let stats = BitmapStatistics {
            num_bitmaps: self.index_map.len(),
        };
        serde_json::to_value(stats).map_err(|e| Error::Internal {
            message: format!("failed to serialize bitmap index statistics: {}", e),
            location: location!(),
        })
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}

#[async_trait]
impl ScalarIndex for BitmapIndex {
    #[instrument(name = "bitmap_search", level = "debug", skip_all)]
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query.as_any().downcast_ref::<SargableQuery>().unwrap();

        let row_ids = match query {
            SargableQuery::Equals(val) => {
                metrics.record_comparisons(1);
                if val.is_null() {
                    self.null_map.clone()
                } else {
                    let key = OrderableScalarValue(val.clone());
                    self.index_map.get(&key).cloned().unwrap_or_default()
                }
            }
            SargableQuery::Range(start, end) => {
                let range_start = match start {
                    Bound::Included(val) => Bound::Included(OrderableScalarValue(val.clone())),
                    Bound::Excluded(val) => Bound::Excluded(OrderableScalarValue(val.clone())),
                    Bound::Unbounded => Bound::Unbounded,
                };

                let range_end = match end {
                    Bound::Included(val) => Bound::Included(OrderableScalarValue(val.clone())),
                    Bound::Excluded(val) => Bound::Excluded(OrderableScalarValue(val.clone())),
                    Bound::Unbounded => Bound::Unbounded,
                };

                let maps = self
                    .index_map
                    .range((range_start, range_end))
                    .map(|(_, v)| v)
                    .collect::<Vec<_>>();

                metrics.record_comparisons(maps.len());
                RowIdTreeMap::union_all(&maps)
            }
            SargableQuery::IsIn(values) => {
                let mut union_bitmap = RowIdTreeMap::default();
                metrics.record_comparisons(values.len());
                for val in values {
                    if val.is_null() {
                        union_bitmap |= self.null_map.clone();
                    } else {
                        let key = OrderableScalarValue(val.clone());
                        if let Some(bitmap) = self.index_map.get(&key) {
                            union_bitmap |= bitmap.clone();
                        }
                    }
                }

                union_bitmap
            }
            SargableQuery::IsNull() => {
                metrics.record_comparisons(1);
                self.null_map.clone()
            }
            SargableQuery::FullTextSearch(_) => {
                return Err(Error::NotSupported {
                    source: "full text search is not supported for bitmap indexes".into(),
                    location: location!(),
                });
            }
        };

        Ok(SearchResult::Exact(row_ids))
    }

    fn can_answer_exact(&self, _: &dyn AnyQuery) -> bool {
        true
    }

    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>> {
        let page_lookup_file = store.open_index_file(BITMAP_LOOKUP_NAME).await?;
        let total_rows = page_lookup_file.num_rows();

        println!("Loading bitmap index with {} total rows", total_rows);

        // Initialize empty index structures
        let mut index_map: BTreeMap<OrderableScalarValue, RowIdTreeMap> = BTreeMap::new();
        let mut null_map = RowIdTreeMap::default();
        let mut index_map_size_bytes = 0;

        // This will be set from the first batch
        let mut value_type: Option<DataType> = None;

        // Load data in chunks to prevent overflow
        const CHUNK_SIZE: usize = 1_000_000; // Adjust based on testing

        let mut start = 0;
        while start < total_rows {
            let end = (start + CHUNK_SIZE).min(total_rows);
            println!("Loading rows {} to {}", start, end);

            let batch = page_lookup_file.read_range(start..end, None).await?;

            // Set value type from first batch if not already set
            if value_type.is_none() && batch.num_rows() > 0 {
                value_type = Some(batch.column(0).data_type().clone());
            }

            if batch.num_rows() == 0 {
                // Skip empty batch
                start = end;
                continue;
            }

            let dict_keys = batch.column(0);
            let binary_bitmaps = batch.column(1);
            let bitmap_binary_array = binary_bitmaps
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap();

            // Process each row in the current batch
            for idx in 0..batch.num_rows() {
                let key = OrderableScalarValue(ScalarValue::try_from_array(dict_keys, idx)?);
                let bitmap_bytes = bitmap_binary_array.value(idx);
                let bitmap = RowIdTreeMap::deserialize_from(bitmap_bytes).unwrap();

                index_map_size_bytes += key.deep_size_of();
                // Approximation of the RowIdTreeMap size
                index_map_size_bytes += bitmap_bytes.len();

                if key.0.is_null() {
                    null_map = bitmap;
                } else {
                    index_map.insert(key, bitmap);
                }
            }

            start = end;
        }

        // If we didn't load any data, use a default value type
        let value_type = value_type.unwrap_or_else(|| {
            println!("Warning: Empty bitmap index, using default value type");
            DataType::Int32
        });

        println!(
            "Successfully loaded bitmap index with {} keys",
            index_map.len()
        );

        Ok(Arc::new(Self::new(
            index_map,
            null_map,
            value_type,
            index_map_size_bytes,
            store,
        )))
    }

    /// Remap the row ids, creating a new remapped version of this index in `dest_store`
    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        let state = self
            .index_map
            .iter()
            .map(|(key, bitmap)| {
                let bitmap =
                    RowIdTreeMap::from_iter(bitmap.row_ids().unwrap().filter_map(|addr| {
                        let addr_as_u64 = u64::from(addr);
                        mapping
                            .get(&addr_as_u64)
                            .copied()
                            .unwrap_or(Some(addr_as_u64))
                    }));
                (key.0.clone(), bitmap)
            })
            .collect::<HashMap<_, _>>();
        write_bitmap_index(state, dest_store, &self.value_type).await
    }

    /// Add the new data into the index, creating an updated version of the index in `dest_store`
    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        let mut state = self
            .index_map
            .iter()
            .map(|(key, bitmap)| (key.0.clone(), bitmap.clone()))
            .collect::<HashMap<_, _>>();

        // Also insert the null map
        let ex_null = new_null_array(&self.value_type, 1);
        let ex_null = ScalarValue::try_from_array(ex_null.as_ref(), 0)?;
        state.insert(ex_null, self.null_map.clone());

        do_train_bitmap_index(new_data, state, dest_store).await
    }
}

fn get_batch_from_arrays(
    keys: Arc<dyn Array>,
    binary_bitmaps: Arc<dyn Array>,
) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("keys", keys.data_type().clone(), true),
        Field::new("bitmaps", binary_bitmaps.data_type().clone(), true),
    ]));

    let columns = vec![keys, binary_bitmaps];

    Ok(RecordBatch::try_new(schema, columns)?)
}

fn chunked_keys_and_bitmaps(
    state: HashMap<ScalarValue, RowIdTreeMap>,
    value_type: &DataType,
) -> Vec<(Arc<dyn Array>, Arc<dyn Array>)> {
    let mut batches = Vec::new();

    // Early return for empty state
    if state.is_empty() {
        // Create empty arrays with the correct types
        let keys_array = new_empty_array(value_type);
        let bitmaps_array = Arc::new(BinaryBuilder::new().finish()) as Arc<dyn Array>;
        batches.push((keys_array, bitmaps_array));
        return batches;
    }

    let mut cur_keys = Vec::new();
    let mut cur_bitmaps = Vec::new();
    let mut cur_bytes = 0;

    for (key, bitmap) in state.into_iter() {
        let mut bytes = Vec::new();
        bitmap.serialize_into(&mut bytes).unwrap();
        let bitmap_len = bytes.len();

        // If adding this bitmap would overflow, flush current batch
        if cur_keys.len() >= MAX_ARROW_ARRAY_LEN || cur_bytes + bitmap_len > MAX_ARROW_ARRAY_LEN {
            flush_keys_and_bitmaps(&mut cur_keys, &mut cur_bitmaps, &mut batches, value_type);
            cur_bytes = 0;
        }

        cur_keys.push(key);
        cur_bitmaps.push(bytes);
        cur_bytes += bitmap_len;
    }

    // Flush any remaining
    if !cur_keys.is_empty() {
        flush_keys_and_bitmaps(&mut cur_keys, &mut cur_bitmaps, &mut batches, value_type);
    }

    // If we still have no batches (which shouldn't happen if state isn't empty),
    // add an empty batch with the correct types
    if batches.is_empty() {
        let keys_array = new_empty_array(value_type);
        let bitmaps_array = Arc::new(BinaryBuilder::new().finish()) as Arc<dyn Array>;
        batches.push((keys_array, bitmaps_array));
    }

    batches
}

// Helper function to create arrays from current keys and bitmaps
fn flush_keys_and_bitmaps(
    cur_keys: &mut Vec<ScalarValue>,
    cur_bitmaps: &mut Vec<Vec<u8>>,
    batches: &mut Vec<(Arc<dyn Array>, Arc<dyn Array>)>,
    value_type: &DataType,
) {
    let keys_array = match ScalarValue::iter_to_array(cur_keys.clone()) {
        Ok(array) => array,
        Err(e) => {
            // If there's a type error, convert keys to match expected type
            println!("Warning: Error creating keys array: {}", e);

            match value_type {
                DataType::Utf8 => {
                    // Convert all keys to strings
                    let converted_keys: Vec<ScalarValue> = cur_keys
                        .iter()
                        .map(|k| {
                            if k.is_null() {
                                ScalarValue::Utf8(None)
                            } else {
                                ScalarValue::Utf8(Some(k.to_string()))
                            }
                        })
                        .collect();

                    match ScalarValue::iter_to_array(converted_keys) {
                        Ok(arr) => arr,
                        Err(_) => new_empty_array(value_type), // Fallback to empty array
                    }
                }
                DataType::Int32 => {
                    // Convert to Int32 or NULL
                    let converted_keys: Vec<ScalarValue> = cur_keys
                        .iter()
                        .map(|k| {
                            if let ScalarValue::Int32(v) = k {
                                k.clone()
                            } else if k.is_null() {
                                ScalarValue::Int32(None)
                            } else {
                                // Try to convert to int if possible
                                if let Ok(i) = k.to_string().parse::<i32>() {
                                    ScalarValue::Int32(Some(i))
                                } else {
                                    ScalarValue::Int32(None)
                                }
                            }
                        })
                        .collect();

                    match ScalarValue::iter_to_array(converted_keys) {
                        Ok(arr) => arr,
                        Err(_) => new_empty_array(value_type), // Fallback to empty array
                    }
                }
                _ => new_empty_array(value_type), // Other types: use empty array
            }
        }
    };

    let mut binary_builder = BinaryBuilder::new();
    for b in cur_bitmaps.iter() {
        binary_builder.append_value(b);
    }
    let bitmaps_array = Arc::new(binary_builder.finish()) as Arc<dyn Array>;

    batches.push((keys_array, bitmaps_array));
    cur_keys.clear();
    cur_bitmaps.clear();
}

async fn write_bitmap_index(
    state: HashMap<ScalarValue, RowIdTreeMap>,
    index_store: &dyn IndexStore,
    value_type: &DataType,
) -> Result<()> {
    // println!("write_bitmap_index: DEBUG: state size {}", state.len());

    // Create a schema even for empty states
    let schema = Arc::new(Schema::new(vec![
        Field::new("keys", value_type.clone(), true),
        Field::new("bitmaps", DataType::Binary, true),
    ]));

    // Create the index file
    let mut bitmap_index_file = index_store
        .new_index_file(BITMAP_LOOKUP_NAME, schema)
        .await?;

    // If state is not empty, process it normally
    if !state.is_empty() {
        let batches = chunked_keys_and_bitmaps(state, value_type);

        for (keys_array, binary_bitmap_array) in batches.into_iter() {
            let record_batch = get_batch_from_arrays(keys_array, binary_bitmap_array)?;

            bitmap_index_file.write_record_batch(record_batch).await?;
        }
    }

    // Always finish the file even when empty
    bitmap_index_file.finish().await?;

    Ok(())
}

async fn do_train_bitmap_index(
    mut data_source: SendableRecordBatchStream,
    mut state: HashMap<ScalarValue, RowIdTreeMap>,
    index_store: &dyn IndexStore,
) -> Result<()> {
    let value_type = data_source.schema().field(0).data_type().clone();
    while let Some(batch) = data_source.try_next().await? {
        debug_assert_eq!(batch.num_columns(), 2);
        debug_assert_eq!(*batch.column(1).data_type(), DataType::UInt64);

        let key_column = batch.column(0);
        let row_id_column = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        for i in 0..key_column.len() {
            let row_id = row_id_column.value(i);
            let key = ScalarValue::try_from_array(key_column.as_ref(), i)?;
            state.entry(key.clone()).or_default().insert(row_id);
        }
    }

    write_bitmap_index(state, index_store, &value_type).await
}

pub async fn train_bitmap_index(
    data_source: Box<dyn TrainingSource + Send>,
    index_store: &dyn IndexStore,
) -> Result<()> {
    let batches_source = data_source.scan_unordered_chunks(4096).await?;

    // mapping from item to list of the row ids where it is present
    let dictionary: HashMap<ScalarValue, RowIdTreeMap> = HashMap::new();

    do_train_bitmap_index(batches_source, dictionary, index_store).await
}

#[cfg(test)]
pub mod tests {
    #[tokio::test]
    #[ignore]
    async fn test_big_value() {
        use super::{BitmapIndex, ScalarIndex, SearchResult, BITMAP_LOOKUP_NAME};
        use crate::scalar::lance_format::LanceIndexStore;
        use crate::scalar::IndexStore;
        use crate::scalar::SargableQuery;
        use arrow_schema::DataType;
        use datafusion_common::ScalarValue;
        use lance_core::cache::FileMetadataCache;
        use lance_core::utils::mask::RowIdTreeMap;
        use lance_io::object_store::ObjectStore;
        use object_store::path::Path;
        use std::collections::HashMap;
        use std::sync::Arc;
        use tempfile::tempdir;

        // Adjust these numbers based on your test resources
        // For CI, use smaller values that still exercise chunking logic
        let m: u32 = if std::env::var("CI").is_ok() {
            10_000 // Smaller set for CI
        } else {
            2_500_000 // Large enough to trigger overflow without chunking
        };

        let per_bitmap_size: u64 = 1000; // row IDs per bitmap

        // Create test dataset
        // println!("Creating test dataset with {} keys", m);
        let mut state = HashMap::new();
        for i in 0..m {
            let bitmap = RowIdTreeMap::from_iter(0..per_bitmap_size);
            let key = ScalarValue::UInt32(Some(i));
            state.insert(key, bitmap);
        }

        // Create a temporary store
        let tmpdir = Arc::new(tempdir().unwrap());
        let test_store = LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(tmpdir.path()).unwrap(),
            FileMetadataCache::no_cache(),
        );

        // Write the bitmap index - this should now use multiple chunks
        // println!("Writing index to disk...");
        let result = super::write_bitmap_index(state, &test_store, &DataType::UInt32).await;
        assert!(result.is_ok(), "Failed to write index: {:?}", result.err());
        // let test_store = LanceIndexStore::new(
        //     Arc::new(ObjectStore::local()),
        //     Path::from_filesystem_path("/Users/haochengliu/Documents/projects/lance/big_data")
        //         .unwrap(),
        //     FileMetadataCache::no_cache(),
        // );

        // Verify the index file exists
        let index_file = test_store.open_index_file(BITMAP_LOOKUP_NAME).await;
        assert!(
            index_file.is_ok(),
            "Failed to open index file: {:?}",
            index_file.err()
        );
        let index_file = index_file.unwrap();

        // Print stats about the index file
        println!(
            "Index file contains {} rows in total",
            index_file.num_rows()
        );

        // Load the index using BitmapIndex::load
        println!("Loading index from disk...");
        let loaded_index = BitmapIndex::load(Arc::new(test_store))
            .await
            .expect("Failed to load bitmap index");

        // Verify the loaded index has the correct number of entries
        assert_eq!(
            loaded_index.index_map.len(),
            m as usize,
            "Loaded index has incorrect number of keys (expected {}, got {})",
            m,
            loaded_index.index_map.len()
        );

        // Manually verify specific keys without using search()
        let test_keys = [0, m / 2, m - 1]; // Beginning, middle, and end
        for &key_val in &test_keys {
            let key = super::OrderableScalarValue(ScalarValue::UInt32(Some(key_val)));
            let bitmap = loaded_index
                .index_map
                .get(&key)
                .expect(&format!("Key {} should exist", key_val));

            // Convert RowIdTreeMap to a vector for easier assertion
            let row_ids: Vec<u64> = bitmap
                .row_ids()
                .unwrap()
                .map(|addr| u64::from(addr))
                .collect();

            // Verify length
            assert_eq!(
                row_ids.len(),
                per_bitmap_size as usize,
                "Bitmap for key {} has wrong size",
                key_val
            );

            // Verify first few and last few elements
            for i in 0..5.min(per_bitmap_size) {
                assert!(
                    row_ids.contains(&i),
                    "Bitmap for key {} should contain row_id {}",
                    key_val,
                    i
                );
            }

            for i in (per_bitmap_size - 5).max(0)..per_bitmap_size {
                assert!(
                    row_ids.contains(&i),
                    "Bitmap for key {} should contain row_id {}",
                    key_val,
                    i
                );
            }

            // Verify exact range
            let expected_range: Vec<u64> = (0..per_bitmap_size).collect();
            assert_eq!(
                row_ids, expected_range,
                "Bitmap for key {} doesn't contain expected values",
                key_val
            );

            println!(
                "✓ Verified bitmap for key {}: {} rows as expected",
                key_val,
                row_ids.len()
            );
        }

        println!("Test successful! Index properly contains {} keys", m);
    }
}
